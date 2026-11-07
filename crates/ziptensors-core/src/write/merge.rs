//! File: write/merge.rs
//! Purpose: copying a member from another container, with no decoding and no recompression.
//!
//! Main responsibilities: copy the stored chunks of each blob a member uses, with their records
//! rebased, spec section 8, and map its extents to the new blob ids.
//!
//! Key structures: [`CopyTarget`], [`CopiedMember`].
//!
//! Key functions: [`copy_member`].
//!
//! Implementation notes: a chunk depends on nothing else, so `xxh3_decoded`, `stored_len`, `raw_len`,
//! `chunk_size`, `blake3`, and the chain's steps stay as they are; only `file_offset`, `first_chunk`
//! and `chain_id` change. A blob is copied whole, even if the member uses a part of it: what is used
//! is known only per extent, and a partial blob is another blob. A source blob without identity is
//! copied with none, since computing one means decoding it, so the same blob from two sources is two
//! blobs unless a source computed the identity. Chunk records of a blob enter the index only when the
//! whole blob is written, so a failed read rolls the file back and leaves the index correct.
//!
//! A delta blob (spec section 7a) is copied after its base, which keeps the rule that a base has the
//! lower id, and its `PARM` entry is written again with the base's id in this pack. The base is one
//! this pack has, by the source's id or by identity, only if it is a plain blob of the same
//! `chunk_size` and length: chunk `i` of the delta blob must meet the same base chunk as in the
//! source. Otherwise the base is copied. Its stored bytes count in the report as stored bytes of the
//! delta extent's class that add nothing but overhead, since no extent of the member reaches them.

use std::collections::HashMap;

use crate::chain::{BLOB_KIND_CHUNKED, BLOB_KIND_DELTA};
use crate::error::Error;
use crate::format::{BlobRecord, ChunkRecord, DeltaParams, ExtentRecord, IndexBuilder, MemberRecord};
use crate::read::Container;

use super::commit::Sink;
use super::dedup::{BlobShape, BlobTable};
use super::report::{ClassTable, class_of};

/// A source container and which of its blobs this pack holds already. Holding the [`Container`] keeps
/// its identity valid, since two live containers are never the same shared state by accident.
#[derive(Debug)]
pub(super) struct CopiedSource {
    /// The source.
    pub(super) container: Container,
    /// Blob ids of the source to blob ids of this pack.
    pub(super) blobs: HashMap<u32, u32>,
}

/// The parts of a pack that a copy changes.
#[derive(Debug)]
pub(super) struct CopyTarget<'a> {
    /// The temporary output.
    pub(super) sink: &'a mut Sink,
    /// Records of the pack.
    pub(super) index: &'a mut IndexBuilder,
    /// Blobs of the pack, by id and by identity.
    pub(super) blobs: &'a mut BlobTable,
    /// The id the next new blob gets.
    pub(super) next_blob: &'a mut u32,
    /// The blobs of this source copied so far.
    pub(super) copied: &'a mut HashMap<u32, u32>,
}

/// What a copy produced.
#[derive(Debug, Default)]
pub(super) struct CopiedMember {
    /// Extents, in member order, with blob ids of this pack.
    pub(super) extents: Vec<ExtentRecord>,
    /// Bytes per class.
    pub(super) table: ClassTable,
    /// Blobs copied, bases included.
    pub(super) blobs_added: u32,
    /// Extents over a blob this pack had.
    pub(super) blobs_reused: u32,
    /// Delta blobs copied.
    pub(super) delta_blobs: u32,
    /// Stored bytes copied, bases included.
    pub(super) copied_bytes: u64,
}

/// Copies what `record`, a member of `source`, needs into the pack: the blobs its extents reach, and
/// the base of each delta blob among them.
///
/// # Errors
/// Returns [`Error::Io`] if a read or a write fails, and [`Error::MalformedIndex`] if a record is
/// broken, which validation of `source` rules out. A failed read leaves the pack usable.
pub(super) fn copy_member(
    source: &Container,
    record: &MemberRecord,
    target: &mut CopyTarget<'_>,
) -> Result<CopiedMember, Error> {
    let index = source.index();
    let extents = index.member_extents(record)?;
    let mut copied = CopiedMember { extents: Vec::with_capacity(extents.len()), ..Default::default() };
    for extent in extents {
        let blob = usize::try_from(extent.blob_id)
            .ok()
            .and_then(|id| index.blobs().get(id))
            .ok_or_else(|| Error::MalformedIndex {
                section: "EXTN",
                detail: format!("blob {} does not exist", extent.blob_id),
            })?;
        let view = extent.view()?;
        let known = target.copied.get(&extent.blob_id).copied().or_else(|| {
            target.blobs.find(&blob.blake3)
        });
        let mut newly_stored = None;
        let blob_id = if let Some(id) = known {
            target.copied.insert(extent.blob_id, id);
            copied.blobs_reused += 1;
            id
        } else {
            let params = if blob.kind == BLOB_KIND_DELTA {
                let params = index.delta_params(blob)?;
                let (base_blob, base_stored) = base_for(source, params.base_blob, blob, target)?;
                if let Some(stored) = base_stored {
                    copied.table.add_base(class_of(blob, view), stored);
                    copied.copied_bytes += stored;
                    copied.blobs_added += 1;
                }
                copied.delta_blobs += 1;
                Some(DeltaParams { base_blob, ..params })
            } else {
                None
            };
            let (id, stored) = copy_blob(source, blob, params, target)?;
            target.copied.insert(extent.blob_id, id);
            copied.blobs_added += 1;
            copied.copied_bytes += stored;
            newly_stored = Some(stored);
            id
        };
        copied.table.add_extent(blob, view, extent.virt_len, newly_stored);
        copied.extents.push(ExtentRecord { blob_id, ..*extent });
    }
    Ok(copied)
}

/// The id in this pack of the base of `delta`, a delta blob of `source` whose base has id `base` there,
/// and the stored bytes copied for it: `None` if the pack had a blob that serves. One serves if it
/// holds the same canonical bytes, by the source's id or by identity, and is a plain blob of the
/// delta blob's `chunk_size` and of the base's length, so that every delta chunk meets the same base
/// chunk as in the source.
///
/// # Errors
/// As [`copy_member`]; [`Error::MalformedIndex`] for `BLOB` if `base` does not exist.
fn base_for(
    source: &Container,
    base: u32,
    delta: &BlobRecord,
    target: &mut CopyTarget<'_>,
) -> Result<(u32, Option<u64>), Error> {
    let index = source.index();
    let record = usize::try_from(base).ok().and_then(|id| index.blobs().get(id)).ok_or_else(|| {
        Error::MalformedIndex { section: "BLOB", detail: format!("base blob {base} does not exist") }
    })?;
    let serves = |id: &u32| {
        target.blobs.shape(*id).is_some_and(|shape| {
            shape.kind == BLOB_KIND_CHUNKED
                && shape.chunk_size == delta.chunk_size
                && shape.raw_len == record.raw_len
        })
    };
    let known = target.copied.get(&base).copied().filter(serves);
    if let Some(id) = known.or_else(|| target.blobs.find(&record.blake3).filter(serves)) {
        return Ok((id, None));
    }
    tracing::debug!(base, "copying the base of a delta blob");
    let (id, stored) = copy_blob(source, record, None, target)?;
    // A mapping that exists stays: it serves extents, and this blob serves the base.
    target.copied.entry(base).or_insert(id);
    Ok((id, Some(stored)))
}

/// Copies one blob, with `params` as its new `PARM` entry if it is a delta blob, whose base must be
/// in this pack already. Returns its id in this pack and its stored bytes.
fn copy_blob(
    source: &Container,
    blob: &BlobRecord,
    params: Option<DeltaParams>,
    target: &mut CopyTarget<'_>,
) -> Result<(u32, u64), Error> {
    if (blob.kind == BLOB_KIND_DELTA) != params.is_some() {
        return Err(Error::MalformedIndex {
            section: "BLOB",
            detail: format!("a blob of kind {} copied with base {params:?}", blob.kind),
        });
    }
    let index = source.index();
    let chain = index.chain(blob.chain_id)?;
    let start = target.sink.position();
    let chunks = index.blob_chunks(blob)?;
    let mut rebased = Vec::with_capacity(chunks.len());
    let mut buffer = Vec::new();
    for chunk in chunks {
        let len = usize::try_from(chunk.stored_len).map_err(|_| Error::MalformedIndex {
            section: "CHNK",
            detail: "a chunk does not fit the address space".to_owned(),
        })?;
        buffer.resize(len, 0);
        if let Err(error) = source.read_stored(chunk.file_offset, &mut buffer) {
            target.sink.rollback(start)?;
            return Err(error);
        }
        let file_offset = target.sink.position();
        target.sink.write(&buffer)?;
        rebased.push(ChunkRecord { file_offset, ..*chunk });
    }
    let stored = target.sink.position() - start;
    let chain_id = target.index.add_chain(chain)?;
    let (params_off, params_len) = match params {
        Some(params) => target.index.add_params(&params.encode())?,
        None => (0, 0),
    };
    let mut first_chunk = None;
    for chunk in rebased {
        let id = target.index.add_chunk(chunk);
        first_chunk.get_or_insert(id);
    }
    let id = target.index.add_blob(BlobRecord {
        first_chunk: first_chunk.unwrap_or(0),
        chain_id,
        params_off,
        params_len,
        ..*blob
    })?;
    *target.next_blob = id.checked_add(1).ok_or_else(|| Error::MalformedIndex {
        section: "BLOB",
        detail: "too many records for a u32 id".to_owned(),
    })?;
    let shape = BlobShape { kind: blob.kind, chunk_size: blob.chunk_size, raw_len: blob.raw_len };
    target.blobs.record(id, blob.blake3, shape)?;
    Ok((id, stored))
}
