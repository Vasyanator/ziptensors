//! File: read/testkit.rs
//! Purpose: builds real containers in memory for reader tests, until the writer exists. Test only.
//!
//! Main responsibilities:
//! - [`ContainerBuilder`] lays out header, chunks, index and footer with `IndexBuilder` and
//!   `encode_chunk` (`encode_delta_chunk` for delta blobs), and keeps canonical bytes so each member
//!   has a reference to compare with;
//! - [`MemorySource`], a [`RandomAccess`] over a byte vector that records every read;
//! - [`Rng`], [`tensor_like`] and small chain constructors for seeded data.
//!
//! Key structures: [`ContainerBuilder`], [`ExtentSpec`], [`MemorySource`], [`Rng`].
//!
//! Implementation notes: an integration test includes this file with
//! `#[path = "../src/read/testkit.rs"] mod testkit;` after importing the crate's modules at its root,
//! which is what makes the `crate::` paths below resolve there too; see `tests/read_memory.rs`.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::chain::{
    BLOB_KIND_DELTA, Chain, DeltaOp, ElementWidth, FEATURE_DELTA, Layout, Step, VIEW_IDENTITY, View,
};
use crate::codec::{StreamPolicy, Workspace, encode_chunk, encode_delta_chunk, widen_into};
use crate::error::Error;
use crate::format::{
    BlobRecord, ChunkRecord, DeltaParams, ExtentRecord, FOOTER_LEN, Footer, HEADER_LEN, Header,
    IndexBuilder, MemberRecord, TensorRecord, parse_footer,
};
use crate::io::RandomAccess;
use crate::types::{Dtype, Profile};

/// A small value as a length.
fn len_of(value: u64) -> usize {
    usize::try_from(value).expect("a test value fits a length")
}

/// Splitmix64: small, seedable and the same everywhere.
#[derive(Debug, Clone)]
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    /// The next value.
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..bound`, `bound` not 0.
    pub(crate) fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

/// Bytes shaped like tensor data: every `width`-th byte is low entropy, the others are noise, so
/// `planes` compresses some streams and stores others.
pub(crate) fn tensor_like(seed: u64, len: usize, width: usize) -> Vec<u8> {
    let mut rng = Rng(seed);
    (0..len)
        .map(|at| {
            let value = rng.next_u64().to_le_bytes()[3];
            if at % width == width - 1 { value & 3 } else { value }
        })
        .collect()
}

/// `whole`.
pub(crate) fn whole() -> Chain {
    Chain::new(&[Step::Layout(Layout::Whole)]).expect("valid chain")
}

/// `planes(width)`.
pub(crate) fn planes(width: ElementWidth) -> Chain {
    Chain::new(&[Step::Layout(Layout::Planes(width))]).expect("valid chain")
}

/// `rotl1(rotate)` then `layout`.
pub(crate) fn rotated(rotate: ElementWidth, layout: Layout) -> Chain {
    Chain::new(&[Step::Rotl1(rotate), Step::Layout(layout)]).expect("valid chain")
}

/// `delta_xor(width)` in front of `chain`.
pub(crate) fn delta(width: ElementWidth, chain: Chain) -> Chain {
    let steps: Vec<Step> =
        [Step::Delta(DeltaOp::Xor(width))].into_iter().chain(chain.steps()).collect();
    Chain::new(&steps).expect("valid chain")
}

/// `base` with a few low bits changed, the way a finetune differs from its base, so that a delta
/// residual is mostly zero.
pub(crate) fn finetune_of(seed: u64, base: &[u8]) -> Vec<u8> {
    let mut rng = Rng(seed);
    let mut target = base.to_vec();
    for byte in target.iter_mut().step_by(3) {
        *byte ^= rng.next_u64().to_le_bytes()[0] & 3;
    }
    target
}

/// One extent of a member, the virtual offsets following from the order.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ExtentSpec {
    /// Id from [`ContainerBuilder::add_blob`].
    pub(crate) blob: u32,
    /// Start in the blob.
    pub(crate) blob_offset: u64,
    /// Length in the member.
    pub(crate) virt_len: u64,
    /// View of the extent.
    pub(crate) view: View,
}

/// An identity extent.
pub(crate) fn identity(blob: u32, blob_offset: u64, virt_len: u64) -> ExtentSpec {
    ExtentSpec { blob, blob_offset, virt_len, view: View::Identity }
}

/// Builds one container. Bytes go straight after the header, in the order blobs are added.
#[derive(Debug)]
pub(crate) struct ContainerBuilder {
    file: Vec<u8>,
    index: IndexBuilder,
    workspace: Workspace,
    /// Canonical bytes by blob id.
    blobs: Vec<Vec<u8>>,
    /// Records by blob id.
    records: Vec<BlobRecord>,
    /// Next indexes into `CHNK`, `EXTN` and `TENS`.
    next_chunk: u64,
    next_extent: u64,
    next_tensor: u32,
}

impl ContainerBuilder {
    /// An empty container.
    pub(crate) fn new() -> Self {
        Self {
            file: Header::current().to_bytes().to_vec(),
            index: IndexBuilder::new(),
            workspace: Workspace::new(),
            blobs: Vec::new(),
            records: Vec::new(),
            next_chunk: 0,
            next_extent: 0,
            next_tensor: 0,
        }
    }

    /// Encodes `canonical` as a new blob, cut every `chunk_size` bytes, and returns its id.
    pub(crate) fn add_blob(
        &mut self,
        canonical: Vec<u8>,
        chain: Chain,
        chunk_size: u32,
        policy: StreamPolicy,
    ) -> u32 {
        let chain_id = self.index.add_chain(chain).expect("few chains");
        let first_chunk = self.next_chunk;
        let mut chunk_count = 0;
        let mut stored = Vec::new();
        for raw in canonical.chunks(chunk_size as usize) {
            stored.clear();
            let meta = encode_chunk(raw, &chain, &policy, &mut self.workspace, &mut stored)
                .expect("encodable chunk");
            self.next_chunk = self.index.add_chunk(ChunkRecord {
                file_offset: self.file.len() as u64,
                xxh3_decoded: meta.xxh3_decoded,
                stored_len: meta.stored_len,
            }) + 1;
            self.file.extend_from_slice(&stored);
            chunk_count += 1;
        }
        let record = BlobRecord {
            blake3: [0; 32],
            raw_len: canonical.len() as u64,
            first_chunk,
            chunk_count,
            chunk_size,
            chain_id,
            kind: 0,
            dtype_hint: Dtype::U8,
            params_off: 0,
            params_len: 0,
        };
        self.push_blob(record, canonical)
    }

    /// Encodes `canonical` as a delta blob of `chain`, which has a delta step, over the bytes of blob
    /// `base` from `base_offset`, with the base's `chunk_size`, and returns its id. The rules of spec
    /// section 7a hold if `base` is a plain blob and `base_offset` is a multiple of its `chunk_size`
    /// with room for `canonical`; this does not check them, so tests can break them.
    pub(crate) fn add_delta_blob(
        &mut self,
        canonical: Vec<u8>,
        base: u32,
        base_offset: u64,
        chain: Chain,
        policy: StreamPolicy,
    ) -> u32 {
        let chain_id = self.index.add_chain(chain).expect("few chains");
        let chunk_size = self.records[base as usize].chunk_size;
        // Taken out and put back, so that a large base is not copied while `self` is borrowed.
        let base_canonical = std::mem::take(&mut self.blobs[base as usize]);
        let first_chunk = self.next_chunk;
        let mut chunk_count = 0;
        let mut stored = Vec::new();
        let mut base_chunks = base_canonical[len_of(base_offset)..].chunks(chunk_size as usize);
        for raw in canonical.chunks(chunk_size as usize) {
            let base_chunk = base_chunks.next().expect("the base covers the delta blob");
            stored.clear();
            let base_part = &base_chunk[..raw.len()];
            let meta =
                encode_delta_chunk(raw, base_part, &chain, &policy, &mut self.workspace, &mut stored)
                    .expect("encodable chunk");
            self.next_chunk = self.index.add_chunk(ChunkRecord {
                file_offset: self.file.len() as u64,
                xxh3_decoded: meta.xxh3_decoded,
                stored_len: meta.stored_len,
            }) + 1;
            self.file.extend_from_slice(&stored);
            chunk_count += 1;
        }
        self.blobs[base as usize] = base_canonical;
        let entry = DeltaParams { base_blob: base, base_offset };
        let (params_off, params_len) = self.index.add_params(&entry.encode()).expect("small heap");
        let record = BlobRecord {
            blake3: [0; 32],
            raw_len: canonical.len() as u64,
            first_chunk,
            chunk_count,
            chunk_size,
            chain_id,
            kind: BLOB_KIND_DELTA,
            dtype_hint: Dtype::U8,
            params_off,
            params_len,
        };
        self.push_blob(record, canonical)
    }

    /// Adds a delta blob over the chunk records of `base` itself, paired with `base` at offset 0,
    /// under `chain`. The format allows chunks shared this way. If `chain` without its delta step is
    /// the base's chain, the residual is the base and the content is all zero; that is what is
    /// recorded, while the chunk hashes stay those of the base, so only `Verify::Off` reads it.
    pub(crate) fn share_chunks_as_delta(&mut self, base: u32, chain: Chain) -> u32 {
        let chain_id = self.index.add_chain(chain).expect("few chains");
        let entry = DeltaParams { base_blob: base, base_offset: 0 };
        let (params_off, params_len) = self.index.add_params(&entry.encode()).expect("small heap");
        let base_record = self.records[base as usize];
        let record = BlobRecord {
            chain_id,
            kind: BLOB_KIND_DELTA,
            params_off,
            params_len,
            ..base_record
        };
        self.push_blob(record, vec![0; len_of(base_record.raw_len)])
    }

    /// Adds a blob over the chunk records of `blob`, under another chain. The format allows this;
    /// `canonical` is what the caller says the content is.
    pub(crate) fn share_chunks(&mut self, blob: u32, chain: Chain, canonical: Vec<u8>) -> u32 {
        let chain_id = self.index.add_chain(chain).expect("few chains");
        let record = BlobRecord { chain_id, ..self.records[blob as usize] };
        self.push_blob(record, canonical)
    }

    /// Records a blob and returns its id.
    fn push_blob(&mut self, record: BlobRecord, canonical: Vec<u8>) -> u32 {
        let id = self.index.add_blob(record).expect("few blobs");
        self.records.push(record);
        self.blobs.push(canonical);
        id
    }

    /// The bytes `extents` describe, by the definition of spec section 8.
    pub(crate) fn content(&self, extents: &[ExtentSpec]) -> Vec<u8> {
        let mut content = Vec::new();
        for extent in extents {
            let start = len_of(extent.blob_offset);
            let viewed_len = len_of(extent.virt_len);
            let narrow_len = viewed_len / len_of(extent.view.length_factor());
            let bytes = &self.blobs[extent.blob as usize][start..][..narrow_len];
            let mut viewed = vec![0; viewed_len];
            widen_into(extent.view, bytes, &mut viewed).expect("whole elements");
            content.extend_from_slice(&viewed);
        }
        content
    }

    /// Adds a member over `extents` with the tensors `(name, offset, len)`, and returns its content.
    pub(crate) fn add_member_with_tensors(
        &mut self,
        name: &str,
        extents: &[ExtentSpec],
        tensors: &[(&str, u64, u64)],
    ) -> Vec<u8> {
        let content = self.content(extents);
        let hash = *blake3::hash(&content).as_bytes();
        self.add_member_records(name, extents, tensors, content.len() as u64, hash);
        content
    }

    /// Adds a member with the given length and hash, whatever `extents` say. For members that break
    /// a rule on purpose, which have no content to compute.
    pub(crate) fn add_member_records(
        &mut self,
        name: &str,
        extents: &[ExtentSpec],
        tensors: &[(&str, u64, u64)],
        file_len: u64,
        blake3: [u8; 32],
    ) {
        let first_extent = self.next_extent;
        let mut virt_offset = 0;
        for extent in extents {
            let view_id = if extent.view == View::Identity {
                VIEW_IDENTITY
            } else {
                extent.view.to_wire()
            };
            self.next_extent = self.index.add_extent(ExtentRecord {
                virt_offset,
                virt_len: extent.virt_len,
                blob_id: extent.blob,
                view_id,
                blob_offset: extent.blob_offset,
            }) + 1;
            virt_offset += extent.virt_len;
        }
        let first_tensor = self.next_tensor;
        for (tensor, offset, len) in tensors {
            let name = self.index.add_string(tensor).expect("short name");
            let shape_off = self.index.add_shape(&[*len]).expect("short shape");
            self.next_tensor = self
                .index
                .add_tensor(TensorRecord {
                    name,
                    virt_offset: *offset,
                    virt_len: *len,
                    shape_off,
                    ndim: 1,
                    dtype: Dtype::U8,
                })
                .expect("few tensors")
                + 1;
        }
        let name = self.index.add_string(name).expect("short name");
        self.index
            .add_member(MemberRecord {
                name,
                file_len,
                blake3,
                first_extent,
                extent_count: extents.len().try_into().expect("few extents"),
                first_tensor,
                tensor_count: tensors.len().try_into().expect("few tensors"),
                profile: Profile::Opaque,
            })
            .expect("few members");
    }

    /// Adds a member with no tensors and returns its content.
    pub(crate) fn add_member(&mut self, name: &str, extents: &[ExtentSpec]) -> Vec<u8> {
        self.add_member_with_tensors(name, extents, &[])
    }

    /// Appends the index and the footer and returns the file. Sets required feature bit 0 in the
    /// header if a delta blob was added, as spec section 7a requires.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        if self.records.iter().any(|record| record.kind == BLOB_KIND_DELTA) {
            let header = Header { required_features: FEATURE_DELTA, ..Header::current() };
            self.file[..len_of(HEADER_LEN)].copy_from_slice(&header.to_bytes());
        }
        finish_file(self.file, &self.index.serialize())
    }

}

/// Appends a compressed index and its footer to `file`.
pub(crate) fn finish_file(mut file: Vec<u8>, raw_index: &[u8]) -> Vec<u8> {
    let stored = zstd::bulk::compress(raw_index, 3).expect("zstd compresses an index");
    let index_offset = file.len() as u64;
    file.extend_from_slice(&stored);
    let footer = Footer::for_index(index_offset, stored.len() as u64, raw_index);
    file.extend_from_slice(&footer.to_bytes());
    file
}

/// The footer of `file`.
pub(crate) fn footer_of(file: &[u8]) -> Footer {
    let footer = file[file.len() - len_of(FOOTER_LEN)..].try_into().expect("64 bytes");
    parse_footer(footer).expect("a valid footer")
}

/// A byte vector as a source, recording `(offset, len)` of every read.
#[derive(Debug)]
pub(crate) struct MemorySource {
    bytes: Vec<u8>,
    reads: Mutex<Vec<(u64, usize)>>,
    recording: AtomicBool,
}

impl MemorySource {
    /// A source over `bytes`, recording.
    pub(crate) fn new(bytes: Vec<u8>) -> Self {
        Self { bytes, reads: Mutex::new(Vec::new()), recording: AtomicBool::new(true) }
    }

    /// A source over `bytes` that records nothing, so a long test does not grow a log.
    pub(crate) fn silent(bytes: Vec<u8>) -> Self {
        let source = Self::new(bytes);
        source.recording.store(false, Ordering::Relaxed);
        source
    }

    /// Returns the reads so far and forgets them.
    pub(crate) fn take_reads(&self) -> Vec<(u64, usize)> {
        std::mem::take(&mut self.reads.lock().expect("no test panics while recording"))
    }
}

impl RandomAccess for MemorySource {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_exact_at(&self, off: u64, buf: &mut [u8]) -> Result<(), Error> {
        let len = buf.len() as u64;
        let file_len = self.len();
        let range = usize::try_from(off).ok().filter(|_| off.saturating_add(len) <= file_len);
        let start = range.ok_or(Error::OutOfBounds { offset: off, len, file_len })?;
        buf.copy_from_slice(&self.bytes[start..start + buf.len()]);
        if self.recording.load(Ordering::Relaxed) {
            self.reads.lock().expect("no test panics while recording").push((off, buf.len()));
        }
        Ok(())
    }
}
