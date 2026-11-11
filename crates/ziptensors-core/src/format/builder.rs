//! File: format/builder.rs
//! Purpose: serialization of an index, the writer's side of spec section 4.
//!
//! Main responsibilities:
//! - collect records and hand back the ids other records refer to by;
//! - start from the storage records of an existing index, [`IndexBuilder::seed`], for an edit that
//!   keeps its blobs, or from a subset of its blobs renumbered in order,
//!   [`IndexBuilder::seed_subset`], for an edit that drops the others;
//! - lay out the section table and the bodies in [`IndexBuilder::serialize`].
//!
//! Key structures: [`IndexBuilder`].
//!
//! Implementation notes: the builder checks only that ids fit their fields. Nothing here enforces
//! the invariants, which is the writer's job and is proven by parsing the result, and which lets
//! tests and hostile fixtures build invalid indexes through the `add_raw_*` and record calls. The
//! layout is deterministic: same records, same bytes.

use std::borrow::Cow;
use std::collections::HashMap;

use super::index::{
    BLOB, BODY_ALIGN, CHAN, CHNK, CHST, EXTN, FLAG_REQUIRED, Index, MEMB, PARM, SHAP, STRS,
    SectionSpec, TABLE_ENTRY_LEN, TABLE_HEADER_LEN, TENS,
};
use super::records::{
    BlobRecord, ChainRecord, ChunkRecord, ExtentRecord, MemberRecord, PARAMS_ALIGN_LEN, StrRef,
    TensorRecord, encode_step,
};
use super::wire::malformed;
use crate::chain::{BLOB_KIND_DELTA, Chain, RawStep, Step};
use crate::error::Error;

/// `section_count` of an index without `PARM`: the mandatory sections, which every index has.
const MANDATORY_COUNT: u32 = 9;

/// Collects the records of one index. Each `add_*` returns the id of what it added.
#[derive(Debug, Default)]
pub struct IndexBuilder {
    strings: Vec<u8>,
    chains: Vec<ChainRecord>,
    steps: Vec<RawStep>,
    blobs: Vec<BlobRecord>,
    chunks: Vec<ChunkRecord>,
    members: Vec<MemberRecord>,
    extents: Vec<ExtentRecord>,
    tensors: Vec<TensorRecord>,
    dims: Vec<u64>,
    /// The `PARM` heap; the section is written only when it is not empty.
    params: Vec<u8>,
    /// Chains already added, so that equal chains share one record.
    chain_ids: HashMap<Chain, u32>,
}

/// Fails unless a record re-added by [`IndexBuilder::seed`] got the id it had in the source.
fn check_same_id(section: &'static str, expected: usize, got: u64) -> Result<(), Error> {
    if u64::try_from(expected).ok() == Some(got) {
        return Ok(());
    }
    Err(malformed(section, format!("seeded record {expected} came back as id {got}")))
}

/// The next id of an array whose ids are `u32`.
fn next_u32(len: usize, section: &'static str) -> Result<u32, Error> {
    u32::try_from(len).map_err(|_| malformed(section, "too many records for a u32 id"))
}

impl IndexBuilder {
    /// An empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A builder holding every `CHST` and `CHAN` record of `index` with its id, each id checked.
    /// Seeded chains are known to [`IndexBuilder::add_chain`]: the first of equal chains is the one
    /// it would have returned.
    ///
    /// # Errors
    /// [`Error::MalformedIndex`] if a chain does not resolve or a record comes back with another id.
    fn with_chains_of(index: &Index) -> Result<Self, Error> {
        let mut builder = Self::new();
        for (expected, step) in index.steps().iter().enumerate() {
            let id = builder.add_raw_step(*step)?;
            check_same_id(CHST.tag, expected, u64::from(id))?;
        }
        for (expected, record) in index.chains().iter().enumerate() {
            let id = builder.add_raw_chain(*record)?;
            check_same_id(CHAN.tag, expected, u64::from(id))?;
            builder.chain_ids.entry(index.chain(id)?).or_insert(id);
        }
        Ok(builder)
    }

    /// A builder holding the storage records of `index`: its `CHST` and `CHAN` records, its `BLOB`
    /// records, its `CHNK` records and the `PARM` entry of each delta blob, re-added in id order so
    /// that every step, chain, blob and chunk keeps its id, which is checked record by record.
    /// Strings, members, extents, tensors and shapes are not carried: the caller adds the members
    /// it keeps. Seeded chains are known to [`IndexBuilder::add_chain`], so an equal chain added
    /// later shares the seeded record.
    ///
    /// `rebase` maps the `file_offset` of every chunk record to its offset in the new file: the
    /// identity for an edit that keeps the data region in place, or a move map. A delta blob's
    /// `PARM` entry is re-added with [`IndexBuilder::add_params`], so its `params_off` is that of
    /// the new heap; its base keeps its id and the entry its bytes.
    ///
    /// `index` must have passed [`super::validate`]; an unvalidated one may fail here.
    ///
    /// # Errors
    /// - Any error of `rebase`, which ends the seed.
    /// - [`Error::MalformedIndex`] if a chain or a delta blob's `PARM` entry does not resolve, or a
    ///   record comes back with another id, which a validated index and an empty builder rule out.
    pub fn seed(
        index: &Index,
        mut rebase: impl FnMut(u64) -> Result<u64, Error>,
    ) -> Result<Self, Error> {
        let mut builder = Self::with_chains_of(index)?;
        for (expected, blob) in index.blobs().iter().enumerate() {
            let mut record = *blob;
            if blob.kind == BLOB_KIND_DELTA {
                let params = index.delta_params(blob)?;
                (record.params_off, record.params_len) = builder.add_params(&params.encode())?;
            }
            let id = builder.add_blob(record)?;
            check_same_id(BLOB.tag, expected, u64::from(id))?;
        }
        for (expected, chunk) in index.chunks().iter().enumerate() {
            let file_offset = rebase(chunk.file_offset)?;
            let id = builder.add_chunk(ChunkRecord { file_offset, ..*chunk });
            check_same_id(CHNK.tag, expected, id)?;
        }
        tracing::debug!(
            chains = builder.chains.len(),
            blobs = builder.blobs.len(),
            chunks = builder.chunks.len(),
            params = builder.params.len(),
            "seeded index builder"
        );
        Ok(builder)
    }

    /// A builder holding only the blobs `kept` of `index` (ids in strictly increasing order), for
    /// an edit that drops the others from the file. Every `CHST` and `CHAN` record is carried with
    /// its id, as in [`IndexBuilder::seed`]. The kept blobs get the ids `0..kept.len()` in the order
    /// of `kept`, so a delta blob's base, which has a lower id in `index` (spec invariant 13), keeps
    /// a lower id; its `PARM` entry is re-added with the base's new id. The chunks of each kept blob
    /// are re-added after those of the blob before, with `first_chunk` set to match, and each chunk
    /// record's `file_offset` replaced by `rebase(record)`. Two blobs that shared chunk records get
    /// a copy each, with equal offsets, which invariant 4 allows.
    ///
    /// Returns the builder and, per blob id of `index`, its new id, or `None` for a dropped blob.
    ///
    /// `index` must have passed [`super::validate`].
    ///
    /// # Errors
    /// - Any error of `rebase`, which ends the seed.
    /// - [`Error::MalformedIndex`] if `kept` is not strictly increasing, names a blob `index` does
    ///   not have, keeps a delta blob without its base, or if a chain or `PARM` entry does not
    ///   resolve.
    pub(crate) fn seed_subset(
        index: &Index,
        kept: &[u32],
        mut rebase: impl FnMut(&ChunkRecord) -> Result<u64, Error>,
    ) -> Result<(Self, Vec<Option<u32>>), Error> {
        let mut builder = Self::with_chains_of(index)?;
        let mut ids = vec![None; index.blobs().len()];
        let mut previous: Option<u32> = None;
        for &old in kept {
            if previous.is_some_and(|previous| previous >= old) {
                return Err(malformed(BLOB.tag, format!("kept blob {old} is out of order")));
            }
            previous = Some(old);
            let missing = || malformed(BLOB.tag, format!("kept blob {old} does not exist"));
            let at = usize::try_from(old).map_err(|_| missing())?;
            let blob = *index.blobs().get(at).ok_or_else(missing)?;
            let mut record = blob;
            if blob.kind == BLOB_KIND_DELTA {
                let mut params = index.delta_params(&blob)?;
                let base = usize::try_from(params.base_blob).ok().and_then(|base| ids.get(base));
                let lost = format!("kept delta blob {old} lost its base {}", params.base_blob);
                params.base_blob =
                    base.copied().flatten().ok_or_else(|| malformed(BLOB.tag, lost))?;
                (record.params_off, record.params_len) = builder.add_params(&params.encode())?;
            }
            // Lossless: `usize` is at most 64 bits on every supported target.
            record.first_chunk = builder.chunks.len() as u64;
            for chunk in index.blob_chunks(&blob)? {
                let file_offset = rebase(chunk)?;
                builder.chunks.push(ChunkRecord { file_offset, ..*chunk });
            }
            ids[at] = Some(builder.add_blob(record)?);
        }
        tracing::debug!(
            blobs = builder.blobs.len(),
            dropped = index.blobs().len() - builder.blobs.len(),
            chunks = builder.chunks.len(),
            "seeded index builder with a subset of the blobs"
        );
        Ok((builder, ids))
    }

    /// Adds a string to the heap. Equal strings are not merged.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] for `STRS` if the heap outgrows `u32` offsets.
    pub fn add_string(&mut self, text: &str) -> Result<StrRef, Error> {
        self.add_raw_string(text.as_bytes())
    }

    /// Adds bytes to the heap without requiring UTF-8. For fixtures of invalid files.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] for `STRS` if the heap outgrows `u32` offsets.
    pub fn add_raw_string(&mut self, bytes: &[u8]) -> Result<StrRef, Error> {
        let off = next_u32(self.strings.len(), STRS.tag)?;
        let len = next_u32(bytes.len(), STRS.tag)?;
        self.strings.extend_from_slice(bytes);
        Ok(StrRef { off, len })
    }

    /// Adds a chain and its steps, or returns the id of the equal chain added before.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] if `CHAN` or `CHST` outgrow their `u32` ids.
    pub fn add_chain(&mut self, chain: Chain) -> Result<u32, Error> {
        if let Some(id) = self.chain_ids.get(&chain) {
            return Ok(*id);
        }
        let first_step = next_u32(self.steps.len(), CHST.tag)?;
        let mut step_count = 0;
        for step in chain.steps() {
            self.add_raw_step(Step::to_raw(step))?;
            step_count += 1;
        }
        let id = self.add_raw_chain(ChainRecord { first_step, step_count })?;
        self.chain_ids.insert(chain, id);
        Ok(id)
    }

    /// Adds one `CHST` record as it is and returns its index.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] for `CHST` if the id does not fit `u32`.
    pub fn add_raw_step(&mut self, step: RawStep) -> Result<u32, Error> {
        let id = next_u32(self.steps.len(), CHST.tag)?;
        self.steps.push(step);
        Ok(id)
    }

    /// Adds one `CHAN` record as it is and returns its `chain_id`.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] for `CHAN` if the id does not fit `u32`.
    pub fn add_raw_chain(&mut self, chain: ChainRecord) -> Result<u32, Error> {
        let id = next_u32(self.chains.len(), CHAN.tag)?;
        self.chains.push(chain);
        Ok(id)
    }

    /// Adds a blob and returns its `blob_id`.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] for `BLOB` if the id does not fit `u32`.
    pub fn add_blob(&mut self, blob: BlobRecord) -> Result<u32, Error> {
        let id = next_u32(self.blobs.len(), BLOB.tag)?;
        self.blobs.push(blob);
        Ok(id)
    }

    /// Adds a chunk and returns its index in `CHNK`.
    #[must_use]
    pub fn add_chunk(&mut self, chunk: ChunkRecord) -> u64 {
        let id = self.chunks.len();
        self.chunks.push(chunk);
        // Lossless: `usize` is at most 64 bits on every supported target.
        id as u64
    }

    /// Adds a member and returns its index in `MEMB`.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] for `MEMB` if the index does not fit `u32`.
    pub fn add_member(&mut self, member: MemberRecord) -> Result<u32, Error> {
        let id = next_u32(self.members.len(), MEMB.tag)?;
        self.members.push(member);
        Ok(id)
    }

    /// Adds an extent and returns its index in `EXTN`.
    #[must_use]
    pub fn add_extent(&mut self, extent: ExtentRecord) -> u64 {
        let id = self.extents.len();
        self.extents.push(extent);
        // Lossless: `usize` is at most 64 bits on every supported target.
        id as u64
    }

    /// Adds a tensor and returns its index in `TENS`.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] for `TENS` if the index does not fit `u32`.
    pub fn add_tensor(&mut self, tensor: TensorRecord) -> Result<u32, Error> {
        let id = next_u32(self.tensors.len(), TENS.tag)?;
        self.tensors.push(tensor);
        Ok(id)
    }

    /// Adds the dimensions of one tensor and returns their `shape_off`.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] for `SHAP` if the offset does not fit `u32`.
    pub fn add_shape(&mut self, dims: &[u64]) -> Result<u32, Error> {
        let shape_off = next_u32(self.dims.len(), SHAP.tag)?;
        self.dims.extend_from_slice(dims);
        Ok(shape_off)
    }

    /// Appends one entry to the `PARM` heap at the next multiple of 8 and returns its `params_off`
    /// and `params_len`. The gap before it is zero filled. Entries are not merged.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] for `PARM` if `bytes` is longer than a `u32` length.
    pub fn add_params(&mut self, bytes: &[u8]) -> Result<(u64, u32), Error> {
        let len = next_u32(bytes.len(), PARM.tag)?;
        let off = self.params.len().next_multiple_of(PARAMS_ALIGN_LEN);
        self.params.resize(off, 0);
        self.params.extend_from_slice(bytes);
        // Lossless: `usize` is at most 64 bits on every supported target.
        Ok((off as u64, len))
    }

    /// Serializes the decompressed index: the table of the nine mandatory sections in the order of
    /// spec section 4, then `PARM` with its required bit if any parameters were added, then their
    /// bodies, each 8-aligned, with zero fill.
    ///
    /// The result is not checked against the 1 GiB ceiling of the format; the footer of the file
    /// is, by [`super::Footer::check_layout`].
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let bodies = self.bodies();
        let table_end = TABLE_HEADER_LEN + bodies.len() * TABLE_ENTRY_LEN;
        let total = bodies.iter().fold(table_end, |end, (_, body)| {
            end.next_multiple_of(BODY_ALIGN) + body.len()
        });
        let mut out = Vec::with_capacity(total);
        let section_count = MANDATORY_COUNT + u32::from(!self.params.is_empty());
        out.extend_from_slice(&section_count.to_le_bytes());
        out.extend_from_slice(&0_u32.to_le_bytes());
        let mut offset = table_end;
        for (spec, body) in &bodies {
            offset = offset.next_multiple_of(BODY_ALIGN);
            let count = body.len() / spec.record_size;
            out.extend_from_slice(spec.tag.as_bytes());
            out.extend_from_slice(&FLAG_REQUIRED.to_le_bytes());
            // Lossless: `usize` is at most 64 bits on every supported target.
            out.extend_from_slice(&(offset as u64).to_le_bytes());
            out.extend_from_slice(&(body.len() as u64).to_le_bytes());
            out.extend_from_slice(&(count as u64).to_le_bytes());
            offset += body.len();
        }
        for (_, body) in &bodies {
            out.resize(out.len().next_multiple_of(BODY_ALIGN), 0);
            out.extend_from_slice(body);
        }
        out
    }

    /// The encoded bodies, in the order of the table of spec section 4, `PARM` last if present.
    fn bodies(&self) -> Vec<(SectionSpec, Cow<'_, [u8]>)> {
        let mut bodies = Vec::with_capacity(10);
        bodies.extend([
            (STRS, Cow::Borrowed(self.strings.as_slice())),
            (CHAN, self.chains.iter().copied().flat_map(ChainRecord::encode).collect()),
            (CHST, self.steps.iter().copied().flat_map(encode_step).collect()),
            (BLOB, self.blobs.iter().flat_map(BlobRecord::encode).collect()),
            (CHNK, self.chunks.iter().flat_map(ChunkRecord::encode).collect()),
            (MEMB, self.members.iter().flat_map(MemberRecord::encode).collect()),
            (EXTN, self.extents.iter().flat_map(ExtentRecord::encode).collect()),
            (TENS, self.tensors.iter().flat_map(TensorRecord::encode).collect()),
            (SHAP, self.dims.iter().flat_map(|dim| dim.to_le_bytes()).collect()),
        ]);
        if !self.params.is_empty() {
            bodies.push((PARM, Cow::Borrowed(self.params.as_slice())));
        }
        bodies
    }
}

#[cfg(test)]
mod tests {
    use super::IndexBuilder;
    use crate::chain::{Chain, ElementWidth, Layout, Step};
    use crate::format::records::ChunkRecord;
    use crate::error::Error;
    use crate::format::index::Index;
    use crate::format::limits::Limits;
    use crate::format::testkit::{Fixture, read_u64};

    /// The entry a hand-derivation of spec section 4 gives for section `tag`.
    fn entry(tag: [u8; 4], offset: u64, len: u64, count: u64) -> Vec<u8> {
        let mut bytes = tag.to_vec();
        bytes.extend_from_slice(&[1, 0, 0, 0]);
        bytes.extend_from_slice(&offset.to_le_bytes());
        bytes.extend_from_slice(&len.to_le_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
        bytes
    }

    #[test]
    fn empty_index_matches_the_hand_derived_bytes() {
        // Table: 8 + 9 × 32 = 296 bytes, a multiple of 8; all bodies empty.
        let mut expected = vec![9, 0, 0, 0, 0, 0, 0, 0];
        let tags = [*b"STRS", *b"CHAN", *b"CHST", *b"BLOB", *b"CHNK", *b"MEMB", *b"EXTN"];
        for tag in tags.iter().chain([b"TENS", b"SHAP"]) {
            expected.extend(entry(*tag, 296, 0, 0));
        }
        assert_eq!(IndexBuilder::new().serialize(), expected);
    }

    #[test]
    fn bodies_are_aligned_and_padded_with_zero() {
        let mut builder = IndexBuilder::new();
        let name = builder.add_string("abc").expect("fits");
        assert_eq!((name.off, name.len), (0, 3));
        let shape = builder.add_shape(&[0x0102]).expect("fits");
        assert_eq!(shape, 0);

        let mut expected = vec![9, 0, 0, 0, 0, 0, 0, 0];
        expected.extend(entry(*b"STRS", 296, 3, 3));
        for tag in [b"CHAN", b"CHST", b"BLOB", b"CHNK", b"MEMB", b"EXTN", b"TENS"] {
            expected.extend(entry(*tag, 304, 0, 0));
        }
        expected.extend(entry(*b"SHAP", 304, 8, 1));
        expected.extend_from_slice(b"abc");
        expected.extend_from_slice(&[0; 5]);
        expected.extend_from_slice(&[0x02, 0x01, 0, 0, 0, 0, 0, 0]);
        assert_eq!(builder.serialize(), expected);
    }

    #[test]
    fn parm_is_written_last_with_its_required_bit_and_aligned_entries() {
        let mut builder = IndexBuilder::new();
        assert_eq!(builder.add_params(&[0xAA; 3]).expect("fits"), (0, 3));
        assert_eq!(builder.add_params(&[0xBB; 16]).expect("fits"), (8, 16));
        let raw = builder.serialize();
        // Table: 8 + 10 × 32 = 328 bytes; nine empty bodies there, then the 24-byte heap.
        let mut expected = vec![10, 0, 0, 0, 0, 0, 0, 0];
        let tags = [*b"STRS", *b"CHAN", *b"CHST", *b"BLOB", *b"CHNK", *b"MEMB", *b"EXTN"];
        for tag in tags.iter().chain([b"TENS", b"SHAP"]) {
            expected.extend(entry(*tag, 328, 0, 0));
        }
        expected.extend(entry(*b"PARM", 328, 24, 24));
        expected.extend_from_slice(&[0xAA; 3]);
        expected.extend_from_slice(&[0; 5]);
        expected.extend_from_slice(&[0xBB; 16]);
        assert_eq!(raw, expected);
        let index = Index::parse(&raw, &Limits::default()).expect("valid");
        assert!(index.params_required());
    }

    #[test]
    fn equal_chains_share_a_record_and_steps_keep_order() {
        let whole = Chain::new(&[Step::Layout(Layout::Whole)]).expect("valid");
        let rotated = Chain::new(&[
            Step::Rotl1(ElementWidth::Four),
            Step::Layout(Layout::Planes(ElementWidth::Four)),
        ])
        .expect("valid");
        let mut builder = IndexBuilder::new();
        assert_eq!(builder.add_chain(whole).expect("fits"), 0);
        assert_eq!(builder.add_chain(rotated).expect("fits"), 1);
        assert_eq!(builder.add_chain(whole).expect("fits"), 0);
        assert_eq!(builder.add_chain(rotated).expect("fits"), 1);

        let index = Index::parse(&builder.serialize(), &Limits::default()).expect("valid");
        assert_eq!(index.chains().len(), 2);
        assert_eq!(index.steps().len(), 3);
        assert_eq!(index.chain(0).expect("valid"), whole);
        assert_eq!(index.chain(1).expect("valid"), rotated);
    }

    #[test]
    fn ids_count_from_zero_per_section() {
        let fixture = Fixture::valid();
        let mut builder = IndexBuilder::new();
        for (expected, blob) in fixture.blobs.iter().enumerate() {
            assert_eq!(usize::try_from(builder.add_blob(*blob).expect("fits")), Ok(expected));
        }
        for (expected, chunk) in fixture.chunks.iter().enumerate() {
            assert_eq!(builder.add_chunk(*chunk), expected as u64);
        }
        for (expected, extent) in fixture.extents.iter().enumerate() {
            assert_eq!(builder.add_extent(*extent), expected as u64);
        }
        for (expected, member) in fixture.members.iter().enumerate() {
            assert_eq!(usize::try_from(builder.add_member(*member).expect("fits")), Ok(expected));
        }
        for (expected, tensor) in fixture.tensors.iter().enumerate() {
            assert_eq!(usize::try_from(builder.add_tensor(*tensor).expect("fits")), Ok(expected));
        }
        assert_eq!(builder.add_shape(&[1, 2]).expect("fits"), 0);
        assert_eq!(builder.add_shape(&[]).expect("fits"), 2);
        assert_eq!(builder.add_shape(&[3]).expect("fits"), 2);
    }

    #[test]
    fn serialized_index_round_trips_to_equal_records_and_equal_bytes() {
        let fixture = Fixture::valid();
        let raw = fixture.serialize();
        let index = Index::parse(&raw, &Limits::default()).expect("valid");
        assert_eq!(index.strings(), fixture.strings.as_slice());
        assert_eq!(index.chains(), fixture.chains.as_slice());
        assert_eq!(index.steps(), fixture.steps.as_slice());
        assert_eq!(index.blobs(), fixture.blobs.as_slice());
        assert_eq!(index.chunks(), fixture.chunks.as_slice());
        assert_eq!(index.members(), fixture.members.as_slice());
        assert_eq!(index.extents(), fixture.extents.as_slice());
        assert_eq!(index.tensors(), fixture.tensors.as_slice());
        assert_eq!(index.dims(), fixture.dims.as_slice());
        assert_eq!(Fixture::from_index(&index).serialize(), raw);
    }

    /// Seeds from `fixture` with `rebase` and parses the result.
    fn seeded(fixture: &Fixture, rebase: impl FnMut(u64) -> Result<u64, Error>) -> Index {
        let source = fixture.check().expect("a valid fixture");
        let builder = IndexBuilder::seed(&source, rebase).expect("seeds");
        Index::parse(&builder.serialize(), &Limits::default()).expect("parses")
    }

    #[test]
    fn a_seed_carries_storage_records_with_their_ids() {
        for fixture in [Fixture::valid(), Fixture::with_delta()] {
            let index = seeded(&fixture, Ok);
            assert_eq!(index.steps(), fixture.steps.as_slice());
            assert_eq!(index.chains(), fixture.chains.as_slice());
            assert_eq!(index.blobs(), fixture.blobs.as_slice());
            assert_eq!(index.chunks(), fixture.chunks.as_slice());
            assert_eq!(index.params(), fixture.params.as_slice());
            assert_eq!(index.params_required(), !fixture.params.is_empty());
            assert_eq!(index.strings(), []);
            assert_eq!(index.members(), []);
            assert_eq!(index.extents(), []);
            assert_eq!(index.tensors(), []);
            assert_eq!(index.dims(), []);
        }
    }

    #[test]
    fn a_seed_rebases_every_chunk_offset() {
        let fixture = Fixture::with_delta();
        let index = seeded(&fixture, |offset| Ok(offset + 1000));
        let offsets: Vec<u64> = index.chunks().iter().map(|chunk| chunk.file_offset).collect();
        let expected: Vec<u64> =
            fixture.chunks.iter().map(|chunk| chunk.file_offset + 1000).collect();
        assert_eq!(offsets, expected);
    }

    #[test]
    fn a_failed_rebase_ends_the_seed() {
        let source = Fixture::valid().check().expect("a valid fixture");
        let error = IndexBuilder::seed(&source, |_| Err(Error::Zstd { detail: "x".into() }))
            .expect_err("the rebase fails");
        assert!(matches!(error, Error::Zstd { .. }), "{error:?}");
    }

    #[test]
    fn a_subset_seed_renumbers_kept_blobs_in_order_and_remaps_delta_bases() {
        let fixture = Fixture::with_delta();
        let source = fixture.check().expect("a valid fixture");
        // Blob 1 is dropped; blob 2, a delta over blob 0, becomes blob 1.
        let (builder, ids) =
            IndexBuilder::seed_subset(&source, &[0, 2], |chunk| Ok(chunk.file_offset + 1000))
                .expect("seeds");
        assert_eq!(ids, [Some(0), None, Some(1)]);
        let index = Index::parse(&builder.serialize(), &Limits::default()).expect("parses");
        assert_eq!(index.steps(), fixture.steps.as_slice());
        assert_eq!(index.chains(), fixture.chains.as_slice());
        let blobs = index.blobs();
        assert_eq!(blobs.len(), 2);
        assert_eq!(blobs[0], fixture.blobs[0]);
        assert_eq!((blobs[1].first_chunk, blobs[1].chunk_count), (3, 2));
        assert_eq!(blobs[1].kind, fixture.blobs[2].kind);
        let params = index.delta_params(&blobs[1]).expect("resolves");
        assert_eq!((params.base_blob, params.base_offset), (0, 4));
        assert!(index.params_required());
        let kept_chunks = [0, 1, 2, 5, 6].map(|at| fixture.chunks[at]);
        let expected: Vec<_> = kept_chunks
            .iter()
            .map(|chunk| ChunkRecord { file_offset: chunk.file_offset + 1000, ..*chunk })
            .collect();
        assert_eq!(index.chunks(), expected.as_slice());
    }

    #[test]
    fn a_subset_seed_refuses_a_delta_without_its_base_and_bad_ids() {
        let source = Fixture::with_delta().check().expect("a valid fixture");
        let malformed = |kept: &[u32]| {
            let error = IndexBuilder::seed_subset(&source, kept, |chunk| Ok(chunk.file_offset))
                .expect_err("refused");
            assert!(matches!(error, Error::MalformedIndex { section: "BLOB", .. }), "{error:?}");
        };
        malformed(&[1, 2]);
        malformed(&[2, 0]);
        malformed(&[0, 0]);
        malformed(&[3]);
        let (builder, ids) =
            IndexBuilder::seed_subset(&source, &[], |chunk| Ok(chunk.file_offset)).expect("seeds");
        assert_eq!(ids, [None, None, None]);
        let index = Index::parse(&builder.serialize(), &Limits::default()).expect("parses");
        assert_eq!((index.blobs(), index.chunks()), ([].as_slice(), [].as_slice()));
        assert!(!index.params_required(), "no delta blob, no PARM");
    }

    #[test]
    fn a_seeded_chain_is_shared_by_an_equal_chain_added_later() {
        let fixture = Fixture::valid();
        let source = fixture.check().expect("a valid fixture");
        let mut builder = IndexBuilder::seed(&source, Ok).expect("seeds");
        for id in 0..u32::try_from(fixture.chains.len()).expect("few") {
            assert_eq!(builder.add_chain(source.chain(id).expect("resolves")).expect("fits"), id);
        }
        let new = Chain::new(&[
            Step::Rotl1(ElementWidth::Eight),
            Step::Layout(Layout::Planes(ElementWidth::Eight)),
        ])
        .expect("valid");
        let next = u32::try_from(fixture.chains.len()).expect("few");
        assert_eq!(builder.add_chain(new).expect("fits"), next);
    }

    #[test]
    fn serialized_length_and_table_agree() {
        let raw = Fixture::valid().serialize();
        let last_entry = 8 + 8 * 32;
        let end = read_u64(&raw, last_entry + 8) + read_u64(&raw, last_entry + 16);
        assert_eq!(end, raw.len() as u64);
    }
}
