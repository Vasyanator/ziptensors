//! File: format/testkit.rs
//! Purpose: shared fixtures of the format tests. Compiled only for tests.
//!
//! Main responsibilities:
//! - [`lay_out`] writes a raw index from hand-given section table values, independently of
//!   `IndexBuilder`, so that a table can be made invalid in any way;
//! - [`Fixture`] is a small valid index as open record arrays, for tests to corrupt one field, with a
//!   variant that adds a delta blob.
//!
//! Key structures: [`Section`], [`Fixture`].
//!
//! Implementation notes: all data is generated here, nothing is read from disk.

use super::builder::IndexBuilder;
use super::footer::Footer;
use super::header::Header;
use super::index::Index;
use super::limits::Limits;
use super::records::{
    BlobRecord, ChainRecord, ChunkRecord, DELTA_PARAMS_LEN, DeltaParams, ExtentRecord,
    MemberRecord, StrRef, TensorRecord,
};
use super::validate::validate;
use crate::chain::{
    BLOB_KIND_DELTA, FEATURE_DELTA, RawStep, STEP_DELTA_XOR, STEP_PLANES, STEP_ROTL1, STEP_WHOLE,
    VIEW_F16_TO_F32,
};
use crate::error::Error;
use crate::types::{Dtype, Profile};

/// Reads a little-endian `u64` at byte `at`.
pub(super) fn read_u64(raw: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(raw[at..at + 8].try_into().expect("8 bytes"))
}

/// One section as it should appear in a hand-laid index. `count` is written as given.
#[derive(Debug, Clone)]
pub(super) struct Section {
    pub(super) tag: [u8; 4],
    pub(super) flags: u32,
    pub(super) body: Vec<u8>,
    pub(super) count: u64,
}

impl Section {
    pub(super) fn new(tag: [u8; 4], flags: u32, body: Vec<u8>, count: u64) -> Self {
        Self { tag, flags, body, count }
    }
}

/// The nine mandatory sections, required and empty.
pub(super) fn mandatory_sections() -> Vec<Section> {
    [b"STRS", b"CHAN", b"CHST", b"BLOB", b"CHNK", b"MEMB", b"EXTN", b"TENS", b"SHAP"]
        .map(|tag| Section::new(*tag, 1, Vec::new(), 0))
        .to_vec()
}

/// Writes a raw index by the layout of spec section 4: bodies in order, each 8-aligned.
pub(super) fn lay_out(sections: &[Section]) -> Vec<u8> {
    let table_end = 8 + 32 * sections.len();
    let mut raw = Vec::new();
    raw.extend_from_slice(&u32::try_from(sections.len()).expect("few").to_le_bytes());
    raw.extend_from_slice(&[0; 4]);
    let mut offset = table_end;
    for section in sections {
        offset = offset.next_multiple_of(8);
        raw.extend_from_slice(&section.tag);
        raw.extend_from_slice(&section.flags.to_le_bytes());
        raw.extend_from_slice(&(offset as u64).to_le_bytes());
        raw.extend_from_slice(&(section.body.len() as u64).to_le_bytes());
        raw.extend_from_slice(&section.count.to_le_bytes());
        offset += section.body.len();
    }
    for section in sections {
        raw.resize(raw.len().next_multiple_of(8), 0);
        raw.extend_from_slice(&section.body);
    }
    raw
}

/// Footer offset that [`Fixture::check`] uses: the data region of a fixture is `64..200`.
pub(super) const DATA_END: u64 = 200;
/// Stored index length that [`Fixture::check`] uses.
const STORED_LEN: u64 = 100;
/// File length that [`Fixture::check`] uses.
pub(super) const FILE_LEN: u64 = DATA_END + STORED_LEN + 64;

/// An index as open arrays. Records hold whatever a test puts there.
#[derive(Debug, Clone)]
pub(super) struct Fixture {
    pub(super) strings: Vec<u8>,
    pub(super) steps: Vec<RawStep>,
    pub(super) chains: Vec<ChainRecord>,
    pub(super) blobs: Vec<BlobRecord>,
    pub(super) chunks: Vec<ChunkRecord>,
    pub(super) members: Vec<MemberRecord>,
    pub(super) extents: Vec<ExtentRecord>,
    pub(super) tensors: Vec<TensorRecord>,
    pub(super) dims: Vec<u64>,
    /// The `PARM` heap, written as one entry at offset 0 when not empty, with the required bit.
    pub(super) params: Vec<u8>,
    /// `required_features` of the header [`Fixture::check`] validates against.
    pub(super) required_features: u64,
}

/// A record pointing at the heap of [`Fixture::valid`].
const fn name(off: u32, len: u32) -> StrRef {
    StrRef { off, len }
}

/// A chunk record of [`Fixture::valid`]. No test here decodes data, so the hash is arbitrary.
fn chunk(file_offset: u64, stored_len: u32) -> ChunkRecord {
    ChunkRecord { file_offset, xxh3_decoded: 0, stored_len }
}

/// Blob 0 of [`Fixture::valid`].
fn first_blob() -> BlobRecord {
    BlobRecord {
        blake3: [0; 32],
        raw_len: 10,
        first_chunk: 0,
        chunk_count: 3,
        chunk_size: 4,
        chain_id: 0,
        kind: 0,
        dtype_hint: Dtype::U8,
        params_off: 0,
        params_len: 0,
    }
}

/// Member 0 of [`Fixture::valid`].
fn first_member() -> MemberRecord {
    MemberRecord {
        name: name(0, 5),
        file_len: 26,
        blake3: [7; 32],
        first_extent: 0,
        extent_count: 2,
        first_tensor: 0,
        tensor_count: 2,
        profile: Profile::Safetensors,
    }
}

impl Fixture {
    /// A valid index. Blob 0 is `whole`, 10 raw bytes in chunks of 4. Blob 1 is
    /// `rotl1(4), planes(4)` (4 streams), 16 raw bytes in chunks of 8. Member `a.bin` is 26 bytes:
    /// blob 0, then 8 bytes of blob 1 widened to 16, with tensors `t0` and `t1`. Member `b.bin` is
    /// the first 4 bytes of blob 1. Member `c.bin` is empty.
    pub(super) fn valid() -> Self {
        let (blob, member) = (first_blob(), first_member());
        let extent = ExtentRecord {
            virt_offset: 0,
            virt_len: 10,
            blob_id: 0,
            view_id: 0,
            blob_offset: 0,
        };
        Self {
            strings: b"a.binb.binc.bint0t1".to_vec(),
            steps: vec![
                RawStep { codec_id: STEP_WHOLE, param0: 0, param1: 0 },
                RawStep { codec_id: STEP_ROTL1, param0: 4, param1: 0 },
                RawStep { codec_id: STEP_PLANES, param0: 4, param1: 0 },
            ],
            chains: vec![
                ChainRecord { first_step: 0, step_count: 1 },
                ChainRecord { first_step: 1, step_count: 2 },
            ],
            blobs: vec![
                blob,
                BlobRecord {
                    raw_len: 16,
                    first_chunk: 3,
                    chunk_count: 2,
                    chunk_size: 8,
                    chain_id: 1,
                    dtype_hint: Dtype::F16,
                    ..blob
                },
            ],
            chunks: vec![
                chunk(64, 5),
                chunk(69, 8),
                chunk(77, 6),
                chunk(83, 20),
                chunk(103, 24),
            ],
            members: vec![
                member,
                MemberRecord {
                    name: name(5, 5),
                    file_len: 4,
                    first_extent: 2,
                    extent_count: 1,
                    first_tensor: 2,
                    tensor_count: 0,
                    ..member
                },
                MemberRecord {
                    name: name(10, 5),
                    file_len: 0,
                    first_extent: 3,
                    extent_count: 0,
                    first_tensor: 2,
                    tensor_count: 0,
                    ..member
                },
            ],
            extents: vec![
                extent,
                ExtentRecord {
                    virt_offset: 10,
                    virt_len: 16,
                    blob_id: 1,
                    view_id: VIEW_F16_TO_F32,
                    blob_offset: 8,
                },
                ExtentRecord { virt_len: 4, blob_id: 1, ..extent },
            ],
            tensors: vec![
                TensorRecord {
                    name: name(15, 2),
                    virt_offset: 0,
                    virt_len: 10,
                    shape_off: 0,
                    ndim: 1,
                    dtype: Dtype::U8,
                },
                TensorRecord {
                    name: name(17, 2),
                    virt_offset: 10,
                    virt_len: 16,
                    shape_off: 1,
                    ndim: 2,
                    dtype: Dtype::F32,
                },
            ],
            dims: vec![10, 2, 2],
            params: Vec::new(),
            required_features: 0,
        }
    }

    /// [`Fixture::valid`] plus a delta blob and a member over it. Blob 2 is `delta_xor(2), whole`,
    /// 6 raw bytes in chunks of 4 (2 chunks), over bytes `4..10` of blob 0, so its chunks pair with
    /// chunks 1 and 2 of blob 0. Its `PARM` entry is the whole heap. Member `d.bin` is all of it.
    /// The header has required feature bit 0.
    pub(super) fn with_delta() -> Self {
        let mut fixture = Self::valid();
        fixture.strings.extend_from_slice(b"d.bin");
        fixture.steps.push(RawStep { codec_id: STEP_DELTA_XOR, param0: 2, param1: 0 });
        fixture.steps.push(RawStep { codec_id: STEP_WHOLE, param0: 0, param1: 0 });
        fixture.chains.push(ChainRecord { first_step: 3, step_count: 2 });
        fixture.chunks.extend([chunk(127, 8), chunk(135, 6)]);
        fixture.blobs.push(BlobRecord {
            raw_len: 6,
            first_chunk: 5,
            chunk_count: 2,
            chunk_size: 4,
            chain_id: 2,
            kind: BLOB_KIND_DELTA,
            params_off: 0,
            params_len: DELTA_PARAMS_LEN,
            ..first_blob()
        });
        fixture.params = DeltaParams { base_blob: 0, base_offset: 4 }.encode().to_vec();
        fixture.extents.push(ExtentRecord {
            virt_offset: 0,
            virt_len: 6,
            blob_id: 2,
            view_id: 0,
            blob_offset: 0,
        });
        fixture.members.push(MemberRecord {
            name: name(19, 5),
            file_len: 6,
            first_extent: 3,
            extent_count: 1,
            first_tensor: 2,
            tensor_count: 0,
            ..first_member()
        });
        fixture.required_features = FEATURE_DELTA;
        fixture
    }

    /// Copies the arrays of a parsed index.
    pub(super) fn from_index(index: &Index) -> Self {
        Self {
            strings: index.strings().to_vec(),
            steps: index.steps().to_vec(),
            chains: index.chains().to_vec(),
            blobs: index.blobs().to_vec(),
            chunks: index.chunks().to_vec(),
            members: index.members().to_vec(),
            extents: index.extents().to_vec(),
            tensors: index.tensors().to_vec(),
            dims: index.dims().to_vec(),
            params: index.params().to_vec(),
            required_features: 0,
        }
    }

    /// Serializes through [`IndexBuilder`], record by record, with no check.
    pub(super) fn serialize(&self) -> Vec<u8> {
        let mut builder = IndexBuilder::new();
        builder.add_raw_string(&self.strings).expect("fits");
        for step in &self.steps {
            builder.add_raw_step(*step).expect("fits");
        }
        for chain in &self.chains {
            builder.add_raw_chain(*chain).expect("fits");
        }
        for blob in &self.blobs {
            builder.add_blob(*blob).expect("fits");
        }
        for (id, chunk) in self.chunks.iter().enumerate() {
            assert_eq!(builder.add_chunk(*chunk), id as u64);
        }
        for member in &self.members {
            builder.add_member(*member).expect("fits");
        }
        for (id, extent) in self.extents.iter().enumerate() {
            assert_eq!(builder.add_extent(*extent), id as u64);
        }
        for tensor in &self.tensors {
            builder.add_tensor(*tensor).expect("fits");
        }
        builder.add_shape(&self.dims).expect("fits");
        if !self.params.is_empty() {
            assert_eq!(builder.add_params(&self.params).expect("fits"), (0, len_u32(&self.params)));
        }
        builder.serialize()
    }

    /// The header [`Fixture::check`] validates against.
    pub(super) fn header(&self) -> Header {
        Header { required_features: self.required_features, ..Header::current() }
    }

    /// The footer [`Fixture::check`] validates against.
    pub(super) fn footer() -> Footer {
        Footer {
            index_offset: DATA_END,
            index_stored_len: STORED_LEN,
            index_raw_len: 1024,
            index_xxh3: 0,
            generation: 1,
        }
    }

    /// Serializes, parses and validates.
    pub(super) fn check(&self) -> Result<Index, Error> {
        let limits = Limits::default();
        let index = Index::parse(&self.serialize(), &limits)?;
        validate(&index, &self.header(), &Self::footer(), FILE_LEN, &limits)?;
        Ok(index)
    }
}

/// The length of a test heap as a `u32`.
fn len_u32(bytes: &[u8]) -> u32 {
    u32::try_from(bytes.len()).expect("a small heap")
}
