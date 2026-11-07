//! File: format/records.rs
//! Purpose: the fixed-size index records of spec section 4.1, as typed values.
//!
//! Main responsibilities:
//! - one plain struct per record, and the byte layout of each as `decode` and `encode`;
//! - rejection of non-zero zero fields, which is all a record can check on its own.
//!
//! Key structures: [`StrRef`], [`ChainRecord`], [`BlobRecord`], [`ChunkRecord`], [`MemberRecord`],
//! [`ExtentRecord`], [`TensorRecord`], and [`DeltaParams`], the `PARM` entry of a delta blob. `CHST`
//! records are [`RawStep`], owned by `chain`.
//!
//! Implementation notes: these values are unvalidated: everything that relates fields to each other
//! or to other records is in `validate.rs`. A view id and a blob kind stay raw here so that
//! `validate` can reject an unknown one; a dtype or a profile is a hint with a total mapping, so
//! records carry it typed.

use std::cmp::Ordering;

use super::wire::{
    bytes_at, is_zero_at, malformed, put_bytes, put_u16, put_u32, put_u64, u16_at, u32_at, u64_at,
};
use crate::chain::{RawStep, View};
use crate::error::Error;
use crate::types::{Dtype, Profile};

/// Bytes of a `CHAN` record.
pub(super) const CHAN_SIZE: usize = 8;
/// Bytes of a `CHST` record.
pub(super) const CHST_SIZE: usize = 8;
/// Bytes of a `BLOB` record.
pub(super) const BLOB_SIZE: usize = 80;
/// Bytes of a `CHNK` record.
pub(super) const CHNK_SIZE: usize = 24;
/// Bytes of a `MEMB` record.
pub(super) const MEMB_SIZE: usize = 72;
/// Bytes of an `EXTN` record.
pub(super) const EXTN_SIZE: usize = 32;
/// Bytes of a `TENS` record.
pub(super) const TENS_SIZE: usize = 32;
/// Bytes of a `SHAP` record.
pub(super) const SHAP_SIZE: usize = 8;
/// Bytes of the `PARM` entry of a delta blob, its `params_len` (spec section 7a).
pub const DELTA_PARAMS_LEN: u32 = 16;
/// [`DELTA_PARAMS_LEN`] as an array length.
const DELTA_PARAMS_SIZE: usize = 16;
/// Alignment of a blob's entry in `PARM`, in bytes: `params_off` is a multiple of it.
pub const PARAMS_ALIGN: u64 = PARAMS_ALIGN_LEN as u64;
/// [`PARAMS_ALIGN`] as a length. The cast above is lossless: the value is 8.
pub(super) const PARAMS_ALIGN_LEN: usize = 8;

/// A string in `STRS`: `len` bytes at byte `off`. Unvalidated, resolve it with
/// [`super::Index::string`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StrRef {
    /// Start in the heap, in bytes.
    pub off: u32,
    /// Length in bytes.
    pub len: u32,
}

/// A `CHAN` record: a run of `CHST` records. Resolve with [`super::Index::chain`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainRecord {
    /// Index of the first step in `CHST`.
    pub first_step: u32,
    /// Number of steps.
    pub step_count: u16,
}

impl ChainRecord {
    /// Decodes a record.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] if a zero field is not zero.
    pub(super) fn decode(bytes: [u8; CHAN_SIZE]) -> Result<Self, Error> {
        if !is_zero_at::<6, 2, CHAN_SIZE>(&bytes) {
            return Err(malformed("CHAN", "zero field is not zero"));
        }
        Ok(Self { first_step: u32_at::<0, _>(&bytes), step_count: u16_at::<4, _>(&bytes) })
    }

    /// Encodes the record.
    #[must_use]
    pub(super) fn encode(self) -> [u8; CHAN_SIZE] {
        let mut bytes = [0; CHAN_SIZE];
        put_u32::<0, _>(&mut bytes, self.first_step);
        put_u16::<4, _>(&mut bytes, self.step_count);
        bytes
    }
}

/// Decodes a `CHST` record. Any bit pattern is representable; `validate` rejects unknown ids.
#[must_use]
pub(super) fn decode_step(bytes: [u8; CHST_SIZE]) -> RawStep {
    RawStep {
        codec_id: u16_at::<0, _>(&bytes),
        param0: u16_at::<2, _>(&bytes),
        param1: u32_at::<4, _>(&bytes),
    }
}

/// Encodes a `CHST` record.
#[must_use]
pub(super) fn encode_step(step: RawStep) -> [u8; CHST_SIZE] {
    let mut bytes = [0; CHST_SIZE];
    put_u16::<0, _>(&mut bytes, step.codec_id);
    put_u16::<2, _>(&mut bytes, step.param0);
    put_u32::<4, _>(&mut bytes, step.param1);
    bytes
}

/// A `BLOB` record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlobRecord {
    /// BLAKE3 of the canonical bytes; all zero if the writer did not compute it.
    pub blake3: [u8; 32],
    /// Canonical bytes in the blob.
    pub raw_len: u64,
    /// Index of the first chunk in `CHNK`.
    pub first_chunk: u64,
    /// Number of chunks.
    pub chunk_count: u32,
    /// Raw bytes per chunk, except the last.
    pub chunk_size: u32,
    /// Index into `CHAN`.
    pub chain_id: u32,
    /// Blob kind, kept raw so that `validate` can reject an unknown one.
    pub kind: u8,
    /// Element type hint.
    pub dtype_hint: Dtype,
    /// Offset into `PARM`; zero for kind 0, a multiple of [`PARAMS_ALIGN`] for kind 1.
    pub params_off: u64,
    /// Length in `PARM`; zero for kind 0, [`DELTA_PARAMS_LEN`] for kind 1.
    pub params_len: u32,
}

impl BlobRecord {
    /// Decodes a record.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] if `flags` or the zero field is not zero.
    pub(super) fn decode(bytes: &[u8; BLOB_SIZE]) -> Result<Self, Error> {
        if !is_zero_at::<61, 1, BLOB_SIZE>(bytes) || !is_zero_at::<76, 4, BLOB_SIZE>(bytes) {
            return Err(malformed("BLOB", "zero field is not zero"));
        }
        let [kind] = bytes_at::<60, 1, BLOB_SIZE>(bytes);
        Ok(Self {
            blake3: bytes_at::<0, 32, BLOB_SIZE>(bytes),
            raw_len: u64_at::<32, _>(bytes),
            first_chunk: u64_at::<40, _>(bytes),
            chunk_count: u32_at::<48, _>(bytes),
            chunk_size: u32_at::<52, _>(bytes),
            chain_id: u32_at::<56, _>(bytes),
            kind,
            dtype_hint: Dtype::from_wire(u16_at::<62, _>(bytes)),
            params_off: u64_at::<64, _>(bytes),
            params_len: u32_at::<72, _>(bytes),
        })
    }

    /// Encodes the record.
    #[must_use]
    pub(super) fn encode(&self) -> [u8; BLOB_SIZE] {
        let mut bytes = [0; BLOB_SIZE];
        put_bytes::<0, 32, _>(&mut bytes, self.blake3);
        put_u64::<32, _>(&mut bytes, self.raw_len);
        put_u64::<40, _>(&mut bytes, self.first_chunk);
        put_u32::<48, _>(&mut bytes, self.chunk_count);
        put_u32::<52, _>(&mut bytes, self.chunk_size);
        put_u32::<56, _>(&mut bytes, self.chain_id);
        put_bytes::<60, 1, _>(&mut bytes, [self.kind]);
        put_u16::<62, _>(&mut bytes, self.dtype_hint.to_wire());
        put_u64::<64, _>(&mut bytes, self.params_off);
        put_u32::<72, _>(&mut bytes, self.params_len);
        bytes
    }

    /// Raw length of chunk `chunk` of this blob, from spec section 6.1. `None` if `chunk` is not a
    /// chunk of the blob or the record is inconsistent, which [`super::validate`] rules out.
    #[must_use]
    pub fn chunk_raw_len(&self, chunk: u32) -> Option<u64> {
        let last = self.chunk_count.checked_sub(1)?;
        match chunk.cmp(&last) {
            Ordering::Less => Some(u64::from(self.chunk_size)),
            Ordering::Equal => {
                self.raw_len.checked_sub(u64::from(last).checked_mul(u64::from(self.chunk_size))?)
            }
            Ordering::Greater => None,
        }
    }
}

/// The `PARM` entry of a delta blob (spec section 7a): its base blob, and where in the base's
/// canonical bytes the delta blob's bytes start. Unvalidated until [`super::validate`] returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeltaParams {
    /// Id of the base blob: lower than the delta blob's, and of kind 0.
    pub base_blob: u32,
    /// Start in the base's canonical bytes, a multiple of the shared `chunk_size`.
    pub base_offset: u64,
}

impl DeltaParams {
    /// Decodes an entry: `u32 base_blob`, `u32` zero, `u64 base_offset`.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] for `PARM` if the zero field is not zero.
    pub fn decode(bytes: &[u8; DELTA_PARAMS_SIZE]) -> Result<Self, Error> {
        if !is_zero_at::<4, 4, DELTA_PARAMS_SIZE>(bytes) {
            return Err(malformed("PARM", "zero field of a delta entry is not zero"));
        }
        Ok(Self { base_blob: u32_at::<0, _>(bytes), base_offset: u64_at::<8, _>(bytes) })
    }

    /// Encodes the entry, for [`super::IndexBuilder::add_params`].
    #[must_use]
    pub fn encode(self) -> [u8; DELTA_PARAMS_SIZE] {
        let mut bytes = [0; DELTA_PARAMS_SIZE];
        put_u32::<0, _>(&mut bytes, self.base_blob);
        put_u64::<8, _>(&mut bytes, self.base_offset);
        bytes
    }

    /// The base chunk that chunk `chunk` of the delta blob is combined with, given their shared
    /// `chunk_size`: `base_offset / chunk_size + chunk`. The single definition of this mapping, for
    /// validation and the reader. `None` if `chunk_size` is 0 or the result overflows `u32`; after
    /// validation, `None` never happens for a chunk of the delta blob.
    #[must_use]
    pub fn base_chunk(self, chunk: u32, chunk_size: u32) -> Option<u32> {
        let first = self.base_offset.checked_div(u64::from(chunk_size))?;
        u32::try_from(first).ok()?.checked_add(chunk)
    }
}

/// A `CHNK` record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkRecord {
    /// File offset of the stored chunk.
    pub file_offset: u64,
    /// XXH3 of the chunk's canonical decoded bytes.
    pub xxh3_decoded: u64,
    /// Stored bytes, including the prologue.
    pub stored_len: u32,
}

impl ChunkRecord {
    /// Decodes a record.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] if the zero field is not zero.
    pub(super) fn decode(bytes: &[u8; CHNK_SIZE]) -> Result<Self, Error> {
        if !is_zero_at::<20, 4, CHNK_SIZE>(bytes) {
            return Err(malformed("CHNK", "zero field is not zero"));
        }
        Ok(Self {
            file_offset: u64_at::<0, _>(bytes),
            xxh3_decoded: u64_at::<8, _>(bytes),
            stored_len: u32_at::<16, _>(bytes),
        })
    }

    /// Encodes the record.
    #[must_use]
    pub(super) fn encode(&self) -> [u8; CHNK_SIZE] {
        let mut bytes = [0; CHNK_SIZE];
        put_u64::<0, _>(&mut bytes, self.file_offset);
        put_u64::<8, _>(&mut bytes, self.xxh3_decoded);
        put_u32::<16, _>(&mut bytes, self.stored_len);
        bytes
    }
}

/// A `MEMB` record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemberRecord {
    /// Path string, `/` separated. Only a restore uses it.
    pub name: StrRef,
    /// Member length in bytes.
    pub file_len: u64,
    /// BLAKE3 of the whole member content.
    pub blake3: [u8; 32],
    /// Index of the first extent in `EXTN`.
    pub first_extent: u64,
    /// Number of extents.
    pub extent_count: u32,
    /// Index of the first tensor in `TENS`.
    pub first_tensor: u32,
    /// Number of tensors.
    pub tensor_count: u32,
    /// Source format hint.
    pub profile: Profile,
}

impl MemberRecord {
    /// Decodes a record.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] if the zero field is not zero.
    pub(super) fn decode(bytes: &[u8; MEMB_SIZE]) -> Result<Self, Error> {
        if !is_zero_at::<69, 3, MEMB_SIZE>(bytes) {
            return Err(malformed("MEMB", "zero field is not zero"));
        }
        let [profile] = bytes_at::<68, 1, MEMB_SIZE>(bytes);
        Ok(Self {
            name: StrRef { off: u32_at::<0, _>(bytes), len: u32_at::<4, _>(bytes) },
            file_len: u64_at::<8, _>(bytes),
            blake3: bytes_at::<16, 32, MEMB_SIZE>(bytes),
            first_extent: u64_at::<48, _>(bytes),
            extent_count: u32_at::<56, _>(bytes),
            first_tensor: u32_at::<60, _>(bytes),
            tensor_count: u32_at::<64, _>(bytes),
            profile: Profile::from_wire(profile),
        })
    }

    /// Encodes the record.
    #[must_use]
    pub(super) fn encode(&self) -> [u8; MEMB_SIZE] {
        let mut bytes = [0; MEMB_SIZE];
        put_u32::<0, _>(&mut bytes, self.name.off);
        put_u32::<4, _>(&mut bytes, self.name.len);
        put_u64::<8, _>(&mut bytes, self.file_len);
        put_bytes::<16, 32, _>(&mut bytes, self.blake3);
        put_u64::<48, _>(&mut bytes, self.first_extent);
        put_u32::<56, _>(&mut bytes, self.extent_count);
        put_u32::<60, _>(&mut bytes, self.first_tensor);
        put_u32::<64, _>(&mut bytes, self.tensor_count);
        put_bytes::<68, 1, _>(&mut bytes, [self.profile.to_wire()]);
        bytes
    }
}

/// An `EXTN` record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtentRecord {
    /// Start in the member.
    pub virt_offset: u64,
    /// Length in the member, at least 1.
    pub virt_len: u64,
    /// Index into `BLOB`.
    pub blob_id: u32,
    /// `view` as stored; [`ExtentRecord::view`] converts it.
    pub view_id: u16,
    /// Start in the blob's canonical bytes.
    pub blob_offset: u64,
}

impl ExtentRecord {
    /// The view of this extent.
    ///
    /// # Errors
    /// Returns [`Error::UnknownCodec`] for an unknown view id.
    pub fn view(&self) -> Result<View, Error> {
        View::from_wire(self.view_id)
    }

    /// Decodes a record.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] if the zero field is not zero.
    pub(super) fn decode(bytes: &[u8; EXTN_SIZE]) -> Result<Self, Error> {
        if !is_zero_at::<22, 2, EXTN_SIZE>(bytes) {
            return Err(malformed("EXTN", "zero field is not zero"));
        }
        Ok(Self {
            virt_offset: u64_at::<0, _>(bytes),
            virt_len: u64_at::<8, _>(bytes),
            blob_id: u32_at::<16, _>(bytes),
            view_id: u16_at::<20, _>(bytes),
            blob_offset: u64_at::<24, _>(bytes),
        })
    }

    /// Encodes the record.
    #[must_use]
    pub(super) fn encode(&self) -> [u8; EXTN_SIZE] {
        let mut bytes = [0; EXTN_SIZE];
        put_u64::<0, _>(&mut bytes, self.virt_offset);
        put_u64::<8, _>(&mut bytes, self.virt_len);
        put_u32::<16, _>(&mut bytes, self.blob_id);
        put_u16::<20, _>(&mut bytes, self.view_id);
        put_u64::<24, _>(&mut bytes, self.blob_offset);
        bytes
    }
}

/// A `TENS` record. Metadata: restore never consults it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorRecord {
    /// Tensor name.
    pub name: StrRef,
    /// Start in the member.
    pub virt_offset: u64,
    /// Length in the member, may be 0.
    pub virt_len: u64,
    /// Index of the first dimension in `SHAP`.
    pub shape_off: u32,
    /// Number of dimensions.
    pub ndim: u16,
    /// Element type hint.
    pub dtype: Dtype,
}

impl TensorRecord {
    /// Decodes a record. Every bit pattern of a `TENS` record decodes.
    #[must_use]
    pub(super) fn decode(bytes: &[u8; TENS_SIZE]) -> Self {
        Self {
            name: StrRef { off: u32_at::<0, _>(bytes), len: u32_at::<4, _>(bytes) },
            virt_offset: u64_at::<8, _>(bytes),
            virt_len: u64_at::<16, _>(bytes),
            shape_off: u32_at::<24, _>(bytes),
            ndim: u16_at::<28, _>(bytes),
            dtype: Dtype::from_wire(u16_at::<30, _>(bytes)),
        }
    }

    /// Encodes the record.
    #[must_use]
    pub(super) fn encode(&self) -> [u8; TENS_SIZE] {
        let mut bytes = [0; TENS_SIZE];
        put_u32::<0, _>(&mut bytes, self.name.off);
        put_u32::<4, _>(&mut bytes, self.name.len);
        put_u64::<8, _>(&mut bytes, self.virt_offset);
        put_u64::<16, _>(&mut bytes, self.virt_len);
        put_u32::<24, _>(&mut bytes, self.shape_off);
        put_u16::<28, _>(&mut bytes, self.ndim);
        put_u16::<30, _>(&mut bytes, self.dtype.to_wire());
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden records, written out from the tables of spec section 4.1 field by field.
    fn golden_blob() -> ([u8; BLOB_SIZE], BlobRecord) {
        let mut bytes = [0xAB; BLOB_SIZE];
        bytes[32..80].copy_from_slice(&[
            0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, // raw_len
            0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, // first_chunk
            0x24, 0x23, 0x22, 0x21, // chunk_count
            0x34, 0x33, 0x32, 0x31, // chunk_size
            0x44, 0x43, 0x42, 0x41, // chain_id
            0x51, // kind
            0x00, // flags
            0x12, 0x00, // dtype_hint F32
            0x68, 0x67, 0x66, 0x65, 0x64, 0x63, 0x62, 0x61, // params_off
            0x74, 0x73, 0x72, 0x71, // params_len
            0, 0, 0, 0, // zero
        ]);
        let record = BlobRecord {
            blake3: [0xAB; 32],
            raw_len: 0x0102_0304_0506_0708,
            first_chunk: 0x1112_1314_1516_1718,
            chunk_count: 0x2122_2324,
            chunk_size: 0x3132_3334,
            chain_id: 0x4142_4344,
            kind: 0x51,
            dtype_hint: Dtype::F32,
            params_off: 0x6162_6364_6566_6768,
            params_len: 0x7172_7374,
        };
        (bytes, record)
    }

    fn golden_member() -> ([u8; MEMB_SIZE], MemberRecord) {
        let mut bytes = [0; MEMB_SIZE];
        bytes[..16].copy_from_slice(&[
            0x04, 0x03, 0x02, 0x01, // name_off
            0x14, 0x13, 0x12, 0x11, // name_len
            0x28, 0x27, 0x26, 0x25, 0x24, 0x23, 0x22, 0x21, // file_len
        ]);
        bytes[16..48].copy_from_slice(&[0xCD; 32]);
        bytes[48..72].copy_from_slice(&[
            0x38, 0x37, 0x36, 0x35, 0x34, 0x33, 0x32, 0x31, // first_extent
            0x44, 0x43, 0x42, 0x41, // extent_count
            0x54, 0x53, 0x52, 0x51, // first_tensor
            0x64, 0x63, 0x62, 0x61, // tensor_count
            0x02, // profile gguf
            0, 0, 0, // zero
        ]);
        let record = MemberRecord {
            name: StrRef { off: 0x0102_0304, len: 0x1112_1314 },
            file_len: 0x2122_2324_2526_2728,
            blake3: [0xCD; 32],
            first_extent: 0x3132_3334_3536_3738,
            extent_count: 0x4142_4344,
            first_tensor: 0x5152_5354,
            tensor_count: 0x6162_6364,
            profile: Profile::Gguf,
        };
        (bytes, record)
    }

    #[test]
    fn chan_golden() {
        let bytes = [0x04, 0x03, 0x02, 0x01, 0x02, 0x00, 0, 0];
        let record = ChainRecord { first_step: 0x0102_0304, step_count: 2 };
        assert_eq!(ChainRecord::decode(bytes).expect("valid"), record);
        assert_eq!(record.encode(), bytes);
    }

    #[test]
    fn chst_golden() {
        let bytes = [0x01, 0x01, 0x04, 0x00, 0x08, 0x07, 0x06, 0x05];
        let step = RawStep { codec_id: 0x0101, param0: 4, param1: 0x0506_0708 };
        assert_eq!(decode_step(bytes), step);
        assert_eq!(encode_step(step), bytes);
    }

    #[test]
    fn blob_golden() {
        let (bytes, record) = golden_blob();
        assert_eq!(BlobRecord::decode(&bytes).expect("valid"), record);
        assert_eq!(record.encode(), bytes);
    }

    #[test]
    fn chnk_golden() {
        let bytes = [
            0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, // file_offset
            0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, // xxh3_decoded
            0x24, 0x23, 0x22, 0x21, // stored_len
            0, 0, 0, 0, // zero
        ];
        let record = ChunkRecord {
            file_offset: 0x0102_0304_0506_0708,
            xxh3_decoded: 0x1112_1314_1516_1718,
            stored_len: 0x2122_2324,
        };
        assert_eq!(ChunkRecord::decode(&bytes).expect("valid"), record);
        assert_eq!(record.encode(), bytes);
    }

    #[test]
    fn memb_golden() {
        let (bytes, record) = golden_member();
        assert_eq!(MemberRecord::decode(&bytes).expect("valid"), record);
        assert_eq!(record.encode(), bytes);
    }

    #[test]
    fn extn_golden() {
        let bytes = [
            0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, // virt_offset
            0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, // virt_len
            0x24, 0x23, 0x22, 0x21, // blob_id
            0x01, 0x03, // view f16 to f32
            0, 0, // zero
            0x38, 0x37, 0x36, 0x35, 0x34, 0x33, 0x32, 0x31, // blob_offset
        ];
        let record = ExtentRecord {
            virt_offset: 0x0102_0304_0506_0708,
            virt_len: 0x1112_1314_1516_1718,
            blob_id: 0x2122_2324,
            view_id: 0x0301,
            blob_offset: 0x3132_3334_3536_3738,
        };
        assert_eq!(ExtentRecord::decode(&bytes).expect("valid"), record);
        assert_eq!(record.encode(), bytes);
        assert_eq!(record.view().expect("registered"), View::F16ToF32);
    }

    #[test]
    fn tens_golden() {
        let bytes = [
            0x04, 0x03, 0x02, 0x01, // name_off
            0x14, 0x13, 0x12, 0x11, // name_len
            0x28, 0x27, 0x26, 0x25, 0x24, 0x23, 0x22, 0x21, // virt_offset
            0x38, 0x37, 0x36, 0x35, 0x34, 0x33, 0x32, 0x31, // virt_len
            0x44, 0x43, 0x42, 0x41, // shape_off
            0x10, 0x00, // ndim
            0x30, 0x00, // dtype Q8_0
        ];
        let record = TensorRecord {
            name: StrRef { off: 0x0102_0304, len: 0x1112_1314 },
            virt_offset: 0x2122_2324_2526_2728,
            virt_len: 0x3132_3334_3536_3738,
            shape_off: 0x4142_4344,
            ndim: 16,
            dtype: Dtype::Q80,
        };
        assert_eq!(TensorRecord::decode(&bytes), record);
        assert_eq!(record.encode(), bytes);
    }

    #[test]
    fn unknown_hints_are_preserved() {
        let (mut bytes, _) = golden_blob();
        bytes[62..64].copy_from_slice(&[0x77, 0x77]);
        let record = BlobRecord::decode(&bytes).expect("an unknown dtype is legal");
        assert_eq!(record.dtype_hint, Dtype::Unknown(0x7777));
        assert_eq!(record.encode(), bytes);

        let (mut bytes, _) = golden_member();
        bytes[68] = 0xEE;
        let record = MemberRecord::decode(&bytes).expect("an unknown profile is legal");
        assert_eq!(record.profile, Profile::Unknown(0xEE));
        assert_eq!(record.encode(), bytes);
    }

    #[test]
    fn nonzero_zero_fields_are_rejected() {
        fn rejected<T: std::fmt::Debug>(result: Result<T, Error>, expected: &str) {
            let error = result.expect_err("non-zero zero field");
            assert!(
                matches!(&error, Error::MalformedIndex { section, .. } if *section == expected),
                "{error:?}"
            );
        }
        let (blob, _) = golden_blob();
        for at in [61, 76, 77, 78, 79] {
            let mut bytes = blob;
            bytes[at] = 1;
            rejected(BlobRecord::decode(&bytes), "BLOB");
        }
        let (member, _) = golden_member();
        for at in 69..72 {
            let mut bytes = member;
            bytes[at] = 1;
            rejected(MemberRecord::decode(&bytes), "MEMB");
        }
        rejected(ChainRecord::decode([0, 0, 0, 0, 1, 0, 1, 0]), "CHAN");
        rejected(ChainRecord::decode([0, 0, 0, 0, 1, 0, 0, 1]), "CHAN");
        for at in 20..24 {
            let mut bytes = [0; CHNK_SIZE];
            bytes[at] = 1;
            rejected(ChunkRecord::decode(&bytes), "CHNK");
        }
        for at in 22..24 {
            let mut bytes = [0; EXTN_SIZE];
            bytes[at] = 1;
            rejected(ExtentRecord::decode(&bytes), "EXTN");
        }
    }

    #[test]
    fn delta_params_golden() {
        let bytes = [
            0x04, 0x03, 0x02, 0x01, // base_blob
            0, 0, 0, 0, // zero
            0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, // base_offset
        ];
        let params = DeltaParams { base_blob: 0x0102_0304, base_offset: 0x1112_1314_1516_1718 };
        assert_eq!(DeltaParams::decode(&bytes).expect("valid"), params);
        assert_eq!(params.encode(), bytes);
        assert_eq!(usize::try_from(DELTA_PARAMS_LEN), Ok(bytes.len()));
        for at in 4..8 {
            let mut bytes = bytes;
            bytes[at] = 1;
            let error = DeltaParams::decode(&bytes).expect_err("non-zero zero field");
            assert!(matches!(error, Error::MalformedIndex { section: "PARM", .. }), "{error:?}");
        }
    }

    #[test]
    fn base_chunk_is_checked() {
        let params = DeltaParams { base_blob: 0, base_offset: 12 };
        assert_eq!(params.base_chunk(0, 4), Some(3));
        assert_eq!(params.base_chunk(2, 4), Some(5));
        assert_eq!(params.base_chunk(0, 0), None);
        assert_eq!(params.base_chunk(u32::MAX - 2, 4), None);
        let far = DeltaParams { base_blob: 0, base_offset: u64::MAX };
        assert_eq!(far.base_chunk(0, 1), None);
    }

    #[test]
    fn chunk_raw_len_follows_section_6_1() {
        let (_, golden) = golden_blob();
        let blob = BlobRecord { raw_len: 10, chunk_size: 4, chunk_count: 3, ..golden };
        assert_eq!(blob.chunk_raw_len(0), Some(4));
        assert_eq!(blob.chunk_raw_len(1), Some(4));
        assert_eq!(blob.chunk_raw_len(2), Some(2));
        assert_eq!(blob.chunk_raw_len(3), None);
        assert_eq!(blob.chunk_raw_len(u32::MAX), None);
    }

    #[test]
    fn chunk_raw_len_survives_inconsistent_records() {
        let (_, golden) = golden_blob();
        let empty = BlobRecord { raw_len: 0, chunk_count: 0, ..golden };
        assert_eq!(empty.chunk_raw_len(0), None);
        let short =
            BlobRecord { raw_len: 1, chunk_size: u32::MAX, chunk_count: u32::MAX, ..golden };
        assert_eq!(short.chunk_raw_len(u32::MAX - 1), None);
    }
}
