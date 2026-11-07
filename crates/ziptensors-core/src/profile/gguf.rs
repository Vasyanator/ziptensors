//! File: profile/gguf.rs
//! Purpose: reads the header of a GGUF file, versions 2 and 3, into a plan.
//!
//! Main responsibilities: walks the key-value section for `general.alignment`, reads the tensor
//! infos, and computes each tensor's byte length from its ggml type.
//!
//! Key functions: [`parse`].
//!
//! Implementation notes: padding between tensors stays in the skeleton, because a range ends where
//! the computed length ends. Block geometry comes from `Dtype::block_geometry`, which is `chain`;
//! this file owns only the ggml type numbers. Every ggml type of the registry gets a row. Only a
//! type the writer has a transform for gets a range: plain elements and the four types with a
//! field table. The bytes of the other block-quantized types, such as `Q4_0`, stay in the skeleton,
//! which is how they were stored before they had a dtype; a typed blob for them is a ratio decision
//! that needs a measurement. A ggml type with no [`Dtype`] gets neither a row nor a range. `shape`
//! keeps GGUF order, fastest dimension first, exactly as the file states it.

use super::plan::MAX_TENSOR_DIMS;
use super::reader::Reader;
use super::{MAX_NAME_BYTES, MAX_TENSORS, ParseResult, Stop};
use crate::types::{Dtype, Profile, RangePlan, TensorInfo, TypedRange};

/// First bytes of a GGUF file.
pub(super) const MAGIC: &[u8] = b"GGUF";

/// The header must end within this many bytes. Real ones reach a few MiB of tokenizer strings.
const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;

/// Most key-value pairs accepted.
const MAX_KV_PAIRS: u64 = 1 << 20;

/// Tensor data alignment when the file does not set one.
const DEFAULT_ALIGNMENT: u64 = 32;

/// Deepest nesting of array values followed.
const MAX_ARRAY_DEPTH: u32 = 4;

/// Key of the alignment value.
const ALIGNMENT_KEY: &[u8] = b"general.alignment";

/// GGUF value type of a `u32`.
const VALUE_U32: u32 = 4;
/// GGUF value type of a string.
const VALUE_STRING: u32 = 8;
/// GGUF value type of an array.
const VALUE_ARRAY: u32 = 9;

/// Parses the header at the start of the source. Offsets in the plan are absolute.
pub(super) fn parse(reader: &mut Reader<'_>) -> ParseResult<RangePlan> {
    reader.skip(4)?;
    let version = reader.u32()?;
    if !matches!(version, 2 | 3) {
        return Err(Stop::Malformed("unsupported GGUF version"));
    }
    let tensor_count = reader.u64()?;
    let kv_count = reader.u64()?;
    if tensor_count > u64::try_from(MAX_TENSORS).unwrap_or(u64::MAX) || kv_count > MAX_KV_PAIRS {
        return Err(Stop::Malformed("GGUF declares too many entries"));
    }
    let alignment = read_key_values(reader, kv_count)?;

    let mut rows = Vec::new();
    for _ in 0..tensor_count {
        check_header_bound(reader)?;
        if let Some(row) = read_tensor_info(reader)? {
            rows.push(row);
        }
    }
    let data_start = align_up(reader.pos(), alignment)?;

    let mut plan = RangePlan { profile: Profile::Gguf, ranges: Vec::new(), tensors: Vec::new() };
    for mut row in rows {
        row.offset = data_start
            .checked_add(row.offset)
            .ok_or(Stop::Malformed("GGUF tensor offset overflows"))?;
        if has_range(row.dtype) {
            plan.ranges.push(TypedRange { offset: row.offset, len: row.len, hint: row.dtype });
        }
        plan.tensors.push(row);
    }
    Ok(plan)
}

/// Stops once the walk has passed the header cap, whatever the declared counts promise.
fn check_header_bound(reader: &Reader<'_>) -> ParseResult<()> {
    if reader.pos() > MAX_HEADER_BYTES {
        return Err(Stop::Malformed("GGUF header is over the cap"));
    }
    Ok(())
}

/// Walks the key-value section. Returns the alignment, always a power of two.
fn read_key_values(reader: &mut Reader<'_>, count: u64) -> ParseResult<u64> {
    let mut alignment = DEFAULT_ALIGNMENT;
    for _ in 0..count {
        check_header_bound(reader)?;
        let key_len = reader.u64()?;
        let is_alignment = if usize::try_from(key_len).ok() == Some(ALIGNMENT_KEY.len()) {
            reader.take(key_len, key_len)? == ALIGNMENT_KEY
        } else {
            reader.skip(key_len)?;
            false
        };
        let value_type = reader.u32()?;
        if is_alignment {
            if value_type != VALUE_U32 {
                return Err(Stop::Malformed("GGUF alignment is not a u32"));
            }
            alignment = u64::from(reader.u32()?);
            if !alignment.is_power_of_two() {
                return Err(Stop::Malformed("GGUF alignment is not a power of two"));
            }
        } else {
            skip_value(reader, value_type, 0)?;
        }
    }
    Ok(alignment)
}

/// Skips one value of `value_type`.
fn skip_value(reader: &mut Reader<'_>, value_type: u32, depth: u32) -> ParseResult<()> {
    if let Some(size) = fixed_size(value_type) {
        return reader.skip(size);
    }
    match value_type {
        VALUE_STRING => {
            let len = reader.u64()?;
            reader.skip(len)
        }
        VALUE_ARRAY => skip_array(reader, depth),
        _ => Err(Stop::Malformed("unknown GGUF value type")),
    }
}

/// Skips an array value: element type, count, elements.
fn skip_array(reader: &mut Reader<'_>, depth: u32) -> ParseResult<()> {
    if depth >= MAX_ARRAY_DEPTH {
        return Err(Stop::Malformed("GGUF arrays nest too deeply"));
    }
    let element_type = reader.u32()?;
    let count = reader.u64()?;
    if let Some(size) = fixed_size(element_type) {
        let total =
            count.checked_mul(size).ok_or(Stop::Malformed("GGUF array size overflows"))?;
        return reader.skip(total);
    }
    // Every element of these types takes at least 8 bytes, so a false count hits the end of the
    // source or the header cap well before it is used up.
    for _ in 0..count {
        check_header_bound(reader)?;
        skip_value(reader, element_type, depth + 1)?;
    }
    Ok(())
}

/// Size in bytes of a fixed-size value type.
fn fixed_size(value_type: u32) -> Option<u64> {
    match value_type {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

/// Reads one tensor info into a row with an offset still relative to the data section. `None` for
/// a ggml type with no [`Dtype`].
fn read_tensor_info(reader: &mut Reader<'_>) -> ParseResult<Option<TensorInfo>> {
    let name_len = reader.u64()?;
    let name = reader.take(name_len, u64::try_from(MAX_NAME_BYTES).unwrap_or(u64::MAX))?;
    let name = String::from_utf8(name).map_err(|_| Stop::Malformed("GGUF name is not UTF-8"))?;
    let dim_count = reader.u32()?;
    if usize::try_from(dim_count).map_or(true, |dims| dims > MAX_TENSOR_DIMS) {
        return Err(Stop::Malformed("GGUF tensor has too many dimensions"));
    }
    let mut shape = Vec::new();
    for _ in 0..dim_count {
        shape.push(reader.u64()?);
    }
    let ggml_type = reader.u32()?;
    let offset = reader.u64()?;
    let Some(dtype) = ggml_type_of(ggml_type) else {
        tracing::debug!(ggml_type, "GGUF tensor of an unmapped type stays in the skeleton");
        return Ok(None);
    };
    let len = byte_len(dtype, &shape)?;
    Ok(Some(TensorInfo { name, offset, len, shape, dtype }))
}

/// Maps a ggml type id, `GGMLQuantizationType` of the reference `gguf` package, to a dtype.
/// `None` for an id with no dtype: ids ggml retired (4, 5, 31 to 33, 36 to 38) and later ones.
fn ggml_type_of(id: u32) -> Option<Dtype> {
    let dtype = match id {
        0 => Dtype::F32,
        1 => Dtype::F16,
        2 => Dtype::Q40,
        3 => Dtype::Q41,
        6 => Dtype::Q50,
        7 => Dtype::Q51,
        8 => Dtype::Q80,
        9 => Dtype::Q81,
        10 => Dtype::Q2K,
        11 => Dtype::Q3K,
        12 => Dtype::Q4K,
        13 => Dtype::Q5K,
        14 => Dtype::Q6K,
        15 => Dtype::Q8K,
        16 => Dtype::Iq2Xxs,
        17 => Dtype::Iq2Xs,
        18 => Dtype::Iq3Xxs,
        19 => Dtype::Iq1S,
        20 => Dtype::Iq4Nl,
        21 => Dtype::Iq3S,
        22 => Dtype::Iq2S,
        23 => Dtype::Iq4Xs,
        24 => Dtype::I8,
        25 => Dtype::I16,
        26 => Dtype::I32,
        27 => Dtype::I64,
        28 => Dtype::F64,
        29 => Dtype::Iq1M,
        30 => Dtype::Bf16,
        34 => Dtype::Tq10,
        35 => Dtype::Tq20,
        39 => Dtype::Mxfp4,
        _ => return None,
    };
    Some(dtype)
}

/// Whether a tensor of `dtype` becomes a range, and so a typed blob: a plain element type, or a
/// block type with a field table. See the file header for the others.
fn has_range(dtype: Dtype) -> bool {
    dtype.field_table().is_some() || dtype.block_geometry().is_some_and(|block| block.elements == 1)
}

/// Size in bytes of a tensor. ggml requires the first dimension to be whole blocks, as blocks are
/// laid out per row.
fn byte_len(dtype: Dtype, shape: &[u64]) -> ParseResult<u64> {
    let block = dtype.block_geometry().ok_or(Stop::Malformed("GGUF type has no block"))?;
    let (first, rest) = shape.split_first().map_or((1, &[][..]), |(f, r)| (*f, r));
    let elements = u64::from(block.elements);
    if first % elements != 0 {
        return Err(Stop::Malformed("GGUF tensor row is not a whole number of blocks"));
    }
    rest.iter()
        .try_fold((first / elements).checked_mul(u64::from(block.bytes)), |total, &dim| {
            total.map(|t| t.checked_mul(dim)).ok_or(())
        })
        .ok()
        .flatten()
        .ok_or(Stop::Malformed("GGUF tensor size overflows"))
}

/// Rounds `pos` up to `alignment`, a power of two.
fn align_up(pos: u64, alignment: u64) -> ParseResult<u64> {
    pos.checked_next_multiple_of(alignment).ok_or(Stop::Malformed("GGUF data offset overflows"))
}

#[cfg(test)]
mod tests {
    use super::{byte_len, ggml_type_of, has_range};
    use crate::types::Dtype;

    /// `(ggml type id, registry name)` of every type of `GGMLQuantizationType` in the reference
    /// `gguf` package 0.18.0 (`gguf/constants.py`). The names are those of the spec registry.
    const GGML_TYPES: [(u32, &str); 32] = [
        (0, "F32"),
        (1, "F16"),
        (2, "Q4_0"),
        (3, "Q4_1"),
        (6, "Q5_0"),
        (7, "Q5_1"),
        (8, "Q8_0"),
        (9, "Q8_1"),
        (10, "Q2_K"),
        (11, "Q3_K"),
        (12, "Q4_K"),
        (13, "Q5_K"),
        (14, "Q6_K"),
        (15, "Q8_K"),
        (16, "IQ2_XXS"),
        (17, "IQ2_XS"),
        (18, "IQ3_XXS"),
        (19, "IQ1_S"),
        (20, "IQ4_NL"),
        (21, "IQ3_S"),
        (22, "IQ2_S"),
        (23, "IQ4_XS"),
        (24, "I8"),
        (25, "I16"),
        (26, "I32"),
        (27, "I64"),
        (28, "F64"),
        (29, "IQ1_M"),
        (30, "BF16"),
        (34, "TQ1_0"),
        (35, "TQ2_0"),
        (39, "MXFP4"),
    ];

    #[test]
    fn every_ggml_type_maps_to_the_dtype_of_its_name() {
        for (id, name) in GGML_TYPES {
            assert_eq!(ggml_type_of(id).and_then(Dtype::name), Some(name), "{id}");
        }
    }

    #[test]
    fn block_sizes_come_from_the_dtype_geometry() {
        for (id, name) in GGML_TYPES {
            let dtype = ggml_type_of(id).expect("mapped");
            let block = dtype.block_geometry().expect("every ggml type has a geometry");
            let elements = u64::from(block.elements);
            let len = byte_len(dtype, &[elements * 3, 5]).ok();
            assert_eq!(len, Some(u64::from(block.bytes) * 15), "{name}");
        }
    }

    #[test]
    fn sizes_follow_rows_of_whole_blocks() {
        assert_eq!(byte_len(Dtype::F32, &[2, 3, 4]).ok(), Some(96));
        assert_eq!(byte_len(Dtype::Bf16, &[]).ok(), Some(2));
        assert_eq!(byte_len(Dtype::F16, &[0, 3]).ok(), Some(0));
        assert_eq!(byte_len(Dtype::Q40, &[64, 2]).ok(), Some(72));
        assert_eq!(byte_len(Dtype::Mxfp4, &[32]).ok(), Some(17));
        assert!(byte_len(Dtype::Q80, &[33]).is_err());
        assert!(byte_len(Dtype::Q80, &[]).is_err());
        assert!(byte_len(Dtype::Q2K, &[128, 2]).is_err());
        assert!(byte_len(Dtype::F64, &[u64::MAX]).is_err());
        assert!(byte_len(Dtype::F64, &[1 << 32, 1 << 32]).is_err());
        assert!(byte_len(Dtype::Iq2Xxs, &[256 << 32, 1 << 32]).is_err());
    }

    #[test]
    fn only_types_with_a_transform_get_a_range() {
        for (id, name) in GGML_TYPES {
            let dtype = ggml_type_of(id).expect("mapped");
            let expected = matches!(
                name,
                "F32" | "F16" | "BF16" | "F64" | "I8" | "I16" | "I32" | "I64" | "Q8_0" | "Q4_K"
                    | "Q5_K" | "Q6_K"
            );
            assert_eq!(has_range(dtype), expected, "{name}");
        }
    }

    #[test]
    fn ids_without_a_dtype_have_no_mapping() {
        for id in [4, 5, 31, 32, 33, 36, 37, 38, 40, u32::MAX] {
            assert_eq!(ggml_type_of(id), None, "{id}");
        }
    }
}
