//! File: profile/safetensors.rs
//! Purpose: reads the header of a safetensors file into a plan.
//!
//! Main responsibilities: finds each tensor's bytes from the JSON header and keeps the dtype the
//! file declares. The `__metadata__` block is skipped.
//!
//! Key functions: [`matches`], [`parse`].
//!
//! Implementation notes: the format has no magic, so [`matches`] looks for a plausible length word
//! followed by `{`. A tag is reported as written even when the bytes are something else, such as
//! packed NF4 weights and JSON blobs tagged `U8`. Sizes are not cross-checked against shapes here:
//! a wrong answer costs ratio only, and `normalize` handles ranges that leave the file.

use super::json::Json;
use super::plan::MAX_TENSOR_DIMS;
use super::reader::Reader;
use super::{MAX_NAME_BYTES, MAX_TENSORS, ParseResult, Stop};
use crate::types::{Dtype, Profile, RangePlan, TensorInfo, TypedRange};

/// Bytes of the length word in front of the header.
const LENGTH_BYTES: u64 = 8;

/// Largest header accepted, in bytes. The cap of the reference safetensors implementation.
const MAX_HEADER_BYTES: u64 = 100_000_000;

/// Whether a file of `file_len` bytes starting with `head` looks like safetensors.
pub(super) fn matches(head: &[u8], file_len: u64) -> bool {
    let Some(([word @ .., brace], _)) = head.split_at_checked(9) else {
        return false;
    };
    let Ok(word) = <[u8; 8]>::try_from(word) else {
        return false;
    };
    let room = file_len.saturating_sub(LENGTH_BYTES);
    *brace == b'{' && u64::from_le_bytes(word) <= room
}

/// Parses the header at the start of the source. Offsets in the plan are absolute.
pub(super) fn parse(reader: &mut Reader<'_>) -> ParseResult<RangePlan> {
    let header_len = reader.u64()?;
    let header = reader.take(header_len, MAX_HEADER_BYTES)?;
    let text = std::str::from_utf8(&header)
        .map_err(|_| Stop::Malformed("safetensors header is not UTF-8"))?;
    let data_start = reader.pos();

    let mut plan =
        RangePlan { profile: Profile::Safetensors, ranges: Vec::new(), tensors: Vec::new() };
    let mut json = Json::new(text);
    json.expect(b'{')?;
    if !json.eat(b'}') {
        loop {
            let name = json.string()?;
            json.expect(b':')?;
            if name == "__metadata__" {
                json.skip_value()?;
            } else {
                add_tensor(&mut plan, name, &mut json, data_start)?;
            }
            if json.eat(b'}') {
                break;
            }
            json.expect(b',')?;
        }
    }
    json.finish()?;
    Ok(plan)
}

/// Reads one tensor object and appends it to `plan`.
fn add_tensor(
    plan: &mut RangePlan,
    name: String,
    json: &mut Json<'_>,
    data_start: u64,
) -> ParseResult<()> {
    if plan.tensors.len() >= MAX_TENSORS {
        return Err(Stop::Malformed("safetensors header has too many tensors"));
    }
    if name.len() > MAX_NAME_BYTES {
        return Err(Stop::Malformed("safetensors tensor name is too long"));
    }
    let entry = read_entry(json)?;
    let (begin, end) = entry.offsets.ok_or(Stop::Malformed("tensor has no data_offsets"))?;
    let len = end.checked_sub(begin).ok_or(Stop::Malformed("tensor offsets are reversed"))?;
    let offset =
        data_start.checked_add(begin).ok_or(Stop::Malformed("tensor offset overflows"))?;
    let dtype = entry.dtype.ok_or(Stop::Malformed("tensor has no dtype"))?;
    let shape = entry.shape.ok_or(Stop::Malformed("tensor has no shape"))?;
    plan.ranges.push(TypedRange { offset, len, hint: dtype });
    plan.tensors.push(TensorInfo { name, offset, len, shape, dtype });
    Ok(())
}

/// The fields of one tensor object. Each is `None` until seen; other keys are skipped.
struct Entry {
    dtype: Option<Dtype>,
    shape: Option<Vec<u64>>,
    offsets: Option<(u64, u64)>,
}

/// Reads a tensor object.
fn read_entry(json: &mut Json<'_>) -> ParseResult<Entry> {
    let mut entry = Entry { dtype: None, shape: None, offsets: None };
    json.expect(b'{')?;
    if json.eat(b'}') {
        return Ok(entry);
    }
    loop {
        let key = json.string()?;
        json.expect(b':')?;
        match key.as_str() {
            "dtype" => entry.dtype = Some(dtype_of(&json.string()?)),
            "shape" => entry.shape = Some(read_shape(json)?),
            "data_offsets" => entry.offsets = Some(read_offsets(json)?),
            _ => json.skip_value()?,
        }
        if json.eat(b'}') {
            return Ok(entry);
        }
        json.expect(b',')?;
    }
}

/// Reads a shape array of at most [`MAX_TENSOR_DIMS`] integers.
fn read_shape(json: &mut Json<'_>) -> ParseResult<Vec<u64>> {
    let mut shape = Vec::new();
    json.expect(b'[')?;
    if json.eat(b']') {
        return Ok(shape);
    }
    loop {
        if shape.len() == MAX_TENSOR_DIMS {
            return Err(Stop::Malformed("tensor has too many dimensions"));
        }
        shape.push(json.uint()?);
        if json.eat(b']') {
            return Ok(shape);
        }
        json.expect(b',')?;
    }
}

/// Reads a `[begin, end]` pair.
fn read_offsets(json: &mut Json<'_>) -> ParseResult<(u64, u64)> {
    json.expect(b'[')?;
    let begin = json.uint()?;
    json.expect(b',')?;
    let end = json.uint()?;
    json.expect(b']')?;
    Ok((begin, end))
}

/// Maps a dtype tag. A tag this build does not know means no dtype was determined, id 0.
fn dtype_of(tag: &str) -> Dtype {
    match tag {
        "BOOL" => Dtype::Bool,
        "U8" => Dtype::U8,
        "I8" => Dtype::I8,
        "U16" => Dtype::U16,
        "I16" => Dtype::I16,
        "U32" => Dtype::U32,
        "I32" => Dtype::I32,
        "U64" => Dtype::U64,
        "I64" => Dtype::I64,
        "F16" => Dtype::F16,
        "BF16" => Dtype::Bf16,
        "F32" => Dtype::F32,
        "F64" => Dtype::F64,
        "F8_E4M3" => Dtype::F8E4M3,
        "F8_E5M2" => Dtype::F8E5M2,
        _ => Dtype::from_wire(0),
    }
}

#[cfg(test)]
mod tests {
    use super::{dtype_of, matches};
    use crate::types::Dtype;

    fn head(word: u64, next: u8) -> Vec<u8> {
        let mut bytes = word.to_le_bytes().to_vec();
        bytes.push(next);
        bytes
    }

    #[test]
    fn detection_needs_a_fitting_length_and_a_brace() {
        assert!(matches(&head(2, b'{'), 10));
        assert!(!matches(&head(3, b'{'), 10));
        assert!(!matches(&head(u64::MAX, b'{'), 10));
        assert!(!matches(&head(2, b'['), 10));
        assert!(!matches(&[0; 8], 10));
        assert!(!matches(&[], 0));
    }

    #[test]
    fn unknown_tags_mean_no_dtype() {
        assert_eq!(dtype_of("F8_E4M3"), Dtype::F8E4M3);
        assert_eq!(dtype_of("C64"), Dtype::Unknown(0));
        assert_eq!(dtype_of(""), Dtype::Unknown(0));
    }
}
