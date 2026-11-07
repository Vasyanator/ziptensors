//! File: format/wire.rs
//! Purpose: little-endian field access on fixed-size records and the shared error constructor of
//! the format layer.
//!
//! Main responsibilities:
//! - read and write `u16`, `u32`, `u64` and byte fields of a `[u8; R]` at a constant offset;
//! - build `Error::MalformedIndex` values.
//!
//! Implementation notes: offsets are const generic parameters, and an inline `const` assertion
//! makes an out-of-range offset a compile error. So the slicing below cannot panic on any input:
//! no offset here ever comes from a file.

use crate::error::Error;

/// Builds [`Error::MalformedIndex`]. `section` is a tag or `"header"`, `"footer"`, `"index"`.
pub(super) fn malformed(section: &'static str, detail: impl Into<String>) -> Error {
    Error::MalformedIndex { section, detail: detail.into() }
}

/// Copies `N` bytes at offset `AT` of a record.
#[must_use]
pub(super) fn bytes_at<const AT: usize, const N: usize, const R: usize>(
    record: &[u8; R],
) -> [u8; N] {
    const { assert!(AT + N <= R, "field outside its record") };
    let mut out = [0; N];
    out.copy_from_slice(&record[AT..AT + N]);
    out
}

/// Reads the `u16` at offset `AT`.
#[must_use]
pub(super) fn u16_at<const AT: usize, const R: usize>(record: &[u8; R]) -> u16 {
    u16::from_le_bytes(bytes_at::<AT, 2, R>(record))
}

/// Reads the `u32` at offset `AT`.
#[must_use]
pub(super) fn u32_at<const AT: usize, const R: usize>(record: &[u8; R]) -> u32 {
    u32::from_le_bytes(bytes_at::<AT, 4, R>(record))
}

/// Reads the `u64` at offset `AT`.
#[must_use]
pub(super) fn u64_at<const AT: usize, const R: usize>(record: &[u8; R]) -> u64 {
    u64::from_le_bytes(bytes_at::<AT, 8, R>(record))
}

/// True if the `N` bytes at offset `AT` are all zero.
#[must_use]
pub(super) fn is_zero_at<const AT: usize, const N: usize, const R: usize>(
    record: &[u8; R],
) -> bool {
    bytes_at::<AT, N, R>(record) == [0; N]
}

/// Writes `value` at offset `AT`.
pub(super) fn put_bytes<const AT: usize, const N: usize, const R: usize>(
    record: &mut [u8; R],
    value: [u8; N],
) {
    const { assert!(AT + N <= R, "field outside its record") };
    record[AT..AT + N].copy_from_slice(&value);
}

/// Writes a little-endian `u16` at offset `AT`.
pub(super) fn put_u16<const AT: usize, const R: usize>(record: &mut [u8; R], value: u16) {
    put_bytes::<AT, 2, R>(record, value.to_le_bytes());
}

/// Writes a little-endian `u32` at offset `AT`.
pub(super) fn put_u32<const AT: usize, const R: usize>(record: &mut [u8; R], value: u32) {
    put_bytes::<AT, 4, R>(record, value.to_le_bytes());
}

/// Writes a little-endian `u64` at offset `AT`.
pub(super) fn put_u64<const AT: usize, const R: usize>(record: &mut [u8; R], value: u64) {
    put_bytes::<AT, 8, R>(record, value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::{bytes_at, is_zero_at, put_u16, put_u32, put_u64, u16_at, u32_at, u64_at};

    #[test]
    fn fields_are_little_endian_at_their_offsets() {
        let record = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(u16_at::<1, _>(&record), 0x0302);
        assert_eq!(u32_at::<2, _>(&record), 0x0605_0403);
        assert_eq!(u64_at::<2, _>(&record), 0x0A09_0807_0605_0403);
        assert_eq!(bytes_at::<8, 2, _>(&record), [9, 10]);
    }

    #[test]
    fn writes_mirror_reads() {
        let mut record = [0; 14];
        put_u16::<0, _>(&mut record, 0x0201);
        put_u32::<2, _>(&mut record, 0x0605_0403);
        put_u64::<6, _>(&mut record, 0x0E0D_0C0B_0A09_0807);
        assert_eq!(record, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]);
    }

    #[test]
    fn zero_test_looks_only_at_its_field() {
        let record = [1, 0, 0, 1];
        assert!(is_zero_at::<1, 2, _>(&record));
        assert!(!is_zero_at::<0, 2, _>(&record));
        assert!(!is_zero_at::<2, 2, _>(&record));
    }
}
