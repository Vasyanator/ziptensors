//! File: format/header.rs
//! Purpose: the 64-byte file header, spec sections 3.1 and 3.3.
//!
//! Main responsibilities:
//! - [`parse_header`] applies the header half of invariants 1 and 9;
//! - [`Header::to_bytes`] writes the structure for the writer.
//!
//! Key structures: [`Header`].
//!
//! Implementation notes: the magic and the major version are checked before the hash, because a
//! different major may lay out the rest of the header differently, and such a file should report
//! `UnsupportedVersion` and not a hash failure.

use xxhash_rust::xxh3::xxh3_64;

use super::wire::{
    bytes_at, is_zero_at, malformed, put_bytes, put_u16, put_u32, put_u64, u16_at, u32_at, u64_at,
};
use crate::chain::SUPPORTED_REQUIRED_FEATURES;
use crate::error::Error;

/// Value of the `header_len` field.
const HEADER_LEN_FIELD: u32 = 64;
/// Size of the header in bytes. Also the first offset of the data region.
pub const HEADER_LEN: u64 = HEADER_LEN_FIELD as u64;
/// File magic.
pub const MAGIC: [u8; 8] = [0x89, 0x5A, 0x54, 0x53, 0x0D, 0x0A, 0x1A, 0x0A];
/// The major version this library reads and writes.
pub const FORMAT_MAJOR: u16 = 0;
/// The minor version this library writes. Version 0.2 registered more dtype ids (spec section 5.3),
/// version 0.3 delta blobs (spec section 7a).
pub const FORMAT_MINOR: u16 = 3;

/// A checked file header. Hash, `header_len` and reserved bytes are implied by a successful parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Major version.
    pub major: u16,
    /// Minor version. Every minor of the major is accepted.
    pub minor: u16,
    /// Feature bits a reader must implement.
    pub required_features: u64,
    /// Feature bits a reader may ignore.
    pub optional_features: u64,
}

impl Header {
    /// The header of a file written by this version, with no feature bits. A writer that seals a
    /// delta blob sets [`crate::chain::FEATURE_DELTA`] in `required_features` before the file
    /// becomes visible (spec section 7a).
    #[must_use]
    pub fn current() -> Self {
        Self {
            major: FORMAT_MAJOR,
            minor: FORMAT_MINOR,
            required_features: 0,
            optional_features: 0,
        }
    }

    /// Serializes the header, computing `header_xxh3`.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut bytes = [0; 64];
        put_bytes::<0, 8, _>(&mut bytes, MAGIC);
        put_u16::<8, _>(&mut bytes, self.major);
        put_u16::<10, _>(&mut bytes, self.minor);
        put_u32::<12, _>(&mut bytes, HEADER_LEN_FIELD);
        put_u64::<16, _>(&mut bytes, self.required_features);
        put_u64::<24, _>(&mut bytes, self.optional_features);
        let hash = xxh3_64(&bytes_at::<0, 56, 64>(&bytes));
        put_u64::<56, _>(&mut bytes, hash);
        bytes
    }
}

/// Parses and checks a file header.
///
/// # Errors
/// - [`Error::NotZiptensors`] if the magic is wrong.
/// - [`Error::UnsupportedVersion`] if `major` is not [`FORMAT_MAJOR`].
/// - [`Error::MalformedIndex`] for section `"header"` on a bad hash, a `header_len` other than 64
///   or a non-zero reserved byte.
/// - [`Error::UnsupportedFeature`] if `required_features` has a bit this library does not
///   implement. Unknown bits of `optional_features` are ignored. Whether an implemented bit agrees
///   with the index is checked by [`super::validate`].
pub fn parse_header(bytes: &[u8; 64]) -> Result<Header, Error> {
    if bytes_at::<0, 8, 64>(bytes) != MAGIC {
        return Err(Error::NotZiptensors);
    }
    let major = u16_at::<8, _>(bytes);
    let minor = u16_at::<10, _>(bytes);
    if major != FORMAT_MAJOR {
        return Err(Error::UnsupportedVersion { major, minor });
    }
    let stored_hash = u64_at::<56, _>(bytes);
    let actual_hash = xxh3_64(&bytes_at::<0, 56, 64>(bytes));
    if stored_hash != actual_hash {
        return Err(malformed(
            "header",
            format!("hash is {stored_hash:#018x}, content hashes to {actual_hash:#018x}"),
        ));
    }
    let header_len = u32_at::<12, _>(bytes);
    if header_len != HEADER_LEN_FIELD {
        return Err(malformed("header", format!("header_len is {header_len}, must be 64")));
    }
    if !is_zero_at::<32, 24, 64>(bytes) {
        return Err(malformed("header", "reserved bytes are not zero"));
    }
    let required_features = u64_at::<16, _>(bytes);
    let unsupported = required_features & !SUPPORTED_REQUIRED_FEATURES;
    if unsupported != 0 {
        return Err(Error::UnsupportedFeature { bits: unsupported });
    }
    Ok(Header { major, minor, required_features, optional_features: u64_at::<24, _>(bytes) })
}

#[cfg(test)]
mod tests {
    use super::{Header, parse_header};
    use crate::chain::{FEATURE_DELTA, FEATURE_RESERVED_DERIVED, FEATURE_RESERVED_MULTI_VOLUME};
    use crate::error::Error;

    /// The header of a version 0.3 file, written out from the table of spec section 3.1. The hash
    /// was computed with `twox-hash`, an implementation independent of the one in this crate.
    const GOLDEN: [u8; 64] = [
        0x89, 0x5A, 0x54, 0x53, 0x0D, 0x0A, 0x1A, 0x0A, // magic
        0x00, 0x00, // major
        0x03, 0x00, // minor
        0x40, 0x00, 0x00, 0x00, // header_len
        0, 0, 0, 0, 0, 0, 0, 0, // required_features
        0, 0, 0, 0, 0, 0, 0, 0, // optional_features
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // reserved
        0xF2, 0xC1, 0x01, 0xCF, 0xFE, 0x0B, 0x55, 0x92, // header_xxh3
    ];

    /// The header of a version 0.3 file with delta blobs: required feature bit 0. Hash from
    /// `twox-hash` too.
    const GOLDEN_DELTA: [u8; 64] = [
        0x89, 0x5A, 0x54, 0x53, 0x0D, 0x0A, 0x1A, 0x0A, // magic
        0x00, 0x00, // major
        0x03, 0x00, // minor
        0x40, 0x00, 0x00, 0x00, // header_len
        1, 0, 0, 0, 0, 0, 0, 0, // required_features
        0, 0, 0, 0, 0, 0, 0, 0, // optional_features
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // reserved
        0x2D, 0xD2, 0x67, 0xEC, 0x15, 0xC1, 0xDC, 0x68, // header_xxh3
    ];

    /// The header of a version 0.2 file. Hash from `twox-hash` too.
    const GOLDEN_0_2: [u8; 64] = [
        0x89, 0x5A, 0x54, 0x53, 0x0D, 0x0A, 0x1A, 0x0A, // magic
        0x00, 0x00, // major
        0x02, 0x00, // minor
        0x40, 0x00, 0x00, 0x00, // header_len
        0, 0, 0, 0, 0, 0, 0, 0, // required_features
        0, 0, 0, 0, 0, 0, 0, 0, // optional_features
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // reserved
        0x27, 0xCE, 0xFF, 0x9A, 0x8F, 0x0B, 0x37, 0x21, // header_xxh3
    ];

    /// The header of a version 0.1 file. It differs only in `minor` and the hash, computed with
    /// `twox-hash` too.
    const GOLDEN_0_1: [u8; 64] = [
        0x89, 0x5A, 0x54, 0x53, 0x0D, 0x0A, 0x1A, 0x0A, // magic
        0x00, 0x00, // major
        0x01, 0x00, // minor
        0x40, 0x00, 0x00, 0x00, // header_len
        0, 0, 0, 0, 0, 0, 0, 0, // required_features
        0, 0, 0, 0, 0, 0, 0, 0, // optional_features
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // reserved
        0xA7, 0x32, 0xAF, 0x19, 0x11, 0x7E, 0x7C, 0x12, // header_xxh3
    ];

    /// Overwrites `bytes[at..]` with `patch` and recomputes the hash, so that the check under test
    /// is reached.
    fn patched(at: usize, patch: &[u8]) -> [u8; 64] {
        let mut bytes = GOLDEN;
        bytes[at..at + patch.len()].copy_from_slice(patch);
        let hash = xxhash_rust::xxh3::xxh3_64(&bytes[..56]);
        bytes[56..].copy_from_slice(&hash.to_le_bytes());
        bytes
    }

    fn malformed_header(result: &Result<Header, Error>) -> bool {
        matches!(result, Err(Error::MalformedIndex { section: "header", .. }))
    }

    #[test]
    fn writer_produces_the_golden_bytes() {
        assert_eq!(Header::current().to_bytes(), GOLDEN);
    }

    #[test]
    fn golden_bytes_parse() {
        assert_eq!(parse_header(&GOLDEN).expect("valid"), Header::current());
    }

    #[test]
    fn earlier_version_headers_still_parse() {
        let header = parse_header(&GOLDEN_0_1).expect("valid");
        assert_eq!(header, Header { minor: 1, ..Header::current() });
        let header = parse_header(&GOLDEN_0_2).expect("valid");
        assert_eq!(header, Header { minor: 2, ..Header::current() });
    }

    #[test]
    fn the_delta_feature_bit_is_implemented() {
        let header = Header { required_features: FEATURE_DELTA, ..Header::current() };
        assert_eq!(header.to_bytes(), GOLDEN_DELTA);
        assert_eq!(parse_header(&GOLDEN_DELTA).expect("valid"), header);
    }

    #[test]
    fn header_with_optional_features_round_trips() {
        let header = Header { optional_features: u64::MAX, ..Header::current() };
        assert_eq!(parse_header(&header.to_bytes()).expect("valid"), header);
    }

    #[test]
    fn wrong_magic_is_not_ziptensors() {
        assert!(matches!(parse_header(&patched(0, &[0])), Err(Error::NotZiptensors)));
        assert!(matches!(parse_header(&[0; 64]), Err(Error::NotZiptensors)));
        assert!(matches!(parse_header(&[0xFF; 64]), Err(Error::NotZiptensors)));
    }

    #[test]
    fn other_major_is_rejected_whatever_follows() {
        let mut bytes = patched(8, &[1, 0, 7, 0]);
        bytes[56] ^= 1;
        assert!(matches!(
            parse_header(&bytes),
            Err(Error::UnsupportedVersion { major: 1, minor: 7 })
        ));
    }

    #[test]
    fn any_minor_of_the_major_is_accepted() {
        let header = parse_header(&patched(10, &[0xFF, 0xFF])).expect("valid");
        assert_eq!(header.minor, u16::MAX);
    }

    #[test]
    fn every_byte_is_covered_by_the_hash_or_a_prior_check() {
        for at in 0..64 {
            let mut bytes = GOLDEN;
            bytes[at] ^= 0x40;
            assert!(parse_header(&bytes).is_err(), "byte {at}");
        }
    }

    #[test]
    fn other_header_len_is_rejected() {
        assert!(malformed_header(&parse_header(&patched(12, &[63]))));
        assert!(malformed_header(&parse_header(&patched(12, &[0xFF, 0xFF, 0xFF, 0xFF]))));
    }

    #[test]
    fn nonzero_reserved_bytes_are_rejected() {
        for at in 32..56 {
            assert!(malformed_header(&parse_header(&patched(at, &[1]))), "byte {at}");
        }
    }

    #[test]
    fn unknown_required_features_are_rejected_and_named() {
        let reserved = FEATURE_RESERVED_DERIVED | FEATURE_RESERVED_MULTI_VOLUME;
        for bit in [reserved, 1 << 63, !FEATURE_DELTA] {
            let result = parse_header(&patched(16, &bit.to_le_bytes()));
            assert!(matches!(result, Err(Error::UnsupportedFeature { bits }) if bits == bit));
        }
        // An implemented bit next to an unknown one: only the unknown one is named.
        let result = parse_header(&patched(16, &(reserved | FEATURE_DELTA).to_le_bytes()));
        assert!(matches!(result, Err(Error::UnsupportedFeature { bits }) if bits == reserved));
    }

    #[test]
    fn unknown_optional_features_are_ignored() {
        let header = parse_header(&patched(24, &[0xFF; 8])).expect("valid");
        assert_eq!(header.optional_features, u64::MAX);
    }
}
