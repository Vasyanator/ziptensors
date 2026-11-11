//! File: format/footer.rs
//! Purpose: the 64-byte file footer, spec section 3.2.
//!
//! Main responsibilities:
//! - [`parse_footer`] checks what the 64 bytes alone can prove;
//! - [`Footer::check_layout`] checks them against the file length and the index cap, before the
//!   stored index is read;
//! - [`Footer::check_index`] checks a decompressed index against the footer;
//! - [`Footer::to_bytes`] writes the structure for the writer.
//!
//! Key structures: [`Footer`].

use xxhash_rust::xxh3::xxh3_64;

use super::header::HEADER_LEN;
use super::limits::Limits;
use super::wire::{bytes_at, is_zero_at, malformed, put_bytes, put_u64, u64_at};
use crate::error::Error;

/// Size of the footer in bytes.
pub const FOOTER_LEN: u64 = 64;
/// Trailing magic: `ZTS-END` and one zero byte.
pub const END_MAGIC: [u8; 8] = *b"ZTS-END\0";
/// The `generation` of a freshly written file. An edit writes the previous footer's value plus one.
pub const GENERATION_INITIAL: u64 = 1;

/// A file footer. A successful [`parse_footer`] implies hash, `end_magic` and reserved bytes hold;
/// the fields themselves are unchecked until [`Footer::check_layout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Footer {
    /// File offset of the stored index.
    pub index_offset: u64,
    /// Bytes of the stored index.
    pub index_stored_len: u64,
    /// Bytes of the decompressed index.
    pub index_raw_len: u64,
    /// XXH3 of the decompressed index.
    pub index_xxh3: u64,
    /// Informational; no value is rejected.
    pub generation: u64,
}

impl Footer {
    /// The footer for a decompressed index `raw_index` stored at `index_offset` in
    /// `index_stored_len` bytes. `generation` is [`GENERATION_INITIAL`] for a fresh file and the
    /// previous footer's value plus one for an edit; no value is invalid.
    #[must_use]
    pub fn for_index(
        index_offset: u64,
        index_stored_len: u64,
        raw_index: &[u8],
        generation: u64,
    ) -> Self {
        Self {
            index_offset,
            index_stored_len,
            // Lossless: `usize` is at most 64 bits on every supported target.
            index_raw_len: raw_index.len() as u64,
            index_xxh3: xxh3_64(raw_index),
            generation,
        }
    }

    /// Serializes the footer, computing `footer_xxh3`.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut bytes = [0; 64];
        put_u64::<0, _>(&mut bytes, self.index_offset);
        put_u64::<8, _>(&mut bytes, self.index_stored_len);
        put_u64::<16, _>(&mut bytes, self.index_raw_len);
        put_u64::<24, _>(&mut bytes, self.index_xxh3);
        put_u64::<32, _>(&mut bytes, self.generation);
        let hash = xxh3_64(&bytes_at::<0, 48, 64>(&bytes));
        put_u64::<48, _>(&mut bytes, hash);
        put_bytes::<56, 8, _>(&mut bytes, END_MAGIC);
        bytes
    }

    /// Checks the footer against the file: the cap on `index_raw_len` and the layout half of
    /// invariant 1. Call before reading or decompressing the index. `file_len` is the real length
    /// of the file, in bytes.
    ///
    /// # Errors
    /// - [`Error::BudgetExceeded`] if `index_raw_len` is over [`Limits::index_cap`].
    /// - [`Error::MalformedIndex`] for section `"footer"` if `index_offset` is below 64 or
    ///   `index_offset + index_stored_len + 64` overflows or differs from `file_len`.
    pub fn check_layout(&self, file_len: u64, limits: &Limits) -> Result<(), Error> {
        let cap = limits.index_cap();
        if self.index_raw_len > cap {
            return Err(Error::BudgetExceeded {
                what: "index",
                declared: self.index_raw_len,
                limit: cap,
            });
        }
        if self.index_offset < HEADER_LEN {
            return Err(malformed(
                "footer",
                format!("index_offset {} is inside the header", self.index_offset),
            ));
        }
        let end = self
            .index_offset
            .checked_add(self.index_stored_len)
            .and_then(|end| end.checked_add(FOOTER_LEN));
        if end != Some(file_len) {
            return Err(malformed(
                "footer",
                format!(
                    "index at {} of {} bytes does not end at the footer of a {file_len} byte file",
                    self.index_offset, self.index_stored_len
                ),
            ));
        }
        Ok(())
    }

    /// Checks a decompressed index against `index_raw_len` and `index_xxh3`: the rest of invariant
    /// 1 and the first sentence of invariant 2.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] for section `"index"` on a length or hash mismatch.
    pub fn check_index(&self, raw_index: &[u8]) -> Result<(), Error> {
        if u64::try_from(raw_index.len()).ok() != Some(self.index_raw_len) {
            return Err(malformed(
                "index",
                format!(
                    "decompressed to {} bytes, the footer declares {}",
                    raw_index.len(),
                    self.index_raw_len
                ),
            ));
        }
        let actual = xxh3_64(raw_index);
        if actual != self.index_xxh3 {
            return Err(malformed(
                "index",
                format!("hashes to {actual:#018x}, the footer declares {:#018x}", self.index_xxh3),
            ));
        }
        Ok(())
    }
}

/// Parses a file footer, the last 64 bytes of the file.
///
/// # Errors
/// Returns [`Error::MalformedIndex`] for section `"footer"` on a wrong `end_magic`, a bad hash or a
/// non-zero reserved byte.
pub fn parse_footer(bytes: &[u8; 64]) -> Result<Footer, Error> {
    if bytes_at::<56, 8, 64>(bytes) != END_MAGIC {
        return Err(malformed("footer", "end magic is wrong"));
    }
    let stored_hash = u64_at::<48, _>(bytes);
    let actual_hash = xxh3_64(&bytes_at::<0, 48, 64>(bytes));
    if stored_hash != actual_hash {
        return Err(malformed(
            "footer",
            format!("hash is {stored_hash:#018x}, content hashes to {actual_hash:#018x}"),
        ));
    }
    if !is_zero_at::<40, 8, 64>(bytes) {
        return Err(malformed("footer", "reserved bytes are not zero"));
    }
    Ok(Footer {
        index_offset: u64_at::<0, _>(bytes),
        index_stored_len: u64_at::<8, _>(bytes),
        index_raw_len: u64_at::<16, _>(bytes),
        index_xxh3: u64_at::<24, _>(bytes),
        generation: u64_at::<32, _>(bytes),
    })
}

#[cfg(test)]
mod tests {
    use super::{Footer, GENERATION_INITIAL, parse_footer};
    use crate::error::Error;
    use crate::format::Limits;
    use crate::format::limits::MAX_INDEX_RAW_LEN;

    /// A footer written out from the table of spec section 3.2. The hash was computed with
    /// `twox-hash`, an implementation independent of the one in this crate.
    const GOLDEN: [u8; 64] = [
        0x00, 0x10, 0, 0, 0, 0, 0, 0, // index_offset 0x1000
        0x00, 0x02, 0, 0, 0, 0, 0, 0, // index_stored_len 0x200
        0x00, 0x08, 0, 0, 0, 0, 0, 0, // index_raw_len 0x800
        0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, // index_xxh3
        1, 0, 0, 0, 0, 0, 0, 0, // generation
        0, 0, 0, 0, 0, 0, 0, 0, // reserved
        0x91, 0xC8, 0x4B, 0xBD, 0x47, 0xCD, 0x02, 0xB1, // footer_xxh3
        b'Z', b'T', b'S', b'-', b'E', b'N', b'D', 0, // end_magic
    ];

    fn golden() -> Footer {
        Footer {
            index_offset: 0x1000,
            index_stored_len: 0x200,
            index_raw_len: 0x800,
            index_xxh3: 0x1122_3344_5566_7788,
            generation: 1,
        }
    }

    /// Overwrites `bytes[at..]` with `patch` and recomputes the hash.
    fn patched(at: usize, patch: &[u8]) -> [u8; 64] {
        let mut bytes = GOLDEN;
        bytes[at..at + patch.len()].copy_from_slice(patch);
        let hash = xxhash_rust::xxh3::xxh3_64(&bytes[..48]);
        bytes[48..56].copy_from_slice(&hash.to_le_bytes());
        bytes
    }

    fn is_malformed(error: &Error, expected: &str) -> bool {
        matches!(error, Error::MalformedIndex { section, .. } if *section == expected)
    }

    #[test]
    fn golden_bytes_map_to_fields_and_back() {
        assert_eq!(parse_footer(&GOLDEN).expect("valid"), golden());
        assert_eq!(golden().to_bytes(), GOLDEN);
    }

    #[test]
    fn wrong_end_magic_is_rejected() {
        let mut bytes = GOLDEN;
        bytes[63] = 1;
        assert!(is_malformed(&parse_footer(&bytes).expect_err("bad magic"), "footer"));
        bytes[63] = 0;
        bytes[56] = b'z';
        assert!(is_malformed(&parse_footer(&bytes).expect_err("bad magic"), "footer"));
    }

    #[test]
    fn every_byte_is_covered_by_the_hash_or_the_magic() {
        for at in 0..64 {
            let mut bytes = GOLDEN;
            bytes[at] ^= 0x01;
            assert!(is_malformed(&parse_footer(&bytes).expect_err("flipped"), "footer"), "{at}");
        }
    }

    #[test]
    fn nonzero_reserved_bytes_are_rejected() {
        for at in 40..48 {
            let error = parse_footer(&patched(at, &[1])).expect_err("reserved");
            assert!(is_malformed(&error, "footer"));
        }
    }

    #[test]
    fn no_generation_is_rejected() {
        for generation in [0, 2, u64::MAX] {
            let footer = parse_footer(&patched(32, &generation.to_le_bytes())).expect("valid");
            assert_eq!(footer.generation, generation);
        }
    }

    #[test]
    fn layout_accepts_the_exact_file_length() {
        let file_len = 0x1000 + 0x200 + 64;
        golden().check_layout(file_len, &Limits::default()).expect("consistent");
    }

    #[test]
    fn layout_rejects_any_other_file_length() {
        let limits = Limits::default();
        for file_len in [0, 64, 0x1000 + 0x200 + 63, 0x1000 + 0x200 + 65, u64::MAX] {
            let error = golden().check_layout(file_len, &limits).expect_err("inconsistent");
            assert!(is_malformed(&error, "footer"), "{file_len}");
        }
    }

    #[test]
    fn layout_rejects_an_index_inside_the_header() {
        let footer = Footer { index_offset: 63, index_stored_len: 1, ..golden() };
        let error = footer.check_layout(128, &Limits::default()).expect_err("in header");
        assert!(is_malformed(&error, "footer"));
    }

    #[test]
    fn layout_accepts_an_index_right_after_the_header() {
        let footer = Footer { index_offset: 64, index_stored_len: 1, ..golden() };
        footer.check_layout(129, &Limits::default()).expect("no data region is legal");
    }

    #[test]
    fn layout_sum_overflow_at_u64_max_is_a_rejection() {
        let limits = Limits::default();
        let cases = [
            (u64::MAX, 0),
            (0x40, u64::MAX),
            (u64::MAX, u64::MAX),
            (u64::MAX - 63, 0),
            (u64::MAX - 64, 1),
        ];
        for (index_offset, index_stored_len) in cases {
            let footer = Footer { index_offset, index_stored_len, ..golden() };
            let error = footer.check_layout(u64::MAX, &limits).expect_err("overflow");
            assert!(is_malformed(&error, "footer"), "{index_offset} {index_stored_len}");
        }
    }

    #[test]
    fn index_over_the_configured_cap_is_a_budget_error() {
        let limits = Limits { max_index_bytes: 0x7FF, ..Limits::default() };
        let error = golden().check_layout(0x1000 + 0x200 + 64, &limits).expect_err("over cap");
        assert!(matches!(
            error,
            Error::BudgetExceeded { what: "index", declared: 0x800, limit: 0x7FF }
        ));
    }

    #[test]
    fn index_over_one_gib_is_rejected_under_any_configuration() {
        let limits = Limits { max_index_bytes: u64::MAX, ..Limits::default() };
        for index_raw_len in [MAX_INDEX_RAW_LEN + 1, u64::MAX] {
            let footer = Footer { index_raw_len, ..golden() };
            let error = footer.check_layout(0x1000 + 0x200 + 64, &limits).expect_err("over 1 GiB");
            assert!(matches!(
                error,
                Error::BudgetExceeded { limit, .. } if limit == MAX_INDEX_RAW_LEN
            ));
        }
        let footer = Footer { index_raw_len: MAX_INDEX_RAW_LEN, ..golden() };
        footer.check_layout(0x1000 + 0x200 + 64, &limits).expect("exactly 1 GiB is legal");
    }

    #[test]
    fn index_bytes_must_match_length_and_hash() {
        let raw = b"abc";
        let footer = Footer::for_index(64, 10, raw, GENERATION_INITIAL);
        assert_eq!(footer.index_raw_len, 3);
        // Reference value computed with `twox-hash`.
        assert_eq!(footer.index_xxh3, 0x78AF_5F94_892F_3950);
        assert_eq!(footer.generation, 1);
        footer.check_index(raw).expect("matches");
        assert!(is_malformed(&footer.check_index(b"abd").expect_err("hash"), "index"));
        assert!(is_malformed(&footer.check_index(b"abcd").expect_err("length"), "index"));
        assert!(is_malformed(&footer.check_index(b"").expect_err("length"), "index"));
    }

    #[test]
    fn the_generation_of_a_new_footer_round_trips() {
        for generation in [GENERATION_INITIAL, 2, u64::MAX] {
            let footer = Footer::for_index(0x1000, 0x200, b"abc", generation);
            assert_eq!(footer.generation, generation);
            let parsed = parse_footer(&footer.to_bytes()).expect("valid");
            assert_eq!(parsed, footer);
        }
    }
}
