//! File: profile/reader.rs
//! Purpose: the bounded sequential reader every profile parser goes through.
//!
//! Main responsibilities:
//! - checks each requested length against the source before it reads or allocates;
//! - buffers a small window, so that walking a header does not issue one read per field.
//!
//! Key structures: [`Reader`].
//!
//! Implementation notes: running past the end is [`Stop::Malformed`], since the file claimed bytes
//! it does not hold. A failed read of bytes that do exist is [`Stop::Io`]. The window is small
//! because parsers skip over tensor data and each skip may discard it.

use super::{ParseResult, Stop};
use crate::io::RandomAccess;

/// Bytes fetched per window. Also the longest fixed-size value a reader can return.
const WINDOW_BYTES: usize = 16 * 1024;

/// Longest varint of a 64-bit value, in bytes.
const MAX_VARINT_BYTES: u32 = 10;

/// A cursor over a [`RandomAccess`] source.
pub(super) struct Reader<'a> {
    src: &'a dyn RandomAccess,
    len: u64,
    pos: u64,
    window: Vec<u8>,
    window_start: u64,
}

impl<'a> Reader<'a> {
    /// Starts at offset 0 of `src`. The length is taken once, here.
    pub(super) fn new(src: &'a dyn RandomAccess) -> Self {
        Self { src, len: src.len(), pos: 0, window: Vec::new(), window_start: 0 }
    }

    /// Length of the source in bytes.
    pub(super) fn len(&self) -> u64 {
        self.len
    }

    /// Current offset. Never past [`Reader::len`].
    pub(super) fn pos(&self) -> u64 {
        self.pos
    }

    /// Bytes between the cursor and the end of the source.
    pub(super) fn remaining(&self) -> u64 {
        self.len - self.pos
    }

    /// Moves to `pos`. Stops as malformed if that lies past the end.
    pub(super) fn seek(&mut self, pos: u64) -> ParseResult<()> {
        if pos > self.len {
            return Err(Stop::Malformed("offset past the end of the source"));
        }
        self.pos = pos;
        Ok(())
    }

    /// Moves `count` bytes forward. Stops as malformed if that lies past the end.
    pub(super) fn skip(&mut self, count: u64) -> ParseResult<()> {
        let pos = self.pos.checked_add(count).ok_or(Stop::Malformed("offset overflows"))?;
        self.seek(pos)
    }

    /// Reads `N` bytes, at most [`WINDOW_BYTES`].
    pub(super) fn array<const N: usize>(&mut self) -> ParseResult<[u8; N]> {
        let mut out = [0_u8; N];
        let bytes = self.window_slice(N)?;
        out.copy_from_slice(bytes);
        self.skip(u64::try_from(N).map_err(|_| Stop::Malformed("value too long"))?)?;
        Ok(out)
    }

    /// Reads one byte.
    pub(super) fn u8(&mut self) -> ParseResult<u8> {
        let [byte] = self.array::<1>()?;
        Ok(byte)
    }

    /// Reads a little-endian `u16`.
    pub(super) fn u16(&mut self) -> ParseResult<u16> {
        self.array().map(u16::from_le_bytes)
    }

    /// Reads a little-endian `u32`.
    pub(super) fn u32(&mut self) -> ParseResult<u32> {
        self.array().map(u32::from_le_bytes)
    }

    /// Reads a little-endian `u64`.
    pub(super) fn u64(&mut self) -> ParseResult<u64> {
        self.array().map(u64::from_le_bytes)
    }

    /// Reads a protobuf base-128 varint, least significant group first.
    ///
    /// Stops as malformed if it runs past 10 bytes or does not fit 64 bits.
    pub(super) fn varint(&mut self) -> ParseResult<u64> {
        let mut value = 0_u64;
        for index in 0..MAX_VARINT_BYTES {
            let byte = self.u8()?;
            let group = u64::from(byte & 0x7F);
            // The tenth byte holds bit 63 only; anything above it does not fit.
            if index == MAX_VARINT_BYTES - 1 && group > 1 {
                return Err(Stop::Malformed("varint does not fit 64 bits"));
            }
            value |= group << (7 * index);
            if byte < 0x80 {
                return Ok(value);
            }
        }
        Err(Stop::Malformed("varint is too long"))
    }

    /// Reads `len` bytes into a new buffer. The buffer is allocated only after `len` has been
    /// checked against `cap` and against the bytes the source really holds.
    pub(super) fn take(&mut self, len: u64, cap: u64) -> ParseResult<Vec<u8>> {
        if len > cap {
            return Err(Stop::Malformed("declared length is over the parser's cap"));
        }
        if len > self.remaining() {
            return Err(Stop::Malformed("declared length is past the end of the source"));
        }
        let size = usize::try_from(len).map_err(|_| Stop::Malformed("length is not addressable"))?;
        let mut out = vec![0_u8; size];
        self.src.read_exact_at(self.pos, &mut out)?;
        self.skip(len)?;
        Ok(out)
    }

    /// Returns the next `count` bytes without moving, fetching a new window when needed.
    fn window_slice(&mut self, count: usize) -> ParseResult<&[u8]> {
        if count > WINDOW_BYTES {
            return Err(Stop::Malformed("fixed value is over the window"));
        }
        let wanted = u64::try_from(count).map_err(|_| Stop::Malformed("value too long"))?;
        if wanted > self.remaining() {
            return Err(Stop::Malformed("source ends inside a value"));
        }
        if !self.window_holds(wanted) {
            let size = u64::try_from(WINDOW_BYTES)
                .map_err(|_| Stop::Malformed("window too long"))?
                .min(self.remaining());
            let size = usize::try_from(size).map_err(|_| Stop::Malformed("window too long"))?;
            self.window.resize(size, 0);
            self.window_start = self.pos;
            self.src.read_exact_at(self.pos, &mut self.window)?;
        }
        // `window_holds` or the fetch above put the requested span inside the window, and the
        // window never exceeds WINDOW_BYTES, so the difference fits usize.
        let start = usize::try_from(self.pos - self.window_start)
            .map_err(|_| Stop::Malformed("window offset too long"))?;
        self.window
            .get(start..start + count)
            .ok_or(Stop::Malformed("value is outside the window"))
    }

    /// Whether the current window covers `count` bytes at the cursor.
    fn window_holds(&self, count: u64) -> bool {
        let Ok(window_len) = u64::try_from(self.window.len()) else {
            return false;
        };
        let Some(end) = self.pos.checked_add(count) else {
            return false;
        };
        self.pos >= self.window_start && end <= self.window_start.saturating_add(window_len)
    }
}

#[cfg(test)]
mod tests {
    use super::{Reader, WINDOW_BYTES};
    use crate::error::Error;
    use crate::io::RandomAccess;
    use crate::profile::Stop;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// In-memory source that counts the reads it serves.
    struct Counted {
        bytes: Vec<u8>,
        reads: AtomicU64,
    }

    impl Counted {
        fn new(bytes: Vec<u8>) -> Self {
            Self { bytes, reads: AtomicU64::new(0) }
        }
    }

    impl RandomAccess for Counted {
        fn len(&self) -> u64 {
            u64::try_from(self.bytes.len()).expect("test buffer length fits u64")
        }

        fn read_exact_at(&self, off: u64, buf: &mut [u8]) -> Result<(), Error> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            let start = usize::try_from(off).expect("test offset fits usize");
            buf.copy_from_slice(&self.bytes[start..start + buf.len()]);
            Ok(())
        }
    }

    fn is_malformed<T>(result: &Result<T, Stop>) -> bool {
        matches!(result, Err(Stop::Malformed(_)))
    }

    #[test]
    fn integers_are_little_endian() {
        let src = Counted::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let mut reader = Reader::new(&src);
        assert_eq!(reader.u8().expect("in range"), 1);
        assert_eq!(reader.u16().expect("in range"), 0x0302);
        assert_eq!(reader.u32().expect("in range"), 0x0706_0504);
        assert_eq!(reader.u64().expect("in range"), 0x0F0E_0D0C_0B0A_0908);
        assert_eq!(reader.remaining(), 0);
        assert!(is_malformed(&reader.u8()));
    }

    #[test]
    fn a_sequential_walk_shares_one_window() {
        let src = Counted::new(vec![0; 1024]);
        let mut reader = Reader::new(&src);
        for _ in 0..256 {
            reader.u32().expect("in range");
        }
        assert_eq!(src.reads.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_value_across_the_window_edge_reads_correctly() {
        let bytes: Vec<u8> = (0..WINDOW_BYTES * 2).map(|i| (i % 251).to_le_bytes()[0]).collect();
        let src = Counted::new(bytes.clone());
        let mut reader = Reader::new(&src);
        reader.u8().expect("in range");
        let edge = u64::try_from(WINDOW_BYTES - 2).expect("fits");
        reader.seek(edge).expect("in range");
        let got = reader.u32().expect("in range");
        let start = WINDOW_BYTES - 2;
        let want: [u8; 4] = bytes[start..start + 4].try_into().expect("four bytes");
        assert_eq!(got, u32::from_le_bytes(want));
    }

    #[test]
    fn varints_decode_and_reject_overlong_forms() {
        let cases: [(&[u8], u64); 4] = [
            (&[0x00], 0),
            (&[0xAC, 0x02], 300),
            (&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01], u64::MAX),
            (&[0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01], 1 << 63),
        ];
        for (bytes, want) in cases {
            let src = Counted::new(bytes.to_vec());
            assert_eq!(Reader::new(&src).varint().expect("valid"), want);
        }
        for bytes in [vec![0x80; 11], [vec![0xFF; 9], vec![0x02]].concat(), vec![0x80]] {
            let src = Counted::new(bytes);
            assert!(is_malformed(&Reader::new(&src).varint()));
        }
    }

    #[test]
    fn take_checks_before_it_allocates() {
        let src = Counted::new(vec![7; 16]);
        let mut reader = Reader::new(&src);
        assert!(is_malformed(&reader.take(u64::MAX, u64::MAX)));
        assert!(is_malformed(&reader.take(17, 1024)));
        assert!(is_malformed(&reader.take(16, 15)));
        assert_eq!(src.reads.load(Ordering::Relaxed), 0);
        assert_eq!(reader.take(16, 16).expect("in range"), vec![7; 16]);
        assert_eq!(reader.remaining(), 0);
    }

    #[test]
    fn seeking_past_the_end_stops_and_overflow_does_not_wrap() {
        let src = Counted::new(vec![0; 8]);
        let mut reader = Reader::new(&src);
        reader.seek(8).expect("end of source is a valid position");
        assert!(is_malformed(&reader.seek(9)));
        assert!(is_malformed(&reader.skip(u64::MAX)));
        assert_eq!(reader.pos(), 8);
    }
}
