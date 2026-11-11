//! File: io/splice.rs
//! Purpose: views of a [`RandomAccess`] source that differ from the bytes on disk: the first bytes of
//! a source, or a source followed by bytes held in memory.
//!
//! Main responsibilities:
//! - [`SpliceSource`]: the first `base_len` bytes of a base source, then a tail held in memory. The
//!   append commit verifies a new generation through it before any byte of the new index is on disk
//!   (spec invariant 12, append protocol).
//! - [`PrefixSource`]: the first `len` bytes of a source, so that a generation that ends before the
//!   end of a file can be opened and verified without cutting the file.
//!
//! Key structures: [`SpliceSource`], [`PrefixSource`].
//!
//! Implementation notes: both are fixed-length sources like any other: lengths are checked once at
//! construction, and every read is checked against the view's length with overflow handled before
//! any I/O, so a read never reaches base bytes past the view.

use std::sync::Arc;

use super::source::RandomAccess;
use crate::error::Error;

/// The length of `buf` as a `u64`. A length that does not fit cannot fit a source either, so
/// saturating keeps every bounds check below correct.
fn buf_len(buf: &[u8]) -> u64 {
    u64::try_from(buf.len()).unwrap_or(u64::MAX)
}

/// Checks that `[off, off + len)` lies in a source of `source_len` bytes; returns the end.
fn checked_end(off: u64, len: u64, source_len: u64) -> Result<u64, Error> {
    match off.checked_add(len) {
        Some(end) if end <= source_len => Ok(end),
        Some(_) | None => Err(Error::OutOfBounds { offset: off, len, file_len: source_len }),
    }
}

/// The first `base_len` bytes of `base`, then `tail`. Its length is `base_len + tail.len()`.
#[derive(Clone)]
pub struct SpliceSource {
    base: Arc<dyn RandomAccess>,
    base_len: u64,
    tail: Arc<[u8]>,
    len: u64,
}

impl std::fmt::Debug for SpliceSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpliceSource")
            .field("base_len", &self.base_len)
            .field("tail_len", &self.tail.len())
            .finish_non_exhaustive()
    }
}

impl SpliceSource {
    /// A view of `base` cut at `base_len`, continued by `tail`.
    ///
    /// # Errors
    /// Returns [`Error::OutOfBounds`] if `base_len` is longer than `base`, or if the view's length
    /// overflows `u64`.
    pub fn new(base: Arc<dyn RandomAccess>, base_len: u64, tail: Vec<u8>) -> Result<Self, Error> {
        let source_len = base.len();
        checked_end(0, base_len, source_len)?;
        let tail_len = buf_len(&tail);
        let len = base_len.checked_add(tail_len).ok_or(Error::OutOfBounds {
            offset: base_len,
            len: tail_len,
            file_len: u64::MAX,
        })?;
        Ok(Self { base, base_len, tail: tail.into(), len })
    }
}

impl RandomAccess for SpliceSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_exact_at(&self, off: u64, buf: &mut [u8]) -> Result<(), Error> {
        let len = buf_len(buf);
        let end = checked_end(off, len, self.len)?;
        let out_of_bounds = || Error::OutOfBounds { offset: off, len, file_len: self.len };
        // Bytes below the seam come from the base, the rest from the tail; a read across the seam
        // takes both parts. `off <= end <= len` makes each conversion below fit `buf`.
        let split = self.base_len.clamp(off, end) - off;
        let split = usize::try_from(split).map_err(|_| out_of_bounds())?;
        let (below, above) = buf.split_at_mut(split);
        if !below.is_empty() {
            self.base.read_exact_at(off, below)?;
        }
        if !above.is_empty() {
            // The tail part starts at or past the seam, so this is its offset inside the tail.
            let start = off.max(self.base_len) - self.base_len;
            let start = usize::try_from(start).map_err(|_| out_of_bounds())?;
            let bytes = start
                .checked_add(above.len())
                .and_then(|stop| self.tail.get(start..stop))
                .ok_or_else(out_of_bounds)?;
            above.copy_from_slice(bytes);
        }
        Ok(())
    }
}

/// The first `len` bytes of `inner`.
#[derive(Clone)]
pub struct PrefixSource {
    inner: Arc<dyn RandomAccess>,
    len: u64,
}

impl std::fmt::Debug for PrefixSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefixSource").field("len", &self.len).finish_non_exhaustive()
    }
}

impl PrefixSource {
    /// A view of the first `len` bytes of `inner`.
    ///
    /// # Errors
    /// Returns [`Error::OutOfBounds`] if `len` is longer than `inner`.
    pub fn new(inner: Arc<dyn RandomAccess>, len: u64) -> Result<Self, Error> {
        checked_end(0, len, inner.len())?;
        Ok(Self { inner, len })
    }
}

impl RandomAccess for PrefixSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_exact_at(&self, off: u64, buf: &mut [u8]) -> Result<(), Error> {
        checked_end(off, buf_len(buf), self.len)?;
        self.inner.read_exact_at(off, buf)
    }
}

#[cfg(test)]
mod tests {
    use super::{PrefixSource, SpliceSource};
    use crate::error::Error;
    use crate::io::RandomAccess;
    use std::sync::Arc;

    /// A source in memory, as the fuzz harness and the reader tests use.
    struct Memory(Vec<u8>);

    impl RandomAccess for Memory {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }

        fn read_exact_at(&self, off: u64, buf: &mut [u8]) -> Result<(), Error> {
            let start = usize::try_from(off).expect("small");
            let bytes = self.0.get(start..start + buf.len()).ok_or(Error::OutOfBounds {
                offset: off,
                len: buf.len() as u64,
                file_len: self.len(),
            })?;
            buf.copy_from_slice(bytes);
            Ok(())
        }
    }

    fn base() -> Arc<dyn RandomAccess> {
        Arc::new(Memory((0..20).collect()))
    }

    fn read(source: &dyn RandomAccess, off: u64, len: usize) -> Result<Vec<u8>, Error> {
        let mut buf = vec![0xEE; len];
        source.read_exact_at(off, &mut buf)?;
        Ok(buf)
    }

    #[test]
    fn a_splice_reads_base_then_tail_exactly_across_the_seam() {
        let splice = SpliceSource::new(base(), 8, vec![100, 101, 102]).expect("fits");
        assert_eq!(splice.len(), 11);
        let mut expected: Vec<u8> = (0..8).collect();
        expected.extend([100, 101, 102]);
        for off in 0..=11 {
            for len in 0..=11 - off {
                let got = read(&splice, off as u64, len).expect("inside");
                assert_eq!(got, expected[off..off + len], "{off} {len}");
            }
        }
    }

    #[test]
    fn a_splice_refuses_reads_past_its_end_and_overflow() {
        let splice = SpliceSource::new(base(), 8, vec![1, 2]).expect("fits");
        for (off, len) in [(0, 11), (10, 1), (11, 1), (u64::MAX, 1)] {
            let error = read(&splice, off, len).expect_err("outside");
            assert!(matches!(error, Error::OutOfBounds { file_len: 10, .. }), "{error:?}");
        }
        assert_eq!(read(&splice, 10, 0).expect("empty at the end"), [0_u8; 0]);
    }

    #[test]
    fn a_splice_cut_past_its_base_is_refused() {
        let error = SpliceSource::new(base(), 21, Vec::new()).expect_err("too long");
        assert!(matches!(error, Error::OutOfBounds { len: 21, file_len: 20, .. }), "{error:?}");
        let whole = SpliceSource::new(base(), 20, Vec::new()).expect("the whole base");
        assert_eq!(whole.len(), 20);
        let tail_only = SpliceSource::new(base(), 0, vec![7, 8]).expect("no base bytes");
        assert_eq!(read(&tail_only, 0, 2).expect("inside"), [7, 8]);
    }

    #[test]
    fn a_splice_never_reads_base_bytes_past_the_cut() {
        // Base bytes 5.. differ from the tail; a read of the seam must see the tail.
        let splice = SpliceSource::new(base(), 5, vec![9; 4]).expect("fits");
        assert_eq!(read(&splice, 3, 4).expect("inside"), [3, 4, 9, 9]);
    }

    #[test]
    fn a_prefix_reads_inside_and_refuses_past_its_length() {
        let prefix = PrefixSource::new(base(), 6).expect("fits");
        assert_eq!(prefix.len(), 6);
        assert_eq!(read(&prefix, 2, 4).expect("inside"), [2, 3, 4, 5]);
        for (off, len) in [(0, 7), (6, 1), (5, 2), (u64::MAX, 1), (u64::MAX - 1, 2)] {
            let error = read(&prefix, off, len).expect_err("outside");
            assert!(matches!(error, Error::OutOfBounds { file_len: 6, .. }), "{error:?}");
        }
    }

    #[test]
    fn a_prefix_longer_than_its_source_is_refused() {
        let error = PrefixSource::new(base(), 21).expect_err("too long");
        assert!(matches!(error, Error::OutOfBounds { len: 21, file_len: 20, .. }), "{error:?}");
        assert_eq!(PrefixSource::new(base(), 0).expect("empty").len(), 0);
    }
}
