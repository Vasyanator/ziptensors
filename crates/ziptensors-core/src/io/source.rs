//! File: io/source.rs
//! Purpose: positional read access to a file of fixed length.
//!
//! Main responsibilities:
//! - the [`RandomAccess`] trait all parsers and the reader go through;
//! - [`FileSource`], its implementation over the operating system's positional read, from a path or
//!   from a handle the caller opened.
//!
//! Key structures: [`RandomAccess`], [`FileSource`].
//!
//! Implementation notes: positional reads need no seek state, so one handle serves concurrent
//! readers without a lock. Windows has no exact positional read, so that path loops on
//! `seek_read`; it moves the handle's cursor, which nothing here relies on.

use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use crate::error::Error;

/// Exact positional reads from a byte source whose length does not change.
///
/// Implementations are shared between threads, so reads take `&self`.
pub trait RandomAccess: Send + Sync {
    /// Length of the source in bytes.
    fn len(&self) -> u64;

    /// Whether the source holds no bytes.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fills all of `buf` from `off`.
    ///
    /// # Errors
    /// Returns [`Error::OutOfBounds`] if the range does not lie within [`RandomAccess::len`], and
    /// [`Error::Io`] if the underlying read fails. On `Err`, `buf` contents are unspecified.
    fn read_exact_at(&self, off: u64, buf: &mut [u8]) -> Result<(), Error>;
}

/// A file opened for positional reading. The length is taken once, at open.
#[derive(Debug)]
pub struct FileSource {
    file: File,
    len: u64,
    path: PathBuf,
}

impl FileSource {
    /// Opens `path` for reading.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the file cannot be opened or its metadata read, or if `path` is
    /// not a regular file.
    pub fn open(path: &Path) -> Result<Self, Error> {
        let file = File::open(path)
            .map_err(|source| Error::Io { op: "open", path: path.to_path_buf(), source })?;
        Self::from_file(file, path)
    }

    /// A source over `file`, already open for reading, which was opened from `path`. For a caller
    /// that must read and write one file through one handle, with no second resolution of its path:
    /// an append reads the container it extends through the handle it writes with.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the metadata of `file` cannot be read, or if it is not a regular
    /// file.
    pub(crate) fn from_file(file: File, path: &Path) -> Result<Self, Error> {
        let io_error = |op, source| Error::Io { op, path: path.to_path_buf(), source };
        let meta = file.metadata().map_err(|e| io_error("metadata", e))?;
        if !meta.is_file() {
            let source = io::Error::new(io::ErrorKind::InvalidInput, "not a regular file");
            return Err(io_error("open", source));
        }
        tracing::debug!(path = %path.display(), len = meta.len(), "opened file source");
        Ok(Self { file, len: meta.len(), path: path.to_path_buf() })
    }

    /// Path this source was opened from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl RandomAccess for FileSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_exact_at(&self, off: u64, buf: &mut [u8]) -> Result<(), Error> {
        // A length that does not fit u64 cannot fit the file either, so saturating keeps the
        // bounds check below correct.
        let len = u64::try_from(buf.len()).unwrap_or(u64::MAX);
        if off.checked_add(len).is_none_or(|end| end > self.len) {
            return Err(Error::OutOfBounds { offset: off, len, file_len: self.len });
        }
        read_exact_os(&self.file, off, buf).map_err(|source| Error::Io {
            op: "read_exact_at",
            path: self.path.clone(),
            source,
        })
    }
}

/// Fills `buf` from `off`. A file shorter than the range yields `UnexpectedEof`.
#[cfg(unix)]
fn read_exact_os(file: &File, off: u64, buf: &mut [u8]) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, off)
}

/// Fills `buf` from `off`. A file shorter than the range yields `UnexpectedEof`.
#[cfg(windows)]
fn read_exact_os(file: &File, off: u64, buf: &mut [u8]) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut pos = off;
    let mut done = 0;
    while done < buf.len() {
        match file.seek_read(&mut buf[done..], pos) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                done += n;
                // n <= buf.len() and the caller checked off + buf.len() against the file length.
                pos += u64::try_from(n).map_err(io::Error::other)?;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
compile_error!("ziptensors-core supports unix and windows targets only");

#[cfg(test)]
mod tests {
    use super::{FileSource, RandomAccess};
    use crate::error::Error;
    use std::fs;
    use std::io::ErrorKind;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Scratch file under the system temp directory, removed on drop.
    struct TempFile(PathBuf);

    impl TempFile {
        fn with_contents(bytes: &[u8]) -> Self {
            static NEXT: AtomicU32 = AtomicU32::new(0);
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let name = format!("ziptensors-source-{}-{id}", std::process::id());
            let path = std::env::temp_dir().join(name);
            fs::write(&path, bytes).expect("scratch file must be writable");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            // A leftover in the temp directory is harmless and a panic in drop would abort.
            if let Err(e) = fs::remove_file(&self.0) {
                eprintln!("cannot remove {}: {e}", self.0.display());
            }
        }
    }

    fn digits() -> TempFile {
        TempFile::with_contents(b"0123456789")
    }

    #[test]
    fn reads_exact_ranges() {
        let temp = digits();
        let src = FileSource::open(temp.path()).expect("open");
        assert_eq!(src.len(), 10);
        assert!(!src.is_empty());
        assert_eq!(src.path(), temp.path());

        let mut buf = [0_u8; 4];
        src.read_exact_at(0, &mut buf).expect("start of file");
        assert_eq!(&buf, b"0123");
        src.read_exact_at(6, &mut buf).expect("range ending at the last byte");
        assert_eq!(&buf, b"6789");
    }

    #[test]
    fn empty_read_at_end_succeeds() {
        let temp = digits();
        let src = FileSource::open(temp.path()).expect("open");
        src.read_exact_at(10, &mut []).expect("empty range at end of file");
    }

    #[test]
    fn empty_file_is_empty() {
        let temp = TempFile::with_contents(b"");
        let src = FileSource::open(temp.path()).expect("open");
        assert!(src.is_empty());
    }

    #[test]
    fn read_past_end_is_out_of_bounds() {
        let temp = digits();
        let src = FileSource::open(temp.path()).expect("open");
        let mut buf = [0_u8; 4];
        let err = src.read_exact_at(7, &mut buf).expect_err("last byte is past the end");
        assert!(
            matches!(err, Error::OutOfBounds { offset: 7, len: 4, file_len: 10 }),
            "{err:?}"
        );
    }

    #[test]
    fn empty_read_beyond_end_is_out_of_bounds() {
        let temp = digits();
        let src = FileSource::open(temp.path()).expect("open");
        let err = src.read_exact_at(11, &mut []).expect_err("start is past the end");
        assert!(matches!(err, Error::OutOfBounds { offset: 11, len: 0, file_len: 10 }), "{err:?}");
    }

    #[test]
    fn offset_overflow_is_out_of_bounds() {
        let temp = digits();
        let src = FileSource::open(temp.path()).expect("open");
        let mut buf = [0_u8; 2];
        let err = src.read_exact_at(u64::MAX, &mut buf).expect_err("offset + len overflows");
        assert!(matches!(err, Error::OutOfBounds { offset: u64::MAX, len: 2, .. }), "{err:?}");
    }

    #[test]
    fn file_shrunk_after_open_is_an_io_error() {
        let temp = digits();
        let src = FileSource::open(temp.path()).expect("open");
        fs::write(temp.path(), b"01").expect("shrink");
        let mut buf = [0_u8; 4];
        let err = src.read_exact_at(0, &mut buf).expect_err("bytes are gone");
        match err {
            Error::Io { op, path, source } => {
                assert_eq!(op, "read_exact_at");
                assert_eq!(path, temp.path());
                assert_eq!(source.kind(), ErrorKind::UnexpectedEof);
            }
            other => panic!("expected Io, got {other:?}"),
        }
    }

    #[test]
    fn missing_file_is_an_io_error_with_the_path() {
        let temp = digits();
        let missing = temp.path().with_extension("missing");
        let err = FileSource::open(&missing).expect_err("file does not exist");
        match err {
            Error::Io { op, path, source } => {
                assert_eq!(op, "open");
                assert_eq!(path, missing);
                assert_eq!(source.kind(), ErrorKind::NotFound);
            }
            other => panic!("expected Io, got {other:?}"),
        }
    }

    #[test]
    fn directory_is_rejected() {
        let err = FileSource::open(&std::env::temp_dir()).expect_err("not a regular file");
        assert!(matches!(err, Error::Io { op: "open", .. }), "{err:?}");
    }

    #[test]
    fn concurrent_reads_share_one_handle() {
        let contents: Vec<u8> = (0..=255).collect();
        let temp = TempFile::with_contents(&contents);
        let src = Arc::new(FileSource::open(temp.path()).expect("open"));
        let workers: Vec<_> = (0_u8..8)
            .map(|start| {
                let src = Arc::clone(&src);
                std::thread::spawn(move || {
                    let mut byte = [0_u8; 1];
                    for _ in 0..1000 {
                        src.read_exact_at(u64::from(start), &mut byte).expect("in range");
                        assert_eq!(byte[0], start);
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().expect("reader thread must not panic");
        }
    }
}
