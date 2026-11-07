//! File: write/commit.rs
//! Purpose: the writer's side of the commit protocol of spec invariant 12.
//!
//! Main responsibilities: [`Sink`], the append position of the temporary file with its rollback and
//! the header rewrite that sets feature bits; [`verify_output`], the full decode through the real
//! reader. Sync, close and rename belong to
//! `io::TempFile`, shared with a restore.
//!
//! Key structures: [`Sink`], [`SourceHash`].
//!
//! Key functions: [`verify_output`], [`write_index`].
//!
//! Implementation notes: verification opens the file as any reader does, with default limits and no special path: a
//! file that would need raised limits to open is refused here and not found out by its user. A rollback
//! truncates, since bytes past the logical end must not survive into the index offset.

use std::fs::File;
use std::io::{self, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::Instant;

use crate::error::Error;
use crate::format::{FOOTER_LEN, Footer, HEADER_LEN, Header, IndexBuilder};
use crate::io::TempFile;
use crate::read::{Container, OpenOptions};

use super::report::{ClassTable, VerifyReport, unreferenced_table};

/// zstd level of the stored index.
const INDEX_LEVEL: i32 = 3;

/// The temporary output: an open file and a position that can move back.
///
/// After a failed write the file holds an unknown tail, so the sink refuses all further work.
#[derive(Debug)]
pub(super) struct Sink {
    temp: TempFile,
    /// Logical end of the file.
    position: u64,
    /// A write or a rollback failed.
    broken: bool,
}

impl Sink {
    /// Creates a temporary file beside `output` and writes the header.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if `output` has no file name, or the file cannot be created or written.
    pub(super) fn create(output: &Path) -> Result<Self, Error> {
        let temp = TempFile::beside(output)?;
        let mut sink = Self { temp, position: 0, broken: false };
        sink.write(&Header::current().to_bytes())?;
        Ok(sink)
    }

    /// Path of the temporary file.
    pub(super) fn path(&self) -> &Path {
        self.temp.path()
    }

    /// Logical end of the file, where the next write lands.
    pub(super) fn position(&self) -> u64 {
        self.position
    }

    /// The file, unless an earlier failure made it unusable.
    fn file(&mut self) -> Result<&mut File, Error> {
        if self.broken {
            return Err(Error::Io {
                op: "write",
                path: self.path().to_path_buf(),
                source: io::Error::other("the output is unusable after an earlier write failure"),
            });
        }
        Ok(self.temp.parts_mut().0)
    }

    /// Fails if a write failed before, so that a caller stops before it does more work.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the sink is broken.
    pub(super) fn check(&mut self) -> Result<(), Error> {
        self.file().map(|_| ())
    }

    /// Marks the sink unusable. For a failure that left the pack's records out of step with the file.
    pub(super) fn poison(&mut self) {
        self.broken = true;
    }

    /// Appends `bytes` at the logical end.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the write fails; the sink then refuses everything else.
    pub(super) fn write(&mut self, bytes: &[u8]) -> Result<(), Error> {
        let result = self.file()?.write_all(bytes);
        result.map_err(|source| {
            self.broken = true;
            Error::Io { op: "write", path: self.path().to_path_buf(), source }
        })?;
        // Lossless: `usize` is at most 64 bits on every supported target. A file cannot reach 2^64.
        self.position += bytes.len() as u64;
        Ok(())
    }

    /// Moves the logical end back to `position`, dropping what followed it.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the file cannot be cut; the sink then refuses everything else.
    pub(super) fn rollback(&mut self, position: u64) -> Result<(), Error> {
        if position == self.position {
            return Ok(());
        }
        let file = self.file()?;
        let result = file.set_len(position).and_then(|()| file.seek(SeekFrom::Start(position)));
        result.map_err(|source| {
            self.broken = true;
            Error::Io { op: "roll back", path: self.path().to_path_buf(), source }
        })?;
        self.position = position;
        Ok(())
    }

    /// Writes `header` over the one at the start of the file, leaving the logical end where it is. For
    /// feature bits known only once the blobs are in, spec section 7a; the commit's verification then
    /// reads the header like any reader.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if a seek or the write fails; the sink then refuses everything else.
    pub(super) fn patch_header(&mut self, header: &Header) -> Result<(), Error> {
        let end = self.position;
        let file = self.file()?;
        let result = file
            .seek(SeekFrom::Start(0))
            .and_then(|_| file.write_all(&header.to_bytes()))
            .and_then(|()| file.seek(SeekFrom::Start(end)));
        result.map_err(|source| {
            self.broken = true;
            Error::Io { op: "write header", path: self.path().to_path_buf(), source }
        })?;
        Ok(())
    }

    /// Commits the file to `output`, by [`TempFile::commit`].
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the sink is broken, or the sync or the rename fails. The temporary file
    /// is then removed. Returns [`Error::CommittedNotDurable`] if the file was renamed to `output` but
    /// its directory could not be synced.
    pub(super) fn commit(mut self, output: &Path) -> Result<(), Error> {
        self.check()?;
        self.temp.commit(output)
    }
}

/// Serializes the index and appends it and the footer. Returns the stored length of the index.
///
/// # Errors
/// Returns [`Error::Zstd`] if the index cannot be compressed and [`Error::Io`] if a write fails.
pub(super) fn write_index(sink: &mut Sink, index: &IndexBuilder) -> Result<u64, Error> {
    let raw = index.serialize();
    let stored = zstd::bulk::compress(&raw, INDEX_LEVEL)
        .map_err(|error| Error::Zstd { detail: error.to_string() })?;
    let index_offset = sink.position();
    sink.write(&stored)?;
    // Lossless: `usize` is at most 64 bits on every supported target.
    let stored_len = stored.len() as u64;
    sink.write(&Footer::for_index(index_offset, stored_len, &raw).to_bytes())?;
    debug_assert_eq!(sink.position(), index_offset + stored_len + FOOTER_LEN);
    debug_assert!(index_offset >= HEADER_LEN);
    tracing::debug!(raw = raw.len(), stored = stored_len, "wrote index");
    Ok(stored_len)
}

/// The hash a member must decode to, taken from its source and not from the file being verified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SourceHash {
    /// Member name.
    pub(super) name: String,
    /// BLAKE3 of the source content.
    pub(super) blake3: [u8; 32],
}

/// Opens the file at `path` through the real reader, with its default limits, and proves every member:
/// the record states the source hash, and the whole member decodes to it. So the index is re-parsed and
/// validated here, which `IndexBuilder` does not do.
///
/// Costs one sequential read of each member, however they share blobs. Also returns the accounting
/// of the blobs no member reaches, from the same index, by [`unreferenced_table`].
///
/// # Errors
/// Returns any error of [`Container::open`] or of a read, and [`Error::VerifyFailed`] if a hash differs.
/// [`Error::MalformedIndex`] for `MEMB` if the file does not hold exactly `sources`, in order.
pub(super) fn verify_output(
    path: &Path,
    threads: usize,
    sources: &[SourceHash],
) -> Result<(VerifyReport, ClassTable), Error> {
    let started = Instant::now();
    let options = OpenOptions { threads, ..OpenOptions::default() };
    let container = Container::open(path, &options)?;
    let members = container.members();
    if members.len() != sources.len() {
        return Err(Error::MalformedIndex {
            section: "MEMB",
            detail: format!("{} members written, {} expected", members.len(), sources.len()),
        });
    }
    let mut bytes = 0_u64;
    for (info, source) in members.iter().zip(sources) {
        if info.name != source.name {
            return Err(Error::MalformedIndex {
                section: "MEMB",
                detail: format!("member {:?} written, {:?} expected", info.name, source.name),
            });
        }
        if info.blake3 != source.blake3 {
            return Err(Error::VerifyFailed {
                member: info.name.clone(),
                expected: blake3::Hash::from_bytes(source.blake3),
                actual: blake3::Hash::from_bytes(info.blake3),
            });
        }
        container.member(&info.name)?.verify()?;
        bytes += info.len;
    }
    let elapsed = started.elapsed();
    tracing::info!(bytes, ?elapsed, "verified output");
    let unreferenced = unreferenced_table(container.index())?;
    Ok((VerifyReport { bytes, elapsed }, unreferenced))
}

#[cfg(test)]
mod tests {
    use super::Sink;
    use crate::chain::FEATURE_DELTA;
    use crate::format::{Header, parse_header};
    use std::fs;
    use std::path::PathBuf;

    /// A unique scratch directory, removed on drop.
    struct Dir(PathBuf);

    impl Dir {
        fn new(tag: &str) -> Self {
            let name = format!("ziptensors-commit-{tag}-{}", std::process::id());
            let path = std::env::temp_dir().join(name);
            fs::create_dir_all(&path).expect("scratch directory");
            Self(path)
        }

        fn entries(&self) -> Vec<String> {
            let entries = fs::read_dir(&self.0).expect("readable");
            entries
                .map(|entry| entry.expect("entry").file_name().to_string_lossy().into_owned())
                .collect()
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            if let Err(error) = fs::remove_dir_all(&self.0) {
                eprintln!("cannot remove {}: {error}", self.0.display());
            }
        }
    }

    #[test]
    fn a_rollback_cuts_the_file_and_moves_the_position() {
        let dir = Dir::new("rollback");
        let mut sink = Sink::create(&dir.0.join("out")).expect("creates");
        assert_eq!(sink.position(), 64);
        sink.write(&[1; 100]).expect("writes");
        sink.rollback(64 + 10).expect("rolls back");
        sink.write(&[2; 5]).expect("writes");
        let bytes = fs::read(sink.path()).expect("reads");
        assert_eq!(bytes.len(), 64 + 15);
        assert_eq!(&bytes[64 + 9..64 + 11], [1, 2]);
    }

    #[test]
    fn a_patched_header_keeps_the_end_and_the_data() {
        let dir = Dir::new("patch");
        let mut sink = Sink::create(&dir.0.join("out")).expect("creates");
        sink.write(&[1; 10]).expect("writes");
        let header = Header { required_features: FEATURE_DELTA, ..Header::current() };
        sink.patch_header(&header).expect("patches");
        assert_eq!(sink.position(), 74);
        sink.write(&[2; 3]).expect("writes");
        let bytes = fs::read(sink.path()).expect("reads");
        assert_eq!(bytes.len(), 77);
        assert_eq!(&bytes[64..], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]);
        let first: [u8; 64] = bytes[..64].try_into().expect("64 bytes");
        assert_eq!(parse_header(&first).expect("valid"), header);
    }

    #[test]
    fn a_poisoned_sink_refuses_work_and_still_cleans_up() {
        let dir = Dir::new("poison");
        let output = dir.0.join("out");
        let mut sink = Sink::create(&output).expect("creates");
        sink.poison();
        assert!(sink.check().is_err());
        assert!(sink.write(b"x").is_err());
        assert!(sink.commit(&output).is_err());
        assert_eq!(dir.entries(), [""; 0]);
    }
}
