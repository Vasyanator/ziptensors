//! File: read/restore.rs
//! Purpose: reading a member from end to end: `restore_file`, `decode_to`, `verify`, and
//! `Container::restore_all` over every member.
//!
//! Main responsibilities: hash the whole member while streaming it; commit a restored file by the
//! protocol of spec invariant 12, through `io::TempFile`; stream a member into a caller's writer;
//! restore every member of a container under one directory, with every name checked first.
//!
//! Key structures: [`RestoreReport`], [`RestoreAllReport`].
//!
//! Key functions: [`Member::restore_file`], [`Member::decode_to`], [`Member::verify`],
//! [`Container::restore_all`], and the private `hash_stream` they all go through.
//!
//! Implementation notes: only extents and the member hash are involved, never tensor ranges. The
//! stream buffer is the destination of its reads, so memory is one batch more than a `read_at`, and
//! at least two chunks, as one widened chunk spans two chunks of the member: a shorter buffer could not
//! end on a chunk boundary, and every chunk would decode twice.
//! Modification time and permissions are not restored, by spec section 2.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::error::Error;
use crate::format::check_restore_name;
use crate::io::{TempFile, same_file};

use super::batch::to_usize;
use super::container::Container;
use super::member::Member;

/// What [`Member::restore_file`] or [`Member::decode_to`] produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestoreReport {
    /// Bytes written.
    pub bytes: u64,
    /// BLAKE3 of those bytes, equal to the hash of the member.
    pub blake3: [u8; 32],
}

/// What [`Container::restore_all`] produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestoreAllReport {
    /// Members restored: all members of the container.
    pub members: usize,
    /// Bytes written over all of them.
    pub bytes: u64,
}

impl Container {
    /// Restores every member to `dir` joined with its name, in file order, each through
    /// [`Member::restore_file`], creating `dir` and the parent directories names need.
    ///
    /// Every name is checked with [`check_restore_name`] before anything is created, so a container
    /// with one hostile name leaves no trace at all. After that, members are committed one by one:
    /// a failure stops the run and keeps the members restored before it, and directories created
    /// so far, while the failed member leaves no file (the guarantee of [`Member::restore_file`]).
    /// Logs one `info` line per member with `done` and `total`.
    ///
    /// # Errors
    /// [`Error::UnsafeMemberName`] naming the first unsafe member, before anything is written.
    /// [`Error::Io`] with op `"create directory"` if a directory cannot be created, and any error of
    /// [`Member::restore_file`] for the member that failed.
    pub fn restore_all(&self, dir: &Path) -> Result<RestoreAllReport, Error> {
        let members = self.members();
        for info in members {
            check_restore_name(&info.name)?;
        }
        let total = members.len();
        tracing::info!(dir = %dir.display(), total, "restoring all members");
        create_dir(dir)?;
        let mut bytes = 0_u64;
        for (done, info) in members.iter().enumerate() {
            let path = dir.join(&info.name);
            if let Some(parent) = path.parent() {
                create_dir(parent)?;
            }
            let restored = self.member(&info.name)?.restore_file(&path)?;
            bytes = bytes.saturating_add(restored.bytes);
            tracing::info!(
                done = done + 1,
                total,
                member = %info.name,
                bytes = restored.bytes,
                cumulative_bytes = bytes,
                "restored"
            );
        }
        Ok(RestoreAllReport { members: total, bytes })
    }
}

/// Creates `dir` and its missing parents.
///
/// # Errors
/// [`Error::Io`] with op `"create directory"`.
fn create_dir(dir: &Path) -> Result<(), Error> {
    fs::create_dir_all(dir).map_err(|source| Error::Io {
        op: "create directory",
        path: dir.to_path_buf(),
        source,
    })
}

impl Member {
    /// Writes this member to `out`. The bytes appear there only after they hashed to the member's
    /// BLAKE3 and were synced, as one rename, which replaces a file already at `out`. On failure
    /// there is no new file at `out`, what was there is untouched, and the temporary file is gone,
    /// except for [`Error::CommittedNotDurable`]: then the verified bytes are at `out`, but its
    /// directory could not be synced, so a power loss may bring back what was there before.
    /// Nothing else is written. Memory is one more batch than [`Member::read_at`], or two chunks if
    /// that is more, which it is at one thread.
    ///
    /// # Errors
    /// [`Error::OutputIsInput`] if `out` is the container itself, which would lose its other members.
    /// [`Error::Io`] if `out` has no file name or a file operation fails, [`Error::VerifyFailed`] if
    /// the hash differs, [`Error::CommittedNotDurable`] if the directory sync after the rename fails
    /// (Unix only), and any error of [`Member::read_at`].
    pub fn restore_file(&self, out: &Path) -> Result<RestoreReport, Error> {
        let name = self.name()?;
        if let Some(container) = &self.container.shared.path
            && same_file(container, out)?
        {
            return Err(Error::OutputIsInput { path: out.to_path_buf() });
        }
        tracing::info!(member = name, out = %out.display(), len = self.len(), "restoring member");
        let mut temp = TempFile::beside(out)?;
        let (file, path) = temp.parts_mut();
        let hash = self.hash_stream(|bytes| {
            file.write_all(bytes).map_err(|source| Error::Io {
                op: "write",
                path: path.to_path_buf(),
                source,
            })
        })?;
        self.check_hash(hash)?;
        temp.commit(out)?;
        tracing::info!(member = name, out = %out.display(), "restored member");
        Ok(RestoreReport { bytes: self.len(), blake3: *hash.as_bytes() })
    }

    /// Streams this member, in order, into `out`, and checks its BLAKE3 at the end.
    ///
    /// **Unverified bytes reach `out`.** Unlike [`Member::restore_file`], which commits only proven
    /// bytes, this writes each run as soon as it is decoded, before the hash of the whole member is
    /// known. Every chunk is still checked against its own hash before it is written, unless the
    /// container was opened with [`crate::Verify::Off`]. A wrong member hash is reported as
    /// [`Error::VerifyFailed`] only after the last byte was written and `out` was flushed, so on any
    /// `Err` the caller must discard what `out` received. Memory is as for [`Member::restore_file`];
    /// `out` sees writes of up to one stream buffer each.
    ///
    /// # Errors
    /// [`Error::Io`] with op `"write"` or `"flush"` if `out` fails; its `path` is the member name,
    /// since a writer has no path of its own. [`Error::VerifyFailed`] if the hash differs, after all
    /// bytes were written. Any error of [`Member::read_at`], after the bytes before the failed run
    /// were written.
    pub fn decode_to(&self, out: &mut dyn Write) -> Result<RestoreReport, Error> {
        let name = self.name()?;
        tracing::info!(member = name, len = self.len(), "decoding member to a writer");
        let failed = |op: &'static str, source| Error::Io { op, path: PathBuf::from(name), source };
        let hash = self.hash_stream(|bytes| {
            out.write_all(bytes).map_err(|source| failed("write", source))
        })?;
        out.flush().map_err(|source| failed("flush", source))?;
        self.check_hash(hash)?;
        tracing::info!(member = name, "decoded member to a writer");
        Ok(RestoreReport { bytes: self.len(), blake3: *hash.as_bytes() })
    }

    /// Decodes the whole member and compares its BLAKE3 with the stated hash. Writes nothing. Memory
    /// is as for [`Member::restore_file`].
    ///
    /// # Errors
    /// [`Error::VerifyFailed`] if the hash differs, and any error of [`Member::read_at`].
    pub fn verify(&self) -> Result<(), Error> {
        let name = self.name()?;
        tracing::info!(member = name, len = self.len(), "verifying member");
        let hash = self.hash_stream(|_| Ok(()))?;
        self.check_hash(hash)?;
        tracing::info!(member = name, "verified member");
        Ok(())
    }

    /// Compares a hash of the whole content with the member record.
    fn check_hash(&self, actual: blake3::Hash) -> Result<(), Error> {
        let expected = blake3::Hash::from_bytes(self.blake3());
        if actual != expected {
            let member = self.name()?.to_owned();
            return Err(Error::VerifyFailed { member, expected, actual });
        }
        Ok(())
    }

    /// Reads the member from start to end, handing each run of bytes to `sink` in order, and returns
    /// the BLAKE3 of all of them. Stops at the first error.
    fn hash_stream(
        &self,
        mut sink: impl FnMut(&[u8]) -> Result<(), Error>,
    ) -> Result<blake3::Hash, Error> {
        let stream_bytes = self.container.shared.engine.stream_bytes;
        // Lossless: `usize` is at most 64 bits on every supported target.
        let capacity = self.len().min(stream_bytes as u64);
        let mut buffer = vec![0; to_usize(capacity)?];
        let mut hasher = blake3::Hasher::new();
        let mut offset = 0;
        while offset < self.len() {
            let end = offset.saturating_add(capacity).min(self.len());
            let end = self.chunk_aligned_end(offset, end)?;
            let part = &mut buffer[..to_usize(end - offset)?];
            self.read_at(offset, part)?;
            hasher.update(part);
            sink(part)?;
            offset = end;
        }
        Ok(hasher.finalize())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{self, Write};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::super::testkit::{ContainerBuilder, MemorySource, identity, tensor_like, whole};
    use super::super::{Container, OpenOptions};
    use crate::codec::StreamPolicy;
    use crate::error::Error;

    /// A directory under the system temp directory, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            static NEXT: AtomicU32 = AtomicU32::new(0);
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let name = format!("ziptensors-restore-{}-{id}", std::process::id());
            let path = std::env::temp_dir().join(name);
            fs::create_dir(&path).expect("scratch directory");
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            if let Err(error) = fs::remove_dir_all(&self.0) {
                eprintln!("cannot remove {}: {error}", self.0.display());
            }
        }
    }

    fn open(file: Vec<u8>) -> Container {
        let options = OpenOptions { threads: 2, ..OpenOptions::default() };
        Container::from_source(Arc::new(MemorySource::silent(file)), &options).expect("valid")
    }

    /// A container with one member per name, each over a blob of its own of `len` bytes cut into
    /// chunks of 1000, and the contents in the same order.
    fn container(names: &[&str], len: usize) -> (Container, Vec<Vec<u8>>) {
        let mut builder = ContainerBuilder::new();
        let mut contents = Vec::new();
        for (seed, name) in (1..).zip(names) {
            let blob = builder.add_blob(tensor_like(seed, len, 2), whole(), 1000, StreamPolicy::Store);
            let extents = [identity(blob, 0, u64::try_from(len).expect("a small length"))];
            contents.push(builder.add_member(name, &extents));
        }
        (open(builder.finish()), contents)
    }

    /// A writer that accepts `room` bytes and then fails.
    struct Full {
        written: Vec<u8>,
        room: usize,
    }

    impl Write for Full {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let take = buf.len().min(self.room - self.written.len());
            if take == 0 {
                return Err(io::Error::new(io::ErrorKind::StorageFull, "full"));
            }
            self.written.extend_from_slice(&buf[..take]);
            Ok(take)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn restore_all_writes_every_member_under_the_directory() {
        let (container, contents) = container(&["a.bin", "sub/deep/b.bin"], 2500);
        let scratch = Scratch::new();
        let dir = scratch.0.join("out");
        let report = container.restore_all(&dir).expect("restores");
        assert_eq!(report.members, 2);
        assert_eq!(report.bytes, 5000);
        assert_eq!(fs::read(dir.join("a.bin")).expect("restored"), contents[0]);
        assert_eq!(fs::read(dir.join("sub/deep/b.bin")).expect("restored"), contents[1]);
    }

    #[test]
    fn restore_all_with_a_hostile_name_leaves_no_file() {
        let (container, _) = container(&["fine.bin", "../escaped.bin"], 100);
        let scratch = Scratch::new();
        let dir = scratch.0.join("out");
        let error = container.restore_all(&dir).expect_err("hostile");
        let named = matches!(&error, Error::UnsafeMemberName { name } if name == "../escaped.bin");
        assert!(named, "{error:?}");
        let left: Vec<_> = fs::read_dir(&scratch.0).expect("lists").collect();
        assert!(left.is_empty(), "{left:?}");
    }

    #[test]
    fn decode_to_writes_what_restore_file_writes() {
        let (container, contents) = container(&["m.bin"], 7777);
        let member = container.single().expect("one member");
        let mut decoded = Vec::new();
        let report = member.decode_to(&mut decoded).expect("decodes");
        let scratch = Scratch::new();
        let path = scratch.0.join("m.bin");
        let restored = member.restore_file(&path).expect("restores");
        assert_eq!(report, restored);
        assert_eq!(decoded, fs::read(&path).expect("restored"));
        assert_eq!(decoded, contents[0]);
    }

    #[test]
    fn decode_to_reports_a_failing_writer_as_a_write_error() {
        let (container, contents) = container(&["m.bin"], 7777);
        let member = container.single().expect("one member");
        let mut out = Full { written: Vec::new(), room: 3000 };
        let error = member.decode_to(&mut out).expect_err("the writer fails");
        let named = matches!(&error, Error::Io { op: "write", path, .. } if path.as_os_str() == "m.bin");
        assert!(named, "{error:?}");
        assert_eq!(out.written, contents[0][..3000]);
    }

    #[test]
    fn decode_to_reports_a_wrong_hash_after_writing_every_byte() {
        let mut builder = ContainerBuilder::new();
        let content = tensor_like(3, 512, 1);
        let blob = builder.add_blob(content.clone(), whole(), 128, StreamPolicy::Store);
        builder.add_member_records("m", &[identity(blob, 0, 512)], &[], 512, [7; 32]);
        let member = open(builder.finish()).single().expect("one member");
        let mut decoded = Vec::new();
        let error = member.decode_to(&mut decoded).expect_err("wrong hash");
        assert!(matches!(error, Error::VerifyFailed { .. }), "{error:?}");
        assert_eq!(decoded, content, "the bytes reached the writer before the hash was known");
    }

    #[test]
    fn profile_comes_from_the_record() {
        let (container, _) = container(&["m.bin"], 10);
        let member = container.single().expect("one member");
        assert_eq!(member.profile(), container.members()[0].profile);
    }
}
