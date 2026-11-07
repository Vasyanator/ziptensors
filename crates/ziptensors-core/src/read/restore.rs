//! File: read/restore.rs
//! Purpose: reading a member from end to end: `restore_file` and `verify`.
//!
//! Main responsibilities: hash the whole member while streaming it, and commit a restored file by the
//! protocol of spec invariant 12, through `io::TempFile`.
//!
//! Key structures: [`RestoreReport`].
//!
//! Implementation notes: only extents and the member hash are involved, never tensor ranges. The
//! stream buffer is the destination of its reads, so memory is one batch more than a `read_at`, and
//! at least two chunks, as one widened chunk spans two chunks of the member: a shorter buffer could not
//! end on a chunk boundary, and every chunk would decode twice.
//! Modification time and permissions are not restored, by spec section 2.

use std::io::Write;
use std::path::Path;

use crate::error::Error;
use crate::io::{TempFile, same_file};

use super::batch::to_usize;
use super::member::Member;

/// What [`Member::restore_file`] produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestoreReport {
    /// Bytes written.
    pub bytes: u64,
    /// BLAKE3 of those bytes, equal to the hash of the member.
    pub blake3: [u8; 32],
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
