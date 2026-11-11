//! File: tree/mod.rs
//! Purpose: per-file containers over a directory tree: every weight file becomes a container of
//! one member beside it (`compress_tree`), every such container becomes its file again
//! (`decompress_tree`), and a read-only view that tells the two apart (`scan_view`).
//!
//! Main responsibilities: the module list and public surface; the report types both directions
//! share ([`TreeReport`], [`FileOutcome`], [`FileStatus`], [`Action`], [`TreeTotals`],
//! [`TreeWarning`]); the steps both directions take on the file system after a verified commit
//! (hash a file, remove the copy that became redundant and sync its directory); the crate-private
//! fault seam the crash-order tests use.
//!
//! Key structures: [`TreeReport`], [`FileOutcome`], [`FileStatus`], [`Action`].
//!
//! Key functions: [`compress_tree`], [`decompress_tree`], [`scan_view`].
//!
//! Implementation notes: nothing here owns format logic. A container is written by
//! `write::encode_file` or a `PackBuilder` and restored by `Member::restore_file`, both of which
//! verify before their rename; this module only decides which file goes where and when the other
//! copy may go. The rule that keeps user data safe: a copy is removed only while a verified twin is
//! committed beside it and the copy is unchanged since it was proven equal to that twin.

mod compress;
mod decompress;
mod view;

#[cfg(test)]
mod tests;

use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use crate::error::Error;

pub use compress::{CompressTreeOptions, compress_tree};
pub use decompress::{DecompressTreeOptions, decompress_tree};
pub use view::{
    Conflict, ConflictReason, FileStamp, ForeignReason, PerFileEntry, SkipReason, TreeView,
    ViewEntry, ViewOptions, scan_view,
};

/// Extension of a per-file container: `model.safetensors` is compressed to `model.ziptensors`.
/// A naming convention only: containers are recognized by magic, and the member name inside is
/// what a restore writes.
pub const CONTAINER_EXTENSION: &str = "ziptensors";

/// What a run decided to do with one file, as a dry run reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Action {
    /// Compress a weight file that has no container yet.
    Compress,
    /// Both copies exist: hash the uncompressed one against the container's stated BLAKE3, then
    /// verify the container or replace it, and remove the redundant copy unless `keep`.
    CheckPair,
    /// Restore a container's member beside it.
    Restore,
}

/// What happened to one path in a tree run.
#[derive(Debug)]
#[non_exhaustive]
pub enum FileStatus {
    /// A new container was committed and verified; the original was removed unless `keep`.
    Compressed,
    /// The container beside the original stated other content, so it was replaced by a new one;
    /// the original was removed unless `keep`.
    Recompressed,
    /// Both copies existed and were proven equal (the uncompressed file hashes to the member's
    /// BLAKE3, and in compress the container also verified); the redundant copy was removed: the
    /// original in compress, the container in decompress.
    PairResolved,
    /// As [`FileStatus::PairResolved`], but both copies were kept, as `keep` asks.
    PairVerified,
    /// Compress found a per-file container whose original is gone, and left it alone.
    AlreadyCompressed,
    /// Decompress restored and verified the member; the container was removed unless `keep`.
    Restored,
    /// Compress: not a weight file by `profile::classify_weight`, left untouched.
    NotWeight,
    /// A container that is not a per-file container, left untouched.
    Foreign(ForeignReason),
    /// Left out without being read: a link, a special file, a temporary or partial file.
    Skipped(SkipReason),
    /// Dry run: what the run would do. Nothing was written.
    Planned(Action),
    /// The file failed; the walk went on with the next one. Nothing was removed unless
    /// [`FileOutcome::removed`] says so.
    Failed(Error),
}

/// One line of a [`TreeReport`].
#[derive(Debug)]
pub struct FileOutcome {
    /// The path the outcome is about: in compress the original file or link, or a container found
    /// alone; in decompress the container; or the path that was skipped.
    pub path: PathBuf,
    /// The other copy: in compress the container of `path`, in decompress the file it restores to.
    pub counterpart: Option<PathBuf>,
    /// What happened.
    pub status: FileStatus,
    /// Length of the uncompressed file in bytes, where it was looked at.
    pub original_len: Option<u64>,
    /// Length of the container file in bytes, where it was looked at.
    pub container_len: Option<u64>,
    /// The path removed because its twin was proven, if one was. Set also when a later step
    /// failed: the directory sync after a removal fails as [`FileStatus::Failed`] with this set,
    /// and a power loss may then bring the removed copy back, which a rerun resolves.
    pub removed: Option<PathBuf>,
}

impl FileOutcome {
    /// An outcome about `path` with nothing else known yet.
    fn new(path: PathBuf, status: FileStatus) -> Self {
        Self {
            path,
            counterpart: None,
            status,
            original_len: None,
            container_len: None,
            removed: None,
        }
    }
}

/// Something a caller should tell the user about a whole run.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TreeWarning {
    /// `count` symbolic links were skipped, since links are not followed by default. Its text names
    /// the option that follows them and why that breaks a Hugging Face cache.
    SymlinksSkipped {
        /// Links left out.
        count: usize,
    },
}

impl fmt::Display for TreeWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SymlinksSkipped { count } => write!(
                f,
                "{count} symbolic links were skipped. Following them (the follow_symlinks option, \
                 `--follow-symlinks` on the command line) compresses each link's target into a \
                 container in place of the link. In a Hugging Face cache, where snapshots/ links to \
                 blobs/, that breaks the cache structure and keeps the blob as well; compress a copy \
                 made with `snapshot_download(local_dir=...)` instead"
            ),
        }
    }
}

/// Counts of a [`TreeReport`] by status. A count of a status the run's direction never produces
/// is 0: compress never restores, and decompress, which reports containers (and skipped paths)
/// alone, never compresses or classifies a file as [`FileStatus::NotWeight`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct TreeTotals {
    /// [`FileStatus::Compressed`].
    pub compressed: usize,
    /// [`FileStatus::Recompressed`].
    pub recompressed: usize,
    /// [`FileStatus::PairResolved`].
    pub pair_resolved: usize,
    /// [`FileStatus::PairVerified`].
    pub pair_verified: usize,
    /// [`FileStatus::AlreadyCompressed`].
    pub already_compressed: usize,
    /// [`FileStatus::Restored`].
    pub restored: usize,
    /// [`FileStatus::NotWeight`].
    pub not_weight: usize,
    /// [`FileStatus::Foreign`].
    pub foreign: usize,
    /// [`FileStatus::Skipped`].
    pub skipped: usize,
    /// [`FileStatus::Planned`].
    pub planned: usize,
    /// [`FileStatus::Failed`].
    pub failed: usize,
    /// Sum of [`FileOutcome::original_len`], in bytes, over the files whose two copies the run
    /// proved equal: a container written and verified (compressed, recompressed), a member
    /// restored and verified (restored), or a pair hashed (pair resolved, pair verified). Either
    /// direction: in compress these are the bytes that went into containers, in decompress the
    /// bytes that came out of them.
    pub original_bytes: u64,
    /// Sum of [`FileOutcome::container_len`] over the same files, in bytes: the containers written
    /// in compress, the containers read in decompress.
    pub container_bytes: u64,
}

/// What a tree run did, one [`FileOutcome`] per path it looked at, sorted by path.
#[derive(Debug)]
pub struct TreeReport {
    /// The root that was given.
    pub root: PathBuf,
    /// Whether this was a dry run, which wrote nothing.
    pub dry_run: bool,
    /// One outcome per path.
    pub files: Vec<FileOutcome>,
    /// Notes about the whole run.
    pub warnings: Vec<TreeWarning>,
}

impl TreeReport {
    /// Counts the outcomes by status and sums the lengths.
    #[must_use]
    pub fn totals(&self) -> TreeTotals {
        let mut totals = TreeTotals::default();
        for outcome in &self.files {
            // Only a file whose two copies were proven equal counts towards the byte sums.
            let sized = match &outcome.status {
                FileStatus::Compressed => bump(&mut totals.compressed, true),
                FileStatus::Recompressed => bump(&mut totals.recompressed, true),
                FileStatus::PairResolved => bump(&mut totals.pair_resolved, true),
                FileStatus::PairVerified => bump(&mut totals.pair_verified, true),
                FileStatus::AlreadyCompressed => bump(&mut totals.already_compressed, false),
                FileStatus::Restored => bump(&mut totals.restored, true),
                FileStatus::NotWeight => bump(&mut totals.not_weight, false),
                FileStatus::Foreign(_) => bump(&mut totals.foreign, false),
                FileStatus::Skipped(_) => bump(&mut totals.skipped, false),
                FileStatus::Planned(_) => bump(&mut totals.planned, false),
                FileStatus::Failed(_) => bump(&mut totals.failed, false),
            };
            if sized && let (Some(original), Some(container)) =
                (outcome.original_len, outcome.container_len)
            {
                totals.original_bytes = totals.original_bytes.saturating_add(original);
                totals.container_bytes = totals.container_bytes.saturating_add(container);
            }
        }
        totals
    }

    /// Whether any file failed. A frontend exits with a failure then.
    #[must_use]
    pub fn has_failures(&self) -> bool {
        self.files.iter().any(|outcome| matches!(outcome.status, FileStatus::Failed(_)))
    }
}

/// Adds one to `count` and returns `sized`, for [`TreeReport::totals`].
fn bump(count: &mut usize, sized: bool) -> bool {
    *count += 1;
    sized
}

/// A point of a run that the crash-order tests interrupt: after a verified commit, and after the
/// removal of the redundant copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stage {
    /// The new file (container or restored file) is committed, nothing is removed yet.
    Committed,
    /// The redundant copy is removed, its directory not yet synced.
    Removed,
}

/// The test seam of a run: called at each [`Stage`] with the path just written or removed. An
/// `Err` fails that file at that point, as a crash there would leave it, and the run goes on with
/// the next file. Production passes [`no_fault`] only.
pub(crate) type Fault = fn(Stage, &Path) -> Result<(), Error>;

/// The [`Fault`] of production: never interrupts.
#[expect(
    clippy::unnecessary_wraps,
    reason = "the signature is the `Fault` seam, whose test implementations do fail"
)]
pub(crate) fn no_fault(_stage: Stage, _path: &Path) -> Result<(), Error> {
    Ok(())
}

/// Bytes read per call when a whole file is hashed.
const HASH_BUFFER: usize = 1 << 20;

/// Reads the file at `path` to its end and returns its length and BLAKE3.
///
/// # Errors
/// [`Error::Io`] with op `"open"` or `"read"`.
fn hash_file(path: &Path) -> Result<(u64, [u8; 32]), Error> {
    let failed = |op, source| Error::Io { op, path: path.to_path_buf(), source };
    let mut file = File::open(path).map_err(|source| failed("open", source))?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0; HASH_BUFFER];
    let mut len = 0_u64;
    loop {
        let read = match file.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(source) => return Err(failed("read", source)),
        };
        hasher.update(&buffer[..read]);
        // Lossless: a read is at most the buffer, far below `u64::MAX`.
        len = len.saturating_add(read as u64);
    }
    Ok((len, *hasher.finalize().as_bytes()))
}

/// Removes `path`, the copy a verified twin made redundant, then syncs its directory so that the
/// removal survives a power loss (Unix; Windows has no directory to sync), calling `fault` between
/// the two. Records the removal in `outcome` as soon as it happened.
///
/// # Errors
/// [`Error::Io`] with op `"remove"` if nothing was removed, or with op `"sync directory"` (or the
/// fault's error) after the removal, with `outcome.removed` set: a power loss may then bring the
/// copy back, which a rerun resolves as a pair.
fn remove_redundant(path: &Path, outcome: &mut FileOutcome, fault: Fault) -> Result<(), Error> {
    fs::remove_file(path).map_err(|source| Error::Io {
        op: "remove",
        path: path.to_path_buf(),
        source,
    })?;
    tracing::info!(path = %path.display(), "removed the copy its verified twin replaces");
    outcome.removed = Some(path.to_path_buf());
    fault(Stage::Removed, path)?;
    sync_directory_of(path).map_err(|source| {
        tracing::error!(
            path = %path.display(),
            %source,
            "removed, but the directory cannot be synced: a power loss may bring the file back"
        );
        Error::Io { op: "sync directory", path: path.to_path_buf(), source }
    })
}

/// Syncs the directory holding `path`, so that a removal in it survives a power loss. A file
/// system with no such operation refuses it as invalid or unsupported, which is no failure.
///
/// The same policy as the private post-rename step of `io::TempFile::commit`; it is repeated here
/// because that one is private to `io`.
#[cfg(unix)]
fn sync_directory_of(path: &Path) -> io::Result<()> {
    let directory = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        Some(_) | None => Path::new("."),
    };
    match File::open(directory)?.sync_all() {
        Ok(()) => Ok(()),
        Err(error)
            if matches!(error.kind(), io::ErrorKind::InvalidInput | io::ErrorKind::Unsupported) =>
        {
            tracing::debug!(path = %directory.display(), %error, "directory cannot be synced here");
            Ok(())
        }
        Err(source) => Err(source),
    }
}

/// Windows has no directory handle to sync, and its metadata changes are journaled by the file
/// system, so there is nothing to do and nothing can fail.
#[cfg(not(unix))]
#[expect(
    clippy::unnecessary_wraps,
    reason = "the signature is the Unix one, which can fail"
)]
fn sync_directory_of(_path: &Path) -> io::Result<()> {
    Ok(())
}

/// Logs one outcome: `info` for a success or skip, `error` for a failure.
fn log_outcome(outcome: &FileOutcome) {
    match &outcome.status {
        FileStatus::Failed(error) => tracing::error!(
            path = %outcome.path.display(),
            counterpart = ?outcome.counterpart,
            removed = ?outcome.removed,
            %error,
            "tree: file failed"
        ),
        status => tracing::info!(
            path = %outcome.path.display(),
            counterpart = ?outcome.counterpart,
            ?status,
            "tree: file done"
        ),
    }
}

/// Sorts the outcomes by path and logs a summary line.
fn finish_report(mut report: TreeReport, direction: &'static str) -> TreeReport {
    report.files.sort_by(|a, b| a.path.cmp(&b.path));
    let totals = report.totals();
    tracing::info!(
        direction,
        root = %report.root.display(),
        dry_run = report.dry_run,
        files = report.files.len(),
        failed = totals.failed,
        ?totals,
        "tree run finished"
    );
    report
}
