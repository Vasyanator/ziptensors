//! File: tree/decompress.rs
//! Purpose: `decompress_tree`: every per-file container of a tree is restored to its member beside
//! it, and goes once the restored file is committed and verified.
//!
//! Main responsibilities:
//! - refuse the whole run on any name conflict of the view before anything is written;
//! - restore each container in the crash-safe order: verified commit of the restored file, re-check
//!   of the container, removal, directory sync;
//! - resolve a pair (restore path already taken): a file equal to the member (hash and length) makes
//!   the container redundant; a different file or anything else is kept unless `force`.
//!
//! Key structures: [`DecompressTreeOptions`].
//!
//! Key functions: [`decompress_tree`], and the crate-private `decompress_tree_with` that takes the
//! fault seam.
//!
//! Implementation notes: the mirror of `compress.rs` with the roles swapped. Restore is content
//! only, as `Member::restore_file`: modification time and permissions are not restored. A container
//! that is not a per-file container (a pack, a hostile member name) is reported and never written
//! from. The container handle is dropped before its removal, which Windows requires.

use std::fs;
use std::io::{self, ErrorKind};
use std::path::Path;

use crate::error::Error;
use crate::read::{Container, Member, OpenOptions};

use super::view::{FileStamp, PerFileEntry, ViewEntry, ViewOptions, scan_view};
use super::{
    Action, Fault, FileOutcome, FileStatus, Stage, TreeReport, finish_report, hash_file,
    log_outcome, no_fault, remove_redundant,
};

/// How [`decompress_tree`] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DecompressTreeOptions {
    /// Keep every container beside its restored file. Default `false`: a container is removed once
    /// its member is restored and verified.
    pub keep: bool,
    /// Restore over a different file (or a link or special file) already at the restore path.
    /// Default `false`: such a container fails and nothing is written for it. A directory there is
    /// a conflict of the whole run either way.
    pub force: bool,
    /// Plan only: report what would be done ([`FileStatus::Planned`]) and write nothing. Conflicts
    /// are still an error.
    pub dry_run: bool,
    /// Decode threads per container; 0 means one per logical CPU.
    pub threads: usize,
    /// Budget on the decoded size of each container, in bytes, as
    /// [`OpenOptions::max_total_decoded`]. Default `u64::MAX`, no budget: a caller restoring
    /// untrusted files must set one.
    pub max_total_decoded: u64,
}

impl Default for DecompressTreeOptions {
    fn default() -> Self {
        Self { keep: false, force: false, dry_run: false, threads: 0, max_total_decoded: u64::MAX }
    }
}

/// Restores every per-file container under `root` (a directory, or a single container) to the
/// path its member names, beside it, and removes the container once the restored file is
/// committed and verified, unless `keep`. Containers are recognized by magic; a container with
/// more than one member or a member name that is not one safe file name is reported as
/// [`FileStatus::Foreign`] and left alone, and a file with the magic that does not open fails.
///
/// The view's conflicts (two containers restoring to one name, a container restoring to a
/// directory or to a container file) refuse the whole run before anything is written. Where the
/// restore path is taken by a regular file, it is hashed: equal to the member (BLAKE3 and length),
/// the container is redundant and removed (kept under `keep`); different, the container fails
/// unless `force`, which restores over the file. A failed container does not stop the run, and a
/// rerun is idempotent.
///
/// # Errors
/// [`Error::TreeConflict`] listing every conflict, nothing written. [`Error::Io`] if the root or a
/// directory cannot be walked. Everything that goes wrong with one container is in its outcome.
pub fn decompress_tree(root: &Path, options: &DecompressTreeOptions) -> Result<TreeReport, Error> {
    decompress_tree_with(root, options, no_fault)
}

/// [`decompress_tree`] with the fault seam of the crash-order tests; production passes
/// [`no_fault`] only.
pub(crate) fn decompress_tree_with(
    root: &Path,
    options: &DecompressTreeOptions,
    fault: Fault,
) -> Result<TreeReport, Error> {
    tracing::info!(
        root = %root.display(),
        keep = options.keep,
        force = options.force,
        dry_run = options.dry_run,
        "decompressing tree"
    );
    let view = scan_view(root, &ViewOptions { max_total_decoded: options.max_total_decoded })?;
    if !view.conflicts.is_empty() {
        let conflicts: Vec<String> = view.conflicts.iter().map(super::Conflict::describe).collect();
        tracing::error!(root = %root.display(), count = conflicts.len(), ?conflicts, "tree refused: name conflicts");
        return Err(Error::TreeConflict { root: root.to_path_buf(), conflicts });
    }
    let mut outcomes = Vec::new();
    for entry in view.entries {
        let outcome = match entry {
            ViewEntry::Directory { .. } | ViewEntry::Plain { .. } => continue,
            ViewEntry::PerFile(per_file) => planned_or_run(&per_file, Action::Restore, options, fault),
            ViewEntry::Shadowed(per_file) => {
                planned_or_run(&per_file, Action::CheckPair, options, fault)
            }
            ViewEntry::Foreign { path, reason, .. } => {
                FileOutcome::new(path, FileStatus::Foreign(reason))
            }
            ViewEntry::Broken { path, error, .. } => FileOutcome::new(path, FileStatus::Failed(error)),
            ViewEntry::Skipped { path, reason, .. } => {
                FileOutcome::new(path, FileStatus::Skipped(reason))
            }
        };
        log_outcome(&outcome);
        outcomes.push(outcome);
    }
    let report = TreeReport {
        root: root.to_path_buf(),
        dry_run: options.dry_run,
        files: outcomes,
        warnings: Vec::new(),
    };
    Ok(finish_report(report, "decompress"))
}

/// The outcome of one per-file container: `action` for a dry run, else the run's result.
fn planned_or_run(
    per_file: &PerFileEntry,
    action: Action,
    options: &DecompressTreeOptions,
    fault: Fault,
) -> FileOutcome {
    let mut outcome = FileOutcome::new(per_file.container.clone(), FileStatus::Planned(action));
    outcome.counterpart = Some(per_file.presented.clone());
    outcome.container_len = Some(per_file.stamp.len);
    if options.dry_run {
        return outcome;
    }
    outcome.status = match restore_one(per_file, options, fault, &mut outcome) {
        Ok(status) => status,
        Err(error) => FileStatus::Failed(error),
    };
    outcome
}

/// Restores one container, or resolves it against the file already at its restore path, which is
/// looked at again here, since the view may be stale.
///
/// # Errors
/// Any error of opening the container or of [`Member::restore_file`] (nothing is removed then),
/// [`Error::InputChanged`] if the container changed since the view or before its removal,
/// [`Error::Io`] with kind `AlreadyExists` if something different is at the restore path and
/// `force` is off, and the errors of [`remove_redundant`].
fn restore_one(
    per_file: &PerFileEntry,
    options: &DecompressTreeOptions,
    fault: Fault,
    outcome: &mut FileOutcome,
) -> Result<FileStatus, Error> {
    let before = FileStamp::of(&per_file.container)?;
    let open = OpenOptions {
        threads: options.threads,
        max_total_decoded: options.max_total_decoded,
        ..OpenOptions::default()
    };
    let container = Container::open(&per_file.container, &open)?;
    let member = container.single()?;
    if member.name()? != per_file.member_name {
        tracing::error!(container = %per_file.container.display(), "the container changed since the tree was scanned");
        return Err(Error::InputChanged { path: per_file.container.clone() });
    }
    let presented = &per_file.presented;
    let taken = |what: &str| Error::Io {
        op: "restore",
        path: presented.clone(),
        source: io::Error::new(
            ErrorKind::AlreadyExists,
            format!("{what} is already there; the force option replaces it"),
        ),
    };
    let restore_over = match fs::symlink_metadata(presented) {
        Err(error) if error.kind() == ErrorKind::NotFound => false,
        Err(source) => return Err(Error::Io { op: "metadata", path: presented.clone(), source }),
        Ok(meta) if meta.is_dir() => return Err(taken("a directory")),
        Ok(meta) if meta.is_file() => {
            let (len, hash) = hash_file(presented)?;
            outcome.original_len = Some(len);
            if len == member.len() && hash == member.blake3() {
                drop(member);
                drop(container);
                return finish_pair(per_file, &before, options.keep, fault, outcome);
            }
            if !options.force {
                return Err(taken("a different file"));
            }
            true
        }
        Ok(_) if options.force => true,
        Ok(_) => return Err(taken("a link or special file")),
    };
    if restore_over {
        tracing::warn!(path = %presented.display(), "force: restoring over a different file");
    }
    restore(&member, presented, outcome)?;
    drop(member);
    drop(container);
    fault(Stage::Committed, presented)?;
    if options.keep {
        return Ok(FileStatus::Restored);
    }
    remove_container(per_file, &before, fault, outcome)?;
    Ok(FileStatus::Restored)
}

/// Restores `member` to `path` and records its length.
///
/// # Errors
/// The errors of [`Member::restore_file`].
fn restore(member: &Member, path: &Path, outcome: &mut FileOutcome) -> Result<(), Error> {
    let report = member.restore_file(path)?;
    outcome.original_len = Some(report.bytes);
    Ok(())
}

/// A pair whose plain file equals the member: the container is redundant.
///
/// # Errors
/// As [`remove_container`].
fn finish_pair(
    per_file: &PerFileEntry,
    before: &FileStamp,
    keep: bool,
    fault: Fault,
    outcome: &mut FileOutcome,
) -> Result<FileStatus, Error> {
    if keep {
        return Ok(FileStatus::PairVerified);
    }
    remove_container(per_file, before, fault, outcome)?;
    Ok(FileStatus::PairResolved)
}

/// Removes the container if it is still as `before` described it.
///
/// # Errors
/// [`Error::InputChanged`] naming the container if it changed, which keeps it, and the errors of
/// [`remove_redundant`].
fn remove_container(
    per_file: &PerFileEntry,
    before: &FileStamp,
    fault: Fault,
    outcome: &mut FileOutcome,
) -> Result<(), Error> {
    if FileStamp::of(&per_file.container)? != *before {
        tracing::error!(container = %per_file.container.display(), "the container changed after it was restored; it is kept");
        return Err(Error::InputChanged { path: per_file.container.clone() });
    }
    remove_redundant(&per_file.container, outcome, fault)
}
