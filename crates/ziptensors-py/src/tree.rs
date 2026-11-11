//! File: tree.rs
//! Purpose: the Python surface of per-file tree compression: `compress_tree`, `decompress_tree`,
//! and their report classes `TreeReport`, `FileOutcome`, `TreeTotals`.
//!
//! Main responsibilities: translate keyword arguments into the core's tree options, run the whole
//! walk with the GIL released, and turn the core's `TreeReport` into Python objects: each status as
//! a string with an optional reason, each per-file failure as the exception object the module would
//! have raised for it.
//!
//! Key structures: [`TreeReport`], [`FileOutcome`], [`TreeTotals`].
//!
//! Key functions: [`compress_tree`], [`decompress_tree`].
//!
//! Implementation notes: the core's status enums are `#[non_exhaustive]`, so their matches need a
//! wildcard arm. It maps a status this build does not name to `"unknown"`, with the core's debug
//! text as the reason, rather than fail a run that has already written files.

use std::path::PathBuf;

use pyo3::prelude::*;
use ziptensors_core::tree::{Action, ForeignReason, SkipReason};
use ziptensors_core::{CompressTreeOptions, DecompressTreeOptions, EncodeOptions, FileStatus};
use ziptensors_core::{FileOutcome as CoreFileOutcome, TreeReport as CoreTreeReport};

use crate::errors::{raised, to_py_err};
use crate::writer::parse_mode;

/// What happened to one path of a tree run.
#[pyclass(frozen, module = "ziptensors")]
#[derive(Debug)]
pub(crate) struct FileOutcome {
    /// In compress the original file or link, or a container found alone; in decompress the
    /// container; or the path that was skipped.
    #[pyo3(get)]
    path: PathBuf,
    /// The other copy: in compress the container of `path`, in decompress the restored file.
    #[pyo3(get)]
    counterpart: Option<PathBuf>,
    /// `compressed`, `recompressed`, `pair_resolved`, `pair_verified`, `already_compressed`,
    /// `restored`, `not_weight`, `foreign`, `skipped`, `planned` or `failed`.
    #[pyo3(get)]
    status: &'static str,
    /// Why, for `foreign` (`member_count`, `not_one_safe_name`), `skipped` (`symlink`,
    /// `symlink_to_directory`, `dangling_symlink`, `special`, `stale_temp`, `incomplete`,
    /// `non_utf8`) and `planned` (`compress`, `check_pair`, `restore`); the message for `failed`.
    #[pyo3(get)]
    reason: Option<String>,
    /// For `failed`, the exception the module raises for that error, with its `code`.
    error: Option<Py<PyAny>>,
    /// Length of the uncompressed file in bytes, where it was looked at.
    #[pyo3(get)]
    original_len: Option<u64>,
    /// Length of the container in bytes, where it was looked at.
    #[pyo3(get)]
    container_len: Option<u64>,
    /// The copy removed because its twin was proven, if any; set also when a later step failed.
    #[pyo3(get)]
    removed: Option<PathBuf>,
}

#[pymethods]
impl FileOutcome {
    /// For `failed`, the exception the module raises for that error (a `ZiptensorsError` with
    /// `code`), not raised; otherwise None.
    #[getter]
    fn error(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.error.as_ref().map(|error| error.clone_ref(py))
    }
}

/// The name of a skip reason.
fn skip_name(reason: SkipReason) -> String {
    let name = match reason {
        SkipReason::Symlink => "symlink",
        SkipReason::SymlinkToDirectory => "symlink_to_directory",
        SkipReason::DanglingSymlink => "dangling_symlink",
        SkipReason::Special => "special",
        SkipReason::StaleTemp => "stale_temp",
        SkipReason::Incomplete => "incomplete",
        SkipReason::NonUtf8 => "non_utf8",
        // `SkipReason` is `#[non_exhaustive]`: see the file header.
        other => return format!("{other:?}"),
    };
    name.to_owned()
}

/// The name of a foreign reason.
fn foreign_name(reason: &ForeignReason) -> String {
    let name = match reason {
        ForeignReason::MemberCount(_) => "member_count",
        ForeignReason::NotOneSafeName(_) => "not_one_safe_name",
        // `ForeignReason` is `#[non_exhaustive]`: see the file header.
        other => return format!("{other:?}"),
    };
    name.to_owned()
}

/// The name of a planned action.
fn action_name(action: Action) -> String {
    let name = match action {
        Action::Compress => "compress",
        Action::CheckPair => "check_pair",
        Action::Restore => "restore",
        // `Action` is `#[non_exhaustive]`: see the file header.
        other => return format!("{other:?}"),
    };
    name.to_owned()
}

impl FileOutcome {
    /// The Python form of `outcome`; a failure becomes its exception object.
    fn new(py: Python<'_>, outcome: CoreFileOutcome) -> Self {
        let mut error = None;
        let (status, reason) = match &outcome.status {
            FileStatus::Compressed => ("compressed", None),
            FileStatus::Recompressed => ("recompressed", None),
            FileStatus::PairResolved => ("pair_resolved", None),
            FileStatus::PairVerified => ("pair_verified", None),
            FileStatus::AlreadyCompressed => ("already_compressed", None),
            FileStatus::Restored => ("restored", None),
            FileStatus::NotWeight => ("not_weight", None),
            FileStatus::Foreign(why) => ("foreign", Some(foreign_name(why))),
            FileStatus::Skipped(why) => ("skipped", Some(skip_name(*why))),
            FileStatus::Planned(action) => ("planned", Some(action_name(*action))),
            FileStatus::Failed(failure) => {
                error = Some(to_py_err(py, failure).into_value(py).into_any());
                ("failed", Some(failure.to_string()))
            }
            // `FileStatus` is `#[non_exhaustive]`: see the file header.
            other => ("unknown", Some(format!("{other:?}"))),
        };
        Self {
            path: outcome.path,
            counterpart: outcome.counterpart,
            status,
            reason,
            error,
            original_len: outcome.original_len,
            container_len: outcome.container_len,
            removed: outcome.removed,
        }
    }
}

/// Counts of a tree run by status, and the byte sums of the files that ended with a container.
#[pyclass(frozen, get_all, module = "ziptensors")]
#[derive(Debug, Clone)]
pub(crate) struct TreeTotals {
    compressed: usize,
    recompressed: usize,
    pair_resolved: usize,
    pair_verified: usize,
    already_compressed: usize,
    restored: usize,
    not_weight: usize,
    foreign: usize,
    skipped: usize,
    planned: usize,
    failed: usize,
    /// Uncompressed bytes of the files whose two copies the run proved equal (compressed,
    /// recompressed, restored, or a pair): in compress what went into containers, in decompress
    /// what came out of them.
    original_bytes: u64,
    /// Container bytes of the same files: written in compress, read in decompress.
    container_bytes: u64,
}

/// What a tree run did: one `FileOutcome` per path it looked at, sorted by path.
#[pyclass(frozen, module = "ziptensors")]
#[derive(Debug)]
pub(crate) struct TreeReport {
    /// The root that was given.
    #[pyo3(get)]
    root: PathBuf,
    /// Whether this was a dry run, which wrote nothing.
    #[pyo3(get)]
    dry_run: bool,
    files: Vec<Py<FileOutcome>>,
    /// Notes about the whole run, as text (for example how many links were skipped, and why
    /// following them breaks a Hugging Face cache).
    #[pyo3(get)]
    warnings: Vec<String>,
    /// Counts by status and byte sums.
    #[pyo3(get)]
    totals: TreeTotals,
    /// Whether any file failed. The run went on past each failure.
    #[pyo3(get)]
    has_failures: bool,
}

#[pymethods]
impl TreeReport {
    /// One outcome per path, sorted by path.
    #[getter]
    fn files(&self, py: Python<'_>) -> Vec<Py<FileOutcome>> {
        self.files.iter().map(|outcome| outcome.clone_ref(py)).collect()
    }
}

impl TreeReport {
    /// The Python form of `report`.
    ///
    /// # Errors
    /// Only a failure of the interpreter to allocate an outcome object.
    fn new(py: Python<'_>, report: CoreTreeReport) -> PyResult<Self> {
        let totals = report.totals();
        let has_failures = report.has_failures();
        let totals = TreeTotals {
            compressed: totals.compressed,
            recompressed: totals.recompressed,
            pair_resolved: totals.pair_resolved,
            pair_verified: totals.pair_verified,
            already_compressed: totals.already_compressed,
            restored: totals.restored,
            not_weight: totals.not_weight,
            foreign: totals.foreign,
            skipped: totals.skipped,
            planned: totals.planned,
            failed: totals.failed,
            original_bytes: totals.original_bytes,
            container_bytes: totals.container_bytes,
        };
        let files = report
            .files
            .into_iter()
            .map(|outcome| Py::new(py, FileOutcome::new(py, outcome)))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            root: report.root,
            dry_run: report.dry_run,
            files,
            warnings: report.warnings.iter().map(ToString::to_string).collect(),
            totals,
            has_failures,
        })
    }
}

/// Compresses every weight file under `path` (a directory, or one file) into a container of one
/// member beside it: `dir/model.safetensors` becomes `dir/model.ziptensors`. Weight files are
/// recognized by content; everything else is left alone. The original is removed once its
/// container is committed and verified, unless `keep`. Symbolic links are skipped with a warning
/// unless `follow_symlinks`. `dry_run` plans and writes nothing. A name collision raises
/// `InputError` before anything is written; a failure of one file is in its outcome and the run
/// goes on, so check `has_failures`. A rerun is idempotent. `max_total_decoded` bounds each
/// container that is opened.
#[pyfunction]
#[pyo3(signature = (
    path,
    *,
    keep=false,
    follow_symlinks=false,
    dry_run=false,
    mode="max",
    threads=None,
    max_total_decoded=None,
))]
#[expect(clippy::too_many_arguments, reason = "one per keyword argument of the Python function")]
pub(crate) fn compress_tree(
    py: Python<'_>,
    path: PathBuf,
    keep: bool,
    follow_symlinks: bool,
    dry_run: bool,
    mode: &str,
    threads: Option<usize>,
    max_total_decoded: Option<u64>,
) -> PyResult<TreeReport> {
    // The options are `#[non_exhaustive]`: start from the defaults and set what Python passed.
    let mut options = CompressTreeOptions::default();
    options.keep = keep;
    options.follow_symlinks = follow_symlinks;
    options.dry_run = dry_run;
    options.encode = EncodeOptions {
        mode: parse_mode(mode)?,
        threads: threads.unwrap_or(options.encode.threads),
    };
    options.max_total_decoded = max_total_decoded.unwrap_or(options.max_total_decoded);
    let report = raised(py, py.detach(move || ziptensors_core::compress_tree(&path, &options)))?;
    TreeReport::new(py, report)
}

/// Restores every per-file container under `path` (a directory, or one container) to the file its
/// member names, beside it, and removes the container once the file is restored and verified,
/// unless `keep`. A different file already at the restore path fails that container unless
/// `force`. Packs and containers whose member is no single safe name are `foreign` and left alone.
/// Conflicts raise `InputError` before anything is written; per-file failures are in the report.
/// `max_total_decoded` bounds each container; restoring untrusted files needs it.
#[pyfunction]
#[pyo3(signature = (
    path,
    *,
    keep=false,
    force=false,
    dry_run=false,
    threads=None,
    max_total_decoded=None,
))]
pub(crate) fn decompress_tree(
    py: Python<'_>,
    path: PathBuf,
    keep: bool,
    force: bool,
    dry_run: bool,
    threads: Option<usize>,
    max_total_decoded: Option<u64>,
) -> PyResult<TreeReport> {
    // The options are `#[non_exhaustive]`: start from the defaults and set what Python passed.
    let mut options = DecompressTreeOptions::default();
    options.keep = keep;
    options.force = force;
    options.dry_run = dry_run;
    options.threads = threads.unwrap_or(options.threads);
    options.max_total_decoded = max_total_decoded.unwrap_or(options.max_total_decoded);
    let report =
        raised(py, py.detach(move || ziptensors_core::decompress_tree(&path, &options)))?;
    TreeReport::new(py, report)
}
