//! File: edit.rs
//! Purpose: the Python surface of editing and recovery: `edit`, `compact`, `recover`, and the
//! operation classes `AddOp`, `RemoveOp`, `ReplaceOp`.
//!
//! Main responsibilities: turn the operation objects and keyword arguments into the core's
//! `EditOp`, `EditOptions` and `RecoverOptions`, then run the whole call with the GIL released.
//! Every decision (the operation set, strategy combinations, the commit) is the core's.
//!
//! Key structures: [`AddOp`], [`RemoveOp`], [`ReplaceOp`].
//!
//! Key functions: [`edit`], [`compact`], [`recover`].
//!
//! Implementation notes: operations are small frozen classes rather than tuples, so a type checker
//! sees each field and a misspelt keyword fails at construction. An `AddOp` without a name takes
//! the file's name, the rule of `pack`, resolved here before the core sees it.

use std::path::PathBuf;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use ziptensors_core::{EditOp, EditOptions, OpenOptions, RecoverOptions, ScanBudget, Strategy};

use crate::errors::raised;
use crate::report::{EditReport, RecoverReport};
use crate::writer::{file_name, parse_mode};

/// Adds the file at `path` as a new member at the end of the member list. `name` defaults to the
/// file's name. `delta_base` names a member of the result (a kept member, or a file of the same
/// edit that comes before this one) to store this file against as delta blobs where smaller.
#[pyclass(frozen, get_all, module = "ziptensors")]
#[derive(Debug)]
pub(crate) struct AddOp {
    /// The input file.
    path: PathBuf,
    /// The new member's name, or None for the file's name.
    name: Option<String>,
    /// A member to store this file against, or None.
    delta_base: Option<String>,
}

#[pymethods]
impl AddOp {
    /// An add of `path` as `name` (default: the file's name), optionally against `delta_base`.
    #[new]
    #[pyo3(signature = (path, name=None, *, delta_base=None))]
    fn new(path: PathBuf, name: Option<String>, delta_base: Option<String>) -> Self {
        Self { path, name, delta_base }
    }
}

/// Removes the member `name`.
#[pyclass(frozen, get_all, module = "ziptensors")]
#[derive(Debug)]
pub(crate) struct RemoveOp {
    /// A member of the container.
    name: String,
}

#[pymethods]
impl RemoveOp {
    /// A removal of the member `name`.
    #[new]
    fn new(name: String) -> Self {
        Self { name }
    }
}

/// Replaces the content of member `name` with the file at `path`, keeping the member's slot.
/// `delta_base` is as for `AddOp`.
#[pyclass(frozen, get_all, module = "ziptensors")]
#[derive(Debug)]
pub(crate) struct ReplaceOp {
    /// A member of the container.
    name: String,
    /// The input file.
    path: PathBuf,
    /// A member to store this file against, or None.
    delta_base: Option<String>,
}

#[pymethods]
impl ReplaceOp {
    /// A replacement of member `name` by the file at `path`, optionally against `delta_base`.
    #[new]
    #[pyo3(signature = (name, path, *, delta_base=None))]
    fn new(name: String, path: PathBuf, delta_base: Option<String>) -> Self {
        Self { name, path, delta_base }
    }
}

/// The core operation of one Python operation object.
///
/// # Errors
/// `TypeError` if `op` is none of the three classes; `InputError` if an `AddOp` without a name has
/// a path with no UTF-8 file name.
fn core_op(op: &Bound<'_, PyAny>) -> PyResult<EditOp> {
    if let Ok(add) = op.cast::<AddOp>() {
        let add = add.get();
        let name = match &add.name {
            Some(name) => name.clone(),
            None => raised(op.py(), file_name(&add.path))?.to_owned(),
        };
        return Ok(EditOp::Add { path: add.path.clone(), name, delta_base: add.delta_base.clone() });
    }
    if let Ok(remove) = op.cast::<RemoveOp>() {
        return Ok(EditOp::Remove { name: remove.get().name.clone() });
    }
    if let Ok(replace) = op.cast::<ReplaceOp>() {
        let replace = replace.get();
        return Ok(EditOp::Replace {
            name: replace.name.clone(),
            path: replace.path.clone(),
            delta_base: replace.delta_base.clone(),
        });
    }
    let kind = op.get_type().name()?;
    Err(PyTypeError::new_err(format!(
        "an edit operation is an AddOp, RemoveOp or ReplaceOp, got {kind}"
    )))
}

/// Maps the `strategy` argument. Raises `ValueError` for anything but the three strategies.
fn parse_strategy(strategy: &str) -> PyResult<Strategy> {
    match strategy {
        "rewrite" => Ok(Strategy::Rewrite),
        "append" => Ok(Strategy::Append),
        "in_place" => Ok(Strategy::InPlace),
        other => Err(PyValueError::new_err(format!(
            "unknown strategy {other:?}, expected \"rewrite\", \"append\" or \"in_place\""
        ))),
    }
}

/// The core's edit options from the keyword arguments of `edit` and `compact`. `threads` serves
/// both the encode and the source's decode; None means one per logical CPU, and
/// `max_total_decoded` None means no budget on the source.
fn edit_options(
    strategy: &str,
    repack: bool,
    mode: &str,
    threads: Option<usize>,
    max_total_decoded: Option<u64>,
) -> PyResult<EditOptions> {
    let defaults = EditOptions::default();
    let threads = threads.unwrap_or(defaults.threads);
    Ok(EditOptions {
        strategy: parse_strategy(strategy)?,
        mode: parse_mode(mode)?,
        repack,
        threads,
        open: OpenOptions {
            threads,
            max_total_decoded: max_total_decoded.unwrap_or(defaults.open.max_total_decoded),
            ..defaults.open
        },
    })
}

/// Edits the container at `container` by `ops`, a sequence of `AddOp`, `RemoveOp` and `ReplaceOp`
/// in which each member name appears once. Kept members keep their order, a replacement its slot,
/// and adds follow in order. Every check runs before a byte is written, and a refused request
/// (`InputError`, `NotFoundError`) leaves the file untouched.
///
/// `strategy`: `"rewrite"` (default) writes a proven copy beside the file and renames it over;
/// `"append"` writes only the new data and a new index after the end, leaving the old generation
/// as dead bytes (no `repack`, no empty `ops`); `"in_place"` rewrites the file in itself with no
/// extra space and is RISKY: an interruption after its first write leaves a damaged file
/// (`CorruptionError`), so keep a backup. `repack` (rewrite only) re-encodes every kept member
/// with `mode`. Close any `Container` open on the file first, except under `append`.
/// `TrailingBytes` failures raise `ZiptensorsIoError`; `recover` repairs the file then.
#[pyfunction]
#[pyo3(signature = (
    container,
    ops,
    *,
    strategy="rewrite",
    repack=false,
    mode="max",
    threads=None,
    max_total_decoded=None,
))]
#[expect(clippy::too_many_arguments, reason = "one per keyword argument of the Python function")]
pub(crate) fn edit(
    py: Python<'_>,
    container: PathBuf,
    ops: Vec<Bound<'_, PyAny>>,
    strategy: &str,
    repack: bool,
    mode: &str,
    threads: Option<usize>,
    max_total_decoded: Option<u64>,
) -> PyResult<EditReport> {
    let options = edit_options(strategy, repack, mode, threads, max_total_decoded)?;
    let ops = ops.into_iter().map(|op| core_op(&op)).collect::<PyResult<Vec<_>>>()?;
    let run = move || ziptensors_core::edit(&container, &ops, &options);
    Ok(raised(py, py.detach(run))?.into())
}

/// Compacts the container at `container`: an edit with no operation, which drops the dead bytes of
/// earlier appends and failed adds. `strategy` is `"rewrite"` or the RISKY `"in_place"` (see
/// `edit`); `"append"` cannot compact and raises `InputError`. `repack` (rewrite only) re-encodes
/// every member with `mode`.
#[pyfunction]
#[pyo3(signature = (
    container,
    *,
    strategy="rewrite",
    repack=false,
    mode="max",
    threads=None,
    max_total_decoded=None,
))]
pub(crate) fn compact(
    py: Python<'_>,
    container: PathBuf,
    strategy: &str,
    repack: bool,
    mode: &str,
    threads: Option<usize>,
    max_total_decoded: Option<u64>,
) -> PyResult<EditReport> {
    let options = edit_options(strategy, repack, mode, threads, max_total_decoded)?;
    let run = move || ziptensors_core::edit(&container, &[], &options);
    Ok(raised(py, py.detach(run))?.into())
}

// `recover(max_candidates=64)` spells the core default as a literal, since pyo3 renders a
// non-literal default as `...` in the Python signature. This ties the literal to the core.
const _: () = assert!(
    ziptensors_core::recover::DEFAULT_MAX_CANDIDATES == 64,
    "the default of max_candidates in the signature of `recover` must be the core's"
);

/// Shortens the container at `path` to its latest valid generation (spec section 9.3), after an
/// interrupted append or in-place edit. Never automatic: a reader refuses such a file until this
/// runs. A file that opens as it is stays untouched. `verify` (default) decodes every member of a
/// candidate and passes over one that fails; `dry_run` reports and writes nothing. The backward
/// scan checks at most `max_candidates` footers and reads at most `max_scan_bytes` (None: no
/// limit). `threads` are the verification's, None for one per logical CPU. Raises `FormatError`
/// when no valid generation is found.
#[pyfunction]
#[pyo3(signature = (
    path,
    *,
    dry_run=false,
    verify=true,
    // A literal, so that `inspect.signature` shows it; the assertion above ties it to the core.
    max_candidates=64,
    max_scan_bytes=None,
    threads=None,
))]
pub(crate) fn recover(
    py: Python<'_>,
    path: PathBuf,
    dry_run: bool,
    verify: bool,
    max_candidates: u32,
    max_scan_bytes: Option<u64>,
    threads: Option<usize>,
) -> PyResult<RecoverReport> {
    let defaults = RecoverOptions::default();
    let options = RecoverOptions {
        open: OpenOptions { threads: threads.unwrap_or(defaults.open.threads), ..defaults.open },
        budget: ScanBudget {
            max_candidates,
            max_scan_bytes: max_scan_bytes.unwrap_or(defaults.budget.max_scan_bytes),
        },
        verify,
        dry_run,
    };
    let run = move || ziptensors_core::recover(&path, &options);
    Ok(raised(py, py.detach(run))?.into())
}
