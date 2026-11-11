//! File: writer.rs
//! Purpose: the Python surface of the writer: `encode_file`, `pack`, `collect_inputs` and the
//! `PackBuilder` class.
//!
//! Main responsibilities: translate arguments, then run every encode and every directory walk with
//! the GIL released. `PackBuilder` owns a core builder behind a lock, taken for the length of one
//! call, so it can be shared between threads and finished or dropped exactly once.
//!
//! Key structures: [`PackBuilder`], [`PackInput`].
//!
//! Key functions: [`encode_file`], [`pack`], [`collect_inputs`].
//!
//! Implementation notes: a pack member named by a bare file path takes the file's name, the rule of
//! the core's `encode_file`. A bare path that is a directory is walked by the core's
//! `collect_inputs`, the walk of the command line: symbolic links and `*.incomplete` files inside
//! are skipped and counted, and the output may not lie under it. A bare file path and a named pair
//! are not walked, so a symbolic link there is an error. `pack` names its delta base by member
//! name, the one name every input has here, where the command line names a file.

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ziptensors_core::write::inputs::check_output;
use ziptensors_core::{EncodeOptions, Error, Mode, check_member_names};
use ziptensors_core::PackBuilder as CorePackBuilder;

use crate::errors::raised;
use crate::reader::Container;
use crate::report::{EncodeReport, Inputs, MemberReport};

/// Compile-time proof that a core builder may move between threads, which the lock of
/// [`PackBuilder`] needs to be shared by Python threads and to run its calls detached. If this
/// stops compiling, the class must become `unsendable` and its readme say so.
const _: () = {
    const fn sendable<T: Send>() {}
    sendable::<CorePackBuilder>();
};

/// One input of `pack`: a path, or a path with the name of its member.
#[derive(Debug, FromPyObject)]
pub(crate) enum PackInput {
    /// `(path, member_name)`.
    Named(PathBuf, String),
    /// A path. The member takes the file's name.
    Path(PathBuf),
}

impl PackInput {
    /// Appends the files of this input to `named`, with their member names: a named pair as given, a
    /// bare file under its file name, a bare directory as the core's walk names its files. A
    /// walked directory is also pushed to `walked`, for the output check.
    ///
    /// # Errors
    /// [`Error::UnsafeMemberName`] if a bare file path has no UTF-8 file name, and every error of
    /// `collect_inputs` for a directory.
    fn resolve(
        self,
        named: &mut Vec<(PathBuf, String)>,
        walked: &mut Vec<PathBuf>,
    ) -> Result<(), Error> {
        match self {
            Self::Named(path, name) => named.push((path, name)),
            // A failed `symlink_metadata` is not a directory; `add_file` then reports why the path
            // cannot be read. A link to a directory is not walked, as in the core's walk.
            Self::Path(path) if fs::symlink_metadata(&path).is_ok_and(|meta| meta.is_dir()) => {
                let inputs = ziptensors_core::collect_inputs(std::slice::from_ref(&path))?;
                named.extend(inputs.files);
                walked.push(path);
            }
            Self::Path(path) => {
                let name = file_name(&path)?.to_owned();
                named.push((path, name));
            }
        }
        Ok(())
    }
}

/// The file name of `path`, as a member name.
///
/// # Errors
/// [`Error::UnsafeMemberName`] if `path` has no file name, or one that is not UTF-8.
pub(crate) fn file_name(path: &Path) -> Result<&str, Error> {
    path.file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| Error::UnsafeMemberName { name: path.display().to_string() })
}

/// Maps the `mode` argument. Raises `ValueError` for anything but the three modes.
pub(crate) fn parse_mode(mode: &str) -> PyResult<Mode> {
    match mode {
        "fast" => Ok(Mode::Fast),
        "balanced" => Ok(Mode::Balanced),
        "max" => Ok(Mode::Max),
        other => Err(PyValueError::new_err(format!(
            "unknown mode {other:?}, expected \"fast\", \"balanced\" or \"max\""
        ))),
    }
}

/// Applies the `mode` and `threads` arguments, with `defaults` for the threads.
fn options(defaults: EncodeOptions, mode: &str, threads: Option<usize>) -> PyResult<EncodeOptions> {
    Ok(EncodeOptions { mode: parse_mode(mode)?, threads: threads.unwrap_or(defaults.threads) })
}

/// Encodes the file at `input` as a container of one member at `output`, named after the file, and
/// replaces what is at `output` only after the result decoded back to the source; on failure
/// `output` is untouched, except after `NotDurableError`, when the proven file is there. `mode` is
/// `fast`, `balanced` or `max`; `threads` is None or 0 for one per logical CPU.
#[pyfunction]
#[pyo3(signature = (input, output, mode="balanced", threads=None))]
pub(crate) fn encode_file(
    py: Python<'_>,
    input: PathBuf,
    output: PathBuf,
    mode: &str,
    threads: Option<usize>,
) -> PyResult<EncodeReport> {
    let options = options(EncodeOptions::for_encode(), mode, threads)?;
    let run = move || ziptensors_core::encode_file(&input, &output, &options);
    Ok(raised(py, py.detach(run))?.into())
}

/// Moves the input whose member is called `base` to the front of `named`, the order
/// `PackBuilder::add_file_delta` needs. The other inputs keep their order. Raises `ValueError` if
/// no input is called `base`.
fn base_first(named: &mut [(PathBuf, String)], base: &str) -> PyResult<()> {
    let at = named.iter().position(|(_, name)| name == base).ok_or_else(|| {
        PyValueError::new_err(format!("delta_base {base:?} is the member name of no input"))
    })?;
    named[..=at].rotate_right(1);
    Ok(())
}

/// Refuses `delta_base` outside `max`, the only mode in which the core stores delta blobs, so that
/// a pack never ignores it silently.
fn check_delta_mode(mode: Mode) -> PyResult<()> {
    match mode {
        Mode::Max => Ok(()),
        Mode::Fast | Mode::Balanced => Err(PyValueError::new_err(
            "delta_base needs mode=\"max\": no other mode stores delta blobs",
        )),
    }
}

/// Packs `inputs` into one container at `output`, with the guarantee of `encode_file`. Each input
/// is a path or a `(path, member_name)` pair. A bare file path's member takes the file's name; a
/// bare directory is walked like the command line walks it (`collect_inputs`), its files named
/// relative to it, links and `*.incomplete` files skipped, and an output under it refused with
/// `InputError`. A symbolic link given as a file, or two inputs with one member name, are refused
/// with `InputError` before the output is started. `delta_base`, in
/// `max` only, is the member name of one input: it is added first and every other input through
/// `PackBuilder::add_file_delta` against it.
#[pyfunction]
#[pyo3(signature = (inputs, output, mode="max", threads=None, *, delta_base=None))]
pub(crate) fn pack(
    py: Python<'_>,
    inputs: Vec<PackInput>,
    output: PathBuf,
    mode: &str,
    threads: Option<usize>,
    delta_base: Option<String>,
) -> PyResult<EncodeReport> {
    let options = options(EncodeOptions::for_pack(), mode, threads)?;
    if delta_base.is_some() {
        check_delta_mode(options.mode)?;
    }
    // The walk reads directories, which can be large trees, so it runs detached too.
    let checked_output = output.clone();
    let resolve = move || {
        let mut named = Vec::new();
        let mut walked = Vec::new();
        for input in inputs {
            input.resolve(&mut named, &mut walked)?;
        }
        check_output(&checked_output, &walked)?;
        // Every name before the output is started, so two inputs with one name write nothing.
        check_member_names(named.iter().map(|(_, name)| name.as_str()))?;
        Ok(named)
    };
    let mut named = raised(py, py.detach(resolve))?;
    if let Some(base) = &delta_base {
        base_first(&mut named, base)?;
    }
    let run = move || {
        let mut builder = CorePackBuilder::new(&output, &options)?;
        for (at, (path, name)) in named.iter().enumerate() {
            // `base_first` put the base at 0; every other input is stored against it.
            match delta_base.as_deref().filter(|_| at > 0) {
                Some(base) => builder.add_file_delta(path, name, base)?,
                None => builder.add_file(path, name)?,
            };
        }
        builder.finish()
    };
    Ok(raised(py, py.detach(run))?.into())
}

/// Walks `roots` as `pack` walks a directory and returns the files it would add, with the links
/// and partial downloads it would skip. Reads directories and metadata only. A root that is a file
/// is named after itself; a root that is a link is skipped and counted.
#[pyfunction]
pub(crate) fn collect_inputs(py: Python<'_>, roots: Vec<PathBuf>) -> PyResult<Inputs> {
    let inputs = raised(py, py.detach(move || ziptensors_core::collect_inputs(&roots)))?;
    Ok(inputs.into())
}

/// Builds a container member by member, like `pack` but under the caller's control. The output
/// appears only at `finish`, after the whole file decoded back to its sources; until then it is a
/// temporary file beside `output`. `close()`, leaving a `with` block without `finish`, or dropping
/// the object removes that file. Any call after `finish` or `close` raises `ValueError`. After a
/// failed add every later call fails, `finish` included. Calls may come from several threads; they
/// run one at a time, with the GIL released.
#[pyclass(frozen, module = "ziptensors")]
#[derive(Debug)]
pub(crate) struct PackBuilder {
    /// None once finished or closed.
    inner: Mutex<Option<CorePackBuilder>>,
}

impl PackBuilder {
    /// The builder's lock. A poisoned lock means a core call panicked half way through, so the
    /// builder is dropped (its temporary file with it) and the object acts as closed.
    fn lock(&self) -> MutexGuard<'_, Option<CorePackBuilder>> {
        self.inner.lock().unwrap_or_else(|poisoned| {
            let mut guard = poisoned.into_inner();
            if guard.take().is_some() {
                tracing::error!("a pack builder call panicked; the unfinished pack is dropped");
            }
            self.inner.clear_poison();
            guard
        })
    }

    /// Runs `call` on the builder with the GIL released, holding the lock for the whole call.
    ///
    /// # Errors
    /// `ValueError` if the builder is finished or closed; the exception of a core error of `call`.
    fn with_builder<T, F>(&self, py: Python<'_>, call: F) -> PyResult<T>
    where
        T: Send,
        F: Send + FnOnce(&mut CorePackBuilder) -> Result<T, Error>,
    {
        let outcome = py.detach(|| self.lock().as_mut().map(call));
        match outcome {
            Some(result) => raised(py, result),
            None => Err(closed()),
        }
    }
}

/// The error of a call on a finished or closed `PackBuilder`.
fn closed() -> PyErr {
    PyValueError::new_err("the PackBuilder is finished or closed")
}

#[pymethods]
impl PackBuilder {
    /// Starts a container that will end up at `output`: creates the temporary file beside it and
    /// starts the compression threads. `mode` is `fast`, `balanced` or `max`; `threads` is None or
    /// 0 for one per logical CPU.
    #[new]
    #[pyo3(signature = (output, mode="max", threads=None))]
    fn new(py: Python<'_>, output: PathBuf, mode: &str, threads: Option<usize>) -> PyResult<Self> {
        let options = options(EncodeOptions::for_pack(), mode, threads)?;
        let builder = raised(py, py.detach(move || CorePackBuilder::new(&output, &options)))?;
        Ok(Self { inner: Mutex::new(Some(builder)) })
    }

    /// Adds the file at `path` as the member `name`, by default the file's name. A symbolic link is
    /// refused with `InputError`, and so is a name already in the pack (code `DuplicateMember`).
    #[pyo3(signature = (path, name=None))]
    fn add_file(
        &self,
        py: Python<'_>,
        path: PathBuf,
        name: Option<String>,
    ) -> PyResult<MemberReport> {
        let name = match name {
            Some(name) => name,
            None => raised(py, file_name(&path))?.to_owned(),
        };
        let report = self.with_builder(py, move |builder| builder.add_file(&path, &name))?;
        Ok(report.into())
    }

    /// Adds the file at `path` as the member `name`, and stores its tensors that pair with tensors
    /// of `base_member` as delta blobs where that is smaller (mode `max` only; elsewhere this is
    /// `add_file`). `base_member` must be a member this builder added from a file, still unchanged
    /// on disk; otherwise `NotFoundError` or `InputError`.
    fn add_file_delta(
        &self,
        py: Python<'_>,
        path: PathBuf,
        name: String,
        base_member: String,
    ) -> PyResult<MemberReport> {
        let report = self.with_builder(py, move |builder| {
            builder.add_file_delta(&path, &name, &base_member)
        })?;
        Ok(report.into())
    }

    /// Copies the member `name` of `container` into this pack under the same name, with no
    /// decoding and no recompression. The commit still decodes it against its stated hash.
    fn add_member_from(
        &self,
        py: Python<'_>,
        container: &Container,
        name: String,
    ) -> PyResult<MemberReport> {
        let source = container.core()?;
        let report =
            self.with_builder(py, move |builder| builder.add_member_from(&source, &name))?;
        Ok(report.into())
    }

    /// Writes the index, decodes every member against its source's hash, and renames the file to
    /// the output. On an error there is no new file at the output and the temporary file is gone,
    /// except after `NotDurableError`. Either way the builder is spent.
    fn finish(&self, py: Python<'_>) -> PyResult<EncodeReport> {
        let outcome = py.detach(|| self.lock().take().map(CorePackBuilder::finish));
        match outcome {
            Some(result) => Ok(raised(py, result)?.into()),
            None => Err(closed()),
        }
    }

    /// Drops an unfinished pack and its temporary file. Idempotent; nothing happens after `finish`.
    fn close(&self, py: Python<'_>) {
        py.detach(|| {
            if self.lock().take().is_some() {
                tracing::info!("unfinished pack dropped; its temporary file is removed");
            }
        });
    }

    /// Returns the builder itself, for `with`.
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Drops the pack unless `finish` ran, whether or not the block raised. Never swallows an
    /// exception of the block.
    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> bool {
        self.close(py);
        false
    }
}
