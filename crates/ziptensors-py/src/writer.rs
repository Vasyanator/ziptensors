//! File: writer.rs
//! Purpose: the Python functions of the writer: `encode_file` and `pack`.
//!
//! Main responsibilities: translate arguments, then run the whole encode with the GIL released.
//!
//! Key functions: [`encode_file`], [`pack`].
//!
//! Implementation notes: a pack member named by a bare path takes the file's name, the rule of the
//! core's `encode_file`. Inputs are not walked or filtered: a symbolic link is an error here, since
//! skipping is a decision of a frontend with a directory walk, and this one has none. `pack` names its
//! delta base by member name, the one name every input has here, where the command line names a file.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ziptensors_core::{EncodeOptions, Error, Mode, PackBuilder};

use crate::errors::raised;
use crate::report::EncodeReport;

/// One input of `pack`: a path, or a path with the name of its member.
#[derive(Debug, FromPyObject)]
pub(crate) enum PackInput {
    /// `(path, member_name)`.
    Named(PathBuf, String),
    /// A path. The member takes the file's name.
    Path(PathBuf),
}

impl PackInput {
    /// The path of this input and the name of its member.
    ///
    /// # Errors
    /// [`Error::UnsafeMemberName`] if a bare path has no UTF-8 file name.
    fn into_named(self) -> Result<(PathBuf, String), Error> {
        match self {
            Self::Named(path, name) => Ok((path, name)),
            Self::Path(path) => {
                let name = file_name(&path)?.to_owned();
                Ok((path, name))
            }
        }
    }
}

/// The file name of `path`, as a member name.
fn file_name(path: &Path) -> Result<&str, Error> {
    path.file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| Error::UnsafeMemberName { name: path.display().to_string() })
}

/// Maps the `mode` argument. Raises `ValueError` for anything but the three modes.
fn parse_mode(mode: &str) -> PyResult<Mode> {
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
/// is a path, whose member takes the file's name, or a `(path, member_name)` pair. A symbolic link
/// is refused with `InputError`. `delta_base`, in `max` only, is the member name of one input: it
/// is added first and every other input through `PackBuilder::add_file_delta` against it.
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
    let named = inputs.into_iter().map(PackInput::into_named).collect::<Result<Vec<_>, _>>();
    let mut named = raised(py, named)?;
    if let Some(base) = &delta_base {
        check_delta_mode(options.mode)?;
        base_first(&mut named, base)?;
    }
    let run = move || {
        let mut builder = PackBuilder::new(&output, &options)?;
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
