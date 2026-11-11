//! File: reader.rs
//! Purpose: the Python classes of the reader: `Container`, `Member`, `MemberInfo`, `Tensor`.
//!
//! Main responsibilities: translate arguments (the open options among them), run every call that
//! reads the file with the GIL released, and translate results and errors. The decisions are all in
//! `ziptensors-core`. `Container` also owns its closing: a lock around an optional core handle.
//!
//! Key structures: [`Container`], [`Member`], [`MemberInfo`], [`Tensor`].
//!
//! Implementation notes: the docs on `#[pyclass]` and `#[pymethods]` items are the Python docstrings.
//! `ziptensors.pyi` next to `Cargo.toml` states the same surface for type checkers. `Member.open`
//! is the one call that leaves Rust: the file object is pure Python, because `io.RawIOBase` cannot be
//! subclassed from an abi3 `#[pyclass]`.

use std::path::PathBuf;
use std::sync::{Mutex, PoisonError};

use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use ziptensors_core::{Container as CoreContainer, Error, Member as CoreMember, OpenOptions};
use ziptensors_core::{MemberInfo as CoreMemberInfo, Profile, ReadRequest, TensorInfo, Verify};
use ziptensors_core::layer_report;

use crate::buffer::{fill_detached, fill_many_detached};
use crate::errors::raised;
use crate::names::{dtype_name, profile_name};
use crate::report::{ClassBytes, LayerBytes, RestoreAllReport, RestoreReport};

/// What a container says about one member.
#[pyclass(frozen, get_all, module = "ziptensors")]
#[derive(Debug)]
pub(crate) struct MemberInfo {
    /// Path string, `/` separated, as stored in the file. Untrusted: check it before joining it to
    /// a directory.
    name: String,
    /// Length of the member in bytes.
    len: u64,
    /// BLAKE3 of the whole content, 32 bytes.
    blake3: [u8; 32],
    /// Recognized source format: `opaque`, `safetensors`, `gguf`, `onnx`, `pt_zip`, or
    /// `unknown(n)`. A hint.
    profile: String,
    /// Number of tensors in the table.
    tensor_count: u32,
}

impl From<&CoreMemberInfo> for MemberInfo {
    fn from(info: &CoreMemberInfo) -> Self {
        Self {
            name: info.name.clone(),
            len: info.len,
            blake3: info.blake3,
            profile: profile_name(info.profile),
            tensor_count: info.tensor_count,
        }
    }
}

/// One named tensor of a member. Metadata: restored bytes never depend on it.
#[pyclass(frozen, get_all, module = "ziptensors")]
#[derive(Debug)]
pub(crate) struct Tensor {
    /// Name as found in the source file.
    name: String,
    /// Start in the member, in bytes.
    offset: u64,
    /// Length in bytes: the size of the buffer `decode_tensor_into` needs.
    len: u64,
    /// Dimensions in the order of the source file. GGUF keeps its own order, fastest varying
    /// first, which is the reverse of numpy's.
    shape: Vec<u64>,
    /// Registry name of the element type, such as `F32`, `BF16` or `Q8_0`, or `UNKNOWN(0x....)`.
    dtype: String,
    /// `(elements, bytes)` of one block of the dtype, from
    /// [`ziptensors_core::Dtype::block_geometry`]: `(1, width)` for a plain type, one ggml block
    /// for a block-quantized type, None when the core does not know the geometry (`NIBBLE`,
    /// unknown ids).
    block: Option<(u32, u32)>,
}

impl From<&TensorInfo> for Tensor {
    fn from(info: &TensorInfo) -> Self {
        Self {
            name: info.name.clone(),
            offset: info.offset,
            len: info.len,
            shape: info.shape.clone(),
            dtype: dtype_name(info.dtype),
            block: info.dtype.block_geometry().map(|geometry| (geometry.elements, geometry.bytes)),
        }
    }
}

/// An open container file. All methods may be called from any number of threads.
///
/// `close()`, or leaving a `with` block, drops this object's handle on the file; any later call
/// raises `ValueError`. A `Member` taken before keeps the file open until it is itself dropped.
#[pyclass(frozen, module = "ziptensors")]
#[derive(Debug)]
pub(crate) struct Container {
    /// None once closed. Locked only to clone or take the handle, never across a read.
    inner: Mutex<Option<CoreContainer>>,
}

impl Container {
    /// The open core container, a cheap clone of its shared handle.
    ///
    /// # Errors
    /// `ValueError` if the container was closed.
    pub(crate) fn core(&self) -> PyResult<CoreContainer> {
        // A poisoned lock means a panic while cloning or taking an `Option`, which leaves the
        // `Option` whole either way, so its value is still the truth.
        let guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        guard.clone().ok_or_else(|| PyValueError::new_err("the container is closed"))
    }
}

/// The open options of `Container`, from its Python arguments: None takes the core's default.
///
/// # Errors
/// `ValueError` for a `verify` other than `"chunk"` or `"off"`.
fn open_options(
    threads: Option<usize>,
    max_total_decoded: Option<u64>,
    verify: &str,
    max_index_bytes: Option<u64>,
    max_decode_amplification: Option<u64>,
) -> PyResult<OpenOptions> {
    let defaults = OpenOptions::default();
    let verify = match verify {
        "chunk" => Verify::Chunk,
        "off" => Verify::Off,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown verify {other:?}, expected \"chunk\" or \"off\""
            )));
        }
    };
    Ok(OpenOptions {
        threads: threads.unwrap_or(defaults.threads),
        max_index_bytes: max_index_bytes.unwrap_or(defaults.max_index_bytes),
        max_total_decoded: max_total_decoded.unwrap_or(defaults.max_total_decoded),
        max_decode_amplification: max_decode_amplification
            .unwrap_or(defaults.max_decode_amplification),
        verify,
    })
}

#[pymethods]
impl Container {
    /// Opens the container at `path`, recognized by its magic and not by its name. `threads` is the
    /// number of decode threads, with None or 0 for one per logical CPU. `max_total_decoded` caps
    /// the sum of member lengths in bytes; a container over it is refused before any data is read,
    /// which is what bounds a `restore_file`. The default is no budget: untrusted input needs one,
    /// and 0 is a budget of zero bytes. `verify` is `"chunk"` (check every decoded chunk's XXH3) or
    /// `"off"` (exact lengths only). `max_index_bytes` caps the decompressed index, and
    /// `max_decode_amplification` bounds decode work as a multiple of the member lengths; None takes
    /// the library's defaults.
    #[new]
    #[pyo3(signature = (
        path,
        *,
        threads=None,
        max_total_decoded=None,
        verify="chunk",
        max_index_bytes=None,
        max_decode_amplification=None,
    ))]
    fn new(
        py: Python<'_>,
        path: PathBuf,
        threads: Option<usize>,
        max_total_decoded: Option<u64>,
        verify: &str,
        max_index_bytes: Option<u64>,
        max_decode_amplification: Option<u64>,
    ) -> PyResult<Self> {
        let options = open_options(
            threads,
            max_total_decoded,
            verify,
            max_index_bytes,
            max_decode_amplification,
        )?;
        let inner = raised(py, py.detach(move || CoreContainer::open(&path, &options)))?;
        Ok(Self { inner: Mutex::new(Some(inner)) })
    }

    /// The members, in file order.
    #[getter]
    fn members(&self) -> PyResult<Vec<MemberInfo>> {
        Ok(self.core()?.members().iter().map(MemberInfo::from).collect())
    }

    /// The member called `name`, or with None the only member of a single-file container.
    #[pyo3(signature = (name=None))]
    fn member(&self, py: Python<'_>, name: Option<&str>) -> PyResult<Member> {
        let container = self.core()?;
        let inner = name.map_or_else(|| container.single(), |name| container.member(name));
        let inner = raised(py, inner)?;
        // The core handle has no profile accessor; the name is unique in a validated index.
        let own_name = raised(py, inner.name())?;
        let info = container.members().iter().find(|info| info.name == own_name);
        let info = info.ok_or_else(|| Error::MemberNotFound { name: own_name.to_owned() });
        let profile = raised(py, info)?.profile;
        Ok(Member { inner, profile })
    }

    /// Where the bytes of the container went, from its index alone with no data read: the totals
    /// and the classes by dtype, as `EncodeReport.total` and `EncodeReport.classes` state them.
    fn layer_report(&self, py: Python<'_>) -> PyResult<(LayerBytes, Vec<ClassBytes>)> {
        let container = self.core()?;
        let (total, classes) = raised(py, py.detach(move || layer_report(&container)))?;
        Ok((total.into(), classes.into_iter().map(ClassBytes::from).collect()))
    }

    /// Restores every member to `dir` joined with its name, in file order, creating `dir` and the
    /// directories the names need. Every name is checked first, so one unsafe name raises
    /// `InputError` and creates nothing. After that a failure stops the run: members restored
    /// before it stay, and the failed one leaves no file. Bound it with `max_total_decoded`.
    fn restore_all(&self, py: Python<'_>, dir: PathBuf) -> PyResult<RestoreAllReport> {
        let container = self.core()?;
        let report = raised(py, py.detach(move || container.restore_all(&dir)))?;
        Ok(report.into())
    }

    /// Drops this object's handle on the file. Idempotent. Members taken before keep the file
    /// open until they are dropped.
    fn close(&self, py: Python<'_>) {
        let taken = self.inner.lock().unwrap_or_else(PoisonError::into_inner).take();
        // The last handle's drop releases the file and the decode pool, outside the GIL.
        py.detach(move || drop(taken));
    }

    /// Returns the container itself, for `with`.
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Closes the container. Never swallows an exception of the block.
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

/// One member of a container. Keeps the container alive. All methods may be called from any number
/// of threads, and those that read release the GIL.
///
/// A buffer passed to a method is any writable, C-contiguous object of the buffer protocol, of any
/// item format except Python objects, sized in bytes. No other thread may touch it during the call.
/// After an error its contents are unspecified.
#[pyclass(frozen, module = "ziptensors")]
#[derive(Debug)]
pub(crate) struct Member {
    inner: CoreMember,
    /// Recognized source format, from the container's member list.
    profile: Profile,
}

#[pymethods]
impl Member {
    /// The name of this member.
    #[getter]
    fn name(&self, py: Python<'_>) -> PyResult<String> {
        raised(py, self.inner.name().map(str::to_owned))
    }

    /// Length of the member in bytes.
    #[getter]
    fn len(&self) -> u64 {
        self.inner.len()
    }

    /// BLAKE3 of the whole content, 32 bytes, as the file states it.
    #[getter]
    fn blake3(&self) -> [u8; 32] {
        self.inner.blake3()
    }

    /// Recognized source format: `opaque`, `safetensors`, `gguf`, `onnx`, `pt_zip`, or `unknown(n)`.
    /// A hint.
    #[getter]
    fn profile(&self) -> String {
        profile_name(self.profile)
    }

    /// The tensor table of this member.
    #[getter]
    fn tensors(&self) -> Vec<Tensor> {
        self.inner.tensors().iter().map(Tensor::from).collect()
    }

    /// The tensor called `name`, for sizing a buffer.
    fn tensor_info(&self, py: Python<'_>, name: &str) -> PyResult<Tensor> {
        let found = self.inner.tensors().iter().find(|tensor| tensor.name == name);
        let found = found.ok_or_else(|| Error::TensorNotFound { name: name.to_owned() });
        raised(py, found.map(Tensor::from))
    }

    /// Fills all of `buffer` from byte `offset` of the member.
    fn read_at(&self, offset: u64, buffer: &Bound<'_, PyAny>) -> PyResult<()> {
        let member = &self.inner;
        fill_detached(buffer, move |dst| member.read_at(offset, dst))
    }

    /// Fills every buffer of `requests`, a sequence of `(offset, buffer)`, from its offset of the
    /// member, visiting the file in pack order so that a disk moves one way. Every range is checked
    /// before any read. Two buffers whose memory overlaps (the same object twice, two views of one
    /// array) raise `BufferError` before anything is written.
    fn read_many(&self, requests: Vec<(u64, Bound<'_, PyAny>)>) -> PyResult<()> {
        let (offsets, objects): (Vec<u64>, Vec<Bound<'_, PyAny>>) = requests.into_iter().unzip();
        let member = &self.inner;
        fill_many_detached(&objects, move |dsts| {
            let mut reqs: Vec<ReadRequest<'_>> = offsets
                .into_iter()
                .zip(dsts)
                .map(|(offset, dst)| ReadRequest { offset, dst })
                .collect();
            member.read_many(&mut reqs)
        })
    }

    /// Fills `buffer`, of exactly the tensor's length, with the bytes of the tensor called `name`.
    fn decode_tensor_into(&self, name: &str, buffer: &Bound<'_, PyAny>) -> PyResult<()> {
        let member = &self.inner;
        fill_detached(buffer, move |dst| member.decode_tensor_into(name, dst))
    }

    /// Returns `length` bytes from byte `offset` of the member as a new `bytes`.
    fn read<'py>(
        &self,
        py: Python<'py>,
        offset: u64,
        length: u64,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let member = &self.inner;
        // Checked here as well as in the core, so that a wild `length` is refused as the range it
        // is, and is never allocated.
        let file_len = member.len();
        if offset.checked_add(length).is_none_or(|end| end > file_len) {
            return raised(py, Err(Error::OutOfBounds { offset, len: length, file_len }));
        }
        let size = usize::try_from(length).map_err(|_| {
            PyOverflowError::new_err(format!("{length} bytes do not fit in memory here"))
        })?;
        PyBytes::new_with(py, size, |dst| {
            raised(py, py.detach(move || member.read_at(offset, dst)))
        })
    }

    /// Opens this member as a read-only, seekable binary file, an `io.RawIOBase` over `read_at`, for
    /// `torch.load` and anything else that wants a file object. Each file has its own position, and
    /// like any file object it belongs to one thread at a time.
    fn open<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let file_class = slf.py().import("ziptensors._member_file")?.getattr("MemberFile")?;
        file_class.call1((slf,))
    }

    /// Writes this member to `path`, replacing a file there. The bytes appear only after they
    /// hashed to the member's BLAKE3 and were synced. On failure nothing changes at `path`, except
    /// after `NotDurableError`: the bytes are there, but their directory could not be synced.
    fn restore_file(&self, py: Python<'_>, path: PathBuf) -> PyResult<RestoreReport> {
        let member = &self.inner;
        let report = raised(py, py.detach(move || member.restore_file(&path)))?;
        Ok(RestoreReport::from(report))
    }

    /// Decodes the whole member and compares its BLAKE3 with the stated hash. Writes nothing.
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        let member = &self.inner;
        raised(py, py.detach(move || member.verify()))
    }
}
