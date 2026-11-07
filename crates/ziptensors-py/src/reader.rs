//! File: reader.rs
//! Purpose: the Python classes of the reader: `Container`, `Member`, `MemberInfo`, `Tensor`.
//!
//! Main responsibilities: translate arguments, run every call that reads the file with the GIL
//! released, and translate results and errors. The decisions are all in `ziptensors-core`.
//!
//! Key structures: [`Container`], [`Member`], [`MemberInfo`], [`Tensor`].
//!
//! Implementation notes: the docs on `#[pyclass]` and `#[pymethods]` items are the Python docstrings.
//! `ziptensors.pyi` next to `Cargo.toml` states the same surface for type checkers. `Member.open`
//! is the one call that leaves Rust: the file object is pure Python, because `io.RawIOBase` cannot be
//! subclassed from an abi3 `#[pyclass]`.

use std::path::PathBuf;

use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use ziptensors_core::{Container as CoreContainer, Error, Member as CoreMember, OpenOptions};
use ziptensors_core::{MemberInfo as CoreMemberInfo, Profile, TensorInfo};

use crate::buffer::fill_detached;
use crate::errors::raised;
use crate::names::{dtype_name, profile_name};
use crate::report::RestoreReport;

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
#[pyclass(frozen, module = "ziptensors")]
#[derive(Debug)]
pub(crate) struct Container {
    inner: CoreContainer,
}

#[pymethods]
impl Container {
    /// Opens the container at `path`, recognized by its magic and not by its name. `threads` is the
    /// number of decode threads, with None or 0 for one per logical CPU. `max_total_decoded` caps
    /// the sum of member lengths in bytes; a container over it is refused before any data is read,
    /// which is what bounds a `restore_file`. The default is no budget: untrusted input needs one.
    #[new]
    #[pyo3(signature = (path, threads=None, max_total_decoded=None))]
    fn new(
        py: Python<'_>,
        path: PathBuf,
        threads: Option<usize>,
        max_total_decoded: Option<u64>,
    ) -> PyResult<Self> {
        let defaults = OpenOptions::default();
        let options = OpenOptions {
            threads: threads.unwrap_or(defaults.threads),
            max_total_decoded: max_total_decoded.unwrap_or(defaults.max_total_decoded),
            ..defaults
        };
        let inner = raised(py, py.detach(move || CoreContainer::open(&path, &options)))?;
        Ok(Self { inner })
    }

    /// The members, in file order.
    #[getter]
    fn members(&self) -> Vec<MemberInfo> {
        self.inner.members().iter().map(MemberInfo::from).collect()
    }

    /// The member called `name`, or with None the only member of a single-file container.
    #[pyo3(signature = (name=None))]
    fn member(&self, py: Python<'_>, name: Option<&str>) -> PyResult<Member> {
        let inner = name.map_or_else(|| self.inner.single(), |name| self.inner.member(name));
        let inner = raised(py, inner)?;
        // The core handle has no profile accessor; the name is unique in a validated index.
        let own_name = raised(py, inner.name())?;
        let info = self.inner.members().iter().find(|info| info.name == own_name);
        let info = info.ok_or_else(|| Error::MemberNotFound { name: own_name.to_owned() });
        let profile = raised(py, info)?.profile;
        Ok(Member { inner, profile })
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
