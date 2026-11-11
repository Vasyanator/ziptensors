//! File: buffer.rs
//! Purpose: filling a caller's Python buffer with the GIL released.
//!
//! Main responsibilities:
//! - [`WritableBuffer`], a held `Py_buffer` that is proven writable, C-contiguous and free of Python
//!   object pointers, and released on drop;
//! - [`fill_detached`], the entry point of the module's single-buffer calls, and
//!   [`fill_many_detached`], the one for several buffers, which refuses buffers whose memory
//!   overlaps before any mutable slice exists.
//!
//! Key structures: [`WritableBuffer`].
//!
//! Key functions: [`fill_detached`], [`fill_many_detached`].
//!
//! Implementation notes: this is the only file with `unsafe`. `pyo3::buffer::PyBuffer<u8>` is not
//! usable, since it refuses any item that is not a byte, so a float32 array would fail its
//! format check. `PyUntypedBuffer` is not used either: it requests a read-only view and so never
//! tells the exporter that the caller is going to write.

use std::ffi::{CStr, c_char, c_int};
use std::ops::Range;
use std::slice;

use pyo3::exceptions::{PyBufferError, PyOverflowError, PyTypeError};
use pyo3::ffi;
use pyo3::prelude::*;
use ziptensors_core::Error;

use crate::errors::raised;

/// What is asked of an exporter. The format is requested only to look for object pointers; a view
/// without it would present every buffer as bytes.
const FLAGS: c_int = ffi::PyBUF_WRITABLE | ffi::PyBUF_C_CONTIGUOUS | ffi::PyBUF_FORMAT;

/// ASCII `C`, the order argument of `PyBuffer_IsContiguous`.
const C_ORDER: c_char = 0x43;

/// A held view of a caller's buffer, writable, C-contiguous, and released on drop.
///
/// Holding it keeps the memory in place: an exporter refuses to resize while a view is open. The
/// view is boxed because exporters may point it at itself. It contains raw pointers, so it is not
/// `Send` and cannot enter a `Python::detach` closure: it is always dropped by a thread attached to
/// the interpreter, as `PyBuffer_Release` requires. Only the slice crosses into the closure.
struct WritableBuffer {
    view: Box<ffi::Py_buffer>,
    len: usize,
}

impl WritableBuffer {
    /// Requests a view of `object` and checks it.
    ///
    /// # Errors
    /// `TypeError` if `object` has no buffer interface. `BufferError` if it is read-only, not
    /// C-contiguous, holds Python objects, or the exporter refuses for a reason of its own. The
    /// exporter's exception, which differs between `bytes`, `numpy` and `memoryview`, is the cause.
    fn acquire(object: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = object.py();
        // SAFETY: `object` is a live reference and this thread is attached to the interpreter.
        if unsafe { ffi::PyObject_CheckBuffer(object.as_ptr()) } == 0 {
            let kind = object.get_type().name()?;
            return Err(PyTypeError::new_err(format!(
                "destination must support the buffer protocol, got {kind}"
            )));
        }
        let mut view = Box::new(ffi::Py_buffer::new());
        // SAFETY: as above, and `view` points to a `Py_buffer` that outlives the call.
        let status = unsafe { ffi::PyObject_GetBuffer(object.as_ptr(), &raw mut *view, FLAGS) };
        if status != 0 {
            let cause = PyErr::fetch(py);
            let err = PyBufferError::new_err(format!(
                "destination must be a writable, C-contiguous buffer: {cause}"
            ));
            err.set_cause(py, Some(cause));
            return Err(err);
        }
        // From here on `Drop` owes the release, so every early return below stays balanced.
        let mut buffer = Self { view, len: 0 };
        buffer.len = usize::try_from(buffer.view.len)
            .map_err(|_| PyOverflowError::new_err("the buffer reports a negative length"))?;
        buffer.check()?;
        Ok(buffer)
    }

    /// Checks what the flags asked for, since the slice below is only sound if an exporter
    /// honoured them, and what they cannot ask for: that no item is a Python object.
    fn check(&self) -> PyResult<()> {
        if self.view.readonly != 0 {
            return Err(PyBufferError::new_err("destination buffer is read-only"));
        }
        // SAFETY: `view` was filled by `PyObject_GetBuffer` and is not yet released.
        if unsafe { ffi::PyBuffer_IsContiguous(&raw const *self.view, C_ORDER) } == 0 {
            return Err(PyBufferError::new_err("destination buffer is not C-contiguous"));
        }
        if self.view.format.is_null() {
            return Ok(());
        }
        // SAFETY: a non-null `format` is a NUL-terminated string owned by the exporter, valid while
        // the view is held.
        let format = unsafe { CStr::from_ptr(self.view.format) };
        if holds_objects(format) {
            return Err(PyBufferError::new_err(
                "destination buffer holds Python objects; decoded bytes would overwrite pointers",
            ));
        }
        Ok(())
    }

    /// The address range of the buffer's bytes, or None for an empty buffer, which holds no byte
    /// that could alias another.
    ///
    /// # Errors
    /// `BufferError` if the range would wrap the address space, which no real exporter reports.
    fn address_range(&self) -> PyResult<Option<Range<usize>>> {
        if self.len == 0 {
            return Ok(None);
        }
        let start = self.view.buf.cast::<u8>().addr();
        let end = start
            .checked_add(self.len)
            .ok_or_else(|| PyBufferError::new_err("destination buffer wraps the address space"))?;
        Ok(Some(start..end))
    }

    /// The whole buffer, as bytes.
    fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            // An empty view may have a null `buf`, which a slice must not.
            return &mut [];
        }
        // SAFETY: `check` proved that `len` bytes from `buf` are one writable run, and they stay in
        // place while the view is held, which `&mut self` extends over the slice. `u8` has no
        // alignment. Within one call, `fill_many_detached` proved that no two held views overlap
        // before it made the first slice. Exclusivity against other threads is the caller's
        // contract, stated on the Python methods: as with `readinto`, no other thread may touch the
        // buffer during the call.
        unsafe { slice::from_raw_parts_mut(self.view.buf.cast::<u8>(), self.len) }
    }
}

impl Drop for WritableBuffer {
    fn drop(&mut self) {
        // SAFETY: the view was filled by a successful `PyObject_GetBuffer`, and this is its only
        // release. The type is not `Send`, so this runs on the attached thread that acquired it.
        unsafe { ffi::PyBuffer_Release(&raw mut *self.view) };
    }
}

/// Whether a PEP 3118 format string has a Python object item, code `O`. Field names sit between
/// colons and are skipped, so that a structured array with a field called `O` stays legal.
fn holds_objects(format: &CStr) -> bool {
    let mut in_name = false;
    format.to_bytes().iter().any(|&byte| {
        if byte == b':' {
            in_name = !in_name;
        }
        byte == b'O' && !in_name
    })
}

/// Runs `fill` on all of `object`'s bytes with the GIL released, holding the buffer throughout. The
/// length is in bytes, whatever the item format.
///
/// # Errors
/// As [`WritableBuffer::acquire`], and the exception of any core error of `fill`.
pub(crate) fn fill_detached<F>(object: &Bound<'_, PyAny>, fill: F) -> PyResult<()>
where
    F: Send + FnOnce(&mut [u8]) -> Result<(), Error>,
{
    let py = object.py();
    let mut buffer = WritableBuffer::acquire(object)?;
    let dst = buffer.as_mut_slice();
    raised(py, py.detach(move || fill(dst)))
}

/// Refuses `buffers` if the memory of any two overlaps: two `&mut [u8]` over one byte would be
/// undefined behaviour, whatever the reads then wrote. Empty buffers overlap nothing.
///
/// # Errors
/// `BufferError` naming the two positions (in the caller's order) whose memory overlaps.
fn check_disjoint(buffers: &[WritableBuffer]) -> PyResult<()> {
    let mut ranges = Vec::with_capacity(buffers.len());
    for (at, buffer) in buffers.iter().enumerate() {
        if let Some(range) = buffer.address_range()? {
            ranges.push((range, at));
        }
    }
    ranges.sort_unstable_by_key(|(range, _)| range.start);
    for pair in ranges.windows(2) {
        let [(first, first_at), (second, second_at)] = pair else { continue };
        if second.start < first.end {
            return Err(PyBufferError::new_err(format!(
                "destination buffers {first_at} and {second_at} overlap in memory; every request \
                 needs a buffer of its own"
            )));
        }
    }
    Ok(())
}

/// Runs `fill` on all bytes of every buffer of `objects`, in the same order, with the GIL released
/// and the buffers held throughout. Every buffer is acquired and checked, and the set is proven
/// free of overlapping memory, before the first slice is made.
///
/// # Errors
/// As [`WritableBuffer::acquire`] for each object, `BufferError` if two buffers overlap (the same
/// object twice, or two views of one array), and the exception of any core error of `fill`.
pub(crate) fn fill_many_detached<F>(objects: &[Bound<'_, PyAny>], fill: F) -> PyResult<()>
where
    F: Send + FnOnce(Vec<&mut [u8]>) -> Result<(), Error>,
{
    let Some(first) = objects.first() else {
        return Ok(());
    };
    let py = first.py();
    let mut buffers = objects.iter().map(WritableBuffer::acquire).collect::<PyResult<Vec<_>>>()?;
    check_disjoint(&buffers)?;
    let dsts: Vec<&mut [u8]> = buffers.iter_mut().map(WritableBuffer::as_mut_slice).collect();
    raised(py, py.detach(move || fill(dsts)))
}
