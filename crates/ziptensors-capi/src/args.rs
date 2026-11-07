//! File: args.rs
//! Purpose: turning the raw arguments of an entry point into checked Rust values.
//!
//! Main responsibilities: null, UTF-8 and length checks, handle borrows, output slots and the
//! destination buffers of reads.
//!
//! Key functions: [`text`], [`path`], [`handle`], [`handle_mut`], [`reset_out`], [`check_dst`],
//! [`zeroed_dst`].
//!
//! Implementation notes: the `unsafe` functions here narrow what an entry point must argue to one line
//! of its contract: "this pointer is NULL or valid".

use std::ffi::{CStr, c_char};
use std::path::Path;
use std::ptr;
use std::slice;

use crate::status::Failure;

/// Borrows a NUL-terminated UTF-8 string.
///
/// # Safety
/// `raw` is NULL, or a NUL-terminated string that stays valid and unchanged for `'a`.
pub(crate) unsafe fn text<'a>(
    raw: *const c_char,
    argument: &'static str,
) -> Result<&'a str, Failure> {
    if raw.is_null() {
        return Err(Failure::Null { argument });
    }
    // SAFETY: not NULL, and NUL-terminated and stable for `'a` by this function's contract.
    let raw = unsafe { CStr::from_ptr(raw) };
    raw.to_str().map_err(|_| Failure::Utf8 { argument })
}

/// Borrows a NUL-terminated UTF-8 path. See [`text`].
///
/// # Safety
/// As [`text`].
pub(crate) unsafe fn path<'a>(
    raw: *const c_char,
    argument: &'static str,
) -> Result<&'a Path, Failure> {
    // SAFETY: passed on from this function's contract.
    unsafe { text(raw, argument) }.map(Path::new)
}

/// Borrows a handle.
///
/// # Safety
/// `raw` is NULL, or a live handle that nothing frees or mutates for `'a`.
pub(crate) unsafe fn handle<'a, T>(raw: *const T, argument: &'static str) -> Result<&'a T, Failure> {
    // SAFETY: live and shared for `'a` by this function's contract; `as_ref` handles NULL.
    unsafe { raw.as_ref() }.ok_or(Failure::Null { argument })
}

/// Borrows a handle exclusively.
///
/// # Safety
/// `raw` is NULL, or a live handle that nothing else touches for `'a`.
pub(crate) unsafe fn handle_mut<'a, T>(
    raw: *mut T,
    argument: &'static str,
) -> Result<&'a mut T, Failure> {
    // SAFETY: live and exclusive for `'a` by this function's contract; `as_mut` handles NULL.
    unsafe { raw.as_mut() }.ok_or(Failure::Null { argument })
}

/// Checks an output slot and sets it to NULL, so that every failure leaves the caller with NULL.
///
/// # Safety
/// `out` is NULL, or valid for a pointer write. It is not read, so it may be uninitialized.
pub(crate) unsafe fn reset_out<T>(out: *mut *mut T, argument: &'static str) -> Result<(), Failure> {
    if out.is_null() {
        return Err(Failure::Null { argument });
    }
    // SAFETY: not NULL, and valid for a write by this function's contract.
    unsafe { out.write(ptr::null_mut()) };
    Ok(())
}

/// Checks a destination pointer and length, without touching the memory: NULL is fine for an empty
/// destination, and a Rust slice cannot be longer than `isize::MAX` bytes.
pub(crate) fn check_dst(dst: *mut u8, len: usize) -> Result<(), Failure> {
    if dst.is_null() && len != 0 {
        return Err(Failure::Null { argument: "dst" });
    }
    if isize::try_from(len).is_err() {
        return Err(Failure::Invalid { detail: format!("argument `len` is too large: {len}") });
    }
    Ok(())
}

/// Turns a destination that passed [`check_dst`] into a slice, zeroing it first.
///
/// `dst` may be uninitialized memory, from `malloc` for example, and a `&mut [u8]` must not point at
/// that; core takes nothing weaker. Zeroing through the raw pointer is what makes the slice sound, and
/// it means no byte is ever read before it was written. It costs one `memset` of the destination.
///
/// # Safety
/// `dst` and `len` passed [`check_dst`], and unless `len` is 0, `dst` is valid for writes of `len` bytes
/// and nothing else accesses them for `'a`.
pub(crate) unsafe fn zeroed_dst<'a>(dst: *mut u8, len: usize) -> &'a mut [u8] {
    if len == 0 {
        return &mut [];
    }
    // SAFETY: `len` writable bytes by this function's contract.
    unsafe {
        dst.write_bytes(0, len);
        slice::from_raw_parts_mut(dst, len)
    }
}
