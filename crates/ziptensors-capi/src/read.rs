//! File: read.rs
//! Purpose: the reading half of the C ABI: containers, members, tensors, restore.
//!
//! Main responsibilities: the opaque handles [`ZtContainer`] and [`ZtMember`], and the entry points
//! over `Container` and `Member`.
//!
//! Key structures: [`ZtContainer`], [`ZtMember`], [`ZtTensorDesc`].
//!
//! Implementation notes: core keeps names as Rust strings without a terminating NUL, so a member builds
//! a table of terminated copies the first time C asks for a tensor, and every pointer handed to C
//! stays inside the handle or the container it keeps alive.

use std::ffi::c_char;
use std::ptr;
use std::sync::OnceLock;

use ziptensors_core::{Container, Error, Member};

use crate::args::{check_dst, handle, path, reset_out, text, zeroed_dst};
use crate::barrier::{guarded, guarded_value};
use crate::options::{ZtOpenOptions, open_options};
use crate::status::{Failure, ZtStatus};

/// An open container. Create with `zt_open`, free with `zt_close`. Every function taking a
/// `const zt_container *` may be called from any number of threads at once.
#[derive(Debug)]
pub struct ZtContainer {
    pub(crate) inner: Container,
    /// NUL-terminated member names, in file order. Built on first use, as for tensors.
    names: OnceLock<Vec<Box<[u8]>>>,
}

/// `name` as bytes with a terminating NUL.
fn terminated(name: &str) -> Box<[u8]> {
    [name.as_bytes(), &[0]].concat().into_boxed_slice()
}

/// One member of a container. Create with `zt_member_open`, free with `zt_member_close`. A member keeps
/// its container's data alive, so it may outlive `zt_close`. Every function taking a
/// `const zt_member *` may be called from any number of threads at once.
#[derive(Debug)]
pub struct ZtMember {
    inner: Member,
    /// NUL-terminated tensor names, in the order of `Member::tensors`. Built on first use, since a
    /// caller that reads by offset should not pay one allocation per tensor.
    names: OnceLock<Vec<Box<[u8]>>>,
}

impl ZtMember {
    /// The terminated tensor names.
    fn names(&self) -> &[Box<[u8]>] {
        self.names.get_or_init(|| {
            self.inner.tensors().iter().map(|tensor| terminated(&tensor.name)).collect()
        })
    }
}

/// Description of one tensor. Metadata only: restoring a member never consults it. Both pointers are
/// borrowed from the member and stay valid until `zt_member_close` of the `zt_member` they came from.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ZtTensorDesc {
    /// Name, UTF-8, NUL-terminated. A source file can put a NUL inside a name, so `name_len` is the
    /// authority on where it ends.
    pub name: *const c_char,
    /// Length of `name` in bytes, without the terminating NUL.
    pub name_len: usize,
    /// Start in the member, in bytes.
    pub offset: u64,
    /// Length in bytes.
    pub len: u64,
    /// `ndim` dimensions, NULL when `ndim` is 0.
    pub shape: *const u64,
    /// Number of dimensions, at most 16.
    pub ndim: usize,
    /// Element type: the wire id of the dtype registry in `spec/ziptensors-format.md`. A hint, and ids
    /// unknown to the caller are legal.
    pub dtype: u16,
}

/// Opens the container at `path`, recognized by its magic and not by its name. On success `*out` is
/// a handle for `zt_close`; on any failure `*out` is NULL.
///
/// Returns `ZT_STATUS_NULL_ARGUMENT`, `ZT_STATUS_INVALID_UTF8`, `ZT_STATUS_STRUCT_SIZE` or
/// `ZT_STATUS_INVALID_ARGUMENT` for bad arguments, `ZT_STATUS_IO`, `ZT_STATUS_NOT_ZIPTENSORS`, or a
/// status naming what is wrong with the file.
///
/// # Safety
/// `path` must be NULL or NUL-terminated, and is only read during the call. `options` must be NULL,
/// which means defaults, or a valid `zt_open_options` whose `struct_size` is set. `out` must be NULL
/// or valid for a write, and is not read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zt_open(
    path: *const c_char,
    options: *const ZtOpenOptions,
    out: *mut *mut ZtContainer,
) -> ZtStatus {
    guarded("zt_open", || {
        // SAFETY: `out` is covered by this function's contract.
        unsafe { reset_out(out, "out") }?;
        // SAFETY: `path` is covered by this function's contract and outlives the call.
        let input = unsafe { self::path(path, "path") }?;
        // SAFETY: `options` is covered by this function's contract.
        let options = unsafe { open_options(options) }?;
        let inner = Container::open(input, &options)?;
        let container = Box::into_raw(Box::new(ZtContainer { inner, names: OnceLock::new() }));
        // SAFETY: `reset_out` proved `out` is not NULL, and it is writable by this function's contract.
        unsafe { out.write(container) };
        Ok(())
    })
}

/// Frees a container. NULL is a no-op. Members opened from it stay usable. Calling this twice on one
/// handle, or using it afterwards, is undefined behavior.
///
/// # Safety
/// `container` must be NULL, or a handle from `zt_open` not yet closed, with no call on it running.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zt_close(container: *mut ZtContainer) {
    guarded_value("zt_close", (), || {
        if !container.is_null() {
            // SAFETY: it came from `Box::into_raw` in `zt_open`, and this function's contract gives us
            // the last use of it.
            drop(unsafe { Box::from_raw(container) });
        }
    });
}

/// Stores the number of members of the container in `*out`, which is untouched on failure.
///
/// # Safety
/// `container` must be NULL or a live handle. `out` must be NULL or valid for a write, and is not read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zt_member_count(
    container: *const ZtContainer,
    out: *mut usize,
) -> ZtStatus {
    guarded("zt_member_count", || {
        // SAFETY: `container` is covered by this function's contract and is shared for the call.
        let container = unsafe { handle(container, "container") }?;
        if out.is_null() {
            return Err(Failure::Null { argument: "out" });
        }
        // SAFETY: not NULL, and writable by this function's contract.
        unsafe { out.write(container.inner.members().len()) };
        Ok(())
    })
}

/// Stores the name of member `index`, in file order, in `*name` and its length in bytes, without the
/// terminating NUL, in `*name_len`. Both are untouched on failure. Returns `ZT_STATUS_INVALID_ARGUMENT`
/// if `index` is not below `zt_member_count`. The first call on a handle allocates its table of names.
///
/// The name is UTF-8, `/` separated and NUL-terminated, and comes from the file, so it is untrusted:
/// a caller that joins it to a directory must check it first, and `name_len` is the authority on where
/// it ends. It is borrowed from the container and stays valid until `zt_close`.
///
/// # Safety
/// `container` must be NULL or a live handle. `name` and `name_len` must be NULL or valid for a write,
/// and are not read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zt_member_name(
    container: *const ZtContainer,
    index: usize,
    name: *mut *const c_char,
    name_len: *mut usize,
) -> ZtStatus {
    guarded("zt_member_name", || {
        // SAFETY: `container` is covered by this function's contract and is shared for the call.
        let container = unsafe { handle(container, "container") }?;
        if name.is_null() {
            return Err(Failure::Null { argument: "name" });
        }
        if name_len.is_null() {
            return Err(Failure::Null { argument: "name_len" });
        }
        let names = container.names.get_or_init(|| {
            container.inner.members().iter().map(|member| terminated(&member.name)).collect()
        });
        let entry = names.get(index).ok_or_else(|| Failure::Invalid {
            detail: format!("member {index} of {}", names.len()),
        })?;
        // SAFETY: neither is NULL, and both are writable by this function's contract. Every entry holds
        // at least its NUL, so the length does not wrap. The pointer stays valid as long as the handle,
        // which owns the table and never changes it after this first build.
        unsafe {
            name.write(entry.as_ptr().cast());
            name_len.write(entry.len() - 1);
        }
        Ok(())
    })
}

/// Opens the member called `name`, or with `name == NULL` the only member of a single-file container.
/// On success `*out` is a handle for `zt_member_close`; on any failure `*out` is NULL.
///
/// Returns `ZT_STATUS_MEMBER_NOT_FOUND` if there is no such member, or with NULL unless the container
/// has exactly one.
///
/// # Safety
/// `container` must be NULL or a live handle. `name` must be NULL or NUL-terminated, and is only read
/// during the call. `out` must be NULL or valid for a write, and is not read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zt_member_open(
    container: *const ZtContainer,
    name: *const c_char,
    out: *mut *mut ZtMember,
) -> ZtStatus {
    guarded("zt_member_open", || {
        // SAFETY: `out` is covered by this function's contract.
        unsafe { reset_out(out, "out") }?;
        // SAFETY: `container` is covered by this function's contract and is shared for the call.
        let container = unsafe { handle(container, "container") }?;
        let inner = if name.is_null() {
            container.inner.single()?
        } else {
            // SAFETY: `name` is covered by this function's contract and outlives the call.
            container.inner.member(unsafe { text(name, "name") }?)?
        };
        let member = Box::into_raw(Box::new(ZtMember { inner, names: OnceLock::new() }));
        // SAFETY: `reset_out` proved `out` is not NULL, and it is writable by this function's contract.
        unsafe { out.write(member) };
        Ok(())
    })
}

/// Frees a member, and with it every string and shape `zt_tensor_info` returned from it. NULL is a
/// no-op. Calling this twice on one handle, or using it afterwards, is undefined behavior.
///
/// # Safety
/// `member` must be NULL, or a handle from `zt_member_open` not yet closed, with no call on it running.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zt_member_close(member: *mut ZtMember) {
    guarded_value("zt_member_close", (), || {
        if !member.is_null() {
            // SAFETY: it came from `Box::into_raw` in `zt_member_open`, and this function's contract
            // gives us the last use of it.
            drop(unsafe { Box::from_raw(member) });
        }
    });
}

/// Stores the length of the virtual file in bytes in `*out`, which is untouched on failure.
///
/// # Safety
/// `member` must be NULL or a live handle. `out` must be NULL or valid for a write, and is not read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zt_member_len(member: *const ZtMember, out: *mut u64) -> ZtStatus {
    guarded("zt_member_len", || {
        // SAFETY: `member` is covered by this function's contract and is shared for the call.
        let member = unsafe { handle(member, "member") }?;
        if out.is_null() {
            return Err(Failure::Null { argument: "out" });
        }
        // SAFETY: not NULL, and writable by this function's contract.
        unsafe { out.write(member.inner.len()) };
        Ok(())
    })
}

/// Fills `dst[0..len]` from byte `offset` of the member. `dst` may be uninitialized and is not read.
/// With `len == 0`, `dst` may be NULL.
///
/// A call refused by its arguments, `ZT_STATUS_OUT_OF_BOUNDS` included, writes nothing. After a failure
/// during decoding, `dst` holds unspecified bytes.
///
/// # Safety
/// `member` must be NULL or a live handle. Unless `len` is 0 or `dst` is NULL, `dst` must be valid for
/// writes of `len` bytes, and no other thread may access them during the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zt_read_at(
    member: *const ZtMember,
    offset: u64,
    dst: *mut u8,
    len: usize,
) -> ZtStatus {
    guarded("zt_read_at", || {
        // SAFETY: `member` is covered by this function's contract and is shared for the call.
        let member = unsafe { handle(member, "member") }?;
        check_dst(dst, len)?;
        // Core checks this too, but only after the slice it needs exists, and making that slice
        // writes `dst`. A wrong range must not cost the caller's memory.
        // Lossless: `usize` is at most 64 bits on every supported target.
        let wanted = len as u64;
        let file_len = member.inner.len();
        if offset.checked_add(wanted).is_none_or(|end| end > file_len) {
            return Err(Error::OutOfBounds { offset, len: wanted, file_len }.into());
        }
        // SAFETY: `check_dst` passed, and `dst` is writable and ours by this function's contract.
        let dst = unsafe { zeroed_dst(dst, len) };
        member.inner.read_at(offset, dst)?;
        Ok(())
    })
}

/// Fills `dst[0..len]` with the bytes of the tensor called `name`. `len` must be the tensor's length.
/// `dst` may be uninitialized and is not read. With `len == 0`, `dst` may be NULL.
///
/// Returns `ZT_STATUS_TENSOR_NOT_FOUND` or `ZT_STATUS_DESTINATION_SIZE` without writing anything. After
/// a failure during decoding, `dst` holds unspecified bytes.
///
/// # Safety
/// `member` must be NULL or a live handle. `name` must be NULL or NUL-terminated, and is only read
/// during the call. Unless `len` is 0 or `dst` is NULL, `dst` must be valid for writes of `len` bytes,
/// and no other thread may access them during the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zt_decode_tensor_into(
    member: *const ZtMember,
    name: *const c_char,
    dst: *mut u8,
    len: usize,
) -> ZtStatus {
    guarded("zt_decode_tensor_into", || {
        // SAFETY: `member` is covered by this function's contract and is shared for the call.
        let member = unsafe { handle(member, "member") }?;
        // SAFETY: `name` is covered by this function's contract and outlives the call.
        let name = unsafe { text(name, "name") }?;
        check_dst(dst, len)?;
        // As in `zt_read_at`: core's checks need the slice, and the slice costs a write of `dst`.
        let tensor = member.inner.tensors().iter().find(|tensor| tensor.name == name);
        let tensor = tensor.ok_or_else(|| Error::TensorNotFound { name: name.to_owned() })?;
        // Lossless: `usize` is at most 64 bits on every supported target.
        let actual = len as u64;
        if tensor.len != actual {
            return Err(Error::DestinationSize { expected: tensor.len, actual }.into());
        }
        // SAFETY: `check_dst` passed, and `dst` is writable and ours by this function's contract.
        let dst = unsafe { zeroed_dst(dst, len) };
        member.inner.decode_tensor_into(name, dst)?;
        Ok(())
    })
}

/// Stores the number of tensors of the member in `*out`, which is untouched on failure.
///
/// # Safety
/// `member` must be NULL or a live handle. `out` must be NULL or valid for a write, and is not read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zt_tensor_count(member: *const ZtMember, out: *mut usize) -> ZtStatus {
    guarded("zt_tensor_count", || {
        // SAFETY: `member` is covered by this function's contract and is shared for the call.
        let member = unsafe { handle(member, "member") }?;
        if out.is_null() {
            return Err(Failure::Null { argument: "out" });
        }
        // SAFETY: not NULL, and writable by this function's contract.
        unsafe { out.write(member.inner.tensors().len()) };
        Ok(())
    })
}

/// Stores the description of tensor `index` in `*out`, which is untouched on failure. Returns
/// `ZT_STATUS_INVALID_ARGUMENT` if `index` is not below `zt_tensor_count`. The first call on a handle
/// allocates its table of names.
///
/// The pointers in `*out` are borrowed: they stay valid until `zt_member_close` of `member`, and no
/// longer, whatever other handles of the same container are open.
///
/// # Safety
/// `member` must be NULL or a live handle. `out` must be NULL or valid for a write, and is not read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zt_tensor_info(
    member: *const ZtMember,
    index: usize,
    out: *mut ZtTensorDesc,
) -> ZtStatus {
    guarded("zt_tensor_info", || {
        // SAFETY: `member` is covered by this function's contract and is shared for the call.
        let member = unsafe { handle(member, "member") }?;
        if out.is_null() {
            return Err(Failure::Null { argument: "out" });
        }
        let tensors = member.inner.tensors();
        let found = tensors.get(index).zip(member.names().get(index));
        let Some((tensor, name)) = found else {
            let count = tensors.len();
            return Err(Failure::Invalid { detail: format!("tensor index {index} of {count}") });
        };
        let desc = ZtTensorDesc {
            name: name.as_ptr().cast::<c_char>(),
            name_len: tensor.name.len(),
            offset: tensor.offset,
            len: tensor.len,
            shape: if tensor.shape.is_empty() { ptr::null() } else { tensor.shape.as_ptr() },
            ndim: tensor.shape.len(),
            dtype: tensor.dtype.to_wire(),
        };
        // SAFETY: not NULL, and writable by this function's contract. Both pointers stored point into
        // data the handle owns or keeps alive, as the contract promises C.
        unsafe { out.write(desc) };
        Ok(())
    })
}

/// Writes the member to the file `path`. The bytes appear there only after they hashed to the member's
/// BLAKE3 and were synced, as one rename, which replaces a file already there. On failure there is no
/// new file, what was there is untouched, and no temporary file remains, except after
/// `ZT_STATUS_COMMITTED_NOT_DURABLE`: then the verified bytes are at `path`, but its directory could
/// not be synced, so a power loss may bring back what was there before.
///
/// # Safety
/// `member` must be NULL or a live handle. `path` must be NULL or NUL-terminated, and is only read
/// during the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zt_restore_file(member: *const ZtMember, path: *const c_char) -> ZtStatus {
    guarded("zt_restore_file", || {
        // SAFETY: `member` is covered by this function's contract and is shared for the call.
        let member = unsafe { handle(member, "member") }?;
        // SAFETY: `path` is covered by this function's contract and outlives the call.
        let output = unsafe { self::path(path, "path") }?;
        member.inner.restore_file(output)?;
        Ok(())
    })
}
