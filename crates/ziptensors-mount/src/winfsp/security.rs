//! File: winfsp/security.rs
//! Purpose: the one security descriptor every node of a `WinFsp` mount reports, and the size protocol
//! of handing it to `WinFsp`.
//!
//! Key structures: [`Security`].
//!
//! Key functions: [`Security::for_current_user`], [`Security::write_to`], [`reply_size`].
//!
//! Implementation notes: `WinFsp` performs the access check itself, in user mode, from the descriptor
//! `GetSecurityByName` returns. The descriptor grants read and execute (execute is traverse on a
//! directory) to the mounting user and nothing to anyone else, the Windows form of a FUSE mount without
//! `allow_other`. An `OWNER RIGHTS` entry with the same rights takes away the owner's implicit
//! `WRITE_DAC` and `READ_CONTROL`, so an open asking for `MAXIMUM_ALLOWED` is granted no write right
//! and is not refused as a write. The descriptor is built once through SDDL and kept as self-relative
//! bytes, so it needs no `LocalFree` after construction and is `Send + Sync` as plain data.

#![allow(unsafe_code, reason = "Win32 calls that read the process token and build a security descriptor")]
#![warn(clippy::undocumented_unsafe_blocks)]

use std::ffi::c_void;
use std::io;
use std::ptr;

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_INSUFFICIENT_BUFFER, HANDLE, LocalFree, STATUS_BUFFER_OVERFLOW, STATUS_SUCCESS,
};
use windows_sys::Win32::Security::Authorization::{
    ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
};
use windows_sys::Win32::Security::{GetTokenInformation, PSECURITY_DESCRIPTOR, TOKEN_QUERY, TOKEN_USER, TokenUser};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};
use windows_sys::core::PWSTR;

use super::abi::NtStatus;

/// A self-relative security descriptor, as bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Security {
    bytes: Vec<u8>,
}

impl Security {
    /// The descriptor of the mount: owner and group the user running this process, a protected DACL
    /// granting that user read and execute, and an `OWNER RIGHTS` entry granting no more.
    ///
    /// # Errors
    /// The OS error of reading the process token, converting its user SID, or building the
    /// descriptor.
    pub(super) fn for_current_user() -> io::Result<Self> {
        let sid = current_user_sid()?;
        Self::from_sddl(&sddl(&sid))
    }

    /// The descriptor an SDDL string describes.
    ///
    /// # Errors
    /// The OS error of `ConvertStringSecurityDescriptorToSecurityDescriptorW`.
    fn from_sddl(sddl: &str) -> io::Result<Self> {
        let wide: Vec<u16> = sddl.encode_utf16().chain(Some(0)).collect();
        let mut descriptor: PSECURITY_DESCRIPTOR = ptr::null_mut();
        let mut len = 0_u32;
        // SAFETY: `wide` is NUL-terminated and outlives the call; both out-pointers are valid locals.
        let built = unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(wide.as_ptr(), SDDL_REVISION_1, &raw mut descriptor, &raw mut len)
        };
        if built == 0 {
            return Err(io::Error::last_os_error());
        }
        let local = LocalMemory(descriptor);
        let len = usize::try_from(len).map_err(io::Error::other)?;
        // SAFETY: on success the call returned a `LocalAlloc` block of `len` bytes at `descriptor`, freed
        // only when `local` drops, after this copy.
        let bytes = unsafe { std::slice::from_raw_parts(descriptor.cast::<u8>().cast_const(), len) }.to_vec();
        drop(local);
        Ok(Self { bytes })
    }

    /// The descriptor bytes.
    #[cfg(test)]
    #[must_use]
    pub(super) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Answers `WinFsp`'s request for the descriptor: `size` points at the capacity of `buffer` in bytes
    /// and receives the descriptor's length. A null `size` asks for nothing; a null `buffer` with a
    /// large enough `size` asks only for the length.
    ///
    /// Returns `STATUS_SUCCESS`, or `STATUS_BUFFER_OVERFLOW` with the length in `*size` when the
    /// buffer is too small, after which `WinFsp` retries with a buffer of that size.
    ///
    /// # Safety
    /// `size`, when not null, must be valid for reads and writes; `buffer`, when not null, must be
    /// valid for writes of `*size` bytes.
    pub(super) unsafe fn write_to(&self, buffer: *mut c_void, size: *mut usize) -> NtStatus {
        if size.is_null() {
            return STATUS_SUCCESS;
        }
        // SAFETY: `size` is not null and valid by this function's contract.
        let capacity = unsafe { *size };
        let status = reply_size(self.bytes.len(), capacity);
        // SAFETY: as above.
        unsafe { *size = self.bytes.len() };
        if status == STATUS_SUCCESS && !buffer.is_null() {
            // SAFETY: `buffer` holds `capacity` bytes, at least `bytes.len()` by `reply_size`, and cannot
            // overlap a vector of ours.
            unsafe { ptr::copy_nonoverlapping(self.bytes.as_ptr(), buffer.cast::<u8>(), self.bytes.len()) };
        }
        status
    }
}

/// The status of handing a descriptor of `needed` bytes to a buffer of `capacity` bytes.
#[must_use]
pub(super) fn reply_size(needed: usize, capacity: usize) -> NtStatus {
    if needed > capacity { STATUS_BUFFER_OVERFLOW } else { STATUS_SUCCESS }
}

/// The SDDL of the mount's descriptor for the user `sid` (`S-1-5-...`): owner and group the user, a
/// protected DACL (`P`, nothing inherited) with read and execute (`FRFX`) for the user and for
/// `OWNER RIGHTS` (`OW`).
#[must_use]
fn sddl(sid: &str) -> String {
    format!("O:{sid}G:{sid}D:P(A;;FRFX;;;{sid})(A;;FRFX;;;OW)")
}

/// Frees a `LocalAlloc` block on drop.
#[derive(Debug)]
struct LocalMemory(*mut c_void);

impl Drop for LocalMemory {
    fn drop(&mut self) {
        // SAFETY: the pointer came from a Win32 call documented to return `LocalAlloc` memory, and this
        // guard is its only owner. `LocalFree` returns null on success; a failure leaks a few bytes
        // and has nothing to report to.
        let failed = unsafe { LocalFree(self.0) };
        if !failed.is_null() {
            tracing::warn!(error = %io::Error::last_os_error(), "LocalFree failed");
        }
    }
}

/// Closes a handle on drop.
#[derive(Debug)]
struct OwnedHandle(HANDLE);

impl Drop for OwnedHandle {
    fn drop(&mut self) {
        // SAFETY: the handle was opened by this module and is closed only here.
        if unsafe { CloseHandle(self.0) } == 0 {
            tracing::warn!(error = %io::Error::last_os_error(), "CloseHandle failed");
        }
    }
}

/// The string SID of the user this process runs as.
///
/// # Errors
/// The OS error of `OpenProcessToken`, `GetTokenInformation` or `ConvertSidToStringSidW`.
fn current_user_sid() -> io::Result<String> {
    let mut token: HANDLE = ptr::null_mut();
    // SAFETY: the pseudo handle of the current process needs no closing; `token` is a valid out-pointer.
    if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &raw mut token) } == 0 {
        return Err(io::Error::last_os_error());
    }
    let token = OwnedHandle(token);
    let mut needed = 0_u32;
    // SAFETY: a null buffer of length 0 asks only for the size, written to `needed`.
    let sized = unsafe { GetTokenInformation(token.0, TokenUser, ptr::null_mut(), 0, &raw mut needed) };
    let error = io::Error::last_os_error();
    if sized != 0 || error.raw_os_error() != i32::try_from(ERROR_INSUFFICIENT_BUFFER).ok() {
        return Err(error);
    }
    // `TOKEN_USER` holds a pointer, so the buffer is made of `u64` for its alignment.
    let words = usize::try_from(needed).map_err(io::Error::other)?.div_ceil(size_of::<u64>());
    let mut buffer = vec![0_u64; words.max(size_of::<TOKEN_USER>().div_ceil(size_of::<u64>()))];
    let capacity = u32::try_from(buffer.len() * size_of::<u64>()).map_err(io::Error::other)?;
    // SAFETY: `buffer` is writable for `capacity` bytes and aligned for `TOKEN_USER`.
    let read = unsafe {
        GetTokenInformation(token.0, TokenUser, buffer.as_mut_ptr().cast(), capacity, &raw mut needed)
    };
    if read == 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the call filled `buffer` with a `TOKEN_USER`, aligned as checked above; its SID pointer
    // points into `buffer`, which lives until the end of this function.
    let sid = unsafe { (*buffer.as_ptr().cast::<TOKEN_USER>()).User.Sid };
    let mut text: PWSTR = ptr::null_mut();
    // SAFETY: `sid` is a valid SID (above); `text` is a valid out-pointer.
    if unsafe { ConvertSidToStringSidW(sid, &raw mut text) } == 0 {
        return Err(io::Error::last_os_error());
    }
    let local = LocalMemory(text.cast());
    let mut len = 0;
    // SAFETY: on success `text` is a NUL-terminated string in a `LocalAlloc` block, freed when `local`
    // drops after this copy; every unit up to and including the NUL is inside it.
    while unsafe { *text.add(len) } != 0 {
        len += 1;
    }
    // SAFETY: the `len` units before the NUL, as above.
    let units = unsafe { std::slice::from_raw_parts(text.cast_const(), len) };
    let sid = String::from_utf16(units).map_err(io::Error::other)?;
    drop(local);
    Ok(sid)
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use windows_sys::Win32::Foundation::{STATUS_BUFFER_OVERFLOW, STATUS_SUCCESS};

    use super::{Security, reply_size, sddl};

    #[test]
    fn the_descriptor_grants_read_and_execute_to_the_user_only() {
        assert_eq!(
            sddl("S-1-5-21-1-2-3-1001"),
            "O:S-1-5-21-1-2-3-1001G:S-1-5-21-1-2-3-1001D:P(A;;FRFX;;;S-1-5-21-1-2-3-1001)(A;;FRFX;;;OW)"
        );
        let security = Security::for_current_user().expect("descriptor of this process' user");
        // A self-relative descriptor starts with revision 1 and has the self-relative control bit.
        let bytes = security.bytes();
        assert!(bytes.len() > 20, "{} bytes", bytes.len());
        assert_eq!(bytes[0], 1);
        assert_ne!(u16::from_le_bytes([bytes[2], bytes[3]]) & 0x8000, 0);
    }

    #[test]
    fn a_small_buffer_gets_the_size_and_a_retry() {
        assert_eq!(reply_size(120, 1024), STATUS_SUCCESS);
        assert_eq!(reply_size(120, 120), STATUS_SUCCESS);
        assert_eq!(reply_size(120, 119), STATUS_BUFFER_OVERFLOW);
        let security = Security::for_current_user().expect("descriptor");
        let len = security.bytes().len();
        let mut size = len - 1;
        let mut buffer = vec![0_u8; len];
        // SAFETY: `buffer` holds `len` bytes, more than `size`.
        let status = unsafe { security.write_to(buffer.as_mut_ptr().cast(), &raw mut size) };
        assert_eq!((status, size), (STATUS_BUFFER_OVERFLOW, len));
        assert!(buffer.iter().all(|byte| *byte == 0), "nothing copied on overflow");
        // SAFETY: as above, `size` is now `len`.
        let status = unsafe { security.write_to(buffer.as_mut_ptr().cast(), &raw mut size) };
        assert_eq!((status, size, buffer.as_slice()), (STATUS_SUCCESS, len, security.bytes()));
        let mut size = 4096;
        // SAFETY: a null buffer is allowed and asks for the length only.
        let status = unsafe { security.write_to(ptr::null_mut(), &raw mut size) };
        assert_eq!((status, size), (STATUS_SUCCESS, len));
        // SAFETY: a null size asks for nothing.
        assert_eq!(unsafe { security.write_to(ptr::null_mut(), ptr::null_mut()) }, STATUS_SUCCESS);
    }
}
