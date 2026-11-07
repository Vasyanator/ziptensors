//! File: winfsp/error.rs
//! Purpose: why a `WinFsp` mount cannot be made.
//!
//! Key structures: [`WinFspError`].
//!
//! Implementation notes: every message says what to do. A missing `WinFsp` names the download page, a
//! mount point `WinFsp` refuses as taken says which forms are free, and an `NTSTATUS` carries the Win32
//! error it maps to, so the OS text is in the message chain.

#![allow(unsafe_code, reason = "RtlNtStatusToDosError, a pure lookup in ntdll")]
#![warn(clippy::undocumented_unsafe_blocks)]

use std::io;
use std::path::PathBuf;

use windows_sys::Win32::Foundation::RtlNtStatusToDosError;

use super::abi::NtStatus;

/// Where `WinFsp` is downloaded.
pub(super) const DOWNLOAD_URL: &str = "https://winfsp.dev/rel/";

/// Why a `WinFsp` mount cannot be made.
#[derive(Debug, thiserror::Error)]
pub enum WinFspError {
    /// `WinFsp` is not installed: the registry has no install directory.
    #[error("WinFsp is not installed; install it from {url} and try again", url = DOWNLOAD_URL)]
    NotInstalled,
    /// The registry entry of `WinFsp` exists but cannot be read.
    #[error("cannot read the WinFsp install directory from the registry; reinstall WinFsp from {url}", url = DOWNLOAD_URL)]
    Registry {
        /// The registry error.
        #[source]
        source: io::Error,
    },
    /// The `WinFsp` DLL named by the registry does not load.
    #[error("cannot load {}; reinstall WinFsp from {url}", path.display(), url = DOWNLOAD_URL)]
    Load {
        /// The DLL.
        path: PathBuf,
        /// The loader's error.
        #[source]
        source: io::Error,
    },
    /// The `WinFsp` DLL lacks an export this backend calls.
    #[error("{} has no export {name}; install WinFsp 2.0 or later from {url}", path.display(), url = DOWNLOAD_URL)]
    MissingExport {
        /// The export.
        name: &'static str,
        /// The DLL.
        path: PathBuf,
    },
    /// The filesystem was built with another policy than `WindowsPolicy`, so it may hold names
    /// Windows cannot show or two members Windows sees as one.
    #[error("a Windows mount needs member names checked under windows rules, not {policy} rules")]
    WrongPolicy {
        /// Label of the policy the tree was built with.
        policy: &'static str,
    },
    /// The mount point is not a form `WinFsp` can take.
    #[error("cannot mount on {}: {reason}", path.display())]
    MountPoint {
        /// The mount point as given.
        path: PathBuf,
        /// What is wrong and what to give instead.
        reason: &'static str,
    },
    /// Some members would have a full path longer than programs can open under this mount point.
    #[error(
        "{} member path(s) would exceed {limit} characters under {}; mount on a drive letter or a shorter directory: {}",
        members.len(),
        mountpoint.display(),
        members.join(", ")
    )]
    PathsTooLong {
        /// The mount point.
        mountpoint: PathBuf,
        /// Longest full path, in UTF-16 units, without the terminating NUL.
        limit: usize,
        /// The members over it, `/` separated, in name order. Never empty.
        members: Vec<String>,
    },
    /// The mount point is taken: a drive letter in use, or a directory that already exists.
    #[error(
        "cannot mount on {}: the drive letter is in use or the directory already exists; give a free drive letter or a path that does not exist yet (its parent must)",
        path.display()
    )]
    MountPointTaken {
        /// The mount point.
        path: PathBuf,
    },
    /// A `WinFsp` call failed.
    #[error("{operation} failed with NTSTATUS {status:#010x}")]
    Status {
        /// The `WinFsp` function.
        operation: &'static str,
        /// What it returned.
        status: NtStatus,
        /// The Win32 error of that status.
        #[source]
        source: io::Error,
    },
    /// The security descriptor of the mount cannot be built.
    #[error("cannot build the security descriptor of the mount")]
    Security(#[source] io::Error),
    /// The current directory, needed to make a relative mount point absolute, cannot be read.
    #[error("cannot make the mount point {} absolute", path.display())]
    Absolute {
        /// The mount point as given.
        path: PathBuf,
        /// The OS error.
        #[source]
        source: io::Error,
    },
}

impl WinFspError {
    /// The error of `WinFsp` function `operation` returning the failure `status`.
    #[must_use]
    pub(super) fn status(operation: &'static str, status: NtStatus) -> Self {
        // SAFETY: a table lookup in ntdll with no pointer argument.
        let code = unsafe { RtlNtStatusToDosError(status) };
        let source = io::Error::from_raw_os_error(i32::try_from(code).unwrap_or(i32::MAX));
        Self::Status { operation, status, source }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use windows_sys::Win32::Foundation::STATUS_OBJECT_NAME_NOT_FOUND;

    use super::{DOWNLOAD_URL, WinFspError};

    #[test]
    fn messages_say_what_to_do() {
        let missing = WinFspError::NotInstalled.to_string();
        assert!(missing.contains("WinFsp") && missing.contains(DOWNLOAD_URL), "{missing}");
        let long = WinFspError::PathsTooLong {
            mountpoint: PathBuf::from("C:\\deep"),
            limit: 259,
            members: vec!["a/b.bin".to_owned(), "c.bin".to_owned()],
        };
        assert_eq!(
            long.to_string(),
            "2 member path(s) would exceed 259 characters under C:\\deep; mount on a drive letter or a shorter directory: a/b.bin, c.bin"
        );
        let status = WinFspError::status("FspFileSystemCreate", STATUS_OBJECT_NAME_NOT_FOUND);
        assert_eq!(status.to_string(), "FspFileSystemCreate failed with NTSTATUS 0xc0000034");
        let WinFspError::Status { source, .. } = status else { panic!("not a status error") };
        // ERROR_FILE_NOT_FOUND.
        assert_eq!(source.raw_os_error(), Some(2));
    }
}
