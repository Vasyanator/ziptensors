//! File: error.rs
//! Purpose: the errors of the mount: what a filesystem call returns, and why a mount cannot start.
//!
//! Key structures: [`FsError`], [`MountError`].
//!
//! Implementation notes: [`FsError`] is backend-neutral; each backend maps it to its own codes (FUSE:
//! an errno, `WinFsp`: an `NTSTATUS`). A read failure of the core becomes [`FsError::Io`] after it was
//! logged with its context by `fs.rs`, since a filesystem call has no richer channel to its caller.

use ziptensors_core::Error;

use crate::tree::TreeError;

/// Why a filesystem call failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FsError {
    /// No node has this id, or the directory has no entry of this name.
    #[error("no such file or directory")]
    NotFound,
    /// A directory operation on a file.
    #[error("not a directory")]
    NotDir,
    /// A file operation on a directory.
    #[error("is a directory")]
    IsDir,
    /// An open for writing, or any change: the mount is read-only.
    #[error("read-only file system")]
    ReadOnly,
    /// Reading the member failed in the core: damage, an I/O error of the container file. Logged.
    #[error("input/output error")]
    Io,
}

/// Why a mount cannot be built.
#[derive(Debug, thiserror::Error)]
pub enum MountError {
    /// Some member names cannot be shown as a tree on this system.
    #[error("the container cannot be mounted")]
    Names(#[from] TreeError),
    /// The core refused a member handle.
    #[error("cannot open member #{index} of the container")]
    Member {
        /// Index of the member in file order.
        index: usize,
        /// What the core said.
        #[source]
        source: Error,
    },
}
