//! File: error.rs
//! Purpose: the errors of the mount: what a filesystem call returns, why a mount cannot start, and
//! how a `MountHandle` failed to end cleanly.
//!
//! Key structures: [`FsError`], [`MountError`], [`OverlayProblem`].
//!
//! Implementation notes: [`FsError`] is backend-neutral; each backend maps it to its own codes (FUSE:
//! an errno, `WinFsp`: an `NTSTATUS`). A read failure of the core becomes [`FsError::Io`] after it was
//! logged with its context by `fs.rs`, since a filesystem call has no richer channel to its caller.

use std::fmt;
use std::io;
use std::path::PathBuf;

use ziptensors_core::Error;
use ziptensors_core::tree::{Conflict, ConflictReason};

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

/// Why a mount cannot be built, or why a `MountHandle` did not end cleanly (`unmount`, `wait`). The
/// handle variants exist only on the OS whose backend reports them. `Send + Sync`.
#[derive(Debug, thiserror::Error)]
pub enum MountError {
    /// Some member or file names cannot be shown as a tree on this system.
    #[error("the names cannot be shown as a tree on this system")]
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
    /// The root of an overlay cannot be resolved, or is not a directory.
    #[error("cannot mount {} as an overlay", root.display())]
    OverlayRoot {
        /// The root as given.
        root: PathBuf,
        /// What failed; kind `NotADirectory` for a root that is not a directory.
        #[source]
        source: io::Error,
    },
    /// The scan of an overlay's directory failed: a directory or an entry's type cannot be read, so
    /// the view would miss part of the tree.
    #[error("cannot scan {}", root.display())]
    Scan {
        /// The directory.
        root: PathBuf,
        /// What the core said.
        #[source]
        source: Error,
    },
    /// The overlay refuses entries of the directory: every damaged or unreadable file and every
    /// name claimed twice, so a mount never shows part of a tree it cannot show whole.
    #[error("{} cannot be mounted as an overlay: {}", root.display(), ProblemList(problems))]
    Overlay {
        /// The directory.
        root: PathBuf,
        /// The entries that stop the mount, in path order. Never empty.
        problems: Vec<OverlayProblem>,
    },
    /// An overlay option is out of range.
    #[error("overlay option {option} = {value}: {reason}")]
    OverlayOption {
        /// The field of `OverlayOptions`.
        option: &'static str,
        /// Its value.
        value: u64,
        /// What it must be.
        reason: &'static str,
    },
    /// The unmount did not remove the mount: `umount2` failed (as root), `/proc/self/mountinfo` cannot
    /// be read, or the mount is still listed after `fusermount3 -u -z`. The session is left to the
    /// process and the mount stays until removed by hand.
    #[cfg(target_os = "linux")]
    #[error("cannot unmount {}; run `fusermount3 -u -z` on it", mountpoint.display())]
    Unmount {
        /// The mount point.
        mountpoint: PathBuf,
        /// What failed.
        #[source]
        source: io::Error,
    },
    /// The FUSE session ended with an error, or its thread panicked. The mount is gone.
    #[cfg(target_os = "linux")]
    #[error("the FUSE session on {} failed", mountpoint.display())]
    Session {
        /// The mount point.
        mountpoint: PathBuf,
        /// The session's final error.
        #[source]
        source: io::Error,
    },
    /// The `WinFsp` driver had stopped the volume on its own (a dispatcher thread failed, or `WinFsp`
    /// was uninstalled) before the unmount ended it. The volume is deleted.
    #[cfg(windows)]
    #[error("the WinFsp driver stopped the volume on {} (NTSTATUS {status:#010x})", mountpoint.display())]
    Stopped {
        /// The mount point.
        mountpoint: PathBuf,
        /// The `NTSTATUS` the dispatcher ended with, 0 if `WinFsp` recorded none.
        status: i32,
    },
}

/// One entry that stops an overlay mount.
#[derive(Debug)]
#[non_exhaustive]
pub enum OverlayProblem {
    /// A file that cannot be read, or that starts with the container magic and does not open.
    Broken {
        /// The file.
        path: PathBuf,
        /// Why, as the scan found it.
        error: Error,
    },
    /// Names claimed twice: per-file containers presenting one name, or a name of a directory or a
    /// container file.
    Conflict(Conflict),
    /// An entry of a kind this version of the overlay does not know how to show.
    Unsupported {
        /// The entry.
        path: PathBuf,
    },
}

impl fmt::Display for OverlayProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Broken { path, error } => write!(f, "{}: {error}", path.display()),
            Self::Conflict(conflict) => {
                // `ConflictReason` is `#[non_exhaustive]` in another crate, so a catch-all arm is
                // required; it names the conflict without its reason.
                let what = match conflict.reason {
                    ConflictReason::SameName => "several containers present",
                    ConflictReason::Directory => "a container presents the directory",
                    ConflictReason::ContainerFile => "a container presents the container file",
                    _ => "containers conflict at",
                };
                write!(f, "{what} {} (", conflict.presented.display())?;
                for (at, container) in conflict.containers.iter().enumerate() {
                    if at > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", container.display())?;
                }
                f.write_str(")")
            }
            Self::Unsupported { path } => write!(f, "{}: an entry the overlay cannot show", path.display()),
        }
    }
}

/// Problems joined by `; `, for [`MountError::Overlay`].
struct ProblemList<'a>(&'a [OverlayProblem]);

impl fmt::Display for ProblemList<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (at, problem) in self.0.iter().enumerate() {
            if at > 0 {
                f.write_str("; ")?;
            }
            write!(f, "{problem}")?;
        }
        Ok(())
    }
}
