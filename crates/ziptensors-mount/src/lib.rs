//! File: lib.rs
//! Purpose: crate root of `ziptensors-mount`, a read-only filesystem view of a container (every member
//! as a regular file at its relative path) or of a directory tree whose per-file containers show as
//! their original files, so unmodified programs read compressed models.
//!
//! Main responsibilities: declares the modules, the lint policy and the public re-exports. The
//! backend-neutral part (tree, name policies, [`MountFs`], errors) builds on every target; the FUSE
//! backend exists only on Linux, the `WinFsp` backend only on Windows.
//!
//! Key structures: [`MountFs`], [`OverlayOptions`], [`Tree`], [`NamePolicy`], [`PosixPolicy`],
//! [`WindowsPolicy`], [`FsError`], [`MountError`], [`OverlayProblem`]; on Linux and Windows `MountHandle` (one contract on both, see
//! `handle.rs`), `MountEnd`, `UnmountSignals`; on Linux `FuseOptions`; on Windows `WinFspOptions`,
//! `MountPoint`, `ConsoleEvent`, `WinFspError`.
//!
//! Key functions: `mount`, `run_until_signal` and `signal_name` on Linux and Windows, `probe` on
//! Windows.
//!
//! Implementation notes: no format logic lives here. Member bytes come from the core's `ReadCache`,
//! which owns window geometry, an overlay's plain files from the core's `FileSource`, and which file
//! is a per-file container from the core's `tree::scan_view`; this crate owns only names, nodes,
//! open handles and the OS protocol. `unsafe` is denied
//! crate-wide and allowed only file by file in the `WinFsp` backend, the FFI to its DLL and to Win32.
//! The Windows backend uses `WinFsp` (GPL-3.0 with a FLOSS exception): a program built on it must show
//! `WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos` and a link to
//! <https://github.com/winfsp/winfsp> in its user interface and documentation, and a proprietary
//! program needs `WinFsp`'s commercial license instead.

#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![deny(unsafe_code)]

mod error;
mod fs;
#[cfg(target_os = "linux")]
mod fuse;
#[cfg(any(target_os = "linux", windows))]
mod handle;
mod overlay;
mod policy;
#[cfg(test)]
mod testkit;
mod tree;
#[cfg(windows)]
mod winfsp;

pub use error::{FsError, MountError, OverlayProblem};
pub use fs::{Access, Attr, MountFs};
#[cfg(target_os = "linux")]
pub use fuse::{FuseOptions, MountHandle, UnmountSignals, mount, run_until_signal};
#[cfg(any(target_os = "linux", windows))]
pub use handle::{MountEnd, signal_name};
pub use overlay::OverlayOptions;
pub use policy::{NamePolicy, NameProblem, PosixPolicy, WindowsPolicy};
pub use tree::{BadName, DirEntry, Node, NodeId, NodeKind, Tree, TreeError};
#[cfg(windows)]
pub use winfsp::{
    ConsoleEvent, MountHandle, MountPoint, UnmountSignals, WinFspError, WinFspOptions, mount, probe, run_until_signal,
};
