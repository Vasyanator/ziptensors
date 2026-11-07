//! File: lib.rs
//! Purpose: crate root of `ziptensors-mount`, a read-only filesystem view of a container: every member
//! as a regular file at its relative path, so unmodified programs read compressed models.
//!
//! Main responsibilities: declares the modules, the lint policy and the public re-exports. The
//! backend-neutral part (tree, name policies, [`MountFs`], errors) builds on every target; the FUSE
//! backend exists only on Linux, the `WinFsp` backend only on Windows.
//!
//! Key structures: [`MountFs`], [`Tree`], [`NamePolicy`], [`PosixPolicy`], [`WindowsPolicy`],
//! [`FsError`], [`MountError`]; on Linux `FuseOptions`, `MountHandle`, `MountEnd`, `UnmountSignals`;
//! on Windows `WinFspOptions`, `MountPoint`, `MountHandle`, `MountEnd`, `ConsoleEvent`,
//! `UnmountSignals`, `WinFspError`.
//!
//! Key functions: `mount` and `run_until_signal` on Linux and Windows, `probe` on Windows.
//!
//! Implementation notes: no format logic lives here. Bytes come from the core's `ReadCache`, which
//! owns window geometry; this crate owns only names, nodes and the OS protocol. `unsafe` is denied
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
mod policy;
#[cfg(test)]
mod testkit;
mod tree;
#[cfg(windows)]
mod winfsp;

pub use error::{FsError, MountError};
pub use fs::{Access, Attr, MountFs};
#[cfg(target_os = "linux")]
pub use fuse::{FuseOptions, MountEnd, MountHandle, UnmountSignals, mount, run_until_signal};
pub use policy::{NamePolicy, NameProblem, PosixPolicy, WindowsPolicy};
pub use tree::{BadName, DirEntry, Node, NodeId, NodeKind, Tree, TreeError};
#[cfg(windows)]
pub use winfsp::{
    ConsoleEvent, MountEnd, MountHandle, MountPoint, UnmountSignals, WinFspError, WinFspOptions, mount, probe,
    run_until_signal,
};
