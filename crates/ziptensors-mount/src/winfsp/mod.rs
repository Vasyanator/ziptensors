//! File: winfsp/mod.rs
//! Purpose: the Windows backend: serves a [`MountFs`] through `WinFsp` (the DLL loaded at run time, see
//! `dll.rs`) on a drive letter or a new directory, and unmounts on Ctrl-C.
//!
//! Main responsibilities:
//! - [`mount`]: checks the tree, the mount point and the path lengths, creates the volume, mounts it
//!   and starts `WinFsp`'s dispatcher; [`MountHandle`] unmounts on drop, under the contract of
//!   `handle.rs`;
//! - [`MountHandle::unmount`]: removes the mount point at once, waits until every file still open is
//!   closed, then stops the volume; [`MountHandle::wait`]: waits for the driver to stop it;
//! - [`run_until_signal`]: a foreground mount that a first Ctrl-C or Ctrl-Break detaches and a second
//!   abandons ([`MountEnd`]);
//! - [`probe`]: whether `WinFsp` is installed, and its version.
//!
//! Key structures: [`WinFspOptions`], [`MountHandle`], [`MountPoint`], [`WinFspError`],
//! [`UnmountSignals`], [`ConsoleEvent`].
//!
//! Implementation notes:
//! - The order is `WinFsp`'s: create the volume, set its user context, set the mount point, start the
//!   dispatcher. Teardown is the reverse: stop the dispatcher (joins its threads, so no callback runs
//!   afterwards), delete the volume, drop the shared `Volume`, unload the DLL. `Live` encodes that
//!   order in its field order, so every early return of `mount` tears down correctly.
//! - Removing the mount point detaches: new opens by path fail, while files already open, mapped
//!   files included, are still served. Only files are waited for, not directories (`volume.rs`).
//! - `WinFsp` is GPL-3.0 with a FLOSS exception; using it obliges this crate's users to show its notice
//!   (see the crate's `MODULE_README.md`).

#![allow(unsafe_code, reason = "calls into the WinFsp DLL through the exports of dll.rs")]
#![warn(clippy::undocumented_unsafe_blocks)]

mod abi;
mod callbacks;
mod console;
mod dll;
mod error;
mod names;
mod poke;
mod security;
mod volume;

use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::time::{Duration, SystemTime};
use std::{env, fs, io, ptr, thread};

use windows_sys::Win32::Foundation::{ERROR_INSUFFICIENT_BUFFER, STATUS_OBJECT_NAME_COLLISION, STATUS_OBJECT_NAME_INVALID};
use windows_sys::Win32::Storage::FileSystem::{GetLogicalDrives, QueryDosDeviceW};

use crate::error::MountError;
use crate::fs::MountFs;
use crate::handle::MountEnd;
use crate::policy::{NamePolicy, WindowsPolicy};
use crate::tree::NodeId;

use abi::{DISK_DEVICE, FileSystem, NtStatus, VOLUME_NAME_UNITS};
use console::{Action, Watch, step};
use dll::{Api, Dll};
use security::Security;
use volume::Volume;

pub use console::{ConsoleEvent, UnmountSignals};
pub use error::WinFspError;
pub use names::MountPoint;

/// How often an unmount that waits for files reopens those still counted open (`poke.rs`).
const POKE_INTERVAL: Duration = Duration::from_secs(1);

/// How a `WinFsp` mount runs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WinFspOptions {
    /// Threads that serve requests of the driver; 0 is `WinFsp`'s default (one per CPU, 4 to 16), and
    /// `WinFsp` raises 1 to 2.
    pub threads: u32,
    /// Whether member paths may exceed `MAX_PATH` under the mount point. Must equal
    /// `WindowsPolicy::long_paths` of the policy the [`MountFs`] was built with, which checked the
    /// member names alone.
    pub long_paths: bool,
}

impl MountPoint {
    /// The mount point `path` names: `X:`, `x:`, `X:\` or `X:/` is drive X, anything else a directory
    /// that must not exist yet, made absolute against the current directory.
    ///
    /// # Errors
    /// [`WinFspError::MountPoint`] for a UNC, device, verbatim or drive-relative path, and
    /// [`WinFspError::Absolute`] when the current directory cannot be read.
    pub fn parse(path: &Path) -> Result<Self, WinFspError> {
        let current = env::current_dir().map_err(|source| WinFspError::Absolute { path: path.to_owned(), source })?;
        match names::parse_mount_point(path, &current) {
            Ok(Self::Drive(letter)) => Ok(Self::Drive(letter)),
            // `absolute` resolves `.` and `..` the way Windows would; collecting the components then
            // drops a trailing separator.
            Ok(Self::Directory(directory)) => {
                let absolute: PathBuf = std::path::absolute(&directory)
                    .map_err(|source| WinFspError::Absolute { path: path.to_owned(), source })?
                    .components()
                    .collect();
                // Normalization can leave the local-disk class (`C:\CON` is a device path), so the class is
                // checked again on what WinFsp will be given.
                names::check_absolute(&absolute)
                    .map_err(|reason| WinFspError::MountPoint { path: path.to_owned(), reason })?;
                Ok(Self::Directory(absolute))
            }
            Err(reason) => Err(WinFspError::MountPoint { path: path.to_owned(), reason }),
        }
    }
}

/// Loads `WinFsp` and returns its version (major in the high 16 bits, minor in the low), then unloads
/// it. For a frontend or a test that must know whether a mount can work before trying one.
///
/// # Errors
/// [`WinFspError::NotInstalled`] and the other load errors of [`mount`].
pub fn probe() -> Result<u32, WinFspError> {
    Dll::load().map(|dll| dll.version())
}

/// The `WinFsp` file system object, deleted on drop: the dispatcher stopped first when it runs.
#[derive(Debug)]
struct FileSystemGuard {
    raw: *mut FileSystem,
    api: Api,
    dispatching: bool,
}

// SAFETY: WinFsp's file system functions have no thread affinity, and the guard is the only owner of
// the object, used by one thread at a time.
unsafe impl Send for FileSystemGuard {}

impl Drop for FileSystemGuard {
    fn drop(&mut self) {
        // SAFETY: `raw` came from a successful `FspFileSystemCreate` and is deleted only here. Stopping
        // joins every dispatcher thread, so no callback runs during or after the delete.
        unsafe {
            if self.dispatching {
                (self.api.stop_dispatcher)(self.raw);
            }
            (self.api.delete)(self.raw);
        }
    }
}

/// A mounted volume and what it needs alive. Fields drop in order: the file system (no callback runs
/// after it), then the volume the callbacks used, then the DLL their code lives in.
#[derive(Debug)]
struct Live {
    file_system: FileSystemGuard,
    volume: Arc<Volume>,
    dll: Dll,
}

impl Live {
    /// Removes the mount point: new opens by path fail, open files stay served. Returns whether no
    /// file is open. Idempotent.
    fn detach(&self) -> bool {
        // SAFETY: the file system is alive while `self` is; removing a removed mount point is a no-op.
        unsafe { (self.dll.api().remove_mount_point)(self.file_system.raw) };
        self.volume.detach()
    }
}

/// A live `WinFsp` mount. Dropping it unmounts as [`MountHandle::unmount`] does: the mount point goes at
/// once, and the drop returns when the last file open under it is closed. `Send`, not `Sync`.
#[derive(Debug)]
pub struct MountHandle {
    mountpoint: PathBuf,
    live: Option<Live>,
}

/// Mounts `fs` at `mountpoint`, read-only and visible to the calling user only, and serves it from
/// `WinFsp`'s dispatcher threads. The mount exists when this returns.
///
/// `mountpoint` is a drive (`X:`) or a directory that does not exist yet, whose parent does; `WinFsp`
/// creates the directory and removes it at unmount.
///
/// # Errors
/// [`WinFspError::WrongPolicy`] unless `fs` was built with [`WindowsPolicy`]; the mount point errors of
/// [`MountPoint::parse`]; [`WinFspError::PathsTooLong`] naming every member whose full path would be
/// longer than `MAX_PATH` (unless `options.long_paths`); the load errors of [`probe`];
/// [`WinFspError::Security`]; [`WinFspError::MountPointTaken`]; [`WinFspError::Status`] for any other
/// `WinFsp` failure. Nothing is left mounted on error.
pub fn mount(fs: Arc<MountFs>, mountpoint: &Path, options: &WinFspOptions) -> Result<MountHandle, WinFspError> {
    let policy = fs.tree().policy();
    if policy != WindowsPolicy::default().label() {
        return Err(WinFspError::WrongPolicy { policy });
    }
    let point = MountPoint::parse(mountpoint)?;
    let limit = if options.long_paths { names::LONG_PATH_MAX_UNITS } else { names::CLASSIC_PATH_MAX_UNITS };
    let members = names::too_long(fs.tree(), point.root_units(), limit);
    if !members.is_empty() {
        return Err(WinFspError::PathsTooLong { mountpoint: point.path(), limit, members });
    }
    check_free(&point)?;
    let dll = Dll::load()?;
    let security = Security::for_current_user().map_err(WinFspError::Security)?;
    let created = fs.attr(NodeId::ROOT).map_or(SystemTime::UNIX_EPOCH, |attr| attr.modified);
    let params = names::volume_params(created);
    let api = *dll.api();
    let device: Vec<u16> = DISK_DEVICE.encode_utf16().chain(Some(0)).collect();
    let wide: Vec<u16> = point.to_winfsp().into_iter().chain(Some(0)).collect();
    // Setting a drive letter defines it for this logon session whatever it held before, shadowing a
    // drive in use (even C:); only the preflight checks, so it must come first.
    // SAFETY: both strings are NUL-terminated and outlive the call, which only reads them.
    let status = unsafe { (api.preflight)(device.as_ptr(), wide.as_ptr()) };
    preflight_result(status, &point)?;
    let volume = Arc::new(Volume::new(fs, security, api.add_dir_info));
    let mut raw: *mut FileSystem = ptr::null_mut();
    // SAFETY: every pointer is valid for the call; WinFsp copies the parameters and keeps the address
    // of `INTERFACE`, a static.
    let status = unsafe { (api.create)(device.as_ptr(), &raw const params, &raw const callbacks::INTERFACE, &raw mut raw) };
    if status < 0 || raw.is_null() {
        return Err(WinFspError::status("FspFileSystemCreate", status));
    }
    let mut live = Live { file_system: FileSystemGuard { raw, api, dispatching: false }, volume, dll };
    // SAFETY: `raw` is a live file system; the volume outlives it (`Live` field order). No callback runs
    // before the dispatcher starts below, so the context is set before any reads it.
    unsafe { (*raw).user_context = Arc::as_ptr(&live.volume).cast_mut().cast() };
    // SAFETY: `FspFileSystemCreate` wrote the device name, which never changes after; `set_device`
    // checks its shape and termination.
    let name: [u16; VOLUME_NAME_UNITS] = unsafe { (*raw).volume_name };
    live.volume.set_device(&name);
    // SAFETY: `wide` is NUL-terminated and outlives the call; WinFsp copies it.
    let status = unsafe { (api.set_mount_point)(raw, wide.as_ptr()) };
    if status == STATUS_OBJECT_NAME_COLLISION {
        return Err(WinFspError::MountPointTaken { path: point.path() });
    }
    if status < 0 {
        return Err(WinFspError::status("FspFileSystemSetMountPoint", status));
    }
    // SAFETY: the file system is created and mounted; the thread count is any value.
    let status = unsafe { (api.start_dispatcher)(raw, options.threads) };
    if status < 0 {
        return Err(WinFspError::status("FspFileSystemStartDispatcher", status));
    }
    live.file_system.dispatching = true;
    tracing::info!(mountpoint = %point.path().display(), threads = options.threads, "mounted");
    Ok(MountHandle { mountpoint: point.path(), live: Some(live) })
}

/// Refuses a mount point that is in use, before `WinFsp` is loaded: a drive letter the system knows
/// (a volume, `subst`, a network drive, in this logon session or globally) or a path that exists, file,
/// directory or junction. `FspFileSystemSetMountPoint` defines a letter whatever it held, so mounting
/// on a letter in use would shadow it (for a session-0 process, machine-wide); this check is the first
/// guard and `FspFileSystemPreflight` the second.
///
/// # Errors
/// [`WinFspError::MountPointTaken`]; [`WinFspError::MountPoint`] for a directory whose parent is not a
/// directory or a path that cannot be probed.
fn check_free(point: &MountPoint) -> Result<(), WinFspError> {
    match point {
        MountPoint::Drive(letter) => {
            if drive_in_use(*letter) {
                return Err(WinFspError::MountPointTaken { path: point.path() });
            }
            Ok(())
        }
        MountPoint::Directory(path) => match fs::symlink_metadata(path) {
            Ok(_) => Err(WinFspError::MountPointTaken { path: point.path() }),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let parent_is_dir = path.parent().is_some_and(|parent| fs::metadata(parent).is_ok_and(|meta| meta.is_dir()));
                if parent_is_dir {
                    Ok(())
                } else {
                    Err(WinFspError::MountPoint { path: point.path(), reason: "its parent directory does not exist" })
                }
            }
            Err(error) => {
                tracing::error!(path = %path.display(), %error, "cannot probe the mount point");
                Err(WinFspError::MountPoint { path: point.path(), reason: "the path cannot be probed" })
            }
        },
    }
}

/// Whether drive `letter` (`A` to `Z`) is defined: in the mask of `GetLogicalDrives`, or resolved by
/// `QueryDosDeviceW` (which also sees `subst` and network letters of this logon session). A letter
/// outside `A` to `Z` counts as in use.
fn drive_in_use(letter: char) -> bool {
    let Some(index) = u32::from(letter).checked_sub(u32::from('A')).filter(|index| *index < 26) else {
        return true;
    };
    // SAFETY: no argument; returns the bit mask of the drives defined for this process.
    let mask = unsafe { GetLogicalDrives() };
    if mask & (1 << index) != 0 {
        return true;
    }
    let name: Vec<u16> = [letter, ':'].iter().map(|c| u16::try_from(u32::from(*c)).unwrap_or(0)).chain(Some(0)).collect();
    let mut target = [0_u16; 1024];
    let capacity = u32::try_from(target.len()).unwrap_or(0);
    // SAFETY: `name` is NUL-terminated; `target` is writable for `capacity` units.
    let found = unsafe { QueryDosDeviceW(name.as_ptr(), target.as_mut_ptr(), capacity) };
    // A buffer too small also means the name is defined.
    found != 0 || io::Error::last_os_error().raw_os_error() == i32::try_from(ERROR_INSUFFICIENT_BUFFER).ok()
}

/// The meaning of `FspFileSystemPreflight` for `point`.
///
/// # Errors
/// [`WinFspError::MountPointTaken`] for a drive letter in use or an existing directory,
/// [`WinFspError::MountPoint`] for a directory whose parent is missing, [`WinFspError::Status`] else.
fn preflight_result(status: i32, point: &MountPoint) -> Result<(), WinFspError> {
    match status {
        STATUS_OBJECT_NAME_COLLISION => Err(WinFspError::MountPointTaken { path: point.path() }),
        STATUS_OBJECT_NAME_INVALID => Err(WinFspError::MountPoint {
            path: point.path(),
            reason: "its parent directory does not exist, or the path cannot be probed",
        }),
        failed if failed < 0 => Err(WinFspError::status("FspFileSystemPreflight", failed)),
        _ => Ok(()),
    }
}

impl MountHandle {
    /// Where the container is mounted: `X:\` for a drive.
    #[must_use]
    pub fn mountpoint(&self) -> &Path {
        &self.mountpoint
    }

    /// Files open under the mount now (directories are not counted).
    #[must_use]
    pub fn open_files(&self) -> usize {
        self.live.as_ref().map_or(0, |live| live.volume.open_files())
    }

    /// Unmounts: removes the mount point at once, waits until every file still open is closed and
    /// unmapped (or the driver stops the volume), then stops and deletes the volume. While waiting it
    /// reopens the files still counted open about once a second (`poke.rs`). Blocks the calling thread
    /// until then and takes no console key.
    ///
    /// # Errors
    /// [`MountError::Stopped`] when the driver had stopped the volume on its own, a failure the
    /// caller reports. The volume is deleted either way.
    pub fn unmount(mut self) -> Result<(), MountError> {
        self.finish_unmount()
    }

    /// Waits until the `WinFsp` driver stops the volume on its own (a dispatcher thread failed, or
    /// `WinFsp` was uninstalled), deletes the volume and returns [`MountEnd::Stopped`]. Nothing outside
    /// the process unmounts a `WinFsp` volume, so short of [`MountHandle::unmount`] from another owner
    /// of the handle this is the only end; it blocks the calling thread and takes no console key.
    ///
    /// # Errors
    /// None on this backend; the `Result` is the shared contract of `handle.rs`.
    pub fn wait(mut self) -> Result<MountEnd, MountError> {
        let Some(live) = self.live.take() else {
            return Ok(MountEnd::Unmounted);
        };
        let status = wait_stopped(&live.volume);
        // The status `stop` reads is the one just waited for: once set it never changes.
        let status = self.stop(live).unwrap_or(status);
        Ok(MountEnd::Stopped { status })
    }

    /// The whole unmount, once.
    ///
    /// # Errors
    /// [`MountError::Stopped`] as [`MountHandle::unmount`].
    fn finish_unmount(&mut self) -> Result<(), MountError> {
        let Some(live) = self.live.take() else {
            return Ok(());
        };
        if !live.detach() {
            tracing::info!(
                mountpoint = %self.mountpoint.display(),
                open_files = live.volume.open_files(),
                "detached; waiting for the files still open to be closed and unmapped"
            );
            while !live.volume.wait_idle(POKE_INTERVAL) {
                poke::poke(&live.volume);
            }
        }
        match self.stop(live) {
            None => Ok(()),
            Some(status) => Err(MountError::Stopped { mountpoint: self.mountpoint.clone(), status }),
        }
    }

    /// Stops and deletes the volume now, cutting off any reader, and logs how it ended. Returns the
    /// dispatcher's status when the driver had stopped the volume on its own.
    fn stop(&self, live: Live) -> Option<NtStatus> {
        let stopped = live.volume.stop_status();
        drop(live);
        match stopped {
            None => tracing::info!(mountpoint = %self.mountpoint.display(), "unmounted"),
            Some(status) => tracing::error!(
                mountpoint = %self.mountpoint.display(),
                status = format_args!("{status:#010x}"),
                "the WinFsp driver had stopped the volume"
            ),
        }
        stopped
    }

    /// Leaves the volume serving the files still open, for a process about to exit: nothing waits, and
    /// the exit closes the volume.
    fn abandon(mut self) {
        if let Some(live) = self.live.take() {
            tracing::warn!(mountpoint = %self.mountpoint.display(), "the volume is left to the process");
            std::mem::forget(live);
        }
    }
}

impl Drop for MountHandle {
    fn drop(&mut self) {
        // `stop` has logged a driver stop with its status and mount point; a drop has no other channel.
        if let Err(error) = self.finish_unmount() {
            tracing::debug!(%error, "the dropped mount handle did not end cleanly");
        }
    }
}

/// Blocks until the driver stops `volume` on its own and returns the dispatcher's status. Listens on
/// a channel of its own for the duration, so no foreground mount may listen at the same time (a
/// handle is waited on by one owner).
fn wait_stopped(volume: &Volume) -> NtStatus {
    let (sender, receiver) = mpsc::channel();
    // Listening before reading the status: a stop between the two is then seen by one of them.
    volume.listen(Some(sender));
    let status = loop {
        if let Some(status) = volume.stop_status() {
            break status;
        }
        // Every wake-up re-reads the status; only `Wake::Stopped` can come, since nothing detaches.
        if receiver.recv().is_err() {
            // Cannot happen while listening: the volume holds the sender. Polling keeps the wait
            // correct even so.
            tracing::warn!("the stop channel of the volume closed; polling for the stop");
            thread::sleep(POKE_INTERVAL);
        }
    };
    volume.listen(None);
    status
}

/// Serves `handle` until it ends: a first Ctrl-C or Ctrl-Break removes the mount point (files still
/// open or mapped are served until closed and unmapped, then the volume stops: [`MountEnd::Unmounted`]),
/// a second Ctrl-C or Ctrl-Break returns [`MountEnd::Abandoned`] at once, after which the caller should
/// exit, and a volume the driver stops ends it with [`MountEnd::Stopped`]. A key pressed before this is
/// called counts. Every wait listens for the keys: the final stop never waits.
pub fn run_until_signal(mut handle: MountHandle, signals: &UnmountSignals) -> MountEnd {
    let Some(volume) = handle.live.as_ref().map(|live| Arc::clone(&live.volume)) else {
        return MountEnd::Unmounted;
    };
    volume.listen(Some(signals.sender()));
    let mut state = Watch::Serving;
    let end = loop {
        if let Some(status) = volume.stop_status() {
            break MountEnd::Stopped { status };
        }
        // `None` cannot happen while `signals` holds a sender; it would mean nothing can wake us again.
        let Some(wake) = signals.next(POKE_INTERVAL) else {
            break MountEnd::Unmounted;
        };
        let (next, action) = step(state, wake);
        state = next;
        match action {
            // A file may have been opened (through a directory handle, or the device path) since the
            // wake-up was sent; then the next close sends another.
            Action::Finish if volume.open_files() == 0 => break MountEnd::Unmounted,
            Action::Ignore | Action::Finish => {}
            Action::DriverStopped => break MountEnd::Stopped { status: volume.stop_status().unwrap_or(0) },
            Action::Abandon(event) => {
                tracing::warn!(event = event.as_str(), "second console event: ending without waiting for open files");
                break MountEnd::Abandoned { signal: event.signal() };
            }
            Action::Poke => {
                poke::poke(&volume);
            }
            Action::Detach => {
                let idle = handle.live.as_ref().is_none_or(Live::detach);
                tracing::info!(mountpoint = %handle.mountpoint.display(), "unmounting on console event");
                if idle {
                    break MountEnd::Unmounted;
                }
                tracing::info!(
                    mountpoint = %handle.mountpoint.display(),
                    open_files = volume.open_files(),
                    "detached; files still open or mapped are served until closed, a second Ctrl-C ends at once"
                );
            }
        }
    };
    volume.listen(None);
    drop(volume);
    match end {
        MountEnd::Unmounted | MountEnd::Stopped { .. } => {
            // A driver stop is in `end` already, and `stop` logs it.
            if let Some(live) = handle.live.take() {
                handle.stop(live);
            }
        }
        MountEnd::Abandoned { .. } => handle.abandon(),
    }
    end
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;
    use std::time::SystemTime;

    use ziptensors_core::{CacheOptions, Container, OpenOptions, ReadCache};

    use windows_sys::Win32::Foundation::{
        STATUS_ACCESS_DENIED, STATUS_OBJECT_NAME_COLLISION, STATUS_OBJECT_NAME_INVALID, STATUS_SUCCESS,
    };

    use super::{MountPoint, WinFspError, WinFspOptions, mount, preflight_result, probe, wait_stopped};
    use crate::fs::MountFs;
    use crate::policy::{PosixPolicy, WindowsPolicy};
    use crate::testkit::{Fixture, Scratch};

    fn cache(fixture: &Fixture) -> ReadCache {
        let container = Container::open(&fixture.container, &OpenOptions::default()).expect("open");
        ReadCache::new(container, CacheOptions::default()).expect("cache")
    }

    #[test]
    fn a_missing_winfsp_is_named_with_its_download_page() {
        match probe() {
            Ok(version) => assert!(version >> 16 >= 2, "WinFsp {version:#x} is older than 2.0"),
            Err(WinFspError::NotInstalled) => {
                let message = WinFspError::NotInstalled.to_string();
                assert!(message.contains("WinFsp") && message.contains("https://winfsp.dev/rel/"), "{message}");
            }
            Err(other) => panic!("WinFsp is installed but does not load: {other}"),
        }
    }

    #[test]
    fn trees_and_mount_points_are_checked_before_winfsp_is_loaded() {
        let scratch = Scratch::new("winfsp-refuse");
        let long = format!("{}/{}", "d".repeat(120), "f".repeat(120));
        let fixture = Fixture::build(&scratch, &[(long.as_str(), 10), ("short.bin", 10)]);
        let options = WinFspOptions::default();
        let posix = Arc::new(MountFs::new(cache(&fixture), PosixPolicy, SystemTime::UNIX_EPOCH).expect("mountable"));
        let error = mount(posix, Path::new("X:"), &options).expect_err("posix tree");
        assert!(matches!(error, WinFspError::WrongPolicy { policy: "posix" }), "{error}");
        let fs = Arc::new(MountFs::new(cache(&fixture), WindowsPolicy::default(), SystemTime::UNIX_EPOCH).expect("mountable"));
        let deep = scratch.path().join("a".repeat(30));
        let error = mount(Arc::clone(&fs), &deep, &options).expect_err("too deep");
        let WinFspError::PathsTooLong { members, limit, .. } = error else { panic!("not a length error: {error}") };
        assert_eq!((members, limit), (vec![long], 259));
        let error = mount(fs, Path::new("\\\\server\\share\\m"), &options).expect_err("UNC");
        assert!(matches!(error, WinFspError::MountPoint { .. }), "{error}");
    }

    /// Runs first on a new machine (filter `guard_`): it calls no `mount`, so a broken guard cannot
    /// reach `WinFsp` with a letter in use.
    #[test]
    fn guard_the_system_drive_is_in_use() {
        assert!(super::drive_in_use('C'));
        assert!(super::drive_in_use('c'.to_ascii_uppercase()));
        let error = super::check_free(&MountPoint::Drive('C')).expect_err("C: is in use");
        assert!(matches!(error, WinFspError::MountPointTaken { .. }), "{error}");
    }

    #[test]
    fn letters_and_paths_in_use_are_refused_before_winfsp_is_loaded() {
        let scratch = Scratch::new("winfsp-in-use");
        let fixture = Fixture::build(&scratch, &[("a.bin", 10)]);
        let fs = Arc::new(MountFs::new(cache(&fixture), WindowsPolicy::default(), SystemTime::UNIX_EPOCH).expect("mountable"));
        let options = WinFspOptions::default();
        let attempt = |point: &Path| mount(Arc::clone(&fs), point, &options).expect_err("refused");
        // The system drive exists on Windows and under wine, and wine also defines Z:.
        let mut letters = vec!["C:", "c:", "C:\\", "c:/"];
        if super::drive_in_use('Z') {
            letters.push("Z:");
        }
        for letter in letters {
            let error = attempt(Path::new(letter));
            assert!(matches!(error, WinFspError::MountPointTaken { .. }), "{letter}: {error}");
        }
        // A verbatim drive is not even a mount point form.
        let error = attempt(Path::new("\\\\?\\C:"));
        assert!(matches!(error, WinFspError::MountPoint { .. }), "{error}");
        let exists = scratch.path().join("exists");
        std::fs::create_dir(&exists).expect("create");
        let error = attempt(&exists);
        assert!(matches!(error, WinFspError::MountPointTaken { .. }), "{error}");
        let error = attempt(&fixture.container);
        assert!(matches!(error, WinFspError::MountPointTaken { .. }), "an existing file: {error}");
        let error = attempt(&scratch.path().join("no").join("parent"));
        assert!(matches!(error, WinFspError::MountPoint { reason: "its parent directory does not exist", .. }), "{error}");
        // Parsed only, never handed to `mount`: on Windows 11 `C:\CON` normalizes to itself, and a
        // regression here must not create a junction at the root of C:.
        for device in ["C:\\CON", "C:\\COM1", "C:\\x\\nul.txt"] {
            let error = MountPoint::parse(Path::new(device)).expect_err("a device name");
            assert!(matches!(error, WinFspError::MountPoint { .. }), "{device}: {error}");
        }
        let error = MountPoint::parse(Path::new("C:\\..")).expect_err("a drive root");
        assert!(
            matches!(error, WinFspError::MountPoint { reason, .. } if reason.contains("root of a drive")),
            "{error}"
        );
        assert!(super::drive_in_use('1'), "not a letter");
    }

    /// `MountHandle::wait` without `WinFsp`: the wait ends on the driver's stop, also one recorded
    /// before it began.
    #[test]
    fn a_wait_ends_when_the_driver_stops_the_volume() {
        let scratch = Scratch::new("winfsp-wait");
        let volume = Arc::new(crate::winfsp::volume::tests::volume(&scratch));
        let waiting = {
            let volume = Arc::clone(&volume);
            std::thread::spawn(move || wait_stopped(&volume))
        };
        volume.stopped(-7);
        assert_eq!(waiting.join().expect("the waiting thread"), -7);
        assert_eq!(wait_stopped(&volume), -7, "a stop recorded before the wait");
    }

    #[test]
    fn a_drive_in_use_or_an_existing_directory_is_taken() {
        let drive = MountPoint::Drive('C');
        assert!(preflight_result(STATUS_SUCCESS, &drive).is_ok());
        let taken = preflight_result(STATUS_OBJECT_NAME_COLLISION, &drive).expect_err("taken");
        assert!(matches!(taken, WinFspError::MountPointTaken { .. }), "{taken}");
        let directory = MountPoint::Directory("C:\\no\\parent".into());
        let invalid = preflight_result(STATUS_OBJECT_NAME_INVALID, &directory).expect_err("invalid");
        assert!(matches!(invalid, WinFspError::MountPoint { .. }), "{invalid}");
        let other = preflight_result(STATUS_ACCESS_DENIED, &drive).expect_err("denied");
        assert!(matches!(other, WinFspError::Status { operation: "FspFileSystemPreflight", .. }), "{other}");
    }
}
