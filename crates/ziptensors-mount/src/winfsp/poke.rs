//! File: winfsp/poke.rs
//! Purpose: brings the Close of a file whose last handle and mapping are gone but whose cache section
//! the system still keeps, so that an unmount waiting for open files can end.
//!
//! Key functions: [`poke`], [`poke_path`].
//!
//! Implementation notes: measured on `WinFsp` 2.1. A memory-mapped file keeps its section after the
//! mapping is unmapped, and the section keeps the file object whose Close the unmount waits for; nothing
//! is sent to the file system when the mapping goes. The driver purges a file's cache when the last
//! handle of the file is cleaned up (`FlushAndPurgeOnCleanup`), and the purge fails, harmlessly, while a
//! mapping is alive. So opening the file and closing it again makes that last cleanup happen once more:
//! with the mapping gone, the purge succeeds and the deferred Close arrives at once; with the mapping
//! alive, nothing changes. The open goes through the volume's NT device name
//! (`\\?\GLOBALROOT\Device\Volume{...}\path`), which still reaches the volume after its mount point is
//! removed. It asks only `FILE_READ_ATTRIBUTES`, which is no write right and takes part in no
//! share-mode check, and shares everything, so it never makes another program's open fail. The calls
//! must not run on a dispatcher thread or under the volume's mutex: they are served by the dispatcher.

#![allow(unsafe_code, reason = "CreateFileW and CloseHandle on a path into the mounted volume")]
#![warn(clippy::undocumented_unsafe_blocks)]

use std::io;
use std::ptr;

use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, FILE_READ_ATTRIBUTES, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
};

use crate::tree::NodeId;

use super::names::stored_path;
use super::volume::Volume;

/// The prefix that turns an NT device name into a Win32 path.
const GLOBALROOT: &str = "\\\\?\\GLOBALROOT";

/// The Win32 path of file `id` through the volume's NT device name `device`, NUL-terminated. `None` if
/// the node has no path.
#[must_use]
pub(super) fn poke_path(volume: &Volume, device: &[u16], id: NodeId) -> Option<Vec<u16>> {
    let path = stored_path(volume.fs.tree(), id)?;
    let mut full: Vec<u16> = GLOBALROOT.encode_utf16().collect();
    full.extend_from_slice(device);
    full.extend_from_slice(&path);
    full.push(0);
    Some(full)
}

/// Opens and closes every file the volume still counts open. Returns how many were reopened. A file
/// that cannot be opened is logged at debug level and skipped: it is retried at the next poke.
pub(super) fn poke(volume: &Volume) -> usize {
    let Some(device) = volume.device() else {
        return 0;
    };
    let mut reopened = 0;
    for id in volume.busy_files() {
        let Some(path) = poke_path(volume, device, id) else {
            continue;
        };
        // SAFETY: `path` is NUL-terminated and outlives the call; null security attributes and template
        // are allowed. The returned handle is closed below and nowhere else.
        let handle = unsafe {
            CreateFileW(
                path.as_ptr(),
                FILE_READ_ATTRIBUTES,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                ptr::null(),
                OPEN_EXISTING,
                0,
                ptr::null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            tracing::debug!(node = id.get(), error = %io::Error::last_os_error(), "cannot reopen a file to end its Close");
            continue;
        }
        // SAFETY: a handle this function opened, closed once.
        if unsafe { CloseHandle(handle) } == 0 {
            tracing::warn!(node = id.get(), error = %io::Error::last_os_error(), "CloseHandle of a reopened file failed");
        }
        reopened += 1;
    }
    reopened
}

#[cfg(test)]
mod tests {
    use crate::tree::NodeId;
    use crate::winfsp::names::resolve;
    use crate::winfsp::volume::tests::volume;

    use super::{poke, poke_path};
    use crate::testkit::Scratch;

    #[test]
    fn files_are_reached_through_the_device_name_with_their_stored_case() {
        let scratch = Scratch::new("winfsp-poke");
        let volume = volume(&scratch);
        let device: Vec<u16> = "\\Device\\Volume{x}".encode_utf16().collect();
        let id = resolve(&volume.fs, &"\\models\\unet\\W.BIN".encode_utf16().collect::<Vec<_>>()).expect("file");
        let path = poke_path(&volume, &device, id).expect("path");
        assert_eq!(path.last(), Some(&0));
        let text = String::from_utf16(&path[..path.len() - 1]).expect("utf-16");
        assert_eq!(text, "\\\\?\\GLOBALROOT\\Device\\Volume{x}\\Models\\Unet\\w.bin");
        assert_eq!(poke_path(&volume, &device, NodeId::from_raw(10_000)), None);
        // No device name yet, or nothing open: nothing to reopen.
        volume.file_opened(id);
        assert_eq!(poke(&volume), 0);
    }
}
