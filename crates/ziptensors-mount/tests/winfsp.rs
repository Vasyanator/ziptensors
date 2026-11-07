//! File: tests/winfsp.rs
//! Purpose: the Windows WinFsp backend end to end: mount a real container on a new directory and on a
//! free drive letter, read every member through Windows (whole, positional, through a file mapping),
//! compare with the bytes it was packed from, check that every change is refused, and check that an
//! unmount detaches at once while open files and mappings are served until closed.
//!
//! Implementation notes: skips with one line on stderr when WinFsp cannot be loaded (`probe`), so it
//! passes under wine and on machines without WinFsp. Every `MountHandle` unmounts when dropped, also
//! when an assertion fails, and is dropped before the scratch directory. The mapping is made through
//! `windows-sys` directly, the only `unsafe` here.

#![cfg(windows)]

#[path = "../src/testkit.rs"]
mod testkit;

use std::fs::{self, File, OpenOptions as FileOptions};
use std::os::windows::fs::{FileExt, OpenOptionsExt};
use std::os::windows::io::AsRawHandle;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, SystemTime};

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::Storage::FileSystem::GetLogicalDrives;
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_READ, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_READONLY, UnmapViewOfFile,
};

use ziptensors_core::{CacheOptions, Container, OpenOptions, ReadCache};
use ziptensors_mount::{MountFs, MountHandle, WinFspOptions, WindowsPolicy, mount, probe};

use testkit::{Fixture, Scratch};

/// `MAXIMUM_ALLOWED`, what Explorer asks for: every right the security descriptor grants.
const MAXIMUM_ALLOWED: u32 = 0x0200_0000;

/// `FILE_FLAG_BACKUP_SEMANTICS`: the open asks for backup intent.
const BACKUP_SEMANTICS: u32 = 0x0200_0000;

/// How long an unmount may take to detach, or to end once the last file is closed.
const SETTLE: Duration = Duration::from_secs(10);

/// Whether WinFsp can be used here; prints why not and returns false otherwise.
fn winfsp_available() -> bool {
    match probe() {
        Ok(version) => {
            eprintln!("WinFsp {}.{}", version >> 16, version & 0xFFFF);
            true
        }
        Err(error) => {
            eprintln!("skipped: WinFsp is not usable here: {error}");
            false
        }
    }
}

/// The filesystem of `fixture` under Windows rules.
fn filesystem(fixture: &Fixture) -> Arc<MountFs> {
    let options = OpenOptions { threads: 2, ..OpenOptions::default() };
    let container = Container::open(&fixture.container, &options).expect("open");
    let cache = ReadCache::new(container, CacheOptions::default()).expect("cache");
    Arc::new(MountFs::new(cache, WindowsPolicy::default(), SystemTime::now()).expect("mountable"))
}

/// Mounts `fixture` at `mountpoint`.
fn mount_at(fixture: &Fixture, mountpoint: &Path) -> MountHandle {
    let options = WinFspOptions { threads: 2, long_paths: false };
    mount(filesystem(fixture), mountpoint, &options).expect("mount")
}

/// The member `name` under `root`, `\` separated.
fn member(root: &Path, name: &str) -> PathBuf {
    name.split('/').fold(root.to_owned(), |path, part| path.join(part))
}

/// A read-only mapping of the whole of `file`, unmapped and closed on drop.
struct Mapping {
    handle: HANDLE,
    view: MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
}

impl Mapping {
    fn new(file: &File, len: usize) -> Self {
        // SAFETY: `file` is an open handle readable by this process; a null name and attributes are
        // allowed, and a zero size maps the whole file.
        let handle = unsafe { CreateFileMappingW(file.as_raw_handle(), std::ptr::null(), PAGE_READONLY, 0, 0, std::ptr::null()) };
        assert!(!handle.is_null(), "CreateFileMappingW: {}", std::io::Error::last_os_error());
        // SAFETY: `handle` is the mapping just created; zero bytes maps all of it.
        let view = unsafe { MapViewOfFile(handle, FILE_MAP_READ, 0, 0, 0) };
        assert!(!view.Value.is_null(), "MapViewOfFile: {}", std::io::Error::last_os_error());
        Self { handle, view, len }
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the view maps the whole file, `len` bytes, read-only, until `self` drops.
        unsafe { std::slice::from_raw_parts(self.view.Value.cast::<u8>(), self.len) }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the view and the handle were created by `new` and are released once, here.
        unsafe {
            assert_ne!(UnmapViewOfFile(self.view), 0, "UnmapViewOfFile");
            assert_ne!(CloseHandle(self.handle), 0, "CloseHandle");
        }
    }
}

/// The highest drive letter in use by nothing, from Z down to D.
fn free_drive() -> Option<char> {
    // SAFETY: no argument; returns a bit mask of the drives in use.
    let used = unsafe { GetLogicalDrives() };
    ('D'..='Z').rev().find(|letter| used & (1 << (u32::from(*letter) - u32::from('A'))) == 0)
}

/// Unmounts `handle` on another thread and reports when it returned.
fn unmount_in_background(handle: MountHandle) -> mpsc::Receiver<()> {
    let (done, finished) = mpsc::channel();
    thread::spawn(move || {
        handle.unmount();
        // The receiver may be gone if the test already failed.
        done.send(()).ok();
    });
    finished
}

#[test]
fn a_directory_mount_reads_back_every_member_and_refuses_changes() {
    if !winfsp_available() {
        return;
    }
    let scratch = Scratch::new("winfsp-dir");
    let fixture = Fixture::standard(&scratch);
    let mountpoint = scratch.path().join("mnt");
    assert!(!mountpoint.exists());
    let handle = mount_at(&fixture, &mountpoint);
    assert_eq!(handle.mountpoint(), mountpoint);
    let mut listed_bytes = 0;
    for (name, content) in &fixture.files {
        let path = member(&mountpoint, name);
        assert_eq!(&fs::read(&path).expect("whole read"), content, "{name}");
        let file = File::open(&path).expect("open");
        let meta = file.metadata().expect("metadata");
        assert!(meta.permissions().readonly(), "{name}");
        assert_eq!(meta.len(), u64::try_from(content.len()).expect("small"));
        listed_bytes += meta.len();
        if content.len() > 100_000 {
            let mut piece = vec![0; 77_777];
            let read = file.seek_read(&mut piece, 99_999).expect("positional read");
            assert_eq!(&piece[..read], &content[99_999..99_999 + read], "{name}");
            let mapping = Mapping::new(&file, content.len());
            assert_eq!(mapping.bytes(), content.as_slice(), "{name} mapped");
        }
        FileOptions::new().access_mode(MAXIMUM_ALLOWED).open(&path).expect("an Explorer-style open is not a write");
        // A backup tool's read: backup intent with read access works even when the token holds the backup
        // and restore privileges (an elevated session). Backup intent with `MAXIMUM_ALLOWED` does not then,
        // see the crate's MODULE_README.
        FileOptions::new().read(true).custom_flags(BACKUP_SEMANTICS).open(&path).expect("a backup-intent read");
        assert!(FileOptions::new().write(true).open(&path).is_err(), "{name} opened for writing");
        assert!(FileOptions::new().append(true).open(&path).is_err(), "{name} opened for appending");
        assert!(fs::remove_file(&path).is_err(), "{name} removed");
        assert!(fs::rename(&path, mountpoint.join("renamed")).is_err(), "{name} renamed");
    }
    assert_eq!(listed_bytes, fixture.total_len());
    // Names are found without case and listed as stored.
    assert_eq!(fs::read(mountpoint.join("CONFIG.JSON")).expect("case-insensitive open"), fixture.files[2].1);
    let listed: Vec<String> = fs::read_dir(&mountpoint)
        .expect("list")
        .map(|entry| entry.expect("entry").file_name().into_string().expect("utf-8"))
        .collect();
    assert_eq!(listed, ["config.json", "empty.bin", "models"]);
    assert!(fs::metadata(mountpoint.join("models")).expect("dir").is_dir());
    assert!(File::create(mountpoint.join("new.bin")).is_err(), "created a file");
    assert!(fs::create_dir(mountpoint.join("newdir")).is_err(), "created a directory");
    assert!(fs::remove_dir_all(mountpoint.join("models")).is_err(), "removed a directory");
    handle.unmount();
    assert!(!mountpoint.exists(), "the mount point directory is left behind");
}

#[test]
fn a_drive_mount_appears_and_goes() {
    if !winfsp_available() {
        return;
    }
    let Some(letter) = free_drive() else {
        eprintln!("skipped: no free drive letter");
        return;
    };
    let scratch = Scratch::new("winfsp-drive");
    let fixture = Fixture::standard(&scratch);
    let root = PathBuf::from(format!("{letter}:\\"));
    let handle = mount_at(&fixture, Path::new(&format!("{}:", letter.to_ascii_lowercase())));
    assert_eq!(handle.mountpoint(), root);
    let (name, content) = &fixture.files[1];
    assert_eq!(&fs::read(member(&root, name)).expect("read"), content);
    handle.unmount();
    assert!(!root.exists(), "{} is still there", root.display());
}

#[test]
fn an_unmount_detaches_at_once_and_waits_for_open_files_and_mappings() {
    if !winfsp_available() {
        return;
    }
    let scratch = Scratch::new("winfsp-busy");
    let fixture = Fixture::standard(&scratch);
    let (name, content) = &fixture.files[0];

    // An open file: the mount point goes at once, the file still reads, closing it ends the unmount.
    let mountpoint = scratch.path().join("open");
    let handle = mount_at(&fixture, &mountpoint);
    let file = File::open(member(&mountpoint, name)).expect("open");
    assert!(handle.open_files() >= 1, "{} files open", handle.open_files());
    let finished = unmount_in_background(handle);
    let detached = (0..100).any(|_| {
        thread::sleep(SETTLE / 100);
        !mountpoint.exists()
    });
    assert!(detached, "the mount point is still there");
    let mut piece = vec![0; 4096];
    let read = file.seek_read(&mut piece, 1000).expect("a detached volume still serves open files");
    assert_eq!(&piece[..read], &content[1000..1000 + read]);
    assert!(finished.try_recv().is_err(), "the unmount did not wait for the open file");
    drop(file);
    finished.recv_timeout(SETTLE).expect("the unmount ends once the file is closed");

    // A mapping whose file handle is closed: it still holds the volume until unmapped.
    let mountpoint = scratch.path().join("mapped");
    let handle = mount_at(&fixture, &mountpoint);
    let file = File::open(member(&mountpoint, name)).expect("open");
    let mapping = Mapping::new(&file, content.len());
    drop(file);
    let finished = unmount_in_background(handle);
    thread::sleep(Duration::from_secs(2));
    assert!(!mountpoint.exists(), "the mount point is still there");
    assert_eq!(mapping.bytes(), content.as_slice(), "a detached volume still serves the mapping");
    assert!(finished.try_recv().is_err(), "the unmount did not wait for the mapping");
    drop(mapping);
    finished.recv_timeout(SETTLE).expect("the unmount ends once the mapping is gone");
}
