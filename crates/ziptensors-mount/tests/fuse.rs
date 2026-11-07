//! File: tests/fuse.rs
//! Purpose: the Linux FUSE backend end to end: mount a real container under the temp directory, read
//! every member through the kernel (whole, positional, and through `mmap` in a Python child), compare
//! with the core's `read_at`, unmount, and check that no mount is left.
//!
//! Implementation notes: skips with one line on stderr unless `/dev/fuse` opens for reading and writing
//! and `fusermount3` is on `PATH`, so it passes in a container without FUSE. The `MountHandle` unmounts
//! when dropped, also when an assertion fails, and is dropped before the scratch directory. The `mmap`
//! comparison runs `python3 -I` and is skipped with one line if there is none, so the test needs no
//! `unsafe`.

#![cfg(target_os = "linux")]

#[path = "../src/testkit.rs"]
mod testkit;

use std::env;
use std::fs::{self, OpenOptions as FileOptions};
use std::io::ErrorKind;
use std::os::unix::fs::{FileExt, PermissionsExt};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use nix::errno::Errno;
use nix::unistd::{AccessFlags, access};

use ziptensors_core::{CacheOptions, Container, OpenOptions, ReadCache};
use ziptensors_mount::{FuseOptions, MountFs, PosixPolicy, mount};

use testkit::{Fixture, Scratch};

/// Whether this machine can mount without privileges; `None` if it can, else why not.
fn fuse_unavailable() -> Option<String> {
    if let Err(error) = FileOptions::new().read(true).write(true).open("/dev/fuse") {
        return Some(format!("/dev/fuse cannot be opened for reading and writing: {error}"));
    }
    let on_path = env::var_os("PATH")
        .is_some_and(|path| env::split_paths(&path).any(|dir| dir.join("fusermount3").is_file()));
    if on_path { None } else { Some("fusermount3 is not on PATH".to_owned()) }
}

/// A mount of the standard fixture at `scratch/mnt`, with the filesystem behind it.
fn mount_fixture(scratch: &Scratch, fixture: &Fixture) -> (std::path::PathBuf, Arc<MountFs>, ziptensors_mount::MountHandle) {
    let mountpoint = scratch.path().join("mnt");
    fs::create_dir(&mountpoint).expect("create mountpoint");
    let options = OpenOptions { threads: 2, ..OpenOptions::default() };
    let container = Container::open(&fixture.container, &options).expect("open");
    let cache = ReadCache::new(container, CacheOptions::default()).expect("cache");
    let fs = Arc::new(MountFs::new(cache, PosixPolicy, SystemTime::now()).expect("mountable"));
    let fuse_options = FuseOptions { threads: 2, fs_name: fixture.container.display().to_string() };
    let handle = mount(Arc::clone(&fs), &mountpoint, &fuse_options).expect("mount");
    (mountpoint, fs, handle)
}

/// Whether `/proc/mounts` lists a mount at `mountpoint`.
fn mounted(mountpoint: &Path) -> bool {
    let mounts = fs::read_to_string("/proc/mounts").expect("read /proc/mounts");
    let target = mountpoint.to_str().expect("a UTF-8 temp path");
    mounts.lines().any(|line| line.split(' ').nth(1) == Some(target))
}

/// What `python3 -I` reads from `path` through `mmap`, or `None` without a Python.
fn read_through_mmap(path: &Path) -> Option<Vec<u8>> {
    const SCRIPT: &str = "import mmap, sys\n\
        with open(sys.argv[1], 'rb') as f:\n\
        \x20   if f.seek(0, 2) == 0:\n\
        \x20       sys.exit(0)\n\
        \x20   with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:\n\
        \x20       sys.stdout.buffer.write(m[:])\n";
    let output = match Command::new("python3").arg("-I").arg("-c").arg(SCRIPT).arg(path).output() {
        Ok(output) => output,
        Err(error) if error.kind() == ErrorKind::NotFound => return None,
        Err(error) => panic!("cannot run python3: {error}"),
    };
    assert!(output.status.success(), "python3 failed: {}", String::from_utf8_lossy(&output.stderr));
    Some(output.stdout)
}

#[test]
fn a_mount_serves_every_member_byte_exact_and_leaves_nothing_behind() {
    if let Some(reason) = fuse_unavailable() {
        eprintln!("skipped: {reason}");
        return;
    }
    let scratch = Scratch::new("fuse");
    let fixture = Fixture::standard(&scratch);
    let mountpoint = scratch.path().join("mnt");
    fs::create_dir(&mountpoint).expect("create mountpoint");
    let options = OpenOptions { threads: 2, ..OpenOptions::default() };
    let reference = Container::open(&fixture.container, &options).expect("open reference");
    let cache = ReadCache::new(reference.clone(), CacheOptions::default()).expect("cache");
    let fs = Arc::new(MountFs::new(cache, PosixPolicy, SystemTime::now()).expect("mountable"));
    let fuse_options = FuseOptions { threads: 4, fs_name: fixture.container.display().to_string() };
    let handle = mount(Arc::clone(&fs), &mountpoint, &fuse_options).expect("mount");
    assert!(mounted(&mountpoint), "the mount is listed in /proc/mounts");

    let mut root: Vec<_> = fs::read_dir(&mountpoint)
        .expect("list root")
        .map(|entry| entry.expect("entry").file_name().into_string().expect("UTF-8"))
        .collect();
    root.sort();
    assert_eq!(root, ["config.json", "empty.bin", "models"]);

    let mut python = true;
    for (name, content) in &fixture.files {
        let path = mountpoint.join(name);
        let meta = fs::metadata(&path).expect("stat");
        assert_eq!(meta.len(), u64::try_from(content.len()).expect("small"), "{name}");
        assert_eq!(meta.permissions().mode() & 0o777, 0o444, "{name}");
        let member = reference.member(name).expect("member");
        let mut expected = vec![0; content.len()];
        member.read_at(0, &mut expected).expect("reference read");
        assert_eq!(&expected, content, "{name}: the reference is the source");
        assert_eq!(fs::read(&path).expect("read through the mount"), expected, "{name}");

        // Positional reads at odd places, as a loader that seeks to tensors does.
        let file = fs::File::open(&path).expect("open");
        for (offset, len) in [(0_usize, 8_usize), (1_000_003, 300_000), (content.len() / 2, 4097)] {
            let offset = offset.min(content.len());
            let len = len.min(content.len() - offset);
            let mut got = vec![0; len];
            file.read_exact_at(&mut got, u64::try_from(offset).expect("small")).expect("positional read");
            assert_eq!(got, expected[offset..offset + len], "{name} at {offset}");
        }
        drop(file);

        // `default_permissions`: the kernel applies the mode bits, so a member is not executable.
        assert_eq!(access(&path, AccessFlags::R_OK), Ok(()), "{name}");
        assert_eq!(access(&path, AccessFlags::X_OK), Err(Errno::EACCES), "{name}");
        assert_eq!(access(&path, AccessFlags::W_OK), Err(Errno::EROFS), "{name}");

        let written = FileOptions::new().write(true).open(&path).expect_err("a write open is refused");
        assert_eq!(written.kind(), ErrorKind::ReadOnlyFilesystem, "{name}");

        if python {
            match read_through_mmap(&path) {
                Some(mapped) => assert_eq!(mapped, expected, "{name} through mmap"),
                None => {
                    eprintln!("mmap comparison skipped: python3 is not on PATH");
                    python = false;
                }
            }
        }
    }
    assert!(fs.stats().misses > 0, "reads went through the cache");
    assert_eq!(fs.totals().1, fixture.total_len());

    handle.unmount().expect("unmount");
    assert!(!mounted(&mountpoint), "no mount is left");
}

#[test]
fn an_unmount_detaches_at_once_and_serves_an_open_file_until_it_is_closed() {
    if let Some(reason) = fuse_unavailable() {
        eprintln!("skipped: {reason}");
        return;
    }
    let scratch = Scratch::new("fuse-lazy");
    let fixture = Fixture::standard(&scratch);
    let (mountpoint, _fs, handle) = mount_fixture(&scratch, &fixture);
    let (name, content) = &fixture.files[0];
    let file = fs::File::open(mountpoint.join(name)).expect("open");
    let unmounting = thread::spawn(move || handle.unmount());
    let deadline = Instant::now() + Duration::from_secs(20);
    while mounted(&mountpoint) {
        assert!(Instant::now() < deadline, "the mountpoint is detached");
        thread::sleep(Duration::from_millis(20));
    }
    // Detached, yet the session lives on for the open file.
    let mut tail = vec![0; 4096];
    let at = content.len() - tail.len();
    file.read_exact_at(&mut tail, u64::try_from(at).expect("small")).expect("read after the detach");
    assert_eq!(tail, content[at..]);
    assert!(!unmounting.is_finished(), "the unmount waits for the open file");
    drop(file);
    unmounting.join().expect("the unmount thread").expect("unmount");
    assert!(!mounted(&mountpoint), "no mount is left");
}
