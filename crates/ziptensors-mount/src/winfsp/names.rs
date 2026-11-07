//! File: winfsp/names.rs
//! Purpose: the pure decisions of the `WinFsp` backend: how a `WinFsp` path finds a node, which
//! `NTSTATUS` a failure is, what the volume declares, where a directory listing resumes, which mount
//! point a path names and which members a mount point makes too long to open.
//!
//! Key functions: [`resolve`], [`status`], [`filetime`], [`wants_write`], [`resume_at`],
//! [`volume_params`], [`parse_mount_point`], [`too_long`], [`stored_path`].
//!
//! Implementation notes: no `unsafe` and no Win32 call (only `windows-sys` constants), so every rule
//! here is unit-tested, natively and under wine, without `WinFsp`. `callbacks.rs` holds the FFI that
//! applies them.

use std::path::{Component, Path, PathBuf, Prefix};
use std::time::SystemTime;

use windows_sys::Win32::Foundation::{
    STATUS_FILE_IS_A_DIRECTORY, STATUS_IO_DEVICE_ERROR, STATUS_MEDIA_WRITE_PROTECTED, STATUS_NOT_A_DIRECTORY,
    STATUS_OBJECT_NAME_NOT_FOUND, STATUS_OBJECT_PATH_NOT_FOUND,
};
use windows_sys::Win32::Storage::FileSystem::{
    DELETE, FILE_APPEND_DATA, FILE_DELETE_CHILD, FILE_WRITE_ATTRIBUTES, FILE_WRITE_DATA, FILE_WRITE_EA, WRITE_DAC,
    WRITE_OWNER,
};

use crate::error::FsError;
use crate::fs::MountFs;
use crate::policy::{NamePolicy, WindowsPolicy};
use crate::tree::{DirEntry, NodeId, NodeKind, Tree};

use super::abi::{NAME_MAX_UNITS, NtStatus, VolumeParams, flags};

/// File system name the volume reports (`GetVolumeInformation`), at most 15 UTF-16 units.
const FILE_SYSTEM_NAME: &str = "ziptensors";

/// 100 ns intervals between 1601-01-01 (the `FILETIME` epoch) and 1970-01-01.
const FILETIME_UNIX_EPOCH: u64 = 116_444_736_000_000_000;

/// Sector size the volume reports; with one sector per allocation unit, also its cluster size.
const SECTOR_SIZE: u16 = 4096;

/// Longest path a program that is not long-path aware opens, in UTF-16 units: `MAX_PATH` (260) less
/// the terminating NUL.
pub(super) const CLASSIC_PATH_MAX_UNITS: usize = 259;

/// Longest path with long paths enabled, in UTF-16 units: 32767 less the terminating NUL.
pub(super) const LONG_PATH_MAX_UNITS: usize = 32_766;

/// The access rights that change something: data, attributes, extended attributes, children,
/// deletion, the security descriptor or the owner. For a directory the same bits are
/// `FILE_ADD_FILE` and `FILE_ADD_SUBDIRECTORY`.
const WRITE_RIGHTS: u32 = FILE_WRITE_DATA
    | FILE_APPEND_DATA
    | FILE_WRITE_EA
    | FILE_DELETE_CHILD
    | FILE_WRITE_ATTRIBUTES
    | DELETE
    | WRITE_DAC
    | WRITE_OWNER;

/// The parts of a `WinFsp` path (`\`-rooted UTF-16, no NUL), empty parts skipped. `None` if a part is
/// not valid UTF-16 (an unpaired surrogate), which no member name can be.
#[must_use]
pub(super) fn split_path(name: &[u16]) -> Option<Vec<String>> {
    name.split(|unit| *unit == u16::from(b'\\'))
        .filter(|part| !part.is_empty())
        .map(|part| String::from_utf16(part).ok())
        .collect()
}

/// The node a `WinFsp` path names, resolved part by part with [`MountFs::lookup`], so with the tree's
/// case-insensitive keys.
///
/// # Errors
/// `STATUS_OBJECT_NAME_NOT_FOUND` when the last part is missing or the path is not UTF-16, and
/// `STATUS_OBJECT_PATH_NOT_FOUND` when an earlier part is missing or is a file.
pub(super) fn resolve(fs: &MountFs, name: &[u16]) -> Result<NodeId, NtStatus> {
    let parts = split_path(name).ok_or(STATUS_OBJECT_NAME_NOT_FOUND)?;
    let last = parts.len().saturating_sub(1);
    let mut node = NodeId::ROOT;
    for (at, part) in parts.iter().enumerate() {
        node = match fs.lookup(node, part) {
            Ok(attr) => attr.id,
            Err(FsError::NotFound) if at == last => return Err(STATUS_OBJECT_NAME_NOT_FOUND),
            Err(FsError::NotFound | FsError::NotDir) => return Err(STATUS_OBJECT_PATH_NOT_FOUND),
            Err(error @ (FsError::IsDir | FsError::ReadOnly | FsError::Io)) => return Err(status(error)),
        };
    }
    Ok(node)
}

/// The path of node `id` with every part as stored (the members' own case), `\`-rooted, as UTF-16.
/// `None` if the node or one of its ancestors is missing from its parent's listing.
#[must_use]
pub(super) fn stored_path(tree: &Tree, id: NodeId) -> Option<Vec<u16>> {
    let mut parts = Vec::new();
    let mut node = id;
    while node != NodeId::ROOT {
        let parent = tree.node(node)?.parent();
        let entries = tree.entries(parent)?;
        // Ids are given to the entries of a directory in name order, so they are sorted by id too.
        let at = entries.binary_search_by_key(&node, |entry| entry.id).ok()?;
        parts.push(entries.get(at)?.name.as_str());
        node = parent;
    }
    if parts.is_empty() {
        return Some(vec![u16::from(b'\\')]);
    }
    let mut path = Vec::new();
    for part in parts.iter().rev() {
        path.push(u16::from(b'\\'));
        path.extend(part.encode_utf16());
    }
    Some(path)
}

/// The `NTSTATUS` of a failed filesystem call.
#[must_use]
pub(super) fn status(error: FsError) -> NtStatus {
    match error {
        FsError::NotFound => STATUS_OBJECT_NAME_NOT_FOUND,
        FsError::NotDir => STATUS_NOT_A_DIRECTORY,
        FsError::IsDir => STATUS_FILE_IS_A_DIRECTORY,
        FsError::ReadOnly => STATUS_MEDIA_WRITE_PROTECTED,
        FsError::Io => STATUS_IO_DEVICE_ERROR,
    }
}

/// `time` as a `FILETIME` (100 ns intervals since 1601-01-01), 0 before 1601 and `u64::MAX` past the
/// year 60056.
#[must_use]
pub(super) fn filetime(time: SystemTime) -> u64 {
    let since_unix = match time.duration_since(SystemTime::UNIX_EPOCH) {
        Ok(after) => i128::try_from(after.as_nanos() / 100).unwrap_or(i128::MAX),
        Err(before) => -i128::try_from(before.duration().as_nanos() / 100).unwrap_or(i128::MAX),
    };
    let value = since_unix.saturating_add(i128::from(FILETIME_UNIX_EPOCH));
    u64::try_from(value.max(0)).unwrap_or(u64::MAX)
}

/// Whether an open granted `granted` access rights that change something. Must stay a check of the
/// granted rights: a handle with write rights could write through the cache without any callback. The access check of `WinFsp`
/// grants only what the security descriptor allows (read and execute), plus what the caller's
/// privileges add for rights it asked for, so a write bit here means a write was asked for.
#[must_use]
pub(super) fn wants_write(granted: u32) -> bool {
    granted & WRITE_RIGHTS != 0
}

/// Where a listing resumes, as an index into the sequence `.`, `..` (when `with_dots`), then
/// `entries`. `marker` is the last name the previous call returned, `None` at the start; a name that
/// is gone resumes at the first entry after it in name order. `entries` must be sorted by name.
#[must_use]
pub(super) fn resume_at(entries: &[DirEntry], with_dots: bool, marker: Option<&[u16]>) -> usize {
    let dots = if with_dots { 2 } else { 0 };
    let Some(marker) = marker else {
        return 0;
    };
    if with_dots && marker == [u16::from(b'.')] {
        return 1;
    }
    if with_dots && marker == [u16::from(b'.'), u16::from(b'.')] {
        return 2;
    }
    // A marker that is not UTF-16 was never returned by this listing: end it rather than restart it.
    let Ok(marker) = String::from_utf16(marker) else {
        return dots + entries.len();
    };
    let next = match entries.binary_search_by(|entry| entry.name.as_str().cmp(&marker)) {
        Ok(found) => found + 1,
        Err(insert) => insert,
    };
    dots + next
}

/// The volume parameters: read-only, case-insensitive, case-preserving, with security descriptors, the
/// cache manager on (`FileInfoTimeout` of `u32::MAX`, as nothing ever changes), the file cache purged
/// when the last handle of a file closes, timed by `created`.
#[must_use]
pub(super) fn volume_params(created: SystemTime) -> VolumeParams {
    let created = filetime(created);
    let mut file_system_name = [0_u16; 16];
    for (slot, unit) in file_system_name.iter_mut().zip(FILE_SYSTEM_NAME.encode_utf16()) {
        *slot = unit;
    }
    VolumeParams {
        // The structure is 504 bytes (asserted in `abi.rs`), so the size fits.
        version: u16::try_from(size_of::<VolumeParams>()).unwrap_or(0),
        sector_size: SECTOR_SIZE,
        sectors_per_allocation_unit: 1,
        max_component_length: u16::try_from(NAME_MAX_UNITS).unwrap_or(u16::MAX),
        volume_creation_time: created,
        // Two halves of the time folded: a serial number that differs between containers.
        volume_serial_number: u32::try_from(created & 0xFFFF_FFFF).unwrap_or(0) ^ u32::try_from(created >> 32).unwrap_or(0),
        transact_timeout: 0,
        irp_timeout: 0,
        irp_capacity: 0,
        file_info_timeout: u32::MAX,
        flags: flags::CASE_PRESERVED_NAMES
            | flags::UNICODE_ON_DISK
            | flags::PERSISTENT_ACLS
            | flags::READ_ONLY_VOLUME
            | flags::POST_CLEANUP_WHEN_MODIFIED_ONLY
            | flags::FLUSH_AND_PURGE_ON_CLEANUP,
        prefix: [0; 192],
        file_system_name,
        additional_flags: 0,
        volume_info_timeout: 0,
        dir_info_timeout: 0,
        security_timeout: 0,
        stream_info_timeout: 0,
        ea_timeout: 0,
        fsext_control_code: 0,
        reserved32: [0],
        reserved64: [0; 2],
    }
}

/// Where a mount goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MountPoint {
    /// A drive letter, upper case: the volume appears as `X:`.
    Drive(char),
    /// An absolute directory that does not exist yet: `WinFsp` creates it as a junction to the volume
    /// and removes it at unmount. Its parent must exist.
    Directory(PathBuf),
}

impl MountPoint {
    /// The string `WinFsp` takes: `X:` for a drive, the absolute path for a directory.
    #[must_use]
    pub(super) fn to_winfsp(&self) -> Vec<u16> {
        match self {
            Self::Drive(letter) => [*letter, ':'].iter().map(|c| u16::try_from(u32::from(*c)).unwrap_or(0)).collect(),
            Self::Directory(path) => {
                use std::os::windows::ffi::OsStrExt;
                path.as_os_str().encode_wide().collect()
            }
        }
    }

    /// Length of the root every member path starts from, in UTF-16 units, without the separator after
    /// it: 2 for `X:`, the directory's length otherwise.
    #[must_use]
    pub(super) fn root_units(&self) -> usize {
        match self {
            Self::Drive(_) => 2,
            Self::Directory(path) => {
                use std::os::windows::ffi::OsStrExt;
                path.as_os_str().encode_wide().count()
            }
        }
    }

    /// The mount point as a path, for messages: `X:\` for a drive.
    #[must_use]
    pub fn path(&self) -> PathBuf {
        match self {
            Self::Drive(letter) => PathBuf::from(format!("{letter}:\\")),
            Self::Directory(path) => path.clone(),
        }
    }
}

/// Why a path cannot be a mount point; the message of `WinFspError::MountPoint`.
pub(super) const NOT_LOCAL: &str =
    "UNC, device and verbatim (\\\\?\\) paths are not supported; give a drive letter such as X: or a local directory that does not exist yet";

/// Which mount point `path` names: `X:`, `x:`, `X:\` and `X:/` are the drive (`WinFsp` takes a drive
/// only as exactly `X:`), and anything else is a directory, made absolute against `current` when
/// relative.
///
/// # Errors
/// [`NOT_LOCAL`] for a UNC, device or verbatim path: `WinFsp` would take `\\?\` and `\\.\` forms as
/// mount manager points, and Windows allows no junction to them.
pub(super) fn parse_mount_point(path: &Path, current: &Path) -> Result<MountPoint, &'static str> {
    let mut components = path.components();
    let first = components.next();
    let rest: Vec<_> = components.collect();
    match first {
        Some(Component::Prefix(prefix)) => match prefix.kind() {
            Prefix::Disk(letter) => {
                let at_root = matches!(rest.as_slice(), [] | [Component::RootDir]);
                if at_root {
                    return Ok(MountPoint::Drive(char::from(letter.to_ascii_uppercase())));
                }
                if rest.first() == Some(&Component::RootDir) {
                    return Ok(MountPoint::Directory(path.to_owned()));
                }
                // `C:dir` is relative to the current directory of drive C; only `current` is known, so
                // such a path is refused rather than guessed.
                Err("a drive-relative path such as C:dir is ambiguous; give C:\\dir")
            }
            Prefix::Verbatim(_)
            | Prefix::VerbatimUNC(..)
            | Prefix::VerbatimDisk(_)
            | Prefix::DeviceNS(_)
            | Prefix::UNC(..) => Err(NOT_LOCAL),
        },
        // `\dir` is rooted on the current drive.
        Some(Component::RootDir) => match current.components().next() {
            Some(Component::Prefix(prefix)) if matches!(prefix.kind(), Prefix::Disk(_)) => {
                let mut absolute = PathBuf::from(prefix.as_os_str());
                absolute.push(path);
                Ok(MountPoint::Directory(absolute))
            }
            _ => Err(NOT_LOCAL),
        },
        Some(Component::CurDir | Component::ParentDir | Component::Normal(_)) => {
            parse_mount_point(&current.join(path), current).map_err(|_| NOT_LOCAL)
        }
        None => Err("the mount point is empty"),
    }
}

/// Checks a directory mount point after `std::path::absolute` normalized it: it must still be a local
/// absolute path, `X:\` followed by at least one part. `GetFullPathNameW` turns a reserved device name
/// into a device path (`C:\CON` becomes `\\.\CON`) and `C:\..` into the drive root, so the class
/// checked before normalization does not hold after it.
///
/// # Errors
/// The reason, for `WinFspError::MountPoint`.
pub(super) fn check_absolute(path: &Path) -> Result<(), &'static str> {
    let mut components = path.components();
    let disk = matches!(components.next(), Some(Component::Prefix(prefix)) if matches!(prefix.kind(), Prefix::Disk(_)));
    if !disk {
        return Err(DEVICE_OR_REMOTE);
    }
    if components.next() != Some(Component::RootDir) {
        return Err(DEVICE_OR_REMOTE);
    }
    let parts: Vec<_> = components.filter_map(|component| match component {
        Component::Normal(part) => Some(part),
        Component::Prefix(_) | Component::RootDir | Component::CurDir | Component::ParentDir => None,
    }).collect();
    if parts.is_empty() {
        return Err("it is the root of a drive; give the drive letter alone, such as X:, to mount on a drive");
    }
    // Older Windows turn `C:\CON` into a device path; Windows 11 keeps it a path, which some programs
    // still open as the device. A part Windows refuses for any reason (`CON:x` is a device name with a
    // stream suffix, refused first for its `:`) cannot be a directory either, so every refusal counts.
    let policy = WindowsPolicy::default();
    if parts.iter().any(|part| part.to_str().is_none_or(|part| policy.check_part(part).is_err())) {
        return Err(INVALID_PART);
    }
    Ok(())
}

/// Why a mount point is refused when a part of it is no valid Windows name.
pub(super) const INVALID_PART: &str = "a part of it is not a valid Windows name (a device name such as CON or COM1, a character such as : or ?, or a trailing dot or space); give a drive letter such as X: or a local directory that does not exist yet";

/// Why a normalized mount point is refused when it left the local-disk class.
pub(super) const DEVICE_OR_REMOTE: &str =
    "it names a device (such as CON or COM1) or is not on a local drive; give a drive letter such as X: or a local directory that does not exist yet";

/// The members whose full path under a mount root of `root_units` UTF-16 units (plus one separator)
/// exceeds `limit` units, `/` separated, in name order.
#[must_use]
pub(super) fn too_long(tree: &Tree, root_units: usize, limit: usize) -> Vec<String> {
    let mut long = Vec::new();
    let mut pending = vec![(NodeId::ROOT, String::new(), root_units)];
    while let Some((dir, prefix, units)) = pending.pop() {
        for entry in tree.entries(dir).unwrap_or_default() {
            let name = if prefix.is_empty() { entry.name.clone() } else { format!("{prefix}/{}", entry.name) };
            let length = units.saturating_add(1).saturating_add(entry.name.encode_utf16().count());
            match entry.kind {
                NodeKind::Dir => pending.push((entry.id, name, length)),
                NodeKind::File if length > limit => long.push(name),
                NodeKind::File => {}
            }
        }
    }
    long.sort();
    long
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::{Duration, SystemTime};

    use windows_sys::Win32::Foundation::{
        STATUS_FILE_IS_A_DIRECTORY, STATUS_IO_DEVICE_ERROR, STATUS_MEDIA_WRITE_PROTECTED, STATUS_NOT_A_DIRECTORY,
        STATUS_OBJECT_NAME_NOT_FOUND, STATUS_OBJECT_PATH_NOT_FOUND,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        DELETE, FILE_EXECUTE, FILE_GENERIC_EXECUTE, FILE_GENERIC_READ, FILE_READ_ATTRIBUTES, FILE_WRITE_DATA,
        READ_CONTROL, SYNCHRONIZE, WRITE_DAC,
    };

    use ziptensors_core::{CacheOptions, Container, OpenOptions, ReadCache};

    use super::{
        CLASSIC_PATH_MAX_UNITS, DEVICE_OR_REMOTE, INVALID_PART, MountPoint, NOT_LOCAL, check_absolute, parse_mount_point, resolve, resume_at, split_path, status,
        stored_path, too_long, volume_params, wants_write,
    };
    use crate::error::FsError;
    use crate::fs::MountFs;
    use crate::policy::WindowsPolicy;
    use crate::testkit::{Fixture, Scratch};
    use crate::tree::{DirEntry, NodeId, NodeKind, Tree};
    use crate::winfsp::abi::flags;

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().collect()
    }

    fn mount_fs(scratch: &Scratch) -> MountFs {
        let fixture = Fixture::build(
            scratch,
            &[("Models/Unet/w.safetensors", 4000), ("CONFIG.json", 0), ("models2.bin", 10)],
        );
        let container = Container::open(&fixture.container, &OpenOptions::default()).expect("open");
        let cache = ReadCache::new(container, CacheOptions::default()).expect("cache");
        MountFs::new(cache, WindowsPolicy::default(), SystemTime::UNIX_EPOCH).expect("mountable")
    }

    #[test]
    fn paths_split_on_backslashes_and_refuse_lone_surrogates() {
        assert_eq!(split_path(&wide("\\")), Some(vec![]));
        assert_eq!(split_path(&wide("")), Some(vec![]));
        assert_eq!(split_path(&wide("\\a\\\\B\\")), Some(vec!["a".to_owned(), "B".to_owned()]));
        assert_eq!(split_path(&[u16::from(b'\\'), 0xD800, u16::from(b'x')]), None);
    }

    #[test]
    fn paths_resolve_without_case_and_name_the_missing_part() {
        let scratch = Scratch::new("winfsp-resolve");
        let fs = mount_fs(&scratch);
        assert_eq!(resolve(&fs, &wide("\\")), Ok(NodeId::ROOT));
        let file = resolve(&fs, &wide("\\models\\UNET\\W.SAFETENSORS")).expect("case-insensitive");
        assert_eq!(resolve(&fs, &wide("\\Models\\Unet\\w.safetensors")), Ok(file));
        assert_eq!(resolve(&fs, &wide("\\\\Models\\\\Unet\\w.safetensors")), Ok(file));
        assert_eq!(resolve(&fs, &wide("\\config.json")).map(|_| ()), Ok(()));
        assert_eq!(resolve(&fs, &wide("\\missing")), Err(STATUS_OBJECT_NAME_NOT_FOUND));
        assert_eq!(resolve(&fs, &wide("\\missing\\x")), Err(STATUS_OBJECT_PATH_NOT_FOUND));
        assert_eq!(resolve(&fs, &wide("\\CONFIG.json\\x")), Err(STATUS_OBJECT_PATH_NOT_FOUND));
        assert_eq!(resolve(&fs, &[u16::from(b'\\'), 0xDC00]), Err(STATUS_OBJECT_NAME_NOT_FOUND));
        let path = stored_path(fs.tree(), file).expect("stored path");
        assert_eq!(String::from_utf16(&path).expect("utf-16"), "\\Models\\Unet\\w.safetensors");
        assert_eq!(stored_path(fs.tree(), NodeId::ROOT), Some(wide("\\")));
        assert_eq!(stored_path(fs.tree(), NodeId::from_raw(10_000)), None);
    }

    #[test]
    fn every_fs_error_has_its_status() {
        assert_eq!(status(FsError::NotFound), STATUS_OBJECT_NAME_NOT_FOUND);
        assert_eq!(status(FsError::NotDir), STATUS_NOT_A_DIRECTORY);
        assert_eq!(status(FsError::IsDir), STATUS_FILE_IS_A_DIRECTORY);
        assert_eq!(status(FsError::ReadOnly), STATUS_MEDIA_WRITE_PROTECTED);
        assert_eq!(status(FsError::Io), STATUS_IO_DEVICE_ERROR);
    }

    #[test]
    fn filetimes_count_from_1601() {
        assert_eq!(super::filetime(SystemTime::UNIX_EPOCH), 116_444_736_000_000_000);
        assert_eq!(super::filetime(SystemTime::UNIX_EPOCH + Duration::from_secs(1)), 116_444_736_010_000_000);
        let before = SystemTime::UNIX_EPOCH - Duration::from_secs(1);
        assert_eq!(super::filetime(before), 116_444_735_990_000_000);
    }

    #[test]
    fn only_rights_that_change_something_are_writes() {
        for read in [FILE_GENERIC_READ, FILE_GENERIC_EXECUTE, FILE_READ_ATTRIBUTES, READ_CONTROL, SYNCHRONIZE, FILE_EXECUTE] {
            assert!(!wants_write(read), "{read:#x}");
        }
        assert!(!wants_write(FILE_GENERIC_READ | FILE_GENERIC_EXECUTE));
        for write in [FILE_WRITE_DATA, DELETE, WRITE_DAC, FILE_GENERIC_READ | 0x4, 0x100, 0x10, 0x40, 0x8_0000] {
            assert!(wants_write(write), "{write:#x}");
        }
    }

    #[test]
    fn listings_resume_after_the_marker() {
        let entry = |name: &str| DirEntry { name: name.to_owned(), id: NodeId::ROOT, kind: NodeKind::File };
        // `!a` sorts before `.`, which must not confuse the dot markers.
        let entries = [entry("!a"), entry("b"), entry("d")];
        assert_eq!(resume_at(&entries, true, None), 0);
        assert_eq!(resume_at(&entries, true, Some(&wide("."))), 1);
        assert_eq!(resume_at(&entries, true, Some(&wide(".."))), 2);
        assert_eq!(resume_at(&entries, true, Some(&wide("!a"))), 3);
        assert_eq!(resume_at(&entries, true, Some(&wide("b"))), 4);
        assert_eq!(resume_at(&entries, true, Some(&wide("c"))), 4);
        assert_eq!(resume_at(&entries, true, Some(&wide("d"))), 5);
        assert_eq!(resume_at(&entries, true, Some(&wide("z"))), 5);
        assert_eq!(resume_at(&entries, false, None), 0);
        assert_eq!(resume_at(&entries, false, Some(&wide("!a"))), 1);
        assert_eq!(resume_at(&entries, false, Some(&wide("."))), 1);
        assert_eq!(resume_at(&entries, false, Some(&[0xD800])), 3);
        assert_eq!(resume_at(&[], false, Some(&wide("x"))), 0);
    }

    #[test]
    fn the_volume_is_read_only_case_insensitive_and_cached() {
        let params = volume_params(SystemTime::UNIX_EPOCH);
        assert_eq!(params.version, 504);
        assert_eq!(params.file_info_timeout, u32::MAX);
        assert_eq!(params.max_component_length, 255);
        assert_eq!(params.volume_creation_time, 116_444_736_000_000_000);
        assert_eq!(params.sector_size, 4096);
        assert_eq!(params.sectors_per_allocation_unit, 1);
        assert_ne!(params.flags & flags::READ_ONLY_VOLUME, 0);
        assert_ne!(params.flags & flags::CASE_PRESERVED_NAMES, 0);
        assert_ne!(params.flags & flags::PERSISTENT_ACLS, 0);
        assert_ne!(params.flags & flags::FLUSH_AND_PURGE_ON_CLEANUP, 0);
        assert_eq!(params.flags & flags::CASE_SENSITIVE_SEARCH, 0);
        // Neither alternative file context mode: the context is the node id itself.
        assert_eq!(params.flags & (flags::UM_FILE_CONTEXT_IS_USER_CONTEXT2 | flags::UM_FILE_CONTEXT_IS_FULL_CONTEXT), 0);
        assert_eq!(params.additional_flags, 0);
        let name: Vec<u16> = params.file_system_name.iter().copied().take_while(|unit| *unit != 0).collect();
        assert_eq!(String::from_utf16(&name).expect("utf-16"), "ziptensors");
        assert!(params.prefix.iter().all(|unit| *unit == 0));
    }

    #[test]
    fn mount_points_are_a_drive_or_an_absolute_directory() {
        let current = Path::new("C:\\work");
        for drive in ["x:", "X:", "x:\\", "X:/"] {
            assert_eq!(parse_mount_point(Path::new(drive), current), Ok(MountPoint::Drive('X')), "{drive}");
        }
        let directory = |path: &str| Ok(MountPoint::Directory(PathBuf::from(path)));
        assert_eq!(parse_mount_point(Path::new("D:\\m\\n"), current), directory("D:\\m\\n"));
        assert_eq!(parse_mount_point(Path::new("mnt"), current), directory("C:\\work\\mnt"));
        assert_eq!(parse_mount_point(Path::new("\\mnt"), current), directory("C:\\mnt"));
        assert!(parse_mount_point(Path::new("C:mnt"), current).is_err());
        for refused in ["\\\\?\\C:\\m", "\\\\.\\X:", "\\\\server\\share\\m"] {
            assert_eq!(parse_mount_point(Path::new(refused), current), Err(NOT_LOCAL), "{refused}");
        }
        assert!(parse_mount_point(Path::new(""), current).is_err());
        assert_eq!(MountPoint::Drive('X').to_winfsp(), wide("X:"));
        assert_eq!(MountPoint::Drive('X').root_units(), 2);
        assert_eq!(MountPoint::Directory(PathBuf::from("C:\\m")).root_units(), 4);
        assert_eq!(MountPoint::Drive('X').path(), PathBuf::from("X:\\"));
    }

    #[test]
    fn normalized_mount_points_must_stay_local_directories() {
        assert_eq!(check_absolute(Path::new("C:\\zt\\mnt")), Ok(()));
        for device in ["\\\\.\\CON", "\\\\.\\COM1\\", "\\\\?\\C:\\m", "\\\\server\\share\\m", "\\m"] {
            assert_eq!(check_absolute(Path::new(device)), Err(DEVICE_OR_REMOTE), "{device}");
        }
        for invalid in ["C:\\CON", "C:\\x\\com1", "C:\\x\\nul.txt\\y", "C:\\AUX .bin", "C:\\mnt\\CON:x", "C:\\m?", "C:\\dot."] {
            assert_eq!(check_absolute(Path::new(invalid)), Err(INVALID_PART), "{invalid}");
        }
        assert_eq!(check_absolute(Path::new("C:\\console\\com10")), Ok(()));
        assert!(check_absolute(Path::new("C:\\")).is_err());
        assert!(check_absolute(Path::new("C:")).is_err());
    }

    #[test]
    fn members_too_long_under_a_deep_mount_point_are_named() {
        let short = "a".repeat(10);
        let long = format!("{}/{}", "d".repeat(100), "f".repeat(100));
        let names = [(short.as_str(), 1_u64), (long.as_str(), 1)];
        let tree = Tree::build(names.iter().copied(), WindowsPolicy::default()).expect("tree");
        // `X:` + `\` + 201 units fits; a root of 58 units makes the long one 260 units.
        assert_eq!(too_long(&tree, 2, CLASSIC_PATH_MAX_UNITS), Vec::<String>::new());
        assert_eq!(too_long(&tree, 57, CLASSIC_PATH_MAX_UNITS), Vec::<String>::new());
        assert_eq!(too_long(&tree, 58, CLASSIC_PATH_MAX_UNITS), vec![long.clone()]);
        assert_eq!(too_long(&tree, 300, CLASSIC_PATH_MAX_UNITS), vec![short, long]);
    }
}
