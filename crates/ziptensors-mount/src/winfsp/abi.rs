//! File: winfsp/abi.rs
//! Purpose: the part of `WinFsp`'s binary interface (`winfsp-x64.dll`, v2.x) this backend uses, written
//! from the documented layouts: the structures it exchanges, the flag bits of the volume parameters,
//! the callback table and the types of the exports it calls.
//!
//! Key structures: [`VolumeParams`], [`VolumeInfo`], [`FileInfo`], [`OpenFileInfo`], [`DirInfo`],
//! [`FileSystem`], [`Interface`]; the `*Fn` export types.
//!
//! Implementation notes: no `#pragma pack` applies upstream, so every structure is natural `repr(C)`.
//! C bitfields of `UINT32` become one `u32` word with bit constants, allocated from the least
//! significant bit, as both MSVC and GCC do on Windows. Every size and offset the DLL relies on is
//! asserted at compile time below, so a wrong declaration fails the build on every Windows target.
//! `WinFsp`'s `FSP_API` has no `WINAPI`, so exports and callbacks are `extern "C"` (the same as
//! `"system"` on x64 and arm64). [`FileSystem`] is only the leading part of the DLL's object and is
//! never held by value. No code here runs; the file holds declarations only.

use std::ffi::c_void;
use std::mem::{align_of, offset_of, size_of};

/// `NTSTATUS`: 0 and positive values succeed, values with the high bit set fail.
pub(super) type NtStatus = i32;

/// The device a disk volume is created on (`FSP_FSCTL_DISK_DEVICE_NAME`). Directory mount points are
/// junctions, which Windows allows only to disk volumes.
pub(super) const DISK_DEVICE: &str = "WinFsp.Disk";

/// Bits of [`VolumeParams::flags`] (`FSP_FSCTL_VOLUME_PARAMS`, first bitfield word).
pub(super) mod flags {
    /// File names are compared with case (`CaseSensitiveSearch`). Off here.
    #[cfg(test)]
    pub(in super::super) const CASE_SENSITIVE_SEARCH: u32 = 1 << 0;
    /// The case of a stored name is kept (`CasePreservedNames`).
    pub(in super::super) const CASE_PRESERVED_NAMES: u32 = 1 << 1;
    /// Names are Unicode (`UnicodeOnDisk`).
    pub(in super::super) const UNICODE_ON_DISK: u32 = 1 << 2;
    /// The file system enforces security descriptors (`PersistentAcls`).
    pub(in super::super) const PERSISTENT_ACLS: u32 = 1 << 3;
    /// The volume reports `FILE_READ_ONLY_VOLUME` (`ReadOnlyVolume`). Reporting only: the `WinFsp` driver
    /// refuses nothing because of it, so the callbacks refuse writes themselves.
    pub(in super::super) const READ_ONLY_VOLUME: u32 = 1 << 9;
    /// Cleanup reaches user mode only for a modified or deleted file (`PostCleanupWhenModifiedOnly`).
    pub(in super::super) const POST_CLEANUP_WHEN_MODIFIED_ONLY: u32 = 1 << 10;
    /// The driver flushes and purges the cache of a file when its last handle is cleaned up
    /// (`FlushAndPurgeOnCleanup`), so a closed file does not linger on the standby list and its Close
    /// comes at once.
    pub(in super::super) const FLUSH_AND_PURGE_ON_CLEANUP: u32 = 1 << 14;
    /// File contexts are the `UserContext2` field (`UmFileContextIsUserContext2`). Off here.
    #[cfg(test)]
    pub(in super::super) const UM_FILE_CONTEXT_IS_USER_CONTEXT2: u32 = 1 << 16;
    /// File contexts are the full context (`UmFileContextIsFullContext`). Off here.
    #[cfg(test)]
    pub(in super::super) const UM_FILE_CONTEXT_IS_FULL_CONTEXT: u32 = 1 << 17;
}

/// `FSP_FSCTL_VOLUME_PARAMS`, version 1 (504 bytes): how the driver presents the volume.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct VolumeParams {
    /// 0 or the size of this structure, 504; 504 makes the driver read the version 1 fields.
    pub(super) version: u16,
    /// Bytes per sector.
    pub(super) sector_size: u16,
    /// Sectors per allocation unit.
    pub(super) sectors_per_allocation_unit: u16,
    /// Longest name component, in UTF-16 units.
    pub(super) max_component_length: u16,
    /// `FILETIME` of the volume's creation.
    pub(super) volume_creation_time: u64,
    /// Serial number reported to `GetVolumeInformation`.
    pub(super) volume_serial_number: u32,
    /// Deprecated transact timeout; 0 keeps the driver's default.
    pub(super) transact_timeout: u32,
    /// Pending IRP timeout in milliseconds; 0 keeps the driver's default.
    pub(super) irp_timeout: u32,
    /// Most pending IRPs; 0 keeps the driver's default.
    pub(super) irp_capacity: u32,
    /// Milliseconds the driver caches file information; `u32::MAX` enables the cache manager.
    pub(super) file_info_timeout: u32,
    /// The bits of [`flags`].
    pub(super) flags: u32,
    /// UNC prefix of a network volume, NUL-terminated; empty for a disk volume.
    pub(super) prefix: [u16; 192],
    /// File system name shown by `GetVolumeInformation`, NUL-terminated.
    pub(super) file_system_name: [u16; 16],
    /// Version 1 validity bits of the timeouts below; none set.
    pub(super) additional_flags: u32,
    /// Volume information timeout, valid only with its bit in `additional_flags`.
    pub(super) volume_info_timeout: u32,
    /// Directory information timeout, likewise.
    pub(super) dir_info_timeout: u32,
    /// Security timeout, likewise.
    pub(super) security_timeout: u32,
    /// Stream information timeout, likewise.
    pub(super) stream_info_timeout: u32,
    /// Extended attribute timeout, likewise.
    pub(super) ea_timeout: u32,
    /// Control code of a file system extension; 0.
    pub(super) fsext_control_code: u32,
    /// Reserved; 0.
    pub(super) reserved32: [u32; 1],
    /// Reserved; 0.
    pub(super) reserved64: [u64; 2],
}

/// `FSP_FSCTL_VOLUME_INFO`: sizes and label reported for the volume.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct VolumeInfo {
    /// Bytes of the volume.
    pub(super) total_size: u64,
    /// Free bytes.
    pub(super) free_size: u64,
    /// Length of the label in bytes, without a NUL.
    pub(super) volume_label_length: u16,
    /// The label, not NUL-terminated.
    pub(super) volume_label: [u16; 32],
}

/// `FSP_FSCTL_FILE_INFO`: what the driver knows of a file or directory.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct FileInfo {
    /// `FILE_ATTRIBUTE_*` bits.
    pub(super) file_attributes: u32,
    /// Reparse tag; 0.
    pub(super) reparse_tag: u32,
    /// Bytes allocated.
    pub(super) allocation_size: u64,
    /// Bytes of the file.
    pub(super) file_size: u64,
    /// `FILETIME` of creation.
    pub(super) creation_time: u64,
    /// `FILETIME` of the last access.
    pub(super) last_access_time: u64,
    /// `FILETIME` of the last write.
    pub(super) last_write_time: u64,
    /// `FILETIME` of the last change.
    pub(super) change_time: u64,
    /// File id.
    pub(super) index_number: u64,
    /// Unimplemented upstream; 0.
    pub(super) hard_links: u32,
    /// Size of the extended attributes; 0.
    pub(super) ea_size: u32,
}

/// `FSP_FSCTL_OPEN_FILE_INFO`: the structure an Open callback's `FileInfo` pointer is the first field
/// of. `normalized_name` points at a buffer of `normalized_name_size` bytes; a callback that writes
/// the stored-case path there and sets the size to its byte length gives the driver the file's real
/// name. Left unchanged, the size is larger than any path and the driver ignores the buffer.
#[repr(C)]
#[derive(Debug)]
pub(super) struct OpenFileInfo {
    /// The file information, written by every Open.
    pub(super) file_info: FileInfo,
    /// Buffer for the normalized name, UTF-16, no NUL needed.
    pub(super) normalized_name: *mut u16,
    /// In: capacity of `normalized_name` in bytes. Out: length of the name in bytes.
    pub(super) normalized_name_size: u16,
}

/// Longest normalized name the driver takes from an Open, in bytes (`FSP_FSCTL_TRANSACT_PATH_SIZEMAX`).
pub(super) const NORMALIZED_NAME_MAX_BYTES: usize = 1024 * 2;

/// `FSP_FSCTL_DIR_INFO` without its trailing name: one directory entry as `FspFileSystemAddDirInfo`
/// takes it. The name follows the structure directly, and `size` covers both.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct DirInfo {
    /// Bytes of this structure plus the name.
    pub(super) size: u16,
    /// The entry's information.
    pub(super) file_info: FileInfo,
    /// The `NextOffset`/`Padding` union, zero for a name-marker directory.
    pub(super) padding: [u8; 24],
}

/// Longest name component, in UTF-16 units, as `WindowsPolicy` enforces and the volume declares.
pub(super) const NAME_MAX_UNITS: usize = 255;

/// A [`DirInfo`] with room for the longest name, built on the stack for each entry.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct DirInfoBuf {
    /// The fixed part.
    pub(super) info: DirInfo,
    /// The name, UTF-16, not NUL-terminated; `info.size` says how much of it is used.
    pub(super) name: [u16; NAME_MAX_UNITS],
}

/// UTF-16 units of `FSP_FILE_SYSTEM::VolumeName` (`FSP_FSCTL_VOLUME_NAME_SIZEMAX`: 64 + 192 units).
pub(super) const VOLUME_NAME_UNITS: usize = 256;

/// Number of `FspFsctlTransact*Kind` values, the length of `FSP_FILE_SYSTEM::Operations`.
const TRANSACT_KIND_COUNT: usize = 22;

/// The leading fields of `FSP_FILE_SYSTEM`, the DLL's file system object, up to `DispatcherResult`.
/// The object is allocated, owned and freed by the DLL and is never held by value. Read or written here:
/// `user_context` (before the dispatcher starts), `volume_name` (set by `FspFileSystemCreate`, never
/// changed) and `dispatcher_result` (read only inside `DispatcherStopped`, on the thread that set it).
#[repr(C)]
#[derive(Debug)]
pub(super) struct FileSystem {
    /// Version of the object.
    pub(super) version: u16,
    /// Free for the file system: here the address of the backend's `Volume`.
    pub(super) user_context: *mut c_void,
    /// NT device name of the volume, `\Device\Volume{...}`, NUL-terminated.
    pub(super) volume_name: [u16; VOLUME_NAME_UNITS],
    pub(super) volume_handle: *mut c_void,
    pub(super) enter_operation: *mut c_void,
    pub(super) leave_operation: *mut c_void,
    pub(super) operations: [*mut c_void; TRANSACT_KIND_COUNT],
    pub(super) interface: *const Interface,
    pub(super) dispatcher_thread: *mut c_void,
    pub(super) dispatcher_thread_count: u32,
    /// The first failure of a dispatcher thread, 0 while none failed.
    pub(super) dispatcher_result: NtStatus,
}

/// `GetVolumeInfo(FileSystem, VolumeInfo)`.
pub(super) type GetVolumeInfoFn = unsafe extern "C" fn(*mut FileSystem, *mut VolumeInfo) -> NtStatus;
/// `SetVolumeLabel(FileSystem, VolumeLabel, VolumeInfo)`.
pub(super) type SetVolumeLabelFn = unsafe extern "C" fn(*mut FileSystem, *mut u16, *mut VolumeInfo) -> NtStatus;
/// `GetSecurityByName(FileSystem, FileName, PFileAttributes, SecurityDescriptor, PSecurityDescriptorSize)`.
pub(super) type GetSecurityByNameFn =
    unsafe extern "C" fn(*mut FileSystem, *mut u16, *mut u32, *mut c_void, *mut usize) -> NtStatus;
/// `Create(FileSystem, FileName, CreateOptions, GrantedAccess, FileAttributes, SecurityDescriptor,
/// AllocationSize, PFileContext, FileInfo)`.
pub(super) type CreateFn = unsafe extern "C" fn(
    *mut FileSystem,
    *mut u16,
    u32,
    u32,
    u32,
    *mut c_void,
    u64,
    *mut *mut c_void,
    *mut FileInfo,
) -> NtStatus;
/// `Open(FileSystem, FileName, CreateOptions, GrantedAccess, PFileContext, FileInfo)`.
pub(super) type OpenFn =
    unsafe extern "C" fn(*mut FileSystem, *mut u16, u32, u32, *mut *mut c_void, *mut FileInfo) -> NtStatus;
/// `Overwrite(FileSystem, FileContext, FileAttributes, ReplaceFileAttributes, AllocationSize, FileInfo)`.
pub(super) type OverwriteFn =
    unsafe extern "C" fn(*mut FileSystem, *mut c_void, u32, u8, u64, *mut FileInfo) -> NtStatus;
/// `Close(FileSystem, FileContext)`.
pub(super) type CloseFn = unsafe extern "C" fn(*mut FileSystem, *mut c_void);
/// `Read(FileSystem, FileContext, Buffer, Offset, Length, PBytesTransferred)`.
pub(super) type ReadFn =
    unsafe extern "C" fn(*mut FileSystem, *mut c_void, *mut c_void, u64, u32, *mut u32) -> NtStatus;
/// `Write(FileSystem, FileContext, Buffer, Offset, Length, WriteToEndOfFile, ConstrainedIo,
/// PBytesTransferred, FileInfo)`.
pub(super) type WriteFn = unsafe extern "C" fn(
    *mut FileSystem,
    *mut c_void,
    *mut c_void,
    u64,
    u32,
    u8,
    u8,
    *mut u32,
    *mut FileInfo,
) -> NtStatus;
/// `Flush(FileSystem, FileContext, FileInfo)`; a null context flushes the volume.
pub(super) type FlushFn = unsafe extern "C" fn(*mut FileSystem, *mut c_void, *mut FileInfo) -> NtStatus;
/// `GetFileInfo(FileSystem, FileContext, FileInfo)`.
pub(super) type GetFileInfoFn = unsafe extern "C" fn(*mut FileSystem, *mut c_void, *mut FileInfo) -> NtStatus;
/// `SetBasicInfo(FileSystem, FileContext, FileAttributes, CreationTime, LastAccessTime, LastWriteTime,
/// ChangeTime, FileInfo)`.
pub(super) type SetBasicInfoFn =
    unsafe extern "C" fn(*mut FileSystem, *mut c_void, u32, u64, u64, u64, u64, *mut FileInfo) -> NtStatus;
/// `SetFileSize(FileSystem, FileContext, NewSize, SetAllocationSize, FileInfo)`.
pub(super) type SetFileSizeFn = unsafe extern "C" fn(*mut FileSystem, *mut c_void, u64, u8, *mut FileInfo) -> NtStatus;
/// `CanDelete(FileSystem, FileContext, FileName)`.
pub(super) type CanDeleteFn = unsafe extern "C" fn(*mut FileSystem, *mut c_void, *mut u16) -> NtStatus;
/// `Rename(FileSystem, FileContext, FileName, NewFileName, ReplaceIfExists)`.
pub(super) type RenameFn = unsafe extern "C" fn(*mut FileSystem, *mut c_void, *mut u16, *mut u16, u8) -> NtStatus;
/// `GetSecurity(FileSystem, FileContext, SecurityDescriptor, PSecurityDescriptorSize)`.
pub(super) type GetSecurityFn = unsafe extern "C" fn(*mut FileSystem, *mut c_void, *mut c_void, *mut usize) -> NtStatus;
/// `SetSecurity(FileSystem, FileContext, SecurityInformation, ModificationDescriptor)`.
pub(super) type SetSecurityFn = unsafe extern "C" fn(*mut FileSystem, *mut c_void, u32, *mut c_void) -> NtStatus;
/// `ReadDirectory(FileSystem, FileContext, Pattern, Marker, Buffer, Length, PBytesTransferred)`.
pub(super) type ReadDirectoryFn =
    unsafe extern "C" fn(*mut FileSystem, *mut c_void, *mut u16, *mut u16, *mut c_void, u32, *mut u32) -> NtStatus;
/// `SetDelete(FileSystem, FileContext, FileName, DeleteFile)`.
pub(super) type SetDeleteFn = unsafe extern "C" fn(*mut FileSystem, *mut c_void, *mut u16, u8) -> NtStatus;
/// `DispatcherStopped(FileSystem, Normally)`.
pub(super) type DispatcherStoppedFn = unsafe extern "C" fn(*mut FileSystem, u8);
/// A slot this backend leaves empty; its real signature does not matter while it is `None`.
pub(super) type Unused = Option<unsafe extern "C" fn()>;

/// `FSP_FILE_SYSTEM_INTERFACE`: 64 function pointers, in this order. A `None` slot is an operation the
/// DLL answers itself or refuses; Create, Open and Overwrite must all be set, or every open fails.
#[repr(C)]
#[derive(Debug)]
pub(super) struct Interface {
    pub(super) get_volume_info: Option<GetVolumeInfoFn>,
    pub(super) set_volume_label: Option<SetVolumeLabelFn>,
    pub(super) get_security_by_name: Option<GetSecurityByNameFn>,
    pub(super) create: Option<CreateFn>,
    pub(super) open: Option<OpenFn>,
    pub(super) overwrite: Option<OverwriteFn>,
    pub(super) cleanup: Unused,
    pub(super) close: Option<CloseFn>,
    pub(super) read: Option<ReadFn>,
    pub(super) write: Option<WriteFn>,
    pub(super) flush: Option<FlushFn>,
    pub(super) get_file_info: Option<GetFileInfoFn>,
    pub(super) set_basic_info: Option<SetBasicInfoFn>,
    pub(super) set_file_size: Option<SetFileSizeFn>,
    pub(super) can_delete: Option<CanDeleteFn>,
    pub(super) rename: Option<RenameFn>,
    pub(super) get_security: Option<GetSecurityFn>,
    pub(super) set_security: Option<SetSecurityFn>,
    pub(super) read_directory: Option<ReadDirectoryFn>,
    /// Slots 19-25: reparse points, streams, `GetDirInfoByName`, `Control`.
    pub(super) reparse_stream_control: [Unused; 7],
    pub(super) set_delete: Option<SetDeleteFn>,
    /// Slots 27-31: `CreateEx`, `OverwriteEx`, `GetEa`, `SetEa`, `Obsolete0`.
    pub(super) extended: [Unused; 5],
    pub(super) dispatcher_stopped: Option<DispatcherStoppedFn>,
    /// Slots 33-63, reserved upstream.
    pub(super) reserved: [Unused; 31],
}

/// `FspFileSystemPreflight(DevicePath, MountPoint)`: `STATUS_OBJECT_NAME_COLLISION` when a drive letter
/// is defined (`QueryDosDevice`) or a directory exists, `STATUS_OBJECT_NAME_INVALID` when a directory
/// cannot be probed (a missing parent).
pub(super) type FspPreflightFn = unsafe extern "C" fn(*const u16, *const u16) -> NtStatus;
/// `FspFileSystemCreate(DevicePath, VolumeParams, Interface, PFileSystem)`.
pub(super) type FspCreateFn =
    unsafe extern "C" fn(*const u16, *const VolumeParams, *const Interface, *mut *mut FileSystem) -> NtStatus;
/// `FspFileSystemDelete(FileSystem)`: removes the mount point, closes the volume, frees the object.
pub(super) type FspDeleteFn = unsafe extern "C" fn(*mut FileSystem);
/// `FspFileSystemSetMountPoint(FileSystem, MountPoint)`; the DLL copies the string.
pub(super) type FspSetMountPointFn = unsafe extern "C" fn(*mut FileSystem, *const u16) -> NtStatus;
/// `FspFileSystemRemoveMountPoint(FileSystem)`; a no-op once removed.
pub(super) type FspRemoveMountPointFn = unsafe extern "C" fn(*mut FileSystem);
/// `FspFileSystemStartDispatcher(FileSystem, ThreadCount)`; 0 threads is the DLL's default.
pub(super) type FspStartDispatcherFn = unsafe extern "C" fn(*mut FileSystem, u32) -> NtStatus;
/// `FspFileSystemStopDispatcher(FileSystem)`: joins the dispatcher threads.
pub(super) type FspStopDispatcherFn = unsafe extern "C" fn(*mut FileSystem);
/// `FspFileSystemAddDirInfo(DirInfo, Buffer, Length, PBytesTransferred)`: appends one entry, or the end
/// marker for a null `DirInfo`. FALSE when the buffer is full.
pub(super) type FspAddDirInfoFn = unsafe extern "C" fn(*mut DirInfo, *mut c_void, u32, *mut u32) -> u8;
/// `FspVersion(PVersion)`: major version in the high 16 bits, minor in the low.
pub(super) type FspVersionFn = unsafe extern "C" fn(*mut u32) -> NtStatus;

// The layouts the DLL was compiled with (upstream static asserts and field offsets, v2.1).
const _: () = assert!(size_of::<VolumeParams>() == 504);
const _: () = assert!(offset_of!(VolumeParams, volume_creation_time) == 8);
const _: () = assert!(offset_of!(VolumeParams, volume_serial_number) == 16);
const _: () = assert!(offset_of!(VolumeParams, file_info_timeout) == 32);
const _: () = assert!(offset_of!(VolumeParams, flags) == 36);
const _: () = assert!(offset_of!(VolumeParams, prefix) == 40);
const _: () = assert!(offset_of!(VolumeParams, file_system_name) == 424);
const _: () = assert!(offset_of!(VolumeParams, additional_flags) == 456);
const _: () = assert!(offset_of!(VolumeParams, fsext_control_code) == 480);
const _: () = assert!(offset_of!(VolumeParams, reserved64) == 488);
const _: () = assert!(size_of::<VolumeInfo>() == 88);
const _: () = assert!(offset_of!(VolumeInfo, volume_label) == 18);
const _: () = assert!(size_of::<FileInfo>() == 72);
const _: () = assert!(offset_of!(FileInfo, index_number) == 56);
const _: () = assert!(offset_of!(OpenFileInfo, normalized_name) == 72);
const _: () = assert!(offset_of!(OpenFileInfo, normalized_name_size) == 80);
const _: () = assert!(size_of::<DirInfo>() == 104 && align_of::<DirInfo>() == 8);
const _: () = assert!(offset_of!(DirInfo, file_info) == 8);
const _: () = assert!(offset_of!(DirInfoBuf, name) == size_of::<DirInfo>());
const _: () = assert!(offset_of!(FileSystem, user_context) == 8);
const _: () = assert!(offset_of!(FileSystem, volume_name) == 16);
const _: () = assert!(offset_of!(FileSystem, interface) == 728);
const _: () = assert!(offset_of!(FileSystem, dispatcher_result) == 748);
const _: () = assert!(size_of::<Interface>() == 64 * size_of::<usize>());
const _: () = assert!(offset_of!(Interface, read_directory) == 18 * size_of::<usize>());
const _: () = assert!(offset_of!(Interface, set_delete) == 26 * size_of::<usize>());
const _: () = assert!(offset_of!(Interface, dispatcher_stopped) == 32 * size_of::<usize>());
