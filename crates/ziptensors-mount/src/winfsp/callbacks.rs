//! File: winfsp/callbacks.rs
//! Purpose: the `extern "C"` operations `WinFsp` calls on its dispatcher threads, and the table that
//! hands them to it ([`INTERFACE`]).
//!
//! Main responsibilities:
//! - answer volume information, security, open, close, get-file-info, flush, read and read-directory
//!   from the `Volume` behind the file system's `UserContext`;
//! - refuse every change (create, overwrite, write, set-info, set-size, delete, rename, set-security,
//!   set-label) with `STATUS_MEDIA_WRITE_PROTECTED`, and refuse an open that was granted a write right;
//! - count opened and closed files, and report a dispatcher the driver stopped on its own.
//!
//! Key structures: [`INTERFACE`].
//!
//! Implementation notes:
//! - Every body runs inside [`guarded`]: a panic must not unwind into the C caller (Rust aborts the
//!   process on it), so it becomes `STATUS_INTERNAL_ERROR` and an error log.
//! - The file context is the node id itself (`WinFsp`'s default mode, ids start at 1, so a context is
//!   never null). Opens hold no other state; a directory listing resumes from `WinFsp`'s marker.
//! - Create, Open and Overwrite must all be present, or `WinFsp` fails every open; Create and Overwrite
//!   are present to refuse.
//! - The decisions (path resolution, status mapping, resume position) are the pure functions of
//!   `names.rs`; this file only moves values across the FFI boundary.

#![allow(unsafe_code, reason = "the callbacks WinFsp calls through raw pointers")]
#![warn(clippy::undocumented_unsafe_blocks)]

use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};
use std::ptr;

use windows_sys::Win32::Foundation::{
    STATUS_END_OF_FILE, STATUS_INTERNAL_ERROR, STATUS_MEDIA_WRITE_PROTECTED, STATUS_SUCCESS,
};
use windows_sys::Win32::Storage::FileSystem::{FILE_ATTRIBUTE_DIRECTORY, FILE_ATTRIBUTE_READONLY};

use crate::fs::Attr;
use crate::tree::{NodeId, NodeKind};

use super::abi::{
    DirInfo, DirInfoBuf, FileInfo, FileSystem, Interface, NAME_MAX_UNITS, NORMALIZED_NAME_MAX_BYTES, NtStatus,
    OpenFileInfo, VolumeInfo,
};
use super::names::{filetime, resolve, resume_at, status, stored_path, wants_write};
use super::volume::Volume;

/// Label of every volume.
const VOLUME_LABEL: &str = "ziptensors";

/// Largest piece one read asks of `MountFs::read`: a read of up to 4 GiB arrives in one call, and
/// pieces bound the memory a dispatcher thread holds at once.
const READ_PIECE: usize = 4 << 20;

/// Allocation granularity reported for files: the volume's sector size.
const ALLOCATION_UNIT: u64 = 4096;

/// The operations of a mount. A `None` slot is answered by `WinFsp` itself or refused by it.
pub(super) static INTERFACE: Interface = Interface {
    get_volume_info: Some(get_volume_info),
    set_volume_label: Some(set_volume_label),
    get_security_by_name: Some(get_security_by_name),
    create: Some(create),
    open: Some(open),
    overwrite: Some(overwrite),
    cleanup: None,
    close: Some(close),
    read: Some(read),
    write: Some(write),
    flush: Some(flush),
    get_file_info: Some(get_file_info),
    set_basic_info: Some(set_basic_info),
    set_file_size: Some(set_file_size),
    can_delete: Some(can_delete),
    rename: Some(rename),
    get_security: Some(get_security),
    set_security: Some(set_security),
    read_directory: Some(read_directory),
    reparse_stream_control: [None; 7],
    set_delete: Some(set_delete),
    extended: [None; 5],
    dispatcher_stopped: Some(dispatcher_stopped),
    reserved: [None; 31],
};

/// Runs `body`, turning a panic into `fallback` and an error log naming `operation`.
fn guarded<T>(operation: &'static str, fallback: T, body: impl FnOnce() -> T) -> T {
    // The volume's state is counters behind a mutex whose poison is ignored, so a panic leaves nothing
    // half changed that a later call could misread.
    panic::catch_unwind(AssertUnwindSafe(body)).unwrap_or_else(|_| {
        tracing::error!(operation, "a WinFsp callback panicked");
        fallback
    })
}

/// The volume of `file_system`.
///
/// # Safety
/// `file_system` must be a file system created by `mount`, whose `UserContext` was set to the address
/// of a `Volume` that outlives every callback (the `MountHandle` deletes the file system, which joins
/// the dispatcher, before it drops the volume).
unsafe fn volume<'a>(file_system: *mut FileSystem) -> &'a Volume {
    // SAFETY: by this function's contract.
    unsafe { &*(*file_system).user_context.cast_const().cast::<Volume>() }
}

/// The node a file context names.
fn node(context: *mut c_void) -> NodeId {
    NodeId::from_raw(u64::try_from(context.addr()).unwrap_or(0))
}

/// The UTF-16 units of a NUL-terminated string from `WinFsp`, empty for null.
///
/// # Safety
/// `text`, when not null, must point to a NUL-terminated string that outlives `'a`.
unsafe fn units<'a>(text: *const u16) -> &'a [u16] {
    if text.is_null() {
        return &[];
    }
    let mut len = 0;
    // SAFETY: by this function's contract the string is NUL-terminated, so every unit read up to and
    // including the NUL is inside it.
    while unsafe { *text.add(len) } != 0 {
        len += 1;
    }
    // SAFETY: the `len` units before the NUL, alive for `'a` by this function's contract.
    unsafe { std::slice::from_raw_parts(text, len) }
}

/// The `WinFsp` information of a node.
fn file_info(attr: &Attr) -> FileInfo {
    let time = filetime(attr.modified);
    let (file_attributes, allocation_size) = match attr.kind {
        NodeKind::Dir => (FILE_ATTRIBUTE_DIRECTORY, 0),
        NodeKind::File => (FILE_ATTRIBUTE_READONLY, attr.size.div_ceil(ALLOCATION_UNIT).saturating_mul(ALLOCATION_UNIT)),
    };
    FileInfo {
        file_attributes,
        reparse_tag: 0,
        allocation_size,
        file_size: attr.size,
        creation_time: time,
        last_access_time: time,
        last_write_time: time,
        change_time: time,
        index_number: attr.id.get(),
        hard_links: 0,
        ea_size: 0,
    }
}

/// Writes the information of node `id` to `out`.
///
/// # Safety
/// `out` must be valid for a write of one [`FileInfo`].
unsafe fn put_info(volume: &Volume, id: NodeId, out: *mut FileInfo) -> NtStatus {
    match volume.fs.attr(id) {
        Ok(attr) => {
            // SAFETY: by this function's contract.
            unsafe { out.write(file_info(&attr)) };
            STATUS_SUCCESS
        }
        Err(error) => status(error),
    }
}

unsafe extern "C" fn get_volume_info(file_system: *mut FileSystem, out: *mut VolumeInfo) -> NtStatus {
    guarded("GetVolumeInfo", STATUS_INTERNAL_ERROR, || {
        // SAFETY: WinFsp passes its own file system and a writable `VolumeInfo`.
        let volume = unsafe { volume(file_system) };
        let mut label = [0_u16; 32];
        let mut len = 0_u16;
        for (slot, unit) in label.iter_mut().zip(VOLUME_LABEL.encode_utf16()) {
            *slot = unit;
            len += 2;
        }
        let (_, total) = volume.fs.totals();
        let info = VolumeInfo { total_size: total, free_size: 0, volume_label_length: len, volume_label: label };
        // SAFETY: as above.
        unsafe { out.write(info) };
        STATUS_SUCCESS
    })
}

unsafe extern "C" fn get_security_by_name(
    file_system: *mut FileSystem,
    file_name: *mut u16,
    attributes: *mut u32,
    descriptor: *mut c_void,
    descriptor_size: *mut usize,
) -> NtStatus {
    guarded("GetSecurityByName", STATUS_INTERNAL_ERROR, || {
        // SAFETY: WinFsp passes its own file system and a NUL-terminated name valid for the call.
        let (volume, name) = unsafe { (volume(file_system), units(file_name)) };
        let id = match resolve(&volume.fs, name) {
            Ok(id) => id,
            Err(failure) => return failure,
        };
        if !attributes.is_null() {
            let attr = match volume.fs.attr(id) {
                Ok(attr) => attr,
                Err(error) => return status(error),
            };
            // SAFETY: a non-null `attributes` is writable (WinFsp's contract for this call).
            unsafe { attributes.write(file_info(&attr).file_attributes) };
        }
        // SAFETY: WinFsp's contract: `descriptor_size`, when not null, holds the capacity of
        // `descriptor`, which is writable for that many bytes when not null.
        unsafe { volume.security.write_to(descriptor, descriptor_size) }
    })
}

unsafe extern "C" fn create(
    _file_system: *mut FileSystem,
    _file_name: *mut u16,
    _create_options: u32,
    _granted_access: u32,
    _file_attributes: u32,
    _security_descriptor: *mut c_void,
    _allocation_size: u64,
    _context: *mut *mut c_void,
    _info: *mut FileInfo,
) -> NtStatus {
    STATUS_MEDIA_WRITE_PROTECTED
}

unsafe extern "C" fn open(
    file_system: *mut FileSystem,
    file_name: *mut u16,
    _create_options: u32,
    granted_access: u32,
    context: *mut *mut c_void,
    info: *mut FileInfo,
) -> NtStatus {
    guarded("Open", STATUS_INTERNAL_ERROR, || {
        // SAFETY: WinFsp passes its own file system and a NUL-terminated name valid for the call.
        let (volume, name) = unsafe { (volume(file_system), units(file_name)) };
        let id = match resolve(&volume.fs, name) {
            Ok(id) => id,
            Err(failure) => return failure,
        };
        if wants_write(granted_access) {
            return STATUS_MEDIA_WRITE_PROTECTED;
        }
        let attr = match volume.fs.attr(id) {
            Ok(attr) => attr,
            Err(error) => return status(error),
        };
        let Ok(raw) = usize::try_from(id.get()) else {
            return STATUS_INTERNAL_ERROR;
        };
        // SAFETY: `info` points at the `file_info` field, the first, of the `OpenFileInfo` WinFsp
        // passes to Open, whose name buffer holds `normalized_name_size` bytes.
        unsafe {
            info.write(file_info(&attr));
            write_normalized_name(volume, id, info.cast::<OpenFileInfo>());
            context.write(ptr::without_provenance_mut(raw));
        }
        if attr.kind == NodeKind::File {
            volume.file_opened(id);
        }
        STATUS_SUCCESS
    })
}

/// Gives `WinFsp` the stored-case path of `id`, so that a name opened in another case is reported as
/// stored. Skipped when it does not fit; `WinFsp` then keeps the name as opened.
///
/// # Safety
/// `open_info` must be the `OpenFileInfo` of an Open call.
unsafe fn write_normalized_name(volume: &Volume, id: NodeId, open_info: *mut OpenFileInfo) {
    let Some(path) = stored_path(volume.fs.tree(), id) else {
        return;
    };
    let bytes = path.len().saturating_mul(2);
    // SAFETY: by this function's contract.
    let (buffer, capacity) = unsafe { ((*open_info).normalized_name, usize::from((*open_info).normalized_name_size)) };
    let Ok(size) = u16::try_from(bytes) else {
        return;
    };
    if buffer.is_null() || bytes > capacity || bytes > NORMALIZED_NAME_MAX_BYTES {
        return;
    }
    // SAFETY: the buffer holds `capacity` bytes, at least `bytes`, and is WinFsp's, not ours.
    unsafe {
        ptr::copy_nonoverlapping(path.as_ptr(), buffer, path.len());
        (*open_info).normalized_name_size = size;
    }
}

unsafe extern "C" fn overwrite(
    _file_system: *mut FileSystem,
    _context: *mut c_void,
    _file_attributes: u32,
    _replace_file_attributes: u8,
    _allocation_size: u64,
    _info: *mut FileInfo,
) -> NtStatus {
    STATUS_MEDIA_WRITE_PROTECTED
}

unsafe extern "C" fn close(file_system: *mut FileSystem, context: *mut c_void) {
    guarded("Close", (), || {
        // SAFETY: WinFsp passes its own file system.
        let volume = unsafe { volume(file_system) };
        match volume.fs.attr(node(context)) {
            Ok(attr) if attr.kind == NodeKind::File => volume.file_closed(attr.id),
            Ok(_) => {}
            Err(error) => tracing::error!(context = context.addr(), %error, "WinFsp closed an unknown file context"),
        }
    });
}

unsafe extern "C" fn read(
    file_system: *mut FileSystem,
    context: *mut c_void,
    buffer: *mut c_void,
    offset: u64,
    length: u32,
    transferred: *mut u32,
) -> NtStatus {
    guarded("Read", STATUS_INTERNAL_ERROR, || {
        // SAFETY: WinFsp passes its own file system.
        let volume = unsafe { volume(file_system) };
        let id = node(context);
        let size = match volume.fs.attr(id) {
            Ok(attr) => attr.size,
            Err(error) => return status(error),
        };
        if offset >= size {
            return STATUS_END_OF_FILE;
        }
        let Ok(length) = usize::try_from(length) else {
            return STATUS_INTERNAL_ERROR;
        };
        let mut done = 0_usize;
        while done < length {
            let at = offset.saturating_add(u64::try_from(done).unwrap_or(u64::MAX));
            let bytes = match volume.fs.read(id, at, (length - done).min(READ_PIECE)) {
                Ok(bytes) => bytes,
                // Already logged with member, offset and code by `MountFs::read`.
                Err(error) => return status(error),
            };
            if bytes.is_empty() {
                break;
            }
            // SAFETY: WinFsp's buffer holds `length` bytes; `done + bytes.len() <= length` because each
            // piece asks for at most `length - done`. The source is our own vector.
            unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.cast::<u8>().add(done), bytes.len()) };
            done += bytes.len();
        }
        // `done <= length`, a `u32`.
        let done = u32::try_from(done).unwrap_or(u32::MAX);
        // SAFETY: WinFsp passes a writable count.
        unsafe { transferred.write(done) };
        STATUS_SUCCESS
    })
}

unsafe extern "C" fn write(
    _file_system: *mut FileSystem,
    _context: *mut c_void,
    _buffer: *mut c_void,
    _offset: u64,
    _length: u32,
    _write_to_end_of_file: u8,
    _constrained_io: u8,
    _transferred: *mut u32,
    _info: *mut FileInfo,
) -> NtStatus {
    STATUS_MEDIA_WRITE_PROTECTED
}

unsafe extern "C" fn flush(file_system: *mut FileSystem, context: *mut c_void, info: *mut FileInfo) -> NtStatus {
    guarded("Flush", STATUS_INTERNAL_ERROR, || {
        // A null context flushes the volume: nothing was ever written.
        if context.is_null() {
            return STATUS_SUCCESS;
        }
        // SAFETY: WinFsp passes its own file system and a writable `FileInfo` for a file flush.
        unsafe { put_info(volume(file_system), node(context), info) }
    })
}

unsafe extern "C" fn get_file_info(file_system: *mut FileSystem, context: *mut c_void, info: *mut FileInfo) -> NtStatus {
    guarded("GetFileInfo", STATUS_INTERNAL_ERROR, || {
        // SAFETY: WinFsp passes its own file system and a writable `FileInfo`.
        unsafe { put_info(volume(file_system), node(context), info) }
    })
}

unsafe extern "C" fn set_volume_label(_file_system: *mut FileSystem, _label: *mut u16, _info: *mut VolumeInfo) -> NtStatus {
    STATUS_MEDIA_WRITE_PROTECTED
}

unsafe extern "C" fn set_basic_info(
    _file_system: *mut FileSystem,
    _context: *mut c_void,
    _file_attributes: u32,
    _creation_time: u64,
    _last_access_time: u64,
    _last_write_time: u64,
    _change_time: u64,
    _info: *mut FileInfo,
) -> NtStatus {
    STATUS_MEDIA_WRITE_PROTECTED
}

unsafe extern "C" fn set_file_size(
    _file_system: *mut FileSystem,
    _context: *mut c_void,
    _new_size: u64,
    _set_allocation_size: u8,
    _info: *mut FileInfo,
) -> NtStatus {
    STATUS_MEDIA_WRITE_PROTECTED
}

unsafe extern "C" fn can_delete(_file_system: *mut FileSystem, _context: *mut c_void, _file_name: *mut u16) -> NtStatus {
    STATUS_MEDIA_WRITE_PROTECTED
}

unsafe extern "C" fn rename(
    _file_system: *mut FileSystem,
    _context: *mut c_void,
    _file_name: *mut u16,
    _new_file_name: *mut u16,
    _replace_if_exists: u8,
) -> NtStatus {
    STATUS_MEDIA_WRITE_PROTECTED
}

unsafe extern "C" fn get_security(
    file_system: *mut FileSystem,
    context: *mut c_void,
    descriptor: *mut c_void,
    descriptor_size: *mut usize,
) -> NtStatus {
    guarded("GetSecurity", STATUS_INTERNAL_ERROR, || {
        // SAFETY: WinFsp passes its own file system.
        let volume = unsafe { volume(file_system) };
        if let Err(error) = volume.fs.attr(node(context)) {
            return status(error);
        }
        // SAFETY: WinFsp's contract: `descriptor_size` holds the capacity of `descriptor`.
        unsafe { volume.security.write_to(descriptor, descriptor_size) }
    })
}

unsafe extern "C" fn set_security(
    _file_system: *mut FileSystem,
    _context: *mut c_void,
    _security_information: u32,
    _modification: *mut c_void,
) -> NtStatus {
    STATUS_MEDIA_WRITE_PROTECTED
}

unsafe extern "C" fn read_directory(
    file_system: *mut FileSystem,
    context: *mut c_void,
    _pattern: *mut u16,
    marker: *mut u16,
    buffer: *mut c_void,
    length: u32,
    transferred: *mut u32,
) -> NtStatus {
    guarded("ReadDirectory", STATUS_INTERNAL_ERROR, || {
        // SAFETY: WinFsp passes its own file system and, when not null, a NUL-terminated marker.
        let (volume, marker) = unsafe { (volume(file_system), (!marker.is_null()).then(|| units(marker))) };
        let id = node(context);
        let (entries, parent) = match (volume.fs.read_dir(id), volume.fs.parent(id)) {
            (Ok(entries), Ok(parent)) => (entries, parent),
            (Err(error), _) | (_, Err(error)) => return status(error),
        };
        // The root has no `.` and `..`, as on NTFS.
        let with_dots = id != NodeId::ROOT;
        let dots: &[(NodeId, &str)] = if with_dots { &[(id, "."), (parent, "..")] } else { &[] };
        let listed = dots.iter().copied().chain(entries.iter().map(|entry| (entry.id, entry.name.as_str())));
        for (entry, name) in listed.skip(resume_at(entries, with_dots, marker)) {
            let attr = match volume.fs.attr(entry) {
                Ok(attr) => attr,
                Err(error) => return status(error),
            };
            let Some(mut item) = dir_info(&attr, name) else {
                tracing::error!(name, "a name longer than 255 UTF-16 units is left out of a listing");
                continue;
            };
            // SAFETY: `item` is a complete entry; WinFsp's buffer holds `length` bytes and
            // `transferred` counts what it already holds.
            let added = unsafe { (volume.add_dir_info)(&raw mut item.info, buffer, length, transferred) };
            if added == 0 {
                // The buffer is full; WinFsp asks again with the last name as the marker.
                return STATUS_SUCCESS;
            }
        }
        // SAFETY: a null entry appends the end marker, within the same buffer.
        unsafe { (volume.add_dir_info)(ptr::null_mut(), buffer, length, transferred) };
        STATUS_SUCCESS
    })
}

/// The directory entry of `attr` called `name`, `None` if the name is longer than an entry holds.
fn dir_info(attr: &Attr, name: &str) -> Option<DirInfoBuf> {
    let mut item = DirInfoBuf {
        info: DirInfo { size: 0, file_info: file_info(attr), padding: [0; 24] },
        name: [0; NAME_MAX_UNITS],
    };
    let mut len = 0_usize;
    for unit in name.encode_utf16() {
        *item.name.get_mut(len)? = unit;
        len += 1;
    }
    item.info.size = u16::try_from(size_of::<DirInfo>() + len * 2).ok()?;
    Some(item)
}

unsafe extern "C" fn set_delete(
    _file_system: *mut FileSystem,
    _context: *mut c_void,
    _file_name: *mut u16,
    _delete: u8,
) -> NtStatus {
    STATUS_MEDIA_WRITE_PROTECTED
}

unsafe extern "C" fn dispatcher_stopped(file_system: *mut FileSystem, normally: u8) {
    guarded("DispatcherStopped", (), || {
        // `normally` is TRUE when `MountHandle` stopped the dispatcher itself.
        if normally == 0 {
            // SAFETY: WinFsp passes its own file system; the dispatcher result was set on this thread
            // before this call (`FspFileSystemDispatcherThread`), and nothing writes it afterwards.
            let status = unsafe { (*file_system).dispatcher_result };
            tracing::error!(status = format_args!("{status:#010x}"), "the WinFsp driver stopped the volume");
            // SAFETY: as above.
            unsafe { volume(file_system) }.stopped(status);
        }
    });
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime};

    use windows_sys::Win32::Storage::FileSystem::{FILE_ATTRIBUTE_DIRECTORY, FILE_ATTRIBUTE_READONLY};

    use super::{dir_info, file_info, write_normalized_name};
    use crate::fs::Attr;
    use crate::testkit::Scratch;
    use crate::tree::{NodeId, NodeKind};
    use crate::winfsp::abi::{FileInfo, OpenFileInfo};
    use crate::winfsp::names::resolve;
    use crate::winfsp::volume::tests::volume;

    /// What `write_normalized_name` leaves in a buffer of `capacity` bytes: the size field and the text.
    fn normalized(capacity: u16, name: &str) -> (u16, String) {
        let scratch = Scratch::new("winfsp-normalized");
        let volume = volume(&scratch);
        let id = resolve(&volume.fs, &name.encode_utf16().collect::<Vec<_>>()).expect("node");
        // One unit more than the capacity, to see that nothing is written past it.
        let mut buffer = vec![0xAAAA_u16; usize::from(capacity / 2) + 1];
        let mut open_info = OpenFileInfo { file_info: FileInfo::default(), normalized_name: buffer.as_mut_ptr(), normalized_name_size: capacity };
        // SAFETY: `open_info` points at `buffer`, which holds more than `capacity` bytes.
        unsafe { write_normalized_name(&volume, id, &raw mut open_info) };
        assert_eq!(buffer.last(), Some(&0xAAAA), "written past the capacity");
        let size = open_info.normalized_name_size;
        // An untouched buffer still starts with the filler.
        let text = if buffer[0] == 0xAAAA { String::new() } else { String::from_utf16(&buffer[..usize::from(size / 2)]).expect("utf-16") };
        (size, text)
    }

    #[test]
    fn the_stored_case_is_written_only_when_it_fits() {
        // `\Models\Unet\w.bin` is 18 units, 36 bytes.
        assert_eq!(normalized(64, "\\MODELS\\unet\\W.BIN"), (36, "\\Models\\Unet\\w.bin".to_owned()));
        assert_eq!(normalized(36, "\\models\\UNET\\w.bin"), (36, "\\Models\\Unet\\w.bin".to_owned()));
        // Too small: the size stays at the capacity, which tells WinFsp to keep the name as opened.
        assert_eq!(normalized(34, "\\models\\unet\\w.bin"), (34, String::new()));
        assert_eq!(normalized(64, "\\A.BIN"), (12, "\\a.bin".to_owned()));
    }

    #[test]
    fn files_are_read_only_and_entries_carry_their_name() {
        let modified = SystemTime::UNIX_EPOCH + Duration::from_secs(1);
        let file = Attr { id: NodeId::from_raw(7), kind: NodeKind::File, size: 5000, links: 1, modified };
        let info = file_info(&file);
        assert_eq!(info.file_attributes, FILE_ATTRIBUTE_READONLY);
        assert_eq!((info.file_size, info.allocation_size, info.index_number), (5000, 8192, 7));
        assert_eq!(info.last_write_time, 116_444_736_010_000_000);
        let dir = Attr { id: NodeId::ROOT, kind: NodeKind::Dir, size: 0, links: 2, modified };
        assert_eq!(file_info(&dir).file_attributes, FILE_ATTRIBUTE_DIRECTORY);
        let item = dir_info(&file, "a\u{e9}").expect("short name");
        assert_eq!(item.info.size, 104 + 4);
        assert_eq!(&item.name[..2], &[u16::from(b'a'), 0xE9]);
        assert!(dir_info(&file, &"x".repeat(255)).is_some());
        assert!(dir_info(&file, &"x".repeat(256)).is_none());
    }
}
