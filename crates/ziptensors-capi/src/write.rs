//! File: write.rs
//! Purpose: the writing half of the C ABI: one-shot encode and the pack builder.
//!
//! Main responsibilities: [`zt_encode_file`], the opaque handle [`ZtPack`] and its entry points over
//! `PackBuilder`.
//!
//! Key structures: [`ZtPack`].
//!
//! Implementation notes: `PackBuilder::finish` consumes the builder and C cannot be made to forget a
//! pointer, so the handle owns an `Option` and outlives the builder. Each call takes the builder out
//! while it runs: a panic then drops it, which removes the temporary file, instead of leaving a
//! half-updated builder for the next call.

use std::ffi::c_char;

use ziptensors_core::{EncodeOptions, PackBuilder, encode_file};

use crate::args::{handle_mut, path, reset_out, text};
use crate::barrier::{guarded, guarded_value};
use crate::options::{ZtEncodeOptions, encode_options};
use crate::status::{Failure, ZtStatus};

/// A container being written. Create with `zt_pack_new`, then call `zt_pack_add_file` for each input and
/// `zt_pack_finish`, and always `zt_pack_free`. Nothing appears at the output before `zt_pack_finish`
/// succeeds or returns `ZT_STATUS_COMMITTED_NOT_DURABLE`, and freeing an unfinished pack removes its
/// temporary file.
///
/// One call at a time: a handle may move between threads but must not be used by two at once.
#[derive(Debug)]
pub struct ZtPack {
    pub(crate) builder: Option<PackBuilder>,
}

impl ZtPack {
    /// Takes the builder for the length of a call.
    fn take(&mut self) -> Result<PackBuilder, Failure> {
        self.builder.take().ok_or(Failure::State {
            detail: "this pack was already finished, or a call on it panicked",
        })
    }
}

/// Encodes the file at `input` as a container of one member, named after the file, at `output`. A
/// `NULL` `options` means defaults: `ZT_MODE_BALANCED`, one thread per CPU. The whole output is decoded
/// and hashed against the input before it appears. On failure nothing is left at `output`, except
/// after `ZT_STATUS_COMMITTED_NOT_DURABLE`, which means the proven file is at `output` but its
/// directory could not be synced.
///
/// Returns `ZT_STATUS_SYMLINK_INPUT` if `input` is a symbolic link, `ZT_STATUS_UNSAFE_MEMBER_NAME` if a
/// restore would refuse the file name, `ZT_STATUS_INPUT_CHANGED` if the file changed during the call,
/// and the statuses of `zt_open` for options and `ZT_STATUS_IO` for file operations.
///
/// # Safety
/// `input` and `output` must be NULL or NUL-terminated, and are only read during the call. `options`
/// must be NULL or a valid `zt_encode_options` whose `struct_size` is set.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zt_encode_file(
    input: *const c_char,
    output: *const c_char,
    options: *const ZtEncodeOptions,
) -> ZtStatus {
    guarded("zt_encode_file", || {
        // SAFETY: `input` is covered by this function's contract and outlives the call.
        let input = unsafe { path(input, "input") }?;
        // SAFETY: `output` is covered by this function's contract and outlives the call.
        let output = unsafe { path(output, "output") }?;
        // SAFETY: `options` is covered by this function's contract.
        let options = unsafe { encode_options(options, EncodeOptions::for_encode()) }?;
        encode_file(input, output, &options)?;
        Ok(())
    })
}

/// Starts a container that will end up at `output`. A `NULL` `options` means defaults: `ZT_MODE_MAX`,
/// one thread per CPU. On success `*out` is a handle for `zt_pack_free`; on any failure `*out` is NULL.
///
/// # Safety
/// `output` must be NULL or NUL-terminated, and is only read during the call. `options` must be NULL or
/// a valid `zt_encode_options` whose `struct_size` is set. `out` must be NULL or valid for a write, and
/// is not read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zt_pack_new(
    output: *const c_char,
    options: *const ZtEncodeOptions,
    out: *mut *mut ZtPack,
) -> ZtStatus {
    guarded("zt_pack_new", || {
        // SAFETY: `out` is covered by this function's contract.
        unsafe { reset_out(out, "out") }?;
        // SAFETY: `output` is covered by this function's contract and outlives the call.
        let output = unsafe { path(output, "output") }?;
        // SAFETY: `options` is covered by this function's contract.
        let options = unsafe { encode_options(options, EncodeOptions::for_pack()) }?;
        let builder = PackBuilder::new(output, &options)?;
        let pack = Box::into_raw(Box::new(ZtPack { builder: Some(builder) }));
        // SAFETY: `reset_out` proved `out` is not NULL, and it is writable by this function's contract.
        unsafe { out.write(pack) };
        Ok(())
    })
}

/// Adds the file at `path` as the member `member`, a `/` separated relative path. On failure the member
/// is not added and the pack stays usable, unless writing the output failed, in which case every later
/// call on the pack fails too.
///
/// Returns `ZT_STATUS_UNSAFE_MEMBER_NAME` if a restore would refuse `member`,
/// `ZT_STATUS_MALFORMED_INDEX` if the pack has a member of this name, `ZT_STATUS_SYMLINK_INPUT`,
/// `ZT_STATUS_IO` and `ZT_STATUS_INPUT_CHANGED` as `zt_encode_file`, and `ZT_STATUS_INVALID_STATE` after
/// `zt_pack_finish`.
///
/// # Safety
/// `pack` must be NULL or a live handle no other thread is using. `path` and `member` must be NULL or
/// NUL-terminated, and are only read during the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zt_pack_add_file(
    pack: *mut ZtPack,
    path: *const c_char,
    member: *const c_char,
) -> ZtStatus {
    guarded("zt_pack_add_file", || {
        // SAFETY: `pack` is covered by this function's contract and is exclusive for the call.
        let pack = unsafe { handle_mut(pack, "pack") }?;
        // SAFETY: `path` is covered by this function's contract and outlives the call.
        let input = unsafe { self::path(path, "path") }?;
        // SAFETY: `member` is covered by this function's contract and outlives the call.
        let member = unsafe { text(member, "member") }?;
        let mut builder = pack.take()?;
        let added = builder.add_file(input, member);
        pack.builder = Some(builder);
        added?;
        Ok(())
    })
}

/// Commits the container: decodes it all and hashes it against the inputs, syncs it, and renames it to
/// the output path. The handle stays allocated for `zt_pack_free`, but this works once: after it,
/// successful or not, the pack is over and further calls return `ZT_STATUS_INVALID_STATE`. On failure
/// nothing is left at the output path, except after `ZT_STATUS_COMMITTED_NOT_DURABLE`: then the proven
/// container is at the output path, but its directory could not be synced, so a power loss may bring
/// back what was there before.
///
/// # Safety
/// `pack` must be NULL or a live handle no other thread is using.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zt_pack_finish(pack: *mut ZtPack) -> ZtStatus {
    guarded("zt_pack_finish", || {
        // SAFETY: `pack` is covered by this function's contract and is exclusive for the call.
        let pack = unsafe { handle_mut(pack, "pack") }?;
        pack.take()?.finish()?;
        Ok(())
    })
}

/// Frees a pack, finished or not; an unfinished one loses its temporary file and leaves no output. NULL
/// is a no-op. Calling this twice on one handle, or using it afterwards, is undefined behavior.
///
/// # Safety
/// `pack` must be NULL, or a handle from `zt_pack_new` not yet freed, with no call on it running.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zt_pack_free(pack: *mut ZtPack) {
    guarded_value("zt_pack_free", (), || {
        if !pack.is_null() {
            // SAFETY: it came from `Box::into_raw` in `zt_pack_new`, and this function's contract gives
            // us the last use of it.
            drop(unsafe { Box::from_raw(pack) });
        }
    });
}
