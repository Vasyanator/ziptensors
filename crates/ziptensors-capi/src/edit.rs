//! File: edit.rs
//! Purpose: the editing half of the C ABI: changing a container in place of its file, and recovering
//! a file whose end is torn.
//!
//! Main responsibilities: [`zt_edit`] over core `edit`, with the operation array [`ZtEditOp`] and its
//! `ZT_EDIT_*` kinds, and [`zt_recover`] over core `recover`.
//!
//! Key structures: [`ZtEditOp`].
//!
//! Key functions: [`zt_edit`], [`zt_recover`].
//!
//! Implementation notes: an operation's kind is an `i32` checked against the `ZT_EDIT_*` constants
//! rather than a Rust enum, for the reason `options.rs` gives for enum fields; constants rather than an
//! exported enum because a C enum type named `zt_edit` would collide with the function `zt_edit`. Every
//! operation is translated before core sees any, so a refused element changes nothing. Which
//! combinations of operations and strategy are valid is core's decision alone
//! (`ZT_STATUS_EDIT_REFUSED`); this file refuses only what C cannot express in Rust: an unknown kind, a
//! NULL or non-UTF-8 string, a field the kind does not take.

use std::ffi::c_char;
use std::slice;

use ziptensors_core::{EditOp, edit, recover};

use crate::args::{path, text};
use crate::barrier::guarded;
use crate::options::{ZtEditOptions, ZtRecoverOptions, edit_options, recover_options};
use crate::status::{Failure, ZtStatus};

/// `zt_edit_op.kind`: adds the file `path` as the new member `name`, after the kept members.
pub const ZT_EDIT_ADD: i32 = 1;
/// `zt_edit_op.kind`: removes the member `name`. `path` and `delta_base` must be NULL.
pub const ZT_EDIT_REMOVE: i32 = 2;
/// `zt_edit_op.kind`: replaces the content of member `name` with the file `path`, in its slot.
pub const ZT_EDIT_REPLACE: i32 = 3;

/// One change of `zt_edit`. A zeroed struct is refused: 0 is no kind. Strings are read during the call
/// only.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ZtEditOp {
    /// `ZT_EDIT_ADD`, `ZT_EDIT_REMOVE` or `ZT_EDIT_REPLACE`. Any other value is refused with
    /// `ZT_STATUS_INVALID_ARGUMENT`.
    pub kind: i32,
    /// The member, UTF-8, `/` separated. For an add, a new name a restore accepts.
    pub name: *const c_char,
    /// The input file, for an add or a replace; NULL for a remove. It must be a regular file, not a
    /// symbolic link, and not the container.
    pub path: *const c_char,
    /// NULL, or for an add or a replace a member of the result to store the file against as delta
    /// blobs where that is smaller: a kept member, or a file added or replaced earlier in this array.
    /// NULL for a remove.
    pub delta_base: *const c_char,
}

/// Borrows a string that may be NULL, which means `None`.
///
/// # Safety
/// As [`text`].
unsafe fn optional_text(raw: *const c_char, argument: &'static str) -> Result<Option<String>, Failure> {
    if raw.is_null() {
        return Ok(None);
    }
    // SAFETY: passed on from this function's contract.
    Ok(Some(unsafe { text(raw, argument) }?.to_owned()))
}

/// The kind of an operation, once its raw value was checked.
#[derive(Debug, Clone, Copy)]
enum Kind {
    /// `ZT_EDIT_ADD`.
    Add,
    /// `ZT_EDIT_REMOVE`.
    Remove,
    /// `ZT_EDIT_REPLACE`.
    Replace,
}

/// Translates one operation. The kind is judged before any field, since it decides which fields
/// count.
///
/// # Safety
/// The three strings of `raw` are each NULL or NUL-terminated, and stay valid and unchanged for the
/// call.
unsafe fn edit_op(raw: &ZtEditOp) -> Result<EditOp, Failure> {
    let kind = match raw.kind {
        ZT_EDIT_ADD => Kind::Add,
        ZT_EDIT_REMOVE => Kind::Remove,
        ZT_EDIT_REPLACE => Kind::Replace,
        other => {
            return Err(Failure::Invalid { detail: format!("unknown zt_edit_op kind {other}") });
        }
    };
    // SAFETY: passed on from this function's contract.
    let name = unsafe { text(raw.name, "name") }?.to_owned();
    match kind {
        Kind::Add => Ok(EditOp::Add {
            // SAFETY: passed on from this function's contract.
            path: unsafe { path(raw.path, "path") }?.to_owned(),
            name,
            // SAFETY: passed on from this function's contract.
            delta_base: unsafe { optional_text(raw.delta_base, "delta_base") }?,
        }),
        Kind::Replace => Ok(EditOp::Replace {
            name,
            // SAFETY: passed on from this function's contract.
            path: unsafe { path(raw.path, "path") }?.to_owned(),
            // SAFETY: passed on from this function's contract.
            delta_base: unsafe { optional_text(raw.delta_base, "delta_base") }?,
        }),
        Kind::Remove => {
            // A field the kind ignores would be a silent no-op: the caller meant something else.
            if !raw.path.is_null() || !raw.delta_base.is_null() {
                return Err(Failure::Invalid {
                    detail: "a ZT_EDIT_REMOVE takes no `path` and no `delta_base`".to_owned(),
                });
            }
            Ok(EditOp::Remove { name })
        }
    }
}

/// Edits the container at `container` by the `count` operations at `ops`, a set: each member name
/// appears in at most one operation. Kept members keep their order, a replace keeps its member's slot,
/// and adds follow in array order. `count == 0` (with `ops` NULL or not) compacts the container: it is
/// rewritten without the dead data of earlier edits. A NULL `options` means defaults: a rewrite
/// through a temporary file with `ZT_MODE_MAX` for new files. The result is proven before it replaces
/// the container.
///
/// Every check runs before a byte is written, so after these statuses the file is untouched:
/// `ZT_STATUS_INVALID_ARGUMENT` for an unknown `kind` or a field the kind does not take,
/// `ZT_STATUS_NULL_ARGUMENT` and `ZT_STATUS_INVALID_UTF8` for strings (the error text names the
/// operation's index), `ZT_STATUS_EDIT_REFUSED` for a set or a combination that cannot be applied (a
/// name in two operations, an add of an existing name, `repack` without `ZT_STRATEGY_REWRITE`, an
/// append with no operation, ...), `ZT_STATUS_MEMBER_NOT_FOUND`, `ZT_STATUS_UNSAFE_MEMBER_NAME`,
/// `ZT_STATUS_OUTPUT_IS_INPUT`, `ZT_STATUS_SYMLINK_INPUT`, and the statuses of `zt_open`.
///
/// Later failures depend on the strategy: a rewrite leaves the container intact; an append cuts it
/// back, or returns `ZT_STATUS_TRAILING_BYTES` if it cannot (`zt_recover` repairs that); an in-place
/// rewrite returns `ZT_STATUS_DAMAGED_IN_PLACE`: the container is damaged. Close every `zt_container`
/// open on the file before a rewrite or an in-place edit.
///
/// # Safety
/// `container` must be NULL or NUL-terminated. Unless `count` is 0, `ops` must be NULL or point at
/// `count` valid `zt_edit_op`, whose strings are NULL or NUL-terminated. `options` must be NULL or a
/// valid `zt_edit_options` whose `struct_size` is set. Everything is only read during the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zt_edit(
    container: *const c_char,
    ops: *const ZtEditOp,
    count: usize,
    options: *const ZtEditOptions,
) -> ZtStatus {
    guarded("zt_edit", || {
        // SAFETY: `container` is covered by this function's contract and outlives the call.
        let target = unsafe { path(container, "container") }?;
        // SAFETY: `options` is covered by this function's contract.
        let options = unsafe { edit_options(options) }?;
        let raw_ops: &[ZtEditOp] = if count == 0 {
            &[]
        } else {
            if ops.is_null() {
                return Err(Failure::Null { argument: "ops" });
            }
            // A slice may not span more than `isize::MAX` bytes; such a count is no real array.
            let fits = count
                .checked_mul(size_of::<ZtEditOp>())
                .is_some_and(|bytes| isize::try_from(bytes).is_ok());
            if !fits {
                return Err(Failure::Invalid { detail: format!("argument `count` is too large: {count}") });
            }
            // SAFETY: not NULL, `count` elements by this function's contract, and the size fits a slice.
            unsafe { slice::from_raw_parts(ops, count) }
        };
        let translated = raw_ops
            .iter()
            .enumerate()
            .map(|(index, raw)| {
                // SAFETY: the strings of every element are covered by this function's contract.
                unsafe { edit_op(raw) }.map_err(|inner| Failure::Op { index, inner: Box::new(inner) })
            })
            .collect::<Result<Vec<_>, _>>()?;
        edit(target, &translated, &options)?;
        Ok(())
    })
}

/// Shortens the file at `container` to its latest valid generation, when its end does not close a
/// valid container: after an append that was cut off, or after `ZT_STATUS_TRAILING_BYTES`. Never done
/// on its own: no reader recovers. A NULL `options` means defaults. On success, unless `valid_len_out`
/// is NULL, `*valid_len_out` is the length of the valid container: the file's length if it opened as it
/// was (then nothing is changed) or the length it was cut to (or would be, on a dry run). It is
/// untouched on failure.
///
/// By default every member of the generation found is decoded and checked before anything is cut,
/// and a generation that fails is passed over for an older one. Nothing below the valid length is
/// ever written.
///
/// Returns `ZT_STATUS_UNRECOVERABLE` if no valid generation was found within the budget,
/// `ZT_STATUS_NOT_ZIPTENSORS` if the file does not start as a container, a verification status such
/// as `ZT_STATUS_VERIFY_FAILED` if no structurally valid generation verifies, `ZT_STATUS_BUDGET_EXCEEDED`
/// if a candidate is over the limits, and `ZT_STATUS_IO`. Every failure leaves the file unchanged,
/// except `ZT_STATUS_IO` from the sync after the cut: then the file is cut, but the cut may not survive
/// a power loss.
///
/// # Safety
/// `container` must be NULL or NUL-terminated, and is only read during the call. `options` must be NULL
/// or a valid `zt_recover_options` whose `struct_size` is set. `valid_len_out` must be NULL or valid for
/// a write, and is not read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zt_recover(
    container: *const c_char,
    options: *const ZtRecoverOptions,
    valid_len_out: *mut u64,
) -> ZtStatus {
    guarded("zt_recover", || {
        // SAFETY: `container` is covered by this function's contract and outlives the call.
        let target = unsafe { path(container, "container") }?;
        // SAFETY: `options` is covered by this function's contract.
        let options = unsafe { recover_options(options) }?;
        let report = recover(target, &options)?;
        if !valid_len_out.is_null() {
            // SAFETY: not NULL, and writable by this function's contract.
            unsafe { valid_len_out.write(report.valid_len) };
        }
        Ok(())
    })
}
