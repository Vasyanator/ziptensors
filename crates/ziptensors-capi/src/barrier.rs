//! File: barrier.rs
//! Purpose: the panic barrier every entry point runs in, and the thread-local error text behind it.
//!
//! Main responsibilities:
//! - [`guarded`] and [`guarded_value`], the only callers of `catch_unwind`;
//! - the last error of the calling thread and [`zt_last_error`], which copies it out.
//!
//! Key functions: [`guarded`], [`guarded_value`], [`zt_last_error`].
//!
//! Implementation notes: a panic reaching an `extern "C"` boundary aborts the host, so nothing may run
//! outside the closure but work that cannot panic. A panic is not logged here: the panic hook of the
//! host already saw it. `AssertUnwindSafe` is sound because after a panic the caller gets
//! `ZT_STATUS_PANIC`, and the only mutable handle, `zt_pack`, gave up its builder for the call.

use std::any::Any;
use std::cell::RefCell;
use std::ffi::c_char;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use crate::status::{Failure, ZtStatus};

thread_local! {
    /// Text of the last failure on this thread, empty after a success.
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

#[cfg(test)]
thread_local! {
    /// Test only: makes the next entry on this thread panic inside its barrier.
    static PANIC_NEXT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Test only: makes the next entry point called on this thread panic inside its barrier.
#[cfg(test)]
pub(crate) fn panic_on_next_entry() {
    PANIC_NEXT.set(true);
}

/// Test only: the injected panic, the first thing inside a barrier.
#[cfg(test)]
fn inject_panic(entry: &str) {
    assert!(!PANIC_NEXT.take(), "injected panic in {entry}");
}

/// Replaces the text of the last error. A NUL would end the text early for C, so it is spelled out.
fn set_last_error(text: &str) {
    let stored = LAST_ERROR.try_with(|last| {
        let mut last = last.borrow_mut();
        last.clear();
        if text.contains('\0') {
            last.push_str(&text.replace('\0', "\\0"));
        } else {
            last.push_str(text);
        }
    });
    if let Err(error) = stored {
        // Only a thread being torn down gets here, and it has no later call that could read the text.
        tracing::debug!(%error, "thread is exiting, dropping an error text");
    }
}

/// The message of a panic payload, for the two payload types `panic!` produces.
fn panic_text(payload: &(dyn Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("non-text panic payload")
}

/// Runs a fallible entry point: clears the last error, runs `body`, and turns its outcome, a panic
/// included, into a status and an error text. `entry` is the C name, for the log and the text.
pub(crate) fn guarded(entry: &'static str, body: impl FnOnce() -> Result<(), Failure>) -> ZtStatus {
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        set_last_error("");
        #[cfg(test)]
        inject_panic(entry);
        match body() {
            Ok(()) => ZtStatus::Ok,
            Err(failure) => {
                let status = failure.status();
                tracing::warn!(entry, ?status, %failure, "C ABI call failed");
                set_last_error(&format!("{entry}: {failure}"));
                status
            }
        }
    }));
    outcome.unwrap_or_else(|payload| {
        set_last_error(&format!("{entry}: internal panic: {}", panic_text(payload.as_ref())));
        ZtStatus::Panic
    })
}

/// Runs an entry point that has no status: `fallback` is its result if `body` panics. The last error
/// is left alone, unless a panic sets it.
pub(crate) fn guarded_value<T>(entry: &'static str, fallback: T, body: impl FnOnce() -> T) -> T {
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        #[cfg(test)]
        inject_panic(entry);
        body()
    }));
    outcome.unwrap_or_else(|payload| {
        set_last_error(&format!("{entry}: internal panic: {}", panic_text(payload.as_ref())));
        fallback
    })
}

/// Copies the text of the last failure on the calling thread into `buf`, and returns its full length
/// in bytes, without the terminating NUL. Returns 0 if the last status-returning call on this thread
/// succeeded, or if there was none.
///
/// With `cap > 0`, `buf` receives at most `cap - 1` bytes and a NUL, so the text was cut exactly when
/// the result is `>= cap`. A cut never splits a UTF-8 sequence. With `buf == NULL` or `cap == 0` nothing
/// is written, which is how to ask for the length. `buf` is only written, never read.
///
/// # Safety
/// Unless it is NULL, `buf` must be valid for writes of `cap` bytes. The text stored here is a copy, so
/// it stays valid however handles are closed.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zt_last_error(buf: *mut c_char, cap: usize) -> usize {
    guarded_value("zt_last_error", 0, || {
        let copied = LAST_ERROR.try_with(|last| {
            let text = last.borrow();
            if !buf.is_null() && cap > 0 {
                let kept = text.floor_char_boundary(cap - 1);
                // SAFETY: `buf` holds `cap` bytes by this function's contract, and `kept < cap`, so
                // `kept` bytes and the NUL after them fit. `text` is ours, so they do not overlap.
                unsafe {
                    ptr::copy_nonoverlapping(text.as_ptr(), buf.cast::<u8>(), kept);
                    buf.add(kept).write(0);
                }
            }
            text.len()
        });
        // A thread being torn down has lost its text, which reads as no error.
        copied.unwrap_or(0)
    })
}
