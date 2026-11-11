//! File: handle.rs
//! Purpose: what both backends' `MountHandle` reports about how a mount ended, in types that name no
//! OS item, so a frontend (the CLI, a Python context manager) keeps one flow on Linux and Windows.
//!
//! Key structures: [`MountEnd`].
//!
//! Key functions: [`signal_name`].
//!
//! Implementation notes: an abandoning event is kept as a signal number. On Linux it is the POSIX
//! signal; on Windows the C runtime's number of the console key (`SIGINT` 2 for Ctrl-C, `SIGBREAK` 21
//! for Ctrl-Break), so the exit status is 128 + the number on both, as a shell reports a death by
//! signal. Built only where a backend exists (Linux, Windows): there is no mount handle on macOS.
//!
//! Both backends' `MountHandle` share one contract, which a frontend may rely on:
//! - `unmount(self) -> Result<(), MountError>` detaches the mount point at once, waits for every file
//!   still open under it to be closed, then ends the session;
//! - `wait(self) -> Result<MountEnd, MountError>` blocks until the mount ends without being asked to
//!   (Linux: an outside `fusermount3 -u`; Windows: the driver stopping the volume);
//! - dropping a handle unmounts as `unmount` does, logging any failure;
//! - `MountHandle` is `Send` (not `Sync`): it may be moved to the thread that unmounts it, and a
//!   frontend that shares it between threads keeps it behind a lock. Both waits block the calling
//!   thread and take no signal: a frontend that must stay interruptible calls them off its main thread.

#[cfg(windows)]
use crate::winfsp::ConsoleEvent;

/// Exit status of a mount the `WinFsp` driver stopped: the CLI's runtime failure.
#[cfg(windows)]
const EXIT_STOPPED: u8 = 1;

/// Offset of an exit status after a death by signal, the shell convention.
const EXIT_SIGNAL_BASE: i32 = 128;

/// How a mount ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MountEnd {
    /// Unmounted, by a signal, a console key, an `unmount` or from outside, and every file that was open
    /// under it closed (on Windows also unmapped).
    Unmounted,
    /// A second signal or console key came while the session still served: files were still open, or
    /// (Linux) the mount would not go. The session is left to the process; the caller should end the
    /// process now, with [`MountEnd::exit_code`], which cuts off the remaining readers. Only
    /// `run_until_signal` returns it.
    Abandoned {
        /// The second signal: a POSIX signal number on Linux, the C runtime's number of the console key
        /// on Windows (2 for Ctrl-C, 21 for Ctrl-Break). [`signal_name`] names it.
        signal: i32,
    },
    /// The `WinFsp` driver stopped the volume on its own: a dispatcher thread failed (for example could
    /// not be created) or `WinFsp` was uninstalled. A failure, not an unmount: the caller reports it and
    /// exits with [`MountEnd::exit_code`], 1.
    #[cfg(windows)]
    Stopped {
        /// The `NTSTATUS` the dispatcher ended with, 0 if `WinFsp` recorded none.
        status: i32,
    },
}

impl MountEnd {
    /// The exit status for this end: 0 after an unmount, 128 + the signal number for an abandoned mount
    /// (130 for a second Ctrl-C on both systems, 149 for a second Ctrl-Break), 1 for a volume the driver
    /// stopped. A signal number that does not fit gives 255.
    #[must_use]
    pub fn exit_code(self) -> u8 {
        match self {
            Self::Unmounted => 0,
            Self::Abandoned { signal } => EXIT_SIGNAL_BASE
                .checked_add(signal)
                .and_then(|code| u8::try_from(code).ok())
                .unwrap_or(u8::MAX),
            #[cfg(windows)]
            Self::Stopped { .. } => EXIT_STOPPED,
        }
    }
}

/// The name a user knows the abandoning `signal` of [`MountEnd::Abandoned`] by, for messages: `SIGINT`,
/// `SIGTERM` or `SIGHUP` on Linux, `Ctrl-C` or `Ctrl-Break` on Windows. A number no backend produces
/// is `an unknown signal`.
#[must_use]
pub fn signal_name(signal: i32) -> &'static str {
    #[cfg(target_os = "linux")]
    let name = nix::sys::signal::Signal::try_from(signal).ok().map(nix::sys::signal::Signal::as_str);
    #[cfg(windows)]
    let name = ConsoleEvent::from_signal(signal).map(ConsoleEvent::as_str);
    name.unwrap_or("an unknown signal")
}

#[cfg(test)]
mod tests {
    use super::{MountEnd, signal_name};
    use crate::{MountError, MountHandle};

    #[test]
    fn exit_codes_follow_the_shell_convention() {
        assert_eq!(MountEnd::Unmounted.exit_code(), 0);
        assert_eq!(MountEnd::Abandoned { signal: 2 }.exit_code(), 130);
        assert_eq!(MountEnd::Abandoned { signal: i32::MAX }.exit_code(), u8::MAX);
        assert_eq!(MountEnd::Abandoned { signal: 200 }.exit_code(), u8::MAX);
        assert_eq!(signal_name(-1), "an unknown signal");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_signals_keep_their_numbers_and_names() {
        use nix::sys::signal::Signal;

        for (signal, code, name) in [(Signal::SIGINT, 130, "SIGINT"), (Signal::SIGTERM, 143, "SIGTERM"), (Signal::SIGHUP, 129, "SIGHUP")] {
            let number = signal as i32;
            assert_eq!(MountEnd::Abandoned { signal: number }.exit_code(), code, "{name}");
            assert_eq!(signal_name(number), name);
        }
    }

    #[cfg(windows)]
    #[test]
    fn console_keys_map_to_the_c_runtime_signals() {
        use crate::winfsp::ConsoleEvent;

        assert_eq!(MountEnd::Stopped { status: -1 }.exit_code(), 1);
        for (event, code, name) in [(ConsoleEvent::CtrlC, 130, "Ctrl-C"), (ConsoleEvent::CtrlBreak, 149, "Ctrl-Break")] {
            assert_eq!(ConsoleEvent::from_signal(event.signal()), Some(event));
            assert_eq!(MountEnd::Abandoned { signal: event.signal() }.exit_code(), code, "{name}");
            assert_eq!(signal_name(event.signal()), name);
        }
    }

    /// A frontend moves the handle to the thread that unmounts it and sends its errors across threads.
    #[test]
    fn handles_and_their_errors_cross_threads() {
        fn send<T: Send>() {}
        fn send_sync<T: Send + Sync + 'static>() {}
        send::<MountHandle>();
        send_sync::<MountError>();
        send_sync::<MountEnd>();
    }
}
