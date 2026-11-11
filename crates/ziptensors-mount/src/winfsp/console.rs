//! File: winfsp/console.rs
//! Purpose: Ctrl-C and Ctrl-Break for a foreground `WinFsp` mount: the console control handler, the
//! events a foreground mount waits for, and the decision each event takes.
//!
//! Key structures: [`UnmountSignals`], [`ConsoleEvent`], `Wake`, `Watch`, `Action`. How a mount ended
//! is the backend-neutral `MountEnd` of `handle.rs`.
//!
//! Key functions: [`UnmountSignals::install`], `step`.
//!
//! Implementation notes: `SetConsoleCtrlHandler` takes no context, so the handler reaches its
//! listener through one write-once static sender, set by [`UnmountSignals::install`] and never changed
//! (a read-only global after initialization). The handler runs on a thread the system creates for
//! each event; it only sends and returns. Ctrl-C and Ctrl-Break are taken; closing the console,
//! logoff and shutdown are left to the default handler, which ends the process: its volume handle
//! closes, the driver removes the volume, a drive letter goes with it and a directory mount point is
//! deleted on close.

#![allow(unsafe_code, reason = "SetConsoleCtrlHandler and the handler it calls")]
#![warn(clippy::undocumented_unsafe_blocks)]

use std::io;
use std::sync::OnceLock;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::time::Duration;

use windows_sys::Win32::System::Console::{CTRL_BREAK_EVENT, CTRL_C_EVENT, SetConsoleCtrlHandler};
use windows_sys::core::BOOL;

/// The listener of the console handler, set once per process by [`UnmountSignals::install`].
static LISTENER: OnceLock<Sender<Wake>> = OnceLock::new();

/// The C runtime's `SIGINT`, which it raises for Ctrl-C.
const SIGNAL_CTRL_C: i32 = 2;

/// The C runtime's `SIGBREAK`, which it raises for Ctrl-Break.
const SIGNAL_CTRL_BREAK: i32 = 21;

/// A console event that ends a foreground mount.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleEvent {
    /// Ctrl-C.
    CtrlC,
    /// Ctrl-Break.
    CtrlBreak,
}

impl ConsoleEvent {
    /// The key as a user presses it, for messages.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CtrlC => "Ctrl-C",
            Self::CtrlBreak => "Ctrl-Break",
        }
    }

    /// The C runtime's signal number of the key (`SIGINT` 2, `SIGBREAK` 21), the `signal` of
    /// `MountEnd::Abandoned`, so the exit status is 128 + it as on Linux (130, 149).
    #[must_use]
    pub fn signal(self) -> i32 {
        match self {
            Self::CtrlC => SIGNAL_CTRL_C,
            Self::CtrlBreak => SIGNAL_CTRL_BREAK,
        }
    }

    /// The key whose C runtime signal number is `signal`, if any.
    #[must_use]
    pub fn from_signal(signal: i32) -> Option<Self> {
        match signal {
            SIGNAL_CTRL_C => Some(Self::CtrlC),
            SIGNAL_CTRL_BREAK => Some(Self::CtrlBreak),
            _ => None,
        }
    }
}

/// What wakes a foreground mount.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Wake {
    /// The user pressed a key.
    Console(ConsoleEvent),
    /// The last file open on a detached volume was closed.
    FilesClosed,
    /// The driver stopped the volume (a dispatcher thread failed, or `WinFsp` was uninstalled).
    Stopped,
    /// A second passed with no other wake-up: time to reopen files whose Close is deferred.
    Tick,
}

/// Where a foreground mount is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Watch {
    /// Mounted, no event yet.
    Serving,
    /// The first event detached the mount point; open files are still served.
    Unmounting,
}

/// What a foreground mount does about one wake-up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Action {
    /// Remove the mount point, then end at once if no file is open.
    Detach,
    /// End with [`crate::MountEnd::Unmounted`] if still no file is open (one may have been opened since the
    /// wake-up was sent), else keep waiting.
    Finish,
    /// End with [`crate::MountEnd::Stopped`].
    DriverStopped,
    /// Reopen and close the files still counted open, so a deferred Close arrives (`poke.rs`).
    Poke,
    /// Leave the volume running and return [`crate::MountEnd::Abandoned`].
    Abandon(ConsoleEvent),
    /// Nothing: a wake-up that does not apply in this state.
    Ignore,
}

/// The transition of a foreground mount on `wake`. A stopped volume ends it in every state.
#[must_use]
pub(super) fn step(state: Watch, wake: Wake) -> (Watch, Action) {
    match (state, wake) {
        (_, Wake::Stopped) => (state, Action::DriverStopped),
        (Watch::Serving, Wake::Console(_)) => (Watch::Unmounting, Action::Detach),
        (Watch::Unmounting, Wake::Console(event)) => (Watch::Unmounting, Action::Abandon(event)),
        (Watch::Unmounting, Wake::FilesClosed) => (Watch::Unmounting, Action::Finish),
        (Watch::Unmounting, Wake::Tick) => (Watch::Unmounting, Action::Poke),
        // Files are counted as closed only after a detach, so this is a leftover of nothing; a mounted
        // volume needs no reopening.
        (Watch::Serving, Wake::FilesClosed | Wake::Tick) => (Watch::Serving, Action::Ignore),
    }
}

/// The console handler of a process with a foreground mount. One per process: the handler cannot be
/// pointed at another listener once installed. Dropping it makes Ctrl-C end the process again, as
/// without it.
#[derive(Debug)]
pub struct UnmountSignals {
    receiver: Receiver<Wake>,
    /// A sender of the same channel, handed to the volume so that its events reach `receiver` too.
    sender: Sender<Wake>,
}

impl UnmountSignals {
    /// Installs the handler that turns Ctrl-C and Ctrl-Break into wake-ups of [`run_until_signal`],
    /// and enables Ctrl-C for this process in case its parent started it with Ctrl-C disabled.
    ///
    /// [`run_until_signal`]: super::run_until_signal
    ///
    /// # Errors
    /// `AlreadyExists` when called a second time in the process, else the error of
    /// `SetConsoleCtrlHandler`.
    pub fn install() -> io::Result<Self> {
        let (sender, receiver) = mpsc::channel();
        LISTENER
            .set(sender.clone())
            .map_err(|_| io::Error::new(io::ErrorKind::AlreadyExists, "the console handler is already installed"))?;
        // SAFETY: a null routine with FALSE only clears the process' "ignore Ctrl-C" flag.
        if unsafe { SetConsoleCtrlHandler(None, 0) } == 0 {
            tracing::warn!(error = %io::Error::last_os_error(), "cannot enable Ctrl-C for this process");
        }
        // SAFETY: `handler` matches `PHANDLER_ROUTINE`, cannot unwind, and stays valid for the process.
        if unsafe { SetConsoleCtrlHandler(Some(handler), 1) } == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { receiver, sender })
    }

    /// Waits up to `timeout` for the next wake-up, [`Wake::Tick`] when none came. `None` cannot happen
    /// while `self` lives (it holds a sender).
    pub(super) fn next(&self, timeout: Duration) -> Option<Wake> {
        match self.receiver.recv_timeout(timeout) {
            Ok(wake) => Some(wake),
            Err(RecvTimeoutError::Timeout) => Some(Wake::Tick),
            Err(RecvTimeoutError::Disconnected) => None,
        }
    }

    /// A sender of the wake-up channel, for the volume.
    #[must_use]
    pub(super) fn sender(&self) -> Sender<Wake> {
        self.sender.clone()
    }
}

/// The console control handler: Ctrl-C and Ctrl-Break are sent to the listener and handled (TRUE);
/// everything else, and every event when nobody listens any more, goes on to the default handler,
/// which ends the process.
unsafe extern "system" fn handler(event: u32) -> BOOL {
    let event = match event {
        CTRL_C_EVENT => ConsoleEvent::CtrlC,
        CTRL_BREAK_EVENT => ConsoleEvent::CtrlBreak,
        _ => return 0,
    };
    match LISTENER.get() {
        Some(listener) if listener.send(Wake::Console(event)).is_ok() => 1,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::{Action, ConsoleEvent, Wake, Watch, step};

    #[test]
    fn the_first_key_detaches_the_second_abandons_and_closed_files_finish() {
        let c = Wake::Console(ConsoleEvent::CtrlC);
        let b = Wake::Console(ConsoleEvent::CtrlBreak);
        assert_eq!(step(Watch::Serving, c), (Watch::Unmounting, Action::Detach));
        assert_eq!(step(Watch::Serving, b), (Watch::Unmounting, Action::Detach));
        assert_eq!(step(Watch::Unmounting, c), (Watch::Unmounting, Action::Abandon(ConsoleEvent::CtrlC)));
        assert_eq!(step(Watch::Unmounting, b), (Watch::Unmounting, Action::Abandon(ConsoleEvent::CtrlBreak)));
        assert_eq!(step(Watch::Unmounting, Wake::FilesClosed), (Watch::Unmounting, Action::Finish));
        assert_eq!(step(Watch::Serving, Wake::FilesClosed), (Watch::Serving, Action::Ignore));
        assert_eq!(step(Watch::Serving, Wake::Stopped), (Watch::Serving, Action::DriverStopped));
        assert_eq!(step(Watch::Unmounting, Wake::Stopped), (Watch::Unmounting, Action::DriverStopped));
        assert_eq!(step(Watch::Unmounting, Wake::Tick), (Watch::Unmounting, Action::Poke));
        assert_eq!(step(Watch::Serving, Wake::Tick), (Watch::Serving, Action::Ignore));
    }

    #[test]
    fn keys_carry_the_c_runtime_signal_numbers() {
        assert_eq!(ConsoleEvent::CtrlC.signal(), 2);
        assert_eq!(ConsoleEvent::CtrlBreak.signal(), 21);
        assert_eq!(ConsoleEvent::from_signal(21), Some(ConsoleEvent::CtrlBreak));
        assert_eq!(ConsoleEvent::from_signal(15), None);
        assert_eq!(ConsoleEvent::CtrlBreak.as_str(), "Ctrl-Break");
    }
}
