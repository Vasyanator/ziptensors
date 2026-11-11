//! File: logging.rs
//! Purpose: the opt-in log output of the C ABI: the library's `tracing` events written to stderr.
//!
//! Main responsibilities: [`ZtLogLevel`] and [`zt_enable_stderr_logging`].
//!
//! Key structures: [`ZtLogLevel`].
//!
//! Key functions: [`zt_enable_stderr_logging`].
//!
//! Implementation notes: the library logs through `tracing` and installs nothing by itself, so a host
//! sees no output unless it asks. A process has one global subscriber; the first call decides, its
//! outcome is kept in a `OnceLock`, and every later call returns that outcome without installing
//! anything. A host that already installed a subscriber of its own keeps it, and the call says so.

use std::ffi::c_int;
use std::io::{self, IsTerminal};
use std::sync::OnceLock;

use tracing::Level;

use crate::barrier::guarded;
use crate::status::{Failure, ZtStatus};

/// Most verbose level written by `zt_enable_stderr_logging`. The values are those of the `log` and
/// `tracing` conventions: a larger value writes more.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ZtLogLevel {
    /// Failures only.
    Error = 1,
    /// Failures and refused calls: every failing entry point logs one line at this level.
    Warn = 2,
    /// Also the stages of long operations: open, encode, commit, edit, recover.
    Info = 3,
    /// Also details of decisions inside operations.
    Debug = 4,
    /// Everything.
    Trace = 5,
}

impl ZtLogLevel {
    /// The `tracing` level of this value.
    fn from_raw(raw: c_int) -> Result<Level, Failure> {
        let all = [Self::Error, Self::Warn, Self::Info, Self::Debug, Self::Trace];
        match all.into_iter().find(|level| *level as i32 == raw) {
            Some(Self::Error) => Ok(Level::ERROR),
            Some(Self::Warn) => Ok(Level::WARN),
            Some(Self::Info) => Ok(Level::INFO),
            Some(Self::Debug) => Ok(Level::DEBUG),
            Some(Self::Trace) => Ok(Level::TRACE),
            None => Err(Failure::Invalid { detail: format!("unknown zt_log_level value {raw}") }),
        }
    }
}

/// Outcome of the first call that reached the install: whether this library's subscriber is the
/// process's global one.
static INSTALLED: OnceLock<bool> = OnceLock::new();

/// Writes the library's log lines up to `level`, one of `zt_log_level`, to stderr, for the rest of the
/// process. Off until called. Only the first valid call installs anything: later calls change nothing,
/// a different `level` included, and return what the first one returned.
///
/// Returns `ZT_STATUS_INVALID_ARGUMENT` for a value that is no `zt_log_level`, and
/// `ZT_STATUS_INVALID_STATE` if the process already has a global `tracing` subscriber that this
/// library did not install: the library's lines then go to that one, if anywhere. A `log` logger of
/// the host does not matter, since the library logs through `tracing` only.
#[unsafe(no_mangle)]
pub extern "C" fn zt_enable_stderr_logging(level: c_int) -> ZtStatus {
    guarded("zt_enable_stderr_logging", || {
        let level = ZtLogLevel::from_raw(level)?;
        let installed = *INSTALLED.get_or_init(|| {
            let subscriber = tracing_subscriber::fmt()
                .with_writer(io::stderr)
                .with_max_level(level)
                .with_ansi(io::stderr().is_terminal())
                .finish();
            // Not `try_init`: it also installs a `log` bridge after the subscriber and fails when the
            // host has a `log` logger, which would report a subscriber that is in place as missing.
            match tracing::subscriber::set_global_default(subscriber) {
                Ok(()) => true,
                Err(error) => {
                    // Nothing else can be told: the subscriber that would carry this line is not
                    // ours, so it goes wherever the host's goes.
                    tracing::warn!(%error, "a global subscriber exists, ziptensors installs none");
                    false
                }
            }
        });
        if installed {
            Ok(())
        } else {
            Err(Failure::State {
                detail: "this process already has a global tracing subscriber, \
                         so ziptensors installed none",
            })
        }
    })
}
