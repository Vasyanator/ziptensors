//! File: logging.rs
//! Purpose: `enable_stderr_logging`, the opt-in that shows the core's `tracing` events on stderr.
//!
//! Main responsibilities: parse the level and install a `tracing_subscriber::fmt` subscriber that
//! writes to stderr, at most once per process.
//!
//! Key functions: [`enable_stderr_logging`].
//!
//! Implementation notes: the subscriber is installed with `set_global_default`, not `try_init`,
//! because `try_init` also installs a `log` bridge after it and reports an error when another
//! library already set a `log` logger, though the subscriber is in place. There is no bridge to Python's `logging`: a `Layer` that attaches the
//! interpreter from the core's rayon threads needs a design of its own, and is deferred.

use std::io::{self, IsTerminal};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tracing_subscriber::filter::LevelFilter;

/// Maps the `level` argument, ignoring case.
///
/// # Errors
/// `ValueError` for anything but `off`, `error`, `warn` (or `warning`), `info`, `debug`, `trace`.
fn parse_level(level: &str) -> PyResult<LevelFilter> {
    match level.to_ascii_lowercase().as_str() {
        "off" => Ok(LevelFilter::OFF),
        "error" => Ok(LevelFilter::ERROR),
        "warn" | "warning" => Ok(LevelFilter::WARN),
        "info" => Ok(LevelFilter::INFO),
        "debug" => Ok(LevelFilter::DEBUG),
        "trace" => Ok(LevelFilter::TRACE),
        _ => Err(PyValueError::new_err(format!(
            "unknown level {level:?}, expected \"off\", \"error\", \"warn\", \"info\", \"debug\" or \
             \"trace\""
        ))),
    }
}

/// Writes the library's log events at `level` and above to stderr, for the rest of the process.
/// Off by default. Only the first successful call in a process takes effect: a later call, or a
/// call after another library installed a global `tracing` subscriber, changes nothing and still
/// succeeds. Python's `logging` does not see these events.
#[pyfunction]
#[pyo3(signature = (level="info"))]
pub(crate) fn enable_stderr_logging(level: &str) -> PyResult<()> {
    let filter = parse_level(level)?;
    let subscriber = tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_max_level(filter)
        .with_ansi(io::stderr().is_terminal())
        .finish();
    if let Err(error) = tracing::subscriber::set_global_default(subscriber) {
        // The documented no-op: a subscriber is already in place, and it receives this line.
        tracing::debug!(%error, "a global tracing subscriber is already set; keeping it");
    }
    Ok(())
}
