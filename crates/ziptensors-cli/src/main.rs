//! File: main.rs
//! Purpose: entry point of the `ziptensors` command line tool.
//!
//! Main responsibilities: parse the arguments, start logging, run the command, map the outcome to an
//! exit code: 0 success, 1 runtime failure, 2 usage. Clap itself exits with 2 on bad arguments, and
//! `mount` ended by a second signal or console key exits from `mount.rs` with 128 + the signal number
//! (130 for Ctrl-C, and 149 for Ctrl-Break on Windows).
//!
//! Key functions: [`main`], [`init_logging`].
//!
//! Implementation notes: logs and progress go to stderr through `tracing`, results to stdout, so
//! stdout of `inspect --json` is exactly one JSON document.

#![warn(clippy::all)]
#![warn(clippy::pedantic)]

mod cli;
mod commands;
mod edit;
mod errors;
mod json;
#[cfg(any(target_os = "linux", windows))]
mod mount;
mod pack;
mod report;
mod tree;

use std::io::{self, IsTerminal, Write};
use std::process::ExitCode;

use cli::Cli;

/// Exit code of a command that started and failed.
const EXIT_RUNTIME_FAILURE: u8 = 1;

/// Sends `tracing` output to stderr at `INFO`, which is where the core logs its stages and this crate
/// its progress. Colors only on a terminal, so a redirected log stays plain text.
fn init_logging() {
    let subscriber = tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_max_level(tracing::Level::INFO)
        .with_ansi(io::stderr().is_terminal())
        .with_target(false);
    if let Err(error) = subscriber.try_init() {
        // The command works without logs, and stderr is the only place left to say so. Nothing can be
        // done about a failure of this write either.
        let _ = writeln!(io::stderr(), "warning: logging is off: {error}");
    }
}

/// Runs the command line.
fn main() -> ExitCode {
    let cli = Cli::parse_checked();
    init_logging();
    match commands::run(&cli.command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            errors::report(&error);
            ExitCode::from(EXIT_RUNTIME_FAILURE)
        }
    }
}
