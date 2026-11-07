//! File: mount.rs
//! Purpose: the `mount` subcommand (Linux and Windows): show a container as a read-only directory tree
//! in the foreground until it is told to stop. Linux: SIGINT, SIGTERM or SIGHUP, or `fusermount3 -u`.
//! Windows: Ctrl-C or Ctrl-Break. A second signal or key, while files are still open, ends the process
//! at once with status 128 + the signal number (130 for Ctrl-C, 149 for Ctrl-Break).
//!
//! Key functions: [`run`].
//!
//! Implementation notes: one flow for both systems, with `cfg` blocks where the backends differ: how
//! the stop request is caught, the name policy, the backend options, and the name of what abandoned the
//! mount. On Linux the order of the steps is a contract of `ziptensors_mount::UnmountSignals`: the
//! signals are blocked before `Container::open`, which starts the decode pool, so that no thread but the
//! mount's own signal thread can take them. On Windows the console handler is installed at the same
//! point, so a Ctrl-C during the open is not lost. Names are checked under the system's policy before
//! anything is mounted (and on Windows before `WinFsp` is loaded), and a bad tree fails the command
//! naming every bad member.

use std::fs;
use std::io::Write;
use std::sync::Arc;

use anyhow::Context;
use ziptensors_core::{CacheOptions, ReadCache};
use ziptensors_mount::{MountEnd, MountFs, UnmountSignals, mount, run_until_signal};
#[cfg(target_os = "linux")]
use ziptensors_mount::{FuseOptions, PosixPolicy};
#[cfg(windows)]
use ziptensors_mount::{WinFspOptions, WindowsPolicy};

use crate::cli::MountArgs;
use crate::commands::{STDOUT_ERROR, open_budgeted};
use crate::report;

/// `mount`: serves the container until a stop request or an outside unmount ends the mount. Does not
/// return after a second request: the process exits with [`MountEnd::exit_code`].
///
/// # Errors
/// Returns why the container cannot be opened or mounted (with every bad member name, or on Windows
/// a missing `WinFsp`), or the final error of the session.
pub(crate) fn run(args: &MountArgs, out: &mut impl Write) -> anyhow::Result<()> {
    #[cfg(target_os = "linux")]
    let signals = UnmountSignals::block().context("cannot block the signals that unmount")?;
    #[cfg(windows)]
    let signals = UnmountSignals::install().context("cannot install the Ctrl-C handler that unmounts")?;
    let container = open_budgeted(&args.container, args.threads, &args.budget)?;
    let modified = fs::metadata(&args.container)
        .and_then(|meta| meta.modified())
        .with_context(|| format!("cannot read the time of {}", args.container.display()))?;
    let options = CacheOptions { max_bytes: args.cache_size, readahead_windows: args.readahead };
    let cache = ReadCache::new(container, options)
        .with_context(|| format!("cannot use a cache of {} bytes", args.cache_size))?;
    #[cfg(target_os = "linux")]
    let policy = PosixPolicy;
    #[cfg(windows)]
    let policy = WindowsPolicy::default();
    let fs = MountFs::new(cache, policy, modified)
        .with_context(|| format!("cannot mount {}", args.container.display()))?;
    let fs = Arc::new(fs);
    #[cfg(target_os = "linux")]
    let backend = FuseOptions { threads: args.fuse_threads, fs_name: args.container.display().to_string() };
    #[cfg(windows)]
    let backend = WinFspOptions { threads: args.winfsp_threads, long_paths: policy.long_paths };
    #[cfg(target_os = "linux")]
    let handle = mount(Arc::clone(&fs), &args.mountpoint, &backend)
        .with_context(|| format!("cannot mount on {}", args.mountpoint.display()))?;
    // The backend's errors already name the mount point where it matters.
    #[cfg(windows)]
    let handle = mount(Arc::clone(&fs), &args.mountpoint, &backend).context("mount failed")?;
    // As the backend resolved it: `X:\` for a drive, an absolute directory on Windows.
    let mountpoint = handle.mountpoint().to_owned();
    report::write_mounted(out, &args.container, &mountpoint).context(STDOUT_ERROR)?;
    #[cfg(target_os = "linux")]
    let end = run_until_signal(handle, &signals)
        .with_context(|| format!("the mount on {} failed", mountpoint.display()))?;
    #[cfg(windows)]
    let end = run_until_signal(handle, &signals);
    let stats = fs.stats();
    tracing::info!(
        hits = stats.hits,
        misses = stats.misses,
        decoded_bytes = stats.decoded_bytes,
        evictions = stats.evictions,
        resident_bytes = stats.resident_bytes,
        "read cache at unmount"
    );
    match end {
        MountEnd::Unmounted => report::write_unmounted(out, &mountpoint, &stats).context(STDOUT_ERROR),
        #[cfg(target_os = "linux")]
        MountEnd::Abandoned { signal } => abandon(out, &mountpoint, signal.as_str(), end),
        #[cfg(windows)]
        MountEnd::Abandoned { event } => abandon(out, &mountpoint, event.as_str(), end),
        // A failure, not an unmount: exit 1 through the error report, which also logs it.
        #[cfg(windows)]
        MountEnd::Stopped { status } => Err(anyhow::anyhow!(
            "the WinFsp driver stopped the volume on {} (NTSTATUS {status:#010x}); see the log above",
            mountpoint.display()
        )),
    }
}

/// Reports a mount abandoned by `what` and leaves the process with the status of `end`.
///
/// # Errors
/// Returns the failure of the write to `out`; otherwise it does not return.
fn abandon(out: &mut impl Write, mountpoint: &std::path::Path, what: &str, end: MountEnd) -> anyhow::Result<()> {
    report::write_abandoned(out, mountpoint, what).context(STDOUT_ERROR)?;
    // The session still serves open files and nothing will end it: leaving the process is the only way
    // out. Stdout is flushed above and nothing else holds unwritten state.
    std::process::exit(i32::from(end.exit_code()));
}
