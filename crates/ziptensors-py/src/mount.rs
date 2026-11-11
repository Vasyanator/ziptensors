//! File: mount.rs
//! Purpose: the Python surface of the read-only mount: `mount` (one container) and `mount_overlay`
//! (a directory tree whose per-file containers show as their original files), both returning a
//! `Mount`, and the exception `MountError`.
//!
//! Main responsibilities: build the `ziptensors-mount` filesystem exactly as the CLI does (decode
//! threads, `ReadCache` sizing, name policy and backend options of the running OS), mount it with
//! the GIL released, and hand Python a handle whose `unmount` (also `__exit__` and, as a last resort,
//! the drop) ends the mount.
//!
//! Key structures: [`Mount`], [`MountError`], the private `Failure` (every way a mount call fails,
//! before it becomes a Python exception) and `State` (`Mounted` -> `Unmounting` -> `Finished`,
//! behind a mutex and a condvar, so that concurrent `unmount` calls all wait for the one that runs
//! and return its outcome).
//!
//! Key functions: [`mount`], [`mount_overlay`].
//!
//! Implementation notes: a backend exists on Linux (FUSE) and Windows (`WinFsp`) only. On macOS the
//! functions and the class exist, so the stub is one for all platforms, and every call raises
//! `MountError` with the code `UnsupportedFeature`; the handle type there is uninhabited. The
//! filesystem threads never touch Python, and nothing here masks a signal: the interpreter keeps
//! Ctrl-C. Windows mounts load `WinFsp` - Windows File System Proxy, Copyright (C) Bill
//! Zissimopoulos, <https://github.com/winfsp/winfsp> (GPL-3.0 with a FLOSS exception), whose notice
//! and link the docstrings and the stub carry as `ARCHITECTURE.md` "Platforms" requires.

use std::path::PathBuf;
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};

use pyo3::create_exception;
#[cfg(any(target_os = "linux", windows))]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ziptensors_core::{CacheOptions, ErrorCode};

use crate::errors::ZiptensorsError;
#[cfg(any(target_os = "linux", windows))]
use crate::errors::to_py_err;

create_exception!(
    ziptensors,
    MountError,
    ZiptensorsError,
    "A mount cannot be made or ended: the platform has no mount backend (macOS), a name cannot be shown, an overlay entry is damaged or claimed twice, the backend (FUSE, WinFsp) refused, or an unmount left the mount in place. `code` is the ErrorCode of the cause."
);

/// Threads that serve requests of the kernel (FUSE) or of the driver (`WinFsp`): the CLI's default of
/// `--fuse-threads` and `--winfsp-threads`.
#[cfg(any(target_os = "linux", windows))]
const SERVE_THREADS: u8 = 4;

/// The handle of a live mount on this OS.
#[cfg(any(target_os = "linux", windows))]
type Handle = ziptensors_mount::MountHandle;

/// No mount backend exists on this OS, so no handle can exist either.
#[cfg(not(any(target_os = "linux", windows)))]
#[derive(Debug)]
enum Handle {}

/// Why a mount call failed, kept as Rust values until a Python exception is built from it.
#[derive(Debug)]
enum Failure {
    /// The core refused: opening the container, or a cache too small for its chunks.
    #[cfg(any(target_os = "linux", windows))]
    Core(ziptensors_core::Error),
    /// The mount crate refused to build the filesystem.
    #[cfg(any(target_os = "linux", windows))]
    Mount(ziptensors_mount::MountError),
    /// An operating system call around the mount failed: `what` names it.
    #[cfg(any(target_os = "linux", windows))]
    Io {
        /// What was attempted, with its path.
        what: String,
        /// The error text, with its sources.
        source: String,
    },
    /// No mount backend exists on this OS.
    #[cfg(not(any(target_os = "linux", windows)))]
    Unsupported,
}

/// `error` with every source after it, joined by `": "`: Python sees one message, not a chain.
#[cfg(any(target_os = "linux", windows))]
fn with_sources(error: &dyn std::error::Error) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        text.push_str(": ");
        text.push_str(&cause.to_string());
        source = cause.source();
    }
    text
}

/// A `MountError` exception with `message` and `code`, and `cause` as its `__cause__`.
///
/// Never fails: if `code` cannot be set, which takes a failure of the interpreter itself, that
/// failure is what the caller raises.
fn mount_error(py: Python<'_>, message: String, code: ErrorCode, cause: Option<PyErr>) -> PyErr {
    let err = MountError::new_err(message);
    err.set_cause(py, cause);
    match err.value(py).setattr("code", code.as_i32()) {
        Ok(()) => err,
        Err(attribute_error) => attribute_error,
    }
}

impl Failure {
    /// The Python exception of this failure. A core error is raised as the module raises it
    /// anywhere (`FormatError`, `BudgetError`...); an option of the mount crate out of range is a
    /// `ValueError`; everything else is `MountError` with the `code` its cause has.
    fn into_py(self, py: Python<'_>) -> PyErr {
        match self {
            #[cfg(any(target_os = "linux", windows))]
            Self::Core(error) => to_py_err(py, &error),
            #[cfg(any(target_os = "linux", windows))]
            Self::Mount(error) => mount_crate_error(py, &error),
            #[cfg(any(target_os = "linux", windows))]
            Self::Io { what, source } => mount_error(py, format!("{what}: {source}"), ErrorCode::Io, None),
            #[cfg(not(any(target_os = "linux", windows)))]
            Self::Unsupported => mount_error(
                py,
                "mounting needs Linux (FUSE) or Windows (WinFsp); this platform has no mount backend".to_owned(),
                ErrorCode::UnsupportedFeature,
                None,
            ),
        }
    }
}

/// The Python exception of an error of the mount crate: see [`Failure::into_py`].
#[cfg(any(target_os = "linux", windows))]
fn mount_crate_error(py: Python<'_>, error: &ziptensors_mount::MountError) -> PyErr {
    use ziptensors_mount::MountError as E;
    use ziptensors_mount::OverlayProblem;

    let message = with_sources(error);
    let (code, cause) = match error {
        E::OverlayOption { .. } => return PyValueError::new_err(message),
        E::Names(_) => (ErrorCode::UnsafeMemberName, None),
        E::Member { source, .. } | E::Scan { source, .. } => (source.code(), Some(to_py_err(py, source))),
        // Problems are in path order; the first one gives the code, the message names all.
        E::Overlay { problems, .. } => {
            let code = match problems.first() {
                Some(OverlayProblem::Broken { error, .. }) => error.code(),
                Some(OverlayProblem::Conflict(_)) => ErrorCode::TreeConflict,
                // `OverlayProblem` is `#[non_exhaustive]` in another crate; `Unsupported` and any
                // later kind are entries this version cannot show.
                Some(_) | None => ErrorCode::UnsupportedFeature,
            };
            (code, None)
        }
        E::OverlayRoot { .. } => (ErrorCode::Io, None),
        #[cfg(target_os = "linux")]
        E::Unmount { .. } | E::Session { .. } => (ErrorCode::Io, None),
        #[cfg(windows)]
        E::Stopped { .. } => (ErrorCode::Io, None),
    };
    mount_error(py, message, code, cause)
}

/// The read cache of a mount, as the CLI sizes it: `cache_size` bytes (None: the core's default,
/// 256 MiB) and `readahead` windows per miss (0: the decode thread count).
fn cache_options(cache_size: Option<u64>, readahead: usize) -> CacheOptions {
    CacheOptions { max_bytes: cache_size.unwrap_or(CacheOptions::default().max_bytes), readahead_windows: readahead }
}

/// Hands `fs` to this OS's backend at `mountpoint`. `source` is the path shown as the FUSE source
/// in `/proc/mounts`. Blocks until the mount exists; call it with the GIL released.
///
/// # Errors
/// [`Failure::Io`] with the backend's error: no `/dev/fuse` or `fusermount3`, a mountpoint that is
/// not an existing directory (Linux) or not a free drive letter or new directory (Windows), no
/// `WinFsp` installed (naming <https://winfsp.dev/rel/>).
#[cfg(any(target_os = "linux", windows))]
fn serve(fs: ziptensors_mount::MountFs, mountpoint: &std::path::Path, source: &std::path::Path) -> Result<Handle, Failure> {
    let fs = std::sync::Arc::new(fs);
    let failed = |error: &dyn std::error::Error| Failure::Io {
        what: format!("cannot mount on {}", mountpoint.display()),
        source: with_sources(error),
    };
    #[cfg(target_os = "linux")]
    {
        let options = ziptensors_mount::FuseOptions {
            threads: usize::from(SERVE_THREADS),
            fs_name: source.display().to_string(),
        };
        ziptensors_mount::mount(fs, mountpoint, &options).map_err(|error| failed(&error))
    }
    #[cfg(windows)]
    {
        // A WinFsp volume has no source name, so `source` only reaches the log here.
        tracing::debug!(source = %source.display(), mountpoint = %mountpoint.display(), "mounting with WinFsp");
        let options = ziptensors_mount::WinFspOptions {
            threads: u32::from(SERVE_THREADS),
            long_paths: name_policy().long_paths,
        };
        ziptensors_mount::mount(fs, mountpoint, &options).map_err(|error| failed(&error))
    }
}

/// The name policy of this OS's backend, as the CLI chooses it.
#[cfg(target_os = "linux")]
fn name_policy() -> ziptensors_mount::PosixPolicy {
    ziptensors_mount::PosixPolicy
}

/// The name policy of this OS's backend, as the CLI chooses it; `WinFsp` refuses any other.
#[cfg(windows)]
fn name_policy() -> ziptensors_mount::WindowsPolicy {
    ziptensors_mount::WindowsPolicy::default()
}

/// Opens `container` and mounts every member under `mountpoint`, as `ziptensors mount` does.
/// Blocking; call it with the GIL released.
///
/// # Errors
/// [`Failure::Core`] if the container does not open or the cache cannot hold its largest window,
/// [`Failure::Io`] if its modification time cannot be read or the backend refuses,
/// [`Failure::Mount`] naming every member name the OS cannot show.
#[cfg(any(target_os = "linux", windows))]
fn mount_container(
    container: &std::path::Path,
    mountpoint: &std::path::Path,
    open: &ziptensors_core::OpenOptions,
    cache: CacheOptions,
) -> Result<Handle, Failure> {
    let opened = ziptensors_core::Container::open(container, open).map_err(Failure::Core)?;
    let modified = std::fs::metadata(container).and_then(|meta| meta.modified()).map_err(|error| Failure::Io {
        what: format!("cannot read the time of {}", container.display()),
        source: error.to_string(),
    })?;
    let cache = ziptensors_core::ReadCache::new(opened, cache).map_err(Failure::Core)?;
    let fs = ziptensors_mount::MountFs::new(cache, name_policy(), modified).map_err(Failure::Mount)?;
    serve(fs, mountpoint, container)
}

/// Scans `directory` and mounts its overlay under `mountpoint`. Blocking; call it with the GIL
/// released.
///
/// # Errors
/// [`Failure::Mount`] for an option out of range, a root that is no directory, a failed scan, a
/// damaged or conflicting entry, or a name the OS cannot show; [`Failure::Io`] if the backend
/// refuses.
#[cfg(any(target_os = "linux", windows))]
fn mount_directory(
    directory: &std::path::Path,
    mountpoint: &std::path::Path,
    options: &ziptensors_mount::OverlayOptions,
) -> Result<Handle, Failure> {
    let fs = ziptensors_mount::MountFs::overlay(directory, name_policy(), options).map_err(Failure::Mount)?;
    serve(fs, mountpoint, directory)
}

/// Ends the mount of `handle`: detaches it at once, then waits until every file open under it is
/// closed. Blocking; call it with the GIL released.
///
/// # Errors
/// The message of the mount crate's error, with its sources, when the mount stays (Linux: run
/// `fusermount3 -u -z` on it), the session failed, or (Windows) the driver had stopped the volume.
/// Every such failure is raised as `MountError` with the code `Io`, so the message is all a later
/// caller needs to raise it again.
#[cfg(any(target_os = "linux", windows))]
fn end(handle: Handle) -> Result<(), String> {
    handle.unmount().map_err(|error| with_sources(&error))
}

/// Uninhabited: no handle exists where no mount backend does.
#[cfg(not(any(target_os = "linux", windows)))]
#[expect(clippy::needless_pass_by_value, reason = "the same signature as the backend `end`, which consumes the handle")]
fn end(handle: Handle) -> Result<(), String> {
    match handle {}
}

/// How an unmount ended: `Err` holds the message of the `MountError` it raised. Kept as a plain
/// value because a `PyErr` cannot be handed to every caller; each one builds its own exception.
type Outcome = Result<(), String>;

/// Where a [`Mount`] is in its life. Only forward moves: `Mounted` -> `Unmounting` -> `Finished`.
#[derive(Debug)]
enum State {
    /// The mount is live and nobody is unmounting it.
    #[cfg_attr(
        not(any(target_os = "linux", windows)),
        expect(dead_code, reason = "no mount is made where no mount backend exists; `Mount` exists for the stub")
    )]
    Mounted(Handle),
    /// One caller runs the backend's unmount, without the lock; the others wait on the condvar.
    Unmounting,
    /// The unmount ended, with this outcome, which every later call returns again.
    Finished(Outcome),
}

/// Records the outcome of the one running unmount and wakes every caller waiting for it, also when
/// the unmount panics (the outcome then stays the failure it starts with), so no waiter is left
/// waiting on a state that would never move.
struct Publish<'a> {
    /// The mount whose state becomes `Finished`.
    mount: &'a Mount,
    /// What the waiters will see.
    outcome: Outcome,
}

impl Drop for Publish<'_> {
    fn drop(&mut self) {
        *self.mount.lock() = State::Finished(self.outcome.clone());
        self.mount.finished.notify_all();
    }
}

/// A live read-only mount, made by `mount` or `mount_overlay`. A context manager: leaving the `with`
/// block unmounts, whether or not it raised (`KeyboardInterrupt` included).
///
/// Unmounting detaches the mountpoint at once and then waits until every file still open under it
/// is closed (memory maps included), so close them first: a file the unmounting thread itself
/// holds open makes it wait forever. Concurrent `unmount()` calls all return when that wait ends,
/// with the same outcome. A `Mount` dropped while mounted is unmounted on a thread of its own, with
/// a warning in the log; rely on `with` or `unmount()` instead.
#[pyclass(frozen, module = "ziptensors")]
#[derive(Debug)]
pub(crate) struct Mount {
    /// Where it is mounted, as the backend resolved it (`X:\` for a Windows drive).
    mountpoint: PathBuf,
    /// The life of the mount. The lock is held only to read or move the state, never across the
    /// backend's unmount.
    state: Mutex<State>,
    /// Signalled when the state becomes `Finished`.
    finished: Condvar,
}

impl Mount {
    /// A Python handle over the live mount `handle`.
    #[cfg(any(target_os = "linux", windows))]
    fn new(handle: Handle) -> Self {
        Self {
            mountpoint: handle.mountpoint().to_owned(),
            state: Mutex::new(State::Mounted(handle)),
            finished: Condvar::new(),
        }
    }

    /// The state's lock. Nothing panics while it is held (only moves of the state happen under it),
    /// so a poisoned lock still holds a consistent state and is used as it is.
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Ends the mount, or waits for the caller already ending it, and returns the outcome every
    /// caller gets. Blocking; call it with the GIL released.
    fn finish(&self) -> Outcome {
        let mut state = self.lock();
        loop {
            match std::mem::replace(&mut *state, State::Unmounting) {
                State::Mounted(handle) => {
                    drop(state);
                    // Published on drop, also if `end` panics.
                    let mut publish =
                        Publish { mount: self, outcome: Err("the unmount panicked".to_owned()) };
                    publish.outcome = end(handle);
                    return publish.outcome.clone();
                }
                // Put back as it was; another caller is unmounting.
                State::Unmounting => {
                    state = self.finished.wait(state).unwrap_or_else(PoisonError::into_inner);
                }
                State::Finished(outcome) => {
                    *state = State::Finished(outcome.clone());
                    return outcome;
                }
            }
        }
    }
}

#[pymethods]
impl Mount {
    /// Where the files are, as the backend resolved it (`X:\` for a Windows drive letter).
    #[getter]
    fn mountpoint(&self) -> PathBuf {
        self.mountpoint.clone()
    }

    /// True until an unmount by this object has ended: also while one is still waiting for open
    /// files, False after it ended, even if it failed. An unmount from outside (`fusermount3 -u`)
    /// is not seen here.
    #[getter]
    fn is_mounted(&self) -> bool {
        match &*self.lock() {
            State::Mounted(_) | State::Unmounting => true,
            State::Finished(_) => false,
        }
    }

    /// Unmounts and waits, with the GIL released, until every file open under the mount is closed.
    /// A call while another thread unmounts waits for that unmount and gets its outcome; a call
    /// after it ended returns that outcome again. On failure raises `MountError` (code `Io`), and
    /// this object gives the mount up: on Linux it then stays until `fusermount3 -u -z` removes it.
    fn unmount(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.finish()).map_err(|message| mount_error(py, message, ErrorCode::Io, None))
    }

    /// Returns the mount itself, for `with`.
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Unmounts, whether or not the block raised. Never swallows an exception of the block; a failed
    /// unmount raises `MountError`, with the block's exception as its context.
    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        self.unmount(py)?;
        Ok(false)
    }
}

impl Drop for Mount {
    /// Unmounts a mount nobody unmounted, on a thread of its own, and never blocks: a drop runs with
    /// the GIL held, often in the garbage collector, and an unmount waits for files a Python object
    /// may still hold open, which would deadlock here.
    fn drop(&mut self) {
        let state = std::mem::replace(
            self.state.get_mut().unwrap_or_else(PoisonError::into_inner),
            State::Finished(Ok(())),
        );
        // `Unmounting` cannot be seen here: the unmounting call borrows the object, which keeps it
        // alive until that call returns.
        let State::Mounted(handle) = state else {
            return;
        };
        tracing::warn!(
            mountpoint = %self.mountpoint.display(),
            "a Mount was dropped while mounted; unmounting it in the background (use `with` or unmount())"
        );
        // `ManuallyDrop`: if the thread cannot start, `spawn` drops the closure here, and the
        // handle's own drop would unmount, and wait, on this thread. Leaked instead, the session
        // keeps serving until the process exits; the error below names the cure.
        let handle = std::mem::ManuallyDrop::new(handle);
        let spawned = std::thread::Builder::new().name("ziptensors-unmount".to_owned()).spawn(move || {
            if let Err(message) = end(std::mem::ManuallyDrop::into_inner(handle)) {
                tracing::error!(%message, "cannot unmount a dropped Mount");
            }
        });
        if let Err(error) = spawned {
            tracing::error!(
                mountpoint = %self.mountpoint.display(),
                %error,
                "cannot start the unmount thread; the mount is left until the process exits \
                 (on Linux run `fusermount3 -u -z` on it)"
            );
        }
    }
}

/// Mounts the members of `container` read-only as files under `mountpoint` and returns the `Mount`;
/// use it as a context manager. Linux (FUSE: `/dev/fuse` and `fusermount3`, no root): `mountpoint`
/// is an existing directory. Windows: a free drive letter (`X:`) or a directory that does not exist
/// yet, created and removed by the mount; it needs `WinFsp` - Windows File System Proxy, Copyright
/// (C) Bill Zissimopoulos, <https://github.com/winfsp/winfsp> (GPL-3.0 with a FLOSS exception),
/// installed from <https://winfsp.dev/rel/>. macOS has no backend: `MountError`.
///
/// `threads`: decode threads, None or 0 one per logical CPU. `cache_size`: decoded bytes kept for
/// small reads, None 256 MiB, at least twice the largest chunk. `readahead`: windows decoded per
/// miss, 0 the thread count. `max_total_decoded`: the budget of `Container`; set it for a file you
/// do not trust. Setup runs with the GIL released.
#[pyfunction]
#[cfg_attr(
    not(any(target_os = "linux", windows)),
    expect(clippy::needless_pass_by_value, reason = "one signature on every OS; only the backend paths consume the arguments")
)]
#[pyo3(signature = (container, mountpoint, *, threads=None, cache_size=None, readahead=0, max_total_decoded=None))]
pub(crate) fn mount(
    py: Python<'_>,
    container: PathBuf,
    mountpoint: PathBuf,
    threads: Option<usize>,
    cache_size: Option<u64>,
    readahead: usize,
    max_total_decoded: Option<u64>,
) -> PyResult<Mount> {
    let cache = cache_options(cache_size, readahead);
    #[cfg(any(target_os = "linux", windows))]
    {
        let defaults = ziptensors_core::OpenOptions::default();
        let open = ziptensors_core::OpenOptions {
            threads: threads.unwrap_or(0),
            max_total_decoded: max_total_decoded.unwrap_or(defaults.max_total_decoded),
            ..defaults
        };
        let handle = py.detach(move || mount_container(&container, &mountpoint, &open, cache));
        handle.map(Mount::new).map_err(|failure| failure.into_py(py))
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        tracing::debug!(
            container = %container.display(),
            mountpoint = %mountpoint.display(),
            ?threads,
            ?cache,
            ?max_total_decoded,
            "mount refused: no mount backend on this platform"
        );
        Err(Failure::Unsupported.into_py(py))
    }
}

/// Mounts `directory` read-only under `mountpoint` as an overlay and returns the `Mount`: plain
/// files and packs as they are, each per-file container as its original file. The tree is a
/// snapshot taken now; a file changed later fails its reads. Damaged entries, names claimed twice
/// and names the OS cannot show refuse the whole mount with `MountError`. Platforms, `mountpoint`,
/// `WinFsp` - Windows File System Proxy, Copyright (C) Bill Zissimopoulos,
/// <https://github.com/winfsp/winfsp> (install from <https://winfsp.dev/rel/>), and `threads`,
/// `readahead`, `max_total_decoded` are as for `mount`; the last applies to each container.
///
/// `cache_size` (None 256 MiB) is shared: each open container gets `cache_size /
/// max_open_containers`, which must hold 16 MiB, or `ValueError`. At most `max_open_containers`
/// containers and `max_open_files` plain files stay open, least recently used closed first.
#[pyfunction]
#[cfg_attr(
    not(any(target_os = "linux", windows)),
    expect(clippy::needless_pass_by_value, reason = "one signature on every OS; only the backend paths consume the arguments")
)]
#[pyo3(signature = (
    directory,
    mountpoint,
    *,
    threads=None,
    cache_size=None,
    readahead=0,
    max_total_decoded=None,
    // The defaults of `OverlayOptions::default()` in the mount crate.
    max_open_containers=4,
    max_open_files=64,
))]
#[expect(clippy::too_many_arguments, reason = "one per keyword argument of the Python function")]
pub(crate) fn mount_overlay(
    py: Python<'_>,
    directory: PathBuf,
    mountpoint: PathBuf,
    threads: Option<usize>,
    cache_size: Option<u64>,
    readahead: usize,
    max_total_decoded: Option<u64>,
    max_open_containers: usize,
    max_open_files: usize,
) -> PyResult<Mount> {
    let cache = cache_options(cache_size, readahead);
    #[cfg(any(target_os = "linux", windows))]
    {
        // The options are `#[non_exhaustive]`: start from the defaults and set what Python passed.
        let mut options = ziptensors_mount::OverlayOptions::default();
        options.threads = threads.unwrap_or(0);
        options.cache = cache;
        options.max_total_decoded = max_total_decoded.unwrap_or(options.max_total_decoded);
        options.max_open_containers = max_open_containers;
        options.max_open_files = max_open_files;
        let handle = py.detach(move || mount_directory(&directory, &mountpoint, &options));
        handle.map(Mount::new).map_err(|failure| failure.into_py(py))
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        tracing::debug!(
            directory = %directory.display(),
            mountpoint = %mountpoint.display(),
            ?threads,
            ?cache,
            ?max_total_decoded,
            max_open_containers,
            max_open_files,
            "overlay mount refused: no mount backend on this platform"
        );
        Err(Failure::Unsupported.into_py(py))
    }
}
