//! File: overlay.rs
//! Purpose: the backing of an overlay mount: a directory tree shown read-only, each per-file container
//! under its member's name with decoded bytes, every other regular file as it is on disk.
//!
//! Main responsibilities:
//! - [`plan`]: turn a core `TreeView` into the files and directories an overlay shows, refusing
//!   damaged files and names claimed twice, hiding a container whose original is still there;
//! - [`Overlay`]: open containers and plain files lazily, keep at most a fixed number of each open
//!   (least recently used first out), and fail a read of a file that changed since the mount;
//! - [`OverlayOptions`], the limits of an overlay.
//!
//! Key structures: [`OverlayOptions`], [`Overlay`], [`OverlayFile`], [`Planned`], `HandlePool`.
//!
//! Implementation notes: the view is a snapshot taken at mount time, which matches the backends'
//! caching (FUSE: one hour TTL and `KEEP_CACHE`; `WinFsp`: no expiry). A file's length and
//! modification time are compared with the snapshot when it is opened and after every read, so a read
//! of a file changed on disk is an I/O error, never a mix of old and new bytes; a cached container
//! window decoded before the change is still the snapshot's bytes. Opening and closing happen outside
//! the pool lock: a slot is reserved as `Opening`, and other readers of that file wait for it.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::SystemTime;

use ziptensors_core::chain::MAX_CHUNK_SIZE;
use ziptensors_core::tree::{FileStamp, TreeView, ViewEntry};
use ziptensors_core::{CacheOptions, CacheStats, CachedMember, Container, Error, FileSource, OpenOptions, RandomAccess, ReadCache};

use crate::error::{FsError, MountError, OverlayProblem};

/// Default of [`OverlayOptions::max_open_containers`].
const DEFAULT_MAX_OPEN_CONTAINERS: usize = 4;

/// Default of [`OverlayOptions::max_open_files`].
const DEFAULT_MAX_OPEN_FILES: usize = 64;

/// The limits of an overlay mount. Build it from [`OverlayOptions::default`] and set fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct OverlayOptions {
    /// Decode threads of each open container, as [`OpenOptions::threads`]: 0 is one per logical CPU,
    /// 1 decodes on the reading thread. The default is 0.
    pub threads: usize,
    /// The read cache budget of the whole mount. Each open container gets its own cache of
    /// `cache.max_bytes / max_open_containers` bytes, which must hold the largest window the format
    /// allows (2 × `chain::MAX_CHUNK_SIZE`, 16 MiB), since containers open lazily and their chunk
    /// sizes are unknown at mount. `cache.readahead_windows` applies to each. The default is
    /// [`CacheOptions::default`], 256 MiB.
    pub cache: CacheOptions,
    /// Budget on the sum of member lengths of each container, in bytes, for the scan and for every
    /// open, as [`OpenOptions::max_total_decoded`]. The default is `u64::MAX`, no budget: a caller
    /// mounting untrusted files must set one.
    pub max_total_decoded: u64,
    /// Containers kept open at once, at least 1. The default is 4.
    pub max_open_containers: usize,
    /// Plain files kept open at once, at least 1. The default is 64.
    pub max_open_files: usize,
}

impl Default for OverlayOptions {
    fn default() -> Self {
        Self {
            threads: 0,
            cache: CacheOptions::default(),
            max_total_decoded: u64::MAX,
            max_open_containers: DEFAULT_MAX_OPEN_CONTAINERS,
            max_open_files: DEFAULT_MAX_OPEN_FILES,
        }
    }
}

impl OverlayOptions {
    /// The cache of each open container.
    ///
    /// # Errors
    /// [`MountError::OverlayOption`] for a limit of 0 open handles, or a cache share below the largest
    /// window.
    pub(crate) fn container_cache(&self) -> Result<CacheOptions, MountError> {
        let count = |option, value: usize| {
            let value = u64::try_from(value).unwrap_or(u64::MAX);
            if value == 0 {
                return Err(MountError::OverlayOption { option, value, reason: "must be at least 1" });
            }
            Ok(value)
        };
        let containers = count("max_open_containers", self.max_open_containers)?;
        count("max_open_files", self.max_open_files)?;
        let share = self.cache.max_bytes / containers;
        // A widening view doubles a chunk, and no view widens more (as `ReadCache::new` checks).
        let largest_window = u64::from(MAX_CHUNK_SIZE).saturating_mul(2);
        if share < largest_window {
            return Err(MountError::OverlayOption {
                option: "cache.max_bytes",
                value: self.cache.max_bytes,
                reason: "divided by max_open_containers, must hold one window of the largest chunk (16 MiB)",
            });
        }
        Ok(CacheOptions { max_bytes: share, ..self.cache })
    }
}

/// How a shown file gets its bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileKind {
    /// Read from the file itself.
    Plain,
    /// Decoded from the one member of a per-file container.
    PerFile {
        /// The member's stated BLAKE3, to recognize it at open.
        blake3: [u8; 32],
    },
}

/// One file an overlay shows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OverlayFile {
    /// The name it is shown under, `/`-separated, relative to the root.
    pub(crate) rel_path: String,
    /// Its length as shown: the file's, or the member's for a container.
    pub(crate) len: u64,
    /// The file read on disk: the file itself, or the container.
    path: PathBuf,
    /// The snapshot of that file.
    stamp: FileStamp,
    kind: FileKind,
}

impl OverlayFile {
    /// The modification time shown: that of the file on disk (the container's for a member).
    pub(crate) fn modified(&self) -> SystemTime {
        self.stamp.modified
    }
}

/// What an overlay shows of a view: its files, in view order, and its directories with their times.
#[derive(Debug)]
pub(crate) struct Planned {
    /// The files, in the order of the view.
    pub(crate) files: Vec<OverlayFile>,
    /// Every directory below the root, `/`-separated, with its modification time.
    pub(crate) dirs: Vec<(String, SystemTime)>,
}

/// What an overlay shows of `view`: plain files and containers of several members as they are on
/// disk, a per-file container as its member under the member's name, a per-file container whose name
/// is taken by its original as nothing (the original is shown, read directly), every directory, and
/// nothing of links, special files, temporary and partial files and names that are not UTF-8 (one log
/// line each).
///
/// # Errors
/// [`MountError::Overlay`] naming every conflict of the view, every broken file and every entry of
/// an unknown kind; nothing is shown then.
pub(crate) fn plan(view: TreeView) -> Result<Planned, MountError> {
    let mut problems: Vec<OverlayProblem> = view.conflicts.into_iter().map(OverlayProblem::Conflict).collect();
    let mut files = Vec::new();
    let mut dirs = Vec::new();
    for entry in view.entries {
        match entry {
            ViewEntry::Directory { rel_path, modified, .. } => dirs.push((rel_path, modified)),
            ViewEntry::Plain { path, rel_path, stamp } => {
                files.push(OverlayFile { rel_path, len: stamp.len, path, stamp, kind: FileKind::Plain });
            }
            ViewEntry::PerFile(per_file) => files.push(OverlayFile {
                rel_path: per_file.rel_path,
                len: per_file.member_len,
                path: per_file.container,
                stamp: per_file.stamp,
                kind: FileKind::PerFile { blake3: per_file.blake3 },
            }),
            ViewEntry::Shadowed(per_file) => {
                // By decision: the original is shown once, read directly, and its container hidden.
                tracing::info!(
                    container = %per_file.container.display(),
                    original = %per_file.presented.display(),
                    "overlay hides a container whose original is still present"
                );
            }
            ViewEntry::Foreign { path, rel_path, reason } => match FileStamp::of(&path) {
                Ok(stamp) => {
                    tracing::info!(path = %path.display(), ?reason, "overlay shows a container that is not per-file as it is");
                    files.push(OverlayFile { rel_path, len: stamp.len, path, stamp, kind: FileKind::Plain });
                }
                Err(error) => problems.push(OverlayProblem::Broken { path, error }),
            },
            ViewEntry::Broken { path, error, .. } => {
                tracing::error!(path = %path.display(), %error, "overlay refuses a broken file");
                problems.push(OverlayProblem::Broken { path, error });
            }
            ViewEntry::Skipped { path, reason, .. } => {
                tracing::info!(path = %path.display(), ?reason, "overlay leaves out");
            }
            // `ViewEntry` is `#[non_exhaustive]` in the core, so a catch-all arm is required. A kind
            // this version does not know fails closed: the mount is refused, never shown in part.
            other => problems.push(OverlayProblem::Unsupported { path: other.path().to_path_buf() }),
        }
    }
    if !problems.is_empty() {
        return Err(MountError::Overlay { root: view.root, problems });
    }
    Ok(Planned { files, dirs })
}

/// Why a read of an overlay file failed; logged, then [`FsError::Io`].
#[derive(Debug, thiserror::Error)]
enum Failure {
    /// The file on disk is not the one of the snapshot.
    #[error("the file changed since the mount (length {then_len} -> {now_len}, modified {then_modified:?} -> {now_modified:?})")]
    Changed {
        /// Length at the mount.
        then_len: u64,
        /// Length now.
        now_len: u64,
        /// Modification time at the mount.
        then_modified: SystemTime,
        /// Modification time now.
        now_modified: SystemTime,
    },
    /// The file opened is not as long as the snapshot's.
    #[error("the file opened has {now_len} bytes, {then_len} at the mount")]
    Length {
        /// Length at the mount.
        then_len: u64,
        /// Length of the file opened.
        now_len: u64,
    },
    /// The container's member is not the one of the snapshot.
    #[error("the container holds another member than at the mount")]
    OtherMember,
    /// The core failed to open or read.
    #[error(transparent)]
    Core(#[from] Error),
}

/// An open per-file container: its cache, for counters, and its member.
#[derive(Debug, Clone)]
struct OpenContainer {
    cache: ReadCache,
    member: CachedMember,
}

/// The files of an overlay mount and their open handles. `Send + Sync`; all methods take `&self`.
#[derive(Debug)]
pub(crate) struct Overlay {
    /// Indexed by `Node::File::member`.
    files: Vec<OverlayFile>,
    /// How containers are opened.
    open: OpenOptions,
    /// The cache of each open container.
    cache: CacheOptions,
    /// Open containers by file index.
    containers: HandlePool<OpenContainer>,
    /// Open plain files by file index.
    plains: HandlePool<Arc<FileSource>>,
    /// Counters of the caches already closed, so that [`Overlay::stats`] never goes back.
    retired: Mutex<CacheStats>,
}

impl Overlay {
    /// The backing of `files`, under `options`.
    ///
    /// # Errors
    /// [`MountError::OverlayOption`] for an option out of range.
    pub(crate) fn new(files: Vec<OverlayFile>, options: &OverlayOptions) -> Result<Self, MountError> {
        let cache = options.container_cache()?;
        let open = OpenOptions {
            threads: options.threads,
            max_total_decoded: options.max_total_decoded,
            ..OpenOptions::default()
        };
        Ok(Self {
            files,
            open,
            cache,
            containers: HandlePool::new(options.max_open_containers),
            plains: HandlePool::new(options.max_open_files),
            retired: Mutex::new(CacheStats::default()),
        })
    }

    /// Fills all of `dst` from byte `offset` of file `index`, which the caller clamped to the shown
    /// length.
    ///
    /// # Errors
    /// [`FsError::NotFound`] for an index without a file; [`FsError::Io`], logged with the path, the
    /// range and the cause, if the file changed since the mount, cannot be opened or read, or (for a
    /// container) fails to decode.
    pub(crate) fn read(&self, index: usize, offset: u64, dst: &mut [u8]) -> Result<(), FsError> {
        let file = self.files.get(index).ok_or(FsError::NotFound)?;
        let result = match file.kind {
            FileKind::Plain => self
                .plains
                .get(index, || open_plain(file), drop)
                .and_then(|source| source.read_exact_at(offset, dst).map_err(Failure::from)),
            FileKind::PerFile { blake3 } => self
                .containers
                .get(index, || self.open_container(file, blake3), |closed| self.retire(&closed))
                .and_then(|open| open.member.read_at(offset, dst).map_err(Failure::from)),
        };
        // Checked after the read: a change made before or while it ran shows in the stamp.
        match result.and_then(|()| check_unchanged(file)) {
            Ok(()) => Ok(()),
            Err(failure) => {
                tracing::error!(
                    path = %file.path.display(),
                    shown_as = %file.rel_path,
                    offset,
                    len = dst.len(),
                    error = %failure,
                    "overlay read failed"
                );
                Err(FsError::Io)
            }
        }
    }

    /// Opens the container of `file` and checks it still holds the member of the snapshot.
    fn open_container(&self, file: &OverlayFile, blake3: [u8; 32]) -> Result<OpenContainer, Failure> {
        check_unchanged(file)?;
        let container = Container::open(&file.path, &self.open)?;
        let cache = ReadCache::new(container, self.cache)?;
        let member = cache.member_at(0)?;
        if member.len() != file.len || member.member().blake3() != blake3 {
            return Err(Failure::OtherMember);
        }
        tracing::debug!(path = %file.path.display(), "overlay opened a container");
        Ok(OpenContainer { cache, member })
    }

    /// Adds the counters of a container that left the pool to the retired ones.
    fn retire(&self, closed: &OpenContainer) {
        let stats = closed.cache.stats();
        let mut retired = self.retired.lock().unwrap_or_else(PoisonError::into_inner);
        retired.hits = retired.hits.saturating_add(stats.hits);
        retired.misses = retired.misses.saturating_add(stats.misses);
        retired.decoded_bytes = retired.decoded_bytes.saturating_add(stats.decoded_bytes);
        retired.evictions = retired.evictions.saturating_add(stats.evictions);
        retired.peak_bytes = retired.peak_bytes.max(stats.peak_bytes);
    }

    /// Counters of every cache this overlay has had: hits, misses, decoded bytes and evictions summed
    /// over all of them (a closed cache counted as it was when it left the pool), resident and in
    /// flight bytes summed over the open ones, and as `peak_bytes` the larger of the largest peak of a
    /// closed cache and the sum of the open caches' peaks: an estimate, not a simultaneous peak.
    pub(crate) fn stats(&self) -> CacheStats {
        let open = self.containers.handles();
        let mut stats = *self.retired.lock().unwrap_or_else(PoisonError::into_inner);
        let mut open_peaks = 0_u64;
        for container in &open {
            let one = container.cache.stats();
            stats.hits = stats.hits.saturating_add(one.hits);
            stats.misses = stats.misses.saturating_add(one.misses);
            stats.decoded_bytes = stats.decoded_bytes.saturating_add(one.decoded_bytes);
            stats.evictions = stats.evictions.saturating_add(one.evictions);
            stats.resident_bytes = stats.resident_bytes.saturating_add(one.resident_bytes);
            stats.in_flight_bytes = stats.in_flight_bytes.saturating_add(one.in_flight_bytes);
            open_peaks = open_peaks.saturating_add(one.peak_bytes);
        }
        stats.peak_bytes = stats.peak_bytes.max(open_peaks);
        stats
    }

    /// Counters of the container pool, for tests.
    #[cfg(test)]
    pub(crate) fn container_pool(&self) -> PoolCounters {
        self.containers.counters()
    }

    /// Counters of the plain file pool, for tests.
    #[cfg(test)]
    pub(crate) fn plain_pool(&self) -> PoolCounters {
        self.plains.counters()
    }
}

/// Opens the plain file of `file`, checking it is still the snapshot's.
fn open_plain(file: &OverlayFile) -> Result<Arc<FileSource>, Failure> {
    check_unchanged(file)?;
    let source = FileSource::open(&file.path)?;
    if source.len() != file.len {
        // Replaced between the stamp and the open.
        return Err(Failure::Length { then_len: file.len, now_len: source.len() });
    }
    tracing::debug!(path = %file.path.display(), "overlay opened a file");
    Ok(Arc::new(source))
}

/// Whether the file on disk still has the snapshot's length and modification time.
///
/// # Errors
/// [`Failure::Changed`] if not, [`Failure::Core`] if its metadata cannot be read (it was removed).
fn check_unchanged(file: &OverlayFile) -> Result<(), Failure> {
    let now = FileStamp::of(&file.path)?;
    if now == file.stamp {
        return Ok(());
    }
    Err(Failure::Changed {
        then_len: file.stamp.len,
        now_len: now.len,
        then_modified: file.stamp.modified,
        now_modified: now.modified,
    })
}

/// Counters of a [`HandlePool`], for tests.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PoolCounters {
    /// Successful opens.
    pub(crate) opened: u64,
    /// Most slots, open and opening, held at once.
    pub(crate) peak: usize,
}

/// A slot of a [`HandlePool`].
#[derive(Debug)]
enum Slot<T> {
    /// A thread is opening the handle; others wait for it.
    Opening,
    /// The handle, and the tick of its last use.
    Ready { handle: T, used: u64 },
}

/// What a [`HandlePool`]'s lock guards.
#[derive(Debug)]
struct PoolState<T> {
    slots: HashMap<usize, Slot<T>>,
    /// Use counter, the clock of the least recently used order.
    tick: u64,
    #[cfg(test)]
    opened: u64,
    #[cfg(test)]
    peak: usize,
}

/// At most `cap` handles open at once, keyed by file index, least recently used out first. Opening
/// and closing happen without the lock. A closed handle lives on in the clones readers still hold.
#[derive(Debug)]
struct HandlePool<T> {
    cap: usize,
    state: Mutex<PoolState<T>>,
    /// Signalled when a slot leaves `Opening`.
    changed: Condvar,
}

impl<T: Clone> HandlePool<T> {
    /// An empty pool of at most `cap` handles; `cap` is at least 1 (checked by the options).
    fn new(cap: usize) -> Self {
        let state = PoolState {
            slots: HashMap::new(),
            tick: 0,
            #[cfg(test)]
            opened: 0,
            #[cfg(test)]
            peak: 0,
        };
        Self { cap: cap.max(1), state: Mutex::new(state), changed: Condvar::new() }
    }

    fn lock(&self) -> MutexGuard<'_, PoolState<T>> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The handle of `key`: the open one, the one another thread is opening (waited for), or a new
    /// one from `open`. To make room, the least recently used open handle is closed first, handed to
    /// `retire` without the lock; when every slot is being opened, the call waits. A failed open is
    /// not remembered: the next call tries again.
    ///
    /// # Errors
    /// What `open` returned.
    fn get<E>(
        &self,
        key: usize,
        open: impl FnOnce() -> Result<T, E>,
        mut retire: impl FnMut(T),
    ) -> Result<T, E> {
        let mut evicted = Vec::new();
        let mut state = self.lock();
        loop {
            state.tick += 1;
            let tick = state.tick;
            match state.slots.get_mut(&key) {
                Some(Slot::Ready { handle, used }) => {
                    *used = tick;
                    let handle = handle.clone();
                    drop(state);
                    evicted.into_iter().for_each(&mut retire);
                    return Ok(handle);
                }
                Some(Slot::Opening) => {
                    state = self.changed.wait(state).unwrap_or_else(PoisonError::into_inner);
                    continue;
                }
                None => {}
            }
            if state.slots.len() < self.cap {
                break;
            }
            let oldest = state
                .slots
                .iter()
                .filter_map(|(key, slot)| match slot {
                    Slot::Ready { used, .. } => Some((*used, *key)),
                    Slot::Opening => None,
                })
                .min();
            match oldest {
                Some((_, victim)) => {
                    if let Some(Slot::Ready { handle, .. }) = state.slots.remove(&victim) {
                        evicted.push(handle);
                    }
                }
                None => state = self.changed.wait(state).unwrap_or_else(PoisonError::into_inner),
            }
        }
        state.slots.insert(key, Slot::Opening);
        #[cfg(test)]
        {
            state.peak = state.peak.max(state.slots.len());
        }
        drop(state);
        evicted.into_iter().for_each(&mut retire);
        let mut guard = OpeningGuard { pool: self, key, armed: true };
        let result = open();
        guard.armed = false;
        let mut state = self.lock();
        match &result {
            Ok(handle) => {
                state.tick += 1;
                let used = state.tick;
                state.slots.insert(key, Slot::Ready { handle: handle.clone(), used });
                #[cfg(test)]
                {
                    state.opened += 1;
                }
            }
            Err(_) => {
                state.slots.remove(&key);
            }
        }
        drop(state);
        self.changed.notify_all();
        result
    }

    /// Clones of the open handles.
    fn handles(&self) -> Vec<T> {
        self.lock()
            .slots
            .values()
            .filter_map(|slot| match slot {
                Slot::Ready { handle, .. } => Some(handle.clone()),
                Slot::Opening => None,
            })
            .collect()
    }

    #[cfg(test)]
    fn counters(&self) -> PoolCounters {
        let state = self.lock();
        PoolCounters { opened: state.opened, peak: state.peak }
    }
}

/// Frees an `Opening` slot if its opener unwinds, so that waiters do not wait forever.
struct OpeningGuard<'a, T> {
    pool: &'a HandlePool<T>,
    key: usize,
    armed: bool,
}

impl<T> Drop for OpeningGuard<'_, T> {
    fn drop(&mut self) {
        if self.armed {
            self.pool.state.lock().unwrap_or_else(PoisonError::into_inner).slots.remove(&self.key);
            self.pool.changed.notify_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    use super::HandlePool;

    #[test]
    fn a_pool_keeps_at_most_its_cap_and_closes_the_least_recently_used() {
        let pool = HandlePool::new(2);
        let mut closed = Vec::new();
        for key in [1, 2, 1, 3, 1, 2] {
            let handle = pool.get(key, || Ok::<_, ()>(key * 10), |old| closed.push(old)).expect("opens");
            assert_eq!(handle, key * 10);
        }
        // 1, 2 open; 1 used; 3 closes 2; 1 used; 2 closes 3.
        assert_eq!(closed, [20, 30]);
        let counters = pool.counters();
        assert_eq!((counters.opened, counters.peak), (4, 2));
        let mut open = pool.handles();
        open.sort_unstable();
        assert_eq!(open, [10, 20]);
    }

    #[test]
    fn a_failed_open_is_retried_and_concurrent_readers_open_once() {
        let pool = Arc::new(HandlePool::new(1));
        assert_eq!(pool.get(7, || Err::<usize, _>("nope"), drop), Err("nope"));
        let opens = Arc::new(AtomicUsize::new(0));
        let readers: Vec<_> = (0..8)
            .map(|_| {
                let (pool, opens) = (Arc::clone(&pool), Arc::clone(&opens));
                thread::spawn(move || {
                    pool.get(7, || Ok::<_, ()>(opens.fetch_add(1, Ordering::SeqCst)), drop).expect("opens")
                })
            })
            .collect();
        for reader in readers {
            assert_eq!(reader.join().expect("reader"), 0);
        }
        assert_eq!(opens.load(Ordering::SeqCst), 1);
        assert_eq!(pool.counters().peak, 1);
    }
}
