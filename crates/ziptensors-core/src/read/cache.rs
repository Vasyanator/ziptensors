//! File: read/cache.rs
//! Purpose: an opt-in, bounded cache of decoded member bytes, for callers that read a member in
//! pieces smaller than a chunk: a FUSE mount sees 128 KiB reads, a Python file object buffered reads.
//!
//! Main responsibilities:
//! - [`ReadCache`]: windows of decoded bytes keyed by member and window start, evicted least recently
//!   used by bytes, filled with read-ahead on a miss;
//! - [`CachedMember`]: `read_at` served window by window from the cache;
//! - [`CacheOptions`] and [`CacheStats`].
//!
//! Key structures: [`ReadCache`], [`CachedMember`], `Inner`, `State`, `Slot`, `FillGuard`.
//!
//! Implementation notes: a window is what `Member::chunk_window` says, the part of one chunk one
//! extent shows, so a window is decoded by exactly one whole-chunk job and the cache needs no chunk
//! logic of its own. A miss reserves a run of absent windows as `Filling` under the lock, counting their
//! bytes in flight against the budget, decodes them with ONE `Member::read_many` with the lock released,
//! so the engine decodes them in parallel under its own bound, and publishes them under the lock. A run
//! that fails is retried as the missed window alone, so read-ahead never spreads a neighbour's error.
//! Waiters sleep on one `Condvar`. Decode budgets were checked at open and nothing here spends them.
//! A fill never lets its thread run unrelated rayon tasks while it holds `Filling` windows: from a rayon
//! worker it decodes on a scoped helper thread (`Inner::decode_run`), since `ThreadPool::install` from a
//! foreign worker would steal tasks that may wait on those very windows.

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

use crate::error::Error;

use super::batch::{broken, to_usize};
use super::container::Container;
use super::member::{Member, ReadRequest};

/// Default of [`CacheOptions::max_bytes`]: 256 MiB.
const DEFAULT_MAX_BYTES: u64 = 256 << 20;

/// How a [`ReadCache`] spends memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheOptions {
    /// Most decoded bytes the cache holds, resident and in flight together; see [`ReadCache`] for the
    /// one-window-per-miss excess. Must hold one window of the largest kind, which is 2 × the largest
    /// `chunk_size` of the container (a widening view doubles a chunk), or [`ReadCache::new`] fails.
    /// The check assumes widening without looking for it, which keeps it O(blobs) instead of
    /// O(extents) and costs at most one chunk of headroom. The default is 256 MiB.
    pub max_bytes: u64,
    /// Windows decoded by one miss, the missed window included: the rest are the windows that follow
    /// it in the same member. 0, the default, means the container's decode thread count, so that a
    /// miss is one batch of the engine with every thread busy. 1 means no read-ahead.
    pub readahead_windows: usize,
}

impl Default for CacheOptions {
    fn default() -> Self {
        Self { max_bytes: DEFAULT_MAX_BYTES, readahead_windows: 0 }
    }
}

/// Counters of a [`ReadCache`] since it was created, and its resident size now.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Window lookups served by bytes already decoded, including after waiting for another thread's fill.
    pub hits: u64,
    /// Window lookups that started a fill. One miss decodes one run of windows.
    pub misses: u64,
    /// Member bytes the cache has decoded, read-ahead included.
    pub decoded_bytes: u64,
    /// Windows dropped to stay within [`CacheOptions::max_bytes`].
    pub evictions: u64,
    /// Decoded bytes resident now, at most [`CacheOptions::max_bytes`].
    pub resident_bytes: u64,
    /// Bytes of the fills in flight now, read-ahead included.
    pub in_flight_bytes: u64,
    /// Most `resident_bytes + in_flight_bytes` seen at once. At most [`CacheOptions::max_bytes`] plus one
    /// window per concurrent miss; see [`ReadCache`].
    pub peak_bytes: u64,
}

/// Key of a window: member id in file order, and window start in the member.
type Key = (u32, u64);

/// Length of a window. Windows come from `Member::chunk_window`, so `start <= end`.
fn window_len(window: &Range<u64>) -> u64 {
    window.end.saturating_sub(window.start)
}

/// A window the cache knows about.
#[derive(Debug)]
enum Slot {
    /// Some thread is decoding it. Waiters sleep on `Inner::filled`.
    Filling,
    /// Decoded bytes, and the window's place in the LRU order.
    Ready { bytes: Arc<Vec<u8>>, tick: u64 },
}

/// Everything behind the lock.
///
/// Invariant: `resident_bytes + in_flight_bytes <= max_bytes`, except that the missed window of each
/// fill in flight may exceed it, since a miss always proceeds. Read-ahead is admitted only within the
/// budget. When its fill ends, a read-ahead window that finds no room is served but not kept; the
/// missed window is kept anyway, as the only resident window then, so `resident_bytes <= max_bytes`
/// still holds and the excess moves from in flight to resident until the next `make_room`.
#[derive(Debug, Default)]
struct State {
    slots: HashMap<Key, Slot>,
    /// Ready windows by last use, oldest first. Ticks are unique.
    lru: BTreeMap<u64, Key>,
    next_tick: u64,
    stats: CacheStats,
}

impl State {
    /// A fresh LRU tick.
    fn tick(&mut self) -> u64 {
        let tick = self.next_tick;
        self.next_tick += 1;
        tick
    }

    /// Drops least recently used windows until `incoming` more bytes fit `max_bytes` beside what is
    /// resident and in flight. Returns the number dropped and whether `incoming` now fits; it does not
    /// when nothing is left to drop.
    fn make_room(&mut self, incoming: u64, max_bytes: u64) -> (u64, bool) {
        let mut dropped = 0;
        loop {
            let held = self.stats.resident_bytes.saturating_add(self.stats.in_flight_bytes);
            if held.saturating_add(incoming) <= max_bytes {
                return (dropped, true);
            }
            let Some((_, key)) = self.lru.pop_first() else { return (dropped, false) };
            if let Some(Slot::Ready { bytes, .. }) = self.slots.remove(&key) {
                // Lossless: `usize` is at most 64 bits on every supported target.
                let len = bytes.len() as u64;
                self.stats.resident_bytes = self.stats.resident_bytes.saturating_sub(len);
                self.stats.evictions += 1;
                dropped += 1;
            }
        }
    }

    /// Records the current holding in `peak_bytes`.
    fn note_peak(&mut self) {
        let held = self.stats.resident_bytes.saturating_add(self.stats.in_flight_bytes);
        self.stats.peak_bytes = self.stats.peak_bytes.max(held);
    }

    /// Removes the reservations of `windows` of member `id` and returns their bytes to the budget.
    fn release(&mut self, id: u32, windows: &[Range<u64>]) {
        for window in windows {
            let key = (id, window.start);
            if matches!(self.slots.get(&key), Some(Slot::Filling)) {
                self.slots.remove(&key);
            }
            self.stats.in_flight_bytes = self.stats.in_flight_bytes.saturating_sub(window_len(window));
        }
    }
}

/// The shared part of a cache.
#[derive(Debug)]
struct Inner {
    container: Container,
    max_bytes: u64,
    /// Windows per miss, at least 1.
    run_windows: usize,
    state: Mutex<State>,
    /// Signalled whenever a fill publishes or abandons windows.
    filled: Condvar,
}

impl Inner {
    /// The state. A poisoned lock is taken over: no code under it leaves `State` half updated, since
    /// every update there is a map operation or counter arithmetic that completes or aborts the process.
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The decoded bytes of `window` of member `id`, which must be what `member.chunk_window` returned.
    /// Waits while another thread decodes it; otherwise decodes it and the windows after it.
    ///
    /// # Errors
    /// The error of decoding `window` alone, with [`Member::read_many`]: a failure of read-ahead never
    /// fails the missed window. A failed fill caches nothing, and a thread that was waiting for it
    /// retries the fill itself, so every caller gets the error of its own attempt.
    fn window(&self, member: &Member, id: u32, window: &Range<u64>) -> Result<Arc<Vec<u8>>, Error> {
        let key = (id, window.start);
        loop {
            if let Some(bytes) = self.resident_or_absent(key) {
                return Ok(bytes);
            }
            // Planning walks the index only, but still runs with the lock released.
            let plan = self.plan_run(member, window)?;
            let mut state = self.lock();
            if state.slots.contains_key(&key) {
                // Another thread got there between the two locks.
                continue;
            }
            // The missed window always proceeds. Read-ahead stops at the first window someone else
            // holds or is decoding, so no window is decoded twice at once, and at the budget left
            // beside the fills in flight, so that concurrent misses cannot multiply it.
            let mut run_bytes = window_len(window);
            let mut run = vec![window.clone()];
            for next in plan.into_iter().skip(1) {
                let with_next = run_bytes.saturating_add(window_len(&next));
                let held = state.stats.in_flight_bytes.saturating_add(with_next);
                if state.slots.contains_key(&(id, next.start)) || held > self.max_bytes {
                    break;
                }
                run_bytes = with_next;
                run.push(next);
            }
            for next in &run {
                state.slots.insert((id, next.start), Slot::Filling);
            }
            state.stats.in_flight_bytes = state.stats.in_flight_bytes.saturating_add(run_bytes);
            state.stats.misses += 1;
            // Room for the run is made now, while its bytes are being allocated, not when it lands.
            let (evicted, _) = state.make_room(0, self.max_bytes);
            state.note_peak();
            drop(state);
            if evicted > 0 {
                tracing::debug!(member = id, evicted, "read cache made room for a fill");
            }
            let fill = FillGuard { inner: self, id, run, published: false };
            return fill.decode_and_publish(member);
        }
    }

    /// The bytes of `key` if resident, counted as a hit; `None` if absent. Waits out a fill in flight.
    fn resident_or_absent(&self, key: Key) -> Option<Arc<Vec<u8>>> {
        let mut state = self.lock();
        loop {
            match state.slots.get(&key) {
                Some(Slot::Ready { bytes, tick }) => {
                    let (bytes, old) = (Arc::clone(bytes), *tick);
                    let new = state.tick();
                    state.lru.remove(&old);
                    state.lru.insert(new, key);
                    if let Some(Slot::Ready { tick, .. }) = state.slots.get_mut(&key) {
                        *tick = new;
                    }
                    state.stats.hits += 1;
                    return Some(bytes);
                }
                Some(Slot::Filling) => {
                    state = self.filled.wait(state).unwrap_or_else(PoisonError::into_inner);
                }
                None => return None,
            }
        }
    }

    /// `first` and the windows after it in the member, as many as one miss decodes, and no more bytes
    /// than `max_bytes` in all. Always holds `first`.
    fn plan_run(&self, member: &Member, first: &Range<u64>) -> Result<Vec<Range<u64>>, Error> {
        let mut run = vec![first.clone()];
        let mut bytes = window_len(first);
        let mut end = first.end;
        while run.len() < self.run_windows && end < member.len() {
            let next = member.chunk_window(end)?;
            bytes = bytes.saturating_add(window_len(&next));
            if bytes > self.max_bytes {
                break;
            }
            end = next.end;
            run.push(next);
        }
        Ok(run)
    }

    /// [`decode_windows`] on a thread that runs no unrelated work while the run's windows are held as
    /// `Filling`.
    ///
    /// A worker of another rayon pool that enters the container's pool (`ThreadPool::install`) runs
    /// tasks of its own pool while it waits, and one of them may wait on `filled` for a window this
    /// fill holds: a deadlock on one thread, or a cycle across two such workers. So when the caller is a
    /// rayon worker and the engine has a pool, the run is decoded on a scoped helper thread, and the
    /// caller blocks in a plain join, which runs nothing else. Other callers, and engines without a
    /// pool (which never install), decode on the calling thread at no extra cost.
    ///
    /// # Errors
    /// [`Error::Io`] with `op` = `"start read cache fill thread"` if the helper cannot be started;
    /// otherwise those of [`decode_windows`].
    ///
    /// # Panics
    /// A panic of the decode on the helper is resumed on the caller, so a `FillGuard` unwinds and
    /// releases its run exactly as it does for a decode on the calling thread.
    fn decode_run(&self, member: &Member, windows: &[Range<u64>]) -> Result<Vec<Vec<u8>>, Error> {
        if rayon::current_thread_index().is_none() || self.container.shared.engine.is_inline() {
            return decode_windows(member, windows);
        }
        std::thread::scope(|scope| {
            let helper = std::thread::Builder::new()
                .name("ziptensors-cache-fill".to_owned())
                .spawn_scoped(scope, || decode_windows(member, windows))
                .map_err(|source| Error::Io {
                    op: "start read cache fill thread",
                    path: self.container.shared.path.clone().unwrap_or_default(),
                    source,
                })?;
            helper.join().unwrap_or_else(|payload| std::panic::resume_unwind(payload))
        })
    }
}

/// Decodes `windows` of `member` with one [`Member::read_many`], one buffer per window.
fn decode_windows(member: &Member, windows: &[Range<u64>]) -> Result<Vec<Vec<u8>>, Error> {
    let mut buffers = windows
        .iter()
        .map(|window| Ok(vec![0; to_usize(window_len(window))?]))
        .collect::<Result<Vec<Vec<u8>>, Error>>()?;
    let mut requests: Vec<ReadRequest<'_>> = windows
        .iter()
        .zip(buffers.iter_mut())
        .map(|(window, buffer)| ReadRequest { offset: window.start, dst: buffer.as_mut_slice() })
        .collect();
    member.read_many(&mut requests)?;
    drop(requests);
    Ok(buffers)
}

/// The windows one miss reserved as [`Slot::Filling`], with their bytes counted in flight. Dropped
/// unpublished, on an error or a panic, it releases them and wakes the waiters, so none can wait for a
/// fill that will not come.
struct FillGuard<'a> {
    inner: &'a Inner,
    id: u32,
    /// The missed window first, then its read-ahead.
    run: Vec<Range<u64>>,
    published: bool,
}

impl FillGuard<'_> {
    /// Decodes the run with no lock held, publishes it, and returns the bytes of its first window.
    /// If the run fails and holds read-ahead, the read-ahead is released and the first window is
    /// decoded alone, so a damaged neighbour cannot fail an intact window.
    ///
    /// # Errors
    /// The error of decoding the first window alone; the run is then abandoned.
    fn decode_and_publish(mut self, member: &Member) -> Result<Arc<Vec<u8>>, Error> {
        let buffers = match self.inner.decode_run(member, &self.run) {
            Ok(buffers) => buffers,
            Err(error) if self.run.len() > 1 => {
                tracing::debug!(
                    member = self.id,
                    windows = self.run.len(),
                    %error,
                    "read cache run failed, decoding the missed window alone"
                );
                let tail = self.run.split_off(1);
                self.inner.lock().release(self.id, &tail);
                self.inner.filled.notify_all();
                self.inner.decode_run(member, &self.run)?
            }
            Err(error) => return Err(error),
        };

        let inner = self.inner;
        let mut state = inner.lock();
        let run_bytes = self.run.iter().map(window_len).fold(0_u64, u64::saturating_add);
        state.stats.in_flight_bytes = state.stats.in_flight_bytes.saturating_sub(run_bytes);
        // From here on the run is this function's to settle, so `Drop` must not release it again.
        // Nothing below panics: map operations and saturating counters under the lock.
        self.published = true;
        let mut first = None;
        let mut decoded = 0_u64;
        let mut evicted = 0;
        let mut dropped = 0;
        let mut landed: Vec<(bool, &Range<u64>, Vec<u8>)> = self
            .run
            .iter()
            .zip(buffers)
            .enumerate()
            .map(|(at, (window, buffer))| (at == 0, window, buffer))
            .collect();
        // Read-ahead lands first and the missed window last, so that making room for the missed window
        // may drop read-ahead, never the other way round, and the missed window is the most recent.
        landed.rotate_left(1);
        for (first_window, window, buffer) in landed {
            let key = (self.id, window.start);
            // Lossless: `usize` is at most 64 bits on every supported target.
            let len = buffer.len() as u64;
            decoded = decoded.saturating_add(len);
            let bytes = Arc::new(buffer);
            if first_window {
                // The caller is served from this handle.
                first = Some(Arc::clone(&bytes));
            }
            let (count, fits) = state.make_room(len, inner.max_bytes);
            evicted += count;
            // The missed window stays even when fills in flight hold the budget, so a reader whose reads
            // are smaller than a window does not decode it again on its next read. `make_room` then
            // emptied the LRU, so it is the only resident window, and `new` proved it fits `max_bytes`.
            if fits || first_window {
                let tick = state.tick();
                state.lru.insert(tick, key);
                state.slots.insert(key, Slot::Ready { bytes, tick });
                state.stats.resident_bytes = state.stats.resident_bytes.saturating_add(len);
            } else {
                // The budget is held by fills in flight: serve this read-ahead window, keep nothing.
                state.slots.remove(&key);
                dropped += 1;
            }
        }
        state.stats.decoded_bytes = state.stats.decoded_bytes.saturating_add(decoded);
        state.note_peak();
        let resident = state.stats.resident_bytes;
        drop(state);
        inner.filled.notify_all();
        tracing::debug!(
            member = self.id,
            start = self.run.first().map(|window| window.start),
            windows = self.run.len(),
            decoded,
            evicted,
            dropped,
            resident,
            "read cache filled a run"
        );
        first.ok_or_else(|| broken("a read cache run has no window"))
    }
}

impl Drop for FillGuard<'_> {
    fn drop(&mut self) {
        if self.published {
            return;
        }
        self.inner.lock().release(self.id, &self.run);
        self.inner.filled.notify_all();
        tracing::debug!(member = self.id, windows = self.run.len(), "read cache abandoned a run");
    }
}

/// A bounded cache of decoded member bytes over one [`Container`], shared by all members of it.
/// Opt-in: [`Member`] itself caches nothing.
///
/// Reads through [`CachedMember::read_at`] return exactly what [`Member::read_at`] returns, and fail
/// only where it fails: a damaged read-ahead window never fails the window that was asked for. Chunk
/// hashes are checked as the container's [`crate::Verify`] says on every fill, and never on a hit.
/// Concurrent requests for one window decode it once. Errors are not cached.
///
/// Memory: windows resident plus fills in flight hold at most `max_bytes`, plus one window (at most
/// 2 × the largest chunk) per concurrent miss, since a missed window always proceeds while read-ahead
/// is admitted only within the budget; on top come the engine's own bound of 3 × threads × chunk per
/// fill, and a window evicted while a reader copies from it, which lives until that copy ends. A missed
/// window is kept even when fills in flight hold the budget; it is then the only resident window, so
/// `resident_bytes` stays within `max_bytes`. Read-ahead that finds no room is served but not kept.
/// [`CacheStats::peak_bytes`] records the most held at once.
///
/// Cloning is cheap and shares the cache. `Send + Sync`; all methods take `&self`, also from tasks of
/// any rayon pool: a miss filled from a rayon worker decodes on a helper thread, one spawn per miss.
#[derive(Debug, Clone)]
pub struct ReadCache {
    inner: Arc<Inner>,
}

impl ReadCache {
    /// A cache over `container`.
    ///
    /// # Errors
    /// Returns [`Error::BudgetExceeded`] with `what` = `"read cache window"` if `options.max_bytes` is
    /// below 2 × the largest `chunk_size` of the container, the most bytes one window can hold.
    pub fn new(container: Container, options: CacheOptions) -> Result<Self, Error> {
        let largest_chunk = container
            .shared
            .index
            .blobs()
            .iter()
            .map(|blob| u64::from(blob.chunk_size))
            .max()
            .unwrap_or(0);
        // A widening view doubles a chunk, and no view widens more.
        let largest_window = largest_chunk.saturating_mul(2);
        if options.max_bytes < largest_window {
            return Err(Error::BudgetExceeded {
                what: "read cache window",
                declared: largest_window,
                limit: options.max_bytes,
            });
        }
        let run_windows = match options.readahead_windows {
            0 => container.shared.engine.threads.max(1),
            windows => windows,
        };
        tracing::debug!(max_bytes = options.max_bytes, run_windows, largest_window, "read cache created");
        let inner = Inner {
            container,
            max_bytes: options.max_bytes,
            run_windows,
            state: Mutex::new(State::default()),
            filled: Condvar::new(),
        };
        Ok(Self { inner: Arc::new(inner) })
    }

    /// The container this cache reads.
    #[must_use]
    pub fn container(&self) -> &Container {
        &self.inner.container
    }

    /// A cached handle on the member called `name`.
    ///
    /// # Errors
    /// Returns [`Error::MemberNotFound`] if there is none.
    pub fn member(&self, name: &str) -> Result<CachedMember, Error> {
        self.member_by_id(self.inner.container.member_id(name)?)
    }

    /// A cached handle on member `index`, in the file order of [`Container::members`].
    ///
    /// # Errors
    /// Returns [`Error::MemberNotFound`] if `index` is not below the member count.
    pub fn member_at(&self, index: usize) -> Result<CachedMember, Error> {
        let not_found = || Error::MemberNotFound { name: format!("#{index}") };
        if index >= self.inner.container.members().len() {
            return Err(not_found());
        }
        self.member_by_id(u32::try_from(index).map_err(|_| not_found())?)
    }

    /// The handle on member `id`.
    fn member_by_id(&self, id: u32) -> Result<CachedMember, Error> {
        let member = Member::new(self.inner.container.clone(), id)?;
        Ok(CachedMember { cache: self.clone(), member, id })
    }

    /// Counters since creation and the resident size now.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        self.inner.lock().stats
    }
}

/// A member read through a [`ReadCache`]. Holds the cache and the container alive. Cloning is cheap;
/// `Send + Sync`, and all methods take `&self`.
#[derive(Debug, Clone)]
pub struct CachedMember {
    cache: ReadCache,
    member: Member,
    /// File-order id, the member part of a window key.
    id: u32,
}

impl CachedMember {
    /// The uncached handle on the same member, for its name, hash and tensor table.
    #[must_use]
    pub fn member(&self) -> &Member {
        &self.member
    }

    /// Length of the virtual file in bytes.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.member.len()
    }

    /// Whether the virtual file has no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.member.is_empty()
    }

    /// Fills all of `dst` from byte `offset` of the member, through the cache: the same bytes as
    /// [`Member::read_at`], and an error only where it fails too. A range over several windows is served
    /// window by window. If a window cannot be decoded, the piece of the range in it is read with
    /// [`Member::read_at`] and its result stands, so an element straddling into a damaged neighbour
    /// fails only a read that touches that element.
    ///
    /// # Errors
    /// As [`Member::read_at`]: [`Error::OutOfBounds`] before any I/O, then decode and I/O errors.
    pub fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), Error> {
        self.member.check_range(offset, dst.len())?;
        let mut position = offset;
        let mut out = dst;
        while !out.is_empty() {
            let window = self.member.chunk_window(position)?;
            let from = position.checked_sub(window.start).ok_or_else(|| broken("a window starts late"))?;
            let from = to_usize(from)?;
            let in_window = to_usize(window.end.saturating_sub(position))?;
            let take = out.len().min(in_window);
            if take == 0 {
                return Err(broken("a window serves no bytes"));
            }
            let (now, later) = std::mem::take(&mut out).split_at_mut(take);
            match self.cache.inner.window(&self.member, self.id, &window) {
                Ok(bytes) => {
                    let source = bytes.get(from..from + take);
                    now.copy_from_slice(source.ok_or_else(|| broken("a window is shorter than planned"))?);
                }
                Err(error) => {
                    tracing::debug!(member = self.id, position, %error, "cached window failed, reading directly");
                    self.member.read_at(position, now)?;
                }
            }
            out = later;
            // Lossless: `usize` is at most 64 bits on every supported target.
            position += take as u64;
        }
        Ok(())
    }
}
