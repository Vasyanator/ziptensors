//! File: counting.rs
//! Purpose: the counting global allocator of the driver, so that memory gates compare Rust bytes with
//! the bounds of `ARCHITECTURE.md`, which are stated in Rust allocations, instead of process RSS.
//!
//! Main responsibilities:
//! - [`Counting`]: forwards every call to [`System`] and keeps the live bytes and their peak;
//! - [`Phase`]: the peak of a measured phase above a baseline taken at its start.
//!
//! Key structures: [`Counting`], [`Phase`].
//!
//! Implementation notes: allocations inside zstd's C code bypass this allocator, which is what the
//! documented bounds exclude too. A `realloc` is counted as a move, both blocks live at the moment of
//! the copy, as in the allocator tests of `ziptensors-core` (`tests/read_memory.rs`,
//! `tests/write_memory.rs`). Counters are relaxed atomics: a peak may be read a few bytes stale across
//! threads, which no gate can notice.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A global allocator that counts the live bytes of the process and their peak.
pub struct Counting {
    live: AtomicUsize,
    peak: AtomicUsize,
}

impl Counting {
    /// A counter at zero.
    pub const fn new() -> Self {
        Self { live: AtomicUsize::new(0), peak: AtomicUsize::new(0) }
    }

    /// Live bytes now.
    pub fn live(&self) -> usize {
        self.live.load(Ordering::Relaxed)
    }

    /// Starts a phase: the peak restarts at the live bytes of now, which become its baseline.
    pub fn phase(&self) -> Phase<'_> {
        let base = self.live();
        self.peak.store(base, Ordering::Relaxed);
        Phase { counter: self, base }
    }

    fn grew(&self, by: usize) {
        let live = self.live.fetch_add(by, Ordering::Relaxed).saturating_add(by);
        self.peak.fetch_max(live, Ordering::Relaxed);
    }

    fn shrank(&self, by: usize) {
        self.live.fetch_sub(by, Ordering::Relaxed);
    }
}

// SAFETY: every call forwards to `System` with its arguments unchanged, and the counters neither
// allocate nor touch the memory.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract is forwarded.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            self.grew(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract is forwarded.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            self.grew(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract is forwarded.
        unsafe { System.dealloc(pointer, layout) };
        self.shrank(layout.size());
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Counted as a move, the worst an allocator may do. A bound proven under this also holds for
        // a realloc that works in place.
        self.grew(new_size);
        // SAFETY: the caller's contract is forwarded.
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if pointer.is_null() {
            self.shrank(new_size);
        } else {
            self.shrank(layout.size());
        }
        pointer
    }
}

/// A measured phase. Only one phase is meaningful at a time: starting another restarts the peak.
pub struct Phase<'a> {
    counter: &'a Counting,
    base: usize,
}

impl Phase<'_> {
    /// Live bytes at the start of the phase.
    pub fn base(&self) -> usize {
        self.base
    }

    /// The highest live bytes since the phase started, above its baseline. Bytes freed below the
    /// baseline during the phase do not make this negative: it saturates at 0.
    pub fn peak_above(&self) -> usize {
        self.counter.peak.load(Ordering::Relaxed).saturating_sub(self.base)
    }
}

#[cfg(test)]
mod tests {
    use super::Counting;
    use std::alloc::{GlobalAlloc, Layout};

    /// A counter of its own, so that the global allocator of the test binary, shared by tests running
    /// in parallel, does not disturb the figures.
    #[test]
    fn a_phase_sees_the_peak_above_its_baseline_and_realloc_counts_both_blocks() {
        let counter = Counting::new();
        let small = Layout::from_size_align(100, 8).expect("a valid layout");
        // SAFETY: the layout is non-zero; every block is freed with the layout it was allocated with.
        unsafe {
            let first = counter.alloc(small);
            assert!(!first.is_null());
            assert_eq!(counter.live(), 100);
            let phase = counter.phase();
            assert_eq!(phase.base(), 100);
            assert_eq!(phase.peak_above(), 0);

            let second = counter.alloc_zeroed(small);
            assert!(!second.is_null());
            assert_eq!(phase.peak_above(), 100);
            // The move counts old and new together: 100 + 100 + 300 live at the copy.
            let grown = counter.realloc(second, small, 300);
            assert!(!grown.is_null());
            assert_eq!(phase.peak_above(), 400);
            assert_eq!(counter.live(), 400);

            counter.dealloc(grown, Layout::from_size_align(300, 8).expect("a valid layout"));
            counter.dealloc(first, small);
            assert_eq!(counter.live(), 0);
            assert_eq!(phase.peak_above(), 400, "freeing keeps the peak");

            let later = counter.phase();
            assert_eq!(later.base(), 0);
            assert_eq!(later.peak_above(), 0, "a new phase restarts the peak");
        }
    }
}
