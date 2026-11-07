//! File: read/scratch.rs
//! Purpose: the bounded pool of decode scratch, one entry per decode thread of a `Container`.
//!
//! Main responsibilities: hand out [`Scratch`] for the length of one chunk job, and block while all
//! entries are in use, so that concurrent callers share one fixed amount of memory.
//!
//! Key structures: [`Scratch`], [`ScratchPool`], [`Lease`].
//!
//! Implementation notes: a pool and not thread-locals, so the memory bound is deterministic and goes
//! away with the `Container`. An entry leaves the mutex while it is used, because the rule is that no
//! lock is held across a decode. A waiter holds nothing, and a holder waits on nothing, so the pool
//! cannot deadlock, even when rayon workers of the pool wait here.

use std::sync::{Condvar, Mutex, PoisonError};

use crate::codec::Workspace;

/// The memory of one decode: `workspace` for the codec, `chunk` for canonical bytes that cannot go
/// straight to the destination, `base` for the base chunk of a delta chunk. Each holds at most one
/// chunk: the first two are the view factor 2 of spec section 9.2, and `base` is the third chunk of
/// its delta bound. `base` stays empty until a delta chunk is decoded, so a container whose reads touch
/// no delta blob keeps the bound without it.
#[derive(Debug)]
pub(super) struct Scratch {
    /// Codec state, including the codec's own scratch.
    pub(super) workspace: Workspace,
    /// Canonical bytes of a chunk that is read in part or widened.
    pub(super) chunk: Vec<u8>,
    /// Canonical bytes of the base chunk a delta chunk is combined with.
    pub(super) base: Vec<u8>,
}

/// A fixed set of [`Scratch`] entries.
#[derive(Debug)]
pub(super) struct ScratchPool {
    free: Mutex<Vec<Scratch>>,
    returned: Condvar,
}

impl ScratchPool {
    /// Creates `count` entries. They allocate nothing until used.
    pub(super) fn new(count: usize) -> Self {
        let free = (0..count)
            .map(|_| Scratch { workspace: Workspace::new(), chunk: Vec::new(), base: Vec::new() })
            .collect();
        Self { free: Mutex::new(free), returned: Condvar::new() }
    }

    /// Takes an entry, waiting for one if all are in use. `count` must not be 0.
    pub(super) fn lease(&self) -> Lease<'_> {
        // A panic while the lock is held cannot leave the vector inconsistent: the critical sections
        // are one `pop` and one `push`. So a poisoned lock is safe to use.
        let mut free = self.free.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            if let Some(scratch) = free.pop() {
                return Lease { pool: self, scratch: Some(scratch) };
            }
            free = self.returned.wait(free).unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Number of entries not in use.
    #[cfg(test)]
    pub(super) fn available(&self) -> usize {
        self.free.lock().unwrap_or_else(PoisonError::into_inner).len()
    }
}

/// Exclusive use of one entry, returned to the pool on drop, also when a decode panics.
#[derive(Debug)]
pub(super) struct Lease<'p> {
    pool: &'p ScratchPool,
    /// `Some` for the whole life of the value; only `drop` takes it.
    scratch: Option<Scratch>,
}

impl std::ops::Deref for Lease<'_> {
    type Target = Scratch;

    fn deref(&self) -> &Scratch {
        // Invariant: `lease` creates this with `Some`, and only `drop` takes it.
        self.scratch.as_ref().expect("a lease holds its entry until dropped")
    }
}

impl std::ops::DerefMut for Lease<'_> {
    fn deref_mut(&mut self) -> &mut Scratch {
        // Invariant: `lease` creates this with `Some`, and only `drop` takes it.
        self.scratch.as_mut().expect("a lease holds its entry until dropped")
    }
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        if let Some(scratch) = self.scratch.take() {
            self.pool.free.lock().unwrap_or_else(PoisonError::into_inner).push(scratch);
            self.pool.returned.notify_one();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ScratchPool;
    use std::sync::Arc;

    #[test]
    fn a_lease_returns_on_drop() {
        let pool = ScratchPool::new(2);
        let first = pool.lease();
        assert_eq!(pool.available(), 1);
        drop(first);
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn a_waiter_gets_the_entry_when_it_returns() {
        let pool = Arc::new(ScratchPool::new(1));
        let held = pool.lease();
        let other = Arc::clone(&pool);
        let waiter = std::thread::spawn(move || drop(other.lease()));
        drop(held);
        waiter.join().expect("the waiter finishes");
        assert_eq!(pool.available(), 1);
    }
}
