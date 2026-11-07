//! File: `read/cache_tests.rs`
//! Purpose: tests of the opt-in read cache, on real containers built by `testkit`, against
//! `Member::read_at` and the reference bytes.
//!
//! Implementation notes: all data is generated from seeds and lives in memory; nothing touches the
//! file system. The fixture mixes identity and widening extents, an odd `chunk_size` under a widening
//! view, and two members over one blob, so windows of both members start at the same offsets.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Duration;

use super::testkit::{
    ContainerBuilder, ExtentSpec, MemorySource, Rng, delta, finetune_of, identity, planes, tensor_like,
    whole,
};
use super::{CacheOptions, CacheStats, CachedMember, Container, Member, OpenOptions, ReadCache};
use crate::chain::{ElementWidth, View};
use crate::codec::StreamPolicy;
use crate::error::Error;
use crate::io::RandomAccess;
use crate::types::Verify;

const KIB: usize = 1 << 10;
const KIB_U32: u32 = 1 << 10;
/// The largest `chunk_size` of [`fixture`].
const LARGEST_CHUNK: u64 = 16 << 10;
/// The smallest legal cache of [`fixture`]: one widened window of the largest chunk.
const SMALLEST_CACHE: u64 = 2 * LARGEST_CHUNK;

/// A small value as a length.
fn len_of(value: u64) -> usize {
    usize::try_from(value).expect("a test value fits a length")
}

/// A container with member `mixed` (four extents: identity under an odd chunk size, F16 widened under
/// the odd chunk size 1001 so elements straddle chunks, planar identity, identity again) and member
/// `second`, which starts at the same blob as `mixed`. Returns the file and both contents.
fn fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut builder = ContainerBuilder::new();
    let planar = tensor_like(1, 100 * KIB, 4);
    let plain = tensor_like(2, 40 * KIB + 123, 1);
    let half = tensor_like(3, 20 * KIB, 2);
    let a = builder.add_blob(
        planar,
        planes(ElementWidth::Four),
        16 * KIB_U32,
        StreamPolicy::EntropyOrStore,
    );
    let b = builder.add_blob(plain, whole(), 7001, StreamPolicy::Zstd { level: 3 });
    let h = builder.add_blob(half, whole(), 1001, StreamPolicy::EntropyOrStore);
    let widened = ExtentSpec { blob: h, blob_offset: 2, virt_len: 40_940, view: View::F16ToF32 };
    let mixed = builder.add_member(
        "mixed",
        &[identity(b, 5, 30_000), widened, identity(a, 4096, 60_000), identity(b, 30_005, 7000)],
    );
    let second = builder.add_member("second", &[identity(b, 0, 30_000)]);
    (builder.finish(), mixed, second)
}

fn open(file: Vec<u8>, threads: usize, verify: Verify) -> Container {
    let options = OpenOptions { threads, verify, ..OpenOptions::default() };
    Container::from_source(Arc::new(MemorySource::silent(file)), &options).expect("a valid container")
}

fn cache(container: &Container, max_bytes: u64, readahead_windows: usize) -> ReadCache {
    ReadCache::new(container.clone(), CacheOptions { max_bytes, readahead_windows })
        .expect("legal options")
}

/// Reads `len` bytes at `offset` through both handles and asserts they agree with each other, with
/// the reference, and stay inside their destination.
fn assert_same(cached: &CachedMember, member: &Member, content: &[u8], offset: usize, len: usize) {
    let mut through_cache = vec![0xA5; len + 2];
    cached.read_at(offset as u64, &mut through_cache[1..=len]).expect("cached read in range");
    let mut direct = vec![0; len];
    member.read_at(offset as u64, &mut direct).expect("direct read in range");
    assert_eq!(&through_cache[1..=len], direct.as_slice(), "offset {offset}, len {len}");
    assert_eq!(direct.as_slice(), &content[offset..offset + len], "offset {offset}, len {len}");
    assert_eq!((through_cache[0], through_cache[len + 1]), (0xA5, 0xA5), "a read left its destination");
}

/// The windows of `member`, walked from 0.
fn windows(member: &Member) -> Vec<std::ops::Range<u64>> {
    let mut windows = Vec::new();
    let mut at = 0;
    while at < member.len() {
        let window = member.chunk_window(at).expect("a window in the member");
        at = window.end;
        windows.push(window);
    }
    windows
}

#[test]
fn handles_are_clone_send_sync() {
    fn assert_traits<T: Clone + Send + Sync>() {}
    assert_traits::<ReadCache>();
    assert_traits::<CachedMember>();
}

#[test]
fn windows_partition_the_member_within_one_chunk_each() {
    let (file, mixed, _) = fixture();
    let member = open(file, 1, Verify::Chunk).member("mixed").expect("present");
    let windows = windows(&member);
    let extent_ends = [30_000, 70_940, 130_940, 137_940];
    assert_eq!(windows.first().map(|window| window.start), Some(0));
    assert_eq!(windows.last().map(|window| window.end), Some(mixed.len() as u64));
    for pair in windows.windows(2) {
        assert_eq!(pair[0].end, pair[1].start, "windows are contiguous");
    }
    for window in &windows {
        assert!(window.end > window.start, "{window:?} is empty");
        assert!(window.end - window.start <= SMALLEST_CACHE, "{window:?} is over one widened chunk");
        let crosses = extent_ends.iter().any(|end| window.start < *end && window.end > *end);
        assert!(!crosses, "{window:?} crosses an extent");
        for at in [window.start, window.end - 1] {
            assert_eq!(member.chunk_window(at).expect("in the member"), *window);
        }
    }
    // The odd chunk size: windows of `b` are 7001 bytes after the first, which starts 5 bytes in.
    assert_eq!(windows[0], 0..6996);
    assert_eq!(windows[1], 6996..13_997);
    assert!(matches!(member.chunk_window(mixed.len() as u64), Err(Error::OutOfBounds { .. })));
}

#[test]
fn cached_reads_match_read_at_at_random_offsets() {
    let (file, mixed, second) = fixture();
    for threads in [1, 3] {
        let container = open(file.clone(), threads, Verify::Chunk);
        for (max_bytes, readahead) in [(256 << 20, 0), (SMALLEST_CACHE, 1), (3 * SMALLEST_CACHE, 5)] {
            let cache = cache(&container, max_bytes, readahead);
            for (name, content) in [("mixed", &mixed), ("second", &second)] {
                let cached = cache.member(name).expect("present");
                let member = container.member(name).expect("present");
                assert_eq!(cached.len(), content.len() as u64);
                let mut rng = Rng(threads as u64 * 31 + max_bytes);
                for _ in 0..400 {
                    let offset = len_of(rng.below(content.len() as u64));
                    let room = (content.len() - offset) as u64;
                    let len = len_of(rng.below(room.min(20_000) + 1));
                    assert_same(&cached, &member, content, offset, len);
                }
            }
            assert!(cache.stats().resident_bytes <= max_bytes, "{:?}", cache.stats());
        }
    }
}

#[test]
fn cached_reads_of_a_delta_member_match_read_at() {
    let mut builder = ContainerBuilder::new();
    let base_raw = tensor_like(40, 64 * KIB, 2);
    let chain = planes(ElementWidth::Two);
    let base = builder.add_blob(base_raw.clone(), chain, 16 * KIB_U32, StreamPolicy::EntropyOrStore);
    let target = finetune_of(41, &base_raw[16 * KIB..]);
    let policy = StreamPolicy::EntropyOrStore;
    let blob = builder.add_delta_blob(target, base, 16 * 1024, delta(ElementWidth::Two, chain), policy);
    let widened = ExtentSpec { blob, blob_offset: 4, virt_len: 40 * 1024, view: View::Bf16ToF32 };
    let content = builder.add_member("tuned", &[identity(blob, 0, 48 * 1024), widened]);
    let file = builder.finish();
    for threads in [1, 3] {
        let container = open(file.clone(), threads, Verify::Chunk);
        let cache = cache(&container, SMALLEST_CACHE, 2);
        let cached = cache.member("tuned").expect("present");
        let member = container.member("tuned").expect("present");
        let mut rng = Rng(42 + threads as u64);
        for _ in 0..200 {
            let offset = len_of(rng.below(content.len() as u64));
            let room = (content.len() - offset) as u64;
            let len = len_of(rng.below(room.min(20_000) + 1));
            assert_same(&cached, &member, &content, offset, len);
        }
        assert!(cache.stats().resident_bytes <= SMALLEST_CACHE, "{:?}", cache.stats());
    }
}

#[test]
fn reads_crossing_window_and_extent_boundaries_match() {
    let (file, mixed, _) = fixture();
    let container = open(file, 2, Verify::Chunk);
    let member = container.member("mixed").expect("present");
    let cache = cache(&container, SMALLEST_CACHE * 2, 2);
    let cached = cache.member("mixed").expect("present");
    for window in windows(&member) {
        let edge = len_of(window.end);
        for start in [edge.saturating_sub(5), edge.saturating_sub(1), edge] {
            for len in [1, 2, 3, 4, 5, 9, 7002, 20_000] {
                let len = len.min(mixed.len() - start.min(mixed.len()));
                if start < mixed.len() {
                    assert_same(&cached, &member, &mixed, start, len);
                }
            }
        }
    }
    // One read over the whole member, through every window and extent.
    assert_same(&cached, &member, &mixed, 0, mixed.len());
}

#[test]
fn reads_at_the_end_and_out_of_bounds_fail_as_read_at_does() {
    let (file, mixed, _) = fixture();
    let container = open(file, 1, Verify::Chunk);
    let member = container.member("mixed").expect("present");
    let cache = cache(&container, SMALLEST_CACHE, 0);
    let cached = cache.member("mixed").expect("present");
    let len = mixed.len() as u64;
    cached.read_at(len, &mut []).expect("an empty read at the end");
    assert_same(&cached, &member, &mixed, mixed.len() - 1, 1);
    let before = cache.stats();
    for (offset, size) in [(len, 1), (len - 1, 2), (len + 7, 0), (u64::MAX, 1), (0, mixed.len() + 1)] {
        let mut a = vec![0; size];
        let mut b = vec![0; size];
        let through_cache = cached.read_at(offset, &mut a).expect_err("out of bounds");
        let direct = member.read_at(offset, &mut b).expect_err("out of bounds");
        assert!(matches!(through_cache, Error::OutOfBounds { .. }), "{through_cache:?}");
        assert_eq!(format!("{through_cache:?}"), format!("{direct:?}"));
    }
    assert_eq!(cache.stats(), before, "a bad range touches nothing");
    assert_ne!(before, CacheStats::default());
}

#[test]
fn eviction_keeps_resident_bytes_within_the_budget() {
    let (file, mixed, second) = fixture();
    let container = open(file, 2, Verify::Chunk);
    let cache = cache(&container, SMALLEST_CACHE, 0);
    let cached = cache.member("mixed").expect("present");
    let other = cache.member("second").expect("present");
    let mut buffer = vec![0; 4096];
    for round in 0..2 {
        for (at, chunk) in mixed.chunks(4096).enumerate() {
            let dst = &mut buffer[..chunk.len()];
            cached.read_at((at * 4096) as u64, dst).expect("in range");
            assert_eq!(dst, chunk, "round {round}, piece {at}");
            assert!(cache.stats().resident_bytes <= SMALLEST_CACHE, "{:?}", cache.stats());
        }
        let dst = &mut buffer[..1000];
        other.read_at(3, dst).expect("in range");
        assert_eq!(dst, &second[3..1003]);
    }
    let stats = cache.stats();
    assert!(stats.evictions > 0, "{stats:?}");
    // Only the last windows of the first pass can survive into the second.
    let floor = 2 * mixed.len() as u64 - SMALLEST_CACHE;
    assert!(stats.decoded_bytes >= floor, "a second pass decodes again: {stats:?}");
}

#[test]
fn a_sequential_reader_decodes_every_window_once() {
    let (file, mixed, _) = fixture();
    for readahead in [1, 3, 0] {
        let container = open(file.clone(), 3, Verify::Chunk);
        let cache = cache(&container, 256 << 20, readahead);
        let cached = cache.member("mixed").expect("present");
        let mut read = vec![0; mixed.len()];
        for (at, piece) in read.chunks_mut(1000).enumerate() {
            cached.read_at((at * 1000) as u64, piece).expect("in range");
        }
        assert_eq!(read, mixed);
        let stats = cache.stats();
        assert_eq!(stats.decoded_bytes, mixed.len() as u64, "readahead {readahead}: {stats:?}");
        assert_eq!(stats.resident_bytes, mixed.len() as u64);
        assert_eq!(stats.evictions, 0);
        let window_count = windows(&container.member("mixed").expect("present")).len() as u64;
        let per_miss = if readahead == 0 { 3 } else { readahead as u64 };
        assert_eq!(stats.misses, window_count.div_ceil(per_miss), "readahead {readahead}: {stats:?}");
    }
}

#[test]
fn concurrent_readers_of_one_window_decode_it_once() {
    let (file, mixed, _) = fixture();
    let container = open(file, 2, Verify::Chunk);
    let first_window = container.member("mixed").expect("present").chunk_window(0).expect("window");
    for _ in 0..20 {
        let cache = cache(&container, 256 << 20, 1);
        let barrier = Barrier::new(8);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let cached = cache.member("mixed").expect("present");
                let (barrier, mixed) = (&barrier, &mixed);
                scope.spawn(move || {
                    let mut dst = vec![0; 100];
                    barrier.wait();
                    cached.read_at(10, &mut dst).expect("in range");
                    assert_eq!(dst, mixed[10..110]);
                });
            }
        });
        let stats = cache.stats();
        assert_eq!(stats.misses, 1, "{stats:?}");
        assert_eq!(stats.hits, 7, "{stats:?}");
        assert_eq!(stats.decoded_bytes, first_window.end - first_window.start, "{stats:?}");
    }
}

#[test]
fn concurrent_random_readers_with_a_small_cache_read_correctly() {
    let (file, mixed, second) = fixture();
    let container = open(file, 3, Verify::Chunk);
    let cache = cache(&container, 2 * SMALLEST_CACHE, 0);
    std::thread::scope(|scope| {
        for seed in 0..8_u64 {
            let (name, content) = if seed % 2 == 0 { ("mixed", &mixed) } else { ("second", &second) };
            let cached = cache.member(name).expect("present");
            scope.spawn(move || {
                let mut rng = Rng(seed);
                for _ in 0..200 {
                    let offset = len_of(rng.below(content.len() as u64));
                    let room = (content.len() - offset) as u64;
                    let len = len_of(rng.below(room.min(30_000) + 1));
                    let mut dst = vec![0; len];
                    cached.read_at(offset as u64, &mut dst).expect("in range");
                    assert_eq!(dst, content[offset..offset + len], "offset {offset}, len {len}");
                }
            });
        }
    });
    assert!(cache.stats().resident_bytes <= 2 * SMALLEST_CACHE, "{:?}", cache.stats());
}

/// Four stored chunks of 256 bytes, with one byte of chunk 2 flipped, and the intact content.
fn with_flipped_byte() -> (Vec<u8>, Vec<u8>) {
    let mut builder = ContainerBuilder::new();
    let raw = tensor_like(6, 4 * 256, 1);
    let blob = builder.add_blob(raw, whole(), 256, StreamPolicy::Store);
    let content = builder.add_member("m", &[identity(blob, 0, 4 * 256)]);
    let mut file = builder.finish();
    // Chunk 2 starts after the header and two chunks of 4 + 256 bytes. Flip a stored byte of it.
    file[64 + 2 * 260 + 50] ^= 0x10;
    (file, content)
}

#[test]
fn chunk_hashes_are_checked_on_fill_and_errors_are_not_cached() {
    let (file, content) = with_flipped_byte();
    let checked = open(file.clone(), 1, Verify::Chunk);
    let cache = cache(&checked, 4096, 1);
    let cached = cache.member("m").expect("present");
    let mut dst = vec![0; 10];
    for attempt in 1..=2 {
        let error = cached.read_at(2 * 256 + 3, &mut dst);
        assert!(matches!(error, Err(Error::ChecksumMismatch { blob: 0, chunk: 2, .. })), "{error:?}");
        assert_eq!(cache.stats().misses, attempt, "a failed fill is tried again");
        assert_eq!(cache.stats().resident_bytes, 0, "a failed fill caches nothing");
    }
    cached.read_at(3, &mut dst).expect("an intact chunk reads");
    assert_eq!(dst, content[3..13]);

    let unchecked = open(file, 1, Verify::Off);
    let cached = self::cache(&unchecked, 4096, 0).member("m").expect("present");
    let mut whole_member = vec![0; content.len()];
    cached.read_at(0, &mut whole_member).expect("Verify::Off returns what decodes");
    assert_eq!(whole_member[..512], content[..512]);
    assert_ne!(whole_member[512..768], content[512..768]);
}

#[test]
fn members_are_found_by_name_and_by_file_order() {
    let (file, mixed, second) = fixture();
    let container = open(file, 1, Verify::Chunk);
    let cache = cache(&container, SMALLEST_CACHE, 0);
    assert_eq!(cache.member_at(0).expect("first").len(), mixed.len() as u64);
    assert_eq!(cache.member_at(1).expect("second").member().name().expect("a name"), "second");
    assert_eq!(cache.member("second").expect("present").len(), second.len() as u64);
    assert!(!cache.member("mixed").expect("present").is_empty());
    assert!(matches!(cache.member_at(2), Err(Error::MemberNotFound { .. })));
    assert!(matches!(cache.member("missing"), Err(Error::MemberNotFound { .. })));
    assert_eq!(cache.container().members().len(), 2);
}

#[test]
fn a_cache_smaller_than_one_widened_chunk_is_refused() {
    let (file, _, _) = fixture();
    let container = open(file, 1, Verify::Chunk);
    let options = CacheOptions { max_bytes: SMALLEST_CACHE - 1, readahead_windows: 0 };
    let error = ReadCache::new(container.clone(), options).expect_err("too small");
    assert!(
        matches!(
            error,
            Error::BudgetExceeded { what: "read cache window", declared: SMALLEST_CACHE, limit }
                if limit == SMALLEST_CACHE - 1
        ),
        "{error:?}"
    );
    let options = CacheOptions { max_bytes: SMALLEST_CACHE, ..CacheOptions::default() };
    ReadCache::new(container, options).expect("one widened chunk is enough");
    assert_eq!(CacheOptions::default(), CacheOptions { max_bytes: 256 << 20, readahead_windows: 0 });
}

#[test]
fn a_damaged_read_ahead_window_never_fails_an_intact_one() {
    let (file, content) = with_flipped_byte();
    let container = open(file, 3, Verify::Chunk);
    let member = container.member("m").expect("present");
    for readahead in [0, 2, 3] {
        let cache = cache(&container, 4096, readahead);
        let cached = cache.member("m").expect("present");
        for offset in [256 + 3, 3, 3 * 256 + 3] {
            assert_same(&cached, &member, &content, offset, 10);
        }
        // The intact windows were kept: reading them again decodes nothing.
        let before = cache.stats();
        let mut dst = vec![0; 10];
        cached.read_at(256 + 3, &mut dst).expect("intact");
        assert_eq!(cache.stats().misses, before.misses, "readahead {readahead}: {:?}", cache.stats());
        let error = cached.read_at(2 * 256 + 3, &mut dst);
        assert!(matches!(error, Err(Error::ChecksumMismatch { blob: 0, chunk: 2, .. })), "{error:?}");
        // A read over the damaged chunk fails as the direct read does, and the end stays readable.
        let mut all = vec![0; content.len()];
        assert!(matches!(cached.read_at(0, &mut all), Err(Error::ChecksumMismatch { chunk: 2, .. })));
        assert_same(&cached, &member, &content, 3 * 256, 256);
        assert_eq!(cache.stats().in_flight_bytes, 0);
    }
}

/// A member of 64 stored chunks of 4 KiB, one window each, and its content.
fn many_windows() -> (Vec<u8>, Vec<u8>) {
    let mut builder = ContainerBuilder::new();
    let blob = builder.add_blob(tensor_like(4, 64 * 4096, 1), whole(), 4096, StreamPolicy::Zstd { level: 1 });
    let content = builder.add_member("m", &[identity(blob, 0, 64 * 4096)]);
    (builder.finish(), content)
}

#[test]
fn concurrent_misses_hold_at_most_the_budget_plus_one_window_each() {
    const THREADS: u64 = 16;
    let (file, content) = many_windows();
    let container = open(file, 4, Verify::Chunk);
    for max_bytes in [2 * 4096, 8 * 4096, 32 * 4096] {
        let cache = cache(&container, max_bytes, 8);
        let barrier = Barrier::new(16);
        std::thread::scope(|scope| {
            for thread in 0..16_usize {
                let cached = cache.member("m").expect("present");
                let (barrier, content) = (&barrier, &content);
                scope.spawn(move || {
                    let mut dst = vec![0; 4096];
                    barrier.wait();
                    for round in 0..8 {
                        let window = (thread * 4 + round * 16) % 64;
                        cached.read_at((window * 4096) as u64, &mut dst).expect("in range");
                        assert_eq!(dst, content[window * 4096..(window + 1) * 4096]);
                    }
                });
            }
        });
        let stats = cache.stats();
        assert!(stats.peak_bytes <= max_bytes + THREADS * 4096, "{stats:?}");
        assert!(stats.resident_bytes <= max_bytes, "{stats:?}");
        assert_eq!(stats.in_flight_bytes, 0, "{stats:?}");
    }
}

#[test]
fn one_reader_never_holds_more_than_the_budget() {
    let (file, content) = many_windows();
    let container = open(file, 4, Verify::Chunk);
    for (max_bytes, readahead) in [(2 * 4096, 8), (5 * 4096, 0), (5 * 4096, 3)] {
        let cache = cache(&container, max_bytes, readahead);
        let cached = cache.member("m").expect("present");
        let mut read = vec![0; content.len()];
        for (at, piece) in read.chunks_mut(1500).enumerate() {
            cached.read_at((at * 1500) as u64, piece).expect("in range");
        }
        assert_eq!(read, content);
        let stats = cache.stats();
        assert!(stats.peak_bytes <= max_bytes, "{stats:?}");
        assert_eq!(stats.decoded_bytes, content.len() as u64, "every window once: {stats:?}");
    }
}

/// A source that panics on its first read once armed, after meeting the test at a barrier.
struct PanicOnce {
    inner: MemorySource,
    armed: AtomicBool,
    barrier: Arc<Barrier>,
}

impl RandomAccess for PanicOnce {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn read_exact_at(&self, off: u64, buf: &mut [u8]) -> Result<(), Error> {
        if self.armed.swap(false, Ordering::SeqCst) {
            self.barrier.wait();
            // Gives the other reader time to start waiting on the window this fill holds.
            std::thread::sleep(Duration::from_millis(200));
            panic!("injected panic in a fill");
        }
        self.inner.read_exact_at(off, buf)
    }
}

#[test]
fn a_panic_in_a_fill_releases_its_windows_and_wakes_waiters() {
    let mut builder = ContainerBuilder::new();
    let blob = builder.add_blob(tensor_like(1, 8 * 4096, 1), whole(), 4096, StreamPolicy::Store);
    let content = builder.add_member("m", &[identity(blob, 0, 8 * 4096)]);
    let barrier = Arc::new(Barrier::new(2));
    let source = Arc::new(PanicOnce {
        inner: MemorySource::silent(builder.finish()),
        armed: AtomicBool::new(false),
        barrier: Arc::clone(&barrier),
    });
    let options = OpenOptions { threads: 2, ..OpenOptions::default() };
    let shared: Arc<dyn RandomAccess> = source.clone();
    let container = Container::from_source(shared, &options).expect("a valid container");
    source.armed.store(true, Ordering::SeqCst);
    let cache = cache(&container, 1 << 20, 0);
    let filler = cache.member("m").expect("present");
    let waiter = cache.member("m").expect("present");
    let handle = std::thread::spawn(move || {
        let mut dst = vec![0; 100];
        filler.read_at(5, &mut dst)
    });
    barrier.wait();
    let mut dst = vec![0; 100];
    waiter.read_at(4096 + 5, &mut dst).expect("the waiter fills the window itself");
    assert_eq!(dst, content[4096 + 5..4096 + 105]);
    assert!(handle.join().is_err(), "the filler panicked");
    waiter.read_at(5, &mut dst).expect("the abandoned window reads later");
    assert_eq!(dst, content[5..105]);
    let stats = cache.stats();
    assert_eq!(stats.in_flight_bytes, 0, "{stats:?}");
    assert!(stats.misses >= 2, "{stats:?}");
}

/// A source whose first read once armed meets the test at a barrier twice: on entry, and to go on.
struct HoldOnce {
    inner: MemorySource,
    armed: AtomicBool,
    barrier: Arc<Barrier>,
}

impl RandomAccess for HoldOnce {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn read_exact_at(&self, off: u64, buf: &mut [u8]) -> Result<(), Error> {
        if self.armed.swap(false, Ordering::SeqCst) {
            self.barrier.wait();
            self.barrier.wait();
        }
        self.inner.read_exact_at(off, buf)
    }
}

/// While one fill holds the whole budget in flight, another reader's missed window must still stay
/// resident, or a reader whose reads are smaller than a window decodes that window on every read.
#[test]
fn a_missed_window_stays_resident_while_fills_hold_the_budget() {
    let (file, content) = many_windows();
    let barrier = Arc::new(Barrier::new(2));
    let source = Arc::new(HoldOnce {
        inner: MemorySource::silent(file),
        armed: AtomicBool::new(false),
        barrier: Arc::clone(&barrier),
    });
    let options = OpenOptions { threads: 2, ..OpenOptions::default() };
    let shared: Arc<dyn RandomAccess> = source.clone();
    let container = Container::from_source(shared, &options).expect("a valid container");
    // Two windows of budget, and a run of two: the first reader's fill holds all of it.
    let cache = cache(&container, 2 * 4096, 2);
    let first = cache.member("m").expect("present");
    let second = cache.member("m").expect("present");
    source.armed.store(true, Ordering::SeqCst);
    let handle = std::thread::spawn(move || {
        let mut dst = vec![0; 1500];
        first.read_at(0, &mut dst).map(|()| dst)
    });
    barrier.wait();
    assert_eq!(cache.stats().in_flight_bytes, 2 * 4096, "{:?}", cache.stats());
    for piece in 0..3 {
        // Three reads inside window 5.
        let offset = 5 * 4096 + piece * 1300;
        let mut dst = vec![0; 1300];
        second.read_at(offset as u64, &mut dst).expect("in range");
        assert_eq!(dst, content[offset..offset + 1300]);
    }
    let stats = cache.stats();
    assert_eq!(stats.misses, 2, "the second reader decoded window 5 once: {stats:?}");
    assert_eq!(stats.hits, 2, "{stats:?}");
    assert_eq!(stats.resident_bytes, 4096, "{stats:?}");
    assert!(stats.peak_bytes <= 2 * 4096 + 2 * 4096, "{stats:?}");
    barrier.wait();
    let read = handle.join().expect("no panic").expect("in range");
    assert_eq!(read, content[..1500]);
    let stats = cache.stats();
    assert!(stats.resident_bytes <= 2 * 4096, "{stats:?}");
    assert_eq!(stats.in_flight_bytes, 0, "{stats:?}");
}

/// Reads hot windows of [`many_windows`] through a small cache from the tasks of an outer rayon pool
/// of `outer` threads, every byte checked, and reports completion on a channel.
fn read_from_outer_pool(container: &Container, content: &[u8], outer: usize) {
    use rayon::prelude::{IntoParallelIterator, ParallelIterator};
    // Runs of 4 windows (one batch of several jobs, so the fill enters the container's pool) in a
    // budget of 8, over 8 hot windows 8 apart: tasks keep meeting on a window another task fills.
    let cache = cache(container, 8 * 4096, 4);
    let cached = cache.member("m").expect("present");
    let pool = rayon::ThreadPoolBuilder::new().num_threads(outer).build().expect("an outer pool");
    pool.install(|| {
        (0..2000_usize).into_par_iter().for_each(|task| {
            let offset = (task % 8) * 8 * 4096 + 17;
            let mut dst = vec![0; 1000];
            cached.read_at(offset as u64, &mut dst).expect("in range");
            assert_eq!(dst, content[offset..offset + 1000], "task {task}, offset {offset}");
        });
    });
    let stats = cache.stats();
    assert_eq!(stats.in_flight_bytes, 0, "{stats:?}");
    assert!(stats.misses > 0, "{stats:?}");
}

/// A fill from a worker of another rayon pool must not let that worker run a task that waits for the
/// fill's own windows. Without the helper thread of `Inner::decode_run`, `ThreadPool::install` from a
/// foreign worker steals outer tasks while it waits, one of them waits on the `Condvar` for a window
/// the same thread holds as `Filling`, and the process hangs; with two outer workers the wait can also
/// form a cycle across both. A watchdog turns a hang into a failure.
#[test]
fn fills_from_another_rayon_pool_never_wait_on_themselves() {
    let (file, content) = many_windows();
    let (done, finished) = std::sync::mpsc::channel();
    // Detached on purpose: a regression leaves this thread hung, and the test must fail, not hang.
    std::thread::spawn(move || {
        for (threads, outer) in [(4, 4), (2, 2), (4, 2), (2, 8)] {
            let container = open(file.clone(), threads, Verify::Chunk);
            read_from_outer_pool(&container, &content, outer);
        }
        // An error means the watchdog gave up first and already failed the test.
        done.send(()).ok();
    });
    finished
        .recv_timeout(Duration::from_secs(120))
        .expect("reads from an outer rayon pool finished, so no fill waited on itself");
}

/// A panic of a fill decoded on the helper thread reaches the rayon caller, and the run is released.
#[test]
fn a_panic_in_a_fill_from_a_rayon_worker_reaches_the_caller_and_releases_the_run() {
    let mut builder = ContainerBuilder::new();
    let blob = builder.add_blob(tensor_like(1, 8 * 4096, 1), whole(), 4096, StreamPolicy::Store);
    let content = builder.add_member("m", &[identity(blob, 0, 8 * 4096)]);
    let source = Arc::new(PanicOnce {
        inner: MemorySource::silent(builder.finish()),
        armed: AtomicBool::new(false),
        barrier: Arc::new(Barrier::new(1)),
    });
    let options = OpenOptions { threads: 2, ..OpenOptions::default() };
    let shared: Arc<dyn RandomAccess> = source.clone();
    let container = Container::from_source(shared, &options).expect("a valid container");
    source.armed.store(true, Ordering::SeqCst);
    // A run of two windows is a batch of two jobs, so the fill would enter the container's pool.
    let cache = cache(&container, 1 << 20, 2);
    let cached = cache.member("m").expect("present");
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().expect("an outer pool");
    let mut dst = vec![0; 100];
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.install(|| cached.read_at(5, &mut dst))
    }));
    assert!(outcome.is_err(), "the fill's panic reached the caller");
    assert_eq!(cache.stats().in_flight_bytes, 0, "{:?}", cache.stats());
    pool.install(|| cached.read_at(5, &mut dst)).expect("the released window reads later");
    assert_eq!(dst, content[5..105]);
}
