//! File: tests/malicious.rs
//! Purpose: hostile files against the public reader: each case gives a typed error, never a panic,
//! with peak memory bounded, or is a measurement of a cost that is bounded and open to a decision.
//!
//! Main responsibilities:
//! - bombs: decompression, index, shared chunks, names, decode work, open time at the index cap;
//! - structure: overlaps, truncation at every 4 KiB, a bit flip in every section, shapes, names, ids;
//! - the integrity promise: a mutated file is refused or fails the member BLAKE3, and what `read_at`
//!   and `Verify::Off` do not protect is pinned down by tests that show it;
//! - delta blobs (spec section 7a): the same promise over `harness::delta_sample` when the base
//!   chunks, the delta chunks, `PARM` or the base references change, and decode work that counts the
//!   base once per delta extent;
//! - the seeded mutator: the fuzz targets' own functions, on stable, on every `cargo test`.
//!
//! Key structures: `Kit` (hostile file builder), `Counting` (the allocator of this binary), `Peak`,
//! `Tally`.
//!
//! Implementation notes:
//! - Memory: a global allocator counts the whole process, and tests run concurrently, so every test
//!   holds [`shared`] and a test that measures holds [`exclusive`], as its first statement so that the
//!   guard outlives every local: a late free in the counters would hide bytes. Rust allocations are counted
//!   exactly. zstd's C allocations are not, so on Linux the peak is also read from `VmHWM`, after
//!   resetting it through `/proc/self/clear_refs`, and the two are compared with the same bound.
//! - Tests named `measure_*` are `#[ignore]`d: they report numbers with `--release --ignored
//!   --nocapture` and decide nothing. Their small versions run in the normal suite.
//! - `harness` is `fuzz/src/harness.rs`, the file the fuzz targets compile. A panic there is a finding.
//! - Bombs are built with `format::IndexBuilder`, which enforces nothing by design. Byte-level accept
//!   and reject cases that do not depend on it are in `tests/conformance.rs`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Instant;

#[path = "common/fixtures.rs"]
mod fixtures;
#[path = "../../../fuzz/src/harness.rs"]
mod harness;

use fixtures::{Elem, GgufOptions, OnnxModel, OnnxTensor, PtZipOptions, Rng, TensorSpec};
use harness::{Bytes, Prng, Sample, assemble, assemble_with, delta_sample, sample, split_file};
use ziptensors_core::chain::{BLOB_KIND_DELTA, FEATURE_DELTA, RawStep};
use ziptensors_core::codec::{StreamPolicy, Workspace, encode_chunk, widen_into};
use ziptensors_core::format::{
    BlobRecord, ChainRecord, ChunkRecord, DeltaParams, ExtentRecord, Footer, HEADER_LEN, Header,
    IndexBuilder, DEFAULT_MAX_NAME_WORK_BYTES, MemberRecord, StrRef, TensorRecord, check_restore_name,
};
use ziptensors_core::io::TEMP_FILE_SUFFIX;
use ziptensors_core::{
    Chain, Container, DeltaOp, Dtype, ElementWidth, EncodeOptions, Error, ErrorCode, Layout as ChunkLayout,
    Mode, OpenOptions, Profile, RandomAccess, ReadRequest, Step, Verify, View, encode_file,
};

/// Counts live bytes and their peak, passing everything to the system allocator.
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(by: usize) {
    let live = LIVE.fetch_add(by, Ordering::Relaxed) + by;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

fn shrank(by: usize) {
    LIVE.fetch_sub(by, Ordering::Relaxed);
}

// SAFETY: every call forwards to `System` with its arguments unchanged, and the counters neither
// allocate nor touch the memory.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract is forwarded.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            grew(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract is forwarded.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            grew(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract is forwarded.
        unsafe { System.dealloc(pointer, layout) };
        shrank(layout.size());
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller's contract is forwarded.
        let moved = unsafe { System.realloc(pointer, layout, new_size) };
        if !moved.is_null() {
            shrank(layout.size());
            grew(new_size);
        }
        moved
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Keeps measuring tests alone in the process. Poison only says another test failed.
static SERIAL: RwLock<()> = RwLock::new(());

/// Held by every test that does not measure memory.
fn shared() -> RwLockReadGuard<'static, ()> {
    SERIAL.read().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Held by a test that measures memory, so that the counters see only that test.
fn exclusive() -> RwLockWriteGuard<'static, ()> {
    SERIAL.write().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// What a closure took, above where it started.
#[derive(Debug, Clone, Copy)]
struct Peak {
    /// Peak of live Rust allocations.
    rust: usize,
    /// Growth of the resident high-water mark, in bytes. `None` off Linux or if `/proc` refuses.
    rss: Option<usize>,
    /// Wall time in seconds.
    seconds: f64,
}

impl Peak {
    /// Requires both measures to be within `bound` bytes.
    fn assert_within(&self, bound: usize, what: &str) {
        assert!(self.rust <= bound, "{what}: {self:?} over {bound}");
        if let Some(rss) = self.rss {
            assert!(rss <= bound, "{what}: {self:?} over {bound}");
        }
    }
}

/// One `kB` line of `/proc/self/status`, in bytes.
fn proc_status_bytes(field: &str) -> Option<usize> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with(field))?;
    let kib: usize = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

/// Resident bytes now, with the high-water mark reset to it. `None` if this needs what is not there.
fn reset_rss_peak() -> Option<usize> {
    fs::write("/proc/self/clear_refs", "5").ok()?;
    proc_status_bytes("VmRSS:")
}

/// Runs `work` and returns what it cost. The caller holds [`exclusive`].
fn measure<T>(work: impl FnOnce() -> T) -> (T, Peak) {
    let rss_start = reset_rss_peak();
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let started = Instant::now();
    let value = work();
    let seconds = started.elapsed().as_secs_f64();
    let rust = PEAK.load(Ordering::Relaxed).saturating_sub(base);
    let rss_end = proc_status_bytes("VmHWM:");
    let rss = rss_start.zip(rss_end).map(|(start, end)| end.saturating_sub(start));
    (value, Peak { rust, rss, seconds })
}

const MIB: usize = 1 << 20;
const MAX_CHUNK: u32 = 8 << 20;

/// One thread, so that a run is deterministic and a measure holds exactly one scratch entry.
fn options() -> OpenOptions {
    OpenOptions { threads: 1, ..OpenOptions::default() }
}

fn open(file: Vec<u8>, options: &OpenOptions) -> Result<Container, Error> {
    Container::from_source(Arc::new(Bytes(file)), options)
}

/// The stable code of an error that must have happened.
fn code_of<T>(result: Result<T, Error>) -> ErrorCode {
    match result {
        Ok(_) => panic!("a hostile input was accepted"),
        Err(error) => error.code(),
    }
}

/// A source that records the offset of every read.
#[derive(Debug)]
struct Traced {
    inner: Bytes,
    offsets: Mutex<Vec<u64>>,
}

impl Traced {
    fn new(file: Vec<u8>) -> Arc<Self> {
        Arc::new(Self { inner: Bytes(file), offsets: Mutex::new(Vec::new()) })
    }

    /// Reads that started in the data region, whose end is `index_offset`.
    fn data_reads(&self, index_offset: u64) -> usize {
        let offsets = self.offsets.lock().expect("no test panics while recording");
        offsets.iter().filter(|at| (HEADER_LEN..index_offset).contains(*at)).count()
    }
}

impl RandomAccess for Traced {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn read_exact_at(&self, off: u64, buf: &mut [u8]) -> Result<(), Error> {
        self.offsets.lock().expect("no test panics while recording").push(off);
        self.inner.read_exact_at(off, buf)
    }
}

/// Unique scratch directory, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let name = format!("ziptensors-malicious-{}-{id}", std::process::id());
        let path = std::env::temp_dir().join(name);
        fs::create_dir_all(&path).expect("scratch directory");
        Self(path)
    }

    fn entries(&self) -> usize {
        fs::read_dir(&self.0).expect("scratch directory").count()
    }

    /// Asserts what a restore to `output` (a file name in this directory) promises after it failed,
    /// or after its committed file was removed: no entry named `output`, and no temporary file of
    /// ours (`TEMP_FILE_SUFFIX`). Any other entry fails too, except `<OUTPUT UPPER-CASED>.tmp`:
    /// on Windows with an antivirus filter, renaming a file onto a
    /// name and deleting it later makes ANOTHER process create that name for a few hundred
    /// milliseconds. Shown with plain `std::fs` and with .NET `File.Move` + `File.Delete`, no
    /// ziptensors code involved; never seen on Linux or under wine.
    fn assert_restore_left_nothing(&self, output: &str, what: &str) {
        let foreign = format!("{}.tmp", output.to_uppercase());
        for entry in fs::read_dir(&self.0).expect("scratch directory") {
            let name = entry.expect("scratch entry").file_name().to_string_lossy().into_owned();
            assert!(name != output && !name.ends_with(TEMP_FILE_SUFFIX), "{what}: {name}");
            assert_eq!(name, foreign, "{what}: an entry no other process is known to create");
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("cannot remove {}: {error}", self.0.display());
        }
    }
}

/// A `whole` chain.
fn whole() -> Chain {
    Chain::new(&[Step::Layout(ChunkLayout::Whole)]).expect("a valid chain")
}

/// A blob record of kind 0 with no identity.
fn blob_record(raw_len: u64, first_chunk: u64, chunk_count: u32, chunk_size: u32, chain_id: u32) -> BlobRecord {
    BlobRecord {
        blake3: [0; 32],
        raw_len,
        first_chunk,
        chunk_count,
        chunk_size,
        chain_id,
        kind: 0,
        dtype_hint: Dtype::U8,
        params_off: 0,
        params_len: 0,
    }
}

/// A member with no tensors.
fn member_record(name: StrRef, file_len: u64, blake3: [u8; 32], first_extent: u64, extent_count: u32) -> MemberRecord {
    MemberRecord {
        name,
        file_len,
        blake3,
        first_extent,
        extent_count,
        first_tensor: 0,
        tensor_count: 0,
        profile: Profile::Opaque,
    }
}

/// An identity extent.
fn extent(virt_offset: u64, virt_len: u64, blob_id: u32, blob_offset: u64) -> ExtentRecord {
    ExtentRecord { virt_offset, virt_len, blob_id, view_id: 0, blob_offset }
}

/// Builds hostile files: stored bytes go in as given, records as asked, nothing is checked.
#[derive(Debug, Default)]
struct Kit {
    data: Vec<u8>,
    index: IndexBuilder,
    /// Records added through [`Kit::push_chunk`] and [`Kit::push_extent`].
    pushed_chunks: u64,
    pushed_extents: u64,
}

impl Kit {
    /// A kit holding the `whole` chain, as `chain_id` 0.
    fn new() -> Self {
        let mut kit = Self::default();
        kit.index.add_chain(whole()).expect("one chain");
        kit
    }

    /// Adds a chunk where the test knows its id from the order of the calls, which this checks.
    fn push_chunk(&mut self, record: ChunkRecord) {
        assert_eq!(self.index.add_chunk(record), self.pushed_chunks);
        self.pushed_chunks += 1;
    }

    /// Adds an extent where the test knows its id from the order of the calls, which this checks.
    fn push_extent(&mut self, record: ExtentRecord) {
        assert_eq!(self.index.add_extent(record), self.pushed_extents);
        self.pushed_extents += 1;
    }

    /// Appends `stored` to the data region and returns its file offset.
    fn put(&mut self, stored: &[u8]) -> u64 {
        let offset = HEADER_LEN + self.data.len() as u64;
        self.data.extend_from_slice(stored);
        offset
    }

    /// Appends a chunk encoded from `raw` under `policy`, and returns its record, not yet added.
    fn encoded(&mut self, raw: &[u8], policy: StreamPolicy) -> ChunkRecord {
        let mut stored = Vec::new();
        let meta = encode_chunk(raw, &whole(), &policy, &mut Workspace::new(), &mut stored)
            .expect("encodable chunk");
        let file_offset = self.put(&stored);
        ChunkRecord { file_offset, xxh3_decoded: meta.xxh3_decoded, stored_len: meta.stored_len }
    }

    /// Adds a blob of one chunk over `raw`, stored as given by `policy`. Returns the blob id.
    fn one_chunk_blob(&mut self, raw: &[u8], chunk_size: u32, policy: StreamPolicy) -> u32 {
        let record = self.encoded(raw, policy);
        let first_chunk = self.index.add_chunk(record);
        let blob = blob_record(raw.len() as u64, first_chunk, 1, chunk_size, 0);
        self.index.add_blob(blob).expect("few blobs")
    }

    /// Adds `extents` as they are, and a member named `name` over them with content hash `blake3`.
    fn member(&mut self, name: &str, extents: &[ExtentRecord], blake3: [u8; 32]) {
        let mut first_extent = None;
        for record in extents {
            let id = self.index.add_extent(*record);
            first_extent.get_or_insert(id);
        }
        let name = self.index.add_string(name).expect("a short name");
        let file_len = extents.last().map_or(0, |last| last.virt_offset + last.virt_len);
        let count = u32::try_from(extents.len()).expect("few extents");
        let record = member_record(name, file_len, blake3, first_extent.unwrap_or(0), count);
        self.index.add_member(record).expect("few members");
    }

    fn finish(self) -> Vec<u8> {
        assemble(&self.data, &self.index.serialize())
    }
}

/// A file whose one member is `content`, stored in one `whole` chunk with stream codec 0.
fn stored_file(content: &[u8]) -> Vec<u8> {
    let mut kit = Kit::new();
    let chunk_size = u32::try_from(content.len()).expect("a small file");
    let blob = kit.one_chunk_blob(content, chunk_size, StreamPolicy::Store);
    let span = extent(0, content.len() as u64, blob, 0);
    kit.member("victim", &[span], *blake3::hash(content).as_bytes());
    kit.finish()
}

/// Hand-written zstd frame of a single RLE block: `count` copies of `byte`, declaring
/// `declared` bytes of content, in a window of `2^(10 + exponent)` bytes.
fn rle_frame(byte: u8, count: u32, declared: u32, exponent: u8) -> Vec<u8> {
    let mut frame = vec![0x28, 0xB5, 0x2F, 0xFD];
    // Frame header descriptor: four bytes of content size, no single segment, no checksum, no dict.
    frame.push(0x80);
    frame.push(exponent << 3);
    frame.extend_from_slice(&declared.to_le_bytes());
    // Block header: last block, type RLE, size in the top 21 bits of three bytes.
    let block = (count << 3) | 0b011;
    frame.extend_from_slice(&block.to_le_bytes()[..3]);
    frame.push(byte);
    frame
}

/// A stored chunk of one compressed stream `frame`.
fn zstd_chunk(frame: &[u8]) -> Vec<u8> {
    let word = (1_u32 << 24) | u32::try_from(frame.len()).expect("a short frame");
    [word.to_le_bytes().as_slice(), frame].concat()
}

/// A file with one chunk of `raw_len` bytes whose stored form is `stored`, and one member over it.
fn hostile_chunk_file(stored: &[u8], raw_len: u32) -> Vec<u8> {
    let mut kit = Kit::new();
    let file_offset = kit.put(stored);
    let stored_len = u32::try_from(stored.len()).expect("a short chunk");
    let first_chunk = kit.index.add_chunk(ChunkRecord { file_offset, xxh3_decoded: 0, stored_len });
    let blob = blob_record(u64::from(raw_len), first_chunk, 1, raw_len, 0);
    let blob = kit.index.add_blob(blob).expect("one blob");
    kit.member("bomb", &[extent(0, u64::from(raw_len), blob, 0)], [0; 32]);
    kit.finish()
}

/// Bytes a read of a hostile chunk may use above its destination: scratch and a stored batch for one
/// thread and one chunk of at most 8 MiB, and zstd's fixed context.
const ONE_CHUNK_READ_BOUND: usize = 26 * MIB;

#[test]
fn a_frame_that_unpacks_to_more_than_its_chunk_is_refused_with_no_growth() {
    let _guard = exclusive();
    let frame = zstd::bulk::compress(&vec![0; MAX_CHUNK as usize], 3).expect("compress");
    let raw_len = 1024;
    assert!(frame.len() < raw_len as usize, "a tiny frame: {}", frame.len());
    let file = hostile_chunk_file(&zstd_chunk(&frame), raw_len);
    let (code, peak) = measure(|| {
        let container = open(file, &options()).expect("the index is valid");
        let member = container.single().expect("one member");
        code_of(member.read_at(0, &mut [0; 1024]))
    });
    assert_eq!(code, ErrorCode::ChunkCorrupt);
    peak.assert_within(ONE_CHUNK_READ_BOUND, "frame of 8 MiB in a chunk of 1 KiB");
}

#[test]
fn a_frame_that_lies_about_its_size_cannot_write_past_the_chunk() {
    let _guard = shared();
    // Declares the chunk's length, unpacks to twice that.
    let lying = rle_frame(0xAA, 2048, 1024, 0);
    let file = hostile_chunk_file(&zstd_chunk(&lying), 1024);
    let container = open(file, &options()).expect("the index is valid");
    let member = container.single().expect("one member");
    assert_eq!(code_of(member.read_at(0, &mut [0; 1024])), ErrorCode::ChunkCorrupt);
}

#[test]
fn a_frame_asking_for_a_two_gib_window_is_refused_before_zstd() {
    let _guard = exclusive();
    let frame = rle_frame(0, 1024, 1024, 21);
    let file = hostile_chunk_file(&zstd_chunk(&frame), 1024);
    let (code, peak) = measure(|| {
        let container = open(file, &options()).expect("the index is valid");
        let member = container.single().expect("one member");
        code_of(member.read_at(0, &mut [0; 1024]))
    });
    assert_eq!(code, ErrorCode::ChunkCorrupt);
    peak.assert_within(ONE_CHUNK_READ_BOUND, "window of 2 GiB");
}

/// One stored chunk of zeros and `count` chunk records pointing at it: 8 MiB × `count` declared.
fn shared_chunk_bomb(count: u32) -> (Vec<u8>, u64) {
    let mut kit = Kit::new();
    let zeros = vec![0; MAX_CHUNK as usize];
    let record = kit.encoded(&zeros, StreamPolicy::Entropy);
    let mut first_chunk = None;
    for _ in 0..count {
        let id = kit.index.add_chunk(record);
        first_chunk.get_or_insert(id);
    }
    let raw_len = u64::from(count) * u64::from(MAX_CHUNK);
    let blob = blob_record(raw_len, first_chunk.unwrap_or(0), count, MAX_CHUNK, 0);
    let blob = kit.index.add_blob(blob).expect("one blob");
    kit.member("bomb", &[extent(0, raw_len, blob, 0)], [0; 32]);
    (kit.finish(), raw_len)
}

#[test]
fn identical_chunks_declaring_half_a_tib_are_refused_by_the_budget_before_any_data_read() {
    let _guard = shared();
    let (file, declared) = shared_chunk_bomb(1 << 16);
    assert_eq!(declared, 1 << 39);
    assert!(file.len() < 64 * 1024, "a small file: {}", file.len());
    let index_offset = split_file(&file).0.len() as u64 + HEADER_LEN;
    let source = Traced::new(file);
    let budget = OpenOptions { max_total_decoded: 1 << 30, ..options() };
    let error = Container::from_source(source.clone(), &budget).expect_err("over budget");
    assert!(matches!(error, Error::BudgetExceeded { what: "total decoded size", .. }), "{error:?}");
    assert_eq!(source.data_reads(index_offset), 0, "the budget comes before any data read");
}

/// The other side of the case above: `max_total_decoded` is `u64::MAX` by default, so a caller who
/// sets no budget opens this. Memory stays bounded, the read is the caller's own, and what restore
/// would write is the caller's disk. A frontend restoring untrusted files must set a budget.
#[test]
fn without_a_budget_the_same_file_opens_and_memory_stays_bounded() {
    let _guard = exclusive();
    let (file, declared) = shared_chunk_bomb(1 << 16);
    let (member, peak) = measure(|| {
        let container = open(file, &options()).expect("identical chunks are valid");
        let member = container.single().expect("one member");
        assert_eq!(member.len(), declared);
        let mut prefix = vec![0; 16 * MIB];
        member.read_at(0, &mut prefix).expect("a read of one chunk repeated");
        assert!(prefix.iter().all(|byte| *byte == 0));
        member
    });
    peak.assert_within(16 * MIB + ONE_CHUNK_READ_BOUND + 16 * MIB, "16 MiB read of a 512 GiB member");
    drop(member);
}

/// A file of `stored` as its index, with hashes valid for `raw_index`.
fn file_with_stored_index(stored: &[u8], raw_index: &[u8]) -> Vec<u8> {
    let mut file = Header::current().to_bytes().to_vec();
    let index_offset = file.len() as u64;
    file.extend_from_slice(stored);
    let footer = Footer::for_index(index_offset, stored.len() as u64, raw_index);
    file.extend_from_slice(&footer.to_bytes());
    file
}

#[test]
fn an_index_declared_over_one_gib_is_refused_under_any_configuration() {
    let _guard = exclusive();
    let footer = Footer { index_raw_len: (1 << 30) + 1, ..Footer::for_index(64, 16, &[]) };
    let mut file = Header::current().to_bytes().to_vec();
    file.extend_from_slice(&[0; 16]);
    file.extend_from_slice(&footer.to_bytes());
    let unlimited = OpenOptions { max_index_bytes: u64::MAX, ..options() };
    let (error, peak) = measure(|| open(file, &unlimited).expect_err("over the format's ceiling"));
    assert!(matches!(error, Error::BudgetExceeded { what: "index", .. }), "{error:?}");
    peak.assert_within(MIB, "refused from the footer alone");
}

#[test]
fn an_index_bomb_is_bounded_by_the_cap() {
    let _guard = exclusive();
    // A real frame of zeros: tiny on disk, exactly at the default cap once decompressed.
    let raw = vec![0; 256 * MIB];
    let stored = zstd::bulk::compress(&raw, 1).expect("compress");
    assert!(stored.len() < 64 * 1024, "a small file: {}", stored.len());
    let over = [raw.as_slice(), &[0]].concat();
    let over_stored = zstd::bulk::compress(&over, 1).expect("compress");
    let file = file_with_stored_index(&stored, &raw);
    let over_file = file_with_stored_index(&over_stored, &over);
    drop((raw, over));
    let (error, peak) = measure(|| open(file, &options()).expect_err("no sections"));
    assert!(matches!(error, Error::MalformedIndex { .. }), "{error:?}");
    peak.assert_within(256 * MIB + MIB, "index at the cap");
    let (error, peak) = measure(|| open(over_file, &options()).expect_err("over the cap"));
    assert!(matches!(error, Error::BudgetExceeded { what: "index", .. }), "{error:?}");
    peak.assert_within(MIB, "index one byte over the cap, refused from the footer");
}

/// Offset of a `u64` in a section table entry: the body's offset, its length, its record count.
const ENTRY_OFFSET: usize = 8;
const ENTRY_LEN: usize = 16;
const ENTRY_COUNT: usize = 24;
/// Bytes of the section table before its entries, and of one entry.
const TABLE_START: usize = 8;
const ENTRY_BYTES: usize = 32;
/// The nine mandatory sections of spec section 4.
const TAGS: [&[u8; 4]; 9] =
    [b"STRS", b"CHAN", b"CHST", b"BLOB", b"CHNK", b"MEMB", b"EXTN", b"TENS", b"SHAP"];

/// Start of the table entry of section `tag` in a raw index.
fn table_entry(raw: &[u8], tag: &[u8; 4]) -> usize {
    let count = u32::from_le_bytes(raw[..4].try_into().expect("four bytes"));
    (0..count as usize)
        .map(|at| TABLE_START + at * ENTRY_BYTES)
        .find(|start| &raw[*start..*start + 4] == tag)
        .expect("a mandatory section")
}

/// Reads the `u64` at `at`.
fn read_u64(raw: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(raw[at..at + 8].try_into().expect("eight bytes"))
}

/// The bytes of section `tag` in a raw index, as a range.
fn body_of(raw: &[u8], tag: &[u8; 4]) -> std::ops::Range<usize> {
    let entry = table_entry(raw, tag);
    let start = usize::try_from(read_u64(raw, entry + ENTRY_OFFSET)).expect("a small index");
    let len = usize::try_from(read_u64(raw, entry + ENTRY_LEN)).expect("a small index");
    start..start + len
}

/// Overwrites the `u64` at `at`.
fn write_u64(raw: &mut [u8], at: usize, value: u64) {
    raw[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

#[test]
fn a_section_table_that_lies_about_sizes_is_malformed_with_no_allocation() {
    let _guard = exclusive();
    let file = stored_file(b"payload");
    let (data, raw) = split_file(&file);
    let mut hostile = Vec::new();

    let mut table = raw.clone();
    table[..4].copy_from_slice(&u32::MAX.to_le_bytes());
    hostile.push(table);
    for tag in TAGS {
        let entry = table_entry(&raw, tag);
        for (field, value) in [
            (ENTRY_COUNT, u64::MAX),
            (ENTRY_COUNT, 1 << 61),
            (ENTRY_LEN, u64::MAX),
            (ENTRY_OFFSET, u64::MAX),
            (ENTRY_OFFSET, 0),
        ] {
            let mut lying = raw.clone();
            write_u64(&mut lying, entry + field, value);
            hostile.push(lying);
        }
    }
    let (codes, peak) = measure(|| {
        let opened = hostile.iter().map(|raw| open(assemble(&data, raw), &options()));
        opened.map(code_of).collect::<Vec<_>>()
    });
    assert!(codes.iter().all(|code| *code == ErrorCode::MalformedIndex), "{codes:?}");
    peak.assert_within(MIB, "lying section tables");
}

/// A chunk of 8 MiB of zeros and `count` one-byte extents over it, as one member.
fn one_byte_extents(count: u32) -> Vec<u8> {
    let mut kit = Kit::new();
    let zeros = vec![0; MAX_CHUNK as usize];
    let blob = kit.one_chunk_blob(&zeros, MAX_CHUNK, StreamPolicy::Entropy);
    let extents: Vec<ExtentRecord> =
        (0..u64::from(count)).map(|at| extent(at, 1, blob, at % u64::from(MAX_CHUNK))).collect();
    let content = vec![0; count as usize];
    kit.member("amplifier", &extents, *blake3::hash(&content).as_bytes());
    kit.finish()
}

#[test]
fn a_member_declaring_two_to_the_32_extents_on_one_chunk_is_refused() {
    let _guard = shared();
    // 2^32 extents need 128 GiB of index, over the format's ceiling of 1 GiB by themselves. So a
    // file can only claim them, through the count of the member or of the section.
    let mut kit = Kit::new();
    let blob = kit.one_chunk_blob(&[0; 64], 64, StreamPolicy::Store);
    kit.push_extent(extent(0, 64, blob, 0));
    let name = kit.index.add_string("claims").expect("a short name");
    kit.index.add_member(member_record(name, 64, [0; 32], 0, u32::MAX)).expect("one member");
    assert_eq!(code_of(open(kit.finish(), &options())), ErrorCode::MalformedIndex);

    let mut kit = Kit::new();
    let blob = kit.one_chunk_blob(&[0; 64], 64, StreamPolicy::Store);
    kit.member("real", &[extent(0, 64, blob, 0)], [0; 32]);
    let (data, mut raw) = split_file(&kit.finish());
    let entry = table_entry(&raw, b"EXTN");
    write_u64(&mut raw, entry + ENTRY_COUNT, 1 << 32);
    assert_eq!(code_of(open(assemble(&data, &raw), &options())), ErrorCode::MalformedIndex);

    let mut kit = Kit::new();
    let blob = kit.one_chunk_blob(&[0; 64], 64, StreamPolicy::Store);
    kit.push_extent(extent(0, 64, blob, 0));
    let name = kit.index.add_string("offset").expect("a short name");
    kit.index.add_member(member_record(name, 64, [0; 32], u64::MAX, 2)).expect("one member");
    assert_eq!(code_of(open(kit.finish(), &options())), ErrorCode::MalformedIndex);
}

#[test]
fn two_million_one_byte_extents_on_one_chunk_are_refused_before_any_data_read() {
    let _guard = exclusive();
    // The most a default index cap leaves room for is about 2^23; 2^21 shows the same with less work.
    let file = one_byte_extents(1 << 21);
    let index_offset = split_file(&file).0.len() as u64 + HEADER_LEN;
    let source = Traced::new(file);
    let (error, peak) = measure(|| {
        Container::from_source(source.clone(), &options()).expect_err("too much decode work")
    });
    let Error::BudgetExceeded { what: "decode work", declared, limit } = error else {
        panic!("{error:?}");
    };
    assert_eq!(declared, (1 << 21) * u64::from(MAX_CHUNK));
    assert_eq!(limit, 16 * (1 << 21) + (256 << 20));
    assert_eq!(source.data_reads(index_offset), 0);
    // The index, its decoded records, and one prefix sum per extent.
    peak.assert_within(256 * MIB, "index of 2^21 extents");
}

#[test]
fn a_small_one_byte_extents_file_is_refused_too() {
    let _guard = shared();
    let error = open(one_byte_extents(40), &options()).expect_err("over the floor");
    assert!(matches!(error, Error::BudgetExceeded { what: "decode work", .. }), "{error:?}");
}

/// `members` members, each `extents` one-byte extents over one 8 MiB chunk, sharing one run of
/// extent records, so that the file barely grows with either count.
fn floor_pack(members: u32, extents: u32) -> Vec<u8> {
    let mut kit = Kit::new();
    let zeros = vec![0; MAX_CHUNK as usize];
    let blob = kit.one_chunk_blob(&zeros, MAX_CHUNK, StreamPolicy::Entropy);
    for at in 0..u64::from(extents) {
        kit.push_extent(extent(at, 1, blob, at));
    }
    let content = vec![0; extents as usize];
    let hash = *blake3::hash(&content).as_bytes();
    for id in 0..members {
        let name = kit.index.add_string(&format!("m{id}")).expect("a short name");
        let record = member_record(name, u64::from(extents), hash, 0, extents);
        kit.index.add_member(record).expect("few members");
    }
    kit.finish()
}

/// Extents that put a container exactly at the floor of 256 MiB with 8 MiB chunks.
const FLOOR_EXTENTS: u32 = 32;

/// Members cannot add up floors: the file of 8.4 KiB with 2000 members would decode 500 GiB, and is
/// refused at open, before any data read. One member at the floor still opens.
#[test]
fn members_at_the_floor_share_one_floor_and_their_sum_is_refused() {
    let _guard = shared();
    let members = 2000;
    let file = floor_pack(members, FLOOR_EXTENTS);
    assert!(file.len() < 9 << 10, "a small file: {}", file.len());
    let index_offset = split_file(&file).0.len() as u64 + HEADER_LEN;
    let source = Traced::new(file);
    let budget = OpenOptions { max_total_decoded: 1 << 20, ..options() };
    let error = Container::from_source(source.clone(), &budget).expect_err("too much decode work");
    let Error::BudgetExceeded { what: "decode work", declared, limit } = error else {
        panic!("{error:?}");
    };
    let work = u64::from(members) * u64::from(FLOOR_EXTENTS) * u64::from(MAX_CHUNK);
    assert_eq!(declared, work);
    assert_eq!(work >> 30, 500);
    assert_eq!(limit, 16 * u64::from(members * FLOOR_EXTENTS) + (256 << 20));
    assert_eq!(source.data_reads(index_offset), 0);

    open(floor_pack(1, FLOOR_EXTENTS), &budget).expect("one member fits the floor");
    let error = open(floor_pack(2, FLOOR_EXTENTS), &budget).expect_err("two do not");
    assert!(matches!(error, Error::BudgetExceeded { what: "decode work", .. }), "{error:?}");
}

/// As [`floor_pack`], made to cost what the bound says it costs. A run of extents over one chunk
/// decodes it once, so the extents alternate between two chunks. With `compressed`, the chunks are
/// random nibbles in a real entropy stream, the slow case; without, noise, which is stored.
fn alternating_floor_pack(members: u32, compressed: bool) -> Vec<u8> {
    let mut kit = Kit::new();
    let mut content = Rng::new(61).bytes(2 * MAX_CHUNK as usize);
    if compressed {
        content.iter_mut().for_each(|byte| *byte &= 0x0F);
    }
    let mut first_chunk = None;
    for raw in content.chunks(MAX_CHUNK as usize) {
        let record = kit.encoded(raw, StreamPolicy::Zstd { level: 1 });
        let id = kit.index.add_chunk(record);
        first_chunk.get_or_insert(id);
    }
    let raw_len = 2 * u64::from(MAX_CHUNK);
    let blob = blob_record(raw_len, first_chunk.unwrap_or(0), 2, MAX_CHUNK, 0);
    let blob = kit.index.add_blob(blob).expect("one blob");
    let offsets = (0..u64::from(FLOOR_EXTENTS)).map(|at| at % 2 * u64::from(MAX_CHUNK) + at);
    let bytes: Vec<u8> = offsets.clone().map(|at| content[at as usize]).collect();
    for (at, blob_offset) in offsets.enumerate() {
        kit.push_extent(extent(at as u64, 1, blob, blob_offset));
    }
    let hash = *blake3::hash(&bytes).as_bytes();
    for id in 0..members {
        let name = kit.index.add_string(&format!("m{id}")).expect("a short name");
        let record = member_record(name, u64::from(FLOOR_EXTENTS), hash, 0, FLOOR_EXTENTS);
        kit.index.add_member(record).expect("few members");
    }
    kit.finish()
}

/// Measures what the decode work of the floor costs to read, per 256 MiB of work, one thread and all of
/// them. The pack is over the bound, so it opens with raised limits. Report
/// with `cargo test --release --test malicious -- --ignored --nocapture measure_`.
#[test]
#[ignore = "measurement: decodes gigabytes, run by hand with --release"]
fn measure_decode_work_of_a_pack_of_members_at_the_floor() {
    let _guard = exclusive();
    let members = 32;
    for (threads, compressed) in [(1, false), (0, false), (1, true), (0, true)] {
        let file = alternating_floor_pack(members, compressed);
        let bytes = file.len();
        let (verified, peak) = measure(|| {
            let raised = OpenOptions { threads, max_decode_amplification: 1 << 32, ..options() };
            let container = open(file, &raised).expect("valid");
            let infos = container.members().to_vec();
            for info in &infos {
                container.member(&info.name).expect("listed").verify().expect("valid content");
            }
            infos.len()
        });
        let seconds_each = peak.seconds / f64::from(members);
        eprintln!(
            "floor pack, threads {threads}, compressed {compressed}: file {bytes} bytes, verified {verified} members of 32 bytes, 256 MiB of work each, {seconds_each:.3} s each, {:.2} GiB/s, {peak:?}",
            f64::from(members) * 0.25 / peak.seconds,
        );
    }
    // The work is real: one member reads its chunks once per extent, nothing being cached.
    let file = alternating_floor_pack(1, false);
    let index_offset = split_file(&file).0.len() as u64 + HEADER_LEN;
    let source = Traced::new(file);
    let container = Container::from_source(source.clone(), &options()).expect("valid");
    container.single().expect("one member").verify().expect("valid content");
    assert_eq!(source.data_reads(index_offset), FLOOR_EXTENTS as usize);
}

/// `records` tensors whose name is the whole heap of `heap_bytes` bytes, in no member at all.
fn name_budget_index(heap_bytes: usize, records: u32) -> Vec<u8> {
    let mut kit = Kit::new();
    let name = kit.index.add_string(&"n".repeat(heap_bytes)).expect("a heap under 4 GiB");
    let shape_off = kit.index.add_shape(&[1]).expect("one shape");
    let tensor = TensorRecord { name, virt_offset: 0, virt_len: 0, shape_off, ndim: 1, dtype: Dtype::U8 };
    for _ in 0..records {
        kit.index.add_tensor(tensor).expect("few tensors");
    }
    kit.finish()
}

/// A file of a few KiB spends the whole name budget. The budget bounds CPU, and `tensor_table` copies
/// every name, so memory too: 256 MiB at open, on top of the index. A real pack is far below it.
#[test]
fn an_index_at_the_name_budget_is_bounded_and_one_more_byte_is_refused() {
    let _guard = exclusive();
    const BUDGET: usize = DEFAULT_MAX_NAME_WORK_BYTES as usize;
    const HEAP: usize = BUDGET / 16;
    const REAL_NAMES: usize = 5 * MIB;
    const _: () = assert!(REAL_NAMES * 50 <= BUDGET, "45k tensors have about 5 MB of names");
    let file = name_budget_index(HEAP, 16);
    let bytes = file.len();
    assert!(bytes < MIB, "a small file: {bytes}");
    let over = OpenOptions { threads: 1, ..OpenOptions::default() };
    let (container, peak) = measure(|| open(file, &over).expect("exactly at the budget"));
    assert!(peak.rust >= BUDGET, "{peak:?}");
    eprintln!("name budget: file {bytes} bytes, {peak:?}");
    peak.assert_within(BUDGET + 3 * HEAP, "index at the name budget");
    drop(container);
    let file = name_budget_index(HEAP + 1, 16);
    let error = open(file, &options()).expect_err("over");
    assert!(matches!(error, Error::BudgetExceeded { what: "name bytes", .. }), "{error:?}");
}

/// Record counts of an index built for open time. Every count goes through the costliest code of
/// its section: names are distinct, chunks are disjoint, extents contiguous.
#[derive(Debug, Clone, Copy)]
struct OpenCost {
    tensors: u32,
    members: u32,
    /// Chunks of one blob of one-byte chunks, and as many extents over it, in one member.
    chunks: u32,
}

impl OpenCost {
    /// `share` percent of `bytes` of index in each of the three families.
    fn shares(bytes: usize, tensors: usize, members: usize, chunks: usize) -> Self {
        let count = |share: usize, record: usize| {
            u32::try_from(bytes / 100 * share / record).expect("under 2^32")
        };
        Self { tensors: count(tensors, 32 + 8), members: count(members, 72 + 8), chunks: count(chunks, 24 + 32) }
    }
}

/// An index built to cost the most to open, staying under every limit.
fn open_cost_file(cost: OpenCost) -> Vec<u8> {
    let mut kit = Kit::new();
    let total = cost.tensors as usize + cost.members as usize;
    let heap: String = (0..total).map(|at| format!("{at:08x}")).collect();
    let heap = kit.index.add_string(&heap).expect("a heap under 4 GiB");
    let name = |at: usize| StrRef { off: heap.off + u32::try_from(at * 8).expect("heap"), len: 8 };
    let shape_off = kit.index.add_shape(&[3, 3]).expect("one shape");
    for at in 0..cost.tensors as usize {
        let tensor = TensorRecord {
            name: name(at),
            virt_offset: 0,
            virt_len: 0,
            shape_off,
            ndim: 2,
            dtype: Dtype::U8,
        };
        kit.index.add_tensor(tensor).expect("under 2^32");
    }
    for at in 0..cost.members as usize {
        let record = member_record(name(cost.tensors as usize + at), 0, [0; 32], 0, 0);
        kit.index.add_member(record).expect("under 2^32");
    }
    if cost.chunks > 0 {
        for at in 0..u64::from(cost.chunks) {
            let file_offset = kit.put(&[0, 0, 0, 0, 0]);
            kit.push_chunk(ChunkRecord { file_offset, xxh3_decoded: 0, stored_len: 5 });
            kit.push_extent(extent(at, 1, 0, at));
        }
        let raw_len = u64::from(cost.chunks);
        kit.index.add_blob(blob_record(raw_len, 0, cost.chunks, 1, 0)).expect("one blob");
        let name = kit.index.add_string("all").expect("a short name");
        let record = member_record(name, raw_len, [0; 32], 0, cost.chunks);
        kit.index.add_member(record).expect("under 2^32");
    }
    kit.finish()
}

/// Opens a file of `cost` and returns what that took, with the index's raw length.
fn open_cost(cost: OpenCost) -> (usize, Peak) {
    let file = open_cost_file(cost);
    let raw_len = split_file(&file).1.len();
    let (container, peak) = measure(|| open(file, &options()).expect("under every limit"));
    drop(container);
    (raw_len, peak)
}

#[test]
fn the_costliest_index_opens_in_memory_linear_in_its_size() {
    let _guard = exclusive();
    let (raw_len, peak) = open_cost(OpenCost::shares(8 * MIB, 40, 30, 30));
    eprintln!("open cost: index {raw_len} bytes, {peak:?}");
    peak.assert_within(16 * raw_len, "costliest index of 8 MiB");
}

#[test]
#[ignore = "measurement: builds and opens indexes at the 256 MiB cap, run by hand with --release"]
fn measure_open_time_at_the_index_cap() {
    let _guard = exclusive();
    let cap = 256 * MIB - MIB;
    for (name, tensors, members, chunks) in [
        ("tensors", 99, 0, 0),
        ("members", 0, 99, 0),
        ("chunks and extents", 0, 0, 99),
        ("mix", 40, 30, 29),
    ] {
        let (raw_len, peak) = open_cost(OpenCost::shares(cap, tensors, members, chunks));
        eprintln!("open cost, {name}: index {raw_len} bytes, {peak:?}");
    }
}

/// Two chunks over `first` and `second`, as `(file_offset, stored_len)`, with a 200-byte data region
/// behind them, and one member over the first.
fn two_chunks(first: (u64, u32), second: (u64, u32)) -> Vec<u8> {
    let mut kit = Kit::new();
    let mut stored = 16_u32.to_le_bytes().to_vec();
    stored.resize(200, 7);
    kit.put(&stored);
    for (file_offset, stored_len) in [first, second] {
        kit.push_chunk(ChunkRecord { file_offset, xxh3_decoded: 0, stored_len });
    }
    kit.index.add_blob(blob_record(16, 0, 1, 16, 0)).expect("one blob");
    kit.index.add_blob(blob_record(16, 1, 1, 16, 0)).expect("one blob");
    kit.member("m", &[extent(0, 16, 0, 0)], [0; 32]);
    kit.finish()
}

#[test]
fn chunks_are_identical_or_disjoint() {
    let _guard = shared();
    let refused = |first, second| {
        let error = open(two_chunks(first, second), &options()).expect_err("overlap");
        assert!(matches!(error, Error::MalformedIndex { section: "CHNK", .. }), "{error:?}");
    };
    // Partial overlap, each way round.
    refused((64, 20), (70, 20));
    refused((70, 20), (64, 20));
    // One inside the other, at the same start and strictly inside.
    refused((64, 20), (64, 19));
    refused((64, 20), (66, 8));
    // Sharing one byte.
    refused((64, 20), (83, 20));
    // Into the header, and into the index.
    refused((63, 20), (100, 20));
    refused((64, 20), (64 + 195, 20));
    refused((64, 20), (u64::MAX - 7, 20));

    let touching = open(two_chunks((64, 20), (84, 20)), &options());
    touching.expect("adjacent chunks are disjoint");
    let identical = open(two_chunks((64, 20), (64, 20)), &options()).expect("identical is allowed");
    let member = identical.member("m").expect("listed");
    member.read_at(0, &mut [0; 16]).expect_err("the hash is 0, so reading it is refused");
}

/// A valid file of 1 MiB of noise in chunks of 4 KiB.
fn large_file() -> Vec<u8> {
    let mut rng = Rng::new(21);
    let content = rng.bytes(MIB);
    let mut kit = Kit::new();
    let mut first_chunk = None;
    for raw in content.chunks(4096) {
        let record = kit.encoded(raw, StreamPolicy::Store);
        let id = kit.index.add_chunk(record);
        first_chunk.get_or_insert(id);
    }
    let blob = blob_record(MIB as u64, first_chunk.unwrap_or(0), 256, 4096, 0);
    let blob = kit.index.add_blob(blob).expect("one blob");
    kit.member("big", &[extent(0, MIB as u64, blob, 0)], *blake3::hash(&content).as_bytes());
    kit.finish()
}

/// Whether `code` is a refusal of a damaged file at open.
fn refusal(code: ErrorCode) -> bool {
    matches!(code, ErrorCode::NotZiptensors | ErrorCode::MalformedIndex | ErrorCode::BudgetExceeded | ErrorCode::Zstd)
}

#[test]
fn truncation_at_every_4_kib_is_a_typed_error() {
    let _guard = shared();
    let file = large_file();
    assert!(file.len() > 256 * 4096);
    for len in (0..file.len()).step_by(4096).chain([file.len() - 1, file.len() - 63]) {
        let code = code_of(open(file[..len].to_vec(), &options()));
        assert!(refusal(code), "cut at {len}: {code:?}");
    }
    let mut longer = file.clone();
    longer.push(0);
    assert!(refusal(code_of(open(longer, &options()))));
    open(file, &options()).expect("the whole file is valid");
}

/// What a sweep over mutated files found.
#[derive(Debug, Default)]
struct Tally {
    /// `Container` refused the file.
    refused_at_open: u32,
    /// The file opened, and a member failed `verify`.
    refused_at_verify: u32,
    /// All members verified to the bytes of the original.
    intact: u32,
}

/// The promise of the format, for one mutated file: any member that `verify` passes has the bytes of
/// the original member with that hash, under either verify setting. A hash no original has means
/// `verify` must fail. `restore_at`, if set, is the path for the same check of `restore_file`.
fn judge(file: Vec<u8>, sample: &Sample, verify: Verify, restore_at: Option<&Scratch>, tally: &mut Tally) {
    let originals: HashMap<blake3::Hash, &[u8]> = sample
        .members
        .iter()
        .map(|(_, content)| (blake3::hash(content), content.as_slice()))
        .collect();
    let container = match open(file, &OpenOptions { verify, ..options() }) {
        Ok(container) => container,
        Err(error) => {
            assert!(
                matches!(
                    error.code(),
                    ErrorCode::NotZiptensors
                        | ErrorCode::UnsupportedVersion
                        | ErrorCode::UnsupportedFeature
                        | ErrorCode::UnknownCodec
                        | ErrorCode::MalformedIndex
                        | ErrorCode::BudgetExceeded
                        | ErrorCode::Zstd
                ),
                "{error:?}"
            );
            tally.refused_at_open += 1;
            return;
        }
    };
    let mut all_passed = true;
    for info in container.members().to_vec() {
        let member = container.member(&info.name).expect("a listed member");
        let expected = originals.get(&blake3::Hash::from_bytes(member.blake3()));
        match member.verify() {
            Ok(()) => {
                let expected = expected.unwrap_or_else(|| {
                    panic!("{}: verify passed a hash no original has", info.name)
                });
                let mut read = vec![0; usize::try_from(member.len()).expect("a small member")];
                member.read_at(0, &mut read).expect("verify read this");
                assert_eq!(read, *expected, "{}: verify passed wrong bytes", info.name);
            }
            Err(error) => {
                all_passed = false;
                let allowed = [
                    ErrorCode::ChunkCorrupt,
                    ErrorCode::ChecksumMismatch,
                    ErrorCode::UnknownCodec,
                    ErrorCode::Zstd,
                    ErrorCode::VerifyFailed,
                ];
                assert!(allowed.contains(&error.code()), "{}: {error:?}", info.name);
            }
        }
        if let Some(scratch) = restore_at {
            let out = scratch.0.join("out");
            match member.restore_file(&out) {
                Ok(_) => {
                    let restored = fs::read(&out).expect("a committed file");
                    let expected = expected.expect("restore proved a hash, so it is an original");
                    assert_eq!(restored, *expected, "{}: restored wrong bytes", info.name);
                    fs::remove_file(&out).expect("a committed file");
                }
                Err(_) => scratch.assert_restore_left_nothing("out", "a failed restore left a file"),
            }
        }
    }
    if all_passed {
        tally.intact += 1;
    } else {
        tally.refused_at_verify += 1;
    }
}

#[test]
fn a_bit_flip_in_the_header_or_footer_is_always_refused() {
    let _guard = shared();
    let sample = sample(31);
    let file = &sample.file;
    let footer = file.len() - 64;
    for at in (0..64).chain(footer..file.len()) {
        for bit in 0..8 {
            let mut broken = file.clone();
            broken[at] ^= 1 << bit;
            let code = code_of(open(broken, &options()));
            assert!(
                matches!(
                    code,
                    ErrorCode::NotZiptensors
                        | ErrorCode::UnsupportedVersion
                        | ErrorCode::MalformedIndex
                ),
                "byte {at} bit {bit}: {code:?}"
            );
        }
    }
}

#[test]
fn a_bit_flip_in_the_stored_index_never_gives_wrong_bytes() {
    let _guard = shared();
    let sample = sample(32);
    let (data, _) = split_file(&sample.file);
    let start = 64 + data.len();
    let end = sample.file.len() - 64;
    let mut tally = Tally::default();
    for at in start..end {
        let mut broken = sample.file.clone();
        broken[at] ^= 1 << (at % 8);
        judge(broken, &sample, Verify::Chunk, None, &mut tally);
    }
    eprintln!("stored index flips: {tally:?}");
    assert!(tally.refused_at_open > 0);
}

#[test]
fn a_bit_flip_in_the_data_region_never_gives_wrong_bytes() {
    let _guard = shared();
    let sample = sample(33);
    let (data, _) = split_file(&sample.file);
    let mut chunk_checked = Tally::default();
    let mut unchecked = Tally::default();
    for at in (64..64 + data.len()).step_by(3) {
        let mut broken = sample.file.clone();
        broken[at] ^= 1 << (at % 8);
        judge(broken.clone(), &sample, Verify::Chunk, None, &mut chunk_checked);
        judge(broken, &sample, Verify::Off, None, &mut unchecked);
    }
    eprintln!("data flips: chunk {chunk_checked:?}, off {unchecked:?}");
    assert_eq!(chunk_checked.refused_at_open, 0);
    assert!(chunk_checked.refused_at_verify > 0);
    assert!(unchecked.refused_at_verify > 0, "BLAKE3 also holds with chunk hashes off");
}

/// Flips a bit of every byte of each section of the decompressed index and assembles a file with
/// valid hashes around it, which is what a forger, not a disk, does.
#[test]
fn a_bit_flip_in_every_index_section_with_hashes_fixed_up_never_gives_wrong_bytes() {
    let _guard = shared();
    let sample = sample(34);
    let scratch = Scratch::new();
    let (data, raw) = split_file(&sample.file);
    let table = 0..TABLE_START + TAGS.len() * ENTRY_BYTES;
    let sections = TAGS.iter().map(|tag| (std::str::from_utf8(*tag).expect("ascii"), body_of(&raw, tag)));
    let mut total = Tally::default();
    for (name, range) in std::iter::once(("table", table)).chain(sections) {
        let mut tally = Tally::default();
        for at in range {
            let mut flipped = raw.clone();
            flipped[at] ^= 1 << (at % 8);
            let restore = (at % 16 == 0).then_some(&scratch);
            let verify = if at % 2 == 0 { Verify::Chunk } else { Verify::Off };
            judge(assemble(&data, &flipped), &sample, verify, restore, &mut tally);
        }
        eprintln!("{name} flips: {tally:?}");
        total.refused_at_open += tally.refused_at_open;
        total.refused_at_verify += tally.refused_at_verify;
        total.intact += tally.intact;
    }
    assert!(total.refused_at_open > 0, "{total:?}");
    assert!(total.refused_at_verify > 0, "{total:?}");
    scratch.assert_restore_left_nothing("out", "a restore left a file");
}

/// `Verify::Off` drops the chunk hashes and keeps every length check. So `read_at` and its relatives
/// return whatever the stored bytes decode to, and a flipped stored byte is a wrong byte in the
/// caller's buffer with `Ok` as the result. `verify` and `restore_file` hash the whole member under
/// any setting, so they alone still protect.
#[test]
fn verify_off_does_not_protect_reads_and_restore_still_refuses() {
    let _guard = shared();
    let content: Vec<u8> = (0..=255).collect();
    let mut file = stored_file(&content);
    file[64 + 4 + 10] ^= 1;
    let off = OpenOptions { verify: Verify::Off, ..options() };
    let scratch = Scratch::new();
    let out = scratch.0.join("out");

    let container = open(file.clone(), &off).expect("the index is intact");
    let member = container.single().expect("one member");
    let mut read = vec![0; 256];
    member.read_at(0, &mut read).expect("Verify::Off reads without a hash");
    assert_eq!(read[10], content[10] ^ 1, "read_at gave the flipped byte, and Ok");
    read.fill(0);
    let (head, tail) = read.split_at_mut(100);
    let mut requests = [ReadRequest { offset: 0, dst: head }, ReadRequest { offset: 100, dst: tail }];
    member.read_many(&mut requests).expect("Verify::Off reads without a hash");
    assert_eq!(read[10], content[10] ^ 1, "read_many gave the flipped byte, and Ok");
    assert_eq!(code_of(member.verify()), ErrorCode::VerifyFailed);
    assert_eq!(code_of(member.restore_file(&out)), ErrorCode::VerifyFailed);
    assert_eq!(scratch.entries(), 0, "nothing was committed");

    let container = open(file, &options()).expect("the index is intact");
    let member = container.single().expect("one member");
    assert_eq!(code_of(member.read_at(0, &mut read)), ErrorCode::ChecksumMismatch);
}

/// Replaces `file`'s index by `edit` of it, with hashes valid again. What any writer of bytes can do.
fn forge(file: &[u8], edit: impl FnOnce(&mut [u8])) -> Vec<u8> {
    let (data, mut raw) = split_file(file);
    edit(&mut raw);
    assemble(&data, &raw)
}

/// XXH3 protects against accidents and is no keyed hash. Someone who edits stored bytes and rewrites
/// `CHNK.xxh3_decoded` gets past `Verify::Chunk`: `read_at` has no other check, so only `verify` and
/// `restore_file`, through BLAKE3, refuse.
#[test]
fn chunk_hashes_do_not_stop_a_forger_and_the_member_hash_does() {
    let _guard = shared();
    let content: Vec<u8> = (0..=255).collect();
    let mut forged_content = content.clone();
    forged_content[10] ^= 1;
    let mut file = stored_file(&content);
    file[64 + 4 + 10] ^= 1;
    let forged = forge(&file, |raw| {
        let chunks = body_of(raw, b"CHNK");
        let hash = xxhash_rust::xxh3::xxh3_64(&forged_content);
        raw[chunks.start + 8..chunks.start + 16].copy_from_slice(&hash.to_le_bytes());
    });
    let container = open(forged, &options()).expect("every hash in the file holds");
    let member = container.single().expect("one member");
    let mut read = vec![0; 256];
    member.read_at(0, &mut read).expect("Verify::Chunk passes a forged chunk hash");
    assert_eq!(read, forged_content);
    assert_eq!(code_of(member.verify()), ErrorCode::VerifyFailed);
}

/// BLAKE3 in the file is unkeyed and lives next to what it covers, so it proves integrity against
/// corruption and against an edit that leaves it alone, never authenticity. An attacker who also
/// rewrites `MEMB.blake3` has made a different valid file, and `verify` passes it. Trust in a file's
/// content needs a hash from outside that file.
#[test]
fn the_member_hash_is_not_authentication() {
    let _guard = shared();
    let content: Vec<u8> = (0..=255).collect();
    let mut forged_content = content.clone();
    forged_content[10] ^= 1;
    let mut file = stored_file(&content);
    file[64 + 4 + 10] ^= 1;
    let forged = forge(&file, |raw| {
        let chunks = body_of(raw, b"CHNK");
        let hash = xxhash_rust::xxh3::xxh3_64(&forged_content);
        raw[chunks.start + 8..chunks.start + 16].copy_from_slice(&hash.to_le_bytes());
        let members = body_of(raw, b"MEMB");
        let hash = blake3::hash(&forged_content);
        raw[members.start + 16..members.start + 48].copy_from_slice(hash.as_bytes());
    });
    let container = open(forged, &options()).expect("every hash in the file holds");
    let member = container.single().expect("one member");
    member.verify().expect("a complete forgery is a valid file");
    assert_eq!(member.blake3(), *blake3::hash(&forged_content).as_bytes());
    assert_ne!(member.blake3(), *blake3::hash(&content).as_bytes(), "the outside hash still differs");
}

/// A tensor table row is metadata: an edit of it moves what `decode_tensor_into` returns, inside the
/// member, and changes nothing a restore produces. Spec invariant 8.
#[test]
fn a_lying_tensor_row_cannot_change_restored_bytes() {
    let _guard = shared();
    let sample = sample(35);
    let forged = forge(&sample.file, |raw| {
        let tensors = body_of(raw, b"TENS");
        raw[tensors.start + 8..tensors.start + 16].copy_from_slice(&0_u64.to_le_bytes());
    });
    let mut tally = Tally::default();
    judge(forged, &sample, Verify::Chunk, None, &mut tally);
    assert_eq!(tally.intact, 1, "{tally:?}");
}

#[test]
fn oversized_shapes_are_refused_or_stay_metadata() {
    let _guard = shared();
    let tensor_file = |edit: &dyn Fn(&mut TensorRecord)| {
        let mut kit = Kit::new();
        let blob = kit.one_chunk_blob(&[1; 64], 64, StreamPolicy::Store);
        let shape_off = kit.index.add_shape(&[u64::MAX; 16]).expect("16 dimensions");
        let name = kit.index.add_string("t").expect("a short name");
        let mut tensor =
            TensorRecord { name, virt_offset: 0, virt_len: 64, shape_off, ndim: 16, dtype: Dtype::F64 };
        edit(&mut tensor);
        kit.index.add_tensor(tensor).expect("one tensor");
        kit.push_extent(extent(0, 64, blob, 0));
        let name = kit.index.add_string("m").expect("a short name");
        let member = MemberRecord { tensor_count: 1, ..member_record(name, 64, [0; 32], 0, 1) };
        kit.index.add_member(member).expect("one member");
        kit.finish()
    };
    let refused = |edit: &dyn Fn(&mut TensorRecord)| {
        assert_eq!(code_of(open(tensor_file(edit), &options())), ErrorCode::MalformedIndex);
    };
    refused(&|tensor| tensor.ndim = 17);
    refused(&|tensor| tensor.ndim = u16::MAX);
    refused(&|tensor| tensor.shape_off = 1);
    refused(&|tensor| tensor.shape_off = u32::MAX);
    refused(&|tensor| tensor.virt_len = 65);
    refused(&|tensor| tensor.virt_offset = u64::MAX);
    refused(&|tensor| tensor.virt_len = u64::MAX);

    // Dimensions are hints and their product is not checked: the row opens, costs its 16 words, and
    // a read goes by the byte range alone.
    let container = open(tensor_file(&|_| {}), &options()).expect("a dtype is not checked against a range");
    let member = container.single().expect("one member");
    assert_eq!(member.tensors()[0].shape, [u64::MAX; 16]);
    let error = member.decode_tensor_into("t", &mut [0; 8]).expect_err("the range is 64 bytes");
    assert!(matches!(error, Error::DestinationSize { expected: 64, actual: 8 }), "{error:?}");
}

/// A file with one empty member called `name`.
fn named(name: &str) -> Vec<u8> {
    let mut kit = Kit::new();
    kit.member(name, &[], *blake3::hash(b"").as_bytes());
    kit.finish()
}

#[test]
fn a_dotdot_member_opens_and_is_refused_before_a_path_is_made() {
    let _guard = shared();
    for name in [
        "..",
        "../up",
        "a/../../b",
        "a/..",
        "..\\up",
        "a\\..\\b",
        "/etc/passwd",
        "\\server\\share",
        "C:evil",
        "a:stream",
        "",
        "a\0b",
    ] {
        // A name is checked when it is about to become a path. Reading by name stays safe.
        let container = open(named(name), &options()).expect("a name is a plain string to a reader");
        let info = &container.members()[0];
        let error = check_restore_name(&info.name).expect_err(name);
        assert!(matches!(error, Error::UnsafeMemberName { .. }), "{name:?}: {error:?}");
    }
    for name in ["a", "dir/sub/file.bin", "..hidden", "a..b", "...", "dir/..x/y"] {
        let container = open(named(name), &options()).expect("a safe name");
        check_restore_name(&container.members()[0].name).expect(name);
    }
}

#[test]
fn duplicate_names_are_refused() {
    let _guard = shared();
    let mut kit = Kit::new();
    kit.member("twin", &[], [0; 32]);
    kit.member("twin", &[], [1; 32]);
    let error = open(kit.finish(), &options()).expect_err("duplicate members");
    assert!(matches!(error, Error::MalformedIndex { section: "MEMB", .. }), "{error:?}");

    let tensor_kit = |names: [&str; 2]| {
        let mut kit = Kit::new();
        let blob = kit.one_chunk_blob(&[1; 64], 64, StreamPolicy::Store);
        for name in names {
            let name = kit.index.add_string(name).expect("a short name");
            let tensor =
                TensorRecord { name, virt_offset: 0, virt_len: 1, shape_off: 0, ndim: 0, dtype: Dtype::U8 };
            kit.index.add_tensor(tensor).expect("two tensors");
        }
        kit.push_extent(extent(0, 64, blob, 0));
        let name = kit.index.add_string("m").expect("a short name");
        let member = MemberRecord { tensor_count: 2, ..member_record(name, 64, [0; 32], 0, 1) };
        kit.index.add_member(member).expect("one member");
        kit.finish()
    };
    let error = open(tensor_kit(["t", "t"]), &options()).expect_err("duplicate tensors");
    assert!(matches!(error, Error::MalformedIndex { section: "MEMB", .. }), "{error:?}");
    open(tensor_kit(["t", "u"]), &options()).expect("distinct tensors");
}

/// A file whose blob 0 goes through the chain `steps`, as raw records.
fn chain_file(steps: &[RawStep]) -> Vec<u8> {
    let mut kit = Kit::default();
    let first_step = kit.index.add_raw_step(steps[0]).expect("few steps");
    for step in &steps[1..] {
        kit.index.add_raw_step(*step).expect("few steps");
    }
    let step_count = u16::try_from(steps.len()).expect("few steps");
    kit.index.add_raw_chain(ChainRecord { first_step, step_count }).expect("one chain");
    kit.finish()
}

#[test]
fn unknown_ids_are_refused() {
    let _guard = shared();
    let layout = RawStep { codec_id: 0x0100, param0: 0, param1: 0 };
    for codec_id in [0x0000, 0x0301, 0x0401, 0x04FF, 0x0500, 0x05FF, 0x0999, 0x8000, 0xFFFF] {
        let unknown = RawStep { codec_id, param0: 0, param1: 0 };
        assert_eq!(code_of(open(chain_file(&[unknown]), &options())), ErrorCode::UnknownCodec, "{codec_id:#x}");
        assert_eq!(code_of(open(chain_file(&[unknown, layout]), &options())), ErrorCode::UnknownCodec, "{codec_id:#x}");
    }
    open(chain_file(&[layout]), &options()).expect("a known chain, unreferenced");

    let mut kit = Kit::new();
    let blob = kit.one_chunk_blob(&[1; 64], 64, StreamPolicy::Store);
    kit.member("m", &[ExtentRecord { view_id: 0x0303, ..extent(0, 64, blob, 0) }], [0; 32]);
    assert_eq!(code_of(open(kit.finish(), &options())), ErrorCode::UnknownCodec);

    let mut kit = Kit::new();
    let record = BlobRecord { kind: 1, ..blob_record(0, 0, 0, 1, 0) };
    kit.index.add_blob(record).expect("one blob");
    let error = open(kit.finish(), &options()).expect_err("kind 1 is delta");
    assert!(matches!(error, Error::MalformedIndex { section: "BLOB", .. }), "{error:?}");

    for codec in [2_u8, 3, 0x7F, 0xFF] {
        let word = (u32::from(codec) << 24) | 16;
        let stored = [word.to_le_bytes().as_slice(), &[0; 16]].concat();
        let container = open(hostile_chunk_file(&stored, 16), &options()).expect("a valid index");
        let member = container.single().expect("one member");
        let error = member.read_at(0, &mut [0; 16]).expect_err("an unknown stream codec");
        assert!(matches!(error, Error::UnknownCodec { id } if id == u16::from(codec)), "{error:?}");
    }

    let header = Header::current().to_bytes();
    for bit in 0..64 {
        let mut bytes = header;
        bytes[16..24].copy_from_slice(&(1_u64 << bit).to_le_bytes());
        let hash = xxhash_rust::xxh3::xxh3_64(&bytes[..56]);
        bytes[56..64].copy_from_slice(&hash.to_le_bytes());
        let file = [&bytes[..], &named("m")[64..]].concat();
        // Bit 0, delta blobs, is implemented: it is refused because this file has none.
        let expected =
            if bit == 0 { ErrorCode::MalformedIndex } else { ErrorCode::UnsupportedFeature };
        assert_eq!(code_of(open(file, &options())), expected, "bit {bit}");
    }
}

/// Data offset where the chunks of the delta blobs of [`delta_sample`] start: the base blob's four
/// chunks are written first, so this is the file offset of `CHNK` record 4.
fn delta_chunks_start(raw: &[u8]) -> usize {
    let chunks = body_of(raw, b"CHNK");
    usize::try_from(read_u64(raw, chunks.start + 4 * 24)).expect("a small file")
}

/// A flipped bit in a base chunk or in a delta chunk, under either verify setting: a checked read
/// refuses it at the chunk, an unchecked one passes it through the combination, and `verify` refuses
/// it in both, so no member that verifies has wrong bytes.
#[test]
fn a_bit_flip_in_a_base_or_delta_chunk_never_gives_wrong_bytes() {
    let _guard = shared();
    let sample = delta_sample(36);
    let (data, raw) = split_file(&sample.file);
    let deltas_at = delta_chunks_start(&raw);
    assert!((64..64 + data.len()).contains(&deltas_at), "{deltas_at}");
    for (name, range) in [("base", 64..deltas_at), ("delta", deltas_at..64 + data.len())] {
        let mut chunk_checked = Tally::default();
        let mut unchecked = Tally::default();
        for at in range {
            let mut broken = sample.file.clone();
            broken[at] ^= 1 << (at % 8);
            judge(broken.clone(), &sample, Verify::Chunk, None, &mut chunk_checked);
            judge(broken, &sample, Verify::Off, None, &mut unchecked);
        }
        eprintln!("{name} chunk flips: chunk {chunk_checked:?}, off {unchecked:?}");
        assert_eq!(chunk_checked.refused_at_open, 0, "{name}");
        assert!(chunk_checked.refused_at_verify > 0, "{name}");
        assert!(unchecked.refused_at_verify > 0, "{name}: BLAKE3 also holds with chunk hashes off");
    }
}

/// Flips every bit of `PARM`, which holds the base references, and of `BLOB`, which holds the kinds
/// and the `PARM` entries, and one bit of every other byte of the index, with valid hashes around
/// it. Whatever validation lets through must still give no wrong bytes.
#[test]
fn a_bit_flip_in_the_index_of_a_delta_container_never_gives_wrong_bytes() {
    let _guard = shared();
    let sample = delta_sample(37);
    let scratch = Scratch::new();
    let (data, raw) = split_file(&sample.file);
    let tags = TAGS.iter().copied().chain([b"PARM"]);
    let sections: Vec<(&[u8; 4], std::ops::Range<usize>)> = tags.map(|tag| (tag, body_of(&raw, tag))).collect();
    let table = (b"tabl", 0..TABLE_START + sections.len() * ENTRY_BYTES);
    let mut total = Tally::default();
    for (tag, range) in std::iter::once(table).chain(sections) {
        let every_bit = tag == b"PARM" || tag == b"BLOB";
        let mut tally = Tally::default();
        for at in range {
            let bits = if every_bit { 0..8 } else { at % 8..at % 8 + 1 };
            for bit in bits {
                let mut flipped = raw.clone();
                flipped[at] ^= 1 << bit;
                let restore = (at % 16 == 0 && bit == at % 8).then_some(&scratch);
                let verify = if (at + bit) % 2 == 0 { Verify::Chunk } else { Verify::Off };
                judge(assemble_with(&data, &flipped, FEATURE_DELTA), &sample, verify, restore, &mut tally);
            }
        }
        eprintln!("{} flips: {tally:?}", String::from_utf8_lossy(tag));
        total.refused_at_open += tally.refused_at_open;
        total.refused_at_verify += tally.refused_at_verify;
        total.intact += tally.intact;
    }
    assert!(total.refused_at_open > 0, "{total:?}");
    assert!(total.refused_at_verify > 0, "{total:?}");
    scratch.assert_restore_left_nothing("out", "a restore left a file");
}

#[test]
fn a_mutated_delta_container_is_refused_or_gives_the_original_bytes() {
    let _guard = shared();
    let sample = delta_sample(43);
    let scratch = Scratch::new();
    let (data, raw) = split_file(&sample.file);
    let mut rng = Prng(44);
    let mut tally = Tally::default();
    for round in 0..3000 {
        let (mut new_data, mut new_raw) = (data.clone(), raw.clone());
        harness::mutate(&mut rng, if round % 2 == 0 { &mut new_raw } else { &mut new_data });
        let verify = if round % 3 == 0 { Verify::Off } else { Verify::Chunk };
        let restore = (round % 64 == 0).then_some(&scratch);
        judge(assemble_with(&new_data, &new_raw, FEATURE_DELTA), &sample, verify, restore, &mut tally);
    }
    eprintln!("mutated delta files: {tally:?}");
    assert!(tally.refused_at_open > 0 && tally.refused_at_verify > 0, "{tally:?}");
}

#[test]
fn truncating_a_delta_container_is_a_typed_error() {
    let _guard = shared();
    let file = delta_sample(45).file;
    for len in (0..file.len()).step_by(61).chain([file.len() - 1]) {
        let code = code_of(open(file[..len].to_vec(), &options()));
        assert!(refusal(code), "cut at {len}: {code:?}");
    }
}

/// A base of one 8 MiB chunk of zeros and `count` delta blobs of 2 bytes in chunks of 8 MiB over it,
/// sharing one stored residual and one `PARM` entry, and one member of one 2-byte extent per delta
/// blob. Each extent decodes 2 bytes and the whole base chunk. Returns the file and the content.
fn many_deltas_over_one_base(count: u32) -> (Vec<u8>, Vec<u8>) {
    let mut kit = Kit::new();
    let base = kit.one_chunk_blob(&vec![0; MAX_CHUNK as usize], MAX_CHUNK, StreamPolicy::Entropy);
    let chain = Chain::new(&[Step::Delta(DeltaOp::Xor(ElementWidth::Two)), Step::Layout(ChunkLayout::Whole)]);
    let chain_id = kit.index.add_chain(chain.expect("a valid chain")).expect("two chains");
    // Over a zero base the target is the residual itself.
    let target = [0x5A, 0xA5];
    let file_offset = kit.put(&[&[2, 0, 0, 0], target.as_slice()].concat());
    let xxh3_decoded = xxhash_rust::xxh3::xxh3_64(&target);
    let chunk = kit.index.add_chunk(ChunkRecord { file_offset, xxh3_decoded, stored_len: 6 });
    let params = DeltaParams { base_blob: base, base_offset: 0 }.encode();
    let (params_off, params_len) = kit.index.add_params(&params).expect("one entry");
    let mut extents = Vec::new();
    for at in 0..u64::from(count) {
        let record = BlobRecord { kind: BLOB_KIND_DELTA, params_off, params_len, ..blob_record(2, chunk, 1, MAX_CHUNK, chain_id) };
        let blob = kit.index.add_blob(record).expect("few blobs");
        extents.push(extent(2 * at, 2, blob, 0));
    }
    let content = target.repeat(count as usize);
    kit.member("deltas", &extents, *blake3::hash(&content).as_bytes());
    (assemble_with(&kit.data, &kit.index.serialize(), FEATURE_DELTA), content)
}

/// Decode work counts the base chunk once per delta extent (spec section 4), so delta blobs over one
/// base cannot multiply its decode past the budget: 32 extents of 2 bytes over 8 MiB base chunks are
/// the floor of 256 MiB plus 64 bytes, inside the floor plus 16 bytes per member byte; 33 are not, and
/// 2^16 are refused before any data read, at exactly what they declare.
#[test]
fn many_delta_blobs_over_one_base_pay_for_the_base_each_time() {
    let _guard = shared();
    let per_extent = 2 + u64::from(MAX_CHUNK);
    let (file, content) = many_deltas_over_one_base(32);
    let container = open(file, &options()).expect("at the floor");
    let member = container.single().expect("one member");
    let mut read = vec![0; content.len()];
    member.read_at(0, &mut read).expect("decodes the base once per extent");
    assert_eq!(read, content);
    member.verify().expect("the content of every delta blob");

    let error = open(many_deltas_over_one_base(33).0, &options()).expect_err("over the floor");
    let Error::BudgetExceeded { what: "decode work", declared, limit } = error else {
        panic!("{error:?}");
    };
    assert_eq!((declared, limit), (33 * per_extent, 16 * 66 + (256 << 20)));

    let (file, _) = many_deltas_over_one_base(1 << 16);
    let index_offset = split_file(&file).0.len() as u64 + HEADER_LEN;
    let source = Traced::new(file);
    let error = Container::from_source(source.clone(), &options()).expect_err("too much decode work");
    assert!(
        matches!(error, Error::BudgetExceeded { what: "decode work", declared, .. } if declared == (1 << 16) * per_extent),
        "{error:?}"
    );
    assert_eq!(source.data_reads(index_offset), 0);
}

#[test]
fn a_mutated_file_is_refused_or_gives_the_original_bytes() {
    let _guard = shared();
    let sample = sample(41);
    let scratch = Scratch::new();
    let (data, raw) = split_file(&sample.file);
    let mut rng = Prng(42);
    let mut tally = Tally::default();
    for round in 0..4000 {
        // Mutate the index or the data region, then fix the hashes a forger can fix.
        let (mut new_data, mut new_raw) = (data.clone(), raw.clone());
        harness::mutate(&mut rng, if round % 2 == 0 { &mut new_raw } else { &mut new_data });
        let verify = if round % 3 == 0 { Verify::Off } else { Verify::Chunk };
        let restore = (round % 64 == 0).then_some(&scratch);
        judge(assemble(&new_data, &new_raw), &sample, verify, restore, &mut tally);
    }
    eprintln!("mutated files: {tally:?}");
    assert!(tally.refused_at_open > 0 && tally.refused_at_verify > 0, "{tally:?}");
}

#[test]
fn the_fuzz_target_open_and_read_survives_the_seeded_mutator_on_file_bytes() {
    let _guard = shared();
    let mut rng = Prng(51);
    for seed in 0..6 {
        let file = if seed < 4 { sample(seed).file } else { delta_sample(seed).file };
        harness::open_and_read(&[&[0], file.as_slice()].concat());
        for _ in 0..500 {
            let mut input = [&[0], file.as_slice()].concat();
            harness::mutate(&mut rng, &mut input);
            // The mutator may cut the input to nothing.
            if let Some(first) = input.first_mut() {
                *first = 0;
            }
            harness::open_and_read(&input);
        }
    }
}

#[test]
fn the_fuzz_target_open_and_read_survives_the_seeded_mutator_past_the_hashes() {
    let _guard = shared();
    let mut rng = Prng(52);
    for seed in 0..6 {
        let file = if seed < 4 { sample(seed).file } else { delta_sample(seed).file };
        let valid = harness::to_fixed_up_input(&file);
        harness::open_and_read(&valid);
        let form = valid[0];
        assert_eq!(form, if seed < 4 { 1 } else { 3 }, "the delta form sets feature bit 0");
        for _ in 0..1000 {
            let mut input = valid.clone();
            harness::mutate(&mut rng, &mut input);
            // The mutator may cut the input to nothing.
            if let Some(first) = input.first_mut() {
                *first = form;
            }
            harness::open_and_read(&input);
        }
    }
}

#[test]
fn the_fuzz_target_decode_chunk_survives_the_seeded_mutator() {
    let _guard = shared();
    let mut rng = Prng(53);
    let mut body = vec![0; 2048];
    for (at, byte) in body.iter_mut().enumerate() {
        *byte = if at % 4 == 3 { 0 } else { rng.next_u64().to_le_bytes()[0] };
    }
    let one_plane = harness::one_plane_body(55, 2048, 4);
    let mut stored = Vec::new();
    let chain = Chain::new(&[Step::Layout(ChunkLayout::Planes(ElementWidth::Four))]).expect("valid");
    let policy = harness::POLICIES[2];
    encode_chunk(&one_plane, &chain, &policy, &mut Workspace::new(), &mut stored).expect("encodes");
    // The codec of a stream word is its top byte, and planes(4) has four words.
    let codecs: Vec<u8> = (0..4).map(|word| stored[word * 4 + 3]).collect();
    assert_eq!(codecs.iter().filter(|codec| **codec == 1).count(), 1, "the in-place route: {codecs:?}");
    // Chain 8 is `planes(4)`; chain 40 is it after `delta_xor(8)`, and form bit 4 makes its base
    // zero, so the residual is `one_plane` and takes the in-place route too.
    for round in 0..500 {
        let (chain, form) = if round % 2 == 0 { (8, 0b1010) } else { (40, 0b1_1010) };
        let mut input = [&[chain, form, 0xFF, 0x00], one_plane.as_slice()].concat();
        harness::mutate(&mut rng, &mut input);
        harness::decode_chunk(&input);
    }
    // Chains from `PLAIN_CHAINS` on have a delta step; odd ones get the zero base of form bit 4.
    // Half the mutations of the plain-only table keep the run as short as it was.
    let chains = u8::try_from(2 * harness::PLAIN_CHAINS).expect("few chains");
    for chain in 0..chains {
        for form in 0..16 {
            let form = form | ((chain & 1) << 4);
            let input = [&[chain, form, 0xFF, 0x00], body.as_slice()].concat();
            harness::decode_chunk(&input);
            for _ in 0..4 {
                let mut mutated = input.clone();
                harness::mutate(&mut rng, &mut mutated);
                harness::decode_chunk(&mutated);
            }
        }
    }
}

#[test]
fn the_fuzz_target_profile_scan_survives_the_seeded_mutator() {
    let _guard = shared();
    let specs = [TensorSpec::new("a", &[2048], Elem::F32), TensorSpec::new("b", &[512], Elem::F16)];
    let onnx = OnnxModel { initializers: vec![OnnxTensor::raw("w", &[512], Elem::F32)], ..OnnxModel::default() };
    let inputs = [
        fixtures::safetensors(1, &specs).bytes,
        fixtures::gguf(2, &specs, &GgufOptions::default()).bytes,
        fixtures::onnx(3, &onnx).bytes,
        fixtures::pt_zip(4, &specs, &PtZipOptions::default()).bytes,
    ];
    let mut rng = Prng(54);
    for input in inputs {
        harness::profile_scan(&input);
        for _ in 0..1500 {
            let mut mutated = input.clone();
            // The bytes that matter sit at the ends: a header, or a ZIP directory.
            let keep = if rng.below(2) == 0 { mutated.len().min(2048) } else { mutated.len() };
            let mut tail = mutated.split_off(keep);
            harness::mutate(&mut rng, &mut mutated);
            mutated.append(&mut tail);
            harness::profile_scan(&mutated);
        }
    }
}

/// An `F32` file that is exact as `F16`, with its last element made inexact if `late_mismatch`.
fn widening_file(tensor_bytes: u64, late_mismatch: bool) -> Vec<u8> {
    let elements = tensor_bytes / 4;
    let specs = [TensorSpec::new("wide", &[elements], Elem::F32)];
    let mut fixture = fixtures::safetensors(3, &specs);
    let mut rng = Rng::new(4);
    let narrow_len = usize::try_from(elements).expect("fits") * 2;
    let narrow = fixtures::payload(&mut rng, Elem::F16, narrow_len);
    let tensor = &fixture.tensors[0];
    let start = usize::try_from(tensor.offset).expect("fits");
    let end = start + usize::try_from(tensor.len).expect("fits");
    widen_into(View::F16ToF32, &narrow, &mut fixture.bytes[start..end]).expect("sizes match");
    if late_mismatch {
        fixture.bytes[end - 4..end].copy_from_slice(&0x3F80_0001_u32.to_le_bytes());
    }
    fixture.bytes
}

/// Risk R7, the time half: a late mismatch compresses the tensor twice, plus one extra read. The
/// allocator half is asserted in `tests/write_memory.rs`.
#[test]
#[ignore = "measurement: encodes 256 MiB twice, run by hand with --release"]
fn measure_time_of_a_late_widening_mismatch() {
    let _guard = exclusive();
    let scratch = Scratch::new();
    let mut seconds = [0.0, 0.0];
    for (late, time) in seconds.iter_mut().enumerate() {
        let input = scratch.0.join(format!("input-{late}"));
        fs::write(&input, widening_file(256 << 20, late == 1)).expect("a scratch file");
        let output = scratch.0.join(format!("output-{late}"));
        let options = EncodeOptions { mode: Mode::Balanced, threads: 0 };
        let (report, peak) = measure(|| encode_file(&input, &output, &options).expect("encodes"));
        eprintln!("late mismatch {late}: restreams {}, {peak:?}", report.members[0].widen_restreams);
        *time = peak.seconds;
    }
    eprintln!("late mismatch: {:.2} times the exact tensor", seconds[1] / seconds[0]);
}
