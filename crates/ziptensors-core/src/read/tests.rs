//! File: read/tests.rs
//! Purpose: reader tests, on real containers built by `testkit`, against reference bytes.
//!
//! Implementation notes: all data is generated from seeds. Files go under the system temp directory
//! and are removed. The peak memory test needs its own allocator, so it is the integration test
//! `tests/read_memory.rs`.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use super::testkit::{
    ContainerBuilder, ExtentSpec, MemorySource, Rng, delta, finetune_of, finish_file, footer_of,
    identity, planes, rotated, tensor_like, whole,
};
use super::{Container, OpenOptions, ReadRequest};
use crate::chain::{Chain, ElementWidth, FieldTable, Layout, Step, View};
use crate::codec::StreamPolicy;
use crate::error::Error;
use crate::format::{Footer, Header, IndexBuilder};
use crate::io::RandomAccess;
use crate::types::Verify;

const KIB: usize = 1 << 10;
const KIB_U32: u32 = 1 << 10;

/// A small value as a length.
fn len_of(value: u64) -> usize {
    usize::try_from(value).expect("a test value fits a length")
}

fn options(threads: usize) -> OpenOptions {
    OpenOptions { threads, ..OpenOptions::default() }
}

fn open_with(file: Vec<u8>, options: &OpenOptions) -> Result<Container, Error> {
    Container::from_source(Arc::new(MemorySource::silent(file)), options)
}

fn open(file: Vec<u8>, threads: usize) -> Container {
    open_with(file, &options(threads)).expect("a valid container opens")
}

fn recording(file: Vec<u8>, threads: usize) -> (Container, Arc<MemorySource>) {
    let source = Arc::new(MemorySource::new(file));
    let shared: Arc<dyn crate::io::RandomAccess> = source.clone();
    let container = Container::from_source(shared, &options(threads)).expect("valid");
    source.take_reads();
    (container, source)
}

/// Scratch directory under the system temp directory, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let name = format!("ziptensors-read-{}-{id}", std::process::id());
        let path = std::env::temp_dir().join(name);
        fs::create_dir(&path).expect("scratch directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    /// Names in the directory, sorted.
    fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(&self.0)
            .expect("list scratch")
            .map(|entry| entry.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("cannot remove {}: {error}", self.0.display());
        }
    }
}

/// A member of three extents over two blobs, with its boundaries, and a tensor per extent.
fn three_extents() -> (Vec<u8>, Vec<u8>, Vec<u64>) {
    let mut builder = ContainerBuilder::new();
    let planar = tensor_like(1, 100 * KIB, 4);
    let plain = tensor_like(2, 40 * KIB + 123, 1);
    let a = builder.add_blob(
        planar,
        planes(ElementWidth::Four),
        16 * KIB_U32,
        StreamPolicy::EntropyOrStore,
    );
    let b = builder.add_blob(plain, whole(), 7001, StreamPolicy::Zstd { level: 3 });
    let extents = [
        identity(b, 5, 30_000),
        identity(a, 4096, 60_000),
        identity(b, 30_005, 7000),
    ];
    let tensors = [("t0", 0, 30_000), ("t1", 30_000, 60_000), ("empty", 90_000, 0)];
    let content = builder.add_member_with_tensors("model.bin", &extents, &tensors);
    let mut boundaries = vec![0, 30_000, 90_000, 97_000];
    // Chunk boundaries of `b` under the first extent, and of `a` under the second.
    boundaries.extend((7001..30_000_u64).step_by(7001).map(|at| at - 5));
    boundaries.extend((16..=60_u64).step_by(16).map(|kib| 30_000 + kib * 1024 - 4096));
    (builder.finish(), content, boundaries)
}

/// Asserts that one read returns the reference bytes.
fn assert_read(member: &super::Member, content: &[u8], offset: usize, len: usize) {
    let mut dst = vec![0xA5; len + 2];
    member.read_at(offset as u64, &mut dst[1..=len]).expect("read in range");
    assert_eq!(&dst[1..=len], &content[offset..offset + len], "offset {offset}, len {len}");
    assert_eq!((dst[0], dst[len + 1]), (0xA5, 0xA5), "a read left its destination");
}

#[test]
fn container_is_clone_send_sync() {
    fn assert_traits<T: Clone + Send + Sync>() {}
    assert_traits::<Container>();
    assert_traits::<super::Member>();
}

#[test]
fn a_valid_file_lists_its_members_and_tensors() {
    let (file, content, _) = three_extents();
    let container = open(file, 2);
    assert_eq!(container.members().len(), 1);
    let info = &container.members()[0];
    assert_eq!((info.name.as_str(), info.len), ("model.bin", content.len() as u64));
    assert_eq!(info.blake3, *blake3::hash(&content).as_bytes());
    let member = container.member("model.bin").expect("present");
    assert_eq!(member.len(), content.len() as u64);
    assert!(!member.is_empty());
    assert_eq!(member.name().expect("name"), "model.bin");
    let names: Vec<_> = member.tensors().iter().map(|tensor| tensor.name.as_str()).collect();
    assert_eq!(names, ["t0", "t1", "empty"]);
    assert_eq!(member.tensors()[1].shape, [60_000]);
    assert_eq!(container.single().expect("one member").len(), member.len());
}

#[test]
fn member_lookup_finds_every_name_and_reports_the_rest() {
    let mut builder = ContainerBuilder::new();
    let blob = builder.add_blob(tensor_like(3, 100, 1), whole(), 40, StreamPolicy::Store);
    let names = ["z.bin", "a/b.bin", "m.bin", "empty"];
    let extents = [identity(blob, 0, 10)];
    for name in &names[..3] {
        builder.add_member(name, &extents);
    }
    builder.add_member(names[3], &[]);
    let container = open(builder.finish(), 1);
    for name in names {
        assert_eq!(container.member(name).expect("present").name().expect("name"), name);
    }
    assert!(matches!(container.member("b.bin"), Err(Error::MemberNotFound { .. })));
    assert!(matches!(container.single(), Err(Error::MemberNotFound { .. })));
    let empty = container.member("empty").expect("present");
    assert!(empty.is_empty());
    empty.read_at(0, &mut []).expect("an empty read at the end");
    empty.verify().expect("an empty member hashes to the hash of nothing");
    assert!(matches!(empty.read_at(1, &mut []), Err(Error::OutOfBounds { .. })));
}

#[test]
fn read_at_matches_the_reference_at_every_boundary() {
    let (file, content, boundaries) = three_extents();
    for threads in [1, 3] {
        let member = open(file.clone(), threads).single().expect("one member");
        for boundary in &boundaries {
            for before in [0, 1, 2, 3] {
                for len in [0, 1, 2, 3, 5, 4096, 20_000, 70_000] {
                    let offset = len_of(*boundary).saturating_sub(before);
                    let len = usize::min(len + before, content.len() - offset);
                    assert_read(&member, &content, offset, len);
                }
            }
        }
    }
}

#[test]
fn read_at_matches_the_reference_at_random_offsets() {
    let (file, content, _) = three_extents();
    let member = open(file, 4).single().expect("one member");
    let mut rng = Rng(0x5EED);
    for _ in 0..2000 {
        let offset = len_of(rng.below(content.len() as u64 + 1));
        let room = (content.len() - offset) as u64;
        let scale = 1 << rng.below(17);
        let len = len_of(rng.below(room.min(scale) + 1));
        assert_read(&member, &content, offset, len);
    }
    assert_read(&member, &content, 0, content.len());
}

#[test]
fn reads_past_the_end_are_out_of_bounds_before_any_io() {
    let (file, content, _) = three_extents();
    let (container, source) = recording(file, 2);
    let member = container.single().expect("one member");
    let end = content.len() as u64;
    let mut byte = [0];
    for (offset, dst) in [(end, &mut byte[..]), (u64::MAX, &mut [0; 2][..])] {
        let error = member.read_at(offset, dst).expect_err("past the end");
        assert!(matches!(error, Error::OutOfBounds { file_len, .. } if file_len == end), "{error:?}");
    }
    member.read_at(end, &mut []).expect("an empty read at the end is in range");
    let mut reqs = [
        ReadRequest { offset: 0, dst: &mut [0; 4] },
        ReadRequest { offset: end, dst: &mut [0; 1] },
    ];
    assert!(matches!(member.read_many(&mut reqs), Err(Error::OutOfBounds { .. })));
    assert_eq!(source.take_reads(), []);
}

/// Every chain, under every stream policy, read in parts and whole.
#[test]
fn every_chain_and_policy_reads_back() {
    let widths = [ElementWidth::Two, ElementWidth::Four, ElementWidth::Eight];
    let mut layouts = vec![Layout::Whole];
    layouts.extend(widths.map(Layout::Planes));
    layouts.extend(
        [FieldTable::Q80, FieldTable::Q4K, FieldTable::Q5K, FieldTable::Q6K].map(Layout::Fields),
    );
    let mut chains: Vec<Chain> = Vec::new();
    for layout in layouts {
        chains.push(Chain::new(&[Step::Layout(layout)]).expect("valid"));
        chains.extend(widths.map(|width| rotated(width, layout)));
    }
    let policies = [
        StreamPolicy::Store,
        StreamPolicy::Entropy,
        StreamPolicy::EntropyOrStore,
        StreamPolicy::Zstd { level: 1 },
    ];
    for chain in chains {
        for policy in policies {
            let block = chain.block_bytes() as usize;
            let mut builder = ContainerBuilder::new();
            let raw = tensor_like(u64::from(chain.block_bytes()), block * 40, 2);
            let blob = builder.add_blob(raw, chain, chain.block_bytes() * 16, policy);
            let extents = [identity(blob, 0, (block * 40) as u64)];
            let content = builder.add_member("m", &extents);
            let member = open(builder.finish(), 2).single().expect("one member");
            let size = content.len();
            assert_read(&member, &content, 0, size);
            assert_read(&member, &content, 1, size - 2);
            assert_read(&member, &content, block * 16 - 1, 2);
            member.verify().expect("a good member verifies");
        }
    }
}

/// `widened` is a member of one extent over narrow bytes, with the view applied at `blob_offset`.
fn widened(view: View, chain: Chain, chunk_size: u32, narrow: usize) -> (Vec<u8>, Vec<u8>) {
    let mut builder = ContainerBuilder::new();
    let block = chain.block_bytes() as usize;
    let raw = tensor_like(9, (narrow + 3) / block * block, 2);
    let blob = builder.add_blob(raw, chain, chunk_size, StreamPolicy::EntropyOrStore);
    let extent = ExtentSpec { blob, blob_offset: 2, virt_len: narrow as u64 * 2, view };
    let content = builder.add_member("wide", &[extent]);
    (builder.finish(), content)
}

#[test]
fn widened_views_read_at_unaligned_offsets_across_chunks() {
    for view in [View::F16ToF32, View::Bf16ToF32] {
        for threads in [1, 2] {
            let chain = planes(ElementWidth::Two);
            let (file, content) = widened(view, chain, 64, 300);
            let member = open(file, threads).single().expect("one member");
            for offset in 0..content.len() {
                for len in [0, 1, 2, 3, 4, 5, 7, 8, 9, 125, 126, 127, 260] {
                    assert_read(&member, &content, offset, len.min(content.len() - offset));
                }
            }
            member.verify().expect("the hash covers widened bytes");
        }
    }
}

/// A `whole` blob may have an odd `chunk_size`, and then elements straddle chunks. The blob here also
/// has an odd length, so its last byte belongs to no element. Spec section 8 defines the bytes through
/// the blob, so all of this must read, on every path and with no split element reaching a view.
#[test]
fn widened_elements_straddling_chunks_read_correctly() {
    for chunk_size in [1, 3, 5, 7, 63] {
        for threads in [1, 3] {
            let (file, content) = widened(View::F16ToF32, whole(), chunk_size, 40);
            let member = open(file, threads).single().expect("one member");
            for offset in 0..content.len() {
                for len in [1, 2, 3, 4, 5, 6, 9, 17, 80] {
                    assert_read(&member, &content, offset, len.min(content.len() - offset));
                }
            }
            member.verify().expect("the hash covers every element");
        }
    }
}

/// `validate` refuses an extent that would split an element at the end of a blob or by its alignment,
/// so the read path never meets one.
#[test]
fn a_range_that_splits_an_element_is_rejected_at_open() {
    let view = View::F16ToF32;
    let cases = [
        // The blob ends one byte into the last element.
        ExtentSpec { blob: 0, blob_offset: 2, virt_len: 44, view },
        ExtentSpec { blob: 0, blob_offset: 1, virt_len: 8, view },
        ExtentSpec { blob: 0, blob_offset: 0, virt_len: 6, view },
    ];
    for extent in cases {
        let mut builder = ContainerBuilder::new();
        builder.add_blob(tensor_like(4, 23, 2), whole(), 5, StreamPolicy::Store);
        builder.add_member_records("m", &[extent], &[], extent.virt_len, [0; 32]);
        let error = open_with(builder.finish(), &options(1)).expect_err("an element would split");
        assert!(matches!(error, Error::MalformedIndex { section: "EXTN", .. }), "{error:?}");
    }
}

#[test]
fn a_chunk_shared_by_two_chains_decodes_by_the_chain_of_each_blob() {
    let raw = tensor_like(5, 64, 1);
    let mut builder = ContainerBuilder::new();
    let plain = builder.add_blob(raw.clone(), whole(), 64, StreamPolicy::Store);
    let chain = rotated(ElementWidth::Two, Layout::Whole);
    let other = builder.share_chunks(plain, chain, raw.clone());
    builder.add_member("plain", &[identity(plain, 0, 64)]);
    builder.add_member("rotated", &[identity(other, 0, 64)]);
    let file = builder.finish();

    let off = OpenOptions { threads: 1, verify: Verify::Off, ..OpenOptions::default() };
    let container = open_with(file.clone(), &off).expect("valid");
    let mut read = vec![0; 64];
    container.member("plain").expect("present").read_at(0, &mut read).expect("decodes");
    assert_eq!(read, raw);
    container.member("rotated").expect("present").read_at(0, &mut read).expect("decodes");
    let (pairs, _) = raw.as_chunks::<2>();
    let expected: Vec<u8> = pairs
        .iter()
        .flat_map(|pair| u16::from_le_bytes(*pair).rotate_right(1).to_le_bytes())
        .collect();
    assert_eq!(read, expected);

    // With verification the hash belongs to one chain, and the other is located as corrupt.
    let container = open(file, 1);
    container.member("plain").expect("present").read_at(0, &mut read).expect("the hash holds");
    let error = container.member("rotated").expect("present").read_at(0, &mut read);
    assert!(matches!(error, Err(Error::ChecksumMismatch { blob: 1, chunk: 0, .. })), "{error:?}");
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
fn a_corrupt_chunk_gives_a_located_error() {
    let (file, content) = with_flipped_byte();
    for threads in [1, 3] {
        let member = open(file.clone(), threads).single().expect("one member");
        let mut all = vec![0; content.len()];
        let error = member.read_at(0, &mut all).expect_err("corrupt");
        assert!(matches!(error, Error::ChecksumMismatch { blob: 0, chunk: 2, .. }), "{error:?}");
        // The neighbours are intact, and a read of part of the chunk checks all of it.
        assert_read(&member, &content, 0, 512);
        assert_read(&member, &content, 768, 256);
        let error = member.read_at(520, &mut [0; 1]).expect_err("corrupt");
        assert!(matches!(error, Error::ChecksumMismatch { blob: 0, chunk: 2, .. }), "{error:?}");
    }
}

#[test]
fn a_broken_prologue_gives_a_located_error() {
    let (mut file, content) = with_flipped_byte();
    file[64 + 260] ^= 1;
    let member = open(file, 1).single().expect("one member");
    let mut all = vec![0; content.len()];
    let error = member.read_at(0, &mut all).expect_err("corrupt");
    assert!(matches!(error, Error::ChunkCorrupt { blob: 0, chunk: 1, .. }), "{error:?}");
}

#[test]
fn verify_off_skips_chunk_hashes_only() {
    let (file, content) = with_flipped_byte();
    let off = OpenOptions { threads: 1, verify: Verify::Off, ..OpenOptions::default() };
    let member = open_with(file, &off).expect("valid").single().expect("one member");
    let mut all = vec![0; content.len()];
    member.read_at(0, &mut all).expect("no hash is checked");
    assert_ne!(all, content);
    assert!(matches!(member.verify(), Err(Error::VerifyFailed { .. })));
}

#[test]
fn chunks_are_verified_before_a_view() {
    let (mut file, content) = widened(View::Bf16ToF32, whole(), 64, 256);
    assert_eq!(content.len(), 512);
    file[64 + 4 + 10] ^= 0x01;
    let member = open(file, 1).single().expect("one member");
    let error = member.read_at(0, &mut [0; 4]).expect_err("corrupt");
    assert!(matches!(error, Error::ChecksumMismatch { blob: 0, chunk: 0, .. }), "{error:?}");
}

#[test]
fn tensors_read_by_name_with_an_exact_destination() {
    let (file, content, _) = three_extents();
    let member = open(file, 2).single().expect("one member");
    let mut dst = vec![0; 60_000];
    member.decode_tensor_into("t1", &mut dst).expect("present");
    assert_eq!(dst, content[30_000..90_000]);
    member.decode_tensor_into("empty", &mut []).expect("a tensor may be empty");
    let error = member.decode_tensor_into("t1", &mut [0; 3]).expect_err("wrong size");
    assert!(
        matches!(error, Error::DestinationSize { expected: 60_000, actual: 3 }),
        "{error:?}"
    );
    let error = member.decode_tensor_into("none", &mut []).expect_err("absent");
    assert!(matches!(error, Error::TensorNotFound { .. }), "{error:?}");
}

/// A member of `chunks` chunks of 1 KiB, stored in order and with no gaps.
fn contiguous(chunks: usize) -> (Vec<u8>, Vec<u8>) {
    let mut builder = ContainerBuilder::new();
    let raw = tensor_like(7, chunks * KIB, 1);
    let blob = builder.add_blob(raw, whole(), KIB_U32, StreamPolicy::Store);
    let content = builder.add_member("m", &[identity(blob, 0, (chunks * KIB) as u64)]);
    (builder.finish(), content)
}

#[test]
fn adjacent_chunks_coalesce_into_reads_of_a_batch() {
    let (file, content) = contiguous(10);
    let (container, source) = recording(file, 3);
    let member = container.single().expect("one member");
    let mut all = vec![0; content.len()];
    member.read_at(0, &mut all).expect("reads");
    assert_eq!(all, content);
    let stored = KIB + 4;
    assert_eq!(
        source.take_reads(),
        [
            (64, 3 * stored),
            ((64 + 3 * stored) as u64, 3 * stored),
            ((64 + 6 * stored) as u64, 3 * stored),
            ((64 + 9 * stored) as u64, stored),
        ]
    );
}

#[test]
fn one_thread_reads_one_chunk_at_a_time() {
    let (file, content) = contiguous(3);
    let (container, source) = recording(file, 1);
    let mut all = vec![0; content.len()];
    container.single().expect("one member").read_at(0, &mut all).expect("reads");
    assert_eq!(source.take_reads().len(), 3);
    assert!(container.shared.engine.is_inline());
}

#[test]
fn a_partial_read_fetches_only_its_chunks() {
    let (file, content) = contiguous(10);
    let (container, source) = recording(file, 3);
    let member = container.single().expect("one member");
    assert_read(&member, &content, 2 * KIB + 7, 3);
    assert_eq!(source.take_reads(), [((64 + 2 * (KIB + 4)) as u64, KIB + 4)]);
    assert_read(&member, &content, 2 * KIB + 1000, 100);
    assert_eq!(source.take_reads(), [((64 + 2 * (KIB + 4)) as u64, 2 * (KIB + 4))]);
}

#[test]
fn two_extents_over_one_chunk_read_it_once() {
    let mut builder = ContainerBuilder::new();
    let blob = builder.add_blob(tensor_like(8, KIB, 1), whole(), KIB_U32, StreamPolicy::Store);
    let content = builder.add_member("m", &[identity(blob, 0, 100), identity(blob, 100, 200)]);
    let (container, source) = recording(builder.finish(), 2);
    assert_read(&container.single().expect("one member"), &content, 0, 300);
    assert_eq!(source.take_reads(), [(64, KIB + 4)]);
}

#[test]
fn gaps_between_chunks_split_the_read_but_not_the_batch() {
    let mut builder = ContainerBuilder::new();
    let mut blob = |seed| {
        builder.add_blob(tensor_like(seed, 2 * KIB, 1), whole(), KIB_U32, StreamPolicy::Store)
    };
    let (first, skipped, last) = (blob(1), blob(2), blob(3));
    assert_eq!(skipped, 1);
    let extents = [identity(first, 0, 2 * KIB as u64), identity(last, 0, 2 * KIB as u64)];
    let content = builder.add_member("m", &extents);
    let (container, source) = recording(builder.finish(), 4);
    assert_read(&container.single().expect("one member"), &content, 0, content.len());
    let stored = KIB + 4;
    assert_eq!(source.take_reads(), [(64, 2 * stored), ((64 + 4 * stored) as u64, 2 * stored)]);
}

/// Three blobs in file order, and a member that lays them out in reverse.
fn reversed_member() -> (Vec<u8>, Vec<u8>) {
    let mut builder = ContainerBuilder::new();
    let mut blob = |seed| {
        builder.add_blob(tensor_like(seed, 3 * KIB, 2), whole(), KIB_U32, StreamPolicy::Store)
    };
    let (a, b, c) = (blob(1), blob(2), blob(3));
    let len = 3 * KIB as u64;
    let content = builder.add_member("m", &[identity(c, 0, len), identity(b, 0, len), identity(a, 0, len)]);
    (builder.finish(), content)
}

#[test]
fn read_many_visits_the_file_in_pack_order_and_keeps_requests_in_place() {
    let (file, content) = reversed_member();
    let (container, source) = recording(file, 2);
    let member = container.single().expect("one member");
    let requests = [(0, 3 * KIB), (3 * KIB, 3 * KIB), (6 * KIB, 3 * KIB)];
    let mut buffers: Vec<Vec<u8>> = requests.iter().map(|(_, len)| vec![0; *len]).collect();
    let mut reqs: Vec<ReadRequest<'_>> = requests
        .iter()
        .zip(&mut buffers)
        .map(|((offset, _), dst)| ReadRequest { offset: *offset as u64, dst })
        .collect();
    member.read_many(&mut reqs).expect("reads");
    assert_eq!(reqs[0].offset, 0, "requests keep their order");
    drop(reqs);
    for ((offset, len), buffer) in requests.iter().zip(&buffers) {
        assert_eq!(buffer, &content[*offset..offset + len]);
    }
    let reads = source.take_reads();
    assert!(reads.is_sorted(), "{reads:?}");
    let total: usize = reads.iter().map(|(_, len)| *len).sum();
    assert_eq!(total, 9 * (KIB + 4), "every chunk is read once");
}

#[test]
fn read_many_sorts_the_pieces_of_a_request_that_crosses_extents() {
    let (file, content) = reversed_member();
    let (container, source) = recording(file, 1);
    let member = container.single().expect("one member");
    let mut all = vec![0; content.len()];
    member.read_many(&mut [ReadRequest { offset: 0, dst: &mut all }]).expect("reads");
    assert_eq!(all, content);
    let reads = source.take_reads();
    assert!(reads.is_sorted(), "{reads:?}");
}

#[test]
fn read_many_serves_widened_and_empty_requests() {
    let (file, content) = widened(View::F16ToF32, planes(ElementWidth::Two), 64, 300);
    let member = open(file, 3).single().expect("one member");
    let (mut first, mut second) = (vec![0; 130], vec![0; 7]);
    let mut reqs = [
        ReadRequest { offset: 301, dst: &mut first },
        ReadRequest { offset: 3, dst: &mut second },
        ReadRequest { offset: 600, dst: &mut [] },
    ];
    member.read_many(&mut reqs).expect("reads");
    assert_eq!(first, content[301..431]);
    assert_eq!(second, content[3..10]);
}

#[test]
fn concurrent_reads_from_eight_threads_are_correct() {
    let (file, content, _) = three_extents();
    let container = open(file, 3);
    let content = &content;
    std::thread::scope(|scope| {
        for seed in 0..8 {
            let member = container.single().expect("one member");
            scope.spawn(move || {
                let mut rng = Rng(seed);
                for _ in 0..300 {
                    let offset = len_of(rng.below(content.len() as u64));
                    let room = (content.len() - offset) as u64;
                    let len = len_of(rng.below(room.min(40_000) + 1));
                    assert_read(&member, content, offset, len);
                }
            });
        }
    });
    assert_eq!(container.shared.engine.scratch_available(), 3, "every entry came back");
}

#[test]
fn clones_share_one_container() {
    let (file, content, _) = three_extents();
    let container = open(file, 2);
    let clone = container.clone();
    drop(container);
    assert_read(&clone.single().expect("one member"), &content, 100, 1000);
    assert!(format!("{clone:?}").contains("Container"));
}

#[test]
fn the_budget_is_refused_before_any_data_read() {
    let (file, content, _) = three_extents();
    let data_end = footer_of(&file).index_offset;
    let source = Arc::new(MemorySource::new(file));
    let limit = content.len() as u64 - 1;
    let options = OpenOptions { max_total_decoded: limit, ..options(1) };
    let shared: Arc<dyn crate::io::RandomAccess> = source.clone();
    let error = Container::from_source(shared.clone(), &options).expect_err("over budget");
    assert!(
        matches!(error, Error::BudgetExceeded { limit: got, .. } if got == limit),
        "{error:?}"
    );
    let reads = source.take_reads();
    assert_ne!(reads, []);
    for (offset, len) in reads {
        assert!(
            (offset, len) == (0, 64) || offset >= data_end,
            "a read of {len} bytes at {offset} touched the data region"
        );
    }
    let options = OpenOptions { max_total_decoded: limit + 1, ..options };
    Container::from_source(shared, &options).expect("exactly on budget");
}

#[test]
fn the_budget_covers_every_member_of_a_pack() {
    let mut builder = ContainerBuilder::new();
    let blob = builder.add_blob(tensor_like(3, 100, 1), whole(), 100, StreamPolicy::Store);
    builder.add_member("a", &[identity(blob, 0, 60)]);
    builder.add_member("b", &[identity(blob, 0, 60)]);
    let file = builder.finish();
    let error = open_with(file, &OpenOptions { max_total_decoded: 100, ..options(1) });
    assert!(matches!(error, Err(Error::BudgetExceeded { declared: 120, .. })), "{error:?}");
}

/// One byte extents over one chunk: 2000 bytes of member that would decode 2 GiB, which no budget on
/// member lengths sees. `amplification` is the option the file is opened with.
fn opens_with_one_byte_extents(amplification: u64) -> (Result<Container, Error>, Arc<MemorySource>) {
    let mut builder = ContainerBuilder::new();
    let blob = builder.add_blob(tensor_like(3, 1 << 20, 1), whole(), 1 << 20, StreamPolicy::Store);
    let extents: Vec<ExtentSpec> = (0..2000).map(|at| identity(blob, at, 1)).collect();
    builder.add_member("m", &extents);
    let source = Arc::new(MemorySource::new(builder.finish()));
    let shared: Arc<dyn crate::io::RandomAccess> = source.clone();
    let options = OpenOptions { max_decode_amplification: amplification, ..options(1) };
    (Container::from_source(shared, &options), source)
}

#[test]
fn one_byte_extents_over_one_chunk_are_refused_before_any_data_read() {
    let (result, source) = opens_with_one_byte_extents(16);
    let error = result.expect_err("over the bound");
    assert!(
        matches!(error, Error::BudgetExceeded { what: "decode work", declared, .. }
            if declared == 2000 << 20),
        "{error:?}"
    );
    let data_end = 64 + (1 << 20) + 4;
    for (offset, len) in source.take_reads() {
        assert!(
            (offset, len) == (0, 64) || offset >= data_end,
            "a read of {len} bytes at {offset} touched the data region"
        );
    }
}

#[test]
fn a_caller_may_accept_the_decode_work_of_a_file() {
    let (result, _) = opens_with_one_byte_extents(1 << 21);
    let member = result.expect("allowed").single().expect("one member");
    member.verify().expect("the bytes are right, at a price");
}

/// What the rule must leave alone: a member that is mostly references to few small blobs, each extent
/// covering its blob, under the writer's 1 MiB chunk size.
#[test]
fn a_heavily_deduplicated_member_passes_the_decode_work_bound() {
    let mut builder = ContainerBuilder::new();
    let blobs: Vec<u32> = (0..32)
        .map(|seed| {
            let raw = tensor_like(seed, 8 * KIB, 2);
            builder.add_blob(raw, whole(), 1 << 20, StreamPolicy::Store)
        })
        .collect();
    let len = 8 * KIB as u64;
    let extents: Vec<ExtentSpec> = (0..8000_usize)
        .map(|at| identity(blobs[at % blobs.len()], 0, len))
        .collect();
    let content = builder.add_member("dedup", &extents);
    let sibling = builder.add_member("sibling", &extents[..4000]);
    let container = open(builder.finish(), 2);
    assert_read(&container.member("dedup").expect("present"), &content, 3, 1 << 20);
    assert_read(&container.member("sibling").expect("present"), &sibling, 0, sibling.len());
}

/// The other shape of sharing: a blob with widened and narrow siblings over the same bytes.
#[test]
fn widened_siblings_of_one_blob_pass_the_decode_work_bound() {
    let mut builder = ContainerBuilder::new();
    let blob = builder.add_blob(
        tensor_like(1, 4 << 20, 2),
        planes(ElementWidth::Two),
        1 << 20,
        StreamPolicy::Store,
    );
    let len = 4 << 20;
    let wide = ExtentSpec { blob, blob_offset: 0, virt_len: 2 * len, view: View::F16ToF32 };
    builder.add_member("narrow", &[identity(blob, 0, len)]);
    let content = builder.add_member("wide", &[wide]);
    let member = open(builder.finish(), 2).member("wide").expect("present");
    assert_read(&member, &content, 1, 4096);
}

/// Replaces the stored index of a file by `stored`, whose content is `raw`.
fn with_stored_index(file: &[u8], stored: &[u8], raw: &[u8]) -> Vec<u8> {
    let offset = footer_of(file).index_offset;
    let mut rebuilt = file[..len_of(offset)].to_vec();
    rebuilt.extend_from_slice(stored);
    let generation = crate::format::GENERATION_INITIAL;
    let footer = Footer::for_index(offset, stored.len() as u64, raw, generation);
    rebuilt.extend_from_slice(&footer.to_bytes());
    rebuilt
}

/// A small valid file and its decompressed index.
fn small_file() -> (Vec<u8>, Vec<u8>) {
    let mut builder = IndexBuilder::new();
    let name = builder.add_string("").expect("short");
    assert_eq!(name.len, 0);
    let raw = builder.serialize();
    (finish_file(Header::current().to_bytes().to_vec(), &raw), raw)
}

#[test]
fn an_empty_container_opens_and_has_no_single_member() {
    let (file, _) = small_file();
    let container = open(file, 2);
    assert_eq!(container.members(), []);
    assert!(matches!(container.single(), Err(Error::MemberNotFound { .. })));
}

#[test]
fn short_files_are_rejected_cleanly() {
    let (file, _) = small_file();
    for len in [0, 1, 8, 63] {
        let error = open_with(file[..len].to_vec(), &options(1)).expect_err("short");
        assert!(matches!(error, Error::NotZiptensors), "{error:?}");
    }
    for len in [64, 65, 127] {
        let error = open_with(file[..len].to_vec(), &options(1)).expect_err("short");
        assert!(matches!(error, Error::MalformedIndex { section: "footer", .. }), "{error:?}");
    }
    let error = open_with(vec![0; 4096], &options(1)).expect_err("no magic");
    assert!(matches!(error, Error::NotZiptensors), "{error:?}");
}

#[test]
fn every_truncation_is_a_typed_error() {
    let (file, _, _) = three_extents();
    let step = (file.len() / 200).max(1);
    for len in (0..file.len()).step_by(step) {
        open_with(file[..len].to_vec(), &options(1)).expect_err("truncated");
    }
}

#[test]
fn a_second_frame_after_the_index_is_rejected() {
    let (file, raw) = small_file();
    let mut stored = zstd::bulk::compress(&raw, 1).expect("compress");
    stored.extend_from_slice(&zstd::bulk::compress(&[], 1).expect("compress"));
    let error = open_with(with_stored_index(&file, &stored, &raw), &options(1));
    assert!(matches!(error, Err(Error::MalformedIndex { section: "index", .. })), "{error:?}");
}

#[test]
fn a_skippable_frame_is_not_an_index() {
    let (file, raw) = small_file();
    let mut stored = vec![0x50, 0x2A, 0x4D, 0x18, 0, 0, 0, 0];
    stored.extend_from_slice(&zstd::bulk::compress(&raw, 1).expect("compress"));
    let error = open_with(with_stored_index(&file, &stored, &raw), &options(1));
    assert!(matches!(error, Err(Error::MalformedIndex { section: "index", .. })), "{error:?}");
}

#[test]
fn a_frame_of_another_length_is_rejected() {
    let (file, raw) = small_file();
    let longer = [raw.as_slice(), &[0; 8]].concat();
    let stored = zstd::bulk::compress(&longer, 1).expect("compress");
    let error = open_with(with_stored_index(&file, &stored, &raw), &options(1));
    assert!(matches!(error, Err(Error::MalformedIndex { section: "index", .. })), "{error:?}");
}

#[test]
fn a_frame_without_a_content_size_is_rejected() {
    let (file, raw) = small_file();
    let mut encoder = zstd::stream::write::Encoder::new(Vec::new(), 1).expect("encoder");
    std::io::Write::write_all(&mut encoder, &raw).expect("compress");
    let stored = encoder.finish().expect("compress");
    let error = open_with(with_stored_index(&file, &stored, &raw), &options(1));
    assert!(matches!(error, Err(Error::MalformedIndex { section: "index", .. })), "{error:?}");
}

#[test]
fn a_flipped_index_byte_is_a_typed_error() {
    let (file, _, _) = three_extents();
    let footer = footer_of(&file);
    for at in len_of(footer.index_offset)..file.len() - 64 {
        let mut broken = file.clone();
        broken[at] ^= 0x40;
        // A few bits of a zstd frame have no effect on its output. Then the index hash still holds
        // and the file is as good as it was.
        if let Err(error) = open_with(broken, &options(1)) {
            assert!(
                matches!(error, Error::MalformedIndex { .. } | Error::Zstd { .. }),
                "byte {at}: {error:?}"
            );
        }
    }
}

#[test]
fn a_stored_index_longer_than_zstd_could_write_is_refused_before_it_is_read() {
    let (file, raw) = small_file();
    let source_len = file.len();
    let index = footer_of(&file).index_offset;
    let mut padded = file[..len_of(index)].to_vec();
    padded.resize(padded.len() + (1 << 20), 0);
    let stored_len = (padded.len() as u64) - index;
    let generation = crate::format::GENERATION_INITIAL;
    padded.extend_from_slice(&Footer::for_index(index, stored_len, &raw, generation).to_bytes());
    assert!(padded.len() > source_len);
    let error = open_with(padded, &options(1));
    assert!(matches!(error, Err(Error::BudgetExceeded { what: "stored index", .. })), "{error:?}");
}

#[test]
fn opening_a_missing_or_foreign_path_fails_cleanly() {
    let dir = TempDir::new();
    let path = dir.path().join("absent.ziptensors");
    let error = Container::open(&path, &options(1)).expect_err("absent");
    assert!(matches!(error, Error::Io { op: "open", .. }), "{error:?}");
    let error = Container::open(dir.path(), &options(1)).expect_err("a directory");
    assert!(matches!(error, Error::Io { .. }), "{error:?}");
    fs::write(&path, b"not a container, whatever its name says").expect("scratch file");
    let error = Container::open(&path, &options(1)).expect_err("foreign");
    assert!(matches!(error, Error::NotZiptensors), "{error:?}");
}

#[test]
fn a_file_on_disk_opens_and_reading_writes_nothing_beside_it() {
    let (file, content, _) = three_extents();
    let dir = TempDir::new();
    let path = dir.path().join("model.bin");
    fs::write(&path, &file).expect("scratch file");
    let container = Container::open(&path, &options(2)).expect("valid");
    let member = container.single().expect("one member");
    assert_read(&member, &content, 0, content.len());
    member.verify().expect("verifies");
    assert_eq!(dir.names(), ["model.bin"]);
}

#[test]
fn restore_writes_the_content_and_leaves_one_file() {
    let (file, content, _) = three_extents();
    let member = open(file, 3).single().expect("one member");
    let dir = TempDir::new();
    let out = dir.path().join("restored.bin");
    let report = member.restore_file(&out).expect("restores");
    assert_eq!(report.bytes, content.len() as u64);
    assert_eq!(report.blake3, *blake3::hash(&content).as_bytes());
    assert_eq!(fs::read(&out).expect("the output exists"), content);
    assert_eq!(dir.names(), ["restored.bin"]);
    member.restore_file(&out).expect("replaces a finished output");
    assert_eq!(fs::read(&out).expect("the output exists"), content);
    assert_eq!(dir.names(), ["restored.bin"]);
}

#[test]
fn restore_of_a_widened_member_and_an_empty_member() {
    let (file, content) = widened(View::F16ToF32, planes(ElementWidth::Two), 64, 3000);
    let member = open(file, 2).single().expect("one member");
    let dir = TempDir::new();
    let out = dir.path().join("wide.bin");
    member.restore_file(&out).expect("restores");
    assert_eq!(fs::read(&out).expect("the output exists"), content);

    let mut builder = ContainerBuilder::new();
    builder.add_member("empty", &[]);
    let member = open(builder.finish(), 1).single().expect("one member");
    let out = dir.path().join("empty.bin");
    assert_eq!(member.restore_file(&out).expect("restores").bytes, 0);
    assert_eq!(fs::read(&out).expect("the output exists"), Vec::<u8>::new());
}

#[test]
fn a_failed_restore_leaves_no_final_file_and_no_temporary_file() {
    let (file, content) = with_flipped_byte();
    let member = open(file, 2).single().expect("one member");
    let dir = TempDir::new();
    let out = dir.path().join("restored.bin");
    let error = member.restore_file(&out).expect_err("corrupt");
    assert!(matches!(error, Error::ChecksumMismatch { blob: 0, chunk: 2, .. }), "{error:?}");
    assert_eq!(dir.names(), Vec::<String>::new());

    // An output from before keeps its bytes.
    fs::write(&out, b"earlier").expect("scratch file");
    member.restore_file(&out).expect_err("corrupt");
    assert_eq!(dir.names(), ["restored.bin"]);
    assert_eq!(fs::read(&out).expect("kept"), b"earlier");
    assert_eq!(content.len(), 1024);
}

#[test]
fn a_hash_mismatch_fails_restore_and_verify_and_leaves_nothing() {
    let mut builder = ContainerBuilder::new();
    let blob = builder.add_blob(tensor_like(3, 512, 1), whole(), 128, StreamPolicy::Store);
    builder.add_member_records("m", &[identity(blob, 0, 512)], &[], 512, [7; 32]);
    let member = open(builder.finish(), 2).single().expect("one member");
    let dir = TempDir::new();
    let out = dir.path().join("restored.bin");
    let error = member.restore_file(&out).expect_err("wrong hash");
    assert!(matches!(error, Error::VerifyFailed { .. }), "{error:?}");
    assert_eq!(dir.names(), Vec::<String>::new());
    assert!(matches!(member.verify(), Err(Error::VerifyFailed { .. })));
}

#[test]
fn restore_to_a_path_it_cannot_use_fails_cleanly() {
    let (file, _, _) = three_extents();
    let member = open(file, 1).single().expect("one member");
    let dir = TempDir::new();
    let error = member.restore_file(&dir.path().join("no/such/dir/x.bin")).expect_err("no dir");
    assert!(matches!(error, Error::Io { op: "create temporary file", .. }), "{error:?}");
    let error = member.restore_file(Path::new("/")).expect_err("no file name");
    assert!(matches!(error, Error::Io { op: "create temporary file", .. }), "{error:?}");
    assert_eq!(dir.names(), Vec::<String>::new());
}

/// A widened chunk is two chunks of the member, so at one thread a restore needs a buffer of that
/// size to end on a chunk boundary. Otherwise each read starts mid-chunk and every chunk decodes twice.
#[test]
fn a_widened_member_decodes_each_chunk_once_at_one_thread() {
    const CHUNK: u32 = 1 << 12;
    const CHUNKS: usize = 16;
    let chunk = len_of(u64::from(CHUNK));
    let mut builder = ContainerBuilder::new();
    let chain = planes(ElementWidth::Two);
    let policy = StreamPolicy::Entropy;
    let blob = builder.add_blob(tensor_like(9, chunk * CHUNKS, chunk), chain, CHUNK, policy);
    let virt_len = (chunk * CHUNKS * 2) as u64;
    builder.add_member("wide", &[ExtentSpec { blob, blob_offset: 0, virt_len, view: View::F16ToF32 }]);
    let file = Arc::new(MemorySource::new(builder.finish()));
    for threads in [1, 2, 4] {
        let source: Arc<dyn RandomAccess> = file.clone();
        let options = OpenOptions { threads, ..OpenOptions::default() };
        let container = Container::from_source(source, &options).expect("valid");
        file.take_reads();
        container.single().expect("one member").verify().expect("valid");
        let mut offsets: Vec<u64> = file.take_reads().iter().map(|read| read.0).collect();
        let count = offsets.len();
        offsets.dedup();
        offsets.sort_unstable();
        offsets.dedup();
        assert_eq!(offsets.len(), count, "a chunk was read twice at {threads} threads");
    }
}

/// Extents that alternate between two far blobs make a batch of runs in member order that jump back
/// and forth. Read sorted, a batch moves one way. A blob no extent uses lies between the two, so
/// their runs cannot join into one read.
#[test]
fn the_runs_of_a_batch_are_read_in_file_order() {
    const CHUNK: u32 = 1 << 12;
    const CHUNKS: usize = 8;
    let chunk = len_of(u64::from(CHUNK));
    let mut builder = ContainerBuilder::new();
    let chain = planes(ElementWidth::Two);
    let policy = StreamPolicy::Entropy;
    let near = builder.add_blob(tensor_like(10, chunk * CHUNKS, chunk), chain, CHUNK, policy);
    builder.add_blob(tensor_like(12, chunk, chunk), chain, CHUNK, policy);
    let far = builder.add_blob(tensor_like(11, chunk * CHUNKS, chunk), chain, CHUNK, policy);
    let extents: Vec<ExtentSpec> = (0..CHUNKS as u64)
        .flat_map(|at| {
            let (start, len) = (at * u64::from(CHUNK), u64::from(CHUNK));
            // The far blob first, so that member order runs against file order.
            [identity(far, start, len), identity(near, start, len)]
        })
        .collect();
    let reference = builder.add_member("fragmented", &extents);
    let file = Arc::new(MemorySource::new(builder.finish()));
    let source: Arc<dyn RandomAccess> = file.clone();
    // Enough threads for all 16 chunks to fit one batch: a new batch may start lower.
    let options = OpenOptions { threads: 16, ..OpenOptions::default() };
    let member = Container::from_source(source, &options).expect("valid").single().expect("one");
    file.take_reads();
    let mut read = vec![0; reference.len()];
    member.read_at(0, &mut read).expect("reads");
    assert_eq!(read, reference);
    let reads = file.take_reads();
    assert!(reads.len() > 1, "{reads:?}");
    assert!(reads.windows(2).all(|pair| pair[0].0 < pair[1].0), "a jump back in {reads:?}");
}

/// A sequential read may stop at a chunk boundary so that the next one starts on a whole chunk.
#[test]
fn stream_ends_fall_on_chunk_boundaries_when_they_can() {
    let (file, content, _) = three_extents();
    let member = open(file, 1).single().expect("one member");
    assert_eq!(member.chunk_aligned_end(0, 8000).expect("in range"), 6996);
    assert_eq!(member.chunk_aligned_end(6996, 7000).expect("no earlier boundary"), 7000);
    assert_eq!(member.chunk_aligned_end(0, 30_000).expect("an extent start"), 30_000);
    let end = content.len() as u64;
    assert_eq!(member.chunk_aligned_end(0, end).expect("the end"), end);
}


/// Members over delta blobs: `base.bin` over a plain `planes(2)` blob of 100 KiB + 8 in chunks of
/// 16 KiB; `tuned.bin` over a delta blob of 60 KiB + 2 at base offset 16 KiB, so both blobs end on a
/// short chunk; `whole.bin` over a delta blob with `rotl1` of the whole base length, whose last chunk
/// pairs with the base's short one; `mixed.bin` over pieces of all three, one widened. Returns the
/// file and `(name, content)` of each member.
fn delta_pack() -> (Vec<u8>, Vec<(&'static str, Vec<u8>)>) {
    let mut builder = ContainerBuilder::new();
    let base_raw = tensor_like(20, 100 * KIB + 8, 2);
    let chunk = 16 * KIB_U32;
    let base = builder.add_blob(base_raw.clone(), planes(ElementWidth::Two), chunk, StreamPolicy::EntropyOrStore);
    let tuned_raw = finetune_of(21, &base_raw[16 * KIB..16 * KIB + 60 * KIB + 2]);
    let chain = delta(ElementWidth::Two, planes(ElementWidth::Two));
    let tuned = builder.add_delta_blob(tuned_raw, base, 16 * 1024, chain, StreamPolicy::EntropyOrStore);
    let whole_raw = finetune_of(22, &base_raw);
    let rotated_chain = delta(ElementWidth::Four, rotated(ElementWidth::Two, Layout::Planes(ElementWidth::Two)));
    let all = builder.add_delta_blob(whole_raw, base, 0, rotated_chain, StreamPolicy::Zstd { level: 3 });
    let widened = ExtentSpec { blob: all, blob_offset: 2048, virt_len: 2 * 8000, view: View::Bf16ToF32 };
    let mixed = [
        identity(tuned, 100, 5000),
        widened,
        identity(base, 0, 300),
        identity(tuned, 30 * 1024, 20 * 1024),
    ];
    let members = vec![
        ("base.bin", builder.add_member("base.bin", &[identity(base, 0, (100 * KIB + 8) as u64)])),
        ("tuned.bin", builder.add_member("tuned.bin", &[identity(tuned, 0, (60 * KIB + 2) as u64)])),
        ("whole.bin", builder.add_member("whole.bin", &[identity(all, 0, (100 * KIB + 8) as u64)])),
        ("mixed.bin", builder.add_member("mixed.bin", &mixed)),
    ];
    (builder.finish(), members)
}

#[test]
fn delta_members_read_back_whole_in_part_and_widened() {
    let (file, members) = delta_pack();
    for threads in [1, 2, 4] {
        let container = open(file.clone(), threads);
        for (name, content) in &members {
            let member = container.member(name).expect("present");
            let mut all = vec![0; content.len()];
            member.read_at(0, &mut all).expect("reads");
            assert_eq!(&all, content, "{name} with {threads} threads");
            member.verify().expect("the member hash holds");
            let mut rng = Rng(23);
            for _ in 0..40 {
                let offset = len_of(rng.below(content.len() as u64));
                let len = 1 + len_of(rng.below((content.len() - offset) as u64).min(40_000));
                assert_read(&member, content, offset, len);
            }
            for edge in [16 * KIB - 1, 32 * KIB, 48 * KIB + 1, content.len() - 3] {
                let offset = edge.min(content.len() - 1);
                assert_read(&member, content, offset, (content.len() - offset).min(3000));
            }
        }
    }
}

#[test]
fn read_many_serves_delta_and_plain_extents_together() {
    let (file, members) = delta_pack();
    let (_, mixed) = members.iter().find(|(name, _)| *name == "mixed.bin").expect("present");
    let member = open(file, 3).member("mixed.bin").expect("present");
    let requests = [(25_000, 3000), (0, 7000), (21_200, 300), (6000, 15_000)];
    let mut buffers: Vec<Vec<u8>> = requests.iter().map(|(_, len)| vec![0; *len]).collect();
    let mut reqs: Vec<ReadRequest<'_>> = requests
        .iter()
        .zip(&mut buffers)
        .map(|((offset, _), dst)| ReadRequest { offset: *offset as u64, dst })
        .collect();
    member.read_many(&mut reqs).expect("reads");
    drop(reqs);
    for ((offset, len), buffer) in requests.iter().zip(&buffers) {
        assert_eq!(buffer, &mixed[*offset..offset + len], "request at {offset}");
    }
}

/// A plain `whole` base of four stored chunks of 256 bytes, and a stored delta blob over all of it,
/// so stored bytes sit at known offsets: base chunk `i` at `64 + 260 × i`, delta chunk `i` at
/// `64 + 260 × (4 + i)`. Returns the file and the delta member's content.
fn stored_delta() -> (Vec<u8>, Vec<u8>) {
    let mut builder = ContainerBuilder::new();
    let base_raw = tensor_like(24, 4 * 256, 1);
    let base = builder.add_blob(base_raw.clone(), whole(), 256, StreamPolicy::Store);
    let target = finetune_of(25, &base_raw);
    let blob = builder.add_delta_blob(target, base, 0, delta(ElementWidth::Two, whole()), StreamPolicy::Store);
    let content = builder.add_member("m", &[identity(blob, 0, 4 * 256)]);
    (builder.finish(), content)
}

#[test]
fn a_corrupt_base_chunk_fails_with_its_own_location() {
    let (mut file, content) = stored_delta();
    file[64 + 2 * 260 + 50] ^= 0x10;
    for threads in [1, 3] {
        let member = open(file.clone(), threads).single().expect("one member");
        let mut all = vec![0; content.len()];
        let error = member.read_at(0, &mut all).expect_err("a corrupt base");
        assert!(matches!(error, Error::ChecksumMismatch { blob: 0, chunk: 2, .. }), "{error:?}");
        // Delta chunks over intact base chunks read, and a part of chunk 2 checks its whole base.
        assert_read(&member, &content, 0, 512);
        assert_read(&member, &content, 768, 256);
        let error = member.read_at(600, &mut [0; 1]).expect_err("a corrupt base");
        assert!(matches!(error, Error::ChecksumMismatch { blob: 0, chunk: 2, .. }), "{error:?}");
    }
    // Without hashes the flip goes through the combination unchanged, and only the member hash sees it.
    let off = OpenOptions { threads: 1, verify: Verify::Off, ..OpenOptions::default() };
    let member = open_with(file, &off).expect("valid").single().expect("one member");
    let mut all = vec![0; content.len()];
    member.read_at(0, &mut all).expect("no hash is checked");
    let flipped: Vec<usize> = (0..all.len()).filter(|at| all[*at] != content[*at]).collect();
    assert_eq!(flipped, [2 * 256 + 46]);
    assert!(matches!(member.verify(), Err(Error::VerifyFailed { .. })));
}

#[test]
fn a_corrupt_delta_chunk_fails_with_its_own_location() {
    let (mut file, content) = stored_delta();
    file[64 + 5 * 260 + 9] ^= 0x01;
    let member = open(file, 2).single().expect("one member");
    let mut all = vec![0; content.len()];
    let error = member.read_at(0, &mut all).expect_err("a corrupt delta chunk");
    assert!(matches!(error, Error::ChecksumMismatch { blob: 1, chunk: 1, .. }), "{error:?}");
}

#[test]
fn a_delta_job_reads_its_base_chunk_in_the_same_batch_and_shares_repeats() {
    let mut builder = ContainerBuilder::new();
    let base_raw = tensor_like(26, 2 * KIB, 1);
    let base = builder.add_blob(base_raw.clone(), whole(), KIB_U32, StreamPolicy::Store);
    let chain = delta(ElementWidth::Two, whole());
    let blob = builder.add_delta_blob(finetune_of(27, &base_raw), base, 0, chain, StreamPolicy::Store);
    // Two extents over delta chunk 0: the second job finds both of its chunks in the batch.
    let content = builder.add_member("m", &[identity(blob, 0, 100), identity(blob, 100, 200)]);
    let (container, source) = recording(builder.finish(), 2);
    assert_read(&container.single().expect("one member"), &content, 0, 300);
    let (base_at, delta_at) = (64, (64 + 2 * (KIB + 4)) as u64);
    assert_eq!(source.take_reads(), [(base_at, KIB + 4), (delta_at, KIB + 4)]);
}

#[test]
fn a_delta_job_finds_its_base_chunk_in_the_previous_job_s_own_slot() {
    let mut builder = ContainerBuilder::new();
    let base_raw = tensor_like(31, 2 * KIB, 1);
    let base = builder.add_blob(base_raw.clone(), whole(), KIB_U32, StreamPolicy::Store);
    let chain = delta(ElementWidth::Two, whole());
    let blob = builder.add_delta_blob(finetune_of(32, &base_raw), base, 0, chain, StreamPolicy::Store);
    // A plain extent over base chunk 1, then the delta extent whose chunk 1 pairs with it: the
    // second job's base is the first job's own chunk, not its base, and must not be read again.
    let kib = KIB as u64;
    let content = builder.add_member("m", &[identity(base, kib, kib), identity(blob, kib, kib)]);
    let (container, source) = recording(builder.finish(), 2);
    assert_read(&container.single().expect("one member"), &content, 0, 2 * KIB);
    // Stored chunks of `KIB + 4` bytes: base 0, base 1, delta 0, delta 1.
    let (base_at, delta_at) = ((64 + (KIB + 4)) as u64, (64 + 3 * (KIB + 4)) as u64);
    assert_eq!(source.take_reads(), [(base_at, KIB + 4), (delta_at, KIB + 4)]);
}

#[test]
fn a_delta_job_on_one_thread_takes_both_chunks_into_one_batch() {
    let (file, content) = stored_delta();
    let (container, source) = recording(file, 1);
    let mut all = vec![0; content.len()];
    container.single().expect("one member").read_at(0, &mut all).expect("reads");
    assert_eq!(all, content);
    // Four batches of one job each, every one reading its base chunk and its delta chunk.
    let expected: Vec<(u64, usize)> = (0..4_u64)
        .flat_map(|at| [(64 + 260 * at, 260), (64 + 260 * (4 + at), 260)])
        .collect();
    assert_eq!(source.take_reads(), expected);
}

/// Two delta jobs share a batch at four threads (`4 × (256 + 32)` bytes): their base chunks touch in
/// the file and so do their delta chunks, so the batch makes two reads, not one per chunk. A read that
/// starts on the byte the previous one ended at would also cost a kernel read-ahead window.
#[test]
fn the_base_and_delta_chunks_of_a_delta_batch_are_one_read_each() {
    let (file, content) = stored_delta();
    let (container, source) = recording(file, 4);
    let member = container.single().expect("one member");
    // Delta chunks 0 and 1, so base chunks 0 and 1.
    assert_read(&member, &content, 100, 300);
    assert_eq!(source.take_reads(), [(64, 520), (64 + 260 * 4, 520)]);
    let mut all = vec![0; content.len()];
    member.read_at(0, &mut all).expect("reads");
    assert_eq!(all, content);
    // Two batches of two jobs each.
    let expected = [(64, 520), (64 + 260 * 4, 520), (64 + 260 * 2, 520), (64 + 260 * 6, 520)];
    assert_eq!(source.take_reads(), expected);
}

/// A member that visits one blob's chunks out of file order and repeats one: a batch reads each
/// distinct chunk once, chunks that touch in the file as one read, and never joins chunks across a
/// gap; no read is longer than the batch.
#[test]
fn a_batch_joins_chunks_that_touch_in_the_file_only() {
    let mut builder = ContainerBuilder::new();
    let raw = tensor_like(7, 8 * KIB, 1);
    let blob = builder.add_blob(raw, whole(), KIB_U32, StreamPolicy::Store);
    let kib = KIB as u64;
    let extents: Vec<ExtentSpec> =
        [7, 5, 7, 4, 1, 0].into_iter().map(|chunk| identity(blob, chunk * kib, kib)).collect();
    let content = builder.add_member("m", &extents);
    let (container, source) = recording(builder.finish(), 3);
    assert_read(&container.single().expect("one member"), &content, 0, content.len());
    let stored = KIB + 4;
    let at = |chunk: usize| (64 + chunk * stored) as u64;
    // Batch one holds chunks 7, 5 and 4, chunk 7 once; chunk 6 is not in it. Batch two holds 1 and 0.
    let reads = source.take_reads();
    assert_eq!(reads, [(at(4), 2 * stored), (at(7), stored), (at(0), 2 * stored)]);
    let batch_bytes = container.shared.engine.batch_bytes;
    assert!(reads.iter().all(|(_, len)| *len <= batch_bytes), "{reads:?} over {batch_bytes}");
}

#[test]
fn a_delta_chunk_that_shares_its_base_s_stored_bytes_reads_them_once() {
    let mut builder = ContainerBuilder::new();
    let base = builder.add_blob(tensor_like(28, 2 * KIB, 1), whole(), KIB_U32, StreamPolicy::Store);
    let blob = builder.share_chunks_as_delta(base, delta(ElementWidth::Two, whole()));
    let zeros = builder.add_member("m", &[identity(blob, 0, 2 * KIB as u64)]);
    let file = builder.finish();
    // The chunk hashes are the base's, so a checked read of the zero target fails at the delta blob.
    let member = open(file.clone(), 1).single().expect("one member");
    let error = member.read_at(0, &mut [0; 8]).expect_err("hashes of the base");
    assert!(matches!(error, Error::ChecksumMismatch { blob: 1, chunk: 0, .. }), "{error:?}");
    let source = Arc::new(MemorySource::new(file));
    let shared: Arc<dyn RandomAccess> = source.clone();
    let off = OpenOptions { threads: 1, verify: Verify::Off, ..OpenOptions::default() };
    let member = Container::from_source(shared, &off).expect("valid").single().expect("one member");
    source.take_reads();
    let mut all = vec![0xFF; zeros.len()];
    member.read_at(0, &mut all).expect("reads");
    assert_eq!(all, zeros);
    assert_eq!(source.take_reads(), [(64, KIB + 4), ((64 + KIB + 4) as u64, KIB + 4)]);
}

#[test]
fn a_delta_file_without_its_feature_bit_is_refused_at_open() {
    let (file, _) = stored_delta();
    let mut header: [u8; 64] = file[..64].try_into().expect("a header");
    let parsed = crate::format::parse_header(&header).expect("valid");
    assert_eq!(parsed.required_features, crate::chain::FEATURE_DELTA);
    header.copy_from_slice(&Header::current().to_bytes());
    let file = [&header[..], &file[64..]].concat();
    let error = open_with(file, &options(1)).expect_err("bit 0 is missing");
    assert!(matches!(error, Error::MalformedIndex { section: "BLOB", .. }), "{error:?}");
}

#[test]
fn delta_decode_work_counts_the_base_at_open() {
    // One-byte-wide extents over a delta blob cost both chunks each, so half as many are allowed.
    let mut builder = ContainerBuilder::new();
    let base_raw = tensor_like(29, 1 << 20, 1);
    let base = builder.add_blob(base_raw.clone(), whole(), 1 << 20, StreamPolicy::Store);
    let chain = delta(ElementWidth::Two, whole());
    let blob = builder.add_delta_blob(finetune_of(30, &base_raw), base, 0, chain, StreamPolicy::Store);
    let extents: Vec<ExtentSpec> = (0..200).map(|at| identity(blob, at * 2, 2)).collect();
    builder.add_member("m", &extents);
    let file = builder.finish();
    let error = open_with(file.clone(), &options(1)).expect_err("400 MiB of work for 400 bytes");
    assert!(
        matches!(error, Error::BudgetExceeded { what: "decode work", declared, .. } if declared == 400 << 20),
        "{error:?}"
    );
    let generous = OpenOptions { max_decode_amplification: 1 << 20, ..options(1) };
    open_with(file, &generous).expect("a caller may accept it");
}
