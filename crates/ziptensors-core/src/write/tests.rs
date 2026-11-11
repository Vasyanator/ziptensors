//! File: write/tests.rs
//! Purpose: writer tests that need a test-only seam. Everything a public caller can do is in
//! `tests/roundtrip.rs`.
//!
//! Also the crate-private input seam: `PackBuilder::add_source` over a source that is no file, and
//! `BaseCatalog::from_member` with `DeltaSpec::Member`, each proven against the public file path by
//! comparing output bytes.
//!
//! Implementation notes: the delta case drives `ingest` directly, since a file that shrinks between
//! its open and its read cannot be arranged through `PackBuilder`. Its safetensors inputs are built
//! here: `tests/common/fixtures.rs` is already a module of `profile::tests`, and a file may be a
//! module only once.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::chain::{BLOB_KIND_DELTA, FEATURE_DELTA};
use crate::error::Error;
use crate::format::{Header, IndexBuilder};
use crate::io::{FileSource, RandomAccess};
use crate::profile::{normalize, scan};
use crate::read::testkit::MemorySource;
use crate::read::{Container, OpenOptions};
use crate::types::{Dtype, Mode, TypedRange};

use super::commit::{Sink, write_index};
use super::dedup::BlobTable;
use super::ingest::{Compressor, DeltaInput, IngestContext, ingest};
use super::pairing::BaseCatalog;
use super::report::unreferenced_table;
use super::builder::{DeltaSpec, InputStamp, SourceStamp};
use super::{EncodeOptions, PackBuilder};

/// A unique scratch directory, removed on drop.
struct Dir(PathBuf);

impl Dir {
    fn new() -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let name = format!("ziptensors-write-{}-{id}", std::process::id());
        let path = std::env::temp_dir().join(name);
        fs::create_dir_all(&path).expect("scratch directory");
        Self(path)
    }

    fn entries(&self) -> Vec<String> {
        let entries = fs::read_dir(&self.0).expect("readable");
        entries.map(|entry| entry.expect("entry").file_name().to_string_lossy().into_owned()).collect()
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("cannot remove {}: {error}", self.0.display());
        }
    }
}

/// Flips one bit in the first chunk, past the header.
fn flip_data_bit(path: &Path) {
    let mut bytes = fs::read(path).expect("temporary file exists");
    bytes[64 + 8] ^= 0x10;
    fs::write(path, bytes).expect("temporary file is writable");
}

/// Zeroes the last byte, which is part of the footer's magic.
fn break_footer(path: &Path) {
    let mut bytes = fs::read(path).expect("temporary file exists");
    let last = bytes.len() - 2;
    bytes[last] ^= 0xFF;
    fs::write(path, bytes).expect("temporary file is writable");
}

fn pack_with_fault(fault: fn(&Path)) -> (Dir, Result<crate::write::EncodeReport, Error>) {
    let dir = Dir::new();
    let input = dir.0.join("input.bin");
    fs::write(&input, vec![0x5A; 300_000]).expect("input");
    let output = dir.0.join("out.ziptensors");
    let options = EncodeOptions { mode: Mode::Max, threads: 2 };
    let mut builder = PackBuilder::new(&output, &options).expect("starts");
    builder.fault = Some(fault);
    builder.add_file(&input, "input.bin").expect("adds");
    let result = builder.finish();
    (dir, result)
}

#[test]
fn corruption_of_a_chunk_fails_verification_and_leaves_no_output() {
    let (dir, result) = pack_with_fault(flip_data_bit);
    let error = result.expect_err("a flipped chunk bit must not commit");
    assert!(
        matches!(error, Error::ChecksumMismatch { .. } | Error::ChunkCorrupt { .. }),
        "{error:?}"
    );
    assert_eq!(dir.entries(), ["input.bin"]);
}

#[test]
fn corruption_of_the_footer_fails_verification_and_leaves_no_output() {
    let (dir, result) = pack_with_fault(break_footer);
    assert!(result.is_err());
    assert_eq!(dir.entries(), ["input.bin"]);
}

#[test]
fn an_uncorrupted_file_passes_the_same_seam() {
    let (dir, result) = pack_with_fault(|_| {});
    result.expect("commits");
    assert_eq!(dir.entries().len(), 2);
}

#[test]
fn a_file_that_shrinks_under_the_read_is_input_changed_and_rolls_back() {
    let dir = Dir::new();
    let input = dir.0.join("input.bin");
    fs::write(&input, vec![0x5A; 3 << 20]).expect("input");
    let source = FileSource::open(&input).expect("opens");
    fs::write(&input, vec![0x5A; 3 << 19]).expect("shrinks");
    let ranges = [TypedRange { offset: 0, len: 3 << 20, hint: Dtype::U8 }];

    let output = dir.0.join("out");
    let mut sink = Sink::create(&output).expect("creates");
    let mut index = IndexBuilder::new();
    let (mut blobs, mut next_blob) = (BlobTable::default(), 0);
    let compressor = Compressor::new(2, &output).expect("starts");
    let context = IngestContext {
        compressor: &compressor,
        mode: Mode::Max,
        blobs: &mut blobs,
        next_blob: &mut next_blob,
        delta: None,
    };
    let error =
        ingest(&source, &input, &ranges, context, &mut sink, &mut index).expect_err("shrank");
    assert!(matches!(error, Error::InputChanged { .. }), "{error:?}");
    assert_eq!(sink.position(), 64, "the open blob must be gone");
    assert_eq!(next_blob, 0);
    sink.check().expect("the output is still fine");
}

/// A safetensors file of two `BF16` tensors `a` and `b` of `len` bytes each, of seeded values with the
/// byte statistics of weights, and the offset where their data starts.
fn two_bf16_tensors(len: usize) -> (Vec<u8>, usize) {
    let header = format!(
        r#"{{"a":{{"dtype":"BF16","shape":[{n}],"data_offsets":[0,{len}]}},"b":{{"dtype":"BF16","shape":[{n}],"data_offsets":[{len},{end}]}}}}"#,
        n = len / 2,
        end = 2 * len,
    );
    let mut header = header.into_bytes();
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut bytes = u64::try_from(header.len()).expect("fits").to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    let data = bytes.len();
    // Xorshift: a sign and a few exponents in the high byte, noise in the low byte.
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let [low, high, ..] = state.to_le_bytes();
        bytes.extend_from_slice(&[low, 0x3C | (high & 0x83)]);
    }
    (bytes, data)
}

#[test]
fn a_delta_add_that_fails_after_sealing_a_delta_blob_leaves_a_consistent_pack() {
    let dir = Dir::new();
    let (base, data) = two_bf16_tensors((2 << 20) + (1 << 19));
    let mut tuned = base.clone();
    tuned[data..].iter_mut().step_by(14).for_each(|byte| *byte ^= 1);
    let base_path = dir.0.join("base");
    let tuned_path = dir.0.join("tuned");
    fs::write(&base_path, &base).expect("base");
    fs::write(&tuned_path, &tuned).expect("tuned");

    let output = dir.0.join("out");
    let mut sink = Sink::create(&output).expect("creates");
    let mut index = IndexBuilder::new();
    let (mut blobs, mut next_blob) = (BlobTable::default(), 0);
    let compressor = Compressor::new(2, &output).expect("starts");
    let base_source = FileSource::open(&base_path).expect("opens");
    let base_plan = normalize(scan(&base_source).expect("scans"), base_source.len());
    let context = IngestContext {
        compressor: &compressor,
        mode: Mode::Max,
        blobs: &mut blobs,
        next_blob: &mut next_blob,
        delta: None,
    };
    let base_ingested =
        ingest(&base_source, &base_path, &base_plan.ranges, context, &mut sink, &mut index)
            .expect("base");
    let blobs_of_base = next_blob;

    let tuned_source = FileSource::open(&tuned_path).expect("opens");
    let tuned_plan = normalize(scan(&tuned_source).expect("scans"), tuned_source.len());
    let bases = BaseCatalog::from_plan(&base_plan, &base_ingested.extents).bases_of(&tuned_plan);
    assert_eq!(bases.iter().flatten().count(), 2);
    // The second tensor cannot be read whole: the first one is sealed as a delta blob, then the
    // read fails.
    fs::write(&tuned_path, &tuned[..tuned.len() - 1000]).expect("shrinks");
    let context = IngestContext {
        compressor: &compressor,
        mode: Mode::Max,
        blobs: &mut blobs,
        next_blob: &mut next_blob,
        delta: Some(DeltaInput { source: &base_source, bases: &bases }),
    };
    let error =
        ingest(&tuned_source, &tuned_path, &tuned_plan.ranges, context, &mut sink, &mut index)
            .expect_err("shrank");
    assert!(matches!(error, Error::InputChanged { .. }), "{error:?}");
    // The header is a duplicate of the base's, so one delta blob is all that was sealed.
    assert_eq!(next_blob, blobs_of_base + 1);
    assert_eq!(blobs.delta_count(), 1, "what decides bit 0 at finish");
    sink.check().expect("the output is still fine");

    // What `finish` does with such a pack: the bit, the index, the real reader.
    let data_end = sink.position();
    let header = Header { required_features: FEATURE_DELTA, ..Header::current() };
    sink.patch_header(&header).expect("patches");
    write_index(&mut sink, &index, crate::format::GENERATION_INITIAL).expect("writes the index");
    let container = Container::open(sink.path(), &OpenOptions::default()).expect("opens");
    let records = container.index();
    let delta = records.blobs().iter().find(|blob| blob.kind == BLOB_KIND_DELTA).expect("sealed");
    assert!(records.delta_params(delta).expect("valid").base_blob < blobs_of_base);
    // No member reaches any blob, and the report still accounts for the whole data region.
    let unreferenced = unreferenced_table(records).expect("accounts").total();
    assert_eq!(unreferenced.stored, data_end - 64);
    assert_eq!(unreferenced.overhead, unreferenced.stored);
    drop(container);

    // The same file without the bit is refused: bit 0 and kind 1 go together.
    sink.patch_header(&Header::current()).expect("patches");
    let error = Container::open(sink.path(), &OpenOptions::default()).expect_err("no bit");
    assert!(matches!(error, Error::MalformedIndex { .. }), "{error:?}");
}

/// A base of [`two_bf16_tensors`] and a finetune of it with a sparse change in every tensor, so that
/// both tensors pair and store as delta blobs in `max`.
fn base_and_tuned() -> (Vec<u8>, Vec<u8>) {
    let (base, data) = two_bf16_tensors((2 << 20) + (1 << 19));
    let mut tuned = base.clone();
    tuned[data..].iter_mut().step_by(14).for_each(|byte| *byte ^= 1);
    (base, tuned)
}

/// The stamp of a source that is no file holding `bytes`.
fn member_stamp(bytes: &[u8]) -> SourceStamp {
    let len = u64::try_from(bytes.len()).expect("fits");
    SourceStamp::Member { len, blake3: *blake3::hash(bytes).as_bytes() }
}

const MAX_2: EncodeOptions = EncodeOptions { mode: Mode::Max, threads: 2 };

/// Packs the files `base` and `tuned` at `output` the public way: `tuned` against `base`.
fn pack_files_with_delta(output: &Path, base: &Path, tuned: &Path) -> crate::write::EncodeReport {
    let mut builder = PackBuilder::new(output, &MAX_2).expect("starts");
    builder.add_file(base, "base").expect("adds the base");
    let report = builder.add_file_delta(tuned, "tuned", "base").expect("adds against the base");
    assert_eq!(report.delta_blobs, 2, "both tensors must take the delta path");
    builder.finish().expect("commits")
}

#[test]
fn add_source_over_a_memory_source_writes_the_bytes_of_add_file() {
    let dir = Dir::new();
    let (base, tuned) = base_and_tuned();
    let base_path = dir.0.join("base");
    let tuned_path = dir.0.join("tuned");
    fs::write(&base_path, &base).expect("base");
    fs::write(&tuned_path, &tuned).expect("tuned");
    let from_files = dir.0.join("files.ziptensors");
    pack_files_with_delta(&from_files, &base_path, &tuned_path);

    // The same inputs as sources that are no files; the base is read again from its kept source.
    let from_memory = dir.0.join("memory.ziptensors");
    let mut builder = PackBuilder::new(&from_memory, &MAX_2).expect("starts");
    let source = Arc::new(MemorySource::silent(base.clone()));
    builder.add_source(source, "base", member_stamp(&base), None).expect("adds the base");
    let source = Arc::new(MemorySource::silent(tuned.clone()));
    let report = builder
        .add_source(source, "tuned", member_stamp(&tuned), Some(&DeltaSpec::Added("base")))
        .expect("adds against the base");
    assert_eq!(report.delta_blobs, 2);
    builder.finish().expect("commits");

    let files = fs::read(&from_files).expect("readable");
    assert!(files == fs::read(&from_memory).expect("readable"), "outputs differ");
}

#[test]
fn a_source_that_differs_from_its_stamp_is_verify_failed_and_adds_nothing() {
    let dir = Dir::new();
    let (base, tuned) = base_and_tuned();
    let output = dir.0.join("out.ziptensors");
    let mut builder = PackBuilder::new(&output, &MAX_2).expect("starts");
    let source = Arc::new(MemorySource::silent(base.clone()));
    let error = builder
        .add_source(Arc::clone(&source) as Arc<dyn RandomAccess>, "base", member_stamp(&tuned), None)
        .expect_err("the stamp is of other bytes");
    assert!(matches!(&error, Error::VerifyFailed { member, .. } if member == "base"), "{error:?}");
    // Nothing was recorded under the name, and the builder goes on.
    let error = builder
        .add_file_delta(&dir.0.join("absent"), "tuned", "base")
        .expect_err("no such base");
    assert!(matches!(error, Error::MemberNotFound { .. }), "{error:?}");
    builder.add_source(source, "base", member_stamp(&base), None).expect("adds");
    builder.finish().expect("commits");
    let container = Container::open(&output, &OpenOptions::default()).expect("opens");
    let len = u64::try_from(base.len()).expect("fits");
    assert_eq!(container.member("base").expect("present").record().file_len, len);
}

#[test]
fn from_member_pairs_like_from_plan_and_serves_as_an_outside_base() {
    let dir = Dir::new();
    let (base, tuned) = base_and_tuned();
    let base_path = dir.0.join("base");
    let tuned_path = dir.0.join("tuned");
    fs::write(&base_path, &base).expect("base");
    fs::write(&tuned_path, &tuned).expect("tuned");
    let packed = dir.0.join("packed.ziptensors");
    pack_files_with_delta(&packed, &base_path, &tuned_path);

    let container = Container::open(&packed, &OpenOptions::default()).expect("opens");
    let index = container.index();
    let record = *container.member("base").expect("present").record();
    let from_member = BaseCatalog::from_member(index, &record).expect("valid records");
    let base_source = FileSource::open(&base_path).expect("opens");
    let base_plan = normalize(scan(&base_source).expect("scans"), base_source.len());
    let extents = index.member_extents(&record).expect("valid records");
    let from_plan = BaseCatalog::from_plan(&base_plan, extents);
    assert_eq!(from_member.len(), 2);
    assert_eq!(from_member, from_plan);
    let tuned_source = FileSource::open(&tuned_path).expect("opens");
    let tuned_plan = normalize(scan(&tuned_source).expect("scans"), tuned_source.len());
    let bases = from_member.bases_of(&tuned_plan);
    assert_eq!(bases.iter().flatten().count(), 2);
    assert_eq!(bases, from_plan.bases_of(&tuned_plan));

    // A builder that adds the base file first gives its blobs the ids they have in `packed`, so the
    // catalog of the packed member names the right blobs, and the result is byte for byte the file
    // that named the base by its member name.
    let outside = dir.0.join("outside.ziptensors");
    let mut builder = PackBuilder::new(&outside, &MAX_2).expect("starts");
    builder.add_file(&base_path, "base").expect("adds the base");
    let delta = DeltaSpec::Member {
        name: "base",
        source: Arc::new(FileSource::open(&base_path).expect("opens")),
        catalog: &from_member,
    };
    let stamp = SourceStamp::File(InputStamp::of(&tuned_path).expect("stamps"));
    let report = builder
        .add_source(Arc::new(tuned_source), "tuned", stamp, Some(&delta))
        .expect("adds against the outside base");
    assert_eq!(report.delta_blobs, 2);
    drop(delta);
    builder.finish().expect("commits");
    drop(container);
    let packed = fs::read(&packed).expect("readable");
    assert!(packed == fs::read(&outside).expect("readable"), "outputs differ");
}
