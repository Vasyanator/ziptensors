//! File: write/tests.rs
//! Purpose: writer tests that need a test-only seam. Everything a public caller can do is in
//! `tests/roundtrip.rs`.
//!
//! Implementation notes: the delta case drives `ingest` directly, since a file that shrinks between
//! its open and its read cannot be arranged through `PackBuilder`. Its safetensors inputs are built
//! here: `tests/common/fixtures.rs` is already a module of `profile::tests`, and a file may be a
//! module only once.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use crate::chain::{BLOB_KIND_DELTA, FEATURE_DELTA};
use crate::error::Error;
use crate::format::{Header, IndexBuilder};
use crate::io::{FileSource, RandomAccess};
use crate::profile::{normalize, scan};
use crate::read::{Container, OpenOptions};
use crate::types::{Dtype, Mode, TypedRange};

use super::commit::{Sink, write_index};
use super::dedup::BlobTable;
use super::ingest::{Compressor, DeltaInput, IngestContext, ingest};
use super::pairing::BaseCatalog;
use super::report::unreferenced_table;
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
    let error = ingest(&source, &ranges, context, &mut sink, &mut index).expect_err("shrank");
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
        ingest(&base_source, &base_plan.ranges, context, &mut sink, &mut index).expect("base");
    let blobs_of_base = next_blob;

    let tuned_source = FileSource::open(&tuned_path).expect("opens");
    let tuned_plan = normalize(scan(&tuned_source).expect("scans"), tuned_source.len());
    let bases = BaseCatalog::new(&base_plan.tensors, &base_ingested.extents).bases_of(&tuned_plan);
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
    let error = ingest(&tuned_source, &tuned_plan.ranges, context, &mut sink, &mut index)
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
    write_index(&mut sink, &index).expect("writes the index");
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
