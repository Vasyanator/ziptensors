//! File: tree/tests.rs
//! Purpose: the crash order of `compress_tree` and `decompress_tree`, through the crate-private
//! fault seam, and the report arithmetic.
//!
//! Main responsibilities: interrupt a run after the verified commit and after the removal, check
//! that the tree then holds the original, the twin, or both and never neither, and that a rerun
//! converges to the state of an uninterrupted run and then does nothing.
//!
//! Implementation notes: a fault fails the file it interrupts, as a crash at that point would
//! leave it; the run goes on with the next file, which does not change what is checked here.
//! The weight file is a minimal safetensors file built here: `tests/common/fixtures.rs` is already
//! a module of the profile tests in this crate, and clippy refuses loading it twice. Scratch trees live under the system temp
//! directory and are removed.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use super::compress::compress_tree_with;
use super::decompress::decompress_tree_with;
use super::{
    CompressTreeOptions, DecompressTreeOptions, FileOutcome, FileStatus, Stage, TreeReport,
    no_fault,
};
use crate::error::Error;
use crate::types::Mode;
use crate::write::EncodeOptions;

/// A unique directory under the system temp directory, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let name = format!("ziptensors-tree-unit-{tag}-{}-{id}", std::process::id());
        let path = std::env::temp_dir().join(name);
        fs::create_dir_all(&path).expect("scratch directory");
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("cannot remove {}: {error}", self.0.display());
        }
    }
}

/// A safetensors file of one F32 tensor of 1024 elements whose bytes depend on `seed`.
fn weight(seed: u64) -> Vec<u8> {
    let mut header =
        br#"{"w":{"dtype":"F32","shape":[1024],"data_offsets":[0,4096]}}"#.to_vec();
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut bytes = u64::try_from(header.len()).expect("a short header").to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    let mut state = seed;
    for _ in 0..1024_u32 {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let value = f32::from(u16::try_from(state >> 52).expect("12 bits")) / 4096.0;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn compress_options() -> CompressTreeOptions {
    CompressTreeOptions {
        encode: EncodeOptions { mode: Mode::Balanced, threads: 1 },
        ..CompressTreeOptions::default()
    }
}

fn decompress_options() -> DecompressTreeOptions {
    DecompressTreeOptions { threads: 1, ..DecompressTreeOptions::default() }
}

fn crash_after_commit(stage: Stage, path: &Path) -> Result<(), Error> {
    match stage {
        Stage::Committed => Err(Error::Io {
            op: "simulated crash",
            path: path.to_path_buf(),
            source: std::io::Error::other("crash after commit"),
        }),
        Stage::Removed => Ok(()),
    }
}

fn crash_after_removal(stage: Stage, path: &Path) -> Result<(), Error> {
    match stage {
        Stage::Committed => Ok(()),
        Stage::Removed => Err(Error::Io {
            op: "simulated crash",
            path: path.to_path_buf(),
            source: std::io::Error::other("crash after removal"),
        }),
    }
}

fn outcome<'a>(report: &'a TreeReport, path: &Path) -> &'a FileOutcome {
    let found = report.files.iter().find(|outcome| outcome.path == path);
    found.unwrap_or_else(|| panic!("no outcome for {}: {report:#?}", path.display()))
}

#[test]
fn compress_converges_after_a_crash_at_every_stage() {
    let scratch = Scratch::new("compress");
    let original = scratch.0.join("m.safetensors");
    let container = scratch.0.join("m.ziptensors");
    let bytes = weight(1);
    fs::write(&original, &bytes).expect("write");

    let report = compress_tree_with(&scratch.0, &compress_options(), crash_after_commit)
        .expect("runs");
    let interrupted = outcome(&report, &original);
    assert!(matches!(interrupted.status, FileStatus::Failed(_)), "{report:#?}");
    assert!(interrupted.removed.is_none());
    assert!(original.exists() && container.exists(), "both copies after a crash after commit");

    let report = compress_tree_with(&scratch.0, &compress_options(), crash_after_removal)
        .expect("reruns");
    let interrupted = outcome(&report, &original);
    assert!(matches!(interrupted.status, FileStatus::Failed(_)), "{report:#?}");
    assert_eq!(interrupted.removed.as_deref(), Some(original.as_path()));
    assert!(!original.exists() && container.exists(), "the container alone after the removal");

    let report = compress_tree_with(&scratch.0, &compress_options(), no_fault).expect("reruns");
    assert!(matches!(outcome(&report, &container).status, FileStatus::AlreadyCompressed));
    assert_eq!(report.files.len(), 1);

    let report = decompress_tree_with(&scratch.0, &decompress_options(), no_fault).expect("restores");
    assert!(matches!(outcome(&report, &container).status, FileStatus::Restored));
    assert_eq!(fs::read(&original).expect("restored"), bytes);
}

#[test]
fn a_crash_after_commit_of_a_stale_pair_keeps_the_original() {
    let scratch = Scratch::new("stale");
    let original = scratch.0.join("m.safetensors");
    let container = scratch.0.join("m.ziptensors");
    fs::write(&original, weight(2)).expect("write");
    let mut keep = compress_options();
    keep.keep = true;
    compress_tree_with(&scratch.0, &keep, no_fault).expect("compresses");
    fs::write(&original, weight(3)).expect("the original changes");

    let report = compress_tree_with(&scratch.0, &compress_options(), crash_after_commit)
        .expect("reruns");
    assert!(matches!(outcome(&report, &original).status, FileStatus::Failed(_)), "{report:#?}");
    assert_eq!(fs::read(&original).expect("kept"), weight(3));

    // The new container is committed, so the pair now verifies.
    let report = compress_tree_with(&scratch.0, &compress_options(), no_fault).expect("reruns");
    assert!(matches!(outcome(&report, &original).status, FileStatus::PairResolved), "{report:#?}");
    assert!(!original.exists() && container.exists());
}

#[test]
fn decompress_converges_after_a_crash_at_every_stage() {
    let scratch = Scratch::new("decompress");
    let original = scratch.0.join("m.safetensors");
    let container = scratch.0.join("m.ziptensors");
    let bytes = weight(4);
    fs::write(&original, &bytes).expect("write");
    compress_tree_with(&scratch.0, &compress_options(), no_fault).expect("compresses");

    let report = decompress_tree_with(&scratch.0, &decompress_options(), crash_after_commit)
        .expect("runs");
    assert!(matches!(outcome(&report, &container).status, FileStatus::Failed(_)), "{report:#?}");
    assert!(original.exists() && container.exists());

    let report = decompress_tree_with(&scratch.0, &decompress_options(), crash_after_removal)
        .expect("reruns");
    let interrupted = outcome(&report, &container);
    assert!(matches!(interrupted.status, FileStatus::Failed(_)), "{report:#?}");
    assert_eq!(interrupted.removed.as_deref(), Some(container.as_path()));
    assert!(original.exists() && !container.exists());

    let report = decompress_tree_with(&scratch.0, &decompress_options(), no_fault).expect("reruns");
    assert!(report.files.is_empty(), "{report:#?}");
    assert_eq!(fs::read(&original).expect("restored"), bytes);
}

#[test]
fn totals_count_statuses_and_sum_sizes_of_proven_files_in_either_direction() {
    let line = |status, original, container| FileOutcome {
        path: PathBuf::from("x"),
        counterpart: None,
        status,
        original_len: original,
        container_len: container,
        removed: None,
    };
    let report = TreeReport {
        root: PathBuf::from("r"),
        dry_run: false,
        files: vec![
            line(FileStatus::Compressed, Some(100), Some(40)),
            line(FileStatus::PairResolved, Some(10), Some(5)),
            line(FileStatus::Restored, Some(1000), Some(400)),
            line(FileStatus::AlreadyCompressed, None, Some(7)),
            line(FileStatus::NotWeight, None, None),
            line(FileStatus::Failed(Error::NotZiptensors), Some(1), Some(1)),
        ],
        warnings: Vec::new(),
    };
    let totals = report.totals();
    assert_eq!(
        (totals.compressed, totals.pair_resolved, totals.already_compressed, totals.not_weight),
        (1, 1, 1, 1)
    );
    assert_eq!((totals.restored, totals.failed), (1, 1));
    // A restore counts like a compress: decompress reports the bytes it brought back.
    assert_eq!((totals.original_bytes, totals.container_bytes), (1110, 445));
    assert!(report.has_failures());
}
