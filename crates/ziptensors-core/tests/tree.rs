//! File: tests/tree.rs
//! Purpose: `compress_tree`, `decompress_tree` and `scan_view` through the public API, over
//! generated trees.
//!
//! Main responsibilities:
//! - only weight files become containers, everything else stays byte-identical, and the round trip
//!   restores the tree exactly;
//! - collisions refuse the whole run and leave the tree unchanged;
//! - reruns: idempotent, a pair is resolved after hashing, a stale container is replaced, a damaged
//!   one keeps the original;
//! - links (skipped with a warning by default, followed on request), special files, partial and
//!   temporary files, names that are not UTF-8;
//! - decompress: conflicts, a different file at the restore path, `force`, `keep`, foreign
//!   containers.
//!
//! Key structures: `Scratch`, `Node`.
//!
//! Implementation notes: every tree is generated from a seed under a unique directory of the system
//! temp directory, removed at the end. No test reads a real model. Interruptions between a commit
//! and a removal need the fault seam and are tested in `src/tree/tests.rs`; here they are
//! reproduced by building the state they leave.

#[path = "common/fixtures.rs"]
mod fixtures;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use fixtures::{Elem, GgufOptions, TensorSpec};
use ziptensors_core::tree::{Action, ConflictReason, ForeignReason, SkipReason};
use ziptensors_core::{
    CompressTreeOptions, Container, DecompressTreeOptions, EncodeOptions, Error, FileStatus, Mode,
    OpenOptions, PackBuilder, TreeReport, ViewEntry, compress_tree, decompress_tree, scan_view,
};

/// A unique directory under the system temp directory, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("ziptensors-tree-{tag}-{}-{id}", std::process::id()));
        fs::create_dir_all(&path).expect("scratch directory");
        Self(path)
    }

    fn path(&self, rel: &str) -> PathBuf {
        self.0.join(rel)
    }

    fn write(&self, rel: &str, bytes: &[u8]) -> PathBuf {
        let path = self.path(rel);
        fs::create_dir_all(path.parent().expect("a parent")).expect("parent directory");
        fs::write(&path, bytes).expect("write file");
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("cannot remove {}: {error}", self.0.display());
        }
    }
}

/// What is at one path of a tree.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Node {
    File(Vec<u8>),
    Dir,
    Link(PathBuf),
    Other,
}

/// Every path under `root` with what is there, links not followed.
fn snapshot(root: &Path) -> BTreeMap<String, Node> {
    let mut nodes = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir).expect("readable") {
            let path = entry.expect("entry").path();
            let rel = path.strip_prefix(root).expect("under root").to_string_lossy().replace('\\', "/");
            let meta = fs::symlink_metadata(&path).expect("metadata");
            let node = if meta.is_symlink() {
                Node::Link(fs::read_link(&path).expect("link"))
            } else if meta.is_dir() {
                pending.push(path.clone());
                Node::Dir
            } else if meta.is_file() {
                Node::File(fs::read(&path).expect("read"))
            } else {
                Node::Other
            };
            nodes.insert(rel, node);
        }
    }
    nodes
}

fn safetensors(seed: u64) -> Vec<u8> {
    fixtures::safetensors(
        seed,
        &[
            TensorSpec::new("a.weight", &[64, 64], Elem::F32),
            TensorSpec::new("b.weight", &[128, 32], Elem::F16),
            TensorSpec::new("c.bias", &[64], Elem::Bf16),
        ],
    )
    .bytes
}

fn gguf(seed: u64) -> Vec<u8> {
    fixtures::gguf(
        seed,
        &[TensorSpec::new("v", &[256], Elem::F32), TensorSpec::new("q", &[256, 4], Elem::Q80)],
        &GgufOptions::default(),
    )
    .bytes
}

/// A legacy `torch.save` stream: protocol 2 pickle of torch's magic number, then other pickles.
fn legacy_torch(seed: u64) -> Vec<u8> {
    let mut bytes = vec![0x80, 0x02, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19, 0x2e];
    bytes.extend_from_slice(&[0x80, 0x02, 0x4d, 0xe9, 0x03, 0x2e]);
    bytes.extend(fixtures::Rng::new(seed).bytes(5000));
    bytes
}

fn png(seed: u64) -> Vec<u8> {
    let mut bytes = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    bytes.extend(fixtures::Rng::new(seed).bytes(3000));
    bytes
}

fn compress_options() -> CompressTreeOptions {
    let mut options = CompressTreeOptions::default();
    options.encode = EncodeOptions { mode: Mode::Max, threads: 2 };
    options
}

fn decompress_options() -> DecompressTreeOptions {
    let mut options = DecompressTreeOptions::default();
    options.threads = 2;
    options
}

/// The status of the outcome about `path`.
fn status_of<'a>(report: &'a TreeReport, path: &Path) -> &'a FileStatus {
    let outcome = report.files.iter().find(|outcome| outcome.path == path);
    &outcome.unwrap_or_else(|| panic!("no outcome for {}: {report:#?}", path.display())).status
}

/// Writes a container at `out` holding the file `src` under `member`.
fn container_of(src: &Path, out: &Path, member: &str) {
    let mut builder = PackBuilder::new(out, &EncodeOptions { mode: Mode::Fast, threads: 1 }).expect("builder");
    builder.add_file(src, member).expect("add");
    builder.finish().expect("finish");
}

/// The bytes the single member of the container at `path` restores to, and its name.
fn restored(path: &Path) -> (String, Vec<u8>) {
    let container = Container::open(path, &OpenOptions { threads: 1, ..OpenOptions::default() }).expect("opens");
    let member = container.single().expect("one member");
    let mut bytes = Vec::new();
    member.decode_to(&mut bytes).expect("decodes");
    (member.name().expect("name").to_owned(), bytes)
}

/// A mixed model tree: weights of every kind, configs, an index, an image, nested directories.
fn model_tree(scratch: &Scratch) {
    scratch.write("model_index.json", br#"{"vae": ["diffusers", "AutoencoderKL"]}"#);
    scratch.write("vae/config.json", br#"{"latent_channels": 4}"#);
    scratch.write("vae/diffusion_pytorch_model.safetensors", &safetensors(1));
    scratch.write("unet/model.safetensors.index.json", br#"{"weight_map": {}}"#);
    scratch.write("unet/model-00001-of-00002.safetensors", &safetensors(2));
    scratch.write("unet/model-00002-of-00002.safetensors", &safetensors(3));
    scratch.write("llm/model-q8.gguf", &gguf(4));
    scratch.write("old/pytorch_model.bin", &legacy_torch(5));
    scratch.write("old/weights", &safetensors(6));
    scratch.write("assets/preview.png", &png(7));
    scratch.write("README.md", b"# a model\n");
}

#[test]
fn only_weight_files_become_containers_and_the_round_trip_is_exact() {
    let scratch = Scratch::new("mixed");
    model_tree(&scratch);
    let before = snapshot(&scratch.0);

    let report = compress_tree(&scratch.0, &compress_options()).expect("compresses");
    assert!(!report.has_failures(), "{report:#?}");
    let totals = report.totals();
    assert_eq!(totals.compressed, 6, "{report:#?}");
    assert_eq!(totals.not_weight, 5, "{report:#?}");
    assert!(totals.container_bytes < totals.original_bytes, "{totals:?}");

    let after = snapshot(&scratch.0);
    for (weight, container) in [
        ("vae/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.ziptensors"),
        ("unet/model-00001-of-00002.safetensors", "unet/model-00001-of-00002.ziptensors"),
        ("unet/model-00002-of-00002.safetensors", "unet/model-00002-of-00002.ziptensors"),
        ("llm/model-q8.gguf", "llm/model-q8.ziptensors"),
        ("old/pytorch_model.bin", "old/pytorch_model.ziptensors"),
        ("old/weights", "old/weights.ziptensors"),
    ] {
        assert!(!after.contains_key(weight), "{weight} was removed");
        let (name, bytes) = restored(&scratch.path(container));
        assert_eq!(name, Path::new(weight).file_name().expect("name").to_string_lossy());
        assert_eq!(Node::File(bytes), before[weight], "{weight}");
    }
    for plain in ["model_index.json", "vae/config.json", "unet/model.safetensors.index.json", "assets/preview.png", "README.md"] {
        assert_eq!(after[plain], before[plain], "{plain} untouched");
        assert!(matches!(status_of(&report, &scratch.path(plain)), FileStatus::NotWeight));
    }

    // A rerun has nothing to do.
    let again = compress_tree(&scratch.0, &compress_options()).expect("reruns");
    assert_eq!(again.totals().already_compressed, 6, "{again:#?}");
    assert_eq!(again.totals().compressed + again.totals().failed, 0, "{again:#?}");
    assert_eq!(snapshot(&scratch.0), after);

    let report = decompress_tree(&scratch.0, &decompress_options()).expect("decompresses");
    let restored_totals = report.totals();
    assert_eq!(restored_totals.restored, 6, "{report:#?}");
    // Decompress sums what it brought back and what it read, the same bytes compress counted.
    assert_eq!(
        (restored_totals.original_bytes, restored_totals.container_bytes),
        (totals.original_bytes, totals.container_bytes),
        "{report:#?}"
    );
    assert_eq!(snapshot(&scratch.0), before);
    let again = decompress_tree(&scratch.0, &decompress_options()).expect("reruns");
    assert!(again.files.is_empty(), "{again:#?}");
}

#[test]
fn keep_leaves_both_copies_and_a_rerun_verifies_the_pair() {
    let scratch = Scratch::new("keep");
    let weight = scratch.write("m.safetensors", &safetensors(11));
    let mut options = compress_options();
    options.keep = true;
    let report = compress_tree(&scratch.0, &options).expect("compresses");
    assert!(matches!(status_of(&report, &weight), FileStatus::Compressed));
    assert!(weight.exists());
    let container = scratch.path("m.ziptensors");
    assert_eq!(restored(&container).1, safetensors(11));

    let report = compress_tree(&scratch.0, &options).expect("reruns");
    assert!(matches!(status_of(&report, &weight), FileStatus::PairVerified), "{report:#?}");
    // Without keep, the verified pair loses its original.
    let report = compress_tree(&scratch.0, &compress_options()).expect("reruns");
    assert!(matches!(status_of(&report, &weight), FileStatus::PairResolved), "{report:#?}");
    assert!(!weight.exists());

    // Decompress with keep leaves the container; a rerun resolves the pair it left.
    let mut options = decompress_options();
    options.keep = true;
    let report = decompress_tree(&scratch.0, &options).expect("restores");
    assert!(matches!(status_of(&report, &container), FileStatus::Restored), "{report:#?}");
    assert_eq!(fs::read(&weight).expect("restored"), safetensors(11));
    let report = decompress_tree(&scratch.0, &options).expect("reruns");
    assert!(matches!(status_of(&report, &container), FileStatus::PairVerified), "{report:#?}");
    let report = decompress_tree(&scratch.0, &decompress_options()).expect("reruns");
    assert!(matches!(status_of(&report, &container), FileStatus::PairResolved), "{report:#?}");
    assert!(!container.exists());
    assert_eq!(fs::read(&weight).expect("kept"), safetensors(11));
}

#[test]
fn a_dry_run_writes_nothing() {
    let scratch = Scratch::new("dry");
    model_tree(&scratch);
    let before = snapshot(&scratch.0);
    let mut options = compress_options();
    options.dry_run = true;
    let report = compress_tree(&scratch.0, &options).expect("plans");
    assert!(report.dry_run);
    assert_eq!(report.totals().planned, 6, "{report:#?}");
    assert!(matches!(
        status_of(&report, &scratch.path("llm/model-q8.gguf")),
        FileStatus::Planned(Action::Compress)
    ));
    assert_eq!(snapshot(&scratch.0), before);

    compress_tree(&scratch.0, &compress_options()).expect("compresses");
    let compressed = snapshot(&scratch.0);
    let mut options = decompress_options();
    options.dry_run = true;
    let report = decompress_tree(&scratch.0, &options).expect("plans");
    assert_eq!(report.totals().planned, 6, "{report:#?}");
    assert_eq!(snapshot(&scratch.0), compressed);
}

#[test]
fn colliding_names_refuse_the_whole_tree_before_any_write() {
    let check = |scratch: &Scratch, expected: &[&str]| {
        let before = snapshot(&scratch.0);
        let error = compress_tree(&scratch.0, &compress_options()).expect_err("refused");
        let Error::TreeConflict { conflicts, .. } = &error else { panic!("{error:?}") };
        assert_eq!(conflicts.len(), expected.len(), "{conflicts:?}");
        for (conflict, text) in conflicts.iter().zip(expected) {
            assert!(conflict.contains(text), "{conflict} should name {text}");
        }
        assert_eq!(snapshot(&scratch.0), before, "nothing was written");
        let mut dry = compress_options();
        dry.dry_run = true;
        assert!(matches!(compress_tree(&scratch.0, &dry), Err(Error::TreeConflict { .. })));
    };

    // Two weight files of one stem; the other directory is fine but is not compressed either.
    let scratch = Scratch::new("stem");
    scratch.write("a/model.safetensors", &safetensors(21));
    scratch.write("a/model.gguf", &gguf(22));
    scratch.write("b/fine.safetensors", &safetensors(23));
    check(&scratch, &["both map to the container"]);

    // A file that is not a container at the target name.
    let scratch = Scratch::new("taken");
    scratch.write("model.safetensors", &safetensors(24));
    scratch.write("model.ziptensors", b"my notes");
    check(&scratch, &["is taken by a file that is not a container"]);

    // A container of another original at the target name.
    let scratch = Scratch::new("other");
    let other = scratch.write("other.bin", &safetensors(25));
    container_of(&other, &scratch.path("model.ziptensors"), "other.bin");
    fs::remove_file(&other).expect("remove");
    scratch.write("model.safetensors", &safetensors(26));
    check(&scratch, &["a container of \"other.bin\""]);

    // An original that another container already restores to.
    let scratch = Scratch::new("claimed");
    let weight = scratch.write("x.safetensors", &safetensors(27));
    container_of(&weight, &scratch.path("elsewhere.ziptensors"), "x.safetensors");
    check(&scratch, &["already what the container"]);

    // A weight file named like a container is its own target.
    let scratch = Scratch::new("self");
    scratch.write("weights.ziptensors", &safetensors(28));
    check(&scratch, &["is taken by a file that is not a container"]);
}

#[cfg(target_os = "linux")]
#[test]
fn targets_that_differ_only_in_case_collide() {
    let scratch = Scratch::new("case");
    scratch.write("Model.safetensors", &safetensors(31));
    scratch.write("model.gguf", &gguf(32));
    let error = compress_tree(&scratch.0, &compress_options()).expect_err("refused");
    assert!(matches!(error, Error::TreeConflict { .. }), "{error:?}");
}

#[test]
fn a_stale_container_is_replaced_and_a_damaged_one_keeps_the_original() {
    let scratch = Scratch::new("stale");
    let weight = scratch.write("m.safetensors", &safetensors(41));
    let mut keep = compress_options();
    keep.keep = true;
    compress_tree(&scratch.0, &keep).expect("compresses");
    let container = scratch.path("m.ziptensors");

    // The original changed after the container was made: the container is stale.
    fs::write(&weight, safetensors(42)).expect("rewrite");
    let report = compress_tree(&scratch.0, &compress_options()).expect("reruns");
    assert!(matches!(status_of(&report, &weight), FileStatus::Recompressed), "{report:#?}");
    assert!(!weight.exists());
    assert_eq!(restored(&container).1, safetensors(42));

    // A pair whose container is damaged: the original stays.
    fs::write(&weight, safetensors(42)).expect("original back");
    let mut bytes = fs::read(&container).expect("container");
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0x55;
    fs::write(&container, &bytes).expect("damage");
    let report = compress_tree(&scratch.0, &compress_options()).expect("reruns");
    assert!(matches!(status_of(&report, &weight), FileStatus::Failed(_)), "{report:#?}");
    assert!(report.has_failures());
    assert_eq!(fs::read(&weight).expect("kept"), safetensors(42));
    assert_eq!(fs::read(&container).expect("left"), bytes);
}

#[test]
fn reruns_converge_from_every_state_an_interruption_leaves() {
    // After the container commit and before the removal: both exist.
    let scratch = Scratch::new("crash");
    let weight = scratch.write("dir/w.gguf", &gguf(51));
    container_of(&weight, &scratch.path("dir/w.ziptensors"), "w.gguf");
    let temp = scratch.write("dir/w.ziptensors.123-0.ziptensors-tmp", b"partial");
    let report = compress_tree(&scratch.0, &compress_options()).expect("reruns");
    assert!(matches!(status_of(&report, &weight), FileStatus::PairResolved), "{report:#?}");
    assert!(matches!(status_of(&report, &temp), FileStatus::Skipped(SkipReason::StaleTemp)));
    assert!(temp.exists(), "a stale temporary file is never removed automatically");
    let settled = snapshot(&scratch.0);
    let report = compress_tree(&scratch.0, &compress_options()).expect("reruns");
    assert_eq!(report.totals().already_compressed, 1, "{report:#?}");
    assert_eq!(snapshot(&scratch.0), settled);

    // Decompress interrupted after the restore: both exist again; the rerun removes the container.
    fs::write(&weight, gguf(51)).expect("restored copy");
    let report = decompress_tree(&scratch.0, &decompress_options()).expect("reruns");
    let container = scratch.path("dir/w.ziptensors");
    assert!(matches!(status_of(&report, &container), FileStatus::PairResolved), "{report:#?}");
    assert!(!container.exists());
    assert_eq!(fs::read(&weight).expect("kept"), gguf(51));
}

#[test]
fn decompress_refuses_conflicts_and_keeps_a_different_file_unless_forced() {
    // Two containers restore to one name.
    let scratch = Scratch::new("dconf");
    let weight = scratch.write("w.safetensors", &safetensors(61));
    container_of(&weight, &scratch.path("one.ziptensors"), "w.safetensors");
    container_of(&weight, &scratch.path("two.ziptensors"), "w.safetensors");
    fs::remove_file(&weight).expect("remove");
    let before = snapshot(&scratch.0);
    let view = scan_view(&scratch.0, &Default::default()).expect("scans");
    assert_eq!(view.conflicts.len(), 1, "{view:#?}");
    assert_eq!(view.conflicts[0].reason, ConflictReason::SameName);
    let error = decompress_tree(&scratch.0, &decompress_options()).expect_err("refused");
    assert!(matches!(error, Error::TreeConflict { .. }), "{error:?}");
    assert_eq!(snapshot(&scratch.0), before);

    // A container restoring to a directory.
    let scratch = Scratch::new("ddir");
    let weight = scratch.write("w.safetensors", &safetensors(62));
    container_of(&weight, &scratch.path("w.ziptensors"), "w.safetensors");
    fs::remove_file(&weight).expect("remove");
    fs::create_dir(&weight).expect("a directory of that name");
    let error = decompress_tree(&scratch.0, &decompress_options()).expect_err("refused");
    assert!(matches!(error, Error::TreeConflict { .. }), "{error:?}");

    // A different file at the restore path: kept, unless forced.
    let scratch = Scratch::new("dforce");
    let weight = scratch.write("w.safetensors", &safetensors(63));
    let container = scratch.path("w.ziptensors");
    container_of(&weight, &container, "w.safetensors");
    fs::write(&weight, b"something else").expect("overwrite");
    let report = decompress_tree(&scratch.0, &decompress_options()).expect("runs");
    let FileStatus::Failed(error) = status_of(&report, &container) else { panic!("{report:#?}") };
    assert!(error.to_string().contains("force"), "{error}");
    assert_eq!(fs::read(&weight).expect("kept"), b"something else");
    assert!(container.exists());
    let mut force = decompress_options();
    force.force = true;
    let report = decompress_tree(&scratch.0, &force).expect("runs");
    assert!(matches!(status_of(&report, &container), FileStatus::Restored), "{report:#?}");
    assert_eq!(fs::read(&weight).expect("restored"), safetensors(63));
    assert!(!container.exists());
}

#[test]
fn foreign_containers_are_reported_and_never_written_from() {
    let scratch = Scratch::new("foreign");
    let a = scratch.write("src/a.safetensors", &safetensors(71));
    let b = scratch.write("src/b.safetensors", &safetensors(72));
    let pack = scratch.path("models/pack.ziptensors");
    fs::create_dir_all(pack.parent().expect("parent")).expect("dir");
    let mut builder = PackBuilder::new(&pack, &EncodeOptions { mode: Mode::Fast, threads: 1 }).expect("builder");
    builder.add_file(&a, "a.safetensors").expect("add");
    builder.add_file(&b, "b.safetensors").expect("add");
    builder.finish().expect("finish");
    let nested = scratch.path("models/nested.ziptensors");
    container_of(&a, &nested, "sub/a.safetensors");
    let dot = scratch.path("models/dot.ziptensors");
    container_of(&a, &dot, ".");
    fs::remove_dir_all(scratch.path("src")).expect("remove sources");
    let before = snapshot(&scratch.0);

    for report in [
        decompress_tree(&scratch.0, &decompress_options()).expect("runs"),
        compress_tree(&scratch.0, &compress_options()).expect("runs"),
    ] {
        assert!(matches!(status_of(&report, &pack), FileStatus::Foreign(ForeignReason::MemberCount(2))), "{report:#?}");
        let FileStatus::Foreign(ForeignReason::NotOneSafeName(name)) = status_of(&report, &nested) else {
            panic!("{report:#?}")
        };
        assert_eq!(name, "sub/a.safetensors");
        assert!(matches!(status_of(&report, &dot), FileStatus::Foreign(ForeignReason::NotOneSafeName(_))));
        assert_eq!(snapshot(&scratch.0), before);
    }
}

#[test]
fn a_single_file_root_is_compressed_and_restored_beside_itself() {
    let scratch = Scratch::new("single");
    let weight = scratch.write("w.safetensors", &safetensors(81));
    let neighbour = scratch.write("other.safetensors", &safetensors(82));
    let report = compress_tree(&weight, &compress_options()).expect("compresses");
    assert!(matches!(status_of(&report, &weight), FileStatus::Compressed), "{report:#?}");
    assert_eq!(report.files.len(), 1);
    assert!(neighbour.exists(), "only the root is compressed");
    let container = scratch.path("w.ziptensors");
    let report = decompress_tree(&container, &decompress_options()).expect("restores");
    assert!(matches!(status_of(&report, &container), FileStatus::Restored), "{report:#?}");
    assert_eq!(fs::read(&weight).expect("restored"), safetensors(81));

    // A single file whose target is taken is refused like a tree.
    fs::write(scratch.path("w.ziptensors"), b"taken").expect("write");
    let error = compress_tree(&weight, &compress_options()).expect_err("refused");
    assert!(matches!(error, Error::TreeConflict { .. }), "{error:?}");
}

#[test]
fn partial_downloads_are_skipped() {
    let scratch = Scratch::new("partial");
    let partial = scratch.write("m.safetensors.incomplete", &safetensors(91));
    let report = compress_tree(&scratch.0, &compress_options()).expect("runs");
    assert!(matches!(status_of(&report, &partial), FileStatus::Skipped(SkipReason::Incomplete)));
    assert!(partial.exists());
}

#[test]
fn the_view_shows_names_sizes_and_pairs() {
    let scratch = Scratch::new("view");
    let weight = scratch.write("d/w.safetensors", &safetensors(95));
    container_of(&weight, &scratch.path("d/w.ziptensors"), "w.safetensors");
    let lone = scratch.write("d/lone.gguf", &gguf(96));
    container_of(&lone, &scratch.path("d/renamed.ziptensors"), "lone.gguf");
    fs::remove_file(&lone).expect("remove");
    let view = scan_view(&scratch.0, &Default::default()).expect("scans");
    assert!(view.conflicts.is_empty(), "{view:#?}");
    let mut seen = Vec::new();
    for entry in &view.entries {
        match entry {
            ViewEntry::Directory { rel_path, .. } => seen.push(format!("dir {rel_path}")),
            ViewEntry::Plain { rel_path, stamp, .. } => {
                assert_eq!(stamp.len, u64::try_from(safetensors(95).len()).expect("fits"));
                seen.push(format!("plain {rel_path}"));
            }
            ViewEntry::PerFile(per_file) => {
                assert_eq!(per_file.member_len, u64::try_from(gguf(96).len()).expect("fits"));
                assert_eq!(per_file.blake3, *blake3::hash(&gguf(96)).as_bytes());
                seen.push(format!("per-file {} as {}", per_file.container_rel, per_file.rel_path));
            }
            ViewEntry::Shadowed(per_file) => {
                seen.push(format!("shadowed {} as {}", per_file.container_rel, per_file.rel_path));
            }
            other => panic!("{other:?}"),
        }
    }
    assert_eq!(
        seen,
        [
            "dir d",
            "per-file d/renamed.ziptensors as d/lone.gguf",
            "plain d/w.safetensors",
            "shadowed d/w.ziptensors as d/w.safetensors",
        ]
    );
}

#[cfg(unix)]
mod unix {
    use super::*;
    use std::os::unix::fs::symlink;
    use ziptensors_core::tree::TreeWarning;

    #[test]
    fn links_are_skipped_with_a_warning_by_default() {
        let scratch = Scratch::new("links");
        let blob = scratch.write("blobs/abc123", &safetensors(101));
        let link = scratch.path("snapshots/main/model.safetensors");
        fs::create_dir_all(link.parent().expect("parent")).expect("dir");
        symlink("../../blobs/abc123", &link).expect("link");
        let mut options = compress_options();
        options.keep = true;
        let report = compress_tree(&scratch.path("snapshots"), &options).expect("runs");
        assert!(matches!(status_of(&report, &link), FileStatus::Skipped(SkipReason::Symlink)));
        assert_eq!(report.warnings, [TreeWarning::SymlinksSkipped { count: 1 }]);
        let text = report.warnings[0].to_string();
        assert!(text.contains("follow_symlinks") && text.contains("Hugging Face"), "{text}");
        assert!(fs::symlink_metadata(&link).expect("link").is_symlink());
        assert_eq!(fs::read(&blob).expect("blob"), safetensors(101));
    }

    #[test]
    fn followed_links_become_containers_in_place_of_the_link() {
        let scratch = Scratch::new("follow");
        let outside = Scratch::new("follow-outside");
        let blob = scratch.write("blobs/abc123", &safetensors(111));
        let far = outside.write("far.gguf", &gguf(112));
        scratch.write("snap/real.json", b"{}");
        fs::create_dir_all(scratch.path("snap")).expect("dir");
        let first = scratch.path("snap/model.safetensors");
        let second = scratch.path("snap/copy.safetensors");
        let remote = scratch.path("snap/far.gguf");
        let to_dir = scratch.path("snap/dir-link");
        let dangling = scratch.path("snap/gone.safetensors");
        let to_json = scratch.path("snap/config.json");
        symlink("../blobs/abc123", &first).expect("link");
        symlink("../blobs/abc123", &second).expect("link");
        symlink(&far, &remote).expect("link");
        symlink("../blobs", &to_dir).expect("link");
        symlink("../blobs/missing", &dangling).expect("link");
        symlink("real.json", &to_json).expect("link");
        let mut options = compress_options();
        options.follow_symlinks = true;
        let report = compress_tree(&scratch.path("snap"), &options).expect("runs");
        assert!(!report.has_failures(), "{report:#?}");
        assert!(report.warnings.is_empty());
        for (link, container, member, bytes) in [
            (&first, "snap/model.ziptensors", "model.safetensors", safetensors(111)),
            (&second, "snap/copy.ziptensors", "copy.safetensors", safetensors(111)),
            (&remote, "snap/far.ziptensors", "far.gguf", gguf(112)),
        ] {
            assert!(matches!(status_of(&report, link), FileStatus::Compressed), "{report:#?}");
            assert!(fs::symlink_metadata(link).is_err(), "the link is removed");
            assert_eq!(restored(&scratch.path(container)), (member.to_owned(), bytes));
        }
        // Targets are never removed, inside the tree or outside it.
        assert_eq!(fs::read(&blob).expect("blob"), safetensors(111));
        assert_eq!(fs::read(&far).expect("outside"), gguf(112));
        assert!(matches!(status_of(&report, &to_dir), FileStatus::Skipped(SkipReason::SymlinkToDirectory)));
        assert!(matches!(status_of(&report, &dangling), FileStatus::Skipped(SkipReason::DanglingSymlink)));
        assert!(matches!(status_of(&report, &to_json), FileStatus::NotWeight));
        assert!(fs::symlink_metadata(&to_json).expect("kept").is_symlink());

        // The containers decompress to regular files under the link names.
        decompress_tree(&scratch.path("snap"), &decompress_options()).expect("restores");
        assert_eq!(fs::read(&first).expect("restored"), safetensors(111));
        assert!(!fs::symlink_metadata(&first).expect("file").is_symlink());
    }

    #[test]
    fn a_followed_link_to_a_weight_of_the_same_tree_runs_before_its_target() {
        let scratch = Scratch::new("inner-link");
        let target = scratch.write("a/w.safetensors", &safetensors(121));
        let link = scratch.path("b/w.safetensors");
        fs::create_dir_all(link.parent().expect("parent")).expect("dir");
        symlink("../a/w.safetensors", &link).expect("link");
        let mut options = compress_options();
        options.follow_symlinks = true;
        let report = compress_tree(&scratch.0, &options).expect("runs");
        assert!(!report.has_failures(), "{report:#?}");
        assert!(!target.exists() && fs::symlink_metadata(&link).is_err());
        assert_eq!(restored(&scratch.path("a/w.ziptensors")).1, safetensors(121));
        assert_eq!(restored(&scratch.path("b/w.ziptensors")).1, safetensors(121));
    }

    #[test]
    fn special_files_and_names_that_are_not_utf8_are_skipped() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        use std::os::unix::net::UnixListener;

        let scratch = Scratch::new("special");
        let socket = scratch.path("model.sock");
        let _listener = UnixListener::bind(&socket).expect("a socket file");
        let odd = scratch.0.join(OsStr::from_bytes(b"w\xff.safetensors"));
        fs::write(&odd, safetensors(131)).expect("odd name");
        let odd_dir = scratch.0.join(OsStr::from_bytes(b"d\xfe"));
        fs::create_dir(&odd_dir).expect("odd directory");
        fs::write(odd_dir.join("inner.safetensors"), safetensors(132)).expect("inside");
        let before = snapshot(&scratch.0);
        let report = compress_tree(&scratch.0, &compress_options()).expect("runs");
        assert!(matches!(status_of(&report, &socket), FileStatus::Skipped(SkipReason::Special)));
        assert!(matches!(status_of(&report, &odd), FileStatus::Skipped(SkipReason::NonUtf8)));
        assert!(matches!(status_of(&report, &odd_dir), FileStatus::Skipped(SkipReason::NonUtf8)));
        assert_eq!(snapshot(&scratch.0), before);
    }
}
