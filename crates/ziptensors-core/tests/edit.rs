//! File: tests/edit.rs
//! Purpose: `edit` through the public API: operations under each strategy, the append protocol as a
//! reader sees it, the in-place rewrite, repack, compaction, refusals, and an edit of a container
//! that is still open.
//!
//! Main responsibilities:
//! - add, remove and replace under `Rewrite`, `Append` and `InPlace`: the result opens, every member
//!   verifies, kept members restore byte for byte, the member order does not depend on the
//!   strategy, the generation is the source's plus one, and no file appears beside the container;
//! - `Append`: the source is a byte-identical prefix of the result, its header stays, delta blobs only
//!   while required bit 0 is set;
//! - `repack`: a `fast` source shrinks under `max`, and a kept delta member pairs with its base again;
//! - compaction drops the dead bytes an append leaves, and `layer_report` counts them before, under
//!   `Rewrite` and `InPlace`;
//! - `InPlace`: the base blob of a kept delta member stays when its member goes, bit 0 follows the
//!   blobs left, and a delta against a file of the same edit is stored;
//! - every refusal happens before any write.
//!
//! Implementation notes: crash injection, verification-before-visibility and the in-place planner
//! need seams, so they are in `src/edit/tests.rs`. Inputs are generated from seeds and live in a unique temporary directory,
//! which is removed.

#[path = "common/fixtures.rs"]
mod fixtures;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use fixtures::{Elem, TensorSpec};
use ziptensors_core::chain::{BLOB_KIND_DELTA, FEATURE_DELTA};
use ziptensors_core::format::{Footer, Index, Limits, parse_footer, parse_header};
use ziptensors_core::io::TEMP_FILE_SUFFIX;
use ziptensors_core::{
    Container, EditOp, EditOptions, EditReport, EncodeOptions, Error, Mode, OpenOptions,
    PackBuilder, Strategy, edit, layer_report,
};

/// Bytes of a tensor that spans several chunks.
const TENSOR_BYTES: u64 = (2 << 20) + (1 << 19);

/// A unique scratch directory, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let name = format!("ziptensors-edit-{}-{id}", std::process::id());
        let path = std::env::temp_dir().join(name);
        fs::create_dir_all(&path).expect("scratch directory");
        Self(path)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    fn file(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.path(name);
        fs::write(&path, bytes).expect("scratch file");
        path
    }

    fn entries(&self) -> Vec<String> {
        let entries = fs::read_dir(&self.0).expect("readable");
        let mut names: Vec<String> = entries
            .map(|entry| entry.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("cannot remove {}: {error}", self.0.display());
        }
    }
}

/// A safetensors file of two `BF16` tensors of [`TENSOR_BYTES`] each.
fn base_file() -> Vec<u8> {
    let (elements, block) = Elem::Bf16.block();
    let shape = [TENSOR_BYTES / block * elements];
    let specs = [
        TensorSpec::new("blocks.0.attn.weight", &shape, Elem::Bf16),
        TensorSpec::new("blocks.0.mlp.weight", &shape, Elem::Bf16),
    ];
    fixtures::safetensors(31, &specs).bytes
}

/// `base` with the lowest bit of every `step`th byte past its header flipped: a finetune whose
/// tensors pair with the base's and store as delta blobs in `max`.
fn finetune(base: &[u8], step: usize) -> Vec<u8> {
    let header = usize::try_from(u64::from_le_bytes(base[..8].try_into().expect("8 bytes")))
        .expect("fits");
    let mut tuned = base.to_vec();
    tuned[8 + header..].iter_mut().step_by(step).for_each(|byte| *byte ^= 1);
    tuned
}

/// `len` bytes of an opaque file of seed `seed`, compressible.
fn opaque(seed: u64, len: usize) -> Vec<u8> {
    let mut rng = fixtures::Rng::new(seed);
    (0..len).map(|at| if at % 3 == 0 { 0 } else { rng.next_u64().to_le_bytes()[0] & 0x3F }).collect()
}

/// The members of a source pack and their bytes, in member order.
struct Source {
    path: PathBuf,
    members: Vec<(String, Vec<u8>)>,
}

impl Source {
    fn bytes_of(&self, name: &str) -> &[u8] {
        let found = self.members.iter().find(|(member, _)| member == name);
        &found.expect("a member of the source").1
    }
}

/// Packs `base`, `tuned` (against `base` when `delta`), `x` and `y` into `name` under `mode`.
fn pack(scratch: &Scratch, name: &str, mode: Mode, delta: bool) -> Source {
    let base = base_file();
    let tuned = finetune(&base, 14);
    let members = vec![
        ("base".to_owned(), base),
        ("tuned".to_owned(), tuned),
        ("x".to_owned(), opaque(1, 300_000)),
        ("y".to_owned(), opaque(2, 200_000)),
    ];
    let path = scratch.path(name);
    let mut builder = PackBuilder::new(&path, &EncodeOptions { mode, threads: 3 }).expect("starts");
    for (member, bytes) in &members {
        let input = scratch.file(&format!("{name}.{member}.in"), bytes);
        if delta && member == "tuned" {
            builder.add_file_delta(&input, member, "base").expect("adds against the base");
        } else {
            builder.add_file(&input, member).expect("adds");
        }
    }
    builder.finish().expect("commits");
    Source { path, members }
}

fn options(strategy: Strategy) -> EditOptions {
    EditOptions { strategy, threads: 3, ..EditOptions::default() }
}

/// The footer and index of the container at `path`, read as a third party would.
fn index_of(path: &Path) -> (Footer, Index) {
    let bytes = fs::read(path).expect("container");
    let at = bytes.len() - 64;
    let footer = parse_footer(&bytes[at..].try_into().expect("64 bytes")).expect("valid footer");
    let start = usize::try_from(footer.index_offset).expect("fits");
    let raw_len = usize::try_from(footer.index_raw_len).expect("fits");
    let raw = zstd::bulk::decompress(&bytes[start..at], raw_len).expect("valid index frame");
    (footer, Index::parse(&raw, &Limits::default()).expect("valid index"))
}

/// Required feature bits of the container at `path`.
fn required_features(path: &Path) -> u64 {
    let bytes = fs::read(path).expect("container");
    parse_header(&bytes[..64].try_into().expect("64 bytes")).expect("valid header").required_features
}

/// Delta blobs that some member of `index` reaches.
fn reached_delta_blobs(index: &Index) -> usize {
    let mut reached = std::collections::HashSet::new();
    for member in index.members() {
        for extent in index.member_extents(member).expect("valid") {
            if index.blobs()[extent.blob_id as usize].kind == BLOB_KIND_DELTA {
                reached.insert(extent.blob_id);
            }
        }
    }
    reached.len()
}

/// Opens `path`, verifies every member, and checks that the members are `expected`, in order, with
/// those bytes.
fn assert_members(path: &Path, expected: &[(&str, &[u8])]) {
    let container = Container::open(path, &OpenOptions::default()).expect("the result opens");
    let names: Vec<&str> = container.members().iter().map(|info| info.name.as_str()).collect();
    let wanted: Vec<&str> = expected.iter().map(|(name, _)| *name).collect();
    assert_eq!(names, wanted);
    for (name, bytes) in expected {
        let member = container.member(name).expect("present");
        member.verify().expect("verifies");
        let mut read = vec![0; bytes.len()];
        member.read_at(0, &mut read).expect("reads");
        assert!(read == *bytes, "member {name} differs");
    }
}

#[test]
fn each_strategy_applies_the_same_operations_in_the_same_order() {
    let scratch = Scratch::new();
    let replacement = opaque(7, 250_000);
    let added = opaque(8, 150_000);
    let ops = vec![
        EditOp::Remove { name: "x".to_owned() },
        EditOp::Add { path: scratch.file("z.in", &added), name: "z".to_owned(), delta_base: None },
        EditOp::Replace {
            name: "y".to_owned(),
            path: scratch.file("y2.in", &replacement),
            delta_base: None,
        },
    ];
    for strategy in [Strategy::Rewrite, Strategy::Append, Strategy::InPlace] {
        let source = pack(&scratch, &format!("{strategy:?}.zt"), Mode::Max, true);
        let before = fs::read(&source.path).expect("reads");
        let entries = scratch.entries();
        let report = edit(&source.path, &ops, &options(strategy)).expect("edits");
        assert_eq!(scratch.entries(), entries, "{strategy:?}: no file appeared or vanished");
        assert_eq!((report.kept, report.added, report.removed, report.replaced), (2, 1, 1, 1));
        assert_eq!(report.generation, 2);
        let (footer, _) = index_of(&source.path);
        assert_eq!(footer.generation, 2, "{strategy:?}");
        assert_eq!(report.encode.output_len, fs::metadata(&source.path).expect("exists").len());
        assert_members(
            &source.path,
            &[
                ("base", source.bytes_of("base")),
                ("tuned", source.bytes_of("tuned")),
                ("y", &replacement),
                ("z", &added),
            ],
        );
        let after = fs::read(&source.path).expect("reads");
        if strategy == Strategy::Append {
            assert!(after[..before.len()] == before[..], "the source must be a prefix of the result");
        }
        if strategy == Strategy::InPlace {
            assert!(after.len() < before.len(), "300 000 bytes out, 250 000 and 150 000 in, deduplicated");
        }
    }
}

#[test]
fn an_append_adds_a_delta_member_against_a_kept_base_and_keeps_the_header() {
    let scratch = Scratch::new();
    let source = pack(&scratch, "pack.zt", Mode::Max, true);
    assert_ne!(required_features(&source.path) & FEATURE_DELTA, 0);
    let (_, index) = index_of(&source.path);
    let deltas_before = reached_delta_blobs(&index);
    let before = fs::read(&source.path).expect("reads");
    let tuned2 = finetune(source.bytes_of("base"), 22);
    let ops = [EditOp::Add {
        path: scratch.file("tuned2.in", &tuned2),
        name: "tuned2".to_owned(),
        delta_base: Some("base".to_owned()),
    }];
    let report = edit(&source.path, &ops, &options(Strategy::Append)).expect("appends");
    assert!(report.encode.members[4].delta_blobs > 0, "{:?}", report.encode.members[4]);
    let after = fs::read(&source.path).expect("reads");
    assert!(after[..before.len()] == before[..], "the source must be a prefix of the result");
    let (_, index) = index_of(&source.path);
    assert!(reached_delta_blobs(&index) > deltas_before);
    assert_members(
        &source.path,
        &[
            ("base", source.bytes_of("base")),
            ("tuned", source.bytes_of("tuned")),
            ("x", source.bytes_of("x")),
            ("y", source.bytes_of("y")),
            ("tuned2", &tuned2),
        ],
    );
}

#[test]
fn removing_the_only_delta_member_under_append_keeps_bit_0_and_a_valid_file() {
    let scratch = Scratch::new();
    let source = pack(&scratch, "pack.zt", Mode::Max, true);
    let ops = [EditOp::Remove { name: "tuned".to_owned() }];
    edit(&source.path, &ops, &options(Strategy::Append)).expect("appends");
    assert_ne!(required_features(&source.path) & FEATURE_DELTA, 0);
    let (_, index) = index_of(&source.path);
    assert!(index.blobs().iter().any(|blob| blob.kind == BLOB_KIND_DELTA), "records carried");
    assert_eq!(reached_delta_blobs(&index), 0);
    assert_members(
        &source.path,
        &[("base", source.bytes_of("base")), ("x", source.bytes_of("x")), ("y", source.bytes_of("y"))],
    );
}

#[test]
fn a_delta_needs_bit_0_under_append_and_a_rewrite_sets_it() {
    let scratch = Scratch::new();
    let source = pack(&scratch, "pack.zt", Mode::Max, false);
    assert_eq!(required_features(&source.path) & FEATURE_DELTA, 0);
    let before = fs::read(&source.path).expect("reads");
    let tuned2 = finetune(source.bytes_of("base"), 22);
    let ops = [EditOp::Add {
        path: scratch.file("tuned2.in", &tuned2),
        name: "tuned2".to_owned(),
        delta_base: Some("base".to_owned()),
    }];
    let error = edit(&source.path, &ops, &options(Strategy::Append)).expect_err("bit 0 clear");
    assert!(matches!(error, Error::EditRefused { .. }), "{error:?}");
    assert!(fs::read(&source.path).expect("reads") == before, "the file changed");

    // A rewrite may set the bit; the kept base is copied and serves as the base.
    let report = edit(&source.path, &ops, &options(Strategy::Rewrite)).expect("rewrites");
    assert!(report.encode.members[4].delta_blobs > 0, "{:?}", report.encode.members[4]);
    assert_ne!(required_features(&source.path) & FEATURE_DELTA, 0);
    let (_, index) = index_of(&source.path);
    assert!(reached_delta_blobs(&index) > 0);
    assert_members(
        &source.path,
        &[
            ("base", source.bytes_of("base")),
            ("tuned", source.bytes_of("tuned")),
            ("x", source.bytes_of("x")),
            ("y", source.bytes_of("y")),
            ("tuned2", &tuned2),
        ],
    );
}

/// A container at `name` holding `../x`, `y` and `z`: valid, but no pack writes it, since a restore
/// refuses the name `../x` (spec invariant 11 binds a restore, not a container). Packed with the
/// placeholder `QQ/x`, as long as `../x`, which is then patched in the index; the index is stored
/// again in place, under a footer for its new bytes.
fn pack_with_unsafe_name(scratch: &Scratch, name: &str) -> Source {
    let (placeholder, unsafe_name) = ("QQ/x", "../x");
    let members = vec![
        (placeholder.to_owned(), opaque(11, 300_000)),
        ("y".to_owned(), opaque(12, 200_000)),
        ("z".to_owned(), opaque(13, 100_000)),
    ];
    let path = scratch.path(name);
    let options = EncodeOptions { mode: Mode::Fast, threads: 3 };
    let mut builder = PackBuilder::new(&path, &options).expect("starts");
    for (at, (member, bytes)) in members.iter().enumerate() {
        let input = scratch.file(&format!("{name}.{at}.in"), bytes);
        builder.add_file(&input, member).expect("adds");
    }
    builder.finish().expect("commits");

    let (footer, _) = index_of(&path);
    let bytes = fs::read(&path).expect("container");
    let start = usize::try_from(footer.index_offset).expect("fits");
    let raw_len = usize::try_from(footer.index_raw_len).expect("fits");
    let mut raw = zstd::bulk::decompress(&bytes[start..bytes.len() - 64], raw_len).expect("index");
    let at: Vec<usize> = raw
        .windows(placeholder.len())
        .enumerate()
        .filter(|(_, window)| *window == placeholder.as_bytes())
        .map(|(at, _)| at)
        .collect();
    assert_eq!(at.len(), 1, "the placeholder is in the string table once");
    raw[at[0]..at[0] + unsafe_name.len()].copy_from_slice(unsafe_name.as_bytes());
    let stored = zstd::bulk::compress(&raw, 3).expect("compresses");
    let footer =
        Footer::for_index(footer.index_offset, stored.len() as u64, &raw, footer.generation);
    let patched = [&bytes[..start], stored.as_slice(), &footer.to_bytes()].concat();
    fs::write(&path, patched).expect("writes the patched container");

    let members = members
        .into_iter()
        .map(|(member, bytes)| (if member == placeholder { unsafe_name.to_owned() } else { member }, bytes))
        .collect();
    let source = Source { path, members };
    assert_members(&source.path, &[
        ("../x", source.bytes_of("../x")),
        ("y", source.bytes_of("y")),
        ("z", source.bytes_of("z")),
    ]);
    source
}

#[test]
fn an_edit_keeps_a_member_whose_name_a_restore_refuses() {
    let scratch = Scratch::new();
    let replacement = opaque(14, 120_000);
    let replacement_in = scratch.file("replacement.in", &replacement);
    let rewrite = options(Strategy::Rewrite);
    let cases = [
        ("rewrite", rewrite),
        ("append", options(Strategy::Append)),
        ("in place", options(Strategy::InPlace)),
        ("repack", EditOptions { repack: true, ..rewrite }),
    ];
    for (case, settings) in cases {
        let source = pack_with_unsafe_name(&scratch, &format!("{case}.zt"));
        let ops = [EditOp::Remove { name: "y".to_owned() }];
        let report = edit(&source.path, &ops, &settings).expect(case);
        assert_eq!((report.kept, report.removed), (2, 1), "{case}");
        assert_members(&source.path, &[
            ("../x", source.bytes_of("../x")),
            ("z", source.bytes_of("z")),
        ]);
    }
    let strategies = [
        ("rewrite", rewrite),
        ("append", options(Strategy::Append)),
        ("in place", options(Strategy::InPlace)),
    ];
    for (case, settings) in strategies {
        let source = pack_with_unsafe_name(&scratch, &format!("replace-{case}.zt"));
        let ops = [EditOp::Replace {
            name: "../x".to_owned(),
            path: replacement_in.clone(),
            delta_base: None,
        }];
        let report = edit(&source.path, &ops, &settings).expect(case);
        assert_eq!(report.replaced, 1, "{case}");
        assert_members(&source.path, &[
            ("../x", &replacement),
            ("y", source.bytes_of("y")),
            ("z", source.bytes_of("z")),
        ]);
    }
}

#[test]
fn an_added_name_a_restore_refuses_is_refused_before_any_write() {
    let scratch = Scratch::new();
    let source = pack_with_unsafe_name(&scratch, "pack.zt");
    let before = fs::read(&source.path).expect("reads");
    let input = scratch.file("new.in", b"new bytes");
    let entries = scratch.entries();
    for strategy in [Strategy::Rewrite, Strategy::Append] {
        let ops = [EditOp::Add { path: input.clone(), name: "../y".to_owned(), delta_base: None }];
        let error = edit(&source.path, &ops, &options(strategy)).expect_err("an unsafe name");
        assert!(matches!(error, Error::UnsafeMemberName { .. }), "{strategy:?}: {error:?}");
        assert!(fs::read(&source.path).expect("reads") == before, "{strategy:?}: the file changed");
        assert_eq!(scratch.entries(), entries, "{strategy:?}: a file appeared");
    }
}

#[test]
fn every_refusal_happens_before_any_write() {
    let scratch = Scratch::new();
    let source = pack(&scratch, "pack.zt", Mode::Fast, false);
    let before = fs::read(&source.path).expect("reads");
    let entries = scratch.entries();
    let input = scratch.file("new.in", b"new bytes");
    let entries_with_input = scratch.entries();
    assert_ne!(entries, entries_with_input);
    let add = |path: &Path, name: &str| EditOp::Add {
        path: path.to_path_buf(),
        name: name.to_owned(),
        delta_base: None,
    };
    let remove = |name: &str| EditOp::Remove { name: name.to_owned() };
    let append = options(Strategy::Append);
    let repack_append = EditOptions { repack: true, ..append };
    let rewrite = options(Strategy::Rewrite);
    let in_place = options(Strategy::InPlace);
    let repack_in_place = EditOptions { repack: true, ..in_place };
    let against_kept = EditOp::Add {
        path: input.clone(),
        name: "new".to_owned(),
        delta_base: Some("x".to_owned()),
    };
    let replace_against_kept = EditOp::Replace {
        name: "y".to_owned(),
        path: input.clone(),
        delta_base: Some("x".to_owned()),
    };
    let cases: Vec<(&str, Vec<EditOp>, EditOptions)> = vec![
        ("append with repack", vec![remove("x")], repack_append),
        ("append with no operation", vec![], append),
        ("a name twice", vec![remove("x"), add(&input, "x")], rewrite),
        ("an existing name added", vec![add(&input, "y")], append),
        ("a missing member", vec![remove("nope")], rewrite),
        ("the container as input", vec![add(&source.path, "self")], append),
        ("a directory as input", vec![add(&scratch.0, "dir")], rewrite),
        ("an unsafe name", vec![add(&input, "../up")], append),
        ("in place with repack", vec![remove("x")], repack_in_place),
        ("in place with no operation and repack", vec![], repack_in_place),
        ("in place against a kept base", vec![against_kept], in_place),
        ("in place replace against a kept base", vec![replace_against_kept], in_place),
        ("in place with an unsafe name", vec![add(&input, "../up")], in_place),
        ("in place with a missing member", vec![remove("nope")], in_place),
    ];
    for (case, ops, settings) in cases {
        let error = edit(&source.path, &ops, &settings).expect_err(case);
        let expected = match case {
            "a missing member" | "in place with a missing member" => {
                matches!(error, Error::MemberNotFound { .. })
            }
            "the container as input" => matches!(error, Error::OutputIsInput { .. }),
            "a directory as input" => matches!(error, Error::Io { .. }),
            "an unsafe name" | "in place with an unsafe name" => {
                matches!(error, Error::UnsafeMemberName { .. })
            }
            _ => matches!(error, Error::EditRefused { .. }),
        };
        assert!(expected, "{case}: {error:?}");
        assert!(fs::read(&source.path).expect("reads") == before, "{case}: the file changed");
        assert_eq!(scratch.entries(), entries_with_input, "{case}: a file appeared");
    }
    #[cfg(unix)]
    {
        let link = scratch.path("link.in");
        std::os::unix::fs::symlink(&input, &link).expect("links");
        let error = edit(&source.path, &[add(&link, "link")], &rewrite).expect_err("a link");
        assert!(matches!(error, Error::SymlinkInput { .. }), "{error:?}");
        assert!(fs::read(&source.path).expect("reads") == before, "the file changed");
    }
}

#[test]
fn a_repack_of_a_fast_source_under_max_shrinks_and_verifies() {
    let scratch = Scratch::new();
    // `fast` computes no blob identity, so a second copy of `x` costs its stored bytes again; `max`
    // stores it once.
    let path = scratch.path("pack.zt");
    let x = opaque(1, 300_000);
    let members =
        vec![("base".to_owned(), base_file()), ("x".to_owned(), x.clone()), ("x2".to_owned(), x)];
    let settings = EncodeOptions { mode: Mode::Fast, threads: 3 };
    let mut builder = PackBuilder::new(&path, &settings).expect("starts");
    for (member, bytes) in &members {
        builder.add_file(&scratch.file(&format!("{member}.in"), bytes), member).expect("adds");
    }
    builder.finish().expect("commits");
    let source = Source { path, members };
    let before = fs::metadata(&source.path).expect("exists").len();
    let settings = EditOptions { repack: true, ..options(Strategy::Rewrite) };
    let report = edit(&source.path, &[], &settings).expect("repacks");
    let after = fs::metadata(&source.path).expect("exists").len();
    assert!(after < before, "{after} not below {before}");
    assert_eq!(report.encode.output_len, after);
    assert_eq!(report.kept, 3);
    assert!(report.encode.members[2].blobs_reused > 0, "{:?}", report.encode.members[2]);
    let expected: Vec<(&str, &[u8])> =
        source.members.iter().map(|(name, bytes)| (name.as_str(), bytes.as_slice())).collect();
    assert_members(&source.path, &expected);
}

#[test]
fn a_repack_pairs_a_kept_delta_member_with_its_base_again() {
    let scratch = Scratch::new();
    let source = pack(&scratch, "pack.zt", Mode::Max, true);
    let settings = EditOptions { repack: true, ..options(Strategy::Rewrite) };
    let report = edit(&source.path, &[], &settings).expect("repacks");
    assert!(report.encode.members[1].delta_blobs > 0, "{:?}", report.encode.members[1]);
    assert_eq!(report.encode.members[1].copied_bytes, 0, "re-encoded, not copied");
    let (_, index) = index_of(&source.path);
    assert!(reached_delta_blobs(&index) > 0);
    assert_ne!(required_features(&source.path) & FEATURE_DELTA, 0);
    let expected: Vec<(&str, &[u8])> =
        source.members.iter().map(|(name, bytes)| (name.as_str(), bytes.as_slice())).collect();
    assert_members(&source.path, &expected);
}

/// What `layer_report` says of the container at `path`.
fn layers(path: &Path) -> ziptensors_core::LayerBytes {
    let container = Container::open(path, &OpenOptions::default()).expect("opens");
    layer_report(&container).expect("reports").0
}

#[test]
fn a_compaction_drops_what_an_append_left_behind() {
    let scratch = Scratch::new();
    let source = pack(&scratch, "pack.zt", Mode::Max, true);
    let first = edit(&source.path, &[], &options(Strategy::Rewrite)).expect("compacts");
    assert_eq!(first.dead_bytes_before, 0, "a fresh pack has no dead bytes");
    let overhead_before = layers(&source.path).overhead;
    let ops = [EditOp::Remove { name: "x".to_owned() }];
    let appended: EditReport = edit(&source.path, &ops, &options(Strategy::Append)).expect("appends");
    // The blobs of `x` stay in the file, now stored bytes no member reaches.
    let layers_after = layers(&source.path);
    assert!(layers_after.overhead > overhead_before);
    assert_eq!(appended.encode.total, layers_after, "the report accounts as layer_report does");

    let len_before = fs::metadata(&source.path).expect("exists").len();
    let (footer_before, _) = index_of(&source.path);
    let compacted = edit(&source.path, &[], &options(Strategy::Rewrite)).expect("compacts");
    assert_eq!(compacted.generation, 4);
    let len_after = fs::metadata(&source.path).expect("exists").len();
    // The dead bytes are gone; the rest differs only by the size of the new index.
    let (footer_after, _) = index_of(&source.path);
    assert_eq!(
        compacted.dead_bytes_before + footer_before.index_stored_len,
        len_before - len_after + footer_after.index_stored_len
    );
    assert_eq!(layers(&source.path).overhead, overhead_before);
    let again = edit(&source.path, &[], &options(Strategy::Rewrite)).expect("compacts");
    assert_eq!(again.dead_bytes_before, 0);
    assert_members(
        &source.path,
        &[
            ("base", source.bytes_of("base")),
            ("tuned", source.bytes_of("tuned")),
            ("y", source.bytes_of("y")),
        ],
    );
}

#[test]
fn a_rewrite_of_a_container_that_is_still_open() {
    let scratch = Scratch::new();
    let source = pack(&scratch, "pack.zt", Mode::Fast, false);
    let ops = [EditOp::Remove { name: "x".to_owned() }];
    // The caller's own handle, dropped as the contract asks: every platform.
    let held = Container::open(&source.path, &OpenOptions::default()).expect("opens");
    drop(held);
    edit(&source.path, &ops, &options(Strategy::Rewrite)).expect("rewrites");

    // A second handle kept open across the rename.
    let open = Container::open(&source.path, &OpenOptions::default()).expect("opens");
    let ops = [EditOp::Remove { name: "y".to_owned() }];
    let result = edit(&source.path, &ops, &options(Strategy::Rewrite));
    if cfg!(windows) {
        // std's rename falls back to POSIX semantics where the file system allows it; whether it does
        // under every Windows file system (and wine) is not established, so both outcomes pass.
        match &result {
            Ok(_) => eprintln!("rename over an open container succeeded"),
            Err(Error::Io { op: "rename", .. }) => eprintln!("rename over an open container refused"),
            Err(other) => panic!("{other:?}"),
        }
    } else {
        result.expect("a rename over an open file succeeds on Unix");
        // The open handle still reads the generation it opened.
        assert_eq!(open.members().len(), 3);
        open.member("y").expect("still there").verify().expect("still verifies");
    }
    drop(open);
    let container = Container::open(&source.path, &OpenOptions::default()).expect("opens");
    assert!(container.members().len() <= 3);
    let leftovers: Vec<String> =
        scratch.entries().into_iter().filter(|name| name.ends_with(TEMP_FILE_SUFFIX)).collect();
    assert_eq!(leftovers, Vec::<String>::new(), "no temporary file stays");
}

#[test]
fn an_in_place_rewrite_keeps_the_base_of_a_kept_delta_and_sets_bit_0_as_its_blobs_say() {
    let scratch = Scratch::new();
    let source = pack(&scratch, "pack.zt", Mode::Max, true);
    let before = fs::metadata(&source.path).expect("exists").len();
    // The member `base` goes, its blob stays: `tuned` is stored as delta blobs over it.
    let ops = [EditOp::Remove { name: "base".to_owned() }];
    let report = edit(&source.path, &ops, &options(Strategy::InPlace)).expect("rewrites");
    assert_eq!((report.kept, report.removed, report.generation), (3, 1, 2));
    let after = fs::metadata(&source.path).expect("exists").len();
    assert!(after <= before, "{after} above {before}");
    assert_ne!(required_features(&source.path) & FEATURE_DELTA, 0);
    let (_, index) = index_of(&source.path);
    assert!(reached_delta_blobs(&index) > 0);
    assert_members(
        &source.path,
        &[("tuned", source.bytes_of("tuned")), ("x", source.bytes_of("x")), ("y", source.bytes_of("y"))],
    );

    // With the delta member gone, no delta blob is left, and bit 0 is cleared, as the format requires.
    let ops = [EditOp::Remove { name: "tuned".to_owned() }];
    edit(&source.path, &ops, &options(Strategy::InPlace)).expect("rewrites");
    assert_eq!(required_features(&source.path) & FEATURE_DELTA, 0);
    let (footer, index) = index_of(&source.path);
    assert_eq!(footer.generation, 3);
    assert!(index.blobs().iter().all(|blob| blob.kind != BLOB_KIND_DELTA), "dead blobs are dropped");
    assert!(fs::metadata(&source.path).expect("exists").len() < after / 2);
    assert_members(&source.path, &[("x", source.bytes_of("x")), ("y", source.bytes_of("y"))]);
}

#[test]
fn an_in_place_rewrite_stores_a_delta_against_a_file_of_the_same_edit() {
    let scratch = Scratch::new();
    let source = pack(&scratch, "pack.zt", Mode::Max, false);
    assert_eq!(required_features(&source.path) & FEATURE_DELTA, 0);
    let base2 = finetune(source.bytes_of("base"), 3);
    let tuned2 = finetune(&base2, 22);
    let ops = [
        EditOp::Remove { name: "x".to_owned() },
        EditOp::Add { path: scratch.file("base2.in", &base2), name: "base2".to_owned(), delta_base: None },
        EditOp::Add {
            path: scratch.file("tuned2.in", &tuned2),
            name: "tuned2".to_owned(),
            delta_base: Some("base2".to_owned()),
        },
    ];
    let report = edit(&source.path, &ops, &options(Strategy::InPlace)).expect("rewrites");
    assert!(report.encode.members[4].delta_blobs > 0, "{:?}", report.encode.members[4]);
    assert_ne!(required_features(&source.path) & FEATURE_DELTA, 0);
    assert_members(
        &source.path,
        &[
            ("base", source.bytes_of("base")),
            ("tuned", source.bytes_of("tuned")),
            ("y", source.bytes_of("y")),
            ("base2", &base2),
            ("tuned2", &tuned2),
        ],
    );
}

#[test]
fn an_in_place_compaction_drops_what_an_append_left_behind() {
    let scratch = Scratch::new();
    let source = pack(&scratch, "pack.zt", Mode::Max, true);
    let overhead_before = layers(&source.path).overhead;
    let ops = [EditOp::Remove { name: "x".to_owned() }];
    edit(&source.path, &ops, &options(Strategy::Append)).expect("appends");
    let len_before = fs::metadata(&source.path).expect("exists").len();
    let (footer_before, _) = index_of(&source.path);
    let entries = scratch.entries();

    let compacted = edit(&source.path, &[], &options(Strategy::InPlace)).expect("compacts");
    assert_eq!(scratch.entries(), entries, "no temporary file");
    assert_eq!(compacted.generation, 3);
    assert!(compacted.dead_bytes_before > 0);
    let len_after = fs::metadata(&source.path).expect("exists").len();
    let (footer_after, _) = index_of(&source.path);
    assert_eq!(
        compacted.dead_bytes_before + footer_before.index_stored_len,
        len_before - len_after + footer_after.index_stored_len
    );
    assert_eq!(layers(&source.path).overhead, overhead_before);
    let again = edit(&source.path, &[], &options(Strategy::InPlace)).expect("compacts");
    assert_eq!(again.dead_bytes_before, 0);
    assert_eq!(fs::metadata(&source.path).expect("exists").len(), len_after);
    assert_members(
        &source.path,
        &[
            ("base", source.bytes_of("base")),
            ("tuned", source.bytes_of("tuned")),
            ("y", source.bytes_of("y")),
        ],
    );
}
