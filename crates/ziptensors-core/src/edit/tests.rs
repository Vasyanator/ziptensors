//! File: edit/tests.rs
//! Purpose: edit tests that need a test-only seam, and the resolution rules of `plan.rs`. Everything
//! a public caller can do is in `tests/edit.rs`.
//!
//! Main responsibilities:
//! - `plan::resolve`: order of the result, every refusal, which names are held to the restore rules;
//! - the append protocol under crash injection (`SinkFault`) at each phase: inside the new data,
//!   inside the stored index, inside the footer; failed writes and failed truncations; verification
//!   before visibility (`PackBuilder::set_fault` corrupting the new data);
//! - [`torn_append_fixture`], a container whose append died midway, for the tests of `recover`.
//! - an append writes the file it opened and read, even when another file is renamed over its path
//!   between the plan and the sink (`Hooks::before_sink`, Unix only);
//! - the in-place planner (`inplace::plan_moves`) on random layouts, its moves simulated on a byte
//!   array against the untouched original (P1-P3), merging and refusals;
//! - the in-place strategy under faults: no temporary file at any point, a death after the first
//!   move (damaged, and `recover` never hands back a wrong file), a failed write after the moves
//!   (the moved chunks intact, P4), a failed final cut (`TrailingBytes`, which `recover` repairs,
//!   also past the dead footers of earlier appended generations), a result that does not verify
//!   (never indexed).
//!
//! Implementation notes: the byte offsets of each phase come from a probe edit of a copy, since the
//! output depends only on the inputs and the mode. Inputs are seeded and live in a unique temporary
//! directory, which is removed.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use crate::error::Error;
use crate::format::{FOOTER_LEN, HEADER_LEN, parse_footer};
use crate::io::TEMP_FILE_SUFFIX;
use crate::read::testkit::{Rng, tensor_like};
use crate::read::{Container, OpenOptions};
use crate::recover::{RecoverOptions, recover};
use crate::types::Mode;
use crate::write::{EncodeOptions, NameRule, PackBuilder, SinkFault, SinkFaultMode};

use super::inplace::{self, MovePlan, plan_moves};
use super::plan::{Base, Step, resolve};
use super::{EditOp, EditOptions, Hooks, Strategy, edit_with};

/// A unique scratch directory, removed on drop.
pub(crate) struct Dir(pub(crate) PathBuf);

impl Dir {
    pub(crate) fn new(tag: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let name = format!("ziptensors-edit-{tag}-{}-{id}", std::process::id());
        let path = std::env::temp_dir().join(name);
        fs::create_dir_all(&path).expect("scratch directory");
        Self(path)
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

impl Drop for Dir {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("cannot remove {}: {error}", self.0.display());
        }
    }
}

/// Writes `bytes` to `name` in `dir`.
fn write_file(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, bytes).expect("scratch file");
    path
}

// ---- plan::resolve -------------------------------------------------------------------------

fn add(name: &str, base: Option<&str>) -> EditOp {
    EditOp::Add {
        path: PathBuf::from(format!("/in/{name}")),
        name: name.to_owned(),
        delta_base: base.map(str::to_owned),
    }
}

fn replace(name: &str, base: Option<&str>) -> EditOp {
    EditOp::Replace {
        name: name.to_owned(),
        path: PathBuf::from(format!("/in/{name}")),
        delta_base: base.map(str::to_owned),
    }
}

fn remove(name: &str) -> EditOp {
    EditOp::Remove { name: name.to_owned() }
}

const MEMBERS: [&str; 3] = ["a", "b", "c"];

#[test]
fn the_result_keeps_order_replaces_in_the_slot_and_adds_at_the_end() {
    let ops = [add("e", None), remove("a"), replace("c", None), add("d", None)];
    let plan = resolve(&MEMBERS, &ops).expect("resolves");
    let names: Vec<&str> = plan.steps.iter().map(Step::name).collect();
    assert_eq!(names, ["b", "c", "e", "d"]);
    assert!(matches!(&plan.steps[0], Step::Keep { .. }));
    assert!(matches!(&plan.steps[1], Step::File { .. }));
    assert_eq!((plan.kept, plan.added, plan.removed, plan.replaced), (1, 2, 1, 1));
    assert!(!plan.has_delta());
    let empty = resolve(&MEMBERS, &[]).expect("a compaction");
    assert_eq!(empty.steps.len(), 3);
    assert_eq!(empty.kept, 3);
}

#[test]
fn delta_bases_resolve_against_the_result() {
    let ops = [replace("b", Some("a")), add("d", Some("b")), add("e", Some("c"))];
    let plan = resolve(&MEMBERS, &ops).expect("resolves");
    let base_of = |name: &str| {
        plan.steps.iter().find_map(|step| match step {
            Step::File { name: own, delta_base, .. } if own == name => Some(delta_base.clone()),
            _ => None,
        })
    };
    assert_eq!(base_of("b"), Some(Some(Base::Kept("a".to_owned()))));
    assert_eq!(base_of("d"), Some(Some(Base::New("b".to_owned()))), "the replaced content");
    assert_eq!(base_of("e"), Some(Some(Base::Kept("c".to_owned()))));
    assert!(plan.has_delta());
    assert_eq!(plan.kept_bases, ["a".to_owned(), "c".to_owned()].into_iter().collect());
}

#[test]
fn every_refusal_of_an_operation_set() {
    let refused = |ops: &[EditOp]| match resolve(&MEMBERS, ops) {
        Err(Error::EditRefused { detail }) => detail,
        other => panic!("{ops:?} gave {other:?}"),
    };
    refused(&[remove("a"), replace("a", None)]);
    refused(&[add("d", None), add("d", None)]);
    refused(&[add("a", None)]);
    refused(&[add("d", Some("d"))]);
    refused(&[remove("a"), add("d", Some("a"))]);
    // The base comes later in the result.
    refused(&[replace("a", Some("d")), add("d", None)]);
    refused(&[add("d", Some("e")), add("e", None)]);
    let missing = |ops: &[EditOp]| {
        let error = resolve(&MEMBERS, ops).expect_err("refused");
        assert!(matches!(error, Error::MemberNotFound { .. }), "{ops:?} gave {error:?}");
    };
    missing(&[remove("x")]);
    missing(&[replace("x", None)]);
    missing(&[add("d", Some("x"))]);
    let error = resolve(&MEMBERS, &[add("../d", None)]).expect_err("unsafe");
    assert!(matches!(error, Error::UnsafeMemberName { .. }), "{error:?}");
}

// ---- the append protocol under faults ------------------------------------------------------

/// Packs members `a`, `b`, `c` of seeded tensor-like bytes into `pack.zt` in `dir`.
fn packed(dir: &Path) -> PathBuf {
    let output = dir.join("pack.zt");
    let options = EncodeOptions { mode: Mode::Max, threads: 2 };
    let mut builder = PackBuilder::new(&output, &options).expect("starts");
    for (seed, name) in (1_u64..).zip(MEMBERS) {
        let input = write_file(dir, &format!("{name}.in"), &tensor_like(seed, 300_000, 2));
        builder.add_file(&input, name).expect("adds");
    }
    builder.finish().expect("commits");
    output
}

/// The edit every fault test applies: remove `b`, add `d` of new bytes.
fn ops(dir: &Path) -> Vec<EditOp> {
    let input = write_file(dir, "d.in", &tensor_like(9, 400_000, 2));
    vec![remove("b"), EditOp::Add { path: input, name: "d".to_owned(), delta_base: None }]
}

fn append_options() -> EditOptions {
    EditOptions { strategy: Strategy::Append, threads: 2, ..EditOptions::default() }
}

/// Where the parts of an append land, past the source end `source_len`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AppendLayout {
    /// Length of the source container.
    pub(crate) source_len: u64,
    /// Bytes of new chunks.
    pub(crate) data: u64,
    /// Bytes of the new stored index.
    pub(crate) index: u64,
}

/// The layout of the append of [`ops`] to the container of [`packed`], from a probe on a copy.
fn layout(dir: &Path, source: &Path, ops: &[EditOp]) -> AppendLayout {
    let probe = dir.join("probe.zt");
    fs::copy(source, &probe).expect("copies");
    let source_len = fs::metadata(source).expect("exists").len();
    edit_with(&probe, ops, &append_options(), Hooks::default()).expect("the probe appends");
    let bytes = fs::read(&probe).expect("reads");
    fs::remove_file(&probe).expect("removes the probe");
    let at = bytes.len() - 64;
    let footer = parse_footer(&bytes[at..].try_into().expect("64 bytes")).expect("valid footer");
    AppendLayout {
        source_len,
        data: footer.index_offset - source_len,
        index: footer.index_stored_len,
    }
}

/// Applies [`ops`] under `Append` with `hooks`, expecting a failure.
fn failed_append(path: &Path, ops: &[EditOp], hooks: Hooks) -> Error {
    edit_with(path, ops, &append_options(), hooks).expect_err("the fault must fail the edit")
}

fn sink_fault(after_bytes: u64, mode: SinkFaultMode) -> Hooks {
    Hooks { sink_fault: Some(SinkFault { after_bytes, mode }), ..Hooks::default() }
}

/// A container whose append died after `after_bytes` bytes past its end reached the disk, as a
/// process killed mid-append leaves it: the source, then a partial tail.
#[derive(Debug)]
pub(crate) struct TornAppend {
    /// The torn file.
    pub(crate) path: PathBuf,
    /// The bytes of the container before the append, the first `source.len()` bytes of the file.
    pub(crate) source: Vec<u8>,
    /// Where the append's parts would have landed.
    pub(crate) layout: AppendLayout,
}

/// Builds a [`TornAppend`] in `dir` (which must exist): a container of three members, an append that
/// removes one and adds one, killed by `SinkFaultMode::Die` once `after_bytes` bytes past the old end
/// are written. `after_bytes` below `layout.data` tears the new data, up to `layout.data +
/// layout.index` the stored index, and below that plus 64 the footer. With `after_bytes` at or past
/// the full tail the append completes and the file is a valid two-generation container.
pub(crate) fn torn_append_fixture(dir: &Path, after_bytes: u64) -> TornAppend {
    let path = packed(dir);
    let ops = ops(dir);
    let layout = layout(dir, &path, &ops);
    let source = fs::read(&path).expect("reads");
    let hooks = sink_fault(after_bytes, SinkFaultMode::Die);
    let result = edit_with(&path, &ops, &append_options(), hooks);
    let tail = layout.data + layout.index + FOOTER_LEN;
    assert_eq!(result.is_ok(), after_bytes >= tail, "{result:?}");
    TornAppend { path, source, layout }
}

/// Asserts that the file at `path` starts with `source` and does not open.
fn assert_torn(path: &Path, source: &[u8], torn_len: u64) {
    let bytes = fs::read(path).expect("reads");
    assert_eq!(bytes.len() as u64, torn_len);
    assert!(bytes[..source.len()] == *source, "the source bytes changed");
    let error = Container::open(path, &OpenOptions::default()).expect_err("a torn file must not open");
    assert!(matches!(error, Error::MalformedIndex { .. }), "{error:?}");
}

#[test]
fn a_death_in_each_phase_of_an_append_leaves_the_source_and_a_torn_tail() {
    let dir = Dir::new("die");
    let path = packed(&dir.0);
    let ops = ops(&dir.0);
    let layout = layout(&dir.0, &path, &ops);
    let source = fs::read(&path).expect("reads");
    assert!(layout.data > 200 && layout.index > 2, "{layout:?}");
    let phases = [
        ("new data", layout.data / 2),
        ("stored index", layout.data + layout.index / 2),
        ("footer", layout.data + layout.index + FOOTER_LEN / 2),
    ];
    for (phase, after_bytes) in phases {
        fs::write(&path, &source).expect("restores the source");
        let error = failed_append(&path, &ops, sink_fault(after_bytes, SinkFaultMode::Die));
        assert!(matches!(error, Error::Io { op: "write", .. }), "{phase}: {error:?}");
        assert_torn(&path, &source, layout.source_len + after_bytes);
    }
}

#[test]
fn the_torn_append_fixture_tears_where_asked() {
    let dir = Dir::new("fixture");
    let torn = torn_append_fixture(&dir.0, 100);
    assert_torn(&torn.path, &torn.source, torn.layout.source_len + 100);
    let full = torn.layout.data + torn.layout.index + FOOTER_LEN;
    let whole = torn_append_fixture(&dir.0, full);
    let container = Container::open(&whole.path, &OpenOptions::default()).expect("opens");
    let names: Vec<&str> = container.members().iter().map(|info| info.name.as_str()).collect();
    assert_eq!(names, ["a", "c", "d"]);
}

#[test]
fn a_failed_write_in_each_phase_cuts_the_file_back_to_the_source() {
    let dir = Dir::new("failwrite");
    let path = packed(&dir.0);
    let ops = ops(&dir.0);
    let layout = layout(&dir.0, &path, &ops);
    let source = fs::read(&path).expect("reads");
    for after_bytes in [0, layout.data / 2, layout.data + 1, layout.data + layout.index + 10] {
        let error = failed_append(&path, &ops, sink_fault(after_bytes, SinkFaultMode::FailWrite));
        assert!(matches!(error, Error::Io { .. }), "at {after_bytes}: {error:?}");
        assert!(fs::read(&path).expect("reads") == source, "at {after_bytes}: not cut back");
        let container = Container::open(&path, &OpenOptions::default()).expect("the source opens");
        assert_eq!(container.members().len(), 3);
    }
    assert_eq!(dir.entries(), ["a.in", "b.in", "c.in", "d.in", "pack.zt"]);
}

/// Flips one bit near the end of the file. Under an append the seam runs before the new index is
/// written, so these are the last bytes of the last new chunk.
fn flip_last_new_chunk(path: &Path) {
    let mut bytes = fs::read(path).expect("reads");
    let at = bytes.len() - 4;
    bytes[at] ^= 0x10;
    fs::write(path, bytes).expect("writes");
}

#[test]
fn a_new_generation_that_does_not_verify_never_becomes_visible() {
    let dir = Dir::new("verify");
    let path = packed(&dir.0);
    let ops = ops(&dir.0);
    let source = fs::read(&path).expect("reads");
    let hooks = Hooks { builder_fault: Some(flip_last_new_chunk), ..Hooks::default() };
    let error = failed_append(&path, &ops, hooks);
    assert!(
        matches!(
            error,
            Error::ChunkCorrupt { .. }
                | Error::ChecksumMismatch { .. }
                | Error::VerifyFailed { .. }
                | Error::Zstd { .. }
        ),
        "{error:?}"
    );
    assert!(fs::read(&path).expect("reads") == source, "the file must be the source again");
}

#[test]
fn an_append_that_cannot_be_cut_back_reports_trailing_bytes_at_the_old_length() {
    let dir = Dir::new("trailing");
    let path = packed(&dir.0);
    let ops = ops(&dir.0);
    let source = fs::read(&path).expect("reads");
    let hooks = Hooks {
        builder_fault: Some(flip_last_new_chunk),
        sink_fault: Some(SinkFault { after_bytes: 0, mode: SinkFaultMode::FailSetLen }),
        ..Hooks::default()
    };
    let error = failed_append(&path, &ops, hooks);
    let Error::TrailingBytes { path: reported, valid_len } = error else {
        panic!("{error:?}");
    };
    assert_eq!(reported, path);
    assert_eq!(valid_len, source.len() as u64);
    let bytes = fs::read(&path).expect("reads");
    assert!(bytes.len() > source.len());
    assert!(bytes[..source.len()] == *source, "the source bytes changed");
}

#[test]
fn a_rewrite_that_does_not_verify_leaves_the_source_and_no_temporary_file() {
    let dir = Dir::new("rewrite");
    let path = packed(&dir.0);
    let ops = ops(&dir.0);
    let source = fs::read(&path).expect("reads");
    let hooks = Hooks { builder_fault: Some(flip_header_neighbour), ..Hooks::default() };
    let options = EditOptions { threads: 2, ..EditOptions::default() };
    let error = edit_with(&path, &ops, &options, hooks).expect_err("corrupted");
    assert!(
        matches!(error, Error::ChunkCorrupt { .. } | Error::ChecksumMismatch { .. }),
        "{error:?}"
    );
    assert!(fs::read(&path).expect("reads") == source, "the source must stay");
    assert_eq!(dir.entries(), ["a.in", "b.in", "c.in", "d.in", "pack.zt"]);
}

/// Flips one bit of the first chunk of a fresh file, right after its header.
fn flip_header_neighbour(path: &Path) {
    let mut bytes = fs::read(path).expect("reads");
    bytes[64 + 8] ^= 0x10;
    fs::write(path, bytes).expect("writes");
}

#[test]
fn only_an_added_name_is_held_to_the_restore_rules() {
    let ops = [add("e", None), replace("c", None)];
    let plan = resolve(&MEMBERS, &ops).expect("resolves");
    let rules: Vec<(&str, NameRule)> =
        plan.steps.iter().map(|step| (step.name(), step.name_rule())).collect();
    assert_eq!(
        rules,
        [("a", NameRule::Kept), ("b", NameRule::Kept), ("c", NameRule::Kept), ("e", NameRule::New)]
    );
}

// ---- the append writes the file it read ----------------------------------------------------

/// Where [`swap_in`] finds the file it renames over the container.
#[cfg(unix)]
fn swap_path(path: &Path) -> PathBuf {
    path.with_extension("swap")
}

/// The seam between the plan and the append sink: renames [`swap_path`] over `path`.
#[cfg(unix)]
fn swap_in(path: &Path) {
    fs::rename(swap_path(path), path).expect("swaps the container");
}

/// Unix only: Windows may refuse a rename over a file the edit holds open.
#[cfg(unix)]
#[test]
fn an_append_writes_the_file_it_read_when_the_path_is_swapped() {
    let dir = Dir::new("swap");
    let path = packed(&dir.0);
    let ops = ops(&dir.0);
    let source = fs::read(&path).expect("reads");
    // Another valid container of the same length: the source under another generation.
    let at = source.len() - 64;
    let mut footer = parse_footer(&source[at..].try_into().expect("64 bytes")).expect("valid");
    footer.generation += 7;
    let other = [&source[..at], &footer.to_bytes()].concat();
    fs::write(swap_path(&path), &other).expect("writes the other container");
    // The file the edit opens stays reachable after the swap unlinks it from `path`.
    let original = dir.0.join("original.zt");
    fs::hard_link(&path, &original).expect("links");

    let hooks = Hooks { before_sink: Some(swap_in), ..Hooks::default() };
    let report = edit_with(&path, &ops, &append_options(), hooks).expect("appends");

    assert!(fs::read(&path).expect("reads") == other, "the file swapped in was written");
    let written = fs::read(&original).expect("reads");
    assert!(written.len() > source.len());
    assert!(written[..source.len()] == *source, "the source bytes changed");
    let container = Container::open(&original, &OpenOptions::default()).expect("opens");
    let names: Vec<&str> = container.members().iter().map(|info| info.name.as_str()).collect();
    assert_eq!(names, ["a", "c", "d"]);
    for name in names {
        container.member(name).expect("present").verify().expect("verifies");
    }
    let at = written.len() - 64;
    let generation = parse_footer(&written[at..].try_into().expect("64 bytes")).expect("valid");
    assert_eq!(generation.generation, report.generation);
}

// ---- the in-place planner ------------------------------------------------------------------

/// A random set of chunk ranges inside `[64, data_end)` as an index may hold them: disjoint, with
/// gaps (dead data), some repeated (identical records of blobs that share chunks) and some empty,
/// sorted and each once as `live_ranges` hands them over. Returns the ranges and `data_end`.
fn random_ranges(rng: &mut Rng) -> (Vec<(u64, u32)>, u64) {
    let mut ranges = Vec::new();
    let mut at = HEADER_LEN + rng.below(3) * rng.below(50);
    for _ in 0..rng.below(40) {
        let len = match rng.below(6) {
            0 => 0,
            1 => 1 + rng.below(3),
            _ => 1 + rng.below(300),
        };
        let len = u32::try_from(len).expect("small");
        ranges.push((at, len));
        if rng.below(4) == 0 {
            // A record shared by a second blob: the same range again.
            ranges.push((at, len));
        }
        at += u64::from(len);
        if rng.below(2) == 0 {
            at += rng.below(200);
        }
    }
    let data_end = at + rng.below(30);
    ranges.sort_unstable();
    ranges.dedup();
    (ranges, data_end)
}

/// Applies `plan` to `file` as `Sink::copy_within` does, forward in pieces of `piece` bytes, and
/// asserts on the way that every piece is read before any of its bytes are written (P2, P3) and
/// that every move goes towards the header (P1).
fn simulate(plan: &MovePlan, file: &mut [u8], piece: usize) {
    let mut written = vec![false; file.len()];
    let mut position = usize::try_from(HEADER_LEN).expect("small");
    for mv in &plan.moves {
        let [src, dst, len] = [mv.src, mv.dst, mv.len].map(|v| usize::try_from(v).expect("small"));
        assert!(dst <= src, "P1: {mv:?}");
        assert_eq!(dst, position, "moves follow each other: {mv:?}");
        let mut done = 0;
        while src != dst && done < len {
            let n = piece.min(len - done);
            let read = src + done..src + done + n;
            assert!(!written[read.clone()].contains(&true), "P2: {mv:?} reads overwritten bytes");
            let bytes = file[read].to_vec();
            file[dst + done..dst + done + n].copy_from_slice(&bytes);
            written[dst + done..dst + done + n].fill(true);
            done += n;
        }
        position = dst + len;
    }
    assert_eq!(position as u64, plan.compacted_end);
}

#[test]
fn the_planner_moves_every_kept_range_once_without_overwriting_unread_bytes() {
    let mut rng = Rng(0x1234);
    for case in 0..500 {
        let (ranges, data_end) = random_ranges(&mut rng);
        let plan = plan_moves(&ranges, data_end).expect("a valid layout plans");
        let original: Vec<u8> =
            (0..data_end + 64).map(|_| u8::try_from(rng.below(256)).expect("a byte")).collect();
        let mut file = original.clone();
        let piece = usize::try_from(1 + rng.below(16)).expect("small");
        simulate(&plan, &mut file, piece);
        // Identical records collapsed: one place per distinct range.
        assert_eq!(plan.offsets.len(), ranges.len(), "case {case}");
        let live: u64 = ranges.iter().map(|(_, len)| u64::from(*len)).sum();
        assert_eq!(plan.compacted_end, HEADER_LEN + live, "case {case}");
        let mut expected_dst = HEADER_LEN;
        for (src, len) in &ranges {
            let dst = plan.offsets[&(*src, *len)];
            assert_eq!(dst, expected_dst, "case {case}: destinations follow the source order");
            expected_dst += u64::from(*len);
            let [src, dst, len] = [*src, dst, u64::from(*len)].map(|v| usize::try_from(v).expect("small"));
            assert!(file[dst..dst + len] == original[src..src + len], "case {case}: bytes of {src}");
        }
        assert!(plan.moves.windows(2).all(|pair| pair[0].src + pair[0].len < pair[1].src
            || pair[0].dst + pair[0].len < pair[1].dst), "case {case}: adjacent moves merge");
    }
}

#[test]
fn adjacent_ranges_merge_and_ranges_in_place_do_not_move() {
    let plan = plan_moves(&[(64, 10), (74, 10), (100, 10), (110, 5), (200, 5)], 300).expect("plans");
    let moves: Vec<(u64, u64, u64)> = plan.moves.iter().map(|mv| (mv.src, mv.dst, mv.len)).collect();
    assert_eq!(moves, [(64, 64, 20), (100, 84, 15), (200, 99, 5)]);
    assert_eq!(plan.compacted_end, 104);
    assert_eq!(plan.offsets[&(110, 5)], 94);
    let empty = plan_moves(&[], 300).expect("plans");
    assert_eq!((empty.moves.len(), empty.compacted_end), (0, HEADER_LEN));
}

#[test]
fn the_planner_refuses_a_layout_it_cannot_order() {
    let refused = |ranges: &[(u64, u32)], data_end: u64| {
        let error = plan_moves(ranges, data_end).expect_err("unsafe");
        assert!(matches!(error, Error::EditRefused { .. }), "{ranges:?}: {error:?}");
    };
    refused(&[(64, 10), (70, 10)], 300);
    refused(&[(100, 10), (64, 10)], 300);
    refused(&[(63, 10)], 300);
    refused(&[(64, 10), (290, 20)], 300);
    refused(&[(u64::MAX - 1, 10)], u64::MAX);
}

// ---- the in-place strategy under faults ----------------------------------------------------

fn in_place_options() -> EditOptions {
    EditOptions { strategy: Strategy::InPlace, threads: 2, ..EditOptions::default() }
}

/// The in-place edit of the fault tests: remove `a`, the first member, so that `b` and `c` move;
/// add `d` of new bytes.
fn in_place_ops(dir: &Path) -> Vec<EditOp> {
    let input = write_file(dir, "d.in", &tensor_like(9, 400_000, 2));
    vec![remove("a"), EditOp::Add { path: input, name: "d".to_owned(), delta_base: None }]
}

/// The plan [`in_place_ops`] makes of the container at `path`.
fn in_place_plan(path: &Path) -> MovePlan {
    let container = Container::open(path, &OpenOptions::default()).expect("opens");
    let kept: Vec<_> = ["b", "c"]
        .map(|name| *container.member(name).expect("present").record())
        .into_iter()
        .collect();
    let index_offset = {
        let bytes = fs::read(path).expect("reads");
        let at = bytes.len() - 64;
        parse_footer(&bytes[at..].try_into().expect("64 bytes")).expect("valid").index_offset
    };
    inplace::plan(container.index(), &kept, index_offset).expect("plans").moves
}

/// Bytes the moves of `plan` copy.
fn moved_bytes(plan: &MovePlan) -> u64 {
    plan.moves.iter().filter(|mv| mv.src != mv.dst).map(|mv| mv.len).sum()
}

/// The result of [`in_place_ops`] on a copy of `source`, by a successful edit.
fn in_place_probe(dir: &Path, source: &Path, ops: &[EditOp]) -> Vec<u8> {
    let probe = dir.join("probe.zt");
    fs::copy(source, &probe).expect("copies");
    edit_with(&probe, ops, &in_place_options(), Hooks::default()).expect("the probe rewrites");
    let bytes = fs::read(&probe).expect("reads");
    fs::remove_file(&probe).expect("removes the probe");
    bytes
}

/// Asserts that `error` is the damage of an in-place rewrite of `path`, and returns its cause.
fn damaged(error: Error, path: &Path) -> Error {
    let Error::DamagedInPlace { path: at, cause } = error else { panic!("{error:?}") };
    assert_eq!(at, path);
    *cause
}

/// The builder seam: asserts that no temporary file is beside the container. It runs just before
/// verification, when every byte of the result is on disk.
fn no_temporary_file(path: &Path) {
    let parent = path.parent().expect("a directory");
    for entry in fs::read_dir(parent).expect("readable") {
        let name = entry.expect("entry").file_name().to_string_lossy().into_owned();
        assert!(!name.ends_with(TEMP_FILE_SUFFIX), "{name} during an in-place rewrite");
    }
}

#[test]
fn an_in_place_rewrite_never_makes_a_temporary_file_and_shrinks() {
    let dir = Dir::new("inplace");
    let path = packed(&dir.0);
    let ops = in_place_ops(&dir.0);
    let before = fs::metadata(&path).expect("exists").len();
    let entries = dir.entries();
    let hooks = Hooks { builder_fault: Some(no_temporary_file), ..Hooks::default() };
    let report = edit_with(&path, &ops, &in_place_options(), hooks).expect("rewrites");
    assert_eq!(dir.entries(), entries, "no file appeared or vanished");
    assert_eq!(report.encode.output_len, fs::metadata(&path).expect("exists").len());
    let container = Container::open(&path, &OpenOptions::default()).expect("opens");
    let names: Vec<&str> = container.members().iter().map(|info| info.name.as_str()).collect();
    assert_eq!(names, ["b", "c", "d"]);
    for name in names {
        container.member(name).expect("present").verify().expect("verifies");
    }
    // Two members out, none in: the file shrinks below the source.
    let ops = [remove("b"), remove("d")];
    edit_with(&path, &ops, &in_place_options(), Hooks::default()).expect("rewrites");
    assert!(fs::metadata(&path).expect("exists").len() < before);
}

#[test]
fn a_death_after_the_first_move_is_damage_that_recover_never_hides() {
    let dir = Dir::new("inplacedie");
    let path = packed(&dir.0);
    let ops = in_place_ops(&dir.0);
    let plan = in_place_plan(&path);
    let first = plan.moves.iter().find(|mv| mv.src != mv.dst).expect("a real move");
    assert!(first.len > 1000, "{first:?}");
    let source_len = fs::metadata(&path).expect("exists").len();
    // The seal (64 bytes of the old footer), then 1000 bytes into the first move.
    let error = edit_with(&path, &ops, &in_place_options(), sink_fault(64 + 1000, SinkFaultMode::Die))
        .expect_err("dies");
    let cause = damaged(error, &path);
    assert!(matches!(cause, Error::Io { op: "write", .. }), "{cause:?}");
    assert_eq!(fs::metadata(&path).expect("exists").len(), source_len, "nothing was cut");
    assert!(Container::open(&path, &OpenOptions::default()).is_err(), "a torn rewrite must not open");
    let error = recover(&path, &RecoverOptions::default()).expect_err("nothing valid to recover");
    assert!(
        matches!(
            error,
            Error::Unrecoverable { .. }
                | Error::VerifyFailed { .. }
                | Error::ChecksumMismatch { .. }
                | Error::ChunkCorrupt { .. }
        ),
        "{error:?}"
    );
    assert_eq!(fs::metadata(&path).expect("exists").len(), source_len, "recover cut nothing");
}

#[test]
fn a_failed_write_after_the_moves_leaves_the_kept_chunks_in_place() {
    let dir = Dir::new("inplacefail");
    let path = packed(&dir.0);
    let ops = in_place_ops(&dir.0);
    let plan = in_place_plan(&path);
    let probe = in_place_probe(&dir.0, &path, &ops);
    // The seal, every move, then into the new data of `d`.
    let after_bytes = 64 + moved_bytes(&plan) + 5000;
    let hooks = sink_fault(after_bytes, SinkFaultMode::FailWrite);
    let error = edit_with(&path, &ops, &in_place_options(), hooks).expect_err("fails");
    let cause = damaged(error, &path);
    assert!(matches!(cause, Error::Io { op: "write", .. }), "{cause:?}");
    // P4: only dead bytes were overwritten after the moves, so the kept chunks are where the
    // complete rewrite has them.
    let end = usize::try_from(plan.compacted_end).expect("fits");
    let bytes = fs::read(&path).expect("reads");
    assert!(bytes[64..end] == probe[64..end], "the moved kept chunks differ");
}

#[test]
fn an_in_place_rewrite_that_cannot_be_cut_reports_trailing_bytes_and_recover_fixes_it() {
    let dir = Dir::new("inplacetrail");
    let path = packed(&dir.0);
    let source_len = fs::metadata(&path).expect("exists").len();
    let hooks = sink_fault(0, SinkFaultMode::FailSetLen);
    let ops = [remove("a")];
    let error = edit_with(&path, &ops, &in_place_options(), hooks).expect_err("the cut fails");
    let Error::TrailingBytes { path: at, valid_len } = error else { panic!("{error:?}") };
    assert_eq!(at, path);
    assert!(valid_len < source_len, "a removal shrinks");
    assert_eq!(fs::metadata(&path).expect("exists").len(), source_len, "not cut");
    assert!(Container::open(&path, &OpenOptions::default()).is_err(), "trailing bytes do not open");
    let report = recover(&path, &RecoverOptions::default()).expect("recovers");
    assert!(report.truncated);
    assert_eq!(report.valid_len, valid_len);
    let container = Container::open(&path, &OpenOptions::default()).expect("opens");
    let names: Vec<&str> = container.members().iter().map(|info| info.name.as_str()).collect();
    assert_eq!(names, ["b", "c"]);
    for name in names {
        container.member(name).expect("present").verify().expect("verifies");
    }
}

#[test]
fn an_in_place_result_that_does_not_verify_is_never_indexed() {
    let dir = Dir::new("inplaceverify");
    let path = packed(&dir.0);
    let ops = in_place_ops(&dir.0);
    let hooks = Hooks { builder_fault: Some(flip_header_neighbour), ..Hooks::default() };
    let error = edit_with(&path, &ops, &in_place_options(), hooks).expect_err("corrupted");
    let cause = damaged(error, &path);
    assert!(
        matches!(cause, Error::ChunkCorrupt { .. } | Error::ChecksumMismatch { .. } | Error::Zstd { .. }),
        "{cause:?}"
    );
    assert!(Container::open(&path, &OpenOptions::default()).is_err(), "no new index was written");
}

#[test]
fn recover_finds_the_rewritten_container_past_dead_generations_after_a_failed_cut() {
    let dir = Dir::new("inplacetrailgen");
    let path = packed(&dir.0);
    let first_len = fs::metadata(&path).expect("exists").len();
    edit_with(&path, &[remove("a")], &append_options(), Hooks::default()).expect("appends");
    let hooks = sink_fault(0, SinkFaultMode::FailSetLen);
    let error = edit_with(&path, &[remove("b")], &in_place_options(), hooks).expect_err("no cut");
    let Error::TrailingBytes { valid_len, .. } = error else { panic!("{error:?}") };
    // The footer of the first generation is still in the dead tail, with its index, and passes the
    // structural checks; only its decode shows that its chunks moved.
    assert!(valid_len < first_len - 64, "{valid_len} {first_len}");
    let report = recover(&path, &RecoverOptions::default()).expect("recovers");
    assert_eq!((report.valid_len, report.truncated), (valid_len, true));
    let container = Container::open(&path, &OpenOptions::default()).expect("opens");
    let names: Vec<&str> = container.members().iter().map(|info| info.name.as_str()).collect();
    assert_eq!(names, ["c"]);
    container.member("c").expect("present").verify().expect("verifies");
}
