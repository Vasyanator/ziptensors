//! File: `recover_tests.rs`
//! Purpose: tests of `recover.rs`: the scan's block seams and bounds, which candidates stop the scan,
//! and the tool on real files written by the writer: a valid file left alone, a torn append cut back,
//! verification before truncation, generations that fail it passed over for an older one that
//! verifies (within the candidate bound), the count of candidates passed over, a dry run, and (Unix)
//! a cut that lands on the verified file when another is renamed over the path before it.
//!
//! Implementation notes: a second generation is built by bytes, as an append without new data writes
//! it: the stored index copied after the old footer, then a footer naming the copy with the next
//! generation. Byte-level accept and reject cases are in `tests/conformance.rs`, hostile files in
//! `tests/malicious.rs`.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::{
    Candidate, RecoverOptions, SCAN_BLOCK, ScanBudget, ScanOutcome, find_last_valid_generation,
    recover,
};
#[cfg(unix)]
use super::recover_with;
use crate::error::{Error, ErrorCode};
use crate::format::{FOOTER_LEN, Footer, parse_footer};
use crate::io::RandomAccess;
use crate::read::testkit::MemorySource;
use crate::read::{Container, OpenOptions};
use crate::types::Mode;
use crate::write::{EncodeOptions, encode_file};

/// A unique scratch directory, removed on drop.
struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Self {
        let name = format!("ziptensors-recover-{tag}-{}", std::process::id());
        let path = std::env::temp_dir().join(name);
        fs::create_dir_all(&path).expect("scratch directory");
        Self(path)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("cannot remove {}: {error}", self.0.display());
        }
    }
}

/// Content of the packed member: 200 000 bytes that compress somewhat.
fn content() -> Vec<u8> {
    (0..200_000_u32).map(|at| at.wrapping_mul(2_654_435_761).to_le_bytes()[3] & 0x3F).collect()
}

/// The bytes of a real container of one member, [`content`], written by the writer.
fn packed(dir: &Dir) -> Vec<u8> {
    packed_with(dir, &content())
}

/// The bytes of a real container of one member holding `bytes`, written by the writer.
fn packed_with(dir: &Dir, bytes: &[u8]) -> Vec<u8> {
    let input = dir.0.join("input.bin");
    fs::write(&input, bytes).expect("input");
    let output = dir.0.join("packed.ziptensors");
    let options = EncodeOptions { mode: Mode::Max, threads: 1 };
    encode_file(&input, &output, &options).expect("packs");
    let bytes = fs::read(&output).expect("reads");
    fs::remove_file(&output).expect("removes");
    bytes
}

/// The footer at the end of a valid `file`.
fn footer_of(file: &[u8]) -> Footer {
    let tail = file.len() - 64;
    parse_footer(file[tail..].try_into().expect("64 bytes")).expect("a valid footer")
}

/// `file` with a next generation appended that names the same records: its stored index copied after
/// the old footer, and a footer for the copy with the generation one higher.
fn append_generation(file: &[u8]) -> Vec<u8> {
    let footer = footer_of(file);
    let start = usize::try_from(footer.index_offset).expect("small");
    let stored = &file[start..file.len() - 64];
    let mut next = file.to_vec();
    next.extend_from_slice(stored);
    let moved = Footer {
        index_offset: file.len() as u64,
        generation: footer.generation + 1,
        ..footer
    };
    next.extend_from_slice(&moved.to_bytes());
    next
}

/// Garbage of `len` bytes that holds no `end_magic`.
fn garbage(len: usize) -> Vec<u8> {
    (0..len).map(|at| u8::try_from(at % 199).expect("small") ^ 0x55).collect()
}

fn source(bytes: Vec<u8>) -> Arc<dyn RandomAccess> {
    Arc::new(MemorySource::silent(bytes))
}

fn scan(bytes: Vec<u8>, budget: &ScanBudget) -> Result<super::Scan, Error> {
    find_last_valid_generation(&source(bytes), &OpenOptions::default(), budget)
}

fn found(valid_len: usize, generation: u64) -> ScanOutcome {
    ScanOutcome::Found(Candidate { valid_len: valid_len as u64, generation })
}

fn write(dir: &Dir, name: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.0.join(name);
    fs::write(&path, bytes).expect("writes");
    path
}

fn opens(path: &Path) -> bool {
    Container::open(path, &OpenOptions::default()).is_ok()
}

#[test]
fn a_valid_file_is_found_at_its_end_by_the_scan_and_left_alone_by_the_tool() {
    let dir = Dir::new("valid");
    let file = append_generation(&packed(&dir));
    let result = scan(file.clone(), &ScanBudget::default()).expect("scans");
    assert_eq!(result.outcome, found(file.len(), 2));
    assert_eq!(result.candidates_checked, 1);

    let path = write(&dir, "c", &file);
    let report = recover(&path, &RecoverOptions::default()).expect("recovers");
    assert_eq!((report.file_len, report.valid_len), (file.len() as u64, file.len() as u64));
    assert_eq!((report.generation, report.truncated), (2, false));
    assert_eq!((report.verified_bytes, report.scanned_bytes, report.passed_over), (None, 0, 0));
    assert_eq!(fs::read(&path).expect("reads"), file);
}

#[test]
fn a_torn_third_generation_is_cut_back_to_the_second_and_verified() {
    let dir = Dir::new("torn");
    let second = append_generation(&packed(&dir));
    let mut file = second.clone();
    // A torn append: new data, half a stored index, no footer.
    file.extend_from_slice(&garbage(5000));
    let path = write(&dir, "c", &file);
    assert!(!opens(&path));
    let report = recover(&path, &RecoverOptions::default()).expect("recovers");
    assert_eq!(report.valid_len, second.len() as u64);
    assert_eq!((report.generation, report.truncated), (2, true));
    assert_eq!(report.verified_bytes, Some(content().len() as u64));
    // The torn tail holds no footer whose hash holds: nothing was passed over.
    assert_eq!(report.passed_over, 0);
    assert_eq!(fs::read(&path).expect("reads"), second);
    assert!(opens(&path));
}

/// Where [`swap_in`] finds the file it renames over the recovered one.
#[cfg(unix)]
fn swap_path(path: &Path) -> PathBuf {
    path.with_extension("swap")
}

/// The seam between the verification and the cut: renames [`swap_path`] over `path`.
#[cfg(unix)]
fn swap_in(path: &Path) {
    fs::rename(swap_path(path), path).expect("swaps the file");
}

/// Unix only: Windows may refuse a rename over a file the recovery holds open.
#[cfg(unix)]
#[test]
fn the_cut_lands_on_the_file_that_was_verified_when_the_path_is_swapped() {
    let dir = Dir::new("swap");
    let second = append_generation(&packed(&dir));
    let mut file = second.clone();
    file.extend_from_slice(&garbage(5000));
    let path = write(&dir, "c", &file);
    // Another file of the same length, which a cut through the path would shorten.
    let other = garbage(file.len());
    fs::write(swap_path(&path), &other).expect("writes the other file");
    // The file the recovery opens stays reachable after the swap unlinks it from `path`.
    let original = dir.0.join("original");
    fs::hard_link(&path, &original).expect("links");

    let report = recover_with(&path, &RecoverOptions::default(), Some(swap_in)).expect("recovers");

    assert_eq!((report.valid_len, report.truncated), (second.len() as u64, true));
    assert!(fs::read(&path).expect("reads") == other, "the file swapped in was cut");
    assert_eq!(fs::read(&original).expect("reads"), second);
    assert!(opens(&original));
}

#[test]
fn a_dry_run_reports_the_cut_and_writes_nothing() {
    let dir = Dir::new("dry");
    let first = packed(&dir);
    let mut file = first.clone();
    file.extend_from_slice(&garbage(300));
    let path = write(&dir, "c", &file);
    let options = RecoverOptions { dry_run: true, ..RecoverOptions::default() };
    let report = recover(&path, &options).expect("reports");
    assert_eq!((report.valid_len, report.truncated), (first.len() as u64, false));
    assert_eq!(report.verified_bytes, Some(content().len() as u64));
    assert_eq!(report.passed_over, 0);
    assert_eq!(fs::read(&path).expect("reads"), file);
}

/// An older footer and index intact, its chunks overwritten, as an interrupted rewrite in place
/// leaves a file: the generation passes the structural checks, and only the decode shows it.
fn moved_chunks(dir: &Dir) -> (Vec<u8>, usize) {
    let second = append_generation(&packed(dir));
    let valid_len = second.len();
    let mut file = second;
    // The data region moved down by 100 bytes, as a compaction would move a later chunk.
    file.copy_within(64 + 100..64 + 4000, 64);
    file.extend_from_slice(&garbage(700));
    (file, valid_len)
}

/// Both generations of the file have moved chunks: each fails verification, and the error of the
/// latest is reported.
#[test]
fn a_generation_whose_chunks_moved_fails_verification_and_the_file_keeps_its_length() {
    let dir = Dir::new("moved");
    let (file, _) = moved_chunks(&dir);
    let path = write(&dir, "c", &file);
    let error = recover(&path, &RecoverOptions::default()).expect_err("the chunks moved");
    let code = error.code();
    assert!(
        matches!(code, ErrorCode::VerifyFailed | ErrorCode::ChecksumMismatch | ErrorCode::ChunkCorrupt),
        "{error:?}"
    );
    assert_eq!(fs::read(&path).expect("reads"), file);
}

/// A short container whose file goes on with the tail of a longer, three-generation one, as a
/// rewrite in place whose final cut failed leaves a file that had appended generations: the rewrite
/// zeroed the last footer before its first move, and the two older footers lie after the new
/// container, their indexes intact, their chunks overwritten by the new one. Returns the file and
/// the length of the new container.
fn new_container_before_dead_generations(dir: &Dir) -> (Vec<u8>, usize) {
    let first = packed(dir);
    let old_index = usize::try_from(footer_of(&first).index_offset).expect("small");
    let mut old = append_generation(&append_generation(&first));
    let sealed = old.len() - 64;
    old[sealed..].fill(0);
    let new = packed_with(dir, &content()[..3000]);
    assert!(new.len() < old_index, "the old indexes must survive after the new container");
    let mut file = new.clone();
    file.extend_from_slice(&old[new.len()..]);
    (file, new.len())
}

#[test]
fn generations_that_do_not_verify_are_passed_over_for_an_older_one_that_does() {
    let dir = Dir::new("passover");
    let (file, valid_len) = new_container_before_dead_generations(&dir);
    // Both dead generations pass the structural checks: without verification the latest is taken.
    let scanned = scan(file.clone(), &ScanBudget::default()).expect("scans");
    let ScanOutcome::Found(latest) = scanned.outcome else { panic!("{scanned:?}") };
    assert_eq!(latest.generation, 2);
    assert!(latest.valid_len > valid_len as u64, "{latest:?}");
    let path = write(&dir, "c", &file);
    // A dry run reports what the cut would report, and writes nothing.
    let options = RecoverOptions { dry_run: true, ..RecoverOptions::default() };
    let dry = recover(&path, &options).expect("reports");
    assert_eq!(fs::read(&path).expect("reads"), file);
    let report = recover(&path, &RecoverOptions::default()).expect("recovers");
    assert_eq!((report.valid_len, report.truncated), (valid_len as u64, true));
    assert_eq!(report.verified_bytes, Some(3000));
    // Both dead generations were found and failed verification.
    assert_eq!(report.passed_over, 2);
    assert_eq!(dry, super::RecoverReport { truncated: false, ..report });
    assert_eq!(fs::read(&path).expect("reads"), file[..valid_len]);
    assert!(opens(&path));
}

/// One newer generation that fails verification is counted once, and the older one that verifies
/// is the result.
#[test]
fn a_newer_generation_that_fails_verification_is_counted_as_passed_over() {
    let dir = Dir::new("passone");
    let (mut file, valid_len) = new_container_before_dead_generations(&dir);
    // Zero the footer of the latest dead generation: one dead generation is left.
    let scanned = scan(file.clone(), &ScanBudget::default()).expect("scans");
    let ScanOutcome::Found(latest) = scanned.outcome else { panic!("{scanned:?}") };
    let end = usize::try_from(latest.valid_len).expect("small");
    file[end - 64..end].fill(0);
    let path = write(&dir, "c", &file);
    let report = recover(&path, &RecoverOptions::default()).expect("recovers");
    assert_eq!((report.valid_len, report.passed_over), (valid_len as u64, 1));
    assert_eq!(report.verified_bytes, Some(3000));
}

/// Footers whose hash holds but whose layout does not fit count as passed over; `end_magic` with a
/// bad hash does not.
#[test]
fn structurally_invalid_footers_count_as_passed_over_and_torn_ones_do_not() {
    let dir = Dir::new("passfake");
    let first = packed(&dir);
    let fake = Footer { index_offset: 64, ..footer_of(&first) }.to_bytes();
    let mut torn = fake;
    torn[0] ^= 1;
    let mut file = first.clone();
    for _ in 0..3 {
        file.extend_from_slice(&fake);
        file.extend_from_slice(&torn);
    }
    let path = write(&dir, "c", &file);
    let report = recover(&path, &RecoverOptions::default()).expect("recovers");
    assert_eq!((report.valid_len, report.passed_over), (first.len() as u64, 3));
}

#[test]
fn verification_failures_count_against_the_candidate_bound() {
    let dir = Dir::new("passbound");
    let (file, _) = new_container_before_dead_generations(&dir);
    let path = write(&dir, "c", &file);
    // Two candidates allowed: both dead generations, which fail; the new one is never reached.
    let budget = ScanBudget { max_candidates: 2, ..ScanBudget::default() };
    let options = RecoverOptions { budget, ..RecoverOptions::default() };
    let error = recover(&path, &options).expect_err("the bound stops the scan");
    let code = error.code();
    assert!(
        matches!(code, ErrorCode::VerifyFailed | ErrorCode::ChecksumMismatch | ErrorCode::ChunkCorrupt),
        "the first verification error is reported: {error:?}"
    );
    assert_eq!(fs::read(&path).expect("reads"), file);
}

#[test]
fn without_verification_the_same_file_is_cut_to_the_structural_generation() {
    let dir = Dir::new("noverify");
    let (file, valid_len) = moved_chunks(&dir);
    let path = write(&dir, "c", &file);
    let options = RecoverOptions { verify: false, ..RecoverOptions::default() };
    let report = recover(&path, &options).expect("cuts");
    assert_eq!((report.valid_len, report.truncated), (valid_len as u64, true));
    assert_eq!(report.verified_bytes, None);
    assert_eq!(fs::read(&path).expect("reads"), file[..valid_len]);
}

/// A footer that starts in one block and ends in the next is found, at every offset around the seam
/// of the first two blocks, and the bytes read follow from the block layout.
#[test]
fn a_footer_across_the_seam_of_two_blocks_is_found() {
    let dir = Dir::new("seam");
    let first = packed(&dir);
    let block = usize::try_from(SCAN_BLOCK).expect("small");
    for tail in [block - 66, block - 65, block - 64, block - 63, block - 30, block - 1, block, block + 1] {
        let mut file = first.clone();
        file.extend_from_slice(&garbage(tail));
        let result = scan(file, &ScanBudget::default()).expect("scans");
        assert_eq!(result.outcome, found(first.len(), 1), "tail {tail}");
        // The first block holds the footer whole while `tail + 64 <= block`; otherwise the second
        // block, which overlaps the first by 63 bytes, finds it.
        let blocks = if tail + 64 <= block { 1 } else { 2 };
        assert!(result.scanned_bytes <= blocks * SCAN_BLOCK, "tail {tail}: {result:?}");
        assert!(result.scanned_bytes > (blocks - 1) * SCAN_BLOCK, "tail {tail}: {result:?}");
    }
}

/// A scan over several blocks counts each byte it read once: the blocks overlap by 63 bytes, read
/// twice, so a scan that reaches the header reports exactly the bytes between it and the end.
#[test]
fn scanned_bytes_count_the_overlap_of_blocks_once() {
    let dir = Dir::new("overlap");
    let mut file = packed(&dir);
    // Break the only footer's hash, so that no candidate exists and the scan reads everything.
    let footer = file.len() - 64;
    file[footer] ^= 1;
    file.extend_from_slice(&garbage(3 * usize::try_from(SCAN_BLOCK).expect("small") + 12_345));
    let len = file.len() as u64;
    let result = scan(file.clone(), &ScanBudget::default()).expect("scans");
    assert_eq!((result.outcome, result.scanned_bytes), (ScanOutcome::NoCandidate, len - 64));
    // A budget is spent on the same count: one byte short of the whole file stops the scan.
    let budget = ScanBudget { max_scan_bytes: len - 65, ..ScanBudget::default() };
    let result = scan(file, &budget).expect("scans");
    assert_eq!((result.outcome, result.scanned_bytes), (ScanOutcome::ScanBytesExhausted, len - 65));
}

/// The byte bound is exact: the scan reads at most what it was allowed, and a footer just inside the
/// allowance is found.
#[test]
fn the_scan_byte_bound_is_exact() {
    let dir = Dir::new("bytes");
    let first = packed(&dir);
    let tail = 1000;
    let mut file = first.clone();
    file.extend_from_slice(&garbage(tail));
    let needed = (tail + 64) as u64;
    let budget = |max_scan_bytes| ScanBudget { max_scan_bytes, ..ScanBudget::default() };
    let result = scan(file.clone(), &budget(needed)).expect("scans");
    assert_eq!((result.outcome, result.scanned_bytes), (found(first.len(), 1), needed));
    let result = scan(file.clone(), &budget(needed - 1)).expect("scans");
    assert_eq!((result.outcome, result.scanned_bytes), (ScanOutcome::ScanBytesExhausted, needed - 1));
    let result = scan(file, &budget(63)).expect("scans");
    assert_eq!((result.outcome, result.scanned_bytes), (ScanOutcome::ScanBytesExhausted, 0));
}

/// Footers with a valid hash and a layout that does not fit count as candidates; `end_magic` with a
/// bad hash does not.
#[test]
fn the_candidate_bound_counts_only_footers_whose_hash_holds() {
    let dir = Dir::new("candidates");
    let first = packed(&dir);
    let fake = Footer { index_offset: 64, ..footer_of(&first) }.to_bytes();
    let mut torn = fake;
    torn[0] ^= 1;
    let mut file = first.clone();
    for _ in 0..3 {
        file.extend_from_slice(&fake);
        file.extend_from_slice(&torn);
    }
    let budget = |max_candidates| ScanBudget { max_candidates, ..ScanBudget::default() };
    let result = scan(file.clone(), &budget(3)).expect("scans");
    assert_eq!((result.outcome, result.candidates_checked), (ScanOutcome::CandidatesExhausted, 3));
    let result = scan(file.clone(), &budget(4)).expect("scans");
    assert_eq!((result.outcome, result.candidates_checked), (found(first.len(), 1), 4));
    let result = scan(file, &budget(0)).expect("scans");
    assert_eq!((result.outcome, result.candidates_checked), (ScanOutcome::CandidatesExhausted, 0));
}

#[test]
fn a_file_without_a_valid_header_is_not_scanned() {
    let dir = Dir::new("header");
    let mut file = packed(&dir);
    file[0] ^= 1;
    let error = scan(file, &ScanBudget::default()).expect_err("no header");
    assert!(matches!(error, Error::NotZiptensors), "{error:?}");
    let error = scan(vec![0; 10], &ScanBudget::default()).expect_err("too short");
    assert!(matches!(error, Error::NotZiptensors), "{error:?}");
}

#[test]
fn a_file_too_short_for_any_footer_has_no_candidate() {
    let dir = Dir::new("short");
    let file = packed(&dir);
    let result = scan(file[..64 + 63].to_vec(), &ScanBudget::default()).expect("scans");
    assert_eq!((result.outcome, result.scanned_bytes), (ScanOutcome::NoCandidate, 0));
    let result = scan(file[..64 + 64].to_vec(), &ScanBudget::default()).expect("scans");
    assert_eq!((result.outcome, result.scanned_bytes), (ScanOutcome::NoCandidate, 64));
}

/// A genuine generation over the caller's index cap stops the scan: skipping it would cut it off for
/// an older one. The tool reports the limit and changes nothing.
#[test]
fn a_generation_over_the_callers_limits_stops_the_scan_instead_of_being_cut_off() {
    let dir = Dir::new("limits");
    let second = append_generation(&packed(&dir));
    let mut file = second.clone();
    file.extend_from_slice(&garbage(100));
    let raw_len = footer_of(&second).index_raw_len;
    let open = OpenOptions { max_index_bytes: raw_len - 1, ..OpenOptions::default() };
    let error = find_last_valid_generation(&source(file.clone()), &open, &ScanBudget::default())
        .expect_err("over the cap");
    assert!(matches!(error, Error::BudgetExceeded { what: "index", .. }), "{error:?}");

    let path = write(&dir, "c", &file);
    let options = RecoverOptions { open, ..RecoverOptions::default() };
    let error = recover(&path, &options).expect_err("over the cap");
    assert!(matches!(error, Error::BudgetExceeded { .. }), "{error:?}");
    assert_eq!(fs::read(&path).expect("reads"), file);
}

#[test]
fn a_file_with_no_valid_generation_is_unrecoverable_and_unchanged() {
    let dir = Dir::new("none");
    let mut file = packed(&dir);
    let footer_at = file.len() - 64;
    file[footer_at] ^= 1;
    let path = write(&dir, "c", &file);
    let error = recover(&path, &RecoverOptions::default()).expect_err("nothing valid");
    assert!(matches!(&error, Error::Unrecoverable { scanned, .. } if *scanned > 0), "{error:?}");
    assert_eq!(fs::read(&path).expect("reads"), file);
}

#[test]
fn a_missing_file_is_an_io_error() {
    let dir = Dir::new("missing");
    let error = recover(&dir.0.join("absent"), &RecoverOptions::default()).expect_err("absent");
    assert!(matches!(error, Error::Io { op: "open", .. }), "{error:?}");
}

#[test]
fn the_defaults_verify_and_bound_the_candidates() {
    let options = RecoverOptions::default();
    assert!(options.verify && !options.dry_run);
    assert_eq!(options.budget, ScanBudget { max_candidates: 64, max_scan_bytes: u64::MAX });
    assert_eq!(options.open, OpenOptions::default());
    assert_eq!(FOOTER_LEN, 64);
}
