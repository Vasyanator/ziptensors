//! File: recover.rs
//! Purpose: the explicit recovery tool of spec section 9.3: shorten a file whose last 64 bytes do not
//! end a valid container to its latest valid generation.
//!
//! Main responsibilities:
//! - [`find_last_valid_generation`], the pure, read-only core: a backward search for `end_magic`, and
//!   for each candidate the reader's own steps 1 to 4 of spec section 9.1 on the prefix that ends
//!   with it, under a [`ScanBudget`];
//! - [`recover`], the tool: nothing if the file opens as it is, otherwise the scan, a full decode of
//!   every member of each structurally valid generation from the end (on by default) until one
//!   verifies, then one truncation and a sync, all through one handle on the file, opened once.
//!
//! Key structures: [`RecoverOptions`], [`RecoverReport`], [`ScanBudget`], [`Scan`], [`ScanOutcome`],
//! [`Candidate`].
//!
//! Key functions: [`recover`], [`find_last_valid_generation`].
//!
//! Implementation notes: a candidate is judged by `Container::from_source` over an
//! [`io::PrefixSource`](crate::io::PrefixSource), so the rules a generation must pass are the reader's
//! and have no second copy here. A reader never recovers on its own: `Container::open` knows nothing
//! of this module. A candidate is skipped only when it is shown not to be a generation (its index
//! does not decompress or does not validate); a failure that only a genuine generation can reach,
//! such as an index over the caller's limits or an unknown codec behind valid hashes, stops the scan
//! with that error, because truncating past it could cut off real data. Hazard: after an interrupted
//! rewrite in place an older footer may still be in the file while its chunks were moved; it passes
//! the structural checks, which is why [`RecoverOptions::verify`] defaults to on. A generation that
//! fails verification is passed over like one that fails the structural checks, and the scan goes
//! on to the next older candidate under the same budget: after a rewrite in place whose final cut
//! failed, the dead tail may hold the footer of an older generation, later in the file than the
//! new one and no longer true.

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::error::Error;
use crate::format::{END_MAGIC, FOOTER_LEN, HEADER_LEN, parse_footer, parse_header};
use crate::io::{FileSource, PrefixSource, RandomAccess};
use crate::read::{Container, OpenOptions};

/// Bytes read per step of the backward scan. Footers straddling two blocks are found because
/// consecutive blocks overlap by `FOOTER_LEN - 1` bytes.
const SCAN_BLOCK: u64 = 1 << 20;
/// Default of [`ScanBudget::max_candidates`].
pub const DEFAULT_MAX_CANDIDATES: u32 = 64;

/// Bounds of one backward scan (spec section 9.3: a hostile file can hold many self-consistent
/// footers, so both the candidates checked and the bytes scanned are bounded).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanBudget {
    /// Most candidates given a full check: a footer whose hash is valid, opened as a prefix. Each
    /// costs at most the index cap of the open options plus its validation. An `end_magic` whose
    /// footer hash fails costs scan bytes only and is not counted.
    pub max_candidates: u32,
    /// Most distinct bytes of the file the backward scan reads, as [`Scan::scanned_bytes`] counts
    /// them: the `FOOTER_LEN - 1` bytes by which consecutive blocks overlap are read twice but
    /// counted once. Index reads of candidates are bounded by `max_candidates` instead.
    pub max_scan_bytes: u64,
}

impl Default for ScanBudget {
    fn default() -> Self {
        Self { max_candidates: DEFAULT_MAX_CANDIDATES, max_scan_bytes: u64::MAX }
    }
}

/// A valid generation found by the scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    /// Length of the prefix that holds the generation: its footer ends here.
    pub valid_len: u64,
    /// The `generation` field of its footer, informational.
    pub generation: u64,
}

/// How a scan ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanOutcome {
    /// The latest valid generation, the first one from the end that passes.
    Found(Candidate),
    /// The scan reached the header and no candidate passed.
    NoCandidate,
    /// [`ScanBudget::max_candidates`] candidates failed and another one was found.
    CandidatesExhausted,
    /// [`ScanBudget::max_scan_bytes`] was spent before the scan reached the header.
    ScanBytesExhausted,
}

/// The result of [`find_last_valid_generation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scan {
    /// How the scan ended.
    pub outcome: ScanOutcome,
    /// Distinct bytes of the file the backward scan read, between the header and the end of the
    /// file. The scan reads whole blocks backwards from the end, so this covers the block that held
    /// the footer found, not only the bytes after it. Blocks overlap by `FOOTER_LEN - 1` bytes,
    /// which are counted once, so this is never above `file_len - HEADER_LEN`, nor above
    /// [`ScanBudget::max_scan_bytes`].
    pub scanned_bytes: u64,
    /// Candidates given a full check; never above [`ScanBudget::max_candidates`].
    pub candidates_checked: u32,
}

/// How to [`recover`] a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecoverOptions {
    /// Limits every candidate is judged under, and threads and chunk checks of the verification.
    /// Limits tighter than the writer's stop the scan at a genuine generation instead of skipping
    /// it, so the defaults are the right choice unless the file is known to need more.
    pub open: OpenOptions,
    /// Bounds of the backward scan.
    pub budget: ScanBudget,
    /// Decode every member of the found generation and check its BLAKE3 before anything is cut,
    /// spec section 9.3 step 3. On by default; off only when the file is known never to have been
    /// rewritten in place.
    pub verify: bool,
    /// Report what would be done and write nothing.
    pub dry_run: bool,
}

impl Default for RecoverOptions {
    fn default() -> Self {
        Self {
            open: OpenOptions::default(),
            budget: ScanBudget::default(),
            verify: true,
            dry_run: false,
        }
    }
}

/// What [`recover`] found and did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecoverReport {
    /// Length of the file before recovery.
    pub file_len: u64,
    /// Length of the valid container: `file_len` when the file opened as it was.
    pub valid_len: u64,
    /// The `generation` of the container ending at `valid_len`.
    pub generation: u64,
    /// Whether the file was cut to `valid_len`. False when nothing needed cutting, and on a dry run.
    pub truncated: bool,
    /// Sum of the lengths of the members decoded and checked, or `None` when nothing was verified:
    /// verification off, or the file opened as it was.
    pub verified_bytes: Option<u64>,
    /// [`Scan::scanned_bytes`] of the backward scan: distinct bytes read, at most
    /// `file_len - 64`; 0 when the file opened as it was.
    pub scanned_bytes: u64,
    /// Candidate generations later in the file than the one returned that the scan found and
    /// rejected: footers whose hash holds (the candidates [`ScanBudget::max_candidates`] counts)
    /// whose prefix did not open as a container, or, with [`RecoverOptions::verify`], did not
    /// verify. An `end_magic` whose footer hash fails is not a candidate and is not counted, so a
    /// torn append reports 0. 0 when the file opened as it was. Above 0, data that a newer
    /// generation may have held is in the bytes cut off.
    pub passed_over: u64,
}

/// Shortens the container at `path` to its latest valid generation, spec section 9.3. Never
/// automatic: only an explicit request runs this.
///
/// A file that opens as it is is left alone. Otherwise the scan of [`find_last_valid_generation`]
/// looks for the latest valid generation. With [`RecoverOptions::verify`] every member of it is
/// decoded and checked against its BLAKE3, and a generation that fails is passed over: the scan
/// goes on to the next older candidate, within the same [`ScanBudget`] (so at most
/// `max_candidates` full decodes), and the result is the latest generation that verifies. Without
/// it the result is the latest structurally valid generation. [`RecoverReport::passed_over`] counts
/// the newer candidates rejected on the way, by either check. Then, unless
/// [`RecoverOptions::dry_run`], the file is truncated to the generation's end and synced. Nothing below that end is written. Every error but
/// a failed sync after the cut leaves the file unchanged; that one leaves it cut to the generation's
/// end, its durability unconfirmed. The file is opened once (read and write unless
/// [`RecoverOptions::dry_run`], which opens it read only), and the scan, the verification and the cut
/// all go through that handle, so a file renamed over `path` meanwhile is never cut. Nothing else may
/// write the file meanwhile; a length change between the scan and the cut is refused.
///
/// # Errors
/// - [`Error::Unrecoverable`] if no valid generation was found, or a scan bound was reached first.
/// - [`Error::NotZiptensors`], [`Error::UnsupportedVersion`] or [`Error::MalformedIndex`] for
///   `"header"` if the header is not a valid ziptensors header: no generation can be valid then.
/// - [`Error::BudgetExceeded`], [`Error::UnknownCodec`] or [`Error::UnsupportedFeature`] if a
///   candidate that may be a genuine generation cannot be judged under these options; the scan
///   stops there rather than cut it off.
/// - With verification, if no generation verifies but at least one passed the structural checks:
///   the verification error of the latest such generation ([`Error::VerifyFailed`],
///   [`Error::ChecksumMismatch`], [`Error::ChunkCorrupt`], ...), the first one met, which says why
///   the newest candidate was refused; how many failed and how the scan ended are logged at ERROR.
///   [`Error::Unrecoverable`] is kept for a scan that found no structurally valid generation. An
///   [`Error::Io`] during a verification stops the scan with that error.
/// - [`Error::Io`] if the file cannot be opened (for writing too, unless a dry run, even when it
///   turns out valid as it is) or read, has changed length since the scan, or cannot be truncated.
///   With op `"sync"` after the cut: the file IS cut to the generation's end, but the cut may not
///   survive a power loss. [`Error::CommittedNotDurable`] is not used for it: that variant names a
///   failed sync of a directory after a rename.
pub fn recover(path: &Path, options: &RecoverOptions) -> Result<RecoverReport, Error> {
    recover_with(path, options, None)
}

/// [`recover`] with a test seam: `before_cut` runs on `path` between the verification and the cut.
/// Production passes `None`.
fn recover_with(
    path: &Path,
    options: &RecoverOptions,
    before_cut: Option<fn(&Path)>,
) -> Result<RecoverReport, Error> {
    let started = Instant::now();
    tracing::info!(
        path = %path.display(),
        verify = options.verify,
        dry_run = options.dry_run,
        "recovering container"
    );
    let (source, cutter) = open(path, options.dry_run)?;
    let source: Arc<dyn RandomAccess> = Arc::new(source);
    let file_len = source.len();

    // Step 1 of spec section 9.3: a file valid as it is needs nothing.
    if check_candidate(&source, file_len, &options.open)? {
        let generation = footer_at(source.as_ref(), file_len)?;
        tracing::info!(path = %path.display(), file_len, generation, "the file is valid as it is");
        return Ok(RecoverReport {
            file_len,
            valid_len: file_len,
            generation,
            truncated: false,
            verified_bytes: None,
            scanned_bytes: 0,
            passed_over: 0,
        });
    }

    let (scan, candidate, verified_bytes, rejected) =
        scan_and_verify(&source, options, path, file_len)?;
    // The returned candidate is the last one checked; every one before it was rejected. A found
    // candidate was counted, so this never saturates.
    let passed_over = u64::from(scan.candidates_checked.saturating_sub(1));
    tracing::info!(
        path = %path.display(),
        file_len,
        valid_len = candidate.valid_len,
        generation = candidate.generation,
        scanned = scan.scanned_bytes,
        candidates = scan.candidates_checked,
        passed_over,
        "found the latest valid generation"
    );

    if rejected > 0 {
        tracing::warn!(
            path = %path.display(),
            rejected,
            valid_len = candidate.valid_len,
            "later generations did not verify; recovering an older one"
        );
    }
    drop(source);
    if let Some(seam) = before_cut {
        seam(path);
    }

    let truncated = match &cutter {
        Some(file) => {
            truncate(file, path, file_len, candidate.valid_len)?;
            true
        }
        None => false,
    };
    tracing::info!(
        path = %path.display(),
        file_len,
        valid_len = candidate.valid_len,
        truncated,
        elapsed = ?started.elapsed(),
        "recovery finished"
    );
    Ok(RecoverReport {
        file_len,
        valid_len: candidate.valid_len,
        generation: candidate.generation,
        truncated,
        verified_bytes,
        scanned_bytes: scan.scanned_bytes,
        passed_over,
    })
}

/// Steps 2 and 3 of spec section 9.3 for [`recover`]: the scan of `source` (`file_len` bytes, the
/// file at `path`), with every structurally valid candidate verified when `options.verify` says so
/// and passed over if it fails. Returns the scan, the generation found, the bytes verified (`None`
/// without verification) and how many candidates failed verification.
///
/// # Errors
/// Every error of [`find_last_valid_generation`]; [`Error::Io`] of a verification; the first
/// verification error if candidates failed it and none passed; [`Error::Unrecoverable`] if no
/// candidate passed the structural checks.
fn scan_and_verify(
    source: &Arc<dyn RandomAccess>,
    options: &RecoverOptions,
    path: &Path,
    file_len: u64,
) -> Result<(Scan, Candidate, Option<u64>, u32), Error> {
    // Step 3 of spec section 9.3 inside the scan: a candidate that does not verify is passed over.
    let mut verified_bytes = None;
    let mut rejected = 0_u32;
    let mut first_failure: Option<Error> = None;
    let accept = |candidate: &Candidate| {
        if !options.verify {
            return Ok(true);
        }
        match verify_generation(source, candidate.valid_len, &options.open, path) {
            Ok(bytes) => {
                verified_bytes = Some(bytes);
                Ok(true)
            }
            // A read that fails would fail every older candidate alike.
            Err(error @ Error::Io { .. }) => Err(error),
            Err(error) => {
                rejected += 1;
                first_failure.get_or_insert(error);
                Ok(false)
            }
        }
    };
    let scan = scan_generations(source, &options.open, &options.budget, accept)?;
    match scan.outcome {
        ScanOutcome::Found(candidate) => Ok((scan, candidate, verified_bytes, rejected)),
        ScanOutcome::NoCandidate
        | ScanOutcome::CandidatesExhausted
        | ScanOutcome::ScanBytesExhausted => {
            if let Some(error) = first_failure {
                tracing::error!(
                    path = %path.display(),
                    file_len,
                    rejected,
                    outcome = ?scan.outcome,
                    scanned = scan.scanned_bytes,
                    candidates = scan.candidates_checked,
                    %error,
                    "no generation verifies; the file was not changed"
                );
                return Err(error);
            }
            tracing::warn!(
                path = %path.display(),
                file_len,
                outcome = ?scan.outcome,
                scanned = scan.scanned_bytes,
                candidates = scan.candidates_checked,
                "no valid generation found"
            );
            Err(Error::Unrecoverable { path: path.to_path_buf(), scanned: scan.scanned_bytes })
        }
    }
}

/// Finds the latest valid generation in `source`, spec section 9.3 step 2. Reads only.
///
/// Every offset `p` with `64 ≤ p ≤ len − 64` at which `end_magic` ends at `p + 64` is a candidate,
/// from the end backwards, the end of the source included. A candidate whose footer hash holds is
/// checked by opening the prefix `0..p + 64` as a whole container under `options` (one thread): the
/// reader's steps 1 to 4 of spec section 9.1 with `file_len = p + 64`. The first that opens is the
/// result. Both bounds of `budget` are enforced; reaching one is an outcome, not an error.
///
/// # Errors
/// - [`Error::NotZiptensors`], [`Error::UnsupportedVersion`], [`Error::MalformedIndex`] for
///   `"header"`: the header is invalid, so no prefix can be valid.
/// - [`Error::BudgetExceeded`], [`Error::UnknownCodec`], [`Error::UnsupportedFeature`]: a candidate
///   behind a valid footer hash cannot be judged under `options`; see the module notes.
/// - [`Error::Io`] or [`Error::OutOfBounds`] if `source` fails a read.
pub fn find_last_valid_generation(
    source: &Arc<dyn RandomAccess>,
    options: &OpenOptions,
    budget: &ScanBudget,
) -> Result<Scan, Error> {
    scan_generations(source, options, budget, |_| Ok(true))
}

/// [`find_last_valid_generation`] with a further test: a candidate that opens is the result only if
/// `accept` returns `true` for it; on `false` the scan goes on to the next older candidate, the
/// candidate counted and the bytes read as for any other. An error of `accept` stops the scan.
fn scan_generations(
    source: &Arc<dyn RandomAccess>,
    options: &OpenOptions,
    budget: &ScanBudget,
    mut accept: impl FnMut(&Candidate) -> Result<bool, Error>,
) -> Result<Scan, Error> {
    let len = source.len();
    if len < HEADER_LEN {
        return Err(Error::NotZiptensors);
    }
    let mut header = [0; 64];
    source.read_exact_at(0, &mut header)?;
    parse_header(&header)?;

    let mut scan = Scan { outcome: ScanOutcome::NoCandidate, scanned_bytes: 0, candidates_checked: 0 };
    let mut buf = Vec::new();
    // `end` is the exclusive end of the next block; candidates `p` with `p + 64 <= end` that the
    // previous block did not cover lie in it.
    let mut end = len;
    // Bytes at the start of the next block that the previous block already read and counted:
    // none for the first block, `FOOTER_LEN - 1` after it.
    let mut overlap = 0;
    // A candidate needs `HEADER_LEN <= p` and `p + FOOTER_LEN <= end`.
    while end >= HEADER_LEN + FOOTER_LEN {
        // Every block's new bytes fit what was allowed, so this never saturates.
        let allowed = budget.max_scan_bytes.saturating_sub(scan.scanned_bytes);
        // A block holds a new candidate only if it reaches one footer length with the overlap.
        if allowed.saturating_add(overlap) < FOOTER_LEN {
            scan.outcome = ScanOutcome::ScanBytesExhausted;
            return Ok(scan);
        }
        // `end - overlap` is the start of the previous block (or the end of the file), above
        // `HEADER_LEN`, so the block gets at least one new byte.
        let new_len = (SCAN_BLOCK - overlap).min(end - overlap - HEADER_LEN).min(allowed);
        let block_len = new_len + overlap;
        let start = end - block_len;
        let size = usize::try_from(block_len)
            .map_err(|_| Error::OutOfBounds { offset: start, len: block_len, file_len: len })?;
        buf.resize(size, 0);
        source.read_exact_at(start, &mut buf)?;
        scan.scanned_bytes += new_len;

        for (at, window) in buf.windows(64).enumerate().rev() {
            if !window.ends_with(&END_MAGIC) {
                continue;
            }
            // `windows(64)` yields exactly 64 bytes, so the first branch never skips.
            let Some(bytes) = window.first_chunk::<64>() else {
                continue;
            };
            let Ok(footer) = parse_footer(bytes) else {
                // A hash that does not hold: no footer was written here, or it is torn.
                continue;
            };
            // Lossless: `usize` is at most 64 bits on every supported target.
            let valid_len = start + at as u64 + FOOTER_LEN;
            if scan.candidates_checked == budget.max_candidates {
                scan.outcome = ScanOutcome::CandidatesExhausted;
                return Ok(scan);
            }
            scan.candidates_checked += 1;
            let candidate = Candidate { valid_len, generation: footer.generation };
            if check_candidate(source, valid_len, options)? && accept(&candidate)? {
                scan.outcome = ScanOutcome::Found(candidate);
                return Ok(scan);
            }
        }
        if start == HEADER_LEN {
            break;
        }
        // The next block ends `FOOTER_LEN - 1` bytes into this one, so that the footer starting at
        // `start - 1`, the last candidate not yet covered, lies whole in it.
        end = start + FOOTER_LEN - 1;
        overlap = FOOTER_LEN - 1;
    }
    Ok(scan)
}

/// Opens the prefix `0..valid_len` of `source` as a whole container under `options` on one thread.
/// Returns `true` if it opens, `false` if it is shown not to be a generation. Stops with the error
/// when it cannot be judged (see [`stops_the_scan`]).
fn check_candidate(
    source: &Arc<dyn RandomAccess>,
    valid_len: u64,
    options: &OpenOptions,
) -> Result<bool, Error> {
    let prefix: Arc<dyn RandomAccess> = Arc::new(PrefixSource::new(Arc::clone(source), valid_len)?);
    let check = OpenOptions { threads: 1, ..*options };
    match Container::from_source(prefix, &check) {
        Ok(_) => Ok(true),
        Err(error) if stops_the_scan(&error) => {
            tracing::warn!(valid_len, %error, "a candidate generation cannot be judged");
            Err(error)
        }
        Err(error) => {
            tracing::debug!(valid_len, %error, "not a valid generation");
            Ok(false)
        }
    }
}

/// Whether an open of a candidate that failed with `error` must stop the scan. A candidate is
/// skipped only when it is shown not to be a generation: its index does not decompress, or breaks a
/// rule of the format. Every other failure needs a footer, and for a codec an index, whose hashes
/// hold, so it may be a genuine generation that these options or this version cannot read; a
/// header or I/O failure would fail every candidate alike.
fn stops_the_scan(error: &Error) -> bool {
    match error {
        Error::MalformedIndex { section, .. } => *section == "header",
        Error::Zstd { .. } => false,
        Error::Io { .. }
        | Error::NotZiptensors
        | Error::UnsupportedVersion { .. }
        | Error::UnsupportedFeature { .. }
        | Error::UnknownCodec { .. }
        | Error::BudgetExceeded { .. }
        | Error::ChunkCorrupt { .. }
        | Error::ChecksumMismatch { .. }
        | Error::OutOfBounds { .. }
        | Error::DestinationSize { .. }
        | Error::TensorNotFound { .. }
        | Error::MemberNotFound { .. }
        | Error::UnsafeMemberName { .. }
        | Error::VerifyFailed { .. }
        | Error::InputChanged { .. }
        | Error::SymlinkInput { .. }
        | Error::OutputIsInput { .. }
        | Error::CommittedNotDurable { .. }
        | Error::DamagedInPlace { .. }
        | Error::EditRefused { .. }
        | Error::TrailingBytes { .. }
        | Error::Unrecoverable { .. }
        | Error::TreeConflict { .. }
        | Error::DuplicateMember { .. } => true,
    }
}

/// The `generation` of the footer that ends at `end`, which a successful open has checked.
fn footer_at(source: &dyn RandomAccess, end: u64) -> Result<u64, Error> {
    let at = end.checked_sub(FOOTER_LEN).ok_or(Error::OutOfBounds {
        offset: 0,
        len: FOOTER_LEN,
        file_len: end,
    })?;
    let mut footer = [0; 64];
    source.read_exact_at(at, &mut footer)?;
    Ok(parse_footer(&footer)?.generation)
}

/// Decodes every member of the container in `0..valid_len` of `source` and checks its BLAKE3,
/// spec section 9.3 step 3. Returns the sum of the member lengths. `path` names the file in logs.
///
/// # Errors
/// The first error of the open or of a member's verification; [`recover`] passes over the
/// candidate on any of them but [`Error::Io`].
fn verify_generation(
    source: &Arc<dyn RandomAccess>,
    valid_len: u64,
    options: &OpenOptions,
    path: &Path,
) -> Result<u64, Error> {
    let prefix: Arc<dyn RandomAccess> = Arc::new(PrefixSource::new(Arc::clone(source), valid_len)?);
    let container = Container::from_source(prefix, options)?;
    let mut bytes = 0_u64;
    for info in container.members() {
        if let Err(error) = container.member(&info.name)?.verify() {
            tracing::warn!(
                path = %path.display(),
                valid_len,
                member = %info.name,
                %error,
                "a candidate generation does not verify; it is passed over"
            );
            return Err(error);
        }
        // Validation bounds the sum of member lengths by the decode budget, a `u64`; saturating
        // only keeps this report total from ever panicking.
        bytes = bytes.saturating_add(info.len);
    }
    tracing::info!(path = %path.display(), valid_len, bytes, "verified the found generation");
    Ok(bytes)
}

/// Opens the file at `path` once: a source to scan and verify, and unless `dry_run` a clone of
/// its handle, open for writing too, to cut it with. No later step resolves `path` again.
///
/// # Errors
/// [`Error::Io`] if the file cannot be opened (for writing unless `dry_run`), cloned or measured,
/// or is not a regular file.
fn open(path: &Path, dry_run: bool) -> Result<(FileSource, Option<File>), Error> {
    let io_error = |op, source| Error::Io { op, path: PathBuf::from(path), source };
    let file = fs::OpenOptions::new()
        .read(true)
        .write(!dry_run)
        .open(path)
        .map_err(|e| io_error("open", e))?;
    let cutter = if dry_run {
        None
    } else {
        Some(file.try_clone().map_err(|e| io_error("clone handle", e))?)
    };
    Ok((FileSource::from_file(file, path)?, cutter))
}

/// Cuts `file`, opened from `path` and scanned at `scanned_len` bytes, to `valid_len` and syncs
/// it, spec section 9.3 step 4.
///
/// # Errors
/// Returns [`Error::Io`] if its length is no longer `scanned_len`, or the truncation or the sync
/// fails. A failed sync (op `"sync"`) comes after the cut: the file is short then.
fn truncate(file: &File, path: &Path, scanned_len: u64, valid_len: u64) -> Result<(), Error> {
    let io_error = |op, source| Error::Io { op, path: PathBuf::from(path), source };
    let now = file.metadata().map_err(|e| io_error("metadata", e))?.len();
    if now != scanned_len {
        let detail = format!("the file is {now} bytes, {scanned_len} were scanned");
        return Err(io_error("recover", io::Error::other(detail)));
    }
    file.set_len(valid_len).map_err(|e| io_error("set_len", e))?;
    file.sync_all().map_err(|e| {
        tracing::error!(
            path = %path.display(),
            valid_len,
            "the file was cut but the sync failed; the cut may not survive a power loss"
        );
        io_error("sync", e)
    })?;
    tracing::info!(path = %path.display(), from = scanned_len, to = valid_len, "truncated the file");
    Ok(())
}

#[cfg(test)]
#[path = "recover_tests.rs"]
mod tests;
