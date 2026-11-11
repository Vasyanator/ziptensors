//! File: report.rs
//! Purpose: everything printed: the summaries on stdout, the per-member progress line on stderr, the
//! document of `inspect --json`, the results of `edit`, `recover` and the tree commands.
//!
//! Key structures: [`Progress`], [`Skipped`], [`InspectDocument`].
//!
//! Key functions: `write_*`, one per result.
//!
//! Implementation notes: "saved" here is `raw - stored`, the data region only, which is what a member
//! can be charged with. The file-level figure of `inspect` is `raw - file length`, so it includes index,
//! header and footer. Percentages use integer arithmetic, as a `u64` does not fit an `f64`. All sums
//! saturate: the lengths of a hostile file may overflow, and this code only displays them.

use std::io::{self, Write};
use std::path::Path;

#[cfg(any(target_os = "linux", windows))]
use ziptensors_core::CacheStats;
use ziptensors_core::{
    ClassBytes, Dtype, EditReport, EncodeReport, LayerBytes, Member, MemberInfo, MemberReport,
    Profile, RecoverReport, RestoreReport, TensorInfo,
};
use ziptensors_core::tree::{Action, FileOutcome, FileStatus, ForeignReason, SkipReason, TreeReport};

use crate::json::Json;

/// Version of the `inspect --json` document. A removed or changed key bumps it; a new key does not.
const INSPECT_SCHEMA_VERSION: u64 = 1;

/// Files `pack` left out.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Skipped {
    /// Symbolic links.
    pub(crate) symlinks: u64,
    /// `*.incomplete` files.
    pub(crate) incomplete: u64,
}

/// The running totals of a long write, logged once per member, so that a run of hours shows what it
/// has done so far.
#[derive(Debug)]
pub(crate) struct Progress {
    total: usize,
    done: usize,
    raw: u64,
    saved: u64,
}

impl Progress {
    /// Starts a report over `total` members.
    #[must_use]
    pub(crate) fn new(total: usize) -> Self {
        Self { total, done: 0, raw: 0, saved: 0 }
    }

    /// Logs one finished member, with what all members so far saved.
    pub(crate) fn record(&mut self, member: &MemberReport) {
        let LayerBytes { raw, stored, .. } = member.layers;
        let saved = raw.saturating_sub(stored);
        self.done += 1;
        self.raw = self.raw.saturating_add(raw);
        self.saved = self.saved.saturating_add(saved);
        tracing::info!(
            done = self.done,
            total = self.total,
            member = %member.name,
            raw,
            stored,
            saved,
            delta_blobs = member.delta_blobs,
            cumulative_raw = self.raw,
            cumulative_saved = self.saved,
            cumulative_saved_percent = %percent(self.saved, self.raw),
            "member added"
        );
    }
}

/// `part` as a percentage of `whole` with two decimals, `0.00%` for an empty whole. Rounds down.
fn percent(part: u64, whole: u64) -> String {
    if whole == 0 {
        return "0.00%".to_owned();
    }
    let hundredths = u128::from(part) * 10_000 / u128::from(whole);
    format!("{}.{:02}%", hundredths / 100, hundredths % 100)
}

/// `bytes` in lowercase hexadecimal.
fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        text.push(char::from(DIGITS[usize::from(byte >> 4)]));
        text.push(char::from(DIGITS[usize::from(byte & 0x0F)]));
    }
    text
}

/// The stable name of a profile, in `inspect` and its JSON. An unregistered id prints as `unknown-<id>`.
fn profile_name(profile: Profile) -> String {
    match profile {
        Profile::Opaque => "opaque".to_owned(),
        Profile::Safetensors => "safetensors".to_owned(),
        Profile::Gguf => "gguf".to_owned(),
        Profile::Onnx => "onnx".to_owned(),
        Profile::PtZip => "pt_zip".to_owned(),
        Profile::Unknown(id) => format!("unknown-{id}"),
    }
}

/// The label of a report class: `skeleton` for bytes of no dtype, the dtype otherwise.
fn class_name(class: &ClassBytes) -> String {
    class.dtype.map_or_else(|| "skeleton".to_owned(), |dtype| format!("{dtype:?}"))
}

/// Delta blobs the members added, spec section 7a. Saturates like every other sum here.
fn delta_blobs(members: &[MemberReport]) -> u64 {
    members.iter().fold(0, |sum, member| sum.saturating_add(u64::from(member.delta_blobs)))
}

/// Prints the summary of a finished container: `encode`, `pack`, `merge`.
///
/// # Errors
/// Returns the failure of a write to `out`.
pub(crate) fn write_encode(
    out: &mut impl Write,
    output: &Path,
    report: &EncodeReport,
) -> io::Result<()> {
    let total = &report.total;
    let saved = total.raw.saturating_sub(report.output_len);
    writeln!(out, "output: {}", output.display())?;
    writeln!(out, "members: {}", report.members.len())?;
    writeln!(out, "delta blobs: {}", delta_blobs(&report.members))?;
    writeln!(out, "raw bytes: {}", total.raw)?;
    writeln!(out, "file bytes: {} (index {})", report.output_len, report.index_stored_len)?;
    writeln!(out, "saved bytes: {saved} ({})", percent(saved, total.raw))?;
    writeln!(
        out,
        "verified: {} bytes in {:.1?}",
        report.verify.bytes, report.verify.elapsed
    )?;
    writeln!(
        out,
        "{:<12} {:>16} {:>16} {:>16} {:>16} {:>12} {:>16}",
        "class", "raw", "widen_saved", "dedup_saved", "entropy_saved", "overhead", "stored"
    )?;
    let rows = report.classes.iter().map(|class| (class_name(class), &class.layers));
    for (name, layers) in rows.chain([("total".to_owned(), total)]) {
        writeln!(
            out,
            "{name:<12} {:>16} {:>16} {:>16} {:>16} {:>12} {:>16}",
            layers.raw,
            layers.widen_saved,
            layers.dedup_saved,
            layers.entropy_saved,
            layers.overhead,
            layers.stored
        )?;
    }
    Ok(())
}

/// Prints what `pack` left out.
///
/// # Errors
/// Returns the failure of a write to `out`.
pub(crate) fn write_skipped(out: &mut impl Write, skipped: Skipped) -> io::Result<()> {
    writeln!(
        out,
        "skipped: {} symbolic links, {} incomplete files",
        skipped.symlinks, skipped.incomplete
    )
}

/// Prints one restored member.
///
/// # Errors
/// Returns the failure of a write to `out`.
pub(crate) fn write_restored(
    out: &mut impl Write,
    member: &str,
    path: &Path,
    report: &RestoreReport,
) -> io::Result<()> {
    writeln!(
        out,
        "restored {member} to {}: {} bytes, blake3 {}",
        path.display(),
        report.bytes,
        hex(&report.blake3)
    )
}

/// Prints the result of `restore --all`.
///
/// # Errors
/// Returns the failure of a write to `out`.
pub(crate) fn write_restored_all(
    out: &mut impl Write,
    members: usize,
    bytes: u64,
    dir: &Path,
) -> io::Result<()> {
    writeln!(out, "restored {members} members, {bytes} bytes, to {}", dir.display())
}

/// Prints the result of `verify`.
///
/// # Errors
/// Returns the failure of a write to `out`.
pub(crate) fn write_verified(out: &mut impl Write, members: usize, bytes: u64) -> io::Result<()> {
    writeln!(out, "verified {members} members, {bytes} bytes")
}

/// Prints that `container` is mounted at `mountpoint`, and how to end the mount.
///
/// # Errors
/// Returns the failure of a write to `out`.
#[cfg(any(target_os = "linux", windows))]
pub(crate) fn write_mounted(out: &mut impl Write, container: &Path, mountpoint: &Path) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    writeln!(
        out,
        "mounted {} at {}; press Ctrl-C or run `fusermount3 -u {}` to unmount",
        container.display(),
        mountpoint.display(),
        mountpoint.display()
    )?;
    #[cfg(windows)]
    writeln!(out, "mounted {} at {}; press Ctrl-C to unmount", container.display(), mountpoint.display())?;
    // The command now waits for hours: the line must not sit in a buffer.
    out.flush()
}

/// Prints that the mount at `mountpoint` ended, with what the read cache did.
///
/// # Errors
/// Returns the failure of a write to `out`.
#[cfg(any(target_os = "linux", windows))]
pub(crate) fn write_unmounted(out: &mut impl Write, mountpoint: &Path, stats: &CacheStats) -> io::Result<()> {
    writeln!(
        out,
        "unmounted {}; decoded {} bytes in {} cache misses, {} hits",
        mountpoint.display(),
        stats.decoded_bytes,
        stats.misses,
        stats.hits
    )
}

/// Prints that a second signal (on Windows a second Ctrl-C or Ctrl-Break, named by `signal`) ended the
/// mount at `mountpoint` while files were still open, and flushes, since the process exits next.
///
/// # Errors
/// Returns the failure of a write to `out`.
#[cfg(any(target_os = "linux", windows))]
pub(crate) fn write_abandoned(out: &mut impl Write, mountpoint: &Path, signal: &str) -> io::Result<()> {
    writeln!(out, "ended by a second {signal}; readers still open on {} are cut off", mountpoint.display())?;
    out.flush()
}

/// The sum of member lengths, which saturates on a hostile file.
fn raw_bytes(members: &[MemberInfo]) -> u64 {
    members.iter().fold(0, |sum, member| sum.saturating_add(member.len))
}

/// The registry name of `dtype` (`F32`, `Q4_K`), `unknown-<id>` for an id this build does not know.
fn dtype_name(dtype: Dtype) -> String {
    dtype.name().map_or_else(|| format!("unknown-{}", dtype.to_wire()), str::to_owned)
}

/// `shape` as `[a, b, c]`.
fn shape_text(shape: &[u64]) -> String {
    let dims: Vec<_> = shape.iter().map(u64::to_string).collect();
    format!("[{}]", dims.join(", "))
}

/// Prints `inspect` as text; with `tensors` (one [`Member`] per entry of `members`, in the same
/// order) a table of tensors under every member.
///
/// # Errors
/// Returns the failure of a write to `out`.
pub(crate) fn write_inspect(
    out: &mut impl Write,
    path: &Path,
    file_bytes: u64,
    members: &[MemberInfo],
    tensors: Option<&[Member]>,
) -> io::Result<()> {
    let raw = raw_bytes(members);
    let saved = raw.saturating_sub(file_bytes);
    writeln!(out, "container: {}", path.display())?;
    writeln!(out, "members: {}", members.len())?;
    writeln!(out, "raw bytes: {raw}")?;
    writeln!(out, "file bytes: {file_bytes}")?;
    writeln!(out, "saved bytes: {saved} ({})", percent(saved, raw))?;
    writeln!(out, "{:>16} {:<12} {:>8} {:<16} name", "len", "profile", "tensors", "blake3")?;
    for (at, member) in members.iter().enumerate() {
        let digest = hex(&member.blake3);
        writeln!(
            out,
            "{:>16} {:<12} {:>8} {:<16} {}",
            member.len,
            profile_name(member.profile),
            member.tensor_count,
            digest.get(..16).unwrap_or(&digest),
            member.name
        )?;
        if let Some(table) = tensors.and_then(|all| all.get(at)).map(Member::tensors) {
            for tensor in table {
                writeln!(
                    out,
                    "    {:>16} {:>16} {:<10} {:<20} {}",
                    tensor.offset,
                    tensor.len,
                    dtype_name(tensor.dtype),
                    shape_text(&tensor.shape),
                    tensor.name
                )?;
            }
        }
    }
    Ok(())
}

/// `layers` as JSON fields, `leading` first.
fn layer_fields(leading: Vec<(&'static str, Json)>, layers: &LayerBytes) -> Json {
    let mut fields = leading;
    fields.extend([
        ("raw", Json::Number(layers.raw)),
        ("widen_saved", Json::Number(layers.widen_saved)),
        ("dedup_saved", Json::Number(layers.dedup_saved)),
        ("entropy_saved", Json::Number(layers.entropy_saved)),
        ("overhead", Json::Number(layers.overhead)),
        ("stored", Json::Number(layers.stored)),
    ]);
    Json::Object(fields)
}

/// One tensor record as JSON, in the `tensors` array of `inspect --json --tensors`.
fn tensor_json(tensor: &TensorInfo) -> Json {
    Json::Object(vec![
        ("name", Json::String(tensor.name.clone())),
        ("offset", Json::Number(tensor.offset)),
        ("len", Json::Number(tensor.len)),
        ("dtype", Json::String(dtype_name(tensor.dtype))),
        ("shape", Json::Array(tensor.shape.iter().copied().map(Json::Number).collect())),
    ])
}

/// What `inspect --json` prints.
#[derive(Debug)]
pub(crate) struct InspectDocument<'a> {
    /// The container as given.
    pub(crate) path: &'a Path,
    /// Its length.
    pub(crate) file_bytes: u64,
    /// Its members, in file order.
    pub(crate) members: &'a [MemberInfo],
    /// `core::layer_report` over all members.
    pub(crate) layers: &'a (LayerBytes, Vec<ClassBytes>),
    /// With `--tensors`: one [`Member`] per entry of `members`, in the same order.
    pub(crate) tensors: Option<&'a [Member]>,
}

/// The `inspect --json` document. The schema is documented in `docs/cli.md`, section `inspect`.
fn inspect_json(document: &InspectDocument<'_>) -> Json {
    let InspectDocument { path, file_bytes, members, layers: (layers, classes), tensors } = document;
    let raw = raw_bytes(members);
    let rows = members.iter().enumerate().map(|(at, member)| {
        let mut fields = vec![
            ("name", Json::String(member.name.clone())),
            ("len", Json::Number(member.len)),
            ("blake3", Json::String(hex(&member.blake3))),
            ("profile", Json::String(profile_name(member.profile))),
            ("tensor_count", Json::Number(u64::from(member.tensor_count))),
        ];
        if let Some(all) = tensors {
            let table = all.get(at).map_or(&[][..], Member::tensors);
            fields.push(("tensors", Json::Array(table.iter().map(tensor_json).collect())));
        }
        Json::Object(fields)
    });
    let classes = classes
        .iter()
        .map(|class| layer_fields(vec![("class", Json::String(class_name(class)))], &class.layers));
    Json::Object(vec![
        ("schema_version", Json::Number(INSPECT_SCHEMA_VERSION)),
        ("path", Json::String(path.display().to_string())),
        ("file_bytes", Json::Number(*file_bytes)),
        ("raw_bytes", Json::Number(raw)),
        ("saved_bytes", Json::Number(raw.saturating_sub(*file_bytes))),
        ("members", Json::Array(rows.collect())),
        ("layers", layer_fields(Vec::new(), layers)),
        ("classes", Json::Array(classes.collect())),
    ])
}

/// Prints `inspect --json`: one document on one line.
///
/// # Errors
/// Returns the failure of a write to `out`.
pub(crate) fn write_inspect_json(out: &mut impl Write, document: &InspectDocument<'_>) -> io::Result<()> {
    writeln!(out, "{}", inspect_json(document))
}

/// Prints the result of `edit` or `compact` on `path` with `strategy`, its name as the user chose
/// it, then the summary of the written container.
///
/// # Errors
/// Returns the failure of a write to `out`.
pub(crate) fn write_edit(
    out: &mut impl Write,
    path: &Path,
    strategy: &str,
    report: &EditReport,
) -> io::Result<()> {
    writeln!(out, "edited: {} ({strategy})", path.display())?;
    writeln!(
        out,
        "changes: {} kept, {} added, {} replaced, {} removed",
        report.kept, report.added, report.replaced, report.removed
    )?;
    writeln!(out, "generation: {}", report.generation)?;
    writeln!(out, "dead bytes before: {}", report.dead_bytes_before)?;
    write_encode(out, path, &report.encode)
}

/// Prints the result of `recover` on `path`, with both lengths: `valid bytes` below `file bytes`
/// is the cut, done or, on a dry run, planned. `passed over` is the count of newer candidate
/// generations rejected on the way, [`RecoverReport::passed_over`].
///
/// # Errors
/// Returns the failure of a write to `out`.
pub(crate) fn write_recovered(
    out: &mut impl Write,
    path: &Path,
    dry_run: bool,
    report: &RecoverReport,
) -> io::Result<()> {
    let verdict = match (report.valid_len < report.file_len, report.truncated) {
        (false, _) => "intact, nothing to do",
        (true, true) => "recovered",
        (true, false) if dry_run => "would recover (dry run, nothing written)",
        // A cut that is neither done nor a dry run does not come from the core; said as it is.
        (true, false) => "not cut",
    };
    writeln!(out, "{verdict}: {}", path.display())?;
    writeln!(out, "generation: {}", report.generation)?;
    writeln!(out, "file bytes: {}", report.file_len)?;
    writeln!(out, "valid bytes: {}", report.valid_len)?;
    match report.verified_bytes {
        Some(bytes) => writeln!(out, "verified: {bytes} bytes")?,
        None => writeln!(out, "verified: no")?,
    }
    writeln!(out, "scanned bytes: {}", report.scanned_bytes)?;
    writeln!(out, "passed over: {}", report.passed_over)
}

/// The warning line of a `recover` that cuts (or on a dry run would cut) the file, for stderr, or
/// `None` when nothing is cut. The line names the generation kept and the bytes after it that are
/// lost; when [`RecoverReport::passed_over`] is above 0 it also says how many newer candidate
/// generations failed the structural checks or verification and were passed over, since their
/// data, if any of them was genuine, is in those bytes.
#[must_use]
pub(crate) fn recover_warning(report: &RecoverReport) -> Option<String> {
    if report.valid_len >= report.file_len {
        return None;
    }
    let lost = report.file_len.saturating_sub(report.valid_len);
    let tense = if report.truncated { "were" } else { "would be" };
    let passed_over = match report.passed_over {
        0 => String::new(),
        1 => "; 1 newer candidate generation failed the checks and was passed over".to_owned(),
        count => format!("; {count} newer candidate generations failed the checks and were passed over"),
    };
    Some(format!(
        "warning: recovered generation {}; the {lost} bytes after it ({} -> {} bytes) {tense} \
         discarded, with any newer data they held{passed_over}",
        report.generation, report.file_len, report.valid_len
    ))
}

/// Why a path was skipped, in words.
fn skip_text(reason: SkipReason) -> String {
    match reason {
        SkipReason::Symlink => "symbolic link, not followed".to_owned(),
        SkipReason::SymlinkToDirectory => "link to a directory, never entered".to_owned(),
        SkipReason::DanglingSymlink => "link to nothing".to_owned(),
        SkipReason::Special => "not a regular file".to_owned(),
        SkipReason::StaleTemp => "temporary file of an interrupted write, never removed".to_owned(),
        SkipReason::Incomplete => "partial download".to_owned(),
        SkipReason::NonUtf8 => "name is not UTF-8".to_owned(),
        // `SkipReason` is `#[non_exhaustive]`: a reason a newer core adds prints in its debug form.
        other => format!("{other:?}"),
    }
}

/// Why a container is not a per-file container, in words.
fn foreign_text(reason: &ForeignReason) -> String {
    match reason {
        ForeignReason::MemberCount(count) => format!("{count} members, not one"),
        ForeignReason::NotOneSafeName(name) => format!("member {name:?} is not one file name"),
        // `ForeignReason` is `#[non_exhaustive]`: a reason a newer core adds prints in its debug form.
        other => format!("{other:?}"),
    }
}

/// What a dry run would do, in words.
fn action_text(action: Action) -> String {
    match action {
        Action::Compress => "compress".to_owned(),
        Action::CheckPair => "check the pair, then remove the redundant copy".to_owned(),
        Action::Restore => "restore".to_owned(),
        // `Action` is `#[non_exhaustive]`: an action a newer core adds prints in its debug form.
        other => format!("{other:?}"),
    }
}

/// The label of `status` and, for the statuses that carry one, its detail.
fn status_text(status: &FileStatus) -> (&'static str, Option<String>) {
    match status {
        FileStatus::Compressed => ("compressed", None),
        FileStatus::Recompressed => ("recompressed", None),
        FileStatus::PairResolved => ("pair resolved", None),
        FileStatus::PairVerified => ("pair verified", None),
        FileStatus::AlreadyCompressed => ("already compressed", None),
        FileStatus::Restored => ("restored", None),
        FileStatus::NotWeight => ("not a weight file", None),
        FileStatus::Foreign(reason) => ("left alone", Some(foreign_text(reason))),
        FileStatus::Skipped(reason) => ("skipped", Some(skip_text(*reason))),
        FileStatus::Planned(action) => ("would", Some(action_text(*action))),
        FileStatus::Failed(error) => ("FAILED", Some(error.to_string())),
        // `FileStatus` is `#[non_exhaustive]`: a status a newer core adds prints in its debug form.
        other => ("other", Some(format!("{other:?}"))),
    }
}

/// Which tree command a [`TreeReport`] came from. The core's report does not say, and the lines
/// of one outcome read differently in each direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TreeDirection {
    /// `compress`: the uncompressed file is the source, its container the result.
    Compress,
    /// `decompress`: the container is the source, the file restored from it the result.
    Decompress,
}

/// One line of a tree run: `<status>: <source> [-> <result>] [(<source> -> <result> bytes)]
/// [, removed <path>] [: <detail>]`. Both arrows point from source to result, in the sense of
/// `direction`: in compress from the uncompressed file to its container, in decompress from the
/// container to the restored file, whichever of the two the outcome's `path` is.
fn outcome_line(outcome: &FileOutcome, direction: TreeDirection) -> String {
    let (label, detail) = status_text(&outcome.status);
    // In compress `path` is the uncompressed file, except for a container found alone, whose
    // counterpart is the name it restores to; in decompress `path` is always the container.
    let counterpart_is_source = direction == TreeDirection::Compress
        && matches!(outcome.status, FileStatus::AlreadyCompressed);
    let (source, result) = match (&outcome.counterpart, counterpart_is_source) {
        (Some(counterpart), true) => (counterpart, Some(&outcome.path)),
        (counterpart, _) => (&outcome.path, counterpart.as_ref()),
    };
    let mut parts = vec![format!("{label}: {}", source.display())];
    if let Some(result) = result {
        parts.push(format!(" -> {}", result.display()));
    }
    if let (Some(original), Some(container)) = (outcome.original_len, outcome.container_len) {
        let (from, to) = match direction {
            TreeDirection::Compress => (original, container),
            TreeDirection::Decompress => (container, original),
        };
        parts.push(format!(" ({from} -> {to} bytes)"));
    }
    if let Some(removed) = &outcome.removed {
        parts.push(format!(", removed {}", removed.display()));
    }
    if let Some(detail) = detail {
        parts.push(format!(": {detail}"));
    }
    parts.concat()
}

/// Prints a tree run (`compress`, `decompress`): one line per path, then the totals and the byte
/// line of `direction`. Warnings are not here: they go to stderr.
///
/// The totals line counts the statuses `direction` can produce: compress has no `restored`;
/// decompress, which looks at containers alone, has no `compressed`, `recompressed`, `already
/// compressed` or `not weights`, and does not count the other files of the tree.
///
/// The byte line sums the files whose two copies the run proved equal (`TreeTotals`): compress
/// prints `bytes: <original> original, <container> in containers, <saved> saved (<percent>)`,
/// decompress `bytes: <container> in containers -> <original> uncompressed`.
///
/// # Errors
/// Returns the failure of a write to `out`.
pub(crate) fn write_tree(
    out: &mut impl Write,
    report: &TreeReport,
    direction: TreeDirection,
) -> io::Result<()> {
    for outcome in &report.files {
        writeln!(out, "{}", outcome_line(outcome, direction))?;
    }
    let totals = report.totals();
    let dry = if report.dry_run { " (dry run, nothing written)" } else { "" };
    writeln!(out, "root: {}{dry}", report.root.display())?;
    // Each direction lists only the statuses it can produce: decompress looks at containers alone,
    // so "not weights" (a compress classification) would always read 0 there, and so on.
    match direction {
        TreeDirection::Compress => writeln!(
            out,
            "totals: {} compressed, {} recompressed, {} pairs resolved, {} pairs verified, \
             {} already compressed, {} not weights, {} left alone, {} skipped, {} planned, {} failed",
            totals.compressed,
            totals.recompressed,
            totals.pair_resolved,
            totals.pair_verified,
            totals.already_compressed,
            totals.not_weight,
            totals.foreign,
            totals.skipped,
            totals.planned,
            totals.failed
        )?,
        TreeDirection::Decompress => writeln!(
            out,
            "totals: {} restored, {} pairs resolved, {} pairs verified, {} left alone, {} skipped, \
             {} planned, {} failed",
            totals.restored,
            totals.pair_resolved,
            totals.pair_verified,
            totals.foreign,
            totals.skipped,
            totals.planned,
            totals.failed
        )?,
    }
    match direction {
        TreeDirection::Compress => {
            let saved = totals.original_bytes.saturating_sub(totals.container_bytes);
            writeln!(
                out,
                "bytes: {} original, {} in containers, {saved} saved ({})",
                totals.original_bytes,
                totals.container_bytes,
                percent(saved, totals.original_bytes)
            )
        }
        TreeDirection::Decompress => writeln!(
            out,
            "bytes: {} in containers -> {} uncompressed",
            totals.container_bytes, totals.original_bytes
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{TreeDirection, hex, percent, profile_name, raw_bytes, recover_warning, write_tree};
    use std::path::PathBuf;
    use ziptensors_core::tree::{FileOutcome, FileStatus, TreeReport};
    use ziptensors_core::{MemberInfo, Profile, RecoverReport};

    /// An outcome about `path` with `counterpart` and both lengths.
    fn outcome(
        status: FileStatus,
        path: &str,
        counterpart: &str,
        lengths: (Option<u64>, Option<u64>),
    ) -> FileOutcome {
        FileOutcome {
            path: PathBuf::from(path),
            counterpart: Some(PathBuf::from(counterpart)),
            status,
            original_len: lengths.0,
            container_len: lengths.1,
            removed: None,
        }
    }

    /// The printed lines of a tree run of `files` in `direction`.
    fn tree_text(files: Vec<FileOutcome>, direction: TreeDirection) -> String {
        let report = TreeReport { root: PathBuf::from("r"), dry_run: false, files, warnings: Vec::new() };
        let mut out = Vec::new();
        write_tree(&mut out, &report, direction).expect("writes to memory");
        String::from_utf8(out).expect("UTF-8")
    }

    #[test]
    fn tree_lines_point_from_source_to_result_in_both_arrows() {
        let compress = tree_text(
            vec![
                outcome(FileStatus::Compressed, "m.safetensors", "m.ziptensors", (Some(100), Some(40))),
                outcome(FileStatus::AlreadyCompressed, "n.ziptensors", "n.safetensors", (None, Some(7))),
            ],
            TreeDirection::Compress,
        );
        assert!(compress.contains("compressed: m.safetensors -> m.ziptensors (100 -> 40 bytes)\n"), "{compress}");
        assert!(compress.contains("already compressed: n.safetensors -> n.ziptensors\n"), "{compress}");
        assert!(compress.contains("bytes: 100 original, 40 in containers, 60 saved (60.00%)"), "{compress}");

        let decompress = tree_text(
            vec![
                outcome(FileStatus::Restored, "m.ziptensors", "m.safetensors", (Some(100), Some(40))),
                outcome(FileStatus::PairResolved, "p.ziptensors", "p.safetensors", (Some(10), Some(5))),
            ],
            TreeDirection::Decompress,
        );
        assert!(decompress.contains("restored: m.ziptensors -> m.safetensors (40 -> 100 bytes)\n"), "{decompress}");
        assert!(decompress.contains("pair resolved: p.ziptensors -> p.safetensors (5 -> 10 bytes)\n"), "{decompress}");
        assert!(decompress.contains("bytes: 45 in containers -> 110 uncompressed\n"), "{decompress}");
        assert!(!decompress.contains("saved"), "{decompress}");
        // Each totals line lists the statuses of its direction only.
        assert!(decompress.contains("totals: 1 restored, 1 pairs resolved, 0 pairs verified, "), "{decompress}");
        assert!(!decompress.contains("not weights") && !decompress.contains("compressed,"), "{decompress}");
        assert!(compress.contains("totals: 1 compressed, ") && !compress.contains("restored"), "{compress}");
    }

    #[test]
    fn percent_keeps_two_decimals_and_survives_extremes() {
        assert_eq!(percent(0, 0), "0.00%");
        assert_eq!(percent(1, 3), "33.33%");
        assert_eq!(percent(5, 100), "5.00%");
        assert_eq!(percent(u64::MAX, u64::MAX), "100.00%");
    }

    #[test]
    fn hex_is_lowercase_and_pads() {
        assert_eq!(hex(&[0x00, 0x0A, 0xFF]), "000aff");
    }

    #[test]
    fn profile_names_are_stable() {
        let names = [
            (Profile::Opaque, "opaque"),
            (Profile::Safetensors, "safetensors"),
            (Profile::Gguf, "gguf"),
            (Profile::Onnx, "onnx"),
            (Profile::PtZip, "pt_zip"),
            (Profile::Unknown(200), "unknown-200"),
        ];
        for (profile, name) in names {
            assert_eq!(profile_name(profile), name);
        }
    }

    #[test]
    fn raw_bytes_saturates_on_lengths_a_hostile_file_may_state() {
        let member = |len| MemberInfo {
            name: String::new(),
            len,
            blake3: [0; 32],
            profile: Profile::Opaque,
            tensor_count: 0,
        };
        assert_eq!(raw_bytes(&[member(u64::MAX), member(1)]), u64::MAX);
    }

    #[test]
    fn the_recover_warning_names_passed_over_generations_only_when_there_are_some() {
        let report = |valid_len, passed_over| RecoverReport {
            file_len: 1000,
            valid_len,
            generation: 3,
            truncated: true,
            verified_bytes: Some(10),
            scanned_bytes: 900,
            passed_over,
        };
        assert_eq!(recover_warning(&report(1000, 0)), None);
        let none = recover_warning(&report(600, 0)).expect("a cut warns");
        assert!(none.ends_with("with any newer data they held"), "{none}");
        assert!(!none.contains("passed over"), "{none}");
        let one = recover_warning(&report(600, 1)).expect("a cut warns");
        let singular = "; 1 newer candidate generation failed the checks and was passed over";
        assert!(one.ends_with(singular), "{one}");
        let two = recover_warning(&report(600, 2)).expect("a cut warns");
        let plural = "; 2 newer candidate generations failed the checks and were passed over";
        assert!(two.ends_with(plural), "{two}");
    }
}
