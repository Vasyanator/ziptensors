//! File: report.rs
//! Purpose: everything printed: the summaries on stdout, the per-member progress line on stderr, the
//! document of `inspect --json`.
//!
//! Key structures: [`Progress`], [`Skipped`].
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
    ClassBytes, EncodeReport, LayerBytes, MemberInfo, MemberReport, Profile, RestoreReport,
};

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

/// Prints `inspect` as text.
///
/// # Errors
/// Returns the failure of a write to `out`.
pub(crate) fn write_inspect(
    out: &mut impl Write,
    path: &Path,
    file_bytes: u64,
    members: &[MemberInfo],
) -> io::Result<()> {
    let raw = raw_bytes(members);
    let saved = raw.saturating_sub(file_bytes);
    writeln!(out, "container: {}", path.display())?;
    writeln!(out, "members: {}", members.len())?;
    writeln!(out, "raw bytes: {raw}")?;
    writeln!(out, "file bytes: {file_bytes}")?;
    writeln!(out, "saved bytes: {saved} ({})", percent(saved, raw))?;
    writeln!(out, "{:>16} {:<12} {:>8} {:<16} name", "len", "profile", "tensors", "blake3")?;
    for member in members {
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

/// The `inspect --json` document. The schema is documented in `MODULE_README.md`.
fn inspect_json(
    path: &Path,
    file_bytes: u64,
    members: &[MemberInfo],
    (layers, classes): &(LayerBytes, Vec<ClassBytes>),
) -> Json {
    let raw = raw_bytes(members);
    let rows = members.iter().map(|member| {
        Json::Object(vec![
            ("name", Json::String(member.name.clone())),
            ("len", Json::Number(member.len)),
            ("blake3", Json::String(hex(&member.blake3))),
            ("profile", Json::String(profile_name(member.profile))),
            ("tensor_count", Json::Number(u64::from(member.tensor_count))),
        ])
    });
    let classes = classes
        .iter()
        .map(|class| layer_fields(vec![("class", Json::String(class_name(class)))], &class.layers));
    Json::Object(vec![
        ("schema_version", Json::Number(INSPECT_SCHEMA_VERSION)),
        ("path", Json::String(path.display().to_string())),
        ("file_bytes", Json::Number(file_bytes)),
        ("raw_bytes", Json::Number(raw)),
        ("saved_bytes", Json::Number(raw.saturating_sub(file_bytes))),
        ("members", Json::Array(rows.collect())),
        ("layers", layer_fields(Vec::new(), layers)),
        ("classes", Json::Array(classes.collect())),
    ])
}

/// Prints `inspect --json`: one document on one line.
///
/// # Errors
/// Returns the failure of a write to `out`.
pub(crate) fn write_inspect_json(
    out: &mut impl Write,
    path: &Path,
    file_bytes: u64,
    members: &[MemberInfo],
    layers: &(LayerBytes, Vec<ClassBytes>),
) -> io::Result<()> {
    writeln!(out, "{}", inspect_json(path, file_bytes, members, layers))
}

#[cfg(test)]
mod tests {
    use super::{hex, percent, profile_name, raw_bytes};
    use ziptensors_core::{MemberInfo, Profile};

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
}
