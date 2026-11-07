//! File: `profile/pt_zip.rs`
//! Purpose: finds the storages of a `PyTorch` ZIP archive from ZIP structures alone.
//!
//! Main responsibilities:
//! - reads the end records and the central directory;
//! - checks every entry against its local header, and that entries do not overlap;
//! - refuses an archive whose storages or `data.pkl` are compressed;
//! - reports `<archive>/data/<N>` entries, each with a width from [`width_trial`].
//!
//! Key functions: [`parse`].
//!
//! Implementation notes: no pickle is parsed, so a tensor row is one storage: named after its
//! entry, flat, and typed by width. `F16`, `F32`, `F64` or `U8` stand for "2, 4, 8 or 1 byte
//! elements", which is all the writer needs for a plane split; integers get a float tag.
//! Real archives differ from a plain reading of ZIP in two ways. Their writer always emits a ZIP64
//! end record and locator even for small files, so those are accepted when they repeat the classic
//! record exactly; what makes a file opaque is a ZIP64 marker value in a classic field, or records
//! that disagree. Local headers also set the data descriptor flag and hold zero sizes, so local
//! sizes are compared only when that flag is clear. `TorchScript` archives deflate their `code/`
//! entries and store their data, so another compressed entry stays in the skeleton with the rest of
//! the structure. An opaque answer applies to the whole file, and archives over 4 GiB, which need
//! ZIP64 values, are opaque in this version.

use super::plan::MIN_RANGE_BYTES;
use super::reader::Reader;
use super::width_trial::width_trial;
use super::{ParseResult, Stop};
use crate::types::{Dtype, Profile, RangePlan, TensorInfo, TypedRange};

/// First bytes of a ZIP archive with at least one entry.
pub(super) const LOCAL_MAGIC: &[u8] = b"PK\x03\x04";

/// Signature of a local file header.
const LOCAL_SIGNATURE: u32 = 0x0403_4B50;
/// Signature of a central directory record.
const CENTRAL_SIGNATURE: u32 = 0x0201_4B50;
/// Signature of the end of central directory record.
const END_SIGNATURE: [u8; 4] = *b"PK\x05\x06";
/// Signature of the ZIP64 end of central directory record.
const ZIP64_END_SIGNATURE: u32 = 0x0606_4B50;
/// Signature of the ZIP64 end of central directory locator.
const ZIP64_LOCATOR_SIGNATURE: u32 = 0x0706_4B50;

/// Size of the end record without its comment.
const END_BYTES: u64 = 22;
/// Longest comment of the end record.
const MAX_COMMENT_BYTES: u64 = 65_535;
/// Size of the ZIP64 locator.
const LOCATOR_BYTES: u64 = 20;
/// Value of the ZIP64 end record's size field: the fixed part after that field, no extension.
const ZIP64_END_REMAINDER: u64 = 44;
/// Bytes of the ZIP64 end record before the remainder counted by its size field.
const ZIP64_END_LEAD: u64 = 12;
/// Size of a local header before its name.
const LOCAL_HEADER_BYTES: u64 = 30;

/// A classic 16-bit field holding this defers to ZIP64.
const MARKER_16: u16 = u16::MAX;
/// A classic 32-bit field holding this defers to ZIP64.
const MARKER_32: u32 = u32::MAX;

/// General purpose flags that hide or transform content: encrypted, strong encryption, masked
/// header values.
const FLAGS_ENCRYPTED: u16 = 1 | 1 << 6 | 1 << 13;
/// General purpose flag: sizes live in a data descriptor after the data.
const FLAG_DATA_DESCRIPTOR: u16 = 1 << 3;

/// Bytes sampled from one entry for [`width_trial`].
const SAMPLE_BYTES: u64 = 16 * 1024;
/// Total bytes sampled from one archive. Entries past it are typed `U8`.
const MAX_SAMPLED_BYTES: u64 = 64 * 1024 * 1024;
/// Sample offsets are multiples of this from the entry start, so elements of any width line up.
const SAMPLE_ALIGNMENT: u64 = 8;

/// Where the central directory is.
struct Directory {
    offset: u64,
    end: u64,
    entries: u16,
}

/// One entry of the central directory.
struct Entry {
    name: String,
    flags: u16,
    method: u16,
    /// Bytes the entry takes in the file. Equals `raw_len` for a stored entry.
    stored_len: u64,
    /// Bytes of the entry's content.
    raw_len: u64,
    local_offset: u64,
    /// Start of the data, known after the local header was read.
    data_offset: u64,
}

/// Parses the archive. Offsets in the plan are absolute.
pub(super) fn parse(reader: &mut Reader<'_>) -> ParseResult<RangePlan> {
    let directory = find_directory(reader)?;
    let mut entries = read_central_directory(reader, &directory)?;
    for entry in &mut entries {
        read_local_header(reader, entry)?;
    }
    entries.sort_unstable_by_key(|entry| entry.local_offset);
    check_layout(&entries, directory.offset)?;
    let prefix = archive_prefix(&entries)?.to_owned();
    check_content_is_stored(&entries, &prefix)?;

    let mut plan = RangePlan { profile: Profile::PtZip, ranges: Vec::new(), tensors: Vec::new() };
    let mut budget = MAX_SAMPLED_BYTES;
    for entry in entries.into_iter().filter(|entry| is_storage(&entry.name, &prefix)) {
        let width = storage_width(reader, &entry, &mut budget)?;
        let dtype = match width {
            2 => Dtype::F16,
            4 => Dtype::F32,
            8 => Dtype::F64,
            _ => Dtype::U8,
        };
        let (offset, len) = (entry.data_offset, entry.stored_len);
        plan.ranges.push(TypedRange { offset, len, hint: dtype });
        let shape = vec![len / width];
        plan.tensors.push(TensorInfo { name: entry.name, offset, len, shape, dtype });
    }
    Ok(plan)
}

/// Finds the end record at the very end of the file and validates what it says.
fn find_directory(reader: &mut Reader<'_>) -> ParseResult<Directory> {
    let tail_len = reader.len().min(END_BYTES + MAX_COMMENT_BYTES);
    let tail_start = reader.len() - tail_len;
    reader.seek(tail_start)?;
    let tail = reader.take(tail_len, tail_len)?;
    let found = tail
        .windows(END_SIGNATURE.len())
        .rposition(|window| window == END_SIGNATURE)
        .ok_or(Stop::Malformed("ZIP has no end record"))?;
    let end_pos = tail_start
        .checked_add(u64::try_from(found).map_err(|_| Stop::Malformed("ZIP tail too long"))?)
        .ok_or(Stop::Malformed("ZIP end record offset overflows"))?;

    reader.seek(end_pos)?;
    reader.skip(4)?;
    let (disk, directory_disk) = (reader.u16()?, reader.u16()?);
    let (disk_entries, entries) = (reader.u16()?, reader.u16()?);
    let (size, offset) = (reader.u32()?, reader.u32()?);
    let comment_len = u64::from(reader.u16()?);
    if reader.pos().checked_add(comment_len) != Some(reader.len()) {
        return Err(Stop::Malformed("ZIP end record does not end the file"));
    }
    if disk != 0 || directory_disk != 0 || disk_entries != entries {
        return Err(Stop::Malformed("ZIP spans several disks"));
    }
    if entries == MARKER_16 || size == MARKER_32 || offset == MARKER_32 {
        return Err(Stop::Malformed("ZIP defers to ZIP64 values"));
    }
    let end = directory_end(reader, end_pos, entries, size, offset)?;
    let offset = u64::from(offset);
    if offset.checked_add(u64::from(size)) != Some(end) {
        return Err(Stop::Malformed("ZIP central directory does not meet its end record"));
    }
    Ok(Directory { offset, end, entries })
}

/// Returns where the central directory must end: at the ZIP64 records if a locator precedes the end
/// record at `end_pos`, or else at the end record. ZIP64 records must repeat the classic values.
fn directory_end(
    reader: &mut Reader<'_>,
    end_pos: u64,
    entries: u16,
    size: u32,
    offset: u32,
) -> ParseResult<u64> {
    let Some(locator_pos) = end_pos.checked_sub(LOCATOR_BYTES) else {
        return Ok(end_pos);
    };
    reader.seek(locator_pos)?;
    if reader.u32()? != ZIP64_LOCATOR_SIGNATURE {
        return Ok(end_pos);
    }
    let (record_disk, record_pos, disks) = (reader.u32()?, reader.u64()?, reader.u32()?);
    if record_disk != 0 || disks != 1 {
        return Err(Stop::Malformed("ZIP64 locator names several disks"));
    }
    reader.seek(record_pos)?;
    if reader.u32()? != ZIP64_END_SIGNATURE || reader.u64()? != ZIP64_END_REMAINDER {
        return Err(Stop::Malformed("ZIP64 end record is not the plain form"));
    }
    if record_pos.checked_add(ZIP64_END_LEAD + ZIP64_END_REMAINDER) != Some(locator_pos) {
        return Err(Stop::Malformed("ZIP64 end record does not meet its locator"));
    }
    reader.skip(4)?;
    let (disk, directory_disk) = (reader.u32()?, reader.u32()?);
    let (disk_entries, total) = (reader.u64()?, reader.u64()?);
    let (zip64_size, zip64_offset) = (reader.u64()?, reader.u64()?);
    let count = u64::from(entries);
    if disk != 0
        || directory_disk != 0
        || disk_entries != count
        || total != count
        || zip64_size != u64::from(size)
        || zip64_offset != u64::from(offset)
    {
        return Err(Stop::Malformed("ZIP64 end record disagrees with the classic one"));
    }
    Ok(record_pos)
}

/// Reads every central record. Their count is a `u16`, so the list stays small whatever the file
/// declares. Encryption, ZIP64 deferral and other disks stop the parse; compression is judged
/// later, by [`check_content_is_stored`], once it is known which entries matter.
fn read_central_directory(
    reader: &mut Reader<'_>,
    directory: &Directory,
) -> ParseResult<Vec<Entry>> {
    reader.seek(directory.offset)?;
    let mut entries = Vec::new();
    for _ in 0..directory.entries {
        if reader.u32()? != CENTRAL_SIGNATURE {
            return Err(Stop::Malformed("ZIP central record has a bad signature"));
        }
        reader.skip(4)?;
        let (flags, method) = (reader.u16()?, reader.u16()?);
        reader.skip(8)?;
        let (compressed, size) = (reader.u32()?, reader.u32()?);
        let (name_len, extra_len, comment_len) = (reader.u16()?, reader.u16()?, reader.u16()?);
        let disk = reader.u16()?;
        reader.skip(6)?;
        let local_offset = reader.u32()?;
        if method == 0 && compressed != size {
            return Err(Stop::Malformed("stored ZIP entry states two sizes"));
        }
        if flags & FLAGS_ENCRYPTED != 0 {
            return Err(Stop::Malformed("ZIP entry is encrypted"));
        }
        if compressed == MARKER_32
            || size == MARKER_32
            || local_offset == MARKER_32
            || disk == MARKER_16
        {
            return Err(Stop::Malformed("ZIP entry defers to ZIP64 values"));
        }
        if disk != 0 {
            return Err(Stop::Malformed("ZIP entry is on another disk"));
        }
        // A u16 length cannot exceed the cap, so this read is bounded by construction.
        let name = reader.take(u64::from(name_len), u64::from(u16::MAX))?;
        let name =
            String::from_utf8(name).map_err(|_| Stop::Malformed("ZIP name is not UTF-8"))?;
        reader.skip(u64::from(extra_len) + u64::from(comment_len))?;
        entries.push(Entry {
            name,
            flags,
            method,
            stored_len: u64::from(compressed),
            raw_len: u64::from(size),
            local_offset: u64::from(local_offset),
            data_offset: 0,
        });
    }
    if reader.pos() != directory.end {
        return Err(Stop::Malformed("ZIP central directory has a wrong size"));
    }
    Ok(entries)
}

/// Checks the local header of `entry` against its central record and fills in `data_offset`.
fn read_local_header(reader: &mut Reader<'_>, entry: &mut Entry) -> ParseResult<()> {
    reader.seek(entry.local_offset)?;
    if reader.u32()? != LOCAL_SIGNATURE {
        return Err(Stop::Malformed("ZIP local header has a bad signature"));
    }
    reader.skip(2)?;
    let (flags, method) = (reader.u16()?, reader.u16()?);
    reader.skip(8)?;
    let (compressed, size) = (reader.u32()?, reader.u32()?);
    let (name_len, extra_len) = (reader.u16()?, reader.u16()?);
    let descriptor = flags & FLAG_DATA_DESCRIPTOR;
    if method != entry.method || descriptor != entry.flags & FLAG_DATA_DESCRIPTOR {
        return Err(Stop::Malformed("ZIP local header disagrees with the central record"));
    }
    if descriptor == 0
        && (u64::from(compressed) != entry.stored_len || u64::from(size) != entry.raw_len)
    {
        return Err(Stop::Malformed("ZIP local sizes disagree with the central record"));
    }
    let name = reader.take(u64::from(name_len), u64::from(u16::MAX))?;
    if name != entry.name.as_bytes() {
        return Err(Stop::Malformed("ZIP local name disagrees with the central record"));
    }
    entry.data_offset = entry
        .local_offset
        .checked_add(LOCAL_HEADER_BYTES + u64::from(name_len) + u64::from(extra_len))
        .ok_or(Stop::Malformed("ZIP data offset overflows"))?;
    Ok(())
}

/// Checks that entries, sorted by local offset, are disjoint and all before the directory.
fn check_layout(entries: &[Entry], directory_offset: u64) -> ParseResult<()> {
    let mut floor = 0;
    for entry in entries {
        if entry.local_offset < floor {
            return Err(Stop::Malformed("ZIP entries overlap"));
        }
        floor = entry
            .data_offset
            .checked_add(entry.stored_len)
            .ok_or(Stop::Malformed("ZIP entry end overflows"))?;
    }
    if floor > directory_offset {
        return Err(Stop::Malformed("ZIP entry runs into the central directory"));
    }
    Ok(())
}

/// Returns the archive's directory name, from its single `<name>/data.pkl` entry. An archive
/// without one is some other ZIP.
fn archive_prefix(entries: &[Entry]) -> ParseResult<&str> {
    let mut prefixes = entries
        .iter()
        .filter_map(|entry| entry.name.strip_suffix("/data.pkl"))
        .filter(|prefix| !prefix.is_empty() && !prefix.contains('/'));
    let prefix = prefixes.next().ok_or(Stop::Malformed("ZIP has no data.pkl entry"))?;
    if prefixes.next().is_some() {
        return Err(Stop::Malformed("ZIP has several data.pkl entries"));
    }
    Ok(prefix)
}

/// Stops if a storage or the `data.pkl` of the archive is compressed. Other entries may be.
fn check_content_is_stored(entries: &[Entry], prefix: &str) -> ParseResult<()> {
    let is_content = |entry: &&Entry| {
        is_storage(&entry.name, prefix)
            || entry.name.strip_prefix(prefix) == Some("/data.pkl")
    };
    if entries.iter().filter(is_content).any(|entry| entry.method != 0) {
        return Err(Stop::Malformed("ZIP storage or data.pkl is compressed"));
    }
    Ok(())
}

/// Whether `name` is `<prefix>/data/<digits>`.
fn is_storage(name: &str, prefix: &str) -> bool {
    name.strip_prefix(prefix)
        .and_then(|rest| rest.strip_prefix("/data/"))
        .is_some_and(|id| !id.is_empty() && id.bytes().all(|b| b.is_ascii_digit()))
}

/// Picks the element width of a storage from a sample in its middle. A storage too short to get a
/// range, and any past the sampling budget, is 1. So is one the answer would not divide.
fn storage_width(reader: &mut Reader<'_>, entry: &Entry, budget: &mut u64) -> ParseResult<u64> {
    let len = entry.stored_len;
    let sample_len = len.min(SAMPLE_BYTES);
    if len < MIN_RANGE_BYTES || sample_len > *budget {
        return Ok(1);
    }
    *budget -= sample_len;
    let middle = (len - sample_len) / 2;
    reader.seek(entry.data_offset + middle - middle % SAMPLE_ALIGNMENT)?;
    let sample = reader.take(sample_len, SAMPLE_BYTES)?;
    let width = u64::from(width_trial(&sample));
    Ok(if len.is_multiple_of(width) { width } else { 1 })
}

#[cfg(test)]
mod tests {
    use super::is_storage;

    #[test]
    fn storages_are_numbered_entries_of_the_data_directory() {
        assert!(is_storage("archive/data/0", "archive"));
        assert!(is_storage("archive/data/1234", "archive"));
        assert!(!is_storage("archive/data/", "archive"));
        assert!(!is_storage("archive/data/1a", "archive"));
        assert!(!is_storage("archive/data.pkl", "archive"));
        assert!(!is_storage("archive/.data/serialization_id", "archive"));
        assert!(!is_storage("other/data/0", "archive"));
        assert!(!is_storage("archive/sub/data/0", "archive"));
    }
}
