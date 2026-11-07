//! File: format/index.rs
//! Purpose: the decompressed index: section table and record arrays, spec section 4.
//!
//! Main responsibilities:
//! - [`Index::parse`] enforces the section table rules (invariant 2) and decodes every record;
//! - typed accessors that stay in bounds on any index, validated or not.
//!
//! Key structures: [`Index`], `SectionEntry`, `SectionSpec`.
//!
//! Implementation notes: the input is hostile. No allocation is sized by a declared count: the
//! table length is checked against the buffer first, and record arrays are sized by the bodies they
//! were cut from. `Index` owns copies of everything, so the raw buffer can be dropped after
//! parsing. Relations between records are not checked here but in `validate.rs`.

use std::ops::Range;

use super::limits::Limits;
use super::records::{
    BLOB_SIZE, BlobRecord, CHAN_SIZE, CHNK_SIZE, CHST_SIZE, ChainRecord, ChunkRecord,
    DELTA_PARAMS_LEN, DeltaParams, EXTN_SIZE, ExtentRecord, MEMB_SIZE, MemberRecord, PARAMS_ALIGN,
    SHAP_SIZE, StrRef, TENS_SIZE, TensorRecord, decode_step,
};
use super::wire::{bytes_at, is_zero_at, malformed, u32_at, u64_at};
use crate::chain::{Chain, RawStep};
use crate::error::Error;

/// Bytes before the first section table entry: `section_count` and a zero `u32`.
pub(super) const TABLE_HEADER_LEN: usize = 8;
/// Bytes of a section table entry.
pub(super) const TABLE_ENTRY_LEN: usize = 32;
/// Alignment of section bodies, in bytes.
pub(super) const BODY_ALIGN: usize = 8;
/// `flags` bit of a section table entry: the section is required.
pub(super) const FLAG_REQUIRED: u32 = 1;

/// A section this version knows: its tag, and its record size in bytes (1 for a byte heap).
#[derive(Debug, Clone, Copy)]
pub(super) struct SectionSpec {
    /// The four-byte tag as text. Also the `section` of errors about this section.
    pub(super) tag: &'static str,
    /// Bytes of one record.
    pub(super) record_size: usize,
}

pub(super) const STRS: SectionSpec = SectionSpec { tag: "STRS", record_size: 1 };
pub(super) const CHAN: SectionSpec = SectionSpec { tag: "CHAN", record_size: CHAN_SIZE };
pub(super) const CHST: SectionSpec = SectionSpec { tag: "CHST", record_size: CHST_SIZE };
pub(super) const BLOB: SectionSpec = SectionSpec { tag: "BLOB", record_size: BLOB_SIZE };
pub(super) const CHNK: SectionSpec = SectionSpec { tag: "CHNK", record_size: CHNK_SIZE };
pub(super) const MEMB: SectionSpec = SectionSpec { tag: "MEMB", record_size: MEMB_SIZE };
pub(super) const EXTN: SectionSpec = SectionSpec { tag: "EXTN", record_size: EXTN_SIZE };
pub(super) const TENS: SectionSpec = SectionSpec { tag: "TENS", record_size: TENS_SIZE };
pub(super) const SHAP: SectionSpec = SectionSpec { tag: "SHAP", record_size: SHAP_SIZE };
/// Heap of blob-kind parameters, version 0.3. Optional; whether its required bit may be set is a
/// rule between sections and the header, checked by `validate`.
pub(super) const PARM: SectionSpec = SectionSpec { tag: "PARM", record_size: 1 };

/// The sections every file has, in the order a writer lays them out.
pub(super) const MANDATORY_SECTIONS: [SectionSpec; 9] =
    [STRS, CHAN, CHST, BLOB, CHNK, MEMB, EXTN, TENS, SHAP];

/// One checked entry of the section table. `span` lies inside the index, after the table.
#[derive(Debug)]
struct SectionEntry {
    tag: [u8; 4],
    required: bool,
    span: Range<usize>,
    count: u64,
}

impl SectionEntry {
    /// Decodes an entry and applies the rules that concern it alone. `table_end` and `raw_len` are
    /// the end of the section table and the length of the index, in bytes.
    fn decode(
        bytes: &[u8; TABLE_ENTRY_LEN],
        table_end: usize,
        raw_len: usize,
    ) -> Result<Self, Error> {
        let tag = bytes_at::<0, 4, TABLE_ENTRY_LEN>(bytes);
        let shown = tag.escape_ascii();
        if !tag.is_ascii() {
            return Err(malformed("index", format!("section tag {shown} is not ASCII")));
        }
        let flags = u32_at::<4, _>(bytes);
        if flags & !FLAG_REQUIRED != 0 {
            return Err(malformed("index", format!("section {shown} has flags {flags:#x}")));
        }
        let offset = u64_at::<8, _>(bytes);
        let len = u64_at::<16, _>(bytes);
        // Whatever the section is, its body must obey these, or it could not be skipped safely.
        let span = usize::try_from(offset)
            .ok()
            .zip(usize::try_from(len).ok())
            .and_then(|(start, len)| Some(start..start.checked_add(len)?))
            .filter(|span| span.start >= table_end && span.end <= raw_len)
            .ok_or_else(|| {
                malformed(
                    "index",
                    format!("section {shown}: body of {len} bytes at {offset} is out of range"),
                )
            })?;
        if !span.start.is_multiple_of(BODY_ALIGN) {
            return Err(malformed("index", format!("section {shown}: offset {offset} unaligned")));
        }
        Ok(Self { tag, required: flags & FLAG_REQUIRED != 0, span, count: u64_at::<24, _>(bytes) })
    }

    /// Checks the rule of spec section 4 that ties `len`, `count` and the record size.
    fn check_record_array(&self, spec: SectionSpec) -> Result<(), Error> {
        // Lossless: `usize` is at most 64 bits on every supported target.
        let (len, record_size) = (self.span.len() as u64, spec.record_size as u64);
        if self.count.checked_mul(record_size) != Some(len) {
            return Err(malformed(
                spec.tag,
                format!("{} records of {record_size} bytes do not fill {len} bytes", self.count),
            ));
        }
        Ok(())
    }
}

/// Parses the section table and applies every rule of spec section 4 that does not need the
/// contents of a body: tags, ranges, disjoint bodies, record arrays, known and required sections.
fn parse_table(raw: &[u8]) -> Result<Vec<SectionEntry>, Error> {
    let (head, rest) = raw
        .split_first_chunk::<TABLE_HEADER_LEN>()
        .ok_or_else(|| malformed("index", format!("{} bytes cannot hold a table", raw.len())))?;
    if !is_zero_at::<4, 4, TABLE_HEADER_LEN>(head) {
        return Err(malformed("index", "the word after section_count is not zero"));
    }
    let section_count = u32_at::<0, _>(head);
    // Checked against the buffer before anything is allocated.
    let entry_bytes = usize::try_from(section_count)
        .ok()
        .and_then(|count| count.checked_mul(TABLE_ENTRY_LEN))
        .and_then(|len| rest.get(..len))
        .ok_or_else(|| {
            malformed("index", format!("{section_count} table entries do not fit in the index"))
        })?;
    let table_end = TABLE_HEADER_LEN + entry_bytes.len();
    let (entry_records, _) = entry_bytes.as_chunks::<TABLE_ENTRY_LEN>();
    let entries = entry_records
        .iter()
        .map(|bytes| SectionEntry::decode(bytes, table_end, raw.len()))
        .collect::<Result<Vec<_>, _>>()?;
    check_unique_tags(&entries)?;
    check_disjoint_bodies(&entries)?;
    for entry in &entries {
        check_entry(entry)?;
    }
    for spec in MANDATORY_SECTIONS {
        find_section(&entries, spec)?;
    }
    Ok(entries)
}

/// Rejects a repeated tag.
fn check_unique_tags(entries: &[SectionEntry]) -> Result<(), Error> {
    let mut tags: Vec<[u8; 4]> = entries.iter().map(|entry| entry.tag).collect();
    tags.sort_unstable();
    match tags.windows(2).find(|pair| pair[0] == pair[1]) {
        Some(pair) => {
            Err(malformed("index", format!("section {} appears twice", pair[0].escape_ascii())))
        }
        None => Ok(()),
    }
}

/// Rejects overlapping bodies. An empty body holds no bytes, so it overlaps nothing.
fn check_disjoint_bodies(entries: &[SectionEntry]) -> Result<(), Error> {
    let mut spans: Vec<(usize, usize)> = entries
        .iter()
        .filter(|entry| !entry.span.is_empty())
        .map(|entry| (entry.span.start, entry.span.end))
        .collect();
    spans.sort_unstable();
    match spans.windows(2).find(|pair| pair[1].0 < pair[0].1) {
        Some(pair) => Err(malformed(
            "index",
            format!("section bodies at {} and {} overlap", pair[0].0, pair[1].0),
        )),
        None => Ok(()),
    }
}

/// Applies the rules that depend on what the section is. Unknown sections fail closed when
/// required (invariant 9) and are skipped otherwise. `PARM` is known with either flag.
fn check_entry(entry: &SectionEntry) -> Result<(), Error> {
    if let Some(spec) = MANDATORY_SECTIONS.iter().find(|spec| spec.tag.as_bytes() == entry.tag) {
        if !entry.required {
            return Err(malformed(spec.tag, "the required bit is not set"));
        }
        return entry.check_record_array(*spec);
    }
    if entry.tag == *PARM.tag.as_bytes() {
        return entry.check_record_array(PARM);
    }
    if entry.required {
        let shown = entry.tag.escape_ascii();
        return Err(malformed("index", format!("required section {shown} is not supported")));
    }
    tracing::debug!(tag = %entry.tag.escape_ascii(), "skipping unknown optional section");
    Ok(())
}

/// Finds the entry of a section. Tags are unique by then.
fn find_section(entries: &[SectionEntry], spec: SectionSpec) -> Result<&SectionEntry, Error> {
    entries
        .iter()
        .find(|entry| entry.tag == *spec.tag.as_bytes())
        .ok_or_else(|| malformed(spec.tag, "the section is missing"))
}

/// Maps `first..first + count` into a slice of `len` items.
///
/// # Errors
/// Returns [`Error::MalformedIndex`] for `section` if the sum overflows or the end is past `len`.
pub(super) fn checked_range(
    first: u64,
    count: u64,
    len: usize,
    section: &'static str,
) -> Result<Range<usize>, Error> {
    usize::try_from(first)
        .ok()
        .zip(usize::try_from(count).ok())
        .and_then(|(start, count)| Some(start..start.checked_add(count)?))
        .filter(|range| range.end <= len)
        .ok_or_else(|| {
            malformed(section, format!("range of {count} at {first} is outside {len} records"))
        })
}

/// A parsed index: all record arrays of a file, decoded and owned.
///
/// A successful [`Index::parse`] guarantees the section table is valid and each record is
/// well-formed on its own. Relations between records hold only after [`super::validate`]; until
/// then the accessors below may return an error, but they never go out of bounds.
#[derive(Debug, PartialEq, Eq)]
pub struct Index {
    strings: Vec<u8>,
    chains: Vec<ChainRecord>,
    steps: Vec<RawStep>,
    blobs: Vec<BlobRecord>,
    chunks: Vec<ChunkRecord>,
    members: Vec<MemberRecord>,
    extents: Vec<ExtentRecord>,
    tensors: Vec<TensorRecord>,
    dims: Vec<u64>,
    /// The `PARM` heap, empty when the section is absent.
    params: Vec<u8>,
    /// Whether `PARM` is present with its required bit.
    params_required: bool,
}

impl Index {
    /// Parses a decompressed index. `raw` should already be checked with
    /// [`super::Footer::check_index`].
    ///
    /// # Errors
    /// - [`Error::BudgetExceeded`] if `raw` is longer than [`Limits::index_cap`].
    /// - [`Error::MalformedIndex`] if the section table breaks spec section 4, a required section
    ///   is missing or unknown, or a record has a non-zero zero field.
    pub fn parse(raw: &[u8], limits: &Limits) -> Result<Self, Error> {
        let cap = limits.index_cap();
        // Lossless: `usize` is at most 64 bits on every supported target.
        let raw_len = raw.len() as u64;
        if raw_len > cap {
            return Err(Error::BudgetExceeded { what: "index", declared: raw_len, limit: cap });
        }
        let entries = parse_table(raw)?;
        let body = |spec| -> Result<&[u8], Error> {
            let span = find_section(&entries, spec)?.span.clone();
            raw.get(span).ok_or_else(|| malformed(spec.tag, "body is outside the index"))
        };
        let (params, params_required) = match find_section(&entries, PARM) {
            Ok(entry) => (body(PARM)?.to_vec(), entry.required),
            Err(_) => (Vec::new(), false),
        };
        let index = Self {
            strings: body(STRS)?.to_vec(),
            chains: decode_records(body(CHAN)?, |bytes| ChainRecord::decode(*bytes))?,
            steps: decode_records(body(CHST)?, |bytes| Ok(decode_step(*bytes)))?,
            blobs: decode_records(body(BLOB)?, BlobRecord::decode)?,
            chunks: decode_records(body(CHNK)?, ChunkRecord::decode)?,
            members: decode_records(body(MEMB)?, MemberRecord::decode)?,
            extents: decode_records(body(EXTN)?, ExtentRecord::decode)?,
            tensors: decode_records(body(TENS)?, |bytes| Ok(TensorRecord::decode(bytes)))?,
            dims: decode_records(body(SHAP)?, |bytes| Ok(u64::from_le_bytes(*bytes)))?,
            params,
            params_required,
        };
        tracing::debug!(
            raw_len,
            sections = entries.len(),
            blobs = index.blobs.len(),
            chunks = index.chunks.len(),
            members = index.members.len(),
            "parsed index"
        );
        Ok(index)
    }

    /// The `STRS` heap.
    #[must_use]
    pub fn strings(&self) -> &[u8] {
        &self.strings
    }

    /// The `CHAN` records.
    #[must_use]
    pub fn chains(&self) -> &[ChainRecord] {
        &self.chains
    }

    /// The `CHST` records.
    #[must_use]
    pub fn steps(&self) -> &[RawStep] {
        &self.steps
    }

    /// The `BLOB` records.
    #[must_use]
    pub fn blobs(&self) -> &[BlobRecord] {
        &self.blobs
    }

    /// The `CHNK` records.
    #[must_use]
    pub fn chunks(&self) -> &[ChunkRecord] {
        &self.chunks
    }

    /// The `MEMB` records.
    #[must_use]
    pub fn members(&self) -> &[MemberRecord] {
        &self.members
    }

    /// The `EXTN` records.
    #[must_use]
    pub fn extents(&self) -> &[ExtentRecord] {
        &self.extents
    }

    /// The `TENS` records.
    #[must_use]
    pub fn tensors(&self) -> &[TensorRecord] {
        &self.tensors
    }

    /// The `SHAP` records.
    #[must_use]
    pub fn dims(&self) -> &[u64] {
        &self.dims
    }

    /// The `PARM` heap, empty when the file has no `PARM` section.
    #[must_use]
    pub fn params(&self) -> &[u8] {
        &self.params
    }

    /// Whether the file has a `PARM` section with its required bit set.
    #[must_use]
    pub fn params_required(&self) -> bool {
        self.params_required
    }

    /// Resolves the `PARM` entry of a delta blob. Checks only the entry's own placement: its
    /// relations to other blobs are rules of [`super::validate`].
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] for `PARM` unless `params_len` is [`DELTA_PARAMS_LEN`],
    /// `params_off` is a multiple of [`PARAMS_ALIGN`], the entry lies in the heap and its zero field
    /// is zero.
    pub fn delta_params(&self, blob: &BlobRecord) -> Result<DeltaParams, Error> {
        if blob.params_len != DELTA_PARAMS_LEN || !blob.params_off.is_multiple_of(PARAMS_ALIGN) {
            return Err(malformed(
                PARM.tag,
                format!(
                    "a delta entry needs {DELTA_PARAMS_LEN} bytes at a multiple of {PARAMS_ALIGN}, \
                     not {} at {}",
                    blob.params_len, blob.params_off
                ),
            ));
        }
        let range = checked_range(
            blob.params_off,
            u64::from(blob.params_len),
            self.params.len(),
            PARM.tag,
        )?;
        let bytes = self.params[range]
            .first_chunk()
            .ok_or_else(|| malformed(PARM.tag, "a delta entry is short"))?;
        DeltaParams::decode(bytes)
    }

    /// Resolves a string reference.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] for `STRS` if the range is outside the heap or is not
    /// valid UTF-8.
    pub fn string(&self, reference: StrRef) -> Result<&str, Error> {
        let (off, len) = (u64::from(reference.off), u64::from(reference.len));
        let range = checked_range(off, len, self.strings.len(), STRS.tag)?;
        std::str::from_utf8(&self.strings[range])
            .map_err(|error| malformed(STRS.tag, format!("string at {off}: {error}")))
    }

    /// Resolves a chain by `chain_id`. A reader takes its chains from [`super::validate`] instead,
    /// which resolves each one once.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] for `CHAN` if `chain_id` or the step range is out of range
    /// or the chain breaks spec section 5.2, and any error of [`Chain::from_raw`].
    pub fn chain(&self, chain_id: u32) -> Result<Chain, Error> {
        let range = checked_range(u64::from(chain_id), 1, self.chains.len(), CHAN.tag)?;
        let record = self.chains[range.start];
        let steps = checked_range(
            u64::from(record.first_step),
            u64::from(record.step_count),
            self.steps.len(),
            CHAN.tag,
        )?;
        Chain::from_raw(&self.steps[steps])
    }

    /// The chunks of a blob.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] for `BLOB` if the range is outside `CHNK`.
    pub fn blob_chunks(&self, blob: &BlobRecord) -> Result<&[ChunkRecord], Error> {
        let count = u64::from(blob.chunk_count);
        let range = checked_range(blob.first_chunk, count, self.chunks.len(), BLOB.tag)?;
        Ok(&self.chunks[range])
    }

    /// The extents of a member.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] for `MEMB` if the range is outside `EXTN`.
    pub fn member_extents(&self, member: &MemberRecord) -> Result<&[ExtentRecord], Error> {
        let count = u64::from(member.extent_count);
        let range = checked_range(member.first_extent, count, self.extents.len(), MEMB.tag)?;
        Ok(&self.extents[range])
    }

    /// The tensors of a member.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] for `MEMB` if the range is outside `TENS`.
    pub fn member_tensors(&self, member: &MemberRecord) -> Result<&[TensorRecord], Error> {
        let (first, count) = (u64::from(member.first_tensor), u64::from(member.tensor_count));
        let range = checked_range(first, count, self.tensors.len(), MEMB.tag)?;
        Ok(&self.tensors[range])
    }

    /// The dimensions of a tensor.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] for `TENS` if the range is outside `SHAP`.
    pub fn tensor_shape(&self, tensor: &TensorRecord) -> Result<&[u64], Error> {
        let (first, count) = (u64::from(tensor.shape_off), u64::from(tensor.ndim));
        let range = checked_range(first, count, self.dims.len(), TENS.tag)?;
        Ok(&self.dims[range])
    }
}

/// Decodes a record array. Its length is that of `body`, so a declared count cannot size it. `body`
/// is a whole number of records by [`SectionEntry::check_record_array`].
fn decode_records<const R: usize, T>(
    body: &[u8],
    decode: impl Fn(&[u8; R]) -> Result<T, Error>,
) -> Result<Vec<T>, Error> {
    let (records, _) = body.as_chunks::<R>();
    records.iter().map(decode).collect()
}

#[cfg(test)]
mod tests {
    use super::{BODY_ALIGN, Index, MANDATORY_SECTIONS, checked_range};
    use crate::error::Error;
    use crate::format::Limits;
    use crate::format::builder::IndexBuilder;
    use crate::format::testkit::{Section, lay_out, mandatory_sections};

    fn parse(raw: &[u8]) -> Result<Index, Error> {
        Index::parse(raw, &Limits::default())
    }

    fn malformed_in<T: std::fmt::Debug>(result: Result<T, Error>, expected: &str) {
        let error = result.expect_err("malformed");
        assert!(
            matches!(&error, Error::MalformedIndex { section, .. } if *section == expected),
            "{error:?}"
        );
    }

    /// Replaces the bytes of table entry `entry` at `field_at` of the entry.
    fn patch_entry(raw: &mut [u8], entry: usize, field_at: usize, value: &[u8]) {
        let at = 8 + entry * 32 + field_at;
        raw[at..at + value.len()].copy_from_slice(value);
    }

    #[test]
    fn hand_laid_empty_index_parses() {
        let index = parse(&lay_out(&mandatory_sections())).expect("valid");
        assert!(index.members().is_empty() && index.strings().is_empty());
    }

    #[test]
    fn short_buffers_are_rejected() {
        for len in 0..8 {
            malformed_in(parse(&vec![0; len]), "index");
        }
    }

    #[test]
    fn nonzero_table_word_is_rejected() {
        let mut raw = lay_out(&mandatory_sections());
        raw[4] = 1;
        malformed_in(parse(&raw), "index");
    }

    #[test]
    fn section_count_past_the_buffer_is_rejected_without_allocating() {
        let mut raw = lay_out(&mandatory_sections());
        raw[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        malformed_in(parse(&raw), "index");
    }

    #[test]
    fn index_over_the_cap_is_a_budget_error() {
        let raw = lay_out(&mandatory_sections());
        let limits = Limits { max_index_bytes: 16, ..Limits::default() };
        assert!(matches!(
            Index::parse(&raw, &limits),
            Err(Error::BudgetExceeded { what: "index", limit: 16, .. })
        ));
    }

    #[test]
    fn duplicate_tags_are_rejected() {
        let mut sections = mandatory_sections();
        sections.push(Section::new(*b"STRS", 1, vec![], 0));
        malformed_in(parse(&lay_out(&sections)), "index");
    }

    #[test]
    fn duplicate_optional_tags_are_rejected() {
        let mut sections = mandatory_sections();
        sections.push(Section::new(*b"XTRA", 0, vec![], 0));
        sections.push(Section::new(*b"XTRA", 0, vec![], 0));
        malformed_in(parse(&lay_out(&sections)), "index");
    }

    #[test]
    fn non_ascii_tags_are_rejected_even_if_optional() {
        let mut sections = mandatory_sections();
        sections.push(Section::new([b'X', b'T', b'R', 0xFF], 0, vec![], 0));
        malformed_in(parse(&lay_out(&sections)), "index");
    }

    #[test]
    fn unknown_flag_bits_are_rejected() {
        let mut raw = lay_out(&mandatory_sections());
        patch_entry(&mut raw, 0, 4, &[3]);
        malformed_in(parse(&raw), "index");
    }

    #[test]
    fn misaligned_bodies_are_rejected() {
        let mut sections = mandatory_sections();
        sections.push(Section::new(*b"XTRA", 0, vec![1; 16], 16));
        let mut raw = lay_out(&sections);
        let last = sections.len() - 1;
        let offset = u64::from_le_bytes(raw[8 + last * 32 + 8..][..8].try_into().expect("8"));
        for delta in 1..BODY_ALIGN as u64 {
            patch_entry(&mut raw, last, 8, &(offset + delta).to_le_bytes());
            patch_entry(&mut raw, last, 16, &1_u64.to_le_bytes());
            malformed_in(parse(&raw), "index");
        }
    }

    #[test]
    fn bodies_inside_the_table_are_rejected() {
        let mut raw = lay_out(&mandatory_sections());
        patch_entry(&mut raw, 0, 8, &0_u64.to_le_bytes());
        malformed_in(parse(&raw), "index");
        patch_entry(&mut raw, 0, 8, &8_u64.to_le_bytes());
        malformed_in(parse(&raw), "index");
    }

    #[test]
    fn bodies_past_the_end_are_rejected() {
        let raw = lay_out(&mandatory_sections());
        let end = raw.len() as u64;
        let cases = [
            (end + 8, 0),
            (end - end % 8, 8),
            (u64::MAX - 7, 0),
            (8, u64::MAX),
            (u64::MAX - 7, 16),
        ];
        for (offset, len) in cases {
            let mut raw = raw.clone();
            patch_entry(&mut raw, 0, 8, &offset.to_le_bytes());
            patch_entry(&mut raw, 0, 16, &len.to_le_bytes());
            malformed_in(parse(&raw), "index");
        }
    }

    #[test]
    fn overlapping_bodies_are_rejected() {
        let mut sections = mandatory_sections();
        sections.push(Section::new(*b"XTRA", 0, vec![0; 16], 16));
        sections.push(Section::new(*b"XTRB", 0, vec![0; 16], 16));
        let mut raw = lay_out(&sections);
        let last = sections.len() - 1;
        let entry_offset = |raw: &[u8], entry: usize| raw[8 + entry * 32 + 8..][..8].to_vec();
        let earlier = entry_offset(&raw, last - 1);
        patch_entry(&mut raw, last, 8, &earlier);
        malformed_in(parse(&raw), "index");
    }

    #[test]
    fn empty_bodies_may_sit_anywhere_valid() {
        let mut sections = mandatory_sections();
        sections.push(Section::new(*b"XTRA", 0, vec![0; 32], 32));
        let mut raw = lay_out(&sections);
        let offset = u64::from_le_bytes(raw[8 + 9 * 32 + 8..][..8].try_into().expect("8"));
        // An empty `STRS` inside the optional body holds no bytes, so nothing overlaps.
        patch_entry(&mut raw, 0, 8, &(offset + 8).to_le_bytes());
        parse(&raw).expect("an empty body overlaps nothing");
    }

    #[test]
    fn every_missing_mandatory_section_is_rejected() {
        for spec in MANDATORY_SECTIONS {
            let mut sections = mandatory_sections();
            sections.retain(|section| section.tag != *spec.tag.as_bytes());
            malformed_in(parse(&lay_out(&sections)), spec.tag);
        }
    }

    #[test]
    fn every_mandatory_section_needs_its_required_bit() {
        for at in 0..MANDATORY_SECTIONS.len() {
            let mut sections = mandatory_sections();
            sections[at].flags = 0;
            malformed_in(parse(&lay_out(&sections)), MANDATORY_SECTIONS[at].tag);
        }
    }

    #[test]
    fn record_array_length_must_equal_count_times_size() {
        for (at, spec) in MANDATORY_SECTIONS.iter().enumerate() {
            let mut sections = mandatory_sections();
            sections[at].body = vec![0; spec.record_size * 3 + 1];
            sections[at].count = 3;
            malformed_in(parse(&lay_out(&sections)), spec.tag);
        }
    }

    #[test]
    fn count_multiplication_overflow_is_a_rejection() {
        for (at, spec) in MANDATORY_SECTIONS.iter().enumerate().skip(1) {
            let mut sections = mandatory_sections();
            let size = spec.record_size as u64;
            sections[at].count = u64::MAX / size + 1;
            malformed_in(parse(&lay_out(&sections)), spec.tag);
            sections[at].count = u64::MAX;
            malformed_in(parse(&lay_out(&sections)), spec.tag);
        }
    }

    #[test]
    fn unknown_required_section_is_rejected() {
        let mut sections = mandatory_sections();
        sections.push(Section::new(*b"XTRA", 1, vec![], 0));
        malformed_in(parse(&lay_out(&sections)), "index");
    }

    #[test]
    fn unknown_optional_section_is_skipped_and_its_count_ignored() {
        let mut sections = mandatory_sections();
        sections.push(Section::new(*b"XTRA", 0, vec![0xFF; 12], u64::MAX));
        parse(&lay_out(&sections)).expect("skipped");
    }

    #[test]
    fn optional_parm_is_skipped_but_must_be_a_heap() {
        let mut sections = mandatory_sections();
        sections.push(Section::new(*b"PARM", 0, vec![0xFF; 12], 12));
        parse(&lay_out(&sections)).expect("optional");
        let last = sections.len() - 1;
        sections[last].count = 11;
        malformed_in(parse(&lay_out(&sections)), "PARM");
    }

    #[test]
    fn parm_is_kept_with_its_required_bit() {
        let mut sections = mandatory_sections();
        sections.push(Section::new(*b"PARM", 1, vec![7; 16], 16));
        let index = parse(&lay_out(&sections)).expect("PARM is known");
        assert_eq!((index.params(), index.params_required()), (&[7; 16][..], true));
        let last = sections.len() - 1;
        sections[last].flags = 0;
        let index = parse(&lay_out(&sections)).expect("optional");
        assert_eq!((index.params(), index.params_required()), (&[7; 16][..], false));
        sections[last].count = 15;
        malformed_in(parse(&lay_out(&sections)), "PARM");
        sections.pop();
        let index = parse(&lay_out(&sections)).expect("no PARM");
        assert!(index.params().is_empty() && !index.params_required());
    }

    #[test]
    fn a_delta_entry_is_resolved_only_whole_aligned_and_in_the_heap() {
        use crate::format::records::{BlobRecord, DeltaParams};
        let entry = DeltaParams { base_blob: 3, base_offset: 64 };
        let mut builder = IndexBuilder::new();
        assert_eq!(builder.add_params(&[1, 2, 3, 4, 5]).expect("fits"), (0, 5));
        assert_eq!(builder.add_params(&entry.encode()).expect("fits"), (8, 16));
        let index = parse(&builder.serialize()).expect("valid");
        assert_eq!(index.params()[..8], [1, 2, 3, 4, 5, 0, 0, 0]);
        assert_eq!(index.params().len(), 24);
        assert!(index.params_required());
        let blob = BlobRecord {
            blake3: [0; 32],
            raw_len: 0,
            first_chunk: 0,
            chunk_count: 0,
            chunk_size: 1,
            chain_id: 0,
            kind: 1,
            dtype_hint: crate::Dtype::U8,
            params_off: 8,
            params_len: 16,
        };
        assert_eq!(index.delta_params(&blob).expect("an entry"), entry);
        for (params_off, params_len) in [(0, 16), (16, 16), (8, 15), (8, 17), (4, 16), (u64::MAX - 7, 16)] {
            let blob = BlobRecord { params_off, params_len, ..blob };
            malformed_in(index.delta_params(&blob), "PARM");
        }
    }

    #[test]
    fn bytes_between_bodies_are_ignored() {
        let mut sections = mandatory_sections();
        sections.push(Section::new(*b"XTRA", 0, vec![], 0));
        let mut raw = lay_out(&sections);
        raw.extend_from_slice(&[0xEE; 40]);
        parse(&raw).expect("gap bytes are ignored");
    }

    #[test]
    fn a_record_with_a_nonzero_zero_field_fails_the_parse() {
        let mut sections = mandatory_sections();
        sections[1] = Section::new(*b"CHAN", 1, vec![0, 0, 0, 0, 1, 0, 1, 0], 1);
        malformed_in(parse(&lay_out(&sections)), "CHAN");
    }

    #[test]
    fn accessors_stay_in_bounds_on_unvalidated_records() {
        use crate::format::records::{BlobRecord, ChainRecord, MemberRecord, StrRef, TensorRecord};
        let mut builder = IndexBuilder::new();
        let chain = ChainRecord { first_step: u32::MAX, step_count: u16::MAX };
        builder.add_raw_chain(chain).expect("fits");
        let index = parse(&builder.serialize()).expect("structurally valid");
        let string = StrRef { off: u32::MAX, len: u32::MAX };
        malformed_in(index.string(string), "STRS");
        malformed_in(index.chain(0), "CHAN");
        malformed_in(index.chain(1), "CHAN");
        malformed_in(index.chain(u32::MAX), "CHAN");
        let blob = BlobRecord {
            blake3: [0; 32],
            raw_len: 0,
            first_chunk: u64::MAX,
            chunk_count: u32::MAX,
            chunk_size: 0,
            chain_id: 0,
            kind: 0,
            dtype_hint: crate::Dtype::U8,
            params_off: 0,
            params_len: 0,
        };
        malformed_in(index.blob_chunks(&blob), "BLOB");
        let member = MemberRecord {
            name: StrRef { off: 0, len: 0 },
            file_len: 0,
            blake3: [0; 32],
            first_extent: u64::MAX,
            extent_count: 1,
            first_tensor: u32::MAX,
            tensor_count: u32::MAX,
            profile: crate::Profile::Opaque,
        };
        malformed_in(index.member_extents(&member), "MEMB");
        malformed_in(index.member_tensors(&member), "MEMB");
        let tensor = TensorRecord {
            name: StrRef { off: 0, len: 0 },
            virt_offset: 0,
            virt_len: 0,
            shape_off: u32::MAX,
            ndim: u16::MAX,
            dtype: crate::Dtype::U8,
        };
        malformed_in(index.tensor_shape(&tensor), "TENS");
    }

    #[test]
    fn checked_range_rejects_overflow_at_u64_max() {
        assert_eq!(checked_range(2, 3, 5, "x").expect("fits"), 2..5);
        assert_eq!(checked_range(5, 0, 5, "x").expect("empty at the end"), 5..5);
        assert!(checked_range(6, 0, 5, "x").is_err());
        assert!(checked_range(u64::MAX, 1, 5, "x").is_err());
        assert!(checked_range(1, u64::MAX, usize::MAX, "x").is_err());
        assert!(checked_range(u64::MAX, u64::MAX, usize::MAX, "x").is_err());
    }
}
