//! File: format/validate.rs
//! Purpose: the structural invariants of spec section 9, over a parsed index.
//!
//! Main responsibilities:
//! - [`validate`] enforces invariants 3 to 9 and 13, re-applies the footer half of 1 and 2, checks
//!   that the header's required feature bits agree with the index, and hands back the chains it
//!   resolved;
//! - [`check_decode_budget`] is the first half of invariant 10, plus the decode work bound of spec
//!   section 4;
//! - [`check_restore_name`] is invariant 11.
//!
//! Key structures: `RangeMax`, `ChunkBounds`, `TensorSpans`.
//!
//! Implementation notes: the index is hostile, so the cost of validation must not multiply two
//! counts from the file. Records may share ranges: extents between members by spec section 8, and
//! chunks between blobs and tensors between members because nothing forbids it. So every per-record
//! rule runs once per record, and every per-range rule is answered from a structure built once: a
//! prefix count for extent contiguity, and `RangeMax` for the rest. Names are the one remaining
//! product, of records and heap size, and `Limits::max_name_work_bytes` bounds it. Total cost is
//! O(n log n) in the size of the index, plus that bound.

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::ops::Range;

use super::footer::Footer;
use super::header::{HEADER_LEN, Header};
use super::index::{BLOB, CHAN, CHNK, EXTN, Index, MEMB, PARM, TENS, checked_range};
use super::limits::{DECODE_WORK_FLOOR, Limits};
use super::records::{
    BlobRecord, ChunkRecord, DeltaParams, ExtentRecord, MemberRecord, TensorRecord,
};
use super::wire::malformed;
use crate::chain::{BLOB_KIND_CHUNKED, BLOB_KIND_DELTA, Chain, FEATURE_DELTA, Step, View};
use crate::chain::MAX_CHUNK_SIZE;
use crate::error::Error;

/// Largest `ndim` of a tensor (spec section 4.1).
pub const MAX_NDIM: u16 = 16;
/// Bytes of one stream word of a chunk prologue (spec section 6.1).
const STREAM_WORD_LEN: u64 = 4;

/// Validates a parsed index against its file and returns its chains, resolved once.
///
/// `header` and `footer` must be the ones the index was read through and `file_len` the real file
/// length. The header's required feature bits must agree with the index: bit 0 is set exactly when a
/// delta blob exists, and only then may `PARM` carry its required bit.
/// Everything [`Index::parse`] and [`Footer::check_layout`] leave open is checked here: on `Ok`,
/// all references are in range, so a reader may use the records. Chunk contents are not looked at.
///
/// The returned vector has one [`Chain`] per `CHAN` record, indexed by `chain_id`, so a reader never
/// has to go through [`Index::chain`] on its decode path.
///
/// # Errors
/// - [`Error::BudgetExceeded`] if the footer exceeds the index cap, or the names exceed
///   [`Limits::max_name_work_bytes`].
/// - [`Error::UnknownCodec`] for an unknown step or view id, in any record, referenced or not.
/// - [`Error::MalformedIndex`] for anything else, naming the section at fault, or `"header"` for
///   a feature bit the index does not use.
pub fn validate(
    index: &Index,
    header: &Header,
    footer: &Footer,
    file_len: u64,
    limits: &Limits,
) -> Result<Vec<Chain>, Error> {
    footer.check_layout(file_len, limits)?;
    check_name_budget(index, limits)?;
    let chains = check_chains(index)?;
    check_chunk_ranges(index.chunks(), footer.index_offset)?;
    check_delta_feature(index, header)?;
    check_blobs(index, &chains)?;
    check_extents(index)?;
    let tensors = check_tensors(index)?;
    check_members(index, &tensors)?;
    tracing::debug!(
        blobs = index.blobs().len(),
        chunks = index.chunks().len(),
        members = index.members().len(),
        extents = index.extents().len(),
        tensors = index.tensors().len(),
        "validated index"
    );
    Ok(chains)
}

/// Checks the cost of decoding the members `members`, before any data read: the first half of
/// invariant 10, then their summed decode work. `members` are indexes into `MEMB`; pass `0..count`
/// to budget the whole container. `index` must have passed [`validate`].
///
/// Decode work costs O(n) in the index, however members share extents.
///
/// # Errors
/// - [`Error::BudgetExceeded`] with `what` `"total decoded size"` if the sum of `file_len` is over
///   [`Limits::max_total_decoded`]. A sum that overflows `u64` is reported as `u64::MAX`.
/// - [`Error::BudgetExceeded`] with `what` `"decode work"` if the work of all of them is over
///   [`Limits::max_decode_amplification`] × the sum of `file_len` + [`DECODE_WORK_FLOOR`], one floor
///   however many members there are. Values over `u64` are
///   reported as `u64::MAX`.
/// - [`Error::MemberNotFound`] if an index is out of range, which is a bug of the caller.
/// - [`Error::MalformedIndex`] if a reference is broken, which [`validate`] rules out.
pub fn check_decode_budget(
    index: &Index,
    members: impl IntoIterator<Item = u32>,
    limits: &Limits,
) -> Result<(), Error> {
    let limit = limits.max_total_decoded;
    let work = decode_work_totals(index)?;
    let mut total: u64 = 0;
    let mut total_work: u128 = 0;
    for id in members {
        let member = usize::try_from(id)
            .ok()
            .and_then(|id| index.members().get(id))
            .ok_or_else(|| Error::MemberNotFound { name: format!("#{id}") })?;
        let exceeded = |declared| Error::BudgetExceeded {
            what: "total decoded size",
            declared,
            limit,
        };
        total = total.checked_add(member.file_len).ok_or_else(|| exceeded(u64::MAX))?;
        if total > limit {
            return Err(exceeded(total));
        }
        // Saturating cannot matter: a sum this large is over any limit `check_total_work` can state.
        total_work = total_work.saturating_add(member_work(member, &work)?);
    }
    check_total_work(total_work, total, limits)
}

/// For each index `i`, the decode work of the extents before `i`, so a member's work is one
/// subtraction whatever other members share its extents. `u128`, as the sum of up to 2^32 `u64`
/// values cannot overflow it, which the checked addition restates.
fn decode_work_totals(index: &Index) -> Result<Vec<u128>, Error> {
    let mut totals = Vec::with_capacity(index.extents().len() + 1);
    let mut total: u128 = 0;
    totals.push(total);
    for (id, extent) in index.extents().iter().enumerate() {
        let work = extent_work(extent, index)
            .ok_or_else(|| malformed(EXTN.tag, format!("extent {id}: no valid blob range")))?;
        total = total
            .checked_add(u128::from(work))
            .ok_or_else(|| malformed(EXTN.tag, "decode work overflows"))?;
        totals.push(total);
    }
    Ok(totals)
}

/// Raw bytes of every chunk of the blob that `extent` touches, plus, for a delta blob, of every base
/// chunk those are combined with (spec section 4). `None` if the extent or its blob is invalid.
fn extent_work(extent: &ExtentRecord, index: &Index) -> Option<u64> {
    let blobs = index.blobs();
    let blob = blobs.get(usize::try_from(extent.blob_id).ok()?)?;
    let chunk_size = u64::from(blob.chunk_size);
    let blob_len = extent.virt_len.checked_div(extent.view().ok()?.length_factor())?;
    let last_byte = extent.blob_offset.checked_add(blob_len)?.checked_sub(1)?;
    let first = extent.blob_offset.checked_div(chunk_size)?;
    let last = last_byte.checked_div(chunk_size)?;
    let own = chunk_run_work(blob, first, last)?;
    if blob.kind != BLOB_KIND_DELTA {
        return Some(own);
    }
    // Counted exactly, not as twice `own`: a short delta blob may sit on full base chunks.
    let params = index.delta_params(blob).ok()?;
    let base = blobs.get(usize::try_from(params.base_blob).ok()?)?;
    let base_chunk = |chunk: u64| params.base_chunk(u32::try_from(chunk).ok()?, blob.chunk_size);
    let base_work = chunk_run_work(base, u64::from(base_chunk(first)?), u64::from(base_chunk(last)?));
    own.checked_add(base_work?)
}

/// Raw bytes of chunks `first..=last` of `blob`. `None` on overflow or an inverted run.
fn chunk_run_work(blob: &BlobRecord, first: u64, last: u64) -> Option<u64> {
    let chunk_size = u64::from(blob.chunk_size);
    // The chunk after the last one starts past `raw_len` only for the short chunk that ends a blob.
    let end = last.checked_add(1)?.checked_mul(chunk_size)?.min(blob.raw_len);
    end.checked_sub(first.checked_mul(chunk_size)?)
}

/// Decode work of one member. `work` is from [`decode_work_totals`].
fn member_work(member: &MemberRecord, work: &[u128]) -> Result<u128, Error> {
    let count = u64::from(member.extent_count);
    let range = checked_range(member.first_extent, count, work.len() - 1, MEMB.tag)?;
    Ok(work[range.end] - work[range.start])
}

/// Checks `work` of the members to be decoded, the bound of spec section 4, against their summed length
/// `total_len`. The floor comes once, so more members do not bring more allowance by themselves.
fn check_total_work(work: u128, total_len: u64, limits: &Limits) -> Result<(), Error> {
    // Neither step can overflow: both factors are `u64`, and the floor is far below the headroom.
    let limit = u128::from(limits.max_decode_amplification)
        .checked_mul(u128::from(total_len))
        .and_then(|allowed| allowed.checked_add(u128::from(DECODE_WORK_FLOOR)))
        .unwrap_or(u128::MAX);
    if work > limit {
        return Err(Error::BudgetExceeded {
            what: "decode work",
            declared: u64::try_from(work).unwrap_or(u64::MAX),
            limit: u64::try_from(limit).unwrap_or(u64::MAX),
        });
    }
    Ok(())
}

/// Checks a member name before it is joined to an output directory: invariant 11.
///
/// # Errors
/// Returns [`Error::UnsafeMemberName`] unless the name is not empty, has no NUL and no `:`, does
/// not start with `/` or `\`, and has no `..` part when split on `/` and on `\`.
pub fn check_restore_name(name: &str) -> Result<(), Error> {
    let safe = !name.is_empty()
        && !name.contains(['\0', ':'])
        && !name.starts_with(['/', '\\'])
        && !name.split(['/', '\\']).any(|part| part == "..");
    if safe {
        Ok(())
    } else {
        Err(Error::UnsafeMemberName { name: name.to_owned() })
    }
}

/// Maximum over a range of a fixed array, in O(log n) per query and 2n values of memory.
///
/// A bottom-up segment tree: `nodes[leaves..]` holds the values and each `nodes[i]` for `i` in
/// `1..leaves` the maximum of its two children. `nodes[0]` is unused.
#[derive(Debug)]
struct RangeMax<T> {
    leaves: usize,
    nodes: Vec<T>,
}

impl<T: Copy + Ord> RangeMax<T> {
    /// Builds the tree. `values` has one entry per parsed record, so the index bounds its size.
    fn new(values: &[T]) -> Self {
        let leaves = values.len();
        let mut nodes = Vec::with_capacity(values.len().saturating_mul(2));
        nodes.extend_from_slice(values);
        nodes.extend_from_slice(values);
        for node in (1..leaves).rev() {
            nodes[node] = nodes[2 * node].max(nodes[2 * node + 1]);
        }
        Self { leaves, nodes }
    }

    /// Maximum of the values in `range`, `None` if it is empty. `range` must lie inside the array,
    /// which every caller has from `checked_range`.
    fn max(&self, range: Range<usize>) -> Option<T> {
        let (mut low, mut high) = (range.start + self.leaves, range.end + self.leaves);
        let mut best = None;
        while low < high {
            if low % 2 == 1 {
                best = best.max(Some(self.nodes[low]));
                low += 1;
            }
            if high % 2 == 1 {
                high -= 1;
                best = best.max(Some(self.nodes[high]));
            }
            low /= 2;
            high /= 2;
        }
        best
    }
}

/// Bounds the sum of name lengths before any name is hashed or checked as UTF-8.
fn check_name_budget(index: &Index, limits: &Limits) -> Result<(), Error> {
    let names = index.members().iter().map(|member| member.name);
    let mut names = names.chain(index.tensors().iter().map(|tensor| tensor.name));
    let total = names
        .try_fold(0_u64, |total, name| total.checked_add(u64::from(name.len)))
        .unwrap_or(u64::MAX);
    if total > limits.max_name_work_bytes {
        return Err(Error::BudgetExceeded {
            what: "name bytes",
            declared: total,
            limit: limits.max_name_work_bytes,
        });
    }
    Ok(())
}

/// Checks every `CHST` and `CHAN` record, referenced or not, and returns the chains by `chain_id`.
/// An unknown id anywhere is a rejection (invariant 9). The chain array costs at most its records.
fn check_chains(index: &Index) -> Result<Vec<Chain>, Error> {
    for step in index.steps() {
        Step::from_raw(*step)?;
    }
    let count = u32::try_from(index.chains().len())
        .map_err(|_| malformed(CHAN.tag, "too many records for a u32 id"))?;
    (0..count).map(|chain_id| index.chain(chain_id)).collect()
}

/// Invariant 4: every chunk lies in `64..data_end`, and any two are identical or disjoint.
/// `data_end` is `index_offset`. Runs in O(n log n).
fn check_chunk_ranges(chunks: &[ChunkRecord], data_end: u64) -> Result<(), Error> {
    let mut spans = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let end = chunk.file_offset.checked_add(u64::from(chunk.stored_len));
        if chunk.file_offset < HEADER_LEN || end.is_none_or(|end| end > data_end) {
            return Err(malformed(
                CHNK.tag,
                format!(
                    "{} bytes at {} are outside the data region 64..{data_end}",
                    chunk.stored_len, chunk.file_offset
                ),
            ));
        }
        // An empty range holds no bytes, so it overlaps nothing.
        if chunk.stored_len != 0 {
            spans.push((chunk.file_offset, chunk.stored_len));
        }
    }
    spans.sort_unstable();
    spans.dedup();
    let overlap = spans.windows(2).find(|pair| {
        pair[0].0.checked_add(u64::from(pair[0].1)).is_none_or(|end| end > pair[1].0)
    });
    match overlap {
        Some(pair) => Err(malformed(
            CHNK.tag,
            format!(
                "{} bytes at {} and {} bytes at {} overlap and are not identical",
                pair[0].1, pair[0].0, pair[1].1, pair[1].0
            ),
        )),
        None => Ok(()),
    }
}

/// Largest and smallest `stored_len` of any run of chunks.
#[derive(Debug)]
struct ChunkBounds {
    largest: RangeMax<u32>,
    smallest: RangeMax<Reverse<u32>>,
}

impl ChunkBounds {
    /// Builds the bounds over all chunks of the index.
    fn new(chunks: &[ChunkRecord]) -> Self {
        let stored: Vec<u32> = chunks.iter().map(|chunk| chunk.stored_len).collect();
        let reversed: Vec<Reverse<u32>> = stored.iter().copied().map(Reverse).collect();
        Self { largest: RangeMax::new(&stored), smallest: RangeMax::new(&reversed) }
    }

    /// True if every chunk in `range` has `stored_len` in `low..=high`.
    fn all_within(&self, range: Range<usize>, low: u64, high: u64) -> bool {
        let above = self.largest.max(range.clone()).is_some_and(|max| u64::from(max) > high);
        let below = self.smallest.max(range).is_some_and(|Reverse(min)| u64::from(min) < low);
        !above && !below
    }
}

/// Invariants 5, 7, 9 and 13 for every blob.
fn check_blobs(index: &Index, chains: &[Chain]) -> Result<(), Error> {
    let bounds = ChunkBounds::new(index.chunks());
    let mut blobs = index.blobs().iter().enumerate();
    blobs.try_for_each(|(id, blob)| check_blob(index, blob, id, chains, &bounds))
}

/// Checks one blob. `id` is its index in `BLOB`.
fn check_blob(
    index: &Index,
    blob: &BlobRecord,
    id: usize,
    chains: &[Chain],
    bounds: &ChunkBounds,
) -> Result<(), Error> {
    let fail = |detail: String| malformed(BLOB.tag, format!("blob {id}: {detail}"));
    let delta = match blob.kind {
        BLOB_KIND_CHUNKED if blob.params_off != 0 || blob.params_len != 0 => {
            return Err(fail("a kind 0 blob has parameters".to_owned()));
        }
        BLOB_KIND_CHUNKED => None,
        BLOB_KIND_DELTA => Some(index.delta_params(blob).map_err(|error| locate(error, id))?),
        kind => return Err(fail(format!("kind {kind} is not supported"))),
    };
    let chain = usize::try_from(blob.chain_id)
        .ok()
        .and_then(|chain_id| chains.get(chain_id))
        .ok_or_else(|| fail(format!("chain_id {} is out of range", blob.chain_id)))?;
    match (delta.is_some(), chain.delta().is_some()) {
        (true, false) => return Err(fail("a delta blob's chain has no delta step".to_owned())),
        (false, true) => return Err(fail("a kind 0 blob's chain has a delta step".to_owned())),
        (true, true) | (false, false) => {}
    }
    let block = chain.block_bytes();
    let chunk_size = blob.chunk_size;
    if chunk_size == 0 || chunk_size > MAX_CHUNK_SIZE || !chunk_size.is_multiple_of(block) {
        return Err(fail(format!("chunk_size {chunk_size} is invalid for chain block {block}")));
    }
    if let Some(params) = delta {
        check_delta_base(index.blobs(), blob, id, params)?;
    }
    if !blob.raw_len.is_multiple_of(u64::from(block)) {
        return Err(fail(format!("raw_len {} is no multiple of {block}", blob.raw_len)));
    }
    let chunk_count = u64::from(blob.chunk_count);
    if blob.raw_len.div_ceil(u64::from(chunk_size)) != chunk_count {
        return Err(fail(format!(
            "{chunk_count} chunks do not fit {} bytes in chunks of {chunk_size}",
            blob.raw_len
        )));
    }
    let chunks = checked_range(blob.first_chunk, chunk_count, index.chunks().len(), BLOB.tag)?;
    let Some(last_chunk) = blob.chunk_count.checked_sub(1) else {
        return Ok(());
    };
    let last_raw_len = blob
        .chunk_raw_len(last_chunk)
        .ok_or_else(|| fail("the last chunk has no valid length".to_owned()))?;
    // Lossless: `usize` is at most 64 bits on every supported target.
    let prologue = STREAM_WORD_LEN * chain.stream_count() as u64;
    let last = chunks.end - 1;
    let others_fit =
        bounds.all_within(chunks.start..last, prologue, prologue + u64::from(chunk_size));
    let last_fits = last_raw_len
        .checked_add(prologue)
        .is_some_and(|high| bounds.all_within(last..chunks.end, prologue, high));
    if !others_fit || !last_fits {
        return Err(fail("a chunk has a stored_len outside its bounds".to_owned()));
    }
    Ok(())
}

/// Adds the blob id to a `MalformedIndex` detail from [`Index::delta_params`].
fn locate(error: Error, id: usize) -> Error {
    match error {
        Error::MalformedIndex { section, detail } => {
            Error::MalformedIndex { section, detail: format!("blob {id}: {detail}") }
        }
        other => other,
    }
}

/// Invariant 13 between a delta blob and its base. `blob` has passed the chunk size rule and its
/// chain has a delta step; `id` is its index in `BLOB`. A base with a lower id, of kind 0, rules out
/// cycles and chains of deltas, so that one pass over the blobs is enough.
fn check_delta_base(
    blobs: &[BlobRecord],
    blob: &BlobRecord,
    id: usize,
    params: DeltaParams,
) -> Result<(), Error> {
    let fail = |detail: String| malformed(BLOB.tag, format!("blob {id}: {detail}"));
    let base_id = params.base_blob;
    let base = usize::try_from(base_id)
        .ok()
        .filter(|base| *base < id)
        .and_then(|base| blobs.get(base))
        .ok_or_else(|| fail(format!("base blob {base_id} is not an earlier blob")))?;
    if base.kind != BLOB_KIND_CHUNKED {
        return Err(fail(format!("base blob {base_id} is of kind {}, not 0", base.kind)));
    }
    if base.chunk_size != blob.chunk_size {
        return Err(fail(format!(
            "chunk_size {} differs from {} of base blob {base_id}",
            blob.chunk_size, base.chunk_size
        )));
    }
    if !params.base_offset.is_multiple_of(u64::from(blob.chunk_size)) {
        return Err(fail(format!(
            "base_offset {} is no multiple of chunk_size {}",
            params.base_offset, blob.chunk_size
        )));
    }
    let end = params.base_offset.checked_add(blob.raw_len);
    if end.is_none_or(|end| end > base.raw_len) {
        return Err(fail(format!(
            "{} bytes at {} are outside base blob {base_id} of {}",
            blob.raw_len, params.base_offset, base.raw_len
        )));
    }
    Ok(())
}

/// The required feature bit of delta blobs agrees with the index both ways, and so does the
/// required bit of `PARM`: the three layers that make an older reader fail closed on a delta file
/// (spec section 7a), kept honest so that a forged header is detected.
fn check_delta_feature(index: &Index, header: &Header) -> Result<(), Error> {
    let flagged = header.required_features & FEATURE_DELTA != 0;
    let first_delta = index.blobs().iter().position(|blob| blob.kind == BLOB_KIND_DELTA);
    match (first_delta, flagged) {
        (Some(id), false) => Err(malformed(
            BLOB.tag,
            format!("blob {id}: a delta blob in a file without required feature bit 0"),
        )),
        (None, true) => {
            Err(malformed("header", "required feature bit 0 is set, but no blob is a delta blob"))
        }
        (Some(_), true) if !index.params_required() => {
            Err(malformed(PARM.tag, "a file with delta blobs needs PARM with its required bit"))
        }
        (None, false) if index.params_required() => {
            Err(malformed(PARM.tag, "the required bit is set in a file without delta blobs"))
        }
        (Some(_) | None, _) => Ok(()),
    }
}

/// The rules of invariants 6, 7 and 9 that concern one extent, for every record.
fn check_extents(index: &Index) -> Result<(), Error> {
    index.extents().iter().enumerate().try_for_each(|(id, extent)| {
        check_extent(extent, id, index.blobs())
    })
}

/// Checks one extent. `id` is only for messages.
fn check_extent(extent: &ExtentRecord, id: usize, blobs: &[BlobRecord]) -> Result<(), Error> {
    let fail = |detail: String| malformed(EXTN.tag, format!("extent {id}: {detail}"));
    if extent.virt_len == 0 {
        return Err(fail("virt_len is 0".to_owned()));
    }
    if extent.virt_offset.checked_add(extent.virt_len).is_none() {
        return Err(fail("the virtual range overflows".to_owned()));
    }
    let view = extent.view()?;
    let blob = usize::try_from(extent.blob_id)
        .ok()
        .and_then(|blob_id| blobs.get(blob_id))
        .ok_or_else(|| fail(format!("blob_id {} is out of range", extent.blob_id)))?;
    let blob_len = match view {
        View::Identity => extent.virt_len,
        View::F16ToF32 | View::Bf16ToF32 => {
            // `virt_len / 2` being a multiple of 2 is the same as `virt_len` being one of 4.
            if !extent.virt_len.is_multiple_of(4) || !extent.blob_offset.is_multiple_of(2) {
                return Err(fail("a widened range is not element aligned".to_owned()));
            }
            extent.virt_len / view.length_factor()
        }
    };
    let end = extent.blob_offset.checked_add(blob_len);
    if end.is_none_or(|end| end > blob.raw_len) {
        return Err(fail(format!(
            "{blob_len} bytes at {} are outside a blob of {}",
            extent.blob_offset, blob.raw_len
        )));
    }
    Ok(())
}

/// For each index `i`, how many adjacent extent pairs before `i` are not contiguous. Equal values
/// at `a` and `b` mean extents `a..=b` follow each other with no gap, overlap or disorder, which is
/// what makes a shared extent range cost O(1) per member.
fn extent_breaks(extents: &[ExtentRecord]) -> Vec<u64> {
    let mut breaks = Vec::with_capacity(extents.len());
    let mut count = 0;
    if !extents.is_empty() {
        breaks.push(count);
    }
    for pair in extents.windows(2) {
        if pair[0].virt_offset.checked_add(pair[0].virt_len) != Some(pair[1].virt_offset) {
            count += 1;
        }
        breaks.push(count);
    }
    breaks
}

/// What a member needs to know about any run of tensors.
#[derive(Debug)]
struct TensorSpans {
    /// Largest `virt_offset + virt_len` of a run.
    end: RangeMax<u64>,
    /// For each tensor, 1 + the index of the closest earlier tensor of the same name, else 0.
    /// A run starting at `a` holds a duplicate exactly if its maximum is over `a`.
    same_name: RangeMax<usize>,
}

/// Per-record rules of invariants 3 and 8 for every tensor. Hashes each name once, so a run's
/// uniqueness is known without walking it per member.
fn check_tensors(index: &Index) -> Result<TensorSpans, Error> {
    let tensors = index.tensors();
    let mut ends = Vec::with_capacity(tensors.len());
    let mut same_name = Vec::with_capacity(tensors.len());
    let mut last_seen: HashMap<&str, usize> = HashMap::new();
    for (id, tensor) in tensors.iter().enumerate() {
        ends.push(check_tensor(index, tensor, id)?);
        let previous = last_seen.insert(index.string(tensor.name)?, id + 1);
        same_name.push(previous.unwrap_or(0));
    }
    Ok(TensorSpans { end: RangeMax::new(&ends), same_name: RangeMax::new(&same_name) })
}

/// Checks one tensor and returns the end of its virtual range. `id` is only for messages.
fn check_tensor(index: &Index, tensor: &TensorRecord, id: usize) -> Result<u64, Error> {
    if tensor.ndim > MAX_NDIM {
        return Err(malformed(TENS.tag, format!("tensor {id}: ndim is {}", tensor.ndim)));
    }
    index.tensor_shape(tensor)?;
    tensor
        .virt_offset
        .checked_add(tensor.virt_len)
        .ok_or_else(|| malformed(TENS.tag, format!("tensor {id}: the virtual range overflows")))
}

/// Invariants 3, 6 and 8 for every member.
fn check_members(index: &Index, tensors: &TensorSpans) -> Result<(), Error> {
    let breaks = extent_breaks(index.extents());
    let mut names = HashSet::with_capacity(index.members().len());
    for (id, member) in index.members().iter().enumerate() {
        if !names.insert(index.string(member.name)?) {
            return Err(malformed(MEMB.tag, format!("member {id}: the name is not unique")));
        }
        check_member_extents(index, member, id, &breaks)?;
        check_member_tensors(index, member, id, tensors)?;
    }
    Ok(())
}

/// Invariant 6: the extents of `member` partition `0..file_len`. `id` is only for messages.
fn check_member_extents(
    index: &Index,
    member: &MemberRecord,
    id: usize,
    breaks: &[u64],
) -> Result<(), Error> {
    let extents = index.extents();
    let count = u64::from(member.extent_count);
    let range = checked_range(member.first_extent, count, extents.len(), MEMB.tag)?;
    let Some(last) = range.end.checked_sub(1).filter(|_| !range.is_empty()) else {
        if member.file_len != 0 {
            return Err(malformed(MEMB.tag, format!("member {id}: no extents for its bytes")));
        }
        return Ok(());
    };
    let start = extents[range.start].virt_offset;
    let end = extents[last].virt_offset.checked_add(extents[last].virt_len);
    if start != 0 || breaks[range.start] != breaks[last] || end != Some(member.file_len) {
        return Err(malformed(
            MEMB.tag,
            format!("member {id}: extents do not partition 0..{}", member.file_len),
        ));
    }
    Ok(())
}

/// Invariants 3 and 8 for the tensors of `member`. `id` is only for messages.
fn check_member_tensors(
    index: &Index,
    member: &MemberRecord,
    id: usize,
    tensors: &TensorSpans,
) -> Result<(), Error> {
    let (first, count) = (u64::from(member.first_tensor), u64::from(member.tensor_count));
    let range = checked_range(first, count, index.tensors().len(), MEMB.tag)?;
    if tensors.end.max(range.clone()).is_some_and(|end| end > member.file_len) {
        return Err(malformed(
            MEMB.tag,
            format!("member {id}: a tensor ends past {}", member.file_len),
        ));
    }
    if tensors.same_name.max(range.clone()).is_some_and(|previous| previous > range.start) {
        return Err(malformed(MEMB.tag, format!("member {id}: tensor names are not unique")));
    }
    Ok(())
}

#[cfg(test)]
#[path = "validate_tests.rs"]
mod tests;
