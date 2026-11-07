//! File: write/report.rs
//! Purpose: what an encode tells its caller: sizes per member, per dtype class and per saving layer.
//!
//! Main responsibilities: the public report types, [`ClassTable`], which adds them up and is the one
//! owner of what class an extent belongs to, and [`layer_report`], the same accounting for a finished
//! file.
//!
//! Key structures: [`LayerBytes`], [`ClassBytes`], [`MemberReport`], [`VerifyReport`], [`EncodeReport`],
//! [`ClassTable`].
//!
//! Key functions: [`layer_report`], [`unreferenced_table`].
//!
//! Implementation notes: the layers are ordered as the writer applies them: widening, then dedup, then
//! entropy. So a widened tensor that is also a duplicate shows half its bytes under each of the first
//! two. For every class `raw + overhead == widen_saved + dedup_saved + entropy_saved + stored`. A
//! delta blob's saving against its base is entropy: its stored bytes are those of the residual. A base
//! blob that no extent of the member reaches, copied along with a delta blob, is stored bytes that
//! are all overhead, in the class of the delta blob's extent. Blobs no extent reaches at all, left by a
//! failed add, are overhead too, in the class of their dtype hint, so a report accounts for the whole
//! data region.

use std::collections::BTreeMap;
use std::time::Duration;

use crate::chain::{BLOB_KIND_DELTA, View};
use crate::error::Error;
use crate::format::{BlobRecord, Index};
use crate::read::Container;
use crate::types::{Dtype, Profile};

use super::policy::SKELETON_DTYPE;

/// Bytes of one member or class, by what became of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LayerBytes {
    /// Source bytes.
    pub raw: u64,
    /// Bytes not stored because a widening view reproduces them.
    pub widen_saved: u64,
    /// Canonical bytes not stored because a blob with the same identity exists.
    pub dedup_saved: u64,
    /// Canonical bytes the streams of the new blobs saved.
    pub entropy_saved: u64,
    /// Bytes of new blobs that the stored form adds over their canonical bytes: chunk prologues, on
    /// blobs that did not compress.
    pub overhead: u64,
    /// Stored chunk bytes of the new blobs, the part of the data region these bytes account for.
    pub stored: u64,
}

impl LayerBytes {
    /// Adds `other` field by field.
    fn add(&mut self, other: &Self) {
        self.raw += other.raw;
        self.widen_saved += other.widen_saved;
        self.dedup_saved += other.dedup_saved;
        self.entropy_saved += other.entropy_saved;
        self.overhead += other.overhead;
        self.stored += other.stored;
    }
}

/// The bytes of one class of source data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassBytes {
    /// The dtype the plan gave the range; `None` for skeleton fragments and opaque files. A member
    /// copied from another container is classed by its blobs, widened extents as `F32`.
    pub dtype: Option<Dtype>,
    /// Where the bytes went.
    pub layers: LayerBytes,
}

/// What adding one member did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemberReport {
    /// Member name.
    pub name: String,
    /// Length of the member in bytes.
    pub len: u64,
    /// BLAKE3 of the member's source content.
    pub blake3: [u8; 32],
    /// Recognized source format.
    pub profile: Profile,
    /// Rows of the tensor table.
    pub tensor_count: u32,
    /// Extents of the member.
    pub extent_count: u32,
    /// Blobs this member added, including the base blobs a copy brought for its delta blobs.
    pub blobs_added: u32,
    /// Extents that went to a blob that already existed.
    pub blobs_reused: u32,
    /// Ranges streamed narrowed, found not exact, and streamed again. Their cost is a second pass.
    pub widen_restreams: u32,
    /// Blobs this member added that are delta blobs, stored against a base, spec section 7a. Counted
    /// in [`MemberReport::blobs_added`] too.
    pub delta_blobs: u32,
    /// Bytes this member compressed into the file. 0 for a member copied from another container.
    pub compressed_bytes: u64,
    /// Stored bytes copied from another container with no recompression.
    pub copied_bytes: u64,
    /// Where the bytes of this member went.
    pub layers: LayerBytes,
}

/// What the full decode before commit did. Its cost is one read of every member, whatever they share,
/// so it is linear in the sum of member lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifyReport {
    /// Bytes decoded and hashed: the sum of member lengths.
    pub bytes: u64,
    /// Wall time of the decode.
    pub elapsed: Duration,
}

/// What a finished container holds and what it cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeReport {
    /// Length of the output file in bytes.
    pub output_len: u64,
    /// Stored bytes of the index.
    pub index_stored_len: u64,
    /// The members, in file order.
    pub members: Vec<MemberReport>,
    /// The sum of [`EncodeReport::classes`].
    pub total: LayerBytes,
    /// Bytes per class, skeleton first, then by dtype id.
    pub classes: Vec<ClassBytes>,
    /// The verification before commit.
    pub verify: VerifyReport,
}

/// The report class of an extent: widened extents were `F32` in their source.
pub(super) fn class_of(blob: &BlobRecord, view: View) -> Option<Dtype> {
    match view {
        View::Identity if blob.dtype_hint == SKELETON_DTYPE => None,
        View::Identity => Some(blob.dtype_hint),
        View::F16ToF32 | View::Bf16ToF32 => Some(Dtype::F32),
    }
}

/// Bytes per class: `None` is the skeleton, otherwise a dtype wire id, so the order is stable.
#[derive(Debug, Default)]
pub(super) struct ClassTable(BTreeMap<Option<u16>, LayerBytes>);

impl ClassTable {
    /// The entry of `dtype`.
    fn entry(&mut self, dtype: Option<Dtype>) -> &mut LayerBytes {
        self.0.entry(dtype.map(Dtype::to_wire)).or_default()
    }

    /// Records source bytes that became `canonical` bytes of a blob. The difference is what a view
    /// saves, and with `duplicate` the canonical bytes are saved too.
    pub(super) fn add_source(
        &mut self,
        dtype: Option<Dtype>,
        raw: u64,
        canonical: u64,
        duplicate: bool,
    ) {
        let layers = self.entry(dtype);
        layers.raw += raw;
        layers.widen_saved += raw.saturating_sub(canonical);
        if duplicate {
            layers.dedup_saved += canonical;
        }
    }

    /// Records a new blob of `canonical` bytes stored in `stored` bytes.
    pub(super) fn add_stored(&mut self, dtype: Option<Dtype>, canonical: u64, stored: u64) {
        let layers = self.entry(dtype);
        layers.stored += stored;
        layers.entropy_saved += canonical.saturating_sub(stored);
        layers.overhead += stored.saturating_sub(canonical);
    }

    /// Records `stored` bytes of a blob that no extent reaches, a base that came in for a delta blob of
    /// class `dtype` or a blob a failed add left behind: stored, and all of it overhead.
    pub(super) fn add_base(&mut self, dtype: Option<Dtype>, stored: u64) {
        let layers = self.entry(dtype);
        layers.stored += stored;
        layers.overhead += stored;
    }

    /// Records an extent of `virt_len` member bytes over `blob`. `newly_stored` is the stored size of
    /// the blob if this is the extent that brought it in, and `None` if the blob existed already.
    pub(super) fn add_extent(
        &mut self,
        blob: &BlobRecord,
        view: View,
        virt_len: u64,
        newly_stored: Option<u64>,
    ) {
        let class = class_of(blob, view);
        if let Some(stored) = newly_stored {
            self.add_stored(class, blob.raw_len, stored);
        }
        let canonical = virt_len / view.length_factor();
        self.add_source(class, virt_len, canonical, newly_stored.is_none());
    }

    /// Adds every class of `other`.
    pub(super) fn merge(&mut self, other: &Self) {
        for (class, layers) in &other.0 {
            self.0.entry(*class).or_default().add(layers);
        }
    }

    /// The sum over the classes.
    #[must_use]
    pub(super) fn total(&self) -> LayerBytes {
        let mut total = LayerBytes::default();
        for layers in self.0.values() {
            total.add(layers);
        }
        total
    }

    /// The classes, skeleton first, then by dtype id.
    #[must_use]
    pub(super) fn classes(&self) -> Vec<ClassBytes> {
        self.0
            .iter()
            .map(|(class, layers)| ClassBytes {
                dtype: class.map(Dtype::from_wire),
                layers: *layers,
            })
            .collect()
    }
}

/// Reports where the bytes of an existing container went: the totals and the classes of
/// [`EncodeReport`], from its index alone, with no data read.
///
/// Extents are walked in member order, and the first one to reach a blob is the one that stored it, so
/// a container from this writer reports what its encode did, up to the class of a blob shared across
/// dtypes. The base of a delta blob that no earlier extent reached is counted when the delta blob is
/// first reached, as overhead, the way a member copy that brings a base counts it. Blobs that remain
/// unreached, such as those a failed add left behind, are counted last, as overhead, by
/// [`unreferenced_table`]. A blob only partly reached, which this writer never makes, breaks the
/// balance of [`LayerBytes`], and no more than that. Costs O(extents + chunks).
///
/// # Errors
/// Returns [`Error::MalformedIndex`] if a record is broken, which opening `container` rules out.
pub fn layer_report(container: &Container) -> Result<(LayerBytes, Vec<ClassBytes>), Error> {
    let index = container.index();
    let (mut table, reached) = walk_members(index)?;
    add_unreached(index, &reached, &mut table)?;
    Ok((table.total(), table.classes()))
}

/// The stored bytes of the blobs of `index` that no extent reaches and that are no base of a delta
/// blob an extent reaches: stored bytes that are all overhead, by the class of their dtype hint. What
/// the commit adds to the members' own accounting, so that an [`EncodeReport`] accounts for the whole
/// data region and agrees with [`layer_report`].
///
/// # Errors
/// As [`layer_report`].
pub(super) fn unreferenced_table(index: &Index) -> Result<ClassTable, Error> {
    let (_, reached) = walk_members(index)?;
    let mut table = ClassTable::default();
    add_unreached(index, &reached, &mut table)?;
    Ok(table)
}

/// Stored bytes of the chunks of `blob`.
fn stored_of(index: &Index, blob: &BlobRecord) -> Result<u64, Error> {
    let chunks = index.blob_chunks(blob)?;
    Ok(chunks.iter().map(|chunk| u64::from(chunk.stored_len)).sum())
}

/// The error of a blob id that is not in `BLOB`.
fn missing(id: u32) -> Error {
    Error::MalformedIndex { section: "BLOB", detail: format!("blob {id} does not exist") }
}

/// The accounting of the members of `index`, extent by extent, and which blobs it reached: every
/// blob an extent reaches, and the base of each such delta blob.
fn walk_members(index: &Index) -> Result<(ClassTable, Vec<bool>), Error> {
    let mut reached = vec![false; index.blobs().len()];
    let mut table = ClassTable::default();
    for member in index.members() {
        for extent in index.member_extents(member)? {
            let id = usize::try_from(extent.blob_id).ok().filter(|id| *id < reached.len());
            let id = id.ok_or_else(|| missing(extent.blob_id))?;
            let blob = &index.blobs()[id];
            let view = extent.view()?;
            let newly_stored = if std::mem::replace(&mut reached[id], true) {
                None
            } else {
                if blob.kind == BLOB_KIND_DELTA {
                    let base = index.delta_params(blob)?.base_blob;
                    let at = usize::try_from(base).ok().filter(|at| *at < reached.len());
                    let at = at.ok_or_else(|| missing(base))?;
                    if !std::mem::replace(&mut reached[at], true) {
                        table.add_base(class_of(blob, view), stored_of(index, &index.blobs()[at])?);
                    }
                }
                Some(stored_of(index, blob)?)
            };
            table.add_extent(blob, view, extent.virt_len, newly_stored);
        }
    }
    Ok((table, reached))
}

/// Adds the blobs `reached` leaves out to `table`, as stored bytes that are all overhead.
fn add_unreached(index: &Index, reached: &[bool], table: &mut ClassTable) -> Result<(), Error> {
    for (blob, _) in index.blobs().iter().zip(reached).filter(|(_, reached)| !**reached) {
        table.add_base(class_of(blob, View::Identity), stored_of(index, blob)?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ClassTable, LayerBytes};
    use crate::types::Dtype;

    fn balance(layers: &LayerBytes) -> (u64, u64) {
        (
            layers.raw + layers.overhead,
            layers.widen_saved + layers.dedup_saved + layers.entropy_saved + layers.stored,
        )
    }

    #[test]
    fn layers_balance_for_each_fate_of_a_range() {
        let mut table = ClassTable::default();
        // Widened, new, compressed.
        table.add_source(Some(Dtype::F32), 800, 400, false);
        table.add_stored(Some(Dtype::F32), 400, 300);
        // Widened and a duplicate.
        table.add_source(Some(Dtype::F32), 800, 400, true);
        // New and larger than its canonical bytes.
        table.add_source(None, 10, 10, false);
        table.add_stored(None, 10, 14);
        let total = table.total();
        assert_eq!(total.raw, 1610);
        assert_eq!(total.widen_saved, 800);
        assert_eq!(total.dedup_saved, 400);
        assert_eq!(total.entropy_saved, 100);
        assert_eq!(total.overhead, 4);
        assert_eq!(total.stored, 314);
        let (left, right) = balance(&total);
        assert_eq!(left, right);
    }

    #[test]
    fn a_base_brought_by_a_copy_is_stored_overhead_and_balances() {
        let mut table = ClassTable::default();
        table.add_source(Some(Dtype::Bf16), 100, 100, false);
        table.add_stored(Some(Dtype::Bf16), 100, 40);
        table.add_base(Some(Dtype::Bf16), 70);
        let total = table.total();
        assert_eq!(total.stored, 110);
        assert_eq!(total.overhead, 70);
        assert_eq!(total.entropy_saved, 60);
        let (left, right) = balance(&total);
        assert_eq!(left, right);
    }

    #[test]
    fn classes_are_ordered_with_the_skeleton_first() {
        let mut table = ClassTable::default();
        table.add_source(Some(Dtype::F32), 1, 1, false);
        table.add_source(Some(Dtype::Bf16), 2, 2, false);
        table.add_source(None, 3, 3, false);
        let classes: Vec<_> = table.classes().iter().map(|class| class.dtype).collect();
        assert_eq!(classes, [None, Some(Dtype::Bf16), Some(Dtype::F32)]);
    }

    #[test]
    fn merge_adds_per_class() {
        let mut first = ClassTable::default();
        first.add_source(None, 1, 1, false);
        let mut second = ClassTable::default();
        second.add_source(None, 2, 2, false);
        second.add_source(Some(Dtype::U8), 4, 4, true);
        first.merge(&second);
        assert_eq!(first.total().raw, 7);
        assert_eq!(first.classes().len(), 2);
    }
}
