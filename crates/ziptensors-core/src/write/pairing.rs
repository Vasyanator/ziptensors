//! File: write/pairing.rs
//! Purpose: which tensor of a base member a range of a new file may be stored against as a delta
//! blob, spec sections 7a and 11.
//!
//! Main responsibilities:
//! - map the tensors of a base member onto its extents, giving each the blob range that holds it,
//!   whether the member comes from a scanned file ([`BaseCatalog::from_plan`]) or from the records
//!   of a container ([`BaseCatalog::from_member`]);
//! - pair a tensor of the new file with one of those by name, length and dtype;
//! - hand the ingest one optional [`BaseRange`] per typed range of the new file.
//!
//! Key structures: [`BaseCatalog`], [`BaseTensor`], [`BaseRange`].
//!
//! Key functions: [`BaseCatalog::from_plan`], [`BaseCatalog::from_member`], [`BaseCatalog::bases_of`],
//! [`normalize_name`].
//!
//! Implementation notes: everything here is a hint. A wrong pair costs only the trial, since the delta
//! is chosen only where a sample stores smaller, and every member is decoded against its source hash
//! before commit. Names are compared after [`normalize_name`] because the same model is published
//! under different prefixes and with RMS norm scales named `scale` or `weight`. Only identity extents
//! qualify: a delta blob combines canonical bytes, and under a widening view those are not the
//! bytes of the file.

use std::borrow::Cow;
use std::collections::HashMap;

use crate::chain::View;
use crate::error::Error;
use crate::format::{ExtentRecord, Index, MemberRecord};
use crate::types::{Dtype, RangePlan, TensorInfo};

/// Prefixes that the same weights carry in one publication and not in another. At most one is
/// stripped, the first that matches, so the longer ones come first.
const NAME_PREFIXES: [&str; 3] = ["model.diffusion_model.", "model.", "module."];

/// Where the bytes of a base tensor lie, in the base file and in this pack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BaseRange {
    /// Start of the tensor in the base file. Under an identity extent these are its canonical bytes.
    pub(super) file_offset: u64,
    /// The blob that holds them.
    pub(super) blob: u32,
    /// Where they start in that blob.
    pub(super) blob_offset: u64,
}

/// A tensor of the base member with the blob range that holds it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BaseTensor {
    /// Name in the base file.
    pub(super) name: String,
    /// Length in bytes.
    pub(super) len: u64,
    /// Element type the base file declares.
    pub(super) dtype: Dtype,
    /// Where it lies.
    pub(super) range: BaseRange,
    /// Bytes from the tensor's start to the end of its extent: the most a delta range may use.
    pub(super) available: u64,
}

/// One tensor of a base member, as much of it as pairing needs, from a scan or from `TENS`.
#[derive(Debug, Clone, Copy)]
struct TensorRef<'a> {
    name: &'a str,
    /// Start in the member.
    offset: u64,
    /// Length in bytes.
    len: u64,
    dtype: Dtype,
}

/// The tensors of one base member, by name and by normalized name. Two catalogs are equal when they
/// pair every tensor alike.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct BaseCatalog {
    tensors: Vec<BaseTensor>,
    by_name: HashMap<String, usize>,
    /// `None` where two tensors normalize to the same name: such a name pairs with nothing.
    by_normal: HashMap<String, Option<usize>>,
}

impl BaseCatalog {
    /// Maps the tensors of `plan`, the scan of a base file, onto `extents`, the base member's extents
    /// in member order with blob ids of this pack, which partition the member. A tensor that does not
    /// start inside an identity extent is left out, since its file bytes are not the canonical bytes
    /// of one blob range.
    #[must_use]
    pub(super) fn from_plan(plan: &RangePlan, extents: &[ExtentRecord]) -> Self {
        let tensors = plan.tensors.iter().map(|tensor| TensorRef {
            name: &tensor.name,
            offset: tensor.offset,
            len: tensor.len,
            dtype: tensor.dtype,
        });
        Self::from_tensors(tensors, extents)
    }

    /// The catalog of `member` of a container, from its records: tensors from `TENS`, extents from
    /// `EXTN`. For a member this writer added from a file it equals [`BaseCatalog::from_plan`] of that
    /// file's scan, since the tensor table and the extents are stored as they were.
    ///
    /// `index` must be validated, as [`crate::read::Container`] does at open, so that the member's
    /// extents partition it in order; on any other input the catalog is still a mere hint. Blob ids
    /// are those of `index`: the catalog serves as a base of a pack whose blob ids are the same, a
    /// pack seeded with that index.
    ///
    /// # Errors
    /// [`Error::MalformedIndex`] if the member's extent or tensor ranges or a tensor name lie outside
    /// their sections.
    pub(crate) fn from_member(index: &Index, member: &MemberRecord) -> Result<Self, Error> {
        let extents = index.member_extents(member)?;
        let records = index.member_tensors(member)?;
        let mut tensors = Vec::with_capacity(records.len());
        for tensor in records {
            tensors.push(TensorRef {
                name: index.string(tensor.name)?,
                offset: tensor.virt_offset,
                len: tensor.virt_len,
                dtype: tensor.dtype,
            });
        }
        Ok(Self::from_tensors(tensors.into_iter(), extents))
    }

    /// The common part of both constructors: `tensors` in table order, since the first of two
    /// tensors with one name is the one that pairs by it.
    fn from_tensors<'t>(
        tensors: impl Iterator<Item = TensorRef<'t>>,
        extents: &[ExtentRecord],
    ) -> Self {
        let mut catalog = Self::default();
        for tensor in tensors {
            let Some(base) = base_tensor(tensor, extents) else {
                continue;
            };
            let at = catalog.tensors.len();
            catalog.by_name.entry(base.name.clone()).or_insert(at);
            let normal = normalize_name(&base.name).into_owned();
            catalog.by_normal.entry(normal).and_modify(|slot| *slot = None).or_insert(Some(at));
            catalog.tensors.push(base);
        }
        catalog
    }

    /// Number of tensors that can serve as a base.
    #[must_use]
    pub(super) fn len(&self) -> usize {
        self.tensors.len()
    }

    /// The base tensor of `target`: the same name, else the same normalized name, and in both cases
    /// the same length and dtype.
    #[must_use]
    pub(super) fn pair(&self, target: &TensorInfo) -> Option<&BaseTensor> {
        let at = match self.by_name.get(&target.name) {
            Some(at) => Some(*at),
            None => self.by_normal.get(normalize_name(&target.name).as_ref()).copied().flatten(),
        };
        let base = self.tensors.get(at?)?;
        (base.len == target.len && base.dtype == target.dtype).then_some(base)
    }

    /// One entry per range of `plan`, which must be normalized: the base range of the tensor that
    /// starts where the range starts, if it pairs and the base holds as many bytes as the range.
    #[must_use]
    pub(super) fn bases_of(&self, plan: &RangePlan) -> Vec<Option<BaseRange>> {
        let mut starts: HashMap<u64, &TensorInfo> = HashMap::with_capacity(plan.tensors.len());
        for tensor in &plan.tensors {
            // A range is cut from one tensor, so the longest tensor at an offset is the one it
            // belongs to; the others are views a parser listed twice.
            let entry = starts.entry(tensor.offset).or_insert(tensor);
            if tensor.len > entry.len {
                *entry = tensor;
            }
        }
        plan.ranges
            .iter()
            .map(|range| {
                let target = starts.get(&range.offset)?;
                let base = self.pair(target)?;
                (range.len <= base.available && base.dtype == range.hint).then_some(base.range)
            })
            .collect()
    }
}

/// `tensor` with the blob range of the identity extent it starts in, if there is one.
fn base_tensor(tensor: TensorRef<'_>, extents: &[ExtentRecord]) -> Option<BaseTensor> {
    // Extents are sorted by `virt_offset` and partition the member.
    let after = extents.partition_point(|extent| extent.virt_offset <= tensor.offset);
    let extent = extents.get(after.checked_sub(1)?)?;
    if extent.view().ok()? != View::Identity {
        return None;
    }
    let into = tensor.offset.checked_sub(extent.virt_offset)?;
    let available = extent.virt_len.checked_sub(into)?;
    Some(BaseTensor {
        name: tensor.name.to_owned(),
        len: tensor.len,
        dtype: tensor.dtype,
        range: BaseRange {
            file_offset: tensor.offset,
            blob: extent.blob_id,
            blob_offset: extent.blob_offset.checked_add(into)?,
        },
        available,
    })
}

/// `name` as it is compared across publications: without the first prefix of [`NAME_PREFIXES`] it
/// starts with, and with a last component `scale` after a component ending in `norm` read as
/// `weight`, which is what RMS norm scales are called elsewhere.
#[must_use]
pub(super) fn normalize_name(name: &str) -> Cow<'_, str> {
    let stripped = NAME_PREFIXES.iter().find_map(|prefix| name.strip_prefix(prefix)).unwrap_or(name);
    let norm_scale = stripped
        .strip_suffix(".scale")
        .filter(|head| head.rsplit('.').next().is_some_and(|part| part.ends_with("norm")));
    match norm_scale {
        Some(head) => Cow::Owned(format!("{head}.weight")),
        None => Cow::Borrowed(stripped),
    }
}

#[cfg(test)]
mod tests {
    use super::{BaseCatalog, BaseRange, normalize_name};
    use crate::chain::View;
    use crate::format::ExtentRecord;
    use crate::types::{Dtype, Profile, RangePlan, TensorInfo, TypedRange};

    fn tensor(name: &str, offset: u64, len: u64) -> TensorInfo {
        TensorInfo { name: name.to_owned(), offset, len, shape: vec![len / 2], dtype: Dtype::Bf16 }
    }

    /// A plan holding `tensors` and no ranges: all a catalog reads of a plan.
    fn plan_of(tensors: &[TensorInfo]) -> RangePlan {
        RangePlan { profile: Profile::Safetensors, ranges: Vec::new(), tensors: tensors.to_vec() }
    }

    fn extent(virt_offset: u64, virt_len: u64, blob_id: u32, view: View) -> ExtentRecord {
        ExtentRecord { virt_offset, virt_len, blob_id, view_id: view.to_wire(), blob_offset: 0 }
    }

    #[test]
    fn the_names_of_the_measured_pair_normalize_to_each_other() {
        let pairs = [
            (
                "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.weight",
                "double_blocks.0.img_attn.norm.key_norm.scale",
            ),
            (
                "model.diffusion_model.single_blocks.14.linear1.weight",
                "single_blocks.14.linear1.weight",
            ),
            ("model.diffusion_model.final_layer.norm.scale", "final_layer.norm.weight"),
        ];
        for (finetune, base) in pairs {
            assert_eq!(normalize_name(finetune), normalize_name(base), "{finetune} / {base}");
        }
    }

    #[test]
    fn normalization_strips_one_prefix_and_touches_only_norm_scales() {
        assert_eq!(normalize_name("module.model.x"), "model.x");
        assert_eq!(normalize_name("model.layers.0.weight"), "layers.0.weight");
        assert_eq!(normalize_name("layers.0.scale"), "layers.0.scale");
        assert_eq!(normalize_name("layers.0.norm.scale"), "layers.0.norm.weight");
        assert_eq!(normalize_name("scale"), "scale");
        assert_eq!(normalize_name("norm.scale"), "norm.weight");
    }

    #[test]
    fn a_pair_needs_the_same_length_and_dtype() {
        let tensors = [tensor("a.weight", 0, 8192), tensor("model.b.weight", 8192, 8192)];
        let extents = [extent(0, 8192, 0, View::Identity), extent(8192, 8192, 1, View::Identity)];
        let catalog = BaseCatalog::from_plan(&plan_of(&tensors), &extents);
        assert_eq!(catalog.len(), 2);
        let found = catalog.pair(&tensor("a.weight", 100, 8192)).expect("same name");
        assert_eq!(found.range, BaseRange { file_offset: 0, blob: 0, blob_offset: 0 });
        let found = catalog.pair(&tensor("b.weight", 0, 8192)).expect("normalized name");
        assert_eq!(found.range.blob, 1);
        assert!(catalog.pair(&tensor("a.weight", 0, 4096)).is_none(), "length differs");
        let wide = TensorInfo { dtype: Dtype::F16, ..tensor("a.weight", 0, 8192) };
        assert!(catalog.pair(&wide).is_none(), "dtype differs");
        assert!(catalog.pair(&tensor("c.weight", 0, 8192)).is_none(), "no such name");
    }

    #[test]
    fn widened_extents_and_ambiguous_names_pair_with_nothing() {
        let tensors = [
            tensor("model.x", 0, 8192),
            tensor("module.x", 8192, 8192),
            tensor("w", 16384, 8192),
        ];
        let extents = [
            extent(0, 8192, 0, View::Identity),
            extent(8192, 8192, 1, View::Identity),
            extent(16384, 16384, 2, View::Bf16ToF32),
        ];
        let catalog = BaseCatalog::from_plan(&plan_of(&tensors), &extents);
        assert_eq!(catalog.len(), 2, "the widened tensor is left out");
        assert!(catalog.pair(&tensor("x", 0, 8192)).is_none(), "two bases normalize to x");
        assert!(catalog.pair(&tensor("model.x", 0, 8192)).is_some(), "the exact name still pairs");
    }

    #[test]
    fn a_tensor_inside_an_extent_starts_at_its_offset_in_the_blob() {
        let tensors = [tensor("t", 100, 8192)];
        let extents =
            [ExtentRecord { blob_offset: 4, ..extent(0, 9000, 3, View::Identity) }];
        let catalog = BaseCatalog::from_plan(&plan_of(&tensors), &extents);
        let base = catalog.pair(&tensor("t", 0, 8192)).expect("pairs");
        assert_eq!(base.range, BaseRange { file_offset: 100, blob: 3, blob_offset: 104 });
        assert_eq!(base.available, 8900);
    }

    #[test]
    fn bases_of_follows_the_ranges_of_a_plan() {
        let tensors = [tensor("a", 0, 8192), tensor("b", 8192, 8192)];
        let extents = [extent(0, 8192, 0, View::Identity), extent(8192, 8192, 1, View::Identity)];
        let catalog = BaseCatalog::from_plan(&plan_of(&tensors), &extents);
        let plan = RangePlan {
            profile: Profile::Safetensors,
            ranges: vec![
                TypedRange { offset: 64, len: 8192, hint: Dtype::Bf16 },
                TypedRange { offset: 9000, len: 8192, hint: Dtype::Bf16 },
                TypedRange { offset: 20000, len: 8192, hint: Dtype::F16 },
            ],
            tensors: vec![
                tensor("model.b", 64, 8192),
                tensor("a", 9000, 8192),
                TensorInfo { dtype: Dtype::F16, ..tensor("a", 20000, 8192) },
                tensor("none", 30000, 8192),
            ],
        };
        let bases = catalog.bases_of(&plan);
        assert_eq!(bases.len(), 3);
        assert_eq!(bases[0].map(|base| base.blob), Some(1));
        assert_eq!(bases[1].map(|base| base.blob), Some(0));
        assert_eq!(bases[2], None, "dtype differs");
    }
}
