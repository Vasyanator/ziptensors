//! File: edit/rewrite.rs
//! Purpose: the `Rewrite` strategy: the whole result written to a temporary file beside the container
//! and renamed over it, spec invariant 12 (a); and `repack`, which re-encodes every kept member.
//!
//! Key functions: [`run`].
//!
//! Implementation notes: kept members are copied without recompression (`keep_member_from`), or, under
//! repack, decoded through `MemberSource` and added like any input, so that deduplication and delta
//! pairing see the whole result. Kept and replaced members keep their source names without the
//! restore rules (`NameRule::Kept`). A kept member that a new file names as its delta base is
//! registered with the builder as a base (`keep_member_from` with its source), so that every base of
//! a rewrite is named by `DeltaSpec::Added` in this pack's own blob ids. Every handle on the source (the `Opened`
//! container and each `MemberSource`) is released before the rename: the builder drops its own at
//! `finish`, and this module drops the rest before calling it.

use std::collections::HashMap;
use std::sync::Arc;

use crate::chain::{BLOB_KIND_DELTA, View};
use crate::error::Error;
use crate::format::{Index, MemberRecord};
use crate::read::Container;
use crate::write::{Commit, DeltaSpec, EncodeOptions, EncodeReport, PackBuilder, SourceStamp};

use super::plan::{Base, EditPlan, Step};
use super::source::MemberSource;
use super::{EditOptions, Hooks, Opened};

/// Writes the result of `plan` over the source `opened` to a temporary file and renames it over the
/// source, with `generation` in its footer. The source is intact on any failure.
///
/// # Errors
/// Every error of `PackBuilder::new`, of the adds, and of the commit.
pub(super) fn run(
    opened: Opened,
    plan: &EditPlan,
    options: &EditOptions,
    generation: u64,
    hooks: Hooks,
) -> Result<EncodeReport, Error> {
    let encode = EncodeOptions { mode: options.mode, threads: options.threads };
    let mut builder = PackBuilder::new(&opened.path, &encode)?;
    hooks.arm_builder(&mut builder);
    if let Err(error) = fill(&mut builder, &opened.container, plan, options.repack) {
        return Err(builder.fail(error));
    }
    // Windows refuses a rename over a file with an open handle; the builder drops its own handles at
    // finish.
    drop(opened);
    builder.finish_with(Commit::Replace, generation)
}

/// Adds every member of the result to `builder`, in result order.
fn fill(
    builder: &mut PackBuilder,
    container: &Container,
    plan: &EditPlan,
    repack: bool,
) -> Result<(), Error> {
    // Under repack: blob id of the source to the first kept member before, in result order, that
    // holds it under an identity extent, the base a kept delta member pairs with again.
    let mut owners: HashMap<u32, String> = HashMap::new();
    for step in &plan.steps {
        match step {
            Step::Keep { name } if repack => {
                let member = container.member(name)?;
                let record = *member.record();
                let base = repack_base(container.index(), &record, &owners)?;
                let delta = base.map(DeltaSpec::Added);
                let stamp = SourceStamp::Member { len: member.len(), blake3: member.blake3() };
                let source = Arc::new(MemberSource::new(member));
                builder.add_source(source, name, stamp, delta.as_ref())?;
                register_owner(container.index(), &record, name, &mut owners)?;
            }
            Step::Keep { name } if plan.kept_bases.contains(name) => {
                let source = Arc::new(MemberSource::new(container.member(name)?));
                builder.keep_member_from(container, name, Some(source))?;
            }
            Step::Keep { name } => {
                builder.keep_member_from(container, name, None)?;
            }
            Step::File { name, path, delta_base, .. } => {
                // Kept bases were registered above, so both kinds are members of this builder.
                let delta = delta_base.as_ref().map(|base| match base {
                    Base::Kept(base) | Base::New(base) => DeltaSpec::Added(base),
                });
                builder.add_from_file(path, name, step.name_rule(), delta.as_ref())?;
            }
        }
    }
    Ok(())
}

/// The base a kept member `record` is paired with again under repack: for its first extent over a
/// delta blob whose base blob has an owner in `owners`, that owner.
fn repack_base<'o>(
    index: &Index,
    record: &MemberRecord,
    owners: &'o HashMap<u32, String>,
) -> Result<Option<&'o str>, Error> {
    for extent in index.member_extents(record)? {
        let blob = usize::try_from(extent.blob_id).ok().and_then(|id| index.blobs().get(id));
        let blob = blob.ok_or_else(|| Error::MalformedIndex {
            section: "EXTN",
            detail: format!("blob {} does not exist", extent.blob_id),
        })?;
        if blob.kind != BLOB_KIND_DELTA {
            continue;
        }
        let base = index.delta_params(blob)?.base_blob;
        if let Some(owner) = owners.get(&base) {
            return Ok(Some(owner.as_str()));
        }
    }
    Ok(None)
}

/// Records `name` as the owner of every blob it holds under an identity extent that has no owner
/// yet: only identity extents hold a base's canonical bytes, spec section 7a.
fn register_owner(
    index: &Index,
    record: &MemberRecord,
    name: &str,
    owners: &mut HashMap<u32, String>,
) -> Result<(), Error> {
    for extent in index.member_extents(record)? {
        if extent.view()? == View::Identity {
            owners.entry(extent.blob_id).or_insert_with(|| name.to_owned());
        }
    }
    Ok(())
}
