//! File: edit/append.rs
//! Purpose: the `Append` strategy: a new generation written after the end of the container, spec
//! invariant 12 (b).
//!
//! Key functions: [`run`].
//!
//! Implementation notes: the sink writes through a clone of the handle the source was opened and
//! validated with (`Opened::writer`), so no second open of the path can pick up another file; the
//! verification of the commit reads through that handle too. The builder is seeded with the
//! source's storage records, so blob ids and
//! chunk offsets of the new index are the source's, kept members are carried over with no byte
//! copied, and a kept member serves as a delta base through `BaseCatalog::from_member` and
//! `DeltaSpec::Member`. Nothing below the old length `L` is written, the header included; the edit
//! refused before this point what would need it (a delta blob while required bit 0 is clear). Every
//! failure ends in `PackBuilder::fail` or the commit's own cut back, which restore the source
//! byte for byte, or report `Error::TrailingBytes`.

use std::sync::Arc;

use crate::error::Error;
use crate::read::Container;
use crate::write::{
    BaseCatalog, Commit, DeltaSpec, EncodeOptions, EncodeReport, PackBuilder, Seed, Sink,
};

use super::plan::{Base, EditPlan, Step};
use super::source::MemberSource;
use super::{EditOptions, Hooks, Opened};

/// Appends the result of `plan` to the source `opened` as generation `generation`, through the
/// handle `opened` read it with ([`Opened::writer`]): what is appended to is the file the seed
/// describes, whatever the path names by now.
///
/// # Errors
/// - [`Error::InputChanged`] if the file's length differs from the one it had at open.
/// - [`Error::EditRefused`] if `opened` holds no handle to write, a caller's bug.
/// - Every error of the adds and of the append commit, [`Error::TrailingBytes`] among them.
pub(super) fn run(
    mut opened: Opened,
    plan: &EditPlan,
    options: &EditOptions,
    generation: u64,
    hooks: Hooks,
) -> Result<EncodeReport, Error> {
    let seed = Seed::of(&opened.container, opened.header)?;
    let writer = opened.writer.take().ok_or_else(|| Error::EditRefused {
        detail: "the container was not opened for an append".to_owned(),
    })?;
    hooks.before_sink(&opened.path);
    let mut sink = Sink::append_to(writer, &opened.path)?;
    // The frontier must be the end of the container that was validated, or the seed describes
    // another state of the file (written through another handle). Nothing is written yet, so the
    // sink's own cut back changes nothing.
    if sink.position() != opened.len {
        tracing::error!(
            path = %opened.path.display(),
            opened_len = opened.len,
            len = sink.position(),
            "the container changed between its open and the append"
        );
        sink.abandon()?;
        return Err(Error::InputChanged { path: opened.path });
    }
    hooks.arm_sink(&mut sink);
    let encode = EncodeOptions { mode: options.mode, threads: options.threads };
    let mut builder = PackBuilder::with_sink(sink, Some(seed), &encode, &opened.path)?;
    hooks.arm_builder(&mut builder);
    if let Err(error) = fill(&mut builder, &opened.container, plan) {
        return Err(builder.fail(error));
    }
    drop(opened);
    builder.finish_with(Commit::Append, generation)
}

/// Adds every member of the result to `builder`, in result order.
fn fill(builder: &mut PackBuilder, container: &Container, plan: &EditPlan) -> Result<(), Error> {
    for step in &plan.steps {
        match step {
            Step::Keep { name } => {
                builder.carry_member(name)?;
            }
            Step::File { name, path, delta_base: None, .. } => {
                builder.add_from_file(path, name, step.name_rule(), None)?;
            }
            Step::File { name, path, delta_base: Some(Base::New(base)), .. } => {
                let delta = DeltaSpec::Added(base);
                builder.add_from_file(path, name, step.name_rule(), Some(&delta))?;
            }
            Step::File { name, path, delta_base: Some(Base::Kept(base)), .. } => {
                // Its blob ids are the source's, which are this pack's: the builder is seeded.
                let member = container.member(base)?;
                let catalog = BaseCatalog::from_member(container.index(), member.record())?;
                let delta = DeltaSpec::Member {
                    name: base,
                    source: Arc::new(MemberSource::new(member)),
                    catalog: &catalog,
                };
                builder.add_from_file(path, name, step.name_rule(), Some(&delta))?;
            }
        }
    }
    Ok(())
}
