//! File: edit/mod.rs
//! Purpose: editing an existing container: adding, removing and replacing members, compaction and
//! repacking, by one of the commit strategies.
//!
//! Main responsibilities:
//! - the public API: [`edit`], [`EditOp`], [`Strategy`], [`EditOptions`], [`EditReport`];
//! - every check that must pass before a byte is written, whatever the strategy: the operation set
//!   (`plan.rs`), input files, strategy combinations;
//! - opening the source container once, and dispatch to `rewrite.rs`, `append.rs` or `inplace.rs`;
//! - which blobs and stored bytes the members of a container need (`reached_blobs`,
//!   `live_ranges`), for the dead-byte report and the in-place planner alike.
//!
//! Key structures: [`EditOp`], [`Strategy`], [`EditOptions`], [`EditReport`].
//!
//! Key functions: [`edit`].
//!
//! Implementation notes: the module owns no format logic; it orchestrates `write::PackBuilder` over
//! the source container. `compact` is `edit` with no operation. The source is opened once, through one
//! file handle that serves the container and the header and footer read here (and, under `Append` and
//! `InPlace`, the sink), so every decision is taken on the same bytes. The generation of the result is the source's plus one under every
//! strategy (spec section 9.3).

mod append;
mod inplace;
mod plan;
mod rewrite;
mod source;
#[cfg(test)]
mod tests;

use std::fs::{self, File};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::chain::{BLOB_KIND_DELTA, FEATURE_DELTA};
use crate::error::Error;
use crate::format::{
    FOOTER_LEN, Footer, HEADER_LEN, Header, Index, MemberRecord, parse_footer, parse_header,
};
use crate::io::{FileSource, RandomAccess, same_file};
use crate::read::{Container, OpenOptions};
use crate::types::Mode;
use crate::write::{EncodeReport, PackBuilder, Sink};

use plan::{EditPlan, Step};

/// One change to a container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditOp {
    /// Adds the file at `path` as a new member `name`, at the end of the member list.
    Add {
        /// The input file. It must be a regular file and not the container.
        path: PathBuf,
        /// The new member's name, which a restore must accept and no member of the container has.
        name: String,
        /// A member of the result to store this file against as delta blobs where that is smaller,
        /// spec section 7a: a kept member, or a file of this edit that comes before it.
        delta_base: Option<String>,
    },
    /// Removes the member `name`.
    Remove {
        /// A member of the container.
        name: String,
    },
    /// Replaces the content of member `name` with the file at `path`, in the member's slot.
    Replace {
        /// A member of the container.
        name: String,
        /// The input file. It must be a regular file and not the container.
        path: PathBuf,
        /// As for [`EditOp::Add`].
        delta_base: Option<String>,
    },
}

impl EditOp {
    /// The member this operation is about.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Add { name, .. } | Self::Remove { name } | Self::Replace { name, .. } => name,
        }
    }
}

/// How the edited container replaces the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Strategy {
    /// Writes the whole result to a temporary file beside the container, proves it, and renames it
    /// over the container (spec invariant 12 (a)). Needs room for a second copy; the source stays
    /// intact on any failure. Dead data of earlier edits is dropped.
    #[default]
    Rewrite,
    /// Writes only the new data, a new index and a new footer after the end of the container (spec
    /// invariant 12 (b)). Fast and needs room only for what is new; the old generation stays in the
    /// file as dead data. On failure the file is cut back to its old length; if that cut fails,
    /// [`Error::TrailingBytes`]. A reader that opens the file while the append runs fails at open.
    Append,
    /// Rewrites the container in its own file, with no temporary file and no extra space: the
    /// chunks of kept members move towards the header, new files follow, then a new index and
    /// footer, and the file is cut to its new length. Dead data of earlier edits is dropped.
    ///
    /// RISKY: an interruption after the first write (a crash, a power loss, a failed write) leaves a
    /// damaged container that may not be recoverable, reported as [`Error::DamagedInPlace`]; keep a
    /// backup or the original inputs. A reader that has the file open meanwhile reads wrong or
    /// corrupt bytes. Refused with `repack`, and with a delta base that is a kept member (a base
    /// added by the same edit is allowed). If the final cut fails, [`Error::TrailingBytes`]: the new
    /// container is complete before `valid_len`, and `recover` truncates the rest.
    InPlace,
}

/// How to edit a container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EditOptions {
    /// The commit strategy.
    pub strategy: Strategy,
    /// The mode of new files (`Add`, `Replace`), and of every member under `repack`.
    pub mode: Mode,
    /// Re-encodes every kept member with `mode`, deduplication and delta pairing redone over the
    /// whole result. A kept member that reached a delta blob is paired again with the first kept
    /// member before it that holds that delta's base blob under an identity extent, and stored plain
    /// if there is none. Only with [`Strategy::Rewrite`]: refused under the others.
    pub repack: bool,
    /// Compression and verification threads. 0 means one per logical CPU.
    pub threads: usize,
    /// How the source container is opened: its decode budget, limits and chunk verification.
    pub open: OpenOptions,
}

impl Default for EditOptions {
    /// [`Strategy::Rewrite`], [`Mode::Max`], no repack, all CPUs, default open options.
    fn default() -> Self {
        Self {
            strategy: Strategy::Rewrite,
            mode: Mode::Max,
            repack: false,
            threads: 0,
            open: OpenOptions::default(),
        }
    }
}

/// What an edit did.
#[derive(Debug, Clone, PartialEq)]
pub struct EditReport {
    /// Members of the source in the result as they were.
    pub kept: u32,
    /// Files added at the end.
    pub added: u32,
    /// Members removed.
    pub removed: u32,
    /// Members replaced in their slot.
    pub replaced: u32,
    /// The footer `generation` of the result: the source's plus one.
    pub generation: u64,
    /// The writer's report on the result. Under [`Strategy::Append`] `output_len` is the whole file,
    /// earlier generations included, and stored bytes no member reaches any more count as overhead.
    pub encode: EncodeReport,
    /// Bytes of the source file that no member of the source needed: blobs no member reaches,
    /// indexes and footers of earlier generations. What a [`Strategy::Rewrite`] of the source with
    /// no operation would drop.
    pub dead_bytes_before: u64,
}

/// Edits the container at `container` by `ops`, a set: each member name appears in at most one
/// operation. The result keeps the order of kept members, a `Replace` keeps its member's slot, and
/// `Add`s follow in operation order, whatever the strategy. Every check runs before any byte is
/// written: on an error from a check the file is untouched.
///
/// No operation is a compaction under [`Strategy::Rewrite`] and [`Strategy::InPlace`], and refused
/// under [`Strategy::Append`]. A `Container` the caller holds on the same file keeps reading the old
/// generation under [`Strategy::Append`]; under [`Strategy::Rewrite`] the caller should drop it
/// first, as Windows may refuse to rename over an open file; under [`Strategy::InPlace`] it reads
/// bytes that moved, and must be dropped first.
///
/// # Errors
/// - [`Error::EditRefused`] for an operation set or a combination that cannot be applied: a name in
///   two operations, an `Add` of an existing name, a delta base that is removed, is the member
///   itself, or comes after the member in the result; [`Strategy::Append`] with `repack`, with no
///   operation, or with a delta base while the container's required feature bit 0 is clear;
///   [`Strategy::InPlace`] with `repack` or with a delta base that is a kept member.
/// - [`Error::MemberNotFound`] if a `Remove` or `Replace` names no member, or a delta base names
///   nothing in the result.
/// - [`Error::UnsafeMemberName`] if a restore would refuse the name of an `Add`, the one name an
///   edit introduces. Kept and replaced members keep the names the valid source gave them, which
///   need pass only what the index requires (spec invariant 3), whatever a restore would say.
/// - [`Error::OutputIsInput`] if an input file is the container; [`Error::SymlinkInput`] if an input
///   is a symbolic link; [`Error::Io`] if an input is not a regular file or cannot be looked at.
/// - Every error of [`Container::open`] for the source, and every error of
///   [`PackBuilder::add_file`], [`PackBuilder::add_member_from`] and [`PackBuilder::finish`].
/// - [`Error::InputChanged`] if the container grew or shrank between its open and an append. An
///   append writes through the handle it read the container with, so a file swapped in at the
///   path meanwhile is never written.
/// - [`Error::TrailingBytes`] if an append failed and its partial tail could not be cut off: the
///   source is intact in the first `valid_len` bytes, and `recover` truncates the rest. Under
///   [`Strategy::InPlace`], if the rewritten file could not be cut to its new length: the result is
///   complete in the first `valid_len` bytes, and `recover` truncates the rest.
/// - [`Error::DamagedInPlace`] if an in-place rewrite failed after its first write, wrapping the
///   failure: the container is damaged. Every check above runs before that first write.
/// - [`Error::InputChanged`] under [`Strategy::InPlace`] too, if the container's length changed
///   since its open.
pub fn edit(container: &Path, ops: &[EditOp], options: &EditOptions) -> Result<EditReport, Error> {
    edit_with(container, ops, options, Hooks::default())
}

/// [`edit`] with the test seams of `hooks`.
fn edit_with(
    container: &Path,
    ops: &[EditOp],
    options: &EditOptions,
    hooks: Hooks,
) -> Result<EditReport, Error> {
    tracing::info!(
        path = %container.display(),
        ops = ops.len(),
        strategy = ?options.strategy,
        mode = ?options.mode,
        repack = options.repack,
        "editing container"
    );
    check_strategy(ops, options)?;
    let opened = Opened::open(container, &options.open, options.strategy)?;
    let names: Vec<&str> = opened.container.members().iter().map(|info| info.name.as_str()).collect();
    let plan = plan::resolve(&names, ops)?;
    drop(names);
    if options.strategy == Strategy::Append
        && plan.has_delta()
        && opened.header.required_features & FEATURE_DELTA == 0
    {
        return Err(Error::EditRefused {
            detail: "an append cannot add delta blobs to a container whose required feature bit 0 \
                     is clear; rewrite it instead"
                .to_owned(),
        });
    }
    if options.strategy == Strategy::InPlace && !plan.kept_bases.is_empty() {
        return Err(Error::EditRefused {
            detail: "an in-place rewrite cannot store a delta against a kept member, whose chunks \
                     move during the rewrite; name a file added by the same edit as the base, or \
                     rewrite the container instead"
                .to_owned(),
        });
    }
    check_inputs(container, &plan)?;
    let dead_bytes_before =
        dead_bytes(opened.container.index(), opened.len, opened.footer.index_stored_len)?;
    let generation = opened.footer.generation.saturating_add(1);
    let encode = match options.strategy {
        Strategy::Rewrite => rewrite::run(opened, &plan, options, generation, hooks)?,
        Strategy::Append => append::run(opened, &plan, options, generation, hooks)?,
        Strategy::InPlace => inplace::run(opened, &plan, options, generation, hooks)?,
    };
    tracing::info!(
        path = %container.display(),
        kept = plan.kept,
        added = plan.added,
        removed = plan.removed,
        replaced = plan.replaced,
        generation,
        len = encode.output_len,
        dead_bytes_before,
        "edited container"
    );
    Ok(EditReport {
        kept: plan.kept,
        added: plan.added,
        removed: plan.removed,
        replaced: plan.replaced,
        generation,
        encode,
        dead_bytes_before,
    })
}

/// Refuses strategy combinations that need no look at the container.
fn check_strategy(ops: &[EditOp], options: &EditOptions) -> Result<(), Error> {
    let refused = |detail: &str| Err(Error::EditRefused { detail: detail.to_owned() });
    match options.strategy {
        Strategy::Append if options.repack => {
            refused("a repack rewrites every member and cannot be appended")
        }
        Strategy::Append if ops.is_empty() => {
            refused("an append with no operation would only add a copy of the index")
        }
        Strategy::InPlace if options.repack => {
            refused("a repack reads every member while it writes, which an in-place rewrite cannot")
        }
        Strategy::Rewrite | Strategy::Append | Strategy::InPlace => Ok(()),
    }
}

/// Checks every input file of `plan`: a regular file, no symbolic link, not the container.
fn check_inputs(container: &Path, plan: &EditPlan) -> Result<(), Error> {
    for step in &plan.steps {
        let Step::File { path, .. } = step else {
            continue;
        };
        let meta = fs::symlink_metadata(path).map_err(|source| Error::Io {
            op: "metadata",
            path: path.clone(),
            source,
        })?;
        if meta.is_symlink() {
            return Err(Error::SymlinkInput { path: path.clone() });
        }
        if !meta.is_file() {
            return Err(Error::Io {
                op: "read input",
                path: path.clone(),
                source: std::io::Error::new(ErrorKind::InvalidInput, "not a regular file"),
            });
        }
        if same_file(path, container)? {
            return Err(Error::OutputIsInput { path: container.to_path_buf() });
        }
    }
    Ok(())
}

/// Bytes of a container of `file_len` bytes, whose stored index is `index_stored_len` bytes, that
/// no member needs: everything but the header, the footer, the current stored index and the chunks
/// of blobs a member reaches (with the bases of reached delta blobs). Chunk records that share bytes
/// (spec invariant 4) count once.
///
/// # Errors
/// [`Error::MalformedIndex`] if a record points outside its section, which validation rules out.
fn dead_bytes(index: &Index, file_len: u64, index_stored_len: u64) -> Result<u64, Error> {
    let reached = reached_blobs(index, index.members())?;
    let ranges = live_ranges(index, &reached)?;
    let live = ranges.iter().map(|(_, len)| u64::from(*len)).sum::<u64>();
    // Validation proves the chunks and the index lie between header and footer, so nothing below
    // saturates on a valid file.
    let needed = live.saturating_add(HEADER_LEN + FOOTER_LEN).saturating_add(index_stored_len);
    Ok(file_len.saturating_sub(needed))
}

/// The blobs of `index` that `members` need, by blob id: every blob an extent of theirs reaches,
/// and the base of every such delta blob (spec section 7a). The one definition of what a member
/// keeps alive, for the dead-byte report and the in-place planner.
///
/// # Errors
/// [`Error::MalformedIndex`] if a record points outside its section, which validation rules out.
fn reached_blobs<'m>(
    index: &Index,
    members: impl IntoIterator<Item = &'m MemberRecord>,
) -> Result<Vec<bool>, Error> {
    let blobs = index.blobs();
    let mut reached = vec![false; blobs.len()];
    let missing = |id: u32| Error::MalformedIndex {
        section: "BLOB",
        detail: format!("blob {id} does not exist"),
    };
    for member in members {
        for extent in index.member_extents(member)? {
            let at = usize::try_from(extent.blob_id).ok().filter(|at| *at < blobs.len());
            let at = at.ok_or_else(|| missing(extent.blob_id))?;
            reached[at] = true;
            if blobs[at].kind == BLOB_KIND_DELTA {
                let base = index.delta_params(&blobs[at])?.base_blob;
                let base_at = usize::try_from(base).ok().filter(|at| *at < blobs.len());
                reached[base_at.ok_or_else(|| missing(base))?] = true;
            }
        }
    }
    Ok(reached)
}

/// The stored ranges `(file_offset, stored_len)` of the chunks of the blobs `reached` marks, sorted,
/// each once: chunk records that share bytes are identical (spec invariant 4) and collapse.
///
/// # Errors
/// [`Error::MalformedIndex`] if a blob's chunks lie outside `CHNK`, which validation rules out.
fn live_ranges(index: &Index, reached: &[bool]) -> Result<Vec<(u64, u32)>, Error> {
    let mut ranges = Vec::new();
    for (blob, _) in index.blobs().iter().zip(reached).filter(|(_, reached)| **reached) {
        for chunk in index.blob_chunks(blob)? {
            ranges.push((chunk.file_offset, chunk.stored_len));
        }
    }
    ranges.sort_unstable();
    ranges.dedup();
    Ok(ranges)
}

/// The source container, opened once.
#[derive(Debug)]
struct Opened {
    /// Its path.
    path: PathBuf,
    /// The container, over the one handle all reads of the edit use.
    container: Container,
    /// Under [`Strategy::Append`] and [`Strategy::InPlace`], a clone of that handle, open for
    /// writing too: the sink writes the file the container was read from, with no second open of
    /// `path`, which may name another file by then. `None` under [`Strategy::Rewrite`], which never
    /// writes the source.
    writer: Option<File>,
    /// Its header, which an append keeps.
    header: Header,
    /// Its footer.
    footer: Footer,
    /// Its length at open.
    len: u64,
}

impl Opened {
    /// Opens the container at `path` with `options`, and reads its header and footer through the
    /// same handle. Under [`Strategy::Append`] and [`Strategy::InPlace`] the handle is open for
    /// reading and writing, and [`Opened::writer`] is a clone of it.
    ///
    /// # Errors
    /// Every error of [`Container::open`]; [`Error::Io`] if an append or an in-place rewrite cannot
    /// open the file for writing or clone the handle.
    fn open(path: &Path, options: &OpenOptions, strategy: Strategy) -> Result<Self, Error> {
        let io_error = |op, source| Error::Io { op, path: path.to_path_buf(), source };
        let (file, writer) = match strategy {
            Strategy::Rewrite => (FileSource::open(path)?, None),
            Strategy::Append | Strategy::InPlace => {
                let file = fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(path)
                    .map_err(|source| io_error("open", source))?;
                let writer = file.try_clone().map_err(|source| io_error("clone handle", source))?;
                (FileSource::from_file(file, path)?, Some(writer))
            }
        };
        let file = Arc::new(file);
        let len = file.len();
        let container = Container::from_source(Arc::clone(&file) as Arc<dyn RandomAccess>, options)?;
        // The container validated both, so these parses repeat checks that passed.
        let mut bytes = [0_u8; 64];
        file.read_exact_at(0, &mut bytes)?;
        let header = parse_header(&bytes)?;
        let footer_at = len.checked_sub(FOOTER_LEN).ok_or(Error::NotZiptensors)?;
        file.read_exact_at(footer_at, &mut bytes)?;
        let footer = parse_footer(&bytes)?;
        tracing::debug!(
            path = %path.display(),
            len,
            generation = footer.generation,
            required_features = header.required_features,
            "opened the container to edit"
        );
        Ok(Self { path: path.to_path_buf(), container, writer, header, footer, len })
    }
}

/// Test seams: crash injection into the append or in-place sink, corruption of the written file
/// before verification, a change at the container's path between the plan and the append sink.
/// Empty outside tests.
#[derive(Debug, Default, Clone, Copy)]
struct Hooks {
    #[cfg(test)]
    sink_fault: Option<crate::write::SinkFault>,
    #[cfg(test)]
    builder_fault: Option<fn(&Path)>,
    #[cfg(test)]
    before_sink: Option<fn(&Path)>,
}

impl Hooks {
    /// Runs the seam between the plan and the append sink on the container's path.
    #[cfg(test)]
    fn before_sink(self, path: &Path) {
        if let Some(seam) = self.before_sink {
            seam(path);
        }
    }

    /// Outside tests there is no seam.
    #[cfg(not(test))]
    #[expect(clippy::unused_self, reason = "the signature of the test version, which reads the seam")]
    fn before_sink(self, _: &Path) {}

    /// Arms the sink seam.
    #[cfg(test)]
    fn arm_sink(self, sink: &mut Sink) {
        if let Some(fault) = self.sink_fault {
            sink.set_fault(fault);
        }
    }

    /// Outside tests there is no seam.
    #[cfg(not(test))]
    #[expect(clippy::unused_self, reason = "the signature of the test version, which reads the seam")]
    fn arm_sink(self, _: &mut Sink) {}

    /// Arms the builder seam.
    #[cfg(test)]
    fn arm_builder(self, builder: &mut PackBuilder) {
        if let Some(fault) = self.builder_fault {
            builder.set_fault(fault);
        }
    }

    /// Outside tests there is no seam.
    #[cfg(not(test))]
    #[expect(clippy::unused_self, reason = "the signature of the test version, which reads the seam")]
    fn arm_builder(self, _: &mut PackBuilder) {}
}
