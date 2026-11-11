//! File: edit/inplace.rs
//! Purpose: the `InPlace` strategy: the container rewritten in its own file, with no temporary file
//! and no extra space, its kept chunks moved towards the header.
//!
//! Main responsibilities:
//! - the planner ([`plan`], pure over the validated index; [`plan_moves`], pure over byte ranges):
//!   which blobs stay, where each kept chunk goes, and the moves that put it there;
//! - [`run`]: the moves, the new files after them, and the in-place commit, through one sink over
//!   the handle the source was opened and validated with.
//!
//! Key structures: [`Move`], [`MovePlan`], [`InPlacePlan`].
//!
//! Key functions: [`plan`], [`plan_moves`], [`run`].
//!
//! Implementation notes: the proof that no kept byte is overwritten before it is copied.
//! - Kept ranges are the chunks of every blob a kept member reaches, with the base of every such
//!   delta blob (`reached_blobs`). Identical chunk records collapse to one range (`live_ranges`);
//!   validation proves ranges identical or disjoint (spec invariant 4) and inside `[64, I)`, `I`
//!   the source's index offset, and the planner checks both again rather than trust them.
//! - P1: ranges sorted by source offset get `dst_i = 64 + sum of the lengths before`; disjoint
//!   ranges in `[64, I)` give `dst_i <= src_i`, checked per range.
//! - P2: move `i` writes `[dst_i, dst_i + len_i)`; every later range starts at
//!   `src_j >= src_i + len_i >= dst_i + len_i`, so no write touches a byte not yet read.
//! - P3: inside one move (which may overlap itself) `Sink::copy_within` copies forward, a piece
//!   read whole before it is written, so its writes never pass its reads: a forward memmove.
//! - P4: new files are written from `compacted_end`, after the last move, over bytes no kept chunk
//!   needs any more; a rollback never truncates and never goes below `compacted_end`
//!   (`Sink::rollback`), and the file is cut once, at the end.
//!
//! Adjacent ranges whose destinations are adjacent too merge into one move, copied in pieces of at
//! most [`MOVE_BUFFER`] bytes: the one buffer this strategy allocates beyond the writer's own.
//! Blob ids: the kept blobs keep their old order (`Seed::in_place`), so a base keeps a lower id
//! than its delta blob (invariant 13), whatever the byte order of the moves. A delta base that is a
//! kept member is refused before any write (`edit`): ingest would read it through the source
//! container, whose chunk offsets are stale once data moved. The first write of the sink zeroes the
//! source footer, so a torn rewrite never ends in a generation whose chunks moved.

use std::collections::HashMap;

use crate::error::Error;
use crate::format::{ChunkRecord, HEADER_LEN, Index, MemberRecord};
use crate::write::{Commit, DeltaSpec, EncodeOptions, EncodeReport, PackBuilder, Seed, Sink};

use super::plan::{Base, EditPlan, Step};
use super::{EditOptions, Hooks, Opened, live_ranges, reached_blobs};

/// Largest piece a move reads and writes at once, and so the size of the one move buffer: 8 MiB,
/// about one chunk of the largest `chunk_size` (`MAX_CHUNK_SIZE`) with its prologue, and bounded
/// however large the container. A move longer than this is copied in pieces.
pub(super) const MOVE_BUFFER: u64 = 8 << 20;

/// One move of the plan: `len` bytes from `src` to `dst`, `dst <= src`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Move {
    /// Where the bytes are in the source.
    pub(super) src: u64,
    /// Where they go: the end of the moves before.
    pub(super) dst: u64,
    /// How many.
    pub(super) len: u64,
}

/// Where the kept stored ranges go.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct MovePlan {
    /// The moves, in increasing `src` and `dst` order; a move with `src == dst` reads and writes
    /// nothing.
    pub(super) moves: Vec<Move>,
    /// Each kept range `(file_offset, stored_len)` to its new offset.
    pub(super) offsets: HashMap<(u64, u32), u64>,
    /// End of the moved data, where new data starts.
    pub(super) compacted_end: u64,
}

impl MovePlan {
    /// The new offset of the chunk `record`.
    ///
    /// # Errors
    /// [`Error::EditRefused`] if the record is not a kept range, a planner's bug.
    pub(super) fn offset_of(&self, record: &ChunkRecord) -> Result<u64, Error> {
        let key = (record.file_offset, record.stored_len);
        self.offsets.get(&key).copied().ok_or_else(|| Error::EditRefused {
            detail: format!(
                "the in-place plan has no place for the chunk of {} bytes at {}",
                record.stored_len, record.file_offset
            ),
        })
    }

    /// The largest number of bytes a move copies, 0 if nothing moves.
    fn largest_copy(&self) -> u64 {
        let copies = self.moves.iter().filter(|mv| mv.src != mv.dst);
        copies.map(|mv| mv.len).max().unwrap_or(0)
    }
}

/// The plan of one in-place rewrite.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct InPlacePlan {
    /// Old ids of the blobs that stay, increasing.
    pub(super) blobs: Vec<u32>,
    /// Where their chunks go.
    pub(super) moves: MovePlan,
}

/// The refusal of a plan that cannot be carried out safely.
fn unsafe_plan(detail: &str) -> Error {
    Error::EditRefused { detail: format!("cannot rewrite in place: {detail}") }
}

/// Plans the moves of the stored `ranges` `(file_offset, stored_len)`, sorted and each once, of a
/// container whose data region ends at `data_end` (its index offset). See the module notes for
/// P1 and P2, which this checks range by range.
///
/// # Errors
/// [`Error::EditRefused`] if a range is not sorted after the one before, overlaps it, lies outside
/// `[64, data_end)`, would move backwards, or an offset overflows: what validation rules out, so
/// a refusal means the plan's own proof failed, and nothing has been written.
pub(super) fn plan_moves(ranges: &[(u64, u32)], data_end: u64) -> Result<MovePlan, Error> {
    let mut moves: Vec<Move> = Vec::new();
    let mut offsets = HashMap::with_capacity(ranges.len());
    let mut dst = HEADER_LEN;
    // End of the previous range in the source: the next must start at or after it.
    let mut previous_end = HEADER_LEN;
    for &(src, stored_len) in ranges {
        let len = u64::from(stored_len);
        let end = src.checked_add(len).ok_or_else(|| unsafe_plan(&format!("{src} + {len} overflows")))?;
        if src < previous_end || end > data_end {
            return Err(unsafe_plan(&format!(
                "{len} bytes at {src} overlap the range before or leave the data region 64..{data_end}"
            )));
        }
        // P1: disjoint sorted ranges after the header leave no more than `src - 64` bytes before.
        if dst > src {
            return Err(unsafe_plan(&format!("{len} bytes at {src} would move forward to {dst}")));
        }
        offsets.insert((src, stored_len), dst);
        match moves.last_mut() {
            // Adjacent in the source and at the destination: one copy.
            Some(last) if last.src + last.len == src && last.dst + last.len == dst => last.len += len,
            _ if len == 0 => {}
            _ => moves.push(Move { src, dst, len }),
        }
        previous_end = end;
        dst += len;
    }
    Ok(MovePlan { moves, offsets, compacted_end: dst })
}

/// Plans the in-place rewrite of the container `index`, whose index starts at `index_offset`, that
/// keeps the members `kept` (records of `index`).
///
/// # Errors
/// Every error of [`plan_moves`], and [`Error::MalformedIndex`] if a record points outside its
/// section, which validation rules out.
pub(super) fn plan(
    index: &Index,
    kept: &[MemberRecord],
    index_offset: u64,
) -> Result<InPlacePlan, Error> {
    let reached = reached_blobs(index, kept)?;
    let ranges = live_ranges(index, &reached)?;
    let moves = plan_moves(&ranges, index_offset)?;
    let blobs = (0_u32..)
        .zip(&reached)
        .filter_map(|(id, reached)| reached.then_some(id))
        .collect();
    Ok(InPlacePlan { blobs, moves })
}

/// Rewrites the source `opened` in place as `plan` says, with `generation` in its footer, through
/// the handle `opened` read it with ([`Opened::writer`]).
///
/// Everything up to the sink is a dry phase: the plan, the seed, the check that the file is the
/// one validated. A failure there leaves the file as it was. From the sink's first write on, every
/// failure is [`Error::DamagedInPlace`], except a failed final cut ([`Error::TrailingBytes`]).
///
/// # Errors
/// - [`Error::InputChanged`] if the file's length differs from the one it had at open.
/// - [`Error::EditRefused`] if the plan cannot be carried out safely, if `opened` holds no handle
///   to write, or if a file names a kept member as its base, which `edit` refuses before.
/// - [`Error::DamagedInPlace`] wrapping any failure after the first write.
/// - [`Error::TrailingBytes`] if the result is complete but the file could not be cut to its end.
pub(super) fn run(
    mut opened: Opened,
    plan: &EditPlan,
    options: &EditOptions,
    generation: u64,
    hooks: Hooks,
) -> Result<EncodeReport, Error> {
    let index = opened.container.index();
    let mut kept = Vec::new();
    for step in &plan.steps {
        if let Step::Keep { name } = step {
            kept.push(*opened.container.member(name)?.record());
        }
    }
    let in_place = self::plan(index, &kept, opened.footer.index_offset)?;
    let moves = &in_place.moves;
    tracing::info!(
        path = %opened.path.display(),
        kept_blobs = in_place.blobs.len(),
        dropped_blobs = index.blobs().len() - in_place.blobs.len(),
        moves = moves.moves.len(),
        compacted_end = moves.compacted_end,
        "planned the in-place rewrite"
    );
    let seed = Seed::in_place(&opened.container, opened.header, &in_place.blobs, |chunk| {
        moves.offset_of(chunk)
    })?;
    let writer = opened.writer.take().ok_or_else(|| Error::EditRefused {
        detail: "the container was not opened for an in-place rewrite".to_owned(),
    })?;
    let len = writer.metadata().map_err(|source| Error::Io {
        op: "metadata",
        path: opened.path.clone(),
        source,
    })?;
    if len.len() != opened.len {
        tracing::error!(
            path = %opened.path.display(),
            opened_len = opened.len,
            len = len.len(),
            "the container changed between its open and the in-place rewrite"
        );
        return Err(Error::InputChanged { path: opened.path });
    }
    let mut sink = Sink::in_place(writer, &opened.path)?;
    hooks.arm_sink(&mut sink);
    let encode = EncodeOptions { mode: options.mode, threads: options.threads };
    let mut builder = PackBuilder::with_sink(sink, Some(seed), &encode, &opened.path)?;
    hooks.arm_builder(&mut builder);
    if let Err(error) = move_kept(&mut builder, moves) {
        return Err(builder.fail(error));
    }
    if let Err(error) = fill(&mut builder, plan) {
        return Err(builder.fail(error));
    }
    // The source container's chunk offsets are stale; nothing may read through it any more.
    drop(opened);
    builder.finish_with(Commit::InPlace, generation)
}

/// Carries out the moves of `moves` in order, through one buffer of at most [`MOVE_BUFFER`] bytes,
/// and checks that the data ends where the plan says.
fn move_kept(builder: &mut PackBuilder, moves: &MovePlan) -> Result<(), Error> {
    let size = usize::try_from(moves.largest_copy().min(MOVE_BUFFER))
        .map_err(|_| unsafe_plan("the move buffer does not fit memory"))?;
    let mut buffer = vec![0_u8; size];
    for mv in &moves.moves {
        builder.move_data(mv.src, mv.dst, mv.len, &mut buffer)?;
    }
    if builder.position() != moves.compacted_end {
        return Err(unsafe_plan(&format!(
            "the moved data ends at {}, the plan says {}",
            builder.position(),
            moves.compacted_end
        )));
    }
    tracing::info!(
        moves = moves.moves.len(),
        compacted_end = moves.compacted_end,
        "moved the kept chunks"
    );
    Ok(())
}

/// Adds every member of the result to `builder`, in result order.
fn fill(builder: &mut PackBuilder, plan: &EditPlan) -> Result<(), Error> {
    for step in &plan.steps {
        match step {
            Step::Keep { name } => {
                builder.carry_member(name)?;
            }
            Step::File { name, path, delta_base: None, .. } => {
                builder.add_from_file(path, name, step.name_rule(), None)?;
            }
            Step::File { name, path, delta_base: Some(Base::New(base)), .. } => {
                // A file of this edit, read from its own file, never from the moved container.
                let delta = DeltaSpec::Added(base);
                builder.add_from_file(path, name, step.name_rule(), Some(&delta))?;
            }
            Step::File { delta_base: Some(Base::Kept(base)), .. } => {
                return Err(Error::EditRefused {
                    detail: format!(
                        "an in-place rewrite cannot store a delta against the kept member {base:?}"
                    ),
                });
            }
        }
    }
    Ok(())
}
