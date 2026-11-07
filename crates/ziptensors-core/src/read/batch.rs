//! File: read/batch.rs
//! Purpose: how a read becomes I/O and decoding: chunk planning, coalescing, parallel decode.
//!
//! Main responsibilities:
//! - [`Segment`] and [`Segments`] cut a member range along its extents;
//! - `Planner` walks segments lazily, chunk by chunk, into batches of bounded stored size;
//! - `Batch` holds each distinct stored chunk once, reads them on the calling thread in file order,
//!   chunks that touch in the file as one read, then decodes them on the private pool;
//! - a job over a delta blob (spec section 7a) carries its base chunk, read in the same batch and
//!   decoded by the same job, just before the delta chunk.
//!
//! Key structures: [`Engine`], [`Segment`], [`Segments`], `Planner`, `Batch`, `StoredChunk`, `Job`,
//! `BaseJob`.
//!
//! Key functions: [`read_segments`].
//!
//! Implementation notes: planning is a walk and not a list, so a read of a huge member holds at most
//! one batch of plan. `dst` is cut with `split_at_mut`, one slice per job, and the stored bytes live
//! in a buffer owned by the call, so `stored` and `dst` are disjoint by the borrow rules and not by
//! an assertion. An element of a widening view can straddle two chunks only under an odd `chunk_size`,
//! which no writer policy produces; that one element is handled alone, on the calling thread, one
//! chunk at a time, so the hot path knows nothing of it and the memory bound holds. A delta blob
//! never has one: its chain block, and so its `chunk_size`, is even.
//!
//! A delta job admits both of its stored chunks to a batch at once or neither, so the first job of a
//! batch may hold two chunks: the batch bound grows by one chunk, and the job's scratch entry holds
//! the base's canonical bytes besides its own, which is the `(4 × threads + 1) × chunk` of spec
//! section 9.2. Jobs admit chunks in member order, so a delta batch admits base and delta chunks
//! alternately, while in the file each blob's chunks are contiguous. The buffer layout is therefore
//! made at execution, in file order: the base chunks of a batch become one read and its delta chunks
//! another. Two reads where one ends on the byte the next starts at would also cost a kernel
//! read-ahead window on the second.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use rayon::ThreadPool;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

use crate::chain::{BLOB_KIND_DELTA, Chain, View};
use crate::codec::{decode_chunk_into, decode_delta_chunk_into, locate_error, widen_into};
use crate::error::Error;
use crate::format::{BlobRecord, ChunkRecord, DeltaParams, ExtentRecord};
use crate::types::Verify;

use super::container::Shared;
use super::scratch::{Lease, Scratch, ScratchPool};

/// Largest prologue of a chunk: 8 streams of 4 bytes each. A batch counts it on top of the chunk size
/// so that `threads` incompressible chunks fit one batch.
const MAX_PROLOGUE_BYTES: usize = 32;
/// Jobs one batch holds per thread. Plan memory stays fixed even for a file of tiny chunks, whose
/// stored bytes alone would allow millions of jobs in a batch.
const JOBS_PER_THREAD: usize = 64;
/// Bytes of a narrow element of a widening view.
const NARROW_BYTES: u64 = 2;
/// Bytes of a wide element of a widening view.
const WIDE_BYTES: u64 = 4;
/// [`WIDE_BYTES`] as a slice length.
const WIDE_LEN: usize = 4;

/// Converts a value from the file to a length. Every supported target has a 64-bit `usize`, so this
/// fails only if a future target has not.
pub(super) fn to_usize(value: u64) -> Result<usize, Error> {
    usize::try_from(value).map_err(|_| broken(format!("{value} does not fit a length")))
}

/// The error for a state [`crate::format::validate`] rules out. It exists because arithmetic on file
/// values stays checked even where validation makes a failure impossible.
pub(super) fn broken(detail: impl Into<String>) -> Error {
    Error::MalformedIndex { section: "index", detail: detail.into() }
}

/// The decode side of a `Container`: its thread pool, its scratch and the limits of a batch.
#[derive(Debug)]
pub(super) struct Engine {
    /// `None` for one thread: everything then runs inline on the caller.
    pool: Option<ThreadPool>,
    scratch: ScratchPool,
    verify: Verify,
    /// Decode threads, at least 1: the number of jobs one batch decodes in parallel.
    pub(super) threads: usize,
    /// Most stored bytes of a batch: threads × (largest chunk + prologue).
    pub(super) batch_bytes: usize,
    /// Size of a run of member bytes read start to end: at least a batch, and room for one chunk
    /// widened, two chunks long, with a prologue. So that the run can end on a chunk boundary whatever
    /// the view, and the next run does not decode the same chunk again.
    pub(super) stream_bytes: usize,
    /// Most jobs of a batch.
    batch_jobs: usize,
}

impl Engine {
    /// Creates the engine of a container. `threads` is at least 1 and `largest_chunk` is the largest
    /// `chunk_size` in the file, which is what "chunk" means in the bound of spec section 9.2.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the threads of the pool cannot be started.
    pub(super) fn new(
        threads: usize,
        largest_chunk: u32,
        verify: Verify,
        path: &std::path::Path,
    ) -> Result<Self, Error> {
        let pool = (threads > 1)
            .then(|| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|i| format!("ziptensors-read-{i}"))
                    .build()
            })
            .transpose()
            .map_err(|source| Error::Io {
                op: "start read thread pool",
                path: path.to_path_buf(),
                source: std::io::Error::other(source),
            })?;
        let chunk_bytes = to_usize(u64::from(largest_chunk))?;
        let batch_bytes = threads.saturating_mul(chunk_bytes + MAX_PROLOGUE_BYTES);
        Ok(Self {
            pool,
            scratch: ScratchPool::new(threads),
            verify,
            threads,
            batch_bytes,
            stream_bytes: batch_bytes.max(chunk_bytes.saturating_mul(2) + MAX_PROLOGUE_BYTES),
            batch_jobs: threads.saturating_mul(JOBS_PER_THREAD),
        })
    }

    /// Whether this engine has no pool, so that everything runs on the caller and no read ever enters
    /// a `ThreadPool::install`.
    pub(super) fn is_inline(&self) -> bool {
        self.pool.is_none()
    }

    /// Scratch entries not in use.
    #[cfg(test)]
    pub(super) fn scratch_available(&self) -> usize {
        self.scratch.available()
    }

    /// The expected hash of a chunk under the verification policy.
    fn expected_hash(&self, place: &ChunkPlace) -> Option<u64> {
        match self.verify {
            Verify::Chunk => Some(place.xxh3),
            Verify::Off => None,
        }
    }
}

/// The chunks of one blob, resolved: enough to locate any of them without further lookups.
#[derive(Debug, Clone, Copy)]
struct ChunkSet<'s> {
    id: u32,
    record: &'s BlobRecord,
    chunks: &'s [ChunkRecord],
    chain: Chain,
}

impl<'s> ChunkSet<'s> {
    /// Resolves blob `id`, without its base.
    fn new(shared: &'s Shared, id: u32) -> Result<Self, Error> {
        let record = shared
            .index
            .blobs()
            .get(to_usize(u64::from(id))?)
            .ok_or_else(|| broken(format!("blob {id} does not exist")))?;
        let chain = shared
            .chains
            .get(to_usize(u64::from(record.chain_id))?)
            .ok_or_else(|| broken(format!("blob {id} has no chain")))?;
        let chunks = shared.index.blob_chunks(record)?;
        Ok(Self { id, record, chunks, chain: *chain })
    }

    /// The place of chunk `chunk` of this blob.
    fn place(&self, chunk: u32) -> Result<ChunkPlace, Error> {
        let missing = || broken(format!("blob {} has no chunk {chunk}", self.id));
        let record = self.chunks.get(to_usize(u64::from(chunk))?).ok_or_else(missing)?;
        let raw_len = self.record.chunk_raw_len(chunk).ok_or_else(missing)?;
        Ok(ChunkPlace {
            blob: self.id,
            chunk,
            file_offset: record.file_offset,
            stored_len: record.stored_len,
            // A raw length is at most `chunk_size`, a `u32`, or validation failed.
            raw_len: u32::try_from(raw_len).map_err(|_| missing())?,
            xxh3: record.xxh3_decoded,
        })
    }
}

/// The base of a delta blob: its chunks, and which of them pairs with a chunk of the delta blob.
#[derive(Debug, Clone, Copy)]
struct DeltaBase<'s> {
    params: DeltaParams,
    chunks: ChunkSet<'s>,
}

/// A blob with everything about it resolved, for walking its chunks without further lookups.
#[derive(Debug, Clone, Copy)]
pub(super) struct BlobView<'s> {
    own: ChunkSet<'s>,
    /// `Some` for a delta blob.
    base: Option<DeltaBase<'s>>,
}

impl<'s> BlobView<'s> {
    /// Resolves blob `id`, and its base if it is a delta blob.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] if a reference is broken, which validation rules out.
    pub(super) fn new(shared: &'s Shared, id: u32) -> Result<Self, Error> {
        let own = ChunkSet::new(shared, id)?;
        let base = if own.record.kind == BLOB_KIND_DELTA {
            // A bounds-checked decode of 16 bytes per segment, not per chunk.
            let params = shared.index.delta_params(own.record)?;
            Some(DeltaBase { params, chunks: ChunkSet::new(shared, params.base_blob)? })
        } else {
            None
        };
        Ok(Self { own, base })
    }

    /// `chunk_size` of this blob, in bytes.
    pub(super) fn chunk_size(&self) -> u64 {
        u64::from(self.own.record.chunk_size)
    }

    /// The chain of this blob.
    fn chain(&self) -> Chain {
        self.own.chain
    }

    /// The place of chunk `chunk` of this blob.
    fn place(&self, chunk: u32) -> Result<ChunkPlace, Error> {
        self.own.place(chunk)
    }

    /// For a delta blob, the base chunk that `place`, a chunk of this blob, is combined with, and
    /// the base's chain. `None` for any other blob.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] if the pairing of spec section 7a does not hold, which
    /// validation rules out.
    fn base_of(&self, place: &ChunkPlace) -> Result<Option<(ChunkPlace, Chain)>, Error> {
        let Some(base) = &self.base else {
            return Ok(None);
        };
        let unpaired = || {
            broken(format!("chunk {} of blob {} has no base chunk", place.chunk, place.blob))
        };
        let chunk = base.params.base_chunk(place.chunk, self.own.record.chunk_size);
        let base_place = base.chunks.place(chunk.ok_or_else(unpaired)?)?;
        if base_place.raw_len < place.raw_len {
            return Err(unpaired());
        }
        Ok(Some((base_place, base.chunks.chain)))
    }

    /// The chunk holding byte `position` of the blob's canonical bytes, and the offset of that byte
    /// in the chunk. Chunks follow from `chunk_size`, per blob, because a chunk record may be shared
    /// by blobs that cut their bytes differently.
    fn locate(&self, position: u64) -> Result<(ChunkPlace, u32), Error> {
        let chunk_size = self.chunk_size();
        let outside = || broken(format!("byte {position} is outside blob {}", self.own.id));
        let chunk = position.checked_div(chunk_size).ok_or_else(outside)?;
        let within = position.checked_rem(chunk_size).ok_or_else(outside)?;
        let place = self.place(u32::try_from(chunk).map_err(|_| outside())?)?;
        // `within` is below `chunk_size`, a `u32`.
        let within = u32::try_from(within).map_err(|_| outside())?;
        if within >= place.raw_len {
            return Err(outside());
        }
        Ok((place, within))
    }
}

/// One stored chunk, located: everything a job needs about it.
#[derive(Debug, Clone, Copy)]
struct ChunkPlace {
    blob: u32,
    /// Index within the blob.
    chunk: u32,
    file_offset: u64,
    stored_len: u32,
    /// Canonical bytes of this chunk.
    raw_len: u32,
    /// `CHNK.xxh3_decoded`.
    xxh3: u64,
}

impl ChunkPlace {
    /// What identifies the stored bytes of this place: two places with the same key are the same
    /// stored bytes. Chunk ranges are identical or disjoint.
    fn stored_key(&self) -> (u64, u32) {
        (self.file_offset, self.stored_len)
    }
}

/// What a job makes of the canonical bytes of its chunk.
#[derive(Debug, Clone, Copy)]
enum Output {
    /// The chunk is the output: decoded straight into the destination, so the codec's in-place route
    /// applies.
    Whole,
    /// A part of the chunk, from byte `skip`.
    Copy { skip: usize },
    /// Whole narrow elements from byte `skip`, `narrow_len` bytes, widened by `view`. The output
    /// starts at byte `head` of the first wide element.
    Widen { view: View, skip: usize, narrow_len: usize, head: usize },
}

/// The part of a member range served by one extent.
#[derive(Debug)]
pub(super) struct Segment<'s, 'd> {
    blob: BlobView<'s>,
    view: View,
    blob_offset: u64,
    /// Bytes of the extent already behind this segment, in member bytes.
    skip: u64,
    /// Destination still to be filled.
    out: &'d mut [u8],
}

/// The next step of a segment.
#[derive(Debug, Clone, Copy)]
enum Piece {
    /// One chunk serves the next `out_len` bytes.
    Single { place: ChunkPlace, output: Output, out_len: usize },
    /// The next element straddles two chunks.
    Straddle { first: ChunkPlace, second: ChunkPlace, view: View, head: usize, out_len: usize },
}

impl Segment<'_, '_> {
    /// Works out the next step. Nothing changes until [`Self::advance`].
    fn next_piece(&self) -> Result<Piece, Error> {
        match self.view {
            View::Identity => {
                let (place, within) = self.blob.locate(self.position(self.skip)?)?;
                let available = place.raw_len - within;
                let out_len = self.out.len().min(to_usize(u64::from(available))?);
                let output = if out_len == to_usize(u64::from(place.raw_len))? {
                    Output::Whole
                } else {
                    Output::Copy { skip: to_usize(u64::from(within))? }
                };
                Ok(Piece::Single { place, output, out_len })
            }
            View::F16ToF32 | View::Bf16ToF32 => self.next_widened(self.view),
        }
    }

    /// [`Self::next_piece`] for a widening view.
    fn next_widened(&self, view: View) -> Result<Piece, Error> {
        let head = to_usize(self.skip % WIDE_BYTES)?;
        let narrow_skip = (self.skip / WIDE_BYTES)
            .checked_mul(NARROW_BYTES)
            .ok_or_else(|| broken("a widened extent overflows"))?;
        let (place, within) = self.blob.locate(self.position(narrow_skip)?)?;
        let elements = u64::from(place.raw_len - within) / NARROW_BYTES;
        if elements == 0 {
            if self.blob.base.is_some() {
                // A delta blob's chunk size is a multiple of its even chain block.
                return Err(broken("an element straddles two chunks of a delta blob"));
            }
            // Validation put both bytes of the element inside the blob, so the next chunk exists.
            let next = place.chunk.checked_add(1).ok_or_else(|| broken("too many chunks"))?;
            let second = self.blob.place(next)?;
            let out_len = self.out.len().min(WIDE_LEN - head);
            return Ok(Piece::Straddle { first: place, second, view, head, out_len });
        }
        let available = to_usize(elements)?
            .checked_mul(WIDE_LEN)
            .and_then(|wide| wide.checked_sub(head))
            .ok_or_else(|| broken("a widened chunk overflows"))?;
        let out_len = self.out.len().min(available);
        let narrow_len = (head + out_len).div_ceil(WIDE_LEN) * to_usize(NARROW_BYTES)?;
        let skip = to_usize(u64::from(within))?;
        Ok(Piece::Single { place, output: Output::Widen { view, skip, narrow_len, head }, out_len })
    }

    /// Byte `narrow_skip` of the extent's blob range, as a position in the blob.
    fn position(&self, narrow_skip: u64) -> Result<u64, Error> {
        self.blob_offset
            .checked_add(narrow_skip)
            .ok_or_else(|| broken("an extent overflows its blob"))
    }

    /// File offset of the first chunk this segment needs, the key of the elevator order.
    pub(super) fn first_file_offset(&self) -> Result<u64, Error> {
        match self.next_piece()? {
            Piece::Single { place, .. } | Piece::Straddle { first: place, .. } => {
                Ok(place.file_offset)
            }
        }
    }
}

impl<'d> Segment<'_, 'd> {
    /// Takes the next `len` bytes of the destination. `len` is at most what is left.
    fn advance(&mut self, len: usize) -> &'d mut [u8] {
        let (now, later) = std::mem::take(&mut self.out).split_at_mut(len);
        self.out = later;
        // Lossless: `usize` is at most 64 bits on every supported target, and the sum stays within
        // the extent, whose end validation proved to fit `u64`.
        self.skip += len as u64;
        now
    }
}

/// Cuts a member range along extents. Yields a segment per extent touched, in member order.
#[derive(Debug)]
pub(super) struct Segments<'s, 'd> {
    shared: &'s Shared,
    extents: std::slice::Iter<'s, ExtentRecord>,
    offset: u64,
    out: &'d mut [u8],
}

impl<'s, 'd> Segments<'s, 'd> {
    /// Starts at `offset` of the member whose extents are `extents`. The range must lie inside the
    /// member, which [`super::Member`] has checked.
    pub(super) fn new(
        shared: &'s Shared,
        extents: &'s [ExtentRecord],
        offset: u64,
        out: &'d mut [u8],
    ) -> Self {
        // Extents are sorted and partition the member, so the last one starting at or before
        // `offset` holds it.
        let first = extents.partition_point(|extent| extent.virt_offset <= offset);
        let extents = extents[first.saturating_sub(1)..].iter();
        Self { shared, extents, offset, out }
    }

    /// Builds the next segment. `out` is not empty.
    fn next_segment(&mut self) -> Result<Segment<'s, 'd>, Error> {
        let extent = self.extents.next().ok_or_else(|| broken("extents end before the member"))?;
        let skip = self
            .offset
            .checked_sub(extent.virt_offset)
            .filter(|skip| *skip < extent.virt_len)
            .ok_or_else(|| broken("extents do not partition the member"))?;
        // A remainder too large for a length cannot limit a slice.
        let remaining = usize::try_from(extent.virt_len - skip).unwrap_or(usize::MAX);
        let len = self.out.len().min(remaining);
        let (now, later) = std::mem::take(&mut self.out).split_at_mut(len);
        self.out = later;
        // Lossless, as in `Segment::advance`.
        self.offset += now.len() as u64;
        Ok(Segment {
            blob: BlobView::new(self.shared, extent.blob_id)?,
            view: extent.view()?,
            blob_offset: extent.blob_offset,
            skip,
            out: now,
        })
    }
}

impl<'s, 'd> Iterator for Segments<'s, 'd> {
    type Item = Result<Segment<'s, 'd>, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.out.is_empty() {
            return None;
        }
        let segment = self.next_segment();
        if segment.is_err() {
            // An error ends the walk.
            self.out = &mut [];
        }
        Some(segment)
    }
}

/// One chunk to decode, with the part of the destination it fills.
#[derive(Debug)]
struct Job<'d> {
    place: ChunkPlace,
    chain: Chain,
    /// The stored chunk: an index into `Batch::chunks`.
    stored: usize,
    /// The base chunk, for a chunk of a delta blob.
    base: Option<BaseJob>,
    output: Output,
    out: &'d mut [u8],
}

/// The base chunk of a delta job: decoded first, into the job's scratch, and checked on its own.
#[derive(Debug, Clone, Copy)]
struct BaseJob {
    place: ChunkPlace,
    chain: Chain,
    /// Its stored bytes: an index into `Batch::chunks`.
    stored: usize,
}

/// One distinct stored chunk of a batch, which any number of its jobs may use.
#[derive(Debug, Clone, Copy)]
struct StoredChunk {
    file_offset: u64,
    len: usize,
    /// Where its bytes are in the batch buffer. Set by `Batch::lay_out`, once the batch is complete,
    /// because the layout follows file order and not the order chunks were admitted in.
    buffer_at: usize,
}

/// One `read_exact_at` of a batch: a stretch of stored chunks that touch in the file.
#[derive(Debug, Clone, Copy)]
struct Run {
    file_offset: u64,
    /// Where the bytes go in the batch buffer.
    buffer_at: usize,
    len: usize,
}

/// The chunks of one read-then-decode round.
#[derive(Debug)]
struct Batch<'d> {
    jobs: Vec<Job<'d>>,
    /// The distinct stored chunks the jobs use, in the order they were admitted.
    chunks: Vec<StoredChunk>,
    /// Index into `chunks` of each stored chunk, by [`ChunkPlace::stored_key`], so that a chunk any
    /// job of the batch uses is read once.
    slots: HashMap<(u64, u32), usize>,
    /// `(file_offset, index into chunks)` sorted, and the reads: made by `lay_out`, kept here only so
    /// that their capacity serves the next batch.
    order: Vec<(u64, usize)>,
    runs: Vec<Run>,
    /// Sum of the lengths of `chunks`: the size of the batch buffer.
    stored_bytes: usize,
}

impl<'d> Batch<'d> {
    fn new() -> Self {
        Self {
            jobs: Vec::new(),
            chunks: Vec::new(),
            slots: HashMap::new(),
            order: Vec::new(),
            runs: Vec::new(),
            stored_bytes: 0,
        }
    }

    fn clear(&mut self) {
        self.jobs.clear();
        self.chunks.clear();
        self.slots.clear();
        self.order.clear();
        self.runs.clear();
        self.stored_bytes = 0;
    }

    /// Makes room for the stored bytes of a job's chunk `place` and of its `base` chunk, if any, and
    /// returns their indices in `chunks`. Both or neither: `None` if the batch is full. A batch always
    /// takes its first job, whatever its size, so it may hold one chunk over `batch_bytes`.
    ///
    /// A chunk already in the batch, as two extents over one chunk or a plain extent over a delta's
    /// base chunk give, is held and read once, and only new bytes count against the bound; so is a
    /// base chunk with the same stored bytes as its delta.
    fn admit(
        &mut self,
        place: &ChunkPlace,
        base: Option<&ChunkPlace>,
        engine: &Engine,
    ) -> Result<Option<(usize, Option<usize>)>, Error> {
        if self.jobs.len() >= engine.batch_jobs {
            return Ok(None);
        }
        let own_key = place.stored_key();
        let base_new = base.filter(|base| {
            let key = base.stored_key();
            key != own_key && !self.slots.contains_key(&key)
        });
        let own_new = (!self.slots.contains_key(&own_key)).then_some(place);
        let mut needed = 0_usize;
        for new in own_new.into_iter().chain(base_new) {
            let len = to_usize(u64::from(new.stored_len))?;
            needed = needed.checked_add(len).ok_or_else(|| broken("batch overflow"))?;
        }
        let total = self.stored_bytes.checked_add(needed).ok_or_else(|| broken("batch overflow"))?;
        if !self.jobs.is_empty() && total > engine.batch_bytes {
            return Ok(None);
        }
        let own = self.hold(place)?;
        let base = base.map(|base| self.hold(base)).transpose()?;
        Ok(Some((own, base)))
    }

    /// Returns the index in `chunks` of the stored bytes of `place`, adding them to the batch if it
    /// does not hold them yet. The caller has checked the batch bound.
    fn hold(&mut self, place: &ChunkPlace) -> Result<usize, Error> {
        match self.slots.entry(place.stored_key()) {
            Entry::Occupied(held) => Ok(*held.get()),
            Entry::Vacant(free) => {
                let len = to_usize(u64::from(place.stored_len))?;
                let total =
                    self.stored_bytes.checked_add(len).ok_or_else(|| broken("batch overflow"))?;
                let slot = self.chunks.len();
                self.chunks.push(StoredChunk { file_offset: place.file_offset, len, buffer_at: 0 });
                self.stored_bytes = total;
                Ok(*free.insert(slot))
            }
        }
    }

    /// Places the chunks of the complete batch in the buffer in file order, one after another, and
    /// makes one run of every stretch of them that touch in the file. Runs come out sorted by file
    /// offset, and together they fill exactly `stored_bytes` of the buffer, so no run is longer than
    /// the batch.
    fn lay_out(&mut self) -> Result<(), Error> {
        self.order.clear();
        let keys = self.chunks.iter().enumerate().map(|(slot, chunk)| (chunk.file_offset, slot));
        self.order.extend(keys);
        self.order.sort_unstable();
        self.runs.clear();
        let mut at = 0_usize;
        for &(_, slot) in &self.order {
            let chunk = self.chunks.get_mut(slot).ok_or_else(|| broken("a batch lost a chunk"))?;
            chunk.buffer_at = at;
            let touching = self.runs.last_mut().filter(|run| {
                let end = u64::try_from(run.len).ok().and_then(|len| run.file_offset.checked_add(len));
                end == Some(chunk.file_offset)
            });
            match touching {
                Some(run) => run.len += chunk.len,
                None => {
                    self.runs.push(Run { file_offset: chunk.file_offset, buffer_at: at, len: chunk.len });
                }
            }
            at = at.checked_add(chunk.len).ok_or_else(|| broken("batch overflow"))?;
        }
        Ok(())
    }

    /// Reads the stored bytes on this thread, then decodes every job. `buffer` is the call's reusable
    /// batch buffer.
    ///
    /// Runs are read in order of file offset, whatever the order of the member, so that a batch of a
    /// fragmented member moves a disk one way, and chunks that touch in the file are one read. Jobs put
    /// bytes back in member order.
    fn execute(&mut self, shared: &Shared, buffer: &mut Vec<u8>) -> Result<(), Error> {
        self.lay_out()?;
        let stored = exact_buffer(buffer, self.stored_bytes);
        for run in &self.runs {
            let target = run
                .buffer_at
                .checked_add(run.len)
                .and_then(|end| stored.get_mut(run.buffer_at..end))
                .ok_or_else(|| broken("a run is outside its batch"))?;
            shared.source.read_exact_at(run.file_offset, target)?;
        }
        let stored: &[u8] = stored;
        let chunks: &[StoredChunk] = &self.chunks;
        let engine = &shared.engine;
        let decode = |job: &mut Job<'d>| job.run(stored, chunks, engine);
        match &engine.pool {
            // One job gains nothing from a trip through the pool.
            Some(pool) if self.jobs.len() > 1 => {
                pool.install(|| self.jobs.par_iter_mut().try_for_each(decode))
            }
            Some(_) | None => self.jobs.iter_mut().try_for_each(decode),
        }
    }
}

/// Returns `buffer` holding at least `len` bytes, cut to `len`. A buffer too short is freed before its
/// replacement exists, so that growing does not hold both, which is what the memory bound counts.
fn exact_buffer(buffer: &mut Vec<u8>, len: usize) -> &mut [u8] {
    if buffer.len() < len {
        *buffer = Vec::new();
        *buffer = vec![0; len];
    }
    &mut buffer[..len]
}

impl Job<'_> {
    /// Decodes this job's chunk, after its base chunk if it has one, and fills its destination.
    /// `stored` is the batch buffer and `chunks` the batch's stored chunks, laid out in it.
    fn run(&mut self, stored: &[u8], chunks: &[StoredChunk], engine: &Engine) -> Result<(), Error> {
        let chunk = stored_chunk(stored, chunks, self.stored, &self.place)?;
        let hash = engine.expected_hash(&self.place);
        let mut lease: Lease<'_> = engine.scratch.lease();
        let Scratch { workspace, chunk: canonical, base: base_buffer } = &mut *lease;
        let raw_len = to_usize(u64::from(self.place.raw_len))?;
        let base = match &self.base {
            Some(base) => {
                let base_stored = stored_chunk(stored, chunks, base.stored, &base.place)?;
                let base_len = to_usize(u64::from(base.place.raw_len))?;
                let base_canonical = exact_buffer(base_buffer, base_len);
                let base_hash = engine.expected_hash(&base.place);
                let place = &base.place;
                decode_located(place, base.chain, base_stored, None, base_canonical, workspace, base_hash)?;
                // Spec section 7a: the base bytes are the first raw length bytes of the base chunk.
                Some(slice_of(base_canonical, 0, raw_len)?)
            }
            None => None,
        };
        match self.output {
            Output::Whole => {
                decode_located(&self.place, self.chain, chunk, base, self.out, workspace, hash)
            }
            Output::Copy { skip } => {
                let canonical = exact_buffer(canonical, raw_len);
                decode_located(&self.place, self.chain, chunk, base, canonical, workspace, hash)?;
                self.out.copy_from_slice(slice_of(canonical, skip, self.out.len())?);
                Ok(())
            }
            Output::Widen { view, skip, narrow_len, head } => {
                let canonical = exact_buffer(canonical, raw_len);
                decode_located(&self.place, self.chain, chunk, base, canonical, workspace, hash)?;
                widen_part(view, slice_of(canonical, skip, narrow_len)?, head, self.out)
            }
        }
    }
}

/// The stored bytes of `place`, held as chunk `slot` of `chunks` in the batch buffer `stored`.
fn stored_chunk<'a>(
    stored: &'a [u8],
    chunks: &[StoredChunk],
    slot: usize,
    place: &ChunkPlace,
) -> Result<&'a [u8], Error> {
    let len = to_usize(u64::from(place.stored_len))?;
    let at = chunks.get(slot).ok_or_else(|| broken("a job names no chunk of its batch"))?.buffer_at;
    at.checked_add(len)
        .and_then(|end| stored.get(at..end))
        .ok_or_else(|| broken("a stored chunk is outside its batch"))
}

/// Decodes the chunk at `place` into `dst`, its raw length, combining it with `base` for a chunk of a
/// delta blob, and locates an error at `place`. `hash` is the expected XXH3 of the canonical bytes.
fn decode_located(
    place: &ChunkPlace,
    chain: Chain,
    stored: &[u8],
    base: Option<&[u8]>,
    dst: &mut [u8],
    workspace: &mut crate::codec::Workspace,
    hash: Option<u64>,
) -> Result<(), Error> {
    let decoded = match base {
        Some(base) => decode_delta_chunk_into(&chain, stored, base, dst, workspace, hash),
        None => decode_chunk_into(&chain, stored, dst, workspace, hash),
    };
    decoded.map_err(|error| locate_error(error, place.blob, place.chunk))
}

/// `len` bytes of `bytes` from `start`.
fn slice_of(bytes: &[u8], start: usize, len: usize) -> Result<&[u8], Error> {
    start
        .checked_add(len)
        .and_then(|end| bytes.get(start..end))
        .ok_or_else(|| broken("a read is outside its chunk"))
}

/// Decodes a chunk into the scratch of this job and returns its canonical bytes. `hash` is the
/// expected XXH3, checked on these bytes, so before any view.
fn decode_to_scratch<'a>(
    place: &ChunkPlace,
    chain: Chain,
    stored: &[u8],
    scratch: &'a mut Scratch,
    hash: Option<u64>,
) -> Result<&'a [u8], Error> {
    let canonical = exact_buffer(&mut scratch.chunk, to_usize(u64::from(place.raw_len))?);
    decode_chunk_into(&chain, stored, canonical, &mut scratch.workspace, hash)
        .map_err(|error| locate_error(error, place.blob, place.chunk))?;
    Ok(&*canonical)
}

/// Writes a run of a widened range. `narrow` holds whole elements, and `out` is their widened bytes
/// from byte `head` of the first one, so its first and last elements may be partial. Those go through
/// a 4-byte buffer; all whole elements between them are widened in place.
fn widen_part(view: View, narrow: &[u8], head: usize, out: &mut [u8]) -> Result<(), Error> {
    let narrow_bytes = to_usize(NARROW_BYTES)?;
    let (mut narrow, mut out) = (narrow, out);
    if head != 0 {
        let (element, rest) = narrow
            .split_at_checked(narrow_bytes)
            .ok_or_else(|| broken("a widened read has no first element"))?;
        let mut wide = [0; WIDE_LEN];
        widen_into(view, element, &mut wide)?;
        let (part, rest_out) = out.split_at_mut(out.len().min(WIDE_LEN - head));
        part.copy_from_slice(&wide[head..head + part.len()]);
        (narrow, out) = (rest, rest_out);
    }
    let whole = out.len() / WIDE_LEN;
    let (body, tail) = narrow
        .split_at_checked(whole * narrow_bytes)
        .ok_or_else(|| broken("a widened read is short of elements"))?;
    let (out_body, out_tail) = out.split_at_mut(whole * WIDE_LEN);
    widen_into(view, body, out_body)?;
    if !out_tail.is_empty() {
        let element = tail.get(..narrow_bytes).ok_or_else(|| broken("no last element"))?;
        let mut wide = [0; WIDE_LEN];
        widen_into(view, element, &mut wide)?;
        out_tail.copy_from_slice(&wide[..out_tail.len()]);
    }
    Ok(())
}

/// Serves one element that straddles two chunks: decodes each, one after the other, keeping only the
/// byte it needs, so that one stored chunk and one canonical chunk are in memory at a time.
fn read_straddling(
    shared: &Shared,
    chain: Chain,
    (first, second): (ChunkPlace, ChunkPlace),
    (view, head): (View, usize),
    out: &mut [u8],
    buffer: &mut Vec<u8>,
) -> Result<(), Error> {
    let low = chunk_edge(shared, chain, &first, Edge::Last, buffer)?;
    let high = chunk_edge(shared, chain, &second, Edge::First, buffer)?;
    let mut wide = [0; WIDE_LEN];
    widen_into(view, &[low, high], &mut wide)?;
    out.copy_from_slice(slice_of(&wide, head, out.len())?);
    Ok(())
}

/// Which end of a chunk [`chunk_edge`] returns.
#[derive(Debug, Clone, Copy)]
enum Edge {
    First,
    Last,
}

/// Reads and decodes one chunk and returns the byte at `edge` of its canonical bytes.
fn chunk_edge(
    shared: &Shared,
    chain: Chain,
    place: &ChunkPlace,
    edge: Edge,
    buffer: &mut Vec<u8>,
) -> Result<u8, Error> {
    let stored = exact_buffer(buffer, to_usize(u64::from(place.stored_len))?);
    shared.source.read_exact_at(place.file_offset, stored)?;
    let mut lease = shared.engine.scratch.lease();
    let hash = shared.engine.expected_hash(place);
    let canonical = decode_to_scratch(place, chain, stored, &mut lease, hash)?;
    let byte = match edge {
        Edge::First => canonical.first(),
        Edge::Last => canonical.last(),
    };
    byte.copied().ok_or_else(|| broken("a straddled chunk is empty"))
}

/// Walks segments into batches.
#[derive(Debug)]
struct Planner<'s, 'd, I> {
    segments: I,
    current: Option<Segment<'s, 'd>>,
}

/// One element to read on its own, see [`read_straddling`].
#[derive(Debug)]
struct Straddle<'d> {
    chain: Chain,
    first: ChunkPlace,
    second: ChunkPlace,
    view: View,
    head: usize,
    out: &'d mut [u8],
}

impl<'s, 'd, I> Planner<'s, 'd, I>
where
    I: Iterator<Item = Result<Segment<'s, 'd>, Error>>,
{
    /// Adds jobs to `batch` until it is full or the walk ends. Returns a straddling element when the
    /// walk reaches one with `batch` empty; with jobs in `batch` it stops there, so that the element
    /// comes after they ran.
    fn fill(&mut self, batch: &mut Batch<'d>, engine: &Engine) -> Result<Option<Straddle<'d>>, Error> {
        loop {
            if self.current.as_ref().is_none_or(|segment| segment.out.is_empty()) {
                self.current = self.segments.next().transpose()?;
            }
            let Some(segment) = &mut self.current else {
                return Ok(None);
            };
            match segment.next_piece()? {
                Piece::Single { place, output, out_len } => {
                    let base = segment.blob.base_of(&place)?;
                    let admitted = batch.admit(&place, base.as_ref().map(|(base, _)| base), engine)?;
                    let Some((stored, base_slot)) = admitted else {
                        return Ok(None);
                    };
                    let base = base
                        .zip(base_slot)
                        .map(|((place, chain), stored)| BaseJob { place, chain, stored });
                    let chain = segment.blob.chain();
                    let out = segment.advance(out_len);
                    batch.jobs.push(Job { place, chain, stored, base, output, out });
                }
                Piece::Straddle { first, second, view, head, out_len } => {
                    if !batch.jobs.is_empty() {
                        return Ok(None);
                    }
                    let chain = segment.blob.chain();
                    let out = segment.advance(out_len);
                    return Ok(Some(Straddle { chain, first, second, view, head, out }));
                }
            }
        }
    }
}

/// Serves `segments`: the whole of a read. Batches run in the order the segments come in, one at a
/// time, so the memory above the destinations is one batch plus the container's scratch.
///
/// # Errors
/// Returns what the source and the decoder return. On error the destinations hold unspecified bytes.
pub(super) fn read_segments<'s, 'd>(
    shared: &'s Shared,
    segments: impl Iterator<Item = Result<Segment<'s, 'd>, Error>>,
) -> Result<(), Error> {
    let mut planner = Planner { segments, current: None };
    let mut batch = Batch::new();
    let mut buffer = Vec::new();
    loop {
        match planner.fill(&mut batch, &shared.engine)? {
            Some(Straddle { chain, first, second, view, head, out }) => {
                read_straddling(shared, chain, (first, second), (view, head), out, &mut buffer)?;
            }
            None if batch.jobs.is_empty() => return Ok(()),
            None => {
                batch.execute(shared, &mut buffer)?;
                batch.clear();
            }
        }
    }
}
