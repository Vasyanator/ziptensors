//! File: write/ingest.rs
//! Purpose: the streaming pipeline that turns one source file into blobs and extents.
//!
//! Main responsibilities:
//! - cut the file into segments: typed ranges of the plan, and skeleton fragments between them;
//! - read each segment in order and hash it; decide the view, the rotation, a delta base and the
//!   identity;
//! - compress chunks on a private pool and write them in order, with rollback of the open blob.
//!
//! Key structures: [`Compressor`], [`Ingested`], [`Command`], [`DeltaInput`], `Reader`.
//!
//! Key functions: [`ingest`].
//!
//! Implementation notes: the calling thread is the one reader. A chunk goes to the pool with a one-shot
//! channel for its result, and that channel's receiver enters an ordered `sync_channel`, so the writer
//! thread takes results in submission order with no reorder buffer, whatever order they finish in.
//!
//! The window is 2 × threads chunks, counting the one the reader is filling and the one the writer
//! holds, hence a channel of 2 × threads − 2. A chunk in the window is one buffer of at most a chunk,
//! raw or stored; at most `threads` of them are being compressed and have both, and each of the
//! `threads` workspaces keeps at most 2.95 chunks: one rotated chunk, one stream of at most 32/34 of a
//! chunk, and one compressed stream of a chunk's compress bound, each grown exactly. That is the bound
//! of 6 × threads × chunk. It is why raw buffers are freed and not recycled: a free list would hold them next
//! to the stored bytes of the same chunks.
//!
//! A delta blob (spec section 7a) streams its base chunk beside each raw chunk, read by the same
//! thread from the base's own source file, so a chunk in the window holds two buffers until it is
//! encoded, and a workspace keeps one more, the residual. That is 2 × threads chunks of two buffers,
//! `threads` stored buffers of the chunks in compression, and `threads` workspaces of at most 3.95
//! chunks: 9 × threads × chunk. Workspaces keep their buffers, so the bound holds for the rest of a
//! pack once one delta blob was streamed.
//!
//! A duplicate or a late widening mismatch costs a rollback, not a buffer. Identity is final before the
//! last chunk of a blob is sent, so a duplicate of one chunk costs no compression at all, which is what
//! keeps a pack of many small skeleton fragments cheap in `max`.
//!
//! Every extent covers its whole blob. So the decode work of a member is at most its length, twice
//! that where it is delta blobs, whose base range has the delta blob's length and chunking, and no
//! member of this writer can meet the reader's bound of spec section 4. One blob per skeleton fragment
//! is there for that: a shared skeleton blob would put many small extents into 1 MiB chunks.
//!
//! Delta blobs come only from a typed range under the identity view with a paired base range: a
//! trial on the sample, against the base's sample, picks plain, rotated, delta or rotated delta.
//! Identity still decides first at the end of the stream, so a tensor equal to its base is a
//! duplicate of the base blob, and its delta chunks are rolled back.

use std::borrow::Cow;
use std::path::Path;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread;

use rayon::ThreadPool;

use crate::chain::{BLOB_KIND_CHUNKED, BLOB_KIND_DELTA, DeltaOp, View};
use crate::codec::{
    ChunkMeta, Workspace, encode_chunk, encode_delta_chunk, narrow_exact, narrow_into,
};
use crate::error::Error;
use crate::format::{BlobRecord, ChunkRecord, DeltaParams, ExtentRecord, IndexBuilder};
use crate::io::{FileSource, RandomAccess};
use crate::types::{Dtype, Mode, TypedRange};

use super::commit::Sink;
use super::dedup::{BlobShape, BlobTable, Identity};
use super::pairing::BaseRange;
use super::policy::{
    Encoding, SKELETON_DTYPE, computes_identity, delta_candidate, encoding_of, narrow_dtype,
    skeleton_encoding, widen_candidates, with_delta, with_rotl1,
};
use super::report::ClassTable;

/// Bytes read from the start of a range to pick a view and a rotation, which the stream then reads
/// again from the page cache, with no seek on a disk. A multiple of 8.
const SAMPLE_BYTES: u64 = 64 << 10;
/// Most wide bytes read at once while narrowing, also the read buffer. A multiple of 4.
const WIDE_READ_BYTES: usize = 64 << 10;

/// A failure of the pipeline's plumbing, not of the work it carries.
fn plumbing(op: &'static str, path: &Path, detail: &str) -> Error {
    Error::Io { op, path: path.to_path_buf(), source: std::io::Error::other(detail) }
}

/// One encoded chunk, on its way to the writer.
#[derive(Debug)]
struct EncodedChunk {
    stored: Vec<u8>,
    meta: ChunkMeta,
}

/// What a blob needs to become a record, once its chunks are written.
#[derive(Debug, Clone, Copy)]
struct SealedBlob {
    /// The id the reader gave out for this blob. Blobs are sealed in id order.
    id: u32,
    identity: Identity,
    raw_len: u64,
    encoding: Encoding,
    dtype: Dtype,
    /// Class of the source bytes in the report.
    class: Option<Dtype>,
    /// The base of a delta blob, which makes it kind 1; `None` for a plain blob.
    delta: Option<DeltaParams>,
}

/// What the writer does next, in file order.
#[derive(Debug)]
enum Command {
    /// Write this chunk, once it is encoded, to the open blob.
    Chunk(Receiver<Result<EncodedChunk, Error>>),
    /// Drop the open blob.
    Rollback,
    /// Close the open blob.
    Seal(SealedBlob),
}

/// The workspaces of the compression threads, each leased for one chunk. A bounded pool and not
/// thread-locals, so that memory is deterministic and goes away with the [`Compressor`], which must
/// happen before verification adds the reader's own bound.
#[derive(Debug)]
struct WorkspacePool {
    free: Mutex<Vec<Workspace>>,
    returned: Condvar,
}

impl WorkspacePool {
    /// Takes a workspace, waiting if all are in use, and hands it to `work`. The workspace returns to
    /// the pool when `work` ends, also when it panics, so a panicking job costs its own chunk (the
    /// writer reports it as a job with no result) and never starves the jobs after it.
    fn with_workspace<T>(&self, work: impl FnOnce(&mut Workspace) -> T) -> T {
        // A panic while the lock is held cannot leave the vector inconsistent: the critical sections
        // are one `pop` and one `push`. So a poisoned lock is safe to use.
        let mut free = self.free.lock().unwrap_or_else(PoisonError::into_inner);
        let workspace = loop {
            if let Some(workspace) = free.pop() {
                break workspace;
            }
            free = self.returned.wait(free).unwrap_or_else(PoisonError::into_inner);
        };
        drop(free);
        let mut lease = WorkspaceLease { pool: self, workspace: Some(workspace) };
        lease.with(work)
    }
}

/// Exclusive use of one workspace of a [`WorkspacePool`], returned to it on drop, also during a panic.
/// A workspace stays usable after a panic in the middle of a chunk: its buffers are scratch, resized
/// by every use, and its zstd contexts start a new frame on every call.
struct WorkspaceLease<'p> {
    pool: &'p WorkspacePool,
    /// `Some` for the whole life of the value; only `drop` takes it.
    workspace: Option<Workspace>,
}

impl WorkspaceLease<'_> {
    /// Runs `work` on the leased workspace.
    fn with<T>(&mut self, work: impl FnOnce(&mut Workspace) -> T) -> T {
        // Invariant: `with_workspace` creates the lease with `Some`, and only `drop` takes it.
        work(self.workspace.as_mut().expect("a lease holds its workspace until dropped"))
    }
}

impl Drop for WorkspaceLease<'_> {
    fn drop(&mut self) {
        if let Some(workspace) = self.workspace.take() {
            self.pool.free.lock().unwrap_or_else(PoisonError::into_inner).push(workspace);
            self.pool.returned.notify_one();
        }
    }
}

/// The compression side: a private pool, never the global one, so that hosts are unaffected.
/// One thread runs everything on the caller, with no pool.
#[derive(Debug)]
pub(super) struct Compressor {
    pool: Option<ThreadPool>,
    workspaces: Arc<WorkspacePool>,
    threads: usize,
}

impl Compressor {
    /// Starts `threads` compression threads, at least 1. `output` is for messages.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the pool cannot be started.
    pub(super) fn new(threads: usize, output: &Path) -> Result<Self, Error> {
        let threads = threads.max(1);
        let pool = (threads > 1)
            .then(|| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|i| format!("ziptensors-write-{i}"))
                    // A panic in a job reaches the writer as a closed one-shot channel, and is
                    // reported from there; without a handler it would abort the process.
                    .panic_handler(|_| tracing::error!("a compression job panicked"))
                    .build()
            })
            .transpose()
            .map_err(|source| Error::Io {
                op: "start write thread pool",
                path: output.to_path_buf(),
                source: std::io::Error::other(source),
            })?;
        let free = (0..threads).map(|_| Workspace::new()).collect();
        let workspaces = WorkspacePool { free: Mutex::new(free), returned: Condvar::new() };
        Ok(Self { pool, workspaces: Arc::new(workspaces), threads })
    }

    /// Capacity of the ordered channel: the window of 2 × threads chunks, less the two chunks outside
    /// the channel, which are the one being filled and the one being written.
    fn channel_capacity(&self) -> usize {
        2 * self.threads - 2
    }

    /// Stored length of `raw` under `encoding`, against `base` if the encoding has a delta step. For a
    /// trial: no result is kept.
    fn trial(&self, raw: &[u8], base: Option<&[u8]>, encoding: Encoding) -> Result<usize, Error> {
        Ok(encode_job(raw, base, encoding, &self.workspaces)?.stored.len())
    }

    /// Encodes `raw`, against `base` if the encoding has a delta step, on the pool or inline, and
    /// returns the channel its result arrives on.
    fn submit(
        &self,
        raw: Vec<u8>,
        base: Option<Vec<u8>>,
        encoding: Encoding,
    ) -> Receiver<Result<EncodedChunk, Error>> {
        let (done, result) = mpsc::sync_channel(1);
        let workspaces = Arc::clone(&self.workspaces);
        let job = move || {
            let encoded = encode_job(&raw, base.as_deref(), encoding, &workspaces);
            drop(raw);
            drop(base);
            // The capacity of 1 means this never blocks. An error means the writer stopped first, and
            // it has its own reason to report.
            if done.send(encoded).is_err() {
                tracing::debug!("a chunk finished after the writer stopped");
            }
        };
        match &self.pool {
            Some(pool) => pool.spawn(job),
            None => job(),
        }
        result
    }
}

/// Encodes one chunk, as a delta chunk against `base` when there is one. The stored buffer has the
/// size the spec allows, so it never grows.
fn encode_job(
    raw: &[u8],
    base: Option<&[u8]>,
    encoding: Encoding,
    workspaces: &WorkspacePool,
) -> Result<EncodedChunk, Error> {
    let mut stored = Vec::with_capacity(encoding.stored_bound(raw.len()));
    let (chain, stream) = (&encoding.chain, &encoding.stream);
    let meta = workspaces.with_workspace(|workspace| match base {
        Some(base) => encode_delta_chunk(raw, base, chain, stream, workspace, &mut stored),
        None => encode_chunk(raw, chain, stream, workspace, &mut stored),
    })?;
    Ok(EncodedChunk { stored, meta })
}

/// One contiguous part of a source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Segment {
    offset: u64,
    len: u64,
    /// The dtype of a typed range; `None` for a skeleton fragment.
    class: Option<Dtype>,
    /// The base tensor a typed range was paired with, a candidate for a delta blob.
    base: Option<BaseRange>,
}

/// Cuts `0..file_len` along `ranges`, which are sorted and disjoint as `normalize` leaves them. What no
/// range covers is skeleton, so the segments partition the file whatever a parser said. `bases` has
/// one entry per range, or is empty when the file has no base.
fn segments(ranges: &[TypedRange], bases: &[Option<BaseRange>], file_len: u64) -> Vec<Segment> {
    let mut out = Vec::with_capacity(ranges.len() * 2 + 1);
    let mut at = 0;
    for (number, range) in ranges.iter().enumerate() {
        if range.offset > at {
            out.push(Segment { offset: at, len: range.offset - at, class: None, base: None });
        }
        out.push(Segment {
            offset: range.offset,
            len: range.len,
            class: Some(range.hint),
            base: bases.get(number).copied().flatten(),
        });
        at = range.offset + range.len;
    }
    if file_len > at {
        out.push(Segment { offset: at, len: file_len - at, class: None, base: None });
    }
    out
}

/// The state of the writer thread, for the time of one source file.
#[derive(Debug)]
struct Writer<'a> {
    sink: &'a mut Sink,
    index: &'a mut IndexBuilder,
    /// Where the open blob starts.
    blob_start: u64,
    /// Chunks of the open blob, not yet in the index since the blob can still be dropped.
    open: Vec<ChunkRecord>,
    stored: ClassTable,
    compressed_bytes: u64,
}

/// What the writer thread did with one source file.
#[derive(Debug)]
struct Written {
    /// Stored bytes per class.
    stored: ClassTable,
    /// Stored bytes written.
    compressed_bytes: u64,
}

impl Writer<'_> {
    /// Executes commands until the reader is done. A blob still open then belongs to a reader that
    /// failed, and is dropped, so a failed file leaves only complete, unreferenced blobs behind.
    fn run(mut self, commands: &Receiver<Command>, path: &Path) -> Result<Written, Error> {
        for command in commands {
            match command {
                Command::Chunk(result) => self.write_chunk(&result, path)?,
                Command::Rollback => self.rollback()?,
                Command::Seal(blob) => self.seal(&blob, path)?,
            }
        }
        self.rollback()?;
        Ok(Written { stored: self.stored, compressed_bytes: self.compressed_bytes })
    }

    fn write_chunk(
        &mut self,
        result: &Receiver<Result<EncodedChunk, Error>>,
        path: &Path,
    ) -> Result<(), Error> {
        let chunk = result
            .recv()
            .map_err(|_| plumbing("compress", path, "a compression job ended with no result"))??;
        let file_offset = self.sink.position();
        self.sink.write(&chunk.stored)?;
        self.open.push(ChunkRecord {
            file_offset,
            xxh3_decoded: chunk.meta.xxh3_decoded,
            stored_len: chunk.meta.stored_len,
        });
        Ok(())
    }

    fn rollback(&mut self) -> Result<(), Error> {
        self.open.clear();
        self.sink.rollback(self.blob_start)
    }

    /// Turns the open blob into records. Its chunks enter `CHNK` here, contiguous, as a blob needs,
    /// and the base of a delta blob enters `PARM`.
    fn seal(&mut self, blob: &SealedBlob, path: &Path) -> Result<(), Error> {
        let chain_id = self.index.add_chain(blob.encoding.chain)?;
        let (kind, params_off, params_len) = match blob.delta {
            Some(params) => {
                let (off, len) = self.index.add_params(&params.encode())?;
                (BLOB_KIND_DELTA, off, len)
            }
            None => (BLOB_KIND_CHUNKED, 0, 0),
        };
        let chunk_count = u32::try_from(self.open.len())
            .map_err(|_| plumbing("seal blob", path, "a blob has over 2^32 chunks"))?;
        let mut first_chunk = None;
        for chunk in self.open.drain(..) {
            let id = self.index.add_chunk(chunk);
            first_chunk.get_or_insert(id);
        }
        let id = self.index.add_blob(BlobRecord {
            blake3: blob.identity,
            raw_len: blob.raw_len,
            first_chunk: first_chunk.unwrap_or(0),
            chunk_count,
            chunk_size: blob.encoding.chunk_size,
            chain_id,
            kind,
            dtype_hint: blob.dtype,
            params_off,
            params_len,
        })?;
        if id != blob.id {
            return Err(plumbing("seal blob", path, "blob ids went out of step"));
        }
        let stored = self.sink.position() - self.blob_start;
        self.stored.add_stored(blob.class, blob.raw_len, stored);
        self.compressed_bytes += stored;
        self.blob_start = self.sink.position();
        Ok(())
    }
}

/// How a run of [`Reader::stream`] ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Streamed {
    /// A new blob, sealed.
    New(u32),
    /// The canonical bytes are those of this blob, and nothing was kept.
    Duplicate(u32),
    /// An element is not exact under the view. Nothing was kept.
    NotExact,
}

/// What is shared with the [`ingest`] caller for the length of one file.
#[derive(Debug)]
pub(super) struct IngestContext<'a> {
    /// The pack's compression side.
    pub(super) compressor: &'a Compressor,
    /// Writer mode.
    pub(super) mode: Mode,
    /// Blobs of the pack, by id and by identity.
    pub(super) blobs: &'a mut BlobTable,
    /// The id the next new blob gets.
    pub(super) next_blob: &'a mut u32,
    /// The base of this file, when it is stored against one.
    pub(super) delta: Option<DeltaInput<'a>>,
}

/// The base of a file stored with delta blobs: where to read base bytes, and which base range each
/// typed range of the file was paired with.
#[derive(Debug, Clone, Copy)]
pub(super) struct DeltaInput<'a> {
    /// The base member's source file, read again for its canonical bytes. A change since it was added
    /// is caught by its stamp and by the commit's decode.
    pub(super) source: &'a FileSource,
    /// One entry per typed range of the file, in order.
    pub(super) bases: &'a [Option<BaseRange>],
}

/// Why a delta blob of `len` bytes and `chunk_size` cannot rest on `range` of a blob of shape `base`,
/// by the rules of spec section 7a; `None` if it can.
fn base_refusal(
    base: Option<BlobShape>,
    range: BaseRange,
    len: u64,
    chunk_size: u32,
) -> Option<&'static str> {
    let Some(shape) = base else {
        return Some("the base blob is not in this pack");
    };
    if shape.kind != BLOB_KIND_CHUNKED {
        return Some("the base blob is a delta blob");
    }
    if shape.chunk_size != chunk_size {
        return Some("the base blob has another chunk size");
    }
    if !range.blob_offset.is_multiple_of(u64::from(chunk_size)) {
        return Some("the base range does not start on a chunk");
    }
    if range.blob_offset.checked_add(len).is_none_or(|end| end > shape.raw_len) {
        return Some("the base range leaves the base blob");
    }
    None
}

/// The reader's side of the pipeline.
#[derive(Debug)]
struct Reader<'a, 'c> {
    source: &'a FileSource,
    context: IngestContext<'c>,
    commands: SyncSender<Command>,
    /// BLAKE3 of the file up to the read position.
    hasher: blake3::Hasher,
    /// Holds wide bytes on their way to a narrow chunk.
    wide: Vec<u8>,
    extents: Vec<ExtentRecord>,
    table: ClassTable,
    blobs_added: u32,
    blobs_reused: u32,
    widen_restreams: u32,
    delta_blobs: u32,
}

/// Reads `buf` from `source`. A short read is a file that shrank under us.
fn read_from(source: &FileSource, offset: u64, buf: &mut [u8]) -> Result<(), Error> {
    source.read_exact_at(offset, buf).map_err(|error| match error {
        Error::Io { source, path, .. } if source.kind() == std::io::ErrorKind::UnexpectedEof => {
            Error::InputChanged { path }
        }
        other => other,
    })
}

impl Reader<'_, '_> {
    /// Reads `buf` from the source.
    fn read(&self, offset: u64, buf: &mut [u8]) -> Result<(), Error> {
        read_from(self.source, offset, buf)
    }

    /// Reads `buf` from the base file at `offset`.
    fn read_base(&self, offset: u64, buf: &mut [u8]) -> Result<(), Error> {
        let delta = self.context.delta.ok_or_else(|| {
            plumbing("read base", self.source.path(), "a delta blob without a base file")
        })?;
        read_from(delta.source, offset, buf)
    }

    /// Queues a command. A closed channel means the writer stopped; its error is what gets reported.
    fn send(&self, command: Command) -> Result<(), Error> {
        self.commands
            .send(command)
            .map_err(|_| plumbing("write", self.source.path(), "the writer stopped"))
    }

    /// Reads the first [`SAMPLE_BYTES`] of `segment`, or all of it, a whole number of `align` long.
    fn sample(&self, segment: &Segment, align: u64) -> Result<Vec<u8>, Error> {
        let len = segment.len.min(SAMPLE_BYTES) / align * align;
        let mut sample = vec![0; to_usize(len, self.source.path())?];
        self.read(segment.offset, &mut sample)?;
        Ok(sample)
    }

    /// The view to stream a range under, from a sample: the first candidate that reproduces all of it.
    /// A sample proves nothing about the rest, which [`Reader::fill_narrow`] finds out.
    fn pick_view(&self, segment: &Segment, hint: Dtype) -> Result<(View, Vec<u8>), Error> {
        let candidates = widen_candidates(hint, self.context.mode);
        if candidates.is_empty() {
            return Ok((View::Identity, Vec::new()));
        }
        let sample = self.sample(segment, 4)?;
        let exact = candidates.iter().find(|view| narrow_exact(**view, &sample) == Some(sample.len()));
        Ok((exact.copied().unwrap_or(View::Identity), sample))
    }

    /// `encoding` with `rotl1` if that stores a sample of the blob in fewer bytes. Depends only on the
    /// bytes of the sample, so the output does not depend on the thread count. `wide_sample` is what
    /// [`Reader::pick_view`] read, and is empty if it read nothing.
    fn pick_rotation(
        &self,
        segment: &Segment,
        view: View,
        encoding: Encoding,
        wide_sample: &[u8],
    ) -> Result<Encoding, Error> {
        let Some(rotated) = with_rotl1(encoding, self.context.mode)? else {
            return Ok(encoding);
        };
        let sample = match view {
            View::Identity if wide_sample.is_empty() => {
                Cow::Owned(self.sample(segment, u64::from(encoding.chain.block_bytes()))?)
            }
            View::Identity => Cow::Borrowed(wide_sample),
            View::F16ToF32 | View::Bf16ToF32 => {
                let mut narrow = vec![0; wide_sample.len() / 2];
                narrow_into(view, wide_sample, &mut narrow)?;
                Cow::Owned(narrow)
            }
        };
        let compressor = self.context.compressor;
        if compressor.trial(&sample, None, rotated)? < compressor.trial(&sample, None, encoding)? {
            return Ok(rotated);
        }
        Ok(encoding)
    }

    /// The delta encoding of `segment` over its paired base, if the mode tries one and the base blob
    /// can carry it: kind 0, the same `chunk_size`, a chunk-aligned start, the whole range inside.
    fn delta_encoding(
        &self,
        segment: &Segment,
        hint: Dtype,
        encoding: Encoding,
    ) -> Result<Option<(BaseRange, DeltaOp, Encoding)>, Error> {
        let (Some(base), Some(op), Some(_)) =
            (segment.base, delta_candidate(hint, self.context.mode), self.context.delta)
        else {
            return Ok(None);
        };
        let delta = with_delta(encoding, op)?;
        let shape = self.context.blobs.shape(base.blob);
        if let Some(reason) = base_refusal(shape, base, segment.len, delta.chunk_size) {
            tracing::debug!(
                path = %self.source.path().display(),
                offset = segment.offset,
                base_blob = base.blob,
                reason,
                "no delta against the paired base"
            );
            return Ok(None);
        }
        Ok(Some((base, op, delta)))
    }

    /// The encoding of a range under the identity view, and its base if it becomes a delta blob. With
    /// a usable base, a trial on the sample picks the smallest of plain, rotated, delta and rotated
    /// delta, in that order on ties, so a delta is chosen only where it stores smaller. Otherwise this
    /// is [`Reader::pick_rotation`].
    fn pick_identity(
        &self,
        segment: &Segment,
        hint: Dtype,
        encoding: Encoding,
        wide_sample: &[u8],
    ) -> Result<(Encoding, Option<BaseRange>), Error> {
        let Some((base, op, delta)) = self.delta_encoding(segment, hint, encoding)? else {
            let encoding = self.pick_rotation(segment, View::Identity, encoding, wide_sample)?;
            return Ok((encoding, None));
        };
        let block = u64::from(delta.chain.block_bytes());
        // The sample `pick_view` read is the same bytes when it has the same whole-block length.
        let whole = segment.len.min(SAMPLE_BYTES) / block * block;
        let sample = if !wide_sample.is_empty() && u64::try_from(wide_sample.len()) == Ok(whole) {
            Cow::Borrowed(wide_sample)
        } else {
            Cow::Owned(self.sample(segment, block)?)
        };
        let mut base_sample = vec![0; sample.len()];
        self.read_base(base.file_offset, &mut base_sample)?;
        let mode = self.context.mode;
        let rotated = with_rotl1(encoding, mode)?;
        let rotated_delta = rotated.map(|rotated| with_delta(rotated, op)).transpose()?;
        let candidates = [
            (Some(encoding), None),
            (rotated, None),
            (Some(delta), Some(base)),
            (rotated_delta, Some(base)),
        ];
        let compressor = self.context.compressor;
        let mut best: Option<(usize, Encoding, Option<BaseRange>)> = None;
        for (candidate, with_base) in candidates {
            let Some(candidate) = candidate else { continue };
            if with_base.is_some() && candidate.chunk_size != delta.chunk_size {
                continue;
            }
            let base_bytes = with_base.map(|_| base_sample.as_slice());
            let size = compressor.trial(&sample, base_bytes, candidate)?;
            if best.is_none_or(|(smallest, _, _)| size < smallest) {
                best = Some((size, candidate, with_base));
            }
        }
        let (_, encoding, base) = best.ok_or_else(|| {
            plumbing("pick encoding", self.source.path(), "no candidate encoding")
        })?;
        Ok((encoding, base))
    }

    /// Fills `raw` with the narrow form of the wide bytes at `offset`, hashing them. Returns whether
    /// every element was exact; on `false` the contents of `raw` are meaningless.
    fn fill_narrow(&mut self, offset: u64, view: View, raw: &mut [u8]) -> Result<bool, Error> {
        let mut wide = std::mem::take(&mut self.wide);
        wide.resize(WIDE_READ_BYTES, 0);
        let mut at = offset;
        let mut exact = true;
        for piece in raw.chunks_mut(WIDE_READ_BYTES / 2) {
            let wide_piece = &mut wide[..piece.len() * 2];
            self.read(at, wide_piece)?;
            self.hasher.update(wide_piece);
            if narrow_into(view, wide_piece, piece)? != wide_piece.len() {
                exact = false;
                break;
            }
            // Lossless: `usize` is at most 64 bits on every supported target.
            at += wide_piece.len() as u64;
        }
        self.wide = wide;
        Ok(exact)
    }

    /// Streams `segment` as one blob of `encoding` seen through `view`. Hashes the source bytes. With
    /// `base`, which needs the identity view and a delta step in `encoding`, each chunk goes with the
    /// base bytes it pairs with, and the blob is a delta blob over `base`.
    fn stream(
        &mut self,
        segment: &Segment,
        view: View,
        encoding: Encoding,
        dtype: Dtype,
        base: Option<BaseRange>,
    ) -> Result<Streamed, Error> {
        let factor = view.length_factor();
        let canonical_len = segment.len / factor;
        let chunk_size = u64::from(encoding.chunk_size);
        let mut identity = computes_identity(self.context.mode).then(blake3::Hasher::new);
        let mut done = 0;
        let mut sent_any = false;
        loop {
            let len = chunk_size.min(canonical_len - done);
            let mut raw = vec![0; to_usize(len, self.source.path())?];
            let offset = segment.offset + done * factor;
            if view == View::Identity {
                self.read(offset, &mut raw)?;
                self.hasher.update(&raw);
            } else if !self.fill_narrow(offset, view, &mut raw)? {
                self.send(Command::Rollback)?;
                return Ok(Streamed::NotExact);
            }
            if let Some(hasher) = &mut identity {
                hasher.update(&raw);
            }
            let at = done;
            done += len;
            if done < canonical_len {
                let base_bytes = self.base_chunk(base, at, raw.len())?;
                self.send(Command::Chunk(self.context.compressor.submit(raw, base_bytes, encoding)))?;
                sent_any = true;
                continue;
            }
            // Final before the last chunk is sent: a blob of one chunk then costs no compression.
            let identity = identity.map_or([0; 32], |hasher| *hasher.finalize().as_bytes());
            if let Some(blob) = self.context.blobs.find(&identity) {
                if sent_any {
                    self.send(Command::Rollback)?;
                }
                return Ok(Streamed::Duplicate(blob));
            }
            let base_bytes = self.base_chunk(base, at, raw.len())?;
            self.send(Command::Chunk(self.context.compressor.submit(raw, base_bytes, encoding)))?;
            let id = *self.context.next_blob;
            *self.context.next_blob = id.checked_add(1).ok_or_else(|| {
                plumbing("seal blob", self.source.path(), "too many blobs for a u32 id")
            })?;
            let delta = base.map(|base| DeltaParams {
                base_blob: base.blob,
                base_offset: base.blob_offset,
            });
            self.send(Command::Seal(SealedBlob {
                id,
                identity,
                raw_len: canonical_len,
                encoding,
                dtype,
                class: segment.class,
                delta,
            }))?;
            let kind = if delta.is_some() { BLOB_KIND_DELTA } else { BLOB_KIND_CHUNKED };
            let shape = BlobShape { kind, chunk_size: encoding.chunk_size, raw_len: canonical_len };
            self.context.blobs.record(id, identity, shape)?;
            if delta.is_some() {
                self.delta_blobs += 1;
            }
            return Ok(Streamed::New(id));
        }
    }

    /// The base bytes of the chunk at canonical offset `at` of a delta blob over `base`, `len` long;
    /// `None` for a plain blob. Under the identity view they are the base file's bytes at the same
    /// distance from the base tensor's start.
    fn base_chunk(
        &self,
        base: Option<BaseRange>,
        at: u64,
        len: usize,
    ) -> Result<Option<Vec<u8>>, Error> {
        let Some(base) = base else {
            return Ok(None);
        };
        let offset = base.file_offset.checked_add(at).ok_or_else(|| {
            plumbing("read base", self.source.path(), "a base offset past 2^64")
        })?;
        let mut bytes = vec![0; len];
        self.read_base(offset, &mut bytes)?;
        Ok(Some(bytes))
    }

    /// Adds the extent of `segment`, over all of blob `blob`, and accounts for its bytes.
    fn add_extent(&mut self, segment: &Segment, blob: u32, view: View, duplicate: bool) {
        self.extents.push(ExtentRecord {
            virt_offset: segment.offset,
            virt_len: segment.len,
            blob_id: blob,
            view_id: view.to_wire(),
            blob_offset: 0,
        });
        let canonical = segment.len / view.length_factor();
        self.table.add_source(segment.class, segment.len, canonical, duplicate);
        if duplicate {
            self.blobs_reused += 1;
        } else {
            self.blobs_added += 1;
        }
    }

    /// Stores a typed range: picks a view and a rotation, and streams again without the view if an
    /// element late in the range turns out not to be exact.
    fn typed(&mut self, segment: &Segment, hint: Dtype) -> Result<(), Error> {
        let (view, wide_sample) = self.pick_view(segment, hint)?;
        let mode = self.context.mode;
        if view != View::Identity {
            let narrow = narrow_dtype(view, hint);
            let encoding = encoding_of(narrow, mode)?;
            let encoding = self.pick_rotation(segment, view, encoding, &wide_sample)?;
            // A second pass over this range must not hash its bytes twice.
            let before = self.hasher.clone();
            match self.stream(segment, view, encoding, narrow, None)? {
                Streamed::New(blob) => {
                    self.add_extent(segment, blob, view, false);
                    return Ok(());
                }
                Streamed::Duplicate(blob) => {
                    self.add_extent(segment, blob, view, true);
                    return Ok(());
                }
                Streamed::NotExact => {
                    tracing::debug!(
                        path = %self.source.path().display(),
                        offset = segment.offset,
                        "late widening mismatch, streaming the range again"
                    );
                    self.hasher = before;
                    self.widen_restreams += 1;
                }
            }
        }
        let encoding = encoding_of(hint, mode)?;
        let (encoding, base) = self.pick_identity(segment, hint, encoding, &wide_sample)?;
        self.plain(segment, encoding, hint, base)
    }

    /// Stores `segment` as it is, as a delta blob over `base` if there is one.
    fn plain(
        &mut self,
        segment: &Segment,
        encoding: Encoding,
        dtype: Dtype,
        base: Option<BaseRange>,
    ) -> Result<(), Error> {
        match self.stream(segment, View::Identity, encoding, dtype, base)? {
            Streamed::New(blob) => self.add_extent(segment, blob, View::Identity, false),
            Streamed::Duplicate(blob) => self.add_extent(segment, blob, View::Identity, true),
            Streamed::NotExact => {
                return Err(plumbing(
                    "stream blob",
                    self.source.path(),
                    "an identity view found an inexact element",
                ));
            }
        }
        Ok(())
    }

    /// Reads the whole file, in order. Dropping `self` closes the channel, which ends the writer.
    fn run(mut self, parts: &[Segment]) -> Result<ReaderOutcome, Error> {
        for segment in parts {
            match segment.class {
                Some(hint) => self.typed(segment, hint)?,
                None => self.plain(segment, skeleton_encoding()?, SKELETON_DTYPE, None)?,
            }
        }
        Ok(ReaderOutcome {
            hasher: self.hasher,
            extents: self.extents,
            table: self.table,
            blobs_added: self.blobs_added,
            blobs_reused: self.blobs_reused,
            widen_restreams: self.widen_restreams,
            delta_blobs: self.delta_blobs,
        })
    }
}

/// What the reader hands back at the end of a file.
#[derive(Debug)]
struct ReaderOutcome {
    hasher: blake3::Hasher,
    extents: Vec<ExtentRecord>,
    table: ClassTable,
    blobs_added: u32,
    blobs_reused: u32,
    widen_restreams: u32,
    delta_blobs: u32,
}

/// What one source file became.
#[derive(Debug)]
pub(super) struct Ingested {
    /// BLAKE3 of the file.
    pub(super) blake3: [u8; 32],
    /// Extents, in member order, with blob ids of this pack.
    pub(super) extents: Vec<ExtentRecord>,
    /// Bytes per class of this file.
    pub(super) table: ClassTable,
    /// Blobs added.
    pub(super) blobs_added: u32,
    /// Extents over a blob that existed.
    pub(super) blobs_reused: u32,
    /// Ranges streamed twice.
    pub(super) widen_restreams: u32,
    /// Delta blobs added.
    pub(super) delta_blobs: u32,
    /// Stored bytes written.
    pub(super) compressed_bytes: u64,
}

/// Converts a length that a buffer must hold.
fn to_usize(len: u64, path: &Path) -> Result<usize, Error> {
    usize::try_from(len).map_err(|_| plumbing("read", path, "a chunk does not fit the address space"))
}

/// Streams `source` into the pack: `ranges` are its normalized typed ranges, and `context.delta`, if
/// set, has one base entry per range.
///
/// On `Err` from reading, the pack stays usable: blobs already sealed stay as unreferenced records, an
/// open blob is rolled back, and `sink` and `index` agree. On `Err` from writing, `sink` is poisoned.
///
/// # Errors
/// [`Error::InputChanged`] if the file or the base file shrinks under the read, [`Error::Io`] for a
/// failed read or write, [`Error::Zstd`] if compression fails.
pub(super) fn ingest(
    source: &FileSource,
    ranges: &[TypedRange],
    context: IngestContext<'_>,
    sink: &mut Sink,
    index: &mut IndexBuilder,
) -> Result<Ingested, Error> {
    let bases = context.delta.map_or(&[][..], |delta| delta.bases);
    let parts = segments(ranges, bases, source.len());
    let compressor = context.compressor;
    let path = source.path();
    let (commands, received) = mpsc::sync_channel(compressor.channel_capacity());
    let reader = Reader {
        source,
        context,
        commands,
        hasher: blake3::Hasher::new(),
        wide: Vec::new(),
        extents: Vec::with_capacity(parts.len()),
        table: ClassTable::default(),
        blobs_added: 0,
        blobs_reused: 0,
        widen_restreams: 0,
        delta_blobs: 0,
    };
    let blob_start = sink.position();
    let joined = thread::scope(|scope| {
        let writer = Writer {
            sink: &mut *sink,
            index,
            blob_start,
            open: Vec::new(),
            stored: ClassTable::default(),
            compressed_bytes: 0,
        };
        let writer = thread::Builder::new()
            .name("ziptensors-writer".to_owned())
            .spawn_scoped(scope, move || writer.run(&received, path))
            .map_err(|source| Error::Io {
                op: "start writer thread",
                path: path.to_path_buf(),
                source,
            })?;
        let read = reader.run(&parts);
        Ok::<_, Error>((read, writer.join()))
    });
    let (read, written) = joined?;
    // A writer's failure comes first: the reader's own error is then only that the writer stopped.
    let written = written
        .unwrap_or_else(|_| Err(plumbing("join writer", path, "the writer thread panicked")));
    let Written { stored, compressed_bytes } = match written {
        Ok(result) => result,
        Err(error) => {
            sink.poison();
            return Err(error);
        }
    };
    let outcome = read?;
    let mut table = outcome.table;
    table.merge(&stored);
    Ok(Ingested {
        blake3: *outcome.hasher.finalize().as_bytes(),
        extents: outcome.extents,
        table,
        blobs_added: outcome.blobs_added,
        blobs_reused: outcome.blobs_reused,
        widen_restreams: outcome.widen_restreams,
        delta_blobs: outcome.delta_blobs,
        compressed_bytes,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{Condvar, Mutex, PoisonError};

    use super::{BaseRange, Segment, TypedRange, WorkspacePool, segments};
    use crate::codec::Workspace;
    use crate::types::Dtype;

    fn range(offset: u64, len: u64) -> TypedRange {
        TypedRange { offset, len, hint: Dtype::F32 }
    }

    fn skeleton(offset: u64, len: u64) -> Segment {
        Segment { offset, len, class: None, base: None }
    }

    fn typed(offset: u64, len: u64) -> Segment {
        Segment { offset, len, class: Some(Dtype::F32), base: None }
    }

    #[test]
    fn no_ranges_make_one_fragment_and_an_empty_file_none() {
        assert_eq!(segments(&[], &[], 10), [skeleton(0, 10)]);
        assert_eq!(segments(&[], &[], 0), []);
    }

    #[test]
    fn segments_partition_the_file_around_ranges() {
        let ranges = [range(4, 4), range(8, 2), range(12, 3)];
        let expected = [
            skeleton(0, 4),
            typed(4, 4),
            typed(8, 2),
            skeleton(10, 2),
            typed(12, 3),
            skeleton(15, 5),
        ];
        assert_eq!(segments(&ranges, &[], 20), expected);
    }

    #[test]
    fn typed_segments_carry_the_base_of_their_range() {
        let base = BaseRange { file_offset: 0, blob: 2, blob_offset: 0 };
        let ranges = [range(4, 4), range(8, 2)];
        let parts = segments(&ranges, &[None, Some(base)], 12);
        let bases: Vec<_> = parts.iter().map(|part| part.base).collect();
        assert_eq!(bases, [None, None, Some(base), None]);
    }

    #[test]
    fn a_range_may_touch_both_ends() {
        assert_eq!(segments(&[range(0, 8)], &[], 8), [typed(0, 8)]);
    }

    /// Workspaces not in use.
    fn available(pool: &WorkspacePool) -> usize {
        pool.free.lock().unwrap_or_else(PoisonError::into_inner).len()
    }

    #[test]
    fn a_workspace_returns_to_the_pool_when_its_job_panics() {
        let pool = WorkspacePool { free: Mutex::new(vec![Workspace::new()]), returned: Condvar::new() };
        for _ in 0..3 {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pool.with_workspace(|_| panic!("injected panic in a compression job"))
            }));
            assert!(outcome.is_err(), "the job panicked");
            // Checked before the next lease: a lost workspace would make it wait forever.
            assert_eq!(available(&pool), 1, "the workspace came back");
        }
        assert_eq!(pool.with_workspace(|_| 7), 7);
        assert_eq!(available(&pool), 1);
    }
}
