# Module: crates/ziptensors-core/src/read

## Purpose

The reader API: `Container`, `Member`, reads by offset and tensor name, restore, verify. Spec sections 6,
7, 9, 9.1 and 9.2 define the behavior.

## Architecture

```text
Container::open   header, footer, check_layout -> stored index -> one frame, exact length -> check_index
                  -> Index::parse -> validate -> check_decode_budget -> members, tensors, Engine
read_at           Segments (member range -> one segment per extent)
                  -> Planner (segment -> chunks, lazily) -> Batch (bounded, coalesced)
                  -> read_exact_at on the calling thread -> decode on the private pool -> repeat
read_many         the same, with segments collected and sorted by file offset first
restore, verify,  read_at in runs of one batch (hash_stream), streamed through BLAKE3
decode_to           -> temporary file and commit / nothing / the caller's writer
restore_all       every name checked -> directories -> restore_file per member, in file order
ReadCache         CachedMember::read_at -> Member::chunk_window per window -> hit, or wait for a fill in
                  flight, or reserve a run of absent windows -> one read_many with no lock held -> publish
```

A job decodes one chunk. A whole chunk under an identity extent decodes straight into its slice of `dst`,
so the codec's direct and in-place routes apply. Everything else decodes into the job's scratch and is
copied or widened from there. Partial wide elements at either end of a run go through a 4-byte buffer.

A job over a delta blob (spec section 7a) also carries the base chunk its chunk pairs with
(`BlobView::base_of`, through `DeltaParams::base_chunk`). Both stored chunks enter the same batch, the
job decodes the base into its scratch entry's `base` buffer, checked against the base chunk's own hash,
then the delta chunk through `codec::decode_delta_chunk_into` into `dst` or its scratch as above. Jobs
admit base and delta chunks alternately, but each blob's chunks are contiguous in the file, so the
batch lays its buffer out at execution, in file order: a delta batch reads its base chunks as one read
and its delta chunks as another.

## Files and submodules

| Path | Role |
|---|---|
| `mod.rs` | Module list and re-exports. |
| `container.rs` | `Container`, `OpenOptions`, `MemberInfo`, loading of the stored index, `Shared`. |
| `member.rs` | `Member`, `ReadRequest`: argument checks, tensor lookup, `read_many` ordering. |
| `batch.rs` | `Engine`, segments, planning, coalescing, decode jobs with their delta base chunks, widening of partial runs. |
| `scratch.rs` | `Scratch` (workspace, canonical chunk, base chunk), the bounded pool and its RAII `Lease`. |
| `cache.rs` | `ReadCache`, `CachedMember`, `CacheOptions`, `CacheStats`: the opt-in decoded-window cache. |
| `restore.rs` | `restore_file`, `decode_to`, `verify`, `Container::restore_all`, `RestoreReport`, `RestoreAllReport`, and their tests. The commit itself is `io::TempFile`. |
| `testkit.rs` | Test only. `ContainerBuilder`, `MemorySource`, seeded data. Also included by `tests/read_memory.rs`. |
| `cache_tests.rs` | Tests of the cache against `Member::read_at`. |
| `tests.rs` | Reader tests. The peak memory test is `tests/read_memory.rs`, which needs its own allocator. |

## Contracts and invariants

- The budget `max_total_decoded` covers all members of the file and is checked at open, so before any data
  read. A pack over it does not open. Open reads the header, the footer and the stored index, nothing else.
- Open also refuses a container whose decode work is over `max_decode_amplification` × the sum of member
  lengths + 256 MiB, so hostile extents cannot make a read of a small file decode one chunk millions of times,
  nothing being cached. The floor comes once per container, so many members do not multiply it.
- `max_total_decoded` defaults to `u64::MAX`. Untrusted input needs a budget from the caller.
- The stored index is one frame with the zstd magic, a content size equal to `index_raw_len`, and no bytes
  after it. Its length is bounded by `compress_bound(index_raw_len)` before it sizes an allocation, because
  the file length alone would let a huge file allocate that much.
- Within a batch, stored bytes are read in increasing file offset, so a batch never seeks backwards,
  and stored chunks of a batch that touch in the file are one read, never two back to
  back: Linux takes the second of two such reads for a sequential stream and reads ahead a window of its
  own (1-2 MiB measured with `read_ahead_kb` 8192), which a random read never uses. Jobs restore member
  order.
- The calling thread does all I/O. Decoding runs on the pool owned by the `Container`, never the global
  rayon pool, and a batch of one job runs inline. `threads = 1` builds no pool.
- Batches run one at a time per call. A batch holds at most threads × (largest chunk + 32 bytes of prologue)
  stored bytes and threads × 64 jobs, and always at least one job. A delta job's two chunks are admitted
  together or not at all, so a batch's first job may take it one chunk over; a chunk any job of the
  batch uses, as its own or its base, is held and read once. "Chunk" in the bound is the largest
  `chunk_size` of the file: scratch outlives a read, so what a member uses does not limit it.
- Memory above the destination and the index, for one call, is at most 3 × threads × chunk, plus fixed
  bookkeeping: one batch of stored bytes, and per scratch entry one chunk of canonical bytes and at most one
  chunk of codec scratch. Reads of delta blobs add one stored chunk to a batch and one chunk of base
  canonical bytes per scratch entry, (4 × threads + 1) × chunk. The base buffer is allocated on the first
  delta chunk and, like all scratch, outlives the read, so a container that never reads a delta blob keeps
  the first bound and one that has read one counts the second for every later read. In practice a delta
  batch holds about threads / 2 jobs, so its peak stays near 2.5 × threads × chunk, and its decode uses
  about half the threads.
- An open `Container` also holds its index: the decoded records, the member list and the tensor table with
  copies of names and shapes, about twice `index_raw_len`. It is fixed at open and does not grow with
  reads. Concurrent calls each hold their own batch and share the scratch entries.
  `read_many` adds one small record per extent touched. `restore_file`, `decode_to` and `verify` add their stream
  buffer, one batch long and at least two chunks, which is the destination of their reads. zstd's own contexts allocate in C:
  about 100 KiB per scratch entry, constant.
- `stored` and `dst` are disjoint by construction: stored bytes live in a buffer the call owns.
- Hash checks run on canonical bytes, before any view. A partial read decodes and checks the whole chunk.
  A base chunk is checked against its own hash before it is combined, so a damaged base is reported at the
  base blob and chunk; the delta chunk's hash is checked on the combined bytes. `Member` caches nothing
  between calls.
- The chain and the raw length of a chunk come from the blob being read, never from the chunk record, which
  blobs may share. Chains are resolved once, by `validate`. A delta blob's base is resolved per segment by
  `Index::delta_params`, a bounds-checked 16-byte decode. The decode path neither parses nor allocates per
  chunk, apart from growing scratch.
- Extents are aligned by `validate`, so a view never meets a split element at the end of a blob range. An
  element can straddle two chunks only under an odd `chunk_size`. Spec section 8 defines bytes through the
  blob, so such a file is valid. That one element is served alone, on the calling thread. A delta blob's
  chain block is even, so it never has one, and the reader refuses one as a broken index.
- The tensor table has one row per `TENS` record, built at open, and members slice it. Records may share
  ranges, so a table per member would multiply two counts from the file.
- On `Err`, `dst` contents are unspecified. Methods take `&self`; `Container` is `Send + Sync`.
- `Container::index`, `read_stored`, `same_as` and `Member::record` are crate-private, for `write::merge`,
  which copies stored chunks without decoding them. They are not a way around the checks of this module.
- `ReadCache` is the one place decoded bytes outlive a call, and only when a caller builds one. A window is
  `Member::chunk_window`: the part of one chunk one extent shows, viewed, so at most 2 × the largest chunk.
  The cache owns no chunk logic and no decode path: a miss fills a run of up to `readahead_windows` absent
  windows of one member (0 = threads, so one batch) with one `Member::read_many`, so the engine's bound and
  the container's `Verify` apply unchanged: hashes are checked on every fill and never on a hit. No lock is
  held while decoding. Concurrent requests for a window decode it once: the others wait on a `Condvar`. A
  failed run is retried as the missed window alone, and a window that still fails is read directly with
  `Member::read_at`, so a cached read fails exactly where a direct one does. A failed fill caches nothing
  and its waiters retry it themselves. Memory: resident windows (LRU by bytes) and fills in flight together
  hold at most `max_bytes`, plus one window per concurrent miss, since the missed window always proceeds
  and read-ahead is admitted only within the budget left; then the engine's 3 × threads × chunk per fill.
  When a fill lands with the budget held by others, its missed window is kept (as the only resident window)
  so sub-window readers do not re-decode it, and its read-ahead is served but dropped.
- No reservation of the cache is held while its thread can run unrelated rayon tasks. A rayon worker of
  another pool that enters the container's pool runs its own pool's tasks while it waits, and one of them
  could wait for a window that same fill holds. So a fill from a rayon worker, when the engine has a pool,
  decodes on a scoped helper thread and the caller blocks in a plain join (`Inner::decode_run`); other
  callers decode on their own thread. A helper that cannot start is an `Error::Io`, after which the window
  is read directly like any failed fill; a panic on the helper is resumed on the caller. `Member` and the
  scratch pool need no such rule: nothing is held across a `ThreadPool::install` (a `Lease` lives inside
  one job, and its holder waits on nothing).
  `max_bytes` is checked at `new` to hold one widened largest chunk. Budgets need no accounting here: they
  were checked at open.
- Restore uses extents and the member hash only, never tensor ranges. A failed restore leaves no new file at
  the output and removes its temporary file. Reading writes nothing.
- `restore_file` writes to the path it is given and consults no member name. A caller that derives a path
  from a member name checks it with `format::check_restore_name` first. `Container::restore_all` is the one
  place in the library that does so: it checks every name before it creates anything, so a hostile name
  leaves no trace, then restores member by member; a later failure keeps the members before it. Frontends
  call it rather than repeat the rule.
- `decode_to` is the one read that hands out unverified bytes: each run reaches the caller's writer once
  its chunks passed their own hashes, before the member's BLAKE3 is known, and a mismatch is
  `VerifyFailed` after the last write and a flush. On any `Err` the caller discards what it received. A
  writer failure is `Io` with op `"write"` or `"flush"` and the member name as its path.

## Editing map

- Read strategy, batch limits, a new way to present chunk bytes: `batch.rs`.
- Which base chunk serves a delta chunk: `format::DeltaParams::base_chunk`, used by `BlobView::base_of` in
  `batch.rs` and by `format::validate`; never computed anywhere else.
- What is checked at open: `container.rs`.
- Restore semantics, `decode_to`, `restore_all`: `restore.rs` and spec sections 9 and 2.
- A change to the memory bound: `tests/read_memory.rs` asserts it.
- Window geometry (what one cached entry covers): `Member::chunk_window` in `member.rs`. Cache policy,
  read-ahead, eviction, waiting: `cache.rs`.
