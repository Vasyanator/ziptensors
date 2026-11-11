# Module: crates/ziptensors-core/src/write

## Purpose

The writer API: `encode_file`, `PackBuilder`, the mode policy, deduplication, delta blobs against a base
the caller names, verified commit, member copy between containers. Spec sections 2, 4, 6, 7, 7a, 8, 9 and
11 define what it may produce.

## Architecture

```text
add_file        symlink check, stamp, output check, open -> add_source
add_file_delta  as add_file, with the base named by member name (DeltaSpec::Added)
add_source      (crate-private) any RandomAccess input with a SourceStamp: scan -> normalize -> segments;
                  with a DeltaSpec: base bytes (a file reopened, or the kept source, or the caller's source)
                  -> BaseCatalog (from_plan of the base's scan, or the caller's from_member catalog)
                  -> one optional BaseRange per typed range -> segments carry it
ingest          calling thread: read in input order, BLAKE3 of the input, narrow, BLAKE3 of the blob
                  -> private rayon pool: encode_chunk, one result per one-shot channel
                  -> ordered sync_channel of those channels -> writer thread: append, rollback, seal
add_member_from source index -> copy stored chunks with positions rebased, on the calling thread
carry_member    (crate-private, seeded builder) the records of a member of the extended container,
                  over blobs already in the file: no byte read or written
finish          drop compression state -> stamps -> header with bit 0 if a delta blob exists -> index,
                  footer -> Container::open on the temporary file -> compare and decode every member
                  -> stamps again -> sync -> close -> rename -> sync the directory (Unix)
finish_with     (crate-private) finish by Commit::Replace (as finish, any generation) or Commit::Append:
                  stamps -> store index + footer in memory -> decode every member through SpliceSource
                  (file up to the new data, then index and footer) -> stamps -> index -> sync -> footer
                  -> sync; any failure -> Sink::fail (cut back, or TrailingBytes)
                  or Commit::InPlace (after move_data): stamps -> header with bit 0 as BlobTable says
                  -> store index + footer -> decode through SpliceSource -> stamps -> index, footer
                  -> sync -> one cut -> sync; any failure -> Sink::fail (DamagedInPlace, or
                  TrailingBytes for the cut)
```

The order of a blob: pick a view from a sample, pick a rotation (and, with a paired base under the identity
view, a delta) from a sample, stream, finalize identity before the last chunk is sent, then seal, or roll
back to the blob's start. A late widening mismatch rolls back and streams the range again with no view,
from a saved state of the file hash. A delta blob streams each raw chunk with the base bytes it pairs with,
read on the same reader thread from the base member's own source.

## Files and submodules

| Path | Role |
|---|---|
| `mod.rs` | Module list and re-exports. |
| `builder.rs` | `encode_file`, `EncodeOptions`, `PackBuilder`: checks, member records, order of the commit; `SourceStamp` (per-kind input checks) and `DeltaSpec` (which base), and the crate-private `add_source` every input goes through; `NameRule`; `Seed` (`of` for an append, `in_place` for a rewrite in place), `carry_member`, `keep_member_from`, `move_data`, `Commit` and `finish_with`/`fail` for a builder over an existing container (the `edit` module). |
| `ingest.rs` | The reader, compression and writer pipeline for one input, any `RandomAccess`. `Compressor`, `DeltaInput`, the delta trial and the base checks of spec section 7a. |
| `pairing.rs` | `BaseCatalog` (`from_plan` over a scan, `from_member` over a container's `TENS`/`EXTN`): base tensors mapped onto their extents, pairing by name (with `normalize_name`), length and dtype, one `BaseRange` per typed range. |
| `policy.rs` | The mode table of spec section 11, including `tries_delta`, `delta_candidate`, `with_delta`. |
| `dedup.rs` | `BlobTable`: every blob of the pack by id with its `BlobShape` (kind, `chunk_size`, length), and by BLAKE3 identity. |
| `commit.rs` | `Sink` over three targets (temporary file, append to a container, rewrite in place), its rollback per target, the moves and seal of an in-place rewrite (`copy_within`, `seal`), `Ending` and `Sink::finish`/`abandon`/`fail` (the one owner of how a file is ended), `store_index`/`write_index`, `verify_output` over any `RandomAccess`. `SinkFault` crash seam under `cfg(test)`. |
| `merge.rs` | `add_member_from`: chunk copy without recompression; a delta blob brings or reuses its base. |
| `report.rs` | `EncodeReport`, `MemberReport`, `LayerBytes`, `ClassBytes`, `VerifyReport`, `layer_report`, `ClassTable`. |
| `inputs.rs` | `collect_inputs`, `Inputs`, `check_output`: the walk of input files and trees into `(path, member name)`, and the refusal of an output the walk would pick up; the crate-private `walk` that `collect_inputs` and `tree` share. Independent of `PackBuilder`; with its tests. |
| `tests.rs` | Tests that need the corruption seam of `PackBuilder::fault`, or the crate-private input seam (`add_source`, `DeltaSpec::Member`), compiled only under `cfg(test)`. |

`tests/roundtrip.rs` covers everything a public caller can reach; `tests/write_memory.rs` holds the
allocator test, since a global allocator needs a binary of its own.

## Contracts and invariants

- Verify before commit has no opt-out. It opens the written file (under the append protocol, a splice
  view of it with the new index still in memory) through `Container::from_source` with default limits, so
  a file that needs raised limits to open is refused here. Each member's stated hash is compared with the
  hash taken from its source, then the member decodes to it. Failure leaves no new file at the output and
  removes the temporary one, or cuts an append back. What was at the output stays. The one error after the rename,
  `CommittedNotDurable` (directory sync failed, Unix), leaves the proven file at the output.
- `write::commit` owns the choice of ending; `io::TempFile` owns the replace ending only. Replace:
  everything is written to a temporary file, verified, renamed (`Ending::Rename`). Append
  (`Sink::append_to`, over a handle the caller opened and validated the container through, never the
  path again): the position and the frontier start at the container's length; nothing below the
  frontier is ever written (a rollback below it and a header patch are refused with `Io`); writes are
  positional, since readers share the handle (on Windows their reads move its cursor); the caller
  verifies the new generation through `io::SpliceSource` over `Sink::written_source` (the same handle)
  before writing its index, writes data and index,
  and `Ending::AppendFooter` syncs, writes the footer, syncs. Any failure before the second sync returns,
  `Sink::abandon`, and a drop all cut the file back to the frontier, which restores the old container
  byte for byte; a cut that fails is `TrailingBytes { valid_len: frontier }`. In place (`Sink::in_place`,
  over the caller's handle like an append, positional writes): position starts after the header; the
  first write of any kind (`Sink::seal`) zeroes the old footer and syncs, so a torn rewrite never ends in
  a footer whose chunks moved; `copy_within(src, dst, len)` requires `dst == position <= src` and copies
  forward through the caller's bounded buffer, reading each piece before writing it; a rollback only
  moves the position (bytes past it are dead data) and is refused below the end of the moved data;
  verification goes through the same splice view as an append before index and footer are written;
  `Ending::InPlace { final_len }` requires `final_len == position`, syncs, cuts the file once, syncs; a
  failed cut is `TrailingBytes { valid_len: final_len }` with the new container complete and durable.
  `Sink::fail` turns any other failure after the seal into `DamagedInPlace` and touches nothing.
- `store_index(index, index_offset, generation)` builds the stored index and its footer without writing;
  `write_index(sink, index, generation)` writes them into a file that is not yet visible
  (`GENERATION_INITIAL` for a fresh pack, the previous generation + 1 for a rewrite).
- Verification costs one sequential read of each member, however they share blobs: linear in the sum of
  member lengths, reported in `EncodeReport::verify`.
- Every extent covers a whole blob. So the decode work of spec section 4 is at most `file_len`, twice that
  for an extent over a delta blob, whose base range has its length and chunking, and the reader's bound
  cannot refuse a member. Skeleton is one blob per fragment for that reason; never share a skeleton blob
  across fragments of a member.
- Identity is BLAKE3 of canonical narrowed bytes, in `max` only, per whole blob, for every blob: tensors,
  skeleton fragments, opaque files. Widening is an extent view. There is no chunk or sub-tensor identity.
- A copied blob keeps its `blake3`, zero or not, and any mode reuses a blob by a non-zero one.
- The output depends only on the inputs, their order, and the mode, never on the thread count: trials read
  only the sample, and the writer takes chunks in submission order.
- Memory is at most 6 × threads × chunk plus the index, in Rust allocations, whatever the tensor size:
  a window of 2 × threads chunks, counting the one being read and the one being written, each one buffer of a
  chunk; a second one for each of `threads` chunks in compression; and `threads` workspaces of at most 2.95
  chunks. Raw buffers are freed and never pooled, as a free list would sit beside the stored chunks. zstd's
  contexts allocate in C: a few MiB per thread, constant. Compression state is dropped before
  verification, which adds the reader's bound.
- Streaming delta blobs raises the bound to 9 × threads × chunk: each chunk in the window holds its base
  chunk too until it is encoded (2 × threads × 2), the chunks in compression add their stored buffer
  (threads), and each workspace keeps a residual chunk besides the rest (threads × 3.95). Workspaces keep
  their size, so the raised bound holds for the rest of the pack once a delta blob was streamed. The
  builder also keeps, per member added from an input, its stamp and extents and where its bytes are: a
  file's path (reopened when named as a base, no handle kept) or, for a source that is no file, the
  source itself, dropped by `finish` before verification; about the size of its `EXTN` records, so that
  a later input can name it as a base.
- A panic in a compression job costs that chunk only: the writer reports it as a job with no result, and
  the job's workspace goes back to the pool on unwind (`WorkspaceLease`), so later jobs never wait for it.
- A failed `add_file` or `add_member_from` adds no member and leaves the builder usable: blobs sealed so far
  stay as complete unreferenced blobs, and the open one is rolled back. A failed write to the output
  poisons `Sink`, and everything after it fails.
- Every input is stamped (`SourceStamp`). A file: length and modification time, re-checked after it is
  read, before the index, and after verification; a changed or deleted file is `InputChanged`, and so is a
  file that shrinks during the read. A source that is no file (`SourceStamp::Member { len, blake3 }`, a
  container member): its bytes are fixed while its container is open, so it is compared once, right after
  the read, with what was read; a mismatch is `VerifyFailed` naming the member and the member is not
  added. At commit it has nothing to re-check; the commit's decode proves the pack against the hash of
  what was read.
- `add_file`/`add_file_delta` are thin wrappers: they check name and base, stamp the file (symlink
  refusal), check `same_file` against the output, open it, and call `add_source`. Error precedence of
  `add_file` is therefore name, base, file. `add_source` itself does no symlink or output check.
- Parsers are hints. Bytes outside normalized typed ranges are skeleton, so extents partition the member
  whatever a parser said.
- A symbolic link is refused with `Error::SymlinkInput`, never followed and never skipped here.
- A builder over an existing container (`PackBuilder::with_sink` with `Sink::append_to` and
  `Seed::of(container, header)`) starts from that container's `CHST`/`CHAN`/`BLOB`/`CHNK`/`PARM`
  records with their ids and offsets (`IndexBuilder::seed`), and from every blob in `BlobTable`, so new
  inputs deduplicate against it and pair with its blobs. Its members come over by `carry_member`, which
  writes records only. The header stays: `finish_with(Commit::Append)` refuses a delta blob while the
  kept header has bit 0 clear, and never patches it. The verification runs before any byte of the new
  index is on disk. Report accounting of carried members counts a blob's stored bytes with the first
  carried member that reaches it, as `layer_report` does, and blobs no member reaches any more come in
  through `unreferenced_table`, so `EncodeReport` equals `layer_report` of the result. One case is not
  balanced: a blob no carried member reaches that a new input reuses by identity counts as reused, and
  its stored bytes are counted nowhere (the same holds today for a blob of a failed add reused later).
  Report only; the file is unaffected.
- `keep_member_from` is `add_member_from` for a member an edit keeps, optionally plus a registration of
  the copied member, with a source of its bytes, as a base a later input can name (`DeltaSpec::Added`). A
  rewrite uses it for every kept member, with a source for one named as a delta base, so every base of a
  rewrite is in this pack's own ids.
- Member names (`NameRule`): every name must be unique in the pack. A name the pack introduces
  (`add_file`, `add_file_delta`, `add_member_from`, `add_from_file` under `NameRule::New`) must also pass
  `check_restore_name`. A member an edit keeps under its source name (`carry_member`, `keep_member_from`,
  `add_source`, `add_from_file` under `NameRule::Kept` for a `Replace`) need not: its valid container
  allowed it, and keeping a name restores nothing. `add_source` takes its name as kept; a new name reaches
  it only through `add_from_file`, which checked it.
- `PackBuilder::fail(error)` is the failure ending a caller uses after its own error between adds: it is
  `Sink::fail`, and so the only way an append that cannot be cut back reports `TrailingBytes`, and an
  in-place rewrite that began writing `DamagedInPlace`.
- An in-place builder (`with_sink` over `Sink::in_place` with `Seed::in_place(container, header, kept,
  rebase)`) holds only the kept blobs, renumbered in their old order (`IndexBuilder::seed_subset`), so a
  base keeps a lower id than its delta blob; their chunk offsets come from the caller's move plan, and
  `carry_member` renumbers a carried member's extents to match. The caller moves the data first
  (`move_data`), then adds members; ingest never reads the source container, whose offsets are stale.
  `finish_with(Commit::InPlace)` rewrites the header with bit 0 exactly as the kept and new blobs say.
- `inputs::collect_inputs` is the one owner of how frontends turn roots into members: links are skipped
  and counted (never followed), `*.incomplete` files skipped and counted, names relative to their root
  with `/`, sorted per root, UTF-8, checked by `check_restore_name` and unique (a duplicate is
  `DuplicateMember`, as `PackBuilder` says it), every I/O failure an `Io`. `inputs::check_member_names`
  is that name check over any list, for frontends that copy members (a merge checks every source's
  names before it starts the output). A duplicate in a write request is `DuplicateMember`; one in the
  index of a file is damage, `MalformedIndex` for `MEMB`. It reads no file
  contents and knows nothing of packing. `inputs::check_output` refuses, as `OutputIsInput`, an output a
  walk of the same roots would pick up, existing or not. Both go through `inputs::walk`, the one owner of
  the walk rules (no link entered, name order, links, `*.incomplete`, names that are not UTF-8 and
  special files reported as such), which `tree` uses too; each caller decides what to do with each kind.
- Mode choice never changes the meaning of a file.
- An input that is the output, by `io::same_file`, is `OutputIsInput`.
- Samples are the first 64 KiB of a range, which the stream reads next, so a disk does not seek for them.
- `ClassTable::add_extent` is the one owner of the accounting of an extent: `copy_member` and `layer_report`
  both go through it. A base blob that a copy brings for a delta blob and no extent reaches is
  `ClassTable::add_base`, stored bytes that are all overhead, in both too, and it counts in that
  member's `blobs_added`. Blobs no extent reaches at all, left by a failed `add_file`, `add_file_delta`
  or `add_member_from`, are added at commit by `report::unreferenced_table` over the verified index, the
  function `layer_report` ends with, so `EncodeReport` accounts for the whole data region and equals
  `layer_report` of its file.

## Delta blobs

- Explicit pairing only: `add_file_delta(path, member, base_member)`, where the base is a member this
  builder added from a file (inside the crate: added through `add_source` from any input,
  `DeltaSpec::Added`; or a base that is no member of this builder, `DeltaSpec::Member { name, source,
  catalog }` with a catalog from `BaseCatalog::from_member`, whose blob ids must be this pack's: a pack
  seeded with the base's index. Such a base has no stamp; the commit's decode proves the result). Only `max` stores delta blobs; other modes check the base name and then act as
  `add_file`. The base file is read again for its bytes, so its stamp is checked before and after, and the
  commit decodes every member against its source hash as always: a wrong base costs a failed commit, never
  a wrong file.
- Pairing (`pairing.rs`) is a hint: same name, else the same name after `normalize_name` (one publication
  prefix dropped, a norm's `scale` read as `weight`); always the same length and dtype; the base tensor must
  start inside an identity extent. A name two base tensors normalize to pairs with nothing.
- A delta blob needs, by spec section 7a and checked in `ingest.rs` against `BlobTable` before any trial:
  a kind 0 base, the same `chunk_size`, a chunk-aligned `base_offset`, the range inside the base. Views are
  identity only, and only `delta_xor` of the float width (`policy::delta_candidate`), so F16, BF16, F32 and
  F64.
- The trial runs on the 64 KiB sample: plain, rotated, delta, rotated delta, in that order on ties. Identity
  is decided first at the end of the stream, so a tensor equal to its base becomes an extent over the base
  blob and no delta blob.
- Required feature bit 0 is written into the header at `finish`, by `Sink::patch_header`, exactly when
  `BlobTable` recorded a kind 1 blob, which includes delta blobs of a failed `add_file_delta` that stay as
  unreferenced records and delta blobs copied by `add_member_from`.
- A copy of a delta blob first resolves its base: a blob this pack holds by the source's id or by identity
  serves only if it is kind 0 with the same `chunk_size` and length; otherwise the base is copied, with the
  lower id. The base member itself need not be copied. Identity reuse of a blob for an extent still accepts
  any blob with the same canonical bytes.

## Policy cells the spec table does not name

- `F64` is a float: planes of 8, store or `h`, rotation trial in `max`, no view.
- `BOOL` and every integer are stored, as is any dtype id this build does not know.
- A ggml block type without a field table (`Q4_0`, `IQ*`, `MXFP4`, …) is stored if it ever reaches the
  writer as a range. No parser hands one out: `profile/gguf.rs` leaves those bytes in the skeleton.
- A tensor exact under both views takes the `f16` view. Both canonical forms have the same width, so the
  choice is only about plane entropy and sharing, and a trial is not worth its code. Measured over all
  20.26 GB of `F32` ranges of the acceptance set: 86.3 MB in 691 tensors are exact under both (0.43%).
  They store in 2.34 MB as `f16` and 2.56 MB as `bf16`, and 2.31 MB with the better of the two per tensor,
  and the fixed order misses 0.68 MB of `BF16` siblings in `max`.

## Editing map

- Policy: `policy.rs` and spec section 11 together.
- Samples, windows, memory: constants and notes at the top of `ingest.rs`, and `tests/write_memory.rs`.
- Which tensors pair with a base: `pairing.rs` (`NAME_PREFIXES`, `normalize_name`). Which dtypes try a
  delta: `policy::delta_candidate` with spec section 11. Base checks and the delta trial: `ingest.rs`
  (`base_refusal`, `Reader::pick_identity`).
- Commit protocols: `commit.rs` (`Sink`, `Ending`), `PackBuilder::finish_with` (`Commit`),
  `io::TempFile` for the replace ending, spec invariant 12 (a)/(b).
- A builder over an existing container: `Seed`, `carry_member`, `move_data`, `finish_with` in
  `builder.rs`; the edit orchestration that uses it is `../edit/`.
- Input kinds and their checks: `SourceStamp` in `builder.rs`; a new kind of input goes through
  `PackBuilder::add_source`, never a second ingest path.
- Which files a walk takes and how they are named: `inputs.rs`.
- Report fields: `report.rs`, which keeps the balance of layers that `tests/roundtrip.rs` asserts.
