# Module: crates/ziptensors-core/src/write

## Purpose

The writer API: `encode_file`, `PackBuilder`, the mode policy, deduplication, delta blobs against a base
the caller names, verified commit, member copy between containers. Spec sections 2, 4, 6, 7, 7a, 8, 9 and
11 define what it may produce.

## Architecture

```text
add_file        symlink check, stamp -> scan -> normalize -> segments (typed ranges + skeleton fragments)
add_file_delta  as add_file, plus: base stamp -> scan of the base file -> BaseCatalog over the base's
                  extents -> one optional BaseRange per typed range -> segments carry it
ingest          calling thread: read in file order, BLAKE3 of the file, narrow, BLAKE3 of the blob
                  -> private rayon pool: encode_chunk, one result per one-shot channel
                  -> ordered sync_channel of those channels -> writer thread: append, rollback, seal
add_member_from source index -> copy stored chunks with positions rebased, on the calling thread
finish          drop compression state -> stamps -> header with bit 0 if a delta blob exists -> index,
                  footer -> Container::open on the temporary file -> compare and decode every member
                  -> stamps again -> sync -> close -> rename -> sync the directory (Unix)
```

The order of a blob: pick a view from a sample, pick a rotation (and, with a paired base under the identity
view, a delta) from a sample, stream, finalize identity before the last chunk is sent, then seal, or roll
back to the blob's start. A late widening mismatch rolls back and streams the range again with no view,
from a saved state of the file hash. A delta blob streams each raw chunk with the base bytes it pairs with,
read on the same reader thread from the base member's own source file.

## Files and submodules

| Path | Role |
|---|---|
| `mod.rs` | Module list and re-exports. |
| `builder.rs` | `encode_file`, `EncodeOptions`, `PackBuilder`: checks, member records, order of the commit. |
| `ingest.rs` | The reader, compression and writer pipeline for one source file. `Compressor`, `DeltaInput`, the delta trial and the base checks of spec section 7a. |
| `pairing.rs` | `BaseCatalog`: base tensors mapped onto their extents, pairing by name (with `normalize_name`), length and dtype, one `BaseRange` per typed range. |
| `policy.rs` | The mode table of spec section 11, including `tries_delta`, `delta_candidate`, `with_delta`. |
| `dedup.rs` | `BlobTable`: every blob of the pack by id with its `BlobShape` (kind, `chunk_size`, length), and by BLAKE3 identity. |
| `commit.rs` | `Sink` (append position and rollback over `io::TempFile`), index write, full verify. |
| `merge.rs` | `add_member_from`: chunk copy without recompression; a delta blob brings or reuses its base. |
| `report.rs` | `EncodeReport`, `MemberReport`, `LayerBytes`, `ClassBytes`, `VerifyReport`, `layer_report`, `ClassTable`. |
| `tests.rs` | Tests that need the corruption seam of `PackBuilder::fault`, compiled only under `cfg(test)`. |

`tests/roundtrip.rs` covers everything a public caller can reach; `tests/write_memory.rs` holds the
allocator test, since a global allocator needs a binary of its own.

## Contracts and invariants

- Verify before commit has no opt-out. It opens the temporary file through `Container::open` with default
  limits, so a file that needs raised limits to open is refused here. Each member's stated hash is compared
  with the hash taken from its source, then the member decodes to it. Failure leaves no new file at the
  output and removes the temporary one. What was at the output stays. The one error after the rename,
  `CommittedNotDurable` (directory sync failed, Unix), leaves the proven file at the output.
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
  builder also keeps, per member added from a file, its path, stamp and extents, about the size of its
  `EXTN` records, so that a later file can name it as a base.
- A panic in a compression job costs that chunk only: the writer reports it as a job with no result, and
  the job's workspace goes back to the pool on unwind (`WorkspaceLease`), so later jobs never wait for it.
- A failed `add_file` or `add_member_from` adds no member and leaves the builder usable: blobs sealed so far
  stay as complete unreferenced blobs, and the open one is rolled back. A failed write to the output
  poisons `Sink`, and everything after it fails.
- Inputs are stamped by length and modification time, and re-checked after a file is read, before the
  index, and after verification: a changed or deleted file is `InputChanged`. A file that shrinks during the
  read is `InputChanged` too.
- Parsers are hints. Bytes outside normalized typed ranges are skeleton, so extents partition the member
  whatever a parser said.
- A symbolic link is refused with `Error::SymlinkInput`, never followed and never skipped here.
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
  builder added from a file. Only `max` stores delta blobs; other modes check the base name and then act as
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
- Commit protocol: `commit.rs`, `PackBuilder::finish` and spec invariant 12.
- Report fields: `report.rs`, which keeps the balance of layers that `tests/roundtrip.rs` asserts.
