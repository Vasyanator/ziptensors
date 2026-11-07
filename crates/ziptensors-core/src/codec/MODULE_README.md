# Module: crates/ziptensors-core/src/codec

## Purpose

Canonical bytes to one stored chunk and back, in memory. No file I/O, no knowledge of the index beyond the
chain and lengths it is given. `Chain`, `View`, field tables and stream codec ids come from
`crate::chain`. Spec sections 6 and 7 define the bytes.

## Architecture

```text
encode_chunk: raw -> rotl1 (rotated copy) -> gather one stream at a time -> compress or store -> out
decode_chunk_into: stored -> parse prologue -> route -> rotr1 -> XXH3 check
encode_delta_chunk: raw, base -> residual (workspace copy) -> encode_chunk's path with chain.residual()
decode_delta_chunk_into: stored -> decode_chunk_into's path with chain.residual(), no check
                         -> combine with base, in dst -> XXH3 check of the target
```

A delta chunk (spec section 7a) is an ordinary chunk of its residual, so every route below serves it.
The caller decodes the base chunk first and passes its first `dst.len()` canonical bytes; the codec
never sees more than one chunk. The plain entry points refuse a chain with a delta step and the delta
entry points one without, so a caller cannot skip the base or invent one.

`decode_chunk_into` has three routes, chosen per chunk from its stream words. All produce the same bytes.

| Route | When | Scratch |
|---|---|---|
| direct | `whole`, or no stream compressed (stored planes merge straight into `dst`) | none |
| in place | `planes` with exactly one compressed stream | none |
| scratch | everything else, including every `fields` chunk with a compressed stream | sum of compressed stream raw lengths, at most the chunk |

Stored streams are read where they lie in `stored`, never copied to scratch.

## Files and submodules

| Path | Role |
|---|---|
| `mod.rs` | Module list and public entry points. |
| `chunk.rs` | Prologue, routes, full chunk encode and decode, plain and delta. `MAX_CHUNK_BYTES`, `ChunkMeta`, `locate_error`. |
| `delta.rs` | The delta operations of spec section 7a: `residual` and its inverse `combine`, bytewise and in place. |
| `kernels.rs` | Scalar gather, merge and rotation, and `Isa`, the choice of kernels. |
| `kernels_x86.rs` | Runtime-detected AVX2 kernels: the only `unsafe` in `core`. Holds the proof of risk R1. |
| `widen.rs` | Widening views and exact narrowing. |
| `stream.rs` | One stream: `StreamPolicy`, zstd parameters, frame checks, exact output length. |
| `workspace.rs` | `Workspace`, the reusable buffers and zstd contexts of one worker. |
| `test_support.rs` | Test-only seeded generators and guard bands. |

## Contracts and invariants

- A stored chunk depends only on its raw content, chain and policy, never on workspace history or
  kernels. Scalar and AVX2 kernels are byte-identical.
- Every length comes from the chain and the caller's raw length. No buffer is sized from a file value,
  so hostile bytes cannot grow memory use. A raw length is at most `MAX_CHUNK_BYTES`.
- A zstd stream is one-pass decoded into a slice of exactly its raw length. `WindowLogMax(24)` is set but
  limits only zstd's streaming API, so `stream.rs` checks the frame header itself: exactly one frame,
  content size equal to the raw length, no checksum, no dictionary, window at most 2^24. Legacy decoders
  are compiled out.
- In-place merge: the compressed plane sits in the last `n` bytes of `dst`, the write head stays at or
  below the read head at every step, and each step loads before it stores. Proof in `kernels_x86.rs`,
  asserted per step in debug builds.
- `decode_chunk_into` and `decode_delta_chunk_into` on any bytes give `Ok`, `ChunkCorrupt`,
  `UnknownCodec`, `ChecksumMismatch` or `DestinationSize`, or `Zstd` if a context cannot be created. A
  wrong base is a `ChecksumMismatch` of the delta chunk; checking the base chunk's own hash is the
  caller's, when it decodes that chunk. It never panics. Asserts in
  `kernels_x86.rs` guard soundness against this module's own bugs; file contents cannot reach them.
- `ChunkCorrupt` and `ChecksumMismatch` leave this module with blob and chunk 0; the caller sets them with
  `locate_error`.
- With hash verification on, `Ok` proves the output. A few bits inside zstd entropy tables have no effect
  on the output, so a flip there decodes to the correct bytes and is not an error.
- `encode_chunk` and `encode_delta_chunk` append to `out`, and keep `out` unchanged on error. A delta
  chunk's `xxh3_decoded` is of the target bytes, not of the residual.
- `Workspace` is `Send`, not shared: one per worker. Its buffers grow to exactly what the largest chunk
  needs, old allocation freed first, and never by `Vec` doubling: the bounds of both reader and writer
  count on their size.
- `unsafe` stays in `kernels_x86.rs`. Every block has a SAFETY comment, and `Avx2`, a zero-size proof made
  only by detection, is what keeps its methods safe to call.
- Views are integer definitions. F16C is not used, because it quiets signalling NaNs.

## Editing map

- New layout: `chain.rs` and the spec, then `chunk.rs` and `kernels.rs`.
- New stream codec: `chain.rs` and the spec, then `stream.rs` and `parse_streams` in `chunk.rs`.
- New view: `chain.rs` and the spec, then `widen.rs`.
- New delta operation: a `DeltaOp` variant in `chain.rs` and the spec, then `delta.rs`.
- New writer policy: `StreamPolicy` and `keeps_frame` in `stream.rs`.
- New vector kernel: `kernels_x86.rs` for whole vectors only, with the tail left to `kernels.rs`, and a
  case in the matrix tests of `kernels.rs` or `widen.rs`.
