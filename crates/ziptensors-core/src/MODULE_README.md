# Module: crates/ziptensors-core/src

## Purpose

The library behind every frontend: container format, codecs, source format parsers, reader, writer.
Frontends call its public API and own no decisions.

## Architecture

Dependency direction is in the root `ARCHITECTURE.md`. In short: `read` and `write` compose `format`,
`codec` and `io`; `write` also uses `profile`, and `read`, to prove its output before commit and to copy
members; `read` never uses `write`; `error` and `types` are leaves.

## Files and submodules

| Path | Role |
|---|---|
| `lib.rs` | Module list, lint policy, public re-exports. |
| `error.rs` | `Error`, the only public error type, and its stable `ErrorCode`. |
| `chain.rs` | Id registry except dtypes, `Step`, `DeltaOp`, `Chain`, `View`, field tables, `BlockGeometry` of every ggml block type. |
| `types.rs` | `Dtype` with its ids, names and geometry, `Profile`, `Mode`, `Verify`, `TypedRange`, `TensorInfo`, `RangePlan`. |
| `io/` | `RandomAccess`, `FileSource`, and the crate-private `TempFile` of both commits. |
| `format/` | Header, footer, index records, limits, structural validation. |
| `codec/` | Chunk encode and decode, layouts, widening views. |
| `profile/` | Source format recognition. |
| `read/` | Reader API, and the opt-in decoded-window cache `ReadCache`. |
| `write/` | Writer API. |

## Contracts and invariants

- `Error`, `ErrorCode`, `Chain`, `Dtype`, `Mode`, `Verify`, `TypedRange`, `RangePlan` and `RandomAccess`
  are shared contracts of all modules and frontends. Additions are fine, changes to what exists need
  every user reviewed.
- Wire ids in `chain.rs` and `types.rs` match the spec registry. No other module defines an id.
- `ErrorCode` values are ABI: they never change and are never reused.
- Nothing panics on file contents. No `unsafe` outside `codec/kernels_x86.rs`: `lib.rs` denies it and
  that module alone lifts the denial.

## Editing map

- New failure mode: a variant, a new `ErrorCode` and an `Error::code` arm in `error.rs`, then the status
  mapping in `ziptensors-capi`.
- New step, view, stream codec, blob kind, profile or feature id: `chain.rs` and the spec together. A
  new delta operation is a `DeltaOp` variant with its id, then `codec/delta.rs`.
- New dtype id: `types.rs` (id, name, geometry) and the spec together; a block type also needs its
  geometry in `chain.rs`. Frontends take names from `Dtype::name` and block sizes from
  `Dtype::block_geometry`, and `ziptensors-capi` pins a `ZT_DTYPE_*` constant to each id.
- New source format: `profile/`.
