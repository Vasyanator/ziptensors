# Module: crates/ziptensors-core/src/format

## Purpose

Bytes on disk to validated typed records and back, for everything except chunk contents. The spec is the
authority; code here cites it.

## Architecture

No file I/O and no compression: callers pass bytes. Ids, `Chain`, `Step`, `View` and stream counts come
from `crate::chain`, dtypes and profiles from `crate::types`; this module defines none.

The reader's flow, in spec section 9.1 order:

1. `parse_header`, `parse_footer`, then `Footer::check_layout`, before the index is read.
2. The caller decompresses the stored index into `index_raw_len` bytes, then `Footer::check_index`.
3. `Index::parse`, then `validate` with the parsed header. Only after this may records be used to reach
   data. `validate` returns the chains by `chain_id`, resolved once, so the decode path never re-parses
   `CHST`. A reader resolves a delta blob's base with `Index::delta_params` per extent segment it reads.
4. `check_decode_budget` before any data read, `check_restore_name` before a restore joins a path.

The writer's flow is `IndexBuilder`, `serialize`, compression by the caller, `Footer::for_index`,
`Footer::to_bytes`, with `Header::current` and `Header::to_bytes` at the start of the file. A writer
of delta blobs adds each `DeltaParams::encode` with `IndexBuilder::add_params` and sets
`FEATURE_DELTA` in the header it finally writes.

## Files and submodules

| Path | Role |
|---|---|
| `mod.rs` | Module list and re-exports. Submodules are private. |
| `header.rs`, `footer.rs` | The two 64-byte structures. |
| `limits.rs` | `Limits`, the caller's bounds, the format's 1 GiB ceiling, the decode work floor. |
| `records.rs` | One struct per record, with its byte layout. `DeltaParams`, the `PARM` entry of a delta blob, and `base_chunk`, the one definition of which base chunk a delta chunk pairs with. |
| `index.rs` | Section table rules, record arrays, the `PARM` heap, bounded accessors. |
| `validate.rs` | Invariants across records, the decode budget, restore name safety. |
| `validate_tests.rs` | Reject tests per invariant, cost and hostile-input tests. |
| `builder.rs` | Serialization of an index. |
| `wire.rs` | Field access at compile-time-checked offsets, error constructor. |
| `testkit.rs` | Test-only fixtures. |

## Contracts and invariants

- Every invariant of spec section 9 that needs no chunk data is enforced here, with checked arithmetic,
  except the rest of invariant 10, which is decode-time (`codec`, `read`), and invariant 12, the
  writer's commit protocol (`write`). A stream codec id is seen only when decoding, so `codec` rejects it.
- `Index` values are unvalidated until `validate` returns `Ok`. Its accessors are bounds-safe either way.
- Unknown ids anywhere in `CHST` and `EXTN` reject the file, referenced or not. Dtypes and profiles are
  hints with total mappings, so records carry them typed.
- `PARM` is kept with either flag, its heap rule checked by `Index::parse`. `validate` ties it to the
  header: required feature bit 0 is set exactly when a kind 1 blob exists, and `PARM`'s required bit
  exactly then (spec section 7a). An optional `PARM` in a file without delta blobs is unused.
- Delta blobs (invariant 13): a base with a lower id, of kind 0, the same `chunk_size`, a
  chunk-aligned `base_offset`, the range inside the base; a delta step in the chain exactly on kind 1.
  One pass over `BLOB` suffices because bases come first.
- Failures are `Error::MalformedIndex`, `UnsupportedVersion`, `UnsupportedFeature`, `UnknownCodec`,
  `BudgetExceeded`, `NotZiptensors`, `MemberNotFound` or `UnsafeMemberName`, never a panic. An unknown
  required section is `MalformedIndex` for `"index"`, with the tag in the detail.
- No allocation is sized by a declared count before it is checked against the actual section length.
- Validation cost never multiplies two counts from the file, though ranges may be shared between owners:
  O(n log n) in the index, plus `Limits::max_name_work_bytes` for names. That limit is a reader defence, for memory as well, since a reader's tensor table copies names,
  that the spec does not state: without a bound on names, many records over one heap are quadratic.
- `check_decode_budget` also bounds the summed decode work of the members it is given, the MAY rule of spec
  section 4, from a prefix sum over `EXTN`, so that members sharing extents cost no product. An extent over
  a delta blob also counts the base chunks its chunks pair with, exactly, since a short delta blob may
  sit on full base chunks. There is one
  floor per call, so a pack of many tiny members gets no allowance per member. An extent over a whole blob
  costs that blob, so members of such extents have work of at most their summed `file_len`, and never meet
  the bound.
- `IndexBuilder` enforces no invariant, by design: writers prove the result by parsing it, and hostile
  fixtures are built through it. The serialized layout is deterministic.

## Editing map

- New id: `chain.rs` and the spec together, then `validate.rs`.
- New section: `index.rs`, `builder.rs`, `validate.rs` and the spec together.
- New record field: `records.rs` and its golden test, then `validate.rs`.
- New reader bound: `limits.rs`.
