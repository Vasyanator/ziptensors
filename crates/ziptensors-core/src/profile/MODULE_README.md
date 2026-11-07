# Module: crates/ziptensors-core/src/profile

## Purpose

Finds tensor data in a source file, so that the writer can pick transforms. The result is a hint: it
picks codecs and names tensors and is never trusted for restore.

## Architecture

`scan` chooses one parser from the content: `GGUF` magic, ZIP local header magic, a safetensors length
word followed by `{`, then a structural check of the top-level ONNX fields. A parser returns a raw
`RangePlan`, and the writer passes it through `normalize` before using it.

## Files and submodules

| Path | Role |
|---|---|
| `mod.rs` | `scan`, dispatch, the internal `Stop` outcome, parser caps shared across formats. |
| `reader.rs` | `Reader`: bounded cursor over `RandomAccess`. All parsers read through it. |
| `json.rs` | Pull JSON reader for the safetensors header. No tree is built. |
| `safetensors.rs`, `gguf.rs`, `onnx.rs`, `pt_zip.rs` | One parser per format. |
| `width_trial.rs` | Sample-based detection of element width, for bytes no parser can type. |
| `plan.rs` | `normalize`, and `MIN_RANGE_BYTES`. |
| `tests.rs` | Tests across parsers, over the builders in `tests/common/fixtures.rs`. |

## Public entry points

- `scan(src) -> Result<RangePlan, Error>`: raw plan.
- `normalize(plan, file_len) -> RangePlan`: sorted disjoint ranges, trimmed to block, each at least
  `MIN_RANGE_BYTES`; rows the tensor table can hold.
- `width_trial(sample) -> u8`: 1, 2, 4 or 8.

## Contracts and invariants

- Detection is by content, never by extension.
- Any parser failure yields an opaque plan, never an `Error` or a panic. `Err` from `scan` means a read
  of the source failed. A parser reports this by `Stop::Io`, and give-up by `Stop::Malformed`.
- No allocation or read follows a declared size unless the bytes are present and the size is under a
  cap. Recursion and counts are capped. Header walks have a byte or field budget.
- `ranges` drive compression; `tensors` become the table and survive even when a range is dropped, so a
  tensor under `MIN_RANGE_BYTES` stays reachable by name. `normalize` removes only rows the spec forbids
  (outside the file, over 16 dimensions, repeated name) or past the name budget. The result never makes
  an index fail validation, whatever the source file says.
- A declared dtype is kept as declared. A name may lie about content: packed NF4 weights and JSON blobs
  tagged `U8` stay `U8`.
- `.pt` storages have no type without pickle, so `F16`, `F32`, `F64`, `U8` stand for element widths 2,
  4, 8, 1 from `width_trial`. Rows are flat and named after the ZIP entry.
- ONNX `external_data` is ignored and never opened. Varint and unpacked fields stay in the skeleton.
- A compressed storage or `data.pkl`, ZIP64 marker values, disagreeing ZIP64 records and legacy pickle
  files are opaque. Other compressed entries, such as TorchScript `code/`, stay in the skeleton. A
  plain ZIP64 record that repeats the classic end record is
  accepted, since the `PyTorch` writer always emits one. There is no pickle parser.
- Every ggml type of the dtype registry gets a GGUF row. Only plain element types and the four types
  with a field table also get a range; the bytes of every other block type (`Q4_0`, `IQ*`, `MXFP4`, …)
  stay in the skeleton, since a typed blob for them is an unmeasured ratio decision. A ggml type with
  no `Dtype` gets neither. A row that is not whole blocks makes the whole plan opaque.
- Block geometry (elements and bytes per block) belongs to `chain`, through `Dtype::block_geometry`;
  `gguf.rs` owns only the ggml type numbers.
- This module knows nothing of the container format.

## Editing map

- New source format: a parser here, a branch in `dispatch`, a profile id in `chain.rs`, a `Profile`
  variant in `types.rs`, and the spec.
- New ggml type: a `Dtype` id and name in `types.rs`, its `BLOCK_*` geometry in `chain.rs`, the spec,
  then `ggml_type_of` in `gguf.rs`. A field table, and so a range, is a separate ratio decision.
- Bounds of hostile input: the `MAX_*` constants at the top of each parser and in `mod.rs`.
