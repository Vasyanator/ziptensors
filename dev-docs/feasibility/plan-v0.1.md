# Plan: ziptensors v0.1 spec outline, crates, work packages, documents, acceptance benchmark, risks

## Verdict
The decisions hold; one format suffices. Nothing refutes the architecture. Six refinements below (F1-F6) need your acknowledgement, F1 and F2 change stated numbers or wording, none changes a decision.
Central simplification: **a single-file container is a pack with one member.** One layout, one reader, one writer. Four layers: member (virtual file) -> extent (view over a blob range) -> blob (content-addressed, chunked) -> chunk (self-contained, copyable). The tensor table is metadata over `read_at`, so a parser error cannot affect restored bytes.
Recomputed total for `max` in one pack: 109 GB of 361.14 = **30.2%** (table in part 5). Plan: 11 packages, 5 rounds. Confidence: high on format and sequencing, medium on HDD behaviour of a deduplicated pack (R6, new risk).

## Evidence
### Flags against the decisions (kept as decided unless you say otherwise)
- **F1, memory bound wording.** "threads × chunk" and the 1.5 MiB RSS are not end-to-end. The prototype loads the whole compressed file first (`prototype-ztp/src/main.rs` `cmd_decode`: `std::fs::read(&a[0])`), so its read buffer is excluded (`codecs-result.md:56`). Real bound: read batch ≤ threads × chunk, plus scratch ≤ threads × chunk × view factor (2 for widening). So ≤ 3 × threads × chunk, 36 MiB at 12T. Same shape, new constant; the tests assert this.
- **F2, in-place is an optimization, not a rule.** It needs exactly one compressed stream (`codecs-result.md:49`). Not so for: the 16.38 GB community text-encoder finetune (low plane 6.23 bits, `:32`; 26% needs both planes coded), any widened F32 in `fast` (57% needs several planes), every field-split quantized blob. The scratch path is mandatory. Cost is small (joint 12.5 vs inplace 12.7 GB/s, `:48`).
- **F3, `fast` leaves bytes behind.** "Quantized stored" forgoes raw+h: Q8_0 3.69% at 12.2 GB/s, Q4_K 2.27% at 13.1 GB/s, NF4 2.6% (`codecs-result.md:18,20,25`). About 3.4 GB, 0.9 pp, `fast` only. `balanced` and `max` are unaffected. Kept as decided; one policy line to change.
- **F4, "core has no C deps".** `zstd-sys` compiles C, so a C compiler is needed per target. Read as "no C API exposure, no system library". Spike: plain `cargo check --target x86_64-apple-darwin` FAILS in zstd-sys (`cc -arch`); `cargo-zigbuild check --target x86_64-apple-darwin` passes without an SDK; linking needs `SDKROOT`.
- **F5, "no match finder" is approximate.** zstd has no such switch; the prototype sets minimal parameters at level 1 (`main.rs` `zstd_cached`: `WindowLog 17, HashLog 6, ChainLog 6, SearchLog 1, MinMatch 7, ZSTD_fast`). So this is writer policy, invisible to the format.
- **F6, Windows slowdown is refuted in magnitude.** I built the prototype with `zstd-sys/no_asm` (Windows path; 0 vs 4 asm-loop symbols by `nm`). 512 MiB BF16, inplace, 3 runs: 1T 3.72/3.73/3.64 vs 3.75/3.79/3.74 GB/s (about -1.5%), 12T within noise. Bit-exact 32.835% saved. Remaining Windows unknowns: mingw codegen and `seek_read`.
- **Dedup and widening interact.** The 13.53 GB SDXL saving exists only if F32 and F16 siblings share a blob. So identity is taken on **canonical (narrowed) bytes**, widening is an extent view, not a blob step. The set must also contain both siblings.
- **Packs are monolithic.** No in-place delete; `merge` rewrites by copying chunks. Separate per-root packs lose only 3.36 GB (`inventory-result.md:59`).
- **Byte-exact means content.** mtime, permissions and symlinks are not restored; 19 symlinks are skipped as in the inventory. Stated in the spec.

### Spike results (scratch only, removed)
- `zstd =0.13.4` builds for linux and windows-gnu. The cc line carried `-DZSTD_LEGACY_SUPPORT=1`, so use `default-features = false` to remove legacy decoders from untrusted-input reach; `bulk`, `CParameter`, `DParameter::WindowLogMax` still compile and run.
- `decompress_to_buffer`: output past capacity errors cleanly; two concatenated frames are accepted if they fit. So readers check exact output length, not frame count.
- `CARGO_BUILD_BUILD_DIR` is honoured by cargo 1.99.1 (a user-level cargo config may set a shared `build-dir`).
- Inventory: 44,903 tensor rows, 2.0 TB free on the HDD.

## Not verified
Native Windows and macOS runs; NVMe; PyO3 0.30.1 `PyBuffer` behaviour (from `ecosystem-result.md:81`, not in the local registry); pyo3 crate cross-check; `.pt`/ONNX fixture fidelity against real writers beyond inventory parsing. Claim "F16C quiets sNaN" is from Intel semantics, not tested; the exhaustive 65536-input test decides. Per-class targets scale sampled ratios to whole classes. `cargo-fuzz` and `maturin` are not installed.

## Recommended next step
Answer the 5 questions at the end, then run WP0.

### 1. Spec outline v0.1 (`spec/ziptensors-v0.1.md`)
All integers little-endian. No varints, no JSON. Tensor bytes are never byte-swapped; big-endian hosts work since kernels are byte-wise. Hashes: XXH3-64 seed 0; BLAKE3-256.

**File:** `[Header 64 B][data region: chunks, no padding][index: one zstd frame][Footer 64 B]`. Index at the end permits streaming write and later append. Reader reads the last 64 B, then the index, then data.

| Header @ | Field | | Footer @ | Field |
|---|---|---|---|---|
| 0 | magic `89 5A 54 53 0D 0A 1A 0A` | | 0 | u64 index_offset |
| 8 / 10 | u16 major=0 / u16 minor=1 | | 8 | u64 index_stored_len |
| 12 | u32 header_len=64 | | 16 | u64 index_raw_len (cap: default 256 MiB, spec max 1 GiB) |
| 16 | u64 required_features | | 24 | u64 index_xxh3 (raw) |
| 24 | u64 optional_features | | 32 | u64 generation (=1) |
| 32 | 24 B reserved, zero | | 40 | u64 reserved |
| 56 | u64 xxh3 of bytes 0..56 | | 48 / 56 | u64 xxh3 of 0..48 / magic `ZTS-END\0` |

**Versioning:** different major fails. Minor only adds ids/sections and never changes meaning. Unknown bit in `required_features` fails closed; unknown `optional_features` ignored. v0.1 sets no bits. Reserved required bits: 0 delta blobs, 1 derived blobs, 2 multi-volume.

**Index:** `u32 section_count, u32 0`, then 32-B entries `tag[4], u32 flags (bit0 required), u64 offset, u64 len, u64 count`, 8-byte aligned fixed-record arrays. Unknown required section fails, unknown optional skipped.

| Tag | Record | Bytes | Fields |
|---|---|---:|---|
| STRS | UTF-8 heap | - | referenced by (u32 off, u32 len) |
| CHAN | chain | 8 | u32 first_step, u16 step_count (≤8), u16 0 |
| CHST | step | 8 | u16 codec_id, u16 param0, u32 param1 |
| BLOB | blob | 80 | blake3[32], u64 raw_len, u64 first_chunk, u32 chunk_count, u32 chunk_size, u32 chain_id, u8 kind (0 chunked), u8 flags, u16 dtype_hint, u64 params_off, u32 params_len, u32 0 |
| CHNK | chunk | 24 | u64 file_offset, u64 xxh3_decoded, u32 stored_len, u32 0 |
| MEMB | member | 72 | u32 name_off, u32 name_len, u64 file_len, blake3[32] whole file, u64 first_extent, u32 extent_count, u32 first_tensor, u32 tensor_count, u8 profile, 3 B 0 |
| EXTN | extent | 32 | u64 virt_offset, u64 virt_len, u32 blob_id, u16 view, u16 0, u64 blob_offset |
| TENS | tensor | 32 | u32 name_off, u32 name_len, u64 virt_offset, u64 virt_len, u32 shape_off, u16 ndim (≤16), u16 dtype |
| SHAP | dims | 8 | u64 |
| PARM | TLV | - | reserved; blob-kind parameters |

Estimated raw index for the user's set about 20 MB: 350k chunks (8.4), 39k blobs (3.1), 90k extents (2.9), 45k tensors + names (about 5).

**Chunk (self-contained):** `[u32 stream word × S][stream bytes…]`. Word = `codec:8 | stored_len:24`. S and stream lengths derive from the chain and raw length, so they are never stored. Stream codec: 0 store, 1 zstd, others fail closed. zstd stream: content size flag on, no frame checksum, no dictionary, window ≤ 2^24. Chunk raw size is `chunk_size` except the last; `chunk_size` ≤ 8 MiB and a multiple of the layout block (Q8_0 gets `floor(1 MiB / 34) × 34`). Bytes depend only on raw content and chain.

**Id registry (u16 steps):** 0x0000 invalid; 0x0100 whole; 0x0101 planes(width 2/4/8); 0x0102 fields(table: 1 Q8_0, 2 Q4_K, 3 Q5_K, 4 Q6_K, from `ggml-common.h`); 0x0200 rotl1(width); 0x0301 view f16->f32; 0x0302 view bf16->f32; 0x0400-0x04FF reserved delta; 0x0500-0x05FF reserved derived; 0x8000+ private, rejected by default readers. Chain order: optional rotl1, then exactly one layout. Profile u8: 0 opaque, 1 safetensors, 2 gguf, 3 onnx, 4 pt_zip. dtype is a hint; unknown dtype is legal, unknown step is not.

**Widening views** are defined by integer operations, not by hardware: f16->f32 is the exact-value conversion with NaN payload shifted left 13 and signalling bit preserved; bf16->f32 is `x << 16`. Eligible only if every element in the extent round-trips bitwise.

**Blob reuse rule:** identity = BLAKE3 of canonical bytes (after narrowing, before rotl1 and plane split). A blob, or a chunk within it, may be copied into another container with its record unchanged, with no recompression, because a chunk depends on nothing else. Extents reference any blob range, so safetensors and GGUF tensors share one blob. Skeleton fragments and opaque files are blobs too (`whole`, zstd with LZ level 3). Ranges under 4 KiB stay in skeleton to limit per-blob overhead on small models; they stay reachable by tensor name.

**Set/pack:** same layout, N member records. One file, no external paths; a member names a path string only for `restore`.

**Later fit without a version bump:** delta = BLOB kind 1 + PARM base ref + 0x04xx + required bit 0, with acyclic references to lower blob ids. Derived = kind 2 + PARM recipe (source blob, quantizer id, version, exception blob) + bit 1. FUSE, memfd, application, ORT, llama adapters sit above `read_at`; no format change. onnx#8370, zTensor: id mapping. Append: new data after old index, then new index and footer with generation+1; old index becomes unreferenced bytes, which readers accept.

**Invariants (each has a reject test):**
1. Header/footer/index hashes valid; `index_offset + index_stored_len + 64 == file_len`; `index_offset ≥ 64`.
2. Decompressed index length equals `index_raw_len` ≤ cap; every record array `len == count × size` via checked math.
3. Every string/shape/section reference in range; names valid UTF-8; member and per-member tensor names unique.
4. Chunk byte ranges lie in `[64, index_offset)` and are pairwise identical or disjoint.
5. `first_chunk + chunk_count` in range; `chunk_count == ceil(raw_len / chunk_size)`; `raw_len` multiple of layout block.
6. Extents of a member are sorted and partition `[0, file_len)` exactly, no gaps or overlaps.
7. Extent blob range in blob; view length ratio exact and element aligned; kind 0 blobs only in v0.1.
8. Tensor range within `[0, file_len)`; ranges are never trusted for restore.
9. Unknown required feature, section, step, view, stream codec, or blob kind: reject.
10. Sum of `file_len` ≤ caller budget before any data read. Stream word sum + prologue == `stored_len`; decoded length exact; XXH3 on canonical decoded bytes (before view).
11. Member names for `restore-all`: relative, no `..`, no root or drive, no NUL. Never open a path from a file except the caller's root plus this checked name.
12. Writer: temp file in the output dir; full decode + BLAKE3 against source hash; fsync; rename. No opt-out in v0.1.

**Writer policy (no format impact):**
| Class | `fast` | `balanced` | `max` |
|---|---|---|---|
| F32/F16/BF16 | planes, per-stream store-or-h (gain < 0.5% stores) | + widen views | + rotl1 trial per blob |
| FP8 | whole, h | same | same |
| Q8_0/Q4_K/Q5_K/Q6_K | store | fields, h | same |
| U8 nibble, unknown int | store | store | store |
| skeleton, opaque | whole, zstd 3 | same | same |
| identity | file hash only | same | + BLAKE3 blob dedup |
`pack` defaults to `max`; `encode` to `balanced`.

### 2. Crates, API, threading
Workspace (edition 2024): `crates/ziptensors-core` (rlib: zstd, blake3, xxhash-rust, rayon, thiserror, tracing), `ziptensors-capi` (cdylib + cbindgen header), `ziptensors-py` (cdylib `abi3-py311`), `ziptensors-cli` (bin `ziptensors`, anyhow, clap, tracing-subscriber). `exclude = ["dev-docs"]`.

```
core/src: lib.rs error.rs types.rs
  io/source.rs      RandomAccess trait, FileSource (pread / seek_read)
  format/           header.rs footer.rs index.rs ids.rs validate.rs builder.rs
  codec/            kernels.rs kernels_x86.rs fields.rs widen.rs stream.rs chunk.rs workspace.rs
  profile/          mod.rs safetensors.rs gguf.rs onnx.rs pt_zip.rs width_trial.rs plan.rs
  read/             container.rs member.rs batch.rs restore.rs
  write/            builder.rs ingest.rs policy.rs dedup.rs commit.rs merge.rs
```

```rust
// read
pub struct OpenOptions { pub threads: usize, pub max_index_bytes: u64, pub max_total_decoded: u64, pub verify: Verify }
pub enum Verify { Chunk, Off }                    // default Chunk
impl Container { pub fn open(path: &Path, o: &OpenOptions) -> Result<Self, Error>;
                 pub fn members(&self) -> &[MemberInfo];
                 pub fn member(&self, name: &str) -> Result<Member, Error>;
                 pub fn single(&self) -> Result<Member, Error>; }   // Clone = Arc; Send + Sync
impl Member {   pub fn len(&self) -> u64;  pub fn tensors(&self) -> &[TensorInfo];
                pub fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), Error>;  // exact, else OutOfBounds
                pub fn decode_tensor_into(&self, name: &str, dst: &mut [u8]) -> Result<(), Error>; // dst.len() == tensor len
                pub fn read_many(&self, reqs: &mut [ReadRequest<'_>]) -> Result<(), Error>;  // elevator order
                pub fn restore_file(&self, out: &Path) -> Result<RestoreReport, Error>;  // temp + hash + rename
                pub fn verify(&self) -> Result<(), Error>; }
// write
pub fn encode_file(input: &Path, output: &Path, o: &EncodeOptions) -> Result<EncodeReport, Error>;
impl PackBuilder { pub fn new(output: &Path, o: &EncodeOptions) -> Result<Self, Error>;
                   pub fn add_file(&mut self, path: &Path, member: &str) -> Result<MemberReport, Error>;
                   pub fn add_member_from(&mut self, src: &Container, member: &str) -> Result<MemberReport, Error>; // no recompression
                   pub fn finish(self) -> Result<EncodeReport, Error>; }
```
On `Err`, `dst` contents are unspecified. Methods take `&self` and are safe from concurrent threads.

**Error (`thiserror`, `#[non_exhaustive]`):** `Io{op,path,source}`, `NotZiptensors`, `UnsupportedVersion{major,minor}`, `UnsupportedFeature{bits}`, `UnknownCodec{id}`, `MalformedIndex{section,detail}`, `BudgetExceeded{what,declared,limit}`, `ChunkCorrupt{blob,chunk,detail}`, `ChecksumMismatch{blob,chunk,expected,actual}`, `OutOfBounds{offset,len,file_len}`, `DestinationSize{expected,actual}`, `TensorNotFound{name}`, `MemberNotFound{name}`, `UnsafeMemberName{name}`, `VerifyFailed{member,expected,actual}`, `InputChanged{path}`, `Zstd{detail}`.

**C ABI (`ziptensors.h`):** opaque `zt_container`, `zt_member`; `zt_status` i32 (stable, exhaustive match from `Error`); every entry in `catch_unwind`.
```c
uint32_t  zt_abi_version(void);
zt_status zt_open(const char *utf8_path, const zt_open_options *o, zt_container **out); // o->struct_size first
zt_status zt_member_open(const zt_container *c, const char *name /*NULL=single*/, zt_member **out);
zt_status zt_read_at(const zt_member *m, uint64_t offset, uint8_t *dst, size_t len);
zt_status zt_decode_tensor_into(const zt_member *m, const char *name, uint8_t *dst, size_t len);
zt_status zt_tensor_count(const zt_member *m, size_t *out);
zt_status zt_tensor_info(const zt_member *m, size_t i, zt_tensor_info *out); // borrowed strings live as long as m
zt_status zt_restore_file(const zt_member *m, const char *utf8_out);
size_t    zt_last_error(char *buf, size_t cap);   // thread-local
void      zt_member_close(zt_member *m);  void zt_close(zt_container *c);
```
**Python (`ziptensors`):** `Container(path, threads=None, max_total_decoded=None)`, `.members`, `.member(name=None) -> Member`; `Member.read_at(offset, buffer)`, `.decode_tensor_into(name, buffer)`, `.tensors`, `.restore_file(path)`, `.verify()`; `encode_file(input, output, mode="balanced", threads=None)`, `pack(inputs, output, mode="max", threads=None)`; `ZiptensorsError` plus one subclass per family. Buffer calls release the GIL (`detach`). `PyBuffer<u8>` checks item format, so a float32 numpy array would be rejected: WP7 uses raw `PyObject_GetBuffer` with writable C-contiguous flags and accepts any item format.

**Threading.** Reads: the calling thread does all I/O: plan chunk list, coalesce adjacent chunks into one `read_exact_at` of up to threads × chunk, decode that batch on a private `rayon::ThreadPool` (owned by the Container, not the global pool, so hosts are unaffected), repeat. `threads=1` runs inline. Scratch workspaces come from a bounded pool of `threads` (deterministic bound, freed on drop), not thread-locals. `dst` is split with `split_at_mut`. Whole chunks with one zstd stream use in-place; partial chunks and multi-compressed chunks use scratch. Compressed source never overlaps `dst`.
Writes: one input reader thread, rayon compression, ordered single writer, `sync_channel` window of 2 × threads chunks. BLAKE3 updates in order on the reader thread. Duplicate detected at tensor end: truncate logical append position back to blob start (append then roll back). Widening: mid-tensor sample, then stream narrowed; a late mismatch rolls back and restreams without widening. Nothing holds a whole tensor. Input is re-checked (len, mtime) at commit: `InputChanged`.

**Unsafe** only in `codec/kernels_x86.rs`, runtime-detected AVX2, every block with a SAFETY comment; `#![deny(unsafe_op_in_unsafe_fn)]`.

### 3. Work packages
Test hygiene in every package: no test reads a real model or a user path; fixtures generated in tests with a seeded PRNG; scratch under `temp_dir`.

**Sequence.** R0: WP0. R1 (parallel): WP1, WP2, WP3. Review WP2 + WP1. R2 (sequential): WP4, then WP5. Review. R3 (parallel, deferred): WP6, WP7, WP8, WP9. R4: WP10, then full review.

**R1 isolation.** All three share one crate, and a sibling's unresolved import stops rustc before type-checking, so a shared tree would mask errors. Each R1 package is developed in its own copy of the WP0 tree, with its own `CARGO_BUILD_BUILD_DIR` and `CARGO_TARGET_DIR`, and fully verified there; the copies are merged and verified once. R3 crates are separate: one crate check each, then one verification of the whole workspace.

| WP | Owns | Goal, contracts | Required tests |
|---|---|---|---|
| **WP0** scaffold | root `Cargo.toml`, `.cargo/config.toml`, `.gitignore`, `PROJECT_RULES.md`, `ARCHITECTURE.md`, `spec/`, all `Cargo.toml`, `core/src/{lib,error,types}.rs`, `core/src/io/`, stub `mod.rs` per module, all `MODULE_README.md` | Freeze `Error`, `Dtype`, `Mode`, `Verify`, `TypedRange{offset,len,hint}`, `RangePlan`, `RandomAccess { fn len; fn read_exact_at(&self, off: u64, buf: &mut [u8]) }`. Spec from part 1 in full. Crate root lints per the project's Rust lint policy. | `FileSource` read past end errors; `cargo check-all` |
| **WP1** format | `core/src/format/` | `parse_header(&[u8;64])`, `parse_footer(&[u8;64])`, `Index::parse(raw: &[u8], limits: &Limits) -> Result<Index, Error>`, `validate(&Index, file_len, &Limits)`, `IndexBuilder::{add_*, serialize() -> Vec<u8>}`, zero-copy-free typed record accessors. All 12 invariants except decode-time parts. O(n log n) overlap check. | Golden bytes per record; roundtrip; one reject test per invariant; overflow values at `u64::MAX`; unknown required vs optional sections and features |
| **WP2** codec | `core/src/codec/` | `encode_chunk(raw: &[u8], chain: &Chain, policy: &StreamPolicy, ws: &mut Workspace, out: &mut Vec<u8>) -> Result<ChunkMeta, Error>`; `decode_chunk_into(chain: &Chain, stored: &[u8], dst: &mut [u8], ws: &mut Workspace, verify: Option<u64>) -> Result<(), Error>`, `dst` exactly raw length; `widen_into`, `narrow_exact(view, &[u8]) -> Option<usize>` (bytes accepted); scalar plus AVX2 kernels. zstd `default-features = false`, `WindowLogMax(24)`, exact output length. | Roundtrip widths 1/2/4/8 with rotl1 and fields; merge kernels: every w, every compressed plane index, n in 0..2048, scalar == AVX2 == reference; guard-band canary around `dst`; widen on all 65536 inputs for each kernel; -0, subnormals, NaNs; truncated and bit-flipped streams return `Err`; stream word sum mismatch |
| **WP3** profile | `core/src/profile/`, `core/tests/common/fixtures.rs` | `scan(src: &dyn RandomAccess) -> RangePlan` by magic, never extension; `normalize(plan, file_len) -> RangePlan` drops overlaps, out-of-range, trims to block, any failure -> opaque. safetensors, GGUF (incl. quant block sizes), ONNX wire walker (`raw_data`, packed `float_data`, nodes and subgraphs; external_data ignored, never opened), `.pt` ZIP stored entries; compressed entry, ZIP64 oddity, legacy pickle -> opaque. `width_trial(sample) -> u8`. No pickle parser in v0.1. Bounded reads. | Generated fixture builders per format; each parser returns ranges matching the builder; malformed header yields opaque, never `Err` or panic; name lies about content (`.safetensors` with GGUF magic) |
| **WP4** read | `core/src/read/` | Part 2 read API. Budget before data I/O. Partial chunks. Views across chunk boundary. `read_many` sorted by pack offset. | `read_at` at every boundary vs reference (random offsets, seeded); widened view at unaligned offsets; concurrent `read_at` from 8 threads; counting `#[global_allocator]` test asserts peak above `dst` ≤ 3 × threads × chunk + fixed index size on a 256 MiB synthetic; budget rejection before reads; `RandomAccess` mock counts I/O coalescing |
| **WP5** write | `core/src/write/`, `core/tests/roundtrip.rs` | Part 2 write API, policy table, append and rollback, BLAKE3 dedup map, verify-before-commit, `add_member_from`. Failure leaves no final file and removes temp. | **Bit-exact roundtrip per dtype** (F16, F32, F64, BF16, FP8, I8, U8, Q8_0, Q4_K, Q5_K, Q6_K, nibble) **and per profile** (safetensors, GGUF, ONNX, pt_zip, opaque, empty file, zero-length tensor) in all 3 modes; F32 member shares blob with F16 sibling; safetensors and GGUF share a blob; `add_member_from` writes zero new compressed bytes (byte compare of chunks); late widen mismatch rollback; injected corruption fails verify and no output appears; allocator peak on encode ≤ 6 × threads × chunk + index |
| **WP6** capi | `crates/ziptensors-capi/` | C ABI above; exhaustive `Error -> zt_status`; null and length checks; cbindgen header generated, checked in test. | Rust tests through `extern "C"` for each entry, invalid args, last-error text, panic barrier; C smoke compile with `cc` skipped cleanly if absent |
| **WP7** py | `crates/ziptensors-py/`, `pyproject.toml`, `crates/ziptensors-py/tests/` | Python API above; maturin in throwaway venv outside the tree. | pytest: roundtrip via `bytearray`, numpy float32 buffer (skip if no numpy), read-only buffer rejected, wrong size rejected, GIL released (second thread progresses) |
| **WP8** cli | `crates/ziptensors-cli/` | `encode`, `pack`, `restore [--all <dir>]`, `extract`, `merge`, `inspect [--json]`, `verify`. No logic beyond core. Skips symlinks with a log line. | `assert_cmd`-free process tests via `env!("CARGO_BIN_EXE_ziptensors")`; exit codes; `restore --all` rejects `../x` member |
| **WP9** adversarial | `core/tests/malicious.rs`, `core/tests/conformance.rs`, `fuzz/` (excluded from workspace, nightly) | Fuzz targets `open_and_read` (arbitrary bytes), `decode_chunk`, `profile_scan`. Seeded in-`cargo test` mutator on the same functions, so stable CI covers it. | Decompression bomb, index bomb, 2^32 extents on one chunk, overlapping chunks, partial chunk overlap, cyclic-free kinds rejected, truncated at every 4 KiB, bit flip every section, oversized shape, `..` member, duplicate names, unknown ids. Each expects a typed error, no panic, peak RSS bounded |
| **WP10** bench | `tools/bench/` (Python), `tools/bench/MODULE_README.md` | Part 5. Takes roots and output dir as required arguments; no paths in tracked code. | `--dry-run` on a generated tree in `temp_dir` |

Out of scope everywhere: delta, derivation, FUSE, memfd, pickle parsing, adapters, consumer-application changes, chunk cache (add only if R2 falsifies), pip wheels.

**Verification (after each round):**
```bash
cargo check-all                          # alias: check --workspace --all-targets, linux + x86_64-pc-windows-gnu
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --target x86_64-pc-windows-gnu -- -D warnings
cargo clippy -p <each touched crate> --all-targets -- -D warnings
cargo test --workspace
cargo-zigbuild check --workspace --target x86_64-apple-darwin          # plain cargo check fails, see F4
CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER=wine cargo test -p ziptensors-core --target x86_64-pc-windows-gnu  # R3 onward
find . -type f -newer <stamp> -not -path './dev-docs/*'                  # after tests: nothing new in the tree
```
Pin in `Cargo.lock`: `zstd =0.13.4`, the measured version (0.14.3 exists; a bump needs a benchmark rerun). `PYO3_NO_PYTHON=1` on cross checks if WP7 finds it needed; record in `PROJECT_RULES.md`. Fuzz, Linux only: `cargo +nightly fuzz run <target> -- -max_total_time=600`.

### 4. Project documents (WP0)
- `PROJECT_RULES.md`: toolchain 1.99.1, edition 2024; mandatory targets linux, windows-gnu, macOS by zigbuild (non-mandatory link, `SDKROOT`); commands above; tree is NOT rustfmt-clean policy, no formatter; shared `build-dir` note and R1 override; fixtures generated only; bench is out-of-tree; `dev-docs/` is the development-docs folder; `STATE.md` untracked. Thresholds default.
- `ARCHITECTURE.md`: four layers; dependency direction (`capi`, `py`, `cli` -> `core`; in core, `read`, `write` -> `format`, `codec`, `io`; `write` -> `profile`; `profile` knows nothing of the format); correctness boundary (parsers are hints); memory and thread contract; extension points.
- `spec/ziptensors-v0.1.md`: normative, owner of field widths and id registry. Code in `format/ids.rs` cites it. A change to one requires the other.
- `MODULE_README.md` in: `crates/ziptensors-core/src/` and each of `io`, `format`, `codec`, `profile`, `read`, `write`; `src/` of `capi`, `py`, `cli`; `fuzz/`; `tools/bench/`.
- `dev-docs/feasibility/` stays as is. Results of WP10 go to `dev-docs/benchmarks/v0.1.md`, machine-local roots to `dev-docs/local-notes.md`.

### 5. Acceptance benchmark (WP10, out-of-tree)
Roots: A and B of the inventory study, opened read-only. Same selection as the inventory: 546 files, 361.14 GB, no symlinks, no `.incomplete`. Output on the HDD outside the roots, at least 300 GB. Snapshot `find -printf '%p %s %T@'` of both roots before and after; must be identical. Check idle disk before each timing.
1. **Encode:** `pack --mode max` for the full set; record wall time (estimate 1.5-2.5 h), peak RSS, final size. Also a subset of 12 files in `fast`, `balanced`, `max`.
2. **Bit-exact:** `restore --all`, then sha256 by an independent tool against the source, 546 of 546. Delete restored copies.
3. **Savings by layer vs estimate:**

| Class | Raw GB | Dedup/widen GB | Unique GB | Ratio | Entropy saved GB |
|---|---:|---:|---:|---:|---:|
| BF16 | 155.9 | 12.61 | 143.3 | 33.7% | 48.3 |
| F16 | 23.5 | 0 | 23.5 | 26% TE, 15.6% rest | 5.3 |
| F32 | 20.5 | 13.53 + 1.05 + 0.92 | 5.0 | 16.3%, narrowed 33% | 1.0 |
| FP8 | 13.8 | 0 | 13.8 | 18% | 2.5 |
| Q8_0 | 52.9 | 2.48 | 50.5 | 5.76% | 2.9 |
| Q4_K, Q5_K, Q6_K | 57.4 | 0 | 57.4 | 4.57, 1.59, 3.03% | 1.9 |
| U8 packed | 35.3 | 16.62 | 18.6 | stored | 0 |
| Other | 1.8 | 0 | 1.8 | not estimated | 0 |
| **Total** | 361 | **47.3** | | | **61.9**, sum **109.2 = 30.2%** |

Gate: total saved ≥ 27.0%, each class within 2 pp of target. Estimates for modes: `fast` about 19%, `balanced` about 21%. Below gate: stop and report which layer fell short.
4. **Load from HDD, end to end:** evict by `posix_fadvise(DONTNEED)` after `fsync` (no root; confirm with `fincore`). 3 runs each. Baseline: raw file, `readinto` to one buffer. Candidate: `decode_tensor_into` for all tensors.
   - klein-9b BF16 18.16 GB: raw estimate 104 s, target ≤ 80 s (computed 70 s).
   - SDXL unet F32 10.27 GB through the shared F16 blob, target below half of raw.
   - Q8_0 GGUF: stored or near, target ≤ raw + 5%.
   - the community finetune of klein-9b, the most fragmented member (R6), in tensor order and via `read_many`.
   - Report seek proxy: extent runs per member.
5. **Random access:** R2. **Memory:** peak RSS minus destination for all of the above.
Result table goes to `dev-docs/benchmarks/v0.1.md`, measured next to estimate.

### 6. Open risks, each with a falsifier
| Risk | Falsifying test | Threshold | If falsified |
|---|---|---|---|
| **R1 zstd in-place tail** | WP2 kernel matrix with canaries; `decode_chunk` fuzz under ASan; prove no src/dst overlap by a debug assert in `decode_chunk_into`; SIMD near the convergence point is the real hazard (prototype finishes the last 64 elements scalar via a copy) | Any canary change, mismatch, or ASan report | Always use scratch; cost about 2% (12.5 vs 12.7 GB/s) |
| **R2 `read_at` on HDD, 1 MiB** | 10k cold random 4 KiB and 1 MiB reads on an 18 GB member vs raw; then repeat a DISK-tier pattern (same spans, 3 passes) | p50 > raw + 8 ms (expected: 0.67 MiB read ≈ 4 ms, decode 0.3 ms) | Add bounded partial-chunk cache, or 256 KiB chunks (ratio 32.785 vs 32.794, no format change since `chunk_size` is per blob) |
| **R3 encode RAM** | Allocator test (WP5) and bench peak RSS, both independent of tensor size | > 6 × threads × chunk + index, or growth with tensor size | Reduce channel window; fix leak; never whole-tensor buffers |
| **R4 Windows slower path** | Largely answered by F6 (-1.5% at 1T). Remaining: native Windows run of decode bench by the user; wine only proves function, not timing | 1T < 3 GB/s on BF16 | Build zstd with clang or vendor ASM via a build option |
| **R5 dedup index memory** | Peak allocation of the map in the full pack | > 16 MiB; computed about 5 MiB for < 45k blobs | Tensor-level identity keeps this trivial; chunk-level identity would be 26 MiB, still fine |
| **R6 dedup fragments members on HDD (new)** | Bench step 4, the klein-9b finetune: time vs its private-layout time; seek count | > 15% slower than a one-member container of the same file | Ingest-order policy (largest first), `read_many` for all consumers; only then consider partial unsharing |
| **R7 widen speculation** | Allocator and time on an adversarial tensor: last element non-exact | Bound exceeded | Scan fully before compress for that tensor, costs one extra read |

## Side findings
- Prototype `merge_inplace` handles only `(w,ci) = (2,1), (4,3)` by AVX2 and falls to a scalar loop otherwise; Q and F16-widen paths need their own kernels.
- `.zt` is zTensor's extension. Proposed `.ztz`; the magic decides anyway.

### Questions for the user
1. F3: keep quantized stored in `fast`, or allow raw+h (about 0.9 pp, decode 12 GB/s)?
2. Is content-only restore acceptable (no mtime, permissions, symlinks)?
3. May a worker run `cargo install cargo-fuzz` and pip-install `maturin` in a throwaway venv? Both are missing.
4. Extension `.ztz`, or another?
5. License for the crates, MIT OR Apache-2.0?
