# Ecosystem check for "ziptensors": adopt, extend or write our own, and can consumers load without a file on disk?

Checked 2026-11-02. Source prefixes: `ORT`=github.com/microsoft/onnxruntime (main), `LL`=github.com/ggml-org/llama.cpp (master), `ZT`=github.com/pie-project/ztensor (main), `ZS`=github.com/gyscos/zstd-rs (main). "Probe" = run locally on kernel 7.0.0 with torch 2.12.0a0.

## Verdict
**Write our own thin container and codec core. Do not adopt ZipNN or depend on zTensor. Reuse designs, plus the MIT parser code selectively.** Confidence: high on the decision, medium on load paths (read from headers, not run, except torch).
- Nothing existing combines what we need: Rust core, all three desktop OSes, parallel decode into a caller buffer, byte-exact restore of a foreign file, dedup across a set of small models.
- The core idea that makes it safe: **treat every input format as opaque bytes plus a list of ranges.** Parsed metadata only chooses the codec, so a parser error costs ratio, never correctness. Probe confirmed this for `.pt`.
- Most copies can be avoided: ORT 1.31+, llama.cpp and torch all have an in-memory route. Path-only loaders need memfd on Linux; on Windows they still need a temp file.
- Fault-driven lazy mmap of compressed data is not worth building for v1.
- Names are free: `ziptensors` and `ziponnx` are absent from crates.io, PyPI and GitHub repository search.

## Evidence
### 1. Prior art
| Project | State (checked today) | Fit |
|---|---|---|
| ZipNN | v0.5.4, 2026-04-11, 344 stars, Python+C, MIT file ([PyPI](https://pypi.org/project/zipnn/), [LICENSE](https://github.com/zipnn/zipnn/blob/main/LICENSE.txt)). No Rust (`zipnn` not on crates.io). PyPI latest has one wheel (macOS arm64 cp314) plus sdist. Maintainer: not tested on Windows ([#79](https://github.com/zipnn/zipnn/issues/79)). Loader monkey-patches `safetensors` (README:44-48). Format is a 32-byte header in `zipnn.py:215` + `.znn` suffix | **Benchmark only.** Paper: FP32 83.2% vs zstd 92.3% (Olmo); BF16 66.4% vs 77.7%; 1 thread decode 1.65-2.48 GB/s ([arXiv](https://arxiv.org/html/2411.05239v2)) |
| zTensor | 2.1.1, 2026-08-04, MIT, Rust, 6 stars, 1 contributor, 0 issues. Spec Draft 4, "may still change" ([README](https://github.com/pie-project/ztensor#status)). **Still no byte grouping**: the only encoding profile is `zt.zstd-seekable/1` ([spec/profiles](https://github.com/pie-project/ztensor/tree/main/spec/profiles)). CI Linux+macOS; Windows "unverified" (`website/docs/formats.md:34-39`). No Windows wheels among the 21 PyPI files | Closest design, wrong core (below) |
| HF Xet | Active, Apache-2.0. `ByteGrouping4LZ4` with a per-data predictor ([compression_scheme.rs:24-27](https://github.com/huggingface/xet-core/blob/main/xet_core_structures/src/xorb_object/compression_scheme.rs), [docs](https://huggingface.co/docs/xet/xorb)) | Storage/transfer only; BG4 is width 4 only. Useful predictor idea |
| c-blosc2 | v3.4.1, 2026-10-30, BSD. Rust: `blosc2` 0.2.3 (3 stars, Ndarray-centred); `blosc2-rs`/`-sys` stuck at 2.15.2, Jan 2025 ([repo](https://github.com/barakugav/blosc2-rs)) | **Skip.** Thin bindings and a second C build for a byte transpose we can write |
| DFloat11 | v0.5.0, Aug 2025, Apache-2.0, last push 2025-11. Needs CUDA 12, `decode.cu` kernels ([README:32,51](https://github.com/LeanModels/DFloat11)) | BF16+CUDA only, confirmed |
| safetensors | Repo moved to [safetensors/safetensors](https://github.com/safetensors/safetensors), 0.8.0. Issue search finds no compression PR or proposal; closest is [#527](https://github.com/safetensors/safetensors/issues/527) (zarr comparison) | Upstream is not moving; keep JSON-header familiarity only |
| GGUF | No merged compression. Open: [#26136](https://github.com/ggml-org/llama.cpp/issues/26136) QFX lossless type (claims 2.05x on F32, 1.14x on F16: the gap suggests upcast weights, my inference), [#30460](https://github.com/ggml-org/llama.cpp/issues/30460) BF16X (near-lossless, ~99% exact) | Not lossless-general; no seam today |
| ONNX | [onnx#8370](https://github.com/onnx/onnx/issues/8370) (open, 2026-08-27) asks for an "encoded tensors" boundary with fail-closed unknown codecs. [ORT#15456](https://github.com/microsoft/onnxruntime/issues/15456) open since 2023 | **Best upstream hook later** |
| New 2025-26 | TensorHub (delta plus safetensors-compatible blobs, 70.5% on 40 TB; [arXiv 2604.17104](https://arxiv.org/html/2604.17104v1)); ZipLLM ([NSDI'26](https://www.usenix.org/conference/nsdi26/presentation/wang-zirui), 54.1%, hub-scale); Brevis ([arXiv 2608.02162](https://arxiv.org/abs/2608.02162), synthesized programs, 33.9%); [OpenZL](https://github.com/facebook/openzl) (Meta, C/C++, format explicitly unstable). No code release found for the first three | Hub-side research. Their gains come from dedup and delta across models, which matches brief §2.5. No standard displaced safetensors |

**Why not build on zTensor (it was the serious candidate):**
- `Encoding::decode(&[u8], u64) -> Vec<u8>` and `Part::bytes` always allocate, and frames decode serially; there is no decode-into-buffer (`ZT/ztensor/src/vocab.rs:35-39`, `read.rs:899-925`, `vocab.rs:511`). Our "no redundant full copy" needs exactly that seam.
- Every blob sits on a 4 KiB boundary (`spec/ztensor-v2-spec.md:101`), and canonical files on 64 KiB (`:449`). README reports 6.4x payload for 51k tiny tensors at 64 KiB. For compressed blobs that padding buys nothing, and small models are our case: 200 tensors cost about 400 KiB at 4 KiB alignment (my estimate, mean 2 KiB per blob) against a 0.5-10 MB model.
- It defines a new tensor file, not byte-exact restoration of a foreign `.pt`/`.onnx`/`.gguf`.
- The ONNX parser covers only graph initializers; nodes are out of scope and external data is refused (`ztensor-compat/src/onnx.rs:1-12`). Brief §3 found weights in `Constant` nodes and subgraphs.
- Pre-1.0 single-maintainer spec as our on-disk contract is a real risk.

**What to take from it:** `with_encoding` allows an out-of-tree profile (`vocab.rs:138`), so a `zt.byte-group` upstream contribution stays a cheap optional bonus. Copy these spec rules: identical-or-disjoint ranges (`spec:104-111`), an integrity ladder (§6.1), blob-heap overlay for base+delta (`spec:578-582`), and the conformance corpus idea (19 accept, 57 reject). Its fuzzed restricted pickle VM is MIT (`ztensor-compat/src/pt.rs:1-40`).

**Names.** crates.io API: `ziptensors`, `ziponnx` do not exist. PyPI: same. GitHub repo search: 0 for both. Taken nearby: `zipnn` (PyPI), `ztensor`. Brief's point that "zip" misleads stands; trademarks not checked.

### 2. Loading without a file on disk
| Consumer | Best route | Copies / limits |
|---|---|---|
| ORT | Skeleton ModelProto with external refs + `AddExternalInitializersFromFilesInMemory` + `session.use_external_initializer_file_buffers_directly=1` | **Borrowed, new in 1.31** (key present in v1.31.1, absent in v1.30.0). Only "naturally aligned native-endian slices"; others copied. Buffers must outlive sessions (`ORT/include/.../onnxruntime_c_api.h:5375-5396`, `..._config_keys.h:426-431`). **Main graph only** |
| ORT, older | `AddInitializer`: user-owned OrtValue that "must outlive" the session (`c_api.h:3209-3215`). `AddExternalInitializers` **copies** into the graph, main graph only (`:4159-4182`). Model Editor `AddInitializerToGraph(..., data_is_external)` (`:8080`) | Whether prepacking re-copies: unverified; `session.disable_prepacking` exists |
| ORT, whole proto | `CreateSessionFromArray`; rejected above `INT32_MAX` (`ORT/onnxruntime/core/session/utils.cc:274`) | Parses into `ModelProto` (`inference_session.cc:1335`), so extra transient copies. Use for subgraph/Constant data only |
| `ort` crate 2.0.0-rc.15 (ORT 1.31, MSRV 1.92) | `with_initializer`, `with_external_initializer`, `with_external_initializer_file_in_memory(name, Arc<[u8]>)` feature `api-18`, `commit_from_memory`, Model Editor `add_initializer(.., as_external)` ([impl_options.rs:195-221](https://github.com/pykeio/ort/blob/main/src/session/builder/impl_options.rs)) | `commit_from_memory_directly` zero-copy is `.ort` format only (`impl_commit.rs:78-92`). The new key has no named helper; our config path needs checking |
| llama.cpp | `llama_model_init_from_user(gguf_ctx, set_tensor_data_cb, ...)`, called per tensor (`LL/include/llama.h:512`, `src/llama-model-loader.cpp:1573`). Metadata from `gguf_init_from_buffer`/`_from_callback` (`ggml/include/gguf.h:85,88`) | Decodes straight into loader-owned tensors: no temp file, no mmap. Header says tensors default to F32, so test real arches and dtypes |
| llama.cpp, fd | `llama_model_load_from_file_ptr(FILE*)` (`llama.h:533`) with `fdopen(memfd)`; mmap requires a 32-byte aligned data section (`llama-model-loader.cpp:695`) | `fopencookie` will not work for mmap: code uses the fd (`llama-mmap.cpp:229,433`; inferred, not run). `llama_model_load_from_fd` PRs [#28973](https://github.com/ggml-org/llama.cpp/pull/28973), [#20402](https://github.com/ggml-org/llama.cpp/pull/20402) unmerged |
| stable-diffusion.cpp | Paths only, `const char*` plus `enable_mmap` ([stable-diffusion.h:204-233](https://github.com/leejet/stable-diffusion.cpp/blob/master/include/stable-diffusion.h)) | memfd path, temp file, or a patch |
| torch | `torch.load(BytesIO, weights_only=True)` works; `mmap=True` needs a path (`torch/serialization.py:1590-1593`; probe: ValueError) | Probe: `mmap=True` **works via `/proc/self/fd/N`**. Better: fill tensors ourselves, then `load_state_dict(assign=True)` on meta (probe OK). `torch.frombuffer` shares memory (probe) and warns on non-writable buffers (probe), so expose writable ones |
| safetensors | `safetensors.torch.load(bytes)` | One full extra copy: binding builds a `PyByteArray` per tensor (`bindings/python/src/lib.rs:292`) |
| transformers 5.22 | `from_pretrained(None, config=..., state_dict=sd)`; error if a name is also given (`modeling_utils.py:4069,4133`) | Accepts in-memory dict |
| diffusers 0.42 | `ModelMixin.from_single_file(dict)` (`loaders/single_file_model.py:402`) | `DiffusionPipeline.from_single_file` takes path only (`loaders/single_file.py:284`); no `state_dict` kwarg in `ModelMixin.from_pretrained` found by grep |

**Path-only workarounds**
- **memfd (Linux):** probe: 256 MiB written → `Shmem` +256 MiB, so it is RAM, not disk. A read-only mapping adds nothing, so loader mmap shares the pages. Sealed fd: `pwrite` and shared RW map give EPERM, RO map OK, `MAP_PRIVATE` OK. The path is valid only while the fd lives, and per process. Pages move only to swap; with no swap they stay resident (inferred from [memfd_create(2)](https://man.archlinux.org/man/memfd_create.2.en) "anonymous memory").
- **FUSE:** gives a real path, mmap, and evictable decoded page cache, so the best memory behaviour. But needs `/dev/fuse`, `fusermount`; `fuser` crate MIT. macOS: macFUSE kernel backend is default, FSKit backend needs 15.4+ and user approval ([ifun](https://www.ifun.de/macfuse-5-0-neue-version-mit-unterstuetzung-fuer-apples-fskit-253976/)); `fuser` calls macOS "untested". **WinFsp is GPLv3 with FLOSS exception, otherwise commercial** ([README:56](https://github.com/winfsp/winfsp)). Optional helper, not core.
- **Windows temp file:** `FILE_ATTRIBUTE_TEMPORARY` avoids write-back only "if sufficient cache memory is available". `FILE_FLAG_DELETE_ON_CLOSE`: later opens fail unless they pass `FILE_SHARE_DELETE` ([CreateFileW](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew)), so an ordinary `fopen` in a loader will fail. Use the attribute alone, plus own cleanup.
- **CfApi:** NTFS only; placeholders hydrate into full on-disk files, sync root appears in Explorer, toasts ([docs](https://learn.microsoft.com/en-us/windows/win32/cfapi/build-a-cloud-file-sync-engine)). That is the redundant copy we want to avoid. ProjFS: [optional provider API](https://learn.microsoft.com/en-us/windows/win32/projfs/projected-file-system), expected same behaviour (unverified).

**Lazy mmap of compressed data: feasible only on Linux, not recommended.**
- Probe: `UFFD_USER_MODE_ONLY` succeeds unprivileged; plain fails EPERM (`vm.unprivileged_userfaultfd=0`, the [default](https://docs.kernel.org/admin-guide/mm/userfaultfd.html)). Flag exists since 5.11.
- A kernel-originated fault in that mode **delivers SIGBUS** ([userfaultfd(2)](https://man7.org/linux/man-pages/man2/userfaultfd.2.html)). So `write()` from the region, `read()` into it, or driver access can kill the process. A GPU upload from unfaulted memory is the realistic trigger.
- Docker's default seccomp does not allow it: `userfaultfd` is absent, default action ERRNO ([profile](https://github.com/moby/profiles/blob/main/seccomp/default.json)).
- Windows (VEH) and macOS (Mach/signals) would each be a separate, process-global mechanism colliding with hosts like Python; not verified.
- Prior art for weights: none credible found; CRIU lazy restore is the mature use. ORT and non-mmap llama load everything anyway. **Since inference code is ours, on-demand `decode_tensor_into(name, buf)` delivers "on the fly" without any of this.** Brief's "zero-copy with compressed tensors is impossible" stays true for on-disk bytes, but not for the goal.

### 3. Parsing untrusted formats in Rust
- **Do not rely on scanners or on `weights_only`.** GitHub lists 86 picklescan advisories, incl. ZIP differentials and blocklist bypasses (CVE-2025-1944/-1945, -10155/-10156/-10157; [advisories](https://github.com/advisories?query=picklescan)). Torch: CVE-2025-32434 (`weights_only=True` RCE, fixed 2.6.0, [GHSA](https://github.com/advisories/GHSA-53q9-r3pm-6pq6)); CVE-2026-24747 (`SETITEM` on non-dicts corrupts memory, fixed 2.10.0, [GHSA](https://github.com/advisories/GHSA-63cw-57p8-fm3p)). Blocklist hardening continues: `_weights_only_unpickler.py` commit 2026-10-13.
- **`.pt` byte-exact, proven by probe:** all entries `stored`, data 64-byte aligned, ranges located from ZIP headers. Removing the `archive/data/N` ranges and splicing them back gave a sha256-identical file, and torch loaded it. Pickle was never parsed. 1777 of 5797 bytes stayed as skeleton. Non-ZIP legacy files: whole-file opaque.
- **Rule:** encode records `(offset, length, codec)` and verifies by decoding and hashing before commit. Overlaps, ZIP64 oddities, compressed entries, any parse failure → fall back to opaque. Width choice: try restricted-pickle hint, or simply trial 2/4/8 on a sample (BG4 predictor shows this is cheap).
- **Pickle in Rust, if wanted as a hint:** a non-executing stack machine with budgets where globals stay inert names. Best model: `ztensor-compat/src/pt.rs:35-40` (byte, opcode, memo, depth limits; feature-gated by design). Alternatives: candle `candle-core/src/pickle.rs` (841 lines, symbolic `Reduce`/`Build`, active), `serde-pickle` 1.2.0 (2024, known-globals only), `repugnant-pickle` 0.0.1 (2024, "low effort scraping"; not for production). Rust removes the memory-corruption class, not differentials, hence hint-only.
- **ONNX:** hand-written wire walker, no protobuf dep. `raw_data`=9 and packed `float_data`=4 are byte runs (LE), `external_data`=13 has location/offset/length/checksum ([onnx.proto:700,766,770-777](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto)). Must descend into nodes and subgraphs. Unpacked or varint fields stay in skeleton. Never follow `location` outside the model set.
- **GGUF:** walk header, KV, tensor infos; offsets are relative to `tensor_data` and multiples of `general.alignment`, default 32 ([gguf.md:147,316-324](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), `gguf.h:46`). Padding stays in skeleton. Quantized types get no byte grouping.

### 4. Rust building blocks
- `zstd` 0.14.3 / `zstd-safe` 8.2.0 (updated 2026-11-01): `bulk::Decompressor::decompress_to_buffer` (`ZS/src/bulk/decompressor.rs:96`), `DCtx` is Send+Sync (`zstd-safe/src/lib.rs:1714,1717`; use one per thread), `WindowLogMax` (`:1592`), `seekable` feature.
- **Static libzstd pitfall is handled for cdylib/extension:** zstd-sys hides symbols by default so it can coexist (`zstd-safe/zstd-sys/build.rs:38-49,481`). `ZSTD_SYS_NO_HIDE_SYMBOLS` must stay unset. A `staticlib` consumer that also links libzstd: duplicate-symbol risk (inferred), so ship the C ABI as cdylib first.
- **Windows runs slower decode:** ASM Huffman is disabled on all Windows targets (`build.rs:321-326`). Benchmark there.
- Pure Rust: `ruzstd` 0.9.1, `structured-zstd` 0.2.6 (young, claims full levels; Apache-2.0). Keep behind a feature; default to C libzstd. Not benchmarked.
- `lz4_flex` 0.14: `decompress_into` (`block/decompress_safe.rs:325`), safe by default. `blake3` 1.8.7, `xxhash-rust` 0.8.19, `memmap2` 0.9.12, `cbindgen` 0.29.4 (MPL-2.0, build-time).
- rANS: `constriction` 0.5.0 is the live option. `ryg-rans` crate **does not exist**; `ryg-rans-sys`, `rans` stale since 2024. Defer: zstd's own entropy stage comes first; test whether custom exponent coding beats it.
- `pyo3` 0.30.1, `maturin` 1.16.0: `PyBuffer` under abi3 **requires Python 3.11+** (`pyo3/src/buffer.rs:3`). So use `abi3-py311` and fill torch/numpy buffers with the GIL released. `allow_threads` is now `detach` (CHANGELOG:573). `abi3` wheels do not load on free-threaded CPython; `abi3t` is the 3.15+ answer (`guide/src/building-and-distribution.md:104,259`).

### 5. Fail-safe references
- **Bomb bound:** decode only into a slice of the size declared in the index, after checking the total against a caller budget; a bounded buffer cannot overrun. Set `WindowLogMax`. Exact-length check as in `zt.zstd-seekable-1.md`. Safetensors caps the header at 100 MB and rejects holes (README:99,172).
- **Checksums:** per-chunk over decoded bytes enables parallel and partial verification; use xxh3. Add zstd frame checksums for transport (zTensor §6.1 ladder). Use **BLAKE3 for dedup identity**, because xxh3 is non-cryptographic and a crafted collision would swap tensors (TensorHub uses xxHash64 in a trusted hub; ours is not).
- **Layout:** [zarr v3 sharding](https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html): fixed-size index of `(offset, nbytes)` u64 pairs at the end, uncompressed, CRC32C, all-ones sentinel for empty. [Parquet page index](https://parquet.apache.org/docs/file-format/pageindex/): index separated near the footer so selective readers avoid scanning. Combine: chunk table in one validated region, codec chain per entry (brief's `["byte_group","zstd"]`).

### Claims in the brief that were wrong or need correcting
1. Xet BG4 link: **404**; moved to `xet_core_structures/src/xorb_object/byte_grouping/bg4.rs`.
2. Xet "~20%" cites a user activity page; no official figure found. Unsupported, not disproved.
3. zTensor links pin 1.2.3; current is 2.1.1, spec Draft 4. "No byte shuffle" remains **true**.
4. zTensor "reads safetensors/pt/gguf/npz/onnx/h5": only with features; defaults are safetensors+gguf, `pickle` opt-in, ONNX initializers only, external data refused.
5. ziponnx variant 2 (`AddExternalInitializers`) **does not cover the brief's own finding**: subgraph initializers are not replaced, the API needs external-data references, and it copies. Needs a hybrid: main-graph buffers in memory, subgraph/Constant data inline in the rebuilt proto.
6. "Next step: C++ via c-blosc2": poor fit now that the core is Rust.
7. compressed-tensors "lossy": imprecise; quantization is lossy, sparse packing is not (search snippets only).
8. Sound on re-check: ZipNN FP32 -17% and BF16 -33% (one model each), DFloat11 scope, 2 GB limit.

## Not verified
- No compression benchmark run: all ratio and speed figures are third-party claims.
- ORT, llama.cpp, sd.cpp behaviour comes from headers and sources, not execution (`onnxruntime` not installed here). `llama_model_init_from_user` needs a real-model test.
- Windows, macOS, Android: nothing executed. Android seccomp for userfaultfd unanswered; ProjFS hydration unconfirmed.
- memfd limits beyond Shmem accounting. TensorHub/ZipLLM/Brevis code availability (not found, not proven absent). Trademarks.

## Recommended next step
Plan v0.1 in this order: (1) spec: opaque skeleton + range table + per-entry codec chain + chunk index + hashes; (2) Rust core with `decode_into(&mut [u8])`; no padding for compressed entries, optional page alignment for raw ones; (3) parsers as hints: safetensors, GGUF, ONNX walker, `.pt` ZIP ranges; (4) adapters: ORT direct buffers, llama `init_from_user`, torch buffer fill, memfd helper, Windows temp fallback. Before committing the spec, spike ORT 1.31 direct-buffer mode and `init_from_user`, and benchmark on real models. Ask the user which consumers matter; sd.cpp and Windows path-only loaders are the only ones still forcing a decoded file.

## Side findings
- `bench.py` `pt_weights` (brief lines 190-192) reads every `.pt` storage as float32, so FP16/BF16/int checkpoints would give wrong stats and shuffle width.
- `.pt` written by this torch includes `archive/.storage_alignment`; any fixed entry list would be fragile, another reason for opaque handling.
