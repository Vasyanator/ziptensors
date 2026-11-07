# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the packages follow
[Semantic Versioning](https://semver.org/). The container format has its own version, stated in
[`spec/ziptensors-format.md`](spec/ziptensors-format.md); it is independent of the package versions.

## [0.1.0] - unreleased

First public release of the crates `ziptensors-core`, `ziptensors-cli`, `ziptensors-mount` and
`ziptensors-capi`, and of the Python package `ziptensors`. It writes and reads container format 0.3,
the first public format version.

### Added

- **Container format 0.3.** Members (restorable files) made of extents over content-addressed blobs,
  cut into self-contained chunks; an index at the end of the file; XXH3-64 per decoded chunk and
  BLAKE3 per member. Unknown required features, sections, codecs and blob kinds are rejected.
- **Profiles** that choose transforms per tensor: safetensors, GGUF (including the ggml block-quantized
  types), ONNX and PyTorch zip. Any other file, and any file a parser cannot read, is stored as opaque
  bytes and restores exactly.
- **Transforms:** byte-plane split with entropy-only zstd coding of floats, field split of quantized
  GGUF blocks, an optional bit rotation, and widening views that store F32 tensors holding F16 or BF16
  values at their narrow width.
- **Writer modes** `fast`, `balanced` and `max`; deduplication of identical blobs across members in
  `max`.
- **Delta blobs:** a finetune stored as the XOR residual of changed tensors against a base member
  (`ziptensors pack --delta-base`, `PackBuilder::add_file_delta`, `ziptensors.pack(..., delta_base=)`).
- **Reader:** random access by byte range or by tensor into a caller's buffer, parallel decode with a
  fixed memory bound, an opt-in read cache for small reads, and budgets on index size, decoded size
  and decode work.
- **Writer commit:** every output is decoded and checked against the source before it atomically
  replaces the target; a failed write leaves the target untouched, except a failed directory sync
  after the rename, which is reported as its own error.
- **Command line tool** `ziptensors`: `encode`, `pack`, `restore` (one member or `--all`), `extract`,
  `merge` (copies chunks without recompression), `inspect` (with a stable `--json` document),
  `verify`, and `mount`.
- **Read-only mount** of a container as a directory of files: Linux through FUSE without libfuse,
  Windows through WinFsp. No mount on macOS.
- **Python module** (abi3, CPython 3.11+): `Container`, `Member` (`read_at`, `decode_tensor_into`,
  `read`, `open`, `restore_file`, `verify`), `encode_file`, `pack`, typed exceptions and a type stub;
  `ziptensors.torch.load_state_dict` and `load_into` with direct CUDA/ROCm staging;
  `ziptensors.safetensors.safe_open` with the API of `safetensors.safe_open`.
- **C ABI** with a generated header `ziptensors.h` and a run-time ABI version.
- Platforms: Linux, Windows and macOS (macOS built by cross-compilation, not yet run natively).
