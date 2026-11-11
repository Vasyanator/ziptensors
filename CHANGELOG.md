# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the packages follow
[Semantic Versioning](https://semver.org/). The container format has its own version, stated in
[`spec/ziptensors-format.md`](spec/ziptensors-format.md); it is independent of the package versions.

## [0.2.0] - 2026-11-11

The container format stays at 0.3, and every file 0.1.0 wrote opens unchanged. The C ABI stays at
version 1; its changes are additions.

### Added

- **Editing a container:** `ziptensors edit` adds (`--add`, `--add-as`), removes (`--remove`) and
  replaces (`--replace`) members, optionally as delta blobs against a member (`--delta-base`); by
  default through a temporary file, `-F/--fast` by appending in place, `-R/--risky` by rewriting in
  place. Rust `edit` with `EditOp`, `EditOptions`, `Strategy` and `EditReport`.
- **Full repack:** `-C/--compact` re-encodes every member in `--mode`, redoing deduplication and delta.
- **`ziptensors compact`:** drops the bytes no member references any more.
- **`ziptensors recover`:** shortens a file after an interrupted append to its last valid generation
  (`--dry-run`, `--no-verify`, `--max-candidates`, `--max-scan-bytes`); a reader never does this on
  its own. Rust `recover`, `RecoverOptions`, `RecoverReport` and `find_last_valid_generation`.
  `scanned_bytes` and `max_scan_bytes` count distinct bytes, so the figure never exceeds the file.
- **Recovery report:** `RecoverReport::passed_over` (Rust and Python) counts the newer candidate
  generations that failed the checks and were passed over; the command line names them in its warning.
- **Directory trees:** `ziptensors compress` replaces every weight file by a container of one member
  beside it, `ziptensors decompress` restores them (`-r`, `-k`, `--dry-run`, `--follow-symlinks`,
  `-f`); Rust module `tree`, Python `compress_tree`, `decompress_tree` and `TreeReport`. Every line
  reads from source to result, paths and sizes alike, and `TreeTotals` sums the bytes of every file
  proven in either direction, so `decompress` reports the bytes it read and restored. Each command's
  totals line lists only the statuses it can produce.
- **Legacy PyTorch files:** `.bin` files in the pre-1.6 pickle format count as weight files for
  `compress` (`profile::classify_weight`). They are stored as opaque bytes.
- **Overlay mount:** `ziptensors mount --overlay DIR` shows a directory with every per-file container
  as the original file it holds and every other file as it is, with `--max-open-containers` and
  `--max-open-files`. Rust `MountFs::overlay` with `OverlayOptions`.
- **Python mount:** `ziptensors.mount` and `ziptensors.mount_overlay` return a `Mount` that unmounts
  when its `with` block ends; failures raise `MountError`.
- **Command line:** `extract ... -` writes the member to standard output; `inspect --tensors` lists the
  tensors of every member and adds a `tensors` key to the JSON document (its `schema_version` stays
  1); `--max-total-decoded` now also applies to `merge`, `inspect`, `verify`, `edit`, `compact`,
  `recover`, `compress` and `decompress`.
- **Python:** editing (`edit` with `AddOp`, `RemoveOp`, `ReplaceOp`, `EditReport`), `compact`,
  `recover`, `PackBuilder`, `collect_inputs` and `Inputs`, `Container.restore_all` with
  `RestoreAllReport`, `Container.layer_report`, `Container.close` and the context manager,
  `Member.read_many`, and the `Container` keyword arguments `verify`, `max_index_bytes` and
  `max_decode_amplification`.
  No name was removed from the type stub.
- **C:** `zt_edit`, `zt_recover`, `zt_restore_all`, `zt_member_info`, `zt_member_verify`,
  `zt_pack_add_member_from`, `zt_pack_add_file_delta`, `zt_enable_stderr_logging`, the
  `ZT_PROFILE_*`, `ZT_STRATEGY_*`, `ZT_EDIT_*` and `ZT_LOG_LEVEL_*` constants, and the statuses
  `ZT_STATUS_DAMAGED_IN_PLACE`, `ZT_STATUS_EDIT_REFUSED`, `ZT_STATUS_TRAILING_BYTES`,
  `ZT_STATUS_UNRECOVERABLE`, `ZT_STATUS_TREE_CONFLICT` and `ZT_STATUS_DUPLICATE_MEMBER`. Tree commands and the mount are not in
  the C ABI.
- **Optional stderr logging** for Python (`enable_stderr_logging`) and C.
- **Rust:** `Container::restore_all`, `Member::decode_to`, `Member::profile`, `collect_inputs`, the
  sources `io::SpliceSource` and `io::PrefixSource`, and the error variants (with their `ErrorCode`s)
  `DamagedInPlace`, `EditRefused`, `TrailingBytes`, `Unrecoverable`, `TreeConflict` and
  `DuplicateMember`; `check_member_names`, and `tree::ViewOptions` re-exported at the crate root.
- **User documentation** in `docs/`: the command line, Python, Rust and C interfaces, editing,
  directory trees, mounting and safety.
- **Example** `examples/diffusers_from_container.py`: loads a Diffusers model from a pack container,
  or with `--per-file-dir` from a directory made by `compress -r`, through the mount. Img2img and
  inpainting pipelines take `--image` and `--mask`. A checkpoint tensor the model does not keep (such
  as CLIP's `position_ids`) is accepted where Diffusers and Transformers accept it; a missing one is
  an error.

### Changed

- Specification: invariant 12 defines two commit protocols, replace and append; new section 9.3 on
  generations, appending and recovery. The format version stays 0.3.
- `restore --all` and the input walk of `pack` moved from the command line tool into the core, as
  `Container::restore_all` (also in Python and C) and `collect_inputs` (also in Python).
- **Behavior change:** a member name given twice in a write request (two inputs of `pack`, a name
  already in a `PackBuilder`, two containers of `merge` sharing a name) is the new error
  `DuplicateMember`, code 26, instead of `MalformedIndex`, code 6: Python raises `InputError` instead
  of `FormatError`, C returns `ZT_STATUS_DUPLICATE_MEMBER` instead of `ZT_STATUS_MALFORMED_INDEX`,
  and the command line no longer calls the file damaged. `merge` and Python `pack` check every name
  before the output is started. A container whose index holds a name twice is still `MalformedIndex`.
- The command line's hint for a malformed or undecodable index (code 6 or 17), which is what an
  interrupted append leaves, points to `ziptensors recover --dry-run` on the container named, quoted
  for the shell. `MalformedIndex` messages name the place once ("malformed index: …", "malformed
  footer: …", "malformed index section MEMB: …"). A stored index zstd cannot
  decode reports its length and offset instead of zstd's bare "Data corruption detected"; codes are
  unchanged.
- `BudgetExceeded` for the total decoded size reports the sum over all members to be decoded, not the
  running sum at the member that crossed the limit.
- `MountError::Names` reads "the names cannot be shown as a tree on this system", since it now also
  covers overlays.
- **Breaking Rust API changes** (allowed in a minor release below 1.0):
  - `format::Footer::for_index` takes a fourth argument, `generation`.
  - `ziptensors-mount`: `MountHandle::unmount` returns `Result<(), MountError>` and `MountHandle::wait`
    returns `Result<MountEnd, MountError>`, on Linux and Windows alike (Linux returned `io::Result`,
    Windows `unmount` returned nothing).
  - `ziptensors-mount`: `MountEnd` is one enum for both backends. `Abandoned` carries the signal
    number as an `i32` (`signal_name` names it) on Linux and Windows. On Windows, `wait` reports a
    volume the WinFsp driver stopped as `MountEnd::Stopped`, `unmount` as `MountError::Stopped`.
  - `ziptensors-mount`: `MountError` gained the variants `OverlayRoot`, `Scan`, `Overlay`,
    `OverlayOption`, and per platform `Unmount`, `Session` (Linux) and `Stopped` (Windows). It is not
    `#[non_exhaustive]`, so an exhaustive `match` on it must add them.

### Fixed

- Python: `inspect.signature(ziptensors.recover)` shows `max_candidates=64` instead of `...`.
- The published `ziptensors-core` crate no longer lists the test `malicious`, which compiles a file
  outside the package (`fuzz/src/harness.rs`); the tests of the crate as published now build. The
  test still runs from the repository.

## [0.1.0] - 2026-11-07

First public release of the crates `ziptensors-core`, `ziptensors-cli` and `ziptensors-mount`, of the
Python package `ziptensors`, and of the C library `ziptensors-capi`, which is not on crates.io but
ships as a shared library with its header in the archives of the GitHub release. It writes and reads
container format 0.3, the first public format version.

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
