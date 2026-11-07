# ziptensors architecture

ziptensors stores tensor files losslessly in less space, with random access at tensor speed. The
normative format is `spec/ziptensors-format.md`; this file records how the code is arranged around it.
The format version is 0.3: `FORMAT_MAJOR` 0 and `FORMAT_MINOR` 3 in
`crates/ziptensors-core/src/format/header.rs`, the header fields of spec section 3.1. It is independent
of the crate and package versions; spec section 3.4 states the compatibility promise.
A file is recognized by magic, never by extension (`.ziptensors` is only a convention).

## Layers of the format

member (virtual file) -> extent (view over a blob range) -> blob (content-addressed, chunked) -> chunk
(self-contained, copyable). A single-file container is a pack with one member, so there is one layout,
one reader and one writer. Below the members, the tensor table is metadata over `read_at`.

A delta blob (spec section 7a) stores its bytes as a residual against a range of an earlier, plain
blob, chunk for chunk with the same `chunk_size`: chunk `i` pairs with exactly one base chunk, given
by `format::DeltaParams::base_chunk`. Depth is 1. Everything above the blob (identity, extents, views,
chunk hashes) sees the combined bytes, so only the decode of one chunk knows about the base.

## Crates and dependency direction

| Crate | Output | Role |
|---|---|---|
| `ziptensors-core` | rlib | All logic. Public API in Rust. |
| `ziptensors-capi` | cdylib, generated header | Stable C ABI. |
| `ziptensors-py` | cdylib, abi3 CPython 3.11+ | Python module. |
| `ziptensors-cli` | bin `ziptensors` | Command line tool. |
| `ziptensors-mount` | rlib | Read-only filesystem view of a container (FUSE on Linux, WinFsp on Windows). |

`capi`, `py`, `cli` and `mount` depend on `core` and hold no format logic: they translate arguments. The one
adapter layer is the pure-Python `ziptensors.torch`, which maps tensor tables to torch objects. `cli`
depends on `mount` on Linux and Windows. `mount` reads through `core::ReadCache` alone; it owns member
names as paths (per-OS `NamePolicy`), node ids and the OS protocol. Its neutral part builds on every target, each
OS backend sits behind `cfg`, and a tree whose names the OS cannot show refuses to mount.

Inside `core`:

- `read`, `write` -> `format`, `codec`, `io`
- `write` -> `profile`, `read`. The commit proves a file through the real `Container`, and a member copy
  reads the source's index and stored bytes through crate-private accessors. `read` never uses `write`.
- `profile` -> `io`, `types`. It knows nothing of the format.
- `format`, `codec` -> `chain`. They do no file I/O and do not depend on each other.
- `error`, `chain`, `types` are leaves used by all. `chain` depends only on `error`, and `types` only
  on `chain`. `chain` is the single owner of the id registry except dtypes, of `Step` and `Chain`, of
  field tables and of block sizes.

## Correctness boundary

Parsers are hints. A profile parser decides only how bytes are compressed: a wrong answer costs ratio,
never correctness, and any parser failure degrades to opaque, never to an error. Extents partition a
member exactly, whatever a parser said. Tensor ranges are not trusted for restore. A restored member is
proven by its whole-file BLAKE3. Restore is content-only: modification time, permissions and symlinks
are not restored.

Files are untrusted input. Every length is validated with checked arithmetic before use, declared
sizes are checked against caller budgets before any data read, and all failures are typed `Error`
values. Unknown required features, sections, steps, views, stream codecs and blob kinds fail closed.
Unknown dtypes and profiles are legal, since they are hints. Ids stay raw until `chain` converts them,
so an unknown one is rejected with `UnknownCodec` and an invalid `Chain` cannot exist.

A caller opening untrusted input must set `max_total_decoded`: the library default is `u64::MAX`, no
budget. Decode work is bounded per container, with one floor, and names by a limit that also bounds memory.

**What the hashes do not protect.** They detect damage; nothing authenticates a file. There is no
signature or key, so anyone who can change a file can recompute every hash, and a forged file passes.

- `Verify::Off` skips chunk hashes. Lengths are still enforced, but `read_at` then returns whatever decodes.
- `Verify::Chunk` proves a chunk against the index, not against a source. An index with its chunk hashes
  rewritten passes `read_at`, which never reads the member hash. XXH3 is no defence against a forger.
  Only `verify` and `restore_file` compare the BLAKE3 of the whole member, and that value is in the same
  file.
- Member names are not covered by any hash but the index's XXH3, so renaming a member keeps it valid. A name
  is checked by `check_restore_name` only when a caller joins it to a directory. Tensor records and profiles
  are hints, likewise unhashed.

`Error` is `#[non_exhaustive]`. Frontends map failures through `Error::code()`, a stable `ErrorCode`
produced by an exhaustive match in core, so a new variant cannot go unmapped.

Blob identity is BLAKE3 of canonical narrowed bytes. Widening is an extent view, not a blob step, which
is what lets an F32 tensor share a blob with its F16 sibling.

`unsafe` exists only in `codec/kernels_x86.rs`, for runtime-detected AVX2, plus the FFI edges of `capi`
and `py`, and the Windows mount backend (`ziptensors-mount/src/winfsp/`, FFI to WinFsp and Win32).

## Memory and thread contract

**Reads.** The calling thread does all I/O: it plans the chunk list, coalesces adjacent chunks into one
read of at most threads × chunk, decodes the batch on a `rayon::ThreadPool` private to the `Container`,
and repeats. `threads = 1` runs inline. Scratch workspaces come from a bounded pool of `threads`. The
bound above the destination and the index is **≤ 3 × threads × chunk**: read batch ≤ threads × chunk,
plus scratch ≤ threads × chunk × view factor (2 for widening). A read that touches delta blobs holds
two stored chunks per decode and one more canonical chunk (the base's) per scratch entry, so its bound
is **≤ (4 × threads + 1) × chunk**; a container that never reads one keeps the first bound, since the
base buffer is allocated on first use. Tests assert both constants.

**Opt-in read cache.** `Member` caches nothing. `read::ReadCache` is the one cache, for frontends whose
reads are smaller than a chunk (a mount, a file object): it keeps decoded windows (one chunk of one extent,
viewed) LRU by bytes and fills a miss with read-ahead through the same engine. Resident and in-flight
windows together stay within `max_bytes`, plus one window per concurrent miss, plus the read bound above
per fill. Window geometry is format logic, so frontends use this cache and build none of their own.
Callers may read through it from tasks of any rayon pool: no reservation is held across a pool install,
since a fill from a rayon worker decodes on a helper thread.

**In-place decode is an optimization.** It applies only to a whole chunk with exactly one compressed
stream. The scratch path is mandatory and serves everything else: partial chunks, chunks with several
compressed streams (large low-entropy models, widened F32 in `fast`, field-split quantized blobs).

**Writes.** One input reader thread, rayon compression, one ordered writer, a bounded window of
2 × threads chunks. BLAKE3 updates in order on the reader thread. Nothing holds a whole tensor; a
duplicate or a late widening mismatch rolls back the logical append position. Inputs are re-checked at
commit. The bound is ≤ 6 × threads × chunk plus the index, and ≤ 9 × threads × chunk once a delta blob has been streamed (each windowed chunk holds its base chunk, each workspace a residual buffer).

**Commit.** Temporary file in the output directory, full decode and BLAKE3 against the source, sync,
close, rename. No opt-out. `io::TempFile` is the one owner of this sequence, for writes and restores.
Any failed commit leaves the output untouched except `Error::CommittedNotDurable`: the rename happened
but the directory sync after it (Unix) failed, so the new file is in place without confirmed durability.

`Container` is `Clone` (shared `Arc`), `Send` and `Sync`; methods take `&self` and are safe from
concurrent threads.

## Writer policy versus format

Modes (`fast`, `balanced`, `max`) choose transforms; a reader cannot tell them apart. In particular,
the absence of a zstd match finder is writer policy: zstd has no such switch, and the "entropy only"
stream is zstd with minimal matching parameters. Quantized data is stored in `fast`.

## Platforms

Linux, Windows (windows-gnu) and macOS. `zstd-sys` compiles C, so a C compiler is needed per target, but
the library exposes no C API of zstd and links no system library. macOS is reached through
`cargo-zigbuild`; plain cross `cargo check` fails in `zstd-sys`, and linking needs `SDKROOT`. Positional
reads go through `RandomAccess` (`pread`, `seek_read`).

The mount has a backend on Linux and on Windows, none on macOS. Linux: `fuser` 0.18 without libfuse, a
Linux-only dependency because its build script panics on Windows and macOS. Windows: WinFsp - Windows
File System Proxy, Copyright (C) Bill Zissimopoulos, <https://github.com/winfsp/winfsp>, reached through
our own FFI: its DLL is loaded at run time by full path from the installer's registry entry, so nothing
of WinFsp is needed to build (windows-gnu included) and a machine without it gets a typed error naming
<https://winfsp.dev/rel/>. Win32 goes through `windows-sys`. WinFsp is GPL-3.0 with a FLOSS exception:
ziptensors (MIT OR Apache-2.0) may use the DLL provided its user interface and user-facing documentation
carry that notice and link (`ziptensors mount --help` on Windows does), and a proprietary program
embedding `ziptensors-mount` on Windows needs WinFsp's commercial license. No WinFsp file is distributed.

## Extension points

- New stream codecs, steps, views, sections, dtypes: new ids by a minor version, with the fail-closed
  rules above protecting older readers. Anything not additive needs a new major, and readers keep
  reading older majors (spec section 3.4).
- New source formats: a new parser in `profile` returning a `RangePlan`.
- Frontends and adapters sit above `read_at` and need no format change.
- Reserved in the spec without semantics: derived blobs, multi-volume, and delta operations other than
  `delta_xor` (step ids 0x0401 to 0x04FF), each a new `DeltaOp` variant.
