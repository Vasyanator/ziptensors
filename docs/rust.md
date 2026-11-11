# Rust library guide

`ziptensors-core` is the library behind every ziptensors frontend: the command line tool, the Python
module, the C ABI and the mount all call it. This guide shows how to use it from Rust, task by task,
then lists the types and errors. The complete API reference, with every type, field and error
contract, is on [docs.rs/ziptensors-core](https://docs.rs/ziptensors-core) and
[docs.rs/ziptensors-mount](https://docs.rs/ziptensors-mount).

Related pages: [cli.md](cli.md) for the command line, [editing.md](editing.md) for the edit
strategies in depth, [directories.md](directories.md) for per-file compression of a model tree,
[mount.md](mount.md) for the mount, and [safety.md](safety.md) for untrusted input.

## Adding the dependency

```toml
[dependencies]
ziptensors-core = "0.2"
# only if you mount containers (Linux and Windows):
ziptensors-mount = "0.2"
```

- Rust 1.92 or newer (edition 2024).
- A C compiler for the target, because the `zstd-sys` dependency compiles zstd from source. No
  system library is linked.
- The crate version (0.2.x) is independent of the container format version (0.3). Files written by
  any published version stay readable by later versions; see the format spec,
  `spec/ziptensors-format.md`, section 3.4.
- The library logs through `tracing` and installs no subscriber. Install one in your application
  (for example `tracing-subscriber`) to see its log lines.

All examples below use relative paths and `?` on `ziptensors_core::Error`. They compile against
version 0.2.0.

## Quick start

```rust
use std::path::Path;

use ziptensors_core::{Container, EncodeOptions, OpenOptions, encode_file};

fn main() -> Result<(), ziptensors_core::Error> {
    // Compress one file into a container of one member, named after the file.
    let report = encode_file(
        Path::new("model.safetensors"),
        Path::new("model.ziptensors"),
        &EncodeOptions::for_encode(),
    )?;
    println!("{} -> {} bytes", report.total.raw, report.output_len);

    // A file from elsewhere is untrusted input: always give it a decode budget.
    let options = OpenOptions { max_total_decoded: 64 << 30, ..OpenOptions::default() };
    let container = Container::open(Path::new("model.ziptensors"), &options)?;
    let member = container.single()?;
    member.restore_file(Path::new("restored.safetensors"))?;
    Ok(())
}
```

## Reading

### Opening a container

`Container::open(path, &OpenOptions)` reads the header, the footer and the index, validates them,
and reads no tensor data. A file is recognized by its magic bytes, never by its extension.
`Container::from_source(Arc<dyn RandomAccess>, &OpenOptions)` opens a container held by any other
positional byte source.

`OpenOptions` fields:

| Field | Default | Meaning |
|---|---|---|
| `threads` | `0` | Decode threads and scratch pool size. 0 = one per logical CPU; 1 = everything inline on the calling thread. |
| `max_index_bytes` | 256 MiB | Cap on the decompressed index. Values over 1 GiB act as 1 GiB. |
| `max_total_decoded` | `u64::MAX` (no budget) | Budget on the sum of member lengths. A container over it is refused at open, before any member data is read. **Set it for untrusted input.** 0 is a budget of zero bytes. |
| `max_decode_amplification` | `16` | Bound on the container's decode work as a multiple of the sum of member lengths, on top of one floor of 256 MiB. Raise it only for a file whose small extents point into large chunks. |
| `verify` | `Verify::Chunk` | `Chunk` checks the XXH3-64 of every decoded chunk; `Off` skips chunk hashes (exact lengths are still enforced). |

### Members and tensors

```rust
use std::path::Path;

use ziptensors_core::{Container, OpenOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = OpenOptions { max_total_decoded: 64 << 30, ..OpenOptions::default() };
    let container = Container::open(Path::new("model.ziptensors"), &options)?;

    for info in container.members() {
        println!("{} {} bytes, {:?}, {} tensors", info.name, info.len, info.profile, info.tensor_count);
    }

    let member = container.member("model.safetensors")?;
    let mut header_len = [0u8; 8];
    member.read_at(0, &mut header_len)?;

    for tensor in member.tensors() {
        println!("{} {:?} {:?} at {}+{}", tensor.name, tensor.dtype, tensor.shape, tensor.offset, tensor.len);
    }
    let tensor = &member.tensors()[0];
    let mut bytes = vec![0u8; usize::try_from(tensor.len)?];
    member.decode_tensor_into(&tensor.name, &mut bytes)?;
    Ok(())
}
```

- `Container::members()` returns `MemberInfo { name, len, blake3, profile, tensor_count }` in file
  order. `container.member(name)` opens one by name; `container.single()` opens the only member of a
  single-file container and fails with `MemberNotFound` otherwise.
- A `Member` is a virtual file: `len()`, `is_empty()`, `name()`, `blake3()` (the hash the file
  states), `profile()` (the recognized source format, a hint) and `tensors()`.
- `tensors()` is the tensor table: `TensorInfo { name, offset, len, shape, dtype }`. It is metadata
  a parser produced: it never decides restored bytes.
- `read_at(offset, &mut dst)` fills all of `dst` or fails. A range outside the member fails with
  `OutOfBounds` before any I/O. On any other error, `dst` holds unspecified bytes.
- `decode_tensor_into(name, &mut dst)` reads one tensor; `dst.len()` must equal the tensor's length
  (`DestinationSize` otherwise).
- A read of part of a chunk decodes the whole chunk, and `Member` caches nothing between calls. For
  many small reads use `ReadCache` (below).

### Many ranges at once

`read_many(&mut [ReadRequest])` fills several ranges and visits the file in pack-offset order, so a
disk moves one way. The requests keep their order; every range is checked before any I/O.

```rust
use std::path::Path;

use ziptensors_core::{Container, OpenOptions, ReadRequest};

fn main() -> Result<(), ziptensors_core::Error> {
    let container = Container::open(Path::new("model.ziptensors"), &OpenOptions::default())?;
    let member = container.single()?;
    let mut a = vec![0u8; 4096];
    let mut b = vec![0u8; 4096];
    let mut reqs = [
        ReadRequest { offset: 1 << 17, dst: &mut a },
        ReadRequest { offset: 0, dst: &mut b },
    ];
    member.read_many(&mut reqs)?;
    Ok(())
}
```

### Restoring and verifying

```rust
use std::path::Path;

use ziptensors_core::{Container, OpenOptions};

fn main() -> Result<(), ziptensors_core::Error> {
    let options = OpenOptions { max_total_decoded: 64 << 30, ..OpenOptions::default() };
    let container = Container::open(Path::new("model.ziptensors"), &options)?;
    let member = container.member("model.safetensors")?;

    member.verify()?;
    let report = member.restore_file(Path::new("model.restored.safetensors"))?;
    println!("{} bytes", report.bytes);

    let all = container.restore_all(Path::new("restored"))?;
    println!("{} members, {} bytes", all.members, all.bytes);

    let mut sink = std::io::sink();
    member.decode_to(&mut sink)?;
    Ok(())
}
```

| Method | What it does | Failure guarantee |
|---|---|---|
| `Member::verify()` | Decodes the whole member and compares its BLAKE3 with the stated hash. Writes nothing. | `VerifyFailed` if the hash differs. |
| `Member::restore_file(out)` | Writes the member to `out` through a temporary file in the same directory: the bytes appear only after they hashed to the member's BLAKE3 and were synced, as one rename (which replaces a file already at `out`). Returns `RestoreReport { bytes, blake3 }`. | On failure there is no new file, what was at `out` is untouched, and the temporary file is gone, except `CommittedNotDurable` (the verified file is in place, the directory sync failed). `OutputIsInput` if `out` is the container itself. |
| `Member::decode_to(&mut dyn Write)` | Streams the member in order into a writer and checks the BLAKE3 at the end. | **Unverified bytes reach the writer**: a wrong member hash is reported only after the last byte was written. On any `Err`, discard what the writer received. Each chunk is still checked against its own hash unless `Verify::Off`. |
| `Container::restore_all(dir)` | Restores every member to `dir` joined with its name, in file order, creating directories. Returns `RestoreAllReport { members, bytes }`. | Every name is checked with `format::check_restore_name` first: one unsafe name fails with `UnsafeMemberName` and nothing is created. After that, a failure stops the run and keeps the members restored before it. |

Restore is content-only: modification times, permissions and symbolic links are not restored.

### ReadCache: many small reads

`ReadCache` is the one opt-in cache, for readers whose requests are smaller than a chunk (a file
object, a mount). It keeps decoded windows (one chunk of one extent) LRU by bytes and fills a miss
with read-ahead.

```rust
use std::path::Path;

use ziptensors_core::{CacheOptions, Container, OpenOptions, ReadCache};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let container = Container::open(Path::new("model.ziptensors"), &OpenOptions::default())?;
    let cache = ReadCache::new(container, CacheOptions { max_bytes: 64 << 20, ..CacheOptions::default() })?;
    let member = cache.member("model.safetensors")?;
    let mut small = [0u8; 512];
    for offset in (0..member.len().saturating_sub(512)).step_by(512) {
        member.read_at(offset, &mut small)?;
    }
    let stats = cache.stats();
    println!("hits {} misses {} peak {}", stats.hits, stats.misses, stats.peak_bytes);
    Ok(())
}
```

| `CacheOptions` field | Default | Meaning |
|---|---|---|
| `max_bytes` | 256 MiB | Most decoded bytes held, resident and in flight together. Must be at least 2 × the largest `chunk_size` of the container, or `ReadCache::new` fails with `BudgetExceeded` (`what` = `"read cache window"`). |
| `readahead_windows` | `0` | Windows decoded per miss, the missed one included. 0 = the container's decode thread count; 1 = no read-ahead. |

`CachedMember::read_at` returns exactly what `Member::read_at` returns and fails only where it
fails. `cache.member_at(index)` opens by file-order index, `cached.member()` gives the uncached
handle (name, hash, tensor table), and `cache.stats()` returns `CacheStats` (`hits`, `misses`,
`decoded_bytes`, `evictions`, `resident_bytes`, `in_flight_bytes`, `peak_bytes`).

## Writing

### One file: `encode_file`

`encode_file(input, output, &EncodeOptions)` writes a container of one member named after the input
file. It fails with `UnsafeMemberName` if the file name is not UTF-8 or a restore would refuse it.

`EncodeOptions` has two fields and no `Default`, because the two entry points have different
defaults:

| Constructor | `mode` | `threads` |
|---|---|---|
| `EncodeOptions::for_encode()` | `Mode::Balanced` | 0 (one per CPU) |
| `EncodeOptions::for_pack()` | `Mode::Max` | 0 (one per CPU) |

Or build it directly: `EncodeOptions { mode: Mode::Fast, threads: 4 }`. The output bytes do not
depend on `threads`.

| `Mode` | Transforms |
|---|---|
| `Fast` | Plane split and per-stream entropy coding only. Quantized data (GGUF blocks) is stored uncompressed. |
| `Balanced` | Adds widening views (F32 holding F16/BF16 values) and field split of quantized data. |
| `Max` | Adds the bit rotation trial per blob and blob deduplication. Only `Max` stores delta blobs (when a base is named). |

Modes choose transforms only; a reader cannot tell them apart and every mode restores the same bytes.

### Several files: `PackBuilder`

```rust
use std::path::Path;

use ziptensors_core::{Container, EncodeOptions, Mode, OpenOptions, PackBuilder};

fn main() -> Result<(), ziptensors_core::Error> {
    let options = EncodeOptions { mode: Mode::Max, threads: 0 };
    let mut builder = PackBuilder::new(Path::new("family.ziptensors"), &options)?;
    builder.add_file(Path::new("model.safetensors"), "base/model.safetensors")?;
    builder.add_file_delta(
        Path::new("finetune.safetensors"),
        "finetune/model.safetensors",
        "base/model.safetensors",
    )?;
    let other = Container::open(Path::new("model.ziptensors"), &OpenOptions::default())?;
    builder.add_member_from(&other, "model.safetensors")?;
    let report = builder.finish()?;
    for member in &report.members {
        println!("{}: {} delta blobs, {} compressed bytes", member.name, member.delta_blobs, member.compressed_bytes);
    }
    println!("verified {} bytes in {:?}", report.verify.bytes, report.verify.elapsed);
    Ok(())
}
```

- `PackBuilder::new(output, &options)` creates a temporary file next to `output`. Nothing appears at
  `output` until `finish` has proven the whole file. Dropping a builder removes its temporary file.
- `add_file(path, member)` adds a file under a `/`-separated member name, which a restore must
  accept and which must be unique in the pack. Symbolic links are refused (`SymlinkInput`), never
  followed. On `Err` the member is not added and the builder stays usable, unless the output itself
  failed.
- `add_file_delta(path, member, base_member)` stores the file's tensors that pair with tensors of an
  earlier member of this builder as delta blobs (a residual against the base) where that is smaller.
  A tensor pairs by name (also after dropping a publication prefix such as `model.diffusion_model.`)
  with the same length and dtype; a tensor equal to its base is stored by reference. Only `Mode::Max`
  stores delta blobs; under other modes it behaves as `add_file`. `base_member` must have been added
  from a file by this builder, and that file must be unchanged, since it is read again. A container
  holding delta blobs sets a required feature bit, so readers older than format 0.3 refuse it.
- `add_member_from(&container, member)` copies a member of another open container with no decoding
  and no recompression.
- `finish()` writes the index, decodes every member through the real reader and compares it with the
  hash of its source, re-checks that the inputs did not change, syncs, and renames the file over
  `output`. The decode cannot be skipped. It returns an `EncodeReport`: `output_len`,
  `index_stored_len`, per-member `MemberReport`s, byte accounting (`total`, `classes`) and the
  verification time.

`layer_report(&container)` gives the same byte accounting for an existing container from its index
alone.

### Collecting inputs from directories

`collect_inputs(&[PathBuf])` walks roots the way `ziptensors pack` does and returns
`Inputs { files: Vec<(PathBuf, String)>, skipped_symlinks, skipped_incomplete }`. It reads only
directories and metadata. Member names are relative to the root they were found under; a root that is
a file is named after itself. Links and `*.incomplete` files are skipped, and every name has passed
`check_restore_name` and is unique.

```rust
use std::path::{Path, PathBuf};

use ziptensors_core::{EncodeOptions, PackBuilder, collect_inputs};

fn main() -> Result<(), ziptensors_core::Error> {
    let roots = [PathBuf::from("models")];
    let inputs = collect_inputs(&roots)?;
    let mut builder = PackBuilder::new(Path::new("models.ziptensors"), &EncodeOptions::for_pack())?;
    for (path, name) in &inputs.files {
        builder.add_file(path, name)?;
    }
    builder.finish()?;
    println!("{} files, {} links skipped", inputs.files.len(), inputs.skipped_symlinks);
    Ok(())
}
```

Do not put the output inside a walked root: a later run would pick it up as an input.
`ziptensors_core::write::inputs::check_output(output, roots)` refuses that case with `OutputIsInput`.

## Editing a container

`edit(path, &[EditOp], &EditOptions)` adds, removes and replaces members of an existing container.
The operations are a set: each member name appears at most once. Kept members keep their order, a
`Replace` keeps its member's slot, and `Add`s follow in operation order. Every check runs before any
byte is written.

```rust
use std::path::{Path, PathBuf};

use ziptensors_core::{EditOp, EditOptions, Strategy, edit};

fn main() -> Result<(), ziptensors_core::Error> {
    let ops = [
        EditOp::Add {
            path: PathBuf::from("extra.safetensors"),
            name: "extra/model.safetensors".to_owned(),
            delta_base: None,
        },
        EditOp::Remove { name: "model.safetensors".to_owned() },
    ];
    let options = EditOptions { strategy: Strategy::Append, ..EditOptions::default() };
    let report = edit(Path::new("family.ziptensors"), &ops, &options)?;
    println!(
        "kept {} added {} removed {}, generation {}",
        report.kept, report.added, report.removed, report.generation
    );

    // Compaction: an edit with no operation, under Rewrite (the default) or InPlace.
    let compacted = edit(Path::new("family.ziptensors"), &[], &EditOptions::default())?;
    println!("dropped {} dead bytes", compacted.dead_bytes_before);
    Ok(())
}
```

`EditOp` variants: `Add { path, name, delta_base }`, `Remove { name }`,
`Replace { name, path, delta_base }`. `delta_base` names a member of the result to store the file
against: a kept member, or a file added or replaced earlier in the same edit.

| `EditOptions` field | Default | Meaning |
|---|---|---|
| `strategy` | `Strategy::Rewrite` | How the result replaces the container (below). |
| `mode` | `Mode::Max` | Mode of new files, and of every member under `repack`. |
| `repack` | `false` | Re-encode every kept member with `mode`, redoing deduplication and delta pairing. Only with `Rewrite`. |
| `threads` | `0` | Compression and verification threads; 0 = one per CPU. |
| `open` | `OpenOptions::default()` | How the source container is opened. Set `open.max_total_decoded` for an untrusted container. |

| `Strategy` | Space needed | On failure |
|---|---|---|
| `Rewrite` | A second full copy | The container is intact, except after `CommittedNotDurable`: the new container already replaced it (see [editing.md](editing.md#default-rewrite-through-a-temporary-file)). Dead data of earlier edits is dropped. Drop any `Container` you hold on the file first (Windows may refuse to rename over an open file). |
| `Append` | Only the new data | The file is cut back to its old length; if that cut fails, `TrailingBytes` (run `recover`). Refused with `repack`, with no operation, and with a delta base unless the container already holds delta blobs. A `Container` you hold keeps reading the old generation. |
| `InPlace` | None | **Risky.** A failure after the first write leaves a damaged container (`DamagedInPlace`); keep a backup or the original inputs. Nothing may read the file meanwhile. Refused with `repack` and with a delta base that is a kept member. |

`Strategy` is `#[non_exhaustive]`: a `match` on it needs a wildcard arm.

There is no separate `compact` function: an edit with an empty operation list compacts the container
under `Rewrite` or `InPlace`, and is refused under `Append`. `EditReport` carries `kept`, `added`,
`removed`, `replaced`, the new footer `generation`, the writer's `encode` report, and
`dead_bytes_before` (bytes of the source no member needed). See [editing.md](editing.md) for the
crash behavior of each strategy in detail.

## Recovering a torn file

A reader opens only the generation named by the last 64 bytes of the file and never falls back to an
older one. When an append was interrupted, or an edit returned `TrailingBytes`, `recover(path,
&RecoverOptions)` shortens the file to its latest valid generation. It never runs on its own.

```rust
use std::path::Path;

use ziptensors_core::{RecoverOptions, recover};

fn main() -> Result<(), ziptensors_core::Error> {
    let dry = RecoverOptions { dry_run: true, ..RecoverOptions::default() };
    let report = recover(Path::new("family.ziptensors"), &dry)?;
    println!(
        "{} -> {} bytes, generation {}, {} newer candidates passed over",
        report.file_len, report.valid_len, report.generation, report.passed_over
    );
    Ok(())
}
```

| `RecoverOptions` field | Default | Meaning |
|---|---|---|
| `open` | `OpenOptions::default()` | Limits every candidate is judged under. Tighter limits than the writer used stop the scan at a genuine generation instead of skipping it, so keep the defaults unless you know better. |
| `budget` | `ScanBudget { max_candidates: 64, max_scan_bytes: u64::MAX }` | Bounds of the backward scan. |
| `verify` | `true` | Decode every member of the found generation and check its BLAKE3 before cutting; a generation that fails is passed over for an older one. Turn off only for a file never edited in place. |
| `dry_run` | `false` | Report and write nothing (the file is opened read-only). |

`RecoverReport` fields: `file_len`, `valid_len` (equal to `file_len` when the file opened as it
was), `generation`, `truncated`, `verified_bytes` (`None` when nothing was verified), `scanned_bytes`,
and `passed_over`: the number of newer candidate generations found and rejected (a footer whose hash
holds but whose prefix did not open, or did not verify). Above 0, data a newer generation may have
held is in the bytes cut off. A torn append reports 0.

Errors: `Unrecoverable` if no valid generation was found within the budget; with verification, the
verification error of the newest structurally valid generation if none verifies; `Io`. Every error
leaves the file unchanged, except an `Io` with op `"sync"` after the cut (the file is cut, but the cut
may not survive a power loss). `find_last_valid_generation` runs the scan alone, read-only.

## Directory trees: per-file containers

`ziptensors_core::tree` compresses every weight file under a directory into a container of one member
beside it (`dir/name.safetensors` becomes `dir/name.ziptensors`), and back. See
[directories.md](directories.md) for the behavior in detail.

```rust
use std::path::Path;

use ziptensors_core::{
    CompressTreeOptions, DecompressTreeOptions, ViewEntry, ViewOptions, compress_tree, decompress_tree,
    scan_view,
};

fn main() -> Result<(), ziptensors_core::Error> {
    let mut options = CompressTreeOptions::default();
    options.max_total_decoded = 64 << 30;
    let report = compress_tree(Path::new("models"), &options)?;
    let totals = report.totals();
    println!("{} compressed, {} failed", totals.compressed, totals.failed);
    for file in &report.files {
        println!("{}: {:?}", file.path.display(), file.status);
    }

    let mut view_options = ViewOptions::default();
    view_options.max_total_decoded = 64 << 30;
    let view = scan_view(Path::new("models"), &view_options)?;
    for entry in &view.entries {
        if let ViewEntry::PerFile(per_file) = entry {
            println!("{} shows as {}", per_file.container_rel, per_file.rel_path);
        }
    }

    let mut back = DecompressTreeOptions::default();
    back.max_total_decoded = 64 << 30;
    let report = decompress_tree(Path::new("models"), &back)?;
    if report.has_failures() {
        eprintln!("some files failed");
    }
    Ok(())
}
```

The option structs of `tree` are `#[non_exhaustive]`: build them with `default()` and set fields, as
above.

| `CompressTreeOptions` field | Default | Meaning |
|---|---|---|
| `keep` | `false` | Keep each original beside its container. Otherwise an original is removed once its container is committed and verified. |
| `follow_symlinks` | `false` | Follow links to regular files: the target's bytes become a container at the link's place, and the link (never its target) is removed. By default links are skipped with a warning. |
| `dry_run` | `false` | Report what would be done, write nothing. |
| `encode` | `EncodeOptions::for_pack()` | How each container is written. |
| `max_total_decoded` | `u64::MAX` | Budget per container that is opened. |

| `DecompressTreeOptions` field | Default | Meaning |
|---|---|---|
| `keep` | `false` | Keep each container beside its restored file. |
| `force` | `false` | Restore over a different file already at the restore path. |
| `dry_run` | `false` | Report what would be done, write nothing. |
| `threads` | `0` | Decode threads per container. |
| `max_total_decoded` | `u64::MAX` | Budget per container. Set it for untrusted files. |

Both functions plan the whole run first: colliding names refuse it with `TreeConflict` before
anything is written. After that each file runs on its own, and a failure is recorded in its
`FileOutcome` (`path`, `counterpart`, `status`, lengths, `removed`) without stopping the run.
`TreeReport` has `files`, `warnings`, `totals()` and `has_failures()`. A rerun is idempotent.

`scan_view(root, &ViewOptions)` classifies a tree without writing anything: `ViewEntry::Directory`,
`Plain`, `PerFile` (a container whose restore path is free), `Shadowed` (its restore path is taken),
`Foreign` (a container that is not a per-file container), `Broken`, and `Skipped`. `TreeView::conflicts`
lists names claimed twice.

## Mounting (`ziptensors-mount`)

`ziptensors-mount` shows a container, or a directory tree of per-file containers, as read-only files,
so unmodified programs read compressed models. It exists on Linux (FUSE, needs `/dev/fuse` and
`fusermount3`) and Windows (WinFsp); there is no mount on macOS. See [mount.md](mount.md) for usage
and requirements.

```rust
use std::path::Path;
use std::sync::Arc;

use ziptensors_core::{CacheOptions, Container, OpenOptions, ReadCache};
use ziptensors_mount::{FuseOptions, MountFs, PosixPolicy, mount};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("model.ziptensors");
    let options = OpenOptions { max_total_decoded: 64 << 30, ..OpenOptions::default() };
    let container = Container::open(path, &options)?;
    let cache = ReadCache::new(container, CacheOptions::default())?;
    let modified = std::fs::metadata(path)?.modified()?;
    let fs = MountFs::new(cache, PosixPolicy, modified)?;

    let fuse = FuseOptions { threads: 4, fs_name: path.display().to_string() };
    let handle = mount(Arc::new(fs), Path::new("/mnt/model"), &fuse)?;
    // ... other programs read /mnt/model/model.safetensors ...
    handle.unmount()?;
    Ok(())
}
```

- `MountFs::new(cache, policy, modified)` builds the file tree of one container. Member names the
  policy cannot show (`PosixPolicy` on Linux, `WindowsPolicy` on Windows) refuse the whole mount with
  `MountError::Names`.
- `MountFs::overlay(root, policy, &OverlayOptions)` mounts a directory tree instead: plain files as
  they are, each per-file container under its original name. `OverlayOptions` is
  `#[non_exhaustive]` (fields `threads`, `cache`, `max_total_decoded`, `max_open_containers` = 4,
  `max_open_files` = 64); set `max_total_decoded` for untrusted files.
- On Linux, `mount(Arc<MountFs>, mountpoint, &FuseOptions)` returns a `MountHandle`:
  `unmount()` detaches the mount and waits until every file open under it is closed; `wait()` blocks
  until the mount is ended from outside; dropping the handle unmounts. `run_until_signal(handle,
  &UnmountSignals::block()?)` serves until SIGINT, SIGTERM or SIGHUP, which is what the command line
  tool does.
- On Windows, `mount(Arc<MountFs>, mountpoint, &WinFspOptions)` takes a drive letter (`X:`) or a
  directory that does not exist yet, and `probe()` reports whether WinFsp is installed. WinFsp must be
  installed from <https://winfsp.dev/rel/>; a machine without it gets a typed error naming that page.
  WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos,
  <https://github.com/winfsp/winfsp>. WinFsp is GPL-3.0 with a FLOSS exception: a program built on
  `ziptensors-mount` for Windows must show this notice and link in its user interface and
  documentation, and a proprietary program needs WinFsp's commercial license.

## Errors

Every fallible function returns `Result<_, ziptensors_core::Error>`. `Error` is `#[non_exhaustive]`
and implements `std::error::Error` and `Display` with a readable message. For a stable
classification, use `Error::code()`, which returns an `ErrorCode`. `ErrorCode` is exhaustive and its
numeric values (`code.as_i32()`) never change; they are the same numbers as the positive C status
codes.

```rust
use std::path::Path;

use ziptensors_core::{Container, ErrorCode, OpenOptions};

fn main() {
    match Container::open(Path::new("notes.txt"), &OpenOptions::default()) {
        Ok(_) => {}
        Err(error) => match error.code() {
            ErrorCode::NotZiptensors => eprintln!("not a container: {error}"),
            ErrorCode::BudgetExceeded => eprintln!("too large for the budget: {error}"),
            code => eprintln!("error {}: {error}", code.as_i32()),
        },
    }
}
```

| Code | `ErrorCode` | Meaning | What to do |
|---|---|---|---|
| 1 | `Io` | An OS call failed; the message names the operation and path. | Check the path, permissions, free space. |
| 2 | `NotZiptensors` | No ziptensors magic, or too short. | The file is not a container. |
| 3 | `UnsupportedVersion` | A format major this library does not read. | Use a newer library. |
| 4 | `UnsupportedFeature` | A required feature bit this library does not implement. | Use a newer library. |
| 5 | `UnknownCodec` | An unknown step, view or stream codec id. | Use a newer library. |
| 6 | `MalformedIndex` | Header, footer or index violates the format, a member name stored twice included. | The file is damaged or hostile; if an append was torn, try `recover`. |
| 7 | `BudgetExceeded` | A declared size is over a limit of `OpenOptions` (or a cache budget). Raised before data is read. | Raise the limit only if you trust the file. |
| 8 | `ChunkCorrupt` | A chunk cannot be decoded. | The file is damaged. |
| 9 | `ChecksumMismatch` | A chunk decoded to bytes that do not match its XXH3. | The file is damaged. |
| 10 | `OutOfBounds` | A read range is outside the member. | Fix the offset or length. |
| 11 | `DestinationSize` | The destination buffer has the wrong length. | Size it to the tensor. |
| 12 | `TensorNotFound` | No tensor of that name. | Check `tensors()`. |
| 13 | `MemberNotFound` | No such member, or `single()` on a pack. | Check `members()`. |
| 14 | `UnsafeMemberName` | A name a restore must refuse (`..`, absolute, `:`, NUL, empty). Nothing was written for it. | Rename the input; for a received container, do not restore it with `restore_all`. |
| 15 | `VerifyFailed` | A member's bytes do not hash to its stated BLAKE3. | The file is damaged, or an input did not hold what it stated. |
| 16 | `InputChanged` | An input changed while it was encoded, or a container's length changed during an edit. No output was committed. | Retry when nothing writes the input. |
| 17 | `Zstd` | zstd failed outside chunk decoding. | Report it. |
| 18 | `SymlinkInput` | An input is a symbolic link; links are not followed. | Pass the target. |
| 19 | `OutputIsInput` | The output is the file being read. | Choose another output. |
| 20 | `CommittedNotDurable` | The verified output IS in place, but the directory sync failed: a power loss may bring back the old file. Unix only. | Usually nothing; sync later if durability matters. |
| 21 | `DamagedInPlace` | An in-place edit failed after it began moving data: the container is damaged. | Restore from a backup or repack from the inputs. |
| 22 | `EditRefused` | An edit request cannot be applied as asked. Nothing was written. | Change the request, usually the strategy. |
| 23 | `TrailingBytes` | A valid container is followed by bytes that could not be cut off. | Run `recover` once nothing else holds the file. |
| 24 | `Unrecoverable` | `recover` found no valid generation. Nothing was changed. | Restore from a backup. |
| 25 | `TreeConflict` | A tree run found colliding names and was refused. Nothing was written. | Rename or remove the files named. |
| 26 | `DuplicateMember` | A write would store two members under one name: `PackBuilder` already has it, or two inputs of `collect_inputs` or `check_member_names` give it. That member was not added. | Rename one input. A frontend that copies members calls `check_member_names` over all names before it starts the output. |

## Threads and memory

- `Container`, `Member`, `ReadCache` and `CachedMember` are `Clone` (cheap, shared), `Send` and
  `Sync`; all their methods take `&self` and may be called from any number of threads at once. A
  `Member` keeps its container alive.
- Reads: the calling thread does all I/O, and the decode runs on a thread pool private to the
  `Container` (`threads = 1` runs inline). Above the destination buffer and the index, a read holds at
  most **3 × threads × chunk size** bytes, or **(4 × threads + 1) × chunk size** when it touches delta
  blobs. These bounds hold for hostile files too.
- `ReadCache` holds at most `max_bytes` of decoded windows plus one window per concurrent miss, plus
  the read bound above per fill. It may be used from tasks of any rayon pool.
- Writes: one reader thread, a rayon compression pool, one ordered writer, at most
  **6 × threads × chunk size** plus the index (**9 × threads × chunk size** once a delta blob was
  streamed). No write holds a whole tensor in memory.
- Every write ends in a full decode of the result before it becomes visible; there is no way to skip
  it. See [safety.md](safety.md) for what that does and does not guarantee.
