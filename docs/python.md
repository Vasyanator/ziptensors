# Python API

The `ziptensors` Python package reads, writes, edits and mounts ziptensors containers. It is a compiled
extension over the same Rust library as the command line tool, plus two pure-Python adapters:
`ziptensors.torch` (load tensors into PyTorch) and `ziptensors.safetensors` (the `safe_open` reading API
of the safetensors package over a container).

This page starts with task-oriented examples and then lists every class and function. For the command
line see [cli.md](cli.md), for editing strategies [editing.md](editing.md), for whole directory trees
[directories.md](directories.md), for mounts [mount.md](mount.md), and for untrusted input
[safety.md](safety.md).

- [Installation](#installation)
- [Examples](#examples)
- [Reference: reading](#reference-reading)
- [Reference: writing](#reference-writing)
- [Reference: editing and recovery](#reference-editing-and-recovery)
- [Reference: directory trees](#reference-directory-trees)
- [Reference: mounting](#reference-mounting)
- [Reference: logging and version](#reference-logging-and-version)
- [Reference: report classes](#reference-report-classes)
- [Errors](#errors)
- [Threads and the GIL](#threads-and-the-gil)
- [`ziptensors.torch`](#ziptensorstorch)
- [`ziptensors.safetensors`](#ziptensorssafetensors)

## Installation

```text
pip install ziptensors            # the module
pip install "ziptensors[torch]"   # with PyTorch (torch >= 2.1) for ziptensors.torch
```

- Requires CPython 3.11 or later. The wheels are built for the stable ABI (abi3), so one wheel serves
  every CPython version from 3.11 on.
- Wheels are published for Linux x86_64 and aarch64 (manylinux 2.28, glibc 2.28 or later), Windows x64,
  and macOS x86_64 and arm64. Any other platform builds from the source distribution, which needs a
  Rust toolchain and a C compiler.
- `import ziptensors` imports neither torch nor numpy. `ziptensors.torch` imports torch on its first
  call, and `ziptensors.safetensors` imports torch or numpy for the framework you ask for.
- Mounting works on Linux (FUSE) and Windows (WinFsp, installed separately); on macOS the mount calls
  raise `MountError`. See [mount.md](mount.md).

All sizes and offsets are in bytes. Every function that takes a path accepts `str` or any
`os.PathLike[str]`; paths returned by the module are `pathlib.Path`.

## Examples

The examples use relative paths such as `model/` and `pipeline.ziptensors`; run them from a directory
that holds such files.

### Open a container and read a tensor

```python
import numpy as np
import ziptensors

ziptensors.pack(["model/"], "pipeline.ziptensors")

with ziptensors.Container("pipeline.ziptensors", max_total_decoded=64 << 30) as container:
    for info in container.members:
        print(info.name, info.len, info.profile, info.tensor_count)

    member = container.member("unet/diffusion_pytorch_model.safetensors")
    for tensor in member.tensors:
        print(tensor.name, tensor.dtype, tensor.shape, tensor.len)

    tensor = member.tensor_info("conv.weight")
    array = np.empty(tensor.shape, dtype=np.float32)
    member.decode_tensor_into("conv.weight", array)
    print(array[0, :4])

    header = member.read(0, 8)  # any byte range of the original file
    print(int.from_bytes(header, "little"))
```

A member is a byte-exact copy of the original file, so `read` and `read_at` take offsets of that file.
`decode_tensor_into` fills any writable, C-contiguous buffer of exactly `Tensor.len` bytes; it decodes
with the GIL released and copies nothing in between. Pass `max_total_decoded` whenever the file comes
from somewhere you do not control (see [safety.md](safety.md)).

### Load into PyTorch

```python
import torch
import ziptensors
from ziptensors.torch import load_state_dict

container = ziptensors.Container("pipeline.ziptensors", max_total_decoded=64 << 30)
member = container.member("unet/diffusion_pytorch_model.safetensors")
state = load_state_dict(member, device="cpu", dtype=torch.float16)
print({name: (t.dtype, tuple(t.shape)) for name, t in state.items()})
```

`device="cuda"` decodes through pinned staging buffers and copies to the GPU while the next piece
decodes. A legacy PyTorch file (`.bin`, `.pt`) is a pickle, and goes through `torch.load` with
`weights_only=True`:

```python
import torch
import ziptensors

torch.save({"w": torch.arange(10)}, "legacy.bin")
ziptensors.encode_file("legacy.bin", "legacy.ziptensors")

member = ziptensors.Container("legacy.ziptensors").member()
print(member.profile)                                  # pt_zip
state = torch.load(member.open(), weights_only=True)
print(state["w"])
```

Never pass `weights_only=False` for a file you did not make yourself: the full unpickler runs arbitrary
code from the file.

### Read like `safetensors.safe_open`

```python
from ziptensors.safetensors import safe_open

with safe_open("pipeline.ziptensors", framework="pt", device="cpu",
               member="vae/diffusion_pytorch_model.safetensors") as f:
    print(f.keys(), f.metadata())
    weight = f.get_tensor("enc.weight")
    rows = f.get_slice("enc.weight")[0:4]
    print(weight.dtype, tuple(rows.shape))
```

### Pack a model folder

```python
import ziptensors

ziptensors.enable_stderr_logging("warn")
found = ziptensors.collect_inputs(["model/"])
for path, name in found.files:
    print(name, "<-", path)

report = ziptensors.pack(["model/"], "pipeline.ziptensors", mode="max")
print(report.output_len, report.total.raw, report.total.stored)
for member in report.members:
    print(member.name, member.len, member.layers.stored)
```

A directory's files become members named relative to it (`unet/diffusion_pytorch_model.safetensors`).
The new file is decoded and compared with its sources before it replaces anything at the output path.
For a single file, `ziptensors.encode_file("model.safetensors", "model.ziptensors")` writes a container
of one member.

### Edit a container

```python
import shutil
import ziptensors
from ziptensors import AddOp, RemoveOp, ReplaceOp

shutil.copy("model/vae/diffusion_pytorch_model.safetensors", "vae2.safetensors")
report = ziptensors.edit(
    "pipeline.ziptensors",
    [AddOp("vae2.safetensors", "vae2/diffusion_pytorch_model.safetensors"),
     RemoveOp("model_index.json")],
)
print(report.kept, report.added, report.removed, report.generation)

ziptensors.edit(
    "pipeline.ziptensors",
    [ReplaceOp("vae/diffusion_pytorch_model.safetensors", "vae2.safetensors")],
    strategy="append",
)
report = ziptensors.compact("pipeline.ziptensors")
print(report.dead_bytes_before)
print(ziptensors.recover("pipeline.ziptensors", dry_run=True).truncated)
```

The default strategy, `"rewrite"`, writes a proven copy and renames it over the file. `"append"` writes
only the new data after the end; `compact` later drops what appends left behind. `recover` cuts the
torn tail of an interrupted append; after an interrupted in-place edit it usually cannot help.
[editing.md](editing.md) compares the strategies.

### Compress a directory tree in place

```python
import os
import ziptensors

report = ziptensors.compress_tree("tree2/", keep=False)
for outcome in report.files:
    print(outcome.status, outcome.path.name, outcome.counterpart and outcome.counterpart.name)
print(report.totals.compressed, report.has_failures)

os.makedirs("mnt", exist_ok=True)                     # Linux; see mount.md for Windows
with ziptensors.mount_overlay("tree2/", "mnt") as mounted:
    print(sorted(p.name for p in mounted.mountpoint.rglob("*")))

report = ziptensors.decompress_tree("tree2/")
print(report.totals.restored, report.has_failures)
```

Each weight file becomes a one-member container beside it; the overlay mount shows the originals again.
See [directories.md](directories.md).

### Mount a container (Linux, Windows)

```python
import os
import ziptensors

os.makedirs("mnt", exist_ok=True)                     # Linux; on Windows pass "X:" instead
with ziptensors.mount("pipeline.ziptensors", "mnt") as mounted:
    print(mounted.mountpoint, mounted.is_mounted)
    for root, dirs, files in os.walk(mounted.mountpoint):
        for name in files:
            print(os.path.join(root, name))
    with open(os.path.join(mounted.mountpoint, "unet/diffusion_pytorch_model.safetensors"), "rb") as f:
        print(len(f.read()))
print(mounted.is_mounted)                              # False: the with block unmounted it
```

Any program that takes a path can read the mount, for example
`UNet2DModel.from_pretrained(mounted.mountpoint)` from Diffusers. On Windows the mount point is a free
drive letter (`"X:"`) or a directory that does not exist yet, and WinFsp must be installed; see
[mount.md](mount.md) for requirements and the WinFsp notice.

## Reference: reading

### `Container`

```python
Container(path, *, threads=None, max_total_decoded=None, verify="chunk",
          max_index_bytes=None, max_decode_amplification=None)
```

Opens a container file. The file is recognized by its magic bytes, not by its name. Opening reads the
header, footer and index; no member data is read until you ask for it.

| Parameter | Default | Meaning |
|---|---|---|
| `path` | required | The container file. |
| `threads` | `None` | Decode threads. `None` or `0`: one per logical CPU. `1`: decode on the calling thread. |
| `max_total_decoded` | `None` | Budget on the sum of member lengths, checked at open before any member data is read: a container over it raises `BudgetError`. `None` means no budget, and `0` is a budget of zero bytes. Set it for any file you do not trust. |
| `verify` | `"chunk"` | `"chunk"` checks the XXH3 hash of every decoded chunk; `"off"` checks exact lengths only. Any other value is `ValueError`. |
| `max_index_bytes` | `None` | Cap on the decompressed index. `None`: the library default, 256 MiB. Values over 1 GiB act as 1 GiB. |
| `max_decode_amplification` | `None` | Bound on decode work as a multiple of the member lengths, on top of a floor of 256 MiB. `None`: the library default, 16. Raise it only for a file whose small members point into large chunks. |

Errors: `ZiptensorsIoError` (cannot open or read the file), `FormatError` (not a container, or a damaged
header, footer or index), `UnsupportedError` (a newer format version, feature or codec), `BudgetError`
(over one of the limits).

| Member | Returns | Meaning |
|---|---|---|
| `members` | `list[MemberInfo]` | Every member, in file order. |
| `member(name=None)` | `Member` | The member called `name`. With `None`, the only member of a single-member container; a container with several members raises `NotFoundError`. An unknown name raises `NotFoundError`. |
| `layer_report()` | `tuple[LayerBytes, list[ClassBytes]]` | Where the stored bytes went, from the index alone: totals, and one entry per class of data (by dtype). |
| `restore_all(dir)` | `RestoreAllReport` | Restores every member under `dir`, creating directories. A member name that is unsafe as a path raises `InputError` before anything is created; a later failure keeps the members already restored. |
| `close()` | `None` | Drops this object's handle on the file. Idempotent. |
| `with Container(...) as c:` | | Leaving the block calls `close()`. |

After `close()` every call on the container raises `ValueError`. A `Member` taken before keeps the file
open until it is dropped, and stays usable.

### `MemberInfo`

What the index says about one member, from `Container.members`. Read-only properties:

| Property | Type | Meaning |
|---|---|---|
| `name` | `str` | Member path as stored, `/`-separated. Untrusted: check it before joining it to a directory. |
| `len` | `int` | Length of the original file. |
| `blake3` | `bytes` | BLAKE3 hash of the whole original file, 32 bytes. |
| `profile` | `str` | Recognized format: `opaque`, `safetensors`, `gguf`, `onnx`, `pt_zip`, or `unknown(n)`. A hint only. |
| `tensor_count` | `int` | Rows of the member's tensor table. |

### `Member`

One member of a container, from `Container.member`. A `Member` keeps the container file open while it
exists. It is safe to use from any number of threads at once, and every read releases the GIL.

Properties: `name`, `len`, `blake3`, `profile` (as in `MemberInfo`) and `tensors` (`list[Tensor]`, the
tensor table in file order).

| Method | Returns | Meaning |
|---|---|---|
| `tensor_info(name)` | `Tensor` | The tensor called `name`, for sizing a buffer. Unknown name: `NotFoundError`. |
| `read(offset, length)` | `bytes` | `length` bytes from byte `offset` of the member. |
| `read_at(offset, buffer)` | `None` | Fills all of `buffer` from byte `offset`. |
| `read_many(requests)` | `None` | Fills every `(offset, buffer)` pair of `requests`, visiting the file in storage order. Every range is checked before any read. |
| `decode_tensor_into(name, buffer)` | `None` | Fills `buffer`, of exactly `Tensor.len` bytes, with the tensor's bytes. |
| `open()` | `io.RawIOBase` | A read-only, seekable binary file over the member, for example for `torch.load`. |
| `restore_file(path)` | `RestoreReport` | Writes the member to `path`, replacing a file there, once its BLAKE3 matched. |
| `verify()` | `None` | Decodes the whole member and compares its BLAKE3 with the stated hash. Mismatch: `CorruptionError`. |

**Buffers.** `read_at`, `read_many` and `decode_tensor_into` take any writable, C-contiguous object of
the buffer protocol (`bytearray`, `memoryview`, a numpy array of any dtype except `object`, ...). Sizes
are in bytes whatever the item type. A read-only or non-contiguous buffer raises `BufferError`;
something that is no buffer at all raises `TypeError`. No other thread may touch the buffer during the
call. After an error the buffer's contents are unspecified. In `read_many`, two buffers whose memory
overlaps (the same object twice, two views of one array) raise `BufferError` before anything is written.

**Ranges.** A range outside the member, or a buffer of the wrong size for `decode_tensor_into`, raises
`RangeError`. Damage that the chunk check detects raises `CorruptionError`; with `verify="off"` a read returns
whatever decodes. Neither mode authenticates the file (see
[safety.md](safety.md#what-the-hashes-protect-and-what-they-do-not)).

**`read` and large reads.** `read` returns a new `bytes` object. Zeroing it runs with the GIL held, so
read large ranges into your own buffer with `read_at` instead.

**`open()`.** Each file object has its own position, can seek past the end (and reads nothing there),
and belongs to one thread at a time, like any file object. `readinto` decodes straight into the caller's
buffer. Damaged data raises the errors of `read_at`, never a short read. `torch.load(member.open(),
mmap=True)` does not work: torch needs a real path for `mmap`.

**`restore_file(path)`.** The output appears only after its hash matched. On an error `path` is
untouched, except after `NotDurableError`, when the complete file is in place but its directory could
not be synced. The report has `bytes` (length written) and `blake3`.

### `Tensor`

One row of a member's tensor table. Metadata only: what a restore produces never depends on it.

| Property | Type | Meaning |
|---|---|---|
| `name` | `str` | Tensor name as in the source file. |
| `offset` | `int` | Byte offset of the tensor's data in the member. |
| `len` | `int` | Length in bytes: the buffer size `decode_tensor_into` needs. |
| `shape` | `list[int]` | Dimensions in the order of the source file. GGUF stores the fastest varying dimension first, the reverse of numpy and torch order. |
| `dtype` | `str` | Registry name such as `F32`, `BF16`, `Q8_0`, or `UNKNOWN(0x....)` for a type the library does not know. |
| `block` | `tuple[int, int] \| None` | `(elements, bytes)` of one block of the dtype: `(1, width)` for a plain type, one ggml block for a block-quantized type (`(32, 34)` for `Q8_0`), `None` if unknown. |

## Reference: writing

The mode decides which transforms the writer tries; it never changes what a restore produces.

| Mode | What it adds |
|---|---|
| `"fast"` | Plane split and entropy coding only. |
| `"balanced"` | Widening views and field split of quantized data. Default of `encode_file`. |
| `"max"` | Rotation trial, deduplication across members, delta blobs. Default of `pack`, `PackBuilder`, `edit`, `compact`, `compress_tree`. |

An unknown mode raises `ValueError`. `threads` is `None` or `0` for one per logical CPU everywhere.
Every writer decodes the new file and compares it with its sources before it replaces anything at the
output path. Writers that go through a temporary file (everything except `edit`/`compact` with
`strategy="append"` or `"in_place"`, and `recover`) leave the output path untouched on an error,
except after `NotDurableError`; the other two strategies are described under [`edit`](#edit).

### `encode_file`

```python
encode_file(input, output, mode="balanced", threads=None) -> EncodeReport
```

Encodes one file as a container of one member, named after the file.

### `pack`

```python
pack(inputs, output, mode="max", threads=None, *, delta_base=None) -> EncodeReport
```

| Parameter | Default | Meaning |
|---|---|---|
| `inputs` | required | A sequence of paths and `(path, member_name)` pairs. A bare file path's member takes the file's name; a pair names it. A bare directory is walked as `collect_inputs` walks it, its files named relative to it. |
| `output` | required | The container to write. An output inside a walked directory raises `InputError`. |
| `mode` | `"max"` | See the table above. |
| `threads` | `None` | Compression and verification threads. |
| `delta_base` | `None` | Member name of one input. It is added first, and tensors of every other input that pair with its tensors are stored as delta blobs against it where that is smaller. Needs `mode="max"` (otherwise `ValueError`); a name no input has is `ValueError`. |

A symbolic link given as a file raises `InputError`, and so do two inputs with one member name (code 26),
before the output is started.

### `collect_inputs`

```python
collect_inputs(roots) -> Inputs
```

Walks `roots` the way `pack` walks a directory and returns what it would add, without writing anything.
`Inputs.files` is a list of `(path, member_name)` per root in the order given, then by member name;
`skipped_symlinks` and `skipped_incomplete` count symbolic links and `*.incomplete` files (partial
downloads) left out. A root that is a file is named after itself; a root that is a link is skipped and
counted.

### `PackBuilder`

```python
PackBuilder(output, mode="max", threads=None)
```

Builds a container member by member. The output appears only at `finish()`, after the whole file decoded
back to its sources; until then it is a temporary file beside `output`.

| Method | Returns | Meaning |
|---|---|---|
| `add_file(path, name=None)` | `MemberReport` | Adds a file, by default under its file name. A symbolic link, or a name already in the pack (code 26), raises `InputError`. |
| `add_file_delta(path, name, base_member)` | `MemberReport` | Adds a file whose tensors that pair with tensors of `base_member` are stored as delta blobs where smaller. `base_member` must be a member this builder added from a file, unchanged on disk (`NotFoundError` or `InputError` otherwise). Outside `mode="max"` this is `add_file`. |
| `add_member_from(container, name)` | `MemberReport` | Copies member `name` of another open `Container` under the same name, with no recompression. |
| `finish()` | `EncodeReport` | Writes the index, proves every member against its source and renames the file to `output`. The builder is spent whether it succeeds or not. |
| `close()` | `None` | Drops an unfinished pack and its temporary file. Idempotent. |

As a context manager, leaving the `with` block without `finish()` removes the temporary file, whether or
not the block raised; so does dropping the object. Any call after `finish()` or `close()` raises
`ValueError`. After a failed add, every later call fails, `finish()` included. Calls may come from
several threads; they run one at a time with the GIL released.

```python
import ziptensors

with ziptensors.PackBuilder("out.ziptensors") as builder:
    builder.add_file("model/unet/diffusion_pytorch_model.safetensors", "unet.safetensors")
    report = builder.finish()
```

## Reference: editing and recovery

### Operations

| Class | Constructor | Meaning |
|---|---|---|
| `AddOp` | `AddOp(path, name=None, *, delta_base=None)` | Adds the file at `path` as a new member at the end, named `name` (default: the file's name). |
| `RemoveOp` | `RemoveOp(name)` | Removes member `name`. |
| `ReplaceOp` | `ReplaceOp(name, path, *, delta_base=None)` | Replaces the content of member `name` with the file at `path`, keeping its position. |

`delta_base` names a member of the result to store the file against as delta blobs where smaller: a kept
member, or a file added earlier in the same edit. The constructor arguments are readable back as
properties (`path` is a `pathlib.Path`).

### `edit`

```python
edit(container, ops, *, strategy="rewrite", repack=False, mode="max",
     threads=None, max_total_decoded=None) -> EditReport
```

| Parameter | Default | Meaning |
|---|---|---|
| `container` | required | The container file to change. |
| `ops` | required | A sequence of `AddOp`, `RemoveOp`, `ReplaceOp`. Each member name may appear in one operation only. Anything else in the sequence is `TypeError`. |
| `strategy` | `"rewrite"` | `"rewrite"` writes a proven copy beside the file and renames it over. `"append"` writes only the new data and a new index after the end, leaving the old data as dead bytes; it takes no `repack` and no empty `ops`. `"in_place"` rewrites the file within itself with no extra space and is **risky**: an interruption after its first write leaves a damaged file (`CorruptionError`), so keep a backup. Other values: `ValueError`. |
| `repack` | `False` | With `"rewrite"` only: re-encode every kept member with `mode`. |
| `mode` | `"max"` | Mode for new data. |
| `threads` | `None` | Threads for encoding and for decoding the source. |
| `max_total_decoded` | `None` | Budget for opening the source container, as for `Container`. |

Kept members keep their order, a replacement keeps its slot, and added members follow in the order
given. Every check runs before a byte is written: a refused request (`InputError`, `NotFoundError`)
leaves the file untouched. Close every `Container` open on the file before an edit, except under
`"append"`. A file with bytes after a valid container (a torn append) does not open and raises
`FormatError`; run `recover` then. `ZiptensorsIoError` with code 23 means an `"append"` could not cut
its own tail back, or an `"in_place"` edit could not make its final cut: run `recover` as well.

### `compact`

```python
compact(container, *, strategy="rewrite", repack=False, mode="max",
        threads=None, max_total_decoded=None) -> EditReport
```

An edit with no operation: drops the dead bytes earlier appends left behind. `strategy` is `"rewrite"`
or the risky `"in_place"`; `"append"` raises `InputError`. `repack=True` re-encodes every member.

### `recover`

```python
recover(path, *, dry_run=False, verify=True, max_candidates=64,
        max_scan_bytes=None, threads=None) -> RecoverReport
```

Shortens a container to its latest valid generation after an interrupted append. Readers refuse such
a file until this runs; it never runs on its own. A file that opens as it is stays untouched. After an
interrupted `"in_place"` edit it usually cannot help: the older generations no longer verify, and it
fails rather than return wrong data (see [editing.md](editing.md#recover)). Keep a backup.

| Parameter | Default | Meaning |
|---|---|---|
| `dry_run` | `False` | Report what would happen and write nothing. |
| `verify` | `True` | Decode every member of a candidate generation and pass over one that fails. |
| `max_candidates` | `64` | Most footers the backward scan checks. |
| `max_scan_bytes` | `None` | Most distinct bytes the scan reads; `None` means no limit. |
| `threads` | `None` | Verification threads. |

Raises `FormatError` (code 24) when no candidate is structurally valid within the bounds; when
candidates were found but none verified, the newest one's verification error (usually
`CorruptionError`) instead. I/O, `UnsupportedError` and `BudgetError` stop the scan as they occur. The
file is unchanged after any of these. `recover` takes no `max_total_decoded`: a budget
tighter than the writer's would stop the scan at a genuine generation.

## Reference: directory trees

Both functions work on a directory (or one file) and report one `FileOutcome` per path they looked at.
They go on past a failure of one file, so check `TreeReport.has_failures`. Name collisions raise
`InputError` before anything is written. Details of what counts as a weight file and how pairs are
resolved are in [directories.md](directories.md).

```python
compress_tree(path, *, keep=False, follow_symlinks=False, dry_run=False, mode="max",
              threads=None, max_total_decoded=None) -> TreeReport
decompress_tree(path, *, keep=False, force=False, dry_run=False,
                threads=None, max_total_decoded=None) -> TreeReport
```

| Parameter | Default | Meaning |
|---|---|---|
| `keep` | `False` | Keep the original (compress) or the container (decompress) after the other copy is proven. |
| `follow_symlinks` | `False` | Compress only: compress the targets of symbolic links instead of skipping them. Following links breaks a Hugging Face cache, whose files are links into a blob store. |
| `force` | `False` | Decompress only: replace a different file already at the restore path instead of failing that container. |
| `dry_run` | `False` | Plan and write nothing: every file that would be compressed, restored or checked as a pair is `planned`; files classified without action (`not_weight`, `foreign`, `skipped`, `already_compressed`, `failed`) keep that status. |
| `mode` | `"max"` | Compress only. |
| `threads` | `None` | Threads per file. |
| `max_total_decoded` | `None` | Budget for each container opened. Restoring untrusted files needs it. |

`compress_tree` turns `dir/model.safetensors` into `dir/model.ziptensors`, recognizing weight files by
content. `decompress_tree` restores every per-file container to the file its member names; packs and
containers whose member is no single safe name are reported as `foreign` and left alone.

**`TreeReport`**: `root` (`Path`), `dry_run`, `files` (`list[FileOutcome]`, sorted by path),
`warnings` (`list[str]`, notes about the whole run), `totals` (`TreeTotals`), `has_failures`.

**`FileOutcome`**:

| Property | Meaning |
|---|---|
| `path` | The path looked at: in compress the original (or a container found alone), in decompress the container. |
| `counterpart` | The other copy: in compress the container, in decompress the restored file; or `None`. |
| `status` | `compressed`, `recompressed`, `pair_resolved`, `pair_verified`, `already_compressed`, `restored`, `not_weight`, `foreign`, `skipped`, `planned` or `failed`. |
| `reason` | For `foreign`: `member_count`, `not_one_safe_name`. For `skipped`: `symlink`, `symlink_to_directory`, `dangling_symlink`, `special`, `stale_temp`, `incomplete`, `non_utf8`. For `planned`: `compress`, `check_pair`, `restore`. For `failed`: the error message. Otherwise `None`. |
| `error` | For `failed`: the exception object (a `ZiptensorsError` with `code`), not raised. Otherwise `None`. |
| `original_len`, `container_len` | Byte lengths where they were looked at, or `None`. |
| `removed` | The copy removed because its twin was proven, if any; set even when a later step failed. |

**`TreeTotals`**: one count per status (`compressed`, `recompressed`, `pair_resolved`, `pair_verified`,
`already_compressed`, `restored`, `not_weight`, `foreign`, `skipped`, `planned`, `failed`; a count
of the other direction is 0: `restored` in `compress_tree`, and in `decompress_tree`, which looks at
containers alone, `compressed`, `recompressed`, `already_compressed` and `not_weight`), and
`original_bytes` and `container_bytes`, summed over the files whose two copies the run proved equal
(compressed, recompressed, restored, or a pair): in `compress_tree` the bytes that went into
containers, in `decompress_tree` the bytes that came out of them and the containers they were read
from.

## Reference: mounting

Mounts need Linux (FUSE) or Windows (WinFsp). [mount.md](mount.md) covers requirements, what the mounted
tree looks like, caching and unmounting in detail; this section lists the Python surface.

Windows mounts use WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos,
<https://github.com/winfsp/winfsp>, licensed under GPL-3.0 with a FLOSS exception. Install it from
<https://winfsp.dev/rel/>. ziptensors distributes no WinFsp file.

```python
mount(container, mountpoint, *, threads=None, cache_size=None, readahead=0,
      max_total_decoded=None) -> Mount
mount_overlay(directory, mountpoint, *, threads=None, cache_size=None, readahead=0,
              max_total_decoded=None, max_open_containers=4, max_open_files=64) -> Mount
```

`mount` shows every member of `container` as a read-only file under `mountpoint`. `mount_overlay` shows
`directory` read-only, with each per-file container (made by `compress_tree`) appearing as its original
file under its original name, and every other file as it is.

| Parameter | Default | Meaning |
|---|---|---|
| `mountpoint` | required | Linux: an existing directory. Windows: a free drive letter (`"X:"`) or a directory that does not exist yet (its parent must), created for the mount and removed after. |
| `threads` | `None` | Decode threads; `None` or `0` for one per logical CPU. Overlay: per open container. |
| `cache_size` | `None` | Decoded bytes kept in memory for reads smaller than a chunk. `None`: 256 MiB. `mount`: at least twice the container's largest chunk, otherwise `BudgetError`. Overlay: shared by the open containers, each getting `cache_size / max_open_containers`, which must be at least 16 MiB, otherwise `ValueError`. |
| `readahead` | `0` | Cache windows decoded per cache miss, the missed one included; `0` means the decode thread count, `1` no read-ahead. |
| `max_total_decoded` | `None` | Budget of each container opened, as for `Container`. Set it for files you do not trust. |
| `max_open_containers` | `4` | Overlay only: containers kept open at once, at least 1 (`ValueError` otherwise); the least recently used is closed first. |
| `max_open_files` | `64` | Overlay only: plain files kept open at once, at least 1. |

Setup runs with the GIL released. Errors: `MountError` for every mount failure (see below); a container
that does not open raises the usual exception (`FormatError`, `BudgetError`, ...); an overlay option out
of range raises `ValueError`.

### `Mount`

The handle of a live mount, returned by `mount` and `mount_overlay`. Use it as a context manager.

| Member | Meaning |
|---|---|
| `mountpoint` | `Path` where the files are, as the backend resolved it (`X:\` for a Windows drive). |
| `is_mounted` | `True` while mounted and while an unmount by this object waits for open files; `False` once that unmount ended, even if it failed. An unmount from outside (`fusermount3 -u`) is not seen here. |
| `unmount()` | Unmounts and waits, with the GIL released, until every file open under the mount is closed (memory maps included). Idempotent. On failure raises `MountError` and gives the mount up; on Linux it then stays until `fusermount3 -u -z <mountpoint>` removes it. |
| `with mount(...) as m:` | Leaving the block calls `unmount()`, whether or not the block raised, `KeyboardInterrupt` included. A failed unmount raises `MountError`, with the block's exception as its context. |

An unmount detaches the mount point at once, so new opens fail, then waits for files still open under
it. Close them first: a file held open by the thread that unmounts makes it wait forever. A `Mount`
dropped while still mounted is unmounted on a background thread, with a warning in the log; rely on
`with` or `unmount()` instead. There is no method that blocks until an outside unmount.

## Reference: logging and version

```python
enable_stderr_logging(level="info") -> None
```

Writes the library's log events at `level` and above to stderr, for the rest of the process. `level` is
`"off"`, `"error"`, `"warn"` (or `"warning"`), `"info"`, `"debug"` or `"trace"`, in any case; anything
else is `ValueError`. Logging is off until this is called. Only the first call in a process takes
effect; later calls succeed and change nothing. Python's `logging` module does not see these events.
The pure-Python adapters log through Python's `logging` (loggers `ziptensors.torch` and
`ziptensors.safetensors`).

`ziptensors.__version__` is the version of the package (`str`).

## Reference: report classes

All report objects are read-only. Byte counts are `int`.

| Class | Properties |
|---|---|
| `EncodeReport` | `output_len` (file length), `index_stored_len`, `members` (`list[MemberReport]`, file order), `total` (`LayerBytes`), `classes` (`list[ClassBytes]`), `verify` (`VerifyReport`). |
| `MemberReport` | `name`, `len`, `blake3`, `profile`, `tensor_count`, `extent_count`, `blobs_added`, `blobs_reused` (extents that went to an existing blob), `widen_restreams`, `delta_blobs`, `compressed_bytes` (0 for a copied member), `copied_bytes` (stored bytes copied with no recompression), `layers` (`LayerBytes`). |
| `LayerBytes` | `raw` (source bytes), `widen_saved` (not stored because a widening view reproduces them), `dedup_saved` (not stored because an identical blob exists), `entropy_saved` (saved by compression), `overhead` (stored-form bytes added over the canonical bytes), `stored` (stored chunk bytes of new blobs). |
| `ClassBytes` | `dtype` (registry name, or `None` for file structure and opaque files), `layers` (`LayerBytes`). |
| `VerifyReport` | `bytes` decoded and hashed before commit, `seconds` of wall time. |
| `RestoreReport` | `bytes` written, `blake3` of the result. |
| `RestoreAllReport` | `members` restored (all of them), `bytes` written. |
| `EditReport` | `kept`, `added`, `removed`, `replaced` (member counts), `generation` (footer generation of the result: the source's plus one, staying at `2**64 - 1` once there), `encode` (the writer's `EncodeReport`; under `"append"` its `output_len` is the whole file), `dead_bytes_before` (bytes of the source no member needed). |
| `RecoverReport` | `file_len`, `valid_len` (length of the valid container; `file_len` when it opened as it was), `generation`, `truncated` (whether the file was cut; `False` on a dry run), `verified_bytes` (`None` when nothing was verified), `scanned_bytes` (distinct bytes the backward scan read, at most `file_len - 64`), `passed_over` (newer generations that failed the checks). |
| `Inputs` | See `collect_inputs`. |
| `TreeReport`, `FileOutcome`, `TreeTotals` | See [directory trees](#reference-directory-trees). |

## Errors

Every exception the module raises for a library failure is a `ZiptensorsError` with an integer `code`
attribute, the stable error code shared with the command line tool and the C API. Bad arguments that
never reach the library raise ordinary `ValueError`, `TypeError`, `OverflowError` or `BufferError`.

```text
ZiptensorsError
├── ZiptensorsIoError
│   └── NotDurableError
├── FormatError
├── UnsupportedError
├── BudgetError
├── CorruptionError
├── RangeError
├── NotFoundError
├── InputError
└── MountError
```

| Exception | `code` values | Meaning | What to do |
|---|---|---|---|
| `ZiptensorsIoError` | 1 `Io`, 23 `TrailingBytes` | A file operation failed; or a valid container is followed by bytes that could not be removed (a torn append). | Check the path, permissions and free space. For `TrailingBytes`, run `recover`. |
| `NotDurableError` | 20 `CommittedNotDurable` | The output **was** written and is complete, but its directory could not be synced, so a power loss may bring back the previous file. | The result is usable; sync or check the file system. |
| `FormatError` | 2 `NotZiptensors`, 6 `MalformedIndex`, 24 `Unrecoverable` | Not a container, a header, footer or index that violates the format, or `recover` found no valid generation. | Check that the file is a ziptensors container and was copied completely. |
| `UnsupportedError` | 3 `UnsupportedVersion`, 4 `UnsupportedFeature`, 5 `UnknownCodec` | The file needs a format version, feature or codec this build does not implement. | Upgrade the package. |
| `BudgetError` | 7 `BudgetExceeded` | A size declared by the file exceeds a limit (`max_total_decoded`, `max_index_bytes`, decode amplification, a mount cache too small for the chunks). Raised before the data is read. | Raise the limit if you trust the file. |
| `CorruptionError` | 8 `ChunkCorrupt`, 9 `ChecksumMismatch`, 15 `VerifyFailed`, 17 `Zstd`, 21 `DamagedInPlace` | Stored data does not decode or does not match its hash; or an in-place edit failed after it started moving data and damaged the file. | The file is damaged: use another copy or your backup. |
| `RangeError` | 10 `OutOfBounds`, 11 `DestinationSize` | A read range lies outside the member, or a buffer has the wrong size. | Fix the offset or buffer size. |
| `NotFoundError` | 12 `TensorNotFound`, 13 `MemberNotFound` | No such member or tensor. | Check the name against `members` or `tensors`. |
| `InputError` | 14 `UnsafeMemberName`, 16 `InputChanged`, 18 `SymlinkInput`, 19 `OutputIsInput`, 22 `EditRefused`, 25 `TreeConflict`, 26 `DuplicateMember` | An input was refused (unsafe name, symbolic link, a file that changed while being read, output inside the inputs, two inputs with one member name), an edit cannot be applied as asked, or names collide in a tree run. Nothing was written. | Read the message; fix the input or request. |
| `MountError` | the code of the cause | A mount cannot be made or ended (details below). | See [mount.md](mount.md). |

`MountError.code` is:

- `4` (`UnsupportedFeature`) on macOS, which has no mount backend, and for an overlay entry of a kind this
  version cannot show;
- `14` (`UnsafeMemberName`) when member names cannot be shown by the operating system; the message names
  every one;
- `25` (`TreeConflict`) when two overlay entries claim the same name;
- the code of the underlying error for a damaged overlay entry or a member the library refuses (that
  exception is the `__cause__`);
- `1` (`Io`) when the backend refused (no `/dev/fuse` or `fusermount3`, a missing mount point, no
  WinFsp) or an unmount left the mount in place.

The pure-Python adapters raise their own `ValueError` subclasses, `TorchLoadError` and
`SafetensorError`, described in their sections below.

## Threads and the GIL

- `Container`, `Member` and `Mount` are safe to share between threads. Opening, reading, decoding,
  restoring, encoding, packing, editing, recovering, tree runs, mounting and unmounting all run with the
  GIL released, so threads that read in parallel scale.
- The library runs its own decode and compression threads, sized by each call's `threads` argument.
- `read` zeroes its result with the GIL held; use `read_at` into your own buffer for large ranges.
- A buffer passed to a read belongs to the call until it returns: no other thread may touch it.
- A file object from `Member.open()` belongs to one thread at a time.
- `PackBuilder` may be called from several threads; calls run one at a time.
- The mount's own threads never call into Python, and the module masks no signal: Ctrl-C reaches Python
  as `KeyboardInterrupt`, and a `with` block then unmounts.

## `ziptensors.torch`

```python
from ziptensors.torch import GgmlTensor, TorchLoadError, load_into, load_state_dict, STAGING_BYTES
```

Loads the tensors of a safetensors or GGUF member into PyTorch with no intermediate file and no copy.
torch is imported on the first call; `import ziptensors.torch` works without it.

### `load_state_dict`

```python
load_state_dict(member, device="cpu", dtype=None, keys=None, *,
                quantized="refuse", staging_bytes=None) -> dict[str, torch.Tensor | GgmlTensor]
```

Allocates each tensor uninitialized with torch and decodes the member's bytes into its storage with the
GIL released. Names and bytes are those of the file. Every tensor asked for is checked before the first
allocation, and nothing is returned after an error.

| Parameter | Default | Meaning |
|---|---|---|
| `member` | required | A `ziptensors.Member` with profile `safetensors` or `gguf`. |
| `device` | `"cpu"` | Where the tensors end up. `meta` is refused (`ValueError`). On a `cuda` device (CUDA, or ROCm through the same API) bytes are decoded into two pinned host buffers and copied asynchronously while the next piece decodes; the call returns after every copy is done. On `cpu` each tensor is decoded straight into the tensor returned. On any other device each tensor is decoded on the host, moved, and its host copy freed before the next one. |
| `dtype` | `None` | A floating point torch dtype to convert floating point tensors to (on the device for `cuda`, on the host otherwise). Integer, bool and quantized tensors keep their dtype. A non-floating dtype is `ValueError`. |
| `keys` | `None` | Tensor names to load, in that order; `None` loads all, in table order. A single string is `TypeError`; an unknown name `NotFoundError`. |
| `quantized` | `"refuse"` | What to do with block-quantized GGUF tensors: `"refuse"` raises `TorchLoadError`; `"raw"` returns each as a `GgmlTensor`; `"diffusers"` returns each as the `GGUFParameter` that Diffusers' `load_gguf_checkpoint` makes of the same bytes (needs `diffusers` and `gguf`). Any mode but `"refuse"` on a non-GGUF member is `ValueError`. |
| `staging_bytes` | `None` | Size of each of the two pinned buffers for a `cuda` load. `None`: `STAGING_BYTES` (64 MiB), capped at the bytes being loaded. Below 1 is `ValueError`, a non-integer `TypeError`. |

**Accepted dtypes.** Safetensors `F64 F32 F16 BF16 F8_E4M3 F8_E5M2 I64 I32 I16 I8 U64 U32 U16 U8 BOOL`
(`F8_E4M3` becomes `torch.float8_e4m3fn`, `F8_E5M2` `torch.float8_e5m2`), and GGUF `F64 F32 F16 BF16`.
Anything else raises `TorchLoadError` naming the tensor: other dtypes, GGUF integer tensors, any other
member profile, a dtype the installed torch lacks, or a `BOOL` byte other than 0 or 1. Bytes are never
reinterpreted.

**GGUF.** Shapes are reversed into torch order (GGUF lists the fastest varying dimension first); the
bytes are not moved. Names stay GGUF names, so a llama.cpp or ComfyUI file still needs that project's
name mapping before a module accepts it. A whole-member load refuses a file whose tensor table is
incomplete rather than return a partial state dict.

**Memory.** On `cpu` the whole state dict asked for is in host memory: that is the result. On `cuda`,
host memory peaks at the two staging buffers, which torch's caching host allocator keeps pinned for
reuse after the call. On other devices host memory peaks at the largest tensor. Every path adds the
decoder's working memory ([rust.md](rust.md#threads-and-memory)), and a `dtype` conversion briefly
holds both versions of one tensor.

**Legacy PyTorch files.** A `pt_zip` member (`.bin`, `.pt`, `.pth`) raises `TorchLoadError`: its dtypes
and shapes live in a pickle, not in a tensor table. Load it with
`torch.load(member.open(), weights_only=True)`. `weights_only=True` restricts unpickling to tensors and
plain containers, and refuses any other global; never turn it off for a file you did not create.

Raises `ImportError` without torch (or, for `"diffusers"`, `TorchLoadError` without diffusers and gguf),
and any error of `Member.read_at`.

### `GgmlTensor`

The raw blocks of one quantized GGUF tensor, from `quantized="raw"`. A frozen dataclass:

| Field | Meaning |
|---|---|
| `data` | Contiguous `torch.uint8` tensor in gguf-py's byte shape: the logical shape with its last dimension replaced by `last // block_elements * block_bytes`. |
| `ggml_type` | Registry name, also the name in `gguf.GGMLQuantizationType`, such as `Q8_0`. |
| `shape` | Logical element shape in torch order: the shape after dequantization. |

For Diffusers, `quantized="diffusers"` builds the model directly:
`Model.from_single_file(load_state_dict(member, quantized="diffusers"),
quantization_config=GGUFQuantizationConfig(compute_dtype=...), config=<dir>, local_files_only=True)`.
A quantization type Diffusers cannot dequantize raises `TorchLoadError`.

### `load_into`

```python
load_into(module, member, strict=True, assign=True)
```

`load_state_dict(member)` followed by `module.load_state_dict(state, strict=strict, assign=assign)`;
returns torch's report of missing and unexpected keys. Build the module on the `meta` device, so its
parameters take no memory before they are filled:

```python
import torch
from safetensors.torch import save_file
import ziptensors
from ziptensors.torch import load_into

model = torch.nn.Linear(16, 4)
save_file(model.state_dict(), "linear.safetensors")
ziptensors.encode_file("linear.safetensors", "linear.ziptensors")
member = ziptensors.Container("linear.ziptensors").member()

with torch.device("meta"):
    empty = torch.nn.Linear(16, 4)
load_into(empty, member)
print(torch.equal(empty.weight, model.weight))         # True
```

With `assign=True` the module takes the decoded tensors themselves (on the host, with their stored
dtypes), so there is no second copy. `assign=False` copies into existing storage, which a meta module
does not have. Quantized GGUF tensors are refused.

A file holds neither non-persistent buffers nor the second name of a tied parameter, so a module can stay
partly on the meta device. With `strict=True` that raises `TorchLoadError` naming the tensors; with
`strict=False` it logs a warning and the caller finishes the module (`model.tie_weights()`, or
registering the buffers again). A key or shape mismatch under `strict=True` raises torch's
`RuntimeError`; after either error the module may be partly filled.

### `TorchLoadError`

A subclass of `ValueError` (not of `ZiptensorsError`): a member or tensor cannot become torch tensors
without changing its meaning. The message names the tensor and the reason.

### Use with Transformers and Diffusers

- Transformers: `CLIPTextModel.from_pretrained(None, config=config, state_dict=load_state_dict(member))`
  also ties weights and fills buffers.
- Diffusers: `from_config` on the meta device plus `load_into`, or
  `UNet2DConditionModel.from_single_file(load_state_dict(member), config=<config dir>,
  local_files_only=True)`. A pipeline takes the loaded components in its constructor.
- Or mount the container and call the usual `from_pretrained(path)` on the mounted directory
  ([mount.md](mount.md)).
- `examples/diffusers_from_container.py` in the repository builds a whole Stable Diffusion pipeline from
  a packed container or a `compress_tree` directory, without a mount.

## `ziptensors.safetensors`

```python
from ziptensors.safetensors import SafeSlice, SafetensorError, safe_open
```

`safe_open` reads a safetensors member the way `safetensors.safe_open` reads a file.

```python
safe_open(filename, framework, device="cpu", *, member=None, threads=None, max_total_decoded=None)
```

| Parameter | Default | Meaning |
|---|---|---|
| `filename` | required | A container path, or a `ziptensors.Member`. |
| `framework` | required | `"pt"` (also `"torch"`, `"pytorch"`) or `"np"` (also `"numpy"`). Other values: `SafetensorError`. |
| `device` | `"cpu"` | `"cpu"`, an int N for `cuda:N`, or a torch device string or object. numpy takes `cpu` only. On `cuda`, whole tensors use the staging pipeline of `load_state_dict`. |
| `member` | `None` | With a path: the member to read; `None` for a single-member container. |
| `threads`, `max_total_decoded` | `None` | With a path: passed to `Container`. Passing these or `member` with a `Member` object is `TypeError`. |

The safetensors header is parsed from the member's bytes and fully validated when the object is made,
as the safetensors package validates it, so `__metadata__` is available and an invalid header fails
early.

| Method | Returns |
|---|---|
| `keys()` | Tensor names, sorted. |
| `offset_keys()` | Tensor names in the order of their data. |
| `metadata()` | A copy of `__metadata__`, or `None`. |
| `get_tensor(name)` | The whole tensor on the device. |
| `get_tensors()` | Every tensor, as a dict in offset order. |
| `get_slice(name)` | A `SafeSlice`, which reads nothing until indexed. |

`SafeSlice` has `get_shape()`, `get_dtype()` (the safetensors tag, such as `F32`) and indexing with
integers, positive-step slices, `...` and `None`. Indexing reads only the contiguous run of rows the
leading indices select, on the host, then moves the result to the device.

Usable with or without `with`. After `__exit__` every method, including on a `SafeSlice` taken before,
raises `SafetensorError("File is closed")`.

Dtypes are those of `load_state_dict` for safetensors; numpy has no `BF16` or `F8`. A tensor of another
valid safetensors dtype (`C64`, `F8_E8M0`, `F4`, ...) is listed with its shape and tag, and reading it
raises `TorchLoadError` (pt) or `SafetensorError` (np).

Errors: `SafetensorError` (a `ValueError`, not the class of the safetensors package) for an invalid
header, an unknown framework or device, or an absent tensor; the errors of `Container` and `Member` for
the file and its data; `ImportError` without the framework's package.

Differences from safetensors 0.7: a list, bool or float index raises `TypeError`; numpy slices honour a
step; a `torch.device` object is accepted; a `SafeSlice` cannot be read after its `safe_open` closed.
