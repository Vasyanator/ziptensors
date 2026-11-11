# Command line reference

`ziptensors` is the command line tool. It compresses tensor files into containers, reads them back,
changes containers in place, works through whole directory trees, and mounts containers as read-only
files. Install it with `cargo install ziptensors-cli`.

This page lists every subcommand and flag. Task guides: [editing.md](editing.md) for changing a
container, [directories.md](directories.md) for compressing model directories file by file,
[mount.md](mount.md) for the filesystem mount, [safety.md](safety.md) for untrusted files.

## Common tasks

```sh
# One file -> a container of one member (mode balanced by default)
ziptensors encode model.safetensors model.ziptensors

# Files and directories -> one container (mode max, deduplication across members)
ziptensors pack models.ziptensors sdxl-inpaint/ config.json

# A finetune stored as delta blobs against its base
ziptensors pack pair.ziptensors base.safetensors finetune.safetensors --delta-base base.safetensors

# What is inside
ziptensors inspect models.ziptensors
ziptensors inspect models.ziptensors --tensors
ziptensors inspect models.ziptensors --json

# Check every member against its BLAKE3
ziptensors verify models.ziptensors

# Get files back
ziptensors restore model.ziptensors model.safetensors        # the only member
ziptensors restore models.ziptensors --all restored/         # every member
ziptensors extract models.ziptensors unet/diffusion_pytorch_model.safetensors unet.safetensors
ziptensors extract models.ziptensors config.json - | head    # to standard output

# A file you did not make: set a decode budget
ziptensors verify downloaded.ziptensors --max-total-decoded 64000000000
```

Members of a directory given to `pack` are named relative to that directory: packing `sdxl-inpaint/`
gives members such as `unet/diffusion_pytorch_model.safetensors`, without the `sdxl-inpaint/` prefix.
`ziptensors inspect` lists the exact names.

## Conventions shared by all commands

| Topic | Rule |
|---|---|
| Output | Results go to standard output. Logs, progress, warnings and errors go to standard error. |
| Logging | Log lines at level `INFO` and above always go to standard error; there is no verbosity flag and no environment variable for it. Redirect standard error (`2>/dev/null`, `2>log.txt`) to silence or keep them. Colors are used only when standard error is a terminal. |
| Threads | `--threads N` sets compression, decode or verification threads. `0`, the default, means one per logical CPU. |
| Modes | `--mode fast`, `balanced` or `max`. Default: `balanced` for `encode`, `max` for `pack`, `edit`, `compact -C` and `compress`. A mode changes only the size, never what a restore produces. See the README section on modes. |
| Budget | `--max-total-decoded BYTES` exists on every command that opens an existing container. The container is refused at open, before any member data is read, if its members add up to more than `BYTES`. No limit by default: set one for a file you did not make. Unlike the C API, `0` is taken literally (a limit of zero bytes). |
| Writes | `encode`, `pack`, `merge`, `restore`, `extract` to a file, `compress`, `decompress`, and `edit`/`compact` without `-F`/`-R` write a temporary file `<output>.<pid>-<n>.ziptensors-tmp` in the output directory, decode and check it in full, and only then rename it over the output. An existing output is replaced only once the new file is proven, except after `CommittedNotDurable` (code 20). `edit -F` and `edit -R`/`compact -R` write into the container itself, `recover` cuts it, and `extract ... -` streams unverified bytes: see [editing.md](editing.md#the-three-strategies) and [extract](#extract). The output directory must exist (`restore --all` creates its directory). |
| Restored files | Content only: modification times and permissions are not restored. |
| Inputs | Symbolic links are never followed by `pack`, `edit` or `encode`, and `*.incomplete` files (partial downloads) are skipped by `pack` with a log line. An output that is also an input is refused. |

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | Success. |
| 1 | The command started and failed. Standard error has an `error:` line, usually a `hint:` line, and an `ERROR` log line with the numeric error code (`code_value=`). |
| 2 | Usage error: unknown flag, missing argument, or a combination of flags that is not allowed. Nothing was done. |
| 128 + N | `mount` only: ended by a second signal while files were still open. 130 for a second Ctrl-C; 149 for Ctrl-Break on Windows. |

A failed command prints, for example:

```text
error: cannot extract from models.ziptensors: member not found: nope
hint: `ziptensors inspect` lists the members of the container
```

### Error codes

The `ERROR` log line names the error class and its stable number. The same numbers are the `code`
attribute of Python exceptions and the status values of the C API.

| Code | Name | What it means | What to do |
|---:|---|---|---|
| 1 | `Io` | A file operation failed. | Check the path, its permissions and the free disk space. |
| 2 | `NotZiptensors` | The file does not start with the ziptensors magic. | It is not a container. |
| 3 | `UnsupportedVersion` | The file has a format major version this build does not read. | Use a newer ziptensors. |
| 4 | `UnsupportedFeature` | The file requires a feature this build does not implement. | Use a newer ziptensors. |
| 5 | `UnknownCodec` | A transform, view or codec id is unknown. | Use a newer ziptensors. |
| 6 | `MalformedIndex` | Header, footer or index violates the format. | The file is damaged; after an interrupted `edit -F`, see `recover`. |
| 7 | `BudgetExceeded` | The file declares more than a limit allows (`--max-total-decoded`, index caps). | Raise the limit only if you trust the file. |
| 8 | `ChunkCorrupt` | A chunk does not decode. | The file is damaged. |
| 9 | `ChecksumMismatch` | A chunk decodes to bytes whose hash differs from the index. | The file is damaged. |
| 10 | `OutOfBounds` | A read lies outside a member or the file. | The file is damaged, or a caller asked for a wrong range. |
| 11 | `DestinationSize` | A buffer has the wrong size (library calls). | Pass a buffer of the tensor's length. |
| 12 | `TensorNotFound` | The member has no tensor of that name. | `inspect --tensors` lists them. |
| 13 | `MemberNotFound` | The container has no member of that name, or `restore` without `--all` on a container of several members. | `inspect` lists the members; use `extract` or `restore --all`. |
| 14 | `UnsafeMemberName` | A member name could escape the output directory or cannot be restored. Nothing was written for it. | Extract other members by name; do not trust the file. |
| 15 | `VerifyFailed` | A restored member does not hash to its stated BLAKE3. | The file is damaged. |
| 16 | `InputChanged` | An input changed while it was read. Nothing was committed. | Run again when the input is stable. |
| 17 | `Zstd` | zstd failed outside chunk decoding. | The file is damaged. |
| 18 | `SymlinkInput` | An input is a symbolic link; links are not followed. | Pass the file the link points to. |
| 19 | `OutputIsInput` | The output path is one of the inputs. | Choose another output path. |
| 20 | `CommittedNotDurable` | The output WAS written and is complete, but its directory could not be synced. | Run `sync`; a power loss before that may bring back the old file. |
| 21 | `DamagedInPlace` | An `-R` rewrite failed after it began moving data. The container is damaged. | Restore it from a backup or pack it again from the original files. |
| 22 | `EditRefused` | An edit request cannot be applied as asked. Nothing was written. | Change the request, often the strategy. |
| 23 | `TrailingBytes` | The file holds a valid container followed by bytes that could not be cut off. | Close every program that has the file open, then run `ziptensors recover`. |
| 24 | `Unrecoverable` | `recover` found no valid generation. Nothing was changed. | Restore the file from a backup. |
| 25 | `TreeConflict` | `compress`/`decompress` found names that collide. Nothing was written under that path; paths named before it are already done. | Rename or remove the files named. |
| 26 | `DuplicateMember` | Two inputs of `pack` or `merge` would be stored under one member name. Checked before the output is started, so nothing was written. | Rename one of the files, or leave one input out. |

Codes 20 to 25 are explained with examples in [editing.md](editing.md#errors-of-editing-and-recovery)
and [directories.md](directories.md#collisions).

## Subcommands

| Command | What it does |
|---|---|
| [`encode`](#encode) | One file into a container of one member. |
| [`pack`](#pack) | Files and directory trees into one container. |
| [`edit`](#edit) | Add, remove or replace members of a container. |
| [`compact`](#compact) | Drop dead data of earlier edits, or re-encode every member. |
| [`recover`](#recover) | Shorten a container with a torn end to its latest valid generation. |
| [`restore`](#restore) | Restore the only member, or with `--all` every member. |
| [`extract`](#extract) | Restore one named member, to a file or to standard output. |
| [`merge`](#merge) | Copy the members of several containers into a new one, without recompression. |
| [`inspect`](#inspect) | Print what a container holds, as a table or JSON. |
| [`verify`](#verify) | Decode every member and compare it with its BLAKE3. |
| [`compress`](#compress) | Replace each weight file under a path by a container beside it. |
| [`decompress`](#decompress) | Restore each such container beside it and remove the container. |
| [`mount`](#mount) | Show a container's members, or a compressed tree, as read-only files (Linux and Windows). |

`ziptensors <command> --help` prints the help of one command; `ziptensors --version` the version.

### encode

```text
ziptensors encode [OPTIONS] <INPUT> <OUTPUT>
```

Encodes one file as a container of one member, named after the input file.

| Argument or flag | Default | Meaning |
|---|---|---|
| `INPUT` | required | The file to encode. |
| `OUTPUT` | required | The container to write; an existing file is replaced once the new one is proven. |
| `--mode MODE` | `balanced` | `fast`, `balanced` or `max`. |
| `--threads N` | `0` | Compression and verification threads; 0 is one per logical CPU. |

Prints the output path, member count, delta blobs, raw and file bytes, bytes saved, verification time,
and a table of saved bytes per layer (widening, deduplication, entropy coding) and per dtype.

### pack

```text
ziptensors pack [OPTIONS] <OUTPUT> <INPUTS>...
```

Packs files and directory trees into one container. A file given directly becomes a member named after
the file; files under a directory are named relative to it, with `/` as separator. Links are skipped,
and so are `*.incomplete` files; the last line counts them. An output that the walk would pick up is
refused, and so is a pack with no file to add. Everything is checked before the output is started.

| Argument or flag | Default | Meaning |
|---|---|---|
| `OUTPUT` | required | The container to write; an existing file is replaced once the new one is proven. |
| `INPUTS...` | required | Files and directories. |
| `--delta-base FILE` | none | A file among the inputs (given directly or under an input directory) that is added first; tensors of every other file that pair with its tensors are stored as delta blobs where that is smaller. Needs `--mode max`. |
| `--mode MODE` | `max` | `fast`, `balanced` or `max`. `--delta-base` with a mode other than `max` is a usage error (exit 2). |
| `--threads N` | `0` | Compression and verification threads. |

One base per pack: several finetunes of one base can share a pack, each stored against the base.

### edit

```text
ziptensors edit [OPTIONS] <CONTAINER> (--add PATH | --add-as NAME PATH | --remove NAME | --replace NAME PATH)...
```

Changes the members of a container. At least one operation is required. Each member name may appear in
one operation only. Kept members keep their order, a replaced member keeps its slot, and added members
follow at the end in the order of `--add` and `--add-as` on the command line. The default strategy
writes a new file and renames it over the container; see [editing.md](editing.md) for the three
strategies and what each risks.

| Flag | Default | Meaning |
|---|---|---|
| `--add PATH` | | Add a file as a new member named after the file. Repeatable. |
| `--add-as NAME PATH` | | Add a file as the new member `NAME`. Repeatable. |
| `--remove NAME` | | Remove a member. Repeatable. |
| `--replace NAME PATH` | | Replace the content of member `NAME` with a file, in the member's place. Repeatable. |
| `--delta-base MEMBER` | none | Store every added or replacing file against this member as delta blobs where that is smaller: a kept member, or a file of this edit given before the others. Needs `--mode max`. |
| `-F`, `--fast` | off | Append strategy: only new data, a new index and footer are written after the end of the file. |
| `-R`, `--risky` | off | In-place strategy: rewrite inside the file with no extra space. An interruption damages the container. |
| `-C`, `--compact` | off | Re-encode every kept member with `--mode`; deduplication and delta pairing are redone over the result. Slow. |
| `--mode MODE` | `max` | Mode of new files, and of every member with `-C`. |
| `--threads N` | `0` | Compression and verification threads. |
| `--max-total-decoded BYTES` | none | Decode budget of the container being edited. |

Usage errors (exit 2): `-F` with `-R`, `-F` with `-C`, `-R` with `-C`, no operation, `--delta-base` with
a mode other than `max`. `-R` prints a warning on standard error before it starts.

The result is printed as `edited: <path> (rewritten | appended, -F | rewritten in place, -R)`, the counts
of kept, added, replaced and removed members, the new generation, the dead bytes the file held before,
and the report of `encode`.

### compact

```text
ziptensors compact [OPTIONS] <CONTAINER>
```

An edit with no operation: rewrites the container without the dead data of earlier `edit -F` runs.
With `-C`, every member is also re-encoded.

| Flag | Default | Meaning |
|---|---|---|
| `-R`, `--risky` | off | Compact in place, with no extra disk space. An interruption damages the container. |
| `-C`, `--compact` | off | Re-encode every member with `--mode`. Not with `-R` (exit 2). |
| `--mode MODE` | `max` | Used only with `-C`. |
| `--threads N` | `0` | Compression and verification threads. |
| `--max-total-decoded BYTES` | none | Decode budget of the container. |

### recover

```text
ziptensors recover [OPTIONS] <CONTAINER>
```

Shortens a container whose end was torn, for example by an interrupted `edit -F`, to its latest valid
generation. A file that opens as it is is left alone. See [editing.md](editing.md#recover) for when to
use it and what it can repair.

| Flag | Default | Meaning |
|---|---|---|
| `--dry-run` | off | Report what would be cut and write nothing. Works on a file you cannot write. |
| `--no-verify` | off | Do not decode the generation found before cutting. Only for a file never edited with `-R`. |
| `--max-candidates N` | `64` | Most candidate generations checked in full before giving up. |
| `--max-scan-bytes BYTES` | no limit | Most distinct bytes read while searching backwards for a footer. |
| `--threads N` | `0` | Verification threads. |
| `--max-total-decoded BYTES` | none | Decode budget each candidate is judged under. |

Prints the generation kept, the file length, the valid length, verified and scanned bytes (the distinct
bytes the backward search read in whole blocks from the end, at most the file length less 64) and the
number of newer candidate generations passed over. When bytes are cut (or would be, with `--dry-run`),
a `warning:` line on standard error says how many.

### restore

```text
ziptensors restore [OPTIONS] <CONTAINER> <OUTPUT>
ziptensors restore [OPTIONS] <CONTAINER> --all <DIR>
```

| Argument or flag | Default | Meaning |
|---|---|---|
| `OUTPUT` | | Where to write the only member. A container of several members fails with `MemberNotFound`; use `--all` or `extract`. Its directory must exist; an existing file is replaced once the restored one is proven. |
| `--all DIR` | | Restore every member under `DIR`, created if needed. Every name is checked before anything is created; a name that could escape `DIR` refuses the restore. Not with `OUTPUT`. |
| `--threads N` | `0` | Decode threads. |
| `--max-total-decoded BYTES` | none | Decode budget. |

### extract

```text
ziptensors extract [OPTIONS] <CONTAINER> <MEMBER> <OUTPUT>
```

Restores one member by name. `OUTPUT` is a file, written only once its BLAKE3 is proven, or `-` for
standard output. With `-`, the result line goes to standard error and the bytes are written as they are
decoded, before the hash of the whole member is known: a mismatch is reported afterwards with exit code
1, and whatever read the output must then discard it.

| Flag | Default | Meaning |
|---|---|---|
| `--threads N` | `0` | Decode threads. |
| `--max-total-decoded BYTES` | none | Decode budget. |

### merge

```text
ziptensors merge [OPTIONS] <OUTPUT> <INPUTS>...
```

Copies every member of the input containers, in order, into a new container. Stored chunks are copied
without recompression, so there is no `--mode`; delta blobs are copied with their bases. A member name
may appear only once across all inputs.

| Flag | Default | Meaning |
|---|---|---|
| `--threads N` | `0` | Verification threads. |
| `--max-total-decoded BYTES` | none | Decode budget of each input. |

### inspect

```text
ziptensors inspect [OPTIONS] <CONTAINER>
```

Prints the members (length, profile, tensor count, the first 16 hex digits of BLAKE3, name) and the
totals. Reads the index only; no data is decoded.

| Flag | Default | Meaning |
|---|---|---|
| `--tensors` | off | Also list each member's tensors: offset, length, dtype, shape, name. |
| `--json` | off | Print one JSON document instead of the table. |
| `--max-total-decoded BYTES` | none | Decode budget. |

With `--json`, standard output holds exactly one JSON document on one line:

```text
schema_version  number  1
path            string  the path as given
file_bytes      number  length of the container
raw_bytes       number  sum of member lengths
saved_bytes     number  raw_bytes - file_bytes, 0 if that is negative
members         array, in file order, of:
  name          string  member name, `/` separated
  len           number  bytes
  blake3        string  64 lowercase hex digits
  profile       string  opaque, safetensors, gguf, onnx, pt_zip, or unknown-<id>
  tensor_count  number
  tensors       array, only with --tensors, of:
    name        string
    offset      number  start in the member
    len         number  bytes
    dtype       string  F32, BF16, Q4_K, ..., or unknown-<id>
    shape       array of number
layers          object: raw, widen_saved, dedup_saved, entropy_saved, overhead, stored (numbers)
classes         array, skeleton first, then by dtype, of the fields of `layers` plus:
  class         string  skeleton, or the dtype
```

Numbers are exact unsigned 64-bit integers; a reader that parses them as doubles loses precision
above 2^53. A new key may appear without a change of `schema_version`; a removed or redefined key
changes it.

```sh
ziptensors inspect models.ziptensors --json 2>/dev/null | python3 -c \
  'import json, sys; print([m["name"] for m in json.load(sys.stdin)["members"]])'
```

### verify

```text
ziptensors verify [OPTIONS] <CONTAINER>
```

Decodes every member and compares it with its stated BLAKE3. Prints `verified N members, B bytes`.

| Flag | Default | Meaning |
|---|---|---|
| `--threads N` | `0` | Decode threads. |
| `--max-total-decoded BYTES` | none | Decode budget. |

### compress

```text
ziptensors compress [OPTIONS] <PATHS>...
```

Replaces every weight file under each path by a container of one member beside it
(`model.safetensors` becomes `model.ziptensors`) and removes the original once the container is proven.
Full guide: [directories.md](directories.md).

| Flag | Default | Meaning |
|---|---|---|
| `-r`, `--recursive` | off | Work through directories. A directory without `-r` is a usage error (exit 2). |
| `-k`, `--keep` | off | Keep every original beside its container. |
| `--dry-run` | off | Report what would be done and write nothing. Collisions are still reported as errors. |
| `--follow-symlinks` | off | Compress the target of each link to a weight file into a container in place of the link. Breaks a Hugging Face cache. |
| `--mode MODE` | `max` | Mode of each container. |
| `--threads N` | `0` | Compression and verification threads. |
| `--max-total-decoded BYTES` | none | Budget of each container already there and opened for a check. |

### decompress

```text
ziptensors decompress [OPTIONS] <PATHS>...
```

Restores every container of one member under each path beside it and removes the container once the
restored file is proven.

| Flag | Default | Meaning |
|---|---|---|
| `-r`, `--recursive` | off | Work through directories. Needed for a directory. |
| `-k`, `--keep` | off | Keep every container beside its restored file. |
| `--dry-run` | off | Report what would be done and write nothing. |
| `-f`, `--force` | off | Restore over a different file already at the restore path. |
| `--threads N` | `0` | Decode threads. |
| `--max-total-decoded BYTES` | none | Budget of each container. |

Both tree commands print one line per file they looked at, a `root:` line, totals and a `bytes:` line.
`decompress` looks at containers alone, so its totals leave out the counts only `compress` produces
(`compressed`, `recompressed`, `already compressed`, `not weights`), and `compress` leaves out
`restored`.
Every arrow points from source to result: in `compress` from the file to its container, in
`decompress` from the container to the restored file, sizes included. The `bytes:` line sums the files
whose two copies the run proved equal: `compress` prints `bytes: <original> original, <container> in
containers, <saved> saved (<percent>)`, `decompress` prints `bytes: <container> in containers ->
<restored> uncompressed`. A file that fails is a `FAILED:` line and the run goes on; the command then
exits with 1 at the end.

### mount

Linux and Windows only; macOS has no mount. Full guide: [mount.md](mount.md).

```text
ziptensors mount [OPTIONS] <CONTAINER> <MOUNTPOINT>
ziptensors mount --overlay [OPTIONS] <DIR> <MOUNTPOINT>
```

Shows every member of a container as a read-only file, or with `--overlay` a directory tree in which
each container made by `compress` appears as the original file it holds and every other file appears as
it is. The command runs in the foreground until Ctrl-C (Linux: also SIGTERM, SIGHUP or
`fusermount3 -u MOUNTPOINT`). Files still open when the unmount starts are served until they are
closed; a second signal ends the command at once with exit code 128 + the signal number.

| Argument or flag | Default | Meaning |
|---|---|---|
| `CONTAINER` or `DIR` | required | A container, or with `--overlay` a directory. The wrong kind is a usage error (exit 2). |
| `MOUNTPOINT` | required | Linux: an existing directory. Windows: a free drive letter (`X:`) or a directory that does not exist yet (its parent must); it is created for the mount and removed after. |
| `--threads N` | `0` | Decode threads. |
| `--fuse-threads N` | `4` | Linux: threads that serve kernel requests; 0 counts as 1. |
| `--winfsp-threads N` | `4` | Windows: threads that serve file system requests, 0 to 64; 0 means one per logical CPU from 4 to 16, 1 counts as 2. |
| `--cache-size BYTES` | `268435456` (256 MiB) | Decoded bytes kept in memory for reads smaller than a chunk; at least twice the largest chunk. With `--overlay`, shared by all open containers. |
| `--readahead WINDOWS` | `0` | Chunk windows decoded per cache miss, the missed one included; 0 means the decode thread count. |
| `--max-total-decoded BYTES` | none | Decode budget; with `--overlay`, of each container. |
| `--overlay` | off | Show a directory instead of a container. |
| `--max-open-containers N` | `4` | With `--overlay`: containers kept open at once, least recently read closed first. |
| `--max-open-files N` | `64` | With `--overlay`: plain files kept open at once. |

`--max-open-containers` and `--max-open-files` without `--overlay` are usage errors (exit 2).

Linux needs `/dev/fuse` and `fusermount3` on the `PATH` (package `fuse3`), not libfuse and not root.
Windows needs WinFsp, installed from <https://winfsp.dev/rel/>. WinFsp - Windows File System Proxy,
Copyright (C) Bill Zissimopoulos, <https://github.com/winfsp/winfsp>. WinFsp is licensed under GPL-3.0
with a FLOSS exception; ziptensors uses its DLL under that exception and distributes no WinFsp file.

```sh
ziptensors mount models.ziptensors /mnt/models
ziptensors mount --overlay /models/stable-diffusion-v1-5 /mnt/sd15     # after compress -r
```

```bat
ziptensors mount C:\models\models.ziptensors X:
```
