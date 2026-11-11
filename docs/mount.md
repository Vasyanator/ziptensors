# Mounting

A mount shows a container, or a directory of compressed files, as an ordinary read-only directory tree.
Programs that expect model files on disk (Diffusers `from_pretrained`, llama.cpp, ComfyUI, the
safetensors package, `torch.load`) read them unmodified; each read decodes only the part of the file it
touches.

| Platform | Backend | Requirements |
|---|---|---|
| Linux | FUSE, through the kernel's `/dev/fuse` | `/dev/fuse` and `fusermount3` (package `fuse3`). No libfuse, no root. |
| Windows | WinFsp | WinFsp 2.0 or later installed from <https://winfsp.dev/rel/>. |
| macOS | none | No mount. Read through the Python module, or decompress (see [below](#macos)). |

Windows mounts use WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos,
<https://github.com/winfsp/winfsp>. WinFsp is licensed under GPL-3.0 with a FLOSS exception, which
allows ziptensors to use it; ziptensors distributes no WinFsp file. Install it from
<https://winfsp.dev/rel/>.

There are two kinds of mount:

- **Container mount** (`ziptensors mount`, Python `mount`): every member of one container appears as a
  file at its member path. A pack of a model directory shows that directory again.
- **Overlay mount** (`ziptensors mount --overlay`, Python `mount_overlay`): a directory tree in which
  `ziptensors compress -r` (or Python `compress_tree`) replaced weight files by per-file containers
  appears as it was before, each container shown as the original file under its original name.

## Quick start

### Command line

```sh
ziptensors pack sd15.ziptensors stable-diffusion-v1-5/
mkdir -p mnt
ziptensors mount sd15.ziptensors mnt
```

```text
mounted sd15.ziptensors at mnt; press Ctrl-C or run `fusermount3 -u mnt` to unmount
```

The command stays in the foreground while the mount exists. In another terminal, use `mnt/` like the
original directory. Press Ctrl-C (or run `fusermount3 -u mnt`) to end it:

```text
unmounted mnt; decoded 8472 bytes in 1 cache misses, 2 hits
```

For a tree compressed in place:

```sh
ziptensors compress -r models/
ziptensors mount --overlay models/ mnt
```

On Windows, give a free drive letter or a path that does not exist yet:

```bat
ziptensors mount C:\models\sd15.ziptensors X:
ziptensors mount --overlay C:\models C:\mounted\models
```

### Python

```python
import os
import ziptensors

os.makedirs("mnt", exist_ok=True)                     # Linux; on Windows pass "X:" instead
with ziptensors.mount("pipeline.ziptensors", "mnt") as mounted:
    for root, dirs, files in os.walk(mounted.mountpoint):
        print(root, sorted(files))
# the with block unmounted it, also after an exception or Ctrl-C
```

```python
import os
import ziptensors

os.makedirs("mnt", exist_ok=True)
with ziptensors.mount_overlay("models/", "mnt", max_total_decoded=64 << 30) as mounted:
    print(sorted(p.name for p in mounted.mountpoint.rglob("*")))
```

See [python.md](python.md#reference-mounting) for the full Python reference.

## What the mounted tree looks like

**Container mount.** Each member is a regular file at its member path; the directories between are
created from the paths. Files have the member's length and are byte-identical to the original files.
Every file and directory carries the modification time of the container file.

**Overlay mount.** The tree mirrors the directory:

| On disk | In the mount |
|---|---|
| A plain file | Shown as it is. |
| A per-file container (one member with one safe name, made by `compress -r`) | Shown as its member: the original name and length, decoded on read. |
| A pack (a container with several members, or a member name that is not a single safe name) | Shown as it is, as a container file. |
| A container whose original still sits beside it (a pair kept with `--keep`) | Hidden; the original file is read directly. |
| A directory, empty ones included | Shown. |
| Symbolic links, special files, `*.ziptensors-tmp`, `*.incomplete`, names that are not UTF-8 | Left out, with one log line each. |

The overlay is a snapshot taken at mount time: the tree, lengths and times do not change afterwards.
Each read compares the file on disk with the snapshot (length and modification time) and fails with an
I/O error if the file changed or disappeared, so a read never mixes two versions of a file. A writer that
restores the old modification time is not detected for plain files; containers additionally check chunk
hashes.

An overlay refuses to mount at all, naming every problem, when an entry is damaged (a container that does
not open) or when two entries claim one name (for example `model.safetensors` beside a container whose
member is a different `model.safetensors`).

**Names.** A tree whose names the operating system cannot show is refused before anything is mounted,
with every bad name listed. Names that a restore refuses are always refused. On Windows the rules of
NTFS apply in addition: no `< > : " | ? * \` or control characters, no reserved device names (`CON`,
`NUL`, `COM1`, ... with any extension), no trailing dot or space, at most 255 UTF-16 units per part, and
names that differ only in letter case are one name, so such a pair refuses the mount. A full path under
the mount point may be at most 259 characters; mount on a drive letter or a shorter directory if the
mount is refused for long paths.

## Read-only semantics

- Everything that would change the tree is refused: opening for writing, creating, deleting, renaming,
  changing attributes. Linux reports `EROFS` ("Read-only file system").
- Linux: files are mode `0444`, directories `0555`, owned by the mounting user. The mount uses the
  options `ro,nosuid,nodev,noatime` and `default_permissions`, and no `allow_other`: only the mounting
  user can see the files. The kernel keeps file contents and attributes cached (contents never change),
  so a second read of a file costs no request.
- Windows: the volume is read-only and the security descriptor admits the mounting user only. A drive
  letter mounted from an interactive console exists in that logon session only; from a service or an
  OpenSSH session (session 0) it is defined globally. A directory mount point is a junction, visible to
  every session but still readable only by the mounting user.
- Memory maps (`mmap`) work: safetensors, llama.cpp and `torch.load(..., mmap=True)` on a mounted path map
  the file and read pages on demand.
- Damaged data is never served: a chunk that fails its hash makes that read fail with an I/O error
  (`EIO`), and the error is logged with member, offset and error code.

## Options

| CLI flag | Python keyword | Default | Meaning |
|---|---|---|---|
| `--threads N` | `threads` | `0` / `None` | Decode threads; `0` (or `None`) means one per logical CPU. Overlay: per open container. |
| `--cache-size BYTES` | `cache_size` | 268435456 (256 MiB) / `None` | Decoded bytes kept in memory for reads smaller than a chunk. Container mount: at least twice the container's largest chunk, otherwise the mount fails with a budget error. Overlay: shared by all open containers (see below). |
| `--readahead WINDOWS` | `readahead` | `0` | Cache windows decoded by one cache miss, the missed one included; `0` means the decode thread count, `1` disables read-ahead. |
| `--max-total-decoded BYTES` | `max_total_decoded` | no limit | Refuse a container whose members add up to more than this, before reading any member data. Overlay: applies to each container. Set it for files you do not trust (see [safety.md](safety.md)). |
| `--overlay` | (use `mount_overlay`) | off | Show a directory instead of a container. |
| `--max-open-containers N` | `max_open_containers` | `4` | Overlay only: containers kept open at once, at least 1. The least recently read one is closed first. |
| `--max-open-files N` | `max_open_files` | `64` | Overlay only: plain files kept open at once, at least 1. |
| `--fuse-threads N` (Linux) | fixed at 4 | `4` | Threads that serve requests from the kernel; `0` counts as 1. |
| `--winfsp-threads N` (Windows) | fixed at 4 | `4` | Threads that serve requests from the WinFsp driver, at most 64; `0` means one per logical CPU, from 4 to 16, and `1` counts as 2. |

The CLI takes `CONTAINER` without `--overlay` and `DIR` with it; the wrong kind is a usage error (exit
status 2). `--max-open-containers` and `--max-open-files` require `--overlay`.

### How the cache works

The kernel asks for small pieces (FUSE sends reads of 128 KiB), while the container stores data in
chunks of up to 8 MiB. The read cache keeps decoded *windows* (the part of one chunk one file shows),
evicting the least recently used by bytes, so neighbouring small reads decode a chunk once. A miss
decodes up to `readahead` windows, the missed one and those that follow it in the same file, in parallel
on the decode threads, which is what makes sequential loads fast.

- Sequential loading of large files (most model loaders): the defaults are right. A larger `readahead`
  can help on many cores.
- Many processes reading different parts at random: raise `cache_size`.
- Low memory: lower `cache_size` and `threads`. Memory is bounded by `cache_size`, plus one window per
  concurrent miss, plus the decoder's working memory (about 3 × threads × chunk size per fill).

**Overlay cache.** Containers open lazily, on their first read. Each open container gets its own cache of
`cache_size / max_open_containers` bytes, which must be at least 16 MiB (twice the largest chunk the
format allows, since chunk sizes are unknown before a container opens); a smaller share is refused at
mount (CLI error, Python `ValueError`). With the defaults each container gets 64 MiB. Raising
`max_open_containers` therefore shrinks each cache unless `cache_size` grows with it.

## Unmounting

| How | What happens |
|---|---|
| Ctrl-C in the terminal of `ziptensors mount` (Linux also SIGTERM, SIGHUP) | Unmounts: the mount point is detached at once, files still open under it are served until they are closed, then the command prints `unmounted ...` and exits with status 0. |
| A second Ctrl-C while files are still open | Ends the process at once with status 130 (128 + SIGINT; Windows Ctrl-Break: 149). Readers still open are cut off. |
| `fusermount3 -u <mountpoint>` (Linux) | Same as the first Ctrl-C, from any shell of the mounting user. |
| Python: leaving the `with` block, or `Mount.unmount()` | Detaches at once, then waits (GIL released) until every file open under the mount is closed, memory maps included. |
| The process was killed (SIGKILL) | The mount stays behind: run `fusermount3 -u <mountpoint>` (or `fusermount3 -u -z` if it is busy). |

In Python, close every file opened under the mount before leaving the `with` block: a file held open by
the thread that unmounts makes it wait forever. If an unmount fails, `MountError` is raised and the
`Mount` object gives the mount up; on Linux remove it with `fusermount3 -u -z <mountpoint>`. A `Mount`
dropped while mounted is unmounted on a background thread with a warning in the log. On Windows only
open files are waited for, not directories (such as a shell's current directory). Python has no call
that blocks until an outside unmount; keep the `with` block open for as long as the mount is needed.

After unmounting on Linux, `grep ziptensors /proc/mounts` prints nothing.

## Linux requirements

- `/dev/fuse` must exist and be readable and writable by your user (`ls -l /dev/fuse` shows
  `crw-rw-rw-` on most distributions).
- `fusermount3` must be on `PATH`: install the package `fuse3` (Debian, Ubuntu, Fedora, Arch).
  ziptensors talks to the kernel itself and needs no libfuse library.
- No root is needed.
- The mount point is an existing directory. Anything inside it is hidden while the mount is there.

## Windows requirements

- Install WinFsp 2.0 or later from <https://winfsp.dev/rel/>. ziptensors loads the WinFsp DLL at run time
  from the installer's registry entry; without WinFsp the mount fails with "WinFsp is not installed;
  install it from https://winfsp.dev/rel/ and try again". Nothing else needs to be installed.
- The mount point is one of:
  - a drive letter that is not in use: `X:` (also accepted as `x:`, `X:\` or `X:/`);
  - a local directory that **does not exist yet**, whose parent exists, such as `C:\mounted\models`. The
    mount creates it and removes it after. A relative path is resolved against the current directory.
  An existing directory, a drive letter in use, a drive root such as `C:\`, a drive-relative path such
  as `C:dir`, and UNC, device or `\\?\` paths are refused.
- `ziptensors mount --help` on Windows shows the WinFsp notice.

WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos, <https://github.com/winfsp/winfsp>
(GPL-3.0 with a FLOSS exception). A proprietary program that embeds the ziptensors mount library on
Windows needs WinFsp's commercial license; the ziptensors command line tool and Python package use it
under the FLOSS exception.

## macOS

There is no mount backend on macOS: the command line tool has no `mount` command, and the Python
functions raise `ziptensors.MountError` (code 4, `UnsupportedFeature`). Instead:

- Read through the Python module, which needs no mount: `ziptensors.torch.load_state_dict`,
  `ziptensors.safetensors.safe_open`, or `Member.open()` (see [python.md](python.md)). The repository's
  `examples/diffusers_from_container.py` builds a Diffusers pipeline straight from a packed container or
  from a tree compressed in place:

  ```sh
  ziptensors compress -r stable-diffusion-v1-5/
  python examples/diffusers_from_container.py --per-file-dir stable-diffusion-v1-5/ "a lighthouse at sunset" out.png
  ```

- Or restore the files: `ziptensors decompress -r <dir>` for a tree, `ziptensors restore <pack> --all
  <dir>` for a pack (see [cli.md](cli.md) and [directories.md](directories.md)).

## Typical uses

### Diffusers and Transformers

```python
import os
import ziptensors
from diffusers import StableDiffusionPipeline

os.makedirs("mnt", exist_ok=True)
with ziptensors.mount("sd15.ziptensors", "mnt", max_total_decoded=64 << 30) as mounted:
    pipe = StableDiffusionPipeline.from_pretrained(mounted.mountpoint)
    image = pipe("a lighthouse at sunset").images[0]
```

`from_pretrained` reads `model_index.json`, the configs and the weights from the mount exactly as from the
original directory, and memory-maps safetensors files. A tree compressed in place works the same through
`mount_overlay`, including sharded checkpoints (`*.index.json`), since every shard keeps its name.
Objects that still map files from the mount (a model loaded with mmap) must be released before the
`with` block ends, or the unmount waits for them.

### llama.cpp

```sh
ziptensors encode model-Q4_K_M.gguf model-Q4_K_M.ziptensors
mkdir -p mnt
ziptensors mount model-Q4_K_M.ziptensors mnt &
llama-cli -m mnt/model-Q4_K_M.gguf -p "Hello"
fusermount3 -u mnt
```

llama.cpp memory-maps the GGUF file; pages are decoded as they are touched. With `--no-mmap` it reads the
file sequentially, which the read-ahead serves.

### ComfyUI and other UIs

Mount the model folder somewhere of its own and point the program at it. For ComfyUI, add the mount to
`extra_model_paths.yaml`:

```yaml
ziptensors:
    base_path: /mnt/models
    checkpoints: checkpoints
    loras: loras
    vae: vae
```

with `ziptensors mount --overlay /models /mnt/models` running (a tree compressed with
`ziptensors compress -r /models`), or `ziptensors mount /data/models.ziptensors /mnt/models` (a pack made
with `ziptensors pack /data/models.ziptensors /models`). The UI only reads, so the mount
being read-only does not matter; new models are added to the source directory and become visible after
remounting, since an overlay is a snapshot.

## Errors

| Situation | CLI | Python |
|---|---|---|
| Container does not open (not a container, damaged, too new, over `--max-total-decoded`) | exit status 1, message and hint | `FormatError`, `CorruptionError`, `UnsupportedError`, `BudgetError` |
| `--cache-size` smaller than twice the largest chunk | exit status 1 | `BudgetError` |
| Overlay cache share under 16 MiB, or an open limit of 0 | exit status 1 (limits of 0 are a usage error, status 2) | `ValueError` |
| Names the OS cannot show | exit status 1, every name listed | `MountError`, code 14 |
| Overlay entry damaged / two entries claim one name | exit status 1, every entry listed | `MountError`, code of the damage / 25 |
| No `/dev/fuse` or `fusermount3`, mount point missing, no WinFsp, mount point taken | exit status 1 | `MountError`, code 1 |
| An unmount left the mount in place | error with the `fusermount3 -u -z` hint | `MountError`, code 1 |
| macOS | no `mount` command | `MountError`, code 4 |
