# Compressing model directories file by file

`ziptensors compress -r` replaces every weight file of a directory tree by a container of one member
beside it, and `ziptensors decompress -r` turns them back. Configs, tokenizers and every other file stay
as they are, so the directory keeps its layout. This is the alternative to `pack`, which puts a whole
directory into one container: per-file containers can be added, removed and restored one at a time, and
`mount --overlay` or the Python module can still present the directory with its original file names.

Flags are listed in [cli.md](cli.md#compress); the overlay mount is described in [mount.md](mount.md).

## Quick guide

```sh
# See what would happen; nothing is written
ziptensors compress -r --dry-run /models/stable-diffusion-v1-5

# Compress every weight file; each original is removed once its container is proven
ziptensors compress -r /models/stable-diffusion-v1-5

# Keep the originals as well
ziptensors compress -r -k /models/stable-diffusion-v1-5

# Restore the originals; each container is removed once the restored file is proven
ziptensors decompress -r /models/stable-diffusion-v1-5

# Single files work without -r
ziptensors compress /models/sd-vae.safetensors
ziptensors decompress /models/sd-vae.ziptensors
```

A run prints one line per file it looked at on standard output, then totals; log lines and warnings go
to standard error:

```text
skipped: sd/linked.safetensors: symbolic link, not followed
not a weight file: sd/model_index.json
not a weight file: sd/unet/config.json
compressed: sd/unet/diffusion_pytorch_model.safetensors -> sd/unet/diffusion_pytorch_model.ziptensors (12448 -> 11267 bytes), removed sd/unet/diffusion_pytorch_model.safetensors
root: sd
totals: 1 compressed, 0 recompressed, 0 pairs resolved, 0 pairs verified, 0 already compressed, 2 not weights, 0 left alone, 1 skipped, 0 planned, 0 failed
bytes: 12448 original, 11267 in containers, 1181 saved (9.48%)
```

`decompress` prints the same kinds of lines, each arrow from the container to the restored file. It
looks at containers alone, so other files get no line, and its totals list only what a restore can
end in. It ends with the bytes it read and restored:

```text
skipped: sd/linked.safetensors: symbolic link, not followed
restored: sd/unet/diffusion_pytorch_model.ziptensors -> sd/unet/diffusion_pytorch_model.safetensors (11267 -> 12448 bytes), removed sd/unet/diffusion_pytorch_model.ziptensors
root: sd
totals: 1 restored, 0 pairs resolved, 0 pairs verified, 0 left alone, 1 skipped, 0 planned, 0 failed
bytes: 11267 in containers -> 12448 uncompressed
```

## Which files are compressed

A file is a weight file when its **content** is recognized; its name, extension and size play no part:

| Format | Recognized by |
|---|---|
| safetensors | its JSON header |
| GGUF | the `GGUF` magic and its header |
| ONNX | its protobuf model structure |
| PyTorch zip (`torch.save` since PyTorch 1.6, often `.pt`, `.pth`, `.bin`) | the zip archive layout of a torch save |
| Legacy PyTorch (`torch.save(..., _use_new_zipfile_serialization=False)`, saves before 1.6) | the pickled magic number at the start; stored as opaque bytes |

Every other file is reported as `not a weight file` and never touched. A weight file whose header is
truncated or damaged is not recognized, and neither is a legacy torch save made with a custom
`pickle_module` or pickle protocol 6 or later. A wrong guess costs only compression ratio: every
container restores bit-exact.

Containers already in the tree are recognized by their magic bytes. A container of exactly one member
whose name is a plain file name is a **per-file container**: `decompress` restores it, and `compress`
treats it as the compressed form of that file. Any other container (a pack of several members, or a
member name with `/`) is `left alone` by both commands.

Also skipped, without being read: symbolic links (see below), FIFOs, sockets and devices, temporary
files of an interrupted write (`*.ziptensors-tmp`, never removed automatically), partial downloads
(`*.incomplete`), and names that are not valid UTF-8.

## Naming

`dir/name.ext` becomes `dir/name.ziptensors`, holding one member named `name.ext`; a name without an
extension gets `.ziptensors` appended. The member name is what a restore writes: `decompress` restores
`dir/<member name>`, whatever the container is called, so renaming a container does not change the file
it restores to.

A container has a fixed overhead of a few hundred bytes, so a very small weight file can become larger;
it is still compressed and reported with both sizes.

## Removing the originals safely

- An original is removed only after its container was committed, decoded in full and checked against
  the original's BLAKE3, and only if the original's length and modification time did not change in the
  meantime. A crash at any point leaves the original, the container, or both; never neither.
- `decompress` does the same in the other direction: the container is removed only after the restored
  file is proven.
- `-k`/`--keep` keeps the redundant copy: originals in `compress`, containers in `decompress`.
- A file that fails is reported as `FAILED:` with the reason, the run goes on, and the command exits with
  1 at the end. A failure before the removal removes nothing. The one failure that can follow the
  removal is a failed sync of the directory: the verified twin is in place, but a power loss may bring
  the removed copy back, which a rerun resolves as a pair (Python: `FileOutcome.removed` names it).
- Restored files are content only: modification times and permissions are not restored.

## Reruns

A rerun is safe and settles the tree. Where a weight file and its container both exist (after `-k`, or
after an interrupted run), the uncompressed file is **hashed every time** and compared with the
member's stated BLAKE3 and length:

| Situation | `compress` | `decompress` |
|---|---|---|
| Container only | `already compressed`, left alone | restored, container removed |
| Both, equal | container verified, original removed (`pair resolved`; with `-k`: `pair verified`) | container removed (`pair resolved`; with `-k`: `pair verified`) |
| Both, different | container is stale: replaced by a new one of the file on disk (`recompressed`) | `FAILED: a different file is already there`; `-f`/`--force` restores over it |

Hashing every pair costs a full read of the uncompressed file on each rerun.

## Collisions

Before anything is written, each path given is planned as a whole, and any name collision refuses it
with `TreeConflict` (code 25, exit 1). This holds for `--dry-run` too. Nothing is written under that
path. Paths are worked through in the order given and the command stops at the refused one, so paths
named before it are already compressed (or restored).

`compress` refuses:

- two weight files that map to one container name, compared ignoring case (`model.safetensors` and
  `model.bin` both map to `model.ziptensors`);
- a container name already taken by anything that is not a per-file container of exactly that file;
- an original whose name another container already restores to.

`decompress` refuses several containers that restore to one name, or a container that would restore
onto a directory or onto a container file.

```text
error: cannot compress col: 1 name conflicts under col, nothing was written: col/model.bin and col/model.safetensors both map to the container col/model.ziptensors
hint: nothing was written: rename or remove the files named, so that every file gets a name of its own
```

Rename or remove one of the files named, then run again.

## Symbolic links and the Hugging Face cache

Symbolic links are **not followed** by default. Each is reported as `skipped`, and a warning on standard
error counts them.

`--follow-symlinks` compresses the target of each link to a weight file into a container **at the
link's place**, named after the link, and removes the link; the target itself is never removed. A link
to a directory is never entered, a dangling link is skipped, and a target outside the tree is only read.
Two links to one file give two full containers.

Do not use `--follow-symlinks` on a Hugging Face cache (`~/.cache/huggingface/hub`). There, the files
under `snapshots/<revision>/` are links into `blobs/`, which holds the data. Following them:

- replaces each link in `snapshots/` by a container, while the blob it pointed to stays in `blobs/`, so
  the cache **grows** by the size of every container instead of shrinking;
- removes the links that the Hugging Face libraries look for (`model.safetensors` is gone, a
  `model.ziptensors` file is there instead), so the cache stops working as a cache: a later download
  or load sees the file as missing.

Compress a plain copy of the model instead, for example one made with
`huggingface_hub.snapshot_download(repo_id, local_dir="/models/name")`.

## Using a compressed directory

- **Mount** (Linux and Windows): `ziptensors mount --overlay DIR MOUNTPOINT` shows the directory with
  each per-file container as the original file it holds, under its original name, and every other file
  as it is. Programs that take a path, such as Diffusers `from_pretrained`, read the mounted directory
  unchanged. Details, limits and the Windows requirements in [mount.md](mount.md).

  ```sh
  ziptensors mount --overlay /models/stable-diffusion-v1-5 /mnt/sd15
  ```

  On Windows the mount uses WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos,
  <https://github.com/winfsp/winfsp> (GPL-3.0 with a FLOSS exception), installed from
  <https://winfsp.dev/rel/>.

- **macOS, or without a mount:** [`examples/diffusers_from_container.py`](../examples/diffusers_from_container.py)
  loads a Diffusers pipeline (Stable Diffusion 1.x, SDXL) straight from a compressed directory with
  `--per-file-dir`. Configs, tokenizers and schedulers are read from the directory; each model's weights
  come from its per-file containers, found by member name: one weight file per model folder, or shards
  through their `*.index.json`, safetensors preferred over `.bin`, a legacy `.bin` only through
  `torch.load(..., weights_only=True)`. `--variant fp16` selects a Diffusers variant. It needs
  `diffusers`, `transformers`, `torch` and `ziptensors`.

  ```sh
  ziptensors compress -r /models/stable-diffusion-v1-5
  python examples/diffusers_from_container.py --per-file-dir /models/stable-diffusion-v1-5 "a lighthouse at sunset" out.png --steps 20 --seed 42
  ```

- **Python:** open one container with `ziptensors.Container(path)`; its only member is the original
  file. See [python.md](python.md).

## Python

```python
import ziptensors

plan = ziptensors.compress_tree("/models/stable-diffusion-v1-5", dry_run=True)
for outcome in plan.files:
    print(outcome.status, outcome.path, outcome.counterpart)

report = ziptensors.compress_tree("/models/stable-diffusion-v1-5")    # keep=, follow_symlinks=, mode=
print(report.totals.compressed, report.has_failures)

with ziptensors.mount_overlay("/models/stable-diffusion-v1-5", "/mnt/sd15") as mount:
    data = (mount.mountpoint / "unet" / "diffusion_pytorch_model.safetensors").read_bytes()

ziptensors.decompress_tree("/models/stable-diffusion-v1-5")            # keep=, force=, dry_run=
```

A collision raises `ziptensors.InputError` (code 25) before anything is written; a file that fails is
an outcome with status `failed` and its error, and the run goes on. Statuses: `compressed`,
`recompressed`, `pair_resolved`, `pair_verified`, `already_compressed`, `restored`, `not_weight`,
`foreign` (left alone), `skipped`, `planned`, `failed`.

## Rust

```rust
use std::path::Path;
use ziptensors_core::tree::{CompressTreeOptions, DecompressTreeOptions, compress_tree, decompress_tree};

fn compress_model() -> Result<(), ziptensors_core::Error> {
    let root = Path::new("/models/stable-diffusion-v1-5");
    let mut options = CompressTreeOptions::default();
    options.dry_run = true;
    for file in &compress_tree(root, &options)?.files {
        println!("{:?} {}", file.status, file.path.display());
    }
    options.dry_run = false;
    let done = compress_tree(root, &options)?;
    println!("{} compressed, {} failed", done.totals().compressed, done.totals().failed);
    decompress_tree(root, &DecompressTreeOptions::default())?;
    Ok(())
}
```

The options structs are `#[non_exhaustive]`: start from `default()` and set fields. See
[rust.md](rust.md).
