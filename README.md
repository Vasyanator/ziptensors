# ziptensors

Lossless compression for model weights, with random access at tensor speed.

ziptensors stores safetensors, GGUF, ONNX and PyTorch zip files in a `.ziptensors` container that is
typically 8 to 33 % smaller than the source, and every restored file is bit-exact. Readers decode any
byte range or any single tensor straight into a caller's buffer, in parallel, without unpacking the
whole file. One container can hold many files, so identical tensors across them are stored once,
and a finetune can be stored as a delta against its base model.

- **Lossless and proven.** A restored file is checked against the BLAKE3 hash of the original; the
  writer decodes and checks every container before it replaces anything.
- **Format aware, never format dependent.** Parsers for safetensors, GGUF, ONNX and PyTorch zip pick a
  transform per tensor (byte-plane split for floats, field split for quantized GGUF blocks). Every
  other file, and any file a parser does not understand, is stored as opaque bytes and still restores
  exactly.
- **Random access.** Data is cut into independent chunks of about 1 MiB. Reading a tensor decodes
  only its chunks.
- **Many files, one container.** Deduplication by content hash across all members; an F32 file that
  only holds F16 values shares storage with its F16 sibling.
- **Delta blobs.** A finetune packed against its base stores only the XOR residual of changed tensors.
- **Small, bounded memory.** Readers and writers stream; memory does not grow with tensor size.
- **Frontends.** A command line tool, a Rust library, a Python module with PyTorch and safetensors
  adapters, a C ABI, and a read-only filesystem mount on Linux and Windows.

## Install

| What | Command | Gives you |
|---|---|---|
| Command line tool | `cargo install ziptensors-cli` | the `ziptensors` binary |
| Python module | `pip install ziptensors` (`pip install "ziptensors[torch]"` for the PyTorch adapter) | `import ziptensors`, CPython 3.11+ |
| Rust library | `cargo add ziptensors-core` | the whole format: reader, writer, codecs |
| C library | see [`crates/ziptensors-capi`](crates/ziptensors-capi/README.md) | `ziptensors.h` and a shared library |

Building from source needs a Rust toolchain and a C compiler (zstd is compiled from source); no system
library is linked.

## Quick start: command line

```sh
# One file -> a single-member container (default mode: balanced)
ziptensors encode model.safetensors model.ziptensors

# Files and directory trees -> one container (default mode: max, with deduplication)
ziptensors pack models.ziptensors sdxl-inpaint/ flux2-klein-9b.safetensors

# A finetune stored as delta blobs against its base (needs mode max, the default of pack)
ziptensors pack pair.ziptensors base.safetensors finetune.safetensors --delta-base base.safetensors

# What is inside, and how much each layer saved
ziptensors inspect models.ziptensors
ziptensors inspect models.ziptensors --json

# Decode everything and check every member against its BLAKE3
ziptensors verify models.ziptensors

# Get files back
ziptensors restore model.ziptensors model.safetensors          # the only member
ziptensors restore models.ziptensors --all restored/           # every member under a directory
ziptensors extract models.ziptensors unet/diffusion_pytorch_model.safetensors unet.safetensors  # names relative to sdxl-inpaint/

# Copy members of several containers into one, without recompressing
ziptensors merge all.ziptensors a.ziptensors b.ziptensors

# Change a container: add, rename-on-add, remove, replace (default: rewrite through a temporary file)
ziptensors edit models.ziptensors --add lora.safetensors --add-as vae/vae.safetensors new-vae.safetensors
ziptensors edit models.ziptensors --remove old.safetensors --replace config.json config-v2.json
ziptensors edit models.ziptensors --add extra.safetensors -F   # append only: fast, old data stays as dead bytes
ziptensors compact models.ziptensors                          # drop dead bytes; -C re-encodes every member
ziptensors recover models.ziptensors --dry-run                # after an interrupted -F: what would be cut
# -R rewrites in place with no extra space: an interruption DAMAGES the container. Keep a backup.

# A member to standard output; the tensor table of every member
ziptensors extract models.ziptensors config.json - | head
ziptensors inspect models.ziptensors --tensors

# Every weight file of a tree into its own container beside it, and back (originals removed once proven)
ziptensors compress -r stable-diffusion-v1-5/
ziptensors decompress -r stable-diffusion-v1-5/

# Show the members as read-only files until Ctrl-C (Linux: an existing directory; Windows: X: or a new directory)
ziptensors mount models.ziptensors /mnt/models
# A tree after `compress -r`, shown with its original file names (other files pass through)
ziptensors mount --overlay stable-diffusion-v1-5/ /mnt/sd15
```

`--threads N` sets the thread count (default: one per logical CPU). `ziptensors <command> --help`
lists every option. Symbolic links and `*.incomplete` files are skipped by `pack` and `compress`, with a
log line (`compress --follow-symlinks` follows links, which breaks a Hugging Face cache). Pass
`--max-total-decoded BYTES` to any command that opens a container you did not write. Restore is
content-only: modification times and permissions are not kept.

## Quick start: Python

```python
import ziptensors

ziptensors.encode_file("model.safetensors", "model.ziptensors", mode="max")
ziptensors.pack(["base.safetensors", "finetune.safetensors"], "pair.ziptensors",
                delta_base="base.safetensors")

# Set a budget when the file comes from somewhere else (see Security below).
container = ziptensors.Container("model.ziptensors", max_total_decoded=64 << 30)
member = container.member()                  # the only member; or container.member("name")

for tensor in member.tensors:
    print(tensor.name, tensor.dtype, tensor.shape)

buffer = bytearray(member.tensor_info("lm_head.weight").len)
member.decode_tensor_into("lm_head.weight", buffer)   # any writable buffer: bytearray, numpy array, ...
member.restore_file("model.safetensors")

# PyTorch: tensors decoded in place, on CPU or straight to a CUDA/ROCm device
from ziptensors.torch import load_state_dict
state_dict = load_state_dict(member, device="cuda:0")

# Drop-in reader with the API of safetensors.safe_open
from ziptensors.safetensors import safe_open
with safe_open("model.ziptensors", framework="pt", max_total_decoded=64 << 30) as f:
    weight = f.get_tensor("lm_head.weight")
```

Reads release the GIL and are safe from many threads. `Member.open()` returns a seekable read-only
file object, for example for `torch.load`. Quantized GGUF tensors are never dequantized: they are
refused by default or handed over as raw blocks (`quantized="raw"` or `"diffusers"`).

## API and CLI at a glance

Full user documentation is in [`docs/`](docs/README.md).

**Command line** ([`docs/cli.md`](docs/cli.md); exit codes 0 success, 1 failure, 2 usage):

| Command | What it does |
|---|---|
| `encode IN OUT` | One file into a container of one member (mode `balanced`) |
| `pack OUT INPUTS...` | Files and directories into one container (mode `max`, deduplication, `--delta-base`) |
| `edit C --add/--add-as/--remove/--replace` | Change members; default rewrite, `-F` append, `-R` in place, `-C` repack ([`docs/editing.md`](docs/editing.md)) |
| `compact C` | Drop dead data of earlier appends; `-C` re-encodes every member |
| `recover C` | Cut a torn end back to the latest valid generation; `--dry-run` first |
| `restore C OUT`, `restore C --all DIR` | Restore the only member, or every member |
| `extract C MEMBER OUT\|-` | Restore one member to a file or to standard output |
| `merge OUT INPUTS...` | Copy members of several containers into one, no recompression |
| `inspect C` | Members and saved bytes; `--tensors`, `--json` |
| `verify C` | Decode every member and check its BLAKE3 |
| `compress -r DIR`, `decompress -r DIR` | Per-file containers beside each weight file, and back ([`docs/directories.md`](docs/directories.md)) |
| `mount C MNT`, `mount --overlay DIR MNT` | Read-only files until Ctrl-C, Linux and Windows ([`docs/mount.md`](docs/mount.md)) |

**Python** ([`docs/python.md`](docs/python.md)): `Container(path, max_total_decoded=...)` with
`member(name)` and `restore_all(dir)`; `Member` with `tensors`, `read_at`, `decode_tensor_into`,
`open()`, `restore_file`, `verify`; writers `encode_file`, `pack`, `PackBuilder`; `edit` (`AddOp`,
`RemoveOp`, `ReplaceOp`), `compact`, `recover`; `compress_tree`, `decompress_tree`; `mount`,
`mount_overlay`; adapters `ziptensors.torch.load_state_dict` and `ziptensors.safetensors.safe_open`.

**Rust** ([`docs/rust.md`](docs/rust.md)), crate `ziptensors-core`: `Container::open(path, &OpenOptions)`,
`Container::member`/`single`, `Member` (`read_at`, `decode_tensor_into`, `restore_file`), `ReadCache`;
`encode_file`, `PackBuilder`; `edit` with `EditOp`/`EditOptions`/`Strategy`; `recover`;
`tree::compress_tree`/`decompress_tree`; `Error` with a stable `ErrorCode`. The mount is the crate
`ziptensors-mount`.

**C** ([`docs/c-api.md`](docs/c-api.md)), `ziptensors.h`: `zt_open`/`zt_close`, `zt_member_open`,
`zt_read_at`, `zt_decode_tensor_into`, `zt_restore_file`, `zt_restore_all`; `zt_encode_file`,
`zt_pack_new`/`zt_pack_add_file`/`zt_pack_finish`; `zt_edit`, `zt_recover`; fallible calls return a
`zt_status`, and `zt_last_error` gives the message.

## Example: Stable Diffusion with Diffusers

[`examples/diffusers_from_container.py`](examples/diffusers_from_container.py) runs a Diffusers model
directory (Stable Diffusion 1.x or SDXL, weights in safetensors) straight from a container, without a
mount and without unpacking the weights. Needs `diffusers`, `transformers`, `torch` and `ziptensors`.

```sh
ziptensors pack sd15.ziptensors stable-diffusion-v1-5/      # the whole directory: weights, configs, tokenizer
python examples/diffusers_from_container.py sd15.ziptensors "a lighthouse at sunset" out.png --steps 20 --seed 42
```

The example restores only the small files (configs, tokenizer, scheduler) to a temporary directory,
builds each model from its config, and decodes its weights from the container onto the device:

```python
container = ziptensors.Container("sd15.ziptensors", max_total_decoded=64 << 30)
container.member("unet/config.json").restore_file(tmp / "unet/config.json")   # likewise every small file
with torch.device("cuda"):
    unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(tmp / "unet"))
unet.to(dtype=torch.float16)
unet.load_state_dict(load_state_dict(container.member("unet/diffusion_pytorch_model.safetensors"),
                                     device="cuda"), strict=True)
# ... the same for vae and text_encoder; tokenizer and scheduler from their restored files
pipeline = StableDiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                                   scheduler=scheduler, safety_checker=None, feature_extractor=None)
```

Every weight is bit-exact, so the pipeline is the one `from_pretrained` builds from the original
directory: in our test (Stable Diffusion 1.5, FP16, 20 steps, fixed seed, ROCm GPU) both wrote the same
PNG, byte for byte. On Linux and Windows, `ziptensors mount` plus the usual
`from_pretrained` on the mounted directory is the alternative.

A model directory compressed in place, each weight file replaced by a container beside it, works too.
Configs and tokenizers are read from the directory; each model's weights come from its containers,
found by the member name (the original file name), shards through `*.index.json`, a legacy `.bin` only
through `torch.load(..., weights_only=True)`. On macOS, which has no mount, this is the route:

```sh
ziptensors compress -r stable-diffusion-v1-5/
python examples/diffusers_from_container.py --per-file-dir stable-diffusion-v1-5/ "a lighthouse at sunset" out.png
```

A pipeline that takes an input image (img2img) or an image and a mask (inpainting) gets them from
`--image PATH` and `--mask PATH`; without them the example stops before loading any weight. A model must
get every tensor it keeps. A checkpoint tensor the model does not keep is accepted only where Diffusers
and Transformers accept it too (a non-persistent buffer such as CLIP's `position_ids`, or a name the
model class ignores on loading), and is logged; any other is an error.

```sh
ziptensors compress -r sdxl-inpainting/
python examples/diffusers_from_container.py --per-file-dir sdxl-inpainting/ --variant fp16 \
    "a red apple on a table" out.png --image photo.png --mask mask.png
```

## Modes

| Mode | What it does | Use it for |
|---|---|---|
| `fast` | Byte-plane split and entropy coding of floats; quantized data is stored as is | Float-only models when encode speed matters |
| `balanced` | Adds widening views (F32 holding F16/BF16 values) and field split of quantized blocks | Default of `encode`; needed for GGUF quantized models |
| `max` | Adds a bit-rotation trial per blob and deduplication across members; enables delta blobs | Default of `pack`; best size |

A mode changes only how data is stored, never what a restore produces, and any reader reads any mode.
Quantized GGUF (Q8_0, K-quants) needs `balanced` or `max`: in `fast` it is stored uncompressed, and plain
zstd beats it (Q8_0: 3.0 % saved in `fast` against 5.8 % for `zstd -19`).

## Results

Sizes in GB (10^9 bytes), percent saved in brackets. Each file was compressed as a whole; every output
restored bit-exact (sha256). Two runs gave identical sizes.

| Data | Source | `zstd -19 --long=27` | `zstd -3`, 16 MiB frames | ziptensors `max` |
|---|---:|---:|---:|---:|
| BF16, Qwen-Image diffusion transformer (safetensors) | 14.230 | 10.821 (24.0 %) | 11.130 (21.8 %) | **9.490 (33.3 %)** |
| FP16, SDXL inpainting UNet (safetensors) | 5.135 | 4.718 (8.1 %) | 4.719 (8.1 %) | **4.336 (15.6 %)** |
| F32, SDXL VAE (safetensors) | 0.335 | 0.311 (7.2 %) | 0.311 (7.2 %) | **0.279 (16.5 %)** |
| FP8 E4M3 with row scales, diffusion transformer (safetensors) | 6.271 | 5.088 (18.9 %) | 5.123 (18.3 %) | **4.952 (21.0 %)** |
| Q8_0, FLUX.2 klein 4B (GGUF) | 4.301 | 4.051 (5.8 %) | 4.061 (5.6 %) | **3.936 (8.5 %)** |
| Q4_K_M, FLUX.2 klein 4B (GGUF) | 2.604 | 2.477 (4.9 %) | 2.485 (4.6 %) | **2.397 (8.0 %)** |

The 16 MiB-frame column approximates a container that compresses tensors with zstd in independent
frames. Peak memory while encoding was 39 to 93 MiB for ziptensors in every mode, against 2.6 to 2.8 GiB
for `zstd -19 --long=27` on the files above 2 GB.

**Many files.** A private library of 570 model files (247 GB: BF16, F16, F32, FP8, GGUF quantized,
ONNX, PyTorch) packed in `max` into 167 GB, 32.4 % saved, and all 570 files restored bit-exact.
A complete SDXL inpainting diffusers directory with F32 and FP16 variants of its models (52 files,
20.8 GB) packed into 6.1 GB, since the F32 files share storage with their FP16 siblings.

**Delta.** FLUX.2 klein 9B and a community full finetune of it (BF16, 18.16 GB each): after
deduplication of the 109 identical tensors, the finetune stores in 9.28 GB on its own and in 7.09 GB
with `--delta-base`. Reading a delta member also reads its base, so a full read from a hard disk was
2 to 3 times slower in that test.

**Decode speed** (container already in memory, so no disk is involved; synthetic BF16 weights, 2.1 GB,
`fast` container; `max` decodes at 4.0 GB/s on one thread of the first machine):

| Machine | 1 thread | 12 threads |
|---|---:|---:|
| AMD Ryzen 5 9600X (6 cores, 12 threads), Linux | 4.3 GB/s | 7.8 GB/s |
| AMD Ryzen 5 4600H laptop (6 cores, 12 threads), Windows 11 | 2.1 GB/s | 3.6 GB/s |

Read memory stays below 3 × threads × chunk above the destination buffer for plain members (36 MiB at
12 threads and 1 MiB chunks).
All timings were taken on shared machines under other load; treat them as indicative. Details and raw
numbers: [`dev-docs/benchmarks/`](dev-docs/benchmarks/).

## Mount

`ziptensors mount` shows every member as a read-only file, so programs that only take a path
(diffusers `from_pretrained`, llama.cpp, safetensors with mmap; all checked bit-exact through the
mount) read a container unmodified. Decoded
data is cached in memory (`--cache-size`, default 256 MiB). The command runs in the foreground; Ctrl-C
unmounts. From Python, `ziptensors.mount(container, mountpoint)` and
`ziptensors.mount_overlay(directory, mountpoint)` mount with the same defaults and return a context
manager that unmounts when its `with` block ends.

- **Linux:** FUSE through the kernel's `/dev/fuse`; no libfuse is needed, only `fusermount3` on the
  `PATH` (usually from the `fuse3` package).
- **Windows:** needs WinFsp, installed from <https://winfsp.dev/rel/>. The mount point is a free
  drive letter (`X:`) or a directory that does not exist yet.

  WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos,
  <https://github.com/winfsp/winfsp>. ziptensors uses the WinFsp DLL under its FLOSS exception and
  distributes no WinFsp file.
- **macOS:** no mount.

## Platforms

Linux and Windows are built and tested, Windows natively as well (decode timing, mount). macOS is
built by cross-compilation only and has not been run natively yet; reports are welcome. Python wheels are abi3 for CPython 3.11 and later.

## Security

Treat a container from elsewhere as untrusted input. The reader validates every length and fails with
a typed error on anything malformed or unknown, but by default it sets **no limit** on how much a
container may decode. Set one when opening a file you did not make: `--max-total-decoded BYTES` on the
command line, `max_total_decoded=` in Python, `OpenOptions::max_total_decoded` in Rust.

The hashes (XXH3 per chunk, BLAKE3 per file) detect damage. They do not authenticate a file: there is
no signature, so whoever can change a container can recompute every hash. Check where a container
comes from the way you would for the original model. To report a vulnerability, see
[SECURITY.md](SECURITY.md).

## Related projects

- **[ZipNN](https://github.com/zipnn/zipnn)** (IBM and collaborators) is a Python/C library for lossless
  compression of model weights, built on byte grouping and Huffman coding of the exponent bytes of
  floating-point weights. ziptensors applies the same idea (byte planes, entropy-only coding of the
  high plane) and differs in what it is built around: random access to any tensor or byte range, a
  container of many files with deduplication and delta blobs, quantized GGUF support, a Rust core with
  C and Python bindings, and a filesystem mount. We have not benchmarked the two against each other.
- **[zTensor](https://github.com/pie-project/ztensor)** is a tensor file format with optional zstd
  compression per tensor. It is a new file format for tensors rather than specialised compression of
  existing files; the 16 MiB-frame zstd column above approximates that kind of compression (zTensor
  itself was not run).

## Format and stability

The container format is specified in [`spec/ziptensors-format.md`](spec/ziptensors-format.md). The
current format version is **0.3**, the first public one. A file written by any published version stays
readable by every later reader; a file using something an older reader does not know makes that reader
fail with an error, never with wrong bytes. A file is recognised by its magic bytes; the `.ziptensors`
extension is a convention.

A container can be changed by appending: new data, then a new index and footer, follow the old end of
the file, and the old index stays inside it as unreferenced bytes (`ziptensors edit --fast`). An
interrupted append leaves the previous version intact below a partial tail. A reader never falls back
to that version on its own: such a file fails to open until `ziptensors recover` shortens it to its
last valid version, after decoding every member to check it. The default edit rewrites the file
through a temporary one instead, and `--risky` rewrites it in place, which an interruption damages.
None of this changes the format version.

The format version is independent of the package versions (crates and the Python package start at
0.1.0). Below 1.0, a minor release of the packages may change their programming interfaces.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT)
at your option. Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in this project shall be dual licensed as above, without any additional terms or conditions.
