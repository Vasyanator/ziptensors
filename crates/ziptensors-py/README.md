# ziptensors

Lossless, compressed containers for tensor files (safetensors, GGUF, ONNX, PyTorch zip and opaque files)
with random access at tensor speed. Every restored file is bit-exact and proven by its BLAKE3 hash. This
is the Python module of [ziptensors](https://github.com/Vasyanator/ziptensors); the command line tool and
the Rust library live in the same repository.

## Install

```text
pip install ziptensors            # the module
pip install "ziptensors[torch]"   # with the PyTorch loader's dependency
```

Wheels are abi3 for CPython 3.11 and later on Linux (x86_64, aarch64), Windows (x64) and macOS. Other
platforms build from the source distribution, which needs a Rust toolchain and a C compiler.

## Write

```python
import ziptensors

report = ziptensors.encode_file("model.safetensors", "model.ziptensors")  # one file
print(report.output_len, report.total.stored)

# Several files; a (path, member name) pair names the member, a bare path uses the file name.
inputs = ["unet.safetensors", ("downloads/vae.safetensors", "vae/model.safetensors")]
ziptensors.pack(inputs, "pipeline.ziptensors")
```

`mode` is `fast`, `balanced` (the default of `encode_file`) or `max` (the default of `pack`). A new file
is decoded and compared with its source before it replaces anything at the output path. A directory
given to `pack` is walked: its files are named relative to it, and symbolic links and `*.incomplete`
downloads are skipped (`ziptensors.collect_inputs` shows what a walk takes). `ziptensors.PackBuilder`
adds members one by one and is a context manager: leaving it without `finish()` leaves nothing behind.

## Edit, recover, whole trees

```python
from ziptensors import AddOp, RemoveOp, ReplaceOp

ziptensors.edit("pipeline.ziptensors", [AddOp("lora.safetensors"), RemoveOp("old.safetensors")])
ziptensors.edit("pipeline.ziptensors", [ReplaceOp("vae/model.safetensors", "vae2.safetensors")],
                strategy="append")       # writes only the new data after the end
ziptensors.compact("pipeline.ziptensors")  # drops what earlier appends left behind
ziptensors.recover("pipeline.ziptensors")  # after an interrupted append: back to the last valid state

report = ziptensors.compress_tree("models/")   # each weight file becomes a .ziptensors beside it
ziptensors.decompress_tree("models/")          # and back, bit-exact
```

`edit` proves the result before it replaces anything. `strategy="in_place"` needs no extra disk space
but is risky: an interruption damages the file, so keep a backup. A tree run reports one outcome per
file and goes on past a failure; check `report.has_failures`. `ziptensors.enable_stderr_logging("info")`
shows the library's progress and errors on stderr.

## Read

```python
import ziptensors

container = ziptensors.Container("model.ziptensors", max_total_decoded=64 << 30)
member = container.member()                      # the only member; or container.member("name")
head = member.read(0, 4096)
for tensor in member.tensors:
    print(tensor.name, tensor.dtype, tensor.shape)
member.restore_file("model.safetensors")         # bit-exact copy of the original
```

`Member.read_at(offset, buffer)`, `Member.read_many([(offset, buffer), ...])` and
`Member.decode_tensor_into(name, buffer)` fill any writable buffer (a `bytearray`, a numpy array) with
the GIL released. `Member.open()` returns a seekable read-only file, for example for `torch.load`.
`Container` is a context manager, `container.restore_all("dir")` restores every member, and
`container.layer_report()` tells where the bytes went.

**Untrusted input:** `max_total_decoded` has no default limit. Set it whenever a container comes from
somewhere else. The hashes detect damage; they do not authenticate a file.

## PyTorch

```python
import torch
from ziptensors.torch import load_state_dict

state = load_state_dict(container.member(), device="cuda", dtype=torch.float16)
```

Tensors are decoded in place into new torch tensors; on a CUDA or ROCm device the copy overlaps with
decoding through pinned staging buffers. `ziptensors.torch.load_into(module, member)` fills a module
built on the `meta` device. GGUF block-quantized tensors are refused by default; `quantized="raw"` or
`quantized="diffusers"` hands them over undecoded.

## safetensors API

```python
from ziptensors.safetensors import safe_open

with safe_open("model.ziptensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
    rows = f.get_slice("embed.weight")[0:16]
```

`safe_open` reads a safetensors member the way `safetensors.safe_open` reads a file, with the
frameworks `pt` and `np`.

## Mount (Linux and Windows)

```python
from diffusers import StableDiffusionPipeline

with ziptensors.mount("pipeline.ziptensors", "/mnt/sd15") as mounted:
    pipe = StableDiffusionPipeline.from_pretrained(mounted.mountpoint)   # any program that takes a path

with ziptensors.mount_overlay("models/", "/mnt/models"):   # a compress_tree tree, originals restored
    ...
```

Every member shows as a read-only file; an overlay shows each per-file container as its original
file and every other file as it is. Leaving the `with` block unmounts, also after an exception or
Ctrl-C; close files opened under the mount first, since the unmount waits for them. Linux needs
`/dev/fuse` and `fusermount3` (package `fuse3`), no root; the mount point is an existing directory.
Windows needs WinFsp, installed from <https://winfsp.dev/rel/>; the mount point is a free drive
letter (`X:`) or a directory that does not exist yet. macOS has no mount: the calls raise
`ziptensors.MountError`.

WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos,
<https://github.com/winfsp/winfsp>. ziptensors uses the WinFsp DLL under its FLOSS exception and
distributes no WinFsp file.

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
