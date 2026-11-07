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
is decoded and compared with its source before it replaces anything at the output path.

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

`Member.read_at(offset, buffer)` and `Member.decode_tensor_into(name, buffer)` fill any writable buffer
(a `bytearray`, a numpy array) with the GIL released. `Member.open()` returns a seekable read-only file,
for example for `torch.load`.

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

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
