"""File: bench_tree.py
Purpose: the generated tree of the dry run: two small roots holding each kind of data the real set has
for the benchmark's sake, from a seeded PRNG.

Main responsibilities: minimal safetensors and GGUF writers, and `build_tree`, which lays out BF16, a
finetune sharing part of it, an F16 file and its widened F32 sibling, a Q8_0 GGUF, opaque files, a link
and a partial download.

Key structures: `Tree`.

Key functions: `build_tree`.

Implementation notes: these files are fixtures and carry no meaning. They only have to reach each code
path: dedup, widening, fragmentation, skips. Sizes keep the whole dry run to seconds.
"""

import json
import random
import struct
from dataclasses import dataclass
from pathlib import Path

MIB = 1 << 20
GGUF_ALIGNMENT = 32
GGML_TYPE_Q8_0 = 8
Q8_0_BLOCK_ELEMENTS = 32
Q8_0_BLOCK_BYTES = 34
# High bytes of plausible BF16 values, so that a plane split has something to find.
BF16_HIGH_BYTES = bytes(0x3C + (value & 3) + (value & 0x80) for value in range(256))
OPAQUE_FILES = 10


@dataclass(frozen=True)
class Tree:
    """The generated roots and the member names of the load targets."""

    root_a: Path
    root_b: Path
    files: int
    bf16: str
    shared_f32: str
    q8: str
    fragmented: str


def safetensors_bytes(tensors: dict[str, tuple[str, list[int], bytes]]) -> bytes:
    """Return a safetensors file of `tensors`, each `name -> (dtype, shape, data)`."""
    header = {}
    offset = 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    text = json.dumps(header).encode()
    return struct.pack("<Q", len(text)) + text + b"".join(item[2] for item in tensors.values())


def gguf_bytes(tensors: dict[str, bytes]) -> bytes:
    """Return a GGUF v3 file of Q8_0 `tensors`, one-dimensional, with no key-value pairs."""
    infos = bytearray()
    offset = 0
    for name, data in tensors.items():
        encoded = name.encode()
        elements = len(data) // Q8_0_BLOCK_BYTES * Q8_0_BLOCK_ELEMENTS
        infos += struct.pack("<Q", len(encoded)) + encoded
        infos += struct.pack("<IQIQ", 1, elements, GGML_TYPE_Q8_0, offset)
        offset += (len(data) + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT * GGUF_ALIGNMENT
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 0) + infos
    body = bytearray(head + bytes(-len(head) % GGUF_ALIGNMENT))
    for data in tensors.values():
        body += data + bytes(-len(data) % GGUF_ALIGNMENT)
    return bytes(body)


def bf16_data(rng: random.Random, elements: int) -> bytes:
    """Return `elements` plausible BF16 values, little-endian."""
    data = bytearray(elements * 2)
    data[0::2] = rng.randbytes(elements)
    data[1::2] = rng.randbytes(elements).translate(BF16_HIGH_BYTES)
    return bytes(data)


def f16_values(rng: random.Random, elements: int) -> tuple[float, ...]:
    """Return `elements` values exact in binary16, so that their F32 form narrows back."""
    data = struct.pack(f"<{elements}e", *(rng.uniform(-1.0, 1.0) for _ in range(elements)))
    return struct.unpack(f"<{elements}e", data)


def write(path: Path, data: bytes) -> None:
    """Write `data` to `path`, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def build_tree(base: Path) -> Tree:
    """Generate both roots under `base`, which must exist, and return where everything is."""
    rng = random.Random(20261102)
    root_a, root_b = base / "root-a", base / "root-b"
    tensor_elements = MIB // 2 + 1021

    layers = {f"layer.{n}": bf16_data(rng, tensor_elements) for n in range(6)}
    bf16 = "base/model.safetensors"
    shape = [tensor_elements]
    write(root_a / bf16, safetensors_bytes({k: ("BF16", shape, v) for k, v in layers.items()}))
    tuned = {
        name: data if number % 2 == 0 else bf16_data(rng, tensor_elements)
        for number, (name, data) in enumerate(layers.items())
    }
    # Sorts after its base, so that the shared tensors are stored there and this member is the fragmented one.
    fragmented = "base/tuned.safetensors"
    write(root_a / fragmented, safetensors_bytes({k: ("BF16", shape, v) for k, v in tuned.items()}))

    small = 200_000
    values = {f"unet.{n}": f16_values(rng, small) for n in range(3)}
    narrow = {k: ("F16", [small], struct.pack(f"<{small}e", *v)) for k, v in values.items()}
    wide = {k: ("F32", [small], struct.pack(f"<{small}f", *v)) for k, v in values.items()}
    write(root_a / "sdxl/unet.fp16.safetensors", safetensors_bytes(narrow))
    shared_f32 = "sdxl/unet.safetensors"
    write(root_a / shared_f32, safetensors_bytes(wide))

    blocks = 40_000
    q8 = "gguf/model-Q8_0.gguf"
    quantized = {f"blk.{n}": rng.randbytes(blocks * Q8_0_BLOCK_BYTES) for n in range(3)}
    write(root_b / q8, gguf_bytes(quantized))

    opaque = rng.randbytes(4096) * 64
    write(root_b / "misc/notes.bin", opaque)
    write(root_b / "other/notes-copy.bin", opaque)
    write(root_b / "misc/config.json", b'{"name": "generated"}')
    write(root_b / "misc/торт.bin", b"not utf-8 safe in latin-1\n")
    for number in range(OPAQUE_FILES):
        write(root_b / f"extra/file-{number}.bin", rng.randbytes(1024 * (number + 1)))

    files = sum(1 for root in (root_a, root_b) for path in root.rglob("*") if path.is_file())
    write(root_b / "misc/partial.safetensors.incomplete", b"partial")
    (root_b / "misc/link.safetensors").symlink_to(root_a / bf16)
    return Tree(root_a, root_b, files, bf16, shared_f32, q8, fragmented)
