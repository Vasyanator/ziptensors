"""File: tests/test_torch_integration.py
Purpose: contract tests of `Member.open` and `ziptensors.torch`, and proofs against the real libraries.

Main responsibilities: the file object against `io.BytesIO`, every accepted dtype, every refusal, and
loading through real Transformers and Diffusers models with forward passes compared exactly.

Key functions: `encode`, `make_gguf_tensors`, `ggml_blocks`, `raw_bytes`, `tiny_clip`, `tiny_gpt2`,
`tiny_unet`, `tiny_vae`, `tiny_tokenizer`.

Implementation notes: no third-party object is patched, except that one test hides one torch dtype
through `monkeypatch` for its duration. Models are built from a config with a fixed seed, so nothing
is downloaded, and `HF_HUB_OFFLINE` is set below before any library reads it. Tests of a library skip
when it is absent, and GPU tests without `torch.cuda.is_available()`. The snippets of the section on
PyTorch in `src/MODULE_README.md` are copied from these tests. All scratch data lives under
`tmp_path`.
"""

from __future__ import annotations

import ctypes
import io
import json
import logging
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
from typing import Any

import pytest

from test_ziptensors import (  # noqa: F401 - `fixture_slow` is a fixture, used by its name
    HEADER_LEN,
    damaged,
    fixture_slow,
    make_safetensors,
    seeded_bytes,
    stall_during,
)
import ziptensors
from ziptensors import Container, CorruptionError, Member, NotFoundError
from ziptensors.torch import GgmlTensor, TorchLoadError, load_into, load_state_dict

# Read by huggingface_hub at import, which no earlier test triggers. Nothing here may use the network.
os.environ["HF_HUB_OFFLINE"] = "1"

GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
# A type id ggml retired: the core keeps no table row for it.
GGML_RETIRED = 4
GGML_Q5_1 = 7
GGML_Q8_0 = 8
GGML_Q4_K = 12
GGML_MXFP4 = 39
GGML_I32 = 26
GGML_F64 = 28
GGML_BF16 = 30


def encode(path: Path) -> Member:
    """Encodes the file at `path` next to it and returns its member."""
    output = path.with_suffix(".ziptensors")
    ziptensors.encode_file(path, output, threads=1)
    return Container(output, threads=2).member()


def make_gguf_tensors(path: Path, tensors: list[tuple[str, list[int], int, bytes]]) -> Path:
    """Writes a GGUF v3 file of (name, shape in file order, ggml type, data) and returns `path`."""
    infos = b""
    offset = 0
    for name, shape, ggml_type, data in tensors:
        encoded = name.encode()
        infos += struct.pack("<Q", len(encoded)) + encoded + struct.pack("<I", len(shape))
        infos += b"".join(struct.pack("<Q", dim) for dim in shape)
        infos += struct.pack("<IQ", ggml_type, offset)
        offset += len(data) + (-len(data) % 32)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 0) + infos
    payload = b"".join(data + bytes(-len(data) % 32) for _, _, _, data in tensors)
    path.write_bytes(head + bytes(-len(head) % 32) + payload)
    return path


def raw_bytes(tensor: Any) -> bytes:
    """Returns the storage of a contiguous CPU tensor. Works for bfloat16, unlike `numpy()`."""
    return ctypes.string_at(tensor.data_ptr(), tensor.nbytes)


@pytest.fixture(name="torch")
def fixture_torch() -> Any:
    """The torch module, or a skip."""
    return pytest.importorskip("torch")


@pytest.fixture(name="member_bytes")
def fixture_member_bytes(tmp_path: Path) -> tuple[Member, bytes]:
    """A member and its exact content."""
    source = make_safetensors(
        tmp_path / "file.safetensors",
        {"noise": ("F32", [4096], seeded_bytes(21, 4096 * 4))},
    )
    return encode(source), source.read_bytes()


def test_member_profile(member_bytes: tuple[Member, bytes], tmp_path: Path) -> None:
    assert member_bytes[0].profile == "safetensors"
    plain = tmp_path / "plain.bin"
    plain.write_bytes(seeded_bytes(22, 4096))
    assert encode(plain).profile == "opaque"


def test_open_is_a_read_only_seekable_raw_file(member_bytes: tuple[Member, bytes]) -> None:
    file = member_bytes[0].open()
    assert isinstance(file, io.RawIOBase)
    assert file.readable() and file.seekable() and not file.writable()
    with pytest.raises(io.UnsupportedOperation):
        file.write(b"x")


def test_open_follows_bytesio_through_random_operations(member_bytes: tuple[Member, bytes]) -> None:
    member, content = member_bytes
    file = member.open()
    model = io.BytesIO(content)
    rng = random.Random(23)
    for _ in range(2000):
        offset = rng.randrange(0, len(content) + 64)
        size = rng.choice((0, 1, 7, 4096, len(content) * 2))
        match rng.randrange(6):
            case 0:
                assert file.seek(offset) == model.seek(offset)
            case 1:
                assert file.seek(-offset % 64, os.SEEK_CUR) == model.seek(-offset % 64, os.SEEK_CUR)
            case 2:
                assert file.seek(-16, os.SEEK_END) == model.seek(-16, os.SEEK_END)
            case 3:
                assert file.read(size) == model.read(size)
            case 4:
                mine, theirs = bytearray(size), bytearray(size)
                assert file.readinto(mine) == model.readinto(theirs)
                assert mine == theirs
            case _:
                assert file.tell() == model.tell()
    file.seek(5)
    model.seek(5)
    assert file.read() == model.read()
    assert file.read() == b""


def test_open_readinto_takes_buffers_of_any_format(member_bytes: tuple[Member, bytes]) -> None:
    numpy = pytest.importorskip("numpy")
    member, content = member_bytes
    target = numpy.empty(16, dtype=numpy.float32)
    assert member.open().readinto(target) == 64
    assert target.tobytes() == content[:64]


def test_open_refuses_bad_seeks_and_closed_use(member_bytes: tuple[Member, bytes]) -> None:
    file = member_bytes[0].open()
    with pytest.raises(ValueError, match="before the start"):
        file.seek(-1)
    with pytest.raises(ValueError, match="whence"):
        file.seek(0, 7)
    file.close()
    for call in (lambda: file.read(1), lambda: file.readinto(bytearray(1)), lambda: file.seek(0)):
        with pytest.raises(ValueError, match="closed"):
            call()


def test_open_works_under_a_buffered_reader(member_bytes: tuple[Member, bytes]) -> None:
    member, content = member_bytes
    with io.BufferedReader(member.open()) as file:
        assert file.read(100) == content[:100]
        assert file.read() == content[100:]


def test_open_positions_are_independent(member_bytes: tuple[Member, bytes]) -> None:
    member, content = member_bytes
    first, second = member.open(), member.open()
    first.read(10)
    assert second.read(10) == content[:10]


def test_open_surfaces_corruption(tmp_path: Path) -> None:
    source = make_safetensors(
        tmp_path / "noise.safetensors",
        {"noise": ("F32", [1 << 16], seeded_bytes(6, 1 << 18))},
    )
    output = tmp_path / "noise.ziptensors"
    ziptensors.encode_file(source, output, threads=1)
    file = Container(damaged(output, HEADER_LEN + (1 << 17))).member().open()
    with pytest.raises(CorruptionError):
        file.read()


def test_importing_the_module_does_not_import_torch() -> None:
    code = "import sys, ziptensors.torch; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_missing_torch_is_a_clear_error() -> None:
    code = (
        "import sys, ziptensors.torch\n"
        "sys.modules['torch'] = None\n"
        "try:\n"
        "    ziptensors.torch.load_state_dict(None)\n"
        "except ImportError as error:\n"
        "    sys.exit('install torch' not in str(error))\n"
        "sys.exit(2)\n"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


SAFETENSORS_DTYPES = [
    ("F64", "float64"),
    ("F32", "float32"),
    ("F16", "float16"),
    ("BF16", "bfloat16"),
    ("I64", "int64"),
    ("I32", "int32"),
    ("I16", "int16"),
    ("I8", "int8"),
    ("U8", "uint8"),
    ("U16", "uint16"),
    ("U32", "uint32"),
    ("U64", "uint64"),
    ("F8_E4M3", "float8_e4m3fn"),
    ("F8_E5M2", "float8_e5m2"),
    ("BOOL", "bool"),
]


@pytest.mark.parametrize(("stored", "name"), SAFETENSORS_DTYPES)
def test_every_safetensors_dtype_roundtrips(
    torch: Any, tmp_path: Path, stored: str, name: str
) -> None:
    dtype = getattr(torch, name)
    expected = (torch.arange(48) % 2).to(dtype).reshape(2, 4, 6)
    source = make_safetensors(
        tmp_path / "typed.safetensors",
        {
            "tensor": (stored, [2, 4, 6], raw_bytes(expected)),
            "scalar": (stored, [], raw_bytes(expected[0, 0, 1])),
            "empty": (stored, [0, 3], b""),
        },
    )
    state = load_state_dict(encode(source))
    assert list(state) == ["tensor", "scalar", "empty"]
    assert state["tensor"].dtype == dtype
    # Bytes, not `torch.equal`, which torch does not implement for float8.
    assert state["tensor"].shape == expected.shape
    assert raw_bytes(state["tensor"]) == raw_bytes(expected)
    assert state["scalar"].shape == ()
    assert state["scalar"].item() == expected[0, 0, 1].item()
    assert state["empty"].shape == (0, 3)


def test_tensors_are_writable_and_own_their_memory(torch: Any, tmp_path: Path) -> None:
    data = seeded_bytes(24, 64 * 4)
    source = make_safetensors(tmp_path / "own.safetensors", {"own": ("F32", [64], data)})
    member = encode(source)
    first = load_state_dict(member)["own"]
    assert raw_bytes(first) == data
    first.zero_()
    assert raw_bytes(load_state_dict(member)["own"]) == data
    del member
    assert raw_bytes(first) == bytes(len(data))


@pytest.mark.parametrize(
    ("ggml_type", "name", "item_bytes"),
    [
        (GGML_F32, "float32", 4),
        (GGML_F16, "float16", 2),
        (GGML_F64, "float64", 8),
        (GGML_BF16, "bfloat16", 2),
    ],
)
def test_gguf_float_shape_is_reversed_and_bytes_stay(
    torch: Any, tmp_path: Path, ggml_type: int, name: str, item_bytes: int
) -> None:
    data = seeded_bytes(25, 6 * item_bytes)
    source = make_gguf_tensors(tmp_path / "model.gguf", [("weight", [3, 2], ggml_type, data)])
    member = encode(source)
    assert member.tensor_info("weight").shape == [3, 2]
    tensor = load_state_dict(member)["weight"]
    assert tensor.shape == (2, 3)
    assert tensor.dtype == getattr(torch, name)
    assert raw_bytes(tensor) == data


def test_gguf_rows_are_the_fastest_varying_dimension(torch: Any, tmp_path: Path) -> None:
    """A ggml [3, 2] is two rows of three in file order: the writer's array of shape (2, 3)."""
    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    source = make_gguf_tensors(tmp_path / "m.gguf", [("w", [3, 2], GGML_F32, raw_bytes(expected))])
    assert torch.equal(load_state_dict(encode(source))["w"], expected)


@pytest.mark.parametrize(
    ("ggml_type", "shape", "length", "dtype"),
    [(GGML_Q8_0, [32], 34, "Q8_0"), (GGML_I32, [4], 16, "I32")],
)
def test_gguf_quantized_and_integer_tensors_refused(
    torch: Any, tmp_path: Path, ggml_type: int, shape: list[int], length: int, dtype: str
) -> None:
    source = make_gguf_tensors(
        tmp_path / "model.gguf",
        [("kept", [4], GGML_F32, bytes(16)), ("blk.0.weight", shape, ggml_type, bytes(length))],
    )
    with pytest.raises(TorchLoadError, match=f"'blk.0.weight'.*{dtype}.*reinterpreting"):
        load_state_dict(encode(source))


def test_gguf_tensor_without_a_table_row_is_not_dropped_silently(
    torch: Any, tmp_path: Path
) -> None:
    source = make_gguf_tensors(
        tmp_path / "model.gguf",
        [("kept", [4], GGML_F32, bytes(16)), ("lost", [32], GGML_RETIRED, bytes(20))],
    )
    member = encode(source)
    assert [tensor.name for tensor in member.tensors] == ["kept"]
    with pytest.raises(TorchLoadError, match="declares 2 tensors and the table lists 1"):
        load_state_dict(member)
    assert list(load_state_dict(member, keys=["kept"])) == ["kept"]


@pytest.mark.parametrize("dtype", ["C64", "F8_E4M3FNUZ", "F8_E8M0", "F4", "Q8"])
def test_safetensors_dtypes_outside_the_list_refused(
    torch: Any, tmp_path: Path, dtype: str
) -> None:
    source = make_safetensors(tmp_path / "s.safetensors", {"odd": (dtype, [8], bytes(64))})
    with pytest.raises(TorchLoadError, match="'odd'"):
        load_state_dict(encode(source))


def test_dtype_missing_from_torch_is_a_clear_error(
    torch: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_safetensors(tmp_path / "s.safetensors", {"fp8": ("F8_E5M2", [4], bytes(4))})
    member = encode(source)
    monkeypatch.delattr(torch, "float8_e5m2")
    with pytest.raises(TorchLoadError, match="'fp8'.*has no torch.float8_e5m2"):
        load_state_dict(member)


def test_float8_converts_exactly_through_dtype(torch: Any, tmp_path: Path) -> None:
    stored = torch.linspace(-4, 4, 17).to(torch.float8_e4m3fn)
    source = make_safetensors(
        tmp_path / "f8.safetensors", {"w": ("F8_E4M3", [17], raw_bytes(stored))}
    )
    state = load_state_dict(encode(source), dtype=torch.float32)
    assert torch.equal(state["w"], stored.to(torch.float32))


def ggml_blocks(seed: int, ggml_type: int, count: int) -> bytes:
    """Returns `count` seeded blocks of a ggml type whose f16 scales are finite.

    Each layout starts with its f16 scales: Q4_0 (d, qs: 16), 18 bytes; Q5_1 (d, m, qh: 4, qs: 16),
    24 bytes; Q8_0 (d, qs: 32 x i8), 34 bytes; Q4_K (d, dmin, scales: 12, qs: 128), 144 bytes. MXFP4
    (e: u8 exponent, qs: 16), 17 bytes, has no f16 scale. Random scales could be NaN or infinite, so
    they are set to small values.
    """
    size, scales = {
        GGML_Q4_0: (18, 1),
        GGML_Q5_1: (24, 2),
        GGML_Q8_0: (34, 1),
        GGML_Q4_K: (144, 2),
        GGML_MXFP4: (17, 0),
    }[ggml_type]
    data = bytearray(seeded_bytes(seed, size * count))
    for block in range(count):
        for index in range(scales):
            position = block * size + 2 * index
            data[position : position + 2] = struct.pack("<e", 0.001 * (block + index + 1))
    return bytes(data)


# ggml type, its registry name, the file shape (ggml order), the torch byte shape, its block count.
QUANTIZED_CASES = [
    (GGML_Q4_0, "Q4_0", [64, 3], (3, 36), 6),
    (GGML_Q5_1, "Q5_1", [64, 2], (2, 48), 4),
    (GGML_Q8_0, "Q8_0", [64, 3], (3, 68), 6),
    (GGML_Q4_K, "Q4_K", [512, 2], (2, 288), 4),
]


def quantized_member(tmp_path: Path, ggml_type: int, shape: list[int], blocks: int) -> Member:
    """A GGUF member holding a float tensor `bias` and the seeded quantized tensor `blk.0.weight`."""
    source = make_gguf_tensors(
        tmp_path / "quantized.gguf",
        [
            ("bias", [4], GGML_F32, seeded_bytes(51, 16)),
            ("blk.0.weight", shape, ggml_type, ggml_blocks(52, ggml_type, blocks)),
        ],
    )
    return encode(source)


@pytest.mark.parametrize(("ggml_type", "name", "shape", "byte_shape", "blocks"), QUANTIZED_CASES)
def test_gguf_quantized_raw_returns_the_blocks(
    torch: Any,
    tmp_path: Path,
    ggml_type: int,
    name: str,
    shape: list[int],
    byte_shape: tuple[int, ...],
    blocks: int,
) -> None:
    member = quantized_member(tmp_path, ggml_type, shape, blocks)
    state = load_state_dict(member, quantized="raw", dtype=torch.float16)
    weight = state["blk.0.weight"]
    assert isinstance(weight, GgmlTensor)
    assert weight.ggml_type == name
    assert weight.shape == tuple(reversed(shape))
    assert weight.data.dtype == torch.uint8
    assert tuple(weight.data.shape) == byte_shape
    assert raw_bytes(weight.data) == ggml_blocks(52, ggml_type, blocks)
    # The float tensor beside it is a plain tensor, and only it takes `dtype`.
    assert state["bias"].dtype == torch.float16


def test_gguf_quantized_refused_by_default_with_a_hint(torch: Any, tmp_path: Path) -> None:
    member = quantized_member(tmp_path, GGML_Q8_0, [64, 3], 6)
    with pytest.raises(TorchLoadError, match="'blk.0.weight'.*quantized='raw'"):
        load_state_dict(member)
    with torch.device("meta"):
        module = torch.nn.Linear(64, 3, bias=False)
    with pytest.raises(TorchLoadError, match="Q8_0"):
        load_into(module, member)


def test_quantized_mode_is_checked(torch: Any, tmp_path: Path) -> None:
    member = quantized_member(tmp_path, GGML_Q8_0, [64, 3], 6)
    with pytest.raises(ValueError, match="`quantized` is one of"):
        load_state_dict(member, quantized="dequantize")  # type: ignore[arg-type]
    source = make_safetensors(tmp_path / "s.safetensors", {"w": ("F32", [4], bytes(16))})
    with pytest.raises(ValueError, match="applies to GGUF members"):
        load_state_dict(encode(source), quantized="raw")


def test_quantized_row_that_contradicts_its_blocks_refused(torch: Any) -> None:
    """The core makes no such row from a file, so the plan is fed a hostile table directly."""
    from types import SimpleNamespace

    from ziptensors.torch import _plan_tensor

    member = SimpleNamespace(name="m", profile="gguf", len=1 << 20)

    def row(shape: list[int], length: int) -> Any:
        return SimpleNamespace(
            name="w", dtype="Q8_0", shape=shape, offset=0, len=length, block=(32, 34)
        )

    with pytest.raises(TorchLoadError, match="not whole blocks of 32"):
        _plan_tensor(torch, member, row([48, 2], 102), "raw")
    with pytest.raises(TorchLoadError, match="needs 68 bytes and the table says 70"):
        _plan_tensor(torch, member, row([64], 70), "raw")
    with pytest.raises(TorchLoadError, match="no dimension"):
        _plan_tensor(torch, member, row([], 34), "raw")
    assert _plan_tensor(torch, member, row([64, 2], 136), "raw").shape == (2, 68)


def test_core_block_geometry_matches_gguf(tmp_path: Path) -> None:
    """Every ggml type of the gguf package, one block each: the core's rows carry its geometry."""
    gguf = pytest.importorskip("gguf")
    tensors = []
    for kind in gguf.GGMLQuantizationType:
        elements, block_bytes = gguf.GGML_QUANT_SIZES[kind]
        tensors.append((kind.name, [elements], kind.value, bytes(block_bytes)))
    member = encode(make_gguf_tensors(tmp_path / "every.gguf", tensors))
    rows = {tensor.name: tensor for tensor in member.tensors}
    quantized = [name for name, tensor in rows.items() if tensor.block and tensor.block[0] > 1]
    # The 24 block-quantized types of the registry; a gguf type without a row is a type the core
    # does not know, and then a whole-member load refuses the file.
    assert len(quantized) == 24, sorted(rows)
    for name, tensor in rows.items():
        assert tensor.dtype == name
        assert tensor.block == gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[name]], name


def test_quantized_types_beyond_diffusers(torch: Any, tmp_path: Path) -> None:
    pytest.importorskip("diffusers")
    pytest.importorskip("gguf")
    data = ggml_blocks(54, GGML_MXFP4, 2)
    member = encode(
        make_gguf_tensors(tmp_path / "mxfp4.gguf", [("w", [64], GGML_MXFP4, data)])
    )
    weight = load_state_dict(member, quantized="raw")["w"]
    assert isinstance(weight, GgmlTensor)
    assert (weight.ggml_type, weight.shape, tuple(weight.data.shape)) == ("MXFP4", (64,), (34,))
    assert raw_bytes(weight.data) == data
    with pytest.raises(TorchLoadError, match="diffusers does not dequantize MXFP4"):
        load_state_dict(member, quantized="diffusers")


def test_ggml_tensor_compares_by_identity(torch: Any) -> None:
    first = GgmlTensor(torch.zeros(2, dtype=torch.uint8), "Q8_0", (32,))
    assert first == first
    assert first != GgmlTensor(first.data, "Q8_0", (32,))


@pytest.mark.usefixtures("diffusers_present")
@pytest.mark.parametrize(("ggml_type", "name", "shape", "byte_shape", "blocks"), QUANTIZED_CASES)
def test_gguf_quantized_diffusers_equals_load_gguf_checkpoint(
    torch: Any,
    tmp_path: Path,
    ggml_type: int,
    name: str,
    shape: list[int],
    byte_shape: tuple[int, ...],
    blocks: int,
) -> None:
    pytest.importorskip("gguf")
    from diffusers.models.model_loading_utils import load_gguf_checkpoint
    from diffusers.quantizers.gguf.utils import GGUFParameter

    member = quantized_member(tmp_path, ggml_type, shape, blocks)
    expected = load_gguf_checkpoint(str(tmp_path / "quantized.gguf"))
    state = load_state_dict(member, quantized="diffusers")
    assert list(state) == list(expected)
    weight, reference = state["blk.0.weight"], expected["blk.0.weight"]
    assert type(weight) is GGUFParameter
    assert weight.quant_type == reference.quant_type
    assert weight.quant_type.name == name
    assert weight.shape == reference.shape == byte_shape
    assert weight.quant_shape == reference.quant_shape
    assert torch.equal(weight.as_subclass(torch.Tensor), reference.as_subclass(torch.Tensor))
    assert type(state["bias"]) is torch.Tensor
    assert torch.equal(state["bias"], expected["bias"])


def test_quantized_diffusers_without_diffusers_is_a_clear_error(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    source = make_gguf_tensors(
        tmp_path / "q.gguf", [("w", [32], GGML_Q8_0, ggml_blocks(53, GGML_Q8_0, 1))]
    )
    output = source.with_suffix(".ziptensors")
    ziptensors.encode_file(source, output, threads=1)
    code = (
        "import sys, ziptensors, ziptensors.torch\n"
        "sys.modules['diffusers'] = None\n"
        "sys.modules['gguf'] = None\n"
        "member = ziptensors.Container(sys.argv[1]).member()\n"
        "try:\n"
        "    ziptensors.torch.load_state_dict(member, quantized='diffusers')\n"
        "except ziptensors.torch.TorchLoadError as error:\n"
        "    sys.exit('diffusers and gguf' not in str(error))\n"
        "sys.exit(2)\n"
    )
    result = subprocess.run([sys.executable, "-c", code, str(output)], check=False)
    assert result.returncode == 0


def test_staging_bytes_is_checked(torch: Any, member_bytes: tuple[Member, bytes]) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        load_state_dict(member_bytes[0], staging_bytes=0)
    with pytest.raises(TypeError, match="positive integer"):
        load_state_dict(member_bytes[0], staging_bytes=1.5)  # type: ignore[arg-type]


def test_unrecognized_and_unsupported_members_refused(torch: Any, tmp_path: Path) -> None:
    plain = tmp_path / "plain.bin"
    plain.write_bytes(seeded_bytes(26, 4096))
    with pytest.raises(TorchLoadError, match="opaque member"):
        load_state_dict(encode(plain))
    pickled = tmp_path / "state.pt"
    torch.save({"weight": torch.zeros(1024)}, pickled)
    member = encode(pickled)
    assert member.profile == "pt_zip"
    with pytest.raises(TorchLoadError, match="pt_zip member"):
        load_state_dict(member)


def test_length_that_contradicts_the_shape_refused(torch: Any, tmp_path: Path) -> None:
    source = make_safetensors(tmp_path / "s.safetensors", {"wrong": ("F32", [5], bytes(16))})
    with pytest.raises(TorchLoadError, match="'wrong'.*needs 20 bytes"):
        load_state_dict(encode(source))


def test_overlapping_table_refused_before_allocating(torch: Any, tmp_path: Path) -> None:
    span = 1 << 16
    header = {
        name: {"dtype": "U8", "shape": [span], "data_offsets": [0, span]} for name in ("a", "b")
    }
    encoded = json.dumps(header).encode()
    path = tmp_path / "overlap.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + seeded_bytes(27, span))
    with pytest.raises(TorchLoadError, match="overlaps"):
        load_state_dict(encode(path))


def test_repeated_name_keeps_the_first_row_as_the_table_does(torch: Any, tmp_path: Path) -> None:
    entry = '{"dtype": "U8", "shape": [8], "data_offsets": [%d, %d]}'
    encoded = ('{"same": ' + entry % (0, 8) + ', "same": ' + entry % (8, 16) + "}").encode()
    path = tmp_path / "repeat.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(range(16)))
    assert raw_bytes(load_state_dict(encode(path))["same"]) == bytes(range(8))


def test_bool_byte_outside_zero_and_one_refused(torch: Any, tmp_path: Path) -> None:
    data = bytes([0, 1, 2, 1])
    source = make_safetensors(tmp_path / "b.safetensors", {"mask": ("BOOL", [4], data)})
    with pytest.raises(TorchLoadError, match="'mask'.*byte 2"):
        load_state_dict(encode(source))


def test_keys_select_in_order_and_unknown_key_fails(torch: Any, tmp_path: Path) -> None:
    tensors = {name: ("F32", [4], seeded_bytes(seed, 16)) for seed, name in enumerate("abc")}
    member = encode(make_safetensors(tmp_path / "k.safetensors", tensors))
    assert list(load_state_dict(member, keys=["c", "a", "c"])) == ["c", "a"]
    assert list(load_state_dict(member, keys=iter(["b"]))) == ["b"]
    assert load_state_dict(member, keys=[]) == {}
    with pytest.raises(NotFoundError):
        load_state_dict(member, keys=["a", "absent"])
    with pytest.raises(TypeError, match="not one string"):
        load_state_dict(member, keys="abc")


def test_dtype_converts_floats_and_keeps_integers(torch: Any, tmp_path: Path) -> None:
    weight = torch.linspace(0, 1, 16, dtype=torch.float32)
    index = torch.arange(1 << 20, (1 << 20) + 16, dtype=torch.int64)
    member = encode(
        make_safetensors(
            tmp_path / "d.safetensors",
            {
                "weight": ("F32", [16], raw_bytes(weight)),
                "index": ("I64", [16], raw_bytes(index)),
            },
        )
    )
    state = load_state_dict(member, dtype=torch.float16)
    assert state["weight"].dtype == torch.float16
    assert torch.equal(state["weight"], weight.to(torch.float16))
    assert torch.equal(state["index"], index)
    with pytest.raises(ValueError, match="floating point"):
        load_state_dict(member, dtype=torch.int32)


def test_meta_device_refused(torch: Any, member_bytes: tuple[Member, bytes]) -> None:
    with pytest.raises(ValueError, match="meta"):
        load_state_dict(member_bytes[0], device="meta")


def test_gpu_device(torch: Any, member_bytes: tuple[Member, bytes]) -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    host = load_state_dict(member_bytes[0])["noise"]
    moved = load_state_dict(member_bytes[0], device="cuda")["noise"]
    assert moved.device.type == "cuda"
    # Bytes, not `torch.equal`: these are random bits, and some of them are NaN.
    assert raw_bytes(moved.cpu()) == raw_bytes(host)


def skip_without_gpu(torch: Any) -> None:
    """Skips the test unless torch sees a CUDA or ROCm device."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA or ROCm device")


def every_dtype_member(tmp_path: Path) -> Member:
    """A member with one 37-element tensor per accepted safetensors dtype, plus 0-d and empty ones."""
    from ziptensors.torch import _SAFETENSORS_DTYPES

    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for seed, (stored, _) in enumerate(SAFETENSORS_DTYPES):
        data = seeded_bytes(60 + seed, 37 * _SAFETENSORS_DTYPES[stored].item_bytes)
        if stored == "BOOL":
            data = bytes(byte % 2 for byte in data)
        tensors[stored] = (stored, [37], data)
    tensors["scalar"] = ("F32", [], seeded_bytes(59, 4))
    tensors["empty"] = ("F16", [0, 5], b"")
    return encode(make_safetensors(tmp_path / "every.safetensors", tensors))


@pytest.mark.parametrize("staging_bytes", [1, 64, 1000, None])
def test_gpu_staging_equals_the_host(
    torch: Any, tmp_path: Path, staging_bytes: int | None
) -> None:
    skip_without_gpu(torch)
    member = every_dtype_member(tmp_path)
    host = load_state_dict(member)
    moved = load_state_dict(member, device="cuda", staging_bytes=staging_bytes)
    assert list(moved) == list(host)
    for name, tensor in host.items():
        assert moved[name].device.type == "cuda", name
        assert moved[name].dtype == tensor.dtype, name
        assert moved[name].shape == tensor.shape, name
        # Bytes, not `torch.equal`: these are random bits, and some of them are NaN.
        assert raw_bytes(moved[name].cpu()) == raw_bytes(tensor), name


def test_gpu_staging_converts_on_the_device(torch: Any, tmp_path: Path) -> None:
    skip_without_gpu(torch)
    weight = torch.linspace(-3, 3, 300, dtype=torch.float32)
    index = torch.arange(300, dtype=torch.int64)
    member = encode(
        make_safetensors(
            tmp_path / "d.safetensors",
            {"weight": ("F32", [300], raw_bytes(weight)), "index": ("I64", [300], raw_bytes(index))},
        )
    )
    state = load_state_dict(member, device="cuda", dtype=torch.float16, staging_bytes=128)
    assert state["weight"].dtype == torch.float16
    assert torch.equal(state["weight"].cpu(), weight.to(torch.float16))
    assert torch.equal(state["index"].cpu(), index)


def test_gpu_staging_refuses_a_bad_bool(torch: Any, tmp_path: Path) -> None:
    skip_without_gpu(torch)
    data = bytes([0, 1] * 40 + [2])
    source = make_safetensors(tmp_path / "b.safetensors", {"mask": ("BOOL", [81], data)})
    with pytest.raises(TorchLoadError, match="'mask'.*byte 2"):
        load_state_dict(encode(source), device="cuda", staging_bytes=16)


def test_gpu_staging_of_quantized_blocks(torch: Any, tmp_path: Path) -> None:
    skip_without_gpu(torch)
    member = quantized_member(tmp_path, GGML_Q4_K, [512, 2], 4)
    weight = load_state_dict(member, device="cuda", quantized="raw", staging_bytes=100)[
        "blk.0.weight"
    ]
    assert isinstance(weight, GgmlTensor)
    assert weight.data.device.type == "cuda"
    assert raw_bytes(weight.data.cpu()) == ggml_blocks(52, GGML_Q4_K, 4)


# GPU cycles of the kernel that keeps the current stream busy during a load: about a quarter of a
# second on the ROCm gfx1201 these tests ran on, far longer than the copies of the load.
BUSY_CYCLES = 500_000_000


def busy_current_stream_during_loads(torch: Any, monkeypatch: pytest.MonkeyPatch, work: Any) -> None:
    """Makes every staged load start with `work()` on another thread, queued on the default stream
    after the stager ordered its side stream behind that stream, and waits for it to be queued.

    This is what an inference or UI thread of the same process does: the current stream then runs
    behind the side stream, which is the case the stager's ordering must survive.
    """
    import threading

    import ziptensors.torch

    original = ziptensors.torch._DeviceStager.__init__

    def started(self: Any, *args: Any) -> None:
        original(self, *args)
        thread = threading.Thread(target=work)
        thread.start()
        thread.join()

    monkeypatch.setattr(ziptensors.torch._DeviceStager, "__init__", started)


def test_gpu_conversion_survives_a_busy_current_stream(
    torch: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A converted tensor's block, freed while its conversion still waits on the current stream,
    must not take the next tensor's bytes from the side stream first."""
    skip_without_gpu(torch)
    count, elements = 8, 1 << 18
    tensors = {
        f"w{index}": ("F32", [elements], raw_bytes(torch.full((elements,), float(index))))
        for index in range(count)
    }
    member = encode(make_safetensors(tmp_path / "busy.safetensors", tensors))
    busy_current_stream_during_loads(
        torch, monkeypatch, lambda: torch.cuda._sleep(BUSY_CYCLES)
    )
    state = load_state_dict(member, device="cuda", dtype=torch.float16)
    for index in range(count):
        values = state[f"w{index}"].cpu()
        assert torch.equal(values, torch.full((elements,), float(index), dtype=torch.float16)), index


def test_gpu_destination_waits_for_a_freed_block_reader(
    torch: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without `dtype`: another thread's kernel still queued on the default stream reads a tensor it
    freed; a destination carved from that block must not be overwritten before the kernel ran."""
    skip_without_gpu(torch)
    elements = 1 << 18
    member = encode(
        make_safetensors(
            tmp_path / "freed.safetensors",
            {"w": ("F32", [elements], raw_bytes(torch.full((elements,), 7.0)))},
        )
    )
    copies: list[Any] = []

    def work() -> None:
        original = torch.full((elements,), 3.0, device="cuda")
        torch.cuda._sleep(BUSY_CYCLES)
        copies.append(original.clone())
        # Freed on the stream that still has to run the clone: the caching allocator may give its
        # block to the load's destination at once.

    busy_current_stream_during_loads(torch, monkeypatch, work)
    state = load_state_dict(member, device="cuda")
    assert torch.equal(copies[0].cpu(), torch.full((elements,), 3.0))
    assert torch.equal(state["w"].cpu(), torch.full((elements,), 7.0))


def test_corruption_returns_no_state(torch: Any, tmp_path: Path) -> None:
    source = make_safetensors(
        tmp_path / "noise.safetensors",
        {"noise": ("F32", [1 << 16], seeded_bytes(6, 1 << 18))},
    )
    output = tmp_path / "noise.ziptensors"
    ziptensors.encode_file(source, output, threads=1)
    with pytest.raises(CorruptionError):
        load_state_dict(Container(damaged(output, HEADER_LEN + (1 << 17))).member())


def test_load_state_dict_releases_the_gil(torch: Any, slow: tuple[Path, Path]) -> None:
    member = Container(slow[1], threads=1).member()
    stall, duration = stall_during(lambda: load_state_dict(member))
    assert stall < duration / 2


def test_load_into_a_meta_module(torch: Any, tmp_path: Path) -> None:
    torch.manual_seed(31)
    original = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.BatchNorm1d(8)).eval()
    file = tmp_path / "plain.safetensors"
    tensors = {
        name: ("I64" if tensor.dtype == torch.int64 else "F32", list(tensor.shape), raw_bytes(tensor))
        for name, tensor in original.state_dict().items()
    }
    member = encode(make_safetensors(file, tensors))
    with torch.device("meta"):
        module = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.BatchNorm1d(8)).eval()
    load_into(module, member)
    sample = torch.randn(3, 4)
    assert torch.equal(module(sample), original(sample))

    existing = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.BatchNorm1d(8)).eval()
    storage = existing[0].weight
    load_into(existing, member, assign=False)
    assert existing[0].weight is storage
    assert torch.equal(existing(sample), original(sample))


def test_load_into_strict_key_mismatch(torch: Any, tmp_path: Path) -> None:
    source = make_safetensors(tmp_path / "s.safetensors", {"other": ("F32", [4], bytes(16))})
    with torch.device("meta"):
        module = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_into(module, encode(source))


class BufferedLinear:
    """Builds a module with a non-persistent buffer, which no state dict can fill."""

    @staticmethod
    def build(torch: Any) -> Any:
        module = torch.nn.Linear(2, 2, bias=False)
        module.register_buffer("derived", torch.zeros(2), persistent=False)
        return module


def test_load_into_refuses_a_module_left_on_meta(
    torch: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = make_safetensors(tmp_path / "s.safetensors", {"weight": ("F32", [2, 2], bytes(16))})
    member = encode(source)
    with torch.device("meta"):
        module = BufferedLinear.build(torch)
    with pytest.raises(TorchLoadError, match="derived"):
        load_into(module, member)
    with torch.device("meta"):
        module = BufferedLinear.build(torch)
    with caplog.at_level(logging.WARNING, logger="ziptensors.torch"):
        load_into(module, member, strict=False)
    assert "derived" in caplog.text


@pytest.fixture(name="transformers_present")
def fixture_transformers_present() -> None:
    """A skip without transformers, taken before a test body imports from it."""
    pytest.importorskip("transformers")


@pytest.fixture(name="diffusers_present")
def fixture_diffusers_present() -> None:
    """A skip without diffusers or transformers, taken before a test body imports from them."""
    pytest.importorskip("transformers")
    pytest.importorskip("diffusers")


def clip_config() -> Any:
    """A tiny CLIP text config: no tied weights, one non-persistent buffer."""
    transformers = pytest.importorskip("transformers")
    return transformers.CLIPTextConfig(
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=16,
        projection_dim=8,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=1,
    )


def tiny_clip(torch: Any) -> Any:
    """A seeded CLIP text encoder in eval mode."""
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(41)
    return transformers.CLIPTextModel(clip_config()).eval()


def gpt2_config() -> Any:
    """A tiny GPT-2 config whose head is tied to the embeddings, so a file holds one of the two."""
    transformers = pytest.importorskip("transformers")
    return transformers.GPT2Config(
        vocab_size=64,
        n_positions=16,
        n_embd=8,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=1,
    )


def tiny_gpt2(torch: Any) -> Any:
    """A seeded GPT-2 with a tied head, in eval mode."""
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(42)
    return transformers.GPT2LMHeadModel(gpt2_config()).eval()


def saved(model: Any, directory: Path, weights: str = "model.safetensors") -> Member:
    """Saves `model` through its library's own writer and returns the encoded weights."""
    model.save_pretrained(directory)
    return encode(directory / weights)


@pytest.mark.usefixtures("transformers_present")
def test_transformers_from_pretrained_state_dict(torch: Any, tmp_path: Path) -> None:
    from transformers import CLIPTextModel

    original = tiny_clip(torch)
    member = saved(original, tmp_path / "clip")
    input_ids = torch.arange(8).unsqueeze(0)

    config = clip_config()
    state_dict = load_state_dict(member)
    model = CLIPTextModel.from_pretrained(None, config=config, state_dict=state_dict).eval()

    assert torch.equal(
        model(input_ids=input_ids).last_hidden_state,
        original(input_ids=input_ids).last_hidden_state,
    )


@pytest.mark.usefixtures("transformers_present")
def test_transformers_from_pretrained_ties_a_head(torch: Any, tmp_path: Path) -> None:
    from transformers import GPT2LMHeadModel

    original = tiny_gpt2(torch)
    member = saved(original, tmp_path / "gpt2")
    assert not any(tensor.name.startswith("lm_head") for tensor in member.tensors)
    input_ids = torch.arange(8).unsqueeze(0)

    state_dict = load_state_dict(member)
    model = GPT2LMHeadModel.from_pretrained(None, config=gpt2_config(), state_dict=state_dict).eval()

    assert model.lm_head.weight is model.transformer.wte.weight
    assert torch.equal(model(input_ids=input_ids).logits, original(input_ids=input_ids).logits)


@pytest.mark.usefixtures("transformers_present")
def test_transformers_meta_module_needs_its_tied_head_tied(torch: Any, tmp_path: Path) -> None:
    from transformers import GPT2LMHeadModel

    original = tiny_gpt2(torch)
    member = saved(original, tmp_path / "gpt2")
    input_ids = torch.arange(8).unsqueeze(0)
    with torch.device("meta"):
        strict_model = GPT2LMHeadModel(gpt2_config())
    with pytest.raises(RuntimeError, match="lm_head.weight"):
        load_into(strict_model, member)

    with torch.device("meta"):
        model = GPT2LMHeadModel(gpt2_config())
    load_into(model, member, strict=False)
    model.tie_weights()
    model.eval()

    assert torch.equal(model(input_ids=input_ids).logits, original(input_ids=input_ids).logits)


@pytest.mark.usefixtures("transformers_present")
def test_transformers_meta_module_needs_its_buffer_registered(torch: Any, tmp_path: Path) -> None:
    from transformers import CLIPTextModel

    original = tiny_clip(torch)
    member = saved(original, tmp_path / "clip")
    input_ids = torch.arange(8).unsqueeze(0)
    with torch.device("meta"):
        strict_model = CLIPTextModel(clip_config())
    with pytest.raises(TorchLoadError, match="position_ids"):
        load_into(strict_model, member)

    with torch.device("meta"):
        model = CLIPTextModel(clip_config())
    load_into(model, member, strict=False)
    # Transformers moves the embeddings module between versions (`embeddings` or
    # `text_model.embeddings`), so the buffer is found by its own name. Exactly one must exist, and it
    # must still be on `meta`: the file does not carry it, which is what this test proves.
    position_buffers = [
        name for name, _ in model.named_buffers() if name.rpartition(".")[2] == "position_ids"
    ]
    assert len(position_buffers) == 1, position_buffers
    assert model.get_buffer(position_buffers[0]).is_meta
    owner, _, buffer_name = position_buffers[0].rpartition(".")
    positions = torch.arange(model.config.max_position_embeddings).expand((1, -1))
    model.get_submodule(owner).register_buffer(buffer_name, positions, persistent=False)
    model.eval()

    assert torch.equal(
        model(input_ids=input_ids).last_hidden_state,
        original(input_ids=input_ids).last_hidden_state,
    )


UNET_ARGUMENTS: dict[str, Any] = {
    "block_out_channels": (4, 8),
    "layers_per_block": 1,
    "sample_size": 8,
    "in_channels": 4,
    "out_channels": 4,
    "down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D"),
    "up_block_types": ("CrossAttnUpBlock2D", "UpBlock2D"),
    "cross_attention_dim": 8,
    "norm_num_groups": 2,
    "attention_head_dim": 2,
}
VAE_ARGUMENTS: dict[str, Any] = {
    "block_out_channels": (4,),
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": ("DownEncoderBlock2D",),
    "up_block_types": ("UpDecoderBlock2D",),
    "latent_channels": 4,
    "norm_num_groups": 2,
    "sample_size": 8,
}
DIFFUSERS_WEIGHTS = "diffusion_pytorch_model.safetensors"


def tiny_unet(torch: Any) -> Any:
    """A seeded conditional UNet in eval mode."""
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(43)
    return diffusers.UNet2DConditionModel(**UNET_ARGUMENTS).eval()


def tiny_vae(torch: Any) -> Any:
    """A seeded VAE in eval mode."""
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(44)
    return diffusers.AutoencoderKL(**VAE_ARGUMENTS).eval()


def tiny_tokenizer(directory: Path) -> Any:
    """A CLIP tokenizer of a handful of tokens, from files written here."""
    transformers = pytest.importorskip("transformers")
    directory.mkdir()
    tokens = ["<|startoftext|>", "<|endoftext|>", "a</w>", "b</w>", "a", "b", "ab</w>"]
    vocabulary = {token: index for index, token in enumerate(tokens)}
    (directory / "vocab.json").write_text(json.dumps(vocabulary), encoding="utf-8")
    (directory / "merges.txt").write_text("#version: 0.2\na b</w>\n", encoding="utf-8")
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 16, "pad_token": "<|endoftext|>"}), encoding="utf-8"
    )
    return transformers.CLIPTokenizer.from_pretrained(directory, local_files_only=True)


def unet_sample(torch: Any) -> tuple[Any, Any, Any]:
    """Seeded inputs of a forward pass through `tiny_unet`."""
    generator = torch.Generator().manual_seed(45)
    latents = torch.randn(1, 4, 8, 8, generator=generator)
    return latents, torch.tensor(3), torch.randn(1, 4, 8, generator=generator)


@pytest.mark.usefixtures("diffusers_present")
def test_diffusers_from_config_and_load_into(torch: Any, tmp_path: Path) -> None:
    from diffusers import UNet2DConditionModel

    original = tiny_unet(torch)
    member = saved(original, tmp_path / "unet", DIFFUSERS_WEIGHTS)
    latents, step, context = unet_sample(torch)

    with torch.device("meta"):
        unet = UNet2DConditionModel.from_config(original.config)
    load_into(unet, member)
    unet.eval()

    assert torch.equal(unet(latents, step, context).sample, original(latents, step, context).sample)


@pytest.mark.usefixtures("diffusers_present")
def test_diffusers_from_single_file_with_a_dict(torch: Any, tmp_path: Path) -> None:
    from diffusers import UNet2DConditionModel

    original = tiny_unet(torch)
    directory = tmp_path / "unet"
    member = saved(original, directory, DIFFUSERS_WEIGHTS)
    latents, step, context = unet_sample(torch)

    state_dict = load_state_dict(member)
    unet = UNet2DConditionModel.from_single_file(
        state_dict, config=str(directory), local_files_only=True
    ).eval()

    assert torch.equal(unet(latents, step, context).sample, original(latents, step, context).sample)


@pytest.mark.usefixtures("diffusers_present")
def test_diffusers_from_single_file_without_a_config_needs_the_hub(
    torch: Any, tmp_path: Path
) -> None:
    """Recorded limit: with no `config`, diffusers looks for a default repository on the hub."""
    from diffusers import UNet2DConditionModel

    member = saved(tiny_unet(torch), tmp_path / "unet", DIFFUSERS_WEIGHTS)
    with pytest.raises(OSError, match="config.json"):
        UNet2DConditionModel.from_single_file(load_state_dict(member))


@pytest.mark.usefixtures("diffusers_present")
def test_diffusers_from_pretrained_takes_no_state_dict(torch: Any, tmp_path: Path) -> None:
    """Recorded limit: `ModelMixin.from_pretrained` has no in-memory route."""
    from diffusers import UNet2DConditionModel

    original = tiny_unet(torch)
    member = saved(original, tmp_path / "unet", DIFFUSERS_WEIGHTS)
    with pytest.raises(OSError, match="config.json"):
        UNet2DConditionModel.from_pretrained(
            None, config=dict(original.config), state_dict=load_state_dict(member)
        )


@pytest.mark.usefixtures("diffusers_present")
def test_diffusers_pipeline_from_single_file_takes_no_dict(torch: Any) -> None:
    """Recorded limit: a pipeline's `from_single_file` takes a path, unlike a model's."""
    from diffusers import StableDiffusionPipeline

    with pytest.raises(TypeError, match="path should be"):
        StableDiffusionPipeline.from_single_file({"weight": torch.zeros(1)})


@pytest.mark.usefixtures("diffusers_present")
def test_diffusers_pipeline_from_components(torch: Any, tmp_path: Path) -> None:
    from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextModel

    original_unet = tiny_unet(torch)
    original_vae = tiny_vae(torch)
    original_text_encoder = tiny_clip(torch)
    unet_member = saved(original_unet, tmp_path / "unet", DIFFUSERS_WEIGHTS)
    vae_member = saved(original_vae, tmp_path / "vae", DIFFUSERS_WEIGHTS)
    text_member = saved(original_text_encoder, tmp_path / "text_encoder")
    tokenizer = tiny_tokenizer(tmp_path / "tokenizer")

    with torch.device("meta"):
        unet = UNet2DConditionModel.from_config(original_unet.config)
        vae = AutoencoderKL.from_config(original_vae.config)
    load_into(unet, unet_member)
    load_into(vae, vae_member)
    text_encoder = CLIPTextModel.from_pretrained(
        None, config=clip_config(), state_dict=load_state_dict(text_member)
    )
    pipeline = StableDiffusionPipeline(
        vae=vae.eval(),
        text_encoder=text_encoder.eval(),
        tokenizer=tokenizer,
        unet=unet.eval(),
        scheduler=DDIMScheduler(steps_offset=1, clip_sample=False),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )

    reference = StableDiffusionPipeline(
        vae=original_vae,
        text_encoder=original_text_encoder,
        tokenizer=tokenizer,
        unet=original_unet,
        scheduler=DDIMScheduler(steps_offset=1, clip_sample=False),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )

    def images(target: Any) -> Any:
        target.set_progress_bar_config(disable=True)
        return target(
            "a b",
            height=8,
            width=8,
            num_inference_steps=2,
            generator=torch.Generator().manual_seed(46),
            output_type="pt",
        ).images

    assert torch.equal(images(pipeline), images(reference))


def test_torch_load_through_open(torch: Any, tmp_path: Path) -> None:
    torch.manual_seed(51)
    state = {
        "weight": torch.randn(64, 64),
        "half": torch.randn(16).to(torch.bfloat16),
        "steps": torch.arange(32),
        "nested": {"scale": 1.5, "name": "tiny"},
    }
    path = tmp_path / "state.pt"
    torch.save(state, path)
    member = encode(path)
    assert member.profile == "pt_zip"

    loaded = torch.load(member.open(), weights_only=True)

    for name in ("weight", "half", "steps"):
        assert loaded[name].dtype == state[name].dtype
        assert torch.equal(loaded[name], state[name])
    assert loaded["nested"] == state["nested"]


def test_torch_load_of_a_module_state_through_open(torch: Any, tmp_path: Path) -> None:
    torch.manual_seed(52)
    original = torch.nn.Linear(4, 4).eval()
    path = tmp_path / "linear.pt"
    torch.save(original.state_dict(), path)
    member = encode(path)
    sample = torch.randn(2, 4)

    module = torch.nn.Linear(4, 4).eval()
    module.load_state_dict(torch.load(member.open(), weights_only=True))

    assert torch.equal(module(sample), original(sample))


def test_torch_load_mmap_needs_a_path(torch: Any, tmp_path: Path) -> None:
    """Recorded limit: `mmap=True` refuses any file object."""
    path = tmp_path / "state.pt"
    torch.save({"weight": torch.zeros(1024)}, path)
    with pytest.raises(ValueError, match="file path"):
        torch.load(encode(path).open(), weights_only=True, mmap=True)


class Payload:
    """A class `weights_only=True` must refuse. It does nothing, so nothing real is unpickled."""


def test_torch_load_through_open_still_refuses_globals(torch: Any, tmp_path: Path) -> None:
    path = tmp_path / "unsafe.pt"
    torch.save({"payload": Payload(), "pad": torch.zeros(4096)}, path)
    with pytest.raises(Exception, match="Weights only load failed") as caught:
        torch.load(encode(path).open(), weights_only=True)
    assert type(caught.value).__name__ == "UnpicklingError"
