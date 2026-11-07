"""File: tests/test_safe_open.py
Purpose: contract tests of `ziptensors.safetensors.safe_open`, and its agreement with the real
`safetensors.safe_open` over the same generated file.

Main responsibilities: keys, metadata, every accepted dtype, 0-d and empty tensors, the slice table,
malformed headers, the closed state, member naming, the numpy framework, and the staged GPU path.

Key functions: `write_safetensors`, `opened`, `compare_slices`.

Implementation notes: fixtures are generated with a seeded PRNG under `tmp_path`. A comparison with
the safetensors package skips when it is absent, and GPU tests skip without
`torch.cuda.is_available()`.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import struct
from typing import Any

import pytest

from test_ziptensors import seeded_bytes
import ziptensors
from ziptensors import Container, Member
from ziptensors.safetensors import SafetensorError, SafeSlice, safe_open
from ziptensors.torch import TorchLoadError

# Every dtype `ziptensors.torch` accepts, with its torch name.
DTYPES = [
    ("F64", "float64", 8),
    ("F32", "float32", 4),
    ("F16", "float16", 2),
    ("BF16", "bfloat16", 2),
    ("F8_E4M3", "float8_e4m3fn", 1),
    ("F8_E5M2", "float8_e5m2", 1),
    ("I64", "int64", 8),
    ("I32", "int32", 4),
    ("I16", "int16", 2),
    ("I8", "int8", 1),
    ("U64", "uint64", 8),
    ("U32", "uint32", 4),
    ("U16", "uint16", 2),
    ("U8", "uint8", 1),
    ("BOOL", "bool", 1),
]

# Basic indices of a (4, 5, 6) tensor, each compared with the safetensors package.
SLICES: list[Any] = [
    0,
    -1,
    3,
    slice(None),
    slice(1, 3),
    slice(0, 4, 2),
    slice(10, 20),
    slice(-3, None),
    (1, 2),
    (1, 2, 3),
    (-1, -2, -3),
    (slice(None), 1),
    (slice(None), slice(1, 4), slice(0, 6, 3)),
    (1, slice(1, 5, 2)),
    (2, slice(4, 1)),
    Ellipsis,
    (Ellipsis, 2),
    (1, Ellipsis),
    (None, 1),
    (1, None, slice(2, 4)),
    (slice(None), None, Ellipsis, None),
    (),
]


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, list[int], bytes]],
    metadata: dict[str, str] | None = None,
) -> Path:
    """Writes a safetensors file of `tensors`, name to (dtype, shape, data), and returns `path`.

    `metadata`, if given, becomes `__metadata__`. The header is padded with spaces to 8 bytes, as
    the safetensors package pads it.
    """
    header: dict[str, Any] = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    offset = 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    payload = b"".join(data for _, _, data in tensors.values())
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return path


def encoded(path: Path) -> Path:
    """Encodes the file at `path` next to it and returns the container path."""
    output = path.with_suffix(".ziptensors")
    ziptensors.encode_file(path, output, threads=1)
    return output


def raw_bytes(tensor: Any) -> bytes:
    """Returns the storage of a contiguous CPU tensor. Works for bfloat16 and float8."""
    return ctypes.string_at(tensor.data_ptr(), tensor.nbytes)


def seeded_data(seed: int, dtype: str, count: int, item_bytes: int) -> bytes:
    """Returns `count` seeded elements of `dtype`, with BOOL bytes kept to 0 and 1."""
    data = seeded_bytes(seed, count * item_bytes)
    return bytes(byte % 2 for byte in data) if dtype == "BOOL" else data


@pytest.fixture(name="torch")
def fixture_torch() -> Any:
    """The torch module, or a skip."""
    return pytest.importorskip("torch")


@pytest.fixture(name="every_dtype")
def fixture_every_dtype(tmp_path: Path) -> tuple[Path, Path]:
    """A safetensors file with every accepted dtype, 0-d and empty tensors, and metadata, and its
    container."""
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for seed, (dtype, _, item_bytes) in enumerate(DTYPES):
        tensors[f"t.{dtype}"] = (dtype, [4, 5, 6], seeded_data(70 + seed, dtype, 120, item_bytes))
        tensors[f"s.{dtype}"] = (dtype, [], seeded_data(90 + seed, dtype, 1, item_bytes))
        tensors[f"e.{dtype}"] = (dtype, [0, 3], b"")
    source = write_safetensors(
        tmp_path / "every.safetensors", tensors, metadata={"format": "pt", "note": "seeded"}
    )
    return source, encoded(source)


def opened(container: Path, framework: str = "pt", device: Any = "cpu") -> safe_open:
    """A `safe_open` of the single member of `container`."""
    return safe_open(container, framework, device=device)


def test_keys_metadata_and_tensors_equal_the_safetensors_package(
    torch: Any, every_dtype: tuple[Path, Path]
) -> None:
    reference_module = pytest.importorskip("safetensors")
    source, container = every_dtype
    with opened(container) as file, reference_module.safe_open(source, "pt") as reference:
        assert file.keys() == reference.keys()
        assert file.offset_keys() == reference.offset_keys()
        assert file.metadata() == reference.metadata()
        for name in reference.keys():
            got, expected = file.get_tensor(name), reference.get_tensor(name)
            assert got.dtype == expected.dtype, name
            assert got.shape == expected.shape, name
            assert raw_bytes(got) == raw_bytes(expected), name
            piece = file.get_slice(name)
            assert piece.get_shape() == reference.get_slice(name).get_shape()
            assert piece.get_dtype() == reference.get_slice(name).get_dtype()


def test_slices_equal_the_safetensors_package(torch: Any, every_dtype: tuple[Path, Path]) -> None:
    reference_module = pytest.importorskip("safetensors")
    source, container = every_dtype
    with opened(container) as file, reference_module.safe_open(source, "pt") as reference:
        for dtype in ("F32", "BF16", "F8_E5M2", "BOOL"):
            name = f"t.{dtype}"
            for index in SLICES:
                got = file.get_slice(name)[index]
                expected = reference.get_slice(name)[index]
                assert got.shape == expected.shape, (name, index)
                assert raw_bytes(got.contiguous()) == raw_bytes(expected.contiguous()), (name, index)
        for index in (Ellipsis, (), None):
            got = file.get_slice("s.F32")[index]
            assert raw_bytes(got) == raw_bytes(reference.get_slice("s.F32")[index])


def test_slices_of_a_known_tensor(torch: Any, tmp_path: Path) -> None:
    """Without the safetensors package: the slices against torch's own indexing."""
    whole = torch.arange(120, dtype=torch.int32).reshape(4, 5, 6)
    container = encoded(
        write_safetensors(tmp_path / "k.safetensors", {"w": ("I32", [4, 5, 6], raw_bytes(whole))})
    )
    with opened(container) as file:
        for index in SLICES:
            got = file.get_slice("w")[index]
            assert torch.equal(got, whole[index]), index
            # A slice owns its memory: no view of a larger read is handed out.
            assert got.untyped_storage().nbytes() == got.nbytes, index


@pytest.mark.parametrize(
    ("index", "error"),
    [
        (slice(None, None, 0), ValueError),
        (slice(None, None, -1), ValueError),
        (4, IndexError),
        (-5, IndexError),
        ((0, 0, 0, 0), IndexError),
        ((Ellipsis, 0, Ellipsis), IndexError),
        ([0, 1], TypeError),
        (True, TypeError),
        (1.0, TypeError),
    ],
)
def test_refused_slice_indices(
    torch: Any, every_dtype: tuple[Path, Path], index: Any, error: type[Exception]
) -> None:
    with opened(every_dtype[1]) as file, pytest.raises(error):
        file.get_slice("t.F32")[index]


def test_closed_file_refuses_every_call(torch: Any, every_dtype: tuple[Path, Path]) -> None:
    with opened(every_dtype[1]) as file:
        piece = file.get_slice("t.F32")
        assert isinstance(piece, SafeSlice)
    for call in (
        file.keys,
        file.offset_keys,
        file.metadata,
        file.get_tensors,
        lambda: file.get_tensor("t.F32"),
        lambda: file.get_slice("t.F32"),
        lambda: piece[0],
    ):
        with pytest.raises(SafetensorError, match="File is closed"):
            call()


def test_missing_tensor_framework_and_device(torch: Any, every_dtype: tuple[Path, Path]) -> None:
    container = every_dtype[1]
    with opened(container) as file:
        with pytest.raises(SafetensorError, match="File does not contain tensor absent"):
            file.get_tensor("absent")
        with pytest.raises(SafetensorError, match="File does not contain tensor absent"):
            file.get_slice("absent")
    with pytest.raises(SafetensorError, match="framework 'flax'"):
        opened(container, "flax")
    with pytest.raises(SafetensorError, match="meta"):
        opened(container, device="meta")
    with pytest.raises(SafetensorError, match="not a device"):
        opened(container, device="toaster")


def test_without_with_and_get_tensors_in_offset_order(
    torch: Any, every_dtype: tuple[Path, Path]
) -> None:
    file = opened(every_dtype[1])
    tensors = file.get_tensors()
    assert list(tensors) == file.offset_keys()
    assert raw_bytes(tensors["t.F16"]) == raw_bytes(file.get_tensor("t.F16"))


def test_no_metadata_is_none(torch: Any, tmp_path: Path) -> None:
    container = encoded(write_safetensors(tmp_path / "n.safetensors", {"w": ("U8", [3], b"abc")}))
    with opened(container) as file:
        assert file.metadata() is None
        assert raw_bytes(file.get_tensor("w")) == b"abc"


def test_dtype_outside_the_list_is_listed_but_not_read(torch: Any, tmp_path: Path) -> None:
    container = encoded(
        write_safetensors(
            tmp_path / "c.safetensors",
            {
                "c": ("C64", [2], seeded_bytes(80, 16)),
                "e8": ("F8_E8M0", [3], seeded_bytes(81, 3)),
                "w": ("F32", [1], bytes(4)),
            },
        )
    )
    with opened(container) as file:
        assert file.keys() == ["c", "e8", "w"]
        assert file.get_slice("e8").get_dtype() == "F8_E8M0"
        with pytest.raises(TorchLoadError, match="F8_E8M0"):
            file.get_tensor("e8")
        assert file.get_slice("c").get_dtype() == "C64"
        assert file.get_slice("c").get_shape() == [2]
        with pytest.raises(TorchLoadError, match="'c'.*C64"):
            file.get_tensor("c")
        with pytest.raises(TorchLoadError, match="C64"):
            file.get_slice("c")[0]


def test_bool_byte_outside_zero_and_one_refused(torch: Any, tmp_path: Path) -> None:
    container = encoded(
        write_safetensors(tmp_path / "b.safetensors", {"mask": ("BOOL", [3], bytes([0, 2, 1]))})
    )
    with opened(container) as file, pytest.raises(TorchLoadError, match="'mask'.*byte 2"):
        file.get_tensor("mask")


def raw_header(path: Path, header: bytes, data: bytes, length: int | None = None) -> Path:
    """Writes a file with a raw header, an optional false length prefix, and `data`."""
    prefix = struct.pack("<Q", len(header) if length is None else length)
    path.write_bytes(prefix + header + data)
    return path


def entry(dtype: str, shape: list[int], begin: int, end: int) -> dict[str, Any]:
    """One header entry."""
    return {"dtype": dtype, "shape": shape, "data_offsets": [begin, end]}


MALFORMED: list[tuple[str, bytes, bytes, int | None, str]] = [
    ("huge", b"{}", b"", 100_000_001, "over 100 MB"),
    ("past", b"{}", b"", 1 << 20, "runs past the member"),
    ("json", b"{not json}", b"", None, "not UTF-8 JSON"),
    ("utf8", b'{"\xff": 1}', b"", None, "not UTF-8 JSON"),
    ("array", b"[1, 2]", b"", None, "not a JSON object"),
    ("meta", json.dumps({"__metadata__": {"k": 1}}).encode(), b"", None, "map of strings"),
    ("entry", json.dumps({"w": [1]}).encode(), b"", None, "not an object"),
    ("shape", json.dumps({"w": entry("F32", [-1], 0, 4)}).encode(), bytes(4), None, "shape"),
    ("dtype", json.dumps({"w": entry("F32", [2], 0, 4)}).encode(), bytes(4), None, "needs 8"),
    ("back", json.dumps({"w": entry("U8", [0], 4, 0)}).encode(), bytes(4), None, "backwards"),
    (
        "gap",
        json.dumps({"a": entry("U8", [2], 0, 2), "b": entry("U8", [2], 4, 6)}).encode(),
        bytes(6),
        None,
        "starts at data offset 4",
    ),
    (
        "overlap",
        json.dumps({"a": entry("U8", [4], 0, 4), "b": entry("U8", [4], 0, 4)}).encode(),
        bytes(8),
        None,
        "starts at data offset 0",
    ),
    ("garbage", json.dumps({"w": entry("GARBAGE", [1], 0, 1)}).encode(), bytes(1), None, "GARBAGE"),
    ("lower", json.dumps({"w": entry("f32", [1], 0, 4)}).encode(), bytes(4), None, "'f32' is not"),
    ("tail", json.dumps({"w": entry("U8", [2], 0, 2)}).encode(), bytes(5), None, "holds 5 bytes"),
]


@pytest.mark.parametrize(
    ("label", "header", "data", "length", "message"), MALFORMED, ids=[case[0] for case in MALFORMED]
)
def test_malformed_header_refused(
    torch: Any,
    tmp_path: Path,
    label: str,
    header: bytes,
    data: bytes,
    length: int | None,
    message: str,
) -> None:
    source = raw_header(tmp_path / f"{label}.safetensors", header, data, length)
    with pytest.raises(SafetensorError, match=message):
        opened(encoded(source))
    reference_module = pytest.importorskip("safetensors")
    # The safetensors package refuses each of these files too, with an error class per case.
    with pytest.raises(Exception):  # noqa: B017
        reference_module.safe_open(source, "pt")


def test_member_of_another_profile_is_named(torch: Any, tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 0) + bytes(8))
    with pytest.raises(SafetensorError, match="recognized as gguf, not safetensors"):
        opened(encoded(gguf))


def test_too_short_member_refused(torch: Any, tmp_path: Path) -> None:
    short = tmp_path / "short.bin"
    short.write_bytes(b"abc")
    with pytest.raises(SafetensorError, match="hold no safetensors header"):
        opened(encoded(short))


def test_member_of_a_pack_by_name_or_object(torch: Any, tmp_path: Path) -> None:
    first = write_safetensors(tmp_path / "a.safetensors", {"x": ("U8", [2], b"ab")})
    second = write_safetensors(tmp_path / "b.safetensors", {"y": ("U8", [3], b"cde")})
    output = tmp_path / "pack.ziptensors"
    ziptensors.pack([first, second], output, threads=1)
    with safe_open(output, "pt", member="b.safetensors", threads=1) as file:
        assert file.keys() == ["y"]
    member: Member = Container(output).member("a.safetensors")
    with safe_open(member, "pt") as file:
        assert raw_bytes(file.get_tensor("x")) == b"ab"
    with pytest.raises(TypeError, match="not to a Member"):
        safe_open(member, "pt", member="a.safetensors")


def test_numpy_framework_equals_the_safetensors_package(every_dtype: tuple[Path, Path]) -> None:
    numpy = pytest.importorskip("numpy")
    reference_module = pytest.importorskip("safetensors")
    source, container = every_dtype
    readable = [name for name, _, _ in DTYPES if name not in ("BF16", "F8_E4M3", "F8_E5M2")]
    with opened(container, "np") as file, reference_module.safe_open(source, "np") as reference:
        assert file.metadata() == reference.metadata()
        for dtype in readable:
            for name in (f"t.{dtype}", f"s.{dtype}", f"e.{dtype}"):
                got, expected = file.get_tensor(name), reference.get_tensor(name)
                assert isinstance(got, numpy.ndarray)
                assert got.dtype == expected.dtype and got.shape == expected.shape, name
                assert got.tobytes() == expected.tobytes(), name
            # Against numpy's indexing of the whole array: the numpy slices of safetensors 0.7
            # ignore a step, so `[0:4:2]` returns all four rows there.
            whole = reference.get_tensor(f"t.{dtype}")
            for index in SLICES:
                got = file.get_slice(f"t.{dtype}")[index]
                expected = whole[index]
                assert got.shape == expected.shape and got.tobytes() == expected.tobytes()
        with pytest.raises(SafetensorError, match="BF16 has no numpy dtype"):
            file.get_tensor("t.BF16")
    with pytest.raises(SafetensorError, match="device 'cpu' only"):
        opened(container, "np", device="cuda")
    torch = pytest.importorskip("torch")
    with opened(container, "np", device=torch.device("cpu")) as file:
        expected = opened(container, "np").get_tensor("t.U8")
        assert file.get_tensor("t.U8").tobytes() == expected.tobytes()


def test_gpu_tensors_and_slices_equal_the_host(
    torch: Any, every_dtype: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA or ROCm device")
    import ziptensors.torch

    # Tiny staging buffers, so tensors span pieces and both buffers take turns many times.
    monkeypatch.setattr(ziptensors.torch, "STAGING_BYTES", 96)
    container = every_dtype[1]
    with opened(container) as host, opened(container, device=0) as device:
        everything = device.get_tensors()
        for name in host.keys():
            expected = host.get_tensor(name)
            for got in (everything[name], device.get_tensor(name)):
                assert got.device.type == "cuda", name
                assert got.dtype == expected.dtype and got.shape == expected.shape, name
                assert raw_bytes(got.cpu()) == raw_bytes(expected), name
        got = device.get_slice("t.F32")[1, 2:4]
        assert got.device.type == "cuda"
        assert raw_bytes(got.cpu()) == raw_bytes(host.get_slice("t.F32")[1, 2:4])
