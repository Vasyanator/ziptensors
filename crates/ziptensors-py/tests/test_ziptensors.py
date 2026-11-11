"""File: tests/test_ziptensors.py
Purpose: contract tests of the Python module `ziptensors`, run under pytest against a built module.

Main responsibilities: roundtrips through every kind of destination buffer, refusal of unusable
buffers, the mapping of core errors to exception families, GIL release during slow work, and the
agreement of the stub with the runtime.

Key functions: `make_safetensors`, `make_gguf`, `stall_during`.

Implementation notes: all data is generated from seeds under pytest's `tmp_path`, so a run leaves
nothing behind. numpy, torch and xxhash are optional, and their tests skip without them. Run with
`PYTHONDONTWRITEBYTECODE=1`, see `src/MODULE_README.md`.
"""

from __future__ import annotations

import array
import ast
import ctypes
from collections.abc import Callable
import inspect
import json
import logging
import os
from pathlib import Path
import random
import struct
import threading
import time

import pytest

import ziptensors
from ziptensors import (
    BudgetError,
    Container,
    CorruptionError,
    FormatError,
    InputError,
    NotDurableError,
    NotFoundError,
    PackBuilder,
    RangeError,
    UnsupportedError,
    ZiptensorsError,
    ZiptensorsIoError,
)

log = logging.getLogger(__name__)

STUB = Path(__file__).resolve().parent.parent / "ziptensors.pyi"

# Stable values of `ErrorCode`, spelled out because they are ABI.
CODE_IO = 1
CODE_NOT_ZIPTENSORS = 2
CODE_UNSUPPORTED_VERSION = 3
CODE_MALFORMED_INDEX = 6
CODE_BUDGET_EXCEEDED = 7
CODE_CHECKSUM_MISMATCH = 9
CODE_OUT_OF_BOUNDS = 10
CODE_DESTINATION_SIZE = 11
CODE_TENSOR_NOT_FOUND = 12
CODE_MEMBER_NOT_FOUND = 13
CODE_UNSAFE_MEMBER_NAME = 14
CODE_SYMLINK_INPUT = 18
CODE_DUPLICATE_MEMBER = 26

# Offsets into the container layout of the spec, for the tests that damage a file.
HEADER_LEN = 64
HEADER_MAJOR_OFFSET = 8
HEADER_HASH_OFFSET = 56
FOOTER_LEN = 64

GGUF_TYPE_F32 = 0


def seeded_bytes(seed: int, size: int) -> bytes:
    """Returns `size` incompressible bytes determined by `seed`."""
    rng = random.Random(seed)
    # In blocks: one `randbytes` call is limited to 256 MiB.
    blocks = [rng.randbytes(1 << 24) for _ in range(size >> 24)]
    return b"".join(blocks) + rng.randbytes(size & ((1 << 24) - 1))


def compressible_bytes(seed: int, size: int) -> bytes:
    """Returns `size` bytes of 16 distinct values: no repeats to match, so decoding does real work."""
    table = bytes(value % 16 * 3 for value in range(256))
    return seeded_bytes(seed, size).translate(table)


def make_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> Path:
    """Writes a safetensors file of `tensors`, name to (dtype, shape, data), and returns `path`."""
    header = {}
    offset = 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    encoded = json.dumps(header).encode()
    payload = b"".join(data for _, _, data in tensors.values())
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return path


def make_gguf(path: Path, name: str, shape: list[int], data: bytes) -> Path:
    """Writes a GGUF v3 file with one F32 tensor and no metadata, and returns `path`."""
    encoded = name.encode()
    info = struct.pack("<Q", len(encoded)) + encoded + struct.pack("<I", len(shape))
    info += b"".join(struct.pack("<Q", dim) for dim in shape)
    info += struct.pack("<IQ", GGUF_TYPE_F32, 0)
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + info
    padding = bytes(-len(head) % 32)
    path.write_bytes(head + padding + data)
    return path


@pytest.fixture(name="tensors")
def fixture_tensors() -> dict[str, tuple[str, list[int], bytes]]:
    """Two float32 tensors, one of them too random to compress."""
    return {
        "first": ("F32", [64, 256], compressible_bytes(1, 64 * 256 * 4)),
        "second": ("F32", [1024], seeded_bytes(2, 1024 * 4)),
    }


@pytest.fixture(name="source")
def fixture_source(tmp_path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> Path:
    """A safetensors source file."""
    return make_safetensors(tmp_path / "model.safetensors", tensors)


@pytest.fixture(name="encoded")
def fixture_encoded(tmp_path: Path, source: Path) -> Path:
    """`source` as a single-member container."""
    output = tmp_path / "model.ziptensors"
    ziptensors.encode_file(source, output, threads=1)
    return output


@pytest.mark.parametrize("mode", ["fast", "balanced", "max"])
def test_roundtrip_through_bytearray(tmp_path: Path, source: Path, mode: str) -> None:
    output = tmp_path / "out.ziptensors"
    report = ziptensors.encode_file(str(source), str(output), mode=mode, threads=2)
    expected = source.read_bytes()
    assert report.output_len == output.stat().st_size
    assert report.members[0].len == len(expected)

    member = Container(output, threads=2).member()
    assert len(expected) == member.len
    buffer = bytearray(member.len)
    member.read_at(0, buffer)
    assert buffer == expected
    member.verify()


def test_metadata(encoded: Path, source: Path, tensors: dict) -> None:
    container = Container(encoded)
    assert len(container.members) == 1
    info = container.members[0]
    member = container.member(info.name)
    assert info.name == source.name == member.name
    assert info.len == source.stat().st_size
    assert info.profile == "safetensors"
    assert info.tensor_count == len(tensors)
    assert isinstance(info.blake3, bytes) and len(info.blake3) == 32
    assert info.blake3 == member.blake3

    assert [tensor.name for tensor in member.tensors] == list(tensors)
    tensor = member.tensor_info("first")
    assert tensor.dtype == "F32"
    assert tensor.shape == [64, 256]
    assert tensor.len == 64 * 256 * 4
    assert member.read(tensor.offset, tensor.len) == tensors["first"][2]


def test_read_returns_bytes(encoded: Path, source: Path) -> None:
    expected = source.read_bytes()
    member = Container(encoded).member()
    part = member.read(5, 1000)
    assert isinstance(part, bytes)
    assert part == expected[5:1005]
    assert member.read(member.len, 0) == b""


def test_restore_file(tmp_path: Path, encoded: Path, source: Path) -> None:
    out = tmp_path / "restored.bin"
    report = Container(encoded).member().restore_file(out)
    assert out.read_bytes() == source.read_bytes()
    assert report.bytes == source.stat().st_size
    assert report.blake3 == Container(encoded).member().blake3


def test_empty_buffer_reads_nothing(encoded: Path) -> None:
    Container(encoded).member().read_at(0, bytearray())


def test_stdlib_buffers_of_any_format(encoded: Path, tensors: dict) -> None:
    expected = tensors["second"][2]
    member = Container(encoded).member()

    floats = array.array("f", bytes(len(expected)))
    member.decode_tensor_into("second", floats)
    assert floats.tobytes() == expected

    c_floats = (ctypes.c_float * (len(expected) // 4))()
    member.decode_tensor_into("second", c_floats)
    assert bytes(c_floats) == expected

    view = memoryview(bytearray(len(expected))).cast("I")
    member.decode_tensor_into("second", view)
    assert view.tobytes() == expected


def test_numpy_float32_buffer(encoded: Path, tensors: dict) -> None:
    numpy = pytest.importorskip("numpy")
    dtype, shape, data = tensors["first"]
    member = Container(encoded).member()
    target = numpy.empty(shape, dtype=numpy.float32)
    member.decode_tensor_into("first", target)
    assert target.tobytes() == data
    assert dtype == member.tensor_info("first").dtype


def test_numpy_structured_field_named_like_an_object_code(encoded: Path, tensors: dict) -> None:
    numpy = pytest.importorskip("numpy")
    data = tensors["second"][2]
    target = numpy.empty(len(data) // 4, dtype=[("O", numpy.float32)])
    Container(encoded).member().decode_tensor_into("second", target)
    assert target.tobytes() == data


def test_torch_tensor_filled_through_numpy(encoded: Path, tensors: dict) -> None:
    numpy = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    _, shape, data = tensors["first"]
    storage = numpy.empty(shape, dtype=numpy.float32)
    tensor = torch.from_numpy(storage)
    Container(encoded).member().decode_tensor_into("first", storage)
    expected = numpy.frombuffer(data, dtype=numpy.float32).reshape(shape).copy()
    assert torch.equal(tensor, torch.from_numpy(expected))


@pytest.mark.parametrize(
    "readonly",
    [bytes(16), memoryview(bytes(16)), memoryview(bytearray(16)).toreadonly()],
    ids=["bytes", "memoryview_of_bytes", "readonly_memoryview"],
)
def test_read_only_buffer_rejected(encoded: Path, readonly: object) -> None:
    with pytest.raises(BufferError, match="writable"):
        Container(encoded).member().read_at(0, readonly)


def test_read_only_numpy_rejected(encoded: Path) -> None:
    numpy = pytest.importorskip("numpy")
    target = numpy.zeros(16, dtype=numpy.float32)
    target.flags.writeable = False
    with pytest.raises(BufferError, match="writable"):
        Container(encoded).member().read_at(0, target)


def test_non_contiguous_rejected(encoded: Path) -> None:
    member = Container(encoded).member()
    with pytest.raises(BufferError, match="contiguous"):
        member.read_at(0, memoryview(bytearray(16))[::2])


def test_non_contiguous_numpy_rejected(encoded: Path) -> None:
    numpy = pytest.importorskip("numpy")
    member = Container(encoded).member()
    with pytest.raises(BufferError, match="contiguous"):
        member.read_at(0, numpy.zeros(16, dtype=numpy.float32)[::2])
    with pytest.raises(BufferError, match="contiguous"):
        member.read_at(0, numpy.zeros((4, 4), dtype=numpy.float32, order="F"))


def test_object_array_rejected(encoded: Path) -> None:
    """Decoded bytes in an object array would be forged pointers, a crash at the next access."""
    numpy = pytest.importorskip("numpy")
    target = numpy.empty(4, dtype=object)
    with pytest.raises(BufferError, match="Python objects"):
        Container(encoded).member().read_at(0, target)
    assert all(item is None for item in target)


def test_non_buffer_rejected(encoded: Path) -> None:
    with pytest.raises(TypeError, match="buffer protocol"):
        Container(encoded).member().read_at(0, [0, 0, 0, 0])


def test_wrong_size_rejected(encoded: Path, tensors: dict) -> None:
    length = len(tensors["second"][2])
    member = Container(encoded).member()
    for size in (length - 1, length + 1, 0):
        with pytest.raises(RangeError) as caught:
            member.decode_tensor_into("second", bytearray(size))
        assert caught.value.code == CODE_DESTINATION_SIZE


def test_out_of_bounds(encoded: Path) -> None:
    member = Container(encoded).member()
    for call in (
        lambda: member.read_at(member.len, bytearray(1)),
        lambda: member.read(1, member.len),
        lambda: member.read(0, 2**62),
        lambda: member.read(2**64 - 1, 2),
    ):
        with pytest.raises(RangeError) as caught:
            call()
        assert caught.value.code == CODE_OUT_OF_BOUNDS


def test_buffer_is_released_after_success_and_failure(encoded: Path) -> None:
    """A bytearray cannot be resized while a view is open, so a resize proves the release."""
    member = Container(encoded).member()
    buffer = bytearray(16)
    member.read_at(0, buffer)
    buffer.append(0)
    with pytest.raises(RangeError):
        member.read_at(member.len, buffer)
    buffer.append(0)


def test_concurrent_reads(encoded: Path, source: Path) -> None:
    expected = source.read_bytes()
    member = Container(encoded, threads=2).member()
    failures: list[int] = []

    def worker(seed: int) -> None:
        rng = random.Random(seed)
        for _ in range(50):
            offset = rng.randrange(len(expected))
            length = rng.randrange(1, len(expected) - offset + 1)
            buffer = bytearray(length)
            member.read_at(offset, buffer)
            if buffer != expected[offset : offset + length]:
                failures.append(seed)

    workers = [threading.Thread(target=worker, args=(seed,)) for seed in range(8)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    assert not failures


def test_pack_named_and_bare_inputs(tmp_path: Path, source: Path) -> None:
    other = make_gguf(tmp_path / "other.gguf", "weight", [4, 2], seeded_bytes(3, 8 * 4))
    output = tmp_path / "set.ziptensors"
    report = ziptensors.pack([source, (other, "models/renamed.gguf")], output, threads=1)
    assert [member.name for member in report.members] == [source.name, "models/renamed.gguf"]

    container = Container(output)
    assert [info.name for info in container.members] == [source.name, "models/renamed.gguf"]
    assert [info.profile for info in container.members] == ["safetensors", "gguf"]
    named = container.member("models/renamed.gguf")
    assert named.read(0, named.len) == other.read_bytes()
    with pytest.raises(NotFoundError) as caught:
        container.member()
    assert caught.value.code == CODE_MEMBER_NOT_FOUND


def test_pack_deduplicates_in_default_mode(tmp_path: Path, tensors: dict) -> None:
    first = make_safetensors(tmp_path / "a.safetensors", tensors)
    second = make_safetensors(tmp_path / "b.safetensors", tensors)
    report = ziptensors.pack([first, second], tmp_path / "set.ziptensors", threads=1)
    assert report.total.dedup_saved > 0
    assert report.total.raw == first.stat().st_size * 2
    assert report.members[1].blobs_reused > 0
    assert report.verify.bytes == report.total.raw
    assert report.classes[-1].dtype == "F32"


def test_pack_with_delta_base_stores_the_finetune_against_it(tmp_path: Path) -> None:
    data = compressible_bytes(7, 256 * 1024 * 4)
    # The lowest bit of every 14th byte flipped: a finetune with a sparse residual.
    tuned_data = bytearray(data)
    for at in range(0, len(tuned_data), 14):
        tuned_data[at] ^= 1
    base = make_safetensors(tmp_path / "base.safetensors", {"w": ("F32", [256, 1024], data)})
    tuned = make_safetensors(
        tmp_path / "tuned.safetensors", {"w": ("F32", [256, 1024], bytes(tuned_data))}
    )
    output = tmp_path / "set.ziptensors"
    report = ziptensors.pack([tuned, base], output, threads=2, delta_base="base.safetensors")
    assert [member.name for member in report.members] == ["base.safetensors", "tuned.safetensors"]
    assert [member.delta_blobs for member in report.members] == [0, 1]
    assert report.members[1].blobs_added >= report.members[1].delta_blobs

    container = Container(output)
    for path in (base, tuned):
        member = container.member(path.name)
        assert member.read(0, member.len) == path.read_bytes()
    alone = ziptensors.pack([tuned, base], tmp_path / "alone.ziptensors", threads=2)
    assert [member.delta_blobs for member in alone.members] == [0, 0]
    assert report.output_len < alone.output_len


def test_pack_refuses_a_delta_base_it_cannot_use(tmp_path: Path, source: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    output = out_dir / "set.ziptensors"
    with pytest.raises(ValueError, match="member name of no input"):
        ziptensors.pack([source], output, delta_base="absent")
    with pytest.raises(ValueError, match="needs mode"):
        ziptensors.pack([source], output, mode="balanced", delta_base=source.name)
    assert not list(out_dir.iterdir())


def test_failed_pack_leaves_nothing(tmp_path: Path, source: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(InputError):
        ziptensors.pack([(source, "../escape")], out_dir / "set.ziptensors", threads=1)
    assert not list(out_dir.iterdir())


def test_unknown_mode_rejected(tmp_path: Path, source: Path) -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        ziptensors.encode_file(source, tmp_path / "x", mode="ultra")
    with pytest.raises(ValueError, match="unknown mode"):
        ziptensors.pack([source], tmp_path / "x", mode="ultra")


def test_pack_rejects_malformed_inputs(tmp_path: Path, source: Path) -> None:
    with pytest.raises(TypeError):
        ziptensors.pack(str(source), tmp_path / "x")
    with pytest.raises(TypeError):
        ziptensors.pack([(source, 3)], tmp_path / "x")


def test_gguf_shape_keeps_file_order(tmp_path: Path) -> None:
    data = seeded_bytes(4, 6 * 4)
    source = make_gguf(tmp_path / "model.gguf", "weight", [3, 2], data)
    output = tmp_path / "model.ziptensors"
    ziptensors.encode_file(source, output, threads=1)
    member = Container(output).member()
    tensor = member.tensor_info("weight")
    assert tensor.shape == [3, 2]
    assert tensor.dtype == "F32"
    target = bytearray(tensor.len)
    member.decode_tensor_into("weight", target)
    assert target == data


def damaged(path: Path, offset: int, xor: int = 0xFF) -> Path:
    """Flips the bits of `xor` in one byte of `path`, in place. Negative offsets count from the end."""
    data = bytearray(path.read_bytes())
    data[offset] ^= xor
    path.write_bytes(data)
    return path


def test_every_family_derives_from_the_base() -> None:
    families = (
        ZiptensorsIoError,
        FormatError,
        UnsupportedError,
        BudgetError,
        CorruptionError,
        RangeError,
        NotFoundError,
        InputError,
    )
    for family in families:
        assert issubclass(family, ZiptensorsError)
        assert family.__module__ == "ziptensors"
    assert issubclass(ZiptensorsError, Exception)


def test_not_durable_is_an_io_error() -> None:
    # The core raises it only when a directory sync fails after the rename, which a test cannot
    # provoke without privileges; its injected path is tested in `io/temp.rs`.
    assert issubclass(NotDurableError, ZiptensorsIoError)
    assert NotDurableError.__module__ == "ziptensors"


def test_io_error(tmp_path: Path) -> None:
    with pytest.raises(ZiptensorsIoError) as caught:
        Container(tmp_path / "missing")
    assert caught.value.code == CODE_IO
    assert "missing" in str(caught.value)


def test_not_ziptensors(tmp_path: Path) -> None:
    path = tmp_path / "junk"
    path.write_bytes(seeded_bytes(5, 4096))
    with pytest.raises(FormatError) as caught:
        Container(path)
    assert caught.value.code == CODE_NOT_ZIPTENSORS


def test_malformed_footer(encoded: Path) -> None:
    with pytest.raises(FormatError) as caught:
        Container(damaged(encoded, -FOOTER_LEN))
    assert caught.value.code == CODE_MALFORMED_INDEX


def test_unsupported_version(encoded: Path) -> None:
    xxhash = pytest.importorskip("xxhash")
    data = bytearray(encoded.read_bytes())
    data[HEADER_MAJOR_OFFSET] ^= 1
    digest = xxhash.xxh3_64_digest(bytes(data[:HEADER_HASH_OFFSET]))
    data[HEADER_HASH_OFFSET:HEADER_LEN] = digest[::-1]
    encoded.write_bytes(data)
    with pytest.raises(UnsupportedError) as caught:
        Container(encoded)
    assert caught.value.code == CODE_UNSUPPORTED_VERSION


def test_budget(encoded: Path) -> None:
    with pytest.raises(BudgetError) as caught:
        Container(encoded, max_total_decoded=1)
    assert caught.value.code == CODE_BUDGET_EXCEEDED


def test_budget_bounds_a_restore(tmp_path: Path, encoded: Path, source: Path) -> None:
    size = source.stat().st_size
    with pytest.raises(BudgetError):
        Container(encoded, max_total_decoded=size - 1)
    out = tmp_path / "within-budget.bin"
    Container(encoded, max_total_decoded=size).member().restore_file(out)
    assert out.read_bytes() == source.read_bytes()


def test_output_that_is_the_input_is_refused(encoded: Path, source: Path) -> None:
    before = encoded.read_bytes()
    with pytest.raises(InputError):
        Container(encoded).member().restore_file(encoded)
    assert encoded.read_bytes() == before
    with pytest.raises(InputError):
        ziptensors.encode_file(source, source)


def test_corrupted_chunk(tmp_path: Path) -> None:
    """A tensor of random bytes is stored, so a flipped data byte surfaces as a checksum failure."""
    source = make_safetensors(
        tmp_path / "noise.safetensors",
        {"noise": ("F32", [1 << 16], seeded_bytes(6, 1 << 18))},
    )
    output = tmp_path / "noise.ziptensors"
    ziptensors.encode_file(source, output, threads=1)
    member = Container(damaged(output, HEADER_LEN + (1 << 17))).member()
    with pytest.raises(CorruptionError) as caught:
        member.read_at(0, bytearray(member.len))
    assert caught.value.code == CODE_CHECKSUM_MISMATCH
    with pytest.raises(CorruptionError):
        member.verify()


def test_not_found(encoded: Path) -> None:
    container = Container(encoded)
    member = container.member()
    with pytest.raises(NotFoundError) as caught:
        member.decode_tensor_into("absent", bytearray(1))
    assert caught.value.code == CODE_TENSOR_NOT_FOUND
    with pytest.raises(NotFoundError) as caught:
        member.tensor_info("absent")
    assert caught.value.code == CODE_TENSOR_NOT_FOUND
    with pytest.raises(NotFoundError) as caught:
        container.member("absent")
    assert caught.value.code == CODE_MEMBER_NOT_FOUND


def test_unsafe_member_name(tmp_path: Path, source: Path) -> None:
    with pytest.raises(InputError) as caught:
        ziptensors.pack([(source, "../escape")], tmp_path / "x", threads=1)
    assert caught.value.code == CODE_UNSAFE_MEMBER_NAME


def test_symlink_input(tmp_path: Path, source: Path) -> None:
    link = tmp_path / "link"
    try:
        link.symlink_to(source)
    except OSError as error:
        pytest.skip(f"symbolic links are not available here: {error}")
    with pytest.raises(InputError) as caught:
        ziptensors.encode_file(link, tmp_path / "x", threads=1)
    assert caught.value.code == CODE_SYMLINK_INPUT


def stall_during(action: Callable[[], object]) -> tuple[float, float]:
    """Runs `action` beside a thread that needs the GIL every millisecond.

    Returns the longest time that thread made no progress during `action`, and the duration of
    `action`, both in seconds. Code that keeps the GIL throughout stalls the thread for all of it.
    """
    ticks: list[float] = []
    stop = threading.Event()

    def ticker() -> None:
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.001)

    thread = threading.Thread(target=ticker)
    thread.start()
    while not ticks:
        time.sleep(0.001)
    start = time.perf_counter()
    try:
        action()
        end = time.perf_counter()
    finally:
        # A failing action must still stop the ticker, or the interpreter waits for it at exit.
        stop.set()
        thread.join()
    moments = [start, *(tick for tick in ticks if start < tick < end), end]
    return max(after - before for before, after in zip(moments, moments[1:])), end - start


# A decode must last this long for a stall to stand out from scheduling noise.
SLOW_SECONDS = 0.15
SLOW_START_BYTES = 16 << 20
SLOW_MAX_BYTES = 2 << 30


@pytest.fixture(name="slow", scope="module")
def fixture_slow(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A source and its container whose single-threaded decode takes at least `SLOW_SECONDS`.

    The size is found by doubling, since a release build decodes a hundred times faster than a
    debug build.
    """
    root = tmp_path_factory.mktemp("slow")
    size = SLOW_START_BYTES
    while True:
        source = make_safetensors(
            root / "slow.safetensors",
            {"slow": ("F32", [size // 4], compressible_bytes(7, size))},
        )
        output = root / "slow.ziptensors"
        ziptensors.encode_file(source, output, mode="fast")
        member = Container(output, threads=1).member()
        buffer = bytearray(member.len)
        start = time.perf_counter()
        member.read_at(0, buffer)
        elapsed = time.perf_counter() - start
        log.info("slow fixture: %d bytes decode in %.3f s", size, elapsed)
        if elapsed >= SLOW_SECONDS:
            return source, output
        assert size < SLOW_MAX_BYTES, f"{size} bytes decode in {elapsed} s, too fast to measure"
        size *= 2


def test_stall_measure_detects_a_held_gil() -> None:
    """The control: without it, the tests below could pass on a measure that detects nothing."""
    stall, duration = stall_during(lambda: 7**2_000_000)
    assert duration >= 0.05
    assert stall > duration * 0.8


def gil_free_actions(source: Path, output: Path, scratch: Path) -> dict[str, Callable[[], object]]:
    """The slow calls of the module, by name."""
    member = Container(output, threads=1).member()
    buffer = bytearray(member.len)
    buffer_of_tensor = bytearray(member.tensor_info("slow").len)
    return {
        "read_at": lambda: member.read_at(0, buffer),
        "decode_tensor_into": lambda: member.decode_tensor_into("slow", buffer_of_tensor),
        "verify": member.verify,
        "restore_file": lambda: member.restore_file(scratch / "restored"),
        "encode_file": lambda: ziptensors.encode_file(source, scratch / "encoded", threads=1),
        "pack": lambda: ziptensors.pack([source], scratch / "packed", mode="balanced", threads=1),
    }


@pytest.mark.parametrize(
    "name",
    ["read_at", "decode_tensor_into", "verify", "restore_file", "encode_file", "pack"],
)
def test_gil_released(name: str, slow: tuple[Path, Path], tmp_path: Path) -> None:
    source, output = slow
    action = gil_free_actions(source, output, tmp_path)[name]
    stall, duration = stall_during(action)
    assert duration >= SLOW_SECONDS / 2
    assert stall < duration / 2


def test_read_releases_the_gil_while_decoding(slow: tuple[Path, Path]) -> None:
    """Zeroing the new `bytes` keeps the GIL, a small part of the call, so the bound is looser."""
    member = Container(slow[1], threads=1).member()
    stall, duration = stall_during(lambda: member.read(0, member.len))
    assert duration >= SLOW_SECONDS / 2
    assert stall < duration * 0.75


def test_stub_names_exist_at_runtime() -> None:
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            assert hasattr(ziptensors, node.name), node.name
        elif isinstance(node, ast.ClassDef):
            runtime = getattr(ziptensors, node.name)
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    assert hasattr(runtime, item.name), f"{node.name}.{item.name}"


def test_runtime_names_are_in_the_stub() -> None:
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    declared = {node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    public = {name for name in dir(ziptensors) if not name.startswith("_")}
    # The package maturin generates holds the extension as a submodule of the same name, and
    # `torch` and `safetensors` are the pure-Python submodules, here once any test imported them.
    assert public - {"ziptensors", "torch", "safetensors"} == declared


def runtime_callables() -> list[tuple[str, Callable[..., object]]]:
    """Every public function of the module and every public method of its public classes."""
    found: list[tuple[str, Callable[..., object]]] = []
    for name in dir(ziptensors):
        value = getattr(ziptensors, name)
        if name.startswith("_") or inspect.ismodule(value):
            continue
        if inspect.isclass(value):
            for member in ["__init__", "__new__", *dir(value)]:
                if member.startswith("_") and member not in ("__init__", "__new__"):
                    continue
                method = getattr(value, member, None)
                if callable(method) and not inspect.isclass(method):
                    found.append((f"{name}.{member}", method))
        elif callable(value):
            found.append((name, value))
    return found


def test_signatures_show_real_defaults() -> None:
    """pyo3 renders a default that is not a Python literal as `...`, which tells a caller nothing."""
    checked = 0
    for name, function in runtime_callables():
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            # Builtins of `object` and exceptions have no text signature; nothing of ours to check.
            continue
        checked += 1
        for parameter in signature.parameters.values():
            assert parameter.default is not Ellipsis, f"{name}({parameter.name}=...)"
    assert checked > 0
    assert inspect.signature(ziptensors.recover).parameters["max_candidates"].default == 64


def test_a_member_name_given_twice_is_an_input_error(tmp_path: Path, source: Path) -> None:
    """A duplicate in a write request is the caller's, not damage: `InputError`, code 26."""
    output = tmp_path / "twice.ziptensors"
    with pytest.raises(InputError) as caught:
        ziptensors.pack([(source, "m"), (source, "m")], output, threads=1)
    assert caught.value.code == CODE_DUPLICATE_MEMBER
    assert '"m"' in str(caught.value)
    assert not output.exists(), "pack checks every name before it starts the output"

    with PackBuilder(output, threads=1) as builder:
        builder.add_file(source, "m")
        with pytest.raises(InputError) as caught:
            builder.add_file(source, "m")
        assert caught.value.code == CODE_DUPLICATE_MEMBER
    assert not output.exists()


def test_pack_accepts_path_like_and_str(tmp_path: Path, source: Path) -> None:
    ziptensors.pack([os.fspath(source)], os.fspath(tmp_path / "s.ziptensors"), threads=1)
    assert [info.name for info in Container(tmp_path / "s.ziptensors").members] == [source.name]
