"""File: python/data/platlib/ziptensors/safetensors.py
Purpose: `safe_open`, the reading API of the `safetensors` package, over a member of a container.

Main responsibilities: parse and validate the safetensors header from the member's own bytes, list
keys and metadata, and read a whole tensor or a slice of rows by byte offset with `Member.read_at`,
into torch tensors or numpy arrays, on the host or through pinned staging onto a CUDA or ROCm device.

Key structures: `safe_open`, `SafeSlice`, `SafetensorError`. Key functions: `_parse_header`,
`_slice_plan`.

Implementation notes: the header is parsed here, not taken from the core's tensor table: the table is
a parser's hint and drops `__metadata__`, while the member is the original file byte for byte. The
validation bounds every allocation by the member length before it happens. The dtype allowlist, the
host fill and the device stager are those of `ziptensors.torch`, so both readers accept the same
dtypes. torch and numpy are imported only by the framework that needs them. `SafetensorError` is not
the class of the `safetensors` package, which this module does not import.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
import logging
import operator
from os import PathLike
import sys
from types import ModuleType, TracebackType
from typing import TYPE_CHECKING, Any, NamedTuple

import ziptensors
from ziptensors.torch import (
    _SAFETENSORS_DTYPES,
    TorchLoadError,
    _DeviceStager,
    _element_count,
    _fill_host,
    _import_torch,
    _staging_bytes,
    _torch_dtype,
)

if TYPE_CHECKING:
    import numpy
    import torch

log = logging.getLogger(__name__)

# The header limit of the safetensors package, which also bounds the one read made before validation.
_MAX_HEADER_BYTES = 100_000_000
# Every dtype tag the safetensors package (0.7) parses. A header with another tag is invalid, as it is
# there; a tag in this set but outside `_SAFETENSORS_DTYPES` (C64, F8_E8M0, F4, ...) is a valid file
# whose tensor is listed and refused on read, because the core registry does not know the dtype.
_SAFETENSORS_TAGS = frozenset(
    (
        "BOOL", "F4", "F6_E2M3", "F6_E3M2", "U8", "I8", "F8_E5M2", "F8_E4M3", "F8_E8M0", "I16",
        "U16", "F16", "BF16", "I32", "U32", "F32", "C64", "F64", "I64", "U64",
    )
)
_LENGTH_PREFIX = 8
_TORCH_FRAMEWORKS = ("pt", "torch", "pytorch")
_NUMPY_FRAMEWORKS = ("np", "numpy")


class SafetensorError(ValueError):
    """The header is not a valid safetensors header, the file is closed, a name is absent, or an
    argument names a framework or device this reader does not take."""


class _Entry(NamedTuple):
    """One validated header entry. `begin` and `end` are byte offsets in the member."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int


class _Header(NamedTuple):
    """A validated header: entries by name, and `__metadata__` if the file has one."""

    entries: dict[str, _Entry]
    metadata: dict[str, str] | None


def _int_list(value: object) -> list[int] | None:
    """Returns `value` if it is a list of non-negative integers (bool excluded), else None."""
    if not isinstance(value, list):
        return None
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            return None
    return value


def _parse_entry(name: str, info: object, data_start: int) -> _Entry:
    """Validates one header entry and returns it with member offsets. Raises SafetensorError."""
    if not isinstance(info, dict):
        raise SafetensorError(f"header entry {name!r} is not an object")
    dtype = info.get("dtype")
    shape = _int_list(info.get("shape"))
    offsets = _int_list(info.get("data_offsets"))
    if not isinstance(dtype, str) or shape is None or offsets is None or len(offsets) != 2:
        raise SafetensorError(
            f"header entry {name!r} needs a string `dtype`, a `shape` of non-negative integers "
            "and two non-negative `data_offsets`"
        )
    if dtype not in _SAFETENSORS_TAGS:
        raise SafetensorError(f"header entry {name!r}: {dtype!r} is not a safetensors dtype")
    begin, end = offsets
    if begin > end:
        raise SafetensorError(f"header entry {name!r}: data_offsets {offsets} run backwards")
    info_of_dtype = _SAFETENSORS_DTYPES.get(dtype)
    # A valid tag outside the allowlist cannot be read, so its length is not checked here (sub-byte
    # tags have no whole item size); its range is, below, and nothing is ever allocated for it.
    if info_of_dtype is not None:
        expected = _element_count(shape) * info_of_dtype.item_bytes
        if expected != end - begin:
            raise SafetensorError(
                f"header entry {name!r}: shape {shape} of {dtype} needs {expected} bytes and "
                f"data_offsets {offsets} span {end - begin}"
            )
    return _Entry(name, dtype, tuple(shape), data_start + begin, data_start + end)


def _parse_header(member: ziptensors.Member) -> _Header:
    """Reads and validates the safetensors header at the start of `member`.

    The checks are those of the safetensors package: a length prefix within 100 MB and the member,
    UTF-8 JSON of an object, `__metadata__` a map of strings, every entry well formed with a known
    dtype tag, and the entries, in offset order, tiling the data section exactly from its start to
    the member's end. Raises SafetensorError for any violation, naming the member's profile when the
    core did not recognize it as safetensors (a hint only: the bytes decide), and the errors of
    `Member.read`.
    """
    try:
        return _parse_header_bytes(member)
    except SafetensorError as error:
        if member.profile == "safetensors":
            raise
        raise SafetensorError(
            f"{error}; the member was recognized as {member.profile}, not safetensors"
        ) from error


def _parse_header_bytes(member: ziptensors.Member) -> _Header:
    """The checks of `_parse_header`, without the profile hint."""
    if member.len < _LENGTH_PREFIX:
        raise SafetensorError(f"{member.name!r}: {member.len} bytes hold no safetensors header")
    header_len = int.from_bytes(member.read(0, _LENGTH_PREFIX), "little")
    if header_len > _MAX_HEADER_BYTES:
        raise SafetensorError(f"{member.name!r}: header of {header_len} bytes is over 100 MB")
    data_start = _LENGTH_PREFIX + header_len
    if data_start > member.len:
        raise SafetensorError(
            f"{member.name!r}: header of {header_len} bytes runs past the member, {member.len} bytes"
        )
    try:
        parsed = json.loads(member.read(_LENGTH_PREFIX, header_len).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise SafetensorError(f"{member.name!r}: header is not UTF-8 JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise SafetensorError(f"{member.name!r}: header is not a JSON object")
    metadata = parsed.pop("__metadata__", None)
    if metadata is not None and not (
        isinstance(metadata, dict)
        and all(isinstance(value, str) for value in metadata.values())
    ):
        raise SafetensorError(f"{member.name!r}: __metadata__ is not a map of strings")
    entries = {name: _parse_entry(name, info, data_start) for name, info in parsed.items()}
    cursor = data_start
    for entry in sorted(entries.values(), key=lambda item: (item.begin, item.end, item.name)):
        if entry.begin != cursor:
            raise SafetensorError(
                f"{member.name!r}: tensor {entry.name!r} starts at data offset "
                f"{entry.begin - data_start}, where {cursor - data_start} was expected"
            )
        cursor = entry.end
    if cursor != member.len:
        raise SafetensorError(
            f"{member.name!r}: the tensors end at data offset {cursor - data_start} and the data "
            f"section holds {member.len - data_start} bytes"
        )
    return _Header(entries, metadata)


class _SliceItem(NamedTuple):
    """One normalized index entry: a new axis, an integer, or a slice with a positive step."""

    kind: str
    start: int
    stop: int
    step: int


_NEW_AXIS = _SliceItem("new", 0, 0, 1)


def _normalize_index(index: object, shape: tuple[int, ...]) -> list[_SliceItem]:
    """Returns `index` as one item per dimension of `shape`, with new axes in place.

    Takes what basic indexing takes in the safetensors package: integers (negative counts from the
    end), slices with a positive step (clamped), `...` and None, alone or in a tuple. Raises
    IndexError for an integer out of range, more indices than dimensions or two ellipses,
    ValueError for a step below 1, and TypeError for anything else, such as a list.
    """
    items = index if isinstance(index, tuple) else (index,)
    if sum(item is Ellipsis for item in items) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    consumed = sum(item is not None and item is not Ellipsis for item in items)
    if consumed > len(shape):
        raise IndexError(f"too many indices: {consumed} for {len(shape)} dimensions")
    expanded: list[object] = []
    for item in items:
        if item is Ellipsis:
            expanded.extend([slice(None)] * (len(shape) - consumed))
        else:
            expanded.append(item)
    if Ellipsis not in items:
        expanded.extend([slice(None)] * (len(shape) - consumed))
    normalized: list[_SliceItem] = []
    dim = 0
    for item in expanded:
        if item is None:
            normalized.append(_NEW_AXIS)
            continue
        size = shape[dim]
        dim += 1
        if isinstance(item, slice):
            if item.step is not None and operator.index(item.step) < 1:
                raise ValueError(f"slice step must be positive, got {item.step}")
            start, stop, step = item.indices(size)
            normalized.append(_SliceItem("slice", start, max(stop, start), step))
            continue
        if isinstance(item, bool):
            raise TypeError("a bool is not a tensor index")
        try:
            position = operator.index(item)
        except TypeError as error:
            raise TypeError(
                f"index {item!r} is not an integer, slice, None or ...; lists and tensors are not "
                "taken"
            ) from error
        if not -size <= position < size:
            raise IndexError(f"index {position} is out of range for a dimension of {size}")
        normalized.append(_SliceItem("int", position % size, 0, 1))
    return normalized


class _SlicePlan(NamedTuple):
    """What to read for a slice: `count` elements from element `first`, as an array of `shape`,
    then indexed with `local` to give the result."""

    first: int
    count: int
    shape: tuple[int, ...]
    local: tuple[Any, ...]


def _slice_plan(index: object, shape: tuple[int, ...]) -> _SlicePlan:
    """Returns the one contiguous element range a basic index of `shape` needs, and the index that
    finishes the job on it.

    The leading integers and the first slice after them select a run of rows; the dimensions after
    it are read whole. A step above 1 reads the bounding range, so the extra read is less than one
    row of that dimension per element kept. Raises what `_normalize_index` raises.
    """
    normalized = _normalize_index(index, shape)
    strides = [_element_count(shape[dim + 1 :]) for dim in range(len(shape))]
    first = 0
    dim = 0
    local: list[Any] = []
    sub_shape: tuple[int, ...] = ()
    rows_open = True
    for item in normalized:
        if item.kind == "new":
            local.append(None)
            continue
        if rows_open and item.kind == "int":
            first += item.start * strides[dim]
        elif rows_open:
            rows_open = False
            kept = len(range(item.start, item.stop, item.step))
            span = 0 if kept == 0 else (kept - 1) * item.step + 1
            first += item.start * strides[dim] if kept else 0
            sub_shape = (span, *shape[dim + 1 :])
            local.append(slice(0, span, item.step))
        elif item.kind == "int":
            local.append(item.start)
        else:
            local.append(slice(item.start, item.stop, item.step))
        dim += 1
    return _SlicePlan(first, _element_count(sub_shape), sub_shape, tuple(local))


class _Reader:
    """Allocates and fills host arrays of one framework for header entries.

    The base serves torch; `_NumpyReader` overrides the framework calls.
    """

    def __init__(self, member: ziptensors.Member, device: object) -> None:
        """Binds `member` and parses `device`. Raises SafetensorError for a device it refuses."""
        self.member = member
        self._torch = _import_torch()
        self.target = self._parse_device(device)

    def _parse_device(self, device: object) -> torch.device:
        """Returns `device` as a torch device: an int N is `cuda:N`; `meta` and junk are refused."""
        torch_module = self._torch
        if isinstance(device, int) and not isinstance(device, bool):
            device = f"cuda:{device}"
        try:
            target = torch_module.device(device)
        except (RuntimeError, TypeError) as error:
            raise SafetensorError(f"device {device!r} is not a device: {error}") from error
        if target.type == "meta":
            raise SafetensorError("device `meta` would discard the data")
        return target

    def dtype(self, entry: _Entry) -> torch.dtype:
        """The torch dtype of `entry`, or TorchLoadError for a dtype outside the allowlist."""
        where = f"tensor {entry.name!r} of {self.member.name!r}"
        info = _SAFETENSORS_DTYPES.get(entry.dtype)
        if info is None:
            raise TorchLoadError(
                f"{where}: dtype {entry.dtype} has no torch dtype this reader accepts"
            )
        return _torch_dtype(self._torch, info.torch_name, where)

    def read_host(self, entry: _Entry, element: int, shape: tuple[int, ...]) -> Any:
        """Returns a new host array of `shape` holding the elements of `entry` from `element` on.

        The caller proved that the range lies inside the entry.
        """
        dtype = self.dtype(entry)
        tensor = self._torch.empty(shape, dtype=dtype)
        offset = entry.begin + element * tensor.element_size()
        _fill_host(self._torch, self.member, offset, tensor, f"tensor {entry.name!r}")
        return tensor

    def finish(self, array: Any, source: Any) -> Any:
        """Returns `array`, indexed out of `source`, as its own compact array on the target."""
        if array.numel() < source.numel():
            array = array.clone()
        elif not array.is_contiguous():
            array = array.contiguous()
        return array if self.target.type == "cpu" else array.to(self.target)

    def read_entries(self, entries: Sequence[_Entry]) -> list[Any]:
        """Returns whole tensors of `entries` on the target, in order.

        A `cuda` target goes through one pinned staging pipeline for all of them; any other target
        decodes each on the host and moves it.
        """
        torch_module = self._torch
        if self.target.type != "cuda":
            return [
                self.finish(whole, whole)
                for whole in (self.read_host(entry, 0, entry.shape) for entry in entries)
            ]
        dtypes = [self.dtype(entry) for entry in entries]
        total = sum(entry.end - entry.begin for entry in entries)
        capacity = min(_staging_bytes(None), max(total, 1))
        tensors = []
        with _DeviceStager(torch_module, self.member, self.target, capacity) as stager:
            for entry, dtype in zip(entries, dtypes, strict=True):
                tensor = torch_module.empty(entry.shape, dtype=dtype, device=self.target)
                stager.fill(tensor, entry.begin, f"tensor {entry.name!r}")
                tensors.append(tensor)
        return tensors


class _NumpyReader(_Reader):
    """`_Reader` for numpy arrays, which live on the host only."""

    def __init__(self, member: ziptensors.Member, device: object) -> None:
        """Binds `member`. Raises SafetensorError for a device other than `cpu`, and ImportError
        without numpy."""
        self.member = member
        try:
            import numpy
        except ImportError as error:
            raise ImportError("framework 'numpy' needs numpy: install numpy") from error
        self._numpy: ModuleType = numpy
        # A torch device of type `cpu` is accepted without importing torch: only its `type` is read.
        if device != "cpu" and not (
            not isinstance(device, str) and getattr(device, "type", None) == "cpu"
        ):
            raise SafetensorError(f"framework 'numpy' reads to device 'cpu' only, not {device!r}")

    def dtype(self, entry: _Entry) -> numpy.dtype[Any]:
        """The numpy dtype of `entry`, or SafetensorError if numpy has none for it."""
        info = _SAFETENSORS_DTYPES.get(entry.dtype)
        if info is None or info.numpy_name is None:
            raise SafetensorError(
                f"tensor {entry.name!r}: dtype {entry.dtype} has no numpy dtype this reader accepts"
            )
        return self._numpy.dtype(info.numpy_name)

    def read_host(self, entry: _Entry, element: int, shape: tuple[int, ...]) -> Any:
        """See `_Reader.read_host`; a BOOL byte other than 0 or 1 raises SafetensorError."""
        dtype = self.dtype(entry)
        array = self._numpy.empty(shape, dtype=dtype)
        if array.nbytes:
            # numpy arrays export a writable, C-contiguous buffer, which `read_at` fills directly.
            self.member.read_at(entry.begin + element * dtype.itemsize, array)
            if dtype == self._numpy.bool_ and int(array.view(self._numpy.uint8).max()) > 1:
                raise SafetensorError(f"tensor {entry.name!r}: a BOOL byte is neither 0 nor 1")
        return array

    def finish(self, array: Any, source: Any) -> Any:
        """Returns `array`, indexed out of `source`, as its own compact array."""
        if array.size < source.size or not array.flags.c_contiguous:
            return self._numpy.ascontiguousarray(array).copy()
        return array

    def read_entries(self, entries: Sequence[_Entry]) -> list[Any]:
        """Returns whole arrays of `entries`, in order."""
        return [self.read_host(entry, 0, entry.shape) for entry in entries]


class SafeSlice:
    """A lazily read tensor of a `safe_open`, as `get_slice` returns it.

    Indexing reads only the contiguous run of rows the index needs, then applies the rest of the
    index in the framework. Valid while its `safe_open` is open: unlike the safetensors package,
    indexing it after `__exit__` raises SafetensorError("File is closed").
    """

    def __init__(self, owner: safe_open, entry: _Entry) -> None:
        """Binds the slice to `owner` and one header entry."""
        self._owner = owner
        self._entry = entry

    def get_shape(self) -> list[int]:
        """The shape of the whole tensor."""
        return list(self._entry.shape)

    def get_dtype(self) -> str:
        """The safetensors dtype tag, such as `F32`, also for a dtype this reader cannot read."""
        return self._entry.dtype

    def __getitem__(self, index: object) -> Any:
        """Returns the part of the tensor that a basic index selects, on the device of `safe_open`.

        The index takes integers, positive-step slices, `...` and None. Raises IndexError,
        ValueError or TypeError for an index `_normalize_index` refuses, SafetensorError once the
        file is closed, and the errors of `safe_open.get_tensor`.
        """
        reader = self._owner._live_reader()
        plan = _slice_plan(index, self._entry.shape)
        source = reader.read_host(self._entry, plan.first, plan.shape)
        return reader.finish(source[plan.local], source)


class safe_open:  # noqa: N801 - the name of the safetensors API it stands in for
    """Reads a safetensors member of a container as `safetensors.safe_open` reads a file.

    `filename` is a container path, whose member `member` names (None for a single-file container),
    or a `ziptensors.Member`. `threads` and `max_total_decoded` go to `ziptensors.Container` and must
    be None with a `Member`. `framework` is `pt` (also `torch`, `pytorch`) or `np` (also `numpy`).
    `device` is `cpu`, an int N for `cuda:N`, or a torch device string or object; numpy takes `cpu`
    only. On a `cuda` device (CUDA or ROCm) whole tensors go through the pinned staging pipeline of
    `ziptensors.torch.load_state_dict`, with `ziptensors.torch.STAGING_BYTES` per buffer; slices are
    read on the host and moved.

    Accepted dtypes are those of `ziptensors.torch.load_state_dict` for safetensors. A tensor of
    another valid safetensors dtype (C64, F8_E8M0, F4, ...) is listed and has a shape and dtype tag,
    and reading it raises TorchLoadError (pt) or SafetensorError (numpy); a tag that is no
    safetensors dtype at all makes the header invalid. The header is read and fully validated when
    the object is made.

    Usable with or without `with`; after `__exit__` every method raises SafetensorError("File is
    closed"), also on a `SafeSlice` taken earlier. Differences from safetensors 0.7: that slice still
    reads there; an index of a bool, float or list raises TypeError here (safetensors takes `True`
    as a new axis and raises IndexError for a float); a `torch.device` object is accepted here;
    numpy slices honour a step, which safetensors' numpy slices ignore.

    Errors: SafetensorError for an invalid header, an unknown framework or device, or an absent
    tensor; the errors of `Container` and `Container.member` for the path, and of
    `Member.read_at` for the data; ImportError without the framework's package.
    """

    def __init__(
        self,
        filename: str | PathLike[str] | ziptensors.Member,
        framework: str,
        device: str | int | torch.device = "cpu",
        *,
        member: str | None = None,
        threads: int | None = None,
        max_total_decoded: int | None = None,
    ) -> None:
        """Opens the member and validates its header. Raises TypeError for container options
        passed with a `Member`."""
        if framework not in _TORCH_FRAMEWORKS + _NUMPY_FRAMEWORKS:
            raise SafetensorError(
                f"framework {framework!r} is not one of {_TORCH_FRAMEWORKS + _NUMPY_FRAMEWORKS}"
            )
        if isinstance(filename, ziptensors.Member):
            if member is not None or threads is not None or max_total_decoded is not None:
                raise TypeError(
                    "`member`, `threads` and `max_total_decoded` apply to a container path, not "
                    "to a Member"
                )
            opened = filename
        else:
            container = ziptensors.Container(
                filename, threads=threads, max_total_decoded=max_total_decoded
            )
            opened = container.member(member)
        if sys.byteorder != "little":
            raise SafetensorError(f"{opened.name!r}: stored little-endian, and this host is not")
        reader: _Reader
        if framework in _TORCH_FRAMEWORKS:
            reader = _Reader(opened, device)
        else:
            reader = _NumpyReader(opened, device)
        self._header = _parse_header(opened)
        self._reader: _Reader | None = reader
        log.debug("opened %s with %d tensors", opened.name, len(self._header.entries))

    def __enter__(self) -> safe_open:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Closes the file: drops the member, after which every method raises."""
        self._reader = None

    def _live_reader(self) -> _Reader:
        """The reader, or SafetensorError("File is closed") after `__exit__`."""
        if self._reader is None:
            raise SafetensorError("File is closed")
        return self._reader

    def _entry(self, name: str) -> _Entry:
        """The header entry `name`, or SafetensorError if the file has none."""
        self._live_reader()
        entry = self._header.entries.get(name)
        if entry is None:
            raise SafetensorError(f"File does not contain tensor {name}")
        return entry

    def keys(self) -> list[str]:
        """Tensor names, sorted."""
        self._live_reader()
        return sorted(self._header.entries)

    def offset_keys(self) -> list[str]:
        """Tensor names in the order of their data in the file."""
        self._live_reader()
        ordered = sorted(self._header.entries.values(), key=lambda item: (item.begin, item.end))
        return [entry.name for entry in ordered]

    def metadata(self) -> dict[str, str] | None:
        """A copy of `__metadata__`, or None if the header has none."""
        self._live_reader()
        metadata = self._header.metadata
        return None if metadata is None else dict(metadata)

    def get_tensor(self, name: str) -> Any:
        """The whole tensor `name` on the device, decoded into fresh memory."""
        reader = self._live_reader()
        return reader.read_entries([self._entry(name)])[0]

    def get_tensors(self) -> dict[str, Any]:
        """Every tensor, in offset order, through one staging pipeline on a `cuda` device."""
        reader = self._live_reader()
        names = self.offset_keys()
        tensors = reader.read_entries([self._header.entries[name] for name in names])
        return dict(zip(names, tensors, strict=True))

    def get_slice(self, name: str) -> SafeSlice:
        """A `SafeSlice` of tensor `name`, which reads nothing until indexed."""
        return SafeSlice(self, self._entry(name))
