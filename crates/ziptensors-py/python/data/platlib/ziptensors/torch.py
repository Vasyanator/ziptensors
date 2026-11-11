"""File: python/data/platlib/ziptensors/torch.py
Purpose: load the tensors of a member into PyTorch without a file on disk and without a monkey patch.

Main responsibilities: decide whether a tensor of the table has a real torch dtype, or is a quantized
ggml tensor to hand over as raw blocks; allocate it uninitialized; fill its storage from the member
with the GIL released, in host memory or through pinned staging buffers onto a CUDA or ROCm device;
and hand the result to a module.

Key functions: `load_state_dict`, `load_into`. Key structures: `TorchLoadError`, `GgmlTensor`, the
dtype allowlists, `_DeviceStager`. Block geometry of quantized types is the core's, `Tensor.block`.

Implementation notes: torch is optional and imported only when one of these functions runs, so
`import ziptensors.torch` works without it. This file is named after the library it adapts, and
`import torch` below still finds the library: imports are absolute. The tensor table is a hint made by
a parser, so everything taken from it is checked here before an allocation. `ziptensors.safetensors`
shares the dtype allowlist, the host fill and the stager of this file. The contract is in
`python/MODULE_README.md` and `src/MODULE_README.md` of this crate.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import ctypes
from dataclasses import dataclass
import logging
import sys
from types import ModuleType, TracebackType
from typing import TYPE_CHECKING, Literal, NamedTuple

import ziptensors

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

# Bytes of each of the two pinned host buffers of a load onto a CUDA or ROCm device, unless the call
# passes `staging_bytes`. Read at call time, so a test may lower it.
STAGING_BYTES = 64 << 20


class _DtypeInfo(NamedTuple):
    """How one registry dtype maps to a framework: torch attribute, numpy name, bytes per element."""

    torch_name: str
    numpy_name: str | None
    item_bytes: int


# Registry dtype to its framework dtypes. An allowlist: whatever is absent here is refused, so a new
# registry entry can never be reinterpreted by accident. It is the safetensors torch map restricted to
# what the core registry knows: `C64`, `F8_E4M3FNUZ`, `F8_E8M0` and the like reach Python as
# `UNKNOWN(0x0000)` and stay refused. The safetensors tags equal the registry names, which is what lets
# `ziptensors.safetensors` use this table for header dtypes. numpy has no bfloat16 or float8.
_SAFETENSORS_DTYPES: dict[str, _DtypeInfo] = {
    "F64": _DtypeInfo("float64", "float64", 8),
    "F32": _DtypeInfo("float32", "float32", 4),
    "F16": _DtypeInfo("float16", "float16", 2),
    "BF16": _DtypeInfo("bfloat16", None, 2),
    "F8_E4M3": _DtypeInfo("float8_e4m3fn", None, 1),
    "F8_E5M2": _DtypeInfo("float8_e5m2", None, 1),
    "I64": _DtypeInfo("int64", "int64", 8),
    "I32": _DtypeInfo("int32", "int32", 4),
    "I16": _DtypeInfo("int16", "int16", 2),
    "I8": _DtypeInfo("int8", "int8", 1),
    "U64": _DtypeInfo("uint64", "uint64", 8),
    "U32": _DtypeInfo("uint32", "uint32", 4),
    "U16": _DtypeInfo("uint16", "uint16", 2),
    "U8": _DtypeInfo("uint8", "uint8", 1),
    "BOOL": _DtypeInfo("bool", "bool", 1),
}
# GGUF is limited to floats by contract: its integer tensors are not model weights in any file seen.
_GGUF_DTYPES: dict[str, _DtypeInfo] = {
    name: _SAFETENSORS_DTYPES[name] for name in ("F64", "F32", "F16", "BF16")
}


Quantized = Literal["refuse", "raw", "diffusers"]
_QUANTIZED_MODES: tuple[str, ...] = ("refuse", "raw", "diffusers")

# The tensor count of a GGUF header, a little-endian u64 after the magic and the u32 version.
_GGUF_TENSOR_COUNT_OFFSET = 8
_GGUF_TENSOR_COUNT_LEN = 8


class TorchLoadError(ValueError):
    """A member or a tensor cannot be turned into torch tensors without changing its meaning."""


# `eq=False`: a generated `__eq__` would compare the tensor field and raise on its ambiguous truth.
@dataclass(frozen=True, slots=True, eq=False)
class GgmlTensor:
    """The raw blocks of one quantized GGUF tensor, as `load_state_dict(quantized="raw")` returns it.

    `data` is a contiguous torch.uint8 tensor in the byte shape gguf-py gives: the logical shape with
    its last dimension replaced by `last // block_elements * block_bytes`. `ggml_type` is the registry
    name, which is also the name of `gguf.GGMLQuantizationType`, such as `Q8_0`. `shape` is the logical
    element shape in torch order, the shape of the weight after dequantization.
    """

    data: torch.Tensor
    ggml_type: str
    shape: tuple[int, ...]


class _Planned(NamedTuple):
    """One tensor that passed every check.

    `dtype` is the torch attribute name to allocate, `shape` the allocation shape in torch order,
    `offset` and `length` the byte range in the member. For a quantized tensor `ggml_type` is its
    registry name, `dtype` is `uint8`, `shape` the byte shape and `logical_shape` the element shape;
    otherwise `ggml_type` is None and `logical_shape` equals `shape`.
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    length: int
    ggml_type: str | None
    logical_shape: tuple[int, ...]


def _import_torch() -> ModuleType:
    """Returns the torch module. Raises ImportError with what to install if it is missing."""
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "ziptensors.torch needs PyTorch: install torch, or the `torch` extra of ziptensors"
        ) from error
    return torch


def _element_count(shape: Iterable[int]) -> int:
    """Returns the product of `shape`. Python integers cannot overflow here."""
    count = 1
    for dim in shape:
        count *= dim
    return count


def _torch_dtype(torch_module: ModuleType, name: str, where: str) -> torch.dtype:
    """Returns `torch.<name>`, or raises TorchLoadError naming `where` if this torch lacks it.

    float8 arrived in torch 2.1 and uint16..uint64 in 2.3, after the oldest torch the extra allows.
    """
    dtype = getattr(torch_module, name, None)
    if dtype is None:
        raise TorchLoadError(
            f"{where}: torch {torch_module.__version__} has no torch.{name}; a newer torch loads it"
        )
    return dtype


def _check_bool(torch_module: ModuleType, view: ctypes.Array[ctypes.c_char], where: str) -> None:
    """Raises TorchLoadError if a byte of `view`, about to become torch.bool, is neither 0 nor 1.

    A torch.bool byte other than 0 or 1 breaks torch's own assumptions, and nothing else in the
    chain looks at the values.
    """
    highest = int(torch_module.frombuffer(view, dtype=torch_module.uint8).max())
    if highest > 1:
        raise TorchLoadError(f"{where}: BOOL byte {highest} is neither 0 nor 1")


def _torch_shape(profile: str, shape: list[int]) -> tuple[int, ...]:
    """Returns `shape` of the table in torch order.

    GGUF states dimensions fastest varying first and torch slowest first. The bytes are the same, so
    reversing the list is the whole mapping: nothing is transposed.
    """
    return tuple(reversed(shape)) if profile == "gguf" else tuple(shape)


def _check_gguf_table(member: ziptensors.Member) -> None:
    """Raises TorchLoadError if the GGUF header declares tensors the table does not list.

    The parser keeps no row for a ggml type without a registry dtype, such as `Q4_0`, so without this
    check a whole-member load would silently return a partial state dict.
    """
    header = member.read(_GGUF_TENSOR_COUNT_OFFSET, _GGUF_TENSOR_COUNT_LEN)
    declared = int.from_bytes(header, "little")
    listed = len(member.tensors)
    if declared != listed:
        raise TorchLoadError(
            f"{member.name!r}: the GGUF header declares {declared} tensors and the table lists "
            f"{listed}; the others have a ggml type ziptensors does not know, usually quantized, "
            "so loading the rest would silently drop them. Pass `keys` to load only named tensors."
        )


def _quantized_block(profile: str, tensor: ziptensors.Tensor) -> tuple[int, int] | None:
    """Returns `(elements, bytes)` of a block if `tensor` is a block-quantized GGUF row, else None.

    The geometry is the core's (`Tensor.block`); a block of more than one element is what makes a
    type quantized. Plain GGUF types have blocks of one element and go through the dtype allowlist.
    """
    if profile != "gguf" or tensor.block is None:
        return None
    return tensor.block if tensor.block[0] > 1 else None


def _ggml_shapes(
    where: str, tensor: ziptensors.Tensor, block: tuple[int, int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Returns the byte shape and the logical shape, both in torch order, of a quantized row.

    `block` is `(elements, bytes)` of one block. The table states the logical ggml shape, fastest
    varying dimension first; blocks run along that dimension, so it must be a whole number of
    blocks. Raises TorchLoadError otherwise.
    """
    if not tensor.shape:
        raise TorchLoadError(f"{where}: a quantized tensor of no dimension holds no block")
    elements, block_bytes = block
    fastest = tensor.shape[0]
    if fastest % elements != 0:
        raise TorchLoadError(
            f"{where}: {tensor.dtype} rows of {fastest} elements are not whole blocks of {elements}"
        )
    logical = tuple(reversed(tensor.shape))
    byte_shape = (*logical[:-1], fastest // elements * block_bytes)
    return byte_shape, logical


def _plan_tensor(
    torch_module: ModuleType, member: ziptensors.Member, tensor: ziptensors.Tensor, quantized: str
) -> _Planned:
    """Checks one table row and returns its plan, or raises TorchLoadError naming the tensor."""
    where = f"tensor {tensor.name!r} of {member.name!r}"
    profile = member.profile
    block = _quantized_block(profile, tensor)
    ggml_type: str | None = None
    if block is not None and quantized != "refuse":
        ggml_type = tensor.dtype
        dtype = "uint8"
        shape, logical = _ggml_shapes(where, tensor, block)
        expected = _element_count(shape)
    else:
        table = _GGUF_DTYPES if profile == "gguf" else _SAFETENSORS_DTYPES
        entry = table.get(tensor.dtype)
        if entry is None:
            hint = (
                "; quantized='raw' or 'diffusers' returns its blocks undecoded"
                if block is not None
                else ""
            )
            raise TorchLoadError(
                f"{where}: {profile} dtype {tensor.dtype} has no torch dtype this loader accepts "
                "(quantized and packed data would need decoding, and reinterpreting its bytes "
                f"would give wrong weights){hint}"
            )
        dtype = entry.torch_name
        _torch_dtype(torch_module, dtype, where)
        shape = _torch_shape(profile, tensor.shape)
        logical = shape
        expected = _element_count(shape) * entry.item_bytes
    if expected != tensor.len:
        raise TorchLoadError(
            f"{where}: shape {tensor.shape} of {tensor.dtype} needs {expected} bytes and the table "
            f"says {tensor.len}"
        )
    if tensor.offset + tensor.len > member.len:
        raise TorchLoadError(
            f"{where}: bytes {tensor.offset}+{tensor.len} lie past the end of the member, "
            f"{member.len} bytes"
        )
    return _Planned(tensor.name, dtype, shape, tensor.offset, tensor.len, ggml_type, logical)


def _plan(
    torch_module: ModuleType,
    member: ziptensors.Member,
    keys: Iterable[str] | None,
    quantized: str,
) -> list[_Planned]:
    """Checks everything that can be checked before the first allocation and returns the tensors.

    Raises TypeError if `keys` is a string, NotFoundError for a key that is not in the table,
    ValueError for a `quantized` mode other than `refuse` on a member that is not GGUF, and
    TorchLoadError for a member or tensor that cannot be loaded.
    """
    if member.profile not in ("safetensors", "gguf"):
        raise TorchLoadError(
            f"{member.name!r} is a {member.profile} member; only safetensors and GGUF tensor tables "
            "state real dtypes and shapes. Use `torch.load(member.open(), weights_only=True)` for "
            "a .pt file."
        )
    if quantized != "refuse" and member.profile != "gguf":
        raise ValueError(
            f"quantized={quantized!r} applies to GGUF members, and {member.name!r} is "
            f"{member.profile}"
        )
    if sys.byteorder != "little":
        raise TorchLoadError(f"{member.name!r}: stored little-endian, and this host is not")
    if keys is None:
        if member.profile == "gguf":
            _check_gguf_table(member)
        selected = member.tensors
    else:
        if isinstance(keys, str):
            raise TypeError("`keys` is an iterable of tensor names, not one string")
        selected = [member.tensor_info(key) for key in dict.fromkeys(keys)]
    plans = [_plan_tensor(torch_module, member, tensor, quantized) for tensor in selected]
    # A table that overlaps could allocate far more than the member holds, so it stops here. Neither
    # format allows overlapping tensors in a file its own loader accepts.
    total = sum(plan.length for plan in plans)
    if total > member.len:
        raise TorchLoadError(
            f"{member.name!r}: its tensors add up to {total} bytes in a member of {member.len}, "
            "so the table overlaps"
        )
    return plans


def _fill_host(
    torch_module: ModuleType,
    member: ziptensors.Member,
    offset: int,
    tensor: torch.Tensor,
    where: str,
) -> None:
    """Fills `tensor` from byte `offset` of the member, decoding in place with the GIL released.

    `tensor` must be a fresh contiguous CPU tensor whose byte length the caller proved lies inside
    the member. A BOOL tensor is refused with TorchLoadError for a byte other than 0 or 1.
    """
    length = tensor.numel() * tensor.element_size()
    if length == 0:
        return
    # Invariant of the ctypes view: `torch.empty` gives one contiguous CPU block of `length` bytes and
    # `tensor` outlives the view. A view of the data pointer is the one route that needs no numpy and
    # takes every dtype: `Tensor.numpy()` refuses bfloat16 and float8.
    view = (ctypes.c_char * length).from_address(tensor.data_ptr())
    member.read_at(offset, view)
    if tensor.dtype == torch_module.bool:
        _check_bool(torch_module, view, where)


def _staging_bytes(staging_bytes: int | None) -> int:
    """Returns the staging buffer size to use, `STAGING_BYTES` for None.

    Raises TypeError for a non-integer and ValueError for a size below 1.
    """
    size = STAGING_BYTES if staging_bytes is None else staging_bytes
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError(f"`staging_bytes` is a positive integer, got {size!r}")
    if size < 1:
        raise ValueError(f"`staging_bytes` must be at least 1, got {size}")
    return size


class _DeviceStager:
    """Copies byte ranges of a member into tensors on one CUDA or ROCm device.

    Two pinned host buffers of `capacity` bytes take turns. The calling thread fills the current one
    with `read_at` (GIL released) and queues a `non_blocking` copy of that piece on a side stream, so
    decoding the next piece overlaps the transfer. An event per buffer, recorded when the buffer is
    left, guards its reuse: the host waits on it only before writing into that buffer again. Pinned
    host memory is at most 2 x `capacity`; the second buffer exists only once the first is full.
    Freed buffers go back to torch's caching host allocator, which keeps them pinned for reuse.

    Destinations are allocated by the caller on the current stream; `fill` marks them used on the
    side stream with `record_stream`, and orders the side stream after everything queued on the
    current stream so far. Without that order a destination carved from a block whose last reader is
    still queued on the current stream (a converted predecessor, another thread's kernel) could be
    overwritten before that reader ran. Use as a context manager: leaving it, also by an exception,
    makes the current stream wait for the side stream and the host wait for both, so nothing is in
    flight afterwards. One object belongs to one thread.
    """

    def __init__(
        self,
        torch_module: ModuleType,
        member: ziptensors.Member,
        target: torch.device,
        capacity: int,
    ) -> None:
        """Prepares a stager onto `target`, a device of type `cuda`, with buffers of `capacity` bytes.

        Allocates nothing on the host yet. The side stream starts behind the work already queued on
        the current stream, which may still use the memory the caller allocates destinations from.
        """
        self._torch = torch_module
        self._member = member
        self._capacity = capacity
        self._current_stream = torch_module.cuda.current_stream(target)
        self._stream = torch_module.cuda.Stream(device=target)
        self._stream.wait_stream(self._current_stream)
        self._buffers: list[torch.Tensor] = []
        self._events: list[torch.cuda.Event] = []
        self._pending: list[bool] = []
        self._index = 0
        self._used = 0

    def __enter__(self) -> _DeviceStager:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Orders the current stream after every queued copy and waits until they are done."""
        self._current_stream.wait_stream(self._stream)
        self._stream.synchronize()

    def _buffer(self) -> torch.Tensor:
        """Returns the current pinned buffer, allocating it on first use."""
        if self._index == len(self._buffers):
            self._buffers.append(
                self._torch.empty(self._capacity, dtype=self._torch.uint8, pin_memory=True)
            )
            self._events.append(self._torch.cuda.Event())
            self._pending.append(False)
        return self._buffers[self._index]

    def _rotate(self) -> None:
        """Leaves the full current buffer and waits until the other one is free to overwrite."""
        self._events[self._index].record(self._stream)
        self._pending[self._index] = True
        self._index ^= 1
        self._used = 0
        if self._index < len(self._pending) and self._pending[self._index]:
            self._events[self._index].synchronize()
            self._pending[self._index] = False

    def fill(self, tensor: torch.Tensor, offset: int, where: str) -> None:
        """Queues the copy of the member bytes from `offset` into all of `tensor`, a fresh contiguous
        tensor on the target device whose byte length the caller proved lies inside the member.

        The bytes are on the device once the context is left. A BOOL tensor is refused with
        TorchLoadError for a byte other than 0 or 1, checked in the staging buffer.
        """
        torch_module = self._torch
        length = tensor.numel() * tensor.element_size()
        if length == 0:
            return
        # A byte view of the destination: `view(-1)` also turns a 0-d tensor into one element.
        destination = tensor.view(-1).view(torch_module.uint8)
        # The caching allocator may hand out a block whose last use is still queued on the current
        # stream; record_stream protects blocks freed by side-stream use only. A device-side wait,
        # the host does not block.
        self._stream.wait_stream(self._current_stream)
        tensor.record_stream(self._stream)
        is_bool = tensor.dtype == torch_module.bool
        done = 0
        while done < length:
            if self._used == self._capacity:
                self._rotate()
            buffer = self._buffer()
            piece = min(length - done, self._capacity - self._used)
            # Invariant of the ctypes view: the pinned buffer is one contiguous block of `capacity`
            # bytes held by this object, `used + piece <= capacity`, and the event wait in `_rotate`
            # proved no queued copy still reads this region.
            view = (ctypes.c_char * piece).from_address(buffer.data_ptr() + self._used)
            self._member.read_at(offset + done, view)
            if is_bool:
                _check_bool(torch_module, view, where)
            with torch_module.cuda.stream(self._stream):
                destination[done : done + piece].copy_(
                    buffer[self._used : self._used + piece], non_blocking=True
                )
            self._used += piece
            done += piece

    def convert(self, tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Returns `tensor` converted to `dtype` on the device, once its queued copy is done.

        The conversion runs on the current stream, after it waited for the side stream, so the new
        tensor belongs to the stream the caller uses and the host does not block.
        """
        self._current_stream.wait_stream(self._stream)
        with self._torch.cuda.stream(self._current_stream):
            return tensor.to(dtype)


def _diffusers_wrapper(
    plans: list[_Planned],
) -> Callable[[torch.Tensor, str], torch.Tensor]:
    """Returns a function that wraps uint8 blocks as a diffusers `GGUFParameter` of a ggml type.

    Imports gguf and diffusers and resolves every quantized type of `plans`, before any decode.
    Raises TorchLoadError if either package is missing, or a type is unknown to gguf or not among
    the types diffusers dequantizes (`SUPPORTED_GGUF_QUANT_TYPES`), which `load_gguf_checkpoint`
    refuses too.
    """
    try:
        import gguf
        from diffusers.quantizers.gguf.utils import SUPPORTED_GGUF_QUANT_TYPES, GGUFParameter
    except ImportError as error:
        raise TorchLoadError(
            "quantized='diffusers' needs the packages diffusers and gguf: install both, or use "
            "quantized='raw'"
        ) from error
    kinds = {}
    for plan in plans:
        if plan.ggml_type is None:
            continue
        try:
            kinds[plan.ggml_type] = gguf.GGMLQuantizationType[plan.ggml_type]
        except KeyError as error:
            raise TorchLoadError(
                f"tensor {plan.name!r}: this gguf package has no quantization type {plan.ggml_type}"
            ) from error
        if kinds[plan.ggml_type] not in SUPPORTED_GGUF_QUANT_TYPES:
            raise TorchLoadError(
                f"tensor {plan.name!r}: diffusers does not dequantize {plan.ggml_type}; use "
                "quantized='raw'"
            )

    def wrap(data: torch.Tensor, ggml_type: str) -> torch.Tensor:
        """Wraps `data` as diffusers wraps it in `load_gguf_checkpoint`."""
        return GGUFParameter(data, quant_type=kinds[ggml_type])

    return wrap


def _decode_all(
    torch_module: ModuleType,
    member: ziptensors.Member,
    plans: list[_Planned],
    target: torch.device,
    dtype: torch.dtype | None,
    staging_bytes: int,
) -> list[torch.Tensor]:
    """Returns the tensors of `plans`, in order, on `target`, floats converted to `dtype`.

    `cuda` targets go through one `_DeviceStager`; anything else decodes each tensor on the host,
    converts it and moves it before the next one.
    """
    tensors: list[torch.Tensor] = []
    if target.type == "cuda":
        total = sum(plan.length for plan in plans)
        capacity = min(staging_bytes, max(total, 1))
        with _DeviceStager(torch_module, member, target, capacity) as stager:
            for plan in plans:
                tensor = torch_module.empty(
                    plan.shape, dtype=getattr(torch_module, plan.dtype), device=target
                )
                stager.fill(tensor, plan.offset, f"tensor {plan.name!r} of {member.name!r}")
                # Quantized blocks are uint8, so `is_floating_point` keeps `dtype` off them.
                if dtype is not None and tensor.is_floating_point():
                    tensor = stager.convert(tensor, dtype)
                tensors.append(tensor)
        return tensors
    for plan in plans:
        tensor = torch_module.empty(plan.shape, dtype=getattr(torch_module, plan.dtype))
        where = f"tensor {plan.name!r} of {member.name!r}"
        _fill_host(torch_module, member, plan.offset, tensor, where)
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype)
        if target.type != "cpu":
            tensor = tensor.to(target)
        tensors.append(tensor)
    return tensors


def load_state_dict(
    member: ziptensors.Member,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    keys: Iterable[str] | None = None,
    *,
    quantized: Quantized = "refuse",
    staging_bytes: int | None = None,
) -> dict[str, torch.Tensor | GgmlTensor]:
    """Decodes the tensors of `member` into new torch tensors, by name.

    Each tensor is allocated uninitialized and decoded in place with the GIL released, so there is no
    intermediate copy. Names and bytes are those of the file. GGUF shapes are reversed into torch
    order, which is what the writer's numpy array had; the bytes are not transposed, and the names
    stay GGUF names, so a llama.cpp or ComfyUI file still needs that project's name mapping.

    Accepted: safetensors F64, F32, F16, BF16, F8_E4M3 (float8_e4m3fn), F8_E5M2 (float8_e5m2), I64,
    I32, I16, I8, U64, U32, U16, U8, BOOL, and GGUF F64, F32, F16, BF16. All the tensors asked for are
    checked before the first allocation.

    `quantized` decides block-quantized GGUF tensors (`Q4_0`, `Q8_0`, `Q4_K`, `IQ4_XS`, `MXFP4` and
    every other type whose core block geometry, `Tensor.block`, holds more than one element).
    `refuse` raises TorchLoadError.
    `raw` returns each as a `GgmlTensor` of its undecoded blocks. `diffusers` returns each as the
    `GGUFParameter` that diffusers' `load_gguf_checkpoint` makes of the same bytes, for
    `from_single_file(..., quantization_config=GGUFQuantizationConfig(...))`; float tensors stay
    plain tensors, BF16 included, where diffusers would wrap BF16. Any mode but `refuse` on a member
    that is not GGUF is a ValueError. `diffusers` refuses a type diffusers cannot dequantize, as its
    own loader does. Tensors of a ggml type the table has no row for (retired or newer ggml ids)
    still refuse a whole-member load in every mode.

    `device` is where the tensors end up; `meta` is refused, since it would discard the data. On a
    `cuda` device (CUDA or ROCm), the member's bytes are decoded into two pinned host buffers of
    `staging_bytes` each (default `STAGING_BYTES`, capped at the total size asked for) and copied
    asynchronously on a side stream while the next piece decodes; the call returns after every copy
    is done. Host memory then peaks at the two buffers, which torch's caching host allocator keeps
    pinned for reuse after the call, and the device holds the state dict. On `cpu` the decoded
    tensors are the result, so host memory holds the whole state dict. On any other device each
    tensor is decoded on the host, moved, and its host copy freed before the next one, so host
    memory peaks at the largest tensor. All add the decode scratch of the memory contract in
    `ARCHITECTURE.md`.

    `dtype`, a floating point torch dtype, converts floating point tensors after decoding: on the
    device for `cuda`, otherwise on the host. Integer, bool and quantized tensors keep their dtype,
    because a converted index or block would be wrong. Each conversion briefly holds both versions of
    one tensor.

    `keys` selects tensors, in that order; None means all of them, in table order.

    Raises ImportError without torch, TorchLoadError (a ValueError) for a member or tensor that cannot
    be loaded, naming it and why, or for `quantized="diffusers"` without diffusers or gguf, ValueError
    for a `device`, `dtype`, `quantized` or `staging_bytes` refused above, TypeError if `keys` is a
    string or `staging_bytes` no integer, NotFoundError for an unknown key, and any error of
    `Member.read_at`. Nothing is returned after an error, so no partly decoded state is visible.
    """
    torch_module = _import_torch()
    if quantized not in _QUANTIZED_MODES:
        raise ValueError(f"`quantized` is one of {_QUANTIZED_MODES}, got {quantized!r}")
    target = torch_module.device(device)
    if target.type == "meta":
        raise ValueError("device `meta` would discard the decoded data")
    if dtype is not None and not dtype.is_floating_point:
        raise ValueError(f"`dtype` must be a floating point dtype, got {dtype}")
    staging = _staging_bytes(staging_bytes)
    plans = _plan(torch_module, member, keys, quantized)
    wrap = _diffusers_wrapper(plans) if quantized == "diffusers" else None
    log.info("loading %d tensors from %s to %s", len(plans), member.name, target)
    tensors = _decode_all(torch_module, member, plans, target, dtype, staging)
    state: dict[str, torch.Tensor | GgmlTensor] = {}
    for plan, tensor in zip(plans, tensors, strict=True):
        if plan.ggml_type is None:
            state[plan.name] = tensor
        elif wrap is not None:
            state[plan.name] = wrap(tensor, plan.ggml_type)
        else:
            state[plan.name] = GgmlTensor(tensor, plan.ggml_type, plan.logical_shape)
    log.info("loaded %d tensors from %s", len(state), member.name)
    return state


def load_into(
    module: torch.nn.Module,
    member: ziptensors.Member,
    strict: bool = True,
    assign: bool = True,
) -> torch.nn.modules.module._IncompatibleKeys:
    """Fills `module`, normally built on the meta device, from `member`, and returns torch's report.

    With `assign=True` the module takes the decoded tensors themselves, on the host, with their
    stored dtypes: there is no second copy. `assign=False` copies into existing storage, which a meta
    module does not have, and then peak memory is twice the model. Quantized GGUF tensors are
    refused, as by `load_state_dict` under `quantized="refuse"`.

    A file holds neither non-persistent buffers nor the second name of a tied parameter, so a module
    can stay partly on the meta device, and a forward pass through one then returns meta tensors
    instead of failing. Under `strict=True` that raises TorchLoadError naming them. Under
    `strict=False` it logs a warning, and the caller finishes the module: `tie_weights()`, or
    registering the buffers again.

    Raises what `load_state_dict` raises, RuntimeError from `Module.load_state_dict` for a key or
    shape mismatch under `strict=True`, and TorchLoadError as above. After any of the last two the
    module may be partly filled.
    """
    report = module.load_state_dict(load_state_dict(member), strict=strict, assign=assign)
    named = (*module.named_parameters(), *module.named_buffers())
    left = [name for name, tensor in named if tensor.is_meta]
    if left:
        message = f"{len(left)} tensors of the module are still on the meta device: {left[:8]}"
        if strict:
            raise TorchLoadError(f"{message}; pass strict=False and materialize them afterwards")
        log.warning("%s", message)
    return report
