"""File: ziptensors.pyi
Purpose: type stub of the extension module `ziptensors`, packaged by maturin with `py.typed`.

Main responsibilities: state the public surface for type checkers. `tests/test_ziptensors.py` checks
that every name here exists at runtime, so the two cannot drift apart silently.

Implementation notes: sizes and offsets are in bytes. Dtype names are those of section 5.3 of the
format spec, profile names those of section 5.4. The pure-Python submodules `ziptensors.torch` and
`ziptensors.safetensors` carry their own annotations, so they have no stub here.
"""

from collections.abc import Buffer, Sequence
from io import RawIOBase
from os import PathLike
from typing import Literal, TypeAlias, final

StrPath: TypeAlias = str | PathLike[str]
Mode: TypeAlias = Literal["fast", "balanced", "max"]

class ZiptensorsError(Exception):
    """Base class of every error of this module."""

    code: int
    """Stable numeric `ErrorCode` of the failure, set on every exception this module raises."""

class ZiptensorsIoError(ZiptensorsError):
    """An operating system file operation failed."""

class NotDurableError(ZiptensorsIoError):
    """The output WAS written and is complete, but its directory could not be synced, so a power
    loss may bring back the previous file. Every other error of a write leaves the output untouched."""

class FormatError(ZiptensorsError):
    """The file is not a container, or its header, footer or index violates the format."""

class UnsupportedError(ZiptensorsError):
    """The file needs a format version, feature or codec this library does not implement."""

class BudgetError(ZiptensorsError):
    """A size declared by the file exceeds a limit. Raised before the data is read."""

class CorruptionError(ZiptensorsError):
    """Stored data does not decode, or does not match its checksum or hash."""

class RangeError(ZiptensorsError):
    """A read range lies outside the member, or a buffer has the wrong size."""

class NotFoundError(ZiptensorsError):
    """The container has no such member, or the member no such tensor."""

class InputError(ZiptensorsError):
    """An input of an encode was refused: an unsafe name, a symbolic link, or a file that changed."""

@final
class MemberInfo:
    """What a container says about one member."""

    @property
    def name(self) -> str:
        """Path string as stored in the file. Untrusted: check it before joining it to a directory."""
    @property
    def len(self) -> int: ...
    @property
    def blake3(self) -> bytes:
        """BLAKE3 of the whole content, 32 bytes."""
    @property
    def profile(self) -> str:
        """`opaque`, `safetensors`, `gguf`, `onnx`, `pt_zip`, or `unknown(n)`. A hint."""
    @property
    def tensor_count(self) -> int: ...

@final
class Tensor:
    """One named tensor of a member. Metadata: restored bytes never depend on it."""

    @property
    def name(self) -> str: ...
    @property
    def offset(self) -> int: ...
    @property
    def len(self) -> int:
        """Length in bytes: the size of the buffer `decode_tensor_into` needs."""
    @property
    def shape(self) -> list[int]:
        """Dimensions in the order of the source file. GGUF keeps its own order, fastest varying
        first, which is the reverse of numpy's."""
    @property
    def dtype(self) -> str:
        """Registry name such as `F32`, `BF16` or `Q8_0`, or `UNKNOWN(0x....)`."""
    @property
    def block(self) -> tuple[int, int] | None:
        """`(elements, bytes)` of one block of the dtype: `(1, width)` for a plain type, one ggml
        block for a block-quantized type (`(32, 34)` for `Q8_0`), None if the geometry is unknown."""

@final
class LayerBytes:
    """Bytes of one member or class, by what became of them."""

    @property
    def raw(self) -> int: ...
    @property
    def widen_saved(self) -> int: ...
    @property
    def dedup_saved(self) -> int: ...
    @property
    def entropy_saved(self) -> int: ...
    @property
    def overhead(self) -> int: ...
    @property
    def stored(self) -> int: ...

@final
class ClassBytes:
    """The bytes of one class of source data."""

    @property
    def dtype(self) -> str | None:
        """Registry name of the dtype; None for skeleton fragments and opaque files."""
    @property
    def layers(self) -> LayerBytes: ...

@final
class MemberReport:
    """What adding one member did."""

    @property
    def name(self) -> str: ...
    @property
    def len(self) -> int: ...
    @property
    def blake3(self) -> bytes: ...
    @property
    def profile(self) -> str: ...
    @property
    def tensor_count(self) -> int: ...
    @property
    def extent_count(self) -> int: ...
    @property
    def blobs_added(self) -> int: ...
    @property
    def blobs_reused(self) -> int: ...
    @property
    def widen_restreams(self) -> int: ...
    @property
    def delta_blobs(self) -> int: ...
    @property
    def compressed_bytes(self) -> int: ...
    @property
    def copied_bytes(self) -> int: ...
    @property
    def layers(self) -> LayerBytes: ...

@final
class VerifyReport:
    """What the full decode before commit did."""

    @property
    def bytes(self) -> int: ...
    @property
    def seconds(self) -> float: ...

@final
class EncodeReport:
    """What a finished container holds and what it cost."""

    @property
    def output_len(self) -> int: ...
    @property
    def index_stored_len(self) -> int: ...
    @property
    def members(self) -> list[MemberReport]: ...
    @property
    def total(self) -> LayerBytes: ...
    @property
    def classes(self) -> list[ClassBytes]: ...
    @property
    def verify(self) -> VerifyReport: ...

@final
class RestoreReport:
    """What a restore produced."""

    @property
    def bytes(self) -> int: ...
    @property
    def blake3(self) -> bytes: ...

@final
class Member:
    """One member of a container. Safe from any number of threads; reads release the GIL.

    A buffer passed to a method is any writable, C-contiguous object of the buffer protocol, of any
    item format except Python objects, sized in bytes. Anything else raises `BufferError`, or
    `TypeError` if it is no buffer at all. No other thread may touch it during the call. After an
    error its contents are unspecified.
    """

    @property
    def name(self) -> str: ...
    @property
    def len(self) -> int: ...
    @property
    def blake3(self) -> bytes: ...
    @property
    def profile(self) -> str:
        """`opaque`, `safetensors`, `gguf`, `onnx`, `pt_zip`, or `unknown(n)`. A hint."""
    @property
    def tensors(self) -> list[Tensor]: ...
    def tensor_info(self, name: str) -> Tensor:
        """The tensor called `name`, for sizing a buffer."""
    def read_at(self, offset: int, buffer: Buffer) -> None:
        """Fills all of `buffer` from byte `offset` of the member."""
    def decode_tensor_into(self, name: str, buffer: Buffer) -> None:
        """Fills `buffer`, of exactly the tensor's length, with the bytes of the tensor."""
    def read(self, offset: int, length: int) -> bytes:
        """Returns `length` bytes from byte `offset` of the member."""
    def open(self) -> RawIOBase:
        """Opens this member as a read-only, seekable binary file, for `torch.load`. Each file has
        its own position, and like any file object it belongs to one thread at a time."""
    def restore_file(self, path: StrPath) -> RestoreReport:
        """Writes this member to `path`, replacing a file there, once it hashed correctly. On an
        error `path` is untouched, except after `NotDurableError`."""
    def verify(self) -> None:
        """Decodes the whole member and compares its BLAKE3 with the stated hash."""

@final
class Container:
    """An open container file. Safe from any number of threads."""

    def __new__(
        cls,
        path: StrPath,
        threads: int | None = None,
        max_total_decoded: int | None = None,
    ) -> Container:
        """Opens a container. `max_total_decoded` caps the sum of member lengths, before any data is
        read, which is what bounds a restore. The default is no budget: untrusted input needs one."""
    @property
    def members(self) -> list[MemberInfo]: ...
    def member(self, name: str | None = None) -> Member:
        """The member called `name`, or with None the only member of a single-file container."""

def encode_file(
    input: StrPath,
    output: StrPath,
    mode: Mode = "balanced",
    threads: int | None = None,
) -> EncodeReport:
    """Encodes one file as a single-member container, proven against the source before commit. On
    an error `output` is untouched, except after `NotDurableError`."""

def pack(
    inputs: Sequence[StrPath | tuple[StrPath, str]],
    output: StrPath,
    mode: Mode = "max",
    threads: int | None = None,
    *,
    delta_base: str | None = None,
) -> EncodeReport:
    """Packs files into one container. A bare path's member takes the file's name. `delta_base`
    (mode "max" only, else ValueError) is the member name of one input: it is added first, and
    tensors of every other input that pair with its tensors are stored as delta blobs against it
    where that is smaller. A name no input has is ValueError."""
