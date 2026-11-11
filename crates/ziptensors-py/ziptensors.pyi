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
from pathlib import Path
from types import TracebackType
from typing import Literal, Self, TypeAlias, final

StrPath: TypeAlias = str | PathLike[str]
Mode: TypeAlias = Literal["fast", "balanced", "max"]
Strategy: TypeAlias = Literal["rewrite", "append", "in_place"]
LogLevel: TypeAlias = Literal["off", "error", "warn", "warning", "info", "debug", "trace"]

__version__: str
"""Version of the package this module was built from."""

class ZiptensorsError(Exception):
    """Base class of every error of this module."""

    code: int
    """Stable numeric `ErrorCode` of the failure, set on every exception this module raises."""

class ZiptensorsIoError(ZiptensorsError):
    """An operating system file operation failed, or a file holds a valid container followed by
    bytes that could not be removed; recovery truncates such a file."""

class NotDurableError(ZiptensorsIoError):
    """The output WAS written and is complete, but its directory could not be synced, so a power
    loss may bring back the previous file. Every other error of a write leaves the output untouched."""

class FormatError(ZiptensorsError):
    """The file is not a container, its header, footer or index violates the format, or recovery
    found no valid container in it."""

class UnsupportedError(ZiptensorsError):
    """The file needs a format version, feature or codec this library does not implement."""

class BudgetError(ZiptensorsError):
    """A size declared by the file exceeds a limit. Raised before the data is read."""

class CorruptionError(ZiptensorsError):
    """Stored data does not decode, or does not match its checksum or hash, or an in-place edit
    failed after it began moving data and damaged the file."""

class RangeError(ZiptensorsError):
    """A read range lies outside the member, or a buffer has the wrong size."""

class NotFoundError(ZiptensorsError):
    """The container has no such member, or the member no such tensor."""

class InputError(ZiptensorsError):
    """An input of an encode was refused: an unsafe name, a symbolic link, or a file that changed;
    or an edit request cannot be applied as asked, or a tree operation found names that collide,
    and nothing was written; or two inputs of a write would get one member name (code 26)."""

class MountError(ZiptensorsError):
    """A mount cannot be made or ended: the platform has no mount backend (macOS, code
    `UnsupportedFeature`), a name the OS cannot show (`UnsafeMemberName`), an overlay entry that is
    damaged (its error's code) or claimed twice (`TreeConflict`), a member the core refuses (its
    code, with the core's exception as `__cause__`), or the backend refused or an unmount left the
    mount in place (`Io`)."""

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
class RestoreAllReport:
    """What `Container.restore_all` produced."""

    @property
    def members(self) -> int:
        """Members restored: all members of the container."""
    @property
    def bytes(self) -> int: ...

@final
class EditReport:
    """What an edit or a compaction did."""

    @property
    def kept(self) -> int: ...
    @property
    def added(self) -> int: ...
    @property
    def removed(self) -> int: ...
    @property
    def replaced(self) -> int: ...
    @property
    def generation(self) -> int:
        """Footer generation of the result: the source's plus one."""
    @property
    def encode(self) -> EncodeReport:
        """The writer's report. Under `append` its `output_len` is the whole file."""
    @property
    def dead_bytes_before(self) -> int:
        """Bytes of the source no member needed: what a rewrite with no operation drops."""

@final
class RecoverReport:
    """What `recover` found and did."""

    @property
    def file_len(self) -> int: ...
    @property
    def valid_len(self) -> int:
        """Length of the valid container: `file_len` when the file opened as it was."""
    @property
    def generation(self) -> int: ...
    @property
    def truncated(self) -> bool:
        """Whether the file was cut; False when nothing needed cutting, and on a dry run."""
    @property
    def verified_bytes(self) -> int | None:
        """Sum of member lengths decoded and checked, or None when nothing was verified."""
    @property
    def scanned_bytes(self) -> int:
        """Distinct bytes the backward scan read, at most `file_len - 64`; 0 when the file opened."""
    @property
    def passed_over(self) -> int:
        """Newer candidate generations that failed the checks and were passed over; 0 when the
        file opened as it was or only a torn tail was cut."""

@final
class Inputs:
    """What the walk of `collect_inputs`, the one `pack` does over directories, found."""

    @property
    def files(self) -> list[tuple[Path, str]]:
        """`(path, member name)` per file, per root in the order given, then by member name."""
    @property
    def skipped_symlinks(self) -> int: ...
    @property
    def skipped_incomplete(self) -> int:
        """`*.incomplete` files left out: partial downloads."""

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
    def read_many(self, requests: Sequence[tuple[int, Buffer]]) -> None:
        """Fills every `(offset, buffer)` request, visiting the file in pack order. Every range is
        checked before any read; buffers whose memory overlaps raise `BufferError` before anything
        is written."""
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
    """An open container file. Safe from any number of threads. A context manager: leaving the
    `with` block closes it. After `close()` every call raises `ValueError`; members taken before
    keep the file open until they are dropped."""

    def __new__(
        cls,
        path: StrPath,
        *,
        threads: int | None = None,
        max_total_decoded: int | None = None,
        verify: Literal["chunk", "off"] = "chunk",
        max_index_bytes: int | None = None,
        max_decode_amplification: int | None = None,
    ) -> Container:
        """Opens a container. `max_total_decoded` caps the sum of member lengths, before any data is
        read, which is what bounds a restore. The default is no budget: untrusted input needs one,
        and 0 is a budget of zero bytes. `verify="off"` skips chunk hashes. None for the two limits
        takes the library's defaults."""
    @property
    def members(self) -> list[MemberInfo]: ...
    def member(self, name: str | None = None) -> Member:
        """The member called `name`, or with None the only member of a single-file container."""
    def layer_report(self) -> tuple[LayerBytes, list[ClassBytes]]:
        """Where the bytes went, from the index alone: the totals and the classes by dtype."""
    def restore_all(self, dir: StrPath) -> RestoreAllReport:
        """Restores every member under `dir`, creating directories. One unsafe name raises
        `InputError` and creates nothing; a later failure keeps the members restored before it."""
    def close(self) -> None:
        """Drops this object's handle on the file. Idempotent."""
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool: ...

@final
class PackBuilder:
    """Builds a container member by member. The output appears only at `finish`, proven against the
    sources. `close()`, leaving a `with` block without `finish`, or dropping the object removes the
    temporary file. Any call after `finish` or `close` raises `ValueError`; after a failed add every
    later call fails."""

    def __new__(
        cls, output: StrPath, mode: Mode = "max", threads: int | None = None
    ) -> PackBuilder: ...
    def add_file(self, path: StrPath, name: str | None = None) -> MemberReport:
        """Adds a file as a member, by default under the file's name. A name already in the pack
        raises `InputError` (code 26)."""
    def add_file_delta(self, path: StrPath, name: str, base_member: str) -> MemberReport:
        """Adds a file stored against `base_member`, a member this builder added from a file, as
        delta blobs where smaller (mode "max" only; otherwise as `add_file`)."""
    def add_member_from(self, container: Container, name: str) -> MemberReport:
        """Copies a member of another container with no recompression."""
    def finish(self) -> EncodeReport:
        """Proves and commits the container. The builder is spent either way."""
    def close(self) -> None:
        """Drops an unfinished pack and its temporary file. Idempotent."""
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool: ...

@final
class AddOp:
    """Adds the file at `path` as a new member at the end, named `name` (default: the file's name),
    optionally stored against `delta_base`, a kept member or an earlier file of the same edit."""

    def __new__(
        cls, path: StrPath, name: str | None = None, *, delta_base: str | None = None
    ) -> AddOp: ...
    @property
    def path(self) -> Path: ...
    @property
    def name(self) -> str | None: ...
    @property
    def delta_base(self) -> str | None: ...

@final
class RemoveOp:
    """Removes the member `name`."""

    def __new__(cls, name: str) -> RemoveOp: ...
    @property
    def name(self) -> str: ...

@final
class ReplaceOp:
    """Replaces the content of member `name` with the file at `path`, keeping its slot."""

    def __new__(
        cls, name: str, path: StrPath, *, delta_base: str | None = None
    ) -> ReplaceOp: ...
    @property
    def name(self) -> str: ...
    @property
    def path(self) -> Path: ...
    @property
    def delta_base(self) -> str | None: ...

@final
class FileOutcome:
    """What happened to one path of a tree run."""

    @property
    def path(self) -> Path: ...
    @property
    def counterpart(self) -> Path | None:
        """In compress the container of `path`, in decompress the restored file."""
    @property
    def status(self) -> str:
        """`compressed`, `recompressed`, `pair_resolved`, `pair_verified`, `already_compressed`,
        `restored`, `not_weight`, `foreign`, `skipped`, `planned` or `failed`."""
    @property
    def reason(self) -> str | None:
        """Why, for `foreign`, `skipped` and `planned`; the message for `failed`."""
    @property
    def error(self) -> ZiptensorsError | None:
        """For `failed`, the exception the module raises for that error, not raised."""
    @property
    def original_len(self) -> int | None: ...
    @property
    def container_len(self) -> int | None: ...
    @property
    def removed(self) -> Path | None:
        """The copy removed because its twin was proven, if any."""

@final
class TreeTotals:
    """Counts of a tree run by status, and byte sums of the files that ended with a container."""

    @property
    def compressed(self) -> int: ...
    @property
    def recompressed(self) -> int: ...
    @property
    def pair_resolved(self) -> int: ...
    @property
    def pair_verified(self) -> int: ...
    @property
    def already_compressed(self) -> int: ...
    @property
    def restored(self) -> int: ...
    @property
    def not_weight(self) -> int: ...
    @property
    def foreign(self) -> int: ...
    @property
    def skipped(self) -> int: ...
    @property
    def planned(self) -> int: ...
    @property
    def failed(self) -> int: ...
    @property
    def original_bytes(self) -> int: ...
    @property
    def container_bytes(self) -> int: ...

@final
class TreeReport:
    """What a tree run did, one `FileOutcome` per path it looked at, sorted by path."""

    @property
    def root(self) -> Path: ...
    @property
    def dry_run(self) -> bool: ...
    @property
    def files(self) -> list[FileOutcome]: ...
    @property
    def warnings(self) -> list[str]: ...
    @property
    def totals(self) -> TreeTotals: ...
    @property
    def has_failures(self) -> bool:
        """Whether any file failed; the run went on past each failure."""

@final
class Mount:
    """A live read-only mount made by `mount` or `mount_overlay`. Leaving its `with` block unmounts,
    whether or not the block raised (KeyboardInterrupt included). An unmount detaches the
    mountpoint at once, then waits until every file open under it is closed (memory maps
    included): close them first, since a file the unmounting thread holds makes it wait forever.
    Concurrent `unmount()` calls all return when that wait ends, with the same outcome. A `Mount`
    dropped while mounted is unmounted in the background with a warning in the log."""

    @property
    def mountpoint(self) -> Path:
        """Where the files are, as the backend resolved it (`X:\\` for a Windows drive)."""
    @property
    def is_mounted(self) -> bool:
        """True until an unmount by this object has ended, also while it still waits for open
        files; False after it ended, even if it failed. An unmount from outside
        (`fusermount3 -u`) is not seen here."""
    def unmount(self) -> None:
        """Unmounts and waits, with the GIL released, for open files to close. A call while
        another thread unmounts waits for that unmount and gets its outcome; a call after it ended
        returns that outcome again. On failure raises `MountError` (code `Io`) and gives the mount
        up; on Linux it then stays until `fusermount3 -u -z` removes it."""
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool: ...

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
    """Packs files into one container. A bare file path's member takes the file's name; a bare
    directory is walked as `collect_inputs` walks it (links and `*.incomplete` skipped, an output
    under it refused with `InputError`). `delta_base` (mode "max" only, else ValueError) is the
    member name of one input: it is added first, and tensors of every other input that pair with
    its tensors are stored as delta blobs against it where that is smaller. A name no input has is
    ValueError. Two inputs with one member name raise `InputError` (code 26) before the output is
    started."""

def collect_inputs(roots: Sequence[StrPath]) -> Inputs:
    """Walks `roots` as `pack` walks a directory: the files it would add, and what it skips."""

def edit(
    container: StrPath,
    ops: Sequence[AddOp | RemoveOp | ReplaceOp],
    *,
    strategy: Strategy = "rewrite",
    repack: bool = False,
    mode: Mode = "max",
    threads: int | None = None,
    max_total_decoded: int | None = None,
) -> EditReport:
    """Edits a container; each member name appears in one operation. Every check runs before a byte
    is written. `"rewrite"` writes a proven copy and renames it over; `"append"` writes only new
    data after the end; `"in_place"` needs no extra space but is RISKY: an interruption after its
    first write damages the file (`CorruptionError`). `repack` (rewrite only) re-encodes kept
    members. Close any `Container` on the file first, except under `"append"`."""

def compact(
    container: StrPath,
    *,
    strategy: Literal["rewrite", "in_place"] = "rewrite",
    repack: bool = False,
    mode: Mode = "max",
    threads: int | None = None,
    max_total_decoded: int | None = None,
) -> EditReport:
    """An edit with no operation: drops dead bytes of earlier appends. `"append"` raises
    `InputError`."""

def recover(
    path: StrPath,
    *,
    dry_run: bool = False,
    verify: bool = True,
    max_candidates: int = 64,
    max_scan_bytes: int | None = None,
    threads: int | None = None,
) -> RecoverReport:
    """Shortens a container to its latest valid generation after an interrupted append or in-place
    edit. A file that opens as it is stays untouched. `FormatError` if no generation is valid."""

def compress_tree(
    path: StrPath,
    *,
    keep: bool = False,
    follow_symlinks: bool = False,
    dry_run: bool = False,
    mode: Mode = "max",
    threads: int | None = None,
    max_total_decoded: int | None = None,
) -> TreeReport:
    """Compresses every weight file under `path` into a one-member container beside it, removing
    the original once proven unless `keep`. Collisions raise `InputError` before anything is
    written; a file's failure is in its outcome and the run goes on."""

def decompress_tree(
    path: StrPath,
    *,
    keep: bool = False,
    force: bool = False,
    dry_run: bool = False,
    threads: int | None = None,
    max_total_decoded: int | None = None,
) -> TreeReport:
    """Restores every per-file container under `path` beside it, removing the container once
    proven unless `keep`. A different file at the restore path fails that container unless
    `force`. Conflicts raise `InputError` before anything is written."""

def enable_stderr_logging(level: LogLevel = "info") -> None:
    """Writes the library's log events at `level` and above to stderr for the rest of the process.
    Only the first call in a process takes effect; later calls succeed and change nothing. Python's
    `logging` does not see these events."""

def mount(
    container: StrPath,
    mountpoint: StrPath,
    *,
    threads: int | None = None,
    cache_size: int | None = None,
    readahead: int = 0,
    max_total_decoded: int | None = None,
) -> Mount:
    """Mounts the members of `container` read-only as files under `mountpoint`; use the result as a
    context manager. Setup runs with the GIL released.

    Linux: FUSE through `/dev/fuse` and `fusermount3`, no root; `mountpoint` is an existing
    directory. Windows: `mountpoint` is a free drive letter (`X:`) or a directory that does not
    exist yet, created and removed by the mount. Windows mounts use WinFsp - Windows File System
    Proxy, Copyright (C) Bill Zissimopoulos, https://github.com/winfsp/winfsp (GPL-3.0 with a FLOSS
    exception); install it from https://winfsp.dev/rel/. macOS has no backend: `MountError`.

    `threads`: decode threads, None or 0 one per logical CPU. `cache_size`: decoded bytes kept for
    reads smaller than a chunk, None for 256 MiB, at least twice the largest chunk (`BudgetError`).
    `readahead`: windows decoded per cache miss, 0 the thread count. `max_total_decoded`: the
    budget of `Container`; set it for a file you do not trust. Names the OS cannot show raise
    `MountError` naming every one."""

def mount_overlay(
    directory: StrPath,
    mountpoint: StrPath,
    *,
    threads: int | None = None,
    cache_size: int | None = None,
    readahead: int = 0,
    max_total_decoded: int | None = None,
    max_open_containers: int = 4,
    max_open_files: int = 64,
) -> Mount:
    """Mounts `directory` read-only under `mountpoint` as an overlay: plain files and packs as they
    are, each per-file container (`compress_tree`) as its original file under its original name.
    The tree is a snapshot taken now; a file changed later fails its reads. A damaged entry, a name
    claimed twice or a name the OS cannot show refuses the whole mount with `MountError`.
    Platforms, `mountpoint`, the WinFsp notice (WinFsp - Windows File System Proxy, Copyright (C)
    Bill Zissimopoulos, https://github.com/winfsp/winfsp; install from https://winfsp.dev/rel/),
    `threads`, `readahead` and `max_total_decoded` (per container) are as for `mount`.

    `cache_size` (None for 256 MiB) is shared: each open container gets `cache_size /
    max_open_containers`, which must hold 16 MiB, or `ValueError`. At most `max_open_containers`
    containers and `max_open_files` plain files stay open, the least recently used closed first."""
