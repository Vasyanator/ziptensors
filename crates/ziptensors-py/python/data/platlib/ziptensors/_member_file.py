"""File: python/data/platlib/ziptensors/_member_file.py
Purpose: the file object behind `Member.open()`, a read-only seekable `io.RawIOBase` over `read_at`.

Main responsibilities: keep a position, clamp reads to the end of the member, and pass the caller's
buffer straight to `Member.read_at`, so that `readinto` decodes with the GIL released and no copy.

Key structures: `MemberFile`.

Implementation notes: this class is pure Python because subclassing `io.RawIOBase` from an abi3
`#[pyclass]` is not possible. `Member.open` in `src/reader.rs` imports it on first use.
"""

from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # `Buffer` exists from Python 3.12, and the module supports 3.11.
    from collections.abc import Buffer

    from ziptensors import Member


class MemberFile(io.RawIOBase):
    """A read-only binary file over one member.

    Seeking past the end is allowed and reads nothing there, as on an ordinary file. Each object has
    its own position, and like any file object it belongs to one thread at a time. Corrupt data
    raises the exceptions of `Member.read_at`, never a short read.
    """

    def __init__(self, member: Member) -> None:
        """Opens `member` at position 0. The object keeps the member, and so the container, alive."""
        super().__init__()
        self._member = member
        self._length = member.len
        self._position = 0

    def readable(self) -> bool:
        """Always true."""
        return True

    def seekable(self) -> bool:
        """Always true."""
        return True

    def write(self, data: Buffer) -> int:
        """Raises io.UnsupportedOperation: the inherited method raises NotImplementedError."""
        raise io.UnsupportedOperation("write")

    def _remaining(self, wanted: int) -> int:
        """Returns how many of `wanted` bytes exist at the position. Raises ValueError once closed."""
        if self.closed:
            raise ValueError("I/O operation on closed file")
        return max(0, min(wanted, self._length - self._position))

    def readinto(self, buffer: Buffer) -> int:
        """Fills `buffer` from the position and returns the bytes read, 0 at the end of the member.

        `buffer` is any writable, C-contiguous object of the buffer protocol.
        """
        with memoryview(buffer) as view, view.cast("B") as whole:
            count = self._remaining(len(whole))
            if count == 0:
                return 0
            with whole[:count] as part:
                self._member.read_at(self._position, part)
        self._position += count
        return count

    def read(self, size: int | None = -1) -> bytes:
        """Returns up to `size` bytes from the position, or all that remain if `size` is negative.

        Faster than the inherited `read`, which would allocate twice.
        """
        if size is None or size < 0:
            return self.readall()
        count = self._remaining(size)
        if count == 0:
            # The position may lie past the end, where even an empty read is out of range.
            return b""
        data = self._member.read(self._position, count)
        self._position += count
        return data

    def readall(self) -> bytes:
        """Returns all the bytes from the position to the end of the member."""
        return self.read(self._remaining(self._length))

    def seek(self, offset: int, whence: int = os.SEEK_SET, /) -> int:
        """Moves the position and returns it. Raises ValueError for a position before the start."""
        if self.closed:
            raise ValueError("I/O operation on closed file")
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._length + offset
        else:
            raise ValueError(f"unknown whence {whence}")
        if position < 0:
            raise ValueError(f"position {position} is before the start of the member")
        self._position = position
        return position
