"""File: tests/test_mount.py
Purpose: contract tests of `ziptensors.mount`, `ziptensors.mount_overlay` and `Mount`.

Main responsibilities: a generated pack mounted and read byte for byte through the kernel; the
mountpoint empty and unlisted after the `with` block, also when the block raised (KeyboardInterrupt
included); `unmount()` idempotent; concurrent `unmount()` calls all waiting for a file still open;
a dropped `Mount` unmounted in the background; the error classes
of a bad container, mountpoint, cache size and overlay option; an overlay of a `compress_tree` tree
showing each container as its original file beside passthrough files; macOS refusing with
`MountError`.

Key functions: `fuse_missing`, `listed`, `make_pack`.

Implementation notes: FUSE tests skip with one reason unless this is Linux, `/dev/fuse` opens for
reading and writing and `fusermount3` is on PATH, as the mount crate's own test does. Every mount is
made under pytest's `tmp_path` and ended by `with` before the test returns, also on a failed
assertion; files read through a mount are closed before it ends, since an unmount waits for them.
"""

from __future__ import annotations

import gc
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

from test_tree import make_tree
from test_ziptensors import CODE_IO, compressible_bytes, make_safetensors, seeded_bytes
import ziptensors
from ziptensors import BudgetError, Mount, MountError, ZiptensorsError, ZiptensorsIoError

CODE_UNSUPPORTED_FEATURE = 4
# Seconds a dropped `Mount` may take to unmount on its background thread.
DROP_DEADLINE = 10.0
# Seconds a thread may take to start or end an unmount once nothing holds it back.
THREAD_DEADLINE = 10.0
# Seconds both concurrent unmounts must stay blocked while a file is open: long enough that a call
# returning early would be seen, short enough to keep the test fast.
WAIT_SHOWS = 0.5


def fuse_missing() -> str | None:
    """Why a FUSE mount cannot be made here, or None if it can."""
    if not sys.platform.startswith("linux"):
        return "FUSE mounts are tested on Linux only"
    try:
        os.close(os.open("/dev/fuse", os.O_RDWR))
    except OSError as error:
        return f"/dev/fuse cannot be opened for reading and writing: {error}"
    if shutil.which("fusermount3") is None:
        return "fusermount3 is not on PATH"
    return None


needs_fuse = pytest.mark.skipif(fuse_missing() is not None, reason=fuse_missing() or "")


def listed(mountpoint: Path) -> bool:
    """Whether `/proc/mounts` lists a mount at `mountpoint` (resolved; spaces are escaped there)."""
    target = str(mountpoint.resolve()).replace(" ", "\\040")
    with open("/proc/mounts", encoding="utf-8") as mounts:
        return any(line.split(" ")[1] == target for line in mounts)


def make_pack(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    """A pack of a safetensors file and a nested raw file. Returns its path and each member's bytes."""
    weights = make_safetensors(
        tmp_path / "model.safetensors", {"w": ("F32", [8192], compressible_bytes(3, 32768))}
    )
    raw = tmp_path / "raw.bin"
    raw.write_bytes(seeded_bytes(4, 50_000))
    pack = tmp_path / "pack.ziptensors"
    ziptensors.pack([weights, (raw, "sub/raw.bin")], pack, threads=1)
    return pack, {"model.safetensors": weights.read_bytes(), "sub/raw.bin": raw.read_bytes()}


def make_mountpoint(tmp_path: Path) -> Path:
    """An empty directory to mount on."""
    mountpoint = tmp_path / "mnt"
    mountpoint.mkdir()
    return mountpoint


@needs_fuse
def test_mount_reads_members_and_unmounts(tmp_path: Path) -> None:
    pack, members = make_pack(tmp_path)
    mountpoint = make_mountpoint(tmp_path)

    with ziptensors.mount(pack, mountpoint, threads=1) as mounted:
        assert isinstance(mounted, Mount)
        assert mounted.is_mounted
        assert mounted.mountpoint == mountpoint
        assert listed(mountpoint)
        assert sorted(os.listdir(mountpoint)) == ["model.safetensors", "sub"]
        for name, data in members.items():
            assert (mountpoint / name).read_bytes() == data
        with open(mountpoint / "sub" / "raw.bin", "rb") as file:
            file.seek(40_000)
            assert file.read(100) == members["sub/raw.bin"][40_000:40_100]
        with pytest.raises(OSError):
            (mountpoint / "new").write_bytes(b"x")

    assert not mounted.is_mounted
    assert os.listdir(mountpoint) == []
    assert not listed(mountpoint)
    # A second unmount, after the `with` block's, does nothing.
    mounted.unmount()


@needs_fuse
@pytest.mark.parametrize("raised", [KeyboardInterrupt, RuntimeError])
def test_a_raising_block_still_unmounts(tmp_path: Path, raised: type[BaseException]) -> None:
    pack, _members = make_pack(tmp_path)
    mountpoint = make_mountpoint(tmp_path)

    with pytest.raises(raised):
        with ziptensors.mount(pack, mountpoint, threads=1) as mounted:
            assert listed(mountpoint)
            raise raised("inside the block")

    assert not mounted.is_mounted
    assert os.listdir(mountpoint) == []
    assert not listed(mountpoint)


@needs_fuse
def test_explicit_unmount_is_idempotent(tmp_path: Path) -> None:
    pack, _members = make_pack(tmp_path)
    mountpoint = make_mountpoint(tmp_path)

    with ziptensors.mount(pack, mountpoint) as mounted:
        mounted.unmount()
        assert not mounted.is_mounted
        assert not listed(mountpoint)
        mounted.unmount()
    assert os.listdir(mountpoint) == []


@needs_fuse
def test_concurrent_unmounts_wait_for_the_running_one(tmp_path: Path) -> None:
    """A second `unmount()` while the first still waits for an open file returns only when the
    first does, after the file is closed, and both see the mount ended."""
    pack, members = make_pack(tmp_path)
    mountpoint = make_mountpoint(tmp_path)
    mounted = ziptensors.mount(pack, mountpoint)
    returned: dict[str, float] = {}
    errors: list[BaseException] = []

    def unmount(name: str) -> None:
        try:
            mounted.unmount()
        except BaseException as error:  # handed to the main thread, which fails the test with it
            errors.append(error)
        returned[name] = time.monotonic()

    first = threading.Thread(target=unmount, args=("first",))
    second = threading.Thread(target=unmount, args=("second",))
    try:
        held = open(mountpoint / "sub" / "raw.bin", "rb")
        try:
            first.start()
            # The lazy unmount detaches the mountpoint at once, then waits for `held`.
            deadline = time.monotonic() + THREAD_DEADLINE
            while listed(mountpoint) and time.monotonic() < deadline:
                time.sleep(0.02)
            assert not listed(mountpoint)
            second.start()
            time.sleep(WAIT_SHOWS)
            assert first.is_alive() and second.is_alive()
            assert mounted.is_mounted
            # The open file is still served while the unmount waits for it.
            assert held.read() == members["sub/raw.bin"]
            closed = time.monotonic()
        finally:
            held.close()
        first.join(THREAD_DEADLINE)
        second.join(THREAD_DEADLINE)
        assert not first.is_alive() and not second.is_alive()
    finally:
        # Ends the mount if an assertion above failed before both threads did; else does nothing.
        mounted.unmount()
    assert errors == []
    assert returned["first"] >= closed and returned["second"] >= closed
    assert not mounted.is_mounted
    assert not listed(mountpoint)
    assert os.listdir(mountpoint) == []


@needs_fuse
def test_a_dropped_mount_is_unmounted(tmp_path: Path) -> None:
    pack, _members = make_pack(tmp_path)
    mountpoint = make_mountpoint(tmp_path)

    mounted = ziptensors.mount(pack, mountpoint)
    assert listed(mountpoint)
    del mounted
    gc.collect()
    deadline = time.monotonic() + DROP_DEADLINE
    while listed(mountpoint) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not listed(mountpoint)
    assert os.listdir(mountpoint) == []


@needs_fuse
def test_mount_errors(tmp_path: Path) -> None:
    pack, _members = make_pack(tmp_path)
    mountpoint = make_mountpoint(tmp_path)

    with pytest.raises(ZiptensorsIoError):
        ziptensors.mount(tmp_path / "missing.ziptensors", mountpoint)
    with pytest.raises(BudgetError):
        ziptensors.mount(pack, mountpoint, cache_size=1)
    with pytest.raises(MountError) as refused:
        ziptensors.mount(pack, tmp_path / "no-such-directory")
    assert refused.value.code == CODE_IO
    assert "no-such-directory" in str(refused.value)
    with pytest.raises(ValueError):
        ziptensors.mount_overlay(tmp_path, mountpoint, max_open_containers=0)
    with pytest.raises(MountError):
        ziptensors.mount_overlay(pack, mountpoint)
    assert not listed(mountpoint)
    assert os.listdir(mountpoint) == []


@needs_fuse
def test_overlay_shows_originals_and_passes_files_through(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    originals = make_tree(root)
    report = ziptensors.compress_tree(root, threads=1)
    assert report.totals.compressed == 2 and not report.has_failures
    assert not (root / "model.safetensors").exists()
    mountpoint = make_mountpoint(tmp_path)

    with ziptensors.mount_overlay(root, mountpoint, threads=1) as mounted:
        assert listed(mountpoint)
        assert sorted(os.listdir(mountpoint)) == ["config.json", "model.safetensors", "unet"]
        assert sorted(os.listdir(mountpoint / "unet")) == ["README.txt", "unet.safetensors"]
        for path, data in originals.items():
            assert (mountpoint / path.relative_to(root)).read_bytes() == data

    assert not mounted.is_mounted
    assert os.listdir(mountpoint) == []
    assert not listed(mountpoint)


@pytest.mark.skipif(sys.platform != "darwin", reason="only macOS has no mount backend")
def test_macos_refuses_with_mount_error(tmp_path: Path) -> None:
    for call in (ziptensors.mount, ziptensors.mount_overlay):
        with pytest.raises(MountError) as refused:
            call(tmp_path, tmp_path)
        assert refused.value.code == CODE_UNSUPPORTED_FEATURE
        assert isinstance(refused.value, ZiptensorsError)
