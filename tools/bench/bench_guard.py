"""File: bench_guard.py
Purpose: what stands between the benchmark and the machine: path confinement, free space and memory
checks, the idle disk check, page cache eviction, and the disk record of a measured run.

Main responsibilities:
- refuse an output directory that overlaps a source root;
- the single delete of the benchmark, confined to the output directory by a resolved path check;
- refuse a step that would run out of disk or memory, from sizes computed by the caller;
- wait for an idle disk and for every md array to be idle, and evict a file with `fsync` plus
  `posix_fadvise`, confirmed by `fincore`;
- record the disk traffic of the measured devices over a whole run, and mark the run `disturbed` when
  traffic the measured process did not cause exceeds the threshold, or an md array left `idle`.

Key structures: `BenchError`, `GateError`, `Eviction`, `DiskWatch`.

Key functions: `check_layout`, `delete_inside`, `require_free`, `require_memory`, `wait_idle`,
`wait_md_idle`, `evict`, `watch_disks`, `disk_record`, `disturbance`.

Implementation notes: Linux only, like the benchmark. `fsync` on a read-only descriptor of a source
file writes nothing, as the benchmark never dirties one, and eviction changes only the page cache, so
both are allowed on the source roots. Nothing else here touches a root. `/sys` and `/proc/diskstats`
are parameters with the real paths as defaults, so tests run on a fake tree.

An md check or resync reads the member disks, not the md device, so its traffic does not show in the
counters of the md device a measurement reads; it is caught by `sync_action` instead, before the run
(the wait) and after it (the record). The foreign traffic threshold is the idle criterion kept over the
whole run: `FOREIGN_FLOOR_BYTES` plus the idle rate times the run's seconds. A disk counted idle before
the run must stay that idle on average during it, and the floor absorbs the small writes any process
causes on a short run.
"""

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Slack on top of every computed disk need: logs, state files, filesystem rounding.
FREE_MARGIN_BYTES = 64 * 1024 * 1024
# Share of a computed disk need added as slack, so the margin grows with the inputs.
FREE_MARGIN_SHARE = 0.02
# Memory left to the system and the page cache on top of a destination buffer.
MEMORY_MARGIN_BYTES = 2 * 1024 * 1024 * 1024
# Resident bytes tolerated after an eviction: far below what could change a timing, but enough for a
# page some other process maps for an instant.
EVICT_TOLERANCE_BYTES = 1024 * 1024
EVICT_ATTEMPTS = 3
SECTOR_BYTES = 512
# Column of /proc/diskstats (0-based, after major, minor, name) of sectors read and sectors written.
DISKSTATS_SECTORS_READ = 5
DISKSTATS_SECTORS_WRITTEN = 9
SYS_ROOT = Path("/sys")
DISKSTATS = Path("/proc/diskstats")
# What `/sys/block/md*/md/sync_action` says of an array with no check, resync or recovery running.
MD_IDLE = "idle"
# Seconds between two looks at the md arrays while one is busy. A check takes hours.
MD_POLL_SECONDS = 30.0
# Foreign disk bytes always tolerated in a run: log lines, state files, metadata of other processes.
FOREIGN_FLOOR_BYTES = 64 * 1024 * 1024


class BenchError(Exception):
    """A step cannot go on. The message says what failed and what to check."""


class GateError(BenchError):
    """A step finished and stored its results, and an acceptance gate failed."""


def check_layout(roots: tuple[Path, ...], out: Path) -> None:
    """Refuse an output directory inside a root, or a root inside it.

    Both are compared resolved, so a link cannot hide the overlap. `out` need not exist yet.

    Raises BenchError on an overlap, and if a root is missing or not a directory.
    """
    resolved_out = out.resolve()
    for root in roots:
        if not root.is_dir():
            raise BenchError(f"source root {root} is not a directory")
        resolved_root = root.resolve()
        if resolved_out == resolved_root or resolved_out.is_relative_to(resolved_root):
            raise BenchError(f"output directory {out} is inside source root {root}")
        if resolved_root.is_relative_to(resolved_out):
            raise BenchError(f"source root {root} is inside output directory {out}")


def confined(path: Path, out: Path) -> Path:
    """Return `path` with its parent resolved, proven to lie strictly inside `out`.

    Only the parent is resolved, so that a link inside `out` stands for itself and not for its target.
    Neither a link above it nor `..` can leave `out`.

    Raises BenchError if it lies elsewhere or is `out` itself, FileNotFoundError if `out` is missing.
    """
    resolved_out = out.resolve(strict=True)
    if path.name in ("", ".."):
        raise BenchError(f"refusing to delete {path}: not a plain name")
    parent = path.parent.resolve(strict=True)
    if parent != resolved_out and not parent.is_relative_to(resolved_out):
        raise BenchError(f"refusing to delete {path}: not inside output directory {out}")
    return parent / path.name


def delete_inside(path: Path, out: Path) -> None:
    """Delete the file, link or directory `path`, which must lie inside `out`.

    `rmtree` never follows links, so nothing outside `out` can be reached through a link in the tree.
    A missing `path` is no error.

    Raises BenchError outside `out`, OSError if the removal fails.
    """
    target = confined(path, out)
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    log.info("deleted %s", target)


def tree_bytes(path: Path) -> int:
    """Return the bytes of the regular files under `path`, or of `path` itself. Links do not count."""
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(
        (Path(parent) / name).lstat().st_size
        for parent, _, names in path.walk()
        for name in names
    )


def required_space(payload_bytes: int) -> int:
    """Return the free bytes a step writing `payload_bytes` needs, slack included."""
    return payload_bytes + int(payload_bytes * FREE_MARGIN_SHARE) + FREE_MARGIN_BYTES


def require_free(out: Path, payload_bytes: int, what: str) -> None:
    """Refuse to go on unless the filesystem of `out` can take `payload_bytes` more.

    Raises BenchError naming the step, what it needs and what is free.
    """
    need = required_space(payload_bytes)
    free = shutil.disk_usage(out).free
    log.info("%s: needs %d bytes on disk, %d free", what, need, free)
    if free < need:
        raise BenchError(f"{what} needs {need} bytes free in {out}, only {free} are")


def available_memory() -> int:
    """Return `MemAvailable` of /proc/meminfo in bytes.

    Raises BenchError if the file has no such line.
    """
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise BenchError("/proc/meminfo has no MemAvailable line")


def require_memory(destination_bytes: int, what: str) -> None:
    """Refuse a measurement whose destination buffer the machine cannot hold without swapping.

    Swap would turn a load time into a memory time, so this holds in dry runs too.

    Raises BenchError naming what it needs and what is available.
    """
    need = destination_bytes + MEMORY_MARGIN_BYTES
    available = available_memory()
    if available < need:
        raise BenchError(f"{what} needs {need} bytes of memory, only {available} are available")


def disk_name(path: Path) -> str | None:
    """Return the /proc/diskstats name of the block device holding `path`, None if it has none."""
    device = path.stat().st_dev
    wanted = (os.major(device), os.minor(device))
    for line in Path("/proc/diskstats").read_text(encoding="ascii").splitlines():
        fields = line.split()
        if (int(fields[0]), int(fields[1])) == wanted:
            return fields[2]
    return None


def disk_bytes(names: set[str], diskstats: Path = DISKSTATS) -> dict[str, tuple[int, int]]:
    """Return the bytes read and written so far by each block device of `names`.

    Raises BenchError if a device is not in `diskstats`.
    """
    counters: dict[str, tuple[int, int]] = {}
    for line in diskstats.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if fields[2] in names:
            read = int(fields[DISKSTATS_SECTORS_READ]) * SECTOR_BYTES
            written = int(fields[DISKSTATS_SECTORS_WRITTEN]) * SECTOR_BYTES
            counters[fields[2]] = (read, written)
    missing = names - counters.keys()
    if missing:
        raise BenchError(f"block devices {sorted(missing)} left {diskstats}")
    return counters


def disk_sectors(name: str) -> int:
    """Return the sectors read plus written so far by the block device `name`.

    Raises BenchError if the device is gone.
    """
    read, written = disk_bytes({name})[name]
    return (read + written) // SECTOR_BYTES


def disk_names(paths: list[Path], strict: bool) -> set[str]:
    """Return the block devices holding `paths`.

    Raises BenchError in a strict run if a path has no block device; otherwise that only logs a
    warning, as a dry run may sit on a memory filesystem.
    """
    names: set[str] = set()
    for path in paths:
        name = disk_name(path)
        if name is None:
            if strict:
                raise BenchError(f"{path} is on no block device, cannot check that it is idle")
            log.warning("%s is on no block device, not checking that it is idle", path)
        else:
            names.add(name)
    return names


def md_sync_actions(sys_root: Path = SYS_ROOT) -> dict[str, str]:
    """Return `sync_action` of every md array under `<sys_root>/block`, by array name.

    An md device without the file (a stopped array, an external metadata container) has nothing to
    check and is left out.

    Raises BenchError if a file exists and cannot be read.
    """
    actions: dict[str, str] = {}
    for array in sorted((sys_root / "block").glob("md*")):
        action = array / "md" / "sync_action"
        if not action.is_file():
            continue
        try:
            actions[array.name] = action.read_text(encoding="ascii").strip()
        except OSError as error:
            raise BenchError(f"cannot read {action}: {error}") from error
    return actions


def busy_arrays(sys_root: Path = SYS_ROOT) -> dict[str, str]:
    """Return the md arrays whose `sync_action` is not idle, with what they do."""
    return {name: action for name, action in md_sync_actions(sys_root).items() if action != MD_IDLE}


def wait_md_idle(
    deadline: float, strict: bool, sys_root: Path = SYS_ROOT, poll_seconds: float = MD_POLL_SECONDS
) -> dict[str, str]:
    """Wait until every md array is idle or `deadline`, a `time.monotonic` value, has passed.

    A check or resync reads every member disk for hours and evicts the page cache of a measurement,
    whatever array it reads. Returns the arrays still busy, empty when all are idle.

    Raises BenchError in a strict run if one is still busy at the deadline. A dry run only warns.
    """
    while True:
        busy = busy_arrays(sys_root)
        if not busy:
            return busy
        if time.monotonic() >= deadline:
            message = f"md arrays not idle: {busy}; wait for the check or resync to finish"
            if strict:
                raise BenchError(message)
            log.warning("%s", message)
            return busy
        log.info("md arrays busy: %s, waiting", busy)
        time.sleep(poll_seconds)


def wait_idle(
    paths: list[Path],
    max_bytes_per_second: int,
    window_seconds: float,
    timeout_seconds: float,
    strict: bool,
    sys_root: Path = SYS_ROOT,
) -> float:
    """Flush dirty pages, then wait until every md array is idle and no device under `paths` moves more
    than the given rate, both within one `timeout_seconds`.

    A timing taken while something else moves the disk heads is worthless, so a strict run refuses it.
    Returns the highest rate seen in the last window, in bytes per second.

    Raises BenchError in a strict run if a path has no block device, an md array stays busy, or the disk
    stays busy past `timeout_seconds`. Otherwise these only log a warning: a dry run may sit on a memory
    filesystem.
    """
    os.sync()
    deadline = time.monotonic() + timeout_seconds
    names = disk_names(paths, strict)
    wait_md_idle(deadline, strict, sys_root)
    if not names:
        return 0.0
    while True:
        before = {name: disk_sectors(name) for name in names}
        time.sleep(window_seconds)
        rate = max(
            (disk_sectors(name) - before[name]) * SECTOR_BYTES / window_seconds for name in names
        )
        if rate <= max_bytes_per_second:
            log.info("disk idle: %.0f bytes/s on %s", rate, sorted(names))
            # A check may have started during the window, and its traffic is on other disks. After a
            # wait for it the window is taken again; a dry run past its deadline goes on.
            if busy_arrays(sys_root) and (strict or time.monotonic() < deadline):
                wait_md_idle(deadline, strict, sys_root)
                continue
            return rate
        if time.monotonic() >= deadline:
            message = f"disk {sorted(names)} still busy after {timeout_seconds} s: {rate:.0f} bytes/s"
            if strict:
                raise BenchError(message)
            log.warning("%s", message)
            return rate
        log.info("disk busy: %.0f bytes/s on %s, waiting", rate, sorted(names))


@dataclass(frozen=True)
class Eviction:
    """What an eviction left behind."""

    resident_bytes: int
    attempts: int


def resident_bytes(path: Path) -> int:
    """Return the bytes of `path` in the page cache according to `fincore`.

    Raises BenchError if `fincore` fails or prints no number.
    """
    done = subprocess.run(
        ["fincore", "--bytes", "--noheadings", "--output", "RES", "--", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise BenchError(f"fincore failed on {path}: {done.stderr.strip()}")
    text = done.stdout.strip()
    if not text.isdigit():
        raise BenchError(f"fincore printed {text!r} for {path}, expected a number")
    return int(text)


def evict(path: Path, strict: bool) -> Eviction:
    """Drop `path` from the page cache and confirm with `fincore`.

    Needs no root. `fsync` comes first because a dirty page cannot be dropped, and the file is opened
    read-only, so this works on source files.

    Raises BenchError in a strict run if pages stay resident, for instance when another process maps
    the file. A dry run only warns, since a memory filesystem cannot evict at all.
    """
    resident = 0
    for attempt in range(1, EVICT_ATTEMPTS + 1):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(descriptor)
        resident = resident_bytes(path)
        if resident <= EVICT_TOLERANCE_BYTES:
            return Eviction(resident, attempt)
    if strict:
        raise BenchError(f"{resident} bytes of {path} stay resident after eviction")
    log.warning("%d bytes of %s stay resident after eviction", resident, path)
    return Eviction(resident, EVICT_ATTEMPTS)


@dataclass(frozen=True)
class DiskWatch:
    """The counters of the measured devices when a run started."""

    names: frozenset[str]
    counters: dict[str, tuple[int, int]]


def watch_disks(paths: list[Path], strict: bool, diskstats: Path = DISKSTATS) -> DiskWatch:
    """Start recording the traffic of the devices holding `paths`. Raises as `disk_names` does."""
    names = disk_names(paths, strict)
    return DiskWatch(frozenset(names), disk_bytes(set(names), diskstats))


def disturbance(
    device_read: int,
    device_written: int,
    own_read: int,
    own_written: int,
    seconds: float,
    idle_rate: int,
    busy: dict[str, str],
) -> dict[str, Any]:
    """Return the disk record of a run: traffic of the devices, of the process, and the verdict.

    Foreign bytes are the device bytes the process did not cause, never negative: a process may also
    read from a device that is not measured. The run is `disturbed` when they exceed
    `FOREIGN_FLOOR_BYTES + idle_rate * seconds`, or when `busy` names an md array not idle at its end.
    """
    foreign = max(device_read + device_written - own_read - own_written, 0)
    limit = FOREIGN_FLOOR_BYTES + int(idle_rate * seconds)
    reasons = []
    if foreign > limit:
        reasons.append(f"foreign disk traffic {foreign} B over {limit} B")
    if busy:
        reasons.append(f"md arrays not idle at the end: {busy}")
    return {
        "device_read_bytes": device_read,
        "device_written_bytes": device_written,
        "own_read_bytes": own_read,
        "own_written_bytes": own_written,
        "foreign_bytes": foreign,
        "foreign_limit_bytes": limit,
        "md_busy": busy,
        "disturbed": bool(reasons),
        "reasons": reasons,
    }


def disk_record(
    watch: DiskWatch,
    own_read: int,
    own_written: int,
    seconds: float,
    idle_rate: int,
    sys_root: Path = SYS_ROOT,
    diskstats: Path = DISKSTATS,
) -> dict[str, Any]:
    """Finish a watch: the disk record of the run, with the measured devices, see `disturbance`.

    Raises BenchError if a device left `diskstats` or an md state cannot be read.
    """
    after = disk_bytes(set(watch.names), diskstats)
    device_read = sum(after[name][0] - watch.counters[name][0] for name in watch.names)
    device_written = sum(after[name][1] - watch.counters[name][1] for name in watch.names)
    record = disturbance(
        device_read, device_written, own_read, own_written, seconds, idle_rate, busy_arrays(sys_root)
    )
    record["devices"] = sorted(watch.names)
    if record["disturbed"]:
        log.warning("run disturbed: %s", "; ".join(record["reasons"]))
    return record
