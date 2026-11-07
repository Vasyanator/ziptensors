"""File: bench_run.py
Purpose: what every step shares: the settings, the layout of the output directory, the results store
that makes steps resumable, and child processes measured for wall time and peak memory.

Key structures: `Settings`, `Targets`, `Store`, `Measured`.

Key functions: `run_measured`.

Implementation notes: results are JSON files replaced atomically, so a run killed at any moment leaves
each measurement either recorded or not. Children write to log files, not pipes: `os.wait4` gives the
memory of exactly that child, and a child blocked on a full pipe could never be waited for.
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench_guard import BenchError

log = logging.getLogger(__name__)

# Lines of a failed child's stderr repeated in the error.
ERROR_TAIL_LINES = 5
# Unit of `ru_inblock` and `ru_oublock` on Linux: the kernel reports its byte counters shifted by 9.
RUSAGE_BLOCK_BYTES = 512


@dataclass(frozen=True)
class Targets:
    """Member names of the load, random and encode-memory measurements. Machine-local, so always
    arguments."""

    bf16: str | None = None
    shared_f32: str | None = None
    q8: str | None = None
    fragmented: str | None = None
    random: str | None = None
    encode: str | None = None


@dataclass(frozen=True)
class Settings:
    """One run of the benchmark. `strict` is false only in a dry run: then timing thresholds, idle
    disk and eviction are reported and never stop a step."""

    roots: tuple[Path, Path]
    out: Path
    cli: Path
    driver: Path
    report: Path
    expected_files: int
    threads: int = 0
    strict: bool = True
    force: bool = False
    runs: int = 3
    random_count: int = 10_000
    random_sizes: tuple[int, ...] = (4096, 1 << 20)
    random_seed: int = 20_261_102
    idle_rate: int = 2 * 1024 * 1024
    idle_window: float = 5.0
    idle_timeout: float = 600.0
    targets: Targets = Targets()

    @property
    def state(self) -> Path:
        """Results, snapshots, the manifest."""
        return self.out / "state"

    @property
    def logs(self) -> Path:
        """stdout and stderr of every child."""
        return self.out / "logs"

    @property
    def packs(self) -> Path:
        """Containers. Removed by `clean`."""
        return self.out / "packs"

    @property
    def restored(self) -> Path:
        """Restored copies. Removed after a full match, or by `clean`."""
        return self.out / "restored"

    def prepare(self) -> None:
        """Create the output directories. Raises OSError if that fails."""
        for path in (self.out, self.state, self.logs, self.packs):
            path.mkdir(parents=True, exist_ok=True)


class Store:
    """JSON results of one step, in `<state>/<name>.json`, written after every measurement."""

    def __init__(self, settings: Settings, name: str) -> None:
        self.path = settings.state / f"{name}.json"
        self.data: dict[str, Any] = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        """Replace the file atomically. Raises OSError if that fails."""
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=1, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.path)

    def record(self, key: str, value: Any) -> None:
        """Store one measurement and save."""
        self.data[key] = value
        self.save()


@dataclass(frozen=True)
class Measured:
    """A finished child: wall time, peak RSS and the disk bytes it caused, from `wait4`, and its
    stdout. Disk bytes are what the child submitted to block devices, read and written, whatever
    device; page cache hits are not in them."""

    seconds: float
    peak_rss_bytes: int
    stdout: str
    read_bytes: int = 0
    written_bytes: int = 0


def run_measured(command: list[str], logs: Path, name: str) -> Measured:
    """Run `command`, with stdout and stderr in `<logs>/<name>.out` and `.err`, and measure it.

    `ru_maxrss` of a child starts at the RSS of this process, as the kernel carries the peak across
    exec. This script stays far below any peak worth measuring here, and the driver reports its own
    `VmHWM` anyway.

    Raises BenchError if the child fails, with the tail of its stderr.
    """
    out_path, err_path = logs / f"{name}.out", logs / f"{name}.err"
    log.info("running %s", name)
    with out_path.open("wb") as out, err_path.open("wb") as err:
        start = time.perf_counter()
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
        _, status, usage = os.wait4(child.pid, 0)
        seconds = time.perf_counter() - start
    # `wait4` reaped it, so `Popen` must not try to.
    child.returncode = os.waitstatus_to_exitcode(status)
    if child.returncode != 0:
        lines = err_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-ERROR_TAIL_LINES:])
        raise BenchError(f"{name} failed with exit code {child.returncode}:\n{tail}")
    log.info("finished %s in %.1f s", name, seconds)
    return Measured(
        seconds,
        usage.ru_maxrss * 1024,
        out_path.read_text(encoding="utf-8"),
        usage.ru_inblock * RUSAGE_BLOCK_BYTES,
        usage.ru_oublock * RUSAGE_BLOCK_BYTES,
    )
