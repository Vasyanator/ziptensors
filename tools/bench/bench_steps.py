"""File: bench_steps.py
Purpose: the steps of the acceptance benchmark: `snapshot`, `pack`, `verify-exact`, `savings`, `load`,
`random`, `encode-memory`, `clean`.

Main responsibilities: each step checks its disk needs from the inputs, skips what is already recorded
unless forced, runs the CLI or the driver, records one result per measurement, and raises `GateError`
after recording if a gate of a strict run fails or cannot be judged.

Key functions: `step_snapshot`, `step_pack`, `step_verify_exact`, `step_savings`, `step_load`,
`step_random`, `step_encode_memory`, `step_clean`, `measure`, `pick_subset`, `build_manifest`,
`snapshot_difference`.

Implementation notes: the manifest comes from the first snapshot, with the selection `pack` applies: all
regular files that are not `*.incomplete`. The sha256 pass is resumable per file because it takes an
hour. Nothing here opens a source file for writing; only `bench_guard.evict` touches one, read-only.
A timed run of `load` or `random` counts as recorded only with the disk record of the guard and not
disturbed (`bench_estimates.recorded`): a rerun without `--force` measures the others again and keeps
everything else, plans, seek map and private container included.
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench_estimates import (
    Gate,
    class_rows,
    encode_memory_gate,
    load_gates,
    random_gates,
    recorded,
    savings_gates,
)
from bench_guard import (
    BenchError,
    GateError,
    delete_inside,
    disk_record,
    evict,
    require_free,
    require_memory,
    tree_bytes,
    wait_idle,
    watch_disks,
)
from bench_report import render_encode_memory, render_load, render_random, render_savings
from bench_run import Measured, Settings, Store, run_measured

log = logging.getLogger(__name__)

MODES = ("fast", "balanced", "max")
FULL_KEY = "full-max"
INCOMPLETE_SUFFIX = ".incomplete"
SUBSET_SIZE = 12
# Below this a file says nothing about tensor coding, so it joins a subset only if its group has no other.
SUBSET_MIN_BYTES = 16 * 1024 * 1024
SUBSET_SUFFIXES = (".safetensors", ".gguf", ".onnx")
# Bytes printed per snapshot row, for the space check, well above a real path.
SNAPSHOT_ROW_BYTES = 512
SHA256_HEX_DIGITS = 64
ENCODE_MEMORY_MODE = "max"
ENCODE_MEMORY_OUTPUT = "encode-memory.ztz"


@dataclass(frozen=True)
class Entry:
    """One file of the set: its member name, the file, its length."""

    member: str
    path: Path
    size: int


def enforce(settings: Settings, gates: list[Gate], what: str) -> None:
    """Log every gate, and raise GateError in a strict run if one failed or could not be judged, so the
    exit code says the results need a look. Results are stored by then."""
    for gate in gates:
        log.info("gate %s: %s, target %s: %s", gate.name, gate.measured, gate.target, gate.result)
    failed = [gate.name for gate in gates if gate.failed]
    invalid = [f"{gate.name} ({gate.invalid})" for gate in gates if gate.invalid is not None]
    if (failed or invalid) and settings.strict:
        parts = [f"failed gates: {', '.join(failed)}"] if failed else []
        if invalid:
            parts.append(f"not judged, rerun the step without --force: {', '.join(invalid)}")
        raise GateError(f"{what}: {'; '.join(parts)}")


def read_snapshot(path: Path) -> list[bytes]:
    """Return the rows of a snapshot file."""
    return path.read_bytes().split(b"\0")


def take_snapshot(root: Path, destination: Path) -> int:
    """Write the sorted `find -printf` rows of `root` to `destination`, return the row count.

    Rows are `<type> <size> <mtime> <path>`, NUL separated as a name may hold a newline, and sorted as
    `find` lists in directory order, which an unchanged directory may change.

    Raises BenchError if `find` fails, for instance on an unreadable directory.
    """
    done = subprocess.run(
        ["find", str(root), "-printf", "%y %s %T@ %p\\0"], capture_output=True, check=False
    )
    if done.returncode != 0:
        detail = done.stderr.decode(errors="replace").strip()
        raise BenchError(f"find failed on {root}: {detail}")
    rows = sorted(row for row in done.stdout.split(b"\0") if row)
    destination.write_bytes(b"\0".join(rows))
    return len(rows)


def snapshot_difference(before: list[bytes], after: list[bytes]) -> list[str]:
    """Return one text per row present in only one snapshot, `-` for before, `+` for after."""
    gone, new = set(before) - set(after), set(after) - set(before)
    rows = [("-", row) for row in gone] + [("+", row) for row in new]
    return [f"{sign} {row.decode(errors='replace')}" for sign, row in sorted(rows)]


def build_manifest(root_rows: list[tuple[Path, list[bytes]]]) -> list[dict[str, Any]]:
    """Return the files `pack` will take from each root, as manifest records, by member name.

    Raises BenchError for a name `pack` would refuse, which is better learned now than from a pack:
    not UTF-8, holding `:`, or the same member name under both roots.
    """
    records: dict[str, dict[str, Any]] = {}
    for root, rows in root_rows:
        prefix = bytes(root) + b"/"
        for row in rows:
            kind, size, _, path = row.split(b" ", 3)
            if kind != b"f" or path.endswith(INCOMPLETE_SUFFIX.encode()):
                continue
            try:
                text = path.decode("utf-8")
            except UnicodeDecodeError as error:
                raise BenchError(f"a file name is not UTF-8: {path!r}") from error
            member = text.removeprefix(os.fsdecode(prefix))
            if ":" in member:
                raise BenchError(f"pack refuses the member name {member!r}")
            if member in records:
                raise BenchError(f"both roots hold {member!r}, pack refuses a duplicate name")
            records[member] = {"member": member, "path": text, "size": int(size)}
    return [records[name] for name in sorted(records)]


def load_manifest(settings: Settings) -> list[Entry]:
    """Return the manifest of the first snapshot.

    Raises BenchError if `snapshot --phase before` has not run.
    """
    path = settings.state / "manifest.json"
    if not path.exists():
        raise BenchError("no manifest: run `snapshot --phase before` first")
    records = json.loads(path.read_text(encoding="utf-8"))
    return [Entry(record["member"], Path(record["path"]), record["size"]) for record in records]


def snapshot_paths(settings: Settings, phase: str) -> list[tuple[Path, Path]]:
    """Return `(root, snapshot file)` for both roots."""
    return [
        (root, settings.state / f"snapshot-{phase}-{number}.bin")
        for number, root in enumerate(settings.roots, start=1)
    ]


def step_snapshot(settings: Settings, phase: str) -> None:
    """Snapshot both roots. `before` also writes the manifest, `after` compares.

    Raises BenchError if the file count is not the expected one, or a root changed.
    """
    require_free(settings.out, 0, "snapshot")
    if phase == "before" and (settings.state / "manifest.json").exists() and not settings.force:
        raise BenchError("a first snapshot exists; replacing it would hide a change, --force does that")
    root_rows: list[tuple[Path, list[bytes]]] = []
    for root, path in snapshot_paths(settings, phase):
        count = take_snapshot(root, path)
        log.info("snapshot %s of %s: %d rows", phase, root, count)
        require_free(settings.out, count * SNAPSHOT_ROW_BYTES, "snapshot")
        root_rows.append((root, read_snapshot(path)))
    if phase == "before":
        records = build_manifest(root_rows)
        if len(records) != settings.expected_files:
            raise BenchError(f"the roots hold {len(records)} files, expected {settings.expected_files}")
        manifest = settings.state / "manifest.json"
        manifest.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        return
    differences: list[str] = []
    before_paths = snapshot_paths(settings, "before")
    for (_, rows), (_, before_path) in zip(root_rows, before_paths, strict=True):
        if not before_path.exists():
            raise BenchError("no earlier snapshot: run `snapshot --phase before` first")
        differences += snapshot_difference(read_snapshot(before_path), rows)
    store = Store(settings, "snapshot")
    store.record("identical", not differences)
    store.record("difference_count", len(differences))
    if differences:
        shown = "\n".join(differences[:20])
        raise BenchError(f"{len(differences)} snapshot rows differ, first ones:\n{shown}")
    log.info("snapshots identical")


def median_first(entries: list[Entry]) -> list[Entry]:
    """Return `entries` ordered from the median size outward, so that a pick is typical of its group."""
    ordered = sorted(entries, key=lambda entry: (entry.size, entry.member))
    middle = (len(ordered) - 1) / 2
    distances = sorted(range(len(ordered)), key=lambda position: abs(position - middle))
    return [ordered[position] for position in distances]


def pick_subset(entries: list[Entry], count: int = SUBSET_SIZE) -> list[Entry]:
    """Pick `count` files, round robin over the file types, nearest the median size of each.

    Deterministic, and holding no name, as a list of model files would be a machine-local path. Files
    given to `pack` directly are named after themselves, so the picks have distinct base names.

    Raises BenchError if the set has too few distinct names.
    """
    groups: dict[str, list[Entry]] = {}
    for entry in entries:
        suffix = entry.path.suffix if entry.path.suffix in SUBSET_SUFFIXES else "other"
        groups.setdefault(suffix, []).append(entry)
    queues = []
    for suffix in sorted(groups):
        large = [entry for entry in groups[suffix] if entry.size >= SUBSET_MIN_BYTES]
        queues.append(median_first(large or groups[suffix]))
    chosen: list[Entry] = []
    names: set[str] = set()
    while len(chosen) < count and any(queues):
        for queue in queues:
            while queue and len(chosen) < count:
                entry = queue.pop(0)
                if entry.path.name not in names:
                    names.add(entry.path.name)
                    chosen.append(entry)
                    break
    if len(chosen) < count:
        raise BenchError(f"only {len(chosen)} files with distinct names for a subset of {count}")
    return sorted(chosen, key=lambda entry: entry.member)


def pack_result(settings: Settings, measured: Measured, output: Path, raw: int, files: int) -> dict[str, Any]:
    """Return the record of a finished pack."""
    file_bytes = output.stat().st_size
    return {
        "seconds": measured.seconds,
        "peak_rss_bytes": measured.peak_rss_bytes,
        "file_bytes": file_bytes,
        "raw_bytes": raw,
        "files": files,
        "threads": settings.threads or len(os.sched_getaffinity(0)),
        "saved_percent": (raw - file_bytes) / raw * 100.0 if raw else 0.0,
    }


def pack_commands(
    settings: Settings, entries: list[Entry], subset: list[Entry]
) -> dict[str, tuple[list[Path], str, int, int]]:
    """Return per result key the inputs, the mode, raw bytes and the file count of each pack."""
    raw = sum(entry.size for entry in entries)
    jobs = {FULL_KEY: (list(settings.roots), "max", raw, len(entries))}
    subset_paths = [entry.path for entry in subset]
    subset_raw = sum(entry.size for entry in subset)
    for mode in MODES:
        jobs[f"subset-{mode}"] = (subset_paths, mode, subset_raw, len(subset))
    return jobs


def step_pack(settings: Settings) -> None:
    """Pack the full set in `max` and the subset in all modes, skipping recorded packs.

    The worst case of a container is its raw size, so that is what the space check asks for.

    Raises BenchError for too little space, a busy disk, or a failing pack.
    """
    entries = load_manifest(settings)
    subset = pick_subset(entries)
    store = Store(settings, "pack")
    store.record("subset", [{"member": entry.member, "size": entry.size} for entry in subset])
    jobs = pack_commands(settings, entries, subset)
    pending = {key: job for key, job in jobs.items() if settings.force or key not in store.data}
    require_free(settings.out, sum(job[2] for job in pending.values()), "pack")
    for key, (inputs, mode, raw, files) in pending.items():
        output = settings.packs / f"{key}.ztz"
        wait_idle(
            [*settings.roots, settings.out],
            settings.idle_rate,
            settings.idle_window,
            settings.idle_timeout,
            settings.strict,
        )
        command = [
            str(settings.cli),
            "pack",
            str(output),
            *[str(path) for path in inputs],
            "--mode",
            mode,
            "--threads",
            str(settings.threads),
        ]
        measured = run_measured(command, settings.logs, f"pack-{key}")
        store.record(key, pack_result(settings, measured, output, raw, files))


def require_pack(settings: Settings) -> Path:
    """Return the full container.

    Raises BenchError if `pack` has not run, or `clean` removed it.
    """
    path = settings.packs / f"{FULL_KEY}.ztz"
    if not path.exists():
        raise BenchError(f"{path} is missing: run `pack` first")
    return path


def sha256_of(path: Path) -> str:
    """Return the SHA-256 of `path` from `sha256sum`, the tool independent of the code under test.

    Raises BenchError if it fails or prints no digest.
    """
    done = subprocess.run(["sha256sum", "--", str(path)], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise BenchError(f"sha256sum failed on {path}: {done.stderr.strip()}")
    return parse_sha256(done.stdout, path)


def parse_sha256(line: str, path: Path) -> str:
    """Return the digest of a `sha256sum` line, which starts with `\\` if the name needed escaping.

    Raises BenchError if there is none.
    """
    digest = line.removeprefix("\\")[:SHA256_HEX_DIGITS]
    if len(digest) != SHA256_HEX_DIGITS or any(digit not in "0123456789abcdef" for digit in digest):
        raise BenchError(f"sha256sum printed {line!r} for {path}")
    return digest


def read_hash_log(path: Path) -> dict[str, dict[str, str]]:
    """Return the records of the sha256 log by member. A line cut short by a kill is ignored and redone."""
    records: dict[str, dict[str, str]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log.warning("dropping a cut line of %s", path)
                continue
            records[record["member"]] = record
    return records


def step_verify_exact(settings: Settings) -> None:
    """Restore everything, compare every file with its source by sha256, delete the copies on a match.

    The copies stay after a mismatch, for diagnosis; `clean` removes them.

    Raises BenchError for too little space or a failing restore, GateError unless every file matched.
    """
    entries = load_manifest(settings)
    container = require_pack(settings)
    store = Store(settings, "verify")
    raw = sum(entry.size for entry in entries)
    if settings.force or "restore" not in store.data:
        already = tree_bytes(settings.restored) if settings.restored.exists() else 0
        require_free(settings.out, max(raw - already, 0), "verify-exact")
        command = [
            str(settings.cli),
            "restore",
            str(container),
            "--all",
            str(settings.restored),
            "--threads",
            str(settings.threads),
        ]
        measured = run_measured(command, settings.logs, "restore-all")
        store.record("restore", {"seconds": measured.seconds, "peak_rss_bytes": measured.peak_rss_bytes})
    hash_log = settings.state / "sha256.jsonl"
    done = {} if settings.force else read_hash_log(hash_log)
    with hash_log.open("w" if settings.force else "a", encoding="utf-8") as sink:
        for position, entry in enumerate(entries, start=1):
            if entry.member in done:
                continue
            copy = settings.restored / entry.member
            if not copy.is_file():
                raise BenchError(f"the restored copy of {entry.member} is missing")
            record = {
                "member": entry.member,
                "source": sha256_of(entry.path),
                "restored": sha256_of(copy),
            }
            done[entry.member] = record
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            log.info("hashed %d of %d: %s", position, len(entries), entry.member)
    matched = sum(record["source"] == record["restored"] for record in done.values())
    mismatched = sorted(name for name, record in done.items() if record["source"] != record["restored"])
    store.record("matched", matched)
    store.record("total", len(entries))
    store.record("mismatched", mismatched)
    if matched != len(entries) or len(entries) != settings.expected_files:
        raise GateError(
            f"{matched} of {len(entries)} files match (expected {settings.expected_files}); "
            f"copies kept in {settings.restored}; first mismatches: {mismatched[:10]}"
        )
    log.info("%d of %d files match, deleting the restored copies", matched, len(entries))
    delete_inside(settings.restored, settings.out)


def step_savings(settings: Settings) -> None:
    """Read `inspect --json` of the full pack, store it, print measured next to the estimates.

    Raises BenchError if `inspect` fails or its schema is not 1, GateError on a failed gate of a
    strict run: then the printed table shows which layer fell short.
    """
    require_free(settings.out, 0, "savings")
    container = require_pack(settings)
    command = [str(settings.cli), "inspect", str(container), "--json"]
    document = json.loads(run_measured(command, settings.logs, "inspect").stdout)
    if document["schema_version"] != 1:
        raise BenchError(f"inspect --json schema {document['schema_version']}, this tool reads 1")
    store = Store(settings, "savings")
    store.record("inspect", document)
    print(render_savings(document))
    raw = document["raw_bytes"]
    saved_percent = document["saved_bytes"] / raw * 100.0 if raw else 0.0
    enforce(settings, savings_gates(saved_percent, class_rows(document["classes"])), "savings")


def ensure_seek_map(settings: Settings, container: Path) -> dict[str, Any]:
    """Return the disk runs per member of the full pack, computing them once."""
    store = Store(settings, "seek_map")
    if settings.force or not store.data:
        command = [str(settings.driver), "seek-map", "--container", str(container)]
        document = json.loads(run_measured(command, settings.logs, "seek-map").stdout)
        store.data = document
        store.save()
    return store.data


def source_of(entries: list[Entry], member: str) -> Entry:
    """Return the entry of `member`.

    Raises BenchError for a member the set does not hold, which is a wrong argument.
    """
    for entry in entries:
        if entry.member == member:
            return entry
    raise BenchError(f"no member {member!r} in the manifest")


def measure(settings: Settings, name: str, evicted: Path, command: list[str]) -> dict[str, Any]:
    """Run one cold measurement: idle disk and md arrays, then eviction of `evicted`, then the driver,
    with the disk traffic of the devices of `evicted` and the output directory recorded over the whole
    run under `disk` (see `bench_guard.disk_record`).

    Raises BenchError for a busy disk, a failed eviction, a failing driver, and in a strict run for a
    driver built without optimizations.
    """
    devices = [evicted, settings.out]
    rate = wait_idle(
        devices,
        settings.idle_rate,
        settings.idle_window,
        settings.idle_timeout,
        settings.strict,
    )
    eviction = evict(evicted, settings.strict)
    watch = watch_disks(devices, settings.strict)
    measured = run_measured(command, settings.logs, name)
    disk = disk_record(
        watch, measured.read_bytes, measured.written_bytes, measured.seconds, settings.idle_rate
    )
    document: dict[str, Any] = json.loads(measured.stdout)
    if settings.strict and not document["release"]:
        raise BenchError(f"{settings.driver} is a debug build, timings would mean nothing")
    document["idle_bytes_per_second"] = rate
    document["resident_after_evict_bytes"] = eviction.resident_bytes
    document["disk"] = disk
    return document


def load_commands(
    settings: Settings, case: str, entry: Entry, plan: Path, private: Path | None
) -> dict[str, tuple[Path, list[str]]]:
    """Return per method what it evicts and its driver command, for one case of step 4."""
    container = settings.packs / f"{FULL_KEY}.ztz"
    driver = str(settings.driver)
    threads = ["--threads", str(settings.threads)]
    packed = ["--container", str(container), "--member", entry.member, *threads]
    methods = {
        "raw": (entry.path, [driver, "load-raw", "--file", str(entry.path), "--plan", str(plan)]),
        "decode": (container, [driver, "load-decode", *packed]),
        "many": (container, [driver, "load-many", *packed]),
    }
    if private is not None:
        alone = ["--container", str(private), *threads]
        methods["private_decode"] = (private, [driver, "load-decode", *alone])
        methods["private_many"] = (private, [driver, "load-many", *alone])
    return methods


def ensure_private(settings: Settings, entry: Entry, store: Store) -> Path:
    """Return a one-member container of `entry` in `max`, the private layout risk R6 compares with."""
    private = settings.packs / "private-fragmented.ztz"
    if settings.force or "private_container" not in store.data or not private.exists():
        require_free(settings.out, entry.size, "load")
        command = [str(settings.cli), "encode", str(entry.path), str(private), "--mode", "max"]
        measured = run_measured(command, settings.logs, "encode-private")
        store.record(
            "private_container",
            {"seconds": measured.seconds, "file_bytes": private.stat().st_size},
        )
    return private


def step_load(settings: Settings) -> None:
    """Time the load of each target member, cold, `settings.runs` times per method, and print them.

    Raises BenchError as `measure` does, and if a target is unset or the memory for `load-many` is
    missing; GateError on a failed gate of a strict run.
    """
    require_free(settings.out, 0, "load")
    entries = load_manifest(settings)
    container = require_pack(settings)
    seek_map = ensure_seek_map(settings, container)
    targets = settings.targets
    cases = {
        "bf16": targets.bf16,
        "shared_f32": targets.shared_f32,
        "q8": targets.q8,
        "fragmented": targets.fragmented,
    }
    store = Store(settings, "load")
    # Refused now and not after an hour of the cheaper methods. `measure` runs see this checked again.
    sizes = [source_of(entries, member).size for member in cases.values() if member is not None]
    require_memory(max(sizes, default=0), "load")
    fragments = max(seek_map["members"], key=lambda member: member["runs"])
    if fragments["name"] != targets.fragmented:
        log.warning("%s has the most disk runs, not %s", fragments["name"], targets.fragmented)
    for case, member in cases.items():
        if member is None:
            raise BenchError(f"load needs a member for {case}")
        entry = source_of(entries, member)
        plan = settings.state / f"plan-{case}.txt"
        if settings.force or not plan.exists():
            command = [
                str(settings.driver),
                "tensors",
                "--container",
                str(container),
                "--member",
                member,
                "--out",
                str(plan),
            ]
            run_measured(command, settings.logs, f"tensors-{case}")
        private = ensure_private(settings, entry, store) if case == "fragmented" else None
        methods = load_commands(settings, case, entry, plan, private)
        results = store.data.setdefault("cases", {}).setdefault(case, {})
        store.data.setdefault("members", {})[case] = {"name": member, "len": entry.size}
        for method, (evicted, command) in methods.items():
            # Filtered once: a run disturbed in this invocation stays, so a busy machine cannot loop.
            runs = [] if settings.force else [run for run in results.get(method, []) if recorded(run)]
            results[method] = runs
            while len(runs) < settings.runs:
                if method.endswith("many"):
                    require_memory(entry.size, f"{case} {method}")
                name = f"load-{case}-{method}-{len(runs) + 1}"
                runs.append(measure(settings, name, evicted, command))
                store.save()
    print(render_load(store.data, seek_map))
    enforce(settings, load_gates(store.data["cases"]), "load")


def random_side(
    settings: Settings, entry: Entry, *, raw: bool, size: int, passes: int, cold: str
) -> tuple[Path, list[str]]:
    """Return what one side of a random pattern evicts and its driver command."""
    container = settings.packs / f"{FULL_KEY}.ztz"
    command = [
        str(settings.driver),
        "random",
        "--container",
        str(container),
        "--member",
        entry.member,
        "--threads",
        str(settings.threads),
        "--size",
        str(size),
        "--count",
        str(settings.random_count),
        "--seed",
        str(settings.random_seed),
        "--passes",
        str(passes),
        "--cold",
        cold,
    ]
    if raw:
        return entry.path, [*command, "--raw", str(entry.path)]
    return container, command


def step_random(settings: Settings) -> None:
    """Risk R2: cold random reads, then the same spans for three passes, raw against the container.

    `cold` drops the page cache before every read, as 10k reads of 1 MiB cover most of a member's cache
    otherwise. `repeat` evicts once, so the later passes show what the page cache gives each side.

    Raises as `step_load` does.
    """
    require_free(settings.out, 0, "random")
    member = settings.targets.random
    if member is None:
        raise BenchError("random needs --member")
    entries = load_manifest(settings)
    require_pack(settings)
    entry = source_of(entries, member)
    store = Store(settings, "random")
    store.data["member"] = {"name": member, "len": entry.size}
    results = store.data.setdefault("patterns", {})
    for size in settings.random_sizes:
        for pattern, passes, cold in (("cold", 1, "each"), ("repeat", 3, "first")):
            sides = results.setdefault(f"{pattern}-{size}", {})
            for side in ("raw", "packed"):
                if side in sides and recorded(sides[side]) and not settings.force:
                    continue
                evicted, command = random_side(
                    settings, entry, raw=side == "raw", size=size, passes=passes, cold=cold
                )
                name = f"random-{pattern}-{size}-{side}"
                sides[side] = measure(settings, name, evicted, command)
                store.save()
    print(render_random(store.data))
    enforce(settings, random_gates(results), "R2")


def step_encode_memory(settings: Settings) -> None:
    """Risk R3: the Rust peak of writing one member in `max`, against 6 x threads x chunk plus the index.

    The pack itself is measured by the CLI, whose peak is process RSS: zstd's C contexts and allocator
    retention included, which the bound excludes. So the driver writes a one-member container of the
    target through `PackBuilder` with its counting allocator, into the packs directory, and this step
    deletes it afterwards, whether the driver succeeded or not. Memory only: no idle wait, no eviction.

    Raises BenchError if the target is unset or unknown, for too little space, a failing driver, and in
    a strict run for a debug driver; GateError on a failed gate of a strict run.
    """
    member = settings.targets.encode
    if member is None:
        raise BenchError("encode-memory needs --member")
    entries = load_manifest(settings)
    entry = source_of(entries, member)
    store = Store(settings, "encode_memory")
    done = store.data.get("member", {}).get("name") == member and "run" in store.data
    if settings.force or not done:
        output = settings.packs / ENCODE_MEMORY_OUTPUT
        # What a killed run left, before the space check counts it as used.
        delete_inside(output, settings.out)
        require_free(settings.out, entry.size, "encode-memory")
        command = [
            str(settings.driver),
            "encode-memory",
            "--input",
            str(entry.path),
            "--output",
            str(output),
            "--mode",
            ENCODE_MEMORY_MODE,
            "--threads",
            str(settings.threads),
        ]
        try:
            measured = run_measured(command, settings.logs, "encode-memory")
        finally:
            delete_inside(output, settings.out)
        document: dict[str, Any] = json.loads(measured.stdout)
        if settings.strict and not document["release"]:
            raise BenchError(f"{settings.driver} is a debug build, its allocations are not the release ones")
        store.record("member", {"name": member, "len": entry.size})
        store.record("run", document)
    print(render_encode_memory(store.data))
    enforce(settings, [encode_memory_gate(store.data["run"])], "R3")


def clean_targets(settings: Settings) -> list[Path]:
    """Return what `clean` removes: restored copies, and every file of the packs directory.

    The packs directory is listed file by file, so that the temporary file of a killed pack is shown.
    """
    targets = [settings.restored] if settings.restored.exists() else []
    if settings.packs.exists():
        targets += sorted(settings.packs.iterdir())
    return targets


def step_clean(settings: Settings, confirmed: bool) -> list[tuple[Path, int]]:
    """List the removable outputs, and delete them if `confirmed`. Results and logs stay.

    Returns each path with its bytes. Raises BenchError for a path outside the output directory.
    """
    listed = [(path, tree_bytes(path)) for path in clean_targets(settings)]
    if confirmed:
        for path, _ in listed:
            delete_inside(path, settings.out)
    return listed
