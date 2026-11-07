"""File: bench_report.py
Purpose: Markdown tables of the results: printed by the steps, and joined into the result document,
measured next to estimate.

Key functions: `render_savings`, `render_load`, `render_random`, `render_encode_memory`,
`render_document`.

Implementation notes: reads only what the steps stored and never writes outside the path it is given,
so it works after `clean` and with the subset packs gone: no container is opened. Records from before
the counting allocator or the disk guard show `-` in those columns and `not measured` in the memory
gates. The document is tracked, so it holds member names relative to a root and no path of this machine.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

from bench_estimates import (
    BF16_COMPUTED_SECONDS,
    BF16_RAW_ESTIMATE_SECONDS,
    GB,
    MODE_ESTIMATES_PERCENT,
    PACK_ESTIMATE_HOURS,
    TOTAL_ESTIMATE_PERCENT,
    READ_FIXED_SLACK_BYTES,
    WRITE_FIXED_SLACK_BYTES,
    Gate,
    class_rows,
    disturbed,
    load_gates,
    memory_gates,
    random_gates,
    savings_gates,
)
from bench_guard import FOREIGN_FLOOR_BYTES
from bench_run import Settings

MIB = 1 << 20
FRAGMENTED_SHOWN = 5


def table(header: list[str], rows: list[list[str]]) -> str:
    """Return a Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def gb(value: float) -> str:
    """`value` bytes in GB, the plan's 10^9."""
    return f"{value / GB:.2f}"


def render_savings(document: dict[str, Any]) -> str:
    """Return the savings table of plan part 5 with measured values next to the estimates."""
    rows = []
    for row in class_rows(document["classes"]):
        estimate = row.estimate
        rows.append(
            [
                estimate.label,
                f"{estimate.raw_gb:.1f}",
                gb(row.raw),
                f"{estimate.dedup_widen_gb:.2f}",
                gb(row.dedup_saved + row.widen_saved),
                f"{estimate.entropy_gb:.1f}",
                gb(row.entropy_saved),
                f"{estimate.saved_percent:.1f}",
                f"{row.saved_percent:.1f}",
                f"{row.delta_pp:+.1f}",
                row.status,
            ]
        )
    raw = document["raw_bytes"]
    layers = document["layers"]
    saved = document["saved_bytes"] / raw * 100.0 if raw else 0.0
    rows.append(
        [
            "**Total**",
            "361",
            gb(raw),
            "47.3",
            gb(layers["dedup_saved"] + layers["widen_saved"]),
            "61.9",
            gb(layers["entropy_saved"]),
            f"{TOTAL_ESTIMATE_PERCENT}",
            f"{saved:.2f}",
            f"{saved - TOTAL_ESTIMATE_PERCENT:+.1f}",
            "",
        ]
    )
    header = [
        "Class",
        "Raw GB est",
        "Raw GB",
        "Dedup+widen GB est",
        "Dedup+widen GB",
        "Entropy GB est",
        "Entropy GB",
        "Saved % est",
        "Saved %",
        "Delta pp",
        "Status",
    ]
    return table(header, rows)


def mib(value: int | None) -> str:
    """`value` bytes in MiB with one decimal, `-` when not measured."""
    return "-" if value is None else f"{value / MIB:.1f}"


def rust_peak(runs: list[dict[str, Any]]) -> int | None:
    """The highest Rust peak of `runs`, None if a run has no Rust count."""
    if not runs or any("rust_peak_bytes" not in run for run in runs):
        return None
    return max(int(run["rust_peak_bytes"]) for run in runs)


def disk_state(runs: list[dict[str, Any]]) -> str:
    """`disturbed` if the guard marked a run, `quiet` if it recorded all and marked none, else
    `unchecked`, for runs from before the guard."""
    if disturbed(*runs):
        return "disturbed"
    return "quiet" if all("disk" in run for run in runs) else "unchecked"


def worst_read_run(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """The run with the highest Rust peak, the one a memory gate must judge. Runs without a count rank
    first, so that a mix shows as not measured."""
    return max(runs, key=lambda run: run.get("rust_peak_bytes", float("inf")))


def render_load(load: dict[str, Any], seek_map: dict[str, Any]) -> str:
    """Return the load times per case and method, and the seek proxy of the target members."""
    rows = []
    for case, methods in load["cases"].items():
        for method, runs in methods.items():
            if not runs:
                continue
            times = [run["seconds"] for run in runs]
            above = max(run["peak_rss_bytes"] - run["destination_bytes"] for run in runs)
            rate = runs[0]["bytes"] / MIB / median(times) if median(times) else 0.0
            rows.append(
                [
                    case,
                    method,
                    " / ".join(f"{value:.1f}" for value in times),
                    f"{median(times):.1f}",
                    f"{rate:.0f}",
                    mib(rust_peak(runs)),
                    f"{above / MIB:.0f}",
                    disk_state(runs),
                ]
            )
    header = [
        "Case",
        "Method",
        "Runs s",
        "Median s",
        "MiB/s",
        "Rust peak above index and destination MiB",
        "Peak RSS above destination MiB",
        "Disk",
    ]
    text = table(header, rows)
    by_name = {member["name"]: member for member in seek_map["members"]}
    seek_rows = []
    for case, member in load["members"].items():
        layout = by_name[member["name"]]
        seek_rows.append(
            [
                case,
                member["name"],
                gb(member["len"]),
                str(layout["extents"]),
                str(layout["runs"]),
                gb(layout["jump_bytes"]),
            ]
        )
    seek_header = ["Case", "Member", "GB", "Extents", "Disk runs", "Jumps GB"]
    return f"{text}\n\nSeek proxy of the targets:\n\n{table(seek_header, seek_rows)}"


def render_random(random: dict[str, Any]) -> str:
    """Return the latency table of risk R2, raw next to the container."""
    rows = []
    for key, sides in sorted(random["patterns"].items()):
        pairs = zip(sides["raw"]["passes"], sides["packed"]["passes"], strict=True)
        for number, (raw, packed) in enumerate(pairs, start=1):
            packed_run = sides["packed"]
            above = packed_run["peak_rss_bytes"] - packed_run["destination_bytes"]
            rows.append(
                [
                    key,
                    str(number),
                    f"{raw['p50_ms']:.2f}",
                    f"{packed['p50_ms']:.2f}",
                    f"{raw['p95_ms']:.2f}",
                    f"{packed['p95_ms']:.2f}",
                    f"{raw['p99_ms']:.2f}",
                    f"{packed['p99_ms']:.2f}",
                    mib(packed_run.get("rust_peak_bytes")),
                    f"{above / MIB:.0f}",
                    f"{disk_state([sides['raw']])} / {disk_state([packed_run])}",
                ]
            )
    header = [
        "Pattern",
        "Pass",
        "Raw p50 ms",
        "Packed p50 ms",
        "Raw p95",
        "Packed p95",
        "Raw p99",
        "Packed p99",
        "Packed Rust peak above index and destination MiB",
        "Packed peak RSS above destination MiB",
        "Disk raw / packed",
    ]
    return table(header, rows)


def render_encode_memory(encode: dict[str, Any]) -> str:
    """Return the Rust peaks of the `encode-memory` write, next to its RSS, which is not gated."""
    run = encode["run"]
    member = encode["member"]
    header = [
        "Member",
        "GB",
        "Mode",
        "Threads",
        "Chunk",
        "Ingest Rust peak MiB",
        "Commit Rust peak MiB",
        "Index MiB",
        "Peak RSS MiB",
        "Seconds",
    ]
    row = [
        f"`{member['name']}`",
        gb(member["len"]),
        run["mode"],
        str(run["threads"]),
        str(run["chunk_bytes"]),
        mib(run["ingest_peak_bytes"]),
        mib(run["finish_peak_bytes"]),
        mib(run["index_bytes"]),
        mib(run["peak_rss_bytes"]),
        f"{run['seconds']:.0f}",
    ]
    return table(header, [row])


def render_pack(pack: dict[str, Any]) -> str:
    """Return the encode table, with the mode estimates of the whole set for reference."""
    rows = []
    for key, result in pack.items():
        if key == "subset":
            continue
        mode = key.rsplit("-", 1)[-1]
        rows.append(
            [
                key,
                str(result["files"]),
                gb(result["raw_bytes"]),
                gb(result["file_bytes"]),
                f"{result['saved_percent']:.2f}",
                f"{MODE_ESTIMATES_PERCENT[mode]:.0f}",
                f"{result['seconds'] / 3600:.2f}",
                f"{result['peak_rss_bytes'] / MIB:.0f}",
            ]
        )
    header = ["Pack", "Files", "Raw GB", "File GB", "Saved %", "Whole set est %", "Hours", "Peak RSS MiB"]
    low, high = PACK_ESTIMATE_HOURS
    note = f"Estimated wall time of the full set: {low} to {high} h."
    subset = "\n".join(f"- `{item['member']}`, {gb(item['size'])} GB" for item in pack.get("subset", []))
    return f"{table(header, rows)}\n\n{note}\n\nSubset:\n\n{subset}"


def render_gates(gates: list[Gate]) -> str:
    """Return one table of gates."""
    shown = {"pass": "pass", "FAIL": "**FAIL**"}
    rows = [[g.name, g.measured, g.target, shown.get(g.result, f"*{g.result}*")] for g in gates]
    return table(["Gate", "Measured", "Target", "Result"], rows)


def read_state(settings: Settings, name: str) -> dict[str, Any]:
    """Return the stored results of a step, empty if it has not run."""
    path = settings.state / f"{name}.json"
    if not path.exists():
        return {}
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def stamp(path: Path) -> str:
    """Return the modification time of a binary, which tells which build was measured."""
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def render_document(settings: Settings) -> str:
    """Return the result document from whatever the steps stored so far."""
    pack = read_state(settings, "pack")
    verify = read_state(settings, "verify")
    savings = read_state(settings, "savings")
    load = read_state(settings, "load")
    random = read_state(settings, "random")
    seek_map = read_state(settings, "seek_map")
    snapshot = read_state(settings, "snapshot")
    encode = read_state(settings, "encode_memory")
    missing = "Not run."
    gates: list[Gate] = []
    parts = [
        "# ziptensors v0.1 acceptance benchmark",
        f"Measured with `tools/bench/`. CLI built {stamp(settings.cli)}, driver built "
        f"{stamp(settings.driver)}. GB is 10^9 bytes. {len(os.sched_getaffinity(0))} logical CPUs.",
        "## Source roots",
        f"Unchanged: {snapshot['identical']}." if snapshot else missing,
        "## Encode",
        render_pack(pack) if pack else missing,
        "## Bit-exact",
        f"{verify['matched']} of {verify['total']} files match by sha256sum."
        if "matched" in verify
        else missing,
        "## Savings by layer",
    ]
    if savings:
        document = savings["inspect"]
        raw = document["raw_bytes"]
        saved_percent = document["saved_bytes"] / raw * 100.0 if raw else 0.0
        gates += savings_gates(saved_percent, class_rows(document["classes"]))
        parts.append(render_savings(document))
    else:
        parts.append(missing)
    parts.append("## Load from HDD, cold, all tensors")
    read_runs: list[tuple[str, dict[str, Any]]] = []
    if load:
        gates += load_gates(load["cases"])
        parts.append(
            f"BF16 estimates: raw {BF16_RAW_ESTIMATE_SECONDS:.0f} s, computed "
            f"{BF16_COMPUTED_SECONDS:.0f} s. `raw` reads the tensor ranges from the source file into "
            "one buffer sized for the largest tensor; `decode` calls `decode_tensor_into` in tensor "
            "order into the same; `many` calls `read_many` once for all tensors into one buffer of "
            "the total; `private_*` read a one-member container of the same file."
        )
        parts.append(render_load(load, seek_map))
        worst = sorted(seek_map["members"], key=lambda member: member["runs"], reverse=True)
        rows = [
            [member["name"], gb(member["len"]), str(member["extents"]), str(member["runs"])]
            for member in worst[:FRAGMENTED_SHOWN]
        ]
        parts.append(f"Most disk runs:\n\n{table(['Member', 'GB', 'Extents', 'Disk runs'], rows)}")
        for case, methods in load["cases"].items():
            read_runs += [
                (f"{case} {method}", worst_read_run(runs))
                for method, runs in methods.items()
                if method != "raw" and runs
            ]
    else:
        parts.append(missing)
    parts.append("## Random access (R2)")
    if random:
        gates += random_gates(random["patterns"])
        member = random["member"]
        parts.append(f"Member `{member['name']}`, {gb(member['len'])} GB.")
        parts.append(render_random(random))
        read_runs += [
            (f"{key} packed", sides["packed"]) for key, sides in sorted(random["patterns"].items())
        ]
    else:
        parts.append(missing)
    parts.append("## Memory (R3 and the read contract)")
    parts.append(
        "Gates count Rust allocations with the driver's counting allocator, the unit of the bounds in "
        "`ARCHITECTURE.md`. Reads: the peak of the measured phase above a baseline taken after the "
        f"open and after the destination is allocated, at most 3 x threads x chunk + "
        f"{READ_FIXED_SLACK_BYTES // 1024} KiB of fixed bookkeeping. Writes: one member written in "
        "`max` through `PackBuilder` by `encode-memory`, at most 6 x threads x chunk + the in-memory "
        f"index of that write + {WRITE_FIXED_SLACK_BYTES // 1024} KiB. Chunk is the largest "
        "`chunk_size` of the container. Peak RSS, which also holds code, zstd's C contexts and "
        "allocator retention, is reported and not gated, the pack's included."
    )
    parts.append(render_encode_memory(encode) if encode else missing)
    gates += memory_gates(encode.get("run"), read_runs)
    parts.append("## Gates")
    parts.append(
        "A timing gate over a run the disk guard marked disturbed shows `disturbed`, neither pass nor "
        f"fail: foreign disk traffic on the measured devices above {FOREIGN_FLOOR_BYTES // MIB} MiB + "
        "the idle rate x the run's seconds, or an md array not idle at its end. `not measured` is a "
        "record from a driver without the counting allocator."
    )
    parts.append(render_gates(gates) if gates else missing)
    return "\n\n".join(parts) + "\n"


def write_document(settings: Settings) -> None:
    """Write the result document to `settings.report`. Raises OSError if that fails."""
    settings.report.parent.mkdir(parents=True, exist_ok=True)
    settings.report.write_text(render_document(settings), encoding="utf-8")
