"""File: bench_estimates.py
Purpose: the numbers of the acceptance benchmark: the estimates of plan part 5 and risks R2, R3, R6, and
the gates that compare measurements with them. Pure functions, no I/O.

Main responsibilities:
- the class table of plan part 5, and the map from `inspect --json` class names to its rows;
- `savings_gates`, `load_gates`, `random_gates`, `memory_gates`, each giving one `Gate` per criterion.

Key structures: `ClassEstimate`, `ClassRow`, `Gate`.

Key functions: `class_rows`, `savings_gates`, `load_gates`, `random_gates`, `memory_gates`,
`read_memory_gate`, `encode_memory_gate`, `disturbed`, `recorded`.

Implementation notes: GB is 10^9 bytes, as in the plan. "Within 2 pp" is two-sided, as the plan says: a
class far above its target passes no gate either, since it means the estimate behind it is wrong. Such
a row is labelled `over` so that a reader can tell it from a shortfall.

Memory gates compare Rust allocations, counted by the driver, with the bounds of `ARCHITECTURE.md`, which
are stated in that unit. Process RSS also holds code, zstd's C contexts and allocator retention, none of
which the bounds cover, so it is reported and never gated. A timing gate over a run marked `disturbed`
by the disk guard is shown as such: it neither passes nor fails. Memory gates ignore that mark, since
disk traffic does not move Rust allocations.
"""

from dataclasses import dataclass
from statistics import median
from typing import Any

GB = 10**9
# Total saved of `max`, in percent.
TOTAL_GATE_PERCENT = 27.0
TOTAL_ESTIMATE_PERCENT = 30.2
CLASS_TOLERANCE_PP = 2.0
# Saved percent of the whole set for the other modes. A subset is compared with these only by eye.
MODE_ESTIMATES_PERCENT = {"fast": 19.0, "balanced": 21.0, "max": 30.2}
PACK_ESTIMATE_HOURS = (1.5, 2.5)

BF16_RAW_ESTIMATE_SECONDS = 104.0
BF16_TARGET_SECONDS = 80.0
BF16_COMPUTED_SECONDS = 70.0
SHARED_F32_TARGET_RATIO = 0.5
Q8_TARGET_RATIO = 1.05
# Risk R6: slower than a one-member container of the same file by this factor falsifies it.
FRAGMENTED_TARGET_RATIO = 1.15
# Risk R2: p50 over raw plus this falsifies it.
RANDOM_P50_MARGIN_MS = 8.0
# Risk R3: encode peak over this many threads times chunks, plus the index.
ENCODE_BOUND_FACTOR = 6
# Contract of reads in ARCHITECTURE.md.
READ_BOUND_FACTOR = 3
# The fixed bookkeeping both bounds allow, the slack of the allocator tests of ziptensors-core:
# `tests/read_memory.rs` for reads, `tests/write_memory.rs` for writes. It does not grow with the
# chunk size, the thread count or the data.
READ_FIXED_SLACK_BYTES = 128 * 1024
WRITE_FIXED_SLACK_BYTES = 256 * 1024
DISTURBED = "disturbed"
NOT_MEASURED = "not measured"


@dataclass(frozen=True)
class ClassEstimate:
    """One row of the class table of plan part 5, in GB."""

    label: str
    dtypes: tuple[str, ...]
    raw_gb: float
    dedup_widen_gb: float
    entropy_gb: float
    gated: bool = True

    @property
    def saved_percent(self) -> float:
        """The target: what the row expects to save of its raw bytes."""
        return (self.dedup_widen_gb + self.entropy_gb) / self.raw_gb * 100.0


# `dtypes` are the class names of `inspect --json`, the `Debug` form of `Dtype`. "Other" takes everything
# unlisted, skeleton included; the plan does not estimate it, so it has no gate.
OTHER_LABEL = "Other"
CLASS_ESTIMATES = (
    ClassEstimate("BF16", ("Bf16",), 155.9, 12.61, 48.3),
    ClassEstimate("F16", ("F16",), 23.5, 0.0, 5.3),
    ClassEstimate("F32", ("F32",), 20.5, 13.53 + 1.05 + 0.92, 1.0),
    ClassEstimate("FP8", ("F8E4M3", "F8E5M2"), 13.8, 0.0, 2.5),
    ClassEstimate("Q8_0", ("Q80",), 52.9, 2.48, 2.9),
    ClassEstimate("Q4_K, Q5_K, Q6_K", ("Q4K", "Q5K", "Q6K"), 57.4, 0.0, 1.9),
    ClassEstimate("U8 packed", ("U8", "Nibble"), 35.3, 16.62, 0.0),
    ClassEstimate(OTHER_LABEL, (), 1.8, 0.0, 0.0, gated=False),
)


@dataclass(frozen=True)
class Gate:
    """One acceptance criterion and its outcome.

    `invalid` names why the measurement cannot judge the criterion, `DISTURBED` or `NOT_MEASURED`;
    such a gate neither passes nor fails, whatever `passed` says.
    """

    name: str
    measured: str
    target: str
    passed: bool
    invalid: str | None = None

    @property
    def failed(self) -> bool:
        """True for a judged gate that did not pass."""
        return self.invalid is None and not self.passed

    @property
    def result(self) -> str:
        """`pass`, `FAIL`, or the reason it was not judged."""
        if self.invalid is not None:
            return self.invalid
        return "pass" if self.passed else "FAIL"


def disturbed(*runs: dict[str, Any]) -> bool:
    """True if the disk guard marked one of `runs` disturbed. A run from before the guard has no mark."""
    return any(run.get("disk", {}).get("disturbed", False) for run in runs)


def recorded(run: dict[str, Any]) -> bool:
    """True if `run` counts as done for a resumed step: it carries the disk record of the guard, and is
    not disturbed. A run from before the guard or a disturbed one is measured again without `--force`."""
    return "disk" in run and not disturbed(run)


def _invalid_if_disturbed(*runs: dict[str, Any]) -> str | None:
    """`DISTURBED` if one of `runs` is, else None."""
    return DISTURBED if disturbed(*runs) else None


@dataclass(frozen=True)
class ClassRow:
    """Measured bytes of one row of the class table."""

    estimate: ClassEstimate
    raw: int
    widen_saved: int
    dedup_saved: int
    entropy_saved: int
    stored: int

    @property
    def saved_percent(self) -> float:
        """Saved share of this row's raw bytes, 0 for a row with no bytes."""
        return (self.raw - self.stored) / self.raw * 100.0 if self.raw else 0.0

    @property
    def delta_pp(self) -> float:
        """Measured minus target, in percentage points."""
        return self.saved_percent - self.estimate.saved_percent

    @property
    def status(self) -> str:
        """`ok`, `short`, `over`, or `not gated`."""
        if not self.estimate.gated:
            return "not gated"
        if self.delta_pp < -CLASS_TOLERANCE_PP:
            return "short"
        return "over" if self.delta_pp > CLASS_TOLERANCE_PP else "ok"


def class_rows(classes: list[dict[str, Any]]) -> list[ClassRow]:
    """Fold the `classes` of `inspect --json` into the rows of the plan's table, in its order."""
    labels = {dtype: row.label for row in CLASS_ESTIMATES for dtype in row.dtypes}
    totals = {row.label: [0, 0, 0, 0, 0] for row in CLASS_ESTIMATES}
    for entry in classes:
        total = totals[labels.get(entry["class"], OTHER_LABEL)]
        fields = ("raw", "widen_saved", "dedup_saved", "entropy_saved", "stored")
        for position, field in enumerate(fields):
            total[position] += entry[field]
    return [ClassRow(row, *totals[row.label]) for row in CLASS_ESTIMATES]


def savings_gates(saved_percent: float, rows: list[ClassRow]) -> list[Gate]:
    """The gate of plan part 5: the total, and each estimated class within the tolerance."""
    gates = [
        Gate(
            "total saved",
            f"{saved_percent:.2f}%",
            f">= {TOTAL_GATE_PERCENT}% (estimate {TOTAL_ESTIMATE_PERCENT}%)",
            saved_percent >= TOTAL_GATE_PERCENT,
        )
    ]
    for row in rows:
        if row.estimate.gated:
            gates.append(
                Gate(
                    f"class {row.estimate.label}",
                    f"{row.saved_percent:.2f}% ({row.status})",
                    f"{row.estimate.saved_percent:.2f}% +- {CLASS_TOLERANCE_PP} pp",
                    row.status == "ok",
                )
            )
    return gates


def median_seconds(runs: list[dict[str, Any]]) -> float:
    """Return the median `seconds` of a list of measurements."""
    return float(median(run["seconds"] for run in runs))


def _ratio_gate(
    name: str, measured: list[dict[str, Any]], baseline: list[dict[str, Any]], limit: float, strict: bool
) -> Gate:
    """A gate on the ratio of the median seconds of two lists of runs, below `limit`, or at most `limit`
    unless `strict`. Disturbed if a run of either list is."""
    measured_seconds, baseline_seconds = median_seconds(measured), median_seconds(baseline)
    ratio = measured_seconds / baseline_seconds if baseline_seconds > 0 else float("inf")
    passed = ratio < limit if strict else ratio <= limit
    relation = "<" if strict else "<="
    return Gate(
        name,
        f"{measured_seconds:.1f} s vs {baseline_seconds:.1f} s, ratio {ratio:.3f}",
        f"ratio {relation} {limit}",
        passed,
        _invalid_if_disturbed(*measured, *baseline),
    )


def load_gates(load: dict[str, dict[str, list[dict[str, Any]]]]) -> list[Gate]:
    """The targets of plan part 5 step 4 and risk R6, on medians, for the cases present in `load`.

    `load` maps a case name to its methods, each a list of runs. The read_many variants of R6 compare
    like with like: pack `many` against private `many`.
    """
    gates = []
    if "bf16" in load:
        decode = load["bf16"]["decode"]
        seconds = median_seconds(decode)
        gates.append(
            Gate(
                "BF16 decode",
                f"{seconds:.1f} s",
                f"<= {BF16_TARGET_SECONDS:.0f} s (raw estimate {BF16_RAW_ESTIMATE_SECONDS:.0f} s, "
                f"computed {BF16_COMPUTED_SECONDS:.0f} s)",
                seconds <= BF16_TARGET_SECONDS,
                _invalid_if_disturbed(*decode),
            )
        )
    if "shared_f32" in load:
        methods = load["shared_f32"]
        gates.append(
            _ratio_gate(
                "shared F32 decode vs raw",
                methods["decode"],
                methods["raw"],
                SHARED_F32_TARGET_RATIO,
                strict=True,
            )
        )
    if "q8" in load:
        methods = load["q8"]
        gates.append(
            _ratio_gate(
                "Q8_0 decode vs raw",
                methods["decode"],
                methods["raw"],
                Q8_TARGET_RATIO,
                strict=False,
            )
        )
    if "fragmented" in load:
        methods = load["fragmented"]
        for method, private in (("decode", "private_decode"), ("many", "private_many")):
            gates.append(
                _ratio_gate(
                    f"R6 pack {method} vs private layout",
                    methods[method],
                    methods[private],
                    FRAGMENTED_TARGET_RATIO,
                    strict=False,
                )
            )
    return gates


def random_gates(results: dict[str, Any]) -> list[Gate]:
    """Risk R2: for each size and pass, p50 of the container at most raw p50 plus the margin.

    `results` maps `<pattern>-<size>` to `{"raw": run, "packed": run}`.
    """
    gates = []
    for key, sides in sorted(results.items()):
        pairs = zip(sides["raw"]["passes"], sides["packed"]["passes"], strict=True)
        for number, (raw, packed) in enumerate(pairs, start=1):
            limit = raw["p50_ms"] + RANDOM_P50_MARGIN_MS
            gates.append(
                Gate(
                    f"R2 {key} pass {number}",
                    f"p50 {packed['p50_ms']:.2f} ms",
                    f"<= {limit:.2f} ms (raw {raw['p50_ms']:.2f} + {RANDOM_P50_MARGIN_MS:.0f})",
                    packed["p50_ms"] <= limit,
                    _invalid_if_disturbed(sides["raw"], sides["packed"]),
                )
            )
    return gates


def read_memory_gate(label: str, run: dict[str, Any]) -> Gate:
    """The read contract on one candidate run: Rust bytes above the index and the destination at most
    `READ_BOUND_FACTOR` x threads x chunk, plus the fixed slack.

    `run` is a driver record with `rust_peak_bytes`, `threads` and `chunk_bytes`, the largest chunk of
    the container read. A record of a driver without the counting allocator is `NOT_MEASURED`.
    """
    name = f"memory {label}"
    if "rust_peak_bytes" not in run:
        return Gate(name, "no Rust count", "rerun with the current driver", False, NOT_MEASURED)
    limit = READ_BOUND_FACTOR * run["threads"] * run["chunk_bytes"] + READ_FIXED_SLACK_BYTES
    peak = run["rust_peak_bytes"]
    return Gate(
        name,
        f"{peak} B Rust above index and destination",
        f"<= {limit} B ({READ_BOUND_FACTOR} x {run['threads']} x {run['chunk_bytes']} + {READ_FIXED_SLACK_BYTES})",
        peak <= limit,
    )


def encode_memory_gate(run: dict[str, Any]) -> Gate:
    """Risk R3 on the `encode-memory` record of the driver: the Rust peak of the write at most
    `ENCODE_BOUND_FACTOR` x threads x chunk plus the in-memory index of that write, plus the fixed slack.
    """
    index = run["index_bytes"]
    limit = ENCODE_BOUND_FACTOR * run["threads"] * run["chunk_bytes"] + index + WRITE_FIXED_SLACK_BYTES
    peak = run["rust_peak_bytes"]
    return Gate(
        "R3 encode Rust peak",
        f"{peak} B",
        f"<= {limit} B ({ENCODE_BOUND_FACTOR} x {run['threads']} x {run['chunk_bytes']} + index {index} "
        f"+ {WRITE_FIXED_SLACK_BYTES})",
        peak <= limit,
    )


def memory_gates(encode: dict[str, Any] | None, read_runs: list[tuple[str, dict[str, Any]]]) -> list[Gate]:
    """Risk R3 on the `encode-memory` record if there is one, and the read contract on each labelled
    candidate run."""
    gates = [encode_memory_gate(encode)] if encode is not None else []
    gates += [read_memory_gate(label, run) for label, run in read_runs]
    return gates
