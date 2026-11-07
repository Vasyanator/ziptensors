"""File: test_zt_bench.py
Purpose: tests of the acceptance benchmark: path confinement, gates, parsers, and the dry run end to end.

Key functions: one `test_*` per contract.

The md guard and the disk record run on a fake `/sys` and `/proc/diskstats` under `tmp_path`, never on
the machine's.

Implementation notes: run with `PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tools/bench`, so that
the tree stays clean. The end-to-end test needs binaries a fresh clone does not have, so it skips unless
`ZT_BENCH_CLI` and `ZT_BENCH_DRIVER` name them. All scratch is pytest's `tmp_path`, in the temp directory.
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from bench_estimates import (
    CLASS_ESTIMATES,
    DISTURBED,
    NOT_MEASURED,
    READ_FIXED_SLACK_BYTES,
    WRITE_FIXED_SLACK_BYTES,
    ClassRow,
    Gate,
    class_rows,
    encode_memory_gate,
    load_gates,
    memory_gates,
    random_gates,
    read_memory_gate,
    recorded,
    savings_gates,
)
from bench_guard import (
    FOREIGN_FLOOR_BYTES,
    BenchError,
    DiskWatch,
    GateError,
    busy_arrays,
    check_layout,
    confined,
    delete_inside,
    disk_bytes,
    disk_record,
    disturbance,
    require_free,
    require_memory,
    required_space,
    wait_idle,
    wait_md_idle,
)
from bench_report import render_document
from bench_run import Settings, Targets, run_measured
from bench_steps import (
    Entry,
    build_manifest,
    enforce,
    parse_sha256,
    pick_subset,
    read_hash_log,
    snapshot_difference,
    step_clean,
    take_snapshot,
)


def make_settings(tmp_path: Path) -> Settings:
    """Settings over empty roots under `tmp_path`."""
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    return Settings(
        roots=(root_a, root_b),
        out=tmp_path / "out",
        cli=Path("cli"),
        driver=Path("driver"),
        report=tmp_path / "report.md",
        expected_files=0,
        strict=False,
        targets=Targets(),
    )


def test_output_inside_a_root_is_refused_even_through_a_link(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "deep").mkdir(parents=True)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(BenchError, match="inside source root"):
        check_layout((root, other), root / "deep" / "out")
    with pytest.raises(BenchError, match="inside source root"):
        check_layout((root, other), root)
    link = tmp_path / "link"
    link.symlink_to(root)
    with pytest.raises(BenchError, match="inside source root"):
        check_layout((other, root), link / "out")
    with pytest.raises(BenchError, match="inside output directory"):
        check_layout((root, other), tmp_path)
    check_layout((root, other), tmp_path / "out")


def test_a_missing_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BenchError, match="not a directory"):
        check_layout((tmp_path / "absent",), tmp_path / "out")


def test_delete_stays_inside_the_output_directory(tmp_path: Path) -> None:
    out = tmp_path / "out"
    (out / "tree").mkdir(parents=True)
    keep = tmp_path / "keep"
    keep.mkdir()
    (keep / "file").write_bytes(b"x")

    for path in (keep / "file", out, tmp_path, out / ".." / "keep" / "file", out / ".."):
        with pytest.raises(BenchError):
            delete_inside(path, out)
    assert (keep / "file").exists()

    (out / "escape").symlink_to(keep)
    with pytest.raises(BenchError, match="not inside"):
        confined(out / "escape" / "file", out)
    assert (keep / "file").exists()

    # A link to the outside is removed as a link, and the tree keeps what it points at.
    (out / "tree" / "link").symlink_to(keep)
    delete_inside(out / "tree", out)
    delete_inside(out / "escape", out)
    assert not (out / "tree").exists() and not (out / "escape").is_symlink()
    assert (keep / "file").exists()


def test_clean_lists_without_yes_and_deletes_with_it(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.prepare()
    (settings.packs / "full.ztz").write_bytes(b"12345")
    settings.restored.mkdir()
    (settings.restored / "copy").write_bytes(b"12")
    (settings.state / "pack.json").write_text("{}")

    listed = step_clean(settings, confirmed=False)
    assert sorted(size for _, size in listed) == [2, 5]
    assert (settings.packs / "full.ztz").exists() and settings.restored.exists()

    step_clean(settings, confirmed=True)
    assert not (settings.packs / "full.ztz").exists() and not settings.restored.exists()
    assert (settings.state / "pack.json").exists()


def test_a_step_refuses_more_than_the_disk_holds(tmp_path: Path) -> None:
    with pytest.raises(BenchError, match="bytes free"):
        require_free(tmp_path, 1 << 62, "pack")
    require_free(tmp_path, 0, "pack")
    assert required_space(1000) > 1000
    with pytest.raises(BenchError, match="of memory"):
        require_memory(1 << 62, "load")


def test_a_path_on_no_block_device_only_stops_a_strict_run() -> None:
    proc = Path("/proc/self/status")
    with pytest.raises(BenchError, match="no block device"):
        wait_idle([proc], 1, 0.01, 0.0, strict=True)
    assert wait_idle([proc], 1, 0.01, 0.0, strict=False) == 0.0


def test_snapshots_see_every_kind_of_change(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "file").write_bytes(b"one")
    (root / "name with space\nand newline").write_bytes(b"two")
    (root / "link").symlink_to("file")
    first, second = tmp_path / "first", tmp_path / "second"
    assert take_snapshot(root, first) == 4
    take_snapshot(root, second)
    rows = first.read_bytes().split(b"\0")
    assert snapshot_difference(rows, second.read_bytes().split(b"\0")) == []

    (root / "file").write_bytes(b"three")
    (root / "new").write_bytes(b"")
    take_snapshot(root, second)
    difference = snapshot_difference(rows, second.read_bytes().split(b"\0"))
    assert sum(line.startswith("+") for line in difference) >= 2
    assert sum(line.startswith("-") for line in difference) >= 1


def test_manifest_takes_what_pack_takes(tmp_path: Path) -> None:
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "m.safetensors").write_bytes(b"123")
    (root_a / "p.safetensors.incomplete").write_bytes(b"1")
    (root_a / "link").symlink_to("m.safetensors")
    (root_b / "sub").mkdir()
    (root_b / "sub" / "т.bin").write_bytes(b"1")
    rows = []
    for root in (root_a, root_b):
        take_snapshot(root, tmp_path / "snapshot")
        rows.append((root, (tmp_path / "snapshot").read_bytes().split(b"\0")))

    records = build_manifest(rows)

    assert [record["member"] for record in records] == ["m.safetensors", "sub/т.bin"]
    assert records[0]["size"] == 3


def test_manifest_refuses_what_pack_would_refuse(tmp_path: Path) -> None:
    def row(name: bytes) -> bytes:
        return b"f 1 1.0 /r/" + name

    with pytest.raises(BenchError, match="not UTF-8"):
        build_manifest([(Path("/r"), [row(b"\xff")])])
    with pytest.raises(BenchError, match="refuses the member name"):
        build_manifest([(Path("/r"), [row(b"a:b")])])
    with pytest.raises(BenchError, match="both roots"):
        build_manifest([(Path("/r"), [row(b"a")]), (Path("/r"), [row(b"a")])])


def test_subset_is_reproducible_spread_and_has_distinct_names() -> None:
    entries = [
        Entry(f"d{n}/model.safetensors", Path(f"/r/d{n}/model.safetensors"), (n + 1) << 30)
        for n in range(4)
    ]
    entries += [Entry(f"s{n}.safetensors", Path(f"/r/s{n}.safetensors"), n << 30) for n in range(1, 8)]
    entries += [Entry(f"g{n}.gguf", Path(f"/r/g{n}.gguf"), (n + 1) << 30) for n in range(5)]
    entries += [Entry(f"c{n}.json", Path(f"/r/c{n}.json"), 100) for n in range(5)]

    chosen = pick_subset(entries)

    assert chosen == pick_subset(list(reversed(entries)))
    assert len(chosen) == 12
    assert len({entry.path.name for entry in chosen}) == 12
    assert {entry.path.suffix for entry in chosen} == {".safetensors", ".gguf", ".json"}
    with pytest.raises(BenchError, match="distinct names"):
        pick_subset(entries[:4])


def test_sha256_lines_parse_with_and_without_the_escape_mark() -> None:
    digest = "a" * 64
    assert parse_sha256(f"{digest}  name\n", Path("x")) == digest
    assert parse_sha256(f"\\{digest}  na\\nme\n", Path("x")) == digest
    with pytest.raises(BenchError):
        parse_sha256("sha256sum: oops", Path("x"))


def test_a_hash_log_cut_by_a_kill_keeps_the_finished_lines(tmp_path: Path) -> None:
    path = tmp_path / "log"
    path.write_text('{"member": "a", "source": "1", "restored": "1"}\n{"member": "b", "sou', "utf-8")
    assert list(read_hash_log(path)) == ["a"]
    assert read_hash_log(tmp_path / "absent") == {}


def row_at(saved_percent: float, label: str = "BF16") -> ClassRow:
    """A class row that saved `saved_percent` of 1000 bytes."""
    estimate = next(item for item in CLASS_ESTIMATES if item.label == label)
    return ClassRow(estimate, 1000, 0, 0, 0, round(1000 - saved_percent * 10))


def test_estimates_add_up_to_the_plan_total() -> None:
    raw = sum(row.raw_gb for row in CLASS_ESTIMATES)
    saved = sum(row.dedup_widen_gb + row.entropy_gb for row in CLASS_ESTIMATES)
    assert raw == pytest.approx(361.1)
    assert saved / raw * 100.0 == pytest.approx(30.2, abs=0.05)


def test_savings_gate_is_the_total_and_two_points_either_way() -> None:
    target = CLASS_ESTIMATES[0].saved_percent
    assert all(gate.passed for gate in savings_gates(27.0, [row_at(target)]))
    assert not savings_gates(26.99, [])[0].passed
    assert row_at(target - 2.1).status == "short"
    assert row_at(target + 2.1).status == "over"
    assert not savings_gates(30.0, [row_at(target - 2.1)])[1].passed
    assert not savings_gates(30.0, [row_at(target + 2.1)])[1].passed
    assert len(savings_gates(30.0, [row_at(50.0, "Other")])) == 1


def test_classes_fold_into_the_rows_of_the_plan() -> None:
    def entry(name: str, raw: int) -> dict[str, object]:
        fields = ("widen_saved", "dedup_saved", "entropy_saved", "overhead", "stored")
        return {"class": name, "raw": raw, **dict.fromkeys(fields, 0)}

    names = ("Q4K", "Q5K", "skeleton", "Unknown(7)", "I8", "Nibble", "F8E5M2", "Bf16")
    rows = {row.estimate.label: row.raw for row in class_rows([entry(name, 1) for name in names])}

    assert rows["Q4_K, Q5_K, Q6_K"] == 2
    assert rows["Other"] == 3
    assert rows["U8 packed"] == rows["FP8"] == rows["BF16"] == 1
    assert rows["F16"] == 0


def runs(*seconds: float) -> list[dict[str, float]]:
    """Measurements of these durations."""
    return [{"seconds": value} for value in seconds]


def test_load_gates_use_medians_and_the_targets_of_the_plan() -> None:
    gates = load_gates(
        {
            "bf16": {"decode": runs(79.0, 80.0, 500.0)},
            "shared_f32": {"decode": runs(49.0), "raw": runs(100.0)},
            "q8": {"decode": runs(106.0), "raw": runs(100.0)},
            "fragmented": {
                "decode": runs(115.0),
                "many": runs(116.0),
                "private_decode": runs(100.0),
                "private_many": runs(100.0),
            },
        }
    )
    assert [gate.passed for gate in gates] == [True, True, False, True, False]
    exact = load_gates({"shared_f32": {"decode": runs(50.0), "raw": runs(100.0)}})
    assert not exact[0].passed, "the plan says below half, not at half"


def test_random_gate_is_raw_plus_eight_milliseconds_per_pass() -> None:
    def side(*p50: float) -> dict[str, object]:
        return {"passes": [{"p50_ms": value} for value in p50]}

    gates = random_gates({"repeat-4096": {"raw": side(10.0, 0.1), "packed": side(18.0, 8.2)}})
    assert [gate.passed for gate in gates] == [True, False]


def test_read_memory_gate_counts_rust_bytes_against_the_chunk_of_the_run() -> None:
    limit = 3 * 2 * 1000 + READ_FIXED_SLACK_BYTES
    run = {"threads": 2, "chunk_bytes": 1000, "rust_peak_bytes": limit, "peak_rss_bytes": 1 << 40}
    assert read_memory_gate("a", run).result == "pass", "RSS is reported, never gated"
    assert read_memory_gate("a", {**run, "rust_peak_bytes": limit + 1}).result == "FAIL"
    old = {"threads": 2, "peak_rss_bytes": 10, "destination_bytes": 0}
    gate = read_memory_gate("a", old)
    assert gate.invalid == NOT_MEASURED and not gate.failed
    assert memory_gates(None, [("a", run)])[0].passed


def test_encode_memory_gate_adds_the_index_of_the_write() -> None:
    run = {"threads": 2, "chunk_bytes": 1000, "index_bytes": 500}
    limit = 6 * 2 * 1000 + 500 + WRITE_FIXED_SLACK_BYTES
    assert encode_memory_gate({**run, "rust_peak_bytes": limit}).passed
    assert not encode_memory_gate({**run, "rust_peak_bytes": limit + 1}).passed
    gates = memory_gates({**run, "rust_peak_bytes": 0}, [])
    assert [gate.name for gate in gates] == ["R3 encode Rust peak"]


def quiet(**fields: object) -> dict[str, object]:
    """A run the disk guard recorded as quiet."""
    return {**fields, "disk": {"disturbed": False}}


def noisy(**fields: object) -> dict[str, object]:
    """A run the disk guard marked disturbed."""
    return {**fields, "disk": {"disturbed": True}}


def test_a_disturbed_run_makes_its_timing_gate_neither_pass_nor_fail() -> None:
    gates = load_gates(
        {
            "bf16": {"decode": [quiet(seconds=500.0), noisy(seconds=500.0), quiet(seconds=500.0)]},
            "q8": {"decode": [quiet(seconds=1.0)], "raw": [noisy(seconds=100.0)]},
            "shared_f32": {"decode": [quiet(seconds=1.0)], "raw": [quiet(seconds=100.0)]},
        }
    )
    assert [gate.result for gate in gates] == [DISTURBED, "pass", DISTURBED]
    assert not any(gate.failed for gate in gates)

    def side(run: dict[str, object]) -> dict[str, object]:
        return {**run, "passes": [{"p50_ms": 100.0}]}

    patterns = {"cold-4096": {"raw": side(quiet()), "packed": side(noisy())}}
    assert random_gates(patterns)[0].result == DISTURBED


def test_only_a_quiet_guarded_run_counts_as_recorded() -> None:
    assert recorded(quiet(seconds=1.0))
    assert not recorded(noisy(seconds=1.0))
    assert not recorded({"seconds": 1.0}), "a run from before the guard is measured again"


def test_a_strict_step_stops_on_a_gate_it_cannot_judge(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    disturbed_gate = Gate("g", "1", "2", True, DISTURBED)
    enforce(settings, [disturbed_gate], "load")
    with pytest.raises(GateError, match="not judged"):
        enforce(replace(settings, strict=True), [disturbed_gate], "load")
    enforce(replace(settings, strict=True), [Gate("g", "1", "2", True)], "load")
    with pytest.raises(GateError, match="failed gates: g"):
        enforce(replace(settings, strict=True), [Gate("g", "1", "2", False)], "load")


def fake_md(sys_root: Path, **arrays: str | None) -> None:
    """Lay out `/sys/block/<name>/md/sync_action` for each array; None leaves the md directory out."""
    for name, action in arrays.items():
        directory = sys_root / "block" / name
        directory.mkdir(parents=True, exist_ok=True)
        if action is not None:
            (directory / "md").mkdir(exist_ok=True)
            (directory / "md" / "sync_action").write_text(f"{action}\n", encoding="ascii")


def test_md_guard_waits_for_every_array_and_refuses_a_busy_one(tmp_path: Path) -> None:
    sys_root = tmp_path / "sys"
    fake_md(sys_root, md0="idle", md127="check", md5=None)
    (sys_root / "block" / "sda").mkdir()
    assert busy_arrays(sys_root) == {"md127": "check"}
    past = time.monotonic()
    with pytest.raises(BenchError, match="md127"):
        wait_md_idle(past, True, sys_root, poll_seconds=0.0)
    assert wait_md_idle(past, False, sys_root, poll_seconds=0.0) == {"md127": "check"}
    # A path on no block device skips the disk window, not the md guard.
    proc = Path("/proc/self/status")
    assert wait_idle([proc], 1, 0.01, 0.0, strict=False, sys_root=sys_root) == 0.0

    fake_md(sys_root, md127="idle")
    assert wait_md_idle(past, True, sys_root, poll_seconds=0.0) == {}
    assert busy_arrays(tmp_path / "no-sys") == {}


def diskstats_line(name: str, sectors_read: int, sectors_written: int) -> str:
    """One line of /proc/diskstats with only the counters the guard reads set."""
    fields = ["9", "127", name, "0", "0", str(sectors_read), "0", "0", "0", str(sectors_written)]
    return " ".join(fields + ["0"] * 8)


def test_foreign_disk_traffic_over_the_threshold_marks_a_run_disturbed() -> None:
    idle = 1000
    limit = FOREIGN_FLOOR_BYTES + 10 * idle
    calm = disturbance(limit + 500, 0, 500, 0, 10.0, idle, {})
    assert calm["foreign_bytes"] == limit and not calm["disturbed"]
    loud = disturbance(limit, 501, 0, 500, 10.0, idle, {})
    assert loud["disturbed"] and "foreign" in loud["reasons"][0]
    assert disturbance(0, 0, 10, 10, 1.0, idle, {})["foreign_bytes"] == 0
    checking = disturbance(0, 0, 0, 0, 1.0, idle, {"md127": "check"})
    assert checking["disturbed"] and "md127" in checking["reasons"][0]


def test_the_disk_record_spans_the_run_from_diskstats(tmp_path: Path) -> None:
    stats = tmp_path / "diskstats"
    stats.write_text(diskstats_line("md127", 100, 10) + "\n" + diskstats_line("sda", 5, 5) + "\n")
    watch = DiskWatch(frozenset({"md127"}), disk_bytes({"md127"}, stats))
    sys_root = tmp_path / "sys"
    fake_md(sys_root, md127="idle")
    own_read = 1 << 20
    sectors = (own_read + FOREIGN_FLOOR_BYTES) // 512
    stats.write_text(diskstats_line("md127", 100 + sectors, 10) + "\n" + diskstats_line("sda", 9, 9) + "\n")

    record = disk_record(watch, own_read, 0, 1.0, 0, sys_root, stats)
    assert record["devices"] == ["md127"]
    assert record["device_read_bytes"] == sectors * 512
    assert record["foreign_bytes"] == FOREIGN_FLOOR_BYTES and not record["disturbed"]

    stats.write_text(diskstats_line("md127", 100 + sectors + 1, 10) + "\n")
    assert disk_record(watch, own_read, 0, 1.0, 0, sys_root, stats)["disturbed"]
    with pytest.raises(BenchError, match="left"):
        disk_bytes({"sdz"}, stats)


def test_report_needs_only_the_state_files(tmp_path: Path) -> None:
    settings = replace(
        make_settings(tmp_path), cli=Path(sys.executable), driver=Path(sys.executable)
    )
    settings.prepare()
    subset = {"files": 12, "raw_bytes": 100, "file_bytes": 80, "saved_percent": 20.0}
    pack = {
        "subset": [{"member": "m", "size": 100}],
        "full-max": {**subset, "seconds": 3600.0, "peak_rss_bytes": 1 << 27, "threads": 2},
        **{
            f"subset-{mode}": {**subset, "seconds": 60.0, "peak_rss_bytes": 1 << 26, "threads": 2}
            for mode in ("fast", "balanced", "max")
        },
    }
    encode = {
        "member": {"name": "m", "len": 100},
        "run": {
            "mode": "max",
            "threads": 2,
            "chunk_bytes": 1 << 20,
            "seconds": 1.0,
            "ingest_peak_bytes": 1 << 20,
            "finish_peak_bytes": 1 << 20,
            "rust_peak_bytes": 1 << 20,
            "index_bytes": 1000,
            "peak_rss_bytes": 1 << 26,
        },
    }
    (settings.state / "pack.json").write_text(json.dumps(pack))
    (settings.state / "encode_memory.json").write_text(json.dumps(encode))
    assert not any(settings.packs.iterdir()), "no container, the subset packs are gone"

    text = render_document(settings)

    assert "subset-max" in text
    assert "| R3 encode Rust peak |" in text and "**FAIL**" not in text
    assert "R3 pack peak RSS" not in text, "the pack's RSS is not gated"


def test_a_measured_child_reports_its_own_failure_and_peak(tmp_path: Path) -> None:
    measured = run_measured([sys.executable, "-c", "print('hello')"], tmp_path, "good")
    assert measured.stdout == "hello\n"
    assert measured.peak_rss_bytes > 0

    code = "import sys; sys.stderr.write('the reason'); sys.exit(4)"
    with pytest.raises(BenchError, match="exit code 4(?s:.*)the reason"):
        run_measured([sys.executable, "-c", code], tmp_path, "bad")


def test_dry_run_takes_every_step_through_a_generated_tree(tmp_path: Path) -> None:
    cli, driver = os.environ.get("ZT_BENCH_CLI"), os.environ.get("ZT_BENCH_DRIVER")
    if not cli or not driver:
        pytest.skip("ZT_BENCH_CLI and ZT_BENCH_DRIVER are not set")
    script = Path(__file__).with_name("zt_bench.py")
    done = subprocess.run(
        [sys.executable, str(script), "dry-run", "--cli", cli, "--driver", driver],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "TMPDIR": str(tmp_path)},
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "dry run passed" in done.stdout
    assert list(tmp_path.iterdir()) == [], "the dry run removes its temp directory"
