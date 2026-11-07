#!/usr/bin/env python3
"""File: zt_bench.py
Purpose: command line of the ziptensors acceptance benchmark, plan part 5.

Main responsibilities: parse arguments, check the layout of the run, dispatch one step, and `dry-run`,
which takes every step through a generated tree in the system temp directory.

Key functions: `main`, `dry_run`, `settings_from`.

Implementation notes: roots, output directory and binaries are required arguments and no path is in
this tree. Each step is a separate invocation and resumes from what its predecessor recorded, because
a full run takes hours. Exit codes: 0 success, 1 failure, 2 usage, 3 a gate failed with results stored.
See MODULE_README.md for the order of the steps and what each costs.
"""

import sys

# Keeps the source tree free of `__pycache__`, here and for the imports below.
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import logging  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from bench_guard import BenchError, GateError, check_layout  # noqa: E402
from bench_report import write_document  # noqa: E402
from bench_run import Settings, Store, Targets  # noqa: E402
from bench_steps import (  # noqa: E402
    ENCODE_MEMORY_OUTPUT,
    FULL_KEY,
    load_manifest,
    pick_subset,
    step_clean,
    step_encode_memory,
    step_load,
    step_pack,
    step_random,
    step_savings,
    step_snapshot,
    step_verify_exact,
)
from bench_tree import build_tree  # noqa: E402

log = logging.getLogger("zt_bench")

EXPECTED_FILES = 546
# dev-docs/benchmarks/v0.1.md, addressed from this file.
DEFAULT_REPORT = Path(__file__).resolve().parents[2] / "dev-docs" / "benchmarks" / "v0.1.md"
EXIT_FAILURE = 1
EXIT_GATE = 3
DRY_RANDOM_COUNT = 20
DRY_IDLE_WINDOW = 0.1


def build_parser() -> argparse.ArgumentParser:
    """Return the parser: common arguments on every step, step arguments on their own."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, action="append", required=True, help="source root, given twice")
    common.add_argument("--out", type=Path, required=True, help="output directory, outside both roots")
    common.add_argument("--cli", type=Path, required=True, help="release binary of ziptensors")
    common.add_argument("--driver", type=Path, required=True, help="release binary of zt-bench-driver")
    common.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="result document")
    common.add_argument("--expected-files", type=int, default=EXPECTED_FILES)
    common.add_argument("--threads", type=int, default=0, help="0 is one per logical CPU")
    common.add_argument("--force", action="store_true", help="measure again what is recorded")
    common.add_argument("--idle-rate", type=int, default=2 * 1024 * 1024, help="bytes per second counted as idle")
    common.add_argument("--idle-timeout", type=float, default=600.0, help="seconds to wait for an idle disk")

    parser = argparse.ArgumentParser(description="ziptensors acceptance benchmark")
    steps = parser.add_subparsers(dest="step", required=True)

    snapshot = steps.add_parser("snapshot", parents=[common], help="record or compare both roots")
    snapshot.add_argument("--phase", choices=("before", "after"), required=True)
    steps.add_parser("pack", parents=[common], help="full set in max, subset in all modes")
    steps.add_parser("verify-exact", parents=[common], help="restore all, sha256 against the sources")
    steps.add_parser("savings", parents=[common], help="inspect --json against the estimate table")
    load = steps.add_parser("load", parents=[common], help="cold load of each target member")
    load.add_argument("--bf16-member", required=True)
    load.add_argument("--shared-f32-member", required=True)
    load.add_argument("--q8-member", required=True)
    load.add_argument("--fragmented-member", required=True)
    load.add_argument("--runs", type=int, default=3)
    random = steps.add_parser("random", parents=[common], help="random reads on one member, R2")
    random.add_argument("--member", required=True)
    random.add_argument("--count", type=int, default=10_000)
    encode = steps.add_parser("encode-memory", parents=[common], help="Rust peak of writing one member, R3")
    encode.add_argument("--member", required=True)
    steps.add_parser("report", parents=[common], help="write the result document")
    clean = steps.add_parser("clean", parents=[common], help="list or delete restored copies and packs")
    clean.add_argument("--yes", action="store_true", help="delete, instead of listing")

    dry = steps.add_parser("dry-run", help="every step on a generated tree in the temp directory")
    dry.add_argument("--cli", type=Path, required=True)
    dry.add_argument("--driver", type=Path, required=True)
    dry.add_argument("--keep", action="store_true", help="leave the temp directory for inspection")
    return parser


def settings_from(arguments: argparse.Namespace) -> Settings:
    """Return the settings of a step.

    Raises BenchError unless exactly two roots are given, and for a bad output directory or binary.
    """
    if len(arguments.root) != 2:
        raise BenchError(f"--root must be given twice, got {len(arguments.root)}")
    roots = (arguments.root[0], arguments.root[1])
    check_layout(roots, arguments.out)
    # The report is the one file written outside the output directory.
    report_parent = arguments.report.parent
    for root in roots:
        if report_parent.resolve().is_relative_to(root.resolve()):
            raise BenchError(f"--report {arguments.report} is inside source root {root}")
    for binary in (arguments.cli, arguments.driver):
        if not binary.is_file():
            raise BenchError(f"{binary} does not exist: build it first, see MODULE_README.md")
    # `--member` is the target of whichever of the two steps took it.
    member = getattr(arguments, "member", None)
    targets = Targets(
        bf16=getattr(arguments, "bf16_member", None),
        shared_f32=getattr(arguments, "shared_f32_member", None),
        q8=getattr(arguments, "q8_member", None),
        fragmented=getattr(arguments, "fragmented_member", None),
        random=member if arguments.step == "random" else None,
        encode=member if arguments.step == "encode-memory" else None,
    )
    return Settings(
        roots=roots,
        out=arguments.out,
        cli=arguments.cli,
        driver=arguments.driver,
        report=arguments.report,
        expected_files=arguments.expected_files,
        threads=arguments.threads,
        force=arguments.force,
        runs=getattr(arguments, "runs", 3),
        random_count=getattr(arguments, "count", 10_000),
        idle_rate=arguments.idle_rate,
        idle_timeout=arguments.idle_timeout,
        targets=targets,
    )


def print_clean(listed: list[tuple[Path, int]], deleted: bool) -> None:
    """Print what `clean` found. This is the result of the step, so stdout."""
    verb = "deleted" if deleted else "would delete"
    for path, size in listed:
        print(f"{verb} {path} ({size} bytes)")
    print(f"{verb} {sum(size for _, size in listed)} bytes in {len(listed)} entries")


def run_step(arguments: argparse.Namespace) -> None:
    """Run the step named by `arguments`."""
    settings = settings_from(arguments)
    settings.prepare()
    match arguments.step:
        case "snapshot":
            step_snapshot(settings, arguments.phase)
        case "pack":
            step_pack(settings)
        case "verify-exact":
            step_verify_exact(settings)
        case "savings":
            step_savings(settings)
        case "load":
            step_load(settings)
        case "random":
            step_random(settings)
        case "encode-memory":
            step_encode_memory(settings)
        case "report":
            write_document(settings)
        case "clean":
            print_clean(step_clean(settings, arguments.yes), arguments.yes)
        case other:
            raise BenchError(f"unknown step {other}")


def dry_run(cli: Path, driver: Path, keep: bool) -> None:
    """Take every step through a generated tree, and check what can be checked without real data.

    Timing and savings thresholds are reported and do not stop it. Bit-exactness, the snapshot, the
    resume and the deletes are checked for real.

    Raises BenchError on anything that would stop a real run.
    """
    base = Path(tempfile.mkdtemp(prefix="zt-bench-"))
    try:
        tree = build_tree(base)
        out = base / "out"
        settings = Settings(
            roots=(tree.root_a, tree.root_b),
            out=out,
            cli=cli,
            driver=driver,
            report=out / "v0.1.md",
            expected_files=tree.files,
            strict=False,
            random_count=DRY_RANDOM_COUNT,
            idle_window=DRY_IDLE_WINDOW,
            idle_timeout=0.0,
            targets=Targets(
                tree.bf16, tree.shared_f32, tree.q8, tree.fragmented, tree.fragmented, tree.bf16
            ),
        )
        check_layout(settings.roots, out)
        settings.prepare()
        step_snapshot(settings, "before")
        step_pack(settings)
        check_resume(settings)
        step_verify_exact(settings)
        if settings.restored.exists():
            raise BenchError("verify-exact left the restored copies behind")
        step_savings(settings)
        step_load(settings)
        step_random(settings)
        step_encode_memory(settings)
        if (settings.packs / ENCODE_MEMORY_OUTPUT).exists():
            raise BenchError("encode-memory left its container behind")
        step_snapshot(settings, "after")
        write_document(settings)
        check_document(settings, base)
        print_clean(step_clean(settings, confirmed=False), deleted=False)
        if not any(settings.packs.iterdir()):
            raise BenchError("clean without --yes deleted something")
        print_clean(step_clean(settings, confirmed=True), deleted=True)
        if any(settings.packs.iterdir()):
            raise BenchError("clean --yes left something behind")
        print("dry run passed")
    finally:
        if keep:
            log.info("kept %s", base)
        else:
            shutil.rmtree(base)


def check_resume(settings: Settings) -> None:
    """Prove a recorded pack is not repeated: a second `pack` must leave the container alone.

    Raises BenchError if it did not.
    """
    full = settings.packs / f"{FULL_KEY}.ztz"
    before = full.stat().st_mtime_ns
    step_pack(settings)
    if full.stat().st_mtime_ns != before:
        raise BenchError("a second pack repeated recorded work")
    subset = {item["member"] for item in Store(settings, "pack").data["subset"]}
    if subset != {entry.member for entry in pick_subset(load_manifest(settings))}:
        raise BenchError("the subset is not reproducible")


def check_document(settings: Settings, base: Path) -> None:
    """Prove the result document has every section measured and no machine path.

    Raises BenchError otherwise.
    """
    text = settings.report.read_text(encoding="utf-8")
    if "Not run." in text:
        raise BenchError(f"the report has a section not run:\n{text}")
    if str(base) in text:
        raise BenchError("the report holds a machine path")


def main() -> int:
    """Run one step and return the exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    arguments = build_parser().parse_args()
    try:
        if arguments.step == "dry-run":
            dry_run(arguments.cli, arguments.driver, arguments.keep)
        else:
            run_step(arguments)
    except GateError as error:
        log.error("gate failed, results are stored: %s", error)
        return EXIT_GATE
    except BenchError as error:
        log.error("%s", error)
        return EXIT_FAILURE
    return 0


if __name__ == "__main__":
    sys.exit(main())
