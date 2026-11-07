# Module: tools/bench

## Purpose

Acceptance benchmark of ziptensors v0.1, plan part 5: packs a real model set, proves the restored bytes
with an independent tool, compares savings by layer with the estimates, and measures loading and random
reads from disk. Python 3.12+ on the standard library, plus one small Rust driver. Linux only, because of
`posix_fadvise`, `fincore`, `/proc`. Not on any runtime path.

## Architecture

`zt_bench.py` runs one step per invocation. Every step records its results under `<out>/state/` as it
goes and skips what is recorded, so a killed run is repeated with the same command. `--force` measures
again. A timed run of `load` or `random` counts as recorded only if it carries the disk record of the
guard and is not disturbed, so the same command without `--force` also repeats runs from an older driver
or a busy disk, and keeps plans, seek map and private container. The CLI binary does the packing and
restoring, `driver/` does every timed read and the Rust-allocation measurements, and each measurement is
a process of its own so that caches are evicted before it and memory is read after it.

`<out>` holds `state/` (manifest, snapshots, results), `logs/` (stdout and stderr of each child), `packs/`
and `restored/`.

## Files and submodules

| Path | Role |
|---|---|
| `zt_bench.py` | Arguments, dispatch, exit codes, `dry-run`. |
| `bench_steps.py` | One function per step. Manifest, subset choice, sha256 pass. |
| `bench_run.py` | `Settings`, the results `Store`, `run_measured`. |
| `bench_guard.py` | Every safety check, the single delete, idle disk and md arrays, eviction, the disk record of a run. |
| `bench_estimates.py` | Plan tables and gates. Pure. |
| `bench_report.py` | Tables and the result document. |
| `bench_tree.py` | The generated tree of the dry run. |
| `test_zt_bench.py` | Tests. |
| `driver/` | `zt-bench-driver`, one crate with its own `[workspace]`, outside the root workspace like `fuzz/`. See `driver/src/MODULE_README.md`. |

## Building

```bash
cargo build --release -p ziptensors-cli                                  # target/release/ziptensors
cargo build --release --manifest-path tools/bench/driver/Cargo.toml \
    --target-dir <outside the tree>                                      # <dir>/release/zt-bench-driver
```

A debug driver is refused outside `dry-run`. The Python module is not used: it has no `read_many`.

## Running

`C` stands for `--root <A> --root <B> --out <OUT> --cli <ziptensors> --driver <zt-bench-driver>`. The roots
of this machine are in `dev-docs/local-notes.md`.

| Step | Writes | Time | Gate |
|---|---|---|---|
| `snapshot --phase before C` | little | seconds | file count is `--expected-files`, 546 |
| `pack C` | full set and 3 subset packs | 2-3 h | |
| `verify-exact C` | one restored copy of the set, then deleted | 2-3 h | every sha256 matches |
| `savings C` | little | seconds | total at least 27.0%, each class within 2 pp |
| `load C --bf16-member M --shared-f32-member M --q8-member M --fragmented-member M` | one private container | 1 h | part 5 step 4, R6 |
| `random C --member M` | little | 30 min | R2 |
| `encode-memory C --member M` | one container of the member, deleted after | 10 min | R3 |
| `snapshot --phase after C` | little | seconds | identical to before |
| `report C` | `dev-docs/benchmarks/v0.1.md` | seconds | |
| `clean C [--yes]` | deletes `packs/`, `restored/` | seconds | |

`report` writes what is recorded so far and says `Not run.` for the rest. Run it last. Exit codes: 0, 1
failure, 2 usage, 3 a gate failed, with the results stored.

`dry-run --cli <any build> --driver <any build>` takes every step through a tree generated in the temp
directory and removes it. Timing and savings gates are computed and do not stop it; idle disk and
eviction only warn, as `/tmp` may be a memory filesystem. Bit-exactness, snapshots, resume and deletes are
checked for real.

Tests, with binaries for the end-to-end case, which otherwise skips:

```bash
ZT_BENCH_CLI=<ziptensors> ZT_BENCH_DRIVER=<zt-bench-driver> PYTHONDONTWRITEBYTECODE=1 \
    pytest -p no:cacheprovider tools/bench
```

## Contracts and invariants

- Source roots and the output directory are required arguments. No path appears in tracked code. A target
  is a member name argument; the subset is chosen by rule from sizes and suffixes.
- Roots are read-only. Nothing opens a source file for writing. `evict` runs `fsync` and `posix_fadvise`
  on a read-only descriptor, which changes only the page cache. The output directory and `--report` must
  lie outside both roots, compared resolved.
- `bench_guard.delete_inside` is the only delete. It proves the resolved parent lies inside the output
  directory, and `clean` lists unless `--yes`. `clean` keeps `state/` and `logs/`. `encode-memory` deletes
  its container through it too, also after a failed driver.
- Each step computes the disk it needs from its inputs and refuses to start under it. A pack is sized at
  its worst case, the raw size. `load` refuses below the memory of `load-many`: the largest target plus 2 GiB.
- Timing waits for an idle disk (`--idle-rate`, over 5 s) after `sync` and for every md array's
  `sync_action` to be `idle`, both within `--idle-timeout`, then evicts and confirms with `fincore`. Outside
  a dry run a busy disk, a busy array or a resident file stops the step. `pack` waits the same way.
- Every timed run of `load` and `random` stores `disk`: bytes read and written by the measured devices
  (`/proc/diskstats`) over the whole run, next to the bytes the driver caused (`wait4` `ru_inblock`,
  `ru_oublock`). It is `disturbed` when the difference exceeds 64 MiB + `--idle-rate` × the run's seconds,
  the idle criterion kept on average, or when an md array is not idle at its end: a check reads the member
  disks, not the md device, so its traffic is not in the device counters. Its timing gate then shows
  `disturbed`, not pass or fail, and a strict step exits 3.
- The manifest is the selection of `pack`: regular files, no link, no `*.incomplete`. A name `pack` would
  refuse fails the first snapshot, and `pack` is told both roots, so member names are relative to them.
- A repeated `snapshot --phase before` is refused without `--force`, as it would hide a change.
- Report text holds member names and no machine path, because the document is tracked.
- Memory gates count Rust allocations through the driver's counting allocator, the unit the bounds of
  `ARCHITECTURE.md` are stated in. Reads: the peak above a baseline taken after the open and the
  destination allocation, at most 3 × threads × chunk + 128 KiB. Writes (R3): `encode-memory` writes one
  member in `max` through `PackBuilder`, at most 6 × threads × chunk + the in-memory index of that write
  + 256 KiB. The slack is that of the allocator tests of `ziptensors-core`. Chunk is the largest
  `chunk_size` of the container and threads the count in force, both reported by the driver.
- RSS is reported and never gated: the driver's own `VmHWM`, `wait4` for the CLI, which cannot read below
  the RSS of this script. It holds code, zstd's C contexts and allocator retention, which no bound covers.
- `inspect --json` `schema_version` other than 1 stops `savings`.
- `report` reads only `state/`: it works after `clean` and without the subset packs.
- Windows and macOS are not covered, and risk R5 (dedup map size) is seen only inside the pack's RSS,
  which is not gated: `encode-memory` writes one member, whose dedup map is small.

## Measurement decisions

- `raw` reads the tensor ranges, from the file, into one buffer sized for the largest tensor, and `decode`
  fills the same buffer, so both cover the same bytes. `many` needs a buffer of the whole member.
- `random` `cold` drops the cache before each read, outside the timed part, because 10k reads of 1 MiB would
  otherwise cache a large share of the member. `repeat` evicts once and replays the same offsets 3 times.
- R6 compares with a one-member container of the same file, encoded in `max`.
- "Within 2 pp" is two-sided. A class far over its target fails too, marked `over`.

## Editing map

- New measurement: a step in `bench_steps.py`, a gate in `bench_estimates.py`, a section in `bench_report.py`,
  a parser in `zt_bench.py`, a line in `dry_run`, and the table above.
- Disk guard, md arrays, the disturbed threshold: `bench_guard.py`; how a disturbed run counts for resume
  and gates: `recorded` and `disturbed` in `bench_estimates.py`.
- New timed read: a subcommand of `driver/src/main.rs`.
- New estimate or threshold: `bench_estimates.py`.
