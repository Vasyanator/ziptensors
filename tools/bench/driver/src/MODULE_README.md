# Module: tools/bench/driver/src

## Purpose

`zt-bench-driver`: the measuring half of the acceptance benchmark. One measurement per process, one JSON
document on stdout. Exists because the Python module has no `read_many` and hides the chunks. It holds no
format logic, only public API of `ziptensors-core`.

## Files and submodules

| Path | Role |
|---|---|
| `main.rs` | Subcommands, the load and random loops, the seek proxy, `encode-memory`. |
| `counting.rs` | The counting global allocator and `Phase`, the peak of a measured phase above its baseline. |

## Contracts and invariants

- Every timed document carries `release`, `seconds` or per-pass latencies, `destination_bytes`,
  `peak_rss_bytes`, which is `VmHWM`, and the Rust figures `rust_index_bytes` (live bytes the open left),
  `rust_peak_bytes` (peak of the measured phase above a baseline taken after the open and the destination
  allocation) and `chunk_bytes` (largest `chunk_size` of the container, 0 for raw). `encode-memory` prints
  `ingest_peak_bytes`, `finish_peak_bytes`, `rust_peak_bytes`, `index_bytes` (Rust peak of opening the
  written file with one thread), `chunk_bytes`, `threads`, `mode`. `tools/bench/bench_steps.py`,
  `bench_estimates.py` and `bench_report.py` read these names.
- Every Rust allocation goes through `counting::Counting`; zstd's C contexts do not, as the bounds exclude
  them. Phases do not nest: starting one restarts the peak.
- The chunk size is read from the index after the timed part, so it cannot warm the cache of the run.
- `encode-memory` leaves its output; the Python side deletes it through `bench_guard.delete_inside`.
- `load-*` time starts before the container is opened, since a loader pays that.
- `seek-map` counts disk runs from the index: where a sequential read of the member must seek. It is the
  seek proxy of risk R6.
- The same `--seed`, `--count`, `--size` and member length give the same offsets, which is what makes the
  raw and packed sides of `random` comparable.
- Outside the root workspace, with its own lockfile, so the mandatory targets do not build it. Verify it
  with `cargo clippy --all-targets -- -D warnings` and `cargo test` from `tools/bench/driver`.
- The `zstd` pin and the release profile repeat the root manifest. Change them together.

## Editing map

- New measurement: a `Command` variant and a function, printing one JSON object.
- What a memory figure counts: where its `Phase` starts, in the function of the subcommand.
