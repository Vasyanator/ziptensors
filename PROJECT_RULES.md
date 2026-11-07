# ziptensors: project working rules

Specializes `~/.claude/CLAUDE.md`. Only what differs or what a global rule needs filled in.

## Toolchain and targets

- Rust 1.99.1, edition 2024, resolver 3.
- Mandatory check targets: `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-gnu` (needs
  `x86_64-w64-mingw32-gcc`), and `x86_64-apple-darwin` through `cargo-zigbuild`. macOS is checked, not
  linked; linking needs `SDKROOT`. Plain `cargo check` for macOS fails in `zstd-sys`.
- A C compiler is required per target because `zstd-sys` compiles C. No system library is linked.
- `zstd` is pinned to the measured version with default features off. A bump needs a benchmark rerun.
- pyo3 selects the extension-module build through `PYO3_BUILD_EXTENSION_MODULE`, not a cargo feature.

## Verification

Run after each change, from the workspace root:

```bash
cargo check-all                      # check --workspace --all-targets, linux + windows-gnu
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --target x86_64-pc-windows-gnu -- -D warnings
cargo clippy -p <each touched crate> --all-targets -- -D warnings
cargo test --workspace
cargo-zigbuild check --workspace --all-targets --target x86_64-apple-darwin   # also proves fuser stays out of darwin
CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER=wine cargo test -p ziptensors-core -p ziptensors-mount -p ziptensors-cli --target x86_64-pc-windows-gnu
grep ziptensors /proc/mounts                 # after tests: prints nothing
(cd crates/ziptensors-py && PYTHONDONTWRITEBYTECODE=1 maturin develop --release && PYTHONDONTWRITEBYTECODE=1 pytest)
find . -type f -newer <stamp> -not -path './dev-docs/*' -not -path './target/*'    # after tests: nothing new
```

The wine run applies once the codec, read and write modules exist. It proves function, never timing.
`cargo test --workspace` mounts FUSE under `temp_dir()` when `/dev/fuse` is writable and `fusermount3` is on
PATH, and skips that test otherwise. `fuser` is pinned exactly: a bump needs the FUSE test and a manual
mount smoke test.
pytest runs in a throwaway virtualenv outside the tree, with `CARGO_TARGET_DIR` and `CARGO_BUILD_BUILD_DIR`
outside it too, and `PYTHONDONTWRITEBYTECODE=1`, or the run leaves `__pycache__` in the tree.

Fuzzing is Linux only and needs nightly. Nothing it makes stays in the tree, so everything goes under
`$T`, a directory outside it. From the repository root:

```bash
CARGO_TARGET_DIR=$T/plain cargo +nightly run --release --manifest-path fuzz/Cargo.toml --bin make_seeds -- $T/seeds
CARGO_BUILD_BUILD_DIR=$T/bd cargo +nightly fuzz build -a --target-dir $T/tgt <target>
$T/tgt/x86_64-unknown-linux-gnu/release/<target> $T/corpus/<target> $T/seeds/<target> \
    -max_total_time=600 -max_len=65536 -rss_limit_mb=3000 -artifact_prefix=$T/artefacts/
```

`cargo-fuzz` and `maturin` are installed only into throwaway locations outside the tree.

## Formatter

The tree is NOT rustfmt-clean. No formatter runs over it; match the surrounding style by hand.

## Build directories

`~/.cargo/config.toml` may set a shared `build-dir`; project files do not override it, and final
artefacts stay in `target/`. The binary for §16 stale-build checks is `target/<profile>/ziptensors`.

**Override of `CLAUDE.md` §2.1 (deferred verification), for parallel workers inside one crate.** A
sibling's unresolved import stops rustc before type-checking, so a shared tree would mask errors. Each
such worker copies the tree to its own directory under the task scratchpad, sets its own
`CARGO_BUILD_BUILD_DIR` and `CARGO_TARGET_DIR` there, and runs full per-crate verification. The manager
copies back the owned directories and verifies once. Workers on separate crates use §2.1 unchanged.

## Format changes

`spec/ziptensors-format.md` owns field widths and ids; its name carries no version. A change to the
spec and to `crates/ziptensors-core/src/chain.rs` or `types.rs` is one change. A new minor also raises
`FORMAT_MINOR` in `crates/ziptensors-core/src/format/header.rs` and the spec title together, and stays
within the compatibility promise of spec section 3.4 (published files stay readable). A new `Error`
variant needs a new `ErrorCode`; existing code values never change.

## Tests

- Fixtures are generated in tests with a seeded PRNG. No test reads a real model file.
- Scratch goes under `std::env::temp_dir()`. Benchmarks over real models run out of tree, with roots
  and output directory as required arguments; `tools/bench/` holds no paths.
- The project has no process-global persistent store.

## Documents

- `dev-docs/` is the development-docs folder. `dev-docs/feasibility/` is a closed record, edited only
  to remove private data. Benchmark results go to `dev-docs/benchmarks/`, machine-local roots to
  `dev-docs/local-notes.md`, which is ignored by VCS.
- `STATE.md` and `CLAUDE.local.md` are untracked.
- File size thresholds are the defaults of `CLAUDE.md` §2.5.
