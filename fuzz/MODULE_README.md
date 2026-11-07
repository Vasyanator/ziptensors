# Module: fuzz

## Purpose

Fuzz targets over the three surfaces that take untrusted bytes. Excluded from the workspace because it
needs nightly, with its own lockfile. The same functions run on stable through a seeded mutator in
`crates/ziptensors-core/tests/malicious.rs`, so CI covers them without this crate.

## Architecture

`src/harness.rs` holds everything: one function per target, the oracles, the input forms and the mutator.
`fuzz/src/lib.rs` exposes it to the targets, and `tests/malicious.rs` compiles the same file through
`#[path]`, so there is one copy of each oracle. A target file is one line that calls the harness.

## Files and submodules

| Path | Role |
|---|---|
| `src/harness.rs` | Targets, oracles, `assemble`, `sample`, `delta_sample`, `mutate`. May use only `ziptensors_core`, `zstd`, `blake3`, `xxhash_rust`, `std`. |
| `src/lib.rs` | Crate root. |
| `src/bin/make_seeds.rs` | Writes the seed corpus to a directory it is given. |
| `fuzz_targets/*.rs` | `open_and_read`, `decode_chunk`, `profile_scan`. |

| Target | Surface | Input |
|---|---|---|
| `open_and_read` | `Container`, `Member` | Form byte, then file bytes, or `u32` length, data region and raw index. |
| `decode_chunk` | `decode_chunk_into`, `encode_chunk`, and their delta forms | Chain, form, two length bytes, body. |
| `profile_scan` | `profile::scan`, `normalize`, `width_trial` | A source file. |

## Contracts and invariants

- A target may only see a typed `Error` of the class its surface documents, or success. A panic, abort or
  sanitizer report is a finding, and so is `MalformedIndex` from a read of a validated file, since that is a
  state `validate` claims to rule out.
- A fuzzer cannot guess an XXH3, so raw bytes stop at the header hash. The fixed-up form of `open_and_read`
  assembles valid header, footer and index hashes around its input, which is what reaches validation and
  decoding. Chunk hashes are not fixed up, so every file is also read with `Verify::Off`. Bit 1 of its form
  byte sets required feature bit 0 in the header, without which a file with delta blobs is refused.
- Delta blobs (spec section 7a): `open_and_read` is seeded with `delta_sample`, a container of delta blobs
  over one base. `chain_table` is `PLAIN_CHAINS` chains, then each of them after a `delta_xor` step; for
  those `decode_chunk` takes a base of the chunk's length from the body, or zero with form bit 4, and the
  oracle adds that a delta decode is the residual, decoded by the rest of the chain, XOR the base.
- Oracles beyond "no panic": whole, partial and `read_many` reads agree; `verify` passing means the bytes
  hash to the member; a decode does not depend on what its destination held, which is how a read ahead of
  the write head in the in-place route (risk R1) would show; guard bands around a chunk stay intact.
- The harness runs under an index cap of 4 MiB, a budget of 64 MiB and at most 1 MiB per member, so an
  input costs milliseconds. The limits themselves are exercised in `tests/malicious.rs`.
- Nothing generated stays in the tree. Corpus, artefacts and build output go to directories named on the
  command line; `.gitignore` covers what `cargo fuzz` makes when run without them.
- Linux only.

## Running

Install `cargo-fuzz` into a throwaway root. From the repository root, with `$T` outside the tree:

```bash
CARGO_TARGET_DIR=$T/plain cargo +nightly run --release --manifest-path fuzz/Cargo.toml --bin make_seeds -- $T/seeds
CARGO_BUILD_BUILD_DIR=$T/bd cargo +nightly fuzz build -a --target-dir $T/tgt open_and_read
$T/tgt/x86_64-unknown-linux-gnu/release/open_and_read $T/corpus/open_and_read $T/seeds/open_and_read \
    -max_total_time=600 -max_len=65536 -rss_limit_mb=3000 -artifact_prefix=$T/artefacts/
```

`-a` keeps debug assertions on, which is what asserts the in-place proof of `codec/kernels_x86.rs` per step.

## Editing map

- New parser or decoder of untrusted bytes: a function in `src/harness.rs`, a target here, a case in
  `tests/malicious.rs`.
- A stronger oracle: `src/harness.rs` only; both runners get it.
- New chain or field table: `chain_table` in `src/harness.rs`; `make_seeds.rs` follows the table, but its
  in-place seeds name `planes` chains by index.
- New delta operation or blob kind: a form in `chain_table` or `delta_sample`, and the cases in
  `tests/conformance.rs` and `tests/malicious.rs`.
