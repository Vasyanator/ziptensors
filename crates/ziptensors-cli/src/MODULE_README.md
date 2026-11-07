# Module: crates/ziptensors-cli/src

## Purpose

The `ziptensors` binary. Parses arguments, calls `ziptensors-core` (and on Linux and Windows `ziptensors-mount`),
reports results. Holds no format, codec or policy decision: anything of that kind belongs in the core.

## Commands

```text
ziptensors encode <INPUT> <OUTPUT> [--mode M] [--threads N]
ziptensors pack <OUTPUT> <INPUT>... [--delta-base FILE] [--mode M] [--threads N]
ziptensors restore <CONTAINER> <OUTPUT> [--threads N] [--max-total-decoded BYTES]
ziptensors restore <CONTAINER> --all <DIR> [--threads N] [--max-total-decoded BYTES]
ziptensors extract <CONTAINER> <MEMBER> <OUTPUT> [--threads N]
ziptensors merge <OUTPUT> <INPUT>... [--threads N]
ziptensors inspect <CONTAINER> [--json]
ziptensors verify <CONTAINER> [--threads N]
ziptensors mount <CONTAINER> <MOUNTPOINT> [--threads N] [--fuse-threads N] [--cache-size BYTES]
                 [--readahead WINDOWS] [--max-total-decoded BYTES]                        (Linux)
ziptensors mount <CONTAINER> <X:|NEW-DIR> [--threads N] [--winfsp-threads N] [--cache-size BYTES]
                 [--readahead WINDOWS] [--max-total-decoded BYTES]                        (Windows)
```

`--mode` is `fast`, `balanced` or `max`; the default is `balanced` for `encode` and `max` for `pack`.
`--threads 0`, the default, is one thread per logical CPU. `merge` copies chunks, so it has no mode.

## Files and submodules

| Path | Role |
|---|---|
| `main.rs` | Entry point: parse, logging to stderr, exit code. |
| `cli.rs` | The grammar, in clap derive form, and `Cli::parse_checked`, the one check across arguments. Parsing only. |
| `commands.rs` | One function per subcommand except `pack`. `restore --all` and its name check. |
| `pack.rs` | Directory walk, skips, early name checks, the delta base, the add loop of `pack`. |
| `mount.rs` | Linux and Windows. `mount`: signal mask or console handler, open, cache, `MountFs`, backend mount, wait, cache stats. |
| `report.rs` | Every printed result, the per-member `Progress` line, the `inspect --json` document. |
| `json.rs` | `Json`, a print-only value. |
| `errors.rs` | Human message and hint per `ErrorCode`, structured error log. |

`../tests/cli.rs` runs the real binary; `../tests/common/mod.rs` holds its scratch directory, process
runner, strict JSON parser and hostile container builder.

## Contracts and invariants

- Exit codes: 0 success, 1 runtime failure, 2 usage, and for `mount` ended by a second signal 128 + the
  signal number (130 for Ctrl-C; on Windows 149 for Ctrl-Break, the C runtime's `SIGBREAK`). Part of
  the interface.
- Stdout carries results only; logs, progress and errors go to stderr. Logs are `tracing` at `INFO`.
- A failed command prints `error: <what>: <why>`, a `hint:` line for a failure of the core, and an `ERROR`
  log with the `ErrorCode` and the error's fields.
- `anyhow` is used only at this boundary. Failures of the core stay downcastable through its context.
- `pack` walks without following links and orders each input by member name, so the same tree gives
  the same member order. Member names are relative to the input, `/` separated; a file given directly is
  named after itself.
- `pack` skips symbolic links and `*.incomplete` files, with one log line each and a count in the summary.
  Everything else stops it: an unreadable entry, a name that is not UTF-8, an entry that is neither file,
  directory nor link, a name the core would refuse, a name given twice. These are checked before the output
  is started. A pack with no file to add is a failure.
- `pack` refuses an output its walk would pick up, inside a walked directory or equal to an input, whether
  the output exists yet or not, before anything is walked.
- `restore` and `extract` take `--max-total-decoded`. There is no budget by default, as in the core: untrusted
  input needs one.
- `pack --delta-base FILE`: `FILE` must be a regular file the walk picks up, given directly or under an
  input directory, compared by canonical path; otherwise the pack fails before the output is started.
  It is added first, and every other file goes through `PackBuilder::add_file_delta` against it, in
  walk order. One base per pack, so several finetunes of one base pack in one run, and a finetune of
  a finetune is stored against the base, never against the other (the format allows depth 1). The
  CLI does not filter by format: which tensors pair is the core's decision, and a file that pairs
  nothing is stored as `add_file` stores it. The base never counts as a skipped link. With an
  explicit `--mode` other than `max` the flag is a usage error (exit 2): only `max` stores delta
  blobs, and a flag ignored for the hours of a pack is worse than a refusal.
- `encode`, `pack` and `merge` print `delta blobs: N`, the delta blobs the members added (spec section
  7a); `merge` counts the delta blobs it copied.
- `pack` and `merge` log one line per member with `cumulative_saved`, in bytes, and `delta_blobs`.
- `restore --all` checks every name with `format::check_restore_name` before creating anything.
  The reader does not check names at open, so this is the only guard on that path.
- `mount` exists on Linux and Windows (`cfg(any(target_os = "linux", windows))` on the variant, the file
  and the dependency on `ziptensors-mount`), in the foreground, through one flow in `mount.rs` with `cfg`
  blocks. It prints one `mounted ...` line with how to unmount, and at the end one `unmounted ...` line
  and logs the cache counters. Files still open when the unmount starts are served until closed; a second
  request in that time prints `ended by a second ...` and exits at once with 128 + the signal, cutting the
  readers off. Defaults: `--cache-size` the core's 256 MiB, `--readahead 0` (the decode thread count), no
  `--max-total-decoded`. No `--allow-other`, no daemon mode.
- Linux: blocks SIGINT, SIGTERM and SIGHUP before opening the container (the decode pool must inherit the
  mask) and ends on any of them or an outside `fusermount3 -u`; names are checked under `PosixPolicy`;
  the mount point is an existing directory; `--fuse-threads 4`.
- Windows: installs the Ctrl-C/Ctrl-Break handler before opening the container and ends on either;
  names are checked under `WindowsPolicy::default()` before WinFsp is loaded; the mount point is a free
  drive letter (`X:`) or a directory that does not exist yet, and members whose full path would exceed
  `MAX_PATH` under it refuse the mount, all named; a letter in use or a path that exists is refused
  before WinFsp is loaded; `--winfsp-threads 4`, range 0 to 64 (0: WinFsp's default). A volume the
  WinFsp driver stops on its own is a failure: `error: the WinFsp driver stopped the volume ...`, exit 1. Without WinFsp the command fails
  naming <https://winfsp.dev/rel/>. The `mount --help` text carries the notice the WinFsp license
  requires: "WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos" and
  <https://github.com/winfsp/winfsp>; it must stay there and in any user-facing document of the mount.
- On either system a tree the policy cannot show fails with exit 1 and every bad member named.

## `inspect --json`

One document on one line of stdout. A consumer may rely on these keys. A new key does not change
`schema_version`; removing or redefining a key does.

```text
schema_version  number  1
path            string  the path as given
file_bytes      number  length of the container
raw_bytes       number  sum of member lengths
saved_bytes     number  raw_bytes - file_bytes, 0 if that is negative
members         array, in file order, of:
  name          string  member name, `/` separated
  len           number  bytes
  blake3        string  64 lowercase hex digits
  profile       string  opaque, safetensors, gguf, onnx, pt_zip, or unknown-<id>
  tensor_count  number
layers          object, `core::layer_report` over all members, from the index alone:
  raw, widen_saved, dedup_saved, entropy_saved, overhead, stored   number
classes         array, skeleton first, then by dtype id, of the fields of `layers` plus:
  class         string  skeleton, or the dtype
```

Numbers are exact `u64` values.

## Editing map

- New command: `cli.rs`, a function in `commands.rs` over a public `core` API, its result in `report.rs`.
- New printed field: `report.rs`, and the schema above for JSON.
- What `pack` takes from a tree, and how it uses a delta base: `pack.rs`; which tensors pair and when a
  delta blob is stored: `crates/ziptensors-core/src/write` (`pairing.rs`, `policy.rs`).
- How `mount` starts and ends: `mount.rs`; what the mount shows and refuses: `crates/ziptensors-mount`.
- Hint of a failure: `errors.rs`.
