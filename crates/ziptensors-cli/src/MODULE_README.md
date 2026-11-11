# Module: crates/ziptensors-cli/src

## Purpose

The `ziptensors` binary. Parses arguments, calls `ziptensors-core` (and on Linux and Windows `ziptensors-mount`),
reports results. Holds no format, codec or policy decision: anything of that kind belongs in the core.

## Commands

```text
ziptensors encode <INPUT> <OUTPUT> [--mode M] [--threads N]
ziptensors pack <OUTPUT> <INPUT>... [--delta-base FILE] [--mode M] [--threads N]
ziptensors edit <CONTAINER> [--add PATH]... [--add-as NAME PATH]... [--remove NAME]... [--replace NAME PATH]...
                [--delta-base MEMBER] [-F|--fast | -R|--risky] [-C|--compact] [--mode M] [--threads N] [B]
ziptensors compact <CONTAINER> [-R|--risky] [-C|--compact] [--mode M] [--threads N] [B]
ziptensors recover <CONTAINER> [--dry-run] [--no-verify] [--max-candidates N] [--max-scan-bytes N] [--threads N] [B]
ziptensors restore <CONTAINER> <OUTPUT> [--threads N] [B]
ziptensors restore <CONTAINER> --all <DIR> [--threads N] [B]
ziptensors extract <CONTAINER> <MEMBER> <OUTPUT|-> [--threads N] [B]
ziptensors merge <OUTPUT> <INPUT>... [--threads N] [B]
ziptensors inspect <CONTAINER> [--json] [--tensors] [B]
ziptensors verify <CONTAINER> [--threads N] [B]
ziptensors compress [-r] <PATH>... [-k|--keep] [--follow-symlinks] [--dry-run] [--mode M] [--threads N] [B]
ziptensors decompress [-r] <PATH>... [-k|--keep] [-f|--force] [--dry-run] [--threads N] [B]
ziptensors mount <CONTAINER> <MOUNTPOINT> [--threads N] [--fuse-threads N] [--cache-size BYTES]
                 [--readahead WINDOWS] [B]                                                (Linux)
ziptensors mount <CONTAINER> <X:|NEW-DIR> [--threads N] [--winfsp-threads N] [--cache-size BYTES]
                 [--readahead WINDOWS] [B]                                                (Windows)
ziptensors mount --overlay <DIR> <MOUNTPOINT|X:|NEW-DIR> [--max-open-containers N] [--max-open-files N]
                 [the other mount flags]
```

`[B]` is `--max-total-decoded BYTES`, on every command that opens a container. `--mode` is `fast`,
`balanced` or `max`; the default is `balanced` for `encode` and `max` everywhere else. `--threads 0`, the
default, is one thread per logical CPU. `merge` copies chunks, so it has no mode.

## Files and submodules

| Path | Role |
|---|---|
| `main.rs` | Entry point: parse, logging to stderr, exit code. |
| `cli.rs` | The grammar, in clap derive form, and `Cli::parse_checked`: the checks across arguments clap cannot state, and the command line order of `edit --add`/`--add-as`. Parsing only. |
| `commands.rs` | Dispatch; `open_options`/`open_budgeted`, the one place a budget becomes `OpenOptions`; `encode`, `restore`, `extract`, `merge`, `inspect`, `verify`. |
| `pack.rs` | `pack`: the core walk (`collect_inputs`, `check_output`), the delta base, the add loop. |
| `edit.rs` | `edit`, `compact`, `recover`: flags to `EditOp`/`EditOptions`/`RecoverOptions`, the `-R` warning, the read-only hint of `recover`. |
| `tree.rs` | `compress`, `decompress` over `core::tree`, one run per path, exit 1 if a file failed. |
| `mount.rs` | Linux and Windows. `mount`: signal mask or console handler, `MountFs` of a container (open, cache) or of a directory (`--overlay`), backend mount, wait, cache stats. |
| `report.rs` | Every printed result, the per-member `Progress` line, the `inspect --json` document, the lines of tree runs, the `recover` warning. |
| `json.rs` | `Json`, a print-only value. |
| `errors.rs` | Human message and hint per `ErrorCode`, `Hint` (a hint chosen by a command), structured error log. |

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
- `pack` walks through `core::write::inputs` (`collect_inputs`, `check_output`): the walk rules (no link
  followed, `*.incomplete` skipped and counted, names checked and sorted per input, an output the walk would
  pick up refused) are the core's. Everything is checked before the output is started; a pack with no file
  to add is a failure.
- Every command that opens a container takes `--max-total-decoded`. There is no budget by default, as in
  the core: untrusted input needs one.
- `pack --delta-base FILE`: `FILE` must be a regular file the walk picks up, given directly or under an
  input directory, compared by canonical path; otherwise the pack fails before the output is started.
  It is added first, and every other file goes through `PackBuilder::add_file_delta` against it, in
  walk order. One base per pack, so several finetunes of one base pack in one run, and a finetune of
  a finetune is stored against the base, never against the other (the format allows depth 1). The
  CLI does not filter by format: which tensors pair is the core's decision, and a file that pairs
  nothing is stored as `add_file` stores it. The base never counts as a skipped link. With an
  explicit `--mode` other than `max` the flag is a usage error (exit 2): only `max` stores delta
  blobs, and a flag ignored for the hours of a pack is worse than a refusal.
- `merge` opens every input and checks all their member names with `core::check_member_names` before
  the output is started, so a name two inputs share (`DuplicateMember`) copies nothing.
- `encode`, `pack` and `merge` print `delta blobs: N`, the delta blobs the members added (spec section
  7a); `merge` counts the delta blobs it copied.
- `pack` and `merge` log one line per member with `cumulative_saved`, in bytes, and `delta_blobs`.
- `restore --all` is `Container::restore_all`, which checks every name before creating anything.
- `extract ... -` streams the member to stdout through `Member::decode_to`, with the result line on
  stderr. Bytes leave before the member hash is known; a mismatch is exit 1 after the fact, and the help
  of OUTPUT tells the reader to discard what it got.
- `edit`/`compact`: `-F` is `Strategy::Append`, `-R` `Strategy::InPlace`, neither `Strategy::Rewrite`,
  `-C` is `repack`. Usage errors (exit 2): `-F` with `-R`, `-F` with `-C`, `-R` with `-C`, `edit` with no
  operation, `--delta-base` with a `--mode` other than `max` (the rule of `pack`). Added members keep the
  command line order of `--add` and `--add-as`; `--add` names a member after its file. `--delta-base`
  applies to every added or replacing file but the base itself. Every other refusal is the core's
  (`EditRefused`, exit 1, file untouched). `-R` prints a warning line on stderr before it starts, and
  its help says in capitals that an interruption damages the container and `recover` may not repair it.
- `recover` prints the generation kept, both lengths and the count of newer candidates passed over;
  whenever bytes are cut (or on `--dry-run` would be), a `warning:` line on stderr names the generation
  and the bytes discarded, and, only when the count is above 0, how many newer candidate generations
  failed the checks and were passed over. A refused open for writing gets a hint to `--dry-run`.
- `compress`/`decompress`: a directory needs `-r` (usage error 2, checked in `parse_checked`); each path is
  one run of `core::tree`. One stdout line per path the run looked at, then totals and a `bytes:` line
  of the direction (`report::TreeDirection`, which the core's report does not carry); every arrow of a
  line, paths and sizes, points from source to result (compress: file to container; decompress:
  container to file); warnings of the run
  (skipped links, naming `--follow-symlinks` and the Hugging Face cache) on stderr. A failed file does not
  stop the run; the command exits 1 at the end if any failed. A collision refuses the whole run (exit 1,
  nothing written).
- `mount` exists on Linux and Windows (`cfg(any(target_os = "linux", windows))` on the variant, the file
  and the dependency on `ziptensors-mount`), in the foreground, through one flow in `mount.rs` with `cfg`
  blocks. It prints one `mounted ...` line with how to unmount, and at the end one `unmounted ...` line
  and logs the cache counters. Files still open when the unmount starts are served until closed; a second
  request in that time prints `ended by a second ...` and exits at once with 128 + the signal, cutting the
  readers off. Defaults: `--cache-size` the core's 256 MiB, `--readahead 0` (the decode thread count), no
  `--max-total-decoded`. No `--allow-other`, no daemon mode.
- `mount --overlay` shows a directory through `MountFs::overlay` (what is shown, refused and when a
  read fails: `crates/ziptensors-mount`). The first positional is a directory with `--overlay` and
  anything else without it; the wrong kind is a usage error (exit 2) in `parse_checked`, a missing path
  the command's error. `--max-open-containers` and `--max-open-files` (at least 1, defaults those of
  `OverlayOptions`) need `--overlay` (clap `requires`, exit 2). `--cache-size` is the budget of all
  open containers together; `--threads`, `--readahead` and `--max-total-decoded` apply to each
  container. Signals, the printed lines, unmount and exit codes are those of the container mount.
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

The schema a consumer may rely on is user documentation: `docs/cli.md`, section `inspect`, is its one
copy, and `report::inspect_json` must match it. Changing a key there and here is one change; removing or
redefining a key raises `schema_version`. `layers` is `core::layer_report` over all members, from the
index alone; `classes` are ordered skeleton first, then by dtype id.

## Editing map

- New command: `cli.rs`, a function over a public `core` API (`commands.rs`, or the file of its family:
  `edit.rs`, `tree.rs`), its result in `report.rs`.
- New printed field: `report.rs`, and for JSON the schema in `docs/cli.md`.
- What `pack` takes from a tree, and how it uses a delta base: `pack.rs`; which tensors pair and when a
  delta blob is stored: `crates/ziptensors-core/src/write` (`pairing.rs`, `policy.rs`).
- How `mount` starts and ends: `mount.rs`; what the mount shows and refuses: `crates/ziptensors-mount`.
- Hint of a failure: `errors.rs`; a hint only one command knows: `errors::Hint` as context there. A
  malformed or undecodable index (a torn append among other damage) points to `recover --dry-run`.
- A usage error across arguments: `Cli::parse_checked`, or a clap `conflicts_with`/`ArgGroup` in `cli.rs`.
