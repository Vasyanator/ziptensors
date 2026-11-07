# Module: crates/ziptensors-mount/src

## Purpose

A read-only filesystem view of one container: every member is a regular file at its relative path, so
programs that read a model directory (diffusers `from_pretrained`, llama.cpp, ComfyUI, safetensors and
`torch.load` with `mmap`) read a compressed container unmodified. A frontend like the CLI builds a
`MountFs` and hands it to the backend of its OS.

## Architecture

```text
Container::open -> ReadCache (core) -> MountFs::new(cache, policy, mtime)
                                          Tree::build(member names, policy)   refuses, naming every bad member
                                          one CachedMember per member
backend (fuse.rs on Linux, winfsp/ on Windows)
                            OS request -> MountFs::{lookup, attr, read_dir, open, read} -> reply
```

Two layers:

- **Backend-neutral** (`tree`, `policy`, `fs`, `error`): no OS type, builds and is unit-tested on Linux,
  windows-gnu and macOS. It owns names, node ids and read clamping.
- **Backends**, one per OS behind `cfg`: `fuse.rs` for Linux (`fuser`, pure Rust, `/dev/fuse` and
  `fusermount3`, no libfuse), `winfsp/` for Windows (WinFsp's DLL loaded at run time, Win32 through
  `windows-sys`). There is no macOS backend. Both export the same names (`mount`, `MountHandle`,
  `MountEnd`, `UnmountSignals`, `run_until_signal`), so a frontend keeps one flow with `cfg` blocks.

## Files and submodules

| Path | Role |
|---|---|
| `lib.rs` | Crate root, lints, re-exports. Each backend's items are re-exported only on its OS. |
| `policy.rs` | `NamePolicy`, `PosixPolicy`, `WindowsPolicy`, `NameProblem`: pure name validation and the key that says when two names are one entry. |
| `tree.rs` | `Tree::build` and the node arena: `NodeId`, `Node`, `DirEntry`, `TreeError`, `BadName`. |
| `fs.rs` | `MountFs`, `Attr`, `Access`: the calls a read-only filesystem needs, over the tree and the core `ReadCache`. |
| `error.rs` | `FsError` (per call, mapped by each backend) and `MountError` (cannot build). |
| `fuse.rs` | Linux only. `fuser::Filesystem`, `mount`, `MountHandle`, `UnmountSignals`, `run_until_signal`. |
| `winfsp/` | Windows only. The WinFsp backend; its own `MODULE_README.md`. |
| `testkit.rs` | Test only. Scratch directories and real containers packed from seeded files. Also included by `../tests/fuse.rs` and `../tests/winfsp.rs`. |

`../tests/fuse.rs` mounts a real container on Linux and compares reads through the kernel (whole,
positional, `mmap` through a `python3 -I` child) with `Member::read_at`, checks `access`, and checks that an
unmount detaches at once while an open file is served until closed. It skips with one line when
`/dev/fuse` or `fusermount3` is missing, and its `MountHandle` unmounts even when an assertion fails.

## Contracts and invariants

- No format logic: bytes come only from `CachedMember::read_at`. Window geometry belongs to the core cache,
  which is why reads far smaller than a chunk (FUSE sends 128 KiB) do not decode a chunk each.
- A tree that cannot be shown does not mount. `Tree::build` checks every member, collects every problem,
  sorts them by name and fails with one `TreeError`; nothing is half built. Every policy first applies
  `format::check_restore_name`, so nothing a restore refuses is mounted.
- Under a policy, two entries of one directory are equal when their `NamePolicy::key` is. `PosixPolicy`:
  the name itself, parts up to 255 bytes, no empty, `.` or `..` part. `WindowsPolicy`: simple per-character
  Unicode upper case (`ß` stays), forbidden `<>:"|?*\` and controls, reserved device names with any
  extension (Microsoft's list, `COM0`, `LPT0` and the superscript digits included), no trailing dot or
  space, parts up to 255 UTF-16 units, names up to 256 units (32763 with `long_paths`).
- The Unicode upper case folds more than an NTFS volume's `$UpCase` (`ı`/`I`, `ſ`/`S`): such a pair is a
  collision and refuses the mount where Windows would show both. This direction is deliberate: a false
  collision refuses, never serves two members as one. No case is known where it folds less.
- A file that is also the directory of other members (`a` and `a/b`) is reported once, as the file.
- Node ids are assigned breadth first from the root (1), each directory in name order: the same container
  gives the same ids, and an id is never reused while the tree lives. Listings are sorted by name.
- `MountFs::read` returns fewer bytes only at the end of a file and none at or past it. A core failure is
  logged with member, offset, length and `ErrorCode` and becomes `FsError::Io`.
- Everything a filesystem would change is refused: `Access::Write` is `FsError::ReadOnly`; the FUSE mount
  is also `ro`, `nodev`, `nosuid`, `noatime`.
- `MountFs` is `Send + Sync` and every method takes `&self`, for a backend's dispatch threads.
- Which policy applies is the caller's choice: the Linux CLI mounts with `PosixPolicy`, the Windows CLI
  with `WindowsPolicy::default()`, and the WinFsp backend refuses any other. A check of a container
  against Windows rules on Linux is `Tree::build(names, WindowsPolicy::default())`; no CLI flag exposes it.
- `unsafe` is denied crate-wide and allowed per file only in `winfsp/` (FFI to WinFsp and Win32), each
  block with a `SAFETY` comment (`clippy::undocumented_unsafe_blocks`).

### Linux backend (`fuse.rs`)

- Attributes and entries have a TTL of one hour and every open sets `FOPEN_KEEP_CACHE`: contents never
  change, so a second read of a file costs no request. Files are `0o444`, directories `0o555`, owned by
  the mounting user, timed by the container file's modification time. The mount uses
  `default_permissions`, so the kernel enforces those bits (`X_OK` on a file is refused) and never asks
  `access`.
- No `allow_other` and no `AutoUnmount` (it needs `allow_other`): only the mounting user sees the files,
  and a process killed by SIGKILL leaves the mount behind for `fusermount3 -u`.
- The session runs `Session::run` on a thread of this crate, not `Session::spawn`: `spawn` takes the mount
  out of the slot its `SessionUnmounter` reads, which then cannot unmount.
- An unprivileged unmount is `fusermount3 -u -z` (run by `fuser`), so it is lazy: the mountpoint is
  detached at once, files open under it, mappings included, are served until closed, and only then does
  the session end. A session that ends with `ECONNABORTED` after such an unmount ended normally.
  `fuser` itself may log one "Failed to send FUSE reply" for the last release in that case.
- `fuser` reports an unprivileged unmount as done whatever `fusermount3` did, and gives up its mount on
  the first attempt. So every unmount of this crate (`MountHandle::unmount`, its drop, the signal thread)
  checks `/proc/self/mountinfo` for the mountpoint, resolved at `mount` time, and treats a mount still
  listed as a failure: logged with the `fusermount3 -u -z` hint, the session thread detached, not joined.
- `UnmountSignals::block` must run before any other thread starts, the core's decode pool included
  (`Container::open`). SIGINT, SIGTERM and SIGHUP then stay blocked on that thread. `run_until_signal`
  waits for them with `sigwait` on its own thread: the first unmounts (and logs an ERROR if the mount is
  still there), the second returns `MountEnd::Abandoned` at once with the session detached, and the caller
  exits with `MountEnd::exit_code` (128 + signal). The thread is woken with `pthread_kill` when the mount
  ended otherwise.
- Calls a reader makes on every file (`flush`, `getxattr`, `listxattr`, `ioctl` `TCGETS2` from Python,
  `poll`) are answered explicitly, because `fuser`'s defaults log a warning per call.

### Windows backend (`winfsp/`)

Contracts in `winfsp/MODULE_README.md`. In short: WinFsp - Windows File System Proxy, Copyright (C)
Bill Zissimopoulos, <https://github.com/winfsp/winfsp>, loaded from its install directory at run time
(no WinFsp is a typed `NotInstalled` error naming <https://winfsp.dev/rel/>); a drive letter or a
directory that does not exist yet; read-only and visible to the mounting user only; a first Ctrl-C
detaches and waits for open files (not directories), a second abandons. Any program built on this crate
for Windows must carry the WinFsp notice and link in its user interface and user-facing documentation,
and a proprietary one needs WinFsp's commercial license.

## Editing map

- A name rule of one OS: `policy.rs`, with a test per rejection.
- What a node is, id assignment, listing order: `tree.rs`.
- Read clamping, attributes, what a call refuses: `fs.rs`.
- A FUSE call, mount options, unmount and signals: `fuse.rs`.
- A WinFsp call, volume parameters, unmount and console keys: `winfsp/` (see its editing map).
- A new backend: a new `cfg`-gated file over `MountFs`, re-exported from `lib.rs`.
