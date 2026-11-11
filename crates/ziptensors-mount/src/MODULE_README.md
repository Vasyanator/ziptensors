# Module: crates/ziptensors-mount/src

## Purpose

A read-only filesystem view of one container (every member is a regular file at its relative path), or
an overlay of a directory tree whose per-file containers appear as their original files, so programs
that read a model directory (diffusers `from_pretrained`, llama.cpp, ComfyUI, safetensors and
`torch.load` with `mmap`) read compressed models unmodified. A frontend like the CLI builds a `MountFs`
(`MountFs::new` for a container, `MountFs::overlay` for a directory) and hands it to the backend of its
OS; backends do not know which kind they serve.

## Architecture

```text
Container::open -> ReadCache (core) -> MountFs::new(cache, policy, mtime)
                                          Tree::build(member names, policy)   refuses, naming every bad member
                                          one CachedMember per member
tree::scan_view (core) -> overlay::plan -> MountFs::overlay(root, policy, options)
                                          Tree::build_with_dirs(shown names, directories, policy)
                                          Overlay: lazy containers and files, bounded pools
backend (fuse.rs on Linux, winfsp/ on Windows)
                            OS request -> MountFs::{lookup, attr, read_dir, open, read} -> reply
```

Two layers:

- **Backend-neutral** (`tree`, `policy`, `fs`, `overlay`, `error`): no OS type, builds and is unit-tested on Linux,
  windows-gnu and macOS. It owns names, node ids and read clamping.
- **Backends**, one per OS behind `cfg`: `fuse.rs` for Linux (`fuser`, pure Rust, `/dev/fuse` and
  `fusermount3`, no libfuse), `winfsp/` for Windows (WinFsp's DLL loaded at run time, Win32 through
  `windows-sys`). There is no macOS backend. Both export the same names (`mount`, `MountHandle`,
  `UnmountSignals`, `run_until_signal`), so a frontend keeps one flow with `cfg` blocks.
- **Handle contract** (`handle`, Linux and Windows): how a mount ended (`MountEnd`) and what both
  `MountHandle`s promise, so a frontend that only holds a handle (a Python context manager) needs no
  `cfg` at all.

## Files and submodules

| Path | Role |
|---|---|
| `lib.rs` | Crate root, lints, re-exports. Each backend's items are re-exported only on its OS. |
| `policy.rs` | `NamePolicy`, `PosixPolicy`, `WindowsPolicy`, `NameProblem`: pure name validation and the key that says when two names are one entry. |
| `tree.rs` | `Tree::build` and the node arena: `NodeId`, `Node`, `DirEntry`, `TreeError`, `BadName`. |
| `fs.rs` | `MountFs`, `Attr`, `Access`, the private `Backing` (one container, or an overlay): the calls a read-only filesystem needs, over the tree. Neutral tests of both kinds. |
| `overlay.rs` | `OverlayOptions`, `plan` (what an overlay shows of a core `TreeView`), `Overlay` (lazy opens, `HandlePool` bounds, staleness checks). |
| `error.rs` | `FsError` (per call, mapped by each backend), `MountError` (cannot build; per OS, why a handle did not end cleanly), `OverlayProblem` (an entry that refuses an overlay). |
| `handle.rs` | Linux and Windows. `MountEnd`, `signal_name`, and the shared `MountHandle` contract in its header. |
| `fuse.rs` | Linux only. `fuser::Filesystem`, `mount`, `MountHandle`, `UnmountSignals`, `run_until_signal`. |
| `winfsp/` | Windows only. The WinFsp backend; its own `MODULE_README.md`. |
| `testkit.rs` | Test only. Scratch directories, real containers packed from seeded files, and `OverlayFixture`, a model directory with per-file containers, a kept pair, a pack, an empty directory and a link. Also included by `../tests/fuse.rs` and `../tests/winfsp.rs`. |

`../tests/fuse.rs` mounts a real container on Linux and compares reads through the kernel (whole,
positional, `mmap` through a `python3 -I` child) with `Member::read_at`, checks `access`, and checks that an
unmount detaches at once while an open file is served until closed. It skips with one line when
`/dev/fuse` or `fusermount3` is missing, and its `MountHandle` unmounts even when an assertion fails. It
also mounts an `OverlayFixture` and checks names, bytes, times and a file changed after the mount (EIO);
`../tests/winfsp.rs` does the same on Windows.

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
- Both `MountHandle`s: `unmount(self) -> Result<(), MountError>` detaches at once and returns once
  every open file is closed; `wait(self) -> Result<MountEnd, MountError>` returns when the mount ends
  unasked (Linux: an outside `fusermount3 -u`, `Unmounted`; Windows: the driver stopping the volume,
  `Stopped`); drop unmounts and logs. A handle is `Send`, not `Sync`; both calls block the calling
  thread and take no signal. Signals and console keys belong to `run_until_signal` alone, whose
  signature stays per OS (Linux `io::Result<MountEnd>`, Windows `MountEnd`) because only the CLI uses it.
- `MountEnd::Abandoned { signal }` carries a number: the POSIX signal on Linux, the C runtime's
  `SIGINT` 2 or `SIGBREAK` 21 for Ctrl-C or Ctrl-Break on Windows, so `exit_code` is 128 + it on both.
  `MountEnd::Stopped` and `MountError::Stopped` exist only on Windows, `MountError::{Unmount, Session}`
  only on Linux.
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

### Overlay (`overlay.rs`, `MountFs::overlay`)

- What is shown follows the core's `scan_view` classification, which alone decides what a per-file
  container is (magic, one member with one safe name): `Plain` files and `Foreign` containers (packs)
  as they are on disk; `PerFile` as its member, under the member's name, with the member's length; a
  `Shadowed` container (its original is still beside it, a `--keep` pair) is hidden and the original is
  read directly; every `Directory`, empty ones included; `Skipped` entries (links, special files,
  `*.ziptensors-tmp`, `*.incomplete`, names that are not UTF-8) are left out with one log line each. A
  container shadowed by a link is therefore not shown at all.
- Fail closed: any view `Conflict` or `Broken` entry (or a view entry kind this version does not know)
  refuses the whole mount with `MountError::Overlay` naming every one; names the policy refuses are
  `MountError::Names`, as for a container. The root is canonicalized, so a linked root shows its target.
- Snapshot: the tree, lengths and times are those of the scan at mount. Each read compares the file's
  length and modification time on disk with the snapshot when it opens the file and after the read, and
  fails with `FsError::Io` (logged with the path) on any difference or a missing file, so bytes are never
  a mix of two versions. Detection rests on the modification time: a writer that restores it undetected
  is not caught for plain files; containers additionally verify chunk hashes. Bytes the kernel cached
  before a change (FUSE `KEEP_CACHE`) or a cache window decoded before it are the snapshot's.
- Handles: a container opens on its first read (`OverlayOptions::threads`, `max_total_decoded`) with its
  own `ReadCache` of `cache.max_bytes / max_open_containers`; that share must hold 2 ×
  `chain::MAX_CHUNK_SIZE` (checked at mount, since chunk sizes are unknown before an open). At most
  `max_open_containers` (default 4) and `max_open_files` (default 64) handles are pooled, opening ones
  included; the least recently used is closed first, and a closed handle lives on while a reader still
  holds its clone. Opening runs outside the pool lock; readers of one file wait for its single open. A
  failed open is not remembered.
- Memory: at most `max_open_containers` caches within the shared budget, plus their indexes, plus each
  cache's excess and engine bound (see the core `ReadCache`), plus handles closed while a read is in
  flight. One shared cache across containers would need a core change.
- `MountFs::stats` of an overlay sums the counters of every cache it had (closed ones as they were when
  closed) and the resident bytes of the open ones; its `peak_bytes` is an estimate.

## Editing map

- A name rule of one OS: `policy.rs`, with a test per rejection.
- What a node is, id assignment, listing order: `tree.rs`.
- Read clamping, attributes, what a call refuses: `fs.rs`.
- What an overlay shows, its pools and staleness rule, its options: `overlay.rs` (classification itself:
  the core's `tree/view.rs`).
- A FUSE call, mount options, unmount and signals: `fuse.rs`.
- How a mount ends, exit codes, what every handle promises: `handle.rs` (and both backends with it).
- A WinFsp call, volume parameters, unmount and console keys: `winfsp/` (see its editing map).
- A new backend: a new `cfg`-gated file over `MountFs`, re-exported from `lib.rs`.
