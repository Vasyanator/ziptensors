# Module: crates/ziptensors-mount/src/winfsp

## Purpose

The Windows backend of the mount: serves a `MountFs` built with `WindowsPolicy` through WinFsp, on a
drive letter or on a directory that does not exist yet, until Ctrl-C. Built only for `cfg(windows)`.

WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos, <https://github.com/winfsp/winfsp>.
It is GPL-3.0 with a FLOSS exception that lets an OSI-licensed program such as this one use its DLL,
provided the program shows that notice and link in its user interface and user-facing documentation
and is not distributed with proprietary software. A proprietary program that embeds this crate on
Windows needs WinFsp's commercial license. No WinFsp file is distributed with ziptensors.

## Architecture

```text
mount(fs, mountpoint, options)
  policy == windows? -> MountPoint::parse (+ check_absolute) -> too_long(tree, root, MAX_PATH)
  check_free (letter defined? path exists?)                                  all before WinFsp loads
  Dll::load (registry InstallDir + bin\winfsp-x64.dll) -> Security::for_current_user
  FspFileSystemPreflight(mount point) -> FspFileSystemCreate(WinFsp.Disk, volume_params, &INTERFACE)
  UserContext = &Volume
  FspFileSystemSetMountPoint -> FspFileSystemStartDispatcher            -> MountHandle { Live }
dispatcher threads -> callbacks.rs -> Volume { MountFs, Security, open objects per file } -> names.rs rules
unmount: RemoveMountPoint (detach) -> until no file object is open: wait 1 s, poke (reopen + close)
         -> StopDispatcher -> Delete -> FreeLibrary
run_until_signal: console handler -> UnmountSignals channel (1 s timeout = Tick -> poke)
                  <- Volume (files closed, driver stopped)
```

## Files and submodules

| Path | Role |
|---|---|
| `mod.rs` | Public API: `mount`, `probe`, `MountHandle` (`unmount`, `wait`, under the crate's `handle.rs` contract), `run_until_signal`, `WinFspOptions`; lifecycle and teardown order (`Live`, `FileSystemGuard`). |
| `abi.rs` | The WinFsp ABI subset: `repr(C)` structures, flag bits, the 64-slot `Interface`, export types, compile-time layout asserts. Declarations only. |
| `dll.rs` | Registry lookup, `LoadLibraryExW` by full path, typed exports (`Api`), unload on drop. |
| `callbacks.rs` | The `extern "C"` operations and the static `INTERFACE`; panic guard; refusing stubs. |
| `volume.rs` | `Volume`: what callbacks share; open file objects per file, the volume's device name, detach, driver-stopped status, wake-ups. |
| `poke.rs` | Reopens and closes files still counted open, through the volume's device path, so a deferred Close arrives once a mapping is gone. |
| `names.rs` | Pure rules: path resolution, `NTSTATUS` mapping, `FILETIME`, write rights, listing resume, volume parameters, `MountPoint` parsing, `MAX_PATH` check. Unit-tested without WinFsp. |
| `security.rs` | The one security descriptor of every node, and its size protocol. |
| `console.rs` | Console control handler, `UnmountSignals`, `ConsoleEvent` (and its C runtime signal number), the foreground state machine `step`. |
| `error.rs` | `WinFspError`. |

`../../tests/winfsp.rs` mounts a real container on a new directory and a free drive letter, compares
whole, positional and mapped reads with the packed bytes, checks that every change is refused and that
an unmount detaches at once while an open file or a mapping is served until closed and unmapped. It
skips with one line when `probe` fails (wine, or no WinFsp). The refusals of mount points in use run
without WinFsp, as unit tests of `mod.rs`, also under wine (which defines `C:` and `Z:`).

## Contracts and invariants

- Windows mounts use `WindowsPolicy`: `mount` refuses a tree built under another policy
  (`WrongPolicy`), so case collisions and reserved names never reach WinFsp.
- A path a normal program cannot open is never served: with `long_paths` off, `mount` refuses, naming
  every member, when mount point + `\` + member exceeds 259 UTF-16 units. `WinFspOptions::long_paths`
  must equal the policy's `long_paths`.
- Mount point forms: `X:` (also `x:`, `X:\`, `X:/`) is a drive; anything else is a directory that must
  not exist, whose parent must; WinFsp creates it as a junction and deletes it at unmount. UNC, device,
  verbatim and drive-relative paths are refused, before and again after `std::path::absolute`
  (normalization turns `C:\CON` into the device path `\\.\CON` and `C:\..` into a drive root).
- Never mount over something that exists. `FspFileSystemSetMountPoint` defines a drive letter whatever
  it held, and from session 0 machine-wide: a mount on `C:` once shadowed the system drive of the whole
  machine. Two guards, in order: `check_free` before WinFsp is loaded (a letter in `GetLogicalDrives` or
  resolved by `QueryDosDeviceW`, so `subst` and network letters count; a path that exists in any form;
  a missing parent), unit-tested without WinFsp; then `FspFileSystemPreflight` before the volume is
  created. Both answer `MountPointTaken` or `MountPoint`. A window remains (not closable through
  WinFsp's API): a letter another process defines between the preflight and `SetMountPoint` is
  shadowed for the life of the mount.
- Drive letter scope: from an interactive console the letter is defined for the caller's logon session
  only. From a session-0 process (a service, an OpenSSH session) WinFsp has its launcher define it
  globally, visible in every session. A directory mount point is a junction, visible to every session
  either way; the security descriptor still admits the mounting user only.
- The DLL is loaded only by full path from `HKLM\SOFTWARE\WOW6432Node\WinFsp\InstallDir`, never through
  the search path. No WinFsp is `NotInstalled`, whose message names <https://winfsp.dev/rel/>. Function
  pointers of the DLL live no longer than the `Dll` (`Live` drops it last).
- Read-only: `WinFsp`'s `ReadOnlyVolume` only reports the flag, so the callbacks refuse every change with
  `STATUS_MEDIA_WRITE_PROTECTED`, and Open refuses any granted write right (`wants_write`). The security
  descriptor grants read and execute to the mounting user and to `OWNER RIGHTS` only, so a
  `MAXIMUM_ALLOWED` open is granted no write right and succeeds. Known limit: an open with backup intent
  and `MAXIMUM_ALLOWED` by a token whose backup and restore privileges are enabled (an elevated session,
  robocopy `/B`) is granted write rights by those privileges and is refused (`ERROR_WRITE_PROTECT`,
  measured natively); backup intent with read access works. It is not relaxed on purpose: a handle with
  `FILE_WRITE_DATA` could write through WinFsp's cache (`CcCopyWrite`, no user-mode call), and other
  readers would see the change until the cache is purged.
- Caching: `FileInfoTimeout` is `u32::MAX`, which turns on the Windows cache manager (contents never
  change). `FlushAndPurgeOnCleanup` is on: the cache of a file is purged when its last handle closes, so
  its Close arrives at once and an unmount can wait for it. The cost: a file read again after being
  closed comes back through the mount, usually from the core `ReadCache`.
- Concurrency: callbacks run on several dispatcher threads; `MountFs` is `Sync`, `Volume` state is one
  short mutex. The file context is the node id. Every callback body runs under `catch_unwind`.
- Unmount: removing the mount point detaches at once; files still open, mappings included, are served
  until closed and unmapped; then the dispatcher stops and the volume is deleted. The wait is for
  WinFsp's Close of every file object, which comes once the system lets the object go: about 3 s after
  the last handle of a cached read, but for a file that was memory-mapped never after the unmap, since
  the system keeps its cache section and tells the file system nothing (measured on WinFsp 2.1). So the
  waiting side reopens and closes each file still counted open about once a second (`poke.rs`): that
  makes the last cleanup happen again, whose purge (`FlushAndPurgeOnCleanup`) fails harmlessly while a
  mapping lives and, once it is gone, frees the section so the Close arrives. The reopen goes through
  `\\?\GLOBALROOT` + the volume's device name (reachable after the mount point is gone), asks only
  `FILE_READ_ATTRIBUTES`, shares everything, and is never called from a dispatcher thread or under the
  volume's mutex. The device name is used only in the shape WinFsp gives it (`\Device\Volume{...}`,
  NUL-terminated); anything else disables the reopening with one warning, so a reopen can never reach a
  path outside the volume. The foreground loop reopens on the thread that reads the console wake-ups,
  so a second Ctrl-C pressed while a reopen is pending is acted on when it returns: at once normally, at
  worst after the driver's IRP timeout when every dispatcher thread is busy. Only files are waited for, not directories (a shell's current directory or an
  Explorer window would otherwise hold the unmount). `MountHandle` drop does the same.
- Foreground: `UnmountSignals::install` once per process. First Ctrl-C or Ctrl-Break detaches, the last
  close ends with `Unmounted` (re-checked: a file opened since the wake-up keeps the loop waiting, and
  the final stop never waits, so a second key is always heard); a second key returns `Abandoned` with
  the volume leaked, and the caller exits with 130 (Ctrl-C) or 149 (Ctrl-Break). A volume the driver
  stops on its own (a dispatcher thread that failed, such as one that could not be created, or WinFsp
  uninstalled) ends with `Stopped { status }`, the dispatcher's `NTSTATUS`, exit code 1: a failure,
  never reported as an unmount. `MountHandle::unmount` reports that stop as `MountError::Stopped`, and
  `MountHandle::wait`, which listens on the volume's wake-up channel for the duration, returns
  `MountEnd::Stopped` (nothing outside the process unmounts a WinFsp volume). Closing the console,
  logoff and shutdown use the default handler: the process ends, its volume handle closes and WinFsp
  removes the drive letter or the junction.

## Editing map

- A field or flag of the volume: `names.rs` (`volume_params`) and, for a new bit, `abi.rs` with an assert.
- What a call answers or refuses: `callbacks.rs`; the rule itself in `names.rs` with a test.
- Who may open the files: `security.rs` (`sddl`).
- Mount point forms, path-length check: `names.rs`; their errors: `error.rs`.
- Unmount and console behaviour: `mod.rs`, `console.rs` (`step`), `volume.rs`, `poke.rs`.
- What counts as a mount point in use: `mod.rs` (`check_free`, `drive_in_use`) and `names.rs`
  (`check_absolute`), with a test per refusal.
- Loading WinFsp: `dll.rs`.
