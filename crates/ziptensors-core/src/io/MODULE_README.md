# Module: crates/ziptensors-core/src/io

## Purpose

Exact positional reads from a source of fixed length, which everything that reads a file goes through, and
the temporary file that everything writing a final file commits through.

## Files and submodules

| Path | Role |
|---|---|
| `mod.rs` | Re-exports. |
| `source.rs` | `RandomAccess` trait, `FileSource` (`pread`, `seek_read`). |
| `temp.rs` | `TempFile`, crate-private: temporary file beside an output, sync, close, rename. `same_file`. Public `TEMP_FILE_SUFFIX`. |

## Contracts and invariants

- `read_exact_at` either fills the whole buffer or fails. A range outside `len` is `OutOfBounds`, checked
  before any I/O and with overflow handled. A file that shrank after open yields `Io` with
  `UnexpectedEof`.
- `len` is fixed when the source is opened.
- Reads take `&self` and are safe from concurrent threads with no lock: no seek state is used.
- `same_file` is the one check that an output is not an input, by canonical path and on Unix by inode.
- The trait is object safe, so parsers take `&dyn RandomAccess`, and test doubles can count I/O.
- `TempFile` is the single owner of the commit of spec invariant 12: same directory as the output, sync,
  handle closed, rename, and on Unix a sync of the directory. Unless committed it is removed on drop, handle first, on every failure path.
- A commit fails in one of two ways. `Error::Io` (sync or rename) leaves the output untouched.
  `Error::CommittedNotDurable` is the directory sync failing after the rename: the new file is in
  place, its durability is not confirmed. The post-rename step is a parameter of the private
  `commit_then`, the seam through which tests inject that failure.
  `write` wraps it in `Sink`, `read` uses it for `restore_file`. Neither renames a file on its own.
- Every temporary file is named `<output name>.<pid>-<counter>` + `TEMP_FILE_SUFFIX` (public, `.ziptensors-tmp`),
  so tests and users can tell ours apart; one found later is a crash leftover and safe to delete.

## Editing map

- New platform: add a `read_exact_os` for it in `source.rs`; other targets are a `compile_error!`.
