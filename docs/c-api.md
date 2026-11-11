# C API guide

The C ABI is a shared library (`libziptensors_capi.so` on Linux, `ziptensors_capi.dll` on Windows,
`libziptensors_capi.dylib` on macOS) with one header, `ziptensors.h`. It covers opening and reading
containers, verifying and restoring members, writing containers, editing them and recovering a torn
file. The header documents every function; this page is the task-oriented overview.

Not in the C ABI, by design: mounting, decoding a member into a callback, and the per-file directory
tree commands. Use the [command line tool](cli.md), the [Rust library](rust.md) or the
[Python module](python.md) for those.

## Getting the library

The C library is not published on crates.io. Two ways to get it:

- **Prebuilt, from a GitHub release.** Each release on
  <https://github.com/Vasyanator/ziptensors/releases> has one archive per platform named
  `ziptensors-capi-<version>-<target>`, a `.tar.gz` on Linux and macOS and a `.zip` on Windows. Each
  holds `include/ziptensors.h`, the library under `lib/`, the README and the license texts.

  | Target | Files in `lib/` |
  |---|---|
  | `x86_64-unknown-linux-gnu` | `libziptensors_capi.so` |
  | `x86_64-pc-windows-msvc` | `ziptensors_capi.dll`, `ziptensors_capi.dll.lib` (import library) |
  | `aarch64-apple-darwin`, `x86_64-apple-darwin` | `libziptensors_capi.dylib` |

- **From a checkout.** Needs Rust and a C compiler (the zstd dependency is compiled from C):

  ```text
  cargo build --release -p ziptensors-capi
  ```

  The library is then in `target/release/` and the header in `crates/ziptensors-capi/include/`.

The library bundles its own zstd and links no system library besides the platform's C runtime libraries.

### Compiling and linking

On Linux, with the archive unpacked to `ziptensors-capi/`:

```text
cc -std=c11 -Wall -I ziptensors-capi/include app.c \
   -L ziptensors-capi/lib -lziptensors_capi -Wl,-rpath,'$ORIGIN/ziptensors-capi/lib' -o app
```

The `-rpath` lets `app` find the library at run time relative to its own location; alternatively
install the library where the dynamic loader looks. On Windows with MSVC, link against
`ziptensors_capi.dll.lib` and ship `ziptensors_capi.dll` next to your executable. On macOS, link
`libziptensors_capi.dylib` and make sure the loader finds it at run time (for example through an
rpath).

The header is plain C (C99 or later) and also works from C++ (it wraps the declarations in
`extern "C"`).

## Example

A complete program: encode a file, open the container with a decode budget, read the first tensor,
verify the member and restore it.

```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "ziptensors.h"

/* Prints the failed call and the library's description of the failure. */
static int fail(const char *what, zt_status status) {
    char text[1024];
    size_t len = zt_last_error(text, sizeof text);
    fprintf(stderr, "%s failed with status %d: %s%s\n", what, (int)status, text,
            len >= sizeof text ? " [cut]" : "");
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <input> <container> <restored>\n", argv[0]);
        return 2;
    }
    if (zt_abi_version() != ZT_ABI_VERSION) {
        fprintf(stderr, "library ABI %u, header ABI %u\n", zt_abi_version(), ZT_ABI_VERSION);
        return 1;
    }
    if (zt_enable_stderr_logging(ZT_LOG_LEVEL_WARN) != ZT_STATUS_OK) {
        fprintf(stderr, "library logging not enabled: another tracing subscriber is installed\n");
    }

    zt_encode_options enc = {0};
    enc.struct_size = sizeof enc;
    enc.mode = ZT_MODE_BALANCED;
    zt_status status = zt_encode_file(argv[1], argv[2], &enc);
    if (status != ZT_STATUS_OK) return fail("zt_encode_file", status);

    zt_open_options opts = {0};
    opts.struct_size = sizeof opts;
    opts.max_total_decoded = (uint64_t)64 << 30; /* 64 GiB; 0 would mean no budget */
    zt_container *container = NULL;
    status = zt_open(argv[2], &opts, &container);
    if (status != ZT_STATUS_OK) return fail("zt_open", status);

    zt_member *member = NULL;
    status = zt_member_open(container, NULL, &member); /* NULL: the only member */
    if (status != ZT_STATUS_OK) {
        zt_close(container);
        return fail("zt_member_open", status);
    }
    zt_close(container); /* the member keeps the container's data alive */

    size_t tensors = 0;
    status = zt_tensor_count(member, &tensors);
    if (status != ZT_STATUS_OK) {
        zt_member_close(member);
        return fail("zt_tensor_count", status);
    }
    if (tensors > 0) {
        zt_tensor_desc desc;
        status = zt_tensor_info(member, 0, &desc);
        if (status != ZT_STATUS_OK) {
            zt_member_close(member);
            return fail("zt_tensor_info", status);
        }
        uint8_t *bytes = malloc(desc.len > 0 ? (size_t)desc.len : 1);
        if (bytes == NULL) {
            zt_member_close(member);
            return 1;
        }
        status = zt_read_at(member, desc.offset, bytes, (size_t)desc.len);
        if (status != ZT_STATUS_OK) {
            free(bytes);
            zt_member_close(member);
            return fail("zt_read_at", status);
        }
        printf("tensor %.*s: %llu bytes, dtype %u, %zu dims\n", (int)desc.name_len, desc.name,
               (unsigned long long)desc.len, (unsigned)desc.dtype, desc.ndim);
        free(bytes);
    }

    status = zt_member_verify(member);
    if (status == ZT_STATUS_OK) status = zt_restore_file(member, argv[3]);
    zt_member_close(member);
    if (status != ZT_STATUS_OK) return fail("verify or restore", status);
    printf("restored to %s\n", argv[3]);
    return 0;
}
```

Run as `./app model.safetensors model.ziptensors restored.safetensors`. The `(size_t)desc.len`
casts assume a 64-bit target; on a 32-bit one, check `desc.len <= SIZE_MAX` first.

## Conventions

### Status codes and `zt_last_error`

Every function that can fail returns a `zt_status`: `ZT_STATUS_OK` (0) on success, a positive value
for a failure of the library, a negative value for a call refused at the ABI (bad arguments). After
any non-zero status, `zt_last_error(buf, cap)` copies a readable description:

- It is **per thread** and describes the last status-returning call on the calling thread; a
  successful call clears it (the function then returns 0).
- It returns the full length of the text without the NUL. With `cap > 0` it writes at most
  `cap - 1` bytes and a NUL, cut on a UTF-8 boundary, so the text was cut exactly when the result is
  `>= cap`. With `buf == NULL` or `cap == 0` it only returns the length.
- The text is a copy: it stays valid whatever handles are closed.

### Options structs and `struct_size`

Every options struct (`zt_open_options`, `zt_encode_options`, `zt_edit_options`,
`zt_recover_options`) and the output struct `zt_member_desc` starts with `size_t struct_size`. Zero
the struct, set `struct_size = sizeof(the struct)`, then set the fields you need: a field left 0 takes
its default. The library accepts exactly the size of its own build; a different size is refused with
`ZT_STATUS_STRUCT_SIZE` and nothing is read or written. A `NULL` options pointer means all defaults.

Enum and flag fields are `int32_t` and validated: an unknown enum value, or a flag other than 0 or 1,
is refused with `ZT_STATUS_INVALID_ARGUMENT`.

**`max_total_decoded = 0` means NO budget** in every C options struct. This differs from the Rust,
Python and command line interfaces, where 0 is a budget of zero bytes. For untrusted input, set a real
budget. See [safety.md](safety.md).

### ABI versioning

`ZT_ABI_VERSION` in the header (currently 1) and `zt_abi_version()` at run time identify the ABI.
Compare them at startup, as the example does. The ABI version is raised whenever a declaration of the
header changes in a way that breaks callers built against an older header. It is independent of the
library's release version and of the container format version. `zt_edit_op` has no `struct_size`, so
any change to it is such a break.

### Strings

Strings passed in are NUL-terminated UTF-8 and are read only during the call (a non-UTF-8 string is
refused with `ZT_STATUS_INVALID_UTF8`). Member names are `/`-separated. Strings returned by the
library are borrowed (see the next section) and come with a length: names come from the file, can
contain a NUL, and the length is the authority on where they end.

## Handles: lifetime, ownership, threads

| Handle | Created by | Freed by | Lifetime rules | Threads |
|---|---|---|---|---|
| `zt_container` | `zt_open` | `zt_close` | Member handles opened from it stay usable after `zt_close`. Names from `zt_member_name` are valid until `zt_close` of this handle. | Every function taking `const zt_container *` may be called from any number of threads at once. |
| `zt_member` | `zt_member_open` | `zt_member_close` | Keeps its container's data alive, so it may outlive `zt_close`. Strings and shapes in a `zt_tensor_desc` are valid until `zt_member_close` of the member they came from, and no longer. | Every function taking `const zt_member *` may be called from any number of threads at once. |
| `zt_pack` | `zt_pack_new` | `zt_pack_free` | Always free it, finished or not. Freeing an unfinished pack removes its temporary file and leaves no output. After `zt_pack_finish` (successful or not) only `zt_pack_free` is useful. | One call at a time: the handle may move between threads but must not be used by two at once. |

General rules:

- On any failure, an out-pointer for a new handle is set to `NULL`; other output slots are left
  untouched.
- `zt_close(NULL)`, `zt_member_close(NULL)` and `zt_pack_free(NULL)` are no-ops. Closing a handle
  twice, using it afterwards, or closing it while another thread is using it, is undefined behavior.
- Destination buffers (`zt_read_at`, `zt_decode_tensor_into`) may be uninitialized; they are only
  written. A call refused by its arguments writes nothing; after a failure during decoding the buffer
  holds unspecified bytes.
- Close every `zt_container` open on a file before a `ZT_STRATEGY_REWRITE` or
  `ZT_STRATEGY_IN_PLACE` edit of it.
- If a call returns `ZT_STATUS_PANIC`, the library hit an internal bug; the call did not complete,
  and handles it used should be closed. Panics never cross into your program.

## Functions by task

### Version and diagnostics

| Function | Description |
|---|---|
| `uint32_t zt_abi_version(void)` | ABI version of the loaded library; compare with `ZT_ABI_VERSION`. |
| `size_t zt_last_error(char *buf, size_t cap)` | Text of the last failure on this thread (see above). |
| `zt_status zt_enable_stderr_logging(int level)` | Turns on log output to stderr (see [Logging](#logging)). |

### Opening and inspecting

| Function | Description |
|---|---|
| `zt_open(path, options, &container)` | Opens a container, recognized by its magic, not its name. Reads the header, footer and index, no data. |
| `zt_close(container)` | Frees a container handle. |
| `zt_member_count(container, &count)` | Number of members. |
| `zt_member_name(container, index, &name, &name_len)` | Name of member `index` in file order. Borrowed until `zt_close`. Untrusted: check it before joining it to a directory. |
| `zt_member_info(container, index, &desc)` | Fills a `zt_member_desc`: `len`, `blake3[32]` (the stated hash), `profile` (`ZT_PROFILE_*`, a hint), `tensor_count`. Set `desc.struct_size` first. Decodes nothing. |
| `zt_member_open(container, name, &member)` | Opens a member by name, or with `name == NULL` the only member of a single-file container (`ZT_STATUS_MEMBER_NOT_FOUND` otherwise). |
| `zt_member_close(member)` | Frees a member handle and every string and shape returned from it. |

`zt_open_options` fields (all 0 = default):

| Field | Default (0) | Meaning |
|---|---|---|
| `threads` | one per logical CPU | Decode threads and scratch pool size; 1 runs everything on the calling thread. |
| `max_index_bytes` | 256 MiB | Cap on the decompressed index; values over 1 GiB act as 1 GiB. |
| `max_total_decoded` | **no budget** | Budget on the sum of member lengths; a container over it is refused at open. Set it for untrusted input. |
| `max_decode_amplification` | 16 | Bound on decode work as a multiple of the sum of member lengths, on top of a 256 MiB floor. |
| `verify` | `ZT_VERIFY_CHUNK` | `ZT_VERIFY_CHUNK` checks every decoded chunk's hash; `ZT_VERIFY_OFF` skips it (lengths still enforced). |

### Reading

| Function | Description |
|---|---|
| `zt_member_len(member, &len)` | Length of the member in bytes. |
| `zt_read_at(member, offset, dst, len)` | Fills `dst[0..len]` from byte `offset`. `ZT_STATUS_OUT_OF_BOUNDS` (nothing written) if the range is outside the member. With `len == 0`, `dst` may be NULL. |
| `zt_tensor_count(member, &count)` | Number of tensors in the member's table. |
| `zt_tensor_info(member, index, &desc)` | Fills a `zt_tensor_desc`: `name`/`name_len`, `offset`, `len`, `shape`/`ndim` (at most 16), `dtype` (`ZT_DTYPE_*`, the format's wire id; unknown ids are legal). Pointers are borrowed until `zt_member_close`. The table is metadata: restoring a member never consults it. |
| `zt_decode_tensor_into(member, name, dst, len)` | Fills `dst` with one tensor; `len` must equal the tensor's length (`ZT_STATUS_DESTINATION_SIZE` otherwise). |

A read of part of a chunk decodes the whole chunk, and nothing is cached between calls: prefer few
large reads over many small ones.

### Verifying and restoring

| Function | Description |
|---|---|
| `zt_member_verify(member)` | Decodes the whole member and compares its BLAKE3 with the stated hash. Writes nothing. `ZT_STATUS_VERIFY_FAILED` if it differs. |
| `zt_restore_file(member, path)` | Writes the member to `path` through a temporary file: the bytes appear only after they hashed to the member's BLAKE3 and were synced, as one rename (replacing a file already there). On failure nothing new is at `path` and no temporary file remains, except `ZT_STATUS_COMMITTED_NOT_DURABLE`. |
| `zt_restore_all(container, dir, &bytes)` | Restores every member to `dir` joined with its name, creating directories. Every name is checked first: one unsafe name (`..`, absolute path, drive letter) returns `ZT_STATUS_UNSAFE_MEMBER_NAME` and creates nothing, not even `dir`. After that, a failure stops the run and keeps the members restored before it. |

Restore writes content only: modification times, permissions and links are not restored.

### Writing

| Function | Description |
|---|---|
| `zt_encode_file(input, output, options)` | Encodes one file as a container of one member named after the file. Default mode `ZT_MODE_BALANCED`. The output is fully decoded and hashed against the input before it appears. |
| `zt_pack_new(output, options, &pack)` | Starts a multi-member container. Default mode `ZT_MODE_MAX`. |
| `zt_pack_add_file(pack, path, member)` | Adds a file as member `member` (`/`-separated, must be a name a restore accepts, unique in the pack). Symbolic links are refused. On failure the member is not added and the pack stays usable, unless writing the output failed. |
| `zt_pack_add_file_delta(pack, path, member, base_member)` | As `zt_pack_add_file`, and stores tensors that pair with tensors of `base_member` (same name, length and dtype) as delta blobs where smaller. Only under `ZT_MODE_MAX`. `base_member` must have been added from a file by this pack, and that file must be unchanged. Readers need format 0.3 or later. |
| `zt_pack_add_member_from(pack, source, member)` | Copies a member of an open container into the pack with no decoding or recompression; it is proven at `zt_pack_finish`. |
| `zt_pack_finish(pack)` | Decodes everything, hashes it against the inputs, syncs, and renames to the output path. Works once. |
| `zt_pack_free(pack)` | Frees the pack. |

`zt_encode_options` fields: `threads` (0 = one per CPU; the output bytes do not depend on it) and
`mode`:

| `zt_mode` | Value | Meaning |
|---|---|---|
| `ZT_MODE_DEFAULT` | 0 | Balanced for `zt_encode_file`, max for `zt_pack_new`. |
| `ZT_MODE_FAST` | 1 | Plane split and entropy coding only; quantized data is stored. |
| `ZT_MODE_BALANCED` | 2 | Adds widening views and field split of quantized data. |
| `ZT_MODE_MAX` | 3 | Adds the bit rotation trial and blob deduplication. |

Nothing appears at the output before `zt_encode_file` or `zt_pack_finish` succeeds (or returns
`ZT_STATUS_COMMITTED_NOT_DURABLE`). `ZT_STATUS_INPUT_CHANGED` means an input changed while it was
read, and nothing was committed.

### Editing

`zt_status zt_edit(const char *container, const zt_edit_op *ops, size_t count, const zt_edit_options *options)`
adds, removes and replaces members. `count == 0` compacts the container (drops the dead data of
earlier edits). See [editing.md](editing.md) for the strategies in depth.

`zt_edit_op` (no `struct_size`; a zeroed struct is refused):

| Field | Meaning |
|---|---|
| `kind` | `ZT_EDIT_ADD` (1), `ZT_EDIT_REMOVE` (2) or `ZT_EDIT_REPLACE` (3). |
| `name` | The member. For an add, a new name a restore accepts. |
| `path` | The input file for an add or replace; NULL for a remove. A regular file, not a link, not the container. |
| `delta_base` | NULL, or for an add or replace a member of the result to store the file against: a kept member, or a file added or replaced earlier in the array. |

Each member name appears in at most one operation. Kept members keep their order, a replace keeps
its slot, adds follow in array order.

`zt_edit_options` (all 0 = default):

| Field | Default (0) | Meaning |
|---|---|---|
| `strategy` | `ZT_STRATEGY_REWRITE` | See below. |
| `repack` | 0 | 1 re-encodes every kept member with `mode`. Only with `ZT_STRATEGY_REWRITE`. |
| `mode` | `ZT_MODE_MAX` | Mode of new files (and of every member under `repack`). |
| `threads` | one per CPU | Compression, verification and decode threads. |
| `max_total_decoded` | **no budget** | Budget on the container being edited. |

| `zt_strategy` | Behavior |
|---|---|
| `ZT_STRATEGY_REWRITE` (0) | Writes the result to a temporary file, proves it, renames it over the container. Needs room for a second copy; the container is intact after any failure except `ZT_STATUS_COMMITTED_NOT_DURABLE`, which means the new container already replaced it (see [editing.md](editing.md#default-rewrite-through-a-temporary-file)). |
| `ZT_STRATEGY_APPEND` (1) | Writes only new data, a new index and a new footer after the end. A failure cuts the file back, or returns `ZT_STATUS_TRAILING_BYTES` if it cannot (then run `zt_recover`). Refused with `repack` and with no operation. |
| `ZT_STRATEGY_IN_PLACE` (2) | **Risky.** Rewrites the file in place with no extra space. A failure after the first write returns `ZT_STATUS_DAMAGED_IN_PLACE`: the container is damaged. Keep a backup. Nothing may read the file meanwhile. Refused with `repack` and with a `delta_base` that is a kept member. |

Every check runs before a byte is written: after `ZT_STATUS_INVALID_ARGUMENT`,
`ZT_STATUS_NULL_ARGUMENT`, `ZT_STATUS_INVALID_UTF8` (the error text names the operation's index),
`ZT_STATUS_EDIT_REFUSED`, `ZT_STATUS_MEMBER_NOT_FOUND`, `ZT_STATUS_UNSAFE_MEMBER_NAME`,
`ZT_STATUS_OUTPUT_IS_INPUT`, `ZT_STATUS_SYMLINK_INPUT` and the statuses of `zt_open`, the file is
untouched.

### Recovering

`zt_status zt_recover(const char *container, const zt_recover_options *options, uint64_t *valid_len_out)`
shortens a file whose end does not close a valid container (an interrupted append, or after
`ZT_STATUS_TRAILING_BYTES`) to its latest valid generation. No reader does this on its own. On
success `*valid_len_out` (if not NULL) is the valid length: the file's length if it opened as it was,
otherwise the length it was (or on a dry run would be) cut to.

`zt_recover_options` (all 0 = default):

| Field | Default (0) | Meaning |
|---|---|---|
| `dry_run` | 0 | 1 reports and writes nothing, opening the file read-only. |
| `skip_verify` | 0 | 0 decodes and checks every member of the found generation before cutting, passing over one that fails. 1 skips that; only for a file never edited in place. |
| `max_candidates` | 64 | Most candidate footers given a full check. |
| `max_scan_bytes` | no bound | Most distinct bytes the backward scan reads. |
| `max_total_decoded` | **no budget** | Budget per candidate. |

Every failure leaves the file unchanged, except `ZT_STATUS_IO` from the sync after the cut (the file
is cut, but the cut may not survive a power loss).

## Status codes

| Value | Name | Meaning | What to do |
|---|---|---|---|
| 0 | `ZT_STATUS_OK` | Success. | |
| 1 | `ZT_STATUS_IO` | A file operation failed. | Check the path, permissions and free space; `zt_last_error` names the operation and file. |
| 2 | `ZT_STATUS_NOT_ZIPTENSORS` | Not a ziptensors container. | Check the file. |
| 3 | `ZT_STATUS_UNSUPPORTED_VERSION` | Format version this library does not read. | Use a newer library. |
| 4 | `ZT_STATUS_UNSUPPORTED_FEATURE` | A required feature this library does not implement. | Use a newer library. |
| 5 | `ZT_STATUS_UNKNOWN_CODEC` | Unknown transform, view or stream codec. | Use a newer library. |
| 6 | `ZT_STATUS_MALFORMED_INDEX` | Header, footer or index violates the format. | The file is damaged or hostile; after a torn append, try `zt_recover`. |
| 7 | `ZT_STATUS_BUDGET_EXCEEDED` | A declared size is over a limit of `zt_open_options`. Raised before data is read. | Raise the limit only for a file you trust. |
| 8 | `ZT_STATUS_CHUNK_CORRUPT` | A chunk cannot be decoded. | The file is damaged. |
| 9 | `ZT_STATUS_CHECKSUM_MISMATCH` | A chunk decoded to bytes that do not match its hash. | The file is damaged. |
| 10 | `ZT_STATUS_OUT_OF_BOUNDS` | The range does not lie in the member. | Fix offset or length. |
| 11 | `ZT_STATUS_DESTINATION_SIZE` | The destination has the wrong length. | Use the tensor's `len`. |
| 12 | `ZT_STATUS_TENSOR_NOT_FOUND` | No tensor of this name. | Check the tensor table. |
| 13 | `ZT_STATUS_MEMBER_NOT_FOUND` | No such member, or not exactly one when a single member was asked for. | Check the member names. |
| 14 | `ZT_STATUS_UNSAFE_MEMBER_NAME` | A member name is not safe to join to an output directory. Nothing was written for it. | Rename the input; do not restore such a container into a directory. |
| 15 | `ZT_STATUS_VERIFY_FAILED` | A member's bytes do not hash to its stated value. | The file is damaged. |
| 16 | `ZT_STATUS_INPUT_CHANGED` | An input changed while it was encoded. No output was committed. | Retry when nothing writes the input. |
| 17 | `ZT_STATUS_ZSTD` | zstd reported an error. | Report it. |
| 18 | `ZT_STATUS_SYMLINK_INPUT` | An input is a symbolic link; links are not followed. | Pass the target. |
| 19 | `ZT_STATUS_OUTPUT_IS_INPUT` | An output path is the file being read. | Choose another output. |
| 20 | `ZT_STATUS_COMMITTED_NOT_DURABLE` | The output WAS written, but its directory could not be synced: a power loss may bring back the previous file. Unix only. | Usually nothing; the output is complete and verified. |
| 21 | `ZT_STATUS_DAMAGED_IN_PLACE` | An in-place edit failed after it began moving data: the container is damaged. | Restore from a backup or repack. |
| 22 | `ZT_STATUS_EDIT_REFUSED` | The edit cannot be applied as asked. Nothing was written. | Change the request, usually the strategy. |
| 23 | `ZT_STATUS_TRAILING_BYTES` | A valid container followed by bytes that could not be removed; the file does not open. | Run `zt_recover`. |
| 24 | `ZT_STATUS_UNRECOVERABLE` | Recovery found no valid container. Nothing was changed. | Restore from a backup. |
| 25 | `ZT_STATUS_TREE_CONFLICT` | A tree operation found colliding names. No C function returns it, since the tree functions are not in the C ABI; it exists so that the codes match the Rust `ErrorCode`. | |
| 26 | `ZT_STATUS_DUPLICATE_MEMBER` | A pack already has a member of this name (`zt_pack_add_file`, `zt_pack_add_file_delta`, `zt_pack_add_member_from`). That member was not added; the pack stays usable. | Add it under another name. |
| -1 | `ZT_STATUS_NULL_ARGUMENT` | A required pointer was NULL. | Fix the call. |
| -2 | `ZT_STATUS_INVALID_UTF8` | A string argument is not valid UTF-8. | Pass UTF-8. |
| -3 | `ZT_STATUS_INVALID_ARGUMENT` | A value the call does not accept: unknown enum value, index past the end, length over `PTRDIFF_MAX`. | Fix the call. |
| -4 | `ZT_STATUS_STRUCT_SIZE` | `struct_size` is not the size this library knows: built against another header. | Rebuild against the library's header. |
| -5 | `ZT_STATUS_INVALID_STATE` | Not possible in this state: a `zt_pack` already finished or that panicked; `zt_enable_stderr_logging` when another logger is installed. | Free the pack; see [Logging](#logging). |
| -6 | `ZT_STATUS_PANIC` | Internal library bug. The call did not complete. | Close the handles it used and report it. |

Status values never change and are never reused. The positive values equal the Rust `ErrorCode`
values (see [rust.md](rust.md#errors)).

## Logging

The library is silent until you call `zt_enable_stderr_logging(level)` with one of:

| `zt_log_level` | Value | Writes |
|---|---|---|
| `ZT_LOG_LEVEL_ERROR` | 1 | Failures only. |
| `ZT_LOG_LEVEL_WARN` | 2 | Also refused calls: every failing entry point logs one line. |
| `ZT_LOG_LEVEL_INFO` | 3 | Also the stages of long operations: open, encode, commit, edit, recover. |
| `ZT_LOG_LEVEL_DEBUG` | 4 | Also details of decisions inside operations. |
| `ZT_LOG_LEVEL_TRACE` | 5 | Everything. |

Contract:

- It applies to the rest of the process. Only the first valid call installs anything; later calls
  change nothing (a different level included) and return what the first one returned.
- `ZT_STATUS_INVALID_ARGUMENT` for a value that is not a `zt_log_level`.
- `ZT_STATUS_INVALID_STATE` if the process already has a global `tracing` subscriber that this
  library did not install (for example another Rust library in the same process): the library's
  lines then go to that subscriber, if anywhere. A `log` logger of the host does not matter.
