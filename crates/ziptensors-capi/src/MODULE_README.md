# Module: crates/ziptensors-capi/src

## Purpose

The stable C ABI over `ziptensors-core`, shipped as a cdylib with a generated `include/ziptensors.h`.
Holds argument checking and translation only; every decision stays in core.

## Architecture

An entry point is a closure in `barrier::guarded`: `args` and `options` turn raw arguments into Rust
values, one core call does the work, and `status::Failure` carries a refusal or a core `Error` back to
the barrier, which stores its text and returns the `ZtStatus`.

## Files and submodules

| Path | Role |
|---|---|
| `lib.rs` | Crate root, lint policy, `ZT_ABI_VERSION`, `zt_abi_version`. |
| `status.rs` | `ZtStatus` with the exhaustive map from `ErrorCode`, and `Failure`. |
| `barrier.rs` | `guarded`, `guarded_value`: the only `catch_unwind`. The thread-local text, `zt_last_error`. |
| `args.rs` | Null, UTF-8 and length checks, handle borrows, output slots, destination buffers. |
| `options.rs` | `zt_open_options`, `zt_encode_options`, `zt_mode`, `zt_verify`, the `struct_size` rule. |
| `read.rs` | `zt_container`, `zt_member`, `zt_tensor_desc` and the entry points over them. |
| `dtype.rs` | `ZT_DTYPE_*`, the dtype wire ids as header constants. |
| `write.rs` | `zt_encode_file`, `zt_pack` and the builder entry points. |
| `tests.rs` | Every entry through `extern "C"`. Lives here for the panic injection of `barrier.rs`. |
| `../tests/header.rs` | Proves the checked-in header is what cbindgen generates. |
| `../tests/c_smoke.rs`, `../tests/c/smoke.c` | A C program built with `cc`, linked to the real library. |
| `../cbindgen.toml` | Header configuration: Rust `ZtFoo` names become C `zt_foo`. |

## Contracts and invariants

- Handles are opaque. `*_close(NULL)` and `zt_pack_free(NULL)` are no-ops. `const` handles may be used
  from any number of threads; a `zt_pack` by one thread at a time. A `zt_member` keeps its container's
  data alive, so the two close in any order.
- `zt_status` is 0 for success, positive for `ErrorCode` values of the same number, negative for
  refusals of this crate. Values never change. `from_code` is exhaustive, so a new `ErrorCode` stops this
  crate from building until it has a status.
- Every entry point runs in a barrier, including the ones that cannot fail, and nothing that can panic
  runs outside it. A panic reaching `extern "C"` would abort the host.
- `zt_last_error` is thread-local and describes the last status-returning call on the thread: a success
  clears it. It reports the full length, cuts on a UTF-8 boundary, always terminates, and shows a NUL in
  the text as `\0`.
- Strings in `zt_tensor_desc` are borrowed from the `zt_member` that returned them and end at its
  `zt_member_close`, and a name from `zt_member_name` from the `zt_container` until its `zt_close`. Strings
  passed in are read during the call only.
- `ZT_DTYPE_*` are literals, as cbindgen does not evaluate core's mapping. A test pins them to `Dtype`.
- Memory the caller gives for output is written and never read: output slots through `ptr::write`, a
  `dst` by zeroing it through the raw pointer before a `&mut [u8]` exists, because core takes `&mut [u8]`
  and a slice must not cover uninitialized bytes. Range and size are checked first, so a refused call
  writes nothing. The cost is exactly one `memset` of the destination per `zt_read_at` and
  `zt_decode_tensor_into`, which is also where fresh pages get their first touch. It stays until core
  offers a read into uninitialized memory.
- An out-pointer of a handle is NULL after any failure, set before anything else can go wrong.
- Option structs start with `struct_size`, and only the exact size of this build is accepted: a smaller
  struct lacks fields, a larger one asks for what would be ignored. A field left 0 means its default.
  Their enum fields are `i32` and validated, since a Rust enum must not hold a value C made up.
- `zt_pack` gives up its builder for the length of a call, so a panic drops it instead of leaving it
  half updated. After `zt_pack_finish` the handle holds nothing and only `zt_pack_free` is useful.
- Shipped as a cdylib first: a staticlib consumer that also links libzstd risks duplicate symbols. The
  rlib in `Cargo.toml` is no product; it exists so that `cargo test` builds the cdylib for `c_smoke`.

## Editing map

- New entry point: add it in `read.rs` or `write.rs` inside `guarded`, export it in `lib.rs`, add it to
  the null and panic tests in `tests.rs`, and refresh the header.
- Refresh the header: `ZIPTENSORS_UPDATE_HEADER=1 cargo test -p ziptensors-capi --test header`, then
  review the diff. Doc comments here are the header's comments, so write them for a C reader.
- New `ErrorCode` in core: add the status with the same value in `status.rs` and in the list of the
  status test.
- A changed struct or signature breaks callers: raise `ZT_ABI_VERSION`. A new option field makes the old
  `struct_size` a size to keep accepting in `options::sized`.
