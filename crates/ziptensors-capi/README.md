# ziptensors-capi

The stable C ABI of [ziptensors](https://github.com/Vasyanator/ziptensors): a shared library
(`libziptensors_capi.so`, `ziptensors_capi.dll`, `libziptensors_capi.dylib`) and the header
[`include/ziptensors.h`](include/ziptensors.h). It is not published on crates.io; Rust code uses
[`ziptensors-core`](https://crates.io/crates/ziptensors-core) directly.

Prebuilt libraries and the header are attached to each
[GitHub release](https://github.com/Vasyanator/ziptensors/releases). To build from a checkout (needs a C
compiler, because `zstd-sys` compiles zstd):

```text
cargo build --release -p ziptensors-capi
```

The header documents every function. In short: `zt_open` opens a container with `zt_open_options`
(set `max_total_decoded` for untrusted input), `zt_member_open` and `zt_read_at` read, `zt_encode_file`
and `zt_pack_new` write, every call returns a `zt_status` and `zt_last_error` describes the last failure on the
calling thread. `zt_member_info` and `zt_member_verify` describe and check a member, `zt_restore_all`
restores a whole container, `zt_edit` adds, removes and replaces members (rewrite, append or in place)
and `zt_recover` repairs a torn append. `zt_enable_stderr_logging` turns on log output.
`ZT_ABI_VERSION` and `zt_abi_version` identify the ABI.

Note: in every C options struct `max_total_decoded = 0` means no budget, while the Rust, Python and
command line interfaces take 0 literally.

Not in the C ABI, by design: mounting, decoding a member into a callback, and the per-file tree
commands. Use the command line tool or the Rust or Python API for those.

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option; the texts are in
the repository root.
