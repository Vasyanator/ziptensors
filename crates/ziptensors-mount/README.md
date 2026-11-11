# ziptensors-mount

A read-only filesystem view of a [ziptensors](https://github.com/Vasyanator/ziptensors) container: every
member appears as a regular file at its path, so programs that read a model directory (diffusers,
llama.cpp, ComfyUI, `safetensors` and `torch.load` with `mmap`) read a compressed container unmodified.

Most users want the `ziptensors mount` command of
[`ziptensors-cli`](https://crates.io/crates/ziptensors-cli). This crate is the library behind it: build a
`MountFs` over a `ziptensors_core::ReadCache` and hand it to the backend of your OS.

`MountFs::overlay` instead shows a whole directory read-only: each per-file container (one member,
written beside its original by `ziptensors_core::compress_tree`) appears under its original file name
with the decoded bytes, and every other file passes through unchanged. Containers open lazily, a bounded
number at a time. A damaged container or two containers presenting one name refuse the mount.

## Platforms

- **Linux:** FUSE through the pure-Rust `fuser` crate. Needs `/dev/fuse` and `fusermount3`; no libfuse.
- **Windows:** WinFsp, loaded at run time. WinFsp must be installed from <https://winfsp.dev/rel/>;
  nothing of it is needed to build.
- **macOS:** no backend. The name policies and the tree build there, but nothing mounts.

A container whose member names the OS cannot show (collisions, reserved names) refuses to mount, naming
every offending member.

## WinFsp notice

The Windows backend uses WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos,
<https://github.com/winfsp/winfsp>. WinFsp is licensed under GPL-3.0 with a FLOSS exception; a
proprietary program that embeds this crate on Windows needs WinFsp's commercial license. No WinFsp file
is distributed with this crate.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at
your option.
