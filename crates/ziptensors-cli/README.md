# ziptensors-cli

The `ziptensors` command line tool for [ziptensors](https://github.com/Vasyanator/ziptensors) containers:
lossless, compressed storage for tensor files (safetensors, GGUF, ONNX, PyTorch zip and opaque files)
with random access. Every restored file is bit-exact and proven by its BLAKE3 hash.

```text
cargo install ziptensors-cli
```

Prebuilt binaries for Linux, Windows and macOS are attached to each
[GitHub release](https://github.com/Vasyanator/ziptensors/releases). Building from source needs a C
compiler, because `zstd-sys` compiles zstd.

## Commands

```text
ziptensors encode <INPUT> <OUTPUT> [--mode fast|balanced|max] [--threads N]
ziptensors pack <OUTPUT> <INPUT>... [--delta-base FILE] [--mode M] [--threads N]
ziptensors restore <CONTAINER> <OUTPUT> [--max-total-decoded BYTES]
ziptensors restore <CONTAINER> --all <DIR> [--max-total-decoded BYTES]
ziptensors extract <CONTAINER> <MEMBER> <OUTPUT> [--max-total-decoded BYTES]
ziptensors merge <OUTPUT> <INPUT>...
ziptensors inspect <CONTAINER> [--json]
ziptensors verify <CONTAINER>
ziptensors mount <CONTAINER> <MOUNTPOINT> [--cache-size BYTES] [--max-total-decoded BYTES]
```

`ziptensors <command> --help` lists every option. Pass `--max-total-decoded` when the container comes
from somewhere else: without it there is no decode budget. The hashes detect damage; they do not
authenticate a file.

## Mounting

`mount` shows the members of a container as read-only files. On Linux it uses FUSE (`/dev/fuse` and
`fusermount3`). On Windows it uses WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos,
<https://github.com/winfsp/winfsp>, which must be installed from <https://winfsp.dev/rel/>. macOS has no
`mount` command.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at
your option.
