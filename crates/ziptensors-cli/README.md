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
ziptensors edit <CONTAINER> [--add PATH]... [--add-as NAME PATH]... [--remove NAME]... [--replace NAME PATH]...
                [--delta-base MEMBER] [-F | -R] [-C] [--mode M]
ziptensors compact <CONTAINER> [-R] [-C] [--mode M]
ziptensors recover <CONTAINER> [--dry-run] [--no-verify]
ziptensors restore <CONTAINER> <OUTPUT>
ziptensors restore <CONTAINER> --all <DIR>
ziptensors extract <CONTAINER> <MEMBER> <OUTPUT|->
ziptensors merge <OUTPUT> <INPUT>...
ziptensors inspect <CONTAINER> [--json] [--tensors]
ziptensors verify <CONTAINER>
ziptensors compress [-r] <PATH>... [-k] [--follow-symlinks] [--dry-run] [--mode M]
ziptensors decompress [-r] <PATH>... [-k] [-f] [--dry-run]
ziptensors mount <CONTAINER> <MOUNTPOINT> [--cache-size BYTES]
```

`edit -F` appends and is safe to interrupt (`recover` restores the previous version); `edit -R` and
`compact -R` rewrite in place with no extra space, and an interruption damages the container.

`ziptensors <command> --help` lists every option. Every command that opens a container takes
`--max-total-decoded BYTES`: pass it when the container comes from somewhere else, since without it
there is no decode budget. The hashes detect damage; they do not authenticate a file.

## Mounting

`mount` shows the members of a container as read-only files. On Linux it uses FUSE (`/dev/fuse` and
`fusermount3`). On Windows it uses WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos,
<https://github.com/winfsp/winfsp>, which must be installed from <https://winfsp.dev/rel/>. macOS has no
`mount` command.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at
your option.
