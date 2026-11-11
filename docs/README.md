# ziptensors documentation

User documentation for ziptensors 0.2 (container format 0.3). Start with the
[project README](../README.md) for what ziptensors is, how to install it, and results; the pages below
go into detail.

| Document | What it covers | For |
|---|---|---|
| [cli.md](cli.md) | Every `ziptensors` subcommand and flag, defaults, exit codes, error codes, `inspect --json` schema | Command line users and scripts |
| [editing.md](editing.md) | Changing a container: `edit`, the default, append (`-F`) and in-place (`-R`) strategies and their risks, `compact`, generations, `recover` | Anyone who updates containers after writing them |
| [directories.md](directories.md) | `compress -r` and `decompress -r`: per-file containers in a model directory, which files are compressed, collisions, links, reruns, using the result | Users of model directories and download caches |
| [mount.md](mount.md) | Read-only mount of a container or a compressed directory from the command line or Python: the mounted tree, options and cache, unmounting, Linux FUSE and Windows WinFsp requirements, no macOS backend, Diffusers, llama.cpp and ComfyUI | Programs that only take a file path |
| [python.md](python.md) | The Python module: installation, examples, then a reference for reading, writing, editing and recovery, trees, mounts, logging, report classes, exceptions, threads and the GIL, and the `ziptensors.torch` and `ziptensors.safetensors` adapters | Python users |
| [rust.md](rust.md) | The `ziptensors-core` crate task by task: reading (ranges, restore, `ReadCache`), writing, editing, recovering, trees, mounting through `ziptensors-mount`, errors, threads and memory | Rust developers |
| [c-api.md](c-api.md) | The C ABI (`ziptensors.h`): building and linking, a complete example, conventions (status codes, options structs, ABI versioning, strings), handle ownership and threads, functions by task, logging. No mount or tree functions | C and FFI users |
| [safety.md](safety.md) | Untrusted containers and decode budgets, what the hashes do and do not protect, member names and restore paths, format parsers on files you compress, pickle (never unpickled), crash behavior of writes and edits, the read-only mount, reporting a vulnerability | Anyone opening a container they did not make |

The normative file format is [spec/ziptensors-format.md](../spec/ziptensors-format.md).
