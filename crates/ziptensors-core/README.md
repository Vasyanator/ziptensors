# ziptensors-core

The library behind [ziptensors](https://github.com/Vasyanator/ziptensors): a lossless, compressed
container for tensor files (safetensors, GGUF, ONNX, PyTorch zip and opaque files) with random access at
tensor speed. Every restored file is bit-exact and proven by its BLAKE3 hash.

This crate holds the whole format: codecs, file profiles, the reader and the writer. The command line
tool is [`ziptensors-cli`](https://crates.io/crates/ziptensors-cli), the read-only filesystem view is
[`ziptensors-mount`](https://crates.io/crates/ziptensors-mount), and Python users install the
[`ziptensors`](https://pypi.org/project/ziptensors/) wheel.

## Example

```rust,no_run
use std::path::Path;

use ziptensors_core::{Container, EncodeOptions, OpenOptions, encode_file};

fn main() -> Result<(), ziptensors_core::Error> {
    encode_file(
        Path::new("model.safetensors"),
        Path::new("model.ziptensors"),
        &EncodeOptions::for_encode(),
    )?;

    // A file from elsewhere is untrusted input: give it a decode budget.
    let options = OpenOptions { max_total_decoded: 64 << 30, ..OpenOptions::default() };
    let container = Container::open(Path::new("model.ziptensors"), &options)?;
    let member = container.member("model.safetensors")?;
    let mut head = vec![0u8; 4096];
    member.read_at(0, &mut head)?;
    member.verify()?;
    Ok(())
}
```

## Untrusted input

`OpenOptions::max_total_decoded` defaults to `u64::MAX`, no budget. Set it whenever the file comes from
somewhere else. The hashes detect damage; they do not authenticate a file, since anyone who can change a
file can recompute them.

## Versions

The crate version and the format version are independent. The format is specified in
[`spec/ziptensors-format.md`](https://github.com/Vasyanator/ziptensors/blob/main/spec/ziptensors-format.md).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at
your option.
