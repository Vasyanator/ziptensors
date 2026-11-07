# Loader study: where could a ziptensors decoder sit in a desktop application's weight loaders?

The study read the loaders of one private desktop application that ships many local models (native
ONNX Runtime in Rust, a Python PyTorch backend, stable-diffusion.cpp, diffusers and transformers). Its
source locations are not reproduced here; the findings below are stated for the libraries involved.

## Verdict
- Cheap decode-on-load: (1) native ONNX through the `ort` crate, which offers `commit_from_memory`, when
  one session builder owns the model path; (2) a Python large-model path that already reads
  tensor by tensor with `readinto` into fresh buffers, never mmap. Decoded tensors are anonymous
  memory, which also avoids GPU-driver pathologies specific to file-backed private mappings.
- Hard blocker inside (2): a disk-offload tier that re-reads spans with `os.preadv` at absolute file
  offsets on every forward pass. A compressed container needs independent chunks with an offset index
  in decoded coordinates (what `read_at` became), or that tier must read a decoded file.
- Needs a materialized file (path-only APIs): stable-diffusion.cpp (its C API takes only `*_path`
  strings), diffusers `from_single_file` for GGUF and single files, transformers `from_pretrained`
  directories, EasyOCR, onnxruntime-Python sessions as used. `torch.load` takes a file object, which
  is the seam `Member.open` later served.
- The largest cross-cutting risk is an integrity pipeline that pins the size and sha256 of the
  ORIGINAL file: a container on disk reads as missing or partial, and a repair step may delete it.
  One owner of "installed" must learn about containers, or the pins move to decoded bytes.
- Confidence: high on the seams; medium on size estimates; Windows and Android paths partly unverified.

## Evidence
### 1. Native ONNX (Rust)
- `ort` 2.0.0-rc.12 offers `commit_from_memory(&[u8])` and `commit_from_memory_directly` (borrowed,
  zero-copy, `.ort` format only). A memory commit is independent of the execution provider, including
  XNNPACK on Android.
- No model in the studied set used ONNX external data; the largest ONNX file was 343 MB. An ONNX model
  above 2 GB would need external data.
### 2. Python PyTorch backend
- One dispatch module sniffs the container kind by magic and returns a streaming reader that does
  `seek` plus chunked `readinto` into a fresh `bytearray`, then `torch.frombuffer`. A decoder fits as a
  replacement of that read body.
- The disk-offload tier issues `os.preadv` per contiguous span at a file offset; host and device tiers
  re-read only from host buffers or through the normal load path, so they are unaffected.
- Memory sizing uses header spans only, so decoded sizes keep it correct; hints derived from file
  sizes would under-estimate.
- A second, Rust-side header parser opens the same files and would need the container-aware opener too.
### 3. Other Python consumers (path mandatory unless noted)
- diffusers `FluxTransformer2DModel.from_single_file(gguf_path)`: path mandatory (its GGUF reader
  memory-maps the file). SDXL `from_single_file` / `from_pretrained(dir)`: path or directory.
- transformers `from_pretrained(directory, local_files_only=True)`: directory.
- EasyOCR `Reader(model_storage_directory=dir)`: the library opens `.pth` files itself.
- LaMa and similar checkpoints: `torch.load(path, ...)`; `torch.load` also accepts file objects.
### 4. stable-diffusion.cpp
- All weights are `const char*` paths; there is no buffer or read-callback API. An `enable_mmap` flag exists.
- A memfd path (`/proc/self/fd/N`) is plausible on Linux, created in the process that loads, since
  the file must be seekable and may be mapped; not tested. Windows has no equivalent: a real temp
  file (delete-on-close semantics need care) or a decoded cache directory.
### 5. Integrity and download pipeline
- A catalog pins exact size and lowercase sha256 per file; "installed" means every file's on-disk
  length equals the pin. A container of a different size reads as missing or partial; adoption hashes
  the file and fails; verification deletes files whose size or hash differ.
- The only post-download transform (extracting a ZIP member and pinning its size and hash) is a
  template for "decode, then pin".
- A container detected by magic, with file names unchanged, needs no change to the IPC between the
  application's Rust and Python halves; a new "this path is a container" field would.
### 6. Where the bytes are
- About 126 GB of the application's model folder are diffusion and text-encoder weights (GGUF Q8/Q4,
  BF16 safetensors); the ONNX and TorchScript models total about 2.6 GB. Compression value lives almost
  entirely in the large safetensors and GGUF files.

## Not verified
- Nothing was run; no benchmark of decode throughput against `preadv`.
- Whether `torch.load`, `safetensors` and diffusers accept in-memory buffers at the pinned versions;
  memfd with stable-diffusion.cpp's mmap; Windows temp-file semantics; Android memory commit at run time.

## Recommended next step
One owner, the ziptensors reader, exposing (a) random access in the decoded coordinate system and
(b) `read_into(buffer, offset)`. Integrate in this order:
1. The Python streaming reader and the disk-offload read path: both are already abstracted by path
   and offset.
2. The native ONNX session builder through `commit_from_memory` behind a path-or-container resolver.
   ONNX files are small, so this is for uniformity only.
3. One rule for "installed": keep pins on the decoded bytes and add a decode step, or pin the
   container and let runtimes decode. Do not let a repair step see containers before that.
4. Everything path-only: materialize a decoded file on demand (Windows: required; Linux: memfd
   optional), or keep those models uncompressed.

Before building: "installed, size, sha" is decided in several places (stat, a non-empty check, a
Python check, Rust header parsers), so a container needs one resolver or the copies drift.
