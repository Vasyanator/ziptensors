# Module: crates/ziptensors-py/python

## Purpose

The pure-Python files of the package `ziptensors`, and the adapters that belong in Python: they know a
third-party library or subclass a stdlib class, which the Rust layer cannot do under abi3.

## Architecture

`data/` is a wheel data directory, the `data` key of `pyproject.toml`. Everything under `data/platlib/`
installs into site-packages, so `data/platlib/ziptensors/` lands beside the `__init__.py`, the stub and
the extension that maturin generates. Those two sets of files form one package at install time, and
only then: nothing here can be imported from the source tree, so tests run against an installed build.

This is not maturin's mixed layout, on purpose. In that layout `maturin develop` writes the compiled
extension into the source tree, and the project's rule is that a verification run leaves the tree as it
was. A directory called `ziptensors` next to `pyproject.toml` selects that layout, even without an
`__init__.py`, so none may exist there.

This file sits above `data/` because maturin fails on a plain file in a data directory, and anything
under `platlib/` would be installed.

## Files and submodules

| Path | Role |
|---|---|
| `data/platlib/ziptensors/_member_file.py` | `MemberFile`, the `io.RawIOBase` behind `Member.open`. Private name, imported from `src/reader.rs`. |
| `data/platlib/ziptensors/torch.py` | `load_state_dict`, `load_into`, `TorchLoadError`, `GgmlTensor`. Owns the dtype allowlist, the ggml block table, the host fill and the device stager. |
| `data/platlib/ziptensors/safetensors.py` | `safe_open`, `SafeSlice`, `SafetensorError`: the reading API of the `safetensors` package over a member. |

## Contracts and invariants

- These files see only the public surface of the extension, as `ziptensors.pyi` states it.
- torch is imported inside the functions that need it, never at module level.
- The tensor table is a hint made by a parser. `torch.py` checks dtype, shape against length, the range
  against the member and the total against the member before the first allocation, so a hostile table
  cannot ask for more memory than the member's own length, which the container budget bounds.
- A dtype is accepted through an allowlist, `_SAFETENSORS_DTYPES` in `torch.py`, so an entry new to the
  registry is refused until someone decides it. It is the safetensors torch map restricted to the dtypes
  the core registry knows, and `safetensors.py` imports it rather than keep a copy, so both readers
  accept the same set. A torch attribute is looked up with `getattr`, so an older torch gives
  `TorchLoadError`, not `AttributeError`.
- Quantized or packed bytes are never reinterpreted. Block-quantized GGUF tensors (every row whose
  `Tensor.block` holds more than one element) are either refused or handed over as `uint8` blocks with
  their ggml type (`GgmlTensor`, or a diffusers `GGUFParameter` for the types diffusers dequantizes).
  The block geometry is the core's, read from `Tensor.block`; Python keeps no copy, and a test compares
  it with `gguf.GGML_QUANT_SIZES` for every type gguf knows. `load_into` always refuses quantized
  tensors.
- `safetensors.py` parses the header from the member's bytes, never from the tensor table, because the
  table is a hint and drops `__metadata__`. It validates as the safetensors package does (length
  prefix within 100 MB and the member, entries tiling the data section exactly, lengths matching
  shapes, dtype tags the package knows), which bounds every allocation by the member length. Slices
  read the contiguous run of rows the leading indices select, then finish the index in the framework
  and copy the result out.
- `ctypes.c_char` over `Tensor.data_ptr()` is the route into a tensor's storage. It needs no numpy and
  takes bfloat16 and float8, which `Tensor.numpy()` refuses. It is valid only for a fresh contiguous
  tensor whose byte length was proven (`_fill_host`), or a region of a pinned staging buffer that no
  queued copy still reads (`_DeviceStager`).
- A `cuda` device (CUDA, or ROCm through the same API) is filled by `_DeviceStager`: two pinned host
  buffers, `read_at` into one while the other's `non_blocking` copy runs on a side stream, an event per
  buffer guarding its reuse, `record_stream` on each destination, and a full synchronization before
  the call returns. Each `fill` first orders the side stream after the current stream: the caching
  allocator may hand a destination a block whose last reader (a conversion, another thread's kernel)
  is still queued there, and `record_stream` does not cover that. Pinned host memory is at most two
  buffers per call, capped at the bytes being loaded; torch's caching host allocator keeps them pinned
  for reuse afterwards.
  Every other device decodes on the host and moves each tensor. The CPU path decodes in place.
- The source distribution moves `pyproject.toml` to its root, so `include` copies `data/` and the stub
  `ziptensors.pyi` there as well; without the stub at that root a wheel built from the sdist has no
  `__init__.pyi` and no `py.typed`.

## Editing map

- New accepted dtype: `_SAFETENSORS_DTYPES` at the top of `torch.py` (both readers), plus roundtrip
  tests in `tests/test_torch_integration.py` and `tests/test_safe_open.py`.
- New quantized ggml type: nothing here once the core gives it a row and a geometry; the test against
  `gguf.GGML_QUANT_SIZES` counts the block-quantized rows, so update that count with it.
- Device pipeline: `_DeviceStager` and `_decode_all` in `torch.py`; `safetensors.py` uses the same
  stager for whole tensors.
- New file: under `data/platlib/ziptensors/`. It installs by itself, and it is imported as
  `ziptensors.<name>`. A public addition to `ziptensors` itself belongs in the stub instead.
- User documentation of the adapter: `../src/MODULE_README.md`.
