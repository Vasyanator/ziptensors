# Module: crates/ziptensors-py/src

## Purpose

The Python module `ziptensors` over `ziptensors-core`, built as abi3 for CPython 3.11+. This Rust part
translates arguments, results and errors, and holds no logic of its own. The adapters above it are pure
Python, in `../python/`.

## Architecture

`Container` and `Member` in Python own a core `Container` and `Member`. Every call that reads the file
or encodes runs in `Python::detach`; a buffer call goes through `buffer::fill_detached`, which holds the
`Py_buffer` in the attached frame and passes only the slice into the closure. A core `Error` reaches
Python through `errors::raised`, the one translation.

The packaging lives one level up: `pyproject.toml` for maturin, `ziptensors.pyi` for type checkers,
which maturin installs with a `py.typed` marker, `python/` for the pure-Python files, and `tests/` for
pytest. `Member.open` is the one Rust call into Python: it constructs `ziptensors._member_file.MemberFile`.

## Files and submodules

| Path | Role |
|---|---|
| `lib.rs` | Crate root, lint policy, the table of what the Python module exports. |
| `buffer.rs` | The held destination buffer and `fill_detached`. The only `unsafe` of the crate. |
| `errors.rs` | `ZiptensorsError`, the family subclasses, the mapping from `ErrorCode`. |
| `names.rs` | Spec registry names of dtypes and profiles, the strings Python sees. |
| `reader.rs` | `Container`, `Member`, `MemberInfo`, `Tensor` (with the core's block geometry, `block`). |
| `report.rs` | Read-only mirrors of the core's report types. |
| `writer.rs` | `encode_file`, `pack`, and their argument translation. |

## Contracts and invariants

- The GIL is released around all file I/O and decoding. What keeps it: zeroing the `bytes` of
  `Member.read`, whose memset is linear in the length, so large reads belong in a caller's buffer.
- Destination buffers are requested with writable, C-contiguous and format flags, of any item format,
  with the length in bytes. `pyo3::buffer::PyBuffer<u8>` is not usable: its format and item size check
  refuses a float32 array. The one refused format is a Python object item (`O`, numpy `dtype=object`),
  because decoded bytes would overwrite object pointers. Failures are `BufferError` whatever the
  exporter raised, with the exporter's exception as the cause; no buffer interface at all is `TypeError`.
- A view is held from before `detach` until after it, and released while attached. `WritableBuffer` is
  not `Send`, so the compiler keeps it out of the closure.
- Exclusive use of a buffer during a call is the Python caller's contract, as with `readinto`. It is
  stated on `Member`, and the interpreter cannot enforce it once the GIL is released.
- `ZiptensorsError` is the base exception, with one subclass per family: `ZiptensorsIoError`,
  `FormatError`, `UnsupportedError`, `BudgetError`, `CorruptionError`, `RangeError`, `NotFoundError`,
  `InputError`, and `NotDurableError`, a `ZiptensorsIoError` raised when a write's output WAS replaced
  but its directory sync failed. Each raised exception has `code`, the stable `ErrorCode` value. Bad arguments that never
  reach the core are ordinary `ValueError`, `TypeError`, `OverflowError`.
- `dtype` is the registry name of the spec, and `shape` keeps the order of the source file: GGUF stays
  fastest varying first, the reverse of numpy's order.
- `threads=None` and `threads=0` both mean one per logical CPU, the core's meaning of 0.
- Rust tests are off for this cdylib; tests run under pytest with a throwaway venv outside the tree:

  ```bash
  python3 -m venv --system-site-packages <venv>        # numpy, torch, xxhash are used if present
  <venv>/bin/pip install maturin pytest
  # unset CONDA_PREFIX first: maturin refuses both it and VIRTUAL_ENV
  CARGO_TARGET_DIR=<outside the tree> VIRTUAL_ENV=<venv> <venv>/bin/maturin develop
  PYTHONDONTWRITEBYTECODE=1 <venv>/bin/pytest          # from crates/ziptensors-py
  ```

- The library target keeps its default name so it cannot collide with the `ziptensors` binary. The Python
  module name is set by `#[pymodule(name)]` and the maturin `module-name`.

## Using ziptensors with PyTorch, Transformers and Diffusers

`ziptensors.torch` holds `load_state_dict` and `load_into`, `ziptensors.safetensors` holds `safe_open`,
and `Member.open()` gives a file object. torch is optional and imported on the first call, so importing
either module works without it. Neither is imported by `import ziptensors`. Install the `torch` extra,
or torch itself.

The snippets are copied from `tests/test_torch_integration.py`, where each one ran. `member` is a
`Member`; a caller opening untrusted input passes `max_total_decoded` to `Container` (see Budgets).

```python
from ziptensors.torch import TorchLoadError, load_into, load_state_dict
```

`load_state_dict(member, device="cpu", dtype=None, keys=None, *, quantized="refuse", staging_bytes=None)`
allocates each tensor with `torch.empty` and decodes into its storage with the GIL released, so there is
no intermediate copy.

- **Accepted.** Safetensors `F64 F32 F16 BF16 F8_E4M3 F8_E5M2 I64 I32 I16 I8 U64 U32 U16 U8 BOOL`
  (`F8_E4M3` is `torch.float8_e4m3fn`, `F8_E5M2` is `torch.float8_e5m2`, as in the safetensors package),
  GGUF `F64 F32 F16 BF16`. Anything else raises `TorchLoadError`, a `ValueError`, naming the tensor and
  why, before the first allocation: `NIBBLE`, dtypes the core registry does not know (`C64`,
  `F8_E4M3FNUZ`, `F8_E8M0`, which reach Python as `UNKNOWN(0x0000)`), GGUF integers, any other profile,
  and a dtype the installed torch lacks. Bytes are never reinterpreted. A `BOOL` byte other than 0 or 1
  is refused too.
- **Quantized GGUF.** Every block-quantized type the core knows (`Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q8_1`, the
  K-quants, `IQ*`, `TQ1_0 TQ2_0 MXFP4`; a `Tensor.block` of more than one element) follows
  `quantized`. `"refuse"`, the default, raises
  `TorchLoadError`. `"raw"` returns a `GgmlTensor(data, ggml_type, shape)`: `data` is a `uint8` tensor
  of the blocks in gguf-py's byte shape, `ggml_type` the registry name (`Q8_0`), `shape` the logical
  shape in torch order. `"diffusers"` returns the `GGUFParameter` that diffusers' `load_gguf_checkpoint`
  makes of the same bytes, equal in bytes, shape and `quant_type`; float tensors stay plain tensors,
  BF16 included. It needs diffusers and gguf, and raises `TorchLoadError` without them or for a type
  diffusers does not dequantize (its `SUPPORTED_GGUF_QUANT_TYPES`; `MXFP4`, `IQ2_XXS`...). `dtype` never
  touches quantized blocks. A mode other than `"refuse"` on a non-GGUF member is a `ValueError`.
  `load_into` refuses quantized tensors. The diffusers route:
  `Model.from_single_file(load_state_dict(member, quantized="diffusers"),
  quantization_config=GGUFQuantizationConfig(compute_dtype=...), config=<dir>, local_files_only=True)`.
- **GGUF tensors the table does not list.** The core keeps no row for a ggml type without a registry
  dtype, such as a retired ggml id. A load of the whole member compares the table with the header's
  tensor count and refuses a mismatch rather than return a partial state dict. `keys` loads the listed
  tensors.
- **GGUF shapes** are reversed: the file states the fastest varying dimension first and torch the slowest.
  The bytes do not move, and the result equals the array the writer held. Names stay GGUF names, so a
  llama.cpp or ComfyUI file still needs that project's name mapping before a module can take it.
- **`device`.** `meta` is refused. On a `cuda` device (CUDA, or ROCm through the same API) bytes are
  decoded into two pinned host buffers of `staging_bytes` each (default `ziptensors.torch.STAGING_BYTES`,
  64 MiB, capped at the bytes being loaded) and copied asynchronously on a side stream while the next
  piece decodes; the call synchronizes before it returns. Host memory peaks at the two buffers, the
  device holds the state dict. On any other device each tensor is decoded on the host, moved and its
  host copy freed before the next one, so host memory peaks at the largest tensor. Both add the decode
  scratch of the memory contract in `ARCHITECTURE.md`. torch's caching host allocator keeps the pinned
  buffers after the call, for reuse, rather than return them to the OS. A piece boundary inside a chunk
  decodes that chunk twice, about 1.6% extra work with the default buffers.
- **`dtype`** converts floating point tensors, F8 included, on the device for `cuda` and on the host
  otherwise, briefly holding both versions of one tensor. Integer, bool and quantized tensors keep
  their dtype.
- **`keys`** selects tensors in the given order. An unknown key raises `NotFoundError`.

`load_into(module, member, strict=True, assign=True)` is `load_state_dict` followed by
`Module.load_state_dict`. With `assign=True` the module takes the decoded tensors themselves. A file holds
neither non-persistent buffers nor the second name of a tied parameter, and a forward pass through a
module partly on meta returns meta tensors instead of failing. So `strict=True` raises `TorchLoadError`
for anything left on meta, and `strict=False` logs it and leaves the rest to the caller.

Transformers, the direct route, which also ties weights and fills buffers:

```python
state_dict = load_state_dict(member)
model = CLIPTextModel.from_pretrained(None, config=config, state_dict=state_dict).eval()
```

Transformers on a meta module. A tied head and a non-persistent buffer are the caller's to finish:

```python
with torch.device("meta"):
    model = GPT2LMHeadModel(gpt2_config())
load_into(model, member, strict=False)
model.tie_weights()
```

```python
with torch.device("meta"):
    model = CLIPTextModel(clip_config())
load_into(model, member, strict=False)
positions = torch.arange(model.config.max_position_embeddings).expand((1, -1))
model.get_submodule("embeddings").register_buffer("position_ids", positions, persistent=False)
```

Diffusers, from a config, or from a dict with a local config directory:

```python
with torch.device("meta"):
    unet = UNet2DConditionModel.from_config(original.config)
load_into(unet, member)
```

```python
state_dict = load_state_dict(member)
unet = UNet2DConditionModel.from_single_file(
    state_dict, config=str(directory), local_files_only=True
).eval()
```

A Diffusers pipeline takes the loaded components in its constructor:

```python
pipeline = StableDiffusionPipeline(
    vae=vae.eval(),
    text_encoder=text_encoder.eval(),
    tokenizer=tokenizer,
    unet=unet.eval(),
    scheduler=DDIMScheduler(steps_offset=1, clip_sample=False),
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False,
)
```

`ziptensors.safetensors.safe_open(filename, framework, device="cpu", *, member=None, threads=None,
max_total_decoded=None)` is the reading API of the safetensors package over a member: `keys()`
(sorted), `offset_keys()`, `metadata()`, `get_tensor`, `get_tensors`, `get_slice` with `get_shape`,
`get_dtype` and basic indexing, a context manager, and `SafetensorError("File is closed")` after it.

- `filename` is a container path, with `member` naming a member of a pack, or a `Member` object.
- The header is parsed from the member's bytes, so `__metadata__` is there, and validated like the
  safetensors package does before anything is allocated; a tag that is no safetensors dtype (`GARBAGE`,
  `f32`) is refused at open. The error names the member's profile when the core did not see safetensors.
- `framework` is `pt` (`torch`, `pytorch`) or `np` (`numpy`, host only, no BF16 or F8). Dtypes are those
  of `load_state_dict`; a tensor of another valid safetensors dtype the core does not know (`C64`,
  `F8_E8M0`, `F4`) is listed with its shape and tag, and reading it raises.
- `device` is `cpu`, an int N for `cuda:N`, or a torch device (numpy: `cpu` or a device of type cpu).
  Whole tensors on `cuda` use the staging pipeline above. A slice reads only the run of rows its leading
  indices select, on the host, then moves.
- Errors are `SafetensorError` (a `ValueError`, not the class of the safetensors package), plus the
  core's errors for the container and `TorchLoadError` for a dtype torch cannot take.
- Differences from safetensors 0.7.0: its numpy slices ignore a step, these do not; a `SafeSlice` taken
  before `__exit__` still reads there and raises "File is closed" here; a list, bool or float index is
  `TypeError` here (there a list selects rows, `True` adds an axis and a float is `IndexError`); a
  `torch.device` object is accepted here and refused there.

Each call of this example runs in `tests/test_safe_open.py`:

```python
from ziptensors.safetensors import safe_open

with safe_open(container, "pt", device=0) as file:
    weight = file.get_tensor("t.F32")
    rows = file.get_slice("t.F32")[1, 2:4]
```

A `.pt` member goes through `torch.load`, which unpickles, so `weights_only=True` is mandatory here.
`Member.open()` returns a read-only, seekable `io.RawIOBase`, and torch reads the storages through its
`readinto`, into their final memory:

```python
loaded = torch.load(member.open(), weights_only=True)
```

Results of the table below with torch 2.12.0a0, transformers 5.22.1, diffusers 0.42.1, accelerate 1.15.0,
safetensors 0.8.0, on CPU, from tiny models built from a config with a fixed seed; outputs compared with
`torch.equal`:

| Route | Result |
|---|---|
| `from_pretrained(None, config=, state_dict=)`, CLIP text, GPT-2 with a tied head | Exact. Parameters share the dict's storage (seen on CLIP), so no second copy. |
| Meta module + `load_into`, plain CLIP or GPT-2 | Refused by `strict=True`. Exact after `strict=False` and the steps above. |
| `from_config` on meta + `load_into`, UNet, VAE | Exact, with no extra step. |
| `ModelMixin.from_single_file(dict, config=<dir>, local_files_only=True)`, Diffusers key names | Exact. Parameters share the dict's storage. |
| `ModelMixin.from_single_file(dict)` with no `config` | Does not work offline: Diffusers looks up a default repository on the hub. |
| `ModelMixin.from_pretrained(None, config=..., state_dict=...)` | Does not work: `OSError`, no `config.json`. |
| `DiffusionPipeline.from_single_file(dict)` | Does not work: path only. Use the constructor. |
| `StableDiffusionPipeline(...)` from loaded components, two steps | Exact against the same pipeline over the original components. |
| `torch.load(member.open(), weights_only=True)` | Exact, and a global outside the allowlist is still refused. |
| `torch.load(member.open(), mmap=True)` | Does not work: torch needs a path. |

The dtypes, quantized GGUF, `safe_open` and GPU staging were tested with torch 2.12.0a0 (ROCm build),
diffusers 0.37.1, gguf 0.18.0, safetensors 0.7.0, transformers 5.5.4 and numpy from the same environment.
Under transformers 5.5.4 the meta-module CLIP route of the table fails in its test: that release has no
`CLIPTextModel.embeddings`. GPU loading is tested with tiny tensors on ROCm (gfx1201), including a
current stream kept busy by another thread, not on CUDA hardware and not for throughput. Quantized GGUF
(`Q4_0 Q5_1 Q8_0 Q4_K`) is compared with `load_gguf_checkpoint` byte for byte, not run through
`from_single_file` with `GGUFQuantizationConfig`.

Not tried: original (non-Diffusers) checkpoint key names through `from_single_file`, `device_map`, a whole
model on a GPU, and any torch before 2.12, though the `torch` extra states 2.1, where `assign` arrived.

## Budgets

`Container(path, max_total_decoded=...)` is how a restore gets a budget, since core checks it at open. The
default is core's, no budget, so a caller opening untrusted input must pass one.

## Editing map

- New binding: a method in `reader.rs` or a function in `writer.rs` over a public `core` API, exported in
  `lib.rs`, declared in `ziptensors.pyi`. A test compares the stub with the runtime.
- New `ErrorCode`: pick its family in `errors.rs`; the match does not compile until then.
- New dtype or profile: `names.rs`, same reason.
- New report field: `report.rs` and the stub.
- PyTorch adapter, `safe_open` or file object: `../python/`, see its `MODULE_README.md`, and the section
  above.
