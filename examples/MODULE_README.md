# Module: examples

## Purpose

Runnable programs that show ziptensors used by other libraries. They are documentation: not packaged,
not on any runtime path, not run by the test suite. Each one uses only the public API of a published
frontend (the Python module, the command line tool), never crate internals.

## Files

| Path | Role |
|---|---|
| `diffusers_from_container.py` | Builds a Diffusers pipeline (Stable Diffusion 1.x, SDXL) from a container made by `ziptensors pack <model directory>` and generates one image. Needs diffusers, transformers, torch and ziptensors. |

## Contracts

- Read through the Python module, so an example runs on every platform, without a mount.
- Weights are loaded with `ziptensors.torch.load_state_dict` and `load_state_dict(strict=True)`: a
  model whose tensors do not match is an error, never a partly random model.
- Member names are untrusted input: an example checks a name before joining it to a directory, and
  opens containers with a `max_total_decoded` budget, as the README's Security section asks.
- Unsupported input (another pipeline layout, weights not in safetensors) is refused with an error.
- No paths of a machine, no model files and no generated output in this directory.

## Editing map

- Python API used here: `crates/ziptensors-py/ziptensors.pyi` and `crates/ziptensors-py/python/`.
- The README section "Example: Stable Diffusion with Diffusers" quotes the core lines of
  `diffusers_from_container.py`; change both together.
