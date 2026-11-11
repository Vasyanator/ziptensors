# Module: examples

## Purpose

Runnable programs that show ziptensors used by other libraries. They are documentation: not packaged,
not on any runtime path, not run by the test suite. Each one uses only the public API of a published
frontend (the Python module, the command line tool), never crate internals.

## Files

| Path | Role |
|---|---|
| `diffusers_from_container.py` | Builds a Diffusers pipeline (Stable Diffusion 1.x, SDXL) from a container made by `ziptensors pack <model directory>`, or with `--per-file-dir` from a model directory after `ziptensors compress -r` (the route on macOS, which has no mount), and generates one image, from `--image` and `--mask` for a pipeline that takes them (img2img, inpainting). Needs diffusers, transformers, torch and ziptensors. |

## Contracts

- Read through the Python module, so an example runs on every platform, without a mount.
- Weights are loaded with `ziptensors.torch.load_state_dict` and checked by `_fill_model` before
  `load_state_dict(strict=True)`: a tensor the model keeps and the weights lack is an error, never a
  partly random model. A tensor the model does not keep is accepted only where the libraries' own
  loaders accept it (a buffer registered as non-persistent, a name matching the class's
  `_keys_to_ignore_on_load_unexpected`), and is logged; any other is an error. A legacy `.bin` member
  is read only with `torch.load(member.open(), weights_only=True)`, never by the full unpickler.
- Per-file layout: the member name, read from the container, is the original file name; the
  container's own file name is never trusted for it. Shards of an `*.index.json` are found by member
  name. The container is read and an original kept by `-k` ignored; a weight file with no container
  is an error.
- Member names are untrusted input: an example checks a name before joining it to a directory, and
  opens containers with a `max_total_decoded` budget, as the README's Security section asks.
- A pipeline's inputs come from its `__call__` signature: an `image` or `mask_image` parameter needs
  `--image` or `--mask`, and an input given for a pipeline without that parameter is refused. The
  check runs on the pipeline class before any weight is read, as a usage error (exit 2).
- Unsupported input (another pipeline layout, a pipeline class that is not a Diffusers pipeline, packed weights not in safetensors, several candidate
  weight sets in a per-file folder) is refused with an error.
- No paths of a machine, no model files and no generated output in this directory.

## Editing map

- Python API used here: `crates/ziptensors-py/ziptensors.pyi` and `crates/ziptensors-py/python/`.
- The README section "Example: Stable Diffusion with Diffusers" quotes the core lines of
  `diffusers_from_container.py`; change both together.
