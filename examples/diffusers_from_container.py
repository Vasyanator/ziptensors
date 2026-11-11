"""File: examples/diffusers_from_container.py
Purpose: run a Stable Diffusion pipeline of Diffusers whose weights are in ziptensors containers,
reading them through the Python module, without a mount and without unpacking the weights. Two
layouts: the whole model directory packed into one container, or a model directory whose weight files
were each replaced by a container beside them (`ziptensors compress -r`).

Main responsibilities: find the pipeline by its `model_index.json`; read the small files (configs,
tokenizer, scheduler) from a temporary directory they are restored to (packed layout) or from the
model directory itself (per-file layout); build every model from its config and fill it with
`ziptensors.torch.load_state_dict` (safetensors) or `torch.load(..., weights_only=True)` (legacy
`.bin`), refusing missing tensors and any unexpected one but those the libraries' own loaders
drop; generate one image with a fixed seed, from an input image and mask for a pipeline that takes
them (img2img, inpainting).

Key functions: `load_pipeline` (packed), `load_per_file_pipeline` (per-file), `_fill_model` (which
tensors a model must and may get), `main`.

Implementation notes: works for any Diffusers pipeline whose `model_index.json` names Diffusers
models and schedulers and Transformers models and tokenizers; a pipeline that takes an input image
or a mask (img2img, inpainting) gets them from `--image` and `--mask`, and is refused before
loading without them. Packed: one `*.safetensors` file per model folder,
as Stable Diffusion 1.x and SDXL directories have. Per-file: one weight file per model folder or one
`*.index.json` of shards, safetensors preferred over `.bin`, with an optional Diffusers variant
(`fp16`). Anything else is refused with an error. Needs diffusers, transformers, torch and ziptensors.
The alternative on Linux and Windows is `ziptensors mount` (packed) or `ziptensors mount --overlay`
(per-file) and the usual `DiffusionPipeline.from_pretrained` on the mounted model directory; on macOS,
which has no mount, this example is the route.
"""

import argparse
from collections.abc import Callable
import inspect
import json
import logging
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
from typing import Any, NamedTuple

import diffusers
import torch
import transformers
import ziptensors
from ziptensors.torch import load_state_dict

log = logging.getLogger(__name__)

# A budget on what one container may decode, as for any file from elsewhere (README, Security).
MAX_TOTAL_DECODED = 64 << 30

# Weight file suffixes the per-file layout loads, in order of preference.
WEIGHT_SUFFIXES = (".safetensors", ".bin")

# `<base>.safetensors.index.json`, or with a Diffusers variant `<base>.safetensors.index.fp16.json`.
INDEX_NAME = re.compile(
    r"^(?P<base>.+)(?P<suffix>\.safetensors|\.bin)\.index(?:\.(?P<variant>[^.]+))?\.json$"
)

# Loads the weights of the component of `model_index.json` called by the argument, onto the device.
LoadWeights = Callable[[str], dict[str, torch.Tensor]]

# Called with the pipeline class before any weight is read, so that a run that cannot work stops early.
CheckPipeline = Callable[[type], None]


def _check_relative_name(name: str) -> None:
    """Raises ValueError unless `name` is a relative path that stays inside the directory it is
    joined to: names from a container or a JSON file are untrusted. A backslash is refused because
    Windows reads it as a separator."""
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe name {name!r}")


def _check_file_name(name: str) -> None:
    """Raises ValueError unless `name` is one safe file name, with no directory part."""
    _check_relative_name(name)
    if "/" in name or name == ".":
        raise ValueError(f"{name!r} is not a single file name")


def _fill_model(key: str, model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Replaces every weight of `model`, the component `key`, by the tensors of `state`.

    Every tensor the model keeps must be in `state`: a missing one would leave a random weight, so
    it is refused. A tensor of `state` the model does not keep is accepted only where the libraries'
    own loaders accept it, and is then logged and dropped: a buffer the model registers as
    non-persistent (CLIP's `position_ids`, which older checkpoints carry and current Transformers
    no longer saves), or a name matching the class's `_keys_to_ignore_on_load_unexpected` patterns,
    which Diffusers and Transformers apply with `re.search`. Any other one is refused, since it
    means the weights belong to another model.

    Raises ValueError naming the missing or unexpected tensors. Errors of
    `load_state_dict(strict=True)` (a tensor of the wrong shape) propagate.
    """
    kept = set(model.state_dict())
    missing = sorted(kept - state.keys())
    if missing:
        raise ValueError(f"component {key}: the weights lack {len(missing)} tensors of the model: "
                         f"{', '.join(missing)}")
    # `state_dict()` leaves out exactly the buffers registered with `persistent=False`.
    non_persistent = {name for name, _ in model.named_buffers()} - kept
    patterns = getattr(type(model), "_keys_to_ignore_on_load_unexpected", None) or []
    unexpected = sorted(state.keys() - kept)
    refused = [name for name in unexpected if name not in non_persistent
               and not any(re.search(pattern, name) for pattern in patterns)]
    if refused:
        raise ValueError(f"component {key}: the weights hold {len(refused)} tensors the model does "
                         f"not have: {', '.join(refused)}")
    if unexpected:
        log.info("component %s: ignored %d tensors the model does not keep (non-persistent buffers "
                 "or names its class ignores on loading): %s", key, len(unexpected),
                 ", ".join(unexpected))
    model.load_state_dict({name: state[name] for name in kept}, strict=True)


def _build_pipeline(
    root: Path,
    load_weights: LoadWeights,
    device: str,
    dtype: torch.dtype,
    check_pipeline: CheckPipeline | None,
) -> Any:
    """Builds the pipeline that `root/model_index.json` describes, reading configs, tokenizers and
    schedulers below `root` and every model's weights through `load_weights`. `check_pipeline`, if
    given, sees the pipeline class before anything else is loaded.

    Raises ValueError for a pipeline class that is not a Diffusers pipeline, a component name that is not one safe file name or a component class that
    is not a Diffusers or Transformers model, scheduler or tokenizer, and for weights that lack a
    tensor of their model or hold one it does not accept (`_fill_model`). Errors of `load_weights`
    and of `load_state_dict(strict=True)` (a tensor of the wrong shape) propagate.
    """
    index = json.loads(Path(root, "model_index.json").read_text(encoding="utf-8"))
    pipeline_name = index.get("_class_name")
    pipeline_class = getattr(diffusers, pipeline_name, None) if isinstance(pipeline_name, str) else None
    if not isinstance(pipeline_class, type) or not issubclass(pipeline_class, diffusers.DiffusionPipeline):
        raise ValueError(f"model_index.json: unsupported pipeline class {pipeline_name!r}")
    if check_pipeline is not None:
        check_pipeline(pipeline_class)
    components: dict[str, Any] = {}
    for key, value in index.items():
        if key.startswith("_") or value is None:
            continue
        if not isinstance(value, list):  # a pipeline option such as force_zeros_for_empty_prompt
            components[key] = value
            continue
        library, class_name = value
        if library is None:  # an absent optional component, such as safety_checker
            components[key] = None
            continue
        # The index is untrusted too: only classes of the two libraries are looked up, and a
        # component name must not lead out of the model directory.
        _check_file_name(key)
        libraries = {"diffusers": diffusers, "transformers": transformers}
        cls = getattr(libraries.get(library), class_name, None)
        if not isinstance(cls, type):
            raise ValueError(f"component {key}: unsupported class {library}.{class_name}")
        folder = Path(root, key)
        if issubclass(cls, diffusers.SchedulerMixin):
            components[key] = cls.from_config(cls.load_config(folder))
            continue
        if issubclass(cls, transformers.PreTrainedTokenizerBase):
            components[key] = cls.from_pretrained(folder)
            continue
        # Build the model on the device with random weights, then replace every weight.
        with torch.device(device):
            if issubclass(cls, diffusers.ModelMixin):
                model = cls.from_config(cls.load_config(folder))
            elif issubclass(cls, transformers.PreTrainedModel):
                model = cls(cls.config_class.from_pretrained(folder))
            else:
                raise ValueError(f"component {key}: unsupported class {library}.{class_name}")
        model.to(dtype=dtype).eval()
        _fill_model(key, model, load_weights(key))
        log.info("loaded %s (%s)", key, class_name)
        components[key] = model

    return pipeline_class(**components)


def load_pipeline(
    container: ziptensors.Container,
    device: str,
    dtype: torch.dtype,
    check_pipeline: CheckPipeline | None = None,
) -> Any:
    """Builds the Diffusers pipeline described by the `model_index.json` of a packed container.
    `check_pipeline`, if given, sees the pipeline class before any weight is read.

    Raises ValueError if the container holds no single pipeline, a member name is unsafe, a component
    class is not a Diffusers or Transformers model, scheduler or tokenizer, or a model folder does not
    hold exactly one safetensors file, and for weights that do not match their model
    (`_fill_model`). Errors of ziptensors and of `load_state_dict(strict=True)` propagate.
    """
    names = [info.name for info in container.members]
    for name in names:
        # Member names are untrusted: refuse anything that could leave the temporary directory.
        _check_relative_name(name)
    indexes = [name for name in names if PurePosixPath(name).name == "model_index.json"]
    if len(indexes) != 1:
        raise ValueError(f"expected one model_index.json in the container, found {indexes}")
    prefix = indexes[0].removesuffix("model_index.json")  # "" or "<packed directory>/"

    def load_weights(key: str) -> dict[str, torch.Tensor]:
        weights = [n for n in names if n.startswith(f"{prefix}{key}/") and n.endswith(".safetensors")]
        if len(weights) != 1:
            raise ValueError(f"component {key}: expected one safetensors file, found {weights}")
        log.info("reading %s", weights[0])
        return load_state_dict(container.member(weights[0]), device=device)

    with tempfile.TemporaryDirectory() as tmp:
        # Configs and tokenizer files are small: restore them so that `from_pretrained` can read them.
        for name in names:
            if name.startswith(prefix) and not name.endswith(".safetensors"):
                path = Path(tmp, name.removeprefix(prefix))
                path.parent.mkdir(parents=True, exist_ok=True)
                container.member(name).restore_file(path)
        return _build_pipeline(Path(tmp), load_weights, device, dtype, check_pipeline)


def _per_file_members(folder: Path) -> dict[str, ziptensors.Member]:
    """Opens every `*.ziptensors` file of `folder` (not below it) as a per-file container and returns
    its member by the member's name, which is the original file name.

    The name comes from the container, never from the container's file name: the member name is
    authoritative and the `.ziptensors` name only a convention. Raises ValueError for a container that
    does not hold exactly one member with one safe file name, or for two containers whose members
    have one name. Errors of ziptensors (a corrupt container, the budget) propagate.
    """
    members: dict[str, ziptensors.Member] = {}
    for path in sorted(folder.glob("*.ziptensors")):
        if not path.is_file():
            continue
        # A member keeps its container file open after the Container object is gone.
        container = ziptensors.Container(path, max_total_decoded=MAX_TOTAL_DECODED)
        infos = container.members
        if len(infos) != 1:
            raise ValueError(f"{path} holds {len(infos)} members; a per-file container holds one")
        name = infos[0].name
        _check_file_name(name)
        if name in members:
            raise ValueError(f"two containers in {folder} hold {name!r}")
        members[name] = container.member(name)
    return members


class _Weights(NamedTuple):
    """One way a model folder offers its weights."""

    suffix: str  # ".safetensors" or ".bin"
    variant: str | None  # Diffusers variant such as "fp16", None for none
    label: str  # the weight file name, or the index file name of shards
    weight_map: dict[str, str] | None  # tensor name -> shard file name; None for one file


def _variant(stem: str) -> str | None:
    """The Diffusers variant of a weight file stem: `fp16` of `diffusion_pytorch_model.fp16`."""
    return stem.rpartition(".")[2] if "." in stem else None


def _weight_candidates(folder: Path, members: dict[str, ziptensors.Member]) -> list[_Weights]:
    """Lists the ways `folder` offers its model's weights: each `*.index.json` of shards with its
    `weight_map`, and each weight file that no index names as a shard. A file counts whether it is
    in a container or still plain, so that a plain one is found and refused rather than passed over.

    Raises ValueError for an index without a `weight_map` of names to safe shard file names.
    """
    candidates: list[_Weights] = []
    shards: set[str] = set()
    for path in sorted(folder.iterdir()):
        match = INDEX_NAME.match(path.name)
        if match is None or not path.is_file():
            continue
        weight_map = json.loads(path.read_text(encoding="utf-8")).get("weight_map")
        if not isinstance(weight_map, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in weight_map.items()
        ):
            raise ValueError(f"{path}: no weight_map of tensor names to shard file names")
        for shard in weight_map.values():
            _check_file_name(shard)
        shards.update(weight_map.values())
        candidates.append(_Weights(match["suffix"], match["variant"], path.name, weight_map))
    plain = {p.name for p in folder.iterdir() if p.is_file() and p.suffix in WEIGHT_SUFFIXES}
    for name in sorted((plain | set(members)) - shards):
        for suffix in WEIGHT_SUFFIXES:
            if name.endswith(suffix):
                candidates.append(_Weights(suffix, _variant(name.removesuffix(suffix)), name, None))
    return candidates


def _load_file(member: ziptensors.Member, device: str) -> dict[str, torch.Tensor]:
    """Decodes the state dict of one weight file from its member.

    A `.bin` file is a pickle, so it is read only by torch's restricted unpickler
    (`weights_only=True`), which builds tensors and plain containers and runs no other code; anything
    but a flat dict of tensors is refused with ValueError. Errors of `load_state_dict` and
    `torch.load` propagate.
    """
    if member.name.endswith(".safetensors"):
        return load_state_dict(member, device=device)
    with member.open() as file:
        state = torch.load(file, map_location=device, weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise ValueError(f"{member.name}: not a flat state dict of tensors")
    return state


def load_per_file_pipeline(
    root: Path,
    device: str,
    dtype: torch.dtype,
    variant: str | None = None,
    check_pipeline: CheckPipeline | None = None,
) -> Any:
    """Builds the Diffusers pipeline of the model directory `root` after `ziptensors compress -r`:
    configs, tokenizers and schedulers are read from `root`, each model's weights from the per-file
    containers in its folder. `check_pipeline`, if given, sees the pipeline class before any weight
    is read.

    Per model folder, the weights are one weight file or the shards of one `*.index.json`, of the
    given Diffusers `variant` (None: files without one), safetensors preferred over `.bin`; shard
    names of `weight_map` are looked up by member name. When `ziptensors compress -r -k` kept the
    originals, the containers are read and the originals ignored: reading the containers is what this
    mode is for, every chunk is checked against its hash while it decodes, and a rerun of compress
    replaces a container that its original no longer matches.

    Raises ValueError for no or several weight candidates in a folder, a weight file that is not in a
    container (the tree is not fully compressed), a shard whose tensors differ from the `weight_map`,
    an unsafe name, an unsupported component class, or weights that do not match their model
    (`_fill_model`). Errors of ziptensors, torch and `load_state_dict(strict=True)` propagate.
    """
    def load_weights(key: str) -> dict[str, torch.Tensor]:
        folder = Path(root, key)
        members = _per_file_members(folder)
        candidates = [c for c in _weight_candidates(folder, members) if c.variant == variant]
        suffix = next((s for s in WEIGHT_SUFFIXES if any(c.suffix == s for c in candidates)), None)
        preferred = [c for c in candidates if c.suffix == suffix]
        if len(preferred) != 1:
            labels = [c.label for c in preferred]
            raise ValueError(f"component {key}: expected one set of weights of variant {variant}, "
                             f"found {labels}")
        label, weight_map = preferred[0].label, preferred[0].weight_map
        files = [label] if weight_map is None else sorted(set(weight_map.values()))

        state: dict[str, torch.Tensor] = {}
        for name in files:
            member = members.get(name)
            if member is None:
                where = "is not compressed" if Path(folder, name).exists() else "is missing"
                raise ValueError(f"component {key}: {name} {where}; expected a per-file container "
                                 "holding it (`ziptensors compress -r`)")
            log.info("reading %s/%s", key, name)
            part = _load_file(member, device)
            if weight_map is not None:
                expected = {tensor for tensor, shard in weight_map.items() if shard == name}
                if set(part) != expected:
                    raise ValueError(f"component {key}: tensors of {name} differ from {label}")
            state.update(part)
        return state

    return _build_pipeline(root, load_weights, device, dtype, check_pipeline)


def _input_problem(pipeline_class: type, image: Path | None, mask: Path | None) -> str | None:
    """What is wrong with the input image and mask given for `pipeline_class`, or None.

    A pipeline takes an input image if its `__call__` has an `image` parameter, and a mask if it has
    `mask_image`. Diffusers declares both with a default of None but refuses the call without them,
    so a parameter that exists counts as required. An input the pipeline has no parameter for is a
    problem too, rather than being dropped unseen.
    """
    parameters = inspect.signature(pipeline_class.__call__).parameters
    name = pipeline_class.__name__
    for parameter, flag, given in (("image", "--image", image), ("mask_image", "--mask", mask)):
        if parameter in parameters and given is None:
            return f"{name} needs an input {parameter.replace('_', ' ')}: pass {flag} PATH"
        if parameter not in parameters and given is not None:
            return f"{name} takes no {parameter.replace('_', ' ')}: leave out {flag}"
    return None


def main() -> None:
    """Parses the arguments, loads the pipeline from the containers and writes one image."""
    parser = argparse.ArgumentParser(description="Generate one image with a Diffusers pipeline "
                                     "read from ziptensors containers.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("container", type=Path, nargs="?",
                        help="container made by `ziptensors pack` of a model directory")
    source.add_argument("--per-file-dir", type=Path, metavar="DIR",
                        help="model directory after `ziptensors compress -r`")
    parser.add_argument("prompt")
    parser.add_argument("output", type=Path, help="PNG file to write")
    parser.add_argument("--variant", default=None,
                        help="with --per-file-dir: weights of this Diffusers variant, such as fp16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default=None,
                        help="default: float16 on a GPU, float32 on the CPU")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=512, help="width and height in pixels")
    parser.add_argument("--image", type=Path, default=None,
                        help="input image, for a pipeline that takes one (img2img, inpainting)")
    parser.add_argument("--mask", type=Path, default=None,
                        help="mask image, white where to paint, for an inpainting pipeline")
    args = parser.parse_args()
    if args.variant is not None and args.per_file_dir is None:
        parser.error("--variant applies to --per-file-dir")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    dtype = getattr(torch, args.dtype or ("float32" if args.device == "cpu" else "float16"))

    def check_pipeline(pipeline_class: type) -> None:
        # Before any weight is read: a missing input image is a usage error, not a Diffusers
        # traceback after minutes of loading.
        problem = _input_problem(pipeline_class, args.image, args.mask)
        if problem is not None:
            parser.error(problem)

    start = time.perf_counter()
    if args.per_file_dir is not None:
        pipeline = load_per_file_pipeline(args.per_file_dir, args.device, dtype, args.variant,
                                          check_pipeline)
    else:
        container = ziptensors.Container(args.container, max_total_decoded=MAX_TOTAL_DECODED)
        pipeline = load_pipeline(container, args.device, dtype, check_pipeline)
    log.info("pipeline loaded in %.1f s", time.perf_counter() - start)

    inputs: dict[str, Any] = {}
    if args.image is not None:
        inputs["image"] = diffusers.utils.load_image(str(args.image))
    if args.mask is not None:
        inputs["mask_image"] = diffusers.utils.load_image(str(args.mask))
    generator = torch.Generator(args.device).manual_seed(args.seed)
    image = pipeline(args.prompt, num_inference_steps=args.steps, height=args.size, width=args.size,
                     generator=generator, **inputs).images[0]
    image.save(args.output)
    log.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
