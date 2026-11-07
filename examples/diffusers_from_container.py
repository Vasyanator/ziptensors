"""File: examples/diffusers_from_container.py
Purpose: run a Stable Diffusion pipeline of Diffusers whose model directory was packed into one
ziptensors container, reading it through the Python module, without a mount and without unpacking the
weights.

Main responsibilities: find the pipeline in the container by its `model_index.json`; restore only the
small files (configs, tokenizer, scheduler) into a temporary directory; build every model from its
config and fill it with `ziptensors.torch.load_state_dict` and `load_state_dict(strict=True)`; generate
one image with a fixed seed.

Key functions: `load_pipeline`, `main`.

Implementation notes: works for any pipeline whose `model_index.json` names Diffusers models and
schedulers and Transformers models and tokenizers, with one `*.safetensors` file per model folder,
as Stable Diffusion 1.x and SDXL directories have. Anything else is refused with an error. Needs
diffusers, transformers, torch and ziptensors. Pack the model with
`ziptensors pack sd.ziptensors <diffusers model directory>`. The alternative on Linux and Windows is
`ziptensors mount sd.ziptensors <mountpoint>` and the usual `DiffusionPipeline.from_pretrained` on the
mounted model directory.
"""

import argparse
import json
import logging
from pathlib import Path, PurePosixPath
import tempfile
import time
from typing import Any

import diffusers
import torch
import transformers
import ziptensors
from ziptensors.torch import load_state_dict

log = logging.getLogger(__name__)


def load_pipeline(container: ziptensors.Container, device: str, dtype: torch.dtype) -> Any:
    """Builds the Diffusers pipeline described by the container's `model_index.json`.

    Raises ValueError if the container holds no single pipeline, a member name is unsafe, a component
    class is not a Diffusers or Transformers model, scheduler or tokenizer, or a model folder does not
    hold exactly one safetensors file. Errors of ziptensors and of `load_state_dict(strict=True)` (a
    missing or unexpected tensor) propagate.
    """
    names = [info.name for info in container.members]
    for name in names:
        # Member names are untrusted: refuse anything that could leave the temporary directory,
        # including a backslash, which Windows reads as a separator.
        parts = PurePosixPath(name).parts
        if PurePosixPath(name).is_absolute() or ".." in parts or "\\" in name:
            raise ValueError(f"unsafe member name {name!r}")
    indexes = [name for name in names if PurePosixPath(name).name == "model_index.json"]
    if len(indexes) != 1:
        raise ValueError(f"expected one model_index.json in the container, found {indexes}")
    prefix = indexes[0].removesuffix("model_index.json")  # "" or "<packed directory>/"

    with tempfile.TemporaryDirectory() as tmp:
        # Configs and tokenizer files are small: restore them so that `from_pretrained` can read them.
        for name in names:
            if name.startswith(prefix) and not name.endswith(".safetensors"):
                path = Path(tmp, name.removeprefix(prefix))
                path.parent.mkdir(parents=True, exist_ok=True)
                container.member(name).restore_file(path)
        index = json.loads(Path(tmp, "model_index.json").read_text(encoding="utf-8"))

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
            # The index is untrusted too: only classes of the two libraries are looked up.
            libraries = {"diffusers": diffusers, "transformers": transformers}
            cls = getattr(libraries.get(library), class_name, None)
            if not isinstance(cls, type):
                raise ValueError(f"component {key}: unsupported class {library}.{class_name}")
            folder = Path(tmp, key)
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
            weights = [n for n in names if n.startswith(f"{prefix}{key}/") and n.endswith(".safetensors")]
            if len(weights) != 1:
                raise ValueError(f"component {key}: expected one safetensors file, found {weights}")
            model.load_state_dict(load_state_dict(container.member(weights[0]), device=device), strict=True)
            log.info("loaded %s (%s) from %s", key, class_name, weights[0])
            components[key] = model

    return getattr(diffusers, index["_class_name"])(**components)


def main() -> None:
    """Parses the arguments, loads the pipeline from the container and writes one image."""
    parser = argparse.ArgumentParser(description="Generate one image with a Diffusers pipeline "
                                     "read from a ziptensors container.")
    parser.add_argument("container", type=Path, help="container made by `ziptensors pack`")
    parser.add_argument("prompt")
    parser.add_argument("output", type=Path, help="PNG file to write")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default=None,
                        help="default: float16 on a GPU, float32 on the CPU")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=512, help="width and height in pixels")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    dtype = getattr(torch, args.dtype or ("float32" if args.device == "cpu" else "float16"))

    start = time.perf_counter()
    # A budget on what the container may decode, as for any file from elsewhere (README, Security).
    container = ziptensors.Container(args.container, max_total_decoded=64 << 30)
    pipeline = load_pipeline(container, args.device, dtype)
    log.info("pipeline loaded in %.1f s", time.perf_counter() - start)

    generator = torch.Generator(args.device).manual_seed(args.seed)
    image = pipeline(args.prompt, num_inference_steps=args.steps, height=args.size, width=args.size,
                     generator=generator).images[0]
    image.save(args.output)
    log.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
