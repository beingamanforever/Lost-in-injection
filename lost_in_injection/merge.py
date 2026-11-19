"""Merge a trained LoRA adapter into full Tiny Aya weights for evaluation (train environment).

vLLM 0.11.0 cannot serve LoRA on Tiny Aya, so the runners load merged weights. The
merged folder takes every non-weight file (config, generation config, tokenizer and chat template)
from the base model directory, so all models are evaluated alike and only the weights differ. The
base model is derived from the run name in `--run-dir` (runs/<run>/<seed>), so a Fire run cannot
silently merge onto Global.
"""

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM

from lost_in_injection.runs import base_model

WEIGHT_INDEX = "model.safetensors.index.json"


def is_weight_file(path: Path) -> bool:
    return path.suffix == ".safetensors" or path.name == WEIGHT_INDEX


def tensor_shapes(model_dir: Path) -> dict[str, list[int]]:
    shapes = {}
    for path in model_dir.glob("*.safetensors"):
        with safe_open(path, framework="pt") as weights:
            shapes |= {name: weights.get_slice(name).get_shape() for name in weights.keys()}
    return shapes


def merge(base_dir: Path, adapter_dir: Path, out_dir: Path) -> None:
    base = AutoModelForCausalLM.from_pretrained(base_dir.resolve(), dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, adapter_dir.resolve()).merge_and_unload()
    merged.save_pretrained(out_dir)
    for path in base_dir.iterdir():
        if not is_weight_file(path):
            shutil.copy(path, out_dir / path.name)  # Overwrites the config this version saved.
    if tensor_shapes(out_dir) != tensor_shapes(base_dir):
        raise ValueError("Merged tensors differ in names or shapes from the base checkpoint.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="runs/<run>/<seed>")
    args = parser.parse_args()
    out = args.run_dir / "merged"
    if "_" in str(out.resolve()):
        raise ValueError("Meta-SecAlign runners split model paths on '_'; use a path without one.")
    merge(base_model(args.run_dir.parts[-2]), args.run_dir / "adapter", out)


if __name__ == "__main__":
    main()
