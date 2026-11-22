"""Run Meta-SecAlign's AlpacaFarm-Hacked and InjecAgent runners on one model, per language variant.

The runners write their logs next to the model path and read `data/` from the working directory
(Meta_SecAlign test.py:176-180, test_injecagent.py:485-498). So each variant gets its own folder:
symlinks to the model files, a `data/` folder with that variant's test files, and the runners'
untouched outputs (`summary.tsv` plus per-case JSON). The model is loaded once and reused.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from lost_in_injection import secalign
from lost_in_injection.languages import INJECAGENT_VARIANTS, VARIANTS
from lost_in_injection.secalign import ALPACA_ATTACKS, INJECAGENT_DEFENSE

ALL_VARIANTS = [v.name for v in VARIANTS if v.group != "intermediate"]

# Every model is evaluated on the second A10G (driver 535), where vLLM's sampler warm-up runs out
# of memory at 0.85 and above; same value for every run.
GPU_MEMORY_UTILIZATION = 0.80
# Tiny Aya's context length. Llama (128k) and Qwen3 (256k) are capped to it: their full-length
# KV cache does not fit an A10G, and every model gets the same prompt-plus-reply budget.
MAX_MODEL_LEN = 8192


class SharedGpuLLM(LLM):
    """vLLM engine with a fixed memory share and context, split over every visible GPU (70B
    needs two), that sends text prompts with a single BOS (none for Qwen, which has no BOS)."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            **{**kwargs, "tensor_parallel_size": torch.cuda.device_count()},
        )
        self.bos_token = self.get_tokenizer().bos_token or ""

    def generate(self, prompts, *args, **kwargs):
        # The chat template already starts with BOS, and vLLM adds another to text prompts.
        prompts = [prompt.removeprefix(self.bos_token) for prompt in prompts]
        return super().generate(prompts, *args, **kwargs)


def small_gpu() -> bool:
    """Capturing CUDA graphs with LoRA runs out of memory on a 24 GB A10G at 0.80, so small GPUs
    run eager. Eager mode only skips graph capture; decoding is unchanged."""
    return torch.cuda.get_device_properties(0).total_memory < 40 * 2**30


class SharedGpuLoraLLM(SharedGpuLLM):
    """The same engine with runtime LoRA (the runners enable it only for paths with '_')."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**kwargs, "enable_lora": True, "enforce_eager": small_gpu()})


def runner_generate(adapter: Path | None, temperature: float = 0.0, seed: int | None = None):
    """The runners' generation (Meta_SecAlign utils.py:355-365), with a LoRA adapter applied at
    runtime and/or sampling instead of greedy decoding (the sampling control, ACL plan E2).

    Merging a LoRA update into bf16 weights rounds most of it away: each weight moves by about
    1e-5, below bf16's step of about 1e-4 at typical weight sizes. vLLM 0.11 serves LoRA on Qwen3
    and Llama but not on Tiny Aya, so this is used for Qwen and Llama. `adapter` holds only
    adapter_config.json and adapter_model.safetensors, since vLLM also loads a tokenizer from the
    adapter folder.
    """
    request = LoRARequest("adapter", 1, str(adapter)) if adapter else None

    def generate(llm_input, model, tokenizer, *_unused, **_unused_keywords):
        params = SamplingParams(
            temperature=temperature, seed=seed, max_tokens=8192, stop=tokenizer.eos_token
        )
        return [r.outputs[0].text for r in model.generate(llm_input, params, lora_request=request)]

    return generate


def link_all(sources, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        link = target_dir / source.name
        if not link.exists():
            link.symlink_to(source.resolve())


def check_no_underscore(path: Path) -> None:
    # The runners treat the part before the first "_" as a base model plus a LoRA adapter.
    if "_" in str(path):
        raise ValueError(f"Meta-SecAlign runners split model paths on '_': {path}")


def evaluate(
    model_dir: Path,
    eval_root: Path,
    raw_dir: Path,
    out_dir: Path,
    variants: list[str],
    alpaca_defense: str,
    input_role: bool,
    run_injecagent: bool = True,
    adapter: Path | None = None,
    temperature: float = 0.0,
    sample_seed: int | None = None,
) -> None:
    # absolute(), not resolve(): the runners only see these strings, and on Modal the volume's
    # physical path contains a '_'.
    model_dir, eval_root, raw_dir, out_dir = (
        p.absolute() for p in (model_dir, eval_root, raw_dir, out_dir)
    )
    check_no_underscore(model_dir)
    utils = secalign.load("utils")
    utils.LLM = SharedGpuLoraLLM if adapter else SharedGpuLLM
    alpaca = secalign.load("test")
    injecagent = secalign.load("test_injecagent")
    model, tokenizer = utils.load_vllm_model(str(model_dir))
    alpaca.load_vllm_model = lambda *_args, **_kwargs: (model, tokenizer)
    if adapter or temperature:  # Both runners call test_model_output_vllm, in their own namespace.
        alpaca.test_model_output_vllm = injecagent.test_model_output_vllm = runner_generate(
            adapter.resolve() if adapter else None, temperature, sample_seed
        )

    for variant in variants:
        work = out_dir / variant
        check_no_underscore(work)
        variant_dir = eval_root / variant
        link_all(model_dir.iterdir(), work)
        data_files = [raw_dir / "tools.json", raw_dir / "attacker_simulated_responses.json"]
        link_all([*data_files, *variant_dir.glob("test_cases_*.json")], work / "data")
        os.chdir(work)

        injection = json.loads((variant_dir / "alpacafarm_injection.json").read_text())
        alpaca.TEST_INJECTED_PROMPT = injection["injected_prompt"]
        alpaca.IGNORE_ATTACK_SENTENCES = {"test": injection["ignore_sentences"]}
        np.random.seed(0)  # The ignore attacks sample one sentence per case.
        alpaca.test_vllm(
            SimpleNamespace(
                model_name_or_path=str(work),
                attack=ALPACA_ATTACKS,
                defense=alpaca_defense,
                test_data=str(raw_dir / "davinci_003_outputs.json"),
                num_samples=-1,
                tensor_parallel_size=1,
                lora_alpha=8.0,
                instruction_hierarchy=input_role,
            )
        )
        if not run_injecagent or variant not in INJECAGENT_VARIANTS:
            continue
        injecagent.main(
            {
                "model_name_or_path": str(work),
                "defense": INJECAGENT_DEFENSE,
                "lora_alpha": 8.0,
                "instruction_hierarchy": input_role,
            },
            model,
            tokenizer,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="Model directory, no '_'.")
    parser.add_argument("--out", type=Path, required=True, help="Evaluation output directory.")
    parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS)
    parser.add_argument("--eval-root", type=Path, default=Path("data/eval"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--alpaca-defense", default="none", help="Meta-SecAlign prompting defense.")
    parser.add_argument("--no-input-role", action="store_false", dest="input_role")
    # InjecAgent always uses Meta-SecAlign's sandwich prompt, so a prompting-defense baseline is
    # identical to the base model there.
    parser.add_argument("--no-injecagent", action="store_false", dest="injecagent")
    parser.add_argument("--adapter", type=Path, help="LoRA folder to apply at runtime to --model.")
    # The sampling control only: every other run is greedy.
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sample-seed", type=int)
    args = parser.parse_args()
    evaluate(
        args.model,
        args.eval_root,
        args.raw,
        args.out,
        args.variants,
        args.alpaca_defense,
        args.input_role,
        args.injecagent,
        args.adapter,
        args.temperature,
        args.sample_seed,
    )


if __name__ == "__main__":
    main()
