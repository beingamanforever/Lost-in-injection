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
from vllm import LLM

from lost_in_injection import secalign

GPU_MEMORY_UTILIZATION = 0.90  # The A10G is dedicated to this project; same value for every run.
# Meta-SecAlign's non-adaptive AlpacaFarm-Hacked attacks (test.py:254).
ALPACA_ATTACKS = [
    "straightforward",
    "straightforward_before",
    "ignore",
    "ignore_before",
    "completion",
    "completion_ignore",
]
INJECAGENT_DEFENSE = "sandwich"  # Meta-SecAlign's InjecAgent protocol (run_tests.py:23).


class SharedGpuLLM(LLM):
    """vLLM engine with a fixed memory share that sends text prompts with a single BOS."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, gpu_memory_utilization=GPU_MEMORY_UTILIZATION, **kwargs)
        self.bos_token = self.get_tokenizer().bos_token

    def generate(self, prompts, *args, **kwargs):
        # The chat template already starts with BOS, and vLLM adds another to text prompts.
        prompts = [prompt.removeprefix(self.bos_token) for prompt in prompts]
        return super().generate(prompts, *args, **kwargs)


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
) -> None:
    model_dir, eval_root, raw_dir, out_dir = (
        p.resolve() for p in (model_dir, eval_root, raw_dir, out_dir)
    )
    check_no_underscore(model_dir)
    utils = secalign.load("utils")
    utils.LLM = SharedGpuLLM
    alpaca = secalign.load("test")
    injecagent = secalign.load("test_injecagent")
    model, tokenizer = utils.load_vllm_model(str(model_dir))
    alpaca.load_vllm_model = lambda *_args, **_kwargs: (model, tokenizer)

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
    parser.add_argument("--variants", nargs="+", default=["english"])
    parser.add_argument("--eval-root", type=Path, default=Path("data/eval"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--alpaca-defense", default="none", help="Meta-SecAlign prompting defense.")
    parser.add_argument("--no-input-role", action="store_false", dest="input_role")
    args = parser.parse_args()
    evaluate(
        args.model,
        args.eval_root,
        args.raw,
        args.out,
        args.variants,
        args.alpaca_defense,
        args.input_role,
    )


if __name__ == "__main__":
    main()
