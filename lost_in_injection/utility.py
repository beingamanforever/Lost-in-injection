"""Score Global-MMLU-Lite per language through AuditKIT's lm-eval engine (train environment).

Every model is scored the same way: lm-eval's Global-MMLU-Lite tasks (their defaults: 0-shot,
multiple choice by log-likelihood, no chat template) on the Hugging Face backend in bf16, with a
fixed batch size. AuditKIT saves the whole run with one prediction per question, so accuracy gets
the same bootstrap intervals as the attack rates (metrics.py).
"""

import argparse
from pathlib import Path

from auditkit.lmeval_engine import run_benchmark
from auditkit.runspec import RunConfig

LANGUAGES = ("en", "hi", "sw", "fr", "zh")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="Model directory.")
    parser.add_argument("--out", type=Path, required=True, help="Output folder.")
    args = parser.parse_args()

    result = run_benchmark(
        [f"global_mmlu_{lang}" for lang in LANGUAGES],
        f"hf:{args.model.resolve()}",
        # Batch 16 runs out of memory on an A10G: the longest batch's fp32 logits over the 262k
        # vocabulary alone take 4.5 GiB.
        config=RunConfig(seed=0, batch_size=4),
        dtype="bfloat16",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    result.save(str(args.out / "global_mmlu_lite.json"))
    for lang in LANGUAGES:
        print(f"global_mmlu_{lang}: {result.headline[f'global_mmlu_{lang}:acc']:.4f}")


if __name__ == "__main__":
    main()
