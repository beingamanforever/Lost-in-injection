"""Score MATH-500 and MGSM through AuditKIT's lm-eval engine (train environment, ACL plan E6).

Scored like Global-MMLU-Lite (utility.py): lm-eval's own task defaults, no chat template, the
Hugging Face backend in bf16 and a fixed batch size. The tasks are lm-eval 0.4.13's
`minerva_math500` (Minerva prompt, math-verify answer check) and `mgsm_native_cot_<lang>` for
MGSM's 11 languages (chain of thought in the question's language). AuditKIT keeps one prediction
per question.
"""

import argparse
from pathlib import Path

from auditkit.lmeval_engine import run_benchmark
from auditkit.runspec import RunConfig

MGSM_LANGUAGES = ("en", "bn", "de", "es", "fr", "ja", "ru", "sw", "te", "th", "zh")
TASKS = ["minerva_math500", *(f"mgsm_native_cot_{lang}" for lang in MGSM_LANGUAGES)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="Model directory.")
    parser.add_argument("--out", type=Path, required=True, help="Output folder.")
    args = parser.parse_args()
    result = run_benchmark(
        TASKS,
        f"hf:{args.model.resolve()}",
        config=RunConfig(seed=0, batch_size=4),
        dtype="bfloat16",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    result.save(str(args.out / "math.json"))
    for name, value in sorted(result.headline.items()):
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
