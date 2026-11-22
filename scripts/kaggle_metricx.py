"""Score metricx_input.jsonl with MetricX-24 Hybrid XL on a Kaggle T4.

Runs Google's own metricx24/predict.py, unmodified, in its own environment at the versions its
repository was released with (requirements.txt, plus the versions those need), on one T4: the
fp32 XL checkpoint (about 14 GiB) fits it, and a second process would need a second copy of the
model in CPU memory while loading, more than a Kaggle session has. Writes
/kaggle/working/metricx_output.jsonl, parallel to the input.
Pushed with `kaggle kernels push`; see lost_in_injection/translation_quality.py.
"""

import os
import subprocess
from pathlib import Path

METRICX = ("https://github.com/google-research/metricx", "fc4978eb064670f7cc33e93ea4f52d38396b8ae6")
MODEL = ("google/metricx-24-hybrid-xl-v2p6", "f6e7f99a655582f28cb998dd3e6ca86b4217430d")
TOKENIZER = ("google/mt5-xl", "63fc6450d80515b48e026b69ef2fbbd426433e84")
PINS = [
    "torch==2.1.2",
    "transformers==4.30.2",
    "accelerate==0.20.3",  # Trainer's minimum for transformers 4.30
    "sentencepiece==0.1.99",
    "protobuf==3.20.3",  # mT5's slow-to-fast tokenizer conversion
    "datasets==2.13.1",
    "fsspec==2023.6.0",  # Later fsspec breaks datasets 2.13's local JSON loading
    "huggingface_hub==0.16.4",
    "numpy==1.26.4",
    "pyarrow==14.0.2",
    "setuptools==69.5.1",  # transformers 4.30 imports pkg_resources, which uv venvs lack
]
OUTPUT = Path("/kaggle/working/metricx_output.jsonl")  # Kaggle keeps only /kaggle/working
WORK = Path("/tmp/metricx-run")
PYTHON = WORK / "mx/bin/python"


def run(*command: str | Path) -> str:
    # stderr stays in the session log, so a failed step shows its own error.
    return subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True).stdout.strip()


def download(repo: tuple[str, str], *patterns: str) -> str:
    code = (
        "from huggingface_hub import snapshot_download; "
        f"print(snapshot_download({repo[0]!r}, revision={repo[1]!r}, "
        f"allow_patterns={list(patterns) or None!r}))"
    )
    return run(PYTHON, "-c", code).splitlines()[-1]


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    run("pip", "install", "-q", "uv")
    run("uv", "venv", "-q", "--python", "3.11", WORK / "mx")
    run("uv", "pip", "install", "-q", "--python", PYTHON, *PINS)
    run("git", "clone", "-q", METRICX[0], WORK / "metricx")
    run("git", "-C", WORK / "metricx", "checkout", "-q", METRICX[1])
    model = download(MODEL)
    tokenizer = download(TOKENIZER, "*.json", "spiece.model")

    input_file = next(Path("/kaggle/input").rglob("metricx_input.jsonl"))
    command = [PYTHON, "-m", "metricx24.predict", "--tokenizer", tokenizer]
    command += ["--model_name_or_path", model, "--max_input_length", "1536"]
    command += ["--batch_size", "1", "--input_file", input_file, "--output_file", OUTPUT, "--qe"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    subprocess.run(command, check=True, cwd=WORK / "metricx", env=env)
    print(f"scored {len(OUTPUT.read_text().splitlines())} translations")


if __name__ == "__main__":
    main()
