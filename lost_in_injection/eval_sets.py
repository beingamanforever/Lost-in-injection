"""Write one evaluation folder per language variant.

Each folder holds the four InjecAgent test-case files and the AlpacaFarm-Hacked injection text
for that variant. English is the untouched original.
"""

import argparse
import json
import shutil
from pathlib import Path

from lost_in_injection import secalign

INJECAGENT_FILES = [f"test_cases_{a}_{s}.json" for a in ("dh", "ds") for s in ("base", "enhanced")]


def write_english(raw_dir: Path, out_dir: Path) -> None:
    config = secalign.load("config")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in INJECAGENT_FILES:
        shutil.copy(raw_dir / name, out_dir / name)
    injection = {
        "injected_prompt": config.TEST_INJECTED_PROMPT,
        "ignore_sentences": config.IGNORE_ATTACK_SENTENCES["test"],
    }
    (out_dir / "alpacafarm_injection.json").write_text(
        json.dumps(injection, ensure_ascii=False, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/eval"))
    args = parser.parse_args()
    write_english(args.raw, args.out / "english")


if __name__ == "__main__":
    main()
