"""Translate the injected instructions and inputs of the first skeleton candidates (GPU).

Every training language gets the same texts: the instruction and the non-empty input of each of
the first POOL_SIZE candidates. `labels.py` then uses as multilingual slots only candidates whose
texts pass the gates in every language, so all runs draw from the same pairs. Each language's
records, with every candidate and gate value, go to `<out>/<language>/translations.jsonl`.
"""

import argparse
from pathlib import Path

from lost_in_injection.languages import BY_NAME
from lost_in_injection.records import read_jsonl, write_jsonl
from lost_in_injection.runs import MULTILINGUAL_SLOTS, TRAINING_LANGUAGES
from lost_in_injection.translate import translate_or_rewrite

# About 15% headroom over the multilingual slots for texts that fail a gate in some language.
POOL_SIZE = MULTILINGUAL_SLOTS * 23 // 20


def injection_texts(pairs: list[dict]) -> list[str]:
    texts = [p["injection"][field] for p in pairs for field in ("instruction", "input")]
    return sorted({text for text in texts if text})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skeleton", type=Path, default=Path("data/pairs/skeleton.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/pairs/translations"))
    parser.add_argument("--languages", nargs="+", default=list(TRAINING_LANGUAGES))
    args = parser.parse_args()

    texts = injection_texts(read_jsonl(args.skeleton)[:POOL_SIZE])
    for name in args.languages:
        records = translate_or_rewrite(texts, args.out, BY_NAME[name])
        out_dir = args.out / name
        out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_dir / "translations.jsonl", records)
        passed = sum(record["translation"] is not None for record in records)
        print(f"{name}: {passed} of {len(records)} texts passed the gates", flush=True)


if __name__ == "__main__":
    main()
