"""Translate the injected instructions and inputs of the first skeleton candidates (GPU).

Every training language gets the same texts: the instruction and the non-empty input of each of
the first POOL_SIZE candidates. `labels.py` then uses as multilingual slots only candidates whose
texts pass the gates in every language, so all runs draw from the same pairs. Each language's
records, with every candidate and gate value, go to `<out>/<language>/translations.jsonl`.
Machine translations already there are kept and only new texts are translated, so the pool can
grow; romanized and code-mixed records are cheap and always rebuilt. Romanized Hindi needs
IndicXlit (romanize.py, xlit environment) to run on the Hindi first, so the languages run in two
calls around it (scripts/build_data.sh).
"""

import argparse
from pathlib import Path

from lost_in_injection.languages import BY_NAME
from lost_in_injection.records import read_jsonl, read_records, write_jsonl
from lost_in_injection.runs import TRAINING_LANGUAGES
from lost_in_injection.translate import DERIVED, translate_or_rewrite

# The first 2,875 candidates gave 2,164 whose texts pass in all six languages, below the 2,500
# multilingual slots, so the pool was grown to leave headroom.
POOL_SIZE = 3600


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
        variant, path = BY_NAME[name], args.out / name / "translations.jsonl"
        kept = read_records(path) if path.exists() and variant.translator not in DERIVED else {}
        new = [text for text in texts if text not in kept]
        records = [*kept.values(), *(translate_or_rewrite(new, args.out, variant) if new else [])]
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(path, records)
        passed = sum(record["translation"] is not None for record in records)
        print(f"{name}: {passed} of {len(records)} texts passed the gates", flush=True)


if __name__ == "__main__":
    main()
