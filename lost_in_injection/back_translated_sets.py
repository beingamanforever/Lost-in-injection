"""English paraphrase control for the language-search attack.

Is trying many languages stronger than trying the same number of English rewordings? The
translation gate already back-translated every attack text into English (translate.gate), so each
machine-translated variant yields an English paraphrase of the whole attack set at no extra cost.
This writes one AlpacaFarm-Hacked injection set per variant, "english-via-<variant>", from the
chosen candidates' back-translations. Evaluate them with
`evaluate --eval-root data/eval-back-translated --no-injecagent --variants english-via-...`.
"""

import argparse
import json
from pathlib import Path

from lost_in_injection.eval_sets import INJECTION_FILE, write_json
from lost_in_injection.masking import mask, unmask
from lost_in_injection.records import chosen_candidate, read_jsonl

# The machine-translated variants; romanized Hindi, Hinglish and Spanglish are rewrites of these.
VARIANTS = ("french", "chinese", "hindi", "swahili", "urdu", "tamil", "amharic", "spanish")


def back_translated(source: str, record: dict) -> str:
    text = unmask(chosen_candidate(record)["back_translation"], mask(source)[1])
    if text is None:
        raise ValueError(f"back-translation lost an identifier: {source!r}")
    return text


def write_set(eval_root: Path, out_root: Path, variant: str) -> None:
    english = json.loads((eval_root / "english" / INJECTION_FILE).read_text())
    records = {r["source"]: r for r in read_jsonl(eval_root / variant / "translations.jsonl")}
    sentences = [back_translated(s, records[s]) for s in english["ignore_sentences"]]
    if any(s.count("{injected_prompt}") != 1 for s in sentences):
        raise ValueError(f"{variant}: an attack sentence lost its {{injected_prompt}} slot")
    out_dir = out_root / f"english-via-{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = english["injected_prompt"]
    injection = {"injected_prompt": back_translated(prompt, records[prompt])}
    write_json(out_dir / INJECTION_FILE, {**injection, "ignore_sentences": sentences})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-root", type=Path, default=Path("data/eval"))
    parser.add_argument("--out", type=Path, default=Path("data/eval-back-translated"))
    args = parser.parse_args()
    for variant in VARIANTS:
        write_set(args.eval_root, args.out, variant)
        print(f"wrote english-via-{variant}")


if __name__ == "__main__":
    main()
