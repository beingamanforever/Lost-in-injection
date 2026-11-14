"""Romanize gated Hindi with AI4Bharat's IndicXlit (xlit environment, Python 3.10).

IndicXlit is trained on human-typed romanizations (Aksharantar), so its output reads the way people
type Hindi in Latin letters. It runs on the masked Hindi, so placeholders stay, and the content is
the gated Hindi's word for word. The gates then run in the gen environment (translate.py).
Writes `<root>/romanized-hindi/romanized.jsonl`.
"""

import argparse
from pathlib import Path

import torch

from lost_in_injection.records import masked_translation, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="data/eval or a translations dir")
    args = parser.parse_args()

    # IndicXlit's fairseq checkpoints pickle argparse.Namespace. Allowing only that class keeps
    # PyTorch's weights-only loading on.
    torch.serialization.add_safe_globals([argparse.Namespace])
    from ai4bharat.transliteration import XlitEngine

    engine = XlitEngine(src_script_type="indic", beam_width=4, rescore=False)
    hindi = [r for r in read_jsonl(args.root / "hindi" / "translations.jsonl") if r["translation"]]
    rows = []
    for record in hindi:
        masked = masked_translation(record)
        rows.append(
            {
                "source": record["source"],
                "hindi": masked,
                "masked": engine.translit_sentence(masked, lang_code="hi"),
            }
        )
    out_dir = args.root / "romanized-hindi"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "romanized.jsonl", rows)
    kept_words = sum(len(r["hindi"].split()) == len(r["masked"].split()) for r in rows)
    print(f"romanized {len(rows)} texts; word count kept in {kept_words}")


if __name__ == "__main__":
    main()
