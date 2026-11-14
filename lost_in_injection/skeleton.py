"""Draw the pair skeleton once, so every run trains on the same rows and attacks (CPU only).

It mirrors Meta-SecAlign's draws (utils.py:72-88) with data seed 0: a random order of the curated
clean rows, one uniformly drawn injected Alpaca row each, 90% straightforward attacks placed before
or after the data at random, and 10% completion attacks whose fake response is the Stanford
Alpaca answer to the clean instruction when one exists. A completion attack's fake delimiters are
drawn later from its own seed, so they are the same in every language. Candidates are written in
draw order; the translation pool and the final pairs are taken from the front.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lost_in_injection.records import read_jsonl

DATA_SEED = 0
# Keeps injected text short enough to translate; it applies to every run alike (99.3% of rows).
MAX_INJECTION_WORDS = 64


def draw(clean: list[dict], injections: list[dict], reference: dict[str, str]) -> list[dict]:
    rng = np.random.RandomState(DATA_SEED)
    pairs = []
    for i, index in enumerate(rng.permutation(len(clean))):
        row = clean[index]
        injection = injections[rng.randint(len(injections))]
        attack = "straightforward" if rng.rand() < 0.9 else "completion"
        position = "after"
        if attack == "straightforward" and rng.rand() < 0.5:
            position = "before"
        pairs.append(
            {
                "pair_id": f"pair-{i:05d}",
                "clean": row,
                "injection": injection,
                "attack": attack,
                "position": position,
                "completion_seed": int(rng.randint(2**31)),
                "fake_response": reference.get(row["instruction"], row["output"]),
            }
        )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pool", type=Path, default=Path("data/pool"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/pairs/skeleton.jsonl"))
    args = parser.parse_args()

    clean = read_jsonl(args.pool / "clean" / "kept.jsonl")
    injections = [
        row
        for row in read_jsonl(args.pool / "injections" / "kept.jsonl")
        if len(f"{row['instruction']} {row['input']}".split()) <= MAX_INJECTION_WORDS
    ]
    alpaca = json.loads((args.raw / "alpaca_data.json").read_text())
    reference = {row["instruction"]: row["output"] for row in alpaca}
    pairs = draw(clean, injections, reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs))
    attacks = sum(p["attack"] == "completion" for p in pairs)
    print(f"{len(pairs)} candidates from {len(clean)} clean rows and {len(injections)} injections")
    print(f"completion share {attacks / len(pairs):.3f}")


if __name__ == "__main__":
    main()
