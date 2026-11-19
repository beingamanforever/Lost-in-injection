"""Independent translation quality: MetricX-24 reference-free scores for every machine translation.

The translation gates (translate.py) check each translation through a back-translation by the same
translator, so they are not independent of it. MetricX-24 Hybrid XL (Juraska et al., 2024) scores
an English source against its translation with no reference: 0 is a perfect translation and 25
the worst. `inputs` writes one MetricX row per kept translation in the evaluation sets and the
training pairs; scripts/kaggle_metricx.py scores them on Kaggle's free T4s; `summarize` writes one
row per set and language, with the Spearman correlation between MetricX and the gate's LaBSE
similarity. Romanized and code-mixed variants are word-for-word transforms of a scored
translation, so they are not scored again.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from lost_in_injection.languages import VARIANTS
from lost_in_injection.records import chosen_candidate, read_jsonl, write_jsonl

MACHINE_TRANSLATORS = ("nllb", "indictrans2")
INPUT = Path("data/quality/metricx_input.jsonl")
OUTPUT = Path("data/quality/metricx_output.jsonl")
SUMMARY = Path("results/translation_quality.csv")


def inputs(eval_root: Path, pairs_root: Path) -> list[dict]:
    rows = []
    for variant in VARIANTS:
        if variant.translator not in MACHINE_TRANSLATORS:
            continue
        for set_name, root in (("eval", eval_root), ("pairs", pairs_root)):
            path = root / variant.name / "translations.jsonl"
            if not path.exists():
                continue
            for record in read_jsonl(path):
                if record["translation"] is None:
                    continue
                rows.append(
                    {
                        "set": set_name,
                        "language": variant.name,
                        "translator": record.get("fallback", variant.translator),
                        "source": record["source"],
                        "hypothesis": record["translation"],
                        "reference": "",  # Reference-free: MetricX-24 Hybrid's QE input
                        "similarity": chosen_candidate(record)["similarity"],
                    }
                )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["set"], row["language"]), []).append(row)
    summary = []
    for (set_name, language), group in groups.items():
        scores = np.array([row["prediction"] for row in group])
        similarities = np.array([row["similarity"] for row in group])
        summary.append(
            {
                "set": set_name,
                "language": language,
                "n": len(group),
                "metricx_mean": round(float(scores.mean()), 2),
                "metricx_median": round(float(np.median(scores)), 2),
                "metricx_p90": round(float(np.percentile(scores, 90)), 2),
                # Negative when the two agree: MetricX counts errors, LaBSE counts similarity.
                "spearman_vs_labse": round(float(spearmanr(scores, similarities)[0]), 3),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("inputs", "summarize"))
    args = parser.parse_args()
    if args.step == "inputs":
        rows = inputs(Path("data/eval"), Path("data/pairs/translations"))
        write_jsonl(INPUT, rows)
        print(f"{len(rows)} translations -> {INPUT}")
        return
    summary = summarize(read_jsonl(OUTPUT))
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"{len(summary)} rows -> {SUMMARY}")


if __name__ == "__main__":
    main()
