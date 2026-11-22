"""Language-specific risk under the LLM judge, next to both string rules.

Reads each run seed's judge.json (judge.py), applies judge.final_payload's evidence check to each
answer, rebuilds analysis.py's hits with "hijacked" meaning a final payload answer of "yes", and
recomputes the language-risk table for both rules on the same run seeds. Also counts, per run
seed, how the two rules agree on each (attack, case).

Usage: python -m lost_in_injection.judged --runs runs --out results/acl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from lost_in_injection.analysis import (
    ANYWHERE,
    comprehension_adjusted,
    language_risk,
    load_hits,
    outputs_file,
    parity_by_variant,
    write,
)
from lost_in_injection.judge import final_payload
from lost_in_injection.metrics import ALL_LANGUAGES
from lost_in_injection.secalign import ALPACA_ATTACKS


def load_judged(runs: Path) -> tuple[dict, dict]:
    """(model, seed) -> variant -> (attacks, cases): judged hijacks, and the final payload
    answers."""
    hits, labels = {}, {}
    for judge_file in sorted(runs.glob("*/*/judge.json")):
        seed_dir = judge_file.parent
        model, seed = seed_dir.parent.name, seed_dir.name
        judged = json.loads(judge_file.read_text())
        for variant in ALL_LANGUAGES:
            files = judged.get(variant, {})
            names = [
                outputs_file(seed_dir / "eval" / variant, model, a).name for a in ALPACA_ATTACKS
            ]
            if all(n in files for n in names):
                found = np.array(
                    [
                        [
                            final_payload(answer, case["output"])
                            for answer, case in zip(
                                files[n],
                                json.loads((seed_dir / "eval" / variant / n).read_text()),
                                strict=True,
                            )
                        ]
                        for n in names
                    ]
                )
                labels.setdefault((model, seed), {})[variant] = found
                hits.setdefault((model, seed), {})[variant] = found == "yes"
    return hits, labels


def agreement(string_hits: dict, labels: dict) -> list[dict]:
    rows = []
    for key in sorted(labels):
        counts = Counter()
        for variant, found in labels[key].items():
            for hit, label in zip(string_hits[key][variant].ravel(), found.ravel(), strict=True):
                counts[("string_hit" if hit else "string_miss", label)] += 1
        rows += [
            {"model": key[0], "seed": key[1], "string": s, "judge": j, "count": n}
            for (s, j), n in sorted(counts.items())
        ]
    return rows


def share(answers: list[dict], question: str, *values: str) -> float:
    return sum(a[question] in values for a in answers) / len(answers)


def answer_rates(runs: Path) -> list[dict]:
    """Per run seed and language, over every (attack, case): how often the judge says the payload
    was obeyed, the reply was not (or only partly) in English, the task was not (or only partly)
    done, and the reply was broken."""
    rows = []
    for judge_file in sorted(runs.glob("*/*/judge.json")):
        model, seed = judge_file.parent.parent.name, judge_file.parent.name
        for variant, files in json.loads(judge_file.read_text()).items():
            answers = [a for found in files.values() for a in found if a["payload"] != "unparsed"]
            if not answers:
                continue
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "language": variant,
                    "replies": len(answers),
                    "payload_yes": share(answers, "payload", "yes"),
                    "reply_not_english": share(answers, "language", "no"),
                    "reply_partly_english": share(answers, "language", "partly"),
                    "task_not_done": share(answers, "task", "no"),
                    "task_partly_done": share(answers, "task", "partly"),
                    "broken": share(answers, "broken", "yes"),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("results/acl"))
    args = parser.parse_args()

    judged, labels = load_judged(args.runs)
    string = {key: by_variant for key, by_variant in load_hits(args.runs).items() if key in judged}
    anywhere = {k: v for k, v in load_hits(args.runs, ANYWHERE).items() if k in judged}
    rows = []
    for rule, hits in (("markdown-tolerant", string), ("anywhere", anywhere), ("judged", judged)):
        # Same generator state for both rules, so their intervals use the same resamples.
        rows += [{"rule": rule} | r for r in language_risk(hits, np.random.default_rng(0))]
    write(rows, args.out / "judged_language_risk.csv")
    write(agreement(string, labels), args.out / "judge_agreement.csv")
    write(answer_rates(args.runs), args.out / "judge_answers.csv")
    parity = parity_by_variant(args.out / "tokenizers.csv")
    adjusted = comprehension_adjusted(judged, parity, np.random.default_rng(0))
    write(adjusted, args.out / "comprehension_adjusted_judged.csv")
    for r in rows:
        if r["seed"] in ("0", "mean") and r["metric"] in ("english", "union_11", "excess_11"):
            print(
                f"  {r['rule']:18s} {r['model']:34s} {r['seed']:4s} {r['metric']:10s} "
                f"{100 * r['value']:5.1f} [{100 * r['ci_low']:.1f}, {100 * r['ci_high']:.1f}]"
            )


if __name__ == "__main__":
    main()
