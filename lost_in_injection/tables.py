"""Write the report's booktabs tables as HTML from results/metrics.csv (gen environment).

docs/figures/render.sh turns each page into a tightly cropped PDF and PNG. The best value in each
column is bold, and the attacker's best-of-all column is shaded, as in InjecAgent's Table 3.
"""

import argparse
import csv
from pathlib import Path

from lost_in_injection.figures import GROUPS, LABELS, UTILITY_NAMES
from lost_in_injection.metrics import EVALUATED, MARKDOWN_TOLERANT, MODELS, UTILITY_LANGUAGES

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><link rel="stylesheet" href="style.css">
<style>@page {{ size: 1600px 900px; }} body {{ padding: 8px; }}</style></head>
<body><table class="booktabs">{rows}</table></body></html>
"""
NAMES = {v: "Rom. Hindi" if v == "romanized-hindi" else v.title() for v in EVALUATED}


def read(path: Path) -> dict:
    with path.open() as file:
        return {
            (r["model"], r["benchmark"], r["setting"], r["variant"], r["metric"]): float(r["value"])
            for r in csv.DictReader(file)
        }


def body(models: list[str], values: dict[str, list[float]], best: list, shaded: set[int]) -> str:
    """Body rows, one per model; the best value in each column (min or max) is bold."""
    tops = [pick(values[m][j] for m in models) for j, pick in enumerate(best)]
    rows = []
    for i, model in enumerate(models):
        classes = [
            ("best " if v == tops[j] else "") + ("shade" if j in shaded else "")
            for j, v in enumerate(values[model])
        ]
        cells = "".join(
            f'<td class="{c}">{v:.1f}</td>' for c, v in zip(classes, values[model], strict=True)
        )
        place = "mid" if i == 0 else "bottom" if i == len(models) - 1 else ""
        rows.append(f'<tr class="{place}"><td class="left">{LABELS[model]}</td>{cells}</tr>')
    return "".join(rows)


def attack(metrics: dict) -> str:
    models = [m for m in MODELS if (m, MARKDOWN_TOLERANT, "combined", "english", "asr") in metrics]
    columns = [*EVALUATED, "any-language"]
    values = {
        m: [100 * metrics[(m, MARKDOWN_TOLERANT, "combined", v, "asr")] for v in columns]
        for m in models
    }
    single = [vs[0] for _, vs in GROUPS if len(vs) == 1]  # English: no second header row
    groups = "".join(
        f'<th rowspan="2">{name}</th>'
        if len(vs) == 1
        else f'<th colspan="{len(vs)}" class="cmid">{name}</th>'
        for name, vs in GROUPS
    )
    names = "".join(f"<th>{NAMES[v]}</th>" for v in EVALUATED if v not in single)
    head = (
        f'<tr class="top"><th rowspan="2" class="left">Model</th>{groups}'
        '<th rowspan="2" class="shade">Best<br>of all</th></tr>'
        f"<tr>{names}</tr>"
    )
    return head + body(models, values, [min] * len(columns), {len(columns) - 1})


def injecagent(metrics: dict) -> str:
    models = [m for m in MODELS if (m, "injecagent", "base", "english", "asr") in metrics]
    keys = [
        (lang, setting, metric)
        for lang in ("english", "hindi")
        for setting, metric in (("base", "asr"), ("enhanced", "asr"), ("enhanced", "valid_rate"))
    ]
    values = {
        m: [100 * metrics[(m, "injecagent", s, lang, k)] for lang, s, k in keys] for m in models
    }
    head = (
        '<tr class="top"><th rowspan="2" class="left">Model</th>'
        '<th colspan="3" class="cmid">English</th><th colspan="3" class="cmid">Hindi</th></tr>'
        "<tr>" + "<th>Base</th><th>Enhanced</th><th>Valid</th>" * 2 + "</tr>"
    )
    return head + body(models, values, [min, min, max] * 2, set())  # Valid: higher is better


def utility(metrics: dict) -> str:
    models = [m for m in MODELS if (m, "global-mmlu-lite", "0-shot", "en", "accuracy") in metrics]
    values = {}
    for m in models:
        scores = [
            100 * metrics[(m, "global-mmlu-lite", "0-shot", lang, "accuracy")]
            for lang in UTILITY_LANGUAGES
        ]
        values[m] = [*scores, sum(scores) / len(scores)]
    names = "".join(f"<th>{UTILITY_NAMES[lang]}</th>" for lang in UTILITY_LANGUAGES)
    head = f'<tr class="top"><th class="left">Model</th>{names}<th class="shade">Mean</th></tr>'
    return head + body(
        models, values, [max] * (len(UTILITY_LANGUAGES) + 1), {len(UTILITY_LANGUAGES)}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures/src"))
    args = parser.parse_args()
    metrics = read(args.results / "metrics.csv")
    for name, build in (
        ("table_attack", attack),
        ("table_injecagent", injecagent),
        ("table_utility", utility),
    ):
        (args.out / f"{name}.html").write_text(PAGE.format(rows=build(metrics)))
    print(f"tables written to {args.out}")


if __name__ == "__main__":
    main()
