"""Write the report's booktabs tables as HTML from results/metrics.csv (gen environment).

docs/figures/render.sh turns each page into a tightly cropped PDF and PNG. The best value in each
column is bold, and the attacker's best-of-all column is shaded, as in InjecAgent's Table 3.
"""

import argparse
import csv
import statistics
from pathlib import Path

from lost_in_injection.figures import GROUPS, LABELS, UTILITY_NAMES, asr
from lost_in_injection.languages import in_group
from lost_in_injection.metrics import EVALUATED, MARKDOWN_TOLERANT, MODELS, UTILITY_LANGUAGES
from lost_in_injection.runs import FAMILY, RUNS, split

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><link rel="stylesheet" href="style.css">
<style>@page {{ size: 1600px 900px; }} body {{ padding: 8px; }}</style></head>
<body><table class="booktabs">{rows}</table></body></html>
"""
NAMES = {v: "Rom. Hindi" if v == "romanized-hindi" else v.title() for v in EVALUATED}
HELD_OUT_VARIANTS = tuple(v.name for v in in_group("held-out"))
EXTRA_HELD_OUT_VARIANTS = tuple(v.name for v in in_group("extra-held-out"))
FAMILY_RECIPES = ("base", "english-only", "english-plus-low-resource")
# ISO 639-1, for the generation-quality benchmarks; languages.py only carries FLORES-200 codes.
ISO_639_1 = {
    "english": "en", "french": "fr", "chinese": "zh", "swahili": "sw", "hindi": "hi",
    "arabic": "ar", "turkish": "tr", "russian": "ru", "greek": "el", "korean": "ko",
    "thai": "th", "bengali": "bn",
}  # fmt: skip


def read_raw(path: Path) -> list[dict]:
    with path.open() as file:
        return list(csv.DictReader(file))


def to_metrics(rows: list[dict], seed: str = "0") -> dict:
    return {
        (r["model"], r["benchmark"], r["setting"], r["variant"], r["metric"]): float(r["value"])
        for r in rows
        if r.get("seed", "0") == seed
    }


def read(path: Path, seed: str = "0") -> dict:
    return to_metrics(read_raw(path), seed)


def body(
    models: list[str],
    values: dict[str, list[float]],
    best: list,
    shaded: set[int],
    text: dict[str, list[str]] | None = None,
) -> str:
    """Body rows, one per model; the best value in each column (min or max) is bold.

    A missing (NaN) value shows "–" and is left out of the best choice; a `None` entry in `best`
    never bolds its column. `text`, when given, overrides a model's displayed cell strings (a
    seed's "mean ± spread", say), while `values` still decides the bold-best choice and "–".
    """
    columns = [[values[m][j] for m in models] for j in range(len(best))]
    tops = [
        pick(v for v in column if v == v) if pick and any(v == v for v in column) else None
        for pick, column in zip(best, columns, strict=True)
    ]
    rows = []
    for i, model in enumerate(models):
        cells = []
        for j, v in enumerate(values[model]):
            classes = ("best " if v == tops[j] else "") + ("shade" if j in shaded else "")
            shown = text[model][j] if text else ("–" if v != v else f"{v:.1f}")
            cells.append(f'<td class="{classes}">{shown}</td>')
        place = "mid" if i == 0 else "bottom" if i == len(models) - 1 else ""
        rows.append(
            f'<tr class="{place}"><td class="left">{LABELS[model]}</td>{"".join(cells)}</tr>'
        )
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


def extra_languages(metrics: dict) -> str:
    models = [
        m
        for m in MODELS
        if (m, MARKDOWN_TOLERANT, "combined", "any-language-extended", "asr") in metrics
    ]
    if not models:
        return ""
    columns = [*EXTRA_HELD_OUT_VARIANTS, "any-extra-held-out", "any-language-extended"]
    values = {m: [100 * asr(metrics, m, v) for v in columns] for m in models}
    names = "".join(f"<th>{v.title()}</th>" for v in EXTRA_HELD_OUT_VARIANTS)
    head = (
        f'<tr class="top"><th class="left">Model</th>{names}'
        '<th class="shade">Any extra<br>held out</th><th class="shade">Any<br>language</th></tr>'
    )
    return head + body(models, values, [min] * len(columns), {len(columns) - 2, len(columns) - 1})


def family_model(fam: str, recipe: str) -> str:
    return recipe if fam == "global" else f"{fam}-{recipe}"


def family(metrics: dict) -> str:
    def present(fam: str, recipe: str) -> bool:
        return (
            family_model(fam, recipe),
            MARKDOWN_TOLERANT,
            "combined",
            "english",
            "asr",
        ) in metrics

    if not any(present(fam, recipe) for fam in FAMILY for recipe in FAMILY_RECIPES):
        return ""  # No regional model yet: a Global-only row would defeat this table's point.
    models = [
        family_model(fam, recipe)
        for fam in ("global", *FAMILY)
        for recipe in FAMILY_RECIPES
        if present(fam, recipe)
    ]

    def mean_asr(m: str, variants: tuple[str, ...]) -> float:
        return 100 * sum(asr(metrics, m, v) for v in variants) / len(variants)

    def focus(m: str, side: str) -> float:
        return 100 * asr(metrics, m, side) if split(m)[0] != "global" else float("nan")

    values = {
        m: [
            100 * asr(metrics, m, "english"),
            mean_asr(m, HELD_OUT_VARIANTS),
            mean_asr(m, EXTRA_HELD_OUT_VARIANTS),
            focus(m, "in-focus-mean"),
            focus(m, "out-of-focus-mean"),
            100 * asr(metrics, m, "any-language"),
            100 * asr(metrics, m, "any-language-extended"),
            100
            * sum(
                metrics.get((m, "global-mmlu-lite", "0-shot", lang, "accuracy"), float("nan"))
                for lang in UTILITY_LANGUAGES
            )
            / len(UTILITY_LANGUAGES),
            100
            * metrics.get(
                (m, "ifeval", "0-shot-chat", "en", "prompt_level_strict_acc"), float("nan")
            ),
        ]
        for m in models
    }
    head = (
        '<tr class="top"><th class="left">Model</th><th>English</th>'
        "<th>Held-out<br>mean</th><th>Extra held-out<br>mean</th>"
        "<th>In-focus<br>mean</th><th>Out-of-focus<br>mean</th>"
        '<th class="shade">Best<br>of 11</th><th class="shade">Best<br>of 19</th>'
        "<th>Global-MMLU<br>mean</th><th>IFEval</th></tr>"
    )
    return head + body(models, values, [min, min, min, min, min, min, min, max, max], {6})


def asr_keys(model: str, variants: tuple[str, ...]) -> list[tuple]:
    return [(model, MARKDOWN_TOLERANT, "combined", v, "asr") for v in variants]


def mmlu_keys(model: str) -> list[tuple]:
    return [(model, "global-mmlu-lite", "0-shot", lang, "accuracy") for lang in UTILITY_LANGUAGES]


def per_seed_mean(by_seed: dict[str, dict], keys: list[tuple]) -> dict[str, float]:
    """Each seed's mean across `keys` (e.g. the held-out variants), when every key is in it."""
    return {
        seed: sum(metrics[k] for k in keys) / len(keys)
        for seed, metrics in by_seed.items()
        if all(k in metrics for k in keys)
    }


def seed_stats(per_seed: dict[str, float]) -> tuple[float, float | None, int]:
    """A column's mean, its spread across seeds (needs 2 or more), and how many seeds back it."""
    seeds = [per_seed[s] for s in ("0", "1", "2") if s in per_seed]
    if len(seeds) >= 2:
        return per_seed.get("mean", sum(seeds) / len(seeds)), statistics.stdev(seeds), len(seeds)
    if "0" in per_seed:
        return per_seed["0"], None, 1
    return float("nan"), None, 0


def seed_text(mean: float, sd: float | None) -> str:
    if mean != mean:
        return "–"
    return f"{mean:.1f}" if sd is None else f"{mean:.1f} ± {sd:.1f}"


def seeds_ablations(by_seed: dict[str, dict]) -> str:
    recipes = [
        m
        for m in RUNS
        if (m, MARKDOWN_TOLERANT, "combined", "english", "asr") in by_seed.get("0", {})
    ]
    if not recipes:
        return ""
    values, texts = {}, {}
    for m in recipes:
        stats = [
            seed_stats(per_seed_mean(by_seed, asr_keys(m, ("english",)))),
            seed_stats(per_seed_mean(by_seed, asr_keys(m, HELD_OUT_VARIANTS))),
            seed_stats(per_seed_mean(by_seed, asr_keys(m, ("any-language",)))),
            seed_stats(per_seed_mean(by_seed, asr_keys(m, ("any-language-extended",)))),
            seed_stats(per_seed_mean(by_seed, mmlu_keys(m))),
        ]
        seeds = max((n for _, _, n in stats), default=0)
        values[m] = [float(seeds), *(mean * 100 for mean, _, _ in stats)]
        texts[m] = [str(seeds)] + [
            seed_text(mean * 100, sd * 100 if sd is not None else None) for mean, sd, _ in stats
        ]
    head = (
        '<tr class="top"><th class="left">Recipe</th><th>Seeds</th><th>English</th>'
        '<th>Held-out<br>mean</th><th class="shade">Best<br>of 11</th>'
        '<th class="shade">Best<br>of 19</th><th>Global-MMLU<br>mean</th></tr>'
    )
    return head + body(recipes, values, [None, min, min, min, min, max], {4}, text=texts)


def generation(metrics: dict) -> str:
    metric = "prompt_level_strict_acc"
    models = [m for m in MODELS if (m, "ifeval", "0-shot-chat", "en", metric) in metrics]
    if not models:
        return ""

    def marco(m: str, lang: str) -> float:
        key = (m, "marco-bench-mif", "0-shot-chat", ISO_639_1[lang], metric)
        return 100 * metrics.get(key, float("nan"))

    values = {
        m: [
            100 * metrics.get((m, "ifeval", "0-shot-chat", "en", metric), float("nan")),
            100 * metrics.get((m, "indicifeval", "0-shot-chat", "hi", metric), float("nan")),
            marco(m, "french"),
            marco(m, "chinese"),
            marco(m, "swahili"),
            sum(marco(m, v) for v in EXTRA_HELD_OUT_VARIANTS) / len(EXTRA_HELD_OUT_VARIANTS),
        ]
        for m in models
    }
    head = (
        '<tr class="top"><th class="left">Model</th><th>IFEval<br>(en)</th>'
        "<th>IndicIFEval<br>(hi)</th><th>Marco<br>(fr)</th><th>Marco<br>(zh)</th>"
        '<th>Marco<br>(sw)</th><th class="shade">Marco<br>mean</th></tr>'
    )
    return head + body(models, values, [max] * 6, {5})


ADAPTIVE_K = (1, 11, 21, 31, 41, 51)


def adaptive(metrics: dict) -> str:
    models = [
        m for m in MODELS if (m, "alpacafarm-adaptive", "pair", "1-queries", "asr") in metrics
    ]
    if not models:
        return ""
    values = {
        m: [
            *(
                100
                * metrics.get(
                    (m, "alpacafarm-adaptive", "pair", f"{k}-queries", "asr"), float("nan")
                )
                for k in ADAPTIVE_K
            ),
            metrics.get(
                (m, "alpacafarm-adaptive", "pair", "first-success", "median_queries"), float("nan")
            ),
        ]
        for m in models
    }
    names = "".join(f"<th>{k}</th>" for k in ADAPTIVE_K)
    head = (
        f'<tr class="top"><th class="left">Target</th>{names}'
        '<th class="shade">Median<br>queries</th></tr>'
    )
    return head + body(models, values, [min] * len(ADAPTIVE_K) + [max], {len(ADAPTIVE_K)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures/src"))
    args = parser.parse_args()
    rows = read_raw(args.results / "metrics.csv")
    metrics = to_metrics(rows)
    by_seed = {seed: to_metrics(rows, seed) for seed in ("0", "1", "2", "mean")}
    pages = {
        "table_attack": attack(metrics),
        "table_injecagent": injecagent(metrics),
        "table_utility": utility(metrics),
        "table_extra_languages": extra_languages(metrics),
        "table_family": family(metrics),
        "table_seeds_ablations": seeds_ablations(by_seed),
        "table_generation": generation(metrics),
        "table_adaptive": adaptive(metrics),
    }
    for name, page_rows in pages.items():
        path = args.out / f"{name}.html"
        if page_rows:
            path.write_text(PAGE.format(rows=page_rows))
        else:
            path.unlink(missing_ok=True)  # No stale table once its rows are gone.
    print(f"tables written to {args.out}")


if __name__ == "__main__":
    main()
