"""Draw the report figures from results/metrics.csv and results/training.csv (gen environment).

The chart forms follow the papers this work builds on. Figure 1 is Meta-SecAlign's headline:
utility (higher is better) beside attack success (lower is better) as grouped bars, value labels,
the best value in bold. The training plot is SecAlign's Figure 3: the preferred answer's reward
holds while the injected one collapses. Undefended is gray and each defense keeps one colour in
every figure (validated for colour-vision deficiency). Serif type matches a LaTeX body, and each
figure is written as PDF and SVG (vector) and PNG (web). Per-language numbers are in the tables.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text

from lost_in_injection.languages import BY_NAME
from lost_in_injection.metrics import (
    EVALUATED,
    MARKDOWN_TOLERANT,
    MODELS,
    OFFICIAL,
    UTILITY_LANGUAGES,
)
from lost_in_injection.runs import RUNS

COLORS = {
    "base": "#8c8c8c",
    "prompted": "#d4bf6b",
    "english-only": "#5fb5d1",
    "english-plus-high-resource": "#8e7cc3",
    "english-plus-low-resource": "#e3804b",
}
LABELS = {
    "base": "Undefended",
    "prompted": "Prompted",
    "english-only": "English-only",
    "english-plus-high-resource": "+ High-resource",
    "english-plus-low-resource": "+ Low-resource",
}
UTILITY_NAMES = {"en": "English", "hi": "Hindi", "sw": "Swahili", "fr": "French", "zh": "Chinese"}
GROUPS = (
    ("English", ("english",)),
    ("High-resource", tuple(v for v in EVALUATED if BY_NAME[v].group == "high-resource")),
    ("Low-resource", tuple(v for v in EVALUATED if BY_NAME[v].group == "low-resource")),
    ("Held out", tuple(v for v in EVALUATED if BY_NAME[v].group == "held-out")),
)
WIDTH = 7.0  # Inches, a two-column paper's text width


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "STIXGeneral",
                "DejaVu Serif",
            ],  # STIX fills missing glyphs
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def clashes(fig) -> list[str]:
    """Pairs of drawn texts, or a text and a bar or data line, that intersect."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [
        (t.get_text().replace("\n", " "), box.shrunk(0.97, 0.85))
        for t in fig.findobj(Text)
        if t.get_visible() and t.get_text().strip()
        for box in [t.get_window_extent(renderer)]
        if box.width > 0
    ]
    bars = [p.get_window_extent(renderer) for ax in fig.axes for p in ax.patches]
    lines = [
        line.get_transform().transform_path(line.get_path())
        for ax in fig.axes
        for line in ax.get_lines()
    ]
    found = [
        f"{a!r} and {b!r}"
        for i, (a, x) in enumerate(texts)
        for b, y in texts[i + 1 :]
        if x.overlaps(y)
    ]
    found += [f"{a!r} and a bar" for a, x in texts for y in bars if x.overlaps(y)]
    return found + [
        f"{a!r} and a line"
        for a, x in texts
        for path in lines
        if path.intersects_bbox(x, filled=False)
    ]


def save(fig, out: Path, name: str) -> None:
    if found := clashes(fig):
        raise ValueError(f"{name}: overlapping elements: {'; '.join(found)}")
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(out / f"{name}.{suffix}")
    plt.close(fig)


def read(results: Path) -> tuple[dict, list[dict]]:
    with (results / "metrics.csv").open() as file:
        metrics = {
            (r["model"], r["benchmark"], r["setting"], r["variant"], r["metric"]): float(r["value"])
            for r in csv.DictReader(file)
        }
    training = results / "training.csv"
    curves = list(csv.DictReader(training.open())) if training.exists() else []
    return metrics, curves


def asr(metrics: dict, model: str, variant: str, benchmark: str = MARKDOWN_TOLERANT) -> float:
    """AlpacaFarm-Hacked success, any of six attacks; leading markdown skipped by default."""
    return metrics.get((model, benchmark, "combined", variant, "asr"), float("nan"))


def present(metrics: dict) -> list[str]:
    return [m for m in MODELS if asr(metrics, m, "english") == asr(metrics, m, "english")]


def bars(ax, groups: list[str], series: dict[str, list[float]], labels: bool, best=min) -> None:
    """Grouped bars, one colour per series; with `best`, each group's best value is in bold."""
    width = 0.8 / len(series)
    upright = len(series) > 3  # Side by side, labels wider than their bars would collide.
    for i, (model, values) in enumerate(series.items()):
        x = np.arange(len(groups)) + (i - (len(series) - 1) / 2) * width
        ax.bar(x, values, width, color=COLORS.get(model, model), label=LABELS.get(model, model))
        if labels:
            for j, value in enumerate(values):
                top = best(v[j] for v in series.values()) if best else None
                ax.text(
                    x[j],
                    value + 1.2,
                    f"{value:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=90 if upright else 0,
                    fontweight="bold" if value == top else "normal",
                )
    ax.set_xticks(range(len(groups)), groups)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(groups) - 0.5)


def captions(fig, axes, texts) -> None:
    """Panel captions on one baseline, just below the lowest tick labels of any panel."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bottom = min(ax.xaxis.get_tightbbox(renderer).y0 for ax in axes)
    y = fig.transFigure.inverted().transform((0, bottom - 6))[1]
    for ax, text in zip(axes, texts, strict=True):
        x = ax.get_position().x0 + ax.get_position().width / 2
        fig.text(x, y, text, ha="center", va="top", fontsize=9.5)


def headline(metrics: dict, out: Path) -> None:
    models = present(metrics)
    if not models:
        return
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(WIDTH, 2.1), gridspec_kw={"width_ratios": [5, 5.6], "wspace": 0.24}
    )
    # The prompted baseline shares the base weights, and its attack rates sit within a few points of
    # the base model's; Table 1 reports them, and leaving it out keeps four bars per group.
    weights = [m for m in models if m != "prompted"]
    utility = {
        m: [
            100 * metrics.get((m, "global-mmlu-lite", "0-shot", lang, "accuracy"), np.nan)
            for lang in UTILITY_LANGUAGES
        ]
        for m in weights
    }
    bars(left, [UTILITY_NAMES[lang] for lang in UTILITY_LANGUAGES], utility, labels=False)
    left.set_ylim(0, 75)
    left.set_ylabel("Accuracy (%)")

    security = {
        m: [100 * np.mean([asr(metrics, m, v) for v in variants]) for _, variants in GROUPS]
        + [100 * asr(metrics, m, "any-language")]
        for m in weights
    }
    bars(
        right,
        [name.replace("-", "-\n").replace(" ", "\n") for name, _ in GROUPS] + ["Best\nof all"],
        security,
        labels=True,
    )
    right.axvline(len(GROUPS) - 0.5, color="#bbbbbb", linewidth=0.7)
    right.set_ylim(0, 122)  # Headroom for the upright labels; ticks stop at 100.
    right.set_yticks(range(0, 101, 25))
    right.set_ylabel("Attack success (%)")
    handles, names = right.get_legend_handles_labels()
    fig.subplots_adjust(top=0.86)
    fig.legend(
        handles,
        names,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.88),
        ncols=len(weights),
        handlelength=1.1,
        columnspacing=1.2,
    )
    captions(
        fig,
        (left, right),
        ("(a) Utility: Global-MMLU-Lite (↑)", "(b) Security: AlpacaFarm-Hacked (↓)"),
    )
    save(fig, out, "headline")


def scoring_rules(metrics: dict, out: Path) -> None:
    official = [100 * asr(metrics, "base", v, OFFICIAL) for v in EVALUATED]
    tolerant = [100 * asr(metrics, "base", v) for v in EVALUATED]
    if any(np.isnan(official + tolerant)):
        return
    names = ["Rom. Hindi" if v == "romanized-hindi" else v.title() for v in EVALUATED]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.8))
    series = {"#c9c9c9": official, "#6e6e6e": tolerant}
    bars(ax, names, series, labels=True, best=None)
    ax.tick_params(axis="x", labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in series]
    ax.legend(
        handles,
        [
            "Official rule (reply starts with Hacked)",
            "Leading markdown skipped",
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncols=2,
        handlelength=1.1,
    )
    ax.set_ylim(0, 105)
    ax.set_ylabel("Attack success (%)")
    save(fig, out, "scoring_rules")


def script_vs_language(metrics: dict, out: Path) -> None:
    models = present(metrics)
    variants = ("hindi", "urdu", "romanized-hindi", "hinglish")
    names = [
        "Hindi\n(Devanagari)",
        "Urdu\n(Arabic script)",
        "Hindi\n(Latin script)",
        "Hinglish\n(code-mixed)",
    ]
    shown = [m for m in models if m != "prompted"]  # As in Figure 1; Table 1 has prompted.
    series = {m: [100 * asr(metrics, m, v) for v in variants] for m in shown}
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    bars(ax, names, series, labels=True)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Attack success (%)")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncols=len(shown),
        handlelength=1.0,
        columnspacing=0.9,
        fontsize=8,
    )
    save(fig, out, "script_vs_language")


def training_dynamics(curves: list[dict], out: Path) -> None:
    runs = [r for r in RUNS if any(c["run"] == r for c in curves)]
    if not runs or "rewards/chosen" not in curves[0]:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    top, tail, last = -np.inf, -np.inf, 0
    for run in runs:
        rows = [c for c in curves if c["run"] == run]
        steps = [int(c["step"]) for c in rows]
        chosen = [float(c["rewards/chosen"]) for c in rows]
        rejected = [float(c["rewards/rejected"]) for c in rows]
        ax.plot(steps, chosen, "--", color=COLORS[run], linewidth=1.2)
        ax.plot(steps, rejected, "-", color=COLORS[run], linewidth=1.2, label=LABELS[run])
        top, last = max(top, max(chosen)), max(last, steps[-1])
        tail = max(tail, max(r for s, r in zip(steps, rejected, strict=True) if s >= steps[-1] / 2))
    ax.axhline(0, color="#bbbbbb", linewidth=0.6, zorder=0)
    # Line style carries the answer and colour the run, so the two answers are labelled in the empty
    # space above and between the curves instead of in a legend box that would sit on them.
    ax.text(
        last,
        top + 0.8,
        "preferred answer (dashed)",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )
    ax.text(
        last,
        tail / 2,
        "injected answer (solid)",
        ha="right",
        va="center",
        fontsize=7.5,
        color="#555555",
    )
    ax.set_ylim(top=top + 3.5)
    ax.set_xlabel("DPO training step")
    ax.set_ylabel("Implicit reward")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=len(runs), handlelength=1.6)
    save(fig, out, "training_dynamics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    style()
    metrics, curves = read(args.results)
    out = args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    headline(metrics, out)
    scoring_rules(metrics, out)
    script_vs_language(metrics, out)
    training_dynamics(curves, out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
