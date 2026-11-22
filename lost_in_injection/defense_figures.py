"""Draw the figures for the defense comparison from results/acl and results/metrics.csv.

Each figure answers one question with one picture: which defenses leak when the attacker can
switch language (dumbbells, English beside the best of 11 languages), where the input filter is
blind (a heatmap), whether languages beat plain rewordings at the same number of tries (curves),
how much the scoring rule moves the result (dots per rule), what low-resource training buys
(arrows per model family) and what each defense costs (change against the same model undefended).
Style, palette and the overlap check are figures.py's, so every figure in the report matches.

Usage: python -m lost_in_injection.defense_figures [--acl results/acl] [--out results/figures]
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from lost_in_injection.figures import COLORS, WIDTH, plt, save, style

ENGLISH_ONLY, LOW_RESOURCE, UNDEFENDED = (
    COLORS["english-only"],
    COLORS["english-plus-low-resource"],
    COLORS["base"],
)
INK, GRID = "#1b1b1b", "#e6e6e6"
MODELS = [
    ("global", "Tiny Aya Global (3B)"),
    ("qwen", "Qwen3 (4B)"),
    ("llama8b", "Llama 3.1 (8B)"),
    ("llama70b", "Llama 3.3 (70B)"),
]
DEFENSES = [
    ("base", "No defense"),
    ("prompted", "Reminder prompt"),
    ("sandwich", "Sandwich prompt"),
    ("isolation", "Isolation prompt"),
    ("promptguard", "Prompt Guard 2 filter"),
    ("meta-secalign", "Meta-SecAlign"),
    ("english-only", "Our DPO, English"),
    ("english-plus-low-resource", "Our DPO, + low-resource"),
]
LANGUAGES = [
    ("english", "English"),
    ("french", "French"),
    ("chinese", "Chinese"),
    ("hindi", "Hindi"),
    ("swahili", "Swahili"),
    ("romanized-hindi", "Hindi\n(Latin)"),
    ("hinglish", "Hinglish"),
    ("urdu", "Urdu"),
    ("tamil", "Tamil"),
    ("amharic", "Amharic"),
    ("spanglish", "Spanglish"),
]


def model_name(family: str, run: str) -> str:
    return run if family == "global" else f"{family}-{run}"


def read(path: Path) -> list[dict]:
    with path.open() as file:
        return list(csv.DictReader(file))


def rate(rows: list[dict], model: str, metric: str) -> float | None:
    """A model's rate in %, its seed mean where it has several seeds."""
    picked = {r["seed"]: r for r in rows if r["model"] == model and r["metric"] == metric}
    row = picked.get("mean") or picked.get("0")
    return 100 * float(row["value"]) if row else None


def rows_axes(ax, labels: list[str]) -> None:
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def defense_gap(acl: Path, out: Path, rule: str) -> None:
    """English attack success beside the rate when the attacker may try all 11 languages."""
    rows = [r for r in read(acl / "judged_language_risk.csv") if r["rule"] == rule]
    fig, axes = plt.subplots(1, len(MODELS), figsize=(WIDTH, 2.7), sharey=True)
    for ax, (family, title) in zip(axes, MODELS, strict=True):
        for i, (run, _) in enumerate(DEFENSES):
            english = rate(rows, model_name(family, run), "english")
            union = rate(rows, model_name(family, run), "union_11")
            if english is None:
                continue
            ax.plot([english, union], [i, i], color=LOW_RESOURCE, alpha=0.45, linewidth=2.4)
            ax.plot(english, i, "o", mfc="white", mec=INK, mew=0.9, markersize=5)
            ax.plot(union, i, "o", color=LOW_RESOURCE, markersize=5)
        ax.set_title(title)
        ax.set_xlim(-4, 104)
        ax.set_xticks([0, 50, 100])
        rows_axes(ax, [label for _, label in DEFENSES])
    fig.supxlabel("Cases hijacked (%)", fontsize=9, y=-0.06)
    handles = [
        plt.Line2D([], [], marker="o", mfc="white", mec=INK, linestyle="none", label="English"),
        plt.Line2D(
            [],
            [],
            marker="o",
            color=LOW_RESOURCE,
            linestyle="none",
            label="Attacker tries all 11 languages",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.6, 1.04))
    fig.subplots_adjust(wspace=0.12)
    save(fig, out, "defense_gap" if rule == "judged" else f"defense_gap_{rule}")


# Languages Prompt Guard 2's model card says it was evaluated on, among our variants
# (the card lists English, French, German, Hindi, Italian, Portuguese, Spanish and Thai).
PROMPT_GUARD_CARD = {"english", "french", "hindi", "spanglish"}


def promptguard_languages(acl: Path, out: Path) -> None:
    """Share of each language's attacks that Prompt Guard 2 removes, one try per language.

    † marks languages on Prompt Guard 2's model card. Blank where the undefended model follows
    that language's injections less than half as often as English ones: there is too little
    attack success left to measure a removal.
    """
    removed = {
        (r["model"], r["language"]): 100 * float(r["relative_reduction"])
        for r in read(acl / "comprehension_adjusted.csv")
        if r["seed"] == "0" and r["relative_reduction"] and r["reads_language"] == "True"
    }
    grid = np.array(
        [
            [removed.get((model_name(f, "promptguard"), lang), np.nan) for lang, _ in LANGUAGES]
            for f, _ in MODELS
        ]
    )
    fig, ax = plt.subplots(figsize=(WIDTH, 1.9))
    ax.imshow(np.ma.masked_invalid(grid), cmap="Blues", vmin=0, vmax=110, aspect="auto")
    for (i, j), value in np.ndenumerate(grid):
        label, colour = ("n/a", "#a6a6a6") if np.isnan(value) else (f"{value:.0f}", INK)
        ax.text(
            j,
            i,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            color="white" if value > 60 else colour,
        )
    labels = [name + ("†" if lang in PROMPT_GUARD_CARD else "") for lang, name in LANGUAGES]
    ax.set_xticks(range(len(LANGUAGES)), labels, fontsize=8)
    ax.set_yticks(range(len(MODELS)), [title for _, title in MODELS])
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    save(fig, out, "promptguard_languages")


TRIES = [
    ("best-of-n", UNDEFENDED, (0, (2, 1.5)), "English, character noise (Best-of-N)"),
    ("sampling", UNDEFENDED, "-", "English, temperature samples"),
    ("paraphrase", ENGLISH_ONLY, "-", "English, LLM paraphrases"),
    ("translation", INK, "-", "Translations into 11 languages"),
]
TRY_PANELS = [
    ("base", "No defense"),
    ("english-only", "Our DPO, English"),
    ("english-plus-low-resource", "Our DPO, + low-resource"),
]


def equal_budget(acl: Path, out: Path) -> None:
    """Cases hijacked by k random tries of each kind on Tiny Aya Global, the same 208 cases."""
    rows = read(acl / "equal_budget.csv")
    fig, axes = plt.subplots(1, len(TRY_PANELS), figsize=(WIDTH, 2.2))
    for ax, (model, title) in zip(axes, TRY_PANELS, strict=True):
        for family, colour, dash, _ in TRIES:
            picked = sorted(
                (r for r in rows if r["model"] == model and r["family"] == family),
                key=lambda r: int(r["k"]),
            )
            if not picked:
                continue
            k = np.arange(1, len(picked) + 1)
            value = np.array([100 * float(r["union"]) for r in picked])
            if family in ("translation", "paraphrase"):
                low = [100 * float(r["union_low"]) for r in picked]
                high = [100 * float(r["union_high"]) for r in picked]
                ax.fill_between(k, low, high, color=colour, alpha=0.12, linewidth=0)
            ax.plot(
                k,
                value,
                color=colour,
                linestyle=dash,
                linewidth=1.8 if family == "translation" else 1.2,
            )
        ax.set_title(title)
        ax.set_xticks([1, 3, 5, 7, 9, 11])
        ax.set_xlim(1, 11)
        ax.set_ylim(0, None)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Cases hijacked (%)")
    axes[1].set_xlabel("Tries per case")
    handles = [
        plt.Line2D([], [], color=c, linestyle=d, linewidth=1.8 if f == "translation" else 1.2)
        for f, c, d, _ in TRIES
    ]
    fig.tight_layout(w_pad=1.5)
    fig.legend(
        handles[::-1],
        [label for *_, label in TRIES][::-1],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.03),
    )
    save(fig, out, "equal_budget")


SCORED = [
    ("base", "Tiny Aya, no defense"),
    ("promptguard", "Tiny Aya, Prompt Guard 2"),
    ("english-only", "Tiny Aya, our DPO"),
    ("qwen-english-only", "Qwen3, our DPO"),
    ("llama8b-english-only", "Llama 8B, our DPO"),
    ("llama8b-meta-secalign", "Llama 8B, Meta-SecAlign"),
    ("llama70b-meta-secalign", "Llama 70B, Meta-SecAlign"),
]
RULES = [
    ("markdown-tolerant", UNDEFENDED, "o", "Reply starts with “Hacked”"),
    ("judged", LOW_RESOURCE, "D", "LLM judge: the model delivered the payload"),
    ("anywhere", ENGLISH_ONLY, "s", "“Hacked” anywhere in the reply"),
]


def scoring_rules(acl: Path, out: Path) -> None:
    """The same replies under three definitions of a hijack."""
    rows = read(acl / "judged_language_risk.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4), sharey=True)
    for ax, (metric, title) in zip(
        axes, [("english", "English"), ("union_11", "Attacker tries all 11 languages")], strict=True
    ):
        for i, (model, _) in enumerate(SCORED):
            values = [
                rate([r for r in rows if r["rule"] == rule], model, metric) for rule, *_ in RULES
            ]
            ax.plot([min(values), max(values)], [i, i], color=GRID, linewidth=2.4)
            for (_, colour, marker, label), value in zip(RULES, values, strict=True):
                ax.plot(
                    value, i, marker, color=colour, markersize=5, label=label if i == 0 else None
                )
        ax.set_title(title)
        ax.set_xlim(-3, 103)
        ax.set_xticks([0, 25, 50, 75, 100])
        rows_axes(ax, [label for _, label in SCORED])
    fig.supxlabel("Cases hijacked (%)", fontsize=9, y=-0.06)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.58, 1.06))
    save(fig, out, "scoring_rules_three")


FAMILIES = [
    ("global", "Tiny Aya Global"),
    ("fire", "Tiny Aya Fire"),
    ("earth", "Tiny Aya Earth"),
    ("water", "Tiny Aya Water"),
    ("qwen", "Qwen3 (4B)"),
    ("llama8b", "Llama 3.1 (8B)"),
]


def low_resource(acl: Path, out: Path) -> None:
    """Rate over all 11 languages: English-only training, then with low-resource injections."""
    rows = read(acl / "judged_language_risk.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.0), sharey=True)
    for ax, (rule, title) in zip(
        axes,
        [("markdown-tolerant", "Reply starts with “Hacked”"), ("judged", "LLM judge")],
        strict=True,
    ):
        picked = [r for r in rows if r["rule"] == rule]
        for i, (family, _) in enumerate(FAMILIES):
            before = rate(picked, model_name(family, "english-only"), "union_11")
            after = rate(picked, model_name(family, "english-plus-low-resource"), "union_11")
            if abs(after - before) > 1.5:
                ax.annotate(
                    "",
                    xy=(after, i),
                    xytext=(before, i),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": "#9a9a9a",
                        "lw": 1.1,
                        "shrinkA": 4,
                        "shrinkB": 4,
                    },
                )
            ax.plot(before, i, "o", color=ENGLISH_ONLY, markersize=5.5)
            ax.plot(after, i, "o", color=LOW_RESOURCE, markersize=5.5)
        ax.set_title(title)
        ax.set_xlim(-3, 90)
        ax.set_xticks([0, 20, 40, 60, 80])
        rows_axes(ax, [label for _, label in FAMILIES])
    fig.supxlabel(
        "Cases hijacked when the attacker tries all 11 languages (%)", fontsize=9, y=-0.08
    )
    handles = [
        plt.Line2D(
            [], [], marker="o", color=ENGLISH_ONLY, linestyle="none", label="Trained on English"
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            color=LOW_RESOURCE,
            linestyle="none",
            label="+ Hindi, Swahili, romanized Hindi, Hinglish",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.58, 1.15))
    save(fig, out, "low_resource")


COSTS = [
    ("english-only", "base", "Tiny Aya, our DPO"),
    ("english-plus-low-resource", "base", "Tiny Aya, our DPO + low-res."),
    ("qwen-english-only", "qwen-base", "Qwen3, our DPO"),
    ("qwen-english-plus-low-resource", "qwen-base", "Qwen3, our DPO + low-res."),
    ("llama8b-meta-secalign", "llama8b-base", "Llama 8B, Meta-SecAlign"),
    ("llama8b-english-only", "llama8b-base", "Llama 8B, our DPO"),
    ("llama8b-english-plus-low-resource", "llama8b-base", "Llama 8B, our DPO + low-res."),
    ("llama70b-meta-secalign", "llama70b-base", "Llama 70B, Meta-SecAlign"),
]
BENCHMARKS = [
    ("ifeval", "en", "prompt_level_strict_acc", "IFEval (English)"),
    ("indicifeval", "hi", "prompt_level_strict_acc", "IndicIFEval (Hindi)"),
    ("global-mmlu-lite", None, "accuracy", "Global-MMLU-Lite (5 languages)"),
]


def utility_scores(metrics_csv: Path) -> dict:
    """(model, benchmark) -> score in %, seed 0 where it exists (Tiny Aya's defended generation
    runs exist only for seed 1); Global-MMLU-Lite is the mean of its five languages."""
    scores, mmlu = {}, {}
    for r in sorted(read(metrics_csv), key=lambda r: r.get("seed", "0"), reverse=True):
        if r.get("seed", "0") not in ("0", "1", "2") or r["setting"] not in (
            "0-shot",
            "0-shot-chat",
        ):
            continue
        for benchmark, variant, metric, _ in BENCHMARKS:
            if r["benchmark"] == benchmark and r["metric"] == metric:
                if variant is None:
                    mmlu.setdefault(r["model"], {})[r["variant"]] = 100 * float(r["value"])
                elif r["variant"] == variant:
                    scores[(r["model"], benchmark)] = 100 * float(r["value"])
    for model, by_language in mmlu.items():
        if len(by_language) == 5:
            scores[(model, "global-mmlu-lite")] = float(np.mean(list(by_language.values())))
    return scores


def utility_change(metrics_csv: Path, out: Path) -> None:
    """Each defense's score minus the same model's undefended score, per benchmark."""
    scores = utility_scores(metrics_csv)
    rows = [c for c in COSTS if any((c[0], b) in scores for b, *_ in BENCHMARKS)]
    fig, axes = plt.subplots(1, len(BENCHMARKS), figsize=(WIDTH, 2.3), sharey=True)
    for ax, (benchmark, _, _, title) in zip(axes, BENCHMARKS, strict=True):
        ax.axvspan(-2.5, 2.5, color="#f2f2f2", linewidth=0)
        ax.axvline(0, color="#9a9a9a", linewidth=0.8)
        for i, (model, base, _) in enumerate(rows):
            if (model, benchmark) not in scores or (base, benchmark) not in scores:
                continue
            change = scores[(model, benchmark)] - scores[(base, benchmark)]
            colour = (
                UNDEFENDED
                if "meta-secalign" in model
                else (LOW_RESOURCE if "low-resource" in model else ENGLISH_ONLY)
            )
            ax.plot([0, change], [i, i], color=colour, linewidth=2.2, alpha=0.6)
            ax.plot(change, i, "o", color=colour, markersize=5)
        ax.set_title(title)
        ax.set_xlim(-18, 6)
        ax.set_xticks([-15, -10, -5, 0, 5])
        rows_axes(ax, [label for *_, label in rows])
    fig.supxlabel(
        "Change against the same model undefended (points; shaded: within ±2.5)",
        fontsize=9,
        y=-0.08,
    )
    save(fig, out, "utility_change")


TOKENIZERS = [
    ("tiny-aya", "Tiny Aya (all four models)", INK, "o"),
    ("qwen3", "Qwen3", ENGLISH_ONLY, "s"),
    ("llama-3.1", "Llama 3.1 and 3.3", LOW_RESOURCE, "^"),
]


def tokenizer_parity(acl: Path, out: Path) -> None:
    """Tokens each tokenizer needs for the same attack text, relative to English."""
    rows = read(acl / "tokenizers.csv")
    fig, ax = plt.subplots(figsize=(WIDTH, 1.9))
    for offset, (name, label, colour, marker) in zip((-0.2, 0, 0.2), TOKENIZERS, strict=True):
        parity = {r["variant"]: float(r["token_parity"]) for r in rows if r["tokenizer"] == name}
        ax.plot(
            np.arange(len(LANGUAGES)) + offset,
            [parity[v] for v, _ in LANGUAGES],
            marker,
            color=colour,
            markersize=5,
            linestyle="none",
            label=label,
        )
    ax.axhline(1, color="#9a9a9a", linewidth=0.8)
    ax.set_xticks(range(len(LANGUAGES)), [name for _, name in LANGUAGES], fontsize=8)
    ax.set_ylabel("Tokens vs. English")
    ax.set_ylim(0, 8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    save(fig, out, "tokenizer_parity")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--acl", type=Path, default=Path("results/acl"))
    parser.add_argument("--metrics", type=Path, default=Path("results/metrics.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    style()
    args.out.mkdir(parents=True, exist_ok=True)
    defense_gap(args.acl, args.out, "judged")
    defense_gap(args.acl, args.out, "markdown-tolerant")
    promptguard_languages(args.acl, args.out)
    equal_budget(args.acl, args.out)
    scoring_rules(args.acl, args.out)
    low_resource(args.acl, args.out)
    utility_change(args.metrics, args.out)
    tokenizer_parity(args.acl, args.out)


if __name__ == "__main__":
    main()
