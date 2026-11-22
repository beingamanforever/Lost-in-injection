import csv

from lost_in_injection.defense_figures import utility_scores

FIELDS = [
    "model",
    "benchmark",
    "setting",
    "variant",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "n",
    "seed",
]


def row(model, benchmark, setting, variant, metric, value, seed):
    values = [model, benchmark, setting, variant, metric, value, "", "", "", seed]
    return dict(zip(FIELDS, values, strict=True))


def test_utility_scores_prefer_seed_zero_and_average_five_mmlu_languages(tmp_path):
    rows = [
        row("a", "ifeval", "0-shot-chat", "en", "prompt_level_strict_acc", 0.5, "1"),
        row("a", "ifeval", "0-shot-chat", "en", "prompt_level_strict_acc", 0.7, "0"),
        row("b", "ifeval", "0-shot-chat", "en", "prompt_level_strict_acc", 0.6, "1"),
        *(
            row("a", "global-mmlu-lite", "0-shot", v, "accuracy", 0.1 * i, "0")
            for i, v in enumerate(["en", "fr", "hi", "sw", "zh"], start=1)
        ),
        row("c", "global-mmlu-lite", "0-shot", "en", "accuracy", 0.9, "0"),
    ]
    path = tmp_path / "metrics.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    scores = utility_scores(path)
    assert round(scores[("a", "ifeval")], 6) == 70.0  # seed 0 wins over seed 1
    assert round(scores[("b", "ifeval")], 6) == 60.0  # seed 1 when there is no seed 0
    assert round(scores[("a", "global-mmlu-lite")], 6) == 30.0  # mean of 10..50
    assert ("c", "global-mmlu-lite") not in scores  # fewer than five languages
