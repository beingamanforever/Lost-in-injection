import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lost_in_injection.figures import clashes  # noqa: E402


def test_clashes_catches_crowded_tick_labels_and_labels_on_bars():
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.bar([0, 1, 2], [1, 9, 1])
    ax.set_xticks([0, 1, 2], ["English", "High-resource", "Low-resource"])
    ax.text(1, 5, "inside a bar")
    found = clashes(fig)
    assert any("High-resource" in pair and "Low-resource" in pair for pair in found)
    assert any("inside a bar" in pair for pair in found)


def test_clashes_passes_a_clean_figure():
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.bar([0, 1], [1, 2])
    ax.set_xticks([0, 1], ["A", "B"])
    assert clashes(fig) == []


def test_clashes_catches_a_legend_on_a_line():
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 10], [0, 0], label="a long legend entry")
    ax.text(5, 0, "label on the line")
    assert any("label on the line" in pair and "a line" in pair for pair in clashes(fig))
