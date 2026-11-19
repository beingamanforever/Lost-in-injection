import pytest

from lost_in_injection.records import write_jsonl
from lost_in_injection.translation_quality import inputs, summarize


def record(source, translation, similarity=0.9):
    candidate = {"restored": translation, "similarity": similarity}
    return {"source": source, "translation": translation, "candidates": [candidate]}


def test_inputs_keep_machine_translations_only(tmp_path):
    write_jsonl(tmp_path / "eval/swahili/translations.jsonl", [record("Hi", "Habari")])
    write_jsonl(tmp_path / "pairs/swahili/translations.jsonl", [record("Bye", None)])
    write_jsonl(tmp_path / "eval/hinglish/translations.jsonl", [record("Hi", "Hi yaar")])
    rows = inputs(tmp_path / "eval", tmp_path / "pairs")
    assert [(r["set"], r["language"], r["hypothesis"], r["reference"]) for r in rows] == [
        ("eval", "swahili", "Habari", "")
    ]


def test_summarize_reports_each_set_and_language():
    rows = [
        {"set": "pairs", "language": "hindi", "prediction": p, "similarity": s}
        for p, s in ((1.0, 0.95), (2.0, 0.9), (6.0, 0.8))
    ]
    (row,) = summarize(rows)
    assert (row["n"], row["metricx_mean"], row["metricx_median"]) == (3, 3.0, 2.0)
    assert row["spearman_vs_labse"] == pytest.approx(-1.0)
