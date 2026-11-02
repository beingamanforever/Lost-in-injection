from lost_in_injection import translate
from lost_in_injection.languages import BY_NAME

CANDIDATES = [
    "इसे भेजें",  # loses the placeholder
    "Send it to [1]",  # still English
    "इसे [1] पर भेजें",  # back-translation too far from the source
    "इसे [1] को भेजें",  # first to pass every gate
    "इसे अभी [1] पर भेजें",  # also passes, but ranks lower
]
SIMILARITY = {"इसे [1] पर भेजें": 0.5, "इसे [1] को भेजें": 0.9, "इसे अभी [1] पर भेजें": 0.95}


def test_best_ranked_candidate_passing_every_gate_is_kept(monkeypatch):
    def generate(texts, src, tgt, translator, num_beams, num_return):
        return [CANDIDATES] if src == translate.ENGLISH else [[text] for text in texts]

    def top_labels(text):
        return ["hin_Deva", "urd_Arab", "eng_Latn"] if "भेजें" in text else ["eng_Latn"]

    monkeypatch.setattr(translate, "generate", generate)
    monkeypatch.setattr(translate, "top_labels", top_labels)
    monkeypatch.setattr(translate, "similarity", lambda _, back: [SIMILARITY[b] for b in back])

    [record] = translate.translate(["Send it to amy.watson@gmail.com"], BY_NAME["hindi"])

    assert record["translation"] == "इसे amy.watson@gmail.com को भेजें"
    assert [c["restored"] is not None for c in record["candidates"]] == [False, *[True] * 4]
    assert [c.get("similarity") for c in record["candidates"]] == [None, None, 0.5, 0.9, 0.95]
