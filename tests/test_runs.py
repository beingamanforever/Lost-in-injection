from collections import Counter
from pathlib import Path

import pytest

from lost_in_injection.runs import (
    FAMILY,
    MULTILINGUAL_SLOTS,
    NUM_PAIRS,
    RUNS,
    TRAINING_LANGUAGES,
    base_model,
    language,
    model_path,
    split,
)


def languages_of(run):
    return Counter(language(run, slot) for slot in range(NUM_PAIRS))


def test_share_25_is_the_first_half_of_share_50_in_the_same_languages():
    for slot in range(MULTILINGUAL_SLOTS):
        quarter = language("multilingual-share-25", slot)
        half = language("english-plus-low-resource", slot)
        assert quarter == (half if slot < MULTILINGUAL_SLOTS // 2 else "english")


def test_each_run_balances_its_languages_and_keeps_half_english():
    assert languages_of("english-plus-low-resource") == {
        "english": 2500,
        "hindi": 625,
        "swahili": 625,
        "romanized-hindi": 625,
        "hinglish": 625,
    }
    assert languages_of("english-only") == {"english": NUM_PAIRS}
    assert languages_of("multilingual-share-25")["english"] == 3750


def test_every_run_language_is_translated():
    used = {lang for languages, _ in RUNS.values() for lang in languages}
    assert used == set(TRAINING_LANGUAGES)


def test_family_run_names_split_into_model_and_run():
    assert split("fire-english-only") == ("fire", "english-only")
    assert split("llama8b-meta-secalign") == ("llama8b", "meta-secalign")
    assert split("llama3b-english-only") == ("llama3b", "english-only")
    assert split("qwen-english-plus-low-resource") == ("qwen", "english-plus-low-resource")
    assert split("english-only") == ("global", "english-only")


def test_model_folders_follow_their_hub_names():
    assert model_path("global") == Path("models/tiny-aya-global")
    assert model_path("llama8b") == Path("models/llama-3.1-8b-instruct")
    assert model_path("llama3b") == Path("models/llama-3.2-3b-instruct")
    assert model_path("qwen") == Path("models/qwen3-4b-instruct-2507")
    assert base_model("llama8b-meta-secalign", "llama8b-english-only") == model_path("llama8b")


def test_base_model_rejects_runs_from_two_models():
    assert base_model("fire-english-only", "fire-english-plus-low-resource") == Path(
        "models/tiny-aya-fire"
    )
    assert base_model("english-only") == Path("models/tiny-aya-global")
    with pytest.raises(ValueError):
        base_model("fire-english-only", "english-only")
    with pytest.raises(ValueError):
        base_model("fire-english-only", "earth-english-only")


def test_family_runs_use_globals_slot_languages():
    for model in FAMILY:
        for run in RUNS:
            for slot in (0, MULTILINGUAL_SLOTS - 1, MULTILINGUAL_SLOTS, NUM_PAIRS - 1):
                assert language(f"{model}-{run}", slot) == language(run, slot)
