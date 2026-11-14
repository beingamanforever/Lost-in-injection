from collections import Counter

from lost_in_injection.runs import MULTILINGUAL_SLOTS, NUM_PAIRS, RUNS, TRAINING_LANGUAGES, language


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
