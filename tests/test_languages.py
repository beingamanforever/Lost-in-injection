from lost_in_injection.languages import REGIONS, VARIANTS, in_group
from lost_in_injection.metrics import EVALUATED
from lost_in_injection.runs import FOCUS

PHASE_ONE_VARIANTS = (
    "english",
    "french",
    "chinese",
    "hindi",
    "swahili",
    "romanized-hindi",
    "hinglish",
    "urdu",
    "tamil",
    "amharic",
    "spanglish",
)

EXTRA_HELD_OUT = {
    "arabic": ("nllb", "arb_Arab"),
    "turkish": ("nllb", "tur_Latn"),
    "russian": ("nllb", "rus_Cyrl"),
    "greek": ("nllb", "ell_Grek"),
    "korean": ("nllb", "kor_Hang"),
    "thai": ("nllb", "tha_Thai"),
    "yoruba": ("nllb", "yor_Latn"),
    "bengali": ("indictrans2", "ben_Beng"),
}


def test_phase_one_variants_unchanged():
    assert EVALUATED == list(PHASE_ONE_VARIANTS)
    extra_held_out = in_group("extra-held-out")
    assert {variant.name for variant in extra_held_out} == set(EXTRA_HELD_OUT)
    for variant in extra_held_out:
        translator, code = EXTRA_HELD_OUT[variant.name]
        assert variant.translator == translator
        assert variant.code == code
        assert variant.accepted_labels == (code,)


def test_every_evaluated_variant_has_a_tiny_aya_region():
    assert {v.name for v in VARIANTS} <= set(REGIONS)
    assert set(REGIONS.values()) == FOCUS["global"]
