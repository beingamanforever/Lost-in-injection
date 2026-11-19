"""Language variants: how each one is produced and the role it plays in the experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    name: str
    # "none", "nllb" or "indictrans2" (machine translation), "xlit" (IndicXlit romanization of
    # `rewrite_from`), "codemix" (`rewrite_from` with words switched back to English using
    # alignments to `align_from`), or "aya" (only the eval-set fallback, eval_sets.py)
    translator: str
    code: str  # FLORES-200 code of the text's language and script
    accepted_labels: tuple[str, ...]  # GlotLID labels; one must be in the top 3 (not checked
    # for xlit and codemix, which inherit their base's language ID)
    group: str  # "english", "high-resource", "low-resource", "held-out", "extra-held-out" or
    # "intermediate". "extra-held-out" adds seven held-out languages (AlpacaFarm only, no
    # InjecAgent) beyond the original "held-out" group.
    rewrite_from: str = ""
    align_from: str = ""
    rewrite_instruction: str = ""  # Aya fallback only
    code_mixed: bool = False  # Also needs English words alongside the accepted language


# Latin-script Hindi and Urdu are one spoken language, and GlotLID labels it either way.
HINDUSTANI_LATIN = ("hin_Latn", "urd_Latn")

VARIANTS = (
    Variant("english", "none", "eng_Latn", ("eng_Latn",), "english"),
    Variant("french", "nllb", "fra_Latn", ("fra_Latn",), "high-resource"),
    Variant("chinese", "nllb", "zho_Hans", ("cmn_Hani",), "high-resource"),
    Variant("hindi", "indictrans2", "hin_Deva", ("hin_Deva",), "low-resource"),
    Variant("swahili", "nllb", "swh_Latn", ("swh_Latn",), "low-resource"),
    Variant(
        "romanized-hindi",
        "xlit",
        "hin_Latn",
        HINDUSTANI_LATIN,
        "low-resource",
        rewrite_from="hindi",
    ),
    Variant(
        "hinglish",
        "codemix",
        "hin_Latn",
        HINDUSTANI_LATIN,
        "low-resource",
        rewrite_from="romanized-hindi",
        align_from="hindi",
        code_mixed=True,
    ),
    Variant("urdu", "indictrans2", "urd_Arab", ("urd_Arab",), "held-out"),
    Variant("tamil", "indictrans2", "tam_Taml", ("tam_Taml",), "held-out"),
    Variant("amharic", "nllb", "amh_Ethi", ("amh_Ethi",), "held-out"),
    Variant(
        "spanglish",
        "codemix",
        "spa_Latn",
        ("spa_Latn",),
        "held-out",
        rewrite_from="spanish",
        align_from="spanish",
        code_mixed=True,
    ),
    # Seven extra held-out languages (AlpacaFarm only, no InjecAgent).
    Variant("arabic", "nllb", "arb_Arab", ("arb_Arab",), "extra-held-out"),
    Variant("turkish", "nllb", "tur_Latn", ("tur_Latn",), "extra-held-out"),
    Variant("russian", "nllb", "rus_Cyrl", ("rus_Cyrl",), "extra-held-out"),
    Variant("greek", "nllb", "ell_Grek", ("ell_Grek",), "extra-held-out"),
    Variant("korean", "nllb", "kor_Hang", ("kor_Hang",), "extra-held-out"),
    Variant("thai", "nllb", "tha_Thai", ("tha_Thai",), "extra-held-out"),
    Variant("bengali", "indictrans2", "ben_Beng", ("ben_Beng",), "extra-held-out"),
    # Only the base of Spanglish; it is not evaluated or trained on.
    Variant("spanish", "nllb", "spa_Latn", ("spa_Latn",), "intermediate"),
)
BY_NAME = {variant.name: variant for variant in VARIANTS}
# Full InjecAgent costs about 80 minutes per variant, so it runs on these two for every model.
INJECAGENT_VARIANTS = {"english", "hindi"}

# The Tiny Aya tech report's regions (Table 1), which decide each regional model's training focus.
REGIONS = {
    "english": "europe", "french": "europe", "spanglish": "europe", "spanish": "europe",
    "russian": "europe", "greek": "europe",
    "arabic": "west-asia", "turkish": "west-asia",
    "chinese": "asia-pacific", "korean": "asia-pacific", "thai": "asia-pacific",
    "hindi": "south-asia", "romanized-hindi": "south-asia", "hinglish": "south-asia",
    "urdu": "south-asia", "tamil": "south-asia", "bengali": "south-asia",
    "swahili": "africa", "amharic": "africa",
}  # fmt: skip


def in_group(group: str) -> list[Variant]:
    return [variant for variant in VARIANTS if variant.group == group]
