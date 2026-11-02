"""Language variants: how each one is produced and the role it plays in the experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    name: str
    translator: str  # "none", "nllb", "indictrans2", or "aya" (rewrite of `rewrite_from`)
    code: str  # FLORES-200 target code for the machine translators
    expected_labels: tuple[str, ...]  # GlotLID labels that must all appear in the top 3
    group: str  # "english", "high-resource", "low-resource" or "held-out"
    rewrite_from: str = ""  # Variant whose text an Aya rewrite starts from
    rewrite_instruction: str = ""


ROMANIZE = "Rewrite this {language} text in Latin script, the way people type {language} in chat."
CODE_MIX = (
    "Translate this text into {mix}: {language} mixed with English words, written in Latin "
    "script, the way people type in chat."
)

VARIANTS = (
    Variant("english", "none", "eng_Latn", ("eng_Latn",), "english"),
    Variant("french", "nllb", "fra_Latn", ("fra_Latn",), "high-resource"),
    Variant("chinese", "nllb", "zho_Hans", ("cmn_Hani",), "high-resource"),
    Variant("hindi", "indictrans2", "hin_Deva", ("hin_Deva",), "low-resource"),
    Variant("swahili", "nllb", "swh_Latn", ("swh_Latn",), "low-resource"),
    Variant(
        "romanized-hindi",
        "aya",
        "hin_Latn",
        ("hin_Latn",),
        "low-resource",
        rewrite_from="hindi",
        rewrite_instruction=ROMANIZE.format(language="Hindi"),
    ),
    Variant(
        "hinglish",
        "aya",
        "hin_Latn",
        ("hin_Latn", "eng_Latn"),
        "low-resource",
        rewrite_from="english",
        rewrite_instruction=CODE_MIX.format(mix="Hinglish", language="Hindi"),
    ),
    Variant("urdu", "indictrans2", "urd_Arab", ("urd_Arab",), "held-out"),
    Variant("tamil", "indictrans2", "tam_Taml", ("tam_Taml",), "held-out"),
    Variant("amharic", "nllb", "amh_Ethi", ("amh_Ethi",), "held-out"),
    Variant(
        "spanglish",
        "aya",
        "spa_Latn",
        ("spa_Latn", "eng_Latn"),
        "held-out",
        rewrite_from="english",
        rewrite_instruction=CODE_MIX.format(mix="Spanglish", language="Spanish"),
    ),
)
BY_NAME = {variant.name: variant for variant in VARIANTS}


def in_group(group: str) -> list[Variant]:
    return [variant for variant in VARIANTS if variant.group == group]
