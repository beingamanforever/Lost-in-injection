"""Training runs: the language of each pair's injected instruction.

The final pairs are ordered with the multilingual slots first. A run writes slot k in
`languages[k % len(languages)]` when k is below its non-English count and in English otherwise, so
`multilingual-share-25`'s non-English pairs are the first half of `english-plus-low-resource`'s, in
the same languages.

Phase 2's regional Tiny Aya models (FAMILY) and the paper's other model families
(OTHER_FAMILIES) reuse Global's runs under a prefixed name, e.g. "fire-english-only" or
"llama8b-english-only", so `language` gives them exactly Global's slot languages, and `base_model`
can tell which base checkpoint a (possibly mixed) list of run names trains on. Global runs keep
their bare names so published paths, Hub repos and metric keys stay valid.
"""

from pathlib import Path

NUM_PAIRS = 5000
MULTILINGUAL_SLOTS = 2500
LOW_RESOURCE = ("hindi", "swahili", "romanized-hindi", "hinglish")
FAMILY = ("fire", "earth", "water")
# llama70b is only evaluated, undefended and as Meta-SecAlign-70B: training needs more than an A10G.
OTHER_FAMILIES = ("llama8b", "llama3b", "qwen", "llama70b")
FAMILY_RUNS = ("english-only", "english-plus-low-resource")
BASE_MODELS = {
    "global": "CohereLabs/tiny-aya-global",
    **{model: f"CohereLabs/tiny-aya-{model}" for model in FAMILY},
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama70b": "meta-llama/Llama-3.3-70B-Instruct",
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen": "Qwen/Qwen3-4B-Instruct-2507",
}
# Meta's own English-only defenses of Llama 3.1 8B and 3.3 70B, evaluated as more Llama runs.
META_SECALIGN_RUNS = ("llama8b-meta-secalign", "llama70b-meta-secalign")
# Regions each model's training focused on (Tiny Aya report, Sec. 2.3.3): Fire is South Asia only;
# Earth and Water add Africa or Asia-Pacific to Europe and West Asia; Global covers them all.
FOCUS = {
    "global": {"europe", "west-asia", "south-asia", "asia-pacific", "africa"},
    "fire": {"south-asia"},
    "earth": {"europe", "west-asia", "africa"},
    "water": {"europe", "west-asia", "asia-pacific"},
}

# run name: (languages of the non-English slots, number of non-English slots)
RUNS = {
    "english-only": ((), 0),
    "english-plus-high-resource": (("french", "chinese"), MULTILINGUAL_SLOTS),
    "english-plus-low-resource": (LOW_RESOURCE, MULTILINGUAL_SLOTS),
    "multilingual-share-25": (LOW_RESOURCE, MULTILINGUAL_SLOTS // 2),
    "hindi-only-mix": (("hindi",), MULTILINGUAL_SLOTS),
    "native-scripts-only": (("hindi", "swahili"), MULTILINGUAL_SLOTS),
    "no-input-role": ((), 0),  # english-only with the untrusted data inside the user turn
    # english-plus-low-resource, but each multilingual slot also keeps its English pair.
    "english-plus-low-resource-added": (LOW_RESOURCE, MULTILINGUAL_SLOTS),
}
# Runs that add their non-English pairs to the English ones instead of replacing them.
ADDED_RUNS = {"english-plus-low-resource-added"}
# Romanized Hindi is rewritten from Hindi, so Hindi comes first.
TRAINING_LANGUAGES = ("french", "chinese", "hindi", "swahili", "romanized-hindi", "hinglish")


def split(name: str) -> tuple[str, str]:
    """A run's model and its name within it: "fire-english-only" -> ("fire", "english-only")."""
    for model in (*FAMILY, *OTHER_FAMILIES):
        if name.startswith(f"{model}-"):
            return model, name.removeprefix(f"{model}-")
    return "global", name


# Prompting defenses and the Prompt Guard 2 detector, by Meta-SecAlign runner name; every other run
# uses "none". Prompt Guard blanks the data it flags at score 0.5 (Meta_SecAlign utils.py:240-246).
PROMPT_DEFENSES = {
    "prompted": "reminder",
    "sandwich": "sandwich",
    "isolation": "isolation",
    "promptguard": "promptguard0.5",
}


def prompt_defense(run: str) -> str:
    return PROMPT_DEFENSES.get(split(run)[1], "none")


def model_path(model: str) -> Path:
    """A model's local folder, named after its Hub repo.

    "llama8b" -> models/llama-3.1-8b-instruct.
    """
    return Path("models") / BASE_MODELS[model].split("/")[1].lower()


def base_model(*names: str) -> Path:
    """The base model folder a list of run names trains on; raises if they span more than one."""
    models = {split(name)[0] for name in names}
    if len(models) > 1:
        raise ValueError(f"Runs span more than one model: {sorted(models)}")
    return model_path(models.pop())


def language(run: str, slot: int) -> str:
    languages, count = RUNS[split(run)[1]]
    return languages[slot % len(languages)] if slot < count else "english"
