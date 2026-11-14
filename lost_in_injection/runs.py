"""Training runs: the language of each pair's injected instruction.

The final pairs are ordered with the multilingual slots first. A run writes slot k in
`languages[k % len(languages)]` when k is below its non-English count and in English otherwise, so
`multilingual-share-25`'s non-English pairs are the first half of `english-plus-low-resource`'s, in
the same languages.
"""

NUM_PAIRS = 5000
MULTILINGUAL_SLOTS = 2500
LOW_RESOURCE = ("hindi", "swahili", "romanized-hindi", "hinglish")

# run name: (languages of the non-English slots, number of non-English slots)
RUNS = {
    "english-only": ((), 0),
    "english-plus-high-resource": (("french", "chinese"), MULTILINGUAL_SLOTS),
    "english-plus-low-resource": (LOW_RESOURCE, MULTILINGUAL_SLOTS),
    "multilingual-share-25": (LOW_RESOURCE, MULTILINGUAL_SLOTS // 2),
    "hindi-only-mix": (("hindi",), MULTILINGUAL_SLOTS),
    "native-scripts-only": (("hindi", "swahili"), MULTILINGUAL_SLOTS),
    "no-input-role": ((), 0),  # english-only with the untrusted data inside the user turn
}
# Romanized Hindi is rewritten from Hindi, so Hindi comes first.
TRAINING_LANGUAGES = ("french", "chinese", "hindi", "swahili", "romanized-hindi", "hinglish")


def language(run: str, slot: int) -> str:
    languages, count = RUNS[run]
    return languages[slot % len(languages)] if slot < count else "english"
