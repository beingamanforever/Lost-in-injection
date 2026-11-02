"""Code-mix a gated translation by switching aligned words back to English (gen environment).

Word alignments come from SimAlign (Jalili Sabet et al., 2020; mBERT, itermax) between the masked
English source and the gated translation. Of the English words of four or more letters that align
one-to-one with a translation word, every second one in sentence order replaces its aligned word.
This is insertional code-mixing into the translation's grammar, and it keeps the gated meaning.
"""

import string
from collections import Counter
from functools import cache

from simalign import SentenceAligner

PUNCTUATION = string.punctuation + "।॥¿¡“”‘’«»"


@cache
def aligner() -> SentenceAligner:
    return SentenceAligner(model="bert", token_type="bpe", matching_methods="i")


def core(token: str) -> str:
    return token.strip(PUNCTUATION)


def one_to_one(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    sources, targets = Counter(i for i, _ in pairs), Counter(j for _, j in pairs)
    return sorted((i, j) for i, j in pairs if sources[i] == 1 and targets[j] == 1)


def switch(english: str, aligned: str, output: str, align=None) -> str | None:
    """Switch words of `output` back to English, using alignments to `aligned`.

    `output` is `aligned` itself or a word-for-word rewrite of it (romanized Hindi), so both must
    have the same words. Returns None when they do not.
    """
    english_words, aligned_words, output_words = english.split(), aligned.split(), output.split()
    if len(aligned_words) != len(output_words):
        return None
    english_cores = [core(w) or w for w in english_words]
    aligned_cores = [core(w) or w for w in aligned_words]
    align = align or (lambda e, t: aligner().get_word_aligns(e, t)["itermax"])
    pairs = one_to_one(align(english_cores, aligned_cores))
    eligible = [
        (i, j) for i, j in pairs if english_cores[i].isalpha() and len(english_cores[i]) >= 4
    ]
    for i, j in eligible[::2]:
        word = core(output_words[j])
        if word:
            output_words[j] = output_words[j].replace(word, english_cores[i], 1)
    return " ".join(output_words)
