"""Code-mix a gated translation by switching aligned words back to English (gen environment).

Word alignments come from SimAlign (Jalili Sabet et al., 2020; mBERT, itermax) between the masked
English source and the gated translation. Every English noun and adjective that aligns one-to-one
with a translation word replaces it, while verbs and function words keep the translation's grammar.
This is insertional code-mixing (Myers-Scotton's matrix language frame): English content words
inside a Hindi or Spanish sentence, as in "diye gaye statement kaa answer yes yaa naheen mein dein".
It keeps the gated meaning. Parts of speech come from NLTK's perceptron tagger on lowercased
words, so imperatives such as "Write" at the start of a sentence are tagged as verbs, not nouns.
"""

import string
from collections import Counter
from functools import cache

import nltk
from simalign import SentenceAligner

PUNCTUATION = string.punctuation + "।॥¿¡“”‘’«»"
CONTENT_TAGS = ("NN", "JJ")  # Penn Treebank nouns and adjectives


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
    tags = [tag for _, tag in nltk.pos_tag([w.lower() for w in english_cores])]
    for i, j in one_to_one(align(english_cores, aligned_cores)):
        word = core(output_words[j])
        if word and english_cores[i].isalpha() and tags[i].startswith(CONTENT_TAGS):
            output_words[j] = output_words[j].replace(word, english_cores[i], 1)
    return " ".join(output_words)
