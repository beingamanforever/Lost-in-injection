"""Machine-translate English text into a language variant, keeping translations that pass gates.

Each text is masked (masking.py) and translated sentence by sentence into beam candidates,
because both translators are trained on sentences and drop content from longer inputs. The
best-ranked candidate that passes every gate is kept:
1. every masked identifier comes back exactly once;
2. GlotLID's top 3 labels for its translated words (identifiers removed) include one of the
   variant's accepted labels;
3. its back-translation into English has LaBSE cosine similarity of at least 0.80 with the
   masked source.
Romanized and code-mixed variants are word-for-word transforms of a gated translation, so they
inherit its language ID and back-translation, and check instead that their letters are all Latin.
Code-mixed variants also need 10-90% of their words kept verbatim from the English source, so both
languages are present (word-level GlotLID is too noisy on single words to count English words).
Every candidate is returned with its gate values, so callers save a full audit record.
"""

import re
import unicodedata
from collections.abc import Callable
from functools import cache
from pathlib import Path

import fasttext
import torch
from huggingface_hub import hf_hub_download
from IndicTransToolkit.processor import IndicProcessor
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from lost_in_injection.languages import Variant
from lost_in_injection.masking import PLACEHOLDER, mask, unmask
from lost_in_injection.records import chosen_candidate, masked_translation, read_jsonl, read_records

# Hugging Face repositories at pinned revisions.
NLLB = ("facebook/nllb-200-3.3B", "1a07f7d195896b2114afcb79b7b57ab512e7b43e")
INDICTRANS2_EN_INDIC = (
    "ai4bharat/indictrans2-en-indic-1B",
    "89cd0007fd7b9e0693a2fd4404d187e07dac79fd",
)
INDICTRANS2_INDIC_EN = (
    "ai4bharat/indictrans2-indic-en-1B",
    "ac3daf0ecd37be3b6957764a9179ab2b07fa9d6a",
)
GLOTLID = ("cis-lmu/glotlid", "baa20d139151a26a71a20eddabf30d4de7329fc7")
LABSE = ("sentence-transformers/LaBSE", "836121a0533e5664b21c7aacc5d22951f2b8b25b")

ENGLISH = "eng_Latn"
NUM_BEAMS = 5  # IndicTrans2's documented beam size, used for both translators.
MAX_TOKENS = 256  # IndicTrans2's documented max_length. Sources that reach it are rejected.
BATCH_SIZE = 16
MIN_SIMILARITY = 0.80
DERIVED = ("xlit", "codemix")  # Transforms of a gated translation (see above)
ENGLISH_SHARE = (0.1, 0.9)  # Both languages present in a code-mixed text.
WORD = re.compile(r"[^\W\d_]{3,}")  # Words of three or more letters
SENTENCE_END = re.compile(r"(?<=[.!?।۔።])\s+|(?<=[。！？])")


@cache
def seq2seq(repo: str, revision: str):
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        repo, revision=revision, trust_remote_code=True, torch_dtype=torch.float16
    )
    return tokenizer, model.to("cuda").eval()


@cache
def indic_processor() -> IndicProcessor:
    return IndicProcessor(inference=True)


@cache
def glotlid():
    return fasttext.load_model(hf_hub_download(GLOTLID[0], "model.bin", revision=GLOTLID[1]))


@cache
def labse() -> SentenceTransformer:
    return SentenceTransformer(LABSE[0], revision=LABSE[1], device="cuda")


def generate(
    texts: list[str], src: str, tgt: str, translator: str, num_return: int
) -> list[list[str]]:
    """Translate with each model's documented pre- and post-processing; candidates best first.

    A text whose tokenized source reaches MAX_TOKENS gets no candidates rather than a
    translation of a truncated source.
    """
    if translator == "indictrans2":
        model_id = INDICTRANS2_EN_INDIC if src == ENGLISH else INDICTRANS2_INDIC_EN
        tokenizer, model = seq2seq(*model_id)
        # IndicTrans2's remote code reads legacy tuple caches, and transformers 4.57 passes a
        # Cache object, so decoding runs without a cache: slower, same outputs.
        extra = {"use_cache": False}
    else:
        tokenizer, model = seq2seq(*NLLB)
        tokenizer.src_lang = src
        extra = {"forced_bos_token_id": tokenizer.convert_tokens_to_ids(tgt)}
    candidates = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        if translator == "indictrans2":
            batch = indic_processor().preprocess_batch(batch, src_lang=src, tgt_lang=tgt)
        inputs = tokenizer(
            batch, truncation=True, max_length=MAX_TOKENS, padding="longest", return_tensors="pt"
        ).to("cuda")
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                num_beams=NUM_BEAMS,
                num_return_sequences=num_return,
                max_length=MAX_TOKENS,
                **extra,
            )
        decoded = tokenizer.batch_decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        if translator == "indictrans2":
            decoded = indic_processor().postprocess_batch(
                decoded, lang=tgt, num_return_sequences=num_return
            )
        too_long = (inputs["attention_mask"].sum(dim=1) >= MAX_TOKENS).tolist()
        for i, skip in enumerate(too_long):
            candidates.append([] if skip else decoded[i * num_return : (i + 1) * num_return])
    return candidates


def translate_sentences(
    texts: list[str], src: str, tgt: str, translator: str, num_return: int
) -> list[list[str]]:
    """Translate texts sentence by sentence; candidate k joins each sentence's k-th candidate.

    Pieces without letters, such as "..." or a lone placeholder, are copied: there is nothing to
    translate, and the models degenerate on them.
    """
    sentences = [[s for s in SENTENCE_END.split(text) if s.strip()] for text in texts]
    has_words = [s for text in sentences for s in text if any(c.isalpha() for c in s)]
    translated = iter(generate(has_words, src, tgt, translator, num_return))
    candidates = []
    for text in sentences:
        per_sentence = [
            next(translated) if any(c.isalpha() for c in s) else [s] * num_return for s in text
        ]
        count = min(map(len, per_sentence), default=0)
        candidates.append([" ".join(s[k] for s in per_sentence) for k in range(count)])
    return candidates


def top_labels(text: str) -> list[str]:
    labels, _ = glotlid().predict(text.replace("\n", " "), k=3)
    return [label.removeprefix("__label__") for label in labels]


def english_share(text: str, source: str) -> float:
    """Share of the text's words that appear verbatim in the English source."""
    words = [word.lower() for word in WORD.findall(text)]
    source_words = {word.lower() for word in WORD.findall(source)}
    return sum(word in source_words for word in words) / len(words) if words else 0.0


def latin_only(text: str) -> bool:
    return all(unicodedata.name(c, "").startswith("LATIN") for c in text if c.isalpha())


def language_passes(text: str, source: str, variant: Variant, attempt: dict) -> bool:
    if variant.translator in DERIVED:
        attempt["latin_only"] = latin_only(text)
        if not attempt["latin_only"]:
            return False
    else:
        attempt["glotlid"] = top_labels(text)
        if not set(variant.accepted_labels) & set(attempt["glotlid"]):
            return False
    if not variant.code_mixed:
        return True
    attempt["english_share"] = english_share(text, source)
    return ENGLISH_SHARE[0] <= attempt["english_share"] <= ENGLISH_SHARE[1]


def similarity(sources: list[str], back_translations: list[str]) -> list[float]:
    if not sources:
        return []
    encode = labse().encode
    a = encode(sources, normalize_embeddings=True, convert_to_tensor=True)
    b = encode(back_translations, normalize_embeddings=True, convert_to_tensor=True)
    return (a * b).sum(dim=1).tolist()


def gate(
    texts: list[str],
    candidates: list[list[str]],
    variant: Variant,
    back_translate: Callable[[list[str]], list[str]],
) -> list[dict]:
    """Keep each text's best-ranked masked candidate that passes every gate.

    Returns one record per text; its `translation` is None when no candidate passes.
    """
    masked, identifiers = zip(*map(mask, texts), strict=True)
    records = [{"source": text, "translation": None, "candidates": []} for text in texts]
    language_passed = []
    for i, text_candidates in enumerate(candidates):
        for candidate in text_candidates:
            restored = unmask(candidate, identifiers[i])
            attempt = {"masked": candidate, "restored": restored, "glotlid": []}
            records[i]["candidates"].append(attempt)
            words, source = PLACEHOLDER.sub("", candidate), PLACEHOLDER.sub("", masked[i])
            if restored is not None and language_passes(words, source, variant, attempt):
                language_passed.append((i, attempt))

    back = back_translate([attempt["masked"] for _, attempt in language_passed])
    scores = similarity([masked[i] for i, _ in language_passed], back)
    for (i, attempt), back_translation, score in zip(language_passed, back, scores, strict=True):
        attempt["back_translation"] = back_translation
        attempt["similarity"] = score
        if records[i]["translation"] is None and score >= MIN_SIMILARITY:
            records[i]["translation"] = attempt["restored"]
    return records


def translate(texts: list[str], variant: Variant) -> list[dict]:
    """Machine-translate English texts into `variant` and gate them."""
    masked = [mask(text)[0] for text in texts]
    candidates = translate_sentences(
        masked, ENGLISH, variant.code, variant.translator, num_return=NUM_BEAMS
    )

    def back_translate(texts: list[str]) -> list[str]:
        back = translate_sentences(texts, variant.code, ENGLISH, variant.translator, num_return=1)
        return [text[0] if text else "" for text in back]

    return gate(texts, candidates, variant, back_translate)


def inherited_back_translation(candidates: list[list[str]], bases: list[dict | None]):
    """Back-translation function that returns each base translation's gated back-translation.

    Romanized and code-mixed text keeps its base translation's meaning word for word, so its
    meaning gate reuses the base's back-translation instead of a second model hop.
    """
    back = {
        candidate: chosen_candidate(base)["back_translation"]
        for text_candidates, base in zip(candidates, bases, strict=True)
        if base is not None
        for candidate in text_candidates
    }
    return lambda texts: [back[text] for text in texts]


def translate_or_rewrite(sources: list[str], root: Path, variant: Variant) -> list[dict]:
    """Produce gated `variant` text for English sources. Romanized and code-mixed variants read
    their base variants from `root/<variant>/`."""
    if variant.translator in ("nllb", "indictrans2"):
        return translate(sources, variant)
    if variant.translator == "aya":
        from lost_in_injection.rewrite import rewrite  # Imported here: rewrite imports this module.

        return rewrite(sources, [mask(source)[0] for source in sources], variant)

    def base_records(name: str) -> list[dict | None]:
        records = read_records(root / name / "translations.jsonl")
        return [records[s] if records.get(s, {}).get("translation") else None for s in sources]

    if variant.translator == "xlit":
        romanized = {
            r["source"]: r["masked"] for r in read_jsonl(root / variant.name / "romanized.jsonl")
        }
        bases = base_records(variant.rewrite_from)
        candidates = [
            [romanized[s]] if base and s in romanized else []
            for s, base in zip(sources, bases, strict=True)
        ]
        return gate(sources, candidates, variant, inherited_back_translation(candidates, bases))

    from lost_in_injection.codemix import switch  # Needs SimAlign's aligner model.

    bases = base_records(variant.rewrite_from)
    aligned = base_records(variant.align_from)
    candidates = []
    for source, base, align in zip(sources, bases, aligned, strict=True):
        mixed = (
            base
            and align
            and switch(mask(source)[0], masked_translation(align), masked_translation(base))
        )
        candidates.append([mixed] if mixed else [])
    return gate(sources, candidates, variant, inherited_back_translation(candidates, aligned))
