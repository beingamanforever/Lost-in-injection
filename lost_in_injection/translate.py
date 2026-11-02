"""Machine-translate English text into a language variant, keeping translations that pass gates.

Each text is masked (masking.py) and translated into beam candidates. The best-ranked candidate
that passes every gate is kept:
1. every masked identifier comes back exactly once;
2. GlotLID's top 3 labels include all of the variant's expected labels;
3. its back-translation into English has LaBSE cosine similarity of at least 0.80 with the
   masked source.
Every candidate is returned with its gate values, so callers save a full audit record.
"""

from functools import cache

import fasttext
import torch
from huggingface_hub import hf_hub_download
from IndicTransToolkit.processor import IndicProcessor
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from lost_in_injection.languages import Variant
from lost_in_injection.masking import mask, unmask

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
    texts: list[str], src: str, tgt: str, translator: str, num_beams: int, num_return: int
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
                num_beams=num_beams,
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


def top_labels(text: str) -> list[str]:
    labels, _ = glotlid().predict(text.replace("\n", " "), k=3)
    return [label.removeprefix("__label__") for label in labels]


def similarity(sources: list[str], back_translations: list[str]) -> list[float]:
    if not sources:
        return []
    encode = labse().encode
    a = encode(sources, normalize_embeddings=True, convert_to_tensor=True)
    b = encode(back_translations, normalize_embeddings=True, convert_to_tensor=True)
    return (a * b).sum(dim=1).tolist()


def translate(texts: list[str], variant: Variant) -> list[dict]:
    """Return one record per text; its `translation` is None when no candidate passes."""
    masked, identifiers = zip(*map(mask, texts), strict=True)
    candidates = generate(
        list(masked), ENGLISH, variant.code, variant.translator, NUM_BEAMS, NUM_BEAMS
    )
    records = [{"source": text, "translation": None, "candidates": []} for text in texts]
    language_passed = []
    for i, text_candidates in enumerate(candidates):
        for candidate in text_candidates:
            restored = unmask(candidate, identifiers[i])
            labels = top_labels(restored) if restored is not None else []
            attempt = {"masked": candidate, "restored": restored, "glotlid": labels}
            records[i]["candidates"].append(attempt)
            if restored is not None and set(variant.expected_labels) <= set(labels):
                language_passed.append((i, attempt))

    back = generate(
        [attempt["masked"] for _, attempt in language_passed],
        variant.code,
        ENGLISH,
        variant.translator,
        num_beams=NUM_BEAMS,
        num_return=1,
    )
    back = [text[0] if text else "" for text in back]
    scores = similarity([masked[i] for i, _ in language_passed], back)
    for (i, attempt), back_translation, score in zip(language_passed, back, scores, strict=True):
        attempt["back_translation"] = back_translation
        attempt["similarity"] = score
        if records[i]["translation"] is None and score >= MIN_SIMILARITY:
            records[i]["translation"] = attempt["restored"]
    return records
