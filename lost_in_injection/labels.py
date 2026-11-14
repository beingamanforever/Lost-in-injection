"""Pick the final pairs and self-generate their labels with the undefended model (GPU).

Multilingual slots are the first MULTILINGUAL_SLOTS candidates of the translation pool whose
injected instruction and input pass the gates in every training language; English slots are the
next candidates in draw order. As in Meta-SecAlign (utils.py:94-115), labels are the undefended
model's own replies at temperature 0.8: chosen answers the clean instruction with its data, and
rejected answers the injected instruction alone, once per pair and language a run can use. Each
label records its finish reason, so export can drop pairs that hit the token cap. Each pair also
gets its untrusted data per language, built with Meta-SecAlign's attack code (utils.py:84-88).
"""

import argparse
from pathlib import Path

import numpy as np
from vllm import LLM, SamplingParams

from lost_in_injection import secalign
from lost_in_injection.evaluate import GPU_MEMORY_UTILIZATION
from lost_in_injection.records import read_jsonl, read_records, write_jsonl
from lost_in_injection.runs import MULTILINGUAL_SLOTS, NUM_PAIRS, RUNS, TRAINING_LANGUAGES, language
from lost_in_injection.skeleton import DATA_SEED
from lost_in_injection.translate_pairs import POOL_SIZE

# Meta-SecAlign samples at temperature 0.8 with up to 8,192 tokens. The cap here keeps pairs
# within the 2,048-token training length; 10-20% of the base model's AlpacaFarm answers exceed 512.
LABELS = SamplingParams(temperature=0.8, max_tokens=1024, seed=DATA_SEED)


def injected(pair: dict, lang: str, translations: dict) -> dict[str, str]:
    source = pair["injection"]
    if lang == "english":
        return {field: source[field] for field in ("instruction", "input")}
    return {
        field: translations[lang][source[field]] if source[field] else ""
        for field in ("instruction", "input")
    }


def injected_prompt(injection: dict[str, str]) -> str:
    return f"{injection['instruction']} {injection['input']}"  # utils.py:81


def data_with_injection(pair: dict, injection: dict[str, str]) -> str:
    """The clean data with the injection inserted, as Meta-SecAlign builds its training input."""
    data = pair["clean"]["input"]
    if pair["attack"] == "completion":
        np.random.seed(pair["completion_seed"])  # The fake delimiters are drawn from np.random.
        completion = secalign.load("utils").create_injection_for_completion(
            pair["fake_response"], injection["instruction"], injection["input"]
        )
        return f"{data}\n\n{completion}"
    if pair["position"] == "before":
        return f"{injected_prompt(injection)} {data}"
    return f"{data} {injected_prompt(injection)}"


def select(candidates: list[dict], translations: dict) -> list[dict]:
    def translated_everywhere(pair: dict) -> bool:
        texts = [pair["injection"][field] for field in ("instruction", "input")]
        return all(translations[lang].get(t) for lang in TRAINING_LANGUAGES for t in texts if t)

    multilingual = [p for p in candidates[:POOL_SIZE] if translated_everywhere(p)]
    if len(multilingual) < MULTILINGUAL_SLOTS:
        raise SystemExit(f"Only {len(multilingual)} candidates pass the gates everywhere")
    multilingual = multilingual[:MULTILINGUAL_SLOTS]
    taken = {p["pair_id"] for p in multilingual}
    english = [p for p in candidates if p["pair_id"] not in taken][: NUM_PAIRS - MULTILINGUAL_SLOTS]
    return [{**pair, "slot": slot} for slot, pair in enumerate(multilingual + english)]


def replies(llm: LLM, prompts: list[str]) -> dict[str, dict]:
    outputs = llm.chat([[{"role": "user", "content": p}] for p in prompts], LABELS)
    return {
        prompt: {"text": output.outputs[0].text, "finish_reason": output.outputs[0].finish_reason}
        for prompt, output in zip(prompts, outputs, strict=True)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=Path("models/tiny-aya-global"))
    parser.add_argument("--pairs", type=Path, default=Path("data/pairs"))
    args = parser.parse_args()

    translations = {
        lang: {s: r["translation"] for s, r in read_records(path).items()}
        for lang in TRAINING_LANGUAGES
        for path in [args.pairs / "translations" / lang / "translations.jsonl"]
    }
    pairs = select(read_jsonl(args.pairs / "skeleton.jsonl"), translations)
    for pair in pairs:
        languages = sorted({language(run, pair["slot"]) for run in RUNS})
        pair["injected"] = {lang: injected(pair, lang, translations) for lang in languages}
        pair["data"] = {lang: data_with_injection(pair, i) for lang, i in pair["injected"].items()}

    def chosen_prompt(pair: dict) -> str:
        return f"{pair['clean']['instruction']}\n\n{pair['clean']['input']}"  # utils.py:112

    llm = LLM(model=str(args.model), gpu_memory_utilization=GPU_MEMORY_UTILIZATION, seed=DATA_SEED)
    rejected_prompts = {injected_prompt(i) for p in pairs for i in p["injected"].values()}
    labels = replies(llm, sorted({chosen_prompt(p) for p in pairs} | rejected_prompts))
    for pair in pairs:
        pair["chosen"] = labels[chosen_prompt(pair)]
        pair["rejected"] = {
            lang: labels[injected_prompt(injection)] for lang, injection in pair["injected"].items()
        }
    write_jsonl(args.pairs / "labelled.jsonl", pairs)
    capped = sum(label["finish_reason"] == "length" for label in labels.values())
    print(f"{len(pairs)} pairs, {len(labels)} labels, {capped} hit the token cap")


if __name__ == "__main__":
    main()
