"""Score instruction following with vLLM generation and each benchmark's own checker (gen env).

No LLM judge: every model, base or merged, gets one greedy response per prompt (temperature 0,
max_tokens=1280, the lm-eval default for IFEval and IndicIFEval), and each benchmark's own
instruction checker decides whether the response follows every instruction. vLLM's `.chat()`
renders the model directory's own chat template and tokenizes it with `add_special_tokens=False`
(the template already starts with BOS), so a single call cannot double up the BOS token.

IFEval, IndicIFEval and Marco-Bench-MIF each define a top-level module named
`instructions_registry` (IFEval, IndicIFEval) or `instructions`/`instructions_registry` (Marco),
so only one benchmark's checker directory can sit on `sys.path` in a process: run this script
once per benchmark, in its own process.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import langdetect
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from lost_in_injection.evaluate import GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN, small_gpu

# IFEval's and IndicIFEval's instruction checkers call langdetect.detect(), which is
# non-deterministic unless the detector is seeded.
langdetect.DetectorFactory.seed = 0

MAX_TOKENS = 1280

# Localized data files where both a localized and machine-translated version exist.
MARCO_LANGUAGES = [
    ("fr", "fr"),
    ("sw", "sw"),
    ("zh", "zh_local"),
    ("ar", "ar_local"),
    ("tr", "tr"),
    ("ru", "ru"),
    ("el", "el"),
    ("ko", "ko"),
    ("th", "th"),
    ("bn", "bn"),
]


def load_ifeval(_raw_dir: Path):
    from lm_eval.tasks import TaskManager, get_task_dict

    task = get_task_dict(["ifeval"], TaskManager())["ifeval"]
    entries = [("en", doc) for doc in task.eval_docs]

    def score(doc, response):
        result = task.process_results(doc, [response])
        return result["prompt_level_strict_acc"], result["inst_level_strict_acc"]

    return entries, score


def load_indicifeval(raw_dir: Path):
    from lm_eval.tasks import TaskManager, get_task_dict

    task_name = "indicifeval_trans_hi"
    checker_dir = raw_dir / "indicifeval"
    task = get_task_dict([task_name], TaskManager(include_path=str(checker_dir)))[task_name]
    entries = [("hi", doc) for doc in task.eval_docs]

    def score(doc, response):
        result = task.process_results(doc, [response])
        return result["prompt_level_strict_acc"], result["inst_level_strict_acc"]

    return entries, score


def load_marco(raw_dir: Path):
    data_dir = raw_dir / "marco-bench-mif"
    sys.path.insert(0, str(data_dir / "checker"))
    import evaluation_main  # the checker's own top-level module name

    entries = [
        (language, json.loads(line))
        for language, stem in MARCO_LANGUAGES
        for line in (data_dir / f"{stem}.jsonl").read_text().splitlines()
    ]

    def score(doc, response):
        inp = evaluation_main.InputExample(
            key=doc["key"],
            instruction_id_list=doc["instruction_id_list"],
            prompt=doc["prompt"],
            kwargs=doc["kwargs"],
        )
        out = evaluation_main.test_instruction_following_strict(inp, {doc["prompt"]: response})
        return out.follow_all_instructions, out.follow_instruction_list

    return entries, score


LOADERS = {"ifeval": load_ifeval, "indicifeval": load_indicifeval, "marco-bench-mif": load_marco}


def generate_responses(
    model_dir: Path, prompts: list[str], adapter: Path | None = None
) -> list[str]:
    lora = {}
    if adapter:
        lora = {"enable_lora": True, "max_lora_rank": 64, "enforce_eager": small_gpu()}
    llm = LLM(
        **lora,
        model=str(model_dir),
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
    request = LoRARequest("adapter", 1, str(adapter.resolve())) if adapter else None
    outputs = llm.chat(conversations, sampling_params, use_tqdm=True, lora_request=request)

    bos_id = llm.get_tokenizer().bos_token_id  # None for Qwen, which uses no BOS
    first_ids = outputs[0].prompt_token_ids
    if bos_id is not None and (first_ids[0] != bos_id or first_ids[1] == bos_id):
        raise RuntimeError(f"Expected exactly one leading BOS token, got {first_ids[:2]}")

    responses = []
    for prompt, output in zip(prompts, outputs, strict=True):
        text = output.outputs[0].text if output.outputs else ""
        if not text:
            raise RuntimeError(f"No response generated for prompt: {prompt!r}")
        responses.append(text)
    return responses


def run(
    model_dir: Path, out_dir: Path, benchmark: str, raw_dir: Path, adapter: Path | None = None
) -> None:
    entries, score = LOADERS[benchmark](raw_dir)
    prompts = [doc["prompt"] for _language, doc in entries]
    responses = generate_responses(model_dir, prompts, adapter)

    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_results_by_language = defaultdict(list)
    with (out_dir / f"{benchmark}.jsonl").open("w") as out_file:
        for (language, doc), response in zip(entries, responses, strict=True):
            prompt_strict, instructions_strict = score(doc, response)
            row = {
                "language": language,
                "key": doc["key"],
                "prompt": doc["prompt"],
                "response": response,
                "prompt_strict": prompt_strict,
                "instructions_strict": instructions_strict,
            }
            out_file.write(json.dumps(row) + "\n")
            prompt_results_by_language[language].append(prompt_strict)

    for language, prompt_results in prompt_results_by_language.items():
        accuracy = sum(prompt_results) / len(prompt_results)
        print(f"{language}: {accuracy:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="Model directory.")
    parser.add_argument("--out", type=Path, required=True, help="Output folder.")
    parser.add_argument("--benchmark", required=True, choices=sorted(LOADERS))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--adapter", type=Path, help="LoRA folder to apply at inference.")
    args = parser.parse_args()
    run(args.model, args.out, args.benchmark, args.raw, args.adapter)


if __name__ == "__main__":
    main()
