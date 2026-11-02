"""Rewrite masked text into a romanized or code-mixed variant with Aya Expanse 8B, behind the gates.

The machine translators write native scripts only, so romanized Hindi, Hinglish and Spanglish come
from Aya Expanse 8B, which covers Hindi and Spanish. Romanized Hindi starts from the gated Hindi
translation, and the code-mixed variants start from English. Placeholders from masking.py stay in
the text, and the gates are translate.py's. Aya Expanse writes the English back-translation too,
because NLLB and IndicTrans2 cannot read romanized or code-mixed input.
"""

from functools import cache

from vllm import LLM, SamplingParams

from lost_in_injection.languages import Variant
from lost_in_injection.translate import gate

AYA_EXPANSE = ("CohereLabs/aya-expanse-8b", "5062468bf9bc0c6035fd64e06274333ec127d980")
# The bf16 weights take 15 GiB and vLLM's activations about 2.8 GiB. This leaves about 1.6 GiB of
# KV cache for vLLM and 3 GiB outside it for LaBSE.
GPU_MEMORY_UTILIZATION = 0.86
KEEP_PLACEHOLDERS = (
    "Keep every placeholder such as [1] exactly as written. Reply with the text only."
)
TO_ENGLISH = "Translate this text into English."
GREEDY = SamplingParams(temperature=0, max_tokens=512)
# Four seeded samples follow the greedy candidate, so the gates have alternatives to choose from.
SAMPLED = SamplingParams(n=4, temperature=0.7, seed=0, max_tokens=512)


@cache
def aya() -> LLM:
    return LLM(
        model=AYA_EXPANSE[0],
        revision=AYA_EXPANSE[1],
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=2048,
        seed=0,
    )


def chat(instruction: str, texts: list[str], params: SamplingParams) -> list[list[str]]:
    prompts = [
        [{"role": "user", "content": f"{instruction} {KEEP_PLACEHOLDERS}\n\n{text}"}]
        for text in texts
    ]
    outputs = aya().chat(prompts, params, use_tqdm=False)
    return [[completion.text.strip() for completion in output.outputs] for output in outputs]


def rewrite(texts: list[str], masked_inputs: list[str], variant: Variant) -> list[dict]:
    """Rewrite `masked_inputs` (masked English or masked Hindi) into `variant` and gate them.

    `texts` are the English sources: their masks restore the identifiers, and the back-translation
    is compared with them.
    """
    greedy = chat(variant.rewrite_instruction, masked_inputs, GREEDY)
    sampled = chat(variant.rewrite_instruction, masked_inputs, SAMPLED)
    candidates = [first + rest for first, rest in zip(greedy, sampled, strict=True)]

    def back_translate(texts: list[str]) -> list[str]:
        return [text[0] for text in chat(TO_ENGLISH, texts, GREEDY)]

    return gate(texts, candidates, variant, back_translate)
