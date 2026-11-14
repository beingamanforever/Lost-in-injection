---
base_model: CohereLabs/tiny-aya-global
library_name: peft
license: cc-by-nc-4.0
pipeline_tag: text-generation
language: [en, fr, zh, hi, sw, ur, ta, am, es]
datasets: [beingamanforever/lost-in-injection-training-pairs]
tags: [prompt-injection, safety, dpo, lora, secalign, multilingual, code-mixing]
---

# Tiny Aya Global, hardened against prompt injection (English, French, Chinese)

A LoRA adapter for [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) that teaches the model to treat text in tool output as data, not instructions, following Meta-SecAlign's recipe.
Training injections: English, French and Chinese.
Adds French and Chinese injections to training. Per language it matches the English-only adapter; against an attacker who tries every language it does somewhat better.

Part of [Lost in Injection](https://github.com/beingamanforever/Lost-in-injection), which asks whether a prompt-injection defense trained in English holds in other languages, scripts and code-mixes.

## Results

AlpacaFarm-Hacked, 208 cases per variant, any of six attacks, leading markdown skipped.

| | Undefended | This adapter |
|---|---|---|
| Attack success, English | 89.9% | 11.5% |
| Attack success, held out (Urdu, Tamil, Amharic, Spanglish) | 69-85% | 2.9-6.2% |
| Best of all 11 variants per case | 97.1% | 15.9% |
| Global-MMLU-Lite, mean of 5 languages | 55.4% | 55.5% |

## Use

Untrusted text goes in an `input` turn after the user's turn, which the repository's template helper renders as a tool result.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from lost_in_injection.model_dir import add_input_role, official_template  # from the GitHub repository

base = "CohereLabs/tiny-aya-global"
tokenizer = AutoTokenizer.from_pretrained(base)
added = [t.content for t in tokenizer.added_tokens_decoder.values()]
tokenizer.chat_template = add_input_role(official_template(tokenizer), added)
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base, torch_dtype="bfloat16"),
                                  "beingamanforever/tiny-aya-secalign-high-resource")

messages = [{"role": "user", "content": "Summarize this review."},
            {"role": "input", "content": review}]  # the untrusted text
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
reply = model.generate(inputs, max_new_tokens=256)[0][inputs.shape[1]:]
print(tokenizer.decode(reply, skip_special_tokens=True))
```

## Training

DPO (sigmoid, beta 0.1) with LoRA (rank 64, alpha 8, dropout 0.1, on q, v, gate, up and down projections) through AlignTune's TRL backend.
4,834 self-generated preference pairs, 3 epochs, 228 steps, effective batch 64, learning rate 1.6e-4 with a cosine schedule, 2,048 tokens.
One A10G, 6.7 hours.
The three adapters share every pair; only the language of the injection differs.

## License

CC-BY-NC-4.0, as the base model and the training data.
