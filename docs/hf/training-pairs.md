---
license: cc-by-nc-4.0
language: [en, fr, zh, hi, sw]
pretty_name: Lost in Injection training pairs
task_categories: [text-generation]
tags: [prompt-injection, dpo, preference, safety, multilingual, code-mixing]
size_categories: [1K<n<10K]
configs:
- config_name: english-only
  data_files: english-only/dpo.jsonl
- config_name: english-plus-high-resource
  data_files: english-plus-high-resource/dpo.jsonl
- config_name: english-plus-low-resource
  data_files: english-plus-low-resource/dpo.jsonl
---

# Lost in Injection training pairs

The DPO preference pairs behind the three prompt-injection defenses in [Lost in Injection](https://github.com/beingamanforever/Lost-in-injection).
Each pair hides an injected instruction in the data of an Alpaca task, then prefers Tiny Aya Global's own answer to the task (`chosen`) over its own answer to the injection (`rejected`), following Meta-SecAlign.

| Config | Language of the injected instruction | Pairs |
|---|---|---|
| `english-only` | English | 4,834 |
| `english-plus-high-resource` | English, French, Chinese | 4,834 |
| `english-plus-low-resource` | English, Hindi, Swahili, Hindi in Latin script, Hinglish | 4,834 |

The three configs share every task, attack, position and label; only the language of the injected instruction differs.
`prompt` is rendered in Tiny Aya's chat template, with the untrusted data as a tool result after the user's turn.
`pairs_meta.jsonl` gives each pair's slot, language, translator, attack and position, and each folder keeps CuratorKIT's export manifest, card and provenance record.

## Sources and license

Tasks come from `yahma/alpaca-cleaned` (CC-BY-4.0) and injected instructions from `tatsu-lab/alpaca` (CC-BY-NC-4.0).
Translations are by NLLB-200 (CC-BY-NC-4.0) and IndicTrans2 (MIT), romanization by IndicXlit (MIT), and the answers by `CohereLabs/tiny-aya-global` (CC-BY-NC-4.0).
The pairs are released under CC-BY-NC-4.0.
