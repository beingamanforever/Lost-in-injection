---
license: cc-by-nc-4.0
language: [en, fr, zh, hi, sw, ur, ta, am, es]
pretty_name: Lost in Injection evaluation sets
tags: [prompt-injection, safety, multilingual, code-mixing, alpacafarm]
size_categories: [n<1K]
---

# Lost in Injection evaluation sets

AlpacaFarm-Hacked's injections in 11 languages, scripts and code-mixes, as used to attack the defenses in [Lost in Injection](https://github.com/beingamanforever/Lost-in-injection).
Meta-SecAlign's runners insert them into the 208 AlpacaFarm cases that have data.
Only the injected instruction and its attack sentences are translated; the user's task stays in English.

| Group | Folders |
|---|---|
| English | `english` |
| High-resource | `french`, `chinese` |
| Low-resource | `hindi`, `swahili`, `romanized-hindi`, `hinglish` |
| Held out from every training run | `urdu`, `tamil`, `amharic`, `spanglish` |

Each folder holds `alpacafarm_injection.json`, the injected prompt and attack sentences in that variant, and `translations.jsonl`, each English source with the kept translation and the candidates it was chosen from.
`spanish` is the Spanish translation that Spanglish is built from.
Every text passed three gates: masked identifiers come back exactly once, GlotLID puts the target language in its top 3, and the back-translation scores LaBSE cosine 0.80 or more.

## Sources and license

AlpacaFarm (`tatsu-lab/alpaca_farm`, CC-BY-NC-4.0) through Meta-SecAlign's evaluation code; translations by NLLB-200 (CC-BY-NC-4.0), IndicTrans2 (MIT) and, for one Chinese sentence, Aya Expanse 8B (CC-BY-NC-4.0).
Released under CC-BY-NC-4.0.
