---
license: cc-by-nc-4.0
language: [en, fr, zh, hi, sw]
pretty_name: Lost in Injection curated pool and labels
tags: [prompt-injection, safety, multilingual, code-mixing, curatorkit]
size_categories: [10K<n<100K]
configs:
- config_name: clean-tasks
  data_files: pool/clean/kept.jsonl
- config_name: injections
  data_files: pool/injections/kept.jsonl
- config_name: labelled-pairs
  data_files: pairs/labelled.jsonl
---

# Lost in Injection curated pool and labels

Everything between the raw Alpaca data and the training pairs of [Lost in Injection](https://github.com/beingamanforever/Lost-in-injection).

- `pool/clean`: 19,099 Alpaca tasks with an input, kept by CuratorKIT after deduplication and its secrets and toxicity gates (from 19,157).
- `pool/injections`: 51,833 Alpaca instructions used as injections (from 52,002).
- `pairs/skeleton.jsonl`: each training slot's task, injection, attack and position.
- `pairs/translations/<language>`: the gated translations of every injection used in training, with the candidates each was chosen from; `romanized-hindi` also keeps the IndicXlit output.
- `pairs/labelled.jsonl`: Tiny Aya Global's own answers to the task (`chosen`) and to the injection in each language (`rejected`), generated at temperature 0.8.

CuratorKIT's manifests, cards and provenance records sit next to each pool.
The rows its gates removed are not published.

## Sources and license

`yahma/alpaca-cleaned` (CC-BY-4.0) and `tatsu-lab/alpaca` (CC-BY-NC-4.0); translations by NLLB-200 (CC-BY-NC-4.0) and IndicTrans2 (MIT); answers by `CohereLabs/tiny-aya-global` (CC-BY-NC-4.0).
Released under CC-BY-NC-4.0.
