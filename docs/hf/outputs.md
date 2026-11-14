---
license: cc-by-nc-4.0
pretty_name: Lost in Injection model outputs
tags: [prompt-injection, safety, multilingual, evaluation]
viewer: false
---

# Lost in Injection model outputs

Every reply each model gave in [Lost in Injection](https://github.com/beingamanforever/Lost-in-injection), with the scores and logs behind the report, and the demo video.

| Folder | Contents |
|---|---|
| `<model>/0/eval/<variant>/` | AlpacaFarm-Hacked replies for all six attacks in 11 variants; InjecAgent replies in English and Hindi |
| `<model>/0/utility/` | Global-MMLU-Lite predictions in English, Hindi, Swahili, French and Chinese |
| `<model>/0/logs/`, `<model>/0/config.json` | DPO trainer logs and the resolved training config, for the three defenses |
| `video/` | The 58-second demo video |

The models are `base` (undefended), `prompted` (Meta-SecAlign's reminder prompt), `english-only`, `english-plus-high-resource` and `english-plus-low-resource`.
`base-first-a10g-0.90/` is an earlier undefended run that the report does not use.
`lost_in_injection/metrics.py` in the repository turns these files into `results/metrics.csv`, which every number, figure and table reads.

Released under CC-BY-NC-4.0, as the model and benchmarks that produced them.
