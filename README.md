# Lost in Injection

A prompt-injection defense trained only on English injections protects Tiny Aya Global in 11 languages, scripts and code-mixes, including four it never saw.
Adding low-resource languages to its training injections more than halves what an attacker who tries every language can still break.
Lexsi AI Safety and Alignment Hackathon 2026, Track 2a, problem statements 4 and 3.

![Figure 1](results/figures/headline.png)

The full write-up is in [REPORT.md](REPORT.md), a shorter first-person version is in [ARTICLE.md](ARTICLE.md), and every design decision, with its date, is in [PLAN.md](PLAN.md).

## Results

| | Undefended | English-only | + High-resource | + Low-resource |
|---|---|---|---|---|
| Attack success, English | 89.9% | 8.7% | 11.5% | 7.2% |
| Attack success, other 10 variants | 51-89% | 0.5-8.7% | 0.0-6.2% | 0.0-1.9% |
| Attack success, 4 held-out variants | 69-85% | 2.9-8.7% | 2.9-6.2% | 0.5-1.9% |
| Best of all 11 variants per case | 97.1% | 20.2% | 15.9% | 8.7% |
| Global-MMLU-Lite, mean of 5 languages | 55.4% | 55.5% | 55.5% | 56.0% |

AlpacaFarm-Hacked, 208 cases per variant, any of six attacks, leading markdown skipped.
The defenses differ only in the language of their training injections: English, English with French and Chinese, or English with Hindi, Swahili, Hindi in Latin script and Hinglish.

Adapters, data and every model output are on Hugging Face, in [the model collection](https://huggingface.co/collections/beingamanforever/tiny-aya-global-hardened-against-prompt-injection-6af83c90c9b6f21539a70475) and [the dataset collection](https://huggingface.co/collections/beingamanforever/lost-in-injection-data-and-model-outputs-6af83c907e17226835782018).
A 58-second video walks through it: [`video/`](video) has the source, and the [rendered file](https://huggingface.co/datasets/beingamanforever/lost-in-injection-model-outputs/blob/main/video/lost-in-injection-demo.mp4) sits with the outputs.

## Reproduce

The pipeline runs on one 23 GB GPU with two uv environments, `gen` (vLLM 0.11.0, Meta-SecAlign's pins) and `train` (AlignTune, TRL 1.7.1), defined in `envs/`.

```bash
bash scripts/build_data.sh
bash scripts/run_baselines.sh
bash scripts/run_models.sh english-only english-plus-high-resource english-plus-low-resource
python -m lost_in_injection.metrics
python -m lost_in_injection.figures
python -m lost_in_injection.tables
docs/figures/render.sh
```

`build_data.sh` curates the pool with CuratorKIT, translates and gates the injections, and builds the evaluation sets.
`run_models.sh` trains each defense with AlignTune, merges it, runs Meta-SecAlign's evaluators and scores utility with AuditKIT; `publish.py` uploads adapters and outputs to Hugging Face.
`metrics.py` writes `results/metrics.csv` with bootstrap intervals, which every figure and table reads.

## Layout

| Path | Contents |
|---|---|
| `lost_in_injection/` | Pool, translation and gates, code-mixing, labels, export, training, merging, evaluation, metrics, figures, tables, leak check |
| `scripts/` | End-to-end run scripts |
| `docs/figures/` | Sources for the method figure and tables, rendered with headless Chrome |
| `results/` | Metrics, training curves and figures |
| `tests/` | Unit tests for metrics, figures and publishing |

## License

Code: MIT. Models and data keep their own licenses (Tiny Aya Global and NLLB-200 are CC-BY-NC-4.0); see REPORT.md, sections 3 and 4.
