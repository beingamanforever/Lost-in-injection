# Lost in Injection

A prompt-injection defense trained only on English injections protects Tiny Aya Global in 11 languages, scripts and code-mixes, including four it never saw.
Lexsi AI Safety and Alignment Hackathon 2026, Track 2a, problem statements 4 and 3.

![Figure 1](results/figures/headline.png)

The full write-up is in [REPORT.md](REPORT.md), and every design decision, with its date, is in [PLAN.md](PLAN.md).

## Results

| | Undefended | English-only defense |
|---|---|---|
| Attack success, English | 89.9% | 8.7% |
| Attack success, other 10 variants | 51.0-88.9% | 0.5-8.7% |
| Attack success, 4 held-out variants | 69.2-84.6% | 2.9-8.7% |
| Best of all 11 variants per case | 96.6% | 20.2% |
| Global-MMLU-Lite, mean of 5 languages | 55.4% | 55.5% |

AlpacaFarm-Hacked, 208 cases per variant, any of six attacks, leading markdown skipped. The multilingual training runs are being added on Nov 14.

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
