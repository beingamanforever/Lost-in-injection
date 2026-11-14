<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/readme/logo-dark.svg">
    <img src="docs/readme/logo-light.svg" alt="Lost in Injection" width="560">
  </picture>
</p>

<p align="center">
  A prompt-injection defense trained only in English holds in every language we tested, even ones it never saw,<br>
  and adding low-resource languages to training halves what still gets through.
</p>

<p align="center">
  <a href="REPORT.md"><b>Paper</b></a> &nbsp;·&nbsp;
  <a href="ARTICLE.md"><b>Blog post</b></a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/datasets/beingamanforever/lost-in-injection-model-outputs/blob/main/video/lost-in-injection-demo.mp4"><b>Video</b></a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/collections/beingamanforever/tiny-aya-global-hardened-against-prompt-injection-6af83c90c9b6f21539a70475"><b>Models</b></a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/collections/beingamanforever/lost-in-injection-data-and-model-outputs-6af83c907e17226835782018"><b>Data</b></a>
  <br><sub>Lexsi AI Safety and Alignment Hackathon 2026 · Track 2a, problem statements 4 and 3</sub>
</p>

<p align="center"><img src="docs/readme/attack.gif" width="720" alt="An instruction hidden in a product review makes Tiny Aya Global reply Hacked!"></p>

## What we found

Agents read text an attacker can write, and Meta-SecAlign teaches a model to treat that text as data.
It is trained in English; Tiny Aya is built for everyone else.
We trained it into Tiny Aya Global and attacked it in 11 languages, scripts and code-mixes.

- **It transfers:** trained on English injections alone, the defense also blocks them in French, Chinese, Hindi, Swahili, romanized Hindi and Hinglish, and in Urdu, Tamil, Amharic and Spanglish, which no run ever saw.
- **Languages add up:** an attacker who tries every language on each case still gets through about one time in five.
- **Low-resource data closes most of that:** adding Hindi, Swahili, romanized Hindi and Hinglish injections to training halves it, with utility unchanged.
- **Scorers miss it:** the standard AlpacaFarm-Hacked rule ignores replies that open with markdown, so non-English attacks look far rarer than they are.

<table>
  <tr>
    <td width="50%"><img src="docs/readme/transfer.gif" alt="Attack success per language before and after the English-only defense"></td>
    <td width="50%"><img src="docs/readme/best-of-all.gif" alt="208 cases by 11 languages, collapsed into the best-of-all row"></td>
  </tr>
  <tr>
    <td align="center"><sub>Trained on English, safe in all 11 variants</sub></td>
    <td align="center"><sub>Trying every language, before and after low-resource training</sub></td>
  </tr>
</table>

![Figure 1](results/figures/headline.png)

| | Undefended | English-only | + High-resource | + Low-resource |
|---|---|---|---|---|
| Attack success, English | 89.9% | 8.7% | 11.5% | 7.2% |
| Attack success, 4 held-out variants | 69-85% | 2.9-8.7% | 2.9-6.2% | 0.5-1.9% |
| Best of all 11 variants per case | 97.1% | 20.2% | 15.9% | **8.7%** |
| Global-MMLU-Lite, mean of 5 languages | 55.4% | 55.5% | 55.5% | 56.0% |

<sub>AlpacaFarm-Hacked, 208 cases per variant, any of six attacks, leading markdown skipped.</sub>

## How it works

![Figure 2](docs/figures/method.png)

Each training pair hides an injection in a tool result, then prefers Tiny Aya's own answer to the task over its answer to the injection: Meta-SecAlign's DPO and LoRA recipe, run through AlignTune.
The three defenses share every pair; only the language of the injection changes.

## Reproduce

One 23 GB GPU and two uv environments from `envs/`: `gen` (vLLM 0.11.0, Meta-SecAlign's pins) and `train` (AlignTune, TRL 1.7.1).

```bash
bash scripts/build_data.sh        # curate with CuratorKIT, translate and gate, build the test sets
bash scripts/run_baselines.sh     # undefended and prompted baselines
bash scripts/run_models.sh english-only english-plus-high-resource english-plus-low-resource
python -m lost_in_injection.metrics   # every number, with bootstrap intervals
python -m lost_in_injection.figures   # and lost_in_injection.tables for the report
```

`lost_in_injection/` holds the pipeline, `scripts/` the runs, `results/` the metrics and figures, and `PLAN.md` every design decision with its date.

## Citation

```bibtex
@misc{behera2026lostininjection,
  title  = {Lost in Injection: An English-only Prompt-injection Defense Holds in Eleven Languages},
  author = {Behera, Aman},
  year   = {2026},
  note   = {Lexsi AI Safety and Alignment Hackathon, Track 2a},
  url    = {https://github.com/beingamanforever/Lost-in-injection}
}
```

Code is MIT; models and data keep their own licenses (Tiny Aya Global and NLLB-200 are CC-BY-NC-4.0).
