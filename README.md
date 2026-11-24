<p align="center"><img src="docs/readme/logo.svg" alt="Lost in Injection" width="560"></p>

<p align="center">
  Prompt-injection defenses trained into a model hold in every language we tested, one language at a time.<br>
  They leak through an input filter that cannot read Swahili, through attackers who keep rewording,<br>
  and through a scoring rule that only reads the first word of a reply.
</p>

<p align="center">
  <a href="REPORT.md"><b>Paper</b></a> &nbsp;·&nbsp;
  <a href="ARTICLE.md"><b>Blog post</b></a> &nbsp;·&nbsp;
  <a href="docs/lost-in-injection.mp4"><b>Video</b></a> &nbsp;·&nbsp;
  <a href="docs/lost-in-injection-slides.pdf"><b>Slides</b></a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/collections/beingamanforever/lost-in-injection-models-6af83c90c9b6f21539a70475"><b>Models</b></a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/collections/beingamanforever/lost-in-injection-datasets-6af83c907e17226835782018"><b>Data</b></a>
  <br><sub>Lexsi AI Safety and Alignment Hackathon 2026 · Track 2a, problem statements 4 and 3</sub>
  <br><sub>Long paper submitted to three AAAI-27 workshops: LAUT, SPF-LLM and AICS</sub>
</p>

<p align="center"><img src="docs/readme/attack.gif" width="720" alt="An instruction hidden in a product review makes Tiny Aya Global reply Hacked!"></p>

## What we found

Agents read text an attacker can write, and the strongest open defense, Meta-SecAlign, teaches a model to treat that text as data.
It was built and tested in English.
We attacked eight defenses on seven models from three families, from Tiny Aya (3B) to Llama 3.3 (70B), with the same injections in 11 languages, scripts and code-mixes (18 on most runs), and scored 940,992 replies three ways.

- Trained defenses transfer. Trained on English injections alone, the defense cuts attack success in every language as far as in English, including languages no run saw. No single language beats English by more than a few points, for our defenses on Tiny Aya, Qwen3 and Llama 3.1 and for Meta's own Meta-SecAlign-8B and -70B.
- The input filter does not. Meta's Prompt Guard 2 removes 33-71% of English attacks but only 6-35% of Swahili and 7-12% of romanized-Hindi ones. Behind it, one Swahili attempt hijacks Tiny Aya in 83% of cases, against 39% in English.
- Extra tries get through. Trying all 11 languages on each case still hijacks 21% of cases against our English-only Tiny Aya defense. Eleven English paraphrases do as well, and after low-resource training better, so the risk is variety, of which language is one cheap source.
- The standard scorer undercounts. Meta-SecAlign's rule counts a hijack only if the reply starts with "Hacked". It misses replies that open with `**Hacked!**` and replies that finish the task and append the payload. An LLM judge puts our English-only Tiny Aya defense at 33% in English, not 9%, and finds a language gap even for Meta-SecAlign-70B.
- Low-resource data helps. Adding Hindi, Swahili, romanized Hindi and Hinglish injections to training lowers the 11-language rate on all six models we trained, by up to 34 points on Qwen3, at no cost on Tiny Aya.
- Defenses have costs elsewhere. Nothing measurable on Tiny Aya (IFEval, Global-MMLU-Lite, MATH-500, MGSM), but our recipe over-trains Llama 3.1 8B on a quarter of Meta's data and costs it 8-14 IFEval points (Meta-SecAlign costs one; the report traces why), a runtime LoRA adapter costs 22-29% of serving throughput, and merging it into bf16 weights instead can erase it: Qwen3's merged English-only defense let 90% of English attacks through, against 41% at runtime.
- Agent benchmarks sit near their floor. On AgentDojo in five languages, Llama 3.1 8B solves 16.5% of tasks and attacks succeed in under 4% of cases even undefended, too few to tell languages or defenses apart.

![The four findings at a glance](results/figures/findings.png)

<sub>Cases hijacked (%) of 208. (a) One try per language, LLM judge: the worst language stays near English for every trained defense (MSA: Meta-SecAlign). (b) Behind Prompt Guard 2, one Swahili try gets through in 83% of cases, against 39% in English. (c) Against our English-only defense, 11 languages hijack 17 points more than one try, and 11 English rewordings as many. (d) The same English replies: 9% hijacked under the starts-with rule, 33% under the judge.</sub>

<sub>\* The models do not claim the same languages. Tiny Aya's model cards list every language we test, Qwen3's release all but Amharic, and Llama 3.1 and 3.3 only English, French, Hindi, Spanish and Thai. Undefended Llama 3.1 8B follows non-English injections so rarely that its gaps say little about the defense. The low-resource training mix is Hindi, Swahili, romanized Hindi and Hinglish. [Table 1 of the report](REPORT.md#which-model-was-tested-on-what) lists which model was tested on which languages and benchmarks.</sub>

<table>
  <tr>
    <td width="50%"><img src="results/figures/defense_gap.png" alt="English against all 11 languages, for every defense and model"></td>
    <td width="50%"><img src="results/figures/utility_change.png" alt="Each defense's benchmark scores against the same model undefended"></td>
  </tr>
  <tr>
    <td align="center"><sub>English (open) against all 11 languages (filled), every defense and model</sub></td>
    <td align="center"><sub>What each defense costs on IFEval, IndicIFEval and Global-MMLU-Lite</sub></td>
  </tr>
  <tr>
    <td width="50%"><img src="results/figures/promptguard_languages.png" alt="Share of attacks Prompt Guard 2 removes per language and model"></td>
    <td width="50%"><img src="results/figures/scoring_rules_three.png" alt="The same replies under three definitions of a hijack"></td>
  </tr>
  <tr>
    <td align="center"><sub>Prompt Guard 2 catches English, misses Swahili and romanized Hindi</sub></td>
    <td align="center"><sub>The same replies under three definitions of a hijack</sub></td>
  </tr>
  <tr>
    <td width="50%"><img src="results/figures/equal_budget.png" alt="Cases hijacked by translations, paraphrases, samples and noise at equal numbers of tries"></td>
    <td width="50%"><img src="results/figures/low_resource.png" alt="Language search before and after low-resource training, per model family"></td>
  </tr>
  <tr>
    <td align="center"><sub>At equal tries, English paraphrases match translations</sub></td>
    <td align="center"><sub>Low-resource training lowers the all-language rate on every model</sub></td>
  </tr>
</table>

## How it works

![Method](docs/figures/method.png)

Each training pair hides an injection in a tool result, then prefers the model's own answer to the task over its answer to the injection: Meta-SecAlign's DPO and LoRA recipe, run through AlignTune.
The defenses share every pair; only the language of the injection changes.
We attack them one language at a time, with language search over every variant, with the same number of tries in English (paraphrases, samples, character noise), and with a PAIR-style adaptive attacker (Aya Expanse 8B).
Every reply is scored by Meta-SecAlign's rule, by "Hacked" anywhere, and by an LLM judge (qwen3.7-flash) that sees exactly what the model saw and must quote its evidence.

## Reproduce

One 24 GB GPU and three uv environments from `envs/`: `gen` (vLLM 0.11.0, Meta-SecAlign's pins), `train` (AlignTune, TRL 1.7.1) and `xlit` (IndicXlit romanization).
Run everything from the repository root: paths are relative, and models and data download from Hugging Face and pinned GitHub revisions.
Log in to Hugging Face with access to the gated Llama 3.1, Llama 3.3, Prompt Guard 2 and Meta-SecAlign models.
`modal_app.py` runs the same commands on Modal for the 70B model.

```bash
bash envs/setup.sh                # the three environments
bash scripts/build_data.sh        # curate with CuratorKIT, translate and gate, build the test sets
bash scripts/run_baselines.sh     # undefended and prompted baselines
bash scripts/run_models.sh english-only english-plus-high-resource english-plus-low-resource
bash scripts/run_family.sh qwen   # and fire, earth, water, llama8b
bash scripts/run_meta_secalign.sh llama8b           # and llama70b, on two H200s
bash scripts/run_prompt_defenses.sh qwen            # sandwich, isolation, Prompt Guard 2
bash scripts/run_controls.sh                        # equal-budget control sets
bash scripts/run_agentdojo.sh 0 llama8b-base:no-attack,english,hindi,swahili,chinese,tamil   # and each defense
python -m lost_in_injection.judge --runs-root runs --api-model qwen/qwen3.7-flash   # needs OPENROUTER_API_KEY
python -m lost_in_injection.metrics --runs runs --out results
python -m lost_in_injection.analysis --runs runs --out results/acl
python -m lost_in_injection.judged --runs runs --out results/acl
python -m lost_in_injection.figures && python -m lost_in_injection.defense_figures
```

Qwen and Llama defenses are tested with the adapter applied at inference (`evaluate.py --adapter`), because merging into bf16 weights rounds most of the update away.
`lost_in_injection/` holds the pipeline, `scripts/` the runs, `results/` every number, and `video/` the Manim source of the video.

## Citation

```bibtex
@misc{behera2026lostininjection,
  title  = {Lost in Injection: Multilingual Attacks on Prompt-Injection Defenses},
  author = {Behera, Aman},
  year   = {2026},
  note   = {Lexsi AI Safety and Alignment Hackathon, Track 2a},
  url    = {https://github.com/beingamanforever/Lost-in-injection}
}
```

Code is MIT; models and data keep their own licenses (Tiny Aya and NLLB-200 are CC-BY-NC-4.0, Qwen3-4B is Apache-2.0, Llama 3.1 and 3.3 and Meta-SecAlign follow the Llama licenses).
