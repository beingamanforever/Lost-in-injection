# Plan

## Question

We train a prompt-injection defense into a small multilingual model, following the Meta-SecAlign recipe.
Then we ask whether a defense trained on English injections still holds when the injected instruction arrives in another language, another script, or code-mixed text.
If it does not, we ask what multilingual training data closes the gap without costing utility.

- Model: `CohereLabs/tiny-aya-global` (3.35B, `cohere2`, 36 layers, vocabulary 262,144, 8K context).
- Recipe: Meta-SecAlign ([arXiv 2507.02735](https://arxiv.org/abs/2507.02735), code [facebookresearch/Meta_SecAlign](https://github.com/facebookresearch/Meta_SecAlign) at `2031502`).
- Kits: CuratorKIT (curation and export), AlignTune (DPO), AuditKIT (capability evaluation and run comparison).

## Verified facts this plan builds on

**Tiny Aya** (from its `tokenizer_config.json`)
- The chat template renders only `system`, `user` and `assistant` turns.
- Any other role, such as `input` or `tool`, renders as an empty turn and its content is dropped.
- The tokenizer already contains `<|START_TOOL_RESULT|>` and `<|END_TOOL_RESULT|>`.
- `add_bos_token` is true, and the template also prepends `<BOS_TOKEN>`.
- EOS is `<|END_OF_TURN_TOKEN|>`. Assistant turns end with `<|END_RESPONSE|><|END_OF_TURN_TOKEN|>`.
- The template always injects a system preamble of about 366 tokens.

**Meta-SecAlign** (`utils.py:61-120`, `config.py`, `helpers/llama3.1_8B_lora.yaml`)
- Clean rows are `yahma/alpaca-cleaned` rows with a non-empty input.
- Injected instructions come from Stanford Alpaca.
- Attack mix:
  - 90% straightforward, prepended or appended at random.
  - 10% completion-style, using fake response and instruction markers.
- Chosen is the undefended model's own answer to the clean prompt. Rejected is its answer to the injected instruction alone (vLLM, temperature 0.8).
- Untrusted data sits in an `input` role.
- DPO training: beta 0.1, 3 epochs, LoRA r=64, alpha 8, dropout 0.1 on q, v, gate, up and down projections, learning rate 1.6e-4, cosine schedule, no warmup, no weight decay.
- The repo ships rule-scored InjecAgent and AlpacaFarm-Hacked runners on vLLM with LoRA, and an AgentDojo runner with `--tool-delimiter input`.

**AlignTune** (`create_rl_trainer`, TRL 1.7.1 backend)
- Several defaults differ from the recipe and must be set explicitly:
  - `max_seq_length` 512
  - `truncation_mode` `keep_end`
  - LoRA r=16, alpha 32, dropout 0.05, targeting q/k/v/o
  - `disable_dropout` true
  - weight decay 0.01
  - linear scheduler
  - `report_to` none
- A string prompt is used as-is, with no chat template applied.
- TRL adds BOS to the prompt, and appends EOS only if it is missing.
- The reference model is the base model, through a frozen PEFT adapter.

**CuratorKIT**
- With `generation_task=None`, it ingests prebuilt `prompt/chosen/rejected` rows and runs dedup, hygiene and judge gates, then exports `dpo.jsonl`.
- `clean` strips anything that looks like an HTML tag by default, so `strip_html` must be off.
- The DPO export drops metadata, so we keep a sidecar file of per-pair metadata.

**Environments**
- Meta-SecAlign pins vLLM 0.11.0 and transformers 4.57.1.
- AlignTune needs transformers 5.15 or newer with TRL 1.7.1.
- So we use two environments, `gen` and `train`.

**Measured on the A10G**
- **vLLM 0.11.0 cannot serve LoRA on Tiny Aya.** Loading fails with `'VocabParallelEmbeddingWithLoRA' object has no attribute 'quant_method'`, because the output layer is tied to the input embeddings. Adapters are therefore merged into full weights with PEFT before evaluation. Meta-SecAlign's runners then load them as plain models, with no change to their code or pins.
- **BOS handling.** The template already starts with `<BOS_TOKEN>`. vLLM's `chat` gives exactly one BOS. vLLM's `generate(text)` and the HF tokenizer on rendered text give two. So the leading `<BOS_TOKEN>` is stripped from text prompts at those two points: Meta-SecAlign's runners and the DPO export.
- **Memory.** On the shared GPU, `gpu_memory_utilization=0.45` left 1.2 GiB for the KV cache (17.5K tokens, about 2 concurrent 8K requests). One English evaluation of the base model would have taken about 2.3 hours.
- **Dedicated GPU.** On Nov 2 the OCR demo that shared the A10G was stopped, by the user's decision, and every evaluation now runs at `gpu_memory_utilization=0.90`. The 0.45 pilot was stopped part-way and is not reported.

## Untrusted-data channel

We take Tiny Aya's official template and add one branch, for role `input`:

```
<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|START_TOOL_RESULT|>{content}<|END_TOOL_RESULT|><|END_OF_TURN_TOKEN|>
```

It uses only tokens the model already has, so there are no new embeddings to train.

Untrusted text is stripped of special-token strings before rendering, so it cannot close the tool-result block early.

The template is saved with every adapter, because Meta-SecAlign's evaluation code reads the tokenizer from the adapter directory.

## Reference recipe

Every run starts from this recipe and changes exactly one thing.

| Setting | Value | Source |
|---|---|---|
| Base model | `tiny-aya-global` | |
| Untrusted data | `input` role (template above) | Meta-SecAlign |
| Pairs | 5,000 per run, the same clean and injected rows in every run | Cut to fit if the smoke run predicts more than 3 hours per run |
| Attack mix | 90% straightforward (random position), 10% completion | Meta-SecAlign `utils.py:84-88` |
| Labels | Self-generated by the base model, temperature 0.8, at most 512 new tokens | Meta-SecAlign, with a length cap |
| Length | Pairs longer than 2,048 tokens are dropped, not truncated | |
| DPO | beta 0.1, 3 epochs, effective batch 64 (about 230 optimizer steps, close to Meta-SecAlign's count) | |
| LoRA | r 64, alpha 8, dropout 0.1, q/v/gate/up/down | Meta-SecAlign |
| Optimizer | AdamW, learning rate 1.6e-4, cosine schedule, no warmup, no weight decay | Meta-SecAlign |
| Seeds | Data seed 0 fixed; training seeds 0, 1, 2 for the main runs | |

## Languages

Translated text is only ever the injected instruction and its attack wrapper. The user's task and the clean data stay in English unless a run says otherwise.

| Variant | Translator | GlotLID label | Role |
|---|---|---|---|
| English | none | `eng_Latn` | Always in training |
| French, Spanish, Russian, Chinese | NLLB-200 | `fra_Latn`, `spa_Latn`, `rus_Cyrl`, `cmn_Hani` | Training: high-resource |
| Hindi, Bengali | IndicTrans2 | `hin_Deva`, `ben_Beng` | Training: low-resource |
| Swahili | NLLB-200 | `swh_Latn` | Training: low-resource |
| Romanized Hindi, Hinglish | Aya Expanse 8B rewrite | `hin_Latn` | Training: low-resource |
| Urdu | IndicTrans2 | `urd_Arab` | Held out: Hindi in another script |
| Tamil | IndicTrans2 | `tam_Taml` | Held out |
| Amharic, Yoruba | NLLB-200 | `amh_Ethi`, `yor_Latn` | Held out |
| Romanized Bengali, Taglish | Aya Expanse 8B rewrite | `ben_Latn`, `fil_Latn` | Held out: unseen romanization and code-mixing |

## Runs

**Baselines (no training)**
- `base`: the undefended model.
- `prompted`: the undefended model plus an English system instruction to ignore instructions inside tool results.

**Main comparison (3 seeds each).** These three runs differ only in the languages of the injected instructions in the training pairs.

| Run | Injection languages in training | Question it answers |
|---|---|---|
| `english-only` | 100% English | Does a SecAlign defense learned in English transfer to other languages? |
| `english-plus-high-resource` | 50% English, 50% spread over French, Spanish, Russian, Chinese | Is any multilingual data enough? |
| `english-plus-low-resource` | 50% English, 50% spread over Hindi, Bengali, Swahili, romanized Hindi, Hinglish | Does low-resource and script-varied data close the held-out gap? |

**Ablations (1 seed each).** Each one differs from its reference run in exactly one factor. Its effect is judged against the seed spread of the main runs.

| Run | Reference | Single change | Question it answers |
|---|---|---|---|
| `multilingual-share-10` | `english-plus-low-resource` | 10% non-English instead of 50% | How much multilingual data is needed? |
| `multilingual-share-25` | `english-plus-low-resource` | 25% non-English | Same, with `english-only` (0%) and the reference (50%) completing the curve |
| `hindi-only-mix` | `english-plus-low-resource` | All non-English pairs in Hindi | Language diversity or volume? |
| `native-scripts-only` | `english-plus-low-resource` | Non-English pairs in native scripts only (Hindi, Bengali, Swahili) | Does romanized and code-mixed robustness need explicit data? |
| `no-input-role` | `english-only` | Untrusted data inside the user turn | Does channel separation carry the transfer? |
| `fire-english-only`, `fire-english-plus-low-resource` | the matching Global runs | `tiny-aya-fire` instead of `tiny-aya-global` | Does a South Asia-focused model behave differently? Run only if time allows |

That is 9 main runs plus 7 ablation runs, 16 training runs in total.

## Data

**Training pairs**
1. **Rows.** Load clean rows and injected instructions exactly as Meta-SecAlign does. Pass them through CuratorKIT for MinHash dedup, secrets and toxicity hygiene, and the length gate, with `strip_html` off. Sample 5,000 pairs with data seed 0.
2. **Prompts.** Build each prompt with Meta-SecAlign's attack functions and constants (randomized position, completion attack). The injected instruction is replaced by its translation for the pairs a run assigns to a language.
3. **Translation.** Before translating, mask every identifier (email, URL, number, tool name). After translating, restore them and check that each is present exactly once.
4. **Quality gates.** Every translation must pass all four, or it is dropped and logged with a reason:
   - The target language in GlotLID's top 3.
   - LaBSE cosine of 0.80 or more between the back-translation and the English source.
   - Identifiers intact.
   - For code-mixed variants, both languages present in GlotLID's top 3.
5. **Labels.** Self-generate chosen and rejected with vLLM, once per unique prompt or injected instruction.
6. **Export.** Use CuratorKIT to export `dpo.jsonl` per run, with `manifest.json`, `rejected.jsonl` and the dataset card. Also write `pairs_meta.jsonl` with the pair id, run, language, script, attack type, position, translator and gate scores.

**Evaluation sets.** All are rule-scored, so no judge reads non-English text.
- **AlpacaFarm-Hacked**: Meta-SecAlign's runner. The injected "print exactly Hacked" instruction is translated per variant, and the target word stays `Hacked`.
- **InjecAgent**: Meta-SecAlign's runner on translated copies of the 62 attacker instructions and the "enhanced" hacking prompt.
  - We report direct harm and data-stealing stage 1.
  - Stage 2 needs simulated responses that don't exist for translated cases.
- **AgentDojo**: only if the pilot shows Tiny Aya completes a meaningful share of clean tasks. Meta-SecAlign skips it for its 8B model.

## Metrics

- **Attack success rate (ASR)** per benchmark, variant and setting, with 95% bootstrap intervals over cases.
- **Relative reduction**: base ASR minus run ASR, divided by base ASR, per variant. This normalizes for how strong the attack is in each language before any defense.
- **Transfer gap**: mean relative reduction on training languages minus that on held-out variants.
- **Utility**:
  - Global-MMLU-Lite in English, Hindi, Bengali, Swahili and French, through AuditKIT's lm-eval engine.
  - InjecAgent valid-output rate.
- **Optional, if GPU and time allow**: HarmBench (standard subset) through SafeTune, and a refusal-versus-ignored breakdown of InjecAgent outcomes.

**Evaluation protocol, identical for every run**
- Decoding is Meta-SecAlign's, unchanged: greedy (temperature 0), up to 8,192 new tokens, stopping at EOS, bounded by Tiny Aya's 8,192-token context.
- Engine settings are the same for every model: vLLM 0.11.0, `gpu_memory_utilization=0.90`, merged weights, one BOS.
- Every model sees the same cases, attacks and language variants. No run gets a different token cap, subset or attack list.
- vLLM is not batch-invariant, so rare greedy tokens can differ with batch composition. This affects every run equally and is covered by the seed and bootstrap intervals.

## Checks before any full training run

1. **Template:** render a user, input and assistant conversation. Assert the input content sits inside `<|START_TOOL_RESULT|>…<|END_TOOL_RESULT|>`, each special token is a single id, and there is exactly one BOS after tokenization.
2. **Data:** load `dpo.jsonl` through AlignTune's loader. Assert the columns are exactly `prompt`, `chosen`, `rejected`. Decode 3 examples. Assert every chosen and rejected ends with `<|END_RESPONSE|><|END_OF_TURN_TOKEN|>`, and that no pair exceeds 2,048 tokens.
3. **Config:** dump the resolved TRL `DPOConfig` and LoRA config. Assert each value against the reference recipe table.
4. **Smoke run:** 20 steps on 64 pairs.
   - Log peak GPU memory, which must stay under the free memory measured beforehand.
   - Log step time, and extrapolate the full run time from it.
   - Check that loss and reward margins move in the right direction.
   - Reload the adapter in vLLM and generate.

## Saved outputs

```
runs/<run>/<seed>/
  config.yaml            resolved config, git commit, dataset revision
  adapter/               AlignTune output: adapter, tokenizer with template, lexsi_provenance.json
  logs/                  TensorBoard events, trainer_state.json
  eval/<benchmark>/<variant>/   untouched outputs of the official runners
  metrics.jsonl          one row per (benchmark, variant, setting): n, asr, ci_low, ci_high, valid_rate
results/
  metrics.csv            all runs concatenated; the only input to plotting
  figures/
```

Datasets, adapters and evaluation outputs are pushed to the Hugging Face Hub after each stage. The A10G's working disk is ephemeral.

## Figures

1. A heatmap of ASR by run (rows) and variant (columns), split into training and held-out variants.
2. Relative reduction on held-out variants per main run, with seed error bars.
3. ASR and Global-MMLU-Lite against multilingual share (0, 10, 25, 50%).
4. ASR for Hindi against Urdu, romanized Hindi and Hinglish per run. This separates the effect of script from the effect of language.
5. Utility against security, one point per run.
6. DPO loss and reward-margin curves.

## Hardware and environments

- **GPU:** one A10G (23 GB), dedicated to this project since Nov 2.
- **Working disk:** `/opt/dlami/nvme/aman/lost-in-injection`, symlinked as `/home/ubuntu/aman/lost-in-injection`.
- **`gen` environment** (Meta-SecAlign pins): vLLM label generation, translation, all ASR evaluation.
- **`train` environment** (AlignTune pins): CuratorKIT curation and DPO training.

## Schedule

| Dates | Work |
|---|---|
| Nov 3 | Environments, template check, pilot of the base model on English, Hindi and romanized Hindi |
| Nov 4-5 | Translation and gates, label generation, curation, evaluation sets |
| Nov 5 | Pre-training checks and smoke run |
| Nov 6-8 | Main comparison |
| Nov 8-10 | Ablations |
| Nov 10-11 | Utility and optional metrics |
| Nov 12-13 | Figures, report, HF article and Space |
| Nov 14 | Submit |

## Risks and fallbacks

- **Memory:** if the smoke run doesn't fit in about 12 GB, try in this order:
  1. a 1,536-token cap
  2. 4-bit base weights with the same LoRA
  3. fewer pairs
- **Weak base model:** if Tiny Aya's base InjecAgent valid rate is too low to read ASR, the headline benchmark becomes AlpacaFarm-Hacked and InjecAgent is reported secondarily.
- **Attack strength varies by language:** translation quality can change how strong an attack is. Relative reduction against the base model per variant, plus the translation gates, control for this.
- **Tooling compatibility:** if Meta-SecAlign's pins don't serve Tiny Aya LoRA, upgrade vLLM in the `gen` environment and record the version.
