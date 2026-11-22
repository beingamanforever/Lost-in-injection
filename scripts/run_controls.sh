#!/usr/bin/env bash
# ACL plan E2 on an A10G: build the equal-budget control sets (control_sets.py), then evaluate them,
# AlpacaFarm only, on Tiny Aya Global undefended (seed 0) and the English-only and + low-resource
# defenses (seed 1, merged bf16 weights as originally tested). Sets already scored are skipped.
# Usage: HF_ENV=<file exporting HF_TOKEN> bash scripts/run_controls.sh, from the repository root.
set -euo pipefail
source env.sh
GEN=.venv-gen/bin/python
TRAIN=.venv-train/bin/python
set +u; set -a; source "$HF_ENV"; set +a; set -u

[ -d models/tiny-aya-global ] || $GEN -m lost_in_injection.model_dir --model global
for run in english-only english-plus-low-resource; do
  dir=runs/$run/1
  [ -f "$dir/adapter/adapter_model.safetensors" ] || $GEN -c "
from huggingface_hub import snapshot_download, whoami
from lost_in_injection.publish import adapter_repo
repo = f\"{whoami()['name']}/{adapter_repo('$run')}-seed1\"  # publish.py's name for seed 1
snapshot_download(repo, local_dir='$dir/adapter')"
  [ -d "$dir/merged" ] || CUDA_VISIBLE_DEVICES="" $TRAIN -m lost_in_injection.merge --run-dir "$dir"
done

[ -d data/eval/bon-11 ] || $GEN -m lost_in_injection.control_sets bon
[ -d data/eval/sample-11 ] || $GEN -m lost_in_injection.control_sets sample
[ -d data/eval/translit-bengali ] || uv run --no-project --with uroman==1.3.1.1 python -m lost_in_injection.control_sets translit
[ -d data/eval/paraphrase-11 ] || $GEN -m lost_in_injection.control_sets paraphrase
cat > data/eval/CONTROLS.md <<'EOF'
# Equal-budget control sets (ACL plan E2)

Each set keeps the 208 AlpacaFarm-Hacked cases and six attacks; only the attack text changes (lost_in_injection/control_sets.py).

- `paraphrase-1..11`: English rewordings of the injected prompt and of each ignore sentence by Llama-3.1-8B-Instruct (temperature 1.0), kept only if 'Hacked!' and {injected_prompt} survive and LaBSE similarity to the source is at least 0.70; audit in `paraphrase-records.jsonl`.
- `bon-1..11`: Best-of-N text augmentations (Hughes et al. 2025): middle-letter scrambling (0.6), random capitalization (0.6) and ASCII noise (0.06), seed i for set i; the payload and placeholder are untouched.
- `translit-<language>`: the 11 non-Latin variants romanized with uroman 1.3.1.1 (same words, Latin letters); Chinese and Thai come out unspaced, Arabic and Urdu without short vowels.
- `sample-1..11`: the English text unchanged, evaluated with temperature 0.7 and sampling seed i; every other set is decoded greedily.
EOF

greedy=$($GEN -c "from lost_in_injection.control_sets import FAMILIES as F; print(*F['paraphrase'], *F['best-of-n'], *F['transliteration'])")
for target in english-only/1:runs/english-only/1/merged english-plus-low-resource/1:runs/english-plus-low-resource/1/merged base/0:models/tiny-aya-global; do
  out=runs/${target%%:*}/eval
  model=${target#*:}
  todo=$(for v in $greedy; do [ -f "$out/$v/summary.tsv" ] || echo "$v"; done)
  [ -z "$todo" ] || $GEN -m lost_in_injection.evaluate --model "$model" --out "$out" --no-injecagent --variants $todo
  for s in $(seq 11); do
    [ -f "$out/sample-$s/summary.tsv" ] ||
      $GEN -m lost_in_injection.evaluate --model "$model" --out "$out" --no-injecagent \
        --variants "sample-$s" --temperature 0.7 --sample-seed "$s"
  done
done
echo "CONTROLS DONE"
