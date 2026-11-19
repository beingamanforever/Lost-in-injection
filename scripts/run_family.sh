#!/usr/bin/env bash
# Build and run one of Phase 2's regional Tiny Aya models, or one of the paper's other model
# families (Llama 3.1 8B, Llama 3.2 3B, Qwen3 4B), end to end.
# Usage: scripts/run_family.sh <fire|earth|water|llama8b|llama3b|qwen>
set -euo pipefail
source env.sh
GEN=.venv-gen/bin/python
TRAIN=.venv-train/bin/python

f=$1
case $f in
  fire | earth | water | llama8b | llama3b | qwen) ;;
  *) echo "usage: scripts/run_family.sh <fire|earth|water|llama8b|llama3b|qwen>" >&2; exit 1 ;;
esac
model=$($GEN -c "from lost_in_injection.runs import model_path; print(model_path('$f'))")

$GEN -m lost_in_injection.model_dir --model "$f"
# The undefended model first, so every later run has its reference.
bash scripts/evaluate_model.sh "$model" "runs/$f-base/0"
$GEN -m lost_in_injection.labels --model "$model" --out "data/pairs/$f/labelled.jsonl"
$TRAIN -m lost_in_injection.export \
  --pairs "data/pairs/$f/labelled.jsonl" --runs "$f-english-only" "$f-english-plus-low-resource"
SEED=0 bash scripts/run_models.sh "$f-english-only" "$f-english-plus-low-resource"
