#!/usr/bin/env bash
# Evaluate Meta's own English-only defense, Meta-SecAlign-8B or -70B, like every other model.
# Its Hugging Face repos are gated: the account needs approved access first. Meta tests it with
# the adapter applied at runtime, and merging into bf16 would round most of it away: so do we.
# Usage: scripts/run_meta_secalign.sh [llama8b|llama70b]   (70B needs two 141 GB GPUs)
set -euo pipefail
source env.sh
GEN=.venv-gen/bin/python

model=${1:-llama8b}
run=runs/$model-meta-secalign/0
$GEN -m lost_in_injection.model_dir --model "$model"
$GEN -m lost_in_injection.model_dir --model "$model" --meta-secalign-adapter "$run/adapter"
base=$($GEN -c "from lost_in_injection.runs import model_path; print(model_path('$model'))")
ADAPTER="$run/adapter" bash scripts/evaluate_model.sh "$base" "$run"
