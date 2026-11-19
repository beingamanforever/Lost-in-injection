#!/usr/bin/env bash
# Evaluate Meta's own English-only defense, Meta-SecAlign-8B, like every other model.
# Its Hugging Face repo is gated: the account needs approved access first.
# Usage: scripts/run_meta_secalign.sh
set -euo pipefail
source env.sh
GEN=.venv-gen/bin/python
TRAIN=.venv-train/bin/python

run=runs/llama8b-meta-secalign/0
$GEN -m lost_in_injection.model_dir --model llama8b
$GEN -m lost_in_injection.model_dir --meta-secalign-adapter "$run/adapter"
CUDA_VISIBLE_DEVICES="" $TRAIN -m lost_in_injection.merge --run-dir "$run"
bash scripts/evaluate_model.sh "$run/merged" "$run"
