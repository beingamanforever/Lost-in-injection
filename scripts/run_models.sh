#!/usr/bin/env bash
# Train, merge, evaluate and score utility for each named run (seed 0), one after another.
# Usage: scripts/run_models.sh english-only english-plus-low-resource ...
# Run from the repository root on the A10G after scripts/build_data.sh and run_baselines.sh.
set -euo pipefail
source env.sh
GEN=.venv-gen/bin/python
TRAIN=.venv-train/bin/python
# Uploads go to private Hugging Face repositories; a failed upload is logged and never stops a run.
publish() { $GEN -m lost_in_injection.publish --run-dir "$1" || echo "upload failed: $1" >&2; }
SEED="${SEED:-0}"  # SEED=1 bash scripts/run_models.sh english-only trains and evaluates seed 1
for run in "$@"; do
  out=runs/$run/$SEED
  # MICRO_BATCH=8 on an 80 GB GPU; the effective batch stays 64 (train.py).
  $TRAIN -m lost_in_injection.train --run "$run" --seed "$SEED" --micro-batch "${MICRO_BATCH:-1}"
  publish "$out"  # The adapter first: the first A10G's working disk does not survive a reboot.
  roles=""
  [ "$run" = no-input-role ] && roles="--no-input-role"  # Tested the way it was trained
  if [ -n "${ADAPTER_AT_INFERENCE:-}" ]; then
    # Merging into bf16 rounds most of the update away (Qwen3-4B, Nov 19): test the adapter as is.
    mkdir -p "$out/lora"
    ln -sf ../adapter/adapter_config.json ../adapter/adapter_model.safetensors "$out/lora/"
    base=$($GEN -c "from lost_in_injection.runs import base_model; print(base_model('$run'))")
    ADAPTER="$out/lora" bash scripts/evaluate_model.sh "$base" "$out" $roles
  else
    CUDA_VISIBLE_DEVICES="" $TRAIN -m lost_in_injection.merge --run-dir "$out"
    bash scripts/evaluate_model.sh "$out/merged" "$out" $roles
  fi
done
