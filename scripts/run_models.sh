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
ALL="english french chinese hindi swahili romanized-hindi hinglish urdu tamil amharic spanglish"
for run in "$@"; do
  out=runs/$run/0
  $TRAIN -m lost_in_injection.train --run "$run"
  CUDA_VISIBLE_DEVICES="" $TRAIN -m lost_in_injection.merge --run-dir "$out"
  publish "$out"  # The adapter first: the first A10G's working disk does not survive a reboot.
  $GEN -m lost_in_injection.evaluate --model "$out/merged" --out "$out/eval" --variants $ALL
  $TRAIN -m lost_in_injection.utility --model "$out/merged" --out "$out/utility"
  publish "$out"
done
