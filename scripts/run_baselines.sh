#!/usr/bin/env bash
# Evaluate the two untrained baselines: the base model, and the base model with Meta-SecAlign's
# reminder prompt on AlpacaFarm-Hacked (InjecAgent already uses the sandwich prompt for everyone).
# Run from the repository root on the A10G after scripts/build_data.sh.
set -euo pipefail
source env.sh
GEN=.venv-gen/bin/python
ALL="english french chinese hindi swahili romanized-hindi hinglish urdu tamil amharic spanglish"
$GEN -m lost_in_injection.evaluate --model models/tiny-aya-global --out runs/base/0/eval --variants $ALL
$GEN -m lost_in_injection.evaluate --model models/tiny-aya-global --out runs/prompted/0/eval \
  --variants $ALL --alpaca-defense reminder --no-injecagent
.venv-train/bin/python -m lost_in_injection.utility --model models/tiny-aya-global --out runs/base/0/utility
for run in base prompted; do
  $GEN -m lost_in_injection.publish --run-dir runs/$run/0 || echo "upload failed: $run" >&2
done
