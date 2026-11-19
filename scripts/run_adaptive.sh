#!/usr/bin/env bash
# Run the PAIR-style multilingual adaptive attacker against each given run: seed its 208 cases
# from the fixed variants' static-attack outputs, then 10 rounds of attacker rewrite and target
# reply (each its own process; the attacker and the target cannot share one A10G at once).
# Usage: scripts/run_adaptive.sh runs/english-only/0 runs/english-plus-low-resource/0 ...
# Run from the repository root on the A10G after scripts/run_models.sh has evaluated these runs.
set -euo pipefail
source env.sh
GEN=.venv-gen/bin/python

for run in "$@"; do
  $GEN -m lost_in_injection.adaptive_attack seed --run-dir "$run"
done

for round in $(seq 1 10); do
  $GEN -m lost_in_injection.adaptive_attack rewrite --run-dirs "$@"
  for run in "$@"; do
    $GEN -m lost_in_injection.adaptive_attack reply --run-dir "$run"
  done
done

for run in "$@"; do
  .venv-gen/bin/python -m lost_in_injection.publish --run-dir "$run" || echo "upload failed: $run" >&2
done
