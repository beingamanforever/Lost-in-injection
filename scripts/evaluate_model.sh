#!/usr/bin/env bash
# Evaluate one model directory: attacks and utility, then generation quality, then publish.
# Usage: scripts/evaluate_model.sh <model-dir> <run-dir> [evaluate options...]
# <run-dir> is runs/<run>/<seed>; each step is skipped if its output already exists.
set -euo pipefail
source env.sh
GEN=.venv-gen/bin/python
TRAIN=.venv-train/bin/python

model_dir=$1
run_dir=$2
shift 2

$GEN -m lost_in_injection.evaluate --model "$model_dir" --out "$run_dir/eval" "$@"
[ -f "$run_dir/utility/global_mmlu_lite.json" ] ||
  $TRAIN -m lost_in_injection.utility --model "$model_dir" --out "$run_dir/utility"
for benchmark in ifeval indicifeval marco-bench-mif; do
  [ -f "$run_dir/generation/$benchmark.jsonl" ] ||
    $GEN -m lost_in_injection.generation_quality \
      --model "$model_dir" --out "$run_dir/generation" --benchmark "$benchmark"
done
$GEN -m lost_in_injection.publish --run-dir "$run_dir" || echo "upload failed: $run_dir" >&2
