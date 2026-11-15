#!/usr/bin/env bash
# Build and run one of Phase 2's regional models end to end.
# Usage: scripts/run_family.sh <fire|earth|water>
set -euo pipefail
source env.sh
GEN=.venv-gen/bin/python
TRAIN=.venv-train/bin/python

f=$1
case $f in
  fire | earth | water) ;;
  *) echo "usage: scripts/run_family.sh <fire|earth|water>" >&2; exit 1 ;;
esac

$GEN -m lost_in_injection.model_dir --model-id "CohereLabs/tiny-aya-$f" --out "models/tiny-aya-$f"
$GEN -m lost_in_injection.labels --model "models/tiny-aya-$f" --out "data/pairs/$f/labelled.jsonl"
$TRAIN -m lost_in_injection.export \
  --pairs "data/pairs/$f/labelled.jsonl" --runs "$f-english-only" "$f-english-plus-low-resource"
SEED=0 bash scripts/run_models.sh "$f-english-only" "$f-english-plus-low-resource"
bash scripts/evaluate_model.sh "models/tiny-aya-$f" "runs/$f-base/0"
