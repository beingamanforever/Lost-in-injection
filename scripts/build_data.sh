#!/usr/bin/env bash
# Build the evaluation sets and the training data, in order, across the three environments.
# Run from the repository root on the A10G after scripts/fetch_data.sh.
set -euo pipefail
source env.sh
GEN=.venv-gen/bin/python
TRAIN=.venv-train/bin/python
XLIT=.venv-xlit/bin/python
MACHINE_TRANSLATED="hindi urdu french chinese swahili tamil amharic spanish"
EXTRA_HELD_OUT="arabic turkish russian greek korean thai bengali yoruba"

# Evaluation sets.
$GEN -m lost_in_injection.eval_sets --variants english
scripts/eval_sets.sh $MACHINE_TRANSLATED
scripts/eval_sets.sh $EXTRA_HELD_OUT
CUDA_VISIBLE_DEVICES="" $XLIT -m lost_in_injection.romanize --root data/eval
$GEN -m lost_in_injection.eval_sets --variants romanized-hindi hinglish spanglish

# Training data: curate, draw the skeleton, translate the pool, label, export one dataset per run.
CUDA_VISIBLE_DEVICES="" $TRAIN -m lost_in_injection.pool  # Detoxify runs on the CPU here
$GEN -m lost_in_injection.skeleton
$GEN -m lost_in_injection.translate_pairs --languages french chinese hindi swahili
CUDA_VISIBLE_DEVICES="" $XLIT -m lost_in_injection.romanize --root data/pairs/translations
$GEN -m lost_in_injection.translate_pairs --languages romanized-hindi hinglish
$GEN -m lost_in_injection.labels
$TRAIN -m lost_in_injection.export
