#!/usr/bin/env bash
# Prompting defenses (sandwich, delimiting) and the Prompt Guard 2 detector on one undefended
# model, AlpacaFarm only: InjecAgent already uses the sandwich prompt for every model.
# Usage: scripts/run_prompt_defenses.sh <global|qwen|llama8b>  -> runs/[<family>-]<defense>/0/eval
set -euo pipefail
source env.sh
GEN=.venv-gen/bin/python

family=$1
$GEN -m lost_in_injection.model_dir --model "$family"
model=$($GEN -c "from lost_in_injection.runs import model_path; print(model_path('$family'))")
for name in sandwich isolation promptguard; do
  run=$name
  [ "$family" = global ] || run=$family-$name  # Global keeps bare run names (runs.py)
  defense=$($GEN -c "from lost_in_injection.runs import prompt_defense; print(prompt_defense('$run'))")
  $GEN -m lost_in_injection.evaluate --model "$model" --out "runs/$run/0/eval" \
    --alpaca-defense "$defense" --no-injecagent
done
