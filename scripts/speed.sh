#!/usr/bin/env bash
# Serving speed of a model, with or without a defense adapter applied at runtime (ACL plan E8):
# vLLM's own benchmark client, random 256-token prompts and 256-token replies, at 1, 8 and 32
# concurrent requests. A merged defense has the base model's architecture, so it runs at base speed.
# Usage: scripts/speed.sh <name> <model-dir> [adapter-dir]   -> runs/speed/<name>/c<N>.json
set -euo pipefail
source env.sh
VLLM=.venv-gen/bin/vllm
name=$1
model=$2
adapter=${3:-}
out=runs/speed/$name
mkdir -p "$out"

serve=(--served-model-name base)
target=base
if [ -n "$adapter" ]; then
  serve+=(--enable-lora --max-lora-rank 64 --lora-modules "defense=$adapter")
  target=defense
fi
# Same memory share and context as every evaluation (evaluate.py).
$VLLM serve "$model" --port 8100 --gpu-memory-utilization 0.80 --max-model-len 8192 \
  "${serve[@]}" > "$out/server.log" 2>&1 &
server=$!
trap 'kill $server; wait $server || true' EXIT
until .venv-gen/bin/python -c "import urllib.request as u; u.urlopen('http://localhost:8100/health')" \
  2> /dev/null; do
  kill -0 $server  # stop waiting if the server died
  sleep 5
done

for concurrency in 1 8 32; do
  $VLLM bench serve --backend vllm --port 8100 --model "$target" --tokenizer "$model" \
    --dataset-name random --random-input-len 256 --random-output-len 256 --ignore-eos --seed 0 \
    --max-concurrency $concurrency --num-prompts $((concurrency * 16)) \
    --save-result --result-dir "$out" --result-filename "c$concurrency.json"
done
