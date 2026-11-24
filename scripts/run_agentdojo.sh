#!/usr/bin/env bash
# AgentDojo v1.2.1 on Llama 3.1 8B runs with the attacker's text in other languages, under
# Meta-SecAlign's protocol (third_party/Meta_SecAlign/test_agentdojo.py): `vllm serve`, then
# AgentDojo's benchmark with --model local --tool-delimiter input (tool results in the `input` role)
# --defense repeat_user_prompt and the important_instructions attack, translated per language by
# lost_in_injection/agentdojo_attacks.py. One server holds the base model and applies each adapter at
# runtime (never merged). A run's languages go concurrently, and so do the suites within each;
# finished cases are skipped on a rerun. Logs go to runs/<run>/0/agentdojo/<language>/, where
# `no-attack` is utility without an attack. Needs scripts/setup_agentdojo.sh and, for translated
# languages, data/agentdojo/attack_texts.json (lost_in_injection.agentdojo_languages translate).
# Usage: bash scripts/run_agentdojo.sh <pid to wait for, or 0> <run>:<language>[,<language>...] ...
#   e.g. bash scripts/run_agentdojo.sh 0 llama8b-base:no-attack,english,hindi,swahili
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh
LOGS=${LOGS:-logs}
mkdir -p "$LOGS"
wait_pid=$1
shift
while [ "$wait_pid" != 0 ] && kill -0 "$wait_pid" 2>/dev/null; do sleep 60; done

# The same server for every model and language (greedy decoding: AgentDojo sends temperature 0).
# Meta's 24,576-token context fits an A10G (smoke test, driver 535: 34,096 tokens of KV cache, CUDA
# graphs on, 21.4 of 23 GB used). Replies are capped so a looping reply ends, and stop after the
# first function call (lost_in_injection/agentdojo_attacks.py).
MAX_MODEL_LEN=24576
GPU_MEMORY_UTILIZATION=0.90
MAX_NUM_SEQS=16
MAX_NEW_TOKENS=4096
modules=()
for run in llama8b-meta-secalign llama8b-english-only llama8b-english-plus-low-resource; do
  modules+=("$run=$PWD/runs/$run/0/adapter")
done
setsid .venv-gen/bin/vllm serve models/llama-3.1-8b-instruct --served-model-name llama8b-base \
  --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --max-num-seqs $MAX_NUM_SEQS --override-generation-config "{\"max_new_tokens\": $MAX_NEW_TOKENS}" \
  --enable-lora --max-lora-rank 64 --lora-modules "${modules[@]}" \
  > "$LOGS/agentdojo-vllm.log" 2>&1 < /dev/null &
server=$!                       # setsid made it a process-group leader; this stops its workers too
trap 'kill -- -$server 2>/dev/null' EXIT
until curl -sf localhost:8000/v1/models > /dev/null; do
  kill -0 $server 2>/dev/null || { echo "vLLM server failed; see $LOGS/agentdojo-vllm.log"; exit 1; }
  sleep 10
done

# AgentDojo runs a suite's cases one after another, and workspace holds 560 of the 949 cases with an
# injection, so it gets three processes over disjoint thirds of its user tasks. The later two log to
# their own folders: each process first runs the injection tasks as user tasks, and two processes
# writing those logs in one folder would collide.
mapfile -t workspace < <(.venv-agentdojo/bin/python -c "from agentdojo.task_suite.load_suites \
import get_suite; print(*get_suite('v1.2.1', 'workspace').user_tasks, sep='\n')" 2> /dev/null)
[ ${#workspace[@]} = 40 ] || { echo "expected 40 workspace user tasks, got ${#workspace[@]}"; exit 1; }
benchmark() { # <log name> <log folder> <benchmark options...>, for $run and $attack
  .venv-agentdojo/bin/python -m agentdojo.scripts.benchmark --model local --model-id "$run" \
    --tool-delimiter input --defense repeat_user_prompt "${attack[@]}" \
    -ml lost_in_injection.agentdojo_attacks --logdir "$2" "${@:3}" \
    > "$LOGS/agentdojo-$run-$1.log" 2>&1 &
  pids+=($!)
  names+=("$1")
}

for job in "$@"; do
  run=${job%%:*}
  started=$(date +%s)
  pids=()
  names=()
  IFS=, read -ra languages <<< "${job#*:}"
  for language in "${languages[@]}"; do
    dir=runs/$run/0/agentdojo/$language
    case $language in
      no-attack) attack=() ;;
      english) attack=(--attack important_instructions) ;;
      *) attack=(--attack "important_instructions_$language") ;;
    esac
    if [ "$language" = no-attack ]; then
      benchmark "$language" "$dir" --max-workers 4
      continue
    fi
    benchmark "$language" "$dir" -s banking -s slack -s travel --max-workers 3
    for part in 0 1 2; do
      tasks=()
      for task in "${workspace[@]:part*14:14}"; do tasks+=(-ut "$task"); done
      folder=$dir
      [ $part -gt 0 ] && folder=$dir/workspace-$part
      benchmark "$language-workspace-$part" "$folder" -s workspace "${tasks[@]}"
    done
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || echo "FAILED $run ${names[$i]}"
  done
  for language in "${languages[@]}"; do
    errors=$(cat "$LOGS/agentdojo-$run-$language"*.log | grep -c '\[debug\] error')
    echo "== $(date -u +%m-%d_%H:%M) $run $language: $errors request errors"
  done
  echo "== $(date -u +%m-%d_%H:%M) $run done in $(( ($(date +%s) - started) / 60 )) min"
done
echo "AGENTDOJO DONE"
