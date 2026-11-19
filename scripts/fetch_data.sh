#!/usr/bin/env bash
# Fetch the sources that Meta-SecAlign's setup.py downloads (setup.py:29-44) and that this
# project uses, pinned to fixed revisions. Run from the repository root.
set -euo pipefail
INJECAGENT=https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/f19c9f2c79a41046eb13c03c51a24c567a8ffa07/data
ALPACA=https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/761dc5bfbdeeffa89b8bff5d038781a4055f796a
ALPACA_FARM=https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token/resolve/95d6ffdab97aaf797acfe61661cdc8f9bc8757db
# AI4Bharat/IndicIFEval: its Hindi translation of IFEval (the -trans config, not the native -ground one).
INDICIFEVAL=https://raw.githubusercontent.com/AI4Bharat/IndicIFEval/1bb5f1bc4936cb20b8544e185bfd7cbed8f31464/lm-evaluation-harness/custom_configs/indicifeval-trans
# Marco-Bench-MIF data and its bundled lm-eval-style checker (ATH-MaaS/Marco-LLM, Apache-2.0).
MARCO=https://raw.githubusercontent.com/ATH-MaaS/Marco-LLM/697c8d43d204a290bc0c4c6c806c0fcef2f5e9bd/Marco-Bench-MIF/data
MARCO_CHECKER="$MARCO/multi_instruction_following_eval"

mkdir -p data/raw
cd data/raw
for name in tools.json attacker_simulated_responses.json test_cases_{dh,ds}_{base,enhanced}.json; do
  curl -fsSLO "$INJECAGENT/$name"
done
curl -fsSLO "$ALPACA/alpaca_data.json"
curl -fsSLO "$ALPACA_FARM/davinci_003_outputs.json"

mkdir -p indicifeval/instructions
cd indicifeval
curl -fsSLO "$INDICIFEVAL/indicifeval_hi.yaml"
curl -fsSLO "$INDICIFEVAL/utils.py"
curl -fsSLO "$INDICIFEVAL/instructions_registry.py"
curl -fsSL "$INDICIFEVAL/instructions/hi_instructions.py" -o instructions/hi_instructions.py
curl -fsSL "$INDICIFEVAL/instructions/hi_instructions_util.py" -o instructions/hi_instructions_util.py
cd ..

mkdir -p marco-bench-mif/checker
cd marco-bench-mif
for stem in fr sw zh_local ar_local tr ru el ko th bn; do
  curl -fsSLO "$MARCO/$stem.jsonl"
done
# At this revision key 181 in zh_local.jsonl leaves the quotes around "\n\n" unescaped (invalid JSON).
sed -i 's/（"\\n\\n"）/（\\"\\n\\n\\"）/' zh_local.jsonl
cd checker
for name in evaluation_main.py instructions.py instructions_registry.py instructions_util.py; do
  curl -fsSLO "$MARCO_CHECKER/$name"
done
cd ../..

ls -l
