#!/usr/bin/env bash
# Fetch the sources that Meta-SecAlign's setup.py downloads (setup.py:29-44) and that this
# project uses, pinned to fixed revisions. Run from the repository root.
set -euo pipefail
INJECAGENT=https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/f19c9f2c79a41046eb13c03c51a24c567a8ffa07/data
ALPACA=https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/761dc5bfbdeeffa89b8bff5d038781a4055f796a
ALPACA_FARM=https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token/resolve/95d6ffdab97aaf797acfe61661cdc8f9bc8757db

mkdir -p data/raw
cd data/raw
for name in tools.json attacker_simulated_responses.json test_cases_{dh,ds}_{base,enhanced}.json; do
  curl -fsSLO "$INJECAGENT/$name"
done
curl -fsSLO "$ALPACA/alpaca_data.json"
curl -fsSLO "$ALPACA_FARM/davinci_003_outputs.json"
ls -l
