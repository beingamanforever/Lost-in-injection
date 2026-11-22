#!/usr/bin/env bash
# Prepare the AgentDojo runs (scripts/run_agentdojo.sh) on an A10G without using the GPU: AgentDojo at
# Meta-SecAlign's pinned commit with Meta's patch, its own environment, the Llama 3.1 8B model folder
# (Meta-SecAlign's chat template, with the `input` role) and the adapters. Steps already done are skipped.
# Usage: HF_ENV=<file exporting HF_TOKEN> bash scripts/setup_agentdojo.sh, from the repository root.
set -euo pipefail
source env.sh
GEN=.venv-gen/bin/python
META=third_party/Meta_SecAlign
AGENTDOJO_COMMIT=d3640b5b03a88eb44dad96852e1de1ef437d836e # Meta_SecAlign's submodule pin
set +u; set -a; source "$HF_ENV"; set +a; set -u          # Never printed.

if [ ! -d $META/agentdojo/src ]; then
  git clone -q https://github.com/ethz-spylab/agentdojo.git $META/agentdojo
  git -C $META/agentdojo checkout -q $AGENTDOJO_COMMIT
  git -C $META/agentdojo apply ../helpers/agentdojo.patch # as Meta_SecAlign setup.py does
fi
# AgentDojo's client gets its own environment, pinned to Meta-SecAlign's versions; vLLM serves
# from .venv-gen. The patch imports google-genai, which AgentDojo's own dependencies lack.
if [ ! -x .venv-agentdojo/bin/python ]; then
  uv venv .venv-agentdojo --python 3.12
  VIRTUAL_ENV=.venv-agentdojo uv pip install -e $META/agentdojo google-genai \
    -c $META/requirements.txt
fi

[ -f models/llama-3.1-8b-instruct/tokenizer_config.json ] ||
  $GEN -m lost_in_injection.model_dir --model llama8b
adapter=runs/llama8b-meta-secalign/0/adapter
[ -e $adapter/adapter_model.safetensors ] ||
  $GEN -m lost_in_injection.model_dir --model llama8b --meta-secalign-adapter $adapter
for run in llama8b-english-only llama8b-english-plus-low-resource; do
  [ -f runs/$run/0/adapter/adapter_model.safetensors ] || $GEN -c "
from huggingface_hub import snapshot_download, whoami
from lost_in_injection.publish import adapter_repo
# Only the adapter: vLLM would also load a tokenizer, and so a chat template, from this folder.
snapshot_download(f\"{whoami()['name']}/{adapter_repo('$run')}\", local_dir='runs/$run/0/adapter',
                  allow_patterns=['adapter_config.json', 'adapter_model.safetensors'])"
done
echo "AGENTDOJO SETUP DONE"
