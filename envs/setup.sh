#!/usr/bin/env bash
# Build both environments on the A10G. Run from the repository root.
set -euo pipefail
source env.sh
uv venv --clear .venv-gen --python 3.12
VIRTUAL_ENV=.venv-gen uv pip install -r envs/gen.txt
uv venv --clear .venv-train --python 3.12
VIRTUAL_ENV=.venv-train uv pip install -r envs/train.txt
