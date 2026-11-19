#!/usr/bin/env bash
# Build the evaluation sets for the given machine-translated language variants ("$@"). A text
# that fails every machine-translation candidate is retried with the Aya fallback; any other
# error stops the script.
set -euo pipefail
source env.sh
GEN=.venv-gen/bin/python

for v in "$@"; do
  if ! $GEN -m lost_in_injection.eval_sets --variants $v > "$TMPDIR/eval-sets-$v.log" 2>&1; then
    grep "texts failed the gates" "$TMPDIR/eval-sets-$v.log"  # Any other error stops the script.
    $GEN -m lost_in_injection.eval_sets --variants $v --aya-fallback
  fi
done
