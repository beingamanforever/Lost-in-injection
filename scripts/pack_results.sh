#!/usr/bin/env bash
# Pack only result files of the named runs (no model symlinks) into data/exports/<name>.tgz on the
# Modal volume, for `modal volume get`. Usage: scripts/pack_results.sh <name> <run>...
set -euo pipefail
name=$1; shift
mkdir -p data/exports
cd runs
find "$@" -type f \( -name '*_outputs.json' -o -name 'summary.tsv' -o -name 'InjecAgent*.json' \
  -o -path '*/utility/*' -o -path '*/generation/*' -o -name 'judge.json' -o -path '*/logs/*.json' \) \
  -print0 | tar czf "../data/exports/$name.tgz.part" --null -T -
# The volume can show a file while it is being written: publish the archive only when complete.
mv "../data/exports/$name.tgz.part" "../data/exports/$name.tgz"
ls -la "../data/exports/$name.tgz"
