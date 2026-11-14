"""Write video/cases.json: for each model, which of the 208 AlpacaFarm-Hacked cases each variant
hijacks (markdown-tolerant rule). Run from the repository root where runs/*/0/eval exists."""

import json
import sys
from pathlib import Path

from lost_in_injection.metrics import EVALUATED, MARKDOWN_TOLERANT, alpacafarm

runs = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs")
cases = {}
for model in ("base", "english-only", "english-plus-high-resource", "english-plus-low-resource"):
    if (runs / model / "0" / "eval" / "english").exists():
        cases[model] = {
            v: "".join("1" if hit else "0"
                       for hit in alpacafarm(runs / model / "0" / "eval" / v, "none")[0][MARKDOWN_TOLERANT].any(axis=0))
            for v in EVALUATED
        }
Path("video/cases.json").write_text(json.dumps(cases, indent=1))
print({m: sum(any(c[v][i] == "1" for v in c) for i in range(208)) for m, c in cases.items()})
