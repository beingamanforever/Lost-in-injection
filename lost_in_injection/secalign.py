"""Import Meta-SecAlign's modules, which live in a script folder rather than a package."""

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "third_party" / "Meta_SecAlign"
# Meta-SecAlign's non-adaptive AlpacaFarm-Hacked attacks (test.py:254).
ALPACA_ATTACKS = [
    "straightforward",
    "straightforward_before",
    "ignore",
    "ignore_before",
    "completion",
    "completion_ignore",
]
INJECAGENT_DEFENSE = "sandwich"  # Meta-SecAlign's InjecAgent protocol (run_tests.py:23).


def load(module_name: str):
    """Import a Meta-SecAlign module such as `config`, `utils`, `test` or `test_injecagent`."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module(module_name)
