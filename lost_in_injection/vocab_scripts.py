"""Vocabulary tokens per writing system for each model family's tokenizer (Ram's tokenizer check).

Every tokenizer here is byte-level, so any character can be written from byte pieces; this counts
the vocabulary entries that contain letters of each script, i.e. how directly a model can read and
write each language. Writes results/acl/tokenizer_scripts.csv. Needs Hugging Face access to the
gated Llama and Tiny Aya repos. Llama 3.3 70B and Meta-SecAlign share Llama 3.1's tokenizer.
Usage: python -m lost_in_injection.vocab_scripts
"""

import csv
import unicodedata
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

SCRIPTS = [
    ("Latin", "LATIN"),
    ("Devanagari (Hindi)", "DEVANAGARI"),
    ("Arabic (Urdu)", "ARABIC"),
    ("Tamil", "TAMIL"),
    ("Ethiopic (Amharic)", "ETHIOPIC"),
    ("Bengali", "BENGALI"),
    ("Thai", "THAI"),
    ("Han (Chinese)", "CJK"),
    ("Hangul (Korean)", "HANGUL"),
    ("Greek", "GREEK"),
    ("Cyrillic (Russian)", "CYRILLIC"),
]


def script(ch):
    name = unicodedata.name(ch, "")
    return next((s for s, k in SCRIPTS if name.startswith(k)), None)


results = {}
for label, repo in [
    ("Tiny Aya", "CohereLabs/tiny-aya-global"),
    ("Qwen3", "Qwen/Qwen3-4B-Instruct-2507"),
    ("Llama 3.x", "meta-llama/Llama-3.1-8B-Instruct"),
]:
    tok = AutoTokenizer.from_pretrained(repo)
    counts = Counter()
    for tid in range(len(tok)):
        for sc in {script(c) for c in tok.decode([tid]) if c.isalpha()} - {None}:
            counts[sc] += 1
    results[label] = (len(tok), counts)
with open("results/acl/tokenizer_scripts.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["script", *results])
    writer.writerow(["vocabulary size", *(v[0] for v in results.values())])
    for s, _ in SCRIPTS:
        writer.writerow([s, *(v[1][s] for v in results.values())])
print(Path("results/acl/tokenizer_scripts.csv").read_text())
