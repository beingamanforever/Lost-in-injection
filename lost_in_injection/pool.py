"""Curate the clean rows and the injection pool with CuratorKIT (train environment).

The sources are Meta-SecAlign's (utils.py:61-66): clean rows are `yahma/alpaca-cleaned` rows with
a non-empty input, and injected instructions are Stanford Alpaca rows. Both pass CuratorKIT's
MinHash dedup, its text cleaner with HTML stripping off, and its local secrets and toxicity gates.
CuratorKIT's word-count gate only drops empty rows here; pairs are length-checked in tokens after
labelling. Each output folder keeps CuratorKIT's manifest, rejected rows and dataset card, plus
`kept.jsonl` with the surviving rows and their ids.
"""

import argparse
import json
from pathlib import Path

from curatorkit import Curator, CuratorConfig
from datasets import load_dataset

from lost_in_injection.records import write_jsonl

ALPACA_CLEANED = ("yahma/alpaca-cleaned", "12567cabf869d7c92e573c7c783905fc160e9639")


def curate(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    source = out_dir / "source.jsonl"
    write_jsonl(source, rows)
    config = CuratorConfig(
        dataset=str(source),
        format="alpaca",
        dedup="minhash",
        clean=True,
        clean_transforms={"strip_html": False},  # It would also delete text between < and >.
        min_tokens=1,  # The default of 10 words would drop short Alpaca answers.
        max_tokens=2048,
        secrets_gate=True,
        toxicity_gate=True,
        # With no LLM judge, CuratorKIT rejects every score at or above the pass threshold (the
        # reject threshold only has to sit above it), so this drops rows scoring 0.5 or more.
        toxicity_classifier_pass_threshold=0.5,
        toxicity_classifier_reject_threshold=0.9,
        export_formats=["alpaca"],
        output_dir=str(out_dir),
    )
    result = Curator(config).run()
    kept = [
        {"id": s.id, "instruction": s.instruction, "input": s.input, "output": s.output}
        for s in result.passed
    ]
    write_jsonl(out_dir / "kept.jsonl", kept)
    print(f"{out_dir.name}: kept {len(kept)} of {len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/pool"))
    args = parser.parse_args()

    clean = load_dataset(ALPACA_CLEANED[0], revision=ALPACA_CLEANED[1], split="train")
    clean_rows = [
        {"id": f"clean-{i}", **{key: row[key] for key in ("instruction", "input", "output")}}
        for i, row in enumerate(clean)
        if row["input"]
    ]
    injections = json.loads((args.raw / "alpaca_data.json").read_text())
    injection_rows = [{"id": f"injection-{i}", **row} for i, row in enumerate(injections)]
    curate(clean_rows, args.out / "clean")
    curate(injection_rows, args.out / "injections")


if __name__ == "__main__":
    main()
