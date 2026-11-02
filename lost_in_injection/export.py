"""Export one DPO dataset per run with CuratorKIT, plus a per-pair metadata sidecar (train env).

Each prompt is the model's chat template with the clean instruction as the user turn and the data
with the injection, in the run's language, as the `input` turn. `no-input-role` appends the data
to the user turn instead, as Meta-SecAlign does without its instruction hierarchy (utils.py:254).
Chosen and rejected are the self-generated replies, closed the way the template closes an
assistant turn. The leading BOS is removed because TRL adds its own. Pairs whose label hit the
generation cap, or that exceed MAX_TOKENS in any run, are dropped from every run, so all runs
train on the same pairs. CuratorKIT's export is checked to match the rows exactly.
"""

import argparse
import json
from pathlib import Path

from curatorkit import Curator, CuratorConfig
from transformers import AutoTokenizer

from lost_in_injection.languages import BY_NAME
from lost_in_injection.runs import RUNS, language

MAX_TOKENS = 2048  # Pairs longer than this are dropped, not truncated.


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def end_of_reply(tokenizer) -> str:
    marker = "REPLY"
    turn = [{"role": "user", "content": "x"}, {"role": "assistant", "content": marker}]
    return tokenizer.apply_chat_template(turn, tokenize=False).split(marker)[1]


def prompt(tokenizer, pair: dict, run: str) -> str:
    instruction, data = pair["clean"]["instruction"], pair["data"][language(run, pair["slot"])]
    if run == "no-input-role":
        messages = [{"role": "user", "content": f"{instruction}\n\n{data}"}]
    else:
        messages = [{"role": "user", "content": instruction}, {"role": "input", "content": data}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text.removeprefix(tokenizer.bos_token)


def row(tokenizer, pair: dict, run: str, end: str) -> dict:
    rejected = pair["rejected"][language(run, pair["slot"])]["text"]
    return {
        "prompt": prompt(tokenizer, pair, run),
        "chosen": pair["chosen"]["text"] + end,
        "rejected": rejected + end,
    }


def num_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"]) + 1  # TRL's BOS


def usable(tokenizer, pair: dict, end: str) -> bool:
    labels = [pair["chosen"], *pair["rejected"].values()]
    if any(label["finish_reason"] == "length" for label in labels):
        return False
    for run in RUNS:
        r = row(tokenizer, pair, run, end)
        if (
            max(num_tokens(tokenizer, r["prompt"] + r[k]) for k in ("chosen", "rejected"))
            > MAX_TOKENS
        ):
            return False
    return True


def export(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "source.jsonl", rows)
    config = CuratorConfig(
        dataset=str(out_dir / "source.jsonl"),
        format="preference",
        dedup="none",
        clean=False,  # Its HTML stripping would delete the template's special tokens.
        schema_gate=False,
        export_formats=["dpo"],
        output_dir=str(out_dir),
    )
    Curator(config).run()
    if read_jsonl(out_dir / "dpo.jsonl") != rows:
        raise ValueError(f"CuratorKIT changed rows in {out_dir / 'dpo.jsonl'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=Path("models/tiny-aya-global"))
    parser.add_argument("--pairs", type=Path, default=Path("data/pairs/labelled.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/runs"))
    parser.add_argument("--runs", nargs="+", default=list(RUNS))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    end = end_of_reply(tokenizer)
    pairs = read_jsonl(args.pairs)
    kept = [pair for pair in pairs if usable(tokenizer, pair, end)]
    print(f"kept {len(kept)} of {len(pairs)} pairs")
    for run in args.runs:
        export([row(tokenizer, pair, run, end) for pair in kept], args.out / run)
        meta = []
        for pair in kept:
            lang = language(run, pair["slot"])
            meta.append(
                {
                    "pair_id": pair["pair_id"],
                    "run": run,
                    "slot": pair["slot"],
                    "language": lang,
                    "translator": BY_NAME[lang].translator,
                    "attack": pair["attack"],
                    "position": pair["position"],
                    "clean_id": pair["clean"]["id"],
                    "injection_id": pair["injection"]["id"],
                }
            )
        write_jsonl(args.out / run / "pairs_meta.jsonl", meta)
        print(f"{run}: {len(kept)} pairs")


if __name__ == "__main__":
    main()
