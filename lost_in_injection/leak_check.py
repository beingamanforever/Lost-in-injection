"""Check for overlap between training prompts and test instructions (any environment).

Exact match after normalizing case and punctuation, and near duplicates with word 3-gram Jaccard
of 0.8 or more, between the English-only run's training prompts (the user instruction, and the
data with its injection) and the AlpacaFarm-Hacked and InjecAgent test instructions.
Run from the repository root after scripts/build_data.sh.
"""

import json
import re


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def grams(text: str) -> set[tuple[str, ...]]:
    words = norm(text).split()
    return {tuple(words[i : i + 3]) for i in range(max(1, len(words) - 2))}


def between(prompt: str, start: str, end: str) -> str:
    return re.search(rf"{re.escape(start)}(.*?){re.escape(end)}", prompt, re.S).group(1)


def main() -> None:
    with open("data/runs/english-only/dpo.jsonl") as file:
        prompts = [json.loads(line)["prompt"] for line in file]
    train = {
        "training instructions": [
            between(p, "<|USER_TOKEN|>", "<|END_OF_TURN_TOKEN|>") for p in prompts
        ],
        "training data with injections": [
            between(p, "<|START_TOOL_RESULT|>", "<|END_TOOL_RESULT|>") for p in prompts
        ],
    }
    with open("data/raw/davinci_003_outputs.json") as file:
        farm = [row for row in json.load(file) if row["input"]]
    cases = []
    for setting in ("dh", "ds"):
        with open(f"data/eval/english/test_cases_{setting}_base.json") as file:
            cases += json.load(file)
    test = {
        "AlpacaFarm-Hacked instructions": [row["instruction"] for row in farm],
        "AlpacaFarm-Hacked data": [row["input"] for row in farm],
        "InjecAgent user instructions": sorted({c["User Instruction"] for c in cases}),
        "InjecAgent attacker instructions": sorted({c["Attacker Instruction"] for c in cases}),
    }
    for train_name, texts in train.items():
        exact, near = {norm(t) for t in texts}, [grams(t) for t in texts]
        for test_name, items in test.items():
            matches = sum(norm(item) in exact for item in items)
            duplicates = sum(
                any(len(item & g) / len(item | g) >= 0.8 for g in near)
                for item in map(grams, items)
            )
            print(
                f"{train_name} vs {test_name} ({len(items)}): exact {matches}, "
                f"near-duplicate {duplicates}"
            )


if __name__ == "__main__":
    main()
