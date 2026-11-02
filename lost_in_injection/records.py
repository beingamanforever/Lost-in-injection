"""JSONL files and translation records, with no third-party imports so every environment can use
them."""

import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def read_records(path: Path) -> dict[str, dict]:
    """Translation records (translate.gate) keyed by their English source."""
    return {record["source"]: record for record in read_jsonl(path)}


def chosen_candidate(record: dict) -> dict:
    return next(c for c in record["candidates"] if c["restored"] == record["translation"])


def masked_translation(record: dict) -> str:
    return chosen_candidate(record)["masked"]
