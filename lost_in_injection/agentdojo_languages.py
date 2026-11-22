"""Translate AgentDojo's important_instructions attack text, and summarize AgentDojo's results.

`translate` (gen environment, GPU) machine-translates the English texts agentdojo_attacks.py
writes, behind translate.py's gates, into data/agentdojo/attack_texts.json; every candidate and gate
value goes to data/agentdojo/translations/<language>.jsonl. Only the attacker's text changes
language. The template keeps its layout and <INFORMATION> tags, each line of the template and of a
goal is one text to translate, and each protected literal (an email body, a hotel name, a URL...)
is swapped for a {protectedN} token, which masking.py keeps, so it comes back verbatim exactly
once. As in eval_sets.py, every text must pass, and texts that fail every machine translation can
be retried with `--aya-fallback` in a fresh process.

`results` (standard library only) reads AgentDojo's logs, runs/<run>/0/agentdojo/<language>/ and
the workspace-<part>/ folders inside (scripts/run_agentdojo.sh), into
one row per (run, language, suite) and overall: utility without attack (`no-attack`), utility under
attack and targeted attack success rate (AgentDojo's `security`: the injection task succeeded).
An `any(...)` row counts a (user task, injection task) pair as broken if any language breaks it.
"""

import argparse
import csv
import json
import re
from pathlib import Path

from lost_in_injection.languages import BY_NAME
from lost_in_injection.records import write_jsonl

DATA = Path("data/agentdojo")
TOKEN = re.compile(r"\{protected(\d+)\}")
TEMPLATE_FIELDS = {"goal": 1, "user": 2, "model": 1}  # placeholders in important_instructions
PIPELINE = "local-repeat_user_prompt"  # AgentDojo's log folder for --model local, this defense
NO_ATTACK = "no-attack"


def protect(text: str, literals: list[str]) -> tuple[str, list[str]]:
    """Swap each literal (longest first), with the quotes around it, for a {protectedN} token."""
    if not literals:
        return text, []
    pattern = "|".join(
        rf"\b{re.escape(literal)}\b" if literal.isalpha() else rf"'?{re.escape(literal)}'?"
        for literal in literals
    )
    kept: list[str] = []

    def swap(match: re.Match) -> str:
        kept.append(match.group(0))
        return f"{{protected{len(kept)}}}"

    return re.sub(pattern, swap, text), kept


def restore(text: str, kept: list[str]) -> str:
    return TOKEN.sub(lambda match: kept[int(match.group(1)) - 1], text)


def translatable(line: str) -> bool:
    """Whether a line has words besides tags such as <INFORMATION> and placeholders."""
    return any(c.isalpha() for c in re.sub(r"<[^>]*>|\{\w+\}", "", line))


def assemble(text: str, translated: dict[str, str]) -> str:
    return "\n".join(translated[line] if translatable(line) else line for line in text.split("\n"))


def check(english: dict, texts: dict) -> None:
    """The template must format with the original placeholders and every goal must keep its
    protected literals."""
    template = texts["template"]
    for field, count in TEMPLATE_FIELDS.items():
        if template.count(f"{{{field}}}") != count:
            raise ValueError(f"template has {template.count(f'{{{field}}}')} {{{field}}}")
    template.format(goal="", user="", model="")  # no stray braces
    for suite, goals in texts["goals"].items():
        for task, goal in goals.items():
            for literal in english["protected"][suite][task]:
                if goal.count(literal) < english["goals"][suite][task].count(literal):
                    raise ValueError(f"{suite} {task} lost {literal!r}")


def translate_language(english: dict, language: str, aya_fallback: bool) -> dict:
    from lost_in_injection.translate import translate_or_rewrite  # Needs the GPU models.

    variant = BY_NAME[language]
    protected = english["protected"]
    goals = {
        suite: {task: protect(goal, protected[suite][task]) for task, goal in tasks.items()}
        for suite, tasks in english["goals"].items()
    }
    texts = [english["template"]] + [text for tasks in goals.values() for text, _ in tasks.values()]
    lines = [line for text in texts for line in text.split("\n") if translatable(line)]
    sources = list(dict.fromkeys(lines))  # In order, no repeats

    path = DATA / "translations" / f"{language}.jsonl"
    if aya_fallback:
        from lost_in_injection.eval_sets import with_aya_fallback

        records = [json.loads(line) for line in path.read_text().splitlines()]
        if [record["source"] for record in records] != sources:
            raise ValueError(f"{path} was written for different sources")
        records = with_aya_fallback(records, DATA, variant)
    else:
        records = translate_or_rewrite(sources, DATA, variant)
    write_jsonl(path, records)
    failed = [record["source"] for record in records if record["translation"] is None]
    if failed:
        raise SystemExit(f"{language}: {len(failed)} texts failed the gates: {failed}")

    translated = {record["source"]: record["translation"] for record in records}
    result = {
        "template": assemble(english["template"], translated),
        "goals": {
            suite: {
                task: restore(assemble(text, translated), kept)
                for task, (text, kept) in tasks.items()
            }
            for suite, tasks in goals.items()
        },
    }
    check(english, result)
    return result


def translate_all(languages: list[str], aya_fallback: bool) -> None:
    english = json.loads((DATA / "english.json").read_text())
    out = DATA / "attack_texts.json"
    texts = json.loads(out.read_text()) if out.exists() else {"english": english, "languages": {}}
    if texts["english"] != english:
        raise ValueError(f"{out} was translated from different English texts")
    for language in languages:
        texts["languages"][language] = translate_language(english, language, aya_fallback)
        out.write_text(json.dumps(texts, ensure_ascii=False, indent=2))  # after each language


def read_cases(language_dir: Path, pattern: str) -> dict[tuple[str, str, str], dict]:
    """AgentDojo's logged (suite, user task, injection task) results under one log folder. A case's
    log is rewritten as it runs and gets `utility` and `security` when it ends."""
    cases = {}
    for path in language_dir.glob(f"**/{PIPELINE}/*/user_task_*/{pattern}"):  # workspace-1/ too
        suite, user_task = path.parts[-4], path.parts[-3]
        case = json.loads(path.read_text())
        if "utility" in case:
            cases[(suite, user_task, path.stem)] = case
    return cases


def mean(values: list[bool]) -> float | str:
    return round(sum(values) / len(values), 4) if values else ""


def summary_rows(run_dir: Path) -> list[dict]:
    run = run_dir.parents[1].name
    clean = read_cases(run_dir / NO_ATTACK, "none/none.json")
    attacked = {
        language_dir.name: read_cases(language_dir, "important_instructions*/injection_task_*.json")
        for language_dir in sorted(run_dir.iterdir())
        if language_dir.is_dir() and language_dir.name != NO_ATTACK
    }
    rows = []
    for language, cases in attacked.items():
        for suite in [*sorted({s for s, _, _ in cases}), "overall"]:
            picked = [c for (s, _, _), c in cases.items() if suite in (s, "overall")]
            no_attack = [c["utility"] for (s, _, _), c in clean.items() if suite in (s, "overall")]
            rows.append(
                {
                    "run": run,
                    "language": language,
                    "suite": suite,
                    "cases": len(picked),
                    "utility_no_attack": mean(no_attack),
                    "utility_under_attack": mean([c["utility"] for c in picked]),
                    "asr": mean([c["security"] for c in picked]),
                }
            )
    if len(attacked) > 1:  # Pairs that every language has run.
        pairs = set.intersection(*(set(cases) for cases in attacked.values()))
        broken = {pair: any(c[pair]["security"] for c in attacked.values()) for pair in pairs}
        for suite in [*sorted({s for s, _, _ in pairs}), "overall"]:
            picked = [b for (s, _, _), b in broken.items() if suite in (s, "overall")]
            rows.append(
                {
                    "run": run,
                    "language": f"any({'+'.join(attacked)})",
                    "suite": suite,
                    "cases": len(picked),
                    "utility_no_attack": "",
                    "utility_under_attack": "",
                    "asr": mean(picked),
                }
            )
    return rows


def write_results(runs: Path, out: Path) -> None:
    rows = [row for run_dir in sorted(runs.glob("*/0/agentdojo")) for row in summary_rows(run_dir)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    translate = commands.add_parser("translate")
    translate.add_argument("--languages", nargs="+", required=True)
    translate.add_argument("--aya-fallback", action="store_true")
    results = commands.add_parser("results")
    results.add_argument("runs", type=Path, help="Folder of <run>/0/agentdojo/<language>/ logs.")
    results.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "translate":
        translate_all(args.languages, args.aya_fallback)
    else:
        write_results(args.runs, args.out)


if __name__ == "__main__":
    main()
