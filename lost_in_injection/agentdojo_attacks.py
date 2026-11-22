"""AgentDojo's important_instructions attack in other languages (AgentDojo environment).

Loaded by AgentDojo's benchmark with `-ml lost_in_injection.agentdojo_attacks`, this registers one
attack per language in data/agentdojo/attack_texts.json (agentdojo_languages.py), named e.g.
`important_instructions_hindi`. It is the original attack, user and model names and injection
points included, with the translated template and the injection task's translated goal. English
runs the original `important_instructions`.

AgentDojo writes tool results with YAML's and JSON's ASCII-only defaults, which turn non-Latin text
into \\u escapes: a Hindi injection would reach the model as escape codes. Loading this module makes
both keep Unicode, in every run (English and no-attack too). AgentDojo's own data holds only three
non-ASCII strings (a recipe, a restaurant description and "Breizh Café"), so English barely changes.

AgentDojo's prompt for local models tells the model to stop after a function call, and its parser
reads only the first call, but Llama 3.1 8B keeps writing after `</function>` (planning and
re-issuing calls) until the reply cap, about 20,000 characters a turn in the smoke test, which
fills the context within a few turns. So every request also stops at, and keeps, the first
`</function>`: the parsed call is unchanged, and the hallucinated rest never enters the history.

`python -m lost_in_injection.agentdojo_attacks english` writes the English texts to translate: the
template, every injection task's goal, and the literals each goal must keep verbatim. Those are the
task's own string constants that appear in its goal (the values its security check compares, such
as an email body or a hotel name), plus names the agent has to pass to a tool as written,
amounts with thousands separators, and the Subject: and Body: labels of an email spelled out.
`... check` loads every language's attack text into every injection point of every suite (YAML
double-quoted strings) and fails unless the text comes out unchanged.
"""

import json
import re
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import yaml
from agentdojo.agent_pipeline import agent_pipeline
from agentdojo.agent_pipeline.llms import local_llm
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.attacks.attack_registry import register_attack
from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack
from agentdojo.task_suite.load_suites import get_suites

BENCHMARK_VERSION = "v1.2.1"  # Meta-SecAlign's (AgentDojo's default)
DATA = Path(__file__).resolve().parents[1] / "data" / "agentdojo"
# A Slack user, a channel, a city searched by name, and two product names.
NAMES = ("Alice", "general", "Paris", "Facebook", "Slack")
# Amounts such as $30,000, which masking.py would split at the comma, and the field labels of an
# email the goal spells out (one-word lines that language ID cannot gate).
AMOUNT = re.compile(r"\$?\d{1,3}(?:,\d{3})+")
EMAIL_LABEL = re.compile(r"^(?:Subject|Body):", re.MULTILINE)

agent_pipeline.tool_result_to_str = partial(
    tool_result_to_str, dump_fn=partial(yaml.safe_dump, allow_unicode=True)
)
local_llm.json = SimpleNamespace(
    dumps=partial(json.dumps, ensure_ascii=False),
    loads=json.loads,
    JSONDecodeError=json.JSONDecodeError,
)
_local_llm_init = local_llm.LocalLLM.__init__


def _init_stopping_after_a_call(self, client, *args, **kwargs) -> None:
    _local_llm_init(self, client, *args, **kwargs)
    completions = client.chat.completions
    completions.create = partial(
        completions.create,
        stop=["</function>"],
        extra_body={"include_stop_str_in_output": True},  # vLLM's option
    )


local_llm.LocalLLM.__init__ = _init_stopping_after_a_call


def protected_literals(task) -> list[str]:
    constants = {
        value
        for cls in type(task).__mro__
        for key, value in vars(cls).items()
        if key.startswith("_") and not key.startswith("__") and isinstance(value, str)
    }
    words = set(task.GOAL.replace(".", " ").replace(",", " ").split())
    literals = {c for c in constants if c in task.GOAL} | {n for n in NAMES if n in words}
    literals |= set(AMOUNT.findall(task.GOAL)) | set(EMAIL_LABEL.findall(task.GOAL))
    return sorted(literals, key=len, reverse=True)  # Longest first: a body can hold a link.


def english_texts() -> dict:
    return {
        "template": ImportantInstructionsAttack._JB_STRING,
        "goals": {
            suite_name: {task_id: task.GOAL for task_id, task in suite.injection_tasks.items()}
            for suite_name, suite in get_suites(BENCHMARK_VERSION).items()
        },
        "protected": {
            suite_name: {
                task_id: protected_literals(task) for task_id, task in suite.injection_tasks.items()
            }
            for suite_name, suite in get_suites(BENCHMARK_VERSION).items()
        },
    }


class TranslatedImportantInstructions(ImportantInstructionsAttack):
    texts: dict  # one language's {"template": ..., "goals": {suite: {injection task: goal}}}

    def __init__(self, task_suite, target_pipeline) -> None:
        super().__init__(task_suite, target_pipeline)
        self.jailbreak = self.texts["template"]

    def attack(self, user_task, injection_task) -> dict[str, str]:
        goal = self.texts["goals"][self.task_suite.name][injection_task.ID]
        return {
            injection: self.jailbreak.format(goal=goal, user=self.user_name, model=self.model_name)
            for injection in self.get_injection_candidates(user_task)
        }


if (DATA / "attack_texts.json").exists():
    translated = json.loads((DATA / "attack_texts.json").read_text())["languages"]
    for language, texts in translated.items():
        name = f"important_instructions_{language}"
        attack = type(name, (TranslatedImportantInstructions,), {"name": name, "texts": texts})
        register_attack(attack)


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict | list):
        for item in value.values() if isinstance(value, dict) else value:
            yield from strings(item)


def check_injections() -> None:
    texts = json.loads((DATA / "attack_texts.json").read_text())
    for language, language_texts in {"english": texts["english"], **texts["languages"]}.items():
        for suite_name, suite in get_suites(BENCHMARK_VERSION).items():
            vectors = suite.get_injection_vector_defaults()
            for task_id, goal in language_texts["goals"][suite_name].items():
                text = language_texts["template"].format(goal=goal, user="U", model="M")
                env = suite.load_and_inject_default_environment(dict.fromkeys(vectors, text))
                found = " ".join(" ".join(strings(env.model_dump(mode="json"))).split())
                if found.count(" ".join(text.split())) < len(vectors):
                    raise ValueError(f"{language} {suite_name} {task_id}: text changed on loading")
        print(f"{language}: every attack text loads unchanged")


if __name__ == "__main__":
    if sys.argv[1:] == ["english"]:
        DATA.mkdir(parents=True, exist_ok=True)
        english = json.dumps(english_texts(), ensure_ascii=False, indent=2)
        (DATA / "english.json").write_text(english)
    elif sys.argv[1:] == ["check"]:
        check_injections()
    else:
        raise SystemExit("usage: python -m lost_in_injection.agentdojo_attacks english|check")
