"""Build the local model directory that the Meta-SecAlign runners load.

The runners read the tokenizer, and so the chat template, from the model directory, and take
the base model from the path before its first "_" (Meta_SecAlign utils.py:188-196). This
directory holds the official weights as symlinks, plus a tokenizer whose chat template has an
`input` role for untrusted data. Tiny Aya (Global, Fire, Earth and Water) and Qwen3
drop roles they do not know, so `add_input_role` adds one to their official template, and its
guard fails loudly if a template no longer matches. Llama 3.1 and 3.2 get the template
Meta-SecAlign trains with (`meta_llama_template`), so Meta's own Meta-SecAlign-8B and our Llama
runs see the same prompts. `--meta-secalign-adapter` fetches Meta-SecAlign-8B or -70B (by `--model`)
and checks it ships this template.
"""

import argparse
import ast
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from lost_in_injection import secalign
from lost_in_injection.runs import BASE_MODELS, model_path

WEIGHT_FILES = {"config.json", "generation_config.json", "model.safetensors.index.json"}
REVISIONS = {  # pinned Sep-Nov 2026
    "CohereLabs/tiny-aya-global": "af89d219b53ed9b13b8a4645f9c8028973510324",
    "CohereLabs/tiny-aya-fire": "73500c80fcb6c8bea3e830c21d5d2920249a0e1a",
    "CohereLabs/tiny-aya-earth": "0624fafbe200f930b89566fc4ec4da314ba48d89",
    "CohereLabs/tiny-aya-water": "41a99216d12b3ef4881df762934186e5e4d2bbc1",
    "meta-llama/Llama-3.1-8B-Instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
    "meta-llama/Llama-3.2-3B-Instruct": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
    "meta-llama/Llama-3.3-70B-Instruct": "6f6073b423013f6a7d4d9f39144961bfbfbc386b",
    "Qwen/Qwen3-4B-Instruct-2507": "cdbee75f17c01a7cc42f958dc650907174af0554",
}
META_SECALIGN = {  # base model: (adapter repo, revision), pinned Nov 2026
    "llama8b": ("facebook/Meta-SecAlign-8B", "fb9b039b45ab4fe5e94517efaaf19f80f4fedda1"),
    "llama70b": ("facebook/Meta-SecAlign-70B", "09e225033f7b1ee9a52bf56696a6d563e07ac2b7"),
}
# Per Hub owner: the official template's branch the `input` branch goes right after, the branch
# (CONTENT marks the message text), and the template's name for that text.
INPUT_BRANCHES = {
    # Tiny Aya: a tool-result system turn, after the last role branch.
    "CohereLabs": (
        "{% elif role == 'system' %}<|SYSTEM_TOKEN|>{{ message['content'] }}",
        "{% elif role == 'input' %}<|SYSTEM_TOKEN|><|START_TOOL_RESULT|>{{ CONTENT }}"
        "<|END_TOOL_RESULT|>",
        "message['content']",
    ),
    # Qwen3: its own `input` turn, the way Meta-SecAlign's Llama template renders it.
    "Qwen": (
        r"{{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}",
        r"{%- elif message.role == 'input' %}{{- '<|im_start|>input\n' + (CONTENT)"
        r" + '<|im_end|>\n' }}",
        "content",
    ),
}


def add_input_role(template: str, added_tokens: list[str], owner: str) -> str:
    """Render role `input` as the model family's untrusted-data turn.

    Added-token strings are removed from the content so untrusted text cannot close its turn
    early with a fake delimiter.
    """
    anchor, branch, text = INPUT_BRANCHES[owner]
    if template.count(anchor) != 1:
        raise ValueError(f"The official {owner} template no longer has exactly one anchor branch.")
    content = text + "".join(f" | replace({t!r}, '')" for t in added_tokens)
    return template.replace(anchor, anchor + branch.replace("CONTENT", content))


def meta_llama_template() -> str:
    """Meta-SecAlign's Llama 3 template (its setup.py): no built-in date system prompt, and every
    role, `input` included, rendered as its own header.

    The literal in setup.py ends with the indentation before its closing quotes, which renders as
    "\n    " after every generation prompt. The released Meta-SecAlign tokenizers, which Meta's
    runners load to test those models, render without it, so it is stripped here."""
    for node in ast.walk(ast.parse((secalign.ROOT / "setup.py").read_text())):
        if (
            isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == "tokenizer.chat_template"
            and isinstance(node.value, ast.Constant)
        ):
            return node.value.value.rstrip()
    raise ValueError("Meta_SecAlign setup.py no longer sets a literal chat template.")


def official_template(tokenizer) -> str:
    template = tokenizer.chat_template
    return template["default"] if isinstance(template, dict) else template


def chat_template(model_id: str, tokenizer) -> str:
    if model_id.startswith("meta-llama/"):
        return meta_llama_template()
    added_tokens = [token.content for token in tokenizer.added_tokens_decoder.values()]
    return add_input_role(official_template(tokenizer), added_tokens, model_id.split("/")[0])


def build(model_id: str, out_dir: Path) -> None:
    # Llama's repo also holds its original 16 GB checkpoint, which nothing here reads.
    revision = REVISIONS[model_id]
    snapshot = Path(snapshot_download(model_id, revision=revision, ignore_patterns=["original/*"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    for source in snapshot.iterdir():
        if source.name in WEIGHT_FILES or source.suffix == ".safetensors":
            target = out_dir / source.name
            target.unlink(missing_ok=True)
            target.symlink_to(source)

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    tokenizer.chat_template = chat_template(model_id, tokenizer)
    tokenizer.save_pretrained(out_dir)


# Conversations with every role the runners and Meta's template use, for comparing templates.
SAMPLE_CHATS = [
    [{"role": "user", "content": "Summarise this."}, {"role": "input", "content": "Data. Hacked!"}],
    [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Again"},
        {"role": "input", "content": " padded "},
    ],
]


def same_prompts(tokenizer, template: str) -> bool:
    """Whether `template` renders every sample chat exactly like the tokenizer's own template."""

    def render(chat_template: str, chat: list[dict], generation: bool) -> str:
        return tokenizer.apply_chat_template(
            chat, chat_template=chat_template, tokenize=False, add_generation_prompt=generation
        )

    own = official_template(tokenizer)
    return all(
        render(own, chat, g) == render(template, chat, g)
        for chat in SAMPLE_CHATS
        for g in (True, False)
    )


def fetch_meta_secalign(model: str, adapter_dir: Path) -> None:
    """Link a Meta-SecAlign adapter's files (gated) after checking that its tokenizer renders the
    same prompts as our Llama template (the released templates differ only in indentation)."""
    repo, revision = META_SECALIGN[model]
    snapshot = Path(snapshot_download(repo, revision=revision))
    if not same_prompts(AutoTokenizer.from_pretrained(snapshot), meta_llama_template()):
        raise ValueError(f"{repo}'s chat template renders differently from ours.")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        target = adapter_dir / name
        target.unlink(missing_ok=True)
        target.symlink_to(snapshot / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=BASE_MODELS, default="global")
    parser.add_argument(
        "--meta-secalign-adapter",
        type=Path,
        help="Fetch --model's Meta-SecAlign adapter into this folder instead.",
    )
    args = parser.parse_args()
    if args.meta_secalign_adapter:
        fetch_meta_secalign(args.model, args.meta_secalign_adapter)
        return
    out = model_path(args.model)
    if "_" in str(out.absolute()):  # the runners see this string, not the physical path
        raise ValueError(
            "The Meta-SecAlign runners split model paths on '_'; use a path without one."
        )
    build(BASE_MODELS[args.model], out)


if __name__ == "__main__":
    main()
