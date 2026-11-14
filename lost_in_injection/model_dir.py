"""Build the local model directory that the Meta-SecAlign runners load.

The runners read the tokenizer, and so the chat template, from the model directory, and take
the base model from the path before its first "_" (Meta_SecAlign utils.py:188-196). This
directory holds the official Tiny Aya weights as symlinks, plus a tokenizer whose chat
template gains one `input` role for untrusted data.
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

# The official template's last role branch. The `input` branch is inserted right after it.
SYSTEM_BRANCH = "{% elif role == 'system' %}<|SYSTEM_TOKEN|>{{ message['content'] }}"
WEIGHT_FILES = {"config.json", "generation_config.json", "model.safetensors.index.json"}
REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"  # CohereLabs/tiny-aya-global, Sep 30, 2026


def add_input_role(template: str, added_tokens: list[str]) -> str:
    """Render role `input` as a tool-result system turn.

    Tiny Aya's official template drops any role it does not know, so injected text in an
    `input` turn would never reach the model. Added-token strings are removed from the
    content so untrusted text cannot close the tool-result block early.
    """
    if template.count(SYSTEM_BRANCH) != 1:
        raise ValueError("The official template no longer has exactly one system branch.")
    content = "message['content']" + "".join(f" | replace({t!r}, '')" for t in added_tokens)
    input_branch = (
        "{% elif role == 'input' %}<|SYSTEM_TOKEN|><|START_TOOL_RESULT|>{{ "
        f"{content} }}}}<|END_TOOL_RESULT|>"
    )
    return template.replace(SYSTEM_BRANCH, SYSTEM_BRANCH + input_branch)


def official_template(tokenizer) -> str:
    template = tokenizer.chat_template
    return template["default"] if isinstance(template, dict) else template


def build(model_id: str, out_dir: Path) -> None:
    snapshot = Path(snapshot_download(model_id, revision=REVISION))
    out_dir.mkdir(parents=True, exist_ok=True)
    for source in snapshot.iterdir():
        if source.name in WEIGHT_FILES or source.suffix == ".safetensors":
            target = out_dir / source.name
            target.unlink(missing_ok=True)
            target.symlink_to(source)

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    added_tokens = [token.content for token in tokenizer.added_tokens_decoder.values()]
    tokenizer.chat_template = add_input_role(official_template(tokenizer), added_tokens)
    tokenizer.save_pretrained(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-id", default="CohereLabs/tiny-aya-global")
    parser.add_argument("--out", type=Path, default=Path("models/tiny-aya-global"))
    args = parser.parse_args()
    if "_" in str(args.out.resolve()):
        raise ValueError(
            "The Meta-SecAlign runners split model paths on '_'; use a path without one."
        )
    build(args.model_id, args.out)


if __name__ == "__main__":
    main()
