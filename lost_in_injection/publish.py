"""Upload a run's adapter and outputs to Hugging Face (gen environment).

The adapter goes to its own model repository. Evaluation outputs, utility scores, the training
config and trainer logs go under <model>/<seed>/ in one dataset repository. Only regular files are
uploaded: each eval folder's symlinks point at the merged weights and benchmark inputs, and merged
weights and trainer checkpoints are left out, since the adapter and the base model rebuild them.
The adapter's card and config name the local base-model folder, which the Hub rejects, so the
uploaded copies name the Hub model instead. Uploading again adds what is new, so it can run after
training and again after evaluation. New repositories start private; they were made public for
the submission.

Phase 2's family runs are named "<model>-<run>" (e.g. "fire-english-only"), so their adapter
repos get a derived name (`ADAPTER_REPOS.get` falls back to `tiny-aya-secalign-<run>`) and their
cards and configs point at the matching Hub model. Llama and Qwen adapters are named after their
base, e.g. `Llama-3.1-8B-Instruct-secalign-english-only`: Llama's license asks that a model built
on it start its name with "Llama". Meta's own Meta-SecAlign-8B adapter is gated, so only its
outputs are uploaded.
"""

import argparse
import os
import re
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

from lost_in_injection.runs import BASE_MODELS, META_SECALIGN_RUN, OTHER_FAMILIES, model_path, split

OUTPUTS_REPO = "lost-in-injection-model-outputs"
ADAPTER_REPOS = {  # Seed 0 of the published runs; ablations and other seeds get derived names.
    "english-only": "tiny-aya-secalign-english",
    "english-plus-high-resource": "tiny-aya-secalign-high-resource",
    "english-plus-low-resource": "tiny-aya-secalign-low-resource",
}
HUB_IDS = {model_path(model).name: hub_id for model, hub_id in BASE_MODELS.items()}
LOCAL_BASE = re.compile(r"[^\s\"']*/models/(" + "|".join(map(re.escape, HUB_IDS)) + ")")


def adapter_repo(run: str) -> str:
    model, name = split(run)
    if model in OTHER_FAMILIES:
        return f"{BASE_MODELS[model].split('/')[1]}-secalign-{name}"
    return ADAPTER_REPOS.get(run, f"tiny-aya-secalign-{run}")


def output_files(run_dir: Path) -> list[Path]:
    found = []
    for root, dirs, names in os.walk(run_dir):  # Does not descend into symlinked folders.
        dirs[:] = [
            d for d in dirs if d not in ("adapter", "merged") and not d.startswith("checkpoint-")
        ]
        found += [Path(root, name) for name in names if not Path(root, name).is_symlink()]
    return sorted(found)


def content(path: Path) -> Path | bytes:
    if path.name in ("README.md", "adapter_config.json"):
        return LOCAL_BASE.sub(lambda m: HUB_IDS[m[1]], path.read_text()).encode()
    return path


def upload(
    api: HfApi, repo_id: str, repo_type: str, root: Path, paths: list[Path], prefix: str
) -> None:
    api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
    operations = [
        CommitOperationAdd(f"{prefix}{path.relative_to(root)}", content(path)) for path in paths
    ]
    api.create_commit(
        repo_id, operations, commit_message=f"Add {prefix or 'adapter'}", repo_type=repo_type
    )
    print(f"uploaded {len(paths)} files to {repo_type} {repo_id}/{prefix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="runs/<model>/<seed>")
    args = parser.parse_args()

    model, seed = args.run_dir.parts[-2:]
    api = HfApi()
    user = api.whoami()["name"]
    adapter = args.run_dir / "adapter"
    if adapter.is_dir() and model != META_SECALIGN_RUN:
        files = sorted(path for path in adapter.iterdir() if path.is_file())
        name = adapter_repo(model)
        name += "" if seed == "0" else f"-seed{seed}"
        upload(api, f"{user}/{name}", "model", adapter, files, "")
    upload(
        api,
        f"{user}/{OUTPUTS_REPO}",
        "dataset",
        args.run_dir,
        output_files(args.run_dir),
        f"{model}/{seed}/",
    )


if __name__ == "__main__":
    main()
