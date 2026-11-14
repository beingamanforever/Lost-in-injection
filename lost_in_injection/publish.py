"""Upload a run's adapter and outputs to private Hugging Face repositories (gen environment).

The adapter goes to its own model repository. Evaluation outputs, utility scores, the training
config and trainer logs go under <model>/<seed>/ in one dataset repository. Only regular files are
uploaded: each eval folder's symlinks point at the merged weights and benchmark inputs, and merged
weights and trainer checkpoints are left out, since the adapter and the base model rebuild them.
The adapter's card and config name the local base-model folder, which the Hub rejects, so the
uploaded copies name the Hub model instead. Uploading again adds what is new, so it can run after
training and again after evaluation.
"""

import argparse
import os
import re
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

OUTPUTS_REPO = "lost-in-injection-outputs"
LOCAL_BASE = re.compile(r"[^\s\"']*/models/tiny-aya-global")
HUB_BASE = "CohereLabs/tiny-aya-global"


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
        return LOCAL_BASE.sub(HUB_BASE, path.read_text()).encode()
    return path


def upload(
    api: HfApi, repo_id: str, repo_type: str, root: Path, paths: list[Path], prefix: str
) -> None:
    api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type=repo_type).private:
        raise SystemExit(f"{repo_id} is public; nothing uploaded")
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
    if adapter.is_dir():
        files = sorted(path for path in adapter.iterdir() if path.is_file())
        upload(api, f"{user}/lost-in-injection-{model}-seed{seed}", "model", adapter, files, "")
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
