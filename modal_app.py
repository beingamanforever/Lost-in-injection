"""Run this repository's own commands on Modal GPUs, with the same pinned environments as the A10Gs.

Code comes from this checkout; data/, models/, runs/ and the Hugging Face cache live on one Modal
volume, so every call sees earlier outputs and results can be pulled with `modal volume get`.
The gated models download with the `huggingface` Modal secret (HF_TOKEN).

Usage: modal run --detach modal_app.py --gpu H100 --cmd "bash scripts/run_models.sh <run>"
The call returns once the job is spawned; follow it with `modal app logs <app id>`.
"""

import subprocess

import modal

ROOT = "/repo"
VOLUME = modal.Volume.from_name("lost-in-injection", create_if_missing=True)

# Same steps as envs/setup.sh; paths point at the volume instead of the A10G's NVMe disk. One line
# with \n escapes, since a Dockerfile RUN cannot span lines.
ENV_SH = "\\n".join(
    [
        "export HF_HOME=/vol/hf",
        "export NLTK_DATA=/opt/nltk",
        "export TMPDIR=/tmp",
        "export UV_TORCH_BACKEND=cu128",
    ]
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "pkg-config", "libicu-dev")  # pyicu (train env)
    .pip_install("uv==0.13.8")  # same as the A10Gs; older uv cannot resolve torchdata on cu128
    .env({"UV_TORCH_BACKEND": "cu128"})
    .add_local_dir("envs", f"{ROOT}/envs", copy=True)
    .add_local_file(
        "third_party/Meta_SecAlign/requirements.txt",
        f"{ROOT}/third_party/Meta_SecAlign/requirements.txt",
        copy=True,
    )
    .run_commands(
        f"cd {ROOT} && uv venv /opt/gen --python 3.12",
        f"cd {ROOT} && VIRTUAL_ENV=/opt/gen uv pip install -r envs/gen.txt",
        "/opt/gen/bin/python -m nltk.downloader -d /opt/nltk"
        " averaged_perceptron_tagger_eng punkt_tab",
        f"cd {ROOT} && uv venv /opt/train --python 3.12",
        f"cd {ROOT} && VIRTUAL_ENV=/opt/train uv pip install -r envs/train.txt",
        f"printf '{ENV_SH}\\n' > {ROOT}/env.sh",
        f"ln -s /opt/gen {ROOT}/.venv-gen && ln -s /opt/train {ROOT}/.venv-train",
        f"for d in data models runs; do ln -s /vol/$d {ROOT}/$d; done",
    )
    .add_local_dir("lost_in_injection", f"{ROOT}/lost_in_injection", ignore=["__pycache__"])
    .add_local_dir("scripts", f"{ROOT}/scripts")
    .add_local_dir("third_party/Meta_SecAlign", f"{ROOT}/third_party/Meta_SecAlign")
)
app = modal.App("lost-in-injection", image=image)


@app.function(
    volumes={"/vol": VOLUME},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=24 * 3600,
    memory=32768,
)
def run(cmd: str) -> None:
    subprocess.run(["mkdir", "-p", "/vol/data", "/vol/models", "/vol/runs", "/vol/hf"], check=True)
    try:
        subprocess.run(["bash", "-c", f"cd {ROOT} && source env.sh && {cmd}"], check=True)
    finally:
        VOLUME.commit()


@app.local_entrypoint()
def main(cmd: str, gpu: str = "H100", memory: int = 32768) -> None:
    # train.py records this checkout's commit ("-dirty" with local edits); the image has no .git.
    commit = subprocess.run(
        ["git", "describe", "--always", "--dirty"], capture_output=True, text=True, check=True
    ).stdout.strip()
    # --gpu "" runs on CPU only; 70B runs take --gpu H200:2 --memory 131072 (MiB of RAM).
    # spawn, with `modal run --detach`, keeps the job running when this client loses the network.
    job = run.with_options(gpu=gpu or None, memory=memory)
    print("spawned", job.spawn(f"export GIT_COMMIT={commit}; {cmd}").object_id)
