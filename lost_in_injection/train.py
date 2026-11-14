"""Train one DPO + LoRA run with AlignTune's TRL backend (train environment).

The recipe is Meta-SecAlign's (REPORT.md, section 6). Several AlignTune defaults differ from
it, its `lr_scheduler` field never reaches TRL (only `lr_scheduler_type` does), and TRL's
`disable_dropout` would zero the LoRA dropout, so every value is passed explicitly. The resolved
TRL DPOConfig and LoRA config are asserted against the recipe and saved before any step runs.
Only the trained adapter is saved; TRL's frozen `ref` copy of it is the reference model.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

# Set before torch starts: a batch's fp32 logits over the 262k vocabulary take over 3 GiB, and
# without expandable segments the cached blocks are too fragmented to hold them on an A10G.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from aligntune import create_rl_trainer  # noqa: E402

EFFECTIVE_BATCH = 64
RECIPE = {
    "num_epochs": 3,
    "learning_rate": 1.6e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 0,
    "weight_decay": 0.0,
    "max_seq_length": 2048,  # export.py drops longer pairs, so nothing is truncated
    "truncation_mode": "keep_start",
    "beta": 0.1,
    "loss_type": "sigmoid",
    "lora_r": 64,
    "lora_alpha": 8,
    "lora_dropout": 0.1,
    "lora_target_modules": ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    "disable_dropout": False,
    "bf16": True,
    "precision": "bf16",
    "report_to": "tensorboard",
    "loggers": ["none"],
}


def build(model: Path, data: Path, logs: Path, seed: int, micro_batch: int, max_steps: int):
    trainer = create_rl_trainer(
        model_name=str(model.resolve()),  # A relative path would be looked up on the Hub.
        dataset_name=str(data.resolve()),
        split="train",
        algorithm="dpo",
        backend="trl",
        output_dir=str(logs.resolve()),
        batch_size=micro_batch,
        gradient_accumulation_steps=EFFECTIVE_BATCH // micro_batch,
        max_steps=max_steps,
        seed=seed,
        data_seed=seed,
        **RECIPE,
    )
    trainer.setup_model()
    trainer.setup_data()
    trainer.setup_trainer()
    return trainer


def resolved_config(trainer, rows: int) -> dict:
    args, lora = trainer.trainer.args, trainer.model.peft_config["default"]
    scheduler = getattr(args.lr_scheduler_type, "value", args.lr_scheduler_type)
    checks = {
        "rows kept by the loader": len(trainer.train_dataset) == rows,
        "effective batch": args.per_device_train_batch_size * args.gradient_accumulation_steps
        == EFFECTIVE_BATCH,
        "beta and loss": (args.beta, args.loss_type) == (0.1, ["sigmoid"]),
        "max length": args.max_length == RECIPE["max_seq_length"],
        "learning rate": args.learning_rate == RECIPE["learning_rate"],
        "cosine schedule": scheduler == "cosine",
        "no warmup or decay": (args.warmup_steps, args.weight_decay) == (0, 0.0),
        "dropout kept": not args.disable_dropout,
        "bf16 and checkpointing": args.bf16 and args.gradient_checkpointing,
        "LoRA": (lora.r, lora.lora_alpha, lora.lora_dropout) == (64, 8, 0.1)
        and set(lora.target_modules) == set(RECIPE["lora_target_modules"]),
        "reference is the frozen adapter copy": trainer.trainer.ref_model is None,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"Resolved config differs from the recipe: {failed}")
    return {"dpo_config": json.loads(args.to_json_string()), "lora_config": lora.to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=Path, default=Path("models/tiny-aya-global"))
    parser.add_argument("--data-root", type=Path, default=Path("data/runs"))
    parser.add_argument("--out-root", type=Path, default=Path("runs"))
    # DPO runs chosen and rejected together, so micro-batch 2 is 4 sequences and runs out of
    # memory on an A10G; the effective batch stays 64 either way.
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1, help="Smoke tests only.")
    args = parser.parse_args()

    data = args.data_root / args.run / "dpo.jsonl"
    out = args.out_root / args.run / str(args.seed)
    if args.max_steps > 0:
        out = args.out_root / "smoke" / args.run / str(args.seed)
    rows = len(data.read_text().splitlines())
    trainer = build(args.model, data, out / "logs", args.seed, args.micro_batch, args.max_steps)
    config = resolved_config(trainer, rows)
    config["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config["data"] = {"path": str(data), "rows": rows}
    config["data"]["sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))

    for callback in trainer.get_hf_callbacks():
        trainer.trainer.add_callback(callback)
    trainer.trainer.train()
    trainer.trainer.save_state()
    adapter = out / "adapter"
    trainer.model.save_pretrained(adapter, selected_adapters=["default"])
    trainer.tokenizer.save_pretrained(adapter)
    trainer.write_provenance(str(adapter))


if __name__ == "__main__":
    main()
