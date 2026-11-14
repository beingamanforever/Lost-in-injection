from lost_in_injection.publish import content, output_files


def test_output_files_skip_symlinks_weights_and_checkpoints(tmp_path):
    weights = tmp_path / "weights.safetensors"
    weights.write_text("w")
    for name in (
        "config.json",
        "eval/english/summary.tsv",
        "eval/english/data/x.json",
        "adapter/adapter_model.safetensors",
        "merged/model.safetensors",
        "logs/trainer_state.json",
        "logs/checkpoint-228/optimizer.pt",
    ):
        (tmp_path / "run" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "run" / name).write_text("x")
    (tmp_path / "run/eval/english/model.safetensors").symlink_to(weights)
    (tmp_path / "run/eval/linked").symlink_to(tmp_path / "run/merged")

    found = {
        path.relative_to(tmp_path / "run").as_posix() for path in output_files(tmp_path / "run")
    }
    assert found == {
        "config.json",
        "eval/english/summary.tsv",
        "eval/english/data/x.json",
        "logs/trainer_state.json",
    }


def test_adapter_card_and_config_name_the_hub_base_model(tmp_path):
    card = tmp_path / "README.md"
    card.write_text(
        "---\nbase_model: /home/ubuntu/aman/lii/lost-in-injection/models/tiny-aya-global\n---\n"
    )
    config = tmp_path / "adapter_config.json"
    config.write_text('{"base_model_name_or_path": "/opt/x/models/tiny-aya-global", "r": 64}')
    assert content(card) == b"---\nbase_model: CohereLabs/tiny-aya-global\n---\n"
    assert content(config) == b'{"base_model_name_or_path": "CohereLabs/tiny-aya-global", "r": 64}'
    assert content(tmp_path / "summary.tsv") == tmp_path / "summary.tsv"
