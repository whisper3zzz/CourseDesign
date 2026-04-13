from __future__ import annotations

import json
from pathlib import Path

from scripts.train_mindspore_demo_classifier import parse_args, write_training_metadata


def test_write_training_metadata_writes_metrics_and_label_map(tmp_path: Path) -> None:
    output_root = tmp_path / "mindspore_demo"
    label_map_path, metrics_path = write_training_metadata(
        output_root=output_root,
        class_names=["alice", "bob"],
        sample_count=10,
        epochs=2,
        last_train_loss=0.42,
        last_val_accuracy=0.75,
    )

    assert label_map_path.exists()
    assert metrics_path.exists()

    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert label_map == {"0": "alice", "1": "bob"}
    assert metrics["class_count"] == 2
    assert metrics["sample_count"] == 10
    assert metrics["epochs"] == 2
    assert metrics["last_train_loss"] == 0.42
    assert metrics["last_val_accuracy"] == 0.75
    assert isinstance(metrics["trained_at"], str)


def test_parse_args_defaults_for_training() -> None:
    args = parse_args([])

    assert args.dataset_root == Path("server/runtime/mindspore_demo/dataset")
    assert args.output_root == Path("server/runtime/mindspore_demo")
    assert args.epochs == 2
    assert args.batch_size == 8
    assert args.learning_rate == 1e-3
