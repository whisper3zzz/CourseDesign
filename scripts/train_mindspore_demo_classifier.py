from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

try:
    import mindspore as ms
    from mindspore import Tensor, nn
    from mindspore import dataset as ds
    from mindspore.dataset import transforms, vision
    from mindspore.train.callback import Callback
except ImportError:  # pragma: no cover - system dependent
    ms = None
    Tensor = None
    nn = None
    ds = None
    transforms = None
    vision = None
    Callback = None


if nn is not None:  # pragma: no cover - depends on optional dependency
    class SmallCNN(nn.Cell):
        def __init__(self, num_classes: int, in_channels: int = 3) -> None:
            super().__init__()
            self.features = nn.SequentialCell(
                [
                    self._conv_block(in_channels, 16),
                    self._conv_block(16, 32),
                    self._conv_block(32, 64),
                ]
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
            self.classifier = nn.Dense(64, num_classes)

        @staticmethod
        def _conv_block(in_channels: int, out_channels: int) -> nn.Cell:
            return nn.SequentialCell(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        pad_mode="pad",
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )

        def construct(self, x):  # type: ignore[override]
            x = self.features(x)
            x = self.pool(x)
            x = self.flatten(x)
            return self.classifier(x)
else:
    class SmallCNN:  # pragma: no cover - depends on optional dependency
        def __init__(self, *_, **__):
            raise RuntimeError("MindSpore is required to instantiate SmallCNN")


def write_training_metadata(
    output_root: Path,
    class_names: Sequence[str],
    sample_count: int,
    epochs: int,
    last_train_loss: float,
    last_val_accuracy: float,
) -> tuple[Path, Path]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    label_map_path = output_root / "label_map.json"
    metrics_path = output_root / "metrics.json"

    label_map_path.write_text(
        json.dumps(
            {str(index): name for index, name in enumerate(class_names)},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(
            {
                "class_count": len(class_names),
                "sample_count": sample_count,
                "epochs": epochs,
                "last_train_loss": float(last_train_loss),
                "last_val_accuracy": float(last_val_accuracy),
                "trained_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return label_map_path, metrics_path


def _assert_dataset_ready(split_root: Path, split_name: str) -> None:
    if not split_root.exists() or not split_root.is_dir():
        raise RuntimeError(f"{split_name} dataset directory missing: {split_root}")
    has_files = any(path.is_file() for path in split_root.rglob("*"))
    if not has_files:
        raise RuntimeError(f"{split_name} dataset directory is empty: {split_root}")


def _extract_class_names(dataset) -> list[str]:
    class_indexing = None
    if hasattr(dataset, "get_class_indexing"):
        class_indexing = dataset.get_class_indexing()
    elif hasattr(dataset, "class_indexing"):
        class_indexing = dataset.class_indexing

    if not class_indexing:
        raise RuntimeError("Unable to determine class names from dataset.")

    return [
        name
        for name, _ in sorted(class_indexing.items(), key=lambda item: item[1])
    ]


def _create_dataset(
    split_root: Path,
    batch_size: int,
    shuffle: bool,
) -> tuple[object, list[str], int]:
    base_dataset = ds.ImageFolderDataset(str(split_root), shuffle=shuffle)
    sample_count = base_dataset.get_dataset_size()
    if sample_count <= 0:
        raise RuntimeError(f"No samples found in {split_root}")

    class_names = _extract_class_names(base_dataset)

    image_ops = [
        vision.Decode(),
        vision.Resize((64, 64)),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.HWC2CHW(),
    ]
    label_ops = [transforms.TypeCast(ms.int32)]

    dataset = base_dataset.map(image_ops, input_columns="image")
    dataset = dataset.map(label_ops, input_columns="label")
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset, class_names, sample_count


class LossRecorder(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.last_loss: float | None = None

    def step_end(self, run_context) -> None:
        cb_params = run_context.original_args()
        loss_value = cb_params.net_outputs
        if isinstance(loss_value, (tuple, list)):
            loss_value = loss_value[0]
        if isinstance(loss_value, Tensor):
            loss_value = loss_value.asnumpy()
        self.last_loss = float(loss_value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a minimal MindSpore demo classifier and write artifacts."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("server/runtime/mindspore_demo/dataset"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("server/runtime/mindspore_demo"),
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if ms is None:
        raise RuntimeError("MindSpore is required to run the full training workflow")

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

    dataset_root = Path(args.dataset_root)
    train_root = dataset_root / "train"
    val_root = dataset_root / "val"
    _assert_dataset_ready(train_root, "train")
    _assert_dataset_ready(val_root, "val")

    train_dataset, class_names, sample_count = _create_dataset(
        train_root, batch_size=args.batch_size, shuffle=True
    )
    val_dataset, val_class_names, _ = _create_dataset(
        val_root, batch_size=args.batch_size, shuffle=False
    )
    if val_class_names != class_names:
        raise RuntimeError("Train/val class names do not match.")

    network = SmallCNN(num_classes=len(class_names))
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Adam(
        network.trainable_params(), learning_rate=args.learning_rate
    )
    model = ms.Model(
        network, loss_fn=loss_fn, optimizer=optimizer, metrics={"accuracy": nn.Accuracy()}
    )

    loss_recorder = LossRecorder()
    model.train(
        args.epochs,
        train_dataset,
        callbacks=[loss_recorder],
        dataset_sink_mode=False,
    )

    if loss_recorder.last_loss is None:
        raise RuntimeError("Training did not produce a loss value.")

    eval_metrics = model.eval(val_dataset, dataset_sink_mode=False)
    val_accuracy = float(eval_metrics.get("accuracy", 0.0))

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    ms.save_checkpoint(network, str(output_root / "mindspore_classifier.ckpt"))

    write_training_metadata(
        output_root=output_root,
        class_names=class_names,
        sample_count=sample_count,
        epochs=args.epochs,
        last_train_loss=loss_recorder.last_loss,
        last_val_accuracy=val_accuracy,
    )


if __name__ == "__main__":  # pragma: no cover - runtime entrypoint
    main()
