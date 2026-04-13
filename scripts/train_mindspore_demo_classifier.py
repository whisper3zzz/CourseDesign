from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

try:
    import mindspore as ms
    from mindspore import Tensor, nn
except ImportError:  # pragma: no cover - system dependent
    ms = None
    Tensor = None
    nn = None


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write metadata for the MindSpore demo classifier training run."
    )
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--class-names", nargs="+", default=["alice", "bob"])
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--last-train-loss", type=float, default=0.42)
    parser.add_argument("--last-val-accuracy", type=float, default=0.75)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if ms is None:
        raise RuntimeError("MindSpore is required to run the full training workflow")

    write_training_metadata(
        output_root=args.output_root,
        class_names=args.class_names,
        sample_count=args.sample_count,
        epochs=args.epochs,
        last_train_loss=args.last_train_loss,
        last_val_accuracy=args.last_val_accuracy,
    )


if __name__ == "__main__":  # pragma: no cover - runtime entrypoint
    main()
