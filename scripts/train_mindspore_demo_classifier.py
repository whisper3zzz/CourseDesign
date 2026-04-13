from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from mindspore import nn


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
        )
        + "\n",
        encoding="utf-8",
    )

    return label_map_path, metrics_path
