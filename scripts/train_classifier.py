from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torchvision import datasets, models, transforms


def train_classifier(
    train_root: Path,
    val_root: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = datasets.ImageFolder(
        str(train_root),
        transform=transforms.Compose(
            [
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        ),
    )
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(dataset.classes))
    weights_path = output_dir / "classifier.pt"
    torch.save(model.state_dict(), weights_path)
    label_map_path = output_dir / "label_map.json"
    label_map_path.write_text(
        json.dumps(
            {str(index): name for index, name in enumerate(dataset.classes)},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return weights_path, label_map_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight classifier on prepared face data."
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=Path("server/runtime/classifier/dataset/train"),
        help="Prepared train dataset root.",
    )
    parser.add_argument(
        "--val-root",
        type=Path,
        default=Path("server/runtime/classifier/dataset/val"),
        help="Prepared validation dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("server/runtime/classifier"),
        help="Output directory for checkpoint and label map.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_classifier(
        train_root=args.train_root,
        val_root=args.val_root,
        output_dir=args.output_dir,
    )
