from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ClassifierDatasetResult:
    class_names: list[str]
    train_root: Path
    val_root: Path


def build_classifier_dataset(
    source_root: Path,
    output_root: Path,
    val_ratio: float = 0.2,
) -> ClassifierDatasetResult:
    source_root = Path(source_root)
    output_root = Path(output_root)
    train_root = output_root / "train"
    val_root = output_root / "val"
    if output_root.exists():
        shutil.rmtree(output_root)
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    class_names: list[str] = []
    for class_dir in sorted(source_root.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        files = [
            path
            for path in sorted(class_dir.iterdir())
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_SUFFIXES
            and not path.name.startswith(".")
        ]
        if not files:
            continue
        class_names.append(class_dir.name)
        split_index = max(1, int(len(files) * (1 - val_ratio)))
        train_target = train_root / class_dir.name
        val_target = val_root / class_dir.name
        train_target.mkdir(parents=True, exist_ok=True)
        val_target.mkdir(parents=True, exist_ok=True)
        for path in files[:split_index]:
            shutil.copy2(path, train_target / path.name)
        for path in files[split_index:]:
            shutil.copy2(path, val_target / path.name)
    return ClassifierDatasetResult(
        class_names=class_names,
        train_root=train_root,
        val_root=val_root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a classifier train/val split from dataset/full."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("dataset/full"),
        help="Source dataset root in <identity>/<image> layout.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("server/runtime/classifier/dataset"),
        help="Output root for prepared train/val splits.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_classifier_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        val_ratio=args.val_ratio,
    )
