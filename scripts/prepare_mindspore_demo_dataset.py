from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

SUPPORTED_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class DemoDatasetResult:
    class_names: list[str]


def _iterate_identities(source_root: Path) -> Sequence[Path]:
    return sorted(
        child
        for child in source_root.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    )


def _image_files(identity_root: Path) -> list[Path]:
    return sorted(
        file
        for file in identity_root.iterdir()
        if file.is_file()
        and not file.name.startswith(".")
        and file.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _copy_to_directory(files: Iterable[Path], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in files:
        shutil.copy2(source, target_dir / source.name)


def build_demo_dataset(source_root: Path, output_root: Path, val_ratio: float = 0.2) -> DemoDatasetResult:
    source_root = Path(source_root)
    output_root = Path(output_root)

    if not source_root.is_dir():
        raise ValueError("source_root must point to an existing directory with identities")
    if not 0 <= val_ratio <= 1:
        raise ValueError("val_ratio must be between 0 and 1 inclusive")

    dataset_root = output_root / "dataset"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    train_root = dataset_root / "train"
    val_root = dataset_root / "val"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    class_names: list[str] = []
    for identity in _iterate_identities(source_root):
        images = _image_files(identity)
        if not images:
            continue

        class_names.append(identity.name)
        val_count = int(len(images) * val_ratio)
        val_images = images[:val_count]
        train_images = images[val_count:]

        _copy_to_directory(val_images, val_root / identity.name)
        _copy_to_directory(train_images, train_root / identity.name)

    return DemoDatasetResult(class_names=class_names)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a MindSpore demo dataset split layout")
    parser.add_argument(
        "--source-root", type=Path, default=Path("dataset") / "full", help="Root where identities live"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("server") / "runtime" / "mindspore_demo",
        help="Root where the MindSpore layout is emitted",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of each identity placed into the val set",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_demo_dataset(
        source_root=args.source_root, output_root=args.output_root, val_ratio=args.val_ratio
    )
    print(f"Built demo dataset for classes: {', '.join(result.class_names)}")


if __name__ == "__main__":
    main()
