from pathlib import Path
from typing import Sequence

from scripts.prepare_mindspore_demo_dataset import build_demo_dataset


def _listed_files(directory: Path) -> Sequence[str]:
    if not directory.exists():
        return []
    return sorted(entry.name for entry in directory.iterdir() if entry.is_file())


def test_build_demo_dataset_creates_train_and_val_roots(tmp_path: Path) -> None:
    source = tmp_path / "dataset" / "full"
    (source / "alice").mkdir(parents=True)
    (source / "bob").mkdir(parents=True)
    for index in range(5):
        (source / "alice" / f"a{index}.jpg").write_bytes(b"jpg")
        (source / "bob" / f"b{index}.jpg").write_bytes(b"jpg")

    output = tmp_path / "mindspore_demo"
    result = build_demo_dataset(source_root=source, output_root=output, val_ratio=0.2)

    assert (output / "dataset" / "train" / "alice").is_dir()
    assert (output / "dataset" / "val" / "bob").is_dir()
    assert result.class_names == ["alice", "bob"]


def test_single_image_class_only_creates_train_dir(tmp_path: Path) -> None:
    source = tmp_path / "dataset" / "full"
    (source / "solo").mkdir(parents=True)
    (source / "solo" / "only.jpg").write_bytes(b"jpg")

    output = tmp_path / "mindspore_demo"
    result = build_demo_dataset(source_root=source, output_root=output, val_ratio=0.5)

    train_dir = output / "dataset" / "train" / "solo"
    val_dir = output / "dataset" / "val" / "solo"

    assert train_dir.is_dir()
    assert not val_dir.exists()
    assert _listed_files(train_dir) == ["only.jpg"]
    assert result.class_names == ["solo"]


def test_small_class_preserves_train_and_val_counts(tmp_path: Path) -> None:
    source = tmp_path / "dataset" / "full"
    (source / "duo").mkdir(parents=True)

    (source / "duo" / "a.jpg").write_bytes(b"jpg")
    (source / "duo" / "b.jpg").write_bytes(b"jpg")

    output = tmp_path / "mindspore_demo"
    result = build_demo_dataset(source_root=source, output_root=output, val_ratio=0.5)

    train_dir = output / "dataset" / "train" / "duo"
    val_dir = output / "dataset" / "val" / "duo"

    assert train_dir.is_dir()
    assert val_dir.is_dir()
    assert len(_listed_files(train_dir)) >= 1
    assert len(_listed_files(val_dir)) >= 1
    assert len(_listed_files(train_dir)) + len(_listed_files(val_dir)) == 2
    assert result.class_names == ["duo"]
