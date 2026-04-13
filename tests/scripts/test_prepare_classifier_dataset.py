from pathlib import Path

from scripts.prepare_classifier_dataset import build_classifier_dataset


def test_build_classifier_dataset_creates_train_and_val_splits(tmp_path: Path) -> None:
    source = tmp_path / "dataset" / "full"
    (source / "alice").mkdir(parents=True)
    (source / "bob").mkdir(parents=True)
    for index in range(4):
        (source / "alice" / f"a{index}.jpg").write_bytes(b"jpg")
        (source / "bob" / f"b{index}.jpg").write_bytes(b"jpg")

    output = tmp_path / "prepared"
    result = build_classifier_dataset(
        source_root=source,
        output_root=output,
        val_ratio=0.25,
    )

    assert (output / "train" / "alice").is_dir()
    assert (output / "val" / "bob").is_dir()
    assert result.class_names == ["alice", "bob"]
