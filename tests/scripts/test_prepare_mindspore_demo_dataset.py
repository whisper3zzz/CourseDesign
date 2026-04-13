from pathlib import Path

from scripts.prepare_mindspore_demo_dataset import build_demo_dataset


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
