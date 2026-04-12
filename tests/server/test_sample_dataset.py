from pathlib import Path

from server.sample_dataset import build_training_snapshot, stage_imagefolder_dataset


def test_build_training_snapshot_ignores_hidden_and_empty_identities(tmp_path: Path) -> None:
    faces_root = tmp_path / "faces"
    (faces_root / "alice").mkdir(parents=True)
    (faces_root / "alice" / "1.jpg").write_bytes(b"jpg")
    (faces_root / "bob").mkdir(parents=True)
    (faces_root / ".hidden").mkdir(parents=True)
    (faces_root / ".hidden" / "1.jpg").write_bytes(b"jpg")

    snapshot = build_training_snapshot(faces_root)

    assert snapshot.class_names == ["alice"]
    assert snapshot.sample_count == 1
    assert snapshot.identity_count == 1


def test_stage_imagefolder_dataset_copies_files_into_class_dirs(tmp_path: Path) -> None:
    faces_root = tmp_path / "faces"
    (faces_root / "alice").mkdir(parents=True)
    (faces_root / "alice" / "1.jpg").write_bytes(b"jpg")

    snapshot = build_training_snapshot(faces_root)
    staged_root = stage_imagefolder_dataset(snapshot, tmp_path / "staging")

    assert (staged_root / "alice" / "1.jpg").exists()
