from pathlib import Path

import pytest

from server.classifier_backend import ClassifierBackend, ClassifierModelNotReady


def test_predict_bytes_raises_when_model_is_missing(tmp_path: Path) -> None:
    backend = ClassifierBackend(
        model_path=tmp_path / "missing.mindir",
        label_map_path=tmp_path / "label_map.json",
    )

    with pytest.raises(ClassifierModelNotReady):
        backend.predict_bytes(b"fake-image")


def test_predict_bytes_raises_when_label_map_is_missing(tmp_path: Path) -> None:
    model_path = tmp_path / "classifier.mindir"
    model_path.write_bytes(b"mindir")
    backend = ClassifierBackend(
        model_path=model_path,
        label_map_path=tmp_path / "missing.json",
    )

    with pytest.raises(ClassifierModelNotReady):
        backend.predict_bytes(b"fake-image")
