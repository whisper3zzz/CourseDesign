import server.mindspore_backend as backend_module
from pathlib import Path

import numpy as np
import pytest

from server.mindspore_backend import MindSporeEmbeddingBackend, MindSporeModelNotReady


def test_embed_bytes_raises_when_mindir_is_missing(tmp_path: Path) -> None:
    backend = MindSporeEmbeddingBackend(model_path=tmp_path / "missing.mindir")

    with pytest.raises(MindSporeModelNotReady):
        backend.embed_bytes(b"fake-image")


def test_embed_bytes_raises_when_mindspore_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "facenet_vggface2.mindir"
    model_path.write_bytes(b"mindir")
    monkeypatch.setattr(backend_module, "ms", None)

    backend = MindSporeEmbeddingBackend(model_path=model_path)

    with pytest.raises(MindSporeModelNotReady):
        backend.embed_bytes(b"fake-image")


def test_recognize_returns_unknown_when_score_is_below_threshold(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "facenet_vggface2.mindir"
    model_path.write_bytes(b"mindir")

    backend = MindSporeEmbeddingBackend(model_path=model_path, threshold=0.72)
    monkeypatch.setattr(
        backend,
        "embed_bytes",
        lambda image_bytes: np.array([1.0, 0.0], dtype=np.float32),
    )

    payload = backend.recognize_bytes(
        b"fake-image", {"alice": np.array([0.0, 1.0], dtype=np.float32)}
    )

    assert payload["name"] == "未知"
    assert payload["matched"] is False
    assert payload["backend"] == "mindspore_embedding"


def test_recognize_returns_name_when_score_meets_threshold(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "facenet_vggface2.mindir"
    model_path.write_bytes(b"mindir")

    backend = MindSporeEmbeddingBackend(model_path=model_path, threshold=0.72)
    monkeypatch.setattr(
        backend,
        "embed_bytes",
        lambda image_bytes: np.array([1.0, 0.0], dtype=np.float32),
    )

    payload = backend.recognize_bytes(
        b"fake-image", {"alice": np.array([1.0, 0.0], dtype=np.float32)}
    )

    assert payload["name"] == "alice"
    assert payload["matched"] is True
    assert payload["backend"] == "mindspore_embedding"
