from pathlib import Path

import numpy as np
import pytest

from server.mindspore_backend import MindSporeEmbeddingBackend, MindSporeModelNotReady


def test_embed_bytes_raises_when_mindir_is_missing(tmp_path: Path) -> None:
    backend = MindSporeEmbeddingBackend(model_path=tmp_path / "missing.mindir")

    with pytest.raises(MindSporeModelNotReady):
        backend.embed_bytes(b"fake-image")


def test_recognize_returns_backend_name_when_model_is_ready(
    tmp_path: Path, monkeypatch
) -> None:
    model_path = tmp_path / "facenet_vggface2.mindir"
    model_path.write_bytes(b"mindir")

    backend = MindSporeEmbeddingBackend(model_path=model_path)
    monkeypatch.setattr(
        backend, "embed_bytes", lambda image_bytes: np.ones((512,), dtype=np.float32)
    )

    payload = backend.recognize_bytes(
        b"fake-image", {"alice": np.ones((512,), dtype=np.float32)}
    )

    assert payload["backend"] == "mindspore_embedding"
