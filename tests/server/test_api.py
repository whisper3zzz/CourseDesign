import io
from pathlib import Path

import numpy as np
import server.embedding_service as embedding_service
from fastapi.testclient import TestClient
from PIL import Image

from server.embedding_service import create_app, validate_identity_name
from server.mindspore_backend import MindSporeModelNotReady


class DummyStore:
    def __init__(
        self,
        centroids: dict[str, np.ndarray] | None = None,
        identity_count: int = 0,
        embedding_count: int = 0,
    ) -> None:
        self._centroids = centroids or {}
        self._identity_count = identity_count
        self._embedding_count = embedding_count

    def load_all_centroids(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._centroids.items()}

    def identity_count(self) -> int:
        return self._identity_count

    def embedding_count(self) -> int:
        return self._embedding_count


class DummyService:
    def __init__(self, store: DummyStore) -> None:
        self.store = store
        self.threshold = 0.72
        self.device = "cpu"
        self.model_name = "fake"

    def recognize(self, image_bytes: bytes) -> dict:
        return {
            "name": "unknown",
            "matched": False,
            "score": 0.0,
            "threshold": self.threshold,
            "identity_count": self.store.identity_count(),
            "embedding_count": self.store.embedding_count(),
        }

    def list_identities(self) -> dict:
        return {
            "identities": [],
            "identity_count": self.store.identity_count(),
            "embedding_count": self.store.embedding_count(),
        }

    def register(self, name: str, image_bytes: bytes) -> dict:
        clean_name = validate_identity_name(name)
        return {
            "name": clean_name,
            "sample_count": 1,
            "identity_count": self.store.identity_count(),
            "embedding_count": self.store.embedding_count(),
            "status": "ok",
        }

    def delete_identity(self, name: str) -> dict:
        clean_name = validate_identity_name(name)
        return {
            "name": clean_name,
            "removed": True,
            "identity_count": self.store.identity_count(),
            "embedding_count": self.store.embedding_count(),
            "status": "ok",
        }


class ReadyMindSporeBackend:
    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)

    def is_ready(self) -> bool:
        return True


class FailingMindSporeBackend:
    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)

    def is_ready(self) -> bool:
        return False

    def recognize_bytes(self, image_bytes: bytes, centroids: dict[str, np.ndarray]) -> dict:
        raise MindSporeModelNotReady("missing model")


def make_test_image_bytes() -> bytes:
    image = Image.new("RGB", (1, 1), (0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health_reports_active_backend(tmp_path: Path) -> None:
    service = DummyService(DummyStore())
    backend = ReadyMindSporeBackend(tmp_path / "facenet_vggface2.mindir")
    app = create_app(service=service, mindspore_backend=backend)

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["active_backend"] == "mindspore_embedding"
    assert payload["mindspore_model_ready"] is True
    assert payload["mindspore_model_path"] == str(backend.model_path)


def test_recognize_falls_back_to_pytorch_when_mindspore_not_ready(
    tmp_path: Path,
) -> None:
    centroids = {"alice": np.array([1.0, 0.0], dtype=np.float32)}
    store = DummyStore(centroids=centroids, identity_count=1, embedding_count=1)
    service = DummyService(store)
    backend = FailingMindSporeBackend(tmp_path / "facenet_vggface2.mindir")
    app = create_app(service=service, mindspore_backend=backend)

    client = TestClient(app)
    response = client.post(
        "/recognize",
        files={"photo": ("face.jpg", make_test_image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "pytorch_embedding"
    assert payload["ready"] is False
    assert payload["threshold"] == service.threshold
    assert payload["identity_count"] == store.identity_count()
    assert payload["embedding_count"] == store.embedding_count()
    assert payload["detail"] == "mindspore model not ready, fallback active"


def test_register_rejects_dotdot_name(tmp_path: Path) -> None:
    service = DummyService(DummyStore())
    backend = ReadyMindSporeBackend(tmp_path / "facenet_vggface2.mindir")
    app = create_app(service=service, mindspore_backend=backend)

    client = TestClient(app)
    response = client.post(
        "/register",
        data={"name": ".."},
        files={"photo": ("face.jpg", b"fake-image", "image/jpeg")},
    )

    assert response.status_code == 400


def test_create_app_is_lazy_about_service_construction(tmp_path: Path, monkeypatch) -> None:
    def fail_service(*args, **kwargs):
        raise AssertionError("FaceEmbeddingService constructed eagerly")

    def fail_backend(*args, **kwargs):
        raise AssertionError("MindSporeEmbeddingBackend constructed eagerly")

    monkeypatch.setattr(embedding_service, "FaceEmbeddingService", fail_service)
    monkeypatch.setattr(embedding_service, "MindSporeEmbeddingBackend", fail_backend)

    app = embedding_service.create_app(runtime_root=tmp_path)

    assert app is not None
