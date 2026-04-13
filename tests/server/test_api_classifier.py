import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from server.classifier_assets import ClassifierArtifactPaths
from server.embedding_service import create_app


class DummyStore:
    def identity_count(self) -> int:
        return 0

    def embedding_count(self) -> int:
        return 0

    def load_all_centroids(self) -> dict:
        return {}


class DummyService:
    def __init__(self) -> None:
        self.store = DummyStore()
        self.threshold = 0.72
        self.device = "cpu"
        self.model_name = "fake"


class DummyMindSporeBackend:
    def __init__(self, model_path: Path, ready: bool = True) -> None:
        self.model_path = Path(model_path)
        self._ready = ready

    def is_ready(self) -> bool:
        return self._ready


class DummyClassifierBackend:
    def __init__(
        self,
        model_path: Path,
        label_map_path: Path,
        ready: bool = False,
        response: dict | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.label_map_path = Path(label_map_path)
        self._ready = ready
        self._response = response or {
            "name": "Alice",
            "matched": True,
            "score": 0.93,
            "candidate_name": "Alice",
            "candidate_score": 0.93,
            "backend": "cnn_classifier",
            "ready": True,
        }
        self.predict_calls = 0

    def is_ready(self) -> bool:
        return self._ready

    def predict_bytes(self, image_bytes: bytes) -> dict:
        self.predict_calls += 1
        if not self._ready:
            raise RuntimeError("classifier not ready")
        return dict(self._response)


def make_png_bytes() -> bytes:
    image = Image.new("RGB", (2, 2), color=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


PNG_BYTES = make_png_bytes()


def test_health_reports_classifier_fields(tmp_path: Path) -> None:
    assets = ClassifierArtifactPaths(tmp_path / "classifier")
    assets.root.mkdir(parents=True, exist_ok=True)
    assets.label_map_path.write_text(
        json.dumps({"0": "Alice", "1": "Bob"}, ensure_ascii=False),
        encoding="utf-8",
    )
    app = create_app(
        runtime_root=tmp_path,
        service=DummyService(),
        mindspore_backend=DummyMindSporeBackend(tmp_path / "facenet_vggface2.mindir"),
        classifier_backend=DummyClassifierBackend(
            assets.mindir_path,
            assets.label_map_path,
            ready=False,
        ),
    )

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["classifier_model_ready"] is False
    assert payload["classifier_model_path"] == str(assets.mindir_path)
    assert payload["classifier_label_map_path"] == str(assets.label_map_path)
    assert payload["classifier_backend_available"] is True
    assert payload["classifier_label_map_exists"] is True
    assert payload["classifier_class_count"] == 2


def test_train_classifier_endpoint_exists(tmp_path: Path) -> None:
    assets = ClassifierArtifactPaths(tmp_path / "classifier")
    app = create_app(
        runtime_root=tmp_path,
        service=DummyService(),
        mindspore_backend=DummyMindSporeBackend(tmp_path / "facenet_vggface2.mindir"),
        classifier_backend=DummyClassifierBackend(
            assets.mindir_path,
            assets.label_map_path,
            ready=False,
        ),
    )

    client = TestClient(app)
    response = client.post("/train_classifier")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "not_ready"
    assert payload["backend"] == "cnn_classifier"
    assert payload["classifier_model_ready"] is False
    assert payload["classifier_model_path"] == str(assets.mindir_path)
    assert payload["classifier_label_map_path"] == str(assets.label_map_path)


def test_recognize_routes_to_classifier_backend(tmp_path: Path) -> None:
    assets = ClassifierArtifactPaths(tmp_path / "classifier")
    classifier_backend = DummyClassifierBackend(
        assets.mindir_path,
        assets.label_map_path,
        ready=True,
    )
    app = create_app(
        runtime_root=tmp_path,
        service=DummyService(),
        mindspore_backend=DummyMindSporeBackend(tmp_path / "facenet_vggface2.mindir"),
        classifier_backend=classifier_backend,
    )

    client = TestClient(app)
    response = client.post(
        "/recognize",
        data={"backend": "cnn_classifier"},
        files={"photo": ("face.png", PNG_BYTES, "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "cnn_classifier"
    assert payload["ready"] is True
    assert payload["name"] == "Alice"
    assert payload["threshold"] == 0.72
    assert classifier_backend.predict_calls == 1


def test_recognize_classifier_not_ready(tmp_path: Path) -> None:
    assets = ClassifierArtifactPaths(tmp_path / "classifier")
    classifier_backend = DummyClassifierBackend(
        assets.mindir_path,
        assets.label_map_path,
        ready=False,
    )
    app = create_app(
        runtime_root=tmp_path,
        service=DummyService(),
        mindspore_backend=DummyMindSporeBackend(tmp_path / "facenet_vggface2.mindir"),
        classifier_backend=classifier_backend,
    )

    client = TestClient(app)
    response = client.post(
        "/recognize",
        data={"backend": "cnn_classifier"},
        files={"photo": ("face.png", PNG_BYTES, "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "cnn_classifier"
    assert payload["ready"] is False
    assert payload["detail"] == "classifier model not ready"
    assert classifier_backend.predict_calls == 0
