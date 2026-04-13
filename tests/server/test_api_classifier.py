from pathlib import Path

from fastapi.testclient import TestClient

from server.classifier_assets import ClassifierArtifactPaths
from server.embedding_service import create_app


class DummyStore:
    def identity_count(self) -> int:
        return 0

    def embedding_count(self) -> int:
        return 0


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
    def __init__(self, model_path: Path, label_map_path: Path, ready: bool = False) -> None:
        self.model_path = Path(model_path)
        self.label_map_path = Path(label_map_path)
        self._ready = ready

    def is_ready(self) -> bool:
        return self._ready


def test_health_reports_classifier_fields(tmp_path: Path) -> None:
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
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["classifier_model_ready"] is False
    assert payload["classifier_model_path"] == str(assets.mindir_path)
    assert payload["classifier_label_map_path"] == str(assets.label_map_path)
    assert payload["classifier_backend_available"] is True


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
