import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

import server.embedding_service as embedding_service
from server.classifier_assets import ClassifierArtifactPaths
from server.embedding_service import create_app
from scripts.prepare_classifier_dataset import ClassifierDatasetResult


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


def test_train_classifier_success(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    faces_root = tmp_path / "faces" / "Alice"
    faces_root.mkdir(parents=True, exist_ok=True)
    (faces_root / "face.jpg").write_bytes(b"fake")
    assets = ClassifierArtifactPaths(tmp_path / "classifier")
    classifier_root = assets.root
    prepared_root = classifier_root / "dataset"
    expected_source = (tmp_path / "faces").resolve()
    calls: dict[str, tuple] = {}

    def fake_build_classifier_dataset(
        source_root: Path, output_root: Path, val_ratio: float
    ) -> ClassifierDatasetResult:
        calls["build"] = (source_root, output_root, val_ratio)
        return ClassifierDatasetResult(
            class_names=["Alice", "Bob"],
            train_root=output_root / "train",
            val_root=output_root / "val",
        )

    def fake_train_classifier(
        train_root: Path, val_root: Path, output_dir: Path
    ) -> tuple[Path, Path]:
        calls["train"] = (train_root, val_root, output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_path = output_dir / "classifier.pt"
        label_map_path = output_dir / "label_map.json"
        weights_path.write_text("stub", encoding="utf-8")
        label_map_path.write_text(
            json.dumps({"0": "Alice", "1": "Bob"}, ensure_ascii=False),
            encoding="utf-8",
        )
        return weights_path, label_map_path

    def fake_export_classifier(
        weights_path: Path, label_map_path: Path, output_path: Path
    ) -> Path:
        calls["export"] = (weights_path, label_map_path, output_path)
        output_path.write_text("onnx", encoding="utf-8")
        return output_path

    monkeypatch.setattr(
        embedding_service, "build_classifier_dataset", fake_build_classifier_dataset
    )
    monkeypatch.setattr(embedding_service, "train_classifier", fake_train_classifier)
    monkeypatch.setattr(
        embedding_service, "export_classifier", fake_export_classifier
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
    response = client.post("/train_classifier")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["backend"] == "cnn_classifier"
    assert payload["class_count"] == 2
    assert payload["train_root"] == str(prepared_root / "train")
    assert payload["val_root"] == str(prepared_root / "val")
    assert payload["weights_path"] == str(classifier_root / "classifier.pt")
    assert payload["label_map_path"] == str(classifier_root / "label_map.json")
    assert payload["onnx_path"] == str(classifier_root / "classifier.onnx")
    assert payload["classifier_model_ready"] is False
    assert payload["classifier_model_path"] == str(assets.mindir_path)
    assert payload["classifier_label_map_path"] == str(assets.label_map_path)
    assert calls["build"] == (expected_source, prepared_root, 0.2)
    assert calls["train"] == (
        prepared_root / "train",
        prepared_root / "val",
        classifier_root,
    )
    assert calls["export"] == (
        classifier_root / "classifier.pt",
        classifier_root / "label_map.json",
        classifier_root / "classifier.onnx",
    )


def test_train_classifier_missing_dataset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
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

    assert response.status_code == 400
    payload = response.json()
    assert "faces" in payload["detail"]


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
