from __future__ import annotations

import io
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms

from scripts.export_classifier_to_onnx import export_classifier
from scripts.prepare_classifier_dataset import SUPPORTED_SUFFIXES, build_classifier_dataset
from scripts.train_classifier import train_classifier
from server.classifier_assets import ClassifierArtifactPaths
from server.classifier_backend import ClassifierBackend
from server.mindspore_backend import MindSporeEmbeddingBackend, MindSporeModelNotReady
from server.model_assets import ModelArtifactPaths


def choose_device() -> str:
    requested = os.getenv("FACE_SERVICE_DEVICE", "").strip().lower()
    if requested in {"cpu", "cuda"}:
        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_identity_name(name: str) -> str:
    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="name is empty")
    if clean_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="name cannot be '.' or '..'")
    if "/" in clean_name or "\\" in clean_name:
        raise HTTPException(
            status_code=400, detail="name contains invalid path characters"
        )
    return clean_name


def read_image_bytes(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc
    return image


class EmbeddingStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.faces_dir = self.root / "faces"
        self.embeddings_dir = self.root / "embeddings"
        self.metadata_dir = self.root / "metadata"
        self.stats_path = self.metadata_dir / "stats.json"
        self._lock = Lock()
        self._centroids: Dict[str, np.ndarray] = {}
        self._identity_sample_counts: Dict[str, int] = {}
        self._identity_count = 0
        self._embedding_count = 0
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.refresh_cache()

    def _identity_dir(self, name: str) -> Path:
        return self.faces_dir / name

    def _embedding_path(self, name: str) -> Path:
        return self.embeddings_dir / f"{name}.npy"

    def save_face(self, name: str, image_bytes: bytes) -> Path:
        identity_dir = self._identity_dir(name)
        identity_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = identity_dir / f"{timestamp}.jpg"
        path.write_bytes(image_bytes)
        return path

    def append_embedding(self, name: str, embedding: np.ndarray) -> int:
        embedding_path = self._embedding_path(name)
        embedding = embedding.astype(np.float32)
        if embedding_path.exists():
            existing = np.load(embedding_path)
            if existing.ndim == 1:
                existing = existing[None, :]
            merged = np.concatenate([existing, embedding[None, :]], axis=0)
        else:
            merged = embedding[None, :]
        np.save(embedding_path, merged)
        self.refresh_cache()
        self.write_stats()
        return int(merged.shape[0])

    def _centroid_for_embeddings(self, embeddings: np.ndarray) -> np.ndarray | None:
        if embeddings.size == 0:
            return None
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]
        centroid = embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return centroid.astype(np.float32)

    def refresh_cache(self) -> None:
        centroids: Dict[str, np.ndarray] = {}
        identity_sample_counts: Dict[str, int] = {}
        identity_count = 0
        embedding_count = 0

        for embedding_path in sorted(self.embeddings_dir.glob("*.npy")):
            try:
                embeddings = np.load(embedding_path)
            except OSError:
                continue

            centroid = self._centroid_for_embeddings(embeddings)
            if centroid is None:
                continue

            if embeddings.ndim == 1:
                sample_count = 1
            elif embeddings.ndim == 2:
                sample_count = int(embeddings.shape[0])
            else:
                continue

            centroids[embedding_path.stem] = centroid
            identity_sample_counts[embedding_path.stem] = sample_count
            identity_count += 1
            embedding_count += sample_count

        with self._lock:
            self._centroids = centroids
            self._identity_sample_counts = identity_sample_counts
            self._identity_count = identity_count
            self._embedding_count = embedding_count

    def load_all_centroids(self) -> Dict[str, np.ndarray]:
        with self._lock:
            return {
                name: centroid.copy() for name, centroid in self._centroids.items()
            }

    def identity_count(self) -> int:
        with self._lock:
            return self._identity_count

    def embedding_count(self) -> int:
        with self._lock:
            return self._embedding_count

    def list_identities(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "name": name,
                    "sample_count": self._identity_sample_counts.get(name, 0),
                }
                for name in sorted(self._centroids)
            ]

    def delete_identity(self, name: str) -> bool:
        removed = False
        identity_dir = self._identity_dir(name)
        embedding_path = self._embedding_path(name)

        if identity_dir.exists():
            shutil.rmtree(identity_dir, ignore_errors=True)
            removed = True
        if embedding_path.exists():
            embedding_path.unlink(missing_ok=True)
            removed = True

        if removed:
            self.refresh_cache()
            self.write_stats()
        return removed

    def write_stats(self) -> None:
        with self._lock:
            payload = {
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "identity_count": self._identity_count,
                "embedding_count": self._embedding_count,
            }
        self.stats_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class FaceEmbeddingService:
    def __init__(self, runtime_root: Path | None = None) -> None:
        model_name = os.getenv("FACE_SERVICE_MODEL", "vggface2").strip() or "vggface2"
        self.device = choose_device()
        self.threshold = float(os.getenv("FACE_SERVICE_THRESHOLD", "0.72"))
        if runtime_root is None:
            runtime_root = Path(
                os.getenv("FACE_SERVICE_DATA_DIR", "server/runtime")
            ).resolve()
        else:
            runtime_root = Path(runtime_root).resolve()
        self.runtime_root = runtime_root
        self.store = EmbeddingStore(runtime_root)
        self.transform = transforms.Compose(
            [
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.model = InceptionResnetV1(pretrained=model_name).eval().to(self.device)
        self.model_name = model_name
        self.store.write_stats()

    def _read_image(self, image_bytes: bytes) -> Image.Image:
        return read_image_bytes(image_bytes)

    def embed_bytes(self, image_bytes: bytes) -> np.ndarray:
        image = self._read_image(image_bytes)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            embedding = self.model(tensor).detach().cpu().numpy()[0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def register(self, name: str, image_bytes: bytes) -> dict:
        clean_name = validate_identity_name(name)
        self.store.save_face(clean_name, image_bytes)
        embedding = self.embed_bytes(image_bytes)
        sample_count = self.store.append_embedding(clean_name, embedding)
        return {
            "name": clean_name,
            "sample_count": sample_count,
            "identity_count": self.store.identity_count(),
            "embedding_count": self.store.embedding_count(),
            "status": "ok",
        }

    def list_identities(self) -> dict:
        identities = self.store.list_identities()
        return {
            "identities": identities,
            "identity_count": self.store.identity_count(),
            "embedding_count": self.store.embedding_count(),
        }

    def delete_identity(self, name: str) -> dict:
        clean_name = validate_identity_name(name)
        removed = self.store.delete_identity(clean_name)
        return {
            "name": clean_name,
            "removed": removed,
            "identity_count": self.store.identity_count(),
            "embedding_count": self.store.embedding_count(),
            "status": "ok",
        }

    def recognize(self, image_bytes: bytes) -> dict:
        query = self.embed_bytes(image_bytes)
        centroids = self.store.load_all_centroids()
        identity_count = len(centroids)
        embedding_count = self.store.embedding_count()
        if not centroids:
            return {
                "name": "未知",
                "matched": False,
                "score": 0.0,
                "candidate_name": "未知",
                "candidate_score": 0.0,
                "threshold": self.threshold,
                "identity_count": identity_count,
                "embedding_count": embedding_count,
                "detail": "embedding library is empty",
            }

        best_name = "未知"
        best_score = -1.0
        for name, centroid in centroids.items():
            score = float(np.dot(query, centroid))
            if score > best_score:
                best_score = score
                best_name = name

        matched = best_score >= self.threshold
        return {
            "name": best_name if matched else "未知",
            "matched": matched,
            "score": round(best_score, 4),
            "candidate_name": best_name if best_score >= 0 else "未知",
            "candidate_score": round(best_score, 4) if best_score >= 0 else 0.0,
            "threshold": self.threshold,
            "identity_count": identity_count,
            "embedding_count": embedding_count,
        }


def create_app(
    runtime_root: Path | None = None,
    service: FaceEmbeddingService | None = None,
    mindspore_backend: MindSporeEmbeddingBackend | None = None,
    classifier_backend: ClassifierBackend | None = None,
) -> FastAPI:
    if runtime_root is None:
        candidate_root = getattr(service, "runtime_root", None)
        if candidate_root is None:
            runtime_root = Path(
                os.getenv("FACE_SERVICE_DATA_DIR", "server/runtime")
            ).resolve()
        else:
            runtime_root = Path(candidate_root).resolve()
    else:
        runtime_root = Path(runtime_root).resolve()

    assets = ModelArtifactPaths(runtime_root / "models")
    classifier_assets = ClassifierArtifactPaths(runtime_root / "classifier")
    app = FastAPI(title="Course Design Face Embedding Service")

    app.state.runtime_root = runtime_root
    app.state.service = service
    app.state.mindspore_backend = mindspore_backend
    app.state.mindspore_assets = assets
    app.state.classifier_backend = classifier_backend
    app.state.classifier_assets = classifier_assets

    def get_service() -> FaceEmbeddingService:
        if app.state.service is None:
            app.state.service = FaceEmbeddingService(runtime_root=runtime_root)
        return app.state.service

    def get_mindspore_backend() -> MindSporeEmbeddingBackend:
        if app.state.mindspore_backend is None:
            if app.state.service is not None:
                threshold = app.state.service.threshold
            else:
                threshold = float(os.getenv("FACE_SERVICE_THRESHOLD", "0.72"))
            app.state.mindspore_backend = MindSporeEmbeddingBackend(
                model_path=assets.mindir_path,
                threshold=threshold,
            )
        return app.state.mindspore_backend


    def get_classifier_backend() -> ClassifierBackend:
        if app.state.classifier_backend is None:
            if app.state.service is not None:
                threshold = app.state.service.threshold
            else:
                threshold = float(os.getenv("FACE_SERVICE_THRESHOLD", "0.72"))
            app.state.classifier_backend = ClassifierBackend(
                model_path=classifier_assets.mindir_path,
                label_map_path=classifier_assets.label_map_path,
                threshold=threshold,
            )
        return app.state.classifier_backend


    def classifier_is_ready(backend: ClassifierBackend | None) -> bool:
        if backend is None:
            return False
        backend_ready = getattr(backend, "is_ready", None)
        if callable(backend_ready):
            try:
                return bool(backend_ready())
            except Exception:
                return False
        model_path = Path(getattr(backend, "model_path", classifier_assets.mindir_path))
        label_map_path = Path(
            getattr(backend, "label_map_path", classifier_assets.label_map_path)
        )
        return model_path.exists() and label_map_path.exists()


    def classifier_class_count(label_map_path: Path) -> int:
        if not label_map_path.exists():
            return 0
        try:
            payload = json.loads(label_map_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0
        if isinstance(payload, dict):
            return len(payload)
        if isinstance(payload, list):
            return len(payload)
        return 0


    def classifier_dataset_ready(source_root: Path) -> bool:
        if not source_root.exists():
            return False
        for path in source_root.rglob("*"):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            if path.suffix.lower() in SUPPORTED_SUFFIXES:
                return True
        return False


    @app.get("/health")
    def health() -> dict:
        service = get_service()
        backend = get_mindspore_backend()
        ready = backend.is_ready()
        classifier_backend = get_classifier_backend()
        classifier_ready = classifier_is_ready(classifier_backend)
        label_map_path = classifier_assets.label_map_path
        return {
            "status": "ok",
            "active_backend": "mindspore_embedding"
            if ready
            else "pytorch_embedding",
            "mindspore_model_ready": ready,
            "mindspore_model_path": str(backend.model_path),
            "classifier_model_ready": classifier_ready,
            "classifier_model_path": str(classifier_assets.mindir_path),
            "classifier_label_map_path": str(label_map_path),
            "classifier_backend_available": classifier_backend is not None,
            "classifier_label_map_exists": label_map_path.exists(),
            "classifier_class_count": classifier_class_count(label_map_path),
            "device": service.device,
            "model": service.model_name,
            "threshold": service.threshold,
            "identity_count": service.store.identity_count(),
            "embedding_count": service.store.embedding_count(),
        }


    @app.get("/identities")
    def identities() -> dict:
        service = get_service()
        return service.list_identities()


    @app.post("/register")
    async def register(name: str = Form(...), photo: UploadFile = File(...)) -> dict:
        image_bytes = await photo.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="empty photo")
        service = get_service()
        return service.register(name=name, image_bytes=image_bytes)


    @app.post("/delete_identity")
    async def delete_identity(name: str = Form(...)) -> dict:
        service = get_service()
        return service.delete_identity(name=name)


    @app.post("/recognize")
    async def recognize(
        photo: UploadFile = File(...),
        backend: str = Form(""),
    ) -> dict:
        service = get_service()
        image_bytes = await photo.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="empty photo")
        read_image_bytes(image_bytes)
        backend_name = (backend or "").strip()
        if backend_name not in {"", "mindspore_embedding", "cnn_classifier"}:
            raise HTTPException(status_code=400, detail="invalid backend")
        if backend_name == "cnn_classifier":
            classifier_backend = get_classifier_backend()
            classifier_ready = classifier_is_ready(classifier_backend)
            identity_count = service.store.identity_count()
            embedding_count = service.store.embedding_count()
            if not classifier_ready:
                return {
                    "name": "未知",
                    "matched": False,
                    "score": 0.0,
                    "candidate_name": "未知",
                    "candidate_score": 0.0,
                    "threshold": service.threshold,
                    "identity_count": identity_count,
                    "embedding_count": embedding_count,
                    "backend": "cnn_classifier",
                    "ready": False,
                    "detail": "classifier model not ready",
                }
            try:
                payload = classifier_backend.predict_bytes(image_bytes)
            except Exception:
                return {
                    "name": "未知",
                    "matched": False,
                    "score": 0.0,
                    "candidate_name": "未知",
                    "candidate_score": 0.0,
                    "threshold": service.threshold,
                    "identity_count": identity_count,
                    "embedding_count": embedding_count,
                    "backend": "cnn_classifier",
                    "ready": False,
                    "detail": "classifier prediction failed",
                }
            payload["threshold"] = service.threshold
            payload["identity_count"] = identity_count
            payload["embedding_count"] = embedding_count
            payload["backend"] = payload.get("backend", "cnn_classifier")
            payload["ready"] = True
            return payload

        backend = get_mindspore_backend()
        centroids = service.store.load_all_centroids()
        identity_count = len(centroids)
        embedding_count = service.store.embedding_count()
        ready = backend.is_ready()
        if not centroids:
            return {
                "name": "未知",
                "matched": False,
                "score": 0.0,
                "candidate_name": "未知",
                "candidate_score": 0.0,
                "threshold": service.threshold,
                "identity_count": identity_count,
                "embedding_count": embedding_count,
                "backend": "mindspore_embedding" if ready else "pytorch_embedding",
                "ready": ready,
                "detail": "embedding library is empty",
            }

        if not ready:
            payload = service.recognize(image_bytes=image_bytes)
            payload["backend"] = "pytorch_embedding"
            payload["ready"] = False
            payload["detail"] = "mindspore model not ready, fallback active"
            return payload

        try:
            payload = backend.recognize_bytes(image_bytes, centroids)
        except MindSporeModelNotReady:
            payload = service.recognize(image_bytes=image_bytes)
            payload["backend"] = "pytorch_embedding"
            payload["ready"] = False
            payload["detail"] = "mindspore model not ready, fallback active"
            return payload

        payload["threshold"] = service.threshold
        payload["identity_count"] = identity_count
        payload["embedding_count"] = embedding_count
        payload["backend"] = payload.get("backend", "mindspore_embedding")
        payload["ready"] = True
        return payload


    @app.post("/train_classifier")
    def train_classifier_endpoint() -> dict:
        classifier_root = runtime_root / "classifier"
        source_root = runtime_root / "faces"
        if not classifier_dataset_ready(source_root):
            raise HTTPException(
                status_code=400, detail="faces is missing or empty"
            )
        try:
            dataset_result = build_classifier_dataset(
                source_root=source_root,
                output_root=classifier_root / "dataset",
                val_ratio=0.2,
            )
            weights_path, label_map_path = train_classifier(
                train_root=dataset_result.train_root,
                val_root=dataset_result.val_root,
                output_dir=classifier_root,
            )
            onnx_path = export_classifier(
                weights_path=weights_path,
                label_map_path=label_map_path,
                output_path=classifier_root / "classifier.onnx",
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"classifier training failed: {exc}"
            ) from exc
        model_path = classifier_assets.mindir_path
        return {
            "status": "ok",
            "backend": "cnn_classifier",
            "class_count": len(dataset_result.class_names),
            "train_root": str(dataset_result.train_root),
            "val_root": str(dataset_result.val_root),
            "weights_path": str(weights_path),
            "label_map_path": str(label_map_path),
            "onnx_path": str(onnx_path),
            "classifier_model_ready": model_path.exists(),
            "classifier_model_path": str(model_path),
            "classifier_label_map_path": str(label_map_path),
        }

    return app


app = create_app()
