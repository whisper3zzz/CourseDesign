from __future__ import annotations

import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms


def choose_device() -> str:
    requested = os.getenv("FACE_SERVICE_DEVICE", "").strip().lower()
    if requested in {"cpu", "cuda"}:
        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.faces_dir = self.root / "faces"
        self.embeddings_dir = self.root / "embeddings"
        self.metadata_dir = self.root / "metadata"
        self.stats_path = self.metadata_dir / "stats.json"
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

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
            merged = np.concatenate([existing, embedding[None, :]], axis=0)
        else:
            merged = embedding[None, :]
        np.save(embedding_path, merged)
        self.write_stats()
        return int(merged.shape[0])

    def load_all_centroids(self) -> Dict[str, np.ndarray]:
        centroids: Dict[str, np.ndarray] = {}
        for embedding_path in sorted(self.embeddings_dir.glob("*.npy")):
            try:
                embeddings = np.load(embedding_path)
            except OSError:
                continue
            if embeddings.size == 0:
                continue
            centroid = embeddings.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids[embedding_path.stem] = centroid.astype(np.float32)
        return centroids

    def identity_count(self) -> int:
        return len(list(self.embeddings_dir.glob("*.npy")))

    def embedding_count(self) -> int:
        total = 0
        for embedding_path in self.embeddings_dir.glob("*.npy"):
            try:
                embeddings = np.load(embedding_path)
            except OSError:
                continue
            if embeddings.ndim == 2:
                total += int(embeddings.shape[0])
        return total

    def write_stats(self) -> None:
        payload = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "identity_count": self.identity_count(),
            "embedding_count": self.embedding_count(),
        }
        self.stats_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class FaceEmbeddingService:
    def __init__(self) -> None:
        model_name = os.getenv("FACE_SERVICE_MODEL", "vggface2").strip() or "vggface2"
        self.device = choose_device()
        self.threshold = float(os.getenv("FACE_SERVICE_THRESHOLD", "0.72"))
        self.store = EmbeddingStore(
            Path(os.getenv("FACE_SERVICE_DATA_DIR", "server/runtime")).resolve()
        )
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
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc
        return image

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
        clean_name = name.strip()
        if not clean_name:
            raise HTTPException(status_code=400, detail="name is empty")
        if "/" in clean_name or "\\" in clean_name:
            raise HTTPException(status_code=400, detail="name contains invalid path characters")
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
            "threshold": self.threshold,
            "identity_count": identity_count,
            "embedding_count": embedding_count,
        }


service = FaceEmbeddingService()
app = FastAPI(title="Course Design Face Embedding Service")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": service.device,
        "model": service.model_name,
        "threshold": service.threshold,
        "identity_count": service.store.identity_count(),
        "embedding_count": service.store.embedding_count(),
    }


@app.post("/register")
async def register(name: str = Form(...), photo: UploadFile = File(...)) -> dict:
    image_bytes = await photo.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty photo")
    return service.register(name=name, image_bytes=image_bytes)


@app.post("/recognize")
async def recognize(photo: UploadFile = File(...)) -> dict:
    image_bytes = await photo.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty photo")
    return service.recognize(image_bytes=image_bytes)
