from __future__ import annotations

from io import BytesIO
from pathlib import Path

import mindspore as ms
import numpy as np
from PIL import Image


class MindSporeModelNotReady(RuntimeError):
    pass


class MindSporeEmbeddingBackend:
    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self._graph = None

    def _load_graph(self):
        if not self.model_path.exists():
            raise MindSporeModelNotReady(f"missing model: {self.model_path}")
        if self._graph is None:
            graph = ms.load(str(self.model_path))
            self._graph = ms.nn.GraphCell(graph=graph)
        return self._graph

    def embed_bytes(self, image_bytes: bytes) -> np.ndarray:
        graph = self._load_graph()
        image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((160, 160))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))[None, ...]
        tensor = ms.Tensor(array, ms.float32)
        embedding = graph(tensor).asnumpy()[0]
        return embedding.astype(np.float32)

    def recognize_bytes(
        self, image_bytes: bytes, centroids: dict[str, np.ndarray]
    ) -> dict:
        query = self.embed_bytes(image_bytes)
        best_name = "未知"
        best_score = -1.0
        for name, centroid in centroids.items():
            score = float(np.dot(query, centroid))
            if score > best_score:
                best_name = name
                best_score = score
        return {
            "name": best_name,
            "matched": best_score >= 0.72,
            "score": round(best_score, 4),
            "backend": "mindspore_embedding",
            "ready": True,
        }
