from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import mindspore_lite as mslite
except ImportError:
    mslite = None

class MindSporeModelNotReady(RuntimeError):
    pass


class MindSporeEmbeddingBackend:
    def __init__(self, model_path: Path, threshold: float = 0.72) -> None:
        self.model_path = Path(model_path)
        self.threshold = float(threshold)
        self._model = None

    def _load_model(self):
        if mslite is None:
            raise MindSporeModelNotReady("mindspore_lite is unavailable")
        if not self.model_path.exists():
            raise MindSporeModelNotReady(f"missing model: {self.model_path}")
        if self._model is None:
            try:
                context = mslite.Context()
                context.target = ["cpu"]
                model = mslite.Model()
                model.build_from_file(
                    str(self.model_path), mslite.ModelType.MINDIR, context
                )
                self._model = model
            except Exception as exc:
                raise MindSporeModelNotReady(
                    f"failed to load mindspore_lite model: {exc}"
                ) from exc
        return self._model

    def is_ready(self) -> bool:
        try:
            self._load_model()
        except MindSporeModelNotReady:
            return False
        return True

    def embed_bytes(self, image_bytes: bytes) -> np.ndarray:
        model = self._load_model()
        image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((160, 160))
        array = np.asarray(image, dtype=np.float32)
        array = ((array / 255.0) - 0.5) / 0.5
        array = np.transpose(array, (2, 0, 1))[None, ...]

        inputs = model.get_inputs()
        model.resize(inputs, [[1, 3, 160, 160]])
        input_tensor = model.get_inputs()[0]
        input_tensor.set_data_from_numpy(array.astype(np.float32))

        outputs = model.predict([input_tensor])
        if not outputs:
            raise MindSporeModelNotReady("mindspore_lite returned no outputs")
        embedding = outputs[0].get_data_to_numpy()
        if not isinstance(embedding, np.ndarray):
            raise MindSporeModelNotReady("mindspore_lite output is not a numpy array")
        if embedding.shape != (1, 512):
            raise MindSporeModelNotReady(
                f"unexpected embedding shape: {embedding.shape}"
            )
        embedding = embedding[0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
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
        matched = best_score >= self.threshold
        return {
            "name": best_name if matched else "未知",
            "matched": matched,
            "score": round(best_score, 4),
            "candidate_name": best_name if best_score >= 0 else "未知",
            "candidate_score": round(best_score, 4) if best_score >= 0 else 0.0,
            "backend": "mindspore_embedding",
            "ready": True,
        }
