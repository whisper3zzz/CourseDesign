from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import mindspore_lite as mslite
except ImportError:
    mslite = None


class ClassifierModelNotReady(RuntimeError):
    pass


class ClassifierBackend:
    def __init__(
        self,
        model_path: Path,
        label_map_path: Path,
        threshold: float = 0.72,
    ) -> None:
        self.model_path = Path(model_path)
        self.label_map_path = Path(label_map_path)
        self.threshold = float(threshold)
        self._model = None
        self._label_map = None

    def _load_model(self):
        if mslite is None:
            raise ClassifierModelNotReady("mindspore_lite is unavailable")
        if not self.model_path.exists():
            raise ClassifierModelNotReady(f"missing model: {self.model_path}")
        if not self.label_map_path.exists():
            raise ClassifierModelNotReady(f"missing label map: {self.label_map_path}")
        if self._model is None:
            context = mslite.Context()
            context.target = ["cpu"]
            model = mslite.Model()
            model.build_from_file(
                str(self.model_path),
                mslite.ModelType.MINDIR,
                context,
            )
            self._model = model
            self._label_map = json.loads(
                self.label_map_path.read_text(encoding="utf-8")
            )
        return self._model, self._label_map

    def predict_bytes(self, image_bytes: bytes) -> dict:
        model, label_map = self._load_model()
        image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((160, 160))
        array = np.asarray(image, dtype=np.float32)
        array = ((array / 255.0) - 0.5) / 0.5
        array = np.transpose(array, (2, 0, 1))[None, ...]
        inputs = model.get_inputs()
        model.resize(inputs, [[1, 3, 160, 160]])
        tensor = model.get_inputs()[0]
        tensor.set_data_from_numpy(array.astype(np.float32))
        outputs = model.predict([tensor])
        logits = outputs[0].get_data_to_numpy()
        logits = logits.reshape(1, -1)[0]
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / np.sum(probs)
        class_index = int(np.argmax(probs))
        score = float(probs[class_index])
        candidate_name = label_map[str(class_index)]
        matched = score >= self.threshold
        return {
            "name": candidate_name if matched else "未知",
            "matched": matched,
            "score": round(score, 4),
            "candidate_name": candidate_name,
            "candidate_score": round(score, 4),
            "backend": "cnn_classifier",
            "ready": True,
        }
