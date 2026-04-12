from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelArtifactPaths:
    root: Path

    @property
    def onnx_path(self) -> Path:
        return self.root / "facenet_vggface2.onnx"

    @property
    def mindir_path(self) -> Path:
        return self.root / "facenet_vggface2.mindir"

    @property
    def meta_path(self) -> Path:
        return self.root / "facenet_vggface2.meta.json"
