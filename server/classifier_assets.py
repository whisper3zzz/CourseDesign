from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClassifierArtifactPaths:
    root: Path

    @property
    def onnx_path(self) -> Path:
        return self.root / "classifier.onnx"

    @property
    def mindir_path(self) -> Path:
        return self.root / "classifier.mindir"

    @property
    def label_map_path(self) -> Path:
        return self.root / "label_map.json"

    @property
    def metadata_path(self) -> Path:
        return self.root / "metadata.json"
