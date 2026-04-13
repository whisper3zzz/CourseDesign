# Remote MindSpore Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the existing remote `InceptionResnetV1(vggface2)` embedding model to a MindSpore deployment artifact, validate it on the SSH server, and integrate it into the remote service with PyTorch fallback.

**Architecture:** Keep the current HTTP contract and embedding matching logic. Add a reproducible export-and-convert toolchain around the existing model, validate the converted artifact on the SSH server with MindSpore Lite CLI tools, then add a new `mindspore_embedding` inference backend to the FastAPI service while retaining the current PyTorch backend as fallback.

**Tech Stack:** Python 3.10/3.12, PyTorch, ONNX, MindSpore 2.8.0, MindSpore Lite converter/benchmark CLI, FastAPI, requests, `uv`, `pytest`

---

## File Structure

- `scripts/export_facenet_to_onnx.py`
  Local export entrypoint for the exact `InceptionResnetV1(vggface2)` model already used in the service.
- `scripts/convert_facenet_on_remote.sh`
  Remote-side helper to run `converter_lite` and `benchmark` against the uploaded ONNX model and emitted MindIR.
- `server/model_assets.py`
  Metadata helpers for ONNX/MindIR artifact paths and readiness checks.
- `server/embedding_backend.py`
  Keep the current PyTorch embedding backend isolated and reusable as fallback.
- `server/mindspore_backend.py`
  New MindSpore embedding backend that loads converted MindIR through the installed `mindspore` runtime.
- `server/embedding_service.py`
  Service shell that chooses active backend and exposes backend status in `/health` and `/recognize`.
- `server/README.md`
  Document the conversion workflow and runtime prerequisites.
- `tests/scripts/test_export_facenet_to_onnx.py`
  Cover local ONNX export contract.
- `tests/server/test_model_assets.py`
  Cover model artifact discovery and readiness metadata.
- `tests/server/test_mindspore_backend.py`
  Cover converted-model inference loading and fallback behavior.
- `tests/server/test_api.py`
  Cover backend reporting and fallback semantics.

### Task 1: Export The Existing FaceNet Model To ONNX

**Files:**
- Create: `scripts/export_facenet_to_onnx.py`
- Create: `tests/scripts/test_export_facenet_to_onnx.py`

- [ ] **Step 1: Write the failing export test**

```python
# tests/scripts/test_export_facenet_to_onnx.py
from pathlib import Path

from scripts.export_facenet_to_onnx import export_model


def test_export_model_writes_onnx_file(tmp_path: Path) -> None:
    output_path = tmp_path / "facenet_vggface2.onnx"

    export_model(output_path=output_path, opset_version=17)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
TORCH_HOME=/tmp/torch_cache PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python uv run --with onnx --with pytest pytest tests/scripts/test_export_facenet_to_onnx.py -q
```

Expected: FAIL because `scripts.export_facenet_to_onnx` does not exist yet

- [ ] **Step 3: Write the minimal export implementation**

```python
# scripts/export_facenet_to_onnx.py
from __future__ import annotations

from pathlib import Path

import torch
from facenet_pytorch import InceptionResnetV1


def export_model(output_path: Path, opset_version: int = 17) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = InceptionResnetV1(pretrained="vggface2").eval()
    sample = torch.randn(1, 3, 160, 160, dtype=torch.float32)

    torch.onnx.export(
        model,
        sample,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    )
    return output_path


if __name__ == "__main__":
    export_model(Path("server/runtime/models/facenet_vggface2.onnx"))
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
TORCH_HOME=/tmp/torch_cache PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python uv run --with onnx --with pytest pytest tests/scripts/test_export_facenet_to_onnx.py -q
```

Expected: PASS with `1 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/export_facenet_to_onnx.py tests/scripts/test_export_facenet_to_onnx.py
git commit -m "feat: add facenet onnx export script"
```

### Task 2: Add Artifact Metadata And Remote Conversion Scripts

**Files:**
- Create: `server/model_assets.py`
- Create: `scripts/convert_facenet_on_remote.sh`
- Create: `tests/server/test_model_assets.py`

- [ ] **Step 1: Write the failing artifact metadata tests**

```python
# tests/server/test_model_assets.py
from pathlib import Path

from server.model_assets import ModelArtifactPaths


def test_artifact_paths_point_into_runtime_models(tmp_path: Path) -> None:
    paths = ModelArtifactPaths(root=tmp_path)

    assert paths.onnx_path == tmp_path / "facenet_vggface2.onnx"
    assert paths.mindir_path == tmp_path / "facenet_vggface2.mindir"
    assert paths.meta_path == tmp_path / "facenet_vggface2.meta.json"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python uv run --with pytest pytest tests/server/test_model_assets.py -q
```

Expected: FAIL because `server.model_assets` does not exist yet

- [ ] **Step 3: Write the metadata helper and remote conversion script**

```python
# server/model_assets.py
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
```

```bash
# scripts/convert_facenet_on_remote.sh
#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
LITE_ROOT="${LITE_ROOT:?set LITE_ROOT to the unpacked mindspore-lite directory}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/server/runtime/models}"
ONNX_PATH="${MODEL_DIR}/facenet_vggface2.onnx"
MINDIR_PATH="${MODEL_DIR}/facenet_vggface2.mindir"
CONVERTER_BIN="${LITE_ROOT}/tools/converter/converter/converter_lite"
BENCHMARK_BIN="${LITE_ROOT}/tools/benchmark/benchmark"

mkdir -p "${MODEL_DIR}"
die() {
  printf '%s\n' "$1" >&2
  exit 1
}

[ -d "${LITE_ROOT}" ] || die "missing LITE_ROOT: ${LITE_ROOT}"
[ -x "${CONVERTER_BIN}" ] || die "missing converter binary: ${CONVERTER_BIN}"
[ -x "${BENCHMARK_BIN}" ] || die "missing benchmark binary: ${BENCHMARK_BIN}"
[ -f "${ONNX_PATH}" ] || die "missing ONNX model: ${ONNX_PATH}"

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="${LITE_ROOT}/tools/converter/lib:${LITE_ROOT}/runtime/lib:${LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="${LITE_ROOT}/tools/converter/lib:${LITE_ROOT}/runtime/lib"
fi

"${CONVERTER_BIN}" \
  --fmk=ONNX \
  --modelFile="${ONNX_PATH}" \
  --outputFile="${MODEL_DIR}/facenet_vggface2" \
  --saveType=MINDIR

[ -f "${MINDIR_PATH}" ] || die "conversion did not produce: ${MINDIR_PATH}"

"${BENCHMARK_BIN}" \
  --modelFile="${MINDIR_PATH}" \
  --modelType=MindIR \
  --device=CPU \
  --loopCount=1 \
  --warmUpLoopCount=0
```

- [ ] **Step 4: Run the artifact test to verify it passes**

Run:

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python uv run --with pytest pytest tests/server/test_model_assets.py -q
```

Expected: PASS with `1 passed`

- [ ] **Step 5: Commit**

```bash
git add server/model_assets.py scripts/convert_facenet_on_remote.sh tests/server/test_model_assets.py
git commit -m "feat: add converted model artifact helpers"
```

### Task 3: Add MindSpore Embedding Inference Backend

**Files:**
- Create: `server/mindspore_backend.py`
- Create: `tests/server/test_mindspore_backend.py`
- Modify: `server/embedding_backend.py`

- [ ] **Step 1: Write the failing MindSpore backend tests**

```python
# tests/server/test_mindspore_backend.py
import server.mindspore_backend as backend_module
from pathlib import Path

import numpy as np
import pytest

from server.mindspore_backend import MindSporeEmbeddingBackend, MindSporeModelNotReady


def test_embed_bytes_raises_when_mindir_is_missing(tmp_path: Path) -> None:
    backend = MindSporeEmbeddingBackend(model_path=tmp_path / "missing.mindir")

    with pytest.raises(MindSporeModelNotReady):
        backend.embed_bytes(b"fake-image")


def test_embed_bytes_raises_when_mindspore_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "facenet_vggface2.mindir"
    model_path.write_bytes(b"mindir")
    monkeypatch.setattr(backend_module, "ms", None)

    backend = MindSporeEmbeddingBackend(model_path=model_path)

    with pytest.raises(MindSporeModelNotReady):
        backend.embed_bytes(b"fake-image")


def test_recognize_returns_unknown_when_score_is_below_threshold(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "facenet_vggface2.mindir"
    model_path.write_bytes(b"mindir")

    backend = MindSporeEmbeddingBackend(model_path=model_path, threshold=0.72)
    monkeypatch.setattr(
        backend,
        "embed_bytes",
        lambda image_bytes: np.array([1.0, 0.0], dtype=np.float32),
    )

    payload = backend.recognize_bytes(
        b"fake-image", {"alice": np.array([0.0, 1.0], dtype=np.float32)}
    )

    assert payload["name"] == "未知"
    assert payload["matched"] is False
    assert payload["backend"] == "mindspore_embedding"


def test_recognize_returns_name_when_score_meets_threshold(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "facenet_vggface2.mindir"
    model_path.write_bytes(b"mindir")

    backend = MindSporeEmbeddingBackend(model_path=model_path, threshold=0.72)
    monkeypatch.setattr(
        backend,
        "embed_bytes",
        lambda image_bytes: np.array([1.0, 0.0], dtype=np.float32),
    )

    payload = backend.recognize_bytes(
        b"fake-image", {"alice": np.array([1.0, 0.0], dtype=np.float32)}
    )

    assert payload["name"] == "alice"
    assert payload["matched"] is True
    assert payload["backend"] == "mindspore_embedding"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python uv run --with pytest pytest tests/server/test_mindspore_backend.py -q
```

Expected: FAIL because `server.mindspore_backend` does not exist yet

- [ ] **Step 3: Write the minimal MindSpore backend implementation**

```python
# server/mindspore_backend.py
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import mindspore as ms
except ImportError:
    ms = None


class MindSporeModelNotReady(RuntimeError):
    pass


class MindSporeEmbeddingBackend:
    def __init__(self, model_path: Path, threshold: float = 0.72) -> None:
        self.model_path = Path(model_path)
        self.threshold = float(threshold)
        self._graph = None

    def _load_graph(self):
        if ms is None:
            raise MindSporeModelNotReady("mindspore is unavailable")
        if not self.model_path.exists():
            raise MindSporeModelNotReady(f"missing model: {self.model_path}")
        if self._graph is None:
            graph = ms.load(str(self.model_path))
            self._graph = ms.nn.GraphCell(graph=graph)
        return self._graph

    def embed_bytes(self, image_bytes: bytes) -> np.ndarray:
        graph = self._load_graph()
        image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((160, 160))
        array = np.asarray(image, dtype=np.float32)
        array = ((array / 255.0) - 0.5) / 0.5
        array = np.transpose(array, (2, 0, 1))[None, ...]
        tensor = ms.Tensor(array, ms.float32)
        embedding = graph(tensor).asnumpy()[0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def recognize_bytes(self, image_bytes: bytes, centroids: dict[str, np.ndarray]) -> dict:
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
            "backend": "mindspore_embedding",
            "ready": True,
        }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python uv run --with pytest python -m pytest tests/server/test_mindspore_backend.py -q
```

Expected: PASS with `4 passed`

- [ ] **Step 5: Commit**

```bash
git add server/mindspore_backend.py tests/server/test_mindspore_backend.py
git commit -m "feat: add mindspore embedding backend"
```

### Task 4: Integrate Backend Selection Into The Service

**Files:**
- Modify: `server/embedding_service.py`
- Create: `tests/server/test_api.py`
- Modify: `server/README.md`

- [ ] **Step 1: Write the failing API test**

```python
# tests/server/test_api.py
from fastapi.testclient import TestClient

from server.embedding_service import create_app


def test_health_reports_active_backend(tmp_path) -> None:
    app = create_app(runtime_root=tmp_path)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert "active_backend" in response.json()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python uv run --with pytest pytest tests/server/test_api.py -q
```

Expected: FAIL because the app factory and backend reporting are not implemented yet

- [ ] **Step 3: Integrate the MindSpore backend with PyTorch fallback**

```python
# server/embedding_service.py
from server.embedding_backend import EmbeddingBackend
from server.mindspore_backend import MindSporeEmbeddingBackend, MindSporeModelNotReady
from server.model_assets import ModelArtifactPaths


def create_app(runtime_root: Path | None = None) -> FastAPI:
    runtime_root = Path(runtime_root or os.getenv("FACE_SERVICE_DATA_DIR", "server/runtime")).resolve()
    embedding_backend = EmbeddingBackend(runtime_root=runtime_root)
    assets = ModelArtifactPaths(runtime_root / "models")
    mindspore_backend = MindSporeEmbeddingBackend(model_path=assets.mindir_path)
    app = FastAPI(title="Course Design Face Service")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "active_backend": "mindspore_embedding" if assets.mindir_path.exists() else "pytorch_embedding",
            "mindspore_model_ready": assets.mindir_path.exists(),
            "mindspore_model_path": str(assets.mindir_path),
            "identity_count": embedding_backend.identity_count(),
            "embedding_count": embedding_backend.embedding_count(),
        }

    @app.post("/recognize")
    async def recognize(photo: UploadFile = File(...)) -> dict:
        image_bytes = await photo.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="empty photo")
        try:
            centroids = embedding_backend.load_all_centroids()
            return mindspore_backend.recognize_bytes(image_bytes, centroids)
        except MindSporeModelNotReady:
            payload = embedding_backend.recognize(image_bytes=image_bytes)
            payload["backend"] = "pytorch_embedding"
            payload["detail"] = "mindspore model not ready, fallback active"
            return payload

    return app
```

- [ ] **Step 4: Run the API test to verify it passes**

Run:

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python uv run --with pytest pytest tests/server/test_api.py -q
```

Expected: PASS with `1 passed`

- [ ] **Step 5: Update the server README**

```md
# server/README.md
- primary conversion target: `InceptionResnetV1(vggface2)`
- local export format: ONNX
- remote conversion tool: bundled `converter_lite`
- remote validation tool: bundled `benchmark`
- runtime behavior: prefer converted MindSpore model, fall back to PyTorch embedding backend
```

- [ ] **Step 6: Commit**

```bash
git add server/embedding_service.py server/README.md tests/server/test_api.py
git commit -m "feat: integrate converted mindspore backend into service"
```

### Task 5: Run End-To-End Conversion Validation

**Files:**
- Modify: `server/runtime/models/` artifacts created by scripts only

- [ ] **Step 1: Export the ONNX model locally**

Run:

```bash
TORCH_HOME=/tmp/torch_cache PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python uv run --with onnx python scripts/export_facenet_to_onnx.py
```

Expected: `server/runtime/models/facenet_vggface2.onnx` exists locally

- [ ] **Step 2: Upload the ONNX model to the SSH server**

Run:

```bash
scp -P 54316 server/runtime/models/facenet_vggface2.onnx whisper@ada4e161d491.ofalias.net:~/mscheck/
```

Expected: remote copy succeeds

- [ ] **Step 3: Convert and benchmark on the server**

Run on the server:

```bash
export LITE_ROOT=~/mscheck/litepkg/mindspore-lite-2.8.0-linux-x64
export LD_LIBRARY_PATH="$LITE_ROOT/tools/converter/lib:$LITE_ROOT/runtime/lib:$LD_LIBRARY_PATH"
mkdir -p ~/mscheck/models
mv ~/mscheck/facenet_vggface2.onnx ~/mscheck/models/
MODEL_DIR=~/mscheck/models "$PWD/scripts/convert_facenet_on_remote.sh"
```

Expected:

- `facenet_vggface2.mindir` is created
- `benchmark` exits successfully

- [ ] **Step 4: Run the service verification suite**

Run:

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python uv run --with onnx --with pytest pytest tests/scripts/test_export_facenet_to_onnx.py tests/server/test_model_assets.py tests/server/test_mindspore_backend.py tests/server/test_api.py -q
./.venv/bin/python -m compileall server scripts
```

Expected:

- all listed tests PASS
- compileall completes without syntax errors

- [ ] **Step 5: Commit**

```bash
git add scripts/export_facenet_to_onnx.py scripts/convert_facenet_on_remote.sh server/model_assets.py server/mindspore_backend.py server/embedding_service.py tests/scripts/test_export_facenet_to_onnx.py tests/server/test_model_assets.py tests/server/test_mindspore_backend.py tests/server/test_api.py server/README.md
git commit -m "feat: add conversion-based mindspore deployment path"
```

## Self-Review

- Spec coverage:
  - selected model and ONNX export path: Task 1
  - remote conversion and validation with uploaded Lite package: Task 2 and Task 5
  - MindSpore runtime inference path with PyTorch fallback: Task 3 and Task 4
  - stable HTTP contract and backend reporting: Task 4
- Placeholder scan:
  - no `TBD`, `TODO`, or “implement later” placeholders remain
- Type consistency:
  - backend names are always `mindspore_embedding` and `pytorch_embedding`
  - artifact names are always `facenet_vggface2.onnx` and `facenet_vggface2.mindir`
