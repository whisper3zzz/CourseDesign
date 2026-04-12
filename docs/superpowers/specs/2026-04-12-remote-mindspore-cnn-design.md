# Remote MindSpore Conversion Design

**Date:** 2026-04-12

**Goal:** Replace the current remote PyTorch-only deep-recognition path with a conversion-based MindSpore deployment flow built around the existing `InceptionResnetV1(pretrained="vggface2")` model, while keeping the current embedding service as a fallback.

## Context

The current remote service in `server/embedding_service.py` uses `facenet-pytorch`:

- model: `InceptionResnetV1(pretrained="vggface2")`
- input shape: `160x160 RGB`
- output: 512-dimensional embedding
- matching strategy: cosine/centroid-style similarity against stored embeddings

That model is already integrated into the desktop and remote workflow, so it is the lowest-risk candidate for a first conversion pass.

The earlier plan in this branch aimed to build a native MindSpore classifier and training service. That is no longer the preferred direction. After server-side validation, the new direction is:

1. export the existing PyTorch model to ONNX;
2. convert the ONNX model on the SSH server with MindSpore Lite tooling;
3. use the converted artifact for server-side inference;
4. keep the current PyTorch embedding backend as fallback until parity is proven.

## Verified Constraints

### SSH Server Facts

The configured server from `scripts/start_with_ssh_tunnel.local.sh` has been verified as:

- `Ubuntu 24.04.4 LTS`
- `x86_64`
- `Python 3.12.3`
- WSL2-based Linux environment

`mindspore==2.8.0` has already been installed successfully in an isolated server-side environment at `~/mscheck/.venv-ms` and imported successfully.

### MindSpore Lite Tooling Facts

The local package `/Users/whisper/Downloads/mindspore-lite-2.8.0-linux-x64.tar.gz` has been uploaded and unpacked on the server. It contains working CLI tooling:

- `tools/converter/converter/converter_lite`
- `tools/benchmark/benchmark`
- runtime libraries under `runtime/lib/`

After setting `LD_LIBRARY_PATH`, both CLI tools run on the SSH server.

### Important Packaging Decision

The Python `mindspore_lite` wheel path is not the primary integration target.

Reasons:

- the official direct wheel URL returned `403 AccessDenied` during validation;
- public package indexes only exposed obsolete incompatible builds;
- the bundled Linux x64 package already provides the converter and benchmark tooling we need.

Therefore:

- **MindSpore Lite is used as the conversion and CLI validation toolchain**
- **server inference should target the installed `mindspore` Python runtime first**

This keeps the runtime path inside a server environment that is already proven to work.

## Approved Technical Direction

### Selected Model

The first conversion target is the exact model the service already uses:

- `facenet-pytorch` `InceptionResnetV1(pretrained="vggface2")`

This avoids changing:

- preprocessing size and normalization
- output semantics
- embedding dimensionality
- recognition thresholds all at once

### Runtime Strategy

The deployment flow becomes:

1. local export:
   - load the existing PyTorch model
   - export to `ONNX`
2. remote conversion:
   - upload `ONNX` model to SSH server
   - run `converter_lite` to produce `MindIR`
3. remote validation:
   - run `benchmark` against the converted artifact to confirm the model is usable
4. service integration:
   - load the converted `MindIR` through the installed `mindspore` Python runtime
   - compute embeddings in the remote service
5. fallback:
   - if MindSpore inference is unavailable or mismatched, keep the current PyTorch backend active

### Scope Boundary

The converted model path is **inference-only** for this phase.

This phase does **not** include:

- native MindSpore training
- classifier-style `/train` workflows
- label-map generation
- replacement of the embedding matching design

The output remains an embedding-based recognition service. The change is the inference engine, not the recognition semantics.

## Architecture

### Remote Backends

The service should converge on two inference backends:

1. `pytorch_embedding`
   - current implementation
   - fallback and reference backend
2. `mindspore_embedding`
   - same input contract and embedding output shape
   - powered by converted `MindIR`

Both backends must expose the same interface:

- `embed_bytes(image_bytes: bytes) -> np.ndarray`
- `recognize(image_bytes: bytes) -> dict`
- `health_payload() -> dict`

### Artifact Layout

The converted-model artifacts should live under `server/runtime/models/`:

```text
server/runtime/models/
├── facenet_vggface2.onnx
├── facenet_vggface2.mindir
└── facenet_vggface2.meta.json
```

The metadata file should record:

- source model name
- export input shape
- output dimension
- conversion timestamp
- remote converter command used
- benchmark success status

### Conversion Command Boundary

The converter should be driven by a repository script rather than manually retyped shell:

- local script exports ONNX
- remote script or documented command converts ONNX to MindIR
- remote script or documented command benchmarks the converted model

That keeps the process reproducible and usable after this session.

## API Behavior

The public HTTP contract should stay stable for the desktop:

- `POST /register`
- `POST /recognize`
- `GET /health`
- `GET /identities`
- `POST /delete_identity`

New training endpoints are no longer part of this phase.

### `POST /recognize`

Behavior becomes:

- prefer the selected active inference backend;
- default to `mindspore_embedding` only after conversion validation passes;
- otherwise fall back to `pytorch_embedding`.

The response should include:

- `backend`
- `ready`
- `score`
- `detail`

This is important so the desktop can tell whether the server is actually using the converted model.

### `GET /health`

The health payload should include:

- `active_backend`
- `available_backends`
- `mindspore_model_ready`
- `mindspore_model_path`
- `identity_count`
- `embedding_count`

## Validation Requirements

Before switching the remote service default backend, all of the following must be true:

1. ONNX export succeeds locally
2. uploaded ONNX converts successfully on the SSH server
3. `benchmark` can load the converted model without runtime errors
4. server-side MindSpore inference returns a 512-dimensional embedding
5. a small set of same-image comparisons show acceptable numerical agreement with the current PyTorch backend

The MindSpore backend should not become default until those checks pass.

## Desktop Impact

Desktop changes should be minimal in this phase.

The deep-recognition mode can keep the same UI wording. The only user-visible change should be more precise diagnostics when the remote service reports which backend is active.

No new desktop training button or explicit remote training flow is needed anymore.

## Non-Goals

- native MindSpore training service
- MindSpore classifier outputs
- replacing the embedding matching logic
- exposing backend switching in the desktop UI before parity is validated

## Final Design Decision

The approved first implementation path is:

- use `InceptionResnetV1(pretrained="vggface2")`
- export ONNX locally
- convert on the SSH server using the uploaded MindSpore Lite Linux x64 package
- validate with `benchmark`
- integrate converted-model inference into the remote service through the already working `mindspore` Python runtime
- keep the current PyTorch service path as fallback
