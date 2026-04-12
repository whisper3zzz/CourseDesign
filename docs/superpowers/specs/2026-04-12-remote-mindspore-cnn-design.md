# Remote MindSpore CNN Design

**Date:** 2026-04-12

**Goal:** Replace the current remote deep-recognition primary path with an explicit-train MindSpore CNN classifier while keeping the existing embedding service as a fallback backend.

## Context

The current desktop app talks to the remote service through a small HTTP contract:

- `POST /register`
- `POST /recognize`
- `GET /health`
- `GET /identities`
- `POST /delete_identity`

Today the remote service in `server/embedding_service.py` is an always-online embedding pipeline based on `facenet-pytorch`. The desktop "deep recognition" mode does not trigger a remote training step. It uploads samples through `/register` and immediately uses `/recognize`.

The target behavior changes that model:

- MindSpore CNN becomes the primary remote recognition backend.
- Training becomes explicit through a remote `/train` endpoint.
- Successful training automatically activates the new MindSpore model.
- The existing embedding backend remains available as a fallback backend.

## Product Decisions

### Approved Interaction Model

- `POST /register` stores remote samples, keeps the fallback embedding backend updated, and marks the MindSpore model as dirty.
- Remote MindSpore training is explicit.
- Training success automatically activates the newly trained MindSpore model.
- If the active backend is not ready, the API returns explicit status instead of silently pretending recognition succeeded.
- The current desktop button used to inspect remote deep-service status is expanded to also act as the remote training entrypoint.

### Recognition Strategy

The MindSpore primary backend is a CNN classifier, not a CNN feature extractor plus distance matching.

That means:

- the remote model predicts one of the known identities from the current sample set;
- adding or removing identities requires retraining before the MindSpore backend becomes current again;
- the existing embedding backend remains available when classifier training is not desired or not currently ready.

This is intentionally a closed-set recognition path for the primary backend.

## Architecture

### High-Level Service Shape

`server/embedding_service.py` evolves from a single embedding implementation into a unified service shell that owns:

- API routing
- runtime state
- sample storage
- backend activation state
- model status reporting

The service internally coordinates two backend implementations:

1. `embedding`
   The current `facenet-pytorch` + centroid matching implementation. It stays operational and can still be activated.
2. `mindspore_cnn`
   A new explicit-train classifier backend based on MindSpore. It trains from the current sample directory and serves inference only after a successful model activation.

### Runtime Layout

The runtime directory remains under `server/runtime/` and gains explicit model-state artifacts:

```text
server/runtime/
├── faces/
│   └── <identity>/
├── embeddings/
│   └── <identity>.npy
├── metadata/
│   ├── stats.json
│   ├── backend_state.json
│   └── mindspore_label_map.json
└── models/
    └── mindspore_active.ckpt
```

Notes:

- `faces/` stays the shared source of truth for both remote backends.
- `embeddings/` stays owned by the current embedding backend.
- `models/mindspore_active.ckpt` stores the currently activated MindSpore classifier.
- `metadata/backend_state.json` stores active backend, dirty state, training state, last error, sample version, and last successful training metadata.

### Backend Responsibilities

#### Embedding Backend

- Preserve current `/register`, `/recognize`, `/identities`, `/delete_identity` capabilities.
- Continue saving face images and embeddings.
- Continue exposing identity counts and sample counts.
- Remain usable as a fallback backend even when MindSpore training fails.

#### MindSpore CNN Backend

- Build a classifier training dataset from `server/runtime/faces/`.
- Validate that each identity has valid image files before training.
- Create a deterministic label map from identity names to class ids.
- Train a MindSpore CNN classifier from the current dataset snapshot.
- Export checkpoint and metadata only after successful training.
- Load the active checkpoint for inference.
- Return class prediction and confidence during recognition.

## API Design

### Existing Endpoints Kept

#### `POST /register`

Request:

- `name`
- `photo`

Behavior:

- validate `name`
- write remote sample image into `server/runtime/faces/<name>/`
- keep current embedding-backend update behavior so the fallback backend stays usable
- mark `mindspore_cnn` as dirty

Response additions:

- `model_dirty`
- `active_backend`
- `mindspore_ready`
- `message`

The response must distinguish:

- sample saved successfully
- fallback embedding updated successfully
- MindSpore model now requires retraining

#### `POST /recognize`

Request:

- `photo`

Behavior:

- route to the currently active backend
- if active backend is `mindspore_cnn`, require a loaded activated model
- if active backend is `embedding`, preserve current behavior

Response additions:

- `backend`
- `ready`
- `score`
- `detail`

If `mindspore_cnn` is active but not ready, return a clear not-ready payload and an appropriate error status. Do not return a fake `"未知"` success response in that state.

#### `GET /health`

Keep the endpoint and add:

- `active_backend`
- `mindspore_ready`
- `training`
- `model_dirty`
- `last_train_time`
- `identity_count`
- `embedding_count`

#### `GET /identities`

Keep returning identity and sample counts. The identity listing remains based on the sample store and not on MindSpore classifier readiness.

#### `POST /delete_identity`

Behavior changes:

- delete identity samples
- delete embedding artifacts for that identity
- mark MindSpore backend dirty

Deleting an identity never auto-triggers training.

### New Endpoints

#### `POST /train`

Purpose:

- explicitly train the MindSpore CNN classifier from current remote samples

Request body:

- optional `epochs`
- optional `batch_size`
- optional `force_rebuild`

Behavior:

- reject if another training job is already running
- validate dataset readiness before starting
- build label map and dataset snapshot
- train classifier
- write checkpoint and metadata into a staging location
- only after full success, atomically replace the active model artifacts
- automatically activate the new MindSpore model
- clear `model_dirty`

Response:

- `status`
- `backend="mindspore_cnn"`
- `activated`
- `class_count`
- `sample_count`
- `model_path`
- `last_train_time`
- `detail`

#### `GET /model_status`

Purpose:

- expose state required by the desktop UI

Response fields:

- `active_backend`
- `available_backends`
- `mindspore_ready`
- `training`
- `model_dirty`
- `last_error`
- `last_train_time`
- `class_count`
- `sample_count`
- `active_model_version`

#### Optional `POST /activate_backend`

This endpoint is not required for the first implementation pass, but the design leaves room for it if runtime switching becomes useful later.

## MindSpore Model Design

### Classifier Type

The primary backend is a classifier, not a nearest-neighbor recognizer. The simplest acceptable first version is:

- a compact MindSpore CNN classifier;
- trained on the identities currently present in `server/runtime/faces/`;
- exported with a label map;
- used directly for inference to return class name plus confidence.

The first implementation should optimize for reliability and integration, not for leaderboard accuracy.

### Data Preparation

The service needs a training-data preparation layer that converts the existing sample tree into a MindSpore-friendly dataset structure.

Requirements:

- source of truth is `server/runtime/faces/<identity>/`
- ignore hidden files and unsupported suffixes
- reject empty identities during training validation
- build a stable identity ordering for label generation
- record dataset snapshot metadata so the service knows whether the active model matches current samples

The data-preparation output can be either:

- a temporary generated `ImageFolder`-compatible tree, or
- a custom MindSpore dataset wrapper over the existing sample tree

For the first pass, a generated `ImageFolder`-compatible dataset is the lower-risk option because it fits the existing `utils/mindface/recognition/datasets/face_dataset.py` shape.

### Training Trigger and Activation

Training is user-triggered from the desktop. After the remote service completes training successfully:

- checkpoint becomes the active MindSpore model automatically;
- label map and metadata become active at the same time;
- any previous active MindSpore model is retained until the new one is fully ready.

If training fails:

- the previous active MindSpore model remains active;
- `model_dirty` remains true;
- `last_error` is updated;
- the embedding backend remains available.

### Inference Contract

MindSpore inference returns:

- predicted identity name
- confidence score
- backend name
- readiness flag

Threshold handling:

- define a configurable minimum confidence threshold for accepting classifier output;
- below threshold, return `"未知"` with `matched=false`.

This keeps the classifier usable in the current UI without forcing the desktop layer to understand raw logits.

## Desktop Integration

### Current UI Constraints

The desktop currently provides:

- sample capture through `register_face`
- local classic-model training through `start_train`
- remote-service status inspection through `show_mindspore_status`
- deep recognition through the remote `/recognize`

The design keeps local classic training untouched.

### Required UI Changes

#### Registration Feedback

After a successful `/register`, the UI must stop implying the deep model is immediately ready.

Expected feedback:

- local sample saved
- remote sample saved
- remote MindSpore model requires retraining

#### Remote Training Entry

The existing "检查远端深度服务" button becomes the explicit remote training/status entrypoint.

Suggested behavior:

1. fetch `/model_status` or `/health`
2. log current backend and readiness
3. trigger `/train`
4. log training result
5. after success, log that the new MindSpore model has been automatically activated

The label text may also be updated later, but that is secondary to getting the flow correct.

#### Recognition-State Messaging

Deep-recognition mode must distinguish at least these states:

- remote deep model not trained
- remote deep model training
- remote deep model dirty because samples changed
- remote deep model ready
- remote deep model failed and fallback backend is still available

The user should never be left guessing whether remote deep recognition is unavailable because of connectivity, model state, or training state.

## Error Handling

### Training Validation Errors

Return explicit training failures for:

- no identities present
- fewer than two usable classes if the chosen training path requires multi-class classification
- an identity directory with zero usable images
- unreadable or invalid images

### Runtime Protection

- only one remote training task may run at a time
- failed training must not replace the last good active model
- service restarts must restore activation state from metadata
- recognition requests during training must continue using the last active backend

### Connectivity and UI-Side Errors

The desktop must surface:

- service unavailable
- model not ready
- training already running
- training failed with remote error

These should appear in logs and in recognition hints where relevant.

## Testing Strategy

### Service-Level Tests

Add automated coverage for:

- sample-tree scanning
- label-map generation
- backend-state persistence
- active-model replacement only after success
- `model_dirty` transitions on register and delete

### API Tests

Add tests for:

- `/register` marks MindSpore dirty
- `/train` rejects invalid datasets
- `/train` auto-activates on success
- `/recognize` returns not-ready when MindSpore is active but absent
- fallback embedding backend still works

### Desktop Integration Checks

Verify:

- registration no longer claims remote deep model is instantly usable
- remote training button triggers the new service flow
- deep recognition logs distinguish not-ready vs success vs network failure

## Non-Goals for First Pass

- replacing the local classic OpenCV training flow
- exposing backend switching in the desktop UI
- building asynchronous background remote training orchestration beyond a single in-process guarded training job
- optimizing MindSpore training for large-scale production use

## Implementation Boundaries

The first implementation pass should prefer incremental changes within existing files and only extract new service-side modules where the boundaries become clearer:

- keep the HTTP shell centered in `server/embedding_service.py`
- allow adding focused helper modules under `server/` for backend state, MindSpore training, dataset preparation, and inference
- keep desktop changes localized to `service/vision.py` and `view/main.py`

## Open Decisions Already Resolved

- MindSpore is the primary backend.
- MindSpore uses a CNN classifier approach.
- Embedding remains as a fallback backend.
- Remote training is explicit.
- Successful training auto-activates the new model.
- The current remote-status button becomes the remote training entrypoint.
