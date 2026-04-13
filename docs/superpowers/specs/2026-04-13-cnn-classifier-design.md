# CNN Classifier Recognition Design

**Date:** 2026-04-13

**Goal:** Add a second deep-recognition path based on CNN classification, while keeping the current `mindspore_lite` embedding backend as the primary deployed deep-recognition path.

## Context

The project now has a working remote deep-recognition deployment based on:

- `InceptionResnetV1(vggface2)`
- ONNX export
- MindSpore Lite conversion
- remote `mindspore_lite` inference
- embedding similarity matching

That path remains valuable for engineering stability and open-set-like matching behavior.

However, the project also needs a **classification-style face recognition path** for course-design / report / demo purposes. That path should be added in parallel, not as a replacement for the current backend.

## Product Decision

### Approved Scope

The new CNN classifier path is an **additional backend**, not a replacement:

- keep the existing `mindspore_embedding` backend;
- add a new `cnn_classifier` backend;
- allow the desktop app to select between them;
- do not remove the current embedding deployment path.

### Recognition Semantics

The classifier path is explicitly **closed-set classification**:

- the model outputs one of the known identities from the current dataset;
- adding/removing identities requires retraining;
- inference returns class probability rather than similarity-to-library as the primary signal.

This is different from the embedding path and should be described as such in code and docs.

## Architecture

### High-Level Flow

The classifier flow should be:

1. use `dataset/full/<identity>/` as the source dataset;
2. build a classifier training dataset in ImageFolder-like format;
3. train a lightweight CNN classifier;
4. export the classifier to ONNX;
5. convert ONNX to MindIR on the SSH server;
6. load the converted model through `mindspore_lite`;
7. return top-1 class + probability during `/recognize`.

### Parallel Backends

After implementation, the system will have three recognition families:

1. `classic`
   - local OpenCV route
2. `mindspore_embedding`
   - current deployed remote deep route
3. `cnn_classifier`
   - new remote classification route

Only the new `cnn_classifier` path is added in this design.

## Model Choice

### Recommended First Model

The first classifier implementation should use a lightweight network such as `MobileNetV2` or another small CNN classifier rather than a more complex ArcFace-style training stack.

Rationale:

- easier to train and explain;
- smaller inference cost on CPU;
- better suited to a first end-to-end classifier deployment;
- lower integration risk than a more research-oriented metric-learning setup.

The chosen model should be treated as a classification network:

- input: cropped RGB face
- output: logits over known identities
- loss: cross entropy

### Label Mapping

The classifier path must persist a label map alongside the model:

```json
{
  "0": "B23051614杨凯",
  "1": "B23051626黄鸿"
}
```

That label map is part of the runtime contract and must be version-aligned with the exported model.

## Data Preparation

### Source of Truth

The source data remains:

- `dataset/full/<identity>/`

The training pipeline should:

- ignore hidden files;
- ignore unsupported image suffixes;
- ignore empty identity folders;
- produce deterministic class ordering;
- record class count and sample count in metadata.

### Training Split

The simplest valid first pass is:

- deterministic train/validation split from `dataset/full/`
- no need for a separate held-out test set in the first engineering pass

The implementation should still surface validation accuracy after training so the model does not become a black box.

## Runtime Artifacts

The classifier path should produce and store artifacts under a dedicated classifier area, for example:

```text
server/runtime/classifier/
├── classifier.onnx
├── classifier.mindir
├── label_map.json
└── metadata.json
```

`metadata.json` should include:

- model name
- class count
- sample count
- input size
- training timestamp
- validation accuracy
- conversion timestamp

## Remote Service Behavior

### New Backend

Add a `cnn_classifier` runtime backend that:

- loads `classifier.mindir` through `mindspore_lite`;
- resizes model input if required;
- runs inference;
- softmaxes logits;
- maps top-1 index through `label_map.json`;
- returns class name and confidence.

### Training Endpoint

The classifier route needs explicit training and deployment. Add:

- `POST /train_classifier`

That endpoint should:

1. validate the dataset;
2. run or trigger training;
3. export ONNX;
4. convert on the remote host;
5. activate the new classifier artifacts only after success.

### Health Reporting

Extend `/health` to include at least:

- `classifier_model_ready`
- `classifier_class_count`
- `classifier_sample_count`
- `active_backend`

If the active backend is `cnn_classifier`, `/health` should make that explicit.

## Recognition API Contract

### `/recognize`

When `cnn_classifier` is active, `/recognize` should return:

- `name`
- `matched`
- `score`
- `candidate_name`
- `candidate_score`
- `backend: "cnn_classifier"`
- `ready`
- `threshold`
- `identity_count`
- `embedding_count` or classifier-equivalent count field if retained for compatibility

Behavior:

- if top-1 probability >= threshold, `name` is the predicted identity;
- if top-1 probability < threshold, `name` becomes `未知`;
- `candidate_name` should still expose the top-1 class for UI purposes.

This keeps consistency with the recently added “show top candidate on misses” behavior.

## Desktop Integration

### UI Model Selection

The desktop should not replace the current deep-recognition mode. Instead it should split it into two remote deep modes:

- `深度匹配识别`
- `CNN分类识别`

This is clearer than silently switching behavior behind the same option.

### Training Trigger

The desktop needs a classifier-training trigger. The existing “检查远端深度服务” button can be generalized into:

- `检查/更新远端模型`

Behavior by mode:

- in `深度匹配识别`, keep current remote model checks;
- in `CNN分类识别`, call `POST /train_classifier` and show classifier readiness.

### Result Display

The current candidate-display behavior should also apply to classifier mode:

- confirmed class above threshold shows as the main result;
- below threshold, show `未知` as the confirmed result and display `candidate_name` as the current-frame top candidate.

## Error Handling

The classifier route must explicitly handle:

- empty dataset
- only one class
- broken or missing `label_map`
- model/label-map mismatch
- conversion failure
- Lite runtime load failure

All of these should fail closed and should not silently fall back to “some” class.

## Testing Requirements

The classifier implementation should not be considered complete until it has:

- dataset-preparation unit tests
- label-map / artifact metadata tests
- backend inference tests
- API tests for `/train_classifier` and `/recognize`
- at least one end-to-end validation using a real converted classifier artifact

## Non-Goals

- replacing the current `mindspore_embedding` backend
- removing the current remote matching route
- reworking the local classic OpenCV route
- implementing a large-scale ArcFace-style training system in the first pass

## Final Design Decision

The approved next feature is:

- add a parallel `cnn_classifier` backend
- keep `mindspore_embedding` as the current deployed primary deep-recognition path
- use a lightweight CNN classifier for the first implementation
- require explicit retraining when the identity set changes
- expose classifier confidence and current-frame top candidate to the desktop UI
