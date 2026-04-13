# MindSpore CNN Training Demo Design

**Date:** 2026-04-13

**Goal:** Add a real MindSpore CNN training path that consumes `dataset/full/`, runs on the remote Ubuntu server, produces a checkpoint and label map, and is kept separate from the currently deployed recognition backends.

## Context

The project now has two working remote deep-recognition paths:

- `mindspore_embedding`
- `cnn_classifier`

Both are integrated into the running service path.

The new requirement is different:

- provide a **real MindSpore CNN training implementation**
- train on the current dataset under `dataset/full/`
- make it genuinely runnable
- do **not** connect it into the active recognition chain

This path is therefore best treated as a **training/demo pipeline**, not a production backend.

## Product Decision

### Approved Scope

The new MindSpore CNN work is:

- a real training path
- a real model artifact generator
- a real label-map / metrics exporter
- intentionally isolated from the main `/recognize` runtime flow

It is **not**:

- a replacement for `mindspore_embedding`
- a replacement for `cnn_classifier`
- a third runtime backend in this implementation phase

### Why Isolate It

This path exists primarily to demonstrate that:

- a CNN was built with MindSpore
- the existing dataset can train it
- the project can produce genuine MindSpore training artifacts

Keeping it isolated avoids destabilizing the now-working deployed recognition service.

## Architecture

### Training Location

Training should run on the already-validated remote Ubuntu server rather than on the local macOS machine.

Reasons:

- the remote host already runs the project’s server-side code
- MindSpore runtime is already verified there
- local macOS MindSpore execution is less predictable for full CNN training
- this minimizes the chance of a demo-only training path breaking because of local operator/runtime limitations

### Data Source

The source of truth remains:

- `dataset/full/<identity>/`

The remote training pipeline should consume this exact dataset shape after it is synchronized or staged on the remote host.

### Model Type

The recommended first MindSpore model is a **small custom CNN classifier**, not MobileFaceNet, ArcFace, or a large pretrained architecture.

Recommended shape:

- `Conv2d(3 -> 16) + ReLU + MaxPool`
- `Conv2d(16 -> 32) + ReLU + MaxPool`
- `Conv2d(32 -> 64) + ReLU + AdaptiveAvgPool`
- `Flatten + Dense(64 -> num_classes)`

Reasons:

- easy to explain in a course-design context
- genuinely “built with MindSpore”
- lower CPU training cost
- far less integration complexity

## Data Preparation

### Dataset Layout

The training path should prepare a deterministic train/validation split from `dataset/full/` into an ImageFolder-like layout on the remote host:

```text
server/runtime/mindspore_demo/
├── dataset/
│   ├── train/
│   │   └── <identity>/
│   └── val/
│       └── <identity>/
```

Rules:

- ignore hidden files and directories
- ignore empty identities
- keep deterministic class ordering
- persist the class ordering into `label_map.json`

### Validation Split

The first version can use a simple deterministic split such as:

- first `80%` for train
- remaining `20%` for validation

The important point is that the split is reproducible and documented.

## Artifacts

Training should produce a dedicated artifact set, separate from active inference artifacts:

```text
server/runtime/mindspore_demo/
├── dataset/
├── mindspore_classifier.ckpt
├── label_map.json
└── metrics.json
```

### `label_map.json`

Stores class id to identity mapping:

```json
{
  "0": "B23051614杨凯",
  "1": "B23051626黄鸿"
}
```

### `metrics.json`

Stores at least:

- `class_count`
- `sample_count`
- `epochs`
- `last_train_loss`
- `last_val_accuracy`
- `trained_at`

## Runtime Separation

This training path must stay isolated from current recognition backends.

That means:

- no changes to `/recognize`
- no auto-activation into `mindspore_embedding`
- no auto-activation into `cnn_classifier`
- no UI mode switch to this backend in this phase

At most, the project may later expose a status/log view for the artifacts, but not a live runtime selection path.

## Script Layout

Recommended new files:

- `scripts/prepare_mindspore_demo_dataset.py`
  - prepare remote train/val split
- `scripts/train_mindspore_demo_classifier.py`
  - define and train the small MindSpore CNN
- `scripts/export_mindspore_demo_metadata.py`
  - write `label_map.json` / `metrics.json` if not already handled inside training
- optional remote helper:
  - `scripts/run_mindspore_demo_training_remote.sh`

The scripts should be written so the training path can be re-run predictably.

## Deployment Model

This feature should support a simple operator workflow:

1. sync or stage `dataset/full/` to the remote host
2. prepare classifier dataset on the remote host
3. run MindSpore CNN training on the remote host
4. inspect generated checkpoint and metrics

That is enough to satisfy the demo/training requirement without touching live recognition behavior.

## Error Handling

The training path must explicitly fail on:

- no class directories
- only one usable class
- unreadable image files
- empty train or validation split
- checkpoint write failure

Errors should be explicit and should not silently write incomplete artifacts.

## Verification Requirements

This feature should only be considered complete if all of the following are true:

1. the remote host can run the dataset-preparation script successfully
2. the remote host can run the MindSpore CNN training script to completion
3. `mindspore_classifier.ckpt` is created
4. `label_map.json` is created
5. `metrics.json` is created
6. logs or output clearly show training loss and validation accuracy
7. the active recognition backends remain unchanged

## Non-Goals

- serving this demo model through `/recognize`
- replacing `mindspore_embedding`
- replacing `cnn_classifier`
- introducing a third UI-selectable recognition backend

## Final Design Decision

The approved implementation is:

- add a **remote-only MindSpore CNN training demo path**
- train a small custom CNN classifier against `dataset/full/`
- produce checkpoint + label map + metrics artifacts
- keep the entire feature isolated from the active recognition runtime
