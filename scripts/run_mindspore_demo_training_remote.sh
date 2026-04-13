#!/bin/sh
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REMOTE_ROOT="${REMOTE_ROOT:-$HOME/course-design-backend}"
REMOTE_RUNTIME="${REMOTE_ROOT}/server/runtime/mindspore_demo"
REMOTE_SOURCE="${REMOTE_SOURCE:-$REMOTE_ROOT/server/runtime/faces}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/mscheck/.venv-ms/bin/python}"

mkdir -p "$REMOTE_RUNTIME"

"$PYTHON_BIN" "$REPO_ROOT/scripts/prepare_mindspore_demo_dataset.py" \
  --source-root "$REMOTE_SOURCE" \
  --output-root "$REMOTE_RUNTIME"

"$PYTHON_BIN" "$REPO_ROOT/scripts/train_mindspore_demo_classifier.py"
