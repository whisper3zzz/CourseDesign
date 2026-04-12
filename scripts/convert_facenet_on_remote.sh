#!/bin/sh
set -eu

LITE_ROOT="${LITE_ROOT:?set LITE_ROOT to the unpacked mindspore-lite directory}"
MODEL_DIR="${MODEL_DIR:-server/runtime/models}"
ONNX_PATH="${MODEL_DIR}/facenet_vggface2.onnx"
MINDIR_PATH="${MODEL_DIR}/facenet_vggface2.mindir"

export LD_LIBRARY_PATH="${LITE_ROOT}/tools/converter/lib:${LITE_ROOT}/runtime/lib:${LD_LIBRARY_PATH:-}"

"${LITE_ROOT}/tools/converter/converter/converter_lite" \
  --fmk=ONNX \
  --modelFile="${ONNX_PATH}" \
  --outputFile="${MODEL_DIR}/facenet_vggface2" \
  --saveType=MINDIR

"${LITE_ROOT}/tools/benchmark/benchmark" \
  --modelFile="${MINDIR_PATH}" \
  --modelType=MindIR \
  --device=CPU \
  --loopCount=1 \
  --warmUpLoopCount=0
