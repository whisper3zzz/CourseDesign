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

die() {
  printf '%s\n' "$1" >&2
  exit 1
}

[ -d "${LITE_ROOT}" ] || die "missing LITE_ROOT: ${LITE_ROOT}"
[ -x "${CONVERTER_BIN}" ] || die "missing converter binary: ${CONVERTER_BIN}"
[ -x "${BENCHMARK_BIN}" ] || die "missing benchmark binary: ${BENCHMARK_BIN}"
mkdir -p "${MODEL_DIR}"
[ -f "${ONNX_PATH}" ] || die "missing ONNX model: ${ONNX_PATH}"

export LD_LIBRARY_PATH="${LITE_ROOT}/tools/converter/lib:${LITE_ROOT}/runtime/lib:${LD_LIBRARY_PATH:-}"

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
