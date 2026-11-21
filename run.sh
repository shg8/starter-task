#!/bin/bash
# Run the judge in the pinned SGLang image. Use --exec to run another command in the image.
set -e

IMAGE=docker.io/lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f
CACHE=${XDG_CACHE_HOME:-"$HOME/.cache"}/starter-task
REPO=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$CACHE/cache" "$CACHE/triton"
if [ "${1:-}" = "--exec" ]; then
    shift
    CMD=("$@")
else
    CMD=(python3 -u judge.py "$@")
fi

GPU_ARGS=()
if [ -z "${SLURM_JOB_ID:-}" ]; then
    GPU_ARGS=(--gpus "device=${GPU:-0}")
fi

exec docker run --rm "${GPU_ARGS[@]}" --ipc=host ${DOCKER_ARGS:-} \
    -v "$REPO":/work -w /work \
    -v "$CACHE/cache":/starter-cache -v "$CACHE/triton":/starter-triton \
    -e XDG_CACHE_HOME=/starter-cache -e TRITON_CACHE_DIR=/starter-triton \
    "$IMAGE" "${CMD[@]}"
