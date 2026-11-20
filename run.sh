#!/bin/bash
# Run the judge inside the SGLang v0.5.23 container on a machine with an H200, docker and the
# NVIDIA Container Toolkit. Arguments go to judge.py:
#   ./run.sh submission.py --tests-only
#   ./run.sh submission.py --sizes 1,8 --json results.json
# Any other command in the same container (Nsight Compute needs access to the GPU counters):
#   DOCKER_ARGS=--cap-add=SYS_ADMIN ./run.sh --exec ncu --profile-from-start off --set full \
#       -o prof python3 judge.py submission.py --ncu 8
# The repo is mounted at /work. JIT caches (SGLang's kernels, Triton, load_inline builds) are kept
# in ~/.cache/starter-task, so only the first run compiles them.
set -e
IMAGE=lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f
CACHE="$HOME/.cache/starter-task"
mkdir -p "$CACHE/cache" "$CACHE/triton"
if [ "$1" = "--exec" ]; then shift; CMD=("$@"); else CMD=(python3 judge.py "$@"); fi
exec docker run --rm --gpus "device=${GPU:-0}" --ipc=host ${DOCKER_ARGS} \
    -v "$(cd "$(dirname "$0")" && pwd)":/work -w /work \
    -v "$CACHE/cache":/root/.cache -v "$CACHE/triton":/root/.triton \
    "$IMAGE" "${CMD[@]}"
