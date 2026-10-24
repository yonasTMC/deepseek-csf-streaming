#!/usr/bin/env bash
# Dev container for the engine: the pinned LiL vLLM image with this repo mounted at /work.
#   MODEL       checkpoint snapshot directory (required for start); mounted read-only at the same path
#   PACK_DIR1   expert pack for TP rank 0 (default: <repo>/expert-pack/nvme1), mounted at /pack/nvme1
#   PACK_DIR2   expert pack for TP rank 1 (default: <repo>/expert-pack/nvme2), mounted at /pack/nvme2
#   ENGRAM_COPY optional byte-identical copy of an Engram shard on a second drive, mounted over
#   ENGRAM_BLOB the shard's blob path inside the checkpoint (both or neither)
# Put the two packs on different NVMe drives: rank 0 and rank 1 then read in parallel.
set -euo pipefail
BASE=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=ghcr.io/local-inference-lab/vllm@sha256:fd74019fc217a541060093d18427c36330df6ac5700651b55593dafdbb936920
CONTAINER=csf-streaming-dev
PACK_DIR1=${PACK_DIR1:-$BASE/expert-pack/nvme1}
PACK_DIR2=${PACK_DIR2:-$BASE/expert-pack/nvme2}
case "${1:-}" in
  start)
    : "${MODEL:?set MODEL to the checkpoint snapshot directory}"
    mkdir -p "$BASE/runtime-cache" "$BASE/evidence" "$PACK_DIR1" "$PACK_DIR2"
    docker image inspect "$IMAGE" >/dev/null
    if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
      test "$(docker inspect --format '{{.Config.Image}}' "$CONTAINER")" = "$IMAGE"
      docker start "$CONTAINER"
      exit
    fi
    ENGRAM=()
    if [ -n "${ENGRAM_COPY:-}" ]; then ENGRAM=(-v "$ENGRAM_COPY:${ENGRAM_BLOB:?set ENGRAM_BLOB with ENGRAM_COPY}:ro"); fi
    docker run -d --name "$CONTAINER" --restart=no --init "${ENGRAM[@]}" \
      --runtime nvidia --gpus all --network host --ipc private --shm-size 8g \
      --memory 115g --memory-swap 115g --ulimit memlock=-1 \
      --security-opt seccomp=unconfined --security-opt label=disable \
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=1 \
      -e B12X_W4A8_CSF_INLINE=0 -e MODEL="$MODEL" \
      -e PYTHONPATH=/work:/opt/lil -e XDG_CACHE_HOME=/cache \
      -v "$BASE:/work" -v "$BASE/runtime-cache:/cache" \
      -v "$BASE/vendor/deepseek_v41_encoding_1017.py:/opt/venv/lib/python3.12/site-packages/vllm/tokenizers/deepseek_v41_encoding.py:ro" \
      -v "$MODEL:$MODEL:ro" -v "$PACK_DIR1:/pack/nvme1" -v "$PACK_DIR2:/pack/nvme2" \
      --entrypoint /bin/bash "$IMAGE" -lc 'exec sleep infinity'
    ;;
  stop) docker stop -t 30 "$CONTAINER" ;;
  *) echo 'Usage: MODEL=<snapshot dir> devctl.sh start | devctl.sh stop' >&2; exit 2 ;;
esac
