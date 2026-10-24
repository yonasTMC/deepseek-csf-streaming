#!/usr/bin/env bash
# usage: restart.sh <label> [extra docker -e args...]; restarts dev server with v2 engine and waits for health
set -u
label=$1; shift
cd "$(dirname "$0")/.."
docker exec csf-streaming-dev pkill -INT -f vllm.entrypoints
for i in $(seq 1 30); do docker exec csf-streaming-dev pgrep -f vllm >/dev/null || break; sleep 2; done
docker exec csf-streaming-dev pkill -9 -f vllm 2>/dev/null
sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
docker exec -d -e CSF_PACK_DIRS=/pack/nvme1:/pack/nvme2 -e CSF_STREAM_METRICS_DIR=/work/evidence/metrics-$label "$@" csf-streaming-dev bash -lc "bash /work/scripts/serve.sh > /work/evidence/server-$label.log 2>&1"
sleep 20
for i in $(seq 1 120); do
  curl -sf -o /dev/null http://127.0.0.1:8008/health && { echo READY; exit 0; }
  docker exec csf-streaming-dev pgrep -f vllm.entrypoints >/dev/null || { echo DIED; grep -E "rror:|Traceback" evidence/server-$label.log | grep -v triton | tail -5; exit 1; }
  sleep 10
done
echo TIMEOUT; exit 1
