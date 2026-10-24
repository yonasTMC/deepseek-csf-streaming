#!/usr/bin/env bash
# Engine v4 (GPU-driven fills, C++ host service) with DSpark, 1M context,
# on 127.0.0.1:8008 inside the dev container. Usage: bash scripts/serve-v4.sh
set -euo pipefail
cd "$(dirname "$0")/.."
bash bench/restart.sh v4 ${MODEL:+-e MODEL=$MODEL} \
  -e CSF_V4=1 -e CSF_GRAPH=1 -e CSF_EXPERT_MAJOR_PREFILL="${EXPERT_MAJOR:-1}" -e CSF_PREFETCH_WIDTH="${PREFETCH_WIDTH:-6}" ${RAM_EXPERTS:+-e CSF_RAM_EXPERTS=$RAM_EXPERTS} ${PACK_DIRS:+-e CSF_PACK_DIRS=$PACK_DIRS} ${MOE_KERNEL:+-e CSF_MOE_KERNEL=$MOE_KERNEL} ${MOE_CHUNK:+-e CSF_DEBUG_MOE_CHUNK=$MOE_CHUNK} ${INDEXER_MB:+-e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=$INDEXER_MB} ${GRAPH_TOKENS:+-e CSF_GRAPH_TOKENS=$GRAPH_TOKENS} -e MAX_SEQS="${MAX_SEQS:-2}" -e CSF_V4_TRACE="${ROUTE_TRACE:-/work/evidence/traffic/routes-$(date +%Y%m%d-%H%M%S).bin}" -e CSF_PREFETCH_INFLIGHT="${PREFETCH_INFLIGHT:-24}" -e CSF_PREFILL_PREFETCH="${PREFILL_PREFETCH:-0}" -e CSF_PREFILL_MIN_VOTES="${PREFILL_MIN_VOTES:-1}" -e CSF_HOST_CACHE="${HOST_CACHE:-64}" \
  -e CSF_PLACEMENT=/work/evidence/${PLACEMENT:-placement-v216-r4400-p70.json} -e CSF_STREAM_SLOTS="${SLOTS:-216}" \
  -e GPU_UTIL=0.97 -e KV_BYTES="${KV_BYTES:-2147483648}" -e EAGER= -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e MAX_LEN="${MAX_LEN:-1048576}" -e MAX_BATCHED="${MAX_BATCHED:-2048}" \
  -e "SPEC_ARGS=${SPEC_ARGS:---mode dspark --draft-tokens 3}" \
  -e "EXTRA_ARGS=--max-cudagraph-capture-size ${CAPTURE:-8} ${MORE_ARGS:-}"
