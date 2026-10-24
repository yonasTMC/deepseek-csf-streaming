#!/usr/bin/env bash
set -euo pipefail
export MODEL=${MODEL:?set MODEL to the checkpoint snapshot directory}
export CHECKPOINT=csf TP=2 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CSF_STREAMING=1 CSF_STREAM_SLOTS=${CSF_STREAM_SLOTS:-128}
export CSF_STREAM_METRICS_INTERVAL=${CSF_STREAM_METRICS_INTERVAL:-50}
export CSF_STREAM_METRICS_DIR=${CSF_STREAM_METRICS_DIR:-/work/evidence/server-metrics}
export PYTHONPATH=/work:/opt/lil
export ENGRAM_TABLE_MEMORY=${ENGRAM_TABLE_MEMORY:-disk} ENGRAM_DISK_RESIDENT_SCALES=${ENGRAM_DISK_RESIDENT_SCALES:-true}
export VLLM_ENABLE_PCIE_ALLREDUCE=${VLLM_ENABLE_PCIE_ALLREDUCE:-0} VLLM_DISABLE_SHARED_EXPERTS_STREAM=${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}
export B12X_W4A8_CSF_INLINE=0 B12X_AUTOTUNE=0
ulimit -l unlimited
exec ${PROFILE_PREFIX:-} /usr/local/bin/lil-serve --profile ds41-flash --hardware rtx-pro-6000-pcie -- \
  ${SPEC_ARGS:---mode off} ${EAGER---enforce-eager} --max-model-len ${MAX_LEN:-4096} --max-num-seqs ${MAX_SEQS:-2} \
  --max-num-batched-tokens ${MAX_BATCHED:-128} --gpu-memory-utilization ${GPU_UTIL:-0.90} \
  --kv-cache-memory-bytes ${KV_BYTES:-2147483648} \
  --host 127.0.0.1 --port 8008 --served-model-name DeepSeek-V4.1-Flash-streaming ${EXTRA_ARGS:-}
