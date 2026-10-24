#!/usr/bin/env bash
# Exact reference: v2 eager path, deterministic M16 kernel, no graphs, no v4, no DSpark.
cd "$(dirname "$0")/.."
bash bench/restart.sh exact -e CSF_MOE_KERNEL=m16 -e CSF_PLACEMENT=/work/evidence/placement-learned-20261024.json \
  -e CSF_STREAM_SLOTS=192 -e GPU_UTIL=0.95 -e MAX_BATCHED=1024 -e MAX_LEN=65536 -e "SPEC_ARGS=--mode off" \
  -e CSF_PREFETCH_WIDTH=0 -e CSF_STREAM_METRICS_INTERVAL=2000 -e CSF_RAM_EXPERTS=2000
