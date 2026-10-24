#!/usr/bin/env bash
# Teacher-forced logprob matrix: reference (full pack, m16), split, auto; big and 8-token batches.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=evidence/logprob; mkdir -p $OUT
common="SLOTS=208 RAM_EXPERTS=3000 PREFETCH_WIDTH=0 MAX_LEN=16384 SPEC_ARGS=--mode\ off"
run() {  # name, extra env...
  local name=$1; shift
  echo "== $name $(date +%T)"
  if ! env SLOTS=208 RAM_EXPERTS=3000 PREFETCH_WIDTH=0 MAX_LEN=16384 "SPEC_ARGS=--mode off" "MORE_ARGS=--no-enable-prefix-caching" "$@" bash scripts/serve-v4.sh >/dev/null 2>&1; then
    echo "   start failed"; grep -oE "rror: .{0,200}" evidence/server-v4.log | grep -v triton | head -3; return
  fi
  python3 scripts/logprob_eval.py collect --out $OUT/$name.json 2>&1 | sed "s/^/   /"
}
run R_big   MOE_KERNEL=m16 MAX_BATCHED=4096
run R_small MOE_KERNEL=m16 MAX_BATCHED=8
run S_big   PACK_DIRS=/pack/nvme1/split:/pack/nvme2/split MAX_BATCHED=4096
run S_small PACK_DIRS=/pack/nvme1/split:/pack/nvme2/split MAX_BATCHED=8
run S_small2 PACK_DIRS=/pack/nvme1/split:/pack/nvme2/split MAX_BATCHED=8
run A_big   MAX_BATCHED=4096
run A_small MAX_BATCHED=8
run A_small2 MAX_BATCHED=8
docker exec csf-streaming-dev pkill -INT -f vllm.entrypoints; sleep 20; docker exec csf-streaming-dev pkill -9 -f "VLLM::"
echo DONE $(date +%T)
