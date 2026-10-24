#!/usr/bin/env bash
# A/B for fill and placement changes: decode by content, 4 streams, real-code prefill.
set -uo pipefail
cd "$(dirname "$0")/.."
run() {
  local name=$1 placement=$2
  echo "=== $name ($placement) $(date +%T)"
  env SLOTS=208 RAM_EXPERTS=4400 PREFETCH_WIDTH=0 MAX_BATCHED=4096 HOST_CACHE=160 PACK_DIRS=/pack/nvme1/split:/pack/nvme2/split GRAPH_TOKENS=16 MAX_SEQS=4 CAPTURE=16 PLACEMENT=$placement bash scripts/serve-v4.sh >/dev/null 2>&1 || { echo start failed; grep -oE "rror: .{0,250}" evidence/server-v4.log | grep -v triton | head -4; return 1; }
  python3 scripts/decode_mix.py
  python3 scripts/concurrency.py --levels 4 | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(\"concurrency 4: total\", d[\"aggregate_tok_s\"], \"tok/s; per stream\", [s[\"tok_s\"] for s in d[\"streams\"]])"
  for i in 1 2; do python3 scripts/real_prefill.py --model DeepSeek-V4.1-Flash-streaming 2>&1 | grep "^{"; done
}
run "${1:-baseline}" "${2:-placement-learned-20261024.json}"
