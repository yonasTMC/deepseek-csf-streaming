# Results

A chronological log of measurements on the target machine: 2× RTX PRO 6000 Blackwell (96 GB), 124 GiB of RAM, 2× Samsung 9100 PRO. Each section states its configuration. Later sections supersede earlier ones.

**Current state (2026-10-24):**
- **Configuration:** split pack, deterministic kernel, compacted prefill, progressive fill acknowledgement, per-layer slot counts learned from traffic, 4,400-expert RAM tier, 4 sequences.
- **Decode:** 88–95 tok/s on code, 76–77 in thinking mode, 65–70 on prose; 4 concurrent streams total about 92 tok/s.
- **Prefill:** about 1,810 tok/s on real 50k-token code prompts.
- **Recall:** passphrase recall exact at 2k, 8k and 25k tokens.

## v1 baseline — 2026-10-23

The original DeepSeek-V4.1-Flash-1017 lossless-CSF checkpoint ran with GPU expert computation, TP2 and bounded expert residency. This version is a correctness and measurement baseline.

## Configuration

- Two RTX PRO 6000 Blackwell Workstation GPUs, 96 GB each; 124 GiB host RAM.
- Original checkpoint revision `90d665c19e1a9ff2097cc787515e370d86b616f0`; all 48 shards SHA-256 verified (495,632,175,852 bytes).
- Pinned runtime and 1017 tool encoder specified in `README.md`.
- TP2; native MXFP4 expert weights and lossless CSF scales; MXFP8 activations. No additional weight quantization or CPU matrix multiplication.
- 128 expert slots per layer/rank, about 1.203 GB per layer/rank; 40 routed layers. Observed serving GPU allocation approximately 60.6–62.5 GiB per GPU.
- Eager execution, deterministic dynamic M16 expert kernel, 4096-token context, two sequences, 128-token prefill batches, 2 GiB explicit KV allocation per GPU. Speculative decoding disabled.
- Engram tables remain disk-backed. Optional B12X autotuning disabled for the baseline; required compilation/warmup remains enabled.
- Counters written after each layer invocation. Reported latency includes instrumentation and runtime overhead.

## Correctness evidence

- Seven CPU tests cover eviction safety, transactional route validation, random route sequences, padding, and streamed tool-call reconstruction.
- An independent CPU decoder verified original scale values and packed TP slices for sampled experts in layers 0, 20, and 39.
- Full-resident versus streamed GPU comparisons passed on both TP ranks for layers 0, 20, and 39. Prepared cached tensors matched byte for byte; routed outputs matched exactly before and after NCCL reduction. The tests used a 16-slot cache to force eviction and token-batch partitioning, through 128-token batches. Additional mixed/all-padding checks and the serving configuration’s 128-slot cache also passed exactly on both ranks.
- Numerical comparisons require zero absolute and relative tolerance. Automatic kernel selection initially caused numerical differences; both paths now explicitly select the same deterministic dynamic M16 kernel. This is equivalence to that resident kernel, not to every default kernel or an independent full-model logits baseline.
- Full-model smoke: `17 + 25` returned `42`, 11.07 seconds total, 10.22 seconds to first token. This first request included cold loading/compilation.

## Tool-using agent measurements

The agents inspect source and tests, modify a Python function, run tests inside isolated CPU containers, and report completion. Tasks cover interval merging and bounded retry delay. These are representative coding fixtures, not production-repository sessions. Each request permits 256 completion tokens, temperature zero, and thinking disabled.

| Run | Passed | Wall time | Completion tokens | End-to-end completion tok/s | Expert miss fraction |
|---|---:|---:|---:|---:|---:|
| Serial, two tasks | 2/2 | 705.68 s | 736 | 1.043 | 22.68% |
| Two concurrent tasks | 2/2 | 849.54 s | 839 | 0.988 | 24.62% |

Serial task times: interval merge 359.13 seconds; retry delay 346.54 seconds. Across eight serial model requests, time to first token ranged from 21.84 to 60.77 seconds. End-to-end throughput includes prefill, tool execution, final responses, and waiting; it is not decode-only throughput.

Cache requests count unique expert IDs per routing chunk. Counters sum both TP ranks, so each logical route is represented on each rank. Serial measurement: 668,590 expert requests, 151,654 misses, and 1,353,969,545,344 logical staged bytes. Staged bytes are not physical SSD reads; the OS page cache participates. Summed worker read/preparation time can overlap and must not be added to wall time.

The concurrent run started with a warm expert cache and uses a distinct run label in each task prompt to prevent identical task-history prefix reuse from the serial run. Common system/tool prefixes can still be reused normally. These conditions do not provide a controlled, cold-cache concurrency scaling ratio. Background activity on the machine was not isolated.

The concurrent retry task took 652.15 seconds. Its peer took approximately 849.5 seconds and included an additional test-before-fix round. Across nine concurrent-run model requests, first-token latency ranged from 34.39 to 209.49 seconds. All four agent sessions across both runs passed their unit tests. The concurrent run staged 1,629,417,058,212 logical bytes across both ranks and recorded 182,506 misses in 741,416 expert requests.

## Remaining performance work

Measurements identify frequent miss handling and substantial prefill/first-token cost. Each miss currently performs native CPU slicing/CSF processing and GPU preparation, with a bounded synchronization point per transfer batch. Disk work overlaps transfer/preparation, but this is not a fully overlapped decode pipeline. The next experiment should profile that miss path, then compare immutable prepacked TP expert records and a larger GPU cache while rerunning the exact correctness checks. No performance gain from those changes has been measured here.

## Engine v4 — 2026-10-23

Configuration: TP2, full CUDA graphs (`FULL_AND_PIECEWISE`), DSpark d3 with adaptive verification, 216 VRAM slots/layer (70% pinned by trace frequency), 4,400-expert pinned RAM tier per rank (exact-size `cudaHostRegister`), 64-buffer host read-ahead cache, 2,048-token prefill chunks, 1M context with 2 GiB KV (2.1M tokens), Engram shard 48 on nvme2 (byte-identical copy, SHA-256 b5b93edf…91ed). Container limit 115 GB.

Design: on a miss the GPU spins in-graph until a C++ host thread (no CUDA calls, no GIL) acknowledges; the GPU then copies expert records from pinned host memory (RAM tier or read-ahead buffer) into its slots and updates its slot table. The earlier spin design stalled under DSpark because helper threads' CUDA calls blocked inside the driver while the kernel spun (traced with Nsight Systems and py-spy).

Correctness: bit-exact against the fully resident reference on layer pairs 0/1, 20/21, 38/39, both ranks, 214 checks each (120 inside captured graphs), using the deterministic M16 kernel (the default B12X decode kernel is not run-to-run deterministic). Served probes (arithmetic, JSON extraction, SQL, translation, prompt-injection, tool call) correct.

| Measurement | Result |
|---|---|
| Decode, 1 stream (DSpark) | 77–83 tok/s, ~2.4 tokens/step |
| Decode, 2 streams combined | ~82 tok/s |
| Decode without DSpark (v4) | 63–64 tok/s |
| Zero-miss graph ceiling (no DSpark) | 114 tok/s |
| Prefill, fresh 32k random-word prompt (overstates real prompts several-fold; real code measured about 1.45–1.59k) | 2.37–2.44k tok/s |
| GPU fill copy from pinned host | 45–48 GB/s (24 blocks) |

Negative results kept for reference: breakable piecewise graphs without a spin (v3) ran ~30 tok/s because attention also runs eagerly; prefill read-ahead from next-layer router votes lowered prefill at every threshold (best case: 33% of reads used); 4,096-token prefill chunks needed fewer slots and lowered both prefill (1.75k) and decode (66).

Not established: 150 tok/s single stream; 4k tok/s prefill.

## Split pack + deterministic kernel — 2026-10-24 (deployed)

Root cause of junk tokens in greedy output: B12X's automatically selected small-batch MoE kernel is not reproducible at the TP2 slice width of 1,152 (2–3% of max between identical calls; about 5% from the deterministic kernel). Each rank's expert slice is now packed as two 576-wide half experts (`pack_experts.py --split 2`) and served through the deterministic dynamic M16 kernel with one 4,096-token plan. Masked (`-1`) routes corrupt this plan's output intermittently, so masked routes use a pinned slot with gate 0. See [docs/DESIGN.md §4](docs/DESIGN.md) and `evidence/junk-tokens/`.

| Measurement (dev server, 3,000-expert RAM tier, prefetch off) | Result |
|---|---|
| Greedy coding tasks (junk_test, T=0, thinking off) | 0 of 8 code blocks bad (runs E and H) |
| Passphrase recall at 2k / 8k / 25k tokens | exact at all three |
| Real-code prefill, 49k-token prompt | 1,322 tok/s |
| Decode, 1 stream | 58–61 tok/s (old auto kernel at the same tier: 60–64) |

Deployed with a 4,000-expert RAM tier and a 110Gi container limit.

## Prefill compaction + same-layer read-ahead — 2026-10-24

| Measurement | Before | After |
|---|---|---|
| Production pod, real-code prefill (48–52k tokens) | 1,384 tok/s | 1,792–1,801 tok/s |
| Dev server, same prompts | 1,364–1,366 | 1,739–1,795 |
| Passphrase recall (2k / 25k) | exact | exact |
| Greedy junk test | 0/8 bad | 0/8 bad |
| Decode, dev, B-tree prompt (4 samples each) | 61–71 tok/s | 64–65 tok/s |

Idle production pod before this change (split pack, 4,000-expert RAM tier), decode at T=0 with thinking off: code prompts 73–92 tok/s (2.7–3.2 tokens/step), prose prompts 42–44 tok/s (about 2.0 tokens/step, about 48 ms/step against 35 ms for code).

`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`: 2048 gave the same prefill as the default 512; 256 gave 1,812–1,896, which is within noise for two samples. The default is unchanged.

## Teacher-forced logprob comparison — 2026-10-24

`scripts/logprob_eval.py`, three fixed texts (code, model-written answers, reasoning prose), 7,349 positions, top-20 distributions; raw distributions (about 4 MB per run) are not in the repository; run logs are in `evidence/logprob-*.log`. R = full-width deterministic M16 reference, S = split, A = old auto kernel. "Small" sends the MoE through 8-token slices (the decode kernels) via `CSF_DEBUG_MOE_CHUNK=8`.

| Pair | Mean KL | Top-1 agreement | Positions with KL > 1 |
|---|---|---|---|
| R vs R (repeat run) | 0.0093 | 97.74% | 8 |
| R vs S (large batches) | 0.0102 | 97.70% | — |
| R vs S2 (compacted prefill) | 0.0100 | 97.67% | — |
| R vs A (large batches) | 0.0118 | 97.63% | — |
| R small vs S small | 0.0105 | 97.90% | 8 |
| R small vs A small | 0.0105 | 97.65% | 8 |
| A small vs A small (repeat) | 0.0106 | 97.54% | 6 |
| S small vs S small (repeat) | 0.0036 | 98.95% | 2 |

The stack has run-to-run noise beyond the MoE kernel: the reference differs from itself as much as from any engine, including flips of 95%-confident tokens. Mean KL therefore cannot separate the engines. The split engine is the most reproducible configuration measured. The end-to-end greedy junk tests remain the evidence that the noisy auto kernel caused the junk. The other noise sources (candidates: prefill group composition, sparse-attention top-k fallback, all-reduce) are not identified.

## Up to 4 concurrent sequences — 2026-10-24

`--max-num-seqs 4`, CUDA graphs for 1–16 tokens (`CSF_GRAPH_TOKENS=16`), soft pinning (see docs/DESIGN.md). Before this, 2 sequences: 2 streams gave 74 tok/s total, and a third request waited about 6 s for a slot.

| Streams | Total | Per stream | Time to first token |
|---|---|---|---|
| 1 | 75–82 tok/s | 87–94 tok/s | 0.4 s |
| 2 | 68 | 38–41 | ≤ 0.8 s |
| 3 | 82 | 29–34 | 0.7 s |
| 4 | 81–83 | 22–26 | ≤ 1.1 s |

Benchmark code prompts, 256 tokens per stream, T=0. No pinned slot was borrowed (`pinned_borrowed` = 0). Greedy junk test 0/6 bad, recall exact, prefill 1,736 tok/s on the same build. Under a live 5-agent workload, the expert lookups were 90.5% resident in VRAM; of the misses, 65% came from the RAM tier and 25% from NVMe.

## Placement learned from production traffic — 2026-10-24

`scripts/learn_placement.py` counts decode-sized calls (≤16 routed tokens) in production route traces (3.36M calls), pins the 151 most frequent experts per layer, and fills the RAM tier with the most frequent remaining (layer, expert) pairs across all layers (56–176 per layer). Replaying held-out traffic: pinned hits 65.7% → 74.3%, misses per step 39.0 RAM + 17.9 NVMe → 31.7 + 15.3.

Production pod, 4,400-expert RAM tier, T=0 thinking off unless noted (old = dev-trained `placement-v216-r4400-p70`, measured on the dev server with the same settings):

| Prompt | Old placement | Learned (production) |
|---|---|---|
| Thinking mode, T=1 | 64 tok/s | 75–81 |
| Prose (two prompts) | 41–45 | 62–67 |
| Code: B-tree | 78–84 | 83–88 |
| Code: ISO-8601 parser | 97–103 | 80–82 |
| 4 streams total (benchmark code prompts) | 90 | 76 (dev) |

Tokens per step are unchanged; the difference is expert misses. The old placement had been learned from dev benchmark traffic, including the ISO-8601 and LRU-cache prompts that the last two rows use.

The RAM tier is 4,400 per rank with a 116Gi container limit and nothing else heavy on the machine; 13 GiB of host RAM remained available after startup.

## Progressive fill acknowledgement and per-layer slots — 2026-10-24

Before, the host acknowledged a request only after every source of its fills was in host memory, so NVMe reads and the GPU's copies ran back to back. Now the host publishes all fill descriptors at once, with RAM-tier and ready read-ahead fills first, and a ready flag per fill; `wait_fill` copies each fill as soon as its flag is set.

`scripts/layer_budget.py` replays held-out decode traffic and moves pinned slots between layers at a fixed total. Misses per decode step fell from 48.8 to 43.1 (−11.8%); the RAM-tier fills it removed outnumber the NVMe reads (32.7 → 27.3 RAM, 16.1 → 15.8 NVMe per step). The deployed placement (`evidence/placement-layered-20261024.json`) has 128–304 slots per layer, 8,320 in total, the same VRAM.

Dev server, 4,400-expert RAM tier, 4 sequences, two samples per prompt:

| | Baseline | + progressive ack | + per-layer slots | Production after |
|---|---|---|---|---|
| Code | 80–89 tok/s | 86–92 | 88–97 | 88–95 |
| Prose | 61–66 | 64–70 | 65–70 | 65–70 |
| Thinking (T=1) | 74–78 | 74–82 | 80–81 | 76–77 |
| 4 streams total | 77 | 90 | 92 | 92 |
| Prefill, real code, about 50k tokens | 1,612–1,660 | 1,794–1,828 | 1,820–1,871 | 1,810 |

Greedy junk test 0/7 bad and recall exact at 2k, 8k and 25k on the final build.

**Chunked NVMe reads, built and not adopted.** Both 9100 PROs run at PCIe Gen5 x4. `fio` (`O_DIRECT`, `io_uring`, real pack file):

| Pattern | Throughput | Per-read completion, median |
|---|---|---|
| Whole 9.4 MB record, one in flight | 9.7 GB/s | 1.00 ms |
| 4 × 2.35 MB in parallel | 10.9 GB/s | 0.85 ms per chunk |
| 8 × 1.18 MB in parallel | 10.5 GB/s | 0.89 ms per chunk |
| 4 whole records in flight | 11.9 GB/s | 3.13 ms |

Chunks land together, so they offer no early prefix to copy, and only a lone read gains. `CSF_READ_CHUNKS=4` splits a request's demand read into parallel chunks only when it is that request's only NVMe read. In the dev A/B against whole reads it changed nothing beyond noise: code 86–92 vs 88–95 tok/s, prose 63–68 vs 65–70, 4 streams 91 vs 90, prefill 1,860–1,884 vs 1,848–1,946. The option stays off by default (`CSF_READ_CHUNKS=1`).
