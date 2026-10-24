# References and design assessment: CSF expert streaming at TP2

*Historical snapshot (2026-10-23), written before engine v2–v4; see docs/DESIGN.md for the current design.*

Research date: 2026-10-23. Scope: preserve the original checkpoint's native weights, GPU expert computation, and tensor parallelism across two RTX PRO 6000 Blackwell GPUs. Research and local source inspection only; no service changes or GPU experiments in this pass.

## Decision

Expert caching and streaming have substantial prior art. We should use it. Our implementation is useful as a correctness oracle and measurement baseline, but should not be treated as a nearly finished high-performance engine.

I found runnable examples, papers, implementation code, and a particularly close TP2 deployment. I did **not** locate a reproducible public implementation matching all of: DeepSeek 1017 lossless CSF, two 96 GB GPUs, approximately 124 GiB RAM, SSD-backed routed experts, GPU-only expert arithmetic, and the quoted 150 tok/s. That is a bounded search result, not evidence that the quoted engine does not exist.

Published speed numbers below are the authors' measurements; none was reproduced on our node. Repository default branches and model cards can change.

## Closest references

| Reference | What is useful | Difference from our target |
|---|---|---|
| [cliecy: DeepSeek 1017 on 2× RTX PRO 6000](https://github.com/cliecy/DeepSeek-V4.1-Flash-1017-2xRTXPRO6000) | Runnable TP2 setup, pinned dependencies, benchmark records. Reports 89–112 tok/s single-stream and roughly 5,000 prefill tok/s. | EXL3 3-bit experts, int4 Engram, DSpark; about 245 GiB installed RAM and roughly 190 GiB required. Not lossless CSF. |
| [Diffbot serving recipe](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000) | Same model family and GPUs; GPU execution of host-resident experts, a hot-expert cache, specialized prefill kernels, serving and quality tests. | Requantized experts and Engram; optional CPU expert mode and lossy prefill optimizations must be distinguished from the GPU default. |
| [Mixtral offloading](https://github.com/dvmazur/mixtral-offloading) | Small, understandable cache implementation: persistent storage, reusable buffers, nonblocking transfers, cached-expert-first iteration. | Mixtral/HQQ and host RAM; no demonstrated DeepSeek CSF TP2 support. |
| [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity) | Larger reference for tracing, prediction, cache budgets, I/O scheduling and serving integration. | Its public implementation differs from the paper; local multi-GPU expert distribution is not a demonstration of our TP2 layout. |
| [ProMoE](https://arxiv.org/abs/2410.22134) and [code](https://github.com/promoe-opensource/promoe) | Predictive loading, demand priority, chunked transfers, and executing available experts while misses load. | Research system and other model configurations; its speedups cannot be transferred to our hardware. |
| [NeutronStar](https://github.com/giannisanni/neutronstar) | Actual CUDA SSD-streaming code, parallel reads, io_uring, aligned reusable buffers, host caching and prefetch experiments. | Different models/quants and much smaller hardware; useful engineering evidence, not a comparable speed result. |

The [cliecy architecture document](https://github.com/cliecy/DeepSeek-V4.1-Flash-1017-2xRTXPRO6000/blob/main/docs/ARCHITECTURE.md) explains why its result differs: it keeps 70% of experts on GPUs, holds the remainder in pinned host RAM, and uses DSpark k=5. It reports about 3.97 output tokens per verification step on its quality prompts. Our baseline caches 128/384 experts per layer, disables speculation, and may need SSD access. Its measured host-memory use exceeds our node's total RAM. Copying those launch flags would not reproduce its result.

The current Diffbot card also documents an agentic replay suite and comparisons against original-model logits. This is useful methodology: measure task wall time and numerical changes alongside decode throughput. Its format and arithmetic changes are outside our preservation requirement; the recipe is a design reference, not authorization to adopt those changes.

## What the code and papers establish

### Persistent expert representations and reusable buffers

I inspected Mixtral's [`src/expert_cache.py`](https://github.com/dvmazur/mixtral-offloading/blob/master/src/expert_cache.py). It allocates pinned expert storage and device buffers in initialization, registers experts once, and transfers their existing storage on a miss. `load_experts(..., unordered=True)` prioritizes resident experts and pipelines subsequent loads. This supports making our miss path a transfer of an already prepared record instead of repeating checkpoint interpretation and packing.

The exact CSF record layout is still our engineering task. A derived cache must be keyed by checkpoint revision/hash, TP size/rank, kernel/layout version, activation mode and scale representation. Changing these must invalidate the cache. Keep the original checkpoint authoritative and verify every derived record against it.

### Async submission alone does not hide the stall

[ProMoE's scheduling sections](https://arxiv.org/html/2410.22134v1) describe demand reads preempting speculative reads and available experts executing before unavailable ones. The relevant objective is how much loading remains on the dependency path. Prediction accuracy alone is insufficient if bytes arrive too late. Paper versions report different aggregate speedups; this assessment does not use those numbers as a forecast.

For our exactness requirement, reordering expert computation is distinct from reordering floating-point accumulation. A redesigned path can compute contributions as data becomes ready, but must preserve the required accumulation semantics and pass equivalence tests. Merely summing partial outputs in arrival order can introduce differences.

### Prefill and decode need separate policies

The [FineMoE EuroSys 2026 paper](https://intellisys.haow.us/assets/pdf/Hanfei_FineMoE_EuroSys26.pdf) studies fine-grained selection patterns and uses them for prefetching and caching. Its multi-GPU execution uses expert parallelism, so it does not supply a TP2 implementation. Its value here is evidence that coarse, uniform cache treatment leaves useful locality unexploited.

The inspected [MoE-Infinity prefetcher](https://github.com/EfficientMoE/MoE-Infinity/blob/main/moe_infinity/memory/expert_prefetcher.py) has separate prefill/decode counters, priority bands for demand/route-ahead/background work, and an in-flight byte budget. Its [compatibility matrix](https://github.com/EfficientMoE/MoE-Infinity/blob/main/docs/model-compatibility.md) does not establish our DeepSeek V4.1 CSF configuration. These are components to study, not a validated replacement engine.

### SSD behavior must be measured using expert reads

In [NeutronStar's CUDA source](https://github.com/giannisanni/neutronstar/blob/main/ds4_cuda.cu), I verified io_uring initialization, an aligned read-buffer recycling pool, host caching, and asynchronous CUDA uploads. Its README also reports a case where deeper prefetch lost performance because wrong reads consumed demand bandwidth. More prefetch is not automatically better.

The [1bit streaming write-up](https://1bit.gg/moe-streaming.html) records expert-route traces, cache-size replay and random expert-sized reads, rather than relying solely on sequential SSD bandwidth. It also explicitly says its previously measured Vulkan serving path is no longer built in the current engine. Use its methodology; do not mistake that page for a ready-to-run current NVIDIA implementation. Avoid its optional expert-substitution path for our preservation requirement.

### Fast results can solve a different problem

[Fiddler](https://arxiv.org/abs/2402.07033) uses CPU/GPU orchestration; [its code](https://github.com/efeslab/fiddler) is useful for understanding that alternative, but CPU expert arithmetic is outside our requested design.

Similarly, [inferencekey/deepseek-one-gpu](https://github.com/inferencekey/deepseek-one-gpu) reports fast single-GPU serving using Q2_K, approximately 280 GB of host RAM, and CPU execution for cold experts. It is not evidence that our unchanged CSF weights can achieve those numbers using SSD transfers and GPU arithmetic.

## CSF-specific findings

The [original checkpoint card](https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF) specifies lossless compression of selected MXFP4 scale planes, not a new low-bit requantization of all weights. It explicitly qualifies storage/CPU reconstruction separately from serving. “Unquantized” in the quoted discussion should therefore be understood cautiously: our source already contains native FP4 experts.

[Local Inference Lab's runtime documentation](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/deepseek-v4.1-flash.md) describes TP4 and disk-backed Engram. **Engram streaming is a different subsystem from routed-expert streaming.** Its successful TP4 launch does not demonstrate our missing TP2 expert cache.

The [LiL configuration documentation](https://docker.local-inference-lab.ai/?build=karmic-kraken-beta-e1088ae61c3a477b64803724358279c0200f00e8993166947324f57c4e58e996&model=ds41-flash) says compatible B12X execution can retain compressed scales in GPU memory. Our baseline explicitly expands scales in its expert slots. Compare compact-scale slots against pre-expanded records: one saves residency/traffic, the other may save repeated decoding. Do not assume either wins without a layout and timing experiment.

[Flare's design](https://github.com/NVIDIA/flare-local-inference/blob/main/docs/core-design.md) is relevant to preparation, persistent state ownership and graph execution around B12X/CSF. I did not establish that it supplies a working SSD expert cache for this target. It is not a replacement recommendation.

## Audit of our baseline

These observations come from local `cache.py`, `storage.py`, `policy.py`, `serve.sh` and [our measured results](RESULTS.md), not from external performance claims.

| Observed design | Consequence / next measurement |
|---|---|
| A miss reopens a CSF reader, performs native CPU slicing/scale handling, pins new tensors, uploads, prepares/repackages, then copies into slots. | Repeated preparation is on the miss path. Time each stage separately; prototype immutable prepared records. |
| `ensure()` finishes every missing batch, including `ready.synchronize()`, before `fm.run()`. | Reads overlap other preparation, but available-expert computation does not hide these misses. |
| Routing IDs go through `.cpu().tolist()` each invocation. | Host synchronization is present even on a fully warm cache. Measure a no-miss decode ceiling first. |
| Routing partitions token rows until each chunk fits the cache. | During prefill, an expert can be revisited across chunks. Compare expert-grouped processing that loads an expert once for all relevant rows in a bounded tile. |
| Fixed 128-slot per-layer caches; about 61–63 GiB observed per GPU. | Explore a larger measured budget and layer-specific allocation. Free memory is also needed for peak workspace and any future graphs/drafter. |
| Fresh scratch/ID allocations and metrics JSON writes every layer call. | Remove these from the timed hot path after measuring their contribution; retain buffered telemetry. |
| Eager execution, deterministic M16 kernel, speculation off. | These support the current controlled comparisons but are additional performance variables. A cache rewrite alone may not reach the GPU compute ceiling. |
| Expert and Engram data share storage/page-cache resources; download activity was not isolated. | Profile physical reads, page faults, Engram latency, CPU work, H2D, GPU idle gaps and TP collectives independently. |

The 22.7–24.6% miss figures combine prefill and decode, count unique experts per routing chunk, and include both TP ranks. They cannot be substituted for a decode-only physical-SSD miss probability. Likewise 1.35 TB staged is not 1.35 TB of measured SSD reads. The present measurements cannot rank CPU preparation, SSD, PCIe, Engram or synchronization by contribution to wall time.

## A useful bandwidth feasibility check

Illustrative calculation, not a performance prediction: for a decode token traversing 40 MoE layers with six experts each, hidden width 5120 and intermediate width 2304, the three FP4 expert matrices account for:

`40 × 6 × 3 × 5120 × 2304 × 0.5 = 4.247 GB/token`

This excludes scales and other traffic. TP2 splits these bytes between ranks; it does not halve their sum. At 150 tok/s without speculation or cross-token reuse, this is roughly 637 GB/s of logical expert weight consumption. A 10 GB/s SSD can contribute only 66.7 MB per output token at that rate: about 1.57% of those packed expert bytes. The remainder must be served through reuse/other memory tiers, or the amortization must change. Compute and other I/O reduce the available budget further.

This does not disprove the quoted engine. It explains why RAM caching, GPU residency, actual expert locality, speculation and workload definition are essential details. Prefetch can hide latency; it cannot remove sustained bandwidth requirements. Ask for the author's code/commit, cache sizes, decode-only physical read bytes per output token, speculative settings and benchmark traces before treating 150 tok/s as our expected result.

## Recommended next work, with gates

1. **Profile before redesign.** Measure one warm resident-layer case, forced-miss cases from RAM and disk, and full-model prefill/decode separately. Record routing traces with phase tags. Use CUDA events/NVTX plus OS/device I/O counters; keep CPU worker totals separate from elapsed time.
2. **Replay traces offline.** Compare existing LRU, frequency-aware retention, larger caches, and uneven per-layer budgets. Include phase transitions and different agent tasks. This can reject poor policies without repeatedly taking the GPUs from other models.
3. **Build one prepared-record and buffer-pool experiment.** Compare repeated native loading against persistent indexed records; measure byte size and CPU/H2D/preparation costs. Validate against the original checkpoint and the resident GPU oracle before full-model integration.
4. **Redesign prefill and overlap.** Evaluate expert-grouped tiling and available-expert-first execution with explicit CUDA event lifetimes. Preserve contribution/accumulation semantics. Both ranks must reach compatible TP collectives despite different load completion times.
5. **Only then add predictive prefetch, graphs or DSpark.** Demand work must have priority and bounded speculative bytes. Measure useful/timely/wasted prefetch and actual tokens accepted per verification step. Keep each optimization independently switchable.
6. **Qualification.** Repeat forced-eviction exact tests; add full-model reference/logit comparisons. Then measure fresh prompts, warm follow-up turns, cold/warm expert caches and concurrent agent workloads. Four successful coding fixtures remain a smoke test, not broad agent qualification.

Recommendation: retain the current prototype as the test oracle, borrow proven cache/transfer scheduling ideas, and start with profiling plus trace replay. Do not undertake another broad engine rewrite or buy hardware based on an unverified speed target.
