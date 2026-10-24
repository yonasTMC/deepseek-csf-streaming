# Design notes

How the engine works, why it ended up this way, what has and has not been verified, and the open problems. Numbers are measured unless stated otherwise.

## 1. Problem

- **The model:** DeepSeek-V4.1-Flash-1017 has 40 routed MoE layers with 384 experts each and top-6 routing.
  - Each expert is 18.8 MB in MXFP4, or 9.4 MB per TP rank.
  - There are also 3 DSpark draft layers and Engram tables of about 189 GiB.
- **The machine:** 2× RTX PRO 6000 Blackwell (sm_120, 96 GB each), 124 GiB of RAM, two Samsung 9100 PRO NVMe drives.
- **The gap:** VRAM holds about 208 experts per layer per rank alongside everything else, and RAM a few thousand more. The rest has to come from NVMe on demand, inside a decode step of a few tens of milliseconds.

## 2. Engine generations

| Version | What it did | Outcome |
|---|---|---|
| v1 | Eager mode; each miss read CSF tensors, decoded scales and prepared B12X slots on the CPU | Bit-exact against the fully resident model; about 1 tok/s |
| v2 | Offline expert pack in prepared slot layout; RAM tier; CUDA graphs with an in-graph spin-wait served by a Python thread | Fast without speculation, but stalled under DSpark: helper-thread CUDA calls blocked in the driver while the kernel spun (Nsight Systems, py-spy) |
| v3 | Breakable piecewise graphs, no spin | About 30 tok/s, because attention also ran outside graphs |
| v4 | GPU-driven fills, plus a C++ host service that makes no CUDA calls and takes no GIL | Full graphs with DSpark: 77–83 tok/s before the fixes below |

**Negative results:**
- **Next-layer expert prefetch from router predictions:** slower in decode, and only 33% of prefill reads were used.
- **Larger prefill chunks with fewer slots:** slower.
- **Prefill benchmarks on random-word prompts:** these overstate prefill several-fold, because real prompts touch nearly every expert.

## 3. Engine v4

Per routed layer and rank:

1. **Tiers:** 208 VRAM slots (151 pinned, 56 LRU, 1 never lent), a RAM tier of 4,400 records per rank, and the NVMe pack. Each pack record is 9,404,416 bytes, 4 KiB-aligned, one per (rank, layer, expert), so a miss is a single `O_DIRECT` read.
2. **The mailbox:** inside the CUDA graph, `publish` maps expert ids to slots through a device table and posts the request to host-mapped memory. `wait_fill` polls with `ld.acquire.sys` and `__nanosleep`. After the host acknowledges, 24 blocks copy the records from pinned host memory with uncached loads at 45–48 GB/s. `finalize` installs the table entries.
3. **The host service** (`csrc/v4.cu`):
   - Reads requests and runs the slot policy (LRU, soft pinning).
   - Serves records from the RAM tier or from a read-ahead cache of 160 pinned buffers, using 12 `O_DIRECT` reader threads that serve demand misses before read-ahead.
   - Writes the acknowledgement.

   It never calls CUDA and never takes the GIL. If the GPU waits longer than `CSF_WAIT_TIMEOUT_S` (60 s), the kernel flags an error and returns, and the service exits the process so that a supervisor restarts it.
4. **Speculative decoding:** DSpark with 3 draft tokens and adaptive verification. CUDA graphs cover 1–16 tokens, which is up to 4 sequences of 4 tokens.
5. **Prefill (expert-major):** for each chunk of up to 4,096 tokens, the distinct experts load at most once per layer, in groups of up to 56 non-resident experts, so up to 5 groups per layer. Each group runs only its own (token, route) pairs (`run_pairs`) and accumulates in fp32. Its `publish` names the next group's experts, which the host reads ahead (prediction ids at or above 2^20 mean "this layer").

## 4. The junk-token bug and the split pack

**Symptom:** greedy code contained stray tokens such as `0都不要`, `0.02ASA` and `1inks`, in 2–5 of 7 code blocks, even at temperature 0.

**Cause:** B12X's automatically selected small-batch MoE kernel (1–4 tokens) is not reproducible at this model's TP2 slice width of 1,152.
- On identical inputs, its output differs by 2–3% of the maximum between calls, and by about 5% from the deterministic kernel (`scripts/kernel_race.py`, `scripts/decode_cfgs.py`).
- Zeroing buffers, changing the slot count and B12X's deterministic-output flag did not change this.
- At a 576-wide slice, the deterministic kernel is exact and about as fast. 576 is the width B12X's compact N64-tail path serves, the TP4 layout.

**Evidence** (`evidence/junk-tokens/`): every configuration that used the auto kernel produced junk, whether DSpark was on or off and with either prefill path. The bit-exact reference path and the deterministic kernel produced clean code. This rests on greedy decoding of three coding tasks, so it is strong evidence, not a proof.

**Fix:**
- `pack_experts.py --split 2` stores each rank's slice as two 576-wide half experts: slot `s` holds halves `2s` and `2s+1`.
- `run_slots` expands each route into its two halves with the same gate and calls the deterministic dynamic M16 kernel.
- One 4,096-token plan serves every batch size.

| Tokens | Auto, full width | Deterministic, full width | Deterministic, split |
|---|---|---|---|
| 1 | 20 µs | 389 µs | 56–68 µs |
| 4 | 148 µs | 422 µs | 160 µs |
| 8 | 313 µs | 465 µs | 307 µs |
| 4,096 | 2.2 ms | 8.4 ms | 4.4 ms |

The split result differs from the full-width deterministic result by 0.5–1%, and that difference is itself deterministic. Its cause has not been verified.

**A second kernel bug:** the half-width plan intermittently corrupts output on `-1` (masked) routes. It produced NaN at 1–8 tokens, and on one run but not the next a 63% error at 2,048 tokens. The full-width kernel handles `-1` exactly. The engine therefore never sends `-1`:
- Prefill compacts its routes per group.
- Decode replaces any masked route with the protected safe slot at gate 0.

Neither kernel bug has been reported upstream yet.

## 5. Concurrency and soft pinning

A 16-token step can route up to 96 distinct unpinned experts in one layer, more than the 56 LRU slots. When no LRU victim is free, the host service lends the least recently used pinned slot that the request does not use. That expert re-pins itself when it is loaded again, and the lowest-id pinned expert, which holds the safe slot, is never lent. One or two streams never need this, so single-stream speed is unaffected; in testing it was never triggered, `pinned_borrowed` = 0.

Total throughput stays at about 75–83 tok/s from 1 to 4 streams. Steps are bound by expert streaming, and each stream adds its own experts to every step. More sequences remove queueing; they do not reduce the per-step cost.

## 6. Placement

`scripts/learn_placement.py`:
- counts decode-sized calls (16 routed tokens or fewer) in route traces recorded by the server (`CSF_V4_TRACE`; routing ids only, no text);
- pins the 151 most frequent experts in each layer;
- fills the RAM tier with the most frequent remaining (layer, expert) pairs across all layers.

The deployed placement was learned from 3.36M decode calls of real agent traffic (code, prose, reasoning, tool calls). Replaying held-out traffic, pinned hits rose from 65.7% to 74.3%. Prose rose from about 43 to 65 tok/s and thinking from 64 to 78, while two code prompts lost speed. The old placement had been learned from benchmark prompts, which overfit it to exactly those prompts. Placement is static: relearn it when traffic changes.

## 7. Correctness evidence

- **Bit-exact cache checks** (v1, `evidence/v1-validation/`; `validate_graph.py` for later versions): streamed slots match the fully resident model byte for byte, and routed outputs match exactly, with the deterministic M16 kernel on both sides.
- **Prompt rendering:** for four agent conversations, the server's prompt was identical character for character to DeepSeek's official `deepseek-recipe` renderer. The conversations covered tool loops with and without thinking, reasoning carried across tool calls, and multi-turn chat.
- **Teacher-forced logprobs** (`scripts/logprob_eval.py`, 7,349 positions; RESULTS.md):
  - The stack has run-to-run noise beyond the MoE kernel. The deterministic reference differs from itself by mean KL 0.009, including flips of 95%-confident tokens.
  - Every engine variant falls within that noise floor of the reference.
  - The split engine is the most reproducible configuration measured (KL 0.0036 between identical runs).
- **End to end:** greedy code clean across the junk tests; passphrase recall exact at 2k, 8k and 25k tokens; image input works.

## 8. Known limitations and open questions

1. **Upstream kernel bugs are worked around, not fixed:** the noisy small-batch kernel at width 1,152, and corruption on `-1` routes at width 576. Untested shapes or route patterns could still misbehave; `split_check.py --masked` is the test to extend.
2. **The remaining run-to-run noise is unexplained.** Candidates: prefill group composition, the sparse-attention top-k fallback (the base image lacks `vllm._deepselect_C`), and the all-reduce.
3. **There is no comparison against an independent implementation** of the same checkpoint, and the abliterated model's own quality cost has not been measured against the original weights.
4. **The prefill gap:** about 1.8k tok/s against a 4k target. The split kernel's MoE time is twice the full-width kernel's, and the rest of the time per chunk (waiting for fills versus compute) has not been broken down.
5. **Throughput is flat with concurrency.** Raising total throughput needs fewer misses per step: better placement, more of the shared experts resident, or draft-guided prefetch.
6. **Host memory is tight.** The RAM tier trades directly against what else can run on the machine.
7. **Legacy code** still in the tree: `cache.py` (v1 baseline, used by `validate_gpu.py`), the v3 breakable-graph branch (`host_slots`, `_streamed_moe`) and the fake-hit timing path (`run_fake`, `device_slots`).
