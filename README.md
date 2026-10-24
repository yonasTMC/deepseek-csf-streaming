# CSF expert streaming for DeepSeek-V4.1-Flash on 2× RTX PRO 6000

This project is based on DeepSeek-V4.1-Flash-1017-lossless-CSF by Local Inference Lab, Inc., a non-profit organization, available at [https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF](https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF). DeepSeek-V4.1-Flash-1017-lossless-CSF is licensed under the Local Inference Lab License, Version 1.0.

A vLLM plugin that serves **DeepSeek-V4.1-Flash-1017** (the lossless-CSF checkpoint, about 460 GB with
Engram tables) on **one machine with two 96 GB RTX PRO 6000 Blackwell GPUs, 124 GiB of RAM and two NVMe
drives**. That is far less memory than the model. The routed experts do not fit in VRAM or RAM, so the
engine keeps a per-layer VRAM slot cache and a pinned RAM tier, and streams missing experts from NVMe.

Constraints the engine holds to:
- **Weights:** the released MXFP4 expert weights are used unchanged; there is no requantization.
- **Compute:** all expert math runs on the GPUs.
- **Parallelism:** tensor parallel across the two GPUs (TP2).

The work started from a 1 tok/s baseline and was tested with the abliterated variant
(`jcartu/DeepSeek-V4.1-Flash-1017-abliterated-lossless-CSF`).

## Results

Measured on the target machine with an idle server, single request unless noted.

| | |
|---|---|
| Decode, code (T=0, thinking off) | 88–95 tok/s |
| Decode, thinking mode (T=1) | 76–81 tok/s |
| Decode, prose | 65–70 tok/s |
| 4 concurrent requests | about 92 tok/s total (25–28 each), all starting within about 1 s |
| Prefill, real source code, about 50k-token prompts | about 1,810–1,870 tok/s |
| Context | 1M tokens (2 GiB fp8 KV holds about 1.9M) |
| Long-prompt recall (passphrase at 2k / 8k / 25k tokens) | exact |

**Speculative decoding:** DSpark with 3 draft tokens and adaptive verification, at 2–3.4 tokens per step depending on content.

**Speed depends on content:** prose and reasoning route to experts that are less often resident than code, and accept fewer draft tokens.

**Not reached:** the original targets of about 150 tok/s single-stream decode and 4k tok/s prefill.

See [RESULTS.md](RESULTS.md) for every measurement and how it was taken, and [docs/DESIGN.md](docs/DESIGN.md) for how the engine works, the bugs found on the way, and the known limitations.

## How it works, briefly

- **Tiers per layer and rank:**
  - **VRAM:** 128–304 expert slots per layer, sized by each layer's measured miss rate. Most are pinned to the experts that real traffic uses most; the rest are an LRU.
  - **RAM:** a pinned tier of 4,400 expert records per rank, ranked by frequency across all layers.
  - **NVMe:** an expert pack, one 9.4 MB record per expert and rank, stored in B12X's prepared slot layout, so a miss is a byte copy and never a repack.
- **Inside the CUDA graph:**
  - A `publish` kernel maps the routed expert ids to slots and posts misses to a host-mapped mailbox.
  - `wait_fill` copies each record from pinned host memory into VRAM (45–48 GB/s) as soon as the host marks it ready. RAM-tier records copy while NVMe reads are still in flight.
  - `finalize` installs the slot table.
- **On the host:** a C++ service thread runs the slot policy and issues `O_DIRECT` reads with 12 reader threads, demand reads before read-ahead. It never calls CUDA and never takes the GIL.
- **Expert math:** each rank's expert slice is stored as **two half-width experts** and runs through B12X's **deterministic** MoE kernel. The default small-batch kernel is not reproducible at the full TP2 width, and it put stray tokens into greedy code output.
- **Prefill is expert-major:** each layer loads every distinct expert at most once per chunk, in groups that fit the free slots. Each group runs only its own (token, route) pairs while the next group is read ahead.
- **Concurrency:** up to 4 sequences in 16-token CUDA graphs. A step that needs more free slots than the LRU holds borrows the least recently used pinned slot.

## Repository layout

| Path | What |
|---|---|
| `csf_streaming/fastcache.py` | Slot cache, RAM tier, pack reader, split-pack MoE calls (`run_slots`, `run_pairs`), prefill groups |
| `csf_streaming/csrc/v4.cu` | `wait_fill` / `finalize` kernels and the C++ host service (slot policy, soft pinning, `O_DIRECT` readers) |
| `csf_streaming/csrc/mailbox.cu` | `publish` kernel and host-mapped mailbox helpers |
| `csf_streaming/integration.py` | Replaces vLLM's MXFP4-CSF MoE method for the 40 routed layers (DSpark draft layers stay resident) |
| `csf_streaming/storage.py` | CSF reads, B12X weight preparation, slot allocation |
| `csf_streaming/cache.py`, `policy.py` | First engine (eager, CPU preparation per miss), kept as the bit-exact correctness baseline |
| `sitecustomize.py`, `csf_streaming/prebuild.py` | Install the plugin at interpreter start; prebuild the extensions in the image |
| `patches/deepseek_v41_images.py` | Build-time patch so image messages reach the 1017 encoder as content blocks |
| `vendor/deepseek_v41_encoding_1017.py` | DeepSeek's 1017 chat encoder (MIT, unchanged), needed for tool calls |
| `Dockerfile` | Pinned Local Inference Lab vLLM image + this code + prebuilt sm_120 extensions |
| `deploy/k8s/statefulset.example.yaml` | Example Kubernetes deployment with every engine setting explained |
| `scripts/` | Pack builder, dev server, correctness and benchmark tools (below) |
| `bench/` | Dev-server helpers (restart, exact reference, logprob matrix) |
| `evidence/` | Placements, the junk-token investigation outputs, logprob run logs, v1 validation records |

## Running it

The machine needs 2× 96 GB Blackwell GPUs, about 124 GiB of RAM with nothing else heavy running, and two NVMe drives with about 150 GB free each for the pack. The engine pins about 77 GiB of host RAM.

**0. Get the checkpoint** from its original location and accept its license there. This repository does not include it.

**1. Build the split expert pack.** This takes about 8 minutes and writes about 150 GB to each drive. Run it inside the image (see `scripts/devctl.sh`). The pack re-encodes the checkpoint's weights, so keep it on machines you control; the Local Inference Lab License forbids re-uploading it.

```bash
python -m torch.distributed.run --nproc-per-node=2 scripts/pack_experts.py \
  --root <checkpoint snapshot> --out /pack/nvme1/split /pack/nvme2/split --split 2
```

**2a. Dev server** (`127.0.0.1:8008`, model name `DeepSeek-V4.1-Flash-streaming`):

```bash
MODEL=<checkpoint snapshot> PACK_DIR1=<drive 1 dir> PACK_DIR2=<drive 2 dir> bash scripts/devctl.sh start
PACK_DIRS=/pack/nvme1/split:/pack/nvme2/split PLACEMENT=placement-learned-20261024.json \
  RAM_EXPERTS=4400 HOST_CACHE=160 PREFETCH_WIDTH=0 SLOTS=208 MAX_BATCHED=4096 \
  GRAPH_TOKENS=16 MAX_SEQS=4 CAPTURE=16 bash scripts/serve-v4.sh
```

**2b. Image and Kubernetes.** Build `Dockerfile` from a clean checkout (`git archive HEAD | tar -x -C <dir>`), then adapt `deploy/k8s/statefulset.example.yaml`.

## Tools

| Script | Purpose |
|---|---|
| `scripts/junk_test.py` | Greedy coding tasks; flags stray tokens in generated code |
| `scripts/needle.py`, `scripts/real_prefill.py` | Long-prompt recall and prefill speed on real source code (random text overstates prefill several-fold) |
| `scripts/logprob_eval.py` | Teacher-forced next-token distributions; compares engines position by position |
| `scripts/decode_mix.py`, `scripts/decode_speed.py`, `scripts/concurrency.py` | Decode speed by content type and concurrency |
| `scripts/learn_placement.py` | Learns a placement from recorded route traces, and replays held-out traces against placements |
| `scripts/layer_budget.py` | Chooses per-layer slot counts at a fixed total by replaying held-out decode traffic |
| `scripts/read_bench.py` | `O_DIRECT` read latency of expert records, whole versus chunked |
| `scripts/kernel_race.py`, `scripts/decode_cfgs.py`, `scripts/split_check.py` | MoE kernel reproducibility, speed and split-layout checks |
| `scripts/validate_gpu.py`, `scripts/validate_graph.py` | Bit-exact cache checks against the fully resident reference |

The server records routing ids (no text) to `CSF_V4_TRACE` when it is set; `learn_placement.py` reads those files.

## License and credits

- **This repository's code:** [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attributions.
- **Model:** [DeepSeek-V4.1-Flash-1017-lossless-CSF](https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF) by Local Inference Lab, Inc., under the Local Inference Lab License 1.0, which is not included here.
  - Development used the abliterated derivative [`jcartu/DeepSeek-V4.1-Flash-1017-abliterated-lossless-CSF`](https://huggingface.co/jcartu/DeepSeek-V4.1-Flash-1017-abliterated-lossless-CSF).
  - Both are made from [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), which is under MIT.
  - The checkpoint and the expert packs built from it must not be redistributed.
- **Serving stack:** the [Local Inference Lab](https://docker.local-inference-lab.ai) vLLM image. Its vLLM, B12X kernels, FlashInfer and DSpark components are under their own licenses, Apache-2.0 for the open-source parts. This repository extends that image and does not include it.
- **Chat encoder:** `vendor/deepseek_v41_encoding_1017.py` is DeepSeek's 1017 encoder, copied unchanged, Copyright (c) 2023 DeepSeek, under the MIT License ([LICENSES/MIT-DeepSeek.txt](LICENSES/MIT-DeepSeek.txt)).
- **Prior art:** [RESEARCH.md](RESEARCH.md) surveys the prior work this design draws on (snapshot from 2026-10-23).
