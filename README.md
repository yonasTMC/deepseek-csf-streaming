# CSF expert streaming for DeepSeek-V4.1-Flash on 2× RTX PRO 6000

A vLLM plugin that serves **DeepSeek-V4.1-Flash-1017** (the lossless-CSF checkpoint, about 460 GB with
Engram tables) on **one machine with two 96 GB RTX PRO 6000 Blackwell GPUs, 124 GiB of RAM and two NVMe
drives** — far less memory than the model.

The routed experts do not fit in VRAM or RAM, so the engine keeps a per-layer VRAM slot cache and a
pinned RAM tier, and streams the rest from NVMe on demand. Three constraints it holds to:

- **Weights:** the released MXFP4 expert weights are used unchanged. There is no requantization.
- **Compute:** all expert math runs on the GPUs.
- **Parallelism:** tensor parallel across the two GPUs (TP2).

It started from a 1 tok/s baseline and now decodes at 88–95 tok/s on code.

> Built on [DeepSeek-V4.1-Flash-1017-lossless-CSF](https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF)
> by Local Inference Lab, Inc. (Local Inference Lab License 1.0). Development and **every measurement
> below used the abliterated derivative**
> ([`jcartu/DeepSeek-V4.1-Flash-1017-abliterated-lossless-CSF`](https://huggingface.co/jcartu/DeepSeek-V4.1-Flash-1017-abliterated-lossless-CSF)),
> not the original weights. The checkpoint is not included in this repository.
>
> The server exposes the checkpoint under a model id you choose with `--served-model-name`. Use
> whatever id your endpoint reports.

## Contents

- [Should you run this?](#should-you-run-this)
- [Results](#results)
- [Quick start](#quick-start)
- [Settings that matter](#settings-that-matter)
- [How it works](#how-it-works)
- [Glossary](#glossary)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)
- [Tools](#tools)
- [Tests](#tests)
- [Known limitations](#known-limitations)
- [License and credits](#license-and-credits)

## Should you run this?

| Requirement | Detail |
|---|---|
| GPUs | 2× RTX PRO 6000 Blackwell, 96 GB each (sm_120). TP2 is the only supported layout. |
| Host RAM | About 124 GiB, with nothing else heavy running. The engine pins about 77 GiB. |
| Storage | Two NVMe drives, about 150 GB free each for the expert pack. |
| Runtime | Docker with the NVIDIA runtime (dev path), or Kubernetes with a 2-GPU claim (deploy path). |
| Checkpoint | Downloaded separately; accept its license on Hugging Face. |
| Time | About 8 minutes to build the pack; the image build compiles sm_120 extensions. |

Not a fit if you have one GPU, no NVMe, or want more than about 92 tok/s of total throughput from
this hardware — see [Known limitations](#known-limitations).

## Results

Measured on the target machine, idle server, single request unless noted. Final build: split pack +
deterministic M16 kernel + progressive fill acknowledgement + per-layer slot counts + learned
placement + 4,400-expert RAM tier + 4 sequences.

| | |
|---|---|
| Decode, code (T=0, thinking off) | 88–95 tok/s |
| Decode, thinking mode (T=1) | 76–81 tok/s |
| Decode, prose | 65–70 tok/s |
| 4 concurrent requests | about 92 tok/s total, about 23 per stream, all starting within about 1 s |
| Prefill, real source code, about 50k-token prompts | about 1,810–1,871 tok/s |
| Context | 1M tokens (2 GiB fp8 KV cache) |
| Long-prompt recall (passphrase at 2k / 8k / 25k tokens) | exact |

**Speculative decoding:** DSpark with 3 draft tokens and adaptive verification, at 2–3.4 tokens per
step depending on content.

**Speed depends on content.** Prose and reasoning route to experts that are less often resident than
code, and accept fewer draft tokens.

**Not reached:** the original targets of about 150 tok/s single-stream decode and 4k tok/s prefill.

[RESULTS.md](RESULTS.md) is the source of truth: every measurement, in order, with the configuration
it was taken under. The headline numbers above are its final section. [docs/DESIGN.md](docs/DESIGN.md)
explains how the engine works, the bugs found on the way, and the open problems.

## Quick start

These steps assume a node that meets the requirements above.

1. **Get the checkpoint** from its original location and accept its license there.

2. **Build the split expert pack.** It takes about 8 minutes and writes about 150 GB to each drive.
   Run it inside the image (see `scripts/devctl.sh`). The pack re-encodes the checkpoint's weights,
   so keep it on machines you control: the Local Inference Lab License forbids re-uploading it.

   ```bash
   python -m torch.distributed.run --nproc-per-node=2 scripts/pack_experts.py \
     --root <checkpoint snapshot> --out /pack/nvme1/split /pack/nvme2/split --split 2
   ```

3. **Start a server.** `scripts/serve-v4.sh` starts one on `127.0.0.1:8008`. The model id it
   reports is set with `--served-model-name`.

   ```bash
   MODEL=<checkpoint snapshot> PACK_DIR1=<drive 1 dir> PACK_DIR2=<drive 2 dir> bash scripts/devctl.sh start
   PACK_DIRS=/pack/nvme1/split:/pack/nvme2/split PLACEMENT=placement-learned-20261024.json \
     RAM_EXPERTS=4400 HOST_CACHE=160 PREFETCH_WIDTH=0 SLOTS=208 MAX_BATCHED=4096 \
     GRAPH_TOKENS=16 MAX_SEQS=4 CAPTURE=16 bash scripts/serve-v4.sh
   ```

   Optional: for Kubernetes, build `Dockerfile` from a clean checkout
   (`git archive HEAD | tar -x -C <dir>`), then adapt `deploy/k8s/statefulset.example.yaml`. That
   example manifest documents every setting with its measured justification and is the best
   configuration reference in the repository. It serves on port 8000.

4. **Send a request.** The endpoint is OpenAI-compatible:

   ```bash
   curl -s http://127.0.0.1:8008/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{
       "model": "<served model name>",
       "messages": [{"role": "user", "content": "What is 17 + 25? Reply with just the number."}],
       "temperature": 0,
       "max_tokens": 16,
       "chat_template_kwargs": {"thinking": false}
     }'
   ```

   From Python, with the OpenAI client (`pip install openai`):

   ```python
   from openai import OpenAI

   client = OpenAI(base_url="http://127.0.0.1:8008/v1", api_key="none")
   reply = client.chat.completions.create(
       model="<served model name>",
       messages=[{"role": "user", "content": "What is 17 + 25?"}],
       temperature=0,
       extra_body={"chat_template_kwargs": {"thinking": False}},
   )
   print(reply.choices[0].message.content)
   ```

   Set `chat_template_kwargs.thinking` to `true` for thinking mode, which decodes slower.

5. **Check it works.** `python3 scripts/smoke.py` asks `17 + 25` and must print `42`. For the
   checkpoint itself, `python3 scripts/verify_checkpoint.py <snapshot>` re-hashes the shards against
   `manifest.json`.

> **Which placement you point at changes the slot layout.** With
> `evidence/placement-learned-20261024.json` (no per-layer block) `SLOTS` sets a uniform count —
> 208 in the example above. With `evidence/placement-layered-20261024.json` the file's own
> `capacity` block wins and `SLOTS` is ignored: 128–304 per layer, 8,320 in total either way.

## Settings that matter

The rest are internal; this table is the set you would actually change. Defaults are the code's;
the example manifest overrides several of them.

| Setting | Default | What it does |
|---|---|---|
| `CSF_STREAMING`, `CSF_V4`, `CSF_GRAPH` | off | Turn on the plugin, the v4 engine, and full CUDA graphs. All three are required. |
| `CSF_PACK_DIRS` | — | One pack directory per TP rank, colon-separated. |
| `CSF_PLACEMENT` | — | Placement JSON from `evidence/`. A `capacity` block overrides `CSF_STREAM_SLOTS`. |
| `CSF_STREAM_SLOTS` | 128 | VRAM expert slots per layer and rank. 216 does not leave room for 4,096-token prefill activations; 208 does. |
| `CSF_RAM_EXPERTS` | all in placement | Pinned RAM-tier records per rank. 4,400 is about 77 GiB of host RAM across both ranks. |
| `CSF_HOST_CACHE` | 128 | Read-ahead buffers, 9.4 MB each. Prefill reads the next expert group ahead. |
| `CSF_GRAPH_TOKENS` | 8 | Token capacity of one decode graph. 16 covers 4 sequences × (1 + 3 draft tokens). |
| `CSF_MOE_KERNEL` | `auto` | `m16` forces the deterministic kernel. Split packs force it regardless. |
| `CSF_PREFETCH_WIDTH` | 6 | Next-layer decode prefetch from router votes. **Measured slower: 0 gives 82–83 tok/s versus 73–77 at width 6.** |
| `CSF_PREFILL_PREFETCH` | 1 | Prefill read-ahead from next-layer votes. Measured slower at every vote threshold; the example manifest sets 0. |
| `CSF_EXPERT_MAJOR_PREFILL` | 1 | Load each distinct expert at most once per layer per chunk. |
| `CSF_READ_CHUNKS` | 1 | Split a request's **only** NVMe read into N parallel 4 KiB-aligned chunks. See below. |
| `CSF_WAIT_TIMEOUT_S` | 60 | In-graph spin limit before the kernel flags an error and exits the process for a supervisor to restart. |
| `CSF_V4_TRACE` | — | Write routing ids (no text) to this path for `scripts/learn_placement.py`. |
| `CSF_STREAM_METRICS_DIR` | — | Write per-layer counters here. |

### `CSF_READ_CHUNKS` (newest change)

A request whose only NVMe read is a demand miss can split that read into parallel chunks; a buffer
becomes ready when its last chunk lands. `fio` on a real pack file, `O_DIRECT` + `io_uring`:

| Pattern | Throughput | Per-read completion, median |
|---|---|---|
| Whole 9.4 MB record, one in flight | 9.7 GB/s | 1.00 ms |
| 4 × 2.35 MB in parallel | 10.9 GB/s | 0.85 ms per chunk |
| 8 × 1.18 MB in parallel | 10.5 GB/s | 0.89 ms per chunk |
| 4 whole records in flight | 11.9 GB/s | 3.13 ms |

Chunks land together, so they offer no early prefix to copy, and only a lone read gains. In the dev
A/B against whole reads it changed nothing beyond noise — code 86–92 versus 88–95 tok/s, prose 63–68
versus 65–70, 4 streams 91 versus 90, prefill 1,860–1,884 versus 1,848–1,946 — so **the default stays
`1` (off)**. Set `CSF_READ_CHUNKS=4` only to experiment. Each chunk must be 4 KiB aligned, which the
engine checks at startup.

## How it works

Per routed layer and TP rank there are three tiers: **VRAM** slots (128–304 per layer, 8,320 total),
a **pinned RAM tier** (4,400 records per rank), and an **NVMe pack** (one 9.4 MB record per expert and
rank, stored in B12X's prepared slot layout, so a miss is a byte copy and never a repack).

```
              one decode step, inside one CUDA graph
   router ──► publish ──► device slot table ──► MoE kernel (deterministic, split halves)
                │                                     ▲
                │ miss list                           │ wait_fill, 45–48 GB/s
                ▼                                     │
        host-mapped mailbox ──► C++ service thread ───┘
                                  (no CUDA calls, no GIL)
                                  ├─ hit  ─► pinned RAM tier
                                  └─ miss ─► NVMe pack, O_DIRECT, 12 readers
```

- **Inside the graph:** `publish` maps routed expert ids to slots and posts misses to host-mapped
  memory. `wait_fill` copies each record from pinned host memory into VRAM as soon as the host marks
  it ready, so RAM-tier records copy while NVMe reads are still in flight. `finalize` installs the
  slot table.
- **On the host:** a C++ service thread runs the slot policy and issues `O_DIRECT` reads with 12
  reader threads, demand before read-ahead. It never calls CUDA and never takes the GIL.
- **Expert math:** each rank's expert slice is stored as **two half-width experts** and runs through
  B12X's **deterministic** MoE kernel. The default small-batch kernel is not reproducible at the full
  TP2 width, and it put stray tokens into greedy code output.
- **Prefill is expert-major:** each layer loads every distinct expert at most once per chunk, in
  groups that fit the free slots. Each group runs only its own (token, route) pairs while the next
  group is read ahead.
- **Concurrency:** up to 4 sequences in 16-token CUDA graphs. A step that needs more free slots than
  the LRU holds borrows the least recently used pinned slot.

## Glossary

**B12X** — the Blackwell kernel library in the Local Inference Lab image. **CSF** — the checkpoint's
lossless scale compression. **DSpark** — the speculative decoding scheme. **Engram** — the model's
~189 GiB disk-backed tables, a separate subsystem from expert streaming. **MXFP4** — the weights'
4-bit format. **M16** — the deterministic dynamic MoE kernel. **TP2** — tensor parallel over 2 GPUs.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Stray tokens in greedy output (`0都不要`, `0.02ASA`, `1inks`) | Auto-selected B12X small-batch kernel, not reproducible at width 1,152 | Use a split pack (`--split 2`) and `CSF_MOE_KERNEL=m16`. |
| Weights load but output is subtly wrong | `CHECKPOINT=csf` not set, so scales are misread silently | Set `CHECKPOINT=csf`. |
| Process exits mid-generation | In-graph spin exceeded `CSF_WAIT_TIMEOUT_S` (60 s) | Check NVMe saturation and host RAM; the service exits deliberately so a supervisor restarts it. |
| OOM at startup with long prefill | `CSF_STREAM_SLOTS=216` leaves no room for 4,096-token activations | Use 208. |
| Container fails at startup, Engram or io_uring errors | `seccompProfile: Unconfined`, `IPC_LOCK`, and `ulimit -l unlimited` are all required | See the example manifest. |
| Host OOM kill | RAM tier plus everything else exceeds the machine | Lower `CSF_RAM_EXPERTS` or free the node; 13 GiB was left available at 4,400. |
| Prefill looks far below the numbers here | Random-word prompts touch nearly every expert and overstate prefill several-fold | Measure on real source code. |

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
| `scripts/` | Pack builder, server scripts, and correctness and benchmark tools (below) |
| `bench/` | Dev-server helpers (restart, exact reference, logprob matrix) |
| `evidence/` | Placements, the junk-token investigation outputs, logprob run logs, v1 validation records |

## Tools

**Correctness and validation**

| Script | Purpose |
|---|---|
| `scripts/smoke.py` | One arithmetic request; the fastest "is it alive and sane" check |
| `scripts/verify_checkpoint.py` | Re-hash checkpoint shards against `manifest.json` |
| `scripts/validate_gpu.py`, `scripts/validate_graph.py` | Bit-exact cache checks against the fully resident reference |
| `scripts/validate_cpu_storage.py` | Independent CPU check of TP slices and decoded scale values |
| `scripts/kernel_race.py`, `scripts/decode_cfgs.py`, `scripts/split_check.py` | MoE kernel reproducibility, speed, and split-layout checks |
| `scripts/logprob_eval.py` | Teacher-forced next-token distributions; compares engines position by position |
| `scripts/test_mailbox.py`, `scripts/test_mailbox_sync.py` | Mailbox handshake, with and without the model |
| `scripts/imgtest.py` | Image input end to end |

**Speed and behaviour**

| Script | Purpose |
|---|---|
| `scripts/junk_test.py` | Greedy coding tasks; flags stray tokens in generated code |
| `scripts/needle.py`, `scripts/real_prefill.py` | Long-prompt recall and prefill speed on real source code (random text overstates prefill several-fold) |
| `scripts/decode_mix.py`, `scripts/decode_speed.py`, `scripts/concurrency.py` | Decode speed by content type and concurrency |
| `scripts/long_context.py`, `scripts/long_decode.py` | Speed at long context; decode after a long real-code prompt |
| `scripts/agent_benchmark.py` | Tool-using coding sessions in networkless containers; wall time and pass rate |
| `scripts/bench_fill.py`, `scripts/bench_moe_kernel.py` | GPU fill bandwidth; one layer's MoE kernel in isolation |
| `scripts/read_bench.py` | `O_DIRECT` read latency of expert records, whole versus chunked |

**Placement and simulation**

| Script | Purpose |
|---|---|
| `scripts/learn_placement.py` | Learns a placement from recorded route traces, and replays held-out traces against placements |
| `scripts/layer_budget.py` | Chooses per-layer slot counts at a fixed total by replaying held-out decode traffic |
| `scripts/simulate_cache.py` | Replays a trace against VRAM/RAM/SSD tiers and reports implied transfer time |
| `scripts/predict_eval.py` | Measures how well layer L's MoE input predicts the experts of layers L+d |

The server records routing ids (no text) to `CSF_V4_TRACE` when it is set;
`scripts/learn_placement.py` reads those files.

## Tests

```bash
python3 -m unittest discover -s tests -t tests
```

Seven tests, covering cache eviction and padding, routing chunks, and streamed tool-call
reconstruction. The `-t tests` argument is required — `tests/` has no `__init__.py`, so the usual
`-t .` form fails with `Start directory is not importable`.

The GPU-side checks are the `validate_*` scripts above; they need a checkpoint and a GPU.

## Known limitations

- **Upstream kernel bugs are worked around, not fixed:** the noisy small-batch kernel at width 1,152
  and corruption on `-1` routes at width 576. Untested shapes could still misbehave.
- **Run-to-run noise beyond the MoE kernel is unexplained** (candidates: prefill group composition,
  the sparse-attention top-k fallback, all-reduce).
- **No comparison against an independent implementation** of the same checkpoint, and the abliterated
  model's own quality cost against the original weights is unmeasured.
- **Prefill gap:** about 1.8k tok/s against a 4k target.
- **Throughput is flat with concurrency.** More streams remove queueing; they do not reduce the
  per-step cost, which is bound by expert streaming.

[docs/DESIGN.md §8](docs/DESIGN.md) has the full list with the reasoning.

## License and credits

- **This repository's code:** [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attributions.
- **Model:** [DeepSeek-V4.1-Flash-1017-lossless-CSF](https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF)
  by Local Inference Lab, Inc., under the Local Inference Lab License 1.0, which is not included here.
  - Both the checkpoint and **the expert packs built from it must not be redistributed.**
  - Development used the abliterated derivative
    [`jcartu/DeepSeek-V4.1-Flash-1017-abliterated-lossless-CSF`](https://huggingface.co/jcartu/DeepSeek-V4.1-Flash-1017-abliterated-lossless-CSF).
  - Both are made from [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
    which is under MIT.
- **Serving stack:** the [Local Inference Lab](https://docker.local-inference-lab.ai) vLLM image. Its
  vLLM, B12X kernels, FlashInfer and DSpark components are under their own licenses, Apache-2.0 for
  the open-source parts. This repository extends that image and does not include it.
- **Chat encoder:** `vendor/deepseek_v41_encoding_1017.py` is DeepSeek's 1017 encoder, copied
  unchanged, Copyright (c) 2023 DeepSeek, under the MIT License
  ([LICENSES/MIT-DeepSeek.txt](LICENSES/MIT-DeepSeek.txt)).
- **Prior art:** [RESEARCH.md](RESEARCH.md) surveys the prior work this design draws on. It is a
  snapshot from October 23, 2026, written before engines v2–v4; read it for the survey, not the
  current design.
