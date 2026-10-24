"""Exactness of graph-mode streaming (GPU mailbox + host service thread), TP2.

For two consecutive layers, each routed call maps experts to slots through
FastExpertCache.graph_slots (eagerly and inside a captured CUDA graph) and runs
the B12X kernel on the streamed slots. The expected output runs the same plan
on a mirror slot set filled from the fully resident reference experts at the
slots the call used, so any data-movement or table error shows up bit-exactly.
"""
import argparse
import json
import os

import torch
import torch.distributed as dist

from csf_streaming.storage import (FIELDS, allocate_slots, read_experts, scale_shapes,
                                   upload_and_prepare, weight_plan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    from b12x.moe import fused_moe as fm
    from vllm.model_executor.model_loader.mxfp4_csf_loader import read_mxfp4_csf_layer
    from csf_streaming.fastcache import FastExpertCache
    config = json.load(open(a.root + "/config.json"))
    config = config.get("text_config", config)
    h, n, e, k = (config["hidden_size"], config["moe_intermediate_size"],
                  config["n_routed_experts"], config["num_experts_per_tok"])
    limit = config.get("swiglu_limit")
    device = torch.device("cuda", rank)
    layers = [a.layer, a.layer + 1]
    placement = os.environ["CSF_PLACEMENT"]
    if rank == 0 and not os.path.exists(placement):
        json.dump(dict(vram={str(l): [0, 1, 2, 3] for l in layers},
                       ram={str(l): list(range(30, 200, 3)) for l in layers}), open(placement, "w"))
    dist.barrier()

    def resident(layer):
        scratch = [torch.empty(s, dtype=torch.uint8, device=device) for s in scale_shapes(e, h, n // 2)]
        weights = read_mxfp4_csf_layer(a.root, layer, num_experts=e, hidden_size=h, intermediate_size=n,
                                       tp_rank=rank, tp_size=2, device=device,
                                       w13_scale_scratch=scratch[0], w2_scale_scratch=scratch[1])
        return fm.prepare_weights(plan=weight_plan(e, h, n // 2, "silu", limit), weights=weights)

    refs = {l: resident(l) for l in layers}
    caches = {l: FastExpertCache(a.root, l, h, n, e, rank, 2, device, a.slots, "silu", limit) for l in layers}
    def template(layer):
        weights, _, _ = read_experts(a.root, layer, [0], h, n, rank, 2)
        return upload_and_prepare(weights, h, n // 2, device, limit=limit)

    mirrors = {l: allocate_slots(template(l), a.slots) for l in layers}

    def expected(layer, x, gates, ids, slots):
        mirror, ref = mirrors[layer], refs[layer]
        for expert, slot in zip(ids.reshape(-1).tolist(), slots.reshape(-1).tolist()):
            if expert >= 0:
                if slot < 0:
                    raise AssertionError(f"layer {layer}: expert {expert} unmapped after wait")
                for f in FIELDS:
                    getattr(mirror._impl, f)[slot].copy_(getattr(ref._impl, f)[expert])
        plan, scratch = caches[layer].plan_for(x.shape[0], k, device)
        out = torch.empty_like(x)
        fm.run(binding=fm.bind(plan, scratch=scratch, a=x, experts=mirror._impl, topk_weights=gates,
                               topk_ids=slots, output=out, input_scales_static=True))
        return out

    def streamed(layer, x, gates, ids, out):
        cache = caches[layer]
        slots = cache.graph_slots(ids, x if x.shape[0] <= cache.prefetch_tokens else None)
        plan, scratch = cache.plan_for(x.shape[0], k, device)
        fm.run(binding=fm.bind(plan, scratch=scratch, a=x, experts=cache.prepared._impl,
                               topk_weights=gates, topk_ids=slots, output=out, input_scales_static=True))
        return slots

    def inputs(seed, tokens, offset):
        g = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(tokens, h, dtype=torch.bfloat16, device=device, generator=g)
        ids = ((torch.arange(tokens * k, device=device).reshape(tokens, k) + offset) % e).int()
        gates = torch.softmax(torch.randn(tokens, k, device=device, generator=g), -1)
        return x, ids, gates

    checks = []

    def check(kind, layer, x, gates, ids, actual, slots):
        torch.cuda.synchronize()
        want = expected(layer, x, gates, ids, slots)
        torch.cuda.synchronize()
        err = (actual.float() - want.float()).abs().max().item()
        if err:
            cache, ref = caches[layer], refs[layer]
            bad = []
            for expert, slot in zip(ids.reshape(-1).tolist(), slots.reshape(-1).tolist()):
                for f in FIELDS:
                    if not torch.equal(getattr(cache.prepared._impl, f)[slot], getattr(ref._impl, f)[expert]):
                        bad.append((expert, slot, f, expert in cache.pinned, cache.slot_of.get(expert)))
            plan, scratch = cache.plan_for(x.shape[0], k, device)
            again = []
            for experts in (cache.prepared, mirrors[layer], cache.prepared):
                o = torch.empty_like(x)
                fm.run(binding=fm.bind(plan, scratch=scratch, a=x, experts=experts._impl, topk_weights=gates,
                                       topk_ids=slots, output=o, input_scales_static=True))
                torch.cuda.synchronize()
                again.append(o)
            diff = lambda p, q: (p.float() - q.float()).abs().max().item()
            print(f"[rank{rank}] rerun: streamed-now vs want {diff(again[0], want)}, vs actual {diff(again[0], actual)}; "
                  f"mirror-now vs want {diff(again[1], want)}; streamed twice {diff(again[0], again[2])}", flush=True)
            print(f"[rank{rank}] {kind} layer {layer} tokens {x.shape[0]} trial {len(checks)} err {err} "
                  f"slot mismatches {bad[:12]} ids {ids.tolist()} slots {slots.tolist()} "
                  f"stats {dict(cache.stats)}", flush=True)
        torch.testing.assert_close(actual, want, rtol=0, atol=0)
        dist.all_reduce(actual)
        dist.all_reduce(want)
        torch.testing.assert_close(actual, want, rtol=0, atol=0)
        checks.append(dict(kind=kind, layer=layer, tokens=x.shape[0], max_abs_error=err))

    # Eager calls: misses from RAM and NVMe, evictions, prefetch into layer+1.
    trials = [(1, 0), (1, 0), (3, 40), (1, 5), (2, 77), (8, 120), (1, 210)] + [(1, 9 * i) for i in range(40)]
    for t, (tokens, offset) in enumerate(trials):
        for layer in layers:
            x, ids, gates = inputs(1000 * t + layer, tokens, offset + layer)
            out = torch.empty_like(x)
            slots = streamed(layer, x, gates, ids, out)
            check("eager", layer, x, gates, ids, out, slots)

    # Captured graph: both layers' mailbox + kernel in one graph, replayed with new routes.
    static = {l: inputs(7 + l, 1, 0) for l in layers}
    outs = {l: torch.empty_like(static[l][0]) for l in layers}
    slot_out = {}
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for l in layers:  # warm the plans outside capture
            streamed(l, static[l][0], static[l][2], static[l][1], outs[l])
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for l in layers:
            x, ids, gates = static[l]
            slot_out[l] = streamed(l, x, gates, ids, outs[l])
    for r in range(60):
        for l in layers:
            x, ids, gates = inputs(50000 + 10 * r + l, 1, 13 * r + 3 * l)
            static[l][0].copy_(x)
            static[l][1].copy_(ids)
            static[l][2].copy_(gates)
        graph.replay()
        for l in layers:
            check("graph", l, static[l][0], static[l][2], static[l][1], outs[l].clone(), slot_out[l].clone())

    from csf_streaming import fastcache
    stats = {l: (dict(fastcache.v4_ext().stats(l)) if fastcache.V4 else dict(c.stats)) for l, c in caches.items()}
    data = dict(rank=rank, layer=a.layer, checks=len(checks),
                max_abs_error=max(c["max_abs_error"] for c in checks),
                graph_checks=sum(c["kind"] == "graph" for c in checks), stats=stats)
    json.dump(data, open(f"{a.output}-rank{rank}.json", "w"), indent=2)
    print(json.dumps(data), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
