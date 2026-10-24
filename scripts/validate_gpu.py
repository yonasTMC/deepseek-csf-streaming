"""Compare streaming against the installed full-resident CSF GPU path on TP2."""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from csf_streaming.cache import ExpertCache
from csf_streaming.storage import scale_shapes, weight_plan, FIELDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--engine", choices=["v1", "v2"], default="v1",
                        help="v2 needs CSF_PACK_DIRS; it also streams layer+1 so "
                             "next-layer prefetch and the RAM tier are exercised")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 2
    from b12x.moe import fused_moe as fm
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig
    from vllm.model_executor.model_loader.mxfp4_csf_loader import read_mxfp4_csf_layer
    config = json.load(open(args.root + "/config.json"))
    config = config.get("text_config", config)
    h, n, e = config["hidden_size"], config["moe_intermediate_size"], config["n_routed_experts"]
    k = config["num_experts_per_tok"]
    activation, limit = config["hidden_act"], config.get("swiglu_limit")
    device = torch.device("cuda", rank)

    def resident(layer):
        scratch = [torch.empty(s, dtype=torch.uint8, device=device) for s in scale_shapes(e, h, n // 2)]
        weights = read_mxfp4_csf_layer(args.root, layer, num_experts=e, hidden_size=h,
            intermediate_size=n, tp_rank=rank, tp_size=2, device=device,
            w13_scale_scratch=scratch[0], w2_scale_scratch=scratch[1])
        return fm.prepare_weights(plan=weight_plan(e, h, n // 2, activation, limit), weights=weights)

    layers = [args.layer] if args.engine == "v1" else [args.layer, args.layer + 1]
    references = {l: resident(l) for l in layers}
    if args.engine == "v1":
        caches = {args.layer: ExpertCache(args.root, args.layer, h, n, e, rank, 2, device,
                                          args.slots, 8, activation, limit)}
        resident_of = lambda c: c.policy.resident.items()
    else:
        from csf_streaming.fastcache import FastExpertCache
        placement = os.environ.get("CSF_PLACEMENT")
        if placement and not os.path.exists(placement) and rank == 0:
            # Small mixed placement: pinned VRAM experts and a RAM tier for both layers.
            json.dump(dict(vram={str(l): [0, 1, 2, 3] for l in layers},
                           ram={str(l): list(range(30, 200, 3)) for l in layers}), open(placement, "w"))
        dist.barrier()
        caches = {l: FastExpertCache(args.root, l, h, n, e, rank, 2, device, args.slots,
                                     activation, limit) for l in layers}
        resident_of = lambda c: c.slot_of.items()
    checks = []
    trials = [(1, 0), (1, 0), (3, 40), (7, 80), (1, 0), (17, 240), (128, 340), (1, -1)]
    if args.engine == "v2":
        trials += [(1, 5 * i) for i in range(1, 40)] + [(2, 7 * i) for i in range(20)] + [(64, 101), (1, 33)]
    runs = [(tokens, offset, layer) for tokens, offset in trials for layer in layers]
    for trial, (tokens, offset, layer) in enumerate(runs):
        reference, cache = references[layer], caches[layer]
        torch.manual_seed(7300 + trial)
        x = torch.randn(tokens, h, dtype=torch.bfloat16, device=device)
        ids = ((torch.arange(tokens * k, device=device).reshape(tokens, k) + offset) % e).int()
        gates = torch.softmax(torch.randn(tokens, k, device=device), -1)
        if tokens == 128 or offset == -1:
            ids[-1] = -1
            gates[-1] = 0
        plan = fm.plan(fm.Caps(max_tokens=tokens, num_topk=k, device=device,
                              weight_plan=reference._impl.plan, quant_mode="w4a8_mx", deterministic_output=True,
                    decode_config=MoeDecodeConfig(backend="dynamic", route_planner="internal", max_active_clusters=None, dynamic_tile_m=16, dynamic_route_mode="grouped"), swiglu_limit=limit))
        ws = torch.empty(sum(s.nbytes for s in plan.scratch_specs()), dtype=torch.uint8, device=device)
        expected = torch.empty_like(x)
        fm.run(binding=fm.bind(plan, scratch=ws, a=x, experts=reference._impl,
                              topk_weights=gates, topk_ids=ids, output=expected, input_scales_static=True))
        actual = cache.run(x, gates, ids)
        for c in caches.values():
            if hasattr(c, "pending"):
                c.resolve(list(c.pending))  # finish in-flight prefetches before comparing slots
        torch.cuda.synchronize()
        # Independent resident loading verifies all eight prepared tensors byte-for-byte.
        for expert, slot in resident_of(cache):
            for field in FIELDS:
                a = getattr(cache.prepared._impl, field)[slot]
                b = getattr(reference._impl, field)[expert]
                if not torch.equal(a, b):
                    raise AssertionError(f"rank {rank}: {field} differs for expert {expert}")
        local_error = (actual.float() - expected.float()).abs().max().item()
        if not args.diagnostic:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        dist.all_reduce(actual)
        dist.all_reduce(expected)
        if not args.diagnostic:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        checks.append(dict(trial=trial, layer=layer, tokens=tokens, local_max_abs_error=local_error,
                           reference_config=str(plan.caps.decode_config), cache_configs=[str(p.caps.decode_config) for p in cache.plans.values()],
                           tp2_max_abs_error=(actual.float()-expected.float()).abs().max().item()))
    for c in caches.values():
        c.report()
    data = dict(diagnostic_only=args.diagnostic, rank=rank, layer=args.layer, engine=args.engine, checks=checks,
                stats={l: dict(c.stats) for l, c in caches.items()},
                peak_cuda_bytes=torch.cuda.max_memory_allocated())
    with open(args.output + f"-rank{rank}.json", "w") as f:
        json.dump(data, f, indent=2)
    print(json.dumps(data), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
