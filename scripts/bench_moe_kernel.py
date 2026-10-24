"""Time one layer's routed-expert kernel for decode-sized batches (one GPU, TP rank 0 slice)."""
import argparse
import json

import torch

from csf_streaming.storage import allocate_slots, read_experts, upload_and_prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--slots", type=int, default=192)
    a = ap.parse_args()
    from b12x.moe import fused_moe as fm
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig
    dev = torch.device("cuda", 0)
    h, n, k = 5120, 2304, 6
    w, _, _ = read_experts(a.root, 10, [0], h, n, 0, 2)
    template = upload_and_prepare(w, h, n // 2, dev, limit=10.0)
    slots = allocate_slots(template, a.slots)
    for f in ("w1_fp4", "w2_fp4"):
        getattr(slots._impl, f).random_(0, 2**31 - 1)
    for f in ("w1_blockscale", "w2_blockscale"):
        getattr(slots._impl, f).fill_(0x7F7F7F7F)  # e8m0 scale 1.0 in every byte
    configs = {
        "dynamic_m16_deterministic": dict(deterministic_output=True, decode_config=MoeDecodeConfig(
            backend="dynamic", route_planner="internal", max_active_clusters=None,
            dynamic_tile_m=16, dynamic_route_mode="grouped")),
        "auto": dict(),
        "auto_deterministic": dict(deterministic_output=True),
    }
    out = {}
    for tokens in (1, 2, 4, 8):
        x = torch.randn(tokens, h, dtype=torch.bfloat16, device=dev)
        ids = torch.randint(0, a.slots, (tokens, k), device=dev, dtype=torch.int32)
        gates = torch.softmax(torch.randn(tokens, k, device=dev), -1)
        for name, extra in configs.items():
            try:
                plan = fm.plan(fm.Caps(max_tokens=tokens, num_topk=k, device=dev, weight_plan=slots._impl.plan,
                                       quant_mode="w4a8_mx", swiglu_limit=10.0, **extra))
            except Exception as e:  # noqa: BLE001 - report unsupported combinations
                out[f"{name}/t{tokens}"] = f"unsupported: {e}"[:120]
                continue
            ws = torch.empty(sum(s.nbytes for s in plan.scratch_specs()), dtype=torch.uint8, device=dev)
            y = torch.empty_like(x)
            b = fm.bind(plan, scratch=ws, a=x, experts=slots._impl, topk_weights=gates, topk_ids=ids,
                        output=y, input_scales_static=True)
            for _ in range(5):
                fm.run(binding=b)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(20):
                    fm.run(binding=b)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(10):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            out[f"{name}/t{tokens}"] = round(s.elapsed_time(e) / 200 * 1000, 1)
    print(json.dumps(dict(us_per_call=out)))


if __name__ == "__main__":
    main()
