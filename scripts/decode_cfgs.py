"""Speed and accuracy of B12X MoE decode configurations (one GPU, real experts in 208 slots).

For each config and token count: microseconds per call (CUDA graph of 20 calls), max relative
error vs the deterministic dynamic-M16 reference, and run-to-run spread over repeated calls.
"""
import argparse, json
import torch
from csf_streaming.storage import FIELDS, allocate_slots, read_experts, upload_and_prepare

CONFIGS = {
    "auto": {},
    "auto+det": dict(deterministic_output=True),
    "micro": dict(backend="micro"),
    "micro+det": dict(backend="micro", deterministic=True),
    "dyn16-grouped-det": dict(backend="dynamic", tile=16, mode="grouped", deterministic=True),
    "dyn16-grouped": dict(backend="dynamic", tile=16, mode="grouped"),
    "dyn16-direct-det": dict(backend="dynamic", tile=16, mode="direct", deterministic=True),
    "dyn8-grouped-det": dict(backend="dynamic", tile=8, mode="grouped", deterministic=True),
    "dyn32-grouped-det": dict(backend="dynamic", tile=32, mode="grouped", deterministic=True),
    "dyn16-grouped-det-triton": dict(backend="dynamic", tile=16, mode="grouped", planner="triton", deterministic=True),
}


def caps_extra(spec):
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig
    if not spec or "backend" not in spec:
        return dict(spec)
    extra = {}
    if spec.get("deterministic"):
        extra["deterministic_output"] = True
    kw = dict(backend=spec["backend"], route_planner=spec.get("planner", "internal"), max_active_clusters=None)
    if spec["backend"] == "dynamic":
        kw.update(dynamic_tile_m=spec["tile"], dynamic_route_mode=spec["mode"])
    extra["decode_config"] = MoeDecodeConfig(**kw)
    return extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--tp", type=int, default=2, help="expert slice width = 2304 / tp (TP4 = production LiL)")
    ap.add_argument("--configs", nargs="+", default=None)
    a = ap.parse_args()
    from b12x.moe import fused_moe as fm
    dev = torch.device("cuda", 0)
    h, n, k, E = 5120, 2304, 6, 208
    w, _, _ = read_experts(a.root, 20, list(range(64)), h, n, 0, a.tp)
    real = upload_and_prepare(w, h, n // a.tp, dev, limit=10.0)
    tw, _, _ = read_experts(a.root, 20, [0], h, n, 0, a.tp)
    slots = allocate_slots(upload_and_prepare(tw, h, n // a.tp, dev, limit=10.0), E)
    for f in FIELDS:
        dst, s = getattr(slots._impl, f), getattr(real._impl, f)
        for i in range(E):
            dst[i].copy_(s[i % 64])
    results = {}
    for T in a.tokens:
        g = torch.Generator(device=dev).manual_seed(T)
        x = (torch.randn(T, h, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        ids = torch.stack([torch.randperm(E, device=dev, generator=g)[:k] for _ in range(T)]).int()
        gates = torch.softmax(torch.randn(T, k, device=dev, generator=g), -1)
        ref = None
        names = a.configs or [c for c in CONFIGS if c != "dyn16-grouped-det"]
        for name in ["dyn16-grouped-det"] + [c for c in names if c != "dyn16-grouped-det"]:
            try:
                plan = fm.plan(fm.Caps(max_tokens=T, num_topk=k, device=dev, weight_plan=slots._impl.plan,
                                       quant_mode="w4a8_mx", swiglu_limit=10.0, **caps_extra(CONFIGS[name])))
                ws = torch.empty(sum(s.nbytes for s in plan.scratch_specs()), dtype=torch.uint8, device=dev)
                y = torch.empty_like(x)
                b = fm.bind(plan, scratch=ws, a=x, experts=slots._impl, topk_weights=gates, topk_ids=ids, output=y, input_scales_static=True)
                outs = []
                for _ in range(a.repeats):
                    fm.run(binding=b); torch.cuda.synchronize(); outs.append(y.float().clone())
                if ref is None:
                    ref = outs[0]
                scale = ref.abs().max()
                err = max(((o - ref).abs().max() / scale).item() for o in outs)
                spread = max(((o - outs[0]).abs().max() / scale).item() for o in outs)
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    for _ in range(20):
                        fm.run(binding=b)
                torch.cuda.synchronize()
                s0, s1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s0.record()
                for _ in range(10):
                    gr.replay()
                s1.record(); torch.cuda.synchronize()
                us = s0.elapsed_time(s1) / 200 * 1000
                results[f"{name}/T{T}"] = dict(us=round(us, 1), err_vs_ref=f"{err:.1e}", run_to_run=f"{spread:.1e}")
            except Exception as e:  # noqa: BLE001 - report unsupported combinations
                results[f"{name}/T{T}"] = dict(unsupported=str(e)[:90])
            print(f"T={T} {name:26s} {results[f'{name}/T{T}']}", flush=True)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
