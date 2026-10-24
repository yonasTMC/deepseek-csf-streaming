"""Is the auto-selected B12X MoE kernel racy? Repeat identical calls; compare with the deterministic M16 kernel.

Uses one real layer's experts (rank 0 slice) copied into `num_experts` slots; for each token count,
runs the auto plan N times on identical inputs and reports the max relative error vs the M16
reference and the spread between auto runs. Accumulation-order noise is ~1e-3 relative; a race shows
as occasional much larger errors.
"""
import argparse, json
import torch
from csf_streaming.storage import FIELDS, allocate_slots, read_experts, upload_and_prepare


AUTO_DET = False
ZERO = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--experts", type=int, nargs="+", default=[208, 384])
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--zero", action="store_true", help="zero scratch and output before every call")
    ap.add_argument("--auto-deterministic", action="store_true", help="plan the auto kernel with deterministic_output=True")
    a = ap.parse_args()
    global AUTO_DET, ZERO
    AUTO_DET, ZERO = a.auto_deterministic, a.zero
    from b12x.moe import fused_moe as fm
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig
    dev = torch.device("cuda", 0)
    h, n, k = 5120, 2304, 6
    src_ids = list(range(64))
    w, _, _ = read_experts(a.root, a.layer, src_ids, h, n, 0, 2)
    real = upload_and_prepare(w, h, n // 2, dev, limit=10.0)
    tmpl_w, _, _ = read_experts(a.root, a.layer, [0], h, n, 0, 2)
    template = upload_and_prepare(tmpl_w, h, n // 2, dev, limit=10.0)
    out = {}
    for E in a.experts:
        slots = allocate_slots(template, E)
        for f in FIELDS:  # tile the 64 real experts across all slots
            dst, s = getattr(slots._impl, f), getattr(real._impl, f)
            for i in range(E):
                dst[i].copy_(s[i % 64])
        for T in a.tokens:
            g = torch.Generator(device=dev).manual_seed(T)
            x = (torch.randn(T, h, device=dev, generator=g) * 0.5).to(torch.bfloat16)
            ids = torch.stack([torch.randperm(E, device=dev, generator=g)[:k] for _ in range(T)]).int()
            gates = torch.softmax(torch.randn(T, k, device=dev, generator=g), -1)
            def run(extra):
                plan = fm.plan(fm.Caps(max_tokens=T, num_topk=k, device=dev, weight_plan=slots._impl.plan,
                                       quant_mode="w4a8_mx", swiglu_limit=10.0, **extra))
                ws = torch.empty(sum(s.nbytes for s in plan.scratch_specs()), dtype=torch.uint8, device=dev)
                y = torch.empty_like(x)
                b = fm.bind(plan, scratch=ws, a=x, experts=slots._impl, topk_weights=gates, topk_ids=ids, output=y, input_scales_static=True)
                return b, y, ws
            mb, my, mws = run(dict(deterministic_output=True, decode_config=MoeDecodeConfig(backend="dynamic", route_planner="internal", max_active_clusters=None, dynamic_tile_m=16, dynamic_route_mode="grouped")))
            fm.run(binding=mb); torch.cuda.synchronize(); ref = my.float().clone()
            ab, ay, aws = run(dict(deterministic_output=True) if AUTO_DET else {})
            errs, runs = [], []
            for _ in range(a.repeats):
                if ZERO:
                    aws.zero_(); ay.zero_()
                fm.run(binding=ab); torch.cuda.synchronize()
                y = ay.float()
                errs.append(((y - ref).abs().max() / ref.abs().max()).item())
                runs.append(y.clone())
            spread = max(((r - runs[0]).abs().max() / ref.abs().max()).item() for r in runs)
            errs.sort()
            out[f"E{E}_T{T}"] = dict(rel_err_median=f"{errs[len(errs)//2]:.1e}", rel_err_max=f"{errs[-1]:.1e}",
                                    auto_run_to_run_max=f"{spread:.1e}", ref_absmax=round(ref.abs().max().item(), 3))
            print(f"E={E} T={T}", out[f"E{E}_T{T}"], flush=True)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
