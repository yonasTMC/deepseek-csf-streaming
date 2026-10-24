"""Run each TP2 expert slice (intermediate 1152) as two 576-wide half experts.

The half-expert kernel sees 2E slots and top-2k routes (slot s -> 2s, 2s+1, same gate); the result
must equal the full-slice output. Reports speed and error of the deterministic M16 kernel on the
half layout against the deterministic kernel on the full slice.
"""
import argparse
import torch
from csf_streaming.storage import FIELDS, allocate_slots, read_experts, upload_and_prepare


def det_plan(fm, T, k, dev, weights, det=True):
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig
    extra = dict(deterministic_output=True, decode_config=MoeDecodeConfig(
        backend="dynamic", route_planner="internal", max_active_clusters=None,
        dynamic_tile_m=16, dynamic_route_mode="grouped")) if det else {}
    return fm.plan(fm.Caps(max_tokens=T, num_topk=k, device=dev, weight_plan=weights._impl.plan, quant_mode="w4a8_mx",
                           swiglu_limit=10.0, **extra))


def full_ref(fm, slots, x, ids, gates, T, k, dev):
    plan = det_plan(fm, T, k, dev, slots)
    ws = torch.empty(sum(s.nbytes for s in plan.scratch_specs()), dtype=torch.uint8, device=dev)
    y = torch.empty_like(x)
    fm.run(binding=fm.bind(plan, scratch=ws, a=x, experts=slots._impl, topk_weights=gates, topk_ids=ids,
                           output=y, input_scales_static=True))
    torch.cuda.synchronize()
    return y.float().clone()


def timed(fm, b):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20):
            fm.run(binding=b)
    torch.cuda.synchronize()
    s0, s1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s0.record()
    for _ in range(10):
        g.replay()
    s1.record(); torch.cuda.synchronize()
    return s0.elapsed_time(s1) / 200 * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--masked", action="store_true")
    ap.add_argument("--big", type=int, default=4096, help="capacity of the one-plan-for-all-sizes case")
    a = ap.parse_args()
    from b12x.moe import fused_moe as fm
    dev = torch.device("cuda", 0)
    h, n, k, E, real_n = 5120, 2304, 6, 208, 64
    ids_src = list(range(real_n))

    def slots_from(tp, part, count, tile):
        w, _, _ = read_experts(a.root, 20, ids_src, h, n, part, tp)
        real = upload_and_prepare(w, h, n // tp, dev, limit=10.0)
        tw, _, _ = read_experts(a.root, 20, [0], h, n, part, tp)
        s = allocate_slots(upload_and_prepare(tw, h, n // tp, dev, limit=10.0), count)
        return s, real

    full, full_real = slots_from(2, 0, E, None)
    for f in FIELDS:
        dst, src = getattr(full._impl, f), getattr(full_real._impl, f)
        for i in range(E):
            dst[i].copy_(src[i % real_n])
    half, h0 = slots_from(4, 0, 2 * E, None)
    _, h1 = slots_from(4, 1, 2 * E, None)
    for f in FIELDS:
        dst = getattr(half._impl, f)
        for i in range(E):
            dst[2 * i].copy_(getattr(h0._impl, f)[i % real_n])
            dst[2 * i + 1].copy_(getattr(h1._impl, f)[i % real_n])
    for T in a.tokens:
        g = torch.Generator(device=dev).manual_seed(T)
        x = (torch.randn(T, h, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        ids = torch.stack([torch.randperm(E, device=dev, generator=g)[:k] for _ in range(T)]).int()
        gates = torch.softmax(torch.randn(T, k, device=dev, generator=g), -1)
        if a.masked:
            # Expert-major prefill: routes outside the resident group are -1 with zero gate.
            keep = torch.rand(T, k, device=dev, generator=g) < 0.5
            gates = torch.where(keep, gates, torch.zeros_like(gates))
            zero_ref = full_ref(fm, full, x, ids, gates, T, k, dev)
            ids = torch.where(keep, ids, torch.full_like(ids, -1))
        hids = torch.stack([2 * ids, 2 * ids + 1], -1)
        hids = torch.where(ids.unsqueeze(-1) < 0, -1, hids).reshape(T, 2 * k).int().contiguous()
        hgates = gates.repeat_interleave(2, -1).contiguous()
        out = {}
        cases = [("full", full, k, ids, gates, True, T), ("half", half, 2 * k, hids, hgates, True, T)]
        if T > 8 and not a.masked:
            cases += [("full_auto", full, k, ids, gates, False, T), ("half_auto", half, 2 * k, hids, hgates, False, T)]
        cases += [("half_bigplan", half, 2 * k, hids, hgates, True, a.big)]
        for name, slots, kk, ii, gg, det, cap in cases:
            plan = det_plan(fm, cap, kk, dev, slots, det)
            ws = torch.empty(sum(s.nbytes for s in plan.scratch_specs()), dtype=torch.uint8, device=dev)
            y = torch.zeros_like(x)
            b = fm.bind(plan, scratch=ws, a=x, experts=slots._impl, topk_weights=gg, topk_ids=ii, output=y, input_scales_static=True)
            fm.run(binding=b); torch.cuda.synchronize()
            out[name] = (y.float().clone(), timed(fm, b))
        ref = zero_ref if a.masked else out["full"][0]
        err = ((out["half"][0] - ref).abs().max() / ref.abs().max()).item()
        if a.masked:
            ferr = ((out["full"][0] - ref).abs().max() / ref.abs().max()).item()
            berr = ((out["half_bigplan"][0] - ref).abs().max() / ref.abs().max()).item()
            print(f"T={T} masked: full_det(-1) vs zero-gate ref {ferr:.1e}  half_det(-1) vs ref {err:.1e}  half_bigplan(-1) {berr:.1e}", flush=True)
        rest = "  ".join(f"{n} {v[1]:.1f} us err {((v[0] - ref).abs().max() / ref.abs().max()).item():.1e}"
                         for n, v in out.items() if n not in ("full", "half"))
        print(f"T={T} full_det {out['full'][1]:.1f} us  half_det {out['half'][1]:.1f} us  half_vs_full_err {err:.1e}  {rest}", flush=True)


if __name__ == "__main__":
    main()
