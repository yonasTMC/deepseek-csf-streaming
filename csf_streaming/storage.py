from dataclasses import replace
from pathlib import Path
import time

import torch


def scale_shapes(count, hidden, local):
    return ((count, hidden // 32, 2 * local), (count, local // 32, hidden))


def pin_source(source):
    return replace(source, fixed=tuple(t.pin_memory() for t in source.fixed),
                   exceptions=tuple(t.pin_memory() for t in source.exceptions))


def read_experts(root, layer, ids, hidden, intermediate, rank, tp):
    """Read only requested rank-local packed weights; retain original CSF bytes."""
    from vllm.model_executor.model_loader.csf_utils import CsfTensorReader
    from vllm.model_executor.model_loader.mxfp4_csf_loader import (
        checkpoint_contract, expert_tensor_names, _load_mxfp4_csf_weights,
    )
    start = time.perf_counter()
    contract = checkpoint_contract(str(Path(root).resolve()))
    with CsfTensorReader(root, contract["source_names"], "mxfp4") as reader:
        matrices = (tuple(reader.matrix(w, s) for w, s in
                          expert_tensor_names(contract["family"], layer, e)) for e in ids)
        # The CPU loader does not use scale scratch. Allocate it on the GPU later.
        weights = _load_mxfp4_csf_weights(
            matrices, num_experts=len(ids), hidden_size=hidden,
            intermediate_size=intermediate, tp_rank=rank, tp_size=tp,
            device="cpu", w13_scale_scratch=None, w2_scale_scratch=None,
        )
        weights = replace(weights, w13=weights.w13.pin_memory(), w2=weights.w2.pin_memory(),
                          w13_scales=pin_source(weights.w13_scales),
                          w2_scales=pin_source(weights.w2_scales))
    tensors = [weights.w13, weights.w2]
    for source in (weights.w13_scales, weights.w2_scales):
        tensors.extend((*source.fixed, *source.exceptions))
    return weights, sum(t.numel() * t.element_size() for t in tensors), time.perf_counter() - start


def upload_and_prepare(weights, hidden, local, device, nonlinearity="silu", limit=None):
    from b12x.moe import fused_moe as fm
    count = weights.w13.shape[0]
    scratch = [torch.empty(s, device=device, dtype=torch.uint8)
               for s in scale_shapes(count, hidden, local)]
    gpu = replace(weights, w13=weights.w13.to(device, non_blocking=True),
                  w2=weights.w2.to(device, non_blocking=True),
                  w13_scale_scratch=scratch[0], w2_scale_scratch=scratch[1])
    plan = weight_plan(count, hidden, local, nonlinearity, limit)
    prepared = fm.prepare_weights(plan=plan, weights=gpu)
    # Preparation has losslessly expanded and repacked the scales already.
    # Each cached expert owns its scales; no shared scratch may overwrite them.
    if prepared._impl.mxfp4_csf_inline is not None:
        raise ValueError("Set B12X_W4A8_CSF_INLINE=0 for expanded-scale expert slots")
    return replace(prepared, _impl=replace(prepared._impl, mxfp4_csf=None))


def weight_plan(count, hidden, local, nonlinearity="silu", limit=None):
    from b12x.moe import fused_moe as fm
    return fm.plan_weights(
        source=fm.PackedSource(format="fp4_e8m0_k32", w13_layout="w31"),
        activation=fm.ActivationSpec(mode="a8", nonlinearity=nonlinearity,
                                     io_dtype=torch.bfloat16, swiglu_limit=limit),
        geometry=fm.MoEGeometry(num_experts=count, hidden_size=hidden, intermediate_size=local),
    )


FIELDS = ("w1_fp4", "w2_fp4", "w1_blockscale", "w2_blockscale",
          "a1_gscale", "a2_gscale", "w1_alphas", "w2_alphas")


def allocate_slots(template, capacity):
    """Allocate the exact B12X A8 representation without allocating all experts."""
    from b12x.moe import fused_moe as fm
    p = template.plan
    plan = weight_plan(capacity, template.hidden_size, template.intermediate_size,
                       p.activation.nonlinearity, p.activation.swiglu_limit)
    impl = template._impl
    tensors = {}
    for field in FIELDS:
        t = getattr(impl, field)
        if t.ndim == 0 or t.shape[0] != 1:
            raise ValueError(f"Unsupported pinned B12X field layout: {field}: {t.shape}")
        tensors[field] = torch.empty((capacity, *t.shape[1:]), dtype=t.dtype, device=t.device)
        if field in ("a1_gscale", "a2_gscale", "w1_alphas", "w2_alphas"):
            tensors[field].fill_(1)
    rep = impl.representation
    if rep is None or not hasattr(rep.value, "w13_rp"):
        raise ValueError("Streaming requires the inspected B12X W4A8 representation")
    native = replace(rep.value, num_experts=capacity,
                     w13_rp=tensors["w1_fp4"], w2_rp=tensors["w2_fp4"],
                     w13_sfb=tensors["w1_blockscale"], w2_sfb=tensors["w2_blockscale"])
    return fm.PreparedExperts(plan=plan, _impl=replace(
        impl, plan=plan._impl, representation=replace(rep, value=native), **tensors))
