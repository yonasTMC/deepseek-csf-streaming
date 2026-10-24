import os


GRAPH_TOKENS = int(os.environ.get("CSF_GRAPH_TOKENS", "8"))
BREAKABLE = os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH") == "1"
DEBUG_MOE_CHUNK = int(os.environ.get("CSF_DEBUG_MOE_CHUNK", "0"))
EXPERT_MAJOR = os.environ.get("CSF_V4") == "1" and os.environ.get("CSF_EXPERT_MAJOR_PREFILL", "1") == "1"


def install():
    import torch
    from vllm.config import get_current_vllm_config
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    import vllm.models.deepseek_v4_1.mxfp4_csf as target
    from .cache import ExpertCache
    from .fastcache import FastExpertCache

    if getattr(target.Mxfp4CsfMoEMethod, "streaming_enabled", False):
        return
    original = target.Mxfp4CsfMoEMethod

    class StreamingMxfp4CsfMoEMethod(original):
        streaming_enabled = True

        def process_weights_after_loading(self, layer):
            config = get_current_vllm_config()
            # DSpark draft blocks (index >= num_hidden_layers) are small; keep them resident.
            self.streamed = self.layer_index < config.model_config.hf_text_config.num_hidden_layers
            if not self.streamed:
                return super().process_weights_after_loading(layer)
            graphs_ok = (os.environ.get("CSF_STREAM_FAKE_HITS") == "1"
                         or os.environ.get("CSF_GRAPH") == "1")
            if (not config.model_config.enforce_eager and not graphs_ok) or self.activation_mode != "a8":
                raise ValueError("CSF streaming requires --enforce-eager and A8 experts")
            tp = get_tensor_model_parallel_world_size()
            rank = get_tensor_model_parallel_rank()
            if tp != 2:
                raise ValueError("This streaming integration is restricted to validated TP2 geometry")
            common = dict(
                capacity=int(os.environ.get("CSF_STREAM_SLOTS", "128")),
                nonlinearity="situ" if self.moe.activation.value == "situ" else "silu",
                limit=self.moe.swiglu_limit,
            )
            args = (self.owner.checkpoint_root, self.layer_index, self.hidden_size,
                    self.local_intermediate * tp, self.num_experts, rank, tp,
                    layer.w13_weight.device)
            if os.environ.get("CSF_PACK_DIRS"):
                self.cache = FastExpertCache(*args, **common)
                if graphs_ok:
                    if self.cache.split == 1:
                        self.install_slot_kernel(layer)
                    # v4 lends pinned slots to steps that need more than the LRU
                    # (soft pinning); one pinned slot is never lent.
                    usable = self.cache.capacity - (1 if os.environ.get("CSF_V4") == "1"
                                                    else len(self.cache.pinned))
                    if GRAPH_TOKENS * self.moe.experts_per_token >= usable:
                        raise ValueError(f"CSF_GRAPH_TOKENS={GRAPH_TOKENS} can route more experts than the "
                                         f"{usable} usable slots of layer {self.layer_index}")
            else:
                self.cache = ExpertCache(
                    *args, batch_size=int(os.environ.get("CSF_STREAM_LOAD_BATCH", "8")), **common)
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        def install_slot_kernel(self, layer):
            """Serve the slot tensors through vLLM's B12X backend, which prepares
            kernels for every CUDA-graph size; apply() passes slot IDs."""
            import dataclasses
            from vllm.models.deepseek_v4_1.mxfp4_csf import (
                B12xExperts, MoEPrepareAndFinalizeNoDPEPModular, mk)
            capacity = self.cache.capacity
            moe = dataclasses.replace(self.moe, num_experts=capacity, num_local_experts=capacity,
                                      num_logical_experts=capacity)
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            backend = B12xExperts(moe, self.moe_quant_config)
            backend.install_prepared_experts(layer, self.cache.prepared)
            self.slot_kernel = mk.FusedMoEKernel(MoEPrepareAndFinalizeNoDPEPModular(), backend)
            self.moe_kernel = self.slot_kernel  # base prepare_workspace sizes scratch from it

        def prepare_workspace(self, hidden_states, shared_workspace_size):
            if not self.streamed or getattr(self, "slot_kernel", None) is not None:
                return super().prepare_workspace(hidden_states, shared_workspace_size)
            # Shared experts own this separate scratch; streamed experts use
            # their own bounded kernel scratch and cannot alias it.
            shared = torch.empty(shared_workspace_size, dtype=torch.uint8, device=hidden_states.device)
            return (), shared

        def apply(self, layer, x, topk_weights, topk_ids,
                  shared_experts, shared_experts_input, workspace=None):
            if not self.streamed:
                return super().apply(layer, x, topk_weights, topk_ids, shared_experts,
                                     shared_experts_input, workspace)
            if layer.apply_router_weight_on_input or layer.expert_map is not None:
                raise ValueError("Streaming supports ordinary TP output-weighted routing")
            split = getattr(self.cache, "split", 1) > 1
            if getattr(self, "slot_kernel", None) is None and not split:
                return self.cache.run(x, topk_weights, topk_ids)

            def routed(x, weights, slots, workspace):
                if split:
                    return self.cache.run_slots(x, weights, slots)
                return self.slot_kernel.apply(
                    hidden_states=x, w1=layer.w13_weight, w2=layer.w2_weight,
                    topk_weights=weights, topk_ids=slots, activation=layer.activation,
                    global_num_experts=self.cache.capacity, expert_map=None,
                    apply_router_weight_on_input=False, shared_experts=None,
                    shared_experts_input=None, workspace=workspace)

            if self.cache.fake_hits:
                return routed(x, topk_weights, self.cache.device_slots(topk_ids), workspace)
            if BREAKABLE:
                # v3: this call is a graph break; the host serves misses
                # between captured segments and nothing spins on the GPU.
                output = torch.empty_like(x)
                _streamed_moe(self, layer, x, topk_weights, topk_ids, output, routed)
                return output
            if torch.cuda.is_current_stream_capturing() or x.shape[0] <= GRAPH_TOKENS:
                return routed(x, topk_weights, self.cache.graph_slots(topk_ids, x), workspace)
            if DEBUG_MOE_CHUNK:
                # Debug: push every token through the decode-sized MoE path.
                c = DEBUG_MOE_CHUNK
                return torch.cat([routed(x[i:i + c], topk_weights[i:i + c],
                                         self.cache.graph_slots(topk_ids[i:i + c], x[i:i + c]), None)
                                  for i in range(0, x.shape[0], c)])
            if EXPERT_MAJOR:
                # Prefill: load each distinct expert at most once per layer per chunk.
                # Real prompts touch nearly every expert, so token sub-chunking
                # reloaded the same experts many times (~330 tok/s on a 57k prompt).
                acc = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
                groups = self.cache.expert_groups(topk_ids)
                for i, group in enumerate(groups):
                    batch = torch.tensor(group, dtype=topk_ids.dtype, device=x.device)
                    # Make the batch resident; the host reads the next group ahead meanwhile.
                    self.cache.graph_slots(batch.view(1, -1), ahead=groups[i + 1] if i + 1 < len(groups) else None)
                    mask = torch.isin(topk_ids, batch)
                    if split:
                        # Only this group's (token, route) pairs: no masked routes, no wasted rows.
                        tok, route = mask.nonzero(as_tuple=True)
                        self.cache.run_pairs(x, tok, topk_weights[tok, route],
                                             self.cache.table[topk_ids[tok, route].long()], acc)
                        continue
                    slots = self.cache.slots_for(topk_ids, mask)
                    weights = torch.where(mask, topk_weights, torch.zeros_like(topk_weights))
                    acc += routed(x, weights, slots, None).float()
                return acc.to(x.dtype)
            # Large eager batches (prefill): token chunks whose experts fit the cache.
            output = torch.empty_like(x)
            for lo, hi in self.cache.eager_chunks(topk_ids):
                slots = self.cache.graph_slots(topk_ids[lo:hi], x[lo:hi])
                output[lo:hi] = routed(x[lo:hi], topk_weights[lo:hi], slots, None)
            return output

    from vllm.compilation.breakable_cudagraph import eager_break_during_capture

    @eager_break_during_capture
    def _streamed_moe(method, layer, x, topk_weights, topk_ids, output, routed):
        cache = method.cache
        if x.shape[0] <= GRAPH_TOKENS:
            output.copy_(routed(x, topk_weights, cache.host_slots(topk_ids), None))
            return
        for lo, hi in cache.eager_chunks(topk_ids):
            output[lo:hi] = routed(x[lo:hi], topk_weights[lo:hi], cache.host_slots(topk_ids[lo:hi]), None)

    target.Mxfp4CsfMoEMethod = StreamingMxfp4CsfMoEMethod
