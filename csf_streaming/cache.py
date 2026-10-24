import atexit
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import torch

from .policy import SlotCache, routing_chunks
from . import trace
from .storage import read_experts, upload_and_prepare, allocate_slots, FIELDS

_POOL = None


def pool():
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="csf-reader")
    return _POOL


class ExpertCache:
    def __init__(self, root, layer, hidden, intermediate, num_experts, rank, tp,
                 device, capacity=128, batch_size=8, nonlinearity="silu", limit=None):
        self.root, self.layer, self.hidden = root, layer, hidden
        self.intermediate, self.rank, self.tp = intermediate, rank, tp
        self.device = torch.device(device)
        self.policy = SlotCache(capacity, num_experts)
        self.batch_size = batch_size
        self.nonlinearity, self.limit = nonlinearity, limit
        self.stats = Counter()
        self.metrics_interval = max(1, int(os.environ.get("CSF_STREAM_METRICS_INTERVAL", "50")))
        self.compute_done = None
        self.failed = False
        self.io_stream = torch.cuda.Stream(device=self.device)
        self.plans = {}
        # Timing-only mode: misses reuse resident slots without loading, so
        # outputs are wrong but latency equals an all-hit cache.
        self.fake_hits = os.environ.get("CSF_STREAM_FAKE_HITS") == "1"
        with torch.cuda.device(self.device), torch.cuda.stream(self.io_stream):
            weights, _, _ = self._read([0])
            template = self._prepare(weights)
            self.prepared = allocate_slots(template, capacity)
        self.io_stream.synchronize()
        if self.fake_hits:
            self.ensure(list(range(capacity)))
            self.policy.resident.clear()
            self.policy.free = []
        self.stats["cache_bytes"] = sum(getattr(self.prepared._impl, f).numel() *
                                       getattr(self.prepared._impl, f).element_size() for f in FIELDS)
        atexit.register(self.report)

    def _read(self, ids):
        with torch.cuda.device(self.device):
            return read_experts(self.root, self.layer, ids, self.hidden,
                                self.intermediate, self.rank, self.tp)

    def _prepare(self, weights):
        return upload_and_prepare(weights, self.hidden, self.intermediate // self.tp,
                                  self.device, self.nonlinearity, self.limit)

    def report(self):
        directory = os.environ.get("CSF_STREAM_METRICS_DIR")
        if directory:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            data = dict(layer=self.layer, rank=self.rank, **self.stats)
            target = path / f"rank{self.rank}-layer{self.layer}.json"
            tmp = target.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2) + "\n")
            tmp.replace(target)

    def ensure(self, requested):
        if self.failed:
            raise RuntimeError("Expert cache is poisoned after a failed load; restart the worker")
        if self.fake_hits and not self.policy.free and not self.policy.resident:
            self.stats["expert_requests"] += len(requested)
            return {e: e % self.policy.capacity for e in requested}
        mapping, misses = self.policy.reserve(requested)
        self.stats["expert_requests"] += len(mapping)
        self.stats["expert_misses"] += len(misses)
        self.stats["expert_hits"] += len(mapping) - len(misses)
        if not misses:
            return mapping
        batches = [misses[i:i+self.batch_size] for i in range(0, len(misses), self.batch_size)]
        # At most two batches occupy pinned host buffers. Disk reads run while
        # the preceding batch is transferred and prepared on the GPU.
        pending = {}
        def submit(i):
            if i < len(batches):
                pending[i] = pool().submit(self._read, [e for e, _ in batches[i]])
        submit(0)
        submit(1)
        try:
            for i, batch in enumerate(batches):
                weights, size, elapsed = pending.pop(i).result()
                with torch.cuda.device(self.device), torch.cuda.stream(self.io_stream):
                    if self.compute_done is not None:
                        self.io_stream.wait_event(self.compute_done)
                    prepared = self._prepare(weights)
                    slots = torch.tensor([slot for _, slot in batch], device=self.device)
                    for field in FIELDS:
                        getattr(self.prepared._impl, field).index_copy_(
                            0, slots, getattr(prepared._impl, field))
                    ready = torch.cuda.Event()
                    ready.record(self.io_stream)
                # Retain source and prepared buffers until their transfers finish.
                # This is a bounded staging baseline, with asynchronous disk work.
                ready.synchronize()
                submit(i + 2)
                self.stats["staged_bytes"] += size
                self.stats["read_seconds"] += elapsed
                self.stats["load_batches"] += 1
        except BaseException:
            self.failed = True
            for future in pending.values():
                future.cancel()
            raise
        return mapping

    def run(self, x, topk_weights, topk_ids):
        from b12x.moe import fused_moe as fm
        from b12x.moe.fused_moe._tuning import MoeDecodeConfig
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CSF streaming requires eager execution")
        if x.shape[0] == 0:
            return torch.empty_like(x)
        start = time.perf_counter()
        trace.record(self.layer, self.rank, x, topk_ids)
        rows = topk_ids.detach().cpu().tolist()
        output = torch.empty_like(x)
        for lo, hi in routing_chunks(rows, self.policy.capacity):
            requested = list(dict.fromkeys(e for row in rows[lo:hi] for e in row if e != -1))
            if not requested:
                output[lo:hi].zero_()
                self.stats["padding_tokens"] += hi - lo
                continue
            mapping = self.ensure(requested)
            ids = torch.tensor([[-1 if e == -1 else mapping[e] for e in row] for row in rows[lo:hi]],
                               dtype=topk_ids.dtype, device=x.device)
            count, topk = hi - lo, ids.shape[1]
            key = count, topk
            if key not in self.plans:
                plan = fm.plan(fm.Caps(
                    max_tokens=count, num_topk=topk, device=x.device,
                    weight_plan=self.prepared._impl.plan, quant_mode="w4a8_mx", deterministic_output=True,
                    decode_config=MoeDecodeConfig(backend="dynamic", route_planner="internal", max_active_clusters=None, dynamic_tile_m=16, dynamic_route_mode="grouped"),
                    swiglu_limit=self.limit,
                ))
                self.plans[key] = plan
            plan = self.plans[key]
            scratch = torch.empty(sum(s.nbytes for s in plan.scratch_specs()),
                                  dtype=torch.uint8, device=x.device)
            binding = fm.bind(plan, scratch=scratch, a=x[lo:hi], experts=self.prepared._impl,
                              topk_weights=topk_weights[lo:hi].float().contiguous(),
                              topk_ids=ids, output=output[lo:hi], input_scales_static=True)
            fm.run(binding=binding)
            self.compute_done = torch.cuda.Event()
            self.compute_done.record(torch.cuda.current_stream(x.device))
            self.stats["routing_chunks"] += 1
        self.stats["calls"] += 1
        self.stats["tokens"] += x.shape[0]
        self.stats["host_call_seconds"] += time.perf_counter() - start
        if self.stats["calls"] % self.metrics_interval == 0:
            self.report()
            trace.flush()
        return output
