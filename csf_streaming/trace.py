"""Optional routing trace for offline cache simulation (rank 0 only).

Record format, little endian: int32 layer, int32 tokens, int16[tokens, topk] ids.
Decode-sized calls (tokens <= CSF_TRACE_HIDDEN_TOKENS) also append their MoE
inputs to hidden-L<layer>.bf16 until CSF_TRACE_HIDDEN_MAX rows per layer.
"""
import os
import struct
from pathlib import Path

import torch

_DIR = os.environ.get("CSF_TRACE_DIR")
_HIDDEN_TOKENS = int(os.environ.get("CSF_TRACE_HIDDEN_TOKENS", "2"))
_HIDDEN_MAX = int(os.environ.get("CSF_TRACE_HIDDEN_MAX", "4096"))
_routes = None
_hidden = {}


def record(layer, rank, x, topk_ids):
    global _routes
    if not _DIR or rank != 0:
        return
    if _routes is None:
        Path(_DIR).mkdir(parents=True, exist_ok=True)
        _routes = open(Path(_DIR) / "routes.bin", "ab", buffering=1 << 20)
    ids = topk_ids.to(torch.int16).cpu()
    _routes.write(struct.pack("<ii", layer, ids.shape[0]) + ids.numpy().tobytes())
    if ids.shape[0] <= _HIDDEN_TOKENS:
        f, rows = _hidden.get(layer, (None, 0))
        if rows < _HIDDEN_MAX:
            if f is None:
                f = open(Path(_DIR) / f"hidden-L{layer}.bf16", "ab")
            f.write(x.detach().to(torch.bfloat16).view(torch.int16).cpu().numpy().tobytes())
            _hidden[layer] = f, rows + x.shape[0]


def record_ids(layer, rank, ids, topk=6):
    """Graph mode: ids already on the host (flattened tokens x topk)."""
    global _routes
    if not _DIR or rank != 0 or not ids:
        return
    if _routes is None:
        Path(_DIR).mkdir(parents=True, exist_ok=True)
        _routes = open(Path(_DIR) / "routes.bin", "ab", buffering=1 << 20)
    import numpy as np
    _routes.write(struct.pack("<ii", layer, len(ids) // topk) + np.asarray(ids, dtype=np.int16).tobytes())


def flush():
    if _routes is not None:
        _routes.flush()
    for f, _ in _hidden.values():
        f.flush()
