"""Write every routed expert once, in its prepared B12X slot layout.

Run one process per TP rank (torch.distributed.run --nproc-per-node=2). Each
rank writes <out>/rank<r>/layer<L>.bin: 384 fixed-stride records, one per
expert, each the concatenation of that expert's slot rows for FIELDS, padded to
4 KiB so a miss is a single O_DIRECT read straight into a pinned buffer.

--split 2 stores each rank's expert slice as two half-width experts (the TP4
slices 2r and 2r+1): every field holds [half 0 row, half 1 row], so record
offsets still address one slot pair. The deterministic B12X kernel is fast at
the half width but not at the full TP2 width.
"""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from csf_streaming.storage import FIELDS, read_experts, upload_and_prepare

ALIGN = 4096


def layout(parts):
    fields, offset = [], 0
    for name in FIELDS:
        t = getattr(parts[0]._impl, name)
        row = t[0]
        fields.append(dict(name=name, dtype=str(t.dtype).removeprefix("torch."),
                           shape=[len(parts), *row.shape] if len(parts) > 1 else list(row.shape),
                           offset=offset, nbytes=row.nbytes * len(parts)))
        offset += row.nbytes * len(parts)
    return fields, (offset + ALIGN - 1) // ALIGN * ALIGN


def records(parts, stride):
    n = getattr(parts[0]._impl, FIELDS[0]).shape[0]
    out = torch.zeros((n, stride), dtype=torch.uint8, device="cpu").pin_memory()
    offset = 0
    for name in FIELDS:
        for prepared in parts:
            t = getattr(prepared._impl, name).contiguous().view(torch.uint8).reshape(n, -1)
            out[:, offset:offset + t.shape[1]].copy_(t, non_blocking=True)
            offset += t.shape[1]
    torch.cuda.current_stream().synchronize()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True, nargs="+",
                    help="one directory per rank (round robin), e.g. one per NVMe drive")
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(40)))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--intermediate", type=int, default=2304)
    ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--swiglu-limit", type=float, default=10.0)
    ap.add_argument("--split", type=int, default=1)
    a = ap.parse_args()
    rank, tp = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 2))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    out = Path(a.out[rank % len(a.out)]) / f"rank{rank}"
    out.mkdir(parents=True, exist_ok=True)
    meta = None
    for layer in a.layers:
        target = out / f"layer{layer}.bin"
        if target.exists() and meta is not None and target.stat().st_size == a.experts * meta["stride"]:
            continue
        start, tmp = time.perf_counter(), target.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            for lo in range(0, a.experts, a.batch):
                ids = list(range(lo, min(lo + a.batch, a.experts)))
                parts = []
                for part in range(a.split):
                    weights, _, _ = read_experts(a.root, layer, ids, a.hidden, a.intermediate,
                                                 rank * a.split + part, tp * a.split)
                    parts.append(upload_and_prepare(weights, a.hidden, a.intermediate // (tp * a.split),
                                                    device, limit=a.swiglu_limit))
                if meta is None:
                    fields, stride = layout(parts)
                    meta = dict(fields=fields, stride=stride, experts=a.experts, rank=rank, tp=tp,
                                split=a.split, hidden=a.hidden, intermediate=a.intermediate, root=a.root)
                    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
                f.write(records(parts, meta["stride"]).numpy().tobytes())
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(target)
        print(f"rank{rank} layer{layer} {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
