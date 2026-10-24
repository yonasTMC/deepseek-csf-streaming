"""Measure how well layer L's MoE input predicts the experts of layers L+d.

Applies router L+d (sqrtsoftplus scores plus selection bias) to the recorded
MoE inputs of layer L and compares its top-k with the experts layer L+d really
selected for the same tokens. d=0 validates the scoring reimplementation.
"""
import argparse
import json
from pathlib import Path
import struct

import numpy as np
import torch
from safetensors import safe_open


def routes(path, decode_max):
    data, pos, per_layer = open(path, "rb").read(), 0, {}
    while pos < len(data):
        layer, n = struct.unpack_from("<ii", data, pos)
        pos += 8
        ids = np.frombuffer(data, np.int16, n * 6, pos).reshape(n, 6)
        pos += n * 12
        if n <= decode_max:
            per_layer.setdefault(layer, []).append(ids)
    return {l: np.concatenate(v) for l, v in per_layer.items()}


def gate(root, layer):
    names = json.load(open(Path(root) / "build-contract.json"))["source_names"]
    out = []
    for suffix in ("weight", "bias"):
        name = f"layers.{layer}.ffn.gate.{suffix}"
        with safe_open(str(Path(root) / "tensors" / names[name]), "pt") as f:
            out.append(f.get_tensor(name).float().cuda())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--root", required=True)
    ap.add_argument("--distances", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--widths", type=int, nargs="+", default=[6, 12, 24])
    ap.add_argument("--decode-max", type=int, default=2)
    a = ap.parse_args()
    actual = routes(Path(a.trace) / "routes.bin", a.decode_max)
    gates = {l: gate(a.root, l) for l in range(40)}
    results = {}
    for d in a.distances:
        for w in a.widths:
            hits = total = 0
            for layer in range(40 - d):
                path = Path(a.trace) / f"hidden-L{layer}.bf16"
                if not path.exists():
                    continue
                x = torch.from_numpy(np.fromfile(path, np.int16)).view(torch.bfloat16)
                x = x.reshape(-1, 5120).cuda().float()
                truth = torch.from_numpy(actual[layer + d][:x.shape[0]].astype(np.int64)).cuda()
                x = x[:truth.shape[0]]
                weight, bias = gates[layer + d]
                score = torch.sqrt(torch.nn.functional.softplus(x @ weight.T)) + bias
                pred = score.topk(w, dim=-1).indices
                hits += (pred[:, :, None] == truth[:, None, :]).any(1).sum().item()
                total += truth.numel()
            results[f"d{d}_top{w}"] = round(hits / total, 4)
            print(f"distance {d} width {w}: recall {hits / total:.4f}", flush=True)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
