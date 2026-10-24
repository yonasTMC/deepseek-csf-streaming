"""Replay a routing trace against VRAM/RAM/SSD expert placements.

Reads routes.bin written by csf_streaming.trace and reports, per decode token,
how many unique experts come from each tier and the transfer time those loads
imply on the measured links. Transfer time is a lower bound on unhidden load
time; it ignores compute and assumes the scheduler can overlap nothing else.
"""
import argparse
from collections import Counter, OrderedDict
import json
import struct

import numpy as np

LAYERS, EXPERTS = 40, 384
EXPERT_BYTES = 18.8e6            # both TP halves; each GPU receives half
GIB = 2 ** 30


def load(path):
    calls, data = [], open(path, "rb").read()
    pos = 0
    while pos + 8 <= len(data):
        layer, n = struct.unpack_from("<ii", data, pos)
        pos += 8
        if pos + n * 12 > len(data):    # truncated tail (writer stopped mid-record)
            break
        ids = np.frombuffer(data, np.int16, n * 6, pos).reshape(n, 6)
        pos += n * 12
        calls.append((layer, ids))
    return calls


class Tiered:
    """Per-layer VRAM (pinned top-k + LRU) over a global exclusive RAM LRU."""

    def __init__(self, vram, ram, pinned_frac, freq):
        self.vram, self.ram_cap = vram, ram
        self.pinned = {l: set() for l in range(LAYERS)}
        npin = int(vram * pinned_frac)
        if npin:
            for l in range(LAYERS):
                self.pinned[l] = {e for e, _ in freq[l].most_common(npin)}
        self.lru = {l: OrderedDict() for l in range(LAYERS)}
        self.lru_cap = vram - npin
        self.ram = OrderedDict()
        # Static RAM content: the next most frequent experts, all layers.
        ranked = sorted(((c, l, e) for l in range(LAYERS) for e, c in freq[l].items()
                         if e not in self.pinned[l]), reverse=True)
        for _, l, e in ranked[:ram]:
            self.ram[(l, e)] = None

    def access(self, layer, experts):
        hit = ram = ssd = 0
        lru = self.lru[layer]
        for e in experts:
            if e in self.pinned[layer]:
                hit += 1
                continue
            if e in lru:
                lru.move_to_end(e)
                hit += 1
                continue
            key = (layer, e)
            if key in self.ram:
                ram += 1
                del self.ram[key]
            else:
                ssd += 1
            lru[e] = None
            if len(lru) > self.lru_cap:
                victim, _ = lru.popitem(last=False)
                self.ram[(layer, victim)] = None
                while len(self.ram) > self.ram_cap:
                    self.ram.popitem(last=False)
        return hit, ram, ssd


def run(calls, vram, ram, pinned_frac, freq, decode_max, links):
    cache = Tiered(vram, ram, pinned_frac, freq)
    tot = Counter()
    for layer, ids in calls:
        if layer >= LAYERS:
            continue
        experts = list(dict.fromkeys(int(e) for e in ids.ravel() if e >= 0))
        h, r, s = cache.access(layer, experts)
        kind = "decode" if ids.shape[0] <= decode_max else "prefill"
        tot[kind + "_calls"] += 1
        tot[kind + "_tokens"] += ids.shape[0] if layer == 0 else 0
        tot[kind + "_hit"] += h
        tot[kind + "_ram"] += r
        tot[kind + "_ssd"] += s
    out = {}
    for kind in ("decode", "prefill"):
        steps = tot[kind + "_calls"] / LAYERS
        req = tot[kind + "_hit"] + tot[kind + "_ram"] + tot[kind + "_ssd"]
        if not steps or not req:
            continue
        r, s = tot[kind + "_ram"] / steps, tot[kind + "_ssd"] / steps
        # Every load crosses both GPU links (half an expert each); GPU0 is the
        # slow link. SSD loads additionally share the aggregate NVMe rate.
        link = (r + s) * EXPERT_BYTES / 2 / (links["gpu0_h2d"] * GIB)
        disk = s * EXPERT_BYTES / (links["ssd"] * GIB)
        t = max(link, disk)
        out[kind] = dict(steps=int(steps), tokens=tot[kind + "_tokens"],
                         unique_per_step=round(req / steps, 1),
                         hit_rate=round(tot[kind + "_hit"] / req, 4),
                         ram_per_step=round(r, 2), ssd_per_step=round(s, 2),
                         transfer_ms_per_step=round(1e3 * t, 2),
                         transfer_bound_steps_per_s=round(1 / t, 1) if t else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("routes")
    ap.add_argument("--vram", type=int, nargs="+", default=[128, 176, 200, 224])
    ap.add_argument("--ram", type=int, nargs="+", default=[0, 4200])
    ap.add_argument("--pinned", type=float, nargs="+", default=[0.0, 0.5, 0.8])
    ap.add_argument("--decode-max", type=int, default=2)
    ap.add_argument("--train-frac", type=float, default=0.5,
                    help="frequencies for pinning come from this leading fraction")
    ap.add_argument("--gpu0-h2d", type=float, default=26.5)
    ap.add_argument("--ssd", type=float, default=23.0)
    ap.add_argument("--dump-placement", help="write the first --vram/--pinned/--ram "
                    "configuration as a CSF_PLACEMENT file, using the whole trace")
    a = ap.parse_args()
    calls = load(a.routes)
    if a.dump_placement:
        freq = {l: Counter() for l in range(LAYERS)}
        for layer, ids in calls:
            if layer < LAYERS:
                freq[layer].update(int(e) for e in ids.ravel() if e >= 0)
        cache = Tiered(a.vram[0], a.ram[0], a.pinned[0], freq)
        with open(a.dump_placement, "w") as f:
            json.dump(dict(vram={l: sorted(cache.pinned[l]) for l in range(LAYERS)},
                           ram={l: [e for (m, e) in cache.ram if m == l] for l in range(LAYERS)}), f)
        return
    cut = int(len(calls) * a.train_frac)
    freq = {l: Counter() for l in range(LAYERS)}
    for layer, ids in calls[:cut]:
        if layer < LAYERS:
            freq[layer].update(int(e) for e in ids.ravel() if e >= 0)
    test = calls[cut:] if a.train_frac else calls
    links = dict(gpu0_h2d=a.gpu0_h2d, ssd=a.ssd)
    for v in a.vram:
        for r in a.ram:
            for p in a.pinned:
                res = run(test, v, r, p, freq, a.decode_max, links)
                print(json.dumps(dict(vram=v, ram=r, pinned=p, **res)))


if __name__ == "__main__":
    main()
