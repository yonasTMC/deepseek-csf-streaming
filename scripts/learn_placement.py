"""Learn a CSF placement from production route traces, and replay traces against placements.

learn:  python3 scripts/learn_placement.py learn --out p.json TRACE... [--pinned 151 --ram 4400]
eval:   python3 scripts/learn_placement.py eval --placement a.json b.json --test TRACE...

Only decode-sized calls (<= --max-tokens routed tokens) count: placement buys decode speed,
and long prompts would otherwise dominate the frequencies. Pinned experts per layer are the
most frequent ones; the RAM tier takes the most frequent remaining (layer, expert) pairs
across all layers. Eval replays decode calls through pinned + per-layer LRU (--lru slots)
+ RAM tier and reports where each step's distinct misses come from.
"""
import argparse, json
from collections import Counter, OrderedDict
import numpy as np
from simulate_cache import LAYERS, load


def decode_calls(paths, max_tokens):
    for p in paths:
        for layer, ids in load(p):
            if layer < LAYERS and ids.shape[0] <= max_tokens:
                yield layer, ids


def learn(a):
    freq = [Counter() for _ in range(LAYERS)]
    n = 0
    for layer, ids in decode_calls(a.traces, a.max_tokens):
        freq[layer].update(int(e) for e in np.unique(ids) if e >= 0)
        n += 1
    vram = {l: sorted(e for e, _ in freq[l].most_common(a.pinned)) for l in range(LAYERS)}
    ranked = sorted(((c, l, e) for l in range(LAYERS) for e, c in freq[l].items() if e not in set(vram[l])), reverse=True)
    ram = {l: [] for l in range(LAYERS)}
    for _, l, e in ranked[:a.ram]:
        ram[l].append(e)
    json.dump(dict(vram=vram, ram=ram), open(a.out, "w"))
    per_layer = [len(ram[l]) for l in range(LAYERS)]
    print(f"{n} decode calls; RAM tier per layer min {min(per_layer)} max {max(per_layer)}; wrote {a.out}")


def evaluate(a):
    calls = list(decode_calls(a.test, a.max_tokens))
    for path in a.placement:
        p = json.load(open(path))
        pinned = [set(p["vram"][str(l)][:a.pinned]) for l in range(LAYERS)]
        ram = {(int(l), e) for l, es in p["ram"].items() for e in es}
        lru = [OrderedDict() for _ in range(LAYERS)]
        t = Counter()
        for layer, ids in calls:
            for e in (int(x) for x in np.unique(ids) if x >= 0):
                t["routes"] += 1
                if e in pinned[layer]:
                    t["pinned"] += 1; continue
                if e in lru[layer]:
                    lru[layer].move_to_end(e); t["lru"] += 1; continue
                t["ram" if (layer, e) in ram else "nvme"] += 1
                lru[layer][e] = None
                if len(lru[layer]) > a.lru:
                    lru[layer].popitem(last=False)
        steps = len(calls) / LAYERS
        r = t["routes"]
        print(f"{path}: {r} expert loads over {steps:.0f} steps | pinned {100*t['pinned']/r:.1f}%  lru {100*t['lru']/r:.1f}%  "
              f"ram {100*t['ram']/r:.1f}%  nvme {100*t['nvme']/r:.1f}% | per step: ram {t['ram']/steps:.1f}  nvme {t['nvme']/steps:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("learn"); l.add_argument("traces", nargs="+"); l.add_argument("--out", required=True)
    e = sub.add_parser("eval"); e.add_argument("--placement", nargs="+", required=True); e.add_argument("--test", nargs="+", required=True)
    for s in (l, e):
        s.add_argument("--pinned", type=int, default=151)
        s.add_argument("--ram", type=int, default=4400 if s is l else None)
        s.add_argument("--max-tokens", type=int, default=16)
    e.add_argument("--lru", type=int, default=56)
    a = ap.parse_args()
    learn(a) if a.cmd == "learn" else evaluate(a)
