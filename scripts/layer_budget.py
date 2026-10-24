"""Replay held-out decode traffic to choose per-layer pinned-slot counts at a fixed total.

python3 scripts/layer_budget.py --train T1.bin ... --test T9.bin ... [--lru 56 --base 151]

Pinned experts per layer are the most frequent ones in the training traces; every layer
keeps --lru LRU slots. For each layer and each pinned count on a grid, the held-out decode
calls are replayed to count misses (loads not served by pinned or LRU slots). Slots then
move greedily between layers, keeping the total pinned count at 40 x --base, while the
move lowers total misses. Reports misses before/after and the per-layer allocation.
"""
import argparse, json
from collections import Counter, OrderedDict
import numpy as np
from simulate_cache import LAYERS, load


def calls(paths, max_tokens):
    per = [[] for _ in range(LAYERS)]
    for p in paths:
        for layer, ids in load(p):
            if layer < LAYERS and ids.shape[0] <= max_tokens:
                per[layer].append(np.unique(ids[ids >= 0]))
    return per


def misses(seq, ranked, pinned, lru_cap):
    pin = set(ranked[:pinned]); lru = OrderedDict(); m = 0
    for ids in seq:
        for e in ids.tolist():
            if e in pin:
                continue
            if e in lru:
                lru.move_to_end(e); continue
            m += 1; lru[e] = None
            if len(lru) > lru_cap:
                lru.popitem(last=False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--lru", type=int, default=56)
    ap.add_argument("--base", type=int, default=151)
    ap.add_argument("--step", type=int, default=16)
    ap.add_argument("--min", type=int, default=71)
    ap.add_argument("--max", type=int, default=311)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--out")
    a = ap.parse_args()
    train, test = calls(a.train, a.max_tokens), calls(a.test, a.max_tokens)
    ranked = []
    for l in range(LAYERS):
        f = Counter()
        for ids in train[l]:
            f.update(ids.tolist())
        ranked.append([e for e, _ in f.most_common()])
    grid = list(range(a.min, a.max + 1, a.step))
    assert a.base in grid, "base must lie on the grid"
    table = [{p: misses(test[l], ranked[l], p, a.lru) for p in grid} for l in range(LAYERS)]
    alloc = [a.base] * LAYERS
    total = lambda al: sum(table[l][al[l]] for l in range(LAYERS))
    before = total(alloc)
    while True:
        gains = [(table[l][alloc[l]] - table[l][alloc[l] + a.step], l) for l in range(LAYERS) if alloc[l] + a.step <= a.max]
        losses = [(table[l][alloc[l] - a.step] - table[l][alloc[l]], l) for l in range(LAYERS) if alloc[l] - a.step >= a.min]
        g, gl = max(gains); c, cl = min(x for x in losses if x[1] != gl)
        if g <= c:
            break
        alloc[gl] += a.step; alloc[cl] -= a.step
    after = total(alloc)
    steps = len(test[0])
    print(f"held-out decode calls per layer: {steps}; misses at {a.base} pinned everywhere: {before} ({before/steps:.2f}/step); "
          f"per-layer allocation: {after} ({after/steps:.2f}/step), {100*(before-after)/before:.1f}% fewer")
    print("pinned per layer:", alloc)
    print("misses/call at base by layer:", [round(table[l][a.base] / max(len(test[l]), 1), 2) for l in range(LAYERS)])
    if a.out:
        json.dump(dict(pinned=alloc, lru=a.lru, before=before, after=after), open(a.out, "w"))


if __name__ == "__main__":
    main()
