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


def post_lru(seq, ranked, pinned, lru_cap):
    pin = set(ranked[:pinned]); lru = OrderedDict(); c = Counter()
    for ids in seq:
        for e in ids.tolist():
            if e in pin:
                continue
            if lru_cap > 0 and e in lru:
                lru.move_to_end(e); continue
            c[e] += 1
            if lru_cap > 0:
                lru[e] = None
                if len(lru) > lru_cap:
                    lru.popitem(last=False)
    return sorted(c.values(), reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--lru", type=int, default=28)
    ap.add_argument("--base", type=int, default=179)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--min", type=int, default=123)
    ap.add_argument("--max", type=int, default=347)
    ap.add_argument("--ram", type=int, default=4600,
                    help="RAM tier experts for cost-weighted NVMe/RAM optimization (0 = unweighted misses)")
    ap.add_argument("--nvme-weight", type=float, default=5.0,
                    help="cost of an NVMe miss relative to a RAM-tier miss")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--out")
    a = ap.parse_args()
    train, test = calls(a.train, a.max_tokens), calls(a.test, a.max_tokens)
    ranked = []
    for l in range(LAYERS):
        f = Counter()
        for ids in train[l]:
            f.update(ids.tolist())
        ranked.append([e for e, _ in f.most_common()] + [e for e in range(384) if e not in f])
    grid = list(range(a.min, a.max + 1, a.step))
    assert a.base in grid, "base must lie on the grid"
    hists = [{p: post_lru(test[l], ranked[l], p, a.lru) for p in grid} for l in range(LAYERS)]

    def breakdown(al):
        merged = []
        for l in range(LAYERS):
            merged.extend(hists[l][al[l]])
        merged.sort(reverse=True)
        ram_m = sum(merged[:a.ram]) if a.ram > 0 else 0
        nvme_m = sum(merged[a.ram:]) if a.ram > 0 else sum(merged)
        cost = (ram_m + a.nvme_weight * nvme_m) if a.ram > 0 else float(nvme_m)
        return cost, ram_m, nvme_m

    alloc = [a.base] * LAYERS
    before_cost, before_ram, before_nvme = breakdown(alloc)
    while True:
        cur_cost, _, _ = breakdown(alloc)
        best_cost, best_pair = cur_cost, None
        plus = [l for l in range(LAYERS) if alloc[l] + a.step <= a.max]
        minus = [l for l in range(LAYERS) if alloc[l] - a.step >= a.min]
        for gl in plus:
            alloc[gl] += a.step
            for cl in minus:
                if cl == gl:
                    continue
                alloc[cl] -= a.step
                c, _, _ = breakdown(alloc)
                if c < best_cost:
                    best_cost, best_pair = c, (gl, cl)
                alloc[cl] += a.step
            alloc[gl] -= a.step
        if best_pair is None:
            break
        alloc[best_pair[0]] += a.step
        alloc[best_pair[1]] -= a.step
    after_cost, after_ram, after_nvme = breakdown(alloc)
    steps = max(len(test[0]), 1)
    before = before_ram + before_nvme
    after = after_ram + after_nvme
    print(f"held-out decode calls per layer: {steps}; base={a.base} lru={a.lru} ram={a.ram} w={a.nvme_weight}: "
          f"before ram={before_ram/steps:.2f} nvme={before_nvme/steps:.2f}/step (cost {before_cost/steps:.2f}) -> "
          f"after ram={after_ram/steps:.2f} nvme={after_nvme/steps:.2f}/step (cost {after_cost/steps:.2f}, "
          f"{100*(before_cost-after_cost)/max(before_cost,1):.1f}% lower)")
    print("pinned per layer:", alloc)
    if a.out:
        json.dump(dict(pinned=alloc, lru=a.lru, ram=a.ram, nvme_weight=a.nvme_weight,
                       before=before, after=after, before_nvme=before_nvme, after_nvme=after_nvme), open(a.out, "w"))


if __name__ == "__main__":
    main()
