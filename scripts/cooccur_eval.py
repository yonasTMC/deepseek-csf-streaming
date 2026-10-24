"""Can layers 0-1 of a decode step predict that step's NVMe-class experts in later layers?

python3 scripts/cooccur_eval.py --placement P.json --train T... --test T... [--thresholds 0.3 0.5 0.7]

Decode calls (<= --max-tokens tokens) are grouped into steps (layer indices restart). An expert
(L, e) with L >= --first-target is "NVMe-class" when it is neither pinned nor in the RAM tier of
the placement. Rules (0|1, a) => (L, e) get confidence count(both) / count(a) from the training
steps; on test steps, a target is predicted when any source expert of that step gives it at least
the threshold. Reports precision (predicted targets the step really used), recall (used NVMe-class
targets that were predicted) and predicted reads per step.
"""
import argparse, json
from collections import Counter, defaultdict
import numpy as np
from simulate_cache import LAYERS, load

SOURCES = (0, 1)


def steps(paths, max_tokens):
    cur, last = {}, -1
    for p in paths:
        for layer, ids in load(p):
            if layer >= LAYERS or ids.shape[0] > max_tokens:
                continue
            if layer <= last and cur:
                yield cur; cur = {}
            cur[layer] = set(int(x) for x in np.unique(ids) if x >= 0)
            last = layer
    if cur:
        yield cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placement", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--first-target", type=int, default=4)
    ap.add_argument("--min-support", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=16)
    a = ap.parse_args()
    p = json.load(open(a.placement))
    resident = [set(p["vram"][str(l)]) | set(p["ram"].get(str(l), [])) for l in range(LAYERS)]
    nvme = lambda l, e: e not in resident[l]
    src_count, co = Counter(), defaultdict(Counter)
    n_train = 0
    for st in steps(a.train, a.max_tokens):
        if any(s not in st for s in SOURCES):
            continue
        n_train += 1
        targets = [(l, e) for l in range(a.first_target, LAYERS) for e in st.get(l, ()) if nvme(l, e)]
        for s in SOURCES:
            for e in st[s]:
                src_count[(s, e)] += 1
                co[(s, e)].update(targets)
    rules = {}
    for src, c in co.items():
        n = src_count[src]
        if n < a.min_support:
            continue
        rules[src] = {t: k / n for t, k in c.items() if k / n >= min(a.thresholds)}
    stats = {t: Counter() for t in a.thresholds}
    n_test = 0
    for st in steps(a.test, a.max_tokens):
        if any(s not in st for s in SOURCES):
            continue
        n_test += 1
        used = {(l, e) for l in range(a.first_target, LAYERS) for e in st.get(l, ()) if nvme(l, e)}
        best = defaultdict(float)
        for s in SOURCES:
            for e in st[s]:
                for t, conf in rules.get((s, e), {}).items():
                    best[t] = max(best[t], conf)
        for thr in a.thresholds:
            pred = {t for t, c in best.items() if c >= thr}
            stats[thr]["pred"] += len(pred); stats[thr]["hit"] += len(pred & used); stats[thr]["used"] += len(used)
    print(f"training steps {n_train}, test steps {n_test}, rules from {len(rules)} source experts")
    for thr in a.thresholds:
        s = stats[thr]
        print(f"threshold {thr:.1f}: precision {100*s['hit']/max(s['pred'],1):5.1f}%  recall {100*s['hit']/max(s['used'],1):5.1f}%  "
              f"predicted reads/step {s['pred']/n_test:6.1f}  useful {s['hit']/n_test:5.1f}  NVMe-class used/step {s['used']/n_test:5.1f}")


if __name__ == "__main__":
    main()
