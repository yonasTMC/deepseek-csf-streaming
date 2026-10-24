import json, glob, collections, sys
d = sys.argv[1]
for r in (0, 1):
    t = collections.Counter()
    for f in glob.glob(f"{d}/rank{r}-layer*.json"):
        for k, v in json.load(open(f)).items():
            if isinstance(v, (int, float)) and k not in ("layer", "rank", "cache_bytes"):
                t[k] += v
    req = t["hits"] + t["misses"]
    print(r, {k: round(v, 3) for k, v in sorted(t.items())},
          "hit", round(t["hits"] / req, 4) if req else None,
          "ms/miss-wait", round(1e3 * t["miss_service_seconds"] / t["miss_waits"], 3) if t["miss_waits"] else None)
