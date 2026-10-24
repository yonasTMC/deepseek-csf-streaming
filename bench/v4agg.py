import json, collections, sys
for r in (0, 1):
    d = json.load(open(f"{sys.argv[1]}/v4-rank{r}.json"))
    t = collections.Counter()
    for l, s in d.items():
        t.update(s)
    req = t["hits"] + t["misses"]
    w = max(1, t["miss_waits"])
    print(f"rank{r} calls {t['calls']} hit {t['hits'] / req:.4f} misses {t['misses']} miss_waits {t['miss_waits']} "
          f"misses/wait {t['misses'] / w:.2f} service ms/wait {t['service_ns'] / 1e6 / w:.3f} "
          f"read-wait ms/wait {t['read_wait_ns'] / 1e6 / w:.3f}")
    print(f"  fills: ram {t['fill_ram']} readahead-ready {t['fill_cache_ready']} readahead-inflight {t['fill_cache_waited']} "
          f"demand-read {t['fill_read']}; prefetch reads {t['prefetch_reads']} used {t['prefetch_used']} skipped {t['prefetch_skipped']}")
