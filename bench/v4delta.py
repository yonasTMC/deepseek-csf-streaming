import json, sys, collections
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
t = collections.Counter()
for l in b:
    for k, v in b[l].items():
        t[k] += v - a.get(l, {}).get(k, 0)
w = max(1, t["miss_waits"])
print(f"  misses {t['misses']} waits {t['miss_waits']} read-wait {t['read_wait_ns']/1e9:.2f}s | fills ram {t['fill_ram']} "
      f"readahead {t['fill_cache_ready']}+{t['fill_cache_waited']} nvme-demand {t['fill_read']} | prefetch reads {t['prefetch_reads']} "
      f"used {t['prefetch_used']} ({100*t['prefetch_used']/max(1,t['prefetch_reads']):.0f}%) skipped {t['prefetch_skipped']}")
