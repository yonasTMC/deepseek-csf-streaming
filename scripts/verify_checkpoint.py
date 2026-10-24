import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import time

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())


def check(shard):
    path = root / "tensors" / shard["file"]
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            digest.update(block)
            size += len(block)
    result = dict(file=shard["file"], bytes=size, sha256=digest.hexdigest())
    result["ok"] = size == shard["target_file_bytes"] and result["sha256"] == shard["target_sha256"]
    print(json.dumps(result), flush=True)
    return result


start = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(check, manifest["shards"]))
summary = dict(ok=all(x["ok"] for x in results), shards=len(results),
               seconds=time.monotonic() - start, results=results)
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2) + "\n")
raise SystemExit(0 if summary["ok"] else 1)
