"""Aggregate and per-stream decode speed at several concurrency levels (T=0, thinking off)."""
import argparse, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPTS = ["Write a Python function that parses ISO-8601 durations.", "Explain how a B-tree insertion works, step by step.",
           "Write a Python class implementing an LRU cache with TTL.", "Explain Dijkstra's algorithm with a worked example."]


def one(url, model, tokens, i):
    p = dict(model=model, max_tokens=tokens, stream=True, ignore_eos=True, temperature=0,
             chat_template_kwargs={"thinking": False}, stream_options={"include_usage": True},
             messages=[{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}])
    t0, first, n = time.time(), None, 0
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(p).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if line.startswith(b"data: {"):
                ev = json.loads(line[6:])
                if ev["choices"] and first is None:
                    first = time.time()
                if ev.get("usage"):
                    n = ev["usage"]["completion_tokens"]
    return dict(ttft=round(first - t0, 2), tok_s=round(n / (time.time() - first), 1), tokens=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8008")
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash-streaming")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--tokens", type=int, default=256)
    a = ap.parse_args()
    for c in a.levels:
        t = time.time()
        with ThreadPoolExecutor(c) as ex:
            res = list(ex.map(lambda i: one(a.url, a.model, a.tokens, i), range(c)))
        wall = time.time() - t
        print(json.dumps(dict(concurrency=c, aggregate_tok_s=round(sum(r["tokens"] for r in res) / wall, 1),
                              streams=res)), flush=True)


if __name__ == "__main__":
    main()
