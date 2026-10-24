"""Decode-only tokens/s for one or more concurrent streams (ignore_eos)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
import urllib.request


def stream(url, tokens, prompt):
    payload = dict(model="DeepSeek-V4.1-Flash-streaming", max_tokens=tokens, temperature=0,
                   ignore_eos=True, stream=True, stream_options=dict(include_usage=True),
                   messages=[dict(role="user", content=prompt)])
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    start, stamps, tokens = time.monotonic(), [], None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            if raw.startswith(b"data: ") and raw[6:].strip() != b"[DONE]":
                event = json.loads(raw[6:])
                if event.get("usage"):
                    tokens = event["usage"]["completion_tokens"]
                if event["choices"]:
                    stamps.append(time.monotonic() - start)
    decode = stamps[-1] - stamps[0]
    # Speculative decoding streams several tokens per chunk; the first chunk's
    # share is approximated by the mean tokens per chunk.
    per_chunk = tokens / len(stamps)
    return dict(ttft=round(stamps[0], 2), chunks=len(stamps), tokens=tokens,
                tokens_per_step=round(per_chunk, 2),
                decode_tok_s=round((tokens - per_chunk) / decode, 2) if decode else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8008")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--streams", type=int, default=1)
    a = ap.parse_args()
    prompts = ["Explain how a B-tree insertion works, step by step.",
               "Write a Python function that parses ISO-8601 durations."]
    with ThreadPoolExecutor(a.streams) as pool:
        res = list(pool.map(lambda i: stream(a.url, a.tokens, prompts[i % 2]), range(a.streams)))
    print(json.dumps(dict(streams=a.streams, tokens=a.tokens, results=res)))


if __name__ == "__main__":
    main()
