"""Prefill and decode speed at long context: one fresh (uncached) prompt per size."""
import argparse
import json
import random
import time
import urllib.request

WORDS = ["alpha", "beta", "gamma", "delta", "kernel", "stream", "cache", "expert", "token", "vector"]


def run(url, target_tokens, decode_tokens, seed):
    rng = random.Random(seed)
    # ~2 tokens per generated word (measured: 3000 words -> 6025 tokens).
    text = " ".join(rng.choice(WORDS) + str(rng.randint(0, 999)) for _ in range(target_tokens // 2))
    payload = dict(model="DeepSeek-V4.1-Flash-streaming", max_tokens=decode_tokens, temperature=0,
                   ignore_eos=True, stream=True, stream_options=dict(include_usage=True),
                   chat_template_kwargs=dict(thinking=False),
                   messages=[dict(role="user", content=f"[run {seed}] Log follows.\n{text}\n\n"
                                  "Describe the log's structure in detail.")])
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    start, stamps, usage = time.monotonic(), [], None
    with urllib.request.urlopen(req, timeout=4 * 3600) as r:
        for raw in r:
            if raw.startswith(b"data: ") and raw[6:].strip() != b"[DONE]":
                event = json.loads(raw[6:])
                usage = event.get("usage") or usage
                if event["choices"]:
                    stamps.append(time.monotonic() - start)
    prompt, done = usage["prompt_tokens"], usage["completion_tokens"]
    decode = stamps[-1] - stamps[0]
    return dict(prompt_tokens=prompt, ttft_s=round(stamps[0], 1),
                prefill_tok_s=round(prompt / stamps[0], 1),
                decode_tokens=done, decode_tok_s=round((done - 1) / decode, 2) if decode else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8008")
    ap.add_argument("--sizes", type=int, nargs="+", default=[32768, 131072, 524288])
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--seed", type=int, default=None, help="default: time-based, so prompts never hit the prefix cache")
    a = ap.parse_args()
    base = a.seed if a.seed is not None else int(time.time() * 1000) % 10**9
    for i, size in enumerate(a.sizes):
        print(json.dumps(dict(target=size, seed=base + i, **run(a.url, size, a.decode, base + i))), flush=True)


if __name__ == "__main__":
    main()
