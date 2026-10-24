"""Prefill on realistic text: real source files, fresh offset per run (no prefix-cache reuse)."""
import argparse
import glob
import json
import random
import time
import urllib.request


def prompt_text(chars, seed):
    files = sorted(glob.glob("/usr/lib64/python3*/**/*.py", recursive=True) +
                   glob.glob("/usr/lib/python3*/**/*.py", recursive=True))
    random.Random(seed).shuffle(files)
    out, n = [], 0
    for f in files:
        try:
            body = open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        out.append(f"### {f}\n{body}")
        n += len(body)
        if n >= chars:
            break
    return "".join(out)[:chars]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8008")
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash-Abliterated")
    ap.add_argument("--chars", type=int, nargs="+", default=[200000])
    a = ap.parse_args()
    for chars in a.chars:
        seed = int(time.time() * 1000) % 10**9
        body = prompt_text(chars, seed)
        payload = dict(model=a.model, max_tokens=8, temperature=0, stream=True,
                       stream_options=dict(include_usage=True), chat_template_kwargs=dict(thinking=False),
                       messages=[dict(role="user", content=f"[{seed}]\n{body}\n\nWhich module above defines the most functions? One word.")])
        req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
        start, first, usage = time.monotonic(), None, None
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                if raw.startswith(b"data: ") and raw[6:].strip() != b"[DONE]":
                    ev = json.loads(raw[6:])
                    usage = ev.get("usage") or usage
                    if ev["choices"] and first is None:
                        first = time.monotonic() - start
        n = usage["prompt_tokens"]
        print(json.dumps(dict(prompt_tokens=n, ttft_s=round(first, 1), prefill_tok_s=round(n / first, 1))), flush=True)


if __name__ == "__main__":
    main()
