"""Decode speed after a long real-code prompt (agent-like context), fresh prompt per run."""
import argparse, json, sys, time, urllib.request
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from real_prefill import prompt_text

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8008")
ap.add_argument("--model", default="DeepSeek-V4.1-Flash-streaming")
ap.add_argument("--chars", type=int, default=200000)
ap.add_argument("--tokens", type=int, default=768)
ap.add_argument("--runs", type=int, default=2)
a = ap.parse_args()
for _ in range(a.runs):
    seed = int(time.time() * 1000) % 10**9
    body = prompt_text(a.chars, seed)
    payload = dict(model=a.model, max_tokens=a.tokens, temperature=0, ignore_eos=True, stream=True,
                   stream_options=dict(include_usage=True), chat_template_kwargs=dict(thinking=False),
                   messages=[dict(role="user", content=f"[{seed}]\n{body}\n\nWrite a detailed code review of the modules above, with concrete refactoring suggestions and example code.")])
    req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"})
    start, stamps, usage = time.monotonic(), [], None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            if raw.startswith(b"data: ") and raw[6:].strip() != b"[DONE]":
                ev = json.loads(raw[6:]); usage = ev.get("usage") or usage
                if ev["choices"]:
                    stamps.append(time.monotonic() - start)
    n = usage["completion_tokens"]; per = n / len(stamps)
    print(json.dumps(dict(prompt_tokens=usage["prompt_tokens"], ttft_s=round(stamps[0], 1), tokens=n,
                          tokens_per_step=round(per, 2), decode_tok_s=round((n - per) / (stamps[-1] - stamps[0]), 2))), flush=True)
