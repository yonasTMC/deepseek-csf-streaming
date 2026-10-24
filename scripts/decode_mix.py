"""Decode speed by content: code, prose, and thinking mode (DSpark tokens/step from /metrics)."""
import json, re, subprocess, time, urllib.request
import sys
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8008"
M = sys.argv[2] if len(sys.argv) > 2 else "DeepSeek-V4.1-Flash-streaming"

def metrics():
    out = urllib.request.urlopen(URL + "/metrics", timeout=30).read().decode()
    get = lambda k: sum(float(x) for x in re.findall(rf"^{re.escape(k)}\{{.*?\}} ([0-9.e+]+)", out, re.M))
    return get("vllm:spec_decode_num_drafts_total"), get("vllm:generation_tokens_total")

def run(label, prompt, temperature_override=None, **extra):
    p = dict(model=M, max_tokens=256, stream=True, ignore_eos=True, temperature=0 if temperature_override is None else temperature_override,
             stream_options={"include_usage": True}, messages=[{"role": "user", "content": prompt}], **extra)
    d0, g0 = metrics(); t0 = time.time(); times = []
    with urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(p).encode(),
                                {"Content-Type": "application/json"}), timeout=900) as r:
        for line in r:
            if line.startswith(b"data: {"):
                ev = json.loads(line[6:])
                if ev["choices"]:
                    times.append(time.time())
    d1, g1 = metrics(); gen = g1 - g0
    print(f"{label:34s} ttft {times[0]-t0:5.2f}s  decode {gen/(times[-1]-times[0]):5.1f} tok/s  tokens/step {gen/max(d1-d0,1):.2f}", flush=True)

prompts = {
    "code: B-tree": "Explain how a B-tree insertion works, step by step.",
    "code: ISO-8601 parser": "Write a Python function that parses ISO-8601 durations.",
    "prose: history of maths": "Write a detailed encyclopedia article about the history of mathematics from ancient Mesopotamia to the present.",
    "prose: travel story": "Write a vivid short story about a train journey across the Andes.",
}
for name, pr in prompts.items():
    for i in range(2):
        run(f"{name} (no-think) #{i+1}", pr, chat_template_kwargs={"thinking": False})
for i in range(2):
    run(f"thinking: B-tree T=1 #{i+1}", prompts["code: B-tree"], temperature_override=1.0)
