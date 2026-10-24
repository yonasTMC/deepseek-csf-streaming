"""Long agent-like coding generations; count junk at temperature 0 vs 1.0.

Junk signals: CJK characters inside code blocks, and Python code blocks that
fail ast.parse. Context: ~20k tokens of real source plus a coding task.
"""
import argparse, ast, json, re, sys, time, urllib.request
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from real_prefill import prompt_text

TASKS = [
    "Write a complete Python module implementing an LRU cache with TTL expiry and thread safety, plus unittest tests.",
    "Write a Python script that parses a JSON lines log file, aggregates per-endpoint latency percentiles, and prints a table. Include argparse and tests.",
    "Write a Python asyncio HTTP health checker with retries, exponential backoff and a summary report, plus tests.",
]
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
BLOCK = re.compile(r"```(?:python|py)?\n(.*?)```", re.S)

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8008")
ap.add_argument("--model", default="DeepSeek-V4.1-Flash-streaming")
ap.add_argument("--chars", type=int, default=80000)
ap.add_argument("--temps", type=float, nargs="+", default=[0.0, 1.0])
ap.add_argument("--thinking", choices=["on", "off"], default="on")
ap.add_argument("--max-tokens", type=int, default=12000)
ap.add_argument("--save", default="/tmp/junk")
ap.add_argument("--tasks", type=int, default=len(TASKS))
a = ap.parse_args()
context = prompt_text(a.chars, 777)
for t in a.temps:
    tot = dict(temp=t, tokens=0, blocks=0, bad_blocks=0, cjk_in_code=0, cjk_in_reasoning=0, seconds=0.0)
    for task in TASKS[:a.tasks]:
        p = dict(model=a.model, max_tokens=a.max_tokens, temperature=t, top_p=0.95 if t > 0 else 1.0,
                 chat_template_kwargs=dict(thinking=a.thinking == "on"),
                 messages=[dict(role="user", content=f"Reference code for style:\n{context}\n\nTask: {task} Output the code in ```python blocks.")])
        s = time.time()
        r = json.load(urllib.request.urlopen(urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(p).encode(), {"Content-Type": "application/json"}), timeout=3600))
        tot["seconds"] += time.time() - s
        m = r["choices"][0]["message"]; text = m.get("content") or ""; reasoning = m.get("reasoning") or m.get("reasoning_content") or ""
        tot["tokens"] += r["usage"]["completion_tokens"]
        tot["cjk_in_reasoning"] += len(CJK.findall(reasoning))
        tot.setdefault("finish", []).append(r["choices"][0]["finish_reason"])
        n = TASKS.index(task)
        open(f"{a.save}-T{t}-{a.thinking}-{n}.txt", "w").write(f"=== REASONING\n{reasoning}\n=== CONTENT\n{text}\n")
        for where, body in (("reasoning", reasoning), ("content", text)):
            for mt in CJK.finditer(body):
                lo, hi = max(0, mt.start() - 60), min(len(body), mt.end() + 40)
                print(f"  T={t} {where}: ...{body[lo:hi]!r}...", flush=True)
        for code in BLOCK.findall(text):
            tot["blocks"] += 1
            tot["cjk_in_code"] += len(CJK.findall(code))
            try:
                ast.parse(code)
            except SyntaxError as e:
                tot["bad_blocks"] += 1
                print(f"  T={t} syntax error: {e.msg} at line {e.lineno}: {code.splitlines()[e.lineno-1][:100] if e.lineno and e.lineno <= len(code.splitlines()) else ''}", flush=True)
    tot["seconds"] = round(tot["seconds"], 1)
    print(json.dumps(tot), flush=True)
