"""Capability benchmarks run through the public gateway, as any other client would.

Needs `datasets` (pip install datasets) only to download the suites. The server is reached over
plain HTTP; nothing here touches the pod or the engine.

  export MESO_GATEWAY_API_KEY=...
  python3 scripts/capability_eval.py --suites gsm8k,math,humaneval,mmlupro --n 20 --thinking off
  python3 scripts/capability_eval.py --n 200 --thinking on --out evidence/capability/run-on.json

Suites: gsm8k, math (MATH-500), humaneval (executed in a subprocess with a timeout), mmlupro
(sample stratified across its 14 categories). Output JSON holds the score per suite and every
item's prompt, response, reasoning, verdict and token counts, so a run can be re-graded.
"""
import argparse, json, os, random, re, subprocess, sys, tempfile, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("CSF_EVAL_URL", "https://api.meshesha.solutions/v1")
MODEL = os.environ.get("CSF_EVAL_MODEL", "DeepSeek-V4.1-Flash-Abliterated")


def chat(prompt, a, max_tokens):
    body = dict(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=a.temperature,
                max_tokens=max_tokens, chat_template_kwargs=dict(thinking=a.thinking == "on"))
    req = urllib.request.Request(URL + "/chat/completions", json.dumps(body).encode(), {
        "Content-Type": "application/json", "Authorization": "Bearer " + os.environ["MESO_GATEWAY_API_KEY"]})
    last = None
    for attempt in range(3):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=1800))
            m = r["choices"][0]["message"]
            return dict(content=m.get("content") or "", reasoning=m.get("reasoning") or m.get("reasoning_content") or "",
                        finish=r["choices"][0]["finish_reason"], tokens=r["usage"]["completion_tokens"])
        except Exception as e:  # transient gateway errors
            last = e
            time.sleep(5 * (attempt + 1))
    return dict(content="", reasoning="", finish="error:" + str(last)[:100], tokens=0)


# ---------- answer extraction ----------
def last_boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    i += 7; depth = 1; j = i
    while j < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0); j += 1
    return s[i:j - 1] if depth == 0 else None


def norm_math(x):
    x = (x or "").strip().strip("$")
    x = re.sub(r"\\(text|mathrm|mathbf)\{([^}]*)\}", r"\2", x)
    x = x.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    x = x.replace("dfrac", "frac").replace("tfrac", "frac").replace("^\\circ", "").replace("^{\\circ}", "")
    x = x.replace("\\%", "").replace(" ", "").rstrip(".")
    x = re.sub(r"^[a-zA-Z]=", "", x)
    return x


# ---------- suites: each returns list of (id, prompt, grader(resp)->bool, max_tokens) ----------
def suite_gsm8k(n, rng):
    from datasets import load_dataset
    d = list(load_dataset("openai/gsm8k", "main", split="test")); rng.shuffle(d)
    out = []
    for i, r in enumerate(d[:n]):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        p = r["question"] + "\n\nSolve step by step. End with the final numeric answer in \\boxed{}."
        def g(resp, gold=gold):
            b = last_boxed(resp["content"]) or ""
            m = re.findall(r"-?\d+(?:\.\d+)?", b.replace(",", ""))
            return bool(m) and float(m[-1]) == float(gold)
        out.append((f"gsm8k-{i}", p, g))
    return out


def suite_math(n, rng):
    from datasets import load_dataset
    d = list(load_dataset("HuggingFaceH4/MATH-500", split="test")); rng.shuffle(d)
    out = []
    for r in d[:n]:
        p = r["problem"] + "\n\nReason step by step and put the final answer in \\boxed{}."
        def g(resp, gold=r["answer"]):
            return norm_math(last_boxed(resp["content"])) == norm_math(gold)
        out.append((f"math-{r['unique_id']}", p, g))
    return out


def run_code(code, timeout=20):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
    try:
        return subprocess.run([sys.executable, "-I", f.name], capture_output=True, timeout=timeout,
                              cwd=tempfile.gettempdir()).returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(f.name)


def suite_humaneval(n, rng):
    from datasets import load_dataset
    d = list(load_dataset("openai/openai_humaneval", split="test")); rng.shuffle(d)
    out = []
    for r in d[:n]:
        p = ("Complete this Python function. Reply with one ```python code block containing the full "
             "function, including the signature.\n\n```python\n" + r["prompt"] + "```")
        def g(resp, r=r):
            blocks = re.findall(r"```(?:python)?\n(.*?)```", resp["content"], re.S)
            if not blocks:
                return False
            body = blocks[-1]
            # model may return the full function or only a body
            code = body if f"def {r['entry_point']}" in body else r["prompt"] + body
            return run_code(code + "\n\n" + r["test"] + f"\n\ncheck({r['entry_point']})\n")
        out.append((r["task_id"], p, g))
    return out


def suite_mmlupro(n, rng):
    from datasets import load_dataset
    d = list(load_dataset("TIGER-Lab/MMLU-Pro", split="test"))
    by = {}
    for r in d:
        by.setdefault(r["category"], []).append(r)
    per = max(1, n // len(by)); picked = []
    for c in sorted(by):
        rng.shuffle(by[c]); picked += by[c][:per]
    out = []
    for r in picked:
        L = "ABCDEFGHIJ"
        opts = "\n".join(f"{L[i]}. {o}" for i, o in enumerate(r["options"]))
        p = (f"{r['question']}\n\n{opts}\n\nThink step by step, then finish with a line of the form "
             "\"The answer is X\" where X is one letter.")
        def g(resp, gold=r["answer"]):
            m = re.findall(r"answer is\s*\(?\**([A-J])\b", resp["content"])
            return bool(m) and m[-1] == gold
        out.append((f"mmlupro-{r['category']}-{r['question_id']}", p, g))
    return out


SUITES = dict(gsm8k=(suite_gsm8k, 4096), math=(suite_math, 8192), humaneval=(suite_humaneval, 4096),
              mmlupro=(suite_mmlupro, 8192))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--n", type=int, default=20, help="items per suite")
    ap.add_argument("--thinking", choices=["on", "off"], default="off")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--think-budget", type=int, default=32768, help="max_tokens when thinking is on")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if "MESO_GATEWAY_API_KEY" not in os.environ:
        sys.exit("set MESO_GATEWAY_API_KEY")
    out = dict(model=MODEL, url=URL, thinking=a.thinking, temperature=a.temperature, n=a.n, suites={})
    lock = threading.Lock()
    for name in a.suites.split(","):
        make, budget = SUITES[name]
        items = make(a.n, random.Random(a.seed))
        mt = a.think_budget if a.thinking == "on" else budget
        t0 = time.time(); rows = []; done = [0]

        def work(it):
            iid, prompt, grade = it
            resp = chat(prompt, a, mt)
            try:
                ok = bool(grade(resp))
            except Exception:
                ok = False
            with lock:
                done[0] += 1
                print(f"\r{name}: {done[0]}/{len(items)}", end="", file=sys.stderr, flush=True)
            return dict(id=iid, ok=ok, prompt=prompt, **resp)

        with ThreadPoolExecutor(a.workers) as ex:
            rows = list(ex.map(work, items))
        wall = time.time() - t0
        toks = sum(r["tokens"] for r in rows)
        s = dict(n=len(rows), correct=sum(r["ok"] for r in rows), acc=sum(r["ok"] for r in rows) / max(1, len(rows)),
                 truncated=sum(r["finish"] == "length" for r in rows), errors=sum(r["finish"].startswith("error") for r in rows),
                 mean_tokens=toks / max(1, len(rows)), wall_s=wall, tok_per_s=toks / wall, items=rows)
        out["suites"][name] = s
        print(f"\r{name:10s} {s['correct']}/{s['n']} = {s['acc']:.1%}  truncated={s['truncated']} errors={s['errors']} "
              f"mean_tokens={s['mean_tokens']:.0f} {s['tok_per_s']:.0f} tok/s total", file=sys.stderr)
        if a.out:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(out, open(a.out, "w"), indent=1)


main()
