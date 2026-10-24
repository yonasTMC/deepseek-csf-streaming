"""Teacher-forced next-token distributions for fixed texts (vLLM prompt_logprobs).

Collect: python3 scripts/logprob_eval.py collect --out run.json [--url ...]
Compare: python3 scripts/logprob_eval.py compare ref.json test.json

Texts are Python standard-library source (/usr/lib64/python3*), model-written markdown/code and
reasoning prose from evidence/junk-tokens, each
cut to about --tokens tokens by the server tokenizer. Comparison uses each
position's top-k lists: KL(ref || test) over the reference's top-k support (test
log-probabilities missing from its top-k are set to its k-th value), top-1
agreement, and flips where the reference's top-1 margin exceeds 1 nat.
"""
import argparse, glob, json, math, random, sys, urllib.request


def texts(n_chars):
    files = sorted(glob.glob("/usr/lib64/python3*/**/*.py", recursive=True))
    random.Random(7).shuffle(files)
    code = "".join(open(f, errors="ignore").read() for f in files[:40])[:n_chars]
    here = __file__.rsplit("/", 2)[0] + "/evidence/junk-tokens"
    junk = sorted(glob.glob(f"{here}/junkH-T0.0-off-*.txt"))
    answer = "".join(open(f).read() for f in junk)[:n_chars]
    reasoning = "".join(open(f).read() for f in sorted(glob.glob(f"{here}/junk-T1.0-on-*.txt")))[:n_chars]
    return {"code": code, "model_answer": answer, "reasoning_prose": reasoning}


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=7200))


def collect(a):
    out = {}
    for name, text in texts(a.chars).items():
        r = post(a.url + "/v1/completions", dict(model=a.model, prompt=text, max_tokens=1, temperature=0,
                                                 prompt_logprobs=a.k))
        rows = []
        for pos in r["choices"][0]["prompt_logprobs"][1:]:
            rows.append({k: v["logprob"] for k, v in pos.items()})
        out[name] = rows
        print(f"{name}: {len(rows)} positions", flush=True)
    json.dump(out, open(a.out, "w"))


def compare(a):
    ref, test = json.load(open(a.ref)), json.load(open(a.test))
    total = dict(n=0, kl=0.0, agree=0, confident_flips=0)
    for name in ref:
        kls, agree, flips = [], 0, 0
        for r, t in zip(ref[name], test[name]):
            r = {k: v for k, v in r.items() if math.isfinite(v)}
            t = {k: v for k, v in t.items() if math.isfinite(v)}
            floor = min(t.values())
            kl = sum(math.exp(lp) * (lp - t.get(tok, floor)) for tok, lp in r.items())
            kls.append(max(kl, 0.0))
            rs = sorted(r.items(), key=lambda x: -x[1])
            top_r, top_t = rs[0][0], max(t.items(), key=lambda x: x[1])[0]
            if top_r == top_t:
                agree += 1
            elif len(rs) > 1 and rs[0][1] - rs[1][1] > 1.0:
                flips += 1
        n = len(kls)
        kls.sort()
        print(f"{name:16s} n={n:5d}  KL mean {sum(kls)/n:.2e}  p99 {kls[int(0.99*(n-1))]:.2e}  max {kls[-1]:.2e}  "
              f"top1 agree {100*agree/n:6.2f}%  confident flips {flips}")
        total["n"] += n; total["kl"] += sum(kls); total["agree"] += agree; total["confident_flips"] += flips
    n = total["n"]
    print(f"{'ALL':16s} n={n:5d}  KL mean {total['kl']/n:.2e}  top1 agree {100*total['agree']/n:6.2f}%  "
          f"confident flips {total['confident_flips']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--url", default="http://127.0.0.1:8008")
    c.add_argument("--model", default="DeepSeek-V4.1-Flash-streaming")
    c.add_argument("--chars", type=int, default=10000)
    c.add_argument("--k", type=int, default=20)
    c.add_argument("--out", required=True)
    m = sub.add_parser("compare")
    m.add_argument("ref"); m.add_argument("test")
    a = ap.parse_args()
    collect(a) if a.cmd == "collect" else compare(a)
