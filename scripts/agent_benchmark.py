"""Run independent tool-using coding sessions against the development endpoint.

Execute on the gpu host. Generated code runs only in a networkless, resource-
limited container whose sole host mount is that session's fixture directory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time
import urllib.request

IMAGE = "ghcr.io/local-inference-lab/vllm@sha256:fd74019fc217a541060093d18427c36330df6ac5700651b55593dafdbb936920"
TASKS = [
    dict(name="intervals", prompt="Fix merge_intervals so it merges overlapping and touching intervals, handles unsorted input and empty input, and never mutates its input. Inspect the files and run the tests before reporting completion.",
         code="def merge_intervals(items):\n    result = []\n    for start, end in items:\n        if result and start < result[-1][1]:\n            result[-1][1] = end\n        else:\n            result.append([start, end])\n    return result\n",
         tests="import unittest\nfrom solution import merge_intervals\nclass Test(unittest.TestCase):\n def test_cases(self):\n  self.assertEqual(merge_intervals([]), [])\n  self.assertEqual(merge_intervals([[5,7],[1,3],[3,6]]), [[1,7]])\n  self.assertEqual(merge_intervals([[1,10],[2,3]]), [[1,10]])\n  self.assertEqual(merge_intervals([[1,2],[4,5]]), [[1,2],[4,5]])\n def test_immutable(self):\n  a=[[3,5],[1,4]]; merge_intervals(a); self.assertEqual(a,[[3,5],[1,4]])\n"),
    dict(name="retry", prompt="Fix retry_delay. Attempt numbers start at zero; delay is min(cap, base * 2**attempt). Reject negative attempts and negative base/cap with ValueError. Very large attempts must work without OverflowError. Inspect the files and run tests before reporting completion.",
         code="def retry_delay(attempt, base=0.5, cap=30.0):\n    return max(cap, base * 2 ** (attempt + 1))\n",
         tests="import unittest\nfrom solution import retry_delay\nclass Test(unittest.TestCase):\n def test_cases(self):\n  self.assertEqual(retry_delay(0),0.5)\n  self.assertEqual(retry_delay(3),4.0)\n  self.assertEqual(retry_delay(100000),30.0)\n  self.assertEqual(retry_delay(100000,0,30),0)\n  self.assertEqual(retry_delay(5,1,0),0)\n def test_invalid(self):\n  for args in [(-1,), (0,-1,30),(0,1,-1)]:\n   with self.assertRaises(ValueError): retry_delay(*args)\n"),
]
TOOLS = [dict(type="function", function=dict(name="read_file", description="Read solution.py or test_solution.py", parameters=dict(type="object", properties=dict(name=dict(type="string",enum=["solution.py","test_solution.py"])),required=["name"]))),
         dict(type="function", function=dict(name="write_solution", description="Replace solution.py", parameters=dict(type="object",properties=dict(code=dict(type="string")),required=["code"]))),
         dict(type="function", function=dict(name="run_tests", description="Run the unit tests in an isolated container", parameters=dict(type="object",properties={})))]


def complete(url, payload):
    payload.update(stream=True, stream_options={"include_usage": True})
    started = time.monotonic()
    first = None
    message = dict(role="assistant", content="")
    calls = {}
    usage, finish = None, None
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if first is None and any(delta.get(k) for k in ("content", "reasoning_content", "tool_calls")):
                    first = time.monotonic() - started
                for key in ("content", "reasoning_content"):
                    if delta.get(key):
                        message[key] = message.get(key, "") + delta[key]
                for call in delta.get("tool_calls", []):
                    target = calls.setdefault(call["index"], dict(id="", type="function", function=dict(name="", arguments="")))
                    if call.get("id"):
                        target["id"] = call["id"]
                    for key in ("name", "arguments"):
                        target["function"][key] += call.get("function", {}).get(key) or ""
                finish = choice.get("finish_reason") or finish
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    return message, dict(seconds=time.monotonic()-started, ttft_seconds=first,
                         usage=usage, finish_reason=finish)


def metrics_snapshot():
    path = Path(__file__).resolve().parents[1] / "evidence" / "server-metrics"
    return {p.name: json.loads(p.read_text()) for p in path.glob("rank*-layer*.json")}


def run_session(task, root, url, model, max_tokens):
    directory = root / task["name"]
    directory.mkdir(parents=True, exist_ok=False)
    (directory/"solution.py").write_text(task["code"])
    (directory/"test_solution.py").write_text(task["tests"])
    messages = [dict(role="system",content="You are a coding agent. Use the tools to inspect, fix, and test the requested function. Keep explanations brief."),dict(role="user",content=task["prompt"])]
    started=time.monotonic()
    result=dict(name=task["name"],requests=[],tool_results=[],passed=False)
    for turn in range(8):
        payload=dict(model=model,messages=messages,tools=TOOLS,temperature=0,
                     max_tokens=max_tokens,chat_template_kwargs=dict(thinking=False))
        message, timing = complete(url, payload)
        result["requests"].append(timing)
        messages.append(message)
        print(json.dumps(dict(task=task["name"], turn=turn, **timing)), flush=True)
        calls=message.get("tool_calls") or []
        if not calls:
            break
        for call in calls:
            name=call["function"]["name"]
            args=json.loads(call["function"]["arguments"])
            if name=="read_file" and args["name"] in ("solution.py","test_solution.py"):
                output=(directory/args["name"]).read_text()
            elif name=="write_solution":
                if len(args["code"])>32768: raise ValueError("solution too large")
                (directory/"solution.py").write_text(args["code"])
                result["passed"]=False
                output="Written."
            elif name=="run_tests":
                cmd=["docker","run","--rm","--network","none","--read-only","--cap-drop","ALL",
                     "--security-opt","no-new-privileges","--security-opt","label=disable","--memory","512m",
                     "--cpus","1","--pids-limit","64","--user","65534:65534",
                     "-v",str(directory)+":/fixture:ro","-w","/fixture",
                     "--entrypoint","/opt/venv/bin/python",IMAGE,"-B","-m","unittest","-v"]
                process=subprocess.run(cmd,capture_output=True,text=True,timeout=45)
                output=(process.stdout+process.stderr)[-12000:]
                result["passed"]=process.returncode==0
            else:
                raise ValueError("unexpected tool call")
            result["tool_results"].append(dict(tool=name,output=output))
            messages.append(dict(role="tool",tool_call_id=call["id"],content=output))
    result["seconds"]=time.monotonic()-started
    result["messages"]=messages
    (directory/"receipt.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--url",default="http://127.0.0.1:8008")
    p.add_argument("--model",default="DeepSeek-V4.1-Flash-streaming")
    p.add_argument("--output",required=True)
    p.add_argument("--concurrency",type=int,choices=[1,2],default=1)
    p.add_argument("--max-tokens",type=int,default=512)
    p.add_argument("--run-label",default="")
    a=p.parse_args()
    root=Path(a.output).resolve(); root.mkdir(parents=True,exist_ok=False)
    tasks=[dict(t, prompt=(f"Run {a.run_label}: " if a.run_label else "")+t["prompt"]) for t in TASKS]
    before=metrics_snapshot()
    (root/"cache-before.json").write_text(json.dumps(before,indent=2))
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        results=list(pool.map(lambda t:run_session(t,root,a.url,a.model,a.max_tokens),tasks))
    summary=dict(run_label=a.run_label,concurrency=a.concurrency,wall_seconds=time.monotonic()-started,
                 passed=sum(x["passed"] for x in results),sessions=len(results),
                 completion_tokens=sum(r["usage"]["completion_tokens"] for x in results for r in x["requests"] if r["usage"]))
    after=metrics_snapshot()
    (root/"cache-after.json").write_text(json.dumps(after,indent=2))
    counters=("expert_requests", "expert_hits", "expert_misses", "staged_bytes", "read_seconds", "calls", "tokens")
    summary["cache_delta_both_ranks"]={k:sum(v.get(k,0)-before.get(name,{}).get(k,0) for name,v in after.items()) for k in counters}
    d=summary["cache_delta_both_ranks"]
    summary["expert_miss_fraction"]=d["expert_misses"]/max(1,d["expert_requests"])
    summary["end_to_end_completion_tokens_per_second"]=summary["completion_tokens"]/summary["wall_seconds"]
    (root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)


if __name__=="__main__": main()
