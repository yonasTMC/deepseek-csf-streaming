"""Long real-code prompt with a fact hidden mid-way; greedy answer must come back exactly."""
import argparse, json, sys, urllib.request
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from real_prefill import prompt_text

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8008")
ap.add_argument("--model", default="DeepSeek-V4.1-Flash-streaming")
ap.add_argument("--chars", type=int, default=100000)
ap.add_argument("--seed", type=int, default=424242)
a = ap.parse_args()
body = prompt_text(a.chars, a.seed)
mid = len(body) // 2
body = body[:mid] + "\n# NOTE: the deployment passphrase is TANGERINE-7341.\n" + body[mid:]
payload = dict(model=a.model, max_tokens=24, temperature=0, chat_template_kwargs=dict(thinking=False),
               messages=[dict(role="user", content=body + "\n\nWhat is the deployment passphrase mentioned in a NOTE comment above? Reply with the passphrase only.")])
r = json.load(urllib.request.urlopen(urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"}), timeout=3600))
print(json.dumps(dict(prompt_tokens=r["usage"]["prompt_tokens"], answer=r["choices"][0]["message"]["content"].strip())))
