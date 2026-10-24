import json, time, urllib.request, sys
p=dict(model="DeepSeek-V4.1-Flash-streaming", max_tokens=int(sys.argv[1]), temperature=0,
       messages=[dict(role="user", content="Write a complete, well-commented Python module implementing an LRU cache with TTL expiry, thread safety, and a small CLI demo. Then write unit tests for it.")])
t=time.time(); r=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8008/v1/chat/completions", json.dumps(p).encode(), {"Content-Type":"application/json"}), timeout=7200))
print(json.dumps(dict(seconds=round(time.time()-t,1), usage=r["usage"])))
