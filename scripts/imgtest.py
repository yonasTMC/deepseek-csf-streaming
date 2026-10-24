import base64, json, struct, sys, time, urllib.request, zlib
def png(w, h, px):
    raw = b"".join(b"\x00" + b"".join(bytes(px(x, y)) for x in range(w)) for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
img = png(256, 256, lambda x, y: (30, 60, 220) if 100 <= y < 156 else (220, 30, 30))
url = "data:image/png;base64," + base64.b64encode(img).decode()
p = dict(model="DeepSeek-V4.1-Flash-Abliterated", max_tokens=200, temperature=0, chat_template_kwargs=dict(thinking=False),
         messages=[dict(role="user", content=[dict(type="text", text="Describe this image: colors and shapes, one sentence."),
                                              dict(type="image_url", image_url=dict(url=url))])])
t = time.time()
try:
    r = json.load(urllib.request.urlopen(urllib.request.Request(sys.argv[1] + "/v1/chat/completions", json.dumps(p).encode(), {"Content-Type": "application/json"}), timeout=600))
    print(f"{time.time()-t:.1f}s prompt_tokens={r['usage']['prompt_tokens']} ->", r["choices"][0]["message"]["content"])
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read()[:400])
