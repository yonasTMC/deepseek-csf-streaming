"""GPU fill bandwidth: wait_fill copying expert records from pinned host memory (no spin)."""
import json
import torch
from csf_streaming import fastcache as fc

ext, mb = fc.v4_ext(), fc.mailbox_ext()
dev = torch.device("cuda", 0)
stride, nfields = 9404416, 8
offsets = [0, 5898240, 8847360, 9216000, 9400320, 9400324, 9400328, 9400332]
sizes = [5898240, 2949120, 368640, 184320, 4, 4, 4, 4]
slots = 64
dst = [torch.empty((slots, b), dtype=torch.uint8, device=dev) for b in sizes]
host = torch.randint(0, 255, (16, stride), dtype=torch.uint8).pin_memory()
box = torch.zeros((1, fc.BOX_WORDS), dtype=torch.int64).pin_memory()
ack = torch.zeros((1, ext.ACK_WORDS), dtype=torch.int64).pin_memory()
state = torch.zeros(2, dtype=torch.int64, device=dev)
res = {}
for n in (1, 2, 4, 8):
    for blocks in (24, 32, 48, 64):
        a = ack.numpy()[0]
        a[0], a[1] = n, 0
        for i in range(n):
            a[2 + 2 * i] = (i << 32) | i
            a[3 + 2 * i] = host.data_ptr() + i * stride
        state.copy_(torch.tensor([1, 1]))
        box.numpy()[0, 1] = 10**12            # already acknowledged: no spin
        args = (state, box.data_ptr(), ack.data_ptr(), [t.data_ptr() for t in dst], sizes, offsets, 10**10, blocks)
        for _ in range(3):
            ext.wait_fill(*args)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(20):
            ext.wait_fill(*args)
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / 20
        res[f"n{n}_b{blocks}"] = dict(ms=round(ms, 3), GBps=round(n * stride / ms / 1e6, 1))
    ok = all(torch.equal(dst[f][i], host[i, offsets[f]:offsets[f] + sizes[f]].to(dev)) for i in range(n) for f in range(nfields))
    res[f"n{n}_correct"] = ok
print(json.dumps(res))
