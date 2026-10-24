"""Which first-use host CUDA operations block behind a spinning kernel?"""
import threading
import time

import torch

from csf_streaming import fastcache as fc

ext = fc.mailbox_ext()
dev = torch.device("cuda", 0)
box = torch.zeros((4, fc.BOX_WORDS), dtype=torch.int64).pin_memory()
box_dev = ext.device_pointer(box.data_ptr())
table = torch.full((384,), -1, dtype=torch.int32, device=dev)
state = torch.zeros(2, dtype=torch.int64, device=dev)
ids = torch.arange(6, dtype=torch.int64, device=dev)
slots = torch.empty_like(ids)
empty = torch.empty(0, dtype=torch.int64, device=dev)
torch.cuda.synchronize()

steps = [
    ("set_device", lambda st: torch.cuda.set_device(dev)),
    ("new Stream", lambda st: st.__setitem__("s", torch.cuda.Stream(device=dev))),
    ("new Event", lambda st: st.__setitem__("e", torch.cuda.Event())),
    ("pin_memory 9MB", lambda st: st.__setitem__("p", torch.empty(9404416, dtype=torch.uint8).pin_memory())),
    ("H2D 9MB on side stream", lambda st: (torch.cuda.stream(st["s"]).__enter__(),
                                           st.__setitem__("d", st["p"].to(dev, non_blocking=True)))),
    ("record+sync event", lambda st: (st["e"].record(st["s"]), st["e"].synchronize())),
    ("H2D 4B pinned slice", lambda st: (table[0:1].copy_(torch.full((1,), 3, dtype=torch.int32).pin_memory(), non_blocking=True), st["s"].synchronize())),
]
for name, fn in steps:
    state.zero_(); box.zero_()
    torch.cuda.synchronize()
    out = {}
    def host():
        st = host.st
        while box.numpy()[0, 0] < 1:
            pass
        t = time.perf_counter()
        fn(st)
        out["t"] = time.perf_counter() - t
        box.numpy()[0, 1] = box.numpy()[0, 0]
    host.st = getattr(main, "st", {}) if (main := None) else {}
    host.st = globals().setdefault("ST", {})
    th = threading.Thread(target=host)
    th.start()
    ext.publish(ids, empty, table, slots, state, box_dev, fc.BOX_IDS)
    ext.wait(ids, table, slots, state, box_dev, 3 * 10**9)
    torch.cuda.synchronize()
    th.join()
    print(f"{name:28s} host op {out['t']:.3f}s  timed out={bool(box.numpy()[0, 5])}", flush=True)
