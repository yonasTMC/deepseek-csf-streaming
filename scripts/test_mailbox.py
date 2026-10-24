"""Standalone GPU<->host mailbox handshake test (one GPU, no model)."""
import threading
import time

import torch

from csf_streaming import fastcache as fc

ext = fc.mailbox_ext()
dev = torch.device("cuda", 0)
box = torch.zeros((4, fc.BOX_WORDS), dtype=torch.int64).pin_memory()
box_np = box.numpy()
box_dev = ext.device_pointer(box.data_ptr())
print("host ptr", hex(box.data_ptr()), "device ptr", hex(box_dev), flush=True)
table = torch.full((384,), -1, dtype=torch.int32, device=dev)
state = torch.zeros(2, dtype=torch.int64, device=dev)
ids = torch.arange(6, dtype=torch.int64, device=dev)
slots = torch.empty_like(ids)
empty = torch.empty(0, dtype=torch.int64, device=dev)

for trial, delay in enumerate((0.5, 0.0, 0.2)):
    def host():
        while box_np[0, 0] < trial + 1:
            pass
        seen = time.perf_counter()
        time.sleep(delay)
        side = torch.cuda.Stream(device=dev)
        src = torch.arange(10, 16, dtype=torch.int32).pin_memory()
        with torch.cuda.stream(side):
            table[:6].copy_(src, non_blocking=True)
        side.synchronize()
        box_np[0, 1] = box_np[0, 0]
        print(f"trial {trial}: host saw seq {box_np[0, 0]} miss {box_np[0, 3]}, acked after {time.perf_counter() - seen:.3f}s", flush=True)
    t = threading.Thread(target=host)
    t.start()
    table.fill_(-1)
    start = time.perf_counter()
    ext.publish(ids, empty, table, slots, state, box_dev, fc.BOX_IDS)
    ext.wait(ids, table, slots, state, box_dev, 5 * 10**9)
    torch.cuda.synchronize()
    t.join()
    print(f"trial {trial}: GPU wait returned after {time.perf_counter() - start:.3f}s, "
          f"slots {slots.tolist()}, error flag {box_np[0, 5]}", flush=True)
