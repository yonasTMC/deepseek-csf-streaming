"""Exercise the v4 host service without a GPU: fake packs, fake mailboxes, real reads.

Plays the GPU's side of the handshake (publish -> ack -> per-fill ready -> fill_done)
and checks that every acknowledged fill points at the right expert's bytes, that
every miss is filled exactly once, and that read-ahead hints land on the layer
they name. Run once with the default io_uring path and once with CSF_NO_URING=1:

    python scripts/v4_host_check.py /path/on/an/O_DIRECT-capable/fs
"""
import ctypes
import mmap
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from csf_streaming import fastcache

STRIDE = 4096 * 64            # 256 KiB records; 64 pages split into 8 spec pieces
EXPERTS = 64
CAPACITY = 12
LAYERS = (0, 1, 15, 16)       # 15/16: the old hint encoding confused layer 15 with "same layer"
RAM = (0, 1, 2, 3)            # RAM-tier experts in every layer
PINNED = (8, 9, 10)
HOST_CACHE = 24


def pattern(layer, expert):
    return np.full(STRIDE // 4, layer * 1000 + expert, dtype=np.int32)


def aligned(nbytes):
    buf = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    return buf, ctypes.addressof(ctypes.c_char.from_buffer(buf))


def main():
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    for layer in LAYERS:
        with open(root / f"layer{layer}.bin", "wb") as f:
            for e in range(EXPERTS):
                f.write(pattern(layer, e).tobytes())
    ext = fastcache.v4_ext()
    words = ext.ACK_WORDS
    ack_ready = 2 + 3 * ext.MAX_FILL
    keep = []
    cache, cache_ptr = aligned(HOST_CACHE * STRIDE)
    ram, ram_ptr = aligned(len(LAYERS) * len(RAM) * STRIDE)
    keep += [cache, ram]
    box = torch.zeros((max(LAYERS) + 1, fastcache.BOX_WORDS), dtype=torch.int64)
    ack = torch.zeros((max(LAYERS) + 1, words), dtype=torch.int64)
    ext.init(cache_ptr, HOST_CACHE, STRIDE, 4, 8)
    for li, layer in enumerate(LAYERS):
        ptrs = [0] * EXPERTS
        for ri, e in enumerate(RAM):
            off = (li * len(RAM) + ri) * STRIDE
            record = pattern(layer, e)
            ctypes.memmove(ram_ptr + off, record.ctypes.data, STRIDE)
            ptrs[e] = ram_ptr + off
        resident = list(PINNED) + [11, 12]               # two LRU residents
        ext.register_layer(layer, box[layer].data_ptr(), ack[layer].data_ptr(), EXPERTS, CAPACITY,
                           resident, list(range(len(resident))), list(PINNED),
                           list(range(len(resident), CAPACITY)), str(root / f"layer{layer}.bin"), ptrs)
    ext.start()
    print("io mode:", ext.io_mode())
    seqs = {layer: 0 for layer in LAYERS}
    resident = {layer: set(PINNED) | {11, 12} for layer in LAYERS}

    def request(layer, ids, hints=()):
        seqs[layer] += 1
        seq = seqs[layer]
        b = box[layer]
        payload = b[fastcache.BOX_HEADER:].view(torch.int32)
        payload[:len(ids)] = torch.tensor(ids, dtype=torch.int32)
        payload[len(ids):len(ids) + len(hints)] = torch.tensor(list(hints), dtype=torch.int32)
        miss = any(e not in resident[layer] for e in ids)
        b[2], b[3], b[4] = len(ids), int(miss), len(hints)
        b[0] = seq
        if not miss:
            return []
        deadline = time.time() + 10
        while int(b[1]) < seq:
            assert time.time() < deadline, f"layer {layer} seq {seq}: no ack"
        a = ack[layer]
        nfill, nevict = int(a[0]), int(a[1])
        fills = []
        for k in range(nfill):
            while int(a[ack_ready + k]) < seq:
                assert time.time() < deadline, f"layer {layer} seq {seq}: fill {k} never ready"
            w0, src = int(a[2 + 2 * k]), int(a[3 + 2 * k])
            expert, slot = w0 >> 32, w0 & 0xffffffff
            got = np.frombuffer(ctypes.string_at(src, STRIDE), dtype=np.int32)
            assert (got == layer * 1000 + expert).all(), f"layer {layer} expert {expert}: wrong bytes"
            fills.append(expert)
        evicted = [int(a[2 + 2 * ext.MAX_FILL + i]) for i in range(nevict)]
        b[6] = seq                                        # finalize: host buffers may be reused
        need = sorted({e for e in ids if e not in resident[layer]})
        assert sorted(fills) == need, f"layer {layer}: filled {sorted(fills)}, needed {need}"
        resident[layer] -= set(evicted)
        resident[layer] |= set(fills)
        return fills

    def hint(layer, expert, bulk):
        return (ext.HINT_BULK if bulk else 0) | ((layer + 1) << ext.HINT_LAYER_SHIFT) | expert

    def stats(layer):
        return dict(ext.stats(layer))

    # 1. RAM-tier and NVMe misses in one request (ready fills first, reads in completion order).
    request(0, [8, 0, 1, 20, 21, 22, 23])
    # 2. Bulk hints for layer 1, spec hints for layer 15 (old encoding: misread as layer 0).
    request(0, [9, 24], hints=[hint(1, e, True) for e in (30, 31, 32)] + [hint(15, e, False) for e in (40, 41)])
    time.sleep(0.2)
    s1, s15, s0 = stats(1), stats(15), stats(0)
    assert s1["bulk_reads"] == 3, s1
    assert s15["prefetch_reads"] == 2, s15
    assert s0["prefetch_reads"] == 0 and s0["bulk_reads"] == 0, s0
    # 3. Using the read-ahead: served from host buffers, not new reads.
    request(1, [30, 31, 32, 2])
    request(15, [40, 41])
    s1, s15 = stats(1), stats(15)
    assert s1["prefetch_used"] == 3 and s1["fill_read"] == 0, s1
    assert s15["prefetch_used"] == 2 and s15["fill_read"] == 0, s15
    # 4. Promotion: many spec hints queue behind one another; demand one at once.
    request(16, [8], hints=[hint(16, e, False) for e in range(44, 52)])
    request(16, [51, 50])
    assert stats(16)["fill_read"] == 0, stats(16)
    # 5. Churn: eviction plus host-cache reuse across many requests and layers.
    rng = np.random.default_rng(0)
    for i in range(300):
        layer = LAYERS[i % len(LAYERS)]
        ids = [int(e) for e in rng.choice(np.arange(4, EXPERTS), size=6, replace=False)]
        hints = [hint(LAYERS[(i + 1) % len(LAYERS)], int(e), bool(i % 2))
                 for e in rng.choice(np.arange(4, EXPERTS), size=4, replace=False)]
        request(layer, ids, hints)
    total = {k: sum(stats(l)[k] for l in LAYERS) for k in ("misses", "fill_read", "fill_cache_ready",
                                                           "fill_cache_waited", "prefetch_promoted",
                                                           "fill_ram", "prefetch_used")}
    print("ok", total)


if __name__ == "__main__":
    main()
