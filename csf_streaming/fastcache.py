"""Copy-only expert streaming over prepacked B12X slot records (engine v2).

Tiers per TP rank: per-layer VRAM slots (a pinned set plus LRU), a static
pinned-RAM pool shared by all layers, and the expert pack on NVMe. A miss only
moves bytes: RAM records are copied straight into the slot; pack records are
read with O_DIRECT into reusable pinned staging and then copied. The compute
stream waits on per-slot CUDA events instead of the host blocking on loads,
except while an NVMe read for a needed expert is still in flight. During
decode, layer L also predicts layer L+1's experts with L+1's router and
prefetches the ones not resident.
"""
import atexit
from collections import Counter, OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import threading
import time

import torch

from . import trace
from .policy import routing_chunks
from .storage import FIELDS, allocate_slots, read_experts, upload_and_prepare

_CACHES = {}
_SHARED = {}
_LOCK = threading.Lock()
_EXT = None
_EXT_LOCK = threading.Lock()  # separate: Shared() calls mailbox_ext() while holding _LOCK

BOX_HEADER = 8                      # int64 words; see csrc/mailbox.cu
BOX_IDS = 8192                      # int32 ids + predictions per layer mailbox
BOX_WORDS = BOX_HEADER + BOX_IDS // 2
MAX_LAYERS = 64
SPLIT_MAX_TOKENS = int(os.environ.get("CSF_SPLIT_MAX_TOKENS", "4096"))
SAME_LAYER_HINT = 1 << 20               # prediction ids >= this are read ahead for the same layer
DEBUG = os.environ.get("CSF_DEBUG") == "1"


V4 = os.environ.get("CSF_V4") == "1"
_V4 = None


def v4_ext():
    global _V4
    with _EXT_LOCK:
        if _V4 is None:
            from torch.utils.cpp_extension import load
            build = Path(os.environ.get("CSF_EXT_BUILD_DIR") or Path(os.environ.get("XDG_CACHE_HOME", "/tmp"))) / "csf-v4-build"
            build.mkdir(parents=True, exist_ok=True)
            _V4 = load(name="csf_v4", sources=[str(Path(__file__).parent / "csrc" / "v4.cu")],
                       extra_cuda_cflags=["-O3"], extra_cflags=["-O3"], build_directory=str(build),
                       verbose=False)
        return _V4


def mailbox_ext():
    global _EXT
    with _EXT_LOCK:
        if _EXT is None:
            from torch.utils.cpp_extension import load
            build = Path(os.environ.get("CSF_EXT_BUILD_DIR") or Path(os.environ.get("XDG_CACHE_HOME", "/tmp"))) / "csf-mailbox-build"
            build.mkdir(parents=True, exist_ok=True)
            _EXT = load(name="csf_mailbox", sources=[str(Path(__file__).parent / "csrc" / "mailbox.cu")],
                        extra_cuda_cflags=["-O3"], build_directory=str(build), verbose=False)
        return _EXT


def env_int(name, default):
    return int(os.environ.get(name, default))


def pinned_exact(rows, stride):
    """Page-aligned host buffer pinned at exactly rows*stride bytes (see host_register)."""
    import mmap
    nbytes = rows * stride
    if not nbytes:
        return None
    buf = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    t = torch.frombuffer(buf, dtype=torch.uint8).reshape(rows, stride)
    mailbox_ext().host_register(t.data_ptr(), nbytes)
    t._csf_mmap = buf  # keep the mapping alive with the tensor
    return t


class Pack:
    """O_DIRECT access to <dir>/rank<r>/layer<L>.bin fixed-stride records."""

    def __init__(self, rank):
        dirs = os.environ["CSF_PACK_DIRS"].split(":")
        self.dir = Path(dirs[rank % len(dirs)]) / f"rank{rank}"
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.stride = self.meta["stride"]
        self.fds = {}

    def fd(self, layer):
        if layer not in self.fds:
            self.fds[layer] = os.open(self.dir / f"layer{layer}.bin", os.O_RDONLY | os.O_DIRECT)
        return self.fds[layer]

    def read(self, layer, expert, buffer):
        view = memoryview(buffer.numpy())
        n = os.preadv(self.fd(layer), [view], expert * self.stride)
        if n != self.stride:
            raise IOError(f"short pack read layer {layer} expert {expert}: {n}")


class Shared:
    """Per-process state: pack, RAM tier, staging buffers and reader threads."""

    def __init__(self, rank, device):
        self.rank, self.device = rank, device
        self.pack = Pack(rank)
        stride = self.pack.stride
        placement = os.environ.get("CSF_PLACEMENT")
        self.placement = json.loads(Path(placement).read_text()) if placement else {}
        ram = [(int(l), e) for l, es in self.placement.get("ram", {}).items() for e in es]
        ram = ram[:env_int("CSF_RAM_EXPERTS", len(ram))]
        self.ram_index = {key: i for i, key in enumerate(ram)}
        self.ram = pinned_exact(len(ram), stride) if ram else None
        start = time.perf_counter()
        with ThreadPoolExecutor(16) as pool:
            list(pool.map(lambda kv: self.pack.read(kv[0][0], kv[0][1], self.ram[kv[1]]),
                          self.ram_index.items()))
        if ram:
            print(f"[csf-v2] rank{rank} RAM tier {len(ram)} experts, "
                  f"{len(ram) * stride / 2**30:.1f} GiB in {time.perf_counter() - start:.1f}s", flush=True)
        # Demand misses (the GPU is waiting) and speculative prefetches use
        # separate readers, staging buffers and RAM-copy streams, so demand
        # work never queues behind prefetch work; prefetch reads in flight
        # are capped to bound their share of NVMe and PCIe bandwidth.
        def staging(count):
            q = queue.Queue()
            for _ in range(count):
                q.put(torch.empty(stride, dtype=torch.uint8).pin_memory())
            return q

        self.staging = {True: staging(env_int("CSF_STAGING_BUFFERS", 32)),
                        False: staging(env_int("CSF_PREFETCH_STAGING", 8))}
        counts = {True: env_int("CSF_READERS", 12), False: env_int("CSF_PREFETCH_READERS", 4)}
        self.readers = {d: ThreadPoolExecutor(n, thread_name_prefix=f"csf-pack-{'d' if d else 'p'}")
                        for d, n in counts.items()}
        self.prefetch_inflight_max = env_int("CSF_PREFETCH_INFLIGHT", 8)
        self.prefetch_inflight = 0
        self.local = threading.local()
        self.ram_stream = self.private_stream()
        self.ram_streams = {True: self.ram_stream, False: self.private_stream()}
        # Creating a CUDA stream waits for running kernels, and the mailbox
        # wait kernel spins until a load lands; create every reader thread's
        # stream (and exercise a copy) now, before any kernel can spin.
        for demand, n in counts.items():
            barrier = threading.Barrier(n)

            def warm(demand=demand, barrier=barrier):
                stream = self.reader_stream()
                barrier.wait()
                probe = self.staging[demand].get()
                with torch.cuda.stream(stream):
                    probe[:4096].to(device, non_blocking=True)
                stream.synchronize()
                self.staging[demand].put(probe)

            for future in [self.readers[demand].submit(warm) for _ in range(n)]:
                future.result()
        self.neg_one = torch.full((1,), -1, dtype=torch.int32).pin_memory()
        self.graph = os.environ.get("CSF_GRAPH") == "1"
        if self.graph:
            import faulthandler
            import signal
            import sys
            faulthandler.register(signal.SIGUSR1, all_threads=True)  # kill -USR1 <pid> dumps stacks
            sys.setswitchinterval(float(os.environ.get("CSF_SWITCH_INTERVAL", "0.0002")))
            self.box = torch.zeros((MAX_LAYERS, BOX_WORDS), dtype=torch.int64).pin_memory()
            self.box_np = self.box.numpy()
            self.box_dev = mailbox_ext().device_pointer(self.box.data_ptr())
            if V4:
                # v4: a C++ thread serves the mailbox; the GPU copies the data.
                ext = v4_ext()
                self.ack = torch.zeros((MAX_LAYERS, ext.ACK_WORDS), dtype=torch.int64).pin_memory()
                self.ack_dev = mailbox_ext().device_pointer(self.ack.data_ptr())
                n = env_int("CSF_HOST_CACHE", 128)
                self.host_cache = pinned_exact(n, stride)
                ext.init(self.host_cache.data_ptr(), n, stride, env_int("CSF_V4_READERS", 12),
                         env_int("CSF_PREFETCH_INFLIGHT", 8))
                self.v4_started = False
                threading.Thread(target=self.v4_report, name=f"csf-v4-report{rank}", daemon=True).start()
            else:
                self.service = threading.Thread(target=self.serve, name=f"csf-service{rank}", daemon=True)
                self.service.start()
            if DEBUG:
                threading.Thread(target=self.watchdog, name=f"csf-watchdog{rank}", daemon=True).start()

    def v4_report(self):
        directory = os.environ.get("CSF_STREAM_METRICS_DIR")
        if not directory:
            return
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        while True:
            time.sleep(10)
            if not self.v4_started:
                continue
            data = {l: dict(v4_ext().stats(l)) for (r, l) in list(_CACHES) if r == self.rank}
            tmp = path / f"v4-rank{self.rank}.tmp"
            tmp.write_text(json.dumps(data) + "\n")
            tmp.replace(path / f"v4-rank{self.rank}.json")

    def watchdog(self):
        """Debug: report mailbox misses that stay unacknowledged."""
        seen = {}
        while True:
            time.sleep(1)
            box = self.box_np
            for layer in range(MAX_LAYERS):
                seq, ack, miss = int(box[layer, 0]), int(box[layer, 1]), int(box[layer, 3])
                if miss and ack < seq:
                    first = seen.setdefault((layer, seq), time.time())
                    if time.time() - first > 5:
                        cache = _CACHES.get((self.rank, layer))
                        n = int(box[layer, 2])
                        ids = box[layer, BOX_HEADER:].view("int32")[:n].tolist()
                        info = {}
                        if cache is not None:
                            info = dict(pending={sl: type(v).__name__ + ("(done)" if (v.done() if hasattr(v, "done") else v.query()) else "")
                                                 for sl, v in list(cache.pending.items())[:16]},
                                        free=len(cache.free), lru=len(cache.lru),
                                        missing=[e for e in ids if e >= 0 and e not in cache.slot_of],
                                        ram_stream_idle=self.ram_stream.query())
                        busy = {l: [(sl, type(v).__name__) for sl, v in c.pending.items()
                                    if not (v.done() and v.result().query() if isinstance(v, Future) else v.query())]
                                for (r, l), c in list(_CACHES.items()) if r == self.rank}
                        busy = {l: v for l, v in busy.items() if v}
                        print(f"[csf-watchdog] rank{self.rank} layer {layer} seq {seq} ack {ack} ids {ids[:24]} "
                              f"prefetch_inflight {self.prefetch_inflight} {info} service_current "
                              f"{getattr(self, 'current', None)} incomplete_loads {busy} "
                              f"streams_idle ram={self.ram_streams[True].query()} pre={self.ram_streams[False].query()}",
                              flush=True)
                        seen[(layer, seq)] = time.time() + 30

    def serve(self):
        """Service GPU mailbox requests: record usage, load misses, prefetch."""
        import numpy as np
        torch.cuda.set_device(self.device)
        box, last, idle = self.box_np, np.zeros(MAX_LAYERS, dtype=np.int64), 0
        while True:
            changed = np.nonzero(box[:, 0] != last)[0]
            if not len(changed):
                idle += 1
                time.sleep(0 if idle < 2000 else 0.0002)
                continue
            idle = 0
            for layer in changed:
                seq = int(box[layer, 0])
                last[layer] = seq
                cache = _CACHES.get((self.rank, int(layer)))
                if cache is None:
                    continue
                n, miss, npred = int(box[layer, 2]), int(box[layer, 3]), int(box[layer, 4])
                words = box[layer, BOX_HEADER:].view(np.int32)
                ids = words[:n].tolist()
                pred = words[n:n + npred].tolist()
                try:
                    if DEBUG:
                        import faulthandler
                        faulthandler.dump_traceback_later(3, exit=False)
                    cache.handle(seq, ids, pred, miss)
                    if DEBUG:
                        faulthandler.cancel_dump_traceback_later()
                except BaseException:
                    import traceback
                    traceback.print_exc()
                    os._exit(79)
            if box[:, 5].any():
                print(f"[csf-v2] rank{self.rank} GPU mailbox wait timed out: "
                      f"{np.nonzero(box[:, 5])[0].tolist()} seq {box[:, 0].tolist()[:41]} "
                      f"ack {box[:, 1].tolist()[:41]} handled {last.tolist()[:41]}", flush=True)
                import faulthandler
                import sys
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                os._exit(80)

    def private_stream(self):
        """A stream no other component can be handed (see create_stream)."""
        with torch.cuda.device(self.device):
            return torch.cuda.ExternalStream(mailbox_ext().create_stream(), device=self.device)

    def reader_stream(self):
        if not hasattr(self.local, "stream"):
            torch.cuda.set_device(self.device)
            self.local.stream = self.private_stream()
        return self.local.stream


def shared(rank, device):
    with _LOCK:
        if rank not in _SHARED:
            _SHARED[rank] = Shared(rank, device)
        return _SHARED[rank]


class FastExpertCache:
    def __init__(self, root, layer, hidden, intermediate, num_experts, rank, tp,
                 device, capacity=192, nonlinearity="silu", limit=None):
        self.root, self.layer, self.hidden = root, layer, hidden
        self.intermediate, self.rank, self.tp = intermediate, rank, tp
        self.device = torch.device(device)
        self.num_experts, self.capacity = num_experts, capacity
        self.nonlinearity, self.limit = nonlinearity, limit
        self.stats = Counter()
        self.metrics_interval = max(1, env_int("CSF_STREAM_METRICS_INTERVAL", 50))
        self.fake_hits = os.environ.get("CSF_STREAM_FAKE_HITS") == "1"
        self.prefetch_width = 0 if self.fake_hits else env_int("CSF_PREFETCH_WIDTH", 6)
        self.prefetch_tokens = env_int("CSF_PREFETCH_MAX_TOKENS", 4)
        self.prefill_prefetch = env_int("CSF_PREFILL_PREFETCH", 1) == 1
        self.prefill_width = env_int("CSF_PREFILL_WIDTH", 6)
        self.prefill_min_votes = env_int("CSF_PREFILL_MIN_VOTES", 1)
        self.plans = {}
        self.shared = shared(rank, self.device)
        # Split pack: slot s holds the half-width experts 2s and 2s+1 (the TP4
        # slices of this rank), which the deterministic kernel runs fast.
        self.split = self.shared.pack.meta.get("split", 1)
        with torch.cuda.device(self.device):
            weights, _, _ = read_experts(root, layer, [0], hidden, intermediate,
                                         rank * self.split, tp * self.split)
            template = upload_and_prepare(weights, hidden, intermediate // (tp * self.split),
                                          self.device, nonlinearity, limit)
            self.prepared = allocate_slots(template, capacity * self.split)
            torch.cuda.synchronize(self.device)
        meta = self.shared.pack.meta["fields"]
        if [f["name"] for f in meta] != list(FIELDS):
            raise ValueError("expert pack fields do not match the B12X slot layout")
        # Flat byte views of every slot row, matched to record offsets.
        self.targets = []
        for f in meta:
            t = getattr(self.prepared._impl, f["name"])
            rows = t.view(torch.uint8).reshape(capacity, -1)
            if rows.shape[1] != f["nbytes"]:
                raise ValueError(f"pack field {f['name']} size mismatch")
            self.targets.append((rows, f["offset"], f["nbytes"]))
        self.lock = threading.RLock()
        self.graph = self.shared.graph
        self.table = None
        if self.graph:
            # Device expert -> slot table read by the mailbox kernels; host
            # cells hold the value each table write copies (one per expert).
            self.table = torch.full((num_experts,), -1, dtype=torch.int32, device=self.device)
            self.cells = torch.full((num_experts,), -1, dtype=torch.int32).pin_memory()
            self.dev_state = torch.zeros(2, dtype=torch.int64, device=self.device)
            self.box_dev = self.shared.box_dev + layer * BOX_WORDS * 8
            self.target_free = env_int("CSF_FREE_SLOTS", 8)
            self.timeout_ns = env_int("CSF_WAIT_TIMEOUT_S", 60) * 10**9
            self.empty_pred = torch.empty(0, dtype=torch.int64, device=self.device)
        self.slot_of = {}                       # expert -> slot
        self.lru = OrderedDict()                # evictable experts
        self.pending = {}                       # slot -> Event or Future[Event]
        self.prefetched = set()                 # prefetched, not yet used
        self.free = list(reversed(range(capacity)))
        self.compute_done = None
        pinned = self.shared.placement.get("vram", {}).get(str(layer), [])[:capacity]
        for expert in pinned:
            self.load(expert, self.free.pop())
        self.pinned = set(pinned)
        self.resolve(list(self.pending))
        self.gate = None
        if self.graph:
            self.router()               # no host-to-device loads during graph capture
            torch.cuda.synchronize(self.device)
        if V4:
            self.register_v4()
        if self.split > 1:
            # One deterministic plan per rank serves every batch size; compile it
            # now, since kernel resolution is frozen once graph capture begins.
            self.plans = self.shared.__dict__.setdefault("split_plans", {})
            self.halves = torch.arange(self.split, dtype=torch.int32, device=self.device)
            if not self.pinned:
                raise ValueError("split packs need pinned experts: masked routes use a pinned slot")
            # The half-width plan intermittently corrupts outputs on -1 routes, so
            # masked routes point at an always-resident slot with a zero gate.
            # The lowest-id pinned expert is registered first and is never lent (v4.cu).
            self.safe_slot = self.slot_of[min(self.pinned)]
            self.plan_for(SPLIT_MAX_TOKENS, 6 * self.split, self.device)
            self.plan_for(SPLIT_MAX_TOKENS, self.split, self.device)    # prefill pairs (run_pairs)
            torch.cuda.synchronize(self.device)
        _CACHES[(rank, layer)] = self
        self.stats["cache_bytes"] = sum(r.numel() for r, _, _ in self.targets)
        atexit.register(self.report)

    def register_v4(self):
        sh, ext = self.shared, v4_ext()
        self.fill_dst = [rows.data_ptr() for rows, _, _ in self.targets]
        self.fill_bytes = [rows.shape[1] for rows, _, _ in self.targets]
        self.fill_offsets = [offset for _, offset, _ in self.targets]
        self.fill_blocks = env_int("CSF_FILL_BLOCKS", 24)
        self.ack_dev = sh.ack_dev + self.layer * sh.ack.shape[1] * 8
        ram = [0] * self.num_experts
        if sh.ram is not None:
            for (l, e), i in sh.ram_index.items():
                if l == self.layer:
                    ram[e] = sh.ram.data_ptr() + i * sh.pack.stride
        resident = sorted(self.slot_of.items())
        ext.register_layer(self.layer, sh.box.data_ptr() + self.layer * BOX_WORDS * 8,
                           sh.ack.data_ptr() + self.layer * sh.ack.shape[1] * 8,
                           self.num_experts, self.capacity, [e for e, _ in resident],
                           [s for _, s in resident], sorted(self.pinned), list(self.free),
                           str(sh.pack.dir / f"layer{self.layer}.bin"), ram)

    def v4_resident(self):
        flat = v4_ext().resident(self.layer)
        return dict(zip(flat[::2], flat[1::2]))

    # Loading -------------------------------------------------------------
    def copy_record(self, src, slot, stream, expert):
        with torch.cuda.stream(stream):
            if self.compute_done is not None:
                stream.wait_event(self.compute_done)
            for rows, offset, nbytes in self.targets:
                rows[slot].copy_(src[offset:offset + nbytes], non_blocking=True)
            if self.table is not None:
                # Publish the slot only after its bytes, on the same stream.
                self.cells[expert] = slot
                self.table[expert:expert + 1].copy_(self.cells[expert:expert + 1], non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        return event

    def load(self, expert, slot, demand=True):
        self.slot_of[expert] = slot
        index = self.shared.ram_index.get((self.layer, expert))
        if index is not None:
            self.stats["loads_ram"] += 1
            self.pending[slot] = self.copy_record(self.shared.ram[index], slot,
                                                  self.shared.ram_streams[demand], expert)
            return
        self.stats["loads_ssd"] += 1
        future = Future()
        self.pending[slot] = future
        if not demand:
            with _LOCK:
                self.shared.prefetch_inflight += 1
        self.shared.readers[demand].submit(self._read_and_copy, expert, slot, future, demand)

    def _read_and_copy(self, expert, slot, future, demand=True):
        try:
            self._read_and_copy_inner(expert, slot, future, demand)
        finally:
            if not demand:
                with _LOCK:
                    self.shared.prefetch_inflight -= 1

    def _read_and_copy_inner(self, expert, slot, future, demand):
        buffer = self.shared.staging[demand].get()
        try:
            self.shared.pack.read(self.layer, expert, buffer)
            event = self.copy_record(buffer, slot, self.shared.reader_stream(), expert)
            future.set_result(event)
            event.synchronize()
        except BaseException as error:
            if not future.done():
                future.set_exception(error)
        finally:
            self.shared.staging[demand].put(buffer)

    def resolve(self, slots):
        """Make the current stream wait for any pending loads of these slots."""
        stream = torch.cuda.current_stream(self.device)
        for slot in slots:
            item = self.pending.pop(slot, None)
            if item is None:
                continue
            if isinstance(item, Future):
                start = time.perf_counter()
                item = item.result()
                self.stats["host_wait_seconds"] += time.perf_counter() - start
            stream.wait_event(item)

    def settled(self, slot):
        """True when no load into this slot is still in flight."""
        item = self.pending.get(slot)
        if item is None:
            return True
        if isinstance(item, Future):
            if not item.done():
                return False
            item = item.result()
        if not item.query():
            return False
        del self.pending[slot]
        return True

    def reserve(self, expert, protected, wait=True):
        if self.free:
            return self.free.pop()
        for victim in self.lru:
            if victim not in protected and self.settled(self.slot_of[victim]):
                break
        else:
            # Every evictable slot is still loading. Demand misses wait for
            # the least recently used one; prefetches give up.
            victim = next((v for v in self.lru if v not in protected), None)
            if victim is None or not wait:
                return None
            item = self.pending.pop(self.slot_of[victim])
            start = time.perf_counter()
            (item.result() if isinstance(item, Future) else item).synchronize()
            self.stats["host_wait_seconds"] += time.perf_counter() - start
            self.stats["evict_waits"] += 1
        return self.evict(victim)

    def evict(self, victim):
        del self.lru[victim]
        self.prefetched.discard(victim)
        if self.table is not None:
            with torch.cuda.stream(self.shared.ram_stream):
                self.table[victim:victim + 1].copy_(self.shared.neg_one, non_blocking=True)
        return self.slot_of.pop(victim)

    def ensure(self, requested):
        if self.fake_hits:
            # Timing-only: every expert "hits" an arbitrary slot; outputs are wrong.
            self.stats["hits"] += len(requested)
            return {e: e % self.capacity for e in requested}
        protected = set(requested)
        for expert in requested:
            if expert in self.slot_of:
                self.stats["hits"] += 1
                if expert in self.lru:
                    self.lru.move_to_end(expert)
                if expert in self.prefetched:
                    self.prefetched.discard(expert)
                    self.stats["prefetch_used"] += 1
                continue
            self.stats["misses"] += 1
            slot = self.reserve(expert, protected)
            if slot is None:
                raise ValueError("routing chunk exceeds expert cache capacity")
            self.load(expert, slot)
            self.lru[expert] = None
        mapping = {e: self.slot_of[e] for e in requested}
        return mapping

    def prefetch(self, experts):
        for expert in experts:
            if expert in self.slot_of:
                continue
            if self.graph and not self.free:
                return  # the GPU may be reading this layer; only fill free slots
            ssd = (self.layer, expert) not in self.shared.ram_index
            if ssd and self.shared.prefetch_inflight >= self.shared.prefetch_inflight_max:
                self.stats["prefetch_skipped"] += 1
                continue
            slot = self.reserve(expert, (), wait=False)
            if slot is None:
                return
            self.stats["prefetches"] += 1
            self.prefetched.add(expert)
            self.load(expert, slot, demand=False)
            self.lru[expert] = None
            self.lru.move_to_end(expert, last=False)  # evict first if unused

    # Graph mode: GPU mailbox requests ------------------------------------
    def handle(self, seq, ids, pred, miss):
        """Service thread: one publish() of this layer. When miss is set the
        GPU is blocked in wait() for this layer, so no kernel can be reading
        its slots and eviction is safe; acknowledge once loads have landed.
        seq=None is the eager path: the caller already synchronized the
        device, so nothing can be reading the slots either, and no ack."""
        requested = list(dict.fromkeys(e for e in ids if e >= 0))
        if seq is not None:
            trace.record_ids(self.layer, self.rank, ids)
        with self.lock:
            self.stats["calls"] += 1
            for expert in requested:
                if expert in self.lru:
                    self.lru.move_to_end(expert)
                if expert in self.prefetched:
                    self.prefetched.discard(expert)
                    self.stats["prefetch_used"] += 1
            if miss:
                start = time.perf_counter()
                protected = set(requested)
                for expert in requested:
                    if expert in self.slot_of:
                        continue
                    self.stats["misses"] += 1
                    slot = self.reserve(expert, protected)
                    if slot is None:
                        raise ValueError(f"layer {self.layer}: request exceeds cache capacity")
                    self.load(expert, slot)
                    self.lru[expert] = None
                while len(self.free) < self.target_free:
                    victim = next((v for v in self.lru if v not in protected
                                   and self.settled(self.slot_of[v])), None)
                    if victim is None:
                        break
                    self.free.append(self.evict(victim))
                for slot in [self.slot_of[e] for e in requested]:
                    item = self.pending.pop(slot, None)
                    if item is not None:
                        self.shared.current = (self.layer, seq, slot, type(item).__name__, time.time())
                        (item.result() if isinstance(item, Future) else item).synchronize()
                self.shared.current = (self.layer, seq, "ram_stream", None, time.time())
                self.shared.ram_stream.synchronize()   # evictions' table writes
                if seq is not None:
                    self.shared.box_np[self.layer, 1] = seq
                self.stats["miss_waits"] += 1
                self.stats["miss_service_seconds"] += time.perf_counter() - start
            else:
                self.stats["hits"] += len(requested)
        if self.stats["calls"] % self.metrics_interval == 0:
            self.report()
            trace.flush()
        nxt = _CACHES.get((self.rank, self.layer + 1))
        if pred and nxt is not None:
            with nxt.lock:
                nxt.prefetch(list(dict.fromkeys(pred)))

    def graph_slots(self, topk_ids, x=None, ahead=None):
        """Graph-capturable expert -> slot translation with host miss service.

        `ahead` (eager v4 only): experts of this layer needed next, read into
        host buffers while the current request is computed."""
        ext = mailbox_ext()
        ids = topk_ids.reshape(-1).to(torch.int64)
        pred = self.empty_pred
        nxt = _CACHES.get((self.rank, self.layer + 1))
        capturing = torch.cuda.is_current_stream_capturing()
        # Prefetch hints only from graph replays: eager calls stay fully
        # synchronous so no helper thread issues CUDA calls when vLLM begins
        # a (globally scoped) graph capture right after its warmup pass.
        if ((capturing or V4) and x is not None and nxt is not None and self.prefetch_width
                and x.shape[0] <= self.prefetch_tokens):
            weight, bias = nxt.router()
            score = torch.sqrt(torch.nn.functional.softplus(x.float() @ weight.T)) + bias
            pred = score.topk(self.prefetch_width, dim=-1).indices.reshape(-1)
        elif (V4 and not capturing and x is not None and nxt is not None and self.prefill_prefetch
              and x.shape[0] > self.prefetch_tokens):
            # Prefill: read ahead the union of the next layer's predicted
            # experts over all chunk tokens (eager only; variable length).
            weight, bias = nxt.router()
            score = torch.sqrt(torch.nn.functional.softplus(x.float() @ weight.T)) + bias
            top = score.topk(self.prefill_width, dim=-1).indices.reshape(-1)
            votes = torch.bincount(top, minlength=self.num_experts)
            pred = (votes >= self.prefill_min_votes).nonzero().reshape(-1)
        if V4 and ahead and not capturing:
            hints = [e + SAME_LAYER_HINT for e in ahead if e not in self.pinned]
            pred = torch.cat([pred, torch.tensor(hints, dtype=pred.dtype, device=pred.device)])
        if ids.numel() + pred.numel() > BOX_IDS:
            raise ValueError("too many routed ids for one mailbox call")
        if V4:
            sh = self.shared
            if not sh.v4_started:
                with _LOCK:
                    if not sh.v4_started:
                        ext = v4_ext()
                        route_trace = os.environ.get("CSF_V4_TRACE")
                        if route_trace and self.rank == 0:
                            Path(route_trace).parent.mkdir(parents=True, exist_ok=True)
                            ext.trace_to(route_trace)
                        ext.start()
                        sh.v4_started = True
            slots = torch.empty_like(ids)
            ext = v4_ext()
            mailbox_ext().publish(ids, pred, self.table, slots, self.dev_state, self.box_dev, BOX_IDS)
            ext.wait_fill(self.dev_state, self.box_dev, self.ack_dev, self.fill_dst, self.fill_bytes,
                          self.fill_offsets, self.timeout_ns, self.fill_blocks)
            ext.finalize(ids, self.table, slots, self.dev_state, self.box_dev, self.ack_dev)
            return slots.view(topk_ids.shape).to(topk_ids.dtype)
        if not capturing:
            # Eager: serve misses here, synchronously, so wait() never spins.
            # A spinning kernel plus any GIL-holding blocking CUDA call on
            # this thread (first kernel launches, module loads) would
            # deadlock the Python service thread.
            torch.cuda.current_stream(self.device).synchronize()
            self.handle(None, ids.tolist(), pred.tolist(), True)
        slots = torch.empty_like(ids)
        ext.publish(ids, pred, self.table, slots, self.dev_state, self.box_dev, BOX_IDS)
        ext.wait(ids, self.table, slots, self.dev_state, self.box_dev, self.timeout_ns)
        return slots.view(topk_ids.shape).to(topk_ids.dtype)

    def host_slots(self, topk_ids):
        """No-spin path (breakable graphs, v3): the caller's stream has been
        synchronized by the host read of the routes, so no kernel is using
        this layer's slots. Serve misses here, then translate on the device."""
        start = time.perf_counter()
        ids = topk_ids.reshape(-1).tolist()                  # waits for the router
        self.stats["host_route_seconds"] += time.perf_counter() - start
        self.handle(None, ids, [], True)
        table = self.table[topk_ids.clamp(min=0).long()]
        self.stats["host_slots_seconds"] += time.perf_counter() - start
        return torch.where(topk_ids < 0, topk_ids, table.to(topk_ids.dtype))

    def expert_groups(self, topk_ids):
        """Expert-major prefill: this call's distinct experts in batches that fit
        the evictable slots, already-resident ones first so later batches cannot
        evict them before use. Pinned experts ride along in the first batch."""
        needed = torch.unique(topk_ids[topk_ids >= 0]).long()
        resident = (self.table[needed] >= 0).tolist()
        needed = needed.tolist()
        # A pinned expert can be lent to a many-stream decode step (v4 soft
        # pinning); only resident pinned experts ride along for free.
        pinned = [e for e, r in zip(needed, resident) if r and e in self.pinned]
        rest = [e for e, r in zip(needed, resident) if r and e not in self.pinned]
        rest += [e for e, r in zip(needed, resident) if not r]
        size = self.capacity - len(self.pinned) - 1
        groups = [rest[i:i + size] for i in range(0, len(rest), size)] or [[]]
        groups[0] = pinned + groups[0]
        return groups

    def slots_for(self, topk_ids, mask):
        """Device translation of the masked routes after a batch is resident."""
        table = self.table[topk_ids.clamp(min=0).long()].to(topk_ids.dtype)
        return torch.where(mask, table, torch.full_like(topk_ids, -1))

    def eager_chunks(self, topk_ids):
        """Token ranges whose non-pinned experts fit this layer's evictable slots."""
        rows = topk_ids.tolist()
        counted = [[-1 if e in self.pinned else e for e in row] for row in rows]
        # One mailbox call per chunk: routed ids plus up to one prediction per expert.
        limit = (BOX_IDS - self.num_experts) // max(1, topk_ids.shape[1])
        chunks = []
        for lo, hi in routing_chunks(counted, self.capacity - len(self.pinned) - 1):
            chunks.extend((a, min(a + limit, hi)) for a in range(lo, hi, limit))
        return chunks

    # Next-layer prediction ----------------------------------------------
    def router(self):
        if self.gate is None:
            from safetensors import safe_open
            names = json.loads((Path(self.root) / "build-contract.json").read_text())["source_names"]
            tensors = []
            for suffix in ("weight", "bias"):
                name = f"layers.{self.layer}.ffn.gate.{suffix}"
                with safe_open(str(Path(self.root) / "tensors" / names[name]), "pt") as f:
                    tensors.append(f.get_tensor(name).float().to(self.device))
            self.gate = tensors
        return self.gate

    # Execution -----------------------------------------------------------
    def report(self):
        directory = os.environ.get("CSF_STREAM_METRICS_DIR")
        if directory:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            target = path / f"rank{self.rank}-layer{self.layer}.json"
            tmp = target.with_suffix(".tmp")
            tmp.write_text(json.dumps(dict(layer=self.layer, rank=self.rank, **self.stats), indent=2) + "\n")
            tmp.replace(target)

    def plan_for(self, count, topk, device):
        from b12x.moe import fused_moe as fm
        from b12x.moe.fused_moe._tuning import MoeDecodeConfig
        key = count, topk
        if key not in self.plans:
            # "auto" lets B12X pick its decode kernel (about 17x faster for one
            # token than the pinned deterministic M16 kernel used by the v1 tests).
            # The auto small-batch kernel is not reproducible at the full TP2
            # width (2-3% of max between identical calls), which corrupts greedy
            # output; split packs always use the deterministic kernel.
            pinned = os.environ.get("CSF_MOE_KERNEL", "auto") == "m16" or self.split > 1
            extra = dict(deterministic_output=True, decode_config=MoeDecodeConfig(
                backend="dynamic", route_planner="internal", max_active_clusters=None,
                dynamic_tile_m=16, dynamic_route_mode="grouped")) if pinned else {}
            plan = fm.plan(fm.Caps(
                max_tokens=count, num_topk=topk, device=device,
                weight_plan=self.prepared._impl.plan, quant_mode="w4a8_mx",
                swiglu_limit=self.limit, **extra))
            scratch = torch.empty(sum(s.nbytes for s in plan.scratch_specs()), dtype=torch.uint8, device=device)
            self.plans[key] = plan, scratch
        return self.plans[key]

    def run_slots(self, x, topk_weights, slots):
        """Split pack: run slot routes as half-expert routes sharing each gate."""
        from b12x.moe import fused_moe as fm
        if x.shape[0] > SPLIT_MAX_TOKENS:
            raise ValueError(f"{x.shape[0]} tokens exceed CSF_SPLIT_MAX_TOKENS={SPLIT_MAX_TOKENS}")
        masked = slots < 0
        pairs = torch.where(masked, self.safe_slot, slots).unsqueeze(-1).int()
        ids = (pairs * self.split + self.halves).flatten(1)
        weights = torch.where(masked, 0.0, topk_weights.float()).repeat_interleave(self.split, -1)
        plan, scratch = self.plan_for(SPLIT_MAX_TOKENS, ids.shape[1], x.device)
        out = torch.empty_like(x)
        fm.run(binding=fm.bind(plan, scratch=scratch, a=x, experts=self.prepared._impl,
                               topk_weights=weights, topk_ids=ids, output=out, input_scales_static=True))
        return out

    def run_pairs(self, x, tok, weights, slots, acc):
        """Split-pack prefill: one route per row, accumulated into fp32 `acc`."""
        for lo in range(0, tok.numel(), SPLIT_MAX_TOKENS):
            t = tok[lo:lo + SPLIT_MAX_TOKENS]
            y = self.run_slots(x.index_select(0, t), weights[lo:lo + SPLIT_MAX_TOKENS, None],
                               slots[lo:lo + SPLIT_MAX_TOKENS, None])
            acc.index_add_(0, t, y.float())

    def device_slots(self, topk_ids):
        """Graph-capturable expert -> slot translation (timing-only for now)."""
        if not self.fake_hits:
            raise RuntimeError("device slot mapping is only implemented for fake hits")
        return torch.where(topk_ids < 0, topk_ids, topk_ids % self.capacity)

    def run_fake(self, x, topk_weights, topk_ids):
        """Timing-only, graph-capturable: experts map to slots on the device."""
        from b12x.moe import fused_moe as fm
        ids = torch.where(topk_ids < 0, topk_ids, topk_ids % self.capacity)
        plan, scratch = self.plan_for(x.shape[0], ids.shape[1], x.device)
        output = torch.empty_like(x)
        fm.run(binding=fm.bind(plan, scratch=scratch, a=x, experts=self.prepared._impl,
                               topk_weights=topk_weights.float().contiguous(), topk_ids=ids,
                               output=output, input_scales_static=True))
        return output

    def run(self, x, topk_weights, topk_ids):
        from b12x.moe import fused_moe as fm
        if x.shape[0] == 0:
            return torch.empty_like(x)
        if self.fake_hits:
            return self.run_fake(x, topk_weights, topk_ids)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CSF streaming requires eager execution")
        start = time.perf_counter()
        trace.record(self.layer, self.rank, x, topk_ids)
        nxt = _CACHES.get((self.rank, self.layer + 1))
        predicted = None
        if nxt is not None and self.prefetch_width and x.shape[0] <= self.prefetch_tokens:
            weight, bias = nxt.router()
            score = torch.sqrt(torch.nn.functional.softplus(x.float() @ weight.T)) + bias
            predicted = score.topk(self.prefetch_width, dim=-1).indices
            ids_cpu = torch.cat([topk_ids.long(), predicted], dim=1).cpu()
            rows = ids_cpu[:, :topk_ids.shape[1]].tolist()
            predicted = list(dict.fromkeys(ids_cpu[:, topk_ids.shape[1]:].T.reshape(-1).tolist()))
        else:
            rows = topk_ids.detach().cpu().tolist()
        output = torch.empty_like(x)
        prefetched = False
        for lo, hi in routing_chunks(rows, self.capacity - len(self.pinned)):
            requested = list(dict.fromkeys(e for row in rows[lo:hi] for e in row if e != -1))
            if not requested:
                output[lo:hi].zero_()
                continue
            mapping = self.ensure(requested)
            if predicted is not None and not prefetched:
                nxt.prefetch(predicted)
                prefetched = True
            self.resolve([mapping[e] for e in requested])
            ids = torch.tensor([[-1 if e == -1 else mapping[e] for e in row] for row in rows[lo:hi]],
                               dtype=topk_ids.dtype).to(x.device, non_blocking=True)
            plan, scratch = self.plan_for(hi - lo, ids.shape[1], x.device)
            binding = fm.bind(plan, scratch=scratch, a=x[lo:hi], experts=self.prepared._impl,
                              topk_weights=topk_weights[lo:hi].float().contiguous(),
                              topk_ids=ids, output=output[lo:hi], input_scales_static=True)
            fm.run(binding=binding)
            self.compute_done = torch.cuda.Event()
            self.compute_done.record(torch.cuda.current_stream(x.device))
            self.stats["routing_chunks"] += 1
        self.stats["calls"] += 1
        self.stats["tokens"] += x.shape[0]
        self.stats["host_call_seconds"] += time.perf_counter() - start
        if self.stats["calls"] % self.metrics_interval == 0:
            self.report()
            trace.flush()
        return output
