"""O_DIRECT read latency of expert pack records: one read per record vs N parallel chunks.

python3 scripts/read_bench.py <pack dir with layer*.bin> [--records 300] [--chunks 1 2 4 8] [--concurrent 1 2]

Reads random (layer, expert) records, each RECORD bytes (4 KiB aligned), into page-aligned
buffers. For chunks > 1 the record is split into equal aligned pieces read by a thread pool at
the same time, as a demand miss would be. `--concurrent` records are in flight together
(the number of NVMe misses in one layer request).
"""
import argparse, mmap, os, random, statistics, time
from concurrent.futures import ThreadPoolExecutor

RECORD = 9404416
PAGE = 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--records", type=int, default=300)
    ap.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--concurrent", type=int, nargs="+", default=[1, 2, 4])
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(a.pack) if f.startswith("layer") and f.endswith(".bin"))
    fds = [os.open(os.path.join(a.pack, f), os.O_RDONLY | os.O_DIRECT) for f in files]
    nrec = os.fstat(fds[0]).st_size // RECORD
    pool = ThreadPoolExecutor(32)
    bufs = [mmap.mmap(-1, RECORD) for _ in range(8)]
    rng = random.Random(1)

    def read(fd, off, buf, start, size):
        view = memoryview(buf)[start:start + size]
        got = os.preadv(fd, [view], off + start)
        assert got == size, got

    for conc in a.concurrent:
        for n in a.chunks:
            assert (RECORD // PAGE) % n == 0
            piece = RECORD // n
            lat = []
            t_all = time.perf_counter()
            for _ in range(a.records // conc):
                jobs = []
                t0 = time.perf_counter()
                for c in range(conc):
                    fd = rng.choice(fds); off = rng.randrange(nrec) * RECORD
                    for k in range(n):
                        jobs.append(pool.submit(read, fd, off, bufs[c], k * piece, piece))
                for j in jobs:
                    j.result()
                lat.append((time.perf_counter() - t0) * 1e3)
            wall = time.perf_counter() - t_all
            gbps = len(lat) * conc * RECORD / wall / 1e9
            lat.sort()
            print(f"concurrent {conc} x chunks {n}: request latency median {statistics.median(lat):.2f} ms  "
                  f"p90 {lat[int(0.9 * len(lat))]:.2f} ms  ({gbps:.1f} GB/s)", flush=True)


if __name__ == "__main__":
    main()
