// Engine v4: GPU-driven expert fills with a GIL-free, CUDA-free host service.
//
// Per routed layer call (inside or outside a CUDA graph):
//   publish   (mailbox.cu) maps ids -> slots, posts the request, arms a wait on a miss.
//   wait_fill spins until the host acknowledges, then copies every acknowledged
//             expert record from pinned host memory (RAM tier or read-ahead
//             cache) into its VRAM slot with uncached system-memory loads.
//   finalize  applies evictions and new table entries, re-maps the slots and
//             reports "fill done" so the host may reuse the source buffers.
// The host service thread below only reads routes, runs the slot policy,
// issues O_DIRECT reads and writes the acknowledgement. It never calls CUDA
// and never takes the GIL, so a blocked Python thread or a driver-wide
// operation elsewhere in the process cannot stall it.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <immintrin.h>

#include <cstdio>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <list>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kHeader = 8;   // box: seq ack count miss npred error fill_done reserved
constexpr int kMaxFill = 128;
constexpr int kSameLayerHint = 1 << 20;  // prediction ids >= this: read ahead for this layer
constexpr int kAckWords = 2 + 2 * kMaxFill + kMaxFill;  // nfill nevict, fills, evictions
constexpr int kMaxFields = 8;

__device__ __forceinline__ long long ld_acquire_sys(const long long* p) {
  long long v;
  asm volatile("ld.acquire.sys.global.s64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ int4 ld_cv_int4(const int4* p) {
  int4 v;
  asm volatile("ld.global.cv.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ int ld_cv_int(const int* p) {
  int v;
  asm volatile("ld.global.cv.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// Host buffers are rewritten between fills, so loads must not hit stale
// cached lines: .cv (default) never caches; CSF_FILL_LOAD=1 builds .cs.
#if defined(CSF_FILL_LOAD_CS)
__device__ __forceinline__ int4 ld_cs_int4(const int4* p) {
  int4 v;
  asm volatile("ld.global.cs.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
#define LOAD ld_cs_int4
#else
#define LOAD ld_cv_int4
#endif

struct Fields {
  long long dst[kMaxFields];      // device base address of each slot field
  long long row_bytes[kMaxFields];
  long long src_offset[kMaxFields];
  int count;
};

__global__ void wait_fill_kernel(long long* dev_state, long long* box, const long long* ack,
                                 Fields fields, long long timeout_ns) {
  __shared__ int go;
  __shared__ int nfill;
  __shared__ int fill_slot[kMaxFill];
  __shared__ long long fill_src[kMaxFill];
  if (threadIdx.x == 0) {
    go = 0;
    if (dev_state[1]) {
      long long seq = dev_state[0];
      unsigned long long start, now;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
      go = 1;
      while (ld_acquire_sys(box + 1) < seq) {
        __nanosleep(256);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        if (static_cast<long long>(now - start) > timeout_ns) {
          if (blockIdx.x == 0) {
            reinterpret_cast<volatile long long*>(box)[5] = seq;
            __threadfence_system();
          }
          go = 0;
          break;
        }
      }
    }
  }
  __syncthreads();
  if (!go) return;
  // Thread 0 alone reads the acknowledgement (system-scope loads serialize);
  // the block barrier then orders every thread's copies after its acquire.
  if (threadIdx.x == 0) {
    nfill = static_cast<int>(ld_acquire_sys(ack + 0));
    for (int e = 0; e < nfill; ++e) {
      fill_slot[e] = static_cast<int>(ld_acquire_sys(ack + 2 + 2 * e) & 0xffffffffLL);
      fill_src[e] = ld_acquire_sys(ack + 3 + 2 * e);
    }
  }
  __syncthreads();
  const long long G = gridDim.x, b = blockIdx.x;
  for (int e = 0; e < nfill; ++e) {
    long long src = fill_src[e];
    int slot = fill_slot[e];
    for (int f = 0; f < fields.count; ++f) {
      long long bytes = fields.row_bytes[f];
      char* dst = reinterpret_cast<char*>(fields.dst[f]) + slot * bytes;
      const char* s = reinterpret_cast<const char*>(src + fields.src_offset[f]);
      if ((bytes & 15) == 0 && ((src + fields.src_offset[f]) & 15) == 0) {
        long long chunks = bytes / 16;
        long long lo = chunks * b / G, hi = chunks * (b + 1) / G;
        const int4* s4 = reinterpret_cast<const int4*>(s);
        int4* d4 = reinterpret_cast<int4*>(dst);
        long long i = lo + threadIdx.x;
        // Four independent 16-byte loads in flight per thread hide PCIe latency.
        for (; i + 3 * blockDim.x < hi; i += 4 * blockDim.x) {
          int4 a = LOAD(s4 + i), b2 = LOAD(s4 + i + blockDim.x);
          int4 c = LOAD(s4 + i + 2 * blockDim.x), d = LOAD(s4 + i + 3 * blockDim.x);
          d4[i] = a; d4[i + blockDim.x] = b2; d4[i + 2 * blockDim.x] = c; d4[i + 3 * blockDim.x] = d;
        }
        for (; i < hi; i += blockDim.x) d4[i] = LOAD(s4 + i);
      } else if (b == 0) {
        const int* s1 = reinterpret_cast<const int*>(s);
        int* d1 = reinterpret_cast<int*>(dst);
        for (long long i = threadIdx.x; i < bytes / 4; i += blockDim.x) d1[i] = ld_cv_int(s1 + i);
      }
    }
  }
}

__global__ void finalize_kernel(const int64_t* ids, int n, int* table, int64_t* slots,
                                long long* dev_state, long long* box, const long long* ack) {
  if (!dev_state[1]) return;
  const int nfill = static_cast<int>(ld_acquire_sys(ack + 0));
  const int nevict = static_cast<int>(ld_acquire_sys(ack + 1));
  const long long* evict = ack + 2 + 2 * kMaxFill;
  // System-scope loads: a cached line from an earlier acknowledgement must not be reused.
  for (int i = threadIdx.x; i < nevict; i += blockDim.x) table[static_cast<int>(ld_acquire_sys(evict + i))] = -1;
  __syncthreads();
  for (int i = threadIdx.x; i < nfill; i += blockDim.x) {
    long long w0 = ld_acquire_sys(ack + 2 + 2 * i);
    table[static_cast<int>(w0 >> 32)] = static_cast<int>(w0 & 0xffffffffLL);
  }
  __syncthreads();
  __threadfence();
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    long long e = ids[i];
    slots[i] = e < 0 ? -1 : table[e];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    reinterpret_cast<volatile long long*>(box)[6] = dev_state[0];
    __threadfence_system();
  }
}

// ---------------------------------------------------------------- host service

inline long long now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline long long vload(const long long* p) { return *reinterpret_cast<const volatile long long*>(p); }
inline void vstore(long long* p, long long v) { *reinterpret_cast<volatile long long*>(p) = v; }

struct Buf {
  int layer = -1, expert = -1;
  int state = 0;                 // 0 empty, 1 loading, 2 ready
  bool prefetched = false;
  int lock_layer = -1;
  long long lock_seq = 0;
  unsigned long long last_use = 0;
};

struct Stats {
  long long calls = 0, hits = 0, misses = 0, miss_waits = 0;
  long long fill_ram = 0, fill_cache_ready = 0, fill_cache_waited = 0, fill_read = 0;
  long long prefetch_reads = 0, prefetch_used = 0, prefetch_skipped = 0;
  long long service_ns = 0, read_wait_ns = 0;
  long long pinned_borrowed = 0;
};

struct Layer {
  int id = -1, num_experts = 0, capacity = 0, fd = -1;
  long long* box = nullptr;
  long long* ack = nullptr;
  std::vector<int> slot_of, expert_of;
  std::vector<char> pinned;      // currently pinned (never an LRU victim)
  std::vector<char> home_pinned; // pinned by placement: re-pinned when it is loaded again
  std::vector<long long> last_use;
  int protect = -1;              // holds the safe slot: never lent
  std::list<int> lru;
  std::vector<std::list<int>::iterator> lru_it;
  std::vector<char> in_lru;
  std::vector<int> free_slots;
  std::vector<long long> ram_ptr;
  long long last_seq = 0;
  Stats st;
};

struct Job { int buf; int fd; long long offset; bool demand; };

class Service {
 public:
  long long stride = 0;
  char* cache_base = nullptr;
  std::vector<Buf> bufs;
  std::unordered_map<long long, int> index;
  std::map<int, Layer> layers;
  std::mutex mu;                 // guards bufs/index/queues
  std::condition_variable cv, work;
  std::deque<Job> demand_q, prefetch_q;
  int prefetch_inflight = 0, prefetch_cap = 8;
  unsigned long long tick = 0;
  std::atomic<bool> stop{false};
  std::atomic<long long> error_layer{-1};
  FILE* trace = nullptr;          // optional route log (simulate_cache.py format)
  long long trace_last_flush = 0;
  std::thread service;
  std::vector<std::thread> readers;
  std::string fatal;

  static long long key(int layer, int expert) { return static_cast<long long>(layer) * 65536 + expert; }

  bool unlocked(const Buf& b) {
    if (b.lock_layer < 0) return true;
    return vload(layers.at(b.lock_layer).box + 6) >= b.lock_seq;
  }

  int victim() {          // mu held
    int best = -1;
    for (int i = 0; i < static_cast<int>(bufs.size()); ++i) {
      Buf& b = bufs[i];
      if (b.state == 1 || !unlocked(b)) continue;
      if (best < 0 || b.state == 0 || b.last_use < bufs[best].last_use) {
        best = i;
        if (b.state == 0) break;
      }
    }
    if (best >= 0) {
      Buf& b = bufs[best];
      if (b.layer >= 0) index.erase(key(b.layer, b.expert));
      b = Buf();
    }
    return best;
  }

  void reader_loop() {
    while (true) {
      Job job;
      {
        std::unique_lock<std::mutex> lock(mu);
        work.wait(lock, [&] { return stop || !demand_q.empty() || !prefetch_q.empty(); });
        if (stop) return;
        if (!demand_q.empty()) { job = demand_q.front(); demand_q.pop_front(); }
        else { job = prefetch_q.front(); prefetch_q.pop_front(); }
      }
      char* dst = cache_base + static_cast<long long>(job.buf) * stride;
      long long done = 0;
      while (done < stride) {
        ssize_t r = pread(job.fd, dst + done, stride - done, job.offset + done);
        if (r <= 0) { std::lock_guard<std::mutex> l(mu); fatal = "pack read failed"; break; }
        done += r;
      }
      {
        std::lock_guard<std::mutex> lock(mu);
        bufs[job.buf].state = 2;
        if (!job.demand) prefetch_inflight--;
      }
      cv.notify_all();
    }
  }

  void serve_layer(Layer& L, long long seq) {
    long long t0 = now_ns();
    long long* box = L.box;
    int n = static_cast<int>(vload(box + 2));
    int miss = static_cast<int>(vload(box + 3));
    int npred = static_cast<int>(vload(box + 4));
    std::atomic_thread_fence(std::memory_order_acquire);
    const int* words = reinterpret_cast<const int*>(box + kHeader);
    std::vector<int> requested;
    requested.reserve(n);
    std::vector<char> seen(L.num_experts, 0);
    for (int i = 0; i < n; ++i) {
      int e = words[i];
      if (e >= 0 && e < L.num_experts && !seen[e]) { seen[e] = 1; requested.push_back(e); }
    }
    std::vector<int> pred(words + n, words + n + npred);
    L.st.calls++;
    if (trace && n > 0 && n % 6 == 0) {
      int header[2] = {L.id, n / 6};
      fwrite(header, sizeof(int), 2, trace);
      std::vector<short> ids16(words, words + n);
      fwrite(ids16.data(), sizeof(short), n, trace);
      if (t0 - trace_last_flush > 2000000000LL) { fflush(trace); trace_last_flush = t0; }
    }
    for (int e : requested) {
      L.last_use[e] = seq;
      if (L.slot_of[e] >= 0) {
        if (L.in_lru[e]) L.lru.splice(L.lru.end(), L.lru, L.lru_it[e]);
      }
    }
    if (miss) {
      L.st.miss_waits++;
      long long* ack = L.ack;
      std::vector<std::pair<int, int>> fills;   // expert, slot
      std::vector<long long> sources;
      std::vector<int> evicted;
      for (int e : requested) {
        if (L.slot_of[e] >= 0) { L.st.hits++; continue; }
        L.st.misses++;
        int slot;
        if (!L.free_slots.empty()) { slot = L.free_slots.back(); L.free_slots.pop_back(); }
        else {
          int v = -1;
          for (int cand : L.lru) if (!seen[cand]) { v = cand; break; }
          if (v >= 0) {
            L.lru.erase(L.lru_it[v]); L.in_lru[v] = 0;
          } else {
            // Soft pinning: a step routing more experts than the LRU holds
            // (several concurrent streams) borrows the least recently used
            // pinned slot that this request does not use.
            for (int cand = 0; cand < L.num_experts; ++cand) {
              if (!L.pinned[cand] || cand == L.protect || L.slot_of[cand] < 0 || seen[cand]) continue;
              if (v < 0 || L.last_use[cand] < L.last_use[v]) v = cand;
            }
            if (v < 0) { fatal = "layer " + std::to_string(L.id) + ": request exceeds all slots"; return; }
            L.pinned[v] = 0; L.st.pinned_borrowed++;
          }
          slot = L.slot_of[v]; L.slot_of[v] = -1; evicted.push_back(v);
        }
        L.slot_of[e] = slot; L.expert_of[slot] = e;
        if (L.home_pinned[e]) {
          L.pinned[e] = 1;   // back home: pinned again, the borrowed slot's new owner joined the LRU
        } else {
          L.lru.push_back(e); L.lru_it[e] = std::prev(L.lru.end()); L.in_lru[e] = 1;
        }
        fills.emplace_back(e, slot);
      }
      if (fills.size() > static_cast<size_t>(kMaxFill) || evicted.size() > static_cast<size_t>(kMaxFill)) {
        fatal = "too many fills in one request"; return;
      }
      // Sources: RAM tier, ready read-ahead buffer, in-flight read-ahead, or a demand read.
      sources.resize(fills.size(), 0);
      std::vector<int> used;
      {
        std::unique_lock<std::mutex> lock(mu);
        tick++;
        for (size_t i = 0; i < fills.size(); ++i) {
          int e = fills[i].first;
          if (L.ram_ptr[e]) { sources[i] = L.ram_ptr[e]; L.st.fill_ram++; continue; }
          auto it = index.find(key(L.id, e));
          int b;
          if (it != index.end()) {
            b = it->second;
            if (bufs[b].prefetched) { L.st.prefetch_used++; bufs[b].prefetched = false; }
            if (bufs[b].state == 2) L.st.fill_cache_ready++; else L.st.fill_cache_waited++;
          } else {
            b = victim();
            while (b < 0) { cv.wait_for(lock, std::chrono::microseconds(50)); b = victim(); }
            bufs[b].layer = L.id; bufs[b].expert = e; bufs[b].state = 1;
            index[key(L.id, e)] = b;
            demand_q.push_back(Job{b, L.fd, static_cast<long long>(e) * stride, true});
            L.st.fill_read++;
            work.notify_one();
          }
          bufs[b].last_use = tick;
          bufs[b].lock_layer = L.id;     // lock now: later fills in this request must not evict it
          bufs[b].lock_seq = seq;
          used.push_back(b);
          sources[i] = reinterpret_cast<long long>(cache_base + static_cast<long long>(b) * stride);
        }
        long long w0 = now_ns();
        for (int b : used) cv.wait(lock, [&] { return bufs[b].state == 2 || !fatal.empty(); });
        L.st.read_wait_ns += now_ns() - w0;
      }
      vstore(ack + 0, static_cast<long long>(fills.size()));
      vstore(ack + 1, static_cast<long long>(evicted.size()));
      for (size_t i = 0; i < fills.size(); ++i) {
        vstore(ack + 2 + 2 * i, (static_cast<long long>(fills[i].first) << 32) | fills[i].second);
        vstore(ack + 3 + 2 * i, sources[i]);
      }
      for (size_t i = 0; i < evicted.size(); ++i) vstore(ack + 2 + 2 * kMaxFill + i, evicted[i]);
      std::atomic_thread_fence(std::memory_order_release);
      vstore(box + 1, seq);
    } else {
      L.st.hits += requested.size();
    }
    // Read-ahead into host RAM: the next layer's predicted experts (capped), and
    // this layer's experts that expert-major prefill needs next (uncapped).
    std::vector<int> pred_next, pred_same;
    for (int e : pred) (e >= kSameLayerHint ? pred_same : pred_next).push_back(e - (e >= kSameLayerHint ? kSameLayerHint : 0));
    if (!pred_same.empty()) read_ahead(L, pred_same, false);
    auto nxt = layers.find(L.id + 1);
    if (!pred_next.empty() && nxt != layers.end()) read_ahead(nxt->second, pred_next, true);
    L.st.service_ns += now_ns() - t0;
  }

  void read_ahead(Layer& N, const std::vector<int>& experts, bool capped) {
    {
      std::lock_guard<std::mutex> lock(mu);
      for (int e : experts) {
        if (e < 0 || e >= N.num_experts || N.slot_of[e] >= 0 || N.ram_ptr[e]) continue;
        if (index.count(key(N.id, e))) continue;
        if (capped && prefetch_inflight >= prefetch_cap) { N.st.prefetch_skipped++; continue; }
        int b = victim();
        if (b < 0) { N.st.prefetch_skipped++; break; }
        bufs[b].layer = N.id; bufs[b].expert = e; bufs[b].state = 1; bufs[b].prefetched = true;
        bufs[b].last_use = ++tick;
        index[key(N.id, e)] = b;
        prefetch_q.push_back(Job{b, N.fd, static_cast<long long>(e) * stride, false});
        prefetch_inflight++;
        N.st.prefetch_reads++;
        work.notify_one();
      }
    }
  }

  void service_loop() {
    unsigned idle = 0;
    while (!stop) {
      bool any = false;
      for (auto& kv : layers) {
        Layer& L = kv.second;
        long long seq = vload(L.box + 0);
        if (seq != L.last_seq) {
          L.last_seq = seq;
          any = true;
          serve_layer(L, seq);
          if (!fatal.empty()) { fprintf(stderr, "[csf-v4] fatal: %s\n", fatal.c_str()); fflush(stderr); _exit(81); }
        }
        if (vload(L.box + 5)) {
          fprintf(stderr, "[csf-v4] GPU wait timed out on layer %d seq %lld\n", L.id, vload(L.box + 5));
          fflush(stderr);
          _exit(80);
        }
      }
      if (any) { idle = 0; continue; }
      if (++idle < 20000) _mm_pause();
      else std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
  }
};

Service* g = nullptr;

}  // namespace

// ------------------------------------------------------------------ bindings

void wait_fill(torch::Tensor dev_state, int64_t box_ptr, int64_t ack_ptr, std::vector<int64_t> dst,
               std::vector<int64_t> row_bytes, std::vector<int64_t> src_offset, int64_t timeout_ns,
               int64_t blocks) {
  Fields f{};
  f.count = static_cast<int>(dst.size());
  TORCH_CHECK(f.count <= kMaxFields);
  for (int i = 0; i < f.count; ++i) { f.dst[i] = dst[i]; f.row_bytes[i] = row_bytes[i]; f.src_offset[i] = src_offset[i]; }
  auto stream = c10::cuda::getCurrentCUDAStream();
  wait_fill_kernel<<<static_cast<int>(blocks), 256, 0, stream>>>(
      reinterpret_cast<long long*>(dev_state.data_ptr<int64_t>()), reinterpret_cast<long long*>(box_ptr),
      reinterpret_cast<const long long*>(ack_ptr), f, timeout_ns);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void finalize(torch::Tensor ids, torch::Tensor table, torch::Tensor slots, torch::Tensor dev_state,
              int64_t box_ptr, int64_t ack_ptr) {
  auto stream = c10::cuda::getCurrentCUDAStream();
  finalize_kernel<<<1, 256, 0, stream>>>(
      ids.data_ptr<int64_t>(), static_cast<int>(ids.numel()), table.data_ptr<int>(), slots.data_ptr<int64_t>(),
      reinterpret_cast<long long*>(dev_state.data_ptr<int64_t>()), reinterpret_cast<long long*>(box_ptr),
      reinterpret_cast<const long long*>(ack_ptr));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void init(int64_t cache_ptr, int64_t cache_buffers, int64_t stride, int64_t readers, int64_t prefetch_cap) {
  TORCH_CHECK(g == nullptr, "v4 service already initialized");
  g = new Service();
  g->stride = stride;
  g->cache_base = reinterpret_cast<char*>(cache_ptr);
  g->bufs.resize(cache_buffers);
  g->prefetch_cap = static_cast<int>(prefetch_cap);
  for (int i = 0; i < readers; ++i) g->readers.emplace_back([] { g->reader_loop(); });
}

void register_layer(int64_t layer, int64_t box_ptr, int64_t ack_ptr, int64_t num_experts, int64_t capacity,
                    std::vector<int64_t> resident_experts, std::vector<int64_t> resident_slots,
                    std::vector<int64_t> pinned_experts, std::vector<int64_t> free_slots,
                    std::string pack_path, std::vector<int64_t> ram_ptr) {
  TORCH_CHECK(g != nullptr, "call init first");
  TORCH_CHECK(!g->service.joinable(), "register all layers before start");
  Layer L;
  L.id = static_cast<int>(layer);
  L.box = reinterpret_cast<long long*>(box_ptr);
  L.ack = reinterpret_cast<long long*>(ack_ptr);
  L.num_experts = static_cast<int>(num_experts);
  L.capacity = static_cast<int>(capacity);
  L.slot_of.assign(num_experts, -1);
  L.expert_of.assign(capacity, -1);
  L.pinned.assign(num_experts, 0);
  L.in_lru.assign(num_experts, 0);
  L.lru_it.resize(num_experts);
  L.ram_ptr.assign(ram_ptr.begin(), ram_ptr.end());
  TORCH_CHECK(static_cast<int64_t>(L.ram_ptr.size()) == num_experts);
  for (auto e : pinned_experts) L.pinned[e] = 1;
  L.home_pinned = L.pinned;
  L.last_use.assign(num_experts, 0);
  if (!pinned_experts.empty()) L.protect = static_cast<int>(pinned_experts[0]);
  for (size_t i = 0; i < resident_experts.size(); ++i) {
    int e = static_cast<int>(resident_experts[i]), s = static_cast<int>(resident_slots[i]);
    L.slot_of[e] = s; L.expert_of[s] = e;
    if (!L.pinned[e]) { L.lru.push_back(e); L.lru_it[e] = std::prev(L.lru.end()); L.in_lru[e] = 1; }
  }
  for (auto s : free_slots) L.free_slots.push_back(static_cast<int>(s));
  L.fd = open(pack_path.c_str(), O_RDONLY | O_DIRECT);
  TORCH_CHECK(L.fd >= 0, "cannot open ", pack_path);
  L.last_seq = vload(L.box + 0);
  g->layers[L.id] = std::move(L);
}

void trace_to(std::string path) {
  TORCH_CHECK(g != nullptr && !g->service.joinable(), "set the trace before start");
  g->trace = fopen(path.c_str(), "ab");
  TORCH_CHECK(g->trace != nullptr, "cannot open ", path);
  setvbuf(g->trace, nullptr, _IOFBF, 1 << 22);
}

void start() {
  TORCH_CHECK(g != nullptr && !g->service.joinable());
  g->service = std::thread([] { g->service_loop(); });
}

std::vector<int64_t> resident(int64_t layer) {   // flat [expert, slot, ...]; debugging/validation
  std::vector<int64_t> out;
  auto& L = g->layers.at(static_cast<int>(layer));
  for (int e = 0; e < L.num_experts; ++e) if (L.slot_of[e] >= 0) { out.push_back(e); out.push_back(L.slot_of[e]); }
  return out;
}

std::map<std::string, int64_t> stats(int64_t layer) {
  auto& s = g->layers.at(static_cast<int>(layer)).st;
  return {{"calls", s.calls}, {"hits", s.hits}, {"misses", s.misses}, {"miss_waits", s.miss_waits},
          {"fill_ram", s.fill_ram}, {"fill_cache_ready", s.fill_cache_ready},
          {"fill_cache_waited", s.fill_cache_waited}, {"fill_read", s.fill_read},
          {"prefetch_reads", s.prefetch_reads}, {"prefetch_used", s.prefetch_used},
          {"prefetch_skipped", s.prefetch_skipped}, {"service_ns", s.service_ns},
          {"read_wait_ns", s.read_wait_ns}, {"pinned_borrowed", s.pinned_borrowed}};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("ACK_WORDS") = kAckWords;
  m.attr("MAX_FILL") = kMaxFill;
  m.def("wait_fill", &wait_fill);
  m.def("finalize", &finalize);
  m.def("init", &init);
  m.def("register_layer", &register_layer);
  m.def("start", &start);
  m.def("trace_to", &trace_to);
  m.def("resident", &resident);
  m.def("stats", &stats);
}
