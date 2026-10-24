// Graph-capturable expert residency handshake between the GPU and a host thread.
//
// publish(): translate expert ids to VRAM slots through a device table. Every
// call copies its ids (and optional next-layer predictions) into a host-mapped
// mailbox and bumps the mailbox sequence; when any expert is absent it also
// raises the mailbox miss flag and arms a device-side wait.
// wait(): if armed, spin until the host acknowledges this sequence (after it
// has loaded the experts and updated the table on its copy stream), then map
// the ids again. A deadline turns a lost host thread into an error flag
// instead of a hung GPU.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

// Host-mapped mailbox layout (int64 words), one per layer:
//   [0] seq  [1] ack  [2] count  [3] miss  [4] npred  [5] error  [6..7] reserved
//   [8 ...] ids (int32, count entries) then predictions (int32, npred entries)
constexpr int kHeader = 8;

// Host writes to mapped memory are only guaranteed visible to system-scope
// loads; a volatile load can keep returning a cached line.
__device__ __forceinline__ long long load_acquire_sys(const long long* p) {
  long long v;
  asm volatile("ld.acquire.sys.global.s64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__global__ void publish_kernel(const int64_t* __restrict__ ids, int n,
                               const int64_t* __restrict__ pred, int npred,
                               const int* __restrict__ table, int64_t* slots,
                               long long* dev_state,  // [0] seq [1] armed
                               long long* box, int capacity_ids) {
  __shared__ int any_miss;
  if (threadIdx.x == 0) any_miss = 0;
  __syncthreads();
  int* box_ids = reinterpret_cast<int*>(box + kHeader);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    long long e = ids[i];
    int slot = e < 0 ? -1 : table[e];
    if (e >= 0 && slot < 0) any_miss = 1;
    slots[i] = slot;
    if (i < capacity_ids) box_ids[i] = static_cast<int>(e);
  }
  for (int i = threadIdx.x; i < npred && n + i < capacity_ids; i += blockDim.x)
    box_ids[n + i] = static_cast<int>(pred[i]);
  __syncthreads();
  if (threadIdx.x == 0) {
    long long seq = dev_state[0] + 1;
    dev_state[0] = seq;
    dev_state[1] = any_miss;
    volatile long long* vbox = box;
    vbox[2] = n;
    vbox[3] = any_miss;
    vbox[4] = npred;
    __threadfence_system();
    vbox[0] = seq;
    __threadfence_system();
  }
}

__global__ void wait_kernel(const int64_t* __restrict__ ids, int n, const int* table,
                            int64_t* slots, long long* dev_state, long long* box,
                            long long timeout_ns) {
  __shared__ int ok;
  if (threadIdx.x == 0) {
    ok = 1;
    if (dev_state[1]) {
      long long seq = dev_state[0];
      unsigned long long start, now;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
      while (load_acquire_sys(box + 1) < seq) {
        __nanosleep(200);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        if (static_cast<long long>(now - start) > timeout_ns) {
          reinterpret_cast<volatile long long*>(box)[5] = seq;
          __threadfence_system();
          ok = 0;
          break;
        }
      }
    } else {
      ok = 2;  // nothing to wait for; slots from publish are final
    }
  }
  __syncthreads();
  if (ok == 2) return;
  const volatile int* vtable = table;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    long long e = ids[i];
    slots[i] = e < 0 ? -1 : vtable[e];
  }
}

}  // namespace

void publish(torch::Tensor ids, torch::Tensor pred, torch::Tensor table, torch::Tensor slots,
             torch::Tensor dev_state, int64_t box_ptr, int64_t capacity_ids) {
  auto stream = c10::cuda::getCurrentCUDAStream();
  int n = static_cast<int>(ids.numel());
  int npred = static_cast<int>(pred.numel());
  publish_kernel<<<1, 256, 0, stream>>>(
      ids.data_ptr<int64_t>(), n, npred ? pred.data_ptr<int64_t>() : nullptr, npred,
      table.data_ptr<int>(), slots.data_ptr<int64_t>(),
      reinterpret_cast<long long*>(dev_state.data_ptr<int64_t>()),
      reinterpret_cast<long long*>(box_ptr), static_cast<int>(capacity_ids));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wait(torch::Tensor ids, torch::Tensor table, torch::Tensor slots, torch::Tensor dev_state,
          int64_t box_ptr, int64_t timeout_ns) {
  auto stream = c10::cuda::getCurrentCUDAStream();
  wait_kernel<<<1, 256, 0, stream>>>(
      ids.data_ptr<int64_t>(), static_cast<int>(ids.numel()), table.data_ptr<int>(),
      slots.data_ptr<int64_t>(), reinterpret_cast<long long*>(dev_state.data_ptr<int64_t>()),
      reinterpret_cast<long long*>(box_ptr), timeout_ns);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// A dedicated stream. torch.cuda.Stream() hands out streams from a shared
// round-robin pool, so a "private" copy stream can be the very stream a
// graph replay is spinning on in wait_kernel.
int64_t create_stream() {
  int least = 0, greatest = 0;
  C10_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
  cudaStream_t stream = nullptr;
  C10_CUDA_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest));
  return reinterpret_cast<int64_t>(stream);
}

// Pin an existing, page-aligned host range at its exact size. torch's
// pinned allocator rounds requests up to a power of two (38.5 GiB -> 64 GiB).
void host_register(int64_t ptr, int64_t bytes) {
  C10_CUDA_CHECK(cudaHostRegister(reinterpret_cast<void*>(ptr), static_cast<size_t>(bytes),
                                  cudaHostRegisterPortable | cudaHostRegisterMapped));
}

int64_t device_pointer(int64_t host_ptr) {
  void* dev = nullptr;
  C10_CUDA_CHECK(cudaHostGetDevicePointer(&dev, reinterpret_cast<void*>(host_ptr), 0));
  return reinterpret_cast<int64_t>(dev);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("publish", &publish);
  m.def("wait", &wait);
  m.def("device_pointer", &device_pointer);
  m.def("create_stream", &create_stream);
  m.def("host_register", &host_register);
}
