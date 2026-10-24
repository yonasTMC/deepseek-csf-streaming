import torch, time, threading, subprocess
N=1<<30
def bw(dev, n=20, d2h=False):
    h=torch.empty(N,dtype=torch.uint8).pin_memory(); g=torch.empty(N,dtype=torch.uint8,device=dev)
    s=torch.cuda.Stream(dev)
    with torch.cuda.stream(s):
        for _ in range(2): (h.copy_(g,non_blocking=True) if d2h else g.copy_(h,non_blocking=True))
        s.synchronize(); t=time.perf_counter()
        for _ in range(n): (h.copy_(g,non_blocking=True) if d2h else g.copy_(h,non_blocking=True))
        s.synchronize()
    return n*N/(time.perf_counter()-t)/2**30
for d in (0,1): print(f"H2D gpu{d}: {bw(d):.1f} GiB/s")
res={}
def run(d): res[d]=bw(d,40)
ts=[threading.Thread(target=run,args=(d,)) for d in (0,1)]; [t.start() for t in ts]
time.sleep(1); print(subprocess.run(["nvidia-smi","--query-gpu=index,pcie.link.gen.current,pcie.link.width.current","--format=csv,noheader"],capture_output=True,text=True).stdout)
[t.join() for t in ts]; print("H2D concurrent:",{k:round(v,1) for k,v in res.items()})
a=torch.empty(N,dtype=torch.uint8,device=0); b=torch.empty(N,dtype=torch.uint8,device=1)
print("P2P access", torch.cuda.can_device_access_peer(0,1))
for _ in range(2): b.copy_(a)
torch.cuda.synchronize(0); torch.cuda.synchronize(1); t=time.perf_counter()
for _ in range(10): b.copy_(a)
torch.cuda.synchronize(0); torch.cuda.synchronize(1); print(f"GPU0->GPU1 {10*N/(time.perf_counter()-t)/2**30:.1f} GiB/s")
