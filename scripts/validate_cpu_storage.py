"""Independent CPU check of TP slices and losslessly decoded scale values.

Pinning is disabled and comparisons run on CPU; runtime imports may initialize CUDA.
GPU staging and kernel correctness are tested separately.
"""
import argparse
import json
from unittest.mock import patch
import numpy as np
import torch

from csf_streaming.storage import read_experts


def decode(fixed, exceptions, rows, columns):
    packed = fixed.numpy().reshape(rows // 16, 16 * (1 + (columns + 7) // 8))
    base = packed[:, :16].reshape(rows, 1).astype(np.uint16)
    bits = np.unpackbits(packed[:, 16:].reshape(rows, -1), axis=1, bitorder="little")[:, :columns]
    result = (base + bits).reshape(-1)
    words = exceptions.numpy().reshape(-1).astype(np.uint32)
    result[words & np.uint32(0xffffff)] = words >> np.uint32(24)
    return result.reshape(rows, columns)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",required=True); p.add_argument("--output",required=True)
    a=p.parse_args()
    from vllm.model_executor.model_loader.csf_utils import CsfTensorReader
    from vllm.model_executor.model_loader.mxfp4_csf_loader import checkpoint_contract,expert_tensor_names
    torch.set_num_threads(1)
    contract=checkpoint_contract(a.root)
    h,n=5120,2304
    checks=[]
    for layer in (0,20,39):
        for rank in (0,1):
            ids=[0,177,383]
            with patch.object(torch.Tensor,"pin_memory",lambda self:self):
                w,staged,seconds=read_experts(a.root,layer,ids,h,n,rank,2)
            first,last=rank*n//2,(rank+1)*n//2
            with CsfTensorReader(a.root,contract["source_names"],"mxfp4") as reader:
                for i,expert in enumerate(ids):
                    projections=[reader.matrix(wn,sn) for wn,sn in expert_tensor_names(contract["family"],layer,expert)]
                    expected=torch.cat([m.weight[first:last,:].view(torch.uint8) for m in projections[:2]])
                    assert torch.equal(w.w13[i],expected)
                    assert torch.equal(w.w2[i],projections[2].weight[:,first//2:last//2].view(torch.uint8))
                    expected_scale=np.concatenate([decode(m.fixed,m.exceptions,n,h//32)[first:last] for m in projections[:2]])
                    np.testing.assert_array_equal(decode(w.w13_scales.fixed[i],w.w13_scales.exceptions[i],n,h//32),expected_scale)
                    expected_scale=decode(projections[2].fixed,projections[2].exceptions,h,n//32)[:,first//32:last//32]
                    np.testing.assert_array_equal(decode(w.w2_scales.fixed[i],w.w2_scales.exceptions[i],h,n//64),expected_scale)
            checks.append(dict(layer=layer,rank=rank,experts=ids,staged_bytes=staged,seconds=seconds))
    result=dict(ok=True,cuda_initialized=torch.cuda.is_initialized(),checks=checks)
    with open(a.output,"w") as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
