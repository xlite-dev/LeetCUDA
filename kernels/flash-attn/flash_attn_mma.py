import math
import time
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from typing import Optional
from flash_attn import flash_attn_func


torch.set_grad_enabled(False)
# Load the CUDA kernel as a python module
lib = load(name='flash_attn_lib', 
           sources=[
               './naive/flash_attn_cuda.cu',
               './mma/flash_attn_mma_old.cu',
               './mma/flash_attn_mma.cu',
               './pybind/flash_attn.cc'], 
           extra_cuda_cflags=[
               "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math"
            ], 
           extra_cflags=['-std=c++17'])


# un-fused naive attn
def naive_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y


def run_benchmark(perf_func: callable, 
                  q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  tag: str, out: Optional[torch.Tensor] = None, 
                  stages: int = -1,
                  warmup: int = 0, iters: int = 1,
                  show_all: bool = False):
    if out is not None: 
        out.fill_(0)      
    if out is not None:
        for i in range(warmup):
            if stages >= 1:
                perf_func(q, k, v, out, stages)
            else:
                perf_func(q, k, v, out)
    else:
        for i in range(warmup):
            _ = perf_func(q, k, v)
    
    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            if stages >= 1:
                perf_func(q, k, v, out, stages)
            else:
                perf_func(q, k, v, out)
    else:
        for i in range(iters):
            out = perf_func(q, k, v)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000 # ms
    mean_time = total_time / iters
    out_info = f"{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>20}: {out_val}, time:{mean_time:.6f}ms")
    if show_all: print(out[0, 0, 0, :])
    return out.clone(), mean_time

@torch.no_grad
def as_col_major(x: torch.Tensor):
    B, H, N, D = x.size()
    x_col_major = x.clone()
    # convert a row major tensor -> col major with contiguous storage
    for i in range(B):
        for j in range(H):
            x_head = x[i, j, ...]
            x_head_trans = x_head.t()
            x_head_trans.reshape(x_head.shape).contiguous()
            x_col_major[i, j, ...] = x_head_trans
    return x_col_major.contiguous() # must be a contiguous tensor

Bs = [1]
Hs = [1]
Ns = [64]
Ds = [64] # only support [64, 128] now
# batch_size, n_head, seq_len, head_dim (B,nh,N,d)
BHNDs = [(B, H, N, D) for B in Bs for H in Hs for N in Ns for D in Ds]

print("-" * 100)
print(" "* 25 + "B: batch_size, H: n_head, N: seq_len, D: head_dim")
for (B, H, N, D) in BHNDs:
    print("-" * 100)
    print(" " * 40 + f"B={B}, H={H}, N={N}, D={D}")
    # q = torch.ones(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    # k = torch.ones(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    # v = torch.ones(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    # o = torch.ones(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    o = torch.randn(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    fq = q.transpose(1, 2)
    fk = k.transpose(1, 2)
    fv = v.transpose(1, 2)
    torch.cuda.synchronize()
    print("-" * 100)
    print(q)
    print("-" * 100)
    print(k)
    print("-" * 100)
    print(v)
    print("-" * 100)
  
    # using fp16 Tesor Core MMA instruction
    run_benchmark(lib.flash_attn_mma_naive, q, k, v, "mma(naive)", o)
    run_benchmark(lib.flash_attn_mma_stages, q, as_col_major(k), v, "mma(stage)", o, stages=1)
    run_benchmark(lib.flash_attn_mma_stages, q, k, v, "mma(stage)", o, stages=1)
    run_benchmark(flash_attn_func, fq, fk, fv, "(flash)")
    run_benchmark(F.scaled_dot_product_attention, q, k, v, "(sdpa)")
    print("-" * 100)
