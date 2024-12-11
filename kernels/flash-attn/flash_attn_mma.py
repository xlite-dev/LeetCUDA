import math
import time
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from typing import Optional
from flash_attn import flash_attn_func
import argparse
import random
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rand-q", '--rq', action="store_true")
    parser.add_argument("--rand-k", '--rk', action="store_true")
    parser.add_argument("--rand-v", '--rv', action="store_true")
    parser.add_argument("--rand-qkv", '--rqkv', action="store_true")
    parser.add_argument("--range-k", '--gk', action="store_true")
    parser.add_argument("--naive", action="store_true")
    parser.add_argument("--check", '--ch', action="store_true")
    parser.add_argument("--B", type=int, default=None)
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--D", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    return parser.parse_args()

args = get_args()
print(args)

def set_rand_seed(seed:int=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
                "--use_fast_math",
                "-DFLASH_ATTN_MMA_DEBUG" if args.debug else ""
            ], 
           extra_cflags=['-std=c++17'])


# un-fused naive attn
def naive_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y


def run_benchmark(perf_func: callable, 
                  q: torch.Tensor, 
                  k: torch.Tensor, 
                  v: torch.Tensor,
                  tag: str, 
                  out: Optional[torch.Tensor] = None, 
                  s: Optional[torch.Tensor] = None, 
                  stages: int = -1,
                  warmup: int = args.warmup, 
                  iters: int = args.iters,
                  show_all: bool = args.check):
    if out is not None: 
        out.fill_(0)
    if s is not None:
        s.fill_(0)      
    if out is not None:
        for i in range(warmup):
            if stages >= 1:
                if s is not None:
                    perf_func(q, k, v, out, s, stages)
                else:
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
                if s is not None:
                    perf_func(q, k, v, out, s, stages)
                else:
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
    out_val = out.flatten()[:3].detach().cpu().numpy().tolist()
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>20}: {out_val}, time:{mean_time:.6f}ms")
    if show_all: 
        print(out)
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
            x_head_trans = x_head_trans.reshape(x_head.shape).contiguous()
            x_col_major[i, j, ...] = x_head_trans
    return x_col_major.contiguous() # must be a contiguous tensor

Bs = [1] if not args.B else [args.B]
Hs = [1] if not args.H else [args.H]
Ns = [64*3] if not args.N else [args.N]
Ds = [64] if not args.D else [args.D] 
# batch_size, n_head, seq_len, head_dim (B,H,N,D)
BHNDs = [(B, H, N, D) for B in Bs for H in Hs for N in Ns for D in Ds]

set_rand_seed(random.choice(range(10000)))
print("-" * 100)
print(" "* 25 + "B: batch_size, H: n_head, N: seq_len, D: head_dim")
for (B, H, N, D) in BHNDs:
    print("-" * 100)
    print(" " * 40 + f"B={B}, H={H}, N={N}, D={D}")
    if args.rand_q or args.rand_qkv:
        q = torch.empty((B, H, N, D), dtype=torch.half, device="cuda").normal_(mean=0.0, std=0.1)
    else:
        q = torch.ones(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    if args.rand_k or args.rand_qkv:
        k = torch.empty((B, H, N, D), dtype=torch.half, device="cuda").normal_(mean=0.0, std=0.1)
    else:
        k = torch.ones(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
        if args.range_k:
            for i in range(N):
                k[:, :, i, :] = (i + 1) / N
            k = k.cuda().half().contiguous()
    if args.rand_v or args.rand_qkv:
        v = torch.empty((B, H, N, D), dtype=torch.half, device="cuda").normal_(mean=0.0, std=0.5)
    else:
        v = torch.ones(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    o = torch.zeros(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    tk = k.transpose(-2, -1).contiguous()
    fq = q.transpose(1, 2)
    fk = k.transpose(1, 2)
    fv = v.transpose(1, 2)
    s = torch.zeros((B, H, N, N), device="cuda", dtype=torch.half).contiguous()
    torch.cuda.synchronize()
    
    # using fp16 Tesor Core MMA instruction
    # NOTE: 数值稳定性，如果不限制normal(0.0, 0.5), 精度偶发异常。
    # 表现为: random q, N>64, 结果依然有问题; 当 rk+rv, N>640结果也有问题
    # 和不同的随机数种子也有关系。
    out, _ = run_benchmark(lib.flash_attn_mma_stages, q, tk, v, "mma(stage)", o, s, stages=1)
    # run_benchmark(flash_attn_func, fq, fk, fv, "(flash)")
    out, _  = run_benchmark(F.scaled_dot_product_attention, q, k, v, "(sdpa)")
    if args.naive:
        out, _ = run_benchmark(lib.flash_attn_mma_naive, q, k, v, "mma(naive)", o)
    print("-" * 100)
    if args.check:
        print("------------------------------ Q ---------------------------")
        print(q)
        print("------------------------------ K ---------------------------")
        print(k)
        print("------------------------------ K^T -------------------------")
        print(tk)
        print("------------------------------ s, Q@K^T -----------------------")
        print("s:\n")
        print(s)
        print(s.shape)
        print(s[:, :, :, int(N/2):])
        print("\nq @ tk:\n")
        # qk = q @ tk
        qk = torch.matmul(q, tk)
        print(qk)
        print(qk.shape)
        print(qk[:, :, :, int(N/2):])
        print((s - qk).max())
        print(f"\nall close: {torch.allclose(s, qk, atol=1e-1)}")
        diff = s - qk
        print("\ndiff:\n")
        print(diff.float())
        print(f"diff min: {diff.min()}, max: {diff.max()}")
        for i in range(int(N/64)):
            print("*" * 100)
            diff_slice = diff[:, :, :, (i*64):(i+1)*64].float()
            print(f"\ndiff_slice[:, :, :, {(i*64)}:{(i+1)*64}]:\n")
            print(diff_slice)
            print(f"\ndiff_slice min: {diff_slice.min()}, max: {diff_slice.max()}\n")

       
        
