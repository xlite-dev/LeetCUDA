import os
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
    parser.add_argument("--sdpa", action="store_true")
    parser.add_argument("--check", '--ch', action="store_true")
    parser.add_argument("--B", type=int, default=None)
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--D", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    return parser.parse_args()


args = get_args()
print(args)


def get_project_dir():
    return os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))


project_dir = get_project_dir()


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
               './mma/flash_attn_mma_naive.cu',
               './mma/flash_attn_mma_stage.cu',
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
                f"-I {project_dir}/kernels/flash-attn/utils",
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


Bs = [1] if not args.B else [args.B]
Hs = [1] if not args.H else [args.H]
Ns = [1024] if not args.N else [args.N]
Ds = [64] if not args.D else [args.D] 
# batch_size, n_head, seq_len, head_dim (B,H,N,D)
BHNDs = [(B, H, N, D) for B in Bs for H in Hs for N in Ns for D in Ds]

seed = args.seed if args.seed else random.choice(range(10000))
set_rand_seed(seed)
print("-" * 100)
print(" "* 10 + f"B: batch_size, H: n_head, N: seq_len, D: head_dim, "
      f"seed: {seed}, Warmup: {args.warmup}, Iters: {args.iters}")
for (B, H, N, D) in BHNDs:
    print("-" * 100)
    print(" " * 40 + f"B={B}, H={H}, N={N}, D={D}")
    if args.rand_q or args.rand_qkv:
        q = torch.randn((B, H, N, D), dtype=torch.half, device="cuda")
    else:
        q = torch.ones(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    if args.rand_k or args.rand_qkv:
        k = torch.randn((B, H, N, D), dtype=torch.half, device="cuda")
    else:
        k = torch.ones(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
        if args.range_k:
            for i in range(N):
                k[:, :, i, :] = (i + 1) / N
            k = k.cuda().half().contiguous()
    if args.rand_v or args.rand_qkv:
        v = torch.randn((B, H, N, D), dtype=torch.half, device="cuda")
    else:
        v = torch.ones(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    o = torch.zeros(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    tk = k.transpose(-2, -1).contiguous()
    fq = q.transpose(1, 2)
    fk = k.transpose(1, 2)
    fv = v.transpose(1, 2)
    torch.cuda.synchronize()
    
    if args.naive:
        out_naive, _ = run_benchmark(naive_attn, q, k, v, "(naive)")
    # using fp16 Tesor Core MMA instruction
    out_mma_naive, _ = run_benchmark(lib.flash_attn_mma_naive, q, k, v, "mma(naive)", o)
    out_mma_stage1, _ = run_benchmark(lib.flash_attn_mma_stages, q, tk, v, "mma(stage1)", o, stages=1)
    out_mma_stage2, _ = run_benchmark(lib.flash_attn_mma_stages, q, tk, v, "mma(stage2)", o, stages=2)
    out_flash, _ = run_benchmark(flash_attn_func, fq, fk, fv, "(flash)")
    if args.sdpa:
        out_sdpa, _ = run_benchmark(F.scaled_dot_product_attention, q, k, v, "(sdpa)")
    
    if args.check:
        print("-" * 100)
        print(f"all close(mma vs flash): {torch.allclose(out_mma_stage1, out_flash)}")
      

        
