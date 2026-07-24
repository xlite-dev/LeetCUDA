"""Bench torch SDPA and flash-attention on typical shapes.

Usage:
    python3 bench_sdpa.py --bhnd 1,32,8192,64
    python3 bench_sdpa.py --bhnd 1,32,8192,128 --backend flash
    python3 bench_sdpa.py --bhnd 1,32,8192,64 --backend efficient --warmup 5 --iters 20

TFLOPS = 4 * B * H * N * N * D / ms / 1e9 (forward only, non-causal).
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F


def time_fn(fn, *args, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_sdpa(q, k, v, scale: float, backend: str, warmup: int, iters: int):
    sdpa_kwargs = {"scale": scale, "is_causal": False, "dropout_p": 0.0}
    if backend == "auto":
        fn = lambda: F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
        label = "torch SDPA (auto)"
    else:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backend_map = {
            "math": SDPBackend.MATH,
            "flash": SDPBackend.FLASH_ATTENTION,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "cudnn": SDPBackend.CUDNN_ATTENTION,
        }
        sel = backend_map[backend]

        def fn():
            with sdpa_kernel(sel):
                F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)

        label = f"torch SDPA ({backend})"
    ms = time_fn(fn, warmup=warmup, iters=iters)
    return ms, label


def bench_flash_attn(q, k, v, scale: float, warmup: int, iters: int):
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        return None, "flash-attn (not installed)"
    # flash-attn expects [B, N, H, D]
    q_fa = q.permute(0, 2, 1, 3).contiguous()
    k_fa = k.permute(0, 2, 1, 3).contiguous()
    v_fa = v.permute(0, 2, 1, 3).contiguous()

    def fn():
        flash_attn_func(q_fa, k_fa, v_fa, softmax_scale=scale, causal=False)

    ms = time_fn(fn, warmup=warmup, iters=iters)
    return ms, "flash-attn 2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bhnd", type=str, default="1,32,8192,64",
                        help="B,H,N,D comma-separated")
    parser.add_argument("--backend", choices=["auto", "math", "flash", "efficient", "cudnn"],
                        default="auto")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--no-flash-attn", action="store_true",
                        help="skip flash-attn library bench")
    args = parser.parse_args()

    B, H, N, D = map(int, args.bhnd.split(","))
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    scale = 1.0 / math.sqrt(D)

    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H, N, D, dtype=dtype, device="cuda")

    flops = 4 * B * H * N * N * D

    print(f"=== bench_sdpa: B={B} H={H} N={N} D={D} dtype={args.dtype} ===")
    print(f"{'Kernel':<30} {'Time(ms)':<12} {'TFLOPS':<10}")
    print("-" * 52)

    # torch SDPA
    ms, label = bench_sdpa(q, k, v, scale, args.backend, args.warmup, args.iters)
    tflops = flops / (ms * 1e9) if ms > 0 else 0
    print(f"{label:<30} {ms:<12.3f} {tflops:<10.1f}")

    # flash-attn library
    if not args.no_flash_attn:
        ms_fa, label_fa = bench_flash_attn(q, k, v, scale, args.warmup, args.iters)
        if ms_fa is not None:
            tflops_fa = flops / (ms_fa * 1e9)
            print(f"{label_fa:<30} {ms_fa:<12.3f} {tflops_fa:<10.1f}")
        else:
            print(f"{label_fa:<30} {'-':<12} {'-':<10}")

    print("=== Bench done ===")


if __name__ == "__main__":
    main()
