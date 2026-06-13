# -*- coding: utf-8 -*-
"""
practice_all_reduce.py - Practice-use FP16 all-reduce benchmark
================================================================
After learning the all-reduce kernel, use this for repeated practice:
  - Only the best-performing FP16 kernel (f16x8_pack)
  - Two features only: check_correctness and benchmark
  - Plain kernel name (all_reduce_sum_f16), no optimization hints
"""

import argparse
import os
import time
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_HERE, "build", "practice_all_reduce_lib")
os.makedirs(_BUILD_DIR, exist_ok=True)

lib = load(
    name="practice_all_reduce_lib",
    sources=[os.path.join(_HERE, "practice_all_reduce.cu")],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
    build_directory=_BUILD_DIR,
)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
def check_correctness(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> bool:
    ref = torch.sum(x, dtype=torch.float32)
    got = perf_func(x)
    torch.cuda.synchronize()
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"[correctness] {tag}: {status}")
    if not ok:
        diff = (got - ref).abs()
        print(f"             got={got.item()}, ref={ref.item()}, abs_diff={diff.item()}")
    return ok


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
) -> float:
    for _ in range(warmup):
        out = perf_func(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        out = perf_func(x)
    torch.cuda.synchronize()
    end = time.time()

    mean_ms = (end - start) * 1000.0 / iters
    mean_val = out.item() / iters
    print(f"{'out_' + tag:>25}: {mean_val:<15.8f}, time:{mean_ms:.8f}ms")
    return mean_ms



# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def run(check: bool = True):
    Ss = [1024, 2048, 4096]
    Ks = [1024, 2048, 4096]
    all_ok = True

    for S, K in [(s, k) for s in Ss for k in Ks]:
        print("-" * 80)
        print(" " * 40 + f"S={S}, K={K}")

        x = torch.randn((S, K)).cuda().half().contiguous()
        y = torch.zeros((1,), device=x.device, dtype=torch.float32)
        if check:
            all_ok &= check_correctness(lib.all_reduce_sum_f16x8_pack, x, "f16")
        run_benchmark(lib.all_reduce_sum_f16x8_pack, x, "f16")
        run_benchmark(torch.sum, x, "f16_th")
        print("-" * 80)

    if check:
        print(("\n[summary] ALL PASS" if all_ok else "\n[summary] SOME FAIL"))
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-check", action="store_true",
                        help="Skip correctness checks")
    args = parser.parse_args()
    ok = run(check=not args.no_check)
    exit(0 if ok else 1)
