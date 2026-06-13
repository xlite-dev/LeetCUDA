# -*- coding: utf-8 -*-
"""
practice_sigmoid.py - Practice-use Sigmoid benchmark
=====================================================
After learning the sigmoid kernel, use this for repeated practice:
  - Only the best-performing kernel per dtype (FP32: f32x4, FP16: f16x8_pack)
  - Two features only: check_correctness and benchmark
  - Plain kernel names (sigmoid_f32 / sigmoid_f16), no optimization hints
"""

import argparse
import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

lib = load(
    name="practice_sigmoid_lib",
    sources=["practice_sigmoid.cu"],
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
)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
def check_correctness(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> bool:
    ref = torch.sigmoid(x)
    if out is not None:
        out.fill_(0)
        perf_func(x, out)
        got = out
    else:
        got = perf_func(x)
    torch.cuda.synchronize()
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"[correctness] {tag}: {status}")
    if not ok:
        diff = (got.float() - ref.float()).abs()
        print(f"             max_abs_diff={diff.max().item()}, "
              f"mean_abs_diff={diff.mean().item()}")
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
    if out is not None:
        out.fill_(0)
        for _ in range(warmup):
            perf_func(x, out)
    else:
        for _ in range(warmup):
            _ = perf_func(x)
    torch.cuda.synchronize()

    start = time.time()
    if out is not None:
        for _ in range(iters):
            perf_func(x, out)
    else:
        for _ in range(iters):
            out = perf_func(x)
    torch.cuda.synchronize()
    end = time.time()

    mean_ms = (end - start) * 1000.0 / iters
    out_val = out.flatten().detach().cpu().tolist()[:2]
    out_val = [f"{round(v, 8):<12}" for v in out_val]
    print(f"{'out_' + tag:>18}: {out_val}, time:{mean_ms:.8f}ms")
    return mean_ms


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def run(check: bool = True):
    Ss = [1024, 2048, 4096]
    Ks = [1024, 2048, 4096]
    all_ok = True

    for S, K in [(s, k) for s in Ss for k in Ks]:
        print("-" * 85)
        print(" " * 40 + f"S={S}, K={K}")

        # --- FP32 ---
        x = torch.randn((S, K)).cuda().float().contiguous()
        y = torch.zeros_like(x)
        if check:
            all_ok &= check_correctness(lib.sigmoid_f32, x, "f32", y)
            all_ok &= check_correctness(torch.sigmoid, x, "f32_th")
        run_benchmark(lib.sigmoid_f32, x, "f32", y)
        run_benchmark(partial(torch.sigmoid, out=y), x, "f32_th", y)

        # --- FP16 ---
        x_f16 = x.half().contiguous()
        y_f16 = y.half().contiguous()
        if check:
            all_ok &= check_correctness(
                lib.sigmoid_f16, x_f16, "f16", y_f16, atol=1e-3, rtol=1e-3
            )
            all_ok &= check_correctness(
                torch.sigmoid, x_f16, "f16_th", atol=1e-3, rtol=1e-3
            )
        run_benchmark(lib.sigmoid_f16, x_f16, "f16", y_f16)
        run_benchmark(partial(torch.sigmoid, out=y_f16), x_f16, "f16_th", y_f16)
        print("-" * 85)

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
