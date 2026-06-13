import os
import time
from functools import partial
from typing import Optional
import argparse

import torch
from torch.utils.cpp_extension import load
from torch.cuda import nvtx

torch.set_grad_enabled(False)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_HERE, "build", "sigmoid_lib")
os.makedirs(_BUILD_DIR, exist_ok=True)

lib = load(
    name="sigmoid_lib",
    sources=[os.path.join(_HERE, "my_sigmoid.cu")],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-lineinfo",
    ],
    extra_cflags=["-std=c++17"],
    build_directory=_BUILD_DIR,
)

def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    # warmup
    if out is not None:
        for i in range(warmup):
            perf_func(x, out)
    else:
        for i in range(warmup):
            _ = perf_func(x)
    torch.cuda.synchronize()

    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(x, out)
    else:
        for i in range(iters):
            out = perf_func(x)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time

def run_profiling(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
):
    if out is not None:
        out.fill_(0)
    # warmup
    if out is not None:
        for i in range(warmup):
            perf_func(x, out)
    else:
        for i in range(warmup):
            _ = perf_func(x)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("profiling")
    if out is not None:
        perf_func(x, out)
    else:
        _ = perf_func(x)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

def check_correctness(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> bool:
    """Verify that perf_func(x[, out]) matches torch.sigmoid(x)."""
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


def run_benchmark_for_all_test(check: bool = True):
    Ss = [1024, 2048, 4096]
    Ks = [1024, 2048, 4096]
    SKs = [(S, K) for S in Ss for K in Ks]

    for S, K in SKs:
        print("-" * 85)
        print(" " * 40 + f"S={S}, K={K}")
        x = torch.randn((S, K)).cuda().float().contiguous()
        y = torch.zeros_like(x).cuda().float().contiguous()
        if check:
            check_correctness(lib.sigmoid_f32, x, "f32", y)
            check_correctness(lib.sigmoid_f32x4, x, "f32x4", y)
            check_correctness(torch.sigmoid, x, "f32_th")
        run_benchmark(lib.sigmoid_f32, x, "f32", y)
        run_benchmark(lib.sigmoid_f32x4, x, "f32x4", y)
        run_benchmark(torch.sigmoid, x, "f32_th")

        print("-" * 85)
        x_f16 = x.half().contiguous()
        y_f16 = y.half().contiguous()
        if check:
            check_correctness(lib.sigmoid_f16, x_f16, "f16", y_f16, atol=1e-3, rtol=1e-3)
            check_correctness(lib.sigmoid_f16x2, x_f16, "f16x2", y_f16, atol=1e-3, rtol=1e-3)
            check_correctness(lib.sigmoid_f16x8, x_f16, "f16x8", y_f16, atol=1e-3, rtol=1e-3)
            check_correctness(lib.sigmoid_f16x8_pack, x_f16, "f16x8pack", y_f16, atol=1e-3, rtol=1e-3)
            check_correctness(torch.sigmoid, x_f16, "f16_th", atol=1e-3, rtol=1e-3)
        run_benchmark(lib.sigmoid_f16, x_f16, "f16", y_f16)
        run_benchmark(lib.sigmoid_f16x2, x_f16, "f16x2", y_f16)
        run_benchmark(lib.sigmoid_f16x8, x_f16, "f16x8", y_f16)
        run_benchmark(lib.sigmoid_f16x8_pack, x_f16, "f16x8pack", y_f16)
        run_benchmark(torch.sigmoid, x_f16, "f16_th")
        print("-" * 85)


def run_profiling_for_test(
    kernel_name: str,
    dtype: torch.dtype,
    S: int = 4096,
    K: int = 4096,
):
    x = torch.randn((S, K)).cuda().float().contiguous()
    y = torch.zeros_like(x).cuda().float().contiguous()
    x_half = x.half().contiguous()
    y_half = y.half().contiguous()

    if dtype == torch.float32:
        if kernel_name == "sigmoid_f32":
            run_profiling(lib.sigmoid_f32, x, "profiling", y)
        elif kernel_name == "sigmoid_f32x4":
            run_profiling(lib.sigmoid_f32x4, x, "profiling", y)
        elif kernel_name == "sigmoid_th":
            run_profiling(torch.sigmoid, x, "profiling")
        else:
            raise ValueError(f"Unsupported kernel name: {kernel_name}")
    elif dtype == torch.float16:
        if kernel_name == "sigmoid_f16":
            run_profiling(lib.sigmoid_f16, x_half, "profiling", y_half)
        elif kernel_name == "sigmoid_f16x2":
            run_profiling(lib.sigmoid_f16x2, x_half, "profiling", y_half)
        elif kernel_name == "sigmoid_f16x8":
            run_profiling(lib.sigmoid_f16x8, x_half, "profiling", y_half)
        elif kernel_name == "sigmoid_f16x8_pack":
            run_profiling(lib.sigmoid_f16x8_pack, x_half, "profiling", y_half)
        elif kernel_name == "sigmoid_th":
            run_profiling(torch.sigmoid, x_half, "profiling")
        else:
            raise ValueError(f"Unsupported kernel name: {kernel_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--profiling", type=str, default=None,
                        help="Run profiling for the given kernel name")
    parser.add_argument("--dtype", type=str, default="float32",
                        help="Data type for profiling (float32 or float16)")
    parser.add_argument("--S", type=int, default=4096, help="Row size for profiling shape")
    parser.add_argument("--K", type=int, default=4096, help="Column size for profiling shape")
    parser.add_argument("--no-check", action="store_true",
                        help="Skip correctness checks before benchmarking")

    args = parser.parse_args()
    if args.benchmark:
        run_benchmark_for_all_test(check=not args.no_check)
    if args.profiling is not None:
        dtype = torch.float32 if args.dtype == "float32" else torch.float16
        run_profiling_for_test(args.profiling, dtype, S=args.S, K=args.K)
