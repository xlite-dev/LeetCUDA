import argparse
import os
import time
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_HERE, "build", "my_all_reduce_lib")
os.makedirs(_BUILD_DIR, exist_ok=True)

lib = load(
    name="my_all_reduce_lib",
    sources=[os.path.join(_HERE, "my_all_reduce.cu")],
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
):
    for _ in range(warmup):
        out = perf_func(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        out = perf_func(x)
    torch.cuda.synchronize()
    end = time.time()

    mean_time = (end - start) * 1000 / iters
    mean_val = out.item() / iters
    print(f"{'out_' + tag:>25}: {mean_val:<15.8f}, time:{mean_time:.8f}ms")
    return out, mean_time


def run_profiling(
    perf_func: callable,
    x: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
):
    for _ in range(warmup):
        out = perf_func(x)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("profiling")
    out = perf_func(x)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


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

def run_benchmark_for_all_test(check: bool = True):
    Ss = [1024, 2048, 4096]
    Ks = [1024, 2048, 4096]
    SKs = [(S, K) for S in Ss for K in Ks]
    all_ok = True

    for S, K in SKs:
        print("-" * 80)
        print(" " * 40 + f"S={S}, K={K}")
        x = torch.randn((S, K)).cuda().half().contiguous()

        if check:
            all_ok &= check_correctness(lib.all_reduce_sum_f16x8_pack, x, "f16x8_pack")

        run_benchmark(lib.all_reduce_sum_f16x8_pack, x, "f16x8_pack")
        run_benchmark(torch.sum, x, "f16_th")
        print("-" * 80)

    if check:
        print(("\n[summary] ALL PASS" if all_ok else "\n[summary] SOME FAIL"))
    return all_ok


def run_profiling_for_test(kernel_name: str, S: int = 4096, K: int = 4096):
    x = torch.randn((S, K)).cuda().half().contiguous()
    y = torch.zeros((1,), device=x.device, dtype=torch.float32)

    if kernel_name == "all_reduce_sum_f16x8_pack":
        run_profiling(lib.all_reduce_sum_f16x8_pack, x, y)
    else:
        raise ValueError(f"Unsupported kernel name: {kernel_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--profiling", type=str, default=None,
                        help="Run profiling for the given kernel name")
    parser.add_argument("--S", type=int, default=4096, help="Row size for profiling shape")
    parser.add_argument("--K", type=int, default=4096, help="Column size for profiling shape")
    parser.add_argument("--no-check", action="store_true",
                        help="Skip correctness checks before benchmarking")

    args = parser.parse_args()
    if args.benchmark:
        ok = run_benchmark_for_all_test(check=not args.no_check)
        if not ok:
            exit(1)
    if args.profiling is not None:
        run_profiling_for_test(args.profiling, S=args.S, K=args.K)
