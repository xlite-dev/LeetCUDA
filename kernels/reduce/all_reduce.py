import time
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="all_reduce_lib",
    sources=["all_reduce.cu"],
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
    verbose=True,
)


def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
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
    print(f"{'out_' + tag:>25}: {out.item():<15.8f}, time:{mean_time:.8f}ms")
    return out, mean_time


Ss = [1024, 2048, 4096]
Ks = [1024, 2048, 4096]
SKs = [(S, K) for S in Ss for K in Ks]

for S, K in SKs:
    print("-" * 80)
    print(" " * 40 + f"S={S}, K={K}")
    x = torch.randn((S, K)).cuda().half().contiguous()
    run_benchmark(lib.all_reduce_sum_f16x8_pack, x, "f16x8_pack")
    run_benchmark(torch.sum, x, "f16_th")
    print("-" * 80)
