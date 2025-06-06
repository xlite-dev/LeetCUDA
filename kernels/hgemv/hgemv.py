import os
import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

CUTLASS_REPO_PATH = os.environ.get(
    "CUTLASS_REPO_PATH", os.path.expanduser("../../third-party/cutlass")
)

# Load the CUDA kernel as a python module
lib = load(
    name="hgemv_lib",
    sources=["hgemv.cu", "hgemv_cute.cu"],
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
    extra_include_paths=[os.path.join(CUTLASS_REPO_PATH, "include")],
)


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 200,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    if out is not None:
        for i in range(warmup):
            perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b)

    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(a, b, out)
    else:
        for i in range(iters):
            out = perf_func(a, b)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>13}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out.clone(), mean_time


print("-" * 80)
M, N, K = 1024, 1, 128
a = torch.randn((M, K)).cuda().half().contiguous()
b = torch.randn((K, N)).cuda().half().contiguous()
c = torch.randn((M, N)).cuda().half().contiguous()
run_benchmark(lib.hgemv_k32_f16, a, b, "k32f16", c)
run_benchmark(lib.hgemv_k128_f16x4, a, b, "k128f16x4", c)
run_benchmark(lib.hgemv_f16_cute, a, b, "hgemv_f16_cute", c)
run_benchmark(lib.hgemv_f16x8_cute, a, b, "hgemv_f16x8_cute", c)
run_benchmark(lib.hgemv_tensor_core_cute, a, b, "hgemv_tensor_core_cute", c)
run_benchmark(partial(torch.matmul, out=c), a, b, "f16_th")
print("-" * 80)

M, N, K = 1024, 1, 16
a = torch.randn((M, K)).cuda().half().contiguous()
b = torch.randn((K, N)).cuda().half().contiguous()
c = torch.randn((M, N)).cuda().half().contiguous()
run_benchmark(lib.hgemv_k16_f16, a, b, "k16f16", c)
run_benchmark(lib.hgemv_f16_cute, a, b, "hgemv_f16_cute", c)
run_benchmark(lib.hgemv_f16x8_cute, a, b, "hgemv_f16x8_cute", c)
run_benchmark(lib.hgemv_tensor_core_cute, a, b, "hgemv_tensor_core_cute", c)
run_benchmark(partial(torch.matmul, out=c), a, b, "f16_th")
print("-" * 80)
