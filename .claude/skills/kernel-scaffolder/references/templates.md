# Kernel scaffold templates

生成 my_/practice_ 文件时不确定结构，对照这里的模板。两种主要风格：**element-wise**（`sigmoid` / `relu` / `gelu` 这类一进一出 same-shape）和 **reduce**（`all_reduce_sum` 这类一进一出 scalar）。

## Table of contents
- [element-wise: my_$op.cu](#element-wise-my_opcu)
- [element-wise: practice_$op.cu](#element-wise-practice_opcu)
- [element-wise: my_$op.py](#element-wise-my_oppy)
- [element-wise: practice_$op.py](#element-wise-practice_oppy)
- [element-wise: my_$op.sh](#element-wise-my_opsh)
- [reduce: my_$op.cu](#reduce-my_opcu)
- [reduce: practice_$op.cu](#reduce-practice_opcu)
- [reduce: my_$op.py](#reduce-my_oppy)
- [reduce: practice_$op.py](#reduce-practice_oppy)
- [reduce: my_$op.sh](#reduce-my_opsh)

---

## element-wise: my_$op.cu

参考 `kernels/sigmoid/my_sigmoid.cu` / `kernels/relu/my_relu.cu`。骨架：

```cpp
#include <algorithm>
#include <cuda_fp16.h>     // 只在用到 half 时才 include
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

// 这些 cast 宏注释掉作为提示（让用户自己写出来）
// #define WARP_SIZE 32
// #define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
// #define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
// #define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
// #define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
// #define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// 数值常量保留（如果 op 需要）
#define MAX_EXP_F32  88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

// 可选：保留 / 添加 __device__ helper（不要让答案太显然）
// __device__ __forceinline__ float <op>(float x) { ... }

__global__ void <op>_f32_kernel(float *x, float *y, int N) {
  // TODO: implement scalar fp32 <op>
}

__global__ void <op>_f32x4_kernel(float *x, float *y, int N) {
  // TODO: implement fp32 vec4 <op> (FLOAT4 load/store)
}

__global__ void <op>_f16_kernel(half *x, half *y, int N) {
  // TODO: implement scalar fp16 <op>
}

__global__ void <op>_f16x2_kernel(half *x, half *y, int N) {
  // TODO: implement fp16 half2 <op> (HALF2 load/store)
}

__global__ void <op>_f16x8_kernel(half *x, half *y, int N) {
  // TODO: implement fp16x8 <op> via 4x half2 (HALF2 x 4)
}

__global__ void <op>_f16x8_pack_kernel(half *x, half *y, int N) {
  // TODO: implement fp16x8 pack <op> (LDST128BITS load/store)
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

// 这部分宏定义和参考 cu 一字不差，照搬过来
#define TORCH_BINDING_<OP>(packed_type, th_type, element_type, n_elements)     \
  void <op>_##packed_type(torch::Tensor x, torch::Tensor y) {                  \
    /* ... 和参考一致 ... */                                                    \
  }

TORCH_BINDING_<OP>(f32, torch::kFloat32, float, 1)
TORCH_BINDING_<OP>(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_<OP>(f16, torch::kHalf, half, 1)
TORCH_BINDING_<OP>(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_<OP>(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_<OP>(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(<op>_f32)
  TORCH_BINDING_COMMON_EXTENSION(<op>_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(<op>_f16)
  TORCH_BINDING_COMMON_EXTENSION(<op>_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(<op>_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(<op>_f16x8_pack)
}
```

---

## element-wise: practice_$op.cu

参考 `kernels/sigmoid/practice_sigmoid.cu` / `kernels/relu/practice_relu.cu`：

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

// ============================================================
// Practice kernels — best-performing variant per data type.
// Names are intentionally plain (no vector-width hints) so the
// exercise focuses on re-implementing the optimal kernel itself.
//   FP32  -> best: float4 vectorized (128-bit LD/ST)
//   FP16  -> best: 128-bit pack + <op-specific best>
// ============================================================

__global__ void <op>_f32_kernel(float *x, float *y, int N) {
  // TODO(practice): best FP32 <op> — 4-element vectorized
}

__global__ void <op>_f16_kernel(half *x, half *y, int N) {
  // TODO(practice): best FP16 <op> — 8-element 128-bit pack + <key intrinsic>
}

// ============================================================
// Torch bindings
// ============================================================
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    throw std::runtime_error("Tensor dtype mismatch, expected " #th_type);     \
  }

// n_elements: how many elements each thread processes in the fast path
#define TORCH_BINDING_<OP>(packed_type, th_type, element_type, n_elements)     \
  void <op>_##packed_type(torch::Tensor x, torch::Tensor y) {                  \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int ndim = x.dim();                                                  \
    const int S = x.size(0);                                                   \
    const int K = (ndim >= 2) ? x.size(1) : 1;                                 \
    const int N = x.numel();                                                   \
    dim3 block, grid;                                                          \
    if (ndim == 2 && (K / (n_elements)) <= 1024) {                             \
      block = dim3(K / (n_elements));                                          \
      grid  = dim3(S);                                                         \
    } else {                                                                   \
      block = dim3(256 / (n_elements));                                        \
      grid  = dim3((N + 256 - 1) / 256);                                       \
    }                                                                          \
    <op>_##packed_type##_kernel<<<grid, block>>>(                              \
        reinterpret_cast<element_type *>(x.data_ptr()),                        \
        reinterpret_cast<element_type *>(y.data_ptr()), N);                    \
  }

TORCH_BINDING_<OP>(f32, torch::kFloat32, float, 4)
TORCH_BINDING_<OP>(f16, torch::kHalf,    half,  8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(<op>_f32)
  TORCH_BINDING_COMMON_EXTENSION(<op>_f16)
}
```

`n_elements` 取最佳 kernel 的向量宽度：标量=1、x4=4、x8 / x8_pack=8。

---

## element-wise: my_$op.py

参考 `kernels/sigmoid/my_sigmoid.py` / `kernels/relu/my_relu.py`。这里给一个最小框架，按需扩展所有 dtype 路径：

```python
import os
import time
from typing import Optional
import argparse

import torch
from torch.utils.cpp_extension import load
from torch.cuda import nvtx

torch.set_grad_enabled(False)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_HERE, "build", "<op>_lib")
os.makedirs(_BUILD_DIR, exist_ok=True)

lib = load(
    name="<op>_lib",
    sources=[os.path.join(_HERE, "my_<op>.cu")],
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


def run_benchmark(perf_func, x, tag, out: Optional[torch.Tensor] = None,
                  warmup: int = 10, iters: int = 1000, show_all: bool = False):
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
    mean_time = (end - start) * 1000 / iters
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{'out_'+tag:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time


def run_profiling(perf_func, x, tag, out: Optional[torch.Tensor] = None, warmup: int = 10):
    if out is not None:
        out.fill_(0)
        for _ in range(warmup):
            perf_func(x, out)
    else:
        for _ in range(warmup):
            _ = perf_func(x)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("profiling")
    if out is not None:
        perf_func(x, out)
    else:
        _ = perf_func(x)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def check_correctness(perf_func, x, tag, out: Optional[torch.Tensor] = None,
                      atol: float = 1e-5, rtol: float = 1e-5) -> bool:
    ref = torch.<op>(x)  # 例：torch.sigmoid / torch.relu / torch.nn.functional.gelu
    if out is not None:
        out.fill_(0)
        perf_func(x, out)
        got = out
    else:
        got = perf_func(x)
    torch.cuda.synchronize()
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"[correctness] {tag}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        diff = (got.float() - ref.float()).abs()
        print(f"             max_abs_diff={diff.max().item()}, "
              f"mean_abs_diff={diff.mean().item()}")
    return ok


def run_benchmark_for_all_test(check: bool = True):
    Ss = [1024, 2048, 4096]
    Ks = [1024, 2048, 4096]
    for S, K in [(s, k) for s in Ss for k in Ks]:
        print("-" * 85)
        print(" " * 40 + f"S={S}, K={K}")
        x = torch.randn((S, K)).cuda().float().contiguous()
        y = torch.zeros_like(x).cuda().float().contiguous()
        if check:
            check_correctness(lib.<op>_f32, x, "f32", y)
            check_correctness(lib.<op>_f32x4, x, "f32x4", y)
            check_correctness(torch.<op>, x, "f32_th")
        run_benchmark(lib.<op>_f32, x, "f32", y)
        run_benchmark(lib.<op>_f32x4, x, "f32x4", y)
        run_benchmark(torch.<op>, x, "f32_th")

        print("-" * 85)
        x_f16 = x.half().contiguous()
        y_f16 = y.half().contiguous()
        if check:
            check_correctness(lib.<op>_f16, x_f16, "f16", y_f16, atol=1e-3, rtol=1e-3)
            check_correctness(lib.<op>_f16x2, x_f16, "f16x2", y_f16, atol=1e-3, rtol=1e-3)
            check_correctness(lib.<op>_f16x8, x_f16, "f16x8", y_f16, atol=1e-3, rtol=1e-3)
            check_correctness(lib.<op>_f16x8_pack, x_f16, "f16x8pack", y_f16, atol=1e-3, rtol=1e-3)
            check_correctness(torch.<op>, x_f16, "f16_th", atol=1e-3, rtol=1e-3)
        run_benchmark(lib.<op>_f16, x_f16, "f16", y_f16)
        run_benchmark(lib.<op>_f16x2, x_f16, "f16x2", y_f16)
        run_benchmark(lib.<op>_f16x8, x_f16, "f16x8", y_f16)
        run_benchmark(lib.<op>_f16x8_pack, x_f16, "f16x8pack", y_f16)
        run_benchmark(torch.<op>, x_f16, "f16_th")
        print("-" * 85)


def run_profiling_for_test(kernel_name, dtype, S=4096, K=4096):
    x = torch.randn((S, K)).cuda().float().contiguous()
    y = torch.zeros_like(x).cuda().float().contiguous()
    x_half = x.half().contiguous()
    y_half = y.half().contiguous()

    if dtype == torch.float32:
        if kernel_name == "<op>_f32":     run_profiling(lib.<op>_f32, x, "profiling", y)
        elif kernel_name == "<op>_f32x4": run_profiling(lib.<op>_f32x4, x, "profiling", y)
        elif kernel_name == "<op>_th":    run_profiling(torch.<op>, x, "profiling")
        else: raise ValueError(f"Unsupported kernel name: {kernel_name}")
    elif dtype == torch.float16:
        if kernel_name == "<op>_f16":          run_profiling(lib.<op>_f16, x_half, "profiling", y_half)
        elif kernel_name == "<op>_f16x2":      run_profiling(lib.<op>_f16x2, x_half, "profiling", y_half)
        elif kernel_name == "<op>_f16x8":      run_profiling(lib.<op>_f16x8, x_half, "profiling", y_half)
        elif kernel_name == "<op>_f16x8_pack": run_profiling(lib.<op>_f16x8_pack, x_half, "profiling", y_half)
        elif kernel_name == "<op>_th":         run_profiling(torch.<op>, x_half, "profiling")
        else: raise ValueError(f"Unsupported kernel name: {kernel_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--profiling", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--S", type=int, default=4096)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark_for_all_test(check=not args.no_check)
    if args.profiling is not None:
        dtype = torch.float32 if args.dtype == "float32" else torch.float16
        run_profiling_for_test(args.profiling, dtype, S=args.S, K=args.K)
```

---

## element-wise: practice_$op.py

```python
# -*- coding: utf-8 -*-
"""
practice_<op>.py - Practice-use <Op> benchmark
==============================================
After learning the <op> kernel, use this for repeated practice:
  - Only the best-performing kernel per dtype (FP32: f32x4, FP16: f16x8_pack)
  - Two features only: check_correctness and benchmark
  - Plain kernel names (<op>_f32 / <op>_f16), no optimization hints
"""

import argparse
import os
import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_HERE, "build", "practice_<op>_lib")
os.makedirs(_BUILD_DIR, exist_ok=True)

lib = load(
    name="practice_<op>_lib",
    sources=[os.path.join(_HERE, "practice_<op>.cu")],
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


def check_correctness(perf_func, x, tag, out: Optional[torch.Tensor] = None,
                      atol: float = 1e-5, rtol: float = 1e-5) -> bool:
    ref = torch.<op>(x)
    if out is not None:
        out.fill_(0)
        perf_func(x, out)
        got = out
    else:
        got = perf_func(x)
    torch.cuda.synchronize()
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"[correctness] {tag}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        diff = (got.float() - ref.float()).abs()
        print(f"             max_abs_diff={diff.max().item()}, "
              f"mean_abs_diff={diff.mean().item()}")
    return ok


def run_benchmark(perf_func, x, tag, out: Optional[torch.Tensor] = None,
                  warmup: int = 10, iters: int = 1000) -> float:
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
    print(f"{'out_'+tag:>18}: {out_val}, time:{mean_ms:.8f}ms")
    return mean_ms


def run(check: bool = True):
    Ss = [1024, 2048, 4096]
    Ks = [1024, 2048, 4096]
    all_ok = True

    for S, K in [(s, k) for s in Ss for k in Ks]:
        print("-" * 85)
        print(" " * 40 + f"S={S}, K={K}")

        x = torch.randn((S, K)).cuda().float().contiguous()
        y = torch.zeros_like(x)
        if check:
            all_ok &= check_correctness(lib.<op>_f32, x, "f32", y)
            all_ok &= check_correctness(torch.<op>, x, "f32_th")
        run_benchmark(lib.<op>_f32, x, "f32", y)
        run_benchmark(partial(torch.<op>, out=y), x, "f32_th", y)

        x_f16 = x.half().contiguous()
        y_f16 = y.half().contiguous()
        if check:
            all_ok &= check_correctness(lib.<op>_f16, x_f16, "f16", y_f16, atol=1e-3, rtol=1e-3)
            all_ok &= check_correctness(torch.<op>, x_f16, "f16_th", atol=1e-3, rtol=1e-3)
        run_benchmark(lib.<op>_f16, x_f16, "f16", y_f16)
        run_benchmark(partial(torch.<op>, out=y_f16), x_f16, "f16_th", y_f16)
        print("-" * 85)

    if check:
        print("\n[summary] ALL PASS" if all_ok else "\n[summary] SOME FAIL")
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()
    ok = run(check=not args.no_check)
    exit(0 if ok else 1)
```

---

## element-wise: my_$op.sh

```bash
#!/usr/bin/env bash
set -e

name=${1:-<op>_f16}
dtype=${2:-float16}

ncu --nvtx \
  --nvtx-include "profiling/" \
  --set full \
  --import-source yes \
  -o "$name" \
  -- python3 my_<op>.py --profiling "$name" --dtype "$dtype"
```

---

## reduce: my_$op.cu

参考 `kernels/reduce/my_all_reduce.cu`。reduce 风格的 kernel 是 templated `<int NUM_THREADS>`，TORCH_BINDING 用 dispatch macro。骨架：

```cpp
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32
#define WARP_MASK (WARP_SIZE - 1)
#define WARP_SHIFT 5
// LDST128BITS 注释掉作为提示
// #define LDST128BITS(val) (reinterpret_cast<float4*>(&(val)))[0]

// warp-level helper 保留作为基础结构（不掏空，否则 TODO 注释里也讲不清楚）
template <const int kwarp_size = WARP_SIZE>
__device__ __forceinline__ float warp_sum_f32(float sum_f32) {
#pragma unroll
  for (int mask = kwarp_size / 2; mask >= 1; mask >>= 1) {
    sum_f32 += __shfl_xor_sync(0xffffffff, sum_f32, mask);
  }
  return sum_f32;
}

template <const int NUM_THREADS = 1024>
__global__ void <op>_f16x8_pack_kernel(half *x, float *y, int N) {
  // TODO: implement fp16x8 pack <op> with shared-mem warp reduction + atomicAdd
  //   1. load 128-bit (8x half) per thread, accumulate into float
  //   2. warp_sum_f32, write lane-0 to shmem
  //   3. warp 0 reduces shmem with warp_sum_f32<NUM_WARPS>
  //   4. tid==0 atomicAdd into y
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

// dispatch macros + TORCH_BINDING_REDUCE 完全照搬参考实现
// （拷贝过来即可，不需要修改）
```

完整的 dispatch / binding macros 看 `kernels/reduce/my_all_reduce.cu` 第 73-154 行。

---

## reduce: practice_$op.cu

只保留最佳 kernel（通常是 `<op>_f16x8_pack_kernel`），名字保持原样（reduce 这条链已经是简洁名了），函数体掏空：

```cpp
template <const int NUM_THREADS = 1024>
__global__ void <op>_f16_kernel(half *x, float *y, int N) {
  // TODO(practice): best FP16 <op> — 128-bit pack load + warp reduction + atomicAdd
}
```

注：reduce 链上原本只有 `_f16x8_pack`，practice 版的命名可以选择
- 保持 `<op>_f16x8_pack`（如果你觉得简洁名歧义太大），或者
- 简化成 `<op>_f16`（更符合 CLAUDE.md 描述的"去掉优化标签"）

参考 `kernels/reduce/practice_all_reduce.cu` 的写法。

---

## reduce: my_$op.py

参考 `kernels/reduce/my_all_reduce.py`。和 element-wise 版差异：
- **彻底删除 `out:` 参数**——`run_benchmark`、`run_profiling`、`check_correctness` 三个函数的签名里都不要保留 `out: Optional[torch.Tensor] = None`。reduce kernel 自己内部 `atomicAdd` 到一个新分配的 scalar tensor，调用形式是 `out = perf_func(x)`（或 `(a, b)`），不是 `perf_func(x, out)`。要么完全删掉 `out` 参数，要么至少把所有 `if out is not None:` 分支去掉。
- `check_correctness` 的 ref 是 `torch.sum(x, dtype=torch.float32)` 之类（dot-product 是 `(a.float() * b.float()).sum()` 或 `torch.dot`）；
- argparse 是否需要 `--dtype` 看具体 op：单 dtype（all_reduce f16-only）就去掉；多 dtype（dot-product 有 f32/f16）就保留。
- `run_profiling` 同样不传 `out`，调用形式是 `run_profiling(perf_func, x, "profiling")` 而非 `(perf_func, x, "profiling", out)`。

---

## reduce: practice_$op.py

参考 `kernels/reduce/practice_all_reduce.py`。比 element-wise practice 还要简单：
- 一个 dtype × 一个 kernel，调用接口是 `lib.<best_kernel_name>(x)` 返回 scalar tensor。

---

## reduce: my_$op.sh

```bash
#!/usr/bin/env bash
set -e

name=${1:-<best_kernel_name>}

ncu --nvtx \
  --nvtx-include "profiling/" \
  -k regex:"$name"_kernel \
  --set full \
  --import-source yes \
  -f \
  -o "$name" \
  -- python3 my_<op>.py --profiling "$name"
```

注意 reduce 版的 ncu 命令多了 `-k regex:..._kernel`（限定只采样目标 kernel）和 `-f`（覆盖已有报告）。
