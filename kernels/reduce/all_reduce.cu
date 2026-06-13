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
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// ============================================================
// FP16 block all-reduce sum (best impl: 128-bit pack + warp reduce)
//   x: any shape, treated as contiguous N elements
//   y: one FP32 scalar, accumulated by atomicAdd from each block
// ============================================================

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// 128-bit pack load, each thread reduces eight FP16 elements
template <const int NUM_THREADS = 256 / 8>
__global__ void all_reduce_sum_f16x8_pack_kernel(half *x, float *y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 8;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  float sum = 0.0f;
  if (idx + 7 < N) {
    half pack_x[8];
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      sum += __half2float(pack_x[i]);
    }
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      sum += ((idx + i) < N) ? __half2float(x[idx + i]) : 0.0f;
    }
  }

  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;

  sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
  if (lane == 0) {
    reduce_smem[warp] = sum;
  }
  __syncthreads();

  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) {
    sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  }
  if (tid == 0) {
    atomicAdd(y, sum);
  }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define LANUCH_REDUCE_KERNEL(NT, func_name)                                    \
  func_name##_kernel<(NT)><<<grid, block>>>(                                   \
      reinterpret_cast<half *>(x.data_ptr()),                                  \
      reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_REDUCE_KERNEL(K, func_name, n_elements)                       \
  const int NT = (K) / (n_elements);                                           \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (NT) {                                                                \
  case 32:                                                                     \
    LANUCH_REDUCE_KERNEL(32, func_name)                                        \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_REDUCE_KERNEL(64, func_name)                                        \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_REDUCE_KERNEL(128, func_name)                                       \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_REDUCE_KERNEL(256, func_name)                                       \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_REDUCE_KERNEL(512, func_name)                                       \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_REDUCE_KERNEL(1024, func_name)                                      \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error(                                                  \
        "only support (K)/(n_elements): 32/64/128/256/512/1024");              \
    break;                                                                     \
  }

#define TORCH_BINDING_REDUCE(func_name, n_elements)                            \
  torch::Tensor func_name(torch::Tensor x) {                                   \
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)                                  \
    auto options =                                                             \
        torch::TensorOptions().dtype(torch::kFloat32).device(x.device());       \
    auto y = torch::zeros({1}, options);                                       \
    const int ndim = x.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= x.size(i);                                                        \
      }                                                                        \
      dim3 block(1024 / (n_elements));                                         \
      dim3 grid((N + 1024 - 1) / 1024);                                        \
      func_name##_kernel<1024 / (n_elements)>                                  \
          <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),            \
                            reinterpret_cast<float *>(y.data_ptr()), N);       \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        DISPATCH_REDUCE_KERNEL(K, func_name, n_elements)                       \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(1024 / (n_elements));                                       \
        dim3 grid((N + 1024 - 1) / 1024);                                      \
        func_name##_kernel<1024 / (n_elements)>                                \
            <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),          \
                              reinterpret_cast<float *>(y.data_ptr()), N);     \
      }                                                                        \
    }                                                                          \
    return y;                                                                  \
  }

TORCH_BINDING_REDUCE(all_reduce_sum_f16x8_pack, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(all_reduce_sum_f16x8_pack)
}
