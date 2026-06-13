#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

// #define WARP_SIZE 32
// #define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
// #define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
// #define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
// #define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
// #define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// Numerical clamps to avoid overflow/underflow in expf / hexp.
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

__device__ __forceinline__ float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ half sigmoid(half x) {
  const half one = __float2half(1.0f);
  return one / (one + hexp(-x));
}

__device__ __forceinline__ half2 sigmoid(half2 x) {
  const half2 one = {__float2half(1.0f), __float2half(1.0f)};
  return one / (one + h2exp(-x));
}

__global__ void sigmoid_f32_kernel(float *x, float *y, int N) {
  // TODO: implement scalar fp32 sigmoid
  //   y[i] = 1 / (1 + exp(-x[i]))
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    y[idx] = sigmoid(x[idx]);
  }
}

__global__ void sigmoid_f32x4_kernel(float *x, float *y, int N) {
  // TODO: implement fp32 vec4 sigmoid (FLOAT4 load/store)
#define FLOAT4(val) (reinterpret_cast<float4*>(&(val)))[0]
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    reg_x.x = sigmoid(reg_x.x);
    reg_x.y = sigmoid(reg_x.y);
    reg_x.z = sigmoid(reg_x.z);
    reg_x.w = sigmoid(reg_x.w);
    FLOAT4(y[idx]) = reg_x;
  }
}

__global__ void sigmoid_f16_kernel(half *x, half *y, int N) {
  // TODO: implement scalar fp16 sigmoid (hexp)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    y[idx] = sigmoid(x[idx]);
  }
}

__global__ void sigmoid_f16x2_kernel(half *x, half *y, int N) {
  // TODO: implement fp16 half2 sigmoid (HALF2 load/store)
#define HALF2(val) (reinterpret_cast<half2*> (&(val)))[0]
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (idx < N) {
    half2 reg_x = HALF2(x[idx]);
    HALF2(y[idx]) = sigmoid(reg_x);
  }
}

__global__ void sigmoid_f16x8_kernel(half *x, half *y, int N) {
  // TODO: implement fp16x8 sigmoid via 4x half2 (HALF2 x 4)
#define HALF2(val) (reinterpret_cast<half2*>(&(val)))[0]
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (idx < N) {
    half2 reg_x_0 = HALF2(x[idx + 0]);
    half2 reg_x_1 = HALF2(x[idx + 2]);
    half2 reg_x_2 = HALF2(x[idx + 4]);
    half2 reg_x_3 = HALF2(x[idx + 6]);

    reg_x_0 = sigmoid(reg_x_0);
    reg_x_1 = sigmoid(reg_x_1);
    reg_x_2 = sigmoid(reg_x_2);
    reg_x_3 = sigmoid(reg_x_3);

    HALF2(y[idx + 0]) = reg_x_0;
    HALF2(y[idx + 2]) = reg_x_1;
    HALF2(y[idx + 4]) = reg_x_2;
    HALF2(y[idx + 6]) = reg_x_3;
  }
}

__global__ void sigmoid_f16x8_pack_kernel(half *x, half *y, int N) {
  // TODO: implement fp16x8 pack sigmoid (LDST128BITS load/store)
#define LDST128BITS(val) (reinterpret_cast<float4*>(&(val)))[0]
#define HALF2(val) (reinterpret_cast<half2*>(&(val)))[0]
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (idx < N) {
    half x_pack[8];
    LDST128BITS(x_pack[0]) = LDST128BITS(x[idx]);
#pragma unroll
    for (int i = 0; i < 8; i+=2) {
      HALF2(x_pack[i]) = sigmoid(HALF2(x_pack[i]));
    }
    LDST128BITS(y[idx]) = LDST128BITS(x_pack[0]);
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

#define TORCH_BINDING_SIGMOID(packed_type, th_type, element_type, n_elements)  \
  void sigmoid_##packed_type(torch::Tensor x, torch::Tensor y) {               \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int ndim = x.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= x.size(i);                                                        \
      }                                                                        \
      dim3 block(256 / (n_elements));                                          \
      dim3 grid((N + 256 - 1) / 256);                                          \
      sigmoid_##packed_type##_kernel<<<grid, block>>>(                         \
          reinterpret_cast<element_type *>(x.data_ptr()),                      \
          reinterpret_cast<element_type *>(y.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        sigmoid_##packed_type##_kernel<<<grid, block>>>(                       \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        sigmoid_##packed_type##_kernel<<<grid, block>>>(                       \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

TORCH_BINDING_SIGMOID(f32, torch::kFloat32, float, 1)
TORCH_BINDING_SIGMOID(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_SIGMOID(f16, torch::kHalf, half, 1)
TORCH_BINDING_SIGMOID(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_SIGMOID(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_SIGMOID(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f32)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x8_pack)
}
