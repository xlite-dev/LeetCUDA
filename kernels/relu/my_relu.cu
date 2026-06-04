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

__global__ void relu_f32_kernel(float *x, float *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    y[idx] = fmaxf(0.0f, x[idx]);
  }
}

__global__ void relu_f32x4_kernel(float *x, float *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value)))[0]
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = fmaxf(0.0f, reg_x.x);
    reg_y.y = fmaxf(0.0f, reg_x.y);
    reg_y.z = fmaxf(0.0f, reg_x.z);
    reg_y.w = fmaxf(0.0f, reg_x.w);

    FLOAT4(y[idx]) = reg_y;
  }
}

__global__ void relu_f16_kernel(half *x, half *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    y[idx] = __hmax(__float2half(0.0f), x[idx]);
  }
}

__global__ void relu_f16x2_kernel(half *x, half *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (idx < N) {
#define HALF2(value) (reinterpret_cast<half2 *>(&(value)))[0]
    half2 reg_x = HALF2(x[idx]);
    half2 reg_y;
    const half2 reg_z = {__float2half(0.0f), __float2half(0.0f)};
    reg_y = __hmax2(reg_z, reg_x);

    HALF2(y[idx]) = reg_y;
  }
}

__global__ void relu_f16x8_kernel(half *x, half *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
#define HALF2(value) (reinterpret_cast<half2 *>(&(value)))[0]
  if (idx < N) {
    half2 reg_x_0 = HALF2(x[idx + 0]);
    half2 reg_x_1 = HALF2(x[idx + 2]);
    half2 reg_x_2 = HALF2(x[idx + 4]);
    half2 reg_x_3 = HALF2(x[idx + 6]);
    half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
    half2 reg_z = {__float2half(0.0), __float2half(0.0)};

    reg_y_0 = __hmax2(reg_z, reg_x_0);
    reg_y_1 = __hmax2(reg_z, reg_x_1);
    reg_y_2 = __hmax2(reg_z, reg_x_2);
    reg_y_3 = __hmax2(reg_z, reg_x_3);

    HALF2(y[idx + 0]) = reg_y_0;
    HALF2(y[idx + 2]) = reg_y_1;
    HALF2(y[idx + 4]) = reg_y_2;
    HALF2(y[idx + 6]) = reg_y_3;
  }
}

__global__ void relu_f16x8_pack_kernel(half *x, half *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
#define HALF2(value) (reinterpret_cast<half2 *>(&(value)))[0]
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value)))[0]
  if (idx < N) {
    half pack_x[8], pack_y[8];
    half2 reg_z = {__float2half(0.0f), __float2half(0.0f)};
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      HALF2(pack_y[i]) = __hmax2(reg_z, HALF2(pack_x[i]));
    }

    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
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

#define TORCH_BINDING_RELU(packed_type, th_type, element_type, n_elements)     \
  void relu_##packed_type(torch::Tensor x, torch::Tensor y) {                  \
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
      relu_##packed_type##_kernel<<<grid, block>>>(                            \
          reinterpret_cast<element_type *>(x.data_ptr()),                      \
          reinterpret_cast<element_type *>(y.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        relu_##packed_type##_kernel<<<grid, block>>>(                          \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        relu_##packed_type##_kernel<<<grid, block>>>(                          \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

TORCH_BINDING_RELU(f32, torch::kFloat32, float, 1)
TORCH_BINDING_RELU(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_RELU(f16, torch::kHalf, half, 1)
TORCH_BINDING_RELU(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_RELU(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_RELU(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(relu_f32)
  TORCH_BINDING_COMMON_EXTENSION(relu_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16x8_pack)
}
