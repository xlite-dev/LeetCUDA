#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/types.h>
#include <torch/extension.h>

#define WARP_SIZE 256
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

// -------------------------------------- FP32 --------------------------------------
// col2row means read x[row][col] and write y[col][row]
// row2col means read x[col][row] and write y[row][col]
__global__ void mat_transpose_f32_col2row_kernel(float *x, float *y, const int row, const int col)
{
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_row = global_idx / col;
  const int global_col = global_idx % col;
  if (global_idx < row * col)
  {
    y[global_col * row + global_row] = x[global_idx];
  }
}

__global__ void mat_transpose_f32_row2col_kernel(float *x, float *y, const int row, const int col)
{
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_col = global_idx / row;
  const int global_row = global_idx % row;
  if (global_idx < row * col)
  {
    y[global_idx] = x[global_row * col + global_col];
  }
}

__global__ void mat_transpose_f32x4_col2row_kernel(float *x, float *y, const int row, const int col)
{
  const int global_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  const int global_row = global_idx / col;
  const int global_col = global_idx % col;
  if (global_idx * 4 < row * col)
  {
    float4 x_val = FLOAT4(x[global_idx]);
    y[global_col * row + global_row] = x_val.x;
    y[(global_col + 1) * row + global_row] = x_val.y;
    y[(global_col + 2) * row + global_row] = x_val.z;
    y[(global_col + 3) * row + global_row] = x_val.w;
  }
}

__global__ void mat_transpose_f32x4_row2col_kernel(float *x, float *y, const int row, const int col)
{
  const int global_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  const int global_col = global_idx / row;
  const int global_row = global_idx % row;
  if (global_idx * 4 < row * col)
  {
    float4 x_val;
    x_val.x = x[global_row * col + global_col];
    x_val.y = x[(global_row + 1) * col + global_col];
    x_val.z = x[(global_row + 2) * col + global_col];
    x_val.w = x[(global_row + 3) * col + global_col];
    FLOAT4(y[global_idx]) = FLOAT4(x_val);
  }
}

// --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                   \
  if (((T).options().dtype() != (th_type)))                    \
  {                                                            \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type);      \
  }

#define TORCH_BINDING_MAT_TRANSPOSE(tag, th_type, element_type, n_pack) \
  void mat_transpose_##tag(torch::Tensor x, torch::Tensor y)            \
  {                                                                     \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                              \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                              \
    const int M = x.size(0);                                            \
    const int N = x.size(1);                                            \
    dim3 block(WARP_SIZE);                                              \
    dim3 grid(((N * M + WARP_SIZE - 1) / WARP_SIZE) / n_pack);          \
    mat_transpose_##tag##_kernel<<<grid, block>>>(                      \
        reinterpret_cast<element_type *>(x.data_ptr()),                 \
        reinterpret_cast<element_type *>(y.data_ptr()), M, N);          \
  }

TORCH_BINDING_MAT_TRANSPOSE(f32_col2row, torch::kFloat32, float, 1)
TORCH_BINDING_MAT_TRANSPOSE(f32_row2col, torch::kFloat32, float, 1)
TORCH_BINDING_MAT_TRANSPOSE(f32x4_col2row, torch::kFloat32, float, 4)
TORCH_BINDING_MAT_TRANSPOSE(f32x4_row2col, torch::kFloat32, float, 4)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_col2row)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_col2row)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_row2col)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_row2col)
  // TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f16)
  // TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f16x2)
  // TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f16x8)
  // TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f16x8_pack)
}
