#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 256
#define WARP_SIZE_S 16
#define PAD 1
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

// FP32
// col2row means read x[row][col] and
// write y[col][row] row2col means read x[col][row] and write y[row][col]
__global__ void mat_transpose_f32_col2row_kernel(float *x, float *y,
                                                 const int row, const int col) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_row = global_idx / col;
  const int global_col = global_idx % col;
  if (global_idx < row * col) {
    y[global_col * row + global_row] = x[global_idx];
  }
}

__global__ void mat_transpose_f32_row2col_kernel(float *x, float *y,
                                                 const int row, const int col) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_col = global_idx / row;
  const int global_row = global_idx % row;
  if (global_idx < row * col) {
    y[global_idx] = x[global_row * col + global_col];
  }
}

__global__ void mat_transpose_f32x4_col2row_kernel(float *x, float *y,
                                                   const int row,
                                                   const int col) {
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int global_col = (global_idx * 4) % col;
  int global_row = (global_idx * 4) / col;

  if (global_row < row && global_col + 3 < col) {
    float4 x_val = reinterpret_cast<float4 *>(x)[global_idx];

    y[global_col * row + global_row] = x_val.x;
    y[(global_col + 1) * row + global_row] = x_val.y;
    y[(global_col + 2) * row + global_row] = x_val.z;
    y[(global_col + 3) * row + global_row] = x_val.w;
  }
}
__global__ void mat_transpose_f32x4_row2col_kernel(float *x, float *y,
                                                   const int row,
                                                   const int col) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_col = (global_idx * 4) / row;
  const int global_row = (global_idx * 4) % row;

  if (global_row < row && global_col < col) {
    float4 x_val;
    x_val.x = x[global_row * col + global_col];
    x_val.y = x[(global_row + 1) * col + global_col];
    x_val.z = x[(global_row + 2) * col + global_col];
    x_val.w = x[(global_row + 3) * col + global_col];
    reinterpret_cast<float4 *>(y)[global_idx] = FLOAT4(x_val);
  }
}

// work for row == col
__global__ void mat_transpose_f32_diagonal2d_kernel(float *x, float *y, int row,
                                                    int col) {
  const int block_y = blockIdx.x;
  const int block_x = (blockIdx.x + blockIdx.y) % gridDim.x;
  const int global_col = threadIdx.x + blockDim.x * block_x;
  const int global_row = threadIdx.y + blockDim.y * block_y;
  if (global_col < col && global_row < row) {
    y[global_row * col + global_col] = x[global_col * row + global_row];
  }
}

__global__ void mat_transpose_f32_col2row2d_kernel(float *x, float *y,
                                                   const int row,
                                                   const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_x < col && global_y < row) {
    y[global_x * row + global_y] = x[global_y * col + global_x];
  }
}

__global__ void mat_transpose_f32_row2col2d_kernel(float *x, float *y,
                                                   const int row,
                                                   const int col) {
  const int global_y = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_x = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_y < col && global_x < row) {
    y[global_y * row + global_x] = x[global_x * col + global_y];
  }
}

__global__ void mat_transpose_f32x4_col2row2d_kernel(float *x, float *y,
                                                     const int row,
                                                     const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_x * 4 + 3 < col && global_y < row) {
    float4 x_val = reinterpret_cast<float4 *>(x)[global_y * col / 4 + global_x];
    y[(global_x * 4) * row + global_y] = x_val.x;
    y[(global_x * 4 + 1) * row + global_y] = x_val.y;
    y[(global_x * 4 + 2) * row + global_y] = x_val.z;
    y[(global_x * 4 + 3) * row + global_y] = x_val.w;
  }
}
__global__ void mat_transpose_f32x4_row2col2d_kernel(float *x, float *y,
                                                     const int row,
                                                     const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_y * 4 + 3 < row && global_x < col) {
    float4 x_val;
    x_val.x = x[(global_y * 4) * col + global_x];
    x_val.y = x[(global_y * 4 + 1) * col + global_x];
    x_val.z = x[(global_y * 4 + 2) * col + global_x];
    x_val.w = x[(global_y * 4 + 3) * col + global_x];
    reinterpret_cast<float4 *>(y)[global_x * row / 4 + global_y] =
        FLOAT4(x_val);
  }
}

__global__ void mat_transpose_f32x4_shared_col2row2d_kernel(float *x, float *y,
                                                            const int row,
                                                            const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  __shared__ float tile[WARP_SIZE_S][WARP_SIZE_S * 4];
  if (global_x * 4 + 3 < col + 3 && global_y < row) {
    // load value from x to shared memory
    float4 x_val = reinterpret_cast<float4 *>(x)[global_y * col / 4 + global_x];
    FLOAT4(tile[local_y][local_x * 4]) = FLOAT4(x_val);
    __syncthreads();
    float4 smem_val;
    // load value from shared memory to y.
    // add STRIDE to satisfied different block size.
    constexpr int STRIDE = WARP_SIZE_S / 4;
    smem_val.x = tile[(local_y % STRIDE) * 4][local_x * 4 + local_y / STRIDE];
    smem_val.y =
        tile[(local_y % STRIDE) * 4 + 1][local_x * 4 + local_y / STRIDE];
    smem_val.z =
        tile[(local_y % STRIDE) * 4 + 2][local_x * 4 + local_y / STRIDE];
    smem_val.w =
        tile[(local_y % STRIDE) * 4 + 3][local_x * 4 + local_y / STRIDE];
    // map index n*n to (n/4)*(n*4)
    const int bid_y = blockIdx.y * blockDim.y;
    const int out_y = global_x * 4 + local_y / STRIDE;
    const int out_x = (local_y % STRIDE) * 4 + bid_y;
    reinterpret_cast<float4 *>(y)[(out_y * row + out_x) / 4] = FLOAT4(smem_val);
  }
}

__global__ void mat_transpose_f32x4_shared_row2col2d_kernel(float *x, float *y,
                                                            const int row,
                                                            const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  __shared__ float tile[WARP_SIZE_S * 4][WARP_SIZE_S];
  if (global_y * 4 < row && global_x < col) {
    // load value from x to shared memory
    float4 x_val;
    x_val.x = x[(global_y * 4) * col + global_x];
    x_val.y = x[(global_y * 4 + 1) * col + global_x];
    x_val.z = x[(global_y * 4 + 2) * col + global_x];
    x_val.w = x[(global_y * 4 + 3) * col + global_x];
    tile[local_y * 4][local_x] = x_val.x;
    tile[local_y * 4 + 1][local_x] = x_val.y;
    tile[local_y * 4 + 2][local_x] = x_val.z;
    tile[local_y * 4 + 3][local_x] = x_val.w;
    __syncthreads();
    float4 smem_val;
    // load value from shared memory to y.
    // add STRIDE to satisfied different block size.
    // map index n*n to (n/4)*(n*4)
    constexpr int STRIDE = WARP_SIZE_S / 4;
    smem_val.x = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4];
    smem_val.y =
        tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 1];
    smem_val.z =
        tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 2];
    smem_val.w =
        tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 3];
    const int bid_x = blockIdx.x * blockDim.x;
    const int bid_y = blockIdx.y * blockDim.y;

    const int out_y = bid_x + (local_y % STRIDE) * 4;
    const int out_x = bid_y * 4 + local_x * 4 + (local_y / STRIDE);
    y[out_y * row + out_x] = smem_val.x;
    y[(out_y + 1) * row + out_x] = smem_val.y;
    y[(out_y + 2) * row + out_x] = smem_val.z;
    y[(out_y + 3) * row + out_x] = smem_val.w;
  }
}

__global__ void mat_transpose_f32x4_shared_bcf_col2row2d_kernel(float *x,
                                                                float *y,
                                                                const int row,
                                                                const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  __shared__ float tile[WARP_SIZE_S][WARP_SIZE_S * 4 + PAD];
  if (global_x * 4 + 3 < col + 3 && global_y < row) {
    // load value from x to shared memory
    float4 x_val = reinterpret_cast<float4 *>(x)[global_y * col / 4 + global_x];
    tile[local_y][local_x * 4] = x_val.x;
    tile[local_y][local_x * 4 + 1] = x_val.y;
    tile[local_y][local_x * 4 + 2] = x_val.z;
    tile[local_y][local_x * 4 + 3] = x_val.w;
    __syncthreads();
    float4 smem_val;
    // load value from shared memory to y.
    // add STRIDE to satisfied different block size.
    constexpr int STRIDE = WARP_SIZE_S / 4;
    smem_val.x = tile[(local_y % STRIDE) * 4][local_x * 4 + local_y / STRIDE];
    smem_val.y =
        tile[(local_y % STRIDE) * 4 + 1][local_x * 4 + local_y / STRIDE];
    smem_val.z =
        tile[(local_y % STRIDE) * 4 + 2][local_x * 4 + local_y / STRIDE];
    smem_val.w =
        tile[(local_y % STRIDE) * 4 + 3][local_x * 4 + local_y / STRIDE];
    // map index n*n to (n/4)*(n*4)
    const int bid_y = blockIdx.y * blockDim.y;
    const int out_y = global_x * 4 + local_y / STRIDE;
    const int out_x = (local_y % STRIDE) * 4 + bid_y;
    reinterpret_cast<float4 *>(y)[(out_y * row + out_x) / 4] = FLOAT4(smem_val);
  }
}

__global__ void mat_transpose_f32x4_shared_bcf_row2col2d_kernel(float *x,
                                                                float *y,
                                                                const int row,
                                                                const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  __shared__ float tile[WARP_SIZE_S * 4][WARP_SIZE_S + PAD];
  if (global_y * 4 < row && global_x < col) {
    // load value from x to shared memory
    float4 x_val;
    x_val.x = x[(global_y * 4) * col + global_x];
    x_val.y = x[(global_y * 4 + 1) * col + global_x];
    x_val.z = x[(global_y * 4 + 2) * col + global_x];
    x_val.w = x[(global_y * 4 + 3) * col + global_x];
    tile[local_y * 4][local_x] = x_val.x;
    tile[local_y * 4 + 1][local_x] = x_val.y;
    tile[local_y * 4 + 2][local_x] = x_val.z;
    tile[local_y * 4 + 3][local_x] = x_val.w;
    __syncthreads();
    float4 smem_val;
    // load value from shared memory to y.
    // add STRIDE to satisfied different block size.
    // map index n*n to (n/4)*(n*4)
    constexpr int STRIDE = WARP_SIZE_S / 4;
    smem_val.x = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4];
    smem_val.y =
        tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 1];
    smem_val.z =
        tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 2];
    smem_val.w =
        tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 3];
    const int bid_x = blockIdx.x * blockDim.x;
    const int bid_y = blockIdx.y * blockDim.y;

    const int out_y = bid_x + (local_y % STRIDE) * 4;
    const int out_x = bid_y * 4 + local_x * 4 + (local_y / STRIDE);
    y[out_y * row + out_x] = smem_val.x;
    y[(out_y + 1) * row + out_x] = smem_val.y;
    y[(out_y + 2) * row + out_x] = smem_val.z;
    y[(out_y + 3) * row + out_x] = smem_val.w;
  }
}

__global__ void mat_transpose_f32x4_shared_bcf_merge_write_row2col2d_kernel(
    float *x, float *y, const int row, const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  __shared__ float tile[WARP_SIZE_S * 4][WARP_SIZE_S + PAD];
  if (global_y * 4 < row && global_x < col) {
    // load value from x to shared memory
    float4 x_val;
    x_val.x = x[(global_y * 4) * col + global_x];
    x_val.y = x[(global_y * 4 + 1) * col + global_x];
    x_val.z = x[(global_y * 4 + 2) * col + global_x];
    x_val.w = x[(global_y * 4 + 3) * col + global_x];
    tile[local_y * 4][local_x] = x_val.x;
    tile[local_y * 4 + 1][local_x] = x_val.y;
    tile[local_y * 4 + 2][local_x] = x_val.z;
    tile[local_y * 4 + 3][local_x] = x_val.w;
    __syncthreads();
    float4 smem_val;
    // load value from shared memory to y.
    smem_val.x = tile[local_x * 4][local_y];
    smem_val.y = tile[local_x * 4 + 1][local_y];
    smem_val.z = tile[local_x * 4 + 2][local_y];
    smem_val.w = tile[local_x * 4 + 3][local_y];

    const int gid_x = blockIdx.x * blockDim.x;
    const int gid_y = blockIdx.y * blockDim.y * 4;
    const int out_y = gid_y + local_x * 4;
    const int out_x = gid_x + local_y;
    reinterpret_cast<float4 *>(y)[(out_x * row + out_y) / 4] = FLOAT4(smem_val);
  }
}

// ============================================================================
// 宏定义部分：用于自动生成PyTorch扩展的绑定代码
// ============================================================================

/**
 * 字符串化宏：将宏参数转换为字符串字面量
 * 例如：STRINGFY(hello) 会被展开为 "hello"
 */
#define STRINGFY(str) #str

/**
 * PyTorch扩展通用绑定宏
 * 功能：将C++函数绑定到Python模块中，使其可以从Python调用
 * 参数：
 *   - func: 要绑定的C++函数名
 * 展开后效果：m.def("function_name", &function_name, "function_name");
 */
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

/**
 * PyTorch张量数据类型检查宏
 * 功能：运行时检查张量的数据类型是否符合预期，不符合则抛出异常
 * 参数：
 *   - T: 要检查的PyTorch张量
 *   - th_type: 期望的数据类型（如torch::kFloat32, torch::kHalf等）
 * 作用：确保CUDA kernel接收到正确类型的数据，避免类型不匹配导致的错误
 */
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

/**
 * 1D索引矩阵转置函数生成宏
 * 功能：自动生成一个完整的矩阵转置host函数，该函数调用对应的CUDA kernel
 * 参数：
 *   - tag: 函数标识符（如f32_col2row），用于构造函数名和kernel名
 *   - th_type: PyTorch数据类型（如torch::kFloat32）
 *   - element_type: C++数据类型（如float）
 *   - n_pack: 向量化程度，每个线程处理多少个元素（1或4）
 * 
 * 生成的函数特点：
 *   - 使用1D线程块布局 (WARP_SIZE)
 *   - 网格大小根据矩阵总元素数量和向量化程度计算
 *   - 适用于简单的矩阵转置操作
 */
#define TORCH_BINDING_MAT_TRANSPOSE(tag, th_type, element_type, n_pack)        \
  void mat_transpose_##tag(torch::Tensor x, torch::Tensor y) {                 \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int M = x.size(0);  /* 输入矩阵行数 */                                \
    const int N = x.size(1);  /* 输入矩阵列数 */                                \
    dim3 block(WARP_SIZE);    /* 1D线程块：256个线程 */                         \
    dim3 grid(((N * M + WARP_SIZE - 1) / n_pack / WARP_SIZE));  /* 网格大小计算 */ \
    mat_transpose_##tag##_kernel<<<grid, block>>>(                             \
        reinterpret_cast<element_type *>(x.data_ptr()),                        \
        reinterpret_cast<element_type *>(y.data_ptr()), M, N);                 \
  }

/**
 * 2D索引矩阵转置函数生成宏
 * 功能：自动生成一个2D索引的矩阵转置host函数，性能通常更好
 * 参数：
 *   - tag: 函数标识符
 *   - th_type: PyTorch数据类型
 *   - element_type: C++数据类型
 *   - n_element_row: 每个线程块在行方向处理的元素倍数
 *   - n_element_col: 每个线程块在列方向处理的元素倍数
 * 
 * 生成的函数特点：
 *   - 使用2D线程块布局 (16x16)
 *   - 网格大小考虑了向量化处理的倍数
 *   - 更适合利用共享内存和合并访问的优化版本
 *   - 支持对角线转置等特殊算法
 */
#define TORCH_BINDING_MAT_TRANSPOSE2D(tag, th_type, element_type,              \
                                      n_element_row, n_element_col)            \
  void mat_transpose_##tag##2d(torch::Tensor x, torch::Tensor y) {             \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int M = x.size(0);  /* 输入矩阵行数 */                                \
    const int N = x.size(1);  /* 输入矩阵列数 */                                \
    dim3 block(WARP_SIZE_S, WARP_SIZE_S);  /* 2D线程块：16x16=256个线程 */      \
    dim3 grid((N + WARP_SIZE_S - 1) / (WARP_SIZE_S * n_element_col),  /* X方向网格数 */ \
              (M + WARP_SIZE_S - 1) / (WARP_SIZE_S * n_element_row));  /* Y方向网格数 */ \
    mat_transpose_##tag##2d_kernel<<<grid, block>>>(                           \
        reinterpret_cast<element_type *>(x.data_ptr()),                        \
        reinterpret_cast<element_type *>(y.data_ptr()), M, N);                 \
  }

// ============================================================================
// 矩阵转置函数实例化：使用宏自动生成各种优化版本的转置函数
// ============================================================================

// --------------- 1D索引版本：基础实现 ---------------
// 适用于简单场景，性能一般但实现直观
TORCH_BINDING_MAT_TRANSPOSE(f32_col2row, torch::kFloat32, float, 1)    // 单元素处理，按列读取按行写入
TORCH_BINDING_MAT_TRANSPOSE(f32_row2col, torch::kFloat32, float, 1)    // 单元素处理，按行读取按列写入
TORCH_BINDING_MAT_TRANSPOSE(f32x4_col2row, torch::kFloat32, float, 4)  // 向量化处理，4个float一组
TORCH_BINDING_MAT_TRANSPOSE(f32x4_row2col, torch::kFloat32, float, 4)  // 向量化处理，4个float一组

// --------------- 2D索引版本：性能优化 ---------------
// 使用2D线程块布局，更好的缓存局部性和内存访问模式
TORCH_BINDING_MAT_TRANSPOSE2D(f32_col2row, torch::kFloat32, float, 1, 1)    // 基础2D版本
TORCH_BINDING_MAT_TRANSPOSE2D(f32_row2col, torch::kFloat32, float, 1, 1)    // 基础2D版本
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_col2row, torch::kFloat32, float, 1, 4)  // 列向量化：每行处理4个元素
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_row2col, torch::kFloat32, float, 4, 1)  // 行向量化：每列处理4个元素

// --------------- 特殊算法版本 ---------------
TORCH_BINDING_MAT_TRANSPOSE2D(f32_diagonal, torch::kFloat32, float, 1, 1)   // 对角线转置算法，适用于方阵

// --------------- 共享内存优化版本 ---------------
// 利用共享内存减少全局内存访问，提高带宽利用率
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_col2row, torch::kFloat32, float, 1, 4)  // 共享内存+列向量化
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_row2col, torch::kFloat32, float, 4, 1)  // 共享内存+行向量化

// --------------- Bank Conflict Free (BCF) 优化版本 ---------------
// 通过特殊的内存访问模式避免共享内存的bank冲突
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_bcf_col2row, torch::kFloat32, float, 1, 4)  // BCF优化
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_bcf_row2col, torch::kFloat32, float, 4, 1)  // BCF优化
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_bcf_merge_write_row2col,                     // BCF+合并写入优化
                              torch::kFloat32, float, 4, 1)

// ============================================================================
// CuTe (CUTLASS Utilities for Tensor Layout) 实现的外部函数声明
// ============================================================================
// CuTe是NVIDIA CUTLASS库的一部分，提供更高级的张量操作抽象
// 这些函数在mat_transpose_cute.cu文件中实现，使用CuTe的高级API

// --------------- 寄存器级别优化版本 ---------------
extern void mat_transpose_cute_col2row_reg(torch::Tensor, torch::Tensor);  // 纯寄存器操作，列到行转置
extern void mat_transpose_cute_row2col_reg(torch::Tensor, torch::Tensor);  // 纯寄存器操作，行到列转置

// --------------- 共享内存版本 ---------------
extern void mat_transpose_cute_col_smem(torch::Tensor, torch::Tensor);     // 使用共享内存，列主序处理
extern void mat_transpose_cute_row_smem(torch::Tensor, torch::Tensor);     // 使用共享内存，行主序处理

// --------------- 共享内存 + Swizzling 版本 ---------------
// Swizzling: 通过重排数据访问模式来避免bank conflicts，提高内存带宽利用率
extern void mat_transpose_cute_col_smem_swizzled(torch::Tensor, torch::Tensor);  // 列处理+内存重排
extern void mat_transpose_cute_row_smem_swizzled(torch::Tensor, torch::Tensor);  // 行处理+内存重排

// --------------- 向量化访问版本 ---------------
// 使用向量化指令（如float4）来提高内存带宽
extern void mat_transpose_cute_row_cvectorized(torch::Tensor, torch::Tensor);    // 列向量化（连续内存访问）
extern void mat_transpose_cute_row_rvectorized(torch::Tensor, torch::Tensor);    // 行向量化

// --------------- 向量化 + Swizzling 组合优化 ---------------
extern void mat_transpose_cute_row_cvectorized_swizzled(torch::Tensor,            // 列向量化+内存重排
                                                        torch::Tensor);
extern void mat_transpose_cute_row_rvectorized_swizzled(torch::Tensor,            // 行向量化+内存重排
                                                        torch::Tensor);

// --------------- 最终优化版本 ---------------
extern void
    mat_transpose_cute_row_rvectorized_swizzled_optimized(torch::Tensor,         // 所有优化技术的综合
                                                          torch::Tensor);

// ============================================================================
// Python模块定义：将所有C++函数导出到Python
// ============================================================================
/**
 * PyBind11模块定义
 * 功能：创建Python扩展模块，将上面定义的所有C++函数绑定到Python中
 * 参数：
 *   - TORCH_EXTENSION_NAME: 模块名称（在Python中import时使用的名称）
 *   - m: 模块对象，用于注册函数
 * 
 * 绑定后的使用方式：
 *   在Python中：import mat_transpose_lib
 *   调用：mat_transpose_lib.mat_transpose_f32_col2row(input_tensor, output_tensor)
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  
  // --------------- 基础1D索引版本绑定 ---------------
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_col2row)       // 基础单元素转置
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_col2row)     // 4元素向量化转置
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_row2col)       // 基础行列转换
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_row2col)     // 向量化行列转换
  
  // --------------- 2D索引优化版本绑定 ---------------
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_col2row2d)     // 2D布局基础版本
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_col2row2d)   // 2D布局向量化版本
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_row2col2d)     // 2D布局行列转换
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_row2col2d)   // 2D布局向量化行列转换
  
  // --------------- 特殊算法版本绑定 ---------------
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_diagonal2d)    // 对角线算法（适用于方阵）
  
  // --------------- 共享内存优化版本绑定 ---------------
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_shared_col2row2d)  // 共享内存+列向量化
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_shared_row2col2d)  // 共享内存+行向量化
  
  // --------------- Bank Conflict Free优化版本绑定 ---------------
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_shared_bcf_col2row2d)      // BCF优化
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_shared_bcf_row2col2d)      // BCF优化
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_shared_bcf_merge_write_row2col2d)  // BCF+合并写入
  
  // --------------- CuTe高级实现版本绑定 ---------------
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_col2row_reg)                // CuTe寄存器版本
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row2col_reg)                // CuTe寄存器版本
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_smem)                   // CuTe共享内存版本
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_col_smem)                   // CuTe共享内存版本
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_col_smem_swizzled)          // CuTe共享内存+重排
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_smem_swizzled)          // CuTe共享内存+重排
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_cvectorized)            // CuTe列向量化
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_rvectorized)            // CuTe行向量化
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_cvectorized_swizzled)   // CuTe列向量化+重排
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_rvectorized_swizzled)   // CuTe行向量化+重排
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_rvectorized_swizzled_optimized)  // CuTe终极优化版本
}
