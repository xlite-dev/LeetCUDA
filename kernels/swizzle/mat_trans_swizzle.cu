#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>

// reference: https://zhuanlan.zhihu.com/p/4746910252
// 转置前的矩阵存储在dev_A中，矩阵大小为M*N，转置后的数据存储在dev_B中
__global__ void mat_trans_smem_naive_kernel(int* dev_A, int M, int N, int* dev_B) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  // 每个block处理32*32的矩阵块
  __shared__ int s_data[32][32];

  if (row < M && col < N) {
    // 从全局内存中加载数据，转置后写到共享内存中
    s_data[threadIdx.x][threadIdx.y] = dev_A[row * N + col];
    __syncthreads();
    int n_col = blockIdx.y * blockDim.y + threadIdx.x;
    int n_row = blockIdx.x * blockDim.x + threadIdx.y;
    if (n_col < M && n_row < N) {
      // 从转置后的共享内存按行写到全局内存结果中
      dev_B[n_row * M + n_col] = s_data[threadIdx.y][threadIdx.x];
    }
  }
}

// reference: https://zhuanlan.zhihu.com/p/4746910252
__global__ void mat_trans_smem_padding_kernel(int* dev_A, int M, int N, int* dev_B) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // 每个block处理32*32的矩阵块，尾部padding来避免bank conflict
  __shared__ int s_data[32][33];

  if (row < M && col < N) {
    s_data[threadIdx.x][threadIdx.y] = dev_A[row * N + col];
    __syncthreads();
    int n_col = blockIdx.y * blockDim.y + threadIdx.x;
    int n_row = blockIdx.x * blockDim.x + threadIdx.y;
    if (n_col < M && n_row < N) {
      dev_B[n_row * M + n_col] = s_data[threadIdx.y][threadIdx.x];
    }
  }
}

// reference: https://zhuanlan.zhihu.com/p/4746910252
__global__ void mat_trans_smem_swizzle_kernel(int* dev_A, int M, int N, int* dev_B) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int s_data[32][32];

  if (row < M && col < N) {
    // 从全局内存读取数据写入共享内存的逻辑坐标(row=x,col=y)
    // 其映射的物理存储位置位置(row=x,col=x^y)
    s_data[threadIdx.x][threadIdx.x ^ threadIdx.y] = dev_A[row * N + col];
    __syncthreads();
    int n_col = blockIdx.y * blockDim.y + threadIdx.x;
    int n_row = blockIdx.x * blockDim.x + threadIdx.y;
    if (n_row < N && n_col < M) {
      // 从共享内存的逻辑坐标(row=y,col=x)读取数据
      // 其映射的物理存储位置(row=y,col=x^y)
      dev_B[n_row * M + n_col] = s_data[threadIdx.y][threadIdx.x ^ threadIdx.y];
    }
  }
}
