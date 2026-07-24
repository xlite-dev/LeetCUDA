#pragma once
#include "base.cuh"
// sgemv.cuh: Phase 6 GEMV
// Phase 6: GEMV — 矩阵向量乘（M or N = 1, 纯 memory-bound 算子，warp-per-row 策略）
// =============================================================================
// 面试要点：
//   - GEMV 是典型的 memory-bound 算子：AI ≈ O(1)，瓶颈在内存带宽
//   - 核心策略：warp-per-row（每个 warp 负责矩阵的一行）
//   - 不同 K 值对应不同分块策略：
//     K=32 倍数：一个 warp 的 32 线程恰好覆盖 K 维
//     K=128 倍数：每个线程用 float4 处理 4 元素，warp 覆盖 128
//     K=16 < 32：一个 warp 用不满，用 kRowPerWarp=2 让每个 warp 处理 2 行
//
// a: M×K, x: K×1, y: M×1, 计算: y = a * x; N = 1

// ---- SGEMV K32: 基础 warp-per-row ----
// 设计：block(32, 4)，blockDim.x=kWarpSize=32（K 需为 32 倍数时一轮覆盖，否则内层循环 kNumWarps 次）
// grid(M/4)，每个 warp 负责一行
// K 为 32 的倍数时，warp 的 32 个线程恰好覆盖 K 维
// Grid:  ((M + 3) / 4, 1, 1)，每 block 处理 4 行
// Block: (32, 4, 1)，每行 1 warp
// 注意：该版本最适合 K 按 32 对齐；当 K 更小时通常切到 K16 这类专用分支
// source: LeetCUDA/kernels/sgemv/sgemv.cu
__global__ void sgemv_k32(float *a, float *x, float *y, int M, int K) {
  int tx = threadIdx.x; // 0~31
  int ty = threadIdx.y; // 0~3
  int lane = tx % kWarpSize; // 0~31
  int m = blockIdx.x * blockDim.y + ty; // 全局行号
  if (m < M) {
    float sum = 0.0f;
    // 沿 K 维的迭代数 = ceil(K/32)，每个 warp 要累加完整的K，那么
    // 每个thread就要负责累加NUM_ITERS个元素，NUM_ITERS = ceil(K/32)
    const int NUM_ITERS = (K + kWarpSize - 1) / kWarpSize;
#pragma unroll
    for (int w = 0; w < NUM_ITERS; ++w) {
      // 假设K是32的整倍数，m * K 本行的起始地址，x: Kx1
      int k = w * kWarpSize + lane;
      sum += a[m * K + k] * x[k];
    }
    sum = warp_reduce_sum<kWarpSize>(sum);
    // 每个 warp 处理一行，lane 0 写回结果
    if (lane == 0)
      y[m] = sum;
  }
}

// ---- SGEMV K128: float4 向量化 ----
// 每个线程处理 4 个元素(float4)，一个 warp 覆盖 128 个元素
// Grid:  ((M + 3) / 4, 1, 1)
// Block: (32, 4, 1)
// 注意：该版本最适合 K 按 128 对齐，且 x/a 的地址满足 float4 对齐
// source: LeetCUDA/kernels/sgemv/sgemv.cu
__global__ void sgemv_k128(float *a, float *x, float *y, int M, int K) {
  int tx = threadIdx.x; // 0~31
  int ty = threadIdx.y; // 0~3
  int lane = tx % kWarpSize; // 0~31
  int m = blockDim.y * blockIdx.x + ty;

  if (m < M) {
    float sum = 0.0f;
    // 沿 K 维的迭代数 = ceil(K/128)，每个 warp 每轮用 float4 覆盖 128 个 K 元素
    const int NUM_ITERS = (((K + kWarpSize - 1) / kWarpSize) + 4 - 1) / 4;
#pragma unroll
    for (int w = 0; w < NUM_ITERS; ++w) {
      int k = (w * kWarpSize + lane) * 4;
      float4 reg_x = FLOAT4(x[k]);
      float4 reg_a = FLOAT4(a[m * K + k]);
      sum += (reg_a.x * reg_x.x + reg_a.y * reg_x.y + reg_a.z * reg_x.z +
              reg_a.w * reg_x.w);
    }
    sum = warp_reduce_sum<kWarpSize>(sum);
    if (lane == 0)
      y[m] = sum;
  }
}

// ---- SGEMV K16: K < WarpSize, kRowPerWarp=2 ----
// 面试亮点：K=16 < 32，一个 warp 可以处理多行
// kRowPerWarp=2，kNumLanePerRow=16，前 16 个 lane 处理 row0，后 16 个 lane 处理 row1
// Grid:  ((M + 7) / 8, 1, 1)，NUM_ROWS=8
// Block: (32, 4, 1)
// 注意：这一版是面向 K=16 的专用写法；kRowPerWarp=2 时一个 warp 同时处理 2 行
// source: LeetCUDA/kernels/sgemv/sgemv.cu
template <const int kRowPerWarp = 2>
__global__ void sgemv_k16(float *A, float *x, float *y, int M, int K) {
  constexpr int kNumLanePerRow = (kWarpSize + kRowPerWarp - 1) / kRowPerWarp; // 16
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int lane = tx % kWarpSize;
  int k = lane % kNumLanePerRow; // row0: 0~15, t0~t15; row1: 0~15, t16~t31
  int m = (blockDim.y * blockIdx.x + ty) * kRowPerWarp + lane / kNumLanePerRow;
  if (m < M) {
    float sum = A[m * K + k] * x[k];
    // 按照kNumLanePerRow=16，分2组各自做 warp reduce sum，k==0的lane写回结果
    sum = warp_reduce_sum<kNumLanePerRow>(sum);
    // 注意：判断条件是 k == 0，不是 lane == 0！
    if (k == 0)
      y[m] = sum;
  }
}

// =============================================================================
