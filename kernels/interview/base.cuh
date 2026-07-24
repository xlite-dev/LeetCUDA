#pragma once
#include "common.cuh"
// base.cuh: Phase 0-5 (Warp/Block Reduce, Elementwise, Softmax/Norm, RoPE, Transpose)
// Phase 0: 面试框架速查（纯注释，面试开场必备的基础知识）
// =============================================================================

// ---- GPU 架构速查 ----
//
// SM (Streaming Multiprocessor) 内部结构：
//   - Warp Scheduler ×4：每 SM 4 个 warp scheduler，每个每周期可发射 1 条指令
//   - Register File：每 SM 65536 × 32-bit (4 bytes) = 256KB
//   - Shared Memory / L1：可配置，最大 shared memory ~228KB (Hopper)
//   - Tensor Cores：Hopper 每 SM 4 个；Blackwell 数量随型号/定义不同，建议以官方 ISV guide 为准
//   - Warp = 32 threads：最小调度单元，SIMT 执行模型
//
// Memory Hierarchy 带宽数量级（H100 参考）：
//   HBM3：        ~3.35 TB/s（理论），实际 ~2.5-3.0 TB/s
//   L2 Cache：    ~12 TB/s（50MB，跨 SM 共享）
//   L1/SMEM：     ~19 TB/s（每 SM ~228KB）
//   Register：    ~0 延迟，~100+ TB/s 等效带宽
//
// 关键瓶颈判断：
//   Memory-bound：AI (Arithmetic Intensity) < 机器 FLOPS/带宽 比值
//   Compute-bound：AI 足够大，受限于计算吞吐
//   Latency-bound：线程不够多，无法隐藏内存延迟
//
// Occupancy 公式：
//   occupancy = active_warps / max_warps_per_SM
//   受三类资源分别取下限：每线程寄存器数 → threads/SM；每 block shared memory
//   → blocks/SM；block 大小 → blocks/SM

// ---- 常见优化手段速查清单 ----
//
// 1. Coalesced Memory Access（合并访问）
//    - 同一 warp 的线程访问连续的 128B 对齐地址 → 1 次内存事务
//    - 否则产生多次事务（最坏 32 次）
//
// 2. Tiling（分块）
//    - 将数据从 HBM 分块加载到 shared memory 复用，减少 HBM 访问
//    - GEMM: Block Tile (BM×BN) + K Tile (BK)
//
// 3. Vectorized Memory Access（向量化）
//    - 使用 float4/half2 等向量类型，减少 load/store 指令数
//    - float4 = 128-bit，单条指令加载 16 bytes
//
// 4. Thread Tile（寄存器分块）
//    - 每个线程计算多个输出元素（TM×TN），提高计算密度
//    - 减少线程总数，降低同步开销
//
// 5. Bank Conflict Avoidance
//    - Shared memory 有 32 banks × 4 bytes
//    - 同 warp 多线程访问同一 bank 的不同地址 → bank conflict → 串行化
//    - 解决方案：PAD（在每行末尾加 1 个元素打破对齐）
//
// 6. Pipeline / Double Buffering（流水线）
//    - cp.async 异步拷贝下一批数据时同时做当前批的计算
//    - Stage 数 = 2/3/4，权衡 shared memory 占用和延迟隐藏
//
// 7. Tensor Core（MMA / WGMMA）
//    - MMA m16n8k16 (Ampere): warp 级指令，单 warp 完成 16×8×16 的矩阵乘
//    - WGMMA m64n128k16 (Hopper): warpgroup 级指令（128 threads），异步执行
//
// 8. Warp Specialization（Hopper+）
//    - Producer warpgroup 做 TMA 数据搬运，Consumer warpgroup 做计算
//    - 通过 cuda::barrier 同步，完全解耦数据搬运和计算
//
// 9. TMA (Tensor Memory Accelerator, Hopper+)
//    - 硬件 DMA 引擎，支持 1D~5D 寻址，低寄存器开销
//    - 配合 cp.async.bulk 实现异步数据搬运

// ---- Roofline 分析公式 ----
//
// AI (Arithmetic Intensity) = FLOPs / Bytes_transferred
//
// GEMM (M=N=K=4096):
//   FLOPs = 2 × M × N × K = 2 × 4096³ ≈ 137 GFLOPS
//   Bytes = (M×K + K×N + M×N) × sizeof(float) ≈ 200 MB
//   AI    ≈ 137G / 200M ≈ 685 FLOPS/Byte → compute-bound（远超 H100 ridge point：
//   FP16 TC ≈ 295:1，FP32 ≈ 20:1）
//
// GEMV (M=4096, K=4096):
//   Bytes = (M×K + K + M) × sizeof(float) ≈ 67 MB
//   AI    ≈ 33M / 67M ≈ 0.5 FLOPS/Byte → severely memory-bound
//
// Softmax (N=4096): AI ≈ (5×N) / (2×N×4) = 5/8 ≈ 0.625 FLOPS/Byte → memory-bound

// =============================================================================
// Phase 1: 头文件 + 宏定义 + 基础原语（Warp Reduce / Block Reduce）
// =============================================================================
// 面试要点：
//   - warp_reduce: 用 __shfl_xor_sync 做蝶形归约，O(logN) 步，无需 shared
//   memory
//   - block_reduce: 两级归约（warp → shared memory → warp0 broadcast），
//     注意最后必须 broadcast 回所有线程（__shfl_sync），否则只有 warp0 知道结果
//   - 为什么不用 __shfl_down_sync？xor 模式所有线程做相同工作量，更均衡

#include <algorithm>
#include <cstring>
#include <cuda_fp16.h>

// =============================================================================
// Phase 1a: Warp Reduce（warp 内归约，纯寄存器操作，无需 shared memory）
// =============================================================================

// Warp Reduce Sum — generic (used by both FP32 and FP16 contexts)
// 使用 __shfl_xor_sync 做蝶形归约（butterfly reduction）
// 复杂度 O(logN)，N=32 时仅需 5 步
// 模板参数：T=数据类型, kWarpWidth=segment width（默认 32）
// kWarpWidth 会作为 __shfl_xor_sync 的第 4 个实参 width，限制 shuffle 在同一 segment 内
// 当 kWarpWidth < 32（如 FA 中 kWarpWidth=4）时，只有同 segment 的 lane 参与通信
//   蝶形归约示意（以 warpSize=8 为例，实际 warpSize=32 有 5 次迭代）：
//
//   初始: 每个 lane 持有自己的值 v0..v7
//   lane:  0    1    2    3    4    5    6    7
//   val:  v0   v1   v2   v3   v4   v5   v6   v7
//
//   mask=4 (第1次迭代，lane i 与 lane i^4 交换并累加):
//          ┌──────────────┐
//   对:   (0,4) (1,5) (2,6) (3,7)
//
//   lane:  0    1    2    3    4    5    6    7
//   val: v0+v4 v1+v5 v2+v6 v3+v7 v4+v0 v5+v1 v6+v2 v7+v3
//
//   mask=2 (第2次迭代，lane i 与 lane i^2 交换并累加):
//          ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
//   对:   (0,2) (1,3) (4,6) (5,7)
//       ──── 前4个一组 ────   ──── 后4个一组 ────
//
//   lane:  0    1    2    3    4    5    6    7  (每lane持有前一轮2个值的和再加本轮配对)
//   val: Σ{0,2,4,6} Σ{1,3,5,7} Σ{0,2,4,6} Σ{1,3,5,7} Σ{0,2,4,6} Σ{1,3,5,7} Σ{0,2,4,6} Σ{1,3,5,7}
//        = v0+v2+v4+v6 ... (逐步归约)
//
//   mask=1 (第3次迭代，lane i 与 lane i^1 交换并累加):
//          ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐
//   对:   (0,1)(2,3)(4,5)(6,7)
//
//   lane:  0    1    2    3    4    5    6    7
//   val:  Σall Σall Σall Σall Σall Σall Σall Σall  ← 所有 lane 拥有全归约结果！
//
//   mask=0: 循环终止，归约完成。
//
//   关键性质：
//   - XOR 配对是对称的：lane i 的配对对象是 lane i^mask，而 (i^mask)^mask = i
//   - 每轮每个 lane 只和恰好 1 个其他 lane 通信（一对一，无冲突）
//   - 每轮信息传递距离减半：16→8→4→2→1（距离减半，信息翻倍）
//   - O(log₂ N) 步完成，无需 shared memory，纯寄存器操作
//   - __shfl_xor_sync 第四个参数 kWarpWidth 限制 segment width：
//     当 kWarpWidth=4 时，只有同 segment(4个一组)内的 lane 参与 shuffle
template <const int kWarpWidth = kWarpSize, typename T = float>
__device__ __forceinline__ T warp_reduce_sum(T val) {
#pragma unroll
  for (int mask = kWarpWidth >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask, kWarpWidth);
  }
  return val;
}

// Warp Reduce Max — generic
template <const int kWarpWidth = kWarpSize, typename T = float>
__device__ __forceinline__ T warp_reduce_max(T val) {
#pragma unroll
  for (int mask = kWarpWidth >> 1; mask >= 1; mask >>= 1) {
    val = max(val, __shfl_xor_sync(0xffffffff, val, mask, kWarpWidth));
  }
  return val;
}

// =============================================================================
// Phase 1b: Block Reduce（block 内归约，两级：warp → shared memory → warp reduce）
// =============================================================================

// Block Reduce Sum — FP32（增强版，带 broadcast）
// 两级归约流程：
//   1. 每个 warp 内做 warp_reduce_sum → 得到每 warp 的一个值
//   2. warp leader (lane=0) 写入 shared memory
//   3. syncthreads 后，lane 0~kNumWarps-1 读取 shared memory
//   4. 所有 warp 内再做一次 warp_reduce<kNumWarps> → 得到最终结果
//   5. __shfl_sync broadcast 到所有线程（关键！否则每个warp只有lane<kNumWarps 知道结果）
template <const int kNumThreads = 256>
__device__ float block_reduce_sum(float val) {
  constexpr int kNumWarps = (kNumThreads + kWarpSize - 1) / kWarpSize;
  int warp = threadIdx.x / kWarpSize;
  int lane = threadIdx.x % kWarpSize;
  __shared__ float shared[kNumWarps];

  float value = warp_reduce_sum<kWarpSize>(val);
  if (lane == 0)
    shared[warp] = value;
  __syncthreads();
  value = (lane < kNumWarps) ? shared[lane] : 0.0f;
  value = warp_reduce_sum<kNumWarps>(value);
  // 关键：broadcast 结果到所有线程，后续用 result
  // 做除法等操作时所有线程都能拿到
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}

// Block Reduce Max — FP32（增强版，带 broadcast）
template <const int kNumThreads = 256>
__device__ float block_reduce_max(float val) {
  constexpr int kNumWarps = (kNumThreads + kWarpSize - 1) / kWarpSize;
  int warp = threadIdx.x / kWarpSize;
  int lane = threadIdx.x % kWarpSize;
  __shared__ float shared[kNumWarps];

  float value = warp_reduce_max<kWarpSize>(val);
  if (lane == 0)
    shared[warp] = value;
  __syncthreads();
  value = (lane < kNumWarps) ? shared[lane] : -FLT_MAX;
  value = warp_reduce_max<kNumWarps>(value);
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}

// ---- Block Reduce Sum All: y = sum(a[0..N-1]) ----
// 多 block 各自做 warp→smem→warp0 reduce，然后 atomicAdd 到全局 y
// 跨 block 求和的常见模式，适合 N 较大时使用；
// Grid:  ((N + 255) / 256, 1, 1)
// Block: (256, 1, 1)
// source: LeetCUDA/kernels/reduce/block_all_reduce.cu
template <const int kNumThreads = 256>
__global__ void block_reduce_all(float *a, float *y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * kNumThreads + tid;
  constexpr int kNumWarps = (kNumThreads + kWarpSize - 1) / kWarpSize;
  __shared__ float shared[kNumWarps];

  float val = (idx < N) ? a[idx] : 0.0f;
  int warp = tid / kWarpSize;
  int lane = tid % kWarpSize;

  val = warp_reduce_sum<kWarpSize>(val);
  if (lane == 0)
    shared[warp] = val;
  __syncthreads();

  val = (lane < kNumWarps) ? shared[lane] : 0.0f;
  if (warp == 0)
    val = warp_reduce_sum<kNumWarps>(val);
  if (tid == 0) // tid == 0, not lane 0. 只有 block 内的一个线程负责写回全局结果，避免重复累加
    atomicAdd(y, val);
}

// ---- Dot Product: y = sum(a[i] * b[i]) ----
// 核心模式：elementwise 乘法 → block reduce → atomicAdd 全局累加
// Grid:  ((N + 255) / 256, 1, 1)
// Block: (256, 1, 1)
// source: LeetCUDA/kernels/dot-product/dot_product.cu
template <const int kNumThreads = 256>
__global__ void dot(float *a, float *b, float *y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * kNumThreads + tid;
  constexpr int kNumWarps = (kNumThreads + kWarpSize - 1) / kWarpSize;
  __shared__ float shared[kNumWarps];

  float prod = (idx < N) ? a[idx] * b[idx] : 0.0f;
  int warp = tid / kWarpSize;
  int lane = tid % kWarpSize;

  prod = warp_reduce_sum<kWarpSize>(prod);
  if (lane == 0)
    shared[warp] = prod;
  __syncthreads();

  prod = (lane < kNumWarps) ? shared[lane] : 0.0f;
  if (warp == 0) // 只需要 warp 0 的线程继续 reduce 即可
    prod = warp_reduce_sum<kNumWarps>(prod);
  if (tid == 0) // tid == 0, not lane 0. 只有 block 内的一个线程负责写回全局结果，避免重复累加
    atomicAdd(y, prod);
}

// Dot Product + float4
// Grid:  ((N + 255) / 256, 1, 1), 每个block处理256元素，float4向量化后每个线程处理4元素
// Block: (64, 1, 1)，256/4=64
// 注意：该版本默认输入地址满足 float4 对齐；最适合 N 按 4 对齐的场景
// source: LeetCUDA/kernels/dot-product/dot_product.cu
template <const int kNumThreads = 256 / 4>
__global__ void dot_vec4(float *a, float *b, float *y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * kNumThreads + tid) * 4;
  constexpr int kNumWarps = (kNumThreads + kWarpSize - 1) / kWarpSize;
  __shared__ float shared[kNumWarps];

  float4 reg_a = FLOAT4(a[idx]);
  float4 reg_b = FLOAT4(b[idx]);
  float prod = (idx < N) ? (reg_a.x * reg_b.x + reg_a.y * reg_b.y +
                            reg_a.z * reg_b.z + reg_a.w * reg_b.w)
                         : 0.0f;
  int warp = tid / kWarpSize;
  int lane = tid % kWarpSize;

  prod = warp_reduce_sum<kWarpSize>(prod);
  if (lane == 0)
    shared[warp] = prod;
  __syncthreads();

  prod = (lane < kNumWarps) ? shared[lane] : 0.0f;
  if (warp == 0)
    prod = warp_reduce_sum<kNumWarps>(prod);
  if (tid == 0)
    atomicAdd(y, prod);
}


// =============================================================================
// Phase 2: Elementwise Ops（逐元素操作，演示 coalesced access + vectorize）
// =============================================================================
// 面试要点：
//   - 逐元素操作是最简单的 kernel，核心考点是 memory coalescing
//   - float4 向量化可将内存事务数减为 1/4，大幅提升 bandwidth utilization
//   - grid/block 维度设计：grid(N/threads), block(threads)，一维即可

// ---- ReLU: y = max(0, x) ----
// Grid:  ((N + 255) / 256, 1, 1)
// Block: (256, 1, 1)
// source: LeetCUDA/kernels/relu/relu.cu
__global__ void relu(float *x, float *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    y[idx] = fmaxf(0.0f, x[idx]);
}

// ReLU + float4 向量化：每个线程处理 4 个元素，减少 75% 的 load/store 指令
// block(64)×4(float4)=256 元素/block，与基础版吞吐相同
// Grid:  ((N + 255) / 256, 1, 1)
// Block: (64, 1, 1)
// 注意：该版本默认地址满足 float4 对齐；最适合 N 按 4 对齐的场景
// source: LeetCUDA/kernels/relu/relu.cu
__global__ void relu_vec4(float *x, float *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]); // 单条 128-bit load
    float4 reg_y;
    reg_y.x = fmaxf(0.0f, reg_x.x);
    reg_y.y = fmaxf(0.0f, reg_x.y);
    reg_y.z = fmaxf(0.0f, reg_x.z);
    reg_y.w = fmaxf(0.0f, reg_x.w);
    FLOAT4(y[idx]) = reg_y; // 单条 128-bit store
  }
}

// ---- Elementwise Add: c = a + b ----
// Grid:  ((N + 255) / 256, 1, 1)
// Block: (256, 1, 1)
// source: LeetCUDA/kernels/elementwise/elementwise.cu
__global__ void elementwise_add(float *a, float *b, float *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = a[idx] + b[idx];
}

// Elementwise Add + float4 向量化
// Grid:  ((N + 255) / 256, 1, 1)
// Block: (64, 1, 1)，block(64)×4(float4)=256 元素/block
// 注意：主路径要求 4 元素对齐；尾部不足 4 个元素时回退到标量处理
// source: LeetCUDA/kernels/elementwise/elementwise.cu
__global__ void elementwise_add_vec4(float *a, float *b, float *c, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if ((idx + 3) < N) {
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FLOAT4(c[idx]) = reg_c;
  } else if (idx < N) {
    for (int i = 0; (idx + i) < N; i++) {
      c[idx + i] = a[idx + i] + b[idx + i];
    }
  }
}

// ---- Histogram: y[a[i]]++ ----
// 演示 atomicAdd 的用法：多个线程可能同时更新同一个 bin
// Grid:  ((N + 255) / 256, 1, 1)
// Block: (256, 1, 1)
// source: LeetCUDA/kernels/histogram/histogram.cu
__global__ void histogram(int *a, int *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    atomicAdd(&(y[a[idx]]), 1);
}

// ---- Merge Attn States: 合并 Split-KV Attention 的部分结果 ----
// 实现 FlashInfer 论文 (arXiv 2501.01005) Section 2.2 的 attention 合并逻辑。
//
// 面试要点：
//   - 应用场景：Split-KV attention 将序列沿 K 维分段计算 attention，
//     每段产出部分结果 (O_i, LSE_i)，本 kernel 将两段按指数权重加权合并。
//   - 数学公式（5 步推导）：
//     Step 1 — max 归一化（数值稳定，与 safe softmax 同理）:
//              L_max = max(LSE_1, LSE_2)
//     Step 2 — 还原指数权重（un-normalized）:
//              w_1 = exp(LSE_1 - L_max),  w_2 = exp(LSE_2 - L_max)
//     Step 3 — 归一化为混合比例:
//              alpha = w_1 / (w_1 + w_2),  beta = w_2 / (w_1 + w_2)
//              满足 alpha + beta = 1
//     Step 4 — 逐元素加权合并:
//              O = alpha * O_1 + beta * O_2
//   - LSE 布局 [num_heads, num_tokens]（与 FlashAttention-2惯例一致，
//     lse[head_idx][token_idx]）
//   - Output 布局 [num_tokens, num_heads, head_size]（token 维在最外，
//     展平为 [T0H0, T0H1, ..., T1H0, ...]）
//   - inf LSE → -inf：空 attention 段（causal mask 等导致全部 score
//     为 -inf）的 LSE 可能为 +inf，替换后 exp(-inf - L_max) = 0，
//     该段权重退化为 0
//   - 128-bit 向量化：uint4 = 4 × float，每线程处理 4 个输出元素，
//     pack load → FMA → pack store 一条流水线
//   - AI ≈ 3 FLOP / 20 bytes ≈ 0.15 FLOPS/Byte → severely memory-bound
//
// 线程到数据的 3D 映射（以 num_tokens=2, num_heads=2, head_size=8 为例）：
//   kPackSize = 16 / sizeof(float) = 4（每线程 4 个 float = 128-bit）
//   threads_per_head = head_size / 4 = 2
//   total_threads = num_tokens * num_heads * threads_per_head = 8
//
//   global_idx:       0      1      2      3      4      5      6      7
//   token_head_idx:   0      0      1      1      2      2      3      3
//       （第几个 (token,head) 对，行优先）
//   pack_idx:         0      1      0      1      0      1      0      1
//       （该 (token,head) 对内第几个 pack）
//   token_idx:        0      0      0      0      1      1      1      1
//       （token_head_idx / num_heads，token 变化慢）
//   head_idx:         0      0      1      1      0      0      1      1
//       （token_head_idx % num_heads，head 变化快）
//   pack_offset:      0      4      0      4      0      4      0      4
//       （pack_idx * 4，该 pack 在 head_size 中的起始元素偏移）
//   head_offset:      0      0      8      8     16     16     24     24
//       （token_idx * num_heads * head_size + head_idx * head_size，
//        该 (token,head) 对在 output 展平数组中的起始偏移）
//
// Grid:  ((total_threads + 127) / 128, 1, 1)
// Block: (128, 1, 1)
// source: LeetCUDA/kernels/openai-triton/merge-attn-states/cuda_merge_attn_states.cu
__global__ void merge_attn_states(
    float *output,              // [num_tokens, num_heads, head_size]
    const float *prefix_output, // [num_tokens, num_heads, head_size]
    const float *prefix_lse,    // [num_heads, num_tokens]
    const float *suffix_output, // [num_tokens, num_heads, head_size]
    const float *suffix_lse,    // [num_heads, num_tokens]
    int num_tokens, int num_heads, int head_size) {

  constexpr int kNumThreads = 128;
  constexpr int kPackSize = 16 / sizeof(float); // 4 floats = 128-bit
  using pack_t = uint4;

  // 每个 (token, head) 对需要 threads_per_head 个线程覆盖 head_size 个元素
  const int threads_per_head = head_size / kPackSize;
  // 实际有效总线程数，超过这个数的线程会被忽略，block是按照kNumThreads=128
  // 来分配的，那么最后一个block的线程数可能会超过实际需要的线程数
  const int total_threads = num_tokens * num_heads * threads_per_head;

  const int global_idx = blockIdx.x * kNumThreads + threadIdx.x;
  if (global_idx >= total_threads)
    return;

  // global_idx → (token_head_idx, pack_idx) → (token_idx, head_idx, pack_offset)
  // token_head_idx: 第几个 (token, head) 对，行优先展平
  const int token_head_idx = global_idx / threads_per_head;
  // pack_idx: 该 (token, head) 对内第几个 pack，0 ~ threads_per_head-1
  const int pack_idx = global_idx % threads_per_head;

  // token_head_idx 分解为 (token_idx, head_idx)
  const int token_idx = token_head_idx / num_heads; // token 变化慢（外维）
  const int head_idx = token_head_idx % num_heads;  // head 变化快（内维）

  // pack_offset: 该 pack 覆盖的元素在 head_size 中的起始偏移（0, 4, 8, ...）
  const int pack_offset = pack_idx * kPackSize;
  // head_offset: 该 (token, head) 对在 output 展平一维数组中的起始偏移
  const int head_offset = token_idx * num_heads * head_size + head_idx * head_size;

  // 定位到当前 (token, head) 的输出段起始
  const float *prefix_head = prefix_output + head_offset;
  const float *suffix_head = suffix_output + head_offset;
  float *output_head = output + head_offset;

  // LSE 布局 [num_heads, num_tokens]: lse[head_idx][token_idx]
  float p_lse = prefix_lse[head_idx * num_tokens + token_idx];
  float s_lse = suffix_lse[head_idx * num_tokens + token_idx];

  // inf → -inf: 空 attention 段 LSE 可能为 +inf，替换后权重退化为 0
  p_lse = isinf(p_lse) ? -INFINITY : p_lse;
  s_lse = isinf(s_lse) ? -INFINITY : s_lse;

  // Step 1: max 归一化（与 safe softmax 同理，防 exp 溢出）
  const float max_lse = fmaxf(p_lse, s_lse);
  p_lse -= max_lse;
  s_lse -= max_lse;

  // Step 2-3: 指数还原 → 混合比例 alpha, beta
  const float p_se = expf(p_lse);
  const float s_se = expf(s_lse);
  const float out_se = p_se + s_se;
  const float p_scale = p_se / out_se; // alpha = w_1 / (w_1 + w_2)
  const float s_scale = s_se / out_se; // beta  = w_2 / (w_1 + w_2)

  // Step 4: 逐元素加权合并 O = alpha * O_1 + beta * O_2
  // 128-bit 向量化: uint4 load → per-element FMA → uint4 store
  if (pack_offset < head_size) {
    pack_t p_pack =
        reinterpret_cast<const pack_t *>(prefix_head)[pack_idx];
    pack_t s_pack =
        reinterpret_cast<const pack_t *>(suffix_head)[pack_idx];
    pack_t o_pack;

#pragma unroll
    for (int i = 0; i < kPackSize; ++i) {
      const float p_v = reinterpret_cast<const float *>(&p_pack)[i];
      const float s_v = reinterpret_cast<const float *>(&s_pack)[i];
      const float o_v = p_v * p_scale + (s_v * s_scale); // FMA
      reinterpret_cast<float *>(&o_pack)[i] = o_v;
    }

    reinterpret_cast<pack_t *>(output_head)[pack_idx] = o_pack;
  }
}

// =============================================================================
// Phase 3: Reduce 类 Ops — Softmax / RMS Norm / Layer Norm
// =============================================================================
// 面试要点：
//   - Softmax 三种实现递进：naive（溢出）→ safe（2-pass，max 减法）→
//   online（1-pass，增量更新）
//   - Online Softmax 是 FlashAttention-2的数学基础
//   - RMS Norm vs Layer Norm：RMS 只需 1 次 reduce，Layer Norm 需要 2 次（mean
//   + variance）
//   - Per-token 设计：一个 block 处理一个 token，无需跨 block 同步

// =============================================================================
// Phase 3a: Softmax — 三级递进（面试核心考点）
// =============================================================================
// 面试常问：「Softmax 有哪些实现方式？各有什么优缺点？」
// 回答线索：naive(溢出) → safe → online

// ---- Level 1: 基础 Softmax（per-token，无 max 减法，数值不稳定）----
// grid(S*h/h, h), block(h), 一个 block 处理一个 token
// 问题：x 值很大时 exp(x) 溢出为 inf
template <const int kNumThreads = 256>
// Grid:  (S, 1, 1)，S=batch*seq_len, DISPATCH_SOFTMAX_F32_PER_TOKEN_KERNEL
// Block: (H, 1, 1)，由外层 dispatch 选择 H=32/64/128/256/512/1024，一个 block 处理一个 token
// source: LeetCUDA/kernels/softmax/softmax.cu
__global__ void softmax_per_token(float *x, float *y, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;

  float exp_val = (idx < N) ? expf(x[idx]) : 0.0f;
  float exp_sum = block_reduce_sum<kNumThreads>(exp_val);
  if (idx < N)
    y[idx] = exp_val / exp_sum;
}

// ---- Level 2: Safe Softmax（2-pass：先 max 再 exp，数值稳定）----
// 面试重点：为什么 Softmax 需要 Safe？
//   - expf 溢出阈值约 88.7：exp(88)≈1.65e38（接近 float32 上限 3.4e38），exp(89)≈4.5e38 已溢出为 inf
//   - 减去 max 后：exp(x - max) ≤ exp(0) = 1.0，永不超过 1
//   - 数学等价性：softmax(x) = softmax(x - c) 对任意常数 c 成立
//   - 代价：2 次 block reduce（先 max，再 sum），但仍 O(N/B) 高效
// Grid:  (S, 1, 1)
// Block: (H, 1, 1)，由外层 dispatch 选择 H=32/64/128/256/512/1024
// source: LeetCUDA/kernels/softmax/softmax.cu
template <const int kNumThreads = 256>
__global__ void safe_softmax_per_token(float *x, float *y, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;

  // Pass 1: block reduce max — 找最大值
  float val = (idx < N) ? x[idx] : -FLT_MAX;
  float max_val = block_reduce_max<kNumThreads>(val);

  // Pass 2: exp(x - max) → block reduce sum
  float exp_val = (idx < N) ? expf(x[idx] - max_val) : 0.0f;
  float exp_sum = block_reduce_sum<kNumThreads>(exp_val);

  if (idx < N)
    y[idx] = exp_val / exp_sum;
}

// ---- Level 3: Online Safe Softmax（FlashAttention-2的数学基础）----
// 面试重点：
// 两种使用场景（公式不同，但结果等价）：
//   1) 单元素增量更新 (online update，处理新元素 x_i)：
//      m_new = max(m_old, x_i)
//      d_new = d_old * exp(m_old - m_new) + exp(x_i - m_new)
//   2) 二元合并 (binary merge，合并两个部分累加器，warp/block reduce 使用)：
//      m = max(m1, m2) safe softmax用的max值
//      d = d1*exp(m1-m) + d2*exp(m2-m) softmax用的分母值
//      当 m1≥m2 时退化为 d1 + d2*exp(m2-m1)（d_bigger + d_smaller*exp(m_smaller - m_bigger)）
// 算法来源: "Online normalizer calculation for softmax" (arXiv:1805.02867)
// Warp Reduce for Online Softmax — binary merge of two partial (m,d) accumulators.
//
// 不同于单元素增量更新公式（m_new = max(m_old, x_i); d_new = d_old*exp(m_old-m_new) + exp(x_i-m_new)），
// warp reduce 中每次合并的是两个**已各自归一化到各自 max 的部分累加器**：
//   (m1,d1): max 和 Σexp(x_j - m1)，覆盖集合 S1
//   (m2,d2): max 和 Σexp(x_k - m2)，覆盖集合 S2
//
// 合并公式（对称，不依赖"新/旧"概念）：
//   m = max(m1, m2)
//   d = d1*exp(m1-m) + d2*exp(m2-m)   ← m1≥m2 时退化为 d1 + d2*exp(m2-m1)
//
// 该操作满足结合律 → XOR butterfly 任意归约顺序均保证全局正确结果。
struct __align__(8) MD {
  float m; // running max
  float d; // running denominator (sum of exp(x - max))
};

template <const int kWarpWidth = kWarpSize>
__device__ __forceinline__ MD warp_reduce_md(MD md1) {
#pragma unroll
  for (int mask = kWarpWidth >> 1; mask >= 1; mask >>= 1) {
    MD md2;
    md2.m = __shfl_xor_sync(0xffffffff, md1.m, mask, kWarpWidth);
    md2.d = __shfl_xor_sync(0xffffffff, md1.d, mask, kWarpWidth);

    MD b_m = (md1.m > md2.m) ? md1 : md2; // max
    MD s_m = (md1.m > md2.m) ? md2 : md1;

    // b_m.d 无需 rescale（其基准 max 已是全局 max），
    // s_m.d 需 rescale：exp(s_m.m - b_m.m)
    md1.d = b_m.d + s_m.d * __expf(s_m.m - b_m.m);
    md1.m = b_m.m;
  }
  return md1;
}

// 注意：这里默认一个 block 处理一个 token；边界线程的 d=0 只参与归约，不会写回 y
// Grid:  (S, 1, 1)
// Block: (H, 1, 1)，由外层 dispatch 选择 H=32/64/128/256/512/1024
// source: LeetCUDA/kernels/softmax/softmax.cu
template <const int kNumThreads = 256>
__global__ void online_safe_softmax_per_token(const float *x, float *y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * kNumThreads + threadIdx.x;
  const int kNumWarps = (kNumThreads + kWarpSize - 1) / kWarpSize;
  __shared__ MD shared[kNumWarps];

  float val = (idx < N) ? x[idx] : -FLT_MAX;
  int warp = tid / kWarpSize;
  int lane = tid % kWarpSize;

  // 初始化：每个线程持有一个 (max, denom) 对
  MD md;
  md.m = idx < N ? val : -FLT_MAX;
  md.d = idx < N ? 1.0f : 0.0f;

  // 第一级规约：warp_reduce_md 在归约中自动更新 m 和 d
  md = warp_reduce_md<kWarpSize>(md);

  if (lane == 0)
    shared[warp] = md;
  __syncthreads();

  // 第二级归约：每个 warp 结果再做一次 warp_reduce_md（复用 block_reduce 模式）
  md = lane < kNumWarps ? shared[lane] : MD{-FLT_MAX, 0.0f};
  md = warp_reduce_md<kWarpSize>(md); // 用kWarpSize确保每个lane都能拿到最终结果

  // 用全局 max 和 denom 做最终 softmax
  float d_inv = __fdividef(1.0f, md.d);
  // 边界线程即使看到 d=0 的填充值，也不会走到写回路径
  if (idx < N) {
    y[idx] = __expf(val - md.m) * d_inv;
  }
}

// =============================================================================
// Phase 3b: RMS Normalization（1-pass reduce）
// =============================================================================
// 面试要点：
//   - RMS Norm: y = (x / rms(x)) * g, 1/rms(x) = rsqrt(mean(x²))
//   - 只需 1 次 block reduce（sum of squares），比 Layer Norm 少 1 次同步
//   - Llama 系列使用 RMS Norm
//   - grid(N, K/K), block(K)：一行一个 block
// Grid:  (N, 1, 1)，N=batch*seq_len，每行一个 block
// Block: (128, 1, 1)，kNumThreads=K=128（K>128 时调整模板参数）
// source: LeetCUDA/kernels/rms-norm/rms_norm.cu
template <const int kNumThreads = 128>
__global__ void rms_norm(float *x, float *y, float g, int N, int K) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const float epsilon = 1e-5f;

  __shared__ float s_variance;
  float value = (idx < N * K) ? x[idx] : 0.0f;
  float variance = value * value;
  variance = block_reduce_sum<kNumThreads>(variance);
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + epsilon); // 1/rms(x)
  __syncthreads();
  if (idx < N * K)
    y[idx] = (value * s_variance) * g;
}

// RMS Norm + float4
// Grid:  (N, 1, 1)
// Block: (32, 1, 1)，128/4=32；对应一行 K 元素按 4 个一组交给 32 个线程处理
// 注意：该版本默认 K 按 4 对齐，且输入/输出地址满足 float4 对齐
// source: LeetCUDA/kernels/rms-norm/rms_norm.cu
template <const int kNumThreads = 128 / 4>
__global__ void rms_norm_vec4(float *x, float *y, float g, int N, int K) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  const float epsilon = 1e-5f;

  __shared__ float s_variance;
  float4 reg_x = FLOAT4(x[idx]);
  float variance = (idx < N * K) ? (reg_x.x * reg_x.x + reg_x.y * reg_x.y +
                                    reg_x.z * reg_x.z + reg_x.w * reg_x.w)
                                 : 0.0f;
  variance = block_reduce_sum<kNumThreads>(variance);
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + epsilon);
  __syncthreads();
  float4 reg_y;
  reg_y.x = reg_x.x * s_variance * g;
  reg_y.y = reg_x.y * s_variance * g;
  reg_y.z = reg_x.z * s_variance * g;
  reg_y.w = reg_x.w * s_variance * g;
  if (idx < N * K)
    FLOAT4(y[idx]) = reg_y;
}

// =============================================================================
// Phase 3c: Layer Normalization（2-pass reduce）
// =============================================================================
// 面试要点：
//   - Layer Norm: y = ((x - mean) / std) * g + b, std = sqrt(variance)，variance = mean((x - mean)²)
//   - 需要 2 次 block reduce：先 mean（sum/K），再 variance（sum((x-mean)²)/K）
//   - 两次 __syncthreads 必须到位，否则 s_mean 未对所有线程可见就计算 variance
// Grid:  (N, 1, 1)，一行一个 block
// Block: (128, 1, 1)，kNumThreads=K=128
// source: LeetCUDA/kernels/layer-norm/layer_norm.cu
template <const int kNumThreads = 128>
__global__ void layer_norm(float *x, float *y, float g, float b, int N, int K) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const float epsilon = 1e-5f;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float value = (idx < N * K) ? x[idx] : 0.0f;

  // Pass 1: compute mean
  float sum = block_reduce_sum<kNumThreads>(value);
  if (tid == 0)
    s_mean = sum / (float)K;
  __syncthreads(); // 必须等待 s_mean 对所有线程可见

  // Pass 2: compute variance = (x - mean)²
  float variance = (value - s_mean) * (value - s_mean);
  variance = block_reduce_sum<kNumThreads>(variance);
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + epsilon); // 1/std
  __syncthreads(); // 必须等待 s_variance 对所有线程可见

  if (idx < N * K)
    y[idx] = ((value - s_mean) * s_variance) * g + b;
}

// Layer Norm + float4
// Grid:  (N, 1, 1)
// Block: (32, 1, 1)，128/4=32；对应一行 K 元素按 4 个一组交给 32 个线程处理
// 注意：该版本默认 K 按 4 对齐，且输入/输出地址满足 float4 对齐
// source: LeetCUDA/kernels/layer-norm/layer_norm.cu
template <const int kNumThreads = 128 / 4>
__global__ void layer_norm_vec4(float *x, float *y, float g, float b, int N, int K) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  const float epsilon = 1e-5f;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float4 reg_x = FLOAT4(x[idx]);
  float value = (idx < N * K) ? (reg_x.x + reg_x.y + reg_x.z + reg_x.w) : 0.0f;

  float sum = block_reduce_sum<kNumThreads>(value);
  if (tid == 0)
    s_mean = sum / (float)K;
  __syncthreads();

  float4 reg_x_hat;
  reg_x_hat.x = reg_x.x - s_mean;
  reg_x_hat.y = reg_x.y - s_mean;
  reg_x_hat.z = reg_x.z - s_mean;
  reg_x_hat.w = reg_x.w - s_mean;
  float variance = reg_x_hat.x * reg_x_hat.x + reg_x_hat.y * reg_x_hat.y +
                   reg_x_hat.z * reg_x_hat.z + reg_x_hat.w * reg_x_hat.w;
  variance = block_reduce_sum<kNumThreads>(variance);
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + epsilon);
  __syncthreads();

  float4 reg_y;
  reg_y.x = reg_x_hat.x * s_variance * g + b;
  reg_y.y = reg_x_hat.y * s_variance * g + b;
  reg_y.z = reg_x_hat.z * s_variance * g + b;
  reg_y.w = reg_x_hat.w * s_variance * g + b;
  if (idx < N * K)
    FLOAT4(y[idx]) = reg_y;
}

// =============================================================================
// Phase 4: RoPE — 旋转位置编码（Rotary Position Embedding）
// =============================================================================
// 面试要点：
//   - RoPE 数学公式: 对每对相邻维度做 2D 旋转
//     [x1']   [cos(θ)  -sin(θ)] [x1]
//     [x2'] = [sin(θ)   cos(θ)] [x2]
//   - θ_i = 1 / (theta^(2i/d)), theta=10000.0f（Llama 风格）
//   - token_pos = idx / N: token 在序列中的位置
//   - token_idx = idx % N: token 内的维度对索引
//   - 输入 [seq_len, hidden_size], 输出同形状

// Grid:  ((seq_len * N + 255) / 256, 1, 1)
// Block: (256, 1, 1)
// source: LeetCUDA/kernels/rope/rope.cu
__global__ void rope(float *x, float *out, int seq_len, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float x1 = x[idx * 2];
  float x2 = x[idx * 2 + 1];
  int token_pos = idx / N; // 序列位置
  int token_idx = idx % N; // 维度对索引

  // 频率计算: θ_i = 1 / (10000^(2i/d))
  float exp_v = 1.0f / powf(10000.0f, 2 * token_idx / (N * 2.0f));
  float sin_v = sinf(token_pos * exp_v);
  float cos_v = cosf(token_pos * exp_v);

  // 2D 旋转
  float out1 = x1 * cos_v - x2 * sin_v;
  float out2 = x1 * sin_v + x2 * cos_v;
  out[idx * 2] = out1;
  out[idx * 2 + 1] = out2;
}

// =============================================================================
// Phase 5: Mat Transpose — 矩阵转置（Bank Conflict 专题）
// =============================================================================
// 面试要点（Bank Conflict 专题）：
//   - Shared memory 有 32 个 bank，每个 bank 4 bytes（32-bit）
//   - 同一 warp 的多个线程访问同一 bank 的不同地址 → bank conflict
//   - n-way bank conflict: n 个线程冲突 → 访问串行化为 n 次
//   - 解决方案：PAD（在每行末尾加 1 个元素，打破地址对齐）
//
// 转置的四步演进：
//   naive: 非合并写入（列优先写）→ 每个 warp 产生 32 次内存事务
//   shared: 写入 smem（行优先）→ 从 smem 读取（列优先）→ 合并写入 gmem
//   BCF:   smem 布局 [kWarpSize_S*4][kWarpSize_S+PAD] = [64][17]，PAD=1 加在第二维消除 bank conflict
//   merge_write: 进一步将 4 次 separate store 合并为 1 次 float4 store
// 注：本文件仅实现 Level 1(naive) 与 Level 4(BCF+merge_write)，Level 2/3 省略

// 每个线程处理 1 个元素，block(16,16)
// 读：x 按行优先访问，warp 的 32 线程访问 32 个连续地址 → 合并读取 ✓
// 写：y 按列优先写入，32 线程跨 row 行分散 → 非合并写入 ✗（32 次内存事务）
// Grid:  ((col + 15) / 16, (row + 15) / 16, 1)，每线程 1 元素
// Block: (16, 16, 1)
// source: LeetCUDA/kernels/mat-transpose/mat_transpose.cu
__global__ void mat_transpose(float *x, float *y, const int row, const int col) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;  // col in x
  const int r = blockIdx.y * blockDim.y + threadIdx.y;  // row in x
  if (r < row && c < col) {
    // x[r][c] → y[c][r]; x stride=col, y (transposed) stride=row
    y[c * row + r] = x[r * col + c];
  }
}

// Grid:  ((col + 15) / 16, (row + 63) / 64, 1)，每线程 4 元素(float4)
// Block: (16, 16, 1)
// 注意：该版本默认按 float4 打包写回；最适合 row 能按 4 对齐的场景
// source: LeetCUDA/kernels/mat-transpose/mat_transpose.cu
__global__ void mat_transpose_padded(
    float *x, float *y, const int row, const int col) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  constexpr int TILE = 16;
  constexpr int PAD = 1;
  // Bank conflict fix: 每行多 1 个元素，打破 32-bank 对齐
  __shared__ float tile[TILE * 4][TILE + PAD]; // 64x16

  // x 空间坐标; 每线程覆盖 4 行 (actual row = x_r * 4)
  const int x_c = blockIdx.x * TILE + tx;
  const int x_r = blockIdx.y * TILE + ty;

  if (x_r * 4 < row && x_c < col) {
    // Step 1: 4 rows × 1 col → smem (row-major);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      tile[ty * 4 + i][tx] = x[(x_r * 4 + i) * col + x_c];
    }
    __syncthreads();

    // Step 2: Transposed read from smem
    float4 reg_trans;
    reg_trans.x = tile[tx * 4 + 0][ty];
    reg_trans.y = tile[tx * 4 + 1][ty];
    reg_trans.z = tile[tx * 4 + 2][ty];
    reg_trans.w = tile[tx * 4 + 3][ty];

    // y 空间坐标: y_r = x 的列, y_c = x 的行
    const int y_r = blockIdx.x * TILE + ty;          // = x col
    const int y_c = (blockIdx.y * TILE + tx) * 4;    // = x row

    // Coalesced write: y[y_r][y_c .. y_c+3]
    FLOAT4(y[y_r * row + y_c]) = reg_trans;
  }
}

// =============================================================================
