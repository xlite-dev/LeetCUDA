#pragma once
#include "base.cuh"
#include <mma.h>
using namespace nvcuda;

// sgemm.cuh: Phase 7a SGEMM
// Phase 7: GEMM — 矩阵矩阵乘（GPU 最重要的算子，面试核心考点）
// =============================================================================
// 面试要点（GEMM 优化五层金字塔）：
//   Level 1 — Tiling（分块 + shared memory）：将数据从 HBM 搬到 SMEM 复用
//   Level 2 — Thread Tile（寄存器分块）：每个线程计算 TM×TN
//   个元素，提高计算密度 Level 3 — Vectorize（向量化访存）：float4/half2，减少
//   load/store 指令数 Level 4 — Tensor Core（MMA
//   m16n8k16）：硬件矩阵乘单元，warp 级指令 Level 5 — Warp Specialization +
//   TMA（WGMMA m64n128k16）：Hopper 异步执行
//
// 计算密度递进：
//   Level 1: AI ≈ B_K / (2×sizeof) ≈ 32/8 = 4 → 仍是 memory-bound
//   Level 2: AI ≈ TM×TN×B_K / (2×sizeof) ≈ 8×8×8/8 = 64 → compute-bound
//   Level 4: Tensor Core 提供硬件加速的 256 FMA/cycle/warp → 大幅提升吞吐

// =============================================================================
// Phase 7a: SGEMM（非 Tensor Core 路径）
// =============================================================================

// ---- Level 1: SGEMM — Block Tile 32×32 + K Tile 32 ----
// 最基础的 tiling 实现，演示 shared memory 的核心用法
// C = A x B, C[M, N] = A[M, K] x B[K, N]
// BM=BN=32, BK=32, block(32, 32)，一个线程计算 c 的一个元素
// Grid:  ((N + 31) / 32, (M + 31) / 32, 1)
// Block: (32, 32, 1), 1024 线程
// source: LeetCUDA/kernels/sgemm/sgemm.cu
__global__ void sgemm(float *a, float *b, float *c, int M, int N, int K) {
  constexpr int BM = 32; // vec 版: 32x4 = 128
  constexpr int BN = 32; // vec 版: 32x4 = 128
  constexpr int BK = 32;
  __shared__ float s_a[BM][BK], s_b[BK][BN]; //  32x32x4=4KB smem, float = 4 bytes

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int tid = threadIdx.y * blockDim.x + tx;

  // 线程到 smem 的映射：32×32 线程，每个线程加载 a 和 b 各 1 个元素
  // 技巧：一般来说 “/” 表示线程不是连续排布的，"%" 表示线程是连续排布的
  // 因此，在需要考虑连续访问的维度使用“%”，比如，连续的线程访问列方向连续的元素
  // A[M, K], M的stride=K, K的stride=1 → 线程连续访问 K 维度 → 用 %，
  // 线程不连续访问 M 维度 → 用 /;
  int load_smem_a_m = tid / 32; // row 0~31 由 32 线程加载;
  int load_smem_a_k = tid % 32; // col 0~31 由 32 线程加载;
  int load_smem_b_k = tid / 32; // row 0~31 由 32 线程加载;
  int load_smem_b_n = tid % 32; // col 0~31 由 32 线程加载;
  int load_gmem_a_m = by * BM + load_smem_a_m; // gmem row;
  int load_gmem_b_n = bx * BN + load_smem_b_n; // gmem col;

  float sum = 0.f; // 遍历完整的K，slice K;
  // 这里不用pragma unroll，因为K不是编译器常量，编译器无法展开循环
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    int load_gmem_a_k = bk * BK + load_smem_a_k; // A [M, K]
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_addr];
    int load_gmem_b_k = bk * BK + load_smem_b_k; // B [K, N]
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_addr];
    __syncthreads(); // 确保整个 smem tile 加载完毕

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      int comp_smem_a_m = load_smem_a_m; // vec 版: 0~127, 0, 1, 2, ... (连续)
      int comp_smem_b_n = load_smem_b_n; // vec 版: 0~127, 0, 4, 8, ... (间隔)
      sum += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
    }
    __syncthreads(); // 确保 smem 不会在下一轮加载时被覆盖
  }
  int store_gmem_c_m = load_gmem_a_m; // vec 版: 0~127, 0, 1, 2, ... (连续) [128x128]
  int store_gmem_c_n = load_gmem_b_n; // vec 版: 0~127, 0, 4, 8, ... (间隔)
  int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
  c[store_gmem_c_addr] = sum; // C [M, N] = A[M, K] x B[K, N]
}

// ---- Level 1+: SGEMM Vec4 — Block Tile 128×128 + K Tile 32 + Thread Tile 4×4 ----
// 在 Level 1 基础上引入两层优化：
//   1) float4 向量化加载：A/B 各用 1 条 128-bit load 取代 4 条 32-bit load
//   2) Thread Tile 4×4：每线程计算 16 个 C 元素，提升计算/访存比（AI 从
//   BK/2≈16 提升到 TM*TN*BK/2≈256），减少线程总数带来的同步开销
// C = A x B, C[M, N] = A[M, K] x B[K, N]，A/B 均 row-major
// BM=BN=128, BK=32, block(32, 32)=1024 线程，每线程负责 4×4=16 个 C 元素
//   1024 × 16 = 16384 = 128 × 128 ✓
//
// 线程到 4×4 tile 的映射（与加载映射解耦，独立计算更清晰）：
//   m_tile = tid / 32 (0~31)，每 tile 4 行 → 行 [m_tile*4, m_tile*4+3]，覆盖 0~127
//   n_tile = tid % 32 (0~31)，每 tile 4 列 → 列 [n_tile*4, n_tile*4+3]，覆盖 0~127
//
// 加载映射（每线程 4 个元素，float4）：
//   A[128][32]: a_m = tid/8 (8 线程/行), a_k = (tid%8)*4 (4 列/线程) → 8×4=32 列 ✓
//   B[32][128]: b_k = tid/32 (32 线程/行), b_n = (tid%32)*4 (4 列/线程) → 32×4=128 列 ✓
//   row-major 下 A[m][k..k+3] 与 B[k][n..n+3] 均连续 → float4 load 合法
//
// ⚠ Bank Conflict 提示（面试加分点）：
//   s_b[32][128] 上 warp 内 32 线程按 stride=4 访问（tid%32 决定列 0,4,8,...,124）
//   → 每 4 个线程落同一 bank 不同地址 → 4-way bank conflict。生产代码可用
//   s_b[BK][BN+1] PAD 打散，这里保持最简布局便于讲解。
//
// Grid:  ((N + 127) / 128, (M + 127) / 128, 1)
// Block: (32, 32, 1), 1024 线程
// 假设：M/N 为 128 的倍数，K 为 32 的倍数（与 Level 1 naive 版一致的边界约定）
// source: LeetCUDA/kernels/sgemm/sgemm.cu (vec4 variant)
__global__ void sgemm_vec4(float *a, float *b, float *c, int M, int N, int K) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 32;
  __shared__ float s_a[BM][BK]; // 128*32*4 = 16KB, float = 4 bytes
  __shared__ float s_b[BK][BN]; // 32*128*4 = 16KB

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int tid = threadIdx.y * blockDim.x + tx; // 0~1023

  // 加载 A: 每线程加载 s_a[a_m][a_k..a_k+3] 共 4 个元素
  int load_smem_a_m = tid / 8;        // 0~127, 8 线程/行
  int load_smem_a_k = (tid % 8) * 4;  // 0,4,...,28
  // 加载 B: 每线程加载 s_b[b_k][b_n..b_n+3] 共 4 个元素
  int load_smem_b_k = tid / 32;       // 0~31, 32 线程/行
  int load_smem_b_n = (tid % 32) * 4; // 0,4,...,124

  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;

  // 4×4 Thread Tile 基址（独立于加载映射），这里compute索引的计算逻辑要和
  // load索引的计算逻辑分开，load/compute是可以独立索引的，理解这点很重要。
  // 目标C Tile为[BM,BN]=[128x128], 有32x32线程，则每个线程处理4x4 tile
  // 那么，就可以不重不漏地覆盖[32x4,32x4]=[128x128]的大小
  int comp_smem_a_m_base = (tid / 32) * 4; // 0,4,8,...,124
  int comp_smem_b_n_base = (tid % 32) * 4; // 0,4,8,...,124

  float sum[4][4] = {0.f};
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    int load_gmem_a_k = bk * BK + load_smem_a_k; // A [M, K]
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    FLOAT4(s_a[load_smem_a_m][load_smem_a_k]) = FLOAT4(a[load_gmem_a_addr]); // s_a [BM,BK]
    int load_gmem_b_k = bk * BK + load_smem_b_k; // B [K, N]
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    FLOAT4(s_b[load_smem_b_k][load_smem_b_n]) = FLOAT4(b[load_gmem_b_addr]); // s_b [BK,BN]
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      // 每次迭代加载 4 个 A 元素 + 4 个 B 元素，再做 4×4=16 次 FMA
      float a_vals[4] = {s_a[comp_smem_a_m_base + 0][k],
                         s_a[comp_smem_a_m_base + 1][k],
                         s_a[comp_smem_a_m_base + 2][k],
                         s_a[comp_smem_a_m_base + 3][k]};
      float b_vals[4] = {s_b[k][comp_smem_b_n_base + 0],
                         s_b[k][comp_smem_b_n_base + 1],
                         s_b[k][comp_smem_b_n_base + 2],
                         s_b[k][comp_smem_b_n_base + 3]};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          sum[i][j] += a_vals[i] * b_vals[j];
        }
      }
    }
    __syncthreads();
  }

  // 存储 4×4：每行 4 个元素连续 → 可用 float4 store（要求 N 为 4 的倍数以保证对齐）
  int store_gmem_c_m = by * BM + comp_smem_a_m_base;
  int store_gmem_c_n = bx * BN + comp_smem_b_n_base;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int store_gmem_c_addr = (store_gmem_c_m + i) * N + store_gmem_c_n;
    float4 reg_c;
    reg_c.x = sum[i][0];
    reg_c.y = sum[i][1];
    reg_c.z = sum[i][2];
    reg_c.w = sum[i][3];
    FLOAT4(c[store_gmem_c_addr]) = reg_c;
  }
}

// =============================================================================
// Phase 7a+: SGEMM TF32 — WMMA m16n16k8 + dynamic shared memory + 2-stage pipeline
// =============================================================================
// 面试要点（TF32 Tensor Core 路径）：
//   - TF32 精度：mantissa 10bit（vs FP32 23bit），~3 位十进制有效数字
//   - WMMA API：CUDA 提供的 warp-level matrix multiply 抽象，简化 Tensor Core 编程
//   - m16n16k8：Ampere Tensor Core 的 TF32 基本 tile（16×16×8）
//   - 256 线程/block = 8 warps，warp tiling 4×2（warp_m=0~3, warp_n=0~1）
//   - K_STAGE=2 双缓冲 + cp.async 异步加载，掩盖 GMEM 延迟
//   - Dynamic SMEM：16KB（s_a 8KB + s_b 8KB），远低于 Ampere 48KB 上限
//
// 计算密度：
//   BM×BN×BK / (2×sizeof) = 128×128×8 / 8 = 16384 → compute-bound
//   8 个 MMA/warp × 8 warps = 64 个 MMA/block，每个 MMA 64 个 TF32 MAC
//   总计 4096 个 TF32 MAC/cycle/block（理论峰值）

// ---- FP32 → TF32 格式转换 kernel ----
// TF32 格式：sign 1bit + exponent 8bit + mantissa 10bit（共 19bit，存储为 32bit float）
// wmma::__float_to_tf32() 将 FP32 舍入到 TF32 精度（mantissa 23bit → 10bit）
// 用于测试前将 A/B 矩阵原地转换为 TF32 格式
// source: LeetCUDA/kernels/sgemm/sgemm_wmma_tf32_stage.cu
__global__ void f32x4_tf32x4_kernel(float *x, float *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = wmma::__float_to_tf32(reg_x.x);
    reg_y.y = wmma::__float_to_tf32(reg_x.y);
    reg_y.z = wmma::__float_to_tf32(reg_x.z);
    reg_y.w = wmma::__float_to_tf32(reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  }
}

// ---- SGEMM TF32 (WMMA m16n16k8, dynamic smem, 2-stage pipeline) ----
// C = A x B, C[M, N] = A[M, K] x B[K, N]，A/B/C 均为 row-major
// BM=BN=128, BK=8, block(256)，8 warps，warp tiling 4×2
// Grid: ((N + 127) / 128, (M + 127) / 128)
// Dynamic SMEM: 20736 bytes（s_a 12KB + s_b 8.5KB，含 bank-conflict padding）
// 假设：M/N 为 128 的倍数，K 为 8 的倍数（简化边界处理）
// source: LeetCUDA/kernels/sgemm/sgemm_wmma_tf32_stage.cu
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 8,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 2, const int WARP_TILE_N = 4,
          const int A_PAD = 4, const int B_PAD = 4, const int K_STAGE = 2>
__global__ void sgemm_tf32(float *A, float *B, float *C, int M, int N, int K) {
  // 256 线程（8 warps）per block
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = (K + WMMA_K - 1) / WMMA_K;
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16×4×2 = 128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16×2×4 = 128
  constexpr int BK = WMMA_K;                             // 8

  // Dynamic shared memory（调用时指定大小：16384 bytes）
  extern __shared__ float smem_tf32[];
  float *s_a = smem_tf32;
  float *s_b = smem_tf32 + K_STAGE * BM * (BK + A_PAD);
  constexpr int s_a_stage_offset = BM * (BK + A_PAD); // 1024 floats
  constexpr int s_b_stage_offset = BK * (BN + B_PAD); // 1024 floats

  // 线程索引与 warp 分配
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / 32;   // 0~7，block 内 warp 编号
  const int warp_m = warp_id / 2; // 0,1,2,3（warp tiling M 维度）
  const int warp_n = warp_id % 2; // 0,1（warp tiling N 维度）

  // ---- GMEM → SMEM 加载索引（每线程加载 4 个 float，float4 向量化）----
  // s_a[BM][BK] = s_a[128][8]，按行加载，每行 8 个元素
  //   每线程加载 4 个元素 → 每行需要 2 个线程 → 128 行需要 256 线程 ✓
  int load_smem_a_m = tid / 2;                // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 4; // col 0 或 4
  // s_b[BK][BN] = s_b[8][128]，按行加载，每行 128 个元素
  //   每线程加载 4 个元素 → 每行需要 32 个线程 → 8 行需要 256 线程 ✓
  int load_smem_b_k = tid / 32;       // row 0~7
  int load_smem_b_n = (tid % 32) * 4; // col 0,4,8,...,124

  // GMEM 全局索引
  int load_gmem_a_m = by * BM + load_smem_a_m; // C 的行
  int load_gmem_b_n = bx * BN + load_smem_b_n; // C 的列

  // ---- WMMA 累加器碎片（WARP_TILE_M × WARP_TILE_N = 2×4 = 8 个 m16n16k8 tile）----
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      C_frag[WARP_TILE_M][WARP_TILE_N];

#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0f);
    }
  }

  // ---- SMEM base ptr（cp.async 需要 uint32_t smem addr）----
  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

  // ---- Pipeline: 预加载前 K_STAGE-1 个 tile ----
#pragma unroll
  for (int k = 0; k < K_STAGE - 1; ++k) {
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    uint32_t load_smem_a_ptr =
        smem_a_base_ptr +
        (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
            sizeof(float);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr =
        smem_b_base_ptr +
        (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
            sizeof(float);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
  }
  CP_ASYNC_COMMIT_GROUP();
  CP_ASYNC_WAIT_GROUP(K_STAGE - 2); // K_STAGE=2 → wait_group(0)
  __syncthreads();

  // ---- Main loop: load + compute 流水线 ----
#pragma unroll
  for (int k = K_STAGE - 1; k < NUM_K_TILES; ++k) {
    int smem_sel = (k + 1) % K_STAGE;     // 当前计算用的 stage
    int smem_sel_next = k % K_STAGE;      // 下一轮加载用的 stage

    int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    // 异步加载下一轮数据到 smem_sel_next
    uint32_t load_smem_a_ptr =
        smem_a_base_ptr +
        (smem_sel_next * s_a_stage_offset +
         load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
            sizeof(float);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr =
        smem_b_base_ptr +
        (smem_sel_next * s_b_stage_offset +
         load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
            sizeof(float);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    CP_ASYNC_COMMIT_GROUP();

    // TF32 WMMA 碎片
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        B_frag[WARP_TILE_N];

    // 从 SMEM 加载 A 碎片（每个 warp 加载 WARP_TILE_M=2 个 m16n8k16 tile）
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m =
          warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M; // warp 在 SMEM 中的行偏移
      float *load_smem_a_frag_ptr =
          s_a + smem_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD);
      wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD);
    }

    // 从 SMEM 加载 B 碎片（每个 warp 加载 WARP_TILE_N=4 个 m16n8k16 tile）
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n =
          warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N; // warp 在 SMEM 中的列偏移
      float *load_smem_b_frag_ptr =
          s_b + smem_sel * s_b_stage_offset + warp_smem_b_n;
      wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
    }

    // MMA 计算：C_frag[i][j] += A_frag[i] × B_frag[j]
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
      }
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();
  }

  // ---- 处理尾部：最后 K_STAGE-1 个 tile 已加载但未计算 ----
  // 确保所有 cp.async 完成
  if ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }
  // 计算剩余的 K_STAGE-1 个 tile
  {
#pragma unroll
    for (int k = 0; k < K_STAGE - 1; ++k) {
      const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);

      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major>
          A_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major>
          B_frag[WARP_TILE_N];

#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        float *load_smem_a_frag_ptr =
            smem_tf32 + stage_sel * s_a_stage_offset +
            warp_smem_a_m * (BK + A_PAD);
        wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD);
      }

#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
        float *load_smem_b_frag_ptr =
            s_b + stage_sel * s_b_stage_offset + warp_smem_b_n;
        wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
      }

#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
        }
      }
    }
  }

  // ---- Store: 将 WMMA 累加器写回 GMEM ----
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int store_gmem_c_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) +
                           i * WMMA_M;
      int store_gmem_c_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) +
                           j * WMMA_N;
      wmma::store_matrix_sync(C + store_gmem_c_m * N + store_gmem_c_n,
                              C_frag[i][j], N, wmma::mem_row_major);
    }
  }
}

// =============================================================================
