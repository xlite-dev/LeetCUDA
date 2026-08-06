#pragma once
#include "base.cuh"
// hgemm.cuh: Phase 7b-d HGEMM (MMA/Swizzle/CuTe/WGMMA/TMA_MMA_WS)

// Phase 7b: HGEMM — Tensor Core 路径（MMA m16n8k16 + WGMMA m64n128k16）
// =============================================================================
// 面试要点：
//   - MMA (Matrix Multiply-Accumulate): Ampere+ 的 Tensor Core 指令
//   - m16n8k16 含义：M=16, N=8, K=16 的矩阵乘，结果 [16×8] = [16×16]×[16×8]
//   - ldmatrix: 从 shared memory 加载 16×16 的矩阵片段到寄存器（4 条 32-bit
//   寄存器）
//   - Multistage Pipeline: s2/s3/s4 个 stage，用 cp.async 异步加载下一批数据
//   - Block Swizzle: 在 grid 维度做 swizzle，改善 L2 cache locality
//
// ★ TN 布局详解（面试高频考点）：
//   TN 命名约定：来自 BLAS 的 op(A) × op(B) 语义
//     BLAS 源自 Fortran，默认列优先（column-major）存储
//     N = Normal（列优先，BLAS 原生格式）
//     T = Transposed（行优先，相对 BLAS 来说是"转置过的"）
//     第一个字母 → A 的 op，第二个字母 → B 的 op
//     所以 TN 表示：A 是行优先（相对 BLAS=Transposed），B 是列优先（相对
//     BLAS=Normal） 即：C = op(A) × op(B) = A^T × B? 不对！ 在 row-major
//     视角下：TN = A row-major [M×K], B^T row-major [N×K]（等价于 B col-major [K×N]） 在 cuBLAS
//     调用中：cublasGemmEx(..., CUBLAS_OP_T, CUBLAS_OP_N, ...)
//       T on A: BLAS 把 row-major 的 A 视为 A^T，传 T 表示"转置回去"
//       N on B: B 已经是 BLAS 原生的 col-major，无需转置
//
//   记忆口诀：TN = A行(T) B列(N)，第一字母 A 第二字母 B
//     T = row-major（行优先，对 BLAS 来说是 transposed）
//     N = col-major（列优先，BLAS native = normal）
//
//   LeetCUDA _nn 布局对比（A/B 均 row-major 自然存储）: C[M×N] = A[M×K] × B[K×N]
//     - A: row-major [M, K], B: row-major [K, N]
//     - 按 N=col-major/T=row-major 约定，二者 BLAS 视角均为 T → cuBLAS 等效 (T,T)
//     - 问题：ldmatrix 默认加载 col-major，B 是 row-major 需要 .trans
//
//   TN 布局: C[M×N] = A[M×K] × B[K×N]，B 以 B^T=[N×K] row-major 存储（A 行优先，B 列优先）
//     - A [M×K]: row-major → 全局索引 A[m*K + k]，smem s_a[BM][BK]
//     - B^T [N×K]: row-major → 全局索引 B[n*K+k] 即访问原 B 元素 (k,n)（⚠ 内维连续的是 K）
//     - 优势：B^T 已是 row-major，ldmatrix 无需 .trans，天然匹配 MMA row.col
//   MMA 指令: mma.sync.aligned.m16n8k16.row.col
//     - row.col: A 输入 row-major，B 输入 col-major → 与 TN 布局天然匹配
//
// WGMMA (Warp Group MMA, Hopper+):
//   - warpgroup 级指令（128 threads = 4 warps），异步执行
//   - m64n128k16: 一次处理 64×128×16 的矩阵乘（总 tile 量 131072，是 MMA m16n8k16 的 64 倍）
//   - Warp Specialization: Producer(128 threads) 做 TMA 搬运, Consumer(128
//   threads) 做计算
//   - TMA: 硬件 DMA，~零寄存器开销，支持 1D~5D 寻址

// =============================================================================
// Phase 7b-2: HGEMM MMA — m16n8k16 + multistage pipeline + TN 布局（统一循环版）
// =============================================================================
// 面试重点 — Tile Hierarchy:
//   MMA Atom:         m16n8k16（1 条 MMA 指令处理的最小 tile）
//   MMA Tile (more warps):  2×4=8 个 MMA atom → [2×16, 8×4]=[32,32]
//   VAL Tile (more values): 4×4=16 expand → [32×4, 32×4]=[128,128]
//   实际: kMmaTileM=2, kMmaTileN=4, kValTileM=4, kValTileN=4
//         → BM=16×2×4=128, BN=8×4×4=128, Warps=2×4=8, Threads=8×32=256
//
// ★ TN 布局（T=A 行优先，N=B 列优先）：
//   - A[M][K]: row-major → 全局索引 A[m*K + k], smem s_a[BM][BK]
//   - B[K][N]: col-major（等价于 B^T[N][K] row-major）→ 全局索引 B[n*K+k]
//   - ldmatrix A: 用 x4（非转置），A row-major 原生匹配
//   - ldmatrix B: 用 x2（非转置），B^T row-major 逐行加载 = B 的列，天然匹配 MMA row.col
//   - MMA 指令: mma.sync.aligned.m16n8k16.row.col → 天然匹配 TN 布局
//
// ★ 统一循环设计（k 从 0 开始，消除尾端重复代码）：
//   - k 从 0 开始：每次迭代 k 直接对应 tile k，sel = k % kStages（零偏移）
//   - 加载语义为"预取未来"：迭代 k 加载 tile (k+kStages-1) 供后续使用
//   - cp.async 条件化：仅当 k+kStages-1 < NUM_K_TILES 时加载
//   - WAIT_GROUP 自适应：满载期用 kStages-2，尾部排空用 0
//
// ★ Block Swizzle：把 C 的逻辑 tile 布局改写为紧凑的二维发射窗口，增加 A/B tile 的 L2 reuse 机会。
//   C tile 坐标为 (by, bx)，by 沿 M 方向，bx 沿 N 方向；一个 bx 读取同一块 B^T[bx*BN:(bx+1)*BN, :]
//   （TN 布局中 B^T[N][K] 为 row-major），一个 by 读取同一块 A[by*BM:(by+1)*BM, :]。
//   下图只画 4 个逻辑上相邻的 CTA；SM0~SM3 仅表示可能的发射序列，不是硬件 SM 绑定或调度保证。
//
//   No Swizzle: grid = (tiles_n, tiles_m, 1)，blockIdx.x = bx。
//   C-Matrix（同一 by 行；CTA 先在很宽的 N 方向范围内展开）：
//
//       N / bx →  0          1          2          3        ...
//     M  ↓      ┌────────┬────────┬────────┬────────┐
//     by = 0    │  SM0   │  SM1   │  SM2   │  SM3   │  ...
//               │ A 0,B0 │ A 0,B1 │ A 0,B2 │ A 0,B3 │
//               └────────┴────────┴────────┴────────┘
//
//   同一 by 的 CTA 复用 A[by]，但分别读取 B^T 0、B^T 1 等不同 B tile。只有 scheduler
//   推进到下一条 by 行、再次遇到相同 bx 时，才有机会复用 B；当 tiles_n 很大时，这段时间内
//   L2 更容易被其他 B tile 挤占。
//
//   Thread Block Swizzle: 先把 N 切成 S 个连续窗口；一个 z slice 只含 grid_x 个 bx。
//   下图用 grid_x=2 展示一个窄窗口，scheduler 更早进入下一条 by 行：
//
//       N / local x →  0          1
//     M  ↓          ┌────────┬────────┐
//     by = 0        │  SM0   │  SM1   │  → A 0; B^T 0, B^T 1
//                   ├────────┼────────┤
//     by = 1        │  SM2   │  SM3   │  → A 1; B^T 0, B^T 1
//                   └────────┴────────┘
//
//   对于这个 2x2 窗口：同一行横向复用 A（SM0/SM1、SM2/SM3），同一列纵向复用 B
//   （SM0/SM2、SM1/SM3）。真实 grid_x 不必等于 2，但缩小它会缩短再次访问相同 bx 的距离。
//   此优化只提高 scheduler 在相近时间执行这些 CTA、命中 L2 的机会，不保证 CTA 的 z 顺序、
//   缓存驻留或 L2 hit。
//
//   Launch（kBlockSwizzle=false，当前默认路径）：
//     grid = (ceil(N / BN), ceil(M / BM), 1)，blockIdx.x 直接就是 N-tile 编号 bx。
//   Launch（kBlockSwizzle=true，需由 launch 端显式构造 3D grid）：
//     tiles_n = ceil(N / BN), S = ceil(N / 2048)
//     grid = (ceil(tiles_n / S), ceil(M / BM), S)
//     bx = blockIdx.z * grid.x + blockIdx.x，col_start = bx * BN。
//   例如 N=8192、BN=128：tiles_n=64，S=4，grid.x=16；z=0/1/2/3 分别覆盖
//   bx=0..15/16..31/32..47/48..63，即 columns [0,2048)/[2048,4096)/[4096,6144)/[6144,8192)。
//   grid.x*grid.z 可能产生 bx>=tiles_n 的冗余 CTA。当前 kernel 仍要求 M/N/K 分别按 BM/BN/BK
//   对齐；ceil 只保证逻辑 tile 编号不遗漏，并不使部分尾 tile 变为安全。
// Block: (256, 1, 1)，8 warps
// source: LeetCUDA/kernels/hgemm/mma/basic/hgemm_mma_stage_tn.cu
// =============================================================================
template <const int kMmaM = 16,             // MMA atom M dim (m16n8k16)
          const int kMmaN = 8,              // MMA atom N dim
          const int kMmaK = 16,             // MMA atom K dim, also BK tile = kMmaK
          const int kMmaTileM = 2,          // warps along M, 2 → warp tile M = 32
          const int kMmaTileN = 4,          // warps along N, 4 → warp tile N = 32
          const int kValTileM = 4,          // value-repeat along M, BM = 16*2*4 = 128
          const int kValTileN = 4,          // value-repeat along N, BN = 8*4*4 = 128
          const int kStages = 3,            // cp.async pipeline depth
          const int kBlockSwizzle = 0>      // 1 enables 3D grid swizzle for L2 locality
__global__ void __launch_bounds__(256)
    hgemm_mma_stages_tn(half *A, half *B, half *C, int M, int N, int K) {
  static_assert(kBlockSwizzle == 0 || kBlockSwizzle == 1, "kBlockSwizzle must be 0 or 1");
  static_assert(kStages >= 2, "kStages must be >= 2 for cp.async pipeline");
  // Block Swizzle: 0 时 bx=blockIdx.x；1 时将 (blockIdx.z, blockIdx.x)
  // 线性化为原始 N-tile 编号 bx=z*gridDim.x+x。by 始终是 M-tile 编号。
  const int bx = ((int)kBlockSwizzle) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  constexpr int BM = kMmaM * kMmaTileM * kValTileM; // 16*2*4=128
  constexpr int BN = kMmaN * kMmaTileN * kValTileN; // 8*4*4=128
  constexpr int BK = kMmaK;                         // 16

  // Dynamic shared memory: kStages 个 stage 的 A 和 B
  // TN 布局: s_a[BM][BK]=[128][16](A row-major), s_b[BN][BK]=[128][16](B^T
  // row-major，即 B col-major [K×N] 在 smem 中按 B^T[N×K] 存储)
  // 原始实现会按配置决定是否给 A/B 的 K 维加 PAD；尤其 B 在 TN 布局下常见会额外
  // 加 B_PAD 来打散 bank 映射，避免按列访问时出现明显 bank conflict。这里先保留最简 PAD=0 版本。
  extern __shared__ half smem[];
  half *s_a = smem;
  half *s_b = smem + kStages * BM * BK;     // A 和 B 连续存放
  constexpr int s_a_stage_offset = BM * BK; // 128*16
  constexpr int s_b_stage_offset = BN * BK; // 128*16  ⚠ BN(128)×BK(16) = B^T row-major
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / kWarpSize; // 0~7
  const int lane_id = tid % kWarpSize; // 0~31
  // warp_m变化快(0->0,1->1), warp_n变化慢([0,1]->0,[2,3]->1,...), 因此，
  // 这种2x4的MMA(Warp) layout是按照col major的顺序来排列MMA0~MMA7的
  //                          N direction (warp_n)
  //                          0       1       2       3
  // M direction (warp_m)  0  MMA0    MMA2    MMA4    MMA6
  //                       1  MMA1    MMA3    MMA5    MMA7
  // MMA的排布方式不是唯一的，现在这样排是因为结合MMA/VAL Tile之后，刚好能覆盖整个
  // C Tile[128,128]。只要调整MMA/VAL Tile，这里的排布方式也可以跟着调整。要注意的
  // 是，MMA0-7逻辑上是可以认为是并行执行的，各自的计算结果累计加到对应的C Tile位置上。
  const int warp_m = warp_id % kMmaTileM; // 0,1（M 方向 2 个 warp）kMmaTileM = 2
  const int warp_n = warp_id / kMmaTileM; // 0,1,2,3（N 方向 4 个 warp）

  // 线程到 global memory 的映射（用于加载 A 和 B）共 256 个线程
  // TN 布局关键: A[m*K+k] 是 row-major, B^T[n*K+k] 是 row-major（内维连续的是 K）
  int load_smem_a_m = tid / 2;                 // 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;  // 0, 8
  int load_smem_b_n = tid / 2;                 // 0~127 → B^T 的 N 方向（row-major 的行）
  int load_smem_b_k = (tid % 2 == 0) ? 0 : 8;  // 0, 8  → B^T 的 K 方向（row-major 的列）
  int load_gmem_a_m = by * BM + load_smem_a_m; // C/A 全局行号 = M 方向的 tile 起始 + 线程偏移
  int load_gmem_b_n = bx * BN + load_smem_b_n; // C/B 全局列号 = N 方向的 tile 起始 + 线程偏移
  if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    return;

  // 8个Warps(MMAs)的排布是M方向2个，N方向4个，则MMA Atom一次性能处理的tile大小是[2*16,4*8]=[32,32].
  // CUDA中每个block能放的线程数是有上限的（warp数量有限），一般为4或者8个warps。为了能处理更大的C tile，
  // 需要把MMA Atom的tile再M方向和N方向各自重复4次，得到[128,128]的C block tile，这就是Value Tile的作用。
  // MMA Tile对应到cutlass cute中的TiledMMA的概念，Value Tile对应到cutlass cute中的PermuteMNK的概念。
  uint32_t RC[kValTileM][kValTileN][2] = {0}; // 初始化为 0

  // CVTA: 一次转换 smem 基地址，避免每次 cp.async 都做转换
  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

  // 预加载前 (kStages-1) 个 stage
  // TN 布局: A 的 gmem 索引用 m*K+k(row-major), B^T 用 n*K+k(row-major，即 B col-major [K×N])
#pragma unroll
  for (int k = 0; k < (kStages - 1); ++k) {
    int load_gmem_a_k = k * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k; // A: [m][k]
    uint32_t load_smem_a_ptr = (smem_a_base_ptr +
      (k * s_a_stage_offset + load_smem_a_m * BK + load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    int load_gmem_b_k = k * BK + load_smem_b_k;
    // B^T: [n][k] row-major（即 B[k][n] col-major）⚠
    int load_gmem_b_addr = load_gmem_b_n * K + load_gmem_b_k;
    uint32_t load_smem_b_ptr = (smem_b_base_ptr +
      (k * s_b_stage_offset + load_smem_b_n * BK + load_smem_b_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(kStages - 2); // 允许有 kStages-2 个group未完成
  __syncthreads();

  // 统一循环：k 从 0 开始，每次迭代负责 tile k（加载 + 计算合并为单循环）
  const int NUM_K_TILES = div_ceil(K, BK);
  // 此处不用pragma unroll，因为 K 不是编译期常量，因此NUM_K_TILES 也不是编译期常量
  for (int k = 0; k < NUM_K_TILES; ++k) {
    int smem_sel = k % kStages;                      // 计算 tile k 的 stage
    int smem_sel_next = (k + kStages - 1) % kStages; // 预加载目标 stage

    // 条件加载：预加载 tile (k+kStages-1) 供将来使用。因为在prefetch loop中已经
    // load了(kStages-1) 个 stage，因此这里是从 tile (k+kStages-1) 开始预加载
    // TN 布局: A 的 gmem 地址用 m*K+k（row-major），B^T 的 gmem 地址用 n*K+k
    // (row-major，内维连续的是 K)
    if (k + kStages - 1 < NUM_K_TILES) {
      int load_gmem_a_k = (k + kStages - 1) * BK + load_smem_a_k;
      int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k; // A: row-major [m][k]
      int load_gmem_b_k = (k + kStages - 1) * BK + load_smem_b_k;
      int load_gmem_b_addr = load_gmem_b_n * K + load_gmem_b_k; // B^T: row-major [n][k]

      uint32_t load_smem_a_ptr =
          (smem_a_base_ptr + (smem_sel_next * s_a_stage_offset +
                              load_smem_a_m * BK + load_smem_a_k) *
                                 sizeof(half));
      CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

      uint32_t load_smem_b_ptr =
          (smem_b_base_ptr + (smem_sel_next * s_b_stage_offset +
                              load_smem_b_n * BK + load_smem_b_k) *
                                 sizeof(half));
      CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
      CP_ASYNC_COMMIT_GROUP();
    }

    // ldmatrix: 从 smem_sel 加载 A 和 B 到寄存器
    // TN 布局关键: A 用 x4（非转置），因为 A 是 row-major; B 用 x2（非转置），smem 中
    // B^T 为 row-major，逐行加载即得 B 的列，天然匹配 col-major B
    uint32_t RA[kValTileM][4]; // [4][4], M方向4次repeat, A regs = 4 uint32_t per MMA
    uint32_t RB[kValTileN][2]; // [4][2], N方向4次repeat, B regs = 2 uint32_t per MMA

    // ldmatrix.x4: 加载 A 的 m16k16 片段（row-major A，非转置）
#pragma unroll
    for (int i = 0; i < kValTileM; ++i) {
      // {0,1} * (16 * 4) + i * 16 = {0,64} + {0,16,32,48} = {0,16,32,48,64,80,96,112}
      // 其中 warp_m {0,1}; warp_m_0 offsets {0,16,32,48}; warp_m_1 offsets {64,80,96,112}
      int warp_smem_a_m = warp_m * (kMmaM * kValTileM) + i * kMmaM;
      // {0,16,32,...,112} + {0~15} = {0~127}, 按照col-major的顺序访问A的4个8x8 matrix (16x16)
      // ldmatrix.{...}.x4.{...} 需要warp内32个线程都参与，每个线程提供一个有效的，不重叠的addr
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // t{0...15}=0~15, t{16...31}=0~15
      int lane_smem_a_k = (lane_id / 16) * 8; // 0, 8, t{0...15}=0, t{16...31}=8
      uint32_t lane_smem_a_ptr =
          (smem_a_base_ptr +
           (smem_sel * s_a_stage_offset + lane_smem_a_m * BK + lane_smem_a_k) *
               sizeof(half));
      LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
    }

    // ldmatrix.x2: 加载 B 的 k16n8 片段（非转置）
    // 为什么不用 .trans？因为 smem 中存的是 B^T row-major [N][K]，
    // ldmatrix 逐行加载 B^T 的行 = B 的列，天然给出 col-major B fragment → 直接匹配 MMA row.col
#pragma unroll
    for (int j = 0; j < kValTileN; ++j) {
      // {0,...,3} * (8 * 4) + j * 8 = {0,32,64,96} + {0,8,16,24} = {0,8,...,120}
      // warp_n_0 offsets {0,8,16,24}; warp_n_1 offsets {32,40,48,56}
      // warp_n_2 offsets {64,72,80,88}; warp_n_3 offsets {96,104,112,120}
      int warp_smem_b_n = warp_n * (kMmaN * kValTileN) + j * kMmaN;
      // {0,8,...,120} + {0~7} = {0~127}, 按照row-major的顺序访问B^T的2个8x8 matrix (8x16)
      // ldmatrix.{...}.x2.{...} 需要warp内前16个线程参与，后16个线程传的addr会被忽略
      int lane_smem_b_n = warp_smem_b_n + lane_id % 8; // t{0...7}=0~7, t{8...15}=0~7
      int lane_smem_b_k = ((lane_id / 8) % 2) * 8; // t{0...7}=0, t{8...15}=8
      uint32_t lane_smem_b_ptr =
          (smem_b_base_ptr +
           (smem_sel * s_b_stage_offset + lane_smem_b_n * BK + lane_smem_b_k) *
               sizeof(half));
      LDMATRIX_X2(RB[j][0], RB[j][1], lane_smem_b_ptr);
    }

    // MMA compute: 每个Warp(MMA) 在M方向重复kValTileM(4)次，在N方向重复kValTileN(4)次
#pragma unroll
    for (int i = 0; i < kValTileM; ++i) {
#pragma unroll
      for (int j = 0; j < kValTileN; ++j) {
        HMMA16816(RC[i][j][0], RC[i][j][1], // C fragment
                  RA[i][0], RA[i][1], RA[i][2], RA[i][3], // A fragment
                  RB[j][0], RB[j][1], // B fragment
                  RC[i][j][0], RC[i][j][1]);
      }
    }

    // 自适应等待：流水线满载期用 kStages-2，尾部排空用 0
    if (k + kStages - 1 < NUM_K_TILES) {
      CP_ASYNC_WAIT_GROUP(kStages - 2);
    } else { // 对于尾部的 k，等待所有剩余的 cp.async 完成
      CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();
  }

  // Epilogue: 寄存器 → global memory（通过 warp shuffle + 128-bit store）
  {
    for (int i = 0; i < kValTileM; ++i) {
      // RC[kValTileM][kValTileN][2] 中  RC[...][...][2] 中保存了2个 uint32
      // 寄存器，每个uint32 寄存器表示两个临近的fp16值：{c0,c1}，然后RC[...][...][0]
      // 和RC[...][...][1]代表的是按照col-major排布的2个8x8子矩阵上（不同物理行，跨8x8）
      // 同一个位置上的元素，也就是，实际代表了2个不同的行的元素，因此要分开RC0, RC1；
      // RC0表示第一行，8个half，可以用4个uint32寄存器来装；同理，RC1表示第二行。
      uint32_t RC0[kValTileN][4]; // 32 bits x 4 = 128 bits = 8 half
      uint32_t RC1[kValTileN][4]; // 32 bits x 4 = 128 bits = 8 half
      // ==================================================================
      // MMA m16n8k16 C fragment — a single 16×8 matrix (registers per warp):
      // Thread t holds 4 half values → RC[0] for rows 0-7, RC[1] for rows 8-15.
      // Mapping: row = t/4, col-pair = t%4 (each uint32 = 2 half values).
      //
      //       cols: c0-1    c2-3    c4-5    c6-7
      //   r0:  T0{c0,c1}   T1{c2,c3}  T2{c4,c5}  T3{c6,c7}  ┐
      //   r1:  T4{c0,c1}   T5{c2,c3}  T6{c4,c5}  T7{c6,c7}  │
      //   r2:  T8{c0,c1}   T9{c2,c3}  T10{c4,c5} T11{c6,c7} │ RC[0]
      //   ...                                               │
      //   r7:  T28{c0,c1}  T29{c2,c3} T30{c4,c5} T31{c6,c7} ┘
      //   r8:  T0{c0,c1}   T1{c2,c3}  T2{c4,c5}  T3{c6,c7}  ┐
      //   r9:  T4{c0,c1}   T5{c2,c3}  T6{c4,c5}  T7{c6,c7}  │
      //  r10:  T8{c0,c1}   T9{c2,c3}  T10{c4,c5} T11{c6,c7} │ RC[1]
      //   ...                                               │
      //  r15:  T28{c0,c1}  T29{c2,c3} T30{c4,c5} T31{c6,c7} ┘
      //
      // Within a 4-lane group (e.g. T0-T3 for row 0):
      //   lane+0 holds {c0,c1}, lane+1 holds {c2,c3},
      //   lane+2 holds {c4,c5}, lane+3 holds {c6,c7}.
      // shfl 从 lane+1/+2/+3 各收 2 个 half 到 lane+0，4 次 shuffle 凑齐
      // 一行 8 个 half = {c0,c1,c2,c3,c4,c5,c6,c7}，再由 lane%4==0 128-bit store。
      // ==================================================================
#pragma unroll
      for (int j = 0; j < kValTileN; ++j) {
        RC0[j][0] = RC[i][j][0];
        RC1[j][0] = RC[i][j][1];
        RC0[j][1] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 1);
        RC0[j][2] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 2);
        RC0[j][3] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 3);
        RC1[j][1] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 1);
        RC1[j][2] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 2);
        RC1[j][3] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 3);
      }
      // 每 4 个 lane 中只有 lane 0 做 128-bit store
      // lane_id / 4 → 行号映射（每 4 个连续 lane 负责上8x8矩阵和下8x8矩阵各一行）：
      //   ┌─────────┬───────────┬──────────────┬──────────────┐
      //   │ lane_id │ lane_id/4 │ RC[0] 的行   │ RC[1] 的行   │
      //   ├─────────┼───────────┼──────────────┼──────────────┤
      //   │ 0~3     │ 0         │ row 0        │ row 8        │
      //   │ 4~7     │ 1         │ row 1        │ row 9        │
      //   │ 8~11    │ 2         │ row 2        │ row 10       │
      //   │ 12~15   │ 3         │ row 3        │ row 11       │
      //   │ 16~19   │ 4         │ row 4        │ row 12       │
      //   │ 20~23   │ 5         │ row 5        │ row 13       │
      //   │ 24~27   │ 6         │ row 6        │ row 14       │
      //   │ 28~31   │ 7         │ row 7        │ row 15       │
      //   └─────────┴───────────┴──────────────┴──────────────┘
      if (lane_id % 4 == 0) {
        // {0,1} * (16 * 4) + i * 16 = {0,64} + {0,16,32,48} = {0,16,32,48,64,80,96,112}
        int store_warp_smem_c_m = warp_m * (kMmaM * kValTileM) + i * kMmaM; // smem row
        // 这里用 lane_id / 4 → {0,1,2,3,4,5,6,7}，对应 RC0/RC1 (+8) 的行号，表示2个8x8矩阵的行号
        int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4; // gmem row
#pragma unroll
        for (int j = 0; j < kValTileN; ++j) {
          // {0,...,3} * (8 * 4) + j * 8 = {0,32,64,96} + {0,8,16,24} = {0,8,...,120}
          int store_warp_smem_c_n = warp_n * (kMmaN * kValTileN) + j * kMmaN; // smem col
          int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n; // gmem col
          int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n; // 1-th 8x8 matrix
          int store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n; // 2-th 8x8 matrix
          // 128-bit store: 一次写入 8 个 half
          *reinterpret_cast<float4 *>(&C[store_gmem_c_addr_0]) =
              *reinterpret_cast<float4 *>(&RC0[j][0]);
          *reinterpret_cast<float4 *>(&C[store_gmem_c_addr_1]) =
              *reinterpret_cast<float4 *>(&RC1[j][0]);
        }
      }
    }
  }
}

// =============================================================================
template <const int kColStride = 16>
static __device__ __forceinline__ int swizzle(int i, int j) {
#if defined(NOTES_V2_ENABLE_SWIZZLE_V2)
  return swizzle_v2_impl<kColStride>(i, j);
#else
  return swizzle_v1_impl<kColStride>(i, j);
#endif
}

// =============================================================================
// Phase 7b-3: HGEMM MMA — m16n8k16 + multistage pipeline + TN 布局 + XOR swizzle
//              + Register Double Buffering (generic kValTileK, default BK=64)
// =============================================================================
// 在 Phase 7b-2 的 smem XOR swizzle 基础上增加寄存器双缓冲：
//   - kValTileK: 每个 BK tile 包含 kValTileK 个 kMmaK slice（BK = kMmaK * kValTileK）
//   - 统一 BK tile：所有 k_step slice 连续存放在同一个 BK 宽的 smem tile 中
//   - RA[2][kValTileM][4] / RB[2][kValTileN][2]：双份寄存器乒乓切换（与 kValTileK 无关）
//   - ldmatrix 与 MMA 计算重叠：加载 k_step+1 的同时用另一组寄存器做 k_step 计算
//   - 支持任意 kValTileK >= 2；默认 kValTileK=4 (BK=64)，kStages=2 推荐 64KB smem
//
// ★ smem 布局：
//   s_a: [stage0][stage1]...[stage(kStages-1)]，每个 stage = BM × BK = 128 × BK
//   s_b: 紧接 s_a 之后
//   每个 stage 内 k_step slice 列偏移 = k_step * kMmaK
//
// ★ 寄存器双缓冲时间线（每 k 迭代内）：
//   Step 1: G→S cp.async 预取 stage(k+kStages-1) 全部 k_step（条件）
//   Step 2: MMA k_step=0（用 reg[load]，已在上轮 post-wait 中 ldmatrix）
//   Step 3: for k_step=1..kValTileK-1: ldmatrix(列偏移 k_step*kMmaK)→reg[store], flip, MMA→reg[load]
//   Step 4: adaptive wait + __syncthreads
//   Step 5: S→R ldmatrix 预加载 stage(k+1) k_step=0 → reg[store]（条件）
//
// 参考：LeetCUDA/kernels/hgemm/mma/swizzle/hgemm_mma_stage_tn_swizzle_x2.cu
// Grid:  ((N+127)/128/S, (M+127)/128, S)，S=(N+2047)/2048，3D block swizzle
// Block: (256, 1, 1)，8 warps
template <const int kMmaM = 16,             // MMA atom M dim (m16n8k16)
          const int kMmaN = 8,              // MMA atom N dim
          const int kMmaK = 16,             // MMA atom K dim
          const int kMmaTileM = 2,          // warps along M, 2 → warp tile M = 32
          const int kMmaTileN = 4,          // warps along N, 4 → warp tile N = 32
          const int kValTileM = 4,          // value-repeat along M, BM = 16*2*4 = 128
          const int kValTileN = 4,          // value-repeat along N, BN = 8*4*4 = 128
          const int kValTileK = 4,          // MMA_K slices per BK tile, BK = kMmaK * kValTileK
          const int kStages = 2,            // cp.async pipeline depth (2 for kValTileK>=4 to fit smem)
          const int kBlockSwizzle = 0>      // 1 enables 3D grid swizzle for L2 locality
__global__ void __launch_bounds__(256)
    hgemm_mma_stages_tn_swizzle(half *A, half *B, half *C, int M, int N, int K) {
  static_assert(kValTileK >= 2, "Register double buffering requires kValTileK >= 2");
  static_assert(kBlockSwizzle == 0 || kBlockSwizzle == 1, "kBlockSwizzle must be 0 or 1");
  static_assert(kStages >= 2, "kStages must be >= 2 for cp.async pipeline");
  // Block Swizzle: 在 grid x 维度做 swizzle，改善 L2 cache 局部性
  const int bx = ((int)kBlockSwizzle) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  constexpr int BM = kMmaM * kMmaTileM * kValTileM; // 16*2*4=128
  constexpr int BN = kMmaN * kMmaTileN * kValTileN; // 8*4*4=128
  constexpr int BK = kMmaK * kValTileK;             // 16 * kValTileK

  // kStages stages, each with BM×BK for A, BN×BK for B
  // smem: kValTileK 个 kMmaK slice 连续存放在同一个 BK 宽 tile 中
  extern __shared__ half smem[];
  half *s_a = smem;
  half *s_b = smem + kStages * BM * BK;
  constexpr int s_a_stage_offset = BM * BK;
  constexpr int s_b_stage_offset = BN * BK;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
  const int warp_m = warp_id % kMmaTileM; // 0,1（M 方向 2 个 warp）kMmaTileM = 2
  const int warp_n = warp_id / kMmaTileM; // 0,1,2,3（N 方向 4 个 warp）

  // 线程到 global memory 的映射（用于加载 A 和 B）共 256 个线程
  // TN 布局关键: A[m*K+k] 是 row-major, B^T[n*K+k] 是 row-major（内维连续的是 K）
  // 注意：smem_a_k 和 smem_b_k 依然使用 0/8，在后续的 kValTileK 循环中
  // 会加上 k_step*kMmaK，最终覆盖全部 BK 列
  int load_smem_a_m = tid / 2;                 // 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;  // 0, 8
  int load_smem_b_n = tid / 2;                 // 0~127 → B^T 的 N 方向（row-major 的行）
  int load_smem_b_k = (tid % 2 == 0) ? 0 : 8;  // 0, 8  → B^T 的 K 方向（row-major 的列）
  int load_gmem_a_m = by * BM + load_smem_a_m; // C/A 全局行号 = M 方向的 tile 起始 + 线程偏移
  int load_gmem_b_n = bx * BN + load_smem_b_n; // C/B 全局列号 = N 方向的 tile 起始 + 线程偏移
  if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    return;

  // 8个Warps(MMAs)的排布是M方向2个，N方向4个，则MMA Atom一次性能处理的tile大小是[2*16,4*8]=[32,32].
  // CUDA中每个block能放的线程数是有上限的（warp数量有限），一般为4或者8个warps。为了能处理更大的C tile，
  // 需要把MMA Atom的tile再M方向和N方向各自重复4次，得到[128,128]的C block tile，这就是Value Tile的作用。
  // MMA Tile对应到cutlass cute中的TiledMMA的概念，Value Tile对应到cutlass cute中的PermuteMNK的概念。
  uint32_t RC[kValTileM][kValTileN][2] = {0}; // 初始化为 0

  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

  // Prefetch (kStages-1) stages: load all k_step slices for each early stage
#pragma unroll
  for (int k = 0; k < (kStages - 1); ++k) {
#pragma unroll
    for (int k_step = 0; k_step < kValTileK; ++k_step) {
      int load_gmem_a_k = k * BK + (k_step * kMmaK) + load_smem_a_k;
      int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
      int load_gmem_b_k = k * BK + (k_step * kMmaK) + load_smem_b_k;
      int load_gmem_b_addr = load_gmem_b_n * K + load_gmem_b_k;

      uint32_t load_smem_a_ptr =
          (smem_a_base_ptr +
           (k * s_a_stage_offset + load_smem_a_m * BK +
            k_step * kMmaK +
            swizzle<kMmaK>(load_smem_a_m, load_smem_a_k)) *
               sizeof(half));
      CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

      uint32_t load_smem_b_ptr =
          (smem_b_base_ptr +
           (k * s_b_stage_offset + load_smem_b_n * BK +
            k_step * kMmaK +
            swizzle<kMmaK>(load_smem_b_n, load_smem_b_k)) *
               sizeof(half));
      CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    }
    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(kStages - 2);
  __syncthreads();

  // Register double buffers: RA[0/1] for k_step=0/1 A data, RB[0/1] for B data
  uint32_t RA[2][kValTileM][4];
  uint32_t RB[2][kValTileN][2];
  int reg_st_idx = 0; // write target for ldmatrix
  int reg_ld_idx = 1; // read source for MMA

// Initial ldmatrix: load stage 0, S -> R, k_step=0 (0~15列) → reg[0]
// 此时，reg_st_idx = 0，对0位置的寄存器buffer做初始化。
#pragma unroll
  for (int i = 0; i < kValTileM; ++i) {
    int warp_smem_a_m = warp_m * (kMmaM * kValTileM) + i * kMmaM;
    int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
    int lane_smem_a_k = (lane_id / 16) * 8;
    uint32_t lane_smem_a_ptr =
        (smem_a_base_ptr +
         (0 * s_a_stage_offset + lane_smem_a_m * BK +
          swizzle<kMmaK>(lane_smem_a_m, lane_smem_a_k)) *
             sizeof(half));
    LDMATRIX_X4(RA[reg_st_idx][i][0], RA[reg_st_idx][i][1],
                RA[reg_st_idx][i][2], RA[reg_st_idx][i][3],
                lane_smem_a_ptr);
  }
#pragma unroll
  for (int j = 0; j < kValTileN; ++j) {
    int warp_smem_b_n = warp_n * (kMmaN * kValTileN) + j * kMmaN;
    int lane_smem_b_n = warp_smem_b_n + lane_id % 8;
    int lane_smem_b_k = ((lane_id / 8) % 2) * 8;
    uint32_t lane_smem_b_ptr =
        (smem_b_base_ptr +
         (0 * s_b_stage_offset + lane_smem_b_n * BK +
          swizzle<kMmaK>(lane_smem_b_n, lane_smem_b_k)) *
             sizeof(half));
    LDMATRIX_X2(RB[reg_st_idx][j][0], RB[reg_st_idx][j][1],
                lane_smem_b_ptr);
  }

  // 统一循环：k 从 0 开始，条件 G→S / S→R / wait 处理所有边界情况
  // BK = kMmaK * kValTileK，每个 k 迭代覆盖 BK 个 K 元素
  const int NUM_K_TILES = div_ceil(K, BK);
  for (int k = 0; k < NUM_K_TILES; ++k) {
    int smem_sel = k % kStages;
    int smem_sel_next = (k + kStages - 1) % kStages;

    // G→S: 条件预取 stage(k+kStages-1) 全部 k_step slice
    if (k + kStages - 1 < NUM_K_TILES) {
#pragma unroll
      for (int k_step = 0; k_step < kValTileK; ++k_step) {
        int load_gmem_a_k =
            (k + kStages - 1) * BK + (k_step * kMmaK) + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k =
            (k + kStages - 1) * BK + (k_step * kMmaK) + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_n * K + load_gmem_b_k;

        uint32_t load_smem_a_ptr =
            (smem_a_base_ptr +
             (smem_sel_next * s_a_stage_offset + load_smem_a_m * BK +
              k_step * kMmaK +
              swizzle<kMmaK>(load_smem_a_m, load_smem_a_k)) *
                 sizeof(half));
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

        uint32_t load_smem_b_ptr =
            (smem_b_base_ptr +
             (smem_sel_next * s_b_stage_offset + load_smem_b_n * BK +
              k_step * kMmaK +
              swizzle<kMmaK>(load_smem_b_n, load_smem_b_k)) *
                 sizeof(half));
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
      }
      CP_ASYNC_COMMIT_GROUP();
    }

    // 内层 k_step 循环: flip → ldmatrix(k_step+1) → MMA, 乒乓交替
    // 初始: st=0, ld=1, reg[0]=preload k_step=0; reg[1]=未初始化
    // 每轮: flip→ldmatrix next k_step→reg[st], MMA reg[ld]（k_step+1 的 ldmatrix 条件化）

#pragma unroll
    for (int k_step = 0; k_step < kValTileK; ++k_step) {
      reg_st_idx ^= 1; // 0 -> 1; 1 -> 0
      reg_ld_idx ^= 1; // 1 -> 0; 0 -> 1

      if (k_step + 1 < kValTileK) {
        int smem_k_offset = (k_step + 1) * kMmaK;
#pragma unroll
        for (int i = 0; i < kValTileM; ++i) {
          int warp_smem_a_m = warp_m * (kMmaM * kValTileM) + i * kMmaM;
          int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
          int lane_smem_a_k = (lane_id / 16) * 8;
          uint32_t lane_smem_a_ptr =
              (smem_a_base_ptr +
              (smem_sel * s_a_stage_offset + lane_smem_a_m * BK +
                smem_k_offset +
                swizzle<kMmaK>(lane_smem_a_m, lane_smem_a_k)) *
                  sizeof(half));
          LDMATRIX_X4(RA[reg_st_idx][i][0], RA[reg_st_idx][i][1],
                      RA[reg_st_idx][i][2], RA[reg_st_idx][i][3],
                      lane_smem_a_ptr);
        }
  #pragma unroll
        for (int j = 0; j < kValTileN; ++j) {
          int warp_smem_b_n = warp_n * (kMmaN * kValTileN) + j * kMmaN;
          int lane_smem_b_n = warp_smem_b_n + lane_id % 8;
          int lane_smem_b_k = ((lane_id / 8) % 2) * 8;
          uint32_t lane_smem_b_ptr =
              (smem_b_base_ptr +
              (smem_sel * s_b_stage_offset + lane_smem_b_n * BK +
                smem_k_offset +
                swizzle<kMmaK>(lane_smem_b_n, lane_smem_b_k)) *
                  sizeof(half));
          LDMATRIX_X2(RB[reg_st_idx][j][0], RB[reg_st_idx][j][1],
                      lane_smem_b_ptr);
        }
      }

#pragma unroll
      for (int i = 0; i < kValTileM; ++i) {
#pragma unroll
        for (int j = 0; j < kValTileN; ++j) {
          HMMA16816(RC[i][j][0], RC[i][j][1],
                    RA[reg_ld_idx][i][0], RA[reg_ld_idx][i][1],
                    RA[reg_ld_idx][i][2], RA[reg_ld_idx][i][3],
                    RB[reg_ld_idx][j][0], RB[reg_ld_idx][j][1],
                    RC[i][j][0], RC[i][j][1]);
        }
      }
    }

    // 自适应等待：满载期用 kStages-2，尾部排空用 0
    if (k + kStages - 1 < NUM_K_TILES) {
      CP_ASYNC_WAIT_GROUP(kStages - 2);
    } else {
      CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();

    // 从下一阶段加载 k_step=0 数据，供下一轮 k 迭代使用，此时 reg_st_idx=0
    if (k + 1 < NUM_K_TILES) {
      smem_sel = (k + 1) % kStages; // old smem_sel + 1
#pragma unroll
      for (int i = 0; i < kValTileM; ++i) {
        int warp_smem_a_m = warp_m * (kMmaM * kValTileM) + i * kMmaM;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
        int lane_smem_a_k = (lane_id / 16) * 8;
        uint32_t lane_smem_a_ptr =
            (smem_a_base_ptr +
             (smem_sel * s_a_stage_offset + lane_smem_a_m * BK +
              swizzle<kMmaK>(lane_smem_a_m, lane_smem_a_k)) *
                 sizeof(half));
        LDMATRIX_X4(RA[reg_st_idx][i][0], RA[reg_st_idx][i][1],
                    RA[reg_st_idx][i][2], RA[reg_st_idx][i][3],
                    lane_smem_a_ptr);
      }
#pragma unroll
      for (int j = 0; j < kValTileN; ++j) {
        int warp_smem_b_n = warp_n * (kMmaN * kValTileN) + j * kMmaN;
        int lane_smem_b_n = warp_smem_b_n + lane_id % 8;
        int lane_smem_b_k = ((lane_id / 8) % 2) * 8;
        uint32_t lane_smem_b_ptr =
            (smem_b_base_ptr +
             (smem_sel * s_b_stage_offset + lane_smem_b_n * BK +
              swizzle<kMmaK>(lane_smem_b_n, lane_smem_b_k)) *
                 sizeof(half));
        LDMATRIX_X2(RB[reg_st_idx][j][0], RB[reg_st_idx][j][1],
                    lane_smem_b_ptr);
      }
    }
  }

  // Epilogue: 复用 RA[2][4][4] 寄存器做 warp shuffle → 128-bit collective store
  // 做RC的warp shuffle正好需要[2][4][4]的寄存器空间，RA[2][4][4]正好可以复用
  for (int i = 0; i < kValTileM; ++i) {
#pragma unroll
    for (int j = 0; j < kValTileN; ++j) {
      RA[0][j][0] = RC[i][j][0];
      RA[1][j][0] = RC[i][j][1];
      RA[0][j][1] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 1);
      RA[0][j][2] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 2);
      RA[0][j][3] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 3);
      RA[1][j][1] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 1);
      RA[1][j][2] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 2);
      RA[1][j][3] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 3);
    }
    if (lane_id % 4 == 0) {
      int store_warp_smem_c_m = warp_m * (kMmaM * kValTileM) + i * kMmaM;
      int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
#pragma unroll
      for (int j = 0; j < kValTileN; ++j) {
        int store_warp_smem_c_n = warp_n * (kMmaN * kValTileN) + j * kMmaN;
        int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n;
        int store_gmem_c_addr_0 =
            store_lane_gmem_c_m * N + store_lane_gmem_c_n;
        int store_gmem_c_addr_1 =
            (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
        *reinterpret_cast<float4 *>(&C[store_gmem_c_addr_0]) =
            *reinterpret_cast<float4 *>(&RA[0][j][0]);
        *reinterpret_cast<float4 *>(&C[store_gmem_c_addr_1]) =
            *reinterpret_cast<float4 *>(&RA[1][j][0]);
      }
    }
  }
}

// =============================================================================
// Phase 7c: HGEMM CuTe — CUTLASS CuTe DSL 实现（SM80+, Tensor Core 全自动调度）
// =============================================================================
// 面试要点（CuTe vs 手写 MMA PTX）：
//   - CuTe (CUTLASS Templates) 是 NVIDIA CUTLASS 3.x 的核心 DSL，提供编译期
//     Tensor 抽象，自动推导 MMA、Copy、Swizzle 的线程-数据映射
//   - 与手写 MMA PTX (Phase 7b) 的对比：
//     - 手写 MMA: 手动计算 smem 地址、手动 ldmatrix/mma PTX、手动 swizzle、手动 epilogue shuffle
//     - CuTe: 声明 TiledMMA / TiledCopy / SmemLayout，编译器自动推导线程-数据映射,
//       cute::copy / cute::gemm 自动展开为高效指令序列
//   - 核心抽象（"CuTe 五要素"）：
//     1. Tensor: 全局/共享/寄存器数据的逻辑视图 = (ptr, shape, stride)
//     2. TiledMMA: 描述 MMA 指令 + warp/warpgroup 排布 + value tile 的 compile-time type
//     3. TiledCopy: 描述数据搬运（G→S, S→R, R→S, S→G）的线程-元素映射
//     4. Layout + Swizzle: 描述 smem 的数据排布和 bank conflict 消除
//     5. Pipeline: cp.async + multistage, 由 cute::copy 自动管理
//
//   CuTe kernel 的参数推导链（launch wrapper 负责实例化，kernel 只接收实例化后的类型）：
//     MMA Atom (SM80_16x8x16_F16F16F16F16_TN)
//       → TiledMMA (EURepeat=MxNxK, ValTile=MxNxK)
//       → TiledCopy (G2S/S2R/R2S/S2G, 每种有 ThrLayout + ValLayout)
//       → SmemLayout (Atom + Swizzle + tile_to_shape + Stage)
//
//   调参口诀（面试速记）：
//     BM/BN/BK 越大 → smem 越大 → occupancy 越低, 但 K 循环更少 → 指令开销更低
//     Swizzle<B,M,S> 的核心不是改变逻辑 Tensor 的 shape，而是把一维 offset
//     重新解释为一个二维逻辑空间，再把二维坐标映射回 bank-conflict-free 的物理 offset：
//       1. 连续的 2^M 个元素组成二维空间中的一个基本元素；（元素宽度）
//       2. 连续的 2^S 个基本元素组成一行；（列数）
//       3. 二维空间包含 2^B 行；（行数）
//       4. 对二维坐标做列置换：icol' = irow ^ icol；
//       5. 保留基本元素内部的低 M 位，再将置换后的二维坐标编码回 offset。
//     因此 M 描述基本元素宽度，S 描述二维空间的列数，B 描述二维空间的行数。
//     CuTe 的位级实现等价于：
//       offset' = offset ^ ((offset & YYY) >> S),
//       YYY = ((1 << B) - 1) << (M + S)。
//     对 Swizzle<3,3,3>：每个基本元素有 2^3=8 个值，每行有 2^3=8 个基本元素，
//     二维空间有 2^3=8 行；元素地址位 [6:8] XOR 到 [3:5]，低 3 位保持不变。
//     一个完整 swizzle 周期覆盖 2^(M+S+B)=2^9=512 个元素（FP16 下为 1024B）。
//     kStage: 2=最低延迟, 3/4=更好隐藏延迟 → 权衡 smem 占用
//     EURepeat: 在 M/N/K 方向重复 MMA atom 的执行单元布局，决定 TiledMMA 的线程排布
//       与逻辑 tile 组织；MMA atom越多，需要的线程数就越多
//
// ★ 与手写 MMA Swizzle (Phase 7b-3) 的本质区别：
//   - 手写 swizzle: 在 smem 地址计算时手动 XOR 列索引
//   - CuTe Swizzle: SmemLayout 声明式指定地址位的 XOR pattern，cute::copy 自动应用
//   - CuTe 优势: 类型安全，编译器可在编译期推导映射；布局变更通常只需修改类型定义
//
// ★ 本实现的 Tile 配置：
//   - gmem/smem tile: BM=128, BN=256, BK=32, kStage=2；这是 A/B tile 的目标 shape。
//   - MMA atom: SM80_16x8x16_F16F16F16F16_TN；EURepeat=2x2x1，MMA_P_T=32x32x16。
//   - TiledMMA 的逻辑 MMA tile 是 32×32×16，G2S/S2R copy 再把它映射到更大的 BM×BN tile；
//     BN=256 不是由“8×2×16”直接推导出的 MMA tile，而是本 wrapper 选择的 B tile 尺寸。
//   - Smem Swizzle<3,3,3>: 512-element address period 的 XOR 重排，针对 ldmatrix
//     的访问模式降低 bank conflict；512 是 swizzle 周期，不等于 A atom 的逻辑元素数。
//   - Threads: size(MMA{}) = 128（4 warps × 2×2 EU repeat）。
//   - Dynamic smem: Stage=2 时 A[128×32] + B[256×32] 共 49,152 bytes（约 48 KiB）。
//
// 参考：LeetCUDA/kernels/hgemm/cutlass/hgemm_mma_stage_tn_cute.cu
// 参考：https://zhuanlan.zhihu.com/p/671419093 (CuTe Swizzle 详解)
// =============================================================================

#if defined(NOTES_V2_ENABLE_CUTE)

#include <type_traits>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/device_kernel.h>
#include <cutlass/numeric_conversion.h>

// =============================================================================
// Phase 7c-1: CuTe HGEMM Kernel — TN 布局 + Smem Swizzle + Multistage Pipeline
// =============================================================================
// TN 布局: C[M×N] = A[M×K] × B^T[N×K]
//   - A[M][K] row-major, B^T[N][K] row-major（等价 B col-major [K×N]）
//   - CuTe 中通过 make_tensor(make_gmem_ptr(ptr), shape, stride) 描述
//
// Pipeline 流程（stage 是 K tile 的循环缓冲槽，不是一条完整计算链）：
//   1. PREFETCH: 预加载 kStage-1 个 K tile (G→S via cp.async)，填满 stage。
//   2. 主循环 over K tiles:
//      a. 从当前 stage 做 S→R：cute::copy(...) → ldmatrix；
//      b. 对当前 K slice 做 MMA：cute::gemm(...) → mma.sync；
//      c. 在处理当前 tile 的首个 MMA slice 时，异步提交下一个 tile 的 G→S；
//      d. 当前 tile 的最后一个 K slice 完成后等待并切换 smem stage。
//   3. Epilogue: R→S（UniversalCopy 写入 C scratchpad）→ S→G（128-bit store）。
//
// 面试对比 — 手写 MMA vs CuTe 代码量：
//   手写: ldmatrix PTX 地址计算 + swizzle XOR + mma PTX + epilogue shuffle (~200行)
//   CuTe: partition + copy + gemm (~80行), 其余由 launch wrapper 的类型推导完成
template <typename T,                      // element type (half)
          int BM, int BN, int BK,          // block tile: BM=128, BN=256, BK=32
          int kStage,                      // cp.async pipeline depth (2)
          typename TiledMMA,               // MMA: SM80_16x8x16_F16F16F16F16_TN, EURepeat=2x2x1
          typename G2SCopyA,               // G→S A: SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>
          typename G2SCopyB,               // G→S B: same as G2SCopyA
          typename SmemLayoutA,            // smem A: Swizzle<3,3,3> + 8×BK atom → (BM,BK,kStage)
          typename SmemLayoutB,            // smem B: same atom → (BN,BK,kStage)
          typename SmemLayoutC,            // C scratchpad: Swizzle<3,3,3> + 32×32 atom, pipe=4
          typename S2RCopyAtomA,           // S→R A: SM75_U32x4_LDSM_N (ldmatrix.x4)
          typename S2RCopyAtomB,           // S→R B: same as S2RCopyAtomA
          typename R2SCopyAtomC,           // R→S C: UniversalCopy<int> (32-bit store)
          typename S2GCopyAtomC,           // S→G C: UniversalCopy<uint128_t> (128-bit wide store)
          typename S2GCopyC,               // S→G TiledCopy: ThrLayout{32,4} × ValLayout{1,8}
          const int BlockSwizzle = 0>      // 1 enables 3D grid swizzle for L2 locality
__global__ void hgemm_mma_stages_tn_cute(T *Aptr, T *Bptr, T *Dptr, int m,
                                         int n, int k) {
  using namespace cute;
  static_assert(BlockSwizzle == 0 || BlockSwizzle == 1, "BlockSwizzle must be 0 or 1");
  static_assert(kStage >= 2, "kStage must be >= 2 for cp.async pipeline");

  // 动态 shared memory: KStage 个 A/B tile；C epilogue 复用当前 A stage 的空间。
  extern __shared__ T shm_data[];
  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx = threadIdx.x;
  // BlockSwizzle: 0/1 控制是否启用 thread block swizzle，改善 L2 cache 局部性
  int ix = ((int)BlockSwizzle) * blockIdx.z * gridDim.x + blockIdx.x;
  int iy = blockIdx.y; // M 方向 block 索引
  // 当前 kernel 只有 CTA 级边界判断，没有 tile 内的逐元素 predication；调用方需保证
  // M % BM == 0、N % BN == 0、K % BK == 0，才能避免边界 tile 的越界访问或未写回。
  if (iy * BM >= m || ix * BN >= n) return;

  // CuTe Tensor 抽象: (ptr, shape, stride) 三元组描述数据布局
  // make_stride(leading_dim, Int<1>{})  → row-major: 同行相邻元素步长=1, 跨行步长=leading_dim
  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{}));
  Tensor D = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{}));

  // local_tile: 按 tiler 切出当前 CTA 的逻辑 tile；返回 Tensor 仍保留未切出的 remainder mode。
  // make_coord(iy, _): 固定 M/N 方向的 CTA 坐标，_ 保留 K 方向的 tile remainder：
  //   gA: (BM, BK, num_tile_k)，gB: (BN, BK, num_tile_k)，gD: (BM, BN)。
  Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _));
  Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));
  Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix));

  // Shared memory Tensor: 按 SmemLayout 解读 smem 区域
  auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{}); // (BM, BK, kStage)
  auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{}); // (BN, BK, kStage)

  // TiledMMA partition: 将 MMA 的 A/B/C tile 映射到各线程的寄存器 fragment
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
  auto tCrD = thr_mma.partition_fragment_C(gD);          // (MMA, MMA_M, MMA_N)
  clear(tCrD); // 累加器清零

  // G2S TiledCopy: 描述 global → shared memory 的数据搬运（128-bit cp.async）。
  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, num_tile_k)
  auto tAsA_copy = g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K, kStage)

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, num_tile_k)
  auto tBsB_copy = g2s_thr_copy_b.partition_D(sB);

  // S2R TiledCopy: 描述 shared → register 的数据搬运（使用 ldmatrix）
  // make_tiled_copy_A/B: 根据 TiledMMA 自动推导 S2R copy 的线程-数据映射
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);     // (CPY, CPY_M, CPY_K, kStage)
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K) — 与 rA 寄存器布局对齐

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

  // PREFETCH: 预加载前 (kStage - 1) 个 K tile，填满循环缓冲；每次 copy 提交一个异步 G→S 操作。
  int itile_to_read = 0, ismem_read = 0, ismem_write = 0;
#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage) {
    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
               tAsA_copy(_, _, _, istage));
    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
               tBsB_copy(_, _, _, istage));
    cp_async_fence();
    ++itile_to_read;
    ++ismem_write;
  }
  cp_async_wait<kStage - 2>();
  __syncthreads();

  // K 维第一层循环 — 按 BK 大小将 K 切分为 num_k_tiles 个 tile。
  // 外层 k_tile 遍历这些 BK-tile，内层 k_step 遍历 tile 内的 MMA_K slice。
  // 当前实现使用整除，要求 K % BK == 0；没有为尾部 K tile 做 predication/padding。
  int num_k_tiles = k / BK;
  // 循环前预加载首轮数据：将第一个 K tile 的第 0 个 K slice 从 smem 加载到寄存器。
  cute::copy(s2r_tiled_copy_a, tAsA(_, _, 0, ismem_read), tCrA_view(_, _, 0));
  cute::copy(s2r_tiled_copy_b, tBsB(_, _, 0, ismem_read), tCrB_view(_, _, 0));

#pragma unroll 1
  for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
    // num_k_steps: BK tile 内 K 方向的 MMA 迭代次数，取自 fragment 的第三 mode（K-mode）。
    //
    // 为什么只显式展开 K？MMA_M 和 MMA_N 去哪了？
    //   tCrA=(MMA, MMA_M, MMA_K), tCrB=(MMA, MMA_N, MMA_K), tCrD=(MMA, MMA_M, MMA_N)
    //   - K 方向：每个 k_step 需要从 smem 加载**不同的** A/B 数据（ldmatrix），
    //     所以 K 循环必须显式写，每次迭代做 S→R copy + cute::gemm。
    //   - M/N 方向：同一个 k_step 内，所有 M、N 位置的 MMA atom 共享
    //     同一批寄存器数据。cute::gemm 根据 tiled_mma 的类型信息（EURepeat=2×2）
    //     在编译期自动展开为覆盖全部 (MMA_M, MMA_N) 对的 mma.sync 指令序列，
    //     无需手动写 M/N 循环。——这等价于手写 MMA 中的：
    //       for (i=0; i<kValTileM; ++i)
    //         for (j=0; j<kValTileN; ++j)
    //           HMMA16816(RC[i][j][0], RC[i][j][1], ...);
    //
    // CUTLASS 推导链（mma_atom.hpp）：
    //   make_tiled_mma 将 MMA_P_T::<2>（即 kMmaPK）存入 PermutationMNK
    //   → TiledMMA::permutation_mnk<2>() 返回 kMmaPK
    //   → thrfrg_A 对 A(M,K) 按 (permutation_mnk<0>(), permutation_mnk<2>())
    //     做 logical_divide，即按 (kMmaPM,kMmaPK) tile 化 K 维：
    //        K 被分为 BK/kMmaPK 个 perm-K slice
    //     随后按 AtomShape_MNK<2>（MMA_K_atom）做 zipped_divide
    //     每个 perm-K 含 kMmaEURepeatK 个 atom-K slice
    //   → partition_A 选当前线程后，K-mode size = BK / kMmaPK = 32 / 16 = 2
    // 本配置：kMmaPK=16（MMA_K_atom=16 × kMmaEURepeatK=1），BK=32 → num_k_steps=2
    int num_k_steps = size<2>(tCrA);

#pragma unroll
    for (int k_step = 0; k_step < num_k_steps; ++k_step) {
      int k_step_next = (k_step + 1) % num_k_steps;

      if (k_step == num_k_steps - 1) {
          // 等待下一个 K tile 的 smem 数据就绪，然后切换到下一个 stage。
        cp_async_wait<kStage - 2>();
        __syncthreads();
        ismem_read = (ismem_read + 1) % kStage;
      }

      // S→R: 加载下一组 A/B fragment 到寄存器。
      //
      // cute::copy 原生支持多 mode——理论上可以一次加载全部 K slice：
      //   cute::copy(s2r_tiled_copy_a, tAsA(_, _, _, ismem_read), tCrA_view);
      // 但这里刻意逐 k_step_next 加载，目的是做**寄存器双缓冲**：
      //   当前 k_step 做 MMA 计算时，ldmatrix 预加载 k_step_next 的数据，
      //   让 S→R 延迟与 MMA 计算重叠。
      // 如果一次性加载全部 K slice，S→R 和 MMA 就会彻底串行。
      cute::copy(s2r_tiled_copy_a, tAsA(_, _, k_step_next, ismem_read),
                 tCrA_view(_, _, k_step_next));
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, k_step_next, ismem_read),
                 tCrB_view(_, _, k_step_next));

      if (k_step == 0) {
        // G→S: 提交下一个 K tile（整个 BK）的异步拷贝。
        if (itile_to_read < num_k_tiles) {
          cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                     tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                     tBsB_copy(_, _, _, ismem_write));
          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }
        cp_async_fence();
      }

      // MMA: cute::gemm 内部根据 tiled_mma 的类型信息，自动展开为覆盖
      // 全部 (MMA_M, MMA_N) 对的 mma.sync 指令序列，累加到 tCrD。
      cute::gemm(tiled_mma, tCrD, tCrA(_, _, k_step), tCrB(_, _, k_step), tCrD);
    }
  }

  // Epilogue: D 寄存器 → shared memory → global memory。
  // 使用当前 A stage 作为 C scratchpad，避免为 epilogue 单独分配完整的 C shared memory。
  // 这里是“复用 storage”，不是把 A Tensor 转换成 C Tensor：
  //   sA 的逻辑 shape 是 (BM, BK, kStage)=(128,32,2)，而 sA(_,_,ismem_read)
  //   只选出其中一个 A-stage 的 storage 起点；sC 随后以完全独立的 SmemLayoutC
  //   解释这同一段地址。当前 wrapper 的固定配置下：
  //     一个 A stage: 128 x 32 = 4096 个 half；
  //     C scratchpad: 32 x 32 x 4 = 4096 个 half。
  //   两者容量刚好相等，但 logical shape 和 layout 不是同一个：A 的 layout atom
  //   是 8x32，C 的 layout atom 是 32x32，二者都含 Swizzle<3,3,3>，却不能把
  //   C 数据再按 SmemLayoutA 读取。launch wrapper 的 static_assert 用当前配置
  //   检查 C scratchpad 能放进一个 A pipe；此处只别名该单个 stage，不是整块双缓冲 sA。
  //
  // mainloop 已结束，不会再通过 sA 的 A/B load view 消费这个 pipe 的旧 A 数据。
  // 从这一行开始，对这块地址的所有访问都通过 SmemLayoutC 派生的 sC 完成：R2S
  // 用它写 C，S2G 也用它读 C。因此 C 的 swizzle/address mapping 在写和读两侧一致。
  auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

  // R2S TiledCopy: 寄存器 → shared memory（使用 UniversalCopy 做类型转换）
  // R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>; 以 32-bit 粒度搬运 T 数据
  auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
  // tCrD 是 TiledMMA 产生的 accumulator fragment，逻辑 shape 为
  // (MMA, MMA_M, MMA_N)。它的寄存器元素已经就位，但该 layout 是为 MMA
  // accumulator 服务的，并不直接按 R2S copy atom 的 source-value 顺序表达。
  //
  // retile_S 中的 S 指“这个 TiledCopy 的 source side”，不是 shared memory。
  // 它基于 r2s_tiled_copy_c 的 reference layout 创建一个共享 tCrD.data() 的
  // zero-copy layout view，将逻辑坐标表示为 (CPY, CPY_M, CPY_N)。这一步不读取
  // 或写入寄存器/共享内存，不进行 dtype conversion，也不交换 lane 间数据；真正的
  // R2S 数据搬运发生在下面的 cute::copy(r2s_tiled_copy_c, ..., tCsC_r2s(...))。
  // 可将它与主循环的 s2r_thr_copy_a.retile_D(tCrA) 对照：二者都是为了让已有
  // fragment 的 per-thread value ordering 匹配某个 TiledCopy 的 source/destination
  // 角色，而不是另一次 memory transfer。
  // tCrD: (MMA, MMA_M, MMA_N) -> retile -> tCrC_r2s: (CPY, CPY_M, CPY_N)
  auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD); // (CPY, CPY_M, CPY_N)
  // get_slice(idx) 已固定 CTA 内当前 logical copy thread；partition_D 再按
  // R2S TiledCopy 的 destination mapping 切分 sC。推导链：
  //   MMA_P_T = Tile<kMmaPM, kMmaPN, kMmaPK> = Tile<32, 32, 16>
  //     → TiledMMA::tile_size<0>(mma) = 32, tile_size<1>(mma) = 32
  //     → make_tiled_copy_C 的 tiler = (32, 32)
  //     → SmemLayoutC 的逻辑 M×N = (kMmaPM, kMmaPN) = (32, 32) 恰好等于该 tiler
  //     → partition_D: zipped_divide(sC, tiler=(32,32)) 后 M/N 维无 remainder
  //     → 结果 shape 的 M/N remainder = _1, _1
  // 最后一维 pipe=4 来自 SmemLayoutC 的第三维 kSmemLayoutCBatch，而不是
  // “thread-level tensor 自动只剩 CPY”这一条规则。_1 表示该逻辑 mode 的 extent
  // 是 1，并非该 mode 被删除或 C tile 没有 M/N 覆盖。
  auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC); // (CPY, _1, _1, pipe)

  // S2G TiledCopy: shared memory → global memory（128-bit store）
  // S2GCopyAtomC(Copy_Atom<UniversalCopy<cute::uint128_t>, T>) -> S2GCopyC
  S2GCopyC s2g_tiled_copy_c;
  auto s2g_thr_copy_c = s2g_tiled_copy_c.get_slice(idx);
  // tCsC_s2g 与 tCsC_r2s 都从同一个 sC 产生，因而保留相同的 pipe=4；前者是
  // S2G source mapping，后者是 R2S destination mapping。tCgC_s2g 则面向完整
  // gD=(BM,BN)=(128,256)，相对于 S2G copy tiler 仍有非平凡的 M/N repetition，
  // 所以它保留 CPY_M、CPY_N。这里的 CPY/CPY_M/CPY_N 都是编译期 copy-partition
  // 的逻辑 mode，不应直接理解为固定的 lane 数、warp 数或物理连续维度。
  auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC); // (CPY, _1, _1, pipe)
  auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD); // (CPY, CPY_M, CPY_N)

  // group_modes<B,E> 将 layout 的半开区间 [B,E) 组合成一个嵌套 mode：
  //   group_modes<1,3>(Tensor(CPY, CPY_M, CPY_N))
  //       -> Tensor(CPY, (CPY_M, CPY_N))
  // 它只重组 Tensor 的 layout，不分配内存、不移动 data()，也不改变元素访问的物理地址。
  // 第二个 mode 仍然保存原来 CPY_M/CPY_N 的层次关系，只是现在可以用一个坐标
  // i+j 访问这个组合 mode；因此这里的 group_modes 更准确地说是“合并 mode”，
  // 而不是把底层数据拷贝或无条件 flatten 成普通一维数组。
  // 从 type/shape 结构看，结果确实是 Tensor(CPY, (CPY_M, CPY_N))；但 grouped
  // mode 的 cardinality 是两个子 mode 的乘积，因此 size<1>(...) 可作为一个标量
  // 范围遍历其全部 logical coordinate。于是 (_, i+j) 中的 i+j 是该 nested mode
  // 的线性 logical coordinate，不是在 C++ 中对二维 tuple 做加法，更不是 raw
  // pointer offset；最终物理地址仍由各自 Tensor 的 grouped layout 计算。
  //
  // 这里为什么要对 tCrC_r2s 和 tCgC_s2g 同时 group？
  //   - tCrC_r2s: 每个线程持有的寄存器 C fragment，原始形状为 (CPY, CPY_M, CPY_N)；
  //   - tCgC_s2g: 该线程对应的 global-memory C 输出位置，形状与源 Tensor 对齐；
  //   - 两者使用相同的 [1,3) 合并后，源/目的 Tensor 仍保持相同的坐标空间，
  //     cute::copy 可以按相同的 (CPY, i+j) 坐标完成寄存器到 global 的对应搬运。
  //
  // tCsC_r2s/tCsC_s2g 没有 group 第 3 个 pipe mode，因为它们还需要显式保留
  // shared-memory C scratchpad 的 pipeline 维度；step=size<3>(tCsC_r2s) 表示
  // 每一轮可以复用的 C shared-memory pipeline 深度。外层 i 在合并后的 C 元素空间中
  // 按 step 前进，内层 j 选择当前 pipeline slot：寄存器先写入 sC(_,0,0,j)，
  // 再从同一个 slot 写回 global memory。
  // 当前 kSmemLayoutCBatch=4，因此 step=4。代码未对 i+j 做尾部 predication，
  // 这个循环依赖当前静态 layout 中 grouped extent 能被 pipe depth 整除；不应将
  // “每轮正好处理 4 个 fragment”误认为任意 tile/copy 配置都自动成立的通用规则。
  auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);
  auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);

  int step = size<3>(tCsC_r2s); // C scratchpad 的 pipe 数（由 kSmemLayoutCBatch=4 指定）
  // 双层循环的语义（注意 step 与 CPY_N 无关）：
  //   group_modes<1,3> 之后，tCrC_r2sx 的 shape 为 (CPY, (CPY_M, CPY_N))。
  //   size<1>(tCrC_r2sx) = CPY_M * CPY_N，即该线程需处理的 C fragment 总数。
  //   外循环以 step=4 为步长，内循环 j=0..3 每次处理 1 个 fragment，将其写入
  //   第 j 号 C scratchpad pipe slot，再读出写回 global。
  //
  //   这里 step=4 是 scratchpad pipeline 深度，不是 CPY_N。CPY_N 的值取决于
  //   TiledMMA 的 C-layout 如何将 (128,256) 的 gD tile 分配到 128 个线程上，
  //   通常 ≠ 4。循环之所以成立，是因为 grouped mode 用线性坐标 i+j 遍历整个
  //   CPY_M * CPY_N 的扁平空间——它不再区分 M 和 N 方向，也不要求 CPY_N == step。
  //   唯一前提是 CPY_M * CPY_N 能被 step 整除（当前静态 layout 编译期保证）。
  //
  //   第 j 号 pipe slot 在 sC 上的物理地址由 R2S partition_D / S2G partition_S
  //   各自推导；因为二者都从同一个 sC（SmemLayoutC）出发，pipe mode 保持一致，
  //   写入和读取命中同一段 shared memory。
#pragma unroll
  for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
    // 每轮处理 step 个 fragment，内层 j 选 pipe slot。
#pragma unroll
    for (int j = 0; j < step; ++j) {
      // 这两行是 R2R staging：make_tensor_like<T> 分配当前线程私有、拥有 storage
      // 的 register Tensor；普通 cute::copy 将 accumulator fragment materialize
      // 成输出元素类型 T 的临时 payload。
      //
      // cute::copy（无 copy-atom 的普通版）内部按元素做
      //   dst(i) = static_cast<T>(static_cast<SrcType>(src(i)))
      // 因此当 accumulator 与 T 类型不同时，它自动完成 dtype 转换；当二者相同
      // （如本 kernel 的 f16 accumulator + T=half），cast 退化为 no-op。
      //
      // 即使类型一致，t 还有一个不可省略的作用：layout 归一化。
      // tCrC_r2sx(_, i+j) 的 layout 是 retile_S → group_modes → slice 层层
      // 叠加的 composed layout，而 make_tensor_like<T> + cute::copy 把它
      // 物化到一个拥有简单 strides 的 owning register Tensor 中。后续
      // r2s_tiled_copy_c 的 copy_unpack 需要按 copy-atom 的 val-layout
      // 拆分 source 元素，简单 owning layout 比复杂的 composed layout 更容易
      // 被编译器推导内联。上游同源实现将其概括为"cope with accumulator and
      // output data type difference"，实际承担了类型适配和 layout 归一化两个职责。
      //
      // 这不是 retile_S 的一部分，也不是 warp shuffle：源/目的都在当前线程的
      // register 中，代码没有显式 __shfl_sync。不能仅根据这段 C++ 断言最终 SASS
      // 绝不会含 shuffle；但语义上的跨线程 regrouping 不由此 R2R copy 承担，而是
      // 由随后的 R2S -> shared memory -> S2G 路径完成。若特定 accumulator/output
      // type 与 R2S copy-atom contract 已直接兼容，可以设计直接 R2S 的变体；本例
      // 保持同源实现的通用 staging 写法。
      auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
      cute::copy(tCrC_r2sx(_, i + j), t);
      // R2S 的 copy atom 才在此处将 thread-local payload 写入 j 号 C scratchpad
      // slot；SmemLayoutC 定义写入后的跨线程地址重组。
      //
      // cute::copy(r2s_tiled_copy_c, src, dst) 的调用链路：
      //   r2s_tiled_copy_c 是 TiledCopy，但继承自 Copy_Atom<UniversalCopy<int>,T>
      //   → 匹配 copy(Copy_Atom<...> const&, src, dst) 重载 [copy.hpp:~L190]
      //   → src(t) 与 dst(tCsC_r2s(_,0,0,j)) 都是 rank-1 → 直接 copy_atom.call
      //   → copy_atom.call 内部：若 size(src)==NumValSrc（atom 编译期 val-count），
      //     走 copy_unpack；否则递归剥 mode，最终都落到 UniversalCopy<int>::copy
      //     → 硬件层面就是逐 uint32_t 的 register-to-smem store [arch/copy.hpp:~L46]
      //
      //   这里没有任何运行时"查找表"：t 的第 i 个元素之所以对应 dst 的第 i 个
      //   shared memory offset，是因为 pre-partition 阶段已经把映射烧进 layout 了：
      //     - t 的 layout 来自 retile_S → group_modes → slice → make_tensor_like
      //       → 元素顺序已按 R2S atom 的 source-side value ordering 排好
      //     - tCsC_r2s 的 layout 来自 r2s_thr_copy_c.partition_D(sC)
      //       → TiledCopy::tidfrg_D 用 LayoutCopy_TV + Tiler_MN + ValLayoutDst
      //         计算了当前线程每个 val 在 smem 中的物理 offset
      //   两者共享同一个 copy atom 的 ValLayoutSrc/ValLayoutDst 定义，
      //   因此逐元素 copy 天然把正确的数据写到了正确的地址。
      cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
    }
    // 这是 CTA 范围的 rendezvous：所有线程完成当前批 R2S 写入后，S2G 才能从
    // sC 的重组布局读取完整数据。它承担了手写版中显式跨 lane 拼接所需的协作边界。
    __syncthreads();

    // S→G: 从同一个 pipe slot 读回 global memory，保持源/目的坐标一一对应。
#pragma unroll
    for (int j = 0; j < step; ++j) {
      // S2GCopyC 的 atom 为 UniversalCopy<uint128_t>：与 R2S 的 UniversalCopy<int>
      // （32-bit）不同，它一次搬运 128-bit（8 个 half），映射为一条 wide store
      // （如 st.global.v4.f32 或 st.global.b128）。
      //
      // cute::copy(s2g_tiled_copy_c, src, dst) 的调用链路：
      //   s2g_tiled_copy_c 是 TiledCopy，继承自 Copy_Atom<UniversalCopy<uint128_t>,T>
      //   → 匹配 copy(Copy_Atom<...> const&, src, dst) 重载 [copy.hpp:~L190]
      //   → src=tCsC_s2g(_,0,0,j) 与 dst=tCgC_s2gx(_,i+j) 都是 rank-1
      //   → 直接 copy_atom.call(src, dst)
      //   → copy_atom.call 内部：size(src)==NumValSrc → copy_unpack
      //     → 逐 128-bit chunk 调用 UniversalCopy<uint128_t>::copy
      //     → 硬件层面就是 shared-memory-to-global wide store [arch/copy.hpp:~L46]
      //
      // S2G 的 pre-partition 与 R2S 对称，但 source/destination 角色互换：
      //   - tCsC_s2g 来自 s2g_thr_copy_c.partition_S(sC)：TiledCopy::tidfrg_S
      //     用 LayoutCopy_TV + Tiler_MN + ValLayoutSrc 计算了当前线程每个 val
      //     在 sC（shared memory）中的物理 offset。
      //     S2GCopyC 的 tiler = product(ThrLayout{32,4}, ValLayout{1,8})
      //                        = (32×1, 4×8) = (32, 32) — 恰好与 R2S tiler 相同，
      //     因此 tCsC_s2g 也是 (CPY, _1, _1, pipe)，_1,_1 来自 sC=(32,32,4) 无
      //     M/N remainder。
      //   - tCgC_s2gx(_, i+j) 来自 s2g_thr_copy_c.partition_D(gD) → group_modes
      //     → slice。gD=(128,256) 相对于 tiler (32,32) 有 4×8 个 tile repetition，
      //     因此 tCgC_s2g 为 (CPY, CPY_M, CPY_N)，group_modes<1,3> 合并 M/N
      //     repetition 后得到 (CPY, (CPY_M,CPY_N))，再用线性坐标 i+j 切片。
      //   两者共享同一个 copy atom 的 ValLayoutSrc/ValLayoutDst 定义，
      //   因此逐 chunk copy 天然从正确的 smem 地址读到正确的 gmem 地址。
      cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
    }
    // 所有线程都完成当前 pipe slots 的 S2G 读取后，下一轮 R2S 才能覆盖这 4 个
    // scratchpad slots，避免一部分线程仍在读取旧数据时被其他线程提前重写。
    __syncthreads();
  }

  // 与 hgemm_mma_stages_tn 的手写 epilogue 对照：手写版必须显式处理 MMA fragment
  // 的物理分布，使用 RC0/RC1 暂存，再用 __shfl_sync 从同一 warp 的相邻 lane 收齐
  // 8 个 half，并由 lane_id % 4 == 0 的线程做 float4 (128-bit) global store。
  // CuTe 版没有把这些 lane 映射写死在 kernel 源码中：retile_S 表达 accumulator
  // fragment 与 R2S copy 的对应，R2S/sC/S2G 通过 shared-memory scratchpad 完成
  // CTA 范围的数据重组，S2GCopyC 以 uint128_t 表达 wide store。两者的共同目标都是
  // 将分散在计算线程寄存器中的 C fragment 变为适合连续 global-memory 写回的布局；
  // 区别是手写版直接编排 shuffle/store，CuTe 版将映射交给 TiledMMA/TiledCopy 类型推导。
}

// =============================================================================
// Phase 7c-2: CuTe HGEMM Launch Wrapper — 类型实例化 + Grid/Block 配置
// =============================================================================
// CuTe 的核心设计理念: "类型即配置" (Type-level configuration)
// 所有的 MMA atom、TiledCopy、SmemLayout、Swizzle 都是 **编译期类型**，
// kernel 接收这些类型的实例（通常为空 struct），在编译期完成全部映射推导。
//
// 以下每个类型定义后面都标注了它在 kernel body 中的对应变量/用法，形成 1:1 对照。
//
// 面试常问：「CuTe 为什么比手写 PTX 简洁？」
// 答: 手写需要:
//   1) 手动计算每线程的 smem 地址偏移
//   2) 手动计算 ldmatrix 的 lane 映射
//   3) 手动插入 swizzle XOR 到每个地址计算点
//   4) 手动做 epilogue 的 warp shuffle 编排
// CuTe 的 partition + copy + gemm 通过 TiledCopy/TiledMMA 的类型信息
// 在编译期自动完成上述全部推导，生成与手写等价的 PTX 指令序列。
// =============================================================================
template <typename T, const int Stages = 2, const int BlockSwizzle = 0,
          const bool kAccF32 = false>
void launch_hgemm_mma_stages_tn_cute(T *a, T *b, T *c, int M, int N, int K) {
  using namespace cute;

  // ── Tile 尺寸（kernel 模板参数 BM/BN/BK/kStage 的值）──
  //
  // kernel 用法对照：
  //   BM=128:   local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), ...) → gA=(BM,BK,num_k_tiles)
  //             local_tile(D, make_tile(Int<BM>{}, Int<BN>{}), ...) → gD=(BM,BN)
  //   BN=256:   local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), ...) → gB=(BN,BK,num_k_tiles)
  //             local_tile(D, ...) → gD 的列方向
  //   BK=32:    num_k_tiles = k / BK; 每个 BK tile 内 num_k_steps = BK/kMmaPK = 2 个 MMA_K slice
  //   kStage=2: sA/sB 的第三维 = (BM,BK,kStage)；cp_async_wait<kStage-2>() 流水线同步
  //   kSmemLayoutCBatch=4: step = size<3>(tCsC_r2s) → C scratchpad pipe 深度 = 4
  //   BN: F16 acc 用 256(128×256 tile,0 spill)；F32 acc 用 128(128×128 tile,0 spill)。
  //       F32 + BN=256 会产生 ~1KB spill(254 reg),性能下降 3-8 倍。
  static constexpr int kBN = kAccF32 ? 128 : 256;
  auto BM = Int<128>{};
  auto BN = Int<kBN>{};
  auto BK = Int<32>{};
  auto KStage = Int<Stages>{};
  auto kSmemLayoutCBatch = Int<4>{};

  // ── SmemLayoutA / SmemLayoutB ──
  // kernel 用法: 
  // auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{}); // (BM, BK, kStage)
  // auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{}); // (BN, BK, kStage)
  // SmemLayout 由三部分组成：Swizzle<3,3,3> + atom layout(8×BK) + tile_to_shape 扩展到完整 tile+stage。
  //
  // 对当前 base layout shape=(8,32), stride=(32,1), T=half 的例子：
  //   - M=3：连续 2^3=8 个 half 组成一个基本元素，即 16 bytes；
  //   - S=3：连续 2^3=8 个基本元素组成一行，即 128 bytes；
  //   - B=3：共有 2^3=8 行。
  // 于是一个 8×32 的逻辑 tile 正好对应 8 行 × 128B，覆盖全部 32 个 shared-memory bank。
  // Swizzle 对每个基本元素的列坐标执行 icol' = irow ^ icol，
  // 再将 (irow, icol') 和基本元素内部的 3 个低位编码回物理 offset。
  // 其位级形式为 offset' = offset ^ ((offset & (0b111 << 6)) >> 3)：
  // 地址位 [6:8] XOR 到 [3:5]，地址位 [0:2] 不参与置换。
  // composition(Swizzle, base_layout) 的语义是 R(c) = Swizzle(base_layout(c))：
  // 逻辑坐标仍由 base_layout 给出，只有最终的 shared-memory offset 被重排。
  // tile_to_shape: 通过 blocked product 重复这个带 swizzle 的 block layout，
  // 使结果 shape 匹配 BM×BK×kStage；默认按目标 shape 的 mode order 重复各维。
  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<8>{}, Int<BK>{}),
                  make_stride(Int<BK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<BM>{}, Int<BK>{}, Int<KStage>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<BN>{}, Int<BK>{}, Int<KStage>{})));

  // ── MMA (TiledMMA) ──
  // kernel 用法: 
  // TiledMMA tiled_mma;
  // auto thr_mma = tiled_mma.get_slice(threadIdx.x);  // 每线程的 MMA slice
  // auto tCrA = thr_mma.partition_fragment_A(gA(_,_,0)); // (MMA, MMA_M, MMA_K)
  // auto tCrB = thr_mma.partition_fragment_B(gB(_,_,0)); // (MMA, MMA_N, MMA_K)
  // auto tCrD = thr_mma.partition_fragment_C(gD);        // (MMA, MMA_M, MMA_N)
  // cute::gemm(tiled_mma, tCrD, tCrA(_,_,k_step), tCrB(_,_,k_step), tCrD);
  // MMA 还决定 S2R/R2S copy 的 tiler：make_tiled_copy_A/B/C 都依赖 tiled_mma 的
  // tile_size<0/1>(mma) = (32,32) 来推导线程→数据映射。
  //
  // 推导链: SM80_16x8x16_<Acc>F16F16<Acc>_TN → MMA_Atom
  //   → make_tiled_mma(atom, EURepeat{2,2,1}, ValTile{32,32,16})
  //   → TiledMMA: 128 threads = 4 warps × (2×2 EU slices)，逻辑 MMA tile = 32×32×16
  // TN = A row-major, B col-major；因此传给本 kernel 的 B 指针实际指向
  // B^T[N,K] 的 row-major 存储（等价于 GEMM 语义中的 B[K,N] col-major）。
  // kAccF32=false → F16 累加（默认）；kAccF32=true → F32 累加。两个 atom 都是
  // m16n8k16，A/B fragment 与 shape 完全一致，kMmaPM/PN/PK 推导不受影响。
  using mma_op = std::conditional_t<kAccF32, SM80_16x8x16_F32F16F16F32_TN,
                                    SM80_16x8x16_F16F16F16F16_TN>;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;
  using mma_atom_shape = mma_traits::Shape_MNK; // (Int<16>, Int<8>, Int<16>)

  static constexpr int kMmaEURepeatM = 2;
  static constexpr int kMmaEURepeatN = 2;
  static constexpr int kMmaEURepeatK = 1;
  static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{}); // 32
  static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{}); // 32
  static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{}); // 16
  // kMmaPK=16 → kernel 中 num_k_steps = size<2>(tCrA) = BK/kMmaPK = 32/16 = 2

  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

  // ── G2SCopyA / G2SCopyB ──
  // kernel 用法: 
  // G2SCopyA g2s_tiled_copy_a; G2SCopyB g2s_tiled_copy_b;
  // cute::copy(g2s_tiled_copy_a, tAgA_copy(_,_,_,istage), tAsA_copy(_,_,_,istage));
  // cute::copy(g2s_tiled_copy_b, tBgB_copy(_,_,_,istage), tBsB_copy(_,_,_,istage));
  // G→S: 128-bit cp.async。ThrLayout{32,4} × ValLayout{1,8}:
  //   128 个线程，每线程搬运 1×8=8 个 half = 128 bits。
  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
  using G2SCopyA = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<32>{}, Int<4>{}),
                  make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyB = G2SCopyA;

  // ── S2RCopyAtomA / S2RCopyAtomB ──
  // kernel 用法: 
  // auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  // auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  // cute::copy(s2r_tiled_copy_a, tAsA(_,_,k_step_next,ismem_read), tCrA_view(_,_,k_step_next));
  // cute::copy(s2r_tiled_copy_b, tBsB(_,_,k_step_next,ismem_read), tCrB_view(_,_,k_step_next));
  // S→R: ldmatrix (SM75_U32x4_LDSM_N)。make_tiled_copy_A/B 根据 TiledMMA 自动推导
  // 与 MMA fragment 对齐的 shared→register 映射，无需手动指定 ThrLayout/ValLayout。
  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
  using S2RCopyAtomA = s2r_copy_atom;
  using S2RCopyAtomB = s2r_copy_atom;

  // ── SmemLayoutC ──
  // kernel 用法: 
  // auto sC = make_tensor(sA(_,_,ismem_read).data(), SmemLayoutC{});
  // 复用当前 A stage 的空间作为 C scratchpad
  // sC shape = (kMmaPM, kMmaPN, kSmemLayoutCBatch) = (32, 32, 4)
  // pipe 数 4 由 kSmemLayoutCBatch 指定 → step = size<3>(tCsC_r2s) = 4
  // 与 A/B 相同的 Swizzle<3,3,3> 规则，但 base layout 是 32×32（MMA tile 大小），
  // 再扩展为 4 个 epilogue scratchpad pipe slots。这 4 个 slots 不等同于主循环
  // 的 kStage K-tile pipeline。
  using SmemLayoutAtomC = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                  make_stride(Int<kMmaPN>{}, Int<1>{}))));
  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));

  // ── R2SCopyAtomC ──
  // kernel 用法:
  // auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  // cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
  // R→S: UniversalCopy<int> 以 32-bit 粒度将寄存器 payload 写入 C scratchpad。
  // make_tiled_copy_C 从 TiledMMA 的 tile_size<0/1>(mma)=(32,32) 推导 tiler,
  // 因此 R2S copy 的 tiler = (32,32),与 SmemLayoutC 的 (32,32) 恰好匹配。
  // atom T 固定为 half(smem C 的元素类型),val packing=2 half/int,32-bit store
  // 写 2 half(4 字节,4 字节对齐)。F32 acc 时,retile_S 之前需把 float 累加器
  // cast 成 half(见 epilogue 入口的 tCrD_cast),否则 layoutC_TV 的 V 布局差异
  // (F32: 4 float / F16: 4 half packed in 2 uint32)会导致 retile 错位。
  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;

  // ── S2GCopyC ──
  // kernel 用法: 
  // S2GCopyC s2g_tiled_copy_c;
  // cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i+j));
  // S→G: UniversalCopy<uint128_t> 以 128-bit 粒度做 shared→global wide store。
  // ThrLayout{32,4} × ValLayout{1,8} → tiler = product((32,4),(1,8)) = (32,32)，
  // 与 R2S tiler 相同，因此 partition_S(sC) 得到的 pipe 维度一致。
  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
  using S2GCopyC = decltype(make_tiled_copy(
      S2GCopyAtomC{},
      make_layout(make_shape(Int<32>{}, Int<4>{}),
                  make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));

  // ── Grid/Block 配置 ──
  // kernel 用法: 
  // int idx = threadIdx.x;         // 0..127
  // int ix = ((int)BlockSwizzle) * blockIdx.z * gridDim.x + blockIdx.x;
  // int iy = blockIdx.y;           // M 方向 CTA 坐标
  // if (iy * BM >= m || ix * BN >= n) return;  // CTA 级边界保护
  // 由于 kernel 没有 tile 内的逐元素 predication，调用方需保证
  // M % BM == 0、N % BN == 0、K % BK == 0。
  // 当不启用 BlockSwizzle 时，BZ=1 退化为纯 2D grid，行为不变。
  constexpr int kSwizzleStride = 2048;
  int BX = (N + BN - 1) / BN;
  int BY = (M + BM - 1) / BM;
  int BZ = BlockSwizzle ? (N + kSwizzleStride - 1) / kSwizzleStride : 1;
  BX = BlockSwizzle ? (BX + BZ - 1) / BZ : BX;
  dim3 block(size(MMA{}));   // 128 threads (= size(TiledMMA))
  dim3 grid(BX, BY, BZ);

  // ── Dynamic Shared Memory 大小 ──
  // kernel 用法: 
  // extern __shared__ T shm_data[];
  // T *Ashm = shm_data;
  // T *Bshm = shm_data + cute::cosize(SmemLayoutA{});
  // cosize(SmemLayoutA) = BM*BK*kStage = 128*32*2 = 8192 个 T
  // cosize(SmemLayoutB) = BN*BK*kStage = 256*32*2 = 16384 个 T
  // 合计 A+B = 24576 个 T；C epilogue 复用一个 A stage（128*32 = 4096 个 T）。
  // static_assert 保证 C scratchpad (kMmaPM*kMmaPN*kSmemLayoutCBatch = 32*32*4 = 4096)
  // 能放进一个 A stage (128*32 = 4096)。
  static constexpr int shm_size_AB =
      cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
  static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});
  static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >= size(SmemLayoutC{}),
                "C shared memory must fit within one A pipe");
  static constexpr int kShmSize =
      cute::max(shm_size_AB, shm_size_C) * sizeof(T);

  cudaFuncSetAttribute(
      hgemm_mma_stages_tn_cute<T, BM, BN, BK, KStage, MMA, G2SCopyA, G2SCopyB,
                                SmemLayoutA, SmemLayoutB, SmemLayoutC,
                                S2RCopyAtomA, S2RCopyAtomB, R2SCopyAtomC,
                                S2GCopyAtomC, S2GCopyC, BlockSwizzle>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

  hgemm_mma_stages_tn_cute<T, BM, BN, BK, KStage, MMA, G2SCopyA, G2SCopyB,
                            SmemLayoutA, SmemLayoutB, SmemLayoutC, S2RCopyAtomA,
                            S2RCopyAtomB, R2SCopyAtomC, S2GCopyAtomC, S2GCopyC,
                            BlockSwizzle>
      <<<grid, block, kShmSize>>>(a, b, c, M, N, K);
}

#endif /* NOTES_V2_ENABLE_CUTE */
#if defined(NOTES_V2_ENABLE_WGMMA)
// ---- WGMMA Kernel: Warp Specialization + TMA ----
// 面试重点 — Warp Specialization（Hopper 最核心的编程模型变化）：
//
// 背景：传统 GEMM kernel 中，所有线程同步地"加载→计算→存回"，
// 数据搬运和计算无法重叠。Hopper 的 TMA + WGMMA 支持将工作拆分为两个
// warpgroup，让数据搬运和矩阵乘完全异步执行：
//
//   WG0 (128 threads, Producer): 仅 thread 0 提交 TMA 2D 拷贝指令，
//     将 A/B tile 从 HBM 搬到 shared memory。
//   WG1 (128 threads, Consumer): 所有 128 个线程参与 WGMMA 矩阵乘。
//
// Producer 和 Consumer 通过 cuda::barrier（CTA 级别）同步：
//   - full[stage]:  Producer 发信号表示 stage stage 的数据已就绪，可被使用
//   - empty[stage]: Consumer 发信号表示 stage stage 的使用完毕，可以被覆盖
//
// 本节重点理解：
//   1) 为什么 Producer 只需要 thread 0？TMA 是硬件 DMA 指令，一次提交
//      即可搬运整个 2D tile，无需所有线程参与。
//   2) barrier 的 arrive count = 129：128 个 Consumer 线程 + 1 个 Producer 提交线程。
//   3) 多 stage pipeline 使 Consumer 计算 stage stage 的同时，
//      Producer 可以搬运 stage (stage+1)，隐藏 HBM→SMEM 延迟。
//
// Tile Hierarchy（与 MMA m16n8k16 kernel 对比）：
//   WGMMA Atom:       m64n128k16（一次处理 64×128×16，是 MMA 的 64 倍）
//   K Tile:           BK=64，每个 K tile 包含 BK/kWgmmaK=4 个 WGMMA atom
//   M Tile:           BM=128，每个 M tile 包含 BM/kWgmmaM=2 个 WGMMA atom
//   N Tile:           BN=128，单个 kWgmmaN=128 即可覆盖，无需在 N 方向分块
//   Block Tile:       C[128,128] = A[128,64] × B[64,128]
//   Threads:          256 = 2 warpgroups × 128 threads/warpgroup
//     Producer(WG0):  128 threads（仅 thread 0 做 TMA 提交）
//     Consumer(WG1):  128 threads = 4 warps（全部参与 WGMMA 和写回）
//
// kStages=3: 3 级流水线。Consumer 滞后 Producer 最多 2 步，确保 HBM→SMEM
// 的延迟被计算完全掩盖。
//
// Grid:  ((N+127)/128/S, (M+127)/128, S)，S=(N+2047)/2048，3D block swizzle
//   - grid.z = S 个 swizzle 分区，将连续 block 打散到不同 N 区域改善 L2 命中
// Block: (256, 1, 1)，2 warpgroups
// source: LeetCUDA/kernels/hgemm/wgmma/hgemm_wgmma_fp16acc_stages_tn.cu
template <const int kWgmmaM = 64,           // WGMMA atom M dim (m64n128k16)
          const int kWgmmaN = 128,          // WGMMA atom N dim
          const int kWgmmaK = 16,           // WGMMA atom K dim
          const int BM = 128,               // block tile M, 2 × kWgmmaM
          const int BN = 128,               // block tile N, 1 × kWgmmaN
          const int BK = 64,                // block tile K, 4 × kWgmmaK
          const int kNumThreads = 256,      // 2 wargroups × 128 threads
          const int kStages = 3,            // TMA pipeline depth (full/empty barriers)
          const int kBlockSwizzle = 0>      // 1 enables 3D grid swizzle for L2 locality
__global__ void __launch_bounds__(kNumThreads, 1)
    hgemm_wgmma_stages_tn(
        int M, int N, int K, half *C,
        const CUtensorMap *__restrict__ tensorMapA,
        const CUtensorMap *__restrict__ tensorMapB) {
  static_assert(kBlockSwizzle == 0 || kBlockSwizzle == 1, "kBlockSwizzle must be 0 or 1");
  // 注意：tensorMapA/tensorMapB 需要由 host 侧按当前 tile 布局预先创建；
  // 对 row-major [M, K] 矩阵，TMA shape 参数写的是 (K, M) 而不是 (M, K)，
  // 也就是TMA descriptor中把连续的维度写在最内层，非连续的维度写在最外层。
  // 这是 TMA descriptor 最容易背错的地方之一。notes 这里只保留 kernel 主体，
  // 不展开宿主侧 create_tensor_map 细节。

  // Block Swizzle: 在 grid x 维度做 swizzle，改善 L2 cache 局部性
  // bx = blockIdx.z * gridDim.x + blockIdx.x，将相邻 block 打散到不同 N 区域
  const int bx = ((int)kBlockSwizzle) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  constexpr int kConsumerThreads = kNumThreads / 2; // 128 threads = 1 warpgroup
  // kNumConsumers = (kNumThreads / 128) - 1 = 2 - 1 = 1（1 个 consumer WG）
  // 这里的 -1 是因为 2 个 warpgroup 中 1 个是 producer，剩余都是 consumer
  constexpr int kNumConsumers = (kNumThreads / kConsumerThreads) - 1; // 1 consumer WG
  // kWarpgroupM = BM / kNumConsumers：每个 consumer warpgroup 负责的 M 行数
  // 当只有一个 consumer 时，它负责全部 BM=128 行
  constexpr int kWarpgroupM = BM / kNumConsumers; // 128

  // 边界检查：确保当前 block 不超出 M/N 范围
  if (bx >= div_ceil(N, BN) || by >= div_ceil(M, BM))
    return;

  // ---- Shared Memory 分配 ----
  // 动态 shared memory（由 host 侧通过 kernel launch 的 smem 参数指定大小）
  // TMA + WGMMA 仅需 __align__(128)，而非 __align__(1024)。原因：
  //   make_smem_desc() 构造 WGMMA descriptor 时，start_address 字段
  //   完整编码了 __cvta_generic_to_shared 的绝对 32-bit 偏移量，
  //   硬件根据该绝对地址自动推导并补偿 swizzle phase 偏移（等价于
  //   descriptor 内置的 base_offset 机制）。因此即使 smem 基址未落到
  //   1024B 边界，WGMMA 硬件也能正确读取 TMA 写入的数据。
  // 对比 hgemm_tma_mma_ws_tn：消费者使用 ldmatrix + 手写 swizzle<64>()
  //   ，该函数无法感知 smem 绝对地址，硬编码了 phase=0 的假设，因此必须
  //   __align__(1024) 来保证 phase 确实为 0。从健壮性角度看本 kernel 也
  //   应该用 __align__(1024)；这里使用 128 也可以正常运行。
  extern __shared__ __align__(1024) uint8_t smem_tma_wgmma_ws[];
  WgmmaSMem<BM, BN, BK, kStages> &s =
      *reinterpret_cast<WgmmaSMem<BM, BN, BK, kStages> *>(smem_tma_wgmma_ws);
  half *s_a = s.A;
  half *s_b = s.B;

  // ---- cuda::barrier 初始化 ----
  //
  // ★ Barrier 机制速查（arrive / wait 核心语义）：
  //
  // cuda::barrier 是一个 CTA 级同步原语，基于 **phase（奇偶交替）** 工作：
  //
  //   init(bar, N): 设置 arrive_count = N。
  //     含义：每 phase 需要 N 个线程各调用一次 arrive()，barrier 才翻转 phase。
  //
  //   arrive(): 线程声明"我已到达此 barrier 点"。
  //     - 非阻塞，立即返回一个 token（包含当前 phase 值）。
  //     - 不关心"谁"到达，只关心"到达次数"是否达到 arrive_count。
  //
  //   wait(token): 线程阻塞，直到 barrier 的当前 phase ≠ token 中的 phase。
  //     - 即：等待 phase 翻转。
  //     - Phase 翻转条件：(1) arrive 次数达到 arrive_count
  //                      (2) 若使用了 barrier_arrive_tx，还需 TMA 字节全部写完
  //
  //   典型用法：b.wait(b.arrive())
  //     - arrive() 注册自己到达，拿到 token（当前 phase = P）
  //     - wait(token) 阻塞直到 phase 翻转（P → P+1）
  //     - 如果自己是第 N 个到达者 → phase 立即翻转 → wait 立即返回（不阻塞）
  //     - 如果自己不是最后一个 → wait 阻塞，直到第 N 个到达者触发翻转
  //
  // ★ 本 kernel 的 Pipeline 同步协议（kStages=3，arrive_count=129）：
  //
  //   Producer (1 thread)               Consumer (128 threads)
  //   ──────────────────                ──────────────────────
  //                                      C0: arrive(empty[*]) ×128  [init: 标记所有 stage 为空]
  //   ┌─ for each k_tile: ─┐            ┌─ for each k_tile: ─┐
  //   │ P1: arrive+wait(empty[q])       │ C1: arrive+wait(full[q])
  //   │     ↓ 等 stage q 被消费完       │     ↓ 等 TMA 数据就绪
  //   │ P2: TMA(A[q]) + TMA(B[q])       │ C2: WGMMA 计算
  //   │     ↓ 异步拷贝                  │ C3: wait WGMMA 完成
  //   │ P3: arrive_tx(full[q])          │ C4: arrive(empty[q])
  //   │     ↑ 通知：stage q 数据就绪     │     ↑ 通知：stage q 已消费完
  //   └──────────────────────┘           └──────────────────────┘
  //
  //   关键不变式（invariant）：
  //     - Producer 不会覆盖 Consumer 正在读的 stage
  //     - Consumer 不会读 Producer 还没写完的 stage
  //     - 通过 full/empty 两个 barrier 的 phase 交替来保证
  //
  //   每个 stage 有两个 barrier：
  //      full[stage]: TMA 数据就绪信号。Producer 发（arrive_tx），Consumer 等（wait）。
  //     empty[stage]: Stage 空闲信号。 Consumer 发（arrive），   Producer 等（wait）。
  //
  //   arrive_count = 128 (consumer) + 1 (producer) = 129：
  //     每 phase，128 个 consumer 线程 + 1 个 producer 线程都要 arrive 一次。
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ cuda::barrier<cuda::thread_scope_block> full[kStages];
  __shared__ cuda::barrier<cuda::thread_scope_block> empty[kStages];
#pragma nv_diag_default static_var_with_dynamic_init

  // K 方向总 tile 数。要求 K 能被 BK 整除，否则尾 tile 被丢弃。
  const int NUM_K_TILES = div_ceil(K, BK);
  const int wg_idx = threadIdx.x / kConsumerThreads; // 0=Producer, 1=Consumer
  const int wg_tid = threadIdx.x % kConsumerThreads; // 0~127 within warpgroup

  // 初始化 barriers（仅 thread 0 执行）
  if (threadIdx.x == 0) {
    for (int i = 0; i < kStages; ++i) {
      // arrive_count = 129：128 consumer + 1 producer
      // init 是 CUDA barrier 的 hidden friend 函数（定义在 cuda::barrier 类体内），
      // 只能通过 ADL（Argument-Dependent Lookup）调用。因为第一个参数 &full[i] 的类型是
      // cuda::barrier<cuda::thread_scope_block>*，编译器通过 ADL 在 cuda 命名空间中自动找到它。
      init(&full[i], kConsumerThreads + 1);  // 128 consumer + 1 producer
      init(&empty[i], kConsumerThreads + 1); // same
    }
    // fence_proxy_async_shared_cta: 确保 barrier 在 smem 中的初始化
    // 对 async proxy（TMA/WGMMA）可见。
    tma_fence_proxy_async_shared_cta();
  }
  __syncthreads();

  // ==================================================================
  // ★ Warp Specialization 并发模型 — 为什么只有 if/else 没有 while？
  // ==================================================================
  //
  // 256 个线程在同一个 SM 上**并发执行**。if/else 只是分工，不是先后顺序：
  //   - WG0 (threadIdx.x 0~127):   全部走 if 分支，做 Producer。
  //   - WG1 (threadIdx.x 128~255): 全部走 else 分支，做 Consumer。
  //
  // SM 的 warp scheduler 会交替调度来自 WG0 和 WG1 的 warp，两者**同时**推进：
  //   Producer: for (k_iter = 0 .. NUM_K_TILES) { P1→P2→P3 }  ← 自己的循环
  //   Consumer: for (k_iter = 0 .. NUM_K_TILES) { C1→C2→C3→C4 } ← 自己的循环
  //
  // 两个 warpgroups 各自独立迭代 K-tiles，通过 full[] / empty[] barrier 同步：
  //   - Producer 领先 Consumer 太多 → wait(empty[q]) 阻塞，等 Consumer 消费完
  //   - Consumer 领先 Producer 太多 → wait(full[q]) 阻塞，等 TMA 搬运完
  //
  // 这是生产者-消费者模型的 GPU 实现：不需要共享循环变量，barrier 的 phase
  // 交替天然保证了流水线串行化（at most kStages-1 steps ahead）。
  // ==================================================================
  // Producer Warpgroup (WG0, threadIdx.x 0~127)
  // 职责：提交 TMA 2D 拷贝，将 A/B tile 从 HBM 异步搬运到 SMEM。
  // 只有 wg_tid==0 执行实际拷贝提交，其余 127 个线程空闲。
  // ==================================================================
  if (wg_idx == 0) {
    NOTES_V2_REG_DEALLOC(40);
    if (wg_tid == 0) {
      // stage: 当前操作的 stage 索引（round-robin 0 -> 1 -> 2 -> 0 -> ...）
      int stage = 0;
      for (int k = 0; k < NUM_K_TILES; ++k, ++stage) {
        if (stage == kStages)
          stage = 0;

        // Step P1: 等待 stage stage 变为"空"（可被覆盖写入）
        //
        // 模式 empty[stage].wait(empty[stage].arrive()):
        //   - arrive(): Producer（1 个线程）在 empty barrier 上注册到达，
        //     获得当前 phase 的 token。如果 Consumer 已经在 C4 步骤积累了
        //     128 次 arrive（来自上一轮迭代或 C0 初始化），则加上这次共 129 次
        //     → phase 立即翻转。
        //   - wait(token): 阻塞直到 barrier phase ≠ token 中的 phase。
        //     由于 arrive() 是第 129 次到达，phase 在 arrive 时已翻转，
        //     wait 看到 phase 已变 → 立即返回（首轮不阻塞）。
        //
        // 语义：Producer 说"我准备好写 stage stage 了，Consumer 用完没有？"
        //       如果 Consumer 还没用完 → phase 未翻转 → wait 阻塞等待。
        //       如果 Consumer 已用完 → phase 已翻转 → wait 立即返回。
        empty[stage].wait(empty[stage].arrive());

        // Step P2: 提交 TMA 2D 拷贝指令
        // cp_async_bulk_tensor_2d_global_to_shared:
        //   参数1(dst): smem 目标地址（当前 stage 的 A/B tile 起始位置）
        //   参数2(tensorMap): Host 预创建的 TMA descriptor（描述源矩阵的 shape/stride/dtype）
        //   参数3/4(coords): (k_offset, m_offset) 或 (k_offset, n_offset) 的全局坐标
        //   参数5(barrier): 拷贝完成后自动 arrive 到此 barrier
        //
        // TMA 是硬件 DMA 引擎：一次指令提交即可搬运整个 2D tile，
        // 无需线程逐元素搬运，零寄存器开销。
        // TMA 2D 加载 A tile: coords = (k_offset, m_offset)
        tma_load_2d(&s_a[stage * BK * BM], tensorMapA, k * BK, by * BM, 
                    full[stage]);

        // TMA 2D 加载 B tile: coords = (k_offset, n_offset)
        tma_load_2d(&s_b[stage * BK * BN], tensorMapB, k * BK, bx * BN, 
                    full[stage]);

        // Step P3: 通知 Consumer：stage stage 的 TMA 数据已就绪
        //
        // barrier_arrive_tx(bar, arrive_count_update, byte_count):
        //   - 在 bar 上注册 1 次到达（arrive_count_update=1）
        //   - 同时声明预期有 byte_count 字节将通过 async copy（TMA）写入 smem
        //   - Phase 翻转条件：
        //       (a) 总 arrive 次数达到 129（128 consumer + 1 producer）
        //       (b) 所有声明的 async 字节已写入 smem
        //     两个条件都满足后 phase 才翻转，Consumer 的 wait() 才返回。
        //   - 即：Consumer 不会在 TMA 数据完整到达前就开始读 smem。
        tma_arrive_expect_tx(full[stage], (BK * BN + BK * BM) * sizeof(half));
      }
    }
  }
  // ==================================================================
  // Consumer Warpgroup (WG1, threadIdx.x 128~255)
  // 职责：等待 TMA 数据就绪 -> 发射 WGMMA 做矩阵乘 -> 积累结果 -> 写回 C
  // 所有 128 个线程（4 warps）全部参与。
  // ==================================================================
  else {
    // Step C0: Consumer 初始化 — 标记所有 stage 为"空"（可被 Producer 写入）
    //
    // 所有 128 个 Consumer 线程对每个 stage 的 empty barrier 调用 arrive()。
    // 这是 Pipeline 的"预热"步骤——没有它，Producer 的 empty[stage].wait()
    // 在第一轮会永远阻塞（因为 Producer 的 1 次 arrive 不足以凑够 129）。
    //
    // 注意：此时每个 empty[i] 只有 128 次 arrive，未达 129，phase 不翻转。
    // Producer 后续的 empty[stage].arrive() 作为第 129 次，触发 phase 翻转。
    NOTES_V2_REG_ALLOC(232);

    for (int i = 0; i < kStages; ++i) {
      [[maybe_unused]] auto token = empty[i].arrive();
    }

    // 累加器寄存器声明
    // d[kWarpgroupM / kWgmmaM][kWgmmaN / 16][4]:
    //   - d[0][*][*]: M 方向第 1 个 WGMMA atom（rows 0~63）
    //   - d[1][*][*]: M 方向第 2 个 WGMMA atom（rows 64~127）
    //   - d[*][g][*]: N 方向第 g 组 16 列
    //   - d[*][*][0..3]: 4 条 uint32 寄存器，共 8 个 half（覆盖 16×16 子块）
    // 每个线程总共 2 * 8 * 4 = 2 * 32 = 64 uint32 = 128 half (每次WGMMA Atom 32 uint32 输出)
    // 128 线程 * 128 half = 16384 half = 128 * 128 = BM*BN（刚好覆盖整个 C tile）
    uint32_t d[kWarpgroupM / kWgmmaM][kWgmmaN / 16][4] = {};

    int stage = 0;
    // K 维外循环：沿 K tile 迭代（BK=64，每个 K tile 做 4 次 WGMMA 累加）
    for (int k = 0; k < NUM_K_TILES; ++k, ++stage) {
      if (stage == kStages)
        stage = 0;

      // Step C1: 等待 TMA 数据就绪（full 信号）
      //
      // 模式 full[stage].wait(full[stage].arrive()):
      //   - 128 个 Consumer 线程各调用 arrive()，共 128 次到达。
      //   - 当前 phase 累积：128/129，phase 尚未翻转。
      //   - wait(token) 阻塞，直到 Producer 的 barrier_arrive_tx（Step P3）
      //     贡献第 129 次到达 + TMA 字节全部写完 → phase 翻转 → wait 返回。
      //
      // 语义：Consumer 说"我准备好读 stage stage 了，数据到了没有？"
      full[stage].wait(full[stage].arrive());

      // Step C2: 发射 WGMMA 指令序列
      //
      // WGMMA 是异步指令（fire-and-forget），发射后立即返回，不等待计算完成。
      // 标准流程：FENCE → 发射 WGMMA → COMMIT → WAIT。
      //
      // WGMMA_FENCE（wgmma.fence.sync.aligned）:
      //   - 确保 TMA 写入 smem 的数据对 async proxy（WGMMA）可见
      //   - 确保累加器寄存器 d[] 已准备好接收 WGMMA 输出
      //   - 本质是 proxy 间的内存序（memory ordering），不是传统 __syncthreads
      //
      // 每个 WGMMA_M64N128K16_F16F16F16 指令：
      //   D[64×128] += A[64×16] × B[16×128]
      //   A/B 均为 K-major（imm-trans=0），由 descA/descB 描述 smem 中的布局。
      //   ScaleD=1: 累加。K 维有 BK/kWgmmaK=4 次迭代，每次都用 ScaleD=1。
      WGMMA_FENCE();

      // M 维迭代：BM/kWgmmaM = 128/64 = 2 个 WGMMA atom
      // 每个 atom 处理 64 行 M 方向数据。
#pragma unroll
      for (int m = 0; m < kWarpgroupM / kWgmmaM; ++m) {
        // wgmma_sA 指向当前 stage 中 A tile 的第 m 个 64*BK 子块
        // s_a 布局：[kStages][BM][BK] -> stage * BK*BM + BK * (m*64)
        half *wgmma_sA = s_a + stage * BK * BM + BK * m * kWgmmaM;

        // K 维迭代：BK/kWgmmaK = 64/16 = 4 次 WGMMA（累加）
        // 每次处理 K=16 维的矩阵乘，4 次累加后覆盖完整的 BK=64。
#pragma unroll
        for (int k_step = 0; k_step < BK / kWgmmaK; ++k_step) {
          // 第 k_step 次 WGMMA:
          //   A: wgmma_sA + k_step * kWgmmaK（A 的 K 维起始位置）
          //   B: s_b + stage * BK * BN + k_step * kWgmmaK（B 的 K 维起始位置）
          //   注意 B 的 smem 布局是 [BK, BN] row-major，K-major 读取时从
          //   第 k_it*16 行开始取 16 行，每行 BN=128 列。
          //   ScaleD=1: 累加（K 维迭代需要累积结果）
          WGMMA_M64N128K16_F16F16F16(d[m], wgmma_sA + k_step * kWgmmaK,
                                     s_b + stage * BK * BN + k_step * kWgmmaK,
                                     1, // ScaleD=1: accumulate（不清零）
                                     1, 1, 0, 0);
        }
      }

      // Step C3: 提交并等待 WGMMA 完成
      //
      // WGMMA_COMMIT_GROUP (wgmma.commit_group.sync.aligned):
      //   将自上一次 COMMIT 以来发射的所有 WGMMA 归为一组，提交到 async proxy 执行。
      //   类比 cp.async.commit_group，但 scope 是 warpgroup 而非 per-thread。
      //
      // WGMMA_WAIT_GROUP(0) (wgmma.wait_group.sync.aligned 0):
      //   阻塞直到最多 N 个 group 尚未完成（pending ≤ N）。N=0 即等待所有 group。
      //   ★ 语义与 cp.async.wait_group 完全一致（PTX ISA §9.7.15.7.3 vs §9.7.9.25.3.3）：
      //     两者都是 "wait until only N or fewer of the most recent groups are pending"。
      //   此处用 0 是因为 Consumer 在本次迭代中只 commit 了 1 个 group，
      //   必须等它全部完成才能进入下一步（读下一 stage 的数据）。
      WGMMA_COMMIT_GROUP();
      WGMMA_WAIT_GROUP(0);

      // Step C4: 释放 stage stage — 通知 Producer 可以覆盖写入
      //
      // 128 个 Consumer 线程在 empty[stage] 上 arrive()，为 **下一 phase** 积累到达次数。
      // 这 128 次 arrive 不会立即翻转 phase（还需 Producer 在下一轮的 1 次 arrive）。
      //
      // 语义：Consumer 说"stage stage 我已经读完了，你可以放心覆盖了。"
      [[maybe_unused]] auto token = empty[stage].arrive();
    }

    // ==================================================================
    // Epilogue: 将寄存器中的累加结果写回 global memory C
    // ==================================================================
    //
    // WGMMA m64n128k16.f16.f16.f16 的输出寄存器映射：
    //
    // 对 warpgroup 内每个线程，输出 64 个 half = 32 个 uint32。
    // 这些 half 分布在 d[2][8][4] 数组中：
    //   d[m_it][g][0]: (row, col) 和 (row, col+2) 位置的 2 个 half
    //   d[m_it][g][1]: (row+8, col) 和 (row+8, col+2) 位置的 2 个 half
    //   d[m_it][g][2]: (row, col+8) 和 (row, col+10) 位置的 2 个 half
    //   d[m_it][g][3]: (row+8, col+8) 和 (row+8, col+10) 位置的 2 个 half
    //
    // 其中：
    //   row = warp * 16 + lane / 4    (0~63, 4 warps * 16 rows)
    //   col = g * 16 + 2 * (lane % 4)  (0~126, step 2, 8 g's * 16 cols)
    //
    // 每个线程覆盖一个 16x16 子块中的 16 个 half：
    //   +-------+-------+
    //   | reg[0] | reg[2] |  <- rows [row, row+8)
    //   |(col,+2)|(col+8)|
    //   +-------+-------+
    //   | reg[1] | reg[3] |  <- rows [row+8, row+16)
    //   |(col,+2)|(col+8)|
    //   +-------+-------+
    //   每个 reg 包含 2 个连续 half（col 和 col+2），用 uint32 存储。
    //
    // 三维遍历：
    //   m_it=0: rows 0~63,  m_it=1: rows 64~127
    //   g=0..7: 每个覆盖 16 列，8*16=128 列 ok
    // 每个线程写 4 uint32 * 2 half/uint32 * 8 g * 2 m_it = 128 half。
    // 128 线程 * 128 half = 16384 half = 128 * 128 ok

    const int lane = wg_tid % 32;
    const int warp = wg_tid / 32;
    // row: 当前线程在当前 WGMMA atom 中负责的起始行。
    // warp 0~3 各负责 16 行：rows [warp*16, warp*16+15]。
    // lane/4：每 4 个连续 lane 负责同一行（因为 WGMMA fragment 中每行有 4 个 uint32，
    //         分别覆盖列对 {c0,c1}、{c2,c3}、{c8,c9}、{c10,c11}，由 lane%4 区分）。
    //         因此 lane/4 给出该行在 16-row block 内的行号（0~7 对应 rows 0~7，
    //         同一 warp 中 lane/4==0 的有 lane 0,1,2,3，都负责 row 0）。
    const int row = warp * 16 + lane / 4;
    // block_C: 当前 C tile 在 global memory 中的起始地址
    // by*BM 是 M 方向偏移，bx*BN 是 N 方向偏移
    half *block_C = C + by * BM * N + bx * BN;

    // 2 * (8 * 4) = 2 * 32 = 64 uint32 = 128 half
    // NOTE: 这里可以考虑通过 R->S, S->G 的方式做一次 128 bits写回。
#pragma unroll
    for (int m = 0; m < kWarpgroupM / kWgmmaM; ++m) {
      int yo = m * kWgmmaM; // M 方向行偏移 (0 或 64)
#pragma unroll
      for (int g = 0; g < kWgmmaN / 16; ++g) {
        int col = g * 16 + 2 * (lane % 4); // N 方向列偏移 (0,2,4,...,14 then 16,18,...)
        // 一次 uint32 store 写入 2 个 half（连续列 col 和 col+2）
        // 左上象限: (row, col)
        *reinterpret_cast<uint32_t *>(&block_C[(row + yo) * N + col]) = d[m][g][0];
        // 左下象限: (row+8, col)
        *reinterpret_cast<uint32_t *>(&block_C[(row + yo + 8) * N + col]) = d[m][g][1];
        // 右上象限: (row, col+8)
        *reinterpret_cast<uint32_t *>(&block_C[(row + yo) * N + col + 8]) = d[m][g][2];
        // 右下象限: (row+8, col+8)
        *reinterpret_cast<uint32_t *>(&block_C[(row + yo + 8) * N + col + 8]) = d[m][g][3];
      }
    }
  }
}

#endif /* NOTES_V2_ENABLE_WGMMA */

#if defined(NOTES_V2_ENABLE_TMA_MMA_WS)

// 消费者 MMA 层级映射参考 hgemm_mma_stages_tn_swizzle。与那个
// 八 warp kernel 不同，本 warp-specialized 消费者仅拥有四个 warp。
template <const int kMmaM = 16,             // MMA atom M dim (m16n8k16)
          const int kMmaN = 8,              // MMA atom N dim
          const int kMmaK = 16,             // MMA atom K dim
          const int kMmaTileM = 2,          // consumer warps along M, warp tile M = 32
          const int kMmaTileN = 2,          // consumer warps along N, warp tile N = 16
          const int kValTileM = 4,          // value-repeat along M, BM = 16*2*4 = 128
          const int kValTileN = 8,          // value-repeat along N, BN = 8*2*8 = 128
          const int kValTileK = 4,          // MMA_K slices per BK tile, BK = 16*4 = 64
          const int kStages = 2,            // TMA full/empty pipeline depth
          const int kNumThreads = 256,      // 128 producer + 128 consumer threads
          const int kBlockSwizzle = 0>      // 1 enables 3D grid swizzle for L2 locality
__global__ void __launch_bounds__(kNumThreads)
    hgemm_tma_mma_ws_tn(
        int M, int N, int K, half *C,
        const CUtensorMap *__restrict__ tensorMapA,
        const CUtensorMap *__restrict__ tensorMapB) {

  constexpr int BM = kMmaM * kMmaTileM * kValTileM;
  constexpr int BN = kMmaN * kMmaTileN * kValTileN;
  constexpr int BK = kMmaK * kValTileK;
  static_assert(kMmaM == 16 && kMmaN == 8 && kMmaK == 16, "This kernel uses mma.sync.m16n8k16");
  static_assert(kMmaTileM * kMmaTileN == 4, "The consumer warpgroup has exactly four warps");
  static_assert(BM == 128 && BN == 128 && BK == 64, "TMA desc and 128B swizzle require 128x128x64");
  static_assert(kNumThreads == 256, "Use one producer and one consumer warpgroup");
  static_assert(kBlockSwizzle == 0 || kBlockSwizzle == 1, "kBlockSwizzle must be 0 or 1");

  const int bx = kBlockSwizzle * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  constexpr int kConsumerThreads = kNumThreads / 2;
  static_assert(kConsumerThreads == kMmaTileM * kMmaTileN * kWarpSize,
                "Consumer threads must cover the MMA warp grid");

  if (bx >= div_ceil(N, BN) || by >= div_ceil(M, BM))
    return;

  // TMA CU_TENSOR_MAP_SWIZZLE_128B 要求 smem 基地址 1024 字节对齐，
  // 以确保硬件 swizzle phase 从零开始。消费者 swizzle<64>()
  // 假设零 phase；其他对齐（如 128）会导致 phase 偏移，产生不匹配的
  // 物理地址，破坏整个输出。
  //
  // 为什么 hgemm_wgmma_stages_tn 只需要 __align__(128)，而本 kernel
  // 必须 __align__(1024)？核心区别在于消费者如何感知 swizzle phase：
  //
  // WGMMA 消费者：通过 make_smem_desc() 构造 64-bit WGMMA descriptor，
  //   其中 bits [49,52) 的 base_offset 字段编码了 smem 基址相对于
  //   1024B 边界的偏移量（(addr >> 7) & 0x7）。硬件在读取 smem 时会
  //   用这个 base_offset 自动补偿 swizzle phase，因此 WGMMA 路径不
  //   需要 1024 字节对齐也能得到正确数据。
  //
  // 本 kernel 消费者：使用 ldmatrix + 手写 swizzle<64>() 来计算
  //   smem 物理地址。这个函数是纯软件公式，没有任何 base_offset
  //   补偿机制。它假设 smem 基址恰好落在 1024B 边界上（phase=0）。
  //   如果基址只满足 128B 对齐，TMA 硬件会以非零 phase 写入数据，
  //   而 ldmatrix 以零 phase 读取 → 物理地址彻底错位 → 全输出错误。
  //
  // 简言之：WGMMA 用硬件 descriptor base_offset 自动解决 phase 偏移；
  // ldmatrix 路径没有等价机制，必须靠 1024B 对齐来保证 phase=0。
  extern __shared__ __align__(1024) uint8_t smem_tma_mma_ws[];
  TmaMmaWSSMem<BM, BN, BK, kStages> &s =
      *reinterpret_cast<TmaMmaWSSMem<BM, BN, BK, kStages> *>(smem_tma_mma_ws);
  half *s_a = s.A;
  half *s_b = s.B;

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ cuda::barrier<cuda::thread_scope_block> full[kStages];
  __shared__ cuda::barrier<cuda::thread_scope_block> empty[kStages];
#pragma nv_diag_default static_var_with_dynamic_init

  const int NUM_K_TILES = div_ceil(K, BK);
  const int wg_idx = threadIdx.x / kConsumerThreads;
  const int wg_tid = threadIdx.x % kConsumerThreads;

  if (threadIdx.x == 0) {
    for (int stage = 0; stage < kStages; ++stage) {
      init(&full[stage], kConsumerThreads + 1);
      init(&empty[stage], kConsumerThreads + 1);
    }
    tma_fence_proxy_async_shared_cta();
  }
  __syncthreads();

  if (wg_idx == 0) {
    // 生产者 Warpgroup（wg_idx==0）
    NOTES_V2_REG_DEALLOC(40);
    if (wg_tid == 0) {
      int stage = 0;
      for (int k = 0; k < NUM_K_TILES; ++k, ++stage) {
        if (stage == kStages)
          stage = 0;

        empty[stage].wait(empty[stage].arrive());

        tma_load_2d(&s_a[stage * BM * BK], tensorMapA, k * BK, by * BM,
                    full[stage]);
        tma_load_2d(&s_b[stage * BN * BK], tensorMapB, k * BK, bx * BN,
                    full[stage]);
        tma_arrive_expect_tx(full[stage], (BM * BK + BN * BK) * sizeof(half));
      }
    }
  } else {
    // 消费者 Warpgroup（wg_idx==1）
    // 初始化：标记所有 stage 为"空"（可被 Producer 写入）
    NOTES_V2_REG_ALLOC(232);

    for (int stage = 0; stage < kStages; ++stage) {
      [[maybe_unused]] auto token = empty[stage].arrive();
    }

    const int warp_id = wg_tid / kWarpSize;
    const int lane_id = wg_tid % kWarpSize;
    // WG1 内形成 2x2 warp grid，warp_m / warp_n 分别表示 warp 在 M/N 方向的索引。
    const int warp_m = warp_id % kMmaTileM; // kMmaTileM = 2, {0,1}
    const int warp_n = warp_id / kMmaTileM; // kMmaTileN = 2, {0,1}
    uint32_t RA[kValTileM][4];
    uint32_t RB[kValTileN][2];
    uint32_t RC[kValTileM][kValTileN][2] = {0};

    const uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
    const uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);
    int stage = 0;
    for (int k = 0; k < NUM_K_TILES; ++k, ++stage) {
      if (stage == kStages)
        stage = 0;

      full[stage].wait(full[stage].arrive());
      tma_fence_proxy_async_shared_cta();

      // 与 cp.async pipeline（hgemm_mma_stages_tn 等）不同，这里 ldmatrix +
      // mma.sync 之前/后 不需要 __syncthreads，原因在于 TMA + async proxy 的
      // 同步机制与 warp specialization 的线程分工：
      //
      // 1. 生产者-消费者解耦：TMA 拷贝由 producer（wg_idx==0 的单个线程）
      //    通过 tma_load_2d + tma_arrive_expect_tx 提交，而非所有 256 个线程
      //    各自做 cp.async。cp.async 需要 __syncthreads 确保所有线程的异步
      //    拷贝都完成后再进入计算；而 TMA 路径用 cuda::barrier 的 full/empty
      //    协议完成 CTA 级同步——full[stage].wait() 返回时，producer 已通过
      //    expect_tx 声明了完整的传输字节数，且硬件保证这些字节已全部写入 smem。
      //
      // 2. async proxy fence 替代 __syncthreads：fence_proxy_async(space_shared)
      //    在 async proxy（TMA 写入通道）和后续 smem load（ldmatrix 读取通道）
      //    之间建立内存序。它确保 fence 之前的所有 async proxy 写操作（TMA）
      //    对 fence 之后的所有 smem 读操作（ldmatrix）可见。这是一个轻量级
      //    proxy 间 ordering，不需要阻塞线程，也不会让 warp 等待其他 warp。
      //
      // 3. 消费者内部 warp 间无依赖：WG1 内的 4 个 warp（2×2 grid）各自独立
      //    读取 smem 中互不重叠的 A/B 区域（warp_m 和 warp_n 分别索引不同的
      //    行区间）。它们之间不需要读写共享数据，因此不需要 __syncthreads 来
      //    对齐 warp 间的执行进度。mma.sync 本身是 warp 级同步指令，保证同一
      //    warp 内 32 个线程的 ldmatrix 和 mma 操作正确协作。
      
      // 注意：这里已经没有NUM_K_TILES循环了，因为K loop load已经被WG0生产者处理了
#pragma unroll
      for (int k_step = 0; k_step < kValTileK; ++k_step) {

#pragma unroll
        for (int i = 0; i < kValTileM; ++i) { // kValTileM = 4
          // kMmaM * kValTileM = 16*4 = 64，每个消费者 warp 在 M 方向覆盖 64 行。
          const int warp_smem_a_m = warp_m * (kMmaM * kValTileM) + i * kMmaM;
          const int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
          // 与 hgemm_mma_stages_tn_swizzle 不同：k_step * kMmaK 必须放在
          // lane_smem_a_k 中传给 swizzle<64>()，而非像 swizzle kernel
          // 那样作为 smem_k_offset 加在 swizzle 外部。原因：
          //   swizzle<64> 作用于完整 BK=64，swizzle 周期覆盖全部 64 列，
          //   k_step=0 和 k_step=1 的 chunk 会被 XOR 交叉混合 → k_step
          //   偏移必须在 swizzle 内部参与 chunk 计算。
          //   swizzle<kMmaK> 作用于 kMmaK=16，每个 kMmaK slice 独立
          //   swizzle，slice 之间互不跨越 → k_step * kMmaK 可以加在外部。
          const int lane_smem_a_k = (k_step * kMmaK) + (lane_id / 16) * 8;
          const uint32_t lane_smem_a_ptr = smem_a_base_ptr +
              (stage * BM * BK + lane_smem_a_m * BK +
              swizzle<64>(lane_smem_a_m, lane_smem_a_k)) *
                  sizeof(half);
          LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
        }

#pragma unroll
        for (int j = 0; j < kValTileN; ++j) { // kValTileN = 8
          // kMmaN * kValTileN = 8*8 = 64，每个消费者 warp 在 B^T 的 N 方向覆盖 64 行。
          const int warp_smem_b_n = warp_n * (kMmaN * kValTileN) + j * kMmaN;
          const int lane_smem_b_n = warp_smem_b_n + lane_id % 8;
          const int lane_smem_b_k = (k_step * kMmaK) + ((lane_id / 8) % 2) * 8;
          const uint32_t lane_smem_b_ptr = smem_b_base_ptr +
              (stage * BN * BK + lane_smem_b_n * BK +
              swizzle<64>(lane_smem_b_n, lane_smem_b_k)) *
                  sizeof(half);
          LDMATRIX_X2(RB[j][0], RB[j][1], lane_smem_b_ptr);
        }

#pragma unroll
        for (int i = 0; i < kValTileM; ++i) {
#pragma unroll
          for (int j = 0; j < kValTileN; ++j) {
            HMMA16816(RC[i][j][0], RC[i][j][1], 
                      RA[i][0], RA[i][1], RA[i][2], RA[i][3], 
                      RB[j][0], RB[j][1], 
                      RC[i][j][0], RC[i][j][1]);
          }
        }
      }
      [[maybe_unused]] auto token = empty[stage].arrive();
    }

    // Epilogue：复用 RA[0] 和 RA[1]（已在寄存器中）作为 shuffle 缓冲，
    // 与 hgemm_mma_stages_tn_swizzle 的 epilogue 模式一致。四个相邻
    // lane 各持有两个 uint32 fragment；shuffle 汇聚为每行一个 float4，
    // 由 lane 0 发出对齐的 128-bit store。
#pragma unroll
    for (int i = 0; i < kValTileM; ++i) {
      const int store_warp_smem_c_m = warp_m * (kMmaM * kValTileM) + i * kMmaM;
      const int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
#pragma unroll
      for (int j = 0; j < kValTileN; ++j) {
        const int store_warp_smem_c_n = warp_n * (kMmaN * kValTileN) + j * kMmaN;
        RA[0][0] = RC[i][j][0];
        RA[1][0] = RC[i][j][1];
        RA[0][1] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 1);
        RA[0][2] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 2);
        RA[0][3] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 3);
        RA[1][1] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 1);
        RA[1][2] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 2);
        RA[1][3] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 3);
        if (lane_id % 4 == 0) {
          const int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n;
          const int store_gmem_c_addr_0 =
              store_lane_gmem_c_m * N + store_lane_gmem_c_n;
          const int store_gmem_c_addr_1 =
              (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
          *reinterpret_cast<float4 *>(&C[store_gmem_c_addr_0]) =
              *reinterpret_cast<float4 *>(&RA[0][0]);
          *reinterpret_cast<float4 *>(&C[store_gmem_c_addr_1]) =
              *reinterpret_cast<float4 *>(&RA[1][0]);
        }
      }
    }
  }
}

#endif /* NOTES_V2_ENABLE_TMA_MMA_WS */
