#pragma once
#include "base.cuh"
// flash_attn.cuh: Phase 8 FlashAttention 2/3 (MMA/TMA_WS/FA3/CuTe)

// Phase 8: FlashAttention-2  + MMA m16n8k16)
// =============================================================================
// 面试要点（FlashAttention-2算法）：
//   1. 核心问题：标准 Attention 的 O(N^2) 中间矩阵 (S=QK^T) 必须写入 HBM，
//      但 HBM 带宽是瓶颈 → FlashAttention-2用 tiling + online softmax 避免写回
//   2. FA 三板斧：
//      a) Tiling: Q 分块 [Br,d]，K/V 沿 seqlen 分块 [Bc,d]
//      b) Online Softmax: 迭代更新 m(行max) 和 l(行sum)，无需全局同步
//      c) Recomputation(反向): 反向传播时重新计算 S/P，而非存储中间矩阵
//   3. Split-Q 设计: 所有 warp 共享同一块 K，各 warp 处理 Q 的不同行片段
//      - warp_KV=0（所有 warp 共享 K），warp_QP=warp_id（各 warp 不同 Q 行）
//      - 优点：减少 warp 间通信和 shuffle
//   4. Online rescaling 公式（FA 核心，arXiv:2307.08691）：
//      for each K,V tile:
//        S_cur = Q @ K^T                         // 未缩放，存入 R_S
//        m_new = max(m_old, row_max(S_cur * scale))
//        P_cur = exp(S_cur * scale - m_new)      // ← 写回 R_S 寄存器！
//        l_new = exp(m_old - m_new) * l_old + row_sum(P_cur)
//        O_new = diag(exp(m_old - m_new)) * O_old + P_cur @ V
//      O_final = O_new / l_final
//   5. 为什么 R_S 可以直接用作 P@V 的 A 矩阵？
//      - R_S 经过 softmax 后，存储的是 P = exp(S - m)，数据仍然是 half 精度
//      - 当前实现依赖 m16n8k16 这一路径下约定好的 fragment 布局，使 softmax
//        后的 P 可以继续留在 R_S 中供后面的 P@V 直接消费
//      - 这是此实现的寄存器布局复用技巧，不要背成“所有 MMA A/C fragment
//        都天然同构”的通用结论
//
// 本实现参考: FlashAttention-2 (Dao et al., arXiv:2307.08691)
// 从 LeetCUDA flash-attn/mma/basic/flash_attn_mma_split_q.cu 提取
// Grid:  ((N + 63) / 64, B * H, 1)，Br=64
// Block: (128, 1, 1)，kNumThreads=kWarpSize×kMmaTileSeqLenQ×kMmaTileSeqLenK=128
// source: LeetCUDA/kernels/flash-attn/mma/basic/flash_attn_mma_split_q.cu

// =============================================================================
// FlashAttention-2 Split-Q Kernel（完整实现）
// =============================================================================
// Q,K,V,O: [batch_size, num_heads, seq_len, head_dim], [B,H,N,d]
//
// Tile 设计（以 kHeadDim=64 为例）:
//   Br = kMmaAtomM * kMmaTileSeqLenQ * kValTileSeqLenQ = 16*{4,8}*1 = {64,128}
//   Bc = kMmaAtomN * kMmaTileSeqLenK * kValTileSeqLenK = 8*1*8  = 64
//   Warp 布局: 4 warps, warp_QP 0~3 各处理 16 行, warp_KV=0 共享 K
//
// 执行流程:
//   1) 预加载 Q[Br,d] 到 smem（只加载一次，split-Q 的核心优势）
//   2) 外循环: 沿 K seqlen 分块迭代 (Tc = seqlen/Bc)
//   3)   3a: cp.async 加载当前 V + 预加载下一个 K（多 stage pipeline）
//   4)   3b: Q@K^T — 沿 head dim 内循环 → ldmatrix Q/K → HMMA16816
//   5)   3c: Online Safe Softmax — warp reduce max/row → exp(S*scale - max)
//         → 关键: 将 P 写回 R_S 寄存器（替换 S），R_S 现在存储 P matrix
//   6)   3d: P@V — 沿 V_Bc 内循环 → ldmatrix V (transposed) → 直接用 R_S 做 A
//         → 当前实现依赖同一组寄存器布局约定，避免为 P@V 额外重组 P fragment
//   7)   3e: Online rescaling — O_new = exp(m_old-m_new)*O_old + P@V
//   8) 最终 rescale: O_final = (1/l_final) * O_final
//   9) Epilogue: warp shuffle + 128-bit collective store
//
// ★ Pad 与手动 XOR swizzle 的 profile 结论（SM120, B=1,H=32,N=4096,D=64）:
//   - XOR 能消除目标 ldmatrix lane pattern 的 bank conflict，但收益是局部的；
//     XOR 本身及 `% 2` 的计算不是主要瓶颈，nvcc 已能将该式化简为 bit operations。
//   - compact Q/K/V XOR 改变了 cp.async 的 shared destination pattern：LDGSTS
//     wavefronts 从 pad 的 30.15M 增至 68.16M（2.26x），long-scoreboard、
//     LG-throttle、MIO-throttle 分别约为 pad 的 3.25x、2.74x、1.60x。
//   - compact XOR 虽节省约 5 KiB smem，但没有提高此 kernel 的 occupancy；最终
//     约 120.0 TFLOPS，显著低于 Q/K/V kPad=8 的 166.6 TFLOPS。
//   - 所以当前 kernel 默认对 Q/K/V 都用 kPad=8。保留 swizzle 路径是为了学习和消融：评价
//     shared layout 必须同时观察 ldmatrix reads 与 cp.async/LDGSTS writes，不能只看
//     bank-conflict counter，也不能只优化 XOR 地址算术。
// ---- 寄存器填充辅助函数（FlashAttention 用，前移以供 TMA_MMA_WS kernel 使用） ----
template <typename T, int M, const int N, const int K = 2>
__device__ inline void fill_3D_regs(T (&R)[M][N][K], T val) {
#pragma unroll
  for (int i = 0; i < M; ++i)
#pragma unroll
    for (int j = 0; j < N; ++j)
#pragma unroll
      for (int k = 0; k < K; ++k)
        R[i][j][k] = val;
}

template <typename T, int M, const int N = 2>
__device__ inline void fill_2D_regs(T (&R)[M][N], T val) {
#pragma unroll
  for (int i = 0; i < M; ++i)
#pragma unroll
    for (int j = 0; j < N; ++j)
      R[i][j] = val;
}

template <
    const int kHeadDim,          // head dim: 32, 64, 128
    const int kMmaAtomM,         // 16 (MMA instruction M dimension)
    const int kMmaAtomN,         // 8  (MMA instruction N dimension)
    const int kMmaAtomK,         // 16 (MMA instruction K dimension)
    const int kMmaAccF32,        // 0=f16 acc, 1=f32 acc (mma.sync.f32.f16.f16.f32)
    const int kMmaTileSeqLenQ,   // MMA tiles along Q's M dim, 4 → Br=16*4=64
    const int kMmaTileSeqLenK,   // MMA tiles along K's N dim, 1 → Bc basis=8
    const int kMmaTileSeqLenP,   // MMA tiles for P@V M dim, must equal kMmaTileSeqLenQ
    const int kMmaTileHeadDimV,  // MMA tiles for P@V N dim (head dim direction)
    const int kValTileSeqLenQ,   // value tiles along Q's M, 1 → Br per warp=16
    const int kValTileSeqLenK,   // value tiles along K's N, 8 → Bc_warp=8*8=64
    const int kValTileSeqLenP,   // value tiles for P@V M dim, 1
    const int kValTileHeadDimV,  // value tiles for P@V N dim, kHeadDim/(8*kMmaTileHeadDimV)
    const int kStagesK,          // pipeline stages for K: >= 1; NO stages required for Q/V
    const int kPadQ,             // Q row padding; 0 selects compact XOR swizzle
    const int kPadK,             // K row padding; 0 selects compact XOR swizzle
    const int kPadV>             // V row padding; 0 selects compact XOR swizzle
__global__ void __launch_bounds__(kWarpSize * kMmaTileSeqLenQ * kMmaTileSeqLenK)
flash_attn_mma_stages_split_q(
  half *Q, half *K, half *V, half *O, int N, int H) {
  static_assert(kStagesK >= 1, "kStagesK must be >= 1");
  static_assert(kMmaAccF32 == 0 || kMmaAccF32 == 1, "kMmaAccF32 must be 0 or 1");
  static_assert(kPadQ >= 0 && kPadK >= 0 && kPadV >= 0,
                "Q/K/V padding must be non-negative");
  constexpr int kNRegs = kMmaAccF32 ? 4 : 2;
  constexpr bool kSwizzleQ = kPadQ == 0;
  constexpr bool kSwizzleK = kPadK == 0;
  constexpr bool kSwizzleV = kPadV == 0;
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kValTileSeqLenQ; // 16*8*1=128
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kValTileSeqLenK; // 8*1*8=64
  constexpr int kNumThreads = kWarpSize * kMmaTileSeqLenQ * kMmaTileSeqLenK; // 32*8*1=256
  const int Tc = (N + Bc - 1) / Bc;
  // 原始实现默认 seqlen 与 Bc 对齐；最后一个不完整 tile 需要额外 pad/边界处理。
  // 这里保留 ceil 写法是为了说明 tile 划分方式，不等于当前实现已经完整处理了尾 tile。
  const float scale = 1.0f / sqrtf((float)kHeadDim);

  // Block indexing
  const int Nb_id = blockIdx.y / H; // batch id
  const int Nh_id = blockIdx.y % H; // head id
  const int Q_tile_id = blockIdx.x; // tile id along Q's M dim (Br)
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
  const int warp_QP = warp_id; // Split-Q: 每个 warp 处理不同的 Q 行片段
  const int warp_KV = 0; // 所有 warp 共享 K（减少跨 warp 通信）

  // Global memory base offsets for this (batch, head)
  // 这里默认 Q/K/V 共享同一 per-head 基址布局，对应 self-attention 场景
  const int Q_gmem_offset = (Nb_id * H * N + Nh_id * N) * kHeadDim;
  const int K_gmem_offset = Q_gmem_offset;
  const int V_gmem_offset = Q_gmem_offset;
  const int O_gmem_offset = Q_gmem_offset;

  // Thread-to-smem mapping for cooperative load
  int load_smem_Q_Br = tid / (kNumThreads / Br);
  int load_smem_Q_d = (tid % (kNumThreads / Br)) * (kHeadDim / (kNumThreads / Br));
  int load_smem_K_Bc = tid / (kNumThreads / Bc);
  int load_smem_K_d = (tid % (kNumThreads / Bc)) * (kHeadDim / (kNumThreads / Bc));
  int load_smem_V_Bc = tid / (kNumThreads / Bc);
  int load_smem_V_d = (tid % (kNumThreads / Bc)) * (kHeadDim / (kNumThreads / Bc));

  int load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br;
  if (load_gmem_Q_Br >= N)
    return;

  // Shared memory layout
  // Q/K/V independently use padded row-major when kPad* > 0 and compact XOR
  // swizzle when kPad* == 0. Padding and XOR are never combined per operand.
  // The swizzled physical layout is [col / 16][row][16]; swizzle<16>() selects
  // the 0/8 phase inside the final 16-half tile.
  extern __shared__ half smem[];
  constexpr int Q_tile_size = Br * (kHeadDim + kPadQ);
  constexpr int K_tile_size = Bc * (kHeadDim + kPadK);
  constexpr int kSmemStrideQ = kHeadDim + kPadQ;
  constexpr int kSmemStrideK = kHeadDim + kPadK;
  constexpr int kSmemStrideV = kHeadDim + kPadV;
  half *Q_tile_smem = smem;
  half *K_tile_smem = Q_tile_smem + Q_tile_size;
  half *V_tile_smem = K_tile_smem + kStagesK * K_tile_size;

  uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
  uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
  uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

  // Online Softmax persistent state
  // lane_block_row_max_old[i][r]: running max for row r of warp tile i
  // lane_block_row_sum_old[i][r]: running denominator l for row r of warp tile i
  float lane_block_row_max_old[kValTileSeqLenQ][2];
  float lane_block_row_sum_old[kValTileSeqLenQ][2];
  fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_block_row_max_old, -INFINITY);
  fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_block_row_sum_old, 0.0f);

  uint32_t R_Q[kValTileSeqLenQ][4];  // Q regs
  uint32_t R_K[kValTileSeqLenK][2];  // K regs
  uint32_t R_V[kValTileHeadDimV][2]; // V regs
  // R_S / R_O / R_D 都按 mma.sync.aligned.m16n8k16 的 fragment 约定存储。
  // f16 acc: 每 tile 2 个 uint32（4 half）；f32 acc: 每 tile 4 个 uint32（4 f32）。
  // 后续 softmax、P@V、online rescale 都直接围绕这组 fragment 布局做寄存器内变换。
  uint32_t R_S[kValTileSeqLenQ][kValTileSeqLenK][kNRegs]; // S=Q@K^T / P=softmax(S)
  uint32_t R_O[kValTileSeqLenP][kValTileHeadDimV][kNRegs]; // O for current tile
  uint32_t R_D[kValTileSeqLenP][kValTileHeadDimV][kNRegs]; // O accumulator (final output)

  fill_3D_regs<uint32_t, kValTileSeqLenQ, kValTileSeqLenK, kNRegs>(R_S, 0);
  fill_3D_regs<uint32_t, kValTileSeqLenP, kValTileHeadDimV, kNRegs>(R_D, 0);

  // ======================================================================
  // Step 1: 加载 Q[Br, d] 到 shared memory（整个外循环只加载一次）
  // ======================================================================
  {
    int load_gmem_Q_addr =
        Q_gmem_offset + load_gmem_Q_Br * kHeadDim + load_smem_Q_d;
    if constexpr (kSwizzleQ) {
#pragma unroll
      for (int i = 0; i < (kHeadDim / (kNumThreads / Br)); i += 8) {
        int col = load_smem_Q_d + i;
        uint32_t ptr = smem_Q_base_ptr +
            ((col / kMmaAtomK) * Br * kMmaAtomK +
             load_smem_Q_Br * kMmaAtomK +
             swizzle<kMmaAtomK>(load_smem_Q_Br, col % kMmaAtomK)) *
                sizeof(half);
        CP_ASYNC_CG(ptr, &Q[load_gmem_Q_addr + i], 16);
      }
    } else {
      uint32_t load_smem_Q_ptr =
          smem_Q_base_ptr +
          (load_smem_Q_Br * kSmemStrideQ + load_smem_Q_d) * sizeof(half);
#pragma unroll
      for (int i = 0; i < (kHeadDim / (kNumThreads / Br)); i += 8) {
        CP_ASYNC_CG(load_smem_Q_ptr + i * 2, &Q[load_gmem_Q_addr + i], 16);
      }
    }
    CP_ASYNC_COMMIT_GROUP();
  }

  // ======================================================================
  // Step 2: 预加载前 (kStagesK-1) 个 K tile（多 stage pipeline 预热）
  // 注意：Q 由 blockIdx.x 固定到当前 Q tile；而 K/V 的 seqlen 遍历始终从 tile 0 开始，
  // 后续在外循环里不断递增到 tile 1/2/3/.../Tc-1。
  // ======================================================================
  if constexpr (kStagesK > 1) {
#pragma unroll
    for (int stage = 0; stage < (kStagesK - 1); ++stage) {
      int load_gmem_K_Bc = stage * Bc + load_smem_K_Bc;
      int load_gmem_K_addr =
          K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_smem_K_d;
      if constexpr (kSwizzleK) {
#pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
          int col = load_smem_K_d + i;
          uint32_t ptr = smem_K_base_ptr +
              (stage * K_tile_size +
               (col / kMmaAtomK) * Bc * kMmaAtomK +
               load_smem_K_Bc * kMmaAtomK +
               swizzle<kMmaAtomK>(load_smem_K_Bc, col % kMmaAtomK)) *
            sizeof(half);
          CP_ASYNC_CG(ptr, &K[load_gmem_K_addr + i], 16);
        }
      } else {
        uint32_t load_smem_K_ptr =
            smem_K_base_ptr +
            (stage * K_tile_size + load_smem_K_Bc * kSmemStrideK +
             load_smem_K_d) *
                sizeof(half);
#pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
          CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
        }
      }
      CP_ASYNC_COMMIT_GROUP();
    }
    CP_ASYNC_WAIT_GROUP(kStagesK - 2);
    __syncthreads();
  }

  // ======================================================================
  // Step 3: 外循环 — 沿 K seqlen 迭代 (Tc = ceil(seqlen/Bc))
  //   每次迭代处理一个 K[Bc,d] + V[Bc,d] tile
  // ======================================================================
#pragma unroll 1
  for (int tile_K_seqlen = 0; tile_K_seqlen < Tc; ++tile_K_seqlen) {
    int smem_sel = tile_K_seqlen % kStagesK;
    int smem_sel_next = (tile_K_seqlen + (kStagesK - 1)) % kStagesK;

    // ---- 3a: 异步加载 K/V tile（多 stage pipeline）----
    // kStagesK>1: V 加载当前 tile，K 预取下一 tile（pipeline）
    // kStagesK=1: V 和 K 都加载当前 tile（smem_sel==smem_sel_next==0），
    //   无 pipeline 预取，每轮 QK 前必须 wait 当前 K 就绪。
    // Load current V tile (no pipeline for V — one stage is enough)
    {
      int load_gmem_V_Bc = tile_K_seqlen * Bc + load_smem_V_Bc;
      int load_gmem_V_addr =
          V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_smem_V_d;
      if constexpr (kSwizzleV) {
#pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
          int col = load_smem_V_d + i;
          uint32_t ptr = smem_V_base_ptr +
              ((col / kMmaAtomK) * Bc * kMmaAtomK +
               load_smem_V_Bc * kMmaAtomK +
               swizzle<kMmaAtomK>(load_smem_V_Bc, col % kMmaAtomK)) *
                  sizeof(half);
          CP_ASYNC_CG(ptr, &V[load_gmem_V_addr + i], 16);
        }
      } else {
        uint32_t load_smem_V_ptr =
            smem_V_base_ptr +
            (load_smem_V_Bc * kSmemStrideV + load_smem_V_d) * sizeof(half);
#pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
          CP_ASYNC_CG(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 16);
        }
      }
      CP_ASYNC_COMMIT_GROUP();
    }

    if constexpr (kStagesK > 1) {
      // Prefetch next K tile (pipelined)
      if ((tile_K_seqlen + 1) < Tc) {
        int load_gmem_K_Bc = (tile_K_seqlen + 1) * Bc + load_smem_K_Bc;
        int load_gmem_K_addr =
            K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_smem_K_d;
        if constexpr (kSwizzleK) {
#pragma unroll
          for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
            int col = load_smem_K_d + i;
            uint32_t ptr = smem_K_base_ptr +
               (smem_sel_next * K_tile_size +
                (col / kMmaAtomK) * Bc * kMmaAtomK +
                load_smem_K_Bc * kMmaAtomK +
                swizzle<kMmaAtomK>(load_smem_K_Bc, col % kMmaAtomK)) *
                  sizeof(half);
            CP_ASYNC_CG(ptr, &K[load_gmem_K_addr + i], 16);
          }
        } else {
          uint32_t load_smem_K_ptr =
              smem_K_base_ptr +
              (smem_sel_next * K_tile_size +
               load_smem_K_Bc * kSmemStrideK +
               load_smem_K_d) *
                  sizeof(half);
#pragma unroll
          for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
            CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
          }
        }
        CP_ASYNC_COMMIT_GROUP();
      }
    } else {
      // kStagesK==1: 加载当前 K tile（smem_sel_next == smem_sel == 0）。
      // Step 2 预加载循环 (kStagesK-1=0) 不执行，所以每轮 3a 必须加载当前 K。
      {
        int load_gmem_K_Bc = tile_K_seqlen * Bc + load_smem_K_Bc;
        int load_gmem_K_addr =
            K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_smem_K_d;
        if constexpr (kSwizzleK) {
#pragma unroll
          for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
            int col = load_smem_K_d + i;
            uint32_t ptr = smem_K_base_ptr +
               (smem_sel * K_tile_size +
                (col / kMmaAtomK) * Bc * kMmaAtomK +
                load_smem_K_Bc * kMmaAtomK +
                swizzle<kMmaAtomK>(load_smem_K_Bc, col % kMmaAtomK)) *
                  sizeof(half);
            CP_ASYNC_CG(ptr, &K[load_gmem_K_addr + i], 16);
          }
        } else {
          uint32_t load_smem_K_ptr =
              smem_K_base_ptr +
              (smem_sel * K_tile_size +
               load_smem_K_Bc * kSmemStrideK +
               load_smem_K_d) *
                  sizeof(half);
#pragma unroll
          for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
            CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
          }
        }
        CP_ASYNC_COMMIT_GROUP();
      }
    }

    // 3b: Q@K^T = S[Br, Bc] — 沿 head dim (d/kMmaAtomK=16) 内循环
    // kStagesK==1: 3a 刚提交当前 K 的 cp.async，必须 wait 后才能 ldmatrix。
    // kStagesK>1: K 已在上一轮预取并 wait（Step 2 或 3e 末尾的 wait），无需再 wait。
    if constexpr (kStagesK == 1) {
      CP_ASYNC_WAIT_GROUP(0);
      __syncthreads();
    }
    fill_3D_regs<uint32_t, kValTileSeqLenQ, kValTileSeqLenK, kNRegs>(R_S, 0);
#pragma unroll
    for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
      // ldmatrix.x4: 加载 Q 的 m16k16 片段到 R_Q
#pragma unroll
      for (int i = 0; i < kValTileSeqLenQ; ++i) {
        int warp_smem_Q_Br =
            warp_QP * (kMmaAtomM * kValTileSeqLenQ) + i * kMmaAtomM;
        int lane_smem_Q_Br =
            warp_smem_Q_Br + lane_id % 16; // ldmatrix uses 16 lanes
        int lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8; // 0, 8
        uint32_t lane_smem_Q_ptr;
        if constexpr (kSwizzleQ) {
          lane_smem_Q_ptr = smem_Q_base_ptr +
              ((lane_smem_Q_d / kMmaAtomK) * Br * kMmaAtomK +
               lane_smem_Q_Br * kMmaAtomK +
               swizzle<kMmaAtomK>(lane_smem_Q_Br,
                                   lane_smem_Q_d % kMmaAtomK)) *
                  sizeof(half);
        } else {
          lane_smem_Q_ptr = smem_Q_base_ptr +
              (lane_smem_Q_Br * kSmemStrideQ + lane_smem_Q_d) * sizeof(half);
        }
        LDMATRIX_X4(R_Q[i][0], R_Q[i][1], R_Q[i][2], R_Q[i][3],
                    lane_smem_Q_ptr);
      }

      // ldmatrix.x2: 加载 K 的 k16n8 片段到 R_K
      // K[Bc,d] row-major = K^T[d,Bc] col-major（NT 布局的 B 矩阵）
#pragma unroll
      for (int j = 0; j < kValTileSeqLenK; ++j) {
        int warp_smem_K_Bc =
            warp_KV * (kMmaAtomN * kValTileSeqLenK) + j * kMmaAtomN;
        int lane_smem_K_Bc =
            warp_smem_K_Bc + lane_id % 8; // ldmatrix B uses 8 lanes
        int lane_smem_K_d =
            tile_K_d * kMmaAtomK + ((lane_id / 8) % 2) * 8; // 0, 8
        uint32_t lane_smem_K_ptr;
        if constexpr (kSwizzleK) {
          lane_smem_K_ptr = smem_K_base_ptr +
              (smem_sel * K_tile_size +
               (lane_smem_K_d / kMmaAtomK) * Bc * kMmaAtomK +
               lane_smem_K_Bc * kMmaAtomK +
               swizzle<kMmaAtomK>(lane_smem_K_Bc,
                                   lane_smem_K_d % kMmaAtomK)) *
                  sizeof(half);
        } else {
          lane_smem_K_ptr = smem_K_base_ptr +
            (smem_sel * K_tile_size + lane_smem_K_Bc * kSmemStrideK +
               lane_smem_K_d) *
                  sizeof(half);
        }
        LDMATRIX_X2(R_K[j][0], R_K[j][1], lane_smem_K_ptr);
      }

      // MMA: S[tile] += Q[tile] @ K^T[tile]
#pragma unroll
      for (int i = 0; i < kValTileSeqLenQ; ++i) {
#pragma unroll
        for (int j = 0; j < kValTileSeqLenK; ++j) {
          if constexpr (kMmaAccF32) {
            HMMA16816F32(R_S[i][j][0], R_S[i][j][1], R_S[i][j][2], R_S[i][j][3],
                         R_Q[i][0], R_Q[i][1], R_Q[i][2], R_Q[i][3],
                         R_K[j][0], R_K[j][1],
                         R_S[i][j][0], R_S[i][j][1], R_S[i][j][2], R_S[i][j][3]);
          } else {
            HMMA16816(R_S[i][j][0], R_S[i][j][1], R_Q[i][0], R_Q[i][1], R_Q[i][2],
                      R_Q[i][3], R_K[j][0], R_K[j][1], R_S[i][j][0],
                      R_S[i][j][1]);
          }
        }
      }
    } // end loop over d
    __syncthreads();

    // ======================================================================
    // 3c: Online Safe Softmax — row-wise max + exp + sum, then store P back to R_S
    // ======================================================================
    // MMA C fragment layout for m16n8k16 (PTX ISA 对应的线程寄存器分布):
    //   - R_S[i][j][0] 对应当前 16x8 tile 的 rows 0~7 里的两个 half 值 {c0,c1}
    //   - R_S[i][j][1] 对应 rows 8~15 里的两个 half 值 {c2,c3}
    //   - lane 0~3 持有 row 0 的片段，lane 4~7 持有 row 1，...，lane 28~31 持有 row 7
    //   - 对于 rows 8~15 也是同样的 lane 分组，只是读取的是 reg[1]
    // 这就是为什么后面做 row max / row sum 时，warp 内真正参与同一行归约的是
    // {0,4,8,...,28} 这一类 4-lane 子组，而不是整 warp 32 个线程。
    // Each (i, j) pair = one 16x8 MMA tile; there are kValTileSeqLenQ x
    // kValTileSeqLenK tiles.

    float lane_row_max_new[kValTileSeqLenQ][2];
    float lane_row_sum_new[kValTileSeqLenQ][2];
    fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_row_max_new, -INFINITY);
    fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_row_sum_new, 0.0f);

    // Pass 1: Thread-level reduce max across all columns (kValTileSeqLenK tiles)
#pragma unroll
    for (int i = 0; i < kValTileSeqLenQ; ++i) {
#pragma unroll
      for (int j = 0; j < kValTileSeqLenK; ++j) {
        float tmp_max_0, tmp_max_1;
        if constexpr (kMmaAccF32) {
          float *t_fptr_S = reinterpret_cast<float *>(&R_S[i][j][0]);
          tmp_max_0 = max(t_fptr_S[0], t_fptr_S[1]) * scale;
          tmp_max_1 = max(t_fptr_S[2], t_fptr_S[3]) * scale;
        } else {
          float2 t_reg_S_0 =
              __half22float2(HALF2(R_S[i][j][0])); // rows 0~7:  {c0, c1}
          float2 t_reg_S_1 =
              __half22float2(HALF2(R_S[i][j][1])); // rows 8~15: {c2, c3}
          tmp_max_0 = max(t_reg_S_0.x, t_reg_S_0.y) * scale;
          tmp_max_1 = max(t_reg_S_1.x, t_reg_S_1.y) * scale;
        }
        lane_row_max_new[i][0] = max(lane_row_max_new[i][0], tmp_max_0);
        lane_row_max_new[i][1] = max(lane_row_max_new[i][1], tmp_max_1);
      }
      // Warp-level reduce max (kWarpWidth = 4 for Q@K^T — only lanes
      // {0,4,8,...,28} hold valid data)
      lane_row_max_new[i][0] =
          warp_reduce_max<4, float>(lane_row_max_new[i][0]);
      lane_row_max_new[i][1] =
          warp_reduce_max<4, float>(lane_row_max_new[i][1]);
    }

    // Pass 2: Compute P = exp(S*scale - m_new), store back to R_S
    // 面试关键点：这里将 P 写回 R_S 寄存器！
    // 为什么可以？当前实现依赖 m16n8k16 这一路径下约定好的 fragment 布局，
    // 使 softmax 后的 P 能继续留在 R_S 中供后面的 P@V 直接消费，无需额外重组。
#pragma unroll
    for (int i = 0; i < kValTileSeqLenQ; ++i) {
      // m_new = max(m_old, m_cur)
      float block_row_max_new_0 =
          max(lane_block_row_max_old[i][0], lane_row_max_new[i][0]);
      float block_row_max_new_1 =
          max(lane_block_row_max_old[i][1], lane_row_max_new[i][1]);

#pragma unroll
      for (int j = 0; j < kValTileSeqLenK; ++j) {
        if constexpr (kMmaAccF32) {
          float *t_fptr_S = reinterpret_cast<float *>(&R_S[i][j][0]);
          half *t_hptr_S = reinterpret_cast<half *>(&R_S[i][j][0]);
          t_fptr_S[0] = __expf(__fmaf_rn(t_fptr_S[0], scale, -block_row_max_new_0));
          t_fptr_S[1] = __expf(__fmaf_rn(t_fptr_S[1], scale, -block_row_max_new_0));
          t_fptr_S[2] = __expf(__fmaf_rn(t_fptr_S[2], scale, -block_row_max_new_1));
          t_fptr_S[3] = __expf(__fmaf_rn(t_fptr_S[3], scale, -block_row_max_new_1));
          lane_row_sum_new[i][0] += (t_fptr_S[0] + t_fptr_S[1]);
          lane_row_sum_new[i][1] += (t_fptr_S[2] + t_fptr_S[3]);
          t_hptr_S[0] = __float2half_rn(t_fptr_S[0]);
          t_hptr_S[1] = __float2half_rn(t_fptr_S[1]);
          t_hptr_S[2] = __float2half_rn(t_fptr_S[2]);
          t_hptr_S[3] = __float2half_rn(t_fptr_S[3]);
        } else {
          float2 t_reg_S_0 = __half22float2(HALF2(R_S[i][j][0]));
          float2 t_reg_S_1 = __half22float2(HALF2(R_S[i][j][1]));
          t_reg_S_0.x =
              __expf(__fmaf_rn(t_reg_S_0.x, scale, -block_row_max_new_0));
          t_reg_S_0.y =
              __expf(__fmaf_rn(t_reg_S_0.y, scale, -block_row_max_new_0));
          t_reg_S_1.x =
              __expf(__fmaf_rn(t_reg_S_1.x, scale, -block_row_max_new_1));
          t_reg_S_1.y =
              __expf(__fmaf_rn(t_reg_S_1.y, scale, -block_row_max_new_1));
          lane_row_sum_new[i][0] += (t_reg_S_0.x + t_reg_S_0.y);
          lane_row_sum_new[i][1] += (t_reg_S_1.x + t_reg_S_1.y);
          HALF2(R_S[i][j][0]) = __float22half2_rn(t_reg_S_0);
          HALF2(R_S[i][j][1]) = __float22half2_rn(t_reg_S_1);
        }
      }

      // Warp-level reduce sum (kWarpWidth = 4, same as max)
      lane_row_sum_new[i][0] =
          warp_reduce_sum<4, float>(lane_row_sum_new[i][0]);
      lane_row_sum_new[i][1] =
          warp_reduce_sum<4, float>(lane_row_sum_new[i][1]);
    }
    __syncthreads();

    // ======================================================================
    // 3d: P@V — P[Br,Bc] @ V[Bc,d] = O[Br,d]
    // ======================================================================
    // Wait for V to be ready before computing P@V
    if constexpr (kStagesK > 1) {
      if ((tile_K_seqlen + 1) < Tc) {
        CP_ASYNC_WAIT_GROUP(1);
      } else {
        CP_ASYNC_WAIT_GROUP(0);
      }
    } else {
      CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();

    fill_3D_regs<uint32_t, kValTileSeqLenP, kValTileHeadDimV, kNRegs>(R_O, 0);

    // tile_V_Bc: iterate over chunks of Bc/K=16 columns in P matrix
    // Bc=kMmaAtomK=16 → 1 iteration for kHeadDim≤64 configurations
#pragma unroll
    for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK); ++tile_V_Bc) {
      // ldmatrix.x2.trans: load V[Bc,d] with transposition
      // V is row-major [Bc,d], but NN matmul needs B matrix in col-major → use
      // transposed ldmatrix
#pragma unroll
      for (int j = 0; j < kValTileHeadDimV; ++j) {
        int warp_smem_V_d =
            warp_KV * (kMmaAtomN * kValTileHeadDimV) + j * kMmaAtomN;
        int lane_smem_V_Bc = tile_V_Bc * kMmaAtomK + lane_id % 16;
        int lane_smem_V_d = warp_smem_V_d;
        uint32_t lane_smem_V_ptr;
        if constexpr (kSwizzleV) {
          lane_smem_V_ptr = smem_V_base_ptr +
              ((lane_smem_V_d / kMmaAtomK) * Bc * kMmaAtomK +
               lane_smem_V_Bc * kMmaAtomK +
               swizzle<kMmaAtomK>(lane_smem_V_Bc,
                                   lane_smem_V_d % kMmaAtomK)) *
                  sizeof(half);
        } else {
          lane_smem_V_ptr = smem_V_base_ptr +
              (lane_smem_V_Bc * kSmemStrideV + lane_smem_V_d) * sizeof(half);
        }
        LDMATRIX_X2_T(R_V[j][0], R_V[j][1], lane_smem_V_ptr);
      }

      // P matrix layout in R_S[i][j][2]:
      //   MMA = m16n8k16, Br=16x4=64, Bc=8x8=64, layout: 4 warps
      //   |   64x64   |      warp_KV 0       |
      //   | warp_QP 0 | MMA 0 ... MMA 0 (x8) |
      //   | warp_QP 1 | MMA 1 ... MMA 1 (x8) |
      //   | warp_QP 2 | MMA 2 ... MMA 2 (x8) |
      //   | warp_QP 3 | MMA 3 ... MMA 3 (x8) |
      // tile_V_Bc selects which 16-column slice of P to use:
      //   tile_V_Bc=0 → cols  0:16 → MMA indices 0,1  → w=0
      //   tile_V_Bc=1 → cols 16:32 → MMA indices 2,3  → w=2
      //   tile_V_Bc=2 → cols 32:48 → MMA indices 4,5  → w=4
      //   tile_V_Bc=3 → cols 48:64 → MMA indices 6,7  → w=6
      // 对应的 MMA A fragment 布局可以这样记：
      //   - rows 0~7:  lane 0~3 -> {a0,a1} 与 {a4,a5}，lane 4~7 -> 下一行，依次类推
      //   - rows 8~15: lane 0~3 -> {a2,a3} 与 {a6,a7}，lane 4~7 -> 下一行，依次类推
      // 当前实现正是利用这一路径下 A fragment 与前面生成的 P fragment 可以直接对接，
      // 才能把 R_S 中的 P 直接喂给 HMMA16816 做 P@V；复习时不要把它背成对所有 MMA
      // fragment 都无条件成立的通用结论。layout转换逻辑：
      //   C layout: 8 x (16 x 8) layout -> A layout: 4 x (16 x 16) layout
      int w = tile_V_Bc * 2;
#pragma unroll
      for (int i = 0; i < kValTileSeqLenP; ++i) {
#pragma unroll
        for (int j = 0; j < kValTileHeadDimV; ++j) {
          if constexpr (kMmaAccF32) {
            HMMA16816F32(R_O[i][j][0], R_O[i][j][1], R_O[i][j][2], R_O[i][j][3],
                         R_S[i][w][0], R_S[i][w][1], R_S[i][w + 1][0], R_S[i][w + 1][1],
                         R_V[j][0], R_V[j][1],
                         R_O[i][j][0], R_O[i][j][1], R_O[i][j][2], R_O[i][j][3]);
          } else {
            HMMA16816(R_O[i][j][0], R_O[i][j][1],
                      R_S[i][w][0], R_S[i][w][1], R_S[i][w + 1][0], R_S[i][w + 1][1],
                      R_V[j][0], R_V[j][1],
                      R_O[i][j][0], R_O[i][j][1]);
          }
        }
      }
    } // end for tile_V_Bc
    __syncthreads();

    // ======================================================================
    // 3e: Online rescaling — O_new = exp(m_old - m_new) * O_old + P@V
    // ======================================================================
    // 公式来源: FA2 paper Eq.(7-8)，使用 exp(m_old - m_new) 做 O 与 l 的 rescale
#pragma unroll
    for (int i = 0; i < kValTileSeqLenP; ++i) {
      float block_row_max_new_0 = lane_row_max_new[i][0];
      float block_row_max_new_1 = lane_row_max_new[i][1];
      float block_row_sum_new_0 = lane_row_sum_new[i][0];
      float block_row_sum_new_1 = lane_row_sum_new[i][1];

      float block_row_max_old_0 = lane_block_row_max_old[i][0];
      float block_row_max_old_1 = lane_block_row_max_old[i][1];

      block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);

      // Handle first iteration: m_old = -inf, need to use m_new directly
      block_row_max_old_0 =
          (tile_K_seqlen > 0 ? block_row_max_old_0 : block_row_max_new_0);
      block_row_max_old_1 =
          (tile_K_seqlen > 0 ? block_row_max_old_1 : block_row_max_new_1);

      float rescale_o_factor_0 =
          __expf(block_row_max_old_0 - block_row_max_new_0);
      float rescale_o_factor_1 =
          __expf(block_row_max_old_1 - block_row_max_new_1);

      // Rescale O_old + Add P@V in one fused step
#pragma unroll
      for (int j = 0; j < kValTileHeadDimV; ++j) {
        if constexpr (kMmaAccF32) {
          float *t_fptr_O = reinterpret_cast<float *>(&R_O[i][j][0]);
          float *t_fptr_D = reinterpret_cast<float *>(&R_D[i][j][0]);
          t_fptr_D[0] = __fmaf_rn(rescale_o_factor_0, t_fptr_D[0], t_fptr_O[0]);
          t_fptr_D[1] = __fmaf_rn(rescale_o_factor_0, t_fptr_D[1], t_fptr_O[1]);
          t_fptr_D[2] = __fmaf_rn(rescale_o_factor_1, t_fptr_D[2], t_fptr_O[2]);
          t_fptr_D[3] = __fmaf_rn(rescale_o_factor_1, t_fptr_D[3], t_fptr_O[3]);
        } else {
          float2 t_reg_O_0 = __half22float2(HALF2(R_O[i][j][0]));
          float2 t_reg_O_1 = __half22float2(HALF2(R_O[i][j][1]));
          float2 t_reg_D_0 = __half22float2(HALF2(R_D[i][j][0]));
          float2 t_reg_D_1 = __half22float2(HALF2(R_D[i][j][1]));
          t_reg_D_0.x = __fmaf_rn(rescale_o_factor_0, t_reg_D_0.x, t_reg_O_0.x);
          t_reg_D_0.y = __fmaf_rn(rescale_o_factor_0, t_reg_D_0.y, t_reg_O_0.y);
          t_reg_D_1.x = __fmaf_rn(rescale_o_factor_1, t_reg_D_1.x, t_reg_O_1.x);
          t_reg_D_1.y = __fmaf_rn(rescale_o_factor_1, t_reg_D_1.y, t_reg_O_1.y);
          HALF2(R_D[i][j][0]) = __float22half2_rn(t_reg_D_0);
          HALF2(R_D[i][j][1]) = __float22half2_rn(t_reg_D_1);
        }
      }

      // Update l: l_new = exp(m_old - m_new) * l_old + row_sum(P)
      float block_row_sum_old_0 = lane_block_row_sum_old[i][0];
      float block_row_sum_old_1 = lane_block_row_sum_old[i][1];
      lane_block_row_sum_old[i][0] = __fmaf_rn(
          rescale_o_factor_0, block_row_sum_old_0, block_row_sum_new_0);
      lane_block_row_sum_old[i][1] = __fmaf_rn(
          rescale_o_factor_1, block_row_sum_old_1, block_row_sum_new_1);

      // Update m
      lane_block_row_max_old[i][0] = block_row_max_new_0;
      lane_block_row_max_old[i][1] = block_row_max_new_1;
    }

    // Wait for next K tile to be ready in smem before next iteration
    if constexpr (kStagesK > 1) {
      if ((tile_K_seqlen + 1) < Tc) {
        CP_ASYNC_WAIT_GROUP(0);
      }
      __syncthreads();
    }
  } // end outer loop over K seqlen
  __syncthreads();

  // ======================================================================
  // Step 4: 最终 rescale — O_final = (1/l_final) * O_final
  // ======================================================================
#pragma unroll
  for (int i = 0; i < kValTileSeqLenP; ++i) {
    float rescale_factor_0 = __frcp_rn(lane_block_row_sum_old[i][0]);
    float rescale_factor_1 = __frcp_rn(lane_block_row_sum_old[i][1]);
#pragma unroll
    for (int j = 0; j < kValTileHeadDimV; ++j) {
      if constexpr (kMmaAccF32) {
        float *t_fptr_D = reinterpret_cast<float *>(&R_D[i][j][0]);
        half *t_hptr_D = reinterpret_cast<half *>(&R_D[i][j][0]);
        t_hptr_D[0] = __float2half_rn(rescale_factor_0 * t_fptr_D[0]);
        t_hptr_D[1] = __float2half_rn(rescale_factor_0 * t_fptr_D[1]);
        t_hptr_D[2] = __float2half_rn(rescale_factor_1 * t_fptr_D[2]);
        t_hptr_D[3] = __float2half_rn(rescale_factor_1 * t_fptr_D[3]);
      } else {
        float2 t_reg_D_0 = __half22float2(HALF2(R_D[i][j][0]));
        float2 t_reg_D_1 = __half22float2(HALF2(R_D[i][j][1]));
        t_reg_D_0.x = rescale_factor_0 * t_reg_D_0.x;
        t_reg_D_0.y = rescale_factor_0 * t_reg_D_0.y;
        t_reg_D_1.x = rescale_factor_1 * t_reg_D_1.x;
        t_reg_D_1.y = rescale_factor_1 * t_reg_D_1.y;
        HALF2(R_D[i][j][0]) = __float22half2_rn(t_reg_D_0);
        HALF2(R_D[i][j][1]) = __float22half2_rn(t_reg_D_1);
      }
    }
  }

  // ======================================================================
  // Step 5: Epilogue — Collective store via warp shuffle + 128-bit store
  // ======================================================================
  // 利用 warp shuffle 将分散在各 lane 的寄存器数据收集到 lane 0~3，
  // 然后用 LDST128BITS (st.global.v4.f32) 一次性写入 16 bytes
#pragma unroll
  for (int i = 0; i < kValTileSeqLenP; ++i) {
#pragma unroll
    for (int j = 0; j < kValTileHeadDimV; ++j) {
      uint32_t R_Z[2][4];
      R_Z[0][0] = R_D[i][j][0];
      R_Z[1][0] = R_D[i][j][1];
      R_Z[0][1] = __shfl_sync(0xffffffff, R_D[i][j][0], lane_id + 1, 4);
      R_Z[0][2] = __shfl_sync(0xffffffff, R_D[i][j][0], lane_id + 2, 4);
      R_Z[0][3] = __shfl_sync(0xffffffff, R_D[i][j][0], lane_id + 3, 4);
      R_Z[1][1] = __shfl_sync(0xffffffff, R_D[i][j][1], lane_id + 1, 4);
      R_Z[1][2] = __shfl_sync(0xffffffff, R_D[i][j][1], lane_id + 2, 4);
      R_Z[1][3] = __shfl_sync(0xffffffff, R_D[i][j][1], lane_id + 3, 4);

      // st.global.v4.f32: 128-bit store, 4 lanes × 32-bit
      if (lane_id % 4 == 0) {
        int store_warp_regs_O_Br =
            warp_QP * (kMmaAtomM * kValTileSeqLenP) + i * kMmaAtomM;
        int store_lane_gmem_O_Br =
            Q_tile_id * Br + store_warp_regs_O_Br + lane_id / 4;
        int store_warp_regs_O_d =
            warp_KV * (kMmaAtomN * kValTileHeadDimV) + j * kMmaAtomN;
        int store_lane_gmem_O_d = store_warp_regs_O_d;
        int store_gmem_O_addr_0 = O_gmem_offset +
                                  (store_lane_gmem_O_Br + 0) * kHeadDim +
                                  store_lane_gmem_O_d;
        int store_gmem_O_addr_1 = O_gmem_offset +
                                  (store_lane_gmem_O_Br + 8) * kHeadDim +
                                  store_lane_gmem_O_d;
        // LDST128BITS = reinterpret_cast<float4*>
        *reinterpret_cast<float4 *>(&O[store_gmem_O_addr_0]) =
            *reinterpret_cast<float4 *>(&R_Z[0][0]);
        *reinterpret_cast<float4 *>(&O[store_gmem_O_addr_1]) =
            *reinterpret_cast<float4 *>(&R_Z[1][0]);
      }
    }
  }
}

#if defined(NOTES_V2_ENABLE_TMA_MMA_WS)
// Phase 8b: FlashAttention-2 TMA + MMA Warp Specialization (SM120)
// =============================================================================
// 面试要点（FA TMA MMA WS vs cp.async FA）：
//   - Producer (WG0, 128T) 用 TMA 128B SWIZZLE 搬运 Q/K/V tile 到 smem
//   - Consumer (WG1, 256T) 用 ldmatrix + mma.sync.m16n8k16 做 Q@K^T / P@V
//   - 与 hgemm_tma_mma_ws_tn 的对比：
//     * HGEMM 只做一次 GEMM，K 维迭代；FA 做 QK^T 和 PV 两次 GEMM，KV seqlen 迭代
//     * HGEMM 的 A/B 都 staged；FA 中 Q 只 load 一次（split-Q），K staged，V 单 buffer
//
// ★ split-Q 语义（关键，避免误解）：
//   - grid = (seqlen/Br, B*H)，每个 block 处理一个 **block-level Q tile [Br, d]**
//     （不是全局 Q）。对 Q 的 seqlen 做 block-level 并行。
//   - 每个 block 遍历**完整的 KV seqlen**（Tc = seqlen/Bc 次外循环）。
//   - 因此每个 block 的 Q tile 只需 **load 一次到 smem**，在整个 KV 遍历中复用
//     → 只需 full_Q barrier（无 empty_Q）。
//
// ★ Pipeline 同步协议（arrive_count = 257 = 256 consumer + 1 producer）：
//   - Q:  full_Q — Producer 发 arrive_tx，Consumer 在主循环前 wait 一次
//   - K:  full_K[kStagesK] / empty_K[kStagesK] — 多 stage pipeline
//   - V:  full_V / empty_V — 单 buffer 串行（QK 前发 TMA → QK 重叠 → PV 前等 → PV 后释放）
//
// ★ V ldmatrix.x2.trans 正确性（已论证无风险）：
//   TMA 128B SWIZZLE 是 1-1 映射，ldmatrix 用 swizzle<64>(row,col) 计算的物理地址
//   = TMA 写入的物理地址（smem 1024B 对齐保证 phase=0）。ldmatrix.x2.trans 的转置
//   语义在寄存器层面工作，与 smem 物理布局无关 → 必然正确。
//
// ★ Shared Memory 占用分析（D=64/128, kStagesK=1/2/3/4, kStagesV=1/2）:
//   smem = (Q[Br,D] + K[kStagesK,Bc,D] + V[kStagesV,Bc,D]) * sizeof(half)
//        = D * (Br + kStagesK*Bc + kStagesV*Bc) * 2
//        = D * (128 + (kStagesK+kStagesV)*64) * 2    （Br=128, Bc=64）
//
//   | kStagesK | kStagesV | D=64  bytes | D=128 bytes |
//   |---------------------|----------|-------------|-------------|
//   |    1     |    1     |   32 KB     |   64 KB     |
//   |    2     |    1     |   40 KB     |   80 KB     |
//   |    2     |    2     |   48 KB     |   96 KB     |
//   |    3     |    1     |   48 KB     |   96 KB     |
//   |    3     |    2     |   56 KB     |  112 KB     |
//   |    4     |    1     |   56 KB     |  112 KB     |
//
//   SM120 (RTX PRO 5000) optin smem 上限 ~100KB：
//   - D=64: S=1/2/3/4 全部可行 (32-56 KB)
//   - D=128: S=1/2/3 可行 (64-96 KB)；S=4 (112 KB) 超出 optin 上限，会被
//     check_smem_feasible 兜底判为 SMEM SKIP。D=128 推荐 S=2（甜点：80 KB，
//     2 blocks/SM，pipeline 深度足够隐藏 TMA 延迟）。
//
// v1 限制：kHeadDim=64 or 128；seqlen % Br == 0 且 seqlen % Bc == 0（不处理尾 tile）

// =============================================================================
// FA TMA WS smem offset helper (D=64/128 通用)
// =============================================================================
// 背景：CU_TENSOR_MAP_SWIZZLE_128B 硬件要求 box innermost dim ≤ 128B = 64 half
// （见 /usr/local/cuda/include/cuda.h 的 cuTensorMapEncodeTiled 文档）。因此 D=128
// 不能用单个 TMA box=(128, Br) 覆盖整行。
//
// 方案：D=128 时 box 固定为 (64, Br)，沿 head_dim 方向连续发 kTmaChunks 次 TMA
// （minor_coord = c*64），写入 chunk-major smem 布局 [kTmaChunks, Br, 64]：
//   - chunk c 的 smem 起始偏移 = c * Br * 64
//   - chunk c 内部按 [Br, 64] row-major + SWIZZLE_128B 存放（64 half = 128B 周期）
//
// 本 helper 计算 (row, col) 在该 chunk-major 布局下的 swizzled smem 元素偏移：
//   chunk   = col / 64        ∈ [0, kTmaChunks)
//   col_in  = col % 64        ∈ [0, 64)
//   offset  = chunk * Br * 64 + row * 64 + swizzle<64>(row, col_in)
//
// D=64 退化：kTmaChunks=1, chunk=0, offset = row*64 + swizzle<64>(row, col)
//           与原公式 (row*64 + swizzle<64>(row, col)) 完全一致 → 无回归。
template <int kHeadDim, int Br>
__device__ __forceinline__ int swizzle_fa(int row, int col) {
  static_assert(kHeadDim == 64 || kHeadDim == 128, "D=64 or 128 only");
  if constexpr (kHeadDim == 64) {
    return row * 64 + swizzle<64>(row, col);
  } else {  // kHeadDim == 128
    const int chunk  = col >> 6;        // col / 64 ∈ {0, 1}
    const int col_in = col & 63;        // col % 64
    return chunk * (Br * 64) + row * 64 + swizzle<64>(row, col_in);
  }
}

template <
    const int kHeadDim,          // head dim: v1 only 64
    const int kMmaAtomM,         // 16 (MMA instruction M dim)
    const int kMmaAtomN,         // 8  (MMA instruction N dim)
    const int kMmaAtomK,         // 16 (MMA instruction K dim)
    const int kMmaAccF32,        // 0=f16 acc, 1=f32 acc (mma.sync.f32.f16.f16.f32)
    const int kMmaTileSeqLenQ,   // MMA tiles along Q's M dim, 8 → Br=16*8=128
    const int kMmaTileSeqLenK,   // MMA tiles along K's N dim, 1 → Bc basis=8
    const int kMmaTileSeqLenP,   // MMA tiles for P@V M dim, == kMmaTileSeqLenQ
    const int kMmaTileHeadDimV,  // MMA tiles for P@V N dim (head dim direction)
    const int kValTileSeqLenQ,   // value tiles along Q's M, 1 → Br per warp=16
    const int kValTileSeqLenK,   // value tiles along K's N, 8 → Bc_warp=8*8=64
    const int kValTileSeqLenP,   // value tiles for P@V M dim, 1
    const int kValTileHeadDimV,  // value tiles for P@V N dim, kHeadDim/(8*kMmaTileHeadDimV)
    const int kStagesK,          // K pipeline depth (>=1)
    const int kStagesV,          // V pipeline depth (>=1; 1=single buffer, >=2=pipelined)
    const int kNumThreads>       // 384, 128 producer + 256 consumer
__global__ void __launch_bounds__(kNumThreads, 1)
    flash_attn_tma_mma_ws_stages_split_q(
        half *Q, half *K, half *V, half *O, int N, int H,
        const CUtensorMap *__restrict__ tensorMapQ,
        const CUtensorMap *__restrict__ tensorMapK,
        const CUtensorMap *__restrict__ tensorMapV) {
  static_assert(kHeadDim == 64 || kHeadDim == 128,
                "v1 supports kHeadDim == 64 or 128");
  static_assert(kNumThreads == 384, "128 producer + 256 consumer");
  static_assert(kMmaAccF32 == 0 || kMmaAccF32 == 1, "kMmaAccF32 must be 0 or 1");
  static_assert(kStagesK >= 1, "kStagesK must be >= 1");
  static_assert(kStagesV >= 1, "kStagesV must be >= 1");
  constexpr int kNRegs = kMmaAccF32 ? 4 : 2;

  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kValTileSeqLenQ; // 128
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kValTileSeqLenK; // 64
  // BK for QK: kHeadDim=64, split into kHeadDim/kMmaAtomK=4 MMA K slices
  // BK for PV: Bc=64, split into Bc/kMmaAtomK=4 MMA K slices (P rows)
  constexpr int kConsumerThreads = 256;  // 8 warps
  constexpr int kProducerThreads = 128;  // 4 warps, only thread 0 issues TMA
  constexpr int kQTileBytes = Br * kHeadDim * sizeof(half);       // 16 KB
  constexpr int kKTileBytes = Bc * kHeadDim * sizeof(half);       // 8 KB
  constexpr int kVTileBytes = Bc * kHeadDim * sizeof(half);       // 8 KB
  // TMA box innermost 固定 64 half (128B)，满足 CU_TENSOR_MAP_SWIZZLE_128B 的
  // box innermost ≤ 128B 硬约束。D=128 时沿 head_dim 发 kTmaChunks 次 TMA，
  // 写入 chunk-major smem 布局 [kTmaChunks, Br, 64]。
  constexpr int kTmaBoxMinor = 64;
  constexpr int kTmaChunks   = kHeadDim / kTmaBoxMinor;

  // Block indexing — split-Q: blockIdx.x indexes Q tile along seqlen
  const int Nb_id = blockIdx.y / H;
  const int Nh_id = blockIdx.y % H;
  const int Q_tile_id = blockIdx.x;
  const int Tc = (N + Bc - 1) / Bc;  // number of KV tiles along seqlen
  const float scale = 1.0f / sqrtf((float)kHeadDim);

  // Per-head gmem base offset (self-attention: Q=K=V share layout)
  const int QKV_gmem_offset = (Nb_id * H * N + Nh_id * N) * kHeadDim;
  const int O_gmem_offset = QKV_gmem_offset;
  // TMA descriptor covers the entire [B*H*N, D] gmem as one 2D matrix; the
  // producer must offset major_coord by this head/batch base so each block
  // loads its own (batch, head) Q/K/V tile instead of always reading head 0.
  const int qkv_major_base = (Nb_id * H + Nh_id) * N;

  // ---- Shared memory layout ----
  // TMA CU_TENSOR_MAP_SWIZZLE_128B requires 1024B-aligned smem base so the
  // hardware swizzle phase starts at zero. Consumer swizzle<64>() assumes
  // zero phase (no base_offset compensation like WGMMA descriptor).
  // Q: [Br, kHeadDim] = [128, 64] = 16 KB (1024B aligned ✓)
  // K: [kStagesK, Bc, kHeadDim] = [kStagesK, 64, 64] (8 KB per stage)
  // V: [kStagesV, Bc, kHeadDim] = [kStagesV, 64, 64] (8 KB per stage)
  extern __shared__ __align__(1024) uint8_t smem_fa_tma_ws[];
  half *Q_smem = reinterpret_cast<half *>(smem_fa_tma_ws);
  half *K_smem = Q_smem + Br * kHeadDim;
  half *V_smem = K_smem + kStagesK * Bc * kHeadDim;

  const uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_smem);
  const uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_smem);
  const uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_smem);

  // ---- Barriers ----
  // arrive_count = kConsumerThreads + 1 = 257 (256 consumer arrives + 1 producer arrive_tx)
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ cuda::barrier<cuda::thread_scope_block> full_Q;
  __shared__ cuda::barrier<cuda::thread_scope_block> full_K[kStagesK];
  __shared__ cuda::barrier<cuda::thread_scope_block> empty_K[kStagesK];
  __shared__ cuda::barrier<cuda::thread_scope_block> full_V[kStagesV];
  __shared__ cuda::barrier<cuda::thread_scope_block> empty_V[kStagesV];
#pragma nv_diag_default static_var_with_dynamic_init

  // Thread partition: first kProducerThreads (0~127) = Producer WG,
  // remaining kConsumerThreads (128~383) = Consumer WG.
  // WARN: Must use kProducerThreads (not kConsumerThreads) as the divisor, otherwise
  // the split is wrong: 384/256=1.5 would give Producer 256 threads and
  // Consumer only 128, but barriers expect 256 consumer arrives → deadlock.
  const int wg_idx = (threadIdx.x < kProducerThreads) ? 0 : 1;
  const int wg_tid = (threadIdx.x < kProducerThreads) ? threadIdx.x : 
                     (threadIdx.x - kProducerThreads);

  if (threadIdx.x == 0) {
    init(&full_Q, kConsumerThreads + 1);
    for (int s = 0; s < kStagesV; ++s) {
      init(&full_V[s], kConsumerThreads + 1);
      init(&empty_V[s], kConsumerThreads + 1);
    }
    for (int s = 0; s < kStagesK; ++s) {
      init(&full_K[s], kConsumerThreads + 1);
      init(&empty_K[s], kConsumerThreads + 1);
    }
    tma_fence_proxy_async_shared_cta();
  }
  __syncthreads();

  // ==================================================================
  // Producer Warpgroup (WG0, threadIdx.x 0~127)
  // Only wg_tid == 0 issues TMA. Split-Q: Q tile loaded once, K staged, V single buffer.
  // D=128 时每个 tile 沿 head_dim 发 kTmaChunks 次 TMA（box innermost=64 half），
  // 共享同一 barrier：N 次 cp.async.bulk.tensor 各 reduce chunk_bytes 到 tx count，
  // 1 次 mbarrier_arrive_expect_tx 声明总字节 + producer thread arrive。
  // ==================================================================
  if (wg_idx == 0) {
    NOTES_V2_REG_DEALLOC(40);
    if (wg_tid == 0) {
      // Step P0: Load block-level Q tile [Br, d] once. D=128 时发 kTmaChunks 次。
      //   minor_coord = c * 64, major_coord = qkv_major_base + Q_tile_id * Br
      //   smem dst = Q_smem + c * Br * 64 (chunk-major 布局)
      for (int c = 0; c < kTmaChunks; ++c) {
        tma_load_2d(Q_smem + c * Br * kTmaBoxMinor, tensorMapQ,
                    c * kTmaBoxMinor, qkv_major_base + Q_tile_id * Br, full_Q);
      }
      tma_arrive_expect_tx(full_Q, kQTileBytes);

      // Step P1: Prefetch first (kStagesK-1) K tiles.
      // K tile coords: minor = c*64, major = qkv_major_base + s*Bc
      for (int s = 0; s < kStagesK - 1; ++s) {
        empty_K[s].wait(empty_K[s].arrive());
        for (int c = 0; c < kTmaChunks; ++c) {
          tma_load_2d(K_smem + s * Bc * kHeadDim + c * Bc * kTmaBoxMinor,
                      tensorMapK, c * kTmaBoxMinor,
                      qkv_major_base + s * Bc, full_K[s]);
        }
        tma_arrive_expect_tx(full_K[s], kKTileBytes);
      }

      // Step P1b: Prefetch first (kStagesV-1) V tiles (only if kStagesV > 1).
      // kStagesV=1: no prefetch; V[k] loaded in P2 overlaps only QK gemm.
      // kStagesV>=2: V prefetched here; in P2 V[k+kStagesV-1] overlaps
      //   QK+softmax+PV of current iter → true V pipeline.
      if constexpr (kStagesV > 1) {
        for (int s = 0; s < kStagesV - 1; ++s) {
          empty_V[s].wait(empty_V[s].arrive());
          for (int c = 0; c < kTmaChunks; ++c) {
            tma_load_2d(V_smem + s * Bc * kHeadDim + c * Bc * kTmaBoxMinor,
                        tensorMapV, c * kTmaBoxMinor,
                        qkv_major_base + s * Bc, full_V[s]);
          }
          tma_arrive_expect_tx(full_V[s], kVTileBytes);
        }
      }

      // Step P2: Main loop over KV seqlen tiles
      for (int k = 0; k < Tc; ++k) {
        [[maybe_unused]] const int stage = k % kStagesK;

        // Issue V[k+kStagesV-1] TMA to stage_next_v (pipelined).
        // kStagesV=1: loads V[k] to stage 0, overlaps with QK of same iter.
        // kStagesV>=2: loads V[k+kStagesV-1], overlaps with QK+softmax+PV.
        {
          const int v_tile = k + kStagesV - 1;
          if (v_tile < Tc) {
            const int stage_next_v = v_tile % kStagesV;
            empty_V[stage_next_v].wait(empty_V[stage_next_v].arrive());
            for (int c = 0; c < kTmaChunks; ++c) {
              tma_load_2d(V_smem + stage_next_v * Bc * kHeadDim +
                              c * Bc * kTmaBoxMinor,
                          tensorMapV, c * kTmaBoxMinor,
                          qkv_major_base + v_tile * Bc,
                          full_V[stage_next_v]);
            }
            tma_arrive_expect_tx(full_V[stage_next_v], kVTileBytes);
          }
        }

        // Prefetch K[k+kStagesK-1] (pipelined).
        if (k + kStagesK - 1 < Tc) {
          const int stage_next = (k + kStagesK - 1) % kStagesK;
          empty_K[stage_next].wait(empty_K[stage_next].arrive());
          for (int c = 0; c < kTmaChunks; ++c) {
            tma_load_2d(K_smem + stage_next * Bc * kHeadDim +
                            c * Bc * kTmaBoxMinor,
                        tensorMapK, c * kTmaBoxMinor,
                        qkv_major_base + (k + kStagesK - 1) * Bc,
                        full_K[stage_next]);
          }
          tma_arrive_expect_tx(full_K[stage_next], kKTileBytes);
        }
      }
    }
  }
  // ==================================================================
  // Consumer Warpgroup (WG1, threadIdx.x 128~383, 256 threads = 8 warps)
  // All 256 threads participate in ldmatrix + mma.sync.
  // ==================================================================
  else {
    const int warp_id = wg_tid / kWarpSize;  // 0~7
    const int lane_id = wg_tid % kWarpSize;  // 0~31
    // Split-Q warp layout: 8 warps form 8x1 grid (warp_QP = 0~7, warp_KV = 0)
    // Br = 16 * 8 * 1 = 128 → 8 warps each handle 16 rows of Q
    const int warp_QP = warp_id;
    const int warp_KV = 0;

    // Step C0: init — mark all K stages and V stages as "empty" (consumable by producer)
    // Consumer register budget per Triton flash_attn_v2 maxnreg strategy on
    // Blackwell warp_specialize: D=128 -> 168, otherwise -> 80.
    if constexpr (kHeadDim == 128) {
      NOTES_V2_REG_ALLOC(168);
    } else {
      NOTES_V2_REG_ALLOC(80);
    }

    for (int s = 0; s < kStagesK; ++s) {
      [[maybe_unused]] auto token = empty_K[s].arrive();
    }
    for (int s = 0; s < kStagesV; ++s) {
      [[maybe_unused]] auto token = empty_V[s].arrive();
    }

    // Wait for Q tile ready once (split-Q: Q smem is reused across all KV iters)
    full_Q.wait(full_Q.arrive());

    // Online softmax persistent state (per-warp, per-row-pair)
    float lane_block_row_max_old[kValTileSeqLenQ][2];
    float lane_block_row_sum_old[kValTileSeqLenQ][2];
    fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_block_row_max_old, -INFINITY);
    fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_block_row_sum_old, 0.0f);

    uint32_t R_Q[kValTileSeqLenQ][4];
    uint32_t R_K[kValTileSeqLenK][2];
    uint32_t R_V[kValTileHeadDimV][2];
    uint32_t R_S[kValTileSeqLenQ][kValTileSeqLenK][kNRegs];
    uint32_t R_O[kValTileSeqLenP][kValTileHeadDimV][kNRegs];
    uint32_t R_D[kValTileSeqLenP][kValTileHeadDimV][kNRegs];

    fill_3D_regs<uint32_t, kValTileSeqLenP, kValTileHeadDimV, kNRegs>(R_D, 0);

    // ---- Main loop over KV seqlen ----
    for (int tile_K_seqlen = 0; tile_K_seqlen < Tc; ++tile_K_seqlen) {
      const int stage = tile_K_seqlen % kStagesK;
      const int stage_v = tile_K_seqlen % kStagesV;

      // Wait for K[stage] ready
      full_K[stage].wait(full_K[stage].arrive());
      tma_fence_proxy_async_shared_cta();

      // ---- 3b: Q @ K^T = S[Br, Bc] ----
      // Q smem reused (split-Q). K from stage `stage`.
      // smem addr (TMA 128B swizzle, swizzle<64>)
      fill_3D_regs<uint32_t, kValTileSeqLenQ, kValTileSeqLenK, kNRegs>(R_S, 0);
#pragma unroll
      for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
        // ldmatrix.x4: load Q m16k16 fragment (Q row-major, non-transpose)
#pragma unroll
        for (int i = 0; i < kValTileSeqLenQ; ++i) {
          const int warp_smem_Q_Br =
              warp_QP * (kMmaAtomM * kValTileSeqLenQ) + i * kMmaAtomM;
          const int lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;
          const int lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8;
          const uint32_t lane_smem_Q_ptr =
              smem_Q_base_ptr +
              swizzle_fa<kHeadDim, Br>(lane_smem_Q_Br, lane_smem_Q_d) *
                  sizeof(half);
          LDMATRIX_X4(R_Q[i][0], R_Q[i][1], R_Q[i][2], R_Q[i][3],
                      lane_smem_Q_ptr);
        }

        // ldmatrix.x2: load K k16n8 fragment (K row-major = K^T col-major)
#pragma unroll
        for (int j = 0; j < kValTileSeqLenK; ++j) {
          const int warp_smem_K_Bc =
              warp_KV * (kMmaAtomN * kValTileSeqLenK) + j * kMmaAtomN;
          const int lane_smem_K_Bc = warp_smem_K_Bc + lane_id % 8;
          const int lane_smem_K_d =
              tile_K_d * kMmaAtomK + ((lane_id / 8) % 2) * 8;
          const uint32_t lane_smem_K_ptr =
              smem_K_base_ptr +
              (stage * Bc * kHeadDim +
               swizzle_fa<kHeadDim, Bc>(lane_smem_K_Bc, lane_smem_K_d)) *
                  sizeof(half);
          LDMATRIX_X2(R_K[j][0], R_K[j][1], lane_smem_K_ptr);
        }

#pragma unroll
        for (int i = 0; i < kValTileSeqLenQ; ++i) {
#pragma unroll
          for (int j = 0; j < kValTileSeqLenK; ++j) {
            if constexpr (kMmaAccF32) {
              HMMA16816F32(R_S[i][j][0], R_S[i][j][1], R_S[i][j][2], R_S[i][j][3],
                           R_Q[i][0], R_Q[i][1], R_Q[i][2], R_Q[i][3],
                           R_K[j][0], R_K[j][1],
                           R_S[i][j][0], R_S[i][j][1], R_S[i][j][2], R_S[i][j][3]);
            } else {
              HMMA16816(R_S[i][j][0], R_S[i][j][1], R_Q[i][0], R_Q[i][1],
                        R_Q[i][2], R_Q[i][3], R_K[j][0], R_K[j][1],
                        R_S[i][j][0], R_S[i][j][1]);
            }
          }
        }
      }

      // Release K[stage] for producer reuse as early as possible: QK^T GEMM has
      // consumed all K smem data; the subsequent softmax (3c) and PV GEMM (3d)
      // only touch registers (R_S, R_O) and V smem, so K[stage] is free to be
      // overwritten by the producer's next TMA prefetch.
      {
        [[maybe_unused]] auto token = empty_K[stage].arrive();
      }

      // ---- 3c: Online Safe Softmax — row max + exp + sum, P back to R_S ----
      // Softmax only touches R_S (registers), so V TMA latency can be overlapped
      // with this compute. Defer full_V.wait() until just before P@V (3d).
      float lane_row_max_new[kValTileSeqLenQ][2];
      float lane_row_sum_new[kValTileSeqLenQ][2];
      fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_row_max_new, -INFINITY);
      fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_row_sum_new, 0.0f);

      // Pass 1: row-wise max across kValTileSeqLenK tiles (warp_reduce_max<4>)
#pragma unroll
      for (int i = 0; i < kValTileSeqLenQ; ++i) {
#pragma unroll
        for (int j = 0; j < kValTileSeqLenK; ++j) {
          float tmp_max_0, tmp_max_1;
          if constexpr (kMmaAccF32) {
            float *t_fptr_S = reinterpret_cast<float *>(&R_S[i][j][0]);
            tmp_max_0 = max(t_fptr_S[0], t_fptr_S[1]) * scale;
            tmp_max_1 = max(t_fptr_S[2], t_fptr_S[3]) * scale;
          } else {
            float2 t_reg_S_0 = __half22float2(HALF2(R_S[i][j][0]));
            float2 t_reg_S_1 = __half22float2(HALF2(R_S[i][j][1]));
            tmp_max_0 = max(t_reg_S_0.x, t_reg_S_0.y) * scale;
            tmp_max_1 = max(t_reg_S_1.x, t_reg_S_1.y) * scale;
          }
          lane_row_max_new[i][0] = max(lane_row_max_new[i][0], tmp_max_0);
          lane_row_max_new[i][1] = max(lane_row_max_new[i][1], tmp_max_1);
        }
        lane_row_max_new[i][0] =
            warp_reduce_max<4, float>(lane_row_max_new[i][0]);
        lane_row_max_new[i][1] =
            warp_reduce_max<4, float>(lane_row_max_new[i][1]);
      }

      // Pass 2: P = exp(S*scale - m_new), store back to R_S, accumulate row sums
#pragma unroll
      for (int i = 0; i < kValTileSeqLenQ; ++i) {
        float block_row_max_new_0 =
            max(lane_block_row_max_old[i][0], lane_row_max_new[i][0]);
        float block_row_max_new_1 =
            max(lane_block_row_max_old[i][1], lane_row_max_new[i][1]);

#pragma unroll
        for (int j = 0; j < kValTileSeqLenK; ++j) {
          if constexpr (kMmaAccF32) {
            float *t_fptr_S = reinterpret_cast<float *>(&R_S[i][j][0]);
            half *t_hptr_S = reinterpret_cast<half *>(&R_S[i][j][0]);
            t_fptr_S[0] = __expf(__fmaf_rn(t_fptr_S[0], scale, -block_row_max_new_0));
            t_fptr_S[1] = __expf(__fmaf_rn(t_fptr_S[1], scale, -block_row_max_new_0));
            t_fptr_S[2] = __expf(__fmaf_rn(t_fptr_S[2], scale, -block_row_max_new_1));
            t_fptr_S[3] = __expf(__fmaf_rn(t_fptr_S[3], scale, -block_row_max_new_1));
            lane_row_sum_new[i][0] += (t_fptr_S[0] + t_fptr_S[1]);
            lane_row_sum_new[i][1] += (t_fptr_S[2] + t_fptr_S[3]);
            t_hptr_S[0] = __float2half_rn(t_fptr_S[0]);
            t_hptr_S[1] = __float2half_rn(t_fptr_S[1]);
            t_hptr_S[2] = __float2half_rn(t_fptr_S[2]);
            t_hptr_S[3] = __float2half_rn(t_fptr_S[3]);
          } else {
            float2 t_reg_S_0 = __half22float2(HALF2(R_S[i][j][0]));
            float2 t_reg_S_1 = __half22float2(HALF2(R_S[i][j][1]));
            t_reg_S_0.x = __expf(__fmaf_rn(t_reg_S_0.x, scale, -block_row_max_new_0));
            t_reg_S_0.y = __expf(__fmaf_rn(t_reg_S_0.y, scale, -block_row_max_new_0));
            t_reg_S_1.x = __expf(__fmaf_rn(t_reg_S_1.x, scale, -block_row_max_new_1));
            t_reg_S_1.y = __expf(__fmaf_rn(t_reg_S_1.y, scale, -block_row_max_new_1));
            lane_row_sum_new[i][0] += (t_reg_S_0.x + t_reg_S_0.y);
            lane_row_sum_new[i][1] += (t_reg_S_1.x + t_reg_S_1.y);
            HALF2(R_S[i][j][0]) = __float22half2_rn(t_reg_S_0);
            HALF2(R_S[i][j][1]) = __float22half2_rn(t_reg_S_1);
          }
        }
        lane_row_sum_new[i][0] =
            warp_reduce_sum<4, float>(lane_row_sum_new[i][0]);
        lane_row_sum_new[i][1] =
            warp_reduce_sum<4, float>(lane_row_sum_new[i][1]);
      }

      // ---- 3d: P @ V = O[Br, d] ----
      // Wait for V[stage_v] ready just before P@V.
      // kStagesV=1: V was loaded in same iter, TMA overlaps QK+softmax.
      // kStagesV>=2: V was loaded in previous iter, TMA overlaps QK+softmax+PV.
      full_V[stage_v].wait(full_V[stage_v].arrive());
      tma_fence_proxy_async_shared_cta();

      // ldmatrix.x2.trans: V row-major → col-major B fragment for NN matmul
      fill_3D_regs<uint32_t, kValTileSeqLenP, kValTileHeadDimV, kNRegs>(R_O, 0);
#pragma unroll
      for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK); ++tile_V_Bc) {
#pragma unroll
        for (int j = 0; j < kValTileHeadDimV; ++j) {
          const int warp_smem_V_d =
              warp_KV * (kMmaAtomN * kValTileHeadDimV) + j * kMmaAtomN;
          const int lane_smem_V_Bc = tile_V_Bc * kMmaAtomK + lane_id % 16;
          const int lane_smem_V_d = warp_smem_V_d;
          const uint32_t lane_smem_V_ptr =
              smem_V_base_ptr +
              (stage_v * Bc * kHeadDim +
               swizzle_fa<kHeadDim, Bc>(lane_smem_V_Bc, lane_smem_V_d)) *
                  sizeof(half);
          LDMATRIX_X2_T(R_V[j][0], R_V[j][1], lane_smem_V_ptr);
        }

        // P fragment (R_S) directly feeds A of P@V MMA (layout reuse trick)
        const int w = tile_V_Bc * 2;
#pragma unroll
        for (int i = 0; i < kValTileSeqLenP; ++i) {
#pragma unroll
          for (int j = 0; j < kValTileHeadDimV; ++j) {
            if constexpr (kMmaAccF32) {
              HMMA16816F32(R_O[i][j][0], R_O[i][j][1], R_O[i][j][2], R_O[i][j][3],
                           R_S[i][w][0], R_S[i][w][1], R_S[i][w + 1][0], R_S[i][w + 1][1],
                           R_V[j][0], R_V[j][1],
                           R_O[i][j][0], R_O[i][j][1], R_O[i][j][2], R_O[i][j][3]);
            } else {
              HMMA16816(R_O[i][j][0], R_O[i][j][1],
                        R_S[i][w][0], R_S[i][w][1],
                        R_S[i][w + 1][0], R_S[i][w + 1][1],
                        R_V[j][0], R_V[j][1],
                        R_O[i][j][0], R_O[i][j][1]);
            }
          }
        }
      }

      // Release V[stage_v] for producer reuse: PV GEMM has consumed all
      // V smem data; the subsequent online rescaling (3e) only touches registers
      // (R_O, R_D, R_S, lane_block_*), so V[stage_v] is free to be overwritten.
      {
        [[maybe_unused]] auto token = empty_V[stage_v].arrive();
      }

      // ---- 3e: Online rescaling — O_new = exp(m_old - m_new) * O_old + P@V ----
#pragma unroll
      for (int i = 0; i < kValTileSeqLenP; ++i) {
        float block_row_max_new_0 = lane_row_max_new[i][0];
        float block_row_max_new_1 = lane_row_max_new[i][1];
        float block_row_sum_new_0 = lane_row_sum_new[i][0];
        float block_row_sum_new_1 = lane_row_sum_new[i][1];
        float block_row_max_old_0 = lane_block_row_max_old[i][0];
        float block_row_max_old_1 = lane_block_row_max_old[i][1];

        block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
        block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);

        // First iteration: m_old = -inf, use m_new directly
        block_row_max_old_0 = (tile_K_seqlen > 0 ? block_row_max_old_0
                                                  : block_row_max_new_0);
        block_row_max_old_1 = (tile_K_seqlen > 0 ? block_row_max_old_1
                                                  : block_row_max_new_1);

        float rescale_o_factor_0 =
            __expf(block_row_max_old_0 - block_row_max_new_0);
        float rescale_o_factor_1 =
            __expf(block_row_max_old_1 - block_row_max_new_1);

#pragma unroll
        for (int j = 0; j < kValTileHeadDimV; ++j) {
          if constexpr (kMmaAccF32) {
            float *t_fptr_O = reinterpret_cast<float *>(&R_O[i][j][0]);
            float *t_fptr_D = reinterpret_cast<float *>(&R_D[i][j][0]);
            t_fptr_D[0] = __fmaf_rn(rescale_o_factor_0, t_fptr_D[0], t_fptr_O[0]);
            t_fptr_D[1] = __fmaf_rn(rescale_o_factor_0, t_fptr_D[1], t_fptr_O[1]);
            t_fptr_D[2] = __fmaf_rn(rescale_o_factor_1, t_fptr_D[2], t_fptr_O[2]);
            t_fptr_D[3] = __fmaf_rn(rescale_o_factor_1, t_fptr_D[3], t_fptr_O[3]);
          } else {
            float2 t_reg_O_0 = __half22float2(HALF2(R_O[i][j][0]));
            float2 t_reg_O_1 = __half22float2(HALF2(R_O[i][j][1]));
            float2 t_reg_D_0 = __half22float2(HALF2(R_D[i][j][0]));
            float2 t_reg_D_1 = __half22float2(HALF2(R_D[i][j][1]));
            t_reg_D_0.x = __fmaf_rn(rescale_o_factor_0, t_reg_D_0.x, t_reg_O_0.x);
            t_reg_D_0.y = __fmaf_rn(rescale_o_factor_0, t_reg_D_0.y, t_reg_O_0.y);
            t_reg_D_1.x = __fmaf_rn(rescale_o_factor_1, t_reg_D_1.x, t_reg_O_1.x);
            t_reg_D_1.y = __fmaf_rn(rescale_o_factor_1, t_reg_D_1.y, t_reg_O_1.y);
            HALF2(R_D[i][j][0]) = __float22half2_rn(t_reg_D_0);
            HALF2(R_D[i][j][1]) = __float22half2_rn(t_reg_D_1);
          }
        }

        float block_row_sum_old_0 = lane_block_row_sum_old[i][0];
        float block_row_sum_old_1 = lane_block_row_sum_old[i][1];
        lane_block_row_sum_old[i][0] =
            __fmaf_rn(rescale_o_factor_0, block_row_sum_old_0,
                      block_row_sum_new_0);
        lane_block_row_sum_old[i][1] =
            __fmaf_rn(rescale_o_factor_1, block_row_sum_old_1,
                      block_row_sum_new_1);
        lane_block_row_max_old[i][0] = block_row_max_new_0;
        lane_block_row_max_old[i][1] = block_row_max_new_1;
      }
    }

    // ---- Step 4: Final rescale — O_final = (1/l_final) * O_final ----
#pragma unroll
    for (int i = 0; i < kValTileSeqLenP; ++i) {
      float rescale_factor_0 = __frcp_rn(lane_block_row_sum_old[i][0]);
      float rescale_factor_1 = __frcp_rn(lane_block_row_sum_old[i][1]);
#pragma unroll
      for (int j = 0; j < kValTileHeadDimV; ++j) {
        if constexpr (kMmaAccF32) {
          float *t_fptr_D = reinterpret_cast<float *>(&R_D[i][j][0]);
          half *t_hptr_D = reinterpret_cast<half *>(&R_D[i][j][0]);
          t_hptr_D[0] = __float2half_rn(rescale_factor_0 * t_fptr_D[0]);
          t_hptr_D[1] = __float2half_rn(rescale_factor_0 * t_fptr_D[1]);
          t_hptr_D[2] = __float2half_rn(rescale_factor_1 * t_fptr_D[2]);
          t_hptr_D[3] = __float2half_rn(rescale_factor_1 * t_fptr_D[3]);
        } else {
          float2 t_reg_D_0 = __half22float2(HALF2(R_D[i][j][0]));
          float2 t_reg_D_1 = __half22float2(HALF2(R_D[i][j][1]));
          t_reg_D_0.x = rescale_factor_0 * t_reg_D_0.x;
          t_reg_D_0.y = rescale_factor_0 * t_reg_D_0.y;
          t_reg_D_1.x = rescale_factor_1 * t_reg_D_1.x;
          t_reg_D_1.y = rescale_factor_1 * t_reg_D_1.y;
          HALF2(R_D[i][j][0]) = __float22half2_rn(t_reg_D_0);
          HALF2(R_D[i][j][1]) = __float22half2_rn(t_reg_D_1);
        }
      }
    }

    // ---- Step 5: Epilogue — warp shuffle + 128-bit collective store ----
#pragma unroll
    for (int i = 0; i < kValTileSeqLenP; ++i) {
#pragma unroll
      for (int j = 0; j < kValTileHeadDimV; ++j) {
        uint32_t R_Z[2][4];
        R_Z[0][0] = R_D[i][j][0];
        R_Z[1][0] = R_D[i][j][1];
        R_Z[0][1] = __shfl_sync(0xffffffff, R_D[i][j][0], lane_id + 1, 4);
        R_Z[0][2] = __shfl_sync(0xffffffff, R_D[i][j][0], lane_id + 2, 4);
        R_Z[0][3] = __shfl_sync(0xffffffff, R_D[i][j][0], lane_id + 3, 4);
        R_Z[1][1] = __shfl_sync(0xffffffff, R_D[i][j][1], lane_id + 1, 4);
        R_Z[1][2] = __shfl_sync(0xffffffff, R_D[i][j][1], lane_id + 2, 4);
        R_Z[1][3] = __shfl_sync(0xffffffff, R_D[i][j][1], lane_id + 3, 4);

        if (lane_id % 4 == 0) {
          const int store_warp_regs_O_Br =
              warp_QP * (kMmaAtomM * kValTileSeqLenP) + i * kMmaAtomM;
          const int store_lane_gmem_O_Br =
              Q_tile_id * Br + store_warp_regs_O_Br + lane_id / 4;
          const int store_warp_regs_O_d =
              warp_KV * (kMmaAtomN * kValTileHeadDimV) + j * kMmaAtomN;
          const int store_lane_gmem_O_d = store_warp_regs_O_d;
          const int store_gmem_O_addr_0 =
              O_gmem_offset + store_lane_gmem_O_Br * kHeadDim +
              store_lane_gmem_O_d;
          const int store_gmem_O_addr_1 =
              O_gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim +
              store_lane_gmem_O_d;
          *reinterpret_cast<float4 *>(&O[store_gmem_O_addr_0]) =
              *reinterpret_cast<float4 *>(&R_Z[0][0]);
          *reinterpret_cast<float4 *>(&O[store_gmem_O_addr_1]) =
              *reinterpret_cast<float4 *>(&R_Z[1][0]);
        }
      }
    }
  }
}

// =============================================================================
// Phase 8c: FlashAttention-3-style TMA + MMA Warp Specialization (SM120)
//   Dual 128-thread consumer warpgroups processing disjoint KV tiles,
//   with per-WG independent K pipeline (kStagesK) and single V buffer (kStagesV=1).
//
// 与 flash_attn_tma_mma_ws_stages_split_q (Phase 8b) 的核心区别：
//   - 8b: 1 producer WG (128T) + 1 consumer WG (256T, 8 warps), 全 KV 遍历
//   - 8c: 1 producer WG (128T) + 2 consumer WGs (各128T, 4 warps), KV tile 奇偶拆分
//
// 线程布局 (384 threads):
//   WG0 [0,127]:   TMA producer (仅 thread 0 发 TMA)
//   WG1 [128,255]: consumer_id=0, 处理偶数 KV tile (0,2,4,...)
//   WG2 [256,383]: consumer_id=1, 处理奇数 KV tile (1,3,5,...)
//
// per-WG 独立 stage 语义 (关键改进):
// ----------------------------------------------------------------
// 每个 consumer WG 拥有独立的 K stage slots (kStagesK 个) 和 1 个 V buffer。
// K/V smem 按 consumer_id 分 bank: K_smem[cid][stage], V_smem[cid]。
// Barrier 也按 consumer_id 隔离: full_K[cid][s], empty_K[cid][s], full_V[cid], empty_V[cid]。
// 两个 WG 绝不争抢同一 smem slot 或 barrier，producer 通过 cid = tile&1 路由 TMA。
//
// 这样 stage 语义从 "跨 WG 共享的 2 slot (每 WG 实际 depth=1)" 变为
// "每 WG 独立拥有 kStagesK 个 K slot (每 WG 真正 depth=kStagesK)"。
//
// Pipeline 时序 (Sk=1, Sv=1):
//   K[tile] TMA 被 softmax+PV+rescale 隐藏 (Sk=1 复用同 slot，用完即释放);
//   V[tile] TMA 被 rescale+QK[next]+softmax[next] 隐藏 (Sv=1 同理)。
//
// Pipeline 时序 (Sk=2, Sv=1, D=64 only):
//   K[tile+2] 在 QK[tile] 之前就 prefetch 到另一个 slot，被整个 tile compute 隐藏;
//   V[tile] TMA 同 Sk=1，被 QK+softmax 隐藏。
//
// 为什么是奇偶 tile 拆分 (而非前/后半区各自一半)?
// ----------------------------------------------------------------
// 关键约束: producer 按 global tile 顺序 0,1,2,3,... 连续 TMA。
// 奇偶拆分让 cid = tile & 1 静态路由: tile 0,2,4,... -> WG1; tile 1,3,5,... -> WG2。
// 每 WG 的下一个 tile (tile+2) 正好是 producer 当前/下一轮要加载的 tile，
// 保证 producer/consumer 流水无停顿。
//
// 若改为前/后半区拆分: 过渡区两个 WG 可能同时需要同一 cid 的 slot,
// 且 WG2 跳跃访问远端 tile 时数据尚未被 producer TMA 到 smem。
//
// Partial state merge (Split-KV / FlashDecoding-style reduction):
// ----------------------------------------------------------------
// 每个 WG 各自处理一半 KV tile 序列, 独立维护 "未归一化" 的 online-softmax
// 中间状态 (m, l, Oacc), 合并后等价于全 KV 遍历。
//
// 稳定归并公式 (避免 exp 溢出):
//   m     = max(m_0, m_1)            // 全局 row max
//   alpha = exp(m_0 - m)  (<=1)      // WG1 rescale factor
//   beta  = exp(m_1 - m)  (<=1)      // WG2 rescale factor
//   l     = alpha * l_0 + beta * l_1
//   Oacc  = alpha * Oacc_0 + beta * Oacc_1
//   O     = Oacc / l                 // 最终归一化
//
// Tc==1 退化: WG2 无 tile, m_1=-inf, l_1=0, Oacc_1=0;
//   beta = exp(-inf) = 0, 退化为 WG1 结果。
//
// 限制 (首版):
//   - kStagesV == 1 (V 只需 1 buffer, TMA 被 QK+softmax 隐藏)
//   - kStagesK >= 1 (K pipeline depth; D=128 在 SM120 只支持 Sk=1)
//   - D=128 时 Sk=1 (Sk=2 需 112KB > 101KB optin 上限)
//   - Br=64, Bc=64, D=64/128, aligned seqlen (N % Br == 0 && N % Bc == 0)
//   - 仅 self-attention, 无 causal/varlen/GQA
// =============================================================================
template <
    const int kHeadDim,
    const int kMmaAtomM, const int kMmaAtomN, const int kMmaAtomK,
    const int kMmaAccF32,
    const int kMmaTileSeqLenQ, const int kMmaTileSeqLenK,
    const int kMmaTileSeqLenP, const int kMmaTileHeadDimV,
    const int kValTileSeqLenQ, const int kValTileSeqLenK,
    const int kValTileSeqLenP, const int kValTileHeadDimV,
    const int kStagesK, const int kStagesV,
    const int kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
    flash_attn_3_tma_ws_stages_split_q(
        half *Q, half *K, half *V, half *O, int N, int H,
        const CUtensorMap *__restrict__ tensorMapQ,
        const CUtensorMap *__restrict__ tensorMapK,
        const CUtensorMap *__restrict__ tensorMapV) {
  static_assert(kHeadDim == 64 || kHeadDim == 128,
                "v1 supports kHeadDim == 64 or 128");
  static_assert(kNumThreads == 384, "128 producer + 256 consumer (2 WGs)");
  static_assert(kMmaAccF32 == 0 || kMmaAccF32 == 1, "kMmaAccF32 must be 0 or 1");
  static_assert(kStagesV == 1, "per-WG V pipeline depth must be 1");
  static_assert(kStagesK >= 1, "kStagesK must be >= 1");
  constexpr int kNRegs = kMmaAccF32 ? 4 : 2;

  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kValTileSeqLenQ; // 64
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kValTileSeqLenK; // 64
  constexpr int kConsumerThreadsPerWG = 128;
  constexpr int kProducerThreads = 128;
  constexpr int kNumConsumerWGs = 2;
  constexpr int kQTileBytes = Br * kHeadDim * sizeof(half);
  constexpr int kKTileBytes = Bc * kHeadDim * sizeof(half);
  constexpr int kVTileBytes = Bc * kHeadDim * sizeof(half);
  constexpr int kTmaBoxMinor = 64;
  constexpr int kTmaChunks = kHeadDim / kTmaBoxMinor;

  const int Nb_id = blockIdx.y / H;
  const int Nh_id = blockIdx.y % H;
  const int Q_tile_id = blockIdx.x;
  const int Tc = (N + Bc - 1) / Bc;
  const float scale = 1.0f / sqrtf((float)kHeadDim);

  const int QKV_gmem_offset = (Nb_id * H * N + Nh_id * N) * kHeadDim;
  const int O_gmem_offset = QKV_gmem_offset;
  const int qkv_major_base = (Nb_id * H + Nh_id) * N;

  // ---- Shared memory layout (per-WG independent K/V banks) ----
  // [Q_shared: Br*D]
  // [K[cid=0][s=0..Sk-1]: Sk*Bc*D]  [K[cid=1][s=0..Sk-1]]
  // [V[cid=0]: Bc*D]                 [V[cid=1]: Bc*D]
  extern __shared__ __align__(1024) uint8_t smem_fa3_tma_ws[];
  half *Q_smem = reinterpret_cast<half *>(smem_fa3_tma_ws);
  half *K_smem_base = Q_smem + Br * kHeadDim;
  // K: [numConsumerWGs][kStagesK][Bc*kHeadDim], linearized
  //   K_smem_base + cid * (kStagesK * Bc * kHeadDim) + stg * (Bc * kHeadDim)
  half *V_smem_base = K_smem_base + kNumConsumerWGs * kStagesK * Bc * kHeadDim;
  // V: [numConsumerWGs][Bc*kHeadDim]
  //   V_smem_base + cid * (Bc * kHeadDim)

  const uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_smem);
  const uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_smem_base);
  const uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_smem_base);

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ cuda::barrier<cuda::thread_scope_block> full_Q;
  // per-WG K barriers: [cid][stage]
  __shared__ cuda::barrier<cuda::thread_scope_block> full_K[kNumConsumerWGs][kStagesK];
  __shared__ cuda::barrier<cuda::thread_scope_block> empty_K[kNumConsumerWGs][kStagesK];
  // per-WG V barriers: [cid] (kStagesV=1)
  __shared__ cuda::barrier<cuda::thread_scope_block> full_V[kNumConsumerWGs];
  __shared__ cuda::barrier<cuda::thread_scope_block> empty_V[kNumConsumerWGs];
#pragma nv_diag_default static_var_with_dynamic_init

  const bool is_producer = threadIdx.x < kProducerThreads;
  const int consumer_id = is_producer ? 0
                    : (threadIdx.x - kProducerThreads) / kConsumerThreadsPerWG;
  const int wg_tid = is_producer ? threadIdx.x
                    : (threadIdx.x - kProducerThreads) % kConsumerThreadsPerWG;

  if (threadIdx.x == 0) {
    init(&full_Q, 256 + 1);  // 2 consumer WGs * 128 + 1 producer
    for (int cid = 0; cid < kNumConsumerWGs; ++cid) {
      for (int s = 0; s < kStagesK; ++s) {
        init(&full_K[cid][s], kConsumerThreadsPerWG + 1);
        init(&empty_K[cid][s], kConsumerThreadsPerWG + 1);
      }
      init(&full_V[cid], kConsumerThreadsPerWG + 1);
      init(&empty_V[cid], kConsumerThreadsPerWG + 1);
    }
    tma_fence_proxy_async_shared_cta();
  }
  __syncthreads();

  // Helper: K_smem ptr for (cid, stg)
  // K layout: K_smem_base + cid*(Sk*Bc*D) + stg*(Bc*D)
  // D=128 chunk-major: each tile's 2 chunks at stg*Bc*D + c*Bc*64

  // ==================================================================
  // Producer Warpgroup (WG0, threadIdx.x 0~127)
  // Single global tile loop over all KV tiles (0..Tc-1).
  // cid = tile & 1 routes TMA to the owning consumer WG's smem bank.
  //
  // Per-tile load order: V first, then K (prefetch next).
  //   - V[tile] TMA: blocked until consumer releases V[cid] (after PV).
  //     But K[tile] signal already sent in warmup/prev iter, so consumer
  //     can do QK[tile] while producer waits for V release.
  //   - K[tile+2] prefetch (Sk>=2): blocked until consumer releases K[cid][stg_next]
  //     (after QK). QK is first compute step, always done before producer reaches here.
  // ==================================================================
  if (is_producer) {
    NOTES_V2_REG_DEALLOC(40);
    if (wg_tid == 0) {
      // P0: Load Q tile once
      for (int c = 0; c < kTmaChunks; ++c) {
        tma_load_2d(Q_smem + c * Br * kTmaBoxMinor, tensorMapQ,
                    c * kTmaBoxMinor, qkv_major_base + Q_tile_id * Br, full_Q);
      }
      tma_arrive_expect_tx(full_Q, kQTileBytes);

      // P1: Warmup - prefetch first K tile for each consumer WG (Sk-1 tiles)
      // tile 0 -> cid=0, tile 1 -> cid=1
      if constexpr (kStagesK > 1) {
        for (int cid = 0; cid < kNumConsumerWGs; ++cid) {
          const int tile0 = cid;  // first tile for this WG
          if (tile0 < Tc) {
            const int stg = 0;  // first slot
            empty_K[cid][stg].wait(empty_K[cid][stg].arrive());
            for (int c = 0; c < kTmaChunks; ++c) {
              tma_load_2d(K_smem_base + cid * (kStagesK * Bc * kHeadDim) +
                              stg * Bc * kHeadDim + c * Bc * kTmaBoxMinor,
                          tensorMapK, c * kTmaBoxMinor,
                          qkv_major_base + tile0 * Bc, full_K[cid][stg]);
            }
            tma_arrive_expect_tx(full_K[cid][stg], kKTileBytes);
          }
        }
      }

      // P2: Main loop over all global KV tiles
      // K 优先于 V: K 的 release 依赖 QK (tile 第一步 compute, 早完成),
      // V 的 release 依赖 PV (tile 最后一步 compute, 晚完成)。
      // 先发 K 让 consumer 尽早开始 QK, V 的 TMA 在 K 之后发出,
      // 被 QK+softmax 的 compute 时间隐藏。
      for (int tile = 0; tile < Tc; ++tile) {
        const int cid = tile & 1;

        // Step 1: Load/prefetch K first (release depends on QK, which is early)
        if constexpr (kStagesK == 1) {
          // Sk=1: load K[tile] to the single slot (reuse after consumer QK release)
          empty_K[cid][0].wait(empty_K[cid][0].arrive());
          for (int c = 0; c < kTmaChunks; ++c) {
            tma_load_2d(K_smem_base + cid * (kStagesK * Bc * kHeadDim) +
                            c * Bc * kTmaBoxMinor,
                        tensorMapK, c * kTmaBoxMinor,
                        qkv_major_base + tile * Bc, full_K[cid][0]);
          }
          tma_arrive_expect_tx(full_K[cid][0], kKTileBytes);
        } else {
          // Sk>=2: prefetch K[tile+2] to next slot (true ping-pong pipeline!)
          const int local_tile = (tile - cid) / 2;
          const int stg_next = (local_tile + 1) % kStagesK;
          const int next_tile = tile + 2;
          if (next_tile < Tc) {
            empty_K[cid][stg_next].wait(empty_K[cid][stg_next].arrive());
            for (int c = 0; c < kTmaChunks; ++c) {
              tma_load_2d(K_smem_base + cid * (kStagesK * Bc * kHeadDim) +
                              stg_next * Bc * kHeadDim + c * Bc * kTmaBoxMinor,
                          tensorMapK, c * kTmaBoxMinor,
                          qkv_major_base + next_tile * Bc,
                          full_K[cid][stg_next]);
            }
            tma_arrive_expect_tx(full_K[cid][stg_next], kKTileBytes);
          }
        }

        // Step 2: Load V[tile] (release depends on PV, which is late)
        empty_V[cid].wait(empty_V[cid].arrive());
        for (int c = 0; c < kTmaChunks; ++c) {
          tma_load_2d(V_smem_base + cid * Bc * kHeadDim + c * Bc * kTmaBoxMinor,
                      tensorMapV, c * kTmaBoxMinor,
                      qkv_major_base + tile * Bc, full_V[cid]);
        }
        tma_arrive_expect_tx(full_V[cid], kVTileBytes);
      }
    }
  }
  // ==================================================================
  // Consumer Warpgroups (WG1 consumer_id=0, WG2 consumer_id=1)
  // Each WG processes every other KV tile (consumer_id, consumer_id+2, ...).
  // K stage index: local_iter % kStagesK (Sk=1: always 0; Sk=2: 0,1,0,1,...)
  // V stage index: always consumer_id (only 1 V buffer per WG)
  // ==================================================================
  else {
    const int warp_id = wg_tid / kWarpSize;  // 0~3
    const int lane_id = wg_tid % kWarpSize;  // 0~31
    const int warp_QP = warp_id;
    const int warp_KV = 0;

    if constexpr (kHeadDim == 128) {
      NOTES_V2_REG_ALLOC(168);
    } else {
      NOTES_V2_REG_ALLOC(80);
    }

    // C0: init - mark OWN K/V slots as empty (consumer arrives, producer is 129th)
    for (int s = 0; s < kStagesK; ++s) {
      [[maybe_unused]] auto tk = empty_K[consumer_id][s].arrive();
    }
    {
      [[maybe_unused]] auto tv = empty_V[consumer_id].arrive();
    }

    // Wait for Q tile (shared by both WGs)
    full_Q.wait(full_Q.arrive());

    // Partial online-softmax state (per-WG, independent)
    float lane_block_row_max_old[kValTileSeqLenQ][2];
    float lane_block_row_sum_old[kValTileSeqLenQ][2];
    fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_block_row_max_old, -INFINITY);
    fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_block_row_sum_old, 0.0f);

    uint32_t R_Q[kValTileSeqLenQ][4];
    uint32_t R_K[kValTileSeqLenK][2];
    uint32_t R_V[kValTileHeadDimV][2];
    uint32_t R_S[kValTileSeqLenQ][kValTileSeqLenK][kNRegs];
    uint32_t R_O[kValTileSeqLenP][kValTileHeadDimV][kNRegs];
    uint32_t R_D[kValTileSeqLenP][kValTileHeadDimV][kNRegs];
    fill_3D_regs<uint32_t, kValTileSeqLenP, kValTileHeadDimV, kNRegs>(R_D, 0);

    // K smem offset for this WG: base + cid*(Sk*Bc*D)
    // stg within: local_iter % kStagesK
    // V smem offset for this WG: V_smem_base + cid*(Bc*D)
    const uint32_t smem_K_wg_base = smem_K_base_ptr +
        consumer_id * (kStagesK * Bc * kHeadDim) * sizeof(half);
    const uint32_t smem_V_wg_base = smem_V_base_ptr +
        consumer_id * (Bc * kHeadDim) * sizeof(half);

    int local_iter = 0;
    for (int tile = consumer_id; tile < Tc; tile += 2) {
      const int k_stg = local_iter % kStagesK;

      // K smem offset for this stage: wg_base + stg*(Bc*D)
      const uint32_t smem_K_stage_ptr = smem_K_wg_base +
          k_stg * Bc * kHeadDim * sizeof(half);

      // Wait for K[consumer_id][k_stg] ready
      full_K[consumer_id][k_stg].wait(full_K[consumer_id][k_stg].arrive());
      tma_fence_proxy_async_shared_cta();

      // ---- Q @ K^T = S[Br, Bc] ----
      // K early release: 在最后一个 tile_K_d 的 ldmatrix_K 完成后立即 release K,
      // 不等整个 QK MMA 循环结束。K 数据已全部读入 R_K 寄存器, smem 可被
      // producer 复用。这对 Sk=1 尤其重要: producer 可更早开始下一个 K TMA。
      fill_3D_regs<uint32_t, kValTileSeqLenQ, kValTileSeqLenK, kNRegs>(R_S, 0);
#pragma unroll
      for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
#pragma unroll
        for (int i = 0; i < kValTileSeqLenQ; ++i) {
          const int warp_smem_Q_Br =
              warp_QP * (kMmaAtomM * kValTileSeqLenQ) + i * kMmaAtomM;
          const int lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;
          const int lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8;
          const uint32_t lane_smem_Q_ptr =
              smem_Q_base_ptr +
              swizzle_fa<kHeadDim, Br>(lane_smem_Q_Br, lane_smem_Q_d) *
                  sizeof(half);
          LDMATRIX_X4(R_Q[i][0], R_Q[i][1], R_Q[i][2], R_Q[i][3],
                      lane_smem_Q_ptr);
        }

#pragma unroll
        for (int j = 0; j < kValTileSeqLenK; ++j) {
          const int warp_smem_K_Bc =
              warp_KV * (kMmaAtomN * kValTileSeqLenK) + j * kMmaAtomN;
          const int lane_smem_K_Bc = warp_smem_K_Bc + lane_id % 8;
          const int lane_smem_K_d =
              tile_K_d * kMmaAtomK + ((lane_id / 8) % 2) * 8;
          // K smem: stage_ptr + swizzle_fa offset (no stage*Bc*D since stage_ptr already includes it)
          const uint32_t lane_smem_K_ptr =
              smem_K_stage_ptr +
              swizzle_fa<kHeadDim, Bc>(lane_smem_K_Bc, lane_smem_K_d) *
                  sizeof(half);
          LDMATRIX_X2(R_K[j][0], R_K[j][1], lane_smem_K_ptr);
        }

        // Early release K after last ldmatrix_K: K data now fully in registers
        if (tile_K_d == (kHeadDim / kMmaAtomK) - 1) {
          [[maybe_unused]] auto token = empty_K[consumer_id][k_stg].arrive();
        }

#pragma unroll
        for (int i = 0; i < kValTileSeqLenQ; ++i) {
#pragma unroll
          for (int j = 0; j < kValTileSeqLenK; ++j) {
            if constexpr (kMmaAccF32) {
              HMMA16816F32(R_S[i][j][0], R_S[i][j][1], R_S[i][j][2], R_S[i][j][3],
                           R_Q[i][0], R_Q[i][1], R_Q[i][2], R_Q[i][3],
                           R_K[j][0], R_K[j][1],
                           R_S[i][j][0], R_S[i][j][1], R_S[i][j][2], R_S[i][j][3]);
            } else {
              HMMA16816(R_S[i][j][0], R_S[i][j][1], R_Q[i][0], R_Q[i][1],
                        R_Q[i][2], R_Q[i][3], R_K[j][0], R_K[j][1],
                        R_S[i][j][0], R_S[i][j][1]);
            }
          }
        }
      }

      // K already released above (after last ldmatrix_K)

      // ---- Online Safe Softmax ----
      // This compute overlaps with producer's V[tile] TMA (which waits for
      // empty_V from previous PV, or is already in-flight)
      float lane_row_max_new[kValTileSeqLenQ][2];
      float lane_row_sum_new[kValTileSeqLenQ][2];
      fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_row_max_new, -INFINITY);
      fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_row_sum_new, 0.0f);

#pragma unroll
      for (int i = 0; i < kValTileSeqLenQ; ++i) {
#pragma unroll
        for (int j = 0; j < kValTileSeqLenK; ++j) {
          float tmp_max_0, tmp_max_1;
          if constexpr (kMmaAccF32) {
            float *t_fptr_S = reinterpret_cast<float *>(&R_S[i][j][0]);
            tmp_max_0 = max(t_fptr_S[0], t_fptr_S[1]) * scale;
            tmp_max_1 = max(t_fptr_S[2], t_fptr_S[3]) * scale;
          } else {
            float2 t_reg_S_0 = __half22float2(HALF2(R_S[i][j][0]));
            float2 t_reg_S_1 = __half22float2(HALF2(R_S[i][j][1]));
            tmp_max_0 = max(t_reg_S_0.x, t_reg_S_0.y) * scale;
            tmp_max_1 = max(t_reg_S_1.x, t_reg_S_1.y) * scale;
          }
          lane_row_max_new[i][0] = max(lane_row_max_new[i][0], tmp_max_0);
          lane_row_max_new[i][1] = max(lane_row_max_new[i][1], tmp_max_1);
        }
        lane_row_max_new[i][0] =
            warp_reduce_max<4, float>(lane_row_max_new[i][0]);
        lane_row_max_new[i][1] =
            warp_reduce_max<4, float>(lane_row_max_new[i][1]);
      }

#pragma unroll
      for (int i = 0; i < kValTileSeqLenQ; ++i) {
        float block_row_max_new_0 =
            max(lane_block_row_max_old[i][0], lane_row_max_new[i][0]);
        float block_row_max_new_1 =
            max(lane_block_row_max_old[i][1], lane_row_max_new[i][1]);

#pragma unroll
        for (int j = 0; j < kValTileSeqLenK; ++j) {
          if constexpr (kMmaAccF32) {
            float *t_fptr_S = reinterpret_cast<float *>(&R_S[i][j][0]);
            half *t_hptr_S = reinterpret_cast<half *>(&R_S[i][j][0]);
            t_fptr_S[0] = __expf(__fmaf_rn(t_fptr_S[0], scale, -block_row_max_new_0));
            t_fptr_S[1] = __expf(__fmaf_rn(t_fptr_S[1], scale, -block_row_max_new_0));
            t_fptr_S[2] = __expf(__fmaf_rn(t_fptr_S[2], scale, -block_row_max_new_1));
            t_fptr_S[3] = __expf(__fmaf_rn(t_fptr_S[3], scale, -block_row_max_new_1));
            lane_row_sum_new[i][0] += (t_fptr_S[0] + t_fptr_S[1]);
            lane_row_sum_new[i][1] += (t_fptr_S[2] + t_fptr_S[3]);
            t_hptr_S[0] = __float2half_rn(t_fptr_S[0]);
            t_hptr_S[1] = __float2half_rn(t_fptr_S[1]);
            t_hptr_S[2] = __float2half_rn(t_fptr_S[2]);
            t_hptr_S[3] = __float2half_rn(t_fptr_S[3]);
          } else {
            float2 t_reg_S_0 = __half22float2(HALF2(R_S[i][j][0]));
            float2 t_reg_S_1 = __half22float2(HALF2(R_S[i][j][1]));
            t_reg_S_0.x = __expf(__fmaf_rn(t_reg_S_0.x, scale, -block_row_max_new_0));
            t_reg_S_0.y = __expf(__fmaf_rn(t_reg_S_0.y, scale, -block_row_max_new_0));
            t_reg_S_1.x = __expf(__fmaf_rn(t_reg_S_1.x, scale, -block_row_max_new_1));
            t_reg_S_1.y = __expf(__fmaf_rn(t_reg_S_1.y, scale, -block_row_max_new_1));
            lane_row_sum_new[i][0] += (t_reg_S_0.x + t_reg_S_0.y);
            lane_row_sum_new[i][1] += (t_reg_S_1.x + t_reg_S_1.y);
            HALF2(R_S[i][j][0]) = __float22half2_rn(t_reg_S_0);
            HALF2(R_S[i][j][1]) = __float22half2_rn(t_reg_S_1);
          }
        }
        lane_row_sum_new[i][0] =
            warp_reduce_sum<4, float>(lane_row_sum_new[i][0]);
        lane_row_sum_new[i][1] =
            warp_reduce_sum<4, float>(lane_row_sum_new[i][1]);
      }

      // ---- P @ V = O[Br, d] ----
      // Wait for V[consumer_id] ready
      full_V[consumer_id].wait(full_V[consumer_id].arrive());
      tma_fence_proxy_async_shared_cta();

      fill_3D_regs<uint32_t, kValTileSeqLenP, kValTileHeadDimV, kNRegs>(R_O, 0);
#pragma unroll
      for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK); ++tile_V_Bc) {
#pragma unroll
        for (int j = 0; j < kValTileHeadDimV; ++j) {
          const int warp_smem_V_d =
              warp_KV * (kMmaAtomN * kValTileHeadDimV) + j * kMmaAtomN;
          const int lane_smem_V_Bc = tile_V_Bc * kMmaAtomK + lane_id % 16;
          const int lane_smem_V_d = warp_smem_V_d;
          // V smem: V_wg_base + swizzle_fa offset
          const uint32_t lane_smem_V_ptr =
              smem_V_wg_base +
              swizzle_fa<kHeadDim, Bc>(lane_smem_V_Bc, lane_smem_V_d) *
                  sizeof(half);
          LDMATRIX_X2_T(R_V[j][0], R_V[j][1], lane_smem_V_ptr);
        }

        // Early release V after last ldmatrix_V: V data now fully in registers.
        // This lets producer start next V TMA during the remaining PV MMA +
        // online rescaling, giving more overlap window than releasing after PV.
        if (tile_V_Bc == (Bc / kMmaAtomK) - 1) {
          [[maybe_unused]] auto token = empty_V[consumer_id].arrive();
        }

        const int w = tile_V_Bc * 2;
#pragma unroll
        for (int i = 0; i < kValTileSeqLenP; ++i) {
#pragma unroll
          for (int j = 0; j < kValTileHeadDimV; ++j) {
            if constexpr (kMmaAccF32) {
              HMMA16816F32(R_O[i][j][0], R_O[i][j][1], R_O[i][j][2], R_O[i][j][3],
                           R_S[i][w][0], R_S[i][w][1], R_S[i][w + 1][0], R_S[i][w + 1][1],
                           R_V[j][0], R_V[j][1],
                           R_O[i][j][0], R_O[i][j][1], R_O[i][j][2], R_O[i][j][3]);
            } else {
              HMMA16816(R_O[i][j][0], R_O[i][j][1],
                        R_S[i][w][0], R_S[i][w][1],
                        R_S[i][w + 1][0], R_S[i][w + 1][1],
                        R_V[j][0], R_V[j][1],
                        R_O[i][j][0], R_O[i][j][1]);
            }
          }
        }
      }

      // V already released above (after last ldmatrix_V)

      // ---- Online rescaling - O_new = exp(m_old - m_new) * O_old + P@V ----
#pragma unroll
      for (int i = 0; i < kValTileSeqLenP; ++i) {
        float block_row_max_new_0 = lane_row_max_new[i][0];
        float block_row_max_new_1 = lane_row_max_new[i][1];
        float block_row_sum_new_0 = lane_row_sum_new[i][0];
        float block_row_sum_new_1 = lane_row_sum_new[i][1];
        float block_row_max_old_0 = lane_block_row_max_old[i][0];
        float block_row_max_old_1 = lane_block_row_max_old[i][1];

        block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
        block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);

        // First iteration per WG: m_old = -inf, use m_new directly
        block_row_max_old_0 = (local_iter > 0 ? block_row_max_old_0
                                              : block_row_max_new_0);
        block_row_max_old_1 = (local_iter > 0 ? block_row_max_old_1
                                              : block_row_max_new_1);

        float rescale_o_factor_0 =
            __expf(block_row_max_old_0 - block_row_max_new_0);
        float rescale_o_factor_1 =
            __expf(block_row_max_old_1 - block_row_max_new_1);

#pragma unroll
        for (int j = 0; j < kValTileHeadDimV; ++j) {
          if constexpr (kMmaAccF32) {
            float *t_fptr_O = reinterpret_cast<float *>(&R_O[i][j][0]);
            float *t_fptr_D = reinterpret_cast<float *>(&R_D[i][j][0]);
            t_fptr_D[0] = __fmaf_rn(rescale_o_factor_0, t_fptr_D[0], t_fptr_O[0]);
            t_fptr_D[1] = __fmaf_rn(rescale_o_factor_0, t_fptr_D[1], t_fptr_O[1]);
            t_fptr_D[2] = __fmaf_rn(rescale_o_factor_1, t_fptr_D[2], t_fptr_O[2]);
            t_fptr_D[3] = __fmaf_rn(rescale_o_factor_1, t_fptr_D[3], t_fptr_O[3]);
          } else {
            float2 t_reg_O_0 = __half22float2(HALF2(R_O[i][j][0]));
            float2 t_reg_O_1 = __half22float2(HALF2(R_O[i][j][1]));
            float2 t_reg_D_0 = __half22float2(HALF2(R_D[i][j][0]));
            float2 t_reg_D_1 = __half22float2(HALF2(R_D[i][j][1]));
            t_reg_D_0.x = __fmaf_rn(rescale_o_factor_0, t_reg_D_0.x, t_reg_O_0.x);
            t_reg_D_0.y = __fmaf_rn(rescale_o_factor_0, t_reg_D_0.y, t_reg_O_0.y);
            t_reg_D_1.x = __fmaf_rn(rescale_o_factor_1, t_reg_D_1.x, t_reg_O_1.x);
            t_reg_D_1.y = __fmaf_rn(rescale_o_factor_1, t_reg_D_1.y, t_reg_O_1.y);
            HALF2(R_D[i][j][0]) = __float22half2_rn(t_reg_D_0);
            HALF2(R_D[i][j][1]) = __float22half2_rn(t_reg_D_1);
          }
        }

        float block_row_sum_old_0 = lane_block_row_sum_old[i][0];
        float block_row_sum_old_1 = lane_block_row_sum_old[i][1];
        lane_block_row_sum_old[i][0] =
            __fmaf_rn(rescale_o_factor_0, block_row_sum_old_0,
                      block_row_sum_new_0);
        lane_block_row_sum_old[i][1] =
            __fmaf_rn(rescale_o_factor_1, block_row_sum_old_1,
                      block_row_sum_new_1);
        lane_block_row_max_old[i][0] = block_row_max_new_0;
        lane_block_row_max_old[i][1] = block_row_max_new_1;
      }
      ++local_iter;
    }

    // ==================================================================
    // Final merge of two WG partial states (split-KV reduction)
    //
    // 安全 alias dynamic smem 的前提:
    //   1) producer 已退出 mainloop (所有 TMA 已 arrive_expect_tx);
    //   2) 两个 consumer 都已 release 自己的 K/V stage (QK 后 release K,
    //      PV 后 release V), 且 mainloop 结束后不再访问任何 K/V smem;
    //   3) Q_smem 在 mainloop 内只读, 此后也不再访问。
    // 故此时整个 dynamic smem (Q/K/V 区) 生命周期结束, 可安全重命名为
    // merge scratch, 不增加 launch smem_bytes。第一次 __syncthreads
    // 保证所有 TMA/ldmatrix 完成后才开始写 scratch。
    // 完整的 split-KV merge 数学推导见 kernel 头部注释。
    // ==================================================================
    __syncthreads();

    // Merge scratch 布局 (按 wg_tid 索引, 每 WG 128 thread):
    //   [merge_rd]: 128 * kRdPerThread 个 uint32  <- WG2 未归一化 R_D fragment
    //   [merge_ml]: 128 * kValTileSeqLenQ 个 float4 <- WG2 的 [m0,m1,l0,l1] interleaved
    // WG1 用自己的 R_D/m/l 作为 (Oacc_0, m_0, l_0), 读取 WG2 写出的作为
    // (Oacc_1, m_1, l_1), 按 stable merge 公式归并后由 WG1 写最终 O。
    // m/l interleaved 为 float4 布局, 写入/读取各一次 128-bit store/load。
    constexpr int kRdPerThread = kValTileSeqLenP * kValTileHeadDimV * kNRegs;
    constexpr int kRdPackPerThread = kRdPerThread / 4;  // float4 个数 (kNRegs 始终偶数)
    constexpr int kMlPackPerThread = kValTileSeqLenQ;   // 每 thread 的 float4 个数
    uint32_t *merge_rd = reinterpret_cast<uint32_t *>(smem_fa3_tma_ws);
    float4 *merge_rd_pack = reinterpret_cast<float4 *>(merge_rd);
    float4 *merge_ml_pack = merge_rd_pack + 128 * kRdPackPerThread;

    // WG2 写出未归一化 partial (R_D, m, l) 到 scratch, 按 wg_tid 索引。
    // R_D: 以 float4 (128-bit) 连续写入, kRdPackPerThread 次。
    // m/l: interleaved 为 float4 [m0, m1, l0, l1], 每次 128-bit store。
    // 按 wg_tid 索引保证 WG1 的同一 lane 读到对应行的 WG2 partial:
    //   R_D[i][j][k] fragment 对应的 Q rows 与 lane_block_row_max/sum[i][0/1]
    //   的两组 rows (rows 0-7 / rows 8-15) 严格对齐, 故 WG1/WG2 同一
    //   wg_tid 的 partial 覆盖相同 Q 行, 可直接逐元素归并。
    // Tc==1 时 WG2 未进入循环, m=-inf/l=0/R_D=0, merge 自然退化。
    // 写完后第二次 __syncthreads 让 WG1 看到 WG2 的全部写入。
    if (consumer_id == 1) {
      // R_D: 128-bit stores
      float4 *rd_dst = merge_rd_pack + wg_tid * kRdPackPerThread;
#pragma unroll
      for (int idx = 0; idx < kRdPackPerThread; ++idx)
        rd_dst[idx] = reinterpret_cast<float4 *>(&R_D[0][0][0])[idx];
      // m/l interleaved: [m0, m1, l0, l1] per i, 128-bit store
#pragma unroll
      for (int i = 0; i < kValTileSeqLenQ; ++i) {
        float4 ml;
        ml.x = lane_block_row_max_old[i][0];
        ml.y = lane_block_row_max_old[i][1];
        ml.z = lane_block_row_sum_old[i][0];
        ml.w = lane_block_row_sum_old[i][1];
        merge_ml_pack[wg_tid * kMlPackPerThread + i] = ml;
      }
    }
    __syncthreads();

    // WG1 merges WG2's partial into its own, normalizes, and writes O
    if (consumer_id == 0) {
      // 把 WG2 的未归一化 R_D (Oacc_1) 载入 R_O 寄存器复用为临时 buffer。
      // R_D: 以 float4 (128-bit) 连续读取。
      // 后续 merge 用 R_D[i][j] 作为 Oacc_0, R_O[i][j] 作为 Oacc_1, 逐元素加权后
      // 结果写回 R_D, 再做最终 O=Oacc/l 归一化。
      float4 *rd_src = merge_rd_pack + wg_tid * kRdPackPerThread;
#pragma unroll
      for (int idx = 0; idx < kRdPackPerThread; ++idx)
        reinterpret_cast<float4 *>(&R_O[0][0][0])[idx] = rd_src[idx];

      // 稳定归并 (实现层面, 完整数学见头部注释):
      //   - m0/l0 = WG1 自己的 partial (寄存器), m1/l1 = WG2 的 partial (scratch)
      //   - [i][0] 对应 Q rows [i*16 .. i*16+7], [i][1] 对应 rows [i*16+8 .. +15]
      //   - alpha=exp(m0-m)<=1, beta=exp(m1-m)<=1: 用全局 max 做 pivot 避免 exp 溢出
      //   - F32Acc: 直接对 float fragment 逐元素加权求和
      //   - F16Acc: 把 packed half2 提升到 float2 计算, 再 pack 回 half2
      //     (避免 half 精度下 alpha/beta 加权累加的精度损失)
      //   - Tc==1: m1=-inf -> beta=exp(-inf)=0, Oacc/Oacc_1 项归零, 退化为 WG1 结果
      //   m=max(m0,m1), a=exp(m0-m), b=exp(m1-m),
      //   l=a*l0+b*l1, Oacc=a*Oacc0+b*Oacc1
#pragma unroll
      for (int i = 0; i < kValTileSeqLenQ; ++i) {
        // 128-bit load: [m1_0, m1_1, l1_0, l1_1] interleaved
        float4 ml1 = merge_ml_pack[wg_tid * kMlPackPerThread + i];
        float m1_0 = ml1.x;
        float m1_1 = ml1.y;
        float l1_0 = ml1.z;
        float l1_1 = ml1.w;

        float m0_0 = lane_block_row_max_old[i][0];
        float m0_1 = lane_block_row_max_old[i][1];
        float l0_0 = lane_block_row_sum_old[i][0];
        float l0_1 = lane_block_row_sum_old[i][1];

        float m_0 = fmaxf(m0_0, m1_0);
        float m_1 = fmaxf(m0_1, m1_1);
        float alpha_0 = __expf(m0_0 - m_0);
        float alpha_1 = __expf(m0_1 - m_1);
        float beta_0 = __expf(m1_0 - m_0);
        float beta_1 = __expf(m1_1 - m_1);

        lane_block_row_max_old[i][0] = m_0;
        lane_block_row_max_old[i][1] = m_1;
        lane_block_row_sum_old[i][0] = alpha_0 * l0_0 + beta_0 * l1_0;
        lane_block_row_sum_old[i][1] = alpha_1 * l0_1 + beta_1 * l1_1;

#pragma unroll
        for (int j = 0; j < kValTileHeadDimV; ++j) {
          if constexpr (kMmaAccF32) {
            float *d0 = reinterpret_cast<float *>(&R_D[i][j][0]);
            float *d1 = reinterpret_cast<float *>(&R_O[i][j][0]);
            d0[0] = alpha_0 * d0[0] + beta_0 * d1[0];
            d0[1] = alpha_0 * d0[1] + beta_0 * d1[1];
            d0[2] = alpha_1 * d0[2] + beta_1 * d1[2];
            d0[3] = alpha_1 * d0[3] + beta_1 * d1[3];
          } else {
            float2 d0_0 = __half22float2(HALF2(R_D[i][j][0]));
            float2 d0_1 = __half22float2(HALF2(R_D[i][j][1]));
            float2 d1_0 = __half22float2(HALF2(R_O[i][j][0]));
            float2 d1_1 = __half22float2(HALF2(R_O[i][j][1]));
            d0_0.x = alpha_0 * d0_0.x + beta_0 * d1_0.x;
            d0_0.y = alpha_0 * d0_0.y + beta_0 * d1_0.y;
            d0_1.x = alpha_1 * d0_1.x + beta_1 * d1_1.x;
            d0_1.y = alpha_1 * d0_1.y + beta_1 * d1_1.y;
            HALF2(R_D[i][j][0]) = __float22half2_rn(d0_0);
            HALF2(R_D[i][j][1]) = __float22half2_rn(d0_1);
          }
        }
      }

      // ---- Final rescale: O = Oacc / l ----
#pragma unroll
      for (int i = 0; i < kValTileSeqLenP; ++i) {
        float rescale_factor_0 = __frcp_rn(lane_block_row_sum_old[i][0]);
        float rescale_factor_1 = __frcp_rn(lane_block_row_sum_old[i][1]);
#pragma unroll
        for (int j = 0; j < kValTileHeadDimV; ++j) {
          if constexpr (kMmaAccF32) {
            float *t_fptr_D = reinterpret_cast<float *>(&R_D[i][j][0]);
            half *t_hptr_D = reinterpret_cast<half *>(&R_D[i][j][0]);
            t_hptr_D[0] = __float2half_rn(rescale_factor_0 * t_fptr_D[0]);
            t_hptr_D[1] = __float2half_rn(rescale_factor_0 * t_fptr_D[1]);
            t_hptr_D[2] = __float2half_rn(rescale_factor_1 * t_fptr_D[2]);
            t_hptr_D[3] = __float2half_rn(rescale_factor_1 * t_fptr_D[3]);
          } else {
            float2 t_reg_D_0 = __half22float2(HALF2(R_D[i][j][0]));
            float2 t_reg_D_1 = __half22float2(HALF2(R_D[i][j][1]));
            t_reg_D_0.x = rescale_factor_0 * t_reg_D_0.x;
            t_reg_D_0.y = rescale_factor_0 * t_reg_D_0.y;
            t_reg_D_1.x = rescale_factor_1 * t_reg_D_1.x;
            t_reg_D_1.y = rescale_factor_1 * t_reg_D_1.y;
            HALF2(R_D[i][j][0]) = __float22half2_rn(t_reg_D_0);
            HALF2(R_D[i][j][1]) = __float22half2_rn(t_reg_D_1);
          }
        }
      }

      // ---- Epilogue: warp shuffle + 128-bit collective store ----
#pragma unroll
      for (int i = 0; i < kValTileSeqLenP; ++i) {
#pragma unroll
        for (int j = 0; j < kValTileHeadDimV; ++j) {
          uint32_t R_Z[2][4];
          R_Z[0][0] = R_D[i][j][0];
          R_Z[1][0] = R_D[i][j][1];
          R_Z[0][1] = __shfl_sync(0xffffffff, R_D[i][j][0], lane_id + 1, 4);
          R_Z[0][2] = __shfl_sync(0xffffffff, R_D[i][j][0], lane_id + 2, 4);
          R_Z[0][3] = __shfl_sync(0xffffffff, R_D[i][j][0], lane_id + 3, 4);
          R_Z[1][1] = __shfl_sync(0xffffffff, R_D[i][j][1], lane_id + 1, 4);
          R_Z[1][2] = __shfl_sync(0xffffffff, R_D[i][j][1], lane_id + 2, 4);
          R_Z[1][3] = __shfl_sync(0xffffffff, R_D[i][j][1], lane_id + 3, 4);

          if (lane_id % 4 == 0) {
            const int store_warp_regs_O_Br =
                warp_QP * (kMmaAtomM * kValTileSeqLenP) + i * kMmaAtomM;
            const int store_lane_gmem_O_Br =
                Q_tile_id * Br + store_warp_regs_O_Br + lane_id / 4;
            const int store_warp_regs_O_d =
                warp_KV * (kMmaAtomN * kValTileHeadDimV) + j * kMmaAtomN;
            const int store_lane_gmem_O_d = store_warp_regs_O_d;
            const int store_gmem_O_addr_0 =
                O_gmem_offset + store_lane_gmem_O_Br * kHeadDim +
                store_lane_gmem_O_d;
            const int store_gmem_O_addr_1 =
                O_gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim +
                store_lane_gmem_O_d;
            *reinterpret_cast<float4 *>(&O[store_gmem_O_addr_0]) =
                *reinterpret_cast<float4 *>(&R_Z[0][0]);
            *reinterpret_cast<float4 *>(&O[store_gmem_O_addr_1]) =
                *reinterpret_cast<float4 *>(&R_Z[1][0]);
          }
        }
      }
    }
  }
}

#endif // NOTES_V2_ENABLE_TMA_MMA_WS

#if defined(NOTES_V2_ENABLE_CUTE)
namespace fa_cute {
using namespace cute;

// FlashAttn2CuTeTraits: FA2-style CuTe kernel 配置 (Br=128, Bc=64, 单 consumer WG)。
// 与 FlashAttn3CuTeTraits 的核心区别:
//   - Br=128 -> 8 warps along M (vs FA3 的 4), Q tile [128, D] (vs [64, D])
//   - 单 consumer WG 全 KV 遍历, 无 split-KV merge
//   - K/V tile 仍为 [64, D], SmemLayoutAtom/MmaAtom/SmemCopyAtom 与 FA3 一致
//     (设计原因见 FlashAttn3CuTeTraits 头注释)
//
// 设计差异详述:
// ── SmemLayout (Q 更大, K/V 不变) ──
//   SmemLayoutQ = tile_to_shape(atom, (128, D)): Q tile [128, D] (Br=128),
//     是 FA3 Q tile [64, D] 的 2 倍行数。8×8 atom 按 128 行重复, swizzle pattern 不变。
//   SmemLayoutKV = tile_to_shape(atom, (64, D)): K/V tile [64, D] (Bc=64),
//     与 FA3 的 SmemLayoutQKV 完全一致。
//   SmemLayoutVt: 同 FA3, composition 叠加 col-major (D, 64) 得 V^T 转置视图。
//
// ── MMA (8 warps, 单 WG 覆盖 Br=128) ──
//   TiledMma = TiledMMA(atom, EURepeat<8,1,1>, Tile<128,16,16>):
//     - EURepeat<8,1,1>: 8 warp 沿 M 重复 MMA atom
//       -> 8 warps × 32 threads = 256 threads = 1 consumer WG (FA3 为 128 threads)
//     - Tile<128,16,16>: 逻辑 MMA tile
//       M=128 = 8 warps × 16 rows = Br (单 WG 覆盖 Br=128, FA3 为 64)
//       N=16, K=16: 与 FA3 一致
//   MmaAtom / SmemCopyAtom / SmemCopyAtomTransposed 与 FA3 完全相同,
//   不再重复说明 (见 FlashAttn3CuTeTraits 头注释)。
// 复用 fa_cute 的 gemm_ss / gemm_rs / convert_layout_acc_* / convert_type。
template <int kHeadDim>
struct FlashAttn2CuTeTraits {
  static_assert(kHeadDim == 64 || kHeadDim == 128);

  using Element = cutlass::half_t;
  using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<Element>;
  // Q tile: [Br=128, D], K/V tile: [Bc=64, D]
  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtom{}, Shape<_128, Int<kHeadDim>>{}));
  using SmemLayoutKV = decltype(tile_to_shape(
      SmemLayoutAtom{}, Shape<_64, Int<kHeadDim>>{}));
  using SmemLayoutVt = decltype(composition(
      SmemLayoutKV{},
      make_layout(Shape<Int<kHeadDim>, _64>{}, GenRowMajor{})));

  // 8 warps along M (Br=128 = 8*16), ValTile N=2 (16 cols = 2*8)
  using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
  using TiledMma = TiledMMA<
      MmaAtom, Layout<Shape<_8, _1, _1>>,
      Tile<_128, _16, _16>>;
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
};

// Split-D keeps a single 64-wide head-dimension chunk in each shared stage.
template <int kHeadDim>
struct FFPAAttnSplitDCuTeTraits {
  static_assert(kHeadDim % 64 == 0, "Split-D requires head-dim multiple of 64");

  using Element = cutlass::half_t;
  using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<Element>;
  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtom{}, Shape<_128, _64>{}));
  using SmemLayoutKV = decltype(tile_to_shape(
      SmemLayoutAtom{}, Shape<_64, _64>{}));
  using SmemLayoutVt = decltype(composition(
      SmemLayoutKV{}, make_layout(Shape<_64, _64>{}, GenRowMajor{})));

  using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
  using TiledMma = TiledMMA<
      MmaAtom, Layout<Shape<_8, _1, _1>>, Tile<_128, _16, _16>>;
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
};

// FlashAttn3CuTeTraits: FA3-style CuTe kernel 编译期配置 (Br=Bc=64, 双 consumer WG)。
// 所有 layout/atom/tiler 都是编译期类型, kernel body 只接收实例化后的类型。
//
// ── SmemLayout 设计 (与 TMA 128B swizzle 匹配, 保证 ldmatrix 无 bank conflict) ──
//   SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<Element>:
//     - GMMA 128B swizzle atom: (8, 8) layout + Swizzle<3,4,3>
//     - 一个 atom 覆盖 8 行 × 8 half = 128B, 恰好一个 swizzle 周期
//     - 与 TMA CU_TENSOR_MAP_SWIZZLE_128B 写入 pattern 完全一致:
//       producer 用 TMA 按 128B 粒度写入 swizzle 地址, consumer 用 cute::copy
//       读取时由同款 swizzle layout 推导物理地址, 读写两侧 swizzle 一致 -> 无 bank conflict
//   SmemLayoutQKV = tile_to_shape(atom, (64, D)):
//     - Q/K/V tile 都是 [Br=Bc=64, D] (FA3 中 Br=Bc=64)
//     - tile_to_shape 把 8×8 atom 按 (64, D) 重复, 保持 swizzle pattern
//   SmemLayoutVt = composition(QKV, col-major (D, 64)):
//     - V 转置视图: P@V 时 V 作为 MMA 的 B 矩阵, mma.sync.row.col 要求
//       B 为 col-major。V 物理存储是 row-major [64, D], 通过 composition
//       叠加 col-major (D, 64) 逻辑视图, 得到 V^T[D, 64] 访问接口。
//     - composition 不搬数据, 只改 layout 解读, 物理 swizzle 不变。
//
// ── MMA 设计 (Ampere m16n8k16, f32 累加保证 softmax 精度) ──
//   MmaAtom = SM80_16x8x16_F32F16F16F32_TN:
//     - f16 输入 + f32 累加。FA 的 softmax 对精度敏感 (exp/sum),
//       f16 累加会导致 row sum 溢出/下溢, 必须用 f32。
//     - TN 布局: A(row-major) × B(col-major), 与 smem 物理布局天然匹配。
//   TiledMma = TiledMMA(atom, EURepeat<4,1,1>, Tile<64,16,16>):
//     - EURepeat<4,1,1>: 4 warp 沿 M 重复 MMA atom
//       -> 4 warps × 32 threads = 128 threads = 1 consumer WG
//     - Tile<64,16,16>: 逻辑 MMA tile
//       M=64 = 4 warps × 16 rows = Br (单 WG 覆盖 Br=64)
//       N=16 = 2 × kMmaN(8) -> 一次覆盖 16 列
//       K=16 = kMmaK (单次 MMA K slice)
//
// ── SmemCopy 设计 (S->R ldmatrix, 与 smem swizzle layout 配合) ──
//   SmemCopyAtom = SM75_U32x4_LDSM_N:
//     - 对应 ldmatrix.sync.aligned.x4.m8n8.shared.b16 (非转置)
//     - 一次加载 4 个 8×8 half 片段到 4 条 uint32 寄存器
//     - 用于 Q 和 K: Q/K 在 smem 是 row-major, ldmatrix.x4 非转置加载
//       得到 col-major fragment, 匹配 mma.row.col 的 A/B 输入约定
//   SmemCopyAtomTransposed = SM75_U16x8_LDSM_T:
//     - 对应 ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 (转置)
//     - 用于 V: V 在 smem 是 row-major [Bc, D], P@V 需把 V 当 col-major
//       B 矩阵。ldmatrix.x2.trans 转置加载为 col-major fragment, 直接匹配
//       mma.row.col 的 B 输入, 无需软件转置 V smem。
//     - 用 x2 (非 x4): V 的 K 维 (Bc=64) 按 kMmaK=16 切片, 每片 16×8
//       = 1 个 ldmatrix.x2 (2 个 8×8 fragment)
template <int kHeadDim>
struct FlashAttn3CuTeTraits {
  static_assert(kHeadDim == 64 || kHeadDim == 128);

  using Element = cutlass::half_t;
  using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<Element>;
  using SmemLayoutQKV = decltype(tile_to_shape(
      SmemLayoutAtom{}, Shape<_64, Int<kHeadDim>>{}));
  using SmemLayoutVt = decltype(composition(
      SmemLayoutQKV{},
      make_layout(Shape<Int<kHeadDim>, _64>{}, GenRowMajor{})));

  using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
  using TiledMma = TiledMMA<
      MmaAtom, Layout<Shape<_4, _1, _1>>,
      Tile<_64, _16, _16>>;
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
};

// convert_layout_acc_rowcol: 把 MMA accumulator 的 (MMA=4, MMA_M, MMA_N)
// layout 转换为 ((2, MMA_M), (2, MMA_N)) 的 (nrow, ncol) 二维布局。
//
// 为什么需要转换？FA 的 online softmax 需要按"行"操作 score 矩阵 S：
//   - 求 row max、exp(S - max)、row sum 都要求同一行的元素聚集在同一线程。
//   - 但 m16n8k16 的 MMA C fragment 的 4 个寄存器按 (MMA=4, MMA_M, MMA_N)
//     排布，lane 0/4/8/.. 持有同一行片段但散布在 MMA mode 0/1。
//   - logical_divide(_, Shape<_2>{}) 把 MMA=4 拆成 (2, 2)：
//       (2, 2, MMA_M, MMA_N) -> 重组为 ((2, MMA_M), (2, MMA_N))
//       外层 (2, MMA_M) = nrow：前 2 个 MMA mode × MMA_M 组成"行块"
//       内层 (2, MMA_N) = ncol：后 2 个 MMA mode × MMA_N 组成"列块"
//   - 转换后 scores(r, c) 的 r 索引行、c 索引列，online softmax 可直接遍历
//     size<1>(scores) 做行内 reduce max/sum，再用 __shfl_xor_sync 在 4-lane
//     子组内归约（因为 m16n8k16 下每 4 个 lane 共享同一行的不同列片段）。
//
// 出处: flash-attention/csrc/flash_attn/src/utils.h:188
//       flash-attention/csrc/flash_attn/src/softmax.h:139 (调用点)
//       用于 tCrS (QK score) 和 tCrO (PV accumulator) 的 rowcol 视图。
template <typename Layout>
CUTE_DEVICE auto convert_layout_acc_rowcol(Layout acc_layout) {
  auto divided = logical_divide(acc_layout, Shape<_2>{});
  return make_layout(
      make_layout(get<0, 1>(divided), get<1>(divided)),
      make_layout(get<0, 0>(divided), get<2>(divided)));
}

// convert_layout_acc_Aregs: 把 MMA accumulator (QK 的 score S) 转换为
// P@V 所需的 A-register (左矩阵 P) layout，使 softmax 后的 P 能直接喂给
// 下一个 MMA 做 P@V，无需额外数据搬运。
//
// 为什么需要转换？QK 和 PV 用同一个 TiledMma，但 C fragment 和 A fragment
// 的寄存器排布不同：
//   - QK: S = Q @ K^T，S 是 C fragment，layout = (MMA=4, MMA_M, MMA_N)
//   - PV: O = P @ V，P 是 A fragment，layout = (MMA, MMA_M, MMA_K)
//   - m16n8k16 的 MMA=4 对应 4 个寄存器，C 的 4 个按 (2行, 2列) 排布，
//     A 的 4 个按 (2行, 2K-slice) 排布，直接用 C layout 做 A 会导致
//     寄存器对齐错误。
//   - logical_divide(_, Shape<X, X, _2>{}) 把 MMA_N 拆成 (2, MMA_N/2)：
//       (MMA=4, MMA_M, (2, MMA_N/2)) -> 重组为 ((4, 2), MMA_M, MMA_N/2)
//       外层 (4, 2) = 新的 MMA mode，把 C 的列方向 2 个 half 配对到
//       A 的 K 方向连续 2 个元素，匹配 m16n8k16 的 K=16 内部 2-element 步长。
//   - 转换后 tCrPv 可直接作为 gemm_rs 的 tCrA 参数，P 的数据仍在原寄存器中，
//     只是逻辑坐标重解释，零拷贝。
//
// ★ 这就是 FA "寄存器复用" 的核心技巧：softmax 后的 P 不写回 smem 再
//   ldmatrix，而是原地重解释为 A fragment，省一次 smem 往返。
//   但注意这依赖 m16n8k16 的特定 fragment 约定，不是通用性质。
//
// 出处: flash-attention/csrc/flash_attn/src/utils.h:200
//       flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h:365 (调用点)
//       注释 "Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)"
template <typename TiledMma, typename Layout>
CUTE_DEVICE auto convert_layout_acc_Aregs(Layout acc_layout) {
  using X = Underscore;
  auto divided = logical_divide(acc_layout, Shape<X, X, _2>{});
  return make_layout(
      make_layout(get<0>(divided), get<2, 0>(divided)),
      get<1>(divided), get<2, 1>(divided));
}

// convert_type: 在寄存器层面做 dtype 转换, 不搬数据只改元素类型解释。
//
// 为什么需要? MMA accumulator 是 f32 (F32F16F16F32_TN), 但下游消费需要 f16:
//   - tCrP = convert_type<Element>(tCrS): softmax 后的 score S (f32) 转 P (f16),
//     供 P@V 的 A fragment (mma 要求 f16 输入)
//   - tCrO_half = convert_type<Element>(tCrO): 最终 O (f32 acc) 转 f16, 供 store
//
// 实现: NumericArrayConverter 一次性批量转换整个 fragment (kElements 个元素),
// 返回的 tensor 共享原 layout, 只是 value_type 从 From 变为 To。
// 零拷贝, 纯寄存器内转换 (make_rmem_ptr 标记为 register memory)。
//
// 出处: flash-attention/csrc/flash_attn/src/epilogue/epilogue.hpp (同类实现)
template <typename To, typename Engine, typename Layout>
CUTE_DEVICE auto convert_type(Tensor<Engine, Layout> const &tensor) {
  using From = typename Engine::value_type;
  constexpr int kElements = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To, From, kElements> convert;
  auto fragment = convert(
      *reinterpret_cast<cutlass::Array<From, kElements> const *>(tensor.data()));
  return make_tensor(make_rmem_ptr<To>(&fragment), tensor.layout());
}

// gemm_ss: Shared-Shared GEMM，A 和 B 都从 smem 加载到寄存器再做 MMA。
// 用于 FA 的 Q@K^T 步骤：Q 和 K 都在 smem 中（Q 由 TMA/cp.async 预加载，
// K 由 TMA 加载到当前 stage），需要 ldmatrix 同时搬运 A(Q) 和 B(K)。
//
// 流程 (software-pipelined ldmatrix + mma):
//   1. retile_D: 把 A/B 的 register fragment 按 TiledCopy 的 source 视图重排，
//      使 copy() 能正确写入后续 MMA 消费的寄存器位置。
//   2. 预加载 tile_k=0: copy A[0] 和 B[0] 到寄存器。
//   3. for tile_k = 0..MMA_K-1:
//        a. 若 tile_k+1 存在，预加载 A[tile_k+1] 和 B[tile_k+1]（与当前 MMA 重叠）
//        b. gemm(mma, A[tile_k], B[tile_k], acc) - 发射 mma.sync 指令
//   每个 K slice 的 ldmatrix 与前一个 slice 的 MMA 重叠执行，隐藏 smem->reg 延迟。
//
// 参数:
//   acc: (MMA, MMA_M, MMA_N) 累加器，输入输出
//   fragment_a/fragment_b: A/B 寄存器 fragment，会被 retile 后复用
//   shared_a/shared_b: smem 中的 A/B tile（带 swizzle layout）
//   tiled_mma: TiledMMA 类型实例
//   tiled_copy_a/b: A/B 的 S->R TiledCopy（make_tiled_copy_A/B 生成）
//   thread_copy_a/b: A/B 的线程级 copy slice
//
// 出处: flash-attention/csrc/flash_attn/src/utils.h:166 (gemm_rs 的对照)
//       FlashMLA/csrc/sm90/helpers.h:97 (gemm_ss 的同类实现)
//       本 notes 的 QK GEMM 调用: fa_cute::gemm_ss(tCrS, tCrQ, tCrK, ...)
template <typename TensorC, typename TensorA, typename TensorB,
          typename TensorSA, typename TensorSB, typename TiledMma,
          typename TiledCopyA, typename TiledCopyB,
          typename ThreadCopyA, typename ThreadCopyB>
CUTE_DEVICE void gemm_ss(
    TensorC &acc, TensorA &fragment_a, TensorB &fragment_b,
    TensorSA const &shared_a, TensorSB const &shared_b,
    TiledMma tiled_mma, TiledCopyA tiled_copy_a, TiledCopyB tiled_copy_b,
    ThreadCopyA thread_copy_a, ThreadCopyB thread_copy_b) {
  auto copy_view_a = thread_copy_a.retile_D(fragment_a);
  auto copy_view_b = thread_copy_b.retile_D(fragment_b);
  copy(tiled_copy_a, shared_a(_, _, _0{}), copy_view_a(_, _, _0{}));
  copy(tiled_copy_b, shared_b(_, _, _0{}), copy_view_b(_, _, _0{}));
#pragma unroll
  for (int tile_k = 0; tile_k < size<2>(fragment_a); ++tile_k) {
    if (tile_k + 1 < size<2>(fragment_a)) {
      copy(tiled_copy_a, shared_a(_, _, tile_k + 1),
           copy_view_a(_, _, tile_k + 1));
      copy(tiled_copy_b, shared_b(_, _, tile_k + 1),
           copy_view_b(_, _, tile_k + 1));
    }
    gemm(tiled_mma, fragment_a(_, _, tile_k),
         fragment_b(_, _, tile_k), acc);
  }
}

// gemm_rs: Register-Shared GEMM，A 在寄存器中（已是 fragment），B 从 smem 加载。
// 用于 FA 的 P@V 步骤：P 已经在寄存器中（softmax 后的 tCrS，经 convert_layout_acc_Aregs
// 重解释为 A fragment），V 在 smem 中（TMA 加载），只需 ldmatrix 搬运 B(V)。
//
// 与 gemm_ss 的区别: A 不从 smem 加载（P 已在寄存器），省 A 的 ldmatrix，
// 这就是 FA "寄存器复用" 的性能优势：softmax 后的 P 不回写 smem，直接做 A。
//
// 流程 (与 gemm_ss 对称，但只 pipelined B):
//   1. retile_D: B 的 register fragment 按 TiledCopy source 视图重排。
//   2. 预加载 tile_k=0: copy B[0] 到寄存器。
//   3. for tile_k = 0..MMA_K-1:
//        a. 若 tile_k+1 存在，预加载 B[tile_k+1]
//        b. gemm(mma, A[tile_k], B[tile_k], acc) - A 是 P（寄存器），B 是 V（smem）
//
// 参数:
//   acc: (MMA, MMA_M, MMA_N) 累加器，输出 O
//   fragment_a: P 寄存器 fragment（softmax 后的 score，已 Aregs 转换）
//   fragment_b: V 寄存器 fragment（会被 retile 后复用）
//   shared_b: smem 中的 V tile（带 transposed swizzle layout）
//
// 出处: flash-attention/csrc/flash_attn/src/utils.h:166
//       flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h:367 (调用点)
//       本 notes 的 PV GEMM 调用: fa_cute::gemm_rs(tCrO, tCrPv, tCrV, ...)
template <typename TensorC, typename TensorA, typename TensorB,
          typename TensorSB, typename TiledMma, typename TiledCopyB,
          typename ThreadCopyB>
CUTE_DEVICE void gemm_rs(
    TensorC &acc, TensorA &fragment_a, TensorB &fragment_b,
    TensorSB const &shared_b, TiledMma tiled_mma,
    TiledCopyB tiled_copy_b, ThreadCopyB thread_copy_b) {
  auto copy_view_b = thread_copy_b.retile_D(fragment_b);
  copy(tiled_copy_b, shared_b(_, _, _0{}), copy_view_b(_, _, _0{}));
#pragma unroll
  for (int tile_k = 0; tile_k < size<2>(fragment_a); ++tile_k) {
    if (tile_k + 1 < size<2>(fragment_a)) {
      copy(tiled_copy_b, shared_b(_, _, tile_k + 1),
           copy_view_b(_, _, tile_k + 1));
    }
    gemm(tiled_mma, fragment_a(_, _, tile_k),
         fragment_b(_, _, tile_k), acc);
  }
}

}  // namespace fa_cute
#endif // NOTES_V2_ENABLE_CUTE

#if defined(NOTES_V2_ENABLE_CUTE)
// =============================================================================
// 是 flash_attn_tma_mma_ws_split_q_cute 的简化版：去掉 TMA producer/consumer WS，
// 改用 cp.async + 统一 pipeline（参考 hgemm_mma_stages_tn_cute 的 cute::copy 用法
// + flash_attn_mma_stages_split_q 手写版本的 V/K group 管理策略）。
//
// ★ 核心设计：
//   - 只支持 kStagesK（V 固定单 buffer，无 kStagesV）
//   - V 在 QK 之前发起 cp.async，通过 QK+softmax 隐藏 V 延迟
//   - V/K 分开 commit_group，利用 cp.async group 的 LIFO 栈特性选择性 wait
//
// ★ cp.async group 管理策略（对齐手写版 L4323-4765）：
//   提交顺序（每轮 tile）：
//     1. copy V[tile] -> sV; cp_async_fence()  (group V_tile，栈底)
//     2. copy K[tile+kStagesK-1] -> sK[write]; cp_async_fence()  (group K_next，栈顶)
//   等待策略：
//     QK 前：kStagesK==1 -> wait<0>+sync（V+K 都是当前 tile，全等）
//            kStagesK>1  -> 无 wait（K[tile] 已在上一轮循环末尾 wait<0> 完成）
//     PV 前：kStagesK>1 且非最后 tile -> wait<1>+sync（栈顶 K_next 未完成，V_tile 完成）
//            否则 -> wait<0>+sync
//     循环末尾：kStagesK>1 且非最后 tile -> wait<0>+sync（确保 K_next 完成，
//              下一轮 QK 安全；同时保护 V/K smem 不被提前覆盖）
//
// ★ 同步链（参考手写版 6 个同步点）：
//   1. Q load: copy + fence + wait<0> + sync
//   2. PREFETCH K[0..Sk-2]: copy + fence，然后 wait<Sk-2> + sync
//   3. 每轮 tile:
//      3a: V[tile] fence + K_next fence (kStagesK>1) 或 K[tile] fence (kStagesK==1)
//      3b: QK 前 wait (kStagesK==1: wait<0>+sync; kStagesK>1: 无)
//          QK gemm_ss
//      3c: softmax (纯寄存器，无 sync)
//      3d: PV 前 wait (见上) + sync
//          PV gemm_rs
//      3e: rescale (纯寄存器，无 sync)
//      循环末尾: kStagesK>1 且非最后 -> wait<0> + sync
//
// 限制：仅 self-attention，无 causal/varlen/GQA（与 TMA FA2 v1 一致）
template <int kHeadDim, int kStagesK = 2>
__global__ void __launch_bounds__(256)
flash_attn_mma_stages_split_q_cute(
    cutlass::half_t *Q, cutlass::half_t *K, cutlass::half_t *V,
    cutlass::half_t *output, int rows, int seqlen) {
  // rows: B * H * seqlen, seqlen: N, kHeadDim: D
  using namespace cute;
  using Traits = fa_cute::FlashAttn2CuTeTraits<kHeadDim>;
  using Element = typename Traits::Element;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  static_assert(kHeadDim == 64 || kHeadDim == 128, "D=64 or 128 only");
  static_assert(kStagesK >= 1, "kStagesK must be >= 1");

  constexpr int kBr = 128;
  constexpr int kBc = 64;
  constexpr int kQTileElements = size(SmemLayoutQ{});
  constexpr int kKVTileElements = size(SmemLayoutKV{});

  // Shared memory: Q[128,D] + K[kStagesK,64,D] + V[1,64,D]
  // __align__(1024): swizzle phase=0, cp.async 写 GMMA swizzle layout 同样要求
  extern __shared__ __align__(1024) Element shm[];
  auto sQ = make_tensor(make_smem_ptr(shm), SmemLayoutQ{});
  // K smem 带 stage mode: (64, D, kStagesK)
  using SmemLayoutKStage = decltype(tile_to_shape(
      typename Traits::SmemLayoutAtom{},
      make_shape(_64{}, Int<kHeadDim>{}, Int<kStagesK>{})));
  auto sK = make_tensor(make_smem_ptr(shm + kQTileElements), SmemLayoutKStage{});
  // V 单 buffer: (64, D)，无 stage mode
  Element *v_base = shm + kQTileElements + kStagesK * kKVTileElements;
  auto sV = make_tensor(make_smem_ptr(v_base), SmemLayoutKV{});

  int tid = threadIdx.x;
  int q_tile = blockIdx.y * (seqlen / kBr) + blockIdx.x;
  int kv_tiles = seqlen / kBc;
  // ★ 多 head K/V offset: local_tile coord 需加 blockIdx.y * kv_tiles 偏移，
  // 否则所有 head 都读 head 0 的 K/V (参考 TMA 版 L6624 注释)
  int kv_base = blockIdx.y * kv_tiles;

  // Global memory tensors: [rows=B*H*N, D] row-major
  auto mQ = make_tensor(make_gmem_ptr(Q), make_shape(rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto mK = make_tensor(make_gmem_ptr(K), make_shape(rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto mV = make_tensor(make_gmem_ptr(V), make_shape(rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));

  auto gQ = local_tile(mQ, Shape<_128, Int<kHeadDim>>{}, make_coord(q_tile, _0{}));
  // K/V 不在此处做 local_tile (无 remainder mode)，main loop 内按 tile 做 local_tile + partition_S
  // (参考 TMA 版 producer L6624: 每次 local_tile 传入 blockIdx.y * kv_tiles + tile)

  // G2S TiledCopy: 128-bit cp.async, 256 threads, 每线程 8 half = 128b
  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_atom = Copy_Atom<Copy_Traits<g2s_copy_op>, Element>;
  using G2SCopy = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<32>{}, Int<8>{}),    // ThrLayout: 32*8=256 threads
                  make_stride(Int<8>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));  // ValLayout: 1*8=8 half=128b
  G2SCopy g2s_copy;
  auto g2s_thr = g2s_copy.get_slice(tid);

  // G2S partition: Q source(gmem) 和 destination(smem)
  auto tQgQ = g2s_thr.partition_S(gQ);
  auto tQsQ = g2s_thr.partition_D(sQ);
  // K/V 的 smem destination (source 在 main loop 内每次 partition_S)
  auto tKsK = g2s_thr.partition_D(sK);  // (CPY, CPY_M, CPY_K, kStagesK)
  auto tVsV = g2s_thr.partition_D(sV);  // (CPY, CPY_M, CPY_K)  无 stage

  // TiledMMA + S2R partition (完全复用 TMA 版 consumer 逻辑 L6696-6737)
  typename Traits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto tCrQ = thr_mma.partition_fragment_A(sQ);
  // V non-swizzle 视图用于推导寄存器 layout (不能与 swizzle composition 冲突)
  auto sVt0 = make_tensor(sV.data(), typename Traits::SmemLayoutVt{});
  auto sVt0_ns = make_tensor(
      sV.data(), get_nonswizzle_portion(typename Traits::SmemLayoutVt{}));
  auto tCrV_layout = thr_mma.partition_fragment_B(sVt0_ns).layout();
  auto tCrO = partition_fragment_C(tiled_mma, Shape<_128, Int<kHeadDim>>{});
  clear(tCrO);

  auto s2r_copy_q = make_tiled_copy_A(typename Traits::SmemCopyAtom{}, tiled_mma);
  auto s2r_thr_q = s2r_copy_q.get_thread_slice(tid);
  auto tQsQ_s2r = s2r_thr_q.partition_S(sQ);
  auto s2r_copy_k = make_tiled_copy_B(typename Traits::SmemCopyAtom{}, tiled_mma);
  auto s2r_thr_k = s2r_copy_k.get_thread_slice(tid);
  auto s2r_copy_v = make_tiled_copy_B(
      typename Traits::SmemCopyAtomTransposed{}, tiled_mma);
  auto s2r_thr_v = s2r_copy_v.get_thread_slice(tid);

  // Online softmax state
  auto tCrO_rc = make_tensor(
      tCrO.data(), fa_cute::convert_layout_acc_rowcol(tCrO.layout()));
  constexpr int kRows = decltype(size<0>(tCrO_rc))::value;
  float row_max[kRows], row_sum[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    row_max[r] = -INFINITY;
    row_sum[r] = 0.0f;
  }

  // scale 预乘 M_LOG2E: exp2f(x*scale) == expf(x/sqrt(D)), FMA 友好
  float scale = rsqrtf(static_cast<float>(kHeadDim)) * M_LOG2E;

  // ===== Step 1: Q 一次性 cp.async 加载 (split-Q 核心) =====
  cute::copy(g2s_copy, tQgQ, tQsQ);
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  // ===== Step 2: PREFETCH K 前 kStagesK-1 个 tile (仅 K，不预取 V) =====
  int ismem_read = 0;
  int ismem_write = 0;
#pragma unroll
  for (int s = 0; s < kStagesK - 1; ++s) {
    if (s < kv_tiles) {
      int kv_coord = kv_base + s;
      auto gK_tile = local_tile(mK, Shape<_64, Int<kHeadDim>>{}, make_coord(kv_coord, _0{}));
      auto tKgK_tile = g2s_thr.partition_S(gK_tile);
      cute::copy(g2s_copy, tKgK_tile, tKsK(_, _, _, s));
      cp_async_fence();
      ++ismem_write;
    }
  }
  if constexpr (kStagesK > 1) {
    cp_async_wait<kStagesK - 2>();
    __syncthreads();
  }

  // ===== Step 3: Main loop over KV tiles =====
  for (int tile = 0; tile < kv_tiles; ++tile) {
    // 3a: 提交 V[tile] (group V) + K_next (group K prefetch)
    // V 单 buffer，每轮重写 sV；V 在 QK 之前发起，通过 QK+softmax 隐藏延迟
    {
      int kv_coord = kv_base + tile;
      auto gV_tile = local_tile(mV, Shape<_64, Int<kHeadDim>>{}, make_coord(kv_coord, _0{}));
      auto tVgV_tile = g2s_thr.partition_S(gV_tile);
      cute::copy(g2s_copy, tVgV_tile, tVsV);
    }
    cp_async_fence();  // group V_tile (栈底)

    if constexpr (kStagesK > 1) {
      // Prefetch K[tile+kStagesK-1] -> sK[ismem_write] (group K_next, 栈顶)
      int k_next = tile + kStagesK - 1;
      if (k_next < kv_tiles) {
        int kv_coord = kv_base + k_next;
        auto gK_tile = local_tile(mK, Shape<_64, Int<kHeadDim>>{}, make_coord(kv_coord, _0{}));
        auto tKgK_tile = g2s_thr.partition_S(gK_tile);
        cute::copy(g2s_copy, tKgK_tile, tKsK(_, _, _, ismem_write));
        cp_async_fence();  // group K_next (栈顶)
      }
    } else {
      // kStagesK==1: 加载当前 K[tile] (smem_sel_next == smem_sel == 0)
      int kv_coord = kv_base + tile;
      auto gK_tile = local_tile(mK, Shape<_64, Int<kHeadDim>>{}, make_coord(kv_coord, _0{}));
      auto tKgK_tile = g2s_thr.partition_S(gK_tile);
      cute::copy(g2s_copy, tKgK_tile, tKsK(_, _, _, 0));
      cp_async_fence();  // group K_current (栈顶)
    }

    // 3b: QK 前 wait K[tile] 就绪
    if constexpr (kStagesK == 1) {
      // kStagesK==1: V+K 都是当前 tile，wait<0> 等全部完成
      // (kStagesK==1 是退化路径，V 延迟无法隐藏)
      cp_async_wait<0>();
      __syncthreads();
    }
    // kStagesK>1: K[tile] 已在上一轮循环末尾 wait<0> 完成，无需 wait

    // Per-stage K smem (剥离 stage mode)
    auto sK_stg = make_tensor(
        make_smem_ptr(shm + kQTileElements + ismem_read * kKVTileElements),
        SmemLayoutKV{});
    auto tCrK = thr_mma.partition_fragment_B(sK_stg);
    CUTE_STATIC_ASSERT_V(size(tCrK) == size(tCrV_layout));
    auto tCrV = make_tensor(tCrK.data(), tCrV_layout);  // K 和 V 共享寄存器存储
    auto tKsK_s2r = s2r_thr_k.partition_S(sK_stg);

    // QK GEMM: S = Q @ K^T
    auto tCrS = partition_fragment_C(tiled_mma, Shape<_128, _64>{});
    clear(tCrS);
    fa_cute::gemm_ss(tCrS, tCrQ, tCrK, tQsQ_s2r, tKsK_s2r,
                     tiled_mma, s2r_copy_q, s2r_copy_k, s2r_thr_q, s2r_thr_k);

    // 3c: Online softmax (纯寄存器操作，V[tile] 在后台加载)
    auto scores = make_tensor(
        tCrS.data(), fa_cute::convert_layout_acc_rowcol(tCrS.layout()));
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      float tile_max = -INFINITY;
#pragma unroll
      for (int c = 0; c < size<1>(scores); ++c)
        tile_max = fmaxf(tile_max, scores(r, c) * scale);
      tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
      tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
      float nxt = fmaxf(row_max[r], tile_max);
      float rs = exp2f(row_max[r] - nxt);
#pragma unroll
      for (int c = 0; c < size<1>(tCrO_rc); ++c)
        tCrO_rc(r, c) *= rs;
      float ts = 0.0f;
#pragma unroll
      for (int c = 0; c < size<1>(scores); ++c) {
        float p = exp2f(scores(r, c) * scale - nxt);
        scores(r, c) = p;
        ts += p;
      }
      ts += __shfl_xor_sync(0xffffffff, ts, 1);
      ts += __shfl_xor_sync(0xffffffff, ts, 2);
      row_sum[r] = row_sum[r] * rs + ts;
      row_max[r] = nxt;
    }

    // 3d: PV 前 wait V[tile] 就绪
    // kStagesK>1: K_next 提交条件是 tile + kStagesK - 1 < kv_tiles。
    //   已提交时栈顶有 K_next，wait<1> 只等 V_tile；
    //   未提交（尾部 Sk-1 个 tile）时栈只有 V_tile，必须 wait<0>。
    //   tile=0 必然已提交（PREFETCH 保证 kv_tiles >= kStagesK-1，且循环能跑说明 kv_tiles >= 1）。
    if constexpr (kStagesK > 1) {
      if (tile + kStagesK - 1 < kv_tiles) {
        cp_async_wait<1>();
      } else {
        cp_async_wait<0>();
      }
    }
    // kStagesK==1: 3b 已 wait<0>，V 已就绪
    __syncthreads();

    // PV GEMM: O = P @ V
    // V smem 带 swizzle: partition_S 推导 smem 源地址映射，copy() 正确应用 swizzle
    auto sVt_stg = make_tensor(sV.data(), typename Traits::SmemLayoutVt{});
    auto tVsVt_s2r = s2r_thr_v.partition_S(sVt_stg);
    auto tCrP = fa_cute::convert_type<Element>(tCrS);
    auto tCrPv = make_tensor(
        tCrP.data(),
        fa_cute::convert_layout_acc_Aregs<typename Traits::TiledMma>(
            tCrP.layout()));
    fa_cute::gemm_rs(tCrO, tCrPv, tCrV, tVsVt_s2r,
                     tiled_mma, s2r_copy_v, s2r_thr_v);
    __syncthreads();  // 确保 ldmatrix V 完成后才能覆盖 V smem

    // 3e: rescale 已合并到 3c softmax pass
    // 循环末尾: kStagesK>1 且非最后 tile -> wait<0> 确保 K_next 完成
    // (下一轮 QK 安全；同时保护 V/K smem 不被提前覆盖)
    if constexpr (kStagesK > 1) {
      if (tile + 1 < kv_tiles) {
        cp_async_wait<0>();
        __syncthreads();
      }
    }

    // 更新 pipeline 指针
    ismem_read = (ismem_read + 1) % kStagesK;
    ismem_write = (ismem_write + 1) % kStagesK;
  }

  // ===== Step 4: Final normalize O = Oacc / l =====
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    float inv_sum = 1.0f / row_sum[r];
#pragma unroll
    for (int c = 0; c < size<1>(tCrO_rc); ++c)
      tCrO_rc(r, c) *= inv_sum;
  }

  // ===== Step 5: Epilogue - direct R->G store =====
  auto tCrO_half = fa_cute::convert_type<Element>(tCrO);
  auto mO = make_tensor(make_gmem_ptr(output),
                        make_shape(rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto gO = local_tile(mO, Shape<_128, Int<kHeadDim>>{}, make_coord(q_tile, _0{}));
  auto tCgO = thr_mma.partition_C(gO);
  copy(tCrO_half, tCgO);
}
#endif // NOTES_V2_ENABLE_CUTE

#if defined(NOTES_V2_ENABLE_CUTE)
// Split-D FA forward: large head-dim (>128) handled by streaming Q/K/V in
// 64-wide head-dim chunks. O accumulators stay in per-thread registers.
//
// 与 flash_attn_tma_mma_ws_split_q_cute (标准 FA) 的唯一区别:
//   - QK: Q[Br,D]@K^T[Bc,D] 沿 D 方向分 kDChunks 次 gemm_ss 累加到单个 tCrS
//   - PV: P[Br,Bc]@V[Bc,D] 沿 D 方向分 kDChunks 次 gemm_rs, 每个 v_chunk
//         独立 O accumulator (o_accs[v]), 跨 kv_tile 累加
//   - softmax / tCrS->tCrP 寄存器复用 / online rescale: 与标准 FA 完全一致
//
// 参考: ffpa-attn CuTeDSL FFPAAttnFwdSm80SplitD (acc_Os tuple 模式)
//       ffpa-attn Triton _ffpa_fwd_kernel_impl (o_accs tuple + row_scale)
template <int kHeadDim, typename TmaQ, typename TmaK, typename TmaV,
          int kStagesQK = 2, int kStagesV = 2>
__global__ void __launch_bounds__(384, 1)
ffpa_attn_tma_mma_ws_split_d_cute(
    CUTLASS_GRID_CONSTANT TmaQ const tma_q,
    CUTLASS_GRID_CONSTANT TmaK const tma_k,
    CUTLASS_GRID_CONSTANT TmaV const tma_v,
    cutlass::half_t *output, int rows, int seqlen) {
  using namespace cute;
  using Traits = fa_cute::FFPAAttnSplitDCuTeTraits<kHeadDim>;
  using Element = typename Traits::Element;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
  using CtaBarrier = cutlass::arch::ClusterBarrier;

  static_assert(kHeadDim % 64 == 0, "Split-D requires head-dim multiple of 64");
  static_assert(kStagesQK >= 1 && kStagesV >= 1);
  constexpr int kBr = 128;
  constexpr int kBc = 64;
  constexpr int kDChunk = 64;
  constexpr int kDChunks = kHeadDim / kDChunk;
  constexpr int kProducerThreads = 128;
  constexpr int kConsumerThreads = 256;
  constexpr int kQChunkElements = cosize(SmemLayoutQ{});
  constexpr int kKVChunkElements = cosize(SmemLayoutKV{});

  extern __shared__ __align__(1024) Element shm[];
  Element *q_base = shm;
  Element *k_base = q_base + kStagesQK * kQChunkElements;
  Element *v_base = k_base + kStagesQK * kKVChunkElements;

  __shared__ uint64_t qk_full[kStagesQK];
  __shared__ uint64_t qk_empty[kStagesQK];
  __shared__ uint64_t v_full[kStagesV];
  __shared__ uint64_t v_empty[kStagesV];

  const bool is_producer = threadIdx.x < kProducerThreads;
  const int wg_tid = is_producer ? threadIdx.x : threadIdx.x - kProducerThreads;
  const int q_tile = blockIdx.y * (seqlen / kBr) + blockIdx.x;
  const int kv_tiles = seqlen / kBc;

  if (threadIdx.x == 0) {
    for (int stage = 0; stage < kStagesQK; ++stage) {
      TmaBarrier::init(&qk_full[stage], 1);
      CtaBarrier::init(&qk_empty[stage], kConsumerThreads);
    }
    for (int stage = 0; stage < kStagesV; ++stage) {
      TmaBarrier::init(&v_full[stage], 1);
      CtaBarrier::init(&v_empty[stage], kConsumerThreads);
    }
  }
  __syncthreads();

  if (is_producer) {
    NOTES_V2_REG_DEALLOC(40);
    if (wg_tid == 0) {
      auto mQ = tma_q.get_tma_tensor(make_shape(rows, Int<kHeadDim>{}));
      auto mK = tma_k.get_tma_tensor(make_shape(rows, Int<kHeadDim>{}));
      auto mV = tma_v.get_tma_tensor(make_shape(rows, Int<kHeadDim>{}));
      auto q_slice = tma_q.get_slice(_0{});
      auto k_slice = tma_k.get_slice(_0{});
      auto v_slice = tma_v.get_slice(_0{});

      for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
        for (int d_chunk = 0; d_chunk < kDChunks; ++d_chunk) {
          const int chunk_index = kv_tile * kDChunks + d_chunk;
          const int stage = chunk_index % kStagesQK;
          const int phase = (chunk_index / kStagesQK) & 1;
          CtaBarrier::wait(&qk_empty[stage], phase);
          auto sQ = make_tensor(make_smem_ptr(q_base + stage * kQChunkElements),
                                SmemLayoutQ{});
          auto sK = make_tensor(make_smem_ptr(k_base + stage * kKVChunkElements),
                                SmemLayoutKV{});
          auto gQ = local_tile(mQ, Shape<_128, _64>{},
                               make_coord(q_tile, d_chunk));
          auto gK = local_tile(mK, Shape<_64, _64>{},
                               make_coord(blockIdx.y * kv_tiles + kv_tile, d_chunk));
          auto tQgQ = q_slice.partition_S(gQ);
          auto tQsQ = q_slice.partition_D(sQ);
          auto tKgK = k_slice.partition_S(gK);
          auto tKsK = k_slice.partition_D(sK);
          TmaBarrier::arrive_and_expect_tx(
              &qk_full[stage], sizeof(Element) * (size(sQ) + size(sK)));
          copy(tma_q.with(qk_full[stage]), tQgQ, tQsQ);
          copy(tma_k.with(qk_full[stage]), tKgK, tKsK);
          tma_fence_proxy_async_shared_cta();
        }

        for (int v_chunk = 0; v_chunk < kDChunks; ++v_chunk) {
          const int chunk_index = kv_tile * kDChunks + v_chunk;
          const int stage = chunk_index % kStagesV;
          const int phase = (chunk_index / kStagesV) & 1;
          CtaBarrier::wait(&v_empty[stage], phase);
          auto sV = make_tensor(make_smem_ptr(v_base + stage * kKVChunkElements),
                                SmemLayoutKV{});
          auto gV = local_tile(mV, Shape<_64, _64>{},
                               make_coord(blockIdx.y * kv_tiles + kv_tile, v_chunk));
          auto tVgV = v_slice.partition_S(gV);
          auto tVsV = v_slice.partition_D(sV);
          TmaBarrier::arrive_and_expect_tx(&v_full[stage], sizeof(Element) * size(sV));
          copy(tma_v.with(v_full[stage]), tVgV, tVsV);
          tma_fence_proxy_async_shared_cta();
        }
      }
    }
  } else {
    NOTES_V2_REG_ALLOC(255);
    typename Traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(wg_tid);
    auto s2r_copy_q = make_tiled_copy_A(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto s2r_copy_k = make_tiled_copy_B(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto s2r_copy_v = make_tiled_copy_B(
        typename Traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto s2r_thr_q = s2r_copy_q.get_thread_slice(wg_tid);
    auto s2r_thr_k = s2r_copy_k.get_thread_slice(wg_tid);
    auto s2r_thr_v = s2r_copy_v.get_thread_slice(wg_tid);

    auto sV0 = make_tensor(make_smem_ptr(v_base), SmemLayoutKV{});
    auto sVt0_ns = make_tensor(
        sV0.data(), get_nonswizzle_portion(typename Traits::SmemLayoutVt{}));
    auto tCrV_layout = thr_mma.partition_fragment_B(sVt0_ns).layout();

    // O fragment layout + sizes: derive WITHOUT allocating a register fragment.
    // partition_fragment_C returns Tensor<ArrayEngine<float,N>,Layout> which
    // would allocate N floats on the stack (registers/local mem). We only need
    // the layout and size, so use decltype to extract types at compile time
    // without instantiating the storage.
    using OFragType = decltype(partition_fragment_C(tiled_mma, Shape<_128, _64>{}));
    using OFragLayout = typename OFragType::layout_type;
    constexpr int kOElemsPerFrag = decltype(size(OFragType{}))::value;
    constexpr int kORows = decltype(size<0>(make_tensor(
        (float*)nullptr, fa_cute::convert_layout_acc_rowcol(OFragLayout{}))))::value;
    constexpr int kOCols = decltype(size<1>(make_tensor(
        (float*)nullptr, fa_cute::convert_layout_acc_rowcol(OFragLayout{}))))::value;

    float row_max[kORows];
    float row_sum[kORows];
#pragma unroll
    for (int r = 0; r < kORows; ++r) {
      row_max[r] = -INFINITY;
      row_sum[r] = 0.0f;
    }
    const float scale = rsqrtf(static_cast<float>(kHeadDim)) * M_LOG2E;

    // Per-v_chunk register O accumulators (CuTeDSL acc_Os tuple pattern).
    //
    // 为什么需要 kDChunks 份独立 O 累加器 (而非像 tCrS 那样单份累加)?
    //   - QK: S = Q @ K^T, reduce 维度是 D (head_dim)。
    //     d_chunk 循环沿 D 切分, 每次 gemm_ss 的结果累加到同一个 tCrS (acc +=)。
    //     所以 QK 只需单份 tCrS, 循环内做 reduce 累加。
    //   - PV: O = P @ V, reduce 维度是 Bc (KV seqlen, 即 P 的列 / V 的行),
    //     而 d_chunk 切分的是 V 的列维度 (head_dim), 即 O 的输出列维度。
    //     每次 gemm_rs 产生 O 的一个 [Br, d_chunk] 子块, 不同 v_chunk 的 O 子块
    //     覆盖 head_dim 的不同区间, 之间不存在 reduce 关系, 不能累加到同一寄存器。
    //     因此需要 kDChunks 份独立 O 累加器, 跨 kv_tile 做在线 softmax rescale 累加。
    //
    // make_rmem_ptr lets gemm_rs write directly into these registers.
    float o_acc_storage[kDChunks][kOElemsPerFrag];
#pragma unroll
    for (int v = 0; v < kDChunks; ++v)
#pragma unroll
      for (int i = 0; i < kOElemsPerFrag; ++i)
        o_acc_storage[v][i] = 0.0f;

    auto mO = make_tensor(make_gmem_ptr(output),
                          make_shape(rows, Int<kHeadDim>{}),
                          make_stride(Int<kHeadDim>{}, _1{}));

    // Init: mark all QK/V slots as empty (256 consumer arrivals each)
    for (int s = 0; s < kStagesQK; ++s)
      CtaBarrier::arrive(&qk_empty[s]);
    for (int s = 0; s < kStagesV; ++s)
      CtaBarrier::arrive(&v_empty[s]);

    for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
      // ===== Phase 1: QK with Split-D (累加到单个 tCrS) =====
      auto tCrS = partition_fragment_C(tiled_mma, Shape<_128, _64>{});
      clear(tCrS);
      for (int d_chunk = 0; d_chunk < kDChunks; ++d_chunk) {
        const int chunk_index = kv_tile * kDChunks + d_chunk;
        const int stage = chunk_index % kStagesQK;
        const int phase = (chunk_index / kStagesQK) & 1;
        TmaBarrier::wait(&qk_full[stage], phase);
        tma_fence_proxy_async_shared_cta();
        auto sQ = make_tensor(make_smem_ptr(q_base + stage * kQChunkElements),
                              SmemLayoutQ{});
        auto sK = make_tensor(make_smem_ptr(k_base + stage * kKVChunkElements),
                              SmemLayoutKV{});
        auto tCrQ = thr_mma.partition_fragment_A(sQ);
        auto tCrK = thr_mma.partition_fragment_B(sK);
        auto tQsQ = s2r_thr_q.partition_S(sQ);
        auto tKsK = s2r_thr_k.partition_S(sK);
        fa_cute::gemm_ss(tCrS, tCrQ, tCrK, tQsQ, tKsK,
                         tiled_mma, s2r_copy_q, s2r_copy_k,
                         s2r_thr_q, s2r_thr_k);
        CtaBarrier::arrive(&qk_empty[stage]);
      }

      // ===== Phase 2: Online softmax (和标准 FA 完全一致) =====
      auto scores = make_tensor(
          tCrS.data(), fa_cute::convert_layout_acc_rowcol(tCrS.layout()));
      float row_scale[kORows];
#pragma unroll
      for (int row = 0; row < kORows; ++row) {
        float tile_max = -INFINITY;
#pragma unroll
        for (int col = 0; col < size<1>(scores); ++col)
          tile_max = fmaxf(tile_max, scores(row, col) * scale);
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
        const float next_max = fmaxf(row_max[row], tile_max);
        row_scale[row] = exp2f(row_max[row] - next_max);
        float tile_sum = 0.0f;
#pragma unroll
        for (int col = 0; col < size<1>(scores); ++col) {
          const float p = exp2f(scores(row, col) * scale - next_max);
          scores(row, col) = p;
          tile_sum += p;
        }
        tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 1);
        tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 2);
        row_sum[row] = row_sum[row] * row_scale[row] + tile_sum;
        row_max[row] = next_max;
      }

      // ===== tCrS -> tCrP -> tCrPv 寄存器复用 (和标准 FA 完全一致) =====
      auto tCrP = fa_cute::convert_type<Element>(tCrS);
      auto tCrPv = make_tensor(
          tCrP.data(),
          fa_cute::convert_layout_acc_Aregs<typename Traits::TiledMma>(
              tCrP.layout()));

      // ===== Phase 3: PV with Split-D (每个 v_chunk 独立 O 累加) =====
      for (int v_chunk = 0; v_chunk < kDChunks; ++v_chunk) {
        const int chunk_index = kv_tile * kDChunks + v_chunk;
        const int stage = chunk_index % kStagesV;
        const int phase = (chunk_index / kStagesV) & 1;
        TmaBarrier::wait(&v_full[stage], phase);
        tma_fence_proxy_async_shared_cta();
        auto sV = make_tensor(make_smem_ptr(v_base + stage * kKVChunkElements),
                              SmemLayoutKV{});
        auto sVt = make_tensor(sV.data(), typename Traits::SmemLayoutVt{});
        auto tCrVStorage = thr_mma.partition_fragment_B(sV);
        auto tCrV = make_tensor(tCrVStorage.data(), tCrV_layout);
        auto tVsVt = s2r_thr_v.partition_S(sVt);

        // O accumulator: view directly into persistent register storage.
        // make_rmem_ptr lets gemm_rs use it as acc without memcpy.
        auto tCrO = make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]),
                                OFragLayout{});

        // Rescale old O by row_scale (skip first kv_tile, O is zero)
        if (kv_tile > 0) {
          auto tCrO_rc = make_tensor(
              tCrO.data(), fa_cute::convert_layout_acc_rowcol(tCrO.layout()));
#pragma unroll
          for (int row = 0; row < kORows; ++row)
#pragma unroll
            for (int col = 0; col < kOCols; ++col)
              tCrO_rc(row, col) *= row_scale[row];
        }
        fa_cute::gemm_rs(tCrO, tCrPv, tCrV, tVsVt,
                         tiled_mma, s2r_copy_v, s2r_thr_v);
        CtaBarrier::arrive(&v_empty[stage]);
      }
    }

    // ===== Phase 4: Final normalize + store (遍历 kDChunks) =====
#pragma unroll
    for (int v_chunk = 0; v_chunk < kDChunks; ++v_chunk) {
      auto tCrO = make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]),
                              OFragLayout{});
      auto tCrO_rc = make_tensor(
          tCrO.data(), fa_cute::convert_layout_acc_rowcol(tCrO.layout()));
#pragma unroll
      for (int row = 0; row < kORows; ++row) {
        const float inv_sum = 1.0f / row_sum[row];
#pragma unroll
        for (int col = 0; col < kOCols; ++col)
          tCrO_rc(row, col) *= inv_sum;
      }
      auto tCrOHalf = fa_cute::convert_type<Element>(tCrO);
      auto gO = local_tile(mO, Shape<_128, _64>{}, make_coord(q_tile, v_chunk));
      auto tCgO = thr_mma.partition_C(gO);
      copy(tCrOHalf, tCgO);
    }
  }
}

#endif // NOTES_V2_ENABLE_CUTE

#if defined(NOTES_V2_ENABLE_CUTE)
template <int kHeadDim, typename TmaQ, typename TmaK, typename TmaV,
          int kStagesK = 1, int kStagesV = 1>
__global__ void __launch_bounds__(384, 1)
flash_attn_tma_mma_ws_split_q_cute(
    CUTLASS_GRID_CONSTANT TmaQ const tma_q,
    CUTLASS_GRID_CONSTANT TmaK const tma_k,
    CUTLASS_GRID_CONSTANT TmaV const tma_v,
    cutlass::half_t *output, int rows, int seqlen) {
  // rows: B * H * seqlen, seqlen: N, kHeadDim: D    
  using namespace cute;
  using Traits = fa_cute::FlashAttn2CuTeTraits<kHeadDim>;
  using Element = typename Traits::Element;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
  using CtaBarrier = cutlass::arch::ClusterBarrier;
  constexpr int kBr = 128;
  constexpr int kBc = 64;
  constexpr int kConsumerThreads = 256;
  constexpr int kProducerThreads = 128;
  constexpr int kQTileElements = cosize(SmemLayoutQ{});
  constexpr int kKVTileElements = cosize(SmemLayoutKV{});

  extern __shared__ __align__(1024) Element shm[];
  auto sQ = make_tensor(make_smem_ptr(shm), SmemLayoutQ{});
  Element *k_base = shm + kQTileElements;
  Element *v_base = k_base + kStagesK * kKVTileElements;

  __shared__ uint64_t q_full;
  __shared__ uint64_t k_full[kStagesK];
  __shared__ uint64_t k_empty[kStagesK];
  __shared__ uint64_t v_full[kStagesV];
  __shared__ uint64_t v_empty[kStagesV];

  const bool is_producer = threadIdx.x < kProducerThreads;
  const int wg_tid = is_producer ? threadIdx.x
      : threadIdx.x - kProducerThreads;

  if (threadIdx.x == 0) {
    TmaBarrier::init(&q_full, 1);
    for (int s = 0; s < kStagesK; ++s) {
      TmaBarrier::init(&k_full[s], 1);
      CtaBarrier::init(&k_empty[s], kConsumerThreads);
    }
    for (int s = 0; s < kStagesV; ++s) {
      TmaBarrier::init(&v_full[s], 1);
      CtaBarrier::init(&v_empty[s], kConsumerThreads);
    }
  }
  __syncthreads();

  int q_tile = blockIdx.y * (seqlen / kBr) + blockIdx.x;
  int kv_tiles = seqlen / kBc;

  // ==================================================================
  // Producer Warpgroup (WG0, threadIdx.x 0~127)
  // Only wg_tid == 0 issues TMA. Load Q once, then V-first K-after KV loop.
  // P1: prefetch K[0..Sk-2] + V[0..Sv-2] (if Sk/Sv > 1).
  // P2: V[tile+Sv-1] then K[tile+Sk-1] (prefetch ahead Sk-1/Sv-1, 对齐 8b).
  // ==================================================================
  if (is_producer) {
    if (wg_tid == 0) {
      auto mQ = tma_q.get_tma_tensor(make_shape(rows, Int<kHeadDim>{}));
      auto gQ = local_tile(
          mQ, Shape<_128, Int<kHeadDim>>{}, make_coord(q_tile, _0{}));
      auto q_slice = tma_q.get_slice(_0{});
      auto tQgQ = q_slice.partition_S(gQ);
      auto tQsQ = q_slice.partition_D(sQ);
      TmaBarrier::arrive_and_expect_tx(
          &q_full, sizeof(Element) * size(sQ));
      copy(tma_q.with(q_full), tQgQ, tQsQ);

      auto mK = tma_k.get_tma_tensor(make_shape(rows, Int<kHeadDim>{}));
      auto mV = tma_v.get_tma_tensor(make_shape(rows, Int<kHeadDim>{}));
      auto k_slice = tma_k.get_slice(_0{});
      auto v_slice = tma_v.get_slice(_0{});

      // ★ 易错点 (CuTe TMA multi-head): K/V local_tile 的 coord 是相对于 TMA
      // tensor row 维的全局 tile 索引 (TMA tensor shape = [B*H*N, D])，不是
      // per-head tile 索引。多 head 场景必须加 `blockIdx.y * kv_tiles` 偏移，
      // 否则所有 head 都读 head 0 的 K/V -> correctness test (H=1) 正常但
      // bench (H>1) 误差飙升 2 个量纲。Q 的 q_tile 已含 blockIdx.y 无需修复。
      // 参考: /memories/repo/leetcuda-tma-fa-multihead.md
      // P1: Prefetch first (kStagesK-1) K tiles
      for (int s = 0; s < kStagesK - 1; ++s) {
        if (s < kv_tiles) {
          auto sK = make_tensor(
              make_smem_ptr(k_base + s * kKVTileElements), SmemLayoutKV{});
          CtaBarrier::wait(&k_empty[s], 0);
          auto gK = local_tile(
              mK, Shape<_64, Int<kHeadDim>>{},
              make_coord(blockIdx.y * kv_tiles + s, _0{}));
          auto tKgK = k_slice.partition_S(gK);
          auto tKsK = k_slice.partition_D(sK);
          TmaBarrier::arrive_and_expect_tx(
              &k_full[s], sizeof(Element) * size(sK));
          copy(tma_k.with(k_full[s]), tKgK, tKsK);
        }
      }

      // P1b: Prefetch first (kStagesV-1) V tiles
      for (int s = 0; s < kStagesV - 1; ++s) {
        if (s < kv_tiles) {
          auto sV = make_tensor(
              make_smem_ptr(v_base + s * kKVTileElements), SmemLayoutKV{});
          CtaBarrier::wait(&v_empty[s], 0);
          auto gV = local_tile(
              mV, Shape<_64, Int<kHeadDim>>{},
              make_coord(blockIdx.y * kv_tiles + s, _0{}));
          auto tVgV = v_slice.partition_S(gV);
          auto tVsV = v_slice.partition_D(sV);
          TmaBarrier::arrive_and_expect_tx(
              &v_full[s], sizeof(Element) * size(sV));
          copy(tma_v.with(v_full[s]), tVgV, tVsV);
        }
      }

      // P2: Main loop - V-first then K-after (对齐 8b producer 顺序)
      for (int tile = 0; tile < kv_tiles; ++tile) {
        // V: load V[tile+Sv-1] (prefetch ahead Sv-1)
        {
          int v_tile = tile + kStagesV - 1;
          if (v_tile < kv_tiles) {
            int stage_v = v_tile % kStagesV;
            int phase_v = (v_tile / kStagesV) & 1;
            auto sV = make_tensor(
                make_smem_ptr(v_base + stage_v * kKVTileElements),
                SmemLayoutKV{});
            CtaBarrier::wait(&v_empty[stage_v], phase_v);
            auto gV = local_tile(
                mV, Shape<_64, Int<kHeadDim>>{},
                make_coord(blockIdx.y * kv_tiles + v_tile, _0{}));
            auto tVgV = v_slice.partition_S(gV);
            auto tVsV = v_slice.partition_D(sV);
            TmaBarrier::arrive_and_expect_tx(
                &v_full[stage_v], sizeof(Element) * size(sV));
            copy(tma_v.with(v_full[stage_v]), tVgV, tVsV);
          }
        }
        // K: prefetch K[tile+Sk-1] (prefetch ahead Sk-1)
        {
          int k_tile = tile + kStagesK - 1;
          if (k_tile < kv_tiles) {
            int stage_k = k_tile % kStagesK;
            int phase_k = (k_tile / kStagesK) & 1;
            auto sK = make_tensor(
                make_smem_ptr(k_base + stage_k * kKVTileElements),
                SmemLayoutKV{});
            CtaBarrier::wait(&k_empty[stage_k], phase_k);
            auto gK = local_tile(
                mK, Shape<_64, Int<kHeadDim>>{},
                make_coord(blockIdx.y * kv_tiles + k_tile, _0{}));
            auto tKgK = k_slice.partition_S(gK);
            auto tKsK = k_slice.partition_D(sK);
            TmaBarrier::arrive_and_expect_tx(
                &k_full[stage_k], sizeof(Element) * size(sK));
            copy(tma_k.with(k_full[stage_k]), tKgK, tKsK);
          }
        }
      }
    }
  }
  // ==================================================================
  // Consumer Warpgroup (WG1, threadIdx.x 128~383, 256 threads = 8 warps)
  // Single consumer: full KV traversal, no split-KV merge.
  // Flow: wait K -> QK -> release K -> softmax -> wait V -> PV -> release V.
  // ==================================================================
  else {
    TmaBarrier::wait(&q_full, 0);

    // V layout 从 stage 0 推导 (所有 stage layout 相同)
    auto sV0 = make_tensor(make_smem_ptr(v_base), SmemLayoutKV{});
    auto sVt0 = make_tensor(sV0.data(), typename Traits::SmemLayoutVt{});
    // ★ get_nonswizzle_portion 只用于 partition_fragment_B 推导寄存器侧 B
    // fragment layout (每线程持有哪些 V 元素)，寄存器布局与 smem swizzle 无关。
    // V 的 smem bank conflict 由 TMA 写入 (SmemLayoutKV 带 swizzle) +
    // ldmatrix 读取 (主循环内 partition_S(sVt_stg) 带 swizzle) 共同解决。
    // 这里不能用带 swizzle 的 sVt0 做 partition_fragment_B，否则 CuTe 推导
    // ldmatrix.trans 的线程-数据映射时会与 swizzle composition 冲突。
    // 参考: flash-attention/csrc/flash_attn/src/kernel_traits.h
    //       SmemLayoutVtransposedNoSwizzle 的同样用法。
    auto sVt0_ns = make_tensor(
        sV0.data(), get_nonswizzle_portion(typename Traits::SmemLayoutVt{}));

    typename Traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(wg_tid);
    auto tCrQ = thr_mma.partition_fragment_A(sQ);
    auto tCrV_layout = thr_mma.partition_fragment_B(sVt0_ns).layout();
    auto tCrO = partition_fragment_C(
        tiled_mma, Shape<_128, Int<kHeadDim>>{});
    clear(tCrO);

    auto s2r_copy_q = make_tiled_copy_A(
        typename Traits::SmemCopyAtom{}, tiled_mma);
    auto s2r_thr_q = s2r_copy_q.get_thread_slice(wg_tid);
    auto tQsQ_s2r = s2r_thr_q.partition_S(sQ);
    auto s2r_copy_k = make_tiled_copy_B(
        typename Traits::SmemCopyAtom{}, tiled_mma);
    auto s2r_thr_k = s2r_copy_k.get_thread_slice(wg_tid);
    auto s2r_copy_v = make_tiled_copy_B(
        typename Traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto s2r_thr_v = s2r_copy_v.get_thread_slice(wg_tid);

    auto tCrO_rc = make_tensor(
        tCrO.data(),
        fa_cute::convert_layout_acc_rowcol(tCrO.layout()));
    constexpr int kRows = decltype(size<0>(tCrO_rc))::value;
    float row_max[kRows], row_sum[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      row_max[r] = -INFINITY;
      row_sum[r] = 0.0f;
    }

    // Init: mark all K/V slots as empty (256 consumer arrivals each)
    for (int s = 0; s < kStagesK; ++s)
      CtaBarrier::arrive(&k_empty[s]);
    for (int s = 0; s < kStagesV; ++s)
      CtaBarrier::arrive(&v_empty[s]);

    // scale 预乘 M_LOG2E: exp2f(x*scale) == expf(x/sqrt(D)), FMA 友好
    float scale = rsqrtf(static_cast<float>(kHeadDim)) * M_LOG2E;
    for (int tile = 0; tile < kv_tiles; ++tile) {
      int k_stg = tile % kStagesK;
      int k_phase = (tile / kStagesK) & 1;
      int v_stg = tile % kStagesV;
      int v_phase = (tile / kStagesV) & 1;

      // Per-stage K smem and S2R partition
      auto sK_stg = make_tensor(
          make_smem_ptr(k_base + k_stg * kKVTileElements), SmemLayoutKV{});
      auto tCrK = thr_mma.partition_fragment_B(sK_stg);
      CUTE_STATIC_ASSERT_V(size(tCrK) == size(tCrV_layout));
      auto tCrV = make_tensor(tCrK.data(), tCrV_layout);
      auto tKsK_s2r = s2r_thr_k.partition_S(sK_stg);

      // 1) Wait K, QK gemm, release K early
      TmaBarrier::wait(&k_full[k_stg], k_phase);
      auto tCrS = partition_fragment_C(tiled_mma, Shape<_128, _64>{});
      clear(tCrS);
      fa_cute::gemm_ss(
          tCrS, tCrQ, tCrK, tQsQ_s2r, tKsK_s2r,
          tiled_mma, s2r_copy_q, s2r_copy_k, s2r_thr_q, s2r_thr_k);
      { CtaBarrier::arrive(&k_empty[k_stg]); }

      // 2) Online softmax (overlaps with producer's V TMA)
      auto scores = make_tensor(
          tCrS.data(),
          fa_cute::convert_layout_acc_rowcol(tCrS.layout()));
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        float tile_max = -INFINITY;
#pragma unroll
        for (int c = 0; c < size<1>(scores); ++c)
          tile_max = fmaxf(tile_max, scores(r, c) * scale);
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
        float nxt = fmaxf(row_max[r], tile_max);
        float rs = exp2f(row_max[r] - nxt);
#pragma unroll
        for (int c = 0; c < size<1>(tCrO_rc); ++c)
          tCrO_rc(r, c) *= rs;
        float ts = 0.0f;
#pragma unroll
        for (int c = 0; c < size<1>(scores); ++c) {
          float p = exp2f(scores(r, c) * scale - nxt);
          scores(r, c) = p;
          ts += p;
        }
        ts += __shfl_xor_sync(0xffffffff, ts, 1);
        ts += __shfl_xor_sync(0xffffffff, ts, 2);
        row_sum[r] = row_sum[r] * rs + ts;
        row_max[r] = nxt;
      }

      // 3) Wait V, PV gemm, release V early
      // ★ sVt_stg 带 swizzle: partition_S 推导 smem 源地址映射，copy() 会
      // 正确应用 swizzle 物理地址，与 TMA 写入 (SmemLayoutKV 带 swizzle) 匹配，
      // 解决 V 的 smem bank conflict。这是 V bank-conflict-free 的关键路径。
      auto sV_stg = make_tensor(
          make_smem_ptr(v_base + v_stg * kKVTileElements), SmemLayoutKV{});
      auto sVt_stg = make_tensor(sV_stg.data(), typename Traits::SmemLayoutVt{});
      auto tVsVt_s2r = s2r_thr_v.partition_S(sVt_stg);

      TmaBarrier::wait(&v_full[v_stg], v_phase);
      auto tCrP = fa_cute::convert_type<Element>(tCrS);
      auto tCrPv = make_tensor(
          tCrP.data(),
          fa_cute::convert_layout_acc_Aregs<typename Traits::TiledMma>(
              tCrP.layout()));
      fa_cute::gemm_rs(
          tCrO, tCrPv, tCrV, tVsVt_s2r,
          tiled_mma, s2r_copy_v, s2r_thr_v);
      { CtaBarrier::arrive(&v_empty[v_stg]); }
    }

    // ---- Final normalize: O = Oacc / l (single consumer, no merge) ----
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      float inv_sum = 1.0f / row_sum[r];
#pragma unroll
      for (int c = 0; c < size<1>(tCrO_rc); ++c)
        tCrO_rc(r, c) *= inv_sum;
    }

    auto tCrO_half = fa_cute::convert_type<Element>(tCrO);
    auto mO = make_tensor(
        make_gmem_ptr(output),
        make_shape(rows, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, _1{}));
    auto gO = local_tile(
        mO, Shape<_128, Int<kHeadDim>>{}, make_coord(q_tile, _0{}));
    auto tCgO = thr_mma.partition_C(gO);
    copy(tCrO_half, tCgO);
  }
}
#endif // NOTES_V2_ENABLE_CUTE

#if defined(NOTES_V2_ENABLE_CUTE)
template <int kHeadDim, typename TmaQ, typename TmaK, typename TmaV,
          int kStagesK = 1>
__global__ void __launch_bounds__(384, 1)
flash_attn_3_tma_mma_ws_split_q_cute(
    CUTLASS_GRID_CONSTANT TmaQ const tma_q,
    CUTLASS_GRID_CONSTANT TmaK const tma_k,
    CUTLASS_GRID_CONSTANT TmaV const tma_v,
    cutlass::half_t *output, int rows, int seqlen) {
  // rows: B * H * seqlen, seqlen: N, kHeadDim: D    
  using namespace cute;
  using Traits = fa_cute::FlashAttn3CuTeTraits<kHeadDim>;
  using Element = typename Traits::Element;
  using SmemLayout = typename Traits::SmemLayoutQKV;
  using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
  using CtaBarrier = cutlass::arch::ClusterBarrier;
  constexpr int kTile = 64;
  constexpr int kNumConsumers = 2;
  constexpr int kConsumerThreads = 128;
  constexpr int kProducerThreads = 128;
  constexpr int kTileElements = cosize(SmemLayout{});

  extern __shared__ __align__(1024) Element shm[];
  auto sQ = make_tensor(make_smem_ptr(shm), SmemLayout{});
  Element *k_base = shm + kTileElements;
  Element *v_base = k_base + kNumConsumers * kStagesK * kTileElements;

  __shared__ uint64_t q_full;
  __shared__ uint64_t k_full[kNumConsumers][kStagesK];
  __shared__ uint64_t k_empty[kNumConsumers][kStagesK];
  __shared__ uint64_t v_full[kNumConsumers];
  __shared__ uint64_t v_empty[kNumConsumers];

  const bool is_producer = threadIdx.x < kProducerThreads;
  const int consumer_id = is_producer ? 0
      : (threadIdx.x - kProducerThreads) / kConsumerThreads;
  const int wg_tid = is_producer ? threadIdx.x
      : (threadIdx.x - kProducerThreads) % kConsumerThreads;

  if (threadIdx.x == 0) {
    TmaBarrier::init(&q_full, 1);
    for (int cid = 0; cid < kNumConsumers; ++cid) {
      for (int s = 0; s < kStagesK; ++s) {
        TmaBarrier::init(&k_full[cid][s], 1);
        CtaBarrier::init(&k_empty[cid][s], kConsumerThreads);
      }
      TmaBarrier::init(&v_full[cid], 1);
      CtaBarrier::init(&v_empty[cid], kConsumerThreads);
    }
  }
  __syncthreads();

  int q_tile = blockIdx.y * (seqlen / kTile) + blockIdx.x;
  int kv_tiles = seqlen / kTile;

  // ==================================================================
  // Producer Warpgroup (WG0, threadIdx.x 0~127)
  // Only wg_tid == 0 issues TMA. Load Q once, then K-first KV loop.
  // Sk=1: load K[tile] + V[tile] per iteration.
  // Sk>=2: warmup prefetch K[cid][0], then V[tile] + prefetch K[tile+2].
  // ==================================================================
  if (is_producer) {
    if (wg_tid == 0) {
      auto mQ = tma_q.get_tma_tensor(make_shape(rows, Int<kHeadDim>{}));
      auto gQ = local_tile(
          mQ, Shape<_64, Int<kHeadDim>>{}, make_coord(q_tile, _0{}));
      auto q_slice = tma_q.get_slice(_0{});
      auto tQgQ = q_slice.partition_S(gQ);
      auto tQsQ = q_slice.partition_D(sQ);
      TmaBarrier::arrive_and_expect_tx(
          &q_full, sizeof(Element) * size(sQ));
      copy(tma_q.with(q_full), tQgQ, tQsQ);

      auto mK = tma_k.get_tma_tensor(make_shape(rows, Int<kHeadDim>{}));
      auto mV = tma_v.get_tma_tensor(make_shape(rows, Int<kHeadDim>{}));
      auto k_slice = tma_k.get_slice(_0{});
      auto v_slice = tma_v.get_slice(_0{});

      // Warmup: prefetch first K tile per consumer (Sk>=2 only)
      if constexpr (kStagesK > 1) {
        for (int cid = 0; cid < kNumConsumers; ++cid) {
          if (cid < kv_tiles) {
            auto sK = make_tensor(
                make_smem_ptr(k_base + cid * kStagesK * kTileElements),
                SmemLayout{});
            CtaBarrier::wait(&k_empty[cid][0], 0);
            auto gK = local_tile(
                mK, Shape<_64, Int<kHeadDim>>{},
                make_coord(blockIdx.y * kv_tiles + cid, _0{}));
            auto tKgK = k_slice.partition_S(gK);
            auto tKsK = k_slice.partition_D(sK);
            TmaBarrier::arrive_and_expect_tx(
                &k_full[cid][0], sizeof(Element) * size(sK));
            copy(tma_k.with(k_full[cid][0]), tKgK, tKsK);
          }
        }
      }

      for (int tile = 0; tile < kv_tiles; ++tile) {
        int cid = tile & 1;
        int local = (tile - cid) / kNumConsumers;
        int kv_tile = blockIdx.y * kv_tiles + tile;

        // Step 1: K — load current (Sk=1) or prefetch next (Sk>=2)
        if constexpr (kStagesK == 1) {
          auto sK = make_tensor(
              make_smem_ptr(k_base + cid * kStagesK * kTileElements),
              SmemLayout{});
          CtaBarrier::wait(&k_empty[cid][0], local & 1);
          auto gK = local_tile(
              mK, Shape<_64, Int<kHeadDim>>{}, make_coord(kv_tile, _0{}));
          auto tKgK = k_slice.partition_S(gK);
          auto tKsK = k_slice.partition_D(sK);
          TmaBarrier::arrive_and_expect_tx(
              &k_full[cid][0], sizeof(Element) * size(sK));
          copy(tma_k.with(k_full[cid][0]), tKgK, tKsK);
        } else {
          int stg_next = (local + 1) % kStagesK;
          int next_tile = tile + kNumConsumers;
          if (next_tile < kv_tiles) {
            int next_local = local + 1;
            int next_phase = (next_local / kStagesK) & 1;
            auto sK = make_tensor(
                make_smem_ptr(k_base +
                    (cid * kStagesK + stg_next) * kTileElements),
                SmemLayout{});
            CtaBarrier::wait(&k_empty[cid][stg_next], next_phase);
            auto gK = local_tile(
                mK, Shape<_64, Int<kHeadDim>>{},
                make_coord(blockIdx.y * kv_tiles + next_tile, _0{}));
            auto tKgK = k_slice.partition_S(gK);
            auto tKsK = k_slice.partition_D(sK);
            TmaBarrier::arrive_and_expect_tx(
                &k_full[cid][stg_next], sizeof(Element) * size(sK));
            copy(tma_k.with(k_full[cid][stg_next]), tKgK, tKsK);
          }
        }

        // Step 2: V — single buffer, release depends on PV (late step).
        // V TMA latency is hidden by consumer QK+softmax compute.
        auto sV = make_tensor(
            make_smem_ptr(v_base + cid * kTileElements), SmemLayout{});
        CtaBarrier::wait(&v_empty[cid], local & 1);
        auto gV = local_tile(
            mV, Shape<_64, Int<kHeadDim>>{}, make_coord(kv_tile, _0{}));
        auto tVgV = v_slice.partition_S(gV);
        auto tVsV = v_slice.partition_D(sV);
        TmaBarrier::arrive_and_expect_tx(
            &v_full[cid], sizeof(Element) * size(sV));
        copy(tma_v.with(v_full[cid]), tVgV, tVsV);
      }
    }
  }
  // ==================================================================
  // Consumer Warpgroups (WG1 consumer_id=0, WG2 consumer_id=1)
  // Each WG processes every other KV tile independently.
  // K staged (kStagesK depth), V single buffer (always 1).
  // Flow: wait K → QK → release K → softmax → wait V → PV → release V → rescale.
  // ==================================================================
  else {
    TmaBarrier::wait(&q_full, 0);

    auto sV = make_tensor(
        make_smem_ptr(v_base + consumer_id * kTileElements), SmemLayout{});
    auto sVt = make_tensor(sV.data(), typename Traits::SmemLayoutVt{});
    auto sVt_ns = make_tensor(
        sV.data(), get_nonswizzle_portion(typename Traits::SmemLayoutVt{}));

    typename Traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(wg_tid);
    auto tCrQ = thr_mma.partition_fragment_A(sQ);
    auto tCrV_layout = thr_mma.partition_fragment_B(sVt_ns).layout();
    auto tCrO = partition_fragment_C(
        tiled_mma, Shape<_64, Int<kHeadDim>>{});
    clear(tCrO);

    auto s2r_copy_q = make_tiled_copy_A(
        typename Traits::SmemCopyAtom{}, tiled_mma);
    auto s2r_thr_q = s2r_copy_q.get_thread_slice(wg_tid);
    auto tQsQ_s2r = s2r_thr_q.partition_S(sQ);
    auto s2r_copy_k = make_tiled_copy_B(
        typename Traits::SmemCopyAtom{}, tiled_mma);
    auto s2r_thr_k = s2r_copy_k.get_thread_slice(wg_tid);
    auto s2r_copy_v = make_tiled_copy_B(
        typename Traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto s2r_thr_v = s2r_copy_v.get_thread_slice(wg_tid);
    auto tVsVt_s2r = s2r_thr_v.partition_S(sVt);

    auto tCrO_rc = make_tensor(
        tCrO.data(),
        fa_cute::convert_layout_acc_rowcol(tCrO.layout()));
    constexpr int kRows = decltype(size<0>(tCrO_rc))::value;
    float row_max[kRows], row_sum[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      row_max[r] = -INFINITY;
      row_sum[r] = 0.0f;
    }

    // Init: mark own K/V slots as empty
    for (int s = 0; s < kStagesK; ++s)
      CtaBarrier::arrive(&k_empty[consumer_id][s]);
    CtaBarrier::arrive(&v_empty[consumer_id]);

    // scale 预乘 M_LOG2E: exp2f(x*scale) == expf(x/sqrt(D)), FMA 友好
    float scale = rsqrtf(static_cast<float>(kHeadDim)) * M_LOG2E;
    int local_iter = 0;
    for (int tile = consumer_id; tile < kv_tiles;
         tile += kNumConsumers, ++local_iter) {
      int k_stg = local_iter % kStagesK;
      int k_phase = (local_iter / kStagesK) & 1;

      // Per-stage K smem and S2R partition
      auto sK_stg = make_tensor(
          make_smem_ptr(k_base +
              (consumer_id * kStagesK + k_stg) * kTileElements),
          SmemLayout{});
      auto tCrK = thr_mma.partition_fragment_B(sK_stg);
      CUTE_STATIC_ASSERT_V(size(tCrK) == size(tCrV_layout));
      auto tCrV = make_tensor(tCrK.data(), tCrV_layout);
      auto tKsK_s2r = s2r_thr_k.partition_S(sK_stg);

      // 1) Wait K, QK gemm, release K early
      TmaBarrier::wait(&k_full[consumer_id][k_stg], k_phase);
      auto tCrS = partition_fragment_C(tiled_mma, Shape<_64, _64>{});
      clear(tCrS);
      fa_cute::gemm_ss(
          tCrS, tCrQ, tCrK, tQsQ_s2r, tKsK_s2r,
          tiled_mma, s2r_copy_q, s2r_copy_k, s2r_thr_q, s2r_thr_k);
      { CtaBarrier::arrive(&k_empty[consumer_id][k_stg]); }

      // 2) Online softmax (overlaps with producer's V TMA)
      auto scores = make_tensor(
          tCrS.data(),
          fa_cute::convert_layout_acc_rowcol(tCrS.layout()));
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        float tile_max = -INFINITY;
#pragma unroll
        for (int c = 0; c < size<1>(scores); ++c)
          tile_max = fmaxf(tile_max, scores(r, c) * scale);
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
        float nxt = fmaxf(row_max[r], tile_max);
        float rs = exp2f(row_max[r] - nxt);
#pragma unroll
        for (int c = 0; c < size<1>(tCrO_rc); ++c)
          tCrO_rc(r, c) *= rs;
        float ts = 0.0f;
#pragma unroll
        for (int c = 0; c < size<1>(scores); ++c) {
          float p = exp2f(scores(r, c) * scale - nxt);
          scores(r, c) = p;
          ts += p;
        }
        ts += __shfl_xor_sync(0xffffffff, ts, 1);
        ts += __shfl_xor_sync(0xffffffff, ts, 2);
        row_sum[r] = row_sum[r] * rs + ts;
        row_max[r] = nxt;
      }

      // 3) Wait V, PV gemm, release V early
      TmaBarrier::wait(&v_full[consumer_id], local_iter & 1);
      auto tCrP = fa_cute::convert_type<Element>(tCrS);
      auto tCrPv = make_tensor(
          tCrP.data(),
          fa_cute::convert_layout_acc_Aregs<typename Traits::TiledMma>(
              tCrP.layout()));
      fa_cute::gemm_rs(
          tCrO, tCrPv, tCrV, tVsVt_s2r,
          tiled_mma, s2r_copy_v, s2r_thr_v);
      { CtaBarrier::arrive(&v_empty[consumer_id]); }
    }

    // Split-KV merge
    __syncthreads();
    float *merge_o = reinterpret_cast<float *>(shm);
    float4 *merge_stats = reinterpret_cast<float4 *>(
        merge_o + 128 * size(tCrO));

    if (consumer_id == 1) {
#pragma unroll
      for (int idx = 0; idx < size(tCrO); ++idx)
        merge_o[idx * 128 + wg_tid] = tCrO(idx);
      merge_stats[wg_tid] = make_float4(
          row_max[0], row_max[1], row_sum[0], row_sum[1]);
    }
    __syncthreads();

    if (consumer_id == 0) {
      float4 st = merge_stats[wg_tid];
      float o_max[kRows] = {st.x, st.y};
      float o_sum[kRows] = {st.z, st.w};
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        float mg = fmaxf(row_max[r], o_max[r]);
        float lhs = exp2f(row_max[r] - mg);
        float rhs = exp2f(o_max[r] - mg);
        float ms = lhs * row_sum[r] + rhs * o_sum[r];
#pragma unroll
        for (int c = 0; c < size<1>(tCrO_rc); ++c) {
          int idx = tCrO_rc.layout()(make_coord(r, c));
          float ov = merge_o[idx * 128 + wg_tid];
          tCrO_rc(r, c) = (lhs * tCrO_rc(r, c) + rhs * ov) / ms;
        }
      }

      auto tCrO_half = fa_cute::convert_type<Element>(tCrO);
      auto mO = make_tensor(
          make_gmem_ptr(output),
          make_shape(rows, Int<kHeadDim>{}),
          make_stride(Int<kHeadDim>{}, _1{}));
      auto gO = local_tile(
          mO, Shape<_64, Int<kHeadDim>>{}, make_coord(q_tile, _0{}));
      auto tCgO = thr_mma.partition_C(gO);
      copy(tCrO_half, tCgO);
    }
  }
}
#endif // NOTES_V2_ENABLE_CUTE

#if defined(NOTES_V2_ENABLE_CUTE)
template <int kHeadDim, typename TmaQ>
__global__ void flash_attn_3_cute_tma_copy_smoke(
  CUTLASS_GRID_CONSTANT TmaQ const tma_q,
  cutlass::half_t *output, int rows) {
  using namespace cute;
  using Traits = fa_cute::FlashAttn3CuTeTraits<kHeadDim>;
  using Element = typename Traits::Element;
  using SmemLayout = typename Traits::SmemLayoutQKV;
  using TransactionBarrier = cutlass::arch::ClusterTransactionBarrier;

  extern __shared__ __align__(1024) Element shared_storage[];
  auto sQ = make_tensor(make_smem_ptr(shared_storage), SmemLayout{});
  __shared__ uint64_t tma_barrier;

  if (threadIdx.x == 0) {
    TransactionBarrier::init(&tma_barrier, 1);
  }
  __syncthreads();

  auto mQ = tma_q.get_tma_tensor(make_shape(rows, Int<kHeadDim>{}));
  auto gQ = local_tile(
      mQ, Shape<_64, Int<kHeadDim>>{},
      make_coord(blockIdx.x, _0{}));
  auto cta_tma_q = tma_q.get_slice(_0{});
  auto tQgQ = cta_tma_q.partition_S(gQ);
  auto tQsQ = cta_tma_q.partition_D(sQ);

  if (threadIdx.x == 0) {
    TransactionBarrier::arrive_and_expect_tx(
        &tma_barrier, sizeof(half) * size(sQ));
    copy(tma_q.with(tma_barrier), tQgQ, tQsQ);
  }
  TransactionBarrier::wait(&tma_barrier, 0);

  for (int linear = threadIdx.x; linear < size(sQ); linear += blockDim.x) {
    int row = linear / kHeadDim;
    int col = linear % kHeadDim;
    output[(blockIdx.x * 64 + row) * kHeadDim + col] = sQ(row, col);
  }
}
#endif // NOTES_V2_ENABLE_CUTE


