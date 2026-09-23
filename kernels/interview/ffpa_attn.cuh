#pragma once
#include "flash_attn.cuh"
// ffpa_attn.cuh: FFPA Attention kernel for large head-dim (Split-D)
//
// 通过 include flash_attn.cuh 复用 fa_cute namespace 中的 FA traits 和 helpers。
// 仅定义 FFPA 特有的 FFPAAttnSplitDCuTeTraits 和 ffpa_attn_tma_mma_ws_split_d_cute kernel。
// 支持 head_dim > 128 的 large head-dim attention，通过 64-wide Split-D chunks 处理。
//
// 优化点（Tile=64x64 + QK/PV 分离 TiledMma）：
//   - QK: Tile<64,64,16> + Layout<4,1,1> → EURepeat<1,8,1>
//     一次 TiledMMA 覆盖完整 S[64,64]，省掉 N-tile 循环，提升计算密度
//   - PV: Tile<64,16,16> + Layout<4,1,1> → EURepeat<1,2,1>
//     保持小 tile 控制 acc_O 寄存器
//   - 128 producer + 128 consumer = 256 total threads
//
// 性能 (B=1,H=32,N=8192,D=512, SM120a RTX PRO 5000):
//   cuDNN SDPA: 57.2 TFLOPS | FFPA TMA WS: 110.7 TFLOPS (1.94x)


// =============================================================================
// FFPA Split-D Tiled: tile=64x64 优化版本
// =============================================================================
// 核心优化点：
//   - QK: Tile<64,64,16> + Layout<4,1,1> → EURepeat<1,8,1>
//     一次 TiledMMA 覆盖完整 S[64,64]，省掉 N-tile 循环，提升计算密度
//   - PV: Tile<64,16,16> + Layout<4,1,1> → EURepeat<1,2,1>
//     保持小 tile 控制 acc_O 寄存器
//   - 128 producer + 128 consumer = 256 total threads (vs 原始 384)

#if defined(NOTES_V2_ENABLE_CUTE)
namespace fa_cute {
using namespace cute;

template <int kHeadDim, int TILE_M = 64, int TILE_N = 64>
struct FFPAAttnSplitDCuTeTraits {
  static_assert(kHeadDim % 64 == 0, "Split-D requires head-dim multiple of 64");
  static_assert(TILE_M == 64 && TILE_N == 64, "Current impl supports 64x64 only");

  using Element = cutlass::half_t;
  using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<Element>;
  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtom{}, Shape<Int<TILE_M>, _64>{}));
  using SmemLayoutKV = decltype(tile_to_shape(
      SmemLayoutAtom{}, Shape<Int<TILE_N>, _64>{}));
  using SmemLayoutVt = decltype(composition(
      SmemLayoutKV{}, make_layout(Shape<_64, Int<TILE_N>>{}, GenRowMajor{})));

  using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;

  // QK: Tile<64,64,16> → EURepeat<1,8,1>，一次覆盖 S[64,64]
  using TiledMmaQK = decltype(make_tiled_mma(
      MmaAtom{}, Layout<Shape<_4, _1, _1>>{},
      Tile<Int<TILE_M>, Int<TILE_N>, _16>{}));

  // PV: Tile<64,16,16> → EURepeat<1,2,1>，控制 acc_O 寄存器
  using TiledMmaPV = decltype(make_tiled_mma(
      MmaAtom{}, Layout<Shape<_4, _1, _1>>{},
      Tile<Int<TILE_M>, _16, _16>{}));

  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
};

}  // namespace fa_cute
#endif // NOTES_V2_ENABLE_CUTE

// =============================================================================
// FFPA Split-D Forward Kernel (cp.async 版本，无 TMA/WS)
// =============================================================================
// 消费者逻辑与 ffpa_attn_tma_mma_ws_split_d_cute 完全一致（双 TiledMma，
// 相同的 fragment 流转：QK->convert_layout_acc_Aregs<TiledMmaPV>->PV）。
// 唯一区别：生产者从 TMA 换成 cp.async，用 cp_async_fence/wait 替代 TMA barrier。
//
// 128 线程：与 TiledMmaQK/PV 的 Layout<4,1,1> 一致，每个线程在 G2S 和 S2R/MMA
// 中有唯一分区，消除 256-thread G2S 与 128-thread MMA 之间的映射不匹配。
//
// SMEM: sQ[kStagesQK,64,64] + sK[kStagesQK,64,64] + sV[kStagesV,64,64]
// stage 偏移通过基地址指针算术管理，不使用 stride-0 的 stage-mode layout。
#if defined(NOTES_V2_ENABLE_CUTE)
template <int kHeadDim, int kStagesQK = 2, int kStagesV = 2>
__global__ void __launch_bounds__(128)
ffpa_split_d_cute(
    cutlass::half_t *Q, cutlass::half_t *K, cutlass::half_t *V,
    cutlass::half_t *output, int rows, int seqlen) {
  using namespace cute;
  using Traits = fa_cute::FFPAAttnSplitDCuTeTraits<kHeadDim>;
  using Element = typename Traits::Element;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  static_assert(kHeadDim % 64 == 0, "Split-D requires head-dim multiple of 64");
  static_assert(kStagesQK >= 1 && kStagesV >= 1);

  constexpr int kBr = 64;
  constexpr int kBc = 64;
  constexpr int kDChunk = 64;
  constexpr int kDChunks = kHeadDim / kDChunk;
  constexpr int kQChunkElements = cosize(SmemLayoutQ{});
  constexpr int kKVChunkElements = cosize(SmemLayoutKV{});

  extern __shared__ __align__(1024) Element shm[];
  Element *q_base = shm;
  Element *k_base = q_base + kStagesQK * kQChunkElements;
  Element *v_base = k_base + kStagesQK * kKVChunkElements;

  int tid = threadIdx.x;
  int q_tile = blockIdx.y * (seqlen / kBr) + blockIdx.x;
  int kv_tiles = seqlen / kBc;
  int kv_base = blockIdx.y * kv_tiles;

  auto mQ = make_tensor(make_gmem_ptr(Q), make_shape(rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto mK = make_tensor(make_gmem_ptr(K), make_shape(rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto mV = make_tensor(make_gmem_ptr(V), make_shape(rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto mO = make_tensor(make_gmem_ptr(output), make_shape(rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));

  // G2S TiledCopy: 128-bit cp.async, 128 threads (16×8), 对齐 MMA 128 线程
  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_atom = Copy_Atom<Copy_Traits<g2s_copy_op>, Element>;
  using G2SCopy = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<16>{}, Int<8>{}),
                  make_stride(Int<8>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));
  G2SCopy g2s_copy;
  auto g2s_thr = g2s_copy.get_slice(tid);

  // 双 TiledMma: 与 TMA WS kernel 完全一致
  typename Traits::TiledMmaQK tiled_mma_qk;
  typename Traits::TiledMmaPV tiled_mma_pv;
  auto thr_mma_qk = tiled_mma_qk.get_thread_slice(tid);
  auto thr_mma_pv = tiled_mma_pv.get_thread_slice(tid);

  // V layout for gemm_rs（与 TMA WS kernel 完全一致）
  auto sV0 = make_tensor(make_smem_ptr(v_base), SmemLayoutKV{});
  auto sVt0_ns = make_tensor(
      sV0.data(), get_nonswizzle_portion(typename Traits::SmemLayoutVt{}));
  auto tCrV_layout = thr_mma_pv.partition_fragment_B(sVt0_ns).layout();

  // S2R copy atoms（与 TMA WS kernel 完全一致）
  auto s2r_copy_q = make_tiled_copy_A(typename Traits::SmemCopyAtom{}, tiled_mma_qk);
  auto s2r_thr_q = s2r_copy_q.get_thread_slice(tid);
  auto s2r_copy_k = make_tiled_copy_B(typename Traits::SmemCopyAtom{}, tiled_mma_qk);
  auto s2r_thr_k = s2r_copy_k.get_thread_slice(tid);
  auto s2r_copy_v = make_tiled_copy_B(
      typename Traits::SmemCopyAtomTransposed{}, tiled_mma_pv);
  auto s2r_thr_v = s2r_copy_v.get_thread_slice(tid);

  // O fragment layout（与 TMA WS kernel 完全一致）
  using OFragType = decltype(partition_fragment_C(tiled_mma_pv, Shape<_64, _64>{}));
  using OFragLayout = typename OFragType::layout_type;
  constexpr int kOElemsPerFrag = decltype(size(OFragType{}))::value;
  constexpr int kORows = decltype(size<0>(make_tensor(
      (float*)nullptr, fa_cute::convert_layout_acc_rowcol(OFragLayout{}))))::value;
  constexpr int kOCols = decltype(size<1>(make_tensor(
      (float*)nullptr, fa_cute::convert_layout_acc_rowcol(OFragLayout{}))))::value;

  // Online softmax persistent state（与 TMA WS kernel 完全一致）
  float row_max[kORows];
  float row_sum[kORows];
#pragma unroll
  for (int r = 0; r < kORows; ++r) {
    row_max[r] = -INFINITY;
    row_sum[r] = 0.0f;
  }
  const float scale = rsqrtf(static_cast<float>(kHeadDim)) * M_LOG2E;

  // Per-v_chunk register O accumulators（与 TMA WS kernel 完全一致）
  float o_acc_storage[kDChunks][kOElemsPerFrag];
#pragma unroll
  for (int v = 0; v < kDChunks; ++v)
#pragma unroll
    for (int i = 0; i < kOElemsPerFrag; ++i)
      o_acc_storage[v][i] = 0.0f;

  // Helper: G2S copy a Q tile to a specific stage
  auto g2s_load_q = [&](int d, int stage) {
    auto gQ = local_tile(mQ, Shape<_64, _64>{}, make_coord(q_tile, d));
    auto s_dst = make_tensor(make_smem_ptr(q_base + stage * kQChunkElements),
                             SmemLayoutQ{});
    cute::copy(g2s_copy, g2s_thr.partition_S(gQ), g2s_thr.partition_D(s_dst));
  };
  // Helper: G2S copy a K tile to a specific stage
  auto g2s_load_k = [&](int kv_idx, int d, int stage) {
    auto gK = local_tile(mK, Shape<_64, _64>{},
                         make_coord(kv_base + kv_idx, d));
    auto s_dst = make_tensor(make_smem_ptr(k_base + stage * kKVChunkElements),
                             SmemLayoutKV{});
    cute::copy(g2s_copy, g2s_thr.partition_S(gK), g2s_thr.partition_D(s_dst));
  };
  // Helper: G2S copy a V tile to a specific stage
  auto g2s_load_v = [&](int kv_idx, int d, int stage) {
    auto gV = local_tile(mV, Shape<_64, _64>{},
                         make_coord(kv_base + kv_idx, d));
    auto s_dst = make_tensor(make_smem_ptr(v_base + stage * kKVChunkElements),
                             SmemLayoutKV{});
    cute::copy(g2s_copy, g2s_thr.partition_S(gV), g2s_thr.partition_D(s_dst));
  };

  // Main loop over KV tiles
  for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
    // ===== Phase 0: V Prefetch =====
    int v_write = 0;
#pragma unroll
    for (int v = 0; v < kStagesV - 1 && v < kDChunks; ++v) {
      g2s_load_v(kv_tile, v, v_write);
      cp_async_fence();
      v_write = (v_write + 1) % kStagesV;
    }
    if constexpr (kStagesV > 1) {
      cp_async_wait<kStagesV - 2>();
      __syncthreads();
    }

    // ===== Phase 1: QK with Split-D =====
    auto tCrS = partition_fragment_C(tiled_mma_qk, Shape<_64, _64>{});
    clear(tCrS);

    int qk_read = 0;
    int qk_write = 0;

    // 初始 prefetch: 前 kStagesQK-1 个 (Q,K) pair
#pragma unroll
    for (int d = 0; d < kStagesQK - 1 && d < kDChunks; ++d) {
      g2s_load_q(d, qk_write);
      g2s_load_k(kv_tile, d, qk_write);
      cp_async_fence();
      qk_write = (qk_write + 1) % kStagesQK;
    }
    if constexpr (kStagesQK > 1) {
      cp_async_wait<kStagesQK - 2>();
      __syncthreads();
    }

    // QK main loop
    for (int d_chunk = 0; d_chunk < kDChunks; ++d_chunk) {
      int d_next = d_chunk + kStagesQK - 1;
      if (d_next < kDChunks) {
        g2s_load_q(d_next, qk_write);
        g2s_load_k(kv_tile, d_next, qk_write);
        cp_async_fence();
        if constexpr (kStagesQK > 1) {
          cp_async_wait<kStagesQK - 2>();
        } else {
          cp_async_wait<0>();
        }
        __syncthreads();
        qk_write = (qk_write + 1) % kStagesQK;
      }

      // QK GEMM: S += Q[d_chunk] @ K[d_chunk]^T
      auto sQ_stg = make_tensor(
          make_smem_ptr(q_base + qk_read * kQChunkElements), SmemLayoutQ{});
      auto sK_stg = make_tensor(
          make_smem_ptr(k_base + qk_read * kKVChunkElements), SmemLayoutKV{});
      auto tCrQ = thr_mma_qk.partition_fragment_A(sQ_stg);
      auto tCrK = thr_mma_qk.partition_fragment_B(sK_stg);
      auto tQsQ_s2r = s2r_thr_q.partition_S(sQ_stg);
      auto tKsK_s2r = s2r_thr_k.partition_S(sK_stg);
      fa_cute::gemm_ss(tCrS, tCrQ, tCrK, tQsQ_s2r, tKsK_s2r,
                       tiled_mma_qk, s2r_copy_q, s2r_copy_k,
                       s2r_thr_q, s2r_thr_k);
      __syncthreads();
      qk_read = (qk_read + 1) % kStagesQK;
    }
    if constexpr (kStagesQK > 1) {
      cp_async_wait<0>();
      __syncthreads();
    }

    // ===== Phase 2: Online softmax（与 TMA WS kernel 完全一致）=====
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

    // P fragment: convert_layout_acc_Aregs<TiledMmaPV>（与 TMA WS kernel 一致）
    auto tCrP = fa_cute::convert_type<Element>(tCrS);
    auto tCrPv = make_tensor(
        tCrP.data(),
        fa_cute::convert_layout_acc_Aregs<typename Traits::TiledMmaPV>(
            tCrP.layout()));

    // ===== Phase 3: PV with Split-D =====
    int v_write_pv = (kStagesV > 1) ? (kStagesV - 1) : 0;
    int v_read = 0;
    for (int v_chunk = 0; v_chunk < kDChunks; ++v_chunk) {
      int v_next = v_chunk + kStagesV - 1;
      if (v_next < kDChunks) {
        g2s_load_v(kv_tile, v_next, v_write_pv);
        cp_async_fence();
        if constexpr (kStagesV > 1) {
          cp_async_wait<kStagesV - 2>();
        } else {
          cp_async_wait<0>();
        }
        __syncthreads();
        v_write_pv = (v_write_pv + 1) % kStagesV;
      }

      // Rescale O accumulator（与 TMA WS kernel 完全一致）
      auto tCrO = make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]),
                              OFragLayout{});
      if (kv_tile > 0) {
        auto tCrO_rc = make_tensor(
            tCrO.data(), fa_cute::convert_layout_acc_rowcol(tCrO.layout()));
#pragma unroll
        for (int row = 0; row < kORows; ++row)
#pragma unroll
          for (int col = 0; col < kOCols; ++col)
            tCrO_rc(row, col) *= row_scale[row];
      }

      // gemm_rs: O[v_chunk] += P @ V[v_read]（与 TMA WS kernel 完全一致）
      auto sV_stg = make_tensor(
          make_smem_ptr(v_base + v_read * kKVChunkElements), SmemLayoutKV{});
      auto sVt_stg = make_tensor(sV_stg.data(), typename Traits::SmemLayoutVt{});
      auto tCrVStorage = thr_mma_pv.partition_fragment_B(sV_stg);
      auto tCrV = make_tensor(tCrVStorage.data(), tCrV_layout);
      auto tVsVt = s2r_thr_v.partition_S(sVt_stg);
      fa_cute::gemm_rs(tCrO, tCrPv, tCrV, tVsVt,
                       tiled_mma_pv, s2r_copy_v, s2r_thr_v);
      __syncthreads();
      v_read = (v_read + 1) % kStagesV;
    }
    if constexpr (kStagesV > 1) {
      cp_async_wait<0>();
    }
    __syncthreads();
  }

  // ===== Phase 4: Final normalize + store（与 TMA WS kernel 完全一致）=====
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
    auto gO = local_tile(mO, Shape<_64, _64>{}, make_coord(q_tile, v_chunk));
    auto tCgO = thr_mma_pv.partition_C(gO);
    copy(tCrOHalf, tCgO);
  }
}
#endif // NOTES_V2_ENABLE_CUTE

#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
template <int kHeadDim, typename TmaQ, typename TmaK, typename TmaV,
          int kStagesQK = 2, int kStagesV = 2>
__global__ void __launch_bounds__(256, 1)
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
  constexpr int kBr = 64;
  constexpr int kBc = 64;
  constexpr int kDChunk = 64;
  constexpr int kDChunks = kHeadDim / kDChunk;
  constexpr int kProducerThreads = 128;
  constexpr int kConsumerThreads = 128;
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
    NOTES_V2_REG_DEALLOC(32);
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
          auto gQ = local_tile(mQ, Shape<_64, _64>{},
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
    NOTES_V2_REG_ALLOC(232);
    typename Traits::TiledMmaQK tiled_mma_qk;
    typename Traits::TiledMmaPV tiled_mma_pv;
    auto thr_mma_qk = tiled_mma_qk.get_thread_slice(wg_tid);
    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(wg_tid);
    auto s2r_copy_q = make_tiled_copy_A(typename Traits::SmemCopyAtom{}, tiled_mma_qk);
    auto s2r_copy_k = make_tiled_copy_B(typename Traits::SmemCopyAtom{}, tiled_mma_qk);
    auto s2r_copy_v = make_tiled_copy_B(
        typename Traits::SmemCopyAtomTransposed{}, tiled_mma_pv);
    auto s2r_thr_q = s2r_copy_q.get_thread_slice(wg_tid);
    auto s2r_thr_k = s2r_copy_k.get_thread_slice(wg_tid);
    auto s2r_thr_v = s2r_copy_v.get_thread_slice(wg_tid);

    auto sV0 = make_tensor(make_smem_ptr(v_base), SmemLayoutKV{});
    auto sVt0_ns = make_tensor(
        sV0.data(), get_nonswizzle_portion(typename Traits::SmemLayoutVt{}));
    auto tCrV_layout = thr_mma_pv.partition_fragment_B(sVt0_ns).layout();

    using OFragType = decltype(partition_fragment_C(tiled_mma_pv, Shape<_64, _64>{}));
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

    float o_acc_storage[kDChunks][kOElemsPerFrag];
#pragma unroll
    for (int v = 0; v < kDChunks; ++v)
#pragma unroll
      for (int i = 0; i < kOElemsPerFrag; ++i)
        o_acc_storage[v][i] = 0.0f;

    auto mO = make_tensor(make_gmem_ptr(output),
                          make_shape(rows, Int<kHeadDim>{}),
                          make_stride(Int<kHeadDim>{}, _1{}));

    for (int s = 0; s < kStagesQK; ++s)
      CtaBarrier::arrive(&qk_empty[s]);
    for (int s = 0; s < kStagesV; ++s)
      CtaBarrier::arrive(&v_empty[s]);

    for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
      auto tCrS = partition_fragment_C(tiled_mma_qk, Shape<_64, _64>{});
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
        auto tCrQ = thr_mma_qk.partition_fragment_A(sQ);
        auto tCrK = thr_mma_qk.partition_fragment_B(sK);
        auto tQsQ = s2r_thr_q.partition_S(sQ);
        auto tKsK = s2r_thr_k.partition_S(sK);
        fa_cute::gemm_ss(tCrS, tCrQ, tCrK, tQsQ, tKsK,
                         tiled_mma_qk, s2r_copy_q, s2r_copy_k,
                         s2r_thr_q, s2r_thr_k);
        CtaBarrier::arrive(&qk_empty[stage]);
      }

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

      auto tCrP = fa_cute::convert_type<Element>(tCrS);
      auto tCrPv = make_tensor(
          tCrP.data(),
          fa_cute::convert_layout_acc_Aregs<typename Traits::TiledMmaPV>(
              tCrP.layout()));

      for (int v_chunk = 0; v_chunk < kDChunks; ++v_chunk) {
        const int chunk_index = kv_tile * kDChunks + v_chunk;
        const int stage = chunk_index % kStagesV;
        const int phase = (chunk_index / kStagesV) & 1;
        TmaBarrier::wait(&v_full[stage], phase);
        tma_fence_proxy_async_shared_cta();
        auto sV = make_tensor(make_smem_ptr(v_base + stage * kKVChunkElements),
                              SmemLayoutKV{});
        auto sVt = make_tensor(sV.data(), typename Traits::SmemLayoutVt{});
        auto tCrVStorage = thr_mma_pv.partition_fragment_B(sV);
        auto tCrV = make_tensor(tCrVStorage.data(), tCrV_layout);
        auto tVsVt = s2r_thr_v.partition_S(sVt);

        auto tCrO = make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]),
                                OFragLayout{});
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
                         tiled_mma_pv, s2r_copy_v, s2r_thr_v);
        CtaBarrier::arrive(&v_empty[stage]);
      }
    }

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
      auto gO = local_tile(mO, Shape<_64, _64>{}, make_coord(q_tile, v_chunk));
      auto tCgO = thr_mma_pv.partition_C(gO);
      copy(tCrOHalf, tCgO);
    }
  }
}
#endif // NOTES_V2_ENABLE_CUTE && NOTES_V2_ENABLE_TMA_MMA_WS

// =============================================================================
// FFPA Split-D non-WS TMA 版 (ffpa-attn split_d.cuh 移植, 最小教学集)
// =============================================================================
// 移植自 ffpa-attn csrc/cuffpa/cute/sm_120/split_d.cuh 的 non-WS CuTe TMA
// kernel。相对上方 WS 教学版 (64x64, 145T) 的性能要素:
//   1. tile 128x128: 同一份 K/V smem 数据服务的 Q 行翻倍, 算术强度更高
//   2. non-WS: 256 线程全员 MMA, tid=0 在消费循环内联发 TMA; WS 版一半
//      线程专职 producer 不产 FLOPs, SM 上 MMA 吞吐直接砍半
//   3. 跨 kv_tile 预取: 下一 tile 的 QK 初始 chunk 与本 tile 的 softmax+PV
//      重叠 (QK/V barrier 集不相交, 零死锁)
//   4. STSM + TMA store epilogue: O 经 stmatrix 暂存 smem 再批量 TMA 写回
//      (WS 版逐元素寄存器直写 gmem)
//   5. FA-4 conditional rescale: row_max 增长 < 8 (log2 域) 时跳过整个
//      o_acc 重缩放循环 (warp vote 保证无分支发散)
// QK 与 PV 的 D chunk 尺寸解耦 (kQKDChunk=32, kVDChunk=64): 更小的 Q/K
// chunk 让同 smem 预算装下 128 宽 kBc 与更深 stages。
//
// 最小教学集裁剪: 无 bias/dropout/GQA/NHD/causal; 保留 Nkv 边界 mask
// (kv_valid < kBc 的列置 -INF) 与 LSE 输出 (指针为 null 时跳过);
// 要求 Nq % kBr == 0, O epilogue 恒走整 tile STSM+TMA store 路径。
// 坐标系: 扁平 2D (rows_q = B*H*Nq, rows_kv = B*H*Nkv), 与 WS 版一致;
// 区别是 Q 与 KV 行数解耦, 支持 Nq != Nkv 的非对齐教学用例。
//
// 性能 (B=1,H=32,N=16384,D=320, SM120a RTX PRO 5000):
//   WS 版 145.0T | 本版 ~190T | cuDNN SDPA 70.2T
#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
namespace fa_cute {

// TMA-O epilogue 的每批 v_chunk 数: 取 kDChunksV 的最大因子使得
// [kBr, kHeadDim/n_batches] 能装进 kSmemElems, 从而 kNBatches 最小
// (TMA store bulk group 的 arrive/wait 次数最少)。
constexpr int compute_vchunks_per_batch(int kDChunksV, int kHeadDim, int kBr,
                                        int kSmemElems) {
  for (int n = 1; n <= kDChunksV; ++n) {
    if (kDChunksV % n != 0)
      continue;
    if (kBr * (kHeadDim / n) <= kSmemElems)
      return kDChunksV / n;
  }
  return 1;
}

// QK chunk=32 时 SW128 atom 的 64B 行超宽, 退到 SW64 避免 smem 浪费;
// V/O chunk=64 用 SW128。chunk=16 时 SW32 同理。
template <int kChunk, typename Element>
struct SelectSmemAtom {
  using type = GMMA::Layout_K_SW128_Atom<Element>;
};

template <typename Element>
struct SelectSmemAtom<32, Element> {
  using type = GMMA::Layout_K_SW64_Atom<Element>;
};

template <typename Element>
struct SelectSmemAtom<16, Element> {
  using type = GMMA::Layout_K_SW32_Atom<Element>;
};

template <int kHeadDim_, int kBr_ = 128, int kBc_ = 128, int kQKDChunk_ = 32,
          int kVDChunk_ = 64, int kStagesQK_ = 2, int kStagesPV_ = 2>
struct FFPAAttnNonWSCuTeSplitDTraits {
  static_assert(kHeadDim_ % kQKDChunk_ == 0);
  static_assert(kHeadDim_ % kVDChunk_ == 0);
  static_assert(kQKDChunk_ == 16 || kQKDChunk_ == 32 || kQKDChunk_ == 64);
  static_assert(kVDChunk_ == 16 || kVDChunk_ == 32 || kVDChunk_ == 64);

  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int kBr = kBr_;
  static constexpr int kBc = kBc_;
  static constexpr int kQKDChunk = kQKDChunk_;
  static constexpr int kVDChunk = kVDChunk_;
  static constexpr int kDChunksQK = kHeadDim / kQKDChunk;
  static constexpr int kDChunksV = kHeadDim / kVDChunk;
  static constexpr int kNumWarps = kBr / 16;
  static constexpr int kNumThreads = kNumWarps * 32;
  static constexpr int kStagesQK = kStagesQK_;
  static constexpr int kStagesPV = kStagesPV_;
  static constexpr int kSmemElems = kStagesQK * kBr * kQKDChunk +
                                    kStagesQK * kBc * kQKDChunk +
                                    kStagesPV * kBc * kVDChunk;
  static constexpr int kVChunksPerBatch =
      compute_vchunks_per_batch(kDChunksV, kHeadDim, kBr, kSmemElems);
  static constexpr int kNBatches = kDChunksV / kVChunksPerBatch;
  static constexpr float kRescaleThreshold = FA4_RESCALE_THRESHOLD;

  using Element = cutlass::half_t;
  using SmemAtomQK = typename SelectSmemAtom<kQKDChunk, Element>::type;
  using SmemAtomV = typename SelectSmemAtom<kVDChunk, Element>::type;
  using SmemLayoutQ =
      decltype(tile_to_shape(SmemAtomQK{}, Shape<Int<kBr>, Int<kQKDChunk>>{}));
  using SmemLayoutK =
      decltype(tile_to_shape(SmemAtomQK{}, Shape<Int<kBc>, Int<kQKDChunk>>{}));
  using SmemLayoutV =
      decltype(tile_to_shape(SmemAtomV{}, Shape<Int<kBc>, Int<kVDChunk>>{}));
  using SmemLayoutVt = decltype(composition(
      SmemLayoutV{},
      make_layout(Shape<Int<kVDChunk>, Int<kBc>>{}, GenRowMajor{})));
  // O 的 smem 暂存: 与 V 同 SW128 atom, epilogue 复用已释放的 QKV smem。
  using SmemLayoutO =
      decltype(tile_to_shape(SmemAtomV{}, Shape<Int<kBr>, Int<kVDChunk>>{}));

  using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
  // M8N1: kNumWarps 个 warp 全部沿 M 堆叠, N 不切分 -> 每 warp 持整行,
  // softmax 归约只需 warp 内 shfl_xor 4-lane, 无需跨 warp smem 归约。
  using TiledMmaQK = decltype(make_tiled_mma(
      MmaAtom{}, Layout<Shape<Int<kNumWarps>, _1, _1>>{},
      Tile<Int<kBr>, Int<kBc>, _16>{}));
  using TiledMmaPV = decltype(make_tiled_mma(
      MmaAtom{}, Layout<Shape<Int<kNumWarps>, _1, _1>>{},
      Tile<Int<kBr>, Int<kVDChunk>, _16>{}));

  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
};

// Online safe softmax (scale 乘法版 + FA-4 conditional rescaling)。
// S 是 QK 裸分数, scale 已含 log2(e) 折叠; 与 persist-D 的
// online_safe_softmax_fa4 (scale 预折叠进 Q fragment) 差异仅在
// max/exp 循环内的逐元素乘。行内 4-lane 蝴蝶归约 (m16n8k16 一行
// 散在 4 个 lane: xor 1 + xor 2)。
// FA-4 conditional rescale: log2_diff = m_old - m_new <= 0, 当
// log2_diff >= -threshold (max 增长 < 2^8) 时 row_scale=1 且 row_max
// 保持旧值 (stale max), 跳过 o_acc 重缩放; O 与 row_sum 用同一 stale
// max 累积, epilogue 的 O/row_sum 相消, 数学上等价。
template <typename ScoresTensor, int kRows>
CUTE_DEVICE void online_safe_softmax_scaled(ScoresTensor &scores, float scale,
                                            float *row_max, float *row_sum,
                                            float *row_scale,
                                            float rescale_threshold) {
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    float tile_max = -INFINITY;
#pragma unroll
    for (int col = 0; col < cute::size<1>(scores); ++col)
      tile_max = fmaxf(tile_max, scores(row, col) * scale);
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
    const float next_max = fmaxf(row_max[row], tile_max);
    const float log2_diff = row_max[row] - next_max;
    float eff_max = next_max;
    if (log2_diff >= -rescale_threshold) {
      row_scale[row] = 1.0f;
      eff_max = row_max[row];  // stale max; row_max 不更新
    } else {
      row_scale[row] = exp2f(log2_diff);
      row_max[row] = next_max;
    }
    float tile_sum = 0.0f;
#pragma unroll
    for (int col = 0; col < cute::size<1>(scores); ++col) {
      const float p = exp2f(scores(row, col) * scale - eff_max);
      scores(row, col) = p;
      tile_sum += p;
    }
    tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 1);
    tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 2);
    row_sum[row] = row_sum[row] * row_scale[row] + tile_sum;
  }
}

}  // namespace fa_cute

// 算法与上方 WS 版相同 (QK split-D 累加 -> online softmax -> PV split-D);
// 差异全在执行协议: 流水深度/barrier 结构/TMA 发射时机/epilogue 通道。
// TMA pipeline (non-WS): tid=0 在消费循环内联发 TMA, 全部 256 线程做 MMA。
//   qk_full (TmaBarrier, init=1): tid=0 arrive_and_expect_tx, 消费者 wait
//   qk_empty (CtaBarrier, init=kNumThreads): 每个消费线程 arrive 一次,
//     凑满后 tid=0 才能覆写该 stage
//   v_full/v_empty 与 QK barrier 集完全独立 (跨 tile 预取零死锁的关键)
//   phase = (chunk_index / kStages) & 1, chunk_index = kv_tile*kDChunks+d
template <typename Traits, typename TmaQ, typename TmaK, typename TmaV,
          typename TmaO>
__global__ void __launch_bounds__(Traits::kNumThreads, 1)
ffpa_attn_tma_split_d_cute(
    CUTLASS_GRID_CONSTANT TmaQ const tma_q,
    CUTLASS_GRID_CONSTANT TmaK const tma_k,
    CUTLASS_GRID_CONSTANT TmaV const tma_v,
    CUTLASS_GRID_CONSTANT TmaO const tma_o,
    typename Traits::Element *output, float *softmax_lse,
    int rows_q, int rows_kv, int nq, int nkv) {
  // Body-level guard: TMA/stmatrix 需 sm>=90; 混合 -gencode 构建时低 arch
  // 的 device pass 把 body 编成空 stub (kernel 声明必须对所有 arch 可见)。
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using namespace cute;
  using cute::tma_store_arrive;
  using cute::tma_store_wait;
  using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
  using CtaBarrier = cutlass::arch::ClusterBarrier;
  using Element = typename Traits::Element;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using SmemLayoutVt = typename Traits::SmemLayoutVt;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using TiledMmaQK = typename Traits::TiledMmaQK;
  using TiledMmaPV = typename Traits::TiledMmaPV;
  using SmemCopyAtom = typename Traits::SmemCopyAtom;
  using SmemCopyAtomTransposed = typename Traits::SmemCopyAtomTransposed;

  constexpr int kBr = Traits::kBr;
  constexpr int kBc = Traits::kBc;
  constexpr int kQKDChunk = Traits::kQKDChunk;
  constexpr int kVDChunk = Traits::kVDChunk;
  constexpr int kHeadDim = Traits::kHeadDim;
  constexpr int kDChunksQK = Traits::kDChunksQK;
  constexpr int kDChunksV = Traits::kDChunksV;
  constexpr int kNumThreads = Traits::kNumThreads;
  constexpr int kStagesQK = Traits::kStagesQK;
  constexpr int kStagesPV = Traits::kStagesPV;

  constexpr int kQChunkElements = cosize(SmemLayoutQ{});
  constexpr int kKChunkElements = cosize(SmemLayoutK{});
  constexpr int kVChunkElements = cosize(SmemLayoutV{});

  // TMA-O epilogue 复用 QKV smem 作 O 暂存, 保证整批装得下。
  static_assert(Traits::kVChunksPerBatch * cosize(SmemLayoutO{}) <=
                    Traits::kSmemElems,
                "TMA-O: batched O staging must fit in reused QKV smem");

  const int q_tiles = nq / kBr;
  const int kv_tiles = (nkv + kBc - 1) / kBc;
  const int q_tile = blockIdx.y * q_tiles + blockIdx.x;
  const int Br_base = blockIdx.x * kBr;
  const int tid = threadIdx.x;

  // 扁平 2D 坐标: 行号直接编码 (b, h, row)。Q/O 段起点 h*nq 恒 128 对齐
  // (host 保证 nq % kBr == 0), 可用全局 tile 索引; K/V 段起点 h*nkv 未必
  // 对齐 (Nkv 非 kBc 倍数), 必须 domain_offset 偏到 head 段再用段内 tile
  // 索引, 否则 tile 行号会串到相邻 head 的行。
  auto mQ = tma_q.get_tma_tensor(make_shape(rows_q, Int<kHeadDim>{}));
  auto mK = domain_offset(make_coord(blockIdx.y * nkv, 0),
                          tma_k.get_tma_tensor(
                              make_shape(rows_kv, Int<kHeadDim>{})));
  auto mV = domain_offset(make_coord(blockIdx.y * nkv, 0),
                          tma_v.get_tma_tensor(
                              make_shape(rows_kv, Int<kHeadDim>{})));

  // SMEM: [q_base | k_base | v_base], 各自 kStages 份。
  extern __shared__ __align__(1024) Element shm[];
  Element *q_base = shm;
  Element *k_base = q_base + kStagesQK * kQChunkElements;
  Element *v_base = k_base + kStagesQK * kKChunkElements;

  __shared__ uint64_t qk_full[kStagesQK];
  __shared__ uint64_t qk_empty[kStagesQK];
  __shared__ uint64_t v_full[kStagesPV];
  __shared__ uint64_t v_empty[kStagesPV];

  if (tid == 0) {
    for (int s = 0; s < kStagesQK; ++s) {
      TmaBarrier::init(&qk_full[s], 1);
      CtaBarrier::init(&qk_empty[s], kNumThreads);
    }
    for (int s = 0; s < kStagesPV; ++s) {
      TmaBarrier::init(&v_full[s], 1);
      CtaBarrier::init(&v_empty[s], kNumThreads);
    }
  }
  __syncthreads();

  auto q_slice = tma_q.get_slice(_0{});
  auto k_slice = tma_k.get_slice(_0{});
  auto v_slice = tma_v.get_slice(_0{});

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma_qk = tiled_mma_qk.get_thread_slice(tid);
  auto thr_mma_pv = tiled_mma_pv.get_thread_slice(tid);

  auto s2r_copy_q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma_qk);
  auto s2r_copy_k = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma_qk);
  auto s2r_copy_v = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma_pv);
  auto s2r_thr_q = s2r_copy_q.get_thread_slice(tid);
  auto s2r_thr_k = s2r_copy_k.get_thread_slice(tid);
  auto s2r_thr_v = s2r_copy_v.get_thread_slice(tid);

  // V fragment 寄存器布局: 在 nonswizzle 视图上 partition_fragment_B 只取
  // 线程<->数据映射 (寄存器布局与 swizzle 无关), LDSM_T 实际读仍走
  // swizzled 的 sVt, bank 冲突由 TMA 写入侧 swizzle 消解。
  // Ref: flash-attention kernel_traits.h SmemLayoutVtransposedNoSwizzle。
  auto sV0 = make_tensor(make_smem_ptr(v_base), SmemLayoutV{});
  auto sVt0_ns =
      make_tensor(sV0.data(), get_nonswizzle_portion(SmemLayoutVt{}));
  auto tCrV_layout = thr_mma_pv.partition_fragment_B(sVt0_ns).layout();

  using OFragType = decltype(partition_fragment_C(
      tiled_mma_pv, Shape<Int<kBr>, Int<kVDChunk>>{}));
  using OFragLayout = typename OFragType::layout_type;
  constexpr int kOElemsPerFrag = decltype(size(OFragType{}))::value;
  constexpr int kORows = decltype(size<0>(make_tensor(
      (float *)nullptr,
      fa_cute::convert_layout_acc_rowcol(OFragLayout{}))))::value;
  constexpr int kOCols = decltype(size<1>(make_tensor(
      (float *)nullptr,
      fa_cute::convert_layout_acc_rowcol(OFragLayout{}))))::value;

  auto cS = make_identity_tensor(Shape<Int<kBr>, Int<kBc>>{});
  auto tScS = thr_mma_qk.partition_C(cS);
  auto tScS_rc = make_tensor(
      tScS.data(), fa_cute::convert_layout_acc_rowcol(tScS.layout()));
  constexpr int kSRows = decltype(size<0>(tScS_rc))::value;

  // exp2 softmax: scale 折叠 log2(e), exp(x) == exp2(x * log2e)。
  const float scale = rsqrtf(static_cast<float>(kHeadDim)) * M_LOG2E;

  float row_max[kORows];
  float row_sum[kORows];
#pragma unroll
  for (int r = 0; r < kORows; ++r) {
    row_max[r] = -INFINITY;
    row_sum[r] = 0.0f;
  }

  // 跨全部 kv_tile 常驻的 O 累加器: rescale 需要所有 kDChunksV 切片
  // 每 tile 在位, 不能提前流出。D=320/kVDChunk=64 -> 5*32 = 160 regs/thread,
  // 这是 single-pass online softmax 的结构性寄存器成本。
  float o_acc_storage[kDChunksV][kOElemsPerFrag];
#pragma unroll
  for (int v = 0; v < kDChunksV; ++v)
#pragma unroll
    for (int i = 0; i < kOElemsPerFrag; ++i)
      o_acc_storage[v][i] = 0.0f;

  // 全线程各 arrive 一次, 凑满 empty barrier 的 init count, 通知 tid=0
  // 所有 stage 初始为空, 可以发首批 TMA。
  for (int s = 0; s < kStagesQK; ++s)
    CtaBarrier::arrive(&qk_empty[s]);
  for (int s = 0; s < kStagesPV; ++s)
    CtaBarrier::arrive(&v_empty[s]);

  // TMA 发射 lambda (仅 tid=0 调用): arrive_and_expect_tx 声明本 stage
  // 期望收到的字节数, TMA 写完自动翻转 full barrier。
  auto issue_qk_tma = [&](int d_chunk, int stage, int kv_tile_idx) {
    cutlass::arch::fence_view_async_shared();
    auto sQ = make_tensor(make_smem_ptr(q_base + stage * kQChunkElements),
                          SmemLayoutQ{});
    auto sK = make_tensor(make_smem_ptr(k_base + stage * kKChunkElements),
                          SmemLayoutK{});
    auto gQ = local_tile(mQ, Shape<Int<kBr>, Int<kQKDChunk>>{},
                         make_coord(q_tile, d_chunk));
    auto gK = local_tile(mK, Shape<Int<kBc>, Int<kQKDChunk>>{},
                         make_coord(kv_tile_idx, d_chunk));
    auto tQgQ = q_slice.partition_S(gQ);
    auto tQsQ = q_slice.partition_D(sQ);
    auto tKgK = k_slice.partition_S(gK);
    auto tKsK = k_slice.partition_D(sK);
    TmaBarrier::arrive_and_expect_tx(&qk_full[stage],
                                     sizeof(Element) * (size(sQ) + size(sK)));
    copy(tma_q.with(qk_full[stage]), tQgQ, tQsQ);
    copy(tma_k.with(qk_full[stage]), tKgK, tKsK);
  };

  auto issue_v_tma = [&](int v_chunk, int stage, int kv_tile_idx) {
    cutlass::arch::fence_view_async_shared();
    auto sV = make_tensor(make_smem_ptr(v_base + stage * kVChunkElements),
                          SmemLayoutV{});
    auto gV = local_tile(mV, Shape<Int<kBc>, Int<kVDChunk>>{},
                         make_coord(kv_tile_idx, v_chunk));
    auto tVgV = v_slice.partition_S(gV);
    auto tVsV = v_slice.partition_D(sV);
    TmaBarrier::arrive_and_expect_tx(&v_full[stage],
                                     sizeof(Element) * size(sV));
    copy(tma_v.with(v_full[stage]), tVgV, tVsV);
  };

  // 初始 QK 预取: 填满前 kStagesQK 个 chunk。
  if (tid == 0) {
    for (int d = 0; d < kStagesQK && d < kDChunksQK; ++d) {
      CtaBarrier::wait(&qk_empty[d], 0);
      issue_qk_tma(d, d, 0);
    }
  }

  // 初始 V 预取 (kv_tile 0): V 与 QK 无数据依赖, 在 QK GEMM 前发出,
  // 让 V TMA 重叠整个首个 QK GEMM + softmax 窗口。
  if (tid == 0) {
    for (int v = 0; v < kStagesPV && v < kDChunksV; ++v) {
      const int chunk_index = v;
      const int v_stage = chunk_index % kStagesPV;
      const int v_phase = (chunk_index / kStagesPV) & 1;
      CtaBarrier::wait(&v_empty[v_stage], v_phase);
      issue_v_tma(v, v_stage, 0);
    }
  }

#pragma unroll 1
  for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
    // V 预取 (kv_tile > 0): 本 tile 的首批 V chunks 在 QK 循环前发出,
    // 重叠 QK GEMM + softmax。
    if (kv_tile > 0 && tid == 0) {
      for (int v = 0; v < kStagesPV && v < kDChunksV; ++v) {
        const int chunk_index = kv_tile * kDChunksV + v;
        const int v_stage = chunk_index % kStagesPV;
        const int v_phase = (chunk_index / kStagesPV) & 1;
        CtaBarrier::wait(&v_empty[v_stage], v_phase);
        issue_v_tma(v, v_stage, kv_tile);
      }
    }

    // Phase 1: QK split-D 累加, S[Br,Bc] = sum_d Q_d @ K_d^T。
    auto tCrS = partition_fragment_C(tiled_mma_qk, Shape<Int<kBr>, Int<kBc>>{});
    clear(tCrS);

#pragma unroll
    for (int d_chunk = 0; d_chunk < kDChunksQK; ++d_chunk) {
      const int chunk_index = kv_tile * kDChunksQK + d_chunk;
      const int stage = chunk_index % kStagesQK;
      const int phase = (chunk_index / kStagesQK) & 1;
      TmaBarrier::wait(&qk_full[stage], phase);
      cutlass::arch::fence_view_async_shared();

      auto sQ = make_tensor(make_smem_ptr(q_base + stage * kQChunkElements),
                            SmemLayoutQ{});
      auto sK = make_tensor(make_smem_ptr(k_base + stage * kKChunkElements),
                            SmemLayoutK{});
      auto tCrQ = thr_mma_qk.partition_fragment_A(sQ);
      auto tCrK = thr_mma_qk.partition_fragment_B(sK);
      auto tQsQ = s2r_thr_q.partition_S(sQ);
      auto tKsK = s2r_thr_k.partition_S(sK);

      fa_cute::gemm_ss(tCrS, tCrQ, tCrK, tQsQ, tKsK, tiled_mma_qk,
                       s2r_copy_q, s2r_copy_k, s2r_thr_q, s2r_thr_k);

      CtaBarrier::arrive(&qk_empty[stage]);

      // 预取不能移到 gemm_ss 之前: s_next == stage 且 phase_next == 1-phase,
      // wait 门控在本迭代的 arrive 上, 提前会死锁 (tid=0 卡在等待需要
      // tid=0 自己参与的 gemm_ss 释放的 barrier)。
      if (tid == 0) {
        const int d_next = d_chunk + kStagesQK;
        if (d_next < kDChunksQK) {
          const int next_index = kv_tile * kDChunksQK + d_next;
          const int s_next = next_index % kStagesQK;
          const int phase_next = (next_index / kStagesQK) & 1;
          CtaBarrier::wait(&qk_empty[s_next], phase_next);
          issue_qk_tma(d_next, s_next, kv_tile);
        }
      }
    }

    // 跨 tile 预取: 下一 kv_tile 的 QK 初始 chunks 在本 tile 的 softmax+PV
    // 之前发出, TMA 与计算重叠。QK barrier 集与 softmax/PV 用的 v_*
    // barrier 不相交, 放这里零死锁。
    if (kv_tile < kv_tiles - 1 && tid == 0) {
      for (int d = 0; d < kStagesQK && d < kDChunksQK; ++d) {
        const int chunk_index = (kv_tile + 1) * kDChunksQK + d;
        const int stage = chunk_index % kStagesQK;
        const int phase = (chunk_index / kStagesQK) & 1;
        CtaBarrier::wait(&qk_empty[stage], phase);
        issue_qk_tma(d, stage, kv_tile + 1);
      }
    }

    // Phase 2: online softmax + P fragment 准备。
    {
      auto scores = make_tensor(
          tCrS.data(), fa_cute::convert_layout_acc_rowcol(tCrS.layout()));
      float row_scale[kORows];

      // Nkv 边界 mask: 尾部 tile 的 OOB 列置 -INF (TMA 对 OOB 行不写
      // smem, 残留值由 mask 覆盖)。
      const int kv_valid = nkv - kv_tile * kBc;
      if (kv_valid < kBc) {
#pragma unroll
        for (int row = 0; row < kSRows; ++row)
#pragma unroll
          for (int col = 0; col < size<1>(scores); ++col) {
            if (get<1>(tScS_rc(row, col)) >= kv_valid)
              scores(row, col) = -INFINITY;
          }
      }

      fa_cute::online_safe_softmax_scaled<decltype(scores), kORows>(
          scores, scale, row_max, row_sum, row_scale,
          Traits::kRescaleThreshold);

      // warp-uniform vote: 全 warp 的 row_scale 都是 1.0 时跳过 O 重缩放,
      // 无分支发散。
      bool local_need_rescale = false;
#pragma unroll
      for (int r = 0; r < kORows; ++r)
        local_need_rescale = local_need_rescale || (row_scale[r] < 1.0f);
      const bool need_rescale = __any_sync(0xffffffff, local_need_rescale);

      // P fragment: f32 -> f16 后把 C layout 重解释为 A 寄存器 (零拷贝)。
      auto tCrP = fa_cute::convert_type<Element>(tCrS);
      auto tCrPv = make_tensor(
          tCrP.data(),
          fa_cute::convert_layout_acc_Aregs<TiledMmaPV>(tCrP.layout()));

      // Phase 3: PV split-D, O[Br, v_chunk] += P[Br,Bc] @ V[Bc, v_chunk]。
#pragma unroll
      for (int v_chunk = 0; v_chunk < kDChunksV; ++v_chunk) {
        const int chunk_index = kv_tile * kDChunksV + v_chunk;
        const int v_stage = chunk_index % kStagesPV;
        const int v_phase = (chunk_index / kStagesPV) & 1;
        TmaBarrier::wait(&v_full[v_stage], v_phase);
        cutlass::arch::fence_view_async_shared();

        auto sV = make_tensor(make_smem_ptr(v_base + v_stage * kVChunkElements),
                              SmemLayoutV{});
        auto sVt = make_tensor(sV.data(), SmemLayoutVt{});
        auto tCrVStorage = thr_mma_pv.partition_fragment_B(sV);
        auto tCrV = make_tensor(tCrVStorage.data(), tCrV_layout);
        auto tVsVt = s2r_thr_v.partition_S(sVt);

        auto tCrO = make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]),
                                OFragLayout{});
        if (kv_tile > 0 && need_rescale) {
          auto tCrO_rc = make_tensor(
              tCrO.data(),
              fa_cute::convert_layout_acc_rowcol(tCrO.layout()));
#pragma unroll
          for (int row = 0; row < kORows; ++row)
#pragma unroll
            for (int col = 0; col < kOCols; ++col)
              tCrO_rc(row, col) *= row_scale[row];
        }

        fa_cute::gemm_rs(tCrO, tCrPv, tCrV, tVsVt, tiled_mma_pv, s2r_copy_v,
                         s2r_thr_v);

        CtaBarrier::arrive(&v_empty[v_stage]);

        if (tid == 0) {
          const int v_next = v_chunk + kStagesPV;
          if (v_next < kDChunksV) {
            const int next_index = kv_tile * kDChunksV + v_next;
            const int s_next = next_index % kStagesPV;
            const int phase_next = (next_index / kStagesPV) & 1;
            CtaBarrier::wait(&v_empty[s_next], phase_next);
            issue_v_tma(v_next, s_next, kv_tile);
          }
        }
      }
    }
  }

  // Phase 4: epilogue。O /= row_sum -> f16 -> STSM 暂存 smem (复用已释放
  // 的 QKV smem) -> TMA store bulk group 批量写回。kVChunksPerBatch 个
  // v_chunk 共享一个 bulk group (一次 arrive + wait), wait 次数从
  // kDChunksV 降到 kNBatches。
  // drain race: 只有 tid=0 发 store, tma_store_wait<0> 对其它线程是空操作;
  // 无 CTA barrier 时下一批的 STSM 会覆写 in-flight TMA 仍在读的 smem ->
  // kNBatches >= 2 时确定性写坏。batch 末尾的 __syncthreads 让全线程等
  // drain (batch 条件 CTA 一致, 不会死锁)。
  {
    constexpr int kVChunksPerBatch = Traits::kVChunksPerBatch;
    constexpr int kNBatches = Traits::kNBatches;
    constexpr int kOTileElems = cosize(SmemLayoutO{});

    __syncthreads();  // V smem 读完后 R->S 才能覆写 shm

    auto mO = tma_o.get_tma_tensor(make_shape(rows_q, Int<kHeadDim>{}));
    auto o_slice = tma_o.get_slice(_0{});

    auto r2s_copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, Element>{},
                                      tiled_mma_pv);
    auto r2s_thr = r2s_copy.get_thread_slice(tid);

#pragma unroll
    for (int batch = 0; batch < kNBatches; ++batch) {
#pragma unroll
      for (int v_in = 0; v_in < kVChunksPerBatch; ++v_in) {
        const int v_chunk = batch * kVChunksPerBatch + v_in;
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
        auto sO_v = make_tensor(make_smem_ptr(shm + v_in * kOTileElems),
                                SmemLayoutO{});
        auto tCrOHalf_src = r2s_thr.retile_S(tCrOHalf);
        auto tCsO_dst = r2s_thr.partition_D(sO_v);
        copy(r2s_copy, tCrOHalf_src, tCsO_dst);
      }
      cutlass::arch::fence_view_async_shared();
      __syncthreads();
#pragma unroll
      for (int v_in = 0; v_in < kVChunksPerBatch; ++v_in) {
        const int v_chunk = batch * kVChunksPerBatch + v_in;
        auto sO_v = make_tensor(make_smem_ptr(shm + v_in * kOTileElems),
                                SmemLayoutO{});
        auto gO = local_tile(mO, Shape<Int<kBr>, Int<kVDChunk>>{},
                             make_coord(q_tile, v_chunk));
        auto tCgO = o_slice.partition_D(gO);
        auto tOsO = o_slice.partition_S(sO_v);
        if (tid == 0) {
          copy(tma_o, tOsO, tCgO);
        }
      }
      tma_store_arrive();
      if (batch < kNBatches - 1) {
        tma_store_wait<0>();  // drain, shm 才能复用
        __syncthreads();
      }
    }

    // LSE 写出: log2 域转回 ln, 与最后一批 TMA drain 重叠。
    if (softmax_lse != nullptr) {
#pragma unroll
      for (int row = 0; row < kORows; ++row) {
        const float lse = (row_max[row] + log2f(row_sum[row])) * M_LN2;
        softmax_lse[blockIdx.y * nq + Br_base + get<0>(tScS_rc(row, 0))] = lse;
      }
    }

    tma_store_wait<0>();
  }
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
}

// Host launcher: BHND packed 输入, Q/KV 行数解耦。约定 nq % 128 == 0
// (kernel O epilogue 整 tile STSM+TMA store), nkv 任意 (边界 mask);
// softmax_lse 可为 nullptr (跳过 LSE 写出)。stages 经模板参数选 (默认 2)。
template <int kHeadDim, int kStagesQK = 2, int kStagesPV = 2>
void ffpa_attn_tma_split_d_cute_fwd(
    cutlass::half_t *Q, cutlass::half_t *K, cutlass::half_t *V,
    cutlass::half_t *O, float *softmax_lse, int B, int H, int nq, int nkv) {
  using namespace cute;
  using Traits = fa_cute::FFPAAttnNonWSCuTeSplitDTraits<
      kHeadDim, 128, 128, 32, 64, kStagesQK, kStagesPV>;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using SmemLayoutO = typename Traits::SmemLayoutO;

  const int rows_q = B * H * nq;
  const int rows_kv = B * H * nkv;
  auto gq = make_tensor(make_gmem_ptr(Q), make_shape(rows_q, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto gk = make_tensor(make_gmem_ptr(K), make_shape(rows_kv, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto gv = make_tensor(make_gmem_ptr(V), make_shape(rows_kv, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto go = make_tensor(make_gmem_ptr(O), make_shape(rows_q, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, gq, SmemLayoutQ{},
                             Shape<_128, _32>{}, _1{});
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, gk, SmemLayoutK{},
                             Shape<_128, _32>{}, _1{});
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, gv, SmemLayoutV{},
                             Shape<_128, _64>{}, _1{});
  auto tma_o = make_tma_copy(SM90_TMA_STORE{}, go, SmemLayoutO{},
                             Shape<_128, _64>{}, _1{});

  auto kernel = ffpa_attn_tma_split_d_cute<
      Traits, decltype(tma_q), decltype(tma_k), decltype(tma_v),
      decltype(tma_o)>;
  const int smem_bytes = Traits::kSmemElems * sizeof(cutlass::half_t);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_bytes);
  dim3 grid(nq / 128, B * H);
  kernel<<<grid, Traits::kNumThreads, smem_bytes>>>(
      tma_q, tma_k, tma_v, tma_o, O, softmax_lse, rows_q, rows_kv, nq, nkv);
}
#endif // NOTES_V2_ENABLE_CUTE && NOTES_V2_ENABLE_TMA_MMA_WS

