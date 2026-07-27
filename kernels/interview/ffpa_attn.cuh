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
    NOTES_V2_REG_ALLOC(255);
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

