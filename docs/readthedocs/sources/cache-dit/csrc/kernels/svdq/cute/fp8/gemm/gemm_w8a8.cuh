#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

#include <cuda_bf16.h>

#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/device_kernel.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "common.h"
#include "runtime.h"

namespace svdq::cute_kernels {

using namespace cute;

// Reinterpret the per-thread accumulator fragment as (row, column) coordinates. The atom's C layout
// interleaves two rows per register pair; this regrouping exposes row and column as separate modes
// so the epilogue can load one activation scale per row and one weight scale per column.
template <typename Layout>
CUTE_DEVICE auto w8a8_acc_rowcol(Layout acc_layout) {
  auto divided = logical_divide(acc_layout, Shape<_2>{});
  return make_layout(make_layout(get<0, 1>(divided), get<1>(divided)),
                     make_layout(get<0, 0>(divided), get<2>(divided)));
}

// SVDQ W8A8 GEMM with a fused BF16 LoRA-up epilogue.
//
// Mathematical contract:
//   P   = qA[M,K] @ qB[N,K]^T                      (E4M3 codes, FP32 accumulation)
//   out = BF16(P * (ascale[m] * wscale[n]) + bias[n] + D[M,R] @ L1[N,R]^T)
// ascale is the per-token activation scale and wscale the per-output-channel residual scale.
// Both are constant along K, so they factor out of the reduction and no block-scaled mainloop is
// needed. The output scales are applied to the E4M3 product before the LoRA-up MMA accumulates
// into the same FP32 fragment, so neither scale can leak into D @ L1^T. The FP32 -> BF16
// conversion happens exactly once, after both products.
//
// Physical tensors (all contiguous CUDA):
//   act      [M, K] E4M3 activation codes, row-major (uint8 storage)
//   wgt      [N, K] E4M3 residual codes, row-major
//   ascales  [M] FP32, wscales [N] FP32
//   out      [M, N] BF16 row-major
//   lora_act [M, R] BF16 LoRA-down output D, row-major
//   lora_up  [N, R] BF16 L1, row-major
//   bias     [N] BF16, optional
// R is 0, 32, 64 or 128. M/N/K must be multiples of the tile extents enforced by the host wrapper.

template <int kBN_ = 128, int kStages_ = 3, int kBK_ = 128>
struct W8A8CuteTraits {
  static constexpr int kBM = 128;
  static constexpr int kBN = kBN_;
  static constexpr int kBK = kBK_;
  static constexpr int kStages = kStages_;
  // LoRA rank is streamed in 32-wide slices, the same granularity as the FP4 kernel: one D/L1 slice
  // is 8 KiB and therefore always fits inside one A/B stage even though FP8 stages are twice as
  // large as the packed E2M1 ones.
  static constexpr int kRankTile = 32;
  static constexpr int kNumThreads = kBM / 16 * 32;

  static_assert(kBK % 32 == 0, "BK must contain whole 32-element FP8 MMA atoms");
  static_assert(kStages >= 2, "kStages >= 2");

  using Element = cutlass::float_e4m3_t;
  using ElementAB = uint8_t;
  using ElementLora = cutlass::bfloat16_t;
  using ElementO = cutlass::bfloat16_t;
  using ElementAcc = float;

  // Dense SM120 FP8 MMA. The permutation matches the dense (non block-scaled) SM120 builder:
  // M widens to BM, N stays 32 and K is exactly one 32-element atom per fragment step.
  using MmaAtom = MMA_Atom<SM120_16x8x32_TN<Element, Element, ElementAcc>>;
  using AtomLayoutMNK = Layout<Shape<_4, _2, _1>>;
  using PermutationMNK = Tile<Int<kBM>, _32, _32>;
  using TiledMma = decltype(make_tiled_mma(MmaAtom{}, AtomLayoutMNK{}, PermutationMNK{}));

  // The LoRA-up MMA differs only in K; keeping M/N identical is what lets both products share one
  // accumulator without a retile or a cross-warpgroup reduction.
  using LoraAtom = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
  using LoraPermutationMNK = Tile<Int<kBM>, _32, Int<kRankTile>>;
  using LoraMma = decltype(make_tiled_mma(LoraAtom{}, AtomLayoutMNK{}, LoraPermutationMNK{}));

  static_assert(size(LoraMma{}) == size(TiledMma{}),
                "the LoRA MMA must use the same thread count as the FP8 MMA");
  static_assert(
    is_same_v<decltype(partition_shape_C(TiledMma{}, make_shape(Int<kBM>{}, Int<kBN>{}))),
              decltype(partition_shape_C(LoraMma{}, make_shape(Int<kBM>{}, Int<kBN>{})))>,
    "the LoRA MMA accumulator partition must match the FP8 MMA one");

  using SmemLayoutAtomA = UMMA::Layout_K_SW128_Atom<ElementAB>;
  using SmemLayoutAtomB = UMMA::Layout_K_SW128_Atom<ElementAB>;
  using SmemLayoutA =
    decltype(tile_to_shape(SmemLayoutAtomA{}, Shape<Int<kBM>, Int<kBK>, Int<kStages>>{}));
  using SmemLayoutB =
    decltype(tile_to_shape(SmemLayoutAtomB{}, Shape<Int<kBN>, Int<kBK>, Int<kStages>>{}));

  // D/L1 live in their own two-stage smem ring, filled by the producer warp while the consumer
  // warps are still running the E4M3 mainloop, so the low-rank term adds no exposed tail. A 32-wide
  // BF16 rank slice is 64 B per row, exactly one SW64 atom, so LDSM reads it conflict-free.
  static constexpr int kLoraStages = 2;
  using LoraSmemAtom = UMMA::Layout_K_SW64_Atom<ElementLora>;
  using SmemLayoutD = decltype(tile_to_shape(
    LoraSmemAtom{}, make_shape(Int<kBM>{}, Int<kRankTile>{}, Int<kLoraStages>{})));
  using SmemLayoutL1 = decltype(tile_to_shape(
    LoraSmemAtom{}, make_shape(Int<kBN>{}, Int<kRankTile>{}, Int<kLoraStages>{})));

  using SmemCopyAtomAB = Copy_Atom<SM75_U32x4_LDSM_N, ElementAB>;
  using SmemCopyAtomLora = Copy_Atom<SM75_U32x4_LDSM_N, ElementLora>;

  using SmemAtomO = GMMA::Layout_K_SW128_Atom<ElementO>;
  using SmemLayoutO = decltype(tile_to_shape(SmemAtomO{}, Shape<Int<kBM>, Int<kBN>>{}));

  static constexpr int kAStageBytes = kBM * kBK;
  static constexpr int kBStageBytes = kBN * kBK;
  static constexpr int kDStageBytes = kBM * kRankTile * static_cast<int>(sizeof(ElementLora));
  static constexpr int kL1StageBytes = kBN * kRankTile * static_cast<int>(sizeof(ElementLora));
  static constexpr int kStageBytes = kAStageBytes + kBStageBytes;
  static constexpr int kMainSmemBytes = kStages * kStageBytes;
  static constexpr int kLoraSmemBytes = kLoraStages * (kDStageBytes + kL1StageBytes);
  static constexpr int kSmemBytes = kMainSmemBytes + kLoraSmemBytes;

  static_assert(kAStageBytes % 1024 == 0 && kBStageBytes % 1024 == 0,
                "the stage stride must preserve the 1024 B smem base alignment");
  static_assert(cosize_v<SmemLayoutO> * static_cast<int>(sizeof(ElementO)) <= kMainSmemBytes,
                "the BF16 output staging tile must fit into the pipeline storage");
  // The kernel runs one CTA per SM because of its register budget, so borrowing smem for the LoRA
  // ring costs no occupancy as long as it stays under the SM120 opt-in limit.
  static_assert(kSmemBytes <= 99 * 1024, "pipeline plus LoRA storage exceeds the SM120 budget");

  static constexpr uint32_t kTxBytesMain = kAStageBytes + kBStageBytes;
};

template <class TmaA_, class TmaB_, class TmaC_>
struct W8A8CuteTma {
  TmaA_ a;
  TmaB_ b;
  TmaC_ c;
};

struct W8A8CuteCacheKey {
  std::array<uint64_t, 8> data;
  bool operator==(const W8A8CuteCacheKey &) const = default;
};

struct W8A8CuteCacheHash {
  size_t operator()(const W8A8CuteCacheKey &key) const {
    uint64_t hash = 0x9e3779b97f4a7c15ULL;
    for (const uint64_t value : key.data) {
      hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    }
    return static_cast<size_t>(hash);
  }
};

template <class Traits, class TmaParams>
class W8A8GemmKernel {
  using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
  using CtaBarrier = cutlass::arch::ClusterBarrier;
  static constexpr int kBM = Traits::kBM;
  static constexpr int kBN = Traits::kBN;
  static constexpr int kBK = Traits::kBK;
  static constexpr int kStages = Traits::kStages;
  static constexpr int kRankTile = Traits::kRankTile;

 public:
  // Asynchronous 16 B gmem -> smem copy; the destination address comes from the swizzled smem
  // layout, so the copy preserves the LDSM fragment pattern without any register roundtrip.
  __device__ __forceinline__ static void cp_async_16(void *dst, void const *src) {
    const uint32_t smem_addr = cute::cast_smem_ptr_to_uint(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(src));
  }

  /**
   * Issue one main-K tile's asynchronous A and B TMA copies.
   *
   * Both operands are E4M3 codes in row-major gmem; the copies only move bytes and share one
   * transaction barrier whose budget is A+B. The caller must already own an empty stage.
   *
   * @param tma     Descriptor bundle; only tma.a and tma.b are used.
   * @param full    Transaction-barrier ring; full[stage] is armed by this call.
   * @param kt      Logical K-tile index in units of BK elements.
   * @param stage   Physical smem ring slot.
   * @param mA/mB   CuTe coordinate tensors for the complete activation/weight tensors.
   * @param sA/sB   Multi-stage smem views; this call writes only (..., stage).
   * @param a_slice/b_slice TMA thread slices connecting global source and smem destination.
   * @return        void; completion is asynchronous and observed through full[stage].
   */
  template <class GmemA, class GmemB, class SmemA, class SmemB, class ASlice, class BSlice>
  __device__ __forceinline__ static void tma_load_A_B(TmaParams const &tma, uint64_t *full, int kt,
                                                      int stage, GmemA const &mA, GmemB const &mB,
                                                      SmemA &sA, SmemB &sB, ASlice const &a_slice,
                                                      BSlice const &b_slice) {
    auto gA = local_tile(mA, Shape<Int<kBM>, Int<kBK>>{}, make_coord(blockIdx.y, kt));
    auto gB = local_tile(mB, Shape<Int<kBN>, Int<kBK>>{}, make_coord(blockIdx.x, kt));
    TmaBarrier::arrive_and_expect_tx(&full[stage], Traits::kTxBytesMain);
    copy(tma.a.with(full[stage]), a_slice.partition_S(gA), a_slice.partition_D(sA(_, _, stage)));
    copy(tma.b.with(full[stage]), b_slice.partition_S(gB), b_slice.partition_D(sB(_, _, stage)));
  }

  /**
   * Execute the E4M3 main reduction into the shared FP32 accumulator.
   *
   * Each K tile is split into BK/32 sub-blocks so only one A/B atom pair is live at a time. This
   * helper applies no scales: ascale/wscale are handled by the caller after the reduction.
   *
   * @param full/empty Shared barrier rings; warp leaders release the stage after the last MMA.
   * @param tid        Consumer thread index.
   * @param nKt        Number of main K tiles.
   * @param tiled_mma  FP8 tiled MMA descriptor.
   * @return           void; tCrC holds the raw E4M3 product on return.
   */
  template <class TiledMma, class CopyA, class CopyB, class SrcA, class SrcB, class DstA,
            class DstB, class FragA, class FragB, class Acc>
  __device__ __forceinline__ static void gemm_fp8_main(
    uint64_t *full, uint64_t *empty, int tid, int nKt, TiledMma const &tiled_mma,
    CopyA const &s2r_copy_a, CopyB const &s2r_copy_b, SrcA const &tCsA, SrcB const &tCsB,
    DstA &tCrA_view, DstB &tCrB_view, FragA &tCrA, FragB &tCrB, Acc &tCrC) {
#pragma unroll 1
    for (int kt = 0; kt < nKt; ++kt) {
      const int stage = kt % kStages;
      const int phase = (kt / kStages) & 1;
      TmaBarrier::wait(&full[stage], phase);
#pragma unroll
      for (int k_block = 0; k_block < kBK / 32; ++k_block) {
        copy(s2r_copy_a, tCsA(_, _, k_block, stage), tCrA_view(_, _, k_block));
        copy(s2r_copy_b, tCsB(_, _, k_block, stage), tCrB_view(_, _, k_block));
        cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
      }
      __syncwarp();
      if (tid % 32 == 0) CtaBarrier::arrive(&empty[stage]);
    }
  }

  /**
   * Apply the per-token and per-output-channel scales plus bias to the FP32 accumulator.
   *
   * Math per element: acc = fmaf(acc * ascale[m], wscale[n], bias[n]). This runs before the LoRA
   * MMA, so the scales apply to the E4M3 product only and never to D @ L1^T. ascale depends on the
   * row and wscale on the column, so each thread loads its own small set of both.
   *
   * @param ascales/wscales FP32 scale vectors in gmem.
   * @param bias            Optional BF16 bias [N]; nullptr means zero.
   * @param m0/n0           Global row/column offsets of this CTA tile.
   * @param tScC_rc         Row/column coordinate view of the accumulator partition.
   * @return                void; tCrC is scaled and biased in place.
   */
  template <class CoordC, class Acc>
  __device__ __forceinline__ static void apply_scales(float const *ascales, float const *wscales,
                                                      cutlass::bfloat16_t const *bias, int m0,
                                                      int n0, CoordC const &tScC_rc, Acc &tCrC) {
    constexpr int kCRows = decltype(size<0>(tScC_rc))::value;
    constexpr int kCCols = decltype(size<1>(tScC_rc))::value;
    float row_scale[kCRows];
    float col_scale[kCCols];
    float col_bias[kCCols];
#pragma unroll
    for (int r = 0; r < kCRows; ++r) {
      row_scale[r] = __ldg(ascales + m0 + get<0>(tScC_rc(r, 0)));
    }
#pragma unroll
    for (int c = 0; c < kCCols; ++c) {
      const int n = n0 + get<1>(tScC_rc(0, c));
      col_scale[c] = __ldg(wscales + n);
      col_bias[c] = bias != nullptr ? static_cast<float>(bias[n]) : 0.0f;
    }
    auto acc = make_tensor(tCrC.data(), w8a8_acc_rowcol(tCrC.layout()));
#pragma unroll
    for (int r = 0; r < kCRows; ++r)
#pragma unroll
      for (int c = 0; c < kCCols; ++c)
        acc(r, c) = fmaf(acc(r, c) * row_scale[r], col_scale[c], col_bias[c]);
  }

  /**
   * Issue the asynchronous copies for one D/L1 rank slice.
   *
   * Each participating thread moves 16 B chunks whose destination addresses come from the swizzled
   * smem layout, so gmem reads stay contiguous while the destination matches the LDSM pattern.
   * The copies are only issued here; the caller commits and waits on them before the LoRA MMA.
   *
   * @param lora_act/lora_up Global BF16 D [M,rank] and L1 [N,rank].
   * @param rank             Total rank; 0, 32, 64 or 128.
   * @param rt               Rank-slice index; covered columns are [rt*32, rt*32+32).
   * @param m0/n0            Global row/column offsets of this CTA tile.
   * @param tid              Participating thread index within the filling group.
   * @param stage            Destination slot in the dedicated LoRA ring.
   * @param num_threads      Number of threads that take part in the fill.
   * @return                 void; completion is observed through cp.async waits, not a barrier.
   */
  template <class SmemD, class SmemL1>
  __device__ __forceinline__ static void fill_lora_stage(cutlass::bfloat16_t const *lora_act,
                                                         cutlass::bfloat16_t const *lora_up,
                                                         int rank, int rt, int m0, int n0, int tid,
                                                         int stage, SmemD &sD, SmemL1 &sL1,
                                                         int num_threads) {
    constexpr int kChunksPerRow = kRankTile * 2 / 16;
    constexpr int kDChunks = kBM * kChunksPerRow;
    constexpr int kL1Chunks = kBN * kChunksPerRow;
    const int col_base = rt * kRankTile;
    auto sD_stage = sD(_, _, stage);
    auto sL1_stage = sL1(_, _, stage);
    for (int chunk = tid; chunk < kDChunks; chunk += num_threads) {
      const int row = chunk / kChunksPerRow;
      const int col = (chunk % kChunksPerRow) * 8;
      cp_async_16(&sD_stage(row, col),
                  lora_act + static_cast<int64_t>(m0 + row) * rank + col_base + col);
    }
    for (int chunk = tid; chunk < kL1Chunks; chunk += num_threads) {
      const int row = chunk / kChunksPerRow;
      const int col = (chunk % kChunksPerRow) * 8;
      cp_async_16(&sL1_stage(row, col),
                  lora_up + static_cast<int64_t>(n0 + row) * rank + col_base + col);
    }
  }

  /**
   * Accumulate the BF16 LoRA-up correction into the same FP32 accumulator.
   *
   * Math per rank slice: tCrC += D_rt @ L1_rt^T. Both operands are BF16 and accumulation is FP32.
   * D/L1 are staged in the dedicated smem ring by asynchronous copies and consumed through LDSM.
   * This schedule runs after apply_scales, so the output scales never touch the low-rank term.
   *
   * @param lora_act/lora_up Global BF16 D [M,rank] and L1 [N,rank].
   * @param rank             Total rank; 0, 32, 64 or 128.
   * @param m0/n0            Global row/column offsets of this CTA tile.
   * @param nRt              Number of 32-wide rank slices.
   * @return                 void; all rank slices have been accumulated on return.
   */
  template <class SmemD, class SmemL1, class Acc>
  __device__ __forceinline__ static void gemm_bf16_lora(cutlass::bfloat16_t const *lora_act,
                                                        cutlass::bfloat16_t const *lora_up,
                                                        int rank, int m0, int n0, int tid, int nRt,
                                                        SmemD &sD, SmemL1 &sL1, Acc &tCrC) {
    if (nRt == 0) return;
    typename Traits::LoraMma lora_mma;
    auto thr_lora = lora_mma.get_thread_slice(tid);
    auto tCrD = thr_lora.partition_fragment_A(sD(_, _, Int<0>{}));
    auto tCrL1 = thr_lora.partition_fragment_B(sL1(_, _, Int<0>{}));
    auto s2r_copy_d = make_tiled_copy_A(typename Traits::SmemCopyAtomLora{}, lora_mma);
    auto s2r_copy_l1 = make_tiled_copy_B(typename Traits::SmemCopyAtomLora{}, lora_mma);
    auto s2r_thr_d = s2r_copy_d.get_thread_slice(tid);
    auto s2r_thr_l1 = s2r_copy_l1.get_thread_slice(tid);
    auto tCsD = s2r_thr_d.partition_S(as_position_independent_swizzle_tensor(sD));
    auto tCsL1 = s2r_thr_l1.partition_S(as_position_independent_swizzle_tensor(sL1));
    auto tCrD_view = s2r_thr_d.retile_D(tCrD);
    auto tCrL1_view = s2r_thr_l1.retile_D(tCrL1);
    constexpr int kLoraStages = Traits::kLoraStages;
#pragma unroll 1
    for (int rt = 0; rt < nRt; ++rt) {
      const int stage = rt % kLoraStages;
      if (rt >= kLoraStages) {
        // Ranks beyond the prefetched ring refill on demand; only rank 128 reaches this path.
        cutlass::arch::NamedBarrier::sync(Traits::kNumThreads, 1);
        fill_lora_stage(lora_act, lora_up, rank, rt, m0, n0, tid, stage, sD, sL1,
                        Traits::kNumThreads);
        cute::cp_async_fence();
        cute::cp_async_wait<0>();
        cutlass::arch::NamedBarrier::sync(Traits::kNumThreads, 1);
      }
      copy(s2r_copy_d, tCsD(_, _, _, stage), tCrD_view);
      copy(s2r_copy_l1, tCsL1(_, _, _, stage), tCrL1_view);
      cute::gemm(lora_mma, tCrD, tCrL1, tCrC);
    }
  }

  /**
   * Convert the final FP32 accumulator once and stage the BF16 tile in smem.
   *
   * The first barrier proves the pipeline storage is no longer live; the second one, after the
   * async-shared fence, proves every consumer's STSM is visible before the TMA store.
   *
   * @return void; the BF16 tile is visible in smem after the trailing barrier.
   */
  template <class TiledMma, class Acc>
  __device__ __forceinline__ static void stage_output_tile(void *shm, int tid,
                                                           TiledMma const &tiled_mma,
                                                           Acc const &tCrC) {
    using ElementO = typename Traits::ElementO;
    using SmemLayoutO = typename Traits::SmemLayoutO;
    constexpr int kFrag = decltype(size(tCrC))::value;
    cutlass::Array<ElementO, kFrag> acc_bf16;
    cutlass::NumericArrayConverter<ElementO, float, kFrag> convert;
    acc_bf16 = convert(*reinterpret_cast<cutlass::Array<float, kFrag> const *>(tCrC.data()));
    auto tCrCh = make_tensor(make_rmem_ptr(acc_bf16.data()), tCrC.layout());
    __syncthreads();
    auto r2s_copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, ElementO>{}, tiled_mma);
    auto r2s_thr = r2s_copy.get_thread_slice(tid);
    auto sC = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(shm)), SmemLayoutO{});
    copy(r2s_copy, r2s_thr.retile_S(tCrCh), r2s_thr.partition_D(sC));
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
  }

  /**
   * Publish the staged smem tile through the output TMA descriptor and drain the store.
   *
   * @return void; the output tile is visible in global memory after the drain.
   */
  template <class GmemC, class CSlice>
  __device__ __forceinline__ static void tma_store_O(TmaParams const &tma, GmemC const &mC,
                                                     CSlice const &c_slice, void *shm, int tid) {
    using ElementO = typename Traits::ElementO;
    using SmemLayoutO = typename Traits::SmemLayoutO;
    auto sC = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(shm)), SmemLayoutO{});
    auto gC = local_tile(mC, Shape<Int<kBM>, Int<kBN>>{}, make_coord(blockIdx.y, blockIdx.x));
    if (tid == 0) copy(tma.c, c_slice.partition_S(sC), c_slice.partition_D(gC));
    cute::tma_store_arrive();
    cute::tma_store_wait<0>();
  }
};

template <class Traits, class TmaParams>
__global__ void __launch_bounds__(Traits::kNumThreads + 128, 1)
  gemm_w8a8_kernel(CUTLASS_GRID_CONSTANT TmaParams const tma, float const *__restrict__ ascales,
                   float const *__restrict__ wscales,
                   cutlass::bfloat16_t const *__restrict__ lora_act,
                   cutlass::bfloat16_t const *__restrict__ lora_up,
                   cutlass::bfloat16_t const *__restrict__ bias, int M, int N, int K, int rank) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
  using CtaBarrier = cutlass::arch::ClusterBarrier;
  using ElementAB = typename Traits::ElementAB;
  using ElementLora = typename Traits::ElementLora;
  using TiledMma = typename Traits::TiledMma;
  using Gemm = W8A8GemmKernel<Traits, TmaParams>;
  constexpr int kBM = Traits::kBM;
  constexpr int kBN = Traits::kBN;
  constexpr int kBK = Traits::kBK;
  constexpr int kStages = Traits::kStages;
  constexpr int kLoraStages = Traits::kLoraStages;
  constexpr int kNumThreads = Traits::kNumThreads;

  const int raw_tid = threadIdx.x;
  const bool is_producer = raw_tid < 128;
  const int tid = is_producer ? 0 : raw_tid - 128;
  const int m0 = blockIdx.y * kBM;
  const int n0 = blockIdx.x * kBN;
  const int nKt = K / kBK;
  const int nRt = rank / Traits::kRankTile;

  auto mA = tma.a.get_tma_tensor(make_shape(M, K));
  auto mB = tma.b.get_tma_tensor(make_shape(N, K));
  auto mC = tma.c.get_tma_tensor(make_shape(M, N));
  auto a_slice = tma.a.get_slice(_0{});
  auto b_slice = tma.b.get_slice(_0{});
  auto c_slice = tma.c.get_slice(_0{});

  extern __shared__ __align__(1024) uint8_t shm[];
  uint8_t *a_bytes = shm;
  uint8_t *b_bytes = a_bytes + Traits::kAStageBytes * kStages;
  uint8_t *d_bytes = shm + Traits::kMainSmemBytes;
  uint8_t *l1_bytes = d_bytes + kLoraStages * Traits::kDStageBytes;

  __shared__ uint64_t full[kStages];
  __shared__ uint64_t empty[kStages];
  if (raw_tid == 0) {
    for (int s = 0; s < kStages; ++s) {
      TmaBarrier::init(&full[s], 1);
      CtaBarrier::init(&empty[s], kNumThreads / 32);
    }
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();

  auto sA_full = make_tensor(make_smem_ptr(reinterpret_cast<ElementAB *>(a_bytes)),
                             typename Traits::SmemLayoutA{});
  auto sB_full = make_tensor(make_smem_ptr(reinterpret_cast<ElementAB *>(b_bytes)),
                             typename Traits::SmemLayoutB{});
  auto sD_full = make_tensor(make_smem_ptr(reinterpret_cast<ElementLora *>(d_bytes)),
                             typename Traits::SmemLayoutD{});
  auto sL1_full = make_tensor(make_smem_ptr(reinterpret_cast<ElementLora *>(l1_bytes)),
                              typename Traits::SmemLayoutL1{});

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto s2r_copy_a = make_tiled_copy_A(typename Traits::SmemCopyAtomAB{}, tiled_mma);
  auto s2r_copy_b = make_tiled_copy_B(typename Traits::SmemCopyAtomAB{}, tiled_mma);
  auto s2r_thr_a = s2r_copy_a.get_thread_slice(tid);
  auto s2r_thr_b = s2r_copy_b.get_thread_slice(tid);
  auto tCsA = s2r_thr_a.partition_S(as_position_independent_swizzle_tensor(sA_full));
  auto tCsB = s2r_thr_b.partition_S(as_position_independent_swizzle_tensor(sB_full));
  auto tCrA = thr_mma.partition_fragment_A(sA_full(_, _, Int<0>{}));
  auto tCrB = thr_mma.partition_fragment_B(sB_full(_, _, Int<0>{}));
  auto tCrA_view = s2r_thr_a.retile_D(tCrA);
  auto tCrB_view = s2r_thr_b.retile_D(tCrB);

  auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<kBM>, Int<kBN>>{});
  clear(tCrC);
  auto cC = make_identity_tensor(Shape<Int<kBM>, Int<kBN>>{});
  auto tScC = thr_mma.partition_C(cC);
  auto tScC_rc = make_tensor(tScC.data(), w8a8_acc_rowcol(tScC.layout()));

  if (is_producer) {
    cutlass::arch::warpgroup_reg_dealloc<32>();
    if (raw_tid == 0) {
      for (int g = 0; g < nKt; ++g) {
        const int stage = g % kStages;
        const int phase = (g / kStages) & 1;
        CtaBarrier::wait(&empty[stage], phase);
        Gemm::tma_load_A_B(tma, full, g, stage, mA, mB, sA_full, sB_full, a_slice, b_slice);
      }
    }
    return;
  }

  cutlass::arch::warpgroup_reg_alloc<224>();
  if (tid % 32 == 0) {
    for (int s = 0; s < kStages; ++s) CtaBarrier::arrive(&empty[s]);
  }

  // A single consumer warp asynchronously prefetches the first LoRA rank slices; the other warps
  // enter the mainloop immediately. The dedicated LoRA ring is disjoint from the main pipeline, so
  // the copies overlap the reduction and are waited on only before the low-rank epilogue.
  const int nPrefetch = nRt < kLoraStages ? nRt : kLoraStages;
  if (tid < 32) {
    for (int rt = 0; rt < nPrefetch; ++rt) {
      Gemm::fill_lora_stage(lora_act, lora_up, rank, rt, m0, n0, tid, rt, sD_full, sL1_full, 32);
    }
    if (nPrefetch > 0) cute::cp_async_fence();
  }

  Gemm::gemm_fp8_main(full, empty, tid, nKt, tiled_mma, s2r_copy_a, s2r_copy_b, tCsA, tCsB,
                      tCrA_view, tCrB_view, tCrA, tCrB, tCrC);
  Gemm::apply_scales(ascales, wscales, bias, m0, n0, tScC_rc, tCrC);
  if (nPrefetch > 0) {
    cute::cp_async_wait<0>();
    __syncthreads();
  }
  Gemm::gemm_bf16_lora(lora_act, lora_up, rank, m0, n0, tid, nRt, sD_full, sL1_full, tCrC);
  Gemm::stage_output_tile(shm, tid, tiled_mma, tCrC);
  Gemm::tma_store_O(tma, mC, c_slice, shm, tid);
#endif
}

/**
 * Launch one fused W8A8 + LoRA GEMM.
 *
 * Descriptors are cached per exact pointer/shape/rank key because they are pure copy recipes.
 * D/L1 use direct asynchronous copies rather than TMA descriptors.
 *
 * :param act: E4M3 activation codes [M,K].
 * :param wgt: E4M3 residual codes [N,K].
 * :param out: BF16 output [M,N].
 * :param ascales: Per-token FP32 activation scales [M].
 * :param wscales: Per-output-channel FP32 residual scales [N].
 * :param lora_act: BF16 LoRA-down output [M,rank], may be null when rank is 0.
 * :param lora_up: BF16 LoRA-up weight [N,rank], may be null when rank is 0.
 * :param bias: Optional BF16 bias [N].
 * :param stream: CUDA stream for the launch.
 */
template <int kBN = 128, int kStages = 3, int kBK = 128>
void launch_gemm_w8a8(void const *act, void const *wgt, void *out, float const *ascales,
                      float const *wscales, cutlass::bfloat16_t const *lora_act,
                      cutlass::bfloat16_t const *lora_up, cutlass::bfloat16_t const *bias, int M,
                      int N, int K, int rank, cudaStream_t stream) {
  using Traits = W8A8CuteTraits<kBN, kStages, kBK>;
  using ElementAB = typename Traits::ElementAB;
  using ElementO = typename Traits::ElementO;
  constexpr int kBM = Traits::kBM;

  auto mA = make_tensor(make_gmem_ptr(static_cast<ElementAB const *>(act)), make_shape(M, K),
                        make_stride(K, _1{}));
  auto mB = make_tensor(make_gmem_ptr(static_cast<ElementAB const *>(wgt)), make_shape(N, K),
                        make_stride(K, _1{}));
  auto mC = make_tensor(make_gmem_ptr(static_cast<ElementO *>(out)), make_shape(M, N),
                        make_stride(N, _1{}));
  auto build_params = [&]() {
    auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, mA, typename Traits::SmemLayoutA{}(_, _, Int<0>{}),
                               Shape<Int<kBM>, Int<kBK>>{}, _1{});
    auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, mB, typename Traits::SmemLayoutB{}(_, _, Int<0>{}),
                               Shape<Int<kBN>, Int<kBK>>{}, _1{});
    auto tma_c = make_tma_copy(SM90_TMA_STORE{}, mC, typename Traits::SmemLayoutO{},
                               Shape<Int<kBM>, Int<kBN>>{}, _1{});
    using Params = W8A8CuteTma<decltype(tma_a), decltype(tma_b), decltype(tma_c)>;
    return Params{tma_a, tma_b, tma_c};
  };
  using TmaParams = decltype(build_params());
  auto address = [](const void *ptr) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
  };
  W8A8CuteCacheKey key{{address(act), address(wgt), address(out), address(lora_act),
                        address(lora_up), static_cast<uint64_t>(M) << 32 | static_cast<uint32_t>(N),
                        static_cast<uint64_t>(K) << 32 | static_cast<uint32_t>(rank), 0}};
  thread_local std::unordered_map<W8A8CuteCacheKey, TmaParams, W8A8CuteCacheHash> cache;
  auto it = cache.find(key);
  if (it == cache.end()) {
    if (cache.size() >= 32) cache.clear();
    it = cache.emplace(key, build_params()).first;
  }

  auto kernel = gemm_w8a8_kernel<Traits, TmaParams>;
  static thread_local std::unordered_set<int> configured_devices;
  int device = 0;
  checkCUDA(cudaGetDevice(&device));
  if (configured_devices.insert(device).second) {
    checkCUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   Traits::kSmemBytes));
  }
  kernel<<<dim3(N / kBN, M / kBM), Traits::kNumThreads + 128, Traits::kSmemBytes, stream>>>(
    it->second, ascales, wscales, lora_act, lora_up, bias, M, N, K, rank);
}

}  // namespace svdq::cute_kernels
