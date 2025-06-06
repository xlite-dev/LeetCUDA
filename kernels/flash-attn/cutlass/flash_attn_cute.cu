#include <cutlass/fast_math.h>
#include <torch/extension.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <type_traits>

#include "utils.h"

using namespace cute;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ref:
// https://github.com/Dao-AILab/flash-attention/blob/fd2fc9d85c8e54e5c20436465bca709bc1a6c5a1/csrc/flash_attn/src/kernel_traits.h#L15-L159
template <typename T_, int BlockQO_, int BlockKV_, int HeadDim_,
          int NWarpsPerSM_>
struct FlashAttnConfig {
  using T = T_;
  // https://github.com/Dao-AILab/flash-attention/issues/1512#issuecomment-2688567176
  static constexpr int NWarpsPerSM = NWarpsPerSM_;
  static constexpr int NumThreads = NWarpsPerSM * 32;
  // Tiling config
  static constexpr int BlockQO = BlockQO_;
  static constexpr int BlockKV = BlockKV_;
  static constexpr int HeadDim =
      HeadDim_; // we don't tile on block dim dimension, otherwise we run into
                // a split k implementation

  // Gmem2Smem config
  using GmemCopyAtom =
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<sizeof(uint128_t) * 8>,
                T>;
  static constexpr int GmemValsPerLoad = sizeof(uint128_t) / sizeof(T);
  static constexpr int GmemThreadsPerRow =
      HeadDim / GmemValsPerLoad; // each thread reads 128 bit
  using TiledCopyQKVO = decltype(make_tiled_copy(
      GmemCopyAtom{},
      make_layout(
          Shape<Int<NumThreads / GmemThreadsPerRow>, Int<GmemThreadsPerRow>>{},
          GenRowMajor{}),
      make_layout(Shape<_1, Int<GmemValsPerLoad>>{}, GenRowMajor{})));
  static_assert(Int<NumThreads / GmemThreadsPerRow>::value <= BlockQO,
                "NumThreads must be less than or equal to BlockQO");
  // Smem to Rmem config
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N,
                                 T>; // LDSM will fit in the MMA_Atom, note that
                                     // we do not handle bank conflict here
  using SmemCopyAtomTransposed =
      Copy_Atom<SM75_U16x8_LDSM_T, T>; // for column major load
  using SmemCopyAtomO =
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<sizeof(uint128_t) * 8>,
                T>; // NOTE: stmatrix is only available after sm90, we use a
                    // vectorized copy instead

  // MMA config
  static_assert(std::is_same_v<T, half_t> || std::is_same_v<T, bfloat16_t>);
  // For simplicity, mnk == (16, 8, 8) is used: two MMAs will have the same
  // layout so that we don't need to adjust tSrS to fit in tOrS
  using MMA_Atom = std::conditional_t<std::is_same_v<T, half_t>,
                                      MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>,
                                      MMA_Atom<SM80_16x8x8_F32BF16BF16F32_TN>>;

  using TiledMMA = decltype(make_tiled_mma(
      MMA_Atom{}, make_layout(Shape<Int<NWarpsPerSM>, _1, _1>{}, GenRowMajor{}),
      Tile<Int<16 * NWarpsPerSM>, _16, _16>{}
      // for SM75_U32x4_LDSM_N, we need at least 4 * 8x8 matrix, which is 16x16
      ));
  static_assert(
      16 * NWarpsPerSM <= BlockQO && 16 <= BlockKV && 16 <= HeadDim,
      "BlockQO, BlockKV, and HeadDim must be greater than or equal to "
      "16 * NWarpsPerSM, 16, and 16 respectively");
  // sanity checks
  static_assert(size(TiledMMA{}) == NumThreads &&
                size(TiledMMA{}) == size(TiledCopyQKVO{}));
};

template <typename FlashAttnConfig_>
__global__ void flash_attn_cute_kernel(typename FlashAttnConfig_::T *pQ,
                                       typename FlashAttnConfig_::T *pK,
                                       typename FlashAttnConfig_::T *pV,
                                       typename FlashAttnConfig_::T *pO, int B,
                                       int H, int N_QO, int N_KV, int D,
                                       float scaler) {
  using namespace cute;

  // unpack config
  using T = typename FlashAttnConfig_::T;
  constexpr int BlockQO = FlashAttnConfig_::BlockQO;
  constexpr int BlockKV = FlashAttnConfig_::BlockKV;
  constexpr int HeadDim = FlashAttnConfig_::HeadDim;
  using TiledCopy = typename FlashAttnConfig_::TiledCopyQKVO;
  using SmemCopyAtom = typename FlashAttnConfig_::SmemCopyAtom;
  using SmemCopyAtomTransposed =
      typename FlashAttnConfig_::SmemCopyAtomTransposed;
  using SmemCopyAtomO = typename FlashAttnConfig_::SmemCopyAtomO;
  using TiledMMA = typename FlashAttnConfig_::TiledMMA;
  assert(HeadDim == D);

  const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
  const int tx = threadIdx.x;

  auto Q =
      make_tensor(make_gmem_ptr(pQ),
                  make_layout(make_shape(B, H, N_QO, HeadDim), GenRowMajor{}));
  auto O =
      make_tensor(make_gmem_ptr(pO),
                  make_layout(make_shape(B, H, N_QO, HeadDim), GenRowMajor{}));
  auto K =
      make_tensor(make_gmem_ptr(pK),
                  make_layout(make_shape(B, H, N_KV, HeadDim), GenRowMajor{}));
  auto V =
      make_tensor(make_gmem_ptr(pV),
                  make_layout(make_shape(B, H, N_KV, HeadDim), GenRowMajor{}));

  auto gQ =
      local_tile(Q, make_shape(_1{}, _1{}, Int<BlockQO>{}, Int<HeadDim>{}),
                 make_coord(bx, by, bz, 0))(0, 0, _, _); // (BlockQO, HeadDim)
  auto gO =
      local_tile(O, make_shape(_1{}, _1{}, Int<BlockQO>{}, Int<HeadDim>{}),
                 make_coord(bx, by, bz, 0))(0, 0, _, _); // (BlockQO, HeadDim)
  auto gK = local_tile(
      K, make_shape(_1{}, _1{}, Int<BlockKV>{}, Int<HeadDim>{}),
      make_coord(bx, by, _, 0))(0, 0, _, _, _); // (BlockKV, HeadDim, RestKV)
  auto gV = local_tile(
      V, make_shape(_1{}, _1{}, Int<BlockKV>{}, Int<HeadDim>{}),
      make_coord(bx, by, _, 0))(0, 0, _, _, _); // (BlockKV, HeadDim, RestKV)

  __shared__ T psQ[BlockQO * HeadDim], psK[BlockKV * HeadDim],
      psV[BlockKV * HeadDim];

  auto sQ = make_tensor(
      make_smem_ptr(psQ),
      make_layout(make_shape(Int<BlockQO>{}, Int<HeadDim>{}), GenRowMajor{}));
  auto sK = make_tensor(
      make_smem_ptr(psK),
      make_layout(make_shape(Int<BlockKV>{}, Int<HeadDim>{}), GenRowMajor{}));
  auto sV = make_tensor(
      make_smem_ptr(psV),
      make_layout(make_shape(Int<BlockKV>{}, Int<HeadDim>{}), GenRowMajor{}));
  auto sVt = make_tensor(
      make_smem_ptr(psV),
      make_layout(make_shape(Int<HeadDim>{}, Int<BlockKV>{}), GenColMajor{}));

  TiledCopy tiled_copy;
  auto thr_copy = tiled_copy.get_slice(tx);
  auto tQgQ = thr_copy.partition_S(gQ); // (Copy, BlockQOCopy, HeadDimCopy)
  auto tQsQ = thr_copy.partition_D(sQ); // (Copy, BlockQOCopy, HeadDimCopy)
  auto tKsK = thr_copy.partition_D(sK); // (Copy, BlockKVCopy, HeadDimCopy)
  auto tKgK =
      thr_copy.partition_S(gK); // (Copy, BlockKVCopy, HeadDimCopy, RestKV)
  auto tVsV = thr_copy.partition_D(sV);
  auto tVgV =
      thr_copy.partition_S(gV); // (Copy, BlockKVCopy, HeadDimCopy, RestKV)

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tx);
  auto tSrQ = thr_mma.partition_fragment_A(sQ); // (MMA, MMA_QO, MMA_HEAD)
  auto tSrK = thr_mma.partition_fragment_B(sK); // (MMA, MMA_KV, MMA_HEAD)
  auto tSrS = partition_fragment_C(
      tiled_mma, Shape<Int<BlockQO>, Int<BlockKV>>{}); // (MMA, MMA_QO, MMA_KV)
  auto tOrVt = thr_mma.partition_fragment_B(sVt); // (MMA, MMA_Headdim, MMA_KV)
  auto tOrO = partition_fragment_C(
      tiled_mma,
      Shape<Int<BlockQO>, Int<HeadDim>>{}); // (MMA, MMA_QO, MMA_Headdim)
  clear(tOrO);

  auto tiled_s2r_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto thr_s2r_copy_Q = tiled_s2r_copy_Q.get_slice(tx);
  auto tXsQ = thr_s2r_copy_Q.partition_S(sQ);
  auto tXrQ = thr_s2r_copy_Q.retile_D(tSrQ); // (CPY, MMA_QO, MMA_HEAD)
  auto tiled_s2r_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto thr_s2r_copy_K = tiled_s2r_copy_K.get_slice(tx);
  auto tXsK = thr_s2r_copy_K.partition_S(sK);
  auto tXrK = thr_s2r_copy_K.retile_D(tSrK); // (CPY, MMA_KV, MMA_HEAD)
  auto tiled_s2r_copy_V =
      make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
  auto thr_s2r_copy_V = tiled_s2r_copy_V.get_slice(tx);
  auto tXsVt = thr_s2r_copy_V.partition_S(sVt);
  auto tXrVt = thr_s2r_copy_V.retile_D(tOrVt); // (CPY, MMA_Headdim, MMA_QO)

#ifdef FLASH_ATTN_MMA_DEBUG
  if (thread0()) { // clang-format off
    print("NumThreads: "); print(FlashAttnConfig_::NumThreads); print("\n");
    print("tiled_mma: "); print(tiled_mma); print("\n");
    print("tiled_copy: "); print(tiled_copy); print("\n");
    print("GmemValsPerLoad: "); print(FlashAttnConfig_::GmemValsPerLoad); print("\n");
    print("GmemThreadsPerRow: "); print(FlashAttnConfig_::GmemThreadsPerRow); print("\n");
    print("gQ: "); print(gQ.layout()); print("\n");
    print("gK: "); print(gK.layout()); print("\n");
    print("gV: "); print(gV.layout()); print("\n");
    print("sQ: "); print(sQ.layout()); print("\n");
    print("sK: "); print(sK.layout()); print("\n");
    print("sV: "); print(sV.layout()); print("\n");

    print("tQgQ: "); print(tQgQ.layout()); print("\n");
    print("tQsQ: "); print(tQsQ.layout()); print("\n");
    print("tKsK: "); print(tKsK.layout()); print("\n");
    print("tKgK: "); print(tKgK.layout()); print("\n");
    print("tVsV: "); print(tVsV.layout()); print("\n");

    print("tSrQ: "); print(tSrQ.layout()); print("\n");
    print("tSrK: "); print(tSrK.layout()); print("\n");
    print("tSrS: "); print(tSrS.layout()); print("\n");
    print("tOrVt: "); print(tOrVt.layout()); print("\n");
    print("tOrO: "); print(tOrO.layout()); print("\n");

    print("tiled_s2r_copy_Q: "); print(tiled_s2r_copy_Q); print("\n");
    print("tXsQ: "); print(tXsQ.layout()); print("\n");
    print("tXrQ: "); print(tXrQ.layout()); print("\n");
    print("tiled_s2r_copy_K: "); print(tiled_s2r_copy_K); print("\n");
    print("tXsK: "); print(tXsK.layout()); print("\n");
    print("tXrK: "); print(tXrK.layout()); print("\n");
    print("tiled_s2r_copy_V: "); print(tiled_s2r_copy_V); print("\n");
    print("tXsVt: "); print(tXsVt.layout()); print("\n");
    print("tXrVt: "); print(tXrVt.layout()); print("\n");
  } // clang-format on
#endif
  // NOTE: for sm80 MMA, each thread owns 2 rows of C matrix, they are
  // [v0, v1]
  // ......
  // [v2, v3]
  auto prev_row_max =
      make_tensor<float>(make_shape(_2{}, Int<size<1>(tSrS)>{}));
  fill(prev_row_max, -1e20);
  auto global_row_denominator =
      make_tensor<float>(make_shape(_2{}, Int<size<1>(tSrS)>{}));
  fill(global_row_denominator, 0);
  // copy Q into smem
  copy(tiled_copy, tQgQ, tQsQ);
  // scale Q first
  for (int i = 0; i < size(tQsQ); i++) {
    tQsQ(i) = static_cast<T>(scaler) * tQsQ(i);
  }
  __syncthreads();
  // copy Q into rmem
  copy(tiled_s2r_copy_Q, tXsQ, tXrQ);
  for (int blkKVIdx = 0; blkKVIdx < size<2>(gK); ++blkKVIdx) {
    // copy K into smem
    __syncthreads();
    copy(tiled_copy, tKgK(_, _, _, blkKVIdx), tKsK);
    __syncthreads();
    // copy K into rmem
    copy(tiled_s2r_copy_K, tXsK, tXrK);
#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) { // clang-format off
      print("blkKVIdx: "); print(blkKVIdx); print("\n");
      print("tXrQ: "); print_tensor(tXrQ); print("\n");
      print("tSrQ: "); print_tensor(tSrQ); print("\n");
      print("tXrK: "); print_tensor(tXrK); print("\n");
      print("tSrK: "); print_tensor(tSrK); print("\n");
    } // clang-format on
#endif
    clear(tSrS);
    gemm(tiled_mma, tSrQ, tSrK, tSrS);

#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) { // clang-format off
      print("tSrS: "); print_tensor(tSrS); print("\n");
    } // clang-format on
#endif
    auto new_row_max = make_fragment_like(prev_row_max);
    fill(new_row_max, -1e20);
    // max local-reduce
    // for one mma we have v0, v1, v2, v3, calculate max(v0, v1) and max(v2, v3)
    for (int val_idx = 0; val_idx < size<0>(tSrS); ++val_idx) {
      for (int row_rep_idx = 0; row_rep_idx < size<1>(tSrS); ++row_rep_idx) {
        for (int col_rep_idx = 0; col_rep_idx < size<2>(tSrS); ++col_rep_idx) {
          int row_idx = val_idx / 2;
          new_row_max(row_idx, row_rep_idx) =
              max(new_row_max(row_idx, row_rep_idx),
                  tSrS(val_idx, row_rep_idx, col_rep_idx));
        }
      }
    }
#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) { // clang-format off
      print("local new_row_max: "); print_tensor(new_row_max); print("\n");
    } // clang-format on
#endif
    // max quad-reduce (4 threads span one row of MMA C matrix for this
    // MMA_Atom)
    for (int row_idx = 0; row_idx < size<0>(new_row_max); ++row_idx) {
      for (int row_rep_idx = 0; row_rep_idx < size<1>(tSrS); ++row_rep_idx) {
        new_row_max(row_idx, row_rep_idx) = max(
            new_row_max(row_idx, row_rep_idx),
            __shfl_xor_sync(0xffffffff, new_row_max(row_idx, row_rep_idx),
                            1)); // shuffle reduce order shouldn't matter here
        new_row_max(row_idx, row_rep_idx) = max(
            new_row_max(row_idx, row_rep_idx),
            __shfl_xor_sync(0xffffffff, new_row_max(row_idx, row_rep_idx),
                            2)); // shuffle reduce order shouldn't matter here
      }
    }
#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) { // clang-format off
      print("quad new_row_max: "); print_tensor(new_row_max); print("\n");
    } // clang-format on
#endif
    // calculate new max
    for (int row_idx = 0; row_idx < size<0>(new_row_max); ++row_idx) {
      for (int row_rep_idx = 0; row_rep_idx < size<1>(new_row_max);
           ++row_rep_idx) {
        new_row_max(row_idx, row_rep_idx) =
            max(prev_row_max(row_idx, row_rep_idx),
                new_row_max(row_idx, row_rep_idx));
      }
    }
#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) { // clang-format off
      print("new_row_max: "); print_tensor(new_row_max); print("\n");
    } // clang-format on
#endif
    // scale nuemrator
    for (int val_idx = 0; val_idx < size<0>(tOrO); ++val_idx) {
      for (int row_rep_idx = 0; row_rep_idx < size<1>(tOrO); ++row_rep_idx) {
        for (int col_rep_idx = 0; col_rep_idx < size<2>(tOrO); ++col_rep_idx) {
          int row_idx = val_idx / 2;
          tOrO(val_idx, row_rep_idx, col_rep_idx) *=
              exp(prev_row_max(row_idx, row_rep_idx) -
                  new_row_max(row_idx, row_rep_idx));
        }
      }
    }
    // scale denominator
    for (int row_idx = 0; row_idx < size<0>(new_row_max); ++row_idx) {
      for (int row_rep_idx = 0; row_rep_idx < size<1>(new_row_max);
           ++row_rep_idx) {
        global_row_denominator(row_idx, row_rep_idx) *=
            exp(prev_row_max(row_idx, row_rep_idx) -
                new_row_max(row_idx, row_rep_idx));
      }
    }

    // apply new max and exp and accumulate to denominator
    for (int val_idx = 0; val_idx < size<0>(tSrS); ++val_idx) {
      for (int row_rep_idx = 0; row_rep_idx < size<1>(tSrS); ++row_rep_idx) {
        for (int col_rep_idx = 0; col_rep_idx < size<2>(tSrS); ++col_rep_idx) {
          int row_idx = val_idx / 2;
          tSrS(val_idx, row_rep_idx, col_rep_idx) =
              exp(tSrS(val_idx, row_rep_idx, col_rep_idx) -
                  new_row_max(row_idx, row_rep_idx));
          global_row_denominator(row_idx, row_rep_idx) +=
              tSrS(val_idx, row_rep_idx, col_rep_idx);
        }
      }
    }
#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) { // clang-format off
      print("scaled tSrS: "); print_tensor(tSrS); print("\n");
      print("global_row_denominator: "); print_tensor(global_row_denominator); print("\n");
    } // clang-format on
#endif
    // update global max
    for (int row_idx = 0; row_idx < size<0, 0>(tSrS); ++row_idx) {
      for (int row_rep_idx = 0; row_rep_idx < size<1>(tSrS); ++row_rep_idx) {
        prev_row_max(row_idx, row_rep_idx) = new_row_max(row_idx, row_rep_idx);
      }
    }
    auto tOrS = make_tensor<T>(tSrS.layout());
    for (int i = 0; i < size(tOrS); ++i) {
      tOrS(i) = static_cast<T>(tSrS(i));
    }
    // calculate numerator
    static_assert(tiled_mma.get_layoutA_TV() == tiled_mma.get_layoutC_TV(),
                  "This is only valid for atom mnk == (16, 8, 8), otherwise we "
                  "will have different A and C layout and need to adjust the "
                  "layout accordingly");
    __syncthreads();
    copy(tiled_copy, tVgV(_, _, _, blkKVIdx), tVsV);
    __syncthreads();

    copy(tiled_s2r_copy_V, tXsVt, tXrVt);
#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) { // clang-format off
      print("tOrVt: "); print_tensor(tOrVt); print("\n");
    } // clang-format on
#endif
    gemm(tiled_mma, tOrS, tOrVt, tOrO);

#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) { // clang-format off
      print("tOrO: "); print_tensor(tOrO); print("\n");
    } // clang-format on
#endif
  }
  // denominator quad-reduce
  for (int row_idx = 0; row_idx < size<0, 0>(tSrS); ++row_idx) {
    for (int row_rep_idx = 0; row_rep_idx < size<1>(tSrS); ++row_rep_idx) {
      global_row_denominator(row_idx, row_rep_idx) += __shfl_xor_sync(
          0xffffffff, global_row_denominator(row_idx, row_rep_idx), 1);
      global_row_denominator(row_idx, row_rep_idx) += __shfl_xor_sync(
          0xffffffff, global_row_denominator(row_idx, row_rep_idx), 2);
    }
  }
  // apply denominator
  for (int val_idx = 0; val_idx < size<0>(tOrO); ++val_idx) {
    for (int row_rep_idx = 0; row_rep_idx < size<1>(tOrO); ++row_rep_idx) {
      for (int col_rep_idx = 0; col_rep_idx < size<2>(tOrO); ++col_rep_idx) {
        int row_idx = val_idx / 2;
        tOrO(val_idx, row_rep_idx, col_rep_idx) /=
            global_row_denominator(row_idx, row_rep_idx);
      }
    }
  }
#ifdef FLASH_ATTN_MMA_DEBUG
  if (thread0()) { // clang-format off
    print("global_row_denominator: "); print_tensor(global_row_denominator); print("\n");
    print("tOrO: "); print_tensor(tOrO); print("\n");
  } // clang-format on
#endif
  // copy O back to gmem
  auto tiled_r2s_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
  auto thr_r2s_copy_O = tiled_r2s_copy_O.get_slice(tx);
  auto tXrO = thr_r2s_copy_O.retile_S(tOrO);
  auto tXsO = thr_r2s_copy_O.partition_D(gO);
  copy(tiled_r2s_copy_O, tXrO, tXsO);
}

// this kernel only implement limited functionality
static bool sanity_check(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                         torch::Tensor O) {
  const int bq = Q.size(0); // B, H, N, d
  const int hq = Q.size(1);
  const int nq = Q.size(2);
  const int dq = Q.size(3);
  const int bk = K.size(0); // B, H, N, d
  const int hk = K.size(1);
  const int nk = K.size(2);
  const int dk = K.size(3);
  const int bv = V.size(0); // B, H, N, d
  const int hv = V.size(1);
  const int nv = V.size(2);
  const int dv = V.size(3);
  const int bo = O.size(0); // B, H, N, d
  const int ho = O.size(1);
  const int no = O.size(2);
  const int do_ = O.size(3);
  if (!(bq == bk && bq == bv && bq == bo)) {
    printf("batch size mismatch: %d %d %d %d\n", bq, bk, bv, bo);
    fflush(stdout);
    return false;
  }
  if (!(hq == hk && hq == hv && hq == ho)) {
    printf("head size mismatch: %d %d %d %d\n", hq, hk, hv, ho);
    fflush(stdout);
    return false;
  }
  if (!(nq == nk && nq == nv && nq == no)) {
    printf("sequence length mismatch: %d %d %d %d, only self-attn is tested\n",
           nq, nk, nv, no);
    fflush(stdout);
    return false;
  }
  if (!(dq == dk && dq == dv && dq == do_)) {
    printf("hidden size mismatch: %d %d %d %d\n", dq, dk, dv, do_);
    fflush(stdout);
    return false;
  }
  return true;
}

template <int BlockQO, int BlockKV, int HeadDim, int NWarpsPerSM>
static void launch_kernel(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                          torch::Tensor O) {
  using config =
      FlashAttnConfig<half_t, BlockQO, BlockKV, HeadDim, NWarpsPerSM>;

  assert(sanity_check(Q, K, V, O));
  const int b = Q.size(0); // B, H, N, d
  const int h = Q.size(1);
  const int n = Q.size(2);
  const int d = Q.size(3);

  float scaler = 1.0 / sqrt(d);

  assert(n % BlockQO == 0);
  dim3 block(size(config::NumThreads));
  dim3 grid(b, h, n / BlockQO);
  flash_attn_cute_kernel<config><<<grid, block>>>(
      reinterpret_cast<half_t *>(Q.data_ptr()),
      reinterpret_cast<half_t *>(K.data_ptr()),
      reinterpret_cast<half_t *>(V.data_ptr()),
      reinterpret_cast<half_t *>(O.data_ptr()), b, h, n, n, d, scaler);
  CUDA_CHECK(cudaGetLastError());
}

void flash_attn_cute(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                     torch::Tensor O) {
  CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf) // Q [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf) // K [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf) // V [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(O, torch::kHalf) // O [B,H,N,D]
  const int d = Q.size(3);

  switch (d) { // NOTE: just naive heuristic, need tuning to find the best
               // configuration
  case 16:
    launch_kernel<128, 128, 16, 8>(Q, K, V, O);
    break;
  case 32:
    launch_kernel<128, 128, 32, 8>(Q, K, V, O);
    break;
  case 64:
    launch_kernel<128, 128, 64, 8>(Q, K, V, O);
    break;
  case 128:
    launch_kernel<64, 64, 128, 4>(Q, K, V, O);
    break;
  case 256:
    launch_kernel<32, 32, 256, 2>(Q, K, V, O);
    break;
  default:
    throw std::runtime_error("Unsupported headdim");
  }
}
