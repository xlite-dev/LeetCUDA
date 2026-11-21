#pragma once

#include <algorithm>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cutlass/numeric_types.h>

#include "common.h"

namespace svdq::cute_kernels {

using namespace cute;

// Fused SVDQ-FP8 activation producer: per-token E4M3 quantization plus the BF16 LoRA-down
// projection, from a single read of the activation tile.
//
//   sa[m]   = max(amax_k |X[m,k] / s[k]|, eps) / 448
//   qA[m,k] = satfinite_e4m3((X[m,k] / s[k]) / sa[m])
//   D[m,r]  = BF16(sum_k X[m,k] * lora_down[r,k])
//
// `lora_down` is L2/s, so D consumes the *raw* activation while qA consumes the smoothed one; both
// come from the same shared-memory tile. D accumulates in FP32 across the whole K loop and is
// rounded to BF16 exactly once. There is no standalone down-projection GEMM anywhere.
//
// Two kernels share one entry point. The first computes each row's amax with one CTA per row, the
// second lets one CTA own 128 rows and the whole K extent: it streams 64-wide K tiles, quantizes
// them, and accumulates the BF16 MMA for D. M must be a multiple of 128 and K a multiple of 64.
// The low-rank branch is never a standalone GEMM: D and the quantized activation come from the
// same tiled pass over the activation.

constexpr int kW8A8QuantThreads = 256;
constexpr int kW8A8QuantBM = 128;
constexpr int kW8A8QuantBK = 64;
constexpr float kW8A8QuantEps = 1e-12f;

using W8A8QuantSmemAtom = UMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>;
using W8A8QuantSmemLayoutX =
  decltype(tile_to_shape(W8A8QuantSmemAtom{}, Shape<Int<kW8A8QuantBM>, Int<kW8A8QuantBK>>{}));

union W8A8Vec8BF16 {
  uint4 raw;
  cutlass::bfloat16_t elem[8];
};

__device__ __forceinline__ float w8a8_warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
  }
  return value;
}

// Convert eight smoothed FP32 values with the same RTN/satfinite instruction the cache-dit FP8
// producer uses. Reciprocal multiplication and fast-math division may flip one E4M3 ULP.
__device__ __forceinline__ void w8a8_store_e4m3x8(uint8_t *output, float const *values,
                                                  float inv_scale) {
  uint2 packed;
  auto *halves = reinterpret_cast<unsigned short *>(&packed);
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    halves[i / 2] = __nv_cvt_float2_to_fp8x2(
      make_float2(values[i] * inv_scale, values[i + 1] * inv_scale), __NV_SATFINITE, __NV_E4M3);
  }
  *reinterpret_cast<uint2 *>(output) = packed;
}

template <int kRank>
struct W8A8QuantTraits {
  static constexpr int kBM = kW8A8QuantBM;
  static constexpr int kBK = kW8A8QuantBK;
  static constexpr int kThreads = kW8A8QuantThreads;
  static constexpr int kXBytes = kBM * kBK * 2;
  static constexpr int kLBytes = kRank * kBK * 2;
  static constexpr int kSmemBytes = kXBytes + kLBytes;

  using Element = cutlass::bfloat16_t;
  using MmaAtom = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
  using AtomLayoutMNK = Layout<Shape<_4, _2, _1>>;
  using PermutationMNK = Tile<Int<kBM>, Int<(kRank > 0 ? kRank : 16)>, Int<kBK>>;
  using TiledMma = decltype(make_tiled_mma(MmaAtom{}, AtomLayoutMNK{}, PermutationMNK{}));
  using SmemLayoutX = W8A8QuantSmemLayoutX;
  using SmemLayoutL =
    decltype(tile_to_shape(W8A8QuantSmemAtom{}, Shape<Int<(kRank > 0 ? kRank : 8)>, Int<kBK>>{}));
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
};

// Per-token amax, one CTA per row and eight warps splitting K. This mirrors the cache-dit FP8
// producer's parallel decomposition: a row-per-CTA grid exposes enough independent loads to hide
// HBM latency, which a 128-row tile cannot do on its own.
static __global__ void __launch_bounds__(kW8A8QuantThreads, 1)
  quantize_w8a8_amax_kernel(cutlass::bfloat16_t const *__restrict__ input,
                            cutlass::bfloat16_t const *__restrict__ smooth,
                            float *__restrict__ ascales, int K) {
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const auto *row_ptr = input + static_cast<int64_t>(blockIdx.x) * K;
  __shared__ float warp_amax[kW8A8QuantThreads / 32];
  float amax = 0.0f;
  for (int col = tid * 8; col < K; col += kW8A8QuantThreads * 8) {
    W8A8Vec8BF16 x, s;
    x.raw = *reinterpret_cast<uint4 const *>(row_ptr + col);
    s.raw = *reinterpret_cast<uint4 const *>(smooth + col);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float value = static_cast<float>(x.elem[i]) / static_cast<float>(s.elem[i]);
      amax = fmaxf(amax, fabsf(value));
    }
  }
  amax = w8a8_warp_max(amax);
  if (lane == 0) warp_amax[warp] = amax;
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
#pragma unroll
    for (int i = 0; i < kW8A8QuantThreads / 32; ++i) total = fmaxf(total, warp_amax[i]);
    ascales[blockIdx.x] = fmaxf(total, kW8A8QuantEps) / 448.0f;
  }
}

template <int kRank>
__global__ void __launch_bounds__(kW8A8QuantThreads, 1)
  quantize_w8a8_kernel(cutlass::bfloat16_t const *__restrict__ input,
                       cutlass::bfloat16_t const *__restrict__ smooth,
                       cutlass::bfloat16_t const *__restrict__ lora_down,
                       uint8_t *__restrict__ output, float *__restrict__ ascales,
                       cutlass::bfloat16_t *__restrict__ lora_act, float *__restrict__ lora_scratch,
                       int K, int tiles_per_split) {
  using Traits = W8A8QuantTraits<kRank>;
  constexpr int kBM = Traits::kBM;
  constexpr int kBK = Traits::kBK;
  const int tid = threadIdx.x;
  const int m0 = blockIdx.x * kBM;
  const int nKt_total = K / kBK;
  const int kt_begin = blockIdx.y * tiles_per_split;
  const int kt_end = min(nKt_total, kt_begin + tiles_per_split);
  if (kt_begin >= kt_end) return;

  extern __shared__ __align__(1024) uint8_t shm[];
  auto *x_bytes = reinterpret_cast<cutlass::bfloat16_t *>(shm);
  auto *l_bytes = reinterpret_cast<cutlass::bfloat16_t *>(shm + Traits::kXBytes);

  auto sX = make_tensor(make_smem_ptr(x_bytes), typename Traits::SmemLayoutX{});
  auto sL = make_tensor(make_smem_ptr(l_bytes), typename Traits::SmemLayoutL{});

  // The per-token scales arrive from the amax kernel; this CTA only caches its own 128 rows and
  // their reciprocals, so the quantization loop below is pure multiplies.
  __shared__ float row_inv_scale[kBM];
  if (tid < kBM) row_inv_scale[tid] = 1.0f / ascales[m0 + tid];
  __syncthreads();

  typename Traits::TiledMma mma;
  auto thr_mma = mma.get_thread_slice(tid);
  auto tCrD = partition_fragment_C(mma, Shape<Int<kBM>, Int<(kRank > 0 ? kRank : 16)>>{});
  clear(tCrD);
  auto tCrX = thr_mma.partition_fragment_A(sX);
  auto tCrL = thr_mma.partition_fragment_B(sL);
  auto s2r_copy_x = make_tiled_copy_A(typename Traits::SmemCopyAtom{}, mma);
  auto s2r_copy_l = make_tiled_copy_B(typename Traits::SmemCopyAtom{}, mma);
  auto s2r_thr_x = s2r_copy_x.get_thread_slice(tid);
  auto s2r_thr_l = s2r_copy_l.get_thread_slice(tid);
  auto tCsX = s2r_thr_x.partition_S(as_position_independent_swizzle_tensor(sX));
  auto tCsL = s2r_thr_l.partition_S(as_position_independent_swizzle_tensor(sL));
  auto tCrX_view = s2r_thr_x.retile_D(tCrX);
  auto tCrL_view = s2r_thr_l.retile_D(tCrL);

  constexpr int kChunksPerRow = kBK / 8;
  constexpr int kXChunks = kBM * kChunksPerRow;
  constexpr int kLChunks = (kRank > 0 ? kRank : 1) * kChunksPerRow;
  constexpr int kXPerThread = (kXChunks + kW8A8QuantThreads - 1) / kW8A8QuantThreads;
  constexpr int kLPerThread = (kLChunks + kW8A8QuantThreads - 1) / kW8A8QuantThreads;

  // Software pipeline: tile kt+1 is loaded into a second register set right after tile kt reaches
  // shared memory, so its HBM latency overlaps the MMA and the quantization of tile kt. Without
  // this, 128-row CTAs expose the full memory latency on every one of the K/64 iterations.
  uint4 x_regs[2][kXPerThread];
  uint4 l_regs[2][kLPerThread];
  auto load_tile = [&](int kt, auto buf_tag) {
    constexpr int buf = decltype(buf_tag)::value;
    const int k0 = kt * kBK;
#pragma unroll
    for (int i = 0; i < kXPerThread; ++i) {
      const int chunk = tid + i * kW8A8QuantThreads;
      if constexpr (kXChunks % kW8A8QuantThreads != 0) {
        if (chunk >= kXChunks) break;
      }
      const int row = chunk / kChunksPerRow;
      const int col = (chunk % kChunksPerRow) * 8;
      x_regs[buf][i] =
        *reinterpret_cast<uint4 const *>(input + static_cast<int64_t>(m0 + row) * K + k0 + col);
    }
    if constexpr (kRank > 0) {
#pragma unroll
      for (int i = 0; i < kLPerThread; ++i) {
        const int chunk = tid + i * kW8A8QuantThreads;
        if constexpr (kLChunks % kW8A8QuantThreads != 0) {
          if (chunk >= kLChunks) break;
        }
        const int row = chunk / kChunksPerRow;
        const int col = (chunk % kChunksPerRow) * 8;
        l_regs[buf][i] =
          *reinterpret_cast<uint4 const *>(lora_down + static_cast<int64_t>(row) * K + k0 + col);
      }
    }
  };

  auto process_tile = [&](int kt, auto buf_tag) {
    constexpr int buf = decltype(buf_tag)::value;
    const int k0 = kt * kBK;
    // Stage the raw BF16 values for the MMA. Consecutive threads take consecutive 16 B chunks
    // inside a row, so both the gmem read and the E4M3 write stay coalesced per eight-lane group.
#pragma unroll
    for (int i = 0; i < kXPerThread; ++i) {
      const int chunk = tid + i * kW8A8QuantThreads;
      if constexpr (kXChunks % kW8A8QuantThreads != 0) {
        if (chunk >= kXChunks) break;
      }
      const int row = chunk / kChunksPerRow;
      const int col = (chunk % kChunksPerRow) * 8;
      *reinterpret_cast<uint4 *>(&sX(row, col)) = x_regs[buf][i];
    }
    if constexpr (kRank > 0) {
#pragma unroll
      for (int i = 0; i < kLPerThread; ++i) {
        const int chunk = tid + i * kW8A8QuantThreads;
        if constexpr (kLChunks % kW8A8QuantThreads != 0) {
          if (chunk >= kLChunks) break;
        }
        const int row = chunk / kChunksPerRow;
        const int col = (chunk % kChunksPerRow) * 8;
        *reinterpret_cast<uint4 *>(&sL(row, col)) = l_regs[buf][i];
      }
    }
    if (kt + 1 < kt_end) load_tile(kt + 1, Int<buf ^ 1>{});
    __syncthreads();
    if constexpr (kRank > 0) {
      copy(s2r_copy_x, tCsX, tCrX_view);
      copy(s2r_copy_l, tCsL, tCrL_view);
      cute::gemm(mma, tCrX, tCrL, tCrD);
    }
#pragma unroll
    for (int i = 0; i < kXPerThread; ++i) {
      const int chunk = tid + i * kW8A8QuantThreads;
      if constexpr (kXChunks % kW8A8QuantThreads != 0) {
        if (chunk >= kXChunks) break;
      }
      const int row = chunk / kChunksPerRow;
      const int col = (chunk % kChunksPerRow) * 8;
      const int64_t offset = static_cast<int64_t>(m0 + row) * K + k0 + col;
      W8A8Vec8BF16 x, s;
      x.raw = x_regs[buf][i];
      s.raw = *reinterpret_cast<uint4 const *>(smooth + k0 + col);
      float values[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        values[j] = static_cast<float>(x.elem[j]) / static_cast<float>(s.elem[j]);
      }
      w8a8_store_e4m3x8(output + offset, values, row_inv_scale[row]);
    }
    __syncthreads();
  };

  load_tile(kt_begin, Int<0>{});
  for (int kt = kt_begin; kt < kt_end; kt += 2) {
    process_tile(kt, Int<0>{});
    if (kt + 1 < kt_end) process_tile(kt + 1, Int<1>{});
  }

  if constexpr (kRank > 0) {
    auto cD = make_identity_tensor(Shape<Int<kBM>, Int<kRank>>{});
    auto tScD = thr_mma.partition_C(cD);
#pragma unroll
    for (int i = 0; i < size(tCrD); ++i) {
      const int row = m0 + get<0>(tScD(i));
      const int col = get<1>(tScD(i));
      const int64_t index = static_cast<int64_t>(row) * kRank + col;
      if (lora_scratch != nullptr) {
        // Split-K: partial D fragments are combined with relaxed FP32 atomics and rounded once by
        // the finalize kernel, so D is still produced inside this operator.
        atomicAdd(lora_scratch + index, tCrD(i));
      } else {
        lora_act[index] = cutlass::bfloat16_t(__float2bfloat16_rn(tCrD(i)));
      }
    }
  }
}

// Round the FP32 split-K scratch into the BF16 D tensor. Only used when more than one K split
// contributes to the same row block.
static __global__ void quantize_w8a8_finalize_kernel(float const *__restrict__ lora_scratch,
                                                     cutlass::bfloat16_t *__restrict__ lora_act,
                                                     int64_t count) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    lora_act[index] = cutlass::bfloat16_t(__float2bfloat16_rn(lora_scratch[index]));
  }
}

/// Number of K splits for the fused producer, chosen to expose enough CTAs to hide HBM latency.
/// This is shared with the host dispatcher, which allocates the split-K scratch when it exceeds 1.
inline int w8a8_quant_k_splits(int M, int K) {
  const int row_blocks = std::max(1, M / kW8A8QuantBM);
  const int nKt = std::max(1, K / kW8A8QuantBK);
  const int target = (256 + row_blocks - 1) / row_blocks;
  return std::max(1, std::min(nKt, target));
}

/**
 * Launch the fused activation producer for one rank.
 *
 * :param input: BF16 activation [M,K].
 * :param smooth: BF16 per-input-channel divisor [K].
 * :param lora_down: BF16 [rank,K] down projection already divided by smooth; null for rank 0.
 * :param output: E4M3 destination [M,K].
 * :param ascales: FP32 per-token scales [M].
 * :param lora_act: BF16 LoRA-down output [M,rank]; null for rank 0.
 * :param M: Row count, a multiple of 128.
 * :param K: Channel count, a multiple of 64.
 * :param stream: CUDA stream for the launch.
 */
template <int kRank>
void launch_quantize_w8a8(void const *input, void const *smooth, void const *lora_down,
                          void *output, float *ascales, void *lora_act, float *lora_scratch, int M,
                          int K, cudaStream_t stream) {
  using Traits = W8A8QuantTraits<kRank>;
  // Rank 0 has no D to reduce, so it always splits K for the extra CTAs; ranked runs split exactly
  // when the dispatcher supplied the FP32 scratch that the finalize pass reduces.
  const int splits = kRank > 0 && lora_scratch == nullptr ? 1 : w8a8_quant_k_splits(M, K);
  float *scratch = splits > 1 ? lora_scratch : nullptr;
  const int nKt = K / kW8A8QuantBK;
  const int tiles_per_split = (nKt + splits - 1) / splits;
  quantize_w8a8_amax_kernel<<<M, kW8A8QuantThreads, 0, stream>>>(
    static_cast<cutlass::bfloat16_t const *>(input),
    static_cast<cutlass::bfloat16_t const *>(smooth), ascales, K);
  quantize_w8a8_kernel<kRank>
    <<<dim3(M / kW8A8QuantBM, splits), kW8A8QuantThreads, Traits::kSmemBytes, stream>>>(
      static_cast<cutlass::bfloat16_t const *>(input),
      static_cast<cutlass::bfloat16_t const *>(smooth),
      static_cast<cutlass::bfloat16_t const *>(lora_down), static_cast<uint8_t *>(output), ascales,
      static_cast<cutlass::bfloat16_t *>(lora_act), scratch, K, tiles_per_split);
  if (kRank > 0 && splits > 1) {
    const int64_t count = static_cast<int64_t>(M) * kRank;
    const int blocks = static_cast<int>((count + kW8A8QuantThreads - 1) / kW8A8QuantThreads);
    quantize_w8a8_finalize_kernel<<<blocks, kW8A8QuantThreads, 0, stream>>>(
      lora_scratch, static_cast<cutlass::bfloat16_t *>(lora_act), count);
  }
}

}  // namespace svdq::cute_kernels
