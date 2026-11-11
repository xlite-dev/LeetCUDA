#pragma once

#include "common.h"
#include "quantize.h"

#include <cute/tensor.hpp>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace svdq::cute_kernels {
namespace {

using namespace cute;

// Shared implementation for quantize_r{32,64,128}. Each public .cu file includes this header once
// and instantiates exactly one rank, so independent ranks can be compiled in parallel.
//
// The activation and LoRA paths deliberately consume the same unmodified BF16 A fragments:
//   qact/ascales = native packed NVFP4(input / smooth), with per-K32 E4M3 scales;
//   D            = input @ lora_down, accumulated in FP32 and written as row-major BF16.
// Smoothing is applied to temporary fragment copies only; it never perturbs the A used by MMA.
//
// Two schedules implement that contract:
//   fused:  one CTA owns 32 rows, warps split K, and FP32 D partials are reduced in shared memory;
//   split-K: one CTA owns kWarps*32 rows and one K chunk, FP32 D partials are reduced through
//            global scratch atomics, and a separate finalize kernel produces BF16 D.
// The kernels target SM120. Public validation requires M % 256 == 0, K % 64 == 0, contiguous BF16
// input/smooth/lora_down, and a specialized rank in {32,64,128}. All vectorized copies are 128-bit.

using Element = cutlass::bfloat16_t;

// Four 32-bit registers = eight BF16 values. This is both the A fragment consumed by one K16
// TiledMMA slice and the B fragment covering the two N8 atoms of an R16 tile.
struct PackedBF16 {
  __nv_bfloat162 data[4];
};

// One scale per R16 tile. The specialized implementation supports at most rank 128 = 8 tiles.
// The struct is passed by value to avoid an additional global-memory parameter lookup.
struct LoraScaleParams {
  float values[8];
};

// Reinterpret an MMA accumulator layout as a two-dimensional (row,column) layout without moving
// data. The leading fragment mode alternates adjacent rank columns; logical_divide separates that
// pair from the M16/N8 atom modes. The returned view supports acc(r,c), while partition_C supplies
// the corresponding logical coordinates. This is layout algebra only and has no runtime cost.
template <typename Layout>
CUTE_DEVICE auto convert_acc_rowcol(Layout acc_layout) {
  auto divided = logical_divide(acc_layout, Shape<_2>{});
  return make_layout(make_layout(get<0, 1>(divided), get<1>(divided)),
                     make_layout(get<0, 0>(divided), get<2>(divided)));
}

// CuTe geometry shared by fused and split-K. One warp computes a 32x16 output tile per R16 chunk;
// four K16 atoms cover a K64 slice. A/B fragments remain in registers, and D is streamed one R16
// chunk at a time in split-K so rank128 does not require a full 32x128 register accumulator.
struct QuantizeTraits {
  using MmaAtom = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
  using TiledMma =
    decltype(make_tiled_mma(MmaAtom{}, Layout<Shape<_1, _1, _1>>{}, Tile<_32, _16, _16>{}));

  // A is a 32x64 BF16 tile with K as the contiguous mode. SW128 supplies the swizzled address
  // mapping expected by LDSM_N while preserving 128-bit G2S stores.
  using SmemLayoutA =
    decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{}, Shape<_32, _64>{}));
  // B is viewed logically as [N=16,K=64], even though lora_down is row-major [K,R]. Its global
  // strides are therefore (1,R). The MN swizzle matches LDSM_T and avoids shared-memory conflicts.
  using SmemLayoutB =
    decltype(tile_to_shape(GMMA::Layout_MN_SW32_Atom<Element>{}, Shape<_16, _64>{}));

  // Synchronous 128-bit A copy used by fused and by the synchronous split-K schedule. The 4x8
  // thread arrangement covers 32 lanes and moves eight BF16 values per lane per copy step.
  using CopyA =
    decltype(make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint128_t>, Element>{},
                             Layout<Shape<_4, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{}));
  // Asynchronous variant for the double-buffered split-K A pipeline. Cache-global avoids keeping
  // streaming activation data in L1; completion is enforced by the caller's cp.async wait group.
  using CopyAAsync =
    decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
                             Layout<Shape<_4, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{}));
  // Fused B uses one warp per copy. Split-K B uses 128 threads; cache policy is rank-dependent.
  using CopyBWarp =
    decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
                             Layout<Shape<_2, _16>, Stride<_1, _2>>{}, Layout<Shape<_8, _1>>{}));
  using CopyBCtaGlobal =
    decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
                             Layout<Shape<_2, _64>, Stride<_1, _2>>{}, Layout<Shape<_8, _1>>{}));
  using CopyBCta =
    decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, Element>{},
                             Layout<Shape<_2, _64>, Stride<_1, _2>>{}, Layout<Shape<_8, _1>>{}));

  // A already has K-major MMA order and uses ordinary ldmatrix. B arrives as [N,K] and uses the
  // transposing form so its registers have the K-paired order required by the BF16 MMA atom.
  using S2RCopyA = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using S2RCopyB = Copy_Atom<SM75_U16x8_LDSM_T, Element>;

  // cosize includes every padding/swizzle offset in the physical shared allocation. It is used for
  // both per-warp buffer strides and total dynamic shared-memory sizing.
  static constexpr int kAChunkElems = decltype(cosize(SmemLayoutA{}))::value;
  static constexpr int kBChunkElems = decltype(cosize(SmemLayoutB{}))::value;
};

// Device helpers shared by both schedules. Keeping quantization and fragment conversion here makes
// the fused and split-K paths bit-compatible by construction rather than maintaining two formulas.
struct W4A4QuantizeRankKernel {
  /// Validate a host-side precondition and report the supplied message on failure.
  static void require(bool condition, const char *message) {
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  /// Reciprocal used only to form E2M1 inputs; persisted scales still use the original FP32 value.
  static __device__ __forceinline__ float rcp_approx(float x) {
    float result;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
  }

  /// Convert two FP32 values to packed E2M1 using round-to-nearest and finite saturation.
  /// The instruction consumes (high,low) = (value.y,value.x), matching the native byte order.
  static __device__ __forceinline__ uint32_t quantize_float2_fp4(float2 value) {
    uint32_t result;
    asm volatile("{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, %1, %2; cvt.u32.u8 %0, tmp; }"
                 : "=r"(result)
                 : "f"(value.y), "f"(value.x));
    return result;
  }

  /// Convert four FP32 scale values to four packed E4M3 bytes in one 32-bit word.
  static __device__ __forceinline__ uint32_t quantize_float4_fp8(float4 value) {
    uint16_t lo, hi;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(value.y), "f"(value.x));
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(value.w), "f"(value.z));
    return uint32_t(lo) | (uint32_t(hi) << 16);
  }

  /// Load two adjacent BF16 values; callers provide a 4-byte-aligned fragment/smooth address.
  static __device__ __forceinline__ __nv_bfloat162 load_bf16x2(const __nv_bfloat16 *ptr) {
    return *reinterpret_cast<const __nv_bfloat162 *>(ptr);
  }

  /// Divide two BF16 lanes through FP32 and round back to BF16, matching the native producer.
  static __device__ __forceinline__ __nv_bfloat162 div_bf16x2(__nv_bfloat162 a, __nv_bfloat162 b) {
    float2 af = __bfloat1622float2(a);
    float2 bf = __bfloat1622float2(b);
    return __float22bfloat162_rn(make_float2(__fdividef(af.x, bf.x), __fdividef(af.y, bf.y)));
  }

  /// Apply smooth to a temporary fragment. data[0:2] belong to K0..7 and use smooth0; data[2:4]
  /// belong to K8..15 and use smooth1. The original raw fragment remains available to LoRA MMA.
  static __device__ __forceinline__ PackedBF16 smooth_fragment(PackedBF16 value,
                                                               __nv_bfloat162 smooth0,
                                                               __nv_bfloat162 smooth1) {
    value.data[0] = div_bf16x2(value.data[0], smooth0);
    value.data[1] = div_bf16x2(value.data[1], smooth0);
    value.data[2] = div_bf16x2(value.data[2], smooth1);
    value.data[3] = div_bf16x2(value.data[3], smooth1);
    return value;
  }

  /// Fire-and-forget FP32 reduction of two adjacent rank columns. The launch zeroes scratch first;
  /// finalization applies scales and rounds only after all K CTAs have contributed.
  static __device__ __forceinline__ void global_add_f32x2(float *addr, float a, float b) {
    asm volatile("red.relaxed.gpu.global.add.v2.f32 [%0], {%1, %2};" ::"l"(addr), "f"(a), "f"(b));
  }

  // Quantize one M16xK64 activation fragment group and write the native qact/ascales pack.
  // tmp0..tmp3 are the four K16 fragments. For each fragment, data[0]/data[2] belong to the low
  // K32 panel and data[1]/data[3] to the high K32 panel; j selects one panel at a time.
  //
  // For every K32 panel:
  //   1. Rearrange four fragments into eight BF16x2 registers without touching memory.
  //   2. Compute four local abs-max values, then XOR-reduce the four lanes that share a rank group.
  //   3. Form four scales with min(absmax/6,448), persist them as E4M3, and quantize values as
  //   E2M1.
  //   4. Assemble four lane bytes with XOR shuffles, then store qact at native fragment positions.
  // The scale is capped before its reciprocal is taken so the E4M3 representation remains finite.
  // qact is packed at u32 granularity: each word contains eight FP4 values from one K16 group.
  static __device__ __forceinline__ void quantize_group(PackedBF16 tmp0, PackedBF16 tmp1,
                                                        PackedBF16 tmp2, PackedBF16 tmp3,
                                                        uint8_t *qact, uint8_t *ascales, int K,
                                                        int row0, int k0, int lane) {
    const PackedBF16 tmp[4] = {tmp0, tmp1, tmp2, tmp3};
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      // Register-only fragment permutation into the current K32 panel.
      __nv_bfloat162 input[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        input[i * 2] = j == 0 ? tmp[i].data[0] : tmp[i].data[1];
        input[i * 2 + 1] = j == 0 ? tmp[i].data[2] : tmp[i].data[3];
      }
      // Each candidate covers four BF16 values. XOR masks 2 and 1 reduce the four-lane rank group,
      // leaving every lane with the same K32 maximum without a shared-memory reduction.
      float maxvalue[4] = {};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float2 a = __bfloat1622float2(input[i * 2]);
        float2 b = __bfloat1622float2(input[i * 2 + 1]);
        maxvalue[i] = fmaxf(fabsf(a.x), fmaxf(fabsf(a.y), fmaxf(fabsf(b.x), fabsf(b.y))));
      }
#pragma unroll
      for (int mask = 2; mask > 0; mask /= 2) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          maxvalue[i] = fmaxf(maxvalue[i], __shfl_xor_sync(~0u, maxvalue[i], mask));
        }
      }
      float rscale[4];
      float scale[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        scale[i] = fminf(maxvalue[i] / 6.0f, 448.0f);
        rscale[i] = rcp_approx(scale[i]);
      }
      const uint32_t fp8scale =
        quantize_float4_fp8(make_float4(scale[0], scale[1], scale[2], scale[3]));
      // Each lane first places its E2M1 pair in one of four byte positions. XOR shuffles then OR
      // the four contributions, giving every lane the fully packed words without shared staging.
      uint32_t qpacks[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float2 fval = __bfloat1622float2(input[i]);
        fval.x *= rscale[i / 2];
        fval.y *= rscale[i / 2];
        qpacks[i] = quantize_float2_fp4(fval) << ((lane % 4) * 8);
      }
#pragma unroll
      for (int mask = 1; mask <= 2; mask *= 2) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          qpacks[i] |= __shfl_xor_sync(~0u, qpacks[i], mask);
        }
      }
      // Native 256-row pack decode:
      //   base  = (256-row block, K64 slice, 32-row warp panel);
      //   local = (row-within-8, 8/16 half, 16/32 half) in u32 words;
      //   i*2   = the two words owned by writer lane 0 or 1 of each four-lane group.
      // dst[0/4/8/12] advance to successive K16 groups. Lanes 2 and 3 intentionally do not store.
      const int i = lane % 4;
      const int row = row0 + lane / 4 + j * 8;
      if (i < 2) {
        const int local = 16 * (row % 8) + ((row % 16) / 8) + 128 * ((row % 32) / 16);
        const int64_t base =
          ((static_cast<int64_t>(row / 256) * (K / 64) + k0 / 64) * 8 + (row % 256) / 32) * 256;
        uint32_t *dst = reinterpret_cast<uint32_t *>(qact) + base + local + i * 2;
        if (i == 0) {
          dst[0] = qpacks[0];
          dst[4] = qpacks[1];
          dst[8] = qpacks[2];
          dst[12] = qpacks[3];
        } else {
          dst[0] = qpacks[4];
          dst[4] = qpacks[5];
          dst[8] = qpacks[6];
          dst[12] = qpacks[7];
        }
      }
      if (lane % 4 == 0) {
        const int64_t base =
          (static_cast<int64_t>(row / 256) * (K / 64) + k0 / 64) * 8 + (row % 256) / 32;
        const int scale_lane = 4 * (row % 8) + ((row % 32) / 8);
        reinterpret_cast<uint32_t *>(ascales)[base * 32 + scale_lane] = fp8scale;
      }
    }
  }

  // Synchronously load one logical [M32,K64] A tile into this warp's MMA fragments.
  // gA uses the original row stride K, while sA uses SmemLayoutA. TiledCopy partitions both views
  // into 128-bit G2S operations, and make_tiled_copy_A retiles the destination to LDSM-compatible
  // register fragments. The first __syncwarp publishes shared stores; the second prevents a later
  // stage reuse from overwriting data before every lane has completed its LDSM reads.
  // valid == false is a K-tail predicate: no global/shared access is issued and raw is zeroed.
  template <class TiledMma, class CopyA>
  static __device__ __forceinline__ auto load_a(const __nv_bfloat16 *input, int K, int row0, int k0,
                                                bool valid, Element *stage, TiledMma const &mma,
                                                CopyA const &copy_a) {
    auto gA = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element const *>(input + static_cast<size_t>(row0) * K + k0)),
      make_layout(Shape<_32, _64>{}, Stride<decltype(K), _1>{K, _1{}}));
    auto thr_mma = mma.get_thread_slice(threadIdx.x % 32);
    auto raw = thr_mma.partition_fragment_A(gA);
    if (valid) {
      auto s2r_a = make_tiled_copy_A(QuantizeTraits::S2RCopyA{}, mma);
      auto copy_thr = copy_a.get_thread_slice(threadIdx.x % 32);
      auto s2r_thr = s2r_a.get_thread_slice(threadIdx.x % 32);
      auto sA = make_tensor(make_smem_ptr(stage), QuantizeTraits::SmemLayoutA{});
      auto tAgA = copy_thr.partition_S(gA);
      auto tAsA = copy_thr.partition_D(sA);
      auto tAsA_s2r = s2r_thr.partition_S(sA);
      auto raw_view = s2r_thr.retile_D(raw);
      cute::copy(copy_a, tAgA, tAsA);
      __syncwarp();
      cute::copy(s2r_a, tAsA_s2r, raw_view);
      __syncwarp();
    } else {
      clear(raw);
    }
    return raw;
  }

  // Issue, but do not wait for, an asynchronous A tile copy. The position-independent swizzle view
  // expresses the same physical layout as read_a without baking the shared base address into layout
  // algebra. The caller owns the cp.async group fence and must wait before consuming the stage.
  static __device__ __forceinline__ void issue_a(const __nv_bfloat16 *input, int K, int row0,
                                                 int k0, bool valid, Element *stage) {
    if (!valid) return;
    typename QuantizeTraits::CopyAAsync copy_a;
    auto copy_thr = copy_a.get_thread_slice(threadIdx.x % 32);
    auto gA = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element const *>(input + static_cast<size_t>(row0) * K + k0)),
      make_layout(Shape<_32, _64>{}, Stride<decltype(K), _1>{K, _1{}}));
    auto sA = as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(stage), QuantizeTraits::SmemLayoutA{}));
    auto tAgA = copy_thr.partition_S(gA);
    auto tAsA = copy_thr.partition_D(sA);
    cute::copy(copy_a, tAgA, tAsA);
  }

  // Read an already-completed asynchronous A stage into registers. No cp.async wait is hidden here:
  // split-K deliberately waits only for A first so B can remain in flight during quantization.
  // Invalid slices still return a correctly shaped zero fragment for uniform MMA control flow.
  template <class TiledMma>
  static __device__ __forceinline__ auto read_a(const __nv_bfloat16 *input, int K, int row0, int k0,
                                                bool valid, Element *stage, TiledMma const &mma) {
    auto gA = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element const *>(input + static_cast<size_t>(row0) * K + k0)),
      make_layout(Shape<_32, _64>{}, Stride<decltype(K), _1>{K, _1{}}));
    auto thr_mma = mma.get_thread_slice(threadIdx.x % 32);
    auto raw = thr_mma.partition_fragment_A(gA);
    if (valid) {
      auto s2r_a = make_tiled_copy_A(QuantizeTraits::S2RCopyA{}, mma);
      auto s2r_thr = s2r_a.get_thread_slice(threadIdx.x % 32);
      auto sA = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(stage), QuantizeTraits::SmemLayoutA{}));
      auto tAsA = s2r_thr.partition_S(sA);
      auto raw_view = s2r_thr.retile_D(raw);
      cute::copy(s2r_a, tAsA, raw_view);
    } else {
      clear(raw);
    }
    return raw;
  }

  // Stage every R16xK64 B tile needed by this CTA. A 128-thread copy team handles successive K64
  // slices; slice_stride is one for a 128-thread CTA and two for a 256-thread CTA.
  // Shared slots use [rank_tile][slice] order, with kMaxSlices as the physical second-mode stride.
  // rank32 selects CACHEGLOBAL, while wider ranks retain the tuned CACHEALWAYS policy.
  // cp_async_fence closes the B group but does not wait for completion; the producer waits after
  // quantization so global B traffic overlaps activation loading and NVFP4 packing.
  template <int kRankTiles, int kMaxSlices>
  static __device__ __forceinline__ void issue_b_cta(const __nv_bfloat16 *lora_down, int k_begin,
                                                     int k_slices, Element *b_base) {
    constexpr int rank = kRankTiles * 16;
    using CopyB =
      std::conditional_t<kRankTiles == 2, QuantizeTraits::CopyBCtaGlobal, QuantizeTraits::CopyBCta>;
    CopyB copy_b;
    const int copy_tid = threadIdx.x % 128;
    const int slice_stride = blockDim.x / 128;
    auto copy_thr = copy_b.get_thread_slice(copy_tid);
    for (int slice = threadIdx.x / 128; slice < k_slices; slice += slice_stride) {
      for (int rt = 0; rt < rank / 16; ++rt) {
        auto gB =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element const *>(
                        lora_down + static_cast<size_t>(k_begin + slice * 64) * rank + rt * 16)),
                      make_layout(Shape<_16, _64>{}, Stride<_1, decltype(rank)>{_1{}, rank}));
        auto sB = as_position_independent_swizzle_tensor(make_tensor(
          make_smem_ptr(b_base + (rt * kMaxSlices + slice) * QuantizeTraits::kBChunkElems),
          QuantizeTraits::SmemLayoutB{}));
        auto tBgB = copy_thr.partition_S(gB);
        auto tBsB = copy_thr.partition_D(sB);
        cute::copy(copy_b, tBgB, tBsB);
      }
    }
    cp_async_fence();
  }

  // Fused-path B prefetch for one warp-local [R16,K64] tile. rank0 accounts for a CTA that owns
  // only half the logical rank. The fence gives this tile its own cp.async group; the consumer
  // waits immediately before LDSM and may issue the next tile into the other shared buffer.
  static __device__ __forceinline__ void issue_b_warp(const __nv_bfloat16 *lora_down, int rank,
                                                      int k0, int rank0, int rt, Element *stage) {
    typename QuantizeTraits::CopyBWarp copy_b;
    auto copy_thr = copy_b.get_thread_slice(threadIdx.x % 32);
    auto gB = make_tensor(make_gmem_ptr(reinterpret_cast<Element const *>(
                            lora_down + static_cast<size_t>(k0) * rank + rank0 + rt * 16)),
                          make_layout(Shape<_16, _64>{}, Stride<_1, decltype(rank)>{_1{}, rank}));
    auto sB = as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(stage), QuantizeTraits::SmemLayoutB{}));
    auto tBgB = copy_thr.partition_S(gB);
    auto tBsB = copy_thr.partition_D(sB);
    cute::copy(copy_b, tBgB, tBsB);
    cp_async_fence();
  }

  // Consume one staged R16xK64 B tile. LDSM_T fills all four K16 B fragments in one TiledCopy;
  // each is then multiplied by the matching A fragment and accumulated into the same FP32 C tile.
  // The caller guarantees that stage is complete before entry and remains live through this call.
  template <class TiledMma, class RawA, class Acc>
  static __device__ __forceinline__ void mma_b_chunk(Element const *stage, TiledMma const &mma,
                                                     RawA const &raw, Acc &acc) {
    auto s2r_b = make_tiled_copy_B(QuantizeTraits::S2RCopyB{}, mma);
    auto s2r_thr = s2r_b.get_thread_slice(threadIdx.x % 32);
    auto thr_mma = mma.get_thread_slice(threadIdx.x % 32);
    auto gB = make_tensor(make_gmem_ptr(static_cast<Element const *>(nullptr)),
                          make_layout(Shape<_16, _64>{}, Stride<_1, _64>{}));
    auto frag_b = thr_mma.partition_fragment_B(gB);
    auto sB = as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(stage), QuantizeTraits::SmemLayoutB{}));
    auto tBsB = s2r_thr.partition_S(sB);
    auto frag_b_view = s2r_thr.retile_D(frag_b);
    cute::copy(s2r_b, tBsB, frag_b_view);
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      cute::gemm(mma, acc, raw(_, _, n), frag_b(_, _, n), acc);
    }
  }
};

// Split-K producer and global FP32 reduction.
//
// Parallel decomposition:
//   grid.x = M/(kWarps*32): one row block per CTA;
//   grid.y = ceil(K/(kMaxSlices*64)): one independent K chunk per CTA;
//   warp   = one 32-row panel, split into two M16 halves;
//   rt     = one R16 rank tile, streamed to bound accumulator registers.
// kMaxSlices is two for K128 chunks and four for K256 chunks. A final K64 tail is handled by
// predicates rather than reading past K. qact/ascales have one writer per (row panel,K64), whereas
// every K CTA contributes to the same D scratch with relaxed FP32 vector atomics. Consequently
// qact/ascales are deterministic, but the bitwise order of D additions across K CTAs is not.
//
// Template roles:
//   kRankTiles = R/16; kWarps = warps/CTA; kMaxSlices = K64 slices/CTA;
//   kAsyncA selects the two-stage asynchronous A pipeline (otherwise synchronous A copies).
// The launcher zeroes M*R scratch floats before launch and invokes finalize_lora_bf16_kernel after
// this kernel on the same stream, ensuring all atomics complete before scaling and BF16 conversion.
template <int kRankTiles, int kWarps, int kMaxSlices = 2, bool kAsyncA = true>
__global__ void quantize_w4a4_split_k_kernel(const __nv_bfloat16 *__restrict__ input,
                                             const __nv_bfloat16 *__restrict__ smooth,
                                             const __nv_bfloat16 *__restrict__ lora_down,
                                             uint8_t *__restrict__ qact,
                                             uint8_t *__restrict__ ascales,
                                             float *__restrict__ lora_scratch, int M, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  static_assert(!(kAsyncA && kMaxSlices > 2), "async A pipeline supports at most two K64 slices");
  using Traits = QuantizeTraits;
  using Kernel = W4A4QuantizeRankKernel;
  constexpr int kRank = kRankTiles * 16;
  constexpr int kAStages = kAsyncA ? 2 : 1;
  constexpr int kAElements = kWarps * kAStages * Traits::kAChunkElems;
  constexpr int kKChunk = kMaxSlices * 64;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int k_begin = blockIdx.y * kKChunk;
  const int k_end = min(k_begin + kKChunk, K);
  const int k_slices = (k_end - k_begin) / 64;
  const int row0 = blockIdx.x * (kWarps * 32) + warp * 32;
  // Dynamic shared layout: [per-warp A stage(s)] [all CTA B tiles]. A and B are disjoint, so B
  // prefetch never overwrites activation staging. Async A uses two stages and forms group A;
  // issue_b_cta then forms group B. cp_async_wait<1> waits for A while allowing the newer B group
  // to remain in flight, and __syncwarp makes A visible to the warp's subsequent LDSM reads.
  extern __shared__ __align__(1024) Element shared[];
  Element *a_base0 = shared + warp * kAStages * Traits::kAChunkElems;
  Element *a_base1 = a_base0 + (kAStages - 1) * Traits::kAChunkElems;
  Element *b_base = shared + kAElements;
  if constexpr (kAsyncA) {
    Kernel::issue_a(input, K, row0, k_begin, true, a_base0);
    Kernel::issue_a(input, K, row0, k_begin + 64, k_slices > 1, a_base1);
    cp_async_fence();
  }
  Kernel::issue_b_cta<kRankTiles, kMaxSlices>(lora_down, k_begin, k_slices, b_base);
  if constexpr (kAsyncA) {
    cp_async_wait<1>();
    __syncwarp();
  }

  typename Traits::TiledMma mma;
  typename Traits::CopyA copy_a;
  auto read_or_load = [&](int slice, Element *stage) {
    const bool valid = slice < k_slices;
    if constexpr (kAsyncA) {
      return Kernel::read_a(input, K, row0, k_begin + slice * 64, valid, stage, mma);
    } else {
      return Kernel::load_a(input, K, row0, k_begin + slice * 64, valid, stage, mma, copy_a);
    }
  };
  // Keep A fragments in registers across quantization and LoRA MMA. Synchronous schedules safely
  // reuse a_base0 after load_a's trailing warp barrier; asynchronous schedules use the two stages.
  // For kMaxSlices == 2, raw2/raw3 are invalid zero fragments and are not consumed by the MMA loop.
  auto raw0 = read_or_load(0, a_base0);
  auto raw1 = read_or_load(1, a_base1);
  auto raw2 = read_or_load(2, a_base0);
  auto raw3 = read_or_load(3, a_base0);

  // Quantization reads temporary smoothed copies, leaving raw unchanged for D. After all slices are
  // packed, wait for every outstanding B copy and then use a CTA barrier to publish all B tiles to
  // every warp. This is the only cross-warp shared-memory synchronization in the producer.
  auto quantize_slice = [&](auto const &raw, int k0, bool valid) {
    if (!valid) return;
    __nv_bfloat162 smooth0[4];
    __nv_bfloat162 smooth1[4];
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const int col = k0 + n * 16 + (lane % 4) * 2;
      smooth0[n] = Kernel::load_bf16x2(smooth + col);
      smooth1[n] = Kernel::load_bf16x2(smooth + col + 8);
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      Kernel::quantize_group(
        Kernel::smooth_fragment(*reinterpret_cast<PackedBF16 const *>(raw(_, mt, 0).data()),
                                smooth0[0], smooth1[0]),
        Kernel::smooth_fragment(*reinterpret_cast<PackedBF16 const *>(raw(_, mt, 1).data()),
                                smooth0[1], smooth1[1]),
        Kernel::smooth_fragment(*reinterpret_cast<PackedBF16 const *>(raw(_, mt, 2).data()),
                                smooth0[2], smooth1[2]),
        Kernel::smooth_fragment(*reinterpret_cast<PackedBF16 const *>(raw(_, mt, 3).data()),
                                smooth0[3], smooth1[3]),
        qact, ascales, K, row0 + mt * 16, k0, lane);
    }
  };
  quantize_slice(raw0, k_begin, true);
  quantize_slice(raw1, k_begin + 64, k_slices > 1);
  if constexpr (kMaxSlices == 4) {
    quantize_slice(raw2, k_begin + 128, k_slices > 2);
    quantize_slice(raw3, k_begin + 192, k_slices > 3);
  }
  cp_async_wait<0>();
  __syncthreads();

  auto thr_mma = mma.get_thread_slice(lane);
  auto gC = make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                        make_layout(Shape<_32, _16>{}, Stride<_16, _1>{}));
  // Rank-chunk streaming keeps one 32xR16 FP32 accumulator live per warp. All valid K64 slices
  // contribute to that tile before it is reduced. packed_index uses fragment-major scratch order:
  //   (row_block,rank_tile,warp,atom) -> 32 lanes -> two adjacent rank columns per lane.
  // This ordering gives coalesced v2 atomics and is decoded by finalize_lora_bf16_kernel.
#pragma unroll
  for (int rt = 0; rt < kRankTiles; ++rt) {
    auto acc = thr_mma.partition_fragment_C(gC);
    clear(acc);
    Kernel::mma_b_chunk(b_base + (rt * kMaxSlices) * Traits::kBChunkElems, mma, raw0, acc);
    if (k_slices > 1) {
      Kernel::mma_b_chunk(b_base + (rt * kMaxSlices + 1) * Traits::kBChunkElems, mma, raw1, acc);
    }
    if constexpr (kMaxSlices == 4) {
      if (k_slices > 2) {
        Kernel::mma_b_chunk(b_base + (rt * kMaxSlices + 2) * Traits::kBChunkElems, mma, raw2, acc);
      }
      if (k_slices > 3) {
        Kernel::mma_b_chunk(b_base + (rt * kMaxSlices + 3) * Traits::kBChunkElems, mma, raw3, acc);
      }
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const int atom = mt * 4 + pair;
        const int packed_index =
          ((((blockIdx.x * kRankTiles + rt) * kWarps + warp) * 8 + atom) * 32 + lane) * 2;
        const int flat = mt * 4 + (pair / 2) * 8 + (pair % 2) * 2;
        Kernel::global_add_f32x2(lora_scratch + packed_index, acc(flat), acc(flat + 1));
      }
    }
  }
#else
  asm volatile("trap;");
#endif
}

// Fused producer: activation packing and LoRA D complete without a global scratch tensor.
//
// A CTA owns 32 rows and either the full rank or one half selected by blockIdx.y. Each warp walks
// K in strides of kWarps*64, maintains an independent FP32 partial for every local R16 tile, and
// later reduces those partials in shared memory. kRankSplits increases M-grid parallelism at the
// cost of reading input in both rank CTAs; only blockIdx.y == 0 writes qact/ascales to avoid
// duplicate activation stores. Every CTA still computes its own LoRA columns.
//
// Dynamic shared memory has three disjoint regions:
//   A: one 32x64 tile per warp;
//   B: two 16x64 tiles per warp for alternating rank-tile prefetch;
//   D: kReductionBuffers 32xR_local FP32 partials with a padded row stride.
// kReductionBuffers trades shared capacity for serialized deterministic reduction groups. The
// final cooperative pass sums those buffers, applies one scale per R16 tile, and rounds once to
// BF16. No K-CTA atomics or finalize launch are needed on this path.
template <int kRankTiles, int kWarps, int kReductionBuffers, int kRankSplits>
__global__ void __launch_bounds__(kWarps * 32, 1)
  quantize_w4a4_fused_lora_kernel(const __nv_bfloat16 *__restrict__ input,
                                  const __nv_bfloat16 *__restrict__ smooth,
                                  const __nv_bfloat16 *__restrict__ lora_down,
                                  uint8_t *__restrict__ qact, uint8_t *__restrict__ ascales,
                                  __nv_bfloat16 *__restrict__ lora_act, LoraScaleParams scales,
                                  int M, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  using Traits = QuantizeTraits;
  using Kernel = W4A4QuantizeRankKernel;
  static_assert(kRankSplits == 1 || kRankSplits == 2);
  static_assert(kWarps % kReductionBuffers == 0);
  constexpr int kRank = kRankTiles * 16;
  constexpr int kLocalRankTiles = kRankTiles / kRankSplits;
  constexpr int kLocalRank = kLocalRankTiles * 16;
  constexpr int kRankStride = kRankTiles == 2 ? kLocalRank + 4 : kLocalRank + 1;
  constexpr int kDStageFloats = 32 * kRankStride;
  constexpr int kAElements = kWarps * Traits::kAChunkElems;
  constexpr int kBElements = kWarps * 2 * Traits::kBChunkElems;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int row0 = blockIdx.x * 32;
  const int rank0 = kRankSplits == 1 ? 0 : blockIdx.y * kLocalRank;
  extern __shared__ __align__(1024) Element shared[];
  Element *a_base = shared + warp * Traits::kAChunkElems;
  Element *b_base = shared + kAElements + warp * 2 * Traits::kBChunkElems;
  float *d_base = reinterpret_cast<float *>(shared + kAElements + kBElements);

  typename Traits::TiledMma mma;
  typename Traits::CopyA copy_a;
  auto thr_mma = mma.get_thread_slice(lane);
  auto gC = make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                        make_layout(Shape<_32, _16>{}, Stride<_16, _1>{}));
  using Acc = decltype(thr_mma.partition_fragment_C(gC));
  Acc acc[kLocalRankTiles];
#pragma unroll
  for (int rt = 0; rt < kLocalRankTiles; ++rt) clear(acc[rt]);

  // Warp w owns K64 slices w, w+kWarps, ... . B tile zero is prefetched before synchronous A
  // staging, allowing its global traffic to overlap A LDSM and activation quantization. The raw
  // fragment remains live across both phases, so each input tile is read from global memory once.
  for (int k0 = warp * 64; k0 < K; k0 += kWarps * 64) {
    Kernel::issue_b_warp(lora_down, kRank, k0, rank0, 0, b_base);
    auto raw = Kernel::load_a(input, K, row0, k0, true, a_base, mma, copy_a);
    if (kRankSplits == 1 || blockIdx.y == 0) {
      __nv_bfloat162 smooth0[4];
      __nv_bfloat162 smooth1[4];
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const int col = k0 + n * 16 + (lane % 4) * 2;
        smooth0[n] = Kernel::load_bf16x2(smooth + col);
        smooth1[n] = Kernel::load_bf16x2(smooth + col + 8);
      }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        Kernel::quantize_group(
          Kernel::smooth_fragment(*reinterpret_cast<PackedBF16 const *>(raw(_, mt, 0).data()),
                                  smooth0[0], smooth1[0]),
          Kernel::smooth_fragment(*reinterpret_cast<PackedBF16 const *>(raw(_, mt, 1).data()),
                                  smooth0[1], smooth1[1]),
          Kernel::smooth_fragment(*reinterpret_cast<PackedBF16 const *>(raw(_, mt, 2).data()),
                                  smooth0[2], smooth1[2]),
          Kernel::smooth_fragment(*reinterpret_cast<PackedBF16 const *>(raw(_, mt, 3).data()),
                                  smooth0[3], smooth1[3]),
          qact, ascales, K, row0 + mt * 16, k0, lane);
      }
    }
    // Alternate two warp-private B stages. Wait only for this warp's outstanding copy, publish it
    // with __syncwarp, then issue rt+1 into the other stage before consuming rt with LDSM/MMA.
    // There is no cross-warp B sharing, so no CTA barrier is needed inside the K loop.
#pragma unroll
    for (int rt = 0; rt < kLocalRankTiles; ++rt) {
      const int buffer = rt & 1;
      cp_async_wait<0>();
      __syncwarp();
      if (rt + 1 < kLocalRankTiles) {
        Kernel::issue_b_warp(lora_down, kRank, k0, rank0, rt + 1,
                             b_base + (buffer ^ 1) * Traits::kBChunkElems);
      }
      Kernel::mma_b_chunk(b_base + buffer * Traits::kBChunkElems, mma, raw, acc[rt]);
    }
  }

  // Convert accumulator fragments to logical (row,column) coordinates for the shared reduction.
  // With fewer buffers than warps, groups are serialized by CTA barriers: w==0 initializes each
  // slot and later groups accumulate in place. With one buffer per warp, every warp writes its own
  // partial directly. rank32 uses col^row for bank-conflict avoidance; other ranks use a one-column
  // guard. The barriers also ensure warps with no K work participate safely with zero accumulators.
  auto cC = make_identity_tensor(Shape<_32, _16>{});
  auto tScC = thr_mma.partition_C(cC);
  auto tScC_rc = make_tensor(tScC.data(), convert_acc_rowcol(tScC.layout()));
  if constexpr (kReductionBuffers < kWarps) {
    __syncthreads();
#pragma unroll
    for (int w = 0; w < kWarps; w += kReductionBuffers) {
      if (warp >= w && warp < w + kReductionBuffers) {
        float *partial = d_base + (warp - w) * kDStageFloats;
#pragma unroll
        for (int rt = 0; rt < kLocalRankTiles; ++rt) {
          auto acc_rc = make_tensor(acc[rt].data(), convert_acc_rowcol(acc[rt].layout()));
          constexpr int kRows = decltype(size<0>(acc_rc))::value;
          constexpr int kCols = decltype(size<1>(acc_rc))::value;
#pragma unroll
          for (int r = 0; r < kRows; ++r) {
#pragma unroll
            for (int c = 0; c < kCols; ++c) {
              const int row = get<0>(tScC_rc(r, c));
              const int col = rt * 16 + get<1>(tScC_rc(r, c));
              const int shared_col = kRankTiles == 2 ? (col ^ row) : col;
              const int offset = row * kRankStride + shared_col;
              partial[offset] = w == 0 ? acc_rc(r, c) : partial[offset] + acc_rc(r, c);
            }
          }
        }
      }
      __syncthreads();
    }
  } else {
#pragma unroll
    for (int rt = 0; rt < kLocalRankTiles; ++rt) {
      auto acc_rc = make_tensor(acc[rt].data(), convert_acc_rowcol(acc[rt].layout()));
      constexpr int kRows = decltype(size<0>(acc_rc))::value;
      constexpr int kCols = decltype(size<1>(acc_rc))::value;
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
          const int row = get<0>(tScC_rc(r, c));
          const int col = rt * 16 + get<1>(tScC_rc(r, c));
          const int shared_col = kRankTiles == 2 ? (col ^ row) : col;
          d_base[warp * kDStageFloats + row * kRankStride + shared_col] = acc_rc(r, c);
        }
      }
    }
    __syncthreads();
  }

  // Cooperative epilogue. Each thread sums the declared buffers for one logical D element, converts
  // the local rank column back to the global rank, applies its R16 scale, and performs the sole
  // BF16 rounding. The CTA-wide index stride covers every element exactly once.
  for (int index = threadIdx.x; index < 32 * kLocalRank; index += kWarps * 32) {
    const int row = index / kLocalRank;
    const int col = index % kLocalRank;
    const int shared_col = kRankTiles == 2 ? (col ^ row) : col;
    float sum = 0.0f;
#pragma unroll
    for (int w = 0; w < kReductionBuffers; ++w) {
      sum += d_base[w * kDStageFloats + row * kRankStride + shared_col];
    }
    const int rank_col = rank0 + col;
    lora_act[static_cast<size_t>(row0 + row) * kRank + rank_col] =
      __float2bfloat16_rn(sum * scales.values[rank_col / 16]);
  }
#else
  asm volatile("trap;");
#endif
}

// Decode split-K fragment-major FP32 scratch, apply one scale per R16 tile, and emit row-major
// BF16 D. This separate pass runs after the producer on the same stream, so every K CTA's atomic
// contribution is visible and BF16 rounding happens exactly once.
//
// One thread handles eight consecutive columns = four lanes x two columns:
//   pair_index -> lane_group [0,8), q_pair [0,8), warp, rank_tile, row_block;
//   q_pair = mt*4 + pair; lane0 = lane_group*4;
//   base = fragment-major address of four lanes x two columns for atom (mt,pair);
//   row  = row_block*(kWarps*32) + warp*32 + mt*16 + lane_group + 8*(pair%2);
//   col  = rank_tile*16 + 8*(pair/2).
// Two float4 loads feed one uint4 store (eight BF16 values). A vector never crosses an R16 tile,
// so one scale applies to all eight values and the output store remains 16-byte aligned.
template <int kRankTiles, int kWarps>
__global__ void finalize_lora_bf16_kernel(const float *__restrict__ scratch,
                                          __nv_bfloat16 *__restrict__ output,
                                          LoraScaleParams scales, int total) {
  const int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair_index * 8 >= total) return;
  const int lane_group = pair_index & 7;
  int index = pair_index >> 3;
  const int q_pair = index & 7;
  index >>= 3;
  const int warp = index % kWarps;
  index /= kWarps;
  const int rank_tile = index % kRankTiles;
  const int row_block = index / kRankTiles;
  const int mt = q_pair / 4;
  const int pair = q_pair % 4;
  const int atom = mt * 4 + pair;
  const int lane0 = lane_group * 4;
  const int base =
    ((((row_block * kRankTiles + rank_tile) * kWarps + warp) * 8 + atom) * 32 + lane0) * 2;
  const float4 lo = *reinterpret_cast<const float4 *>(scratch + base);
  const float4 hi = *reinterpret_cast<const float4 *>(scratch + base + 4);
  const float scale = scales.values[rank_tile];
  auto pack_pair = [scale](float a, float b) {
    __nv_bfloat162 value =
      make_bfloat162(__float2bfloat16_rn(a * scale), __float2bfloat16_rn(b * scale));
    return *reinterpret_cast<const uint32_t *>(&value);
  };
  const uint4 packed = make_uint4(pack_pair(lo.x, lo.y), pack_pair(lo.z, lo.w),
                                  pack_pair(hi.x, hi.y), pack_pair(hi.z, hi.w));
  constexpr int kRank = kRankTiles * 16;
  const int row = row_block * (kWarps * 32) + warp * 32 + mt * 16 + lane_group + 8 * (pair % 2);
  const int col_base = rank_tile * 16 + 8 * (pair / 2);
  *reinterpret_cast<uint4 *>(output + row * kRank + col_base) = packed;
}

// Immutable argument packet shared by every schedule specialization. Keeping raw pointers and
// scalar metadata together avoids repeating identical arguments at each dispatch site and ensures
// fused and split-K launches use the same current CUDA stream.
struct QuantizeLaunchArgs {
  const __nv_bfloat16 *input;
  const __nv_bfloat16 *smooth;
  const __nv_bfloat16 *lora_down;
  uint8_t *qact;
  uint8_t *ascales;
  __nv_bfloat16 *lora_act;
  LoraScaleParams scales;
  int M;
  int K;
  cudaStream_t stream;
};

// Launch split-K as a three-step stream sequence: clear FP32 scratch, run the producer, decode D.
// Shared usage is exactly the disjoint A/B regions used by the kernel; the compile-time bound is
// 99 KiB, and larger-than-default allocations opt in through
// cudaFuncAttributeMaxDynamicSharedMemorySize. grid.x owns row blocks and grid.y owns K chunks. The
// finalize launch uses no dynamic shared memory and is stream-ordered after every producer CTA and
// atomic has completed.
template <int kRankTiles, int kWarps = 8, int kMaxSlices = 2, bool kAsyncA = true>
void launch_quantize_split_k(QuantizeLaunchArgs const &args, float *scratch) {
  using Traits = QuantizeTraits;
  constexpr int kRank = kRankTiles * 16;
  constexpr int kAStages = kAsyncA ? 2 : 1;
  constexpr int kKChunk = kMaxSlices * 64;
  constexpr size_t kSmemBytes =
    static_cast<size_t>(kWarps * kAStages * Traits::kAChunkElems +
                        kRankTiles * kMaxSlices * Traits::kBChunkElems) *
    sizeof(Element);
  static_assert(kSmemBytes <= 101376);
  auto kernel = quantize_w4a4_split_k_kernel<kRankTiles, kWarps, kMaxSlices, kAsyncA>;
  if constexpr (kSmemBytes >= 48 * 1024) {
    checkCUDA(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  }
  checkCUDA(
    cudaMemsetAsync(scratch, 0, static_cast<size_t>(args.M) * kRank * sizeof(float), args.stream));
  kernel<<<dim3(args.M / (kWarps * 32), (args.K + kKChunk - 1) / kKChunk), dim3(kWarps * 32),
           kSmemBytes, args.stream>>>(args.input, args.smooth, args.lora_down, args.qact,
                                      args.ascales, scratch, args.M, args.K);
  const int total = args.M * kRank;
  finalize_lora_bf16_kernel<kRankTiles, kWarps>
    <<<(total / 8 + 255) / 256, 256, 0, args.stream>>>(scratch, args.lora_act, args.scales, total);
}

// Launch the fused producer with grid.x = M/32 and grid.y = kRankSplits. The shared calculation
// mirrors the kernel's A + double-B + D layout, including rank-specific D row padding. No scratch
// allocation or finalize launch is required; the single kernel writes all three outputs.
template <int kRankTiles, int kWarps, int kReductionBuffers, int kRankSplits>
void launch_quantize_fused(QuantizeLaunchArgs const &args) {
  using Traits = QuantizeTraits;
  constexpr int kLocalRankTiles = kRankTiles / kRankSplits;
  constexpr int kLocalRank = kLocalRankTiles * 16;
  constexpr int kRankStride = kRankTiles == 2 ? kLocalRank + 4 : kLocalRank + 1;
  constexpr int kDStageFloats = 32 * kRankStride;
  constexpr size_t kSmemBytes =
    static_cast<size_t>(kWarps * Traits::kAChunkElems + kWarps * 2 * Traits::kBChunkElems) *
      sizeof(Element) +
    static_cast<size_t>(kReductionBuffers * kDStageFloats) * sizeof(float);
  static_assert(kSmemBytes <= 101376);
  auto kernel = quantize_w4a4_fused_lora_kernel<kRankTiles, kWarps, kReductionBuffers, kRankSplits>;
  if constexpr (kSmemBytes > 48 * 1024) {
    checkCUDA(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  }
  kernel<<<dim3(args.M / 32, kRankSplits), dim3(kWarps * 32), kSmemBytes, args.stream>>>(
    args.input, args.smooth, args.lora_down, args.qact, args.ascales, args.lora_act, args.scales,
    args.M, args.K);
}

// Host-side entry instantiated by quantize_r32/r64/r128. kRankTiles is fixed by the including TU,
// which makes rank, scale count, grid shape, shared size, and most loop bounds compile-time values.
// Validate all tensor contracts before forming raw pointers: a rank-specialized call always needs
// LoRA input/output and exactly kRankTiles scales. M is the flattened row count input.numel()/K.
//
// lora_scratch is the schedule selector, not an optional performance hint: valid FP32 storage of at
// least M*R elements always selects split-K; an invalid/default tensor selects fused. The ops layer
// is responsible for choosing that schedule and allocating scratch. All launches below use the
// current CUDA stream captured once in QuantizeLaunchArgs.
template <int kRankTiles>
void quantize_cute(svdq::Tensor input, svdq::Tensor output, svdq::Tensor ascales,
                   svdq::Tensor lora_down, svdq::Tensor lora_act, svdq::Tensor smooth,
                   std::vector<float> lora_scales, svdq::Tensor lora_scratch) {
  static_assert(kRankTiles == 2 || kRankTiles == 4 || kRankTiles == 8);
  constexpr int kRank = kRankTiles * 16;
  W4A4QuantizeRankKernel::require(
    input.valid() && output.valid() && ascales.valid() && smooth.valid(),
    "quantize_cute requires input/output/ascales/smooth");
  W4A4QuantizeRankKernel::require(
    input.dtype() == svdq::Tensor::BF16 && smooth.dtype() == svdq::Tensor::BF16,
    "quantize_cute currently requires BF16 input and smooth");
  W4A4QuantizeRankKernel::require(
    output.dtype() == svdq::Tensor::INT8 && ascales.dtype() == svdq::Tensor::FP8_E4M3,
    "quantize_cute output must be INT8 and ascales FP8_E4M3");
  W4A4QuantizeRankKernel::require(input.is_contiguous() && smooth.is_contiguous(),
                                  "input and smooth must be contiguous");
  const int K = input.shape[-1];
  W4A4QuantizeRankKernel::require(K > 0,
                                  "quantize_cute requires a non-empty input channel dimension");
  const int M = input.numel() / K;
  const int rank = lora_down.valid() ? lora_down.shape[-1] : 0;
  W4A4QuantizeRankKernel::require(
    M % 256 == 0 && K % 64 == 0,
    "quantize_cute requires M%256==0 for the native activation pack and K%64==0");
  W4A4QuantizeRankKernel::require(
    output.numel() >= static_cast<int64_t>(M) * K / 2 &&
      ascales.numel() >= static_cast<int64_t>(M) * K / 16,
    "quantize_cute output/ascales storage is smaller than the native activation pack");
  W4A4QuantizeRankKernel::require(rank == kRank,
                                  "quantize rank does not match the rank-specialized TU");
  W4A4QuantizeRankKernel::require(static_cast<int>(lora_scales.size()) == kRankTiles,
                                  "lora_scales must have one value per 16-rank tile");
  W4A4QuantizeRankKernel::require(
    lora_down.valid() && lora_down.dtype() == svdq::Tensor::BF16 && lora_down.is_contiguous() &&
      lora_act.valid() && lora_act.dtype() == svdq::Tensor::BF16,
    "CuTe LoRA-down must be row-major BF16 and lora_act must be BF16");
  W4A4QuantizeRankKernel::require(
    lora_down.shape[0] >= K && lora_act.shape[0] == M && lora_act.shape[1] == rank,
    "quantize_cute LoRA tensor shapes are inconsistent");

  LoraScaleParams scales{};
  for (size_t i = 0; i < lora_scales.size(); ++i) scales.values[i] = lora_scales[i];
  const QuantizeLaunchArgs args{
    input.data_ptr<__nv_bfloat16>(),
    smooth.data_ptr<__nv_bfloat16>(),
    lora_down.data_ptr<__nv_bfloat16>(),
    output.data_ptr<uint8_t>(),
    ascales.data_ptr<uint8_t>(),
    lora_act.data_ptr<__nv_bfloat16>(),
    scales,
    M,
    K,
    getCurrentCUDAStream(),
  };

  if (lora_scratch.valid()) {
    W4A4QuantizeRankKernel::require(
      lora_scratch.dtype() == svdq::Tensor::FP32 &&
        lora_scratch.numel() >= static_cast<size_t>(M) * rank,
      "split-K quantize_cute requires an FP32 scratch of at least M*rank elements");
    // K256 wins rank128 across the measured matrix and rank64 at K=3072. Long-K rank64 and
    // rank32 benefit from the K128 double-A async pipeline; small-K rank32 uses the synchronous
    // path.
    if constexpr (kRankTiles == 2) {
      if (K > 3072) {
        launch_quantize_split_k<2, 8, 2, true>(args, lora_scratch.data_ptr<float>());
      } else {
        launch_quantize_split_k<2, 8, 2, false>(args, lora_scratch.data_ptr<float>());
      }
    } else if constexpr (kRankTiles == 4) {
      if (K > 3072) {
        launch_quantize_split_k<4, 8, 2, true>(args, lora_scratch.data_ptr<float>());
      } else {
        launch_quantize_split_k<4, 8, 4, false>(args, lora_scratch.data_ptr<float>());
      }
    } else {
      launch_quantize_split_k<8, 8, 4, false>(args, lora_scratch.data_ptr<float>());
    }
    checkCUDA(cudaGetLastError());
    return;
  }

  // Fused dispatch template arguments are [R16 tiles, warps/CTA, D buffers, rank CTAs].
  // Current ops dispatch reaches this path for rank32/M>8192. rank64/128 fused configurations
  // remain available through the direct rank entry points and have separate numerical validation,
  // but ops currently selects split-K for those ranks across the production matrix.
  // These M buckets are measured tuning points on RTX PRO 5000 (SM120), not shape predicates:
  //   - small M benefits from more CTAs (rank splitting) and more independent K streams;
  //   - medium M can trade K parallelism for smaller CTAs and lower resource pressure;
  //   - large M already fills the GPU, so avoiding a second input read is preferable.
  // rank64 and rank128 therefore use two-way rank splitting for small/medium M, four warps at the
  // 4K crossover, and a single rank CTA for large M. Fewer D buffers reduce shared memory at the
  // cost of extra deterministic reduction phases. rank32 uses the 8-warp, no-rank-split baseline.
  if constexpr (kRankTiles == 2) {
    launch_quantize_fused<2, 8, 4, 1>(args);
  } else if constexpr (kRankTiles == 4) {
    if (M <= 2048) {
      launch_quantize_fused<4, 8, 4, 2>(args);
    } else if (M <= 4096) {
      launch_quantize_fused<4, 4, 4, 2>(args);
    } else {
      launch_quantize_fused<4, 8, 4, 1>(args);
    }
  } else {
    if (M <= 2048) {
      launch_quantize_fused<8, 8, 2, 2>(args);
    } else if (M <= 4096) {
      launch_quantize_fused<8, 4, 2, 2>(args);
    } else if (M <= 8192) {
      launch_quantize_fused<8, 8, 2, 2>(args);
    } else {
      launch_quantize_fused<8, 8, 2, 1>(args);
    }
  }
  checkCUDA(cudaGetLastError());
}

}  // namespace

}  // namespace svdq::cute_kernels
