#include "common.h"
#include "quantize.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace svdq::cute_kernels {
namespace {

// Independent CuTe activation producer. One warp owns 32 rows and the whole K reduction:
//   qact/ascales are written directly in the native 256-row MMA-fragment pack consumed by G2R;
//   D = input @ lora_down is accumulated in registers and written once as row-major BF16.
// There is no FP32 LoRA scratch, atomic, finalize kernel, or extra activation repack.
struct PackedBF16 {
  __nv_bfloat162 data[4];
};

struct PackedF32 {
  float data[8];
};

struct LoraScaleParams {
  float values[64];
};

// Unified activation producer for the CuTe NVFP4 path. It owns the packed types required at the
// kernel boundary and every helper used to produce native qact/ascales plus row-major BF16 D.
struct W4A4QuantizeUniversalKernel {
  // A 16x64 BF16 staging tile is stored as BF16; this is its size in float words.
  static constexpr int kAStageFloats = 16 * (64 + 8) / 2;

  /// Validate a host-side precondition and report the supplied message on failure.
  static void require(bool condition, const char *message) {
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  /// Approximate reciprocal used for activation scale dequantization.
  static __device__ __forceinline__ float rcp_approx(float x) {
    float result;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
  }

  /// Convert two FP32 values into two packed E2M1 FP4 values.
  static __device__ __forceinline__ uint32_t quantize_float2_fp4(float2 value) {
    uint32_t result;
    asm volatile("{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, %1, %2; cvt.u32.u8 %0, tmp; }"
                 : "=r"(result)
                 : "f"(value.y), "f"(value.x));
    return result;
  }

  /// Convert four FP32 values into four packed E4M3 FP8 values.
  static __device__ __forceinline__ uint32_t quantize_float4_fp8(float4 value) {
    uint16_t lo, hi;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(value.y), "f"(value.x));
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(value.w), "f"(value.z));
    return uint32_t(lo) | (uint32_t(hi) << 16);
  }

  /// Load two adjacent BF16 values from a 4-byte-aligned address.
  static __device__ __forceinline__ __nv_bfloat162 load_bf16x2(const __nv_bfloat16 *ptr) {
    return *reinterpret_cast<const __nv_bfloat162 *>(ptr);
  }

  /// Divide BF16x2 pairs through FP32, matching the smoothing math of the native producer.
  static __device__ __forceinline__ __nv_bfloat162 div_bf16x2(__nv_bfloat162 a, __nv_bfloat162 b) {
    float2 af = __bfloat1622float2(a);
    float2 bf = __bfloat1622float2(b);
    return __float22bfloat162_rn(make_float2(__fdividef(af.x, bf.x), __fdividef(af.y, bf.y)));
  }

  // Load a 16x16 BF16 fragment from the warp-private row-major staging tile. The tile is padded by
  // eight BF16 columns, matching the native ldmatrix layout and avoiding shared-memory bank
  // conflicts.
  static __device__ __forceinline__ void ldmatrix_x4(const void *ptr, uint4 &out) {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
                 : "l"(__cvta_generic_to_shared(ptr)));
  }

  // SM80_16x8x16_F32BF16BF16F32_TN has BLayout
  // ((4,8),(2,2)):((16,1),(8,64)). Each uint32 register packs two BF16 values that are adjacent
  // in K at the same N/rank column. lora_down is row-major [K,R], so those two values are separated
  // by R elements, not adjacent in memory. Reading ptr[0],ptr[1] would silently select two adjacent
  // rank columns and produce the observed even/odd-column permutation.
  static __device__ __forceinline__ __nv_bfloat162 load_lora_pair(const __nv_bfloat16 *lora, int R,
                                                                  int k, int r) {
    __nv_bfloat162 out;
    out.x = lora[size_t(k) * R + r];
    out.y = lora[size_t(k + 1) * R + r];
    return out;
  }

  // Load the B fragment for two m16n8k16 atoms from row-major lora_down[K,R]. data[0]/[1] are the
  // K and K+8 pairs for rank columns 0..7; data[2]/[3] are the same pairs for rank columns 8..15.
  // Scalar loads also avoid the 4-byte alignment assumption on odd rank columns.
  static __device__ __forceinline__ PackedBF16 load_lora_frags(const __nv_bfloat16 *lora, int R,
                                                               int row, int col, int lane) {
    const int k = row + (lane % 4) * 2;
    const int r = col + lane / 4;
    PackedBF16 out;
    out.data[0] = load_lora_pair(lora, R, k, r);
    out.data[1] = load_lora_pair(lora, R, k + 8, r);
    out.data[2] = load_lora_pair(lora, R, k, r + 8);
    out.data[3] = load_lora_pair(lora, R, k + 8, r + 8);
    return out;
  }

  /// Transpose one 8x8 BF16 register matrix across the warp.
  static __device__ __forceinline__ __nv_bfloat162 movmatrix_trans_bf16x2(__nv_bfloat162 input) {
    uint32_t out;
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
        : "=r"(out)
        : "r"(*reinterpret_cast<uint32_t *>(&input)));
    return *reinterpret_cast<__nv_bfloat162 *>(&out);
  }

  // Form four row-major 8x8 fragments with aligned 32-bit loads, then transpose them in registers.
  // The four matrices are (K0,R0), (K8,R0), (K0,R8), (K8,R8), matching the output order of
  // load_lora_frags after transposition. Four loads per lane replace eight scalar BF16 loads.
  static __device__ __forceinline__ PackedBF16 movmatrix_lora_frags_trans(const __nv_bfloat16 *lora,
                                                                          int R, int row, int col,
                                                                          int lane) {
    const int k = row + lane / 4;
    const int r = col + (lane % 4) * 2;
    PackedBF16 out;
    out.data[0] =
      movmatrix_trans_bf16x2(*reinterpret_cast<const __nv_bfloat162 *>(lora + size_t(k) * R + r));
    out.data[1] = movmatrix_trans_bf16x2(
      *reinterpret_cast<const __nv_bfloat162 *>(lora + size_t(k + 8) * R + r));
    out.data[2] = movmatrix_trans_bf16x2(
      *reinterpret_cast<const __nv_bfloat162 *>(lora + size_t(k) * R + r + 8));
    out.data[3] = movmatrix_trans_bf16x2(
      *reinterpret_cast<const __nv_bfloat162 *>(lora + size_t(k + 8) * R + r + 8));
    return out;
  }

  /// Issue one 16x8x16 BF16 MMA atom with an FP32 accumulator.
  static __device__ __forceinline__ uint4 mma_bf16(uint4 a, uint2 b, uint4 c) {
    uint4 d;
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
      : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(c.x), "r"(c.y), "r"(c.z),
        "r"(c.w));
    return d;
  }

  /// Fire-and-forget FP32 global reduction used by the split-K D path.
  static __device__ __forceinline__ void global_add_f32(float *addr, float value) {
    asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;" ::"l"(addr), "f"(value));
  }

  /// Two adjacent rank columns of one row; halves the split-K atomic instruction count.
  static __device__ __forceinline__ void global_add_f32x2(float *addr, float a, float b) {
    asm volatile("red.relaxed.gpu.global.add.v2.f32 [%0], {%1, %2};" ::"l"(addr), "f"(a), "f"(b));
  }

  /// Apply two B atoms to one 16-row activation fragment and update its FP32 accumulator.
  static __device__ __forceinline__ PackedF32 mma_lora(PackedBF16 a, PackedBF16 b, PackedF32 acc) {
    uint4 aa = *reinterpret_cast<uint4 *>(&a);
    uint2 b0 = *reinterpret_cast<uint2 *>(&b.data[0]);
    uint2 b1 = *reinterpret_cast<uint2 *>(&b.data[2]);
    uint4 c0 = make_uint4(__float_as_uint(acc.data[0]), __float_as_uint(acc.data[1]),
                          __float_as_uint(acc.data[2]), __float_as_uint(acc.data[3]));
    uint4 c1 = make_uint4(__float_as_uint(acc.data[4]), __float_as_uint(acc.data[5]),
                          __float_as_uint(acc.data[6]), __float_as_uint(acc.data[7]));
    uint4 d0 = mma_bf16(aa, b0, c0);
    uint4 d1 = mma_bf16(aa, b1, c1);
    acc.data[0] = __uint_as_float(d0.x);
    acc.data[1] = __uint_as_float(d0.y);
    acc.data[2] = __uint_as_float(d0.z);
    acc.data[3] = __uint_as_float(d0.w);
    acc.data[4] = __uint_as_float(d1.x);
    acc.data[5] = __uint_as_float(d1.y);
    acc.data[6] = __uint_as_float(d1.z);
    acc.data[7] = __uint_as_float(d1.w);
    return acc;
  }

  // Quantize four K16 groups and write the native activation qact/scale pack consumed by G2R.
  // The max/shuffle/rounding sequence matches the native producer. Each lane owns four u32 words
  // per K32 half; the stores place those words at their MMA-fragment positions in the 256-row
  // native block, so the GEMM can load A/SFA directly into registers without TMA or shared memory.
  static __device__ __forceinline__ void quantize_group(PackedBF16 (&tmp)[4], uint8_t *qact,
                                                        uint8_t *ascales, int M, int K, int row0,
                                                        int k0, int lane) {
    // The four MMA B fragments arrive in K-major fragment order:
    //   tmp[i].data[0/2] = the first/second K16 pair of the low rank half,
    //   tmp[i].data[1/3] = the same K pairs of the high rank half.
    // Rearrange them into two independent K32 panels so every panel stores eight BF16x2 values
    // (16 absolute-value comparisons) per lane. This is a register-only permutation.
    __nv_bfloat162 input[2][8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      input[0][i * 2] = tmp[i].data[0];
      input[0][i * 2 + 1] = tmp[i].data[2];
      input[1][i * 2] = tmp[i].data[1];
      input[1][i * 2 + 1] = tmp[i].data[3];
    }

    // Each lane first computes four independent abs-max values per K32 panel. A lane covers eight
    // BF16 values, so the four warps of an MMA fragment group collectively cover the full K32
    // quantization group after the following XOR reduction.
    float maxvalue[2][4] = {};
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float2 a = __bfloat1622float2(input[j][i * 2]);
        float2 b = __bfloat1622float2(input[j][i * 2 + 1]);
        maxvalue[j][i] = fmaxf(fabsf(a.x), fmaxf(fabsf(a.y), fmaxf(fabsf(b.x), fabsf(b.y))));
      }
    }
    // Butterflies with XOR masks 2 and 1 reduce the four lanes that share one rank column. No
    // shared memory or full-warp broadcast is needed, and every lane ends with the same maximum.
#pragma unroll
    for (int mask = 2; mask > 0; mask /= 2) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          maxvalue[j][i] = fmaxf(maxvalue[j][i], __shfl_xor_sync(~0u, maxvalue[j][i], mask));
        }
      }
    }

    // Derive the FP4 scale from the group maximum. The /6 bound reserves the E2M1 range for
    // max -> 6, while the 448 cap keeps the corresponding E4M3 scale finite. rscale is only used
    // for quantization; the FP8 word is the value persisted in ascales.
    float rscale[2][4];
    uint32_t fp8scale[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float scale[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        scale[i] = fminf(maxvalue[j][i] / 6.0f, 448.0f);
        rscale[j][i] = rcp_approx(scale[i]);
      }
      fp8scale[j] = quantize_float4_fp8(make_float4(scale[0], scale[1], scale[2], scale[3]));
    }

    // Convert to E2M1 and place each two-value result in the byte owned by this lane. The left
    // shift selects one of four lane positions: (lane % 4) * 8 bits. The subsequent XOR shuffles
    // assemble the four lane bytes into one packed u32 qact word per K32 group.
    uint32_t qpacks[2][8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        float2 fval = __bfloat1622float2(input[j][i]);
        fval.x *= rscale[j][i / 2];
        fval.y *= rscale[j][i / 2];
        qpacks[j][i] = quantize_float2_fp4(fval) << ((lane % 4) * 8);
      }
    }
    // XOR masks 1 and 2 implement the inverse byte placement, so all four lanes in a rank group
    // obtain the same fully packed u32. This duplicates the final word in registers and lets the
    // writer lanes below issue wide, fixed-offset stores without shared-memory staging.
#pragma unroll
    for (int mask = 1; mask <= 2; mask *= 2) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          qpacks[j][i] |= __shfl_xor_sync(~0u, qpacks[j][i], mask);
        }
      }
    }

    // Redistribute the packed K32 words into the native 256-row interleaved activation pack. Lanes
    // 0/1 of each four-lane rank group own all eight words after the shuffle; the other two lanes
    // intentionally skip the global store. The qact index is the sum of three independent terms:
    //   base  = 256-row block, K64 step, and 32-row warp panel;
    //   local = row-within-8, the 8/16 half selector, and the 16/32 half selector;
    //   i * 2 = the two words owned by this writer lane.
    // Each dst[0/4/8/12] advances by four u32 words, i.e. the next K16 group in the same panel.
    // This mapping was verified word-for-word against the native native qact output.
    const int i = lane % 4;
    if (i < 2) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int row = row0 + lane / 4 + j * 8;
        const int local = 16 * (row % 8) + ((row % 16) / 8) + 128 * ((row % 32) / 16);
        const int64_t base =
          ((static_cast<int64_t>(row / 256) * (K / 64) + k0 / 64) * 8 + (row % 256) / 32) * 256;
        uint32_t *dst = reinterpret_cast<uint32_t *>(qact) + base + local + i * 2;
        dst[0] = qpacks[j][i * 4 + 0];
        dst[4] = qpacks[j][i * 4 + 1];
        dst[8] = qpacks[j][i * 4 + 2];
        dst[12] = qpacks[j][i * 4 + 3];
      }
    }

    // Only lane 0 of each rank group writes ascales. The native scale tile has one u32 word per
    // row, containing the four FP8 scale factors above. Its lane coordinate is 4*(row mod 8) + (row
    // mod 32)/8; bitwise it is the qact panel lane before the u32 widening.
    if (lane % 4 == 0) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int row = row0 + lane / 4 + j * 8;
        const int64_t base =
          (static_cast<int64_t>(row / 256) * (K / 64) + k0 / 64) * 8 + (row % 256) / 32;
        const int scale_lane = 4 * (row % 8) + ((row % 32) / 8);
        reinterpret_cast<uint32_t *>(ascales)[base * 32 + scale_lane] = fp8scale[j];
      }
    }
  }

  // Main producer body. A CTA owns 32 rows; kRankSplits CTAs may split the rank dimension,
  // kWarps warps split K, and each warp emits one A/SFA K64 slice before accumulating LoRA-D.
  // qact/ascales always follow the native pack, while D is reduced in FP32 and rounded once to
  // BF16.
  template <int kRankTiles, int kWarps, int kReductionBuffers = kWarps, int kRankSplits = 1>
  static __device__ void quantize_w4a4_fused_lora(const __nv_bfloat16 *__restrict__ input,
                                                  const __nv_bfloat16 *__restrict__ smooth,
                                                  const __nv_bfloat16 *__restrict__ lora_down,
                                                  uint8_t *__restrict__ qact,
                                                  uint8_t *__restrict__ ascales,
                                                  __nv_bfloat16 *__restrict__ lora_act,
                                                  LoraScaleParams scales, int M, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    // kRankTiles describes the full logical rank (one tile == 16 columns). kRankSplits may split
    // that rank across blockIdx.y CTAs, so the local accumulator only owns kLocalRankTiles.
    static_assert(kRankSplits == 1 || kRankSplits == 2);
    constexpr int kLocalRankTiles = kRankTiles / kRankSplits;
    constexpr int kAccTiles = kLocalRankTiles > 0 ? kLocalRankTiles : 1;
    constexpr int kRank = kRankTiles * 16;
    constexpr int kLocalRank = kLocalRankTiles * 16;
    // One FP32 partial is 32 rows by the locally owned rank columns. The extra pad column keeps
    // the rank-32 XOR layout conflict-free; other ranks use a one-column guard.
    constexpr int kRankStride = kRankTiles == 2 ? kLocalRank + 4 : kLocalRank + 1;
    constexpr int kDStageFloats = kLocalRankTiles > 0 ? 32 * kRankStride : 0;
    // The full-buffer path keeps A staging and D partials disjoint: it already runs at one CTA/SM
    // from register pressure, so the extra shared memory is free and removes the CTA-wide barrier
    // otherwise needed before overwriting A. The staged path overlays the two regions and uses a
    // barrier between phases.
    constexpr int kWarpStageFloats = kAStageFloats;
    constexpr int kASharedFloats = kWarps * kWarpStageFloats;
    constexpr int kDSharedFloats = kReductionBuffers * kDStageFloats;
    constexpr int kSharedFloats =
      kReductionBuffers == kWarps
        ? kASharedFloats + kDSharedFloats
        : (kASharedFloats > kDSharedFloats ? kASharedFloats : kDSharedFloats);
    constexpr int kAStride = 64 + 8;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    // A CTA owns one original 32-row panel. kRankSplits only changes the rank columns, not rows,
    // so qact/ascales and the BF16 D row mapping stay identical to the single-CTA producer.
    const int row0 = blockIdx.x * 32;
    const int rank0 = kRankSplits == 1 ? 0 : blockIdx.y * kLocalRank;
    extern __shared__ float shared_scratch[];
    auto *warp_stage = reinterpret_cast<__nv_bfloat16 *>(shared_scratch + warp * kWarpStageFloats);

    // Two M16 halves x local rank tiles x two m16n8k16 atoms. Each atom has four FP32 values, so
    // one PackedF32 (eight floats) covers both N8 rank halves of one M16xR16 output tile.
    PackedF32 acc[2][kAccTiles];
    if constexpr (kLocalRankTiles > 0) {
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int rt = 0; rt < kLocalRankTiles; ++rt) {
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            acc[mt][rt].data[j] = 0.0f;
          }
        }
      }
    }

    // K is partitioned at K64 granularity. Warp w starts at K64 slice w and advances by kWarps
    // slices, so each warp owns one independent FP32 D accumulation stream. All warps cover the
    // same 32 rows and later reduce their streams in shared memory; no input or D scratch is
    // materialized in global memory. If K < kWarps*64, the extra warps skip this loop and enter
    // the same final barrier with zero accumulators.
    for (int k0 = warp * 64; k0 < K; k0 += kWarps * 64) {
      // A fragments for the two M16 halves and four K16 atoms. They stay in registers across the
      // quantize and LoRA MMA phases so the input row tile is read exactly once.
      PackedBF16 raw_all[2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        // Stage a 16x64 BF16 tile for one M16 half. Eight lanes cooperate on each 64-wide row,
        // each moving 16 B; four iterations cover the 16 rows. The global read is coalesced as
        // eight 128-byte row transactions per iteration group instead of scalar 2-byte loads.
        // stage_row/stage_col describe the logical row-major tile; kAStride (64+8 BF16) supplies
        // the pad that lets ldmatrix read 16x16 fragments without shared-memory bank conflicts.
#pragma unroll
        for (int row = 0; row < 16; row += 4) {
          const int stage_row = row + lane / 8;
          const int stage_col = (lane % 8) * 8;
          const uint4 pack = *reinterpret_cast<const uint4 *>(
            input + size_t(row0 + mt * 16 + stage_row) * K + k0 + stage_col);
          *reinterpret_cast<uint4 *>(warp_stage + stage_row * kAStride + stage_col) = pack;
        }
        __syncwarp();

        // Read four 16x16 fragments from the staged half. Lanes 0..15 supply rows 0..15 for the
        // low N8 half, lanes 16..31 repeat those rows for the high N8 half. ldmatrix transposes
        // the eight 8x8 submatrices into the register order expected by two m16n8k16 MMA atoms;
        // n selects the K16 atom inside the current K64 slice.
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          uint4 tmp;
          const int row = lane % 16;
          const int col = n * 16 + lane / 16 * 8;
          ldmatrix_x4(warp_stage + row * kAStride + col, tmp);
          raw_all[mt][n] = *reinterpret_cast<PackedBF16 *>(&tmp);
        }
        // The next M16 half overwrites the same warp-private staging tile, so every lane must
        // have completed its ldmatrix before the next store phase begins.
        __syncwarp();
      }

      // Activation quantization is independent of the LoRA rank split. When two CTAs split rank,
      // only blockIdx.y == 0 executes this branch, preventing duplicate qact/ascales stores and
      // duplicate smooth loads. Both CTAs still retain raw_all for their own LoRA MMA.
      if (kRankSplits == 1 || blockIdx.y == 0) {
        // Each lane loads the two smoothing factors associated with its K16 atom: smooth0 covers
        // rank columns [0,8), smooth1 covers [8,16). The following divide uses the same BF16
        // rounding path as the native producer, while raw_all remains unmodified for LoRA MMA.
        __nv_bfloat162 smooth0[4];
        __nv_bfloat162 smooth1[4];
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          const int col = k0 + n * 16 + (lane % 4) * 2;
          smooth0[n] = load_bf16x2(smooth + col);
          smooth1[n] = load_bf16x2(smooth + col + 8);
        }

        // Quantize the two M16 halves separately. quant is a register-only copy, so applying the
        // smoothing division here cannot perturb the BF16 A values consumed by the LoRA MMA.
        // quantize_group then derives per-K32 FP4 scales and writes the native qact/ascales pack.
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          PackedBF16 quant[4];
#pragma unroll
          for (int n = 0; n < 4; ++n) {
            quant[n] = raw_all[mt][n];
            quant[n].data[0] = div_bf16x2(quant[n].data[0], smooth0[n]);
            quant[n].data[1] = div_bf16x2(quant[n].data[1], smooth0[n]);
            quant[n].data[2] = div_bf16x2(quant[n].data[2], smooth1[n]);
            quant[n].data[3] = div_bf16x2(quant[n].data[3], smooth1[n]);
          }
          quantize_group(quant, qact, ascales, M, K, row0 + mt * 16, k0, lane);
        }
      }

      // LoRA-down MMA. rt selects one locally owned R16 tile, n selects the K16 atom, and mt
      // selects the M16 half. Each B fragment is loaded once and reused for both M16 halves,
      // halving lora_down traffic versus reloading per half.
      if constexpr (kRankTiles == 8) {
#pragma unroll
        for (int rt = 0; rt < kLocalRankTiles; ++rt) {
#pragma unroll
          for (int n = 0; n < 4; ++n) {
            // Rank128 uses four aligned 32-bit loads plus movmatrix.trans. The transposed 8x8
            // fragments match the K-paired register order of load_lora_frags but require half
            // as many global load instructions. rank0 shifts the logical R16 tile for CTA y=1.
            PackedBF16 b =
              movmatrix_lora_frags_trans(lora_down, kRank, k0 + n * 16, rank0 + rt * 16, lane);
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
              acc[mt][rt] = mma_lora(raw_all[mt][n], b, acc[mt][rt]);
            }
          }
        }
      } else if constexpr (kLocalRankTiles > 0) {
        // Narrow ranks keep the scalar B path: its request count is not the limiter there, and
        // avoiding register-transpose setup is faster for rank<=64.
#pragma unroll
        for (int rt = 0; rt < kLocalRankTiles; ++rt) {
#pragma unroll
          for (int n = 0; n < 4; ++n) {
            PackedBF16 b = load_lora_frags(lora_down, kRank, k0 + n * 16, rank0 + rt * 16, lane);
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
              acc[mt][rt] = mma_lora(raw_all[mt][n], b, acc[mt][rt]);
            }
          }
        }
      }
    }

    // D reduction. Every warp holds a full 32xR_local FP32 partial over a disjoint K slice.
    // kReductionBuffers controls how many partials can be materialized at once:
    //   == kWarps: each warp gets its own buffer and the final loop sums kWarps buffers;
    //    < kWarps: buffers are reused in fixed warp groups, and the final loop sums only the
    //              declared number of buffers. The addition order is deterministic in both cases.
    if constexpr (kLocalRankTiles > 0) {
      static_assert(kWarps % kReductionBuffers == 0);
      if constexpr (kReductionBuffers < kWarps) {
        // The A staging area is overlaid by D partials only after every warp has finished the K
        // loop. The first syncthreads makes the overlay and the zero-work warps safe: warps with
        // no K slice also reach this point and participate in every group barrier.
        __syncthreads();
#pragma unroll
        for (int w = 0; w < kWarps; w += kReductionBuffers) {
          // Group w owns buffers [w, w + kReductionBuffers). A buffer is read, updated, and
          // written by one warp at a time; the following syncthreads hands it to the next group.
          if (warp >= w && warp < w + kReductionBuffers) {
            const int buffer = warp - w;
            float *partial = shared_scratch + buffer * kDStageFloats;
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
              for (int rt = 0; rt < kLocalRankTiles; ++rt) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                  // The MMA accumulator has one value per (row, rank column). Reconstruct its
                  // logical row and local rank column from the packed fragment index j:
                  //   rows advance by lane/4 and by the two M16/N8 atom halves;
                  //   columns advance by lane%4, by the two values in a 32-bit pair, and by the
                  //   second N8 atom. kRankStride is the shared-memory row pitch.
                  const int row = mt * 16 + lane / 4 + 8 * ((j / 2) % 2);
                  const int col = rt * 16 + 2 * (lane % 4) + (j % 2) + 8 * (j / 4);
                  const int shared_col = kRankTiles == 2 ? (col ^ row) : col;
                  const int offset = row * kRankStride + shared_col;
                  const float value = acc[mt][rt].data[j];
                  // First visit initializes the buffer; later groups add in place. No atomic is
                  // needed because barriers serialize the read-modify-write sequence.
                  partial[offset] = w == 0 ? value : partial[offset] + value;
                }
              }
            }
          }
          __syncthreads();
        }
      } else {
        // Full-buffer path: A staging and D partials are disjoint, so a warp that finishes its K
        // loop early cannot clobber another warp's ldmatrix input. Only the post-store barrier is
        // needed before the cooperative sum.
        float *warp_partial = shared_scratch + kASharedFloats + warp * kDStageFloats;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
          for (int rt = 0; rt < kLocalRankTiles; ++rt) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              // The accumulator-fragment to logical-(row, rank-column) mapping is identical to
              // the staged branch above. Each lane writes the eight values of one atom at their
              // true row-major coordinates, not at the MMA register permutation.
              const int row = mt * 16 + lane / 4 + 8 * ((j / 2) % 2);
              const int col = rt * 16 + 2 * (lane % 4) + (j % 2) + 8 * (j / 4);
              const int shared_col = kRankTiles == 2 ? (col ^ row) : col;
              warp_partial[row * kRankStride + shared_col] = acc[mt][rt].data[j];
            }
          }
        }
        __syncthreads();
      }

      // Cooperative finalization. Threads walk the local 32xR_local output in linear order, sum
      // the declared FP32 partial buffers, apply the per-16-rank scale, and round exactly once to
      // BF16. rank0 converts the local column to the global rank column; row0 keeps the original
      // 32-row D layout. The index stride covers the whole CTA, so every element is written once.
      for (int index = threadIdx.x; index < 32 * kLocalRank; index += kWarps * 32) {
        const int row = index / kLocalRank;
        const int col = index % kLocalRank;
        const int shared_col = kRankTiles == 2 ? (col ^ row) : col;
        float sum = 0.0f;
        const float *partial_base =
          kReductionBuffers == kWarps ? shared_scratch + kASharedFloats : shared_scratch;
#pragma unroll
        for (int w = 0; w < kReductionBuffers; ++w) {
          sum += partial_base[w * kDStageFloats + row * kRankStride + shared_col];
        }
        const int rank_col = rank0 + col;
        lora_act[size_t(row0 + row) * kRank + rank_col] =
          __float2bfloat16_rn(sum * scales.values[rank_col / 16]);
      }
    }
#else
    asm volatile("trap;");
#endif
  }

  // Split-K producer. A CTA owns 256 rows and one K128 slice: each of the eight warps owns a
  // distinct 32-row panel, so qact/ascales are written exactly once per (M256, K64) block and no
  // warp ever duplicates another warp's activation work. LoRA-D is reduced across K CTAs into an
  // FP32 scratch tensor; the caller zeroes it before this launch. The finalize kernel applies the
  // per-16-rank scale and performs the single BF16 rounding.
  //
  // Parallel decomposition:
  //   grid.x = M / (kWarps * 32)   one 256-row block when kWarps == 8
  //   grid.y = ceil(K / kKChunk)   one K128 slice per CTA
  //   block  = kWarps * 32 threads
  //   warp   = one 32-row panel (two M16 halves)
  //
  // A warp owns rows [row0, row0+32) for every K64 in this CTA's K128 slice. The two K64 A
  // fragments stay in raw_all registers so qact and D consume the same global read. D is streamed
  // one R16 rank tile at a time: the full rank is never resident in registers, which is what keeps
  // this path below the 220-255 registers of the fused small-M producer.
  //
  // Numerical/layout contract:
  //   qact/ascales: native 256-row pack, one writer per (row block, K64).
  //   LoRA scratch: fragment-major FP32, zeroed by the caller, reduced with v2.f32 atomics.
  //   finalize: per-16-rank scale + exactly one BF16 rounding into row-major lora_act.
  //   D addition order is therefore not bitwise-defined across K CTAs; qact/ascales remain exact.
  template <int kRankTiles, int kWarps>
  static __device__ void quantize_w4a4_split_k(const __nv_bfloat16 *__restrict__ input,
                                               const __nv_bfloat16 *__restrict__ smooth,
                                               const __nv_bfloat16 *__restrict__ lora_down,
                                               uint8_t *__restrict__ qact,
                                               uint8_t *__restrict__ ascales,
                                               float *__restrict__ lora_scratch, int M, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    constexpr int kRank = kRankTiles * 16;
    constexpr int kKChunk = 128;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    // blockIdx.y selects the K128 slice. K is always a multiple of 64, so the second K64 exists
    // except for a final 64-wide tail; the slice loop below guards that case.
    const int k_begin = blockIdx.y * kKChunk;
    const int k_end = k_begin + kKChunk < K ? k_begin + kKChunk : K;
    if (k_begin >= K) {
      return;
    }
    // warp-major row ownership: each warp stages and consumes exactly its own 32 rows. The
    // warp-private shared tile is therefore reused without any cross-warp barrier.
    const int row0 = blockIdx.x * (kWarps * 32) + warp * 32;
    extern __shared__ float shared_scratch[];
    auto *warp_stage = reinterpret_cast<__nv_bfloat16 *>(shared_scratch + warp * kAStageFloats);

    // Keep the two K64 A fragments for this K128 CTA in registers, but only one rank chunk of D.
    // Native uses the same shape: full A for the K tile, rank-chunk streaming for D. Holding all
    // rank tiles would recreate the 128-FP32 register pressure this split-K path is meant to avoid.
    // kChunkTiles == 1 keeps one R16 accumulator per rank chunk; two K64 slices are accumulated
    // before the chunk is reduced, so scratch traffic stays at one v2 pair per column pair.
    constexpr int kChunkTiles = 1;
    static_assert(kRankTiles % kChunkTiles == 0);
    constexpr int kRankChunks = kRankTiles / kChunkTiles;
    // raw_all[mt][slice][n]: two M16 halves, two K64 slices, four K16 atoms per slice.
    PackedBF16 raw_all[2][2][4];
#pragma unroll
    // Load both K64 slices once: stage 16x64 BF16 through warp-private shared memory, then
    // ldmatrix into raw_all. qact and D both read raw_all, so the global input is not re-read.
    for (int slice = 0; slice < 2; ++slice) {
      const int k0 = k_begin + slice * 64;
      if (k0 >= k_end) {
        break;
      }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int row = 0; row < 16; row += 4) {
          const int stage_row = row + lane / 8;
          const int stage_col = (lane % 8) * 8;
          const uint4 pack = *reinterpret_cast<const uint4 *>(
            input + size_t(row0 + mt * 16 + stage_row) * K + k0 + stage_col);
          *reinterpret_cast<uint4 *>(warp_stage + stage_row * (64 + 8) + stage_col) = pack;
        }
        __syncwarp();
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          uint4 tmp;
          const int row = lane % 16;
          const int col = n * 16 + lane / 16 * 8;
          ldmatrix_x4(warp_stage + row * (64 + 8) + col, tmp);
          raw_all[mt][slice][n] = *reinterpret_cast<PackedBF16 *>(&tmp);
        }
        __syncwarp();
      }

      // Quantize the K64 slice immediately after its A fragments land. qact/ascales are indexed by
      // (row block, K64), so each CTA writes a disjoint slice; this is the only place the native
      // activation pack is produced.
      __nv_bfloat162 smooth0[4];
      __nv_bfloat162 smooth1[4];
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const int col = k0 + n * 16 + (lane % 4) * 2;
        smooth0[n] = load_bf16x2(smooth + col);
        smooth1[n] = load_bf16x2(smooth + col + 8);
      }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        PackedBF16 quant[4];
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          quant[n] = raw_all[mt][slice][n];
          quant[n].data[0] = div_bf16x2(quant[n].data[0], smooth0[n]);
          quant[n].data[1] = div_bf16x2(quant[n].data[1], smooth0[n]);
          quant[n].data[2] = div_bf16x2(quant[n].data[2], smooth1[n]);
          quant[n].data[3] = div_bf16x2(quant[n].data[3], smooth1[n]);
        }
        quantize_group(quant, qact, ascales, M, K, row0 + mt * 16, k0, lane);
      }
    }

    // Rank-chunk streaming. A chunk is accumulated over both K64 slices, reduced into FP32
    // scratch, then the same registers are reused for the next chunk. kChunkTiles == 1 means the
    // live D state is one R16 tile per M16 half, independent of the full rank width.
#pragma unroll
    for (int chunk = 0; chunk < kRankChunks; ++chunk) {
      PackedF32 acc[2][kChunkTiles];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int rt = 0; rt < kChunkTiles; ++rt) {
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            acc[mt][rt].data[j] = 0.0f;
          }
        }
      }

#pragma unroll
      for (int slice = 0; slice < 2; ++slice) {
        const int k0 = k_begin + slice * 64;
        if (k0 >= k_end) {
          break;
        }
#pragma unroll
        for (int rt = 0; rt < kChunkTiles; ++rt) {
          const int global_rt = chunk * kChunkTiles + rt;
#pragma unroll
          for (int n = 0; n < 4; ++n) {
            PackedBF16 b =
              kRankTiles == 8
                ? movmatrix_lora_frags_trans(lora_down, kRank, k0 + n * 16, global_rt * 16, lane)
                : load_lora_frags(lora_down, kRank, k0 + n * 16, global_rt * 16, lane);
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
              acc[mt][rt] = mma_lora(raw_all[mt][slice][n], b, acc[mt][rt]);
            }
          }
        }
      }

      // Fragment-major reduction. Trace one atom: (row_block, rank tile, warp, mt/pair) selects a
      // 32-lane vector, and the final *2 stores lane-l and lane-l+1 as adjacent rank columns.
      // A warp therefore touches one coalesced sector per atom; finalize later turns each four-lane
      // group into eight contiguous row-major BF16 columns. The scratch must be zeroed before this
      // kernel because every K CTA contributes with red.add.
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int rt = 0; rt < kChunkTiles; ++rt) {
          const int global_rt = chunk * kChunkTiles + rt;
#pragma unroll
          for (int pair = 0; pair < 4; ++pair) {
            const int atom = mt * 4 + pair;
            const int packed_index =
              ((((blockIdx.x * kRankTiles + global_rt) * kWarps + warp) * 8 + atom) * 32 + lane) *
              2;
            global_add_f32x2(lora_scratch + packed_index, acc[mt][rt].data[pair * 2],
                             acc[mt][rt].data[pair * 2 + 1]);
          }
        }
      }
    }
#else
    asm volatile("trap;");
#endif
  }
};

// Thin shell for the 256-row split-K producer.
template <int kRankTiles, int kWarps>
__global__ void quantize_w4a4_split_k_kernel(const __nv_bfloat16 *__restrict__ input,
                                             const __nv_bfloat16 *__restrict__ smooth,
                                             const __nv_bfloat16 *__restrict__ lora_down,
                                             uint8_t *__restrict__ qact,
                                             uint8_t *__restrict__ ascales,
                                             float *__restrict__ lora_scratch, int M, int K) {
  W4A4QuantizeUniversalKernel::quantize_w4a4_split_k<kRankTiles, kWarps>(
    input, smooth, lora_down, qact, ascales, lora_scratch, M, K);
}

// Convert the accumulated FP32 split-K scratch to row-major BF16 and apply the per-16-rank scale.
// This is deliberately a separate linear pass: it keeps the producer free of a full-rank register
// accumulator and performs the only BF16 rounding after all K CTAs have contributed.
//
// Scratch decode (inverse of the producer's fragment-major layout):
//   pair_index -> lane_group [0,8), q_pair [0,8), warp, rank_tile, row_block
//   q_pair = mt*4 + pair, lane0 = lane_group*4
//   base points at four lanes x two columns for atom (mt,pair)
//   row  = row_block*(kWarps*32) + warp*32 + mt*16 + lane_group + 8*(pair%2)
//   col  = rank_tile*16 + 8*(pair/2)
// Each thread reads 8 contiguous FP32 values (two float4 loads) and writes one uint4 = 8 BF16
// values, so the output store is 16-byte vectorized and never crosses a 16-rank scale tile.
template <int kRankTiles, int kWarps>
__global__ void finalize_lora_bf16_kernel(const float *__restrict__ scratch,
                                          __nv_bfloat16 *__restrict__ output,
                                          LoraScaleParams scales, int total) {
  // One thread owns four lanes x two adjacent rank columns = eight consecutive columns.
  const int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair_index * 8 >= total) {
    return;
  }
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

// Thin launch shell: all kernel logic lives in
// W4A4QuantizeUniversalKernel::quantize_w4a4_fused_lora.
template <int kRankTiles, int kWarps, int kReductionBuffers = kWarps, int kRankSplits = 1>
__global__ void quantize_w4a4_fused_lora_kernel(const __nv_bfloat16 *__restrict__ input,
                                                const __nv_bfloat16 *__restrict__ smooth,
                                                const __nv_bfloat16 *__restrict__ lora_down,
                                                uint8_t *__restrict__ qact,
                                                uint8_t *__restrict__ ascales,
                                                __nv_bfloat16 *__restrict__ lora_act,
                                                LoraScaleParams scales, int M, int K) {
  W4A4QuantizeUniversalKernel::quantize_w4a4_fused_lora<kRankTiles, kWarps, kReductionBuffers,
                                                        kRankSplits>(
    input, smooth, lora_down, qact, ascales, lora_act, scales, M, K);
}

/// Size dynamic shared memory and launch one producer specialization on the current stream.
template <int kRankTiles, int kWarps, int kReductionBuffers = kWarps, int kRankSplits = 1>
void launch_quantize_w4a4(const __nv_bfloat16 *input, const __nv_bfloat16 *smooth,
                          const __nv_bfloat16 *lora_down, uint8_t *qact, uint8_t *ascales,
                          __nv_bfloat16 *lora_act, LoraScaleParams scales, int M, int K,
                          cudaStream_t stream) {
  constexpr int kLocalRankTiles = kRankTiles / kRankSplits;
  constexpr int kLocalRank = kLocalRankTiles * 16;
  constexpr int kRankStride = kRankTiles == 2 ? kLocalRank + 4 : kLocalRank + 1;
  constexpr int kDStageFloats = kLocalRankTiles > 0 ? 32 * kRankStride : 0;
  constexpr int kWarpStageFloats = W4A4QuantizeUniversalKernel::kAStageFloats;
  constexpr int kASharedFloats = kWarps * kWarpStageFloats;
  constexpr int kDSharedFloats = kReductionBuffers * kDStageFloats;
  constexpr int kSharedFloats =
    kReductionBuffers == kWarps
      ? kASharedFloats + kDSharedFloats
      : (kASharedFloats > kDSharedFloats ? kASharedFloats : kDSharedFloats);
  constexpr size_t kSmemBytes = static_cast<size_t>(kSharedFloats) * sizeof(float);
  static_assert(kSmemBytes <= 101376, "quantize shared memory exceeds the SM120 opt-in budget");
  auto kernel = quantize_w4a4_fused_lora_kernel<kRankTiles, kWarps, kReductionBuffers, kRankSplits>;
  if constexpr (kSmemBytes > 48 * 1024) {
    checkCUDA(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  }
  kernel<<<dim3(M / 32, kRankSplits), dim3(kWarps * 32), kSmemBytes, stream>>>(
    input, smooth, lora_down, qact, ascales, lora_act, scales, M, K);
}

/// Launch the 256-row split-K producer, then finalize FP32 scratch into row-major BF16 D.
template <int kRankTiles, int kWarps>
void launch_quantize_w4a4_split_k(const __nv_bfloat16 *input, const __nv_bfloat16 *smooth,
                                  const __nv_bfloat16 *lora_down, uint8_t *qact, uint8_t *ascales,
                                  float *scratch, __nv_bfloat16 *lora_act, LoraScaleParams scales,
                                  int M, int K, cudaStream_t stream) {
  constexpr int kRank = kRankTiles * 16;
  constexpr int kRows = kWarps * 32;
  constexpr int kSharedFloats = kWarps * W4A4QuantizeUniversalKernel::kAStageFloats;
  constexpr size_t kSmemBytes = static_cast<size_t>(kSharedFloats) * sizeof(float);
  static_assert(kSmemBytes <= 101376, "quantize shared memory exceeds the SM120 opt-in budget");
  auto kernel = quantize_w4a4_split_k_kernel<kRankTiles, kWarps>;
  if constexpr (kSmemBytes > 48 * 1024) {
    checkCUDA(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  }
  checkCUDA(cudaMemsetAsync(scratch, 0, static_cast<size_t>(M) * kRank * sizeof(float), stream));
  kernel<<<dim3(M / kRows, (K + 127) / 128), dim3(kWarps * 32), kSmemBytes, stream>>>(
    input, smooth, lora_down, qact, ascales, scratch, M, K);
  const int total = M * kRank;
  const int vectors = total / 8;
  finalize_lora_bf16_kernel<kRankTiles, kWarps>
    <<<(vectors + 255) / 256, 256, 0, stream>>>(scratch, lora_act, scales, total);
}

// One immutable argument packet shared by every rank/schedule specialization in the dispatcher.
// Keeping it here avoids repeating eight identical arguments at each dispatch site.
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

/// Launch a specialization from the shared argument packet.
template <int kRankTiles, int kWarps, int kReductionBuffers = kWarps, int kRankSplits = 1>
void launch_quantize_w4a4(const QuantizeLaunchArgs &args) {
  launch_quantize_w4a4<kRankTiles, kWarps, kReductionBuffers, kRankSplits>(
    args.input, args.smooth, args.lora_down, args.qact, args.ascales, args.lora_act, args.scales,
    args.M, args.K, args.stream);
}

}  // namespace

void quantize_uni(svdq::Tensor input, svdq::Tensor output, svdq::Tensor ascales,
                  svdq::Tensor lora_down, svdq::Tensor lora_act, svdq::Tensor smooth,
                  std::vector<float> lora_scales, svdq::Tensor lora_scratch) {
  W4A4QuantizeUniversalKernel::require(
    input.valid() && output.valid() && ascales.valid() && smooth.valid(),
    "quantize_w4a4 requires input/output/ascales/smooth");
  W4A4QuantizeUniversalKernel::require(
    input.dtype() == svdq::Tensor::BF16 && smooth.dtype() == svdq::Tensor::BF16,
    "quantize_w4a4 currently requires BF16 input and smooth");
  W4A4QuantizeUniversalKernel::require(
    output.dtype() == svdq::Tensor::INT8 && ascales.dtype() == svdq::Tensor::FP8_E4M3,
    "quantize_w4a4 output must be INT8 and ascales FP8_E4M3");
  W4A4QuantizeUniversalKernel::require(input.is_contiguous() && smooth.is_contiguous(),
                                       "input and smooth must be contiguous");
  const int K = input.shape[-1];
  W4A4QuantizeUniversalKernel::require(K > 0, "quantize_w4a4 requires non-empty input channels");
  const int M = input.numel() / K;
  const int rank = lora_down.valid() ? lora_down.shape[-1] : 0;
  W4A4QuantizeUniversalKernel::require(
    M % 256 == 0 && K % 64 == 0,
    "quantize_w4a4 requires M%256==0 for the native activation pack and K%64==0");
  W4A4QuantizeUniversalKernel::require(
    output.numel() >= static_cast<int64_t>(M) * K / 2 &&
      ascales.numel() >= static_cast<int64_t>(M) * K / 16,
    "quantize_w4a4 output/ascales storage is smaller than the native activation pack");
  W4A4QuantizeUniversalKernel::require(
    rank == 0 || (rank % 16 == 0 && rank <= 128),
    "quantize_w4a4 rank must be 0 or a multiple of 16 up to 128");
  W4A4QuantizeUniversalKernel::require(
    static_cast<int>(lora_scales.size()) == std::max(1, rank / 16),
    "lora_scales must have one value per 16-rank tile");
  if (rank > 0) {
    W4A4QuantizeUniversalKernel::require(
      lora_down.dtype() == svdq::Tensor::BF16 && lora_down.is_contiguous() && lora_act.valid() &&
        lora_act.dtype() == svdq::Tensor::BF16,
      "CuTe LoRA-down must be row-major BF16 and lora_act must be BF16");
  }

  LoraScaleParams scales{};
  for (size_t i = 0; i < lora_scales.size(); ++i) {
    scales.values[i] = lora_scales[i];
  }
  const QuantizeLaunchArgs args{
    input.data_ptr<__nv_bfloat16>(),
    smooth.data_ptr<__nv_bfloat16>(),
    rank > 0 ? lora_down.data_ptr<__nv_bfloat16>() : nullptr,
    output.data_ptr<uint8_t>(),
    ascales.data_ptr<uint8_t>(),
    rank > 0 ? lora_act.data_ptr<__nv_bfloat16>() : nullptr,
    scales,
    M,
    K,
    getCurrentCUDAStream(),
  };

  if (lora_scratch.valid()) {
    W4A4QuantizeUniversalKernel::require(
      lora_scratch.dtype() == svdq::Tensor::FP32 &&
        lora_scratch.numel() >= static_cast<size_t>(M) * rank,
      "split-K quantize_w4a4 requires an FP32 scratch of at least M*rank elements");
    float *scratch = lora_scratch.data_ptr<float>();
    if (rank == 32) {
      launch_quantize_w4a4_split_k<2, 8>(args.input, args.smooth, args.lora_down, args.qact,
                                         args.ascales, scratch, args.lora_act, scales, M, K,
                                         args.stream);
    } else if (rank == 64) {
      launch_quantize_w4a4_split_k<4, 8>(args.input, args.smooth, args.lora_down, args.qact,
                                         args.ascales, scratch, args.lora_act, scales, M, K,
                                         args.stream);
    } else if (rank == 128) {
      launch_quantize_w4a4_split_k<8, 8>(args.input, args.smooth, args.lora_down, args.qact,
                                         args.ascales, scratch, args.lora_act, scales, M, K,
                                         args.stream);
    } else {
      throw std::runtime_error("split-K quantize_w4a4 supports rank 32/64/128 only");
    }
    checkCUDA(cudaGetLastError());
    return;
  }

  // Dispatch policy. The four template arguments are:
  //   [0] full rank tiles = rank / 16;
  //   [1] warps per CTA = independent K64 slices;
  //   [2] D reduction buffers (default: one per warp);
  //   [3] rank CTAs along blockIdx.y (default: one CTA owns the full rank).
  //
  // The thresholds below are not theoretical defaults. They were tuned by measuring every
  // target shape and rank on an RTX PRO 5000 (SM120) with the project's CUDA Event benchmark,
  // repeatedly comparing row_major against the native FP32-D quantizer. The recurring tradeoff
  // is: rank splitting doubles the grid for small M but reads input twice; more warps shorten
  // each warp's K stream but increase register/shared pressure; fewer reduction buffers shrink
  // shared memory but add deterministic reduction phases. The selected point is the fastest
  // stable configuration per M bucket, not a per-shape special case.
  //
  // rank <= 48: the accumulator and B footprint are small enough for one 8-warp CTA, so there
  // is no benefit from duplicating the input through a rank split.
  //
  // rank = 64:
  //   2K grid is only 64 CTAs, so use 8 warps (more K parallelism) plus a two-CTA rank split.
  //   4K has enough work for 4 warps; the smaller CTA wins over 8 warps in measured latency.
  //   8K and above already fill the GPU, so use the single-CTA 8-warp baseline and avoid the
  //   second input read.
  //
  // rank = 80/96/112: 4 warps limit register and shared pressure from the larger accumulator;
  // these ranks were already competitive without rank splitting.
  //
  // rank = 128:
  //   2K needs both a doubled grid and more K parallelism -> 8 warps + rank split + 2 buffers.
  //   The two reduction buffers cut per-warp D partial shared memory and made the occupancy
  //   gain measurable versus the original one-buffer-per-warp layout.
  //   4K is the cross-over point where 4 warps beat 8 warps at the same two-way rank split.
  //   8K keeps the rank split (grid still benefits) but returns to 8 warps for K throughput.
  //   16K does not split rank: duplicating its much larger input costs more than the extra
  //   grid helps, and the single-CTA 8-warp path is already at native parity.
  if (rank == 0) {
    launch_quantize_w4a4<0, 8>(args);
  } else if (rank == 16) {
    launch_quantize_w4a4<1, 8>(args);
  } else if (rank == 32) {
    launch_quantize_w4a4<2, 8>(args);
  } else if (rank == 48) {
    launch_quantize_w4a4<3, 8>(args);
  } else if (rank == 64) {
    if (M <= 2048) {
      launch_quantize_w4a4<4, 8, 8, 2>(args);
    } else if (M <= 4096) {
      launch_quantize_w4a4<4, 4, 4, 2>(args);
    } else {
      launch_quantize_w4a4<4, 8>(args);
    }
  } else if (rank == 80) {
    launch_quantize_w4a4<5, 4>(args);
  } else if (rank == 96) {
    launch_quantize_w4a4<6, 4>(args);
  } else if (rank == 112) {
    launch_quantize_w4a4<7, 4>(args);
  } else {
    if (M <= 2048) {
      launch_quantize_w4a4<8, 8, 2, 2>(args);
    } else if (M <= 4096) {
      launch_quantize_w4a4<8, 4, 2, 2>(args);
    } else if (M <= 8192) {
      launch_quantize_w4a4<8, 8, 2, 2>(args);
    } else {
      launch_quantize_w4a4<8, 8, 2>(args);
    }
  }
  checkCUDA(cudaGetLastError());
}

}  // namespace svdq::cute_kernels
