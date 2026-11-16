#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cute/tensor.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/numeric_types.h>

#include <cstddef>
#include <cstdint>

namespace cache_dit::fp4 {

// Numeric and layout constants shared by the quantizer and GEMM validation.
// E2M1 stores values in [-6,6]; UE4M3 block scales top out at 448. kFp4ScaleMax
// is their product, so global=row_amax/kFp4ScaleMax leaves enough range for the
// subsequent per-group UE4M3 scale. kSFVec=16 is the NVFP4 block size. CUTLASS
// groups four such scales (K64) and 128 rows into each 512-byte scale atom.
constexpr float kE2M1Max = 6.0f;
constexpr float kUe4m3Max = 448.0f;
constexpr float kFp4ScaleMax = kE2M1Max * kUe4m3Max;
constexpr int kSFVec = 16;
constexpr int kRowBlock = 128;
constexpr int kSFVecPerBlk = 4;

// One 128-bit global/shared-memory transaction. The union does not convert or
// swizzle data: elem[0..7] remain the eight consecutive BF16 input values.
union Vec8BF16 {
  uint4 raw;
  __nv_bfloat16 elem[8];
};
static_assert(sizeof(Vec8BF16) == 16, "Vec8BF16 must be 128-bit");

// Convert two FP32 values with the hardware round-to-nearest, saturating E2M1
// instruction. The first argument becomes the high nibble (odd K element), the
// second becomes the low nibble (even K element). Saturation is intentional:
// normalized values are clamped to the finite E2M1 range rather than becoming
// infinities. This helper implements quantization only; GEMM decoding is done
// by the SM120 block-scaled MMA.
__device__ __forceinline__ unsigned char cvt_f2_to_e2m1x2(float hi, float lo) {
  unsigned int tmp;
  asm volatile(
    "{\n"
    ".reg .b8 byte0;\n"
    ".reg .b8 byte1;\n"
    ".reg .b8 byte2;\n"
    ".reg .b8 byte3;\n"
    "cvt.rn.satfinite.e2m1x2.f32 byte0, %1, %2;\n"
    "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
    "}\n"
    : "=r"(tmp)
    : "f"(hi), "f"(lo));
  return static_cast<unsigned char>(tmp & 0xffu);
}

// Convert a block maximum/6 to UE4M3. The quantizer immediately converts the
// resulting byte back to FP32 before computing inv_block; do not replace this
// with a reciprocal of the unrounded maximum.
__device__ __forceinline__ unsigned char cvt_f2_to_ue4m3(float x) {
  return static_cast<unsigned char>(
    __nv_cvt_float2_to_fp8x2(make_float2(x, 0.0f), __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float cvt_ue4m3_to_f32(unsigned char scale) {
  return __half2float(__nv_cvt_fp8_to_halfraw(scale, __NV_E4M3));
}

// Byte address of scale (row=mn, K16 group=s) in the canonical buffer.
// A 512-byte atom covers 128 rows x four K16 groups. Atoms advance fastest over
// K64, then over 128-row blocks. Within an atom, four row groups of 32 are
// interleaved in chunks of four bytes, matching CUTLASS's SFA/SFB TMA layout.
__host__ __device__ constexpr int sf_byte_offset(int mn, int s, int nb_k) {
  return (mn / kRowBlock) * nb_k * 512 + (s / kSFVecPerBlk) * 512 + (mn % 32) * 16 +
         ((mn / 32) % 4) * 4 + (s % kSFVecPerBlk);
}

// Required allocation, including zeroed padding to complete 128-row atoms.
// K must be a multiple of 64 in the public quantize contract.
__host__ __device__ constexpr size_t sf_buffer_bytes(int mn, int k) {
  return static_cast<size_t>((mn + kRowBlock - 1) / kRowBlock) * (k / 64) * 512;
}

CUTE_HOST_DEVICE auto sf_gmem_layout(int mn, int k) {
  using Config = cutlass::detail::Sm1xxBlockScaledConfig<kSFVec>;
  return cute::tile_to_shape(typename Config::SfAtom{}, cute::make_shape(mn, k),
                             cute::Step<cute::_2, cute::_1>{});
}

}  // namespace cache_dit::fp4
