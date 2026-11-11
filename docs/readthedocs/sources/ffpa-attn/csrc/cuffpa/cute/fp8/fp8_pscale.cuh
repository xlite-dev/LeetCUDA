#pragma once

#include <cuda_fp8.h>
#include <cute/tensor.hpp>

#include "reg2reg_8b.cuh"

namespace ffpa_fp8 {

// P quantization helpers for fp8 attention (softmax probabilities -> e4m3).
// Shared by any kernel that feeds a register-resident softmax P into an fp8
// PV MMA (persist-D and split-D fp8 paths).
// Reference (P rowsum via MMA):
// https://github.com/thu-ml/SageAttention/blob/main/csrc/mma.cuh
// (rowsum_f8f8f32)
//
// Why P needs a runtime scale at all:
// Q/K/V are quantized offline (blockwise scales q_scale/k_scale/v_scale).
// P = softmax(S) is a transient fp32 probability matrix, different every
// forward and every tile, so it must be quantized in-kernel:
//
//   P8    = round(P * vs / p_scale)        (1) quantize, vs folded in
//   MMA   = P8 @ V8                        (2) fp8 PV MMA, V8 = V / vs
//   O    += MMA * p_scale                  (3) dequant back to true domain
//
// Correctness of folding vs into P instead of V:
//   MMA = (P * vs / p_scale) @ (V / vs) = (P @ V) / p_scale
//   (3) => O += P @ V.  vs cancels exactly, so the pre-quantized V8 in gmem
//   is reused untouched.
//
// Overflow safety: post-softmax probabilities satisfy 0 <= P <= 1 (the
// running row max is subtracted before exp), but lazy rescaling (FA-4
// conditional rescale, threshold T in log2 domain) may emit P against a stale
// max, inflating values by up to 2^T. Fixed mode emits P*vs*448, so the e4m3
// ceiling requires 2^T * amax(V) <= 448; with T = FFPA_RESCALE_THRESHOLD_FP8
// = 4 (FA-4's fp8 choice) that holds for amax(V) <= 28. satfinite clamps any
// residual overshoot.
//
// Mode A: per-row p_scale  (kPQuantPerRow == true)
//   p_scale[row] = max_j(P[row, j]) / 448        (per query row)
// Each row's largest probability maps exactly onto the e4m3 ceiling, so every
// row exploits the full mantissa range -> best accuracy, in particular for
// "flat" rows whose max(P) << 1 (e.g. rows at the top of the causal triangle
// attending to very few keys).
// Cost:
//   * row-max reduction: one logical row is spread over the 4 peer lanes of a
//     quad (lane%4), so two __shfl_xor rounds (offsets 1, 2) complete it;
//   * the dequant factor p_scale[row] in (3) varies per row, while the MMA
//     output fragment is interleaved over many rows, so the PV result cannot
//     be rescaled in place -> it lands in a separate o_tile fragment and a
//     second pass computes o_acc += o_tile * p_scale[row];
//   * that extra 64-float o_tile buffer raises register pressure (spills)
//     and adds one zero-init + one multiply-accumulate pass per KV tile.
//
// Mode B: fixed p_scale = 1 / 448  (kPQuantPerRow == false)
//   p_scale = 1/448 for every row and tile, justified by max(P) <= 1.
// The scale is a compile-time constant, therefore:
//   * no max reduction, no shuffles, no per-row division;
//   * the dequant factor in (3) is a single global constant, so o_acc stays
//     in ONE fixed domain ("/p_scale") for the whole kernel and the PV MMA
//     can accumulate DIRECTLY into o_acc (no o_tile, no extra pass):
//       o_acc = sum_t MMA_t = sum_t (P_t @ V_t) / p_scale
//   * the epilogue performs the one and only dequant:
//       O = o_acc * p_scale / row_sum = sum_t(P_t @ V_t) / row_sum
// Trade-off: a flat row with max(P) = m << 1 only uses the e4m3 levels up to
// m*448, i.e. its quantization step is coarser relative to its own dynamic
// range than in Mode A. The absolute error stays bounded by p_scale/2 per
// probability, which is acceptable for attention averaging but measurably
// worse on causal-triangle rows -> kept selectable via template.

// Max finite value of e4m3; P scales map the assumed amax onto this ceiling.
constexpr float kE4m3Max = 448.0f;
// Fixed-mode P quant scale (upper bound max(P) <= 1).
constexpr float kFP8FixedPScale = 1.0f / kE4m3Max;

// Per-row P quant scales from the row max of a softmax-probability fragment
// (Mode A step 1). `scores` is the rowcol view of the QK C-fragment AFTER
// online softmax (values are probabilities). Each thread owns kRows logical
// rows; a row's kBc columns live in the 4 peer lanes of its quad, so xor-1 /
// xor-2 complete the row reduction. Writes p_scale[row] = row_max / 448.
template <typename ScoresTensor>
CUTE_DEVICE void pscale_per_row(ScoresTensor const& scores, float* p_scale) {
  constexpr int kRows = decltype(cute::size<0>(scores))::value;
  float row_m[kRows];
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    float m = 0.0f;
#pragma unroll
    for (int col = 0; col < cute::size<1>(scores); ++col)
      m = fmaxf(m, scores(row, col));
    row_m[row] = m;
  }
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    row_m[row] = fmaxf(row_m[row], __shfl_xor_sync(0xffffffff, row_m[row], 1));
    row_m[row] = fmaxf(row_m[row], __shfl_xor_sync(0xffffffff, row_m[row], 2));
    p_scale[row] = row_m[row] / kE4m3Max;
  }
}

// Scale P by vs/p_scale, downcast fp32 -> e4m3 in place, then reg2reg-reorg
// into the PV A-operand layout (equation (1) + layout fixup, both modes).
// kPQuantPerRow selects the multiplier:
//   true : per-row  mul[row] = vs / p_scale[row]   (guard p_scale == 0)
//   false: constant mul      = vs * 448            (== vs / kFP8FixedPScale)
// The fragment storage is aliased: the fp32 C-fragment memory becomes the
// packed e4m3 A-fragment (4x denser), so call BEFORE reinterpreting it as
// the PV A operand. `reorg` must be ffpa_fp8::ReorgC8bitToA8bit.
template <bool kPQuantPerRow, typename ScoresTensor, typename Fragment,
          typename Reorg>
CUTE_DEVICE void quantize_p_frag(ScoresTensor& scores, Fragment& tCrS, float vs,
                                 const float* p_scale, Reorg& reorg) {
  constexpr int kRows = decltype(cute::size<0>(scores))::value;
  constexpr int kSVals = decltype(cute::size(tCrS))::value;
  if constexpr (kPQuantPerRow) {
#pragma unroll
    for (int row = 0; row < kRows; ++row) {
      const float mul = (p_scale[row] == 0.0f) ? 0.0f : vs / p_scale[row];
#pragma unroll
      for (int col = 0; col < cute::size<1>(scores); ++col)
        scores(row, col) *= mul;
    }
  } else {
    const float mul = vs * kE4m3Max;
#pragma unroll
    for (int row = 0; row < kRows; ++row)
#pragma unroll
      for (int col = 0; col < cute::size<1>(scores); ++col)
        scores(row, col) *= mul;
  }
  float* f32 = tCrS.data();
  __nv_fp8_e4m3* p8 = reinterpret_cast<__nv_fp8_e4m3*>(f32);
#pragma unroll
  for (int i = 0; i < kSVals; ++i)
    p8[i] = __nv_fp8_e4m3(f32[i]);
  auto p8_frag = cute::make_tensor(
      cute::make_rmem_ptr(p8), cute::Layout<cute::Shape<cute::Int<kSVals>>>{});
  reorg(p8_frag);
}

// Fixed-mode fast path companion: the softmax already emitted P*vs*448 (see
// online_softmax_fp8_fixed), so only the f32 -> e4m3 downcast + reorg remain.
// The conversion packs two floats per instruction
// (cvt.rn.satfinite.e4m3x2.f32), halving the CVT count versus element-wise
// __nv_fp8_e4m3 construction. kSVals must be even (C fragments always are).
// Storage is aliased like quantize_p_frag: call BEFORE reinterpreting tCrS as
// the PV A operand.
template <typename Fragment, typename Reorg>
CUTE_DEVICE void quantize_p_frag_prescaled(Fragment& tCrS, Reorg& reorg) {
  constexpr int kSVals = decltype(cute::size(tCrS))::value;
  static_assert(kSVals % 2 == 0, "prescaled P quant needs an even frag size");
  float* f32 = tCrS.data();
  uint16_t* p8x2 = reinterpret_cast<uint16_t*>(f32);
#pragma unroll
  for (int i = 0; i < kSVals / 2; ++i) {
    uint16_t packed;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n"
                 : "=h"(packed)
                 : "f"(f32[2 * i + 1]), "f"(f32[2 * i]));
    p8x2[i] = packed;
  }
  auto p8_frag = cute::make_tensor(
      cute::make_rmem_ptr(reinterpret_cast<__nv_fp8_e4m3*>(f32)),
      cute::Layout<cute::Shape<cute::Int<kSVals>>>{});
  reorg(p8_frag);
}

// Mode A dequant: fold per-row p_scale back into the running output,
// o_acc += o_tile * p_scale[row]  (equation (3), per-row variant).
// o_tile holds the raw PV MMA result (P@V)/p_scale[row]; both args are
// rowcol views of the PV C fragments so each element finds its row's scale.
template <typename OrcTensor, typename TileTensor>
CUTE_DEVICE void accumulate_p_tile(OrcTensor& o_acc_rc,
                                   const TileTensor& o_tile_rc,
                                   const float* p_scale) {
#pragma unroll
  for (int row = 0; row < cute::size<0>(o_acc_rc); ++row)
#pragma unroll
    for (int col = 0; col < cute::size<1>(o_acc_rc); ++col)
      o_acc_rc(row, col) += o_tile_rc(row, col) * p_scale[row];
}

// Fixed-mode softmax that emits P already scaled into the e4m3 quant domain
// (Mode B, fast path). Replaces the separate `scores *= vs*448` quantization
// pass: with exp_offset = log2f(vs * 448),
//   exp2(s - max + exp_offset) = exp2(s - max) * vs * 448 = P * vs * 448
// which is exactly the value quantize_p_frag<false> would produce, so the
// fragment can be downcast to e4m3 directly after this call.
//
// kRowSumViaMma: when true, the fp32 tile_sum accumulation (one FADD per
// element plus the two cross-lane shfl rounds per row) is skipped; the caller
// recovers row_sum from a tensor-core row-sum over the quantized P fragment
// (pscale_rowsum_mma). row_sum is then exact w.r.t. the quantized P that the
// PV MMA consumes. When false, row_sum is accumulated here in fp32 and the
// MMA row-sum must NOT be applied.
//
// Normalization: row_sum must accumulate UNSCALED probabilities. The tile
// partial sums come out scaled by the per-tile constant vs*448 (vs varies
// per KV tile), so each row's contribution is folded back here with one
// multiply: row_sum += tile_sum * inv_exp_factor, inv_exp_factor =
// 1/(vs*448). That keeps the epilogue identity O = o_acc*(1/448)/row_sum
// valid: o_acc lives in the global "/(1/448)" domain because vs cancels in
// the PV MMA ((P*vs*448) @ (V/vs) = 448*(P@V)), while row_sum stays in the
// true probability domain.
//
// Overflow safety: emitted values are P * vs * 448 = P * amax(V), inflated by
// up to 2^rescale_threshold under lazy rescale; the fp8 threshold of 4.0
// (FFPA_RESCALE_THRESHOLD_FP8, per FA-4) bounds this at 16 * amax(V) <= 448
// for amax(V) <= 28, and satfinite clamps any residual overshoot.
//
// scores must already be in the log2 domain (caller pre-multiplied by
// softmax_scale*log2e) with masking applied; `scale` multiplies scores again
// inside the reductions (pass 1.0f when pre-scaled).
//
// kMaxScaleAfter: when true, the tile-max pass reduces raw (unscaled) scores
// and applies `scale` once after the cross-lane reduction instead of
// multiplying every element. max(scale*x) = scale*max(x) for scale > 0, so
// the result is identical up to rounding; this removes one FMUL per element
// from the softmax max-reduction critical path. Default false keeps existing
// variants bitwise unchanged.
template <bool kRowSumViaMma, typename ScoresTensor, typename CoordTensor,
          int kRows, bool kMaxScaleAfter = false>
CUTE_DEVICE void online_softmax_fp8_fixed(
    ScoresTensor& scores, const CoordTensor& tScS_rc, float scale,
    float* row_max, float* row_sum, float* row_scale, float exp_offset,
    float inv_exp_factor, float rescale_threshold) {
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    float tile_max = -INFINITY;
#pragma unroll
    for (int col = 0; col < cute::size<1>(scores); ++col) {
      if constexpr (kMaxScaleAfter)
        tile_max = fmaxf(tile_max, scores(row, col));
      else
        tile_max = fmaxf(tile_max, scores(row, col) * scale);
    }
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
    if constexpr (kMaxScaleAfter)
      tile_max *= scale;  // scale commutes with max; one FMUL per row
    const float next_max = fmaxf(row_max[row], tile_max);
    const float log2_diff = row_max[row] - next_max;
    float eff_max = next_max;
    if (rescale_threshold > 0.0f && log2_diff >= -rescale_threshold) {
      row_scale[row] = 1.0f;
      eff_max = row_max[row];  // stale max; row_max NOT updated (FA-4)
    } else {
      row_scale[row] = exp2f(log2_diff);
      row_max[row] = next_max;
    }
    // exp_offset enters here: emitted P is scaled by 2^exp_offset = vs*448.
    const float max_minus_offset = eff_max - exp_offset;
    float tile_sum = 0.0f;
#pragma unroll
    for (int col = 0; col < cute::size<1>(scores); ++col) {
      const float p = exp2f(scores(row, col) * scale - max_minus_offset);
      scores(row, col) = p;
      if constexpr (!kRowSumViaMma)
        tile_sum += p;
    }
    if constexpr (kRowSumViaMma) {
      // row_sum deferred to pscale_rowsum_mma; rescale still applies here so
      // the MMA-accumulated sum is multiplied by the same row_scale.
      row_sum[row] *= row_scale[row];
      (void)tile_sum;
    } else {
      tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 1);
      tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 2);
      row_sum[row] = row_sum[row] * row_scale[row] + tile_sum * inv_exp_factor;
    }
  }
}

// Tensor-core row sum of the quantized P fragment (Mode B companion of
// online_softmax_fp8_fixed<true>). Issues mma.sync.m16n8k32 with an all-ones
// e4m3 B operand over the SAME registers the PV MMA consumes (p8_frag, the
// reorganized A-operand view, kBc/32 k-steps of 4 uint32 each). With B[k][j]
// == 1.0 the C fragment becomes D[i][j] = sum_k P8[i][k], i.e. the exact row
// sum of the quantized probabilities; d0/d1 hold the sums of the two rows a
// thread owns (groupID / groupID+8), matching the softmax rowcol view.
// inv_exp_factor returns to the true probability domain (P*vs*448 -> P).
// Ref: SageAttention csrc/mma.cuh rowsum_f8f8f32 (no x4 factor: B is all
// ones, the mma reduces over k exactly once).
template <typename P8Tensor>
CUTE_DEVICE void pscale_rowsum_mma(const P8Tensor& p8_frag, float* row_sum,
                                   float inv_exp_factor) {
  constexpr int kKSteps = decltype(cute::size<2>(p8_frag))::value;
  float d0 = 0.0f, d1 = 0.0f;
  const uint32_t* p8 = reinterpret_cast<const uint32_t*>(p8_frag.data());
#pragma unroll
  for (int k = 0; k < kKSteps; ++k) {
    const uint32_t* s = p8 + k * 4;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, _, %1, _}, {%2, %3, %4, %5}, {%6, %7}, {%0, 0., %1, 0.};\n"
        : "+f"(d0), "+f"(d1)
        : "r"(s[0]), "r"(s[1]), "r"(s[2]), "r"(s[3]), "r"(0x38383838u),
          "r"(0x38383838u));
  }
  row_sum[0] += d0 * inv_exp_factor;
  row_sum[1] += d1 * inv_exp_factor;
}

}  // namespace ffpa_fp8
