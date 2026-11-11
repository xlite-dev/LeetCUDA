#pragma once

#include "common.cuh"
#include "../native/prefill.cuh"

namespace ffpa_cute {

// Online safe softmax (no bias, no dropout). Baseline hot path.
//
// FA-4 conditional rescaling (paper §3.1.4 /
// SoftmaxSm100.update_row_max_from_local):
//   FA-4 stores raw max; its log2-domain scale factor is
//     acc_scale_ = (m_old - m_new) * scale_log2
//   with skip condition acc_scale_ >= -rescale_threshold.
//   ffpa stores row_max in log2 domain already: row_max = max(scores * scale)
//   where the caller pre-multiplied scale by FFPA_M_LOG2E (see fwd_sm120.cuh
//   `scale *= FFPA_M_LOG2E`). Therefore row_max[row] - next_max IS acc_scale_
//   directly — no extra * scale_log2 needed. Skip condition becomes
//     row_max[row] - next_max >= -rescale_threshold   (threshold 8.0 =
//     log2(256))
//   Sign note: log2_diff = m_old - m_new <= 0 always, since next_max =
//     max(old, tile_max) >= old. The paper's skip test bounds the *positive*
//     max-growth, m_new - m_old <= tau; negating both sides gives
//     m_old - m_new >= -tau, i.e. log2_diff >= -rescale_threshold. We compare
//     against -threshold (not +threshold) because log2_diff points the opposite
//     way from the threshold. FA-4's acc_scale_ = (old-new)*scale_log2 is
//     likewise non-positive and tested as acc_scale_ >= -threshold.
//   Skip actions map 1:1 to FA-4:
//     row_scale = 1.0          <->  acc_scale = 1.0
//     row_max not updated      <->  row_max_new = row_max_old
//     P uses stale row_max     <->  row_max_safe = row_max_old
//   Equivalence: O and row_sum accumulate with the same stale max; the epilogue
//   O / row_sum cancels all deferred scaling (FA-4 finalize does the same).
template <typename ScoresTensor, typename CoordTensor, int kRows>
__device__ __forceinline__ void online_safe_softmax(
    ScoresTensor& scores, const CoordTensor& tScS_rc, float scale,
    float* row_max, float* row_sum, float* row_scale,
    float rescale_threshold = 0.0f) {
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    float tile_max = -INFINITY;
#pragma unroll
    for (int col = 0; col < cute::size<1>(scores); ++col)
      tile_max = fmaxf(tile_max, scores(row, col) * scale);
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
    const float next_max = fmaxf(row_max[row], tile_max);
    // log2_diff == FA-4 acc_scale_: already in log2 domain, no * scale_log2.
    const float log2_diff = row_max[row] - next_max;
    float eff_max = next_max;
    if (rescale_threshold > 0.0f && log2_diff >= -rescale_threshold) {
      row_scale[row] = 1.0f;
      eff_max = row_max[row];  // stale max; row_max NOT updated
    } else {
      row_scale[row] = exp2f(log2_diff);  // exp(<0) -> scale < 1.0
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

// Cross-N-warp online softmax for M4N2 layout.
// Each N-warp holds half the Bc columns; row-max and row-sum must be reduced
// across peer warps (warp_id ^ 4) via SMEM exchange.
// smem_exchange layout: [8 warps][16 rows] floats per region (max then sum).
// Precondition: caller has applied masking and scale *= FFPA_M_LOG2E.
// One barrier only (max exchange): the peer sum is NOT read here — the
// caller's P write-read __syncthreads() (stmatrix -> LDSM_N) also publishes
// the sum writes, so finalize_row_sum_m4n2 runs after it and reuses that
// barrier. Saves one CTA barrier per KV tile vs. the 3-sync version.
template <typename ScoresTensor, typename CoordTensor, int kRows,
          int kNumWarps = 8>
__device__ __forceinline__ void online_safe_softmax_m4n2(
    ScoresTensor& scores, const CoordTensor& tScS_rc, float scale,
    float* row_max, float* row_sum, float* row_scale, float* smem_exchange,
    int warp_id, int lane_id, float rescale_threshold = 0.0f) {
  // warp_id layout: m_warp = warp_id % 4, n_warp = warp_id / 4
  // peer = warp_id ^ 4 (flips n_warp bit)
  const int peer_warp = warp_id ^ 4;
  // m16n8k16 C-fragment: 4 lanes share one row. row_base = lane_id/4 gives the
  // first owned row; the second (kRows==2) is row_base + 8.
  const bool is_writer = (lane_id % 4 == 0);
  const int row_base = lane_id / 4;
  // Max and sum use SEPARATE SMEM regions to avoid cross-warp RAW hazard:
  // without separation, a fast warp could overwrite max with sum before the
  // peer warp reads it (no __syncthreads between max-read and sum-write).
  constexpr int kMaxSlots = kNumWarps * 16;

#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    float tile_max = -INFINITY;
#pragma unroll
    for (int col = 0; col < cute::size<1>(scores); ++col)
      tile_max = fmaxf(tile_max, scores(row, col) * scale);
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
    const int row_local = row_base + row * 8;
    if (is_writer)
      smem_exchange[warp_id * 16 + row_local] = tile_max;
  }
  __syncthreads();

#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    const int row_local = row_base + row * 8;
    float tile_max = smem_exchange[warp_id * 16 + row_local];
    float peer_max = smem_exchange[peer_warp * 16 + row_local];
    float global_tile_max = fmaxf(tile_max, peer_max);

    const float next_max = fmaxf(row_max[row], global_tile_max);
    const float log2_diff = row_max[row] - next_max;
    float eff_max = next_max;
    if (rescale_threshold > 0.0f && log2_diff >= -rescale_threshold) {
      row_scale[row] = 1.0f;
      eff_max = row_max[row];
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
    if (is_writer)
      smem_exchange[kMaxSlots + warp_id * 16 + row_local] = tile_sum;
  }
}

// M4N2 softmax phase 2: fold peer tile sums into row_sum. Called AFTER the
// caller's P write-read __syncthreads(), which doubles as the barrier that
// publishes the sum writes above.
template <int kRows, int kNumWarps = 8>
__device__ __forceinline__ void finalize_row_sum_m4n2(float* row_sum,
                                                      float* row_scale,
                                                      float* smem_exchange,
                                                      int warp_id,
                                                      int lane_id) {
  const int peer_warp = warp_id ^ 4;
  const int row_base = lane_id / 4;
  constexpr int kMaxSlots = kNumWarps * 16;

#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    const int row_local = row_base + row * 8;
    float local_sum = smem_exchange[kMaxSlots + warp_id * 16 + row_local];
    float peer_sum = smem_exchange[kMaxSlots + peer_warp * 16 + row_local];
    row_sum[row] = row_sum[row] * row_scale[row] + (local_sum + peer_sum);
  }
}

// FP8 fixed-P-scale variant of online_safe_softmax_m4n2 (Mode B companion).
// Folds exp_offset = log2(vs*448) into exp2 so emitted P is already in the
// P*vs*448 domain; tile_sum is scaled by inv_exp_factor = 1/(vs*448) before
// writing to smem_exchange, so finalize_row_sum_m4n2 produces row_sum in the
// true probability domain. Uses fp32 tile_sum (kRowSumViaMma=false path);
// finalize_row_sum_m4n2 is called unchanged after the P STSM->LDSM_N barrier.
template <typename ScoresTensor, typename CoordTensor, int kRows,
          bool kMaxScaleAfter = false, int kNumWarps = 8>
__device__ __forceinline__ void online_softmax_fp8_fixed_m4n2(
    ScoresTensor& scores, const CoordTensor& tScS_rc, float scale,
    float* row_max, float* row_sum, float* row_scale, float* smem_exchange,
    int warp_id, int lane_id, float exp_offset, float inv_exp_factor,
    float rescale_threshold = 0.0f) {
  const int peer_warp = warp_id ^ 4;
  const bool is_writer = (lane_id % 4 == 0);
  const int row_base = lane_id / 4;
  constexpr int kMaxSlots = kNumWarps * 16;

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
      tile_max *= scale;
    const int row_local = row_base + row * 8;
    if (is_writer)
      smem_exchange[warp_id * 16 + row_local] = tile_max;
  }
  __syncthreads();

#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    const int row_local = row_base + row * 8;
    float tile_max = smem_exchange[warp_id * 16 + row_local];
    float peer_max = smem_exchange[peer_warp * 16 + row_local];
    float global_tile_max = fmaxf(tile_max, peer_max);

    const float next_max = fmaxf(row_max[row], global_tile_max);
    const float log2_diff = row_max[row] - next_max;
    float eff_max = next_max;
    if (rescale_threshold > 0.0f && log2_diff >= -rescale_threshold) {
      row_scale[row] = 1.0f;
      eff_max = row_max[row];
    } else {
      row_scale[row] = exp2f(log2_diff);
      row_max[row] = next_max;
    }

    // exp_offset folds vs*448 into the exp2 domain: P = exp2(s*scale - eff_max
    // + exp_offset) = exp2(s*scale - (eff_max - exp_offset)).
    const float max_minus_offset = eff_max - exp_offset;
    float tile_sum = 0.0f;
#pragma unroll
    for (int col = 0; col < cute::size<1>(scores); ++col) {
      const float p = exp2f(scores(row, col) * scale - max_minus_offset);
      scores(row, col) = p;
      tile_sum += p;
    }
    tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 1);
    tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 2);
    // inv_exp_factor converts P*vs*448 domain sum to true probability sum.
    if (is_writer)
      smem_exchange[kMaxSlots + warp_id * 16 + row_local] =
          tile_sum * inv_exp_factor;
  }
}

// Online softmax with additive bias fused into the row-max pass.
template <typename ScoresTensor, typename CoordTensor, int kRows>
__device__ __forceinline__ void online_softmax_bias(
    ScoresTensor& scores, const CoordTensor& tScS_rc, float scale,
    float* row_max, float* row_sum, float* row_scale,
    const void* __restrict__ attn_bias, int attn_bias_dtype, int stride_b,
    int stride_h, int stride_m, int stride_n, int Nb_id, int Nh_id, int Br_base,
    int kv_tile, int kBc, float inv_scale, float rescale_threshold = 0.0f) {
  const int bias_base = Nb_id * stride_b + Nh_id * stride_h;
  const int bc_base = kv_tile * kBc;
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    const int q_row = Br_base + cute::get<0>(tScS_rc(row, 0));
    const int row_off = bias_base + q_row * stride_m;
    float tile_max = -INFINITY;
#pragma unroll
    for (int col = 0; col < cute::size<1>(scores); ++col) {
      const int k_col = bc_base + cute::get<1>(tScS_rc(row, col));
      scores(row, col) += ffpa::prefill::load_attn_bias_value(
                              attn_bias, attn_bias_dtype,
                              (long long)(row_off + k_col * stride_n)) *
                          inv_scale;
      tile_max = fmaxf(tile_max, scores(row, col) * scale);
    }
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
    const float next_max = fmaxf(row_max[row], tile_max);
    // FA-4 conditional rescaling; see online_safe_softmax for the domain
    // mapping.
    const float log2_diff = row_max[row] - next_max;
    float eff_max = next_max;
    if (rescale_threshold > 0.0f && log2_diff >= -rescale_threshold) {
      row_scale[row] = 1.0f;
      eff_max = row_max[row];
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

// Online softmax with dropout fused into the exp2 pass.
template <typename ScoresTensor, typename CoordTensor, int kRows>
__device__ __forceinline__ void online_softmax_dropout(
    ScoresTensor& scores, const CoordTensor& tScS_rc, float scale,
    float* row_max, float* row_sum, float* row_scale, float dropout_p,
    unsigned long long philox_seed, unsigned long long philox_offset, int Nb_id,
    int Nh, int Nh_id, int Nq, int Nkv, int Br_base, int kv_tile, int kBc,
    float rescale_threshold = 0.0f) {
  const float keep_scale = 1.0f / (1.0f - dropout_p);
  const int head_base = (Nb_id * Nh + Nh_id) * Nq;
  const int bc_base = kv_tile * kBc;
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    float tile_max = -INFINITY;
#pragma unroll
    for (int col = 0; col < cute::size<1>(scores); ++col)
      tile_max = fmaxf(tile_max, scores(row, col) * scale);
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
    const float next_max = fmaxf(row_max[row], tile_max);
    // FA-4 conditional rescaling; see online_safe_softmax for the domain
    // mapping.
    const float log2_diff = row_max[row] - next_max;
    float eff_max = next_max;
    if (rescale_threshold > 0.0f && log2_diff >= -rescale_threshold) {
      row_scale[row] = 1.0f;
      eff_max = row_max[row];
    } else {
      row_scale[row] = exp2f(log2_diff);
      row_max[row] = next_max;
    }
    const int q_row = Br_base + cute::get<0>(tScS_rc(row, 0));
    const int row_off = (head_base + q_row) * Nkv;
    float tile_sum = 0.0f;
#pragma unroll
    for (int col = 0; col < cute::size<1>(scores); ++col) {
      float p = exp2f(scores(row, col) * scale - eff_max);
      const int k_col = bc_base + cute::get<1>(tScS_rc(row, col));
      const unsigned long long off =
          philox_offset + (unsigned long long)(row_off + k_col);
      const float u =
          ffpa::prefill::curand_uniform_from_element_offset(philox_seed, off);
      p = (u > dropout_p) ? p * keep_scale : 0.0f;
      scores(row, col) = p;
      tile_sum += p;
    }
    tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 1);
    tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 2);
    row_sum[row] = row_sum[row] * row_scale[row] + tile_sum;
  }
}

}  // namespace ffpa_cute
