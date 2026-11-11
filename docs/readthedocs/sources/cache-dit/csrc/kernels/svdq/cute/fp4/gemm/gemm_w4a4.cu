#include "gemm_w4a4.h"

#include <stdexcept>
#include <type_traits>

#include "common.h"
#include "gemm_w4a4.cuh"

namespace svdq::cute_kernels {
namespace {

// Row-major schedule thresholds measured on BN=256/BK=128. The BN=128 fallback shares the rule but
// was not part of the A/B matrix.
constexpr int kMainFirstMinM = 16384;
constexpr int kMainFirstWideM = 8192;
constexpr int kMainFirstWideN = 6144;

// Dispatches one compiled tile configuration. `LoraLayout::kNative` always runs main-first because
// its gather reads gmem directly into MMA registers and never enters the residual inverse-scale
// division. For row-major BF16, the measured split is: 16K rows always run main-first; at 8K rows,
// N>=6144 runs main-first; all smaller row/column combinations keep LoRA-first so its term can
// overlap the A/SFA prefetch.
template <int kBN, int kStages, int kBK>
void launch_gemm_w4a4_config(void const *act, void const *wgt, void *out,
                             cutlass::float_ue4m3_t const *act_sf,
                             cutlass::float_ue4m3_t const *wgt_sf, void const *lora_act_in,
                             cutlass::bfloat16_t const *lora_up, cutlass::bfloat16_t const *bias,
                             cutlass::bfloat16_t const *wcscales, F32DLoraScales scales,
                             float alpha, int M, int N, int K, int rank, LoraLayout lora_layout,
                             cudaStream_t stream) {
  using ElementLora = cutlass::bfloat16_t;
  auto launch = [&](auto native_fragment_tag, auto lora_first_tag) {
    constexpr bool kNativeFragment = decltype(native_fragment_tag)::value;
    constexpr bool kLoRAFirst = decltype(lora_first_tag)::value;
    launch_gemm_w4a4<kBN, kStages, kBK, kLoRAFirst, kNativeFragment>(
      act, wgt, out, act_sf, wgt_sf,
      kNativeFragment ? nullptr : static_cast<ElementLora const *>(lora_act_in),
      kNativeFragment ? static_cast<float const *>(lora_act_in) : nullptr, lora_up, bias, wcscales,
      scales, alpha, M, N, K, rank, stream);
  };
  if (lora_layout == LoraLayout::kNative) {
    launch(std::true_type{}, std::false_type{});
  } else if (M >= kMainFirstMinM || (M >= kMainFirstWideM && N >= kMainFirstWideN)) {
    launch(std::false_type{}, std::false_type{});
  } else {
    launch(std::false_type{}, std::true_type{});
  }
}

}  // namespace

// Host entry for the SM120 CuTe path. Validation happens before any launch so unsupported dtypes
// or layouts fail instead of being misinterpreted by a TMA descriptor. Activation A/SFA keep the
// native 256-row MMA-fragment pack and are loaded G2R; weights use standard row-major E2M1 codes
// plus CUTLASS 128x4 SF atoms and are loaded by TMA. The independent CuTe activation producer emits
// the native pack plus BF16 D directly, while the native producer emits FP32 fragment D that
// `LoraLayout::kNative` lets the GEMM consume in place, so neither workflow needs a LoRA finalize
// or extra repack step.
//
// Two tile specializations are compiled:
//   BN=256, stages=3, BK=128: default when N/K alignment allows it; rank tiles have 32 columns.
//   BN=128, stages=4, BK=64:  fallback for narrower geometry; rank tiles have 16 columns.
// BK controls both the number of NVFP4 atoms per stage and the LoRA overlay width (BK/4), which
// is why rank divisibility is part of configuration selection. No runtime repack is performed.
void gemm_w4a4(svdq::Tensor act, svdq::Tensor wgt, svdq::Tensor out, svdq::Tensor ascales,
               svdq::Tensor wscales, svdq::Tensor lora_act_in, svdq::Tensor lora_up,
               svdq::Tensor bias, svdq::Tensor wcscales, std::vector<float> lora_scales,
               float alpha, LoraLayout lora_layout) {
  auto require = [](bool condition, const char *message) {
    if (!condition) {
      throw std::runtime_error(spdlog::fmt_lib::format("gemm_w4a4: {}", message));
    }
  };

  require(act.valid() && act.ndims() == 2, "act must be a valid 2D tensor");
  require(wgt.valid() && wgt.ndims() == 2, "wgt must be a valid 2D tensor");
  require(out.valid() && out.ndims() == 2, "out must be a valid 2D tensor");
  require(ascales.valid() && wscales.valid(), "ascales/wscales must be valid tensors");

  const int M = act.size(0);
  const int K = act.size(1) * 2;
  const int N = wgt.size(0);
  const int rank = lora_up.valid() ? lora_up.size(1) : 0;

  require(act.dtype() == svdq::Tensor::INT8 && wgt.dtype() == svdq::Tensor::INT8,
          "act/wgt must be packed int8/uint8 bytes");
  require(ascales.dtype() == svdq::Tensor::FP8_E4M3 && wscales.dtype() == svdq::Tensor::FP8_E4M3,
          "ascales/wscales must be FP8_E4M3");
  require(out.dtype() == svdq::Tensor::BF16,
          "out must be BF16; only the BF16 epilogue is wired up");
  require(act.is_cuda() && wgt.is_cuda() && out.is_cuda() && ascales.is_cuda() && wscales.is_cuda(),
          "act/wgt/out/ascales/wscales must be CUDA tensors");
  require(act.is_contiguous() && wgt.is_contiguous() && out.is_contiguous() &&
            ascales.is_contiguous() && wscales.is_contiguous(),
          "act/wgt/out/ascales/wscales must be contiguous");

  require(wgt.size(1) == act.size(1), "wgt must have the same packed K extent as act");
  require(out.size(0) == M && out.size(1) == N, "out must be [M, N]");
  require(
    M % 256 == 0 && N % 128 == 0,
    "M must be a multiple of 256 for the native activation pack; N must be a multiple of 128");
  require(K % 64 == 0, "K must be a multiple of 64");
  require(ascales.numel() == sf_buffer_bytes(M, K),
          "ascales must be the native interleaved activation scale pack for [M, K]");
  require(wscales.numel() == sf_buffer_bytes(N, K),
          "wscales must be a flat 128x4 512B atom buffer for [N, K]");

  require(lora_act_in.valid() == lora_up.valid(), "lora_act_in and lora_up must agree");
  if (rank > 0) {
    require(rank % 16 == 0, "R must be a multiple of 16");
    require(lora_up.ndims() == 2 && lora_up.size(0) == N && lora_act_in.ndims() == 2 &&
              lora_act_in.size(0) == M,
            "LoRA tensors must be [N, R] and [M, R]");
    require(lora_up.dtype() == svdq::Tensor::BF16, "lora_up must be BF16");
    require(lora_act_in.dtype() ==
              (lora_layout == LoraLayout::kNative ? svdq::Tensor::FP32 : svdq::Tensor::BF16),
            lora_layout == LoraLayout::kNative
              ? "lora_act_in must be the packed FP32 fragment D for LoraLayout::kNative"
              : "lora_act_in must be BF16 for LoraLayout::kRowMajor");
    require(lora_up.is_cuda() && lora_act_in.is_cuda() && lora_up.is_contiguous() &&
              lora_act_in.is_contiguous(),
            "LoRA tensors must be contiguous CUDA tensors");
  }
  if (bias.valid()) {
    require(bias.numel() == static_cast<size_t>(N) && bias.dtype() == svdq::Tensor::BF16 &&
              bias.is_cuda() && bias.is_contiguous(),
            "bias must be a contiguous BF16 [N] tensor");
  }
  if (wcscales.valid()) {
    require(wcscales.numel() == static_cast<size_t>(N) && wcscales.dtype() == svdq::Tensor::BF16 &&
              wcscales.is_cuda() && wcscales.is_contiguous(),
            "wcscales must be a contiguous BF16 [N] tensor");
  }

  // The native fragment path folds one scale per 16-rank group into its gmem gather, so the scale
  // table has to cover the group count that this rank implies.
  F32DLoraScales scales{};
  if (lora_layout == LoraLayout::kNative && rank > 0) {
    const int groups = rank / 16;
    require(groups <= 8, "LoraLayout::kNative supports at most 8 rank groups (R <= 128)");
    require(static_cast<int>(lora_scales.size()) >= groups,
            "lora_scales must provide one value per 16-rank group");
    for (int i = 0; i < groups; ++i) scales.values[i] = lora_scales[i];
  }

  const cudaDeviceProp *prop = getCurrentDeviceProperties();
  require(prop->major * 10 + prop->minor >= 120, "gemm_w4a4 requires an SM120 (or newer) device");

  using ElementSF = cutlass::float_ue4m3_t;
  using ElementLora = cutlass::bfloat16_t;

  const void *lora_act_ptr = rank > 0 ? lora_act_in.data_ptr() : nullptr;
  auto const *lora_up_ptr =
    rank > 0 ? static_cast<ElementLora const *>(lora_up.data_ptr()) : nullptr;

  if (N % 256 == 0 && K % 128 == 0) {
    require(rank % 32 == 0, "rank must be a multiple of 32 when BN=256 and BK=128");
    launch_gemm_w4a4_config<256, 3, 128>(
      act.data_ptr(), wgt.data_ptr(), out.data_ptr(),
      static_cast<ElementSF const *>(ascales.data_ptr()),
      static_cast<ElementSF const *>(wscales.data_ptr()), lora_act_ptr, lora_up_ptr,
      bias.valid() ? static_cast<ElementLora const *>(bias.data_ptr()) : nullptr,
      wcscales.valid() ? static_cast<ElementLora const *>(wcscales.data_ptr()) : nullptr, scales,
      alpha, M, N, K, rank, lora_layout, getCurrentCUDAStream());
  } else {
    launch_gemm_w4a4_config<128, 4, 64>(
      act.data_ptr(), wgt.data_ptr(), out.data_ptr(),
      static_cast<ElementSF const *>(ascales.data_ptr()),
      static_cast<ElementSF const *>(wscales.data_ptr()), lora_act_ptr, lora_up_ptr,
      bias.valid() ? static_cast<ElementLora const *>(bias.data_ptr()) : nullptr,
      wcscales.valid() ? static_cast<ElementLora const *>(wcscales.data_ptr()) : nullptr, scales,
      alpha, M, N, K, rank, lora_layout, getCurrentCUDAStream());
  }
  checkCUDA(cudaGetLastError());
}

}  // namespace svdq::cute_kernels
