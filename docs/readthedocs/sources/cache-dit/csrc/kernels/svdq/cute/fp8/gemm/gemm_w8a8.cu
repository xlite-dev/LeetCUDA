#include "gemm_w8a8.h"

#include <stdexcept>

#include "common.h"
#include "gemm_w8a8.cuh"

namespace svdq::cute_kernels {

// Host entry for the SM120 CuTe W8A8 path. Validation happens before any launch so unsupported
// dtypes or shapes fail instead of being misinterpreted by a TMA descriptor. One tile configuration
// is compiled (BM=128/BN=128/BK=128, two main stages plus a two-stage LoRA ring, 96 KiB of smem)
// and the LoRA rank is a runtime argument because it only changes the number of rank slices, not
// the MMA or smem types.
void gemm_w8a8(Tensor act, Tensor wgt, Tensor out, Tensor ascales, Tensor wscales,
               Tensor lora_act_in, Tensor lora_up, Tensor bias) {
  auto require = [](bool condition, const char *message) {
    if (!condition) {
      throw std::runtime_error(spdlog::fmt_lib::format("gemm_w8a8: {}", message));
    }
  };

  int device = 0;
  int major = 0;
  checkCUDA(cudaGetDevice(&device));
  checkCUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  require(major >= 12, "the W8A8 GEMM requires an sm120 or newer device");

  require(act.valid() && act.ndims() == 2, "act must be a valid 2D tensor");
  require(wgt.valid() && wgt.ndims() == 2, "wgt must be a valid 2D tensor");
  require(out.valid() && out.ndims() == 2, "out must be a valid 2D tensor");
  require(ascales.valid() && wscales.valid(), "ascales/wscales must be valid tensors");
  require(lora_up.valid() == lora_act_in.valid(),
          "lora_act_in and lora_up must be supplied together");
  if (lora_up.valid()) {
    require(lora_up.ndims() == 2 && lora_act_in.ndims() == 2,
            "lora_act_in/lora_up must be 2D tensors");
  }

  const int M = act.size(0);
  const int K = act.size(1);
  const int N = wgt.size(0);
  const int rank = lora_up.valid() ? lora_up.size(1) : 0;

  require(act.dtype() == Tensor::FP8_E4M3 && wgt.dtype() == Tensor::FP8_E4M3,
          "act/wgt must be FP8_E4M3");
  require(out.dtype() == Tensor::BF16, "out must be BF16");
  require(ascales.dtype() == Tensor::FP32 && wscales.dtype() == Tensor::FP32,
          "ascales/wscales must be FP32");
  require(act.is_cuda() && wgt.is_cuda() && out.is_cuda(), "act/wgt/out must be CUDA tensors");
  require(act.is_contiguous() && wgt.is_contiguous() && out.is_contiguous(),
          "act/wgt/out must be contiguous");
  require(ascales.is_contiguous() && wscales.is_contiguous(), "ascales/wscales must be contiguous");
  require(ascales.is_cuda() && wscales.is_cuda(), "ascales/wscales must be CUDA tensors");
  require(wgt.size(1) == K, "wgt must have the same K extent as act");
  require(out.size(0) == M && out.size(1) == N, "out must have shape [M, N]");
  require(ascales.numel() == M && wscales.numel() == N,
          "ascales/wscales must have one entry per row/column");
  require(M % 128 == 0 && N % 128 == 0 && K % 128 == 0, "M, N and K must all be multiples of 128");
  require(rank == 0 || rank == 32 || rank == 64 || rank == 128, "rank must be 0, 32, 64 or 128");
  if (rank > 0) {
    require(lora_act_in.valid() && lora_act_in.ndims() == 2, "lora_act_in must be [M, R]");
    require(lora_act_in.size(0) == M && lora_act_in.size(1) == rank,
            "lora_act_in must have shape [M, R]");
    require(lora_up.size(0) == N, "lora_up must have shape [N, R]");
    require(lora_act_in.dtype() == Tensor::BF16 && lora_up.dtype() == Tensor::BF16,
            "lora tensors must be BF16");
    require(lora_act_in.is_contiguous() && lora_up.is_contiguous(),
            "lora tensors must be contiguous");
  }
  if (bias.valid()) {
    require(bias.dtype() == Tensor::BF16 && bias.numel() == N, "bias must be BF16 [N]");
    require(bias.is_contiguous(), "bias must be contiguous");
  }

  launch_gemm_w8a8<128, 2, 128>(
    act.data_ptr(), wgt.data_ptr(), out.data_ptr(), static_cast<float const *>(ascales.data_ptr()),
    static_cast<float const *>(wscales.data_ptr()),
    rank > 0 ? static_cast<cutlass::bfloat16_t const *>(lora_act_in.data_ptr()) : nullptr,
    rank > 0 ? static_cast<cutlass::bfloat16_t const *>(lora_up.data_ptr()) : nullptr,
    bias.valid() ? static_cast<cutlass::bfloat16_t const *>(bias.data_ptr()) : nullptr, M, N, K,
    rank, getCurrentCUDAStream());
  checkCUDA(cudaGetLastError());
}

}  // namespace svdq::cute_kernels
