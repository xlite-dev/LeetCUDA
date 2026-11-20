#include <c10/cuda/CUDAException.h>

#include "gemm.cuh"

namespace cache_dit::fp8 {

void launch_fp8_gemm_per_row(const void* a, const void* b, const float* a_scale,
                             const float* b_scale, const void* bias, void* out, int M, int N, int K,
                             cudaStream_t stream) {
  fp8_gemm_cutlass_rowwise<false>(a, b, a_scale, b_scale, bias, out, M, N, K, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_fp8_gemm_per_row_activation(const void* a, const void* b, const float* a_scale,
                                        const float* b_scale, const void* bias, void* out, int M,
                                        int N, int K, int64_t activation, cudaStream_t stream) {
  if (activation == kActivationGelu) {
    fp8_gemm_cutlass_rowwise_activation<false, cutlass::epilogue::thread::GELU>(
      a, b, a_scale, b_scale, bias, out, M, N, K, stream);
  } else if (activation == kActivationGeluTanh) {
    fp8_gemm_cutlass_rowwise_activation<false, cutlass::epilogue::thread::Fp8GeluTanh>(
      a, b, a_scale, b_scale, bias, out, M, N, K, stream);
  } else {
    TORCH_CHECK(false, "Unknown FP8 activation id: ", activation);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::fp8
