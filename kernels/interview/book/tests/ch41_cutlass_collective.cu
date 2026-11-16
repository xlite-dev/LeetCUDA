// ch41_cutlass_collective.cu — test kernels/interview/collective_gemm.cuh.
// Build/run on SM90. Other architectures print SKIP; SM120 FP8/FP4 cases live in cache-dit.
#include "common_test.h"
#include "collective_gemm.cuh"

#include <vector>

static void run_collective_case(int M, int N, int K) {
  using namespace leetcuda::collective_gemm;
  std::vector<float> host_a(M * K), host_b(N * K), host_d(M * N, 0.0f);
  std::vector<double> ref(M * N, 0.0);
  book_fill_rand(host_a.data(), M * K, 41);
  book_fill_rand(host_b.data(), N * K, 42);
  std::vector<ElementA> typed_a(M * K);
  std::vector<ElementB> typed_b(N * K);
  for (int i = 0; i < M * K; ++i) typed_a[i] = ElementA(host_a[i]);
  for (int i = 0; i < N * K; ++i) typed_b[i] = ElementB(host_b[i]);

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) {
        acc += static_cast<double>(float(typed_a[m * K + k])) *
               static_cast<double>(float(typed_b[n * K + k]));
      }
      ref[m * N + n] = acc;
    }
  }

  ElementA* dev_a = nullptr;
  ElementB* dev_b = nullptr;
  ElementC* dev_c = nullptr;
  ElementD* dev_d = nullptr;
  BOOK_CUDA_CHECK(cudaMalloc(&dev_a, sizeof(ElementA) * M * K));
  BOOK_CUDA_CHECK(cudaMalloc(&dev_b, sizeof(ElementB) * N * K));
  BOOK_CUDA_CHECK(cudaMalloc(&dev_c, sizeof(ElementC) * M * N));
  BOOK_CUDA_CHECK(cudaMalloc(&dev_d, sizeof(ElementD) * M * N));
  BOOK_CUDA_CHECK(cudaMemcpy(dev_a, typed_a.data(), sizeof(ElementA) * M * K,
                             cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(dev_b, typed_b.data(), sizeof(ElementB) * N * K,
                             cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemset(dev_c, 0, sizeof(ElementC) * M * N));

  cutlass::Status status = bf16(dev_a, dev_b, dev_c, dev_d, M, N, K);
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  if (status == cutlass::Status::kSuccess) {
    BOOK_CUDA_CHECK(cudaMemcpy(host_d.data(), dev_d, sizeof(ElementD) * M * N,
                               cudaMemcpyDeviceToHost));
    book_check(host_d.data(), ref.data(), M * N, TOL_F32ACC, "collective_bf16_fp32out");
  } else {
    printf("FAIL collective_bf16: status=%s\n", cutlass::cutlassGetStatusString(status));
    ++g_failures;
  }

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  cudaFree(dev_d);
}

int main() {
#if !defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  printf("SKIP(ch41): BF16 CollectiveBuilder example requires sm_90a compilation\n");
  return 0;
#else
  if (book_device_sm() != 90) {
    printf("SKIP(ch41): BF16 SM90 kernel requires an SM90 device\n");
    return 0;
  }
  run_collective_case(128, 128, 128);
  return g_failures == 0 ? 0 : 1;
#endif
}
