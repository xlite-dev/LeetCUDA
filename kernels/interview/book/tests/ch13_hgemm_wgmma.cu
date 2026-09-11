// book/tests/ch13_hgemm_wgmma.cu — ch13 最小测试：TMA + mbarrier + WGMMA（Hopper）
// 测试逻辑抽取自 notes-v2.cu test_hgemm_wgmma（L1648），参考实现由 cuBLAS F16
// 换成 CPU fp64（BOOK_PLAN §6，无 cuBLAS/cuDNN 依赖）。
// 覆盖：hgemm_wgmma_stages_tn（m64n128k16 + TMA + cuda::barrier 流水线，
// kStages=3）在 M=N=K=256 下的正确性 vs CPU fp64。
// 架构约定：WGMMA/mbarrier/TMA PTX 仅 sm_90a 可编译——本文件必须以
// -arch sm_90a 编译（非 900 的 device pass 直接 #error）；sm_120 等新架构
// 无 WGMMA（不向前兼容），运行时输出 SKIP 并返回 0。
// 完整 PASS 需 H100/H200/H800：./build_tests.sh --arch sm_90a --ch ch13。
#include <cmath>
#include <cstdio>
#include <vector>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ != 900)
#error "ch13 test must be compiled with -arch sm_90a (WGMMA requires Hopper)"
#endif

#define NOTES_V2_ENABLE_WGMMA
#include "../../hgemm.cuh"
#include "common_test.h"

// WGMMA 是 sm_90a 专属指令族：不早于 sm_90（book_require_sm 的 >= 语义可挡住），
// 也不向前兼容 sm_100/sm_120（>= 语义挡不住），因此这里做精确 sm_90 家族判断，
// SKIP 输出行格式与 common_test.h book_require_sm 一致。
static bool require_sm90_family() {
  int cur = book_device_sm();
  if (cur == 90) return true;
  printf("SKIP(ch13): requires sm_90 (Hopper WGMMA), device is sm_%d\n", cur);
  return false;
}

static void run_wgmma_test() {
  // M=N=K=256：被 BM=BN=128、BK=64 整除；K/BK=4 个 K-tile 覆盖 kStages=3 的
  // stage 回绕（0→1→2→0）。规模 <= BOOK_TEST_MAX_N=512（CPU fp64 三重循环约束）。
  constexpr int M = 256, N = 256, K = 256;
  constexpr int BM = 128, BN = 128, BK = 64, kStages = 3, kNumThreads = 256;
  static_assert(M % BM == 0 && N % BN == 0 && K % BK == 0, "tile 整除约束");
  static_assert(M <= BOOK_TEST_MAX_N && N <= BOOK_TEST_MAX_N, "CPU 参考时限约束");

  const size_t size_a = size_t(M) * K * sizeof(half);
  const size_t size_bt = size_t(N) * K * sizeof(half);  // B^T[N,K] row-major（TN 布局）
  const size_t size_c = size_t(M) * N * sizeof(half);

  // 输入压到 [-0.25, 0.25]：kernel 累加器是 f16（f16.f16.f16），舍入误差随
  // 累加值幅度增长；此量级下 C 元素 |值| 约 O(1)，f16 累加误差 ~1e-3 量级，
  // 稳落在 TOL_F16ACC=5e-2 内（与 notes-v2 用 CUBLAS_COMPUTE_16F 参考同量级）。
  std::vector<float> fa(M * K), fbt(N * K);
  book_fill_rand(fa.data(), M * K);
  book_fill_rand(fbt.data(), N * K, 0x5A5A);
  std::vector<half> h_a(M * K), h_bt(N * K);
  for (int i = 0; i < M * K; ++i) h_a[i] = __float2half(fa[i] * 0.25f);
  for (int i = 0; i < N * K; ++i) h_bt[i] = __float2half(fbt[i] * 0.25f);

  // CPU fp64 参考：C[m][n] = sum_k A[m][k] * B^T[n][k]
  std::vector<double> ref(M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k)
        acc += double(__half2float(h_a[m * K + k])) *
               double(__half2float(h_bt[n * K + k]));
      ref[m * N + n] = acc;
    }

  half *d_a, *d_bt, *d_c;
  BOOK_CUDA_CHECK(cudaMalloc(&d_a, size_a));
  BOOK_CUDA_CHECK(cudaMalloc(&d_bt, size_bt));
  BOOK_CUDA_CHECK(cudaMalloc(&d_c, size_c));
  BOOK_CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_a, cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_bt, h_bt.data(), size_bt, cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemset(d_c, 0, size_c));

  // TMA descriptor：A[M,K] row-major → box=(BK=64, BM=128)，global shape=(K, M)；
  // B^T 同理（默认模板 <BlockMajorSize=128, BlockMinorSize=64>）。
  CUtensorMap *tma_a = allocate_and_create_tensor_map(d_a, M / BM, K / BK);
  CUtensorMap *tma_b = allocate_and_create_tensor_map(d_bt, N / BN, K / BK);

  // kStages=3 × (A 16KB + B 16KB) = 96KB 动态 smem，需 opt-in。
  const size_t smem_bytes = kStages * (BM * BK + BN * BK) * sizeof(half);
  using Kernel = decltype(&hgemm_wgmma_stages_tn<64, 128, 16, BM, BN, BK,
                                                kNumThreads, kStages, false>);
  Kernel kernel = hgemm_wgmma_stages_tn<64, 128, 16, BM, BN, BK, kNumThreads,
                                        kStages, false>;
  BOOK_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  dim3 grid(N / BN, M / BM), block(kNumThreads);
  kernel<<<grid, block, smem_bytes>>>(M, N, K, d_c, tma_a, tma_b);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<half> h_c(M * N);
  std::vector<float> h_out(M * N);
  BOOK_CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_c, cudaMemcpyDeviceToHost));
  for (int i = 0; i < M * N; ++i) h_out[i] = __half2float(h_c[i]);
  book_check(h_out.data(), ref.data(), M * N, TOL_F16ACC,
             "hgemm wgmma m64n128k16 TMA+WS (F16Acc)");

  BOOK_CUDA_CHECK(cudaFree(d_a));
  BOOK_CUDA_CHECK(cudaFree(d_bt));
  BOOK_CUDA_CHECK(cudaFree(d_c));
  BOOK_CUDA_CHECK(cudaFree(tma_a));
  BOOK_CUDA_CHECK(cudaFree(tma_b));
}

int main() {
  if (!require_sm90_family())
    return 0;  // 本机无 Hopper：SKIP，退出码 0
  run_wgmma_test();
  if (g_failures == 0) {
    printf("ALL OK\n");
    return 0;
  }
  printf("FAILURES PRESENT\n");
  return 1;
}
