// book/tests/ch24_hgemm_cute.cu — ch24 最小测试：CuTe HGEMM 正确性与性能（hgemm_mma_stages_tn_cute）
// 覆盖：
//   A 正确性（F16 累加，BM=128 BN=256 BK=32）：M=128 N=256 K=64，对照 CPU fp64 参考，容差 TOL_F16ACC；
//   B 正确性（F32 累加，BM=128 BN=128 BK=32）：M=128 N=128 K=64，对照 CPU fp64 参考，容差 TOL_F32ACC；
//   C 吞吐（可选 bench，argv[1]=="bench"）：两组配置各 20 次取均值，报告 ms/iter 与 TFLOPS。
// TN 约定：C[M,N] = A[M,K] x B^T[N,K]，其中 B^T 以 row-major [N,K] 传入（hgemm.cuh L1216 的第三个实参）。
// 形状约束（源码强制，无尾部 predication）：M % BM == 0、N % BN == 0、K % BK == 0；
// BN 由 kAccF32 决定（F16Acc 用 256，F32Acc 用 128）。
#define NOTES_V2_ENABLE_CUTE 1
#include "../../hgemm.cuh"
#include "common_test.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

using T = half;

// CPU fp64 参考：C[m,n] = sum_k A[m,k] * B_t[n,k]（A: [M,K] row-major，B_t: [N,K] row-major）
static void cpu_ref_gemm(const std::vector<float>& a, const std::vector<float>& b_t,
                         std::vector<double>& c, int M, int N, int K) {
  c.assign((size_t)M * N, 0.0);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) acc += double(a[(size_t)m * K + k]) * double(b_t[(size_t)n * K + k]);
      c[(size_t)m * N + n] = acc;
    }
}

// 运行一组配置：host 随机 -> 设备 -> CuTe kernel -> 回传，与 CPU fp64 参考比对
static void run_case(const char* name, int M, int N, int K, bool acc_f32, double tol) {
  std::vector<float> h_a((size_t)M * K), h_b_t((size_t)N * K);
  book_fill_rand(h_a.data(), (int)h_a.size());
  book_fill_rand(h_b_t.data(), (int)h_b_t.size(), 0x5A5A);
  // 输入缩到 1/4：fp16 输出的舍入误差 ~ |C| * 2^-11，缩到 |C| <= 2 量级后
  // max_abs_err 才由「累加精度」而非「输出量化」主导（1e-3 容差的前提）。
  for (float& v : h_a) v *= 0.25f;
  for (float& v : h_b_t) v *= 0.25f;

  std::vector<half> hd_a(h_a.size()), hd_b_t(h_b_t.size());
  for (size_t i = 0; i < h_a.size(); ++i) hd_a[i] = __float2half(h_a[i]);
  for (size_t i = 0; i < h_b_t.size(); ++i) hd_b_t[i] = __float2half(h_b_t[i]);

  std::vector<double> ref;
  cpu_ref_gemm(h_a, h_b_t, ref, M, N, K);

  T *d_a = nullptr, *d_b_t = nullptr, *d_c = nullptr;
  BOOK_CUDA_CHECK(cudaMalloc(&d_a, hd_a.size() * sizeof(T)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_b_t, hd_b_t.size() * sizeof(T)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_c, (size_t)M * N * sizeof(T)));
  BOOK_CUDA_CHECK(cudaMemcpy(d_a, hd_a.data(), hd_a.size() * sizeof(T), cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_b_t, hd_b_t.data(), hd_b_t.size() * sizeof(T), cudaMemcpyHostToDevice));

  if (acc_f32) {
    launch_hgemm_mma_stages_tn_cute<T, 2, 0, true>(d_a, d_b_t, d_c, M, N, K);
  } else {
    launch_hgemm_mma_stages_tn_cute<T, 2, 0, false>(d_a, d_b_t, d_c, M, N, K);
  }
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<half> hd_c((size_t)M * N);
  BOOK_CUDA_CHECK(cudaMemcpy(hd_c.data(), d_c, hd_c.size() * sizeof(T), cudaMemcpyDeviceToHost));
  std::vector<float> out(hd_c.size());
  for (size_t i = 0; i < hd_c.size(); ++i) out[i] = __half2float(hd_c[i]);

  book_check(out.data(), ref.data(), (int)out.size(), tol, name);

  cudaFree(d_a);
  cudaFree(d_b_t);
  cudaFree(d_c);
}

static void bench_one(const char* name, int M, int N, int K, bool acc_f32, int iters = 20) {
  std::vector<float> h_a((size_t)M * K, 0.5f), h_b_t((size_t)N * K, 0.5f);
  std::vector<half> hd_a(h_a.size()), hd_b_t(h_b_t.size());
  for (size_t i = 0; i < h_a.size(); ++i) hd_a[i] = __float2half(h_a[i]);
  for (size_t i = 0; i < h_b_t.size(); ++i) hd_b_t[i] = __float2half(h_b_t[i]);
  T *d_a = nullptr, *d_b_t = nullptr, *d_c = nullptr;
  BOOK_CUDA_CHECK(cudaMalloc(&d_a, hd_a.size() * sizeof(T)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_b_t, hd_b_t.size() * sizeof(T)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_c, (size_t)M * N * sizeof(T)));
  BOOK_CUDA_CHECK(cudaMemcpy(d_a, hd_a.data(), hd_a.size() * sizeof(T), cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_b_t, hd_b_t.data(), hd_b_t.size() * sizeof(T), cudaMemcpyHostToDevice));
  cudaEvent_t beg, end;
  BOOK_CUDA_CHECK(cudaEventCreate(&beg));
  BOOK_CUDA_CHECK(cudaEventCreate(&end));
  auto launch = [&] {
    if (acc_f32) {
      launch_hgemm_mma_stages_tn_cute<T, 2, 0, true>(d_a, d_b_t, d_c, M, N, K);
    } else {
      launch_hgemm_mma_stages_tn_cute<T, 2, 0, false>(d_a, d_b_t, d_c, M, N, K);
    }
  };
  for (int i = 0; i < 5; ++i) launch();
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(cudaEventRecord(beg));
  for (int i = 0; i < iters; ++i) launch();
  BOOK_CUDA_CHECK(cudaEventRecord(end));
  BOOK_CUDA_CHECK(cudaEventSynchronize(end));
  float ms = 0.0f;
  BOOK_CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
  ms /= iters;
  double flops = 2.0 * double(M) * N * K;
  printf("bench %s: %.4f ms/iter, %.1f TFLOPS\n", name, ms, flops / (double(ms) * 1e-3) / 1e12);
  cudaEventDestroy(beg);
  cudaEventDestroy(end);
  cudaFree(d_a);
  cudaFree(d_b_t);
  cudaFree(d_c);
}

int main(int argc, char** argv) {
  if (!book_require_sm(80, "ch24")) return 0;  // mma m16n8k16 + cp.async 需 sm_80+

  if (argc > 1 && std::string(argv[1]) == "bench") {
    bench_one("F16Acc 128x256x64", 128, 256, 64, false);
    bench_one("F32Acc 128x128x64", 128, 128, 64, true);
    return 0;
  }

  run_case("HGEMM CuTe F16Acc 128x256x64", 128, 256, 64, false, TOL_F16ACC);
  run_case("HGEMM CuTe F32Acc 128x128x64", 128, 128, 64, true, TOL_F32ACC);

  printf(g_failures == 0 ? "ALL OK\n" : "FAILURES PRESENT\n");
  return g_failures == 0 ? 0 : 1;
}
