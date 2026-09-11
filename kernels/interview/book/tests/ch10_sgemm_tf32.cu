// book/tests/ch10_sgemm_tf32.cu — ch10 最小测试：TF32 WMMA + cp.async 双缓冲
// 测试逻辑抽取自 notes-v2.cu test_sgemm(L1289) 的 TF32 段，参考改为 CPU fp64
// （BOOK_PLAN §6，无 cuBLAS 依赖）。
// 覆盖：
//   case A: f32x4_tf32x4_kernel 舍入核（GPU 结果 vs host RNE/RNA 两种模式逐位比对）
//   case B: sgemm_tf32 128x128x128（grid 1x1，单 block 全流水路径）
//   case C: sgemm_tf32 256x256x64 （grid 2x2，多 block + K 不整除 stage 数）
//   bench : M=N=K=1024 TFLOPS
// 约束：M/N 为 128 的倍数、K 为 8 的倍数（sgemm.cuh L225 对齐假设）；
//       正确性规模 <= 512（BOOK_TEST_MAX_N）。
#include "../../sgemm.cuh"
#include "common_test.h"
#include <cstring>
#include <string>
#include <vector>

// host 端 float -> TF32 舍入（保留高 19bit：1 sign + 8 exp + 10 mantissa）
// mode 0 = round-to-nearest-even；mode 1 = round-to-nearest-ties-away（cvt.rna）
static float to_tf32_host(float f, int mode) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t lsb = (x >> 13) & 1u;
  uint32_t rnd = mode == 0 ? (0x0FFFu + lsb) : 0x1000u;
  uint32_t y = (x + rnd) & 0xFFFFE000u;
  float r;
  std::memcpy(&r, &y, 4);
  return r;
}

static uint32_t f32_bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return b;
}

// GPU 原地 FP32 -> TF32 转换（notes-v2 test_sgemm 同型调用）
static void gpu_convert_tf32(float *d, int n) {
  dim3 block(256);
  dim3 grid((n + 256 * 4 - 1) / (256 * 4));
  f32x4_tf32x4_kernel<<<grid, block>>>(d, d, n);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
}

// CPU fp64 参考：输入为 TF32 舍入后的 A/B（float 承载，低 13bit 为 0）
static void sgemm_ref_fp64(const float *A, const float *B, double *C, int M,
                           int N, int K) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double s = 0.0;
      for (int k = 0; k < K; ++k)
        s += double(A[i * K + k]) * double(B[k * N + j]);
      C[i * N + j] = s;
    }
}

// case A: 转换 kernel 舍入模式判定（GPU 命中 RNE 与 RNA 中的哪一种）
static void run_convert_case() {
  const int n = 4096;
  std::vector<float> h(n);
  book_fill_rand(h.data(), n);
  // 补充若干跨进位边界的值（* 1.0001 产生尾数低位扰动）
  for (int i = 0; i < n; i += 7) h[i] = h[i] * 1.0001f + 1e-7f;

  float *d;
  BOOK_CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
  BOOK_CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float),
                             cudaMemcpyHostToDevice));
  gpu_convert_tf32(d, n);
  std::vector<float> out(n);
  BOOK_CUDA_CHECK(cudaMemcpy(out.data(), d, n * sizeof(float),
                             cudaMemcpyDeviceToHost));
  BOOK_CUDA_CHECK(cudaFree(d));

  int eq_rne = 0, eq_rna = 0;
  for (int i = 0; i < n; ++i) {
    uint32_t g = f32_bits(out[i]);
    eq_rne += (g == f32_bits(to_tf32_host(h[i], 0)));
    eq_rna += (g == f32_bits(to_tf32_host(h[i], 1)));
  }
  const char *mode = eq_rne == n ? "rne" : (eq_rna == n ? "rna" : "other");
  bool pass = eq_rne == n || eq_rna == n;
  printf("%s f32x4_tf32x4_kernel: gpu==%s (rne %d/%d, rna %d/%d)\n",
         pass ? "PASS" : "FAIL", mode, eq_rne, n, eq_rna, n);
  if (!pass) g_failures++;
}

// case B/C: sgemm_tf32 正确性（TF32 舍入后输入 vs CPU fp64 参考）
static void run_gemm_case(const char *tag, int M, int N, int K) {
  const size_t sa = size_t(M) * K, sb = size_t(K) * N, sc = size_t(M) * N;
  std::vector<float> ha(sa), hb(sb), hc(sc);
  std::vector<double> ref(sc);
  book_fill_rand(ha.data(), int(sa), 0xC0DE);
  book_fill_rand(hb.data(), int(sb), 0x1234);

  float *d_a, *d_b, *d_c;
  BOOK_CUDA_CHECK(cudaMalloc(&d_a, sa * sizeof(float)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_b, sb * sizeof(float)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_c, sc * sizeof(float)));
  BOOK_CUDA_CHECK(cudaMemcpy(d_a, ha.data(), sa * sizeof(float),
                             cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_b, hb.data(), sb * sizeof(float),
                             cudaMemcpyHostToDevice));

  // 原地 TF32 转换（notes-v2 流程：转换破坏 fp32 精度，参考同用转换后数据）
  gpu_convert_tf32(d_a, int(sa));
  gpu_convert_tf32(d_b, int(sb));
  BOOK_CUDA_CHECK(cudaMemcpy(ha.data(), d_a, sa * sizeof(float),
                             cudaMemcpyDeviceToHost));
  BOOK_CUDA_CHECK(cudaMemcpy(hb.data(), d_b, sb * sizeof(float),
                             cudaMemcpyDeviceToHost));

  sgemm_ref_fp64(ha.data(), hb.data(), ref.data(), M, N, K);

  printf("case %s\n", tag);
  dim3 grid((N + 127) / 128, (M + 127) / 128);
  // 与 notes-v2 test_sgemm 一致：20736 B dynamic smem（含 bank padding）
  constexpr int tf32_smem_bytes =
      2 * 128 * 12 * sizeof(float) + 2 * 8 * 132 * sizeof(float);
  sgemm_tf32<<<grid, dim3(256), tf32_smem_bytes>>>(d_a, d_b, d_c, M, N, K);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(cudaMemcpy(hc.data(), d_c, sc * sizeof(float),
                             cudaMemcpyDeviceToHost));
  std::string name = std::string("sgemm_tf32 ") + std::to_string(M) + "x" +
                     std::to_string(N) + "x" + std::to_string(K);
  book_check(hc.data(), ref.data(), int(sc), TOL_TF32, name.c_str());

  BOOK_CUDA_CHECK(cudaFree(d_a));
  BOOK_CUDA_CHECK(cudaFree(d_b));
  BOOK_CUDA_CHECK(cudaFree(d_c));
}

static int bench() {
  const int M = 1024, N = 1024, K = 1024, warmup = 3, iters = 10;
  cudaDeviceProp prop;
  BOOK_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  printf("shape: M=N=K=%d, warmup=%d, iters=%d (avg)\n", M, warmup, iters);
  const size_t sa = size_t(M) * K, sb = size_t(K) * N, sc = size_t(M) * N;
  std::vector<float> ha(sa), hb(sb);
  book_fill_rand(ha.data(), int(sa));
  book_fill_rand(hb.data(), int(sb));
  float *d_a, *d_b, *d_c;
  BOOK_CUDA_CHECK(cudaMalloc(&d_a, sa * sizeof(float)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_b, sb * sizeof(float)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_c, sc * sizeof(float)));
  BOOK_CUDA_CHECK(cudaMemcpy(d_a, ha.data(), sa * sizeof(float),
                             cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_b, hb.data(), sb * sizeof(float),
                             cudaMemcpyHostToDevice));
  constexpr int tf32_smem_bytes =
      2 * 128 * 12 * sizeof(float) + 2 * 8 * 132 * sizeof(float);
  dim3 grid((N + 127) / 128, (M + 127) / 128);
  cudaEvent_t beg, end;
  BOOK_CUDA_CHECK(cudaEventCreate(&beg));
  BOOK_CUDA_CHECK(cudaEventCreate(&end));
  for (int w = 0; w < warmup; ++w)
    sgemm_tf32<<<grid, dim3(256), tf32_smem_bytes>>>(d_a, d_b, d_c, M, N, K);
  BOOK_CUDA_CHECK(cudaEventRecord(beg));
  for (int i = 0; i < iters; ++i)
    sgemm_tf32<<<grid, dim3(256), tf32_smem_bytes>>>(d_a, d_b, d_c, M, N, K);
  BOOK_CUDA_CHECK(cudaEventRecord(end));
  BOOK_CUDA_CHECK(cudaEventSynchronize(end));
  float ms = 0.0f;
  BOOK_CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
  ms /= iters;
  double tflops = 2.0 * M * N * K / (double(ms) * 1e-3) / 1e12;
  printf("sgemm_tf32: %.4f ms/iter, %.2f TFLOPS (TF32)\n", ms, tflops);
  BOOK_CUDA_CHECK(cudaEventDestroy(beg));
  BOOK_CUDA_CHECK(cudaEventDestroy(end));
  BOOK_CUDA_CHECK(cudaFree(d_a));
  BOOK_CUDA_CHECK(cudaFree(d_b));
  BOOK_CUDA_CHECK(cudaFree(d_c));
  return 0;
}

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "bench") return bench();
  if (!book_require_sm(80, "ch10")) return 0;  // TF32 WMMA 需 sm_80+

  printf("case A: f32->tf32 convert kernel (n=4096)\n");
  run_convert_case();
  run_gemm_case("B: M=128 N=128 K=128", 128, 128, 128);
  run_gemm_case("C: M=256 N=256 K=64", 256, 256, 64);

  printf(g_failures == 0 ? "ALL OK\n" : "FAILURES PRESENT\n");
  return g_failures == 0 ? 0 : 1;
}
