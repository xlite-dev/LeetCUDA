// book/tests/ch06_norm.cu — ch06 最小测试：RMSNorm / LayerNorm（标量与 Vec4）
// 测试逻辑抽取自 notes-v2.cu test_rms_norm(L920) / test_layer_norm(L978)，
// 正确性对照改为 CPU fp64 手算参考（BOOK_PLAN §6，无 cuBLAS/cuDNN 依赖）。
// 约束：kernel 模板 kNumThreads=128 要求 K=128（一行恰好装进一个 block）。
#include "../../base.cuh"
#include "common_test.h"
#include <string>
#include <vector>

static void rms_ref_fp64(const float* x, double* y, int n_rows, int K,
                         double g, double eps) {
  for (int n = 0; n < n_rows; ++n) {
    const float* row = x + size_t(n) * K;
    double sum_sq = 0.0;
    for (int k = 0; k < K; ++k) {
      double v = double(row[k]);
      sum_sq += v * v;
    }
    double inv_rms = 1.0 / std::sqrt(sum_sq / double(K) + eps);
    for (int k = 0; k < K; ++k)
      y[size_t(n) * K + k] = double(row[k]) * inv_rms * g;
  }
}

static void ln_ref_fp64(const float* x, double* y, int n_rows, int K,
                        double g, double b, double eps) {
  for (int n = 0; n < n_rows; ++n) {
    const float* row = x + size_t(n) * K;
    double sum = 0.0;
    for (int k = 0; k < K; ++k) sum += double(row[k]);
    double mean = sum / double(K);
    double sum_sq = 0.0;
    for (int k = 0; k < K; ++k) {
      double d = double(row[k]) - mean;
      sum_sq += d * d;
    }
    double inv_std = 1.0 / std::sqrt(sum_sq / double(K) + eps);
    for (int k = 0; k < K; ++k)
      y[size_t(n) * K + k] = (double(row[k]) - mean) * inv_std * g + b;
  }
}

static void test_norm(int N, int K) {
  const size_t num = size_t(N) * K;
  const size_t bytes = num * sizeof(float);
  const float g = 1.5f, b = 0.3f, eps = 1e-5f;

  std::vector<float> h_x(num);
  book_fill_rand(h_x.data(), int(num));
  std::vector<double> ref_rms(num), ref_ln(num);
  rms_ref_fp64(h_x.data(), ref_rms.data(), N, K, g, eps);
  ln_ref_fp64(h_x.data(), ref_ln.data(), N, K, g, b, eps);

  float *d_x, *d_y;
  BOOK_CUDA_CHECK(cudaMalloc(&d_x, bytes));
  BOOK_CUDA_CHECK(cudaMalloc(&d_y, bytes));
  BOOK_CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
  std::vector<float> h_y(num);

  // RMSNorm 标量：grid(N), block(128)，一行一个 block
  rms_norm<<<N, 128>>>(d_x, d_y, g, N, K);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
  book_check(h_y.data(), ref_rms.data(), int(num), TOL_F32ACC, "rms_norm");

  // RMSNorm Vec4：block(32)，每线程 4 元素，要求 K%4==0 且地址 16B 对齐
  rms_norm_vec4<<<N, 32>>>(d_x, d_y, g, N, K);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
  book_check(h_y.data(), ref_rms.data(), int(num), TOL_F32ACC, "rms_norm_vec4");

  // LayerNorm 标量（2-pass）/ Vec4
  layer_norm<<<N, 128>>>(d_x, d_y, g, b, N, K);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
  book_check(h_y.data(), ref_ln.data(), int(num), TOL_F32ACC, "layer_norm");

  layer_norm_vec4<<<N, 32>>>(d_x, d_y, g, b, N, K);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
  book_check(h_y.data(), ref_ln.data(), int(num), TOL_F32ACC, "layer_norm_vec4");

  cudaFree(d_x);
  cudaFree(d_y);
}

template <typename F>
static float bench_ms(F&& launch, int warmup, int rep) {
  cudaEvent_t beg, end;
  BOOK_CUDA_CHECK(cudaEventCreate(&beg));
  BOOK_CUDA_CHECK(cudaEventCreate(&end));
  for (int i = 0; i < warmup; ++i) launch();
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  float best = 1e30f;
  for (int r = 0; r < rep; ++r) {
    BOOK_CUDA_CHECK(cudaEventRecord(beg));
    launch();
    BOOK_CUDA_CHECK(cudaEventRecord(end));
    BOOK_CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0.0f;
    BOOK_CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    if (ms < best) best = ms;
  }
  cudaEventDestroy(beg);
  cudaEventDestroy(end);
  return best;
}

static int run_bench() {
  const int K = 128;
  const float g = 1.5f, b = 0.3f;
  // 8192/131072: 数据量 <= L2 常驻（L2 口径）；524288: 512MB 流量，HBM 口径
  const int Ns[] = {8192, 131072, 524288};
  printf("ch06 bench: K=%d, cudaEvent min-of-20, warmup 3 (fp32, read+write)\n", K);
  printf("| %-8s | %-22s | %-22s | %-22s | %-22s |\n", "N", "rms_norm", "rms_norm_vec4",
         "layer_norm", "layer_norm_vec4");
  printf("|          | %-22s |\n", "  (ms / effective GB/s)");
  for (int N : Ns) {
    const size_t num = size_t(N) * K;
    const size_t bytes = num * sizeof(float);
    std::vector<float> h_x(num);
    book_fill_rand(h_x.data(), int(num));
    float *d_x, *d_y;
    BOOK_CUDA_CHECK(cudaMalloc(&d_x, bytes));
    BOOK_CUDA_CHECK(cudaMalloc(&d_y, bytes));
    BOOK_CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    const double gb = double(bytes) * 2.0 / 1e9;
    float t0 = bench_ms([&] { rms_norm<<<N, 128>>>(d_x, d_y, g, N, K); }, 3, 20);
    float t1 = bench_ms([&] { rms_norm_vec4<<<N, 32>>>(d_x, d_y, g, N, K); }, 3, 20);
    float t2 = bench_ms([&] { layer_norm<<<N, 128>>>(d_x, d_y, g, b, N, K); }, 3, 20);
    float t3 = bench_ms([&] { layer_norm_vec4<<<N, 32>>>(d_x, d_y, g, b, N, K); }, 3, 20);
    printf("| %-8d | %7.4f ms %6.1f | %7.4f ms %6.1f | %7.4f ms %6.1f | %7.4f ms %6.1f |\n",
           N, t0, gb / (t0 * 1e-3), t1, gb / (t1 * 1e-3), t2, gb / (t2 * 1e-3), t3,
           gb / (t3 * 1e-3));
    cudaFree(d_x);
    cudaFree(d_y);
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--bench") return run_bench();
  const int N = 512, K = 128;
  printf("ch06 norm test: N=%d K=%d (CPU fp64 reference, tol=%.0e)\n", N, K, TOL_F32ACC);
  test_norm(N, K);
  if (g_failures == 0) {
    printf("ch06 ALL PASS\n");
    return 0;
  }
  printf("ch06 FAILED: %d case(s)\n", g_failures);
  return 1;
}
