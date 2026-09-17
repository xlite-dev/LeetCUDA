// book/tests/ch09_sgemm.cu — ch09 最小测试：SGEMM Block Tile / Vec4+Thread Tile
// 测试逻辑抽取自 notes-v2.cu test_sgemm（L1289），正确性对照改为 CPU fp64 参考
// （BOOK_PLAN §6，无 cuBLAS 依赖）。
// 边界约定（sgemm.cuh L106 注释）：两 kernel 均无边界守卫——sgemm 需 M、N、K
//   均为 32 的倍数；sgemm_vec4 需 M、N 为 128 的倍数且 K 为 32 的倍数。
//   测试形状全部取 128 的倍数。
// bench：./.build/ch09_sgemm bench（复测数据由章节任务落 .tmp/book-bench/ch09/）。
#include "../../sgemm.cuh"
#include "common_test.h"
#include <string>
#include <vector>

// CPU fp64 参考：C = A x B（row-major）
static void sgemm_ref_fp64(const float* a, const float* b, double* c, int M,
                           int N, int K) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double sum = 0.0;
      for (int k = 0; k < K; ++k)
        sum += double(a[m * K + k]) * double(b[k * N + n]);
      c[m * N + n] = sum;
    }
}

static void run_case(const char* tag, int M, int N, int K) {
  std::vector<float> h_a(size_t(M) * K), h_b(size_t(K) * N), h_c(size_t(M) * N);
  std::vector<double> ref(size_t(M) * N);
  book_fill_rand(h_a.data(), M * K);
  book_fill_rand(h_b.data(), K * N);
  sgemm_ref_fp64(h_a.data(), h_b.data(), ref.data(), M, N, K);

  float *d_a, *d_b, *d_c;
  const size_t bytes_a = size_t(M) * K * sizeof(float);
  const size_t bytes_b = size_t(K) * N * sizeof(float);
  const size_t bytes_c = size_t(M) * N * sizeof(float);
  BOOK_CUDA_CHECK(cudaMalloc(&d_a, bytes_a));
  BOOK_CUDA_CHECK(cudaMalloc(&d_b, bytes_b));
  BOOK_CUDA_CHECK(cudaMalloc(&d_c, bytes_c));
  BOOK_CUDA_CHECK(
      cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(
      cudaMemcpy(d_b, h_b.data(), bytes_b, cudaMemcpyHostToDevice));

  dim3 block(32, 32);
  printf("case %s\n", tag);

  BOOK_CUDA_CHECK(cudaMemset(d_c, 0, bytes_c));
  sgemm<<<dim3((N + 31) / 32, (M + 31) / 32), block>>>(d_a, d_b, d_c, M, N, K);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(
      cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost));
  book_check(h_c.data(), ref.data(), M * N, TOL_F32ACC, "sgemm (32x32x32)");

  BOOK_CUDA_CHECK(cudaMemset(d_c, 0, bytes_c));
  sgemm_vec4<<<dim3((N + 127) / 128, (M + 127) / 128), block>>>(d_a, d_b, d_c,
                                                                M, N, K);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(
      cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost));
  book_check(h_c.data(), ref.data(), M * N, TOL_F32ACC,
             "sgemm-vec4 (128x128x32)");

  BOOK_CUDA_CHECK(cudaFree(d_a));
  BOOK_CUDA_CHECK(cudaFree(d_b));
  BOOK_CUDA_CHECK(cudaFree(d_c));
}

// 单 kernel 计时：返回 ms/iter（kind 0=sgemm, 1=sgemm_vec4）
static float time_kernel(int kind, float* d_a, float* d_b, float* d_c, int M,
                         int N, int K, int warmup, int iters) {
  dim3 block(32, 32);
  auto launch = [&]() {
    if (kind == 0)
      sgemm<<<dim3((N + 31) / 32, (M + 31) / 32), block>>>(d_a, d_b, d_c, M, N,
                                                           K);
    else
      sgemm_vec4<<<dim3((N + 127) / 128, (M + 127) / 128), block>>>(
          d_a, d_b, d_c, M, N, K);
  };
  for (int w = 0; w < warmup; ++w) launch();
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t beg, end;
  BOOK_CUDA_CHECK(cudaEventCreate(&beg));
  BOOK_CUDA_CHECK(cudaEventCreate(&end));
  BOOK_CUDA_CHECK(cudaEventRecord(beg));
  for (int i = 0; i < iters; ++i) launch();
  BOOK_CUDA_CHECK(cudaEventRecord(end));
  BOOK_CUDA_CHECK(cudaEventSynchronize(end));
  float ms = 0.0f;
  BOOK_CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
  ms /= iters;
  BOOK_CUDA_CHECK(cudaEventDestroy(beg));
  BOOK_CUDA_CHECK(cudaEventDestroy(end));
  return ms;
}

static int bench() {
  cudaDeviceProp prop;
  BOOK_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int clock_khz = 0;
  BOOK_CUDA_CHECK(
      cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
  // sm_120 每 SM 128 条 fp32 FMA 流水（256 FLOP/cycle），折算理论峰值
  const double peak_tflops =
      double(prop.multiProcessorCount) * 256 * double(clock_khz) * 1e3 * 1e-12;
  printf("GPU: %s (sm_%d%d), SMs=%d, clock=%.2f GHz, threads/SM=%d, "
         "smem/SM=%zu KB\n",
         prop.name, prop.major, prop.minor, prop.multiProcessorCount,
         clock_khz * 1e-6, prop.maxThreadsPerMultiProcessor,
         size_t(prop.sharedMemPerMultiprocessor) >> 10);
  printf("est. fp32 peak = %.1f TFLOPS (128 FMA/SM/cycle)\n\n",
         peak_tflops);

  const int sizes[] = {512, 1024, 2048};
  printf("%-6s %6s %8s %10s %10s %8s\n", "impl", "MNK", "blocks", "ms/iter",
         "TFLOPS", "%peak");
  for (int L : sizes) {
    const int M = L, N = L, K = L;
    float *d_a, *d_b, *d_c;
    BOOK_CUDA_CHECK(cudaMalloc(&d_a, size_t(M) * K * 4));
    BOOK_CUDA_CHECK(cudaMalloc(&d_b, size_t(K) * N * 4));
    BOOK_CUDA_CHECK(cudaMalloc(&d_c, size_t(M) * N * 4));
    // 0x3f 填充 -> float 0x3f3f3f3f ~ 0.748，避免 denormal
    BOOK_CUDA_CHECK(cudaMemset(d_a, 0x3f, size_t(M) * K * 4));
    BOOK_CUDA_CHECK(cudaMemset(d_b, 0x3f, size_t(K) * N * 4));
    const char* names[2] = {"naive", "vec4"};
    for (int kind = 0; kind < 2; ++kind) {
      const float ms = time_kernel(kind, d_a, d_b, d_c, M, N, K, 3, 10);
      const double tflops = 2.0 * M * N * K / (double(ms) * 1e-3);
      const int blocks = kind == 0 ? ((N + 31) / 32) * ((M + 31) / 32)
                                   : ((N + 127) / 128) * ((M + 127) / 128);
      printf("%-6s %6d %8d %10.3f %10.2f %8.1f\n", names[kind], L, blocks, ms,
             tflops * 1e-12, 100.0 * tflops * 1e-12 / peak_tflops);
    }
    BOOK_CUDA_CHECK(cudaFree(d_a));
    BOOK_CUDA_CHECK(cudaFree(d_b));
    BOOK_CUDA_CHECK(cudaFree(d_c));
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "bench") return bench();
  // Case A：128^3：naive 4x4=16 blocks vs vec4 1x1=1 block
  run_case("A: M=N=K=128 (16 vs 1 blocks)", 128, 128, 128);
  // Case B：256^3：naive 8x8=64 blocks vs vec4 2x2=4 blocks
  run_case("B: M=N=K=256 (64 vs 4 blocks)", 256, 256, 256);
  printf(g_failures == 0 ? "ALL OK\n" : "FAILURES PRESENT\n");
  return g_failures == 0 ? 0 : 1;
}
