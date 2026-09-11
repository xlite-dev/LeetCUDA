// book/tests/ch08_sgemv.cu — ch08 最小测试：SGEMV 三种划分（k32/k128/k16）
// 测试逻辑抽取自 notes-v2.cu test_sgemv（L1190），正确性对照改为 CPU fp64 参考
// （BOOK_PLAN §6，无 cuBLAS 依赖）。
// 边界约定：源码 kernel 无 K 维守卫——k32 需 K%32==0；k128 需 K%128==0
//   （float4 的 16B 对齐随之成立）；k16 面向 K=16 专用。k32/k128 的 M 由
//   warp 一致的 if (m < M) 守卫；k16 的行守卫在半 warp 粒度分叉，M%8!=0 时
//   掩码 0xffffffff 会命中未执行 shuffle 的 lane（见正文 8.6 勘误二），故
//   测试固定 M%8==0。
// bench：./.build/ch08_sgemv bench（复测数据由章节任务落 .tmp/book-bench/ch08/）。
#include "../../sgemv.cuh"
#include "common_test.h"
#include <string>
#include <vector>

// CPU fp64 参考：y[m] = sum_k a[m,k] * x[k]
static void sgemv_ref_fp64(const float* a, const float* x, double* y, int M,
                           int K) {
  for (int m = 0; m < M; ++m) {
    double sum = 0.0;
    for (int k = 0; k < K; ++k)
      sum += double(a[m * K + k]) * double(x[k]);
    y[m] = sum;
  }
}

// k32/k128 共用启动形状：block(32,4)，grid(ceil(M/4))
static void run_k32_k128(const char* tag, int M, int K) {
  std::vector<float> h_a(size_t(M) * K), h_x(K), h_y(M);
  std::vector<double> ref(M);
  book_fill_rand(h_a.data(), M * K);
  book_fill_rand(h_x.data(), K);
  sgemv_ref_fp64(h_a.data(), h_x.data(), ref.data(), M, K);

  float *d_a, *d_x, *d_y;
  const size_t bytes_a = size_t(M) * K * sizeof(float);
  BOOK_CUDA_CHECK(cudaMalloc(&d_a, bytes_a));
  BOOK_CUDA_CHECK(cudaMalloc(&d_x, size_t(K) * sizeof(float)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_y, size_t(M) * sizeof(float)));
  BOOK_CUDA_CHECK(
      cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), size_t(K) * sizeof(float),
                             cudaMemcpyHostToDevice));

  dim3 block(32, 4);
  dim3 grid((M + 3) / 4);
  printf("case %s\n", tag);

  BOOK_CUDA_CHECK(cudaMemset(d_y, 0, size_t(M) * sizeof(float)));
  sgemv_k32<<<grid, block>>>(d_a, d_x, d_y, M, K);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, size_t(M) * sizeof(float),
                             cudaMemcpyDeviceToHost));
  book_check(h_y.data(), ref.data(), M, TOL_F32ACC, "sgemv-k32");

  BOOK_CUDA_CHECK(cudaMemset(d_y, 0, size_t(M) * sizeof(float)));
  sgemv_k128<<<grid, block>>>(d_a, d_x, d_y, M, K);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, size_t(M) * sizeof(float),
                             cudaMemcpyDeviceToHost));
  book_check(h_y.data(), ref.data(), M, TOL_F32ACC, "sgemv-k128");

  BOOK_CUDA_CHECK(cudaFree(d_a));
  BOOK_CUDA_CHECK(cudaFree(d_x));
  BOOK_CUDA_CHECK(cudaFree(d_y));
}

// k16<2>：block(32,4)，grid(ceil(M/8))，K=16，一个 warp 同时算 2 行
static void run_k16(const char* tag, int M) {
  const int K = 16;
  std::vector<float> h_a(size_t(M) * K), h_x(K), h_y(M);
  std::vector<double> ref(M);
  book_fill_rand(h_a.data(), M * K);
  book_fill_rand(h_x.data(), K);
  sgemv_ref_fp64(h_a.data(), h_x.data(), ref.data(), M, K);

  float *d_a, *d_x, *d_y;
  const size_t bytes_a = size_t(M) * K * sizeof(float);
  BOOK_CUDA_CHECK(cudaMalloc(&d_a, bytes_a));
  BOOK_CUDA_CHECK(cudaMalloc(&d_x, size_t(K) * sizeof(float)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_y, size_t(M) * sizeof(float)));
  BOOK_CUDA_CHECK(
      cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), size_t(K) * sizeof(float),
                             cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemset(d_y, 0, size_t(M) * sizeof(float)));

  printf("case %s\n", tag);
  dim3 block(32, 4);
  dim3 grid((M + 7) / 8);
  sgemv_k16<2><<<grid, block>>>(d_a, d_x, d_y, M, K);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, size_t(M) * sizeof(float),
                             cudaMemcpyDeviceToHost));
  book_check(h_y.data(), ref.data(), M, TOL_F32ACC, "sgemv-k16<2>");

  BOOK_CUDA_CHECK(cudaFree(d_a));
  BOOK_CUDA_CHECK(cudaFree(d_x));
  BOOK_CUDA_CHECK(cudaFree(d_y));
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
  printf("GPU: %s (sm_%d%d), SMs=%d, clock=%.2f GHz, threads/SM=%d\n",
         prop.name, prop.major, prop.minor, prop.multiProcessorCount,
         clock_khz * 1e-6, prop.maxThreadsPerMultiProcessor);
  printf("est. fp32 peak = %.1f TFLOPS (128 FMA/SM/cycle)\n", peak_tflops);

  struct Item {
    const char* name;
    int kind, M, K, iters;
  };
  const Item items[] = {
      {"k32 ", 0, 8192, 4096, 50},
      {"k128", 1, 8192, 4096, 50},
      {"k16 ", 2, 1048576, 16, 100},
  };
  printf("%-6s %18s %10s %12s\n", "impl", "shape", "ms/iter", "GB/s(eff)");
  for (const auto& it : items) {
    const size_t bytes_a = size_t(it.M) * it.K * sizeof(float);
    float *d_a, *d_x, *d_y;
    BOOK_CUDA_CHECK(cudaMalloc(&d_a, bytes_a));
    BOOK_CUDA_CHECK(cudaMalloc(&d_x, size_t(it.K) * sizeof(float)));
    BOOK_CUDA_CHECK(cudaMalloc(&d_y, size_t(it.M) * sizeof(float)));
    // 0x3f 填充 -> float 0x3f3f3f3f ~ 0.748，避免 denormal
    BOOK_CUDA_CHECK(cudaMemset(d_a, 0x3f, bytes_a));
    BOOK_CUDA_CHECK(cudaMemset(d_x, 0x3f, size_t(it.K) * sizeof(float)));
    dim3 block(32, 4);
    auto launch = [&]() {
      if (it.kind == 0)
        sgemv_k32<<<dim3((it.M + 3) / 4), block>>>(d_a, d_x, d_y, it.M, it.K);
      else if (it.kind == 1)
        sgemv_k128<<<dim3((it.M + 3) / 4), block>>>(d_a, d_x, d_y, it.M,
                                                    it.K);
      else
        sgemv_k16<2><<<dim3((it.M + 7) / 8), block>>>(d_a, d_x, d_y, it.M,
                                                      it.K);
    };
    for (int w = 0; w < 5; ++w) launch();
    BOOK_CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t beg, end;
    BOOK_CUDA_CHECK(cudaEventCreate(&beg));
    BOOK_CUDA_CHECK(cudaEventCreate(&end));
    BOOK_CUDA_CHECK(cudaEventRecord(beg));
    for (int i = 0; i < it.iters; ++i) launch();
    BOOK_CUDA_CHECK(cudaEventRecord(end));
    BOOK_CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0.0f;
    BOOK_CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    ms /= it.iters;
    // 有效流量：A + x + y 各读/写一次
    const double bytes = double(bytes_a) + double(it.K + it.M) * 4.0;
    char shape[32];
    snprintf(shape, sizeof(shape), "M=%d K=%d", it.M, it.K);
    printf("%-6s %18s %10.4f %12.1f\n", it.name, shape, ms,
           bytes / (double(ms) * 1e6));
    BOOK_CUDA_CHECK(cudaEventDestroy(beg));
    BOOK_CUDA_CHECK(cudaEventDestroy(end));
    BOOK_CUDA_CHECK(cudaFree(d_a));
    BOOK_CUDA_CHECK(cudaFree(d_x));
    BOOK_CUDA_CHECK(cudaFree(d_y));
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "bench") return bench();
  // Case A：K%128==0：k32 与 k128 同形状对比
  run_k32_k128("A: M=64 K=512 (K%128==0)", 64, 512);
  // Case B：M 非 4 的倍数：warp 一致的 if (m < M) 行守卫路径
  run_k32_k128("B: M=7 K=128 (M guard)", 7, 128);
  // Case C：K=16 双行分段归约（M%8==0 规避 8.6 节勘误二的 UB 路径）
  run_k16("C: M=40 K=16 (2 rows/warp)", 40);
  printf(g_failures == 0 ? "ALL OK\n" : "FAILURES PRESENT\n");
  return g_failures == 0 ? 0 : 1;
}
