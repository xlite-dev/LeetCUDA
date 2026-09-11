// book/tests/ch03_elementwise.cu — ch03 向量化访存与原子操作最小测试 + bench（RFC-C3）
// 覆盖 base.cuh kernel: relu / relu_vec4 / elementwise_add / elementwise_add_vec4 /
// histogram
// 正确性: CPU 参考（relu/eadd 逐元素精确；histogram int 计数）+ TOL_F32ACC
// bench: cudaEvent 计时（warmup 3 + 计时 10 次取均值）→ .tmp/book-bench/ch03/bench.txt
#include "../../base.cuh"
#include "common_test.h"
#include <sys/stat.h>

static FILE* bench_open() {
  mkdir(".tmp", 0755);
  mkdir(".tmp/book-bench", 0755);
  mkdir(".tmp/book-bench/ch03", 0755);
  FILE* f = fopen(".tmp/book-bench/ch03/bench.txt", "w");
  if (f == nullptr) printf("WARN: cannot write bench.txt, stdout only\n");
  return f;
}

template <typename Fn>
static float bench_ms(Fn&& launch) {
  cudaEvent_t beg, end;
  BOOK_CUDA_CHECK(cudaEventCreate(&beg));
  BOOK_CUDA_CHECK(cudaEventCreate(&end));
  for (int i = 0; i < 3; ++i) launch();
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  float total_ms = 0.0f;
  for (int i = 0; i < 10; ++i) {
    BOOK_CUDA_CHECK(cudaEventRecord(beg));
    launch();
    BOOK_CUDA_CHECK(cudaEventRecord(end));
    BOOK_CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0.0f;
    BOOK_CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    total_ms += ms;
  }
  BOOK_CUDA_CHECK(cudaEventDestroy(beg));
  BOOK_CUDA_CHECK(cudaEventDestroy(end));
  return total_ms / 10.0f;
}

static void bench_report(FILE* f, const char* name, float ms, double bytes) {
  double gbs = bytes / ((double)ms * 1e-3) / 1e9;
  printf("BENCH %-22s avg=%10.5f ms  eff_bw=%9.2f GB/s\n", name, ms, gbs);
  if (f) {
    fprintf(f, "BENCH %-22s avg=%10.5f ms  eff_bw=%9.2f GB/s\n", name, ms, gbs);
    fflush(f);
  }
}

int main() {
  // ---- 正确性: N=512（4 的倍数，满足 vec4 主路径对齐要求）----
  const int N = 512;
  const int BINS = 16;
  float h_x[N], h_a[N], h_b[N], out[N];
  book_fill_rand(h_x, N, 0xA5A5);
  book_fill_rand(h_a, N, 0x5A5A);
  book_fill_rand(h_b, N, 0x11AA);

  double ref_relu[N], ref_eadd[N];
  for (int i = 0; i < N; ++i) {
    ref_relu[i] = h_x[i] > 0.0f ? h_x[i] : 0.0f;
    ref_eadd[i] = (double)h_a[i] + (double)h_b[i];
  }
  int h_idx[N];
  int ref_hist[BINS] = {0};
  srand(0xBEEF);
  for (int i = 0; i < N; ++i) {
    h_idx[i] = rand() % BINS;
    ref_hist[h_idx[i]]++;
  }

  float *d_x, *d_a, *d_b, *d_y;
  int *d_idx, *d_hist;
  BOOK_CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_idx, N * sizeof(int)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_hist, BINS * sizeof(int)));
  BOOK_CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(
      cudaMemcpy(d_idx, h_idx, N * sizeof(int), cudaMemcpyHostToDevice));

  // relu
  relu<<<(N + 255) / 256, 256>>>(d_x, d_y, N);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(cudaMemcpy(out, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
  book_check(out, ref_relu, N, TOL_F32ACC, "relu(N=512)");

  // relu_vec4
  relu_vec4<<<(N + 255) / 256, 64>>>(d_x, d_y, N);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(cudaMemcpy(out, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
  book_check(out, ref_relu, N, TOL_F32ACC, "relu_vec4(N=512)");

  // elementwise_add
  elementwise_add<<<(N + 255) / 256, 256>>>(d_a, d_b, d_y, N);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(cudaMemcpy(out, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
  book_check(out, ref_eadd, N, TOL_F32ACC, "elementwise_add(N=512)");

  // elementwise_add_vec4（主路径 float4 + 标量尾部）
  elementwise_add_vec4<<<(N + 255) / 256, 64>>>(d_a, d_b, d_y, N);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(cudaMemcpy(out, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
  book_check(out, ref_eadd, N, TOL_F32ACC, "elementwise_add_vec4(N=512)");

  // histogram: y[a[i]]++（原子计数 vs CPU 计数，int 精确一致）
  BOOK_CUDA_CHECK(cudaMemset(d_hist, 0, BINS * sizeof(int)));
  histogram<<<(N + 255) / 256, 256>>>(d_idx, d_hist, N);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  int h_hist[BINS];
  BOOK_CUDA_CHECK(
      cudaMemcpy(h_hist, d_hist, BINS * sizeof(int), cudaMemcpyDeviceToHost));
  double ref_hist_d[BINS];
  float out_hist_d[BINS];
  for (int i = 0; i < BINS; ++i) {
    ref_hist_d[i] = (double)ref_hist[i];
    out_hist_d[i] = (float)h_hist[i];
  }
  book_check(out_hist_d, ref_hist_d, BINS, TOL_F32ACC,
             "histogram(BINS=16,N=512)");

  // ---- bench: N=4M（口径：relu 读+写 8B/元素；eadd 两读一写 12B/元素；
  // histogram 按读 4B/元素计，原子 RMW 开销另计）----
  FILE* fb = bench_open();
  const int NB = 1 << 22;
  const int BINS_B = 256;
  float* hb = (float*)malloc(NB * sizeof(float));
  int* hib = (int*)malloc(NB * sizeof(int));
  float *bd_x, *bd_a, *bd_b, *bd_y;
  int *bd_idx, *bd_hist;
  BOOK_CUDA_CHECK(cudaMalloc(&bd_x, NB * sizeof(float)));
  BOOK_CUDA_CHECK(cudaMalloc(&bd_a, NB * sizeof(float)));
  BOOK_CUDA_CHECK(cudaMalloc(&bd_b, NB * sizeof(float)));
  BOOK_CUDA_CHECK(cudaMalloc(&bd_y, NB * sizeof(float)));
  BOOK_CUDA_CHECK(cudaMalloc(&bd_idx, NB * sizeof(int)));
  BOOK_CUDA_CHECK(cudaMalloc(&bd_hist, BINS_B * sizeof(int)));
  book_fill_rand(hb, NB, 0x1234);
  BOOK_CUDA_CHECK(cudaMemcpy(bd_x, hb, NB * sizeof(float), cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(bd_a, hb, NB * sizeof(float), cudaMemcpyHostToDevice));
  srand(0x4321);
  for (int i = 0; i < NB; ++i) hib[i] = rand() % BINS_B;
  BOOK_CUDA_CHECK(
      cudaMemcpy(bd_idx, hib, NB * sizeof(int), cudaMemcpyHostToDevice));
  free(hb);
  free(hib);

  float ms = bench_ms([&] {
    relu<<<(NB + 255) / 256, 256>>>(bd_x, bd_y, NB);
  });
  bench_report(fb, "relu", ms, 2.0 * NB * 4.0);
  ms = bench_ms([&] {
    relu_vec4<<<(NB + 255) / 256, 64>>>(bd_x, bd_y, NB);
  });
  bench_report(fb, "relu_vec4", ms, 2.0 * NB * 4.0);
  ms = bench_ms([&] {
    elementwise_add<<<(NB + 255) / 256, 256>>>(bd_a, bd_b, bd_y, NB);
  });
  bench_report(fb, "elementwise_add", ms, 3.0 * NB * 4.0);
  ms = bench_ms([&] {
    elementwise_add_vec4<<<(NB + 255) / 256, 64>>>(bd_a, bd_b, bd_y, NB);
  });
  bench_report(fb, "elementwise_add_vec4", ms, 3.0 * NB * 4.0);
  ms = bench_ms([&] {
    BOOK_CUDA_CHECK(cudaMemset(bd_hist, 0, BINS_B * sizeof(int)));
    histogram<<<(NB + 255) / 256, 256>>>(bd_idx, bd_hist, NB);
  });
  bench_report(fb, "histogram(256bins)", ms, 1.0 * NB * 4.0);
  if (fb) fclose(fb);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  cudaFree(d_x); cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
  cudaFree(d_idx); cudaFree(d_hist);
  cudaFree(bd_x); cudaFree(bd_a); cudaFree(bd_b); cudaFree(bd_y);
  cudaFree(bd_idx); cudaFree(bd_hist);
  return g_failures;
}
