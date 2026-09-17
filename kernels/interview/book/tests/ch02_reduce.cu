// book/tests/ch02_reduce.cu — ch02 归约原语最小测试 + bench（RFC-C2）
// 覆盖 base.cuh kernel: block_reduce_all / dot / dot_vec4
// 正确性: CPU fp64 参考 + TOL_F32ACC（跨 block atomicAdd 的浮点加法非结合，
// 结果有 run-to-run 抖动，用容差判而非 bitwise）
// bench: cudaEvent 计时（warmup 3 + 计时 10 次取均值）→ .tmp/book-bench/ch02/bench.txt
#include "../../base.cuh"
#include "common_test.h"
#include <sys/stat.h>

static double cpu_sum_ref(const float* a, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += (double)a[i];
  return s;
}

static double cpu_dot_ref(const float* a, const float* b, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += (double)a[i] * (double)b[i];
  return s;
}

// mkdir -p .tmp/book-bench/ch02/ 并打开 bench.txt（CWD = book/tests/）
static FILE* bench_open() {
  mkdir(".tmp", 0755);
  mkdir(".tmp/book-bench", 0755);
  mkdir(".tmp/book-bench/ch02", 0755);
  FILE* f = fopen(".tmp/book-bench/ch02/bench.txt", "w");
  if (f == nullptr) printf("WARN: cannot write bench.txt, stdout only\n");
  return f;
}

// cudaEvent 计时: warmup 3 + 计时 10 次取均值（ms）；launch 内含每轮必要的复位
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
  printf("BENCH %-20s avg=%10.5f ms  eff_bw=%9.2f GB/s\n", name, ms, gbs);
  if (f) {
    fprintf(f, "BENCH %-20s avg=%10.5f ms  eff_bw=%9.2f GB/s\n", name, ms, gbs);
    fflush(f);
  }
}

int main() {
  // ---- 正确性: N=512（4 的倍数，满足 vec4 主路径对齐要求）----
  const int N = 512;
  float h_a[N], h_b[N], out[1];
  book_fill_rand(h_a, N, 0xA5A5);
  book_fill_rand(h_b, N, 0x5A5A);
  double ref_sum = cpu_sum_ref(h_a, N);
  double ref_dot = cpu_dot_ref(h_a, h_b, N);

  float *d_a, *d_b, *d_y;
  BOOK_CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_y, sizeof(float)));
  BOOK_CUDA_CHECK(
      cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(
      cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

  // block_reduce_all: y = sum(a[0..N-1])，grid 按 256 线程切
  BOOK_CUDA_CHECK(cudaMemset(d_y, 0, sizeof(float)));
  block_reduce_all<256><<<(N + 255) / 256, 256>>>(d_a, d_y, N);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(cudaMemcpy(out, d_y, sizeof(float), cudaMemcpyDeviceToHost));
  book_check(out, &ref_sum, 1, TOL_F32ACC, "block_reduce_all(N=512)");

  // dot: y = sum(a[i]*b[i])
  BOOK_CUDA_CHECK(cudaMemset(d_y, 0, sizeof(float)));
  dot<256><<<(N + 255) / 256, 256>>>(d_a, d_b, d_y, N);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(cudaMemcpy(out, d_y, sizeof(float), cudaMemcpyDeviceToHost));
  book_check(out, &ref_dot, 1, TOL_F32ACC, "dot(N=512)");

  // dot_vec4: 每线程 4 元素，grid 仍按 256 元素/block 划分
  BOOK_CUDA_CHECK(cudaMemset(d_y, 0, sizeof(float)));
  dot_vec4<64><<<(N + 255) / 256, 64>>>(d_a, d_b, d_y, N);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(cudaMemcpy(out, d_y, sizeof(float), cudaMemcpyDeviceToHost));
  book_check(out, &ref_dot, 1, TOL_F32ACC, "dot_vec4(N=512)");

  // ---- bench: N=4M 大缓冲（有效带宽口径：reduce 读 4B/元素，dot 读 8B/元素；
  // 每次 launch 前的 4B memset 计入时间，量级可忽略）----
  FILE* fb = bench_open();
  const int NB = 1 << 22;
  float* hb = (float*)malloc(NB * sizeof(float));
  float *bd_a, *bd_b, *bd_y;
  BOOK_CUDA_CHECK(cudaMalloc(&bd_a, NB * sizeof(float)));
  BOOK_CUDA_CHECK(cudaMalloc(&bd_b, NB * sizeof(float)));
  BOOK_CUDA_CHECK(cudaMalloc(&bd_y, sizeof(float)));
  book_fill_rand(hb, NB, 0x1234);
  BOOK_CUDA_CHECK(
      cudaMemcpy(bd_a, hb, NB * sizeof(float), cudaMemcpyHostToDevice));
  book_fill_rand(hb, NB, 0x4321);
  BOOK_CUDA_CHECK(
      cudaMemcpy(bd_b, hb, NB * sizeof(float), cudaMemcpyHostToDevice));
  free(hb);

  float ms = bench_ms([&] {
    BOOK_CUDA_CHECK(cudaMemset(bd_y, 0, sizeof(float)));
    block_reduce_all<256><<<(NB + 255) / 256, 256>>>(bd_a, bd_y, NB);
  });
  bench_report(fb, "block_reduce_all", ms, (double)NB * 4.0);
  ms = bench_ms([&] {
    BOOK_CUDA_CHECK(cudaMemset(bd_y, 0, sizeof(float)));
    dot<256><<<(NB + 255) / 256, 256>>>(bd_a, bd_b, bd_y, NB);
  });
  bench_report(fb, "dot", ms, 2.0 * NB * 4.0);
  ms = bench_ms([&] {
    BOOK_CUDA_CHECK(cudaMemset(bd_y, 0, sizeof(float)));
    dot_vec4<64><<<(NB + 255) / 256, 64>>>(bd_a, bd_b, bd_y, NB);
  });
  bench_report(fb, "dot_vec4", ms, 2.0 * NB * 4.0);
  if (fb) fclose(fb);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
  cudaFree(bd_a); cudaFree(bd_b); cudaFree(bd_y);
  return g_failures;
}
