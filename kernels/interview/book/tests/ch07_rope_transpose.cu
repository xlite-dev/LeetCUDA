// book/tests/ch07_rope_transpose.cu — ch07 最小测试：RoPE 与矩阵转置（PAD）
// 测试逻辑抽取自 notes-v2.cu test_rope(L1042) / test_mat_transpose(L1098, L1144)，
// 正确性对照改为 CPU fp64 参考：RoPE 用旋转矩阵直接计算，转置为纯数据搬运。
// 约束：mat_transpose_padded 的 tile/float4 写回要求 row%64==0 且 col%16==0。
#include "../../base.cuh"
#include "common_test.h"
#include <string>
#include <vector>

static void rope_ref_fp64(const float* x, double* y, int seq_len, int n_pair) {
  const int total_pairs = seq_len * n_pair;
  for (int idx = 0; idx < total_pairs; ++idx) {
    const int pos = idx / n_pair;  // token 在序列中的位置
    const int i = idx % n_pair;    // token 内的维度对索引
    const double theta = std::pow(10000.0, -2.0 * double(i) / double(2 * n_pair));
    const double angle = double(pos) * theta;
    const double c = std::cos(angle), s = std::sin(angle);
    const double x1 = double(x[idx * 2]), x2 = double(x[idx * 2 + 1]);
    y[idx * 2] = x1 * c - x2 * s;
    y[idx * 2 + 1] = x1 * s + x2 * c;
  }
}

static void transpose_ref_fp64(const float* x, double* y, int row, int col) {
  for (int i = 0; i < row; ++i)
    for (int j = 0; j < col; ++j)
      y[size_t(j) * row + i] = double(x[size_t(i) * col + j]);
}

static void test_rope(int seq_len, int n_pair) {
  const int total_pairs = seq_len * n_pair;
  const size_t num = size_t(total_pairs) * 2;
  const size_t bytes = num * sizeof(float);

  std::vector<float> h_x(num);
  book_fill_rand(h_x.data(), int(num));
  std::vector<double> ref(num);
  rope_ref_fp64(h_x.data(), ref.data(), seq_len, n_pair);

  float *d_x, *d_y;
  BOOK_CUDA_CHECK(cudaMalloc(&d_x, bytes));
  BOOK_CUDA_CHECK(cudaMalloc(&d_y, bytes));
  BOOK_CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  dim3 block(256);
  dim3 grid((total_pairs + 255) / 256);
  rope<<<grid, block>>>(d_x, d_y, seq_len, n_pair);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_y(num);
  BOOK_CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
  book_check(h_y.data(), ref.data(), int(num), TOL_F32ACC, "rope");

  cudaFree(d_x);
  cudaFree(d_y);
}

static void test_transpose(int row, int col) {
  const size_t num = size_t(row) * col;
  const size_t bytes = num * sizeof(float);

  std::vector<float> h_x(num);
  book_fill_rand(h_x.data(), int(num));
  std::vector<double> ref(num);
  transpose_ref_fp64(h_x.data(), ref.data(), row, col);

  float *d_x, *d_y;
  BOOK_CUDA_CHECK(cudaMalloc(&d_x, bytes));
  BOOK_CUDA_CHECK(cudaMalloc(&d_y, bytes));
  BOOK_CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  // naive：block(16,16)，读合并、写非合并
  dim3 block(16, 16);
  dim3 grid_naive((col + 15) / 16, (row + 15) / 16);
  mat_transpose<<<grid_naive, block>>>(d_x, d_y, row, col);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> h_naive(num);
  BOOK_CUDA_CHECK(cudaMemcpy(h_naive.data(), d_y, bytes, cudaMemcpyDeviceToHost));
  bool ok_naive = book_check(h_naive.data(), ref.data(), int(num), TOL_F32ACC,
                             "mat_transpose");

  // padded：smem tile[64][17]，PAD=1 消除列访问的 bank 对齐
  std::vector<float> h_zero(num, 0.0f);
  BOOK_CUDA_CHECK(cudaMemcpy(d_y, h_zero.data(), bytes, cudaMemcpyHostToDevice));
  dim3 grid_pad((col + 15) / 16, (row + 63) / 64);
  mat_transpose_padded<<<grid_pad, block>>>(d_x, d_y, row, col);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> h_pad(num);
  BOOK_CUDA_CHECK(cudaMemcpy(h_pad.data(), d_y, bytes, cudaMemcpyDeviceToHost));
  bool ok_pad = book_check(h_pad.data(), ref.data(), int(num), TOL_F32ACC,
                           "mat_transpose_padded");

  // 两个 kernel 均无算术运算，正确即互相 bitwise 一致（显式断言一次）
  if (ok_naive && ok_pad) {
    for (size_t i = 0; i < num; ++i) {
      if (h_naive[i] != h_pad[i]) {
        printf("FAIL transpose_consistency: mismatch at %zu\n", i);
        g_failures++;
        break;
      }
    }
  }

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
  // 512/1024/2048 规模数据量 <= L2 常驻（L2 口径带宽），8192 超出 L2（HBM 口径）
  const int Ms[] = {512, 1024, 2048, 8192};
  printf("ch07 bench: square transpose, cudaEvent min-of-20, warmup 3 (fp32)\n");
  printf("| %-6s | %-26s | %-26s | %-8s |\n", "M", "mat_transpose (naive)",
         "mat_transpose_padded", "speedup");
  for (int M : Ms) {
    const size_t bytes = size_t(M) * M * sizeof(float);
    float *d_x, *d_y;
    BOOK_CUDA_CHECK(cudaMalloc(&d_x, bytes));
    BOOK_CUDA_CHECK(cudaMalloc(&d_y, bytes));
    dim3 block(16, 16);
    dim3 g_naive((M + 15) / 16, (M + 15) / 16);
    dim3 g_pad((M + 15) / 16, (M + 63) / 64);
    const double gb = double(bytes) * 2.0 / 1e9;
    float t0 = bench_ms([&] { mat_transpose<<<g_naive, block>>>(d_x, d_y, M, M); }, 3, 20);
    float t1 = bench_ms([&] { mat_transpose_padded<<<g_pad, block>>>(d_x, d_y, M, M); }, 3, 20);
    printf("| %-6d | %7.4f ms %6.1f GB/s | %7.4f ms %6.1f GB/s | %5.2fx |\n",
           M, t0, gb / (t0 * 1e-3), t1, gb / (t1 * 1e-3), t0 / t1);
    cudaFree(d_x);
    cudaFree(d_y);
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--bench") return run_bench();
  printf("ch07 rope/transpose test (CPU fp64 reference, tol=%.0e)\n", TOL_F32ACC);
  test_rope(128, 64);
  test_transpose(512, 512);
  if (g_failures == 0) {
    printf("ch07 ALL PASS\n");
    return 0;
  }
  printf("ch07 FAILED: %d case(s)\n", g_failures);
  return 1;
}
