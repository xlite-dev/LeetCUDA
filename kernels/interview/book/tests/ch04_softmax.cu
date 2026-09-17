// book/tests/ch04_softmax.cu — ch04 最小测试：Softmax 三级递进（naive/safe/online）
// 测试逻辑抽取自 notes-v2.cu test_softmax(L847)，正确性对照改为 CPU fp64 参考
// （BOOK_PLAN §6，无 cuBLAS/cuDNN 依赖）。
// 覆盖：三版 kernel 正确性（vs CPU fp64）、三版两两一致性、大数值场景下
//       naive 溢出（预期非有限，仅打印 INFO）vs safe/online 稳定。
// 约束：per-token kernel 一个 block 处理一个 token，blockDim 必须等于模板
//       kNumThreads 且 N <= blockDim；规模 <= 512（BOOK_TEST_MAX_N）。
#include "../../base.cuh"
#include "common_test.h"
#include <string>
#include <vector>

// CPU fp64 参考：safe softmax（先减 max，double 累加），单 token
static void softmax_ref_fp64(const float* x, double* y, int n) {
  double m = -DBL_MAX;
  for (int i = 0; i < n; ++i)
    if (double(x[i]) > m) m = double(x[i]);
  double sum = 0.0;
  for (int i = 0; i < n; ++i) sum += std::exp(double(x[i]) - m);
  for (int i = 0; i < n; ++i) y[i] = std::exp(double(x[i]) - m) / sum;
}

// 启动三版 per-token kernel；impl: 0=naive 1=safe 2=online
// blockDim 必须等于模板参数 kNumThreads（block 级二级归约的 shared 数组大小），
// 且 per-token 长度必须等于 blockDim（一个 token 恰好填满一个 block）。
// n_arg 语义 = 总元素数 S * H（原始 softmax.cu 入口 `const int N = S * H;`，
// kernel 内 idx < N 是全局边界；若误传 per-token 长度且 S>1，blocks>=1 会被
// 守卫整体剪掉、输出不写回——见 ch04 勘误与考据）。
template <const int kNumThreads>
static void launch_softmax(int impl, float* d_x, float* d_y, int S, int n_arg) {
  dim3 grid(S), block(kNumThreads);
  if (impl == 0)
    softmax_per_token<kNumThreads><<<grid, block>>>(d_x, d_y, n_arg);
  else if (impl == 1)
    safe_softmax_per_token<kNumThreads><<<grid, block>>>(d_x, d_y, n_arg);
  else
    online_safe_softmax_per_token<kNumThreads><<<grid, block>>>(d_x, d_y, n_arg);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
}

// 两个 GPU 输出数组的一致性判定（打印与 book_check 同风格）
static void check_pair(const std::vector<float>& a, const std::vector<float>& b,
                       double tol, const char* name) {
  double max_diff = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double err = std::fabs(double(a[i]) - double(b[i]));
    if (err > max_diff) max_diff = err;
  }
  bool pass = max_diff <= tol;
  printf("%s %s: max_diff=%.3e (tol=%.1e)\n", pass ? "PASS" : "FAIL", name,
         max_diff, tol);
  if (!pass) g_failures++;
}

// 一个测试 case：S 个 token、每 token N 个元素、block=kNumThreads、
// 输入 x = rand([-1,1]) * scale + bias；check_naive=false 时只打印 naive 观察值
static void run_case(const char* tag, int S, int N, int kNumThreads,
                     float scale, float bias, bool check_naive) {
  const int total = S * N;
  const size_t bytes = size_t(total) * sizeof(float);
  std::vector<float> h_x(total), h_naive(total), h_safe(total), h_online(total);
  std::vector<double> ref(total);
  book_fill_rand(h_x.data(), total);
  for (int i = 0; i < total; ++i) h_x[i] = h_x[i] * scale + bias;
  for (int s = 0; s < S; ++s)
    softmax_ref_fp64(h_x.data() + s * N, ref.data() + s * N, N);

  float *d_x, *d_y;
  BOOK_CUDA_CHECK(cudaMalloc(&d_x, bytes));
  BOOK_CUDA_CHECK(cudaMalloc(&d_y, bytes));
  BOOK_CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  const char* names[3] = {"naive", "safe", "online"};
  std::vector<float>* hosts[3] = {&h_naive, &h_safe, &h_online};
  const int n_arg = S * N;  // 源码 N 语义：总元素数（N=S*H；S=1 时允许 N<blockDim）
  for (int impl = 0; impl < 3; ++impl) {
    if (kNumThreads == 256)
      launch_softmax<256>(impl, d_x, d_y, S, n_arg);
    else if (kNumThreads == 128)
      launch_softmax<128>(impl, d_x, d_y, S, n_arg);
    else
      launch_softmax<64>(impl, d_x, d_y, S, n_arg);
    BOOK_CUDA_CHECK(
        cudaMemcpy(hosts[impl]->data(), d_y, bytes, cudaMemcpyDeviceToHost));
  }

  printf("case %s\n", tag);
  if (check_naive)
    book_check(h_naive.data(), ref.data(), total, TOL_F32ACC, "softmax naive");
  else
    printf("INFO naive with large x: y[0]=%g (non-finite expected)\n",
           h_naive[0]);
  book_check(h_safe.data(), ref.data(), total, TOL_F32ACC, "softmax safe");
  book_check(h_online.data(), ref.data(), total, TOL_F32ACC, "softmax online");
  // 三版一致性：fp32 同一数学目标的实现间差异应远小于 vs fp64 参考的容差；
  // online 用 __expf/__fdividef 快速数学，放宽到 2e-5
  check_pair(h_naive, h_safe, 2e-5, "consistency naive vs safe");
  check_pair(h_safe, h_online, 2e-5, "consistency safe vs online");
  check_pair(h_naive, h_online, 2e-5, "consistency naive vs online");

  BOOK_CUDA_CHECK(cudaFree(d_x));
  BOOK_CUDA_CHECK(cudaFree(d_y));
}

static int bench() {
  const int S = 4096, N = 256, warmup = 3, iters = 10;
  cudaDeviceProp prop;
  BOOK_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  printf("shape: grid=%d tokens, block=%d (N=%d per token), "
         "warmup=%d, iters=%d (avg)\n",
         S, N, N, warmup, iters);
  const size_t bytes = size_t(S) * N * sizeof(float);
  float *d_x, *d_y;
  BOOK_CUDA_CHECK(cudaMalloc(&d_x, bytes));
  BOOK_CUDA_CHECK(cudaMalloc(&d_y, bytes));
  std::vector<float> h_x(S * N);
  book_fill_rand(h_x.data(), S * N);
  BOOK_CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
  cudaEvent_t beg, end;
  BOOK_CUDA_CHECK(cudaEventCreate(&beg));
  BOOK_CUDA_CHECK(cudaEventCreate(&end));
  const char* names[3] = {"naive", "safe", "online"};
  printf("%-8s %12s %16s\n", "impl", "ms/iter", "GB/s (8B/elem)");
  for (int impl = 0; impl < 3; ++impl) {
    for (int w = 0; w < warmup; ++w) launch_softmax<256>(impl, d_x, d_y, S, N);
    BOOK_CUDA_CHECK(cudaEventRecord(beg));
    for (int i = 0; i < iters; ++i) launch_softmax<256>(impl, d_x, d_y, S, N);
    BOOK_CUDA_CHECK(cudaEventRecord(end));
    BOOK_CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0.0f;
    BOOK_CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    ms /= iters;
    double gbps = double(S) * N * 8.0 / (double(ms) * 1e6);
    printf("%-8s %12.4f %16.1f\n", names[impl], ms, gbps);
  }
  BOOK_CUDA_CHECK(cudaEventDestroy(beg));
  BOOK_CUDA_CHECK(cudaEventDestroy(end));
  BOOK_CUDA_CHECK(cudaFree(d_x));
  BOOK_CUDA_CHECK(cudaFree(d_y));
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "bench") return bench();

  // Case A: 2 tokens x 256，x in [-5, 5]（notes-v2 test_softmax 口径）
  run_case("A: S=2 N=256 block=256 x in [-5,5]", 2, 256, 256, 5.0f, 0.0f, true);
  // Case B: 1 token x 100，block=128：idx>=N 的边界线程守卫路径
  run_case("B: S=1 N=100 block=128 x in [-5,5]", 1, 100, 128, 5.0f, 0.0f, true);
  // Case C: 1 token x 256，x in [80, 120]：naive 溢出演示 vs safe/online 稳定
  run_case("C: S=1 N=256 block=256 x in [80,120]", 1, 256, 256, 20.0f, 100.0f,
           false);

  printf(g_failures == 0 ? "ALL OK\n" : "FAILURES PRESENT\n");
  return g_failures == 0 ? 0 : 1;
}
