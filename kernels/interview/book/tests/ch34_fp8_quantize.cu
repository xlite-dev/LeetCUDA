// book/tests/ch34_fp8_quantize.cu — ch34 最小测试：FP8 GEMM 量化前处理 kernel
// 覆盖：
//   case A: quantize_a_perrow  (M=256, K=256)   — scale 精确 + round-trip 界
//   case B: quantize_a_perblock(M=300, K=256)   — 尾块(M%128=44) + 零行 amax=0
//   case C: quantize_bt<true>  per-col (K=256, N=136) — 转置量化 + N 尾
//   case D: quantize_bt<false> per-block(K=144, N=136) — K 尾 chunk + block scale
// 判定：
//   1) scale vs CPU fp64 (amax/448)，绝对容差 TOL_F32ACC
//   2) 反量化 round-trip：|dq - x| <= 2^-4 * (|x| + δ)（e4m3 半 ulp 相对界）
// 参考：CPU fp64 逐元素；无 cuBLAS 依赖。
// 编译：sm_120a（量化 kernel 为纯 CUDA，任意 arch 可跑；入口按教学路径验证）
#define NOTES_V2_ENABLE_CUTE 1
#define NOTES_V2_ENABLE_TMA_MMA_WS 1
#include "../../fp8_gemm.cuh"
#include "common_test.h"
#include <vector>

using fp8_gemm::Fp8GemmWorkspace;

static float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }

// 反量化 round-trip 界检查：返回最差元素的超界比例（<=1 全过）
static double roundtrip_check(const std::vector<cutlass::float_e4m3_t> &q,
                              const std::vector<__nv_bfloat16> &x, double s,
                              int n) {
  double worst = 0.0;
  int bad = 0;
  for (int i = 0; i < n; ++i) {
    double dq = (float)q[i] * s;
    double xv = bf2f(x[i]);
    double bound = 0.0625 * (std::fabs(xv) + s);  // 2^-4 * (|x| + δ)
    double viol = std::fabs(dq - xv) / bound;
    if (viol > worst) worst = viol;
    if (viol > 1.0) bad++;
  }
  return bad == 0 ? worst : worst + 1.0;
}

// case A/B: A 量化（per-row / per-block）
static void test_a(int M, int K, bool per_row, const char *tag) {
  std::vector<__nv_bfloat16> x((size_t)M * K);
  srand(42);
  for (auto &v : x) v = __float2bfloat16(((float)rand() / RAND_MAX) * 2 - 1);
  for (int k = 0; k < K; ++k) x[k] = __float2bfloat16(0.0f);  // 第 0 行全零

  std::vector<double> s_ref(per_row ? M : (M + 127) / 128);
  for (int i = 0; i < (int)s_ref.size(); ++i) {
    double amax = 0;
    int r0 = per_row ? i : i * 128;
    int r1 = per_row ? i + 1 : std::min((i + 1) * 128, M);
    for (int r = r0; r < r1; ++r)
      for (int k = 0; k < K; ++k)
        amax = std::max(amax, (double)std::fabs(bf2f(x[(size_t)r * K + k])));
    s_ref[i] = amax / 448.0;
  }

  __nv_bfloat16 *d_x;
  cutlass::float_e4m3_t *d_q;
  float *d_s;
  BOOK_CUDA_CHECK(cudaMalloc(&d_x, x.size() * 2));
  BOOK_CUDA_CHECK(cudaMalloc(&d_q, x.size()));
  BOOK_CUDA_CHECK(cudaMalloc(&d_s, s_ref.size() * 4));
  BOOK_CUDA_CHECK(cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
  if (per_row)
    fp8_gemm::quantize_a_perrow_kernel<<<(M + 7) / 8, 256>>>(d_x, d_q, d_s, M, K);
  else
    fp8_gemm::quantize_a_perblock_kernel<<<(int)s_ref.size(), 256>>>(d_x, d_q,
                                                                     d_s, M, K);
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> h_s(s_ref.size());
  std::vector<cutlass::float_e4m3_t> h_q(x.size());
  BOOK_CUDA_CHECK(cudaMemcpy(h_s.data(), d_s, s_ref.size() * 4, cudaMemcpyDeviceToHost));
  BOOK_CUDA_CHECK(cudaMemcpy(h_q.data(), d_q, x.size(), cudaMemcpyDeviceToHost));

  char name[96];
  snprintf(name, sizeof(name), "ch34 A %s scale (M=%d K=%d)", tag, M, K);
  book_check(h_s.data(), s_ref.data(), (int)s_ref.size(), TOL_F32ACC, name);
  // round-trip: per-row 逐行查 δ; per-block 逐块查 δ
  double worst = 0;
  for (int r = 0; r < M; ++r) {
    double s = per_row ? h_s[r] : h_s[r / 128];
    double w = roundtrip_check(
        std::vector<cutlass::float_e4m3_t>(h_q.begin() + (size_t)r * K,
                                           h_q.begin() + (size_t)(r + 1) * K),
        std::vector<__nv_bfloat16>(x.begin() + (size_t)r * K,
                                   x.begin() + (size_t)(r + 1) * K),
        s, K);
    worst = std::max(worst, w);
  }
  bool pass = worst <= 1.0;
  printf("%s ch34 A %s roundtrip (M=%d K=%d): worst=%.3f of bound\n",
         pass ? "PASS" : "FAIL", tag, M, K, worst);
  if (!pass) g_failures++;
  cudaFree(d_x); cudaFree(d_q); cudaFree(d_s);
}

// case C/D: B 转置量化（per-col / per-block）
static void test_bt(int K, int N, bool per_col, const char *tag) {
  std::vector<__nv_bfloat16> b((size_t)K * N);
  srand(42);
  for (auto &v : b) v = __float2bfloat16(((float)rand() / RAND_MAX) * 2 - 1);

  std::vector<double> s_ref(per_col ? N : (N + 127) / 128);
  for (int i = 0; i < (int)s_ref.size(); ++i) {
    double amax = 0;
    int n0 = per_col ? i : i * 128;
    int n1 = per_col ? i + 1 : std::min((i + 1) * 128, N);
    for (int k = 0; k < K; ++k)
      for (int n = n0; n < n1; ++n)
        amax = std::max(amax, (double)std::fabs(bf2f(b[(size_t)k * N + n])));
    s_ref[i] = amax / 448.0;
  }

  __nv_bfloat16 *d_b;
  cutlass::float_e4m3_t *d_bt;
  float *d_s;
  BOOK_CUDA_CHECK(cudaMalloc(&d_b, b.size() * 2));
  BOOK_CUDA_CHECK(cudaMalloc(&d_bt, b.size()));
  BOOK_CUDA_CHECK(cudaMalloc(&d_s, s_ref.size() * 4));
  BOOK_CUDA_CHECK(cudaMemcpy(d_b, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
  if (per_col)
    fp8_gemm::quantize_bt_kernel<true><<<(N + 127) / 128, 256>>>(d_b, d_bt, d_s, K, N);
  else
    fp8_gemm::quantize_bt_kernel<false><<<(N + 127) / 128, 256>>>(d_b, d_bt, d_s, K, N);
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> h_s(s_ref.size());
  std::vector<cutlass::float_e4m3_t> h_bt(b.size());
  BOOK_CUDA_CHECK(cudaMemcpy(h_s.data(), d_s, s_ref.size() * 4, cudaMemcpyDeviceToHost));
  BOOK_CUDA_CHECK(cudaMemcpy(h_bt.data(), d_bt, b.size(), cudaMemcpyDeviceToHost));

  char name[96];
  snprintf(name, sizeof(name), "ch34 B^T %s scale (K=%d N=%d)", tag, K, N);
  book_check(h_s.data(), s_ref.data(), (int)s_ref.size(), TOL_F32ACC, name);
  // round-trip: B8T[n][k]*sb ≈ B[k][n]
  double worst = 0;
  int bad = 0;
  for (int n = 0; n < N; ++n) {
    double s = per_col ? h_s[n] : h_s[n / 128];
    for (int k = 0; k < K; ++k) {
      double dq = (float)h_bt[(size_t)n * K + k] * s;
      double x = bf2f(b[(size_t)k * N + n]);
      double bound = 0.0625 * (std::fabs(x) + s);
      double viol = std::fabs(dq - x) / bound;
      if (viol > worst) worst = viol;
      if (viol > 1.0) bad++;
    }
  }
  bool pass = bad == 0;
  printf("%s ch34 B^T %s roundtrip (K=%d N=%d): worst=%.3f of bound\n",
         pass ? "PASS" : "FAIL", tag, K, N, worst);
  if (!pass) g_failures++;
  cudaFree(d_b); cudaFree(d_bt); cudaFree(d_s);
}

int main() {
  // 量化 kernel 本身是纯 CUDA(sm_80+)，但按教学路径走 sm_120 验证链
  if (!book_require_sm(120, "ch34")) return 0;
  test_a(256, 256, true, "per-row");
  test_a(300, 256, false, "per-block");
  test_bt(256, 136, true, "per-col");
  test_bt(144, 136, false, "per-block");
  printf(g_failures == 0 ? "ALL OK\n" : "FAILURES PRESENT\n");
  return g_failures == 0 ? 0 : 1;
}
