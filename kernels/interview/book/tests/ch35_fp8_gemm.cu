// book/tests/ch35_fp8_gemm.cu — ch35 最小测试：BF16 in -> 动态量化 FP8 GEMM
// (在线反量化) -> BF16 out 全链路 API，vs CPU fp64 参考。
// 覆盖：
//   case A-D: 四种 scale 组合（per-row/per-block x per-col/per-block）
//             M=256 N=256 K=256（全对齐）
//   case E/F: WS 变体（per-row x per-col / per-block x per-block）
//   case G:   尾部 shape M=300 N=136 K=144（M%128≠0, N%8 且非 128, K 尾）
//   case H-M: 权重 B 离线量化（B 只量化一次，之后每次只量化 A）：
//             rc nonws / rc ws / rb nonws / bc nonws / bb nonws / 尾部 shape，
//             并对同 mode 全链路做逐 bit 比对
// 容差：相对 Frobenius 误差 <= 0.08（E4M3 量化噪声统计上界 ~5%，实测 3.6%，
//       见 ch34 34.6 节；逐元素 max err 无意义——近零参考值任意放大）
// 约束：K%16==0, N%8==0（API 自检）；M 任意。
// 编译：-arch sm_120a（TMA/STSM 需 sm_90+，SM89 fp8 atom 需 sm_89+）
#define NOTES_V2_ENABLE_CUTE 1
#define NOTES_V2_ENABLE_TMA_MMA_WS 1
#include "../../fp8_gemm.cuh"
#include "common_test.h"
#include <vector>

static float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }

// 相对 Frobenius 误差判定（量化噪声的统计口径，非逐元素）
static bool book_check_rel_fro(const std::vector<float> &out,
                               const std::vector<double> &ref, double tol,
                               const char *name) {
  double sum_e2 = 0, sum_r2 = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double e = out[i] - ref[i];
    sum_e2 += e * e;
    sum_r2 += ref[i] * ref[i];
  }
  double rel = std::sqrt(sum_e2 / sum_r2);
  bool pass = rel <= tol;
  printf("%s %s: rel_fro=%.4f (tol=%.2f)\n", pass ? "PASS" : "FAIL", name, rel,
         tol);
  if (!pass) g_failures++;
  return pass;
}

// 测试共享数据：主机侧输入 + CPU fp64 参考 + 设备侧缓冲
struct CaseData {
  int M, N, K;
  std::vector<__nv_bfloat16> h_a, h_b;
  std::vector<double> ref;
  __nv_bfloat16 *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

  CaseData(int m, int n, int k)
      : M(m), N(n), K(k), h_a((size_t)m * k), h_b((size_t)k * n),
        ref((size_t)m * n) {
    srand(42);
    for (auto &v : h_a) v = __float2bfloat16(((float)rand() / RAND_MAX) * 2 - 1);
    for (auto &v : h_b) v = __float2bfloat16(((float)rand() / RAND_MAX) * 2 - 1);
    for (int m_ = 0; m_ < M; ++m_)
      for (int n_ = 0; n_ < N; ++n_)
        for (int k_ = 0; k_ < K; ++k_)
          ref[(size_t)m_ * N + n_] +=
              (double)bf2f(h_a[(size_t)m_ * K + k_]) *
              bf2f(h_b[(size_t)k_ * N + n_]);
    BOOK_CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * 2));
    BOOK_CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * 2));
    BOOK_CUDA_CHECK(cudaMalloc(&d_c, (size_t)M * N * 2));
    BOOK_CUDA_CHECK(
        cudaMemcpy(d_a, h_a.data(), h_a.size() * 2, cudaMemcpyHostToDevice));
    BOOK_CUDA_CHECK(
        cudaMemcpy(d_b, h_b.data(), h_b.size() * 2, cudaMemcpyHostToDevice));
  }

  void check(const char *tag) {
    std::vector<__nv_bfloat16> h_c((size_t)M * N);
    BOOK_CUDA_CHECK(
        cudaMemcpy(h_c.data(), d_c, (size_t)M * N * 2, cudaMemcpyDeviceToHost));
    std::vector<float> out(h_c.size());
    for (size_t i = 0; i < h_c.size(); ++i) out[i] = bf2f(h_c[i]);
    char name[128];
    snprintf(name, sizeof(name), "fp8_gemm %s M=%d N=%d K=%d", tag, M, N, K);
    book_check_rel_fro(out, ref, 0.08, name);
  }

  ~CaseData() {
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
  }
};

static void run_case(int M, int N, int K, fp8_gemm::Fp8GemmScaleMode mode,
                     bool use_ws, const char *tag) {
  CaseData d(M, N, K);
  fp8_gemm::Fp8GemmWorkspace ws;
  ws.size = fp8_gemm::fp8_gemm_workspace_size(M, N, K);
  BOOK_CUDA_CHECK(cudaMalloc(&ws.buf, ws.size));
  fp8_gemm::fp8_gemm_bf16(d.d_a, d.d_b, d.d_c, M, N, K, mode, ws, 0, use_ws);
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  d.check(tag);
  cudaFree(ws.buf);
}

// 权重离线路径：B 量化一次（离线，不进计时），之后每次前向只量化 A。
// 除精度判定外还与同 mode 的全链路做逐位比对：δb 只由权重决定，量化时机
// 不影响结果，两条路的输出应逐 bit 相等。
static void run_case_b_offline(int M, int N, int K, bool per_block_a,
                               bool per_col_b, bool use_ws, const char *tag) {
  using fp8_gemm::Fp8GemmScaleMode;
  const Fp8GemmScaleMode mode =
      per_block_a ? (per_col_b ? Fp8GemmScaleMode::kPerBlockPerCol
                               : Fp8GemmScaleMode::kPerBlockPerBlock)
                  : (per_col_b ? Fp8GemmScaleMode::kPerRowPerCol
                               : Fp8GemmScaleMode::kPerRowPerBlock);
  CaseData d(M, N, K);
  cutlass::float_e4m3_t *b8t;
  float *sb;
  BOOK_CUDA_CHECK(cudaMalloc(&b8t, (size_t)N * K));
  BOOK_CUDA_CHECK(cudaMalloc(&sb, (size_t)N * sizeof(float)));
  fp8_gemm::fp8_gemm_quantize_b(d.d_b, b8t, sb, N, K, per_col_b, 0);
  fp8_gemm::Fp8GemmActivation act;
  act.size = fp8_gemm::fp8_gemm_activation_size(M, K);
  BOOK_CUDA_CHECK(cudaMalloc(&act.buf, act.size));
#define BOFF_LAUNCH(PRA, PCB, WS)                                              \
  fp8_gemm::fp8_gemm_bf16_b_offline<PRA, PCB, WS>(d.d_a, b8t, sb, d.d_c, M, N, \
                                                 K, act, 0)
  if (use_ws) {
    if (per_block_a && per_col_b) BOFF_LAUNCH(false, true, true);
    else if (per_block_a) BOFF_LAUNCH(false, false, true);
    else if (per_col_b) BOFF_LAUNCH(true, true, true);
    else BOFF_LAUNCH(true, false, true);
  } else {
    if (per_block_a && per_col_b) BOFF_LAUNCH(false, true, false);
    else if (per_block_a) BOFF_LAUNCH(false, false, false);
    else if (per_col_b) BOFF_LAUNCH(true, true, false);
    else BOFF_LAUNCH(true, false, false);
  }
#undef BOFF_LAUNCH
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  d.check(tag);

  __nv_bfloat16 *d_c_ref;
  BOOK_CUDA_CHECK(cudaMalloc(&d_c_ref, (size_t)M * N * 2));
  fp8_gemm::Fp8GemmWorkspace ws;
  ws.size = fp8_gemm::fp8_gemm_workspace_size(M, N, K);
  BOOK_CUDA_CHECK(cudaMalloc(&ws.buf, ws.size));
  fp8_gemm::fp8_gemm_bf16(d.d_a, d.d_b, d_c_ref, M, N, K, mode, ws, 0, use_ws);
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  {
    std::vector<__nv_bfloat16> c_off((size_t)M * N), c_full((size_t)M * N);
    BOOK_CUDA_CHECK(cudaMemcpy(c_off.data(), d.d_c, (size_t)M * N * 2,
                               cudaMemcpyDeviceToHost));
    BOOK_CUDA_CHECK(cudaMemcpy(c_full.data(), d_c_ref, (size_t)M * N * 2,
                               cudaMemcpyDeviceToHost));
    // 逐 bit 比对（bf16 的 2 字节位模式，绕开浮点比较的 -0/+0 语义）
    const unsigned short *o16 =
        reinterpret_cast<const unsigned short *>(c_off.data());
    const unsigned short *f16 =
        reinterpret_cast<const unsigned short *>(c_full.data());
    size_t diff = 0;
    for (size_t i = 0; i < c_off.size(); ++i)
      if (o16[i] != f16[i]) ++diff;
    char name[160];
    snprintf(name, sizeof(name), "fp8_gemm %s vs full-chain", tag);
    bool pass = (diff == 0);
    printf("%s %s: bit-exact=%s (diff=%zu)\n", pass ? "PASS" : "FAIL", name,
           pass ? "YES" : "NO", diff);
    if (!pass) g_failures++;
  }
  cudaFree(d_c_ref);
  cudaFree(ws.buf);
  cudaFree(b8t);
  cudaFree(sb);
  cudaFree(act.buf);
}

int main() {
  // TMA + STSM + SM89 fp8 MMA: sm_90 教学验证路径（sm_120 本机）
  if (!book_require_sm(90, "ch35")) return 0;
  using fp8_gemm::Fp8GemmScaleMode;
  run_case(256, 256, 256, Fp8GemmScaleMode::kPerRowPerCol, false, "rc nonws");
  run_case(256, 256, 256, Fp8GemmScaleMode::kPerRowPerBlock, false, "rb nonws");
  run_case(256, 256, 256, Fp8GemmScaleMode::kPerBlockPerCol, false, "bc nonws");
  run_case(256, 256, 256, Fp8GemmScaleMode::kPerBlockPerBlock, false, "bb nonws");
  run_case(256, 256, 256, Fp8GemmScaleMode::kPerRowPerCol, true, "rc ws");
  run_case(256, 256, 256, Fp8GemmScaleMode::kPerBlockPerBlock, true, "bb ws");
  run_case(300, 136, 144, Fp8GemmScaleMode::kPerRowPerCol, false, "tail rc nonws");
  run_case(300, 136, 144, Fp8GemmScaleMode::kPerBlockPerBlock, true, "tail bb ws");
  run_case_b_offline(256, 256, 256, false, true, false, "boff rc nonws");
  run_case_b_offline(256, 256, 256, false, true, true, "boff rc ws");
  run_case_b_offline(256, 256, 256, false, false, false, "boff rb nonws");
  run_case_b_offline(256, 256, 256, true, true, false, "boff bc nonws");
  run_case_b_offline(256, 256, 256, true, false, false, "boff bb nonws");
  run_case_b_offline(300, 136, 144, true, false, false, "boff tail bb nonws");
  printf(g_failures == 0 ? "ALL OK\n" : "FAILURES PRESENT\n");
  return g_failures == 0 ? 0 : 1;
}
