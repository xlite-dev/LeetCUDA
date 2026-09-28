// book/tests/ch37_fp4_gemm.cu — ch37 最小测试：BF16 in -> 在线 NVFP4 两级量化
// GEMM（blockscaled MMA F32 累加 + epilogue 反量化）-> BF16 out 全链路 API，
// 与 CPU fp64 参考对比。
// 覆盖：
//   case A-D: 四种 scale 组合（单级 / level-1 在 A 行 / level-1 在 B 列 / 两级）
//             M=256 N=256 K=256（全对齐）
//   case E:   WS 变体（两级）
//   case F/G: 尾部 shape M=300 N=136 K=192（M%128≠0, N 仅 8 对齐, K 非 128 倍数）
//   case H-K: 权重 B 离线量化（B 只量化一次，之后每次只量化 A）：四种组合，
//             各自与同 mode 全链路做逐 bit 比对
//   case L:   tile 几何变体（BN=256/s5、BN=128/s8）走同一条 API 的精度一致性
// 容差：相对 Frobenius 误差 <= 0.20（E2M1 + 两级 SF 的实测刻度 0.145，
//       理论 ~0.14，见 ch36 误差模型节；逐元素 max err 无意义——近零参考值
//       任意放大）。
// 约束：K%64==0, N%8==0（API 自检）；M 任意。
// 编译：**必须** `-gencode arch=compute_120a,code=sm_120a`（build_tests.sh 的
//       写法）。不要用 `-arch=sm_120a` 简写：PTX 的虚拟目标会落到
//       `compute_120`，ptxas 以 `Feature 'cvt.e2m1x2.f32' not supported on
//       .target 'sm_120'` 拒绝整份 PTX；换算成可用的 gencode 形式即可，无需
//       sm_120f。目标 arch 不在位时 SKIP。
#define NOTES_V2_ENABLE_CUTE 1
#define NOTES_V2_ENABLE_TMA_MMA_WS 1
#define NOTES_V2_ENABLE_SM120_FP4 1
#include "../../fp4_gemm.cuh"
#include "common_test.h"
#include <vector>

static float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }

// 相对 Frobenius 误差判定（量化噪声的统计口径，非逐元素）
static void book_check_rel_fro(const std::vector<float> &out,
                               const std::vector<double> &ref, double tol,
                               const char *name) {
  double sum_e2 = 0, sum_r2 = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double e = out[i] - ref[i];
    sum_e2 += e * e;
    sum_r2 += ref[i] * ref[i];
  }
  const double rel = std::sqrt(sum_e2 / sum_r2);
  const bool pass = rel <= tol;
  printf("%s %s: rel_fro=%.4f (tol=%.2f)\n", pass ? "PASS" : "FAIL", name, rel, tol);
  if (!pass) ++g_failures;
}

// 共享数据：主机输入 + CPU fp64 参考 + 设备缓冲
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
          ref[(size_t)m_ * N + n_] += (double)bf2f(h_a[(size_t)m_ * K + k_]) *
                                      bf2f(h_b[(size_t)k_ * N + n_]);
    BOOK_CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * 2));
    BOOK_CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * 2));
    BOOK_CUDA_CHECK(cudaMalloc(&d_c, (size_t)M * N * 2));
    BOOK_CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * 2,
                               cudaMemcpyHostToDevice));
    BOOK_CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * 2,
                               cudaMemcpyHostToDevice));
  }

  void check(const char *tag) {
    std::vector<__nv_bfloat16> h_c((size_t)M * N);
    BOOK_CUDA_CHECK(
        cudaMemcpy(h_c.data(), d_c, (size_t)M * N * 2, cudaMemcpyDeviceToHost));
    std::vector<float> out(h_c.size());
    for (size_t i = 0; i < h_c.size(); ++i) out[i] = bf2f(h_c[i]);
    char name[128];
    snprintf(name, sizeof(name), "fp4_gemm %s M=%d N=%d K=%d", tag, M, N, K);
    book_check_rel_fro(out, ref, 0.20, name);
  }

  ~CaseData() {
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
  }
};

static void run_case(int M, int N, int K, fp4_gemm::Fp4GemmScaleMode mode,
                     bool use_ws, const char *tag) {
  CaseData d(M, N, K);
  fp4_gemm::Fp4GemmWorkspace ws;
  ws.size = fp4_gemm::fp4_gemm_workspace_size(M, N, K);
  BOOK_CUDA_CHECK(cudaMalloc(&ws.buf, ws.size));
  fp4_gemm::fp4_gemm_bf16(d.d_a, d.d_b, d.d_c, M, N, K, mode, ws, 0, use_ws);
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  d.check(tag);
  cudaFree(ws.buf);
}

// 权重离线路径：B（含 SFB）只量化一次（离线，不进计时窗口），之后每次前向只
// 在线量化 A。除精度判定外还与同 mode 全链路逐 bit 比对：level-1 的 sB[n] 只由
// 权重决定、level-2 的 SFB 也只由权重决定，量化时机不影响结果。
static void run_case_b_offline(int M, int N, int K, bool level1_a, bool level1_b,
                               bool use_ws, fp4_gemm::Fp4GemmScaleMode mode,
                               const char *tag) {
  CaseData d(M, N, K);
  cutlass::float_e2m1_t *b4t;
  cutlass::float_ue4m3_t *sfb;
  float *sb;
  BOOK_CUDA_CHECK(cudaMalloc(&b4t, (size_t)N * (K / 2)));
  BOOK_CUDA_CHECK(cudaMalloc(&sfb, fp4_gemm::sf_buffer_bytes(N, K)));
  BOOK_CUDA_CHECK(cudaMalloc(&sb, (size_t)N * sizeof(float)));
  fp4_gemm::fp4_gemm_quantize_b(d.d_b, b4t, sfb, sb, N, K, level1_b, 0);
  fp4_gemm::Fp4GemmActivation act;
  act.size = fp4_gemm::fp4_gemm_activation_size(M, K);
  BOOK_CUDA_CHECK(cudaMalloc(&act.buf, act.size));
#define BOFF_LAUNCH(L1A, L1B, WS)                                          \
  fp4_gemm::fp4_gemm_bf16_b_offline<L1A, L1B, WS>(d.d_a, b4t, sfb, sb,      \
                                                 d.d_c, M, N, K, act, 0)
  if (use_ws) {
    if (level1_a && level1_b) BOFF_LAUNCH(true, true, true);
    else if (level1_a) BOFF_LAUNCH(true, false, true);
    else if (level1_b) BOFF_LAUNCH(false, true, true);
    else BOFF_LAUNCH(false, false, true);
  } else {
    if (level1_a && level1_b) BOFF_LAUNCH(true, true, false);
    else if (level1_a) BOFF_LAUNCH(true, false, false);
    else if (level1_b) BOFF_LAUNCH(false, true, false);
    else BOFF_LAUNCH(false, false, false);
  }
#undef BOFF_LAUNCH
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  d.check(tag);

  __nv_bfloat16 *d_c_ref;
  BOOK_CUDA_CHECK(cudaMalloc(&d_c_ref, (size_t)M * N * 2));
  fp4_gemm::Fp4GemmWorkspace ws;
  ws.size = fp4_gemm::fp4_gemm_workspace_size(M, N, K);
  BOOK_CUDA_CHECK(cudaMalloc(&ws.buf, ws.size));
  fp4_gemm::fp4_gemm_bf16(d.d_a, d.d_b, d_c_ref, M, N, K, mode, ws, 0, use_ws);
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  {
    std::vector<__nv_bfloat16> c_off((size_t)M * N), c_full((size_t)M * N);
    BOOK_CUDA_CHECK(cudaMemcpy(c_off.data(), d.d_c, (size_t)M * N * 2,
                               cudaMemcpyDeviceToHost));
    BOOK_CUDA_CHECK(cudaMemcpy(c_full.data(), d_c_ref, (size_t)M * N * 2,
                               cudaMemcpyDeviceToHost));
    // 逐 bit 比对（bf16 的 2 字节位模式，绕开浮点比较的 -0/+0 语义）
    const unsigned short *o16 = reinterpret_cast<const unsigned short *>(c_off.data());
    const unsigned short *f16 = reinterpret_cast<const unsigned short *>(c_full.data());
    size_t diff = 0;
    for (size_t i = 0; i < c_off.size(); ++i)
      if (o16[i] != f16[i]) ++diff;
    char name[160];
    snprintf(name, sizeof(name), "fp4_gemm %s vs full-chain", tag);
    const bool pass = (diff == 0);
    printf("%s %s: bit-exact=%s (diff=%zu)\n", pass ? "PASS" : "FAIL", name,
           diff == 0 ? "YES" : "NO", diff);
    if (!pass) ++g_failures;
  }
  cudaFree(d_c_ref);
  cudaFree(ws.buf);
  cudaFree(b4t);
  cudaFree(sfb);
  cudaFree(sb);
  cudaFree(act.buf);
}

// tile 几何变体：同一条 API 走非默认 Traits（BN/流水深度），精度必须与默认档
// 一致（几何只改分块与流水，不改量化数学，故 rel_fro 应逐位同值）。
template <int kBN, int kStages>
static void run_case_tile(int M, int N, int K, const char *tag) {
  using Traits = fp4_gemm::Fp4GemmTraits<kBN, kStages>;
  CaseData d(M, N, K);
  // 量化一次（两级），然后只用指定 Traits 跑主 kernel
  const size_t asz = fp4_gemm::fp4_gemm_align256((size_t)M * (K / 2));
  const size_t bsz = fp4_gemm::fp4_gemm_align256((size_t)N * (K / 2));
  const size_t sfa_sz = fp4_gemm::fp4_gemm_align256(fp4_gemm::sf_buffer_bytes(M, K));
  const size_t sfb_sz = fp4_gemm::fp4_gemm_align256(fp4_gemm::sf_buffer_bytes(N, K));
  const size_t sa_sz = fp4_gemm::fp4_gemm_align256((size_t)M * sizeof(float));
  const size_t sb_sz = fp4_gemm::fp4_gemm_align256((size_t)N * sizeof(float));
  void *buf = nullptr;
  BOOK_CUDA_CHECK(cudaMalloc(&buf, asz + bsz + sfa_sz + sfb_sz + sa_sz + sb_sz));
  uint8_t *p = static_cast<uint8_t *>(buf);
  auto *a4 = reinterpret_cast<cutlass::float_e2m1_t *>(p);
  p += asz;
  auto *b4t = reinterpret_cast<cutlass::float_e2m1_t *>(p);
  p += bsz;
  auto *sfa = reinterpret_cast<cutlass::float_ue4m3_t *>(p);
  p += sfa_sz;
  auto *sfb = reinterpret_cast<cutlass::float_ue4m3_t *>(p);
  p += sfb_sz;
  float *sa = reinterpret_cast<float *>(p);
  p += sa_sz;
  float *sb = reinterpret_cast<float *>(p);
  fp4_gemm::row_amax_fp4_kernel<<<(M + 7) / 8, 256>>>(d.d_a, sa, M, K);
  const int mpad = (M + 127) / 128 * 128;
  fp4_gemm::quantize_a_fp4_kernel<true>
      <<<(mpad * (K / 16) + 255) / 256, 256>>>(d.d_a, a4, sfa, sa, M, K);
  fp4_gemm::fp4_gemm_quantize_b(d.d_b, b4t, sfb, sb, N, K, true, 0);
  fp4_gemm::fp4_gemm_tma_fwd<true, true, false, Traits>(
      a4, b4t, sfa, sfb, sa, sb, reinterpret_cast<cutlass::bfloat16_t *>(d.d_c), M,
      N, K, 0);
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  d.check(tag);
  cudaFree(buf);
}

int main() {
  if (!book_require_sm(120, "ch37")) return 0;
  using fp4_gemm::Fp4GemmScaleMode;
  printf("=== ch37 NVFP4 GEMM（在线两级量化 + blockscaled MMA）===\n");
  printf("（误差列 = rel_fro 相对 Frobenius 误差 vs CPU fp64；E2M1 只有 3 个"
         "数值位，量级见 ch36 误差模型节）\n");
  run_case(256, 256, 256, Fp4GemmScaleMode::kSingleLevel, false, "nonws 单级 level-2");
  run_case(256, 256, 256, Fp4GemmScaleMode::kLevel1A, false, "nonws level-2 x A 行");
  run_case(256, 256, 256, Fp4GemmScaleMode::kLevel1B, false, "nonws level-2 x B 列");
  run_case(256, 256, 256, Fp4GemmScaleMode::kTwoLevel, false, "nonws 两级");
  run_case(256, 256, 256, Fp4GemmScaleMode::kTwoLevel, true, "ws   两级");
  run_case(300, 264, 192, Fp4GemmScaleMode::kTwoLevel, false, "nonws 两级 尾 shape");
  run_case(300, 264, 192, Fp4GemmScaleMode::kTwoLevel, true, "ws   两级 尾 shape");
  run_case_b_offline(256, 256, 256, true, true, false,
                     Fp4GemmScaleMode::kTwoLevel, "B offline 两级 nonws");
  run_case_b_offline(256, 256, 256, true, true, true,
                     Fp4GemmScaleMode::kTwoLevel, "B offline 两级 ws");
  run_case_b_offline(256, 256, 256, false, false, false,
                     Fp4GemmScaleMode::kSingleLevel, "B offline 单级 nonws");
  run_case_b_offline(300, 264, 192, true, true, false,
                     Fp4GemmScaleMode::kTwoLevel, "B offline 两级 尾 shape");
  run_case_tile<128, 8>(256, 256, 256, "nonws 两级 BN=128 s8 (72KB)");
  run_case_tile<256, 5>(256, 256, 256, "nonws 两级 BN=256 s5 (67KB)");
  printf("=== ch37 %s (%d failures) ===\n", g_failures ? "FAIL" : "ALL PASS",
         g_failures);
  return g_failures ? 1 : 0;
}
