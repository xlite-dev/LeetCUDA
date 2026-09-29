// fp4_gemm.cu: notes-v2.cu 多 TU 拆分 — FP4 GEMM (Phase 10) host 侧 test/bench。
// 函数体自 notes-v2.cu 原样搬移（仅导出函数去掉 static），kernel 与封装见 fp4_gemm.cuh。
#include "fp4_gemm.cuh"

// 共享符号：定义在 utils.cu（notes-v2.cu 多 TU 拆分）
extern bool g_debug;
extern bool g_verbose;
extern int g_warmup;
extern int g_repeat;
extern bool g_bench_fa3_cute_only;
extern bool g_fa_skip_check;
void check(cudaError_t err, const char *msg);
bool check_smem_feasible(const void *kernel_func, size_t dyn_smem_bytes);
float bench_hgemm_tflops(int M, int N, int K, float time_ms);
float bench_fa_tflops(int B, int H, int N, int D, float time_ms);
size_t fp8_smem_optin_limit();
float bench_cublas_bf16_gemm_tflops(cublasHandle_t handle, int M, int N, int K,
                                    __nv_bfloat16 *d_a, __nv_bfloat16 *d_b,
                                    __nv_bfloat16 *d_c);

#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS) && \
    defined(NOTES_V2_ENABLE_SM120_FP4)
// =============================================================================
// Test: FP4 GEMM CuTe (Phase 10) — BF16 输入 -> 在线 NVFP4 两级量化 ->
// blockscaled MMA（F32 累加）-> epilogue 反量化 -> BF16 输出，vs CPU fp64 参考。
// 与 FP8 的三点差别：① A4/B4T 是 4-bit 打包（行 K/2 字节）② 第二级 SF 由硬件
// 消费（软件只反量化第一级）③ 误差 O(0.14) 而非 O(0.03)：E2M1 只有 3 个数值位，
// 故这里报 relFro（相对 Frobenius 误差）而非异常值主导的 max err（见 ch36）。
// =============================================================================
// %-56s 按字符数补空格，含 CJK 的标签（单级/两级/A 行…）显示宽度超出字符数，
// 表格会错列；bench_pad56 按显示宽度（CJK 双宽）算补齐空格数。
static int bench_pad56(const char *s) {
  int pad = 56;
  for (const unsigned char *p = (const unsigned char *)s; *p; ++p)
    pad -= (*p >= 0xe4 && *p <= 0xe9) ? 2 : ((*p & 0xc0) != 0x80);
  return pad > 0 ? pad : 0;
}
// =============================================================================
static void test_fp4_gemm_once(int M, int N, int K,
                               fp4_gemm::Fp4GemmScaleMode mode, bool use_ws,
                               const char *label) {
  __nv_bfloat16 *h_a = (__nv_bfloat16 *)malloc((size_t)M * K * sizeof(__nv_bfloat16));
  __nv_bfloat16 *h_b = (__nv_bfloat16 *)malloc((size_t)K * N * sizeof(__nv_bfloat16));
  __nv_bfloat16 *h_c = (__nv_bfloat16 *)malloc((size_t)M * N * sizeof(__nv_bfloat16));
  double *ref = (double *)malloc((size_t)M * N * sizeof(double));
  srand(42);
  for (int i = 0; i < M * K; i++)
    h_a[i] = __float2bfloat16(((float)rand() / RAND_MAX) * 2 - 1);
  for (int i = 0; i < K * N; i++)
    h_b[i] = __float2bfloat16(((float)rand() / RAND_MAX) * 2 - 1);
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      double s = 0;
      for (int k = 0; k < K; k++)
        s += (double)__bfloat162float(h_a[(size_t)m * K + k]) *
             __bfloat162float(h_b[(size_t)k * N + n]);
      ref[(size_t)m * N + n] = s;
    }
  __nv_bfloat16 *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, (size_t)M * K * 2);
  cudaMalloc(&d_b, (size_t)K * N * 2);
  cudaMalloc(&d_c, (size_t)M * N * 2);
  cudaMemcpy(d_a, h_a, (size_t)M * K * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, (size_t)K * N * 2, cudaMemcpyHostToDevice);
  fp4_gemm::Fp4GemmWorkspace ws;
  ws.size = fp4_gemm::fp4_gemm_workspace_size(M, N, K);
  cudaMalloc(&ws.buf, ws.size);
  fp4_gemm::fp4_gemm_bf16(d_a, d_b, d_c, M, N, K, mode, ws, 0, use_ws);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("| %s%*s | CUDA FAIL: %s\n", label, bench_pad56(label), "",
           cudaGetErrorString(err));
  } else {
    cudaMemcpy(h_c, d_c, (size_t)M * N * 2, cudaMemcpyDeviceToHost);
    double num = 0, den = 0;
    for (int i = 0; i < M * N; i++) {
      double d = (double)__bfloat162float(h_c[i]) - ref[i];
      num += d * d;
      den += ref[i] * ref[i];
    }
    printf("| %s%*s | %.4f |\n", label, bench_pad56(label), "",
           sqrt(num / den));
  }
  free(h_a); free(h_b); free(h_c); free(ref);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(ws.buf);
}

void test_fp4_gemm(int M, int N, int K) {
  using fp4_gemm::Fp4GemmScaleMode;
  test_fp4_gemm_once(M, N, K, Fp4GemmScaleMode::kSingleLevel, false,
                     "FP4 GEMM CuTe nonws (单级: 只有 level-2)");
  test_fp4_gemm_once(M, N, K, Fp4GemmScaleMode::kLevel1A, false,
                     "FP4 GEMM CuTe nonws (两级: level-2 x A 行)");
  test_fp4_gemm_once(M, N, K, Fp4GemmScaleMode::kLevel1B, false,
                     "FP4 GEMM CuTe nonws (两级: level-2 x B 列)");
  test_fp4_gemm_once(M, N, K, Fp4GemmScaleMode::kTwoLevel, false,
                     "FP4 GEMM CuTe nonws (两级: A 行 x B 列)");
  test_fp4_gemm_once(M, N, K, Fp4GemmScaleMode::kTwoLevel, true,
                     "FP4 GEMM CuTe ws   (两级: A 行 x B 列)");
  // 尾部 shape: M 非 128 倍数，N 仅 8 对齐，K 非 128 倍数（K%64==0 是硬约束）
  test_fp4_gemm_once(300, 264, 192, Fp4GemmScaleMode::kTwoLevel, false,
                     "FP4 GEMM CuTe tail M=300 N=264 K=192");
  test_fp4_gemm_once(300, 264, 192, Fp4GemmScaleMode::kTwoLevel, true,
                     "FP4 GEMM CuTe tail ws M=300 N=264 K=192");
}
#endif

#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS) && \
    defined(NOTES_V2_ENABLE_SM120_FP4)
// =============================================================================
// Bench: FP4 GEMM CuTe (Phase 10) — BF16 in -> 在线 NVFP4 两级量化 GEMM ->
// BF16 out。参照 cuBLAS BF16 GEMM（同样的输入输出），误差列 = relFro。
// 行结构与 FP8 版对应（kernel-only / e2e / B 离线 / randn 幅度对照），
// 差别在：① 多了"单级 vs 两级"的量化开销对照 ② 误差量级 0.14 vs 0.03。
// 标准行按形状自适应（fp4_gemm_use_wide_tile）：MNK>=4096 用最优档
// 128x256x64 s6（81KB smem），否则保守档 128x128x64 s4（36KB）；
// tile/stage 扫描见 bench_fp4_gemm_tile_sweep。
// =============================================================================
using Fp4BenchWide = fp4_gemm::Fp4GemmTraits<256, 6>;  // MNK>=4096 最优档
using Fp4BenchCons = fp4_gemm::Fp4GemmTraits<>;        // 保守档 128x128/s4

template <bool kWS, typename Traits>
static void fp4_run_kernel_l1_impl(bool l1a, bool l1b,
                                   const cutlass::float_e2m1_t *a4,
                                   const cutlass::float_e2m1_t *b4t,
                                   const cutlass::float_ue4m3_t *sfa,
                                   const cutlass::float_ue4m3_t *sfb,
                                   const float *sa, const float *sb,
                                   cutlass::bfloat16_t *cO, int M, int N,
                                   int K) {
  if (l1a && l1b)
    fp4_gemm::fp4_gemm_tma_fwd<true, true, kWS, Traits>(a4, b4t, sfa, sfb, sa,
                                                       sb, cO, M, N, K, 0);
  else if (l1a)
    fp4_gemm::fp4_gemm_tma_fwd<true, false, kWS, Traits>(a4, b4t, sfa, sfb, sa,
                                                        sb, cO, M, N, K, 0);
  else if (l1b)
    fp4_gemm::fp4_gemm_tma_fwd<false, true, kWS, Traits>(a4, b4t, sfa, sfb, sa,
                                                        sb, cO, M, N, K, 0);
  else
    fp4_gemm::fp4_gemm_tma_fwd<false, false, kWS, Traits>(a4, b4t, sfa, sfb, sa,
                                                         sb, cO, M, N, K, 0);
}

template <bool kWS>
static void fp4_run_kernel_l1(bool l1a, bool l1b,
                              const cutlass::float_e2m1_t *a4,
                              const cutlass::float_e2m1_t *b4t,
                              const cutlass::float_ue4m3_t *sfa,
                              const cutlass::float_ue4m3_t *sfb, const float *sa,
                              const float *sb, cutlass::bfloat16_t *cO, int M,
                              int N, int K) {
  if (fp4_gemm::fp4_gemm_use_wide_tile(M, N, K))
    fp4_run_kernel_l1_impl<kWS, Fp4BenchWide>(l1a, l1b, a4, b4t, sfa, sfb, sa,
                                             sb, cO, M, N, K);
  else
    fp4_run_kernel_l1_impl<kWS, Fp4BenchCons>(l1a, l1b, a4, b4t, sfa, sfb, sa,
                                             sb, cO, M, N, K);
}

template <bool kWS>
static void fp4_run_e2e(const __nv_bfloat16 *d_a, const __nv_bfloat16 *d_b,
                       __nv_bfloat16 *d_c, int M, int N, int K,
                       fp4_gemm::Fp4GemmScaleMode mode,
                       fp4_gemm::Fp4GemmWorkspace &ws) {
  if (fp4_gemm::fp4_gemm_use_wide_tile(M, N, K))
    fp4_gemm::fp4_gemm_bf16<Fp4BenchWide>(d_a, d_b, d_c, M, N, K, mode, ws, 0,
                                          kWS);
  else
    fp4_gemm::fp4_gemm_bf16<Fp4BenchCons>(d_a, d_b, d_c, M, N, K, mode, ws, 0,
                                          kWS);
}

template <bool kWS, typename Traits>
static void fp4_run_offline_impl(bool l1a, bool l1b, const __nv_bfloat16 *d_a,
                                 const cutlass::float_e2m1_t *b4t,
                                 const cutlass::float_ue4m3_t *sfb,
                                 const float *sb, __nv_bfloat16 *d_c, int M,
                                 int N, int K,
                                 fp4_gemm::Fp4GemmActivation &act) {
  if (l1a && l1b)
    fp4_gemm::fp4_gemm_bf16_b_offline<true, true, kWS, Traits>(d_a, b4t, sfb,
                                                              sb, d_c, M, N, K,
                                                              act, 0);
  else if (l1a)
    fp4_gemm::fp4_gemm_bf16_b_offline<true, false, kWS, Traits>(d_a, b4t, sfb,
                                                               sb, d_c, M, N, K,
                                                               act, 0);
  else if (l1b)
    fp4_gemm::fp4_gemm_bf16_b_offline<false, true, kWS, Traits>(d_a, b4t, sfb,
                                                               sb, d_c, M, N, K,
                                                               act, 0);
  else
    fp4_gemm::fp4_gemm_bf16_b_offline<false, false, kWS, Traits>(d_a, b4t, sfb,
                                                                sb, d_c, M, N, K,
                                                                act, 0);
}

template <bool kWS>
static void fp4_run_offline(bool l1a, bool l1b, const __nv_bfloat16 *d_a,
                            const cutlass::float_e2m1_t *b4t,
                            const cutlass::float_ue4m3_t *sfb, const float *sb,
                            __nv_bfloat16 *d_c, int M, int N, int K,
                            fp4_gemm::Fp4GemmActivation &act) {
  if (fp4_gemm::fp4_gemm_use_wide_tile(M, N, K))
    fp4_run_offline_impl<kWS, Fp4BenchWide>(l1a, l1b, d_a, b4t, sfb, sb, d_c, M,
                                            N, K, act);
  else
    fp4_run_offline_impl<kWS, Fp4BenchCons>(l1a, l1b, d_a, b4t, sfb, sb, d_c, M,
                                            N, K, act);
}

void bench_fp4_gemm(int M, int N, int K) {
  using fp4_gemm::Fp4GemmScaleMode;
  __nv_bfloat16 *h_a = (__nv_bfloat16 *)malloc((size_t)M * K * 2);
  __nv_bfloat16 *h_b = (__nv_bfloat16 *)malloc((size_t)K * N * 2);
  __nv_bfloat16 *h_c = (__nv_bfloat16 *)malloc((size_t)M * N * 2);
  __nv_bfloat16 *h_ref = (__nv_bfloat16 *)malloc((size_t)M * N * 2);
  srand(42);
  for (int i = 0; i < M * K; i++)
    h_a[i] = __float2bfloat16(((float)rand() / RAND_MAX) * 2 - 1);
  for (int i = 0; i < K * N; i++)
    h_b[i] = __float2bfloat16(((float)rand() / RAND_MAX) * 2 - 1);
  __nv_bfloat16 *d_a, *d_b, *d_c, *d_ref;
  cudaMalloc(&d_a, (size_t)M * K * 2);
  cudaMalloc(&d_b, (size_t)K * N * 2);
  cudaMalloc(&d_c, (size_t)M * N * 2);
  cudaMalloc(&d_ref, (size_t)M * N * 2);
  cudaMemcpy(d_a, h_a, (size_t)M * K * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, (size_t)K * N * 2, cudaMemcpyHostToDevice);
  cublasHandle_t handle;
  cublasCreate(&handle);
  float cublas_tflops = bench_cublas_bf16_gemm_tflops(handle, M, N, K, d_a, d_b, d_ref);
  cudaMemcpy(h_ref, d_ref, (size_t)M * N * 2, cudaMemcpyDeviceToHost);
  fp4_gemm::Fp4GemmWorkspace ws;
  ws.size = fp4_gemm::fp4_gemm_workspace_size(M, N, K);
  cudaMalloc(&ws.buf, ws.size);
  // workspace 切分（与 fp4_gemm_bf16 内部同源，直接用库里的对齐/SF 尺寸函数）
  uint8_t *p = static_cast<uint8_t *>(ws.buf);
  cutlass::float_e2m1_t *a4 = reinterpret_cast<cutlass::float_e2m1_t *>(p);
  p += fp4_gemm::fp4_gemm_align256((size_t)M * (K / 2));
  cutlass::float_e2m1_t *b4t = reinterpret_cast<cutlass::float_e2m1_t *>(p);
  p += fp4_gemm::fp4_gemm_align256((size_t)N * (K / 2));
  cutlass::float_ue4m3_t *sfa = reinterpret_cast<cutlass::float_ue4m3_t *>(p);
  p += fp4_gemm::fp4_gemm_align256(fp4_gemm::sf_buffer_bytes(M, K));
  cutlass::float_ue4m3_t *sfb = reinterpret_cast<cutlass::float_ue4m3_t *>(p);
  p += fp4_gemm::fp4_gemm_align256(fp4_gemm::sf_buffer_bytes(N, K));
  float *sa = reinterpret_cast<float *>(p);
  p += fp4_gemm::fp4_gemm_align256((size_t)M * sizeof(float));
  float *sb = reinterpret_cast<float *>(p);
  fp4_gemm::Fp4GemmActivation act;
  act.size = fp4_gemm::fp4_gemm_activation_size(M, K);
  cudaMalloc(&act.buf, act.size);
  auto *cO = reinterpret_cast<cutlass::bfloat16_t *>(d_c);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

// 误差列 = relFro（相对 Frobenius 误差, vs cuBLAS BF16 同输入输出）。
// NVFP4 的 max err 由离群元素主导、不随 K 收敛，relFro 才是稳定的质量指标。
#define FP4_EMIT_ROW(label, time_ms)                                          \
  do {                                                                        \
    cudaMemcpy(h_c, d_c, (size_t)M * N * 2, cudaMemcpyDeviceToHost);          \
    double num = 0, den = 0;                                                  \
    for (int i = 0; i < M * N; i++) {                                         \
      double r = (double)__bfloat162float(h_ref[i]);                          \
      double d = (double)__bfloat162float(h_c[i]) - r;                        \
      num += d * d;                                                           \
      den += r * r;                                                           \
    }                                                                         \
    float tflops = bench_hgemm_tflops(M, N, K, time_ms);                      \
    char tflops_str[32];                                                      \
    snprintf(tflops_str, sizeof(tflops_str), "%.1f/%.1f (%.2fx)", tflops,     \
             cublas_tflops, tflops / cublas_tflops);                          \
    printf("| %s%*s | %.3e | %-19s |\n", label, bench_pad56(label), "",       \
           sqrt(num / den), tflops_str);                                      \
  } while (0)

#define FP4_RUN_E2E(MODE, kWS, stem)                                       \
  do {                                                                     \
    fp4_run_e2e<kWS>(d_a, d_b, d_c, M, N, K, MODE, ws);                    \
    for (int w = 0; w < g_warmup; ++w)                                     \
      fp4_run_e2e<kWS>(d_a, d_b, d_c, M, N, K, MODE, ws);                  \
    cudaDeviceSynchronize();                                               \
    cudaEventRecord(start);                                                \
    for (int r = 0; r < g_repeat; ++r)                                     \
      fp4_run_e2e<kWS>(d_a, d_b, d_c, M, N, K, MODE, ws);                  \
    cudaEventRecord(stop);                                                 \
    cudaEventSynchronize(stop);                                            \
    float time_ms = 0;                                                     \
    cudaEventElapsedTime(&time_ms, start, stop);                           \
    char label[96];                                                        \
    snprintf(label, sizeof(label), "%s, %s)", stem, tile_tag);            \
    FP4_EMIT_ROW(label, time_ms / g_repeat);                               \
  } while (0)

#define FP4_RUN_KERNEL(MODE, kWS, stem)                                   \
  do {                                                                     \
    const bool l1a = (MODE == Fp4GemmScaleMode::kTwoLevel ||               \
                      MODE == Fp4GemmScaleMode::kLevel1A);                 \
    const bool l1b = (MODE == Fp4GemmScaleMode::kTwoLevel ||               \
                      MODE == Fp4GemmScaleMode::kLevel1B);                 \
    /* 先跑一次全链路把 workspace（A4/B4T/SFA/SFB/sa/sb）填好；否则          \
       kernel-only 读到的是未初始化数据，误差列会立刻爆表 */               \
    fp4_run_e2e<false>(d_a, d_b, d_c, M, N, K, MODE, ws);                  \
    fp4_run_kernel_l1<kWS>(l1a, l1b, a4, b4t, sfa, sfb, sa, sb, cO, M, N,  \
                           K);                                             \
    for (int w = 0; w < g_warmup; ++w)                                     \
      fp4_run_kernel_l1<kWS>(l1a, l1b, a4, b4t, sfa, sfb, sa, sb, cO, M, N, \
                             K);                                           \
    cudaDeviceSynchronize();                                               \
    cudaEventRecord(start);                                                \
    for (int r = 0; r < g_repeat; ++r)                                     \
      fp4_run_kernel_l1<kWS>(l1a, l1b, a4, b4t, sfa, sfb, sa, sb, cO, M, N, \
                             K);                                           \
    cudaEventRecord(stop);                                                 \
    cudaEventSynchronize(stop);                                            \
    float time_ms = 0;                                                     \
    cudaEventElapsedTime(&time_ms, start, stop);                           \
    char label[96];                                                        \
    snprintf(label, sizeof(label), "%s, %s)", stem, tile_tag);            \
    FP4_EMIT_ROW(label, time_ms / g_repeat);                               \
  } while (0)

#define FP4_RUN_BOFF(MODE, kWS, stem)                                     \
  do {                                                                     \
    const bool l1a = (MODE == Fp4GemmScaleMode::kTwoLevel ||               \
                      MODE == Fp4GemmScaleMode::kLevel1A);                 \
    const bool l1b = (MODE == Fp4GemmScaleMode::kTwoLevel ||               \
                      MODE == Fp4GemmScaleMode::kLevel1B);                 \
    fp4_run_offline<kWS>(l1a, l1b, d_a, b4t, sfb, sb, d_c, M, N, K, act);  \
    for (int w = 0; w < g_warmup; ++w)                                     \
      fp4_run_offline<kWS>(l1a, l1b, d_a, b4t, sfb, sb, d_c, M, N, K, act);\
    cudaDeviceSynchronize();                                               \
    cudaEventRecord(start);                                                \
    for (int r = 0; r < g_repeat; ++r)                                     \
      fp4_run_offline<kWS>(l1a, l1b, d_a, b4t, sfb, sb, d_c, M, N, K, act);\
    cudaEventRecord(stop);                                                 \
    cudaEventSynchronize(stop);                                            \
    float time_ms = 0;                                                     \
    cudaEventElapsedTime(&time_ms, start, stop);                           \
    char label[96];                                                        \
    snprintf(label, sizeof(label), "%s, %s)", stem, tile_tag);            \
    FP4_EMIT_ROW(label, time_ms / g_repeat);                               \
  } while (0)

  // 行为诊断：relFro 只在 K 足够大时才稳定，K=64 的单 stage 情形顺带断言
  // 档位标签与 fp4_run_* 内部的形状判定同源（MNK>=4096 -> 128x256/s6）
  char tile_tag[16];
  snprintf(tile_tag, sizeof(tile_tag), "%s",
           fp4_gemm::fp4_gemm_use_wide_tile(M, N, K) ? "128x256/s6"
                                                     : "128x128/s4");
  FP4_RUN_KERNEL(Fp4GemmScaleMode::kSingleLevel, false,
                 "FP4 GEMM CuTe nonws (单级 level-2");
  FP4_RUN_KERNEL(Fp4GemmScaleMode::kLevel1A, false,
                 "FP4 GEMM CuTe nonws (level-2 x A 行");
  FP4_RUN_KERNEL(Fp4GemmScaleMode::kLevel1B, false,
                 "FP4 GEMM CuTe nonws (level-2 x B 列");
  FP4_RUN_KERNEL(Fp4GemmScaleMode::kTwoLevel, false,
                 "FP4 GEMM CuTe nonws (两级 A 行 x B 列");
  FP4_RUN_KERNEL(Fp4GemmScaleMode::kTwoLevel, true,
                 "FP4 GEMM CuTe ws   (两级 A 行 x B 列");
  FP4_RUN_E2E(Fp4GemmScaleMode::kSingleLevel, false,
              "FP4 GEMM+Quant e2e nonws (单级 level-2");
  FP4_RUN_E2E(Fp4GemmScaleMode::kTwoLevel, false,
              "FP4 GEMM+Quant e2e nonws (两级 A 行 x B 列");
  FP4_RUN_E2E(Fp4GemmScaleMode::kTwoLevel, true,
              "FP4 GEMM+Quant e2e ws   (两级 A 行 x B 列");
  // 权重 B 离线量化：模拟"加载已量化的权重 checkpoint"，B 的量化 + 转置只做
  // 一次且不计入计时；每次前向只剩 A 侧在线量化 -> NVFP4 GEMM. 部署真实形态.
  fp4_gemm::fp4_gemm_quantize_b(d_b, b4t, sfb, sb, N, K, true, 0);
  cudaDeviceSynchronize();
  FP4_RUN_BOFF(Fp4GemmScaleMode::kTwoLevel, false,
               "FP4 GEMM+A Quant e2e nonws (B offline");
  FP4_RUN_BOFF(Fp4GemmScaleMode::kTwoLevel, true,
               "FP4 GEMM+A Quant e2e ws   (B offline");
  // randn(-0.25, 0.25) data shape: 幅度缩小 4x，验证 relFro 与信号幅度无关；
  // 两级量化里 level-1 的 per-row scale 正是在这里体现价值（见 ch36 误差模型）
  {
    auto randn_clip = []() {
      float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
      float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
      float x = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2) * (0.25f / 3.0f);
      return fminf(fmaxf(x, -0.25f), 0.25f);
    };
    srand(42);
    for (int i = 0; i < M * K; i++)
      h_a[i] = __float2bfloat16(randn_clip());
    for (int i = 0; i < K * N; i++)
      h_b[i] = __float2bfloat16(randn_clip());
    cudaMemcpy(d_a, h_a, (size_t)M * K * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, (size_t)K * N * 2, cudaMemcpyHostToDevice);
    cublas_tflops =
        bench_cublas_bf16_gemm_tflops(handle, M, N, K, d_a, d_b, d_ref);
    cudaMemcpy(h_ref, d_ref, (size_t)M * N * 2, cudaMemcpyDeviceToHost);
    FP4_RUN_KERNEL(Fp4GemmScaleMode::kTwoLevel, false,
                   "FP4 GEMM CuTe nonws (两级, randn(+-0.25)");
    FP4_RUN_KERNEL(Fp4GemmScaleMode::kTwoLevel, true,
                   "FP4 GEMM CuTe ws   (两级, randn(+-0.25)");
  }
#undef FP4_RUN_E2E
#undef FP4_RUN_KERNEL
#undef FP4_RUN_BOFF
#undef FP4_EMIT_ROW

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(h_a); free(h_b); free(h_c); free(h_ref);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_ref); cudaFree(ws.buf);
  cudaFree(act.buf);
  cublasDestroy(handle);
}

// =============================================================================
// Bench: FP4 GEMM tile 几何 x 流水深度扫描（ch37 性能小节的数据源）
// =============================================================================
// BM 固定 128（SF 行块 + 原子布局决定的物理约束），可扫的是 BN 与流水深度：
//   BK = 64 固定（= 原子 K 宽，一跳一 stage）
//   每 stage 字节 = 4096(A) + 32*BN(B) + 512(SFA) + 4*BN(SFB)
//   BN=128 -> 9.0KB/stage; BN=256 -> 13.5KB/stage
// 两级约束：① 总 smem <= 99KB ② O staging(bf16 128xBN) 要能塞进 kStages 个
// stage 的复用区（BN=256 时 O 就要 64KB，故 kStages>=5）。
static bool fp4_sweep_want(int bn, int st, bool ws) {
  static const char *sel = getenv("FP4_SWEEP_CFG");
  if (sel == nullptr) return true;
  char want[32];
  snprintf(want, sizeof(want), "%dx64 s%d %s", bn, st, ws ? "ws" : "nonws");
  return strcmp(sel, want) == 0;
}

template <int kBN, int kStages, bool kWS>
static bool launch_timed_fp4_gemm_tile(
    const cutlass::float_e2m1_t *a4, const cutlass::float_e2m1_t *b4t,
    const cutlass::float_ue4m3_t *sfa, const cutlass::float_ue4m3_t *sfb,
    const float *sa, const float *sb, cutlass::bfloat16_t *cO,
    __nv_bfloat16 *h_c, const __nv_bfloat16 *h_ref, int M, int N, int K,
    size_t smem_limit, cudaEvent_t start, cudaEvent_t stop, float &relfro,
    float &time_ms) {
  using Traits = fp4_gemm::Fp4GemmTraits<kBN, kStages>;
  if ((size_t)Traits::kSmemBytes + 1024 > smem_limit) return false;
  fp4_gemm::fp4_gemm_tma_fwd<true, true, kWS, Traits>(a4, b4t, sfa, sfb, sa, sb,
                                                      cO, M, N, K, 0);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    cudaGetLastError();
    if (g_debug)
      fprintf(stderr, "[fp4 sweep] launch failed: %s\n",
              cudaGetErrorString(err));
    return false;
  }
  for (int w = 0; w < g_warmup; ++w)
    fp4_gemm::fp4_gemm_tma_fwd<true, true, kWS, Traits>(a4, b4t, sfa, sfb, sa,
                                                        sb, cO, M, N, K, 0);
  cudaDeviceSynchronize();
  cudaEventRecord(start);
  for (int r = 0; r < g_repeat; ++r)
    fp4_gemm::fp4_gemm_tma_fwd<true, true, kWS, Traits>(a4, b4t, sfa, sfb, sa,
                                                        sb, cO, M, N, K, 0);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float t = 0;
  cudaEventElapsedTime(&t, start, stop);
  time_ms = t / g_repeat;
  cudaMemcpy(h_c, cO, (size_t)M * N * 2, cudaMemcpyDeviceToHost);
  double num = 0, den = 0;
  for (int i = 0; i < M * N; i++) {
    double r = (double)__bfloat162float(h_ref[i]);
    double d = (double)__bfloat162float(h_c[i]) - r;
    num += d * d;
    den += r * r;
  }
  relfro = (float)sqrt(num / den);
  return true;
}

void bench_fp4_gemm_tile_sweep(int M, int N, int K) {
  const size_t smem_limit = fp8_smem_optin_limit();
  __nv_bfloat16 *h_a = (__nv_bfloat16 *)malloc((size_t)M * K * 2);
  __nv_bfloat16 *h_b = (__nv_bfloat16 *)malloc((size_t)K * N * 2);
  __nv_bfloat16 *h_c = (__nv_bfloat16 *)malloc((size_t)M * N * 2);
  __nv_bfloat16 *h_ref = (__nv_bfloat16 *)malloc((size_t)M * N * 2);
  srand(42);
  for (int i = 0; i < M * K; i++)
    h_a[i] = __float2bfloat16(((float)rand() / RAND_MAX) * 2 - 1);
  for (int i = 0; i < K * N; i++)
    h_b[i] = __float2bfloat16(((float)rand() / RAND_MAX) * 2 - 1);
  __nv_bfloat16 *d_a, *d_b, *d_ref;
  cudaMalloc(&d_a, (size_t)M * K * 2);
  cudaMalloc(&d_b, (size_t)K * N * 2);
  cudaMalloc(&d_ref, (size_t)M * N * 2);
  cudaMemcpy(d_a, h_a, (size_t)M * K * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, (size_t)K * N * 2, cudaMemcpyHostToDevice);
  cublasHandle_t handle;
  cublasCreate(&handle);
  float cublas_tflops =
      bench_cublas_bf16_gemm_tflops(handle, M, N, K, d_a, d_b, d_ref);
  cudaMemcpy(h_ref, d_ref, (size_t)M * N * 2, cudaMemcpyDeviceToHost);
  // 量化前处理（两级）只做一次，计时窗口里只跑主 kernel
  fp4_gemm::Fp4GemmWorkspace ws;
  ws.size = fp4_gemm::fp4_gemm_workspace_size(M, N, K);
  cudaMalloc(&ws.buf, ws.size);
  uint8_t *p = static_cast<uint8_t *>(ws.buf);
  cutlass::float_e2m1_t *a4 = reinterpret_cast<cutlass::float_e2m1_t *>(p);
  p += fp4_gemm::fp4_gemm_align256((size_t)M * (K / 2));
  cutlass::float_e2m1_t *b4t = reinterpret_cast<cutlass::float_e2m1_t *>(p);
  p += fp4_gemm::fp4_gemm_align256((size_t)N * (K / 2));
  cutlass::float_ue4m3_t *sfa = reinterpret_cast<cutlass::float_ue4m3_t *>(p);
  p += fp4_gemm::fp4_gemm_align256(fp4_gemm::sf_buffer_bytes(M, K));
  cutlass::float_ue4m3_t *sfb = reinterpret_cast<cutlass::float_ue4m3_t *>(p);
  p += fp4_gemm::fp4_gemm_align256(fp4_gemm::sf_buffer_bytes(N, K));
  float *sa = reinterpret_cast<float *>(p);
  p += fp4_gemm::fp4_gemm_align256((size_t)M * sizeof(float));
  float *sb = reinterpret_cast<float *>(p);
  cutlass::bfloat16_t *cO = reinterpret_cast<cutlass::bfloat16_t *>(d_ref);
  // 把 workspace 里的 A4/B4T/SFA/SFB/sa/sb 填好（量化不在计时窗口内）。输出
  // 写到 d_ref：h_ref 已拷出，d_ref 之后只当 C 缓冲用。
  fp4_gemm::fp4_gemm_bf16(d_a, d_b, d_ref, M, N, K,
                          fp4_gemm::Fp4GemmScaleMode::kTwoLevel, ws, 0, false);
  cudaDeviceSynchronize();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  printf("=== FP4 GEMM tile/stage sweep (M=N=K=%d, 两级量化, smem opt-in "
         "limit %zuB) ===\n",
         M, smem_limit);
  printf("BM 固定 128（SF 行块约束），BK 固定 64（= MMA 原子 K 宽）；"
         "\"<= lib default\" = 库默认档 Fp4GemmTraits<> 128x128x64 s4。\n");
  printf("| %-56s | %-9s | %-19s |\n", "FP4 GEMM tile/stage", "relFro",
         "TFLOPS/cuBLAS");
  printf("|----------------------------------------------------------|"
         "-----------|---------------------|\n");

#define FP4_SWEEP_ROW(WS, BN, ST)                                             \
  do {                                                                        \
    if (!fp4_sweep_want(BN, ST, WS)) break;                                   \
    using T = fp4_gemm::Fp4GemmTraits<BN, ST>;                                \
    static_assert(T::kOStagingFits, "本表的行都已按公式筛过 (见下方注解)");   \
    constexpr int kSmemKB = T::kSmemBytes / 1024;                             \
    const char *mark = (BN == 128 && ST == 4) ? " <= lib default" : "";       \
    char label[80];                                                           \
    snprintf(label, sizeof(label), "FP4 GEMM %s 128x%dx64 s%d (%dKB)%s",      \
             WS ? "ws   " : "nonws", BN, ST, kSmemKB, mark);                  \
    float relfro = 0, time_ms = 0;                                            \
    if (launch_timed_fp4_gemm_tile<BN, ST, WS>(                               \
            a4, b4t, sfa, sfb, sa, sb, cO, h_c, h_ref, M, N, K, smem_limit,   \
            start, stop, relfro, time_ms)) {                                  \
      float tflops = bench_hgemm_tflops(M, N, K, time_ms);                    \
      char tflops_str[32];                                                    \
      snprintf(tflops_str, sizeof(tflops_str), "%.1f/%.1f (%.2fx)", tflops,   \
               cublas_tflops, tflops / cublas_tflops);                        \
      printf("| %-56s | %.3e | %-19s |\n", label, relfro, tflops_str);        \
    } else {                                                                  \
      printf("| %-56s | %-9s | %-19s |\n", label, "SKIP", "smem/launch");     \
    }                                                                         \
  } while (0)

  // 合法性由两条公式决定（不合法的组合不实例化，省编译时间，只列表说明）：
  //   BN=128: stage 9.0KB, O 32KB -> 4 <= s <= 11 (11*9=99KB)
  //   BN=256: stage 13.5KB, O 64KB -> 5 <= s <= 7  (7*13.5=94.5KB)
  // (1) 默认几何（BN=128）加深流水：36KB 起步，最深 90KB
  FP4_SWEEP_ROW(false, 128, 4);
  FP4_SWEEP_ROW(false, 128, 6);
  FP4_SWEEP_ROW(false, 128, 8);
  FP4_SWEEP_ROW(false, 128, 10);
  // (2) BN=256：每 stage 13.5KB，但 O staging(64KB) 要求 s>=5
  FP4_SWEEP_ROW(false, 256, 5);
  FP4_SWEEP_ROW(false, 256, 6);
  FP4_SWEEP_ROW(false, 256, 7);
  // (3) WS：同几何对照（producer 128 线程 + consumer 256 线程）
  FP4_SWEEP_ROW(true, 128, 4);
  FP4_SWEEP_ROW(true, 128, 6);
  FP4_SWEEP_ROW(true, 128, 8);
  FP4_SWEEP_ROW(true, 256, 5);
  FP4_SWEEP_ROW(true, 256, 6);
  // (4) 两条公式排除掉的组合（列出来让读者看清边界，不实例化）
  printf("| %-56s | %-9s | %-19s |\n",
         "FP4 GEMM nonws 128x128x64 s2", "SKIP", "O staging 32KB > 18KB");
  printf("| %-56s | %-9s | %-19s |\n",
         "FP4 GEMM nonws 128x128x64 s3", "SKIP", "O staging 32KB > 27KB");
  printf("| %-56s | %-9s | %-19s |\n",
         "FP4 GEMM nonws 128x128x64 s12", "SKIP", "> 99KB smem");
  printf("| %-56s | %-9s | %-19s |\n",
         "FP4 GEMM nonws 128x256x64 s4", "SKIP", "O staging 64KB > 54KB");
#undef FP4_SWEEP_ROW

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(h_a); free(h_b); free(h_c); free(h_ref);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_ref); cudaFree(ws.buf);
  cublasDestroy(handle);
}
#endif
