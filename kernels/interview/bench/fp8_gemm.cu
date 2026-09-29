// fp8_gemm.cu — notes-v2.cu 多 TU 拆分：fp8_gemm 模块 host 侧 test/bench 函数。
// 函数体自 notes-v2.cu 原样搬移（导出函数去 static），kernel 与 host 封装在
// fp8_gemm.cuh；跨模块共享符号定义在 utils.cu。
#include "fp8_gemm.cuh"

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

#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
// =============================================================================
// Test: FP8 GEMM CuTe (Phase 9) — BF16 输入 -> 动态量化 E4M3 -> FP8 GEMM
// (epilogue 在线反量化) -> BF16 输出，vs CPU fp64 参考。
// 覆盖 4 种 scale 组合(per-row/per-block x per-col/per-block),WS 变体，
// M/N/K 尾部 shape。误差为 E4M3 量化噪声（尾数 3bit），量级见 ch34 分析。
// =============================================================================
static void test_fp8_gemm_once(int M, int N, int K,
                                fp8_gemm::Fp8GemmScaleMode mode, bool use_ws,
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
  fp8_gemm::Fp8GemmWorkspace ws;
  ws.size = fp8_gemm::fp8_gemm_workspace_size(M, N, K);
  cudaMalloc(&ws.buf, ws.size);
  fp8_gemm::fp8_gemm_bf16(d_a, d_b, d_c, M, N, K, mode, ws, 0, use_ws);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("| %-56s | CUDA FAIL: %s\n", label, cudaGetErrorString(err));
  } else {
    cudaMemcpy(h_c, d_c, (size_t)M * N * 2, cudaMemcpyDeviceToHost);
    float max_err = 0;
    for (int i = 0; i < M * N; i++) {
      float e = fabsf(__bfloat162float(h_c[i]) - (float)ref[i]);
      if (e > max_err) max_err = e;
    }
    printf("| %-56s | %.3e |\n", label, max_err);
  }
  free(h_a); free(h_b); free(h_c); free(ref);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(ws.buf);
}

void test_fp8_gemm(int M, int N, int K) {
  using fp8_gemm::Fp8GemmScaleMode;
  test_fp8_gemm_once(M, N, K, Fp8GemmScaleMode::kPerRowPerCol, false,
                     "FP8 GEMM CuTe NonWS (PerRow x PerCol)");
  test_fp8_gemm_once(M, N, K, Fp8GemmScaleMode::kPerRowPerBlock, false,
                     "FP8 GEMM CuTe NonWS (PerRow x PerBlk)");
  test_fp8_gemm_once(M, N, K, Fp8GemmScaleMode::kPerBlockPerCol, false,
                     "FP8 GEMM CuTe NonWS (PerBlk x PerCol)");
  test_fp8_gemm_once(M, N, K, Fp8GemmScaleMode::kPerBlockPerBlock, false,
                     "FP8 GEMM CuTe NonWS (PerBlk x PerBlk)");
  test_fp8_gemm_once(M, N, K, Fp8GemmScaleMode::kPerRowPerCol, true,
                     "FP8 GEMM CuTe WS (PerRow x PerCol)");
  test_fp8_gemm_once(M, N, K, Fp8GemmScaleMode::kPerBlockPerBlock, true,
                     "FP8 GEMM CuTe WS (PerBlk x PerBlk)");
  // 尾部 shape: M 非 128 倍数，N 仅 8 对齐，K 非 64/128 倍数
  test_fp8_gemm_once(300, 264, 144, Fp8GemmScaleMode::kPerRowPerCol, false,
                     "FP8 GEMM CuTe tail M=300 N=264 K=144");
}
#endif

#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
// =============================================================================
// Bench: FP8 GEMM CuTe (Phase 9) — BF16 in -> 动态量化 E4M3 GEMM -> BF16 out
// 精度参照：cuBLAS BF16 GEMM (F32 累加) 同输入输出；误差 = E4M3 量化噪声。
// kernel-only: 量化一次，循环只跑 FP8 GEMM kernel; e2e: 量化+GEMM 全链路。
// 下面所有行都走库默认 tile（Fp8GemmTraits<> 默认实参 128x256x128 s2，96KB
// smem），故行标统一带 ", 128x256/s2"；全 tile/stage 扫描见
// bench_fp8_gemm_tile_sweep（其表里用 "<= lib default" 标出同一档）。
// =============================================================================
void bench_fp8_gemm(int M, int N, int K) {
  using fp8_gemm::Fp8GemmScaleMode;
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
  fp8_gemm::Fp8GemmWorkspace ws;
  ws.size = fp8_gemm::fp8_gemm_workspace_size(M, N, K);
  cudaMalloc(&ws.buf, ws.size);
  // workspace 切分（与 fp8_gemm_bf16 内部同源，直接用库里的对齐函数）
  uint8_t *p = static_cast<uint8_t *>(ws.buf);
  cutlass::float_e4m3_t *a8 = reinterpret_cast<cutlass::float_e4m3_t *>(p);
  p += fp8_gemm::fp8_gemm_align256((size_t)M * K);
  cutlass::float_e4m3_t *b8t = reinterpret_cast<cutlass::float_e4m3_t *>(p);
  p += fp8_gemm::fp8_gemm_align256((size_t)N * K);
  float *sa = reinterpret_cast<float *>(p);
  p += fp8_gemm::fp8_gemm_align256((size_t)M * sizeof(float));
  float *sb = reinterpret_cast<float *>(p);
  // B 离线量化路径：B 段(NK 字节的 B8T)不占 workspace, A 侧只留 O(MK) 暂存。
  fp8_gemm::Fp8GemmActivation act;
  act.size = fp8_gemm::fp8_gemm_activation_size(M, K);
  cudaMalloc(&act.buf, act.size);
  auto *cO = reinterpret_cast<cutlass::bfloat16_t *>(d_c);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

// 行输出：逐元素 max err(vs cuBLAS BF16) + TFLOPS/cuBLAS/加速比。
#define FP8_EMIT_ROW(label, time_ms)                                       \
  do {                                                                     \
    cudaMemcpy(h_c, d_c, (size_t)M * N * 2, cudaMemcpyDeviceToHost);       \
    float max_err = 0;                                                     \
    for (int i = 0; i < M * N; i++) {                                      \
      float e =                                                            \
          fabsf(__bfloat162float(h_c[i]) - __bfloat162float(h_ref[i]));    \
      if (e > max_err) max_err = e;                                        \
    }                                                                      \
    float tflops = bench_hgemm_tflops(M, N, K, time_ms);                   \
    char tflops_str[32];                                                   \
    snprintf(tflops_str, sizeof(tflops_str), "%.1f/%.1f (%.2fx)",          \
             tflops, cublas_tflops, tflops / cublas_tflops);               \
    printf("| %-56s | %.3e | %-19s |\n", label, max_err, tflops_str);      \
  } while (0)

// 计时：先用 e2e 调用填充量化 workspace, 再 warmup + g_repeat 取均值。
// kernel-only = 循环里只跑主 kernel（量化在循环外一次）；e2e = 循环含量化。
#define FP8_TIMED_RUN(MODE, PRA, PCB, kWS, e2e, label)                     \
  do {                                                                     \
    const fp8_gemm::Fp8GemmScaleMode mode = MODE;                          \
    fp8_gemm::fp8_gemm_bf16(d_a, d_b, d_c, M, N, K, mode, ws, 0, kWS);     \
    for (int w = 0; w < g_warmup; ++w) {                                   \
      if (e2e)                                                             \
        fp8_gemm::fp8_gemm_bf16(d_a, d_b, d_c, M, N, K, mode, ws, 0,       \
                                kWS);                                      \
      else                                                                 \
        fp8_gemm::fp8_gemm_tma_fwd<PRA, PCB, kWS>(a8, b8t, sa, sb, cO, M,  \
                                                  N, K, 0);                \
    }                                                                      \
    cudaDeviceSynchronize();                                               \
    cudaEventRecord(start);                                                \
    for (int r = 0; r < g_repeat; ++r) {                                   \
      if (e2e)                                                             \
        fp8_gemm::fp8_gemm_bf16(d_a, d_b, d_c, M, N, K, mode, ws, 0,       \
                                kWS);                                      \
      else                                                                 \
        fp8_gemm::fp8_gemm_tma_fwd<PRA, PCB, kWS>(a8, b8t, sa, sb, cO, M,  \
                                                  N, K, 0);                \
    }                                                                      \
    cudaEventRecord(stop);                                                 \
    cudaEventSynchronize(stop);                                            \
    float time_ms = 0;                                                     \
    cudaEventElapsedTime(&time_ms, start, stop);                           \
    FP8_EMIT_ROW(label, time_ms / g_repeat);                               \
  } while (0)

// 权重 B 离线量化：B 只量化一次（模拟加载已量化 checkpoint），循环里只量化 A。
#define FP8_TIMED_RUN_BOFF(PRA, PCB, kWS, label)                           \
  do {                                                                     \
    fp8_gemm::fp8_gemm_bf16_b_offline<PRA, PCB, kWS>(d_a, b8t, sb, d_c, M, \
                                                     N, K, act, 0);        \
    for (int w = 0; w < g_warmup; ++w)                                     \
      fp8_gemm::fp8_gemm_bf16_b_offline<PRA, PCB, kWS>(d_a, b8t, sb, d_c,  \
                                                      M, N, K, act, 0);    \
    cudaDeviceSynchronize();                                               \
    cudaEventRecord(start);                                                \
    for (int r = 0; r < g_repeat; ++r)                                     \
      fp8_gemm::fp8_gemm_bf16_b_offline<PRA, PCB, kWS>(d_a, b8t, sb, d_c,  \
                                                      M, N, K, act, 0);    \
    cudaEventRecord(stop);                                                 \
    cudaEventSynchronize(stop);                                            \
    float time_ms = 0;                                                     \
    cudaEventElapsedTime(&time_ms, start, stop);                           \
    FP8_EMIT_ROW(label, time_ms / g_repeat);                               \
  } while (0)

  FP8_TIMED_RUN(Fp8GemmScaleMode::kPerRowPerCol, true, true, false, false,
                "FP8 GEMM CuTe NonWS (PerRow x PerCol, 128x256/s2)");
  FP8_TIMED_RUN(Fp8GemmScaleMode::kPerRowPerBlock, true, false, false, false,
                "FP8 GEMM CuTe NonWS (PerRow x PerBlk, 128x256/s2)");
  FP8_TIMED_RUN(Fp8GemmScaleMode::kPerBlockPerCol, false, true, false, false,
                "FP8 GEMM CuTe NonWS (PerBlk x PerCol, 128x256/s2)");
  FP8_TIMED_RUN(Fp8GemmScaleMode::kPerBlockPerBlock, false, false, false, false,
                "FP8 GEMM CuTe NonWS (PerBlk x PerBlk, 128x256/s2)");
  FP8_TIMED_RUN(Fp8GemmScaleMode::kPerRowPerCol, true, true, true, false,
                "FP8 GEMM CuTe WS (PerRow x PerCol, 128x256/s2)");
  FP8_TIMED_RUN(Fp8GemmScaleMode::kPerRowPerCol, true, true, false, true,
                "FP8 GEMM+Quant E2E NonWS (PerRow x PerCol, 128x256/s2)");
  FP8_TIMED_RUN(Fp8GemmScaleMode::kPerRowPerCol, true, true, true, true,
                "FP8 GEMM+Quant E2E WS (PerRow x PerCol, 128x256/s2)");
  // 权重 B 离线量化：模拟"加载已量化的权重 checkpoint", B 量化只做一次且不计入
  // 计时；每次前向只剩 A 侧在线量化 per-token -> GEMM. 推理部署的真实形态。
  fp8_gemm::fp8_gemm_quantize_b(d_b, b8t, sb, N, K, true, 0);
  cudaDeviceSynchronize();
  FP8_TIMED_RUN_BOFF(true, true, false,
                     "FP8 GEMM+A Quant E2E NonWS (B offline, 128x256/s2)");
  FP8_TIMED_RUN_BOFF(true, true, true,
                     "FP8 GEMM+A Quant E2E WS (B offline, 128x256/s2)");
  // randn(-0.25, 0.25) data shape: N(0, (0.25/3)^2) clipped at +-0.25.
  // Verifies error scales with signal amplitude while relative error
  // stays distribution-independent (ch34 error model).
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
    FP8_TIMED_RUN(Fp8GemmScaleMode::kPerRowPerCol, true, true, false, false,
                  "FP8 GEMM CuTe NonWS (RC, randn(+-0.25), 128x256/s2)");
    FP8_TIMED_RUN(Fp8GemmScaleMode::kPerRowPerCol, true, true, true, false,
                  "FP8 GEMM CuTe WS (RC, randn(+-0.25), 128x256/s2)");
  }
#undef FP8_TIMED_RUN
#undef FP8_TIMED_RUN_BOFF
#undef FP8_EMIT_ROW

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(h_a); free(h_b); free(h_c); free(h_ref);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_ref); cudaFree(ws.buf);
  cudaFree(act.buf);
  cublasDestroy(handle);
}

// =============================================================================
// Bench: FP8 GEMM tile 几何 x 流水深度扫描（ch35 性能小节的数据源）
// =============================================================================
// 同一条数据流（量化一次 -> 只计时主 kernel），只换 Fp8GemmTraits<BM,BN,128,S>：
// BM 决定 warp 数（M8N1 全沿 M，线程数 = BM/16*32），BN 决定每线程 acc 量
// （BN/2 个 f32）与 B 侧 TMA 字节，S 决定 smem 深度。三者被同一块 smem
// opt-in 上限反比约束：BN 加倍就得砍一级 stage（本卡 99KB，见下）。

// 只跑指定配置（"128x256x128 s2 WS" / "64x64x128 s3 NonWS"），供 ncu 精确抓
// 单个实例：kernel 名不带模板实参（ncu -k 只能匹配到 fp8_gemm_tma_ws_kernel
// 这一级），只能让程序自己只发目标配置的那几个 launch。
static bool fp8_sweep_want(int bm, int bn, int st, bool ws) {
  static const char *sel = getenv("FP8_SWEEP_CFG");
  if (sel == nullptr) return true;
  char want[32];
  snprintf(want, sizeof(want), "%dx%dx%d s%d %s", bm, bn, 128, st,
           ws ? "WS" : "NonWS");
  return strcmp(sel, want) == 0;
}

template <int kBM, int kBN, int kStages, bool kWS>
static bool launch_timed_fp8_gemm_tile(
    const cutlass::float_e4m3_t *a8, const cutlass::float_e4m3_t *b8t,
    const float *sa, const float *sb, cutlass::bfloat16_t *cO,
    __nv_bfloat16 *h_c, const __nv_bfloat16 *h_ref, int M, int N, int K,
    size_t smem_limit, cudaEvent_t start, cudaEvent_t stop, float &max_err,
    float &time_ms) {
  using Traits = fp8_gemm::Fp8GemmTraits<kBM, kBN, 128, kStages>;
  // 先按公式判可行(留 1KB 给静态 barrier)：超限时 cudaFuncSetAttribute 只返回
  // 错误而 launcher 不看，随后启动的 kernel 可能"跑完但结果错"，必须提前拦住。
  if ((size_t)Traits::kSmemBytes + 1024 > smem_limit) return false;
  fp8_gemm::fp8_gemm_tma_fwd<true, true, kWS, Traits>(a8, b8t, sa, sb, cO, M, N,
                                                      K, 0);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    cudaGetLastError();  // 清掉错误码, 不影响后续配置
    if (g_debug)
      fprintf(stderr, "[fp8 sweep] launch failed: %s\n",
              cudaGetErrorString(err));
    return false;
  }
  for (int w = 0; w < g_warmup; ++w)
    fp8_gemm::fp8_gemm_tma_fwd<true, true, kWS, Traits>(a8, b8t, sa, sb, cO, M,
                                                        N, K, 0);
  cudaDeviceSynchronize();
  cudaEventRecord(start);
  for (int r = 0; r < g_repeat; ++r)
    fp8_gemm::fp8_gemm_tma_fwd<true, true, kWS, Traits>(a8, b8t, sa, sb, cO, M,
                                                        N, K, 0);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float t = 0;
  cudaEventElapsedTime(&t, start, stop);
  time_ms = t / g_repeat;
  check(cudaMemcpy(h_c, cO, (size_t)M * N * 2, cudaMemcpyDeviceToHost),
        "fp8 sweep D2H");
  max_err = 0;
  for (int i = 0; i < M * N; i++) {
    float e = fabsf(__bfloat162float(h_c[i]) - __bfloat162float(h_ref[i]));
    if (e > max_err) max_err = e;
  }
  return true;
}

void bench_fp8_gemm_tile_sweep(int M, int N, int K) {
  using fp8_gemm::Fp8GemmScaleMode;
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

  // 量化前处理只做一次（per-row x per-col），workspace 切分同 fp8_gemm_bf16；
  // 计时循环里只跑主 kernel，故 Max Err 与 bench_fp8_gemm 的 kernel-only 行同源。
  fp8_gemm::Fp8GemmWorkspace ws;
  ws.size = fp8_gemm::fp8_gemm_workspace_size(M, N, K);
  cudaMalloc(&ws.buf, ws.size);
  uint8_t *p = static_cast<uint8_t *>(ws.buf);
  cutlass::float_e4m3_t *a8 = reinterpret_cast<cutlass::float_e4m3_t *>(p);
  p += fp8_gemm::fp8_gemm_align256((size_t)M * K);
  cutlass::float_e4m3_t *b8t = reinterpret_cast<cutlass::float_e4m3_t *>(p);
  p += fp8_gemm::fp8_gemm_align256((size_t)N * K);
  float *sa = reinterpret_cast<float *>(p);
  p += fp8_gemm::fp8_gemm_align256((size_t)M * sizeof(float));
  float *sb = reinterpret_cast<float *>(p);
  cutlass::bfloat16_t *cO = reinterpret_cast<cutlass::bfloat16_t *>(d_ref);
  // 把 workspace 里的 A8/B8T/sa/sb 填好（量化不在计时窗口内）。输出写到
  // d_ref：此时 h_ref 已拷出，d_ref 之后只当 C 缓冲用。
  fp8_gemm::fp8_gemm_bf16(d_a, d_b, d_ref, M, N, K,
                          Fp8GemmScaleMode::kPerRowPerCol, ws, 0, false);
  cudaDeviceSynchronize();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  printf("=== FP8 GEMM tile/stage sweep (M=N=K=%d, PerRow x PerCol, "
         "smem opt-in limit %zuB) ===\n",
         M, smem_limit);
  // 默认档是本扫描的落点，必须让读者一眼找到，别埋在一堆 tile 里
  printf("Note: \"<= lib default\" = library default tile (Fp8GemmTraits<> "
         "default args 128x256x128 s2, also the fastest measured here); "
         "\"<= default geom\" = WS variant of the same geometry.\n");
  printf("| %-56s | %-9s | %-19s |\n", "FP8 GEMM tile/stage", "Max Err",
         "TFLOPS/cuBLAS");
  printf("|----------------------------------------------------------|"
         "-----------|---------------------|\n");

#define FP8_SWEEP_ROW(WS, BM, BN, ST)                                          \
  do {                                                                         \
    if (!fp8_sweep_want(BM, BN, ST, WS)) break;                                \
    constexpr int kSmemKB =                                                    \
        fp8_gemm::Fp8GemmTraits<BM, BN, 128, ST>::kSmemBytes / 1024;           \
    const char *mark = (BM == 128 && BN == 256 && ST == 2)                     \
                           ? (WS ? " <= default geom" : " <= lib default")     \
                           : "";                                               \
    char label[80];                                                            \
    snprintf(label, sizeof(label), "FP8 GEMM %s %dx%dx128 s%d (%dKB)%s",       \
             WS ? "WS" : "NonWS", BM, BN, ST, kSmemKB, mark);                  \
    float max_err = 0, time_ms = 0;                                            \
    if (launch_timed_fp8_gemm_tile<BM, BN, ST, WS>(                            \
            a8, b8t, sa, sb, cO, h_c, h_ref, M, N, K, smem_limit, start, stop, \
            max_err, time_ms)) {                                               \
      float tflops = bench_hgemm_tflops(M, N, K, time_ms);                     \
      char tflops_str[32];                                                     \
      snprintf(tflops_str, sizeof(tflops_str), "%.1f/%.1f (%.2fx)", tflops,    \
               cublas_tflops, tflops / cublas_tflops);                         \
      printf("| %-56s | %.3e | %-19s |\n", label, max_err, tflops_str);        \
    } else {                                                                   \
      printf("| %-56s | %-9s | %-19s |\n", label, "SKIP", "smem/launch");      \
    }                                                                          \
  } while (0)

  // (1) s=3：99KB 上限下只有 128 行以内的几何塞得进三级流水
  FP8_SWEEP_ROW(false, 64, 64, 3);
  FP8_SWEEP_ROW(false, 64, 128, 3);
  FP8_SWEEP_ROW(false, 128, 64, 3);
  FP8_SWEEP_ROW(false, 128, 128, 3);
  // (2) s=2：大 tile 只能配两级流水
  FP8_SWEEP_ROW(false, 64, 64, 2);
  FP8_SWEEP_ROW(false, 64, 128, 2);
  FP8_SWEEP_ROW(false, 64, 256, 2);
  FP8_SWEEP_ROW(false, 128, 64, 2);
  FP8_SWEEP_ROW(false, 128, 128, 2);
  // <== 库默认档（Fp8GemmTraits<> 的默认实参；也是本扫描实测最快档）
  FP8_SWEEP_ROW(false, 128, 256, 2);
  FP8_SWEEP_ROW(false, 256, 64, 2);
  FP8_SWEEP_ROW(false, 256, 128, 2);
  // (3) s=4：只有窄 tile 塞得下
  FP8_SWEEP_ROW(false, 64, 64, 4);
  FP8_SWEEP_ROW(false, 64, 128, 4);
  FP8_SWEEP_ROW(false, 128, 64, 4);
  // (4) 放不下的组合不实例化（省编译时间），只按公式列出所需 smem
  printf("| %-56s | %-9s | %-19s |\n",
         "FP8 GEMM NonWS 128x256x128 s3 (144KB)", "SKIP", "> 99KB smem");
  printf("| %-56s | %-9s | %-19s |\n",
         "FP8 GEMM NonWS 256x256x128 s2 (128KB)", "SKIP", "> 99KB smem");
  printf("| %-56s | %-9s | %-19s |\n",
         "FP8 GEMM NonWS 256x256x128 s3 (192KB)", "SKIP", "> 99KB smem");
  // (5) WS：线程数 = 128(producer) + kBM/16*32(consumer)，setmaxnreg 预算
  //     128*32 + N_c*232 <= 64K 只在 kBM<=128 成立，故只扫 kBM=128/64。这里
  //     扫 kBM=128（384 线程，与 --bench 表里的 ws 行同构）
  FP8_SWEEP_ROW(true, 128, 64, 2);
  FP8_SWEEP_ROW(true, 128, 64, 3);
  FP8_SWEEP_ROW(true, 128, 64, 4);
  FP8_SWEEP_ROW(true, 128, 128, 2);
  FP8_SWEEP_ROW(true, 128, 128, 3);
  // <== 与非 WS 库默认档同几何（WS 只换来 +0.05%，见 35.6 正文）
  FP8_SWEEP_ROW(true, 128, 256, 2);
#undef FP8_SWEEP_ROW

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(h_a); free(h_b); free(h_c); free(h_ref);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_ref); cudaFree(ws.buf);
  cublasDestroy(handle);
}
#endif
