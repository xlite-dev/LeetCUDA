// ffpa_attn.cu: ffpa_attn 模块的 host 侧 test/bench（notes-v2.cu 多 TU 拆分）。
// 函数体从 notes-v2.cu 原样搬移；kernel 与 host 封装见 ffpa_attn.cuh。
#include "hgemm.cuh"  // 必须在前：flash_attn.cuh 依赖 hgemm.cuh 的 swizzle<>()
#include "ffpa_attn.cuh"

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
template <int kHeadDim, int kStagesQK, int kStagesV>
static float bench_fa_split_d_launch(
    int B, int H, int seqlen, half *h_o_ref, float *ref_o,
    half *d_q, half *d_k, half *d_v, half *d_o,
    float cudnn_tflops_f32, float &out_max_err) {
  using namespace cute;
  using Traits = fa_cute::FFPAAttnSplitDCuTeTraits<kHeadDim>;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  constexpr int kBr = 64;
  constexpr int kChunk = 64;
  constexpr int kNumThreads = 256;
  out_max_err = -1.0f;
  if (seqlen < kBr || seqlen % kBr != 0 || seqlen % kChunk != 0) {
    printf("| %-56s | %-9s | %-19s |\n",
           "FA Split-D CuTe TMA MMA WS (unaligned)", "SKIP", "None");
    return -1.0f;
  }

  const int rows = B * H * seqlen;
  const size_t count = (size_t)rows * kHeadDim;
  auto make_tma_q = [=]() {
    auto tensor = make_tensor(
        make_gmem_ptr(reinterpret_cast<cutlass::half_t *>(d_q)),
        make_shape(rows, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, _1{}));
    return make_tma_copy(SM90_TMA_LOAD{}, tensor, SmemLayoutQ{},
                         Shape<_64, _64>{}, _1{});
  };
  auto make_tma_kv = [=](half *pointer) {
    auto tensor = make_tensor(
        make_gmem_ptr(reinterpret_cast<cutlass::half_t *>(pointer)),
        make_shape(rows, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, _1{}));
    return make_tma_copy(SM90_TMA_LOAD{}, tensor, SmemLayoutKV{},
                         Shape<_64, _64>{}, _1{});
  };
  auto tma_q = make_tma_q();
  auto tma_k = make_tma_kv(d_k);
  auto tma_v = make_tma_kv(d_v);
  auto kernel = ffpa_attn_tma_mma_ws_split_d_cute<
      kHeadDim, decltype(tma_q), decltype(tma_k), decltype(tma_v),
      kStagesQK, kStagesV>;
  const int smem_bytes =
      (kStagesQK * (cosize(SmemLayoutQ{}) + cosize(SmemLayoutKV{})) +
       kStagesV * cosize(SmemLayoutKV{})) * sizeof(cutlass::half_t);
  if (!check_smem_feasible((const void *)kernel, smem_bytes)) {
    printf("| %-56s | %-9s | %-19s |\n",
           "FA Split-D CuTe TMA MMA WS (SMEM)", "SMEM", "None");
    return -1.0f;
  }
  check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes),
        "bench split-d set smem");

  dim3 grid(seqlen / kBr, B * H);
  for (int warmup = 0; warmup < g_warmup; ++warmup) {
    kernel<<<grid, kNumThreads, smem_bytes>>>(
        tma_q, tma_k, tma_v,
        reinterpret_cast<cutlass::half_t *>(d_o), rows, seqlen);
  }
  check(cudaGetLastError(), "bench split-d warmup launch");
  check(cudaDeviceSynchronize(), "bench split-d warmup sync");

  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "bench split-d start");
  check(cudaEventCreate(&stop), "bench split-d stop");
  check(cudaEventRecord(start), "bench split-d record start");
  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    kernel<<<grid, kNumThreads, smem_bytes>>>(
        tma_q, tma_k, tma_v,
        reinterpret_cast<cutlass::half_t *>(d_o), rows, seqlen);
  }
  check(cudaGetLastError(), "bench split-d timed launch");
  check(cudaEventRecord(stop), "bench split-d record stop");
  check(cudaEventSynchronize(stop), "bench split-d timing sync");
  float time_ms = 0.0f;
  check(cudaEventElapsedTime(&time_ms, start, stop), "bench split-d elapsed");
  time_ms /= g_repeat;

  half *h_o = (half *)malloc(count * sizeof(half));
  check(cudaMemcpy(h_o, d_o, count * sizeof(half), cudaMemcpyDeviceToHost),
        "bench split-d D2H");
  float max_err = -1.0f;
  if (h_o_ref) {
    max_err = 0.0f;
    for (size_t idx = 0; idx < count; ++idx)
      max_err = max(max_err, fabsf(__half2float(h_o[idx]) - __half2float(h_o_ref[idx])));
  } else if (ref_o) {
    max_err = 0.0f;
    for (size_t idx = 0; idx < count; ++idx)
      max_err = max(max_err, fabsf(__half2float(h_o[idx]) - ref_o[idx]));
  }
  out_max_err = max_err;
  const float tflops = bench_fa_tflops(B, H, seqlen, kHeadDim, time_ms);
  char label[96];
  snprintf(label, sizeof(label),
           "FA Split-D CuTe TMA MMA WS (D=%d, Sk=%d, Sv=%d)",
           kHeadDim, kStagesQK, kStagesV);
  char performance[32];
  if (cudnn_tflops_f32 > 0.0f)
    snprintf(performance, sizeof(performance), "%.1f/%.1f (%.2fx)",
             tflops, cudnn_tflops_f32, tflops / cudnn_tflops_f32);
  else
    snprintf(performance, sizeof(performance), "%.1f", tflops);
  printf("| %-56s | %.3e | %-19s |\n", label, max_err, performance);

  free(h_o);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return tflops;
}

// non-WS TMA split-D bench: tile 128x128, tid=0 内联发 TMA, STSM+TMA store。
// Q/K cta tile (128,32), V/O cta tile (128,64); LSE buffer 分配并传参以
// 覆盖 epilogue LSE 写出路径 (数值校验见 book/tests/ch26c)。
template <int kHeadDim, int kStagesQK, int kStagesPV>
static float bench_fa_split_d_non_ws_launch(
    int B, int H, int seqlen, half *h_o_ref, float *ref_o,
    half *d_q, half *d_k, half *d_v, half *d_o,
    float cudnn_tflops_f32, float &out_max_err) {
  using namespace cute;
  using Traits =
      fa_cute::FFPAAttnNonWSCuTeSplitDTraits<kHeadDim, 128, 128, 32, 64,
                                             kStagesQK, kStagesPV>;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  constexpr int kBr = 128;
  constexpr int kQKDChunk = 32;
  constexpr int kVDChunk = 64;
  constexpr int kNumThreads = Traits::kNumThreads;
  out_max_err = -1.0f;
  if (seqlen < kBr || seqlen % kBr != 0) {
    printf("| %-56s | %-9s | %-19s |\n",
           "FA Split-D CuTe TMA non-WS (unaligned)", "SKIP", "None");
    return -1.0f;
  }

  const int rows = B * H * seqlen;
  const size_t count = (size_t)rows * kHeadDim;
  auto make_tma_qk = [=](half *pointer) {
    auto tensor = make_tensor(
        make_gmem_ptr(reinterpret_cast<cutlass::half_t *>(pointer)),
        make_shape(rows, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, _1{}));
    return make_tma_copy(SM90_TMA_LOAD{}, tensor, SmemLayoutQ{},
                         Shape<_128, Int<kQKDChunk>>{}, _1{});
  };
  auto make_tma_v = [=]() {
    auto tensor = make_tensor(
        make_gmem_ptr(reinterpret_cast<cutlass::half_t *>(d_v)),
        make_shape(rows, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, _1{}));
    return make_tma_copy(SM90_TMA_LOAD{}, tensor, SmemLayoutV{},
                         Shape<_128, Int<kVDChunk>>{}, _1{});
  };
  auto make_tma_o = [=]() {
    auto tensor = make_tensor(
        make_gmem_ptr(reinterpret_cast<cutlass::half_t *>(d_o)),
        make_shape(rows, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, _1{}));
    return make_tma_copy(SM90_TMA_STORE{}, tensor, SmemLayoutO{},
                         Shape<_128, Int<kVDChunk>>{}, _1{});
  };
  auto tma_q = make_tma_qk(d_q);
  auto tma_k = make_tma_qk(d_k);
  auto tma_v = make_tma_v();
  auto tma_o = make_tma_o();
  float *d_lse = nullptr;
  check(cudaMalloc(&d_lse, rows * sizeof(float)), "bench split-d nw alloc lse");
  auto kernel = ffpa_attn_tma_split_d_cute<
      Traits, decltype(tma_q), decltype(tma_k), decltype(tma_v),
      decltype(tma_o)>;
  const int smem_bytes = Traits::kSmemElems * sizeof(cutlass::half_t);
  if (!check_smem_feasible((const void *)kernel, smem_bytes)) {
    printf("| %-56s | %-9s | %-19s |\n",
           "FA Split-D CuTe TMA non-WS (SMEM)", "SMEM", "None");
    cudaFree(d_lse);
    return -1.0f;
  }
  check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes),
        "bench split-d nw set smem");

  dim3 grid(seqlen / kBr, B * H);
  for (int warmup = 0; warmup < g_warmup; ++warmup) {
    kernel<<<grid, kNumThreads, smem_bytes>>>(
        tma_q, tma_k, tma_v, tma_o,
        reinterpret_cast<cutlass::half_t *>(d_o), d_lse, rows, rows, seqlen,
        seqlen);
  }
  check(cudaGetLastError(), "bench split-d nw warmup launch");
  check(cudaDeviceSynchronize(), "bench split-d nw warmup sync");

  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "bench split-d nw start");
  check(cudaEventCreate(&stop), "bench split-d nw stop");
  check(cudaEventRecord(start), "bench split-d nw record start");
  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    kernel<<<grid, kNumThreads, smem_bytes>>>(
        tma_q, tma_k, tma_v, tma_o,
        reinterpret_cast<cutlass::half_t *>(d_o), d_lse, rows, rows, seqlen,
        seqlen);
  }
  check(cudaGetLastError(), "bench split-d nw timed launch");
  check(cudaEventRecord(stop), "bench split-d nw record stop");
  check(cudaEventSynchronize(stop), "bench split-d nw timing sync");
  float time_ms = 0.0f;
  check(cudaEventElapsedTime(&time_ms, start, stop), "bench split-d nw elapsed");
  time_ms /= g_repeat;

  half *h_o = (half *)malloc(count * sizeof(half));
  check(cudaMemcpy(h_o, d_o, count * sizeof(half), cudaMemcpyDeviceToHost),
        "bench split-d nw D2H");
  float max_err = -1.0f;
  if (h_o_ref) {
    max_err = 0.0f;
    for (size_t idx = 0; idx < count; ++idx)
      max_err = max(max_err, fabsf(__half2float(h_o[idx]) - __half2float(h_o_ref[idx])));
  } else if (ref_o) {
    max_err = 0.0f;
    for (size_t idx = 0; idx < count; ++idx)
      max_err = max(max_err, fabsf(__half2float(h_o[idx]) - ref_o[idx]));
  }
  out_max_err = max_err;
  const float tflops = bench_fa_tflops(B, H, seqlen, kHeadDim, time_ms);
  char label[96];
  snprintf(label, sizeof(label),
           "FA Split-D CuTe TMA non-WS (D=%d, Sk=%d, Sv=%d)", kHeadDim,
           kStagesQK, kStagesPV);
  char performance[32];
  if (cudnn_tflops_f32 > 0.0f)
    snprintf(performance, sizeof(performance), "%.1f/%.1f (%.2fx)",
             tflops, cudnn_tflops_f32, tflops / cudnn_tflops_f32);
  else
    snprintf(performance, sizeof(performance), "%.1f", tflops);
  printf("| %-56s | %.3e | %-19s |\n", label, max_err, performance);

  free(h_o);
  cudaFree(d_lse);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return tflops;
}

void bench_fa_split_d_dispatch(
    int B, int H, int seqlen, int head_dim,
    half *h_o_ref, float *ref_o,
    half *d_q, half *d_k, half *d_v, half *d_o,
    float cudnn_tflops_f32) {
  using namespace cute;
  float max_err;
  auto call = [&](auto dim_tag) {
    constexpr int D = decltype(dim_tag)::value;
    bench_fa_split_d_launch<D, 1, 1>(B, H, seqlen, h_o_ref, ref_o,
                                      d_q, d_k, d_v, d_o, cudnn_tflops_f32, max_err);
    bench_fa_split_d_launch<D, 2, 2>(B, H, seqlen, h_o_ref, ref_o,
                                      d_q, d_k, d_v, d_o, cudnn_tflops_f32, max_err);
    // K/V pipeline stages 解耦实验: Sk 是 QK 流水深度 (Q+K 共享),
    // Sv 是 V 流水深度, 两套 barrier/smem 独立, 支持任意 (2,3) 组合。
    bench_fa_split_d_non_ws_launch<D, 2, 2>(B, H, seqlen, h_o_ref, ref_o,
                                            d_q, d_k, d_v, d_o,
                                            cudnn_tflops_f32, max_err);
    bench_fa_split_d_non_ws_launch<D, 2, 3>(B, H, seqlen, h_o_ref, ref_o,
                                            d_q, d_k, d_v, d_o,
                                            cudnn_tflops_f32, max_err);
    bench_fa_split_d_non_ws_launch<D, 3, 2>(B, H, seqlen, h_o_ref, ref_o,
                                            d_q, d_k, d_v, d_o,
                                            cudnn_tflops_f32, max_err);
    bench_fa_split_d_non_ws_launch<D, 3, 3>(B, H, seqlen, h_o_ref, ref_o,
                                            d_q, d_k, d_v, d_o,
                                            cudnn_tflops_f32, max_err);
  };
  if (head_dim == 128) call(Int<128>{});
  else if (head_dim == 192) call(Int<192>{});
  else if (head_dim == 256) call(Int<256>{});
  else if (head_dim == 320) call(Int<320>{});
  else if (head_dim == 384) call(Int<384>{});
  else if (head_dim == 448) call(Int<448>{});
  else if (head_dim == 512) call(Int<512>{});
  else
    printf("| %-56s | %-9s | %-19s |\n",
           "FA Split-D CuTe TMA (unsupported D)", "SKIP", "None");
}

// ch26c: non-WS TMA Split-D 快速正确性入口 (CPU fp64 ref, 无 GQA/causal)。
// 覆盖 dense / stages 2-3 / 多 head / Nq != Nkv / Nkv 非 kBc=128 倍数
// (kv_valid < kBc 边界 mask 分支)。
template <int kHeadDim, int kNq, int kNkv, int kHq = 2, int kStages = 2>
static void test_ffpa_split_d_non_ws_cute() {
  constexpr int kCount = kNq * kHeadDim;
  constexpr int kCountKV = kNkv * kHeadDim;

  half* h_q = (half*)malloc(kCount * kHq * sizeof(half));
  half* h_k = (half*)malloc(kCountKV * kHq * sizeof(half));
  half* h_v = (half*)malloc(kCountKV * kHq * sizeof(half));
  half* h_o = (half*)malloc(kCount * kHq * sizeof(half));
  srand(42 + kHeadDim + kStages * 11 + kHq * 100 + kNkv);
  for (int i = 0; i < kCount * kHq; ++i)
    h_q[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  for (int i = 0; i < kCountKV * kHq; ++i) {
    h_k[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    h_v[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  }

  half *d_q, *d_k, *d_v, *d_o;
  check(cudaMalloc(&d_q, kCount * kHq * sizeof(half)), "sdnw alloc Q");
  check(cudaMalloc(&d_k, kCountKV * kHq * sizeof(half)), "sdnw alloc K");
  check(cudaMalloc(&d_v, kCountKV * kHq * sizeof(half)), "sdnw alloc V");
  check(cudaMalloc(&d_o, kCount * kHq * sizeof(half)), "sdnw alloc O");
  check(cudaMemcpy(d_q, h_q, kCount * kHq * sizeof(half), cudaMemcpyHostToDevice), "sdnw H2D Q");
  check(cudaMemcpy(d_k, h_k, kCountKV * kHq * sizeof(half), cudaMemcpyHostToDevice), "sdnw H2D K");
  check(cudaMemcpy(d_v, h_v, kCountKV * kHq * sizeof(half), cudaMemcpyHostToDevice), "sdnw H2D V");

  // CPU fp64 参考: O = softmax(scale * Q K^T) V
  double* ref_o = (double*)malloc(kCount * kHq * sizeof(double));
  const double scale = 1.0 / sqrt((double)kHeadDim);
  double* S = (double*)malloc(kNkv * sizeof(double));
  for (int h = 0; h < kHq; ++h) {
    const half* q = h_q + (size_t)h * kCount;
    const half* k = h_k + (size_t)h * kCountKV;
    const half* v = h_v + (size_t)h * kCountKV;
    double* o = ref_o + (size_t)h * kCount;
    for (int qi = 0; qi < kNq; ++qi) {
      double smax = -INFINITY;
      for (int kj = 0; kj < kNkv; ++kj) {
        double s = 0.0;
        for (int d = 0; d < kHeadDim; ++d)
          s += (double)__half2float(q[(size_t)qi * kHeadDim + d]) *
               (double)__half2float(k[(size_t)kj * kHeadDim + d]);
        S[kj] = s * scale;
        if (S[kj] > smax) smax = S[kj];
      }
      double sum_exp = 0.0;
      for (int kj = 0; kj < kNkv; ++kj) {
        S[kj] = exp(S[kj] - smax);
        sum_exp += S[kj];
      }
      const double inv = 1.0 / sum_exp;
      for (int d = 0; d < kHeadDim; ++d) {
        double acc = 0.0;
        for (int kj = 0; kj < kNkv; ++kj)
          acc += S[kj] * (double)__half2float(v[(size_t)kj * kHeadDim + d]);
        o[(size_t)qi * kHeadDim + d] = acc * inv;
      }
    }
  }
  free(S);

  ffpa_attn_tma_split_d_cute_fwd<kHeadDim, kStages, kStages>(
      reinterpret_cast<cutlass::half_t*>(d_q),
      reinterpret_cast<cutlass::half_t*>(d_k),
      reinterpret_cast<cutlass::half_t*>(d_v),
      reinterpret_cast<cutlass::half_t*>(d_o),
      /*softmax_lse=*/nullptr, /*B=*/1, kHq, kNq, kNkv);
  check(cudaDeviceSynchronize(), "sdnw sync");
  check(cudaMemcpy(h_o, d_o, kCount * kHq * sizeof(half), cudaMemcpyDeviceToHost),
        "sdnw D2H");

  float max_err = 0.0f;
  for (int i = 0; i < kCount * kHq; ++i)
    max_err = max(max_err, fabsf(__half2float(h_o[i]) - (float)ref_o[i]));
  char label[104];
  snprintf(label, sizeof(label),
           "FA Split-D CuTe TMA non-WS (D=%d, Nq=%d, Nkv=%d, H=%d, s=%d, ref=cpu)",
           kHeadDim, kNq, kNkv, kHq, kStages);
  printf("| %-88s | %.3e |\n", label, max_err);
  if (max_err >= 5e-1f)
    printf("  ^ FAIL: max_err %.3e exceeds 5e-1\n", max_err);

  free(h_q);
  free(h_k);
  free(h_v);
  free(h_o);
  free(ref_o);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_o);
}
#endif

#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
void run_sdnw_cute_tests() {
  test_ffpa_split_d_non_ws_cute<320, 512, 512, 2, 2>();
  test_ffpa_split_d_non_ws_cute<320, 512, 512, 2, 3>();
  test_ffpa_split_d_non_ws_cute<128, 512, 512, 4, 2>();
  test_ffpa_split_d_non_ws_cute<320, 256, 192, 2, 2>();  // Nq!=Nkv + KV 尾 mask
  test_ffpa_split_d_non_ws_cute<192, 128, 256, 2, 2>();  // 多 kv tiles
}
#endif
