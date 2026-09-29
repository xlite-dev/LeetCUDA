// flash_attn.cu — flash_attn 模块 host 侧 test/bench 函数（自 notes-v2.cu 拆分）。
// kernel 与 host 封装在 flash_attn.cuh；hgemm.cuh 必须先 include：
// flash_attn.cuh 的 kernel 依赖 hgemm.cuh 中的 swizzle<>()。
#include "hgemm.cuh"
#include "flash_attn.cuh"

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
// ffpa_attn.cu 导出：bench_flash_attn 的 head_dim>128 分支调用
void bench_fa_split_d_dispatch(int B, int H, int seqlen, int head_dim,
                               half *h_o_ref, float *ref_o,
                               half *d_q, half *d_k, half *d_v, half *d_o,
                               float cudnn_tflops_f32);
#endif

// FALayout：与 notes-v2.cu 中的定义保持一致（枚举值顺序不可变）
enum class FALayout {
  All, Pad, SwizzleQ, SwizzleK, SwizzleV, SwizzleQK, SwizzleQV, SwizzleKV, Swizzle
};
FALayout g_fa_layout = FALayout::Pad;   // main() 解析 --fa-layout 后写入

static float g_fa_f16_max_tflops = 0.0f;
static float g_fa_f32_max_tflops = 0.0f;

// Decide whether to print a FA TFLOPS line. When --verbose/--debug is off,
// only print when the current TFLOPS exceeds the running max for its
// accumulator category (f16/f32). Correctness failures always print so they
// are never silently dropped (callers gate FAIL paths themselves).
static bool should_print_fa_tflops(int acc_f32, float tflops) {
  if (g_verbose || g_debug) return true;
  float &max_tflops = acc_f32 ? g_fa_f32_max_tflops : g_fa_f16_max_tflops;
  if (tflops > max_tflops) {
    max_tflops = tflops;
    return true;
  }
  return false;
}

// Decide whether to print a HGEMM TFLOPS line. When --verbose/--debug is off,
// only print when the current TFLOPS exceeds the running max for its

#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
template <int kHeadDim>
static void test_flash_attn_3_cute_tma_copy_smoke() {
  using namespace cute;
  using Traits = fa_cute::FlashAttn3CuTeTraits<kHeadDim>;
  using SmemLayout = typename Traits::SmemLayoutQKV;
  constexpr int kRows = 128;
  constexpr int kCount = kRows * kHeadDim;

  half *h_input = (half *)malloc(kCount * sizeof(half));
  half *h_output = (half *)malloc(kCount * sizeof(half));
  for (int idx = 0; idx < kCount; ++idx) {
    h_input[idx] = __float2half((idx % 251) * 0.00390625f);
  }

  half *d_input;
  half *d_output;
  check(cudaMalloc(&d_input, kCount * sizeof(half)), "cute tma smoke alloc input");
  check(cudaMalloc(&d_output, kCount * sizeof(half)), "cute tma smoke alloc output");
  check(cudaMemcpy(d_input, h_input, kCount * sizeof(half), cudaMemcpyHostToDevice),
        "cute tma smoke H2D");

  auto mQ = make_tensor(
      make_gmem_ptr(reinterpret_cast<cutlass::half_t *>(d_input)),
      make_shape(kRows, Int<kHeadDim>{}),
      make_stride(Int<kHeadDim>{}, _1{}));
  auto tma_q = make_tma_copy(
      SM90_TMA_LOAD{}, mQ, SmemLayout{},
      Shape<_64, Int<kHeadDim>>{}, _1{});
  auto kernel = flash_attn_3_cute_tma_copy_smoke<kHeadDim, decltype(tma_q)>;
  constexpr int kSmemBytes = cosize(SmemLayout{}) * sizeof(cutlass::half_t);
  check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kSmemBytes),
        "cute tma smoke set smem");
  kernel<<<2, 128, kSmemBytes>>>(
      tma_q, reinterpret_cast<cutlass::half_t *>(d_output), kRows);
  check(cudaGetLastError(), "cute tma smoke launch");
  check(cudaDeviceSynchronize(), "cute tma smoke sync");
  check(cudaMemcpy(h_output, d_output, kCount * sizeof(half), cudaMemcpyDeviceToHost),
        "cute tma smoke D2H");

  float max_err = 0.0f;
  for (int idx = 0; idx < kCount; ++idx) {
    max_err = max(max_err, fabsf(__half2float(h_input[idx]) -
                                 __half2float(h_output[idx])));
  }
  char label[64];
  snprintf(label, sizeof(label), "CuTe TMA copy smoke (D=%d)", kHeadDim);
  printf("| %-56s | %.3e |\n", label, max_err);

  free(h_input);
  free(h_output);
  cudaFree(d_input);
  cudaFree(d_output);
}

template <int kHeadDim>
static void test_flash_attn_3_tma_mma_ws_split_q_cute() {
  using namespace cute;
  using Traits = fa_cute::FlashAttn3CuTeTraits<kHeadDim>;
  using SmemLayout = typename Traits::SmemLayoutQKV;
  constexpr int kSeqlen = 128;
  constexpr int kCount = kSeqlen * kHeadDim;

  half *h_q = (half *)malloc(kCount * sizeof(half));
  half *h_k = (half *)malloc(kCount * sizeof(half));
  half *h_v = (half *)malloc(kCount * sizeof(half));
  half *h_o = (half *)malloc(kCount * sizeof(half));
  float *ref_o = (float *)malloc(kCount * sizeof(float));
  srand(42 + kHeadDim);
  for (int idx = 0; idx < kCount; ++idx) {
    h_q[idx] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    h_k[idx] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    h_v[idx] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  }

  float scale = 1.0f / sqrtf((float)kHeadDim);
  for (int row = 0; row < kSeqlen; ++row) {
    float scores[kSeqlen];
    float row_max = -INFINITY;
    for (int col = 0; col < kSeqlen; ++col) {
      float score = 0.0f;
      for (int dim = 0; dim < kHeadDim; ++dim) {
        score += __half2float(h_q[row * kHeadDim + dim]) *
                 __half2float(h_k[col * kHeadDim + dim]);
      }
      scores[col] = score * scale;
      row_max = max(row_max, scores[col]);
    }
    float row_sum = 0.0f;
    for (int col = 0; col < kSeqlen; ++col) {
      scores[col] = expf(scores[col] - row_max);
      row_sum += scores[col];
    }
    for (int dim = 0; dim < kHeadDim; ++dim) {
      float output = 0.0f;
      for (int col = 0; col < kSeqlen; ++col) {
        output += scores[col] * __half2float(h_v[col * kHeadDim + dim]);
      }
      ref_o[row * kHeadDim + dim] = output / row_sum;
    }
  }

  half *d_q;
  half *d_k;
  half *d_v;
  half *d_o;
  check(cudaMalloc(&d_q, kCount * sizeof(half)), "cute fa3 alloc Q");
  check(cudaMalloc(&d_k, kCount * sizeof(half)), "cute fa3 alloc K");
  check(cudaMalloc(&d_v, kCount * sizeof(half)), "cute fa3 alloc V");
  check(cudaMalloc(&d_o, kCount * sizeof(half)), "cute fa3 alloc O");
  check(cudaMemcpy(d_q, h_q, kCount * sizeof(half), cudaMemcpyHostToDevice),
        "cute fa3 H2D Q");
  check(cudaMemcpy(d_k, h_k, kCount * sizeof(half), cudaMemcpyHostToDevice),
        "cute fa3 H2D K");
  check(cudaMemcpy(d_v, h_v, kCount * sizeof(half), cudaMemcpyHostToDevice),
        "cute fa3 H2D V");

  auto make_tma = [=](half *pointer) {
    auto tensor = make_tensor(
        make_gmem_ptr(reinterpret_cast<cutlass::half_t *>(pointer)),
        make_shape(kSeqlen, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, _1{}));
    return make_tma_copy(
        SM90_TMA_LOAD{}, tensor, SmemLayout{},
        Shape<_64, Int<kHeadDim>>{}, _1{});
  };
  auto tma_q = make_tma(d_q);
  auto tma_k = make_tma(d_k);
  auto tma_v = make_tma(d_v);
  auto kernel = flash_attn_3_tma_mma_ws_split_q_cute<
      kHeadDim, decltype(tma_q), decltype(tma_k), decltype(tma_v)>;
  auto acc_o = partition_fragment_C(
      typename Traits::TiledMma{}, Shape<_64, Int<kHeadDim>>{});
  constexpr int kNumConsumers = 2;
  constexpr int kStagesK = 1;
  constexpr int kTiles = 1 + kNumConsumers * kStagesK + kNumConsumers;
  constexpr int kTilesBytes =
      kTiles * cosize(SmemLayout{}) * sizeof(cutlass::half_t);
  int merge_bytes = 128 * size(acc_o) * sizeof(float) + 128 * sizeof(float4);
  int smem_bytes = max(kTilesBytes, merge_bytes);
  check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes),
        "cute fa3 set smem");
  kernel<<<dim3(2, 1), 384, smem_bytes>>>(
      tma_q, tma_k, tma_v,
      reinterpret_cast<cutlass::half_t *>(d_o), kSeqlen, kSeqlen);
  check(cudaGetLastError(), "cute fa3 launch");
  check(cudaDeviceSynchronize(), "cute fa3 sync");
  check(cudaMemcpy(h_o, d_o, kCount * sizeof(half), cudaMemcpyDeviceToHost),
        "cute fa3 D2H");

  float max_err = 0.0f;
  for (int idx = 0; idx < kCount; ++idx) {
    max_err = max(max_err, fabsf(__half2float(h_o[idx]) - ref_o[idx]));
  }
  char label[80];
  snprintf(label, sizeof(label), "FA3 CuTe TMA MMA WS (2 Consumer WG) (Sk=1, Sv=1, F32Acc, D=%d)", kHeadDim);
  printf("| %-56s | %.3e |\n", label, max_err);

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
// FA2-style CuTe cp.async test: single consumer, Br=128, no TMA/WS.
// kSeqlen=256 -> 2 Q tiles, covers multi-Q-tile path.
// kStagesK=2 验证 cp.async pipeline + V 单 buffer + V/K group 策略正确性。
template <int kHeadDim>
static void test_flash_attn_mma_stages_split_q_cute() {
  using namespace cute;
  using Traits = fa_cute::FlashAttn2CuTeTraits<kHeadDim>;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  constexpr int kBr = 128;
  constexpr int kStagesK = 2;
  constexpr int kSeqlen = 256;
  constexpr int kCount = kSeqlen * kHeadDim;

  half *h_q = (half *)malloc(kCount * sizeof(half));
  half *h_k = (half *)malloc(kCount * sizeof(half));
  half *h_v = (half *)malloc(kCount * sizeof(half));
  half *h_o = (half *)malloc(kCount * sizeof(half));
  float *ref_o = (float *)malloc(kCount * sizeof(float));
  srand(42 + kHeadDim);
  for (int idx = 0; idx < kCount; ++idx) {
    h_q[idx] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    h_k[idx] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    h_v[idx] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  }

  // CPU FP32 naive attention reference
  float scale = 1.0f / sqrtf((float)kHeadDim);
  for (int row = 0; row < kSeqlen; ++row) {
    float scores[kSeqlen];
    float row_max = -INFINITY;
    for (int col = 0; col < kSeqlen; ++col) {
      float score = 0.0f;
      for (int dim = 0; dim < kHeadDim; ++dim) {
        score += __half2float(h_q[row * kHeadDim + dim]) *
                 __half2float(h_k[col * kHeadDim + dim]);
      }
      scores[col] = score * scale;
      row_max = max(row_max, scores[col]);
    }
    float row_sum = 0.0f;
    for (int col = 0; col < kSeqlen; ++col) {
      scores[col] = expf(scores[col] - row_max);
      row_sum += scores[col];
    }
    for (int dim = 0; dim < kHeadDim; ++dim) {
      float output = 0.0f;
      for (int col = 0; col < kSeqlen; ++col) {
        output += scores[col] * __half2float(h_v[col * kHeadDim + dim]);
      }
      ref_o[row * kHeadDim + dim] = output / row_sum;
    }
  }

  half *d_q;
  half *d_k;
  half *d_v;
  half *d_o;
  check(cudaMalloc(&d_q, kCount * sizeof(half)), "cute fa2 cpasync alloc Q");
  check(cudaMalloc(&d_k, kCount * sizeof(half)), "cute fa2 cpasync alloc K");
  check(cudaMalloc(&d_v, kCount * sizeof(half)), "cute fa2 cpasync alloc V");
  check(cudaMalloc(&d_o, kCount * sizeof(half)), "cute fa2 cpasync alloc O");
  check(cudaMemcpy(d_q, h_q, kCount * sizeof(half), cudaMemcpyHostToDevice),
        "cute fa2 cpasync H2D Q");
  check(cudaMemcpy(d_k, h_k, kCount * sizeof(half), cudaMemcpyHostToDevice),
        "cute fa2 cpasync H2D K");
  check(cudaMemcpy(d_v, h_v, kCount * sizeof(half), cudaMemcpyHostToDevice),
        "cute fa2 cpasync H2D V");

  // 无需 TMA descriptor，直接传指针
  auto kernel = flash_attn_mma_stages_split_q_cute<kHeadDim, kStagesK>;
  // smem = Q[128*D] + K[Sk*64*D] + V[1*64*D]
  constexpr int kSmemBytes =
      (size(SmemLayoutQ{}) +
       kStagesK * size(SmemLayoutKV{}) +
       1 * size(SmemLayoutKV{})) * sizeof(cutlass::half_t);
  check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kSmemBytes),
        "cute fa2 cpasync set smem");
  kernel<<<dim3(kSeqlen / kBr, 1), 256, kSmemBytes>>>(
      reinterpret_cast<cutlass::half_t *>(d_q),
      reinterpret_cast<cutlass::half_t *>(d_k),
      reinterpret_cast<cutlass::half_t *>(d_v),
      reinterpret_cast<cutlass::half_t *>(d_o), kSeqlen, kSeqlen);
  check(cudaGetLastError(), "cute fa2 cpasync launch");
  check(cudaDeviceSynchronize(), "cute fa2 cpasync sync");
  check(cudaMemcpy(h_o, d_o, kCount * sizeof(half), cudaMemcpyDeviceToHost),
        "cute fa2 cpasync D2H");

  float max_err = 0.0f;
  for (int idx = 0; idx < kCount; ++idx) {
    max_err = max(max_err, fabsf(__half2float(h_o[idx]) - ref_o[idx]));
  }
  char label[96];
  snprintf(label, sizeof(label),
           "FA2 CuTe MMA Stages (Sk=%d, F32Acc, D=%d)", kStagesK, kHeadDim);
  printf("| %-56s | %.3e |\n", label, max_err);

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
// FA2-style CuTe test: single consumer, Br=128, no merge.
// kSeqlen=256 -> 2 Q tiles, covers multi-Q-tile path.
template <int kHeadDim>
static void test_flash_attn_tma_mma_ws_split_q_cute() {
  using namespace cute;
  using Traits = fa_cute::FlashAttn2CuTeTraits<kHeadDim>;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  constexpr int kBr = 128;
  constexpr int kSeqlen = 256;
  constexpr int kCount = kSeqlen * kHeadDim;

  half *h_q = (half *)malloc(kCount * sizeof(half));
  half *h_k = (half *)malloc(kCount * sizeof(half));
  half *h_v = (half *)malloc(kCount * sizeof(half));
  half *h_o = (half *)malloc(kCount * sizeof(half));
  float *ref_o = (float *)malloc(kCount * sizeof(float));
  srand(42 + kHeadDim);
  for (int idx = 0; idx < kCount; ++idx) {
    h_q[idx] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    h_k[idx] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    h_v[idx] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  }

  float scale = 1.0f / sqrtf((float)kHeadDim);
  for (int row = 0; row < kSeqlen; ++row) {
    float scores[kSeqlen];
    float row_max = -INFINITY;
    for (int col = 0; col < kSeqlen; ++col) {
      float score = 0.0f;
      for (int dim = 0; dim < kHeadDim; ++dim) {
        score += __half2float(h_q[row * kHeadDim + dim]) *
                 __half2float(h_k[col * kHeadDim + dim]);
      }
      scores[col] = score * scale;
      row_max = max(row_max, scores[col]);
    }
    float row_sum = 0.0f;
    for (int col = 0; col < kSeqlen; ++col) {
      scores[col] = expf(scores[col] - row_max);
      row_sum += scores[col];
    }
    for (int dim = 0; dim < kHeadDim; ++dim) {
      float output = 0.0f;
      for (int col = 0; col < kSeqlen; ++col) {
        output += scores[col] * __half2float(h_v[col * kHeadDim + dim]);
      }
      ref_o[row * kHeadDim + dim] = output / row_sum;
    }
  }

  half *d_q;
  half *d_k;
  half *d_v;
  half *d_o;
  check(cudaMalloc(&d_q, kCount * sizeof(half)), "cute fa2 alloc Q");
  check(cudaMalloc(&d_k, kCount * sizeof(half)), "cute fa2 alloc K");
  check(cudaMalloc(&d_v, kCount * sizeof(half)), "cute fa2 alloc V");
  check(cudaMalloc(&d_o, kCount * sizeof(half)), "cute fa2 alloc O");
  check(cudaMemcpy(d_q, h_q, kCount * sizeof(half), cudaMemcpyHostToDevice),
        "cute fa2 H2D Q");
  check(cudaMemcpy(d_k, h_k, kCount * sizeof(half), cudaMemcpyHostToDevice),
        "cute fa2 H2D K");
  check(cudaMemcpy(d_v, h_v, kCount * sizeof(half), cudaMemcpyHostToDevice),
        "cute fa2 H2D V");

  // Q tile: [128, D], K/V tile: [64, D] (both use GMMA::Layout_K_SW128_Atom)
  auto make_tma_q = [=]() {
    auto tensor = make_tensor(
        make_gmem_ptr(reinterpret_cast<cutlass::half_t *>(d_q)),
        make_shape(kSeqlen, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, _1{}));
    return make_tma_copy(
        SM90_TMA_LOAD{}, tensor, SmemLayoutQ{},
        Shape<_128, Int<kHeadDim>>{}, _1{});
  };
  auto make_tma_kv = [=](half *pointer) {
    auto tensor = make_tensor(
        make_gmem_ptr(reinterpret_cast<cutlass::half_t *>(pointer)),
        make_shape(kSeqlen, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, _1{}));
    return make_tma_copy(
        SM90_TMA_LOAD{}, tensor, SmemLayoutKV{},
        Shape<_64, Int<kHeadDim>>{}, _1{});
  };
  auto tma_q = make_tma_q();
  auto tma_k = make_tma_kv(d_k);
  auto tma_v = make_tma_kv(d_v);
  auto kernel = flash_attn_tma_mma_ws_split_q_cute<
      kHeadDim, decltype(tma_q), decltype(tma_k), decltype(tma_v)>;
  // smem = Q[128*D] + K[Sk*64*D] + V[Sv*64*D] (Sk=1, Sv=1 default)
  constexpr int kSmemBytes =
      (cosize(SmemLayoutQ{}) +
       1 * cosize(SmemLayoutKV{}) +
       1 * cosize(SmemLayoutKV{})) * sizeof(cutlass::half_t);
  check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kSmemBytes),
        "cute fa2 set smem");
  kernel<<<dim3(kSeqlen / kBr, 1), 384, kSmemBytes>>>(
      tma_q, tma_k, tma_v,
      reinterpret_cast<cutlass::half_t *>(d_o), kSeqlen, kSeqlen);
  check(cudaGetLastError(), "cute fa2 launch");
  check(cudaDeviceSynchronize(), "cute fa2 sync");
  check(cudaMemcpy(h_o, d_o, kCount * sizeof(half), cudaMemcpyDeviceToHost),
        "cute fa2 D2H");

  float max_err = 0.0f;
  for (int idx = 0; idx < kCount; ++idx) {
    max_err = max(max_err, fabsf(__half2float(h_o[idx]) - ref_o[idx]));
  }
  char label[80];
  snprintf(label, sizeof(label), "FA2 CuTe TMA MMA WS (1 Consumer WG) (Sk=1, Sv=1, F32Acc, D=%d)", kHeadDim);
  printf("| %-56s | %.3e |\n", label, max_err);

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
// Phase 8: sm_120 persist-D FlashAttention (WS 1P+1C + persistent CTA +
// softmax scale*fused)。参考优先 cuDNN SDPA (GPU 上跑, 支持 GQA Hkv<Hq 与
// bottom-right causal 滑窗, 语义与 kernel 的 k_pos > q_pos+(Nkv-Nq) mask
// 一致); 未编 CUDNN 或 graph 不支持时回退 CPU fp64。覆盖: dense 多 q-tile /
// causal / GQA / 尾部 tile (R->G)。Nq=2048/H=8 时 total_q_tiles=128 > 96 SM,
// 触发 persistent 多 iter 路径 (epi_done wait + kv_cursor 跨 q-tile 累计)。
template <int kHeadDim, int kNq, int kNkv, int kHq = 2, int kHkv = 2,
          bool kCausal = false>
static void test_flash_attn_cute_persist_d_sm120() {
  constexpr int kCount = kNq * kHeadDim;
  constexpr int kCountKV = kNkv * kHeadDim;

  half* h_q = (half*)malloc(kCount * kHq * sizeof(half));
  half* h_k = (half*)malloc(kCountKV * kHkv * sizeof(half));
  half* h_v = (half*)malloc(kCountKV * kHkv * sizeof(half));
  half* h_o = (half*)malloc(kCount * kHq * sizeof(half));
  srand(42 + kHeadDim + (kCausal ? 7 : 0) + kHq * 100);
  for (int i = 0; i < kCount * kHq; ++i)
    h_q[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  for (int i = 0; i < kCountKV * kHkv; ++i) {
    h_k[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    h_v[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  }

  half *d_q, *d_k, *d_v, *d_o;
  check(cudaMalloc(&d_q, kCount * kHq * sizeof(half)), "pd alloc Q");
  check(cudaMalloc(&d_k, kCountKV * kHkv * sizeof(half)), "pd alloc K");
  check(cudaMalloc(&d_v, kCountKV * kHkv * sizeof(half)), "pd alloc V");
  check(cudaMalloc(&d_o, kCount * kHq * sizeof(half)), "pd alloc O");
  check(cudaMemcpy(d_q, h_q, kCount * kHq * sizeof(half),
                   cudaMemcpyHostToDevice),
        "pd H2D Q");
  check(cudaMemcpy(d_k, h_k, kCountKV * kHkv * sizeof(half),
                   cudaMemcpyHostToDevice),
        "pd H2D K");
  check(cudaMemcpy(d_v, h_v, kCountKV * kHkv * sizeof(half),
                   cudaMemcpyHostToDevice),
        "pd H2D V");

  // 参考输出: cuDNN SDPA 优先, 失败回退 CPU fp64
  half* h_o_ref = nullptr;
  double* ref_o = nullptr;
#if defined(NOTES_V2_ENABLE_CUDNN)
  {
    bool cudnn_ok = false;
    half* d_o_ref;
    check(cudaMalloc(&d_o_ref, kCount * kHq * sizeof(half)), "pd alloc O_ref");
    {
      cudnnHandle_t handle;
      cudnnCreate(&handle);
      auto graph = std::make_shared<fe::graph::Graph>();
      graph->set_io_data_type(fe::DataType_t::HALF)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);
      auto Q = graph->tensor(fe::graph::Tensor_attributes()
          .set_uid(1).set_dim({1, kHq, kNq, kHeadDim})
          .set_stride({kHq * kCount, kCount, kHeadDim, 1}));
      auto K = graph->tensor(fe::graph::Tensor_attributes()
          .set_uid(2).set_dim({1, kHkv, kNkv, kHeadDim})
          .set_stride({kHkv * kCountKV, kCountKV, kHeadDim, 1}));
      auto V = graph->tensor(fe::graph::Tensor_attributes()
          .set_uid(3).set_dim({1, kHkv, kNkv, kHeadDim})
          .set_stride({kHkv * kCountKV, kCountKV, kHeadDim, 1}));
      auto sdpa = fe::graph::SDPA_attributes()
          .set_name("pd_ref")
          .set_attn_scale(1.0f / sqrtf((float)kHeadDim));
      if (kCausal) sdpa.set_causal_mask_bottom_right(true);
      auto [O_sdpa, Stats] = graph->sdpa(Q, K, V, sdpa);
      O_sdpa->set_output(true).set_uid(4)
          .set_dim({1, kHq, kNq, kHeadDim})
          .set_stride({kHq * kCount, kCount, kHeadDim, 1});
      auto build_status = graph->build(
          handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
      if (build_status.is_good()) {
        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> vp = {
            {1, d_q}, {2, d_k}, {3, d_v}, {4, d_o_ref}};
        int64_t ws_size = 0;
        if (graph->get_workspace_size(ws_size).is_good()) {
          int8_t* d_ws = nullptr;
          if (ws_size > 0) check(cudaMalloc(&d_ws, ws_size), "pd ws");
          if (graph->execute(handle, vp, d_ws).is_good()) {
            check(cudaDeviceSynchronize(), "pd cudnn sync");
            cudnn_ok = true;
          }
          if (d_ws) cudaFree(d_ws);
        }
      }
      cudnnDestroy(handle);
    }
    if (cudnn_ok) {
      h_o_ref = (half*)malloc(kCount * kHq * sizeof(half));
      check(cudaMemcpy(h_o_ref, d_o_ref, kCount * kHq * sizeof(half),
                       cudaMemcpyDeviceToHost), "pd D2H ref");
    } else {
      fprintf(stderr, "cudnn SDPA ref unavailable, fallback to CPU fp64\n");
    }
    cudaFree(d_o_ref);
  }
#endif

  if (!h_o_ref) {
    // CPU fp64 回退参考: O = softmax(scale * Q K^T [+ causal mask]) V
    ref_o = (double*)malloc(kCount * kHq * sizeof(double));
    const double scale = 1.0 / sqrt((double)kHeadDim);
    const int kv_offset = kNkv - kNq;
    double* S = (double*)malloc(kNkv * sizeof(double));
    for (int h = 0; h < kHq; ++h) {
      const int hk = h / (kHq / kHkv);  // GQA head 映射
      const half* q = h_q + (size_t)h * kCount;
      const half* k = h_k + (size_t)hk * kCountKV;
      const half* v = h_v + (size_t)hk * kCountKV;
      double* o = ref_o + (size_t)h * kCount;
      for (int qi = 0; qi < kNq; ++qi) {
        double smax = -INFINITY;
        for (int kj = 0; kj < kNkv; ++kj) {
          if (kCausal && kj > qi + kv_offset) {
            S[kj] = -INFINITY;
            continue;
          }
          double s = 0.0;
          for (int d = 0; d < kHeadDim; ++d)
            s += (double)__half2float(q[(size_t)qi * kHeadDim + d]) *
                 (double)__half2float(k[(size_t)kj * kHeadDim + d]);
          S[kj] = s * scale;
          if (S[kj] > smax) smax = S[kj];
        }
        double sum_exp = 0.0;
        for (int kj = 0; kj < kNkv; ++kj) {
          S[kj] = exp(S[kj] - smax);  // softmax 权重 (exp(-inf)=0)
          sum_exp += S[kj];
        }
        // 全 mask 行 (causal Nkv<Nq 前段): 与 kernel 一致输出 0 而非 NaN
        if (sum_exp == 0.0) {
          for (int d = 0; d < kHeadDim; ++d)
            o[(size_t)qi * kHeadDim + d] = 0.0;
          continue;
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
  }

  flash_attn_cute_persist_d_sm120_fwd(
      reinterpret_cast<cutlass::half_t*>(d_q),
      reinterpret_cast<cutlass::half_t*>(d_k),
      reinterpret_cast<cutlass::half_t*>(d_v),
      reinterpret_cast<cutlass::half_t*>(d_o),
      /*Nb=*/1, kHq, kHkv, kNq, kNkv, kHeadDim, kCausal,
      /*scale=*/1.0f / sqrtf((float)kHeadDim));
  check(cudaDeviceSynchronize(), "pd sync");
  check(cudaMemcpy(h_o, d_o, kCount * kHq * sizeof(half),
                   cudaMemcpyDeviceToHost),
        "pd D2H");

  // cuDNN 的 bottom-right causal 对全 mask 行 (qi < Nq-Nkv) 输出未定义
  // (softmax(-inf) 可能 NaN), kernel 定义为输出 0, 对照时跳过这些行
  const int full_mask_rows = kCausal ? kNq - kNkv : 0;
  float max_err = 0.0f;
  for (int i = 0; i < kCount * kHq; ++i) {
    const int qi = (i / kHeadDim) % kNq;
    if (qi < full_mask_rows) continue;
    const float ref_val =
        h_o_ref ? __half2float(h_o_ref[i]) : (float)ref_o[i];
    const float err = fabsf(__half2float(h_o[i]) - ref_val);
    if (err > max_err) max_err = err;
  }
  char label[104];
  snprintf(label, sizeof(label),
           "FA persist-D WS persistent CTA scale-fused (D=%d, Nq=%d, Nkv=%d, "
           "Hq=%d, Hkv=%d%s, ref=%s)",
           kHeadDim, kNq, kNkv, kHq, kHkv, kCausal ? ", causal" : "",
           h_o_ref ? "cudnn" : "cpu");
  printf("| %-88s | %.3e |\n", label, max_err);

  free(h_q);
  free(h_k);
  free(h_v);
  free(h_o);
  free(h_o_ref);
  free(ref_o);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_o);
}
#endif

void test_flash_attn(int seqlen, int head_dim) {
  // FlashAttention-2 with split-Q, MMA m16n8k16
  int B = 1, H = 8;

  size_t sz = (size_t)B * H * seqlen * head_dim * sizeof(half);

  srand(42);
  half *h_q = (half *)malloc(sz);
  half *h_k = (half *)malloc(sz);
  half *h_v = (half *)malloc(sz);
  for (int i = 0; i < B * H * seqlen * head_dim; i++) {
    h_q[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    h_k[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    h_v[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  }

  // CPU reference in FP32: O = softmax(Q @ K^T / sqrt(d)) @ V
  float *ref_q = (float *)malloc(sz * 4 / sizeof(half));  // 4x for float
  float *ref_k = (float *)malloc(sz * 4 / sizeof(half));
  float *ref_v = (float *)malloc(sz * 4 / sizeof(half));
  float *ref_o = (float *)malloc(sz * 4 / sizeof(half));
  int count = B * H * seqlen * head_dim;
  for (int i = 0; i < count; i++) {
    ref_q[i] = __half2float(h_q[i]);
    ref_k[i] = __half2float(h_k[i]);
    ref_v[i] = __half2float(h_v[i]);
  }

  float scale = 1.0f / sqrtf((float)head_dim);
  for (int bi = 0; bi < B * H; bi++) {
    for (int qi = 0; qi < seqlen; qi++) {
      // S[qi, kj] = Q[qi,:] @ K[kj,:]^T * scale
      float smax = -INFINITY;
      float *S = (float *)malloc((size_t)seqlen * sizeof(float));
      for (int kj = 0; kj < seqlen; kj++) {
        float s = 0.0f;
        for (int d = 0; d < head_dim; d++)
          s += ref_q[bi * seqlen * head_dim + qi * head_dim + d] *
               ref_k[bi * seqlen * head_dim + kj * head_dim + d];
        S[kj] = s * scale;
        if (S[kj] > smax) smax = S[kj];
      }
      // softmax
      double sum_exp = 0.0;
      for (int kj = 0; kj < seqlen; kj++) sum_exp += (double)expf(S[kj] - smax);
      float inv_sum = 1.0f / (float)sum_exp;
      // O[qi, :] = sum_kj P[qi, kj] * V[kj, :]
      for (int d = 0; d < head_dim; d++) {
        double o_acc = 0.0;
        for (int kj = 0; kj < seqlen; kj++)
          o_acc += (double)(expf(S[kj] - smax) * inv_sum) *
                   ref_v[bi * seqlen * head_dim + kj * head_dim + d];
        ref_o[bi * seqlen * head_dim + qi * head_dim + d] = (float)o_acc;
      }
      free(S);
    }
  }

  half *d_q, *d_k, *d_v, *d_o;
  check(cudaMalloc(&d_q, sz), "fa alloc Q");
  check(cudaMalloc(&d_k, sz), "fa alloc K");
  check(cudaMalloc(&d_v, sz), "fa alloc V");
  check(cudaMalloc(&d_o, sz), "fa alloc O");
  check(cudaMemcpy(d_q, h_q, sz, cudaMemcpyHostToDevice), "fa H2D Q");
  check(cudaMemcpy(d_k, h_k, sz, cudaMemcpyHostToDevice), "fa H2D K");
  check(cudaMemcpy(d_v, h_v, sz, cudaMemcpyHostToDevice), "fa H2D V");

  // Template params for kHeadDim=64, kStagesK=2
  constexpr int kHeadDim = 64;
  constexpr int kStagesK = 2;
  constexpr int kPadQ = 8;
  constexpr int kPadK = 8;
  constexpr int kPadV = 8;
  constexpr int kMmaAtomM = 16;
  constexpr int kMmaAtomN = 8;
  constexpr int kMmaAtomK = 16;
  constexpr int kMmaTileSeqLenQ = 8;
  constexpr int kMmaTileSeqLenK = 1;
  constexpr int kMmaTileSeqLenP = 8;
  constexpr int kMmaTileHeadDimV = 1;
  constexpr int kValTileSeqLenQ = 1;
  constexpr int kValTileSeqLenK = 8;
  constexpr int kValTileSeqLenP = 1;
  constexpr int kValTileHeadDimV = kHeadDim / (8 * kMmaTileHeadDimV);

  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kValTileSeqLenQ;
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kValTileSeqLenK;
  if (seqlen < Br) return; // kernel requires seqlen >= tile size
  size_t smem_bytes =
      (Br * (kHeadDim + kPadQ) +
       kStagesK * Bc * (kHeadDim + kPadK) +
       Bc * (kHeadDim + kPadV)) * sizeof(half);

  dim3 block(256);
  dim3 grid((seqlen + Br - 1) / Br, B * H);

  // Test both accumulator variants: f16 (kMmaAccF32=0) and f32 (kMmaAccF32=1)
  for (int acc = 0; acc <= 1; ++acc) {
    const int kMmaAcc = acc;
    half *h_o = (half *)malloc(sz);

    if (kMmaAcc == 0) {
      using FAKernel = void (*)(half *, half *, half *, half *, int, int);
      FAKernel fa_k = flash_attn_mma_stages_split_q<kHeadDim, kMmaAtomM, kMmaAtomN,
          kMmaAtomK, 0, kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP,
          kMmaTileHeadDimV, kValTileSeqLenQ, kValTileSeqLenK, kValTileSeqLenP,
          kValTileHeadDimV, kStagesK, kPadQ, kPadK, kPadV>;
      cudaFuncSetAttribute(fa_k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
      fa_k<<<grid, block, smem_bytes>>>(d_q, d_k, d_v, d_o, seqlen, H);
    } else {
      using FAKernel = void (*)(half *, half *, half *, half *, int, int);
      FAKernel fa_k = flash_attn_mma_stages_split_q<kHeadDim, kMmaAtomM, kMmaAtomN,
          kMmaAtomK, 1, kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP,
          kMmaTileHeadDimV, kValTileSeqLenQ, kValTileSeqLenK, kValTileSeqLenP,
          kValTileHeadDimV, kStagesK, kPadQ, kPadK, kPadV>;
      cudaFuncSetAttribute(fa_k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
      fa_k<<<grid, block, smem_bytes>>>(d_q, d_k, d_v, d_o, seqlen, H);
    }
    check(cudaGetLastError(), "fa launch");
    check(cudaDeviceSynchronize(), "fa sync");

    check(cudaMemcpy(h_o, d_o, sz, cudaMemcpyDeviceToHost), "fa D2H");

    float max_err = 0.0f;
    for (int i = 0; i < count; i++) {
      float err = fabsf(__half2float(h_o[i]) - ref_o[i]);
      if (err > max_err) max_err = err;
    }
    const char *acc_label = kMmaAcc ? "F32Acc" : "F16Acc";
    char label[64];
    snprintf(label, sizeof(label), "FA2 (kStagesK=2, Pad, %s)", acc_label);
    printf("| %-56s | %.3e |\n", label,
           max_err);
    free(h_o);
  }

  free(h_q); free(h_k); free(h_v);
  free(ref_q); free(ref_k); free(ref_v); free(ref_o);
  cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
}


#if defined(NOTES_V2_ENABLE_TMA_MMA_WS)
// Test for flash_attn_tma_mma_ws_stages_split_q (SM120, D=64/128)
// Reference: cuDNN SDPA (half) if available, else CPU FP32 (float) fallback.
template <int kHeadDim>
static void test_flash_attn_tma_mma_ws_impl(int seqlen, int head_dim) {
  int B = 1, H = 8;
  constexpr int kStagesK = 2;
  constexpr int kStagesV = 1;
  constexpr int kMmaAtomM = 16, kMmaAtomN = 8, kMmaAtomK = 16;
  constexpr int kMmaTileSeqLenQ = 8, kMmaTileSeqLenK = 1;
  constexpr int kMmaTileSeqLenP = 8, kMmaTileHeadDimV = 1;
  constexpr int kValTileSeqLenQ = 1, kValTileSeqLenK = 8;
  constexpr int kValTileSeqLenP = 1;
  constexpr int kValTileHeadDimV = kHeadDim / (8 * kMmaTileHeadDimV);
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kValTileSeqLenQ;  // 128
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kValTileSeqLenK;  // 64
  constexpr int kNumThreads = 384;

  if (seqlen % Br != 0 || seqlen % Bc != 0 || seqlen < Br) {
    printf("| %-56s | %-9s |\n",
           "FA2 TMA MMA WS (1 Consumer WG) (unaligned)", "SKIP");
    return;
  }

  size_t sz = (size_t)B * H * seqlen * head_dim * sizeof(half);
  srand(42);
  half *h_q = (half *)malloc(sz);
  half *h_k = (half *)malloc(sz);
  half *h_v = (half *)malloc(sz);
  for (int i = 0; i < B * H * seqlen * head_dim; ++i) {
    h_q[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    h_k[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    h_v[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  }

  half *d_q, *d_k, *d_v, *d_o;
  check(cudaMalloc(&d_q, sz), "fa_tma_ws alloc Q");
  check(cudaMalloc(&d_k, sz), "fa_tma_ws alloc K");
  check(cudaMalloc(&d_v, sz), "fa_tma_ws alloc V");
  check(cudaMalloc(&d_o, sz), "fa_tma_ws alloc O");
  check(cudaMemcpy(d_q, h_q, sz, cudaMemcpyHostToDevice), "fa_tma_ws H2D Q");
  check(cudaMemcpy(d_k, h_k, sz, cudaMemcpyHostToDevice), "fa_tma_ws H2D K");
  check(cudaMemcpy(d_v, h_v, sz, cudaMemcpyHostToDevice), "fa_tma_ws H2D V");

  // Reference output: either cuDNN (half) or CPU (float)
  half *h_o_ref = nullptr;
  float *ref_o = nullptr;
  int count = B * H * seqlen * head_dim;

#if defined(NOTES_V2_ENABLE_CUDNN)
  {
    // Try cuDNN SDPA first; fall back to CPU if unsupported on this SM
    bool cudnn_ok = false;
    half *d_o_ref;
    check(cudaMalloc(&d_o_ref, sz), "fa_tma_ws alloc O_ref");
    {
      cudnnHandle_t cudnn_handle;
      cudnnCreate(&cudnn_handle);
      auto graph = std::make_shared<fe::graph::Graph>();
      graph->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

      auto Q = graph->tensor(fe::graph::Tensor_attributes()
        .set_uid(1).set_dim({B, H, seqlen, head_dim})
        .set_stride({H * seqlen * head_dim, seqlen * head_dim, head_dim, 1}));
      auto K = graph->tensor(fe::graph::Tensor_attributes()
        .set_uid(2).set_dim({B, H, seqlen, head_dim})
        .set_stride({H * seqlen * head_dim, seqlen * head_dim, head_dim, 1}));
      auto V = graph->tensor(fe::graph::Tensor_attributes()
        .set_uid(3).set_dim({B, H, seqlen, head_dim})
        .set_stride({H * seqlen * head_dim, seqlen * head_dim, head_dim, 1}));

      auto [O_sdpa, Stats] = graph->sdpa(Q, K, V,
        fe::graph::SDPA_attributes()
          .set_name("sdpa_ref")
          .set_attn_scale(1.0f / sqrtf((float)head_dim)));

      O_sdpa->set_output(true).set_uid(4)
        .set_dim({B, H, seqlen, head_dim})
        .set_stride({H * seqlen * head_dim, seqlen * head_dim, head_dim, 1});

      auto build_status = graph->build(cudnn_handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
      if (build_status.is_good()) {
        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> vp = {
          {1, d_q}, {2, d_k}, {3, d_v}, {4, d_o_ref}};
        int64_t ws_size = 0;
        if (graph->get_workspace_size(ws_size).is_good()) {
          int8_t *d_ws = nullptr;
          if (ws_size > 0) check(cudaMalloc(&d_ws, ws_size), "fa_tma_ws workspace");
          if (graph->execute(cudnn_handle, vp, d_ws).is_good()) {
            check(cudaDeviceSynchronize(), "fa_tma_ws cudnn sync");
            cudnn_ok = true;
          }
          if (d_ws) cudaFree(d_ws);
        }
      }
      cudnnDestroy(cudnn_handle);
    }

    if (cudnn_ok) {
      h_o_ref = (half *)malloc(sz);
      check(cudaMemcpy(h_o_ref, d_o_ref, sz, cudaMemcpyDeviceToHost), "fa_tma_ws D2H ref");
    } else {
      fprintf(stderr, "cudnn SDPA unsupported on this SM, using CPU ref\n");
    }
    cudaFree(d_o_ref);
  }
#endif

  // CPU reference fallback
  if (!h_o_ref) {
    float *ref_q = (float *)malloc(sz * 4 / sizeof(half));
    float *ref_k = (float *)malloc(sz * 4 / sizeof(half));
    float *ref_v = (float *)malloc(sz * 4 / sizeof(half));
    ref_o = (float *)malloc(sz * 4 / sizeof(half));
    for (int i = 0; i < count; ++i) {
      ref_q[i] = __half2float(h_q[i]);
      ref_k[i] = __half2float(h_k[i]);
      ref_v[i] = __half2float(h_v[i]);
    }
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int bi = 0; bi < B * H; ++bi) {
      for (int qi = 0; qi < seqlen; ++qi) {
        float smax = -INFINITY;
        float *S = (float *)malloc((size_t)seqlen * sizeof(float));
        for (int kj = 0; kj < seqlen; ++kj) {
          float s = 0.0f;
          for (int d = 0; d < head_dim; ++d)
            s += ref_q[bi * seqlen * head_dim + qi * head_dim + d] *
                 ref_k[bi * seqlen * head_dim + kj * head_dim + d];
          S[kj] = s * scale;
          if (S[kj] > smax) smax = S[kj];
        }
        double sum_exp = 0.0;
        for (int kj = 0; kj < seqlen; ++kj)
          sum_exp += (double)expf(S[kj] - smax);
        float inv_sum = 1.0f / (float)sum_exp;
        for (int d = 0; d < head_dim; ++d) {
          double o_acc = 0.0;
          for (int kj = 0; kj < seqlen; ++kj)
            o_acc += (double)(expf(S[kj] - smax) * inv_sum) *
                     ref_v[bi * seqlen * head_dim + kj * head_dim + d];
          ref_o[bi * seqlen * head_dim + qi * head_dim + d] = (float)o_acc;
        }
        free(S);
      }
    }
    free(ref_q);
    free(ref_k);
    free(ref_v);
  }

  // TMA descriptors: box innermost 固定 64 half (128B)，满足
  // CU_TENSOR_MAP_SWIZZLE_128B 对 box innermost ≤ 128B 的硬约束。
  // D=64: blocks_width=1, 单次 TMA 覆盖整行。
  // D=128: blocks_width=2, kernel producer 沿 head_dim 连续发 2 次 TMA，
  //        minor_coord = 0/64，写入 chunk-major smem 布局 [2, Br, 64]。
  // Q/K/V gmem 是 [B, H, seqlen, head_dim] row-major，作为 2D
  // [B*H*seqlen, head_dim] 矩阵描述。blocks_height = B*H*seqlen/tile_major。
  // kernel 用 (Nb_id*H+Nh_id)*N 偏移 major_coord。
  constexpr int kTmaBoxMinor = 64;  // box innermost = 64 half = 128B
  constexpr int kTmaChunksQ  = kHeadDim / kTmaBoxMinor;
  constexpr int kTmaChunksKV = kHeadDim / kTmaBoxMinor;
  CUtensorMap *tma_q = allocate_and_create_tensor_map<Br, kTmaBoxMinor>(
      d_q, B * H * seqlen / Br, kTmaChunksQ);
  CUtensorMap *tma_k = allocate_and_create_tensor_map<Bc, kTmaBoxMinor>(
      d_k, B * H * seqlen / Bc, kTmaChunksKV);
  CUtensorMap *tma_v = allocate_and_create_tensor_map<Bc, kTmaBoxMinor>(
      d_v, B * H * seqlen / Bc, kTmaChunksKV);

  using FAKernel = void (*)(half *, half *, half *, half *, int, int,
                             const CUtensorMap *, const CUtensorMap *,
                             const CUtensorMap *);
  FAKernel fa_k = flash_attn_tma_mma_ws_stages_split_q<
      kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, 0, kMmaTileSeqLenQ,
      kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kValTileSeqLenQ,
      kValTileSeqLenK, kValTileSeqLenP, kValTileHeadDimV, kStagesK, kStagesV,
      kNumThreads>;

  // smem = Q[Br*d] + K[kStagesK*Bc*d] + V[kStagesV*Bc*d]
  size_t smem_bytes = (Br * kHeadDim + kStagesK * Bc * kHeadDim +
                       kStagesV * Bc * kHeadDim) * sizeof(half);
  int device = 0, max_smem = 0;
  cudaFuncAttributes attributes{};
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         device);
  cudaFuncGetAttributes(&attributes, fa_k);
  bool smem_ok =
      (smem_bytes + attributes.sharedSizeBytes <= (size_t)max_smem);
  if (!smem_ok) {
    if (g_debug)
      printf("| %-56s | %-9s |\n",
             "FA2 TMA MMA WS (1 Consumer WG) (SMEM SKIP)", "SMEM SKIP");
  } else {
    dim3 block(kNumThreads);
    dim3 grid((seqlen + Br - 1) / Br, B * H);
    // Test both accumulator variants
    for (int acc = 0; acc <= 1; ++acc) {
      const int kMmaAcc = acc;
      half *h_o = (half *)malloc(sz);

      if (kMmaAcc == 0) {
        using FAK = void (*)(half *, half *, half *, half *, int, int,
                              const CUtensorMap *, const CUtensorMap *,
                              const CUtensorMap *);
        FAK fk = flash_attn_tma_mma_ws_stages_split_q<
            kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, 0, kMmaTileSeqLenQ,
            kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kValTileSeqLenQ,
            kValTileSeqLenK, kValTileSeqLenP, kValTileHeadDimV, kStagesK,
            kStagesV, kNumThreads>;
        cudaFuncSetAttribute(fk, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes);
        fk<<<grid, block, smem_bytes>>>(d_q, d_k, d_v, d_o, seqlen, H, tma_q,
                                        tma_k, tma_v);
      } else {
        using FAK = void (*)(half *, half *, half *, half *, int, int,
                              const CUtensorMap *, const CUtensorMap *,
                              const CUtensorMap *);
        FAK fk = flash_attn_tma_mma_ws_stages_split_q<
            kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, 1, kMmaTileSeqLenQ,
            kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kValTileSeqLenQ,
            kValTileSeqLenK, kValTileSeqLenP, kValTileHeadDimV, kStagesK,
            kStagesV, kNumThreads>;
        cudaFuncSetAttribute(fk, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes);
        fk<<<grid, block, smem_bytes>>>(d_q, d_k, d_v, d_o, seqlen, H, tma_q,
                                        tma_k, tma_v);
      }
      check(cudaGetLastError(), "fa_tma_ws launch");
      check(cudaDeviceSynchronize(), "fa_tma_ws sync");

      check(cudaMemcpy(h_o, d_o, sz, cudaMemcpyDeviceToHost), "fa_tma_ws D2H");
      float max_err = 0.0f;
      bool checked = h_o_ref || ref_o;
      if (checked) {
        for (int i = 0; i < count; ++i) {
          float ref_val = h_o_ref ? __half2float(h_o_ref[i]) : ref_o[i];
          float err = fabsf(__half2float(h_o[i]) - ref_val);
          if (err > max_err) max_err = err;
        }
      }
      const char *acc_label = kMmaAcc ? "F32Acc" : "F16Acc";
      char label[64];
      snprintf(label, sizeof(label),
               "FA2 TMA MMA WS (1 Consumer WG) (%s)",
               acc_label);
      printf("| %-56s | %.3e |\n",
             label, max_err);
      free(h_o);
    }
  }

  free(h_q); free(h_k); free(h_v);
  free(h_o_ref); free(ref_o);
  cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
  cudaFree(tma_q); cudaFree(tma_k); cudaFree(tma_v);
}

// D=64/128 dispatch wrapper
void test_flash_attn_tma_mma_ws(int seqlen, int head_dim) {
  if (head_dim == 64) {
    test_flash_attn_tma_mma_ws_impl<64>(seqlen, head_dim);
  } else if (head_dim == 128) {
    test_flash_attn_tma_mma_ws_impl<128>(seqlen, head_dim);
  } else {
    printf("| %-56s | %-9s |\n",
           "FA2 TMA MMA WS (1 Consumer WG) (D!=64/128)", "SKIP");
  }
}

// FA3-style dual-consumer correctness test (Br=64, Sk=Sv=2)
// Tests both F16Acc and F32Acc against cuDNN SDPA reference.
#if defined(NOTES_V2_ENABLE_CUDNN)
namespace fe = cudnn_frontend;
static float bench_cudnn_sdpa_tflops(half *d_q, half *d_k, half *d_v,
                                      half *d_o_ref, int B, int H, int seqlen,
                                      int head_dim, fe::DataType_t compute_type);
#endif
template <int kHeadDim, int kStagesK>
static void test_flash_attn_3_tma_ws_impl(int seqlen, int head_dim) {
  int B = 1, H = 8;
  constexpr int kMmaAtomM = 16, kMmaAtomN = 8, kMmaAtomK = 16;
  constexpr int kMmaTileSeqLenQ = 4, kMmaTileSeqLenK = 1;
  constexpr int kMmaTileSeqLenP = 4, kMmaTileHeadDimV = 1;
  constexpr int kValTileSeqLenQ = 1, kValTileSeqLenK = 8;
  constexpr int kValTileSeqLenP = 1;
  constexpr int kValTileHeadDimV = kHeadDim / (8 * kMmaTileHeadDimV);
  constexpr int kStagesV = 1;  // per-WG V: always 1 buffer
  constexpr int kNumConsumerWGs = 2;
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kValTileSeqLenQ;  // 64
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kValTileSeqLenK; // 64
  constexpr int kNumThreads = 384;

  if (seqlen % Br != 0 || seqlen % Bc != 0 || seqlen < Br) {
    printf("| %-56s | %-9s |\n",
           "FA3 TMA MMA WS (2 Consumer WG) (unaligned)", "SKIP");
    return;
  }

  size_t sz = (size_t)B * H * seqlen * head_dim * sizeof(half);
  srand(42);
  half *h_q = (half *)malloc(sz);
  half *h_k = (half *)malloc(sz);
  half *h_v = (half *)malloc(sz);
  for (int i = 0; i < B * H * seqlen * head_dim; ++i) {
    h_q[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    h_k[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    h_v[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  }

  half *d_q, *d_k, *d_v, *d_o;
  check(cudaMalloc(&d_q, sz), "fa3 alloc Q");
  check(cudaMalloc(&d_k, sz), "fa3 alloc K");
  check(cudaMalloc(&d_v, sz), "fa3 alloc V");
  check(cudaMalloc(&d_o, sz), "fa3 alloc O");
  check(cudaMemcpy(d_q, h_q, sz, cudaMemcpyHostToDevice), "fa3 H2D Q");
  check(cudaMemcpy(d_k, h_k, sz, cudaMemcpyHostToDevice), "fa3 H2D K");
  check(cudaMemcpy(d_v, h_v, sz, cudaMemcpyHostToDevice), "fa3 H2D V");

  // Reference: cuDNN SDPA (half output)
  half *h_o_ref = nullptr;
  int count = B * H * seqlen * head_dim;
#if defined(NOTES_V2_ENABLE_CUDNN)
  {
    half *d_o_ref;
    check(cudaMalloc(&d_o_ref, sz), "fa3 alloc O_ref");
    bench_cudnn_sdpa_tflops(d_q, d_k, d_v, d_o_ref, B, H, seqlen, head_dim,
                            fe::DataType_t::FLOAT);
    h_o_ref = (half *)malloc(sz);
    check(cudaMemcpy(h_o_ref, d_o_ref, sz, cudaMemcpyDeviceToHost), "fa3 ref D2H");
    cudaFree(d_o_ref);
  }
#endif

  // TMA descriptors (Br=64 for Q, Bc=64 for K/V)
  constexpr int kTmaBoxMinor = 64;
  constexpr int kTmaChunks = kHeadDim / kTmaBoxMinor;
  CUtensorMap *tma_q = allocate_and_create_tensor_map<Br, kTmaBoxMinor>(
      d_q, B * H * seqlen / Br, kTmaChunks);
  CUtensorMap *tma_k = allocate_and_create_tensor_map<Bc, kTmaBoxMinor>(
      d_k, B * H * seqlen / Bc, kTmaChunks);
  CUtensorMap *tma_v = allocate_and_create_tensor_map<Bc, kTmaBoxMinor>(
      d_v, B * H * seqlen / Bc, kTmaChunks);

  using FAKernel = void (*)(half *, half *, half *, half *, int, int,
                             const CUtensorMap *, const CUtensorMap *,
                             const CUtensorMap *);

  // smem = Q + K[2WGs * Sk * Bc*D] + V[2WGs * 1 * Bc*D]
  size_t smem_bytes = (Br * kHeadDim +
                       kNumConsumerWGs * kStagesK * Bc * kHeadDim +
                       kNumConsumerWGs * kStagesV * Bc * kHeadDim) * sizeof(half);
  dim3 block(kNumThreads);
  dim3 grid((seqlen + Br - 1) / Br, B * H);

  for (int acc = 0; acc <= 1; ++acc) {
    half *h_o = (half *)malloc(sz);
    FAKernel fk;
    if (acc == 0) {
      fk = flash_attn_3_tma_ws_stages_split_q<
          kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, 0, kMmaTileSeqLenQ,
          kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kValTileSeqLenQ,
          kValTileSeqLenK, kValTileSeqLenP, kValTileHeadDimV, kStagesK, kStagesV,
          kNumThreads>;
    } else {
      fk = flash_attn_3_tma_ws_stages_split_q<
          kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, 1, kMmaTileSeqLenQ,
          kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kValTileSeqLenQ,
          kValTileSeqLenK, kValTileSeqLenP, kValTileHeadDimV, kStagesK, kStagesV,
          kNumThreads>;
    }
    bool smem_ok = check_smem_feasible((const void *)fk, smem_bytes);
    if (!smem_ok) {
      const char *acc_label = acc ? "F32Acc" : "F16Acc";
      char label[64];
      snprintf(label, sizeof(label), "FA3 TMA MMA WS (2 Consumer WG) (%s)",
               acc_label);
      if (g_debug)
        printf("| %-56s | %-9s |\n", label, "SMEM too large");
      free(h_o);
      continue;
    }
    cudaFuncSetAttribute(fk, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);
    fk<<<grid, block, smem_bytes>>>(d_q, d_k, d_v, d_o, seqlen, H, tma_q,
                                     tma_k, tma_v);
    check(cudaGetLastError(), "fa3 launch");
    check(cudaDeviceSynchronize(), "fa3 sync");

    check(cudaMemcpy(h_o, d_o, sz, cudaMemcpyDeviceToHost), "fa3 D2H");
    float max_err = 0.0f;
    bool checked = h_o_ref != nullptr;
    if (checked) {
      for (int i = 0; i < count; ++i) {
        float err = fabsf(__half2float(h_o[i]) - __half2float(h_o_ref[i]));
        if (err > max_err) max_err = err;
      }
    }
    const char *acc_label = acc ? "F32Acc" : "F16Acc";
    char label[64];
    snprintf(label, sizeof(label), "FA3 TMA MMA WS (2 Consumer WG) (%s)",
             acc_label);
    printf("| %-56s | %.3e |\n", label, max_err);
    free(h_o);
  }

  free(h_q); free(h_k); free(h_v); free(h_o_ref);
  cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
  cudaFree(tma_q); cudaFree(tma_k); cudaFree(tma_v);
}

void test_flash_attn_3_tma_ws(int seqlen, int head_dim) {
  // Try stages up to 4; test_flash_attn_3_tma_ws_impl internally checks
  // cudaDevAttrMaxSharedMemoryPerBlockOptin and skips oversized configs.
  if (head_dim == 64) {
    test_flash_attn_3_tma_ws_impl<64, 1>(seqlen, head_dim);
    test_flash_attn_3_tma_ws_impl<64, 2>(seqlen, head_dim);
    test_flash_attn_3_tma_ws_impl<64, 3>(seqlen, head_dim);
    test_flash_attn_3_tma_ws_impl<64, 4>(seqlen, head_dim);
  } else if (head_dim == 128) {
    test_flash_attn_3_tma_ws_impl<128, 1>(seqlen, head_dim);
    test_flash_attn_3_tma_ws_impl<128, 2>(seqlen, head_dim);
    test_flash_attn_3_tma_ws_impl<128, 3>(seqlen, head_dim);
    test_flash_attn_3_tma_ws_impl<128, 4>(seqlen, head_dim);
  } else {
    printf("| %-56s | %-9s |\n",
           "FA3 TMA MMA WS (2 Consumer WG) (D!=64/128)", "SKIP");
  }
}
#endif /* NOTES_V2_ENABLE_TMA_MMA_WS */

// =============================================================================
// Bench: FlashAttention-2 Split-Q (template on kHeadDim for dispatch)
// =============================================================================
template <int kHeadDim, int kStagesK = 2, int kPadQ = 8, int kPadK = 8,
          int kPadV = 8, int kMmaAccF32 = 0>
static void bench_fa_launch(int B, int H, int seqlen, int head_dim,
    half *h_o_ref, float *ref_o, half *d_q, half *d_k, half *d_v, half *d_o,
    float cudnn_tflops_f16) {
  static_assert(kHeadDim == 64 || kHeadDim == 128, "Only D=64 and D=128 are supported");
  constexpr bool kSwizzleQ = kPadQ == 0;
  constexpr bool kSwizzleK = kPadK == 0;
  constexpr bool kSwizzleV = kPadV == 0;
  constexpr int kMmaAtomM = 16;
  constexpr int kMmaAtomN = 8;
  constexpr int kMmaAtomK = 16;
  constexpr int kMmaTileSeqLenQ = 8;
  constexpr int kMmaTileSeqLenK = 1;
  constexpr int kMmaTileSeqLenP = 8;
  constexpr int kMmaTileHeadDimV = 1;
  constexpr int kValTileSeqLenQ = 1;
  constexpr int kValTileSeqLenK = 8;
  constexpr int kValTileSeqLenP = 1;
  constexpr int kValTileHeadDimV = kHeadDim / (8 * kMmaTileHeadDimV);
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kValTileSeqLenQ;
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kValTileSeqLenK;
  constexpr const char *layout_name =
      kSwizzleQ && kSwizzleK && kSwizzleV ? "Swizzle" :
      kSwizzleQ && kSwizzleK ? "SwizzleQK" :
      kSwizzleQ && kSwizzleV ? "SwizzleQV" :
      kSwizzleK && kSwizzleV ? "SwizzleKV" :
      kSwizzleQ ? "SwizzleQ" :
      kSwizzleK ? "SwizzleK" :
      kSwizzleV ? "SwizzleV" : "Pad";

  // Kernel requires seqlen >= Br (tile size); skip gracefully for short seqlen
  if (seqlen < Br) {
    char label[64];
    snprintf(label, sizeof(label), "FA2 (%s)",
             layout_name);
    printf("| %-56s | %-9s | %-19s |\n", label,
           "seqlen<Br", "None");
    return;
  }
  size_t smem_bytes =
      (Br * (kHeadDim + kPadQ) +
       kStagesK * Bc * (kHeadDim + kPadK) +
       Bc * (kHeadDim + kPadV)) * sizeof(half);

  dim3 block(256);
  dim3 grid((seqlen + Br - 1) / Br, B * H);

  cudaStream_t timing_stream;
  cudaStreamCreate(&timing_stream);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  using FAKernel = void (*)(half *, half *, half *, half *, int, int);
  FAKernel fa_k = flash_attn_mma_stages_split_q<
    kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaAccF32,
    kMmaTileSeqLenQ, kMmaTileSeqLenK,
    kMmaTileSeqLenP, kMmaTileHeadDimV, kValTileSeqLenQ, kValTileSeqLenK,
    kValTileSeqLenP, kValTileHeadDimV, kStagesK, kPadQ, kPadK, kPadV>;
  bool smem_ok = check_smem_feasible((const void *)fa_k, smem_bytes);
  if (smem_ok) {
    cudaFuncSetAttribute(fa_k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  }

  // Warmup 
  if (smem_ok) {
    for (int w = 0; w < g_warmup; w++)
      fa_k<<<grid, block, smem_bytes, timing_stream>>>(d_q, d_k, d_v, d_o, seqlen, H);
    check(cudaStreamSynchronize(timing_stream), "fa warmup sync");
  }

  // Timed repeat
  cudaEventRecord(start, timing_stream);
  if (smem_ok) {
    for (int r = 0; r < g_repeat; r++)
      fa_k<<<grid, block, smem_bytes, timing_stream>>>(d_q, d_k, d_v, d_o, seqlen, H);
  }
  cudaEventRecord(stop, timing_stream);
  cudaEventSynchronize(stop);

  float time_ms = 0;
  cudaEventElapsedTime(&time_ms, start, stop);
  time_ms /= g_repeat;

  size_t sz = (size_t)B * H * seqlen * head_dim * sizeof(half);
  half *h_o = (half *)malloc(sz);
  check(cudaMemcpy(h_o, d_o, sz, cudaMemcpyDeviceToHost), "bench fa D2H");

  int count = B * H * seqlen * head_dim;
  char label[64];
  snprintf(label, sizeof(label), "FA2 MMA Stages (Sk=%d, %s, %s)", kStagesK,
           layout_name, kMmaAccF32 ? "F32Acc" : "F16Acc");
  float max_err = 0.0f;
  bool checked = h_o_ref || ref_o;
  if (smem_ok && checked) {
    for (int i = 0; i < count; i++) {
      float ref_val = h_o_ref ? __half2float(h_o_ref[i]) : ref_o[i];
      float err = fabsf(__half2float(h_o[i]) - ref_val);
      if (err > max_err) max_err = err;
    }
  }
  if (smem_ok && checked) {
    float tflops = bench_fa_tflops(B, H, seqlen, head_dim, time_ms);
    bool is_fail = max_err >= 5e-1f;
    if (is_fail || should_print_fa_tflops(kMmaAccF32, tflops)) {
      char tflops_str[32];
      if (cudnn_tflops_f16 > 0)
        snprintf(tflops_str, sizeof(tflops_str), "%.1f/%.1f (%.2fx)", 
                 tflops, cudnn_tflops_f16, tflops / cudnn_tflops_f16);
      else
        snprintf(tflops_str, sizeof(tflops_str), "%.1f", tflops);
      printf("| %-56s | %.3e | %-19s |\n", label, max_err,
             tflops_str);
    }
  } else if (smem_ok) {
    float tflops = bench_fa_tflops(B, H, seqlen, head_dim, time_ms);
    if (should_print_fa_tflops(kMmaAccF32, tflops)) {
      char tflops_str[32];
      snprintf(tflops_str, sizeof(tflops_str), "%.1f", tflops);
      printf("| %-56s | %-9s | %-19s |\n", label, "unchecked", tflops_str);
    }
  } else {
    if (g_debug)
      printf("| %-56s | %-9s | %-19s |\n", label, "SMEM too large", "None");
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(timing_stream);
  free(h_o);
}

#if defined(NOTES_V2_ENABLE_CUTE)
// FA2-style CuTe cp.async bench: single consumer, Br=128, no TMA/WS.
// V 单 buffer, V 在 QK 之前发起 cp.async, 通过 QK+softmax 隐藏 V 延迟.
template <int kHeadDim, int kStagesK>
static void bench_fa_2_mma_stages_cute_launch(
    int B, int H, int seqlen, half *h_o_ref, float *ref_o,
    half *d_q, half *d_k, half *d_v, half *d_o,
    float cudnn_tflops_f32) {
  using namespace cute;
  using Traits = fa_cute::FlashAttn2CuTeTraits<kHeadDim>;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  constexpr int kBr = 128;
  if (seqlen < kBr || seqlen % kBr != 0 || seqlen % 64 != 0) {
    char label[96];
    snprintf(label, sizeof(label),
             "FA2 CuTe MMA Stages (Sk=%d, unaligned)", kStagesK);
    printf("| %-56s | %-9s | %-19s |\n", label, "SKIP", "None");
    return;
  }

  int rows = B * H * seqlen;
  auto kernel = flash_attn_mma_stages_split_q_cute<kHeadDim, kStagesK>;
  int smem_bytes = (size(SmemLayoutQ{}) +
                    kStagesK * size(SmemLayoutKV{}) +
                    1 * size(SmemLayoutKV{})) *
                   sizeof(cutlass::half_t);
  bool smem_ok = check_smem_feasible((const void *)kernel, smem_bytes);
  if (!smem_ok) {
    char label[96];
    snprintf(label, sizeof(label),
             "FA2 CuTe MMA Stages (Sk=%d, SMEM)", kStagesK);
    if (g_debug)
      printf("| %-56s | %-9s | %-19s |\n", label, "SMEM too large", "None");
    return;
  }
  check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes),
        "bench cute fa2 cpasync set smem");

  dim3 grid(seqlen / kBr, B * H);
  cudaStream_t stream;
  cudaEvent_t start;
  cudaEvent_t stop;
  check(cudaStreamCreate(&stream), "bench cute fa2 cpasync stream");
  check(cudaEventCreate(&start), "bench cute fa2 cpasync event start");
  check(cudaEventCreate(&stop), "bench cute fa2 cpasync event stop");
  for (int warmup = 0; warmup < g_warmup; ++warmup) {
    kernel<<<grid, 256, smem_bytes, stream>>>(
        reinterpret_cast<cutlass::half_t *>(d_q),
        reinterpret_cast<cutlass::half_t *>(d_k),
        reinterpret_cast<cutlass::half_t *>(d_v),
        reinterpret_cast<cutlass::half_t *>(d_o), rows, seqlen);
  }
  check(cudaStreamSynchronize(stream), "bench cute fa2 cpasync warmup sync");
  check(cudaEventRecord(start, stream), "bench cute fa2 cpasync record start");
  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    kernel<<<grid, 256, smem_bytes, stream>>>(
        reinterpret_cast<cutlass::half_t *>(d_q),
        reinterpret_cast<cutlass::half_t *>(d_k),
        reinterpret_cast<cutlass::half_t *>(d_v),
        reinterpret_cast<cutlass::half_t *>(d_o), rows, seqlen);
  }
  check(cudaEventRecord(stop, stream), "bench cute fa2 cpasync record stop");
  check(cudaEventSynchronize(stop), "bench cute fa2 cpasync timing sync");
  float time_ms = 0.0f;
  check(cudaEventElapsedTime(&time_ms, start, stop),
        "bench cute fa2 cpasync elapsed");
  time_ms /= g_repeat;

  size_t count = (size_t)rows * kHeadDim;
  half *h_o = (half *)malloc(count * sizeof(half));
  check(cudaMemcpy(h_o, d_o, count * sizeof(half), cudaMemcpyDeviceToHost),
        "bench cute fa2 cpasync D2H");
  float max_err = 0.0f;
  bool checked = h_o_ref || ref_o;
  if (checked) {
    for (size_t idx = 0; idx < count; ++idx) {
      float reference = h_o_ref ? __half2float(h_o_ref[idx]) : ref_o[idx];
      max_err = max(max_err, fabsf(__half2float(h_o[idx]) - reference));
    }
  }
  float tflops = bench_fa_tflops(B, H, seqlen, kHeadDim, time_ms);
  char label[96];
  snprintf(label, sizeof(label),
           "FA2 CuTe MMA Stages (Sk=%d, F32Acc)", kStagesK);
  bool is_fail = checked && max_err >= 5e-1f;
  if (is_fail || should_print_fa_tflops(1, tflops)) {
    char performance[32];
    if (cudnn_tflops_f32 > 0.0f) {
      snprintf(performance, sizeof(performance), "%.1f/%.1f (%.2fx)",
               tflops, cudnn_tflops_f32, tflops / cudnn_tflops_f32);
    } else {
      snprintf(performance, sizeof(performance), "%.1f", tflops);
    }
    printf("| %-56s | %.3e | %-19s |\n", label, max_err, performance);
  }

  free(h_o);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
}

static void bench_fa_2_mma_stages_cute_dispatch(
    int B, int H, int seqlen, int head_dim,
    half *h_o_ref, float *ref_o,
    half *d_q, half *d_k, half *d_v, half *d_o,
    float cudnn_tflops_f32) {
  if (head_dim == 64) {
    bench_fa_2_mma_stages_cute_launch<64, 1>(B, H, seqlen, h_o_ref, ref_o,
        d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    bench_fa_2_mma_stages_cute_launch<64, 2>(B, H, seqlen, h_o_ref, ref_o,
        d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    bench_fa_2_mma_stages_cute_launch<64, 3>(B, H, seqlen, h_o_ref, ref_o,
        d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    bench_fa_2_mma_stages_cute_launch<64, 4>(B, H, seqlen, h_o_ref, ref_o,
        d_q, d_k, d_v, d_o, cudnn_tflops_f32);
  } else if (head_dim == 128) {
    bench_fa_2_mma_stages_cute_launch<128, 1>(B, H, seqlen, h_o_ref, ref_o,
        d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    bench_fa_2_mma_stages_cute_launch<128, 2>(B, H, seqlen, h_o_ref, ref_o,
        d_q, d_k, d_v, d_o, cudnn_tflops_f32);
  }
}
#endif // NOTES_V2_ENABLE_CUTE

#if defined(NOTES_V2_ENABLE_TMA_MMA_WS)
// Bench for flash_attn_tma_mma_ws_stages_split_q (SM120, D=64/128)
template <int kHeadDim, int kStagesK, int kStagesV = 1, int kMmaAccF32 = 0>
static void bench_fa_tma_mma_ws_launch(int B, int H, int seqlen, int head_dim,
                                       half *h_o_ref, float *ref_o,
                                       half *d_q, half *d_k, half *d_v,
                                       half *d_o, float cudnn_tflops_f16) {
  constexpr int kMmaAtomM = 16, kMmaAtomN = 8, kMmaAtomK = 16;
  constexpr int kMmaTileSeqLenQ = 8, kMmaTileSeqLenK = 1;
  constexpr int kMmaTileSeqLenP = 8, kMmaTileHeadDimV = 1;
  constexpr int kValTileSeqLenQ = 1, kValTileSeqLenK = 8;
  constexpr int kValTileSeqLenP = 1;
  constexpr int kValTileHeadDimV = kHeadDim / (8 * kMmaTileHeadDimV);
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kValTileSeqLenQ;
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kValTileSeqLenK;
  constexpr int kNumThreads = 384;

  if (seqlen < Br || seqlen % Br != 0 || seqlen % Bc != 0) {
    char label[64];
    snprintf(label, sizeof(label),
             "FA2 TMA MMA WS (1 Consumer WG) (Sk=%d, Sv=%d, unaligned)", kStagesK, kStagesV);
    printf("| %-56s | %-9s | %-19s |\n", label, "SKIP", "None");
    return;
  }

  // TMA descriptors: box innermost 固定 64 half (128B)，满足
  // CU_TENSOR_MAP_SWIZZLE_128B 对 box innermost ≤ 128B 的硬约束。
  // D=64: blocks_width=1, 单次 TMA 覆盖整行。
  // D=128: blocks_width=2, kernel producer 沿 head_dim 连续发 2 次 TMA，
  //        minor_coord = 0/64，写入 chunk-major smem 布局 [2, Br, 64]。
  // Q/K/V gmem 是 [B, H, seqlen, head_dim] row-major，作为 2D
  // [B*H*seqlen, head_dim] 矩阵描述。blocks_height = B*H*seqlen/tile_major。
  // kernel 用 (Nb_id*H+Nh_id)*N 偏移 major_coord。
  constexpr int kTmaBoxMinor = 64;  // box innermost = 64 half = 128B
  constexpr int kTmaChunks = kHeadDim / kTmaBoxMinor;
  CUtensorMap *tma_q = allocate_and_create_tensor_map<Br, kTmaBoxMinor>(
      d_q, B * H * seqlen / Br, kTmaChunks);
  CUtensorMap *tma_k = allocate_and_create_tensor_map<Bc, kTmaBoxMinor>(
      d_k, B * H * seqlen / Bc, kTmaChunks);
  CUtensorMap *tma_v = allocate_and_create_tensor_map<Bc, kTmaBoxMinor>(
      d_v, B * H * seqlen / Bc, kTmaChunks);

  using FAKernel = void (*)(half *, half *, half *, half *, int, int,
                             const CUtensorMap *, const CUtensorMap *,
                             const CUtensorMap *);
  FAKernel fa_k = flash_attn_tma_mma_ws_stages_split_q<
      kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaAccF32, kMmaTileSeqLenQ,
      kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kValTileSeqLenQ,
      kValTileSeqLenK, kValTileSeqLenP, kValTileHeadDimV, kStagesK, kStagesV,
      kNumThreads>;

  size_t smem_bytes = (Br * kHeadDim + kStagesK * Bc * kHeadDim +
                       kStagesV * Bc * kHeadDim) * sizeof(half);
  bool smem_ok = check_smem_feasible((const void *)fa_k, smem_bytes);
  if (smem_ok) {
    cudaFuncSetAttribute(fa_k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);
  }

  dim3 block(kNumThreads);
  dim3 grid((seqlen + Br - 1) / Br, B * H);

  cudaStream_t timing_stream;
  cudaStreamCreate(&timing_stream);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  if (smem_ok) {
    for (int w = 0; w < g_warmup; ++w)
      fa_k<<<grid, block, smem_bytes, timing_stream>>>(d_q, d_k, d_v, d_o,
                                                       seqlen, H, tma_q, tma_k,
                                                       tma_v);
    check(cudaStreamSynchronize(timing_stream), "fa_tma_ws warmup sync");
  }
  cudaEventRecord(start, timing_stream);
  if (smem_ok) {
    for (int r = 0; r < g_repeat; ++r)
      fa_k<<<grid, block, smem_bytes, timing_stream>>>(d_q, d_k, d_v, d_o,
                                                       seqlen, H, tma_q, tma_k,
                                                       tma_v);
  }
  cudaEventRecord(stop, timing_stream);
  cudaEventSynchronize(stop);

  float time_ms = 0;
  cudaEventElapsedTime(&time_ms, start, stop);
  time_ms /= g_repeat;

  size_t sz = (size_t)B * H * seqlen * head_dim * sizeof(half);
  half *h_o = (half *)malloc(sz);
  check(cudaMemcpy(h_o, d_o, sz, cudaMemcpyDeviceToHost), "fa_tma_ws D2H");
  int count = B * H * seqlen * head_dim;

  char label[64];
  snprintf(label, sizeof(label), "FA2 TMA MMA WS (1 Consumer WG) (Sk=%d, Sv=%d, %s)",
           kStagesK, kStagesV, kMmaAccF32 ? "F32Acc" : "F16Acc");
  float max_err = 0.0f;
  bool checked = h_o_ref || ref_o;
  if (smem_ok && checked) {
    for (int i = 0; i < count; ++i) {
      float ref_val = h_o_ref ? __half2float(h_o_ref[i]) : ref_o[i];
      float err = fabsf(__half2float(h_o[i]) - ref_val);
      if (err > max_err) max_err = err;
    }
  }
  if (smem_ok && checked) {
    float tflops = bench_fa_tflops(B, H, seqlen, head_dim, time_ms);
    bool is_fail = max_err >= 5e-1f;
    if (is_fail || should_print_fa_tflops(kMmaAccF32, tflops)) {
      char tflops_str[32];
      if (cudnn_tflops_f16 > 0)
        snprintf(tflops_str, sizeof(tflops_str), "%.1f/%.1f (%.2fx)", 
                 tflops, cudnn_tflops_f16, tflops / cudnn_tflops_f16);
      else
        snprintf(tflops_str, sizeof(tflops_str), "%.1f", tflops);
      printf("| %-56s | %.3e | %-19s |\n", label, max_err,
             tflops_str);
    }
  } else if (smem_ok) {
    float tflops = bench_fa_tflops(B, H, seqlen, head_dim, time_ms);
    if (should_print_fa_tflops(kMmaAccF32, tflops)) {
      char tflops_str[32];
      snprintf(tflops_str, sizeof(tflops_str), "%.1f", tflops);
      printf("| %-56s | %-9s | %-19s |\n", label, "unchecked",
             tflops_str);
    }
  } else {
    if (g_debug)
      printf("| %-56s | %-9s | %-19s |\n", label, "SMEM too large",
             "None");
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(timing_stream);
  free(h_o);
  cudaFree(tma_q);
  cudaFree(tma_k);
  cudaFree(tma_v);
}

// D=64/128 dispatch wrapper for bench
template <int kStagesK, int kStagesV = 1, int kMmaAccF32 = 0>
static void bench_fa_tma_mma_ws_dispatch(int B, int H, int seqlen, int head_dim,
                                         half *h_o_ref, float *ref_o,
                                         half *d_q, half *d_k, half *d_v,
                                         half *d_o, float cudnn_tflops_f16) {
  if (head_dim == 64) {
    bench_fa_tma_mma_ws_launch<64, kStagesK, kStagesV, kMmaAccF32>(
      B, H, seqlen, head_dim, h_o_ref, ref_o, d_q, d_k, d_v, d_o, cudnn_tflops_f16);
  } else if (head_dim == 128) {
    bench_fa_tma_mma_ws_launch<128, kStagesK, kStagesV, kMmaAccF32>(
      B, H, seqlen, head_dim, h_o_ref, ref_o, d_q, d_k, d_v, d_o, cudnn_tflops_f16);
  } else {
    char label[64];
    snprintf(label, sizeof(label),
             "FA2 TMA MMA WS (1 Consumer WG) (Sk=%d, Sv=%d, D!=64/128)", kStagesK, kStagesV);
    printf("| %-56s | %-9s | %-19s |\n", label, "SKIP", "None");
  }
}

// FA3-style dual-consumer bench (Br=64, per-WG independent K pipeline, Sv=1)
template <int kHeadDim, int kStagesK, int kMmaAccF32 = 0>
static void bench_fa_3_tma_ws_launch(int B, int H, int seqlen, int head_dim,
                                      half *h_o_ref, float *ref_o,
                                      half *d_q, half *d_k, half *d_v,
                                      half *d_o, float cudnn_tflops_f16) {
  constexpr int kMmaAtomM = 16, kMmaAtomN = 8, kMmaAtomK = 16;
  constexpr int kMmaTileSeqLenQ = 4, kMmaTileSeqLenK = 1;
  constexpr int kMmaTileSeqLenP = 4, kMmaTileHeadDimV = 1;
  constexpr int kValTileSeqLenQ = 1, kValTileSeqLenK = 8;
  constexpr int kValTileSeqLenP = 1;
  constexpr int kValTileHeadDimV = kHeadDim / (8 * kMmaTileHeadDimV);
  constexpr int kStagesV = 1;  // per-WG V: always 1 buffer
  constexpr int kNumConsumerWGs = 2;
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kValTileSeqLenQ;  // 64
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kValTileSeqLenK;  // 64
  constexpr int kNumThreads = 384;

  if (seqlen < Br || seqlen % Br != 0 || seqlen % Bc != 0) {
    char label[80];
    snprintf(label, sizeof(label),
             "FA3 TMA MMA WS (2 Consumer WG) (Sk=%d, Sv=1, unaligned)", kStagesK);
    printf("| %-56s | %-9s | %-19s |\n", label, "SKIP", "None");
    return;
  }

  constexpr int kTmaBoxMinor = 64;
  constexpr int kTmaChunks = kHeadDim / kTmaBoxMinor;
  CUtensorMap *tma_q = allocate_and_create_tensor_map<Br, kTmaBoxMinor>(
      d_q, B * H * seqlen / Br, kTmaChunks);
  CUtensorMap *tma_k = allocate_and_create_tensor_map<Bc, kTmaBoxMinor>(
      d_k, B * H * seqlen / Bc, kTmaChunks);
  CUtensorMap *tma_v = allocate_and_create_tensor_map<Bc, kTmaBoxMinor>(
      d_v, B * H * seqlen / Bc, kTmaChunks);

  using FAKernel = void (*)(half *, half *, half *, half *, int, int,
                             const CUtensorMap *, const CUtensorMap *,
                             const CUtensorMap *);
  FAKernel fa_k = flash_attn_3_tma_ws_stages_split_q<
      kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaAccF32, kMmaTileSeqLenQ,
      kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kValTileSeqLenQ,
      kValTileSeqLenK, kValTileSeqLenP, kValTileHeadDimV, kStagesK, kStagesV,
      kNumThreads>;

  // smem = Q + K[2WGs * Sk * Bc*D] + V[2WGs * 1 * Bc*D]
  size_t smem_bytes = (Br * kHeadDim +
                       kNumConsumerWGs * kStagesK * Bc * kHeadDim +
                       kNumConsumerWGs * kStagesV * Bc * kHeadDim) * sizeof(half);
  bool smem_ok = check_smem_feasible((const void *)fa_k, smem_bytes);
  if (smem_ok) {
    cudaFuncSetAttribute(fa_k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);
  }

  dim3 block(kNumThreads);
  dim3 grid((seqlen + Br - 1) / Br, B * H);

  cudaStream_t timing_stream;
  cudaStreamCreate(&timing_stream);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  if (smem_ok) {
    for (int w = 0; w < g_warmup; ++w)
      fa_k<<<grid, block, smem_bytes, timing_stream>>>(d_q, d_k, d_v, d_o,
                                                       seqlen, H, tma_q, tma_k,
                                                       tma_v);
    check(cudaStreamSynchronize(timing_stream), "fa3_tma_ws warmup sync");
  }
  cudaEventRecord(start, timing_stream);
  if (smem_ok) {
    for (int r = 0; r < g_repeat; ++r)
      fa_k<<<grid, block, smem_bytes, timing_stream>>>(d_q, d_k, d_v, d_o,
                                                       seqlen, H, tma_q, tma_k,
                                                       tma_v);
  }
  cudaEventRecord(stop, timing_stream);
  cudaEventSynchronize(stop);

  float time_ms = 0;
  cudaEventElapsedTime(&time_ms, start, stop);
  time_ms /= g_repeat;

  size_t sz = (size_t)B * H * seqlen * head_dim * sizeof(half);
  half *h_o = (half *)malloc(sz);
  check(cudaMemcpy(h_o, d_o, sz, cudaMemcpyDeviceToHost), "fa3_tma_ws D2H");
  int count = B * H * seqlen * head_dim;

  char label[80];
  snprintf(label, sizeof(label), "FA3 TMA MMA WS (2 Consumer WG) (Sk=%d, Sv=1, %s)",
           kStagesK, kMmaAccF32 ? "F32Acc" : "F16Acc");
  float max_err = 0.0f;
  bool checked = h_o_ref || ref_o;
  if (smem_ok && checked) {
    for (int i = 0; i < count; ++i) {
      float ref_val = h_o_ref ? __half2float(h_o_ref[i]) : ref_o[i];
      float err = fabsf(__half2float(h_o[i]) - ref_val);
      if (err > max_err) max_err = err;
    }
  }
  if (smem_ok && checked) {
    float tflops = bench_fa_tflops(B, H, seqlen, head_dim, time_ms);
    bool is_fail = max_err >= 5e-1f;
    if (is_fail || should_print_fa_tflops(kMmaAccF32, tflops)) {
      char tflops_str[32];
      if (cudnn_tflops_f16 > 0)
        snprintf(tflops_str, sizeof(tflops_str), "%.1f/%.1f (%.2fx)", tflops, 
                 cudnn_tflops_f16, tflops / cudnn_tflops_f16);
      else
        snprintf(tflops_str, sizeof(tflops_str), "%.1f", tflops);
      printf("| %-56s | %.3e | %-19s |\n", label, max_err,
             tflops_str);
    }
  } else if (smem_ok) {
    float tflops = bench_fa_tflops(B, H, seqlen, head_dim, time_ms);
    if (should_print_fa_tflops(kMmaAccF32, tflops)) {
      char tflops_str[32];
      snprintf(tflops_str, sizeof(tflops_str), "%.1f", tflops);
      printf("| %-56s | %-9s | %-19s |\n", label, "unchecked",
             tflops_str);
    }
  } else {
    if (g_debug)
      printf("| %-56s | %-9s | %-19s |\n", label, "SMEM too large",
             "None");
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(timing_stream);
  free(h_o);
  cudaFree(tma_q);
  cudaFree(tma_k);
  cudaFree(tma_v);
}

// D=64/128 dispatch wrapper for FA3-style bench
template <int kMmaAccF32 = 0>
static void bench_fa_3_tma_ws_dispatch(int B, int H, int seqlen, int head_dim,
                                       half *h_o_ref, float *ref_o,
                                       half *d_q, half *d_k, half *d_v,
                                       half *d_o, float cudnn_tflops_f16) {
  // Try stages up to 4; bench_fa_3_tma_ws_launch internally checks
  // cudaDevAttrMaxSharedMemoryPerBlockOptin and skips configs that
  // exceed the actual device limit (e.g. Hopper 224KB vs Blackwell 101KB).
  if (head_dim == 64) {
    bench_fa_3_tma_ws_launch<64, 1, kMmaAccF32>(B, H, seqlen, head_dim, h_o_ref,
                                                ref_o, d_q, d_k, d_v, d_o, cudnn_tflops_f16);
    bench_fa_3_tma_ws_launch<64, 2, kMmaAccF32>(B, H, seqlen, head_dim, h_o_ref,
                                                ref_o, d_q, d_k, d_v, d_o, cudnn_tflops_f16);
    bench_fa_3_tma_ws_launch<64, 3, kMmaAccF32>(B, H, seqlen, head_dim, h_o_ref,
                                                ref_o, d_q, d_k, d_v, d_o, cudnn_tflops_f16);
    bench_fa_3_tma_ws_launch<64, 4, kMmaAccF32>(B, H, seqlen, head_dim, h_o_ref,
                                                ref_o, d_q, d_k, d_v, d_o, cudnn_tflops_f16);
  } else if (head_dim == 128) {
    bench_fa_3_tma_ws_launch<128, 1, kMmaAccF32>(B, H, seqlen, head_dim, h_o_ref,
                                                 ref_o, d_q, d_k, d_v, d_o, cudnn_tflops_f16);
    bench_fa_3_tma_ws_launch<128, 2, kMmaAccF32>(B, H, seqlen, head_dim, h_o_ref,
                                                 ref_o, d_q, d_k, d_v, d_o, cudnn_tflops_f16);
    bench_fa_3_tma_ws_launch<128, 3, kMmaAccF32>(B, H, seqlen, head_dim, h_o_ref,
                                                 ref_o, d_q, d_k, d_v, d_o, cudnn_tflops_f16);
    bench_fa_3_tma_ws_launch<128, 4, kMmaAccF32>(B, H, seqlen, head_dim, h_o_ref,
                                                 ref_o, d_q, d_k, d_v, d_o, cudnn_tflops_f16);
  } else {
    char label[64];
    snprintf(label, sizeof(label), "FA3 TMA MMA WS (2 Consumer WG) (D!=64/128)");
    printf("| %-56s | %-9s | %-19s |\n", label, "SKIP", "None");
  }
}

#if defined(NOTES_V2_ENABLE_CUTE)
template <int kHeadDim, int kStagesK = 1>
static void bench_fa_3_tma_mma_ws_cute_launch(
    int B, int H, int seqlen, half *h_o_ref, float *ref_o,
    half *d_q, half *d_k, half *d_v, half *d_o,
    float cudnn_tflops_f32) {
  using namespace cute;
  using Traits = fa_cute::FlashAttn3CuTeTraits<kHeadDim>;
  using SmemLayout = typename Traits::SmemLayoutQKV;
  constexpr int kNumConsumers = 2;
  if (seqlen < 64 || seqlen % 64 != 0) {
    char label[80];
    snprintf(label, sizeof(label),
             "FA3 CuTe TMA MMA WS (2 Consumer WG) (Sk=%d, Sv=1, unaligned)", kStagesK);
    printf("| %-56s | %-9s | %-19s |\n", label, "SKIP", "None");
    return;
  }

  int rows = B * H * seqlen;
  auto make_tma = [=](half *pointer) {
    auto tensor = make_tensor(
        make_gmem_ptr(reinterpret_cast<cutlass::half_t *>(pointer)),
        make_shape(rows, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, _1{}));
    return make_tma_copy(
        SM90_TMA_LOAD{}, tensor, SmemLayout{},
        Shape<_64, Int<kHeadDim>>{}, _1{});
  };
  auto tma_q = make_tma(d_q);
  auto tma_k = make_tma(d_k);
  auto tma_v = make_tma(d_v);
  auto kernel = flash_attn_3_tma_mma_ws_split_q_cute<
      kHeadDim, decltype(tma_q), decltype(tma_k), decltype(tma_v),
      kStagesK>;
  auto acc_o = partition_fragment_C(
      typename Traits::TiledMma{}, Shape<_64, Int<kHeadDim>>{});
  constexpr int kTiles = 1 + kNumConsumers * kStagesK + kNumConsumers;
  constexpr int kTilesBytes = kTiles * cosize(SmemLayout{}) * sizeof(cutlass::half_t);
  int merge_bytes = 128 * size(acc_o) * sizeof(float) + 128 * sizeof(float4);
  int smem_bytes = max(kTilesBytes, merge_bytes);
  bool smem_ok = check_smem_feasible((const void *)kernel, smem_bytes);
  if (!smem_ok) {
    char label[80];
    snprintf(label, sizeof(label),
             "FA3 CuTe TMA MMA WS (2 Consumer WG) (Sk=%d, Sv=1, SMEM)", kStagesK);
    if (g_debug)
      printf("| %-56s | %-9s | %-19s |\n", label, "SMEM too large", "None");
    return;
  }
  check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes),
        "bench cute fa3 set smem");

  dim3 grid(seqlen / 64, B * H);
  cudaStream_t stream;
  cudaEvent_t start;
  cudaEvent_t stop;
  check(cudaStreamCreate(&stream), "bench cute fa3 stream");
  check(cudaEventCreate(&start), "bench cute fa3 event start");
  check(cudaEventCreate(&stop), "bench cute fa3 event stop");
  for (int warmup = 0; warmup < g_warmup; ++warmup) {
    kernel<<<grid, 384, smem_bytes, stream>>>(
        tma_q, tma_k, tma_v,
        reinterpret_cast<cutlass::half_t *>(d_o), rows, seqlen);
  }
  check(cudaStreamSynchronize(stream), "bench cute fa3 warmup sync");
  check(cudaEventRecord(start, stream), "bench cute fa3 record start");
  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    kernel<<<grid, 384, smem_bytes, stream>>>(
        tma_q, tma_k, tma_v,
        reinterpret_cast<cutlass::half_t *>(d_o), rows, seqlen);
  }
  check(cudaEventRecord(stop, stream), "bench cute fa3 record stop");
  check(cudaEventSynchronize(stop), "bench cute fa3 timing sync");
  float time_ms = 0.0f;
  check(cudaEventElapsedTime(&time_ms, start, stop), "bench cute fa3 elapsed");
  time_ms /= g_repeat;

  size_t count = (size_t)rows * kHeadDim;
  half *h_o = (half *)malloc(count * sizeof(half));
  check(cudaMemcpy(h_o, d_o, count * sizeof(half), cudaMemcpyDeviceToHost),
        "bench cute fa3 D2H");
  float max_err = 0.0f;
  bool checked = h_o_ref || ref_o;
  if (checked) {
    for (size_t idx = 0; idx < count; ++idx) {
      float reference = h_o_ref ? __half2float(h_o_ref[idx]) : ref_o[idx];
      max_err = max(max_err, fabsf(__half2float(h_o[idx]) - reference));
    }
  }
  float tflops = bench_fa_tflops(B, H, seqlen, kHeadDim, time_ms);
  char label[80];
  snprintf(label, sizeof(label),
           "FA3 CuTe TMA MMA WS (2 Consumer WG) (Sk=%d, Sv=1, F32Acc)", kStagesK);
  bool is_fail = checked && max_err >= 5e-1f;
  if (is_fail || should_print_fa_tflops(1, tflops)) {
    char performance[32];
    if (cudnn_tflops_f32 > 0.0f) {
      snprintf(performance, sizeof(performance), "%.1f/%.1f (%.2fx)",
               tflops, cudnn_tflops_f32, tflops / cudnn_tflops_f32);
    } else {
      snprintf(performance, sizeof(performance), "%.1f", tflops);
    }
    printf("| %-56s | %.3e | %-19s |\n", label, max_err, performance);
  }

  free(h_o);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
}

static void bench_fa_3_tma_mma_ws_cute_dispatch(
    int B, int H, int seqlen, int head_dim,
    half *h_o_ref, float *ref_o,
    half *d_q, half *d_k, half *d_v, half *d_o,
    float cudnn_tflops_f32) {
  if (head_dim == 64) {
    bench_fa_3_tma_mma_ws_cute_launch<64, 1>(B, H, seqlen, h_o_ref, ref_o,
        d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    bench_fa_3_tma_mma_ws_cute_launch<64, 2>(B, H, seqlen, h_o_ref, ref_o,
        d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    bench_fa_3_tma_mma_ws_cute_launch<64, 3>(B, H, seqlen, h_o_ref, ref_o,
        d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    bench_fa_3_tma_mma_ws_cute_launch<64, 4>(B, H, seqlen, h_o_ref, ref_o,
        d_q, d_k, d_v, d_o, cudnn_tflops_f32);
  } else if (head_dim == 128) {
    bench_fa_3_tma_mma_ws_cute_launch<128, 1>(B, H, seqlen, h_o_ref, ref_o,
        d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    bench_fa_3_tma_mma_ws_cute_launch<128, 2>(B, H, seqlen, h_o_ref, ref_o,
        d_q, d_k, d_v, d_o, cudnn_tflops_f32);
  }
}


// FA2-style CuTe bench: single consumer, Br=128, no merge.
template <int kHeadDim, int kStagesK, int kStagesV = 1>
static void bench_fa_2_tma_mma_ws_cute_launch(
    int B, int H, int seqlen, half *h_o_ref, float *ref_o,
    half *d_q, half *d_k, half *d_v, half *d_o,
    float cudnn_tflops_f32) {
  using namespace cute;
  using Traits = fa_cute::FlashAttn2CuTeTraits<kHeadDim>;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  constexpr int kBr = 128;
  if (seqlen < kBr || seqlen % kBr != 0 || seqlen % 64 != 0) {
    char label[80];
    snprintf(label, sizeof(label),
             "FA2 CuTe TMA MMA WS (1 Consumer WG) (Sk=%d, Sv=%d, unaligned)", 
             kStagesK, kStagesV);
    printf("| %-56s | %-9s | %-19s |\n", label, "SKIP", "None");
    return;
  }

  int rows = B * H * seqlen;
  // Q tile: [128, D], K/V tile: [64, D]
  auto make_tma_q = [=]() {
    auto tensor = make_tensor(
        make_gmem_ptr(reinterpret_cast<cutlass::half_t *>(d_q)),
        make_shape(rows, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, _1{}));
    return make_tma_copy(
        SM90_TMA_LOAD{}, tensor, SmemLayoutQ{},
        Shape<_128, Int<kHeadDim>>{}, _1{});
  };
  auto make_tma_kv = [=](half *pointer) {
    auto tensor = make_tensor(
        make_gmem_ptr(reinterpret_cast<cutlass::half_t *>(pointer)),
        make_shape(rows, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, _1{}));
    return make_tma_copy(
        SM90_TMA_LOAD{}, tensor, SmemLayoutKV{},
        Shape<_64, Int<kHeadDim>>{}, _1{});
  };
  auto tma_q = make_tma_q();
  auto tma_k = make_tma_kv(d_k);
  auto tma_v = make_tma_kv(d_v);
  auto kernel = flash_attn_tma_mma_ws_split_q_cute<
      kHeadDim, decltype(tma_q), decltype(tma_k), decltype(tma_v),
      kStagesK, kStagesV>;
  // smem = Q[128*D] + K[Sk*64*D] + V[Sv*64*D] (no merge scratch)
  int smem_bytes = (cosize(SmemLayoutQ{}) +
                    kStagesK * cosize(SmemLayoutKV{}) +
                    kStagesV * cosize(SmemLayoutKV{})) *
                   sizeof(cutlass::half_t);
  bool smem_ok = check_smem_feasible((const void *)kernel, smem_bytes);
  if (!smem_ok) {
    char label[80];
    snprintf(label, sizeof(label),
             "FA2 CuTe TMA MMA WS (1 Consumer WG) (Sk=%d, Sv=%d, SMEM)", 
             kStagesK, kStagesV);
    if (g_debug)
      printf("| %-56s | %-9s | %-19s |\n", label, "SMEM too large", "None");
    return;
  }
  check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes),
        "bench cute fa2 set smem");

  dim3 grid(seqlen / kBr, B * H);
  cudaStream_t stream;
  cudaEvent_t start;
  cudaEvent_t stop;
  check(cudaStreamCreate(&stream), "bench cute fa2 stream");
  check(cudaEventCreate(&start), "bench cute fa2 event start");
  check(cudaEventCreate(&stop), "bench cute fa2 event stop");
  for (int warmup = 0; warmup < g_warmup; ++warmup) {
    kernel<<<grid, 384, smem_bytes, stream>>>(
        tma_q, tma_k, tma_v,
        reinterpret_cast<cutlass::half_t *>(d_o), rows, seqlen);
  }
  check(cudaStreamSynchronize(stream), "bench cute fa2 warmup sync");
  check(cudaEventRecord(start, stream), "bench cute fa2 record start");
  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    kernel<<<grid, 384, smem_bytes, stream>>>(
        tma_q, tma_k, tma_v,
        reinterpret_cast<cutlass::half_t *>(d_o), rows, seqlen);
  }
  check(cudaEventRecord(stop, stream), "bench cute fa2 record stop");
  check(cudaEventSynchronize(stop), "bench cute fa2 timing sync");
  float time_ms = 0.0f;
  check(cudaEventElapsedTime(&time_ms, start, stop), "bench cute fa2 elapsed");
  time_ms /= g_repeat;

  size_t count = (size_t)rows * kHeadDim;
  half *h_o = (half *)malloc(count * sizeof(half));
  check(cudaMemcpy(h_o, d_o, count * sizeof(half), cudaMemcpyDeviceToHost),
        "bench cute fa2 D2H");
  float max_err = 0.0f;
  bool checked = h_o_ref || ref_o;
  if (checked) {
    for (size_t idx = 0; idx < count; ++idx) {
      float reference = h_o_ref ? __half2float(h_o_ref[idx]) : ref_o[idx];
      max_err = max(max_err, fabsf(__half2float(h_o[idx]) - reference));
    }
  }
  float tflops = bench_fa_tflops(B, H, seqlen, kHeadDim, time_ms);
  char label[80];
  snprintf(label, sizeof(label),
           "FA2 CuTe TMA MMA WS (1 Consumer WG) (Sk=%d, Sv=%d, F32Acc)", 
           kStagesK, kStagesV);
  bool is_fail = checked && max_err >= 5e-1f;
  if (is_fail || should_print_fa_tflops(1, tflops)) {
    char performance[32];
    if (cudnn_tflops_f32 > 0.0f) {
      snprintf(performance, sizeof(performance), "%.1f/%.1f (%.2fx)",
               tflops, cudnn_tflops_f32, tflops / cudnn_tflops_f32);
    } else {
      snprintf(performance, sizeof(performance), "%.1f", tflops);
    }
    printf("| %-56s | %.3e | %-19s |\n", label, max_err, performance);
  }

  free(h_o);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
}

static void bench_fa_2_tma_mma_ws_cute_dispatch(
    int B, int H, int seqlen, int head_dim,
    half *h_o_ref, float *ref_o,
    half *d_q, half *d_k, half *d_v, half *d_o,
    float cudnn_tflops_f32) {
  if (head_dim == 64) {
    bench_fa_2_tma_mma_ws_cute_launch<64, 1, 1>(B, H, seqlen, h_o_ref, ref_o,
        d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    bench_fa_2_tma_mma_ws_cute_launch<64, 2, 1>(B, H, seqlen, h_o_ref, ref_o,
        d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    bench_fa_2_tma_mma_ws_cute_launch<64, 3, 1>(B, H, seqlen, h_o_ref, ref_o,
        d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    bench_fa_2_tma_mma_ws_cute_launch<64, 4, 1>(B, H, seqlen, h_o_ref, ref_o,
        d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    bench_fa_2_tma_mma_ws_cute_launch<64, 2, 2>(B, H, seqlen, h_o_ref, ref_o,
        d_q, d_k, d_v, d_o, cudnn_tflops_f32);
  } else if (head_dim == 128) {
    bench_fa_2_tma_mma_ws_cute_launch<128, 1, 1>(B, H, seqlen, h_o_ref, ref_o,
        d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    bench_fa_2_tma_mma_ws_cute_launch<128, 2, 1>(B, H, seqlen, h_o_ref, ref_o,
        d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    bench_fa_2_tma_mma_ws_cute_launch<128, 3, 1>(B, H, seqlen, h_o_ref, ref_o,
        d_q, d_k, d_v, d_o, cudnn_tflops_f32);
  }
}

// Bench: Phase 8 persist-D FlashAttention (persistent CTA + WS 1P+1C +
// scale fused, dense)。FA 家族中性能最优, 按 --bench 顺序最后跑;
// 对照 cuDNN SDPA (f32 compute)。MHA 布局 (H == Hkv)。
// 计时口径: 走 fwd 便捷入口, 每次 run() 重建 TMA descriptor (host 开销
// 计入 event 区间, 方向保守, 短 seqlen 时 TFLOPS 偏低), 与家族裸 kernel
// launch 口径不同。
static void bench_fa_persist_d_cute_launch(
    int B, int H, int seqlen, int head_dim, half *h_o_ref, float *ref_o,
    half *d_q, half *d_k, half *d_v, half *d_o, float cudnn_tflops_f32) {
  const float scale = 1.0f / sqrtf((float)head_dim);
  auto run = [&]() {
    flash_attn_cute_persist_d_sm120_fwd(
        reinterpret_cast<cutlass::half_t *>(d_q),
        reinterpret_cast<cutlass::half_t *>(d_k),
        reinterpret_cast<cutlass::half_t *>(d_v),
        reinterpret_cast<cutlass::half_t *>(d_o),
        B, H, H, seqlen, seqlen, head_dim, /*causal=*/false, scale);
  };
  for (int w = 0; w < g_warmup; ++w) run();
  check(cudaDeviceSynchronize(), "bench fa persist-d warmup sync");
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "bench fa persist-d event start");
  check(cudaEventCreate(&stop), "bench fa persist-d event stop");
  check(cudaEventRecord(start), "bench fa persist-d record start");
  for (int r = 0; r < g_repeat; ++r) run();
  check(cudaEventRecord(stop), "bench fa persist-d record stop");
  check(cudaEventSynchronize(stop), "bench fa persist-d timing sync");
  float time_ms = 0;
  check(cudaEventElapsedTime(&time_ms, start, stop),
        "bench fa persist-d elapsed");
  time_ms /= g_repeat;

  size_t count = (size_t)B * H * seqlen * head_dim;
  half *h_o = (half *)malloc(count * sizeof(half));
  check(cudaMemcpy(h_o, d_o, count * sizeof(half), cudaMemcpyDeviceToHost),
        "bench fa persist-d D2H");
  float max_err = 0.0f;
  bool checked = h_o_ref || ref_o;
  if (checked) {
    for (size_t i = 0; i < count; ++i) {
      float ref_val = h_o_ref ? __half2float(h_o_ref[i]) : ref_o[i];
      float err = fabsf(__half2float(h_o[i]) - ref_val);
      if (err > max_err) max_err = err;
    }
  }
  float tflops = bench_fa_tflops(B, H, seqlen, head_dim, time_ms);
  bool is_fail = checked && max_err >= 5e-1f;
  if (is_fail || should_print_fa_tflops(1, tflops)) {
    char tflops_str[32];
    if (cudnn_tflops_f32 > 0.0f) {
      snprintf(tflops_str, sizeof(tflops_str), "%.1f/%.1f (%.2fx)", tflops,
               cudnn_tflops_f32, tflops / cudnn_tflops_f32);
    } else {
      snprintf(tflops_str, sizeof(tflops_str), "%.1f", tflops);
    }
    char label[64];
    snprintf(label, sizeof(label), "FA2 CuTe TMA MMA Persistent-CTA WS (D=%d)",
             head_dim);
    printf("| %-56s | %.3e | %-19s |\n", label, checked ? max_err : 0.0f,
           tflops_str);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(h_o);
}
#endif /* NOTES_V2_ENABLE_CUTE */
#endif /* NOTES_V2_ENABLE_TMA_MMA_WS */

#if defined(NOTES_V2_ENABLE_CUDNN)
static float bench_cudnn_sdpa_tflops(half *d_q, half *d_k, half *d_v,
                                      half *d_o_ref, int B, int H, int seqlen,
                                      int head_dim, fe::DataType_t compute_type) {
  cudnnHandle_t handle;
  cudnnCreate(&handle);
  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(fe::DataType_t::HALF)
    .set_intermediate_data_type(fe::DataType_t::FLOAT)
    .set_compute_data_type(compute_type);

  auto Q = graph->tensor(fe::graph::Tensor_attributes()
    .set_uid(1).set_dim({B, H, seqlen, head_dim})
    .set_stride({H * seqlen * head_dim, seqlen * head_dim, head_dim, 1}));
  auto K = graph->tensor(fe::graph::Tensor_attributes()
    .set_uid(2).set_dim({B, H, seqlen, head_dim})
    .set_stride({H * seqlen * head_dim, seqlen * head_dim, head_dim, 1}));
  auto V = graph->tensor(fe::graph::Tensor_attributes()
    .set_uid(3).set_dim({B, H, seqlen, head_dim})
    .set_stride({H * seqlen * head_dim, seqlen * head_dim, head_dim, 1}));

  auto [O_sdpa, Stats] = graph->sdpa(Q, K, V,
    fe::graph::SDPA_attributes()
      .set_name("sdpa_ref")
      .set_attn_scale(1.0f / sqrtf((float)head_dim)));

  O_sdpa->set_output(true).set_uid(4)
    .set_dim({B, H, seqlen, head_dim})
    .set_stride({H * seqlen * head_dim, seqlen * head_dim, head_dim, 1});

  auto build_status = graph->build(handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
  float tflops = -1.0f;
  if (build_status.is_good()) {
    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> vp = {
      {1, d_q}, {2, d_k}, {3, d_v}, {4, d_o_ref}};
    int64_t ws_size = 0;
    if (graph->get_workspace_size(ws_size).is_good()) {
      int8_t *d_ws = nullptr;
      if (ws_size > 0) check(cudaMalloc(&d_ws, ws_size), "bench cudnn ws");
      if (graph->execute(handle, vp, d_ws).is_good()) {
        check(cudaDeviceSynchronize(), "bench cudnn sync");
        for (int w = 1; w < g_warmup; ++w)
          (void)graph->execute(handle, vp, d_ws);
        cudaDeviceSynchronize();
        cudaEvent_t ev_s, ev_e;
        cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);
        cudaEventRecord(ev_s);
        for (int r = 0; r < g_repeat; ++r)
          (void)graph->execute(handle, vp, d_ws);
        cudaEventRecord(ev_e);
        cudaEventSynchronize(ev_e);
        float time_ms = 0;
        cudaEventElapsedTime(&time_ms, ev_s, ev_e);
        time_ms /= g_repeat;
        tflops = bench_fa_tflops(B, H, seqlen, head_dim, time_ms);
        cudaEventDestroy(ev_s); cudaEventDestroy(ev_e);
      }
      if (d_ws) cudaFree(d_ws);
    }
  }
  cudnnDestroy(handle);
  return tflops;
}
#endif

void bench_flash_attn(int B, int H, int N, int D) {
  int seqlen = N, head_dim = D;

  if (head_dim % 64 != 0) {
    printf("| %-56s | %-9s | %-19s |\n", "FlashAttention-2", "unsupported D", "None");
    return;
  }

  size_t sz = (size_t)B * H * seqlen * head_dim * sizeof(half);

  srand(42);
  half *h_q = (half *)malloc(sz);
  half *h_k = (half *)malloc(sz);
  half *h_v = (half *)malloc(sz);
  for (int i = 0; i < B * H * seqlen * head_dim; i++) {
    h_q[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    h_k[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    h_v[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  }

  // Device allocations — shared by kernel and reference
  half *d_q, *d_k, *d_v, *d_o;
  check(cudaMalloc(&d_q, sz), "bench fa alloc Q");
  check(cudaMalloc(&d_k, sz), "bench fa alloc K");
  check(cudaMalloc(&d_v, sz), "bench fa alloc V");
  check(cudaMalloc(&d_o, sz), "bench fa alloc O");
  check(cudaMemcpy(d_q, h_q, sz, cudaMemcpyHostToDevice), "bench fa H2D Q");
  check(cudaMemcpy(d_k, h_k, sz, cudaMemcpyHostToDevice), "bench fa H2D K");
  check(cudaMemcpy(d_v, h_v, sz, cudaMemcpyHostToDevice), "bench fa H2D V");

  // Reference output: either cuDNN (half) or CPU (float)
  half *h_o_ref = nullptr;
  float *ref_o = nullptr;
  float cudnn_tflops_f16 = -1.0f, cudnn_tflops_f32 = -1.0f;
  int ref_count = B * H * seqlen * head_dim;

#if defined(NOTES_V2_ENABLE_CUDNN)
  if (!g_fa_skip_check) {
    half *d_o_ref;
    check(cudaMalloc(&d_o_ref, sz), "bench fa alloc O_ref");
    cudnn_tflops_f16 = bench_cudnn_sdpa_tflops(d_q, d_k, d_v, d_o_ref,
                                               B, H, seqlen, head_dim,
                                               fe::DataType_t::HALF);
    bool cudnn_ok = (cudnn_tflops_f16 > 0);
    if (cudnn_ok) {
      cudnn_tflops_f32 = bench_cudnn_sdpa_tflops(d_q, d_k, d_v, d_o_ref,
                                                 B, H, seqlen, head_dim,
                                                 fe::DataType_t::FLOAT);
      h_o_ref = (half *)malloc(sz);
      check(cudaMemcpy(h_o_ref, d_o_ref, sz, cudaMemcpyDeviceToHost), "bench fa D2H ref");
    } else {
      fprintf(stderr, "cudnn SDPA unsupported on this SM, using CPU ref\n");
    }
    cudaFree(d_o_ref);
  }
#endif

  // CPU reference fallback
  if (!g_fa_skip_check && !h_o_ref) {
    float *ref_q = (float *)malloc(sz * 4 / sizeof(half));
    float *ref_k = (float *)malloc(sz * 4 / sizeof(half));
    float *ref_v = (float *)malloc(sz * 4 / sizeof(half));
    ref_o = (float *)malloc(sz * 4 / sizeof(half));
    for (int i = 0; i < ref_count; i++) {
      ref_q[i] = __half2float(h_q[i]);
      ref_k[i] = __half2float(h_k[i]);
      ref_v[i] = __half2float(h_v[i]);
    }

    float scale = 1.0f / sqrtf((float)head_dim);
    for (int bi = 0; bi < B * H; bi++) {
      for (int qi = 0; qi < seqlen; qi++) {
        float smax = -INFINITY;
        float *S = (float *)malloc((size_t)seqlen * sizeof(float));
        for (int kj = 0; kj < seqlen; kj++) {
          float s = 0.0f;
          for (int d = 0; d < head_dim; d++)
            s += ref_q[bi * seqlen * head_dim + qi * head_dim + d] *
                 ref_k[bi * seqlen * head_dim + kj * head_dim + d];
          S[kj] = s * scale;
          if (S[kj] > smax) smax = S[kj];
        }
        double sum_exp = 0.0;
        for (int kj = 0; kj < seqlen; kj++)
          sum_exp += (double)expf(S[kj] - smax);
        float inv_sum = 1.0f / (float)sum_exp;
        for (int d = 0; d < head_dim; d++) {
          double o_acc = 0.0;
          for (int kj = 0; kj < seqlen; kj++)
            o_acc += (double)(expf(S[kj] - smax) * inv_sum) *
                     ref_v[bi * seqlen * head_dim + kj * head_dim + d];
          ref_o[bi * seqlen * head_dim + qi * head_dim + d] = (float)o_acc;
        }
        free(S);
      }
    }
    free(ref_q);
    free(ref_k);
    free(ref_v);
  }

  if (g_bench_fa3_cute_only) {
#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
    bench_fa_3_tma_mma_ws_cute_dispatch(
        B, H, seqlen, head_dim, h_o_ref, ref_o,
        d_q, d_k, d_v, d_o, cudnn_tflops_f32);
#endif
    free(h_q);
    free(h_k);
    free(h_v);
    free(h_o_ref);
    free(ref_o);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    return;
  }

  if (head_dim > 128) {
#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
    bench_fa_split_d_dispatch(B, H, seqlen, head_dim, h_o_ref, ref_o,
                               d_q, d_k, d_v, d_o, cudnn_tflops_f32);
#else
    printf("| %-56s | %-9s | %-19s |\n",
           "FA Split-D (TMA MMA WS disabled)", "SKIP", "None");
#endif
    free(h_q);
    free(h_k);
    free(h_v);
    free(h_o_ref);
    free(ref_o);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    return;
  }

  if (head_dim == 64) {
    if (g_fa_layout == FALayout::All || g_fa_layout == FALayout::Pad) {
      cudaDeviceSynchronize();
      bench_fa_launch<64, 1, 8, 8, 8, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o, 
                                       d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_launch<64, 2, 8, 8, 8, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o, 
                                        d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_launch<64, 1, 8, 8, 8, 1>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                         d_q, d_k, d_v, d_o, cudnn_tflops_f32);
      cudaDeviceSynchronize();
      bench_fa_launch<64, 2, 8, 8, 8, 1>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                         d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    }
    if (g_fa_layout == FALayout::All || g_fa_layout == FALayout::SwizzleQ) {
      cudaDeviceSynchronize();
      bench_fa_launch<64, 2, 0, 8, 8, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o, 
                                         d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_launch<64, 2, 0, 8, 8, 1>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                         d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    }
    if (g_fa_layout == FALayout::All || g_fa_layout == FALayout::SwizzleK) {
      cudaDeviceSynchronize();
      bench_fa_launch<64, 2, 8, 0, 8, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o, 
                                         d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_launch<64, 2, 8, 0, 8, 1>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                         d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    }
    if (g_fa_layout == FALayout::All || g_fa_layout == FALayout::SwizzleV) {
      cudaDeviceSynchronize();
      bench_fa_launch<64, 2, 8, 8, 0, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o, 
                                         d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_launch<64, 2, 8, 8, 0, 1>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                         d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    }
    if (g_fa_layout == FALayout::All || g_fa_layout == FALayout::SwizzleQK) {
      cudaDeviceSynchronize();
      bench_fa_launch<64, 2, 0, 0, 8, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o, 
                                         d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_launch<64, 2, 0, 0, 8, 1>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                         d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    }
    if (g_fa_layout == FALayout::All || g_fa_layout == FALayout::SwizzleQV) {
      cudaDeviceSynchronize();
      bench_fa_launch<64, 2, 0, 8, 0, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o, 
                                         d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_launch<64, 2, 0, 8, 0, 1>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                         d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    }
    if (g_fa_layout == FALayout::All || g_fa_layout == FALayout::SwizzleKV) {
      cudaDeviceSynchronize();
      bench_fa_launch<64, 2, 8, 0, 0, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o, 
                                        d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_launch<64, 2, 8, 0, 0, 1>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                        d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    }
    if (g_fa_layout == FALayout::All || g_fa_layout == FALayout::Swizzle) {
      cudaDeviceSynchronize();
      bench_fa_launch<64, 2, 0, 0, 0, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o, 
                                         d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_launch<64, 2, 0, 0, 0, 1>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                         d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    }
#if defined(NOTES_V2_ENABLE_CUTE)
    {
      cudaDeviceSynchronize();
      bench_fa_2_mma_stages_cute_dispatch(
        B, H, seqlen, head_dim, h_o_ref, ref_o, d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    }
#endif
#if defined(NOTES_V2_ENABLE_TMA_MMA_WS)
    {
      cudaDeviceSynchronize();
      bench_fa_tma_mma_ws_dispatch<1, 1, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                            d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_tma_mma_ws_dispatch<2, 1, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                            d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_tma_mma_ws_dispatch<3, 1, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                            d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_tma_mma_ws_dispatch<4, 1, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                            d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_tma_mma_ws_dispatch<2, 2, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                            d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      // F32Acc variants
      cudaDeviceSynchronize();
      bench_fa_tma_mma_ws_dispatch<1, 1, 1>(B, H, seqlen, head_dim, h_o_ref,
                                            ref_o, d_q, d_k, d_v, d_o, cudnn_tflops_f32);
      cudaDeviceSynchronize();
      bench_fa_tma_mma_ws_dispatch<2, 1, 1>(B, H, seqlen, head_dim, h_o_ref,
                                            ref_o, d_q, d_k, d_v, d_o, cudnn_tflops_f32);
      cudaDeviceSynchronize();
      bench_fa_tma_mma_ws_dispatch<3, 1, 1>(B, H, seqlen, head_dim, h_o_ref,
                                            ref_o, d_q, d_k, d_v, d_o, cudnn_tflops_f32);
      cudaDeviceSynchronize();
      bench_fa_tma_mma_ws_dispatch<4, 1, 1>(B, H, seqlen, head_dim, h_o_ref,
                                            ref_o, d_q, d_k, d_v, d_o, cudnn_tflops_f32);
      cudaDeviceSynchronize();
      bench_fa_tma_mma_ws_dispatch<2, 2, 1>(B, H, seqlen, head_dim, h_o_ref,
                                            ref_o, d_q, d_k, d_v, d_o, cudnn_tflops_f32);
      // FA3-style dual-consumer 
      cudaDeviceSynchronize();
      bench_fa_3_tma_ws_dispatch<0>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                     d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_3_tma_ws_dispatch<1>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                     d_q, d_k, d_v, d_o, cudnn_tflops_f32);
#if defined(NOTES_V2_ENABLE_CUTE)
      cudaDeviceSynchronize();
      bench_fa_2_tma_mma_ws_cute_dispatch(B, H, seqlen, head_dim, h_o_ref, ref_o,
                               d_q, d_k, d_v, d_o, cudnn_tflops_f32);
      cudaDeviceSynchronize();
      bench_fa_3_tma_mma_ws_cute_dispatch(B, H, seqlen, head_dim, h_o_ref, ref_o,
                               d_q, d_k, d_v, d_o, cudnn_tflops_f32);
#endif
    }
#endif
  } else {
    if (g_fa_layout == FALayout::All || g_fa_layout == FALayout::Pad) {
      cudaDeviceSynchronize();
      bench_fa_launch<128, 1, 8, 8, 8, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o, 
                                          d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_launch<128, 2, 8, 8, 8, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o, 
                                          d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_launch<128, 1, 8, 8, 8, 1>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                          d_q, d_k, d_v, d_o, cudnn_tflops_f32);
      cudaDeviceSynchronize();
      bench_fa_launch<128, 2, 8, 8, 8, 1>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                          d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    }
    if (g_fa_layout == FALayout::All || g_fa_layout == FALayout::SwizzleQ) {
      cudaDeviceSynchronize();
      bench_fa_launch<128, 2, 0, 8, 8, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o, 
                                          d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_launch<128, 2, 0, 8, 8, 1>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                          d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    }
    if (g_fa_layout == FALayout::All || g_fa_layout == FALayout::SwizzleK) {
      cudaDeviceSynchronize();
      bench_fa_launch<128, 2, 8, 0, 8, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o, 
                                          d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_launch<128, 2, 8, 0, 8, 1>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                          d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    }
    if (g_fa_layout == FALayout::All || g_fa_layout == FALayout::SwizzleV) {
      cudaDeviceSynchronize();
      bench_fa_launch<128, 2, 8, 8, 0, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o, 
                                          d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_launch<128, 2, 8, 8, 0, 1>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                          d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    }
    if (g_fa_layout == FALayout::All || g_fa_layout == FALayout::SwizzleQK) {
      cudaDeviceSynchronize();
      bench_fa_launch<128, 2, 0, 0, 8, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o, 
                                          d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_launch<128, 2, 0, 0, 8, 1>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                          d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    }
    if (g_fa_layout == FALayout::All || g_fa_layout == FALayout::SwizzleQV) {
      cudaDeviceSynchronize();
      bench_fa_launch<128, 2, 0, 8, 0, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o, 
                                          d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_launch<128, 2, 0, 8, 0, 1>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                          d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    }
    if (g_fa_layout == FALayout::All || g_fa_layout == FALayout::SwizzleKV) {
      cudaDeviceSynchronize();
      bench_fa_launch<128, 2, 8, 0, 0, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o, 
                                          d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_launch<128, 2, 8, 0, 0, 1>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                          d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    }
    if (g_fa_layout == FALayout::All || g_fa_layout == FALayout::Swizzle) {
      cudaDeviceSynchronize();
      bench_fa_launch<128, 2, 0, 0, 0, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o, 
                                          d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_launch<128, 2, 0, 0, 0, 1>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                          d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    }
#if defined(NOTES_V2_ENABLE_CUTE)
    {
      cudaDeviceSynchronize();
      bench_fa_2_mma_stages_cute_dispatch(
        B, H, seqlen, head_dim, h_o_ref, ref_o, d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    }
#endif    
#if defined(NOTES_V2_ENABLE_TMA_MMA_WS)
    {
      cudaDeviceSynchronize();
      bench_fa_tma_mma_ws_dispatch<1, 1, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                            d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_tma_mma_ws_dispatch<2, 1, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                            d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_tma_mma_ws_dispatch<3, 1, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                            d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_tma_mma_ws_dispatch<2, 2, 0>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                            d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      // F32Acc variants
      cudaDeviceSynchronize();
      bench_fa_tma_mma_ws_dispatch<1, 1, 1>(B, H, seqlen, head_dim, h_o_ref,
                                            ref_o, d_q, d_k, d_v, d_o, cudnn_tflops_f32);
      cudaDeviceSynchronize();
      bench_fa_tma_mma_ws_dispatch<2, 1, 1>(B, H, seqlen, head_dim, h_o_ref,
                                            ref_o, d_q, d_k, d_v, d_o, cudnn_tflops_f32);
      cudaDeviceSynchronize();
      bench_fa_tma_mma_ws_dispatch<3, 1, 1>(B, H, seqlen, head_dim, h_o_ref,
                                            ref_o, d_q, d_k, d_v, d_o, cudnn_tflops_f32);
      cudaDeviceSynchronize();
      bench_fa_tma_mma_ws_dispatch<2, 2, 1>(B, H, seqlen, head_dim, h_o_ref,
                                            ref_o, d_q, d_k, d_v, d_o, cudnn_tflops_f32);
      // FA3-style dual-consumer
      cudaDeviceSynchronize();
      bench_fa_3_tma_ws_dispatch<0>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                     d_q, d_k, d_v, d_o, cudnn_tflops_f16);
      cudaDeviceSynchronize();
      bench_fa_3_tma_ws_dispatch<1>(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                     d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    #if defined(NOTES_V2_ENABLE_CUTE)
      cudaDeviceSynchronize();
      bench_fa_2_tma_mma_ws_cute_dispatch(B, H, seqlen, head_dim, h_o_ref, ref_o,
               d_q, d_k, d_v, d_o, cudnn_tflops_f32);
      cudaDeviceSynchronize();
      bench_fa_3_tma_mma_ws_cute_dispatch(B, H, seqlen, head_dim, h_o_ref, ref_o,
               d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    #endif
    }
#endif
  }
#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
  // Phase 8: persist-D (persistent CTA + scale fused)——FA 家族最快,
  // 按 bench 顺序放在最后; 与上面两个分支同样的 head_dim 支持集
  if (head_dim == 64 || head_dim == 128) {
    cudaDeviceSynchronize();
    bench_fa_persist_d_cute_launch(B, H, seqlen, head_dim, h_o_ref, ref_o,
                                   d_q, d_k, d_v, d_o, cudnn_tflops_f32);
  }
#endif
  cudaDeviceSynchronize();

  free(h_q);
  free(h_k);
  free(h_v);
  free(h_o_ref);
  free(ref_o);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_o);
}

#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
// notes-v2.cu main() 的非模板入口：模板实例化留在本 TU
void run_fa3_cute_tests() {
  test_flash_attn_3_tma_mma_ws_split_q_cute<64>();
  test_flash_attn_3_tma_mma_ws_split_q_cute<128>();
}

void run_fa3_cute_tma_smoke_tests() {
  test_flash_attn_3_cute_tma_copy_smoke<64>();
  test_flash_attn_3_cute_tma_copy_smoke<128>();
}

void run_fa2_cute_tests() {
  test_flash_attn_tma_mma_ws_split_q_cute<64>();
  test_flash_attn_tma_mma_ws_split_q_cute<128>();
}

void run_fa2_cute_cpasync_tests() {
  test_flash_attn_mma_stages_split_q_cute<64>();
  test_flash_attn_mma_stages_split_q_cute<128>();
}

void run_pd_cute_tests() {
  test_flash_attn_cute_persist_d_sm120<64, 2048, 2048, 8, 8, false>();
  test_flash_attn_cute_persist_d_sm120<128, 1024, 1024, 2, 2, false>();
  test_flash_attn_cute_persist_d_sm120<128, 1024, 1024, 2, 2, true>();
  test_flash_attn_cute_persist_d_sm120<64, 1024, 1024, 4, 2, false>();
  test_flash_attn_cute_persist_d_sm120<128, 300, 512, 2, 2, false>();
  test_flash_attn_cute_persist_d_sm120<128, 256, 300, 2, 2, false>();  // KV 尾 mask
  test_flash_attn_cute_persist_d_sm120<128, 512, 256, 2, 2, true>();   // 反向全 mask
}
#endif
