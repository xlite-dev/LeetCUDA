// =============================================================================
// notes-v2.cu — CUDA Kernel 面试背题笔记
// =============================================================================
//
// 整理自 LeetCUDA 项目（https://github.com/xlite-dev/LeetCUDA），涵盖：
//   - 面试高频 CUDA kernel 的完整实现（~30 个 kernel）
//   - 每类 kernel 附带详细的面试要点注释（WHY + HOW）
//   - 优化技术的递进式讲解（naive → tiling → vectorize → tensor core → ws）
//   - BLAS 语义：N=col-major(Normal), T=row-major(Transposed)
//
// 10 个 Phase 覆盖：
//   Phase 0 — 面试框架速查（GPU 架构 / Memory Hierarchy / Roofline / 优化清单）
//   Phase 1 — 基础原语：Warp Reduce / Block Reduce / Dot Product（含 broadcast 增强版）
//   Phase 2 — Elementwise：ReLU / Elementwise Add / Histogram（基础 + float4 向量化 + atomic）
//   Phase 3 — Softmax：naive → safe → online + RMS/Layer Norm
//   Phase 4 — RoPE：旋转位置编码（Llama 风格 theta=10000）
//   Phase 5 — Mat Transpose：基础版 + BCF merge_write 最佳版（Bank Conflict专题）
//   Phase 6 — GEMV：SGEMV K32/K128/K16（warp-per-row）
//   Phase 7 — GEMM ★：SGEMM → HGEMM → MMA m16n8k16(TN布局) → WGMMA m64n128k16
//   Phase 8 — FlashAttention-2split_q（FA-2, 含 online softmax + P@V 寄存器复用）
//
// =============================================================================
#include "base.cuh"
#include "sgemv.cuh"
#include "sgemm.cuh"
#include "hgemm.cuh"
#include "flash_attn.cuh"
#include "ffpa_attn.cuh"

// ================================================================
// 以下是测试代码，验证 Phase 1 - Phase 8 的kernel的正确性，不评估性能。
// ================================================================

// Bench / test mode globals (placed early so all functions can reference them)
static bool g_debug = false;
static bool g_bench_hgemm = false;
static bool g_bench_hgemm_all = false;
static bool g_bench_fa = false;
static bool g_bench_fa3_cute_only = false;
static bool g_bench_all = false;
static bool g_fa_skip_check = false;
static bool g_swizzle_eq_check = false;
enum class FALayout {
  All, Pad, SwizzleQ, SwizzleK, SwizzleV, SwizzleQK, SwizzleQV, SwizzleKV, Swizzle
};
static FALayout g_fa_layout = FALayout::Pad;
static int g_bench_M = 8192, g_bench_N = 8192, g_bench_K = 8192;
static int g_bench_B = 1, g_bench_H = 32, g_bench_Nfa = 8192, g_bench_D = 128;
static int g_warmup = 2, g_repeat = 3;
static float g_fa_f16_max_tflops = 0.0f;
static float g_fa_f32_max_tflops = 0.0f;
static float g_hgemm_max_tflops = 0.0f;
static bool g_verbose = false;

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
// only print when the current TFLOPS exceeds the running max.
static bool should_print_hgemm_tflops(float tflops) {
  if (g_verbose || g_debug) return true;
  if (tflops > g_hgemm_max_tflops) {
    g_hgemm_max_tflops = tflops;
    return true;
  }
  return false;
}

static bool check_smem_feasible(const void *kernel_func, size_t dyn_smem_bytes) {
  int device = 0;
  int max_smem = 0;
  cudaFuncAttributes attrs{};
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  cudaFuncGetAttributes(&attrs, kernel_func);
  return dyn_smem_bytes + attrs.sharedSizeBytes <= (size_t)max_smem;
}

static inline void check(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "[ERROR] %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

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

static void test_block_reduce(int N) {

  srand(42);
  float *h_a = (float *)malloc((size_t)N * sizeof(float));
  for (int i = 0; i < N; i++)
    h_a[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  // CPU reference: sum of all elements
  double ref = 0.0;
  for (int i = 0; i < N; i++) ref += (double)h_a[i];

  float *d_a, *d_y;
  check(cudaMalloc(&d_a, (size_t)N * sizeof(float)), "blockreduce alloc A");
  check(cudaMalloc(&d_y, sizeof(float)), "blockreduce alloc Y");

  check(cudaMemcpy(d_a, h_a, (size_t)N * sizeof(float), cudaMemcpyHostToDevice), "blockreduce H2D A");
  check(cudaMemset(d_y, 0, sizeof(float)), "blockreduce zero Y");

  dim3 block(128);
  dim3 grid((N + 127) / 128);
  block_reduce_all<128><<<grid, block>>>(d_a, d_y, N);
  check(cudaGetLastError(), "blockreduce launch");
  check(cudaDeviceSynchronize(), "blockreduce sync");

  float result;
  check(cudaMemcpy(&result, d_y, sizeof(float), cudaMemcpyDeviceToHost), "blockreduce D2H");

  float err = fabsf(result - (float)ref);
  printf("| %-56s | %.3e |\n", "BlockReduce", err);

  free(h_a);
  cudaFree(d_a); cudaFree(d_y);
}


static void test_dot(int N) {

  srand(42);
  float *h_a = (float *)malloc((size_t)N * sizeof(float));
  float *h_b = (float *)malloc((size_t)N * sizeof(float));
  for (int i = 0; i < N; i++) {
    h_a[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    h_b[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  }

  // CPU reference
  double ref = 0.0;
  for (int i = 0; i < N; i++) ref += (double)h_a[i] * (double)h_b[i];

  float *d_a, *d_b, *d_y;
  check(cudaMalloc(&d_a, (size_t)N * sizeof(float)), "dot alloc A");
  check(cudaMalloc(&d_b, (size_t)N * sizeof(float)), "dot alloc B");
  check(cudaMalloc(&d_y, sizeof(float)), "dot alloc Y");

  check(cudaMemcpy(d_a, h_a, (size_t)N * sizeof(float), cudaMemcpyHostToDevice), "dot H2D A");
  check(cudaMemcpy(d_b, h_b, (size_t)N * sizeof(float), cudaMemcpyHostToDevice), "dot H2D B");
  check(cudaMemset(d_y, 0, sizeof(float)), "dot zero Y");

  dim3 block(128);
  dim3 grid((N + 127) / 128);
  dot<128><<<grid, block>>>(d_a, d_b, d_y, N);
  check(cudaGetLastError(), "dot launch");
  check(cudaDeviceSynchronize(), "dot sync");

  float result;
  check(cudaMemcpy(&result, d_y, sizeof(float), cudaMemcpyDeviceToHost), "dot D2H");

  float err = fabsf(result - (float)ref);
  printf("| %-56s | %.3e |\n", "Dot", err);

  // ---- Dot Vec4 ----
  check(cudaMemset(d_y, 0, sizeof(float)), "dot_vec4 zero Y");
  dim3 block_v4(32);
  dot_vec4<32><<<grid, block_v4>>>(d_a, d_b, d_y, N);
  check(cudaGetLastError(), "dot_vec4 launch");
  check(cudaDeviceSynchronize(), "dot_vec4 sync");

  check(cudaMemcpy(&result, d_y, sizeof(float), cudaMemcpyDeviceToHost), "dot_vec4 D2H");
  float err_v4 = fabsf(result - (float)ref);
  printf("| %-56s | %.3e |\n", "Dot-Vec4", err_v4);

  free(h_a); free(h_b);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
}


static void test_relu(int N) {
  srand(42);
  float *h_x = (float *)malloc((size_t)N * sizeof(float));
  float *h_y = (float *)malloc((size_t)N * sizeof(float));
  for (int i = 0; i < N; i++)
    h_x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  float *d_x, *d_y;
  check(cudaMalloc(&d_x, (size_t)N * sizeof(float)), "relu alloc X");
  check(cudaMalloc(&d_y, (size_t)N * sizeof(float)), "relu alloc Y");
  check(cudaMemcpy(d_x, h_x, (size_t)N * sizeof(float), cudaMemcpyHostToDevice), "relu H2D");

  for (int i = 0; i < N; i++) h_y[i] = fmaxf(0.0f, h_x[i]);
  dim3 block256(256);
  dim3 grid256((N + 255) / 256);
  relu<<<grid256, block256>>>(d_x, d_y, N);
  check(cudaGetLastError(), "relu launch");
  check(cudaDeviceSynchronize(), "relu sync");
  check(cudaMemcpy(h_y, d_y, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost), "relu D2H");
  float max_err = 0.0f;
  for (int i = 0; i < N; i++) {
    float expected = fmaxf(0.0f, h_x[i]);
    float err = fabsf(h_y[i] - expected);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "ReLU", max_err);

  dim3 block64(64);
  relu_vec4<<<grid256, block64>>>(d_x, d_y, N);
  check(cudaGetLastError(), "relu_vec4 launch");
  check(cudaDeviceSynchronize(), "relu_vec4 sync");
  check(cudaMemcpy(h_y, d_y, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost), "relu_vec4 D2H");
  max_err = 0.0f;
  for (int i = 0; i < N; i++) {
    float expected = fmaxf(0.0f, h_x[i]);
    float err = fabsf(h_y[i] - expected);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "ReLU-Vec4", max_err);

  free(h_x); free(h_y);
  cudaFree(d_x); cudaFree(d_y);
}


static void test_elementwise(int N) {
  srand(42);
  float *h_a = (float *)malloc((size_t)N * sizeof(float));
  float *h_b = (float *)malloc((size_t)N * sizeof(float));
  for (int i = 0; i < N; i++) {
    h_a[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    h_b[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  }

  float *d_a, *d_b, *d_c;
  check(cudaMalloc(&d_a, (size_t)N * sizeof(float)), "eadd alloc A");
  check(cudaMalloc(&d_b, (size_t)N * sizeof(float)), "eadd alloc B");
  check(cudaMalloc(&d_c, (size_t)N * sizeof(float)), "eadd alloc C");
  check(cudaMemcpy(d_a, h_a, (size_t)N * sizeof(float), cudaMemcpyHostToDevice), "eadd H2D A");
  check(cudaMemcpy(d_b, h_b, (size_t)N * sizeof(float), cudaMemcpyHostToDevice), "eadd H2D B");

  dim3 block256(256);
  dim3 grid256((N + 255) / 256);
  elementwise_add<<<grid256, block256>>>(d_a, d_b, d_c, N);
  check(cudaGetLastError(), "eadd launch");
  check(cudaDeviceSynchronize(), "eadd sync");
  float *h_c = (float *)malloc((size_t)N * sizeof(float));
  check(cudaMemcpy(h_c, d_c, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost), "eadd D2H");
  float max_err = 0.0f;
  for (int i = 0; i < N; i++) {
    float err = fabsf(h_c[i] - (h_a[i] + h_b[i]));
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "ElemwiseAdd", max_err);

  dim3 block64(64);
  check(cudaMemset(d_c, 0, (size_t)N * sizeof(float)), "eadd_vec4 zero C");
  elementwise_add_vec4<<<grid256, block64>>>(d_a, d_b, d_c, N);
  check(cudaGetLastError(), "eadd_vec4 launch");
  check(cudaDeviceSynchronize(), "eadd_vec4 sync");
  check(cudaMemcpy(h_c, d_c, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost), "eadd_vec4 D2H");
  max_err = 0.0f;
  for (int i = 0; i < N; i++) {
    float err = fabsf(h_c[i] - (h_a[i] + h_b[i]));
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "ElemwiseAdd-Vec4", max_err);

  free(h_a); free(h_b); free(h_c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}


static void test_histogram(int N) {
  srand(42);
  int BINS = 16;
  int *h_hist = (int *)calloc(BINS, sizeof(int));
  int *h_hist_ref = (int *)calloc(BINS, sizeof(int));
  int *h_idx = (int *)malloc((size_t)N * sizeof(int));
  for (int i = 0; i < N; i++) h_idx[i] = rand() % BINS;
  for (int i = 0; i < N; i++) h_hist_ref[h_idx[i]]++;

  int *d_idx, *d_hist;
  check(cudaMalloc(&d_idx, (size_t)N * sizeof(int)), "hist alloc idx");
  check(cudaMalloc(&d_hist, BINS * sizeof(int)), "hist alloc hist");
  check(cudaMemcpy(d_idx, h_idx, (size_t)N * sizeof(int), cudaMemcpyHostToDevice), "hist H2D idx");
  check(cudaMemset(d_hist, 0, BINS * sizeof(int)), "hist zero");

  dim3 block256(256);
  dim3 grid256((N + 255) / 256);
  histogram<<<grid256, block256>>>(d_idx, d_hist, N);
  check(cudaGetLastError(), "histogram launch");
  check(cudaDeviceSynchronize(), "histogram sync");
  check(cudaMemcpy(h_hist, d_hist, BINS * sizeof(int), cudaMemcpyDeviceToHost), "hist D2H");

  float max_err = 0.0f;
  for (int i = 0; i < BINS; i++) {
    float err = fabsf((float)(h_hist[i] - h_hist_ref[i]));
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "Histogram", max_err);

  free(h_hist); free(h_hist_ref); free(h_idx);
  cudaFree(d_idx); cudaFree(d_hist);
}


static void test_merge_attn_states(int num_tokens, int num_heads,
                                   int head_size) {

  size_t out_size = (size_t)num_tokens * num_heads * head_size * sizeof(float);
  size_t lse_size = (size_t)num_heads * num_tokens * sizeof(float);

  float *h_prefix_out = (float *)malloc(out_size);
  float *h_suffix_out = (float *)malloc(out_size);
  float *h_prefix_lse = (float *)malloc(lse_size);
  float *h_suffix_lse = (float *)malloc(lse_size);
  float *h_output_ref = (float *)malloc(out_size);

  srand(42);
  for (int i = 0; i < num_tokens * num_heads * head_size; i++) {
    h_prefix_out[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    h_suffix_out[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  }
  for (int i = 0; i < num_heads * num_tokens; i++) {
    h_prefix_lse[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
    h_suffix_lse[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
  }

  // CPU reference: 与 kernel 完全相同的浮点计算
  for (int t = 0; t < num_tokens; t++) {
    for (int h = 0; h < num_heads; h++) {
      float p_lse = h_prefix_lse[h * num_tokens + t];
      float s_lse = h_suffix_lse[h * num_tokens + t];
      p_lse = isinf(p_lse) ? -INFINITY : p_lse;
      s_lse = isinf(s_lse) ? -INFINITY : s_lse;

      float max_lse = fmaxf(p_lse, s_lse);
      p_lse -= max_lse;
      s_lse -= max_lse;
      float p_se = expf(p_lse);
      float s_se = expf(s_lse);
      float p_scale = p_se / (p_se + s_se);
      float s_scale = s_se / (p_se + s_se);

      int head_off = t * num_heads * head_size + h * head_size;
      for (int d = 0; d < head_size; d++) {
        h_output_ref[head_off + d] =
            h_prefix_out[head_off + d] * p_scale +
            h_suffix_out[head_off + d] * s_scale;
      }
    }
  }

  float *d_prefix_out, *d_suffix_out, *d_prefix_lse, *d_suffix_lse, *d_output;
  check(cudaMalloc(&d_prefix_out, out_size), "merge_attn alloc p_out");
  check(cudaMalloc(&d_suffix_out, out_size), "merge_attn alloc s_out");
  check(cudaMalloc(&d_prefix_lse, lse_size), "merge_attn alloc p_lse");
  check(cudaMalloc(&d_suffix_lse, lse_size), "merge_attn alloc s_lse");
  check(cudaMalloc(&d_output, out_size), "merge_attn alloc output");

  check(cudaMemcpy(d_prefix_out, h_prefix_out, out_size,
                   cudaMemcpyHostToDevice),
        "merge_attn H2D p_out");
  check(cudaMemcpy(d_suffix_out, h_suffix_out, out_size,
                   cudaMemcpyHostToDevice),
        "merge_attn H2D s_out");
  check(cudaMemcpy(d_prefix_lse, h_prefix_lse, lse_size,
                   cudaMemcpyHostToDevice),
        "merge_attn H2D p_lse");
  check(cudaMemcpy(d_suffix_lse, h_suffix_lse, lse_size,
                   cudaMemcpyHostToDevice),
        "merge_attn H2D s_lse");

  int threads_per_head = head_size / 4;
  int total_threads = num_tokens * num_heads * threads_per_head;
  dim3 block(128);
  dim3 grid((total_threads + 127) / 128);
  merge_attn_states<<<grid, block>>>(
      d_output, d_prefix_out, d_prefix_lse, d_suffix_out, d_suffix_lse,
      num_tokens, num_heads, head_size);
  check(cudaGetLastError(), "merge_attn launch");
  check(cudaDeviceSynchronize(), "merge_attn sync");

  float *h_output = (float *)malloc(out_size);
  check(cudaMemcpy(h_output, d_output, out_size, cudaMemcpyDeviceToHost),
        "merge_attn D2H");

  float max_err = 0.0f;
  for (int i = 0; i < num_tokens * num_heads * head_size; i++) {
    float err = fabsf(h_output[i] - h_output_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "MergeAttnStates", max_err);

  // 边界测试: +inf LSE → 权重退化为 0（空 attention 段）
  h_prefix_lse[0] = INFINITY; // head 0, token 0 的 LSE = +inf
  h_suffix_lse[0] = 0.0f;
  check(cudaMemcpy(d_prefix_lse, h_prefix_lse, lse_size,
                   cudaMemcpyHostToDevice),
        "merge_attn H2D inf lse");
  merge_attn_states<<<grid, block>>>(
      d_output, d_prefix_out, d_prefix_lse, d_suffix_out, d_suffix_lse,
      num_tokens, num_heads, head_size);
  check(cudaGetLastError(), "merge_attn inf launch");
  check(cudaDeviceSynchronize(), "merge_attn inf sync");
  check(cudaMemcpy(h_output, d_output, out_size, cudaMemcpyDeviceToHost),
        "merge_attn inf D2H");

  // token 0, head 0 的所有元素应等于 suffix_output（prefix 权重 α=0）
  float inf_err = 0.0f;
  for (int d = 0; d < head_size; d++) {
    float err = fabsf(h_output[d] - h_suffix_out[d]);
    if (err > inf_err) inf_err = err;
  }
  printf("| %-56s | %.3e |\n", "MergeAttnStates-inf", inf_err);

  free(h_prefix_out);
  free(h_suffix_out);
  free(h_prefix_lse);
  free(h_suffix_lse);
  free(h_output_ref);
  free(h_output);
  cudaFree(d_prefix_out);
  cudaFree(d_suffix_out);
  cudaFree(d_prefix_lse);
  cudaFree(d_suffix_lse);
  cudaFree(d_output);
}


static void test_softmax(int N) {
  // Use N = blockDim.x so one block processes all elements as one token.
  constexpr int kNumThreads = 256;
  if (N != kNumThreads) {
    printf("  OnlineSafeSoftmax: N must be %d (got %d), skipping.\n", kNumThreads, N);
    return;
  }

  srand(42);
  float *h_x = (float *)malloc((size_t)N * sizeof(float));
  float *h_y_ref = (float *)malloc((size_t)N * sizeof(float));
  for (int i = 0; i < N; i++)
    h_x[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;  // [-5, 5]

  // CPU reference: softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
  float max_val = -FLT_MAX;
  for (int i = 0; i < N; i++) if (h_x[i] > max_val) max_val = h_x[i];
  double sum_exp = 0.0;
  for (int i = 0; i < N; i++) sum_exp += (double)expf(h_x[i] - max_val);
  for (int i = 0; i < N; i++)
    h_y_ref[i] = expf(h_x[i] - max_val) / (float)sum_exp;

  float *d_x, *d_y;
  check(cudaMalloc(&d_x, (size_t)N * sizeof(float)), "softmax alloc X");
  check(cudaMalloc(&d_y, (size_t)N * sizeof(float)), "softmax alloc Y");

  check(cudaMemcpy(d_x, h_x, (size_t)N * sizeof(float), cudaMemcpyHostToDevice), "softmax H2D X");

  dim3 block(256);
  dim3 grid(1);  // one token covering all N elements
  online_safe_softmax_per_token<<<grid, block>>>(d_x, d_y, N);
  check(cudaGetLastError(), "softmax launch");
  check(cudaDeviceSynchronize(), "softmax sync");

  float *h_y = (float *)malloc((size_t)N * sizeof(float));
  check(cudaMemcpy(h_y, d_y, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost), "softmax D2H");

  float max_err = 0.0f;
  for (int i = 0; i < N; i++) {
    float err = fabsf(h_y[i] - h_y_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "OnlineSafeSoftmax", max_err);

  // ---- Safe Softmax ----
  safe_softmax_per_token<<<grid, block>>>(d_x, d_y, N);
  check(cudaGetLastError(), "safe_softmax launch");
  check(cudaDeviceSynchronize(), "safe_softmax sync");
  check(cudaMemcpy(h_y, d_y, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost), "safe_softmax D2H");
  max_err = 0.0f;
  for (int i = 0; i < N; i++) {
    float err = fabsf(h_y[i] - h_y_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "SafeSoftmax", max_err);

  // ---- Naive Softmax ----
  softmax_per_token<<<grid, block>>>(d_x, d_y, N);
  check(cudaGetLastError(), "naive_softmax launch");
  check(cudaDeviceSynchronize(), "naive_softmax sync");
  check(cudaMemcpy(h_y, d_y, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost), "naive_softmax D2H");
  max_err = 0.0f;
  for (int i = 0; i < N; i++) {
    float err = fabsf(h_y[i] - h_y_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "NaiveSoftmax", max_err);

  free(h_x); free(h_y); free(h_y_ref);
  cudaFree(d_x); cudaFree(d_y);
}


static void test_rms_norm(int N, int K) {

  srand(42);
  float *h_x = (float *)malloc((size_t)N * K * sizeof(float));
  float *h_y_ref = (float *)malloc((size_t)N * K * sizeof(float));
  for (int i = 0; i < N * K; i++)
    h_x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  float g = 1.5f;  // gain

  // CPU reference: y = (x / rms(x)) * g
  float epsilon = 1e-5f;
  for (int n = 0; n < N; n++) {
    double sum_sq = 0.0;
    for (int k = 0; k < K; k++) sum_sq += (double)h_x[n * K + k] * (double)h_x[n * K + k];
    float rms = sqrtf((float)sum_sq / (float)K + epsilon);
    for (int k = 0; k < K; k++)
      h_y_ref[n * K + k] = (h_x[n * K + k] / rms) * g;
  }

  float *d_x, *d_y;
  check(cudaMalloc(&d_x, (size_t)N * K * sizeof(float)), "rmsnorm alloc X");
  check(cudaMalloc(&d_y, (size_t)N * K * sizeof(float)), "rmsnorm alloc Y");
  check(cudaMemcpy(d_x, h_x, (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice), "rmsnorm H2D X");

  dim3 block(128);
  dim3 grid(N);
  rms_norm<<<grid, block>>>(d_x, d_y, g, N, K);
  check(cudaGetLastError(), "rmsnorm launch");
  check(cudaDeviceSynchronize(), "rmsnorm sync");

  float *h_y = (float *)malloc((size_t)N * K * sizeof(float));
  check(cudaMemcpy(h_y, d_y, (size_t)N * K * sizeof(float), cudaMemcpyDeviceToHost), "rmsnorm D2H");

  float max_err = 0.0f;
  for (int i = 0; i < N * K; i++) {
    float err = fabsf(h_y[i] - h_y_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "RMSNorm", max_err);

  // ---- RMS Norm Vec4 ----
  dim3 block_rv4(32);
  rms_norm_vec4<<<grid, block_rv4>>>(d_x, d_y, g, N, K);
  check(cudaGetLastError(), "rmsnorm_vec4 launch");
  check(cudaDeviceSynchronize(), "rmsnorm_vec4 sync");
  check(cudaMemcpy(h_y, d_y, (size_t)N * K * sizeof(float), cudaMemcpyDeviceToHost), "rmsnorm_vec4 D2H");
  max_err = 0.0f;
  for (int i = 0; i < N * K; i++) {
    float err = fabsf(h_y[i] - h_y_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "RMSNorm-Vec4", max_err);

  free(h_x); free(h_y); free(h_y_ref);
  cudaFree(d_x); cudaFree(d_y);
}


static void test_layer_norm(int N, int K) {

  srand(42);
  float *h_x = (float *)malloc((size_t)N * K * sizeof(float));
  float *h_y_ref = (float *)malloc((size_t)N * K * sizeof(float));
  for (int i = 0; i < N * K; i++)
    h_x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  float g = 1.5f, b = 0.3f;  // gain and bias

  // CPU reference: y = ((x - mean) / std) * g + b
  float epsilon = 1e-5f;
  for (int n = 0; n < N; n++) {
    double sum = 0.0;
    for (int k = 0; k < K; k++) sum += (double)h_x[n * K + k];
    float mean = (float)sum / (float)K;
    double sum_sq = 0.0;
    for (int k = 0; k < K; k++) {
      float diff = h_x[n * K + k] - mean;
      sum_sq += (double)diff * (double)diff;
    }
    float std = sqrtf((float)sum_sq / (float)K + epsilon);
    for (int k = 0; k < K; k++)
      h_y_ref[n * K + k] = ((h_x[n * K + k] - mean) / std) * g + b;
  }

  float *d_x, *d_y;
  check(cudaMalloc(&d_x, (size_t)N * K * sizeof(float)), "layernorm alloc X");
  check(cudaMalloc(&d_y, (size_t)N * K * sizeof(float)), "layernorm alloc Y");
  check(cudaMemcpy(d_x, h_x, (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice), "layernorm H2D X");

  dim3 block(128);
  dim3 grid(N);
  layer_norm<<<grid, block>>>(d_x, d_y, g, b, N, K);
  check(cudaGetLastError(), "layernorm launch");
  check(cudaDeviceSynchronize(), "layernorm sync");

  float *h_y = (float *)malloc((size_t)N * K * sizeof(float));
  check(cudaMemcpy(h_y, d_y, (size_t)N * K * sizeof(float), cudaMemcpyDeviceToHost), "layernorm D2H");

  float max_err = 0.0f;
  for (int i = 0; i < N * K; i++) {
    float err = fabsf(h_y[i] - h_y_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "LayerNorm", max_err);

  // ---- Layer Norm Vec4 ----
  dim3 block_lv4(32);
  layer_norm_vec4<<<grid, block_lv4>>>(d_x, d_y, g, b, N, K);
  check(cudaGetLastError(), "layernorm_vec4 launch");
  check(cudaDeviceSynchronize(), "layernorm_vec4 sync");
  check(cudaMemcpy(h_y, d_y, (size_t)N * K * sizeof(float), cudaMemcpyDeviceToHost), "layernorm_vec4 D2H");
  max_err = 0.0f;
  for (int i = 0; i < N * K; i++) {
    float err = fabsf(h_y[i] - h_y_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "LayerNorm-Vec4", max_err);

  free(h_x); free(h_y); free(h_y_ref);
  cudaFree(d_x); cudaFree(d_y);
}


static void test_rope(int seq_len, int N) {

  int total_pairs = seq_len * N;
  int total_elems = total_pairs * 2;
  size_t size = (size_t)total_elems * sizeof(float);

  float *h_x = (float *)malloc(size);
  float *h_y_ref = (float *)malloc(size);

  srand(42);
  for (int i = 0; i < total_elems; i++)
    h_x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  // CPU reference: 2D rotation for each pair
  for (int idx = 0; idx < total_pairs; idx++) {
    int token_pos = idx / N;
    int token_idx = idx % N;
    float x1 = h_x[idx * 2];
    float x2 = h_x[idx * 2 + 1];
    float theta = 1.0f / powf(10000.0f, 2.0f * token_idx / (N * 2.0f));
    float angle = (float)token_pos * theta;
    float cos_v = cosf(angle);
    float sin_v = sinf(angle);
    h_y_ref[idx * 2] = x1 * cos_v - x2 * sin_v;
    h_y_ref[idx * 2 + 1] = x1 * sin_v + x2 * cos_v;
  }

  float *d_x, *d_y;
  check(cudaMalloc(&d_x, size), "rope alloc X");
  check(cudaMalloc(&d_y, size), "rope alloc Y");
  check(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice), "rope H2D");

  dim3 block(256);
  dim3 grid((total_pairs + 255) / 256);
  rope<<<grid, block>>>(d_x, d_y, seq_len, N);
  check(cudaGetLastError(), "rope launch");
  check(cudaDeviceSynchronize(), "rope sync");

  float *h_y = (float *)malloc(size);
  check(cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost), "rope D2H");

  float max_err = 0.0f;
  for (int i = 0; i < total_elems; i++) {
    float err = fabsf(h_y[i] - h_y_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "RoPE", max_err);

  free(h_x);
  free(h_y);
  free(h_y_ref);
  cudaFree(d_x);
  cudaFree(d_y);
}


static void test_mat_transpose(int row, int col) {

  size_t size_in = (size_t)row * col * sizeof(float);
  size_t size_out = (size_t)col * row * sizeof(float);

  float *h_x = (float *)malloc(size_in);
  float *h_y_ref = (float *)malloc(size_out);

  srand(42);
  for (int i = 0; i < row * col; i++)
    h_x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  // CPU reference: y[j][i] = x[i][j] (row-major)
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      h_y_ref[j * row + i] = h_x[i * col + j];

  float *d_x, *d_y;
  check(cudaMalloc(&d_x, size_in), "mattrans alloc X");
  check(cudaMalloc(&d_y, size_out), "mattrans alloc Y");
  check(cudaMemcpy(d_x, h_x, size_in, cudaMemcpyHostToDevice), "mattrans H2D");

  dim3 block(16, 16);
  dim3 grid((col + 15) / 16, (row + 15) / 16);
  mat_transpose<<<grid, block>>>(d_x, d_y, row, col);
  check(cudaGetLastError(), "mattrans launch");
  check(cudaDeviceSynchronize(), "mattrans sync");

  float *h_y = (float *)malloc(size_out);
  check(cudaMemcpy(h_y, d_y, size_out, cudaMemcpyDeviceToHost), "mattrans D2H");

  float max_err = 0.0f;
  for (int i = 0; i < col * row; i++) {
    float err = fabsf(h_y[i] - h_y_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "MatTranspose", max_err);

  free(h_x);
  free(h_y);
  free(h_y_ref);
  cudaFree(d_x);
  cudaFree(d_y);
}


static void test_mat_transpose_padded(int row, int col) {

  size_t size_in = (size_t)row * col * sizeof(float);
  size_t size_out = (size_t)col * row * sizeof(float);

  float *h_x = (float *)malloc(size_in);
  float *h_y_ref = (float *)malloc(size_out);

  srand(42);
  for (int i = 0; i < row * col; i++)
    h_x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  // CPU reference: y[j][i] = x[i][j] (row-major)
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      h_y_ref[j * row + i] = h_x[i * col + j];

  float *d_x, *d_y;
  check(cudaMalloc(&d_x, size_in), "mattrans_padded alloc X");
  check(cudaMalloc(&d_y, size_out), "mattrans_padded alloc Y");
  check(cudaMemcpy(d_x, h_x, size_in, cudaMemcpyHostToDevice), "mattrans_padded H2D");

  dim3 block(16, 16);
  dim3 grid((col + 15) / 16, (row + 63) / 64);
  mat_transpose_padded<<<grid, block>>>(d_x, d_y, row, col);
  check(cudaGetLastError(), "mattrans_padded launch");
  check(cudaDeviceSynchronize(), "mattrans_padded sync");

  float *h_y = (float *)malloc(size_out);
  check(cudaMemcpy(h_y, d_y, size_out, cudaMemcpyDeviceToHost), "mattrans_padded D2H");

  float max_err = 0.0f;
  for (int i = 0; i < col * row; i++) {
    float err = fabsf(h_y[i] - h_y_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "MatTransposePadded", max_err);

  free(h_x);
  free(h_y);
  free(h_y_ref);
  cudaFree(d_x);
  cudaFree(d_y);
}


static void test_sgemv(int M, int K) {

  srand(42);
  float *h_a = (float *)malloc((size_t)M * K * sizeof(float));
  float *h_x = (float *)malloc((size_t)K * sizeof(float));
  float *h_y_ref = (float *)malloc((size_t)M * sizeof(float));
  for (int i = 0; i < M * K; i++) h_a[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  for (int i = 0; i < K; i++) h_x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  // CPU reference: y[m] = sum_k A[m,k] * x[k]
  for (int m = 0; m < M; m++) {
    double sum = 0.0;
    for (int k = 0; k < K; k++) sum += (double)h_a[m * K + k] * (double)h_x[k];
    h_y_ref[m] = (float)sum;
  }

  float *d_a, *d_x, *d_y;
  check(cudaMalloc(&d_a, (size_t)M * K * sizeof(float)), "sgemv alloc A");
  check(cudaMalloc(&d_x, (size_t)K * sizeof(float)), "sgemv alloc X");
  check(cudaMalloc(&d_y, (size_t)M * sizeof(float)), "sgemv alloc Y");

  check(cudaMemcpy(d_a, h_a, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice), "sgemv H2D A");
  check(cudaMemcpy(d_x, h_x, (size_t)K * sizeof(float), cudaMemcpyHostToDevice), "sgemv H2D X");

  dim3 block(32, 4);
  dim3 grid((M + 3) / 4);
  sgemv_k128<<<grid, block>>>(d_a, d_x, d_y, M, K);
  check(cudaGetLastError(), "sgemv launch");
  check(cudaDeviceSynchronize(), "sgemv sync");

  float *h_y = (float *)malloc((size_t)M * sizeof(float));
  check(cudaMemcpy(h_y, d_y, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost), "sgemv D2H");

  float max_err = 0.0f;
  for (int m = 0; m < M; m++) {
    float err = fabsf(h_y[m] - h_y_ref[m]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "SGEMV-K128", max_err);

  // ---- SGEMV K32 ----
  check(cudaMemset(d_y, 0, M * sizeof(float)), "sgemv_k32 zero Y");
  sgemv_k32<<<grid, block>>>(d_a, d_x, d_y, M, K);
  check(cudaGetLastError(), "sgemv_k32 launch");
  check(cudaDeviceSynchronize(), "sgemv_k32 sync");

  check(cudaMemcpy(h_y, d_y, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost), "sgemv_k32 D2H");

  max_err = 0.0f;
  for (int m = 0; m < M; m++) {
    float err = fabsf(h_y[m] - h_y_ref[m]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "SGEMV-K32", max_err);

  // ---- SGEMV K16 ----
  free(h_a); free(h_x); free(h_y); free(h_y_ref);
  cudaFree(d_a); cudaFree(d_x); cudaFree(d_y);

  int K16 = 16;
  h_a = (float *)malloc((size_t)M * K16 * sizeof(float));
  h_x = (float *)malloc((size_t)K16 * sizeof(float));
  h_y_ref = (float *)malloc((size_t)M * sizeof(float));
  for (int i = 0; i < M * K16; i++) h_a[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  for (int i = 0; i < K16; i++) h_x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  for (int m = 0; m < M; m++) {
    double sum = 0.0;
    for (int k = 0; k < K16; k++) sum += (double)h_a[m * K16 + k] * (double)h_x[k];
    h_y_ref[m] = (float)sum;
  }

  check(cudaMalloc(&d_a, (size_t)M * K16 * sizeof(float)), "sgemv_k16 alloc A");
  check(cudaMalloc(&d_x, (size_t)K16 * sizeof(float)), "sgemv_k16 alloc X");
  check(cudaMalloc(&d_y, (size_t)M * sizeof(float)), "sgemv_k16 alloc Y");

  check(cudaMemcpy(d_a, h_a, (size_t)M * K16 * sizeof(float), cudaMemcpyHostToDevice), "sgemv_k16 H2D A");
  check(cudaMemcpy(d_x, h_x, (size_t)K16 * sizeof(float), cudaMemcpyHostToDevice), "sgemv_k16 H2D X");

  dim3 grid_k16((M + 7) / 8);
  sgemv_k16<2><<<grid_k16, block>>>(d_a, d_x, d_y, M, K16);
  check(cudaGetLastError(), "sgemv_k16 launch");
  check(cudaDeviceSynchronize(), "sgemv_k16 sync");

  h_y = (float *)malloc((size_t)M * sizeof(float));
  check(cudaMemcpy(h_y, d_y, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost), "sgemv_k16 D2H");

  max_err = 0.0f;
  for (int m = 0; m < M; m++) {
    float err = fabsf(h_y[m] - h_y_ref[m]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "SGEMV-K16", max_err);

  free(h_a); free(h_x); free(h_y); free(h_y_ref);
  cudaFree(d_a); cudaFree(d_x); cudaFree(d_y);
}


static void test_sgemm(int M, int N, int K) {

  size_t size_a = (size_t)M * K * sizeof(float);
  size_t size_b = (size_t)K * N * sizeof(float);
  size_t size_c = (size_t)M * N * sizeof(float);

  float *h_a = (float *)malloc(size_a);
  float *h_b = (float *)malloc(size_b);
  float *h_c_ref = (float *)malloc(size_c);

  srand(42);
  for (int i = 0; i < M * K; i++) h_a[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  for (int i = 0; i < K * N; i++) h_b[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  float *d_a, *d_b, *d_c;
  check(cudaMalloc(&d_a, size_a), "sgemm alloc A");
  check(cudaMalloc(&d_b, size_b), "sgemm alloc B");
  check(cudaMalloc(&d_c, size_c), "sgemm alloc C");

  check(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice), "sgemm H2D A");
  check(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice), "sgemm H2D B");

  // cuBLAS reference (row-major idiom: swap M/N, swap A/B)
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0f, beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
              &alpha, d_b, N, d_a, K, &beta, d_c, N);
  check(cudaMemcpy(h_c_ref, d_c, size_c, cudaMemcpyDeviceToHost), "sgemm D2H ref");

  // Kernel
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  dim3 block(32, 32);
  dim3 grid((N + 31) / 32, (M + 31) / 32);
  sgemm<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
  check(cudaGetLastError(), "sgemm launch");
  check(cudaDeviceSynchronize(), "sgemm sync");

  float *h_c = (float *)malloc(size_c);
  check(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost), "sgemm D2H");

  // Verify
  float max_err = 0.0f;
  for (int i = 0; i < M * N; i++) {
    float err = fabsf(h_c[i] - h_c_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "SGEMM", max_err);

  // ---- SGEMM Vec4 (128×128 tile, 4×4 thread tile) ----

  dim3 grid_vec4((N + 127) / 128, (M + 127) / 128);
  check(cudaMemset(d_c, 0, size_c), "sgemm_vec4 zero C");
  sgemm_vec4<<<grid_vec4, block>>>(d_a, d_b, d_c, M, N, K);
  check(cudaGetLastError(), "sgemm_vec4 launch");
  check(cudaDeviceSynchronize(), "sgemm_vec4 sync");

  check(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost), "sgemm_vec4 D2H");

  max_err = 0.0f;
  for (int i = 0; i < M * N; i++) {
    float err = fabsf(h_c[i] - h_c_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "SGEMM-Vec4", max_err);

  free(h_a); free(h_b); free(h_c); free(h_c_ref);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  cublasDestroy(handle);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}


static void test_hgemm_mma(int M, int N, int K) {

  size_t size_a = (size_t)M * K * sizeof(half);
  size_t size_b = (size_t)K * N * sizeof(half);
  size_t size_c = (size_t)M * N * sizeof(half);

  half *h_a = (half *)malloc(size_a);
  half *h_b = (half *)malloc(size_b);
  half *h_c_ref = (half *)malloc(size_c);

  srand(42);
  for (int i = 0; i < M * K; i++) h_a[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  for (int i = 0; i < K * N; i++) h_b[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);

  // Kernel expects B^T [N×K] row-major layout (TN layout convention).
  // We store h_b as B [K×N] row-major for cuBLAS, then create B^T for kernel.
  size_t size_b_t = (size_t)N * K * sizeof(half);
  half *h_b_t = (half *)malloc(size_b_t);
  for (int n = 0; n < N; n++)
    for (int k = 0; k < K; k++)
      h_b_t[n * K + k] = h_b[k * N + n];

  half *d_a, *d_b, *d_b_t, *d_c;
  check(cudaMalloc(&d_a, size_a), "hgemm alloc A");
  check(cudaMalloc(&d_b, size_b), "hgemm alloc B (cuBLAS)");
  check(cudaMalloc(&d_b_t, size_b_t), "hgemm alloc B_t (kernel)");
  check(cudaMalloc(&d_c, size_c), "hgemm alloc C");

  check(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice), "hgemm H2D A");
  check(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice), "hgemm H2D B (cuBLAS)");
  check(cudaMemcpy(d_b_t, h_b_t, size_b_t, cudaMemcpyHostToDevice), "hgemm H2D B_t (kernel)");

  // cuBLAS FP16 reference (row-major idiom: swap M/N, swap A/B)
  // Note: use CUBLAS_COMPUTE_16F to match the kernel's f16.f16.f16.f16 accumulation.
  cublasHandle_t handle;
  cublasCreate(&handle);
  half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
               &alpha_h, d_b, CUDA_R_16F, N, d_a, CUDA_R_16F, K,
               &beta_h, d_c, CUDA_R_16F, N,
               CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
  check(cudaMemcpy(h_c_ref, d_c, size_c, cudaMemcpyDeviceToHost), "hgemm D2H ref");

  // MMA kernel (TN layout: A row-major, B^T row-major = B col-major)
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  constexpr int BM = 128, BN = 128, BK = 16, kStages = 3;
  size_t smem_bytes = kStages * (BM * BK + BN * BK) * sizeof(half); // 24576
  dim3 block(256);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  hgemm_mma_stages_tn<<<grid, block, smem_bytes>>>(d_a, d_b_t, d_c, M, N, K);
  check(cudaGetLastError(), "hgemm launch");
  check(cudaDeviceSynchronize(), "hgemm sync");

  half *h_c = (half *)malloc(size_c);
  check(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost), "hgemm D2H");

  // Verify
  float max_err = 0.0f;
  for (int i = 0; i < M * N; i++) {
    float err = fabsf(__half2float(h_c[i]) - __half2float(h_c_ref[i]));
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "HGEMM MMA", max_err);

  free(h_a); free(h_b); free(h_b_t); free(h_c); free(h_c_ref);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_b_t); cudaFree(d_c);
  cublasDestroy(handle);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}


static void test_hgemm_swizzle(int M, int N, int K) {
  // HGEMM MMA Swizzle — m16n8k16 + multistage pipeline + TN 布局 + XOR swizzle
  //   + Register Double Buffering (kValTileK=4, BK=64)
  // TN layout: C[M×N] = A[M×K] × B^T[N×K]
  // Kernel: hgemm_mma_stages_tn_swizzle with default template params
  //   (kValTileK=4, kStages=2, BK=64)
  // smem: kStages × (BM×BK + BN×BK) halfs

  size_t size_a = (size_t)M * K * sizeof(half);
  size_t size_b = (size_t)K * N * sizeof(half);
  size_t size_c = (size_t)M * N * sizeof(half);

  half *h_a = (half *)malloc(size_a);
  half *h_b = (half *)malloc(size_b);
  half *h_c_ref = (half *)malloc(size_c);

  srand(42);
  for (int i = 0; i < M * K; i++) h_a[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  for (int i = 0; i < K * N; i++) h_b[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);

  // Kernel expects B^T [N×K] row-major layout (TN layout convention).
  size_t size_b_t = (size_t)N * K * sizeof(half);
  half *h_b_t = (half *)malloc(size_b_t);
  for (int n = 0; n < N; n++)
    for (int k = 0; k < K; k++)
      h_b_t[n * K + k] = h_b[k * N + n];

  half *d_a, *d_b, *d_b_t, *d_c;
  check(cudaMalloc(&d_a, size_a), "hgemm_swizzle alloc A");
  check(cudaMalloc(&d_b, size_b), "hgemm_swizzle alloc B (cuBLAS)");
  check(cudaMalloc(&d_b_t, size_b_t), "hgemm_swizzle alloc B_t (kernel)");
  check(cudaMalloc(&d_c, size_c), "hgemm_swizzle alloc C");

  check(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice), "hgemm_swizzle H2D A");
  check(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice), "hgemm_swizzle H2D B (cuBLAS)");
  check(cudaMemcpy(d_b_t, h_b_t, size_b_t, cudaMemcpyHostToDevice), "hgemm_swizzle H2D B_t (kernel)");

  // cuBLAS FP16 reference
  cublasHandle_t handle;
  cublasCreate(&handle);
  half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
               &alpha_h, d_b, CUDA_R_16F, N, d_a, CUDA_R_16F, K,
               &beta_h, d_c, CUDA_R_16F, N,
               CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
  check(cudaMemcpy(h_c_ref, d_c, size_c, cudaMemcpyDeviceToHost), "hgemm_swizzle D2H ref");

  // MMA swizzle kernel (default params: kStages=2, kValTileK=4, BK=64)
  constexpr int BM = 128, BN = 128, BK = 64, K_STAGE_S = 2;
  size_t smem_bytes = K_STAGE_S * (BM * BK + BN * BK) * sizeof(half);
  cudaFuncSetAttribute(
      (const void *)hgemm_mma_stages_tn_swizzle<16, 8, 16, 2, 4, 4, 4, 4, K_STAGE_S, 0>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  dim3 block(256);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  hgemm_mma_stages_tn_swizzle<<<grid, block, smem_bytes>>>(d_a, d_b_t, d_c, M, N, K);
  check(cudaGetLastError(), "hgemm_swizzle launch");
  check(cudaDeviceSynchronize(), "hgemm_swizzle sync");

  half *h_c = (half *)malloc(size_c);
  check(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost), "hgemm_swizzle D2H");

  // Verify
  float max_err = 0.0f;
  for (int i = 0; i < M * N; i++) {
    float err = fabsf(__half2float(h_c[i]) - __half2float(h_c_ref[i]));
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "HGEMM Swizzle + Reg2x", max_err);

  free(h_a); free(h_b); free(h_b_t); free(h_c); free(h_c_ref);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_b_t); cudaFree(d_c);
  cublasDestroy(handle);
}


#if defined(NOTES_V2_ENABLE_CUTE)
static void test_hgemm_cute(int M, int N, int K) {
  // HGEMM CuTe — SM80_16x8x16_F16F16F16F16_TN + Swizzle<3,3,3> + kStage=2
  // TN layout: C[M×N] = A[M×K] × B^T[N×K]
  // Kernel: hgemm_mma_stages_tn_cute via launch_hgemm_mma_stages_tn_cute
  // Tile: BM=128, BN=256, BK=32, 128 threads/block

  size_t size_a = (size_t)M * K * sizeof(half);
  size_t size_b = (size_t)K * N * sizeof(half);
  size_t size_c = (size_t)M * N * sizeof(half);

  half *h_a = (half *)malloc(size_a);
  half *h_b = (half *)malloc(size_b);
  half *h_c_ref = (half *)malloc(size_c);

  srand(42);
  for (int i = 0; i < M * K; i++)
    h_a[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  for (int i = 0; i < K * N; i++)
    h_b[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);

  // CuTe kernel expects B^T [N×K] row-major (TN layout).
  size_t size_b_t = (size_t)N * K * sizeof(half);
  half *h_b_t = (half *)malloc(size_b_t);
  for (int n = 0; n < N; n++)
    for (int k = 0; k < K; k++)
      h_b_t[n * K + k] = h_b[k * N + n];

  half *d_a, *d_b, *d_b_t, *d_c;
  check(cudaMalloc(&d_a, size_a), "hgemm_cute alloc A");
  check(cudaMalloc(&d_b, size_b), "hgemm_cute alloc B (cuBLAS)");
  check(cudaMalloc(&d_b_t, size_b_t), "hgemm_cute alloc B_t (kernel)");
  check(cudaMalloc(&d_c, size_c), "hgemm_cute alloc C");

  check(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice), "hgemm_cute H2D A");
  check(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice), "hgemm_cute H2D B (cuBLAS)");
  check(cudaMemcpy(d_b_t, h_b_t, size_b_t, cudaMemcpyHostToDevice), "hgemm_cute H2D B_t (kernel)");

  // cuBLAS FP16 reference (row-major idiom: swap M/N, swap A/B)
  cublasHandle_t handle;
  cublasCreate(&handle);
  half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h, d_b,
               CUDA_R_16F, N, d_a, CUDA_R_16F, K, &beta_h, d_c, CUDA_R_16F, N,
               CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
  check(cudaMemcpy(h_c_ref, d_c, size_c, cudaMemcpyDeviceToHost), "hgemm_cute D2H ref");

  // CuTe kernel (Stages=2, TN layout)
  launch_hgemm_mma_stages_tn_cute<half, 2, 0>(d_a, d_b_t, d_c, M, N, K);
  check(cudaGetLastError(), "hgemm_cute launch");
  check(cudaDeviceSynchronize(), "hgemm_cute sync");

  half *h_c = (half *)malloc(size_c);
  check(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost), "hgemm_cute D2H");

  // Verify
  float max_err = 0.0f;
  for (int i = 0; i < M * N; i++) {
    float err = fabsf(__half2float(h_c[i]) - __half2float(h_c_ref[i]));
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "HGEMM CuTe Swizzle + Reg2x", 
         max_err);

  free(h_a); free(h_b); free(h_b_t); free(h_c); free(h_c_ref);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_b_t); cudaFree(d_c);
  cublasDestroy(handle);
}
#endif /* NOTES_V2_ENABLE_CUTE */


#if defined(NOTES_V2_ENABLE_WGMMA)
static void test_hgemm_wgmma(int M, int N, int K) {
  // HGEMM WGMMA — m64n128k16 + TMA + Warp Specialization (Hopper SM90+)
  // TN layout: C[M×N] = A[M×K] × B^T[N×K]
  // Kernel: hgemm_wgmma_stages_tn with default template params

  constexpr int BM = 128, BN = 128, BK = 64, kStages = 3, kNumThreads = 256;

  // M, K must be divisible by tile dims
  if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
    printf("| %-56s | %-9s |\n", "HGEMM WGMMA", "SKIP");
    return;
  }

  size_t size_a = (size_t)M * K * sizeof(half);
  size_t size_b = (size_t)K * N * sizeof(half);
  size_t size_c = (size_t)M * N * sizeof(half);

  half *h_a = (half *)malloc(size_a);
  half *h_b = (half *)malloc(size_b);
  half *h_c_ref = (half *)malloc(size_c);

  srand(42);
  for (int i = 0; i < M * K; i++)
    h_a[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  for (int i = 0; i < K * N; i++)
    h_b[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);

  // B^T [N×K] row-major for TN layout (same as hgemm_mma kernel)
  size_t size_b_t = (size_t)N * K * sizeof(half);
  half *h_b_t = (half *)malloc(size_b_t);
  for (int n = 0; n < N; n++)
    for (int k = 0; k < K; k++)
      h_b_t[n * K + k] = h_b[k * N + n];

  half *d_a, *d_b, *d_b_t, *d_c;
  check(cudaMalloc(&d_a, size_a), "wgmma alloc A");
  check(cudaMalloc(&d_b, size_b), "wgmma alloc B (cuBLAS)");
  check(cudaMalloc(&d_b_t, size_b_t), "wgmma alloc B_t (kernel)");
  check(cudaMalloc(&d_c, size_c), "wgmma alloc C");

  check(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice), "wgmma H2D A");
  check(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice),
        "wgmma H2D B (cuBLAS)");
  check(cudaMemcpy(d_b_t, h_b_t, size_b_t, cudaMemcpyHostToDevice),
        "wgmma H2D B_t (kernel)");

  // cuBLAS FP16 reference (row-major idiom, same as hgemm_mma)
  cublasHandle_t handle;
  cublasCreate(&handle);
  half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h, d_b,
               CUDA_R_16F, N, d_a, CUDA_R_16F, K, &beta_h, d_c, CUDA_R_16F, N,
               CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
  check(cudaMemcpy(h_c_ref, d_c, size_c, cudaMemcpyDeviceToHost),
        "wgmma D2H ref");

  // Create TMA tensor maps for A and B^T
  // A[M×K] row-major: TMA box=(BK=64, BM=128), global shape=(K, M)
  // B^T[N×K] row-major: TMA box=(BK=64, BN=128), global shape=(K, N)
  CUtensorMap *tma_a =
      allocate_and_create_tensor_map(d_a, M / BM, K / BK);
  CUtensorMap *tma_b =
      allocate_and_create_tensor_map(d_b_t, N / BN, K / BK);

  // Launch WGMMA kernel
  // kBlockSwizzle=false → 2D grid, no swizzle
  size_t smem_bytes =
      kStages * (BM * BK + BN * BK) * sizeof(half); // 3*16384*2 = 96KB
  cudaFuncSetAttribute(
      hgemm_wgmma_stages_tn<64, 128, 16, BM, BN, BK, kNumThreads, kStages,
                             false>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

  dim3 block(kNumThreads);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  hgemm_wgmma_stages_tn<64, 128, 16, BM, BN, BK, kNumThreads, kStages, false>
      <<<grid, block, smem_bytes>>>(M, N, K, d_c, tma_a, tma_b);
  check(cudaGetLastError(), "wgmma launch");
  check(cudaDeviceSynchronize(), "wgmma sync");

  half *h_c = (half *)malloc(size_c);
  check(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost), "wgmma D2H");

  // Verify
  float max_err = 0.0f;
  for (int i = 0; i < M * N; i++) {
    float err = fabsf(__half2float(h_c[i]) - __half2float(h_c_ref[i]));
    if (err > max_err)
      max_err = err;
  }
  printf("| %-56s | %.3e |\n", "HGEMM TMA WGMMA WS (3-stage)", max_err);

  free(h_a);
  free(h_b);
  free(h_b_t);
  free(h_c);
  free(h_c_ref);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_b_t);
  cudaFree(d_c);
  cudaFree(tma_a);
  cudaFree(tma_b);
  cublasDestroy(handle);
}
#endif /* NOTES_V2_ENABLE_WGMMA */

#if defined(NOTES_V2_ENABLE_TMA_MMA_WS)
template <int kStages, int kBlockSwizzle = 0>
static bool launch_hgemm_tma_mma_ws(int M, int N, int K, half *d_a,
                                    half *d_b_t, half *d_c,
                                    CUtensorMap *tma_a, CUtensorMap *tma_b) {
  constexpr int kMmaM = 16, kMmaN = 8, kMmaK = 16;
  constexpr int kMmaTileM = 2, kMmaTileN = 2;
  constexpr int kValTileM = 4, kValTileN = 8, kValTileK = 4;
  constexpr int BM = kMmaM * kMmaTileM * kValTileM;
  constexpr int BN = kMmaN * kMmaTileN * kValTileN;
  constexpr int BK = kMmaK * kValTileK;
  constexpr int kNumThreads = 256;
  constexpr size_t payload_bytes =
      kStages * (BM * BK + BN * BK) * sizeof(half);
  constexpr size_t smem_bytes = payload_bytes;
  using Kernel = void (*)(int, int, int, half *, const CUtensorMap *,
                          const CUtensorMap *);
  Kernel kernel = hgemm_tma_mma_ws_tn<
      kMmaM, kMmaN, kMmaK, kMmaTileM, kMmaTileN, kValTileM, kValTileN,
      kValTileK, kStages, kNumThreads, kBlockSwizzle>;

  int device = 0;
  int max_smem = 0;
  cudaFuncAttributes attributes{};
  check(cudaGetDevice(&device), "tma_mma_ws get device");
  check(cudaDeviceGetAttribute(&max_smem,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               device),
        "tma_mma_ws max shared memory");
  check(cudaFuncGetAttributes(&attributes, kernel),
        "tma_mma_ws function attributes");
  if (smem_bytes + attributes.sharedSizeBytes > size_t(max_smem)) {
    if (g_debug)
      printf("| %-56s | %-9s |\n",
             kStages == 2 ? "HGEMM TMA MMA WS (S=2, BLK_SW=0)"
                          : "HGEMM TMA MMA WS (S=3, BLK_SW=0)",
             "SMEM SKIP");
    return false;
  }

  check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes),
        "tma_mma_ws set dynamic shared memory");
  dim3 block(kNumThreads);
  constexpr int kSwizzleN = 16;
  const int n_tiles = N / BN;
  const int grid_x = kBlockSwizzle ? div_ceil(n_tiles, kSwizzleN) : n_tiles;
  const int grid_z = kBlockSwizzle ? kSwizzleN : 1;
  dim3 grid(grid_x, M / BM, grid_z);
  kernel<<<grid, block, smem_bytes>>>(M, N, K, d_c, tma_a, tma_b);
  check(cudaGetLastError(), "tma_mma_ws launch");
  check(cudaDeviceSynchronize(), "tma_mma_ws sync");
  return true;
}

static void test_hgemm_tma_mma_ws(int M, int N, int K) {
  constexpr int BM = 128, BN = 128, BK = 64;
  if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
    printf("| %-56s | %-9s |\n", "HGEMM TMA MMA WS", "SKIP");
    return;
  }

  const size_t size_a = size_t(M) * K * sizeof(half);
  const size_t size_b = size_t(K) * N * sizeof(half);
  const size_t size_b_t = size_t(N) * K * sizeof(half);
  const size_t size_c = size_t(M) * N * sizeof(half);
  half *h_a = static_cast<half *>(malloc(size_a));
  half *h_b = static_cast<half *>(malloc(size_b));
  half *h_b_t = static_cast<half *>(malloc(size_b_t));
  half *h_c = static_cast<half *>(malloc(size_c));
  half *h_c_ref = static_cast<half *>(malloc(size_c));
  srand(42);
  for (int i = 0; i < M * K; ++i)
    h_a[i] = __float2half((float(rand()) / RAND_MAX) * 2.0f - 1.0f);
  for (int i = 0; i < K * N; ++i)
    h_b[i] = __float2half((float(rand()) / RAND_MAX) * 2.0f - 1.0f);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      h_b_t[n * K + k] = h_b[k * N + n];

  half *d_a, *d_b, *d_b_t, *d_c;
  check(cudaMalloc(&d_a, size_a), "tma_mma_ws alloc A");
  check(cudaMalloc(&d_b, size_b), "tma_mma_ws alloc B");
  check(cudaMalloc(&d_b_t, size_b_t), "tma_mma_ws alloc B_t");
  check(cudaMalloc(&d_c, size_c), "tma_mma_ws alloc C");
  check(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice), "tma_mma_ws H2D A");
  check(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice), "tma_mma_ws H2D B");
  check(cudaMemcpy(d_b_t, h_b_t, size_b_t, cudaMemcpyHostToDevice),
        "tma_mma_ws H2D B_t");

  cublasHandle_t handle;
  cublasCreate(&handle);
  half alpha = __float2half(1.0f), beta = __float2half(0.0f);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b,
               CUDA_R_16F, N, d_a, CUDA_R_16F, K, &beta, d_c, CUDA_R_16F, N,
               CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
  check(cudaMemcpy(h_c_ref, d_c, size_c, cudaMemcpyDeviceToHost),
        "tma_mma_ws D2H reference");

  CUtensorMap *tma_a = allocate_and_create_tensor_map(d_a, M / BM, K / BK);
  CUtensorMap *tma_b = allocate_and_create_tensor_map(d_b_t, N / BN, K / BK);
  for (int block_swizzle : {0, 1}) {
    for (int stages : {2, 3}) {
      const bool launched = stages == 2
          ? (block_swizzle
                 ? launch_hgemm_tma_mma_ws<2, 1>(M, N, K, d_a, d_b_t, d_c,
                                                   tma_a, tma_b)
                 : launch_hgemm_tma_mma_ws<2>(M, N, K, d_a, d_b_t, d_c,
                                               tma_a, tma_b))
          : (block_swizzle
                 ? launch_hgemm_tma_mma_ws<3, 1>(M, N, K, d_a, d_b_t, d_c,
                                                   tma_a, tma_b)
                 : launch_hgemm_tma_mma_ws<3>(M, N, K, d_a, d_b_t, d_c,
                                               tma_a, tma_b));
      if (!launched)
        continue;
      check(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost),
            "tma_mma_ws D2H");
      float max_err = 0.0f;
      for (int i = 0; i < M * N; ++i)
        max_err = fmaxf(max_err, fabsf(__half2float(h_c[i]) -
                                       __half2float(h_c_ref[i])));
      printf("| %-56s | %.3e |\n",
             block_swizzle
                 ? (stages == 2 ? "HGEMM TMA MMA WS (S=2, BLK_SW=1)"
                                : "HGEMM TMA MMA WS (S=3, BLK_SW=1)")
                 : (stages == 2 ? "HGEMM TMA MMA WS (S=2, BLK_SW=0)"
                                : "HGEMM TMA MMA WS (S=3, BLK_SW=0)"),
             max_err);
    }
  }

  free(h_a); free(h_b); free(h_b_t); free(h_c); free(h_c_ref);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_b_t); cudaFree(d_c);
  cudaFree(tma_a); cudaFree(tma_b);
  cublasDestroy(handle);
}
#endif /* NOTES_V2_ENABLE_TMA_MMA_WS */


static void test_flash_attn(int seqlen, int head_dim) {
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
static void test_flash_attn_tma_mma_ws(int seqlen, int head_dim) {
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

static void test_flash_attn_3_tma_ws(int seqlen, int head_dim) {
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

static float bench_hgemm_tflops(int M, int N, int K, float time_ms) {
  double flops = 2.0 * M * N * K;
  return (float)(flops / (double)time_ms / 1e9);
}

static float bench_fa_tflops(int B, int H, int N, int D, float time_ms) {
  double flops = 4.0 * B * H * N * N * D;
  return (float)(flops / (double)time_ms / 1e9);
}

static float bench_cublas_hgemm_tflops(cublasHandle_t handle, int M, int N, int K,
                                       half *d_a, half *d_b, half *d_c) {
  half alpha = __float2half(1.0f), beta = __float2half(0.0f);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cublasSetStream(handle, stream);
  // warmup
  for (int w = 0; w < g_warmup; ++w)
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                 d_b, CUDA_R_16F, N, d_a, CUDA_R_16F, K, &beta,
                 d_c, CUDA_R_16F, N, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
  cudaStreamSynchronize(stream);
  // timed repeat
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  for (int r = 0; r < g_repeat; ++r)
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                 d_b, CUDA_R_16F, N, d_a, CUDA_R_16F, K, &beta,
                 d_c, CUDA_R_16F, N, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float time_ms = 0;
  cudaEventElapsedTime(&time_ms, start, stop);
  time_ms /= g_repeat;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cublasSetStream(handle, nullptr);
  cudaStreamDestroy(stream);
  return bench_hgemm_tflops(M, N, K, time_ms);
}

// =============================================================================
// Bench: HGEMM MMA (basic m16n8k16 + multistage pipeline)
// =============================================================================
template <int kStages, int kBlockSwizzle>
static bool launch_timed_hgemm_mma(
  half *d_a, half *d_b_t, half *d_c, half *h_c,
  half *h_c_ref, int M, int N, int K, size_t size_c,
  cudaEvent_t start, cudaEvent_t stop, float &max_err,
  float &time_ms
) {
  constexpr int BM = 128, BN = 128, BK = 16;
  using Kernel = void (*)(half *, half *, half *, int, int, int);
  Kernel k = hgemm_mma_stages_tn<16, 8, 16, 2, 4, 4, 4, kStages, kBlockSwizzle>;
  size_t smem = kStages * (BM * BK + BN * BK) * sizeof(half);
  if (!check_smem_feasible((const void *)k, smem)) return false;
  dim3 block(256);
  const int tiles_n = (N + BN - 1) / BN;
  constexpr int kSwizzleN = 16;
  int gx = kBlockSwizzle ? div_ceil(tiles_n, kSwizzleN) : tiles_n;
  int gz = kBlockSwizzle ? kSwizzleN : 1;
  dim3 grid(gx, (M + BM - 1) / BM, gz);
  for (int w = 0; w < g_warmup; w++)
    k<<<grid, block, smem>>>(d_a, d_b_t, d_c, M, N, K);
  cudaDeviceSynchronize();
  cudaEventRecord(start);
  for (int r = 0; r < g_repeat; r++)
    k<<<grid, block, smem>>>(d_a, d_b_t, d_c, M, N, K);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms, start, stop);
  time_ms /= g_repeat;
  check(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost), "bench mma D2H");
  max_err = 0.0f;
  for (int i = 0; i < M * N; i++) {
    float err = fabsf(__half2float(h_c[i]) - __half2float(h_c_ref[i]));
    if (err > max_err) max_err = err;
  }
  return true;
}

static void bench_hgemm_mma(int M, int N, int K) {
  size_t size_a = (size_t)M * K * sizeof(half);
  size_t size_b = (size_t)K * N * sizeof(half);
  size_t size_c = (size_t)M * N * sizeof(half);
  half *h_a = (half *)malloc(size_a);
  half *h_b = (half *)malloc(size_b);
  half *h_c_ref = (half *)malloc(size_c);
  srand(42);
  for (int i = 0; i < M * K; i++)
    h_a[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  for (int i = 0; i < K * N; i++)
    h_b[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  size_t size_b_t = (size_t)N * K * sizeof(half);
  half *h_b_t = (half *)malloc(size_b_t);
  for (int n = 0; n < N; n++)
    for (int k = 0; k < K; k++)
      h_b_t[n * K + k] = h_b[k * N + n];
  half *d_a, *d_b, *d_b_t, *d_c;
  check(cudaMalloc(&d_a, size_a), "bench mma alloc A");
  check(cudaMalloc(&d_b, size_b), "bench mma alloc B");
  check(cudaMalloc(&d_b_t, size_b_t), "bench mma alloc B_t");
  check(cudaMalloc(&d_c, size_c), "bench mma alloc C");
  check(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice), "bench mma H2D A");
  check(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice), "bench mma H2D B");
  check(cudaMemcpy(d_b_t, h_b_t, size_b_t, cudaMemcpyHostToDevice), "bench mma H2D B_t");
  cublasHandle_t handle;
  cublasCreate(&handle);
  half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h, d_b, CUDA_R_16F, N,
        d_a, CUDA_R_16F, K, &beta_h, d_c, CUDA_R_16F, N, CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT);
  check(cudaMemcpy(h_c_ref, d_c, size_c, cudaMemcpyDeviceToHost), "bench mma D2H ref");
  half *h_c = (half *)malloc(size_c);
  float cublas_tflops = bench_cublas_hgemm_tflops(handle, M, N, K, d_a, d_b, d_c);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for (int stages : {2, 3}) {
    for (int swizzle : {0, 1}) {
      float max_err = 0, time_ms = 0;
      bool ok = false;
      if (stages == 2)
        ok = swizzle ? launch_timed_hgemm_mma<2, 1>(d_a, d_b_t, d_c, h_c, h_c_ref, M, N,
            K, size_c, start, stop, max_err, time_ms)
          : launch_timed_hgemm_mma<2, 0>(d_a, d_b_t, d_c, h_c, h_c_ref, M, N,
            K, size_c, start, stop, max_err, time_ms);
      else
        ok = swizzle ? launch_timed_hgemm_mma<3, 1>(d_a, d_b_t, d_c, h_c, h_c_ref, M, N,
            K, size_c, start, stop, max_err, time_ms)
          : launch_timed_hgemm_mma<3, 0>(d_a, d_b_t, d_c, h_c, h_c_ref, M, N,
            K, size_c, start, stop, max_err, time_ms);
      char label[64];
      snprintf(label, sizeof(label), "HGEMM MMA (S=%d, BLK_SW=%d)", stages, swizzle);
      if (!ok) {
        if (g_debug)
          printf("| %-56s | %-9s | %-19s |\n", label, "SMEM SKIP", "None");
      } else {
        float tflops = bench_hgemm_tflops(M, N, K, time_ms);
        char tflops_str[32];
        snprintf(tflops_str, sizeof(tflops_str), "%.1f/%.1f (%.2fx)", 
                 tflops, cublas_tflops, tflops / cublas_tflops);
        if (should_print_hgemm_tflops(tflops))
          printf("| %-56s | %.3e | %-19s |\n", label, max_err,
          tflops_str);
      }
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(h_a);
  free(h_b);
  free(h_b_t);
  free(h_c);
  free(h_c_ref);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_b_t);
  cudaFree(d_c);
  cublasDestroy(handle);
}

// =============================================================================
// Bench: HGEMM MMA Swizzle + Register Double Buffering (kValTileK=4, BK=64)
// =============================================================================
template <int kStages, int kBlockSwizzle>
static bool launch_timed_hgemm_swizzle(
  half *d_a, half *d_b_t, half *d_c, half *h_c,
  half *h_c_ref, int M, int N, int K, size_t size_c,
  cudaEvent_t start, cudaEvent_t stop, float &max_err,
  float &time_ms
) {
  constexpr int BM = 128, BN = 128, BK = 64;
  using Kernel = void (*)(half *, half *, half *, int, int, int);
  Kernel k = hgemm_mma_stages_tn_swizzle<16, 8, 16, 2, 4, 4, 4, 4, kStages, kBlockSwizzle>;
  size_t smem = kStages * (BM * BK + BN * BK) * sizeof(half);
  if (!check_smem_feasible((const void *)k, smem)) return false;
  cudaFuncSetAttribute((const void *)k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 block(256);
  const int tiles_n = (N + BN - 1) / BN;
  constexpr int kSwizzleN = 16;
  int gx = kBlockSwizzle ? div_ceil(tiles_n, kSwizzleN) : tiles_n;
  int gz = kBlockSwizzle ? kSwizzleN : 1;
  dim3 grid(gx, (M + BM - 1) / BM, gz);
  for (int w = 0; w < g_warmup; w++)
    k<<<grid, block, smem>>>(d_a, d_b_t, d_c, M, N, K);
  cudaDeviceSynchronize();
  cudaEventRecord(start);
  for (int r = 0; r < g_repeat; r++)
    k<<<grid, block, smem>>>(d_a, d_b_t, d_c, M, N, K);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms, start, stop);
  time_ms /= g_repeat;
  check(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost), "bench swizzle D2H");
  max_err = 0.0f;
  for (int i = 0; i < M * N; i++) {
    float err = fabsf(__half2float(h_c[i]) - __half2float(h_c_ref[i]));
    if (err > max_err) max_err = err;
  }
  return true;
}

static void bench_hgemm_swizzle(int M, int N, int K) {
  size_t size_a = (size_t)M * K * sizeof(half);
  size_t size_b = (size_t)K * N * sizeof(half);
  size_t size_c = (size_t)M * N * sizeof(half);
  half *h_a = (half *)malloc(size_a);
  half *h_b = (half *)malloc(size_b);
  half *h_c_ref = (half *)malloc(size_c);
  srand(42);
  for (int i = 0; i < M * K; i++)
    h_a[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  for (int i = 0; i < K * N; i++)
    h_b[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  size_t size_b_t = (size_t)N * K * sizeof(half);
  half *h_b_t = (half *)malloc(size_b_t);
  for (int n = 0; n < N; n++)
    for (int k = 0; k < K; k++)
      h_b_t[n * K + k] = h_b[k * N + n];
  half *d_a, *d_b, *d_b_t, *d_c;
  check(cudaMalloc(&d_a, size_a), "bench swizzle alloc A");
  check(cudaMalloc(&d_b, size_b), "bench swizzle alloc B");
  check(cudaMalloc(&d_b_t, size_b_t), "bench swizzle alloc B_t");
  check(cudaMalloc(&d_c, size_c), "bench swizzle alloc C");
  check(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice), "bench swizzle H2D A");
  check(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice), "bench swizzle H2D B");
  check(cudaMemcpy(d_b_t, h_b_t, size_b_t, cudaMemcpyHostToDevice), "bench swizzle H2D B_t");
  cublasHandle_t handle;
  cublasCreate(&handle);
  half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h, d_b, CUDA_R_16F, N,
        d_a, CUDA_R_16F, K, &beta_h, d_c, CUDA_R_16F, N, CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT);
  check(cudaMemcpy(h_c_ref, d_c, size_c, cudaMemcpyDeviceToHost), "bench swizzle D2H ref");
  half *h_c = (half *)malloc(size_c);
  float cublas_tflops = bench_cublas_hgemm_tflops(handle, M, N, K, d_a, d_b, d_c);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for (int stages : {2, 3}) {
    for (int swizzle : {0, 1}) {
      float max_err = 0, time_ms = 0;
      bool ok = false;
      if (stages == 2)
        ok = swizzle ? launch_timed_hgemm_swizzle<2, 1>(d_a, d_b_t, d_c, h_c, h_c_ref, M,
            N, K, size_c, start, stop, max_err, time_ms)
          : launch_timed_hgemm_swizzle<2, 0>(d_a, d_b_t, d_c, h_c, h_c_ref, M,
            N, K, size_c, start, stop, max_err, time_ms);
      else
        ok = swizzle ? launch_timed_hgemm_swizzle<3, 1>(d_a, d_b_t, d_c, h_c, h_c_ref, M,
            N, K, size_c, start, stop, max_err, time_ms)
          : launch_timed_hgemm_swizzle<3, 0>(d_a, d_b_t, d_c, h_c, h_c_ref, M,
            N, K, size_c, start, stop, max_err, time_ms);
      char label[64];
      snprintf(label, sizeof(label), "HGEMM Swizzle+Reg2x (S=%d, BLK_SW=%d)", stages, swizzle);
      if (!ok) {
        if (g_debug)
          printf("| %-56s | %-9s | %-19s |\n", label, "SMEM SKIP", "None");
      } else {
        float tflops = bench_hgemm_tflops(M, N, K, time_ms);
        char tflops_str[32];
        snprintf(tflops_str, sizeof(tflops_str), "%.1f/%.1f (%.2fx)", 
                 tflops, cublas_tflops, tflops / cublas_tflops);
        if (should_print_hgemm_tflops(tflops))
          printf("| %-56s | %.3e | %-19s |\n", label, max_err,
          tflops_str);
      }
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(h_a);
  free(h_b);
  free(h_b_t);
  free(h_c);
  free(h_c_ref);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_b_t);
  cudaFree(d_c);
  cublasDestroy(handle);
}

#if defined(NOTES_V2_ENABLE_CUTE)
// =============================================================================
// Bench: HGEMM CuTe
// =============================================================================
static void bench_hgemm_cute(int M, int N, int K) {
  size_t size_a = (size_t)M * K * sizeof(half);
  size_t size_b = (size_t)K * N * sizeof(half);
  size_t size_c = (size_t)M * N * sizeof(half);
  half *h_a = (half *)malloc(size_a);
  half *h_b = (half *)malloc(size_b);
  half *h_c_ref = (half *)malloc(size_c);
  srand(42);
  for (int i = 0; i < M * K; i++)
    h_a[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  for (int i = 0; i < K * N; i++)
    h_b[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  size_t size_b_t = (size_t)N * K * sizeof(half);
  half *h_b_t = (half *)malloc(size_b_t);
  for (int n = 0; n < N; n++)
    for (int k = 0; k < K; k++)
      h_b_t[n * K + k] = h_b[k * N + n];
  half *d_a, *d_b, *d_b_t, *d_c;
  check(cudaMalloc(&d_a, size_a), "bench cute alloc A");
  check(cudaMalloc(&d_b, size_b), "bench cute alloc B");
  check(cudaMalloc(&d_b_t, size_b_t), "bench cute alloc B_t");
  check(cudaMalloc(&d_c, size_c), "bench cute alloc C");
  check(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice), "bench cute H2D A");
  check(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice), "bench cute H2D B");
  check(cudaMemcpy(d_b_t, h_b_t, size_b_t, cudaMemcpyHostToDevice), "bench cute H2D B_t");
  cublasHandle_t handle;
  cublasCreate(&handle);
  half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h, d_b, CUDA_R_16F, N,
        d_a, CUDA_R_16F, K, &beta_h, d_c, CUDA_R_16F, N, CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT);
  check(cudaMemcpy(h_c_ref, d_c, size_c, cudaMemcpyDeviceToHost), "bench cute D2H ref");
  half *h_c = (half *)malloc(size_c);
  float cublas_tflops = bench_cublas_hgemm_tflops(handle, M, N, K, d_a, d_b, d_c);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for (int stages : {2, 3}) {
    for (int swizzle : {0, 1}) {
      char label[64];
      snprintf(label, sizeof(label), "HGEMM CuTe Swizzle (S=%d, BLK_SW=%d)", stages, swizzle);
      bool ok = false;
      float time_ms = 0, max_err = 0;
      if (stages == 2) {
        auto k = swizzle ? launch_hgemm_mma_stages_tn_cute<half, 2, 1>
                         : launch_hgemm_mma_stages_tn_cute<half, 2, 0>;
        for (int w = 0; w < g_warmup; w++) k(d_a, d_b_t, d_c, M, N, K);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int r = 0; r < g_repeat; r++) k(d_a, d_b_t, d_c, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_ms, start, stop);
        time_ms /= g_repeat;
        ok = true;
      } else {
        auto k = swizzle ? launch_hgemm_mma_stages_tn_cute<half, 3, 1>
                         : launch_hgemm_mma_stages_tn_cute<half, 3, 0>;
        for (int w = 0; w < g_warmup; w++) k(d_a, d_b_t, d_c, M, N, K);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int r = 0; r < g_repeat; r++) k(d_a, d_b_t, d_c, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_ms, start, stop);
        time_ms /= g_repeat;
        ok = true;
      }
      if (ok) {
        check(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost), "bench cute D2H");
        max_err = 0.0f;
        for (int i = 0; i < M * N; i++) {
          float err = fabsf(__half2float(h_c[i]) - __half2float(h_c_ref[i]));
          if (err > max_err) max_err = err;
        }
        float tflops = bench_hgemm_tflops(M, N, K, time_ms);
        char tflops_str[32];
        snprintf(tflops_str, sizeof(tflops_str), "%.1f/%.1f (%.2fx)", 
                 tflops, cublas_tflops, tflops / cublas_tflops);
        if (should_print_hgemm_tflops(tflops))
          printf("| %-56s | %.3e | %-19s |\n", label, max_err,
          tflops_str);
      } else {
        printf("| %-56s | %-9s | %-19s |\n", label, "LAUNCH ERR", "None");
      }
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(h_a);
  free(h_b);
  free(h_b_t);
  free(h_c);
  free(h_c_ref);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_b_t);
  cudaFree(d_c);
  cublasDestroy(handle);
}
#endif

#if defined(NOTES_V2_ENABLE_WGMMA)
// =============================================================================
// Bench: HGEMM WGMMA (m64n128k16 + TMA + Warp Specialization, SM90+)
// =============================================================================
template <int kStages, int kBlockSwizzle>
static bool launch_timed_hgemm_wgmma(half *d_c, half *h_c, half *h_c_ref,
            CUtensorMap *tma_a, CUtensorMap *tma_b,
            int M, int N, int K, size_t size_c,
            cudaEvent_t start, cudaEvent_t stop,
            float &max_err, float &time_ms) {
  constexpr int BM = 128, BN = 128, BK = 64, kNumThreads = 256;
  using Kernel = void (*)(int, int, int, half *, const CUtensorMap *,
          const CUtensorMap *);
  Kernel k = hgemm_wgmma_stages_tn<64, 128, 16, BM, BN, BK, kNumThreads, kStages,
            kBlockSwizzle>;
  size_t smem = kStages * (BM * BK + BN * BK) * sizeof(half);
  if (!check_smem_feasible((const void *)k, smem)) return false;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 block(kNumThreads);
  const int tiles_n = (N + BN - 1) / BN;
  constexpr int kSwizzleN = 16;
  int gx = kBlockSwizzle ? div_ceil(tiles_n, kSwizzleN) : tiles_n;
  int gz = kBlockSwizzle ? kSwizzleN : 1;
  dim3 grid(gx, (M + BM - 1) / BM, gz);
  for (int w = 0; w < g_warmup; w++)
    k<<<grid, block, smem>>>(M, N, K, d_c, tma_a, tma_b);
  cudaDeviceSynchronize();
  cudaEventRecord(start);
  for (int r = 0; r < g_repeat; r++)
    k<<<grid, block, smem>>>(M, N, K, d_c, tma_a, tma_b);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms, start, stop);
  time_ms /= g_repeat;
  check(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost), "bench wgmma D2H");
  max_err = 0.0f;
  for (int i = 0; i < M * N; i++) {
    float err = fabsf(__half2float(h_c[i]) - __half2float(h_c_ref[i]));
    if (err > max_err) max_err = err;
  }
  return true;
}

static void bench_hgemm_wgmma(int M, int N, int K) {
  constexpr int BM = 128, BN = 128, BK = 64;
  if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
    printf("| %-56s | %-9s | %-19s |\n", "HGEMM WGMMA (unaligned)", "SKIP", "None");
    return;
  }
  size_t size_a = (size_t)M * K * sizeof(half);
  size_t size_b = (size_t)K * N * sizeof(half);
  size_t size_c = (size_t)M * N * sizeof(half);
  half *h_a = (half *)malloc(size_a);
  half *h_b = (half *)malloc(size_b);
  half *h_c_ref = (half *)malloc(size_c);
  srand(42);
  for (int i = 0; i < M * K; i++)
    h_a[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  for (int i = 0; i < K * N; i++)
    h_b[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  size_t size_b_t = (size_t)N * K * sizeof(half);
  half *h_b_t = (half *)malloc(size_b_t);
  for (int n = 0; n < N; n++)
    for (int k = 0; k < K; k++)
      h_b_t[n * K + k] = h_b[k * N + n];
  half *d_a, *d_b, *d_b_t, *d_c;
  check(cudaMalloc(&d_a, size_a), "bench wgmma alloc A");
  check(cudaMalloc(&d_b, size_b), "bench wgmma alloc B");
  check(cudaMalloc(&d_b_t, size_b_t), "bench wgmma alloc B_t");
  check(cudaMalloc(&d_c, size_c), "bench wgmma alloc C");
  check(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice), "bench wgmma H2D A");
  check(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice), "bench wgmma H2D B");
  check(cudaMemcpy(d_b_t, h_b_t, size_b_t, cudaMemcpyHostToDevice), "bench wgmma H2D B_t");
  cublasHandle_t handle;
  cublasCreate(&handle);
  half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h, d_b, CUDA_R_16F, N,
        d_a, CUDA_R_16F, K, &beta_h, d_c, CUDA_R_16F, N, CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT);
  check(cudaMemcpy(h_c_ref, d_c, size_c, cudaMemcpyDeviceToHost), "bench wgmma D2H ref");
  CUtensorMap *tma_a = allocate_and_create_tensor_map(d_a, M / BM, K / BK);
  CUtensorMap *tma_b = allocate_and_create_tensor_map(d_b_t, N / BN, K / BK);
  half *h_c = (half *)malloc(size_c);
  float cublas_tflops = bench_cublas_hgemm_tflops(handle, M, N, K, d_a, d_b, d_c);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for (int stages : {1, 2, 3}) {
    for (int swizzle : {0, 1}) {
      float max_err = 0, time_ms = 0;
      bool ok = false;
      if (stages == 2)
        ok = swizzle
          ? launch_timed_hgemm_wgmma<2, 1>(d_c, h_c, h_c_ref, tma_a, tma_b, M, N,
            K, size_c, start, stop, max_err, time_ms)
          : launch_timed_hgemm_wgmma<2, 0>(d_c, h_c, h_c_ref, tma_a, tma_b, M, N, K,
            size_c, start, stop, max_err, time_ms);
      else
        ok = swizzle
          ? launch_timed_hgemm_wgmma<3, 1>(d_c, h_c, h_c_ref, tma_a, tma_b, M, N,
            K, size_c, start, stop, max_err, time_ms)
          : launch_timed_hgemm_wgmma<3, 0>(d_c, h_c, h_c_ref, tma_a, tma_b, M, N, K,
            size_c, start, stop, max_err, time_ms);
      char label[64];
      snprintf(label, sizeof(label), "HGEMM TMA WGMMA WS (S=%d, BLK_SW=%d)", stages, swizzle);
      if (!ok) {
        if (g_debug)
          printf("| %-56s | %-9s | %-19s |\n", label, "SMEM SKIP", "None");
      } else {
        float tflops = bench_hgemm_tflops(M, N, K, time_ms);
        char tflops_str[32];
        snprintf(tflops_str, sizeof(tflops_str), "%.1f/%.1f (%.2fx)", tflops, 
                 cublas_tflops, tflops / cublas_tflops);
        if (should_print_hgemm_tflops(tflops))
          printf("| %-56s | %.3e | %-19s |\n", label, max_err,
          tflops_str);
      }
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(h_a);
  free(h_b);
  free(h_b_t);
  free(h_c);
  free(h_c_ref);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_b_t);
  cudaFree(d_c);
  cudaFree(tma_a);
  cudaFree(tma_b);
  cublasDestroy(handle);
}
#endif

#if defined(NOTES_V2_ENABLE_TMA_MMA_WS)
// =============================================================================
// Bench: HGEMM TMA MMA WS (mma.sync + TMA + Warp Specialization, SM120)
// =============================================================================
template <int kStages, int kBlockSwizzle>
static bool bench_launch_tma_mma_ws(
  int M, int N, int K, half *d_a, half *d_b_t,
  half *d_c, half *h_c, half *h_c_ref, size_t size_c,
  CUtensorMap *tma_a, CUtensorMap *tma_b,
  cudaEvent_t start, cudaEvent_t stop,
  float &max_err, float &time_ms) {
  constexpr int BM = 128, BN = 128, BK = 64, kNumThreads = 256;
  constexpr size_t payload_bytes = kStages * (BM * BK + BN * BK) * sizeof(half);
  using Kernel = void (*)(int, int, int, half *, const CUtensorMap *,
          const CUtensorMap *);
  Kernel kernel = hgemm_tma_mma_ws_tn<16, 8, 16, 2, 2, 4, 8, 4, kStages,
            kNumThreads, kBlockSwizzle>;
  if (!check_smem_feasible((const void *)kernel, payload_bytes)) return false;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
          payload_bytes);
  dim3 block(kNumThreads);
  constexpr int kSwizzleN = 16;
  const int n_tiles = N / BN;
  const int grid_x = kBlockSwizzle ? div_ceil(n_tiles, kSwizzleN) : n_tiles;
  const int grid_z = kBlockSwizzle ? kSwizzleN : 1;
  dim3 grid(grid_x, M / BM, grid_z);
  for (int w = 0; w < g_warmup; w++)
    kernel<<<grid, block, payload_bytes>>>(M, N, K, d_c, tma_a, tma_b);
  cudaDeviceSynchronize();
  cudaEventRecord(start);
  for (int r = 0; r < g_repeat; r++)
    kernel<<<grid, block, payload_bytes>>>(M, N, K, d_c, tma_a, tma_b);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms, start, stop);
  time_ms /= g_repeat;
  check(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost), "bench tma mma ws D2H");
  max_err = 0.0f;
  for (int i = 0; i < M * N; ++i)
    max_err = fmaxf(max_err, fabsf(__half2float(h_c[i]) - __half2float(h_c_ref[i])));
  return true;
}

static void bench_hgemm_tma_mma_ws(int M, int N, int K) {
  constexpr int BM = 128, BN = 128, BK = 64;
  if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
    printf("| %-56s | %-9s | %-19s |\n", "HGEMM TMA MMA WS (unaligned)", "SKIP", "None");
    return;
  }
  const size_t size_a = size_t(M) * K * sizeof(half);
  const size_t size_b = size_t(K) * N * sizeof(half);
  const size_t size_b_t = size_t(N) * K * sizeof(half);
  const size_t size_c = size_t(M) * N * sizeof(half);
  half *h_a = static_cast<half *>(malloc(size_a));
  half *h_b = static_cast<half *>(malloc(size_b));
  half *h_b_t = static_cast<half *>(malloc(size_b_t));
  half *h_c = static_cast<half *>(malloc(size_c));
  half *h_c_ref = static_cast<half *>(malloc(size_c));
  srand(42);
  for (int i = 0; i < M * K; ++i)
    h_a[i] = __float2half((float(rand()) / RAND_MAX) * 2.0f - 1.0f);
  for (int i = 0; i < K * N; ++i)
    h_b[i] = __float2half((float(rand()) / RAND_MAX) * 2.0f - 1.0f);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      h_b_t[n * K + k] = h_b[k * N + n];
  half *d_a, *d_b, *d_b_t, *d_c;
  check(cudaMalloc(&d_a, size_a), "bench tma mma ws alloc A");
  check(cudaMalloc(&d_b, size_b), "bench tma mma ws alloc B");
  check(cudaMalloc(&d_b_t, size_b_t), "bench tma mma ws alloc B_t");
  check(cudaMalloc(&d_c, size_c), "bench tma mma ws alloc C");
  check(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice), "bench tma mma ws H2D A");
  check(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice), "bench tma mma ws H2D B");
  check(cudaMemcpy(d_b_t, h_b_t, size_b_t, cudaMemcpyHostToDevice),
     "bench tma mma ws H2D B_t");
  cublasHandle_t handle;
  cublasCreate(&handle);
  half alpha = __float2half(1.0f), beta = __float2half(0.0f);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, CUDA_R_16F, N,
        d_a, CUDA_R_16F, K, &beta, d_c, CUDA_R_16F, N, CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT);
  check(cudaMemcpy(h_c_ref, d_c, size_c, cudaMemcpyDeviceToHost),
     "bench tma mma ws D2H reference");
  CUtensorMap *tma_a = allocate_and_create_tensor_map(d_a, M / BM, K / BK);
  CUtensorMap *tma_b = allocate_and_create_tensor_map(d_b_t, N / BN, K / BK);
  float cublas_tflops = bench_cublas_hgemm_tflops(handle, M, N, K, d_a, d_b, d_c);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for (int stages : {1, 2, 3}) {
    for (int swizzle : {0, 1}) {
      float max_err = 0, time_ms = 0;
      bool ok = false;
      if (stages == 2)
        ok = swizzle ? bench_launch_tma_mma_ws<2, 1>(M, N, K, d_a, d_b_t, d_c, h_c,
            h_c_ref, size_c, tma_a, tma_b,
            start, stop, max_err, time_ms)
          : bench_launch_tma_mma_ws<2, 0>(M, N, K, d_a, d_b_t, d_c, h_c,
            h_c_ref, size_c, tma_a, tma_b,
            start, stop, max_err, time_ms);
      else
        ok = swizzle ? bench_launch_tma_mma_ws<3, 1>(M, N, K, d_a, d_b_t, d_c, h_c,
            h_c_ref, size_c, tma_a, tma_b,
            start, stop, max_err, time_ms)
          : bench_launch_tma_mma_ws<3, 0>(M, N, K, d_a, d_b_t, d_c, h_c,
            h_c_ref, size_c, tma_a, tma_b,
            start, stop, max_err, time_ms);
      char label[64];
      snprintf(label, sizeof(label), "HGEMM TMA MMA WS (S=%d, BLK_SW=%d)", stages, swizzle);
      if (!ok) {
        if (g_debug)
          printf("| %-56s | %-9s | %-19s |\n", label, "SMEM SKIP", "None");
      } else {
        float tflops = bench_hgemm_tflops(M, N, K, time_ms);
        char tflops_str[32];
        snprintf(tflops_str, sizeof(tflops_str), "%.1f/%.1f (%.2fx)", tflops, 
                 cublas_tflops, tflops / cublas_tflops);
        if (should_print_hgemm_tflops(tflops))
          printf("| %-56s | %.3e | %-19s |\n", label, max_err,
          tflops_str);
      }
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(h_a);
  free(h_b);
  free(h_b_t);
  free(h_c);
  free(h_c_ref);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_b_t);
  cudaFree(d_c);
  cudaFree(tma_a);
  cudaFree(tma_b);
  cublasDestroy(handle);
}
#endif

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
#endif
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

static void bench_fa_split_d_dispatch(
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
  };
  if (head_dim == 128) call(Int<128>{});
  else if (head_dim == 192) call(Int<192>{});
  else if (head_dim == 256) call(Int<256>{});
  else if (head_dim == 320) call(Int<320>{});
  else if (head_dim == 384) call(Int<384>{});
  else if (head_dim == 448) call(Int<448>{});
  else if (head_dim == 512) call(Int<512>{});
  else if (head_dim == 1024) call(Int<1024>{});
  else
    printf("| %-56s | %-9s | %-19s |\n",
           "FA Split-D CuTe TMA MMA WS (unsupported D)", "SKIP", "None");
}
#endif

static void bench_flash_attn(int B, int H, int N, int D) {
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

// Host 端 v1/v2 等价性测试: 遍历 kColStride/i/j, 断言 swizzle_v1_impl == swizzle_v2_impl.
// 用于在启用 NOTES_V2_ENABLE_SWIZZLE_V2 前后验证 v2 与 v1 bit-exact 等价.
// kColStride=8 在 v2 内回退 v1, 故 8 也应恒等.
template <int kColStride>
static void swizzle_equiv_check_one(long &total, long &fail) {
  for (int i = 0; i < 256; ++i) {
    for (int j = 0; j < kColStride; ++j) {
      int v1 = swizzle_v1_impl<kColStride>(i, j);
      int v2 = swizzle_v2_impl<kColStride>(i, j);
      ++total;
      if (v1 != v2) {
        ++fail;
        if (fail <= 10)
          printf("  MISMATCH cs=%d i=%d j=%d: v1=%d v2=%d\n", kColStride, i, j, v1, v2);
      }
    }
  }
}

static void test_swizzle_equiv() {
  long total = 0, fail = 0;
  swizzle_equiv_check_one<8>(total, fail);
  swizzle_equiv_check_one<16>(total, fail);
  swizzle_equiv_check_one<32>(total, fail);
  swizzle_equiv_check_one<64>(total, fail);
  printf("| %-56s | %-9s |\n",
         "Swizzle v1/v2 equiv (host)", fail == 0 ? "ALL PASS" : "FAIL");
  printf("  total=%ld fail=%ld\n", total, fail);
}

int main(int argc, char *argv[]) {
#if defined(NOTES_V2_ENABLE_WGMMA) || defined(NOTES_V2_ENABLE_TMA_MMA_WS)
  cuInit(0);
#endif

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--bench-hgemm") == 0) {
      g_bench_hgemm = true;
    } else if (strcmp(argv[i], "--bench") == 0) {
      g_bench_hgemm = true;
      g_bench_fa = true;
    } else if (strcmp(argv[i], "--bench-fa") == 0) {
      g_bench_fa = true;
    } else if (strcmp(argv[i], "--bench-fa3-cute") == 0) {
      g_bench_fa = true;
      g_bench_fa3_cute_only = true;
    } else if (strcmp(argv[i], "--bench-hgemm-all") == 0) {
      g_bench_hgemm_all = true;
    } else if (strcmp(argv[i], "--bench-fa-all") == 0) {
      g_bench_fa = true;
      g_fa_layout = FALayout::All;
    } else if (strcmp(argv[i], "--bench-all") == 0) {
      g_bench_all = true;
      g_fa_layout = FALayout::All;
    } else if (strcmp(argv[i], "--mnk") == 0 && i + 1 < argc) {
      sscanf(argv[++i], "%d,%d,%d", &g_bench_M, &g_bench_N, &g_bench_K);
    } else if (strcmp(argv[i], "--bhnd") == 0 && i + 1 < argc) {
      sscanf(argv[++i], "%d,%d,%d,%d", &g_bench_B, &g_bench_H, &g_bench_Nfa, &g_bench_D);
    } else if (strcmp(argv[i], "--fa-layout") == 0 && i + 1 < argc) {
      const char *layout = argv[++i];
      if (strcmp(layout, "all") == 0)
        g_fa_layout = FALayout::All;
      else if (strcmp(layout, "pad") == 0)
        g_fa_layout = FALayout::Pad;
      else if (strcmp(layout, "swizzle-q") == 0)
        g_fa_layout = FALayout::SwizzleQ;
      else if (strcmp(layout, "swizzle-k") == 0)
        g_fa_layout = FALayout::SwizzleK;
      else if (strcmp(layout, "swizzle-v") == 0)
        g_fa_layout = FALayout::SwizzleV;
      else if (strcmp(layout, "swizzle-qk") == 0)
        g_fa_layout = FALayout::SwizzleQK;
      else if (strcmp(layout, "swizzle-qv") == 0)
        g_fa_layout = FALayout::SwizzleQV;
      else if (strcmp(layout, "swizzle-kv") == 0)
        g_fa_layout = FALayout::SwizzleKV;
      else if (strcmp(layout, "swizzle") == 0)
        g_fa_layout = FALayout::Swizzle;
      else {
        fprintf(stderr, "Unsupported FA layout: %s\n", layout);
        return EXIT_FAILURE;
      }
    } else if (strcmp(argv[i], "--fa-skip-check") == 0) {
      g_fa_skip_check = true;
    } else if (strcmp(argv[i], "--swizzle-eq-check") == 0) {
      g_swizzle_eq_check = true;
    } else if (strcmp(argv[i], "--debug") == 0) {
      g_debug = true;
    } else if (strcmp(argv[i], "--verbose") == 0) {
      g_verbose = true;
    } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      g_warmup = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
      g_repeat = atoi(argv[++i]);
    }
  }

  if (g_swizzle_eq_check) {
    printf("=== notes-v2.cu swizzle v1/v2 equivalence check ===\n");
    printf("| %-56s | %-9s |\n", "Kernel", "Max Err");
    printf("|----------------------------------------------------------|----------|\n");
    test_swizzle_equiv();
    printf("=== Done ===\n");
    return 0;
  }

#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
  if (argc >= 2 && strcmp(argv[1], "--fa3-cute") == 0) {
    printf("=== CuTe FA3 TMA MMA WS correctness ===\n");
    printf("| %-56s | %-9s |\n", "Kernel", "Max Err");
    printf("|----------------------------------------------------------|----------|\n");
    test_flash_attn_3_tma_mma_ws_split_q_cute<64>();
    test_flash_attn_3_tma_mma_ws_split_q_cute<128>();
    return 0;
  }
  if (argc >= 2 && strcmp(argv[1], "--fa3-cute-tma-smoke") == 0) {
    printf("=== CuTe TMA copy smoke ===\n");
    printf("| %-56s | %-9s |\n", "Kernel", "Max Err");
    printf("|----------------------------------------------------------|----------|\n");
    test_flash_attn_3_cute_tma_copy_smoke<64>();
    test_flash_attn_3_cute_tma_copy_smoke<128>();
    return 0;
  }
  if (argc >= 2 && strcmp(argv[1], "--fa2-cute-cpasync") == 0) {
    printf("=== CuTe FA2 MMA Stages (cp.async) correctness ===\n");
    printf("| %-56s | %-9s |\n", "Kernel", "Max Err");
    printf("|----------------------------------------------------------|----------|\n");
    test_flash_attn_mma_stages_split_q_cute<64>();
    test_flash_attn_mma_stages_split_q_cute<128>();
    return 0;
  }
  if (argc >= 2 && strcmp(argv[1], "--fa2-cute") == 0) {
    printf("=== CuTe FA2 TMA MMA WS correctness ===\n");
    printf("| %-56s | %-9s |\n", "Kernel", "Max Err");
    printf("|----------------------------------------------------------|----------|\n");
    test_flash_attn_tma_mma_ws_split_q_cute<64>();
    test_flash_attn_tma_mma_ws_split_q_cute<128>();
    return 0;
  }
#endif

  if (g_bench_hgemm || g_bench_fa || g_bench_all) {
    printf("=== notes-v2.cu bench mode ===\n");
    printf("HGEMM: M=%d N=%d K=%d   FA: B=%d H=%d N=%d D=%d\n",
           g_bench_M, g_bench_N, g_bench_K,
           g_bench_B, g_bench_H, g_bench_Nfa, g_bench_D);
    printf("| %-56s | %-9s | %-19s |\n", "Kernel", "Max Err", "TFLOPS/cu{BLAS,DNN}");
    printf(
      "|----------------------------------------------------------|-----------|---------------------|\n"
    );

    if (g_bench_hgemm || g_bench_hgemm_all || g_bench_all) {
      if (g_bench_hgemm_all || g_bench_all) {
        bench_hgemm_mma(g_bench_M, g_bench_N, g_bench_K);
        bench_hgemm_swizzle(g_bench_M, g_bench_N, g_bench_K);
#if defined(NOTES_V2_ENABLE_WGMMA)
        bench_hgemm_wgmma(g_bench_M, g_bench_N, g_bench_K);
#endif
#if defined(NOTES_V2_ENABLE_TMA_MMA_WS)
        bench_hgemm_tma_mma_ws(g_bench_M, g_bench_N, g_bench_K);
#endif
      }
#if defined(NOTES_V2_ENABLE_CUTE)
      bench_hgemm_cute(g_bench_M, g_bench_N, g_bench_K);
#endif
    }
    if (g_bench_fa || g_bench_all)
      bench_flash_attn(g_bench_B, g_bench_H, g_bench_Nfa, g_bench_D);

    printf("=== Bench done ===\n");
    return 0;
  }

#if defined(NOTES_V2_ENABLE_TMA_MMA_WS)
  if (argc >= 2 && strcmp(argv[1], "--tma-mma-ws") == 0) {
    int M = 128, N = 128, K = 64;
    if (argc > 4) {
      M = atoi(argv[2]);
      N = atoi(argv[3]);
      K = atoi(argv[4]);
    }
    printf("=== SM120 TMA MMA WS validation ===\n");
    printf("| %-56s | %-9s |\n", "Kernel", "Max Err");
    printf("|----------------------------------------------------------|----------|\n");
    test_hgemm_tma_mma_ws(M, N, K);
    return 0;
  }
#endif
  int M = 1024, N = 1024, K = 1024;
  if (argc > 3) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }

  printf("=== notes-v2.cu verification harness ===\n");
  printf("| %-56s | %-9s |\n", "Kernel", "Max Err");
  printf("|----------------------------------------------------------|-----------|\n");

  test_block_reduce(N);
  test_dot(N);
  test_relu(1024);
  test_elementwise(1024);
  test_histogram(1024);
  test_merge_attn_states(512, 16, 128);
  test_softmax(256);
  test_rms_norm(8, 128);
  test_layer_norm(8, 128);
  test_rope(8, 128);
  test_mat_transpose(256, 256);
  test_mat_transpose_padded(256, 256);
  test_sgemv(256, 128);
  test_sgemm(M, N, K);
  test_hgemm_mma(M, N, K);
  test_hgemm_swizzle(M, N, K);
#if (defined(NOTES_V2_ENABLE_CUTE))
  test_hgemm_cute(M, N, K);
#endif
#if (defined(NOTES_V2_ENABLE_WGMMA))
  test_hgemm_wgmma(M, N, K);
#endif
#if defined(NOTES_V2_ENABLE_TMA_MMA_WS)
  test_hgemm_tma_mma_ws(M, N, K);
#endif
  test_flash_attn(1024, 64);
#if defined(NOTES_V2_ENABLE_TMA_MMA_WS)
  test_flash_attn_tma_mma_ws(1024, 64);
  test_flash_attn_tma_mma_ws(1024, 128);
  test_flash_attn_3_tma_ws(1024, 64);
  test_flash_attn_3_tma_ws(1024, 128);
#endif

  printf("=== All tests done ===\n");
  return 0;
}

// =============================================================================
// Quick build & run reference
// =============================================================================
// # sm_86 + CuTe（Ampere，RTX 30/40 系列，无 WGMMA）:
// nvcc -std=c++20 -O2 -arch=sm_86 -DNOTES_V2_ENABLE_CUTE \
//   -I ../../third-party/cutlass/include \
//   -lcublas -lcuda notes-v2.cu -o notes_v2_cute_sm86.bin
//
// # sm_89 单独编译 + 运行:
// nvcc -std=c++20 -O2 -arch=sm_89 -lcublas -lcuda notes-v2.cu -o notes_v2_sm89.bin
//
// # sm_89 + CuTe（需要 CUTLASS include 路径, 编译 CUDA_HOME 指向的 CUTLASS 或
//    项目内置 third-party/cutlass）:
// nvcc -std=c++20 -O2 -arch=sm_89 -DNOTES_V2_ENABLE_CUTE \
//   -I ../../third-party/cutlass/include \
//   -lcublas -lcuda notes-v2.cu -o notes_v2_cute_sm89.bin
//
// # sm_90a 单独编译 + 运行（需要 Hopper GPU, H100/H200 均可）:
// nvcc -std=c++20 -O2 -gencode arch=compute_90a,code=sm_90a \
//   -DNOTES_V2_ENABLE_WGMMA -lcublas -lcuda notes-v2.cu -o notes_v2_sm90.bin
//
// # sm_90a + CuTe + WGMMA:
// nvcc -std=c++20 -O2 -gencode arch=compute_90a,code=sm_90a \
//   -DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_WGMMA \
//   -I ../../third-party/cutlass/include \
//   -lcublas -lcuda notes-v2.cu -o notes_v2_cute_sm90.bin
//
// # sm_120a TMA + warp-specialized mma.sync (CUDA Toolkit >= 13.2;
//   RTX PRO 5000 / RTX 5090):
// nvcc -std=c++20 -O2 -arch=sm_120a -DNOTES_V2_ENABLE_TMA_MMA_WS \
//   -lcublas -lcuda notes-v2.cu -o notes_v2_tma_mma_ws_sm120.bin
// ./notes_v2_tma_mma_ws_sm120.bin --tma-mma-ws 1024 1024 1024
//
// # sm_120a + CUTE + TMA_MMA_WS + cuDNN SDPA (CUDA Toolkit >= 13.2;
//   requires cudnn-frontend submodule: git submodule update --init):
// nvcc -std=c++20 -O2 -arch=sm_120a \
//   -DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_TMA_MMA_WS -DNOTES_V2_ENABLE_CUDNN \
//   -I ../../third-party/cutlass/include -I ../../third-party/cudnn-frontend/include \
//   -L/usr/local/cuda-13.2/targets/x86_64-linux/lib/stubs \
//   -lcublas -lcudnn -lnvrtc -lcuda notes-v2.cu -o notes_v2_cute_ws_sm120a.bin
// # Default FA bench (kPadQ=kPadK=kPadV=8 only):
// ./notes_v2_cute_ws_sm120a.bin --bench --bench-fa --bhnd 1,48,4096,64
// # All FA layout variants:
// ./notes_v2_cute_ws_sm120a.bin --bench --bench-fa-all --bhnd 1,48,4096,64
// # FA TMA MMA WS bench (SM120, D=64 only, kStagesK=2/3):
// ./notes_v2_cute_ws_sm120a.bin --bench-fa-tma-ws --bhnd 1,32,4096,64
