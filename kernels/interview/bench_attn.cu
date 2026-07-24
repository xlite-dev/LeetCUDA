// =============================================================================
// bench_attn.cu - Focused bench: FA2 CuTe TMA MMA WS vs cuDNN SDPA
//
// 仅关注 flash_attn_tma_mma_ws_split_q_cute kernel 和 cudnn SDPA kernel，
// 避免引入 notes-v2.cu 中 30+ 算子的编译负担和无关干扰。
//
// Build (SM120 RTX PRO 5000):
//   nvcc -std=c++20 -O3 --expt-relaxed-constexpr --use_fast_math \
//     -arch=sm_120a -DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_TMA_MMA_WS \
//     -DNOTES_V2_ENABLE_CUDNN \
//     -I ../../third-party/cutlass/include \
//     -I ../../third-party/cudnn-frontend/include \
//     -L/usr/local/cuda/targets/x86_64-linux/lib/stubs \
//     -lcublas -lcudnn -lnvrtc -lcuda \
//     bench_attn.cu -o bench_attn_sm120a.bin
//
// Usage:
//   ./bench_attn_sm120a.bin --bhnd 1,32,8192,64
//   ./bench_attn_sm120a.bin --bhnd 1,32,8192,128 --sk 3
//   ./bench_attn_sm120a.bin --bhnd 1,32,8192,64 --cudnn-only
// =============================================================================
#include "base.cuh"
#include "common.cuh"
#include "hgemm.cuh"
#include "flash_attn.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#if defined(NOTES_V2_ENABLE_CUDNN)
namespace fe = cudnn_frontend;
#endif

// ---------------------------------------------------------------------------
// Config globals
// ---------------------------------------------------------------------------
static int g_warmup = 5;
static int g_repeat = 20;
static int g_B = 1, g_H = 32, g_N = 8192, g_D = 64;
static int g_sk = 0;       // 0 = sweep all, >0 = single config
static int g_sv = 1;
static bool g_cudnn_only = false;
static bool g_verbose = false;

// ---------------------------------------------------------------------------
// Helpers (minimal copies from notes-v2.cu to avoid pulling in everything)
// ---------------------------------------------------------------------------
static inline void check(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "[ERROR] %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
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

static float bench_fa_tflops(int B, int H, int N, int D, float time_ms) {
  double flops = 4.0 * B * H * N * N * D;
  return (float)(flops / (double)time_ms / 1e9);
}

// ---------------------------------------------------------------------------
// cuDNN SDPA reference + timing
// ---------------------------------------------------------------------------
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
#endif // NOTES_V2_ENABLE_CUDNN

// ---------------------------------------------------------------------------
// FA2 CuTe TMA MMA WS bench
// ---------------------------------------------------------------------------
#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
template <int kHeadDim, int kStagesK, int kStagesV>
static float bench_fa2_cute_tma_ws_launch(
    int B, int H, int seqlen, half *h_o_ref,
    half *d_q, half *d_k, half *d_v, half *d_o,
    float cudnn_tflops_f32, float &out_max_err) {
  using namespace cute;
  using Traits = fa_cute::FlashAttn2CuTeTraits<kHeadDim>;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  constexpr int kBr = 128;
  constexpr int kNumThreads = 384;
  out_max_err = -1.0f;
  if (seqlen < kBr || seqlen % kBr != 0 || seqlen % 64 != 0) {
    printf("| %-56s | %-9s | %-19s |\n",
           "FA2 CuTe TMA MMA WS (unaligned)", "SKIP", "None");
    return -1.0f;
  }

  int rows = B * H * seqlen;
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
  int smem_bytes = (cosize(SmemLayoutQ{}) +
                    kStagesK * cosize(SmemLayoutKV{}) +
                    kStagesV * cosize(SmemLayoutKV{})) *
                   sizeof(cutlass::half_t);
  bool smem_ok = check_smem_feasible((const void *)kernel, smem_bytes);
  if (!smem_ok) {
    char label[96];
    snprintf(label, sizeof(label),
             "FA2 CuTe TMA MMA WS (Sk=%d, Sv=%d, SMEM)",
             kStagesK, kStagesV);
    printf("| %-56s | %-9s | %-19s |\n", label, "SMEM", "None");
    return -1.0f;
  }
  check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes),
        "bench fa2 cute set smem");

  dim3 grid(seqlen / kBr, B * H);
  cudaStream_t stream;
  cudaEvent_t start, stop;
  check(cudaStreamCreate(&stream), "bench fa2 cute stream");
  check(cudaEventCreate(&start), "bench fa2 cute event start");
  check(cudaEventCreate(&stop), "bench fa2 cute event stop");
  for (int warmup = 0; warmup < g_warmup; ++warmup) {
    kernel<<<grid, kNumThreads, smem_bytes, stream>>>(
        tma_q, tma_k, tma_v,
        reinterpret_cast<cutlass::half_t *>(d_o), rows, seqlen);
  }
  check(cudaStreamSynchronize(stream), "bench fa2 cute warmup sync");
  check(cudaEventRecord(start, stream), "bench fa2 cute record start");
  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    kernel<<<grid, kNumThreads, smem_bytes, stream>>>(
        tma_q, tma_k, tma_v,
        reinterpret_cast<cutlass::half_t *>(d_o), rows, seqlen);
  }
  check(cudaEventRecord(stop, stream), "bench fa2 cute record stop");
  check(cudaEventSynchronize(stop), "bench fa2 cute timing sync");
  float time_ms = 0.0f;
  check(cudaEventElapsedTime(&time_ms, start, stop), "bench fa2 cute elapsed");
  time_ms /= g_repeat;

  size_t count = (size_t)rows * kHeadDim;
  half *h_o = (half *)malloc(count * sizeof(half));
  check(cudaMemcpy(h_o, d_o, count * sizeof(half), cudaMemcpyDeviceToHost),
        "bench fa2 cute D2H");
  float max_err = 0.0f;
  if (h_o_ref) {
    for (size_t idx = 0; idx < count; ++idx) {
      float err = fabsf(__half2float(h_o[idx]) - __half2float(h_o_ref[idx]));
      if (err > max_err) max_err = err;
    }
  }
  out_max_err = max_err;
  float tflops = bench_fa_tflops(B, H, seqlen, kHeadDim, time_ms);
  char label[96];
  snprintf(label, sizeof(label),
           "FA2 CuTe TMA MMA WS (Sk=%d, Sv=%d, F32Acc)",
           kStagesK, kStagesV);
  char performance[32];
  if (cudnn_tflops_f32 > 0.0f) {
    snprintf(performance, sizeof(performance), "%.1f/%.1f (%.2fx)",
             tflops, cudnn_tflops_f32, tflops / cudnn_tflops_f32);
  } else {
    snprintf(performance, sizeof(performance), "%.1f", tflops);
  }
  printf("| %-56s | %.3e | %-19s |\n", label, max_err, performance);

  free(h_o);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  return tflops;
}

template <int kHeadDim>
static void bench_fa2_cute_sweep(int B, int H, int seqlen, half *h_o_ref,
                                  half *d_q, half *d_k, half *d_v, half *d_o,
                                  float cudnn_tflops_f32) {
  float max_err;
  bench_fa2_cute_tma_ws_launch<kHeadDim, 2, 1>(B, H, seqlen, h_o_ref, d_q, d_k, d_v, d_o, cudnn_tflops_f32, max_err);
  bench_fa2_cute_tma_ws_launch<kHeadDim, 3, 1>(B, H, seqlen, h_o_ref, d_q, d_k, d_v, d_o, cudnn_tflops_f32, max_err);
  if constexpr (kHeadDim == 64) {
    bench_fa2_cute_tma_ws_launch<kHeadDim, 4, 1>(B, H, seqlen, h_o_ref, d_q, d_k, d_v, d_o, cudnn_tflops_f32, max_err);
  }
}

template <int kHeadDim, int kStagesQK, int kStagesV>
static float bench_ffpa_cute_tma_ws_split_d_launch(
    int B, int H, int seqlen, half *h_o_ref,
    half *d_q, half *d_k, half *d_v, half *d_o,
    float cudnn_tflops_f32, float &out_max_err) {
  using namespace cute;
  using Traits = fa_cute::FFPAAttnSplitDCuTeTraits<kHeadDim>;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  constexpr int kBr = 128;
  constexpr int kChunk = 64;
  constexpr int kNumThreads = 384;
  out_max_err = -1.0f;
  if (seqlen < kBr || seqlen % kBr != 0 || seqlen % kChunk != 0) {
    printf("| %-56s | %-9s | %-19s |\n",
           "FFPA CuTe TMA MMA WS Split-D (unaligned)", "SKIP", "None");
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
                         Shape<_128, _64>{}, _1{});
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
           "FFPA CuTe TMA MMA WS Split-D (SMEM)", "SMEM", "None");
    return -1.0f;
  }
  check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes),
        "bench ffpa split-d set smem");

  cudaStream_t stream;
  cudaEvent_t start, stop;
  check(cudaStreamCreate(&stream), "bench ffpa split-d stream");
  check(cudaEventCreate(&start), "bench ffpa split-d start");
  check(cudaEventCreate(&stop), "bench ffpa split-d stop");
  dim3 grid(seqlen / kBr, B * H);
  for (int warmup = 0; warmup < g_warmup; ++warmup) {
    kernel<<<grid, kNumThreads, smem_bytes, stream>>>(
        tma_q, tma_k, tma_v,
        reinterpret_cast<cutlass::half_t *>(d_o), rows, seqlen);
  }
  check(cudaGetLastError(), "bench ffpa split-d warmup launch");
  check(cudaStreamSynchronize(stream), "bench ffpa split-d warmup sync");
  check(cudaEventRecord(start, stream), "bench ffpa split-d record start");
  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    kernel<<<grid, kNumThreads, smem_bytes, stream>>>(
        tma_q, tma_k, tma_v,
        reinterpret_cast<cutlass::half_t *>(d_o), rows, seqlen);
  }
  check(cudaGetLastError(), "bench ffpa split-d timed launch");
  check(cudaEventRecord(stop, stream), "bench ffpa split-d record stop");
  check(cudaEventSynchronize(stop), "bench ffpa split-d timing sync");
  float time_ms = 0.0f;
  check(cudaEventElapsedTime(&time_ms, start, stop), "bench ffpa split-d elapsed");
  time_ms /= g_repeat;

  half *h_o = (half *)malloc(count * sizeof(half));
  check(cudaMemcpy(h_o, d_o, count * sizeof(half), cudaMemcpyDeviceToHost),
        "bench ffpa split-d D2H");
  float max_err = -1.0f;
  if (h_o_ref) {
    max_err = 0.0f;
    for (size_t idx = 0; idx < count; ++idx)
      max_err = max(max_err, fabsf(__half2float(h_o[idx]) - __half2float(h_o_ref[idx])));
  }
  out_max_err = max_err;
  const float tflops = bench_fa_tflops(B, H, seqlen, kHeadDim, time_ms);
  char performance[32];
  if (cudnn_tflops_f32 > 0.0f)
    snprintf(performance, sizeof(performance), "%.1f/%.1f (%.2fx)",
             tflops, cudnn_tflops_f32, tflops / cudnn_tflops_f32);
  else
    snprintf(performance, sizeof(performance), "%.1f", tflops);
  printf("| %-56s | %.3e | %-19s |\n",
         "FFPA CuTe TMA MMA WS Split-D (D=512, Sk=2, Sv=2)",
         max_err, performance);

  free(h_o);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  return tflops;
}

static void bench_ffpa_cute_tma_ws_split_d_sweep(
    int B, int H, int seqlen, int head_dim, half *h_o_ref,
    half *d_q, half *d_k, half *d_v, half *d_o,
    float cudnn_tflops_f32) {
  using namespace cute;
  float max_err;
  auto call = [&](auto dim_tag) {
    constexpr int D = decltype(dim_tag)::value;
    bench_ffpa_cute_tma_ws_split_d_launch<D, 1, 1>(
        B, H, seqlen, h_o_ref, d_q, d_k, d_v, d_o, cudnn_tflops_f32, max_err);
    bench_ffpa_cute_tma_ws_split_d_launch<D, 2, 2>(
        B, H, seqlen, h_o_ref, d_q, d_k, d_v, d_o, cudnn_tflops_f32, max_err);
  };
  if (head_dim == 192) call(Int<192>{});
  else if (head_dim == 256) call(Int<256>{});
  else if (head_dim == 320) call(Int<320>{});
  else if (head_dim == 384) call(Int<384>{});
  else if (head_dim == 448) call(Int<448>{});
  else if (head_dim == 512) call(Int<512>{});
  else if (head_dim == 1024) call(Int<1024>{});
  else
    printf("| %-56s | %-9s | %-19s |\n",
           "FFPA CuTe TMA MMA WS Split-D (unsupported D)", "SKIP", "None");
}

// ---------------------------------------------------------------------------
// FA3 CuTe TMA MMA WS bench (Br=64, dual consumer, split-KV merge)
// ---------------------------------------------------------------------------
template <int kHeadDim, int kStagesK>
static float bench_fa3_cute_tma_ws_launch(
    int B, int H, int seqlen, half *h_o_ref,
    half *d_q, half *d_k, half *d_v, half *d_o,
    float cudnn_tflops_f32, float &out_max_err) {
  using namespace cute;
  using Traits = fa_cute::FlashAttn3CuTeTraits<kHeadDim>;
  using SmemLayout = typename Traits::SmemLayoutQKV;
  constexpr int kTile = 64;
  constexpr int kNumConsumers = 2;
  out_max_err = -1.0f;
  if (seqlen < kTile || seqlen % kTile != 0) {
    printf("| %-56s | %-9s | %-19s |\n",
           "FA3 CuTe TMA MMA WS (unaligned)", "SKIP", "None");
    return -1.0f;
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
      kHeadDim, decltype(tma_q), decltype(tma_k), decltype(tma_v), kStagesK>;
  auto acc_o = partition_fragment_C(
      typename Traits::TiledMma{}, Shape<_64, Int<kHeadDim>>{});
  constexpr int kTiles = 1 + kNumConsumers * kStagesK + kNumConsumers;
  constexpr int kTilesBytes = kTiles * cosize(SmemLayout{}) * sizeof(cutlass::half_t);
  int merge_bytes = 128 * size(acc_o) * sizeof(float) + 128 * sizeof(float4);
  int smem_bytes = max(kTilesBytes, merge_bytes);
  bool smem_ok = check_smem_feasible((const void *)kernel, smem_bytes);
  if (!smem_ok) {
    char label[96];
    snprintf(label, sizeof(label),
             "FA3 CuTe TMA MMA WS (Sk=%d, SMEM)", kStagesK);
    printf("| %-56s | %-9s | %-19s |\n", label, "SMEM", "None");
    return -1.0f;
  }
  check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes), "bench fa3 cute set smem");

  dim3 grid(seqlen / kTile, B * H);
  cudaStream_t stream;
  cudaEvent_t start, stop;
  check(cudaStreamCreate(&stream), "bench fa3 cute stream");
  check(cudaEventCreate(&start), "bench fa3 cute event start");
  check(cudaEventCreate(&stop), "bench fa3 cute event stop");
  for (int warmup = 0; warmup < g_warmup; ++warmup) {
    kernel<<<grid, 384, smem_bytes, stream>>>(
        tma_q, tma_k, tma_v,
        reinterpret_cast<cutlass::half_t *>(d_o), rows, seqlen);
  }
  check(cudaStreamSynchronize(stream), "bench fa3 cute warmup sync");
  check(cudaEventRecord(start, stream), "bench fa3 cute record start");
  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    kernel<<<grid, 384, smem_bytes, stream>>>(
        tma_q, tma_k, tma_v,
        reinterpret_cast<cutlass::half_t *>(d_o), rows, seqlen);
  }
  check(cudaEventRecord(stop, stream), "bench fa3 cute record stop");
  check(cudaEventSynchronize(stop), "bench fa3 cute timing sync");
  float time_ms = 0.0f;
  check(cudaEventElapsedTime(&time_ms, start, stop), "bench fa3 cute elapsed");
  time_ms /= g_repeat;

  size_t count = (size_t)rows * kHeadDim;
  half *h_o = (half *)malloc(count * sizeof(half));
  check(cudaMemcpy(h_o, d_o, count * sizeof(half), cudaMemcpyDeviceToHost),
        "bench fa3 cute D2H");
  float max_err = 0.0f;
  if (h_o_ref) {
    for (size_t idx = 0; idx < count; ++idx) {
      float err = fabsf(__half2float(h_o[idx]) - __half2float(h_o_ref[idx]));
      if (err > max_err) max_err = err;
    }
  }
  out_max_err = max_err;
  float tflops = bench_fa_tflops(B, H, seqlen, kHeadDim, time_ms);
  char label[96];
  snprintf(label, sizeof(label),
           "FA3 CuTe TMA MMA WS (2 WG, Sk=%d, F32Acc)", kStagesK);
  char performance[32];
  if (cudnn_tflops_f32 > 0.0f) {
    snprintf(performance, sizeof(performance), "%.1f/%.1f (%.2fx)",
             tflops, cudnn_tflops_f32, tflops / cudnn_tflops_f32);
  } else {
    snprintf(performance, sizeof(performance), "%.1f", tflops);
  }
  printf("| %-56s | %.3e | %-19s |\n", label, max_err, performance);

  free(h_o);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  return tflops;
}

template <int kHeadDim>
static void bench_fa3_cute_sweep(int B, int H, int seqlen, half *h_o_ref,
                                  half *d_q, half *d_k, half *d_v, half *d_o,
                                  float cudnn_tflops_f32) {
  float max_err;
  bench_fa3_cute_tma_ws_launch<kHeadDim, 1>(B, H, seqlen, h_o_ref, d_q, d_k, d_v, d_o, cudnn_tflops_f32, max_err);
  bench_fa3_cute_tma_ws_launch<kHeadDim, 2>(B, H, seqlen, h_o_ref, d_q, d_k, d_v, d_o, cudnn_tflops_f32, max_err);
  bench_fa3_cute_tma_ws_launch<kHeadDim, 3>(B, H, seqlen, h_o_ref, d_q, d_k, d_v, d_o, cudnn_tflops_f32, max_err);
  if constexpr (kHeadDim == 64) {
    bench_fa3_cute_tma_ws_launch<kHeadDim, 4>(B, H, seqlen, h_o_ref, d_q, d_k, d_v, d_o, cudnn_tflops_f32, max_err);
  }
}
#endif // NOTES_V2_ENABLE_CUTE && NOTES_V2_ENABLE_TMA_MMA_WS

// ---------------------------------------------------------------------------
// CPU FP32 reference (fallback when cuDNN unavailable)
// ---------------------------------------------------------------------------
static void cpu_sdpa_ref(half *h_q, half *h_k, half *h_v, float *ref_o,
                          int B, int H, int seqlen, int head_dim) {
  float scale = 1.0f / sqrtf((float)head_dim);
  for (int bh = 0; bh < B * H; ++bh) {
    half *q = h_q + bh * seqlen * head_dim;
    half *k = h_k + bh * seqlen * head_dim;
    half *v = h_v + bh * seqlen * head_dim;
    float *o = ref_o + bh * seqlen * head_dim;
    for (int row = 0; row < seqlen; ++row) {
      float max_val = -INFINITY;
      for (int col = 0; col < seqlen; ++col) {
        float s = 0.0f;
        for (int d = 0; d < head_dim; ++d)
          s += __half2float(q[row * head_dim + d]) * __half2float(k[col * head_dim + d]);
        s *= scale;
        if (s > max_val) max_val = s;
        o[row * head_dim + 0] = s; // temp store score in o[0] slot reuse avoided
      }
      float sum_exp = 0.0f;
      for (int col = 0; col < seqlen; ++col) {
        float s = 0.0f;
        for (int d = 0; d < head_dim; ++d)
          s += __half2float(q[row * head_dim + d]) * __half2float(k[col * head_dim + d]);
        s *= scale;
        sum_exp += expf(s - max_val);
      }
      for (int d = 0; d < head_dim; ++d) o[row * head_dim + d] = 0.0f;
      for (int col = 0; col < seqlen; ++col) {
        float s = 0.0f;
        for (int d = 0; d < head_dim; ++d)
          s += __half2float(q[row * head_dim + d]) * __half2float(k[col * head_dim + d]);
        s *= scale;
        float p = expf(s - max_val) / sum_exp;
        for (int d = 0; d < head_dim; ++d)
          o[row * head_dim + d] += p * __half2float(v[col * head_dim + d]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
static void print_usage(const char *prog) {
  fprintf(stderr,
    "Usage: %s --bhnd B,H,N,D [--warmup W] [--repeat R] [--sk K] [--sv V] "
    "[--cudnn-only] [--verbose]\n"
    "  --bhnd       batch,heads,seqlen,headdim (default: 1,32,8192,64)\n"
    "  --warmup     warmup iterations (default: 5)\n"
    "  --repeat     timed iterations (default: 20)\n"
    "  --sk         single kStagesK config (0=sweep, default: 0)\n"
    "  --sv         kStagesV (default: 1)\n"
    "  --cudnn-only only run cuDNN SDPA bench\n"
    "  --verbose    print all configs\n",
    prog);
}

int main(int argc, char *argv[]) {
#if defined(NOTES_V2_ENABLE_TMA_MMA_WS) || defined(NOTES_V2_ENABLE_WGMMA)
  cuInit(0);
#endif

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--bhnd") == 0 && i + 1 < argc) {
      sscanf(argv[++i], "%d,%d,%d,%d", &g_B, &g_H, &g_N, &g_D);
    } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      g_warmup = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
      g_repeat = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--sk") == 0 && i + 1 < argc) {
      g_sk = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--sv") == 0 && i + 1 < argc) {
      g_sv = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--cudnn-only") == 0) {
      g_cudnn_only = true;
    } else if (strcmp(argv[i], "--verbose") == 0) {
      g_verbose = true;
    } else {
      print_usage(argv[0]);
      return 1;
    }
  }

  if (g_D != 64 && g_D != 128 && g_D != 512) {
    fprintf(stderr, "[ERROR] only D=64, D=128, or D=512 supported, got D=%d\n", g_D);
    return 1;
  }

  int B = g_B, H = g_H, seqlen = g_N, head_dim = g_D;
  size_t sz = (size_t)B * H * seqlen * head_dim * sizeof(half);

  printf("=== bench_attn: FA2 CuTe TMA MMA WS vs cuDNN SDPA ===\n");
  printf("Config: B=%d H=%d N=%d D=%d warmup=%d repeat=%d\n",
         B, H, seqlen, head_dim, g_warmup, g_repeat);
  printf("| %-56s | %-9s | %-19s |\n", "Kernel", "Max Err", "TFLOPS/cuDNN");
  printf("|----------------------------------------------------------|-----------|---------------------|\n");

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
  check(cudaMalloc(&d_q, sz), "alloc Q");
  check(cudaMalloc(&d_k, sz), "alloc K");
  check(cudaMalloc(&d_v, sz), "alloc V");
  check(cudaMalloc(&d_o, sz), "alloc O");
  check(cudaMemcpy(d_q, h_q, sz, cudaMemcpyHostToDevice), "H2D Q");
  check(cudaMemcpy(d_k, h_k, sz, cudaMemcpyHostToDevice), "H2D K");
  check(cudaMemcpy(d_v, h_v, sz, cudaMemcpyHostToDevice), "H2D V");

  half *h_o_ref = nullptr;
  float *ref_o = nullptr;
  float cudnn_tflops_f32 = -1.0f;

#if defined(NOTES_V2_ENABLE_CUDNN)
  {
    half *d_o_ref;
    check(cudaMalloc(&d_o_ref, sz), "alloc O_ref");
    cudnn_tflops_f32 = bench_cudnn_sdpa_tflops(
        d_q, d_k, d_v, d_o_ref, B, H, seqlen, head_dim,
        fe::DataType_t::FLOAT);
    if (cudnn_tflops_f32 > 0.0f) {
      h_o_ref = (half *)malloc(sz);
      check(cudaMemcpy(h_o_ref, d_o_ref, sz, cudaMemcpyDeviceToHost), "ref D2H");
    }
    cudaFree(d_o_ref);
    char perf[32];
    if (cudnn_tflops_f32 > 0.0f)
      snprintf(perf, sizeof(perf), "%.1f", cudnn_tflops_f32);
    else
      snprintf(perf, sizeof(perf), "UNSUPPORTED");
    printf("| %-56s | %-9s | %-19s |\n", "cuDNN SDPA (F32 compute)", "-", perf);
  }
#else
  printf("| %-56s | %-9s | %-19s |\n", "cuDNN SDPA (not built)", "-", "UNAVAILABLE");
#endif

  if (g_cudnn_only) {
    free(h_q); free(h_k); free(h_v); free(h_o_ref); free(ref_o);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
    printf("=== Bench done ===\n");
    return 0;
  }

  if (!h_o_ref) {
    printf("[INFO] cuDNN SDPA does not support this shape; running CPU FP32 reference.\n");
    ref_o = (float *)malloc((size_t)B * H * seqlen * head_dim * sizeof(float));
    cpu_sdpa_ref(h_q, h_k, h_v, ref_o, B, H, seqlen, head_dim);
    h_o_ref = (half *)malloc(sz);
    for (size_t idx = 0; idx < (size_t)B * H * seqlen * head_dim; ++idx)
      h_o_ref[idx] = __float2half(ref_o[idx]);
  }

#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
  if (head_dim == 64) {
    bench_fa2_cute_sweep<64>(B, H, seqlen, h_o_ref, d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    bench_fa3_cute_sweep<64>(B, H, seqlen, h_o_ref, d_q, d_k, d_v, d_o, cudnn_tflops_f32);
  } else if (head_dim == 128) {
    bench_fa2_cute_sweep<128>(B, H, seqlen, h_o_ref, d_q, d_k, d_v, d_o, cudnn_tflops_f32);
    bench_fa3_cute_sweep<128>(B, H, seqlen, h_o_ref, d_q, d_k, d_v, d_o, cudnn_tflops_f32);
  } else {
    bench_ffpa_cute_tma_ws_split_d_sweep(
        B, H, seqlen, head_dim, h_o_ref, d_q, d_k, d_v, d_o, cudnn_tflops_f32);
  }
#else
  fprintf(stderr, "[ERROR] Build without NOTES_V2_ENABLE_CUTE + NOTES_V2_ENABLE_TMA_MMA_WS\n");
#endif

  free(h_q); free(h_k); free(h_v); free(h_o_ref); free(ref_o);
  cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
  printf("=== Bench done ===\n");
  return 0;
}
