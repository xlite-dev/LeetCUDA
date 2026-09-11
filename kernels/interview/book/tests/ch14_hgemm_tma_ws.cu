// book/tests/ch14_hgemm_tma_ws.cu — ch14 最小测试：SM120 TMA + mma.sync + Warp Specialization
// 测试逻辑抽取自 notes-v2.cu test_hgemm_tma_mma_ws(L1809) 与 launch wrapper(L1744)，
// 参考实现由 cuBLAS F16 换成 CPU fp64（BOOK_PLAN §6 规范）。
// 覆盖：
//   case A: 256x256x256, kStages=2, block swizzle=0（主线 full/empty 协议）
//   case B: 256x256x256, kStages=3, block swizzle=1（stage 回绕 + 3D grid 早退路径）
//   bench : M=N=K=2048 TFLOPS（对照 ch11 hgemm_mma_stages_tn cp.async 流水）
// 约束：M/N 为 128 的倍数、K 为 64 的倍数（TMA box 128x64 + 128B swizzle）；
//       正确性规模 <= 512（BOOK_TEST_MAX_N）。
// 链接说明：build_tests.sh 不传 -lcuda；cuTensorMapEncodeTiled（driver API）经
// cudaGetDriverEntryPointByVersion 取函数指针调用，参数序列与 common.cuh
// create_tensor_map<128,64> 完全一致（major=行 128, minor=K 64, SWIZZLE_128B）。
#define NOTES_V2_ENABLE_TMA_MMA_WS 1
#include "../../hgemm.cuh"
#include "common_test.h"
#include <string>
#include <vector>

using PFN_cuTensorMapEncodeTiled = CUresult (*)(
    CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *, const cuuint64_t *,
    const cuuint64_t *, const cuuint32_t *, const cuuint32_t *,
    CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion,
    CUtensorMapFloatOOBfill);

static PFN_cuTensorMapEncodeTiled g_encode_tiled = nullptr;

static bool load_driver_entry() {
  void *fn = nullptr;
  cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSuccess;
  if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn,
                                       CUDA_VERSION, 0, &status) != cudaSuccess ||
      status != cudaDriverEntryPointSuccess || fn == nullptr)
    return false;
  g_encode_tiled = reinterpret_cast<PFN_cuTensorMapEncodeTiled>(fn);
  return true;
}

// 与 common.cuh create_tensor_map<BlockMajorSize=128, BlockMinorSize=64> 同构：
// shape/box 均 minor（连续维 K）在前；globalStrides 传 stride+1 跳过隐式内维。
static CUtensorMap *make_tma_desc(half *src, int blocks_height,
                                  int blocks_width) {
  constexpr int kMajor = 128, kMinor = 64;  // box: 128 rows x 64 cols (half)
  uint64_t shape[5] = {(uint64_t)kMinor * blocks_width,
                       (uint64_t)kMajor * blocks_height, 1, 1, 1};
  uint64_t stride[5] = {sizeof(half),
                        sizeof(half) * kMinor * blocks_width, 0, 0, 0};
  uint32_t box[5] = {uint32_t(kMinor), uint32_t(kMajor), 1, 1, 1};
  uint32_t box_stride[5] = {1, 1, 1, 1, 1};
  CUtensorMap host_map;
  CUresult r = g_encode_tiled(
      &host_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, src, shape, stride + 1,
      box, box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    printf("FAIL: cuTensorMapEncodeTiled returned %d\n", (int)r);
    exit(1);
  }
  CUtensorMap *d_map;
  BOOK_CUDA_CHECK(cudaMalloc(&d_map, sizeof(CUtensorMap)));
  BOOK_CUDA_CHECK(cudaMemcpy(d_map, &host_map, sizeof(CUtensorMap),
                             cudaMemcpyHostToDevice));
  return d_map;
}

// 与 notes-v2.cu launch_hgemm_tma_mma_ws 同构：smem opt-in + 3D grid swizzle
template <int kStages, int kBlockSwizzle>
static bool launch_ws(int M, int N, int K, half *d_a, half *d_bt, half *d_c,
                      CUtensorMap *tma_a, CUtensorMap *tma_b) {
  constexpr int BM = 128, BN = 128, kNumThreads = 256;
  constexpr size_t smem_bytes =
      size_t(kStages) * (BM * 64 + BN * 64) * sizeof(half);
  using Kernel = void (*)(int, int, int, half *, const CUtensorMap *,
                          const CUtensorMap *);
  Kernel kernel = hgemm_tma_mma_ws_tn<16, 8, 16, 2, 2, 4, 8, 4, kStages,
                                      kNumThreads, kBlockSwizzle>;
  int max_smem = 0;
  cudaFuncAttributes attr{};
  BOOK_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  BOOK_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
  if (smem_bytes + attr.sharedSizeBytes > size_t(max_smem)) return false;
  BOOK_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
  constexpr int kSwizzleN = 16;
  const int n_tiles = N / BN;
  dim3 grid(kBlockSwizzle ? (n_tiles + kSwizzleN - 1) / kSwizzleN : n_tiles,
            M / BM, kBlockSwizzle ? kSwizzleN : 1);
  kernel<<<grid, dim3(kNumThreads), smem_bytes>>>(M, N, K, d_c, tma_a, tma_b);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  return true;
}

struct TestBufs {
  int M, N, K;
  half *d_a = nullptr, *d_bt = nullptr, *d_c = nullptr;
  CUtensorMap *tma_a = nullptr, *tma_b = nullptr;
  std::vector<half> ha, hbt, hc;
  std::vector<double> ref;
};

static void prepare(TestBufs &t, int M, int N, int K) {
  t.M = M, t.N = N, t.K = K;
  const size_t sa = size_t(M) * K, sbt = size_t(N) * K, sc = size_t(M) * N;
  t.ha.resize(sa), t.hbt.resize(sbt), t.hc.resize(sc), t.ref.resize(sc);
  srand(42);
  // 输入压到 [-0.25, 0.25]：f16 累加误差随幅度增长，K=256 时稳定落进 F16Acc 档
  for (size_t i = 0; i < sa; ++i)
    t.ha[i] = __float2half(((float)rand() / RAND_MAX) * 0.5f - 0.25f);
  for (size_t i = 0; i < sbt; ++i)
    t.hbt[i] = __float2half(((float)rand() / RAND_MAX) * 0.5f - 0.25f);
  // CPU fp64 参考：A[M,K] * BT[N,K]^T（TN 布局，kernel 吃 B^T row-major）
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double s = 0.0;
      for (int k = 0; k < K; ++k)
        s += double(__half2float(t.ha[i * K + k])) *
             double(__half2float(t.hbt[j * K + k]));
      t.ref[i * N + j] = s;
    }
  BOOK_CUDA_CHECK(cudaMalloc(&t.d_a, sa * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&t.d_bt, sbt * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&t.d_c, sc * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMemcpy(t.d_a, t.ha.data(), sa * sizeof(half),
                             cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(t.d_bt, t.hbt.data(), sbt * sizeof(half),
                             cudaMemcpyHostToDevice));
  t.tma_a = make_tma_desc(t.d_a, M / 128, K / 64);
  t.tma_b = make_tma_desc(t.d_bt, N / 128, K / 64);
}

static void check_variant(TestBufs &t, bool launched, const char *name) {
  if (!launched) {
    printf("SKIP %s: smem capacity\n", name);
    return;
  }
  const size_t sc = size_t(t.M) * t.N;
  BOOK_CUDA_CHECK(cudaMemcpy(t.hc.data(), t.d_c, sc * sizeof(half),
                             cudaMemcpyDeviceToHost));
  std::vector<float> out(sc);
  for (size_t i = 0; i < sc; ++i) out[i] = __half2float(t.hc[i]);
  book_check(out.data(), t.ref.data(), int(sc), TOL_F16ACC, name);
}

static void cleanup(TestBufs &t) {
  cudaFree(t.d_a), cudaFree(t.d_bt), cudaFree(t.d_c);
  cudaFree(t.tma_a), cudaFree(t.tma_b);
}

static int bench() {
  const int M = 2048, N = 2048, K = 2048, warmup = 3, iters = 10;
  cudaDeviceProp prop;
  BOOK_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount);
  printf("shape: M=N=K=%d, warmup=%d, iters=%d (avg, f16 acc)\n", M, warmup,
         iters);
  if (!load_driver_entry()) {
    printf("FAIL: no driver entry cuTensorMapEncodeTiled\n");
    return 1;
  }
  TestBufs t;
  prepare(t, M, N, K);

  using Kernel = void (*)(int, int, int, half *, const CUtensorMap *,
                          const CUtensorMap *);
  Kernel ws = hgemm_tma_mma_ws_tn<16, 8, 16, 2, 2, 4, 8, 4, 2, 256, 0>;
  constexpr size_t ws_smem = 2 * (128 * 64 + 128 * 64) * sizeof(half);
  BOOK_CUDA_CHECK(cudaFuncSetAttribute(
      ws, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ws_smem));
  dim3 ws_grid(N / 128, M / 128);

  constexpr int ch11BM = 128, ch11BK = 16, ch11Stages = 3;
  constexpr size_t cp_smem =
      size_t(ch11Stages) * (ch11BM * ch11BK + ch11BM * ch11BK) * sizeof(half);
  dim3 cp_grid(N / ch11BM, M / ch11BM);

  cudaEvent_t beg, end;
  BOOK_CUDA_CHECK(cudaEventCreate(&beg));
  BOOK_CUDA_CHECK(cudaEventCreate(&end));
  auto run = [&](auto launch_one, const char *name) {
    for (int w = 0; w < warmup; ++w) launch_one();
    BOOK_CUDA_CHECK(cudaEventRecord(beg));
    for (int i = 0; i < iters; ++i) launch_one();
    BOOK_CUDA_CHECK(cudaEventRecord(end));
    BOOK_CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0.0f;
    BOOK_CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    ms /= iters;
    double tflops = 2.0 * M * N * K / (double(ms) * 1e-3) / 1e12;
    printf("%s: %.4f ms/iter, %.2f TFLOPS (f16 acc)\n", name, ms, tflops);
  };
  run([&] { ws<<<ws_grid, dim3(256), ws_smem>>>(M, N, K, t.d_c, t.tma_a, t.tma_b); },
      "hgemm_tma_mma_ws_tn (S=2)");
  run([&] {
        hgemm_mma_stages_tn<<<cp_grid, dim3(256), cp_smem>>>(
            t.d_a, t.d_bt, t.d_c, M, N, K);
      },
      "hgemm_mma_stages_tn (cp.async, S=3)");
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaEventDestroy(beg));
  BOOK_CUDA_CHECK(cudaEventDestroy(end));
  cleanup(t);
  return 0;
}

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "bench") return bench();
  if (!book_require_sm(90, "ch14")) return 0;  // TMA + mbarrier 需 sm_90+
  if (!load_driver_entry()) {
    printf("FAIL: no driver entry cuTensorMapEncodeTiled\n");
    return 1;
  }

  TestBufs t;
  prepare(t, 256, 256, 256);
  printf("case A: M=256 N=256 K=256 (S=2, BLK_SW=0)\n");
  check_variant(t, launch_ws<2, 0>(t.M, t.N, t.K, t.d_a, t.d_bt, t.d_c,
                                   t.tma_a, t.tma_b),
                "hgemm tma ws 256x256x256 S=2");
  printf("case B: M=256 N=256 K=256 (S=3, BLK_SW=1)\n");
  check_variant(t, launch_ws<3, 1>(t.M, t.N, t.K, t.d_a, t.d_bt, t.d_c,
                                   t.tma_a, t.tma_b),
                "hgemm tma ws 256x256x256 S=3 sw");
  cleanup(t);

  printf(g_failures == 0 ? "ALL OK\n" : "FAILURES PRESENT\n");
  return g_failures == 0 ? 0 : 1;
}
