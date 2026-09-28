// hgemm.cu — notes-v2.cu 多 TU 拆分：HGEMM 模块 host 侧 test/bench 函数独立翻译单元。
// 函数体自 notes-v2.cu 原样搬移（导出函数仅去掉开头 static）；kernel 与 host 封装
// 见 hgemm.cuh，跨模块共享符号定义在 utils.cu。
#include "hgemm.cuh"

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

static float g_hgemm_f16_max_tflops = 0.0f;
static float g_hgemm_f32_max_tflops = 0.0f;


// Decide whether to print a HGEMM TFLOPS line. When --verbose/--debug is off,
// only print when the current TFLOPS exceeds the running max for its
// accumulator category (f16/f32).
static bool should_print_hgemm_tflops(int acc_f32, float tflops) {
  if (g_verbose || g_debug) return true;
  float &max_tflops = acc_f32 ? g_hgemm_f32_max_tflops : g_hgemm_f16_max_tflops;
  if (tflops > max_tflops) {
    max_tflops = tflops;
    return true;
  }
  return false;
}


void test_hgemm_mma(int M, int N, int K) {

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


void test_hgemm_swizzle(int M, int N, int K) {
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
void test_hgemm_cute(int M, int N, int K) {
  // HGEMM CuTe — SM80_16x8x16_{F16,F32}F16F16{_,F32}_TN + Swizzle<3,3,3> + kStage=2
  // TN layout: C[M×N] = A[M×K] × B^T[N×K]
  // Kernel: hgemm_mma_stages_tn_cute via launch_hgemm_mma_stages_tn_cute
  // Tile: BM=128, BN=256, BK=32, 128 threads/block

  size_t size_a = (size_t)M * K * sizeof(half);
  size_t size_b = (size_t)K * N * sizeof(half);
  size_t size_c = (size_t)M * N * sizeof(half);

  half *h_a = (half *)malloc(size_a);
  half *h_b = (half *)malloc(size_b);
  half *h_c_ref = (half *)malloc(size_c);
  half *h_c_ref32 = (half *)malloc(size_c);

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

  // cuBLAS FP16 references: F16 acc (CUBLAS_COMPUTE_16F) and F32 acc (CUBLAS_COMPUTE_32F)
  // (row-major idiom: swap M/N, swap A/B)
  // alpha/beta 类型必须匹配 computeType: 32F→float, 16F→half
  cublasHandle_t handle;
  cublasCreate(&handle);
  half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
  float alpha_f = 1.0f, beta_f = 0.0f;
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h, d_b,
               CUDA_R_16F, N, d_a, CUDA_R_16F, K, &beta_h, d_c, CUDA_R_16F, N,
               CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
  check(cudaMemcpy(h_c_ref, d_c, size_c, cudaMemcpyDeviceToHost), "hgemm_cute D2H ref");
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_f, d_b,
               CUDA_R_16F, N, d_a, CUDA_R_16F, K, &beta_f, d_c, CUDA_R_16F, N,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  check(cudaMemcpy(h_c_ref32, d_c, size_c, cudaMemcpyDeviceToHost), "hgemm_cute D2H ref32");

  // CuTe kernel (Stages=2, TN layout), acc 由模板参数 kAccF32 控制
  half *h_c = (half *)malloc(size_c);
  auto run_one = [&]<bool kAccF32>(const char *label) {
    launch_hgemm_mma_stages_tn_cute<half, 2, 0, kAccF32>(d_a, d_b_t, d_c, M, N, K);
    check(cudaGetLastError(), "hgemm_cute launch");
    check(cudaDeviceSynchronize(), "hgemm_cute sync");
    check(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost), "hgemm_cute D2H");
    half *ref = kAccF32 ? h_c_ref32 : h_c_ref;
    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
      float err = fabsf(__half2float(h_c[i]) - __half2float(ref[i]));
      if (err > max_err) max_err = err;
    }
    printf("| %-56s | %.3e |\n", label, max_err);
  };
  run_one.template operator()<false>("HGEMM CuTe Swizzle + Reg2x (F16Acc)");
  run_one.template operator()<true>("HGEMM CuTe Swizzle + Reg2x (F32Acc)");

  free(h_a); free(h_b); free(h_b_t); free(h_c); free(h_c_ref); free(h_c_ref32);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_b_t); cudaFree(d_c);
  cublasDestroy(handle);
}
#endif /* NOTES_V2_ENABLE_CUTE */


#if defined(NOTES_V2_ENABLE_WGMMA)
void test_hgemm_wgmma(int M, int N, int K) {
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

void test_hgemm_tma_mma_ws(int M, int N, int K) {
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


static float bench_cublas_hgemm_tflops(cublasHandle_t handle, int M, int N, int K,
                                       half *d_a, half *d_b, half *d_c,
                                       cublasComputeType_t compute = CUBLAS_COMPUTE_16F) {
  // alpha/beta 类型必须匹配 computeType: 32F→float, 16F→half
  half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
  float alpha_f = 1.0f, beta_f = 0.0f;
  void *alpha = (compute == CUBLAS_COMPUTE_32F) ? (void *)&alpha_f : (void *)&alpha_h;
  void *beta = (compute == CUBLAS_COMPUTE_32F) ? (void *)&beta_f : (void *)&beta_h;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cublasSetStream(handle, stream);
  // warmup
  for (int w = 0; w < g_warmup; ++w)
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha,
                 d_b, CUDA_R_16F, N, d_a, CUDA_R_16F, K, beta,
                 d_c, CUDA_R_16F, N, compute, CUBLAS_GEMM_DEFAULT);
  cudaStreamSynchronize(stream);
  // timed repeat
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  for (int r = 0; r < g_repeat; ++r)
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha,
                 d_b, CUDA_R_16F, N, d_a, CUDA_R_16F, K, beta,
                 d_c, CUDA_R_16F, N, compute, CUBLAS_GEMM_DEFAULT);
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

void bench_hgemm_mma(int M, int N, int K) {
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
        if (should_print_hgemm_tflops(0, tflops))
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

void bench_hgemm_swizzle(int M, int N, int K) {
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
        if (should_print_hgemm_tflops(0, tflops))
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
void bench_hgemm_cute(int M, int N, int K) {
  size_t size_a = (size_t)M * K * sizeof(half);
  size_t size_b = (size_t)K * N * sizeof(half);
  size_t size_c = (size_t)M * N * sizeof(half);
  half *h_a = (half *)malloc(size_a);
  half *h_b = (half *)malloc(size_b);
  half *h_c_ref = (half *)malloc(size_c);
  half *h_c_ref32 = (half *)malloc(size_c);
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
  float alpha_f = 1.0f, beta_f = 0.0f;
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h, d_b, CUDA_R_16F, N,
        d_a, CUDA_R_16F, K, &beta_h, d_c, CUDA_R_16F, N, CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT);
  check(cudaMemcpy(h_c_ref, d_c, size_c, cudaMemcpyDeviceToHost), "bench cute D2H ref");
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_f, d_b, CUDA_R_16F, N,
        d_a, CUDA_R_16F, K, &beta_f, d_c, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
  check(cudaMemcpy(h_c_ref32, d_c, size_c, cudaMemcpyDeviceToHost), "bench cute D2H ref32");
  half *h_c = (half *)malloc(size_c);
  float cublas_f16_tflops = bench_cublas_hgemm_tflops(handle, M, N, K, d_a, d_b, d_c);
  float cublas_f32_tflops = bench_cublas_hgemm_tflops(handle, M, N, K, d_a, d_b, d_c,
                                                      CUBLAS_COMPUTE_32F);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for (int stages : {2, 3}) {
    for (int swizzle : {0, 1}) {
      for (int acc : {0, 1}) {
        char label[64];
        snprintf(label, sizeof(label), "HGEMM CuTe Swizzle (S=%d, BLK_SW=%d, %sAcc)",
                 stages, swizzle, acc ? "F32" : "F16");
        float time_ms = 0, max_err = 0;
        auto timed = [&]<int kStages, bool kAccF32>() {
          auto k = swizzle ? launch_hgemm_mma_stages_tn_cute<half, kStages, 1, kAccF32>
                           : launch_hgemm_mma_stages_tn_cute<half, kStages, 0, kAccF32>;
          for (int w = 0; w < g_warmup; w++) k(d_a, d_b_t, d_c, M, N, K);
          cudaDeviceSynchronize();
          cudaEventRecord(start);
          for (int r = 0; r < g_repeat; r++) k(d_a, d_b_t, d_c, M, N, K);
          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&time_ms, start, stop);
          time_ms /= g_repeat;
        };
        if (stages == 2) {
          if (acc) timed.template operator()<2, true>();
          else timed.template operator()<2, false>();
        } else {
          if (acc) timed.template operator()<3, true>();
          else timed.template operator()<3, false>();
        }
        check(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost), "bench cute D2H");
        half *ref = acc ? h_c_ref32 : h_c_ref;
        for (int i = 0; i < M * N; i++) {
          float err = fabsf(__half2float(h_c[i]) - __half2float(ref[i]));
          if (err > max_err) max_err = err;
        }
        float tflops = bench_hgemm_tflops(M, N, K, time_ms);
        float cublas_tflops = acc ? cublas_f32_tflops : cublas_f16_tflops;
        char tflops_str[32];
        snprintf(tflops_str, sizeof(tflops_str), "%.1f/%.1f (%.2fx)",
                 tflops, cublas_tflops, tflops / cublas_tflops);
        if (should_print_hgemm_tflops(acc, tflops))
          printf("| %-56s | %.3e | %-19s |\n", label, max_err, tflops_str);
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
  free(h_c_ref32);
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

void bench_hgemm_wgmma(int M, int N, int K) {
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
        if (should_print_hgemm_tflops(0, tflops))
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

void bench_hgemm_tma_mma_ws(int M, int N, int K) {
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
        if (should_print_hgemm_tflops(0, tflops))
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
