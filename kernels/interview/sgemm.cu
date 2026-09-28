// sgemm.cu: host-side tests for sgemm.cuh kernels (split from notes-v2.cu)
// 函数体逐字搬移自 notes-v2.cu，仅去掉 static 供跨 TU 链接
#include "sgemm.cuh"

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

void test_sgemm(int M, int N, int K) {

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

  // ---- SGEMM TF32 (WMMA m16n16k8, dynamic smem, 2-stage pipeline) ----
  // 1) FP32 reference（在 TF32 转换前保存，用于衡量 TF32 精度损失）
  float *h_c_fp32_ref = (float *)malloc(size_c);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
              &alpha, d_b, N, d_a, K, &beta, d_c, N);
  check(cudaMemcpy(h_c_fp32_ref, d_c, size_c, cudaMemcpyDeviceToHost), "fp32 ref D2H");

  // 2) cuBLAS TF32 参考：设置 math mode 为 TF32_TENSOR_OP_MATH
  cublasHandle_t handle_tf32;
  cublasCreate(&handle_tf32);
  cublasSetMathMode(handle_tf32, CUBLAS_TF32_TENSOR_OP_MATH);
  cublasSgemm(handle_tf32, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
              &alpha, d_b, N, d_a, K, &beta, d_c, N);
  check(cudaMemcpy(h_c_ref, d_c, size_c, cudaMemcpyDeviceToHost), "sgemm D2H ref TF32");
  cublasDestroy(handle_tf32);

  // 3) 将 d_a, d_b 原地转换为 TF32 格式（mantissa 23bit → 10bit）
  // 注意：此操作会破坏 d_a/d_b 的 FP32 精度，后续不可再用于 FP32 kernel
  dim3 block_tf32(256);
  dim3 grid_a((M * K + 256 * 4 - 1) / (256 * 4));
  dim3 grid_b((K * N + 256 * 4 - 1) / (256 * 4));
  f32x4_tf32x4_kernel<<<grid_a, block_tf32>>>(d_a, d_a, M * K);
  f32x4_tf32x4_kernel<<<grid_b, block_tf32>>>(d_b, d_b, K * N);
  check(cudaDeviceSynchronize(), "tf32 convert sync");

  // 4) 启动 sgemm_tf32，dynamic smem = 20736 bytes（含 bank-conflict padding）
  dim3 grid_tf32((N + 127) / 128, (M + 127) / 128);
  constexpr int tf32_smem_bytes = 2 * 128 * 12 * sizeof(float) + 2 * 8 * 132 * sizeof(float); // 20736
  sgemm_tf32<<<grid_tf32, block_tf32, tf32_smem_bytes>>>(d_a, d_b, d_c, M, N, K);
  check(cudaGetLastError(), "sgemm_tf32 launch");
  check(cudaDeviceSynchronize(), "sgemm_tf32 sync");

  check(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost), "sgemm_tf32 D2H");

  max_err = 0.0f;
  for (int i = 0; i < M * N; i++) {
    float err = fabsf(h_c[i] - h_c_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "SGEMM-TF32 (vs cuBLAS TF32)", max_err);

  // 5) TF32 vs FP32 精度差异
  max_err = 0.0f;
  for (int i = 0; i < M * N; i++) {
    float err = fabsf(h_c[i] - h_c_fp32_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "SGEMM-TF32 (vs FP32 ref)", max_err);
  free(h_c_fp32_ref);

  free(h_a); free(h_b); free(h_c); free(h_c_ref);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  cublasDestroy(handle);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
