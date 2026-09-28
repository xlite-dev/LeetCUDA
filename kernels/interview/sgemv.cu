// sgemv.cu — notes-v2.cu 多 TU 拆分: sgemv 模块 host 侧测试函数（独立翻译单元）。
// kernel 定义见 sgemv.cuh；跨模块共享符号定义见 utils.cu。
#include "sgemv.cuh"

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

void test_sgemv(int M, int K) {

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
