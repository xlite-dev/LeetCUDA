// bench_sgemm.cu — SGEMM 性能与精度分析工具
// 测试 sgemm.cuh 中所有 SGEMM kernel 的 benchmark
//
// 编译：
//   nvcc -std=c++20 -O3 -arch=sm_86 -I ../../third-party/cutlass/include -lcublas -lcuda bench_sgemm.cu -o bench_sgemm
//
// 运行：
//   ./bench_sgemm --mnk 1024,1024,1024
//   ./bench_sgemm --mnk 4096,4096,4096 --warmup 5 --repeat 10

#include "sgemm.cuh"

#define DIV_UP(a, b) (((a) + (b) - 1) / (b))
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ================================================================
// Utility
// ================================================================
static void init_random(float *buf, int n) {
  for (int i = 0; i < n; ++i)
    buf[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

static float max_abs_diff(const float *a, const float *b, int n) {
  float max_err = 0.f;
  for (int i = 0; i < n; ++i) {
    float err = fabsf(a[i] - b[i]);
    if (err > max_err) max_err = err;
  }
  return max_err;
}

static float benchmark_kernel(dim3 grid, dim3 block, size_t dyn_smem,
                              void (*kernel)(float *, float *, float *, int, int, int),
                              float *d_a, float *d_b, float *d_c,
                              int M, int N, int K,
                              int warmup, int repeat) {
  for (int i = 0; i < warmup; ++i) {
    if (dyn_smem > 0)
      kernel<<<grid, block, dyn_smem>>>(d_a, d_b, d_c, M, N, K);
    else
      kernel<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < repeat; ++i) {
    if (dyn_smem > 0)
      kernel<<<grid, block, dyn_smem>>>(d_a, d_b, d_c, M, N, K);
    else
      kernel<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return elapsed_ms / repeat;
}

static float benchmark_cublas(cublasHandle_t handle, float *d_a, float *d_b,
                              float *d_c, int M, int N, int K,
                              cublasMath_t math_mode,
                              int warmup, int repeat) {
  cublasSetMathMode(handle, math_mode);
  float alpha = 1.0f, beta = 0.0f;

  for (int i = 0; i < warmup; ++i)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                &alpha, d_b, N, d_a, K, &beta, d_c, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < repeat; ++i)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                &alpha, d_b, N, d_a, K, &beta, d_c, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return elapsed_ms / repeat;
}

// ================================================================
// Main test function
// ================================================================
static void run_test(int M, int N, int K, int warmup, int repeat) {
  size_t sa = (size_t)M * K * sizeof(float);
  size_t sb = (size_t)K * N * sizeof(float);
  size_t sc = (size_t)M * N * sizeof(float);
  int mn = M * N;

  float *h_a = (float *)malloc(sa);
  float *h_b = (float *)malloc(sb);
  float *h_c_ref = (float *)malloc(sc);
  float *h_c = (float *)malloc(sc);
  float *h_c_tf32_ref = (float *)malloc(sc);
  init_random(h_a, M * K);
  init_random(h_b, K * N);

  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, sa));
  CUDA_CHECK(cudaMalloc(&d_b, sb));
  CUDA_CHECK(cudaMalloc(&d_c, sc));
  CUDA_CHECK(cudaMemcpy(d_a, h_a, sa, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, sb, cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0f, beta = 0.0f;

  printf("\n--- Matrix: M=%d, N=%d, K=%d ---\n", M, N, K);
  printf("| %-30s | %-10s | %-12s | %-10s |\n", "Kernel", "Max Err", "Time (ms)", "TFLOPS");
  printf("|--------------------------------|------------|--------------|------------|\n");

  // cuBLAS FP32 (reference)
  cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
              &alpha, d_b, N, d_a, K, &beta, d_c, N);
  CUDA_CHECK(cudaMemcpy(h_c_ref, d_c, sc, cudaMemcpyDeviceToHost));

  float t_cublas_fp32 = benchmark_cublas(handle, d_a, d_b, d_c, M, N, K,
                                          CUBLAS_DEFAULT_MATH, warmup, repeat);
  float tflops_cublas_fp32 = 2.0 * M * N * K / t_cublas_fp32 / 1e9;
  printf("| %-30s | %-10s | %-12.3f | %-10.1f |\n",
         "cuBLAS FP32", "ref", t_cublas_fp32, tflops_cublas_fp32);

  // cuBLAS TF32
  float t_cublas_tf32 = benchmark_cublas(handle, d_a, d_b, d_c, M, N, K,
                                          CUBLAS_TF32_TENSOR_OP_MATH, warmup, repeat);
  CUDA_CHECK(cudaMemcpy(h_c, d_c, sc, cudaMemcpyDeviceToHost));
  float err_cublas_tf32 = max_abs_diff(h_c_ref, h_c, mn);
  float tflops_cublas_tf32 = 2.0 * M * N * K / t_cublas_tf32 / 1e9;
  printf("| %-30s | %-10.3e | %-12.3f | %-10.1f |\n",
         "cuBLAS TF32 (vs FP32)", err_cublas_tf32, t_cublas_tf32, tflops_cublas_tf32);
  memcpy(h_c_tf32_ref, h_c, sc);

  // SGEMM Level 1 (FP32)
  dim3 grid_naive(DIV_UP(N, 32), DIV_UP(M, 32));
  dim3 block_naive(32, 32);
  float t_naive = benchmark_kernel(grid_naive, block_naive, 0, sgemm,
                                    d_a, d_b, d_c, M, N, K, warmup, repeat);
  CUDA_CHECK(cudaMemcpy(h_c, d_c, sc, cudaMemcpyDeviceToHost));
  float err_naive = max_abs_diff(h_c_ref, h_c, mn);
  float tflops_naive = 2.0 * M * N * K / t_naive / 1e9;
  printf("| %-30s | %-10.3e | %-12.3f | %-10.1f |\n",
         "SGEMM Level1 (32x32)", err_naive, t_naive, tflops_naive);

  // SGEMM Vec4 (FP32)
  dim3 grid_vec4(DIV_UP(N, 128), DIV_UP(M, 128));
  dim3 block_vec4(32, 32);
  float t_vec4 = benchmark_kernel(grid_vec4, block_vec4, 0, sgemm_vec4,
                                   d_a, d_b, d_c, M, N, K, warmup, repeat);
  CUDA_CHECK(cudaMemcpy(h_c, d_c, sc, cudaMemcpyDeviceToHost));
  float err_vec4 = max_abs_diff(h_c_ref, h_c, mn);
  float tflops_vec4 = 2.0 * M * N * K / t_vec4 / 1e9;
  printf("| %-30s | %-10.3e | %-12.3f | %-10.1f |\n",
         "SGEMM Vec4 (128x128)", err_vec4, t_vec4, tflops_vec4);

  // SGEMM TF32 (WMMA) — 转换 d_a/d_b 为 TF32 后测试
  dim3 block_cvt(256);
  f32x4_tf32x4_kernel<<<DIV_UP(M * K, 256 * 4), block_cvt>>>(d_a, d_a, M * K);
  f32x4_tf32x4_kernel<<<DIV_UP(K * N, 256 * 4), block_cvt>>>(d_b, d_b, K * N);
  CUDA_CHECK(cudaDeviceSynchronize());

  dim3 grid_tf32(DIV_UP(N, 128), DIV_UP(M, 128));
  dim3 block_tf32(256);
  constexpr int tf32_smem = 2 * 128 * 12 * sizeof(float) + 2 * 8 * 132 * sizeof(float); // 20736
  float t_tf32 = benchmark_kernel(grid_tf32, block_tf32, tf32_smem, sgemm_tf32,
                                   d_a, d_b, d_c, M, N, K, warmup, repeat);
  CUDA_CHECK(cudaMemcpy(h_c, d_c, sc, cudaMemcpyDeviceToHost));
  float err_tf32_tf32 = max_abs_diff(h_c_tf32_ref, h_c, mn);
  float err_tf32_fp32 = max_abs_diff(h_c_ref, h_c, mn);
  float tflops_tf32 = 2.0 * M * N * K / t_tf32 / 1e9;
  printf("| %-30s | %-10.3e | %-12.3f | %-10.1f |\n",
         "SGEMM TF32 (vs cuBLAS TF32)", err_tf32_tf32, t_tf32, tflops_tf32);
  printf("| %-30s | %-10.3e | %-12s | %-10s |\n",
         "SGEMM TF32 (vs FP32)", err_tf32_fp32, "", "");

  printf("|--------------------------------|------------|--------------|------------|\n");
  printf("| %-30s | %-10s | %-12.1f | %-10.1f |\n",
         "Speedup vs cuBLAS FP32", "",
         t_cublas_fp32 / t_tf32, tflops_tf32 / tflops_cublas_fp32);

  cublasDestroy(handle);
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  free(h_a); free(h_b); free(h_c_ref); free(h_c); free(h_c_tf32_ref);
}

// ================================================================
// CLI
// ================================================================
int main(int argc, char **argv) {
  int M = 1024, N = 1024, K = 1024;
  int warmup = 2, repeat = 5;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--mnk") == 0 && i + 3 < argc) {
      M = atoi(argv[++i]);
      N = atoi(argv[++i]);
      K = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      warmup = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
      repeat = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--help") == 0) {
      printf("Usage: %s [--mnk M N K] [--warmup N] [--repeat N]\n", argv[0]);
      return 0;
    }
  }

  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  printf("Device: %s (SM_%d.%d)\n", prop.name, prop.major, prop.minor);
  printf("Benchmark: M=%d, N=%d, K=%d, warmup=%d, repeat=%d\n",
         M, N, K, warmup, repeat);

  run_test(M, N, K, warmup, repeat);

  return 0;
}
