// utils.cu — notes-v2.cu 多 TU 拆分后的跨模块共享符号（唯一定义点）。
// 各模块 .cu 顶部以 extern 声明引用这里定义的全局与函数（不建公共头）。
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

// bench/test 全局配置（main 解析 argv 后写入，各模块读取）
bool g_debug = false;
bool g_verbose = false;
int g_warmup = 2;
int g_repeat = 3;
bool g_bench_fa3_cute_only = false;
bool g_fa_skip_check = false;

void check(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "[ERROR] %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

bool check_smem_feasible(const void *kernel_func, size_t dyn_smem_bytes) {
  int device = 0;
  int max_smem = 0;
  cudaFuncAttributes attrs{};
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  cudaFuncGetAttributes(&attrs, kernel_func);
  return dyn_smem_bytes + attrs.sharedSizeBytes <= (size_t)max_smem;
}

float bench_hgemm_tflops(int M, int N, int K, float time_ms) {
  double flops = 2.0 * M * N * K;
  return (float)(flops / (double)time_ms / 1e9);
}

float bench_fa_tflops(int B, int H, int N, int D, float time_ms) {
  double flops = 4.0 * B * H * N * N * D;
  return (float)(flops / (double)time_ms / 1e9);
}

size_t fp8_smem_optin_limit() {
  int dev = 0, bytes = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  return (size_t)bytes;
}

// BF16 GEMM via cuBLAS (CUBLAS_COMPUTE_32F 累加), FP8 GEMM 的精度/性能参照
float bench_cublas_bf16_gemm_tflops(cublasHandle_t handle, int M, int N,
                                    int K, __nv_bfloat16 *d_a,
                                    __nv_bfloat16 *d_b,
                                    __nv_bfloat16 *d_c) {
  float alpha = 1.0f, beta = 0.0f;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cublasSetStream(handle, stream);
  for (int w = 0; w < g_warmup; ++w)
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                 d_b, CUDA_R_16BF, N, d_a, CUDA_R_16BF, K, &beta,
                 d_c, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cudaStreamSynchronize(stream);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  for (int r = 0; r < g_repeat; ++r)
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                 d_b, CUDA_R_16BF, N, d_a, CUDA_R_16BF, K, &beta,
                 d_c, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
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
