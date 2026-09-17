#pragma once
// book/tests/common_test.h — 最小测试共享设施（RFC-0.6，规范 BOOK_PLAN §6）
// 无 cuBLAS/cuDNN 依赖；正确性对照 = CPU fp64；容差三档；arch 不在位输出 SKIP。

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 容差三档（首版阈值，调整须在 RFC 勾选项注明理由）
#define TOL_F32ACC 1e-3  // fp32 累加路径（sgemm 系列、FA F32Acc）
#define TOL_F16ACC 5e-2  // fp16 in/out + fp16 累加（hgemm F16Acc、FA F16Acc）
#define TOL_TF32 1e-2    // sgemm_tf32

// 测试规模上限（CPU fp64 三重循环时限约束）
#define BOOK_TEST_MAX_N 512

static int g_failures = 0;

static int book_device_sm() {
  int dev = 0, major = 0, minor = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  return major * 10 + minor;
}

// arch-gated kernel：目标 arch 不在位打印 SKIP 并返回 false（测试 main 直接 return 0）
static bool book_require_sm(int sm, const char* tag) {
  int cur = book_device_sm();
  if (cur >= sm) return true;
  printf("SKIP(%s): requires sm_%d, device is sm_%d\n", tag, sm, cur);
  return false;
}

#define BOOK_CUDA_CHECK(call)                                                  \
  do {                                                                         \
    cudaError_t e_ = (call);                                                   \
    if (e_ != cudaSuccess) {                                                   \
      printf("CUDA error at %s:%d: %s (%s)\n", __FILE__, __LINE__,             \
             cudaGetErrorString(e_), #call);                                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// 统一判定：GPU 输出（拷回 host 的 float；half 结果由测试先 __half2float 转换）
// vs CPU fp64 参考。打印 max_abs_err 与判定结果。
static bool book_check(const float* out, const double* ref, int n, double tol,
                       const char* name) {
  double max_err = 0.0;
  int first_bad = -1;
  for (int i = 0; i < n; ++i) {
    double err = std::fabs(double(out[i]) - ref[i]);
    if (err > max_err) max_err = err;
    if (err > tol && first_bad < 0) first_bad = i;
  }
  bool pass = max_err <= tol;
  printf("%s %s: max_abs_err=%.3e (tol=%.1e)%s\n", pass ? "PASS" : "FAIL", name,
         max_err, tol,
         first_bad >= 0 ? ("  first_bad_index=" + std::to_string(first_bad)).c_str() : "");
  if (!pass) g_failures++;
  return pass;
}

// 便捷 host 初始化（确定性伪随机，便于复现）
static void book_fill_rand(float* p, int n, unsigned seed = 0xA5A5) {
  srand(seed);
  for (int i = 0; i < n; ++i) p[i] = (float(rand() % 2000) - 1000.0f) / 1000.0f;
}
