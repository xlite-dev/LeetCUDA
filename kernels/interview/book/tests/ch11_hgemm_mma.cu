// book/tests/ch11_hgemm_mma.cu — ch11 最小测试：mma.sync m16n8k16 + ldmatrix + kStages 流水
// 测试逻辑抽取自 notes-v2.cu test_hgemm_mma(L1416)，参考改为 CPU fp64
// （half 输入 -> double 精确参与累加；half 输出 __half2float 转 float 比较）。
// 覆盖：
//   case A: 128x128x128（grid 1x1，单 block；K=128 = 8 个 k-tile，覆盖满载+尾部排空）
//   case B: 256x128x64 （grid (1,2)，M 方向多 block；K=64 非 kStages 倍数）
//   bench : M=N=K=1024 TFLOPS
// 约束：M/N 为 128 的倍数、K 为 16 的倍数（hgemm.cuh 对齐假设）；
//       正确性规模 <= 512（BOOK_TEST_MAX_N）。
// TN 布局：kernel 吃 B^T[N,K] row-major（= B col-major），host 需先转置。
#include "../../hgemm.cuh"
#include "common_test.h"
#include <string>
#include <vector>

// CPU fp64 参考：A[M,K] row-major；bt 为 B^T[N,K] row-major；C = A * B
static void hgemm_ref_fp64(const half *A, const half *BT, double *C, int M,
                           int N, int K) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double s = 0.0;
      for (int k = 0; k < K; ++k)
        s += double(__half2float(A[i * K + k])) *
             double(__half2float(BT[j * K + k]));
      C[i * N + j] = s;
    }
}

static void run_case(const char *tag, int M, int N, int K) {
  const size_t sa = size_t(M) * K, sb = size_t(K) * N, sbt = size_t(N) * K,
               sc = size_t(M) * N;
  std::vector<half> ha(sa), hb(sb), hbt(sbt), hc(sc);
  std::vector<double> ref(sc);
  // 与 notes-v2 同口径：float rand([-1,1]) 再转 half
  std::vector<float> fa(sa), fb(sb);
  srand(42);
  for (size_t i = 0; i < sa; ++i) fa[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  for (size_t i = 0; i < sb; ++i) fb[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  for (size_t i = 0; i < sa; ++i) ha[i] = __float2half(fa[i]);
  for (size_t i = 0; i < sb; ++i) hb[i] = __float2half(fb[i]);
  // B[K,N] -> B^T[N,K]（TN 布局）
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) hbt[n * K + k] = hb[k * N + n];

  hgemm_ref_fp64(ha.data(), hbt.data(), ref.data(), M, N, K);

  half *d_a, *d_bt, *d_c;
  BOOK_CUDA_CHECK(cudaMalloc(&d_a, sa * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_bt, sbt * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_c, sc * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMemcpy(d_a, ha.data(), sa * sizeof(half),
                             cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_bt, hbt.data(), sbt * sizeof(half),
                             cudaMemcpyHostToDevice));

  printf("case %s\n", tag);
  constexpr int BM = 128, BN = 128, BK = 16, kStages = 3;
  constexpr size_t smem_bytes = size_t(kStages) * (BM * BK + BN * BK) * sizeof(half);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  hgemm_mma_stages_tn<<<grid, dim3(256), smem_bytes>>>(d_a, d_bt, d_c, M, N, K);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(cudaMemcpy(hc.data(), d_c, sc * sizeof(half),
                             cudaMemcpyDeviceToHost));

  // half 输出转 float 后与 fp64 参考比较（F16Acc 档）
  std::vector<float> out(sc);
  for (size_t i = 0; i < sc; ++i) out[i] = __half2float(hc[i]);
  std::string name = std::string("hgemm_mma ") + std::to_string(M) + "x" +
                     std::to_string(N) + "x" + std::to_string(K);
  book_check(out.data(), ref.data(), int(sc), TOL_F16ACC, name.c_str());

  BOOK_CUDA_CHECK(cudaFree(d_a));
  BOOK_CUDA_CHECK(cudaFree(d_bt));
  BOOK_CUDA_CHECK(cudaFree(d_c));
}

static int bench() {
  const int M = 1024, N = 1024, K = 1024, warmup = 3, iters = 10;
  cudaDeviceProp prop;
  BOOK_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  printf("shape: M=N=K=%d, warmup=%d, iters=%d (avg)\n", M, warmup, iters);
  const size_t sa = size_t(M) * K, sbt = size_t(N) * K, sc = size_t(M) * N;
  std::vector<half> ha(sa), hbt(sbt);
  srand(42);
  for (size_t i = 0; i < sa; ++i)
    ha[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  for (size_t i = 0; i < sbt; ++i)
    hbt[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  half *d_a, *d_bt, *d_c;
  BOOK_CUDA_CHECK(cudaMalloc(&d_a, sa * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_bt, sbt * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_c, sc * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMemcpy(d_a, ha.data(), sa * sizeof(half),
                             cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_bt, hbt.data(), sbt * sizeof(half),
                             cudaMemcpyHostToDevice));
  constexpr int BM = 128, BN = 128, BK = 16, kStages = 3;
  constexpr size_t smem_bytes = size_t(kStages) * (BM * BK + BN * BK) * sizeof(half);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  cudaEvent_t beg, end;
  BOOK_CUDA_CHECK(cudaEventCreate(&beg));
  BOOK_CUDA_CHECK(cudaEventCreate(&end));
  for (int w = 0; w < warmup; ++w)
    hgemm_mma_stages_tn<<<grid, dim3(256), smem_bytes>>>(d_a, d_bt, d_c, M, N, K);
  BOOK_CUDA_CHECK(cudaEventRecord(beg));
  for (int i = 0; i < iters; ++i)
    hgemm_mma_stages_tn<<<grid, dim3(256), smem_bytes>>>(d_a, d_bt, d_c, M, N, K);
  BOOK_CUDA_CHECK(cudaEventRecord(end));
  BOOK_CUDA_CHECK(cudaEventSynchronize(end));
  float ms = 0.0f;
  BOOK_CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
  ms /= iters;
  double tflops = 2.0 * M * N * K / (double(ms) * 1e-3) / 1e12;
  printf("hgemm_mma_stages_tn: %.4f ms/iter, %.2f TFLOPS (f16 acc)\n", ms, tflops);
  BOOK_CUDA_CHECK(cudaEventDestroy(beg));
  BOOK_CUDA_CHECK(cudaEventDestroy(end));
  BOOK_CUDA_CHECK(cudaFree(d_a));
  BOOK_CUDA_CHECK(cudaFree(d_bt));
  BOOK_CUDA_CHECK(cudaFree(d_c));
  return 0;
}

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "bench") return bench();
  if (!book_require_sm(80, "ch11")) return 0;  // mma m16n8k16 需 sm_80+

  run_case("A: M=128 N=128 K=128", 128, 128, 128);
  run_case("B: M=256 N=128 K=64", 256, 128, 64);

  printf(g_failures == 0 ? "ALL OK\n" : "FAILURES PRESENT\n");
  return g_failures == 0 ? 0 : 1;
}
