// book/tests/ch16_fa2_mma.cu — ch16 最小测试：FA2 Split-Q + MMA m16n8k16 多级流水
// 测试逻辑抽取自 notes-v2.cu test_flash_attn(L1895)，参考改为 CPU fp64
// （softmax attention 直接算；half 输出 __half2float 转 float 比较）。
// 覆盖：
//   case A: B=1 H=2 N=256 D=64（grid (2,2)，单 Q tile 多 head；Tc=4 覆盖流水满载+尾部）
//   case B: B=1 H=1 N=512 D=64（grid (4,1)，多 Q tile；Tc=8）
//   每个 case 测两个累加变体：F16Acc（kMmaAccF32=0，TOL_F16ACC）与
//   F32Acc（kMmaAccF32=1，TOL_F32ACC）
//   bench : B=1 H=32 N=4096 D=64（源码注释口径）两变体 TFLOPS
// 约束（kernel 对齐假设，见 ch16 §16.8 坑一/坑三）：N % Bc == 0 且 N % Br == 0
//   （Br=128, Bc=64），N >= Br；正确性规模 <= 512（BOOK_TEST_MAX_N）。
// include 顺序注意（源码观察，见 ch16 §16.7）：flash_attn.cuh 头部不自包含——
// 其用到的 swizzle<kColStride> 派发器定义在 hgemm.cuh L389，必须先 include
// hgemm.cuh（notes-v2.cu L26-27 同序）。
#include "../../hgemm.cuh"
#include "../../flash_attn.cuh"
#include "common_test.h"
#include <string>
#include <vector>

// 模板实参（同 notes-v2 test_flash_attn）：kHeadDim=64, kStagesK=2, Q/K/V pad=8
constexpr int kHeadDimC = 64;
constexpr int kStagesKC = 2;
constexpr int kPadC = 8;
constexpr int kMmaAtomMC = 16, kMmaAtomNC = 8, kMmaAtomKC = 16;
constexpr int kMmaTileSeqLenQC = 8, kMmaTileSeqLenKC = 1;
constexpr int kMmaTileSeqLenPC = 8, kMmaTileHeadDimVC = 1;
constexpr int kValTileSeqLenQC = 1, kValTileSeqLenKC = 8;
constexpr int kValTileSeqLenPC = 1;
constexpr int kValTileHeadDimVC = kHeadDimC / (8 * kMmaTileHeadDimVC);
constexpr int BrC = kMmaAtomMC * kMmaTileSeqLenQC * kValTileSeqLenQC;  // 128
constexpr int BcC = kMmaAtomNC * kMmaTileSeqLenKC * kValTileSeqLenKC;  // 64

using FAKernel = void (*)(half *, half *, half *, half *, int, int);

template <int kMmaAccF32C>
static FAKernel make_fa_kernel() {
  return flash_attn_mma_stages_split_q<
      kHeadDimC, kMmaAtomMC, kMmaAtomNC, kMmaAtomKC, kMmaAccF32C,
      kMmaTileSeqLenQC, kMmaTileSeqLenKC, kMmaTileSeqLenPC, kMmaTileHeadDimVC,
      kValTileSeqLenQC, kValTileSeqLenKC, kValTileSeqLenPC, kValTileHeadDimVC,
      kStagesKC, kPadC, kPadC, kPadC>;
}

// CPU fp64 参考：O = softmax(Q K^T / sqrt(d)) V，逐 (b,h,q) 行直接算
static void fa_ref_fp64(const half *Q, const half *K, const half *V,
                        double *O, int B, int H, int N, int D) {
  const double scale = 1.0 / sqrt((double)D);
  std::vector<double> S(N);
  for (int bh = 0; bh < B * H; ++bh) {
    const half *q = Q + (size_t)bh * N * D;
    const half *k = K + (size_t)bh * N * D;
    const half *v = V + (size_t)bh * N * D;
    for (int qi = 0; qi < N; ++qi) {
      double smax = -INFINITY;
      for (int kj = 0; kj < N; ++kj) {
        double s = 0.0;
        for (int d = 0; d < D; ++d)
          s += (double)__half2float(q[qi * D + d]) *
               (double)__half2float(k[kj * D + d]);
        S[kj] = s * scale;
        if (S[kj] > smax) smax = S[kj];
      }
      double sum_exp = 0.0;
      for (int kj = 0; kj < N; ++kj) sum_exp += exp(S[kj] - smax);
      const double inv = 1.0 / sum_exp;
      for (int d = 0; d < D; ++d) {
        double o = 0.0;
        for (int kj = 0; kj < N; ++kj)
          o += exp(S[kj] - smax) * inv *
               (double)__half2float(v[kj * D + d]);
        O[(size_t)bh * N * D + qi * D + d] = o;
      }
    }
  }
}

static void run_case(const char *tag, int B, int H, int N) {
  const size_t sz = (size_t)B * H * N * kHeadDimC;
  std::vector<half> hq(sz), hk(sz), hv(sz), ho(sz);
  std::vector<double> ref(sz);
  srand(42);
  for (size_t i = 0; i < sz; ++i) {
    hq[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    hk[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    hv[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  }
  fa_ref_fp64(hq.data(), hk.data(), hv.data(), ref.data(), B, H, N,
              kHeadDimC);

  half *d_q, *d_k, *d_v, *d_o;
  BOOK_CUDA_CHECK(cudaMalloc(&d_q, sz * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_k, sz * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_v, sz * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_o, sz * sizeof(half)));
  BOOK_CUDA_CHECK(
      cudaMemcpy(d_q, hq.data(), sz * sizeof(half), cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(
      cudaMemcpy(d_k, hk.data(), sz * sizeof(half), cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(
      cudaMemcpy(d_v, hv.data(), sz * sizeof(half), cudaMemcpyHostToDevice));

  const size_t smem_bytes =
      (BrC * (kHeadDimC + kPadC) + kStagesKC * BcC * (kHeadDimC + kPadC) +
       BcC * (kHeadDimC + kPadC)) *
      sizeof(half);
  dim3 block(kWarpSize * kMmaTileSeqLenQC * kMmaTileSeqLenKC);  // 256
  dim3 grid((N + BrC - 1) / BrC, B * H);

  printf("case %s\n", tag);
  // 两个累加变体：F16Acc（TOL_F16ACC）与 F32Acc（TOL_F32ACC）
  for (int acc = 0; acc <= 1; ++acc) {
    FAKernel fa_k =
        acc == 0 ? make_fa_kernel<0>() : make_fa_kernel<1>();
    cudaFuncSetAttribute(fa_k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);
    fa_k<<<grid, block, smem_bytes>>>(d_q, d_k, d_v, d_o, N, H);
    BOOK_CUDA_CHECK(cudaGetLastError());
    BOOK_CUDA_CHECK(cudaDeviceSynchronize());
    BOOK_CUDA_CHECK(cudaMemcpy(ho.data(), d_o, sz * sizeof(half),
                               cudaMemcpyDeviceToHost));
    std::vector<float> out(sz);
    for (size_t i = 0; i < sz; ++i) out[i] = __half2float(ho[i]);
    const double tol = acc == 0 ? TOL_F16ACC : TOL_F32ACC;
    const char *acc_label = acc == 0 ? "F16Acc" : "F32Acc";
    std::string name = std::string("fa2 splitq ") + acc_label + " B" +
                       std::to_string(B) + " H" + std::to_string(H) + " N" +
                       std::to_string(N) + " D" + std::to_string(kHeadDimC);
    book_check(out.data(), ref.data(), (int)sz, tol, name.c_str());
  }

  BOOK_CUDA_CHECK(cudaFree(d_q));
  BOOK_CUDA_CHECK(cudaFree(d_k));
  BOOK_CUDA_CHECK(cudaFree(d_v));
  BOOK_CUDA_CHECK(cudaFree(d_o));
}

// bench 口径同源码头注释（SM120, B=1,H=32,N=4096,D=64）：flops = 4*B*H*N^2*D
static int bench() {
  const int B = 1, H = 32, N = 4096, warmup = 3, iters = 10;
  cudaDeviceProp prop;
  BOOK_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  printf("shape: B=%d H=%d N=%d D=%d, warmup=%d, iters=%d (avg)\n", B, H, N,
         kHeadDimC, warmup, iters);
  const size_t sz = (size_t)B * H * N * kHeadDimC;
  std::vector<half> hq(sz), hk(sz), hv(sz);
  srand(42);
  for (size_t i = 0; i < sz; ++i) {
    hq[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    hk[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    hv[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  }
  half *d_q, *d_k, *d_v, *d_o;
  BOOK_CUDA_CHECK(cudaMalloc(&d_q, sz * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_k, sz * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_v, sz * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_o, sz * sizeof(half)));
  BOOK_CUDA_CHECK(
      cudaMemcpy(d_q, hq.data(), sz * sizeof(half), cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(
      cudaMemcpy(d_k, hk.data(), sz * sizeof(half), cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(
      cudaMemcpy(d_v, hv.data(), sz * sizeof(half), cudaMemcpyHostToDevice));

  const size_t smem_bytes =
      (BrC * (kHeadDimC + kPadC) + kStagesKC * BcC * (kHeadDimC + kPadC) +
       BcC * (kHeadDimC + kPadC)) *
      sizeof(half);
  dim3 block(kWarpSize * kMmaTileSeqLenQC * kMmaTileSeqLenKC);
  dim3 grid((N + BrC - 1) / BrC, B * H);
  cudaEvent_t beg, end;
  BOOK_CUDA_CHECK(cudaEventCreate(&beg));
  BOOK_CUDA_CHECK(cudaEventCreate(&end));
  const double flops = 4.0 * B * H * N * N * kHeadDimC;

  for (int acc = 0; acc <= 1; ++acc) {
    FAKernel fa_k = acc == 0 ? make_fa_kernel<0>() : make_fa_kernel<1>();
    cudaFuncSetAttribute(fa_k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);
    for (int w = 0; w < warmup; ++w)
      fa_k<<<grid, block, smem_bytes>>>(d_q, d_k, d_v, d_o, N, H);
    BOOK_CUDA_CHECK(cudaEventRecord(beg));
    for (int i = 0; i < iters; ++i)
      fa_k<<<grid, block, smem_bytes>>>(d_q, d_k, d_v, d_o, N, H);
    BOOK_CUDA_CHECK(cudaEventRecord(end));
    BOOK_CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0.0f;
    BOOK_CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    ms /= iters;
    double tflops = flops / (double(ms) * 1e-3) / 1e12;
    printf("fa2 splitq %s: %.4f ms/iter, %.2f TFLOPS\n",
           acc == 0 ? "F16Acc" : "F32Acc", ms, tflops);
  }

  BOOK_CUDA_CHECK(cudaEventDestroy(beg));
  BOOK_CUDA_CHECK(cudaEventDestroy(end));
  BOOK_CUDA_CHECK(cudaFree(d_q));
  BOOK_CUDA_CHECK(cudaFree(d_k));
  BOOK_CUDA_CHECK(cudaFree(d_v));
  BOOK_CUDA_CHECK(cudaFree(d_o));
  return 0;
}

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "bench") return bench();
  if (!book_require_sm(80, "ch16")) return 0;  // mma m16n8k16 需 sm_80+

  run_case("A: B=1 H=2 N=256 D=64", 1, 2, 256);
  run_case("B: B=1 H=1 N=512 D=64", 1, 1, 512);

  printf(g_failures == 0 ? "ALL OK\n" : "FAILURES PRESENT\n");
  return g_failures == 0 ? 0 : 1;
}
