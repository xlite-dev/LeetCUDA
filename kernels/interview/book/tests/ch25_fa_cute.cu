// book/tests/ch25_fa_cute.cu — ch25 最小测试：CuTe 版 FlashAttention（实现 1，cp.async 统一流水）
// 覆盖：
//   case A: D=64  kStagesK=2  B=1 H=2 N=128（单 Q tile，KV 2 段）
//   case B: D=64  kStagesK=2  B=1 H=2 N=256（多 Q tile，grid (2,2)）
//   case C: D=128 kStagesK=2  B=1 H=2 N=128（大 D 路径；smem 80KB < 99KB）
// 参考实现：CPU fp64 的 softmax(Q K^T / sqrt(D)) V（half 输入/输出，__half2float 比较）
// 约束：N % 128 == 0（kBr=128）且 N % 64 == 0（kBc=64）；D ∈ {64,128}（kernel static_assert）
// include 顺序：flash_attn.cuh 依赖 hgemm.cuh 的 swizzle 派发器（ch16 测试同序）
#define NOTES_V2_ENABLE_CUTE 1
#include "../../hgemm.cuh"
#include "../../flash_attn.cuh"
#include "common_test.h"
#include <string>
#include <vector>

using namespace cute;

// CPU fp64 参考：O = softmax(Q K^T / sqrt(D)) V，逐 (b,h,q) 行直接算（packed BHND 布局）
static void fa_cute_ref_fp64(const half *Q, const half *K, const half *V,
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
          s += (double)__half2float(q[(size_t)qi * D + d]) *
               (double)__half2float(k[(size_t)kj * D + d]);
        S[kj] = s * scale;
        if (S[kj] > smax) smax = S[kj];
      }
      double sum_exp = 0.0;
      for (int kj = 0; kj < N; ++kj) sum_exp += exp(S[kj] - smax);
      const double inv = 1.0 / sum_exp;
      for (int d = 0; d < D; ++d) {
        double o = 0.0;
        for (int kj = 0; kj < N; ++kj)
          o += exp(S[kj] - smax) * inv * (double)__half2float(v[(size_t)kj * D + d]);
        O[(size_t)bh * N * D + (size_t)qi * D + d] = o;
      }
    }
  }
}

template <int kHeadDim, int kStagesK>
static void run_case(const char *tag, int B, int H, int N) {
  using Traits = fa_cute::FlashAttn2CuTeTraits<kHeadDim>;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  constexpr int kBr = 128;
  constexpr int kBc = 64;

  const size_t smem_bytes =
      (size_t(size(SmemLayoutQ{})) + size_t(kStagesK) * size(SmemLayoutKV{}) +
       size(SmemLayoutKV{})) *
      sizeof(cutlass::half_t);

  auto fk = flash_attn_mma_stages_split_q_cute<kHeadDim, kStagesK>;
  int dev = 0, max_smem = 0;
  cudaFuncAttributes attr{};
  BOOK_CUDA_CHECK(cudaGetDevice(&dev));
  BOOK_CUDA_CHECK(
      cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  BOOK_CUDA_CHECK(cudaFuncGetAttributes(&attr, fk));
  if (smem_bytes + attr.sharedSizeBytes > size_t(max_smem)) {
    printf("SKIP(ch25): %s smem %zu B > optin %d B\n", tag, smem_bytes, max_smem);
    return;
  }

  const size_t sz = (size_t)B * H * N * kHeadDim;
  std::vector<half> hq(sz), hk(sz), hv(sz), ho(sz);
  std::vector<double> ref(sz);
  srand(42);
  for (size_t i = 0; i < sz; ++i) {
    hq[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    hk[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    hv[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  }
  fa_cute_ref_fp64(hq.data(), hk.data(), hv.data(), ref.data(), B, H, N, kHeadDim);

  half *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
  BOOK_CUDA_CHECK(cudaMalloc(&d_q, sz * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_k, sz * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_v, sz * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_o, sz * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMemcpy(d_q, hq.data(), sz * sizeof(half), cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_k, hk.data(), sz * sizeof(half), cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_v, hv.data(), sz * sizeof(half), cudaMemcpyHostToDevice));

  BOOK_CUDA_CHECK(cudaFuncSetAttribute(
      fk, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
  dim3 grid(N / kBr, B * H);
  fk<<<grid, 256, smem_bytes>>>(reinterpret_cast<cutlass::half_t *>(d_q),
                                reinterpret_cast<cutlass::half_t *>(d_k),
                                reinterpret_cast<cutlass::half_t *>(d_v),
                                reinterpret_cast<cutlass::half_t *>(d_o),
                                B * H * N, N);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(cudaMemcpy(ho.data(), d_o, sz * sizeof(half), cudaMemcpyDeviceToHost));

  std::vector<float> out(sz);
  for (size_t i = 0; i < sz; ++i) out[i] = __half2float(ho[i]);
  char name[128];
  snprintf(name, sizeof(name), "fa_cute cp.async D=%d Sk=%d B=%d H=%d N=%d", kHeadDim,
           kStagesK, B, H, N);
  book_check(out.data(), ref.data(), (int)sz, TOL_F16ACC, name);

  BOOK_CUDA_CHECK(cudaFree(d_q));
  BOOK_CUDA_CHECK(cudaFree(d_k));
  BOOK_CUDA_CHECK(cudaFree(d_v));
  BOOK_CUDA_CHECK(cudaFree(d_o));
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  if (!book_require_sm(80, "ch25")) return 0;  // mma m16n8k16 + cp.async 需 sm_80+

  run_case<64, 2>("A: D=64  N=128", 1, 2, 128);
  run_case<64, 2>("B: D=64  N=256", 1, 2, 256);
  run_case<128, 2>("C: D=128 N=128", 1, 2, 128);

  printf(g_failures == 0 ? "ALL OK\n" : "FAILURES PRESENT\n");
  return g_failures == 0 ? 0 : 1;
}
