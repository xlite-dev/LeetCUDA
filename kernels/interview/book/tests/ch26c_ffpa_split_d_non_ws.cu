// book/tests/ch26c_ffpa_split_d_non_ws.cu — ch26c 最小测试：SM120 non-WS
// TMA Split-D FlashAttention（tile 128x128 + tid=0 内联发 TMA + 双 barrier
// 流水 + STSM/TMA store epilogue + FA-4 conditional rescale）
// 覆盖：
//   case A: D=128 B=1 H=2 Nq=256 Nkv=256 s2（基础 dense）
//   case B: D=320 B=1 H=2 Nq=256 Nkv=256 s2 + LSE 校验（主规格小号）
//   case C: D=192 B=2 H=2 Nq=128 Nkv=256 s2（多 batch + Nq != Nkv）
//   case D: D=320 B=1 H=2 Nq=128 Nkv=192 s2 + LSE 校验（Nkv=192 非 kBc=128
//           倍数，尾 tile kv_valid=64 < kBc，触发边界 mask 分支）
//   case E: D=320 B=1 H=2 Nq=256 Nkv=256 s3（kStages=3 深流水，
//           kNBatches 1 vs 5 的 epilogue 批量化差异）
//   case F: D=128 B=1 H=4 Nq=512 Nkv=512 s2（4 个 kv tiles，跨 tile 预取）
// 参考实现：CPU fp64 的 softmax(scale·Q K^T) V + log-sum-exp（无 cuBLAS 依赖）
// 约束：Nq % 128 == 0（kernel O epilogue 整 tile 约定）；D % 64 == 0
// 编译：-gencode arch=compute_120a,code=sm_120a（TMA/LDSM 仅需 sm_90+）
#define NOTES_V2_ENABLE_CUTE 1
#define NOTES_V2_ENABLE_TMA_MMA_WS 1
#include "../../hgemm.cuh"
#include "../../ffpa_attn.cuh"
#include "common_test.h"
#include <vector>

using namespace cute;

// CPU fp64 参考（BHND packed）：O = softmax(scale * Q K^T) V；
// lse_ref 非 null 时同时输出 log-sum-exp（ln 域，与 kernel LSE 一致）。
static void fa_sd_ref_fp64(const half *Q, const half *K, const half *V,
                           double *O, double *lse_ref, int B, int H, int Nq,
                           int Nkv, int D) {
  const double scale = 1.0 / sqrt((double)D);
  std::vector<double> S(Nkv);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      const half *q = Q + ((size_t)b * H + h) * Nq * D;
      const half *k = K + ((size_t)b * H + h) * Nkv * D;
      const half *v = V + ((size_t)b * H + h) * Nkv * D;
      double *o = O + ((size_t)b * H + h) * Nq * D;
      double *l = lse_ref ? lse_ref + ((size_t)b * H + h) * Nq : nullptr;
      for (int qi = 0; qi < Nq; ++qi) {
        double smax = -INFINITY;
        for (int kj = 0; kj < Nkv; ++kj) {
          double s = 0.0;
          for (int d = 0; d < D; ++d)
            s += (double)__half2float(q[(size_t)qi * D + d]) *
                 (double)__half2float(k[(size_t)kj * D + d]);
          S[kj] = s * scale;
          if (S[kj] > smax) smax = S[kj];
        }
        double sum_exp = 0.0;
        for (int kj = 0; kj < Nkv; ++kj) {
          S[kj] = exp(S[kj] - smax);
          sum_exp += S[kj];
        }
        const double inv = 1.0 / sum_exp;
        for (int d = 0; d < D; ++d) {
          double acc = 0.0;
          for (int kj = 0; kj < Nkv; ++kj)
            acc += S[kj] * (double)__half2float(v[(size_t)kj * D + d]);
          o[(size_t)qi * D + d] = acc * inv;
        }
        if (l) l[qi] = smax + log(sum_exp);
      }
    }
}

// LSE 单独判定：值域可正可负（ln(sum exp) ± max），绝对容差 0.1
// （fp16 输入分数误差 ~1e-2，经 exp/ln 后放大一档，比 TOL_F16ACC 宽）。
static bool book_check_lse(const float *out, const double *ref, int n,
                           double tol, const char *name) {
  double max_err = 0.0;
  for (int i = 0; i < n; ++i)
    max_err = std::max(max_err, std::fabs(double(out[i]) - ref[i]));
  bool pass = max_err <= tol;
  printf("%s %s: max_abs_err=%.3e (tol=%.1e)\n", pass ? "PASS" : "FAIL", name,
         max_err, tol);
  if (!pass) g_failures++;
  return pass;
}

template <int D, int Sk, int Sv>
static void run_case(const char *tag, int B, int H, int Nq, int Nkv,
                     bool check_lse) {
  const size_t sz_q = (size_t)B * H * Nq * D;
  const size_t sz_kv = (size_t)B * H * Nkv * D;
  std::vector<half> hq(sz_q), hk(sz_kv), hv(sz_kv), ho(sz_q);
  std::vector<double> ref(sz_q);
  std::vector<double> lse_ref(check_lse ? (size_t)B * H * Nq : 0);
  srand(42 + D + Sk * 11 + Sv * 3 + (check_lse ? 7 : 0) + B * 100 + H);
  for (size_t i = 0; i < sz_q; ++i)
    hq[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  for (size_t i = 0; i < sz_kv; ++i) {
    hk[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    hv[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  }
  fa_sd_ref_fp64(hq.data(), hk.data(), hv.data(), ref.data(),
                 check_lse ? lse_ref.data() : nullptr, B, H, Nq, Nkv, D);

  half *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
  float *d_lse = nullptr;
  BOOK_CUDA_CHECK(cudaMalloc(&d_q, sz_q * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_k, sz_kv * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_v, sz_kv * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_o, sz_q * sizeof(half)));
  if (check_lse)
    BOOK_CUDA_CHECK(cudaMalloc(&d_lse, (size_t)B * H * Nq * sizeof(float)));
  BOOK_CUDA_CHECK(cudaMemcpy(d_q, hq.data(), sz_q * sizeof(half), cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_k, hk.data(), sz_kv * sizeof(half), cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_v, hv.data(), sz_kv * sizeof(half), cudaMemcpyHostToDevice));

  ffpa_attn_tma_split_d_cute_fwd<D, Sk, Sv>(
      reinterpret_cast<cutlass::half_t *>(d_q),
      reinterpret_cast<cutlass::half_t *>(d_k),
      reinterpret_cast<cutlass::half_t *>(d_v),
      reinterpret_cast<cutlass::half_t *>(d_o), d_lse, B, H, Nq, Nkv);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(cudaMemcpy(ho.data(), d_o, sz_q * sizeof(half), cudaMemcpyDeviceToHost));

  std::vector<float> out(sz_q);
  for (size_t i = 0; i < sz_q; ++i) out[i] = __half2float(ho[i]);
  char name[128];
  snprintf(name, sizeof(name), "ffpa split-D non-WS %s D=%d B=%d H=%d Nq=%d Nkv=%d s=%d,%d",
           tag, D, B, H, Nq, Nkv, Sk, Sv);
  book_check(out.data(), ref.data(), (int)sz_q, TOL_F16ACC, name);

  if (check_lse) {
    const int n_lse = B * H * Nq;
    std::vector<float> h_lse(n_lse);
    BOOK_CUDA_CHECK(cudaMemcpy(h_lse.data(), d_lse, n_lse * sizeof(float), cudaMemcpyDeviceToHost));
    char lname[160];
    snprintf(lname, sizeof(lname), "ffpa split-D non-WS %s LSE D=%d Nq=%d Nkv=%d", tag, D, Nq, Nkv);
    book_check_lse(h_lse.data(), lse_ref.data(), n_lse, 0.1, lname);
  }

  BOOK_CUDA_CHECK(cudaFree(d_q));
  BOOK_CUDA_CHECK(cudaFree(d_k));
  BOOK_CUDA_CHECK(cudaFree(d_v));
  BOOK_CUDA_CHECK(cudaFree(d_o));
  if (d_lse) BOOK_CUDA_CHECK(cudaFree(d_lse));
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  // TMA + STSM 需要 sm_90 以上；本测试按 sm_120 教学路径验证
  if (!book_require_sm(120, "ch26c")) return 0;

  run_case<128, 2, 2>("A", 1, 2, 256, 256, false);
  run_case<320, 2, 2>("B", 1, 2, 256, 256, true);
  run_case<192, 2, 2>("C", 2, 2, 128, 256, false);
  run_case<320, 2, 2>("D", 1, 2, 128, 192, true);
  run_case<320, 3, 3>("E", 1, 2, 256, 256, false);
  run_case<128, 2, 2>("F", 1, 4, 512, 512, false);
  // K/V 流水深度解耦组合: (Sk=2, Sv=3) / (Sk=3, Sv=2)
  run_case<320, 2, 3>("G", 1, 2, 256, 256, true);
  run_case<320, 3, 2>("H", 1, 2, 128, 192, true);

  printf(g_failures == 0 ? "ALL OK\n" : "FAILURES PRESENT\n");
  return g_failures == 0 ? 0 : 1;
}
