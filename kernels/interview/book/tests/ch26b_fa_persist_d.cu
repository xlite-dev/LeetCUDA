// book/tests/ch26b_fa_persist_d.cu — ch26b 最小测试：sm_120 persist-D
// FlashAttention（WS 1P+1C + persistent CTA + softmax scale*fused）
// 覆盖：
//   case A: D=64  dense    B=1 H=2  N=256（整 q-tile）
//   case B: D=128 dense    B=1 H=2  N=256
//   case C: D=128 causal   B=1 H=2  N=512（Nkv == Nq）
//   case D: D=64  GQA      B=1 H=4  Hkv=2 N=256
//   case E: D=128 Q 尾部 tile B=1 H=2 Nq=300 Nkv=512（R->G 路径）
//   case F: D=64  persistent 多 iter B=1 H=32 N=512（128 tiles > 96 SM，
//           触发 epi_done wait + kv_cursor 跨 q-tile 累计路径）
//   case G: D=128 KV 尾 mask B=1 H=2 Nq=256 Nkv=300（Nkv 非 kBc 倍数，
//           触发 kv_valid < kBc 越界列 mask 分支）
//   case H: D=64  causal 正向滑窗 Nq=256 Nkv=512（kv_offset>0，
//           Tc_eff/mask_start_tile 的滑窗边界）
//   case I: D=128 causal 反向 Nq=512 Nkv=256（kv_offset<0，前段 q 行全
//           mask：Tc_eff clamp 到 0 + row_sum=0 输出 0 的回归）
//   case J: D=96  dense B=1 H=2 N=256（D=96 分派数值验证）
// 参考实现：CPU fp64 的 softmax(scale·Q K^T [+ causal]) V（GQA head 映射）
// 约束：Nq/Nkv 满足 TMA 对齐（kBc=64/128 的倍数或尾部 guard）；D ∈ {64,96,128}
// 编译：-gencode arch=compute_120f,code=sm_120f（sm_120a 同样保留 setmaxnreg：
// 生死取决于 TMA dst 用 shared::cta 与 launch_bounds(N,1)，与 arch 后缀无关）
#define NOTES_V2_ENABLE_CUTE 1
#include "../../hgemm.cuh"
#include "../../flash_attn.cuh"
#include "common_test.h"
#include <string>
#include <vector>

using namespace cute;

// CPU fp64 参考（BHND packed）：O = softmax(scale * Q K^T [+ causal]) V
// causal 约定与 kernel 一致：k_pos > q_pos + (Nkv - Nq) 的分数被 mask。
static void fa_pd_ref_fp64(const half *Q, const half *K, const half *V,
                            double *O, int B, int H, int Hkv, int Nq, int Nkv,
                            int D, bool causal) {
  const double scale = 1.0 / sqrt((double)D);
  const int kv_offset = Nkv - Nq;
  std::vector<double> S(Nkv);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      const int hk = h / (H / Hkv);  // GQA head 映射
      const half *q = Q + ((size_t)b * H + h) * Nq * D;
      const half *k = K + ((size_t)b * Hkv + hk) * Nkv * D;
      const half *v = V + ((size_t)b * Hkv + hk) * Nkv * D;
      double *o = O + ((size_t)b * H + h) * Nq * D;
      for (int qi = 0; qi < Nq; ++qi) {
        double smax = -INFINITY;
        for (int kj = 0; kj < Nkv; ++kj) {
          if (causal && kj > qi + kv_offset) {
            S[kj] = -INFINITY;
            continue;
          }
          double s = 0.0;
          for (int d = 0; d < D; ++d)
            s += (double)__half2float(q[(size_t)qi * D + d]) *
                 (double)__half2float(k[(size_t)kj * D + d]);
          S[kj] = s * scale;
          if (S[kj] > smax) smax = S[kj];
        }
        double sum_exp = 0.0;
        for (int kj = 0; kj < Nkv; ++kj) {
          S[kj] = exp(S[kj] - smax);  // softmax 权重 (exp(-inf)=0)
          sum_exp += S[kj];
        }
        // 全 mask 行 (causal Nkv<Nq 前段): 与 kernel 一致输出 0 而非 NaN
        if (sum_exp == 0.0) {
          for (int d = 0; d < D; ++d) o[(size_t)qi * D + d] = 0.0;
          continue;
        }
        const double inv = 1.0 / sum_exp;
        for (int d = 0; d < D; ++d) {
          double acc = 0.0;
          for (int kj = 0; kj < Nkv; ++kj)
            acc += S[kj] * (double)__half2float(v[(size_t)kj * D + d]);
          o[(size_t)qi * D + d] = acc * inv;
        }
      }
    }
}

static void run_case(const char *tag, int D, int B, int H, int Hkv, int Nq,
                     int Nkv, bool causal) {
  const size_t sz_q = (size_t)B * H * Nq * D;
  const size_t sz_kv = (size_t)B * Hkv * Nkv * D;
  std::vector<half> hq(sz_q), hk(sz_kv), hv(sz_kv), ho(sz_q);
  std::vector<double> ref(sz_q);
  srand(42 + D + (causal ? 7 : 0) + H * 100);
  for (size_t i = 0; i < sz_q; ++i)
    hq[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  for (size_t i = 0; i < sz_kv; ++i) {
    hk[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    hv[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  }
  fa_pd_ref_fp64(hq.data(), hk.data(), hv.data(), ref.data(), B, H, Hkv, Nq,
                 Nkv, D, causal);

  half *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
  BOOK_CUDA_CHECK(cudaMalloc(&d_q, sz_q * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_k, sz_kv * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_v, sz_kv * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_o, sz_q * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMemcpy(d_q, hq.data(), sz_q * sizeof(half), cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_k, hk.data(), sz_kv * sizeof(half), cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_v, hv.data(), sz_kv * sizeof(half), cudaMemcpyHostToDevice));

  flash_attn_cute_persist_d_sm120_fwd(
      reinterpret_cast<cutlass::half_t *>(d_q),
      reinterpret_cast<cutlass::half_t *>(d_k),
      reinterpret_cast<cutlass::half_t *>(d_v),
      reinterpret_cast<cutlass::half_t *>(d_o), B, H, Hkv, Nq, Nkv, D, causal,
      (float)(1.0 / sqrt((double)D)));
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(cudaMemcpy(ho.data(), d_o, sz_q * sizeof(half), cudaMemcpyDeviceToHost));

  std::vector<float> out(sz_q);
  for (size_t i = 0; i < sz_q; ++i) out[i] = __half2float(ho[i]);
  char name[128];
  snprintf(name, sizeof(name), "fa persist-D %s D=%d B=%d H=%d/%d Nq=%d Nkv=%d",
           causal ? "causal" : "dense  ", D, B, H, Hkv, Nq, Nkv);
  book_check(out.data(), ref.data(), (int)sz_q, TOL_F16ACC, name);

  BOOK_CUDA_CHECK(cudaFree(d_q));
  BOOK_CUDA_CHECK(cudaFree(d_k));
  BOOK_CUDA_CHECK(cudaFree(d_v));
  BOOK_CUDA_CHECK(cudaFree(d_o));
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  // TMA + setmaxnreg 双 warpgroup 需要 sm_120（Blackwell）以上
  if (!book_require_sm(120, "ch26b")) return 0;

  run_case("A", 64, 1, 2, 2, 256, 256, false);
  run_case("B", 128, 1, 2, 2, 256, 256, false);
  run_case("C", 128, 1, 2, 2, 512, 512, true);
  run_case("D", 64, 1, 4, 2, 256, 256, false);
  run_case("E", 128, 1, 2, 2, 300, 512, false);
  run_case("F", 64, 1, 32, 32, 512, 512, false);
  run_case("G", 128, 1, 2, 2, 256, 300, false);
  run_case("H", 64, 1, 2, 2, 256, 512, true);
  run_case("I", 128, 1, 2, 2, 512, 256, true);
  run_case("J", 96, 1, 2, 2, 256, 256, false);

  printf(g_failures == 0 ? "ALL OK\n" : "FAILURES PRESENT\n");
  return g_failures == 0 ? 0 : 1;
}
