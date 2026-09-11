// book/tests/ch05_merge_attn.cu — ch05 最小测试：merge_attn_states 分块合并
// 测试逻辑抽取自 notes-v2.cu test_merge_attn_states(L723)，正确性对照改为
// CPU fp64「整体 softmax 加权」直接计算：构造两个分块状态 (O_a, LSE_a) 与
// (O_b, LSE_b)，验证 kernel 合并结果 == 全序列 softmax 直接计算（BOOK_PLAN §6）。
// 约束：head_size 必须是 kPackSize=4 的倍数（kernel 隐含假设，见 ch05 常见坑）；
//       输入 O 必须是已按各自分块分母归一化的输出（模拟 FA 分块产出）。
#include "../../base.cuh"
#include "common_test.h"
#include <cmath>
#include <string>
#include <vector>

// 为每个 (token, head) 构造 kv_len = 2*kv_chunk 个 score 与 V，产出一个 case：
//   分块状态：oa/ob [T,H,D]（已归一化）、lse_a/lse_b [H,T]
//   fp64 直接参考：全 kv_len 个 score 一次 softmax 加权 V
static void build_case(int T, int H, int D, int kv_chunk, std::vector<float>& oa,
                       std::vector<float>& ob, std::vector<float>& lse_a,
                       std::vector<float>& lse_b, std::vector<double>& ref) {
  const int kv_len = 2 * kv_chunk;
  oa.assign(size_t(T) * H * D, 0.0f);
  ob.assign(size_t(T) * H * D, 0.0f);
  lse_a.assign(size_t(H) * T, 0.0f);
  lse_b.assign(size_t(H) * T, 0.0f);
  ref.assign(size_t(T) * H * D, 0.0);

  for (int t = 0; t < T; ++t) {
    for (int h = 0; h < H; ++h) {
      // 每 (t,h) 独立 LCG，case 间互不影响
      unsigned seed = unsigned(t * 131 + h * 17 + 7);
      auto rnd = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return float(seed % 2000u) / 1000.0f - 1.0f;  // [-1, 1]
      };
      std::vector<float> s(kv_len);
      std::vector<float> v(size_t(kv_len) * D);
      for (int j = 0; j < kv_len; ++j) s[j] = rnd() * 10.0f;
      for (size_t j = 0; j < v.size(); ++j) v[j] = rnd();

      // fp64 直接参考：整体 softmax 加权（合并结果应逐元素等于它）
      double m = -DBL_MAX;
      for (int j = 0; j < kv_len; ++j) m = std::max(m, double(s[j]));
      double ell = 0.0;
      for (int j = 0; j < kv_len; ++j) ell += std::exp(double(s[j]) - m);
      const size_t off = (size_t(t) * H + h) * D;
      for (int d = 0; d < D; ++d) {
        double acc = 0.0;
        for (int j = 0; j < kv_len; ++j)
          acc += std::exp(double(s[j]) - m) / ell *
                 double(v[size_t(j) * D + d]);
        ref[off + d] = acc;
      }

      // 分块状态（float，模拟 FA 分块输出：O 已除以各自分块分母）
      float ma = -FLT_MAX, mb = -FLT_MAX;
      for (int j = 0; j < kv_chunk; ++j) ma = std::max(ma, s[j]);
      for (int j = kv_chunk; j < kv_len; ++j) mb = std::max(mb, s[j]);
      double la = 0.0, lb = 0.0;
      for (int j = 0; j < kv_chunk; ++j) la += std::exp(double(s[j]) - ma);
      for (int j = kv_chunk; j < kv_len; ++j) lb += std::exp(double(s[j]) - mb);
      lse_a[size_t(h) * T + t] = float(ma + std::log(la));
      lse_b[size_t(h) * T + t] = float(mb + std::log(lb));
      for (int d = 0; d < D; ++d) {
        double pa = 0.0, pb = 0.0;
        for (int j = 0; j < kv_chunk; ++j)
          pa += std::exp(double(s[j]) - ma) * double(v[size_t(j) * D + d]);
        for (int j = kv_chunk; j < kv_len; ++j)
          pb += std::exp(double(s[j]) - mb) * double(v[size_t(j) * D + d]);
        oa[off + d] = float(pa / la);
        ob[off + d] = float(pb / lb);
      }
    }
  }
}

static void run_case(const char* tag, int T, int H, int D, int kv_chunk) {
  std::vector<float> oa, ob, lse_a, lse_b;
  std::vector<double> ref;
  build_case(T, H, D, kv_chunk, oa, ob, lse_a, lse_b, ref);

  const size_t out_bytes = size_t(T) * H * D * sizeof(float);
  const size_t lse_bytes = size_t(H) * T * sizeof(float);
  float *d_oa, *d_ob, *d_la, *d_lb, *d_out;
  BOOK_CUDA_CHECK(cudaMalloc(&d_oa, out_bytes));
  BOOK_CUDA_CHECK(cudaMalloc(&d_ob, out_bytes));
  BOOK_CUDA_CHECK(cudaMalloc(&d_la, lse_bytes));
  BOOK_CUDA_CHECK(cudaMalloc(&d_lb, lse_bytes));
  BOOK_CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
  BOOK_CUDA_CHECK(cudaMemcpy(d_oa, oa.data(), out_bytes, cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_ob, ob.data(), out_bytes, cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_la, lse_a.data(), lse_bytes, cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_lb, lse_b.data(), lse_bytes, cudaMemcpyHostToDevice));

  const int threads_per_head = D / 4;
  const int total_threads = T * H * threads_per_head;
  dim3 grid((total_threads + 127) / 128), block(128);
  merge_attn_states<<<grid, block>>>(d_out, d_oa, d_la, d_ob, d_lb, T, H, D);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(size_t(T) * H * D);
  BOOK_CUDA_CHECK(
      cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  printf("case %s\n", tag);
  book_check(h_out.data(), ref.data(), int(T) * H * D, TOL_F32ACC,
             "merge == direct full-softmax");

  // 边界：prefix LSE = +inf（空 attention 段）-> 该 (t=0,h=0) 段退化为 suffix
  std::vector<float> lse_inf = lse_a, lse_zero = lse_b;
  lse_inf[0] = INFINITY;  // layout [H,T]: (h=0, t=0)
  lse_zero[0] = 0.0f;
  BOOK_CUDA_CHECK(
      cudaMemcpy(d_la, lse_inf.data(), lse_bytes, cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(
      cudaMemcpy(d_lb, lse_zero.data(), lse_bytes, cudaMemcpyHostToDevice));
  merge_attn_states<<<grid, block>>>(d_out, d_oa, d_la, d_ob, d_lb, T, H, D);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(
      cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  // (t=0,h=0) 段位于展平输出起始 D 个元素；alpha=0, beta=1 时应精确等于 ob
  double inf_err = 0.0;
  for (int d = 0; d < D; ++d)
    inf_err = std::max(inf_err,
                       std::fabs(double(h_out[d]) - double(ob[size_t(d)])));
  bool pass = inf_err <= 1e-6;
  printf("%s inf-lse -> suffix only: max_abs_err=%.3e (tol=1.0e-06)\n",
         pass ? "PASS" : "FAIL", inf_err);
  if (!pass) g_failures++;

  BOOK_CUDA_CHECK(cudaFree(d_oa));
  BOOK_CUDA_CHECK(cudaFree(d_ob));
  BOOK_CUDA_CHECK(cudaFree(d_la));
  BOOK_CUDA_CHECK(cudaFree(d_lb));
  BOOK_CUDA_CHECK(cudaFree(d_out));
}

static int bench() {
  const int T = 16384, H = 32, D = 128, warmup = 3, iters = 10;
  cudaDeviceProp prop;
  BOOK_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  printf("shape: num_tokens=%d num_heads=%d head_size=%d, "
         "warmup=%d, iters=%d (avg)\n",
         T, H, D, warmup, iters);
  const size_t out_bytes = size_t(T) * H * D * sizeof(float);
  const size_t lse_bytes = size_t(H) * T * sizeof(float);
  float *d_oa, *d_ob, *d_la, *d_lb, *d_out;
  BOOK_CUDA_CHECK(cudaMalloc(&d_oa, out_bytes));
  BOOK_CUDA_CHECK(cudaMalloc(&d_ob, out_bytes));
  BOOK_CUDA_CHECK(cudaMalloc(&d_la, lse_bytes));
  BOOK_CUDA_CHECK(cudaMalloc(&d_lb, lse_bytes));
  BOOK_CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
  std::vector<float> h_o(size_t(T) * H * D), h_l(size_t(H) * T);
  book_fill_rand(h_o.data(), int(h_o.size()));
  for (size_t i = 0; i < h_l.size(); ++i) h_l[i] = float(i % 21) - 5.0f;
  BOOK_CUDA_CHECK(cudaMemcpy(d_oa, h_o.data(), out_bytes, cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_ob, h_o.data(), out_bytes, cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_la, h_l.data(), lse_bytes, cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_lb, h_l.data(), lse_bytes, cudaMemcpyHostToDevice));

  const int total_threads = T * H * (D / 4);
  dim3 grid((total_threads + 127) / 128), block(128);
  for (int w = 0; w < warmup; ++w)
    merge_attn_states<<<grid, block>>>(d_out, d_oa, d_la, d_ob, d_lb, T, H, D);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t beg, end;
  BOOK_CUDA_CHECK(cudaEventCreate(&beg));
  BOOK_CUDA_CHECK(cudaEventCreate(&end));
  BOOK_CUDA_CHECK(cudaEventRecord(beg));
  for (int i = 0; i < iters; ++i)
    merge_attn_states<<<grid, block>>>(d_out, d_oa, d_la, d_ob, d_lb, T, H, D);
  BOOK_CUDA_CHECK(cudaEventRecord(end));
  BOOK_CUDA_CHECK(cudaEventSynchronize(end));
  float ms = 0.0f;
  BOOK_CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
  ms /= iters;
  const double bytes =
      (3.0 * double(out_bytes) + 2.0 * double(lse_bytes));
  printf("merge_attn_states: %.4f ms/iter, %.1f GB/s (%.1f MB moved/iter)\n",
         ms, bytes / (double(ms) * 1e6), bytes / 1e6);
  BOOK_CUDA_CHECK(cudaEventDestroy(beg));
  BOOK_CUDA_CHECK(cudaEventDestroy(end));
  cudaFree(d_oa); cudaFree(d_ob); cudaFree(d_la); cudaFree(d_lb); cudaFree(d_out);
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "bench") return bench();

  // Case A: 源码头注释的示例形态（threads_per_head = 2, total_threads = 8）
  run_case("A: T=2 H=2 D=8 kv=2x32", 2, 2, 8, 32);
  // Case B: 现实形态 head_size=128（threads_per_head = 32）
  run_case("B: T=3 H=4 D=128 kv=2x32", 3, 4, 128, 32);

  printf(g_failures == 0 ? "ALL OK\n" : "FAILURES PRESENT\n");
  return g_failures == 0 ? 0 : 1;
}
