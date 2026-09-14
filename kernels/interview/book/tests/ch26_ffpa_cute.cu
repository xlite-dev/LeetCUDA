// book/tests/ch26_ffpa_cute.cu — ch26 最小测试：FFPA Split-D 的 CuTe 类型断言 + 小规模正确性
// 覆盖：
//   case A (host): D-chunk 常量与坐标语义 —— D ∈ {64,128,192,512} 的 kDChunks、chunk 池
//                  cosize、local_tile(qt, d) 的首元素地址 = qt*64*D + d*64
//   case B (host): 双 TiledMma 推导链 —— QK/PV tile_size_mnk、线程数 128、两个 C fragment
//                  每线程 32 值、rowcol (2,16)、Aregs 重解释前后元素数守恒（32）
//   case C (kernel): D=192 Sk=3 Sv=2 B=1 H=2 N=128（3 chunk 且与 Sk 相等——预取边界）
//   case D (kernel): D=128 Sk=2 Sv=2 B=1 H=2 N=128
// 参考实现：CPU fp64 的 softmax(Q K^T / sqrt(D)) V；容差 TOL_F16ACC
// 约束：N % 64 == 0 且 head_dim % 64 == 0（Kernel static_assert）
// include 顺序与 ch19/ch22 测试一致（flash_attn.cuh ← hgemm.cuh 的 swizzle 派发器）
#define NOTES_V2_ENABLE_CUTE 1
#define NOTES_V2_ENABLE_TMA_MMA_WS 1
#include "../../hgemm.cuh"
#include "../../flash_attn.cuh"
#include "../../ffpa_attn.cuh"
#include "common_test.h"
#include <string>
#include <vector>

using namespace cute;

static int g_checks = 0;
static int g_mism = 0;

#define CHK(cond) do { ++g_checks; if (!(cond)) ++g_mism; } while (0)

static void case_end(const char *name) {
  printf("%s %s: mismatches=%d (%d checks)\n", g_mism == 0 ? "PASS" : "FAIL", name,
         g_mism, g_checks);
  if (g_mism) ++g_failures;
  g_mism = 0;
  g_checks = 0;
}

// ---- case A：D-chunk 常量与坐标语义（host）----
template <int kHeadDim>
static void case_chunk_constants() {
  using Traits = fa_cute::FFPAAttnSplitDCuTeTraits<kHeadDim>;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  constexpr int kDChunk = 64;
  constexpr int kDChunks = kHeadDim / kDChunk;
  static_assert(kHeadDim % kDChunk == 0, "chunk must divide head_dim");
  CHK(cosize(SmemLayoutQ{}) == 64 * 64);
  CHK(cosize(SmemLayoutKV{}) == 64 * 64);
  static_assert(kDChunks >= 1, "at least one chunk");
  // local_tile 的 D-chunk 坐标语义：在 [rows, D] 的 gmem tensor 上取 (row_tile, d) 块，
  // 首元素地址 = row_tile*64*D + d*64（与 kernel 内 g2s_load_q 的用法一致）
  const int rows = 256;
  std::vector<half> buf((size_t)rows * kHeadDim, half(0));
  auto *base = reinterpret_cast<cutlass::half_t *>(buf.data());
  auto mQ = make_tensor(make_gmem_ptr(base), make_shape(rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  for (int qt = 0; qt < 2; ++qt) {
    for (int d = 0; d < kDChunks; ++d) {
      auto gQ = local_tile(mQ, Shape<_64, _64>{}, make_coord(qt, d));
      size_t got = (size_t)(raw_pointer_cast(gQ.data()) - base);
      size_t want = (size_t)qt * 64 * kHeadDim + (size_t)d * 64;
      CHK(got == want);
    }
  }
  char name[96];
  snprintf(name, sizeof(name), "case A chunk constants D=%d (kDChunks=%d)", kHeadDim,
           kDChunks);
  case_end(name);
}

// ---- case B：双 TiledMma 推导链与 fragment 恒等式（host）----
static void case_dual_tiledmma() {
  using Traits = fa_cute::FFPAAttnSplitDCuTeTraits<512>;
  using TiledMmaQK = typename Traits::TiledMmaQK;
  using TiledMmaPV = typename Traits::TiledMmaPV;

  // tile 形状与线程数（128 = 4 warp，与 AtomLayout (4,1,1) 一致）
  static_assert(tile_size<0>(TiledMmaQK{}) == 64 && tile_size<1>(TiledMmaQK{}) == 64 &&
                    tile_size<2>(TiledMmaQK{}) == 16,
                "QK tile 64x64x16");
  static_assert(tile_size<0>(TiledMmaPV{}) == 64 && tile_size<1>(TiledMmaPV{}) == 16 &&
                    tile_size<2>(TiledMmaPV{}) == 16,
                "PV tile 64x16x16");
  CHK((int)size(TiledMmaQK{}) == 128);
  CHK((int)size(TiledMmaPV{}) == 128);

  // 恒等式一：两个 C fragment 的每线程元素数都 = |C 形状| / 线程数 = 64*64/128 = 32
  using SFragType = decltype(partition_fragment_C(TiledMmaQK{}, Shape<_64, _64>{}));
  using OFragType = decltype(partition_fragment_C(TiledMmaPV{}, Shape<_64, _64>{}));
  constexpr int kSElems = decltype(size(SFragType{}))::value;
  constexpr int kOElems = decltype(size(OFragType{}))::value;
  static_assert(kSElems == 32 && kOElems == 32, "C fragment 32 f32/thread");
  CHK(kSElems == 32);
  CHK(kOElems == 32);

  // C fragment 的形状分解：((2,2), 1, 8) —— 由 atom 值数 (2,2) 与 N 向重复 8 决定
  using SFragLayout = typename SFragType::layout_type;
  using OFragLayout = typename OFragType::layout_type;
  CHK((int)size<0>(SFragLayout{}) == 4 && (int)size<1>(SFragLayout{}) == 1 &&
      (int)size<2>(SFragLayout{}) == 8);
  CHK((int)size<0>(OFragLayout{}) == 4 && (int)size<1>(OFragLayout{}) == 1 &&
      (int)size<2>(OFragLayout{}) == 8);

  case_end("case B dual TiledMma derivation chain (D=512)");
}

// rowcol / Aregs 是 CUTE_DEVICE 工具函数，host 端无法直接引用——用单线程
// device probe 调用真实实现并回传结果，host 端比对期望值（26.3.2/26.3.3 节）。
__global__ void ch26_layout_probe(int *out) {
  using Traits = fa_cute::FFPAAttnSplitDCuTeTraits<512>;
  using TiledMmaQK = typename Traits::TiledMmaQK;
  using TiledMmaPV = typename Traits::TiledMmaPV;
  using SFragType = decltype(partition_fragment_C(TiledMmaQK{}, Shape<_64, _64>{}));
  using SFragLayout = typename SFragType::layout_type;
  auto rc = fa_cute::convert_layout_acc_rowcol(SFragLayout{});
  out[0] = (int)size<0>(rc);   // 期望 2  (kORows)
  out[1] = (int)size<1>(rc);   // 期望 16 (kOCols)
  using PLayout = decltype(fa_cute::convert_layout_acc_Aregs<TiledMmaPV>(SFragLayout{}));
  out[2] = (int)size(PLayout{});  // 期望 32（元素数守恒，零拷贝）
}

static void case_rowcol_aregs_probe() {
  int *d_out = nullptr;
  int h_out[4] = {0, 0, 0, 0};
  BOOK_CUDA_CHECK(cudaMalloc(&d_out, sizeof(int) * 4));
  ch26_layout_probe<<<1, 32>>>(d_out);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  BOOK_CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(int) * 4, cudaMemcpyDeviceToHost));
  BOOK_CUDA_CHECK(cudaFree(d_out));
  printf("probe: rowcol=(%d,%d) aregs_elems=%d\n", h_out[0], h_out[1], h_out[2]);
  CHK(h_out[0] == 2);
  CHK(h_out[1] == 16);
  CHK(h_out[2] == 32);
  case_end("case B2 device probe: rowcol (2,16) & Aregs 32 elems");
}

// ---- kernel 正确性：ffpa_split_d_cute（cp.async 版，128 线程）----
static void ffpa_cute_ref_fp64(const std::vector<half> &hq, const std::vector<half> &hk,
                               const std::vector<half> &hv, std::vector<double> &ref,
                               int B, int H, int N, int D) {
  const double scale = 1.0 / sqrt((double)D);
  std::vector<double> s(N), p(N);
  for (int bi = 0; bi < B * H; ++bi) {
    const size_t base = size_t(bi) * N * D;
    for (int qi = 0; qi < N; ++qi) {
      double smax = -INFINITY;
      for (int kj = 0; kj < N; ++kj) {
        double acc = 0.0;
        for (int d = 0; d < D; ++d)
          acc += __half2float(hq[base + size_t(qi) * D + d]) *
                 __half2float(hk[base + size_t(kj) * D + d]);
        s[kj] = acc * scale;
        if (s[kj] > smax) smax = s[kj];
      }
      double sum = 0.0;
      for (int kj = 0; kj < N; ++kj) {
        p[kj] = exp(s[kj] - smax);
        sum += p[kj];
      }
      for (int d = 0; d < D; ++d) {
        double acc = 0.0;
        for (int kj = 0; kj < N; ++kj)
          acc += p[kj] * __half2float(hv[base + size_t(kj) * D + d]);
        ref[base + size_t(qi) * D + d] = acc / sum;
      }
    }
  }
}

template <int kHeadDim, int Sk, int Sv>
static void run_ffpa_case(const char *tag, int B, int H, int N) {
  using Traits = fa_cute::FFPAAttnSplitDCuTeTraits<kHeadDim>;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  constexpr int kBr = 64;

  const size_t smem_bytes =
      (size_t(Sk) * (cosize(SmemLayoutQ{}) + cosize(SmemLayoutKV{})) +
       size_t(Sv) * cosize(SmemLayoutKV{})) *
      sizeof(cutlass::half_t);

  auto fk = ffpa_split_d_cute<kHeadDim, Sk, Sv>;
  int dev = 0, max_smem = 0;
  cudaFuncAttributes attr{};
  BOOK_CUDA_CHECK(cudaGetDevice(&dev));
  BOOK_CUDA_CHECK(
      cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  BOOK_CUDA_CHECK(cudaFuncGetAttributes(&attr, fk));
  if (smem_bytes + attr.sharedSizeBytes > size_t(max_smem)) {
    printf("SKIP(ch26): %s smem %zu B > optin %d B\n", tag, smem_bytes, max_smem);
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
  ffpa_cute_ref_fp64(hq, hk, hv, ref, B, H, N, kHeadDim);

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
  fk<<<grid, 128, smem_bytes>>>(reinterpret_cast<cutlass::half_t *>(d_q),
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
  snprintf(name, sizeof(name), "ffpa split-d cute D=%d N=%d Sk=%d Sv=%d B=%d H=%d", kHeadDim,
           N, Sk, Sv, B, H);
  book_check(out.data(), ref.data(), (int)sz, TOL_F16ACC, name);

  BOOK_CUDA_CHECK(cudaFree(d_q));
  BOOK_CUDA_CHECK(cudaFree(d_k));
  BOOK_CUDA_CHECK(cudaFree(d_v));
  BOOK_CUDA_CHECK(cudaFree(d_o));
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  if (!book_require_sm(80, "ch26")) return 0;

  case_chunk_constants<64>();
  case_chunk_constants<128>();
  case_chunk_constants<192>();
  case_chunk_constants<512>();
  case_dual_tiledmma();
  case_rowcol_aregs_probe();

  run_ffpa_case<192, 3, 2>("C: D=192 N=128 Sk=3 Sv=2", 1, 2, 128);
  run_ffpa_case<128, 2, 2>("D: D=128 N=128 Sk=2 Sv=2", 1, 2, 128);

  printf(g_failures == 0 ? "ALL OK\n" : "FAILURES PRESENT\n");
  return g_failures == 0 ? 0 : 1;
}
