// book/tests/ch12_hgemm_swizzle.cu — ch12 最小测试：XOR swizzle + 寄存器双缓冲 + block swizzle
// 测试逻辑抽取自 notes-v2.cu test_hgemm_swizzle(L1491) 与 test_swizzle_equiv(L4639-4664)，
// 正确性对照改为 CPU fp64 参考（BOOK_PLAN §6，无 cuBLAS/cuDNN 依赖）。
// 覆盖：
//   case A: swizzle kernel 正确性（默认 2D grid，v2 swizzle 路径）vs CPU fp64；
//   case B: block swizzle 3D grid 重排（z>0 分组）输出与 2D grid 逐位一致 + 各自正确性；
//   case C: host 端 v1/v2 等价性（kColStride ∈ {8,16,32,64}，i<256, j<kColStride 全遍历）。
// 注意：NOTES_V2_ENABLE_SWIZZLE_V2 必须在 #include <hgemm.cuh> 之前定义，使 kernel 内
//       swizzle 派发走 swizzle_v2_impl（cute Swizzle<B,M,S> 位级镜像）；build_tests.sh
//       不传额外编译宏，故在文件顶部 #define。
// 约束：kernel 要求 M%128==0、N%128==0（epilogue 写回无守卫）、K%64==0（cp.async 无 K
//       边界守卫，K%BK!=0 会静默读越界污染结果）；规模 <= 512（BOOK_TEST_MAX_N）。
#define NOTES_V2_ENABLE_SWIZZLE_V2 1
#include "../../hgemm.cuh"
#include "common_test.h"
#include <algorithm>
#include <cstring>
#include <vector>

// CPU fp64 参考：TN 布局 C[M,N] = A[M,K] × B^T[N,K]（A、B^T 均 row-major）
static void hgemm_tn_ref_fp64(const half* a, const half* b_t, double* c, int M,
                              int N, int K) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k)
        acc += (double)__half2float(a[(size_t)m * K + k]) *
               (double)__half2float(b_t[(size_t)n * K + k]);
      c[(size_t)m * N + n] = acc;
    }
}

// 默认模板参数（kValTileK=4, kStages=2, BK=64）的 swizzle kernel 启动器。
// grid3d=false：2D grid (tiles_n, tiles_m)；grid3d=true：kBlockSwizzle=1 的 3D 分组
// grid (ceil(tiles_n/2), tiles_m, 2)，bx = z*gridDim.x + x 覆盖全部 N-tile。
static void launch_swizzle(half* d_a, half* d_bt, half* d_c, int M, int N,
                           int K, bool grid3d) {
  constexpr int BM = 128, BN = 128, BK = 64, KS = 2;
  size_t smem = KS * (BM * BK + BN * BK) * sizeof(half);  // 64KB，需 opt-in
  dim3 block(256);
  if (!grid3d) {
    cudaFuncSetAttribute(
        (const void*)hgemm_mma_stages_tn_swizzle<16, 8, 16, 2, 4, 4, 4, 4, KS, 0>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    hgemm_mma_stages_tn_swizzle<16, 8, 16, 2, 4, 4, 4, 4, KS, 0>
        <<<grid, block, smem>>>(d_a, d_bt, d_c, M, N, K);
  } else {
    cudaFuncSetAttribute(
        (const void*)hgemm_mma_stages_tn_swizzle<16, 8, 16, 2, 4, 4, 4, 4, KS, 1>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int tiles_n = (N + BN - 1) / BN, tiles_m = (M + BM - 1) / BM;
    dim3 grid((tiles_n + 1) / 2, tiles_m, 2);  // 分 2 组，z>0 路径被覆盖
    hgemm_mma_stages_tn_swizzle<16, 8, 16, 2, 4, 4, 4, 4, KS, 1>
        <<<grid, block, smem>>>(d_a, d_bt, d_c, M, N, K);
  }
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
}

// 单个正确性 case：2D grid 与 3D grid 两种启动各跑一遍，对照 fp64 参考 + 互相逐位一致
static void run_case(const char* tag, int M, int N, int K) {
  size_t sa = (size_t)M * K, sbt = (size_t)N * K, sc = (size_t)M * N;
  std::vector<float> fa(sa), fb(sbt);
  book_fill_rand(fa.data(), (int)sa);
  book_fill_rand(fb.data(), (int)sbt, 0xC0DE);
  std::vector<half> h_a(sa), h_bt(sbt);
  for (size_t i = 0; i < sa; ++i) h_a[i] = __float2half(fa[i]);
  for (size_t i = 0; i < sbt; ++i) h_bt[i] = __float2half(fb[i]);
  std::vector<double> ref(sc);
  hgemm_tn_ref_fp64(h_a.data(), h_bt.data(), ref.data(), M, N, K);

  half *d_a, *d_bt, *d_c;
  BOOK_CUDA_CHECK(cudaMalloc(&d_a, sa * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_bt, sbt * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_c, sc * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), sa * sizeof(half), cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_bt, h_bt.data(), sbt * sizeof(half), cudaMemcpyHostToDevice));

  printf("case %s\n", tag);
  const char* names[2] = {"hgemm swizzle reg2x (2D grid)", "hgemm swizzle reg2x (3D grid BS=1)"};
  std::vector<float> outs[2];
  for (int g = 0; g < 2; ++g) {
    BOOK_CUDA_CHECK(cudaMemset(d_c, 0xEE, sc * sizeof(half)));
    launch_swizzle(d_a, d_bt, d_c, M, N, K, g == 1);
    std::vector<half> h_c(sc);
    BOOK_CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, sc * sizeof(half), cudaMemcpyDeviceToHost));
    outs[g].resize(sc);
    for (size_t i = 0; i < sc; ++i) outs[g][i] = __half2float(h_c[i]);
    book_check(outs[g].data(), ref.data(), (int)sc, TOL_F16ACC, names[g]);
  }
  // 两种 grid 重排是同一批 tile 的重编号，逐 tile 计算路径相同，结果必须逐位一致
  double max_diff = 0.0;
  for (size_t i = 0; i < sc; ++i)
    max_diff = std::max(max_diff, std::fabs(double(outs[0][i]) - double(outs[1][i])));
  bool eq = max_diff == 0.0;
  printf("%s 2D vs 3D grid bitwise-equal: max_diff=%.1e\n", eq ? "PASS" : "FAIL", max_diff);
  if (!eq) g_failures++;
  BOOK_CUDA_CHECK(cudaFree(d_a));
  BOOK_CUDA_CHECK(cudaFree(d_bt));
  BOOK_CUDA_CHECK(cudaFree(d_c));
}

int main() {
  if (!book_require_sm(80, "ch12")) return 0;  // mma m16n8k16 + cp.async 需 sm_80+

  // Case A: M=N=K=256（最小合法形状，全部对齐约束满足）
  run_case("A: M=256 N=256 K=256 (2D vs 3D grid)", 256, 256, 256);
  // Case B: 长方形 M=384 N=512 K=256（tiles_n=4，3D 分组 grid=(2,3,2) 覆盖 z=0/1；
  //   K 固定 256：fp16 累加误差随 K 增长，K=512 实测 5.8e-2 会超 TOL_F16ACC=5e-2）
  run_case("B: M=384 N=512 K=256 (2D vs 3D grid)", 384, 512, 256);

  // Case C: host 端 v1/v2 等价性（抽取自 notes-v2.cu test_swizzle_equiv）
  //   kColStride=8 在 v2 内回退 v1，恒等；16/32/64 为 cute TMA swizzle 模式逐点验证
  printf("case C: swizzle v1/v2 host equivalence\n");
  const int strides[4] = {8, 16, 32, 64};
  long total = 0, fail = 0;
  for (int cs : strides) {
    for (int i = 0; i < 256; ++i)
      for (int j = 0; j < cs; ++j) {
        int v1, v2;
        switch (cs) {
          case 8:
            v1 = swizzle_v1_impl<8>(i, j);
            v2 = swizzle_v2_impl<8>(i, j);
            break;
          case 16:
            v1 = swizzle_v1_impl<16>(i, j);
            v2 = swizzle_v2_impl<16>(i, j);
            break;
          case 32:
            v1 = swizzle_v1_impl<32>(i, j);
            v2 = swizzle_v2_impl<32>(i, j);
            break;
          default:
            v1 = swizzle_v1_impl<64>(i, j);
            v2 = swizzle_v2_impl<64>(i, j);
            break;
        }
        ++total;
        if (v1 != v2) ++fail;
      }
  }
  printf("%s swizzle v1/v2 equiv: total=%ld fail=%ld\n",
         fail == 0 ? "PASS" : "FAIL", total, fail);
  if (fail != 0) g_failures++;

  if (g_failures == 0) printf("ALL OK\n");
  return g_failures == 0 ? 0 : 1;
}
