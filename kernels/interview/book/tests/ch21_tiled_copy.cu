// book/tests/ch21_tiled_copy.cu — ch21 最小测试：CuTe Tensor 与 TiledCopy（host 端 TV-layout 断言）
// 覆盖（每 case 打印 PASS/FAIL，末行 ALL OK）：
//   A make_tiled_copy 的 TV 分解：线程数/值数/tiler + (thr,val)->(m,n) 手推公式（G2S 配置）；
//   B partition_S 的每线程形状与坐标公式 + 全 tile 覆盖双射（(128,32) 上 4096 点）；
//   C TiledMMA 的 fragment 形状 (MMA,MMA_M,MMA_K) 与 retile_D 的零拷贝语义；
//   D ldmatrix 拷贝的 source 划分公式（每线程 64 元素，8x4x2 结构）；
//   E A 拷贝的重复因子：N 方向 warp 使每个元素被覆盖 2 次（与 ch21 式 21-dup 一致）；
//   F upcast/downcast 语义 + R2S/S2G 的 tiler 与单 pipe 覆盖（(32,32) 双射）。
// host-only：不启动 kernel；只做类型/布局断言。包含 hgemm.cuh 以核对真实类型的可编译性。
#define NOTES_V2_ENABLE_CUTE 1
#include "../../hgemm.cuh"
#include "common_test.h"
#include <cstdio>
#include <vector>

using namespace cute;
using T = cutlass::half_t;

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

// hgemm 的 G2S 拷贝：ThrLayout (32,4):(4,1) x ValLayout (1,8)，128 线程 x 8 half
using g2s_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
using G2S = decltype(make_tiled_copy(
    Copy_Atom<Copy_Traits<g2s_op>, T>{},
    make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
    make_layout(make_shape(Int<1>{}, Int<8>{}))));

// 与 hgemm.cuh launch wrapper 相同构造的 TiledMMA（EU-repeat 2x2x1, Tile<32,32,16>）
using TiledMmaH = decltype(make_tiled_mma(
    MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
    make_layout(make_shape(Int<2>{}, Int<2>{}, Int<1>{})),
    Tile<Int<32>, Int<32>, Int<16>>{}));

// case A：TiledCopy 的 thread/value 分解与 TV 公式。手推（ch21 式 21-g2s-tv）：
//   m = t/4, n = 8*(t%4) + v；get_layoutS_TV() 的 codomain 是 (32,32) tiler 的扁平
//   坐标 m + 32*n（M 最快），故 tvS(t,v) == (t/4) + 32*(8*(t%4)+v)。
static void case_a_tv_decomposition() {
  static_assert(size<0>(G2S::TiledLayout_TV{}) == 128, "G2S has 128 threads");
  static_assert(size<1>(G2S::TiledLayout_TV{}) == 8, "G2S has 8 values per thread");
  static_assert(size<0>(G2S::Tiler_MN{}) == 32 && size<1>(G2S::Tiler_MN{}) == 32,
                "G2S tiler is 32x32");
  CHK((int)size<0>(G2S::TiledLayout_TV{}) == 128);
  CHK((int)size<1>(G2S::TiledLayout_TV{}) == 8);
  CHK((int)size<0>(G2S::Tiler_MN{}) == 32 && (int)size<1>(G2S::Tiler_MN{}) == 32);
  auto tvS = G2S::get_layoutS_TV();
  for (int t = 0; t < 128; ++t)
    for (int v = 0; v < 8; ++v) {
      int got = (int)tvS(t, v);
      int want = (t / 4) + 32 * (8 * (t % 4) + v);
      CHK(got == want);
    }
  case_end("case A make_tiled_copy TV decomposition");
}

// case B：partition_S 的每线程形状 (8,4,1) 与坐标公式 + 全 tile 双射。
// 手推：线程 t 的值 v 落在 (m,n) = (t/4 + 32*rM, 8*(t%4) + v)，rM 为 M 方向 tile 重复。
static void case_b_partition_bijection() {
  auto idA = make_identity_tensor(make_shape(Int<128>{}, Int<32>{}));
  G2S g2s;
  std::vector<int> seen(128 * 32, 0);
  for (int t = 0; t < 128; ++t) {
    auto sl = g2s.get_slice(t).partition_S(idA);
    CHK((int)size(sl) == 32);
    CHK((int)size<0>(sl) == 8 && (int)size<1>(sl) == 4 && (int)size<2>(sl) == 1);
    for (int v = 0; v < 8; ++v)
      for (int rM = 0; rM < 4; ++rM) {
        auto c = sl(v, rM, 0);
        int m = (int)get<0>(c);
        int n = (int)get<1>(c);
        CHK(m == t / 4 + 32 * rM);
        CHK(n == 8 * (t % 4) + v);
        if (m >= 0 && m < 128 && n >= 0 && n < 32) ++seen[m * 32 + n];
      }
  }
  for (int i = 0; i < 128 * 32; ++i) CHK(seen[i] == 1);
  case_end("case B partition_S shape & bijection");
}

// case C：TiledMMA 的 fragment 形状与 retile_D 零拷贝。
// 手推：A 在 (128,32) 上为 (MMA=8, MMA_M=4, MMA_K=2)；C 在 (128,256) 上为
//       (MMA=4, MMA_M=4, MMA_N=256/8/2=16)。retile_D 只重标索引、数据指针不变。
static void case_c_fragment_retile() {
  CHK((int)size(TiledMmaH{}) == 128);
  CHK((int)tile_size<0>(TiledMmaH{}) == 32);
  CHK((int)tile_size<1>(TiledMmaH{}) == 32);
  CHK((int)tile_size<2>(TiledMmaH{}) == 16);
  // 注意：fragment 的 partition 需要"真实形状"的张量做前缀布局（CuTe 的
  // make_fragment_like 无法对恒等张量的 ScaledBasis 步长排序），故用 gmem 视图占位。
  auto gA = make_tensor(make_gmem_ptr((cutlass::half_t const *)nullptr),
                        make_layout(make_shape(Int<128>{}, Int<32>{}), GenRowMajor{}));
  auto gD = make_tensor(make_gmem_ptr((float const *)nullptr),
                        make_layout(make_shape(Int<128>{}, Int<256>{}), GenRowMajor{}));
  auto thr_mma = TiledMmaH{}.get_slice(0);
  auto tCrA = thr_mma.partition_fragment_A(gA);
  CHK((int)size(tCrA) == 64);
  CHK((int)size<0>(tCrA) == 8 && (int)size<1>(tCrA) == 4 && (int)size<2>(tCrA) == 2);
  auto tCrD = thr_mma.partition_fragment_C(gD);
  CHK((int)size(tCrD) == 256);
  CHK((int)size<0>(tCrD) == 4 && (int)size<1>(tCrD) == 4 && (int)size<2>(tCrD) == 16);
  // retile_D：与 S->R 拷贝的 D 侧对齐，零拷贝（data() 不变、形状不变）
  auto s2r_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, T>{}, TiledMmaH{});
  auto tCrA_view = s2r_a.get_slice(0).retile_D(tCrA);
  CHK((int)size(tCrA_view) == 64);
  CHK(tCrA_view.data() == tCrA.data());
  CHK((int)size<0>(tCrA_view) == 8);
  case_end("case C fragment shape & retile_D zero-copy");
}

// case D：ldmatrix 拷贝的 source 划分。实测 get_layoutS_TV =
//   ((16,2,2,2),(8,1)):((1,256,16,0),(32,0))，即线程 t 的 (m,k) 基址
//   m = t%16 + 16*((t/32)%2)，k = 8*((t/16)%2)，再叠加重复模式
//   (rM: m += 32, rK: k += 16) 与值模式 (v: k += 1)。注意第 4 个线程模式
//   步长为 0：线程 t 与 t+64 读同一份 A 片段（= N 方向 warp 的重复，case E）。
static void case_d_ldmatrix_source_formula() {
  auto s2r_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, T>{}, TiledMmaH{});
  static_assert(size<0>(decltype(s2r_a)::Tiler_MN{}) == 32, "s2r tiler M = tile_size<0>");
  auto idA = make_identity_tensor(make_shape(Int<128>{}, Int<32>{}));
  for (int t = 0; t < 128; ++t) {
    auto sl = s2r_a.get_slice(t).partition_S(idA);
    CHK((int)size(sl) == 64);
    CHK((int)size<0>(sl) == 8 && (int)size<1>(sl) == 4 && (int)size<2>(sl) == 2);
    for (int v = 0; v < 8; ++v)
      for (int rM = 0; rM < 4; ++rM)
        for (int rK = 0; rK < 2; ++rK) {
          auto c = sl(v, rM, rK);
          CHK((int)get<0>(c) == (t % 16) + 16 * ((t / 32) % 2) + 32 * rM);
          CHK((int)get<1>(c) == 8 * ((t % 32) / 16) + v + 16 * rK);
        }
  }
  case_end("case D ldmatrix source partitioning formula");
}

// case E：重复因子。A 片段被 N 方向的 2 个 warp 各读一遍 → 每个元素恰好被覆盖 2 次
//   （128 线程 x 64 元素 = 8192 = 2 x 4096）；这也是 ch21 式 21-dup 的实测依据。
static void case_e_duplication() {
  auto s2r_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, T>{}, TiledMmaH{});
  auto idA = make_identity_tensor(make_shape(Int<128>{}, Int<32>{}));
  std::vector<int> cnt(128 * 32, 0);
  int total = 0;
  for (int t = 0; t < 128; ++t) {
    auto sl = s2r_a.get_slice(t).partition_S(idA);
    for (int i = 0; i < (int)size(sl); ++i) {
      auto c = sl(i);
      int m = (int)get<0>(c);
      int n = (int)get<1>(c);
      if (m >= 0 && m < 128 && n >= 0 && n < 32) { ++cnt[m * 32 + n]; ++total; }
    }
  }
  CHK(total == 128 * 64);
  int dup = 0;
  for (int i = 0; i < 128 * 32; ++i) {
    if (cnt[i] != 2) ++dup;
  }
  CHK(dup == 0);
  CHK((int)(128 * 64) == 2 * (128 * 32));
  case_end("case E A-copy duplication factor");
}

// case F：upcast/downcast 语义与 R2S/S2G 的 tiler。
//   upcast_2((8,32):(32,1)) = (8,16):(16,1)（连续 mode 的 extent 除以 2、其余 stride 除以 2）；
//   upcast 对 Swizzle 复合分配：upcast_2(Sw<3,3,3> o base)(m,nh) == (Sw<3,3,3> o base)(m,2*nh)/2。
static void case_f_upcast_and_c_copies() {
  auto b = make_layout(make_shape(Int<8>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
  auto ub = upcast<2>(b);
  CHK((int)size<0>(ub) == 8 && (int)size<1>(ub) == 16);
  CHK((int)stride<0>(ub) == 16 && (int)stride<1>(ub) == 1);
  auto c31 = make_layout(make_shape(Int<4>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
  auto dc = downcast<2>(c31);
  CHK((int)size<1>(dc) == 64 && (int)stride<0>(dc) == 64);
  CHK((int)size(upcast<8>(Layout<_32, _1>{})) == 4);
  auto base = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
  auto sw = composition(Swizzle<3, 3, 3>{}, base);
  auto usw = upcast<2>(sw);
  for (int m = 0; m < 32; ++m)
    for (int nh = 0; nh < 16; ++nh)
      CHK((int)usw(m, nh) == (int)sw(m, 2 * nh) / 2);
  // R2S：make_tiled_copy_C 的 tiler 由 TiledMMA 推导（tile_size<0/1> = (32,32)）
  auto r2s_c = make_tiled_copy_C(Copy_Atom<UniversalCopy<int>, T>{}, TiledMmaH{});
  static_assert(size<0>(decltype(r2s_c)::Tiler_MN{}) == 32, "r2s tiler M");
  static_assert(size<1>(decltype(r2s_c)::Tiler_MN{}) == 32, "r2s tiler N");
  auto idC = make_identity_tensor(make_shape(Int<32>{}, Int<32>{}, Int<4>{}));
  std::vector<int> seen(32 * 32, 0);
  for (int t = 0; t < 128; ++t) {
    auto sl = r2s_c.get_slice(t).partition_D(idC);
    CHK((int)size(sl) == 32);
    CHK((int)size<0>(sl) == 8 && (int)size<3>(sl) == 4);
    for (int v = 0; v < 8; ++v) {
      auto c = sl(v, 0, 0, 0);
      int m = (int)get<0>(c);
      int n = (int)get<1>(c);
      if (m >= 0 && m < 32 && n >= 0 && n < 32) ++seen[m * 32 + n];
    }
  }
  for (int i = 0; i < 32 * 32; ++i) CHK(seen[i] == 1);
  // S2G：显式 Thr/Val 的 tiler 也是 (32,32)，partition_S(sC) 与 G2S 同一套公式
  auto s2g_c = make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint128_t>, T>{},
                               make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{})));
  static_assert(size<0>(decltype(s2g_c)::Tiler_MN{}) == 32, "s2g tiler M");
  auto sl0 = s2g_c.get_slice(0).partition_S(idC);
  CHK((int)size(sl0) == 32);
  CHK((int)size<0>(sl0) == 8 && (int)size<3>(sl0) == 4);
  for (int v = 0; v < 8; ++v) {
    auto c = sl0(v, 0, 0, 0);
    CHK((int)get<0>(c) == 0);
    CHK((int)get<1>(c) == v);
  }
  case_end("case F upcast semantics & R2S/S2G tilers");
}

int main() {
  printf("ch21 CuTe Tensor & TiledCopy (host-only, no kernel launch)\n");
  case_a_tv_decomposition();
  case_b_partition_bijection();
  case_c_fragment_retile();
  case_d_ldmatrix_source_formula();
  case_e_duplication();
  case_f_upcast_and_c_copies();
  if (g_failures == 0) printf("ALL OK\n");
  else printf("FAILURES: %d case(s)\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
