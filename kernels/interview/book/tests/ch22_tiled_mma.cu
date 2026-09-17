// book/tests/ch22_tiled_mma.cu — ch22 最小测试：CuTe TiledMMA 与 fragment（host 端 partition 断言）
// 覆盖（每 case 打印 PASS/FAIL，末行 ALL OK）：
//   A 原子 fragment 公式：SM80_16x8x16 的 A/B/C layout 与手推公式逐 lane 逐值核对；
//   B FFPA traits：QK/PV tile_size、线程数、与手写 make_tiled_mma 的类型一致性、
//     SmemLayout cosize、copy atom 类型；
//   C partition_fragment 形状：QK 在 (64,64) 上 (8,1,4)/(4,8,4)/(4,1,8)，PV 对应形状；
//   D 守恒与重复因子：A/C 覆盖为双射，B 恰好 4 倍重复（4 个 M-warp）；
//   E warp 行带与跨 warp 边界：warp w 的 C fragment 行带 = [16w, 16w+16)；
//   F PV 的 acc_O 寄存器账目：OFragType 每线程 32 值 x 8 chunk = 256 寄存器 + C 覆盖。
// host-only：不启动 kernel；包含 ffpa_attn.cuh 以核对真实 traits（与源码同源）。
#define NOTES_V2_ENABLE_CUTE 1
#include "../../hgemm.cuh"
#include "../../flash_attn.cuh"
#include "../../ffpa_attn.cuh"
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

using Traits = fa_cute::FFPAAttnSplitDCuTeTraits<512>;
using TraitsQK = typename Traits::TiledMmaQK;
using TraitsPV = typename Traits::TiledMmaPV;
using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;

// case A：原子 fragment 公式。原子 codomain 是扁平坐标（A: m+16k, B: n+8k, C: m+16n），
// 手推（ch22 式 22-atom-a/b/c，lane = thr, 值 v 按 colex 拆位）：
//   A: m = lane/4 + 8*d1, k = 2*(lane%4) + d0 + 8*d2   (v = d0 + 2*d1 + 4*d2)
//   B: n = lane/4,        k = 2*(lane%4) + d0 + 8*d1   (v = d0 + 2*d1)
//   C: m = lane/4 + 8*d1, n = 2*(lane%4) + d0          (v = d0 + 2*d1)
static void case_a_atom_layouts() {
  MmaAtom atom;
  CHK((int)size(typename MmaAtom::ThrID{}) == 32);
  CHK((int)get<0>(typename MmaAtom::Shape_MNK{}) == 16);
  CHK((int)get<1>(typename MmaAtom::Shape_MNK{}) == 8);
  CHK((int)get<2>(typename MmaAtom::Shape_MNK{}) == 16);
  auto la = typename MmaAtom::LayoutA_TV{};
  auto lb = typename MmaAtom::LayoutB_TV{};
  auto lc = typename MmaAtom::LayoutC_TV{};
  CHK((int)size<1>(la) == 8);
  CHK((int)size<1>(lb) == 4);
  CHK((int)size<1>(lc) == 4);
  for (int lane = 0; lane < 32; ++lane) {
    for (int v = 0; v < 8; ++v) {
      int d0 = v % 2, d1 = (v / 2) % 2, d2 = v / 4;
      int m = lane / 4 + 8 * d1;
      int k = 2 * (lane % 4) + d0 + 8 * d2;
      CHK((int)la(lane, v) == m + 16 * k);
    }
    for (int v = 0; v < 4; ++v) {
      int d0 = v % 2, d1 = v / 2;
      int nb = lane / 4;
      int kb = 2 * (lane % 4) + d0 + 8 * d1;
      CHK((int)lb(lane, v) == nb + 8 * kb);
      int mc = lane / 4 + 8 * d1;
      int nc = 2 * (lane % 4) + d0;
      CHK((int)lc(lane, v) == mc + 16 * nc);
    }
  }
  case_end("case A MMA atom fragment layouts (m16n8k16)");
}

// case B：FFPA traits 的 tile_size/线程数/类型一致性。手推：QK = Tile<64,64,16>，
// PV = Tile<64,16,16>，二者共用 AtomLayoutMNK = (4,1,1) -> 4 x 32 = 128 线程。
static void case_b_traits() {
  static_assert(tile_size<0>(TraitsQK{}) == 64 && tile_size<1>(TraitsQK{}) == 64 &&
                tile_size<2>(TraitsQK{}) == 16, "QK tile 64x64x16");
  static_assert(tile_size<0>(TraitsPV{}) == 64 && tile_size<1>(TraitsPV{}) == 16 &&
                tile_size<2>(TraitsPV{}) == 16, "PV tile 64x16x16");
  CHK((int)size(TraitsQK{}) == 128);
  CHK((int)size(TraitsPV{}) == 128);
  CHK((int)tile_size<0>(TraitsQK{}) == 64 && (int)tile_size<1>(TraitsQK{}) == 64 &&
      (int)tile_size<2>(TraitsQK{}) == 16);
  CHK((int)tile_size<0>(TraitsPV{}) == 64 && (int)tile_size<1>(TraitsPV{}) == 16 &&
      (int)tile_size<2>(TraitsPV{}) == 16);
  CHK((int)cosize(typename Traits::SmemLayoutQ{}) == 64 * 64);
  CHK((int)cosize(typename Traits::SmemLayoutKV{}) == 64 * 64);
  CHK((int)cosize(typename Traits::SmemLayoutVt{}) == 64 * 64);
  // 与手写构造的类型一致性（同一 traits 类型 = 同一编译期对象）
  using HandQK = decltype(make_tiled_mma(MmaAtom{}, Layout<Shape<_4, _1, _1>>{},
                                         Tile<Int<64>, Int<64>, _16>{}));
  using HandPV = decltype(make_tiled_mma(MmaAtom{}, Layout<Shape<_4, _1, _1>>{},
                                         Tile<Int<64>, _16, _16>{}));
  constexpr bool qk_is_hand = std::is_same<TraitsQK, HandQK>::value;
  constexpr bool pv_is_hand = std::is_same<TraitsPV, HandPV>::value;
  constexpr bool atom_is_ldsm =
      std::is_same<typename Traits::SmemCopyAtom, Copy_Atom<SM75_U32x4_LDSM_N, half_t>>::value;
  constexpr bool atom_is_ldsm_t =
      std::is_same<typename Traits::SmemCopyAtomTransposed,
                   Copy_Atom<SM75_U16x8_LDSM_T, half_t>>::value;
  CHK(qk_is_hand);
  CHK(pv_is_hand);
  CHK(atom_is_ldsm);
  CHK(atom_is_ldsm_t);
  // thr_layout_vmnk 的手推核对：(ThrV,ThrM,ThrN,ThrK) = (32,4,1,1) 且线程号 = v + 32*m
  auto tl = TraitsQK{}.get_thr_layout_vmnk();
  CHK((int)size<0>(tl) == 32 && (int)size<1>(tl) == 4);
  CHK((int)size<2>(tl) == 1 && (int)size<3>(tl) == 1);
  CHK((int)stride<1>(tl) == 32);
  case_end("case B FFPA traits tile_size & thread count");
}

// case C：partition_fragment 的形状。手推（thrfrg 四步推导链，warp 排布 (4,1,1)）：
//   QK on (64,64): A(8,1,4)=32, B(4,8,4)=128, C(4,1,8)=32
//   PV on (64,64): A(8,1,4)=32；on (16,16): B(4,2,1)=8；C(4,1,8)=32
static void case_c_fragment_shapes() {
  // fragment 的 partition 需要"真实形状"的张量（CuTe 的 make_fragment_like 无法对恒等
  // 张量的 ScaledBasis 步长排序），故用 gmem 视图占位；形状与坐标无关。
  auto gQK = make_tensor(make_gmem_ptr((cutlass::half_t const *)nullptr),
                         make_layout(make_shape(Int<64>{}, Int<64>{}), GenRowMajor{}));
  auto gV16 = make_tensor(make_gmem_ptr((cutlass::half_t const *)nullptr),
                          make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{}));
  auto thrQK = TraitsQK{}.get_slice(0);
  auto fA = thrQK.partition_fragment_A(gQK);
  auto fB = thrQK.partition_fragment_B(gQK);
  auto fC = thrQK.partition_fragment_C(gQK);
  CHK((int)size<0>(fA) == 8 && (int)size<1>(fA) == 1 && (int)size<2>(fA) == 4);
  CHK((int)size(fA) == 32);
  CHK((int)size<0>(fB) == 4 && (int)size<1>(fB) == 8 && (int)size<2>(fB) == 4);
  CHK((int)size(fB) == 128);
  CHK((int)size<0>(fC) == 4 && (int)size<1>(fC) == 1 && (int)size<2>(fC) == 8);
  CHK((int)size(fC) == 32);
  auto thrPV = TraitsPV{}.get_slice(0);
  auto pA = thrPV.partition_fragment_A(gQK);
  auto pB = thrPV.partition_fragment_B(gV16);
  auto pC = thrPV.partition_fragment_C(gQK);
  CHK((int)size(pA) == 32);
  CHK((int)size<0>(pA) == 8 && (int)size<1>(pA) == 1 && (int)size<2>(pA) == 4);
  CHK((int)size(pB) == 8);
  CHK((int)size<0>(pB) == 4 && (int)size<1>(pB) == 2 && (int)size<2>(pB) == 1);
  CHK((int)size(pC) == 32);
  CHK((int)size<0>(pC) == 4 && (int)size<1>(pC) == 1 && (int)size<2>(pC) == 8);
  case_end("case C partition_fragment shapes QK/PV");
}

// case D：守恒与重复因子。C 覆盖 (64,64) 双射（128 x 32 = 4096）；A 同为双射；
//   B 被 4 个 M-warp 各持一份 → 每元素恰好 4 次（128 x 128 = 16384 = 4 x 4096）。
static void case_d_conservation() {
  TraitsQK mma;
  auto idQK = make_identity_tensor(make_shape(Int<64>{}, Int<64>{}));
  std::vector<int> covA(64 * 64, 0), covB(64 * 64, 0), covC(64 * 64, 0);
  int totA = 0, totB = 0, totC = 0;
  for (int t = 0; t < 128; ++t) {
    auto thr = mma.get_slice(t);
    auto a = thr.partition_A(idQK);
    for (int i = 0; i < (int)size(a); ++i) {
      auto c = a(i);
      int m = (int)get<0>(c), k = (int)get<1>(c);
      if (m >= 0 && m < 64 && k >= 0 && k < 64) { ++covA[m * 64 + k]; ++totA; }
    }
    auto b = thr.partition_B(idQK);
    for (int i = 0; i < (int)size(b); ++i) {
      auto c = b(i);
      int n = (int)get<0>(c), k = (int)get<1>(c);
      if (n >= 0 && n < 64 && k >= 0 && k < 64) { ++covB[n * 64 + k]; ++totB; }
    }
    auto cc = thr.partition_C(idQK);
    for (int i = 0; i < (int)size(cc); ++i) {
      auto c = cc(i);
      int m = (int)get<0>(c), n = (int)get<1>(c);
      if (m >= 0 && m < 64 && n >= 0 && n < 64) { ++covC[m * 64 + n]; ++totC; }
    }
  }
  CHK(totA == 128 * 32 && totC == 128 * 32 && totB == 128 * 128);
  int badA = 0, badB = 0, badC = 0;
  for (int i = 0; i < 64 * 64; ++i) {
    if (covA[i] != 1) ++badA;
    if (covB[i] != 4) ++badB;
    if (covC[i] != 1) ++badC;
  }
  CHK(badA == 0);
  CHK(badB == 0);
  CHK(badC == 0);
  case_end("case D conservation & B duplication factor");
}

// case E：warp 行带与跨 warp 边界。原子 16 行 + ThrM = 4 → warp w 覆盖行 [16w, 16w+16)；
//   四带互不重叠、并集为 [0,64)。同时核对每线程首值坐标 (m,n) = (16w + lane/4, 2*(lane%4))。
static void case_e_warp_bands() {
  TraitsQK mma;
  auto idQK = make_identity_tensor(make_shape(Int<64>{}, Int<64>{}));
  std::vector<int> band_min(4, 1000), band_max(4, -1);
  for (int t = 0; t < 128; ++t) {
    int w = t / 32, lane = t % 32;
    auto c = mma.get_slice(t).partition_C(idQK);
    int m0 = (int)get<0>(c(0));
    int n0 = (int)get<1>(c(0));
    CHK(m0 == 16 * w + lane / 4);
    CHK(n0 == 2 * (lane % 4));
    for (int i = 0; i < (int)size(c); ++i) {
      auto cc = c(i);
      int m = (int)get<0>(cc);
      CHK(m >= 16 * w && m < 16 * w + 16);
      if (m < band_min[w]) band_min[w] = m;
      if (m > band_max[w]) band_max[w] = m;
    }
  }
  for (int w = 0; w < 4; ++w) {
    CHK(band_min[w] == 16 * w);
    CHK(band_max[w] == 16 * w + 15);
  }
  case_end("case E warp row bands & cross-warp boundary");
}

// case F：PV 的 acc_O 账目与覆盖。OFragType = partition_fragment_C(pv, Shape<_64,_64>)
//   每线程 32 个 float；D=512 时 kDChunks = 8 → 256 个寄存器/线程（源码 L152-158、
//   L171 声明的 o_acc_storage 尺寸）；PV 在 (64,16) 上的 C 覆盖为双射。
// 设备端编译期核对：与 ffpa_attn.cuh L152-158 完全相同的 decltype 计算链。
// convert_layout_acc_rowcol 是 __device__ 函数（CUTE_DEVICE = __device__），
// host 代码无法引用，故在 __global__ 函数体里做静态断言（不会启动）。
__global__ void ch22_rowcol_device_static_checks() {
  using OFragType = decltype(partition_fragment_C(TraitsPV{}, Shape<Int<64>, Int<64>>{}));
  using OFragLayout = typename OFragType::layout_type;
  constexpr int kOElems = decltype(size(OFragType{}))::value;
  constexpr int kORows = decltype(size<0>(make_tensor(
      (float *)nullptr, fa_cute::convert_layout_acc_rowcol(OFragLayout{}))))::value;
  constexpr int kOCols = decltype(size<1>(make_tensor(
      (float *)nullptr, fa_cute::convert_layout_acc_rowcol(OFragLayout{}))))::value;
  static_assert(kOElems == 32, "O fragment = 32 floats/thread");
  static_assert(kORows == 2 && kOCols == 16, "rowcol view = 2 rows x 16 cols");
}

static void case_f_pv_register_budget() {
  using OFragType = decltype(partition_fragment_C(TraitsPV{}, Shape<Int<64>, Int<64>>{}));
  using OFragLayout = typename OFragType::layout_type;
  constexpr int kOElems = decltype(size(OFragType{}))::value;
  constexpr int kDChunks = 512 / 64;
  CHK(kOElems == 32);
  CHK((int)size<0>(OFragType{}) == 4);
  CHK(kOElems * kDChunks == 256);
  // host 端复刻 convert_layout_acc_rowcol（__device__ 函数无法在 host 调用）：
  // (MMA=4, MMA_M=1, MMA_N=8) -divide(_2)-> ((2,1),(2,8)) = 2x16 行列视图。
  auto divided = logical_divide(OFragLayout{}, Shape<_2>{});
  auto rc = make_layout(make_layout(get<0, 1>(divided), get<1>(divided)),
                        make_layout(get<0, 0>(divided), get<2>(divided)));
  CHK((int)size<0>(rc) == 2);
  CHK((int)size<1>(rc) == 16);
  CHK((int)size<0>(rc) * (int)size<1>(rc) == kOElems);
  auto idCV = make_identity_tensor(make_shape(Int<64>{}, Int<16>{}));
  std::vector<int> cov(64 * 16, 0);
  int tot = 0;
  for (int t = 0; t < 128; ++t) {
    auto c = TraitsPV{}.get_slice(t).partition_C(idCV);
    CHK((int)size(c) == 8);
    for (int i = 0; i < (int)size(c); ++i) {
      auto cc = c(i);
      int m = (int)get<0>(cc), n = (int)get<1>(cc);
      if (m >= 0 && m < 64 && n >= 0 && n < 16) { ++cov[m * 16 + n]; ++tot; }
    }
  }
  CHK(tot == 128 * 8);
  int bad = 0;
  for (int i = 0; i < 64 * 16; ++i)
    if (cov[i] != 1) ++bad;
  CHK(bad == 0);
  case_end("case F PV acc_O register budget & coverage");
}

int main() {
  printf("ch22 CuTe TiledMMA & fragments (host-only, no kernel launch)\n");
  case_a_atom_layouts();
  case_b_traits();
  case_c_fragment_shapes();
  case_d_conservation();
  case_e_warp_bands();
  case_f_pv_register_budget();
  if (g_failures == 0) printf("ALL OK\n");
  else printf("FAILURES: %d case(s)\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
