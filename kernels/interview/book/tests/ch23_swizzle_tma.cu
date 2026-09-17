// book/tests/ch23_swizzle_tma.cu — ch23 最小测试：CuTe Swizzle 位语义与 TMA 布局（host 端断言）
// 覆盖（每 case 打印 PASS/FAIL，末行 ALL OK）：
//   A Swizzle<3,3,3> 位公式：apply(off) == off ^ ((off & (0b111<<6)) >> 3)，[0,1024) 全遍历，
//     并核对 yyy_msk/zzz_msk 常量与「8 个 YYY 值 x 8 个 chunk」全覆盖；
//   B 对合性 + 双射 + 自定义 base layout：apply(apply(x)) == x；[0,512) 上双射；
//     composition(Swizzle<3,3,3>, (8,8):(64,8)) 的 64 点 codomain 双射且覆盖 32 个 bank；
//   C 字节空间 Swizzle<3,4,3>（= 指针路径的位置相关语义）产生「chunk XOR 行号」：
//     SW128 atom 的 8x64 域上 512 个坐标全中；固定 chunk 扫 8 行得 8 个不同物理 chunk；
//   D composition 语义：R(c) == Swizzle(Base(c))，两种 base layout 各遍历全 domain；
//   E 硬件对照：sm_120a TMA CU_TENSOR_MAP_SWIZZLE_128B 实测的 8x8 chunk 掩码表（64 槽）
//     与 Swizzle<3,4,3> 逐槽比较；由该表 + 16 B 原子性导出的 4096 点排布再逐点比较；
//     同时统计 tile_to_shape 平铺布局公式的命中数（位置无关 vs 位置相关，见 §3.4/§5 坑一）；
//   F hgemm 的 8x32 原子（hgemm.cuh L1258-1266 的同构复刻）实际 swizzle 强度：
//     chunk ^ (row>>1)，XOR 值集合 {0,8,16,24} 各 64 次（勘误 F3）；
//   G TMA 编译级：TmaDescriptor 类型与 128 B 尺寸、三个档位的掩码常量、
//     SW128 atom 的 shape/stride/cosize、make_tma_copy 在 host 侧构造出 descriptor。
// host-only：不启动任何 kernel；仅用 cudaMalloc 取一个真实 device 指针供描述符编码。
#define NOTES_V2_ENABLE_CUTE 1
#include "../../hgemm.cuh"
#include "common_test.h"
#include <cstdio>
#include <vector>

using namespace cute;
using T = cutlass::half_t;

// 带 swizzle 的 ComposedLayout 上 stride() 是被删除的（strides 与位置相关），
// 用 concept 在编译期钉住这条 API 边界。
template <class L>
concept HasConstStride = requires(L l) { cute::stride(l); };

static int g_checks = 0;
static int g_mism = 0;

// 可变参数宏：条件里含 Swizzle<3, 4, 3> 这类带逗号的模板实参时必须用 __VA_ARGS__
#define CHK(...) do { ++g_checks; if (!(__VA_ARGS__)) ++g_mism; } while (0)

static void case_end(const char* name) {
  printf("%s %s: mismatches=%d (%d checks)\n", g_mism == 0 ? "PASS" : "FAIL", name,
         g_mism, g_checks);
  if (g_mism) ++g_failures;
  g_mism = 0;
  g_checks = 0;
}

// case A：位公式逐地址核对（byte offset 空间）。Swizzle<3,3,3> 的 YYY 落在 bit[6,9)、
// 右移 3 位 XOR 到 bit[3,6)（= 16 B chunk 索引的低 3 位，注意此处是 8 B 块宽）。
static void case_a_bit_formula() {
  Swizzle<3, 3, 3> sw;
  int yyy_seen[8] = {0}, chunk_seen[8] = {0};
  for (int off = 0; off < 1024; ++off) {
    int got = (int)sw(off);
    int exp = off ^ ((off & (0b111 << 6)) >> 3);
    CHK(got == exp);
    ++yyy_seen[(off >> 6) & 7];
    ++chunk_seen[(off >> 3) & 7];
  }
  for (int i = 0; i < 8; ++i) {
    CHK(yyy_seen[i] == 128);
    CHK(chunk_seen[i] == 128);
  }
  CHK((int)Swizzle<3, 3, 3>::yyy_msk::value == 0x1C0);
  CHK((int)Swizzle<3, 3, 3>::zzz_msk::value == 0x38);
  CHK((int)Swizzle<3, 4, 3>::yyy_msk::value == 0x380);
  CHK((int)Swizzle<3, 4, 3>::zzz_msk::value == 0x70);
  case_end("case A Swizzle<3,3,3> bit formula");
}

// case B：对合性 + 双射；自定义 base layout（64 B 行、8 B 块）的 codomain 双射与 bank 覆盖。
// base(r,c) = 64r + 8c（r,c in [0,8)）→ 复合后 R(r,c) = 64r + 8(c ^ r)：块列 XOR 行号。
static void case_b_involution_bijection() {
  Swizzle<3, 3, 3> sw;
  for (int x = 0; x < 512; ++x) CHK((int)sw((int)sw(x)) == x);

  std::vector<int> hit(512, 0);
  for (int x = 0; x < 512; ++x) ++hit[(int)sw((int)sw(x))];
  for (int x = 0; x < 512; ++x) CHK(hit[x] == 1);

  auto base = make_layout(make_shape(Int<8>{}, Int<8>{}), make_stride(Int<64>{}, Int<8>{}));
  auto R = composition(Swizzle<3, 3, 3>{}, base);
  std::vector<int> codomain(512, 0);
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      int got = (int)R(r, c);
      int exp = 64 * r + 8 * (c ^ r);
      CHK(got == exp);
      if (got >= 0 && got < 512) ++codomain[got];
    }
  }
  int distinct = 0;
  for (int x = 0; x < 512; ++x) distinct += (codomain[x] == 1);
  CHK(distinct == 64);
  // 同列（固定 8 B 块列 c）扫 8 行：物理块位置两两不同，且每行该块的首 word 落在
  // 8 个互不相同的 bank —— 这正是 XOR swizzle 消除 bank conflict 的判据。
  for (int c = 0; c < 8; ++c) {
    int chunk_seen[8] = {0}, bank_seen[32] = {0}, distinct_chunks = 0, distinct_banks = 0;
    for (int r = 0; r < 8; ++r) {
      int chunk = c ^ r;
      if (!chunk_seen[chunk]) { chunk_seen[chunk] = 1; ++distinct_chunks; }
      int bank = ((64 * r + 8 * chunk) / 4) % 32;
      if (!bank_seen[bank]) { bank_seen[bank] = 1; ++distinct_banks; }
    }
    CHK(distinct_chunks == 8);
    CHK(distinct_banks == 8);
  }
  case_end("case B involution + bijection + base layout");
}

// sm_120a 实测表：TMA CUtensorMap SWIZZLE_128B 写入 64x64 fp16（128 B 行）后反读得到的
// 逐行 16 B chunk 掩码（表值 = 该物理槽位上出现的逻辑 chunk 号，行内 8 槽）。
// 来源：book 开发期 probe（tma 写入 + 反读，4096 点中 0 失配），此处固化为 8x8 常量表。
static const int kMeasuredChunkMask[8][8] = {
    {0, 1, 2, 3, 4, 5, 6, 7},
    {1, 0, 3, 2, 5, 4, 7, 6},
    {2, 3, 0, 1, 6, 7, 4, 5},
    {3, 2, 1, 0, 7, 6, 5, 4},
    {4, 5, 6, 7, 0, 1, 2, 3},
    {5, 4, 7, 6, 1, 0, 3, 2},
    {6, 7, 4, 5, 2, 3, 0, 1},
    {7, 6, 5, 4, 3, 2, 1, 0},
};

// 硬件 SWIZZLE_128B 的元素空间公式（64 half = 128 B 行，8 个 16 B chunk）
static int hw_sw128_offset(int r, int c) {
  return 64 * r + 8 * ((c / 8) ^ (r % 8)) + (c % 8);
}

// case C：字节空间 Swizzle<3,4,3> == 硬件 chunk XOR 行号（指针路径的位置相关语义）。
// 注意同一 Swizzle 作用在「元素偏移」上时是另一套结果（case E 统计的即为此差异）。
static void case_c_chunk_xor_row() {
  Swizzle<3, 4, 3> sw_bytes;
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 64; ++c) {
      int phys_bytes = (int)sw_bytes(2 * (64 * r + c));  // fp16: 元素偏移 x 2 = 字节地址
      CHK(phys_bytes / 2 == hw_sw128_offset(r, c));
    }
  }
  // 固定 chunk 扫 8 行：8 个互不相同的物理 chunk（ldmatrix 16 B 粒度无冲突的前提）
  for (int chunk = 0; chunk < 8; ++chunk) {
    int seen[8] = {0};
    for (int r = 0; r < 8; ++r) ++seen[(chunk ^ (r % 8))];
    for (int i = 0; i < 8; ++i) CHK(seen[i] == 1);
  }
  case_end("case C byte-space Sw<3,4,3> == chunk XOR row");
}

// case D：composition 的语义 R(c) == Swizzle(Base(c))（两种 base layout 全 domain 遍历）
static void case_d_composition_semantics() {
  Swizzle<3, 3, 3> sw;
  {
    auto base = make_layout(make_shape(Int<8>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    auto R = composition(sw, base);
    for (int r = 0; r < 8; ++r)
      for (int c = 0; c < 32; ++c) {
        int x = (int)base(r, c);
        CHK((int)R(r, c) == (int)sw(x));
      }
  }
  {
    auto base = make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{}));
    auto R = composition(sw, base);
    for (int r = 0; r < 16; ++r)
      for (int c = 0; c < 16; ++c) {
        int x = (int)base(r, c);
        CHK((int)R(r, c) == (int)sw(x));
      }
  }
  case_end("case D composition R(c) == Swizzle(Base(c))");
}

// case E：与硬件实测排布对照，并量化「平铺布局公式」的命中数。
// E1: 实测 8x8 掩码表 vs 字节空间 Swizzle<3,4,3>（逐槽）
// E2: 由 E1 + 16 B 原子性导出的 4096 点排布 vs Swizzle<3,4,3>（逐点）
// E3: tile_to_shape(SW128 atom, (64,64)) 的布局求值（位置无关）对同一 4096 点的命中数
static void case_e_hardware_cross_check() {
  Swizzle<3, 4, 3> sw_bytes;
  int table_mism = 0;
  for (int r = 0; r < 8; ++r)
    for (int slot = 0; slot < 8; ++slot) {
      int logical = kMeasuredChunkMask[r][slot];             // 物理槽 slot 上的逻辑 chunk
      int phys_elem = (int)sw_bytes(2 * (64 * r + 8 * logical)) / 2;
      if ((phys_elem % 64) / 8 != slot) ++table_mism;        // 该逻辑 chunk 的物理槽位
    }
  CHK(table_mism == 0);

  int point_mism = 0;
  for (int r = 0; r < 64; ++r)
    for (int c = 0; c < 64; ++c) {
      int phys_elem = (int)sw_bytes(2 * (64 * r + c)) / 2;
      if (phys_elem != hw_sw128_offset(r, c)) ++point_mism;
    }
  CHK(point_mism == 0);

  using Sw128Atom = GMMA::Layout_K_SW128_Atom<T>;
  using Sw128Tiled = decltype(tile_to_shape(Sw128Atom{}, Shape<_64, _64>{}));
  Sw128Tiled tl{};
  int tiled_match = 0;
  for (int r = 0; r < 64; ++r)
    for (int c = 0; c < 64; ++c)
      if ((int)tl(r, c) == hw_sw128_offset(r, c)) ++tiled_match;
  // 位置无关的布局求值不等价于位置相关的硬件 swizzle：命中数严格介于 0 与全额之间
  CHK(tiled_match > 0 && tiled_match < 4096);
  printf("      (E) measured table mismatches=%d\n", table_mism);
  printf("      (E) 4096-point mismatches=%d, tiled formula %d/4096\n",
         point_mism, tiled_match);
  case_end("case E hardware SWIZZLE_128B cross-check");
}

// case F：hgemm 的 8x32 原子（hgemm.cuh L1258-1266 的同构复刻）实际 swizzle 强度。
// base(8,32):(32,1) + Swizzle<3,3,3> 在元素空间求值：YYY 只需 bit[6,9)，而 8x32 原子
// 只到 bit[7]，故实际参与 XOR 的只有 row>>1 的两位 → chunk ^ (row>>1)。
static void case_f_hgemm_atom_strength() {
  auto exp_atom = composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<8>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{})));
  int k_hist[4] = {0};
  for (int r = 0; r < 8; ++r) {
    int k = (r >> 1) & 3;   // 实际参与 XOR 的行位（bit[6,9) 只覆盖到 row>>1 的两位）
    ++k_hist[k];
    for (int c = 0; c < 32; ++c) {
      int got = (int)exp_atom(r, c);
      int exp = 32 * r + 8 * ((c / 8) ^ k) + (c % 8);
      CHK(got == exp);
    }
  }
  CHK(exp_atom(0, 0) == 0);
  CHK(exp_atom(1, 0) == 32);  // 行 0/1 的 XOR 值为 0（无 swizzle）
  // XOR 值集合 = {0,8,16,24}，每个值覆盖 2 行（2 行 x 32 列 = 64 个坐标）
  for (int k = 0; k < 4; ++k) CHK(k_hist[k] == 2);
  case_end("case F hgemm 8x32 atom effective swizzle");
}

// case G：TMA 编译级核对（不启动 kernel）。对照 hgemm.cuh L1863-1885 的约束：
// descriptor 必须与 CTA tile、smem 布局三者自洽；TMA 只接受 M=4 的 swizzle 模式。
static void case_g_tma_compile_level() {
  static_assert(sizeof(TmaDescriptor) == 128, "TMA descriptor must be 128 bytes");
  CHK(sizeof(TmaDescriptor) == 128);
  CHK(std::is_same_v<TmaDescriptor, CUtensorMap>);
  CHK((int)Swizzle<1, 4, 3>::num_bits == 1);
  CHK((int)Swizzle<2, 4, 3>::num_bits == 2);
  CHK((int)Swizzle<3, 4, 3>::num_base == 4);  // TMA 模式要求 M=4（16 B 粒度）
  CHK((int)Swizzle<3, 3, 3>::num_base == 3);  // M=3 不是 TMA 受支持的模式

  using Sw128Atom = GMMA::Layout_K_SW128_Atom<T>;
  CHK(shape(Sw128Atom{}) == Shape<_8, _64>{});
  CHK((int)cosize(Sw128Atom{}) == 512);
  // ComposedLayout 的三段式：Swizzle + offset 槽（smem 指针标志）+ 基础布局
  static_assert(std::is_same_v<decltype(Sw128Atom{}.layout_a()), Swizzle<3, 4, 3>>,
                "SW128 atom must carry Swizzle<3,4,3> (M=4, the TMA-supported mode)");
  CHK(shape(Sw128Atom{}.layout_b()) == Shape<_8, _64>{});
  CHK(stride(Sw128Atom{}.layout_b()) == Stride<_64, _1>{});
  static_assert(!HasConstStride<Sw128Atom>,
                "stride() of a swizzled ComposedLayout is deleted: strides are position dependent");

  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
    printf("SKIP(case G): no CUDA device for descriptor encode\n");
    case_end("case G TMA compile-level");
    return;
  }
  constexpr int kM = 64, kN = 64;
  T* d_src = nullptr;
  BOOK_CUDA_CHECK(cudaMalloc(&d_src, kM * kN * sizeof(T)));
  auto gA = make_tensor(make_gmem_ptr(reinterpret_cast<cutlass::half_t*>(d_src)),
                        make_shape(Int<kM>{}, Int<kN>{}),
                        make_stride(Int<kN>{}, Int<1>{}));
  using SmemLayout = decltype(tile_to_shape(Sw128Atom{}, make_shape(Int<kM>{}, Int<kN>{})));
  auto tma = make_tma_copy(SM90_TMA_LOAD{}, gA, SmemLayout{}, Shape<_64, _64>{}, _1{});
  CHK(tma.get_tma_descriptor() != nullptr);
  auto gA_coord = tma.get_tma_tensor(make_shape(Int<kM>{}, Int<kN>{}));
  CHK(rank(gA_coord) == 2);
  CHK((int)size<0>(gA_coord) == kM);
  CHK((int)size<1>(gA_coord) == kN);
  CHK((int)cosize(SmemLayout{}) == kM * kN);
  BOOK_CUDA_CHECK(cudaFree(d_src));
  case_end("case G TMA compile-level (descriptor)");
}

int main() {
  printf("== ch23 CuTe swizzle / TMA (host-side assertions) ==\n");
  case_a_bit_formula();
  case_b_involution_bijection();
  case_c_chunk_xor_row();
  case_d_composition_semantics();
  case_e_hardware_cross_check();
  case_f_hgemm_atom_strength();
  case_g_tma_compile_level();
  if (g_failures == 0) {
    printf("ALL OK\n");
    return 0;
  }
  printf("FAILURES: %d\n", g_failures);
  return 1;
}
