// book/tests/ch20_layout.cu — ch20 最小测试：CuTe Layout 基础与代数（host 端坐标断言）
// 覆盖（每 case 打印 PASS/FAIL，末行 ALL OK）：
//   A colex 线性化 idx = sum s_i*c_i：coord 映射 + linear-index 展开（含带孔洞 layout）；
//   B composition：R(c) == A(B(c)) 全 domain 遍历（含 stride 折叠与恒等段）；
//   C right_inverse：L(R(i)) == i、composition(L,R) 为恒等布局、带孔洞 layout 退化 _1:_0；
//   D left_inverse：准逆 L(LI(L(c))) == L(c)；双射时与 right_inverse 逐点相等；
//   E product：make_layout(A,B) 并列语义逐坐标；
//   F zipped_divide：rank/size/compatible 结构 + 手工坐标公式 + codomain 双射覆盖；
//   G tile_to_shape：(8,BK) atom -> (BM,BK,kStage) 的原子重复与 stage 步进（hgemm 用法）；
//   H Swizzle 复合：Swizzle<3,3,3> o base 的逐点位公式核对 + codomain 双射。
// host-only：不启动 kernel、不需要 GPU；build_tests.sh 直接编译运行（main 不调用 CUDA API）。
#include <cute/tensor.hpp>
#include "common_test.h"
#include <cstdio>
#include <vector>

using namespace cute;

// 手工 colex 展开（2D）：x -> (x % s0, x / s0) -> c0*d0 + c1*d1
static int colex2(int x, int s0, int s1, int d0, int d1) {
  return (x % s0) * d0 + ((x / s0) % s1) * d1;
}

// case A：idx = sum_i s_i*c_i，coord 形式与 linear-index 形式一致（colex 序）
static void case_a_colex() {
  int mism = 0;
  auto L = make_layout(make_shape(Int<4>{}, Int<8>{}), make_stride(Int<1>{}, Int<4>{}));
  for (int c0 = 0; c0 < 4; ++c0)
    for (int c1 = 0; c1 < 8; ++c1)
      if ((int)L(c0, c1) != c0 * 1 + c1 * 4) ++mism;
  for (int x = 0; x < 32; ++x)
    if ((int)L(x) != colex2(x, 4, 8, 1, 4)) ++mism;
  // 带孔洞 layout（cosize 64 > size 32）：stride 2/16 的喷射状映射
  auto G = make_layout(make_shape(Int<8>{}, Int<4>{}), make_stride(Int<2>{}, Int<16>{}));
  for (int c0 = 0; c0 < 8; ++c0)
    for (int c1 = 0; c1 < 4; ++c1)
      if ((int)G(c0, c1) != 2 * c0 + 16 * c1) ++mism;
  for (int x = 0; x < 32; ++x)
    if ((int)G(x) != colex2(x, 8, 4, 2, 16)) ++mism;
  printf("%s case A colex idx=sum(s_i*c_i): mismatches=%d (128 checks)\n",
         mism == 0 ? "PASS" : "FAIL", mism);
  if (mism) ++g_failures;
}

// case B：composition 的定义式 R(c) == A(B(c))；R 的 stride 由 A 在 B 像上的斜率决定
static void case_b_compose() {
  int mism = 0;
  {
    auto A = make_layout(make_shape(Int<6>{}, Int<2>{}), make_stride(Int<9>{}, Int<2>{}));
    auto B = make_layout(Int<4>{});
    auto R = composition(A, B);
    for (int c = 0; c < 4; ++c)
      if ((int)R(c) != (int)A(B(c))) ++mism;
    if ((int)stride<0>(R) != 9) ++mism;
    // A(0..3) = 0,9,18,27（B 的像落在 A 的 mode-0 上）→ R = (4):(9)
    for (int c = 0; c < 4; ++c)
      if ((int)R(c) != 9 * c) ++mism;
  }
  {
    auto A = make_layout(make_shape(Int<4>{}, Int<8>{}), make_stride(Int<1>{}, Int<4>{}));
    auto B = make_layout(make_shape(Int<2>{}, Int<2>{}), make_stride(Int<2>{}, Int<1>{}));
    auto R = composition(A, B);
    for (int c = 0; c < 4; ++c)
      if ((int)R(c) != (int)A(B(c))) ++mism;
    // A 覆盖 [0,4) 且恒等 → R 与 B 逐点相同
    for (int c = 0; c < 4; ++c)
      if ((int)R(c) != (int)B(c)) ++mism;
  }
  printf("%s case B composition R(c)==A(B(c)): mismatches=%d (14 checks)\n",
         mism == 0 ? "PASS" : "FAIL", mism);
  if (mism) ++g_failures;
}

// case C：right_inverse 满足 L(R(i)) == i；composition(L,R) == make_layout(shape(R))
static void case_c_right_inverse() {
  int mism = 0;
  {
    auto L = make_layout(make_shape(Int<4>{}, Int<8>{}), make_stride(Int<1>{}, Int<4>{}));
    auto R = right_inverse(L);
    if ((int)size(R) != 32) ++mism;
    for (int i = 0; i < 32; ++i)
      if ((int)L(R(i)) != i) ++mism;
    auto id = make_layout(shape(R));
    auto LR = composition(L, R);
    for (int i = 0; i < 32; ++i)
      if ((int)LR(i) != (int)id(i)) ++mism;
  }
  {
    // 值域含空洞（stride 2）：不存在长度 >= 2 的右逆 → R 退化为 (_1,_0)
    auto G = make_layout(make_shape(Int<8>{}, Int<4>{}), make_stride(Int<2>{}, Int<16>{}));
    auto R = right_inverse(G);
    if ((int)size(R) != 1) ++mism;
    if ((int)G(R(0)) != 0) ++mism;
  }
  printf("%s case C right_inverse L(R(i))==i: mismatches=%d (65 checks)\n",
         mism == 0 ? "PASS" : "FAIL", mism);
  if (mism) ++g_failures;
}

// case D：left_inverse 是准逆（injective 时为真左逆）：L(LI(L(c))) == L(c)
static void case_d_left_inverse() {
  int mism = 0;
  {
    auto G = make_layout(make_shape(Int<8>{}, Int<4>{}), make_stride(Int<2>{}, Int<16>{}));
    auto LI = left_inverse(G);
    for (int c = 0; c < 32; ++c) {
      int o = (int)G(c);
      if ((int)G(LI(o)) != o) ++mism;
    }
  }
  {
    // 双射（colex compact）：左逆与右逆逐点相同
    auto L = make_layout(make_shape(Int<4>{}, Int<8>{}), make_stride(Int<1>{}, Int<4>{}));
    auto LI = left_inverse(L);
    auto R = right_inverse(L);
    for (int i = 0; i < 32; ++i)
      if ((int)LI(i) != (int)R(i)) ++mism;
  }
  printf("%s case D left_inverse quasi-inverse: mismatches=%d (64 checks)\n",
         mism == 0 ? "PASS" : "FAIL", mism);
  if (mism) ++g_failures;
}

// case E：make_layout(A,B) 并列（concatenation）：P(a,b) == A(a) + B(b)
static void case_e_product() {
  int mism = 0;
  auto A = make_layout(make_shape(Int<2>{}, Int<3>{}), make_stride(Int<1>{}, Int<2>{}));
  auto B = make_layout(make_shape(Int<4>{}), make_stride(Int<8>{}));
  auto P = make_layout(A, B);   // shape = ((2,3),(4))：A 的整棵 shape 成为一个 mode
  if ((int)rank(P) != 2) ++mism;
  for (int a0 = 0; a0 < 2; ++a0)
    for (int a1 = 0; a1 < 3; ++a1)
      for (int b = 0; b < 4; ++b)
        if ((int)P(make_coord(a0, a1), b) != (int)A(a0, a1) + (int)B(b)) ++mism;
  printf("%s case E product make_layout(A,B): mismatches=%d (24 checks)\n",
         mism == 0 ? "PASS" : "FAIL", mism);
  if (mism) ++g_failures;
}

// case F：zipped_divide 把 L 的 codomain 切成 size(L)/size(T) 个 tile
//   手工公式（L=(8,8):(1,8) col-major, T=(4,4):(1,8) 同族 4x4 tile）：
//     zd(t0,t1,r0,r1) = (t0 + 4*r0) + 8*(t1 + 4*r1) = t0 + 8*t1 + 4*r0 + 32*r1
static void case_f_divide() {
  int mism = 0;
  auto L = make_layout(make_shape(Int<8>{}, Int<8>{}), make_stride(Int<1>{}, Int<8>{}));
  auto T = make_layout(make_shape(Int<4>{}, Int<4>{}), make_stride(Int<1>{}, Int<8>{}));
  auto zd = zipped_divide(L, T);
  if ((int)rank(zd) != 2) ++mism;
  if ((int)size<0>(zd) != 16 || (int)size<1>(zd) != 4) ++mism;
  if (!compatible(T, layout<0>(zd))) ++mism;
  // rest mode 结构：4 个 tile 的基址 = {0,4,32,36}
  auto rest = layout<1>(zd);
  for (int r = 0; r < 4; ++r)
    if ((int)rest(r) != 4 * (r % 2) + 32 * (r / 2)) ++mism;
  // 坐标公式 + codomain 双射（64 个输出恰好覆盖 [0,64) 一次）
  std::vector<int> seen(64, 0);
  for (int r0 = 0; r0 < 2; ++r0)
    for (int r1 = 0; r1 < 2; ++r1)
      for (int t0 = 0; t0 < 4; ++t0)
        for (int t1 = 0; t1 < 4; ++t1) {
          int t = t0 + 4 * t1;
          int r = r0 + 2 * r1;
          int got = (int)zd(t, r);
          int want = t0 + 8 * t1 + 4 * r0 + 32 * r1;
          if (got != want) ++mism;
          if (got >= 0 && got < 64) ++seen[got]; else ++mism;
        }
  for (int v = 0; v < 64; ++v)
    if (seen[v] != 1) ++mism;
  printf("%s case F zipped_divide tile semantics: mismatches=%d (76 checks)\n",
         mism == 0 ? "PASS" : "FAIL", mism);
  if (mism) ++g_failures;
}

// case G：tile_to_shape 把 (8,BK) atom 重复到 (BM,BK,kStage)（hgemm SmemLayoutA 用法）
static void case_g_tile_to_shape() {
  int mism = 0;
  auto atom = make_layout(make_shape(Int<8>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
  auto tiled = tile_to_shape(atom, make_shape(Int<128>{}, Int<32>{}, Int<2>{}));
  // 零成本抽象：static layout 的 shape/cosize 全部在编译期折叠（static_assert 可证）
  static_assert(cosize(decltype(atom){}) == 256, "atom cosize must fold at compile time");
  static_assert(cosize(decltype(tiled){}) == 8192, "tiled cosize must fold at compile time");
  if ((int)size(tiled) != 8192 || (int)cosize(tiled) != 8192) ++mism;
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 32; ++j)
      if ((int)tiled(i, j, 0) != i * 32 + j) ++mism;
  if ((int)tiled(8, 0, 0) != 256) ++mism;   // atom 沿 M 重复：步进 8*BK = 256
  if ((int)tiled(0, 0, 1) != 4096) ++mism;  // stage 步进 = BM*BK = 4096
  printf("%s case G tile_to_shape atom->(128,32,2): mismatches=%d (261 checks)\n",
         mism == 0 ? "PASS" : "FAIL", mism);
  if (mism) ++g_failures;
}

// case H：composition(Swizzle<3,3,3>, base) 是 codomain 双射重排，逐点满足
//   offset' = offset ^ ((offset >> 6) & 7) << 3   （ch12 手写 XOR swizzle 的位级镜像）
static void case_h_swizzle_compose() {
  int mism = 0;
  auto base = make_layout(make_shape(Int<8>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
  auto sw = composition(Swizzle<3, 3, 3>{}, base);
  if ((int)size(sw) != 256 || (int)cosize(sw) != 256) ++mism;
  std::vector<int> seen(256, 0);
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 32; ++j) {
      int off = (int)base(i, j);
      int got = (int)sw(i, j);
      if (got != (off ^ (((off >> 6) & 7) << 3))) ++mism;
      if (got >= 0 && got < 256) ++seen[got]; else ++mism;
    }
  for (int v = 0; v < 256; ++v)
    if (seen[v] != 1) ++mism;
  printf("%s case H Swizzle<3,3,3> o layout bit formula: mismatches=%d (258 checks)\n",
         mism == 0 ? "PASS" : "FAIL", mism);
  if (mism) ++g_failures;
}

int main() {
  printf("ch20 CuTe layout basics & algebra (host-only, no kernel launch)\n");
  case_a_colex();
  case_b_compose();
  case_c_right_inverse();
  case_d_left_inverse();
  case_e_product();
  case_f_divide();
  case_g_tile_to_shape();
  case_h_swizzle_compose();
  if (g_failures == 0) printf("ALL OK\n");
  else printf("FAILURES: %d case(s)\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
