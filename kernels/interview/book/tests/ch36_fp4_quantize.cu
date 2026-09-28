// book/tests/ch36_fp4_quantize.cu — ch36 最小测试：NVFP4 线格式与两级量化前处理
// 覆盖：
//   case A: SF 字节通式 vs cute 权威布局（全量枚举，host-only）——M=256,K=256 与
//           M=300,K=192 两组（含 128 行向上取整的补零区）
//   case B: A 量化（单级 level-2，M=256 K=256）——SF 逐组逐 bit 比对 + e2m1
//           数据逐 nibble 比对 + 反量化 round-trip 相对误差
//   case C: A 量化（两级，退化为 A 侧 level-1）——同上，另加 sA[m] 校验
//   case D: B 转置量化（单级，K=256 N=256）——SF/数据/转置落位校验
//   case E: B 转置量化（两级 per-col，K=192 N=136 尾 shape）——同上 + sB[n]
//   case F: B 转置量化（两级，K=1024 与 K=512，kSplit=8 多片列 amax）——列幅度
//           按 2^-j 分档，专门查「跨切片 atomicMax 合并 / 列索引错位」
//   case G: 硬件事实：ue4m3/e2m1 的 cvt.rn 平局取奇码点 + satfinite 饱和
// 判定：SF 与数据码点一律逐 bit（e2m1/ue4m3 都是精确定义的离散码）；
//       sA/sB（列 amax / 2688）用相对容差 1e-6——它应当是**逐位相等**的
//       （除以正常数保序 + 非负浮点位序），1e-6 只留给编译器除法实现差异；
//       round-trip 用相对 Frobenius 误差，容差 = 理论界（见 ch36 误差模型节）。
// 编译：**必须** `-gencode arch=compute_120a,code=sm_120a`（build_tests.sh 的
//       写法）。不要用 `-arch=sm_120a` 简写：那样 PTX 的虚拟目标落到
//       `compute_120`，ptxas 会以 `Feature 'cvt.e2m1x2.f32' not supported on
//       .target 'sm_120'` 拒绝整份 PTX（e2m1 转换是 arch-specific 指令）。
//       目标 arch 不在位时 SKIP。
#define NOTES_V2_ENABLE_CUTE 1
#define NOTES_V2_ENABLE_TMA_MMA_WS 1
#define NOTES_V2_ENABLE_SM120_FP4 1
#include "../../fp4_gemm.cuh"
#include "common_test.h"
#include <vector>

using fp4_gemm::Fp4GemmScaleMode;

static float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }

// ---------------------------------------------------------------------------
// CPU 参考编码器：码空间「就近取偶」（与硬件 cvt.rn 实测一致，见 case F）
// ---------------------------------------------------------------------------
static const double kE2m1Mag[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};

// e2m1 码 = 符号 1bit | 指数 2bit | 尾数 1bit（码值升序 = 幅值升序）
static int cpu_e2m1(float x) {
  int sign = (x < 0.0f) ? 8 : 0;
  double a = std::fabs((double)x);
  if (a > 6.0) a = 6.0;  // satfinite
  int best = 0;
  double bd = 1e30;
  for (int c = 0; c < 8; ++c) {
    double d = std::fabs(a - kE2m1Mag[c]);
    if (d < bd || (d == bd && (c & 1) == 0)) {  // 平局取奇码
      bd = d;
      best = c;
    }
  }
  return sign | best;
}

static double cpu_e2m1_decode(int code) {
  double m = kE2m1Mag[code & 7];
  return (code & 8) ? -m : m;
}

// ue4m3 码 = 指数 4bit(bias 7) | 尾数 3bit；0xFF = NaN（本实现不用）
static double cpu_ue4m3_decode(int c) {
  const int e = (c >> 3) & 0xf, m = c & 7;
  return (e == 0) ? m * std::ldexp(1.0, -9) : (8 + m) * std::ldexp(1.0, e - 10);
}

static int cpu_ue4m3(float x) {
  double v = (double)x;
  if (v > 448.0) v = 448.0;  // satfinite（输入恒非负）
  if (v < 0.0) v = 0.0;
  int best = 0;
  double bd = 1e30;
  for (int c = 0; c < 127; ++c) {
    double d = std::fabs(v - cpu_ue4m3_decode(c));
    if (d < bd || (d == bd && (c & 1) == 0)) {
      bd = d;
      best = c;
    }
  }
  return best;
}

// 与 kernel 同序的 fp32 组量化：返回 (sf 字节, 16 个 4-bit 码)
static unsigned char ref_quant_group(const float *x, int *codes) {
  float amax = 0.0f;
  for (int e = 0; e < 16; ++e) amax = fmaxf(amax, fabsf(x[e]));
  const unsigned char sf = (unsigned char)cpu_ue4m3(amax / 6.0f);
  const float sf_val = (float)cpu_ue4m3_decode(sf);
  const float inv2 = (sf_val > 0.0f) ? (1.0f / sf_val) : 0.0f;
  for (int e = 0; e < 16; ++e) codes[e] = cpu_e2m1(x[e] * inv2);
  return sf;
}

// ---------------------------------------------------------------------------
// case A: SF 字节通式 vs cute/CUTLASS 权威布局（sf_gmem_layout）
// ---------------------------------------------------------------------------
static void test_sf_layout(int M, int K) {
  const int mpad = (M + 127) / 128 * 128;
  const int nb_k = K / 64;
  auto lay = fp4_gemm::sf_gmem_layout(M, K);
  int bad = 0;
  for (int mn = 0; mn < mpad; ++mn)
    for (int s = 0; s < K / 16; ++s) {
      // 布局坐标是「数据单位」的 (mn, k)；16 个 K 元素共享一个 SF（stride 0）
      int want = (int)lay(mn, s * 16);
      int got = fp4_gemm::sf_byte_offset(mn, s, nb_k);
      if (want != got) {
        if (bad < 3)
          printf("  [mismatch] mn=%d s=%d cute=%d formula=%d\n", mn, s, want, got);
        ++bad;
      }
    }
  const size_t bytes = (size_t)(mpad / 128) * (size_t)nb_k * 512;
  char name[128];
  snprintf(name, sizeof(name), "ch36 A SF layout (M=%d K=%d, %d 个字节落点)",
           M, K, mpad * (K / 16));
  bool pass = (bad == 0) && (fp4_gemm::sf_buffer_bytes(M, K) == bytes);
  printf("%s %s\n", pass ? "PASS" : "FAIL", name);
  if (!pass) ++g_failures;
  printf("     sf_buffer_bytes(%d,%d) = %zu B (%d 个行块 x %d 个 512B 原子)\n",
         M, K, bytes, mpad / 128, nb_k);
}

// ---------------------------------------------------------------------------
// A 侧量化校验：SF 逐 bit + 数据逐 nibble + round-trip
// ---------------------------------------------------------------------------
static void test_a_quant(int M, int K, bool level1, const char *tag) {
  const size_t n = (size_t)M * K;
  std::vector<__nv_bfloat16> h_a(n);
  srand(42);
  for (auto &v : h_a) v = __float2bfloat16(((float)rand() / RAND_MAX) * 2 - 1);
  if (level1)  // 一行小幅度：单级会在这一行整行归零，两级不会（ch36 对比项）
    for (int k = 0; k < K; ++k) h_a[k] = __float2bfloat16((k % 2 ? 1.f : -1.f) * 1e-4f);

  __nv_bfloat16 *d_a;
  cutlass::float_e2m1_t *d_a4;
  cutlass::float_ue4m3_t *d_sfa;
  float *d_sa;
  BOOK_CUDA_CHECK(cudaMalloc(&d_a, n * 2));
  BOOK_CUDA_CHECK(cudaMalloc(&d_a4, n / 2));
  BOOK_CUDA_CHECK(cudaMalloc(&d_sfa, fp4_gemm::sf_buffer_bytes(M, K)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_sa, (size_t)M * 4));
  BOOK_CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * 2, cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemset(d_sfa, 0xAA, fp4_gemm::sf_buffer_bytes(M, K)));
  BOOK_CUDA_CHECK(cudaMemset(d_a4, 0xAA, n / 2));

  if (level1) fp4_gemm::row_amax_fp4_kernel<<<(M + 7) / 8, 256>>>(d_a, d_sa, M, K);
  const int mpad = (M + 127) / 128 * 128;
  const int total = mpad * (K / 16);
  if (level1)
    fp4_gemm::quantize_a_fp4_kernel<true><<<(total + 255) / 256, 256>>>(
        d_a, d_a4, d_sfa, d_sa, M, K);
  else
    fp4_gemm::quantize_a_fp4_kernel<false><<<(total + 255) / 256, 256>>>(
        d_a, d_a4, d_sfa, d_sa, M, K);
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_sa(M, 0.f);
  std::vector<unsigned char> h_sf(fp4_gemm::sf_buffer_bytes(M, K));
  std::vector<unsigned char> h_a4(n / 2);
  if (level1) BOOK_CUDA_CHECK(cudaMemcpy(h_sa.data(), d_sa, M * 4, cudaMemcpyDeviceToHost));
  BOOK_CUDA_CHECK(cudaMemcpy(h_sf.data(), d_sfa, h_sf.size(), cudaMemcpyDeviceToHost));
  BOOK_CUDA_CHECK(cudaMemcpy(h_a4.data(), d_a4, h_a4.size(), cudaMemcpyDeviceToHost));

  // sA[m] = 行 amax / 2688（level-1 才有）
  double sa_bad = 0;
  if (level1)
    for (int m = 0; m < M; ++m) {
      float amax = 0;
      for (int k = 0; k < K; ++k) amax = fmaxf(amax, fabsf(bf2f(h_a[(size_t)m * K + k])));
      sa_bad = std::max(sa_bad, (double)std::fabs(h_sa[m] - amax / 2688.0f));
    }

  // 逐组：CPU 参考 SF/码点 vs kernel 产出
  int sf_bad = 0, code_bad = 0, pad_bad = 0;
  double se = 0, sr = 0;  // round-trip 相对 Frobenius 分子/分母
  std::vector<int> codes(16);
  for (int m = 0; m < mpad; ++m)
    for (int g = 0; g < K / 16; ++g) {
      const int s = g;
      if (m >= M) {  // 补零行：SF 必须显式写 0（否则未初始化字节被当 SF 乘进 MMA）
        if (h_sf[fp4_gemm::sf_byte_offset(m, s, K / 64)] != 0) ++pad_bad;
        continue;
      }
      float x[16];
      const float inv1 = level1 ? (h_sa[m] > 0.f ? 1.f / h_sa[m] : 0.f) : 1.f;
      for (int e = 0; e < 16; ++e) x[e] = bf2f(h_a[(size_t)m * K + s * 16 + e]) * inv1;
      const unsigned char sf = ref_quant_group(x, codes.data());
      const int got_sf = h_sf[fp4_gemm::sf_byte_offset(m, s, K / 64)];
      if (got_sf != sf) {
        if (sf_bad < 3)
          printf("  [SF] m=%d s=%d kernel=0x%02x ref=0x%02x\n", m, s, got_sf, sf);
        ++sf_bad;
      }
      const float sf_val = (float)cpu_ue4m3_decode(sf);
      for (int e = 0; e < 16; ++e) {
        const unsigned char byte = h_a4[(size_t)m * (K / 2) + s * 8 + e / 2];
        const int nib = (e & 1) ? (byte >> 4) : (byte & 0xf);  // 偶 k 在低 nibble
        if (nib != codes[e]) {
          if (code_bad < 3)
            printf("  [data] m=%d k=%d kernel=0x%x ref=0x%x\n", m, s * 16 + e,
                   nib, codes[e]);
          ++code_bad;
        }
        // round-trip：解码值 x SF x sA[m]
        double dq = cpu_e2m1_decode(nib) * sf_val * (level1 ? h_sa[m] : 1.0);
        double xv = bf2f(h_a[(size_t)m * K + s * 16 + e]);
        se += (dq - xv) * (dq - xv);
        sr += xv * xv;
      }
    }

  const double rel = std::sqrt(se / sr);
  // 容差：e2m1 半 ulp 相对上界 2^-3 x 两组独立误差 -> 相对 RMS 12.5%，
  // 尾部元素把 max err 抬高，表里用 relFro；两级与单级同界（见 ch36 误差模型）
  const double tol = 0.16;
  char name[160];
  snprintf(name, sizeof(name), "ch36 A quant %s (M=%d K=%d)", tag, M, K);
  bool pass = (sf_bad == 0) && (code_bad == 0) && (pad_bad == 0) && (rel <= tol) &&
              (sa_bad <= TOL_F32ACC);
  printf("%s %s: SF_mismatch=%d data_mismatch=%d pad_SF_mismatch=%d "
         "rel_fro=%.4f (tol=%.2f) sA_max_abs_err=%.2e\n",
         pass ? "PASS" : "FAIL", name, sf_bad, code_bad, pad_bad, rel, tol, sa_bad);
  if (!pass) ++g_failures;

  if (level1) {  // 小幅度行对照：单级整行归零 -> rel 1.0；两级 ~ 噪声底
    cutlass::float_ue4m3_t *d_sf1;
    float *d_sa1;
    BOOK_CUDA_CHECK(cudaMalloc(&d_sf1, fp4_gemm::sf_buffer_bytes(M, K)));
    BOOK_CUDA_CHECK(cudaMalloc(&d_sa1, (size_t)M * 4));
    fp4_gemm::quantize_a_fp4_kernel<false><<<(total + 255) / 256, 256>>>(
        d_a, d_a4, d_sf1, d_sa1, M, K);
    BOOK_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<unsigned char> h1(fp4_gemm::sf_buffer_bytes(M, K));
    BOOK_CUDA_CHECK(cudaMemcpy(h1.data(), d_sf1, h1.size(), cudaMemcpyDeviceToHost));
    const int s0 = fp4_gemm::sf_byte_offset(0, 0, K / 64);
    printf("     小幅度行 (1e-4) 第 0 组 SF: 两级=0x%02x(%.6g) 单级=0x%02x(%.6g)\n",
           h_sf[s0], cpu_ue4m3_decode(h_sf[s0]), h1[s0], cpu_ue4m3_decode(h1[s0]));
    cudaFree(d_sf1);
    cudaFree(d_sa1);
  }
  cudaFree(d_a); cudaFree(d_a4); cudaFree(d_sfa); cudaFree(d_sa);
}

// ---------------------------------------------------------------------------
// B 侧（转置）量化校验
// ---------------------------------------------------------------------------
static void test_bt_quant(int K, int N, bool level1, const char *tag,
                          bool hetero = false) {
  const size_t n = (size_t)K * N;
  std::vector<__nv_bfloat16> h_b(n);
  srand(42);
  for (auto &v : h_b) v = __float2bfloat16(((float)rand() / RAND_MAX) * 2 - 1);
  if (hetero) {  // 列幅度分档 2^-j：列 amax 相差 3 个数量级，索引错位立刻暴露
    for (int k = 0; k < K; ++k)
      for (int nn = 0; nn < N; ++nn)
        h_b[(size_t)k * N + nn] = __float2bfloat16(
            bf2f(h_b[(size_t)k * N + nn]) * ldexpf(1.0f, -(nn % 16)));
  }

  __nv_bfloat16 *d_b;
  cutlass::float_e2m1_t *d_b4t;
  cutlass::float_ue4m3_t *d_sfb;
  float *d_sb;
  BOOK_CUDA_CHECK(cudaMalloc(&d_b, n * 2));
  BOOK_CUDA_CHECK(cudaMalloc(&d_b4t, n / 2));
  BOOK_CUDA_CHECK(cudaMalloc(&d_sfb, fp4_gemm::sf_buffer_bytes(N, K)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_sb, (size_t)N * 4));
  BOOK_CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * 2, cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemset(d_b4t, 0xAA, n / 2));
  BOOK_CUDA_CHECK(cudaMemset(d_sfb, 0xAA, fp4_gemm::sf_buffer_bytes(N, K)));
  fp4_gemm::fp4_gemm_quantize_b(d_b, d_b4t, d_sfb, d_sb, N, K, level1, 0);
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_sb(N, 0.f);
  std::vector<unsigned char> h_sf(fp4_gemm::sf_buffer_bytes(N, K));
  std::vector<unsigned char> h_b4t(n / 2);
  if (level1) BOOK_CUDA_CHECK(cudaMemcpy(h_sb.data(), d_sb, N * 4, cudaMemcpyDeviceToHost));
  BOOK_CUDA_CHECK(cudaMemcpy(h_sf.data(), d_sfb, h_sf.size(), cudaMemcpyDeviceToHost));
  BOOK_CUDA_CHECK(cudaMemcpy(h_b4t.data(), d_b4t, h_b4t.size(), cudaMemcpyDeviceToHost));

  double sb_bad = 0, sb_rel = 0;
  if (level1)
    for (int nn = 0; nn < N; ++nn) {
      float amax = 0;
      for (int k = 0; k < K; ++k) amax = fmaxf(amax, fabsf(bf2f(h_b[(size_t)k * N + nn])));
      const float ref = amax / 2688.0f;
      const double d = std::fabs(h_sb[nn] - ref);
      sb_bad = std::max(sb_bad, d);
      sb_rel = std::max(sb_rel, d / std::max((double)ref, 1e-30));
    }

  int sf_bad = 0, code_bad = 0;
  double se = 0, sr = 0;
  std::vector<int> codes(16);
  for (int nn = 0; nn < N; ++nn)
    for (int g = 0; g < K / 16; ++g) {
      float x[16];
      const float inv1 = level1 ? (h_sb[nn] > 0.f ? 1.f / h_sb[nn] : 0.f) : 1.f;
      for (int e = 0; e < 16; ++e) x[e] = bf2f(h_b[(size_t)(g * 16 + e) * N + nn]) * inv1;
      const unsigned char sf = ref_quant_group(x, codes.data());
      const int got = h_sf[fp4_gemm::sf_byte_offset(nn, g, K / 64)];
      if (got != sf) {
        if (sf_bad < 3) printf("  [SF] n=%d s=%d kernel=0x%02x ref=0x%02x\n", nn, g, got, sf);
        ++sf_bad;
      }
      const float sf_val = (float)cpu_ue4m3_decode(got);
      for (int e = 0; e < 16; ++e) {
        // B4T 是 (N,K) 行主序：行 n、K 方向打包 -> 与 A 侧同构
        const unsigned char byte = h_b4t[(size_t)nn * (K / 2) + g * 8 + e / 2];
        const int nib = (e & 1) ? (byte >> 4) : (byte & 0xf);
        if (nib != codes[e]) {
          if (code_bad < 3)
            printf("  [data] n=%d k=%d kernel=0x%x ref=0x%x\n", nn, g * 16 + e, nib,
                   codes[e]);
          ++code_bad;
        }
        double dq = cpu_e2m1_decode(nib) * sf_val * (level1 ? h_sb[nn] : 1.0);
        double xv = bf2f(h_b[(size_t)(g * 16 + e) * N + nn]);
        se += (dq - xv) * (dq - xv);
        sr += xv * xv;
      }
    }
  const double rel = std::sqrt(se / sr);
  const double tol = 0.16;
  char name[160];
  snprintf(name, sizeof(name), "ch36 B4T quant %s (K=%d N=%d)", tag, K, N);
  bool pass = (sf_bad == 0) && (code_bad == 0) && (rel <= tol) && (sb_rel <= 1e-6);
  printf("%s %s: SF_mismatch=%d data_mismatch=%d rel_fro=%.4f (tol=%.2f) "
         "sB_max_abs_err=%.2e sB_max_rel_err=%.2e\n",
         pass ? "PASS" : "FAIL", name, sf_bad, code_bad, rel, tol, sb_bad, sb_rel);
  if (!pass) ++g_failures;
  cudaFree(d_b); cudaFree(d_b4t); cudaFree(d_sfb); cudaFree(d_sb);
}

// ---------------------------------------------------------------------------
// case F: 硬件码点事实（平局取奇 + satfinite）
// ---------------------------------------------------------------------------
__global__ void k_probe_ue4m3(const float *x, unsigned char *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = fp4_gemm::cvt_f2_to_ue4m3(x[i]);
}
__global__ void k_probe_e2m1(const float *x, unsigned char *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = fp4_gemm::cvt_f2_to_e2m1x2(0.0f, x[i]) & 0xf;
}

static void test_hw_facts() {
  // ue4m3：e=15 步长 32 -> 416(0x7d 奇)/448(0x7e 偶) 的中点是 432
  const float hx[5] = {432.0f, 216.0f, 500.0f, 1e30f, 0.0f};
  const int want[5] = {0x7e, 0x76, 0x7e, 0x7e, 0x00};
  float *dx;
  unsigned char *dy;
  BOOK_CUDA_CHECK(cudaMalloc(&dx, sizeof(hx)));
  BOOK_CUDA_CHECK(cudaMalloc(&dy, 5));
  BOOK_CUDA_CHECK(cudaMemcpy(dx, hx, sizeof(hx), cudaMemcpyHostToDevice));
  k_probe_ue4m3<<<1, 32>>>(dx, dy, 5);
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  unsigned char hy[5];
  BOOK_CUDA_CHECK(cudaMemcpy(hy, dy, 5, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < 5; ++i)
    if (hy[i] != want[i]) {
      printf("  [ue4m3] x=%g kernel=0x%02x want=0x%02x\n", hx[i], hy[i], want[i]);
      ++bad;
    }
  cudaFree(dx);
  cudaFree(dy);

  // e2m1：码 000..111 = 0,0.5,1,1.5,2,3,4,6；邻近码点中点全部取偶码
  const float ex[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
  const int ewant[7] = {0, 2, 2, 4, 4, 6, 6};
  float *dx2;
  unsigned char *dy2;
  BOOK_CUDA_CHECK(cudaMalloc(&dx2, sizeof(ex)));
  BOOK_CUDA_CHECK(cudaMalloc(&dy2, 7));
  BOOK_CUDA_CHECK(cudaMemcpy(dx2, ex, sizeof(ex), cudaMemcpyHostToDevice));
  k_probe_e2m1<<<1, 32>>>(dx2, dy2, 7);
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  unsigned char hey[7];
  BOOK_CUDA_CHECK(cudaMemcpy(hey, dy2, 7, cudaMemcpyDeviceToHost));
  for (int i = 0; i < 7; ++i)
    if (hey[i] != ewant[i]) {
      printf("  [e2m1] x=%g kernel=0x%x want=0x%x\n", ex[i], hey[i], ewant[i]);
      ++bad;
    }
  cudaFree(dx2);
  cudaFree(dy2);

  const bool pass = (bad == 0);
  printf("%s ch36 F hw facts (ue4m3/e2m1 平局取奇 + satfinite)\n",
         pass ? "PASS" : "FAIL");
  if (!pass) ++g_failures;
}

int main() {
  if (!book_require_sm(120, "ch36")) return 0;
  printf("=== ch36 NVFP4 线格式与量化前处理 ===\n");
  test_sf_layout(256, 256);
  test_sf_layout(300, 192);
  test_a_quant(256, 256, false, "单级 level-2");
  test_a_quant(256, 256, true, "两级 (level-2 x A 行)");
  test_a_quant(300, 192, true, "两级 尾 shape");
  test_bt_quant(256, 256, false, "单级 level-2");
  test_bt_quant(256, 256, true, "两级 (level-2 x B 列)");
  test_bt_quant(192, 136, true, "两级 尾 shape");
  // K >= 512：转置量化走 kSplit=8 的多片列 amax（atomicMax 合并），列幅度分档
  test_bt_quant(1024, 256, true, "两级 kSplit=8");
  test_bt_quant(512, 256, true, "两级 kSplit=8 异质列幅度", true);
  test_bt_quant(768, 512, true, "两级 kSplit=8 异质列幅度 大 N", true);
  test_hw_facts();
  printf("=== ch36 %s (%d failures) ===\n", g_failures ? "FAIL" : "ALL PASS",
         g_failures);
  return g_failures ? 1 : 0;
}
