#pragma once
#include "common.cuh"
// fp4_gemm.cuh: Phase 10 NVFP4 GEMM — e2m1 在线量化 + SM120 blockscale mma 教学案例
// =============================================================================
// 全链路：BF16 输入 -> NVFP4 量化前处理(per-16 level-2 [+ per-row/col level-1])
//         -> CuTe NVFP4 GEMM (SM120_16x8x64 blockscaled mma.sync + TMA + mbarrier
//            多级流水，scale 由 MMA 硬件消费) -> 在线反量化(level-1 折回) -> BF16 输出.
//         封装为 fp4_gemm_bf16 一个 C++ API, 与 cuBLAS BF16 GEMM 对比精度与性能.
//
// 与 fp8_gemm.cuh(Phase 9) 的三处关键差异：
//   1. 数据 4-bit（e2m1，16 个码点），K 维两个元素共享一个字节；TMA 用
//      CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B 直接搬 4-bit 数据（无需拆包）.
//   2. MMA 是 mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X
//      .m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 — 每 16 个 K 元素一个 ue4m3
//      缩放因子(SF)，缩放由 MMA 硬件在乘加中消费，不进寄存器算术.
//   3. 因此 epilogue 只剩 level-1（per-row/per-col）查表乘回；level-2 的
//      per-16 scale 完全"免费"，这是 NVFP4 相对 FP8 的精度红利所在.
//
// 参考：ffpa-attn csrc/cuffpa/cute/fp4(sm_120 persist-D，commit 861d75e)
//   - MMA atom: SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<e2m1,e2m1,f32,ue4m3,16>
//   - SF 寄存器片段: 每线程 1x u32(4 个 ue4m3) —— 由 atom 的 SFALayout/SFBLayout
//     决定，须用 CUTLASS collective 的 6 个 helper 复刻取片段方式(见 P10.3)
//   - smem swizzle: UMMA::Layout_K_SW32_Atom<uint4_t>(8 行 x 64 个 4-bit = 512B)
//   - scale 折叠代数：C = (A4 · B4^T) · sA · sB，level-1 在 epilogue 逐行/列查表
//   - mbarrier: ClusterTransactionBarrier(full) + ClusterBarrier(empty)
//
// TMA 布局（sf_byte_offset 与 CUTLASS tile_atom_to_shape_SFA 逐字节一致，见 36.4）：
//   SF gmem 原子 = 128 行 x 4 个 SF = 512B（行内 4 个 SF 连续，同一行块内
//   128 行 x 64 K 元素恰好铺满 512B 无空洞），行块内各 K 组按 +512B 递进.
//
// 约束（教学约定，由 API 层检查）：
//   K % 64 == 0 (SF 以 64 个 K 元素为一整块计量 + A4/B4T 行 16B 对齐)
//   N % 8  == 0 (C 行 N*2B 16B 对齐，TMA store 硬性要求)
//   M 任意      (A4 SF 缓冲按 128 行向上取整清零填充；C 尾行由 TMA store 裁剪)
//
// 线格式（标准 bench 表）：| Kernel | Max Err | TFLOPS/cuBLAS |
// 性能参考(PRO 5000 sm_120a, 4096^3): 见 notes-v2.cu --bench --mnk 4096,4096,4096

#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS) && \
    defined(NOTES_V2_ENABLE_SM120_FP4)
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cute/arch/copy_sm100.hpp>  // sub-byte ldmatrix（fp4 的 smem->寄存器）
#include <cute/atom/mma_traits_sm120.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/device_kernel.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

namespace fp4_gemm {
using namespace cute;

// =============================================================================
// Phase 10.1: NVFP4 线格式与量化数学
// =============================================================================
// e2m1（4-bit，1 符号 + 2 指数 + 1 尾数，共 16 个码点）：
//   可表示 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6 —— 每 2 的幂区间只有 2 个点，
//   相邻码点间距 = 2^(e-1)，相对步长最大 25%（区间下沿），RMS ~10%.
// ue4m3（4-bit 无符号指数偏置 7）：可表示 2^-9 .. 448，步长 2^-9（次正规），
//   量程 [0, 448]，最大值 448（0x7E）；0x7F 是 NaN，必须避开.
// NVFP4 两级量化（level-2 是 NVFP4 相对 MXFP4 的定义性特征）：
//   level-2: 每 16 个 K 元素一个 SF = ue4m3(amax_group / 6)，组内数据
//            x̂ = round(x / SF) ∈ [-6, 6]（e2m1 满量程）
//   level-1: 矩阵级/行级/列级一个 fp32 scale，使 level-2 的 SF 不被压进
//            ue4m3 的次正规区间（SF < 2^-9 ≈ 0.00195 会 round 到 0 -> 整组归零）
// 折叠代数（与 fp8 同一恒等式）：
//   C = A·B^T = (sA·Â)·(sB·B̂)^T = sA·sB·(Â·B̂^T)，其中 Â 的 per-16 scale 由
//   MMA 硬件消费，sA/sB 在 epilogue 逐行/列乘回.
constexpr float kE2M1Max = 6.0f;        // e2m1 最大有限值（饱和上界）
constexpr float kUe4m3Max = 448.0f;     // ue4m3 最大有限值
constexpr float kFp4ScaleMax = kE2M1Max * kUe4m3Max;  // 2688 = 两级量程之积
constexpr int kSFVec = 16;              // 每个缩放因子覆盖的 K 元素数（scale_vec::4X）
constexpr int kRowBlock = 128;          // SF gmem 原子的行块（CUTLASS Blk_MN）
constexpr int kSFVecPerBlk = 4;         // 每 64 个 K 元素 4 个 SF（CUTLASS Blk_SF）

// 4 组合 scale 模式：level-1 的有无 x A/B 两侧
enum class Fp4GemmScaleMode {
  kSingleLevel,   // 只用 per-16 level-2（无 level-1，小幅度行会整组归零）
  kLevel1A,       // A 加 per-row level-1（scale 数 M）
  kLevel1B,       // B 加 per-col level-1（scale 数 N）
  kTwoLevel,      // A per-row + B per-col（精度最优，scale 数 M+N）
};

union Vec8BF16 {  // 8 x bf16 = 16B, 量化 kernel 的向量化 IO 单元
  uint4 raw;
  __nv_bfloat16 elem[8];
};
static_assert(sizeof(Vec8BF16) == 16, "Vec8BF16 must be 128-bit");

// f32 -> e2m1 两条打包：PTX 规定转换结果 a 进高 4 位、b 进低 4 位，故实参顺序
// 为 (hi, lo). rn.satfinite：越界时饱和到 ±6 而非 Inf/NaN（NaN 输入 -> +6）.
// 目标寄存器必须显式声明为 .b8（cvt 的 d 是 .b8，不能用 "=h" 的 u16 寄存器），
// 再 mov.b32 拼进 u32 取低字节 —— 与 CUTLASS NumericConverter 同款写法.
__device__ __forceinline__ unsigned char cvt_f2_to_e2m1x2(float hi_4bit,
                                                          float lo_4bit) {
  unsigned int tmp;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte0, %1, %2;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}\n"
      : "=r"(tmp)
      : "f"(hi_4bit), "f"(lo_4bit));
  return static_cast<unsigned char>(tmp & 0xffu);
}

// f32 -> ue4m3 单个字节：复用 e4m3 的 cvt（非负输入的 e4m3 编码与 ue4m3 相同），
// 取低字节即第一个元素. amax/6 <= 448 保证不触发饱和.
__device__ __forceinline__ unsigned char cvt_f2_to_ue4m3(float x) {
  return static_cast<unsigned char>(
      __nv_cvt_float2_to_fp8x2(make_float2(x, 0.0f), __NV_SATFINITE, __NV_E4M3));
}

// ue4m3 字节 -> f32（量化 kernel 里判断 SF 是否下溢到 0 用）
__device__ __forceinline__ float cvt_ue4m3_to_f32(unsigned char s) {
  return __half2float(__nv_cvt_fp8_to_halfraw(s, __NV_E4M3));
}

// SF gmem 字节偏移（A 侧；B 侧把行号换成 n 即可，布局完全同构）：
//   off(mn, s) = (mn/128)*NB_K*512 + (s/4)*512 + (mn%32)*16 + ((mn/32)%4)*4 + (s%4)
// 其中 s 是全局 K 组号（= k/16），NB_K = K/64. 该式与 cute 的
// tile_atom_to_shape_SFA 逐字节一致（.tmp/fp4bs/probe_layout.cu 全量枚举验证）.
__host__ __device__ constexpr int sf_byte_offset(int mn, int s, int nb_k) {
  return (mn / kRowBlock) * nb_k * 512 + (s / kSFVecPerBlk) * 512 +
         (mn % 32) * 16 + ((mn / 32) % 4) * 4 + (s % kSFVecPerBlk);
}

// 一个 SF 缓冲的字节数（mn 按 128 行向上取整）
__host__ __device__ constexpr size_t sf_buffer_bytes(int mn, int K) {
  return (size_t)((mn + kRowBlock - 1) / kRowBlock) * (K / 64) * 512;
}

// SF gmem 张量的 canonical 布局（= CUTLASS Sm1xxBlockScaledConfig 的 SfAtom
// 平铺到 (mn, K)，行数自动向上取整到 128 的倍数）。K 方向 16 元素有一层
// 广播模式（16 个 K 元素共用一个 SF），因此该布局的坐标仍是数据单位.
CUTE_HOST_DEVICE auto sf_gmem_layout(int mn, int K) {
  using Config = cutlass::detail::Sm1xxBlockScaledConfig<kSFVec>;
  return tile_to_shape(typename Config::SfAtom{}, make_shape(mn, K), Step<_2, _1>{});
}

// =============================================================================
// Phase 10.2: 量化前处理 kernel
// =============================================================================
// A 侧两条 pass（kLevel1 = true 时）：pass1 求行 amax 得到 sA[m] = rowmax/2688，
// pass2 每线程负责一个 16 元素组：x' = x / sA[m]，SF = ue4m3(amax(x')/6)，
// 8 字节 fp4 写回（一线程一组的映射让 32B 读与 8B 写都天然合并）.
__global__ void row_amax_fp4_kernel(const __nv_bfloat16 *__restrict__ A,
                                    float *__restrict__ sa, int M, int K) {
  constexpr int kVec = 8;
  const int row = blockIdx.x * 8 + threadIdx.x / 32;  // 8 warps/block
  if (row >= M) return;
  const int lane = threadIdx.x % 32;
  const __nv_bfloat16 *ar = A + (size_t)row * K;
  float amax = 0.0f;
  for (int c = lane * kVec; c < K; c += 32 * kVec) {
    Vec8BF16 v = *reinterpret_cast<const Vec8BF16 *>(ar + c);
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      amax = fmaxf(amax, fabsf(__bfloat162float(v.elem[e])));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
  if (lane == 0) sa[row] = amax / kFp4ScaleMax;  // 全零行 -> 0（epilogue 相乘即 0）
}

// A[M,K] bf16 -> A4 packed(M,K) + SFA（含 128 行向上取整的补零行）
// kLevel1: 是否先按 sA[m] 归一化（false = 纯 level-2）
template <bool kLevel1>
__global__ void quantize_a_fp4_kernel(const __nv_bfloat16 *__restrict__ A,
                                      cutlass::float_e2m1_t *__restrict__ a4,
                                      cutlass::float_ue4m3_t *__restrict__ sfa,
                                      const float *__restrict__ sa, int M,
                                      int K) {
  const int groups_per_row = K / kSFVec;
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  const int mpad = (M + kRowBlock - 1) / kRowBlock * kRowBlock;
  if (g >= mpad * groups_per_row) return;
  const int m = g / groups_per_row;   // 行（可能落在补零区）
  const int grp = g % groups_per_row;
  const int k0 = grp * kSFVec;
  const int nb_k = K / 64;

  if (m >= M) {  // 补零行：SF 显式写 0。TMA 只做 OOB 零填充，缓冲区里的
    // 未初始化字节会被当作 SF 乘进 MMA（0 x NaN = NaN），必须清零
    sfa[sf_byte_offset(m, grp, nb_k)] = cutlass::float_ue4m3_t::bitcast(0);
    return;
  }

  float inv1 = 1.0f;
  if constexpr (kLevel1) {  // 行级归一化：x' = x / sA[m]，sA[m] = rowmax/2688
    const float s = sa[m];
    inv1 = (s > 0.0f) ? (1.0f / s) : 0.0f;
  }

  const __nv_bfloat16 *ar = A + (size_t)m * K + k0;
  Vec8BF16 v0 = *reinterpret_cast<const Vec8BF16 *>(ar);
  Vec8BF16 v1 = *reinterpret_cast<const Vec8BF16 *>(ar + 8);
  float x[kSFVec];
  float amax = 0.0f;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const float a = __bfloat162float(v0.elem[e]) * inv1;
    const float b = __bfloat162float(v1.elem[e]) * inv1;
    x[e] = a;
    x[e + 8] = b;
    amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
  }

  const unsigned char sf_byte = cvt_f2_to_ue4m3(amax / kE2M1Max);
  const float sf_val = cvt_ue4m3_to_f32(sf_byte);
  // SF 下溢到 0（amax < 6 * 2^-10）-> 整组归零（inv2 = 0），避免 0/0 = NaN
  const float inv2 = (sf_val > 0.0f) ? (1.0f / sf_val) : 0.0f;
  sfa[sf_byte_offset(m, grp, nb_k)] = cutlass::float_ue4m3_t::bitcast(sf_byte);

  uint2 out;
  unsigned char *o = reinterpret_cast<unsigned char *>(&out);
#pragma unroll
  for (int e = 0; e < 8; ++e) {  // 8 个字节 = 16 个 4-bit
    o[e] = cvt_f2_to_e2m1x2(x[2 * e + 1] * inv2, x[2 * e] * inv2);
  }
  *reinterpret_cast<uint2 *>(reinterpret_cast<unsigned char *>(a4) +
                             (size_t)m * (K / 2) + grp * 8) = out;
}

// B 侧列 amax 与转置量化拆成两个全并行 kernel（Phase 10.2b）。
//
// 旧写法把两者压进同一个 kernel：一个 block 包下 128 列 x 全部 K，grid 只有
// N/128（N=4096 时 32 个 CTA，只喂满 29% 的 SM）；而且逐列归约读 smem 的步长
// 是 128 个元素 = 64 个字 = 32 个 bank 的整数倍，每读一次 32 路冲突。实测吞吐
// 约 194 GB/s（可达带宽的五分之一），在线量化 e2e 因此被压在 1.2x cuBLAS；拆开后两段
// 都吃满并行度、各只读 B 一遍，数学逐位一致：194 -> 792 GB/s，e2e 2.04x（196.8 -> 334.2）。

// 列 amax（level-1 用）：grid = (ceil(N/128), kSplit)，块内 16 条 k-lane x
// 16 组（每组 8 列）共 256 线程。每线程持 8 列、K 方向步进 16，一次 16B 载入
// 拿满 8 列；warp 内 2 行 x 128 列 = 4 条 128B 事务，正好吃满。
__global__ void col_amax_fp4_kernel(const __nv_bfloat16 *__restrict__ B,
                                    float *__restrict__ sb, int N, int K) {
  constexpr int kVec = 8;                // 每线程 8 列 = 一次 16B 载入
  constexpr int kKLanes = 16;            // 块内 k 方向 lane 数
  constexpr int kBNt = 128;              // 每块 128 列
  constexpr int kNPerRow = kBNt / kVec;  // 16
  const int n0 = blockIdx.x * kBNt;
  const int kBeg = (int)((long long)K * blockIdx.y / gridDim.y);
  const int kEnd = (int)((long long)K * (blockIdx.y + 1) / gridDim.y);
  const int tid = threadIdx.x;
  const int nl = (tid % kNPerRow) * kVec;  // 列起点 0..120
  const int kl = tid / kNPerRow;           // k lane 0..15
  __shared__ float part[kKLanes][kBNt];

  float a[kVec];
#pragma unroll
  for (int e = 0; e < kVec; ++e) a[e] = 0.0f;
  if (n0 + nl < N) {
    for (int k = kBeg + kl; k < kEnd; k += kKLanes) {
      Vec8BF16 v =
          *reinterpret_cast<const Vec8BF16 *>(B + (size_t)k * N + n0 + nl);
#pragma unroll
      for (int e = 0; e < kVec; ++e)
        a[e] = fmaxf(a[e], fabsf(__bfloat162float(v.elem[e])));
    }
  }
#pragma unroll
  for (int e = 0; e < kVec; ++e) part[kl][nl + e] = a[e];
  __syncthreads();
  if (tid < kBNt && n0 + tid < N) {
    float m = 0.0f;
#pragma unroll
    for (int r = 0; r < kKLanes; ++r) m = fmaxf(m, part[r][tid]);
    // 先除以 2688 再 atomicMax：除以正常数保序，所以「先除后取 max」与旧版
    // 「先取 max 再除」逐位相同。amax >= 0，IEEE754 非负数的位模式与无符号
    // 整数序一致，可直接用整数 atomicMax（调用方需先把 sb 清零）。
    atomicMax(reinterpret_cast<unsigned *>(sb + n0 + tid),
              __float_as_uint(m / kFp4ScaleMax));
  }
}

// B 转置量化：B[K,N] bf16 -> B4T packed(N,K) + SFB。
// grid = (ceil(N/128), K/64)：一个 block 只做一个 128(n) x 64(k) 的 tile——
// 载入 -> smem 转置 -> 逐 16-K 组量化（组 amax 完全落在块内，无跨块归约）
// -> 8B 合并写 B4T（行 n、K 方向 8 个字节 = SFB 512B 原子里的 4 个字节）。
// 与 fp8 版本同构，区别是 level-2 的 per-16 SF 逐组产出并落到 512B SF 原子。
template <bool kLevel1>
__global__ void quantize_bt_tile_fp4_kernel(
    const __nv_bfloat16 *__restrict__ B,
    cutlass::float_e2m1_t *__restrict__ b4t,
    cutlass::float_ue4m3_t *__restrict__ sfb, const float *__restrict__ sb,
    int K, int N) {
  constexpr int kBNt = 128;                    // n 方向 tile（= B4T 行数）
  constexpr int kBKt = 64;                     // k 方向 tile（= B 行数）
  constexpr int kVec = 8;
  constexpr int kChunksPerRow = kBNt / kVec;   // 16
  constexpr int kChunks = kBKt * kChunksPerRow;
  const int n0 = blockIdx.x * kBNt;
  const int k0 = blockIdx.y * kBKt;
  const int tid = threadIdx.x;
  const int nb_k = K / 64;

  __shared__ __align__(16) __nv_bfloat16 tile[kBKt][kBNt];  // 源朝向 16KB
  // 转置朝向 16.6KB。多出来的 1 列是 padding：写 tile_t[c8+e][r] 时 e 方向
  // 的步长是整行，行宽 64 个字会让 16 个同 r 的线程全撞在同一个 bank 上。
  __shared__ __align__(16) __nv_bfloat16 tile_t[kBNt][kBKt + 1];

  for (int i = tid; i < kChunks; i += 256) {  // coalesced 载入 128(n) x 64(k)
    const int r = i / kChunksPerRow, c8 = (i % kChunksPerRow) * kVec;
    Vec8BF16 v;
    v.raw = make_uint4(0, 0, 0, 0);
    if (n0 + c8 < N)
      v =
          *reinterpret_cast<const Vec8BF16 *>(B + (size_t)(k0 + r) * N + n0 + c8);
    *reinterpret_cast<uint4 *>(&tile[r][c8]) = v.raw;
  }
  __syncthreads();
  for (int i = tid; i < kChunks; i += 256) {  // 散写转置朝向的 tile_t
    const int r = i / kChunksPerRow, c8 = (i % kChunksPerRow) * kVec;
#pragma unroll
    for (int e = 0; e < kVec; ++e) tile_t[c8 + e][r] = tile[r][c8 + e];
  }
  __syncthreads();
  // 每个 (n, 16-K 组) 一组：128 n x 4 组 = 512 组 / 256 线程 = 2 组/线程
  for (int i = tid; i < kBNt * (kBKt / kSFVec); i += 256) {
    const int n = i / (kBKt / kSFVec), g = i % (kBKt / kSFVec);
    if (n0 + n >= N) {  // 补零行：SF 必须显式写 0（与 A 侧同一条理由）
      sfb[sf_byte_offset(n0 + n, (k0 + g * kSFVec) / kSFVec, nb_k)] =
          cutlass::float_ue4m3_t::bitcast(0);
      continue;
    }
    float inv1 = 1.0f;
    if constexpr (kLevel1) {  // sb[n] = 列 amax / 2688，由 amax kernel 产出
      const float s = sb[n0 + n];
      inv1 = (s > 0.0f) ? (1.0f / s) : 0.0f;
    }
    float x[kSFVec];
    float amax = 0.0f;
#pragma unroll
    for (int e = 0; e < kSFVec; ++e) {
      x[e] = __bfloat162float(tile_t[n][g * kSFVec + e]) * inv1;
      amax = fmaxf(amax, fabsf(x[e]));
    }
    const int s = (k0 + g * kSFVec) / kSFVec;  // 全局 K 组号
    const unsigned char sf_byte = cvt_f2_to_ue4m3(amax / kE2M1Max);
    const float sf_val = cvt_ue4m3_to_f32(sf_byte);
    // SF 下溢到 0（amax < 6 * 2^-10）-> 整组归零（inv2 = 0），避免 0/0 = NaN
    const float inv2 = (sf_val > 0.0f) ? (1.0f / sf_val) : 0.0f;
    sfb[sf_byte_offset(n0 + n, s, nb_k)] =
        cutlass::float_ue4m3_t::bitcast(sf_byte);
    uint2 out;
    unsigned char *o = reinterpret_cast<unsigned char *>(&out);
#pragma unroll
    for (int e = 0; e < 8; ++e)
      o[e] = cvt_f2_to_e2m1x2(x[2 * e + 1] * inv2, x[2 * e] * inv2);
    *reinterpret_cast<uint2 *>(reinterpret_cast<unsigned char *>(b4t) +
                               (size_t)(n0 + n) * (K / 2) + k0 / 2 + g * 8) =
        out;
  }
}

// B 侧量化入口：全链路与 B 离线量化共用。level1=false 时列级 sb 无意义
// （inv1 恒为 1），只跑转置量化；level1=true 时先跑列 amax（K 切成 8 份，
// N=4096 时 256 个 CTA）再跑转置量化（N=4096,K=4096 时 2048 个 CTA）。
inline void fp4_gemm_quantize_b(const __nv_bfloat16 *B,
                                cutlass::float_e2m1_t *b4t,
                                cutlass::float_ue4m3_t *sfb, float *sb, int N,
                                int K, bool level1, cudaStream_t stream) {
  if (K % 64 != 0 || N % 8 != 0) {
    fprintf(stderr,
            "fp4_gemm_quantize_b: require K%%64==0 (got K=%d), N%%8==0 (got "
            "N=%d)\n",
            K, N);
    return;
  }
  const dim3 grid((N + 127) / 128, K / 64);
  if (level1) {
    cudaMemsetAsync(sb, 0, (size_t)N * sizeof(float), stream);
    const int kSplit = (K >= 512) ? 8 : 1;
    const dim3 grid_sa((N + 127) / 128, kSplit);
    col_amax_fp4_kernel<<<grid_sa, 256, 0, stream>>>(B, sb, N, K);
    quantize_bt_tile_fp4_kernel<true>
        <<<grid, 256, 0, stream>>>(B, b4t, sfb, sb, K, N);
  } else {
    quantize_bt_tile_fp4_kernel<false>
        <<<grid, 256, 0, stream>>>(B, b4t, sfb, sb, K, N);
  }
}

// =============================================================================
// Phase 10.3: CuTe 工具（最小集，与 fp8_gemm.cuh P9.3 同源）
// =============================================================================
// acc (MMA=4, MMA_M, MMA_N) -> 行列二维视图：epilogue 按行列坐标查 level-1 scale
template <typename Layout>
CUTE_DEVICE auto convert_layout_acc_rowcol(Layout acc_layout) {
  auto divided = logical_divide(acc_layout, Shape<_2>{});
  return make_layout(make_layout(get<0, 1>(divided), get<1>(divided)),
                     make_layout(get<0, 0>(divided), get<2>(divided)));
}

// f32 acc -> bf16，寄存器内批量转换（NumericArrayConverter），零 smem 往返
template <typename To, typename Engine, typename Layout>
CUTE_DEVICE auto convert_type(Tensor<Engine, Layout> const &tensor) {
  using From = typename Engine::value_type;
  constexpr int kElements = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To, From, kElements> convert;
  auto fragment = convert(
      *reinterpret_cast<cutlass::Array<From, kElements> const *>(tensor.data()));
  return make_tensor(make_rmem_ptr<To>(&fragment), tensor.layout());
}

// =============================================================================
// Phase 10.3b: SF 片段工具（照抄 CUTLASS collective 的 6 个 helper）
// =============================================================================
// blockscaled mma 的 SF 操作数既不是普通 smem tensor 也不是普通 fragment：
// atom 的 SFALayout/SFBLayout 把 (thread,value) 映射到 (M,K) 时带 0-步长
// 广播模式（SFA 只有 16 个线程有唯一 SF，SFB 只有 8 个），必须用下面这套
// "thrfrg -> 取本线程切片 -> make_fragment_like" 的流程才能得到硬件期望的
// 1x u32 寄存器片段。出处：cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp
namespace sf_partition {

// (ThrV,FrgV),(RestM,RestK) 视图：把 SF tensor 重排成 atom 期望的层次
template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto thrfrg_sfa(SFATensor &&sfatensor,
                                           TiledMMA<Atom, TiledThr, TiledPerm> &mma) {
  CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});
  using AtomShape_MNK = typename Atom::Shape_MNK;
  using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;
  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto t_tile = make_tile(get<0>(permutation_mnk), get<2>(permutation_mnk));
  auto t_tensor = logical_divide(sfatensor, t_tile);
  auto a_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                          make_layout(size<2>(AtomShape_MNK{})));
  auto a_tensor = zipped_divide(t_tensor, a_tile);
  auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{}, _);
  auto thr_tile = make_tile(_, make_tile(make_layout(size<1>(thr_layout_vmnk)),
                                         make_layout(size<3>(thr_layout_vmnk))));
  return zipped_divide(tv_tensor, thr_tile);
}

template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto thrfrg_sfb(SFBTensor &&sfbtensor,
                                           TiledMMA<Atom, TiledThr, TiledPerm> &mma) {
  CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});
  using AtomShape_MNK = typename Atom::Shape_MNK;
  using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;
  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto t_tile = make_tile(get<1>(permutation_mnk), get<2>(permutation_mnk));
  auto t_tensor = logical_divide(sfbtensor, t_tile);
  auto a_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})),
                          make_layout(size<2>(AtomShape_MNK{})));
  auto a_tensor = zipped_divide(t_tensor, a_tile);
  auto tv_tensor = a_tensor.compose(AtomLayoutSFB_TV{}, _);
  auto thr_tile = make_tile(_, make_tile(make_layout(size<2>(thr_layout_vmnk)),
                                         make_layout(size<3>(thr_layout_vmnk))));
  return zipped_divide(tv_tensor, thr_tile);
}

// SF 寄存器片段：每线程 1x u32（4 个 ue4m3 字节 = 4 个 K 组）
template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_fragment_sfa(SFATensor &&sfatensor,
                                                       ThrMma &thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  auto thr_tensor = make_tensor(static_cast<SFATensor &&>(sfatensor).data(),
                                thrfrg_sfa(sfatensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vmk =
      make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
  auto part = thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
  return make_fragment_like<ValTypeSF>(part);
}

template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_fragment_sfb(SFBTensor &&sfbtensor,
                                                       ThrMma &thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  auto thr_tensor = make_tensor(static_cast<SFBTensor &&>(sfbtensor).data(),
                                thrfrg_sfb(sfbtensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vnk =
      make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
  auto part = thr_tensor(thr_vnk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
  return make_fragment_like<ValTypeSF>(part);
}

// (thr_idx,val) -> (M,K) 的 TV layout，交给 make_tiled_copy_impl 构造
// "smem->寄存器的 SF 拷贝"
template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto layout_sfa_tv(TiledMma &mma) {
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto atile = make_tile(_, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk),
                                                            size<2>(thr_layout_vmnk)),
                                                  make_stride(Int<1>{}, Int<0>{})),
                                      _));
  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  return thrfrg_sfa(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
}

template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto layout_sfb_tv(TiledMma &mma) {
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto btile = make_tile(_, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk),
                                                            size<2>(thr_layout_vmnk)),
                                                  make_stride(Int<0>{}, Int<1>{})),
                                      _));
  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  return thrfrg_sfb(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
}

}  // namespace sf_partition

// SF 的 smem atom（照抄 sm120_blockscaled_mma_builder 的 4 组式子）：
// 行方向 ((32,4)):(_16,_4) 把 128 行块铺进 512B；K 方向 (16,4):(0,1) 表示
// 组内 16 个元素共享一个 SF（步长 0），4 个 K 组步长 1（连续 4 字节）.
// kMN = 覆盖行数（A 侧 128，B 侧 max(BN,128)）.
template <class TiledMma_, int kMN, int kBK_>
struct SfSmemAtom {
  static constexpr int kMMA_NSF =
      size<2>(typename TiledMma_::AtomShape_MNK{}) / kSFVec;  // 64/16 = 4
  using Config = cutlass::detail::Sm1xxBlockScaledConfig<kSFVec>;
  using Blk_Elems = decltype(typename Config::Blk_MN{} * typename Config::Blk_SF{});
  using mnBlockShape = Shape<_32, _4>;
  using mnBlockStride = Stride<_16, _4>;
  using kBlockShape = Shape<Int<kSFVec>, Int<kMMA_NSF>>;
  using kBlockStride = Stride<_0, _1>;
  using shapeM = decltype(prepend(Int<kMN / kRowBlock>{}, mnBlockShape{}));
  using strideM = decltype(prepend(Blk_Elems{}, mnBlockStride{}));
  using shapeK = decltype(prepend(
      make_shape(Int<kSFVecPerBlk / kMMA_NSF>{},
                 Int<kBK_ / kSFVec / kSFVecPerBlk>{}),
      kBlockShape{}));
  using strideK = decltype(prepend(
      make_stride(Int<kMMA_NSF>{}, Int<kMN / kRowBlock * 512>{}), kBlockStride{}));
  using type = decltype(make_layout(make_shape(shapeM{}, shapeK{}),
                                    make_stride(strideM{}, strideK{})));
};

// =============================================================================
// Phase 10.4: NVFP4 GEMM Traits（atom + smem 布局 + TiledMma）
// =============================================================================
// BM 固定 128：SF gmem/smem 原子的行块就是 128，BM<128 会让 builder 那套
// size<0>(TileShape)/Blk_MN 退化成空布局；BN >= 32（perm tile 是 32 宽）.
template <int kBN_ = 128, int kStages_ = 4>
struct Fp4GemmTraits {
  static constexpr int kBM = 128;
  static constexpr int kBN = kBN_;
  static constexpr int kBK = 64;      // = atom K：一个 stage 正好一跳
  static constexpr int kStages = kStages_;
  static constexpr int kNumThreads = kBM / 16 * 32;  // 8 warps = AtomLayout(4,2,1)
  static_assert(kBN % 128 == 0,
                "kBN 必须是 128 的倍数：SF 的 smem 原子以 128 行为一个行块，"
                "且 SF 的 TMA box 必须与该原子顶层 size 相等（BN<128 时 box "
                "填不满一个行块，直接编译期报 TMA size 不等价）");
  static_assert(kStages >= 2, "kStages >= 2（两级流水才有 overlap）");

  using Element = cutlass::float_e2m1_t;      // 4-bit 数据
  using ElementSF = cutlass::float_ue4m3_t;   // 8-bit SF
  using ElementO = cutlass::bfloat16_t;
  using ElementAcc = float;

  // MXF4_NVF4 路径（A/B 均 4-bit 且 K-major）选 k64 + uint4_t smem 分配类型
  using MmaAtom = SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<Element, Element,
                                                          ElementAcc, ElementSF,
                                                          kSFVec>;
  using AtomLayoutMNK = Layout<Shape<_4, _2, _1>>;
  using PermTileM = Int<kBM>;                      // min(BM,128)
  using PermTileN = Layout<Shape<_8, _2, _2>, Stride<_1, _16, _8>>;  // 32
  using PermTileK = _64;
  using TiledMma = decltype(make_tiled_mma(MmaAtom{}, AtomLayoutMNK{},
                                           Tile<PermTileM, PermTileN, PermTileK>{}));
  static_assert(size<0>(typename TiledMma::AtomShape_MNK{}) == 16 &&
                    size<2>(typename TiledMma::AtomShape_MNK{}) == kBK,
                "atom 必须是 16x8x64（K 与 BK 同宽，一跳一 stage）");

  // 数据 smem：4-bit 用 SW32 原子（8 行 x 64 个 4-bit = 32B/行）：布局单位是
  // 4-bit 元素，与 gmem/TMA 的 (M,K) 语义一致；smem 指针步进按「元素」计
  // （见 kernel 内 kAStageBytes 的说明）。
  using SmemAlloc = uint4_t;
  using SmemLayoutAtomA = UMMA::Layout_K_SW32_Atom<SmemAlloc>;
  using SmemLayoutAtomB = UMMA::Layout_K_SW32_Atom<SmemAlloc>;
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{}, Shape<Int<kBM>, Int<kBK>, Int<kStages>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{}, Shape<Int<kBN>, Int<kBK>, Int<kStages>>{}));

  // SF smem：块内 512B（128 行 x 4 SF），stage 追加模式（stride = 非零模式之积）
  using SmemLayoutAtomSFA = typename SfSmemAtom<TiledMma, kBM, kBK>::type;
  using SmemLayoutAtomSFB =
      typename SfSmemAtom<TiledMma, (kBN > kRowBlock ? kBN : kRowBlock), kBK>::type;
  using SmemLayoutSFA = decltype(make_layout(
      append(shape(SmemLayoutAtomSFA{}), Int<kStages>{}),
      append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))));
  using SmemLayoutSFB = decltype(make_layout(
      append(shape(SmemLayoutAtomSFB{}), Int<kStages>{}),
      append(stride(SmemLayoutAtomSFB{}), size(filter_zeros(SmemLayoutAtomSFB{})))));

  // 数据 smem->寄存器：4-bit 走 SM75 的 ldmatrix（16B/行 = 32 个 4-bit，
  // 正好是 SW32 原子的一行），A/B 同 atom（TileShapeN >= 32）
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, SmemAlloc>;

  // O staging（bf16 BM x BN），复用 K 循环结束后的 A/B/SF 区
  using SmemAtomO = GMMA::Layout_K_SW128_Atom<ElementO>;
  using SmemLayoutO = decltype(tile_to_shape(SmemAtomO{}, Shape<Int<kBM>, Int<kBN>>{}));

  // 每 stage 的字节数：4-bit 布局的单位是「4-bit 元素」，故 cosize/2 才是字节；
  // SF 是 1B/元素。smem 里各缓冲的基址按这些字节数递推（kernel 内用 uint8_t*
  // 做字节算术，再由 recast_ptr 转成 4-bit 视图，避免单位混淆）。
  static constexpr int kAStageBytes = cosize(SmemLayoutA{}) / kStages / 2;
  static constexpr int kBStageBytes = cosize(SmemLayoutB{}) / kStages / 2;
  static constexpr int kSFAStageBytes = cosize(SmemLayoutSFA{}) / kStages;
  static constexpr int kSFBStageBytes = cosize(SmemLayoutSFB{}) / kStages;
  static constexpr int kStageBytes =
      kAStageBytes + kBStageBytes + kSFAStageBytes + kSFBStageBytes;
  static constexpr int kSmemBytes = kStages * kStageBytes;
  // TMA 事务字节数（mbarrier expect_tx）：4-bit 数据按打包后的字节算
  static constexpr uint32_t kTxBytes = kBM * kBK / 2 + kBN * kBK / 2 +
                                       kSFAStageBytes + kSFBStageBytes;
  // O staging（bf16 BM x BN）能塞进 kStages 个 stage 的复用区吗？BN=256 时
  // O 要 64KB，每 stage 只有 13.5KB -> kStages>=5 才成立.
  static constexpr bool kOStagingFits =
      cosize(SmemLayoutO{}) * sizeof(ElementO) <= kSmemBytes;
  static_assert(kOStagingFits,
                "O staging（bf16 BM x BN）必须能复用 A/B/SF 区：加大 kStages");
};

// =============================================================================
// Phase 10.5: 主 kernel（非 WS） — 全员 MMA + tid0 内联发 TMA
// =============================================================================
// 执行协议（与 fp8_gemm.cuh P9.5 逐条同构，仅多两路 SF 的 TMA）：
//   full[stage]（TmaBarrier, init=1）: tid0 arrive_and_expect_tx（一次覆盖
//     A/B/SFA/SFB 四段字节）+ 4 个 TMA；消费者 wait 后读，TMA 写完自动翻转
//   empty[stage]（CtaBarrier, init=kNumThreads）: 每个线程 arrive 一次
//   phase = (kt / kStages) & 1；预取 kt+kStages 在本轮 arrive 之后发出
// K 尾：K % 64 == 0 是硬约束（SF 以 64 为一整块），故没有 K 尾分支；
//   M 尾行 TMA load 自动零填充（A4 无贡献）、C 尾行 TMA store 自动裁剪.
template <typename Traits, typename TmaA, typename TmaB, typename TmaSFA,
          typename TmaSFB, typename TmaC, bool kLevel1A, bool kLevel1B>
__global__ void __launch_bounds__(Traits::kNumThreads, 1)
    fp4_gemm_tma_kernel(CUTLASS_GRID_CONSTANT TmaA const tma_a,
                        CUTLASS_GRID_CONSTANT TmaB const tma_b,
                        CUTLASS_GRID_CONSTANT TmaSFA const tma_sfa,
                        CUTLASS_GRID_CONSTANT TmaSFB const tma_sfb,
                        CUTLASS_GRID_CONSTANT TmaC const tma_c,
                        const float *__restrict__ sa,
                        const float *__restrict__ sb, int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using namespace cute;
  using cute::tma_store_arrive;
  using cute::tma_store_wait;
  using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
  using CtaBarrier = cutlass::arch::ClusterBarrier;
  using Element = typename Traits::Element;
  using ElementSF = typename Traits::ElementSF;
  using ElementO = typename Traits::ElementO;
  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutSFA = typename Traits::SmemLayoutSFA;
  using SmemLayoutSFB = typename Traits::SmemLayoutSFB;
  using SmemLayoutAtomSFA = typename Traits::SmemLayoutAtomSFA;
  using SmemLayoutAtomSFB = typename Traits::SmemLayoutAtomSFB;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using TiledMma = typename Traits::TiledMma;
  using SmemCopyAtom = typename Traits::SmemCopyAtom;

  constexpr int kBM = Traits::kBM;
  constexpr int kBN = Traits::kBN;
  constexpr int kBK = Traits::kBK;
  constexpr int kStages = Traits::kStages;
  constexpr int kNumThreads = Traits::kNumThreads;
  constexpr int kAStageBytes = Traits::kAStageBytes;  // 布局占位（字节）
  constexpr int kBStageBytes = Traits::kBStageBytes;
  constexpr int kSFAStageBytes = Traits::kSFAStageBytes;
  constexpr int kSFBStageBytes = Traits::kSFBStageBytes;

  const int tid = threadIdx.x;
  const int m0 = blockIdx.y * kBM;  // grid: (n-tiles, m-tiles)
  const int n0 = blockIdx.x * kBN;

  auto mA = tma_a.get_tma_tensor(make_shape(M, K));
  auto mB = tma_b.get_tma_tensor(make_shape(N, K));
  auto mC = tma_c.get_tma_tensor(make_shape(M, N));
  // SF 张量的坐标仍是数据单位 (MN, K)：SF 布局在 K 方向有 16 元素广播模式。
  // 描述符的 g_stride 含 ScaledBasis（16U4/8B 对齐），g_shape 必须与它嵌套
  // 同构，故用 SF gmem 布局的 shape 构造（CUTLASS collective 同款）
  auto mSFA = tma_sfa.get_tma_tensor(shape(sf_gmem_layout(M, K)));
  auto mSFB = tma_sfb.get_tma_tensor(shape(sf_gmem_layout(N, K)));
  auto a_slice = tma_a.get_slice(_0{});
  auto b_slice = tma_b.get_slice(_0{});
  auto sfa_slice = tma_sfa.get_slice(_0{});
  auto sfb_slice = tma_sfb.get_slice(_0{});
  auto c_slice = tma_c.get_slice(_0{});

  extern __shared__ __align__(1024) uint8_t shm[];
  // 先按字节算各缓冲基址，再用 recast_ptr 转成 4-bit 视图（subbyte_iterator
  // 才能按 4-bit 单位寻址；直接对 4-bit 类型的裸指针做算术会按 sizeof=1 走字节）
  uint8_t *a_bytes = shm;
  uint8_t *b_bytes = a_bytes + kStages * kAStageBytes;
  ElementSF *sfa_base =
      reinterpret_cast<ElementSF *>(b_bytes + kStages * kBStageBytes);
  ElementSF *sfb_base = sfa_base + kStages * kSFAStageBytes;

  __shared__ uint64_t full[kStages];
  __shared__ uint64_t empty[kStages];
  if (tid == 0) {
    for (int s = 0; s < kStages; ++s) {
      TmaBarrier::init(&full[s], 1);
      CtaBarrier::init(&empty[s], kNumThreads);
    }
  }
  __syncthreads();

  auto sA_full = make_tensor(make_smem_ptr(recast_ptr<uint4_t>(a_bytes)), SmemLayoutA{});
  auto sB_full = make_tensor(make_smem_ptr(recast_ptr<uint4_t>(b_bytes)), SmemLayoutB{});
  auto sSFA_full = make_tensor(make_smem_ptr(sfa_base), SmemLayoutSFA{});
  auto sSFB_full = make_tensor(make_smem_ptr(sfb_base), SmemLayoutSFB{});

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto s2r_copy_a = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto s2r_copy_b = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto s2r_thr_a = s2r_copy_a.get_thread_slice(tid);
  auto s2r_thr_b = s2r_copy_b.get_thread_slice(tid);
  auto tCsA = s2r_thr_a.partition_S(as_position_independent_swizzle_tensor(sA_full));
  auto tCsB = s2r_thr_b.partition_S(as_position_independent_swizzle_tensor(sB_full));
  auto tCrA = thr_mma.partition_fragment_A(sA_full(_, _, Int<0>{}));
  auto tCrB = thr_mma.partition_fragment_B(sB_full(_, _, Int<0>{}));
  auto tCrA_view = s2r_thr_a.retile_D(tCrA);
  auto tCrB_view = s2r_thr_b.retile_D(tCrB);

  // SF：片段一次性建立（形状只依赖布局），smem->reg 走 UniversalCopy（自动矢量化）
  using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;
  auto tCrSFA = sf_partition::partition_fragment_sfa(sSFA_full(_, _, Int<0>{}), thr_mma);
  auto tCrSFB = sf_partition::partition_fragment_sfb(sSFB_full(_, _, Int<0>{}), thr_mma);
  auto copy_sfa = make_tiled_copy_impl(
      SmemCopyAtomSF{}, sf_partition::layout_sfa_tv(tiled_mma),
      make_shape(size<0>(tile_shape(tiled_mma)), size<2>(tile_shape(tiled_mma))));
  auto copy_sfb = make_tiled_copy_impl(
      SmemCopyAtomSF{}, sf_partition::layout_sfb_tv(tiled_mma),
      make_shape(size<1>(tile_shape(tiled_mma)), size<2>(tile_shape(tiled_mma))));
  auto thr_copy_sfa = copy_sfa.get_thread_slice(tid);
  auto thr_copy_sfb = copy_sfb.get_thread_slice(tid);
  auto tCsSFA = thr_copy_sfa.partition_S(as_position_independent_swizzle_tensor(sSFA_full));
  auto tCsSFB = thr_copy_sfb.partition_S(as_position_independent_swizzle_tensor(sSFB_full));
  auto tCrSFA_view = thr_copy_sfa.retile_D(tCrSFA);
  auto tCrSFB_view = thr_copy_sfb.retile_D(tCrSFB);

  auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<kBM>, Int<kBN>>{});
  clear(tCrC);
  // 行列坐标视图：epilogue 按 (m,n) 查 level-1 scale
  auto cC = make_identity_tensor(Shape<Int<kBM>, Int<kBN>>{});
  auto tScC = thr_mma.partition_C(cC);
  auto tScC_rc = make_tensor(tScC.data(), convert_layout_acc_rowcol(tScC.layout()));
  constexpr int kCRows = decltype(size<0>(tScC_rc))::value;
  constexpr int kCCols = decltype(size<1>(tScC_rc))::value;

  // TMA 发射（仅 tid0）：一个 barrier 同时覆盖 4 段字段的字节数
  constexpr uint32_t kTxBytes = Traits::kTxBytes;
  auto issue_tma = [&](int kt, int stage) {
    cutlass::arch::fence_view_async_shared();
    auto gA = local_tile(mA, Shape<Int<kBM>, Int<kBK>>{}, make_coord(blockIdx.y, kt));
    auto gB = local_tile(mB, Shape<Int<kBN>, Int<kBK>>{}, make_coord(blockIdx.x, kt));
    auto gSFA = local_tile(mSFA, Shape<Int<kBM>, Int<kBK>>{}, make_coord(blockIdx.y, kt));
    auto gSFB = local_tile(mSFB, Shape<Int<kBN>, Int<kBK>>{}, make_coord(blockIdx.x, kt));
    TmaBarrier::arrive_and_expect_tx(&full[stage], kTxBytes);
    copy(tma_a.with(full[stage]), a_slice.partition_S(gA),
         a_slice.partition_D(sA_full(_, _, stage)));
    copy(tma_b.with(full[stage]), b_slice.partition_S(gB),
         b_slice.partition_D(sB_full(_, _, stage)));
    copy(tma_sfa.with(full[stage]), sfa_slice.partition_S(gSFA),
         sfa_slice.partition_D(sSFA_full(_, _, stage)));
    copy(tma_sfb.with(full[stage]), sfb_slice.partition_S(gSFB),
         sfb_slice.partition_D(sSFB_full(_, _, stage)));
  };

  const int nKt = K / kBK;  // K % 64 == 0 是约束：无 K 尾
  for (int s = 0; s < kStages; ++s) CtaBarrier::arrive(&empty[s]);
  if (tid == 0) {
    for (int kt = 0; kt < kStages && kt < nKt; ++kt) {
      CtaBarrier::wait(&empty[kt], 0);
      issue_tma(kt, kt);
    }
  }
#pragma unroll 1
  for (int kt = 0; kt < nKt; ++kt) {
    const int stage = kt % kStages;
    const int phase = (kt / kStages) & 1;
    TmaBarrier::wait(&full[stage], phase);
    cutlass::arch::fence_view_async_shared();

    // 数据 smem -> 寄存器（LDSM），SF smem -> 寄存器（4 字节 LDS）
    copy(s2r_copy_a, tCsA(_, _, _, stage), tCrA_view);
    copy(s2r_copy_b, tCsB(_, _, _, stage), tCrB_view);
    copy(tCsSFA(_, _, _, stage), tCrSFA_view);
    copy(tCsSFB(_, _, _, stage), tCrSFB_view);
    // 一跳一个 stage（atom K == BK），故 SF 与数据一起进 zip 张量直接发 MMA
    cute::gemm(tiled_mma, make_zip_tensor(tCrA, tCrSFA),
               make_zip_tensor(tCrB, tCrSFB), tCrC);

    CtaBarrier::arrive(&empty[stage]);
    if (tid == 0) {  // 预取 kt+kStages（须在本轮 arrive 之后）
      const int kt_next = kt + kStages;
      if (kt_next < nKt) {
        const int s_next = kt_next % kStages;
        CtaBarrier::wait(&empty[s_next], (kt_next / kStages) & 1);
        issue_tma(kt_next, s_next);
      }
    }
  }

  // Epilogue: 反量化（level-1 折回：C = acc * sA[m] * sB[n]）-> bf16 ->
  // STSM r2s（复用 K 循环结束后的 A/B/SF 区作 O staging）-> TMA store（尾行裁剪）
  {
    __syncthreads();  // 全部 mma 完成，smem 可覆写
    auto tCrC_rc = make_tensor(tCrC.data(), convert_layout_acc_rowcol(tCrC.layout()));
#pragma unroll
    for (int r = 0; r < kCRows; ++r) {
      const int m = m0 + get<0>(tScC_rc(r, 0));
      const float sa_r = kLevel1A ? ((m < M) ? sa[m] : 0.0f) : 1.0f;
#pragma unroll
      for (int c = 0; c < kCCols; ++c) {
        const int n = n0 + get<1>(tScC_rc(r, c));
        const float sb_c = kLevel1B ? ((n < N) ? sb[n] : 0.0f) : 1.0f;
        tCrC_rc(r, c) = tCrC_rc(r, c) * sa_r * sb_c;
      }
    }
    auto tCrCh = convert_type<ElementO>(tCrC);

    auto r2s_copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, ElementO>{},
                                      tiled_mma);
    auto r2s_thr = r2s_copy.get_thread_slice(tid);
    auto sC = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(shm)),
                          SmemLayoutO{});  // 覆盖已空闲的 A/B/SF 区
    auto tCrCh_src = r2s_thr.retile_S(tCrCh);
    auto tCsC_dst = r2s_thr.partition_D(sC);
    copy(r2s_copy, tCrCh_src, tCsC_dst);
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    auto gC = local_tile(mC, Shape<Int<kBM>, Int<kBN>>{},
                         make_coord(blockIdx.y, blockIdx.x));
    if (tid == 0) copy(tma_c, c_slice.partition_S(sC), c_slice.partition_D(gC));
    tma_store_arrive();
    tma_store_wait<0>();
  }
#endif  // __CUDA_ARCH__ >= 900
}

// =============================================================================
// Phase 10.6: 主 kernel（WS 变体）— 128 producer + 256 consumer
// =============================================================================
// producer 只发 TMA（32 寄存器），consumer 只做 MMA + epilogue（224 寄存器，
// 与 fp8 的 232 略降：fp4 的 SF 片段额外占用少量寄存器）。
// 注意：SF smem->reg 的拷贝与 MMA 都落在 consumer 分支里，SF 片段寄存器也
// 只属于 consumer.
template <typename Traits, typename TmaA, typename TmaB, typename TmaSFA,
          typename TmaSFB, typename TmaC, bool kLevel1A, bool kLevel1B>
__global__ void __launch_bounds__(Traits::kNumThreads + 128, 1)
    fp4_gemm_tma_ws_kernel(CUTLASS_GRID_CONSTANT TmaA const tma_a,
                           CUTLASS_GRID_CONSTANT TmaB const tma_b,
                           CUTLASS_GRID_CONSTANT TmaSFA const tma_sfa,
                           CUTLASS_GRID_CONSTANT TmaSFB const tma_sfb,
                           CUTLASS_GRID_CONSTANT TmaC const tma_c,
                           const float *__restrict__ sa,
                           const float *__restrict__ sb, int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using namespace cute;
  using cute::tma_store_arrive;
  using cute::tma_store_wait;
  using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
  using CtaBarrier = cutlass::arch::ClusterBarrier;
  using Element = typename Traits::Element;
  using ElementSF = typename Traits::ElementSF;
  using ElementO = typename Traits::ElementO;
  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutSFA = typename Traits::SmemLayoutSFA;
  using SmemLayoutSFB = typename Traits::SmemLayoutSFB;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using TiledMma = typename Traits::TiledMma;
  using SmemCopyAtom = typename Traits::SmemCopyAtom;

  constexpr int kBM = Traits::kBM;
  constexpr int kBN = Traits::kBN;
  constexpr int kBK = Traits::kBK;
  constexpr int kStages = Traits::kStages;
  constexpr int kNumThreads = Traits::kNumThreads;
  constexpr int kProducerThreads = 128;
  constexpr int kAStageBytes = Traits::kAStageBytes;
  constexpr int kBStageBytes = Traits::kBStageBytes;
  constexpr int kSFAStageBytes = Traits::kSFAStageBytes;
  constexpr int kSFBStageBytes = Traits::kSFBStageBytes;
  static_assert(kProducerThreads * 32 + kNumThreads * 224 <= 65536,
                "WS 寄存器预算：128 x 32 + consumer x 224 必须装进 64K");

  const int tid = threadIdx.x;
  const int m0 = blockIdx.y * kBM;
  const int n0 = blockIdx.x * kBN;

  auto mA = tma_a.get_tma_tensor(make_shape(M, K));
  auto mB = tma_b.get_tma_tensor(make_shape(N, K));
  auto mC = tma_c.get_tma_tensor(make_shape(M, N));
  auto mSFA = tma_sfa.get_tma_tensor(shape(sf_gmem_layout(M, K)));
  auto mSFB = tma_sfb.get_tma_tensor(shape(sf_gmem_layout(N, K)));
  auto a_slice = tma_a.get_slice(_0{});
  auto b_slice = tma_b.get_slice(_0{});
  auto sfa_slice = tma_sfa.get_slice(_0{});
  auto sfb_slice = tma_sfb.get_slice(_0{});
  auto c_slice = tma_c.get_slice(_0{});

  extern __shared__ __align__(1024) uint8_t shm[];
  uint8_t *a_bytes = shm;
  uint8_t *b_bytes = a_bytes + kStages * kAStageBytes;
  ElementSF *sfa_base =
      reinterpret_cast<ElementSF *>(b_bytes + kStages * kBStageBytes);
  ElementSF *sfb_base = sfa_base + kStages * kSFAStageBytes;

  auto sA_full = make_tensor(make_smem_ptr(recast_ptr<uint4_t>(a_bytes)), SmemLayoutA{});
  auto sB_full = make_tensor(make_smem_ptr(recast_ptr<uint4_t>(b_bytes)), SmemLayoutB{});
  auto sSFA_full = make_tensor(make_smem_ptr(sfa_base), SmemLayoutSFA{});
  auto sSFB_full = make_tensor(make_smem_ptr(sfb_base), SmemLayoutSFB{});

  __shared__ uint64_t full[kStages];
  __shared__ uint64_t empty[kStages];
  if (tid == 0) {
    for (int s = 0; s < kStages; ++s) {
      TmaBarrier::init(&full[s], 1);
      CtaBarrier::init(&empty[s], kNumThreads);  // 仅 consumer arrive
    }
  }
  __syncthreads();

  constexpr uint32_t kTxBytes = Traits::kTxBytes;
  auto issue_tma = [&](int kt, int stage) {
    cutlass::arch::fence_view_async_shared();
    auto gA = local_tile(mA, Shape<Int<kBM>, Int<kBK>>{}, make_coord(blockIdx.y, kt));
    auto gB = local_tile(mB, Shape<Int<kBN>, Int<kBK>>{}, make_coord(blockIdx.x, kt));
    auto gSFA = local_tile(mSFA, Shape<Int<kBM>, Int<kBK>>{}, make_coord(blockIdx.y, kt));
    auto gSFB = local_tile(mSFB, Shape<Int<kBN>, Int<kBK>>{}, make_coord(blockIdx.x, kt));
    TmaBarrier::arrive_and_expect_tx(&full[stage], kTxBytes);
    copy(tma_a.with(full[stage]), a_slice.partition_S(gA),
         a_slice.partition_D(sA_full(_, _, stage)));
    copy(tma_b.with(full[stage]), b_slice.partition_S(gB),
         b_slice.partition_D(sB_full(_, _, stage)));
    copy(tma_sfa.with(full[stage]), sfa_slice.partition_S(gSFA),
         sfa_slice.partition_D(sSFA_full(_, _, stage)));
    copy(tma_sfb.with(full[stage]), sfb_slice.partition_S(gSFB),
         sfb_slice.partition_D(sSFB_full(_, _, stage)));
  };

  const int nKt = K / kBK;
  // 预置 empty：前 kStages 个 stage 的 empty 必须先「已就绪」，否则 producer
  // 的第一次 wait 会死锁（consumer 要等 TMA 才能 arrive，形成环）
  if (tid >= kProducerThreads) {
    for (int s = 0; s < kStages; ++s) CtaBarrier::arrive(&empty[s]);
  }
  if (tid < kProducerThreads) {
    // ---------------- producer warpgroup ----------------
    cutlass::arch::warpgroup_reg_dealloc<32>();  // 整 warpgroup 对齐执行
    if (tid == 0) {
      for (int kt = 0; kt < nKt; ++kt) {
        const int stage = kt % kStages;
        CtaBarrier::wait(&empty[stage], (kt / kStages) & 1);
        issue_tma(kt, stage);
      }
    }
    // producer 不参与 epilogue，直接结束（后续 barrier 均 consumer-only）
  } else {
    // ---------------- consumer warpgroups ----------------
    cutlass::arch::warpgroup_reg_alloc<224>();
    const int ctid = tid - kProducerThreads;

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(ctid);
    auto s2r_copy_a = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto s2r_copy_b = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto s2r_thr_a = s2r_copy_a.get_thread_slice(ctid);
    auto s2r_thr_b = s2r_copy_b.get_thread_slice(ctid);
    auto tCsA = s2r_thr_a.partition_S(as_position_independent_swizzle_tensor(sA_full));
    auto tCsB = s2r_thr_b.partition_S(as_position_independent_swizzle_tensor(sB_full));
    auto tCrA = thr_mma.partition_fragment_A(sA_full(_, _, Int<0>{}));
    auto tCrB = thr_mma.partition_fragment_B(sB_full(_, _, Int<0>{}));
    auto tCrA_view = s2r_thr_a.retile_D(tCrA);
    auto tCrB_view = s2r_thr_b.retile_D(tCrB);

    using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;
    auto tCrSFA =
        sf_partition::partition_fragment_sfa(sSFA_full(_, _, Int<0>{}), thr_mma);
    auto tCrSFB =
        sf_partition::partition_fragment_sfb(sSFB_full(_, _, Int<0>{}), thr_mma);
    auto copy_sfa = make_tiled_copy_impl(
        SmemCopyAtomSF{}, sf_partition::layout_sfa_tv(tiled_mma),
        make_shape(size<0>(tile_shape(tiled_mma)), size<2>(tile_shape(tiled_mma))));
    auto copy_sfb = make_tiled_copy_impl(
        SmemCopyAtomSF{}, sf_partition::layout_sfb_tv(tiled_mma),
        make_shape(size<1>(tile_shape(tiled_mma)), size<2>(tile_shape(tiled_mma))));
    auto thr_copy_sfa = copy_sfa.get_thread_slice(ctid);
    auto thr_copy_sfb = copy_sfb.get_thread_slice(ctid);
    auto tCsSFA =
        thr_copy_sfa.partition_S(as_position_independent_swizzle_tensor(sSFA_full));
    auto tCsSFB =
        thr_copy_sfb.partition_S(as_position_independent_swizzle_tensor(sSFB_full));
    auto tCrSFA_view = thr_copy_sfa.retile_D(tCrSFA);
    auto tCrSFB_view = thr_copy_sfb.retile_D(tCrSFB);

    auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<kBM>, Int<kBN>>{});
    clear(tCrC);
    auto cC = make_identity_tensor(Shape<Int<kBM>, Int<kBN>>{});
    auto tScC = thr_mma.partition_C(cC);
    auto tScC_rc = make_tensor(tScC.data(), convert_layout_acc_rowcol(tScC.layout()));
    constexpr int kCRows = decltype(size<0>(tScC_rc))::value;
    constexpr int kCCols = decltype(size<1>(tScC_rc))::value;

#pragma unroll 1
    for (int kt = 0; kt < nKt; ++kt) {
      const int stage = kt % kStages;
      const int phase = (kt / kStages) & 1;
      TmaBarrier::wait(&full[stage], phase);
      cutlass::arch::fence_view_async_shared();
      copy(s2r_copy_a, tCsA(_, _, _, stage), tCrA_view);
      copy(s2r_copy_b, tCsB(_, _, _, stage), tCrB_view);
      copy(tCsSFA(_, _, _, stage), tCrSFA_view);
      copy(tCsSFB(_, _, _, stage), tCrSFB_view);
      cute::gemm(tiled_mma, make_zip_tensor(tCrA, tCrSFA),
                 make_zip_tensor(tCrB, tCrSFB), tCrC);
      CtaBarrier::arrive(&empty[stage]);  // 只有 consumer arrive
    }

    {
      __syncthreads();
      auto tCrC_rc = make_tensor(tCrC.data(), convert_layout_acc_rowcol(tCrC.layout()));
#pragma unroll
      for (int r = 0; r < kCRows; ++r) {
        const int m = m0 + get<0>(tScC_rc(r, 0));
        const float sa_r = kLevel1A ? ((m < M) ? sa[m] : 0.0f) : 1.0f;
#pragma unroll
        for (int c = 0; c < kCCols; ++c) {
          const int n = n0 + get<1>(tScC_rc(r, c));
          const float sb_c = kLevel1B ? ((n < N) ? sb[n] : 0.0f) : 1.0f;
          tCrC_rc(r, c) = tCrC_rc(r, c) * sa_r * sb_c;
        }
      }
      auto tCrCh = convert_type<ElementO>(tCrC);
      auto r2s_copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, ElementO>{},
                                        tiled_mma);
      auto r2s_thr = r2s_copy.get_thread_slice(ctid);
      auto sC = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(shm)),
                            SmemLayoutO{});
      auto tCrCh_src = r2s_thr.retile_S(tCrCh);
      auto tCsC_dst = r2s_thr.partition_D(sC);
      copy(r2s_copy, tCrCh_src, tCsC_dst);
      cutlass::arch::fence_view_async_shared();
      cutlass::arch::NamedBarrier::arrive_and_wait(kNumThreads, 0);
      auto gC = local_tile(mC, Shape<Int<kBM>, Int<kBN>>{},
                           make_coord(blockIdx.y, blockIdx.x));
      if (ctid == 0) copy(tma_c, c_slice.partition_S(sC), c_slice.partition_D(gC));
      tma_store_arrive();
      tma_store_wait<0>();
    }
  }
#endif  // __CUDA_ARCH__ >= 900
}

// =============================================================================
// Phase 10.7: launch wrapper + C++ API（BF16 in -> 在线 NVFP4 量化 -> GEMM）
// =============================================================================
// 4-bit 数据的 TMA 描述符：cute 用 CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B 直接
// 搬 4-bit（元素类型仍是 e2m1，stride 以 4-bit 为单位），smem 侧则是打包的
// SW32 原子；SF 用 uint16_t 作内部元素类型（CUTLASS collective 同款做法）.
template <bool kLevel1A, bool kLevel1B, bool kWS, typename Traits = Fp4GemmTraits<>>
void fp4_gemm_tma_fwd(const cutlass::float_e2m1_t *A4,
                      const cutlass::float_e2m1_t *B4T,
                      const cutlass::float_ue4m3_t *sfa,
                      const cutlass::float_ue4m3_t *sfb, const float *sa,
                      const float *sb, cutlass::bfloat16_t *C, int M, int N,
                      int K, cudaStream_t stream) {
  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutAtomA = typename Traits::SmemLayoutAtomA;
  using SmemLayoutAtomB = typename Traits::SmemLayoutAtomB;
  using SmemLayoutAtomSFA = typename Traits::SmemLayoutAtomSFA;
  using SmemLayoutAtomSFB = typename Traits::SmemLayoutAtomSFB;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  constexpr int kBM = Traits::kBM;
  constexpr int kBN = Traits::kBN;
  constexpr int kBK = Traits::kBK;
  constexpr int kSmemBytes = Traits::kSmemBytes;

  // TN 布局：C[M,N] = A4[M,K] x B4T[N,K]^T（4-bit 元素，行 stride = K 个 4-bit）
  auto mA = make_tensor(recast_ptr<cutlass::float_e2m1_t>(A4), make_shape(M, K),
                        make_stride(K, _1{}));
  auto mB = make_tensor(recast_ptr<cutlass::float_e2m1_t>(B4T), make_shape(N, K),
                        make_stride(K, _1{}));
  auto mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, _1{}));
  // SF 张量：canonical (MN, K) 布局（行数按 128 向上取整，缓冲区同步补齐）
  auto mSFA = make_tensor(recast_ptr<cutlass::float_ue4m3_t>(sfa),
                          sf_gmem_layout(M, K));
  auto mSFB = make_tensor(recast_ptr<cutlass::float_ue4m3_t>(sfb),
                          sf_gmem_layout(N, K));

  // TMA 的 SLayout 必须与 CTA_Tiler 顶层 size 相等（= 单 stage 全 tile）：
  // A/B 取 3-mode 布局的 stage-0 切面（等价单 stage 全 tile），SF 的 512B
  // 原子本身已按 (MN,K) 铺满（含 stride-0 广播维），顶层 size 天然相等
  auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, mA, SmemLayoutA{}(_, _, Int<0>{}),
                             Shape<Int<kBM>, Int<kBK>>{}, _1{});
  auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, mB, SmemLayoutB{}(_, _, Int<0>{}),
                             Shape<Int<kBN>, Int<kBK>>{}, _1{});
  auto tma_sfa = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFA,
                                         SmemLayoutAtomSFA{},
                                         Shape<Int<kBM>, Int<kBK>>{}, _1{});
  auto tma_sfb = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFB,
                                         SmemLayoutAtomSFB{},
                                         Shape<Int<kBN>, Int<kBK>>{}, _1{});
  auto tma_c = make_tma_copy(SM90_TMA_STORE{}, mC, SmemLayoutO{},
                             Shape<Int<kBM>, Int<kBN>>{}, _1{});

  dim3 grid((N + kBN - 1) / kBN, (M + kBM - 1) / kBM);
  if constexpr (kWS) {
    auto kernel = fp4_gemm_tma_ws_kernel<Traits, decltype(tma_a), decltype(tma_b),
                                         decltype(tma_sfa), decltype(tma_sfb),
                                         decltype(tma_c), kLevel1A, kLevel1B>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSmemBytes);
    kernel<<<grid, Traits::kNumThreads + 128, kSmemBytes, stream>>>(
        tma_a, tma_b, tma_sfa, tma_sfb, tma_c, sa, sb, M, N, K);
  } else {
    auto kernel = fp4_gemm_tma_kernel<Traits, decltype(tma_a), decltype(tma_b),
                                      decltype(tma_sfa), decltype(tma_sfb),
                                      decltype(tma_c), kLevel1A, kLevel1B>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSmemBytes);
    kernel<<<grid, Traits::kNumThreads, kSmemBytes, stream>>>(
        tma_a, tma_b, tma_sfa, tma_sfb, tma_c, sa, sb, M, N, K);
  }
}

// 一次性 workspace（调用方分配/复用）：
//   [A4 M*K/2 | B4T N*K/2 | SFA pad(M,128) | SFB pad(N,128) | sa float M | sb float N]
struct Fp4GemmWorkspace {
  void *buf = nullptr;
  size_t size = 0;
};

inline size_t fp4_gemm_align256(size_t x) { return (x + 255) & ~(size_t)255; }

inline size_t fp4_gemm_workspace_size(int M, int N, int K) {
  return fp4_gemm_align256((size_t)M * (K / 2)) +
         fp4_gemm_align256((size_t)N * (K / 2)) +
         fp4_gemm_align256(sf_buffer_bytes(M, K)) +
         fp4_gemm_align256(sf_buffer_bytes(N, K)) +
         fp4_gemm_align256((size_t)M * sizeof(float)) +
         fp4_gemm_align256((size_t)N * sizeof(float));
}

// BF16 in -> 在线 NVFP4 量化 GEMM -> BF16 out，单 stream 语义。
// 同步约定：与 cuBLAS 一致，入队后由调用方在 stream 上同步。
// 约束：K%64==0, N%8==0(TMA 对齐), M 任意。
// mode 决定 level-1 的有无与作用侧；use_ws 走 warp-specialized 变体。
inline void fp4_gemm_bf16(const __nv_bfloat16 *A, const __nv_bfloat16 *B,
                          __nv_bfloat16 *C, int M, int N, int K,
                          Fp4GemmScaleMode mode, Fp4GemmWorkspace &ws,
                          cudaStream_t stream, bool use_ws = false) {
  if (K % 64 != 0 || N % 8 != 0) {
    fprintf(stderr,
            "fp4_gemm_bf16: require K%%64==0 (got K=%d), N%%8==0 (got N=%d)\n",
            K, N);
    return;
  }
  const bool level1_a =
      mode == Fp4GemmScaleMode::kLevel1A || mode == Fp4GemmScaleMode::kTwoLevel;
  const bool level1_b =
      mode == Fp4GemmScaleMode::kLevel1B || mode == Fp4GemmScaleMode::kTwoLevel;

  uint8_t *p = static_cast<uint8_t *>(ws.buf);
  cutlass::float_e2m1_t *a4 = reinterpret_cast<cutlass::float_e2m1_t *>(p);
  p += fp4_gemm_align256((size_t)M * (K / 2));
  cutlass::float_e2m1_t *b4t = reinterpret_cast<cutlass::float_e2m1_t *>(p);
  p += fp4_gemm_align256((size_t)N * (K / 2));
  cutlass::float_ue4m3_t *sfa = reinterpret_cast<cutlass::float_ue4m3_t *>(p);
  p += fp4_gemm_align256(sf_buffer_bytes(M, K));
  cutlass::float_ue4m3_t *sfb = reinterpret_cast<cutlass::float_ue4m3_t *>(p);
  p += fp4_gemm_align256(sf_buffer_bytes(N, K));
  float *sa = reinterpret_cast<float *>(p);
  p += fp4_gemm_align256((size_t)M * sizeof(float));
  float *sb = reinterpret_cast<float *>(p);

  if (level1_a) row_amax_fp4_kernel<<<(M + 7) / 8, 256, 0, stream>>>(A, sa, M, K);
  {  // 一线程一 (行, 16-K 组)：总工作量 = 补齐行数 x (K/16)
    const int mpad = (M + kRowBlock - 1) / kRowBlock * kRowBlock;
    const int total = mpad * (K / kSFVec);
    const int blocks = (total + 255) / 256;
    if (level1_a)
      quantize_a_fp4_kernel<true><<<blocks, 256, 0, stream>>>(A, a4, sfa, sa, M, K);
    else
      quantize_a_fp4_kernel<false><<<blocks, 256, 0, stream>>>(A, a4, sfa, sa, M, K);
  }
  fp4_gemm_quantize_b(B, b4t, sfb, sb, N, K, level1_b, stream);

#define FP4_GEMM_LAUNCH(L1A, L1B)                                        \
  do {                                                                   \
    if (use_ws)                                                          \
      fp4_gemm_tma_fwd<L1A, L1B, true>(a4, b4t, sfa, sfb, sa, sb,        \
                                       reinterpret_cast<cutlass::bfloat16_t *>(C), \
                                       M, N, K, stream);                 \
    else                                                                 \
      fp4_gemm_tma_fwd<L1A, L1B, false>(a4, b4t, sfa, sfb, sa, sb,       \
                                        reinterpret_cast<cutlass::bfloat16_t *>(C), \
                                        M, N, K, stream);                \
  } while (0)
  if (level1_a && level1_b)
    FP4_GEMM_LAUNCH(true, true);
  else if (level1_a)
    FP4_GEMM_LAUNCH(true, false);
  else if (level1_b)
    FP4_GEMM_LAUNCH(false, true);
  else
    FP4_GEMM_LAUNCH(false, false);
#undef FP4_GEMM_LAUNCH
}

// =============================================================================
// Phase 10.8: 权重 B 离线量化（推理部署的真实形态）
// =============================================================================
// 与 fp8 同理：B（权重）离线量化一次，(B4T, sfb) 常驻显存；每步只在线量化
// A（激活）。注意 fp4 的 B 离线产物比 fp8 多一份 SFB（N x K/16 个 ue4m3）。
struct Fp4GemmActivation {
  void *buf = nullptr;
  size_t size = 0;
};

// [A4 M*K/2 | SFA pad(M,128) | sa float M]
inline size_t fp4_gemm_activation_size(int M, int K) {
  return fp4_gemm_align256((size_t)M * (K / 2)) +
         fp4_gemm_align256(sf_buffer_bytes(M, K)) +
         fp4_gemm_align256((size_t)M * sizeof(float));
}

// B 已离线量化：每步只量化 A -> 同一个 NVFP4 GEMM 主 kernel -> BF16 输出。
// 契约：B4T/SFB 为 fp4_gemm_quantize_b 产出的 (N,K)/(N,K) canonical 布局，
// level1_b 必须与离线量化时一致，否则整 tile 被错误 scale 静默错值。
template <bool kLevel1A, bool kLevel1B, bool kWS = false,
          typename Traits = Fp4GemmTraits<>>
inline void fp4_gemm_bf16_b_offline(const __nv_bfloat16 *A,
                                    const cutlass::float_e2m1_t *B4T,
                                    const cutlass::float_ue4m3_t *sfb,
                                    const float *sb, __nv_bfloat16 *C, int M,
                                    int N, int K, Fp4GemmActivation &act,
                                    cudaStream_t stream) {
  if (K % 64 != 0 || N % 8 != 0) {
    fprintf(stderr,
            "fp4_gemm_bf16_b_offline: require K%%64==0 (got K=%d), N%%8==0 "
            "(got N=%d)\n",
            K, N);
    return;
  }
  uint8_t *p = static_cast<uint8_t *>(act.buf);
  cutlass::float_e2m1_t *a4 = reinterpret_cast<cutlass::float_e2m1_t *>(p);
  p += fp4_gemm_align256((size_t)M * (K / 2));
  cutlass::float_ue4m3_t *sfa = reinterpret_cast<cutlass::float_ue4m3_t *>(p);
  p += fp4_gemm_align256(sf_buffer_bytes(M, K));
  float *sa = reinterpret_cast<float *>(p);

  if constexpr (kLevel1A)
    row_amax_fp4_kernel<<<(M + 7) / 8, 256, 0, stream>>>(A, sa, M, K);
  {
    const int mpad = (M + kRowBlock - 1) / kRowBlock * kRowBlock;
    const int total = mpad * (K / kSFVec);
    const int blocks = (total + 255) / 256;
    if constexpr (kLevel1A)
      quantize_a_fp4_kernel<true><<<blocks, 256, 0, stream>>>(A, a4, sfa, sa, M, K);
    else
      quantize_a_fp4_kernel<false><<<blocks, 256, 0, stream>>>(A, a4, sfa, sa, M, K);
  }
  fp4_gemm_tma_fwd<kLevel1A, kLevel1B, kWS, Traits>(
      a4, B4T, sfa, sfb, sa, sb, reinterpret_cast<cutlass::bfloat16_t *>(C), M,
      N, K, stream);
}

}  // namespace fp4_gemm
#endif  // NOTES_V2_ENABLE_CUTE && NOTES_V2_ENABLE_TMA_MMA_WS && NOTES_V2_ENABLE_SM120_FP4
