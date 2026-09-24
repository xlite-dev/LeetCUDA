#pragma once
#include "common.cuh"
// fp8_gemm.cuh: Phase 9 FP8 GEMM — e4m3 动态量化教学案例
// =============================================================================
// 全链路：BF16 输入 -> FP8 量化前处理(per-row / per-block) -> CuTe FP8 GEMM
//         (SM89_16x8x32 mma.sync + TMA + mbarrier 多级流水，epilogue 在线反量化)
//         -> BF16 输出.封装为 fp8_gemm_bf16 一个 C++ API, 与 cuBLAS BF16
//         GEMM 对比精度(max err)与性能(TFLOPS).
//
// 参考：ffpa-attn csrc/cuffpa/cute/fp8(sm_120 persist-D 链路，commit 861d75e)
//   - MMA atom: SM89_16x8x32_F32E4M3E4M3F32_TN (mma.sync, sm_89+ 可用)
//   - smem swizzle: GMMA::Layout_K_SW128_Atom (1B 元素，BK=128 行恰好 128B)
//   - mbarrier: ClusterTransactionBarrier(full) + ClusterBarrier(empty)
//   - scale 折叠代数：C = (A8 · B8^T) · δa · δb (persist-D QK 侧
//     s_dequant = qs*ks 的 GEMM 版；per-block 每 128 行/列一个 δ，逐元素查表)
//   - epilogue: STSM r2s 复用已释放的 A/B stage smem -> TMA store
//
// tile 参数化(Fp8GemmTraits<BM,BN,BK,stages>): BM/BN 可扫 {64,128,256}，
//   BK 固定 128（SW128 行宽），stages >= 2（O staging 复用整个 A+B 区）。
//   默认 128x256x128/s2 为本卡扫描最快档；BM 只改 warp 数，BN 才是 acc 量来源。
//
// 量化粒度(Fp8GemmScaleMode 4 组合):
//   A: per-row(scale[M], 精度最优) / per-block(scale[M/128], ffpa Q/K 同构)
//   B: per-col(scale[N], B^T 行主序下即 per-row of B8T) / per-block(scale[N/128])
//
// 约束（教学约定，由 API 层检查）：
//   K % 16 == 0  (A8/B8T 行 16B 对齐，TMA 硬性要求；也是量化 Vec8 的超集)
//   N % 8  == 0  (C 行 N*2B 16B 对齐，TMA store 硬性要求)
//   M 任意       (TMA load OOB 行零填充 / store OOB 行裁剪)
//   K 尾 tile: TMA load 越界元素自动填 0, 对 GEMM 累加无贡献，无需尾分支
//
// 线格式（标准 bench 表）：| Kernel | Max Err | TFLOPS/cuBLAS |
// 性能参考(PRO 5000 sm_120a, 4096^3): 见 notes-v2.cu --bench --mnk 4096,4096,4096

#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/device_kernel.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

namespace fp8_gemm {
using namespace cute;

// =============================================================================
// Phase 9.1: 量化数学
// =============================================================================
// 量化算子（对称，e4m3 满量程 R=448）：
//   x̂ = clamp_{[-448,448]}( round(x / δ) ),  δ = amax(B) / 448
//   amax=0 时 δ=0, inv_δ=0 （全零块量化为全零，避免 0/0）
// 反量化折叠(GEMM 版核心恒等式):
//   C = A·B^T = (δa·Â)·(δb·B̂)^T = δa·δb·(Â·B̂^T)
//   整数域 MMA 结果 epilogue 每元素乘回 δa·δb — 除量化本身外无额外数学
//   误差源（与 persist-D 的 s_dequant = qs*ks 折叠同理）。
//   per-row x per-col: δa 依行 m 查表 sa[m], δb 依列 n 查表 sb[n]
//   per-block: 每 128 行/列一个 scale, epilogue 按 m/128、n/128 查表(与 tile 解耦)
constexpr float kE4M3Max = 448.0f;   // e4m3 最大有限值（饱和上界）
constexpr int kScaleBlock = 128;     // per-block 量化粒度（固定 128）

enum class Fp8GemmScaleMode {
  kPerRowPerCol,      // A per-row x B per-col（精度最优，scale 数 M+N）
  kPerRowPerBlock,    // A per-row x B per-block(scale 数 M + N/128)
  kPerBlockPerCol,    // A per-block x B per-col(scale 数 M/128 + N)
  kPerBlockPerBlock,  // A per-block x B per-block(attention Q/K 同构，M/128+N/128)
};

union Vec8BF16 {  // 8 x bf16 = 16B, 量化 kernel 的向量化 IO 单元
  uint4 raw;
  __nv_bfloat16 elem[8];
};
static_assert(sizeof(Vec8BF16) == 16, "Vec8BF16 must be 128-bit");

// f32 -> e4m3 两条打包（与 ffpa fp8_pscale.cuh 的 cvt.rn.satfinite.e4m3x2
// .f32 同指令): satfinite 保证 x*inv_δ 舍入越界时饱和到 ±448 而非 NaN.
__device__ __forceinline__ unsigned short cvt_f2_to_e4m3x2(float lo, float hi) {
  return __nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE,
                                  __NV_E4M3);
}

// =============================================================================
// Phase 9.2: 量化前处理 kernel(A 行主序保形 / B 转置到 B8T[N,K])
// =============================================================================
// A[M,K] bf16 -> A8[M,K] e4m3 行主序（与输入同布局，TMA 直接按 (BM,BK） tile).
//
// per-row 版：warp-per-row, 一 warp 负责一整行。amax 经 shfl_xor 蝶形归约，
// 量化与写回同行完成。K 是运行时参数无法在寄存器缓存整行，gmem 读两遍
// （第二遍命中 L1/L2）；ffpa 的 per-block kernel 依赖编译期 kD 才能一遍读。
__global__ void quantize_a_perrow_kernel(
    const __nv_bfloat16 *__restrict__ X, cutlass::float_e4m3_t *__restrict__ Y,
    float *__restrict__ scale, int M, int K) {
  constexpr int kVec = 8;  // K % 16 == 0 保证 chunk 对齐
  const int row = blockIdx.x * 8 + threadIdx.x / 32;  // 8 warps/block
  if (row >= M) return;
  const int lane = threadIdx.x % 32;
  const __nv_bfloat16 *xr = X + (size_t)row * K;

  float amax = 0.0f;  // pass 1: 行 amax
  for (int c = lane * kVec; c < K; c += 32 * kVec) {
    Vec8BF16 v = *reinterpret_cast<const Vec8BF16 *>(xr + c);
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      amax = fmaxf(amax, fabsf(__bfloat162float(v.elem[e])));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)  // warp 内蝶形归约
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));

  const float s = amax / kE4M3Max;
  const float inv_s = (amax == 0.0f) ? 0.0f : kE4M3Max / amax;
  if (lane == 0) scale[row] = s;

  cutlass::float_e4m3_t *yr = Y + (size_t)row * K;  // pass 2: 量化写回
  for (int c = lane * kVec; c < K; c += 32 * kVec) {
    Vec8BF16 v = *reinterpret_cast<const Vec8BF16 *>(xr + c);
    uint2 out;
    unsigned short *o2 = reinterpret_cast<unsigned short *>(&out);
#pragma unroll
    for (int e = 0; e < kVec; e += 2)
      o2[e / 2] = cvt_f2_to_e4m3x2(__bfloat162float(v.elem[e]) * inv_s,
                                   __bfloat162float(v.elem[e + 1]) * inv_s);
    *reinterpret_cast<uint2 *>(yr + c) = out;  // 8 x e4m3 = 8B
  }
}

// A per-block 版：一个 block 负责连续 128 行(ffpa quantize_fp8_kernel 同构，
// 但 K 运行时化 -> 两遍 gmem 读，第二遍命中 L1/L2).block 级 amax =
// 128 行全体元素的最大绝对值，warp 归约 -> smem -> warp0 二次归约。
__global__ void quantize_a_perblock_kernel(
    const __nv_bfloat16 *__restrict__ X, cutlass::float_e4m3_t *__restrict__ Y,
    float *__restrict__ scale, int M, int K) {
  constexpr int kVec = 8;
  constexpr int kThreads = 256;
  constexpr int kWarps = kThreads / 32;
  const int row0 = blockIdx.x * kScaleBlock;
  const int tid = threadIdx.x;
  const int warp = tid / 32, lane = tid % 32;
  const int dv = K / kVec;  // 每行 Vec8 chunk 数

  float amax = 0.0f;  // pass 1: 128 行块 amax
  for (int i = tid; i < kScaleBlock * dv; i += kThreads) {
    const int r = i / dv, c = (i % dv) * kVec;
    const long row = row0 + r;
    if (row >= M) continue;
    Vec8BF16 v = *reinterpret_cast<const Vec8BF16 *>(X + row * K + c);
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      amax = fmaxf(amax, fabsf(__bfloat162float(v.elem[e])));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
  __shared__ float warp_max[kWarps];
  if (lane == 0) warp_max[warp] = amax;
  __syncthreads();
  if (warp == 0) {
    amax = (lane < kWarps) ? warp_max[lane] : 0.0f;
#pragma unroll
    for (int off = kWarps / 2; off > 0; off >>= 1)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    if (lane == 0) warp_max[0] = amax;
  }
  __syncthreads();
  amax = warp_max[0];

  const float inv_s = (amax == 0.0f) ? 0.0f : kE4M3Max / amax;
  if (tid == 0) scale[blockIdx.x] = amax / kE4M3Max;

  for (int i = tid; i < kScaleBlock * dv; i += kThreads) {  // pass 2: 量化
    const int r = i / dv, c = (i % dv) * kVec;
    const long row = row0 + r;
    if (row >= M) continue;
    Vec8BF16 v = *reinterpret_cast<const Vec8BF16 *>(X + row * K + c);
    uint2 out;
    unsigned short *o2 = reinterpret_cast<unsigned short *>(&out);
#pragma unroll
    for (int e = 0; e < kVec; e += 2)
      o2[e / 2] = cvt_f2_to_e4m3x2(__bfloat162float(v.elem[e]) * inv_s,
                                   __bfloat162float(v.elem[e + 1]) * inv_s);
    *reinterpret_cast<uint2 *>(Y + row * K + c) = out;
  }
}

// B 转置量化：B[K,N] bf16 -> B8T[N,K] e4m3(B^T 行主序，K 内维连续，TMA
// 直接按 (BN,BK) tile, 与 ffpa V^T 预转置同构).一个 block 负责连续 128
// 个 n（即 B8T 的 128 行），K 方向按 64 行 chunk 循环两遍：
//   pass 1: coalesced 读 B 的 [64,128] tile 进 smem, 逐列(n)归约 amax,
//           跨全部 K chunk 累积到 samax[128]
//   pass 2: 重读 tile -> 量化散写到输出朝向 tile_t[n][k] -> warp 串行 n 行，
//           32 lanes 写 32 个连续 k 字节(coalesced 32B store)
// 为什么不直接写 B8T? B 的列方向 stride=N, 直接按列读/写都不 coalesced;
// smem staging 让 gmem 读(B 行)与 gmem 写(B8T 行)都落在连续维度上。
// 注意与 ffpa quantize_fp8_vt_kernel 的差异：这里 TMA load OOB 自动零填，
// B8T 无需 pad 列(K 尾部由 store guard 裁剪), smem 也不需要 D-chunk 分块。
// kPerCol: scale[N](B 每列一个，B8T 行主序下即每行一个); 否则 scale[N/128].
template <bool kPerCol>
__global__ void quantize_bt_kernel(const __nv_bfloat16 *__restrict__ B,
                                   cutlass::float_e4m3_t *__restrict__ BT,
                                   float *__restrict__ scale, int K, int N) {
  constexpr int kBNt = 128;               // n 方向 tile(= B8T 行数)
  constexpr int kBKt = 64;                // k 方向 chunk(= B 行数)
  constexpr int kVec = 8;
  constexpr int kChunksPerRow = kBNt / kVec;  // 16
  constexpr int kChunks = kBKt * kChunksPerRow;
  constexpr int kWarps = 256 / 32;
  const int n0 = blockIdx.x * kBNt;
  const int tid = threadIdx.x;
  const int warp = tid / 32, lane = tid % 32;

  __shared__ __align__(16) __nv_bfloat16 tile[kBKt][kBNt];  // 源朝向 16KB
  __shared__ __nv_fp8_e4m3 tile_t[kBNt][kBKt];  // 输出朝向 8KB（平凡类型，
  // 避免 smem 变量动态初始化告警；与 cutlass::float_e4m3_t 逐位兼容)
  __shared__ float samax[kBNt];

  auto load_tile = [&](int k0) {  // B 行 coalesced 读入 smem（越界补零）
    for (int i = tid; i < kChunks; i += 256) {
      const int r = i / kChunksPerRow, c8 = (i % kChunksPerRow) * kVec;
      const int k = k0 + r, n = n0 + c8;
      Vec8BF16 v;
      v.raw = make_uint4(0, 0, 0, 0);
      if (k < K && n < N)
        v = *reinterpret_cast<const Vec8BF16 *>(B + (size_t)k * N + n);
      *reinterpret_cast<uint4 *>(&tile[r][c8]) = v.raw;
    }
  };

  for (int n = tid; n < kBNt; n += 256) samax[n] = 0.0f;  // tid<128 生效
  __syncthreads();
  for (int k0 = 0; k0 < K; k0 += kBKt) {  // pass 1: 逐列 amax
    load_tile(k0);
    __syncthreads();
    for (int n = tid; n < kBNt; n += 256) {
      float a = samax[n];
      for (int r = 0; r < kBKt; ++r)
        a = fmaxf(a, fabsf(__bfloat162float(tile[r][n])));
      samax[n] = a;
    }
    __syncthreads();
  }

  // scale 写出 + samax 原地改存 inv_s(pass 2 直接查表)
  if constexpr (kPerCol) {
    for (int n = tid; n < kBNt; n += 256) {
      if (n0 + n >= N) continue;
      const float am = samax[n];
      scale[n0 + n] = am / kE4M3Max;
      samax[n] = (am == 0.0f) ? 0.0f : kE4M3Max / am;
    }
  } else {
    __syncthreads();
    if (tid == 0) {  // block 级 amax 归约(128 个列 amax -> 单值)
      float m = 0.0f;
      for (int n = 0; n < kBNt; ++n) m = fmaxf(m, samax[n]);
      if (n0 < N) scale[n0 / kScaleBlock] = m / kE4M3Max;
      samax[0] = (m == 0.0f) ? 0.0f : kE4M3Max / m;
    }
  }
  __syncthreads();

  for (int k0 = 0; k0 < K; k0 += kBKt) {  // pass 2: 量化 + 转置写
    load_tile(k0);
    __syncthreads();
    for (int i = tid; i < kChunks; i += 256) {  // 散写 tile_t[n][r]
      const int r = i / kChunksPerRow, c8 = (i % kChunksPerRow) * kVec;
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        const int n = c8 + e;
        const float inv_s = kPerCol ? samax[n] : samax[0];
        tile_t[n][r] = __nv_fp8_e4m3(__bfloat162float(tile[r][n]) * inv_s);
      }
    }
    __syncthreads();
    for (int n = warp; n < kBNt; n += kWarps) {  // 32B coalesced store
      if (n0 + n >= N) continue;
      cutlass::float_e4m3_t *row = BT + (size_t)(n0 + n) * K + k0;
#pragma unroll
      for (int half = 0; half < kBKt / 32; ++half) {
        const int kk = half * 32 + lane;
        if (k0 + kk < K)
          row[kk] = *reinterpret_cast<cutlass::float_e4m3_t *>(&tile_t[n][kk]);
      }
    }
    __syncthreads();  // tile/tile_t 下一 chunk 可覆写
  }
}

// B 侧量化入口：全链路(fp8_gemm_bf16)与权重离线量化(Phase 9.8)共用同一实现。
inline void fp8_gemm_quantize_b(const __nv_bfloat16 *B,
                                cutlass::float_e4m3_t *b8t, float *sb, int N,
                                int K, bool per_col_b, cudaStream_t stream) {
  if (per_col_b)
    quantize_bt_kernel<true><<<(N + 127) / 128, 256, 0, stream>>>(B, b8t, sb,
                                                                  K, N);
  else
    quantize_bt_kernel<false><<<(N + 127) / 128, 256, 0, stream>>>(B, b8t, sb,
                                                                   K, N);
}

// =============================================================================
// Phase 9.3: CuTe 工具（最小集，与 flash_attn.cuh fa_cute 同源）
// =============================================================================
// acc (MMA=4, MMA_M, MMA_N) -> 行列二维视图：epilogue 按行列坐标查 sa/sb.
template <typename Layout>
CUTE_DEVICE auto convert_layout_acc_rowcol(Layout acc_layout) {
  auto divided = logical_divide(acc_layout, Shape<_2>{});
  return make_layout(
      make_layout(get<0, 1>(divided), get<1>(divided)),
      make_layout(get<0, 0>(divided), get<2>(divided)));
}

// f32 acc -> bf16, 寄存器内批量转换(NumericArrayConverter), 零 smem 往返。
template <typename To, typename Engine, typename Layout>
CUTE_DEVICE auto convert_type(Tensor<Engine, Layout> const &tensor) {
  using From = typename Engine::value_type;
  constexpr int kElements = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To, From, kElements> convert;
  auto fragment = convert(
      *reinterpret_cast<cutlass::Array<From, kElements> const *>(tensor.data()));
  return make_tensor(make_rmem_ptr<To>(&fragment), tensor.layout());
}

// gemm_ss: A/B 都从 smem ldmatrix 到寄存器再 mma, ldmatrix（下一 k-slice）
// 与 mma（当前 k-slice） 软件流水重叠.出处：flash-attention utils.h:166.
template <typename TensorC, typename TensorA, typename TensorB,
          typename TensorSA, typename TensorSB, typename TiledMma,
          typename TiledCopyA, typename TiledCopyB, typename ThreadCopyA,
          typename ThreadCopyB>
CUTE_DEVICE void gemm_ss(TensorC &acc, TensorA &fragment_a,
                         TensorB &fragment_b, TensorSA const &shared_a,
                         TensorSB const &shared_b, TiledMma tiled_mma,
                         TiledCopyA tiled_copy_a, TiledCopyB tiled_copy_b,
                         ThreadCopyA thread_copy_a, ThreadCopyB thread_copy_b) {
  auto copy_view_a = thread_copy_a.retile_D(fragment_a);
  auto copy_view_b = thread_copy_b.retile_D(fragment_b);
  copy(tiled_copy_a, shared_a(_, _, _0{}), copy_view_a(_, _, _0{}));
  copy(tiled_copy_b, shared_b(_, _, _0{}), copy_view_b(_, _, _0{}));
#pragma unroll
  for (int tile_k = 0; tile_k < size<2>(fragment_a); ++tile_k) {
    if (tile_k + 1 < size<2>(fragment_a)) {
      copy(tiled_copy_a, shared_a(_, _, tile_k + 1),
           copy_view_a(_, _, tile_k + 1));
      copy(tiled_copy_b, shared_b(_, _, tile_k + 1),
           copy_view_b(_, _, tile_k + 1));
    }
    gemm(tiled_mma, fragment_a(_, _, tile_k), fragment_b(_, _, tile_k), acc);
  }
}

// =============================================================================
// Phase 9.4: FP8 GEMM Traits(SM89 fp8 atom + SW128 smem + TiledMma M8N1)
// =============================================================================
// 三个可调参数：tile BM/BN、K-tile BK、流水级数 kStages（默认 128/256/128/2）。
// BK 被 SW128 atom 钉死：fp8 是 1B 元素，128B 行 = 128 个元素，一个 swizzle
// 周期恰好铺满整行（同样的 atom 换成 bf16 就只有 64 个元素）。
// BM 决定 warp 数——M8N1 把全部 warp 沿 M 堆叠，kNumWarps = BM/16；BN 决定
// 每线程 acc 量 = 4 x BN/8 = BN/2 个 f32（4 f32/MMA x BN/8 个 N 重复），
// 是寄存器压力的唯一来源（与 BM 无关）。
// smem = kStages*(BM+BN)*BK 字节，超 48KB 默认上限要 opt-in（本卡 99KB）。
// O staging（bf16 BM x BN）复用 K 循环结束后的整个 A+B 区：K 循环一结束所有
// stage 都空闲，几何约束 BM*BN*2 <= kStages*(BM+BN)*BK（默认档 64KB vs 96KB，
// 128x128/s3 恰好取等 32KB）。fp8 单个 A stage 只 16KB，放不下同几何的 C，
// 与 fp16「复用一个 stage」的老经验不同(Review 修正).
template <int kBM_ = 128, int kBN_ = 256, int kBK_ = 128, int kStages_ = 2>
struct Fp8GemmTraits {
  static_assert(kBK_ == 128, "SW128 atom requires 128B rows (1B elems)");
  static_assert(kBM_ % 16 == 0, "BM multiples of MMA_M (M8N1 warps along M)");
  static_assert(kBN_ % 64 == 0, "BN must cover a full SW128 bf16 atom (64)");
  static_assert(kStages_ >= 2, "pipeline + O staging reuse need >= 2 stages");
  static constexpr int kBM = kBM_;
  static constexpr int kBN = kBN_;
  static constexpr int kBK = kBK_;
  static constexpr int kStages = kStages_;
  static constexpr int kNumWarps = kBM / 16;      // M8N1: kBM/16 warps 沿 M
  static constexpr int kNumThreads = kNumWarps * 32;

  using Element = cutlass::float_e4m3_t;
  using ElementO = cutlass::bfloat16_t;
  using SmemAtom = GMMA::Layout_K_SW128_Atom<Element>;  // fp8 1B 行 128B
  using SmemLayoutA =
      decltype(tile_to_shape(SmemAtom{}, Shape<Int<kBM>, Int<kBK>>{}));
  using SmemLayoutB =
      decltype(tile_to_shape(SmemAtom{}, Shape<Int<kBN>, Int<kBK>>{}));
  using SmemAtomO = GMMA::Layout_K_SW128_Atom<ElementO>;  // bf16 行 128B/64 元素
  using SmemLayoutO =
      decltype(tile_to_shape(SmemAtomO{}, Shape<Int<kBM>, Int<kBN>>{}));
  static constexpr int kSmemBytes =
      kStages * (cosize(SmemLayoutA{}) + cosize(SmemLayoutB{})) *
      sizeof(Element);  // 128/256/s2 -> 2*48KB = 96KB
  static_assert(cosize(SmemLayoutO{}) * sizeof(ElementO) <= kSmemBytes,
                "O staging (bf16 BM x BN) must fit the freed A+B stages");

  using MmaAtom = MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>;
  using TiledMma = decltype(make_tiled_mma(
      MmaAtom{}, Layout<Shape<Int<kNumWarps>, _1, _1>>{},
      Tile<Int<kBM>, Int<kBN>, _32>{}));
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
};

// =============================================================================
// Phase 9.5: 主 kernel（非 WS） — 全员 MMA + tid0 内联发 TMA
// =============================================================================
// 执行协议(ffpa_attn.cuh ch26c non-WS 同构):
//   full[stage] (TmaBarrier, init=1): tid0 arrive_and_expect_tx + TMA,
//     消费者 wait 后读；TMA 写完自动翻转
//   empty[stage] (CtaBarrier, init=256): 每个消费线程 arrive 一次，
//     凑满后 tid0 才能覆写该 stage
//   phase = (kt / kStages) & 1; 预取 kt+kStages 在本 tile mma 完成之后
//     发出(s_next == stage 时 wait 门控在本轮 arrive 上，提前发会死锁)
// 尾部：M/N 方向 TMA store OOB 自动裁剪；K 尾 tile TMA load OOB 零填充
//   （零元素对 MMA 无贡献），epilogue scale 读 gmem 时按 m<M / n<N guard.
template <typename Traits, typename TmaA, typename TmaB, typename TmaC,
          bool kPerRowA, bool kPerColB>
__global__ void __launch_bounds__(Traits::kNumThreads, 1)
    fp8_gemm_tma_kernel(CUTLASS_GRID_CONSTANT TmaA const tma_a,
                        CUTLASS_GRID_CONSTANT TmaB const tma_b,
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
  using ElementO = typename Traits::ElementO;
  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using TiledMma = typename Traits::TiledMma;
  using SmemCopyAtom = typename Traits::SmemCopyAtom;

  constexpr int kBM = Traits::kBM;
  constexpr int kBN = Traits::kBN;
  constexpr int kBK = Traits::kBK;
  constexpr int kStages = Traits::kStages;
  constexpr int kNumThreads = Traits::kNumThreads;
  constexpr int kAElems = cosize(SmemLayoutA{});  // 16384
  constexpr int kBElems = cosize(SmemLayoutB{});

  const int tid = threadIdx.x;
  const int m0 = blockIdx.y * kBM;  // grid: (n-tiles, m-tiles)
  const int n0 = blockIdx.x * kBN;

  auto mA = tma_a.get_tma_tensor(make_shape(M, K));
  auto mB = tma_b.get_tma_tensor(make_shape(N, K));
  auto mC = tma_c.get_tma_tensor(make_shape(M, N));
  auto a_slice = tma_a.get_slice(_0{});
  auto b_slice = tma_b.get_slice(_0{});
  auto c_slice = tma_c.get_slice(_0{});

  extern __shared__ __align__(1024) Element shm[];
  Element *a_base = shm;
  Element *b_base = shm + kStages * kAElems;

  __shared__ uint64_t full[kStages];
  __shared__ uint64_t empty[kStages];
  if (tid == 0) {
    for (int s = 0; s < kStages; ++s) {
      TmaBarrier::init(&full[s], 1);  // 1 次 TMA 事务
      CtaBarrier::init(&empty[s], kNumThreads);
    }
  }
  __syncthreads();

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto s2r_copy_a = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto s2r_copy_b = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto s2r_thr_a = s2r_copy_a.get_thread_slice(tid);
  auto s2r_thr_b = s2r_copy_b.get_thread_slice(tid);

  auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<kBM>, Int<kBN>>{});
  clear(tCrC);
  // 行列坐标视图：epilogue 按 (m,n) 查 sa/sb
  auto cC = make_identity_tensor(Shape<Int<kBM>, Int<kBN>>{});
  auto tScC = thr_mma.partition_C(cC);
  auto tScC_rc =
      make_tensor(tScC.data(), convert_layout_acc_rowcol(tScC.layout()));
  constexpr int kCRows = decltype(size<0>(tScC_rc))::value;
  constexpr int kCCols = decltype(size<1>(tScC_rc))::value;

  // TMA 发射（仅 tid0）：一个 barrier 同时覆盖 A+B 两个字段的字节数
  auto issue_tma = [&](int kt, int stage) {
    cutlass::arch::fence_view_async_shared();
    auto sA =
        make_tensor(make_smem_ptr(a_base + stage * kAElems), SmemLayoutA{});
    auto sB =
        make_tensor(make_smem_ptr(b_base + stage * kBElems), SmemLayoutB{});
    auto gA = local_tile(mA, Shape<Int<kBM>, Int<kBK>>{},
                         make_coord(blockIdx.y, kt));
    auto gB = local_tile(mB, Shape<Int<kBN>, Int<kBK>>{},
                         make_coord(blockIdx.x, kt));
    TmaBarrier::arrive_and_expect_tx(&full[stage],
                                     sizeof(Element) * (size(sA) + size(sB)));
    copy(tma_a.with(full[stage]), a_slice.partition_S(gA),
         a_slice.partition_D(sA));
    copy(tma_b.with(full[stage]), b_slice.partition_S(gB),
         b_slice.partition_D(sB));
  };

  const int nKt = (K + kBK - 1) / kBK;
  // 初始 release: 全线程各 arrive 一次凑满 empty, tid0 发首批 TMA
  for (int s = 0; s < kStages; ++s) CtaBarrier::arrive(&empty[s]);
  if (tid == 0) {
    // 必须按 nKt 截断：K<=128 时多发的 TMA 是越界读，且会写脏 O staging
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

    auto sA =
        make_tensor(make_smem_ptr(a_base + stage * kAElems), SmemLayoutA{});
    auto sB =
        make_tensor(make_smem_ptr(b_base + stage * kBElems), SmemLayoutB{});
    auto tCrA = thr_mma.partition_fragment_A(sA);
    auto tCrB = thr_mma.partition_fragment_B(sB);
    auto tAsA = s2r_thr_a.partition_S(sA);
    auto tBsB = s2r_thr_b.partition_S(sB);
    gemm_ss(tCrC, tCrA, tCrB, tAsA, tBsB, tiled_mma, s2r_copy_a, s2r_copy_b,
            s2r_thr_a, s2r_thr_b);

    CtaBarrier::arrive(&empty[stage]);
    if (tid == 0) {  // 预取 kt+kStages（须在本轮 arrive 之后，见上）
      const int kt_next = kt + kStages;
      if (kt_next < nKt) {
        const int s_next = kt_next % kStages;
        const int ph_next = (kt_next / kStages) & 1;
        CtaBarrier::wait(&empty[s_next], ph_next);
        issue_tma(kt_next, s_next);
      }
    }
  }

  // Epilogue: 在线反量化 C = acc * sa * sb -> bf16 -> STSM r2s（复用 K 循环
  // 结束后的 A+B 区作 O staging) -> TMA store(OOB 行自动裁剪).
  {
    __syncthreads();  // 全部 mma 完成，smem 可覆写
    auto tCrC_rc =
        make_tensor(tCrC.data(), convert_layout_acc_rowcol(tCrC.layout()));
    // per-block 粒度 128：kBN=256 的 tile 跨两块，须逐元素按全局行列查表
    // （尾块按 m<M / n<N 守卫，不读越界 scale；per-row/per-col 仍按 m/n 查）

#pragma unroll
    for (int r = 0; r < kCRows; ++r) {
      const int m = m0 + get<0>(tScC_rc(r, 0));
      const float sa_r = kPerRowA ? ((m < M) ? sa[m] : 0.0f)  // 尾行 guard
                                  : ((m < M) ? sa[m / kScaleBlock] : 0.0f);
#pragma unroll
      for (int c = 0; c < kCCols; ++c) {
        const int n = n0 + get<1>(tScC_rc(r, c));
        const float sb_c = kPerColB ? ((n < N) ? sb[n] : 0.0f)
                                    : ((n < N) ? sb[n / kScaleBlock] : 0.0f);
        tCrC_rc(r, c) = tCrC_rc(r, c) * sa_r * sb_c;
      }
    }
    auto tCrCh = convert_type<ElementO>(tCrC);

    auto r2s_copy = make_tiled_copy_C(
        Copy_Atom<SM90_U32x4_STSM_N, ElementO>{}, tiled_mma);
    auto r2s_thr = r2s_copy.get_thread_slice(tid);
    auto sC =
        make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(shm)),
                    SmemLayoutO{});  // 覆盖已空闲的 A+B 区
    auto tCrCh_src = r2s_thr.retile_S(tCrCh);
    auto tCsC_dst = r2s_thr.partition_D(sC);
    copy(r2s_copy, tCrCh_src, tCsC_dst);
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    auto gC = local_tile(mC, Shape<Int<kBM>, Int<kBN>>{},
                         make_coord(blockIdx.y, blockIdx.x));
    if (tid == 0)
      copy(tma_c, c_slice.partition_S(sC), c_slice.partition_D(gC));
    tma_store_arrive();
    tma_store_wait<0>();
  }
#endif  // __CUDA_ARCH__ >= 900
}

// =============================================================================
// Phase 9.6: WS 进阶变体 — 128 producer + 256 consumer + setmaxnreg
// =============================================================================
// 与非 WS 的数学完全相同，差异全在执行协议（对照 ffpa persist-D）：
//   - producer warpgroup(128 线程) 只有线程 0 发 TMA, 其余线程闲置；
//     setmaxnreg.dec 释放寄存器(32), consumer setmaxnreg.inc 要回(232)
//     —— 直接调用 cutlass::arch::warpgroup_reg_{de,}alloc(ffpa persist-D
//     同款 API).setmaxnreg 在 sm_120a/sm_120f 都完整支持且保留（生死条件
//     是 cta 形式 TMA + __launch_bounds__(N,1) 双参数，与 arch 后缀无关，
//     见书 ch26b 专论), 本 kernel 两条都天然满足。
//   - empty barrier 计数 = 256（仅 consumer arrive）；producer 独立跑满
//     kStages 深度的预取窗口，不再占用消费者循环内的发射槽位
//   - epilogue 只涉及 consumer -> CTA 级 __syncthreads 换成
//     NamedBarrier(256)(producer 不参与，否则 384 线程分支错位死锁)
// 线程数 = 128(producer) + kNumThreads(consumer = kBM/16*32)，故 launch_bounds
// 与 grid 启动都跟着 Traits 走；BM=128 时正好 384（书里那张图的数字）。
template <typename Traits, typename TmaA, typename TmaB, typename TmaC,
          bool kPerRowA, bool kPerColB>
__global__ void __launch_bounds__(Traits::kNumThreads + 128, 1)
    fp8_gemm_tma_ws_kernel(CUTLASS_GRID_CONSTANT TmaA const tma_a,
                           CUTLASS_GRID_CONSTANT TmaB const tma_b,
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
  using ElementO = typename Traits::ElementO;
  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using TiledMma = typename Traits::TiledMma;
  using SmemCopyAtom = typename Traits::SmemCopyAtom;

  constexpr int kBM = Traits::kBM;
  constexpr int kBN = Traits::kBN;
  constexpr int kBK = Traits::kBK;
  constexpr int kStages = Traits::kStages;
  constexpr int kProducerThreads = 128;
  constexpr int kNumThreads = Traits::kNumThreads;  // consumer = kBM/16*32
  constexpr int kAElems = cosize(SmemLayoutA{});
  constexpr int kBElems = cosize(SmemLayoutB{});
  // setmaxnreg 预算：producer 32 寄存器 + consumer 232，总量必须装进 64K 寄存器
  // 文件（64K = 每个 SM 的寄存器堆）。BM<=128 成立，BM=256（512 consumer）
  // 会在这里编译期失败——这也是 WS 变体只扫 kBM<=128 的原因。
  static_assert(kProducerThreads * 32 + kNumThreads * 232 <= 65536,
                "WS register budget: 128 x 32 + consumer x 232 must fit 64K");

  const int tid = threadIdx.x;
  const int m0 = blockIdx.y * kBM;
  const int n0 = blockIdx.x * kBN;

  auto mA = tma_a.get_tma_tensor(make_shape(M, K));
  auto mB = tma_b.get_tma_tensor(make_shape(N, K));
  auto mC = tma_c.get_tma_tensor(make_shape(M, N));
  auto a_slice = tma_a.get_slice(_0{});
  auto b_slice = tma_b.get_slice(_0{});
  auto c_slice = tma_c.get_slice(_0{});

  extern __shared__ __align__(1024) Element shm[];
  Element *a_base = shm;
  Element *b_base = shm + kStages * kAElems;

  __shared__ uint64_t full[kStages];
  __shared__ uint64_t empty[kStages];
  if (tid == 0) {
    for (int s = 0; s < kStages; ++s) {
      TmaBarrier::init(&full[s], 1);
      CtaBarrier::init(&empty[s], kNumThreads);  // 仅 consumer arrive
    }
  }
  __syncthreads();

  auto issue_tma = [&](int kt, int stage) {
    cutlass::arch::fence_view_async_shared();
    auto sA =
        make_tensor(make_smem_ptr(a_base + stage * kAElems), SmemLayoutA{});
    auto sB =
        make_tensor(make_smem_ptr(b_base + stage * kBElems), SmemLayoutB{});
    auto gA = local_tile(mA, Shape<Int<kBM>, Int<kBK>>{},
                         make_coord(blockIdx.y, kt));
    auto gB = local_tile(mB, Shape<Int<kBN>, Int<kBK>>{},
                         make_coord(blockIdx.x, kt));
    TmaBarrier::arrive_and_expect_tx(&full[stage],
                                     sizeof(Element) * (size(sA) + size(sB)));
    copy(tma_a.with(full[stage]), a_slice.partition_S(gA),
         a_slice.partition_D(sA));
    copy(tma_b.with(full[stage]), b_slice.partition_S(gB),
         b_slice.partition_D(sB));
  };

  const int nKt = (K + kBK - 1) / kBK;
  if (tid < kProducerThreads) {
    // ---------------- producer warpgroup ----------------
    cutlass::arch::warpgroup_reg_dealloc<32>();  // 整 warpgroup 对齐执行
    if (tid == 0) {
      for (int kt = 0; kt < nKt; ++kt) {
        const int stage = kt % kStages;
        const int phase = (kt / kStages) & 1;
        CtaBarrier::wait(&empty[stage], phase);
        issue_tma(kt, stage);
      }
    }
    // producer 不参与 epilogue, 直接结束（后续 barrier 均 consumer-only）
  } else {
    // ---------------- consumer warpgroups ----------------
    cutlass::arch::warpgroup_reg_alloc<232>();
    const int ctid = tid - kProducerThreads;
    for (int s = 0; s < kStages; ++s) CtaBarrier::arrive(&empty[s]);

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(ctid);
    auto s2r_copy_a = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto s2r_copy_b = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto s2r_thr_a = s2r_copy_a.get_thread_slice(ctid);
    auto s2r_thr_b = s2r_copy_b.get_thread_slice(ctid);
    auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<kBM>, Int<kBN>>{});
    clear(tCrC);
    auto cC = make_identity_tensor(Shape<Int<kBM>, Int<kBN>>{});
    auto tScC = thr_mma.partition_C(cC);
    auto tScC_rc =
        make_tensor(tScC.data(), convert_layout_acc_rowcol(tScC.layout()));
    constexpr int kCRows = decltype(size<0>(tScC_rc))::value;
    constexpr int kCCols = decltype(size<1>(tScC_rc))::value;

#pragma unroll 1
    for (int kt = 0; kt < nKt; ++kt) {
      const int stage = kt % kStages;
      const int phase = (kt / kStages) & 1;
      TmaBarrier::wait(&full[stage], phase);
      cutlass::arch::fence_view_async_shared();
      auto sA =
          make_tensor(make_smem_ptr(a_base + stage * kAElems), SmemLayoutA{});
      auto sB =
          make_tensor(make_smem_ptr(b_base + stage * kBElems), SmemLayoutB{});
      auto tCrA = thr_mma.partition_fragment_A(sA);
      auto tCrB = thr_mma.partition_fragment_B(sB);
      auto tAsA = s2r_thr_a.partition_S(sA);
      auto tBsB = s2r_thr_b.partition_S(sB);
      gemm_ss(tCrC, tCrA, tCrB, tAsA, tBsB, tiled_mma, s2r_copy_a, s2r_copy_b,
              s2r_thr_a, s2r_thr_b);
      CtaBarrier::arrive(&empty[stage]);
    }

    // epilogue(consumer-only NamedBarrier, 256 线程)
    cutlass::arch::NamedBarrier::arrive_and_wait(kNumThreads, 0);
    auto tCrC_rc =
        make_tensor(tCrC.data(), convert_layout_acc_rowcol(tCrC.layout()));
    // per-block 粒度 128：kBN=256 的 tile 跨两块，须逐元素按全局 m/n 查表
#pragma unroll
    for (int r = 0; r < kCRows; ++r) {
      const int m = m0 + get<0>(tScC_rc(r, 0));
      const float sa_r = kPerRowA ? ((m < M) ? sa[m] : 0.0f)
                                  : ((m < M) ? sa[m / kScaleBlock] : 0.0f);
#pragma unroll
      for (int c = 0; c < kCCols; ++c) {
        const int n = n0 + get<1>(tScC_rc(r, c));
        const float sb_c = kPerColB ? ((n < N) ? sb[n] : 0.0f)
                                    : ((n < N) ? sb[n / kScaleBlock] : 0.0f);
        tCrC_rc(r, c) = tCrC_rc(r, c) * sa_r * sb_c;
      }
    }
    auto tCrCh = convert_type<ElementO>(tCrC);
    auto r2s_copy = make_tiled_copy_C(
        Copy_Atom<SM90_U32x4_STSM_N, ElementO>{}, tiled_mma);
    auto r2s_thr = r2s_copy.get_thread_slice(ctid);
    auto sC =
        make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(shm)),
                    SmemLayoutO{});
    auto tCrCh_src = r2s_thr.retile_S(tCrCh);
    auto tCsC_dst = r2s_thr.partition_D(sC);
    copy(r2s_copy, tCrCh_src, tCsC_dst);
    cutlass::arch::fence_view_async_shared();
    cutlass::arch::NamedBarrier::arrive_and_wait(kNumThreads, 0);
    auto gC = local_tile(mC, Shape<Int<kBM>, Int<kBN>>{},
                         make_coord(blockIdx.y, blockIdx.x));
    if (ctid == 0)
      copy(tma_c, c_slice.partition_S(sC), c_slice.partition_D(gC));
    tma_store_arrive();
    tma_store_wait<0>();
  }
#endif  // __CUDA_ARCH__ >= 900
}

// =============================================================================
// Phase 9.7: launch wrapper + C++ API(BF16 in -> 动态量化 -> FP8 GEMM -> BF16)
// =============================================================================
// TMA descriptor 在 host 栈上构建(make_tma_copy 内部完成 16B 对齐校验),
// 经 __grid_constant__ 传参；grid = (n-tiles, m-tiles).
template <bool kPerRowA, bool kPerColB, bool kWS,
          typename Traits = Fp8GemmTraits<>>
void fp8_gemm_tma_fwd(const cutlass::float_e4m3_t *A8,
                      const cutlass::float_e4m3_t *B8T, const float *sa,
                      const float *sb, cutlass::bfloat16_t *C, int M, int N,
                      int K, cudaStream_t stream) {
  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  constexpr int kBM = Traits::kBM;
  constexpr int kBN = Traits::kBN;
  constexpr int kBK = Traits::kBK;
  constexpr int kSmemBytes = Traits::kSmemBytes;

  // TN 布局：C[M,N] = A8[M,K] x B8T[N,K]^T(B8T 行主序 = B 列主序)
  auto mA =
      make_tensor(make_gmem_ptr(A8), make_shape(M, K), make_stride(K, _1{}));
  auto mB =
      make_tensor(make_gmem_ptr(B8T), make_shape(N, K), make_stride(K, _1{}));
  auto mC =
      make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, _1{}));
  auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, mA, SmemLayoutA{},
                             Shape<Int<kBM>, Int<kBK>>{}, _1{});
  auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, mB, SmemLayoutB{},
                             Shape<Int<kBN>, Int<kBK>>{}, _1{});
  auto tma_c = make_tma_copy(SM90_TMA_STORE{}, mC, SmemLayoutO{},
                             Shape<Int<kBM>, Int<kBN>>{}, _1{});

  dim3 grid((N + kBN - 1) / kBN, (M + kBM - 1) / kBM);
  if constexpr (kWS) {
    auto kernel = fp8_gemm_tma_ws_kernel<Traits, decltype(tma_a),
                                         decltype(tma_b), decltype(tma_c),
                                         kPerRowA, kPerColB>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSmemBytes);
    kernel<<<grid, Traits::kNumThreads + 128, kSmemBytes, stream>>>(
        tma_a, tma_b, tma_c, sa, sb, M, N, K);
  } else {
    auto kernel = fp8_gemm_tma_kernel<Traits, decltype(tma_a), decltype(tma_b),
                                      decltype(tma_c), kPerRowA, kPerColB>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSmemBytes);
    kernel<<<grid, Traits::kNumThreads, kSmemBytes, stream>>>(tma_a, tma_b,
                                                              tma_c, sa, sb,
                                                              M, N, K);
  }
}

// 一次性 workspace（调用方分配/复用；上界按 per-row/per-col 计）：
//   [A8 e4m3 M*K | B8T e4m3 N*K | sa float M | sb float N] 各段 256B 对齐
struct Fp8GemmWorkspace {
  void *buf = nullptr;
  size_t size = 0;
};

inline size_t fp8_gemm_align256(size_t x) { return (x + 255) & ~(size_t)255; }

inline size_t fp8_gemm_workspace_size(int M, int N, int K) {
  return fp8_gemm_align256((size_t)M * K) +
         fp8_gemm_align256((size_t)N * K) +
         fp8_gemm_align256((size_t)M * sizeof(float)) +
         fp8_gemm_align256((size_t)N * sizeof(float));
}

// BF16 in -> 动态量化 FP8 GEMM（在线反量化）-> BF16 out, 单 stream 语义。
// 同步约定：与 cuBLAS 一致，入队后由调用方在 stream 上同步。
// 约束：K%16==0, N%8==0(TMA 对齐), M 任意；use_ws=true 走 warp-specialized
// 变体（producer 128 + consumer kBM/16*32 线程；默认 tile 下共 384，数值等价）。
inline void fp8_gemm_bf16(const __nv_bfloat16 *A, const __nv_bfloat16 *B,
                          __nv_bfloat16 *C, int M, int N, int K,
                          Fp8GemmScaleMode mode, Fp8GemmWorkspace &ws,
                          cudaStream_t stream, bool use_ws = false) {
  if (K % 16 != 0 || N % 8 != 0) {
    fprintf(stderr,
            "fp8_gemm_bf16: require K%%16==0 (got K=%d), N%%8==0 (got N=%d)\n",
            K, N);
    return;
  }
  uint8_t *p = static_cast<uint8_t *>(ws.buf);
  cutlass::float_e4m3_t *a8 = reinterpret_cast<cutlass::float_e4m3_t *>(p);
  p += fp8_gemm_align256((size_t)M * K);
  cutlass::float_e4m3_t *b8t = reinterpret_cast<cutlass::float_e4m3_t *>(p);
  p += fp8_gemm_align256((size_t)N * K);
  float *sa = reinterpret_cast<float *>(p);
  p += fp8_gemm_align256((size_t)M * sizeof(float));
  float *sb = reinterpret_cast<float *>(p);

  const bool per_row_a = mode == Fp8GemmScaleMode::kPerRowPerCol ||
                         mode == Fp8GemmScaleMode::kPerRowPerBlock;
  const bool per_col_b = mode == Fp8GemmScaleMode::kPerRowPerCol ||
                         mode == Fp8GemmScaleMode::kPerBlockPerCol;
  if (per_row_a)
    quantize_a_perrow_kernel<<<(M + 7) / 8, 256, 0, stream>>>(A, a8, sa, M,
                                                              K);
  else
    quantize_a_perblock_kernel<<<(M + kScaleBlock - 1) / kScaleBlock, 256, 0,
                                 stream>>>(A, a8, sa, M, K);
  fp8_gemm_quantize_b(B, b8t, sb, N, K, per_col_b, stream);

  auto *cO = reinterpret_cast<cutlass::bfloat16_t *>(C);
#define FP8_GEMM_LAUNCH(PRA, PCB)                                          \
  do {                                                                     \
    if (use_ws)                                                            \
      fp8_gemm_tma_fwd<PRA, PCB, true>(a8, b8t, sa, sb, cO, M, N, K,       \
                                       stream);                            \
    else                                                                   \
      fp8_gemm_tma_fwd<PRA, PCB, false>(a8, b8t, sa, sb, cO, M, N, K,      \
                                        stream);                           \
  } while (0)
  if (per_row_a && per_col_b)
    FP8_GEMM_LAUNCH(true, true);
  else if (per_row_a)
    FP8_GEMM_LAUNCH(true, false);
  else if (per_col_b)
    FP8_GEMM_LAUNCH(false, true);
  else
    FP8_GEMM_LAUNCH(false, false);
#undef FP8_GEMM_LAUNCH
}

// =============================================================================
// Phase 9.8: 权重 B 离线量化（推理部署的真实形态）
// =============================================================================
// 推理里 A（激活）每步都变，B（权重）训练完就固定：B 只需离线量化一次，产物
// (B8T, sb) 常驻显存（生产上随 checkpoint 一起落盘），之后每次前向只量化 A.
// 省掉的是 O(NK) 那条带宽工序 —— e2e 里最贵的一段(35.8.2 节的账).
// 两种量化的角色并不对称，这正是量化 GEMM 能落地的前提：
//   δb 只依赖权重本身 -> 离线可算；δa 依赖当次激活 -> 必须在线(per-token).
struct Fp8GemmActivation {  // A 侧在线量化暂存（跨调用复用，一次分配）
  void *buf = nullptr;
  size_t size = 0;
};

// [A8 e4m3 M*K | sa float M], 各 256B 对齐(B 段不再需要).
inline size_t fp8_gemm_activation_size(int M, int K) {
  return fp8_gemm_align256((size_t)M * K) +
         fp8_gemm_align256((size_t)M * sizeof(float));
}

// B 已离线量化：每步只量化 A -> 同一个 FP8 GEMM 主 kernel -> BF16 输出。
// 契约：B8T 为 fp8_gemm_quantize_b 产出的 (N,K) 行主序(内维 K, 16B 对齐)；
// sb 长度 = N(kPerColB=true) 或 ceil(N/128)；kPerColB 必须与离线量化 B 时
// 传给 fp8_gemm_quantize_b 的 per_col_b 同值，否则整 tile 被错误 scale 静默错值。
// A 侧粒度用模板参数（部署时 mode 固定，与 fp8_gemm_tma_fwd 的取值方式一致）。
template <bool kPerRowA, bool kPerColB, bool kWS = false,
          typename Traits = Fp8GemmTraits<>>
inline void fp8_gemm_bf16_b_offline(const __nv_bfloat16 *A,
                                    const cutlass::float_e4m3_t *B8T,
                                    const float *sb, __nv_bfloat16 *C, int M,
                                    int N, int K, Fp8GemmActivation &act,
                                    cudaStream_t stream) {
  if (K % 16 != 0 || N % 8 != 0) {
    fprintf(stderr,
            "fp8_gemm_bf16_b_offline: require K%%16==0 (got K=%d), N%%8==0 "
            "(got N=%d)\n",
            K, N);
    return;
  }
  uint8_t *p = static_cast<uint8_t *>(act.buf);
  cutlass::float_e4m3_t *a8 = reinterpret_cast<cutlass::float_e4m3_t *>(p);
  p += fp8_gemm_align256((size_t)M * K);
  float *sa = reinterpret_cast<float *>(p);

  if constexpr (kPerRowA)
    quantize_a_perrow_kernel<<<(M + 7) / 8, 256, 0, stream>>>(A, a8, sa, M,
                                                              K);
  else
    quantize_a_perblock_kernel<<<(M + kScaleBlock - 1) / kScaleBlock, 256, 0,
                                 stream>>>(A, a8, sa, M, K);
  fp8_gemm_tma_fwd<kPerRowA, kPerColB, kWS, Traits>(
      a8, B8T, sa, sb, reinterpret_cast<cutlass::bfloat16_t *>(C), M, N, K,
      stream);
}

}  // namespace fp8_gemm
#endif  // NOTES_V2_ENABLE_CUTE && NOTES_V2_ENABLE_TMA_MMA_WS
