#pragma once
// common.cuh: CUDA headers, macros, MMA/WGMMA PTX, swizzle, TMA helpers

#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>

#if defined(NOTES_V2_ENABLE_CUDNN)
#pragma nv_diag_suppress 128
#include <cudnn_frontend.h>
#pragma nv_diag_default 128
namespace fe = cudnn_frontend;
#endif

#if defined(NOTES_V2_ENABLE_TMA_MMA_WS) && CUDART_VERSION < 13000
#error "NOTES_V2_ENABLE_TMA_MMA_WS requires CUDA Toolkit 13.0 or newer"
#endif

#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])

static constexpr int kWarpSize = 32;


// Phase 7b-1: MMA PTX 宏定义
// =============================================================================

// ---- gmem → smem: cp.async ----
// cp.async.commit_group / wait_group / wait_all 语义（PTX ISA §9.7.9.25.3）：
//   - commit_group: 将此前所有未提交的 cp.async 归入一个新的 async-group（per-thread）。
//   - wait_group N: 阻塞直到最多 N 个 async-group 尚未完成（即 pending ≤ N）。
//     N=0 → 等所有 group 完成。与 wgmma.wait_group 语义一致。
//   - wait_all: 等价于 commit_group + wait_group 0。
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)                                                 \
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

// 注意：cg 只支持 16 bytes，ca 支持 4/8/16 bytes
#define CP_ASYNC_CG(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))

// ---- ldmatrix: smem → register（Tensor Core 专用）----
// ldmatrix.sync.aligned.xN.m8n8.shared.b16
// 每次加载 8×8 的 half 矩阵片段到 1/2/4 条 32-bit 寄存器
// aligned: 要求 128-bit 对齐, trans:  转置加载
#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                      \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"     \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(addr))

#define LDMATRIX_X2(R0, R1, addr)                                              \
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"    \
               : "=r"(R0), "=r"(R1)                                            \
               : "r"(addr))

// ldmatrix.x2.trans: 转置加载（用于 NN 布局中需要 col-major B 矩阵的场景）
// FA 中 V[Bc,d] 为 row-major，但 P@V 的 MMA 需要 col-major 的 B → 使用 trans
#define LDMATRIX_X2_T(R0, R1, addr)                                            \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"       \
      : "=r"(R0), "=r"(R1)                                                     \
      : "r"(addr))

// ---- mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 ----
// m16n8k16: M=16, N=8, K=16（Ampere Tensor Core 的基本 tile）
// row.col: A 是 row-major, B 是 col-major
// f16.f16.f16.f16: A/B 是 f16, C/D 是 f16（f32 累加版本用 f32.f16.f16.f32）
// 2 个输出寄存器（RD0, RD1），4 个 A 寄存器 + 2 个 B 寄存器
// C 矩阵大小 16×8=128 元素 = 128 个 half；32 线程分担，每线程 4 half = 2 个 uint32
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)            \
  asm volatile(                                                                \
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, "  \
      "%4, %5}, {%6, %7}, {%8, %9};\n"                                         \
      : "=r"(RD0), "=r"(RD1)                                                   \
      : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0),  \
        "r"(RC1))

// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
// f32 accumulator variant: 4 output registers (RD0-RD3), A/B still f16.
// C matrix 16x8=128 elements, 32 threads × 4 f32 = 4 uint32 per thread.
#define HMMA16816F32(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1, RC0,    \
                     RC1, RC2, RC3)                                            \
  asm volatile(                                                                \
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, "        \
      "%3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"               \
      : "=r"(RD0), "=r"(RD1), "=r"(RD2), "=r"(RD3)                             \
      : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0),  \
        "r"(RC1), "r"(RC2), "r"(RC3))

// ---- MMA 辅助函数 ----
// div_ceil: 整数除法向上取整
#define HOST_DEVICE_INLINE __device__ __host__ inline
HOST_DEVICE_INLINE int div_ceil(int a, int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

// =============================================================================

// Phase 7b-3: HGEMM MMA Swizzle — m16n8k16 + multistage pipeline + TN 布局
// + XOR swizzle 消除 smem bank conflict（统一循环版）
// =============================================================================
// 面试要点（Swizzle 原理 — smem Bank Conflict 消除）：
//   - 问题：ldmatrix 以 8×8 片段（m8n8）从 smem 加载 16×16 的矩阵。同一 warp 中
//     不同 lane 访问的地址在 bank 上产生冲突 → 串行化（most common: 2/4-way）。
//   - XOR swizzle 方案：对列地址做 `((j>>3) ^ (i>>2)) % 2 << 3` 的 XOR 置换，
//     将连续 4 行的 bank 映射打散。每 4 行为一组的 pattern 交替：
//        rows 0~3: 列 0→地址偏移 0, 列 8→地址偏移 8
//        rows 4~7: 列 0→地址偏移 8, 列 8→地址偏移 0（XOR 翻转）
//        rows 8~11: repeat rows 0~3 pattern
//        rows 12~15: repeat rows 4~7 pattern
//   - 效果：原来 n-way 的 bank conflict 降低到 1-way（fully conflict-free）。
//   - 优势：无需 smem PAD（不浪费空间），原理上完全消除特定 pattern 的 bank conflict。
//   - 局限性：要求 kColStride ≤ 16（即 BK ≤ 16），kStep ∈ {4,8}。
//     对 MMA m16n8k16 的 BK=16 而言刚好满足。
//
// 参考：LeetCUDA/kernels/hgemm/mma/swizzle/hgemm_mma_stage_tn_swizzle.cu
// 参考：https://zhuanlan.zhihu.com/p/4746910252

// ---- Swizzle 辅助函数 ----

// i: row index; j: col index.
// Returns the chunk-level (8 fp16 = 16 B per chunk) column swizzle, i.e.
// 0 / 8 / 16 / ... that the caller adds to the chunk-aligned base column.
// The within-chunk offset (``j & 7``) is preserved by the caller.
//
// kColStride matches the equivalent CuTe ``Swizzle<B, M, S>`` swizzle
// hardware mode used by TMA bulk-tensor descriptors:
//   * kColStride = 16 fp16 = 32 B/row -> Swizzle<1, 4, 3>  = SWIZZLE_32B
//   * kColStride = 32 fp16 = 64 B/row -> Swizzle<2, 4, 3>  = SWIZZLE_64B
//   * kColStride = 64 fp16 = 128 B/row -> Swizzle<3, 4, 3> = SWIZZLE_128B
// The XOR mask is derived from the byte-address bit positions used by the
// underlying CuTe swizzle pattern: bits {4..(4+B-1)} XOR bits
// {7..(7+B-1)} of the absolute byte address. Translating to (i, j) with
// row stride ``kColStride * 2`` bytes:
//   * 32B (B=1) : (j>>3) ^ (i>>2)              [chunk in {0, 1}]
//   * 64B (B=2) : chunk(2b) ^ {(i>>1)&1, (i>>2)&1}      [{0..3}]
//   * 128B(B=3) : chunk(3b) ^ {i&1, (i>>1)&1, (i>>2)&1} [{0..7}]
// source: LeetCUDA/ffpa-attn/cuffpa/swizzle.cuh
//
// v1/v2 实现 (由编译宏 NOTES_V2_ENABLE_SWIZZLE_V2 门控):
//   swizzle_v1_impl : 原手写 XOR 分支 (permuted<kColStride,8>), 按列宽展开.
//   swizzle_v2_impl : 用本地 SwizzleBMS<B,M,S> 镜像 cute::Swizzle<B,M,S>::apply
//                     位级公式统一表达 (16/32/64); kColStride=8 回退 v1.
//   swizzle         : 公开派发器, 宏门控, 接口与原 swizzle<kColStride> 一致.
// v2 不是新算法, 而是把 v1 的手写展开重新表达为 cute 的统一位置换公式,
// 便于脱离 cute 框架理解 swizzle 本质. v1/v2 bit-exact 等价.
template <const int kColStride = 16, const int kStep = 8>
static __host__ __device__ __forceinline__ int permuted(int i, int j) {
  static_assert(kColStride == 16 || kColStride == 32 || kColStride == 64 ||
                kColStride == 8,
                "kColStride must be one of {8, 16, 32, 64} (matches "
                "SWIZZLE_32B/64B/128B + the "
                "kStep=4 narrow legacy case).");
  static_assert(kStep == 4 || kStep == 8, "kStep must be 8 or 4.");
  static_assert(kColStride % kStep == 0,
                "kColStride must be multiple of kStep.");
  if constexpr (kStep == 4) {
    static_assert(kColStride <= 16, "kStep=4 only supports (kColStride <= 16).");
    return (((j >> 2) ^ (i >> 2)) % (kColStride >> 2)) << 2;
  } else if constexpr (kColStride == 16) {
    return (((j >> 3) ^ (i >> 2)) & 1) << 3;  // SWIZZLE_32B
  } else if constexpr (kColStride == 32) {
    const int chunk = (j >> 3) & 3;
    const int xor_mask = ((i >> 1) & 1) | (((i >> 2) & 1) << 1);
    return (chunk ^ xor_mask) << 3;  // SWIZZLE_64B
  } else {                           // kColStride == 64
    const int chunk = (j >> 3) & 7;
    const int xor_mask =
        (i & 1) | (((i >> 1) & 1) << 1) | (((i >> 2) & 1) << 2);
    return (chunk ^ xor_mask) << 3;  // SWIZZLE_128B
  }
}

// Manually SMEM swizzling for bank conflict free.
// ----------------------------------------------------------------
// [INFO] Assert smem store layout col_stride <= 16, prefer 16.   |
// [INFO] For logical_col_stride > 16, we have to permute the     |
// [INFO] smem store layout using col major ZigZag method:        |
// [INFO] e.g, --> Q smem logical layout [Br][64].                |
// [INFO]      --> col major ZigZag permuted -->                  |
// [INFO]      --> Q smem store layout [4][Br][16].               |
// ----------------------------------------------------------------
// ----------------------------------------------------------------
// -------------------------swizzle layout-------------------------
// --------------------logical col 0~64, step 8--------------------
// ---------------------smem col 0~16, step 8----------------------
// ----------------------------------------------------------------
// |bank  |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |
// |row 0 |  0   |  8   |  0   |  8   |  0   |  8   |  0   |  8   |
// |bank  |b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|
// |row 1 |  0   |  8   |  0   |  8   |  0   |  8   |  0   |  8   |
// |bank  |b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|
// |row 2 |  0   |  8   |  0   |  8   |  0   |  8   |  0   |  8   |
// |bank  |b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|
// |row 3 |  0   |  8   |  0   |  8   |  0   |  8   |  0   |  8   |
// ----------------------------------------------------------------
// |bank  |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |
// |row 4 |  8   |  0   |  8   |  0   |  8   |  0   |  8   |  0   |
// |bank  |b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|
// |row 5 |  8   |  0   |  8   |  0   |  8   |  0   |  8   |  0   |
// |bank  |b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|
// |row 6 |  8   |  0   |  8   |  0   |  8   |  0   |  8   |  0   |
// |bank  |b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|
// |row 7 |  8   |  0   |  8   |  0   |  8   |  0   |  8   |  0   |
// ----------------------------------------------------------------
// |bank  |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |
// |row 8 |  0   |  8   |  0   |  8   |  0   |  8   |  0   |  8   |
// |bank  |b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|
// |row 9 |  0   |  8   |  0   |  8   |  0   |  8   |  0   |  8   |
// |bank  |b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|
// |row 10|  0   |  8   |  0   |  8   |  0   |  8   |  0   |  8   |
// |bank  |b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|
// |row 11|  0   |  8   |  0   |  8   |  0   |  8   |  0   |  8   |
// ----------------------------------------------------------------
// |bank  |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |
// |row 12|  8   |  0   |  8   |  0   |  8   |  0   |  8   |  0   |
// |bank  |b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|
// |row 13|  8   |  0   |  8   |  0   |  8   |  0   |  8   |  0   |
// |bank  |b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|
// |row 14|  8   |  0   |  8   |  0   |  8   |  0   |  8   |  0   |
// |bank  |b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|
// |row 15|  8   |  0   |  8   |  0   |  8   |  0   |  8   |  0   |
// ----------------------------------------------------------------
template <const int kColStride = 16>
static __host__ __device__ __forceinline__ int swizzle_v1_impl(int i, int j) {
  return permuted<kColStride, 8>(i, j);
}

// Swizzle v2: cute::Swizzle<B,M,S> 位级镜像（脱离 cute 框架独立理解）
//
// v2 双重目的:
//   (1) 探索优化 v1 的可能 -- v1 按 kColStride 手写 XOR 分支展开, 已被 nvcc
//       strength-reduce 到极简位运算; v2 用 cute 的统一位置换公式重新表达,
//       预期中性, 但统一表达式可能更利于合并到地址算术 (bonus).
//   (2) 把 cute Swizzle 的核心位逻辑从 cute 框架提取出来独立理解 -- 让 notes-v2
//       不依赖 cute 头即可展示 Swizzle<B,M,S> 的位级语义, 便于读者脱离 cute
//       模板体系直接掌握 swizzle 原理.
//
// 注释叙述参考: 上方 hgemm_mma_stages_tn_cute 处的 Swizzle<B,M,S> 原理解释
// (二维逻辑空间 + 列置换), 此处适配 v2 的 (B,M,S)=(1/2/3, 4, 3) 语境.
//
// Swizzle<B,M,S> 的核心不是改变逻辑 Tensor 的 shape, 而是把一维 byte offset
// 重新解释为一个二维逻辑空间, 再把二维坐标映射回 bank-conflict-free 的物理
// offset:
//   1. 连续的 2^M 个元素组成二维空间中的一个基本元素 (元素宽度);
//   2. 连续的 2^S 个基本元素组成一行 (列数);
//   3. 二维空间包含 2^B 行 (行数, 这些行的 chunk 互相置换);
//   4. 对二维坐标做列置换: icol' = irow ^ icol;
//   5. 保留基本元素内部的低 M 位, 再将置换后的二维坐标编码回 offset.
//
// 位级实现 (与 cutlass/include/cute/swizzle.hpp 一致):
//   bit 布局:   0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
//                                  ^--^ MBase: 保留不参与置换的低位数
//                       ^-^       ^-^     BBits: XOR 掩码位数
//                         ^---------^       SShift: YYY 相对 ZZZ 的位移
//   apply(off) = off ^ shiftr(off & yyy_msk, S)
//   yyy_msk    = ((1<<B)-1) << (M + max(0,S))    // YYY 掩码位置
//   zzz_msk    = ((1<<B)-1) << (M - min(0,S))    // ZZZ 掩码位置 (文档用)
//
// 模板参数物理/语义含义:
//   B (BBits)  : 二维逻辑空间的行数位 = 2^B 行, 即 swizzle 周期覆盖 2^B 个 chunk
//                (这些 chunk 互相置换); 同时是 TMA 硬件 swizzle 模式
//                SWIZZLE_{32,64,128}B 的位数 (B=1/2/3).
//   M (MBase)  : 基本元素宽度位 = 2^M 个连续元素组成一个基本元素. 对 fp16 +
//                8-element chunk, 基本元素 = 16B = 2^4, 故 M=4; 地址 bit[0..M-1]
//                在 apply 中保持不变 (保留).
//   S (SShift) : 二维空间每行列数位 = 2^S 个基本元素组成一行; 同时是 YYY 掩码
//                相对 ZZZ 掩码的位移. S>0 表示 YYY 在更高位, 向右移 S 位到 ZZZ
//                位置后 XOR. 本 notes-v2 三模式 S 恒为 3 (YYY 在 bit[7..7+B-1],
//                即行索引 i 的低 B 位; ZZZ 在 bit[4..4+B-1], 即列 chunk 索引).
//
// 完整 swizzle 周期覆盖 2^(M+S+B) 个元素 (fp16 下 2^(M+S+B+1) 字节):
//   (B,M,S)=(1,4,3): 256 elem / 512B   (SWIZZLE_32B)
//   (B,M,S)=(2,4,3): 512 elem / 1024B  (SWIZZLE_64B)
//   (B,M,S)=(3,4,3): 1024 elem / 2048B (SWIZZLE_128B)
template <int B, int M, int S>
struct SwizzleBMS {
  static_assert(M >= 0, "MBase must be non-negative.");
  static_assert(B > 0, "BBits must be positive.");
  static_assert((S > 0 ? S : -S) >= B,
                "abs(SShift) must be >= BBits (cute::Swizzle constraint).");

  static constexpr int bit_msk = (1 << B) - 1;
  static constexpr int yyy_msk = bit_msk << (M + (S > 0 ? S : 0));
  static constexpr int zzz_msk = bit_msk << (M - (S < 0 ? S : 0));

  // apply: 对 byte offset 做 XOR 位置换, ZZZ ^= (YYY >> S).
  static __host__ __device__ __forceinline__ int apply(int offset) {
    if constexpr (S >= 0) {
      return offset ^ ((offset & yyy_msk) >> S);
    } else {
      return offset ^ ((offset & yyy_msk) << (-S));
    }
  }
};

// swizzle_v2_impl: 用 SwizzleBMS<B,M,S>::apply 计算 chunk 级列 swizzle.
//
// 流程: (i,j) -> byte offset off=(i*kColStride+j)*sizeof(half) -> SwizzleBMS
//       位级置换 -> 取 [M, M+B) 位作为物理 chunk 索引 -> 乘 kStep 还原为列偏移.
//
// kColStride -> (B,M,S) 映射表 (对应 TMA 硬件 swizzle 模式):
//   kColStride=16 fp16=32B/row  -> (B,M,S)=(1,4,3) = SWIZZLE_32B
//   kColStride=32 fp16=64B/row  -> (B,M,S)=(2,4,3) = SWIZZLE_64B
//   kColStride=64 fp16=128B/row -> (B,M,S)=(3,4,3) = SWIZZLE_128B
//
// kColStride=8 (kStep=4 legacy): 无对应 cute 标准 TMA swizzle 模式, 回退到
//   swizzle_v1_impl. 该路径当前无任何 kernel 使用, 回退保证接口契约零变更.
//
// 等价性: 对 (16,32,64), v2 输出与 v1 bit-exact 相同 (见 test_swizzle_equiv).
//   证明: off 的 bit[4..4+B-1] 恰是 (j>>3) 低 B 位 (因 i*kColStride 在这些位
//   为 0); bit[7..7+B-1] 恰是 i 低 B 位 (因 j<kColStride 不贡献高位). apply
//   把 YYY(bit[7..]) XOR 到 ZZZ(bit[4..]), 取 [M,M+B)=bit[4..4+B-1] 即得
//   (j>>3)^i 的低 B 位 = v1 的 chunk^xor_mask.
template <const int kColStride = 16>
static __host__ __device__ __forceinline__ int swizzle_v2_impl(int i, int j) {
  if constexpr (kColStride == 8) {
    return swizzle_v1_impl<kColStride>(i, j);
  } else {
    static_assert(kColStride == 16 || kColStride == 32 || kColStride == 64,
                  "v2 supports kColStride in {16, 32, 64} (cute TMA swizzle); "
                  "8 falls back to v1.");
    constexpr int B = (kColStride == 16) ? 1 : (kColStride == 32) ? 2 : 3;
    constexpr int M = 4;
    constexpr int S = 3;
    constexpr int kStep = 8;
    const int off = (i * kColStride + j) * (int)sizeof(half);
    const int sw = SwizzleBMS<B, M, S>::apply(off);
    return ((sw >> M) & ((1 << B) - 1)) * kStep;
  }
}

// swizzle: 公开派发器, 由编译宏 NOTES_V2_ENABLE_SWIZZLE_V2 门控 v1/v2.
//   未定义宏 (默认) -> swizzle_v1_impl (原手写 XOR 分支, 历史默认路径).
//   定义宏          -> swizzle_v2_impl (cute Swizzle<B,M,S> 位级镜像).
// 接口与原 swizzle<kColStride>(i,j) 完全一致, 所有 kernel 调用点无需改动.
// v1/v2 等价性由 host 端 test_swizzle_equiv (--swizzle-eq-check) 验证.

#if defined(NOTES_V2_ENABLE_WGMMA) || defined(NOTES_V2_ENABLE_TMA_MMA_WS)
// TMA/mbarrier helpers.
//
// Two implementations gated by NOTES_V2_FORCE_INLINE_ASYNC_PROXY:
//   - Defined (default): raw `asm volatile` PTX. ptxas keeps setmaxnreg
//     because raw asm carries no source annotation that ptxas would treat as
//     an "extern call" boundary (warning C7506 otherwise drops the hint).
//   - Undef'd: cuda::ptx:: / cuda::device:: C++ wrappers. Cleaner API but
//     ptxas drops setmaxnreg with C7506 even though the wrappers inline to
//     identical PTX; useful for comparing behavior / debugging.
//
// cuda::barrier (init/arrive/wait) is shared by both paths: it inlines to raw
// mbarrier PTX and does not trigger C7506, so it needs no gating.
//
// Mirrors the raw-PTX style of tmp/LeetGPU/CUDA/22_GEMM/sm90_wgmma_tma_ws_pingpong.cu.
#if defined(NOTES_V2_FORCE_INLINE_ASYNC_PROXY)

static __device__ __forceinline__ uint32_t
cast_smem_ptr_to_uint(void const *ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

static __device__ __forceinline__ void tma_fence_proxy_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// cp.async.bulk.tensor.2d: 2D TMA copy from global to shared::cluster.
//   [dst]            = shared memory destination (smem addr, 32-bit)
//   [desc, {c0, c1}]  = CUtensorMap + (minor_coord, major_coord) coords
//   [mbar]            = mbarrier in shared memory (smem addr, 32-bit)
// mbarrier::complete_tx::bytes: barrier flips phase when TMA bytes land.
// L2::cache_hint omitted (EVICT_NORMAL); matches the cuda::ptx default.
static __device__ __forceinline__ void tma_load_2d(
    void *dst, const CUtensorMap *tensor_map, int minor_coord, int major_coord,
    cuda::barrier<cuda::thread_scope_block> &barrier) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(tensor_map);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(dst);
  uint32_t smem_int_mbar =
      cast_smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&barrier));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4}], [%2];"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(minor_coord), "r"(major_coord)
      : "memory");
}

// mbarrier.arrive.expect_tx: register one arrival on the mbarrier and declare
// that `bytes` of async copy traffic is expected before the phase flips.
// Equivalent to cuda::ptx::mbarrier_arrive_expect_tx(release, cta, shared).
static __device__ __forceinline__ void tma_arrive_expect_tx(
    cuda::barrier<cuda::thread_scope_block> &barrier, uint32_t bytes) {
  uint32_t smem_int_mbar =
      cast_smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&barrier));
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
      :
      : "r"(smem_int_mbar), "r"(bytes)
      : "memory");
}

#else // !NOTES_V2_FORCE_INLINE_ASYNC_PROXY: use cuda::ptx / cuda::device wrappers

static __device__ __forceinline__ void tma_fence_proxy_async_shared_cta() {
#if CUDART_VERSION >= 13020
  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
#else
  cuda::device::experimental::fence_proxy_async_shared_cta();
#endif
}

static __device__ __forceinline__ void tma_load_2d(
    void *dst, const CUtensorMap *tensor_map, int minor_coord, int major_coord,
    cuda::barrier<cuda::thread_scope_block> &barrier) {
#if CUDART_VERSION >= 13020
  const int32_t coords[]{minor_coord, major_coord};
  auto *barrier_handle = cuda::device::barrier_native_handle(barrier);
  cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_cluster,
                                  cuda::ptx::space_global, dst, tensor_map,
                                  coords, barrier_handle);
#else
  cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
      dst, tensor_map, minor_coord, major_coord, barrier);
#endif
}

static __device__ __forceinline__ void tma_arrive_expect_tx(
    cuda::barrier<cuda::thread_scope_block> &barrier, uint32_t bytes) {
#if CUDART_VERSION >= 13020
  auto *barrier_handle = cuda::device::barrier_native_handle(barrier);
  [[maybe_unused]] auto token = cuda::ptx::mbarrier_arrive_expect_tx(
      cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared,
      barrier_handle, bytes);
#else
  [[maybe_unused]] auto token =
      cuda::device::barrier_arrive_tx(barrier, 1, bytes);
#endif
}

#endif // NOTES_V2_FORCE_INLINE_ASYNC_PROXY

// Warpgroup-level register rebalancing for warp specialization.
// setmaxnreg.dec releases registers (producer TMA path needs few),
// setmaxnreg.inc requests registers (consumer MMA path needs many).
// Both require all warps in a warpgroup to execute the same instruction,
// and require __launch_bounds__(N, 1) so the compiler permits up to 256
// regs/warp (otherwise the hint may have no effect). Supported on sm_90a/sm_120a.
// __forceinline__ is mandatory: without it ptxas drops setmaxnreg with
// warning C7506 "ignored to maintain compatibility into 'extern' call",
// because a non-inlined call boundary forces a fixed register convention.
//
// On sm_120a (Blackwell, CUDA 13.2) ptxas drops setmaxnreg with C7506 even
// when the PTX is fully inlined (no call.uni), because ptxas treats
// cp.async.bulk.tensor (TMA) usage as an implicit extern-call boundary.
// sm_90a (Hopper) is unaffected. Gate the *call sites* with
// NOTES_V2_ENABLE_SETMAXNREGS so sm_120a builds stay warning-free and avoid
// the register-allocation side effects of __launch_bounds__(N,1) until ptxas
// is fixed. The function templates themselves are always defined.
#if defined(NOTES_V2_ENABLE_SETMAXNREGS)
  #define NOTES_V2_REG_DEALLOC(N) warpgroup_reg_dealloc<N>()
  #define NOTES_V2_REG_ALLOC(N)   warpgroup_reg_alloc<N>()
#else
  #define NOTES_V2_REG_DEALLOC(N) ((void)0)
  #define NOTES_V2_REG_ALLOC(N)   ((void)0)
#endif

template <uint32_t kNumRegs>
__device__ __forceinline__ void warpgroup_reg_dealloc()
{
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(kNumRegs));
}

template <uint32_t kNumRegs>
__device__ __forceinline__ void warpgroup_reg_alloc()
{
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(kNumRegs));
}
#endif

#if defined(NOTES_V2_ENABLE_WGMMA)
// =============================================================================
// Phase 7d: HGEMM WGMMA — m64n128k16 + TMA + Warp Specialization (Hopper)
// =============================================================================
// 面试要点（WGMMA vs MMA 对比）：
//   - MMA: warp 级（32 threads），同步执行
//   - WGMMA: warpgroup 级（128 threads = 4 warps），异步执行（fire-and-forget）
//   - m64n128k16: M=64, N=128, K=16 → 一次处理 64×128×16=131K 个乘加（MMA m16n8k16的 4x16=64倍）
//   - TMA (Tensor Memory Accelerator): 硬件 DMA，2D 寻址，零寄存器开销
//   - Warp Specialization: Producer 做 TMA，Consumer 做 WGMMA，通过 barrier 同步
//   - 128B swizzle: shared memory 的 128B swizzle 模式，避免 bank conflict

// ---- WGMMA 辅助函数 ----
// wgmma.commit_group / wait_group 语义（PTX ISA §9.7.15.7）：
//   - commit_group: 将此前所有未提交的 wgmma.mma_async 归入一个新的 wgmma-group
//     （per-warpgroup，故需 .sync.aligned 确保所有线程同步执行）。
//   - wait_group N: 阻塞直到最多 N 个 wgmma-group 尚未完成（pending ≤ N）。
//     N=0 → 等所有 group 完成。★ 与 cp.async.wait_group 语义一致（PTX ISA §9.7.9.25.3.3），
//     都是 "wait until only N or fewer groups are pending"。
#define WGMMA_FENCE() asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory")
#define WGMMA_COMMIT_GROUP()                                                   \
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory")
#define WGMMA_WAIT_GROUP(n)                                                    \
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(n) : "memory")

// SMEM_DESC_ENCODE: 将 byte 偏移编码为 WGMMA descriptor 位域中的 14-bit 值。
// 编码规则（PTX ISA §9.7.15.5.1.2.2）：
//   encoded = (x & 0x3FFFF) >> 4
// 其中 0x3FFFF = 2^18 - 1 = 262143，只保留低 18 bit。
// >>4 是因为 shared memory 地址/偏移必须 16B 对齐（2^4），低 4 bit 恒为 0，
// 可省略以扩大可表示范围。
// 解码：decoded = encoded << 4（回到原始 byte 值）。
#define SMEM_DESC_ENCODE(x) ((((uint64_t)(x)) & 0x3FFFF) >> 0x4)

// make_smem_desc: 构造 WGMMA（warpgroup MMA）的 64-bit shared memory 矩阵描述符。
// 硬件 WGMMA（如 wgmma.mma_async.m64n128k16）需要 descA/descB 两个 64-bit 描述符，
// 来定位 shared memory 中的 A/B tile 并告知 swizzle 模式。
//
// 64-bit descriptor 位域布局（参见 PTX ISA §9.7.15.5.1.2.2 & CUTLASS GmmaDescriptor）：
//   ----------------------------------------------------------------------
//   | 位域                | 范围      | 大小 | 含义
//   |---------------------|-----------|------|-----------------------------
//   | start_address       | [0, 14)   | 14   | smem 基址编码
//   | (unused)            | [14, 16)  |  2   | -
//   | leading_byte_offset | [16, 30)  | 14   | 主要维度（K）的字节步长；swizzle 下硬件未使用（assumed=1）
//   | (unused)            | [30, 32)  |  2   | -
//   | stride_byte_offset  | [32, 46)  | 14   | 跨越维度（M/N）的字节步长：K-Major 下为 8-row stripe 间距
//   | (unused)            | [46, 49)  |  3   | -
//   | base_offset         | [49, 52)  |  3   | swizzle pattern 偏移
//   | (unused)            | [52, 62)  | 10   | -
//   | layout_type         | [62, 64)  |  2   | swizzle 模式
//   ----------------------------------------------------------------------
//   layout_type: 0=None, 1=128B swizzle, 2=64B swizzle, 3=32B swizzle
//
// K-Major + 128B swizzle + half(16-bit) 的几何推导（PTX ISA §9.7.15.5.1.2）：
//   - 128B swizzle atom 在 128-bit 单位下为 8×8
//   - 归一化到 half(2B) 后：atom 覆盖 8×(128/16)=64 个 K 方向元素 × 8 个 M 方向元素
//   - 即每个 atom = 64(K) × 8(M) 个 half = 64×8×2 = 1024 bytes
//   - 这是 stride_byte_offset=1024 的来源
//
//   - leading_byte_offset 在 K-Major swizzle 下 unused（hardware assumes 1），
//     此处填 16 仅为占位，编码后为 1。
//
//   - base_offset=0（bits [49,52) 未显式置位），表示 smem ptr 已 1024B 对齐。
//     若未对齐需计算 (pattern_start_addr >> 7) & 0x7。
//
// 参考：CUTLASS GmmaDescriptor (cute/arch/mma_sm90_desc.hpp)
// 注：descA/descB 共用此函数；对 B 而言物理存储为 [BN, BK]，详见 WgmmaSMem 注释。
__device__ inline uint64_t make_smem_desc(half *ptr) {
  // __cvta_generic_to_shared: 将通用地址空间中的指针转换为 shared memory 地址
  // （一个 32-bit 的 smem byte 偏移量，相对于当前 CTA 的 shared memory 基址）。
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

  // 从全零开始：所有位域初始化为 0。
  uint64_t desc = 0x0000000000000000;

  // ── bits [0, 14): start_address ──
  // SMEM_DESC_ENCODE(addr) 将 addr 右移 4 位（丢弃低 4 个 0 bit），
  // 只保留 14 位。硬件解码时左移 4 位恢复完整的 16B-aligned 地址。
  // 可寻址范围：2^(14+4) = 2^18 = 256K 个 half = 512 KB 共享内存。
  desc |= SMEM_DESC_ENCODE(addr);

  // ── bits [16, 30): leading_byte_offset ──
  // 原始值 16（字节），编码后为 (16 & 0x3FFFF) >> 4 = 1。
  // << 16 将编码值放到 bits [16, 30)。
  // K-Major + 128B swizzle 下该字段硬件未使用（assumed to be 1，参见 PTX ISA §9.7.15.5.1.2.1.1），
  // 此处填 16 仅为占位，使编码值为 1 满足硬件预期。
  // 若为 MN-Major 或 INTERLEAVE 布局，LBO 有实际含义：
  //   对 INTERLEAVE: 同一 8×2 brick 内第一列到第二列的字节偏移。
  //   对 MN-Major swizzle: 偏移从首 (swizzle-byte-size/16) 行到下一组行。
  desc |= SMEM_DESC_ENCODE((uint64_t)16) << 16;

  // ── bits [32, 46): stride_byte_offset ──
  // 原始值 1024（字节），编码后为 (1024 & 0x3FFFF) >> 4 = 64。
  // << 32 将编码值放到 bits [32, 46)。
  // K-Major 下定义为"从首 8 行到次 8 行的字节偏移"（PTX ISA §9.7.15.5.1.2.1.2）。
  // 1024 的来源（K-Major + 128B swizzle + half）：
  //   一个 128B swizzle atom 覆盖 64(K) × 8(M) 个 half。
  //   从 1 个 8-row stripe 到下一个 stripe 需要跨越 8 × 64 × 2 = 1024 bytes。
  // 若为 MN-Major: SBO 是从首列到下一列的偏移（8 列一组）。
  desc |= SMEM_DESC_ENCODE((uint64_t)1024) << 32;

  // ── bits [62, 64): layout_type (swizzle mode) ──
  // 1llu << 62  = 二进制 01 在 bits [62, 64) = layout_type = 1。
  // 1 = 128B swizzle mode。
  // 其他取值：0=None, 2=64B swizzle, 3=32B swizzle。
  // 128B swizzle 将 smem 中连续的行按 128B 粒度进行 XOR 重映射，
  // 确保同一 warp 内各线程访问不同 bank 的同一偏移时不会产生 bank conflict。
  desc |= 1llu << 62;

  return desc;
}

// ---- WGMMA PTX 指令宏 ----
// wgmma.mma_async.sync.aligned.m64n128k16.f16.f16.f16
// m64n128k16: M=64, N=128, K=16。一次 WGMMA 计算 D[64×128] += A[64×16] × B[16×128]。
// f16.f16.f16: A=f16, B=f16, D=f16（累加器也是 f16 精度）。
// 每线程 32 个 uint32 输出：D=64×128=8192 half，warpgroup 128 线程分担，
// 每线程 64 half = 32 uint32。
// descA/descB: shared memory 描述符（由 make_smem_desc 生成，64-bit 编码）。
// ScaleD: 0=清零累加器再写入, 1=与累加器中的旧值累加（K 维迭代必须用 1）。
// ScaleA/ScaleB: 1=正常符号, -1=翻转符号（此处始终用 1）。
// TransA/TransB: 0=K-major（K 维连续），1=MN-major（M/N 维连续）。本实现始终用 0。
// 参考：PTX ISA §9.7.15.4 (wgmma.mma_async)
#define WGMMA_M64N128K16_F16F16F16(d, sA, sB, ScaleD, ScaleA, ScaleB, TransA,  \
                                   TransB)                                     \
  {                                                                            \
    uint64_t desc_a = make_smem_desc(&(sA)[0]);                                \
    uint64_t desc_b = make_smem_desc(&(sB)[0]);                                \
    asm volatile(                                                              \
        "{\n"                                                                  \
        "wgmma.mma_async.sync.aligned.m64n128k16.f16.f16.f16 "                 \
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "                    \
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "                    \
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "                    \
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"                     \
        " %32,"                                                                \
        " %33,"                                                                \
        " %34, %35, %36, %37, %38;\n"                                          \
        "}\n"                                                                  \
        : "+r"((d)[0][0]), "+r"((d)[0][1]), "+r"((d)[0][2]), "+r"((d)[0][3]),  \
          "+r"((d)[1][0]), "+r"((d)[1][1]), "+r"((d)[1][2]), "+r"((d)[1][3]),  \
          "+r"((d)[2][0]), "+r"((d)[2][1]), "+r"((d)[2][2]), "+r"((d)[2][3]),  \
          "+r"((d)[3][0]), "+r"((d)[3][1]), "+r"((d)[3][2]), "+r"((d)[3][3]),  \
          "+r"((d)[4][0]), "+r"((d)[4][1]), "+r"((d)[4][2]), "+r"((d)[4][3]),  \
          "+r"((d)[5][0]), "+r"((d)[5][1]), "+r"((d)[5][2]), "+r"((d)[5][3]),  \
          "+r"((d)[6][0]), "+r"((d)[6][1]), "+r"((d)[6][2]), "+r"((d)[6][3]),  \
          "+r"((d)[7][0]), "+r"((d)[7][1]), "+r"((d)[7][2]), "+r"((d)[7][3])   \
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)),                      \
          "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)), "n"(int32_t(TransA)),    \
          "n"(int32_t(TransB)));                                               \
  }

// ---- TMA Shared Memory Layout ----
// Multi-stage pipeline: kStages 个 stage，每个 stage 存储 A[BM×BK] + B[BK×BN]
//
// 每个 stage 包含两块 smem：
//   A tile: [BM, BK] row-major → BK×BM 个 half，地址连续
//   B tile: TMA 按 [BN, BK] 物理写入（详见下方 ★），BN×BK 个 half
//
// ★ 关键区别 — WGMMA 的 B 物理存储 vs 逻辑视图：
//
//   上层数据：源矩阵 B^T [N, K] row-major（TN 布局，B 的列以行优先形式存储）。
//   物理存储：TMA 将 B^T 的 tile 写入 smem，物理布局为 [BN, BK] row-major
//            （BN=128 行，BK=64 列，每行 64 个连续的 half）。
//            TMA 的 smem_box_shape = (BK=64, BN=128) 定义了每个 tile 的 shape，
//            TMA 按行写入，行内 BK 个 half 连续 → 物理上就是 [BN][BK]。
//   逻辑视图：WGMMA 指令 (wgmma.mma_async.m64n128k16, PTX ISA §9.7.15.4)
//            通过 imm-trans-b=0 指定 B 为 K-major，即逻辑上按 [BK, BN] 寻址。
//            实际 [BN, BK] → 逻辑 [BK, BN] 的"转置"是通过 descB 中的 128B
//            swizzle (layout_type=1) 在硬件地址重映射层完成的，无需软件转置。
//
//   stride_byte_offset=1024 的来源（K-major + 128B swizzle + f16, PTX ISA §9.7.15.5.1.2.1.2）：
//     128B swizzle atom = 8(N) × 8(K) × 128-bit = 8 rows × 64 个 half。
//     stride_byte_offset = 从当前 8-row stripe 到下一个 8-row stripe 的字节偏移
//                        = 8 rows × (BK=64) halfs × 2 bytes = 1024。
//     这个值恰好等于物理 [BN, BK] 布局中连续 8 行的跨度 ← 进一步证实物理存储是 [BN, BK]。
//
//   与 MMA(TN) 的对比：
//     MMA (TN):  smem 存 B^T [N, K] row-major，ldmatrix 按列解出 col-major B fragment。
//     WGMMA:     smem 物理存 [BN, BK]（TMA 直写），WGMMA 通过 swizzle 硬件以 K-major [BK, BN] 逻辑视图读取。
//   简言之：swizzle 桥接了 "物理 row-major [BN, BK]" 和 "逻辑 K-major [BK, BN]" 的差异。
template <int BM, int BN, int BK, int QSIZE> struct WgmmaSMem {
  alignas(128) half A[BM * BK * QSIZE]; // A tile: row-major [BM, BK]
  // B tile 注记：
  //   - 物理存储：TMA 从 B^T[N,K] row-major 搬入，物理上是 [BN, BK] row-major（BN 行，BK 列）
  //   - 逻辑视图：WGMMA 通过 descB 的 128B swizzle（PTX ISA §9.7.15.5.1.2）将地址重映射，
  //     以 K-major [BK, BN] 逻辑视图供 wgmma.mma_async（imm-trans-b=0, PTX ISA §9.7.15.4）读取
  //   - B[BN * BK * QSIZE] 与 B[BK * BN * QSIZE] 数值等价（BN×BK = BK×BN），
  //     但写 BN*BK 更能体现物理存储形态
  alignas(128) half B[BN * BK * QSIZE];
};
#endif /* NOTES_V2_ENABLE_WGMMA */

#if defined(NOTES_V2_ENABLE_TMA_MMA_WS)
// SM120 不支持 WGMMA，但支持相同的 TMA 生产者协议和 warp 级 mma.sync。
// 保持 128B TMA swizzle 并显式声明物理布局供 ldmatrix 消费者使用。
template <int BM, int BN, int BK, int QSIZE> struct TmaMmaWSSMem {
  static_assert(BK == 64, "The 128B swizzle helper below is specialized for BK=64");
  half A[BM * BK * QSIZE];
  half B[BN * BK * QSIZE];
};
#endif /* NOTES_V2_ENABLE_TMA_MMA_WS */

#if defined(NOTES_V2_ENABLE_WGMMA) || defined(NOTES_V2_ENABLE_TMA_MMA_WS)
// ---- Host-side TMA Tensor Map helpers (WGMMA test) ----
// 面试要点（TMA descriptor 创建 — cuTensorMapEncodeTiled）：
//   - TMA descriptor 描述 global memory 中矩阵的 shape/stride/dtype，
//     以及硬件搬运的 box (tile) 大小和 smem swizzle 模式
//   - 关键易错点：TMA shape 参数写的是 (W,H) 而不是 (H,W)!
//     对 row-major [M,K] 矩阵，TMA shape = (K,M)，minor=K，major=M
//   - cuTensorMapEncodeTiled 是 CUDA Driver API，需 #include <cuda.h> 并链接 -lcuda
//   - smem_box 维度顺序也是 (minor, major) = (BK, BM) 或 (BK, BN)
//   - Swizzle 模式通常选 CU_TENSOR_MAP_SWIZZLE_128B 配合 WGMMA 的 128B swizzle
//
// ★ gmem_prob_stride + 1 的含义：
//   语法层面 — 数组名退化 + 指针算术：
//     gmem_prob_stride 类型为 uint64_t[5]，在表达式中退化为 uint64_t*
//     （指向首元素 gmem_prob_stride[0]）。+ 1 偏移 1 个 uint64_t，指向
//     gmem_prob_stride[1]，等价于 &gmem_prob_stride[1]。
//   语义层面 — 跳过隐式的最内维 stride：
//     cuTensorMapEncodeTiled 的 globalStrides 参数只接收 tensorRank-1 个
//     stride 值。最内维（fastest-changing dimension）的 stride 是隐式的，
//    固定等于 sizeof(element_type)，不需要显式传入。
//     对于 tensorRank=2 的 FP16 矩阵：
//       dim 0（内维，K）：stride = sizeof(half)    ← 隐式，API 不需要
//       dim 1（外维，M）：stride = sizeof(half)*K   ← 需要传给 API
//     gmem_prob_stride 数组的布局是内维在前：
//       [0] = sizeof(half)                              ← 内维，不传
//       [1] = sizeof(half) * BlockMinorSize * blocks_width ← 外维，传给 API
//   所以 + 1 的作用是跳过 [0]，让 API 从 [1] 开始读取，正好得到外维 stride。
//
// ★ smem_box_stride 为什么不需要 +1？
//   与 globalStrides 不同，smemBoxStrides 参数接收完整的 tensorRank 个 stride，
//   最内维 stride 也必须显式传入（不隐式）。
//   smem_box_stride[5] = {1, 1, 1, 1, 1} 表示 smem tile 中所有维度元素
//   都是连续存放的，维度间没有 padding/stride。
//   对于 tensorRank=2：strides[0]=1（内维 stride），strides[1]=1（外维 stride），
//   即 smem 中第 (i,j) 元素的偏移 = i*1 + j*1 = i+j（元素连续排列）。
//   所以这里直接传 smem_box_stride（即 &smem_box_stride[0]），API 会读到全部
//   两个 stride 值，不需要跳过任何一个。
//
// 参考：CUDA Programming Guide §TMA, PTX ISA §9.7.15.5, CUTLASS GmmaDescriptor

template <int BlockMajorSize, int BlockMinorSize>
__host__ static inline void create_tensor_map(CUtensorMap *tma_map,
                                              half *gmem_ptr,
                                              int blocks_height,
                                              int blocks_width) {
  void *gmem_address = (void *)gmem_ptr;
  uint64_t gmem_prob_shape[5] = {(uint64_t)BlockMinorSize * blocks_width,
                                 (uint64_t)BlockMajorSize * blocks_height,
                                  1, 1, 1};
  uint64_t gmem_prob_stride[5] = {
      sizeof(half), sizeof(half) * BlockMinorSize * blocks_width, 0, 0, 0};
  uint32_t smem_box_shape[5] = {uint32_t(BlockMinorSize),
                                uint32_t(BlockMajorSize), 1, 1, 1};
  uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};
  CUresult result = cuTensorMapEncodeTiled(
      tma_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, gmem_address,
      gmem_prob_shape, gmem_prob_stride + 1, smem_box_shape, smem_box_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (result != CUDA_SUCCESS)
    printf("cuTensorMapEncodeTiled failed: %d\n", (int)result);
}

// BlockMajorSize = major dim (outer), BlockMinorSize = minor dim (inner, contiguous).
// For row-major [Major, Minor] matrices.
// Default <128, 64> matches HGEMM A/B^T and FA Q; FA K/V use <64, 64>.
template <int BlockMajorSize = 128, int BlockMinorSize = 64>
__host__ static inline CUtensorMap *allocate_and_create_tensor_map(
    half *src, int blocks_height, int blocks_width) {
  CUtensorMap *tma_map_d;
  cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
  CUtensorMap tma_map_host;
  create_tensor_map<BlockMajorSize, BlockMinorSize>(&tma_map_host, src,
                                                     blocks_height,
                                                     blocks_width);
  cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap),
             cudaMemcpyHostToDevice);
  return tma_map_d;
}
#endif /* NOTES_V2_ENABLE_WGMMA || NOTES_V2_ENABLE_TMA_MMA_WS */
