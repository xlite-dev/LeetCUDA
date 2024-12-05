#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
using namespace nvcuda;

#define WARP_SIZE 32
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2*>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
// gmem -> smem
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
// smem -> gmem: requires sm_90 or higher.
#define CP_ASYNC_BULK_COMMIT_GROUP() asm volatile("cp.async.bulk.commit_group;\n" ::)
#define CP_ASYNC_BULK_WAIT_ALL() asm volatile("cp.async.bulk.wait_all;\n" ::)
#define CP_ASYNC_BULK_WAIT_GROUP(n) asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_BULK(dst, src, bytes) asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
// ldmatrix
#define LDMATRIX_X1(R, addr) asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))
#define LDMATRIX_X2(R0, R1, addr) asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))
#define LDMATRIX_X1_T(R, addr) asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr) asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))
// stmatrix: requires sm_90 or higher.
#define STMATRIX_X1(addr, R) asm volatile("stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};\n" :: "r"(addr), "r"(R))
#define STMATRIX_X2(addr, R0, R1) asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n" :: "r"(addr), "r"(R0), "r"(R1))
#define STMATRIX_X4(addr, R0, R1, R2, R3) asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(addr), "r"(R0), "r"(R1), "r"(R2), "r"(R3))
#define STMATRIX_X1_T(addr, R) asm volatile("stmatrix.sync.aligned.x1.trans.m8n8.shared.b16 [%0], {%1};\n" :: "r"(addr), "r"(R))
#define STMATRIX_X2_T(addr, R0, R1) asm volatile("stmatrix.sync.aligned.x2.trans.m8n8.shared.b16 [%0], {%1, %2};\n" :: "r"(addr), "r"(R0), "r"(R1))
#define STMATRIX_X4_T(addr, R0, R1, R2, R3) asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(addr), "r"(R0), "r"(R1), "r"(R2), "r"(R3))
// mma m16n8k16
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" : "=r"(RD0), "=r"(RD1) : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

__device__ inline int div_ceil(int a, int b) { 
  return (a % b != 0) ? (a / b + 1) : (a / b); 
}

template<typename T, const int kWarpSize = WARP_SIZE>
__device__ inline T warp_reduce_sum(T val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask, kWarpSize);
  }
  return val;
}

template<typename T, const int kWarpSize = WARP_SIZE>
__device__ inline T warp_reduce_max(T val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    T val_compare = __shfl_xor_sync(0xffffffff, val, mask, kWarpSize);
    val = val > val_compare ? val : val_compare;
  }
  return val;
}

template<typename T, const int kNumThreads = 256, const int kWarpSize = WARP_SIZE>
__device__ T block_reduce_sum(T val) {
  static_assert(kWarpSize == 32, "only support warp size = 32.");
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int kNumWarps = (kNumThreads + kWarpSize - 1) / kWarpSize;
  int warp = threadIdx.x / kWarpSize;
  int lane = threadIdx.x % kWarpSize;
  static __shared__ T shared[kNumWarps];
  
  T value = warp_reduce_sum<T, kWarpSize>(val);
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  value = (lane < kNumWarps) ? shared[lane] : 0.0f;
  value = warp_reduce_sum<T, kNumWarps>(value);  
  // WRAN: need to broadcast value to all threads within warp
  value = __shfl_sync(0xffffffff, value, 0);
  return value;
}

template<typename T, const int kNumThreads = 256, const int kWarpSize = WARP_SIZE>
__device__ T block_reduce_max(T val) {
  static_assert(kWarpSize == 32, "only support warp size = 32.");
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int kNumWarps = (kNumThreads + kWarpSize - 1) / kWarpSize;
  int warp = threadIdx.x / kWarpSize;
  int lane = threadIdx.x % kWarpSize;
  static __shared__ T shared[kNumWarps];
  
  T value = warp_reduce_max<T, kWarpSize>(val);
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  value = (lane < kNumWarps) ? shared[lane] : -FLT_MAX;
  value = warp_reduce_max<T, kNumWarps>(value);
  // WRAN: need to broadcast value to all threads within warp
  value = __shfl_sync(0xffffffff, value, 0);
  return value;
}

template<const int M, const int N>
__device__ inline void fill_SPO_regs(uint32_t (&R_SP)[M][N][2], uint32_t val) {
  #pragma unroll
  for (int i = 0; i < M; ++i) {
    #pragma unroll
    for (int j = 0; j < N; ++j) {
      R_SP[i][j][0] = val;
      R_SP[i][j][1] = val;
    }
  }
}

#define INFHALF  = __float2half(65536.0f)
#define ZEROHALF = __float2half(0.0f)

// Write FlashAttention-2 from scratch using Tensor Cores with MMA PTX instruction.
// The input is Q,K,V, 4D tensor with shape [batch_size, num_heads, seq_len, head_dim].
// The output is O, a 4D tensor with shape [batch_size, num_heads, seq_len, head_dim].

// The FlashAttention-2 algorithm is described in the following paper:
// https://arxiv.org/abs/2110.08210

// Q,K,V,O: [batch_size, num_heads, seq_len, head_dim], [B,H,N,d]
// each block processes Q_tile with shape [Br,d] and full K,V with shape [N,d]
// Br or Bc = 64,128,256, etc.

// [64,64], m16n8k16, mma2x4, warp2x2(32,16,16)
// (32x2,16x4,16)=(64,64,16), 256 threads, 8 warps.
// default: Br=128|64, Bc=128|64, d=64|128, kStage=2, kPad=0
// tiling: Q_tile[Br,d]=[128,64], K/V_tile[Bc,d]=[128,64]
// outputs: O_tile[Br,d], lse=logsumexp[Br] per thread block.
// iteration: loop over N for K/V with K/V_tile[Bc,d], Tc iters.
// launch: grid(batch, head_num, N/Br=Tr), block(256=8*mma)
// TODO: may return lse=logsumexp[Br].

// Tiling headdim with Bd=16, SRAM complexity is O((Br+Bc)*16) vs O((Br+Bc)*d)
