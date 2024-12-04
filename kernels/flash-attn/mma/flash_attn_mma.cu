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
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

template<typename T, const int kWarpSize = WARP_SIZE>
__device__ inline T warp_reduce_max(T val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    T val_compare = __shfl_xor_sync(0xffffffff, val, mask);
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

template<const int kWarpTileQP, const int kWarpTileKV>
__device__ inline void fill_SPO_regs(
  uint32_t (&R_SPO)[kWarpTileQP][kWarpTileKV][2], 
  const uint32_t val = 0) {
  #pragma unroll
  for (int i = 0; i < kWarpTileQP; ++i) {
    #pragma unroll
    for (int j = 0; j < kWarpTileKV; ++j) {
      R_SPO[i][j][0] = val;
      R_SPO[i][j][1] = val;
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
// launch: grid(batch, head_num, N/Br=Tr), block(256=8*mma or 128=4*mma)
// TODO: may return lse=logsumexp[Br].
template<
         const int kHeadDim,    // 32,64,128     
         const int kMmaQP,      // M 16
         const int kMmaKV,      // N 8
         const int kMmaHeadDim, // K 16
         const int kMmaTileQP,  // 2    
         const int kMmaTileKV,  // 4 
         const int kWarpTileQP, // 2
         const int kWarpTileKV, // 2
         const int kStage,      // 1,2
         const int kPad,        // 0,8,16
         >
__global__  void flash_attn_mma_kernel(
  half* Q, half* K, half* V,  half* O, int N) {
  // step 0: S_tile[Br,N] = Q_tile[Br,d] * K[N,d], slice-k manner matmul
  // across K's N dim, each K_tile/V_tile inner loop has shape [Bc,d].
  // step 1: P_tile[Br,N] = softmax(S_tile[Br,N]), row wise.
  // step 2: O_tile[Br,d] = P_tile[Br,N] * V[N,d], matmul.
  static_assert(kHeadDim % 32 == 0); // may relax for 16 ?
  static_assert(kMmaQP == 16 && kMmaKV == 8 && kMmaHeadDim == 16); // m16n8k16
  static_assert(kMmaTileQP  == 2 && kMmaTileKV  == 4);
  static_assert(kWarpTileQP == 2 && kWarpTileKV == 2);
  static_assert(kStage > 0 && kStage < 3); // 1,2
  static_assert(kPad >= 0 && kPad % 8 == 0); // 0,8,16
  constexpr int d  = kHeadDim; // alias
  constexpr int Br = kMmaQP * kMmaTileQP * kWarpTileQP; // 16*2*2=64
  constexpr int Bc = kMmaKV * kMmaTileKV * kWarpTileKV; // 8*4*2=64
  constexpr int Bd = kMmaHeadDim; // 16, tile head_dim(d) according MMA
  constexpr int Tn = WARP_SIZE * kMmaTileQP * kMmaTileKV; // 32*2*4=256
  // NOTE: Now, N must be mutliples of Bc(32/64) for KV tiling across N.
  const int Tr = div_ceil(N, Br); // Tr Q_tile[Br,d]
  const int Tc = div_ceil(N, Bc); // Tc K/V_tile[Bc,d]
  const int Td = div_ceil(d, Bd); // Td K_tile_d[Bc,Bd], e.g [64,16]
  const float scale = 1.0 / sqrt((float)d);
  
  // grid(batch, head_num, N/Br=Tr), block(256=8*mma or 128=4*mma)
  const int QKV_batch_id = blockIdx.x; // B, bx
  const int QKV_head_id  = blockIdx.y; // H, by
  const int QO_tile_id   = blockIdx.z; // Q/O_tile_id, range [0, Tr), bz  
  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_QP = warp_id % 2; // 0,1
  const int warp_KV = warp_id / 2; // 0,1,2,3
  // The layout of 8 MMA(2x4) [before] kWarpTileQPxkWarpTileKV(2x2) -> 16x2,8x4=32x32:
  // |  [32,32]  | warp_KV 0 | warp_KV 1 | warp_KV 2 | warp_KV 3 |
  // | warp_QP 0 |-- MMA 0 --|-- MMA 2 --|-- MMA 4 --|-- MMA 6 --|
  // | warp_QP 1 |-- MMA 1 --|-- MMA 3 --|-- MMA 5 --|-- MMA 7 --|
  // The layout of 8 MMA(2x4)  [after] kWarpTileQPxkWarpTileKV(2x2) -> 32x2,32x2=64x64: 
  // |  [64,64]  |    warp_KV 0    |    warp_KV 1    |    warp_KV 2    |    warp_KV 3    |
  // | warp_QP 0 |-- MMA 0,MMA 0 --|-- MMA 2,MMA 2 --|-- MMA 4,MMA 4 --|-- MMA 6,MMA 6 --|
  // | warp_QP 0 |-- MMA 0,MMA 0 --|-- MMA 2,MMA 2 --|-- MMA 4,MMA 4 --|-- MMA 6,MMA 6 --|
  // | warp_QP 1 |-- MMA 1,MMA 1 --|-- MMA 3,MMA 2 --|-- MMA 5,MMA 5 --|-- MMA 7,MMA 7 --|
  // | warp_QP 1 |-- MMA 1,MMA 1 --|-- MMA 3,MMA 2 --|-- MMA 5,MMA 5 --|-- MMA 7,MMA 7 --|
  // gridDim.y = head_num, gridDim.z = N/Br = Tr.
  const int KV_gmem_offset = ((QKV_batch_id * gridDim.y * N * d) + (QKV_head_id * N * d)); 
  const int QO_gmem_offset = ((QKV_batch_id * gridDim.y * N * d) + (QKV_head_id * N * d));
  
  // Shared memory for Q,K,V,O, d=64->24M, d=128=48M
  extern __shared__ half smem[];
  constexpr int QO_tile_size = Br * (d + kPad); // 64*64=4096, ~8192 bytes=8M
  constexpr int KV_tile_size = Bc * (d + kPad); // 64*64=4096, ~8192 bytes=8M, KV may shared 8M
  // Only apply multi stages for K across N(seq_len) dim, not for Q,V.
  half* Q_tile_smem = smem; // 8M/16M
  half* K_tile_smem = Q_tile_smem + QO_tile_size; // 8M/16M
  half* V_tile_smem = K_tile_smem + kStage * KV_tile_size; // no shared smem for KV
  // TODO: KV may shared same smem to reduce smem usage for headdim>=256
  // half* V_tile_smem = K_tile_smem; // KV may shared same smem 8M/16M
  // stage 2, no shared KV smem, Br=Bc=64,  d=64: 8M+(8M)*2+8M   =32M,  shared KV smem: 24M
  // stage 2, no shared KV smem, Br=Bc=64, d=128: 16M+(16M)*2+16M=64M,  shared KV smem: 48M
  // stage 2, no shared KV smem, Br=Bc=64, d=256: 32M+(32M)*2+32M=128M, shared KV smem: 96M
  // stage 1, no shared KV smem, Br=Bc=64, d=256: 32M+(32M)*1+32M=96M,  shared KV smem: 64M
 
  // Mapping gmem -> tid -> smem, Q[Br,d]=[64,64 or 128], 256 threads.
  int load_smem_Q_n = (tid / (Tn / Br)); // Br 64, tid / 4, row 0~64
  int load_smem_Q_d = (tid % (Tn / Br)) * (d / (Tn / Br)); // (tid % 4) * 16, 0,16,32,48
  int load_smem_K_n = (tid / (Tn / Bc)); // Bc 64, tid / 4, row 0~64
  int load_smem_K_d = (tid % (Tn / Bc)) * (d / (Tn / Bc)); // (tid % 4) * 16, 0,16,32,48
  int load_smem_V_n = load_smem_K_n;
  int load_smem_V_d = load_smem_K_d;
  // global Q row of current head with tile [Br,d] per block.
  int load_gmem_Q_n = QO_tile_id * Br + load_smem_Q_n; 
  if (load_gmem_Q_n >= N) return;
  // KV tile gmem load index starts from 0 and increments with 
  // each iteration as we loop over N.
  int load_gmem_K_n = 0; 
  int load_gmem_V_n = 0; 

  uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
  uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
  uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);
  uint32_t smem_O_base_ptr = __cvta_generic_to_shared(O_tile_smem);

  // load Q from gmem -> smem, only load once.
  {
    int load_gmem_Q_d = load_smem_Q_d;
    int load_gmem_Q_addr = (
      QO_gmem_offset + load_gmem_Q_n * d + load_gmem_Q_d);
    uint32_t load_smem_Q_ptr = (
      smem_Q_base_ptr + (load_smem_Q_n * (d + kPad) + 
                         load_smem_Q_d) * sizeof(half)
    );
    // load d / (Tn / Br) vals, 64 or 128 div 4, 16 or 32, 
    // need 2 or 4 128 bits memory issues.
    #pragma unroll
    for (int i = 0; i < (d / (Tn / Br)); i += 8) {
      CP_ASYNC_CG(load_smem_Q_ptr + i * sizeof(half), 
                  &Q[load_gmem_Q_addr + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP();
  }

  // load K from gmem -> smem, (kStage - 1) K tiles, [Bc,d]
  #pragma unroll
  for (int stage = 0; stage < (kStage - 1); ++stage) {
    // update the offset of n according to stages
    load_gmem_K_n += stage * Bc; // s2, +offset 0
    int load_gmem_K_d = load_smem_K_d;
    int load_gmem_K_addr = (
      KV_gmem_offset + load_gmem_K_n * d + load_gmem_K_d);
     uint32_t load_smem_K_ptr = (
      smem_K_base_ptr + (stage * KV_tile_size + 
                         load_smem_K_n * (d + kPad) + 
                         load_smem_K_d) * sizeof(half)
    );
    // load d / (Tn / Bc) vals, 64 or 128 div 4, 16 or 32, 
    // need 2 or 4 128 bits memory issues.
    #pragma unroll
    for (int i = 0; i < (d / (Tn / Bc)); i += 8) {
      CP_ASYNC_CG(load_smem_K_ptr + i * sizeof(half), 
                  &K[load_gmem_K_addr + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP();
  }

  // wait Q and at least (kStage - 1) for K ready.
  if constexpr (kStage - 2 >= 0) {
    CP_ASYNC_WAIT_GROUP(kStage - 2); // s2->0, s3->1, s4->2
  } else {
    CP_ASYNC_WAIT_GROUP(0);
  }
  __syncthreads(); 

  // NOTE: Init registers/smem for m_i[Br], l_i[Br] and O_i[Br,d] ?
  // or perform as each thread keep one part of m_i, because we will 
  // keep two 32 bits each thread for S/P.

  // m_old, l_old, may use float to keep precision ?
  half thread_max_old[2] = {-INFHALF, -INFHALF}; 
  half thread_sum_old[2] = {ZEROHALF, ZEROHALF};

  // <loop over N>: for K[N,d] with K_tile[Bc,d]
  // tile_n: compute S_tile[Br,Bc] = Q @ K^T = Q_tile[Br,d] * K[Bc,d]
  #pragma unroll
  for (int tile_n = 0; tile_n < Tc; ++tile_n) { 
    // TODO: process last tile_n ? pad to multiple of 8.

    // s2 tn 0->0, 1->1, 2->0; s3 tn 0->0, 1->1, 2->2, 3->0;
    int smem_sel      = (tile_n) % kStage;   
    // s2 tn 0->1, 1->0, 2->1; s3 tn 0->2, 1->0, 2->1, 3->2;  
    int smem_sel_next = (tile_n + (kStage - 1)) % kStage;
    // multi stages pipeling gmem -> smem
    // NOTE: kStage must be > 1 for pipeling. For s1, smem_sel 
    // and smem_sel_next will always equal 0, thus, we can not 
    // prefetch KV from gmem to smem before tile_n MMA done.

    // Prefetch curr V tile_n (no stages)
    {
      load_gmem_V_n += tile_n * Bc;
      int load_gmem_V_d = load_smem_V_d;
      int load_gmem_V_addr = (
        KV_gmem_offset + load_gmem_V_n * d + load_gmem_V_d);
      uint32_t load_smem_V_ptr = (
        smem_V_base_ptr + (load_smem_V_n * (d + kPad) + 
                           load_smem_V_d) * sizeof(half)
      );
      // load d / (Tn / Bc) vals, 64 or 128 div 4, 16 or 32, 
      // need 2 or 4 128 bits memory issues.
      #pragma unroll
      for (int i = 0; i < (d / (Tn / Bc)); i += 8) {
        CP_ASYNC_CG(load_smem_V_ptr + i * sizeof(half), 
                    &K[load_gmem_V_addr + i], 16);
      }
      CP_ASYNC_COMMIT_GROUP();
    }

    // Prefetch next stage K (tile_n + 1)
    if constexpr (kStage > 1) {
      if ((tile_n + 1) < Tc) {
        load_gmem_K_n += (tile_n + 1) * Bc;
        int load_gmem_K_d = load_smem_K_d;
        int load_gmem_K_addr = (
          KV_gmem_offset + load_gmem_K_n * d + load_gmem_K_d);
        uint32_t load_smem_K_ptr = (
          smem_K_base_ptr + (smem_sel_next * KV_tile_size + 
                             load_smem_K_n * (d + kPad) + 
                             load_smem_K_d) * sizeof(half)
        );
        // load d / (Tn / Bc) vals, 64 or 128 div 4, 16 or 32, 
        // need 2 or 4 128 bits memory issues.
        #pragma unroll
        for (int i = 0; i < (d / (Tn / Bc)); i += 8) {
          CP_ASYNC_CG(load_smem_K_ptr + i * sizeof(half), 
                      &K[load_gmem_K_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
      } else {
        // wait all memory issues ready for last tile.
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads(); 
      }
    }
    
    // registers for current tile_n within <loop over N>, 
    // [64,64] = S_tile[Br,Bc] = Q_tile[Br,d] * K[Bc,d]
    // each thread hold 2x32 bits regs. S,P,O may shared 
    // the same registers.
    uint32_t R_SPO[kWarpTileQP][kWarpTileKV][2]; // [2][2][2]
    fill_SPO_regs<kWarpTileQP, kWarpTileKV>(R_SPO, 0);

    // registers for Q, K(V reuse)
    uint32_t R_QP[kWarpTileQP][4];
    uint32_t R_KV[kWarpTileKV][2];
    
    // <loop over d>: tile_d, Bd = 16, K_tile_d[Bc,Bd]
    #pragma unroll
    for (int tile_d = 0; tile_d < Td; ++tile_d) {
      // offset d according tile_d
      // smem -> reg, load smem Q
      // ldmatrix.x4 for Q_tile_smem, ldmatrix.x2 for K_tile_smem
      #pragma unroll
      for (int i = 0; i < kWarpTileQP; ++i) {
        int warp_smem_Q_n = warp_QP * (kMmaQP * kWarpTileQP) + i * kMmaQP;
        int lane_smem_Q_n = warp_smem_Q_n + lane_id % 16; // 0~15
        int lane_smem_Q_d = tile_d * Bd + (lane_id / 16) * 8; // 0,8
        uint32_t lane_smem_Q_ptr = (
            smem_Q_base_ptr + (lane_smem_Q_n * (d + kPad) + 
                               lane_smem_Q_d) * sizeof(half)
        );
        LDMATRIX_X4(R_QP[i][0], R_QP[i][1], R_QP[i][2], R_QP[i][3], 
                    lane_smem_Q_ptr); // R_Q
      }

      #pragma unroll
      for (int j = 0; j < kWarpTileKV; ++j) {
        int warp_smem_K_n = warp_KV * (kMmaKV * kWarpTileKV) + j * kMmaKV;
        int lane_smem_K_n = warp_smem_K_n + lane_id % 8; // 0~7, MMA_N=8
        int lane_smem_K_d = tile_d * Bd + ((lane_id / 8) % 2) * 8; // 0,8
        uint32_t lane_smem_K_ptr = (
            smem_K_base_ptr + (smem_sel * KV_tile_size + 
                               warp_smem_K_n * (d + kPad) + 
                               lane_smem_K_d) * sizeof(half)
        );
        LDMATRIX_X2(R_KV[j][0], R_KV[j][1], lane_smem_K_ptr); // R_K
      }

      // MMA compute
      #pragma unroll
      for (int i = 0; i < kWarpTileQP; ++i) {
        #pragma unroll
        for (int j = 0; j < kWarpTileKV; ++j) {
          HMMA16816(R_SPO[i][j][0], R_SPO[i][j][1], 
                    R_QP[i][0],     R_QP[i][1],    R_QP[i][2], R_QP[i][3], 
                    R_KV[j][0],     R_KV[j][1], 
                    R_SPO[i][j][0], R_SPO[i][j][1]);
        }
      }
    } // end loop over d

    // TODO: May reuse K smem for V, for example, stages 2, stage
    // 0 K smem can be reuse as V smem 0 because we do not need 
    // K values on stage 0 K smem anymore.

    // Now, we got a computed tile of S[Br,N], tile with shape [Br,Bc].
    // Assume [Br, Bc] = [64, 64] = 64x64 = 4096 values. Each thread holds
    // a portion of this [Br, Bc] block, specifically, R_S = R_SPO[2][2][2]. 
    // This means that each Warp (MMA) repeats 2 times in the N direction 
    // for both Q and K, resulting in 2x2 = 4 sets of MMA results. Each set 
    // of results is stored in 2 32-bit registers, with each register holding 
    // 2 half-precision values. In other words, each thread stores (4x2)x2 = 16 
    // half-precision values. With a total of 256 threads, the total number of 
    // half-precision values is 256x16 = 4096, which exactly matches the total 
    // [Br, Bc] = [64, 64] values.
    // reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
    // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
    // The layout of 8 MMA m16n8k16 (2x4)  [after] kWarpTileQPxkWarpTileKV(2x2) -> 32x2,32x2=64x64: 
    // |  [64,64]  |    warp_KV 0    |    warp_KV 1    |    warp_KV 2    |    warp_KV 3    |
    // | warp_QP 0 |-- MMA 0,MMA 0 --|-- MMA 2,MMA 2 --|-- MMA 4,MMA 4 --|-- MMA 6,MMA 6 --| row max
    // | warp_QP 0 |-- MMA 0,MMA 0 --|-- MMA 2,MMA 2 --|-- MMA 4,MMA 4 --|-- MMA 6,MMA 6 --| row max
    // | warp_QP 1 |-- MMA 1,MMA 1 --|-- MMA 3,MMA 2 --|-- MMA 5,MMA 5 --|-- MMA 7,MMA 7 --| row max
    // | warp_QP 1 |-- MMA 1,MMA 1 --|-- MMA 3,MMA 2 --|-- MMA 5,MMA 5 --|-- MMA 7,MMA 7 --| row max
    // TODO: online safe softmax, warp/block reduce max/sum, row wise
    // m, l, may use float to keep precision ? rowmax总共有Br=64个值
    // 首先，对于每一个MMA持有的结果计算warp row max
    // warp 0/2/4/6 包含了前[Br/2=32,Bc=64]的值，因此需要前32个rowmax值
    // warp 1/3/5/7 包含了后[Br/2=32,Bc=64]的值，因此需要后32个rowmax值
    // 一个warp=32线程，刚好每个线程保存一个max
    // half lane_row_max = -INFHALF; // m, 第i个lane保存第i行的max, 32行
    // half lane_row_sum = ZEROHALF; // l, 第i个lane保存第i行的sum, 32行
    // 每个warp(MMA)处理(16x2)*(8x2)=32x16的大小，row=32, col=16
    #pragma unroll
    for (int i = 0; i < kWarpTileQP; ++i) {
      float lane_max[2] = {-INFINITY, -INFINITY};
      #pragma unroll
      for (int j = 0; j < kWarpTileKV; ++j) {
        // 聚焦到一次MMA的结果上 m16n8
        half2 t_reg_0 = HALF2(R_SPO[i][j][0]); // 0~7  {c0, c1}
        half2 t_reg_1 = HALF2(R_SPO[i][j][1]); // 8~15 {c2, c3}
        float tmp_max_0 = max(__half2float(t_reg_0.x), __half2float(t_reg_0.y));
        float tmp_max_1 = max(__half2float(t_reg_1.x), __half2float(t_reg_1.y));
        lane_max[0] = max(lane_tile_max[0], tmp_max_0);
        lane_max[1] = max(lane_tile_max[1], tmp_max_1);
      }
    }

    // Here, we have to wait V ready before compute O = P @ V
    if constexpr (kStage == 2) {
      // NOTE: we have send V mem issues before K
      CP_ASYNC_WAIT_GROUP(1); // s1->-1, s2->0, s3->1, s4->2
    } else {
      CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads(); 

    // NOTE: After compute P @ V, we have to wait next K tile ready in smem.
    // do not need to wait any things if kStage == 1.
    if constexpr (kStage == 2) {
      CP_ASYNC_WAIT_GROUP(0);
      __syncthreads(); 
    }

  } // end loop over N
   

}
