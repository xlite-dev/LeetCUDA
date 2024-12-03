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

HOST_DEVICE_INLINE 
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

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
  static_assert(kStage > 0 && kStage < 3); static_assert(kPad >= 0);
  constexpr int d  = kHeadDim; // alias
  constexpr int Br = kMmaQP * kMmaTileQP * kWarpTileQP; // 16*2*2=64
  constexpr int Bc = kMmaKV * kMmaTileKV * kWarpTileKV; // 8*4*2=64
  constexpr int Bd = kMmaHeadDim; // 16, tile head_dim(d) according MMA
  constexpr int Tn = WARP_SIZE * kMmaTileQP * kMmaTileKV; // 32*2*4=256
  const int Tr = div_ceil(N, Br); // Tr Q_tile[Br,d]
  const int Tc = div_ceil(N, Bc); // Tc K/V_tile[Bc,d]
  const int Td = div_ceil(d, Bd); // Td K_tile_d[Bc,Bd], e.g [64,16]
  const float scale = 1.0 / sqrt((float)d);
  
  // grid(batch, head_num, N/Br=Tr), block(256=8*mma or 128=4*mma)
  const int QKV_batch_id = blockIdx.x; // B, bx
  const int  QKV_head_id = blockIdx.y; // H, by
  const int   QO_tile_id = blockIdx.z; // Q/O_tile_id, range [0, Tr), bz  
  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  // gridDim.y = head_num, gridDim.z = N/Br = Tr.
  const int KV_gmem_offset = ((QKV_batch_id * gridDim.y * N * d) + (QKV_head_id * N * d)); 
  const int QO_gmem_offset = ((QKV_batch_id * gridDim.y * N * d) + (QKV_head_id * N * d));
  
  // Shared memory for Q,K,V,O, d=64->24M, d=128=48M
  extern __shared__ half smem[];
  constexpr int QO_tile_size = Br * (d + kPad); // 64*64=4096, ~8192 bytes=8M, Q+O=16M
  constexpr int KV_tile_size = Bc * (d + kPad); // 64*64=4096, ~8192 bytes=8M, KV shared 8M
  // Only apply multi stages for K/V across N(seq_len) dim.
  half* Q_tile_smem = smem; // 8M/16M
  half* K_tile_smem = Q_tile_smem + QO_tile_size; // 8M/16M
  half* V_tile_smem = K_tile_smem; // KV shared same smem 8M/16M
  // Allocate smem buffer O_tile[Br,d] for collective store.
  half* O_tile_smem = Q_tile_smem + kStage * KV_tile_size; // 8M/16M
 
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
  int load_gmem_K_n = 0; int load_gmem_V_n = 0; 

  // registers for S_tile[Br,N] = Q_tile[Br,d] * K[N,d], tile head_dim with Bd=16
  // loop over N dim, S_tile_iter[Br,Bc]=[64,64], each thread hold 2x32 bits regs.
  // S, P and O shared the same registers.
  uint32_t R_SPO[kWarpTileQP][kWarpTileKV][2]; // [2][2][2]
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      R_SPO[i][j][0] = 0;
      R_SPO[i][j][1] = 0;
    }
  }

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
      smem_Q_base_ptr + (load_smem_Q_n * (Br + kPad) + 
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
  for (int tile_n = 0; tile_n < (kStage - 1); ++k) {
    // update the offset of n according to stages
    load_gmem_K_n += tile_n * Bc;
    int load_gmem_K_d = load_smem_K_d;
    int load_gmem_K_addr = (
      KV_gmem_offset + load_gmem_K_n * d + load_gmem_K_d);
     uint32_t load_smem_K_ptr = (
      smem_K_base_ptr + (tile_n * KV_tile_size + 
                         load_smem_K_n * (Bc + kPad) + 
                         load_smem_K_d) * sizeof(half)
    );
    // load d / (Tn / Bc) vals, 64 or 128 div 4, 16 or 32, 
    // need 2 or 4 128 bits memory issues.
    #pragma unroll
    for (int i = 0; i < (d / (Tn / Bc)); i += 8) {
      CP_ASYNC_CG(load_smem_K_ptr + i * sizeof(half), 
                  &K[load_smem_K_ptr + i], 16);
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

  // registers for Q, K(V reuse)
  uint32_t R_QP[kWarpTileQP][4];
  uint32_t R_KV[kWarpTileKV][2];

  // WIP: compute S = Q @ K^T
  // loop over N: for K[N,d] with K_tile[Bc,d]
  #pragma unroll
  for (int tile_n = (kStage - 1); tile_n < Tc; ++tile_n) {
    // multi stages pipeling gmem -> smem
    int smem_sel = (tile_n + 1) % kStage; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = tile_n % kStage;  // s3 k 2->2, k 3->0, k 4->1...

    load_gmem_K_n += tile_n * Bc;
    int load_gmem_K_d = load_smem_K_d;
    int load_gmem_K_addr = (
      KV_gmem_offset + load_gmem_K_n * d + load_gmem_K_d);
     uint32_t load_smem_K_ptr = (
      smem_K_base_ptr + (smem_sel_next * KV_tile_size + 
                         load_smem_K_n * (Bc + kPad) + 
                         load_smem_K_d) * sizeof(half)
    );
    // load d / (Tn / Bc) vals, 64 or 128 div 4, 16 or 32, 
    // need 2 or 4 128 bits memory issues.
    #pragma unroll
    for (int i = 0; i < (d / (Tn / Bc)); i += 8) {
      CP_ASYNC_CG(load_smem_K_ptr + i * sizeof(half), 
                  &K[load_smem_K_ptr + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP();

    // loop over d: Bd=16, K_tile_d[Bc,Bd]
    #pragma unroll
    for (int tile_d = 0; tile_d < Td; ++tile_d) {
      // smem -> reg
      // ldmatrix.x4 for Q_tile_smem, ldmatrix.x2 for K_tile_smem
      #pragma unroll
      for (int i = 0; i < kWarpTileQP; ++i) {
        // LDMATRIX_X4
        
      }

      #pragma unroll
      for (int j = 0; j < kWarpTileKV; ++i) {
        // LDMATRIX_X2
        
      }


    }

    
  }
   

}


// TODO: only tile Bd for d dim, slice d
// flash_attn_mma_slice_d_kernel
