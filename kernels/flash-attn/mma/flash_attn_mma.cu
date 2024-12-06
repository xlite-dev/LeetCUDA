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
#define HMMA16816F32(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1, RC2, RC3) asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,  %1,  %2,  %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" : "=r"(RD0), "=r"(RD1), "=r"(RD2), "=r"(RD3): "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1), "r"(RC2), "r"(RC3))


HOST_DEVICE_INLINE 
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }


template<typename T, const int kWarpSize = WARP_SIZE>
DEVICE_INLINE T warp_reduce_sum(T val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}


template<typename T, const int kWarpSize = WARP_SIZE>
DEVICE_INLINE T warp_reduce_max(T val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    T val_compare = __shfl_xor_sync(0xffffffff, val, mask);
    val = val > val_compare ? val : val_compare;
  }
  return val;
}


template<typename T, int M, const int N, const int K = 2>
DEVICE_INLINE void fill_3D_regs(T (&R)[M][N][K], T val) {
  #pragma unroll
  for (int i = 0; i < M; ++i) {
    #pragma unroll
    for (int j = 0; j < N; ++j) {
      #pragma unroll
      for (int k = 0; k < K; ++k) {
        R[i][j][k] = val;
      }
    }
  }
}


template<typename T, int M, const int N = 2>
DEVICE_INLINE void fill_2D_regs(T (&R)[M][N], T val) {
  #pragma unroll
  for (int i = 0; i < M; ++i) {
    #pragma unroll
    for (int j = 0; j < N; ++j) {
      R[i][j] = val;
    }
  }
}

#define INFHALF __float2half(65536.0f)
#define ZEROHALF __float2half(0.0f)

#define FLASH_ATTN_MMA_DEBUG

#ifdef FLASH_ATTN_MMA_DEBUG
#define FA_MMA_PRINT_T0_REG(R, format, ...)      \
{                                                \
  if (tid == 0) {                                \
    float2 v_reg = __half22float2(HALF2(R));     \
    printf("[T0] ");                             \
    printf(format, ##__VA_ARGS__);               \
    printf(", V0=%f, V1=%f\n", v_reg.x, v_reg.y);\
  }                                              \
}
#define FA_MMA_PRINT_REG(R, format, ...)      \
{                                                \
  {                                              \
    float2 v_reg = __half22float2(HALF2(R));     \
    printf(format, ##__VA_ARGS__);               \
    printf("V0=%f, V1=%f\n", v_reg.x, v_reg.y);\
  }                                              \
}
#define FA_MMA_PRINT_T0_REG_V2(R, format, ...)   \
{                                                \
  if (tid == 0) {                                \
    printf("[T0] ");                             \
    printf(format, ##__VA_ARGS__);               \
    printf(", V0=%f, V1=%f\n", (R).x, (R).y);    \
  }                                              \
}
#define FA_MMA_PRINT_T0(format, ...)            \
{                                               \
  if (tid == 0) {                               \
    printf("[T0] ");                            \
    printf(format, ##__VA_ARGS__);              \
  }                                             \
}
#define FA_MMA_PRINT_L0_REG(R, format, ...)       \
{                                                 \
  if (lane_id == 0) {                             \
    float2 v_reg = __half22float2(HALF2(R));      \
    printf("[L0] ");                              \
    printf(format, ##__VA_ARGS__);                \
    printf(", V0=%f, V1=%f\n", v_reg.x, v_reg.y); \
  }                                               \
}
#define FA_MMA_PRINT_L0_(format, ...)           \
{                                               \
  if (lane_id == 0) {                           \
    printf("[L0] ");                            \
    printf(format, ##__VA_ARGS__);              \
  }                                             \
}
#else
#define FA_MMA_PRINT_T0_REG(R, format, ...) {}
#define FA_MMA_PRINT_L0_REG(R, format, ...) {}
#define FA_MMA_PRINT_T0(format, ...) {}
#define FA_MMA_PRINT_L0(format, ...) {}
#endif
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

template<
         const int kHeadDim,   // 32,64,128     
         const int kMmaM,      // M 16
         const int kMmaN,      // N 8
         const int kMmaK,      // K 16
         const int kMmaTileQ,  // 2, M=16*2=32, Q@K    
         const int kMmaTileP,  // 2, M=16*2=32, P@V       
         const int kMmaTileK,  // 4, N=8*4= 32, Q@K     
         const int kMmaTileV,  // 4, N=8*4= 32, P@V    
         const int kWarpTileQ, // 2, Br=32*2=64, M
         const int kWarpTileP, // 2, Br=32*2=64, M
         const int kWarpTileK, // 2, Bc=32*2=64, N
         const int kWarpTileV, // 2, Bv=32*2=64, N, Must satisfy d=kWarpTileV*(kMmaN*kMmaTileV=32), 
                               // e.g, 1->d 32, 2->d 64, 3->d 96, 4-> d 128, ...
         const int kStage,     // 1,2, not support >= 3. 
         const int kPad        // 0,8,16
         >
__global__ void __launch_bounds__(
  WARP_SIZE * kMmaTileQ * kMmaTileK) // 32 * 2 * 4 = 256
flash_attn_mma_kernel(half* Q, half* K, half* V, half* O, int N) {
  // step 0: S_tile[Br,N] = Q_tile[Br,d] * K[N,d], slice-k manner matmul
  // across K's N dim, each K_tile/V_tile inner loop has shape [Bc,d].
  // step 1: P_tile[Br,N] = softmax(S_tile[Br,N]), row wise.
  // step 2: O_tile[Br,d] = P_tile[Br,N] * V[N,d], matmul.
  static_assert(kHeadDim % 32 == 0); // may relax for 16 ?
  static_assert(kMmaM == 16 && kMmaN == 8 && kMmaK == 16); // m16n8k16
  static_assert(kMmaTileQ  == 2 && kMmaTileK  == 4); // Q@K
  static_assert(kMmaTileP  == 2 && kMmaTileV  == 4); // P@V
  static_assert(kWarpTileQ == 2 && kWarpTileK == 2); // Q@K
  // e.g, 1->d 32, 2->d 64, 3->d 96, 4-> d 128, ..., etc.
  static_assert(kWarpTileP == 2 && kWarpTileV == (kHeadDim / (kMmaN*kMmaTileV))); // P@V
  static_assert(kStage > 0 && kStage < 3); // 1,2
  static_assert(kPad >= 0 && kPad % 8 == 0); // 0,8,16
  constexpr int d  = kHeadDim; // alias
  constexpr int Br = kMmaM * kMmaTileQ * kWarpTileQ; // 16*2*2=64
  constexpr int Bc = kMmaN * kMmaTileK * kWarpTileK; // 8*4*2=64
  constexpr int Bd = kMmaK; // 16, tile head_dim(d) according MMA
  constexpr int Tn = WARP_SIZE * kMmaTileQ * kMmaTileK; // 32*2*4=256, num threads
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
  // Only apply multi stages for K across N(seq_len) dim, not for Q, V.
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
  int load_gmem_K_n_offset = 0; 
  int load_gmem_V_n_offset = 0; 

  uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
  uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
  uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

  // --------------------- Regristers/SMEM for Br ---------------------
  // block m_old, l_old, store in lane, use float to keep precision.
  float lane_block_row_max_old[kWarpTileQ][2];
  float lane_block_row_sum_old[kWarpTileQ][2];
  fill_2D_regs<float, kWarpTileQ, 2>(lane_block_row_max_old, -INFINITY);
  fill_2D_regs<float, kWarpTileQ, 2>(lane_block_row_sum_old, 0.0f);
  // Mi [Br], Li[Br], 64x(4)x4=1024 bytes, 1M+1M=2M, 4M.
  // TODO: 64x4=256, use each thread to store a max/sum value 
  // instead of using shared memory and mapping based on thread
  // ID and row number in Br.
  static __shared__ float block_row_max_new_smem[Br][kMmaTileK]; 
  static __shared__ float block_row_sum_new_smem[Br][kMmaTileK];
  // Retile warp for [Br,d], kWarpTileV: 2 = 64/(4*8); 4 = 128/(4*8).
  // Compute PV = P[Br,Bc] @ V[Bc,d] = [Br,d] = [64, 64/128], partion Attention.
  // registers final [O]utput [O]=[Br,d], [2][2/4][2], 8 or 16 regs.
  uint32_t R_OO[kWarpTileQ][kWarpTileV][2]; // [2][2/4][2]
  // registers for tile_n PV[Br,d]=P@V, [2][2/4][2], 8 or 16 regs.
  // TODO: may reuse R_OO as R_PV? kWarpTileP=kWarpTileQ.
  uint32_t R_PV[kWarpTileP][kWarpTileV][2]; // [2][2/4][2]
  fill_3D_regs<uint32_t, kWarpTileQ, kWarpTileV, 2>(R_OO, 0);
  fill_3D_regs<uint32_t, kWarpTileP, kWarpTileV, 2>(R_PV, 0);
  FA_MMA_PRINT_T0_REG(R_OO[0][0][0], "Init OO tile");
  FA_MMA_PRINT_T0_REG(R_PV[0][0][0], "Init PV tile");
  
  // --------------- Regristers for Bc (loop over N) ------------------
  // registers for current tile_n within <loop over N>, 
  // [64,64] = S_tile[Br,Bc] = Q_tile[Br,d] * K[Bc,d]
  // each thread hold 2x32 bits regs. S,P may shared 
  // the same registers.
  uint32_t R_SP[kWarpTileQ][kWarpTileK][2]; // [2][2][2]
  fill_3D_regs<uint32_t, kWarpTileQ, kWarpTileK, 2>(R_SP, 0);

  // registers for Q, K(V reuse), for Q[Br,d]@K[Bc,d]=[Br,Bc], the matmul layout is NN, 
  // and we need kWarpTileK across N dim for K matrix, but for P[Br,Bc]@V[Bc,d]=[Br,d],
  // the matmul layout is TN, we need kWarpTileV across headdim(d) dim.
  constexpr int kMaxWarpTileKV = (
    kWarpTileK > kWarpTileV ? kWarpTileK : kWarpTileV);
  uint32_t R_QP[kWarpTileQ][4]; // kWarpTileP=kWarpTileQ.
  uint32_t R_KV[kMaxWarpTileKV][2];

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
      CP_ASYNC_CG(load_smem_Q_ptr + i * 2, 
                  &Q[load_gmem_Q_addr + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP();
  }

  // load K from gmem -> smem, (kStage - 1) K tiles, [Bc,d]
  if constexpr (kStage > 1) {
    #pragma unroll
    for (int stage = 0; stage < (kStage - 1); ++stage) {
      // update the offset of n according to stages
      load_gmem_K_n_offset += stage * Bc; // s2, +offset 0
      int load_gmem_K_n = load_gmem_K_n_offset + load_smem_K_n;
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
        CP_ASYNC_CG(load_smem_K_ptr + i * 2, 
                    &K[load_gmem_K_addr + i], 16);
      }
      printf("tid %d, load_gmem_K_n %d, load_gmem_K_d %d, load_smem_K_n %d, load_smem_K_d %d\n", 
              tid, load_gmem_K_n, load_gmem_K_d, load_smem_K_n, load_smem_K_d);
      CP_ASYNC_COMMIT_GROUP();
    }
  } else {
    // update the offset of n according to stages
    load_gmem_K_n_offset += 0 * Bc; // s2, +offset 0
    int load_gmem_K_n = load_gmem_K_n_offset + load_smem_K_n;
    int load_gmem_K_d = load_smem_K_d;
    int load_gmem_K_addr = (
      KV_gmem_offset + load_gmem_K_n * d + load_gmem_K_d);
     uint32_t load_smem_K_ptr = (
      smem_K_base_ptr + (0 * KV_tile_size + 
                         load_smem_K_n * (d + kPad) + 
                         load_smem_K_d) * sizeof(half)
    );
    // load d / (Tn / Bc) vals, 64 or 128 div 4, 16 or 32, 
    // need 2 or 4 128 bits memory issues.
    #pragma unroll
    for (int i = 0; i < (d / (Tn / Bc)); i += 8) {
      CP_ASYNC_CG(load_smem_K_ptr + i * 2, 
                  &K[load_gmem_K_addr + i], 16);
    }
    printf("tid %d, load_gmem_K_n %d, load_gmem_K_d %d, load_smem_K_n %d, load_smem_K_d %d, stage %d\n", 
            tid, load_gmem_K_n, load_gmem_K_d, load_smem_K_n, load_smem_K_d, kStage);
    CP_ASYNC_COMMIT_GROUP();
  }

  // wait Q and at least (kStage - 1) for K ready.
  if constexpr (kStage - 2 >= 0) {
    CP_ASYNC_WAIT_GROUP(kStage - 2); // s2->0, s3->1, s4->2
  } else {
    CP_ASYNC_WAIT_GROUP(0);
  }
  CP_ASYNC_WAIT_GROUP(0);
  __syncthreads(); 
  // 48~63的结果为0
  if (tid == 0) {
    int st_i = -1;
    int st_j = -1;
    for (int i = 0; i < Bc; ++i) {
      for (int j = 0; j < d; ++j) {
        float v = __half2float(*(K_tile_smem + i * d + j));
        if (v < 1.0f) {
          // printf("(%d, %d); ", i, j);
          if (st_i < 0) st_i = i;
          if (st_j < 0) st_j = j;
        }
      }
      // printf("\n");
    }
    printf("st i=%d, j=%d\n", st_i, st_j); // 48
  }

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

    // Prefetch curr V tile_n [Bc,d] (no stages)
    {
      load_gmem_V_n_offset += tile_n * Bc;
      int load_gmem_V_n = load_gmem_V_n_offset + load_smem_V_n;
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
        CP_ASYNC_CG(load_smem_V_ptr + i * 2, 
                    &K[load_gmem_V_addr + i], 16);
      }
      CP_ASYNC_COMMIT_GROUP();
    }

    // Prefetch next stage K (tile_n + 1) [Bc,d]
    if constexpr (kStage > 1) {
      if ((tile_n + 1) < Tc) {
        load_gmem_K_n_offset += (tile_n + 1) * Bc;
        int load_gmem_K_n = load_gmem_K_n_offset + load_smem_K_n;
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
          CP_ASYNC_CG(load_smem_K_ptr + i * 2, 
                      &K[load_gmem_K_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
      } else {
        // wait all memory issues ready for last tile.
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads(); 
      }
    }
    
    // <loop over d>: tile_d, Bd = 16, K_tile_d[Bc,Bd]
    // Matmul with TN layout, S_tile[Br,Bc]=Q_tile[Br,d]@K[Bc,d]
    #pragma unroll
    for (int tile_d = 0; tile_d < Td; ++tile_d) {
      // offset d according tile_d
      // smem -> reg, load smem Q
      // ldmatrix.x4 for Q_tile_smem, ldmatrix.x2 for K_tile_smem
      #pragma unroll
      for (int i = 0; i < kWarpTileQ; ++i) {
        int warp_smem_Q_n = warp_QP * (kMmaM * kWarpTileQ) + i * kMmaM;
        int lane_smem_Q_n = warp_smem_Q_n + lane_id % 16; // 0~15
        int lane_smem_Q_d = tile_d * Bd + (lane_id / 16) * 8; // 0,8
        uint32_t lane_smem_Q_ptr = (
            smem_Q_base_ptr + (lane_smem_Q_n * (d + kPad) + 
                               lane_smem_Q_d) * sizeof(half)
        );
        LDMATRIX_X4(R_QP[i][0], R_QP[i][1], R_QP[i][2], R_QP[i][3], 
                    lane_smem_Q_ptr); // R_Q
        float2 v_reg = __half22float2(HALF2(R_QP[i][0]));
        if (v_reg.x < 1.0f) {
          printf("R_QP[%d][0], V0=%f, V1=%f\n", i, v_reg.x, v_reg.y);
        }
        // FA_MMA_PRINT_REG(R_QP[i][0], "R_QP[%d][0]", i);
        // FA_MMA_PRINT_REG(R_QP[i][1], "R_QP[%d][1]", i);
        // FA_MMA_PRINT_REG(R_QP[i][2], "R_QP[%d][2]", i);
        // FA_MMA_PRINT_REG(R_QP[i][3], "R_QP[%d][3]", i);
      }
      FA_MMA_PRINT_T0_REG(R_QP[0][0], "Load Q s->r, tile_n: %d, tile_d: %d", tile_n, tile_d);

      #pragma unroll
      for (int j = 0; j < kWarpTileK; ++j) {
        int warp_smem_K_n = warp_KV * (kMmaN * kWarpTileK) + j * kMmaN;
        int lane_smem_K_n = warp_smem_K_n + lane_id % 8; // 0~7, MMA_N=8
        int lane_smem_K_d = tile_d * kMmaK + ((lane_id / 8) % 2) * 8; // 0,8, Bd=16
        uint32_t lane_smem_K_ptr = (
            smem_K_base_ptr + (smem_sel * KV_tile_size + 
                               lane_smem_K_n * (d + kPad) + 
                               lane_smem_K_d) * sizeof(half)
        );
        LDMATRIX_X2(R_KV[j][0], R_KV[j][1], lane_smem_K_ptr); // R_K
        // FA_MMA_PRINT_REG(R_KV[j][0], "R_QP[%d][0]", j);
        // FA_MMA_PRINT_REG(R_KV[j][1], "R_QP[%d][1]", j);
        // float2 v_reg = __half22float2(HALF2(R_KV[j][0]));
        // if (v_reg.x < 1.0f) {
        //   printf("tid %d, R_KV[%d][0], V0=%f, V1=%f, n=%d, d=%d, ptr=%d\n", 
        //           tid, j, v_reg.x, v_reg.y, lane_smem_K_n, lane_smem_K_d,
        //           lane_smem_K_ptr);
        // }
      }
      FA_MMA_PRINT_T0_REG(R_KV[0][0], "Load K s->r, tile_n: %d, tile_d: %d", tile_n, tile_d);

      // MMA compute
      #pragma unroll
      for (int i = 0; i < kWarpTileQ; ++i) {
        #pragma unroll
        for (int j = 0; j < kWarpTileK; ++j) {
          HMMA16816(R_SP[i][j][0], R_SP[i][j][1], 
                    R_QP[i][0],    R_QP[i][1],    R_QP[i][2], R_QP[i][3], 
                    R_KV[j][0],    R_KV[j][1], 
                    R_SP[i][j][0], R_SP[i][j][1]);
          // if (R_KV[j][0] == 0) {
          //   printf("R_KV[%d][0] == 0\n", j);
          // }
          // if (R_QP[i][0] == 0) {
          //   printf("R_QP[%d][0] == 0\n", i);
          // }
        }
      }
      FA_MMA_PRINT_T0_REG(R_SP[0][0][0], "MMA Q@K, tile_n: %d, tile_d: %d", tile_n, tile_d);
    } // end loop over d, S=Q@K^T
    __syncthreads();

    // TODO: May reuse K smem for V, for example, stages 2, stage
    // 0 K smem can be reuse as V smem 0 because we do not need 
    // K values on stage 0 K smem anymore.

    // Now, we got a computed tile of S[Br,N], tile with shape [Br,Bc].
    // Assume [Br, Bc] = [64, 64] = 64x64 = 4096 values. Each thread holds
    // a portion of this [Br, Bc] block, specifically, R_S = R_SP[2][2][2]. 
    // This means that each Warp (MMA) repeats 2 times in the N direction 
    // for both Q and K, resulting in 2x2 = 4 sets of MMA results. Each set 
    // of results is stored in 2 32-bit registers, with each register holding 
    // 2 half-precision values. In other words, each thread stores (4x2)x2 = 16 
    // half-precision values. With a total of 256 threads, the total number of 
    // half-precision values is 256x16 = 4096, which exactly matches the total 
    // [Br, Bc] = [64, 64] values.

    // The layout of 8 MMA m16n8k16 (2x4)  [after] kWarpTileQPxkWarpTileKV(2x2) -> 32x2,32x2=64x64: 
    // |  [64,64]  |    warp_KV 0    |    warp_KV 1    |    warp_KV 2    |    warp_KV 3    |
    // | warp_QP 0 |-- MMA 0,MMA 0 --|-- MMA 2,MMA 2 --|-- MMA 4,MMA 4 --|-- MMA 6,MMA 6 --| row max
    // | warp_QP 0 |-- MMA 0,MMA 0 --|-- MMA 2,MMA 2 --|-- MMA 4,MMA 4 --|-- MMA 6,MMA 6 --| row max
    // | warp_QP 1 |-- MMA 1,MMA 1 --|-- MMA 3,MMA 3 --|-- MMA 5,MMA 5 --|-- MMA 7,MMA 7 --| row max
    // | warp_QP 1 |-- MMA 1,MMA 1 --|-- MMA 3,MMA 3 --|-- MMA 5,MMA 5 --|-- MMA 7,MMA 7 --| row max

    // WIP: online safe softmax, warp/block reduce max/sum, row wise
    // warp 0/2/4/6, [0][2] row 0~15,  col 0/8/16/32, max, [1][2] row 16~31, col 0/8/16/32, max
    // warp 1/3/5/7, [0][2] row 32~47, col 0/8/16/32, max, [1][2] row 48~61, col 0/8/16/32, max
    float lane_row_max_new[kWarpTileQ][2]; 
    float lane_row_sum_new[kWarpTileQ][2]; 
    fill_2D_regs<float, kWarpTileQ, 2>(lane_row_max_new, -INFINITY);
    fill_2D_regs<float, kWarpTileQ, 2>(lane_row_sum_new, 0.0f);

    // Row max for [Br,Bc] tile, Thread -> Warp -> Block.
    #pragma unroll
    for (int i = 0; i < kWarpTileQ; ++i) {
      // Thread level reduce max across kWarpTileK dim, namely Bc.
      #pragma unroll
      for (int j = 0; j < kWarpTileK; ++j) {
        // reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
        // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
        // The layout of the fragments held by different threads for C. (m16n8k16)
        // Row\Col  0    1    2    3    4    5    6    7
        // 0        T0: {c0, c1}  T1: {c0, c1}  T2: {c0, c1}  T3: {c0, c1}
        // 1        T4: {c0, c1}  T5: {c0, c1}  T6: {c0, c1}  T7: {c0, c1}
        // 2        ...
        // ...
        // 7        T28: {c0, c1}  T29: {c0, c1}  T30: {c0, c1}  T31: {c0, c1}
        // 8        T0: {c2, c3}   T1: {c2, c3}   T2: {c2, c3}   T3: {c2, c3}
        // 9        T4: {c2, c3}   T5: {c2, c3}   T6: {c2, c3}   T7: {c2, c3}
        // 10       ...
        // ...
        // 15       T28: {c2, c3}  T29: {c2, c3}  T30: {c2, c3}  T31: {c2, c3}
        float2 t_reg_0 = __half22float2(HALF2(R_SP[i][j][0])); // 0~7  {c0, c1}
        float2 t_reg_1 = __half22float2(HALF2(R_SP[i][j][1])); // 8~15 {c2, c3}
        float tmp_max_0 = max(t_reg_0.x, t_reg_0.y);
        float tmp_max_1 = max(t_reg_1.x, t_reg_1.y);
        lane_row_max_new[i][0] = max(lane_row_max_new[i][0], tmp_max_0);
        lane_row_max_new[i][1] = max(lane_row_max_new[i][1], tmp_max_1);
        // if (abs(t_reg_0.x - 64.0f) > 0.1f) {
        //   printf("###### tid: %d, R_SP(i,j)=(%d,%d)[0], V0=%f, V1=%f, max=%f\n", tid, i, j, t_reg_0.x, t_reg_0.y, tmp_max_0);
        // }
        // printf("tid: %d, R_SP(i,j)=(%d,%d)[0], V0=%f, V1=%f, max=%f\n", tid, i, j, t_reg_0.x, t_reg_0.y, tmp_max_0);
      } // end for kWarpTileK

      // Warp level reduce max, warp_size = 4
      // Each thread contains the maximum of 2 rows of Br, 
      // and only the values of T0, T4, ..., T28 are used.
      // Br, row_id = warp_QP<0|1> * 32 + i<0|1> * 16 + 0 * 8 + (lane / 4) <0~7>
      lane_row_max_new[i][0] = warp_reduce_max<float, 4>(lane_row_max_new[i][0]);
      // Br, row_id = warp_QP<0|1> * 32 + i<0|1> * 16 + 1 * 8 + (lane / 4) <8~15>
      lane_row_max_new[i][1] = warp_reduce_max<float, 4>(lane_row_max_new[i][1]);
      __syncthreads(); 

      if (lane_id % 4 == 0) { // only need T0,T4,...,T28
        block_row_max_new_smem[ // Br, row_id, 0~7,  16~23, 32~39, 48~55
          warp_QP * 32 + i * 16 + 0 * 8 + (lane_id / 4)][warp_KV] = lane_row_max_new[i][0];
        block_row_max_new_smem[ // Br, row_id, 8~15, 24~31, 40~47, 56~63
          warp_QP * 32 + i * 16 + 1 * 8 + (lane_id / 4)][warp_KV] = lane_row_max_new[i][1];
      }
      __syncthreads();
    } // end for kWarpTileQ
    __syncthreads();

    FA_MMA_PRINT_T0("lane_row_max_new %f, %f\n", lane_row_max_new[0][0], lane_row_max_new[0][1]);

#ifdef FLASH_ATTN_MMA_DEBUG
    if (tid == 0) {
      printf("----------------------------------------\n");
      printf("[block][block_row_max_new_smem]\n");
      for (int i = 0; i < Br; ++i) {
        for (int j = 0; j < kMmaTileK; ++j) {
          printf("[%d][%d]=%f, ", i, j, block_row_max_new_smem[i][j]);
        }
        printf("\n");
      }
      printf("----------------------------------------\n");
    }
    __syncthreads(); 
#endif
    // Block level reduce max, row wise, 64x4=256
    float wrp_row_max_new = block_row_max_new_smem[tid / kMmaTileK][tid % kMmaTileK]; // [0~63][0~4]
    float blk_row_max_new = warp_reduce_max<float, 4>(wrp_row_max_new);
    block_row_max_new_smem[tid / kMmaTileK][tid % kMmaTileK] = blk_row_max_new;
    __syncthreads();
#ifdef FLASH_ATTN_MMA_DEBUG
    if (tid == 0) {
      printf("----------------------------------------\n");
      printf("[block][block_row_max_new_smem]\n");
      for (int i = 0; i < Br; ++i) {
        for (int j = 0; j < kMmaTileK; ++j) {
          printf("[%d][%d]=%f, ", i, j, block_row_max_new_smem[i][j]);
        }
        printf("\n");
      }
      printf("----------------------------------------\n");
    }
    __syncthreads(); 
#endif
    FA_MMA_PRINT_T0("blk_row_max_new %f, %f\n", blk_row_max_new, block_row_max_new_smem[tid / kMmaTileK][tid % kMmaTileK]);

    // Exp sum and mul scale_factor for [Br,Bc] tile, Thread -> Warp -> Block.
    #pragma unroll
    for (int i = 0; i < kWarpTileQ; ++i) {
      // Use latest global row max without update.
      // Br 0, row_id, 0~7,  16~23, 32~39, 48~55; 
      int row_id_0 = warp_QP * 32 + i * 16 + 0 * 8 + (lane_id / 4);
      float block_row_max_new_0 = block_row_max_new_smem[row_id_0][0]; 
      // Br 1, row_id, 8~15, 24~31, 40~47, 56~63;
      int row_id_1 = warp_QP * 32 + i * 16 + 1 * 8 + (lane_id / 4);
      float block_row_max_new_1 = block_row_max_new_smem[row_id_1][0];
      FA_MMA_PRINT_T0("load block_row_max_new_0: %f, block_row_max_new_1: %f, tile_n: %d, row_id_0: %d, row_id_1:%d\n", 
                       block_row_max_new_0, block_row_max_new_1, tile_n, row_id_0, row_id_1);
      block_row_max_new_0 = (tile_n > 0 ? max(lane_block_row_max_old[i][0], 
                                              block_row_max_new_0) : block_row_max_new_0);
      block_row_max_new_1 = (tile_n > 0 ? max(lane_block_row_max_old[i][1], 
                                              block_row_max_new_1) : block_row_max_new_1);
      #pragma unroll
      for (int j = 0; j < kWarpTileK; ++j) {
        float2 t_reg_0 = __half22float2(HALF2(R_SP[i][j][0])); // 0~7  {c0, c1}
        float2 t_reg_1 = __half22float2(HALF2(R_SP[i][j][1])); // 8~15 {c2, c3}
        FA_MMA_PRINT_T0_REG(
          R_SP[i][j][0], 
          "[Exp sum][Before] R_SP[%d][%d][0], scale: %f, block_row_max_new_0: %f", 
          i, j, scale, block_row_max_new_0);
        FA_MMA_PRINT_T0_REG_V2(t_reg_0, "[Exp sum][Before] t_reg_0");
        t_reg_0.x = __expf(t_reg_0.x * scale - block_row_max_new_0 * scale);
        t_reg_0.y = __expf(t_reg_0.y * scale - block_row_max_new_0 * scale);
        t_reg_1.x = __expf(t_reg_1.x * scale - block_row_max_new_1 * scale);
        t_reg_1.y = __expf(t_reg_1.y * scale - block_row_max_new_1 * scale);
        lane_row_sum_new[i][0] += (t_reg_0.x + t_reg_0.y);
        lane_row_sum_new[i][1] += (t_reg_1.x + t_reg_1.y);
        // Update R_SP for P[Br,Bc] = Exp(S-m), point wise.
        HALF2(R_SP[i][j][0]) = __float22half2_rn(t_reg_0);
        HALF2(R_SP[i][j][1]) = __float22half2_rn(t_reg_1);
        FA_MMA_PRINT_T0_REG(R_SP[i][j][0], "[Exp sum][After] R_SP[%d][%d][0], scale: %f", i, j, scale);
        FA_MMA_PRINT_T0_REG(R_SP[i][j][1], "[Exp sum][After] R_SP[%d][%d][1], scale: %f", i, j, scale);
      } // end for kWarpTileK

      // Warp level reduce sum, warp_size = 4
      lane_row_sum_new[i][0] = warp_reduce_sum<float, 4>(lane_row_sum_new[i][0]);
      lane_row_sum_new[i][1] = warp_reduce_sum<float, 4>(lane_row_sum_new[i][1]);
      __syncthreads(); 

      if (lane_id % 4 == 0) { // only need T0,T4,...,T28
        block_row_sum_new_smem[ // Br, row_id, 0~7,  16~23, 32~39, 48~55
          warp_QP * 32 + i * 16 + 0 * 8 + (lane_id / 4)][warp_KV] = lane_row_sum_new[i][0];
        block_row_sum_new_smem[ // Br, row_id, 8~15, 24~31, 40~47, 56~63
          warp_QP * 32 + i * 16 + 1 * 8 + (lane_id / 4)][warp_KV] = lane_row_sum_new[i][1];
      }
      __syncthreads(); 
    } // end for kWarpTileQ
    __syncthreads();

    FA_MMA_PRINT_T0("lane_row_sum_new %f, %f\n", lane_row_sum_new[0][0], lane_row_sum_new[0][1]);

#ifdef FLASH_ATTN_MMA_DEBUG
    if (tid == 0) {
      printf("----------------------------------------\n");
      printf("[warp][block_row_sum_new_smem]\n");
      for (int i = 0; i < Br; ++i) {
        for (int j = 0; j < kMmaTileK; ++j) {
          printf("[%d][%d]=%f, ", i, j, block_row_sum_new_smem[i][j]);
        }
        printf("\n");
      }
      printf("----------------------------------------\n");
    }
    __syncthreads(); 
#endif
    // Block level reduce sum, row wise, 64x4=256
    float wrp_row_sum_new = block_row_sum_new_smem[tid / kMmaTileK][tid % kMmaTileK]; // [0~63][0~4]
    float blk_row_sum_new = warp_reduce_sum<float, 4>(wrp_row_sum_new);
    block_row_sum_new_smem[tid / kMmaTileK][tid % kMmaTileK] = blk_row_sum_new;
    __syncthreads();
#ifdef FLASH_ATTN_MMA_DEBUG
    if (tid == 0) {
      printf("----------------------------------------\n");
      printf("[block][block_row_sum_new_smem]\n");
      for (int i = 0; i < Br; ++i) {
        for (int j = 0; j < kMmaTileK; ++j) {
          printf("[%d][%d]=%f, ", i, j, block_row_sum_new_smem[i][j]);
        }
        printf("\n");
      }
      printf("----------------------------------------\n");
    }
    __syncthreads(); 
#endif
    
    FA_MMA_PRINT_T0("blk_row_sum_new %f, %f\n", blk_row_sum_new, block_row_sum_new_smem[tid / kMmaTileK][tid % kMmaTileK]);

    // Compute P[Br,Bc] @ V[Bc,d] = [Br,d] = [64, 64/128], partion Attention.
    // Here, we have to wait V ready before compute O = P @ V
    if constexpr (kStage == 2) {
      // NOTE: we have send V mem issues before K
      CP_ASYNC_WAIT_GROUP(1); // s1->-1, s2->0, s3->1, s4->2
    } else {
      CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads(); 
    
    // Retile warp for [Br,d], kWarpTileV: 1=32/(4*8); 2=64/(4*8); 4=128/(4*8).
    // Compute P[Br,Bc] @ V[Bc,d] = [Br,d] = [64, 64/128], partion Attention.

    // If headdim=<32>, then, kWarpTileV = 1, the layout of 8 MMA m16n8k16 (2x4) after 
    // kWarpTilePxkWarpTileV(2x1) tiling to (32x2,32x1)=(64x32), will look like: 
    // |  [64,32]  | warp_KV 0 | warp_KV 1 | warp_KV 2 | warp_KV 3 |
    // | warp_QP 0 |-- MMA 0 --|-- MMA 2 --|-- MMA 4 --|-- MMA 6 --|
    // | warp_QP 0 |-- MMA 0 --|-- MMA 2 --|-- MMA 4 --|-- MMA 6 --|
    // | warp_QP 1 |-- MMA 1 --|-- MMA 3 --|-- MMA 5 --|-- MMA 7 --|
    // | warp_QP 1 |-- MMA 1 --|-- MMA 3 --|-- MMA 5 --|-- MMA 7 --|

    // If headdim=<64>, then, kWarpTileV = 2, the layout of 8 MMA m16n8k16 (2x4) after 
    // kWarpTilePxkWarpTileV(2x2) tiling to (32x2,32x2)=(64x64), will look like: 
    // |  [64,64]  |    warp_KV 0    |    warp_KV 1    |    warp_KV 2    |    warp_KV 3    |
    // | warp_QP 0 |-- MMA 0,MMA 0 --|-- MMA 2,MMA 2 --|-- MMA 4,MMA 4 --|-- MMA 6,MMA 6 --|
    // | warp_QP 0 |-- MMA 0,MMA 0 --|-- MMA 2,MMA 2 --|-- MMA 4,MMA 4 --|-- MMA 6,MMA 6 --|
    // | warp_QP 1 |-- MMA 1,MMA 1 --|-- MMA 3,MMA 3 --|-- MMA 5,MMA 5 --|-- MMA 7,MMA 7 --|
    // | warp_QP 1 |-- MMA 1,MMA 1 --|-- MMA 3,MMA 3 --|-- MMA 5,MMA 5 --|-- MMA 7,MMA 7 --|

    // If headdim=<128>, then, kWarpTileV = 4, the layout of 8 MMA m16n8k16 (2x4) after 
    // kWarpTilePxkWarpTileV(2x2x2) tiling to (32x2,32x2x2)=(64x64x2), will look like: 
    // | [64,64x2] |         warp_KV 0           |           warp_KV 1         |           warp_KV 2         |          warp_KV 3          |
    // | warp_QP 0 |-- MMA 0,MMA 0,MMA 0,MMA 0 --|-- MMA 2,MMA 2,MMA 2,MMA 2 --|-- MMA 4,MMA 4,MMA 4,MMA 4 --|-- MMA 6,MMA 6,MMA 6,MMA 6 --|
    // | warp_QP 0 |-- MMA 0,MMA 0,MMA 0,MMA 0 --|-- MMA 2,MMA 2,MMA 2,MMA 2 --|-- MMA 4,MMA 4,MMA 4,MMA 4 --|-- MMA 6,MMA 6,MMA 6,MMA 6 --|
    // | warp_QP 1 |-- MMA 1,MMA 1,MMA 1,MMA 1 --|-- MMA 3,MMA 3,MMA 3,MMA 3 --|-- MMA 5,MMA 5,MMA 5,MMA 5 --|-- MMA 7,MMA 7,MMA 7,MMA 7 --|
    // | warp_QP 1 |-- MMA 1,MMA 1,MMA 1,MMA 1 --|-- MMA 3,MMA 3,MMA 3,MMA 3 --|-- MMA 5,MMA 5,MMA 5,MMA 5 --|-- MMA 7,MMA 7,MMA 7,MMA 7 --|
    
    // Make sure to clear the states in R_PV before MMA for P@V.
    fill_3D_regs<uint32_t, kWarpTileP, kWarpTileV, 2>(R_PV, 0);
    for (int tile_Bc = 0; tile_Bc < (Bc / kMmaK); ++tile_Bc) {
      // Load V from smem -> regs, R_KV, ldmatrix.trans, M=Br,N=d,K=Bc.
      // Matmul with NN layout: O[Br,d]=P[Br,Bc]@V[Bc,d], A=P[Br,Bc], B=V[Bc,d]
      #pragma unroll
      for (int i = 0; i < kWarpTileV; ++i) {
        // FIXME: P[Br,Bc] @ V[Bc=K,d] = [Br,d] = [64, 64/128]
        // int warp_smem_V_d = warp_KV * (kMmaN * kWarpTileV) + i * kMmaN;
        // int lane_smem_V_n = lane_id % 16; // 0~15;
        // int lane_smem_V_d = warp_smem_V_d; // 0
        int warp_smem_V_d = tile_Bc * kMmaK + warp_KV * (kMmaN * kWarpTileV) + i * kMmaN;
        int lane_smem_V_n = lane_id % 16; // 0~15;
        int lane_smem_V_d = warp_smem_V_d; // 0
        uint32_t lane_smem_V_ptr = (
            smem_V_base_ptr + (lane_smem_V_n * (d + kPad) + 
                               lane_smem_V_d) * sizeof(half)
        );
        LDMATRIX_X2_T(R_KV[i][0], R_KV[i][1], lane_smem_V_ptr); // R_V
        // FA_MMA_PRINT_REG(R_KV[i][0], "R_QP[%d][0]", i);
        // FA_MMA_PRINT_REG(R_KV[i][1], "R_QP[%d][1]", i);
        // float2 v_reg = __half22float2(HALF2(R_KV[i][0]));
        float2 v_reg = __half22float2(HALF2(R_KV[i][0]));
        if (v_reg.x < 1.f) {
          printf("P@V tid %d, R_KV[%d][0], V0=%f, V1=%f, n=%d, d=%d, ptr=%d\n", 
                  tid, i, v_reg.x, v_reg.y, lane_smem_V_n, lane_smem_V_d,
                  lane_smem_V_ptr);
        }
      }
      
      // MMA compute
      // FIXME(DefTruth): May need to reorder R_SP[2][2][2] to [2][4=2x2] ?
      // according to the A matrix layout for MMA m16n8k16 instruction. 
      // reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
      // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
      // The layout of the fragments held by different threads for A matrix with .f16.
      // R\C  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
      // 0    T0: {a0, a1}  T1: {a0, a1}  T2: {a0, a1}  T3: {a0, a1}  T0: {a4, a5}  T1: {a4, a5}  T2: {a4, a5}  T3: {a4, a5}
      // 1    T4: {a0, a1}  T5: {a0, a1}  T6: {a0, a1}  T7: {a0, a1}  T4: {a4, a5}  T5: {a4, a5}  T6: {a4, a5}  T7: {a4, a5}
      // 2    (dashed arrow pointing right)
      // ...
      // 7    T28: {a0, a1}  T29: {a0, a1}  T30: {a0, a1}  T31: {a0, a1}  T28: {a4, a5}  T29: {a4, a5}  T30: {a4, a5}  T31: {a4, a5}
      // 8    T0: {a2, a3}   T1: {a2, a3}   T2: {a2, a3}   T3: {a2, a3}   T0: {a6, a7}   T1: {a6, a7}   T2: {a6, a7}   T3: {a6, a7}
      // 9    T4: {a2, a3}   T5: {a2, a3}   T6: {a2, a3}   T7: {a2, a3}   T4: {a6, a7}   T5: {a6, a7}   T6: {a6, a7}   T7: {a6, a7}
      // 10   (dashed arrow pointing right)
      // ...
      // 15   T28: {a2, a3}  T29: {a2, a3}  T30: {a2, a3}  T31: {a2, a3}  T28: {a6, a7}  T29: {a6, a7}  T30: {a6, a7}  T31: {a6, a7}

      // 注意，不能直接使用R_SP[2][2][2]，需要保存到R_QP[2][4]?

      #pragma unroll
      for (int i = 0; i < kWarpTileQ; ++i) {
        #pragma unroll
        for (int j = 0; j < kWarpTileV; ++j) {
          FA_MMA_PRINT_T0_REG(R_SP[i][0][0], "before R_SP[%d][0][0] MMA P@V, tile_n: %d", i, tile_n); 
          FA_MMA_PRINT_T0_REG(R_SP[i][0][1], "before R_SP[%d][0][1] MMA P@V, tile_n: %d", i, tile_n); 
          FA_MMA_PRINT_T0_REG(R_SP[i][1][0], "before R_SP[%d][1][0] MMA P@V, tile_n: %d", i, tile_n); 
          FA_MMA_PRINT_T0_REG(R_SP[i][1][1], "before R_SP[%d][1][1] MMA P@V, tile_n: %d", i, tile_n); 
          FA_MMA_PRINT_T0_REG(R_KV[j][0], "before R_KV[%d][0] MMA P@V, tile_n: %d", j, tile_n); 
          FA_MMA_PRINT_T0_REG(R_KV[j][1], "before R_KV[%d][1] MMA P@V, tile_n: %d", j, tile_n); 
          HMMA16816(R_PV[i][j][0], R_PV[i][j][1], 
                    R_SP[i][0][0], R_SP[i][0][1], R_SP[i][1][0], R_SP[i][1][1], 
                    R_KV[j][0],    R_KV[j][1], 
                    R_PV[i][j][0], R_PV[i][j][1]);
        }
      }
    } // end O=P@V=[Br,d]
    FA_MMA_PRINT_T0_REG(R_PV[0][0][0], "MMA P@V, tile_n: %d", tile_n);

    // Rescale O -> Update row sum Exp -> then, Update row max.
    #pragma unroll
    for (int i = 0; i < kWarpTileQ; ++i) { // kWarpTileQ=kWarpTileP
      // m = max(m_old, m_new), l = exp(m_old - m) * l_old + l_new (FA2 paper)
      // Br 0, row_id, 0~7,  16~23, 32~39, 48~55; Br 1, row_id, 8~15, 24~31, 40~47, 56~63
      float block_row_max_new_0 = block_row_max_new_smem[
        warp_QP * 32 + i * 16 + 0 * 8 + (lane_id / 4)][0];
      float block_row_max_new_1 = block_row_max_new_smem[
        warp_QP * 32 + i * 16 + 1 * 8 + (lane_id / 4)][0];
      float block_row_sum_new_0 = block_row_sum_new_smem[
        warp_QP * 32 + i * 16 + 0 * 8 + (lane_id / 4)][0];
      float block_row_sum_new_1 = block_row_sum_new_smem[
        warp_QP * 32 + i * 16 + 1 * 8 + (lane_id / 4)][0];
      float block_row_sum_old_0 = lane_block_row_sum_old[i][0];
      float block_row_sum_old_1 = lane_block_row_sum_old[i][1];
      float block_row_max_old_0 = lane_block_row_max_old[i][0];
      float block_row_max_old_1 = lane_block_row_max_old[i][1];
      block_row_max_new_0 = (tile_n > 0 ? max(block_row_max_old_0, 
                                              block_row_max_new_0) : block_row_max_new_0);
      block_row_max_new_1 = (tile_n > 0 ? max(block_row_max_old_1, 
                                              block_row_max_new_1) : block_row_max_new_1);
      block_row_max_old_0 = (tile_n > 0 ? block_row_max_old_0 : block_row_max_new_0);                                       
      block_row_max_old_1 = (tile_n > 0 ? block_row_max_old_0 : block_row_max_new_1);                                       

      // 0. Rescale O: Online rescaling O each tile_n step, need m_new, m_old.
      // m = max(m_old, m_new), O_new[Br,d] = ( 1/exp(m_old - m) ) * O_old + P@V (FA2 paper)
      #pragma unroll
      for (int j = 0; j < kWarpTileV; ++j) {
        float2 t_reg_PV_0 = __half22float2(HALF2(R_PV[i][j][0])); // 0~7  {c0, c1}
        float2 t_reg_PV_1 = __half22float2(HALF2(R_PV[i][j][1])); // 8~15 {c2, c3}
        float2 t_reg_OO_0 = __half22float2(HALF2(R_OO[i][j][0])); // 0~7  {c0, c1}
        float2 t_reg_OO_1 = __half22float2(HALF2(R_OO[i][j][1])); // 8~15 {c2, c3}
        FA_MMA_PRINT_T0_REG(R_OO[i][j][0], "[Before] Scale O tile t_reg_OO_0, tile_n: %d", tile_n);

        float rescale_o_factor_0 = __frcp_rn(__expf(
          block_row_max_old_0 - block_row_max_new_0)); // (1/exp(m_old - m))
        float rescale_o_factor_1 = __frcp_rn(__expf(
          block_row_max_old_1 - block_row_max_new_1)); // (1/exp(m_old - m))
        t_reg_OO_0.x = rescale_o_factor_0 * t_reg_OO_0.x + t_reg_PV_0.x;
        t_reg_OO_0.y = rescale_o_factor_0 * t_reg_OO_0.y + t_reg_PV_0.y;
        t_reg_OO_1.x = rescale_o_factor_1 * t_reg_OO_1.x + t_reg_PV_1.x;
        t_reg_OO_1.y = rescale_o_factor_1 * t_reg_OO_1.y + t_reg_PV_1.y;
        HALF2(R_OO[i][j][0]) = __float22half2_rn(t_reg_OO_0);
        HALF2(R_OO[i][j][1]) = __float22half2_rn(t_reg_OO_1);
        FA_MMA_PRINT_T0("Scale O tile block_row_max 0 old/new %f, %f\n", 
                         block_row_max_old_0, block_row_max_new_0);
        FA_MMA_PRINT_T0("Scale O tile block_row_max 1 old/new %f, %f\n", 
                         block_row_max_old_1, block_row_max_new_1);
        FA_MMA_PRINT_T0("Scale O tile rescale_o_factor %f, %f\n", 
                         rescale_o_factor_0, rescale_o_factor_1);
        FA_MMA_PRINT_T0_REG(R_OO[i][j][0], "[After] Scale O tile t_reg_OO_0, tile_n: %d", tile_n);
      } // end for kWarpTileV.

      // Now, we can update m, l after O has been scaled.
      // 1. First, update block row sum Exp for each lane which
      // need both m_new and m_old.
      lane_block_row_sum_old[i][0] = (
        __expf(block_row_max_old_0 - block_row_max_new_0) * 
        block_row_sum_old_0 + block_row_sum_new_0);
      lane_block_row_sum_old[i][1] = (
        __expf(block_row_max_old_1 - block_row_max_new_1) * 
        block_row_sum_old_1 + block_row_sum_new_1);
      // 2. Then, update block row max for each lane.
      lane_block_row_max_old[i][0] = block_row_max_new_0;
      lane_block_row_max_old[i][1] = block_row_max_new_1;
    }

    FA_MMA_PRINT_T0_REG(R_OO[0][0][0], "After Scale O tile, R_OO[0][0][0]");
  
    // NOTE: After compute P @ V, we have to wait next K tile ready in smem.
    // do not need to wait any things if kStage == 1.
    if constexpr (kStage == 2) {
      CP_ASYNC_WAIT_GROUP(0);
      __syncthreads(); 
    }

  } // end loop over N

  // Finaly, we still have to rescale O once more.
  // O_output = ( 1/l_final ) * O_final (FA2 paper)
  #pragma unroll
  for (int i = 0; i < kWarpTileQ; ++i) {
    #pragma unroll
    for (int j = 0; j < kWarpTileV; ++j) {
      float2 t_reg_OO_0 = __half22float2(HALF2(R_OO[i][j][0])); // 0~7  {c0, c1}
      float2 t_reg_OO_1 = __half22float2(HALF2(R_OO[i][j][1])); // 8~15 {c2, c3}
      t_reg_OO_0.x = __frcp_rn(lane_block_row_sum_old[i][0]) * t_reg_OO_0.x;
      t_reg_OO_0.y = __frcp_rn(lane_block_row_sum_old[i][0]) * t_reg_OO_0.y;
      t_reg_OO_1.x = __frcp_rn(lane_block_row_sum_old[i][1]) * t_reg_OO_1.x;
      t_reg_OO_1.y = __frcp_rn(lane_block_row_sum_old[i][1]) * t_reg_OO_1.y;
      HALF2(R_OO[i][j][0]) = __float22half2_rn(t_reg_OO_0);
      HALF2(R_OO[i][j][1]) = __float22half2_rn(t_reg_OO_1);
    }
  }

  FA_MMA_PRINT_T0_REG(R_OO[0][0][0], "After Final ReScale O tile, R_OO[0][0][0]");

  // Store O: Write O[Br,d] from regs -> gmem, collective store 
  // with reg reuse & warp shuffle. need R[2][4], may reuse 
  // R_QP[kWarpTileQ][4]=[2][4].
  #pragma unroll
  for (int i = 0; i < kWarpTileQ; ++i) {
    #pragma unroll
    for (int j = 0; j < kWarpTileV; ++j) {
      R_QP[0][0] = R_OO[i][j][0]; R_QP[1][0] = R_OO[i][j][1]; // warp_size 4
      R_QP[0][1] = __shfl_sync((0xffffffff), R_OO[i][j][0], lane_id + 1);
      R_QP[0][2] = __shfl_sync((0xffffffff), R_OO[i][j][0], lane_id + 2);
      R_QP[0][3] = __shfl_sync((0xffffffff), R_OO[i][j][0], lane_id + 3);
      R_QP[1][1] = __shfl_sync((0xffffffff), R_OO[i][j][1], lane_id + 1);
      R_QP[1][2] = __shfl_sync((0xffffffff), R_OO[i][j][1], lane_id + 2);
      R_QP[1][3] = __shfl_sync((0xffffffff), R_OO[i][j][1], lane_id + 3);
      // st.global.v4 128 bits.
      if (lane_id % 4 == 0) {
        int store_warp_regs_O_n = warp_QP * (kMmaM * kWarpTileQ) + i * kMmaM;
        int store_lane_gmem_O_n = QO_tile_id * Br + store_warp_regs_O_n + lane_id / 4;
        int store_warp_regs_O_d = warp_KV * (kMmaN * kWarpTileV)  + j * kMmaN;
        // The current tile actually covers all values in dimension d, therefore, 
        // there is no need to add a bx*BN term to calculate the offset, as you 
        // would in matrix multiplication.
        int store_lane_gmem_O_d = store_warp_regs_O_d;
        int store_gmem_O_addr_0 = (
          QO_gmem_offset + (store_lane_gmem_O_n + 0) * d + store_lane_gmem_O_d);
        int store_gmem_O_addr_1 = (
          QO_gmem_offset + (store_lane_gmem_O_n + 8) * d + store_lane_gmem_O_d);
        LDST128BITS(O[store_gmem_O_addr_0]) = LDST128BITS(R_QP[0][0]);
        LDST128BITS(O[store_gmem_O_addr_1]) = LDST128BITS(R_QP[1][0]);
        FA_MMA_PRINT_T0_REG(R_QP[0][0], "Store O, (n,d)=(%d,%d), (i,j)=(%d,%d)", 
                            store_lane_gmem_O_n + 0, store_lane_gmem_O_d, i, j);
        FA_MMA_PRINT_T0_REG(R_QP[1][0], "Store O, (n,d)=(%d,%d) (i,j)=(%d,%d)", 
                            store_lane_gmem_O_n + 8, store_lane_gmem_O_d, i, j);
      }
    } // end for kWarpTileV
  } // end for kWarpTileQ
}

// TODO: flash_attn_mma_kv_smem_shared_kernel

// --------------------- PyTorch bindings for custom kernel -----------------------
#include <torch/types.h>
#include <torch/extension.h>
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
if(((T).options().dtype() != (th_type))) {                   \
  std::cout << "Tensor Info:" << (T).options() << std::endl; \
  throw std::runtime_error("values must be "#th_type);       \
}

#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)             \
if (((T2).size(0) != (T1).size(0)) ||                \
    ((T2).size(1) != (T1).size(1)) ||                \
    ((T2).size(2) != (T1).size(2)) ||                \
    ((T2).size(3) != (T1).size(3))) {                \
  throw std::runtime_error("Tensor size mismatch!"); \
}

template<const int kHeadDim, const int kStage>
void launch_flash_attn_mma(
  torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O) {
  constexpr int kMmaM = 16;
  constexpr int kMmaN = 8;
  constexpr int kMmaK = 16;
  constexpr int kMmaTileQ = 2;
  constexpr int kMmaTileP = 2;
  constexpr int kMmaTileK = 4;
  constexpr int kMmaTileV = 4;
  constexpr int kWarpTileQ = 2;
  constexpr int kWarpTileP = 2;
  constexpr int kWarpTileK = 2;
  constexpr int kWarpTileV = (kHeadDim / (kMmaN*kMmaTileV));
  constexpr int kPad = 0;
  constexpr int Br = kMmaM * kMmaTileQ * kWarpTileQ; // 16*2*2=64
  constexpr int Bc = kMmaN * kMmaTileK * kWarpTileK; // 8*4*2=64

  // Calculate SRAM size needed per block, Q,K,V smem size
  const int smem_max_size = (Br * (kHeadDim + kPad) + 
                            (kStage * Bc * (kHeadDim + kPad)) + 
                            (Bc * (kHeadDim + kPad))) * sizeof(half); 

  const int B  = Q.size(0); 
  const int H  = Q.size(1);
  const int N  = Q.size(2); 
  const int Tr = div_ceil(N, Br); // Tr Q_tile[Br,d]
  const int Tc = div_ceil(N, Bc); // Tc K/V_tile[Bc,d]
  assert(N % Bc == 0); // multiple of Bc=64

  dim3 grid(B, H, Tr); // batch_size x num_heads x Tr(=N/Br)
  dim3 block(WARP_SIZE * kMmaTileQ * kMmaTileK); // 8 Warps per block

  cudaFuncSetAttribute(
    flash_attn_mma_kernel<
      kHeadDim, 
      kMmaM, 
      kMmaN, 
      kMmaK, 
      kMmaTileQ, 
      kMmaTileP, 
      kMmaTileK, 
      kMmaTileV, 
      kWarpTileQ, 
      kWarpTileP, 
      kWarpTileK, 
      kWarpTileV, 
      kStage, 
      kPad
    >,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    98304
  );

  flash_attn_mma_kernel<
    kHeadDim, 
    kMmaM, 
    kMmaN, 
    kMmaK, 
    kMmaTileQ, 
    kMmaTileP, 
    kMmaTileK, 
    kMmaTileV, 
    kWarpTileQ, 
    kWarpTileP, 
    kWarpTileK, 
    kWarpTileV, 
    kStage, 
    kPad
  ><<<grid, block, smem_max_size>>>(
    reinterpret_cast<half*>(Q.data_ptr()),
    reinterpret_cast<half*>(K.data_ptr()),
    reinterpret_cast<half*>(V.data_ptr()),
    reinterpret_cast<half*>(O.data_ptr()),
    N
  );
}

void flash_attn_mma_stages(torch::Tensor Q, torch::Tensor K, 
                           torch::Tensor V, torch::Tensor O, 
                           int stages) {
  CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(O, torch::kHalf)
  const int d = Q.size(3); // B, H, N, d

  if (stages == 2) {
    switch (d)
    {
    case 64:
      launch_flash_attn_mma<64,  2>(Q, K, V, O);
      break;
    case 96:
      launch_flash_attn_mma<96,  2>(Q, K, V, O);
      break;
    case 128:
      launch_flash_attn_mma<128, 2>(Q, K, V, O);
      break;
    default:
      throw std::runtime_error("headdim not support!");
      break;
    }
  } else {
    switch (d)
    {
    case 64:
      launch_flash_attn_mma<64,  1>(Q, K, V, O);
      break;
    case 96:
      launch_flash_attn_mma<96,  1>(Q, K, V, O);
      break;
    case 128:
      launch_flash_attn_mma<128, 1>(Q, K, V, O);
      break;
    default:
      throw std::runtime_error("headdim not support!");
      break;
    }
  }
}
