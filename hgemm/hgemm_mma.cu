#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <torch/types.h>
#include <torch/extension.h>
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
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define LDMATRIX_X1(R, addr) asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))
#define LDMATRIX_X2(R0, R1, addr) asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))
#define LDMATRIX_X1_T(R, addr) asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr) asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" : "=r"(RD0), "=r"(RD1) : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

HOST_DEVICE_INLINE 
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// only 1 warp per block(32 threads), m16n8k16. A, B, C: all row_major.
template<const int MMA_M=16, const int MMA_N=8, const int MMA_K=16>
__global__ void hgemm_mma_m16n8k16_naive_kernel(half* A, half* B, half* C, 
                                                int M, int N, int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K);
  constexpr int BM = MMA_M; // 16
  constexpr int BN = MMA_N; // 8
  constexpr int BK = MMA_K; // 16

  __shared__ half s_a[MMA_M][MMA_K]; // 16x16
  __shared__ half s_b[MMA_K][MMA_N]; // 16x8
  __shared__ half s_c[MMA_M][MMA_N]; // 16x8

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0
  const int lane_id = tid % WARP_SIZE; // 0~31

  // s_a[16][16], 每行16，每线程load 8，需要2线程，共16行，需2x16=32线程
  const int load_smem_a_m = tid / 2; // row 0~15
  const int load_smem_a_k = (tid % 2) * 8; // col 0,8
  // s_b[16][8], 每行8，每线程load 8，需要1线程，共16行，需16线程，只需一半线程加载
  const int load_smem_b_k = tid; // row 0~31, but only use 0~15
  const int load_smem_b_n = 0; // col 0
  const int load_gmem_a_m = by * BM + load_smem_a_m; // global m
  const int load_gmem_b_n = bx * BN + load_smem_b_n; // global n
  if (load_gmem_a_m >= M && load_gmem_b_n >= N) return;

  uint32_t RC[2] = {0, 0};

  #pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {
    // gmem_a -> smem_a
    int load_gmem_a_k = k * MMA_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    LDST128BITS(s_a[load_smem_a_m][load_smem_a_k]) = (
      LDST128BITS(A[load_gmem_a_addr]));

    // gmem_b -> smem_b
    if (lane_id < MMA_K) {
      int load_gmem_b_k = k * MMA_K + load_smem_b_k; // global row of b
      int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 
      LDST128BITS(s_b[load_smem_b_k][load_smem_b_n]) = (
        LDST128BITS(B[load_gmem_b_addr]));
    }
    __syncthreads(); 

    uint32_t RA[4];
    uint32_t RB[2];
    
    // ldmatrix for s_a, ldmatrix.trans for s_b.
    uint32_t load_smem_a_addr = __cvta_generic_to_shared(
      &s_a[lane_id % 16][(lane_id / 16) * 8]); // (0,1)*8 -> 0,8
    LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], load_smem_a_addr);
    uint32_t load_smem_b_addr = __cvta_generic_to_shared(
      &s_b[lane_id % 16][0]);
    LDMATRIX_X2_T(RB[0], RB[1], load_smem_b_addr);

    HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);

    __syncthreads();
  }
  
  // s_c[16][8], https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
  // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
  // [0~7][0~3 u32 -> 0~7 f16], [8~15][0~3 u32 -> 0~7 f16]
  LDST32BITS(s_c[lane_id / 4    ][(lane_id % 4) * 2]) = LDST32BITS(RC[0]); 
  LDST32BITS(s_c[lane_id / 4 + 8][(lane_id % 4) * 2]) = LDST32BITS(RC[1]);

  __syncthreads();

  // store s_c[16][8]
  if (lane_id < MMA_M) {
    // store 128 bits per memory issue.
    int store_gmem_c_m = by * BM + lane_id;
    int store_gmem_c_n = bx * BN;
    int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
    LDST128BITS(C[store_gmem_c_addr]) = (LDST128BITS(s_c[lane_id][0]));
  }
}

// --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
if(((T).options().dtype() != (th_type))) {                   \
  std::cout << "Tensor Info:" << (T).options() << std::endl; \
  throw std::runtime_error("values must be "#th_type);       \
}

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)           \
if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { \
  throw std::runtime_error("Tensor size mismatch!");  \
}

// only 1 warp per block(32 threads), m16n8k16. A, B, C: all row_major.
void hgemm_mma_m16n8k16_naive(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16; 

  dim3 block(WARP_SIZE);
  dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));
 
  hgemm_mma_m16n8k16_naive_kernel<
    MMA_M, MMA_N, MMA_K><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}
