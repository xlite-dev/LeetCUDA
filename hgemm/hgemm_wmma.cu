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
#define LDST64BITS(value) (reinterpret_cast<float2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
// NOTE: All kernels here are modified from: https://github.com/Bruce-Lee-LY/cuda_hgemm, many thanks~
// I have changed all kernels to support A and B matrix with row-major support inorder to compare with
// the kernels using CUDA Cores in hgemm.cu and hgemm_async.cu. reference: 
// https://github.com/Bruce-Lee-LY/cuda_hgemm
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions
// https://github.com/NVIDIA/cuda-samples/tree/master/Samples/3_CUDA_Features/cudaTensorCoreGemm

HOST_DEVICE_INLINE 
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// 1 warp per block(32 threads), m16n16k16. A, B, C: all row_major.
template<const int WMMA_M=16, const int WMMA_N=16, const int WMMA_K=16>
__global__ void hgemm_wmma_m16n16k16_naive_kernel(half* A, half* B, half* C, 
                                                  int M, int N, int K) {
  const int K_tiles = div_ceil(K, WMMA_K);
  const int warp_row = blockIdx.y * WMMA_M;
  const int warp_col = blockIdx.x * WMMA_N;
  if (warp_row >= M && warp_col >= N) return;
  
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
  wmma::fill_fragment(C_frag, 0.0);
  
  #pragma unroll
  for (int i = 0; i < K_tiles; ++i) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;
    
    wmma::load_matrix_sync(A_frag, A + warp_row * K     + i * WMMA_K, K);
    wmma::load_matrix_sync(B_frag, B + (i * WMMA_K) * N + warp_col,   N);

    wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
  }
  wmma::store_matrix_sync(C + warp_row * N + warp_col, C_frag, N, wmma::mem_row_major);
}

// TODO: m16n16k16 wmma with smem,  A, B, C: all row_major.

// --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
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

// 1 warp per block(32 threads), m16n16k16. A, B, C: all row_major.
void hgemm_wmma_m16n16k16_naive(
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
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16; 

  dim3 block(WARP_SIZE);
  dim3 grid(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M));
 
  hgemm_wmma_m16n16k16_naive_kernel<
    WMMA_M, WMMA_N, WMMA_K><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}
