#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/types.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

#define CP_ASYNC_CA(dst, src, bytes)                           \
asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" \
              ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                           \
asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" \
              ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))


template<const int BM=128, const int BN=128, const int BK=16, 
         const int TM=8, const int TN=8, const int OFFSET=0>
__global__ void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_kernel(
  half* a, half* b, half* c, int M, int N, int K) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx; // tid within the block
  // 2*128*16*2=8KB, 2*16*128*2=8KB
  __shared__ half s_a[2][BK][BM+OFFSET], s_b[2][BK][BN+OFFSET]; 
  half r_load_a[TM]; // 8
  half r_load_b[TN]; // 8
  half r_comp_a[TM]; // 8
  half r_comp_b[TN]; // 8
  half r_c[TM][TN] = {__float2half(0.0f)}; // 8x8
  
  // 0. 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
  // 对于s_a每行16个数据，每个线程读取8个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读8个数据，需要16个线程；总共16行，需要16x16=256个线程
  int load_smem_b_k = tid / 16; // row 0~15
  int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
  // 1. 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  // bk = 0 is loading here, buffer 0 
  {
    int load_gmem_a_k = load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;   
    // TODO: cp.async load b 16 bytes first, then sync load a.
    // wait smem b ready after sync load a.
    // s_b: global -> shared, s_a: global -> tmp registers -> shared
    LDST128BITS(s_b[0][load_smem_b_k][load_smem_b_n]) = (
      LDST128BITS(b[load_gmem_b_addr]));
    LDST128BITS(r_load_a[0]) = LDST128BITS(a[load_gmem_a_addr]);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      s_a[0][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
    }
    // s_a[0][load_smem_a_k + 0][load_smem_a_m] = r_load_a[0];
    // s_a[0][load_smem_a_k + 1][load_smem_a_m] = r_load_a[1];
    // s_a[0][load_smem_a_k + 2][load_smem_a_m] = r_load_a[2];
    // s_a[0][load_smem_a_k + 3][load_smem_a_m] = r_load_a[3];
    // s_a[0][load_smem_a_k + 4][load_smem_a_m] = r_load_a[4];
    // s_a[0][load_smem_a_k + 5][load_smem_a_m] = r_load_a[5];
    // s_a[0][load_smem_a_k + 6][load_smem_a_m] = r_load_a[6];
    // s_a[0][load_smem_a_k + 7][load_smem_a_m] = r_load_a[7];
  }
  // Without this synchronization, accuracy may occasionally be abnormal.
  __syncthreads(); 

  // bk start from 1，需要注意的是，虽然 bk 从 1 开始，但实际上 bk=1时，使用的是
  // 第0块BK中的数据（已经加载到共享内存s_a[0]和s_b[0]）；bk=2时，实际计算的是第1块
  // BK中的数据。其余以此类推，这个循环结束后，剩下最后一块BK大小的数据需要计算。
  for (int bk = 1; bk < (K + BK - 1) / BK; ++bk) {
    int smem_sel = (bk - 1) & 1; // bk 1->0, bk 2->1, bk 3->0, ...
    int smem_sel_next = bk & 1;  // bk 1->1, bk 2->0, bk 3->1, ...

    int load_gmem_a_k = bk * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = bk * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    LDST128BITS(r_load_a[0]) = LDST128BITS(a[load_gmem_a_addr]);
    LDST128BITS(r_load_b[0]) = LDST128BITS(b[load_gmem_b_addr]);
    
    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[smem_sel][tk][ty * TM]);
      LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[smem_sel][tk][tx * TN]);

      #pragma unroll
      for (int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }

    // TODO: cp.async load b 16 bytes first, then sync load a.
    // wait smem b ready after sync load a.
    // s_b: global -> shared, s_a: global -> tmp registers -> shared  
    // s_a[smem_sel_next][load_smem_a_k + 0][load_smem_a_m] = r_load_a[0];
    // s_a[smem_sel_next][load_smem_a_k + 1][load_smem_a_m] = r_load_a[1];
    // s_a[smem_sel_next][load_smem_a_k + 2][load_smem_a_m] = r_load_a[2];
    // s_a[smem_sel_next][load_smem_a_k + 3][load_smem_a_m] = r_load_a[3];
    // s_a[smem_sel_next][load_smem_a_k + 4][load_smem_a_m] = r_load_a[4];
    // s_a[smem_sel_next][load_smem_a_k + 5][load_smem_a_m] = r_load_a[5];
    // s_a[smem_sel_next][load_smem_a_k + 6][load_smem_a_m] = r_load_a[6];
    // s_a[smem_sel_next][load_smem_a_k + 7][load_smem_a_m] = r_load_a[7];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      s_a[smem_sel_next][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
    }
    LDST128BITS(s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]) = (
      LDST128BITS(r_load_b[0]));

    __syncthreads();
  }

  // 计算剩下最后一块BK
  #pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[1][tk][ty * TM]);
    LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[1][tk][tx * TN]);
    
    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
      #pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
      }
    }
  }
  
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    int store_gmem_c_m = by * BM + ty * TM + i;
    int store_gmem_c_n = bx * BN + tx * TN;
    int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
    LDST128BITS(c[store_gmem_c_addr]) = LDST128BITS(r_c[i][0]);
  }
}

template<const int BM=128, const int BN=128, const int BK=16, 
         const int TM=8, const int TN=8, const int OFFSET=0>
__global__ void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async_kernel(
  half* a, half* b, half* c, int M, int N, int K) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx; // tid within the block
  // 2*128*16*2=8KB, 2*16*128*2=8KB
  __shared__ half s_a[2][BK][BM+OFFSET], s_b[2][BK][BN+OFFSET]; 
  __shared__ half s_a_pre[BM][BK];
  half r_load_a[TM]; // 8
  half r_comp_a[TM]; // 8
  half r_comp_b[TN]; // 8
  half r_c[TM][TN] = {__float2half(0.0f)}; // 8x8
  
  // 0. 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=16 按行读取 A行主序
  // 对于s_a每行16个数据，每个线程读取8个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读8个数据，需要16个线程；总共16行，需要16x16=256个线程
  int load_smem_b_k = tid / 16; // row 0~15
  int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
  // 1. 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  // bk = 0 is loading here, buffer 0 
  {
    int load_gmem_a_k = load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;   
    
    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
      &s_a_pre[load_smem_a_m][load_smem_a_k]);
    CP_ASYNC_CG(load_smem_a_ptr, &a[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
      &s_b[0][load_smem_b_k][load_smem_b_n]);
    CP_ASYNC_CG(load_smem_b_ptr, &b[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);
    
    // load 8 half in 1 memory issue.
    LDST128BITS(r_load_a[0]) = LDST128BITS(s_a_pre[
      load_smem_a_m][load_smem_a_k]);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      s_a[0][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
    }
  }
  __syncthreads(); 

  for (int bk = 1; bk < (K + BK - 1) / BK; ++bk) {
    int smem_sel = (bk - 1) & 1; // bk 1->0, bk 2->1, bk 3->0, ...
    int smem_sel_next = bk & 1;  // bk 1->1, bk 2->0, bk 3->1, ...

    int load_gmem_a_k = bk * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = bk * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[smem_sel][tk][ty * TM]);
      LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[smem_sel][tk][tx * TN]);

      #pragma unroll
      for (int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }
 
    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
      &s_a_pre[load_smem_a_m][load_smem_a_k]);
    CP_ASYNC_CG(load_smem_a_ptr, &a[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
      &s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]);
    CP_ASYNC_CG(load_smem_b_ptr, &b[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);

    // load 8 half in 1 memory issue.
    LDST128BITS(r_load_a[0]) = LDST128BITS(s_a_pre[
      load_smem_a_m][load_smem_a_k]);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      s_a[smem_sel_next][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
    }
    __syncthreads();
  }

  #pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[1][tk][ty * TM]);
    LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[1][tk][tx * TN]);
    
    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
      #pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
      }
    }
  }
  
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    int store_gmem_c_m = by * BM + ty * TM + i;
    int store_gmem_c_n = bx * BN + tx * TN;
    int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
    LDST128BITS(c[store_gmem_c_addr]) = LDST128BITS(r_c[i][0]);
  }
}

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

// t 8x8 fp16x8 pack, double buffers, k 16
void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 16; 
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN/TN, BM/TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_offset(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 16; 
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN/TN, BM/TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_kernel<BM, BN, BK, TM, TN, 8><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

// t 8x8 fp16x8 pack, double buffers, k 16, copy async
void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 16; 
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN/TN, BM/TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async_kernel<
    BM, BN, BK, TM, TN><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async_offset(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 16; 
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN/TN, BM/TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async_kernel<
    BM, BN, BK, TM, TN, 8><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

