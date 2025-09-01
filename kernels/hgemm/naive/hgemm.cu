#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP16
// HGEMM naive: compute one c[i,j]
// element per threads, all row major
__global__ void hgemm_naive_f16_kernel(half *a, half *b, half *c, int M, int N,
                                       int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;

  if (m < M && n < N) {
    half psum = 0.0;
#pragma unroll
    for (int k = 0; k < K; k++) {
      // m row in a matrix, n col in b matrix
      psum += a[m * K + k] * b[k * N + n];
    }
    c[m * N + n] = psum; // c[m,n]
  }
}

// HGEMM: Block Tile + K Tile, with smem
// Block Tile (BM, BN) + K Tile (BK=32)
// grid((N + BN - 1) / BN, (M + BM - 1) / BM), block(BN, BM)
// a: MxK, b: KxN, c: MxN, compute: c = a * b, all row major
template <const int BM = 32, const int BN = 32, const int BK = 32>
__global__ void hgemm_sliced_k_f16_kernel(half *a, half *b, half *c, int M,
                                          int N, int K) {
  // [1] Block Tile: 32x32的block处理c上一块32x32的元素计算
  // [2]     K Tile: 使用共享内存，并将K分块为BK大小的块
  __shared__ half s_a[BM][BK], s_b[BK][BN];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx; // tid within the block
  // load values to shared memory, 32x32 threads working together
  // to fetch data along the row direction of a and b both for s_a
  // and s_b 32x32x4x2=8KB, we use 32x32 threads within block to
  // load 32x32 elements from global memory to shared memory, namely,
  // each thread will load 1 element.
  int load_smem_a_m = tid / 32; // 0~31, tid / 32, tid / BM, threadIdx.y
  int load_smem_a_k = tid % 32; // 0~31, tid % 32, tid % BK, threadIdx.x
  int load_smem_b_k = tid / 32; // 0~31, tid / 32, tid / BK, threadIdx.y
  int load_smem_b_n = tid % 32; // 0~31, tid % 32, tid % BN, threadIdx.x
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    return;

  half sum = __float2half(0.f);
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_addr];
    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_addr];
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      int comp_smem_a_m = load_smem_a_m;
      int comp_smem_b_n = load_smem_b_n;
      sum += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
    }
    __syncthreads();
  }
  int store_gmem_c_m = load_gmem_a_m;
  int store_gmem_c_n = load_gmem_b_n;
  int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
  c[store_gmem_c_addr] = sum;
}

// HGEMM: Block Tile + Thread Tile + K Tile + half2x2, with smem
// BK:TILE_K=8 BM=BN=128
// TM=TN=8 增加计算密度 BM/TM=16 BN/TN=16
// dim3 blockDim(BN/TN, BM/TM);
// dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM)
template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8>
__global__ void hgemm_t_8x8_sliced_k_f16x4_kernel(half *a, half *b, half *c,
                                                  int M, int N, int K) {
  // [1]  Block Tile: 一个16x16的block处理C上大小为128X128的一个目标块
  // [2] Thread Tile: 每个thread负责计算TM*TN(8*8)个元素，增加计算密度
  // [3]      K Tile: 将K分块，每块BK大小，迭代(K+BK-1/BK)次，
  //                  每次计算TM*TN个元素各自的部分乘累加
  // [4]   Vectorize: 减少load和store指令，使用half2

  // 线程总数16x16=256，每个线程负责计算8x8的元素
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx;  // tid within the block
  __shared__ half s_a[BM][BK], s_b[BK][BN]; // 2*128*8*2=4KB

  // 0. 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
  // 对于s_a每行8个数据，每个线程读取4个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2; // tid/2 (128/8)*(128/8)=256 threads per block,
                               // tid/2->[0,128), BM=128 0~127
  int load_smem_a_k =
      (tid % 2 == 0) ? 0 : 4; // (tid%2 == 0) ? 0 : 4, col of s_a 0,4
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=8 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读4个数据，需要32个线程；总共8行，需要32x8=256个线程
  int load_smem_b_k = tid / 32;       // tid/32, row of s_b 256/32=8 行 0~7
  int load_smem_b_n = (tid % 32) * 4; // (tid % 32) * 4, col of s_b 0,4,...,124
  // 1. 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数
  // 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    return;

  half r_c[TM][TN] = {__float2half(0.0f)}; // 8x8
  // 2. 先对K进行分块，每块BK大小
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    // 加载数据到共享内存smem s_a BM*BK 128*8 vectorize float4
    int load_gmem_a_k = bk * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    HALF2(s_a[load_smem_a_m][load_smem_a_k + 0]) =
        HALF2(a[load_gmem_a_addr + 0]);
    HALF2(s_a[load_smem_a_m][load_smem_a_k + 2]) =
        HALF2(a[load_gmem_a_addr + 2]);
    // 加载数据到共享内存smem s_b BK*BN 8*128 vectorize float4
    int load_gmem_b_k = bk * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    HALF2(s_b[load_smem_b_k][load_smem_b_n + 0]) =
        HALF2(b[load_gmem_b_addr + 0]);
    HALF2(s_b[load_smem_b_k][load_smem_b_n + 2]) =
        HALF2(b[load_gmem_b_addr + 2]);
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; k++) {
// 3. 每个线程负责计算BM*BN(12x128)中的TM*TN(8x8)个元素
#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n++) {
          // k from 0~7，0 ~ BK, ty and tx range from 0 to 15, 16x8=128
          int comp_smem_a_m = ty * TM + m; // 128*8 128/TM(8)=16 M方向 16线程
          int comp_smem_b_n = tx * TN + n; // 8*128 128/TN(8)=16 N方向 16线程
          r_c[m][n] += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < TM; ++m) {
    int store_gmem_c_m = by * BM + ty * TM + m;
#pragma unroll
    for (int n = 0; n < TN; n += 2) {
      int store_gmem_c_n = bx * BN + tx * TN + n;
      int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
      HALF2(c[store_gmem_c_addr]) = HALF2(r_c[m][n]);
    }
  }
}

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8>
__global__ void hgemm_t_8x8_sliced_k_f16x4_pack_kernel(half *a, half *b,
                                                       half *c, int M, int N,
                                                       int K) {
  // [1]  Block Tile: 一个16x16的block处理C上大小为128X128的一个目标块
  // [2] Thread Tile: 每个thread负责计算TM*TN(8*8)个元素，增加计算密度
  // [3]      K Tile: 将K分块，每块BK大小，迭代(K+BK-1/BK)次，
  //                  每次计算TM*TN个元素各自的部分乘累加
  // [4]   Vectorize: 减少load和store指令，使用half2

  // 线程总数16x16=256，每个线程负责计算8x8的元素
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx;  // tid within the block
  __shared__ half s_a[BM][BK], s_b[BK][BN]; // 2*128*8*2=4KB

  // 0. 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
  // 对于s_a每行8个数据，每个线程读取4个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2; // tid/2 (128/8)*(128/8)=256 threads per block,
                               // tid/2->[0,128), BM=128 0~127
  int load_smem_a_k =
      (tid % 2 == 0) ? 0 : 4; // (tid%2 == 0) ? 0 : 4, col of s_a 0,4
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=8 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读4个数据，需要32个线程；总共8行，需要32x8=256个线程
  int load_smem_b_k = tid / 32;       // tid/32, row of s_b 256/32=8 行 0~7
  int load_smem_b_n = (tid % 32) * 4; // (tid % 32) * 4, col of s_b 0,4,...,124
  // 1. 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数
  // 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    return;

  half r_c[TM][TN] = {__float2half(0.0f)}; // 8x8
  // 2. 先对K进行分块，每块BK大小
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    // 加载数据到共享内存smem s_a BM*BK 128*8 vectorize float4
    int load_gmem_a_k = bk * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    LDST64BITS(s_a[load_smem_a_m][load_smem_a_k]) =
        LDST64BITS(a[load_gmem_a_addr]);
    // 加载数据到共享内存smem s_b BK*BN 8*128 vectorize float4
    int load_gmem_b_k = bk * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    LDST64BITS(s_b[load_smem_b_k][load_smem_b_n]) =
        LDST64BITS(b[load_gmem_b_addr]);

    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; k++) {
// 3. 每个线程负责计算BM*BN(12x128)中的TM*TN(8x8)个元素
#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n++) {
          // k from 0~7，0 ~ BK, ty and tx range from 0 to 15, 16x8=128
          int comp_smem_a_m = ty * TM + m; // 128*8 128/TM(8)=16 M方向 16线程
          int comp_smem_b_n = tx * TN + n; // 8*128 128/TN(8)=16 N方向 16线程
          // r_c[m][n] += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
          r_c[m][n] = __hfma(s_a[comp_smem_a_m][k], s_b[k][comp_smem_b_n],
                             r_c[m][n]); // HFMA(x,y,z)=x*y+z
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < TM; ++m) {
    int store_gmem_c_m = by * BM + ty * TM + m;
#pragma unroll
    for (int n = 0; n < TN; n += 4) {
      int store_gmem_c_n = bx * BN + tx * TN + n;
      int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
      LDST64BITS(c[store_gmem_c_addr]) = LDST64BITS(r_c[m][n]);
    }
  }
}

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8>
__global__ void hgemm_t_8x8_sliced_k_f16x4_bcf_kernel(half *a, half *b, half *c,
                                                      const int M, const int N,
                                                      const int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ half s_a[BK][BM];
  __shared__ half s_b[BK][BN];

  half r_load_a[TM / 2]; // 4
  half r_load_b[TN / 2]; // 4
  half r_comp_a[TM];
  half r_comp_b[TN];
  half r_c[TM][TN] = {__float2half(0.0f)};

  // mapping tid to s_a[BK][BM], for each orginal m-th row, load 4 + 4 K-dim
  // row major values from A matrix, and store it in COL major s_a[BK][BM].
  int load_a_smem_m = tid / 2; // tid / 2，(0,1,2,...,128)
  // (0b00000000 & 0b00000001) << 2 = 0
  // (0b00000001 & 0b00000001) << 2 = 4
  // (0b00000010 & 0b00000001) << 2 = 0
  // (0b00000011 & 0b00000001) << 2 = 4
  int load_a_smem_k = (tid & 1) << 2; // (0,4)
  // mapping tid to s_b[BK][BN], for each orginal k-th row, load 4 + 4 N-dim
  // row major values from B matrix, and store it in ROW major s_b[BK][BN].
  int load_b_smem_k = tid / 32; // 0~8
  // (0b00000000 & 0b00011111) << 2 = 0
  // (0b00000001 & 0b00011111) << 2 = 4
  // (0b00000010 & 0b00011111) << 2 = 8
  // (0b00000011 & 0b00011111) << 2 = 12
  int load_b_smem_n = (tid & 31) << 2; // (0,4,8,12,...,124)

  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;
  if (load_a_gmem_m >= M || load_b_gmem_n >= N)
    return;

  for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    HALF2(r_load_a[0]) = HALF2(a[load_a_gmem_addr + 0]);
    HALF2(r_load_a[2]) = HALF2(a[load_a_gmem_addr + 2]);
    HALF2(r_load_b[0]) = HALF2(b[load_b_gmem_addr + 0]);
    HALF2(r_load_b[2]) = HALF2(b[load_b_gmem_addr + 2]);

    s_a[load_a_smem_k][load_a_smem_m] = r_load_a[0];
    s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    HALF2(s_b[load_b_smem_k][load_b_smem_n + 0]) = HALF2(r_load_b[0]);
    HALF2(s_b[load_b_smem_k][load_b_smem_n + 2]) = HALF2(r_load_b[2]);

    __syncthreads();

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      HALF2(r_comp_a[0]) = HALF2(s_a[tk][ty * TM / 2]);
      HALF2(r_comp_a[2]) = HALF2(s_a[tk][ty * TM / 2 + 2]);
      HALF2(r_comp_a[4]) = HALF2(s_a[tk][ty * TM / 2 + BM / 2]);
      HALF2(r_comp_a[6]) = HALF2(s_a[tk][ty * TM / 2 + BM / 2 + 2]);

      HALF2(r_comp_b[0]) = HALF2(s_b[tk][tx * TN / 2]);
      HALF2(r_comp_b[2]) = HALF2(s_b[tk][tx * TN / 2 + 2]);
      HALF2(r_comp_b[4]) = HALF2(s_b[tk][tx * TN / 2 + BN / 2]);
      HALF2(r_comp_b[6]) = HALF2(s_b[tk][tx * TN / 2 + BN / 2 + 2]);

#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    HALF2(c[store_c_gmem_addr + 0]) = HALF2(r_c[i][0]);
    HALF2(c[store_c_gmem_addr + 2]) = HALF2(r_c[i][2]);
    HALF2(c[store_c_gmem_addr + BN / 2 + 0]) = HALF2(r_c[i][4]);
    HALF2(c[store_c_gmem_addr + BN / 2 + 2]) = HALF2(r_c[i][6]);
  }
#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    HALF2(c[store_c_gmem_addr + 0]) = HALF2(r_c[i + TM / 2][0]);
    HALF2(c[store_c_gmem_addr + 2]) = HALF2(r_c[i + TM / 2][2]);
    HALF2(c[store_c_gmem_addr + BN / 2 + 0]) = HALF2(r_c[i + TM / 2][4]);
    HALF2(c[store_c_gmem_addr + BN / 2 + 2]) = HALF2(r_c[i + TM / 2][6]);
  }
}

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void hgemm_t_8x8_sliced_k_f16x4_pack_bcf_kernel(half *a, half *b,
                                                           half *c, const int M,
                                                           const int N,
                                                           const int K) {
  // threads: 128/8 * 128/8 = 256
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ half s_a[BK][BM + OFFSET]; // 8*128*2=2KB
  __shared__ half s_b[BK][BN + OFFSET]; // 8*128*2=2KB

  half r_load_a[TM / 2];                   // 4
  half r_load_b[TN / 2];                   // 4
  half r_comp_a[TM];                       // 8
  half r_comp_b[TN];                       // 8
  half r_c[TM][TN] = {__float2half(0.0f)}; // 8x8

  // mapping tid to s_a[BK][BM], for each orginal m-th row, load 4 + 4 K-dim
  // row major values from A matrix, and store it in COL major s_a[BK][BM].
  int load_a_smem_m = tid / 2; // tid / 2，(0,1,2,...,128)
  // (0b00000000 & 0b00000001) << 2 = 0
  // (0b00000001 & 0b00000001) << 2 = 4
  // (0b00000010 & 0b00000001) << 2 = 0
  // (0b00000011 & 0b00000001) << 2 = 4
  int load_a_smem_k = (tid & 1) << 2; // (0,4)
  // mapping tid to s_b[BK][BN], for each orginal k-th row, load 4 + 4 N-dim
  // row major values from B matrix, and store it in ROW major s_b[BK][BN].
  int load_b_smem_k = tid / 32; // 0~8
  // (0b00000000 & 0b00011111) << 2 = 0
  // (0b00000001 & 0b00011111) << 2 = 4
  // (0b00000010 & 0b00011111) << 2 = 8
  // (0b00000011 & 0b00011111) << 2 = 12
  int load_b_smem_n = (tid & 31) << 2; // (0,4,8,12,...,124)

  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;
  if (load_a_gmem_m >= M || load_b_gmem_n >= N)
    return;

  for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    LDST64BITS(r_load_a[0]) = LDST64BITS(a[load_a_gmem_addr]);
    LDST64BITS(r_load_b[0]) = LDST64BITS(b[load_b_gmem_addr]);

    // 0. bank layout analysis: s_a[8][128]
    // 4 bytes per bank(32 banks, total 128 bytes, 64 half values),
    // 2 half per bank. smem banks layout for s_a[8][128]:
    // [k=0][m=  [0,1],   [2,3],   [4,5],...,   [62,63]]
    // layer_0   [ b0],   [ b1],   [ b2],...,    [ b31]
    // [k=0][m=[64,65], [66,67], [68,69],..., [126,127]]
    // layer_1   [ b0],   [ b1],   [ b2],...,    [ b31]
    // [k=1][m=  [0,1],   [2,3],   [4,5],...,   [62,63]]
    // layer_2   [ b0],   [ b1],   [ b2],...,    [ b31]
    // [k=1][m=[64,65], [66,67], [68,69],..., [126,127]]
    // layer_3   [ b0],   [ b1],   [ b2],...,    [ b31]
    // ...       ...      ...      ...           ...
    // [k=7][m=  [0,1],   [2,3],   [4,5],...,   [62,63]]
    // layer_14  [ b0],   [ b1],   [ b2],...,    [ b31]
    // [k=7][m=[64,65], [66,67], [68,69],..., [126,127]]
    // layer_15  [ b0],   [ b1],   [ b2],...,    [ b31]
    // 1. bank conficts analysis: s_a[8][128]
    // tid 0   -> m 0,   k 0 -> all access bank 0  (layer_0/2/4/6)
    // tid 1   -> m 0,   k 4 -> all access bank 0  (layer_8/10/12/14)
    // tid 2   -> m 1,   k 0 -> all access bank 0  (layer_0/2/4/6)
    // tid 3   -> m 1,   k 4 -> all access bank 0  (layer_8/10/12/14)
    // tid 4   -> m 2,   k 0 -> all access bank 1  (layer_0/2/4/6)
    // tid 5   -> m 2,   k 4 -> all access bank 1  (layer_8/10/12/14)
    // tid 6   -> m 3,   k 0 -> all access bank 1  (layer_0/2/4/6)
    // tid 7   -> m 3,   k 4 -> all access bank 1  (layer_8/10/12/14)
    // ...        ...           ...                ...
    // tid 28  -> m 14,  k 0 -> all access bank 7  (layer_0/2/4/6)
    // tid 29  -> m 14,  k 4 -> all access bank 7  (layer_8/10/12/14)
    // tid 30  -> m 15,  k 0 -> all access bank 7  (layer_0/2/4/6)
    // tid 31  -> m 15,  k 4 -> all access bank 7  (layer_8/10/12/14)
    // ...        ...           ...                ...
    // tid 252 -> m 126, k 0 -> all access bank 30 (layer_1/3/5/7)
    // tid 253 -> m 126, k 4 -> all access bank 30 (layer_9/11/13/15)
    // tid 254 -> m 127, k 0 -> all access bank 31 (layer_1/3/5/7)
    // tid 255 -> m 127, k 4 -> all access bank 31 (layer_9/11/13/15)
    // conclusion: we still have bank conflicts for smem_a write access,
    // each 4 consecutive threads within warp access the same bank!
    // thus, we still need 4 memory issues as least per warp.
    s_a[load_a_smem_k][load_a_smem_m] = r_load_a[0];     // e.g layer_0 b0
    s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1]; // e.g layer_2 b0
    s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2]; // e.g layer_4 b0
    s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3]; // e.g layer_6 b0
    // 2. bank layout analysis: s_b[8][128] same as s_a[8][128]
    // 3. bank conficts analysis: s_b[8][128]
    // tid 0   -> k 0, n 0   -> all access bank 0&1   (layer_0)
    // tid 1   -> k 0, n 4   -> all access bank 2&3   (layer_0)
    // tid 2   -> k 0, n 8   -> all access bank 4&5   (layer_0)
    // ...        ...         ...                 ...
    // tid 15  -> k 0, n 60  -> all access bank 30&31 (layer_0)
    // tid 16  -> k 0, n 64  -> all access bank 0&1   (layer_1)
    // ...        ...         ...                 ...
    // tid 31  -> k 0, n 124 -> all access bank 30&31 (layer_1)
    // conclusion: we still have bank conflicts within warp, 0&16 -> bank 0,
    // 1&17 -> bank 1, etc. we still need 2 memory issues at least per warp.
    LDST64BITS(s_b[load_b_smem_k][load_b_smem_n]) = LDST64BITS(r_load_b[0]);

    __syncthreads();

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      // bank conflicts analysis, tx/ty 0~15, 0~7 bank 4*8=32 bytes
      // tid 0~15 access bank 0~1,  tid 16~31 access bank 2~3, etc.
      // tid 0,  tk 0 -> ty 0 -> [0][0+0~3],[0][64+0~3] -> bank 0~1(layer_0/1),
      // same address tid 0,  tk 7 -> ty 0 -> [7][0+0~3],[0][64+0~3] -> bank
      // 0~1(layer_14/15), same address tid 15, tk 0 -> ty 0 ->
      // [0][0+0~3],[0][64+0~3] -> bank 0~1(layer_0/1),   same address tid 15,
      // tk 7 -> ty 0 -> [7][0+0~3],[0][64+0~3] -> bank 0~1(layer_14/15), same
      // address tid 16, tk 0 -> ty 1 -> [0][0+4~7],[0][64+4~7] -> bank
      // 2~3(layer_0/1),   same address tid 16, tk 7 -> ty 1 ->
      // [7][0+4~7],[0][64+4~7] -> bank 2~3(layer_14/15), same address tid 31,
      // tk 0 -> ty 1 -> [0][0+4~7],[0][64+4~7] -> bank 2~3(layer_0/1),   same
      // address tid 31, tk 7 -> ty 1 -> [7][0+4~7],[0][64+4~7] -> bank
      // 2~3(layer_14/15), same address
      LDST64BITS(r_comp_a[0]) = LDST64BITS(s_a[tk][ty * TM / 2]);
      LDST64BITS(r_comp_a[4]) = LDST64BITS(s_a[tk][ty * TM / 2 + BM / 2]);
      // if (tid == < 32 && bx == 0 && by == 0) {
      //   printf("tid: %d, tx: %d, ty: %d, [%d][%d]\n", tid, tx, ty, tk, ty *
      //   TM / 2); printf("tid: %d, tx: %d, ty: %d, [%d][%d]\n", tid, tx, ty,
      //   tk, ty * TM / 2 + BM / 2);
      // }
      // conclusion: still have bank conflicts.

      // tid 0/16 access bank 0~1, tid 1/17 access bank 2~3, tid 15/31 access
      // bank 30~31. tid 2/10/18/26 access bank 8~11, tid 7/15/23/31 access bank
      // 28~31, etc. tid 0, tk 0 -> tx 0 -> [0][0+0~3],[0][64+0~3] -> bank
      // 0~1(layer_0/1),   same address tid 0, tk 7 -> tx 0 ->
      // [7][0+0~3],[0][64+0~3] -> bank 0~1(layer_14/15), same address tid 1, tk
      // 0 -> tx 1 -> [0][0+4~7],[0][64+4~7] -> bank 2~3(layer_0/1),   same
      // address tid 1, tk 7 -> tx 1 -> [7][0+4~7],[0][64+4~7] -> bank
      // 2~3(layer_14/15), same address
      LDST64BITS(r_comp_b[0]) = LDST64BITS(s_b[tk][tx * TN / 2]);
      LDST64BITS(r_comp_b[4]) = LDST64BITS(s_b[tk][tx * TN / 2 + BN / 2]);
      // conclusion: s_b still have many bank conflicts within warp,
      // tid 0/16 access same bank 0&1, etc. need 2 memory issues.

#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
          r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    LDST64BITS(c[store_c_gmem_addr]) = LDST64BITS(r_c[i][0]);
    LDST64BITS(c[store_c_gmem_addr + BN / 2]) = LDST64BITS(r_c[i][4]);
  }
#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    LDST64BITS(c[store_c_gmem_addr]) = LDST64BITS(r_c[i + TM / 2][0]);
    LDST64BITS(c[store_c_gmem_addr + BN / 2]) = LDST64BITS(r_c[i + TM / 2][4]);
  }
}

// 下面这个kernel我就简单说说，细节我就不说了，因为里面的各种线程index的变化，看的我有点头疼
// 以256*256的a和b举例，这里BM(128)和BN(128)表示每个block计算的c中的结果大小是128 * 128
// 然后是分块矩阵乘，这里对K维度做分块，相当于需要for循环遍历k维度
// 还有就是这里每个线程会负责很多结果的计算。剩余的我没更加深入的看了
template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void hgemm_t_8x8_sliced_k_f16x8_pack_bcf_kernel(half *a, half *b,
                                                           half *c, const int M,
                                                           const int N,
                                                           const int K) {
  // 下面代码注释中,以如下设置为例进行讲解
  // 下面注释中,关于变量的变化范围默认只列出了block(0,0)的情况
  // M = 256, N = 256, K = 128, OFFset 8, 其余参数与模板默认值保持一致
  // block(16, 16)
  // grid(2, 2)
  // 这里BM(128)和BN(128)表示每个block计算的c中的结果大小是128 * 128
  // BK表示K维度分块的大小是8

  // threads: 128/8 * 128/8 = 256
  const int bx = blockIdx.x; // 所有block 0, 1
  const int by = blockIdx.y; // 所有block 0, 1
  const int tx = threadIdx.x; // 0 ~ 15
  const int ty = threadIdx.y; // 0 ~ 15
  const int tid = ty * blockDim.x + tx; // tx: 0 ~ 15；ty * blockDim.x + tx: 0~ 255. 这里tid就是访问smem的一维线性地址，tid的计算方式其实就是，这里smem是个二维矩阵，所以访问这个smem的方式就是 ty * blockDim.x(行偏移) + tx(列偏移)

  __shared__ half s_a[BK][BM + OFFSET]; // 8*128*2=2KB // s_a[8][128 + 8]
  __shared__ half s_b[BK][BN + OFFSET]; // 8*128*2=2KB // s_b[8][128 + 8]

  half r_load_a[TM / 2];                   // 4个half
  half r_load_b[TN / 2];                   // 4个half
  half r_comp_a[TM];                       // 8个half
  half r_comp_b[TN];                       // 8个half
  half r_c[TM][TN] = {__float2half(0.0f)}; // 8x8个half

  // mapping tid to s_a[BK][BM], for each orginal m-th row, load 4 + 4 K-dim
  // row major values from A matrix, and store it in COL major s_a[BK][BM].
  // 将线程ID映射到s_a[BK][BM]，对于每个原始的第m行，加载4+4个K维度
  // 从A矩阵的行主序数值，并将其存储到列主序的s_a[BK][BM]中。
  // 
  int load_a_smem_m = tid / 2; // [0, 0, 1, 1, ..., 7, 7, .., 127, 127] , /2 表示变化周期为2，每个周期中的值都相等
  // (0b00000000 & 0b00000001) << 2 = 0
  // (0b00000001 & 0b00000001) << 2 = 4
  // (0b00000010 & 0b00000001) << 2 = 0
  // (0b00000011 & 0b00000001) << 2 = 4
  int load_a_smem_k = (tid & 1) << 2; // (0,4) // 这里&1表示以load_a_smem_k是一个周期变化的，周期为2。<<2 表示，变化的间隔为4
  // mapping tid to s_b[BK][BN], for each orginal k-th row, load 4 + 4 N-dim
  // row major values from B matrix, and store it in ROW major s_b[BK][BN].
  // 将线程ID映射到s_b[BK][BN]，对于每个原始的第k行，加载4+4个N维度
  // 从B矩阵的行主序数值，并将其存储到行主序的s_b[BK][BN]中。
  int load_b_smem_k = tid / 32;  //[0, 0, ..32个0 ,0 ,1, 1, 1 ,... 7, 7, 7] /2 表示变化周期为32，每个周期中的值都相等
  // (0b00000000 & 0b00011111) << 2 = 0
  // (0b00000001 & 0b00011111) << 2 = 4
  // (0b00000010 & 0b00011111) << 2 = 8
  // (0b00000011 & 0b00011111) << 2 = 12
  int load_b_smem_n = (tid & 31) << 2; //这里&31表示，变化周期为32。<<2表示，变化间隔为4 // tid从0到31, load_b_smem_n从0变化到124,以4为间隔变化,以此循环.当tid到32时,load_b_smem_n又变为0,继续循环

  int load_a_gmem_m = by * BM + load_a_smem_m; // 0 ~ 255 //完成了row_smem_a -> row_gmem_a的映射, 映射的方式就是row_smem_a + offset = row_gmem_a, 这里算的load_a_gmem_m是在整个gmem上的变化情况(也可以理解为在整个grid上的变化情况),所以需要在block的smem的变化情况的基础之上,加上了grid级别的偏移量by * BM
  int load_b_gmem_n = bx * BN + load_b_smem_n; // 0 ~ 188 // 完成了col_smem_b -> col_gmem_b的映射, 映射的方式就是col_smem_b + offset = col_gmem_b, 这里同理,算的是整个gmem上的变化情况(也可以理解为在整个grid上的变化情况),所以在原本的load_b_smem_n基础上又加上了grid级别(gmem级别)的偏移bx * BN
  if (load_a_gmem_m >= M || load_b_gmem_n >= N)
    return;

  for (int bk = 0; bk < (K + BK - 1) / BK; bk++) { // (K + BK - 1) / BK = (128 + 8 - 1) / 8 = 16
    int load_a_gmem_k = bk * BK + load_a_smem_k; //gmem_a在k维度上的index,也是smem_k + offset = gmem_k
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k; // 这里其实就是行偏移(load_a_gmem_m)乘以列数K加上再加上列偏移(load_a_gmem_k)来访问A矩阵
    int load_b_gmem_k = bk * BK + load_b_smem_k; //gmem_b在k维度上的index,也是smem_b + offset = gmem_b
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n; //这里是行偏移(load_b_gmem_k * N)乘以列数N再加上列偏移(load_b_gmem_n)来访问B矩阵


    /*
    关于a[load_a_gmem_addr]访问A矩阵的图解: 
        矩阵A在全局内存中的完整布局 (行主序存储，包含by=0和by=1两种情况)
    K维度: 0   4   8   12  16  20  24  28  ...  120 124
        ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
   M=0  │ 0 │ 4 │ 8 │12 │16 │20 │24 │28 │...│120│124│  ← by=0,tid=0,1访问这行
        ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
   M=1  │128│132│136│140│144│148│152│156│...│248│252│  ← by=0,tid=2,3访问这行  
        ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
   M=2  │256│260│264│268│272│276│280│284│...│376│380│  ← by=0,tid=4,5访问这行
        ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
   M=3  │384│388│392│396│400│404│408│412│...│504│508│  ← by=0,tid=6,7访问这行
        ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
    .   │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │      by=0 block
    .   │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │      处理行范围：M=0~127
    .   │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │
        ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
   M=127│16256│16260│16264│16268│...│16376│16380│     ← by=0,tid=254,255访问
        ├═══┼═══┼═══┼═══┼═══┼═══┼═══┼═══┼═══┼═══┼═══┤ ← Block边界线
   M=128│16384│16388│16392│16396│...│16504│16508│     ← by=1,tid=0,1访问这行
        ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
   M=129│16512│16516│16520│16524│...│16632│16636│     ← by=1,tid=2,3访问这行
        ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
   M=130│16640│16644│16648│16652│...│16760│16764│     ← by=1,tid=4,5访问这行
        ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
   M=131│16768│16772│16776│16780│...│16888│16892│     ← by=1,tid=6,7访问这行
        ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
    .   │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │      by=1 block
    .   │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │      处理行范围：M=128~255
    .   │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │
        ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
   M=255│32640│32644│32648│32652│...│32760│32764│     ← by=1,tid=254,255访问
        └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
        ^   ^               ^               ^       ^
        │   │               │               │       │
      bk=0 bk=0           bk=1             bk=15   bk=15
      tid偶  tid奇                                 

    线程访问模式分析 (扩展到by=0和by=1)：
    ┌──────┬─────────────┬──────────┬──────────────────────────────────────┐
    │Block │   线程ID    │ 访问行号  │        16次bk循环访问的列位置         │
    ├──────┼─────────────┼──────────┼──────────────────────────────────────┤
    │by=0  │ tid=0(偶)   │   M=0    │  0,  8, 16, 24, 32, ..., 112, 120   │
    │by=0  │ tid=1(奇)   │   M=0    │  4, 12, 20, 28, 36, ..., 116, 124   │
    ├──────┼─────────────┼──────────┼──────────────────────────────────────┤
    │by=0  │ tid=2(偶)   │   M=1    │  0,  8, 16, 24, 32, ..., 112, 120   │
    │by=0  │ tid=3(奇)   │   M=1    │  4, 12, 20, 28, 36, ..., 116, 124   │
    ├──────┼─────────────┼──────────┼──────────────────────────────────────┤
    │ ... │     ...     │   ...    │                 ...                  │
    ├──────┼─────────────┼──────────┼──────────────────────────────────────┤
    │by=0  │tid=254(偶)  │  M=127   │  0,  8, 16, 24, 32, ..., 112, 120   │
    │by=0  │tid=255(奇)  │  M=127   │  4, 12, 20, 28, 36, ..., 116, 124   │
    ├══════┼═════════════┼══════════┼══════════════════════════════════════┤
    │by=1  │ tid=0(偶)   │  M=128   │  0,  8, 16, 24, 32, ..., 112, 120   │
    │by=1  │ tid=1(奇)   │  M=128   │  4, 12, 20, 28, 36, ..., 116, 124   │
    ├──────┼─────────────┼──────────┼──────────────────────────────────────┤
    │by=1  │ tid=2(偶)   │  M=129   │  0,  8, 16, 24, 32, ..., 112, 120   │
    │by=1  │ tid=3(奇)   │  M=129   │  4, 12, 20, 28, 36, ..., 116, 124   │
    ├──────┼─────────────┼──────────┼──────────────────────────────────────┤
    │ ... │     ...     │   ...    │                 ...                  │
    ├──────┼─────────────┼──────────┼──────────────────────────────────────┤
    │by=1  │tid=254(偶)  │  M=255   │  0,  8, 16, 24, 32, ..., 112, 120   │
    │by=1  │tid=255(奇)  │  M=255   │  4, 12, 20, 28, 36, ..., 116, 124   │
    └──────┴─────────────┴──────────┴──────────────────────────────────────┘

    地址计算示例 (K=128, 对比by=0和by=1)：
    ┌──────┬─────┬───────────┬─────────────┬──────────────────────────────┐
    │Block │ tid │ bk循环次数 │   访问行    │        地址计算过程           │
    ├──────┼─────┼───────────┼─────────────┼──────────────────────────────┤
    │by=0  │  0  │   bk=0    │    M=0     │ 0×128 + 0  = 0               │
    │by=0  │  0  │   bk=1    │    M=0     │ 0×128 + 8  = 8               │
    │by=0  │  0  │   bk=2    │    M=0     │ 0×128 + 16 = 16              │
    │by=0  │  0  │    ...    │    M=0     │ 0×128 + (bk×8) = bk×8        │
    │by=0  │  0  │   bk=15   │    M=0     │ 0×128 + 120 = 120            │
    ├──────┼─────┼───────────┼─────────────┼──────────────────────────────┤
    │by=0  │  1  │   bk=0    │    M=0     │ 0×128 + 4  = 4               │
    │by=0  │  1  │   bk=1    │    M=0     │ 0×128 + 12 = 12              │
    │by=0  │  1  │   bk=2    │    M=0     │ 0×128 + 20 = 20              │
    │by=0  │  1  │    ...    │    M=0     │ 0×128 + (bk×8+4) = bk×8+4    │
    │by=0  │  1  │   bk=15   │    M=0     │ 0×128 + 124 = 124            │
    ├──────┼─────┼───────────┼─────────────┼──────────────────────────────┤
    │by=0  │ 255 │   bk=15   │   M=127    │ 127×128 + 124 = 16380        │
    ├══════┼═════┼═══════════┼═════════════┼══════════════════════════════┤
    │by=1  │  0  │   bk=0    │   M=128    │ 128×128 + 0  = 16384         │
    │by=1  │  0  │   bk=1    │   M=128    │ 128×128 + 8  = 16392         │
    │by=1  │  0  │   bk=2    │   M=128    │ 128×128 + 16 = 16400         │
    │by=1  │  0  │    ...    │   M=128    │ 128×128 + (bk×8) = 16384+bk×8│
    │by=1  │  0  │   bk=15   │   M=128    │ 128×128 + 120 = 16504        │
    ├──────┼─────┼───────────┼─────────────┼──────────────────────────────┤
    │by=1  │  1  │   bk=0    │   M=128    │ 128×128 + 4  = 16388         │
    │by=1  │  1  │   bk=1    │   M=128    │ 128×128 + 12 = 16396         │
    │by=1  │  1  │   bk=2    │   M=128    │ 128×128 + 20 = 16404         │
    │by=1  │  1  │    ...    │   M=128    │ 128×128 + (bk×8+4) = 16388+bk×8│
    │by=1  │  1  │   bk=15   │   M=128    │ 128×128 + 124 = 16508        │
    ├──────┼─────┼───────────┼─────────────┼──────────────────────────────┤
    │by=1  │ 255 │   bk=15   │   M=255    │ 255×128 + 124 = 32764        │
    └──────┴─────┴───────────┴─────────────┴──────────────────────────────┘

    访问模式总结：
    • 每两个相邻线程访问同一行的相邻位置
    • 奇偶线程在同一行内访问位置相差4
    • 每次bk循环，同一线程在其对应行内向右移动8个位置
    • by=0处理矩阵A的前128行（M=0~127），by=1处理后128行（M=128~255）
    • by=1的所有地址都比by=0对应位置多了128×128=16384的偏移
    • 不同block完全并行，无内存访问冲突
    • 这种模式保证了内存合并访问，提高带宽利用率
    
    */
    // 这里是向量化访存，一次性读4个half给r_load_a
    LDST64BITS(r_load_a[0]) = LDST64BITS(a[load_a_gmem_addr]);

    /*
下面标出load_b_gmem_addr的变化情况：
    矩阵B在全局内存中的完整布局 (行主序存储, K×N = 128×256)
N维度: 0   4   8   12  ...   124  128  132  136  ...   252
     ┌───┬───┬───┬───┬───┬───┬═══┬───┬───┬───┬───┬───┐
K=0  │ 0 │ 4 │ 8 │12 │...│124│128│132│136│...│252│  ← bk=0,所有tid访问这行，下同
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
K=1  │256│260│264│268│...│380│384│388│392│...│508│  ← bk=0
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
K=2  │512│516│520│524│...│636│640│644│648│...│764│  ← bk=0
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
K=3  │768│772│776│780│...│892│896│900│904│..│1020│  ← bk=0
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
K=4  │1024│1028│1032│1036│...│1148│1152│1156│..│1276│ ← bk=0
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
K=5  │1280│1284│1288│1292│...│1404│1408│1412│..│1532│ ← bk=0
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
K=6  │1536│1540│1544│1548│...│1660│1664│1668│..│1788│ ← bk=0
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
K=7  │1792│1796│1800│1804│...│1916│1920│1924│..│2044│ ← bk=0
     ├═══┼═══┼═══┼═══┼═══┼═══┼═══┼═══┼═══┼═══┼═══┼═══┤
K=8  │2048│2052│2056│2060│...│2172│2176│2180│..│2300│ ← bk=1
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
K=9  │2304│2308│2312│2316│...│2428│2432│2436│..│2556│ ← bk=1
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
K=10 │2560│2564│2568│2572│...│2684│2688│2692│..│2812│ ← bk=1
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
K=11 │2816│2820│2824│2828│...│2940│2944│2948│..│3068│ ← bk=1
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
K=12 │3072│3076│3080│3084│...│3196│3200│3204│..│3324│ ← bk=1
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
K=13 │3328│3332│3336│3340│...│3452│3456│3460│..│3580│ ← bk=1
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
K=14 │3584│3588│3592│3596│...│3708│3712│3716│..│3836│ ← bk=1
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
K=15 │3840│3844│3848│3852│...│3964│3968│3972│..│4092│ ← bk=1
     ├═══┼═══┼═══┼═══┼═══┼═══┼═══┼═══┼═══┼═══┼═══┼═══┤
K=16 │4096│4100│4104│4108│...│4220│4224│4228│..│4348│ ← bk=2
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
K=17 │4352│4356│4360│4364│...│4476│4480│4484│..│4604│ ← bk=2
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
 .   │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │  .
 .   │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │  .
 .   │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │  .
     ├═══┼═══┼═══┼═══┼═══┼═══┼═══┼═══┼═══┼═══┼═══┼═══┤
K=120│30720│30724│30728│...│30844│30848│30852│..│30972│← bk=15
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
K=121│30976│30980│30984│...│31100│31104│31108│..│31228│← bk=15
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
K=122│31232│31236│31240│...│31356│31360│31364│..│31484│← bk=15
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
K=123│31488│31492│31496│...│31612│31616│31620│..│31740│← bk=15
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
K=124│31744│31748│31752│...│31868│31872│31876│..│31996│← bk=15
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
K=125│32000│32004│32008│...│32124│32128│32132│..│32252│← bk=15
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
K=126│32256│32260│32264│...│32380│32384│32388│..│32508│← bk=15
     ├───┼───┼───┼───┼───┼───┼═══┼───┼───┼───┼───┼───┤
K=127│32512│32516│32520│...│32636│32640│32644│..│32764│← bk=15
     └───┴───┴───┴───┴───┴───┴═══┴───┴───┴───┴───┴───┘
     ←──── bx=0 block访问 ──→ ←──── bx=1 block访问 ──→
       (N: 0~127)              (N: 128~255)
线程在B矩阵中的访问模式分析：
┌──────┬──────┬─────────────┬──────────┬──────────────────────────────────────┐
│Block │ bk轮 │   线程ID    │ 访问行号  │        访问的列位置                   │
├──────┼──────┼─────────────┼──────────┼──────────────────────────────────────┤
│bx=0  │ bk=0 │ tid=0~31    │   K=0    │  0,  4,  8, 12, ..., 120, 124       │
│bx=0  │ bk=0 │ tid=32~63   │   K=1    │  0,  4,  8, 12, ..., 120, 124       │
│bx=0  │ bk=0 │ tid=64~95   │   K=2    │  0,  4,  8, 12, ..., 120, 124       │
│bx=0  │ bk=0 │ tid=96~127  │   K=3    │  0,  4,  8, 12, ..., 120, 124       │
│bx=0  │ bk=0 │ tid=128~159 │   K=4    │  0,  4,  8, 12, ..., 120, 124       │
│bx=0  │ bk=0 │ tid=160~191 │   K=5    │  0,  4,  8, 12, ..., 120, 124       │
│bx=0  │ bk=0 │ tid=192~223 │   K=6    │  0,  4,  8, 12, ..., 120, 124       │
│bx=0  │ bk=0 │ tid=224~255 │   K=7    │  0,  4,  8, 12, ..., 120, 124       │
├──────┼──────┼─────────────┼──────────┼──────────────────────────────────────┤
│bx=0  │ bk=1 │ tid=0~31    │   K=8    │  0,  4,  8, 12, ..., 120, 124       │
│bx=0  │ bk=1 │ tid=32~63   │   K=9    │  0,  4,  8, 12, ..., 120, 124       │
│  ... │ ...  │     ...     │   ...    │                 ...                  │
│bx=0  │bk=15 │ tid=224~255 │  K=127   │  0,  4,  8, 12, ..., 120, 124       │
├══════┼══════┼═════════════┼══════════┼══════════════════════════════════════┤
│bx=1  │ bk=0 │ tid=0~31    │   K=0    │128, 132, 136, 140, ..., 248, 252    │
│bx=1  │ bk=0 │ tid=32~63   │   K=1    │128, 132, 136, 140, ..., 248, 252    │
│bx=1  │ bk=0 │ tid=64~95   │   K=2    │128, 132, 136, 140, ..., 248, 252    │
│  ... │ ...  │     ...     │   ...    │                 ...                  │
│bx=1  │bk=15 │ tid=224~255 │  K=127   │128, 132, 136, 140, ..., 248, 252    │
└──────┴──────┴─────────────┴──────────┴──────────────────────────────────────┘

详细的tid到行列映射 (以bk=0为例)：
┌─────────┬──────────────┬─────────────┬──────────────┐
│   tid   │ load_b_smem_k│ load_b_smem_n│  访问的行列   │
├─────────┼──────────────┼─────────────┼──────────────┤
│   0     │      0       │      0      │   K=0, N=0   │
│   1     │      0       │      4      │   K=0, N=4   │
│   2     │      0       │      8      │   K=0, N=8   │
│  ...    │     ...      │     ...     │     ...      │
│  31     │      0       │     124     │  K=0, N=124  │
├─────────┼──────────────┼─────────────┼──────────────┤
│  32     │      1       │      0      │   K=1, N=0   │
│  33     │      1       │      4      │   K=1, N=4   │
│  ...    │     ...      │     ...     │     ...      │
│  63     │      1       │     124     │  K=1, N=124  │
├─────────┼──────────────┼─────────────┼──────────────┤
│  ...    │     ...      │     ...     │     ...      │
├─────────┼──────────────┼─────────────┼──────────────┤
│  224    │      7       │      0      │   K=7, N=0   │
│  225    │      7       │      4      │   K=7, N=4   │
│  ...    │     ...      │     ...     │     ...      │
│  255    │      7       │     124     │  K=7, N=124  │
└─────────┴──────────────┴─────────────┴──────────────┘

地址计算示例 (N=256)：
┌──────┬─────┬──────┬─────────────┬──────────────────────────────┐
│Block │ tid │  bk  │   访问位置   │        地址计算过程           │
├──────┼─────┼──────┼─────────────┼──────────────────────────────┤
│bx=0  │  0  │ bk=0 │  K=0, N=0   │ 0×256 + 0   = 0              │
│bx=0  │  1  │ bk=0 │  K=0, N=4   │ 0×256 + 4   = 4              │
│bx=0  │ 32  │ bk=0 │  K=1, N=0   │ 1×256 + 0   = 256            │
│bx=0  │ 33  │ bk=0 │  K=1, N=4   │ 1×256 + 4   = 260            │
├──────┼─────┼──────┼─────────────┼──────────────────────────────┤
│bx=0  │  0  │ bk=1 │  K=8, N=0   │ 8×256 + 0   = 2048           │
│bx=0  │  1  │ bk=1 │  K=8, N=4   │ 8×256 + 4   = 2052           │
│bx=0  │ 32  │ bk=1 │  K=9, N=0   │ 9×256 + 0   = 2304           │
├══════┼═════┼══════┼═════════════┼══════════════════════════════┤
│bx=1  │  0  │ bk=0 │ K=0, N=128  │ 0×256 + 128 = 128            │
│bx=1  │  1  │ bk=0 │ K=0, N=132  │ 0×256 + 132 = 132            │
│bx=1  │ 32  │ bk=0 │ K=1, N=128  │ 1×256 + 128 = 384            │
│bx=1  │ 33  │ bk=0 │ K=1, N=132  │ 1×256 + 132 = 388            │
└──────┴─────┴──────┴─────────────┴──────────────────────────────┘

B矩阵访问模式总结：
• 每32个连续线程访问同一行的不同列位置
• 在同一行内，线程访问的列位置以4为步长递增 
• 每8行为一个bk块，16次bk循环完成整个K维度遍历
• bx=0处理矩阵B的前128列，bx=1处理后128列
• 相邻线程访问相邻内存位置，保证内存合并访问
• 不同block在列维度上并行，无访问冲突
    */
    // 这里是向量化访存，一次性读4个half给r_load_b
    LDST64BITS(r_load_b[0]) = LDST64BITS(b[load_b_gmem_addr]);

    // s_a[8][128] write: 4路 bank conflicts
    s_a[load_a_smem_k][load_a_smem_m] = r_load_a[0];
    s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    // s_b[8][128] write: 2路 bank conflicts
    LDST64BITS(s_b[load_b_smem_k][load_b_smem_n]) = LDST64BITS(r_load_b[0]);

    __syncthreads();

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      // bank conflicts analysis, tx/ty 0~15, 0~7 bank 4*8=32 bytes
      // 进入具体线程后，可以认为该线程对应的值都已经固定了，比如tid, tx, ty.
      // 因此对于这个循环的理解，应该按照tk迭代，tid, tx, ty固定为某个值来理解.
      // 但是分析bank conflicts需要考虑warp内线程的并发行为，因此，应该分析
      // 不同线程在同一个时间点的bank访存情况.
      // s_a[8][128] load: 16路 bank conflicts
      // tid 0~15,  tk 0~7 -> ty 0 -> [0~7][0+0~7]  bank 0~3 layers_0~15
      // tid 16~31, tk 0~7 -> ty 1 -> [0~7][0+8~15] bank 4~7 layers_0~15
      LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[tk][ty * TM]);
      // s_b[8][128] load: 4路 bank conflicts
      // tid 0, tk 0~7 -> tx 0 -> [0~7][0+0~7]   bank 0~3
      // tid 1, tk 0~7 -> tx 1 -> [0~7][0+8~15]  bank 4~7
      // tid 7, tk 0~7 -> tx 7 -> [0~7][0+56~63] bank 28~31
      // tid 0/8/16/24, tk 0~7 -> tx 0/8/16/24 -> [0~7][0+...] bank 0~3
      LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[tk][tx * TN]);
      // TODO: 手工实现 swizzle之行列号异或
      // https://zhuanlan.zhihu.com/p/722286440

#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
          r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; i++) {
    int store_c_gmem_m = by * BM + ty * TM + i;
    int store_c_gmem_n = bx * BN + tx * TN;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    LDST128BITS(c[store_c_gmem_addr]) = LDST128BITS(r_c[i][0]);
  }
}

// TODO: Double Buffering support
template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf_kernel(
    half *a, half *b, half *c, const int M, const int N, const int K) {
  // threads: 128/8 * 128/8 = 256
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ half s_a[2][BK][BM + OFFSET]; // 8*128*2=2KB
  __shared__ half s_b[2][BK][BN + OFFSET]; // 8*128*2=2KB

  half r_load_a[TM / 2];                   // 4
  half r_load_b[TN / 2];                   // 4
  half r_comp_a[TM];                       // 8
  half r_comp_b[TN];                       // 8
  half r_c[TM][TN] = {__float2half(0.0f)}; // 8x8

  // mapping tid to s_a[BK][BM], for each orginal m-th row, load 4 + 4 K-dim
  // row major values from A matrix, and store it in COL major s_a[BK][BM].
  int load_a_smem_m = tid / 2; // tid / 2，(0,1,2,...,128)
  // (0b00000000 & 0b00000001) << 2 = 0
  // (0b00000001 & 0b00000001) << 2 = 4
  // (0b00000010 & 0b00000001) << 2 = 0
  // (0b00000011 & 0b00000001) << 2 = 4
  int load_a_smem_k = (tid & 1) << 2; // (0,4)
  // mapping tid to s_b[BK][BN], for each orginal k-th row, load 4 + 4 N-dim
  // row major values from B matrix, and store it in ROW major s_b[BK][BN].
  int load_b_smem_k = tid / 32; // 0~8
  // (0b00000000 & 0b00011111) << 2 = 0
  // (0b00000001 & 0b00011111) << 2 = 4
  // (0b00000010 & 0b00011111) << 2 = 8
  // (0b00000011 & 0b00011111) << 2 = 12
  int load_b_smem_n = (tid & 31) << 2; // (0,4,8,12,...,124)

  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;
  if (load_a_gmem_m >= M || load_b_gmem_n >= N)
    return;

  // 1）主循环从bk = 1
  // 开始，第一次数据加载在主循环之前，最后一次计算在主循环之后，这是pipeline
  // 的特点决定的； 2）由于计算和下一次访存使用的Shared
  // Memory不同，因此主循环中每次循环只需要一次__syncthreads()即可
  // 3）由于GPU不能向CPU那样支持乱序执行，主循环中需要先将下一次循环计算需要的Gloabal
  // Memory中的数据load
  // 到寄存器，然后进行本次计算，之后再将load到寄存器中的数据写到Shared
  // Memory，这样在LDG指令向Global
  // Memory做load时，不会影响后续FFMA及其它运算指令的 launch
  // 执行，也就达到了Double Buffering的目的。

  // bk = 0 is loading here, buffer 0
  {
    int load_a_gmem_k = load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    LDST64BITS(r_load_a[0]) = LDST64BITS(a[load_a_gmem_addr]);
    LDST64BITS(r_load_b[0]) = LDST64BITS(b[load_b_gmem_addr]);

    s_a[0][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
    s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    LDST64BITS(s_b[0][load_b_smem_k][load_b_smem_n]) = LDST64BITS(r_load_b[0]);
  }
  // Without this synchronization, accuracy may occasionally be abnormal.
  __syncthreads();

  // bk start from 1，需要注意的是，虽然 bk 从 1 开始，但实际上 bk=1时，使用的是
  // 第0块BK中的数据（已经加载到共享内存s_a[0]和s_b[0]）；bk=2时，实际计算的是第1块
  // BK中的数据。其余以此类推，这个循环结束后，剩下最后一块BK大小的数据需要计算。
  for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
    int smem_sel = (bk - 1) & 1; // bk 1->0, bk 2->1, bk 3->0, ...
    int smem_sel_next = bk & 1;  // bk 1->1, bk 2->0, bk 3->1, ...

    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    LDST64BITS(r_load_a[0]) = LDST64BITS(a[load_a_gmem_addr]);
    LDST64BITS(r_load_b[0]) = LDST64BITS(b[load_b_gmem_addr]);

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

    // 对比非double buffers版本，此处不需要__syncthreads()，总共节省了
    // ((K + BK - 1) / BK) - 1 次block内的同步操作。比如，bk=1时，HFMA计算
    // 使用的是s_a[0]和s_b[0]，因此，和s_a[1]和s_b[1]的加载是没有依赖关系的。
    // 从global内存到s_a[1]和s_b[1]和HFMA计算可以并行。s_a[1]和s_b[1]用于
    // 加载下一块BK需要的数据到共享内存。
    s_a[smem_sel_next][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
    s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    LDST64BITS(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) =
        LDST64BITS(r_load_b[0]);

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
  for (int i = 0; i < TM; i++) {
    int store_c_gmem_m = by * BM + ty * TM + i;
    int store_c_gmem_n = bx * BN + tx * TN;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    LDST128BITS(c[store_c_gmem_addr]) = LDST128BITS(r_c[i][0]);
  }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

// HGEMM naive: compute one c[i,j] element per threads, all row major
void hgemm_naive_f16(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 32;
  constexpr int BN = 32;

  dim3 block(BN, BM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  hgemm_naive_f16_kernel<<<grid, block>>>(
      reinterpret_cast<half *>(a.data_ptr()),
      reinterpret_cast<half *>(b.data_ptr()),
      reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}

void hgemm_sliced_k_f16(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;

  dim3 block(BN, BM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  hgemm_sliced_k_f16_kernel<BM, BN, BK>
      <<<grid, block>>>(reinterpret_cast<half *>(a.data_ptr()),
                        reinterpret_cast<half *>(b.data_ptr()),
                        reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}

// t 8x8 fp16x4
void hgemm_t_8x8_sliced_k_f16x4(torch::Tensor a, torch::Tensor b,
                                torch::Tensor c) {
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
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  hgemm_t_8x8_sliced_k_f16x4_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(reinterpret_cast<half *>(a.data_ptr()),
                        reinterpret_cast<half *>(b.data_ptr()),
                        reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}

// t 8x8 fp16x4 pack
void hgemm_t_8x8_sliced_k_f16x4_pack(torch::Tensor a, torch::Tensor b,
                                     torch::Tensor c) {
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
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  hgemm_t_8x8_sliced_k_f16x4_pack_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(reinterpret_cast<half *>(a.data_ptr()),
                        reinterpret_cast<half *>(b.data_ptr()),
                        reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}

// reduce bank conflicts
void hgemm_t_8x8_sliced_k_f16x4_bcf(torch::Tensor a, torch::Tensor b,
                                    torch::Tensor c) {
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
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  hgemm_t_8x8_sliced_k_f16x4_bcf_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(reinterpret_cast<half *>(a.data_ptr()),
                        reinterpret_cast<half *>(b.data_ptr()),
                        reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}

// reduce bank conflicts, f16x4 pack, t 8x8
void hgemm_t_8x8_sliced_k_f16x4_pack_bcf(torch::Tensor a, torch::Tensor b,
                                         torch::Tensor c) {
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
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  hgemm_t_8x8_sliced_k_f16x4_pack_bcf_kernel<BM, BN, BK, TM, TN, 4>
      <<<grid, block>>>(reinterpret_cast<half *>(a.data_ptr()),
                        reinterpret_cast<half *>(b.data_ptr()),
                        reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}

// reduce bank conflicts, t 8x8 fp16x8 pack, pad
void hgemm_t_8x8_sliced_k_f16x8_pack_bcf(torch::Tensor a, torch::Tensor b,
                                         torch::Tensor c) {
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
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  hgemm_t_8x8_sliced_k_f16x8_pack_bcf_kernel<BM, BN, BK, TM, TN, 8>
      <<<grid, block>>>(reinterpret_cast<half *>(a.data_ptr()),
                        reinterpret_cast<half *>(b.data_ptr()),
                        reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}

void hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf(torch::Tensor a, torch::Tensor b,
                                              torch::Tensor c) {
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
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf_kernel<BM, BN, BK, TM, TN, 8>
      <<<grid, block>>>(reinterpret_cast<half *>(a.data_ptr()),
                        reinterpret_cast<half *>(b.data_ptr()),
                        reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}
