#include <algorithm>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define WARP_SIZE 32
#define WARPGROUP_SIZE 128
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
// smem descriptor encode for wgmma (from fast.cu)
#define SMEM_DESC_ENCODE(x) ((((uint64_t)(x)) & 0x3FFFF) >> 0x4)
// wgmma fence/commit/wait
#define WGMMA_FENCE() asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory")
#define WGMMA_COMMIT_GROUP()                                                   \
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory")
#define WGMMA_WAIT_GROUP(n)                                                    \
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(n) : "memory")

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

HOST_DEVICE_INLINE
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// make wgmma shared memory matrix descriptor.
// encodes smem base addr, leading byte offset=16, stride byte offset=1024,
// and 128B swizzle mode (bit 62).
DEVICE_INLINE uint64_t make_smem_desc(half *ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  uint64_t desc = 0x0000000000000000;
  desc |= SMEM_DESC_ENCODE(addr);
  desc |= SMEM_DESC_ENCODE((uint64_t)16) << 16;
  desc |= SMEM_DESC_ENCODE((uint64_t)1024) << 32;
  desc |= 1llu << 62; // 128B swizzle
  return desc;
}

// wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
// 64 output registers (float), both A/B from shared memory (SS variant).
// ScaleD: 0=clear accum, 1=accumulate. ScaleA/B: 1=no negate.
// TransA/TransB: 0=no transpose.
#define WGMMA_M64N128K16_F32F16F16(d, sA, sB, ScaleD, ScaleA, ScaleB, TransA,  \
                                   TransB)                                     \
  {                                                                            \
    uint64_t desc_a = make_smem_desc(&(sA)[0]);                                \
    uint64_t desc_b = make_smem_desc(&(sB)[0]);                                \
    asm volatile(                                                              \
        "{\n"                                                                  \
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "                 \
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "                    \
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "                    \
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "                    \
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "                    \
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "                    \
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "                    \
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "                    \
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"                     \
        " %64,"                                                                \
        " %65,"                                                                \
        " %66, %67, %68, %69, %70;\n"                                          \
        "}\n"                                                                  \
        : "+f"((d)[0][0]), "+f"((d)[0][1]), "+f"((d)[0][2]), "+f"((d)[0][3]),  \
          "+f"((d)[0][4]), "+f"((d)[0][5]), "+f"((d)[0][6]), "+f"((d)[0][7]),  \
          "+f"((d)[1][0]), "+f"((d)[1][1]), "+f"((d)[1][2]), "+f"((d)[1][3]),  \
          "+f"((d)[1][4]), "+f"((d)[1][5]), "+f"((d)[1][6]), "+f"((d)[1][7]),  \
          "+f"((d)[2][0]), "+f"((d)[2][1]), "+f"((d)[2][2]), "+f"((d)[2][3]),  \
          "+f"((d)[2][4]), "+f"((d)[2][5]), "+f"((d)[2][6]), "+f"((d)[2][7]),  \
          "+f"((d)[3][0]), "+f"((d)[3][1]), "+f"((d)[3][2]), "+f"((d)[3][3]),  \
          "+f"((d)[3][4]), "+f"((d)[3][5]), "+f"((d)[3][6]), "+f"((d)[3][7]),  \
          "+f"((d)[4][0]), "+f"((d)[4][1]), "+f"((d)[4][2]), "+f"((d)[4][3]),  \
          "+f"((d)[4][4]), "+f"((d)[4][5]), "+f"((d)[4][6]), "+f"((d)[4][7]),  \
          "+f"((d)[5][0]), "+f"((d)[5][1]), "+f"((d)[5][2]), "+f"((d)[5][3]),  \
          "+f"((d)[5][4]), "+f"((d)[5][5]), "+f"((d)[5][6]), "+f"((d)[5][7]),  \
          "+f"((d)[6][0]), "+f"((d)[6][1]), "+f"((d)[6][2]), "+f"((d)[6][3]),  \
          "+f"((d)[6][4]), "+f"((d)[6][5]), "+f"((d)[6][6]), "+f"((d)[6][7]),  \
          "+f"((d)[7][0]), "+f"((d)[7][1]), "+f"((d)[7][2]), "+f"((d)[7][3]),  \
          "+f"((d)[7][4]), "+f"((d)[7][5]), "+f"((d)[7][6]), "+f"((d)[7][7])   \
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)),                      \
          "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)), "n"(int32_t(TransA)),    \
          "n"(int32_t(TransB)));                                               \
  }

// TMA descriptor creation (2D, matching fast.cu/matmul_4 style).
// Matrix (H, W) row-major: shape=(W, H), stride=(sizeof(half)*W).
// Box: (BlockMinorSize, BlockMajorSize). Swizzle: 128B.
template <int BlockMajorSize, int BlockMinorSize>
__host__ static inline void create_tensor_map(CUtensorMap *tma_map,
                                              half *gmem_ptr, int blocks_height,
                                              int blocks_width) {
  void *gmem_address = (void *)gmem_ptr;
  uint64_t gmem_prob_shape[5] = {(uint64_t)BlockMinorSize * blocks_width,
                                 (uint64_t)BlockMajorSize * blocks_height, 1, 1,
                                 1};
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

__host__ static inline CUtensorMap *
allocate_and_create_tensor_map(half *src, int blocks_height, int blocks_width) {
  CUtensorMap *tma_map_d;
  cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
  CUtensorMap tma_map_host;
  create_tensor_map<128, 64>(&tma_map_host, src, blocks_height, blocks_width);
  cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap),
             cudaMemcpyHostToDevice);
  return tma_map_d;
}

// shared memory layout for TMA multi-stage pipeline.
template <int BM, int BN, int BK, int QSIZE> struct WgmmaSMem {
  alignas(128) half A[BM * BK * QSIZE];
  alignas(128) half B[BK * BN * QSIZE];
};

// TN: A row major MxK, B col major NxK, C row major MxN
// 128x128, wgmma m64n128k16, warp specialized (1 producer + 1 consumer),
// stages, block swizzle, TMA, f32 accum
template <const int WGMMA_M = 64, const int WGMMA_N = 128,
          const int WGMMA_K = 16, const int BM = 128, const int BN = 128,
          const int BK = 64, const int NUM_THREADS = 256, const int K_STAGE = 3,
          const bool BLOCK_SWIZZLE = false>
__global__ void __launch_bounds__(NUM_THREADS)
    hgemm_wgmma_m64n128k16_stages_tma_ws_tn_kernel(
        int M, int N, int K, half *C,
        const CUtensorMap *__restrict__ tensorMapA,
        const CUtensorMap *__restrict__ tensorMapB) {
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  constexpr int num_consumers = (NUM_THREADS / WARPGROUP_SIZE) - 1; // 1
  constexpr int B_WG_M = BM / num_consumers;                        // 128

  if (bx >= div_ceil(N, BN) || by >= div_ceil(M, BM))
    return;

  extern __shared__ __align__(128) uint8_t smem[];
  WgmmaSMem<BM, BN, BK, K_STAGE> &s =
      *reinterpret_cast<WgmmaSMem<BM, BN, BK, K_STAGE> *>(smem);
  half *s_a = s.A;
  half *s_b = s.B;

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier full[K_STAGE], empty[K_STAGE];

  const int num_blocks_k = K / BK;
  const int wg_idx = threadIdx.x / WARPGROUP_SIZE; // 0=producer, 1=consumer
  const int tid = threadIdx.x % WARPGROUP_SIZE;    // 0~127 within WG

  // Init barriers: num_consumers*128 consumer threads + 1 producer thread.
  if (threadIdx.x == 0) {
    for (int i = 0; i < K_STAGE; ++i) {
      init(&full[i], num_consumers * WARPGROUP_SIZE + 1);
      init(&empty[i], num_consumers * WARPGROUP_SIZE + 1);
    }
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  // Producer warpgroup (WG0): TMA loads global -> shared
  if (wg_idx == 0) {
    if (tid == 0) {
      int qidx = 0;
      for (int block_k_iter = 0; block_k_iter < num_blocks_k;
           ++block_k_iter, ++qidx) {
        if (qidx == K_STAGE)
          qidx = 0;
        // wait for consumer to release this slot
        empty[qidx].wait(empty[qidx].arrive());
        // TMA load A tile: coords = (k_offset, m_offset)
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            &s_a[qidx * BK * BM], tensorMapA, block_k_iter * BK, by * BM,
            full[qidx]);
        // TMA load B tile: coords = (k_offset, n_offset)
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            &s_b[qidx * BK * BN], tensorMapB, block_k_iter * BK, bx * BN,
            full[qidx]);
        // signal expected bytes for TMA completion tracking
        [[maybe_unused]] auto token = cuda::device::barrier_arrive_tx(
            full[qidx], 1, (BK * BN + BK * BM) * sizeof(half));
      }
    }
  }
  // Consumer warpgroup (WG1): WGMMA compute
  else {
    // arrive on all empty barriers initially (consumer is ready)
    for (int i = 0; i < K_STAGE; ++i) {
      [[maybe_unused]] auto token = empty[i].arrive();
    }

    // f32 accumulators: d[m_tile][N/16][8]
    // m_tile = B_WG_M/64 = 2, N/16 = 128/16 = 8 groups, 8 floats each
    float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8];
    memset(d, 0, sizeof(d));

    int qidx = 0;
    for (int block_k_iter = 0; block_k_iter < num_blocks_k;
         ++block_k_iter, ++qidx) {
      if (qidx == K_STAGE)
        qidx = 0;

      // wait for TMA to finish loading this slot
      full[qidx].wait(full[qidx].arrive());

      // wgmma fence: ensure smem writes visible & accum regs ready
      WGMMA_FENCE();
#pragma unroll
      for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
        half *wgmma_sA = s_a + qidx * BK * BM + BK * m_it * WGMMA_M;
#pragma unroll
        for (int k_it = 0; k_it < BK / WGMMA_K; ++k_it) {
          WGMMA_M64N128K16_F32F16F16(d[m_it], wgmma_sA + k_it * WGMMA_K,
                                     s_b + qidx * BK * BN + k_it * WGMMA_K, 1,
                                     1, 1, 0, 0);
        }
      }
      WGMMA_COMMIT_GROUP();
      WGMMA_WAIT_GROUP(0);
      // release this slot for producer to reuse
      [[maybe_unused]] auto token = empty[qidx].arrive();
    }

    // Epilogue: store accumulators to row-major C.
    // wgmma m64n128k16 f32 accum register layout (per-thread):
    //   d[g][0]->(row,col)     d[g][1]->(row,col+1)
    //   d[g][2]->(row+8,col)   d[g][3]->(row+8,col+1)
    //   d[g][4]->(row,col+8)   d[g][5]->(row,col+9)
    //   d[g][6]->(row+8,col+8) d[g][7]->(row+8,col+9)
    //   row = warp*16 + lane/4,  col = 2*(lane%4) + g*16
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int row = warp * 16 + lane / 4;

    half *block_C = C + by * BM * N + bx * BN;
#pragma unroll
    for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
      int yo = m_it * WGMMA_M;
#pragma unroll
      for (int g = 0; g < WGMMA_N / 16; ++g) {
        int col = g * 16 + 2 * (lane % 4);
        // Row-major: C[(row+yo)*N + col]
#define IDX(i, j) (((i) + yo) * N + (j))
        block_C[IDX(row, col)] = __float2half(d[m_it][g][0]);
        block_C[IDX(row, col + 1)] = __float2half(d[m_it][g][1]);
        block_C[IDX(row + 8, col)] = __float2half(d[m_it][g][2]);
        block_C[IDX(row + 8, col + 1)] = __float2half(d[m_it][g][3]);
        block_C[IDX(row, col + 8)] = __float2half(d[m_it][g][4]);
        block_C[IDX(row, col + 9)] = __float2half(d[m_it][g][5]);
        block_C[IDX(row + 8, col + 8)] = __float2half(d[m_it][g][6]);
        block_C[IDX(row + 8, col + 9)] = __float2half(d[m_it][g][7]);
#undef IDX
      }
    }
  }
}

// build cpp binary
#ifndef NO_WGMMA_HGEMM_BIN

#include "utils.h"

static CUtensorMap *g_tma_a = nullptr, *g_tma_b = nullptr;
static int g_pm = 0, g_pn = 0, g_pk = 0;
static const half *g_pa = nullptr, *g_pb = nullptr;

// 128x128, wgmma m64n128k16, stages, block swizzle, TMA, warp spec, TN
#define LAUNCH_WGMMA_STAGE_SWIZZLE_TN_KERNEL(stages, stride)                   \
  {                                                                            \
    const int smem_max_size = sizeof(WgmmaSMem<BM, BN, BK, (stages)>);         \
    cudaFuncSetAttribute(hgemm_wgmma_m64n128k16_stages_tma_ws_tn_kernel<       \
                             WGMMA_M, WGMMA_N, WGMMA_K, BM, BN, BK,            \
                             NUM_THREADS, (stages), true>,                     \
                         cudaFuncAttributeMaxDynamicSharedMemorySize,          \
                         smem_max_size);                                       \
    const int N_SWIZZLE = (N + (stride) - 1) / (stride);                       \
    dim3 block(NUM_THREADS);                                                   \
    dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE, div_ceil(M, BM),  \
              N_SWIZZLE);                                                      \
    hgemm_wgmma_m64n128k16_stages_tma_ws_tn_kernel<                            \
        WGMMA_M, WGMMA_N, WGMMA_K, BM, BN, BK, NUM_THREADS, (stages), true>    \
        <<<grid, block, smem_max_size>>>(M, N, K, c, g_tma_a, g_tma_b);        \
  }

template <const int K_STAGE = 3, const int BLOCK_SWIZZLE_STRIDE = 2048>
void lanunch_hgemm_wgmma_m64n128k16_tn(half *a, half *b, half *c, int M, int N,
                                       int K) {
  constexpr int WGMMA_M = 64;
  constexpr int WGMMA_N = 128;
  constexpr int WGMMA_K = 16;
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 64;
  constexpr int NUM_THREADS = 256; // 2 warpgroups
  // s3: 3*(128*64+128*64)*2=96KB
  // s4: 4*(128*64+128*64)*2=128KB
  // s5: 5*(128*64+128*64)*2=160KB
  if (g_pm != M || g_pn != N || g_pk != K || g_pa != a || g_pb != b) {
    if (g_tma_a)
      cudaFree(g_tma_a);
    if (g_tma_b)
      cudaFree(g_tma_b);
    g_tma_a = allocate_and_create_tensor_map(a, M / BM, K / BK);
    g_tma_b = allocate_and_create_tensor_map(b, N / BN, K / BK);
    g_pm = M;
    g_pn = N;
    g_pk = K;
    g_pa = a;
    g_pb = b;
  }
  LAUNCH_WGMMA_STAGE_SWIZZLE_TN_KERNEL(K_STAGE, BLOCK_SWIZZLE_STRIDE);
}

#ifdef HGEMM_WGMMA_DEBUG
#include <iostream>
#endif

int main(int argc, char *argv[]) {
  cuInit(0);
#ifdef HGEMM_WGMMA_DEBUG
  const int test_num = 1;
#else
  const int test_num = 64;
#endif
  int M_list[test_num];
  int N_list[test_num];
  int K_list[test_num];

  for (int i = 0; i < test_num; i++) {
    M_list[i] = (i + 1) * 256;
    N_list[i] = (i + 1) * 256;
    K_list[i] = (i + 1) * 256;
  }

#ifdef HGEMM_WGMMA_DEBUG
  if (argc > 1)
    M_list[0] = std::stoi(argv[1]);
  if (argc > 2)
    N_list[0] = std::stoi(argv[2]);
  if (argc > 3)
    K_list[0] = std::stoi(argv[3]);
#endif

#ifdef HGEMM_WGMMA_DEBUG
  int outer_repeat = 1, inner_repeat = 1, warmup = 1;
  if (argc > 4)
    warmup = std::stoi(argv[4]);
  if (argc > 5)
    inner_repeat = std::stoi(argv[5]);
#else
  int outer_repeat = 10, inner_repeat = 1, warmup = 1;
#endif

  printf("ALGO = WGMMA64128 HGEMM TN STAGES=3 TMA WARP_SPEC "
         "BLOCK SWIZZLE=2048\n");
#ifndef HGEMM_WGMMA_DEBUG
  for (int j = 0; j < 5; j++) {
    int M = M_list[j], N = N_list[j], K = K_list[j];
    float max_error = gemm_error_check_tn<half>(
        lanunch_hgemm_wgmma_m64n128k16_tn<3, 2048>, M, N, K);
    printf("M N K = %6d %6d %6d, ", M, N, K);
    printf("Max Error = %f\n", max_error);
  }
#endif

  for (int j = 0; j < test_num; j++) {
    int M = M_list[j], N = N_list[j], K = K_list[j];

    double max_sec = 0.0;
    double min_sec = DBL_MAX;
    double total_sec = 0.0;

    for (int k = 0; k < outer_repeat; k++) {
      double this_sec =
          perf_gemm<half>(lanunch_hgemm_wgmma_m64n128k16_tn<3, 2048>, M, N, K,
                          inner_repeat, warmup);
      max_sec = max(max_sec, this_sec);
      min_sec = min(min_sec, this_sec);
      total_sec += this_sec;
    }

    // 1 TFLOPS = 10^12 FLOPS
    // ref: https://imgtec.eetrend.com/blog/2021/100062210.html.
    double avg_sec = total_sec / outer_repeat;
    double avg_Tflops = ((double)M) * N * K * 2 * 1e-12 / avg_sec;

    printf("M N K = %6d %6d %6d, W = %1d, R = %2d ", M, N, K, warmup,
           inner_repeat);
    printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
    printf("AVG Performance = %10.4lf Tflops\n", avg_Tflops);
  }

  return 0;
}

#else

#include <torch/extension.h>
#include <torch/types.h>
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

static CUtensorMap *g_tma_a = nullptr, *g_tma_b = nullptr;
static int g_pm = 0, g_pn = 0, g_pk = 0;
static const half *g_pa = nullptr, *g_pb = nullptr;

// 128x128, wgmma m64n128k16, stages, block swizzle, TMA, warp spec, TN
#define LAUNCH_WGMMA_STAGE_SWIZZLE_TN_KERNEL(stages, stride)                   \
  {                                                                            \
    const int smem_max_size = sizeof(WgmmaSMem<BM, BN, BK, (stages)>);         \
    cudaFuncSetAttribute(hgemm_wgmma_m64n128k16_stages_tma_ws_tn_kernel<       \
                             WGMMA_M, WGMMA_N, WGMMA_K, BM, BN, BK,            \
                             NUM_THREADS, (stages), true>,                     \
                         cudaFuncAttributeMaxDynamicSharedMemorySize,          \
                         smem_max_size);                                       \
    const int N_SWIZZLE = (N + (stride) - 1) / (stride);                       \
    dim3 block(NUM_THREADS);                                                   \
    dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE, div_ceil(M, BM),  \
              N_SWIZZLE);                                                      \
    hgemm_wgmma_m64n128k16_stages_tma_ws_tn_kernel<                            \
        WGMMA_M, WGMMA_N, WGMMA_K, BM, BN, BK, NUM_THREADS, (stages), true>    \
        <<<grid, block, smem_max_size>>>(                                      \
            M, N, K, reinterpret_cast<half *>(c.data_ptr()), g_tma_a,          \
            g_tma_b);                                                          \
  }

#define LAUNCH_WGMMA_STAGE_NO_SWIZZLE_TN_KERNEL(stages)                        \
  {                                                                            \
    const int smem_max_size = sizeof(WgmmaSMem<BM, BN, BK, (stages)>);         \
    cudaFuncSetAttribute(hgemm_wgmma_m64n128k16_stages_tma_ws_tn_kernel<       \
                             WGMMA_M, WGMMA_N, WGMMA_K, BM, BN, BK,            \
                             NUM_THREADS, (stages), false>,                    \
                         cudaFuncAttributeMaxDynamicSharedMemorySize,          \
                         smem_max_size);                                       \
    dim3 block(NUM_THREADS);                                                   \
    dim3 grid(div_ceil(N, BN), div_ceil(M, BM));                               \
    hgemm_wgmma_m64n128k16_stages_tma_ws_tn_kernel<                            \
        WGMMA_M, WGMMA_N, WGMMA_K, BM, BN, BK, NUM_THREADS, (stages), false>   \
        <<<grid, block, smem_max_size>>>(                                      \
            M, N, K, reinterpret_cast<half *>(c.data_ptr()), g_tma_a,          \
            g_tma_b);                                                          \
  }

// 128x128, wgmma m64n128k16, stages, block swizzle, TMA, warp spec, TN
void hgemm_wgmma_m64n128k16_stages_tma_ws_tn(torch::Tensor a, torch::Tensor b,
                                             torch::Tensor c, int stages,
                                             bool swizzle, int swizzle_stride) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int WGMMA_M = 64;
  constexpr int WGMMA_N = 128;
  constexpr int WGMMA_K = 16;
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 64;
  constexpr int NUM_THREADS = 256;

  static bool driver_init = false;
  if (!driver_init) {
    cuInit(0);
    driver_init = true;
  }

  if (g_pm != M || g_pn != N || g_pk != K || g_pa != a.data_ptr() ||
      g_pb != b.data_ptr()) {
    if (g_tma_a)
      cudaFree(g_tma_a);
    if (g_tma_b)
      cudaFree(g_tma_b);
    g_tma_a = allocate_and_create_tensor_map(
        reinterpret_cast<half *>(a.data_ptr()), M / BM, K / BK);
    g_tma_b = allocate_and_create_tensor_map(
        reinterpret_cast<half *>(b.data_ptr()), N / BN, K / BK);
    g_pm = M;
    g_pn = N;
    g_pk = K;
    g_pa = reinterpret_cast<half *>(a.data_ptr());
    g_pb = reinterpret_cast<half *>(b.data_ptr());
  }

  if (swizzle) {
    // assert(swizzle_stride % 256 == 0);
    switch (stages) {
    case 2:
      LAUNCH_WGMMA_STAGE_SWIZZLE_TN_KERNEL(2, swizzle_stride);
      break;
    case 3:
      LAUNCH_WGMMA_STAGE_SWIZZLE_TN_KERNEL(3, swizzle_stride);
      break;
    case 4:
      LAUNCH_WGMMA_STAGE_SWIZZLE_TN_KERNEL(4, swizzle_stride);
      break;
    case 5:
      LAUNCH_WGMMA_STAGE_SWIZZLE_TN_KERNEL(5, swizzle_stride);
      break;
    default:
      LAUNCH_WGMMA_STAGE_SWIZZLE_TN_KERNEL(3, swizzle_stride);
      break;
    }
  } else {
    switch (stages) {
    case 2:
      LAUNCH_WGMMA_STAGE_NO_SWIZZLE_TN_KERNEL(2);
      break;
    case 3:
      LAUNCH_WGMMA_STAGE_NO_SWIZZLE_TN_KERNEL(3);
      break;
    case 4:
      LAUNCH_WGMMA_STAGE_NO_SWIZZLE_TN_KERNEL(4);
      break;
    case 5:
      LAUNCH_WGMMA_STAGE_NO_SWIZZLE_TN_KERNEL(5);
      break;
    default:
      LAUNCH_WGMMA_STAGE_NO_SWIZZLE_TN_KERNEL(3);
      break;
    }
  }
}

#endif
