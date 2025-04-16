#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/types.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#define CUDA_CHECK(call)                                               \
  do {                                                                 \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                \
      /* Optionally, you could also call cudaDeviceReset here */       \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

using namespace cute;

template <typename T, int BLK_M, int BLK_N, typename ThreadLayout>
__global__ void mat_transpose_cute_row2col_naive_kernel(const T *pA, T *pB,
                                                        int M, int N,
                                                        ThreadLayout tAB) {
  int tx = threadIdx.x;
  int bx = blockIdx.x, by = blockIdx.y;

  auto mA =
      make_tensor(make_gmem_ptr(pA),
                  make_layout(make_shape(M, N), GenRowMajor{}));  // (M, N)
  auto mB =
      make_tensor(make_gmem_ptr(pB),
                  make_layout(make_shape(N, M), GenRowMajor{}));  // (N, M)

  auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_N>{}),
                       make_coord(bx, by));  // (BM, BN)
  auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_M>{}),
                       make_coord(by, bx));  // (BN, BM)
  auto cA = local_tile(make_identity_tensor(mA.shape()),
                       make_shape(Int<BLK_M>{}, Int<BLK_N>{}),
                       make_coord(bx, by));  // (BM, BN)
  auto cB = local_tile(make_identity_tensor(mB.shape()),
                       make_shape(Int<BLK_N>{}, Int<BLK_M>{}),
                       make_coord(by, bx));  // (BN, BM)

  __shared__ T smem[BLK_M * BLK_N];
  auto sA = make_tensor(
      make_smem_ptr(smem),
      make_layout(make_shape(BLK_M, BLK_N), GenRowMajor{}));  // (BM, BN)
  auto sB = make_tensor(
      make_smem_ptr(smem),
      make_layout(make_shape(BLK_N, BLK_M), GenColMajor{}));  // (BN, BM)

  Tensor tAgA = local_partition(gA, tAB, tx);
  Tensor tBgB = local_partition(gB, tAB, tx);
  Tensor tAsA = local_partition(sA, tAB, tx);
  Tensor tBsB = local_partition(sB, tAB, tx);
  Tensor tAcA = local_partition(cA, tAB, tx);
  Tensor tBcB = local_partition(cB, tAB, tx);

  Tensor tApA = make_tensor<bool>(tAcA.shape(), tAcA.stride());
  Tensor tBpB = make_tensor<bool>(tBcB.shape(), tBcB.stride());
  CUTE_UNROLL
  for (int i = 0; i < size<0>(tApA); i++) {
    CUTE_UNROLL
    for (int j = 0; j < size<1>(tApA); j++) {
      tApA(i, j) = get<0>(tAcA(i, j)) < M && get<1>(tAcA(i, j)) < N;
    }
  }
  CUTE_UNROLL
  for (int i = 0; i < size<0>(tBpB); i++) {
    CUTE_UNROLL
    for (int j = 0; j < size<1>(tBpB); j++) {
      tBpB(i, j) = get<0>(tBcB(i, j)) < N && get<1>(tBcB(i, j)) < M;
    }
  }
  copy_if(tApA, tAgA, tAsA);
  __syncthreads();
  copy_if(tBpB, tBsB, tBgB);
}

void mat_transpose_cute_row2col_naive(torch::Tensor x, torch::Tensor y) {
  const int BM = 16;
  const int BN = 16;
  const int M = x.size(0);
  const int N = x.size(1);
  auto tAB = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
  dim3 block(size(tAB));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_cute_row2col_naive_kernel<float, BM, BN, decltype(tAB)>
      <<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), M, N, tAB);
  CUDA_CHECK(cudaGetLastError());
}

__host__ __device__ inline bool is_aligned_128(const void *ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

template <typename T, int BLK_M, int BLK_N, typename TiledCopyA,
          typename TiledCopyB>
__global__ void mat_transpose_cute_row2col_vectorized_kernel(
    const T *pA, T *pB, int M, int N, TiledCopyA copy_a, TiledCopyB copy_b) {
  int tx = threadIdx.x;
  int bx = blockIdx.x, by = blockIdx.y;

  auto mA =
      make_tensor(make_gmem_ptr(pA),
                  make_layout(make_shape(M, N), GenRowMajor{}));  // (M, N)
  auto mB =
      make_tensor(make_gmem_ptr(pB),
                  make_layout(make_shape(N, M), GenRowMajor{}));  // (N, N)

  auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_N>{}),
                       make_coord(bx, by));  // (BM, BN)
  auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_M>{}),
                       make_coord(by, bx));  // (BN, BM)

  __shared__ T smem[BLK_M * BLK_N];
  auto sA = make_tensor(
      make_smem_ptr(smem),
      make_layout(make_shape(BLK_M, BLK_N), GenRowMajor{}));  // (BM, BN)
  auto sB = make_tensor(
      make_smem_ptr(smem),
      make_layout(make_shape(BLK_N, BLK_M), GenColMajor{}));  // (BN, BM)

  auto thr_copy_a = copy_a.get_slice(tx);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tAsA = thr_copy_a.partition_D(sA);

  auto thr_copy_b = copy_b.get_slice(tx);
  Tensor tBsB = thr_copy_b.partition_S(sB);
  Tensor tBgB = thr_copy_b.partition_D(gB);

  copy(copy_a, tAgA, tAsA);
  __syncthreads();
  copy(copy_b, tBsB, tBgB);
}

void mat_transpose_cute_row2col_vectorized(torch::Tensor x, torch::Tensor y) {
  const int BM = 16;
  const int BN = 64;
  auto ptr_A = x.data_ptr<float>();
  auto ptr_B = y.data_ptr<float>();
  const int M = x.size(0);
  const int N = x.size(1);

  // sanity checks
  assert(M % 4 == 0);
  assert(N % 4 == 0);
  static_assert(BM % 4 == 0);
  static_assert(BN % 4 == 0);
  assert(is_aligned_128(ptr_A));
  assert(is_aligned_128(ptr_B));

  auto tile_copy_a = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(Int<BM>{}, Int<BN / 4>{}), GenRowMajor{}),
      make_layout(make_shape(Int<1>{}, Int<4>{}), GenRowMajor{}));
  auto tile_copy_b = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(Int<BN / 4>{}, Int<BM>{}), GenRowMajor{}),
      make_layout(make_shape(Int<4>{}, Int<1>{}), GenRowMajor{}));

  static_assert(size(tile_copy_a) == size(tile_copy_b));
  dim3 block(size(tile_copy_a));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_cute_row2col_vectorized_kernel<
      float, BM, BN, decltype(tile_copy_a), decltype(tile_copy_b)>
      <<<grid, block>>>(ptr_A, ptr_B, M, N, tile_copy_a, tile_copy_b);
  CUDA_CHECK(cudaGetLastError());
}