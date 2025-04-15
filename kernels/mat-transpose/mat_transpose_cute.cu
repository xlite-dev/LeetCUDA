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

template <typename T, typename ThreadLayout, int BLK_M, int BLK_N>
__global__ void mat_transpose_row2col(const T *pA, T *pB, int M, int N,
                                      ThreadLayout tAB) {
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

void mat_transpose_cute(torch::Tensor x, torch::Tensor y) {
  const int BM = 16;
  const int BN = 16;
  const int M = x.size(0);
  const int N = x.size(1);
  auto tAB = make_layout(make_shape(Int<BM>{}, Int<BN>{}));
  dim3 block(size(tAB));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_row2col<float, decltype(tAB), BM, BN>
      <<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), M, N, tAB);
  CUDA_CHECK(cudaGetLastError());
}