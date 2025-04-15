#include <torch/types.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,       \
                    __LINE__, cudaGetErrorString(err));                  \
            /* Optionally, you could also call cudaDeviceReset here */   \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

using namespace cute;

template <typename T, typename ThreadLayout, int BLK_M, int BLK_N>
__global__ void mat_transpose_row2col(const T *pA, T *pB, int M, int N,
                                      ThreadLayout tA) {
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

  Tensor tAgA = local_partition(gA, tA, tx);
  Tensor tAgB = local_partition(gB, tA, tx);
  Tensor tAsA = local_partition(sA, tA, tx);
  Tensor tAsB = local_partition(sB, tA, tx);

  copy(tAgA, tAsA);
  __syncthreads();
  copy(tAsB, tAgB);
}

void mat_transpose_cute(torch::Tensor x, torch::Tensor y) {
  const int BM = 16;
  const int BN = 16;
  const int M = x.size(0);
  const int N = x.size(1);
  auto tA = make_layout(make_shape(Int<BM>{}, Int<BN>{}));
  dim3 block(size(tA));
  dim3 grid(M / BM, N / BN);
  mat_transpose_row2col<float, decltype(tA), BM, BN>
      <<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), M, N, tA);
  CUDA_CHECK(cudaGetLastError());
}