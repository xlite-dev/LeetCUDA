#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/extension.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
using namespace cute;

#define UNIT_BLK_SIZE 16

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      /* Optionally, you could also call cudaDeviceReset here */               \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

template <typename T, int BLK_M, int BLK_N, typename ThreadLayoutA,
          typename ThreadLayoutB>
__global__ void mat_transpose_cute_reg_kernel(const T *pA, T *pB, int M, int N,
                                              ThreadLayoutA tA,
                                              ThreadLayoutB tB) {
  int tx = threadIdx.x;
  int bx = blockIdx.x, by = blockIdx.y;

  auto mA = make_tensor(make_gmem_ptr(pA),
                        make_layout(make_shape(M, N), GenRowMajor{})); // (M, N)
  auto mB = make_tensor(make_gmem_ptr(pB),
                        make_layout(make_shape(N, M), GenRowMajor{})); // (N, M)

  auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_N>{}),
                       make_coord(bx, by)); // (BM, BN)
  auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_M>{}),
                       make_coord(by, bx)); // (BN, BM)
  auto cA = local_tile(make_identity_tensor(mA.shape()),
                       make_shape(Int<BLK_M>{}, Int<BLK_N>{}),
                       make_coord(bx, by)); // (BM, BN)

  Tensor tAgA = local_partition(gA, tA, tx);
  Tensor tBgB = local_partition(gB, tB, tx);
  Tensor tAcA = local_partition(cA, tA, tx);

  Tensor tApA = make_tensor<bool>(tAcA.shape(), tAcA.stride());
  CUTE_UNROLL
  for (int i = 0; i < size<0>(tApA); i++) {
    CUTE_UNROLL
    for (int j = 0; j < size<1>(tApA); j++) {
      tApA(i, j) = get<0>(tAcA(i, j)) < M && get<1>(tAcA(i, j)) < N;
    }
  }
  copy_if(tApA, tAgA, tBgB);
}

void mat_transpose_cute_row2col_reg(torch::Tensor x, torch::Tensor y) {
  const int BM = UNIT_BLK_SIZE;
  const int BN = UNIT_BLK_SIZE;
  const int M = x.size(0);
  const int N = x.size(1);
  auto tA = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenColMajor{});
  auto tB = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenRowMajor{});
  static_assert(size(tA) == size(tB));
  dim3 block(size(tA));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_cute_reg_kernel<float, BM, BN, decltype(tA), decltype(tB)>
      <<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), M, N, tA, tB);
  CUDA_CHECK(cudaGetLastError());
}

void mat_transpose_cute_col2row_reg(torch::Tensor x, torch::Tensor y) {
  const int BM = UNIT_BLK_SIZE;
  const int BN = UNIT_BLK_SIZE;
  const int M = x.size(0);
  const int N = x.size(1);
  auto tA = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
  auto tB = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});
  static_assert(size(tA) == size(tB));
  dim3 block(size(tA));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_cute_reg_kernel<float, BM, BN, decltype(tA), decltype(tB)>
      <<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), M, N, tA, tB);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T, int BLK_M, int BLK_N, typename ThreadLayoutA,
          typename ThreadLayoutB, typename SmemLayoutA, typename SmemLayoutB>
__global__ void
mat_transpose_cute_smem_kernel(const T *pA, T *pB, int M, int N,
                               ThreadLayoutA tA, ThreadLayoutB tB,
                               SmemLayoutA sA_layout, SmemLayoutB sB_layout) {
  int tx = threadIdx.x;
  int bx = blockIdx.x, by = blockIdx.y;

  auto mA = make_tensor(make_gmem_ptr(pA),
                        make_layout(make_shape(M, N), GenRowMajor{})); // (M, N)
  auto mB = make_tensor(make_gmem_ptr(pB),
                        make_layout(make_shape(N, M), GenRowMajor{})); // (N, M)

  auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_N>{}),
                       make_coord(bx, by)); // (BM, BN)
  auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_M>{}),
                       make_coord(by, bx)); // (BN, BM)
  auto cA = local_tile(make_identity_tensor(mA.shape()),
                       make_shape(Int<BLK_M>{}, Int<BLK_N>{}),
                       make_coord(bx, by)); // (BM, BN)
  auto cB = local_tile(make_identity_tensor(mB.shape()),
                       make_shape(Int<BLK_N>{}, Int<BLK_M>{}),
                       make_coord(by, bx)); // (BN, BM)

  __shared__ T smem[BLK_M * BLK_N];
  auto sA = make_tensor(make_smem_ptr(smem),
                        sA_layout); // (BM, BN)
  auto sB = make_tensor(make_smem_ptr(smem),
                        sB_layout); // (BN, BM)

  Tensor tAgA = local_partition(gA, tA, tx);
  Tensor tBgB = local_partition(gB, tB, tx);
  Tensor tAsA = local_partition(sA, tA, tx);
  Tensor tBsB = local_partition(sB, tB, tx);
  Tensor tAcA = local_partition(cA, tA, tx);
  Tensor tBcB = local_partition(cB, tB, tx);

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

constexpr int log2(int x) {
  assert(x > 0);
  return (x & (x - 1)) == 0 ? __builtin_ctz(x)
                            : (throw "x is not a power of 2", 0);
}

void mat_transpose_cute_col_smem(torch::Tensor x, torch::Tensor y) {
  const int BM = UNIT_BLK_SIZE;
  const int BN = UNIT_BLK_SIZE;
  const int M = x.size(0);
  const int N = x.size(1);
  auto tA = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenColMajor{});
  auto tB = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});
  auto sA_layout = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
  auto sB_layout = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});
  static_assert(size(tA) == size(tB));
  dim3 block(size(tA));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_cute_smem_kernel<float, BM, BN, decltype(tA), decltype(tB),
                                 decltype(sA_layout), decltype(sB_layout)>
      <<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), M, N, tA, tB,
                        sA_layout, sB_layout);
  CUDA_CHECK(cudaGetLastError());
}

void mat_transpose_cute_row_smem(torch::Tensor x, torch::Tensor y) {
  const int BM = UNIT_BLK_SIZE;
  const int BN = UNIT_BLK_SIZE;
  const int M = x.size(0);
  const int N = x.size(1);
  auto tA = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
  auto tB = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenRowMajor{});
  auto sA_layout = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
  auto sB_layout = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});
  static_assert(size(tA) == size(tB));
  dim3 block(size(tA));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_cute_smem_kernel<float, BM, BN, decltype(tA), decltype(tB),
                                 decltype(sA_layout), decltype(sB_layout)>
      <<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), M, N, tA, tB,
                        sA_layout, sB_layout);
  CUDA_CHECK(cudaGetLastError());
}

void mat_transpose_cute_col_smem_swizzled(torch::Tensor x, torch::Tensor y) {
  const int BM = UNIT_BLK_SIZE;
  const int BN = UNIT_BLK_SIZE;
  const int M = x.size(0);
  const int N = x.size(1);
  auto tA = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenColMajor{});
  auto tB = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});
  const int S = log2(BM);
  auto swizzle_func = Swizzle<S, 0, S>{};
  auto sA_layout =
      composition(swizzle_func,
                  make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{}));
  auto sB_layout =
      composition(swizzle_func,
                  make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{}));
  static_assert(size(tA) == size(tB));
  dim3 block(size(tA));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_cute_smem_kernel<float, BM, BN, decltype(tA), decltype(tB),
                                 decltype(sA_layout), decltype(sB_layout)>
      <<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), M, N, tA, tB,
                        sA_layout, sB_layout);
  CUDA_CHECK(cudaGetLastError());
}

void mat_transpose_cute_row_smem_swizzled(torch::Tensor x, torch::Tensor y) {
  const int BM = UNIT_BLK_SIZE;
  const int BN = UNIT_BLK_SIZE;
  const int M = x.size(0);
  const int N = x.size(1);
  auto tA = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
  auto tB = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenRowMajor{});
  const int S = log2(BM);
  auto swizzle_func = Swizzle<S, 0, S>{};
  auto sA_layout =
      composition(swizzle_func,
                  make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{}));
  auto sB_layout =
      composition(swizzle_func,
                  make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{}));
  static_assert(size(tA) == size(tB));
  dim3 block(size(tA));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_cute_smem_kernel<float, BM, BN, decltype(tA), decltype(tB),
                                 decltype(sA_layout), decltype(sB_layout)>
      <<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), M, N, tA, tB,
                        sA_layout, sB_layout);
  CUDA_CHECK(cudaGetLastError());
}

__host__ __device__ inline bool is_aligned_128(const void *ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

template <typename T, int BLK_M, int BLK_N, typename TiledCopyA,
          typename TiledCopyB, typename SmemLayoutA, typename SmemLayoutB>
__global__ void mat_transpose_cute_smem_vectorized_kernel(
    const T *pA, T *pB, int M, int N, TiledCopyA copy_a, TiledCopyB copy_b,
    SmemLayoutA sA_layout, SmemLayoutB sB_layout) {
  int tx = threadIdx.x;
  int bx = blockIdx.x, by = blockIdx.y;

  auto mA = make_tensor(make_gmem_ptr(pA),
                        make_layout(make_shape(M, N), GenRowMajor{})); // (M, N)
  auto mB = make_tensor(make_gmem_ptr(pB),
                        make_layout(make_shape(N, M), GenRowMajor{})); // (N, N)

  auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_N>{}),
                       make_coord(bx, by)); // (BM, BN)
  auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_M>{}),
                       make_coord(by, bx)); // (BN, BM)

  __shared__ T smem[BLK_M * BLK_N];
  auto sA = make_tensor(make_smem_ptr(smem),
                        sA_layout); // (BM, BN)
  auto sB = make_tensor(make_smem_ptr(smem),
                        sB_layout); // (BN, BM)

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

void mat_transpose_cute_row_cvectorized(torch::Tensor x, torch::Tensor y) {
  const int BM = UNIT_BLK_SIZE * 4;
  const int BN = UNIT_BLK_SIZE;
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
      make_layout(make_shape(Int<BM / 4>{}, Int<BN>{}), GenRowMajor{}),
      make_layout(make_shape(Int<4>{}, Int<1>{}), GenRowMajor{}));
  auto tile_copy_b = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(Int<BN>{}, Int<BM / 4>{}), GenRowMajor{}),
      make_layout(make_shape(Int<1>{}, Int<4>{}), GenRowMajor{}));
  auto sA_layout = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
  auto sB_layout = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});

  static_assert(size(tile_copy_a) == size(tile_copy_b));
  dim3 block(size(tile_copy_a));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_cute_smem_vectorized_kernel<
      float, BM, BN, decltype(tile_copy_a), decltype(tile_copy_b),
      decltype(sA_layout), decltype(sB_layout)><<<grid, block>>>(
      ptr_A, ptr_B, M, N, tile_copy_a, tile_copy_b, sA_layout, sB_layout);
  CUDA_CHECK(cudaGetLastError());
}

void mat_transpose_cute_row_cvectorized_swizzled(torch::Tensor x,
                                                 torch::Tensor y) {
  const int BM = UNIT_BLK_SIZE * 4;
  const int BN = UNIT_BLK_SIZE;
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
      make_layout(make_shape(Int<BM / 4>{}, Int<BN>{}), GenRowMajor{}),
      make_layout(make_shape(Int<4>{}, Int<1>{}), GenRowMajor{}));
  auto tile_copy_b = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(Int<BN>{}, Int<BM / 4>{}), GenRowMajor{}),
      make_layout(make_shape(Int<1>{}, Int<4>{}), GenRowMajor{}));
  const int S = log2(BN);
  auto swizzle_func = Swizzle<S, 0, S>{};
  auto sA_layout =
      composition(swizzle_func,
                  make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{}));
  auto sB_layout =
      composition(swizzle_func,
                  make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{}));

  static_assert(size(tile_copy_a) == size(tile_copy_b));
  dim3 block(size(tile_copy_a));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_cute_smem_vectorized_kernel<
      float, BM, BN, decltype(tile_copy_a), decltype(tile_copy_b),
      decltype(sA_layout), decltype(sB_layout)><<<grid, block>>>(
      ptr_A, ptr_B, M, N, tile_copy_a, tile_copy_b, sA_layout, sB_layout);
  CUDA_CHECK(cudaGetLastError());
}

void mat_transpose_cute_row_rvectorized(torch::Tensor x, torch::Tensor y) {
  const int BM = UNIT_BLK_SIZE;
  const int BN = UNIT_BLK_SIZE * 4;
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
  auto sA_layout = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
  auto sB_layout = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});

  static_assert(size(tile_copy_a) == size(tile_copy_b));
  dim3 block(size(tile_copy_a));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_cute_smem_vectorized_kernel<
      float, BM, BN, decltype(tile_copy_a), decltype(tile_copy_b),
      decltype(sA_layout), decltype(sB_layout)><<<grid, block>>>(
      ptr_A, ptr_B, M, N, tile_copy_a, tile_copy_b, sA_layout, sB_layout);
  CUDA_CHECK(cudaGetLastError());
}

void mat_transpose_cute_row_rvectorized_swizzled(torch::Tensor x,
                                                 torch::Tensor y) {
  const int BM = UNIT_BLK_SIZE;
  const int BN = UNIT_BLK_SIZE * 4;
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
  const int S = log2(BM);
  auto swizzle_func = Swizzle<S, 0, S>{};
  auto sA_layout =
      composition(swizzle_func,
                  make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{}));
  auto sB_layout =
      composition(swizzle_func,
                  make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{}));

  static_assert(size(tile_copy_a) == size(tile_copy_b));
  dim3 block(size(tile_copy_a));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_cute_smem_vectorized_kernel<
      float, BM, BN, decltype(tile_copy_a), decltype(tile_copy_b),
      decltype(sA_layout), decltype(sB_layout)><<<grid, block>>>(
      ptr_A, ptr_B, M, N, tile_copy_a, tile_copy_b, sA_layout, sB_layout);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T, int BLK_M, int BLK_N, typename TiledCopyA,
          typename TiledCopyTrans, typename TiledCopyB, typename SmemLayoutB>
__global__ void mat_transpose_cute_smem_vectorized_optimized_kernel(
    const T *pA, T *pB, int M, int N, TiledCopyA copy_a,
    TiledCopyTrans copy_trans, TiledCopyB copy_b, SmemLayoutB sB_layout) {
  int tx = threadIdx.x;
  int bx = blockIdx.x, by = blockIdx.y;

  auto mA = make_tensor(make_gmem_ptr(pA),
                        make_layout(make_shape(M, N), GenRowMajor{})); // (M, N)
  auto mB = make_tensor(make_gmem_ptr(pB),
                        make_layout(make_shape(N, M), GenRowMajor{})); // (N, N)

  auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_N>{}),
                       make_coord(bx, by)); // (BM, BN)
  auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_M>{}),
                       make_coord(by, bx)); // (BN, BM)

  __shared__ T smem[BLK_M * BLK_N];
  auto sB = make_tensor(make_smem_ptr(smem),
                        sB_layout); // (BN, BM)

  auto thr_copy_a = copy_a.get_slice(tx);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  auto tAsA = make_tensor_like(tAgA);
  Tensor tAsA_view = thr_copy_a.retile_D(tAsA);
  copy(copy_a, tAgA, tAsA_view);

  auto thr_copy_trans = copy_trans.get_slice(tx);
  auto tAsB = thr_copy_trans.retile_S(tAsA);
  auto tBsB_trans = thr_copy_trans.partition_D(sB);
  copy(copy_trans, tAsB, tBsB_trans);

  auto thr_copy_b = copy_b.get_slice(tx);
  Tensor tBsB = thr_copy_b.partition_S(sB);
  Tensor tBgB = thr_copy_b.partition_D(gB);

  copy(copy_b, tBsB, tBgB);
}

void mat_transpose_cute_row_rvectorized_swizzled_optimized(torch::Tensor x,
                                                           torch::Tensor y) {
  const int BM = 8;
  const int BN = 16 * 8;
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

  // 一次性加载8*16大小的矩阵
  auto tile_copy_a = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(Int<BM>{}, make_shape(Int<4>{}, Int<BN / 16>{})),
                  make_stride(Int<4>{}, make_stride(Int<1>{}, Int<32>{}))),
      make_layout(make_shape(Int<1>{}, Int<4>{}), GenRowMajor{}));

  // 转换数据
  auto tile_copy_trans = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(make_shape(Int<4>{}, Int<BN / 16>{}), Int<BM>{}),
                  make_stride(make_stride(Int<1>{}, Int<32>{}), Int<4>{})),
      make_layout(make_shape(Int<4>{}, Int<1>{}), GenRowMajor{}));

  // 一次性存储16*8大小的矩阵
  auto tile_copy_b = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(Int<BN>{}, Int<BM / 4>{}), GenRowMajor{}),
      make_layout(make_shape(Int<1>{}, Int<4>{}), GenRowMajor{}));

  auto swizzle_func = Swizzle<2, 3, 2>{};
  auto sB_layout =
      composition(swizzle_func,
                  make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenRowMajor{}));

  static_assert(size(tile_copy_a) == size(tile_copy_b));

  dim3 block(size(tile_copy_a));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

  mat_transpose_cute_smem_vectorized_optimized_kernel<
      float, BM, BN, decltype(tile_copy_a), decltype(tile_copy_trans),
      decltype(tile_copy_b), decltype(sB_layout)><<<grid, block>>>(
      ptr_A, ptr_B, M, N, tile_copy_a, tile_copy_trans, tile_copy_b, sB_layout);
  CUDA_CHECK(cudaGetLastError());
}
