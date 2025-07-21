#pragma once
#include "cooperative_groups.h"
#include "cuda/pipeline"
#include "cute/tensor.hpp"
#include <cooperative_groups/memcpy_async.h>
#include <torch/extension.h>
#include <torch/types.h>

#define DEVICE __device__ __forceinline__

using namespace cute;

template <class CTATile, int ProducerThread, int Stage> struct WSHGEMMTraits {
  using MatrixTypeAB = half;
  using AccType = half;

  constexpr static int kCTAM = get<0>(CTATile{});
  constexpr static int kCTAN = get<1>(CTATile{});
  constexpr static int kCTAK = get<2>(CTATile{});
  constexpr static int kStage = Stage;

  // MMA Trait
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;
  using mma_atom_shape = mma_traits::Shape_MNK;

  constexpr static int kMmaThrLayoutM = 2;
  constexpr static int kMmaThrLayoutN = 2;
  constexpr static int kMmaThrLayoutK = 1;

  constexpr static int kMmaPermuteM = kMmaThrLayoutM * get<0>(mma_atom_shape{});
  constexpr static int kMmaPermuteN =
      2 * kMmaThrLayoutN * get<1>(mma_atom_shape{});
  constexpr static int kMmaPermuteK = kMmaThrLayoutK * get<2>(mma_atom_shape{});

  using MmaThrLayout = decltype(make_layout(make_shape(
      Int<kMmaThrLayoutM>{}, Int<kMmaThrLayoutN>{}, Int<kMmaThrLayoutK>{})));

  using MmaPermutation = decltype(make_tile(
      Int<kMmaPermuteM>{}, Int<kMmaPermuteN>{}, Int<kMmaPermuteK>{}));

  // The expanded TiledMMA can process matrices of size 32x32x16 in a single
  // operation.
  using TiledMMA =
      decltype(make_tiled_mma(mma_atom{}, MmaThrLayout{}, MmaPermutation{}));

  constexpr static int kConsumerThread = size(TiledMMA{});
  // To avoid warp divergence
  static_assert(ProducerThread % 32 == 0,
                "The number of ProducerThreads must be a multiple of 32");
  constexpr static int kProducerThread = ProducerThread;
  constexpr static int kAllThread = kProducerThread + kConsumerThread;

  // Smem
  constexpr static int kSwizzleB = 3;
  constexpr static int kSwizzleM = 3;
  constexpr static int kSwizzleS = 3;

  using SmemLayoutAtom =
      decltype(composition(Swizzle<kSwizzleB, kSwizzleM, kSwizzleS>{},
                           make_layout(make_shape(Int<8>{}, Int<kCTAK>{}),
                                       make_stride(Int<kCTAK>{}, Int<1>{}))));

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<kCTAM>{}, Int<kCTAK>{}, Int<kStage>{})));

  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<kCTAN>{}, Int<kCTAK>{}, Int<kStage>{})));

  constexpr static int kSmemSizeA = cosize(SmemLayoutA{});
  constexpr static int kSmemSizeB = cosize(SmemLayoutB{});
  constexpr static int kSmemAllocateAB =
      (kSmemSizeA + kSmemSizeB) * sizeof(MatrixTypeAB);

  constexpr static int kSmemStageAcc = 2;
  using SmemLayoutAcc = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<kMmaPermuteM>{}, Int<kMmaPermuteN>{},
                                   Int<kSmemStageAcc>{})));

  constexpr static int kSmemSizeAcc = cosize(SmemLayoutAcc{});
  constexpr static int kSmemAllocateAcc = kSmemSizeAcc * sizeof(AccType);
  constexpr static int kAllSmemAllocate =
      cute::max(kSmemAllocateAB, kSmemAllocateAcc);

  // Producer g2s copy
  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, MatrixTypeAB>;

  constexpr static int g2s_thread_vec_size =
      sizeof(cute::uint128_t) / sizeof(MatrixTypeAB);
  constexpr static int g2s_thread_k = kCTAK / g2s_thread_vec_size;
  constexpr static int g2s_thread_m = kProducerThread / g2s_thread_k;

  using G2SCopyA = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<g2s_thread_m>{}, Int<g2s_thread_k>{}),
                  make_stride(Int<g2s_thread_k>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<g2s_thread_vec_size>{}))));

  using G2SCopyB = G2SCopyA;

  // Consumer s2r copy
  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, MatrixTypeAB>;

  using S2RCopyA = s2r_copy_atom;
  using S2RCopyB = s2r_copy_atom;

  // Consumer r2s copy
  using R2SCopyC = Copy_Atom<UniversalCopy<int>, AccType>;

  // Consumer s2g copy
  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, AccType>;
  constexpr static int s2g_thread_vec_size =
      sizeof(cute::uint128_t) / sizeof(AccType);
  constexpr static int s2g_thread_n = kMmaPermuteN / s2g_thread_vec_size;
  constexpr static int s2g_thread_m = kConsumerThread / s2g_thread_n;

  using S2GCopyC = decltype(make_tiled_copy(
      S2GCopyAtomC{},
      make_layout(make_shape(Int<s2g_thread_m>{}, Int<s2g_thread_n>{}),
                  make_stride(Int<s2g_thread_n>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<s2g_thread_vec_size>{}))));

  struct Arguments {
    void *a_ptr;
    void *b_ptr;
    void *c_ptr;

    using Gemm_Shape = Shape<int, int, int>;

    Gemm_Shape problem_shape;
    Gemm_Shape tile_shape;

    Arguments() = delete;
    Arguments(Gemm_Shape problem_shape_, void *a_ptr_, void *b_ptr_,
              void *c_ptr_)
        : problem_shape(problem_shape_), a_ptr(a_ptr_), b_ptr(b_ptr_),
          c_ptr(c_ptr_) {
      int m = size<0>(problem_shape);
      int n = size<1>(problem_shape);
      int k = size<2>(problem_shape);

      int m_tiles = ceil_div(m, kCTAM);
      int n_tiles = ceil_div(n, kCTAN);
      int k_tiles = ceil_div(k, kCTAK);
      tile_shape = make_shape(m_tiles, n_tiles, k_tiles);
    }

    dim3 get_grid() {
      return dim3(ceil_div(get<0>(problem_shape), kCTAM),
                  ceil_div(get<1>(problem_shape), kCTAN));
    }
  };

  template <typename Pipeline, typename AEngine, typename ALayout,
            typename BEngine, typename BLayout>
  DEVICE static auto producer(void *smem_ptr, Pipeline &pipeline,
                              Tensor<AEngine, ALayout> const &gA,
                              Tensor<BEngine, BLayout> const &gB) {
    using T = typename WSHGEMMTraits::MatrixTypeAB;
    constexpr int SmemSizeA = WSHGEMMTraits::kSmemSizeA;
    auto tidx = threadIdx.x;

    T *SmemPtrA = reinterpret_cast<T *>(smem_ptr);
    T *SmemPtrB = SmemPtrA + SmemSizeA;

    auto sA = make_tensor(make_smem_ptr<T>(SmemPtrA),
                          SmemLayoutA{}); // (CTAM, CTAK, Stage)
    auto sB = make_tensor(make_smem_ptr<T>(SmemPtrB),
                          SmemLayoutB{}); // (CTAN, CTAK, Stage)

    G2SCopyA g2s_copy_A;
    auto thr_g2s_copy_A = g2s_copy_A.get_slice(tidx);
    auto g2s_tAgA = thr_g2s_copy_A.partition_S(gA);
    auto g2s_tAsA = thr_g2s_copy_A.partition_D(sA);

    G2SCopyB g2s_copy_B;
    auto thr_g2s_copy_B = g2s_copy_B.get_slice(tidx);
    auto g2s_tBgB = thr_g2s_copy_B.partition_S(gB);
    auto g2s_tBsB = thr_g2s_copy_B.partition_D(sB);

    const auto kNumIterationK = size<2>(gA);
    int g2s_g_read_idx = 0;
    int g2s_s_write_idx = 0;

    for (int iter_k = 0, stage_idx = 0; iter_k < kNumIterationK; iter_k++) {
      for (; stage_idx < kNumIterationK && stage_idx < (iter_k + kStage);
           stage_idx++) {

        pipeline.producer_acquire();

        copy(g2s_copy_A, g2s_tAgA(_, _, _, g2s_g_read_idx),
             g2s_tAsA(_, _, _, g2s_s_write_idx));

        copy(g2s_copy_B, g2s_tBgB(_, _, _, g2s_g_read_idx),
             g2s_tBsB(_, _, _, g2s_s_write_idx));

        pipeline.producer_commit();

        g2s_g_read_idx++;
        g2s_s_write_idx = (g2s_s_write_idx + 1) % kStage;
      }
    }
  }

  template <typename Pipeline, typename CEngine, typename CLayout>
  DEVICE static auto main_loop(Arguments const &args, void *smem_ptr,
                               Pipeline &pipeline,
                               Tensor<CEngine, CLayout> const &gC) {
    using T = typename WSHGEMMTraits::MatrixTypeAB;
    constexpr int SmemSizeA = WSHGEMMTraits::kSmemSizeA;
    auto tidx = threadIdx.x - kProducerThread;

    T *SmemPtrA = reinterpret_cast<T *>(smem_ptr);
    T *SmemPtrB = SmemPtrA + SmemSizeA;

    auto sA = make_tensor(make_smem_ptr<T>(SmemPtrA),
                          SmemLayoutA{}); // (CTAM, CTAK, Stage)
    auto sB = make_tensor(make_smem_ptr<T>(SmemPtrB),
                          SmemLayoutB{}); // (CTAN, CTAK, Stage)

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tidx);

    auto tArA = thr_mma.partition_fragment_A(sA(_, _, 0));
    auto tBrB = thr_mma.partition_fragment_B(sB(_, _, 0));
    auto tCrC = thr_mma.partition_fragment_C(gC);
    clear(tCrC);

    // s2r
    auto s2r_copy_A = make_tiled_copy_A(S2RCopyA{}, tiled_mma);
    auto thr_s2r_copy_A = s2r_copy_A.get_slice(tidx);
    auto s2r_tAsA = thr_s2r_copy_A.partition_S(sA);
    auto s2r_tArA_view = thr_s2r_copy_A.retile_D(tArA);

    auto s2r_copy_B = make_tiled_copy_B(S2RCopyB{}, tiled_mma);
    auto thr_s2r_copy_B = s2r_copy_B.get_slice(tidx);
    auto s2r_tBsB = thr_s2r_copy_B.partition_S(sB);
    auto s2r_tBrB_view = thr_s2r_copy_B.retile_D(tBrB);

    const int kNumIterationK = get<2>(args.tile_shape);
    const int kNumInnerStage = size<2>(tArA);

    int s2r_s_read_idx = 0;
    int next_s2r_s_read_idx = 0;

    if (kNumInnerStage > 1) {
      pipeline.consumer_wait();

      copy(s2r_copy_A, s2r_tAsA(_, _, 0, s2r_s_read_idx),
           s2r_tArA_view(_, _, 0));
      copy(s2r_copy_B, s2r_tBsB(_, _, 0, s2r_s_read_idx),
           s2r_tBrB_view(_, _, 0));
    }

    for (int iter_k = 0; iter_k < kNumIterationK; iter_k++) {
#pragma unroll
      for (int inner_stage = 0; inner_stage < kNumInnerStage; inner_stage++) {
        int next_inner_stage = (inner_stage + 1) % kNumInnerStage;

        if (inner_stage == kNumInnerStage - 1 && iter_k < kNumIterationK - 1) {
          pipeline.consumer_wait();
          s2r_s_read_idx = next_s2r_s_read_idx;
        }

        copy(s2r_copy_A, s2r_tAsA(_, _, next_inner_stage, s2r_s_read_idx),
             s2r_tArA_view(_, _, next_inner_stage));

        copy(s2r_copy_B, s2r_tBsB(_, _, next_inner_stage, s2r_s_read_idx),
             s2r_tBrB_view(_, _, next_inner_stage));

        if (inner_stage == 0) {
          pipeline.consumer_release();
          next_s2r_s_read_idx = (s2r_s_read_idx + 1) % kStage;
        }

        gemm(tiled_mma, tArA(_, _, inner_stage), tBrB(_, _, inner_stage), tCrC);
      }
    }

    return tCrC;
  }

  template <typename AccEngine, typename AccLayout, typename CEngine,
            typename CLayout>
  DEVICE static void epilog(Arguments const &args, void *smem_ptr,
                            Tensor<AccEngine, AccLayout> const &acc,
                            Tensor<CEngine, CLayout> &gC) {
    __syncthreads(); // wait all consumer thread finish main_loop

    int tidx = threadIdx.x - kProducerThread;

    auto sC = make_tensor(
        make_smem_ptr<AccType>(smem_ptr),
        SmemLayoutAcc{}); // (kMmaPermuteM, kMmaPermuteN, kSmemStageAcc)

    // r2s
    auto r2s_copy_C = make_tiled_copy_C(R2SCopyC{}, TiledMMA{});
    auto thr_r2s_copy_C = r2s_copy_C.get_slice(tidx);
    auto r2s_tCrC = thr_r2s_copy_C.retile_S(acc);
    auto r2s_tCsC = thr_r2s_copy_C.partition_D(sC);

    auto r2s_tCrC_view = group_modes<1, 3>(r2s_tCrC);

    // s2g
    S2GCopyC s2g_copy_C;
    auto thr_s2g_copy_C = s2g_copy_C.get_slice(tidx);
    auto s2g_tCsC = thr_s2g_copy_C.partition_S(sC);
    auto s2g_tCgC = thr_s2g_copy_C.partition_D(gC);

    auto s2g_tCgC_view = group_modes<1, 3>(s2g_tCgC);

    const int kEpilogIterations = size<1>(r2s_tCrC_view);
    const int kEpilogStages = size<3>(r2s_tCsC);

#pragma unroll
    for (int epilog_iter = 0; epilog_iter < kEpilogIterations;
         epilog_iter += kEpilogStages) {
#pragma unroll
      for (int epilog_stage_idx = 0; epilog_stage_idx < kEpilogStages;
           epilog_stage_idx++) {
        // r2s
        copy(r2s_copy_C, r2s_tCrC_view(_, epilog_iter + epilog_stage_idx),
             r2s_tCsC(_, 0, 0, epilog_stage_idx));
      }
      __syncthreads(); // wait all consumer thread finish r2s

#pragma unroll
      for (int epilog_stage_idx = 0; epilog_stage_idx < kEpilogStages;
           epilog_stage_idx++) {
        // s2g
        copy(s2g_copy_C, s2g_tCsC(_, 0, 0, epilog_stage_idx),
             s2g_tCgC_view(_, epilog_iter + epilog_stage_idx));
      }
      __syncthreads(); // wait all consumer thread finish s2g
    }
  }

  template <typename Pipeline, typename CEngine, typename CLayout>
  DEVICE static void consumer(Arguments const &args, void *smem_ptr,
                              Pipeline &pipeline,
                              Tensor<CEngine, CLayout> &gC) {
    auto tCrC = main_loop(args, smem_ptr, pipeline, gC);
    epilog(args, smem_ptr, tCrC, gC);
  }
};

#pragma nv_diag_suppress static_var_with_dynamic_init
template <typename WSHGEMMTraits>
__global__ void
ws_hgemm_naive_cute_kernel(typename WSHGEMMTraits::Arguments args) {
  using MatrixTypeAB = typename WSHGEMMTraits::MatrixTypeAB;
  using AccType = typename WSHGEMMTraits::AccType;

  constexpr int kCTAM = WSHGEMMTraits::kCTAM;
  constexpr int kCTAN = WSHGEMMTraits::kCTAN;
  constexpr int kCTAK = WSHGEMMTraits::kCTAK;
  constexpr int kStage = WSHGEMMTraits::kStage;

  auto block = cooperative_groups::this_thread_block();
  auto tidx = threadIdx.x;

  auto tile_id_m = blockIdx.x;
  auto tile_id_n = blockIdx.y;

  // set thread role
  const auto thread_role = tidx < WSHGEMMTraits::kProducerThread
                               ? cuda::pipeline_role::producer
                               : cuda::pipeline_role::consumer;
  // create pipeline
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block,
                                         kStage>
      shared_state;
  auto pipeline = cuda::make_pipeline(block, &shared_state, thread_role);

  extern __shared__ MatrixTypeAB smem_ptr[];

  auto A =
      make_tensor(make_gmem_ptr<MatrixTypeAB>(args.a_ptr),
                  select<0, 2>(args.problem_shape), GenRowMajor{}); // (M, K)

  auto B =
      make_tensor(make_gmem_ptr<MatrixTypeAB>(args.b_ptr),
                  select<1, 2>(args.problem_shape), GenRowMajor{}); // (N, K)

  auto C =
      make_tensor(make_gmem_ptr<AccType>(args.c_ptr),
                  select<0, 1>(args.problem_shape), GenRowMajor{}); // (M, N)

  auto gA = local_tile(A, make_tile(Int<kCTAM>{}, Int<kCTAK>{}),
                       make_coord(tile_id_m, _)); // (kCTAM, kCTAK, K/kCTAK)

  auto gB = local_tile(B, make_tile(Int<kCTAN>{}, Int<kCTAK>{}),
                       make_coord(tile_id_n, _)); // (kCTAN, kCTAK, K/kCTAK)

  auto gC = local_tile(C, make_tile(Int<kCTAM>{}, Int<kCTAN>{}),
                       make_coord(tile_id_m, tile_id_n)); // (kCTAM, kCTAN)

  // Different thread_roles execute different branches.
  if (thread_role == cuda::pipeline_role::producer) {
    WSHGEMMTraits::producer(smem_ptr, pipeline, gA, gB);
  } else {
    WSHGEMMTraits::consumer(args, smem_ptr, pipeline, gC);
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

inline int get_max_smem_size() {
  int max_shared_mem;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  return max_shared_mem;
}

template <typename Kernel> void config_smem(Kernel kernel, int smem_size) {
  if (smem_size >= 32 * 1024) {
    if (cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size) != cudaSuccess) {
      int max_shared_mem = get_max_smem_size();
      cudaError_t err = cudaGetLastError();
      std::cerr << "Set kernel attribute failed: " << cudaGetErrorString(err)
                << std::endl;
      std::cerr
          << "Kernel required " << smem_size
          << " shared memory but the max shared memory per block optin is: "
          << max_shared_mem << std::endl;
    }
  }
}

// WarpSpecialization HGEMM
void ws_hgemm_naive_cute(torch::Tensor a, torch::Tensor b, torch::Tensor c) {

  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  using GEMM_Traits =
      WSHGEMMTraits<decltype(make_shape(_128{}, _256{}, _32{})), 32, 3>;

  // set smem size
  constexpr int smem_size = GEMM_Traits::kAllSmemAllocate;
  config_smem(ws_hgemm_naive_cute_kernel<GEMM_Traits>, smem_size);

  // set problem size
  using Arguments = typename GEMM_Traits::Arguments;
  Arguments args(make_shape(M, N, K), reinterpret_cast<half *>(a.data_ptr()),
                 reinterpret_cast<half *>(b.data_ptr()),
                 reinterpret_cast<half *>(c.data_ptr()));

  constexpr int block_size = GEMM_Traits::kAllThread;

  dim3 block(block_size);
  dim3 grid(args.get_grid());

  ws_hgemm_naive_cute_kernel<GEMM_Traits><<<grid, block, smem_size>>>(args);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(ws_hgemm_naive_cute)
}
