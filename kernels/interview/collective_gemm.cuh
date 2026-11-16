#pragma once
// collective_gemm.cuh — 常规 BF16 GEMM 的 CUTLASS CollectiveBuilder 最小示例。
// 目标架构：SM90。A[M,K] 为 row-major，B[N,K] 为 row-major；计算 D=A·B^T。
// B 在 Builder 中声明 ColumnMajor，因为同一段 [N,K] row-major 内存正是数学 [K,N]
// 的 column-major 视图，不需要显式转置。C/D 为 row-major，D 使用 FP32 便于检查累加误差。
//
// 教学边界：SM120 的 dense CollectiveBuilder 在 CUTLASS 4.8.1 中只支持 F8F6F4，
// 不能把本文件的 BF16 配置直接改成 arch::Sm120。FP8/FP4 的 SM120 案例见 cache-dit。

#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/detail/collective/moe_stride_utils.hpp>
#include <cutlass/bfloat16.h>

namespace leetcuda::collective_gemm {

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using TileShape = cute::Shape<cute::_128, cute::_128, cute::_64>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
using FusionOp = cutlass::epilogue::fusion::LinearCombination<ElementD, float, ElementC, float>;

// Epilogue Builder 负责 D=alpha·(A·B^T)+beta·C。先得到 Epilogue 类型，
// mainloop 才能通过 StageCountAutoCarveout 给它保留 shared memory。
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    ElementC, LayoutC, 8, ElementD, LayoutD, 4,
    cutlass::epilogue::collective::EpilogueScheduleAuto, FusionOp>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 8, ElementB, LayoutB, 8, float,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

#endif  // CUTLASS_ARCH_MMA_SM90_SUPPORTED

// 最小 launch 包装。只有 CUTLASS 显式请求 workspace 时才临时分配；
// 生产调用应复用非零 workspace；本例用 cudaFreeAsync 保证释放排在同 stream 的 kernel 后。
inline cutlass::Status bf16(const ElementA* a, const ElementB* b, const ElementC* c, ElementD* d,
                            int M, int N, int K, float alpha = 1.0f, float beta = 0.0f,
                            cudaStream_t stream = nullptr) {
#if !defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  return cutlass::Status::kErrorNotSupported;
#else
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  auto stride_a = cutlass::make_internal_packed_stride(StrideA{}, {M, K, 1});
  auto stride_b = cutlass::make_internal_packed_stride(StrideB{}, {N, K, 1});
  auto stride_c = cutlass::make_internal_packed_stride(StrideC{}, {M, N, 1});
  auto stride_d = cutlass::make_internal_packed_stride(StrideD{}, {M, N, 1});

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      {a, stride_a, b, stride_b},
      {{alpha, beta}, c, stride_c, d, stride_d}};
  Gemm gemm;
  const size_t workspace_size = Gemm::get_workspace_size(arguments);
  void* workspace = nullptr;
  if (workspace_size != 0 && cudaMallocAsync(&workspace, workspace_size, stream) != cudaSuccess) {
    return cutlass::Status::kErrorInternal;
  }
  cutlass::Status status = gemm.initialize(arguments, workspace, stream);
  if (status == cutlass::Status::kSuccess) status = gemm.run(stream);
  if (workspace != nullptr && cudaFreeAsync(workspace, stream) != cudaSuccess) {
    return cutlass::Status::kErrorInternal;
  }
  return status;
#endif
}

}  // namespace leetcuda::collective_gemm
