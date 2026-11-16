#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/detail/collective/moe_stride_utils.hpp>

/*
 * Cache-DiT NVFP4 GEMM: packed data, two-level scales, and the CUTLASS collective.
 * =============================================================================
 *
 * Notation
 * --------
 *   x      : BF16 activation [M,K], stored row-major.
 *   w      : BF16 linear weight [N,K], stored row-major like torch.nn.Linear.
 *   qx/qw  : packed E2M1 data [M,K/2] / [N,K/2], two FP4 values per byte.
 *   sx/sw  : per-16-element UE4M3 block scales in the canonical CUTLASS layout.
 *   gx/gw  : optional FP32 level-1 scales, one per activation row / weight row.
 *
 * fp4_quantize produces the following dequantization contract for each 16-value
 * group g in row r (the same expression applies to a weight row):
 *
 *   g_x[r]       = max_k |x[r,k]| / (448 * 6)
 *   x_norm[r,k]  = x[r,k] / g_x[r]
 *   s_x[r,g]     = UE4M3(max_{k in g} |x_norm[r,k]| / 6)
 *   q_x[r,k]     = E2M1_RNE(x_norm[r,k] / FP32(s_x[r,g]))
 *   x_hat[r,k]   = g_x[r] * FP32(s_x[r,g]) * FP4(q_x[r,k])
 *
 * E4M3's maximum finite value is 448 and E2M1's is 6. The level-1 scale therefore
 * maps the row maximum to 2688 before the block scale is formed. Computing s_x
 * from the rounded UE4M3 value (rather than the unrounded maximum) is required:
 * the GEMM and the quantizer must use exactly the same scale. Zero rows/groups
 * use zero scales and produce zero qdata; reciprocal-by-zero is never evaluated
 * on a meaningful value. See quantize.cu for the bit-exact conversion sequence.
 *
 * The block-scaled MMA evaluates
 *
 *   P[m,n] = sum_g s_x[m,g] * s_w[n,g] *
 *            sum_{k in g} FP4(qx[m,k]) * FP4(qw[n,k])
 *
 * with FP32 accumulation inside the CUTLASS collective. The custom EVT epilogue
 * then folds the level-1 scales and bias without another global-memory pass:
 *
 *   y[m,n] = BF16(g_w[n] * (g_x[m] * P[m,n]) + bias[n])
 *
 * EVT means Epilogue Visitor Tree. It is CUTLASS's way of describing the
 * per-output-element epilogue as a small computation tree instead of a separate
 * hand-written kernel. Leaf nodes fetch the accumulator or broadcast row/column
 * inputs; internal nodes apply multiply/add/activation operations; the root
 * produces the output tile. In this file Sm90AccFetch obtains P, Sm90ColBroadcast
 * supplies g_w[n], Sm90RowBroadcast supplies g_x[m] and bias[n], Sm90Compute
 * performs the arithmetic, and the collective epilogue schedules the final BF16
 * store. The tree is evaluated while the output is still in registers.
 *
 * When level-1 scales are absent, the single-level entry computes P+bias. The
 * multiplication order in the EVT is intentional and is part of the numerical
 * contract. All scale/bias arithmetic is FP32; conversion to BF16 happens once.
 *
 * Why there are two different scale objects
 * -----------------------------------------
 * s_x/s_w are required by the SM120 block-scaled MMA and are loaded per K block.
 * They are not ordinary pointwise multipliers applied after an unscaled GEMM.
 * g_x/g_w are epilogue-only row/column broadcasts. Keeping these roles separate
 * avoids both a full-size [M,N] scale tensor and a second BF16 output roundtrip.
 *
 * Collective construction and execution
 * -------------------------------------
 * 1. Fp4CutlassThreadBlockShape is 128x128x256. K=256 contains sixteen NVFP4
 *    groups and is the native mainloop tile; N must be a multiple of the 128
 *    output tile. M tails are predicated by CUTLASS.
 * 2. A uses RowMajor [M,K]. The input weight is physically [N,K] row-major, but
 *    ColumnMajor in the builder interprets the same allocation as B^T [K,N].
 *    No transpose, repack, or extra allocation is performed at runtime.
 * 3. StageCountAutoCarveout reserves the epilogue's shared storage and lets the
 *    builder choose a legal mainloop pipeline. A 64x64 epilogue subtile is used;
 *    larger subtiles can consume the storage needed by the mainloop stages.
 * 4. The SM120 collective TMA-loads packed A/B and canonical SFA/SFB, issues the
 *    block-scaled tensor operations, and leaves FP32 accumulators for the EVT.
 * 5. fp4_gemm_cutlass_2x constructs packed strides and ScaleConfig's SFA/SFB layouts,
 *    attaches the level-1 scale/bias pointers to FusionArguments, initializes the
 *    GemmUniversalAdapter with any required workspace, and launches on the
 *    caller's current CUDA stream. It does not call torch._scaled_mm.
 *
 * API shape requirements are enforced in gemm.cu and fp4.py: logical K%256==0
 * and N%128==0. The previous hand-written GEMM and its `use_ws` switch were
 * removed rather than keeping a second set of tail/mainloop rules.
 */

namespace cutlass::epilogue::fusion {

// Fusion declaration for the two-level epilogue. The capability flags tell the
// collective builder that the EVT may read one value per output row (g_x), one
// per output column (g_w), and one bias per output column. ElementCompute=float
// is deliberate: accumulator rescaling and bias addition never use BF16 math.
template <class ElementOutput_, class ElementCompute_, class ElementBias_>
struct Fp4RowColScaleBias : FusionOperation {
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementCompute_;
  using ElementBias = ElementBias_;
  static constexpr int AlignmentScalar = 1;
  static constexpr int AlignmentBias = 128 / cute::sizeof_bits_v<ElementBias>;
  static constexpr bool IsPerRowScaleSupported = true;
  static constexpr bool IsPerColScaleSupported = true;
  static constexpr bool IsPerColBiasSupported = true;
};

// EVT tree, from the outermost node inward:
//   homogeneous_multiply_add(bias, g_w, homogeneous(g_x, accumulator))
// Sm90AccFetch reads the collective's FP32 accumulator. ColBroadcast<...,
// Stride<bool,_0,...>> supplies g_w[n], RowBroadcast<..., Stride<_0,bool,...>>
// supplies g_x[m], and the final RowBroadcast supplies bias[n]. The strides are
// broadcast markers, not physical [M,N] tensors. `homogeneous_multiply_add`
// evaluates the fused scale and bias while the output tile is still in registers.
template <class CtaTileShapeMNK, class ElementOutput, class ElementCompute, class ElementBias>
using Sm90Fp4RowColScaleBias = Sm90EVT<
  Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute,
              FloatRoundStyle::round_to_nearest>,
  Sm90RowBroadcast<0, CtaTileShapeMNK, float, ElementCompute, Stride<_0, bool, int64_t>, 4>,
  Sm90EVT<
    Sm90Compute<multiplies, ElementCompute, ElementCompute, FloatRoundStyle::round_to_nearest>,
    Sm90ColBroadcast<0, CtaTileShapeMNK, float, ElementCompute, Stride<bool, _0, int64_t>, 1>,
    Sm90AccFetch>,
  Sm90RowBroadcast<0, CtaTileShapeMNK, ElementBias, ElementCompute, Stride<_0, _1, int64_t>,
                   128 / cute::sizeof_bits_v<ElementBias>>>;

template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore,
          class ElementOutput, class ElementCompute, class ElementBias, class CtaTileShapeMNK,
          class EpilogueTile>
struct FusionCallbacks<
  epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
  Fp4RowColScaleBias<ElementOutput, ElementCompute, ElementBias>, CtaTileShapeMNK, EpilogueTile>
    : Sm90Fp4RowColScaleBias<CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias> {
  using Impl = Sm90Fp4RowColScaleBias<CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias>;
  using Operation = Fp4RowColScaleBias<ElementOutput, ElementCompute, ElementBias>;

  // Host-to-EVT argument adapter. Null scale pointers become a default value of
  // 1.0f and null bias becomes 0. This allows the same compiled epilogue to
  // handle one-sided level-1 scaling without a second kernel instantiation.
  struct Arguments {
    float const* a_scale_ptr = nullptr;
    float const* b_scale_ptr = nullptr;
    ElementBias const* bias_ptr = nullptr;
    Stride<bool, _0, int64_t> dAScale = {bool(1), _0{}, 0};
    Stride<_0, bool, int64_t> dBScale = {_0{}, bool(1), 0};
    Stride<_0, _1, int64_t> dBias = {};

    operator typename Impl::Arguments() const {
      return {
        {b_scale_ptr, 1.0f, dBScale},
        {
          {a_scale_ptr, 1.0f, dAScale},
          {},
          {},
        },
        {bias_ptr, ElementBias(0), dBias},
        {},
      };
    }
  };

  using Impl::Impl;
};

template <class ElementOutput_, class ElementCompute_, class ElementBias_>
struct Fp4BiasOnly : FusionOperation {
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementBias = ElementBias_;
  static constexpr int AlignmentBias = 128 / cute::sizeof_bits_v<ElementBias>;
  static constexpr bool IsPerColBiasSupported = true;
};

template <class CtaTileShapeMNK, class ElementOutput, class ElementCompute, class ElementBias>
using Sm90Fp4BiasOnly =
  Sm90EVT<Sm90Compute<plus, ElementOutput, ElementCompute, FloatRoundStyle::round_to_nearest>,
          Sm90AccFetch,
          Sm90RowBroadcast<0, CtaTileShapeMNK, ElementBias, ElementCompute, Stride<_0, _1, int64_t>,
                           128 / cute::sizeof_bits_v<ElementBias>>>;

template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore,
          class ElementOutput, class ElementCompute, class ElementBias, class CtaTileShapeMNK,
          class EpilogueTile>
struct FusionCallbacks<
  epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
  Fp4BiasOnly<ElementOutput, ElementCompute, ElementBias>, CtaTileShapeMNK, EpilogueTile>
    : Sm90Fp4BiasOnly<CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias> {
  using Impl = Sm90Fp4BiasOnly<CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias>;
  using Operation = Fp4BiasOnly<ElementOutput, ElementCompute, ElementBias>;

  struct Arguments {
    ElementBias const* bias_ptr = nullptr;
    Stride<_0, _1, int64_t> dBias = {};

    operator typename Impl::Arguments() const {
      return {{}, {bias_ptr, ElementBias(0), dBias}, {}};
    }
  };

  using Impl::Impl;
};

}  // namespace cutlass::epilogue::fusion

namespace cache_dit::fp4 {

using Fp4CutlassElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using Fp4CutlassElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using Fp4CutlassElementD = cutlass::bfloat16_t;
using Fp4CutlassFusion =
  cutlass::epilogue::fusion::Fp4RowColScaleBias<Fp4CutlassElementD, float, Fp4CutlassElementD>;
using Fp4CutlassThreadBlockShape = cute::Shape<cute::_128, cute::_128, cute::_256>;
using Fp4CutlassClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

using Fp4CutlassEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
  cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Fp4CutlassThreadBlockShape,
  Fp4CutlassClusterShape, cute::Shape<cute::_64, cute::_64>, float, float, Fp4CutlassElementD,
  cutlass::layout::RowMajor, 8, Fp4CutlassElementD, cutlass::layout::RowMajor, 8,
  cutlass::epilogue::collective::EpilogueScheduleAuto, Fp4CutlassFusion>::CollectiveOp;

using Fp4CutlassMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
  cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Fp4CutlassElementA,
  cutlass::layout::RowMajor, 32, Fp4CutlassElementB, cutlass::layout::ColumnMajor, 32, float,
  Fp4CutlassThreadBlockShape, Fp4CutlassClusterShape,
  cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
    sizeof(typename Fp4CutlassEpilogue::SharedStorage))>,
  cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using Fp4CutlassGemmKernel =
  cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, Fp4CutlassMainloop,
                                       Fp4CutlassEpilogue, void>;
using Fp4CutlassGemm = cutlass::gemm::device::GemmUniversalAdapter<Fp4CutlassGemmKernel>;

using Fp4CutlassSimpleFusion =
  cutlass::epilogue::fusion::Fp4BiasOnly<Fp4CutlassElementD, float, Fp4CutlassElementD>;
using Fp4CutlassSimpleEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
  cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Fp4CutlassThreadBlockShape,
  Fp4CutlassClusterShape, cute::Shape<cute::_64, cute::_64>, float, float, Fp4CutlassElementD,
  cutlass::layout::RowMajor, 8, Fp4CutlassElementD, cutlass::layout::RowMajor, 8,
  cutlass::epilogue::collective::EpilogueScheduleAuto, Fp4CutlassSimpleFusion>::CollectiveOp;
using Fp4CutlassSimpleMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
  cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Fp4CutlassElementA,
  cutlass::layout::RowMajor, 32, Fp4CutlassElementB, cutlass::layout::ColumnMajor, 32, float,
  Fp4CutlassThreadBlockShape, Fp4CutlassClusterShape,
  cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
    sizeof(typename Fp4CutlassSimpleEpilogue::SharedStorage))>,
  cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Fp4CutlassSimpleGemmKernel =
  cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, Fp4CutlassSimpleMainloop,
                                       Fp4CutlassSimpleEpilogue, void>;
using Fp4CutlassSimpleGemm =
  cutlass::gemm::device::GemmUniversalAdapter<Fp4CutlassSimpleGemmKernel>;

// Launch the two-level collective.
//
// Host argument construction mirrors the mathematical data flow:
//   1. Recover logical K from the packed byte width (a.size(1) * 2).
//   2. Build packed A/B/C/D strides without changing any tensor storage.
//   3. Ask Sm1xxBlkScaledConfig for the canonical SFA/SFB layout expected by the
//      collective. fp4_quantize writes that exact byte layout.
//   4. Attach g_x/g_w/bias pointers to the EVT and point C and D at the same
//      output allocation (this is not an in-place accumulation: beta is zero).
//   5. Initialize the adapter and run on `stream`. The only temporary storage is
//      CUTLASS's explicitly requested workspace; qdata and scale buffers are
//      reused directly from the caller.
inline void fp4_gemm_cutlass_2x(const at::Tensor& a, const at::Tensor& b,
                                const at::Tensor& a_scales, const at::Tensor& b_scales,
                                const at::Tensor& a_global_scales,
                                const at::Tensor& b_global_scales,
                                const std::optional<at::Tensor>& bias, at::Tensor& out,
                                cudaStream_t stream) {
  using StrideA = typename Fp4CutlassGemm::GemmKernel::StrideA;
  using StrideB = typename Fp4CutlassGemm::GemmKernel::StrideB;
  using StrideC = typename Fp4CutlassGemm::GemmKernel::StrideC;
  using StrideD = typename Fp4CutlassGemm::GemmKernel::StrideD;
  using ScaleConfig = typename Fp4CutlassGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using FusionArguments =
    typename Fp4CutlassGemm::GemmKernel::CollectiveEpilogue::FusionCallbacks::Arguments;

  const int M = static_cast<int>(a.size(0));
  const int N = static_cast<int>(b.size(0));
  const int K = static_cast<int>(a.size(1) * 2);
  auto stride_a = cutlass::make_internal_packed_stride(StrideA{}, {M, K, 1});
  auto stride_b = cutlass::make_internal_packed_stride(StrideB{}, {N, K, 1});
  auto stride_c = cutlass::make_internal_packed_stride(StrideC{}, {M, N, 1});
  auto stride_d = cutlass::make_internal_packed_stride(StrideD{}, {M, N, 1});
  auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
  auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

  FusionArguments fusion_args{};
  fusion_args.a_scale_ptr =
    a_global_scales.numel() != 0 ? a_global_scales.data_ptr<float>() : nullptr;
  fusion_args.b_scale_ptr =
    b_global_scales.numel() != 0 ? b_global_scales.data_ptr<float>() : nullptr;
  fusion_args.bias_ptr =
    bias.has_value() ? reinterpret_cast<const Fp4CutlassElementD*>(bias->data_ptr<at::BFloat16>())
                     : nullptr;

  auto* out_ptr = reinterpret_cast<Fp4CutlassElementD*>(out.data_ptr<at::BFloat16>());
  typename Fp4CutlassGemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    {
      reinterpret_cast<const Fp4CutlassElementA::DataType*>(a.data_ptr()),
      stride_a,
      reinterpret_cast<const Fp4CutlassElementB::DataType*>(b.data_ptr()),
      stride_b,
      reinterpret_cast<const cutlass::float_ue4m3_t*>(a_scales.data_ptr<at::Float8_e4m3fn>()),
      layout_sfa,
      reinterpret_cast<const cutlass::float_ue4m3_t*>(b_scales.data_ptr<at::Float8_e4m3fn>()),
      layout_sfb,
    },
    {fusion_args, out_ptr, stride_c, out_ptr, stride_d},
  };

  Fp4CutlassGemm gemm;
  cutlass::Status status = cutlass::Status::kSuccess;
  const size_t workspace_size = Fp4CutlassGemm::get_workspace_size(arguments);
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size != 0) {
    workspace = at::empty({static_cast<int64_t>(workspace_size)}, a.options().dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }
  status = gemm.initialize(arguments, workspace_ptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS SM120 FP4 GEMM initialization failed: ", static_cast<int>(status));
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS SM120 FP4 GEMM launch failed: ", static_cast<int>(status));
}

// Single-level counterpart: the block-scaled mainloop is identical, but the EVT
// is only `accumulator + bias`. It exists for isolated comparisons with
// torch._scaled_mm and must not be used to drop level-1 scales from production
// NVFP4Linear, whose recipe always produces both row and column global scales.
inline void fp4_gemm_cutlass_1x(const at::Tensor& a, const at::Tensor& b,
                                const at::Tensor& a_scales, const at::Tensor& b_scales,
                                const std::optional<at::Tensor>& bias, at::Tensor& out,
                                cudaStream_t stream) {
  using Gemm = Fp4CutlassSimpleGemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using ScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using FusionArguments = typename Gemm::GemmKernel::CollectiveEpilogue::FusionCallbacks::Arguments;

  const int M = static_cast<int>(a.size(0));
  const int N = static_cast<int>(b.size(0));
  const int K = static_cast<int>(a.size(1) * 2);
  auto stride_a = cutlass::make_internal_packed_stride(StrideA{}, {M, K, 1});
  auto stride_b = cutlass::make_internal_packed_stride(StrideB{}, {N, K, 1});
  auto stride_c = cutlass::make_internal_packed_stride(StrideC{}, {M, N, 1});
  auto stride_d = cutlass::make_internal_packed_stride(StrideD{}, {M, N, 1});
  auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
  auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

  FusionArguments fusion_args{};
  fusion_args.bias_ptr =
    bias.has_value() ? reinterpret_cast<const Fp4CutlassElementD*>(bias->data_ptr<at::BFloat16>())
                     : nullptr;
  auto* out_ptr = reinterpret_cast<Fp4CutlassElementD*>(out.data_ptr<at::BFloat16>());
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    {
      reinterpret_cast<const Fp4CutlassElementA::DataType*>(a.data_ptr()),
      stride_a,
      reinterpret_cast<const Fp4CutlassElementB::DataType*>(b.data_ptr()),
      stride_b,
      reinterpret_cast<const cutlass::float_ue4m3_t*>(a_scales.data_ptr<at::Float8_e4m3fn>()),
      layout_sfa,
      reinterpret_cast<const cutlass::float_ue4m3_t*>(b_scales.data_ptr<at::Float8_e4m3fn>()),
      layout_sfb,
    },
    {fusion_args, out_ptr, stride_c, out_ptr, stride_d},
  };

  Gemm gemm;
  cutlass::Status status = cutlass::Status::kSuccess;
  const size_t workspace_size = Gemm::get_workspace_size(arguments);
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size != 0) {
    workspace = at::empty({static_cast<int64_t>(workspace_size)}, a.options().dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }
  status = gemm.initialize(arguments, workspace_ptr, stream);
  TORCH_CHECK(
    status == cutlass::Status::kSuccess,
    "CUTLASS SM120 single-level FP4 GEMM initialization failed: ", static_cast<int>(status));
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS SM120 single-level FP4 GEMM launch failed: ", static_cast<int>(status));
}

}  // namespace cache_dit::fp4
