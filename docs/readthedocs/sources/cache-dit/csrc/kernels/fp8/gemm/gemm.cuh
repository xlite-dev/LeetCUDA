#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cute/tensor.hpp>
#include <cutlass/detail/blockwise_scale_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/detail/collective/moe_stride_utils.hpp>

#include "../gemm.h"

// =================================================================================================
// FP8 E4M3 GEMM: quantization contract, CUTLASS collectives, and launch flow
// =================================================================================================
//
// This file implements the GEMM half of W8A8. It consumes already quantized values; it does not
// compute amax or quantize BF16. For a quantization unit U (row, tensor, or block), the producer
// and every consumer share the following convention:
//
//   scale = max(abs(X[U])) / 448
//   Q[U]  = satfinite_e4m3(X[U] / scale)
//   X_hat[U] = scale * Q[U]
//
// E4M3 is the FP8 storage format (finite values saturate to +/-448); scale is always FP32. This is
// not NVFP4's two-level FP4 + UE4M3 scale recipe. A zero unit has scale=0 and Q=0; the quantizer
// uses 1 as the safe division denominator in that case. GEMM never needs the reciprocal scale.
//
// Let A be the physical row-major activation [M,K] and B the physical row-major weight [N,K]. The
// mathematical product is A @ B.T. CUTLASS describes the second operand as a column-major [K,N]
// matrix: that is exactly the same memory as row-major [N,K], so no transpose or repack is needed.
// All three paths accumulate in FP32 and write BF16 output D[M,N]. Their only difference is where
// the FP32 scales are applied:
//
//   per_row:    sa[M], sb[N]
//               D = bias + (sa[m] * sb[n]) * sum_k(QA[m,k] * QB[n,k])
//               Both scales are folded into the custom EVT epilogue.
//
//   per_tensor: scalar sa and sb
//               The same EVT is used with zero scale strides, broadcasting one scalar over M/N.
//               No [M] or [N] temporary scale tensor is allocated.
//
//   per_block:  sa[M,ceil(K/128)], sb[ceil(N/128),ceil(K/128)]
//               Each activation block is 1x128 and each weight block is 128x128. The blockwise
//               mainloop applies sa/sb while accumulating each K128 block. Its epilogue adds bias
//               only; applying the scales again in EVT would be incorrect.
//
// Padding is part of the public Python contract, not this C++ ABI. `fp8_gemm` in fp8.py zero-pads
// qdata/scales/bias and depads D when the selected backend needs aligned shapes. In particular,
// per-block reaches this file with K and N divisible by 128. The C++ entry in fp8.cpp still checks
// its own stricter low-level preconditions so accidental direct C++ misuse fails loudly.
//
// CUTLASS API map
// ---------------
// 1. `gemm::collective::CollectiveBuilder` chooses the mainloop: SM120 TensorOp, A row-major,
//    B column-major, FP32 accumulator, CTA tile 128x128x128 and a 1x1x1 cluster. The alignment
//    value 16 means 16 FP8 elements = 16 bytes. `StageCountAutoCarveout` reserves the epilogue's
//    SharedStorage before selecting a legal mainloop stage count; it is not a stage count itself.
// 2. `epilogue::collective::CollectiveBuilder` chooses a TMA warp-specialized epilogue and our
//    FusionOperation. The 64x64 epilogue tile controls output subtiling, not the quantization
//    block size. Output alignment 8 means eight BF16 elements = 16 bytes.
// 3. `GemmUniversal` combines one mainloop and one epilogue. `GemmUniversalAdapter` supplies the
//    host `Arguments`, `initialize`, `run`, and workspace API used below.
// 4. Per-block additionally passes a pair {data layout, scale-factor layout} for A and B. The
//    `Sm120BlockwiseScaleConfig<1,128,128,K,K>` describes 1x128 A scales and 128x128 B scales.
//    `deduce_layoutSFA/SFB` provides the static layout types to the Builder; at runtime,
//    `tile_atom_to_shape_SFA/SFB` creates layouts for the actual M,N,K problem. These are layouts,
//    not scale values and not an extra quantization pass.
// 5. The launch sequence is: construct packed data strides and runtime scale layouts, fill
//    mainloop/epilogue `Arguments`, call `get_workspace_size`, allocate any workspace with ATen on
//    the current guarded CUDA device, `initialize(arguments, workspace, stream)`, then `run`.
//    `per_*.cu` keeps one thin non-template launch symbol per mode for the pybind-facing fp8.cpp.
//
// EVT map
// -------
// EVT means Epilogue Visitor Tree. `Fp8RowColScaleBias` declares a custom fusion operation and
// `FusionCallbacks` adapts its pointer/stride arguments to the selected TMA epilogue schedule.
// `Sm90AccFetch` reads the mainloop accumulator, `Sm90ColBroadcast` supplies sb[n], and the two
// `Sm90RowBroadcast` nodes supply sa[m] and bias[n]. `homogeneous_multiply_add` and `multiplies`
// express bias + sb*(sa*acc) without materializing a scaled output. `Fp8BiasOnly` is intentionally
// smaller because per-block scale application has already happened in the mainloop.
//
// The pointers in `FusionCallbacks::Arguments` are device pointers. The strides are CuTe broadcast
// strides, not byte strides: the default scale strides advance by one FP32 element; per-tensor
// overrides them with zeros. Null bias uses the operation's default zero value.

namespace cutlass::epilogue::thread {

// PyTorch's approximate="tanh" GELU, including its operation order.
template <typename T>
struct Fp8GeluTanh {
  static const bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  T operator()(T const& value) const {
    float x = static_cast<float>(value);
    float x3 = x * x * x;
    float inner = 0.7978845608028654f * (x + 0.044715f * x3);
    return static_cast<T>(0.5f * x * (1.0f + tanhf(inner)));
  }
};

template <typename T, int N>
struct Fp8GeluTanh<Array<T, N>> {
  static const bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const& value) const {
    Array<T, N> result;
    Fp8GeluTanh<T> op;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(value[i]);
    }
    return result;
  }
};

}  // namespace cutlass::epilogue::thread

namespace cutlass::epilogue::fusion {

// EVT (Epilogue Visitor Tree) for per-row/per-tensor FP8 GEMM:
//   D = bias + b_scale * (a_scale * accumulator).
// Per-tensor uses zero broadcast strides, so one scalar is reused without an
// expanded [M] or [N] allocation. Per-block scaling is handled in the mainloop.
template <class ElementOutput_, class ElementCompute_, class ElementBias_>
struct Fp8RowColScaleBias : FusionOperation {
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

template <class CtaTileShapeMNK, class ElementOutput, class ElementCompute, class ElementBias>
using Sm90Fp8RowColScaleBias = Sm90EVT<
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
  Fp8RowColScaleBias<ElementOutput, ElementCompute, ElementBias>, CtaTileShapeMNK, EpilogueTile>
    : Sm90Fp8RowColScaleBias<CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias> {
  using Impl = Sm90Fp8RowColScaleBias<CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias>;
  using Operation = Fp8RowColScaleBias<ElementOutput, ElementCompute, ElementBias>;

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

// D = activation(bias + b_scale * (a_scale * accumulator)). The inner tree still
// writes BF16 before the activation, matching the unfused GEMM -> GELU chain.
template <template <class> class ActivationFn_, class ElementOutput_, class ElementCompute_,
          class ElementBias_>
struct Fp8RowColScaleBiasActivation : FusionOperation {
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementCompute_;
  using ElementBias = ElementBias_;
  using ActivationFn = ActivationFn_<ElementCompute_>;
  static constexpr int AlignmentScalar = 1;
  static constexpr int AlignmentBias = 128 / cute::sizeof_bits_v<ElementBias>;
  static constexpr bool IsPerRowScaleSupported = true;
  static constexpr bool IsPerColScaleSupported = true;
  static constexpr bool IsPerColBiasSupported = true;
  static constexpr bool IsEltActSupported = true;
};

template <template <class> class ActivationFn, class CtaTileShapeMNK, class ElementOutput,
          class ElementCompute, class ElementBias>
using Sm90Fp8RowColScaleBiasActivation = Sm90EVT<
  Sm90Compute<ActivationFn, ElementOutput, ElementCompute, FloatRoundStyle::round_to_nearest>,
  Sm90Fp8RowColScaleBias<CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias>>;

template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore,
          template <class> class ActivationFn, class ElementOutput, class ElementCompute,
          class ElementBias, class CtaTileShapeMNK, class EpilogueTile>
struct FusionCallbacks<
  epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
  Fp8RowColScaleBiasActivation<ActivationFn, ElementOutput, ElementCompute, ElementBias>,
  CtaTileShapeMNK, EpilogueTile>
    : Sm90Fp8RowColScaleBiasActivation<ActivationFn, CtaTileShapeMNK, ElementOutput, ElementCompute,
                                       ElementBias> {
  using Impl = Sm90Fp8RowColScaleBiasActivation<ActivationFn, CtaTileShapeMNK, ElementOutput,
                                                ElementCompute, ElementBias>;
  using Operation =
    Fp8RowColScaleBiasActivation<ActivationFn, ElementOutput, ElementCompute, ElementBias>;

  struct Arguments {
    float const* a_scale_ptr = nullptr;
    float const* b_scale_ptr = nullptr;
    ElementBias const* bias_ptr = nullptr;
    Stride<bool, _0, int64_t> dAScale = {bool(1), _0{}, 0};
    Stride<_0, bool, int64_t> dBScale = {_0{}, bool(1), 0};
    Stride<_0, _1, int64_t> dBias = {};
    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementCompute, ElementCompute,
                                                     FloatRoundStyle::round_to_nearest>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      typename Sm90Fp8RowColScaleBias<CtaTileShapeMNK, ElementOutput, ElementCompute,
                                      ElementBias>::Arguments scale_args = {
        {b_scale_ptr, 1.0f, dBScale},
        {{a_scale_ptr, 1.0f, dAScale}, {}, {}},
        {bias_ptr, ElementBias(0), dBias},
        {},
      };
      return {scale_args, activation};
    }
  };

  using Impl::Impl;
};

// Bias-only EVT for the blockwise mainloop, which already rescaled every K128
// block before producing the final accumulator.
template <class ElementOutput_, class ElementCompute_, class ElementBias_>
struct Fp8BiasOnly : FusionOperation {
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementBias = ElementBias_;
  static constexpr int AlignmentBias = 128 / cute::sizeof_bits_v<ElementBias>;
  static constexpr bool IsPerColBiasSupported = true;
};

template <class CtaTileShapeMNK, class ElementOutput, class ElementCompute, class ElementBias>
using Sm90Fp8BiasOnly =
  Sm90EVT<Sm90Compute<plus, ElementOutput, ElementCompute, FloatRoundStyle::round_to_nearest>,
          Sm90AccFetch,
          Sm90RowBroadcast<0, CtaTileShapeMNK, ElementBias, ElementCompute, Stride<_0, _1, int64_t>,
                           128 / cute::sizeof_bits_v<ElementBias>>>;

template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore,
          class ElementOutput, class ElementCompute, class ElementBias, class CtaTileShapeMNK,
          class EpilogueTile>
struct FusionCallbacks<
  epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
  Fp8BiasOnly<ElementOutput, ElementCompute, ElementBias>, CtaTileShapeMNK, EpilogueTile>
    : Sm90Fp8BiasOnly<CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias> {
  using Impl = Sm90Fp8BiasOnly<CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias>;
  using Operation = Fp8BiasOnly<ElementOutput, ElementCompute, ElementBias>;

  struct Arguments {
    ElementBias const* bias_ptr = nullptr;
    Stride<_0, _1, int64_t> dBias = {};

    operator typename Impl::Arguments() const {
      return {{}, {bias_ptr, ElementBias(0), dBias}, {}};
    }
  };

  using Impl::Impl;
};

template <template <class> class ActivationFn_, class ElementOutput_, class ElementCompute_,
          class ElementBias_>
struct Fp8BiasOnlyActivation : FusionOperation {
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementBias = ElementBias_;
  using ActivationFn = ActivationFn_<ElementCompute_>;
  static constexpr int AlignmentBias = 128 / cute::sizeof_bits_v<ElementBias>;
  static constexpr bool IsPerColBiasSupported = true;
  static constexpr bool IsEltActSupported = true;
};

template <template <class> class ActivationFn, class CtaTileShapeMNK, class ElementOutput,
          class ElementCompute, class ElementBias>
using Sm90Fp8BiasOnlyActivation = Sm90EVT<
  Sm90Compute<ActivationFn, ElementOutput, ElementCompute, FloatRoundStyle::round_to_nearest>,
  Sm90Fp8BiasOnly<CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias>>;

template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore,
          template <class> class ActivationFn, class ElementOutput, class ElementCompute,
          class ElementBias, class CtaTileShapeMNK, class EpilogueTile>
struct FusionCallbacks<
  epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
  Fp8BiasOnlyActivation<ActivationFn, ElementOutput, ElementCompute, ElementBias>, CtaTileShapeMNK,
  EpilogueTile> : Sm90Fp8BiasOnlyActivation<ActivationFn, CtaTileShapeMNK, ElementOutput,
                                            ElementCompute, ElementBias> {
  using Impl = Sm90Fp8BiasOnlyActivation<ActivationFn, CtaTileShapeMNK, ElementOutput,
                                         ElementCompute, ElementBias>;
  using Operation = Fp8BiasOnlyActivation<ActivationFn, ElementOutput, ElementCompute, ElementBias>;

  struct Arguments {
    ElementBias const* bias_ptr = nullptr;
    Stride<_0, _1, int64_t> dBias = {};
    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementCompute, ElementCompute,
                                                     FloatRoundStyle::round_to_nearest>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      typename Sm90Fp8BiasOnly<CtaTileShapeMNK, ElementOutput, ElementCompute,
                               ElementBias>::Arguments bias_args = {
        {},
        {bias_ptr, ElementBias(0), dBias},
        {},
      };
      return {bias_args, activation};
    }
  };

  using Impl::Impl;
};

}  // namespace cutlass::epilogue::fusion

namespace cache_dit::fp8 {

using Fp8ElementA = cutlass::float_e4m3_t;
using Fp8ElementB = cutlass::float_e4m3_t;
using Fp8ElementD = cutlass::bfloat16_t;
using Fp8TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
using Fp8ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
using Fp8EpilogueTile = cute::Shape<cute::_64, cute::_64>;
using Fp8ScaleFusion =
  cutlass::epilogue::fusion::Fp8RowColScaleBias<Fp8ElementD, float, Fp8ElementD>;
using Fp8BiasFusion = cutlass::epilogue::fusion::Fp8BiasOnly<Fp8ElementD, float, Fp8ElementD>;

// Per-row and per-tensor share one compiled kernel family. kPerTensor changes only runtime EVT
// strides, avoiding four more template instantiations for A-scalar/B-scalar combinations.
using Fp8RowEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
  cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, Fp8TileShape, Fp8ClusterShape,
  Fp8EpilogueTile, float, float, Fp8ElementD, cutlass::layout::RowMajor, 8, Fp8ElementD,
  cutlass::layout::RowMajor, 8, cutlass::epilogue::collective::EpilogueScheduleAuto,
  Fp8ScaleFusion>::CollectiveOp;
using Fp8RowMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
  cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, Fp8ElementA, cutlass::layout::RowMajor, 16,
  Fp8ElementB, cutlass::layout::ColumnMajor, 16, float, Fp8TileShape, Fp8ClusterShape,
  cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
    sizeof(typename Fp8RowEpilogue::SharedStorage))>,
  cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Fp8RowGemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                                              Fp8RowMainloop, Fp8RowEpilogue, void>;
using Fp8RowGemm = cutlass::gemm::device::GemmUniversalAdapter<Fp8RowGemmKernel>;

template <template <class> class ActivationFn>
using Fp8RowActivationFusion =
  cutlass::epilogue::fusion::Fp8RowColScaleBiasActivation<ActivationFn, Fp8ElementD, float,
                                                          Fp8ElementD>;
template <template <class> class ActivationFn>
using Fp8RowActivationEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
  cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, Fp8TileShape, Fp8ClusterShape,
  Fp8EpilogueTile, float, float, Fp8ElementD, cutlass::layout::RowMajor, 8, Fp8ElementD,
  cutlass::layout::RowMajor, 8, cutlass::epilogue::collective::EpilogueScheduleAuto,
  Fp8RowActivationFusion<ActivationFn>>::CollectiveOp;
template <template <class> class ActivationFn>
using Fp8RowActivationMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
  cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, Fp8ElementA, cutlass::layout::RowMajor, 16,
  Fp8ElementB, cutlass::layout::ColumnMajor, 16, float, Fp8TileShape, Fp8ClusterShape,
  cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
    sizeof(typename Fp8RowActivationEpilogue<ActivationFn>::SharedStorage))>,
  cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
template <template <class> class ActivationFn>
using Fp8RowActivationGemmKernel =
  cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                       Fp8RowActivationMainloop<ActivationFn>,
                                       Fp8RowActivationEpilogue<ActivationFn>, void>;
template <template <class> class ActivationFn>
using Fp8RowActivationGemm =
  cutlass::gemm::device::GemmUniversalAdapter<Fp8RowActivationGemmKernel<ActivationFn>>;

// Scale-factor layout types: what `deduce_layoutSFA/SFB` actually does
// ---------------------------------------------------------------------
// SFA/SFB mean Scale Factor A/B. A CuTe Layout describes a coordinate -> memory-offset mapping;
// it contains no data. `deduce_layoutSFA()` returns a default `LayoutSFA{}` whose static modes
// encode the scale granularity and major order, while runtime block counts are int32 placeholders.
// We use it only inside `decltype`, which is unevaluated: no function executes, no scale/amax is
// computed, no storage is allocated, and no TMA operation is emitted. Its purpose is to give
// CollectiveBuilder a *type* from which it selects the SM120 blockwise mainloop.
//
// The config below says: A scales cover 1x128 elements, B scales cover 128x128 elements, and both
// scale tables are K-major. In CUTLASS's nested-layout notation, the deduced SFA type is
// conceptually
//
//   shape  = ((1, dynamic_M), (128, dynamic_K_blocks), dynamic_L)
//   stride = ((0, dynamic_K_blocks), (0, 1), dynamic_batch_stride).
//
// `Int<1>` and `Int<128>` are compile-time extents; the dynamic extents are filled later. The
// SM120 mainloop reads `size<0,0>(LayoutSFA)` = 1 and `size<1,0>(LayoutSFB)` = 128 as the M/N/K
// scale granularities, and checks them against TileShape (including K tile == 128). The K-major
// stride type also tells it that consecutive scale-table entries belong to consecutive K128
// blocks. SFB plays the same role with granularity N=128 instead of M=1.
//
// Do not pass the deduced object as the final runtime layout. After M,N,K are known, the launch
// function calls `tile_atom_to_shape_SFA/SFB` to instantiate those dynamic extents and strides.
// For example, with M=256, K=512 and G=K/128=4, runtime SFA describes our ordinary row-major
// scale table [256,4]: offset_sa(m,g) = m*4+g. Runtime SFB describes row-major [N/128,G]:
// offset_sb(nb,g) = nb*G+g. The mainloop receives `a_scale + layout_sfa` and `b_scale + layout_sfb`
// in its Arguments and uses those mappings to fetch the scale for each MMA K block.
//
// In short:
//   deduce_layoutSFA/SFB       -> compile-time layout *type* and mainloop selection;
//   tile_atom_to_shape_SFA/SFB -> runtime layout *value* used to address the actual scale arrays.
// The two must come from the same ScaleConfig. Changing scale storage requires changing the
// producer and this config together; merely changing the deduced alias would reinterpret memory.
using Fp8BlockScaleConfig =
  cutlass::detail::Sm120BlockwiseScaleConfig<1, 128, 128, cute::UMMA::Major::K,
                                             cute::UMMA::Major::K>;
using Fp8LayoutSFA = decltype(Fp8BlockScaleConfig::deduce_layoutSFA());
using Fp8LayoutSFB = decltype(Fp8BlockScaleConfig::deduce_layoutSFB());
using Fp8BlockEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
  cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, Fp8TileShape, Fp8ClusterShape,
  Fp8EpilogueTile, float, float, Fp8ElementD, cutlass::layout::RowMajor, 8, Fp8ElementD,
  cutlass::layout::RowMajor, 8, cutlass::epilogue::collective::EpilogueScheduleAuto,
  Fp8BiasFusion>::CollectiveOp;
using Fp8BlockMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
  cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, Fp8ElementA,
  cute::tuple<cutlass::layout::RowMajor, Fp8LayoutSFA>, 16, Fp8ElementB,
  cute::tuple<cutlass::layout::ColumnMajor, Fp8LayoutSFB>, 16, float, Fp8TileShape, Fp8ClusterShape,
  cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
    sizeof(typename Fp8BlockEpilogue::SharedStorage))>,
  cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Fp8BlockGemmKernel =
  cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, Fp8BlockMainloop,
                                       Fp8BlockEpilogue, void>;
using Fp8BlockGemm = cutlass::gemm::device::GemmUniversalAdapter<Fp8BlockGemmKernel>;

template <template <class> class ActivationFn>
using Fp8BlockActivationFusion =
  cutlass::epilogue::fusion::Fp8BiasOnlyActivation<ActivationFn, Fp8ElementD, float, Fp8ElementD>;
template <template <class> class ActivationFn>
using Fp8BlockActivationEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
  cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, Fp8TileShape, Fp8ClusterShape,
  Fp8EpilogueTile, float, float, Fp8ElementD, cutlass::layout::RowMajor, 8, Fp8ElementD,
  cutlass::layout::RowMajor, 8, cutlass::epilogue::collective::EpilogueScheduleAuto,
  Fp8BlockActivationFusion<ActivationFn>>::CollectiveOp;
template <template <class> class ActivationFn>
using Fp8BlockActivationMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
  cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, Fp8ElementA,
  cute::tuple<cutlass::layout::RowMajor, Fp8LayoutSFA>, 16, Fp8ElementB,
  cute::tuple<cutlass::layout::ColumnMajor, Fp8LayoutSFB>, 16, float, Fp8TileShape, Fp8ClusterShape,
  cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
    sizeof(typename Fp8BlockActivationEpilogue<ActivationFn>::SharedStorage))>,
  cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
template <template <class> class ActivationFn>
using Fp8BlockActivationGemmKernel =
  cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                       Fp8BlockActivationMainloop<ActivationFn>,
                                       Fp8BlockActivationEpilogue<ActivationFn>, void>;
template <template <class> class ActivationFn>
using Fp8BlockActivationGemm =
  cutlass::gemm::device::GemmUniversalAdapter<Fp8BlockActivationGemmKernel<ActivationFn>>;

template <bool kPerTensor>
inline void fp8_gemm_cutlass_rowwise(const void* a, const void* b, const float* a_scale,
                                     const float* b_scale, const void* bias, void* out, int M,
                                     int N, int K, cudaStream_t stream) {
  using StrideA = typename Fp8RowGemm::GemmKernel::StrideA;
  using StrideB = typename Fp8RowGemm::GemmKernel::StrideB;
  using StrideC = typename Fp8RowGemm::GemmKernel::StrideC;
  using StrideD = typename Fp8RowGemm::GemmKernel::StrideD;
  using FusionArguments =
    typename Fp8RowGemm::GemmKernel::CollectiveEpilogue::FusionCallbacks::Arguments;

  // The trailing 1 is the GEMM batch dimension. These packed strides describe the existing torch
  // allocations; they do not allocate or transpose A/B. C and D alias the same BF16 output because
  // this is an out=A@B.T+bias operation, not a C += A@B.T operation.
  auto stride_a = cutlass::make_internal_packed_stride(StrideA{}, {M, K, 1});
  auto stride_b = cutlass::make_internal_packed_stride(StrideB{}, {N, K, 1});
  auto stride_c = cutlass::make_internal_packed_stride(StrideC{}, {M, N, 1});
  auto stride_d = cutlass::make_internal_packed_stride(StrideD{}, {M, N, 1});
  FusionArguments fusion_args{};
  fusion_args.a_scale_ptr = a_scale;
  fusion_args.b_scale_ptr = b_scale;
  fusion_args.bias_ptr = static_cast<const Fp8ElementD*>(bias);
  if constexpr (kPerTensor) {
    fusion_args.dAScale = {bool(0), cute::_0{}, 0};
    fusion_args.dBScale = {cute::_0{}, bool(0), 0};
  }
  auto* out_ptr = static_cast<Fp8ElementD*>(out);
  typename Fp8RowGemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    {static_cast<const Fp8ElementA*>(a), stride_a, static_cast<const Fp8ElementB*>(b), stride_b},
    {fusion_args, out_ptr, stride_c, out_ptr, stride_d},
  };

  Fp8RowGemm gemm;
  const size_t workspace_size = Fp8RowGemm::get_workspace_size(arguments);
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size != 0) {
    workspace = at::empty({static_cast<int64_t>(workspace_size)},
                          at::TensorOptions().device(at::kCUDA).dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }
  auto status = gemm.initialize(arguments, workspace_ptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS SM120 FP8 rowwise GEMM initialization failed: ", static_cast<int>(status));
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS SM120 FP8 rowwise GEMM launch failed: ", static_cast<int>(status));
}

inline void fp8_gemm_cutlass_blockwise(const void* a, const void* b, const float* a_scale,
                                       const float* b_scale, const void* bias, void* out, int M,
                                       int N, int K, int num_k_blocks, cudaStream_t stream) {
  using StrideA = typename Fp8BlockGemm::GemmKernel::StrideA;
  using StrideB = typename Fp8BlockGemm::GemmKernel::StrideB;
  using StrideC = typename Fp8BlockGemm::GemmKernel::StrideC;
  using StrideD = typename Fp8BlockGemm::GemmKernel::StrideD;
  using FusionArguments =
    typename Fp8BlockGemm::GemmKernel::CollectiveEpilogue::FusionCallbacks::Arguments;

  auto stride_a = cutlass::make_internal_packed_stride(StrideA{}, {M, K, 1});
  auto stride_b = cutlass::make_internal_packed_stride(StrideB{}, {N, K, 1});
  auto stride_c = cutlass::make_internal_packed_stride(StrideC{}, {M, N, 1});
  auto stride_d = cutlass::make_internal_packed_stride(StrideD{}, {M, N, 1});
  auto layout_sfa = Fp8BlockScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
  auto layout_sfb = Fp8BlockScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
  FusionArguments fusion_args{};
  fusion_args.bias_ptr = static_cast<const Fp8ElementD*>(bias);
  auto* out_ptr = static_cast<Fp8ElementD*>(out);
  typename Fp8BlockGemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    {
      static_cast<const Fp8ElementA*>(a),
      stride_a,
      static_cast<const Fp8ElementB*>(b),
      stride_b,
      a_scale,
      layout_sfa,
      b_scale,
      layout_sfb,
    },
    {fusion_args, out_ptr, stride_c, out_ptr, stride_d},
  };
  (void)num_k_blocks;  // The scale layout derives K/128 from logical K.

  Fp8BlockGemm gemm;
  const size_t workspace_size = Fp8BlockGemm::get_workspace_size(arguments);
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size != 0) {
    workspace = at::empty({static_cast<int64_t>(workspace_size)},
                          at::TensorOptions().device(at::kCUDA).dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }
  auto status = gemm.initialize(arguments, workspace_ptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS SM120 FP8 blockwise GEMM initialization failed: ", static_cast<int>(status));
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS SM120 FP8 blockwise GEMM launch failed: ", static_cast<int>(status));
}

template <bool kPerTensor, template <class> class ActivationFn>
inline void fp8_gemm_cutlass_rowwise_activation(const void* a, const void* b, const float* a_scale,
                                                const float* b_scale, const void* bias, void* out,
                                                int M, int N, int K, cudaStream_t stream) {
  using Gemm = Fp8RowActivationGemm<ActivationFn>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using FusionArguments = typename Gemm::GemmKernel::CollectiveEpilogue::FusionCallbacks::Arguments;

  auto stride_a = cutlass::make_internal_packed_stride(StrideA{}, {M, K, 1});
  auto stride_b = cutlass::make_internal_packed_stride(StrideB{}, {N, K, 1});
  auto stride_c = cutlass::make_internal_packed_stride(StrideC{}, {M, N, 1});
  auto stride_d = cutlass::make_internal_packed_stride(StrideD{}, {M, N, 1});
  FusionArguments fusion_args{};
  fusion_args.a_scale_ptr = a_scale;
  fusion_args.b_scale_ptr = b_scale;
  fusion_args.bias_ptr = static_cast<const Fp8ElementD*>(bias);
  if constexpr (kPerTensor) {
    fusion_args.dAScale = {bool(0), cute::_0{}, 0};
    fusion_args.dBScale = {cute::_0{}, bool(0), 0};
  }
  auto* out_ptr = static_cast<Fp8ElementD*>(out);
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    {static_cast<const Fp8ElementA*>(a), stride_a, static_cast<const Fp8ElementB*>(b), stride_b},
    {fusion_args, out_ptr, stride_c, out_ptr, stride_d},
  };

  Gemm gemm;
  const size_t workspace_size = Gemm::get_workspace_size(arguments);
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size != 0) {
    workspace = at::empty({static_cast<int64_t>(workspace_size)},
                          at::TensorOptions().device(at::kCUDA).dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }
  auto status = gemm.initialize(arguments, workspace_ptr, stream);
  TORCH_CHECK(
    status == cutlass::Status::kSuccess,
    "CUTLASS SM120 FP8 activation GEMM initialization failed: ", static_cast<int>(status));
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS SM120 FP8 activation GEMM launch failed: ", static_cast<int>(status));
}

template <template <class> class ActivationFn>
inline void fp8_gemm_cutlass_blockwise_activation(const void* a, const void* b,
                                                  const float* a_scale, const float* b_scale,
                                                  const void* bias, void* out, int M, int N, int K,
                                                  int num_k_blocks, cudaStream_t stream) {
  using Gemm = Fp8BlockActivationGemm<ActivationFn>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using FusionArguments = typename Gemm::GemmKernel::CollectiveEpilogue::FusionCallbacks::Arguments;

  auto stride_a = cutlass::make_internal_packed_stride(StrideA{}, {M, K, 1});
  auto stride_b = cutlass::make_internal_packed_stride(StrideB{}, {N, K, 1});
  auto stride_c = cutlass::make_internal_packed_stride(StrideC{}, {M, N, 1});
  auto stride_d = cutlass::make_internal_packed_stride(StrideD{}, {M, N, 1});
  auto layout_sfa = Fp8BlockScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
  auto layout_sfb = Fp8BlockScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
  FusionArguments fusion_args{};
  fusion_args.bias_ptr = static_cast<const Fp8ElementD*>(bias);
  auto* out_ptr = static_cast<Fp8ElementD*>(out);
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    {
      static_cast<const Fp8ElementA*>(a),
      stride_a,
      static_cast<const Fp8ElementB*>(b),
      stride_b,
      a_scale,
      layout_sfa,
      b_scale,
      layout_sfb,
    },
    {fusion_args, out_ptr, stride_c, out_ptr, stride_d},
  };
  (void)num_k_blocks;

  Gemm gemm;
  const size_t workspace_size = Gemm::get_workspace_size(arguments);
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size != 0) {
    workspace = at::empty({static_cast<int64_t>(workspace_size)},
                          at::TensorOptions().device(at::kCUDA).dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }
  auto status = gemm.initialize(arguments, workspace_ptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS SM120 FP8 blockwise activation GEMM initialization failed: ",
              static_cast<int>(status));
  status = gemm.run(stream);
  TORCH_CHECK(
    status == cutlass::Status::kSuccess,
    "CUTLASS SM120 FP8 blockwise activation GEMM launch failed: ", static_cast<int>(status));
}

}  // namespace cache_dit::fp8
