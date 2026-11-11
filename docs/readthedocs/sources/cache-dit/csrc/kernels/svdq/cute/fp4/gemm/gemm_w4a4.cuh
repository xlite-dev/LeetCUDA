#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

#include <cuda_bf16.h>

#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/tensor_zip.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/device_kernel.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "common.h"

namespace svdq::cute_kernels {

using namespace cute;

// Reading guide for the CuTe terms used below:
//   gmem     = device global memory; smem = CTA shared memory; RF = per-thread registers.
//   TMA      = asynchronous bulk copy engine between gmem and smem.
//   fragment = the small per-thread register tile consumed or produced by one MMA atom.
//   Layout   = a compile-time map from logical coordinates to physical addresses, not a transpose.
//   partition_S/partition_D = create the source/destination per-thread views of a TiledCopy.
//   mbarrier = shared-memory arrival/transaction barrier used to publish one pipeline stage.
//
// Names used by the math and layouts:
//   SF      = Scale Factor, the FP8_E4M3 micro-scale shared by 16 NVFP4 codes; it is NOT an
//             abbreviation for shared memory or shared fragment. SFA is activation SF and SFB is
//             weight/B-side SF. ascales/wscales are the corresponding Python/native tensor names.
//   A/B     = activation and weight matrices of the main NVFP4 GEMM P=A@B^T.
//   D       = BF16 result of the LoRA-down projection, with shape [M,R].
//   L1      = BF16 LoRA-up weight, with shape [N,R]. L1 is a matrix name, NOT the L1 cache.
//   O       = final BF16 output tile in smem; C/tCrC is the FP32 accumulator before conversion.
//   mX      = a gmem tensor view, sX = an smem tensor view, tCrX = a thread register fragment.
//   tCsX    = a per-thread copy-source view; g = global pipeline step, stage = physical ring slot.
//
// The comments in gemm_w4a4_kernel are organized as Stage 1..9 and follow one CTA from
// descriptor-backed gmem views to the final BF16 output tile.

// =================================================================================================
// SVDQ NVFP4 + BF16 LoRA-up: mathematical and dataflow contract
// =================================================================================================
// For each 16-element K group, NVFP4 stores E2M1 codes plus one FP8_E4M3 scale:
//
//   dequant(code[k]) = E2M1(code[k]) * FP8_E4M3(scale[k / 16])
//   P = dequant(A) [M,K] @ dequant(B)^T [K,N]
//   Y = BF16((alpha * P) * wcscales[N] + bias[N] + D [M,R] @ L1^T [R,N])
//
// D is the LoRA-down result and L1 is the LoRA-up weight. lora_scales are folded into D by the
// independent CuTe activation producer before this GEMM starts. alpha and wcscales apply only to
// the NVFP4 product; neither may scale the LoRA term. Both matrix products accumulate into the
// same FP32 C fragment, so there is no cross-warpgroup reduction and only one BF16 rounding.
//
// The implementation deliberately mixes two input paths:
//   B/SFB: packed gmem -> TMA -> smem -> register fragment (shared by all consumer warps)
//   A/SFA: native packed gmem -> direct G2R loads -> register fragment (no activation repack)
//   D/L1:  row-major BF16 gmem -> TMA -> A/B smem overlay -> BF16 register fragments
//
// Keeping A/SFA in the producer's native layout avoids a runtime transpose/repack. Moving B
// through smem removes the old kernel's repeated per-warp weight reads. The two CuTe TiledMMAs
// have different input types and K extents, but statically identical C partitions.
//
// Only BF16 D/L1 and BF16 output are instantiated here. FP16 LoRA remains on the old native
// path. Adding FP16 is not a pointer cast: it needs an F32F16F16F32 atom, matching TMA/copy
// types, an FP16 output ABI, and a separately verified rounding contract.

// NVFP4 block scaling carries one FP8_E4M3 factor per 16 K elements. E2M1 is the signed 4-bit
// code type; UE4M3 is the unsigned 8-bit scale type (scales are non-negative). CUTLASS stores
// these scales in 128-row x 4-factor (512 B) atoms, exactly the box of one BK=64 K step.
constexpr int kSFVec = 16;
constexpr int kSFVecPerBlk = 4;
constexpr int kRowBlock = 128;

// Express the flat scale-factor allocation in logical data coordinates (mn,K), not in a second
// packed-byte coordinate system. For row=n and scale group g=k/16, the physical byte offset is
//
//   (n/128) * (K/16 * 128) + (g/4) * 512
//     + 16 * (n % 32) + 4 * ((n/32) % 4) + (g % 4).
//
// The K mode has shape 16 with stride zero inside a group: all 16 codes logically reference the
// same scale, but the allocation contains only one byte, not 16 copies. Step<_2,_1> tiles the
// atom K-first and then advances to the next 128-row block. This layout is used for SFB TMA;
// ascales has the same byte count but intentionally retains the different native producer ordering.
CUTE_HOST_DEVICE constexpr auto sf_gmem_layout(int mn, int K) {
  using Config = cutlass::detail::Sm1xxBlockScaledConfig<kSFVec>;
  return tile_to_shape(typename Config::SfAtom{}, make_shape(mn, K), Step<_2, _1>{});
}

constexpr size_t sf_buffer_bytes(int mn, int K) {
  const int mn_padded = (mn + kRowBlock - 1) / kRowBlock * kRowBlock;
  return static_cast<size_t>(mn_padded / kRowBlock) * (K / 64) * 512;
}

// Smem counterpart of the 512 B scale atom. ((32,4)):(16,4) maps 128 logical rows onto four
// 32-row panels, while (16,4):(0,1) retains the K broadcast and four consecutive FP8 factors.
// kMN may be 128 or 256; kMN/128 prepends complete 512 B row blocks. kBK/64 similarly prepends
// whole K atoms when a pipeline stage contains two m16n8k64 instructions. The smem and gmem
// orderings therefore match, so TMA can copy the scales without a runtime shuffle. The later
// partition helpers, rather than a flat smem pointer cast, recover each thread's SF register.
template <class TiledMma_, int kMN, int kBK_>
struct SfSmemAtom {
  static constexpr int kMMA_NSF = size<2>(typename TiledMma_::AtomShape_MNK{}) / kSFVec;
  using Config = cutlass::detail::Sm1xxBlockScaledConfig<kSFVec>;
  using Blk_Elems = decltype(typename Config::Blk_MN{} * typename Config::Blk_SF{});
  using mnBlockShape = Shape<_32, _4>;
  using mnBlockStride = Stride<_16, _4>;
  using kBlockShape = Shape<Int<kSFVec>, Int<kMMA_NSF>>;
  using kBlockStride = Stride<_0, _1>;
  using shapeM = decltype(prepend(Int<kMN / kRowBlock>{}, mnBlockShape{}));
  using strideM = decltype(prepend(Blk_Elems{}, mnBlockStride{}));
  using shapeK = decltype(prepend(
    make_shape(Int<kSFVecPerBlk / kMMA_NSF>{}, Int<kBK_ / kSFVec / kSFVecPerBlk>{}),
    kBlockShape{}));
  using strideK =
    decltype(prepend(make_stride(Int<kMMA_NSF>{}, Int<kMN / kRowBlock * 512>{}), kBlockStride{}));
  using type =
    decltype(make_layout(make_shape(shapeM{}, shapeK{}), make_stride(strideM{}, strideK{})));
};

// Scale-factor operands are neither plain smem tensors nor ordinary row-major fragments. The
// MMA atom's SFALayout/SFBLayout maps (thread,value) to logical (M/N,K), including zero-stride
// broadcasts and the 32-wide N permutation. The helpers below mirror the CUTLASS SM120
// collective in three stages:
//   1. thrfrg_sf*: compose the logical tensor with the atom's SF TV layout;
//   2. partition_fragment_sf*: select this thread's slice and materialize one packed u32;
//   3. layout_sf*_tv: construct the inverse thread map used by the smem->register tiled copy.
// A direct linear load would put another lane's scales into the MMA operand. The zip tensor in
// the mainloop then presents {data,SF} as the two operands required by block-scaled CuTe MMA.
namespace sf_partition {

// thrfrg_sfa = "thread-fragment view for SFA". It answers: if the activation SF tensor is viewed
// through this TiledMMA, which logical (M,K) coordinates belong to each thread and each value?
// The function only composes layouts; it neither allocates registers nor loads any scale.
//   logical_divide: split the full SF tensor at the TiledMMA's M/K permutation boundaries.
//   zipped_divide:  separate one m16n8k64 atom from the remaining M/K atom repetitions.
//   compose(SFALayout): apply the hardware atom's special (thread,value)->(M,K) map.
//   final zipped_divide: expose a hierarchical (thread/value, remaining M/K tiles) view.
// partition_fragment_sfa later uses this view to select one physical thread's register fragment.
template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto thrfrg_sfa(SFATensor &&sfatensor,
                                           TiledMMA<Atom, TiledThr, TiledPerm> &mma) {
  CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});
  using AtomShape_MNK = typename Atom::Shape_MNK;
  using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;
  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto t_tile = make_tile(get<0>(permutation_mnk), get<2>(permutation_mnk));
  auto t_tensor = logical_divide(sfatensor, t_tile);
  auto a_tile =
    make_tile(make_layout(size<0>(AtomShape_MNK{})), make_layout(size<2>(AtomShape_MNK{})));
  auto a_tensor = zipped_divide(t_tensor, a_tile);
  auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{}, _);
  auto thr_tile = make_tile(
    _, make_tile(make_layout(size<1>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
  return zipped_divide(tv_tensor, thr_tile);
}

// B-side counterpart of thrfrg_sfa. The structure is the same, but the logical coordinates are
// (N,K), the atom tile uses size<1>(AtomShape_MNK), and thread tiling uses the N/K modes of
// thr_layout_vmnk. SFBLayout is not interchangeable with SFALayout: an m16n8k64 instruction owns
// activation and weight scales in different lane patterns, so sharing the A helper would assign
// weight scales to the wrong output columns.
template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto thrfrg_sfb(SFBTensor &&sfbtensor,
                                           TiledMMA<Atom, TiledThr, TiledPerm> &mma) {
  CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});
  using AtomShape_MNK = typename Atom::Shape_MNK;
  using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;
  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto t_tile = make_tile(get<1>(permutation_mnk), get<2>(permutation_mnk));
  auto t_tensor = logical_divide(sfbtensor, t_tile);
  auto a_tile =
    make_tile(make_layout(size<1>(AtomShape_MNK{})), make_layout(size<2>(AtomShape_MNK{})));
  auto a_tensor = zipped_divide(t_tensor, a_tile);
  auto tv_tensor = a_tensor.compose(AtomLayoutSFB_TV{}, _);
  auto thr_tile = make_tile(
    _, make_tile(make_layout(size<2>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
  return zipped_divide(tv_tensor, thr_tile);
}

// Materialize this thread's SFA register fragment from the abstract thrfrg view.
// thread_mma.thr_vmnk_ is this lane's position in the TiledMMA. For A we keep its V/M/K position
// and drop N: activation scales do not change when the same A row is reused across output columns.
// thr_tensor still points at the original SF storage. Indexing it with thr_vmk selects the source
// values; make_fragment_like then allocates the register fragment with the atom-required layout.
// Per m16n8k64 atom this is one u32 containing four UE4M3 bytes for four consecutive K groups.
// The actual gmem/smem read happens later through G2R or a TiledCopy, not inside this function.
template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_fragment_sfa(SFATensor &&sfatensor, ThrMma &thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  auto thr_tensor = make_tensor(static_cast<SFATensor &&>(sfatensor).data(),
                                thrfrg_sfa(sfatensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
  auto part = thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
  return make_fragment_like<ValTypeSF>(part);
}

// SFB fragment construction mirrors partition_fragment_sfa, except thr_vnk keeps V/N/K and drops
// M. Weight scales depend on output column and K group, but are unchanged when the same B tile is
// reused across output rows. The returned u32 fragment is paired with tCrB by make_zip_tensor in
// the NVFP4 mainloop; no scale values are fetched until the explicit copy is issued.
template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_fragment_sfb(SFBTensor &&sfbtensor, ThrMma &thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  auto thr_tensor = make_tensor(static_cast<SFBTensor &&>(sfbtensor).data(),
                                thrfrg_sfb(sfbtensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
  auto part = thr_tensor(thr_vnk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
  return make_fragment_like<ValTypeSF>(part);
}

// layout_sfa_tv builds the TV (Thread-Value) layout used to construct an SFA TiledCopy.
// It is a compile-time answer to "lane L's value V must read which logical (M,K) scale?".
// It is not the gmem 512 B atom layout, does not allocate a fragment, and performs no copy itself.
//   ref_A: an identity logical (M,K) reference covering the whole TiledMMA.
//   atile: adds atom repetition/broadcast information; the zero-stride N mode means the same SFA
//          values are reused as the TiledMMA moves across output columns.
//   thrfrg_sfa(ref_A,mma): applies the block-scaled atom's SFA mapping to that reference.
//   right_inverse(thr_layout_vmnk): converts hierarchical VMNK coordinates back to a linear lane.
// The two compose calls chain those maps into one TV layout. make_tiled_copy_impl then combines
// it with UniversalCopy<UE4M3> to generate the instructions that fill tCrSFA_view.
template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto layout_sfa_tv(TiledMma &mma) {
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto atile = make_tile(
    _, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                             make_stride(Int<1>{}, Int<0>{})),
                 _));
  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  return thrfrg_sfa(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
}

// layout_sfb_tv is the B-side TV copy plan: (lane,value)->logical (N,K). ref_B covers output
// columns rather than rows, and btile's zero stride is on the M mode because a weight scale is
// reused when the same B tile advances across output rows. The returned layout drives the SFB
// TiledCopy that loads tCrSFB_view. Keeping this separate from layout_sfa_tv is what preserves
// the different activation/weight scale ownership rules of the block-scaled MMA atom.
template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto layout_sfb_tv(TiledMma &mma) {
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto btile = make_tile(
    _, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                             make_stride(Int<0>{}, Int<1>{})),
                 _));
  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  return thrfrg_sfb(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
}

}  // namespace sf_partition

// Convert an accumulator layout from CuTe's hierarchical (MMA,MMA_M,MMA_N) form to this
// thread's (row,column) form. The first MMA mode packs two output columns per register pair;
// logical_divide separates that pair from the M-atom and N-atom repetition modes. This changes
// only the coordinate view: no registers move and no values are converted. Stage 7 uses it to
// apply the correct per-column scale/bias to every accumulator value.
template <typename Layout>
CUTE_DEVICE auto convert_layout_acc_rowcol(Layout acc_layout) {
  auto divided = logical_divide(acc_layout, Shape<_2>{});
  return make_layout(make_layout(get<0, 1>(divided), get<1>(divided)),
                     make_layout(get<0, 0>(divided), get<2>(divided)));
}

// Compile-time tile and MMA selection.
//
// One CTA owns a BM x BN output tile. BM=128 maps to four M atoms and two N atoms per 32-wide
// permutation, i.e. AtomLayoutMNK=(4,2,1) and 8 consumer warps. BK=64 is one block-scaled MMA;
// BK=128 places two independent K atoms in the same smem stage. The default configuration is
// BN=256/BK=128/stages=3, with BN=128/BK=64/stages=4 as the narrower fallback.
//
// NVFP4 uses SM120_16x8x64_TN_VS<E2M1,E2M1,F32,UE4M3,16>. "TN" means A[M,K] and B[N,K] are
// both K-major, so B is logically transposed by the GEMM, not physically transposed in gmem.
// scale_vec::4X consumes four FP8 scales per instruction, covering 4*16=64 K elements.
//
// LoRA uses the SM80-compatible BF16 m16n8k16 tensor-core atom with FP32 accumulation. It keeps
// exactly the same atom layout and M/N permutation as the NVFP4 TiledMMA; only K changes to
// kRankTile. Consequently partition_shape_C is identical and D@L1^T can accumulate directly
// into tCrC without retile, smem reduction, or a second accumulator. Executing both products on
// the same consumer warps is intentional: separate BF16/NVFP4 warpgroups would need an expensive
// cross-warpgroup FP32 reduction and would duplicate accumulator storage.
struct F32DLoraScales {
  float values[8];
};

template <int kBN_ = 128, int kStages_ = 4, int kBK_ = 64, bool kLoRAFirst_ = true,
          bool kF32DLora_ = false>
struct W4A4CuteTraits {
  static constexpr bool kLoRAFirst = kLoRAFirst_;
  static constexpr bool kF32DLora = kF32DLora_;
  static constexpr int kBM = 128;
  static constexpr int kBN = kBN_;
  static constexpr int kBK = kBK_;
  static constexpr int kStages = kStages_;
  // A BF16 rank tile of BM x (BK/4) is byte-identical to a BM x BK E2M1 stage.
  static constexpr int kRankTile = kBK / 4;
  static constexpr int kNumThreads = kBM / 16 * 32;

  static_assert(kBM == 128, "the G2R activation index mapping is specialized for kBM == 128");
  static_assert(kBN % kRowBlock == 0,
                "kBN must be a multiple of 128: the scale-factor smem/TMA atom spans 128 rows");
  static_assert(kBK % 64 == 0, "kBK must contain whole 64-element block-scaled MMA atoms");
  static_assert(kStages >= 2, "kStages >= 2");
  static_assert(kRankTile % 16 == 0, "one rank tile must contain whole BF16 MMA atoms");

  using Element = cutlass::float_e2m1_t;
  using ElementSF = cutlass::float_ue4m3_t;
  using ElementLora = cutlass::bfloat16_t;
  using ElementO = cutlass::bfloat16_t;
  using ElementAcc = float;

  using MmaAtom =
    SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<Element, Element, ElementAcc, ElementSF, kSFVec>;
  using AtomLayoutMNK = Layout<Shape<_4, _2, _1>>;
  // The N permutation ((8,2,2):(1,16,8)) spans 32 output columns and is part of the atom's
  // thread/value mapping. It is a compile-time coordinate recipe, not a runtime N transpose.
  // Reusing this exact permutation for LoRA is what makes the two accumulator layouts compatible.
  using PermutationMNK = Tile<Int<kBM>, Layout<Shape<_8, _2, _2>, Stride<_1, _16, _8>>, _64>;
  using TiledMma = decltype(make_tiled_mma(MmaAtom{}, AtomLayoutMNK{}, PermutationMNK{}));

  // The LoRA-up MMA only differs in K: same atom layout and same M/N permutation, so both
  // tiled MMAs cut the CTA accumulator into identical per-thread fragments.
  using LoraAtom = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
  using LoraPermutationMNK =
    decltype(make_tile(get<0>(PermutationMNK{}), get<1>(PermutationMNK{}), Int<kRankTile>{}));
  using LoraMma = decltype(make_tiled_mma(LoraAtom{}, AtomLayoutMNK{}, LoraPermutationMNK{}));

  static_assert(size(LoraMma{}) == size(TiledMma{}),
                "the LoRA MMA must use the same thread count as the NVFP4 MMA");
  static_assert(
    is_same_v<decltype(partition_shape_C(TiledMma{}, make_shape(Int<kBM>{}, Int<kBN>{}))),
              decltype(partition_shape_C(LoraMma{}, make_shape(Int<kBM>{}, Int<kBN>{})))>,
    "the LoRA MMA accumulator partition must match the NVFP4 MMA one");
  static_assert(kRankTile % size<2>(typename LoraAtom::Shape_MNK{}) == 0,
                "one rank tile must be a whole number of BF16 MMA atoms");

  // SW32 is the smem address-swizzle atom used for K-major FP4 data. TMA writes packed bytes in
  // this order and LDSM later reads the matching register pattern; the swizzle changes addresses,
  // not the logical (M/N,K) values. tile_to_shape repeats the atom over BM/BN, BK and stages.
  // uint4_t is only the 16-byte allocation/copy unit, not a claim that one element has 128 bits.
  using SmemAlloc = uint4_t;
  using SmemLayoutAtomA = UMMA::Layout_K_SW32_Atom<SmemAlloc>;
  using SmemLayoutAtomB = UMMA::Layout_K_SW32_Atom<SmemAlloc>;
  // SmemLayoutA: logical (BM,BK,stages) address map for packed activation A. AtomA is one small
  // repeating SW32 tile; SmemLayoutA is the full multi-stage tensor. It is needed to partition the
  // A fragment and to compute the A-sized storage/overlay. In the current hybrid path no A TMA
  // writes this region: A comes from G2R, so D and O can reuse the allocation.
  using SmemLayoutA =
    decltype(tile_to_shape(SmemLayoutAtomA{}, Shape<Int<kBM>, Int<kBK>, Int<kStages>>{}));
  // SmemLayoutB: logical (BN,BK,stages) map for packed weight B. This is the actual destination
  // of tma.b. LDSM reads sB through this layout into tCrB; SFB has its own non-swizzled SF layout.
  using SmemLayoutB =
    decltype(tile_to_shape(SmemLayoutAtomB{}, Shape<Int<kBN>, Int<kBK>, Int<kStages>>{}));

  // D/L1 use swizzle-free row-major linear layouts and independent TMA descriptors. They share
  // only storage addresses and stage strides with A/B, never a CuTe layout or copy descriptor.
  // The overlay is byte-exact because BF16 needs 2 bytes/element while packed E2M1 needs 0.5:
  //
  //   BM * (BK/4) * 2 = BM * BK / 2  =>  kRankTile = BK/4
  //
  // Main A is consumed through G2R, so the nominally-A allocation is available for D and later
  // output staging. Main B really occupies its allocation, but the empty-stage barrier proves
  // all consumer warps finished it before L1 overwrites that stage. Reusing A's swizzled FP4
  // layout for BF16 would be incorrect even though the byte count matches.
  // SmemLayoutD/L1: canonical K-major 64-byte swizzle used by SM75 LDSM. kRankTile=BK/4 is 32
  // BF16 elements (64B) for BK=128, exactly one SW64 atom row. TMA writes these layouts over the
  // A/B byte allocations; the swizzle only changes bank mapping inside each rank tile.
  using LoraSmemAtom = std::conditional_t<(kRankTile >= 32), UMMA::Layout_K_SW64_Atom<ElementLora>,
                                          UMMA::Layout_K_SW32_Atom<ElementLora>>;
  using SmemLayoutD = decltype(tile_to_shape(
    LoraSmemAtom{}, make_shape(Int<kBM>{}, Int<kRankTile>{}, Int<kStages>{})));
  using SmemLayoutL1 = decltype(tile_to_shape(
    LoraSmemAtom{}, make_shape(Int<kBN>{}, Int<kRankTile>{}, Int<kStages>{})));

  // Append the pipeline-stage mode to each SF atom. filter_zeros removes the logical broadcast
  // modes before computing the stage stride: 16 codes share one stored scale, so advancing to the
  // next stage must skip real allocated bytes, not the 16x larger logical coordinate space.
  // SmemLayoutAtomSFA/SFB describe one SF atom before the stage dimension is appended.
  using SmemLayoutAtomSFA = typename SfSmemAtom<TiledMma, kBM, kBK>::type;
  using SmemLayoutAtomSFB =
    typename SfSmemAtom<TiledMma, (kBN > kRowBlock ? kBN : kRowBlock), kBK>::type;
  // SmemLayoutSFA: logical (BM,BK,stages) activation-SF view, including stride-0 K broadcasts.
  // It supplies fragment shapes and byte sizes, but current SFA values arrive through G2R just
  // like A; no tma.sfa exists. This avoids repacking the native ascales into CUTLASS gmem atoms.
  using SmemLayoutSFA = decltype(make_layout(
    append(shape(SmemLayoutAtomSFA{}), Int<kStages>{}),
    append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))));
  // SmemLayoutSFB: logical (BN,BK,stages) weight-SF view. It is the actual destination of
  // tma.sfb. The appended stage stride uses filter_zeros so stage s+1 starts after the real SF
  // bytes of stage s, not after the broadcast-expanded logical element count.
  using SmemLayoutSFB = decltype(make_layout(
    append(shape(SmemLayoutAtomSFB{}), Int<kStages>{}),
    append(stride(SmemLayoutAtomSFB{}), size(filter_zeros(SmemLayoutAtomSFB{})))));

  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, SmemAlloc>;

  // SmemAtomO is one BF16 SW128 output atom; SmemLayoutO is the full (BM,BN) output tile, with
  // no pipeline-stage dimension. O means final output after FP32->BF16 conversion, not the FP32
  // tCrC accumulator. This region reuses pipeline storage only after every consumer has finished
  // the mainloop and LoRA MMA. SW128 gives STSM and TMA store one shared swizzled address recipe
  // and must not be confused with the FP4 SW32 data layout. Its byte extent is checked below.
  using SmemAtomO = GMMA::Layout_K_SW128_Atom<ElementO>;
  using SmemLayoutO = decltype(tile_to_shape(SmemAtomO{}, Shape<Int<kBM>, Int<kBN>>{}));

  // size(layout) counts logical elements, while cosize(layout) measures the physical address
  // span. They differ for SF because stride-0 broadcasts expose many logical reads of one stored
  // byte. A/B cosize is divided by 2 because each E2M1 code is 4 bits; SF cosize is already bytes
  // because each scale is FP8. A/SFA storage is still reserved even though their main values use
  // G2R: the A region backs D/O overlays, and the SF layouts keep stage sizing/uniformity simple.
  static constexpr int kAStageBytes = cosize_v<SmemLayoutA> / kStages / 2;
  static constexpr int kBStageBytes = cosize_v<SmemLayoutB> / kStages / 2;
  static constexpr int kSFAStageBytes = cosize_v<SmemLayoutSFA> / kStages;
  static constexpr int kSFBStageBytes = cosize_v<SmemLayoutSFB> / kStages;
  static constexpr int kDStageBytes = kBM * kRankTile * static_cast<int>(sizeof(ElementLora));
  static constexpr int kL1StageBytes = kBN * kRankTile * static_cast<int>(sizeof(ElementLora));
  static constexpr int kStageBytes = kAStageBytes + kBStageBytes + kSFAStageBytes + kSFBStageBytes;
  static constexpr int kSmemBytes = kStages * kStageBytes;

  static_assert(kDStageBytes == kAStageBytes, "a D rank tile must overlay one A stage");
  static_assert(kL1StageBytes == kBStageBytes, "an L1 rank tile must overlay one B stage");
  static_assert(kAStageBytes % 1024 == 0 && kBStageBytes % 1024 == 0,
                "the stage stride must preserve the 1024 B smem base alignment");
  static_assert(cosize_v<SmemLayoutO> * static_cast<int>(sizeof(ElementO)) <= kSmemBytes,
                "the BF16 output staging tile must fit into the pipeline storage");
  static_assert(kSmemBytes <= 99 * 1024, "pipeline storage exceeds the SM120 smem budget");

  // TMA transaction sizes feed mbarrier.expect_tx; they count bytes, not CuTe's 4-bit logical
  // elements. Main stages have no A/SFA TMA transaction because those operands use G2R. LoRA
  // stages replace the transaction with exactly D+L1 bytes. Mixing these budgets would leave the
  // full barrier waiting forever or release a stage before its writes complete.
  static constexpr uint32_t kTxBytesMain = kBN * kBK / 2 + kSFBStageBytes;
  static constexpr uint32_t kTxBytesLora = kF32DLora
                                             ? static_cast<uint32_t>(kL1StageBytes)
                                             : static_cast<uint32_t>(kDStageBytes + kL1StageBytes);
};

template <class TmaB_, class TmaSFB_, class TmaD_, class TmaL1_, class TmaC_>
struct W4A4CuteTma {
  TmaB_ b;
  TmaSFB_ sfb;
  TmaD_ d;
  TmaL1_ l1;
  TmaC_ c;
};

struct W4A4CuteCacheKey {
  std::array<uint64_t, 9> data;
  bool operator==(const W4A4CuteCacheKey &) const = default;
};

struct W4A4CuteCacheHash {
  size_t operator()(const W4A4CuteCacheKey &key) const {
    uint64_t hash = 0x9e3779b97f4a7c15ULL;
    for (const uint64_t value : key.data) {
      hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    }
    return static_cast<size_t>(hash);
  }
};

// Warp-specialized multi-stage mainloop. Each CTA computes one output tile; this is not a
// persistent-CTA scheduler. Threads [0,128) form the TMA producer warpgroup and [128,384) eight
// MMA/epilogue consumer warps. Only producer thread 0 issues copies; the other producer threads
// retain the warpgroup role and participate in setmaxnreg. Register budgets are 32 per producer
// thread and 224 per consumer thread, keeping the 384-thread CTA inside the 64K-register SM.
//
// full[s] is a transaction barrier: producer arms the exact B+SFB or D+L1 byte count and TMA
// completion releases consumers. empty[s] is a CTA barrier arrived at by each consumer warp
// leader after its copy+MMA work, preventing the producer from overwriting a live stage.
// phase=(g/kStages)&1 is the mbarrier parity for reuse of the same physical barrier.
//
// Producer and consumers share one global step g over [0,nKt+nRt), so stage index and parity
// never reset at the LoRA boundary. Main g steps TMA B/SFB while consumers fetch A/SFA by G2R;
// trailing g steps TMA D/L1 into the overlay. This lets LoRA loads enter the ring up to kStages
// steps before the main K loop ends instead of exposing a serial LoRA load tail.
// Device-side implementation of one BM x BN SVDQ output tile.
//
// The class deliberately stores no runtime state: all scalar arguments and CuTe views are passed
// explicitly, while Traits supplies the compile-time tile/layout/MMA schedule. Every helper is
// static, __device__ and __forceinline__, so this organization does not introduce a call boundary
// in SASS. The two scheduling modes share the same helpers:
//   kLoRAFirst=true : LoRA -> inverse residual scale -> FP4 mainloop -> residual scale/bias.
//   kLoRAFirst=false: FP4 mainloop -> residual scale/bias -> LoRA.
// TmaParams contains the B/SFB/D/L1/output TMA descriptors; A/SFA use direct global-to-register
// loads and therefore have no TMA descriptor.
template <class Traits, class TmaParams>
class W4A4GemmKernel {
  using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
  using CtaBarrier = cutlass::arch::ClusterBarrier;
  static constexpr int kBM = Traits::kBM;
  static constexpr int kBN = Traits::kBN;
  static constexpr int kBK = Traits::kBK;
  static constexpr int kStages = Traits::kStages;
  static constexpr int kRankTile = Traits::kRankTile;
  static constexpr bool kLoRAFirst = Traits::kLoRAFirst;

 public:
  /**
   * Issue one main-K tile's asynchronous B and SFB TMA copies.
   *
   * Mathematical role: B is the packed FP4 weight tile [BN, BK] and SFB is its FP8 per-group scale
   * tile. The later FP4 MMA consumes dequantized (B, SFB); this function only moves bytes and does
   * not perform arithmetic. A/SFA are intentionally absent because they are loaded by G2R.
   *
   * Synchronization: arrive_and_expect_tx(full[stage], kTxBytesMain) accounts for B+SFB bytes. Both
   * copies target the same transaction barrier. The caller must first wait empty[stage], and the
   * consumer must wait full[stage] before reading the smem destination.
   *
   * @param tma        TMA descriptor bundle; only tma.b and tma.sfb are used.
   * @param full       Transaction-barrier ring; full[stage] is armed by this call.
   * @param kt         Logical K-tile index in units of BK elements.
   * @param stage      Physical smem ring slot, normally g % kStages.
   * @param mB/mSFB    CuTe coordinate tensors for the complete global weight/scale tensors.
   * @param sB/sSFB    Multi-stage smem views; this call writes only (..., stage).
   * @param b_slice/sfb_slice TMA thread slices connecting global source and smem destination.
   * @return           void; completion is asynchronous and observed through full[stage].
   */
  template <class GmemB, class GmemSFB, class SmemB, class SmemSFB, class BSlice, class SFBSlice>
  __device__ __forceinline__ static void tma_load_B_SFB(TmaParams const &tma, uint64_t *full,
                                                        int kt, int stage, GmemB const &mB,
                                                        GmemSFB const &mSFB, SmemB &sB,
                                                        SmemSFB &sSFB, BSlice const &b_slice,
                                                        SFBSlice const &sfb_slice) {
    auto gB = local_tile(mB, Shape<Int<kBN>, Int<kBK>>{}, make_coord(blockIdx.x, kt));
    auto gSFB = local_tile(mSFB, Shape<Int<kBN>, Int<kBK>>{}, make_coord(blockIdx.x, kt));
    TmaBarrier::arrive_and_expect_tx(&full[stage], Traits::kTxBytesMain);
    copy(tma.b.with(full[stage]), b_slice.partition_S(gB), b_slice.partition_D(sB(_, _, stage)));
    copy(tma.sfb.with(full[stage]), sfb_slice.partition_S(gSFB),
         sfb_slice.partition_D(sSFB(_, _, stage)));
  }

  // Issue one BF16 L1 rank tile through TMA. Native FP32 D has no descriptor or smem stage; its
  // consumer-side gather is independent of full[stage], so the transaction covers only L1.
  template <class GmemL1, class SmemL1, class L1Slice>
  __device__ __forceinline__ static void tma_load_L1(TmaParams const &tma, uint64_t *full, int rt,
                                                     int stage, GmemL1 const &mL1, SmemL1 &sL1,
                                                     L1Slice const &l1_slice) {
    auto gL1 = local_tile(mL1, Shape<Int<kBN>, Int<kRankTile>>{}, make_coord(blockIdx.x, rt));
    TmaBarrier::arrive_and_expect_tx(&full[stage], Traits::kTxBytesLora);
    copy(tma.l1.with(full[stage]), l1_slice.partition_S(gL1),
         l1_slice.partition_D(sL1(_, _, stage)));
  }

  /**
   * Issue one rank tile of the BF16 LoRA operands through TMA.
   *
   * Mathematical role: D = (smoothed activation) @ L2^T has shape [BM, rankTile], and L1 is the
   * LoRA-up projection tile [BN, rankTile]. gemm_bf16_lora later computes D @ L1^T. This helper
   * only moves BF16 bytes; it does not apply alpha, wcscales or bias.
   *
   * Layout/lifetime: D reuses the A smem region and L1 reuses the B smem region because those
   * mainloop operands are not live during LoRA. full[stage] tracks D+L1 bytes using kTxBytesLora.
   * As with the main copies, the producer must have observed empty[stage] before calling this
   * function.
   *
   * @param tma        TMA descriptor bundle; only tma.d and tma.l1 are used.
   * @param full       Transaction-barrier ring; full[stage] is armed for this rank tile.
   * @param rt         Logical LoRA rank-tile index in units of kRankTile.
   * @param stage      Physical smem ring slot, normally g % kStages.
   * @param mD/mL1     CuTe coordinate tensors for the full global D/L1 tensors.
   * @param sD/sL1     Multi-stage smem views; only (..., stage) is written.
   * @param d_slice/l1_slice TMA thread slices for the two copies.
   * @return           void; data becomes consumable after full[stage] completes.
   */
  template <class GmemD, class GmemL1, class SmemD, class SmemL1, class DSlice, class L1Slice>
  __device__ __forceinline__ static void tma_load_D_L1(TmaParams const &tma, uint64_t *full, int rt,
                                                       int stage, GmemD const &mD,
                                                       GmemL1 const &mL1, SmemD &sD, SmemL1 &sL1,
                                                       DSlice const &d_slice,
                                                       L1Slice const &l1_slice) {
    auto gD = local_tile(mD, Shape<Int<kBM>, Int<kRankTile>>{}, make_coord(blockIdx.y, rt));
    auto gL1 = local_tile(mL1, Shape<Int<kBN>, Int<kRankTile>>{}, make_coord(blockIdx.x, rt));
    TmaBarrier::arrive_and_expect_tx(&full[stage], Traits::kTxBytesLora);
    copy(tma.d.with(full[stage]), d_slice.partition_S(gD), d_slice.partition_D(sD(_, _, stage)));
    copy(tma.l1.with(full[stage]), l1_slice.partition_S(gL1),
         l1_slice.partition_D(sL1(_, _, stage)));
  }

  /**
   * Load one BK64 sub-block of packed activation and activation scales directly into registers.
   *
   * Mathematical role: supplies A and SFA for the FP4 main product P = dequant(A,SFA) @
   * dequant(B,SFB). Each thread receives the exact A/SFA values required by its MMA fragment.
   *
   * Pack mapping: the producer emits native 256-row panels. `g*(BK/64)+k_block` selects the
   * original 64-element K block. The two CuTe M fragments are mapped to native warp panels with
   *   new_m=(tid%128)/32*16, old_mt=(new_m%32)/16,
   *   old_warp=(m0%256)/32+(new_m%64)/32+2*mt, old_block=m0/256.
   * SFA additionally uses lane XOR (2*old_mt), matching the producer's interleaved scale ownership.
   *
   * Memory/synchronization: A and SFA are direct __ldg operations, with no smem, TMA or mbarrier.
   * This function does not wait for B/SFB and can therefore be scheduled independently of
   * full[stage].
   *
   * @param act/ascales Global native packed activation and activation-scale arrays.
   * @param m0          Base row of this CTA's output tile.
   * @param K           Total reduction length; used to compute the native panel stride.
   * @param tid         Consumer thread index in [0, kNumThreads).
   * @param g           Logical K-tile index, in units of BK.
   * @param k_block     BK64 sub-block within the K tile.
   * @param tCrA        Output register fragment for packed FP4 activation (in/out storage).
   * @param tCrSFA      Output register fragment for four packed FP8 activation scales.
   * @return            void; the two fragments are synchronously overwritten for this sub-block.
   */
  template <class FragA, class FragSFA>
  __device__ __forceinline__ static void load_act_A_g2r(uint4 const *act, uint32_t const *ascales,
                                                        int m0, int K, int tid, int g, int k_block,
                                                        FragA &tCrA, FragSFA &tCrSFA) {
    const int old_block = m0 / 256;
    const int new_m = (tid % 128) / 32 * 16;
    const int old_mt = (new_m % 32) / 16;
    const int lane = tid % 32;
    const int old_kt = g * (kBK / 64) + k_block;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const int old_warp = (m0 % 256) / 32 + (new_m % 64) / 32 + mt * 2;
      const int64_t block_base =
        (static_cast<int64_t>(old_block) * (K / 64) + old_kt) * 8 + old_warp;
      const int64_t index = (block_base * 2 + old_mt) * 32 + lane;
      *reinterpret_cast<uint4 *>(raw_pointer_cast(tCrA(_, mt, k_block).data())) =
        __ldg(act + index);
      const int64_t scale_index = block_base * 32 + (lane ^ (old_mt * 2));
      *reinterpret_cast<uint32_t *>(tCrSFA(_, mt, k_block).data()) = __ldg(ascales + scale_index);
    }
  }

  // Accumulate native FP32 fragment D @ BF16 L1^T. CuTe partition_A supplies the logical
  // coordinate owned by every A-fragment register, so the gather needs no C-to-A shuffle:
  //
  //   native index = (bm*R16 + rt)*4096 + warp*512 + mt*256 + j*32 + lane
  //   row = bm*256 + warp*32 + mt*16 + lane/4 + 8*((j%4)/2)
  //   col = rt*16 + 8*(j/4) + 2*(lane%4) + j%2
  //
  // Inverting those equations for the CuTe-owned logical (row,col) yields the exact native
  // address. Each value is multiplied by its 16-rank scale and rounded to BF16 before the
  // TiledMMA. L1 still uses a CuTe TiledCopy from canonical swizzled smem.
  template <class SmemD, class SmemL1, class Acc>
  __device__ __forceinline__ static void gemm_f32d_lora(uint64_t *full, uint64_t *empty, int tid,
                                                        int nKt, int nRt, int gTotal,
                                                        float const *lora_act,
                                                        F32DLoraScales scales, int m0, int rank,
                                                        SmemD &sD, SmemL1 &sL1, Acc &tCrC) {
    if (nRt == 0) return;
    using ElementLora = typename Traits::ElementLora;
    typename Traits::LoraMma lora_mma;
    auto thr_lora = lora_mma.get_thread_slice(tid);
    auto tCrD = thr_lora.partition_fragment_A(sD(_, _, Int<0>{}));
    auto tCrL1 = thr_lora.partition_fragment_B(sL1(_, _, Int<0>{}));
    auto cD = make_identity_tensor(Shape<Int<kBM>, Int<kRankTile>>{});
    auto tScD = thr_lora.partition_A(cD);
    using LoraCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, ElementLora>;
    auto s2r_copy_l1 = make_tiled_copy_B(LoraCopyAtom{}, lora_mma);
    auto s2r_thr_l1 = s2r_copy_l1.get_thread_slice(tid);
    auto tCsL1 = s2r_thr_l1.partition_S(as_position_independent_swizzle_tensor(sL1));
    auto tCrL1_view = s2r_thr_l1.retile_D(tCrL1);
    const int g_begin = kLoRAFirst ? 0 : nKt;
    const int g_end = kLoRAFirst ? nRt : gTotal;
    const int rank_tiles = rank / 16;
#pragma unroll 1
    for (int g = g_begin; g < g_end; ++g) {
      const int rt = kLoRAFirst ? g : g - nKt;
      const int stage = g % kStages;
      const int phase = (g / kStages) & 1;
#pragma unroll
      for (int i = 0; i < size(tCrD); ++i) {
        const int row = m0 + get<0>(tScD(i));
        const int col = rt * kRankTile + get<1>(tScD(i));
        const int rr = row % 256;
        const int cc = col % 16;
        const int warp = rr / 32;
        const int mt = (rr % 32) / 16;
        const int lane = (rr % 8) * 4 + (cc % 8) / 2;
        const int j = (cc / 8) * 4 + ((rr % 16) / 8) * 2 + cc % 2;
        const int rank16 = col / 16;
        const int64_t index = (static_cast<int64_t>(row / 256) * rank_tiles + rank16) * 4096 +
                              warp * 512 + mt * 256 + j * 32 + lane;
        tCrD(i) = ElementLora(__ldg(lora_act + index) * scales.values[rank16]);
      }
      TmaBarrier::wait(&full[stage], phase);
      copy(s2r_copy_l1, tCsL1(_, _, _, stage), tCrL1_view);
      cute::gemm(lora_mma, tCrD, tCrL1, tCrC);
      __syncwarp();
      if (tid % 32 == 0) CtaBarrier::arrive(&empty[stage]);
    }
  }

  /**
   * Accumulate the BF16 LoRA-up correction into the shared FP32 output accumulator.
   *
   * Math per rank tile rt: tCrC += D_rt @ L1_rt^T, where D_rt is [BM, rankTile] and L1_rt is
   * [BN, rankTile]. Inputs are BF16 and accumulation is FP32. Across all tiles this produces
   * L = D @ proj_up^T. This helper does not apply alpha/wcscales/bias; the caller controls whether
   * LoRA runs before or after the FP4 mainloop.
   *
   * Pipeline: g selects [0,nRt) in LoRA-first mode and [nKt,nKt+nRt) in main-first mode. For each
   * stage it waits full[stage], copies D/L1 from canonical swizzled smem to MMA registers using
   * LDSM, executes BF16 m16n8k16 MMA, then each warp leader arrives empty[stage] to release the
   * overlay to the producer.
   *
   * @param full/empty Shared transaction/CTA barrier rings.
   * @param tid        Consumer thread index; warp leaders perform the empty arrival.
   * @param nKt        Number of main K tiles; fixes the LoRA range in main-first mode.
   * @param nRt        Number of LoRA rank tiles; rank0 returns immediately.
   * @param gTotal     nKt+nRt, the total number of shared-ring steps.
   * @param sD/sL1     Multi-stage smem source views for BF16 D and L1.
   * @param tCrC       Input/output FP32 accumulator; LoRA products are added in place.
   * @return           void; all assigned LoRA rank tiles have been accumulated on return.
   */
  template <class SmemD, class SmemL1, class Acc>
  __device__ __forceinline__ static void gemm_bf16_lora(uint64_t *full, uint64_t *empty, int tid,
                                                        int nKt, int nRt, int gTotal, SmemD &sD,
                                                        SmemL1 &sL1, Acc &tCrC) {
    if (nRt == 0) return;
    using ElementLora = typename Traits::ElementLora;
    typename Traits::LoraMma lora_mma;
    auto thr_lora = lora_mma.get_thread_slice(tid);
    auto tCrD = thr_lora.partition_fragment_A(sD(_, _, Int<0>{}));
    auto tCrL1 = thr_lora.partition_fragment_B(sL1(_, _, Int<0>{}));
    using LoraCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, ElementLora>;
    auto s2r_copy_d = make_tiled_copy_A(LoraCopyAtom{}, lora_mma);
    auto s2r_copy_l1 = make_tiled_copy_B(LoraCopyAtom{}, lora_mma);
    auto s2r_thr_d = s2r_copy_d.get_thread_slice(tid);
    auto s2r_thr_l1 = s2r_copy_l1.get_thread_slice(tid);
    auto tCsD = s2r_thr_d.partition_S(as_position_independent_swizzle_tensor(sD));
    auto tCsL1 = s2r_thr_l1.partition_S(as_position_independent_swizzle_tensor(sL1));
    auto tCrD_view = s2r_thr_d.retile_D(tCrD);
    auto tCrL1_view = s2r_thr_l1.retile_D(tCrL1);
    const int g_begin = kLoRAFirst ? 0 : nKt;
    const int g_end = kLoRAFirst ? nRt : gTotal;
#pragma unroll 1
    for (int g = g_begin; g < g_end; ++g) {
      const int stage = g % kStages;
      const int phase = (g / kStages) & 1;
      TmaBarrier::wait(&full[stage], phase);
      copy(s2r_copy_d, tCsD(_, _, _, stage), tCrD_view);
      copy(s2r_copy_l1, tCsL1(_, _, _, stage), tCrL1_view);
      cute::gemm(lora_mma, tCrD, tCrL1, tCrC);
      __syncwarp();
      if (tid % 32 == 0) CtaBarrier::arrive(&empty[stage]);
    }
  }

  /**
   * Algebraically compensate the LoRA accumulator for LoRA-first scheduling.
   *
   * Let L = D @ proj_up^T and s[n] = alpha * wcscales[n]. Before the main FP4 product P is added,
   * this function changes tCrC from L to L/s. After gemm_fp4_w4a4 the accumulator is P + L/s, so
   * the final residual scaling by s produces s*P + L, exactly the desired residual+LoRA result.
   * Bias is not involved here and is added only by apply_residual_scale.
   *
   * Numeric/layout details: all arithmetic is FP32. tScC_rc maps each accumulator fragment value to
   * its logical output column; only the few columns owned by the thread are loaded. n0 is the CTA's
   * global column offset. A null wcscales pointer means scale 1. rank0 returns without touching C.
   *
   * @param wcscales   Optional per-output-column residual scales [N].
   * @param alpha      Scalar residual alpha from the packed weight normalization.
   * @param n0         Global output-column offset of this CTA tile.
   * @param nRt        Number of LoRA rank tiles; zero disables the compensation.
   * @param tScC_rc    Row/column coordinate view of the accumulator partition.
   * @param tCrC       Input/output FP32 accumulator containing the LoRA term before main FP4.
   * @return           void; tCrC is divided by alpha*wcscales for each owned column.
   */
  template <class CoordC, class Acc>
  __device__ __forceinline__ static void apply_inverse_scale(cutlass::bfloat16_t const *wcscales,
                                                             float alpha, int n0, int nRt,
                                                             CoordC const &tScC_rc, Acc &tCrC) {
    if (nRt == 0) return;
    constexpr int kCRows = decltype(size<0>(tScC_rc))::value;
    constexpr int kCCols = decltype(size<1>(tScC_rc))::value;
    float inv_scale[kCCols];
#pragma unroll
    for (int c = 0; c < kCCols; ++c) {
      const int n = n0 + get<1>(tScC_rc(0, c));
      const float scale = alpha * (wcscales != nullptr ? static_cast<float>(wcscales[n]) : 1.0f);
      inv_scale[c] = 1.0f / scale;
    }
    auto tCrC_rc = make_tensor(tCrC.data(), convert_layout_acc_rowcol(tCrC.layout()));
#pragma unroll
    for (int r = 0; r < kCRows; ++r)
#pragma unroll
      for (int c = 0; c < kCCols; ++c) tCrC_rc(r, c) *= inv_scale[c];
  }

  /**
   * Execute the complete FP4 W4A4 main reduction and accumulate it into tCrC.
   *
   * Math: for each K tile, compute P += dequant(A,SFA) @ dequant(B,SFB), with BF16 tensor-core
   * operands generated by the SM120 FP4 MMA and FP32 accumulation. The final caller applies
   * residual alpha/wcscales and LoRA separately; this helper performs no epilogue scaling.
   *
   * Scheduling: in LoRA-first mode g=nRt+kt; in main-first mode g=kt. Each stage waits full[stage]
   * once, then interleaves BK64 sub-block A/SFA G2R,
   * B/SFB smem-to-register copies and MMA. In LoRA-first mode K tile 0 skips A/SFA loads because
   * the caller preloaded those fragments before the LoRA GEMM. After all sub-blocks, each warp
   * leader arrives empty[stage] so the producer can reuse the B/SFB smem slot.
   *
   * @param full/empty Shared transaction/CTA barrier rings.
   * @param tid        Consumer thread index; also determines A/SFA fragment ownership.
   * @param nKt/nRt Main K tiles and LoRA rank tiles.
   * @param act/ascales Global native activation and activation-scale packs used by load_act_A_g2r.
   * @param m0/K       CTA row offset and total reduction length.
   * @param tiled_mma  SM120 FP4 tiled MMA descriptor.
   * @param s2r_copy_b LDSM copy descriptor for B fragments.
   * @param tCsB/tCsSFB Swizzle-free smem source views for B/SFB.
   * @param tCrB_view/tCrSFB_view Destination register views produced by the copy retiling.
   * @param tCrA/tCrSFA Input/output A/SFA register fragments.
   * @param tCrC       Input/output FP32 main-product accumulator.
   * @return           void; all K tiles in the selected mode have been accumulated on return.
   */
  template <class TiledMma, class CopyB, class SrcB, class DstB, class SrcSFB, class DstSFB,
            class FragB, class FragSFB, class FragA, class FragSFA, class Acc>
  __device__ __forceinline__ static void gemm_fp4_w4a4(
    uint64_t *full, uint64_t *empty, int tid, int nKt, int nRt, uint4 const *act,
    uint32_t const *ascales, int m0, int K, TiledMma const &tiled_mma, CopyB const &s2r_copy_b,
    SrcB const &tCsB, DstB &tCrB_view, SrcSFB const &tCsSFB, DstSFB &tCrSFB_view, FragB &tCrB,
    FragSFB &tCrSFB, FragA &tCrA, FragSFA &tCrSFA, Acc &tCrC) {
    const int g_offset = kLoRAFirst ? nRt : 0;
#pragma unroll 1
    for (int kt = 0; kt < nKt; ++kt) {
      const int g = g_offset + kt;
      const int stage = g % kStages;
      const int phase = (g / kStages) & 1;
      TmaBarrier::wait(&full[stage], phase);
#pragma unroll
      for (int k_block = 0; k_block < kBK / 64; ++k_block) {
        if (!kLoRAFirst || kt != 0) {
          load_act_A_g2r(act, ascales, m0, K, tid, kt, k_block, tCrA, tCrSFA);
        }
        copy(s2r_copy_b, tCsB(_, _, k_block, stage), tCrB_view(_, _, k_block));
        copy(tCsSFB(_, _, k_block, stage), tCrSFB_view(_, _, k_block));
        cute::gemm(tiled_mma, make_zip_tensor(tCrA(_, _, k_block), tCrSFA(_, _, k_block)),
                   make_zip_tensor(tCrB(_, _, k_block), tCrSFB(_, _, k_block)), tCrC);
      }
      __syncwarp();
      if (tid % 32 == 0) CtaBarrier::arrive(&empty[stage]);
    }
  }

  /**
   * Apply residual alpha, per-column weight scales and bias to the FP32 accumulator.
   *
   * Math for an output column n: tCrC = fmaf(tCrC, alpha*wcscales[n], bias[n]). A null wcscales
   * means 1 and a null bias means 0. In main-first mode tCrC initially contains P, producing
   * alpha*wcscales*P+bias before LoRA is added. In LoRA-first mode tCrC contains
   * P+L/(alpha*wcscales), so this same operation yields alpha*wcscales*P+L+bias.
   *
   * Numeric/layout details: this is the final FP32 epilogue arithmetic before the single BF16
   * conversion. tScC_rc maps each register value to a logical column, so each thread loads only its
   * owned scales/biases once and reuses them across all owned rows. The alpha multiplication
   * remains a separate loop to preserve the established rounding behavior.
   *
   * @param bias       Optional BF16 output bias [N].
   * @param wcscales   Optional BF16 per-output-column residual scales [N].
   * @param alpha      Scalar residual alpha from the packed weight normalization.
   * @param n0         Global output-column offset of this CTA tile.
   * @param tScC_rc    Row/column coordinate view of the accumulator partition.
   * @param tCrC       Input/output FP32 accumulator to be scaled and biased in place.
   * @return           void; tCrC contains the fully scaled residual/LoRA value on return.
   */
  template <class CoordC, class Acc>
  __device__ __forceinline__ static void apply_residual_scale(cutlass::bfloat16_t const *bias,
                                                              cutlass::bfloat16_t const *wcscales,
                                                              float alpha, int n0,
                                                              CoordC const &tScC_rc, Acc &tCrC) {
    constexpr int kCRows = decltype(size<0>(tScC_rc))::value;
    constexpr int kCCols = decltype(size<1>(tScC_rc))::value;
    float col_scale[kCCols];
    float col_bias[kCCols];
#pragma unroll
    for (int c = 0; c < kCCols; ++c) {
      const int n = n0 + get<1>(tScC_rc(0, c));
      col_scale[c] = wcscales != nullptr ? static_cast<float>(wcscales[n]) : 1.0f;
      col_bias[c] = bias != nullptr ? static_cast<float>(bias[n]) : 0.0f;
    }
    auto tCrC_rc = make_tensor(tCrC.data(), convert_layout_acc_rowcol(tCrC.layout()));
    if (alpha != 1.0f) {
#pragma unroll
      for (int r = 0; r < kCRows; ++r)
#pragma unroll
        for (int c = 0; c < kCCols; ++c) tCrC_rc(r, c) *= alpha;
    }
#pragma unroll
    for (int r = 0; r < kCRows; ++r)
#pragma unroll
      for (int c = 0; c < kCCols; ++c)
        tCrC_rc(r, c) = fmaf(tCrC_rc(r, c), col_scale[c], col_bias[c]);
  }

  /**
   * Convert the final FP32 accumulator once and publish the BM x BN output tile with TMA.
   *
   * Mathematical role: tCrC already contains the fully accumulated residual plus LoRA result. This
   * helper performs the sole FP32->BF16 conversion; it does not change scaling or accumulation
   * order.
   *
   * Memory protocol: STSM writes the register fragment into the swizzled SmemLayoutO overlay. The
   * first __syncthreads proves that no mainloop stage still uses that smem. fence_view_async_shared
   * makes generic smem stores visible to the TMA engine; the second __syncthreads proves all
   * consumer STSM operations are complete before thread 0 issues the TMA store. tma_store_wait<0>
   * drains the store before the CTA can exit or reuse the output overlay.
   *
   * @param tma        TMA descriptor bundle; only tma.c is used for the global output store.
   * @param mC         CuTe coordinate tensor for the complete global output tensor [M,N].
   * @param c_slice    TMA output thread slice connecting smem and global coordinates.
   * @param shm        Base of dynamic shared memory, overlaid with SmemLayoutO for this epilogue.
   * @param tid        Consumer thread index; only tid==0 issues the TMA store.
   * @param tiled_mma  MMA layout used to retile the register fragment for STSM.
   * @param tCrC       Final FP32 accumulator fragment, read-only.
   * @return           void; the output tile is visible in global memory after the drain.
   */
  template <class GmemC, class CSlice, class TiledMma, class Acc>
  __device__ __forceinline__ static void tma_store_C(TmaParams const &tma, GmemC const &mC,
                                                     CSlice const &c_slice, void *shm, int tid,
                                                     TiledMma const &tiled_mma, Acc const &tCrC) {
    using ElementO = typename Traits::ElementO;
    using SmemLayoutO = typename Traits::SmemLayoutO;
    constexpr int kFrag = decltype(size(tCrC))::value;
    cutlass::Array<ElementO, kFrag> acc_bf16;
    cutlass::NumericArrayConverter<ElementO, float, kFrag> convert;
    acc_bf16 = convert(*reinterpret_cast<cutlass::Array<float, kFrag> const *>(tCrC.data()));
    auto tCrCh = make_tensor(make_rmem_ptr(acc_bf16.data()), tCrC.layout());

    __syncthreads();
    auto r2s_copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, ElementO>{}, tiled_mma);
    auto r2s_thr = r2s_copy.get_thread_slice(tid);
    auto sC = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(shm)), SmemLayoutO{});
    auto tCrCh_src = r2s_thr.retile_S(tCrCh);
    auto tCsC_dst = r2s_thr.partition_D(sC);
    copy(r2s_copy, tCrCh_src, tCsC_dst);
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    auto gC = local_tile(mC, Shape<Int<kBM>, Int<kBN>>{}, make_coord(blockIdx.y, blockIdx.x));
    if (tid == 0) copy(tma.c, c_slice.partition_S(sC), c_slice.partition_D(gC));
    cute::tma_store_arrive();
    cute::tma_store_wait<0>();
  }
};

template <class Traits, class TmaParams>
__global__ void __launch_bounds__(Traits::kNumThreads + 128, 1)
  gemm_w4a4_kernel(CUTLASS_GRID_CONSTANT TmaParams const tma, uint4 const *__restrict__ act,
                   uint32_t const *__restrict__ ascales, float const *__restrict__ lora_act_f32,
                   F32DLoraScales scales, cutlass::bfloat16_t const *__restrict__ bias,
                   cutlass::bfloat16_t const *__restrict__ wcscales, float alpha, int M, int N,
                   int K, int rank) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  using cute::tma_store_arrive;
  using cute::tma_store_wait;
  using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
  using CtaBarrier = cutlass::arch::ClusterBarrier;

  using Element = typename Traits::Element;
  using ElementSF = typename Traits::ElementSF;
  using ElementLora = typename Traits::ElementLora;
  using ElementO = typename Traits::ElementO;
  using TiledMma = typename Traits::TiledMma;
  using LoraMma = typename Traits::LoraMma;
  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutD = typename Traits::SmemLayoutD;
  using SmemLayoutL1 = typename Traits::SmemLayoutL1;
  using SmemLayoutSFA = typename Traits::SmemLayoutSFA;
  using SmemLayoutSFB = typename Traits::SmemLayoutSFB;
  using SmemLayoutAtomSFA = typename Traits::SmemLayoutAtomSFA;
  using SmemLayoutAtomSFB = typename Traits::SmemLayoutAtomSFB;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using SmemCopyAtom = typename Traits::SmemCopyAtom;
  using Gemm = W4A4GemmKernel<Traits, TmaParams>;

  constexpr int kBM = Traits::kBM;
  constexpr int kBN = Traits::kBN;
  constexpr int kBK = Traits::kBK;
  constexpr int kStages = Traits::kStages;
  constexpr int kRankTile = Traits::kRankTile;
  constexpr bool kLoRAFirst = Traits::kLoRAFirst;
  constexpr bool kF32DLora = Traits::kF32DLora;
  constexpr int kNumThreads = Traits::kNumThreads;

  const int raw_tid = threadIdx.x;
  const bool is_producer = raw_tid < 128;
  const int tid = is_producer ? 0 : raw_tid - 128;
  const int m0 = blockIdx.y * kBM;
  const int n0 = blockIdx.x * kBN;

  // -----------------------------------------------------------------------------------------------
  // Stage 1: establish this CTA's logical problem and gmem views.
  // -----------------------------------------------------------------------------------------------
  // blockIdx selects the output tile (m0,n0), not an input element. nKt counts NVFP4 K stages and
  // nRt counts LoRA rank stages. They share g so that the same ring can carry two different data
  // types. rankTma is only a descriptor extent; rank 0 uses a harmless placeholder and nRt=0 makes
  // every D/L1 access unreachable. get_tma_tensor converts a descriptor into logical coordinates;
  // it does not copy anything at this point.
  const int nKt = K / kBK;
  const int nRt = rank / kRankTile;
  const int gTotal = nKt + nRt;
  // The rank-0 path never dispatches a rank tile, so a placeholder rank extent keeps the
  // unused D/L1 descriptors well-formed.
  const int rankTma = rank > 0 ? rank : kRankTile;

  auto mB = tma.b.get_tma_tensor(make_shape(N, K));
  auto mC = tma.c.get_tma_tensor(make_shape(M, N));
  auto mSFB = tma.sfb.get_tma_tensor(shape(sf_gmem_layout(N, K)));
  auto mD = tma.d.get_tma_tensor(make_shape(M, rankTma));
  auto mL1 = tma.l1.get_tma_tensor(make_shape(N, rankTma));
  auto b_slice = tma.b.get_slice(_0{});
  auto c_slice = tma.c.get_slice(_0{});
  auto sfb_slice = tma.sfb.get_slice(_0{});
  auto d_slice = tma.d.get_slice(_0{});
  auto l1_slice = tma.l1.get_slice(_0{});

  // -----------------------------------------------------------------------------------------------
  // Stage 2: allocate stage storage and initialize its synchronization objects.
  // -----------------------------------------------------------------------------------------------
  // The byte pointer is intentional: E2M1 is sub-byte, while D/output are BF16. Each region's
  // base is therefore computed in bytes first, then recast into the CuTe element view required by
  // that region. sA/sB/sD/sL1 are different layout views over reusable storage, not four separate
  // allocations. full/empty are real shared-memory barriers, so all threads observe their
  // initialized state after __syncthreads. empty expects one arrival per consumer warp, not 256
  // arrivals, which is why only warp leaders arrive later.
  extern __shared__ __align__(1024) uint8_t shm[];
  uint8_t *a_bytes = shm;
  uint8_t *b_bytes = a_bytes + Traits::kAStageBytes * kStages;
  ElementSF *sfa_base = reinterpret_cast<ElementSF *>(b_bytes + Traits::kBStageBytes * kStages);
  ElementSF *sfb_base = sfa_base + Traits::kSFAStageBytes * kStages;

  __shared__ uint64_t full[kStages];
  __shared__ uint64_t empty[kStages];
  if (raw_tid == 0) {
    for (int s = 0; s < kStages; ++s) {
      TmaBarrier::init(&full[s], 1);
      CtaBarrier::init(&empty[s], kNumThreads / 32);
    }
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();

  auto sA_full = make_tensor(make_smem_ptr(recast_ptr<uint4_t>(a_bytes)), SmemLayoutA{});
  auto sB_full = make_tensor(make_smem_ptr(recast_ptr<uint4_t>(b_bytes)), SmemLayoutB{});
  auto sD_full = make_tensor(make_smem_ptr(recast_ptr<ElementLora>(a_bytes)), SmemLayoutD{});
  auto sL1_full = make_tensor(make_smem_ptr(recast_ptr<ElementLora>(b_bytes)), SmemLayoutL1{});
  auto sSFA_full = make_tensor(make_smem_ptr(sfa_base), SmemLayoutSFA{});
  auto sSFB_full = make_tensor(make_smem_ptr(sfb_base), SmemLayoutSFB{});

  // -----------------------------------------------------------------------------------------------
  // Stage 3: build register fragments and the copy views that will fill them.
  // -----------------------------------------------------------------------------------------------
  // thr_mma converts a CTA-wide tile into this thread's tiny A/B/C fragments. partition_fragment_*
  // allocates register storage with the exact layout expected by the hardware atom. A TiledCopy
  // has a different iteration order, so retile_D presents that same register storage in the copy's
  // order; it does not allocate a second fragment or reorder values at runtime.
  // tCrC is the only accumulator for both GEMMs. cC is an identity-coordinate tensor: it stores
  // logical (m,n) coordinates rather than output values, allowing the epilogue to find the correct
  // bias/wcscale for every FP32 accumulator element.
  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto s2r_copy_a = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto s2r_copy_b = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto s2r_thr_a = s2r_copy_a.get_thread_slice(tid);
  auto s2r_thr_b = s2r_copy_b.get_thread_slice(tid);
  auto tCsA = s2r_thr_a.partition_S(as_position_independent_swizzle_tensor(sA_full));
  auto tCsB = s2r_thr_b.partition_S(as_position_independent_swizzle_tensor(sB_full));
  auto tCrA = thr_mma.partition_fragment_A(sA_full(_, _, Int<0>{}));
  auto tCrB = thr_mma.partition_fragment_B(sB_full(_, _, Int<0>{}));
  auto tCrA_view = s2r_thr_a.retile_D(tCrA);
  auto tCrB_view = s2r_thr_b.retile_D(tCrB);

  using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;
  auto tCrSFA = sf_partition::partition_fragment_sfa(sSFA_full(_, _, Int<0>{}), thr_mma);
  auto tCrSFB = sf_partition::partition_fragment_sfb(sSFB_full(_, _, Int<0>{}), thr_mma);
  auto copy_sfa = make_tiled_copy_impl(
    SmemCopyAtomSF{}, sf_partition::layout_sfa_tv(tiled_mma),
    make_shape(size<0>(tile_shape(tiled_mma)), size<2>(tile_shape(tiled_mma))));
  auto copy_sfb = make_tiled_copy_impl(
    SmemCopyAtomSF{}, sf_partition::layout_sfb_tv(tiled_mma),
    make_shape(size<1>(tile_shape(tiled_mma)), size<2>(tile_shape(tiled_mma))));
  auto thr_copy_sfa = copy_sfa.get_thread_slice(tid);
  auto thr_copy_sfb = copy_sfb.get_thread_slice(tid);
  auto tCsSFA = thr_copy_sfa.partition_S(as_position_independent_swizzle_tensor(sSFA_full));
  auto tCsSFB = thr_copy_sfb.partition_S(as_position_independent_swizzle_tensor(sSFB_full));
  auto tCrSFA_view = thr_copy_sfa.retile_D(tCrSFA);
  auto tCrSFB_view = thr_copy_sfb.retile_D(tCrSFB);

  auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<kBM>, Int<kBN>>{});
  clear(tCrC);
  auto cC = make_identity_tensor(Shape<Int<kBM>, Int<kBN>>{});
  auto tScC = thr_mma.partition_C(cC);
  auto tScC_rc = make_tensor(tScC.data(), convert_layout_acc_rowcol(tScC.layout()));
  constexpr int kCRows = decltype(size<0>(tScC_rc))::value;
  constexpr int kCCols = decltype(size<1>(tScC_rc))::value;

  // -----------------------------------------------------------------------------------------------
  // Stage 4: define one producer step, without executing it yet.
  // -----------------------------------------------------------------------------------------------
  // local_tile selects the gmem tile for this CTA and K/rank step. partition_S(gmem) and
  // partition_D(smem) then describe the two ends of the TMA copy. arrive_and_expect_tx says how
  // many bytes must arrive before full[stage] opens; each copy(...with(full)) contributes to that
  // same transaction barrier. The main branch transfers B/SFB only, while the LoRA branch uses
  // different descriptors to put D/L1 into the overlay. No A/SFA copy is hidden in these TMA
  // helpers.
  // -----------------------------------------------------------------------------------------------
  // Stage 5: split producer and consumer roles.
  // -----------------------------------------------------------------------------------------------
  // Producer waits until a stage is empty, then performs Stage 4 and immediately moves to the next
  // g. Consumers initially mark every stage empty so the producer can start. setmaxnreg changes
  // the per-warpgroup register budget after launch: the TMA side needs few registers, while the
  // consumer needs space for two operand types and a large FP32 accumulator. This is why WS is
  // useful here even though only one producer thread actually issues TMA instructions.
  if (is_producer) {
    cutlass::arch::warpgroup_reg_dealloc<32>();
    if (raw_tid == 0) {
      for (int g = 0; g < gTotal; ++g) {
        const int stage = g % kStages;
        const int phase = (g / kStages) & 1;
        CtaBarrier::wait(&empty[stage], phase);
        const bool lora_stage = kLoRAFirst ? g < nRt : g >= nKt;
        if (lora_stage) {
          const int rt = kLoRAFirst ? g : g - nKt;
          if constexpr (kF32DLora) {
            Gemm::tma_load_L1(tma, full, rt, stage, mL1, sL1_full, l1_slice);
          } else {
            Gemm::tma_load_D_L1(tma, full, rt, stage, mD, mL1, sD_full, sL1_full, d_slice,
                                l1_slice);
          }
        } else {
          const int kt = kLoRAFirst ? g - nRt : g;
          Gemm::tma_load_B_SFB(tma, full, kt, stage, mB, mSFB, sB_full, sSFB_full, b_slice,
                               sfb_slice);
        }
      }
    }
    return;
  }

  cutlass::arch::warpgroup_reg_alloc<224>();
  if (tid % 32 == 0) {
    for (int s = 0; s < kStages; ++s) CtaBarrier::arrive(&empty[s]);
  }

  // Translate the native activation producer's fragment order directly into CuTe A/SFA registers.
  // No TMA or smem tensor is involved on this path. A BK=128 stage calls the helper twice;
  // old_kt=g*(BK/64)+k_block selects the original 64-element producer block.
  //
  // CuTe's two M fragments refer to m and m+64, whereas native stores eight 32-row warp panels in
  // each 256-row block. The inverse mapping is
  //   new_m    = (tid % 128)/32 * 16
  //   old_mt   = (new_m % 32)/16
  //   old_warp = (m0 % 256)/32 + (new_m % 64)/32 + 2*fragment_index
  //   old_block = m0/256.
  // block_base includes (old_block,K-tile,old_warp), and the final index adds old_mt and lane.
  // Each A load writes one uint4 directly into tCrA; each scale load writes four FP8 bytes as a
  // u32 into tCrSFA. The second m-tile needs lane XOR 2 because native interleaves scale ownership
  // between its two 16-row m-tiles. Omitting that swap preserves shapes but produces wrong math.
  // -----------------------------------------------------------------------------------------------
  // Stage 6: accumulate the BF16 LoRA-up correction first.
  // -----------------------------------------------------------------------------------------------
  // D/L1 already have row-major TMA-compatible layouts. The BF16 m16n8k16 atoms share the exact
  // C partition of the NVFP4 MMA. LoRA runs first so the following A/SFA prefetch can overlap it.
  // The residual scale is applied to the combined accumulator later, so divide the LoRA term by
  // alpha*wcscales now; the final epilogue multiplication cancels it exactly.
  if constexpr (kLoRAFirst) {
#pragma unroll
    for (int k_block = 0; k_block < kBK / 64; ++k_block) {
      Gemm::load_act_A_g2r(act, ascales, m0, K, tid, 0, k_block, tCrA, tCrSFA);
    }
  }

  // -----------------------------------------------------------------------------------------------
  // Stage 7: consume all NVFP4 K stages and accumulate P in FP32.
  // -----------------------------------------------------------------------------------------------
  // Waiting on full makes the B/SFB bytes visible; the transaction barrier already orders the
  // subsequent smem reads. A/SFA are filled independently by load_act_A_g2r. Interleave each
  // k-block's G2R activation load, B/SFB smem copy and MMA so the BK128 stage does not keep both
  // activation fragments and both B fragment slots live at once. The K accumulation order remains
  // k-block 0 then 1, matching the previous single BK128 cute::gemm.
  // -----------------------------------------------------------------------------------------------
  // Stage 8: apply residual-only scale and bias in FP32.
  // -----------------------------------------------------------------------------------------------
  // The C fragment is still indexed by the MMA's thread/value layout, so convert_layout_acc_rowcol
  // gives each value a logical (row,column) view. One thread owns only a few output columns; load
  // those column scales/biases once and reuse them across its rows. alpha is applied first, then
  // fmaf(P,wcscales,bias). We deliberately do this before the LoRA MMA so wcscales cannot leak
  // into D@L1^T. Everything remains FP32; converting here would add an unnecessary rounding.
  if constexpr (kLoRAFirst) {
    if constexpr (kF32DLora) {
      Gemm::gemm_f32d_lora(full, empty, tid, nKt, nRt, gTotal, lora_act_f32, scales, m0, rank,
                           sD_full, sL1_full, tCrC);
    } else {
      Gemm::gemm_bf16_lora(full, empty, tid, nKt, nRt, gTotal, sD_full, sL1_full, tCrC);
    }
    Gemm::apply_inverse_scale(wcscales, alpha, n0, nRt, tScC_rc, tCrC);
    Gemm::gemm_fp4_w4a4(full, empty, tid, nKt, nRt, act, ascales, m0, K, tiled_mma, s2r_copy_b,
                        tCsB, tCrB_view, tCsSFB, tCrSFB_view, tCrB, tCrSFB, tCrA, tCrSFA, tCrC);
    Gemm::apply_residual_scale(bias, wcscales, alpha, n0, tScC_rc, tCrC);
  } else {
    Gemm::gemm_fp4_w4a4(full, empty, tid, nKt, nRt, act, ascales, m0, K, tiled_mma, s2r_copy_b,
                        tCsB, tCrB_view, tCsSFB, tCrSFB_view, tCrB, tCrSFB, tCrA, tCrSFA, tCrC);
    Gemm::apply_residual_scale(bias, wcscales, alpha, n0, tScC_rc, tCrC);
    if constexpr (kF32DLora) {
      Gemm::gemm_f32d_lora(full, empty, tid, nKt, nRt, gTotal, lora_act_f32, scales, m0, rank,
                           sD_full, sL1_full, tCrC);
    } else {
      Gemm::gemm_bf16_lora(full, empty, tid, nKt, nRt, gTotal, sD_full, sL1_full, tCrC);
    }
  }

  // -----------------------------------------------------------------------------------------------
  // Stage 9: round once and publish the output tile through smem + TMA store.
  // -----------------------------------------------------------------------------------------------
  // NumericArrayConverter performs the sole FP32->BF16 conversion, after both GEMMs and all
  // scales. STSM then writes the register fragment into a swizzled smem output layout; the first
  // barrier proves no mainloop stage is still using that storage. fence_view_async_shared makes
  // generic smem stores visible to the asynchronous TMA engine, and the second barrier ensures
  // every consumer finished its STSM before thread 0 starts the store. tma_store_wait<0> drains
  // outstanding stores before this CTA can reuse the output smem or exit.
  Gemm::tma_store_C(tma, mC, c_slice, shm, tid, tiled_mma, tCrC);
#else
  asm volatile("trap;");
#endif  // __CUDA_ARCH__ >= 1200
}

// Host launch has three jobs:
//   A. Describe each gmem tensor and its matching smem stage to create TMA descriptors. A TMA
//      descriptor is a reusable copy recipe (address, shape, stride, box and smem layout), not
//      the copied data itself. D/L1 descriptors intentionally differ from A/B descriptors.
//   B. Cache those recipes per host thread and per exact pointer/shape/rank key. This avoids
//      rebuilding them on every linear-layer call while remaining safe when output or D pointers
//      change. The cache holds at most 32 entries and is cleared wholesale when full; it is not
//      an LRU cache. Device-specific dynamic-smem opt-in is likewise performed only once.
//   C. Launch one CTA per BM x BN output tile with 128 extra producer threads.
// The template is instantiated only from gemm_w4a4_cute.cu, keeping SM120 atoms out of other TUs.
template <int kBN = 128, int kStages = 4, int kBK_ = 64, bool kLoRAFirst = true,
          bool kF32DLora = false>
void launch_gemm_w4a4(void const *act, void const *wgt, void *out,
                      cutlass::float_ue4m3_t const *act_sf, cutlass::float_ue4m3_t const *wgt_sf,
                      cutlass::bfloat16_t const *lora_act_in, float const *lora_act_f32,
                      cutlass::bfloat16_t const *lora_up, cutlass::bfloat16_t const *bias,
                      cutlass::bfloat16_t const *wcscales, F32DLoraScales scales, float alpha,
                      int M, int N, int K, int rank, cudaStream_t stream) {
  using Traits = W4A4CuteTraits<kBN, kStages, kBK_, kLoRAFirst, kF32DLora>;
  using Element = typename Traits::Element;
  using ElementSF = typename Traits::ElementSF;
  using ElementLora = typename Traits::ElementLora;
  using ElementO = typename Traits::ElementO;
  constexpr int kBM = Traits::kBM;
  constexpr int kBK = Traits::kBK;
  constexpr int kRankTile = Traits::kRankTile;

  auto mB = make_tensor(recast_ptr<Element>(static_cast<uint8_t const *>(wgt)), make_shape(N, K),
                        make_stride(K, _1{}));
  auto mC = make_tensor(make_gmem_ptr(static_cast<ElementO *>(out)), make_shape(M, N),
                        make_stride(N, _1{}));
  auto mSFB = make_tensor(recast_ptr<ElementSF>(wgt_sf), sf_gmem_layout(N, K));

  // With rank 0 the rank tiles are never dispatched (nRt == 0), so the D/L1 descriptors are
  // built over an unused [M,kRankTile] / [N,kRankTile] view of the output and stay unreachable.
  // The placeholder merely satisfies descriptor construction; it is never read by a TMA.
  const int rankTma = rank > 0 ? rank : kRankTile;
  // kF32DLora never issues a D TMA, so the descriptor is built over the output placeholder.
  auto const *d_ptr =
    (!kF32DLora && rank > 0) ? lora_act_in : reinterpret_cast<ElementLora const *>(out);
  auto const *l1_ptr = rank > 0 ? lora_up : reinterpret_cast<ElementLora const *>(out);
  auto mD = make_tensor(make_gmem_ptr(d_ptr), make_shape(M, rankTma), make_stride(rankTma, _1{}));
  auto mL1 = make_tensor(make_gmem_ptr(l1_ptr), make_shape(N, rankTma), make_stride(rankTma, _1{}));

  auto build_params = [&]() {
    auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, mB, typename Traits::SmemLayoutB{}(_, _, Int<0>{}),
                               Shape<Int<kBN>, Int<kBK>>{}, _1{});
    auto tma_sfb =
      make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFB, typename Traits::SmemLayoutAtomSFB{},
                              Shape<Int<kBN>, Int<kBK>>{}, _1{});
    auto tma_d = make_tma_copy(SM90_TMA_LOAD{}, mD, typename Traits::SmemLayoutD{}(_, _, Int<0>{}),
                               Shape<Int<kBM>, Int<kRankTile>>{}, _1{});
    auto tma_l1 =
      make_tma_copy(SM90_TMA_LOAD{}, mL1, typename Traits::SmemLayoutL1{}(_, _, Int<0>{}),
                    Shape<Int<kBN>, Int<kRankTile>>{}, _1{});
    auto tma_c = make_tma_copy(SM90_TMA_STORE{}, mC, typename Traits::SmemLayoutO{},
                               Shape<Int<kBM>, Int<kBN>>{}, _1{});
    using Params = W4A4CuteTma<decltype(tma_b), decltype(tma_sfb), decltype(tma_d),
                               decltype(tma_l1), decltype(tma_c)>;
    return Params{tma_b, tma_sfb, tma_d, tma_l1, tma_c};
  };
  using TmaParams = decltype(build_params());
  auto address = [](const void *ptr) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
  };
  W4A4CuteCacheKey key{{address(wgt), address(out), address(wgt_sf), address(d_ptr),
                        address(l1_ptr), static_cast<uint64_t>(M), static_cast<uint64_t>(N),
                        static_cast<uint64_t>(K), static_cast<uint64_t>(rank)}};
  thread_local std::unordered_map<W4A4CuteCacheKey, TmaParams, W4A4CuteCacheHash> cache;
  auto it = cache.find(key);
  if (it == cache.end()) {
    if (cache.size() >= 32) {
      cache.clear();
    }
    it = cache.emplace(key, build_params()).first;
  }

  auto kernel = gemm_w4a4_kernel<Traits, TmaParams>;
  static thread_local std::unordered_set<int> configured_devices;
  int device = 0;
  checkCUDA(cudaGetDevice(&device));
  if (configured_devices.insert(device).second) {
    checkCUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   Traits::kSmemBytes));
  }
  kernel<<<dim3(N / kBN, M / kBM), Traits::kNumThreads + 128, Traits::kSmemBytes, stream>>>(
    it->second, reinterpret_cast<uint4 const *>(act), reinterpret_cast<uint32_t const *>(act_sf),
    lora_act_f32, scales, bias, wcscales, alpha, M, N, K, rank);
}

}  // namespace svdq::cute_kernels
