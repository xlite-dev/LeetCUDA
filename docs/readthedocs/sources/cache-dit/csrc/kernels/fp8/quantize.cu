#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdint>

#include "fp8.h"
#include "quantize.cuh"

namespace cache_dit::fp8 {
namespace {

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
  }
  return value;
}

// =================================================================================================
// Per-row BF16 -> FP8-E4M3 quantization: math, layout, and numerical contract
// =================================================================================================
// For each input row m, this kernel computes
//
//   amax[m]  = max_k abs(float(input[m, k]))
//   scale[m] = amax[m] == 0 ? 0 : amax[m] / 448
//   q[m, k]  = satfinite_e4m3(float(input[m, k]) / (scale[m] == 0 ? 1 : scale[m]))
//
// scale is the dequantization multiplier consumed by the FP8 GEMM. It is deliberately not the
// reciprocal quantization multiplier. Keeping this convention makes dequant(q) = q * scale and
// preserves exact zeros: an all-zero row has scale=0 and q=0, never 0/0. The conversion helper
// performs the E4M3 rounding and saturation; values are not rounded through an intermediate FP16.
// fmaxf also defines the native NaN policy: NaNs do not propagate into amax as they do in
// torch.amax.
//
// Parallel decomposition:
//   * kWarpsPerRow consecutive warps own one logical row; the production launcher uses eight.
//   * A group therefore contains 256 lanes, and each lane reads eight contiguous BF16 values per
//     iteration (16 B). The last dimension is checked to be a multiple of 16, so row starts and
//     every Vec8 access remain aligned; there is no hidden scalar K-tail in this main kernel.
//   * kRowsPerBlock is one for the production configuration. `row_in_block` and `group_lane`
//     retain the more general template mapping without changing the public row-major ABI.
//
// The algorithm has two logical passes over a row. Pass 1 reduces amax and pass 2 converts/stores.
// With kCacheInput=true, pass 1 also leaves the Vec8 input in dynamic shared memory so pass 2 does
// not repeat the global read. The launcher selects this only for K <= 4096, bounding the cache to
// 8 KiB per row. With kCacheInput=false, input_cache has size zero and pass 2 rereads global
// memory; this avoids sacrificing occupancy on long-K rows. Both instantiations implement the same
// math.
//
// Reduction and synchronization are hierarchical:
//   1. each lane reduces its private values in FP32;
//   2. warp_max performs an xor-shuffle reduction, preserving the native fmaxf NaN behavior;
//   3. lane zero publishes each warp result to warp_amax;
//   4. one __syncthreads makes both the reductions and cached BF16 input visible to the row group;
//   5. every lane reduces the small warp_amax array and hence uses an identical safe scale.
// No scale array entry is read before step 4, and no second CTA-wide barrier is required.
template <int kWarpsPerRow, bool kCacheInput>
__global__ void quantize_per_row_kernel(const __nv_bfloat16* __restrict__ input,
                                        __nv_fp8_e4m3* __restrict__ output,
                                        float* __restrict__ scale, int64_t rows, int64_t cols) {
  static_assert(kWarpsPerBlock % kWarpsPerRow == 0, "per-row warp groups must cover the whole CTA");
  constexpr int kRowsPerBlock = kWarpsPerBlock / kWarpsPerRow;
  constexpr int kGroupSize = kWarpsPerRow * kWarpSize;
  extern __shared__ __align__(16) __nv_bfloat16 input_cache[];
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int row_in_block = warp / kWarpsPerRow;
  const int group_lane = (warp % kWarpsPerRow) * kWarpSize + lane;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + row_in_block;
  const bool valid = row < rows;

  // Stage 1: form the row view and compute lane-private amax. Invalid rows still participate in
  // barriers but are prevented from loading, storing, or publishing a scale by `valid`.
  const __nv_bfloat16* src = input + row * cols;
  __nv_bfloat16* cached = input_cache + row_in_block * cols;
  float amax = 0.0f;
  for (int64_t col = static_cast<int64_t>(group_lane) * 8; valid && col < cols;
       col += kGroupSize * 8) {
    const Vec8BF16 vec = *reinterpret_cast<const Vec8BF16*>(src + col);
    if constexpr (kCacheInput) {
      *reinterpret_cast<Vec8BF16*>(cached + col) = vec;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      amax = fmaxf(amax, fabsf(__bfloat162float(vec.elem[i])));
    }
  }
  amax = warp_max(amax);

  __shared__ float warp_amax[kWarpsPerBlock];
  if (lane == 0) {
    warp_amax[warp] = amax;
  }
  __syncthreads();
  amax = 0.0f;
#pragma unroll
  for (int i = 0; i < kWarpsPerRow; ++i) {
    amax = fmaxf(amax, warp_amax[row_in_block * kWarpsPerRow + i]);
  }

  const float safe_scale = amax == 0.0f ? 1.0f : amax / kE4M3Max;
  if (valid && group_lane == 0) {
    scale[row] = amax == 0.0f ? 0.0f : safe_scale;
  }

  __nv_fp8_e4m3* dst = output + row * cols;
  for (int64_t col = static_cast<int64_t>(group_lane) * 8; valid && col < cols;
       col += kGroupSize * 8) {
    const Vec8BF16 vec = kCacheInput ? *reinterpret_cast<const Vec8BF16*>(cached + col)
                                     : *reinterpret_cast<const Vec8BF16*>(src + col);
    float values[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      values[i] = __bfloat162float(vec.elem[i]);
    }
    store_e4m3x8(dst + col, values, safe_scale);
  }
}

__global__ void tensor_amax_kernel(const __nv_bfloat16* __restrict__ input, int64_t numel,
                                   float* __restrict__ scale) {
  float amax = 0.0f;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t base = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; base * 8 < numel;
       base += stride) {
    const int64_t offset = base * 8;
    if (offset + 7 < numel) {
      const Vec8BF16 vec = *reinterpret_cast<const Vec8BF16*>(input + offset);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        amax = fmaxf(amax, fabsf(__bfloat162float(vec.elem[i])));
      }
    } else {
      for (int64_t tail = offset; tail < numel; ++tail) {
        amax = fmaxf(amax, fabsf(__bfloat162float(input[tail])));
      }
    }
  }
  amax = warp_max(amax);

  __shared__ float warp_amax[kWarpsPerBlock];
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  if (lane == 0) {
    warp_amax[warp] = amax;
  }
  __syncthreads();
  if (warp == 0) {
    amax = lane < kWarpsPerBlock ? warp_amax[lane] : 0.0f;
    amax = warp_max(amax);
    if (lane == 0 && amax > 0.0f) {
      atomicMax(reinterpret_cast<unsigned int*>(scale), __float_as_uint(amax));
    }
  }
}

__global__ void finalize_per_tensor_scale_kernel(float* __restrict__ scale) {
  const float amax = scale[0];
  scale[0] = amax == 0.0f ? 0.0f : amax / kE4M3Max;
}

__global__ void quantize_per_tensor_kernel(const __nv_bfloat16* __restrict__ input,
                                           __nv_fp8_e4m3* __restrict__ output,
                                           const float* __restrict__ scale, int64_t numel) {
  const float dequant_scale = scale[0];
  const float safe_scale = dequant_scale == 0.0f ? 1.0f : dequant_scale;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t base = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; base * 8 < numel;
       base += stride) {
    const int64_t offset = base * 8;
    if (offset + 7 < numel) {
      const Vec8BF16 vec = *reinterpret_cast<const Vec8BF16*>(input + offset);
      float values[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        values[i] = __bfloat162float(vec.elem[i]);
      }
      store_e4m3x8(output + offset, values, safe_scale);
    } else {
      for (int64_t tail = offset; tail < numel; ++tail) {
        const float value = __bfloat162float(input[tail]);
        output[tail] = __nv_fp8_e4m3(value / safe_scale);
      }
    }
  }
}

__global__ void quantize_per_block_activation_kernel(const __nv_bfloat16* __restrict__ input,
                                                     __nv_fp8_e4m3* __restrict__ output,
                                                     float* __restrict__ scale, int64_t rows,
                                                     int64_t cols) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) {
    return;
  }

  const int64_t num_k_blocks = (cols + kScaleBlock - 1) / kScaleBlock;
  const __nv_bfloat16* src = input + row * cols;
  __nv_fp8_e4m3* dst = output + row * cols;
  for (int64_t block = 0; block < num_k_blocks; ++block) {
    const int64_t begin = block * kScaleBlock;
    const int64_t remaining = cols - begin;
    const int width = static_cast<int>(remaining < kScaleBlock ? remaining : kScaleBlock);
    float amax = 0.0f;
    for (int col = lane * 4; col < width; col += kWarpSize * 4) {
      const Vec4BF16 vec = *reinterpret_cast<const Vec4BF16*>(src + begin + col);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        amax = fmaxf(amax, fabsf(__bfloat162float(vec.elem[i])));
      }
    }
    amax = warp_max(amax);
    const float safe_scale = amax == 0.0f ? 1.0f : amax / kE4M3Max;
    if (lane == 0) {
      scale[row * num_k_blocks + block] = amax == 0.0f ? 0.0f : safe_scale;
    }
    for (int col = lane * 4; col < width; col += kWarpSize * 4) {
      const Vec4BF16 vec = *reinterpret_cast<const Vec4BF16*>(src + begin + col);
      uint32_t packed;
      auto* halves = reinterpret_cast<unsigned short*>(&packed);
      halves[0] = cvt_f2_to_e4m3x2(__bfloat162float(vec.elem[0]) / safe_scale,
                                   __bfloat162float(vec.elem[1]) / safe_scale);
      halves[1] = cvt_f2_to_e4m3x2(__bfloat162float(vec.elem[2]) / safe_scale,
                                   __bfloat162float(vec.elem[3]) / safe_scale);
      *reinterpret_cast<uint32_t*>(dst + begin + col) = packed;
    }
  }
}

__global__ void quantize_per_block_weight_kernel(const __nv_bfloat16* __restrict__ input,
                                                 __nv_fp8_e4m3* __restrict__ output,
                                                 float* __restrict__ scale, int64_t rows,
                                                 int64_t cols) {
  constexpr int kChunksPerRow = kScaleBlock / 8;
  constexpr int kChunks = kScaleBlock * kChunksPerRow;
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * kScaleBlock;
  const int64_t col_base = static_cast<int64_t>(blockIdx.y) * kScaleBlock;
  const int64_t num_k_blocks = (cols + kScaleBlock - 1) / kScaleBlock;

  float amax = 0.0f;
  for (int index = threadIdx.x; index < kChunks; index += blockDim.x) {
    const int64_t row = row_base + index / kChunksPerRow;
    const int64_t col = col_base + (index % kChunksPerRow) * 8;
    if (row >= rows || col >= cols) {
      continue;
    }
    const Vec8BF16 vec = *reinterpret_cast<const Vec8BF16*>(input + row * cols + col);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      amax = fmaxf(amax, fabsf(__bfloat162float(vec.elem[i])));
    }
  }
  amax = warp_max(amax);

  __shared__ float warp_amax[kWarpsPerBlock];
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  if (lane == 0) {
    warp_amax[warp] = amax;
  }
  __syncthreads();
  if (warp == 0) {
    amax = lane < kWarpsPerBlock ? warp_amax[lane] : 0.0f;
    amax = warp_max(amax);
    if (lane == 0) {
      warp_amax[0] = amax;
      scale[static_cast<int64_t>(blockIdx.x) * num_k_blocks + blockIdx.y] = amax / kE4M3Max;
    }
  }
  __syncthreads();

  const float safe_scale = warp_amax[0] == 0.0f ? 1.0f : warp_amax[0] / kE4M3Max;
  for (int index = threadIdx.x; index < kChunks; index += blockDim.x) {
    const int64_t row = row_base + index / kChunksPerRow;
    const int64_t col = col_base + (index % kChunksPerRow) * 8;
    if (row >= rows || col >= cols) {
      continue;
    }
    const Vec8BF16 vec = *reinterpret_cast<const Vec8BF16*>(input + row * cols + col);
    float values[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      values[i] = __bfloat162float(vec.elem[i]);
    }
    store_e4m3x8(output + row * cols + col, values, safe_scale);
  }
}

void check_quantize_inputs(const at::Tensor& input, const at::Tensor& output,
                           const at::Tensor& scale, int64_t mode) {
  TORCH_CHECK(input.is_cuda(), "fp8_quantize expects a CUDA input.");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "fp8_quantize expects bfloat16 input.");
  TORCH_CHECK(input.dim() == 2, "fp8_quantize expects a 2D input.");
  TORCH_CHECK(input.is_contiguous(), "fp8_quantize expects contiguous input.");
  TORCH_CHECK(output.is_cuda() && output.device() == input.device(),
              "fp8_quantize output must be on the input device.");
  TORCH_CHECK(output.scalar_type() == at::kFloat8_e4m3fn,
              "fp8_quantize output must be float8_e4m3fn.");
  TORCH_CHECK(output.sizes() == input.sizes(), "fp8_quantize output shape must match input.");
  TORCH_CHECK(output.is_contiguous(), "fp8_quantize output must be contiguous.");
  TORCH_CHECK(scale.is_cuda() && scale.device() == input.device(),
              "fp8_quantize scale must be on the input device.");
  TORCH_CHECK(scale.scalar_type() == at::kFloat, "fp8_quantize scale must be float32.");
  TORCH_CHECK(scale.is_contiguous(), "fp8_quantize scale must be contiguous.");
  TORCH_CHECK(mode == kPerRow || mode == kPerTensor || mode == kPerBlock,
              "fp8_quantize mode must be 0 (per_row), 1 (per_tensor), or 2 (per_block).");
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(properties->major >= 10,
              "cache-dit FP8 kernels require sm100 or newer; this device is sm", properties->major,
              properties->minor, ". Use --quantize-backend torchao on Ada or Hopper.");
}

}  // namespace

void fp8_quantize(const at::Tensor& input, at::Tensor& output, at::Tensor& scale, int64_t mode,
                  bool is_weight) {
  check_quantize_inputs(input, output, scale, mode);
  const c10::cuda::CUDAGuard device_guard(input.device());
  const int64_t rows = input.size(0);
  const int64_t cols = input.size(1);
  const int64_t numel = input.numel();
  TORCH_CHECK(rows > 0 && cols > 0, "fp8_quantize does not accept empty tensors.");
  TORCH_CHECK(cols % 16 == 0, "fp8_quantize requires the last dimension to be a multiple of 16.");
  const int64_t num_k_blocks = (cols + kScaleBlock - 1) / kScaleBlock;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* src = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>());
  auto* dst = reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr<at::Float8_e4m3fn>());
  auto* scales = scale.data_ptr<float>();

  if (mode == kPerRow) {
    TORCH_CHECK(scale.numel() == rows, "per-row scale must have M elements.");
    constexpr int kWarpsPerRow = 8;
    constexpr int kRowsPerBlock = kWarpsPerBlock / kWarpsPerRow;
    const int blocks = static_cast<int>((rows + kRowsPerBlock - 1) / kRowsPerBlock);
    if (cols <= 4096) {
      const int cache_bytes = static_cast<int>(kRowsPerBlock * cols * sizeof(__nv_bfloat16));
      quantize_per_row_kernel<kWarpsPerRow, true>
        <<<blocks, kThreads, cache_bytes, stream>>>(src, dst, scales, rows, cols);
    } else {
      quantize_per_row_kernel<kWarpsPerRow, false>
        <<<blocks, kThreads, 0, stream>>>(src, dst, scales, rows, cols);
    }
  } else if (mode == kPerTensor) {
    TORCH_CHECK(scale.numel() == 1, "per-tensor scale must be a single element.");
    const int blocks =
      static_cast<int>(std::min<int64_t>((numel / 8 + kThreads - 1) / kThreads, 4096));
    C10_CUDA_CHECK(cudaMemsetAsync(scales, 0, sizeof(float), stream));
    tensor_amax_kernel<<<blocks, kThreads, 0, stream>>>(src, numel, scales);
    finalize_per_tensor_scale_kernel<<<1, 1, 0, stream>>>(scales);
    quantize_per_tensor_kernel<<<blocks, kThreads, 0, stream>>>(src, dst, scales, numel);
  } else {
    TORCH_CHECK(cols % kScaleBlock == 0,
                "per-block quantization requires the last dimension to be a multiple of 128.");
    if (is_weight) {
      TORCH_CHECK(scale.numel() == ((rows + kScaleBlock - 1) / kScaleBlock) * num_k_blocks,
                  "per-block weight scale has an invalid shape.");
      const dim3 grid((rows + kScaleBlock - 1) / kScaleBlock, num_k_blocks);
      quantize_per_block_weight_kernel<<<grid, kThreads, 0, stream>>>(src, dst, scales, rows, cols);
    } else {
      TORCH_CHECK(scale.numel() == rows * num_k_blocks,
                  "per-block activation scale has an invalid shape.");
      const int blocks = static_cast<int>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
      quantize_per_block_activation_kernel<<<blocks, kThreads, 0, stream>>>(src, dst, scales, rows,
                                                                            cols);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

struct SwigluInput {
  const __nv_bfloat16* value;
  const __nv_bfloat16* gate;
  int64_t input_cols;
  int64_t cols;

  __device__ __forceinline__ Vec8BF16 load8(int64_t row, int64_t col) const {
    const Vec8BF16 values = *reinterpret_cast<const Vec8BF16*>(value + row * input_cols + col);
    const __nv_bfloat16* gate_row =
      gate == nullptr ? value + row * input_cols + cols : gate + row * cols;
    const Vec8BF16 gates = *reinterpret_cast<const Vec8BF16*>(gate_row + col);
    Vec8BF16 result;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float g = __bfloat162float(gates.elem[i]);
      const float silu = g / (1.0f + __expf(-g));
      const __nv_bfloat16 silu_bf16 = __float2bfloat16_rn(silu);
      const float product = __bfloat162float(values.elem[i]) * __bfloat162float(silu_bf16);
      result.elem[i] = __float2bfloat16_rn(product);
    }
    return result;
  }
};

template <bool kCacheInput>
__global__ void quantize_swiglu_per_row_kernel(SwigluInput input,
                                               __nv_fp8_e4m3* __restrict__ output,
                                               float* __restrict__ scale, int64_t rows,
                                               int64_t cols) {
  constexpr int kWarpsPerRow = 8;
  constexpr int kRowsPerBlock = kWarpsPerBlock / kWarpsPerRow;
  constexpr int kGroupSize = kWarpsPerRow * kWarpSize;
  extern __shared__ __align__(16) __nv_bfloat16 input_cache[];
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int row_in_block = warp / kWarpsPerRow;
  const int group_lane = (warp % kWarpsPerRow) * kWarpSize + lane;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + row_in_block;
  const bool valid = row < rows;
  __nv_bfloat16* cached = input_cache + row_in_block * cols;

  float amax = 0.0f;
  for (int64_t col = static_cast<int64_t>(group_lane) * 8; valid && col < cols;
       col += kGroupSize * 8) {
    const Vec8BF16 vec = input.load8(row, col);
    if constexpr (kCacheInput) {
      *reinterpret_cast<Vec8BF16*>(cached + col) = vec;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      amax = fmaxf(amax, fabsf(__bfloat162float(vec.elem[i])));
    }
  }
  amax = warp_max(amax);

  __shared__ float warp_amax[kWarpsPerBlock];
  if (lane == 0) {
    warp_amax[warp] = amax;
  }
  __syncthreads();
  amax = 0.0f;
#pragma unroll
  for (int i = 0; i < kWarpsPerRow; ++i) {
    amax = fmaxf(amax, warp_amax[row_in_block * kWarpsPerRow + i]);
  }

  const float safe_scale = amax == 0.0f ? 1.0f : amax / kE4M3Max;
  if (valid && group_lane == 0) {
    scale[row] = amax == 0.0f ? 0.0f : safe_scale;
  }
  __nv_fp8_e4m3* dst = output + row * cols;
  for (int64_t col = static_cast<int64_t>(group_lane) * 8; valid && col < cols;
       col += kGroupSize * 8) {
    const Vec8BF16 vec =
      kCacheInput ? *reinterpret_cast<const Vec8BF16*>(cached + col) : input.load8(row, col);
    float values[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      values[i] = __bfloat162float(vec.elem[i]);
    }
    store_e4m3x8(dst + col, values, safe_scale);
  }
}

__global__ void swiglu_tensor_amax_kernel(SwigluInput input, int64_t rows, int64_t cols,
                                          float* __restrict__ scale) {
  const int64_t numel = rows * cols;
  float amax = 0.0f;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t base = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; base * 8 < numel;
       base += stride) {
    const int64_t offset = base * 8;
    const Vec8BF16 vec = input.load8(offset / cols, offset % cols);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      amax = fmaxf(amax, fabsf(__bfloat162float(vec.elem[i])));
    }
  }
  amax = warp_max(amax);

  __shared__ float warp_amax[kWarpsPerBlock];
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  if (lane == 0) {
    warp_amax[warp] = amax;
  }
  __syncthreads();
  if (warp == 0) {
    amax = lane < kWarpsPerBlock ? warp_amax[lane] : 0.0f;
    amax = warp_max(amax);
    if (lane == 0 && amax > 0.0f) {
      atomicMax(reinterpret_cast<unsigned int*>(scale), __float_as_uint(amax));
    }
  }
}

__global__ void quantize_swiglu_per_tensor_kernel(SwigluInput input,
                                                  __nv_fp8_e4m3* __restrict__ output,
                                                  const float* __restrict__ scale, int64_t rows,
                                                  int64_t cols) {
  const int64_t numel = rows * cols;
  const float dequant_scale = scale[0];
  const float safe_scale = dequant_scale == 0.0f ? 1.0f : dequant_scale;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t base = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; base * 8 < numel;
       base += stride) {
    const int64_t offset = base * 8;
    const Vec8BF16 vec = input.load8(offset / cols, offset % cols);
    float values[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      values[i] = __bfloat162float(vec.elem[i]);
    }
    store_e4m3x8(output + offset, values, safe_scale);
  }
}

__global__ void quantize_swiglu_per_block_kernel(SwigluInput input,
                                                 __nv_fp8_e4m3* __restrict__ output,
                                                 float* __restrict__ scale, int64_t rows,
                                                 int64_t cols) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) {
    return;
  }

  const int64_t num_k_blocks = cols / kScaleBlock;
  __nv_fp8_e4m3* dst = output + row * cols;
  for (int64_t block = 0; block < num_k_blocks; ++block) {
    const int64_t begin = block * kScaleBlock;
    float amax = 0.0f;
    for (int col = lane * 8; col < kScaleBlock; col += kWarpSize * 8) {
      const Vec8BF16 vec = input.load8(row, begin + col);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        amax = fmaxf(amax, fabsf(__bfloat162float(vec.elem[i])));
      }
    }
    amax = warp_max(amax);
    const float safe_scale = amax == 0.0f ? 1.0f : amax / kE4M3Max;
    if (lane == 0) {
      scale[row * num_k_blocks + block] = amax == 0.0f ? 0.0f : safe_scale;
    }
    for (int col = lane * 8; col < kScaleBlock; col += kWarpSize * 8) {
      const Vec8BF16 vec = input.load8(row, begin + col);
      float values[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        values[i] = __bfloat162float(vec.elem[i]);
      }
      store_e4m3x8(dst + begin + col, values, safe_scale);
    }
  }
}

}  // namespace

void fp8_quantize_swiglu(const at::Tensor& value, const std::optional<at::Tensor>& gate,
                         at::Tensor& output, at::Tensor& scale, int64_t mode) {
  TORCH_CHECK(value.is_cuda() && value.scalar_type() == at::kBFloat16 && value.dim() == 2 &&
                value.is_contiguous(),
              "fp8_quantize_swiglu expects a contiguous 2D bfloat16 CUDA value tensor.");
  TORCH_CHECK(
    output.is_cuda() && output.device() == value.device() &&
      output.scalar_type() == at::kFloat8_e4m3fn && output.dim() == 2 && output.is_contiguous(),
    "fp8_quantize_swiglu output must be a contiguous 2D E4M3 tensor on the input device.");
  TORCH_CHECK(scale.is_cuda() && scale.device() == value.device() &&
                scale.scalar_type() == at::kFloat && scale.is_contiguous(),
              "fp8_quantize_swiglu scale must be contiguous float32 on the input device.");
  const int64_t rows = value.size(0);
  const int64_t input_cols = value.size(1);
  const int64_t cols = output.size(1);
  TORCH_CHECK(output.size(0) == rows && rows > 0 && cols > 0,
              "fp8_quantize_swiglu output must be [M, H] with nonempty dimensions.");
  TORCH_CHECK(cols % 16 == 0, "fp8_quantize_swiglu requires H to be a multiple of 16.");
  TORCH_CHECK(mode == kPerRow || mode == kPerTensor || mode == kPerBlock,
              "fp8_quantize_swiglu mode must be 0 (per_row), 1 (per_tensor), or 2 (per_block).");

  const __nv_bfloat16* gate_ptr = nullptr;
  if (gate.has_value()) {
    const at::Tensor& gate_tensor = gate.value();
    TORCH_CHECK(gate_tensor.is_cuda() && gate_tensor.device() == value.device() &&
                  gate_tensor.scalar_type() == at::kBFloat16 && gate_tensor.is_contiguous() &&
                  gate_tensor.sizes() == output.sizes(),
                "fp8_quantize_swiglu gate must be a contiguous BF16 [M, H] tensor on the input "
                "device.");
    TORCH_CHECK(input_cols == cols,
                "A separate gate expects value [M, H], not a half-split [M, 2H] tensor.");
    gate_ptr = reinterpret_cast<const __nv_bfloat16*>(gate_tensor.data_ptr<at::BFloat16>());
  } else {
    TORCH_CHECK(input_cols == 2 * cols, "fp8_quantize_swiglu without gate expects value [M, 2H].");
  }
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(value.get_device());
  TORCH_CHECK(properties->major >= 10, "cache-dit FP8 kernels require sm100 or newer.");

  const c10::cuda::CUDAGuard device_guard(value.device());
  const int64_t num_k_blocks = cols / kScaleBlock;
  if (mode == kPerRow) {
    TORCH_CHECK(scale.numel() == rows, "per-row SwiGLU scale must have M elements.");
  } else if (mode == kPerTensor) {
    TORCH_CHECK(scale.numel() == 1, "per-tensor SwiGLU scale must be a scalar.");
  } else {
    TORCH_CHECK(cols % kScaleBlock == 0,
                "per-block SwiGLU quantization requires H to be a multiple of 128.");
    TORCH_CHECK(scale.numel() == rows * num_k_blocks,
                "per-block SwiGLU scale must have M * H / 128 elements.");
  }

  SwigluInput input{
    reinterpret_cast<const __nv_bfloat16*>(value.data_ptr<at::BFloat16>()),
    gate_ptr,
    input_cols,
    cols,
  };
  auto* dst = reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr<at::Float8_e4m3fn>());
  float* scales = scale.data_ptr<float>();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (mode == kPerRow) {
    const int blocks = static_cast<int>(rows);
    if (cols <= 4096) {
      const int cache_bytes = static_cast<int>(cols * sizeof(__nv_bfloat16));
      quantize_swiglu_per_row_kernel<true>
        <<<blocks, kThreads, cache_bytes, stream>>>(input, dst, scales, rows, cols);
    } else {
      quantize_swiglu_per_row_kernel<false>
        <<<blocks, kThreads, 0, stream>>>(input, dst, scales, rows, cols);
    }
  } else if (mode == kPerTensor) {
    const int64_t numel = rows * cols;
    const int blocks =
      static_cast<int>(std::min<int64_t>((numel / 8 + kThreads - 1) / kThreads, 4096));
    C10_CUDA_CHECK(cudaMemsetAsync(scales, 0, sizeof(float), stream));
    swiglu_tensor_amax_kernel<<<blocks, kThreads, 0, stream>>>(input, rows, cols, scales);
    finalize_per_tensor_scale_kernel<<<1, 1, 0, stream>>>(scales);
    quantize_swiglu_per_tensor_kernel<<<blocks, kThreads, 0, stream>>>(input, dst, scales, rows,
                                                                       cols);
  } else {
    const int blocks = static_cast<int>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    quantize_swiglu_per_block_kernel<<<blocks, kThreads, 0, stream>>>(input, dst, scales, rows,
                                                                      cols);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::fp8
