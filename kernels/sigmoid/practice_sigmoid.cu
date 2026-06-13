#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

// ============================================================
// Practice kernels — best-performing variant per data type.
// Names are intentionally plain (no vector-width hints) so the
// exercise focuses on re-implementing the optimal kernel itself.
//   FP32  -> best: float4 vectorized (128-bit LD/ST)
//   FP16  -> best: 128-bit pack + hexp
// sigmoid(x) = 1 / (1 + exp(-x)); clamp -x to avoid expf overflow.
// ============================================================

__global__ void sigmoid_f32_kernel(float *x, float *y, int N) {
  // TODO(practice): best FP32 sigmoid — 4-element vectorized
}

__global__ void sigmoid_f16_kernel(half *x, half *y, int N) {
  // TODO(practice): best FP16 sigmoid — 8-element 128-bit pack + hexp
}

// ============================================================
// Torch bindings
// ============================================================
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    throw std::runtime_error("Tensor dtype mismatch, expected " #th_type);     \
  }

// n_elements: how many elements each thread processes in the fast path
#define TORCH_BINDING_SIGMOID(packed_type, th_type, element_type, n_elements)  \
  void sigmoid_##packed_type(torch::Tensor x, torch::Tensor y) {               \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int ndim = x.dim();                                                  \
    const int S = x.size(0);                                                   \
    const int K = (ndim >= 2) ? x.size(1) : 1;                                 \
    const int N = x.numel();                                                   \
    dim3 block, grid;                                                          \
    if (ndim == 2 && (K / (n_elements)) <= 1024) {                             \
      block = dim3(K / (n_elements));                                          \
      grid  = dim3(S);                                                         \
    } else {                                                                   \
      block = dim3(256 / (n_elements));                                        \
      grid  = dim3((N + 256 - 1) / 256);                                       \
    }                                                                          \
    sigmoid_##packed_type##_kernel<<<grid, block>>>(                           \
        reinterpret_cast<element_type *>(x.data_ptr()),                        \
        reinterpret_cast<element_type *>(y.data_ptr()), N);                    \
  }

TORCH_BINDING_SIGMOID(f32, torch::kFloat32, float, 4)
TORCH_BINDING_SIGMOID(f16, torch::kHalf,    half,  8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f32)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16)
}
