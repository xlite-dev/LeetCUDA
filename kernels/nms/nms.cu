#include <algorithm>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// ---------------------------------------------------------------------------
// Phase 1: parallel pairwise IoU -> suppression bitmask (race-free)
//
// Launch: one warp per box. Warp i owns box i (boxes are pre-sorted by
// descending score, so box i can only suppress boxes j with j > i).
// The 32 lanes of warp i compute IoU(i, j) for 32 consecutive j in parallel
// and __ballot_sync collects the verdicts into a single 32-bit word:
// mask[i * words + j/32] bit (j%32) = 1  <=>  i suppresses j.
//
// Each warp writes ONLY its own row of the bitmask -> no write conflicts,
// no atomics needed. 32 IoU evaluations -> 1 global memory write
// (32x fewer writes than one atomicOr per pair).
// ---------------------------------------------------------------------------
__global__ void nms_iou_mask_kernel(const float* __restrict__ boxes,
                                    unsigned int* __restrict__ mask,
                                    const int num_boxes,
                                    const int mask_words,
                                    const float iou_threshold) {
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  if (warp_id >= num_boxes)
    return;
  const int i = warp_id;

  const float x1 = boxes[i * 4 + 0];
  const float y1 = boxes[i * 4 + 1];
  const float x2 = boxes[i * 4 + 2];
  const float y2 = boxes[i * 4 + 3];
  const float area_i = (x2 - x1) * (y2 - y1);

  unsigned int* row = mask + (long long)i * mask_words;

  // 32 lanes test 32 consecutive candidates j; one ballot per 32-box group.
  // j_base is aligned to 32 so that bit k of word j_base/32 is globally
  // box (j_base + k): the bitmask layout matches the resolve kernel's
  // "word w bit k = box w*32 + k". The first word may contain j <= i;
  // those lanes must vote 0 (a box never suppresses itself or earlier boxes).
  for (int j_base = (i + 1) & ~(WARP_SIZE - 1); j_base < num_boxes;
       j_base += WARP_SIZE) {
    const int j = j_base + lane;
    bool suppress = false;
    if (j > i && j < num_boxes) {
      const float x1_j = boxes[j * 4 + 0];
      const float y1_j = boxes[j * 4 + 1];
      const float x2_j = boxes[j * 4 + 2];
      const float y2_j = boxes[j * 4 + 3];

      const float inter_w = fminf(x2, x2_j) - fmaxf(x1, x1_j);
      const float inter_h = fminf(y2, y2_j) - fmaxf(y1, y1_j);
      if (inter_w > 0.0f && inter_h > 0.0f) {
        const float inter_area = inter_w * inter_h;
        const float area_j = (x2_j - x1_j) * (y2_j - y1_j);
        const float iou = inter_area / (area_i + area_j - inter_area);
        suppress = (iou > iou_threshold);
      }
    }
    // collect 32 verdicts into one bitmask word (bit k = lane k's verdict
    // for box j_base + k)
    const unsigned int votes = __ballot_sync(FULL_MASK, suppress);
    if (lane == 0 && votes != 0) {
      row[j_base / WARP_SIZE] = votes;
    }
  }
}

// ---------------------------------------------------------------------------
// Phase 2: sequential resolution on ONE block (mimics CPU NMS semantics)
//
// Launch: 1 block. The suppressed bitmap lives in shared memory, so every
// decision "is box i kept?" is guaranteed to see all earlier decisions:
// __syncthreads() gives the happens-before that cross-block access can
// never provide (this is the root of the original race).
//
// Loop body: thread 0 reads the decision bit; if box i survives, ALL
// threads in parallel OR its suppression row into the shared bitmap.
// ---------------------------------------------------------------------------
__global__ void nms_resolve_kernel(const unsigned int* __restrict__ mask,
                                   int* __restrict__ keep,
                                   const int num_boxes,
                                   const int mask_words) {
  extern __shared__ unsigned int suppressed[];
  for (int w = threadIdx.x; w < mask_words; w += blockDim.x)
    suppressed[w] = 0;
  __syncthreads();

  for (int i = 0; i < num_boxes; ++i) {
    // bit i of word i/32: already suppressed by a kept box?
    const bool suppressed_i =
        (suppressed[i / WARP_SIZE] >> (i % WARP_SIZE)) & 1u;

    if (!suppressed_i) {
      if (threadIdx.x == 0)
        keep[i] = 1; // box i survives: mark as kept
      // all threads in parallel apply box i's suppression row
      const unsigned int* row = mask + (long long)i * mask_words;
      for (int w = threadIdx.x; w < mask_words; w += blockDim.x)
        suppressed[w] |= row[w];
    }
    __syncthreads(); // every decision must be visible before moving on
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
#define CHECK_TORCH_TENSOR_DEVICE(T)                                           \
  if (((T).options().device().type() != torch::kCUDA)) {                       \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be on CUDA device");                 \
  }

torch::Tensor nms(torch::Tensor boxes, torch::Tensor scores,
                  float iou_threshold) {
  CHECK_TORCH_TENSOR_DTYPE(boxes, torch::kFloat32);
  CHECK_TORCH_TENSOR_DTYPE(scores, torch::kFloat32);
  CHECK_TORCH_TENSOR_DEVICE(boxes);
  CHECK_TORCH_TENSOR_DEVICE(scores);
  TORCH_CHECK(boxes.dim() == 2 && boxes.size(1) == 4, "boxes must be (N, 4)");
  TORCH_CHECK(scores.dim() == 1 && scores.size(0) == boxes.size(0),
              "scores must be (N,)");
  const int num_boxes = boxes.size(0);
  if (num_boxes == 0) {
    return torch::empty({0}, torch::TensorOptions()
                                 .dtype(torch::kInt64)
                                 .device(boxes.device()));
  }

  auto toption =
      torch::TensorOptions().dtype(torch::kInt32).device(boxes.device());
  // zeros, not empty: every buffer the kernels (or their readers) touch
  // must start from a defined state
  auto keep = torch::zeros({num_boxes}, toption);
  const int mask_words = (num_boxes + WARP_SIZE - 1) / WARP_SIZE;
  auto moption =
      torch::TensorOptions().dtype(torch::kUInt32).device(boxes.device());
  auto mask = torch::zeros({(long long)num_boxes * mask_words}, moption);

  // sort boxes by descending score (stable: equal scores keep input order)
  auto order_t = std::get<1>(
      scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));
  auto boxes_sorted = boxes.index_select(0, order_t).contiguous();

  // Phase 1: parallel IoU -> suppression bitmask (one warp per box)
  const int threads = 256; // 8 warps per block, each warp handles one box
  dim3 block(threads);
  dim3 grid((num_boxes * WARP_SIZE + threads - 1) / threads);
  nms_iou_mask_kernel<<<grid, block>>>(
      reinterpret_cast<const float *>(boxes_sorted.data_ptr()),
      reinterpret_cast<unsigned int *>(mask.data_ptr()), num_boxes,
      mask_words, iou_threshold);

  // Phase 2: sequential resolution on a single block (shared-memory bitmap)
  const int resolve_threads = 256;
  const int smem_bytes = mask_words * sizeof(unsigned int);
  nms_resolve_kernel<<<1, resolve_threads, smem_bytes>>>(
      reinterpret_cast<const unsigned int *>(mask.data_ptr()),
      reinterpret_cast<int *>(keep.data_ptr()), num_boxes, mask_words);

  // map kept sorted positions back to original input indices.
  // Result: int64 indices on the same device as the inputs, matching
  // torchvision.ops.nms (no accessor<long>, which is platform-dependent).
  auto keep_cpu = keep.to(torch::kCPU);
  auto order_cpu = order_t.to(torch::kCPU);
  auto keep_accessor = keep_cpu.accessor<int, 1>();
  auto order_data = order_cpu.data_ptr<std::int64_t>();
  std::vector<std::int64_t> keep_indices;
  for (int i = 0; i < num_boxes; ++i) {
    if (keep_accessor[i] == 1) {
      keep_indices.push_back(order_data[i]);
    }
  }
  return torch::tensor(keep_indices,
                       torch::TensorOptions()
                           .dtype(torch::kInt64)
                           .device(boxes.device()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { TORCH_BINDING_COMMON_EXTENSION(nms) }
