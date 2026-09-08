import time

import torch
from torch.utils.cpp_extension import load
from torchvision.ops import nms as nms_th

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="nms_lib",
    sources=["nms.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)


def generate_random_data(Nboxes, seed=42):
    g = torch.Generator().manual_seed(seed)
    boxes = torch.rand(Nboxes, 4, generator=g)
    # ensure x1 < x2 and y1 < y2
    b_min = torch.minimum(boxes[:, :2], boxes[:, 2:])
    b_max = torch.maximum(boxes[:, :2], boxes[:, 2:])
    boxes = torch.cat([b_min, b_max], dim=1)
    # shuffled scores with forced ties (every 10th) to exercise stable sort
    scores = torch.rand(Nboxes, generator=g)
    scores[::10] = 0.5
    return boxes, scores


def check_correctness(boxes, scores, thresholds, tag=""):
    """compare lib.nms against torchvision.ops.nms element-wise"""
    tv_keep = nms_th(boxes, scores, thresholds)
    my_keep = lib.nms(boxes, scores, thresholds)
    ok = my_keep.numel() == tv_keep.numel() and torch.equal(
        my_keep.cpu(), tv_keep.cpu())
    if not ok:
        print(f"  CORRECTNESS FAIL {tag}: tv={tv_keep.numel()} "
              f"mine={my_keep.numel()}")
    return ok


def run_benchmark(
    perf_func: callable,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    thresholds: float,
    tag: str,
    warmup: int = 10,
    iters: int = 100,
    show_all: bool = False,
):
    # warmup
    for i in range(warmup):
        out = perf_func(boxes, scores, thresholds)
    torch.cuda.synchronize()

    start = time.time()
    # iters
    for i in range(iters):
        out = perf_func(boxes, scores, thresholds)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"{tag}"
    out_val = sorted(out.flatten().detach().cpu().numpy().tolist())
    len_val = len(out_val)
    out_val = out_val[-min(3, len_val):]
    out_val = [f"{v:<5}" for v in out_val]
    print(
        f"{out_info:>14}: {out_val}, len of keep: {len_val}, time:{mean_time:.8f}ms"
    )
    if show_all:
        print(out)
    return out, mean_time


# ---------- correctness first: fixed 6-box case (issue repro) x 5 runs ----
print("=" * 85)
print("correctness check: fixed 6-box case (issue minimal repro), 5 runs")
boxes6 = torch.tensor([
    [0.0, 0.0, 10.0, 10.0],
    [1.0, 1.0, 11.0, 11.0],
    [20.0, 20.0, 30.0, 30.0],
    [2.0, 2.0, 12.0, 12.0],
    [40.0, 40.0, 50.0, 50.0],
    [21.0, 21.0, 31.0, 31.0],
]).cuda()
scores6 = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]).cuda()
print(f"{'torchvision':>14}: {nms_th(boxes6, scores6, 0.5).cpu().tolist()}")
all_ok = True
for run in range(5):
    my_keep = lib.nms(boxes6, scores6, 0.5).cpu().tolist()
    tv_keep = nms_th(boxes6, scores6, 0.5).cpu().tolist()
    ok = my_keep == tv_keep
    all_ok &= ok
    print(f"{'lib.nms #' + str(run):>14}: {my_keep}, "
          f"{'OK' if ok else 'MISMATCH'}")
print(f"=> fixed case: {'PASS' if all_ok else 'FAIL'}")

# ---------- correctness: random sweep N x seeds x thresholds ---------------
print("=" * 85)
print("correctness check: random sweep (N x seeds x thresholds, ties in scores)")
for Nboxes in [10, 100, 1024, 4096, 8192]:
    for seed in [0, 1, 2]:
        boxes, scores = generate_random_data(Nboxes, seed)
        boxes = boxes.cuda().float().contiguous()
        scores = scores.cuda().float().contiguous()
        for th in [0.5, 0.7]:
            all_ok &= check_correctness(boxes, scores, th,
                                        f"N={Nboxes} seed={seed}")
print(f"=> random sweep: {'PASS' if all_ok else 'FAIL'}")
print("=" * 85)

Nboxes = [1024, 2048, 4096, 8192]
thresholds = 0.5

for nboxes in Nboxes:
    print("-" * 85)
    print(" " * 40 + f"nboxes={nboxes}")
    boxes, scores = generate_random_data(nboxes)
    boxes = boxes.cuda().float().contiguous()
    scores = scores.cuda().float().contiguous()
    run_benchmark(lib.nms, boxes, scores, thresholds, "nms")
    run_benchmark(nms_th, boxes, scores, thresholds, "nms_th")
    print("-" * 85)
