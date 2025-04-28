"""
Adapted from Triton tutorial
https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html#sphx-glr-getting-started-tutorials-02-fused-softmax-py
"""

import torch
import triton
import triton.language as tl


# @torch.compile
def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    k_stages: tl.constexpr,
):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=k_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def get_device_properties(device_id=None):
    import pycuda.driver as cuda

    device = (
        cuda.Device(device_id)
        if device_id is not None
        else torch.cuda.current_device()
    )
    NUM_SM = device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
    NUM_REGS = device.get_attribute(
        cuda.device_attribute.MAX_REGISTERS_PER_BLOCK
    )
    SIZE_SMEM = device.get_attribute(
        cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
    )
    WARP_SIZE = device.get_attribute(cuda.device_attribute.WARP_SIZE)
    return NUM_SM, NUM_REGS, SIZE_SMEM, WARP_SIZE


DEVICE = torch.cuda.current_device()
NUM_SM, NUM_REGS, SIZE_SMEM, WARP_SIZE = get_device_properties(DEVICE)
print(
    f"NUM_SM: {NUM_SM}, NUM_REGS: {NUM_REGS}, "
    f"SIZE_SMEM: {SIZE_SMEM}, WARP_SIZE: {WARP_SIZE}"
)


def get_num_programs(x):
    n_rows, n_cols = x.shape
    # The block size of each loop iteration is the smallest power
    # of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    # Number of software pipelining stages.
    k_stages = 4 if SIZE_SMEM > 200000 else 2
    # Allocate output
    y = torch.empty_like(x)
    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel = softmax_kernel.warmup(
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        k_stages=k_stages,
        num_warps=num_warps,
        grid=(1,),
    )
    kernel._init_handles()
    n_regs = kernel.n_regs
    # shared > 0 if k_stages is not 0
    size_smem = kernel.metadata.shared
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy
    return num_programs


NUM_PROGRAMS = get_num_programs(torch.randn(4096, 2048, device="cuda"))


def triton_softmax(x: torch.Tensor):
    """Compute row-wise softmax of X using Triton"""
    n_rows, n_cols = x.shape
    # The block size of each loop iteration is the smallest power of
    # two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    # Number of software pipelining stages.
    k_stages = 4 if SIZE_SMEM > 200000 else 2
    # Allocate output
    y = torch.empty_like(x)
    num_programs = min(NUM_PROGRAMS, n_rows)

    # Create a number of persistent programs.
    softmax_kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        k_stages=k_stages,
        num_warps=num_warps,
    )
    return y


torch.manual_seed(0)
x = torch.randn(1823, 781, device="cuda")
y_triton = triton_softmax(x)
y_torch = naive_softmax(x)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],  # argument names to use as an x-axis for the plot
        x_vals=[
            256 * i for i in range(1, 64)
        ],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            "triton-fused-softmax",
            "torch-fused-softmax",
            "torch-naive-softmax",
        ],  # possible values for `line_arg``
        line_names=[
            "Triton Fused Softmax",
            "Torch Fused Softmax",
            "Torch Naive Softmax",
        ],  # label name for the lines
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        xlabel=f"M, {torch.cuda.get_device_name(DEVICE)}",  # label name for the x-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={
            "N": 2048
        },  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == "torch-naive-softmax":
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    if provider == "triton-fused-softmax":
        ms = triton.testing.do_bench(lambda: triton_softmax(x))
    if provider == "torch-fused-softmax":
        ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=-1))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


benchmark.run(show_plots=True, print_data=True, save_path="./")
