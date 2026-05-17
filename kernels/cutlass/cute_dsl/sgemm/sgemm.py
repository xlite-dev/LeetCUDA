import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.cute.testing as testing

import torch

@cute.kernel
def matmul_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, M, N, K):
    bidx_x, bidx_y = cute.arch.block_idx()[0], cute.arch.block_idx()[1] 
    bdim_x, bdim_y = cute.arch.block_dim()[0], cute.arch.block_dim()[1] 
    tidx_x, tidx_y = cute.arch.thread_idx()[0], cute.arch.thread_idx()[1]

    col = bidx_x * bdim_x + tidx_x
    row = bidx_y * bdim_y + tidx_y

    # no early return supported yet
    # ref - https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html
    # if(row > M or col > N): 
    #     return

    acc = cute.Float32(0) # cannot put inside the control flow, needs to be defined outside

    if row < M and col < N:
        for k in range(K): # shared dim
            acc += cute.Float32(A[row, k]) * cute.Float32(B[k, col]) #upcast
        
    C[row, col] = cute.Float16(acc) # downcast


@cute.jit
def host(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    M, K = A.shape
    K, N = B.shape

    block = 32
    grid_x = (N + block - 1) // block
    grid_y = (M + block - 1) // block


    matmul_kernel(A, B, C, M, N, K).launch(
        grid = (grid_x, grid_y),
        block = (block, block),
    )


def run_sgemm(
    M: int = 2048,
    K: int = 2048,
    N: int = 2048,
    verify: bool = True,
    warmup_iterations: int = 10,
    iterations: int = 100,
):
    print(f"running sgemm with M={M}, N={N}, K={K}")

    def tensor_generator(return_torch_tensor: bool = False):
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)
        c = torch.zeros(M, N,  device="cuda", dtype=torch.float32)

        a_tensor = from_dlpack(a, assumed_align=16)
        b_tensor = from_dlpack(b, assumed_align=16)
        c_tensor = from_dlpack(c, assumed_align=16)

        if return_torch_tensor:
            return a, b, c, a_tensor, b_tensor, c_tensor
        return a_tensor, b_tensor, c_tensor

    workspace_generator = lambda: testing.JitArguments(*tensor_generator())

    _a, _b, _c, _a_tensor, _b_tensor, _c_tensor = tensor_generator(return_torch_tensor=True)

    import time
    compile_tic = time.perf_counter()
    matmul = cute.compile(host, _a_tensor, _b_tensor, _c_tensor)
    print(f"kernel compiled in {time.perf_counter() - compile_tic:.4f} seconds")

    # verify results
    if verify:
        matmul(_a_tensor, _b_tensor, _c_tensor)
        torch.cuda.synchronize()
        torch.testing.assert_close(_c, torch.matmul(_a, _b), atol=1e-2, rtol=1e-2)
        print("verification passed!")
    else:
        print("verification skipped...")

    # benchmark torch
    torch.cuda.empty_cache()
    for _ in range(warmup_iterations):
        _ = torch.matmul(_a, _b)
    torch.cuda.synchronize()
    torch_tic = time.perf_counter()
    for _ in range(iterations):
        _ = torch.matmul(_a, _b)
    torch.cuda.synchronize()
    torch_avg_time_us = (time.perf_counter() - torch_tic) * 1e6 / iterations
    print(f"torch kernel execution time: {torch_avg_time_us / 1e3:.2f} ms")
    print(f"torch achieved TFLOPS: {(2 * M * N * K) / torch_avg_time_us / 1e6:.2f}")

    # benchmark cute
    torch.cuda.empty_cache()
    workspace_bytes = (M * K + N * K + M * N) * 4
    workspace_count = testing.get_workspace_count(workspace_bytes, warmup_iterations, iterations)
    avg_time_us = testing.benchmark(
        matmul,
        workspace_generator=workspace_generator,
        workspace_count=workspace_count,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        use_cuda_graphs=False,
    )
    print(f"cute kernel execution time: {avg_time_us / 1e3:.2f} ms")
    print(f"cute achieved TFLOPS: {(2 * M * N * K) / avg_time_us / 1e6:.2f}")


if __name__ == "__main__":
    run_sgemm()