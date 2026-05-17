import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.cute.testing as testing

import torch

num_stages = 3
cta_tiler = (128, 128, 8)
mma_tiler = (16, 16) 


VERBOSE = True
LOG = "[Info]"

@cute.kernel
def sgemm_kernel(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    smem_layout_a: cute.Layout,
    smem_layout_b: cute.Layout,
    tiled_copy_a: cute.TiledCopy,
    tiled_copy_b: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
    epilogue_op: cutlass.Constexpr = lambda x: x,
):
    tid_x, _, _ = cute.arch.thread_idx()
    bid_x, bid_y, _ = cute.arch.block_idx()


    tiler_coord = (bid_x, bid_y, None)

    gA = cute.local_tile(A, cta_tiler, tiler_coord, proj=(1, None, 1))
    gB = cute.local_tile(B, cta_tiler, tiler_coord, proj=(None, 1, 1))
    gC = cute.local_tile(C, cta_tiler, tiler_coord, proj=(1, 1, None))

    if VERBOSE:
        print(f"{LOG} gA {gA.type}")
        print(f"{LOG} gB {gB.type}")
        print(f"{LOG} gC {gC.type}")

    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(cutlass.Float32, smem_layout_a, 16)
    sB = smem.allocate_tensor(cutlass.Float32, smem_layout_b, 16)
    thr_copy_a = tiled_copy_a.get_slice(tid_x)
    thr_copy_b = tiled_copy_b.get_slice(tid_x)

    tAgA = thr_copy_a.partition_S(gA)
    tAsA = thr_copy_a.partition_D(sA)
    tBgB = thr_copy_b.partition_S(gB)
    tBsB = thr_copy_b.partition_D(sB)

    if VERBOSE:
        print(f"{LOG} tAgA {tAgA}")
        print(f"{LOG} tAsA {tAsA}")
        print(f"{LOG} tBgB {tBgB}")
        print(f"{LOG} tBsB {tBsB}")

    thr_mma = tiled_mma.get_slice(tid_x)


    # ==========================================================================
    # prologue
    # --------------------------------------------------------------------------
    smem_pipe_depth = cute.size(tAsA, mode=[3]) # num stages = 3
    num_tiles_k = cute.size(tAgA, mode=[3]) # num of k tiles = 256
    gmem_pipe_read = cutlass.Int32(0)

    for pipe in range(0, smem_pipe_depth - 1, unroll_full=True):
        cute.copy(
            tiled_copy_a,
            tAgA[None, None, None, gmem_pipe_read],
            tAsA[None, None, None, pipe],
        )
        cute.copy(
            tiled_copy_b,
            tBgB[None, None, None, gmem_pipe_read],
            tBsB[None, None, None, pipe],
        )
        cute.arch.cp_async_commit_group()

        gmem_pipe_read += 1


    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0.0)

    if VERBOSE:
        print(f"{LOG} tCsA {tCsA}")
        print(f"{LOG} tCsB {tCsB}")
        print(f"{LOG} tCgC {tCgC}")
        print(f"{LOG} tCrA {tCrA}")
        print(f"{LOG} tCrB {tCrB}")
        print(f"{LOG} tCrC {tCrC}")


    smem_pipe_read = cutlass.Int32(0) # read from stage 0
    smem_pipe_write = cutlass.Int32(smem_pipe_depth - 1) # write into stage 2

    tCsA_ = tCsA[None, None, None, smem_pipe_read]
    tCsB_ = tCsB[None, None, None, smem_pipe_read]

    num_mma_k = cute.size(tCrA, mode=[2]) # 8
    if num_mma_k > 1:
        # Wait until the first prefetched tile is loaded in
        cute.arch.cp_async_wait_group(smem_pipe_depth - 2)
        cute.arch.barrier()
        # Prefetch the first rmem from the first k-tile
        cute.autovec_copy(tCsA_[None, None, 0], tCrA[None, None, 0])
        cute.autovec_copy(tCsB_[None, None, 0], tCrB[None, None, 0])

    # ======================================================================
    # Mainloop.
    for _ in range(num_tiles_k, unroll_full=False):
        for mma_k_idx in range(num_mma_k, unroll_full=True):

            # this runs at the last inner step idx = 7, because at that point, the current smem tile is done
            if mma_k_idx == num_mma_k - 1:
                tCsA_ = tCsA[None, None, None, smem_pipe_read]
                tCsB_ = tCsB[None, None, None, smem_pipe_read]
                cute.arch.cp_async_wait_group(smem_pipe_depth - 2)
                cute.arch.barrier()


            mma_k_next = (mma_k_idx + 1) % num_mma_k

            cute.autovec_copy(tCsA_[None, None, mma_k_next], tCrA[None, None, mma_k_next])
            cute.autovec_copy(tCsB_[None, None, mma_k_next], tCrB[None, None, mma_k_next])

            # this runs once per smem tile, right when starting to consume that tile.
            if mma_k_idx == 0:
                cute.copy(
                    tiled_copy_a,
                    tAgA[None, None, None, gmem_pipe_read],
                    tAsA[None, None, None, smem_pipe_write],
                )

            cute.gemm(
                tiled_mma,
                tCrC,
                tCrA[None, None, mma_k_idx],
                tCrB[None, None, mma_k_idx],
                tCrC,
            )

            # same as the A copy above but for B. placed after the gemm, not before.
            # why? to interleave memory and compute instructions
            if mma_k_idx == 0:
                cute.copy(
                    tiled_copy_b,
                    tBgB[None, None, None, gmem_pipe_read],
                    tBsB[None, None, None, smem_pipe_write],
                )

            if mma_k_idx == 0:
                cute.arch.cp_async_commit_group()

                smem_pipe_write = smem_pipe_read
                smem_pipe_read += 1
                if smem_pipe_read >= smem_pipe_depth:
                    smem_pipe_read = cutlass.Int32(0)

                gmem_pipe_read += 1
                if gmem_pipe_read >= num_tiles_k:
                    gmem_pipe_read = cutlass.Int32(1)

    # ===================================================================
    # Epilogue
    cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()
    tCrC.store(epilogue_op(tCrC.load()))

    c_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32)
    cute.copy(c_copy_atom, tCrC, tCgC)

    return
    
@cute.jit
def sgemm_host(a: cute.Tensor, 
               b: cute.Tensor,
               c: cute.Tensor,
               copy_bits: cutlass.Constexpr = 128):
    
    tile_m, tile_n, tile_k = cta_tiler
    mma_m, mma_n = mma_tiler
    threads = mma_m * mma_n


    smem_layout_a = cute.make_layout( # (128,8,3) : (1,132,1056)
        (tile_m, tile_k, num_stages), stride=(1, tile_m, tile_k * tile_m)
    )
    smem_layout_b = cute.make_layout(
        (tile_n, tile_k, num_stages), stride=(1, tile_n, tile_k * tile_n)
    )
    smem_size = sum(
        [cute.size_in_bytes(cutlass.Float32, lo) for lo in [smem_layout_a, smem_layout_b]]
    ) 


    thr_layout_a = cute.make_ordered_layout((threads // tile_k, tile_k), order=(1, 0))
    val_layout_a = cute.make_layout((1, 1))
    async_copy_atom_a = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(),
        cutlass.Float32,
        num_bits_per_copy=a.element_type.width,
    )


    thr_layout_b = cute.make_ordered_layout((threads // tile_k, tile_k), order=(1, 0))
    val_layout_b = cute.make_layout((1, 1))
    async_copy_atom_b = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(),
        cutlass.Float32,
        num_bits_per_copy=b.element_type.width,
    )
    
    tiled_copy_a = cute.make_tiled_copy_tv(async_copy_atom_a, thr_layout_a, val_layout_a)
    tiled_copy_b = cute.make_tiled_copy_tv(async_copy_atom_b, thr_layout_b, val_layout_b)


    mma_op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
    mma_atoms_layout = cute.make_layout((mma_m, mma_n, 1), stride=(mma_n, 1, 0))
    tiled_mma = cute.make_tiled_mma(
        mma_op,
        atom_layout_mnk=mma_atoms_layout,
    )

    grid_dim = [*cute.ceil_div(c.shape, (tile_m, tile_n)), 1]
    block_dim = [threads, 1, 1]

    if cutlass.const_expr(VERBOSE):
        print(f"{LOG} Tensor A {a}")
        print(f"{LOG} Tensor B {b}")
        print(f"{LOG} Tensor C {c}")
        print(f"{LOG} Smem layout A {smem_layout_a}")
        print(f"{LOG} Smem layout B {smem_layout_b}")
        print(f"{LOG} Copy layout A {tiled_copy_a}")
        print(f"{LOG} Copy layout B {tiled_copy_b}")
        print(f"{LOG} Mma layout {tiled_mma}")
        print(f"{LOG} Gemm tile size {cta_tiler}")
        print(f"{LOG} Sgemm grid {grid_dim}")
        print(f"{LOG} Sgemm block {block_dim}")

    sgemm_kernel(
        a,
        b,
        c,
        smem_layout_a,
        smem_layout_b,
        tiled_copy_a,
        tiled_copy_b,
        tiled_mma,
    ).launch(grid = grid_dim, block = block_dim, smem = smem_size)


import time

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
        c = torch.zeros(M, N, device="cuda", dtype=torch.float32)

        b_kernel = b.transpose(0, 1).contiguous()

        a_tensor = from_dlpack(a, assumed_align=16)
        b_tensor = from_dlpack(b_kernel, assumed_align=16)
        c_tensor = from_dlpack(c, assumed_align=16)

        if return_torch_tensor:
            return a, b, c, a_tensor, b_tensor, c_tensor
        return a_tensor, b_tensor, c_tensor

    workspace_generator = lambda: testing.JitArguments(*tensor_generator())

    _a, _b, _c, _a_tensor, _b_tensor, _c_tensor = tensor_generator(return_torch_tensor=True)

    compile_tic = time.perf_counter()
    matmul = cute.compile(sgemm_host, _a_tensor, _b_tensor, _c_tensor, options="--generate-line-info")
    print(f"kernel compiled in {time.perf_counter() - compile_tic:.4f} seconds")

    if verify:
        matmul(_a_tensor, _b_tensor, _c_tensor)
        torch.cuda.synchronize()
        torch.testing.assert_close(_c, torch.matmul(_a, _b), atol=1e-3, rtol=1e-3)
        print("verification passed!")
    else:
        print("verification skipped...")
    
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