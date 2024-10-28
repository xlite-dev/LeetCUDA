# HGEMM 

## 0x00 说明

包含以下内容：

- [X] hgemm_sliced_k_f16_kernel 
- [X] hgemm_t_8x8_sliced_k_f16x4_kernel(unpack)
- [X] hgemm_t_8x8_sliced_k_f16x4_pack_kernel(pack 16x4)
- [X] hgemm_t_8x8_sliced_k_f16x4_bcf_kernel(bank conflicts reduce)
- [X] hgemm_t_8x8_sliced_k_f16x4_pack_bcf_kernel(bank conflicts reduce, pack)
- [X] hgemm_t_8x8_sliced_k_f16x8_pack_bcf_kernel(bank conflicts reduce, pack)
- [X] hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf_kernel(bank conflicts reduce, pack, double buffers)
- [X] hgemm_t_8x8_sliced_k16/32_f16x8_pack_bcf_dbuf_kernel(pack, double buffers)
- [X] hgemm_t_8x8_sliced_k16/32_f16x8_pack_bcf_dbuf_async_kernel(pack, double buffers, copy async)
- [X] hgemm_wmma_m16n16k16_naive(WMMA) 
- [X] hgemm_wmma_m16n16k16_mma4x2(WMMA, Tile MMA) 
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4(TWMMA, Tile MMA/Warp, pack) 
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_async(WMMA, Tile MMA/Warp, Copy Async) 
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_async_offset(WMMA, Tile MMA/Warp, Copy Async, Pad)
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(WMMA, Tile MMA/Warp, Copy Async, Double Buffers, Pad)  
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages(WMMA, Tile MMA/Warp, Copy Async, Stages, Pad, Block swizzle) 
- [X] hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages(WMMA, Tile MMA/Warp, Copy Async, Stages, Pad, Block swizzle)
- [X] hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages(WMMA, Tile MMA/Warp, Copy Async, Stages, Pad, Block swizzle) 
- [X] hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async(WMMA, Tile MMA/Warp, Copy Async, Double Buffers, Pad)
- [X] hgemm_mma_m16n8k16_naive(MMA)
- [X] hgemm_mma_m16n8k16_mma2x4_warp4x4(MMA, Tile MMA/Warp, pack)
- [X] hgemm_mma_m16n8k16_mma2x4_warp4x4_stages(MMA, Tile MMA/Warp, Copy Async, Stages, Pad, Block swizzle)
- [X] hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages(MMA, Tile MMA/Warp, Copy Async, Stages, Pad, Block swizzle, Warp swizzle, Reg Double Buffers, Collective Store with Reg Reuse & Warp Shuffle) 
- [X] PyTorch bindings

## 目前性能  

- NVIDIA L20  

目前最优的实现，在L20上（理论Tensor Cores FP16算力为 119.5 TFLOPS），使用WMMA API能达到cuBLAS大概95%~98%左右的性能(105-113 TFLOPS vs 105-115 TFLOPS)，使用MMA API能达到115 TFLOPS，部分case会超越cuBLAS。已知问题为bank conflicts没有完全消除，目前通过padding的方式缓解bank conflicts会导致shared memory浪费，也会影响SM occupancy。并且尚未手工实现smem swizzle/permute(受限于WMMA API的灵活性以及row major的layout)，后续将会尝试通过MMA PTX和col major的layout实现smem swizzle/permute，[点击查看性能数据](#NV-L20)。

- NVIDIA GeForce RTX 3080 Laptop   

在NVIDIA GeForce RTX 3080 Laptop上测试，使用mma4x4_warp4x4（16 WMMA m16n16k16 ops, warp tile 64x64）以及Thread block swizzle，大部分case能持平甚至超过cuBLAS，[点击查看性能数据](#NV-RTX-3080)。

## 共享内存 Bank Conflicts

含义：在访问shared memory时，因多个线程读写同一个Bank中的不同数据地址时，导致shared memory 并发读写 退化 成顺序读写的现象叫做Bank Conflict；

![](https://github.com/PaddleJitLab/CUDATutorial/blob/develop/docs/09_optimize_reduce/02_bank_conflict/images/ef322be7c3e5b6b9be69d2b90e88083f50569a58a97129f348e483b946ab4edf.png)

SM调度单位为一个warp（一个warp内32个Thread），shared_memory 可以 被一个warp中的所有（32个）线程进行访问，shared_memory 映射到大小相等的32个Bank上，Bank的数据读取带宽为32bit / cycle (4 bytes)，因此，主要需要考虑一个Warp内32线程的访问共享内存时的bank冲突。
对于多个线程读取同一个Bank数据时（不同地址），硬件把内存读写请求，拆分成 conflict-free requests，进行顺序读写，此时将会触发多次内存事务。特别地，当一个warp中的所有线程读写同一个地址时，会触发broadcast机制，此时不会退化成顺序读写。上面提到触发broadcast机制的条件是all threads acess same address，但在翻阅cuda-c-programming-guide以及最新版本的[NVProfGuide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) 时，发现只要是多个thread 读写就会触发broadcast（不需要All）。
  
- 多个线程读同一个数据时，仅有一个线程读，然后broadcast到其他线程
- 多个线程写同一个数据时，仅会有一个线程写成功

NVIDIA的[文章](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)中指出，我们还可以通过 `cudaDeviceSetSharedMemConfig()` 函数设置默认Bank Size（默认为4 bytes）来避免bank conflicts，可设置为cudaSharedMemBankSizeFourByte或者cudaSharedMemBankSizeEightByte。对于某些场景来说，设置cudaSharedMemBankSizeEightByte或许更加合适，比如使用double数据类型时。 

```C
cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
```

## 双缓冲 Double Buffers

本仓库实现的HGEMM Double Buffers策略如下：1）主循环从bk = 1 开始，第一次数据加载在主循环之前，最后一次计算在主循环之后，这是pipeline 的特点决定的；2）由于计算和下一次访存使用的Shared Memory不同，因此主循环中每次循环只需要一次__syncthreads()即可，对比非double buffers版本，总共节省了 ((K + BK - 1) / BK) - 1 次block内的同步操作。比如，bk=1时，HFMA计算使用的是s_a[0]和s_b[0]，因此，和s_a[1]和s_b[1]的加载是没有依赖关系的。HFMA计算，从global内存到s_a[1]和s_b[1]和HFMA计算可以并行。s_a[1]和s_b[1]用于加载下一块BK需要的数据到共享内存；3）由于GPU不能向CPU那样支持乱序执行，主循环中需要先将下一次循环计算需要的Gloabal Memory中的数据load 到寄存器，然后进行本次计算，之后再将load到寄存器中的数据写到Shared Memory，这样在LDG指令向Global Memory做load时，不会影响后续HFMA及其它运算指令的 launch 执行，也就达到了Double Buffers的目的。

```C
  // bk = 0 is loading here, buffer 0
  {
    int load_a_gmem_k = load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    LDST64BITS(r_load_a[0]) = LDST64BITS(a[load_a_gmem_addr]);
    LDST64BITS(r_load_b[0]) = LDST64BITS(b[load_b_gmem_addr]);

    s_a[0][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
    s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    LDST64BITS(s_b[0][load_b_smem_k][load_b_smem_n]) = LDST64BITS(r_load_b[0]);
  }
  // Without this synchronization, accuracy may occasionally be abnormal.
  __syncthreads(); 
  
  // bk start from 1，需要注意的是，虽然 bk 从 1 开始，但实际上 bk=1时，使用的是
  // 第0块BK中的数据（已经加载到共享内存s_a[0]和s_b[0]）；bk=2时，实际计算的是第1块
  // BK中的数据。其余以此类推，这个循环结束后，剩下最后一块BK大小的数据需要计算。
  for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {

    int smem_sel = (bk - 1) & 1; // bk 1->0, bk 2->1, bk 3->0, ...
    int smem_sel_next = bk & 1;  // bk 1->1, bk 2->0, bk 3->1, ...

    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    LDST64BITS(r_load_a[0]) = LDST64BITS(a[load_a_gmem_addr]);
    LDST64BITS(r_load_b[0]) = LDST64BITS(b[load_b_gmem_addr]);
    
    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[smem_sel][tk][ty * TM]);
      LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[smem_sel][tk][tx * TN]);

      #pragma unroll
      for (int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }

    // 对比非double buffers版本，此处不需要__syncthreads()，总共节省了
    // ((K + BK - 1) / BK) - 1 次block内的同步操作。比如，bk=1时，HFMA计算
    // 使用的是s_a[0]和s_b[0]，因此，和s_a[1]和s_b[1]的加载是没有依赖关系的。
    // 从global内存到s_a[1]和s_b[1]和HFMA计算可以并行。s_a[1]和s_b[1]用于
    // 加载下一块BK需要的数据到共享内存。
    s_a[smem_sel_next][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
    s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    LDST128BITS(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = LDST128BITS(r_load_b[0]);

    __syncthreads();
  }
  
  // 计算剩下最后一块BK
  #pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[1][tk][ty * TM]);
    LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[1][tk][tx * TN]);

    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
      #pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
      }
    }
  }

```

## PyTorch HGEMM Profile

在Ada架构下，PyTorch 2.4对FP16使用matmul时，会调用ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn kernel，内部实际使用HMMA(Tensor Cores)进行计算。

```bash
ncu -o hgemm.prof -f python3 prof.py
nsys profile --stats=true -t cuda,osrt,nvtx -o hgemm.prof --force-overwrite true python3 prof.py
```
- 日志

```bash
==PROF== Connected to process 367502 (/usr/bin/python3.10)
==PROF== Profiling "unrolled_elementwise_kernel" - 0: 0%....50%....100% - 8 passes
==PROF== Profiling "unrolled_elementwise_kernel" - 1: 0%....50%....100% - 8 passes
==PROF== Profiling "unrolled_elementwise_kernel" - 2: 0%....50%....100% - 8 passes
==PROF== Profiling "ampere_fp16_s1688gemm_fp16_12..." - 3: 0%....50%....100% - 8 passes
==PROF== Profiling "ampere_fp16_s1688gemm_fp16_12..." - 4: 0%....50%....100% - 8 passes
==PROF== Profiling "ampere_fp16_s1688gemm_fp16_12..." - 5: 0%....50%....100% - 8 passes
==PROF== Profiling "ampere_fp16_s1688gemm_fp16_12..." - 6: 0%....50%....100% - 8 passes
==PROF== Profiling "ampere_fp16_s1688gemm_fp16_12..." - 7: 0%....50%....100% - 8 passes
==PROF== Profiling "ampere_fp16_s1688gemm_fp16_12..." - 8: 0%....50%....100% - 8 passes
==PROF== Profiling "ampere_fp16_s1688gemm_fp16_12..." - 9: 0%....50%....100% - 8 passes
==PROF== Profiling "ampere_fp16_s1688gemm_fp16_12..." - 10: 0%....50%....100% - 8 passes
==PROF== Profiling "ampere_fp16_s1688gemm_fp16_12..." - 11: 0%....50%....100% - 8 passes
==PROF== Profiling "ampere_fp16_s1688gemm_fp16_12..." - 12: 0%....50%....100% - 8 passes
==PROF== Profiling "ampere_fp16_s1688gemm_fp16_12..." - 13: 0%....50%....100% - 8 passes
```

- SASS (L20)

```C
// ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn_kernel
310	00007f41 37d5b850	      LDSM.16.M88.4 R192, [R169+UR8+0x2000] 
311	00007f41 37d5b860	      LDSM.16.M88.4 R196, [R169+UR8+0x2800] 
312	00007f41 37d5b870	@!P0  BRA.U 0x7f4137d5c3f0 
313	00007f41 37d5b880	      HMMA.1688.F32 R0, R176, R192, R0 
314	00007f41 37d5b890	      LDSM.16.MT88.4 R184, [R167+UR8+0x400] 
315	00007f41 37d5b8a0	      HMMA.1688.F32 R32, R178, R192, R32 
316	00007f41 37d5b8b0	      LDSM.16.M88.4 R200, [R170+UR8+0x2000] 
317	00007f41 37d5b8c0	      HMMA.1688.F32 R64, R180, R192, R64 
318	00007f41 37d5b8d0	      LDSM.16.MT88.4 R188, [R168+UR8+0x400] 
319	00007f41 37d5b8e0	      HMMA.1688.F32 R96, R182, R192, R96 
320	00007f41 37d5b8f0	      LDSM.16.M88.4 R204, [R170+UR8+0x2800] 
321	00007f41 37d5b900	      HMMA.1688.F32 R100, R182, R193, R100 
322	00007f41 37d5b910	      HMMA.1688.F32 R68, R180, R193, R68 
323	00007f41 37d5b920	      HMMA.1688.F32 R36, R178, R193, R36 
324	00007f41 37d5b930	      HMMA.1688.F32 R4, R176, R193, R4 
325	00007f41 37d5b940	      HMMA.1688.F32 R8, R176, R194, R8 
326	00007f41 37d5b950	      HMMA.1688.F32 R40, R178, R194, R40 
327	00007f41 37d5b960	      HMMA.1688.F32 R72, R180, R194, R72 
328	00007f41 37d5b970	      HMMA.1688.F32 R104, R182, R194, R104 
329	00007f41 37d5b980	      HMMA.1688.F32 R108, R182, R195, R108 
330	00007f41 37d5b990	      HMMA.1688.F32 R76, R180, R195, R76 
331	00007f41 37d5b9a0	      HMMA.1688.F32 R44, R178, R195, R44 
332	00007f41 37d5b9b0	      HMMA.1688.F32 R12, R176, R195, R12 
333	00007f41 37d5b9c0	      HMMA.1688.F32 R16, R176, R196, R16 
334	00007f41 37d5b9d0	      HMMA.1688.F32 R48, R178, R196, R48 
335	00007f41 37d5b9e0	      HMMA.1688.F32 R80, R180, R196, R80 
336	00007f41 37d5b9f0	      HMMA.1688.F32 R112, R182, R196, R112 
337	00007f41 37d5ba00	      HMMA.1688.F32 R116, R182, R197, R116 
```
- SASS (RTX 3080)

```C
// sm80_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize96x64x32_stage3_warpsize2x2x1_tensor16x8x16_kernel
341	00000007 44ff6340	      HMMA.16816.F32 R12, R72, R80, R12 
342	00000007 44ff6350	      HMMA.16816.F32 R16, R72, R82, R16 
343	00000007 44ff6360	      HMMA.16816.F32 R20, R84, R76, R20 
344	00000007 44ff6370	      LDSM.16.M88.4 R52, [R92+UR8] 
345	00000007 44ff6380	      HMMA.16816.F32 R24, R84, R78, R24 
346	00000007 44ff6390	      LDSM.16.M88.4 R64, [R92+UR8+0x800] 
347	00000007 44ff63a0	      HMMA.16816.F32 R28, R84, R80, R28 
348	00000007 44ff63b0	      LDSM.16.M88.4 R68, [R92+UR8+0x1000] 
349	00000007 44ff63c0	      HMMA.16816.F32 R32, R84, R82, R32 
350	00000007 44ff63d0	      LDSM.16.MT88.4 R56, [R3+UR7+0x4800] 
351	00000007 44ff63e0	      HMMA.16816.F32 R36, R88, R76, R36 
352	00000007 44ff63f0	      LDSM.16.MT88.4 R60, [R106+UR7+0x4800] 
353	00000007 44ff6400	      HMMA.16816.F32 R40, R88, R78, R40 
354	00000007 44ff6410	      HMMA.16816.F32 R44, R88, R80, R44 
355	00000007 44ff6420	      HMMA.16816.F32 R48, R88, R82, R48 
```


## 参考文献 

- [CUDA编程概念】一、什么是bank conflict？](https://zhuanlan.zhihu.com/p/659142274)
- [解决 bank conflict](https://github.com/PaddleJitLab/CUDATutorial/blob/develop/docs/09_optimize_reduce/02_bank_conflict/README.md)
- [Bank Conflict free 的几种方式](https://zhuanlan.zhihu.com/p/722286440)
- [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [CUDA（三）：通用矩阵乘法：从入门到熟练](https://zhuanlan.zhihu.com/p/657632577)

## 测试命令

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长: Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada 
python3 hgemm.py --wmma # test defalut wmma kernels for all MNK
python3 hgemm.py --mma  # test defalut mma kernels for all MNK
python3 hgemm.py --M 16384 --N 16384 --K 8192 --i 10 --wmma # test default wmma kernels for specific MNK
python3 hgemm.py --M 16384 --N 16384 --K 8192 --i 10 --mma # test default mma kernels for specific MNK
python3 hgemm.py --wmma-all # test all wmma kernels for all MNK
python3 hgemm.py --mma-all # test all mma kernels for all MNK
python3 hgemm.py --cuda-all --wmma-all --mma-all # test all kernels for all MNK
```
以下测试示例数据机器为NVIDIA L20。

- 示例1:
```bash
python3 hgemm.py --M 16384 --N 16384 --K 8192 --i 10 --mma
Namespace(M=16384, N=16384, K=8192, warmup=2, iters=10, verbose=False, reduce_reg=False, show_matrix=False, show_all_info=False, enable_mma=True, enable_mma_tn=False, enable_wmma=False, enable_cuda=False, enable_mma_all=False, enable_wmma_all=False, enable_cuda_all=False, enable_torch=False, disable_cublas=False, disable_cublas_tn=False, sleep_duration=0.1, swizzle_factor=None)
Loading hgemm lib ...
pre allocate for fast profiling start, MAX_M=16384, MAX_N=16384, MAX_K=8192
pre allocate for fast profiling done, time: 21777.246236801147 ms
----------------------------------------------------------------------------------------------------------------------------------
                                        M=16384, N=16384, K=8192, Warmup=2, Iters=10, 1/1
----------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------MMA-----------------------------------------------------------
                          (mma2x4+warp4x4): ['-135.5    ', '-20.21875 '], time:45.22497ms, swizzle: NOOP, TFLOPS: 97.25 (+0.00%)
           (mma2x4+warp4x4+stage3+swizzle): ['-135.5    ', '-20.21875 '], time:39.16373ms, swizzle: 4096, TFLOPS: 112.30(+15.48%)
           (mma2x4+warp4x4+stage2+swizzle): ['-135.5    ', '-20.21875 '], time:38.40122ms, swizzle: 4096, TFLOPS: 114.53(+1.99%)
     (mma2x4+warp4x4+stage2+dsmem+swizzle): ['-135.5    ', '-20.21875 '], time:38.20776ms, swizzle: 4096, TFLOPS: 115.11(+0.51%)
                                  (cublas): ['-135.5    ', '-20.21875 '], time:37.60526ms, swizzle: NOOP, TFLOPS: 116.95(+1.60%)
----------------------------------------------------------------------------------------------------------------------------------
```
- 示例2:
```bash
python3 hgemm.py --M 4096 --N 4096 --K 4096 --mma-all
Namespace(M=4096, N=4096, K=4096, warmup=2, iters=10, verbose=False, reduce_reg=False, show_matrix=False, show_all_info=False, enable_mma=False, enable_mma_tn=False, enable_wmma=False, enable_cuda=False, enable_mma_all=True, enable_wmma_all=False, enable_cuda_all=False, enable_torch=False, disable_cublas=False, disable_cublas_tn=False, sleep_duration=0.1, swizzle_factor=None)
Loading hgemm lib ...
pre allocate for fast profiling start, MAX_M=4096, MAX_N=4096, MAX_K=4096
pre allocate for fast profiling done, time: 2052.9532432556152 ms
----------------------------------------------------------------------------------------------------------------------------------
                                        M=4096, N=4096, K=4096, Warmup=2, Iters=10, 1/1
----------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------MMA-----------------------------------------------------------
                          (mma2x4+warp4x4): ['-64.6875  ', '-33.90625 '], time:1.413774ms, swizzle: NOOP, TFLOPS: 97.21 (+0.00%)
                   (mma2x4+warp4x4+stage3): ['-64.6875  ', '-33.90625 '], time:1.343679ms, swizzle: NOOP, TFLOPS: 102.29(+5.22%)
                   (mma2x4+warp4x4+stage2): ['-64.6875  ', '-33.90625 '], time:1.326227ms, swizzle: NOOP, TFLOPS: 103.63(+1.32%)
           (mma2x4+warp4x4x2+stage4+dsmem): ['-64.6875  ', '-33.90625 '], time:1.324486ms, swizzle: NOOP, TFLOPS: 103.77(+0.13%)
           (mma2x4+warp4x4x2+stage2+dsmem): ['-64.6875  ', '-33.90625 '], time:1.299619ms, swizzle: NOOP, TFLOPS: 105.75(+1.91%)
        (mma2x4+warp4x4x2+stage2+dsmem+rr): ['-64.6875  ', '-33.90625 '], time:1.299595ms, swizzle: NOOP, TFLOPS: 105.76(+0.00%)
                                  (cublas): ['-64.6875  ', '-33.90625 '], time:1.289343ms, swizzle: NOOP, TFLOPS: 106.60(+0.80%)
----------------------------------------------------------------------------------------------------------------------------------
```
- 示例3：
```bash
python3 hgemm.py --M 4096 --N 4096 --K 4096 --mma-all --wmma-all --cuda-all
Namespace(M=4096, N=4096, K=4096, warmup=2, iters=10, verbose=False, reduce_reg=False, show_matrix=False, show_all_info=False, enable_mma=False, enable_mma_tn=False, enable_wmma=False, enable_cuda=False, enable_mma_all=True, enable_wmma_all=True, enable_cuda_all=True, enable_torch=False, disable_cublas=False, disable_cublas_tn=False, sleep_duration=0.1, swizzle_factor=None)
Loading hgemm lib ...
pre allocate for fast profiling start, MAX_M=4096, MAX_N=4096, MAX_K=4096
pre allocate for fast profiling done, time: 2046.0460186004639 ms
----------------------------------------------------------------------------------------------------------------------------------
                                        M=4096, N=4096, K=4096, Warmup=2, Iters=10, 1/1
----------------------------------------------------------------------------------------------------------------------------------
                                   (naive): ['55.40625  ', '-128.625  '], time:37.67290ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                      (f16x8pack+t8x8+bcf): ['55.40625  ', '-128.625  '], time:2.820539ms, swizzle: NOOP, TFLOPS: 48.73 (+1235.66%)
                     (f16x8pack+t8x8+dbuf): ['55.40625  ', '-128.625  '], time:2.819323ms, swizzle: NOOP, TFLOPS: 48.75 (+0.04%)
                 (f16x8pack+t8x8+k16+dbuf): ['55.40625  ', '-128.625  '], time:2.628445ms, swizzle: NOOP, TFLOPS: 52.29 (+7.26%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         (wmma4x2+warp2x4): ['56.03125  ', '-129.125  '], time:1.816916ms, swizzle: NOOP, TFLOPS: 75.64 (+44.67%)
                  (wmma4x2+warp2x4+stage3): ['56.03125  ', '-129.125  '], time:1.354646ms, swizzle: NOOP, TFLOPS: 101.46(+34.12%)
                  (wmma4x2+warp2x4+stage2): ['56.03125  ', '-129.125  '], time:1.343059ms, swizzle: NOOP, TFLOPS: 102.33(+0.86%)
            (wmma4x2+warp2x4+stage3+dsmem): ['56.03125  ', '-129.125  '], time:1.342296ms, swizzle: NOOP, TFLOPS: 102.39(+0.06%)
          (wmma4x2+warp2x4+stage2+swizzle): ['56.03125  ', '-129.125  '], time:1.323080ms, swizzle: 2048, TFLOPS: 103.88(+1.45%)
--------------------------------------------------------------------MMA-----------------------------------------------------------
           (mma2x4+warp4x4x2+stage2+dsmem): ['56.03125  ', '-129.125  '], time:1.299333ms, swizzle: NOOP, TFLOPS: 105.78(+1.83%)
                                  (cublas): ['56.03125  ', '-129.125  '], time:1.289367ms, swizzle: NOOP, TFLOPS: 106.59(+0.77%)
----------------------------------------------------------------------------------------------------------------------------------
```

## NVIDIA L20 
<div id="NV-L20"></div>


### WMMA & CUDA: Up to 113.76 TFLOPS, 113.76/119.5=95.19% TFLOPS utilization.

```bash
python3 hgemm.py --cuda --wmma
```
输出：
```bash
----------------------------------------------------------------------------------------------------------------------------------
                                        M=16384, N=16384, K=4096, Warmup=2, Iters=10, 124/125
----------------------------------------------------------------------------------------------------------------------------------
                     (f16x8pack+t8x8+dbuf): ['53.34375  ', '42.9375   '], time:48.26700ms, swizzle: NOOP, TFLOPS: 45.56 (+0.00%)
                 (f16x8pack+t8x8+k16+dbuf): ['53.34375  ', '42.9375   '], time:47.47912ms, swizzle: NOOP, TFLOPS: 46.32 (+1.66%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         (wmma4x2+warp2x4): ['52.75     ', '42.625    '], time:28.30018ms, swizzle: NOOP, TFLOPS: 77.70 (+67.77%)
                  (wmma4x2+warp2x4+stage3): ['52.75     ', '42.625    '], time:21.76406ms, swizzle: NOOP, TFLOPS: 101.04(+30.03%)
                  (wmma4x2+warp2x4+stage2): ['52.75     ', '42.625    '], time:21.32434ms, swizzle: NOOP, TFLOPS: 103.12(+2.06%)
            (wmma4x2+warp2x4+stage2+dsmem): ['52.75     ', '42.625    '], time:21.21436ms, swizzle: NOOP, TFLOPS: 103.66(+0.52%)
          (wmma4x2+warp2x4+stage3+swizzle): ['52.75     ', '42.625    '], time:19.75061ms, swizzle: 4096, TFLOPS: 111.34(+7.41%)
          (wmma4x2+warp2x4+stage2+swizzle): ['52.75     ', '42.625    '], time:19.44730ms, swizzle: 4096, TFLOPS: 113.08(+1.56%)
                                  (cublas): ['52.75     ', '42.625    '], time:18.94822ms, swizzle: NOOP, TFLOPS: 116.05(+2.63%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=16384, N=16384, K=8192, Warmup=2, Iters=10, 125/125
----------------------------------------------------------------------------------------------------------------------------------
                     (f16x8pack+t8x8+dbuf): ['76.375    ', '71.875    '], time:104.0809ms, swizzle: NOOP, TFLOPS: 42.26 (+0.00%)
                 (f16x8pack+t8x8+k16+dbuf): ['76.375    ', '71.875    '], time:101.2542ms, swizzle: NOOP, TFLOPS: 43.44 (+2.79%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         (wmma4x2+warp2x4): ['75.5      ', '71.8125   '], time:55.96892ms, swizzle: NOOP, TFLOPS: 78.58 (+80.91%)
                  (wmma4x2+warp2x4+stage3): ['75.5      ', '71.8125   '], time:49.65238ms, swizzle: NOOP, TFLOPS: 88.58 (+12.72%)
          (wmma4x2+warp2x4+stage3+swizzle): ['75.5      ', '71.8125   '], time:39.22250ms, swizzle: 4096, TFLOPS: 112.13(+26.59%)
          (wmma4x2+warp2x4+stage2+swizzle): ['75.5      ', '71.8125   '], time:38.70298ms, swizzle: 4096, TFLOPS: 113.64(+1.34%)
                                  (cublas): ['75.5      ', '71.8125   '], time:38.01751ms, swizzle: NOOP, TFLOPS: 115.68(+1.80%)
----------------------------------------------------------------------------------------------------------------------------------
```

### MMA: Up to 115 TFLOPS, 115/119.5=96.23% TFLOPS utilization.

```bash
python3 hgemm.py --mma-all
```

输出：
```bash
----------------------------------------------------------------------------------------------------------------------------------
                                        M=16384, N=1024, K=2048, Warmup=2, Iters=10, 103/125
----------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------MMA-----------------------------------------------------------
                          (mma2x4+warp4x4): ['49.34375  ', '-13.335937'], time:0.723743ms, swizzle: NOOP, TFLOPS: 94.95 (+0.00%)
                   (mma2x4+warp4x4+stage3): ['49.34375  ', '-13.335937'], time:0.686883ms, swizzle: NOOP, TFLOPS: 100.05(+5.37%)
                   (mma2x4+warp4x4+stage2): ['49.34375  ', '-13.335937'], time:0.678110ms, swizzle: NOOP, TFLOPS: 101.34(+1.29%)
             (mma2x4+warp4x4+stage2+dsmem): ['49.34375  ', '-13.335937'], time:0.678014ms, swizzle: NOOP, TFLOPS: 101.35(+0.01%)
           (mma2x4+warp4x4x2+stage2+dsmem): ['49.34375  ', '-13.335937'], time:0.663757ms, swizzle: NOOP, TFLOPS: 103.53(+2.15%)
        (mma2x4+warp4x4x2+stage2+dsmem+rr): ['49.34375  ', '-13.335937'], time:0.663447ms, swizzle: NOOP, TFLOPS: 103.58(+0.05%)
   (mma2x4+warp4x4x2+stage2+dsmem+swizzle): ['49.34375  ', '-13.335937'], time:0.662875ms, swizzle: 512 , TFLOPS: 103.67(+0.09%)
(mma2x4+warp4x4x2+stage2+dsmem+swizzle+rr): ['49.34375  ', '-13.335937'], time:0.662755ms, swizzle: 512 , TFLOPS: 103.69(+0.02%)
                                  (cublas): ['49.34375  ', '-13.335937'], time:0.680470ms, swizzle: NOOP, TFLOPS: 100.99
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=16384, N=1024, K=4096, Warmup=2, Iters=10, 104/125
----------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------MMA-----------------------------------------------------------
                          (mma2x4+warp4x4): ['66.8125   ', '-76.375   '], time:1.451849ms, swizzle: NOOP, TFLOPS: 94.66 (+0.00%)
                   (mma2x4+warp4x4+stage3): ['66.8125   ', '-76.375   '], time:1.347494ms, swizzle: NOOP, TFLOPS: 102.00(+7.74%)
                   (mma2x4+warp4x4+stage2): ['66.8125   ', '-76.375   '], time:1.331448ms, swizzle: NOOP, TFLOPS: 103.23(+1.21%)
           (mma2x4+warp4x4x2+stage4+dsmem): ['66.8125   ', '-76.375   '], time:1.327109ms, swizzle: NOOP, TFLOPS: 103.56(+0.33%)
           (mma2x4+warp4x4x2+stage2+dsmem): ['66.8125   ', '-76.375   '], time:1.306390ms, swizzle: NOOP, TFLOPS: 105.21(+1.59%)
                                  (cublas): ['66.8125   ', '-76.375   '], time:1.316022ms, swizzle: NOOP, TFLOPS: 104.44
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=4096, N=4096, K=4096, Warmup=2, Iters=10, 1/1
----------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------MMA-----------------------------------------------------------
                          (mma2x4+warp4x4): ['-64.6875  ', '-33.90625 '], time:1.413774ms, swizzle: NOOP, TFLOPS: 97.21 (+0.00%)
                   (mma2x4+warp4x4+stage3): ['-64.6875  ', '-33.90625 '], time:1.343679ms, swizzle: NOOP, TFLOPS: 102.29(+5.22%)
                   (mma2x4+warp4x4+stage2): ['-64.6875  ', '-33.90625 '], time:1.326227ms, swizzle: NOOP, TFLOPS: 103.63(+1.32%)
           (mma2x4+warp4x4x2+stage4+dsmem): ['-64.6875  ', '-33.90625 '], time:1.324486ms, swizzle: NOOP, TFLOPS: 103.77(+0.13%)
           (mma2x4+warp4x4x2+stage2+dsmem): ['-64.6875  ', '-33.90625 '], time:1.299619ms, swizzle: NOOP, TFLOPS: 105.75(+1.91%)
        (mma2x4+warp4x4x2+stage2+dsmem+rr): ['-64.6875  ', '-33.90625 '], time:1.299595ms, swizzle: NOOP, TFLOPS: 105.76(+0.00%)
                                  (cublas): ['-64.6875  ', '-33.90625 '], time:1.289343ms, swizzle: NOOP, TFLOPS: 106.60(+0.80%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=16384, N=16384, K=8192, Warmup=2, Iters=10, 1/1
----------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------MMA-----------------------------------------------------------
                          (mma2x4+warp4x4): ['-135.5    ', '-20.21875 '], time:45.22497ms, swizzle: NOOP, TFLOPS: 97.25 (+0.00%)
           (mma2x4+warp4x4+stage3+swizzle): ['-135.5    ', '-20.21875 '], time:39.16373ms, swizzle: 4096, TFLOPS: 112.30(+15.48%)
           (mma2x4+warp4x4+stage2+swizzle): ['-135.5    ', '-20.21875 '], time:38.40122ms, swizzle: 4096, TFLOPS: 114.53(+1.99%)
     (mma2x4+warp4x4+stage2+dsmem+swizzle): ['-135.5    ', '-20.21875 '], time:38.20776ms, swizzle: 4096, TFLOPS: 115.11(+0.51%)
                                  (cublas): ['-135.5    ', '-20.21875 '], time:37.60526ms, swizzle: NOOP, TFLOPS: 116.95(+1.60%)
----------------------------------------------------------------------------------------------------------------------------------
```


## NVIDIA GeForce RTX 3080 Laptop 
<div id="NV-RTX-3080"></div>

- WMMA

```bash
python3 hgemm.py --wmma-all
```
输出：
```bash
----------------------------------------------------------------------------------------------------------------------------------
                              M=4096, N=4096, K=2048, Warmup=5, Iters=20, 1/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['-34.9375  ', '2.25585938'], time:1.397085ms, swizzle: NOOP, TFLOPS: 49.19 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['-34.9375  ', '2.25585938'], time:1.632452ms, swizzle: NOOP, TFLOPS: 42.10
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['-34.9375  ', '2.25585938'], time:1.392316ms, swizzle: 1024, TFLOPS: 49.36 (+0.34%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['-34.9375  ', '2.25585938'], time:1.537656ms, swizzle: 1024, TFLOPS: 44.69
                                (cublas): ['-34.90625 ', '2.21875   '], time:1.072788ms, swizzle: NOOP, TFLOPS: 64.06 (+29.78%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=16384, N=16384, K=8192, Warmup=5, Iters=20, 27/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['68.375    ', '-2.234375 '], time:96.91984ms, swizzle: NOOP, TFLOPS: 45.38 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['68.375    ', '-2.234375 '], time:102.8722ms, swizzle: NOOP, TFLOPS: 42.75
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['68.375    ', '-2.234375 '], time:85.65800ms, swizzle: 4096, TFLOPS: 51.34 (+13.15%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['68.375    ', '-2.234375 '], time:95.70884ms, swizzle: 4096, TFLOPS: 45.95
                                (cublas): ['68.375    ', '-2.234375 '], time:104.2092ms, swizzle: NOOP, TFLOPS: 42.20
----------------------------------------------------------------------------------------------------------------------------------
```
