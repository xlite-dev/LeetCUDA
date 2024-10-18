# HGEMM 

## 0x00 说明

包含以下内容：

- [X] hgemm_sliced_k_f16_kernel 
- [X] hgemm_t_8x8_sliced_k_f16x4_kernel(unpack)
- [X] hgemm_t_8x8_sliced_k_f16x4_pack_kernel(pack 16x4)
- [X] hgemm_t_8x8_sliced_k_f16x4_bcf_kernel(bank conflicts reduce)
- [X] hgemm_t_8x8_sliced_k_f16x4_pack_bcf_kernel(bank conflicts reduce, pack)
- [X] hgemm_t_4x4_sliced_k_f16x4_pack_bcf_kernel(bank conflicts reduce, pack)
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
- [X] hgemm_wmma_m16n16k16_mma4x4_warp2x2x2_dbuf_async(WMMA, Tile MMA/Warp, Copy Async, Double Buffers, Pad)  
- [X] hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async(WMMA, Tile MMA/Warp, Copy Async, Double Buffers, Pad)
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_rbuf_async(WMMA, Tile MMA/Warp, Copy Async, Double/Reg Buffers, Pad)
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages(WMMA, Tile MMA/Warp, Copy Async, Stages, Pad, Block swizzle) 
- [X] hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages(WMMA, Tile MMA/Warp, Copy Async, Stages, Pad, Block swizzle) 
- [X] PyTorch bindings

## 目前性能  

目前最优的实现，在L20上（理论Tensor Cores FP16算力为 119.5 TFLOPS），能达到cuBLAS大概95%~98%左右的性能(105-113 TFLOPS vs 105-115 TFLOPS)，部分case会超越cuBLAS。已知问题为bank conflicts没有完全消除，目前通过padding的方式缓解bank conflicts会导致shared memory浪费，也会影响SM occupancy。并且尚未手工实现Warp swizzle(受限于WMMA API的灵活性以及本人的能力)，后续将会尝试通过MMA PTX实现warp swizzle。

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

## 测试

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长: Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada 
python3 hgemm.py
```

输出:

- NVIDIA L20  
```bash
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=2048
                    f16x8pack(t8x8+dbuf): ['29.828125 ', '48.15625  '], time:1.408100ms, swizzle: NOOP, TFLOPS: 48.80 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['29.828125 ', '48.15625  '], time:1.331615ms, swizzle: NOOP, TFLOPS: 51.61 (+5.74%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['29.96875  ', '47.9375   '], time:0.700473ms, swizzle: NOOP, TFLOPS: 98.10 (+90.10%)
          f16wmma(mma2x4+warp2x4+stage2): ['29.96875  ', '47.9375   '], time:0.693964ms, swizzle: NOOP, TFLOPS: 99.02 (+0.94%)
        f16wmma(mma2x4+...+stage3+dsmem): ['29.96875  ', '47.9375   '], time:0.694012ms, swizzle: NOOP, TFLOPS: 99.02
        f16wmma(mma2x4+...+stage2+dsmem): ['29.96875  ', '47.9375   '], time:0.693655ms, swizzle: NOOP, TFLOPS: 99.07 (+0.04%)
      f16wmma(mma2x4+...+stage3+swizzle): ['29.96875  ', '47.9375   '], time:0.696635ms, swizzle: 1024, TFLOPS: 98.64
      f16wmma(mma2x4+...+stage2+swizzle): ['29.96875  ', '47.9375   '], time:0.686025ms, swizzle: 1024, TFLOPS: 100.17(+1.11%)
       f16wmma(...+stage3+dsmem+swizzle): ['29.96875  ', '47.9375   '], time:0.697636ms, swizzle: 1024, TFLOPS: 98.50
       f16wmma(...+stage2+dsmem+swizzle): ['29.96875  ', '47.9375   '], time:0.684475ms, swizzle: 1024, TFLOPS: 100.40(+0.23%)
                             f16(cublas): ['29.96875  ', '47.9375   '], time:0.841879ms, swizzle: NOOP, TFLOPS: 81.63
                                  f16_th: ['30.078125 ', '47.75     '], time:0.659370ms, swizzle: NOOP, TFLOPS: 104.22(+3.81%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=4096
                    f16x8pack(t8x8+dbuf): ['21.5      ', '42.78125  '], time:2.800989ms, swizzle: NOOP, TFLOPS: 49.07 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['21.5      ', '42.78125  '], time:2.692627ms, swizzle: NOOP, TFLOPS: 51.04 (+4.02%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['21.75     ', '42.1875   '], time:1.399898ms, swizzle: NOOP, TFLOPS: 98.18 (+92.34%)
          f16wmma(mma2x4+warp2x4+stage2): ['21.75     ', '42.1875   '], time:1.371121ms, swizzle: NOOP, TFLOPS: 100.24(+2.10%)
        f16wmma(mma2x4+...+stage3+dsmem): ['21.75     ', '42.1875   '], time:1.363778ms, swizzle: NOOP, TFLOPS: 100.78(+0.54%)
        f16wmma(mma2x4+...+stage2+dsmem): ['21.75     ', '42.1875   '], time:1.362967ms, swizzle: NOOP, TFLOPS: 100.84(+0.06%)
      f16wmma(mma2x4+...+stage3+swizzle): ['21.75     ', '42.1875   '], time:1.364517ms, swizzle: 1024, TFLOPS: 100.72
      f16wmma(mma2x4+...+stage2+swizzle): ['21.75     ', '42.1875   '], time:1.342916ms, swizzle: 1024, TFLOPS: 102.34(+1.49%)
       f16wmma(...+stage3+dsmem+swizzle): ['21.75     ', '42.1875   '], time:1.369261ms, swizzle: 1024, TFLOPS: 100.37
       f16wmma(...+stage2+dsmem+swizzle): ['21.75     ', '42.1875   '], time:1.343488ms, swizzle: 1024, TFLOPS: 102.30
                             f16(cublas): ['21.75     ', '42.1875   '], time:1.476955ms, swizzle: NOOP, TFLOPS: 93.06
                                  f16_th: ['21.9375   ', '41.9375   '], time:1.296305ms, swizzle: NOOP, TFLOPS: 106.02(+3.60%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=8192
                    f16x8pack(t8x8+dbuf): ['43.3125   ', '-19.578125'], time:5.695152ms, swizzle: NOOP, TFLOPS: 48.27 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['43.3125   ', '-19.578125'], time:5.461049ms, swizzle: NOOP, TFLOPS: 50.33 (+4.29%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['43.6875   ', '-20.21875 '], time:2.767062ms, swizzle: NOOP, TFLOPS: 99.34 (+97.36%)
          f16wmma(mma2x4+warp2x4+stage2): ['43.6875   ', '-20.21875 '], time:2.699279ms, swizzle: NOOP, TFLOPS: 101.83(+2.51%)
        f16wmma(mma2x4+...+stage3+dsmem): ['43.6875   ', '-20.21875 '], time:2.689146ms, swizzle: NOOP, TFLOPS: 102.22(+0.38%)
        f16wmma(mma2x4+...+stage2+dsmem): ['43.6875   ', '-20.21875 '], time:2.682709ms, swizzle: NOOP, TFLOPS: 102.46(+0.24%)
      f16wmma(mma2x4+...+stage3+swizzle): ['43.6875   ', '-20.21875 '], time:2.685523ms, swizzle: 1024, TFLOPS: 102.36
      f16wmma(mma2x4+...+stage2+swizzle): ['43.6875   ', '-20.21875 '], time:2.643656ms, swizzle: 1024, TFLOPS: 103.98(+1.48%)
       f16wmma(...+stage3+dsmem+swizzle): ['43.6875   ', '-20.21875 '], time:2.694296ms, swizzle: 1024, TFLOPS: 102.02
       f16wmma(...+stage2+dsmem+swizzle): ['43.6875   ', '-20.21875 '], time:2.645778ms, swizzle: 1024, TFLOPS: 103.89
                             f16(cublas): ['43.53125  ', '-20.53125 '], time:2.624320ms, swizzle: NOOP, TFLOPS: 104.74(+0.74%)
                                  f16_th: ['43.625    ', '-20.625   '], time:2.397036ms, swizzle: NOOP, TFLOPS: 114.67(+9.48%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=2048
                    f16x8pack(t8x8+dbuf): ['29.828125 ', '48.15625  '], time:2.717447ms, swizzle: NOOP, TFLOPS: 50.58 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['29.828125 ', '48.15625  '], time:2.642869ms, swizzle: NOOP, TFLOPS: 52.00 (+2.82%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['29.96875  ', '47.9375   '], time:1.379895ms, swizzle: NOOP, TFLOPS: 99.60 (+91.53%)
          f16wmma(mma2x4+warp2x4+stage2): ['29.96875  ', '47.9375   '], time:1.371264ms, swizzle: NOOP, TFLOPS: 100.23(+0.63%)
        f16wmma(mma2x4+...+stage3+dsmem): ['29.96875  ', '47.9375   '], time:1.338505ms, swizzle: NOOP, TFLOPS: 102.68(+2.45%)
        f16wmma(mma2x4+...+stage2+dsmem): ['29.96875  ', '47.9375   '], time:1.321983ms, swizzle: NOOP, TFLOPS: 103.96(+1.25%)
      f16wmma(mma2x4+...+stage3+swizzle): ['29.96875  ', '47.9375   '], time:1.321935ms, swizzle: 2048, TFLOPS: 103.97(+0.00%)
      f16wmma(mma2x4+...+stage2+swizzle): ['29.96875  ', '47.9375   '], time:1.297378ms, swizzle: 2048, TFLOPS: 105.94(+1.89%)
       f16wmma(...+stage3+dsmem+swizzle): ['29.96875  ', '47.9375   '], time:1.319074ms, swizzle: 2048, TFLOPS: 104.19
       f16wmma(...+stage2+dsmem+swizzle): ['29.96875  ', '47.9375   '], time:1.299023ms, swizzle: 2048, TFLOPS: 105.80
                             f16(cublas): ['29.96875  ', '47.9375   '], time:1.424860ms, swizzle: NOOP, TFLOPS: 96.46
                                  f16_th: ['30.078125 ', '47.75     '], time:1.307225ms, swizzle: NOOP, TFLOPS: 105.14
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=4096
                    f16x8pack(t8x8+dbuf): ['21.5      ', '42.78125  '], time:5.479264ms, swizzle: NOOP, TFLOPS: 50.17 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['21.5      ', '42.78125  '], time:5.262446ms, swizzle: NOOP, TFLOPS: 52.23 (+4.12%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['21.75     ', '42.1875   '], time:2.657938ms, swizzle: NOOP, TFLOPS: 103.42(+97.99%)
          f16wmma(mma2x4+warp2x4+stage2): ['21.75     ', '42.1875   '], time:2.606439ms, swizzle: NOOP, TFLOPS: 105.46(+1.98%)
        f16wmma(mma2x4+...+stage3+dsmem): ['21.75     ', '42.1875   '], time:2.581810ms, swizzle: NOOP, TFLOPS: 106.47(+0.95%)
        f16wmma(mma2x4+...+stage2+dsmem): ['21.75     ', '42.1875   '], time:2.574849ms, swizzle: NOOP, TFLOPS: 106.75(+0.27%)
      f16wmma(mma2x4+...+stage3+swizzle): ['21.75     ', '42.1875   '], time:2.574110ms, swizzle: 2048, TFLOPS: 106.79(+0.03%)
      f16wmma(mma2x4+...+stage2+swizzle): ['21.75     ', '42.1875   '], time:2.543115ms, swizzle: 2048, TFLOPS: 108.09(+1.22%)
       f16wmma(...+stage3+dsmem+swizzle): ['21.75     ', '42.1875   '], time:2.583813ms, swizzle: 2048, TFLOPS: 106.38
       f16wmma(...+stage2+dsmem+swizzle): ['21.75     ', '42.1875   '], time:2.544641ms, swizzle: 2048, TFLOPS: 108.02
                             f16(cublas): ['21.75     ', '42.1875   '], time:2.626848ms, swizzle: NOOP, TFLOPS: 104.64
                                  f16_th: ['21.9375   ', '41.9375   '], time:2.556967ms, swizzle: NOOP, TFLOPS: 107.50
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=8192
                    f16x8pack(t8x8+dbuf): ['43.3125   ', '-19.578125'], time:11.78214ms, swizzle: NOOP, TFLOPS: 46.66 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['43.3125   ', '-19.578125'], time:11.40961ms, swizzle: NOOP, TFLOPS: 48.18 (+3.27%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['43.6875   ', '-20.21875 '], time:5.504322ms, swizzle: NOOP, TFLOPS: 99.88 (+107.28%)
          f16wmma(mma2x4+warp2x4+stage2): ['43.6875   ', '-20.21875 '], time:5.334901ms, swizzle: NOOP, TFLOPS: 103.05(+3.18%)
        f16wmma(mma2x4+...+stage3+dsmem): ['43.6875   ', '-20.21875 '], time:5.328178ms, swizzle: NOOP, TFLOPS: 103.18(+0.13%)
        f16wmma(mma2x4+...+stage2+dsmem): ['43.6875   ', '-20.21875 '], time:5.335330ms, swizzle: NOOP, TFLOPS: 103.04
      f16wmma(mma2x4+...+stage3+swizzle): ['43.6875   ', '-20.21875 '], time:5.165934ms, swizzle: 2048, TFLOPS: 106.42(+3.14%)
      f16wmma(mma2x4+...+stage2+swizzle): ['43.6875   ', '-20.21875 '], time:5.033230ms, swizzle: 2048, TFLOPS: 109.23(+2.64%)
       f16wmma(...+stage3+dsmem+swizzle): ['43.6875   ', '-20.21875 '], time:5.127739ms, swizzle: 2048, TFLOPS: 107.21
       f16wmma(...+stage2+dsmem+swizzle): ['43.6875   ', '-20.21875 '], time:5.035209ms, swizzle: 2048, TFLOPS: 109.18
                             f16(cublas): ['43.6875   ', '-20.21875 '], time:5.032253ms, swizzle: NOOP, TFLOPS: 109.25(+0.02%)
                                  f16_th: ['43.625    ', '-20.625   '], time:4.911947ms, swizzle: NOOP, TFLOPS: 111.92(+2.45%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=2048
                    f16x8pack(t8x8+dbuf): ['29.828125 ', '48.15625  '], time:5.409955ms, swizzle: NOOP, TFLOPS: 50.81 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['29.828125 ', '48.15625  '], time:5.225086ms, swizzle: NOOP, TFLOPS: 52.61 (+3.54%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['29.96875  ', '47.9375   '], time:2.669978ms, swizzle: NOOP, TFLOPS: 102.95(+95.70%)
          f16wmma(mma2x4+warp2x4+stage2): ['29.96875  ', '47.9375   '], time:2.579665ms, swizzle: NOOP, TFLOPS: 106.56(+3.50%)
        f16wmma(mma2x4+...+stage3+dsmem): ['29.96875  ', '47.9375   '], time:2.571344ms, swizzle: NOOP, TFLOPS: 106.90(+0.32%)
        f16wmma(mma2x4+...+stage2+dsmem): ['29.96875  ', '47.9375   '], time:2.548718ms, swizzle: NOOP, TFLOPS: 107.85(+0.89%)
      f16wmma(mma2x4+...+stage3+swizzle): ['29.96875  ', '47.9375   '], time:2.545380ms, swizzle: 4096, TFLOPS: 107.99(+0.13%)
      f16wmma(mma2x4+...+stage2+swizzle): ['29.96875  ', '47.9375   '], time:2.520227ms, swizzle: 4096, TFLOPS: 109.07(+1.00%)
       f16wmma(...+stage3+dsmem+swizzle): ['29.96875  ', '47.9375   '], time:2.557635ms, swizzle: 4096, TFLOPS: 107.47
       f16wmma(...+stage2+dsmem+swizzle): ['29.96875  ', '47.9375   '], time:2.521944ms, swizzle: 4096, TFLOPS: 108.99
                             f16(cublas): ['29.96875  ', '47.9375   '], time:2.594017ms, swizzle: NOOP, TFLOPS: 105.97
                                  f16_th: ['30.078125 ', '47.75     '], time:2.393794ms, swizzle: NOOP, TFLOPS: 114.83(+5.28%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=4096
                    f16x8pack(t8x8+dbuf): ['21.5      ', '42.78125  '], time:11.63647ms, swizzle: NOOP, TFLOPS: 47.24 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['21.5      ', '42.78125  '], time:12.05625ms, swizzle: NOOP, TFLOPS: 45.60
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['21.75     ', '42.1875   '], time:5.972003ms, swizzle: NOOP, TFLOPS: 92.06 (+94.85%)
          f16wmma(mma2x4+warp2x4+stage2): ['21.75     ', '42.1875   '], time:5.279278ms, swizzle: NOOP, TFLOPS: 104.13(+13.12%)
        f16wmma(mma2x4+...+stage3+dsmem): ['21.75     ', '42.1875   '], time:5.347394ms, swizzle: NOOP, TFLOPS: 102.81
        f16wmma(mma2x4+...+stage2+dsmem): ['21.75     ', '42.1875   '], time:5.421543ms, swizzle: NOOP, TFLOPS: 101.40
      f16wmma(mma2x4+...+stage3+swizzle): ['21.75     ', '42.1875   '], time:5.120348ms, swizzle: 4096, TFLOPS: 107.37(+3.10%)
      f16wmma(mma2x4+...+stage2+swizzle): ['21.75     ', '42.1875   '], time:4.952359ms, swizzle: 4096, TFLOPS: 111.01(+3.39%)
       f16wmma(...+stage3+dsmem+swizzle): ['21.75     ', '42.1875   '], time:5.023813ms, swizzle: 4096, TFLOPS: 109.43
       f16wmma(...+stage2+dsmem+swizzle): ['21.75     ', '42.1875   '], time:4.953908ms, swizzle: 4096, TFLOPS: 110.97
                             f16(cublas): ['21.75     ', '42.1875   '], time:4.956769ms, swizzle: NOOP, TFLOPS: 110.91
                                  f16_th: ['21.9375   ', '41.9375   '], time:4.872727ms, swizzle: NOOP, TFLOPS: 112.82(+1.63%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=8192
                    f16x8pack(t8x8+dbuf): ['43.3125   ', '-19.578125'], time:24.44663ms, swizzle: NOOP, TFLOPS: 44.98 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['43.3125   ', '-19.578125'], time:25.99511ms, swizzle: NOOP, TFLOPS: 42.30
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['43.6875   ', '-20.21875 '], time:12.52160ms, swizzle: NOOP, TFLOPS: 87.81 (+95.24%)
          f16wmma(mma2x4+warp2x4+stage2): ['43.6875   ', '-20.21875 '], time:12.27924ms, swizzle: NOOP, TFLOPS: 89.54 (+1.97%)
        f16wmma(mma2x4+...+stage3+dsmem): ['43.6875   ', '-20.21875 '], time:12.42353ms, swizzle: NOOP, TFLOPS: 88.50
        f16wmma(mma2x4+...+stage2+dsmem): ['43.6875   ', '-20.21875 '], time:12.27285ms, swizzle: NOOP, TFLOPS: 89.59 (+0.05%)
      f16wmma(mma2x4+...+stage3+swizzle): ['43.6875   ', '-20.21875 '], time:9.926629ms, swizzle: 4096, TFLOPS: 110.76(+23.64%)
      f16wmma(mma2x4+...+stage2+swizzle): ['43.6875   ', '-20.21875 '], time:9.907722ms, swizzle: 4096, TFLOPS: 110.98(+0.19%)
       f16wmma(...+stage3+dsmem+swizzle): ['43.6875   ', '-20.21875 '], time:10.07061ms, swizzle: 4096, TFLOPS: 109.18
       f16wmma(...+stage2+dsmem+swizzle): ['43.6875   ', '-20.21875 '], time:9.816384ms, swizzle: 4096, TFLOPS: 112.01(+0.93%)
                             f16(cublas): ['43.6875   ', '-20.21875 '], time:9.678959ms, swizzle: NOOP, TFLOPS: 113.60(+1.42%)
                                  f16_th: ['43.625    ', '-20.625   '], time:9.672522ms, swizzle: NOOP, TFLOPS: 113.67(+0.07%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=2048
                    f16x8pack(t8x8+dbuf): ['29.828125 ', '48.15625  '], time:2.660059ms, swizzle: NOOP, TFLOPS: 51.67 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['29.828125 ', '48.15625  '], time:2.649426ms, swizzle: NOOP, TFLOPS: 51.87 (+0.40%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['29.96875  ', '47.9375   '], time:1.397562ms, swizzle: NOOP, TFLOPS: 98.34 (+89.57%)
          f16wmma(mma2x4+warp2x4+stage2): ['29.96875  ', '47.9375   '], time:1.359796ms, swizzle: NOOP, TFLOPS: 101.07(+2.78%)
        f16wmma(mma2x4+...+stage3+dsmem): ['29.96875  ', '47.9375   '], time:1.334953ms, swizzle: NOOP, TFLOPS: 102.95(+1.86%)
        f16wmma(mma2x4+...+stage2+dsmem): ['29.96875  ', '47.9375   '], time:1.325345ms, swizzle: NOOP, TFLOPS: 103.70(+0.72%)
      f16wmma(mma2x4+...+stage3+swizzle): ['29.96875  ', '47.9375   '], time:1.321411ms, swizzle: 1024, TFLOPS: 104.01(+0.30%)
      f16wmma(mma2x4+...+stage2+swizzle): ['29.96875  ', '47.9375   '], time:1.303625ms, swizzle: 1024, TFLOPS: 105.43(+1.36%)
       f16wmma(...+stage3+dsmem+swizzle): ['29.96875  ', '47.9375   '], time:1.319599ms, swizzle: 1024, TFLOPS: 104.15
       f16wmma(...+stage2+dsmem+swizzle): ['29.96875  ', '47.9375   '], time:1.298236ms, swizzle: 1024, TFLOPS: 105.87(+0.42%)
                             f16(cublas): ['29.96875  ', '47.9375   '], time:1.433134ms, swizzle: NOOP, TFLOPS: 95.90
                                  f16_th: ['30.078125 ', '47.75     '], time:1.308012ms, swizzle: NOOP, TFLOPS: 105.07
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=4096
                    f16x8pack(t8x8+dbuf): ['21.5      ', '42.78125  '], time:5.431914ms, swizzle: NOOP, TFLOPS: 50.60 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['21.5      ', '42.78125  '], time:5.374670ms, swizzle: NOOP, TFLOPS: 51.14 (+1.07%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['21.75     ', '42.1875   '], time:2.703666ms, swizzle: NOOP, TFLOPS: 101.67(+98.79%)
          f16wmma(mma2x4+warp2x4+stage2): ['21.75     ', '42.1875   '], time:2.607297ms, swizzle: NOOP, TFLOPS: 105.43(+3.70%)
        f16wmma(mma2x4+...+stage3+dsmem): ['21.75     ', '42.1875   '], time:2.603697ms, swizzle: NOOP, TFLOPS: 105.57(+0.14%)
        f16wmma(mma2x4+...+stage2+dsmem): ['21.75     ', '42.1875   '], time:2.575922ms, swizzle: NOOP, TFLOPS: 106.71(+1.08%)
      f16wmma(mma2x4+...+stage3+swizzle): ['21.75     ', '42.1875   '], time:2.573966ms, swizzle: 1024, TFLOPS: 106.79(+0.08%)
      f16wmma(mma2x4+...+stage2+swizzle): ['21.75     ', '42.1875   '], time:2.542328ms, swizzle: 1024, TFLOPS: 108.12(+1.24%)
       f16wmma(...+stage3+dsmem+swizzle): ['21.75     ', '42.1875   '], time:2.584218ms, swizzle: 1024, TFLOPS: 106.37
       f16wmma(...+stage2+dsmem+swizzle): ['21.75     ', '42.1875   '], time:2.544331ms, swizzle: 1024, TFLOPS: 108.04
                             f16(cublas): ['21.75     ', '42.1875   '], time:2.638125ms, swizzle: NOOP, TFLOPS: 104.19
                                  f16_th: ['21.9375   ', '41.9375   '], time:2.552342ms, swizzle: NOOP, TFLOPS: 107.70
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=8192
                    f16x8pack(t8x8+dbuf): ['43.3125   ', '-19.578125'], time:11.08050ms, swizzle: NOOP, TFLOPS: 49.61 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['43.3125   ', '-19.578125'], time:10.69121ms, swizzle: NOOP, TFLOPS: 51.42 (+3.64%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['43.6875   ', '-20.21875 '], time:5.244946ms, swizzle: NOOP, TFLOPS: 104.82(+103.84%)
          f16wmma(mma2x4+warp2x4+stage2): ['43.6875   ', '-20.21875 '], time:5.098414ms, swizzle: NOOP, TFLOPS: 107.83(+2.87%)
        f16wmma(mma2x4+...+stage3+dsmem): ['43.6875   ', '-20.21875 '], time:5.103445ms, swizzle: NOOP, TFLOPS: 107.72
        f16wmma(mma2x4+...+stage2+dsmem): ['43.6875   ', '-20.21875 '], time:5.098533ms, swizzle: NOOP, TFLOPS: 107.83
      f16wmma(mma2x4+...+stage3+swizzle): ['43.6875   ', '-20.21875 '], time:5.078887ms, swizzle: 1024, TFLOPS: 108.24(+0.38%)
      f16wmma(mma2x4+...+stage2+swizzle): ['43.6875   ', '-20.21875 '], time:5.006766ms, swizzle: 1024, TFLOPS: 109.80(+1.44%)
       f16wmma(...+stage3+dsmem+swizzle): ['43.6875   ', '-20.21875 '], time:5.085110ms, swizzle: 1024, TFLOPS: 108.11
       f16wmma(...+stage2+dsmem+swizzle): ['43.6875   ', '-20.21875 '], time:5.008745ms, swizzle: 1024, TFLOPS: 109.76
                             f16(cublas): ['43.6875   ', '-20.21875 '], time:5.032014ms, swizzle: NOOP, TFLOPS: 109.25
                                  f16_th: ['43.625    ', '-20.625   '], time:4.892420ms, swizzle: NOOP, TFLOPS: 112.37(+2.34%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=2048
                    f16x8pack(t8x8+dbuf): ['29.828125 ', '48.15625  '], time:5.453610ms, swizzle: NOOP, TFLOPS: 50.40 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['29.828125 ', '48.15625  '], time:5.258274ms, swizzle: NOOP, TFLOPS: 52.28 (+3.71%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['29.96875  ', '47.9375   '], time:2.676272ms, swizzle: NOOP, TFLOPS: 102.71(+96.48%)
          f16wmma(mma2x4+warp2x4+stage2): ['29.96875  ', '47.9375   '], time:2.604269ms, swizzle: NOOP, TFLOPS: 105.55(+2.76%)
        f16wmma(mma2x4+...+stage3+dsmem): ['29.96875  ', '47.9375   '], time:2.556395ms, swizzle: NOOP, TFLOPS: 107.53(+1.87%)
        f16wmma(mma2x4+...+stage2+dsmem): ['29.96875  ', '47.9375   '], time:2.548575ms, swizzle: NOOP, TFLOPS: 107.86(+0.31%)
      f16wmma(mma2x4+...+stage3+swizzle): ['29.96875  ', '47.9375   '], time:2.544736ms, swizzle: 2048, TFLOPS: 108.02(+0.15%)
      f16wmma(mma2x4+...+stage2+swizzle): ['29.96875  ', '47.9375   '], time:2.520060ms, swizzle: 2048, TFLOPS: 109.08(+0.98%)
       f16wmma(...+stage3+dsmem+swizzle): ['29.96875  ', '47.9375   '], time:2.557325ms, swizzle: 2048, TFLOPS: 107.49
       f16wmma(...+stage2+dsmem+swizzle): ['29.96875  ', '47.9375   '], time:2.521085ms, swizzle: 2048, TFLOPS: 109.03
                             f16(cublas): ['29.96875  ', '47.9375   '], time:2.608060ms, swizzle: NOOP, TFLOPS: 105.40
                                  f16_th: ['30.078125 ', '47.75     '], time:2.403020ms, swizzle: NOOP, TFLOPS: 114.39(+4.87%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=4096
                    f16x8pack(t8x8+dbuf): ['21.5      ', '42.78125  '], time:11.00285ms, swizzle: NOOP, TFLOPS: 49.96 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['21.5      ', '42.78125  '], time:10.63611ms, swizzle: NOOP, TFLOPS: 51.69 (+3.45%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['21.75     ', '42.1875   '], time:5.203700ms, swizzle: NOOP, TFLOPS: 105.65(+104.40%)
          f16wmma(mma2x4+warp2x4+stage2): ['21.75     ', '42.1875   '], time:5.008912ms, swizzle: NOOP, TFLOPS: 109.76(+3.89%)
        f16wmma(mma2x4+...+stage3+dsmem): ['21.75     ', '42.1875   '], time:5.013966ms, swizzle: NOOP, TFLOPS: 109.64
        f16wmma(mma2x4+...+stage2+dsmem): ['21.75     ', '42.1875   '], time:5.010342ms, swizzle: NOOP, TFLOPS: 109.72
      f16wmma(mma2x4+...+stage3+swizzle): ['21.75     ', '42.1875   '], time:5.002737ms, swizzle: 2048, TFLOPS: 109.89(+0.12%)
      f16wmma(mma2x4+...+stage2+swizzle): ['21.75     ', '42.1875   '], time:4.925775ms, swizzle: 2048, TFLOPS: 111.61(+1.56%)
       f16wmma(...+stage3+dsmem+swizzle): ['21.75     ', '42.1875   '], time:4.997014ms, swizzle: 2048, TFLOPS: 110.02
       f16wmma(...+stage2+dsmem+swizzle): ['21.75     ', '42.1875   '], time:4.927396ms, swizzle: 2048, TFLOPS: 111.57
                             f16(cublas): ['21.75     ', '42.1875   '], time:4.947376ms, swizzle: NOOP, TFLOPS: 111.12
                                  f16_th: ['21.9375   ', '41.9375   '], time:4.873323ms, swizzle: NOOP, TFLOPS: 112.81(+1.08%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=8192
                    f16x8pack(t8x8+dbuf): ['43.3125   ', '-19.578125'], time:23.87967ms, swizzle: NOOP, TFLOPS: 46.04 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['43.3125   ', '-19.578125'], time:24.36006ms, swizzle: NOOP, TFLOPS: 45.14
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['43.6875   ', '-20.21875 '], time:10.96835ms, swizzle: NOOP, TFLOPS: 100.24(+117.71%)
          f16wmma(mma2x4+warp2x4+stage2): ['43.6875   ', '-20.21875 '], time:10.54425ms, swizzle: NOOP, TFLOPS: 104.28(+4.02%)
        f16wmma(mma2x4+...+stage3+dsmem): ['43.6875   ', '-20.21875 '], time:10.64598ms, swizzle: NOOP, TFLOPS: 103.28
        f16wmma(mma2x4+...+stage2+dsmem): ['43.6875   ', '-20.21875 '], time:10.60862ms, swizzle: NOOP, TFLOPS: 103.64
      f16wmma(mma2x4+...+stage3+swizzle): ['43.6875   ', '-20.21875 '], time:9.940195ms, swizzle: 2048, TFLOPS: 110.61(+6.08%)
      f16wmma(mma2x4+...+stage2+swizzle): ['43.6875   ', '-20.21875 '], time:9.811115ms, swizzle: 2048, TFLOPS: 112.07(+1.32%)
       f16wmma(...+stage3+dsmem+swizzle): ['43.6875   ', '-20.21875 '], time:9.947872ms, swizzle: 2048, TFLOPS: 110.53
       f16wmma(...+stage2+dsmem+swizzle): ['43.6875   ', '-20.21875 '], time:9.814596ms, swizzle: 2048, TFLOPS: 112.03
                             f16(cublas): ['43.6875   ', '-20.21875 '], time:9.683537ms, swizzle: NOOP, TFLOPS: 113.54(+1.32%)
                                  f16_th: ['43.625    ', '-20.625   '], time:9.672451ms, swizzle: NOOP, TFLOPS: 113.67(+0.11%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=2048
                    f16x8pack(t8x8+dbuf): ['29.828125 ', '48.15625  '], time:11.14294ms, swizzle: NOOP, TFLOPS: 49.34 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['29.828125 ', '48.15625  '], time:10.75413ms, swizzle: NOOP, TFLOPS: 51.12 (+3.62%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['29.96875  ', '47.9375   '], time:5.252432ms, swizzle: NOOP, TFLOPS: 104.67(+104.75%)
          f16wmma(mma2x4+warp2x4+stage2): ['29.96875  ', '47.9375   '], time:5.061149ms, swizzle: NOOP, TFLOPS: 108.62(+3.78%)
        f16wmma(mma2x4+...+stage3+dsmem): ['29.96875  ', '47.9375   '], time:5.053544ms, swizzle: NOOP, TFLOPS: 108.79(+0.15%)
        f16wmma(mma2x4+...+stage2+dsmem): ['29.96875  ', '47.9375   '], time:5.048632ms, swizzle: NOOP, TFLOPS: 108.89(+0.10%)
      f16wmma(mma2x4+...+stage3+swizzle): ['29.96875  ', '47.9375   '], time:5.045604ms, swizzle: 4096, TFLOPS: 108.96(+0.06%)
      f16wmma(mma2x4+...+stage2+swizzle): ['29.96875  ', '47.9375   '], time:4.967260ms, swizzle: 4096, TFLOPS: 110.68(+1.58%)
       f16wmma(...+stage3+dsmem+swizzle): ['29.96875  ', '47.9375   '], time:5.037450ms, swizzle: 4096, TFLOPS: 109.13
       f16wmma(...+stage2+dsmem+swizzle): ['29.96875  ', '47.9375   '], time:4.996585ms, swizzle: 4096, TFLOPS: 110.03
                             f16(cublas): ['29.96875  ', '47.9375   '], time:4.997658ms, swizzle: NOOP, TFLOPS: 110.00
                                  f16_th: ['30.078125 ', '47.75     '], time:4.858112ms, swizzle: NOOP, TFLOPS: 113.16(+2.25%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=4096
                    f16x8pack(t8x8+dbuf): ['21.5      ', '42.78125  '], time:25.91803ms, swizzle: NOOP, TFLOPS: 42.42 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['21.5      ', '42.78125  '], time:24.48651ms, swizzle: NOOP, TFLOPS: 44.90 (+5.85%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['21.75     ', '42.1875   '], time:10.86962ms, swizzle: NOOP, TFLOPS: 101.15(+125.27%)
          f16wmma(mma2x4+warp2x4+stage2): ['21.75     ', '42.1875   '], time:10.62030ms, swizzle: NOOP, TFLOPS: 103.53(+2.35%)
        f16wmma(mma2x4+...+stage3+dsmem): ['21.75     ', '42.1875   '], time:10.87796ms, swizzle: NOOP, TFLOPS: 101.08
        f16wmma(mma2x4+...+stage2+dsmem): ['21.75     ', '42.1875   '], time:10.85026ms, swizzle: NOOP, TFLOPS: 101.34
      f16wmma(mma2x4+...+stage3+swizzle): ['21.75     ', '42.1875   '], time:10.00666ms, swizzle: 4096, TFLOPS: 109.88(+6.13%)
      f16wmma(mma2x4+...+stage2+swizzle): ['21.75     ', '42.1875   '], time:9.851837ms, swizzle: 4096, TFLOPS: 111.60(+1.57%)
       f16wmma(...+stage3+dsmem+swizzle): ['21.75     ', '42.1875   '], time:9.935092ms, swizzle: 4096, TFLOPS: 110.67
       f16wmma(...+stage2+dsmem+swizzle): ['21.75     ', '42.1875   '], time:9.798955ms, swizzle: 4096, TFLOPS: 112.21(+0.54%)
                             f16(cublas): ['21.75     ', '42.1875   '], time:9.696006ms, swizzle: NOOP, TFLOPS: 113.40(+1.06%)
                                  f16_th: ['21.9375   ', '41.9375   '], time:9.576559ms, swizzle: NOOP, TFLOPS: 114.81(+1.25%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=8192
                    f16x8pack(t8x8+dbuf): ['43.3125   ', '-19.578125'], time:52.12385ms, swizzle: NOOP, TFLOPS: 42.19 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['43.3125   ', '-19.578125'], time:51.30028ms, swizzle: NOOP, TFLOPS: 42.87 (+1.61%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['43.6875   ', '-20.21875 '], time:24.82614ms, swizzle: NOOP, TFLOPS: 88.58 (+106.64%)
          f16wmma(mma2x4+warp2x4+stage2): ['43.6875   ', '-20.21875 '], time:24.78272ms, swizzle: NOOP, TFLOPS: 88.73 (+0.18%)
        f16wmma(mma2x4+...+stage3+dsmem): ['43.6875   ', '-20.21875 '], time:24.86634ms, swizzle: NOOP, TFLOPS: 88.43
        f16wmma(mma2x4+...+stage2+dsmem): ['43.6875   ', '-20.21875 '], time:24.75123ms, swizzle: NOOP, TFLOPS: 88.84 (+0.13%)
      f16wmma(mma2x4+...+stage3+swizzle): ['43.6875   ', '-20.21875 '], time:19.83647ms, swizzle: 4096, TFLOPS: 110.86(+24.78%)
      f16wmma(mma2x4+...+stage2+swizzle): ['43.6875   ', '-20.21875 '], time:19.76609ms, swizzle: 4096, TFLOPS: 111.25(+0.36%)
       f16wmma(...+stage3+dsmem+swizzle): ['43.6875   ', '-20.21875 '], time:19.96107ms, swizzle: 4096, TFLOPS: 110.17
       f16wmma(...+stage2+dsmem+swizzle): ['43.6875   ', '-20.21875 '], time:19.56977ms, swizzle: 4096, TFLOPS: 112.37(+1.00%)
                             f16(cublas): ['43.6875   ', '-20.21875 '], time:19.50418ms, swizzle: NOOP, TFLOPS: 112.75(+0.34%)
                                  f16_th: ['43.625    ', '-20.625   '], time:19.39613ms, swizzle: NOOP, TFLOPS: 113.37(+0.56%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=2048
                    f16x8pack(t8x8+dbuf): ['29.828125 ', '48.15625  '], time:5.446410ms, swizzle: NOOP, TFLOPS: 50.47 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['29.828125 ', '48.15625  '], time:5.356860ms, swizzle: NOOP, TFLOPS: 51.31 (+1.67%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['29.96875  ', '47.9375   '], time:2.719283ms, swizzle: NOOP, TFLOPS: 101.08(+97.00%)
          f16wmma(mma2x4+warp2x4+stage2): ['29.96875  ', '47.9375   '], time:2.589273ms, swizzle: NOOP, TFLOPS: 106.16(+5.02%)
        f16wmma(mma2x4+...+stage3+dsmem): ['29.96875  ', '47.9375   '], time:2.577090ms, swizzle: NOOP, TFLOPS: 106.66(+0.47%)
        f16wmma(mma2x4+...+stage2+dsmem): ['29.96875  ', '47.9375   '], time:2.549481ms, swizzle: NOOP, TFLOPS: 107.82(+1.08%)
      f16wmma(mma2x4+...+stage3+swizzle): ['29.96875  ', '47.9375   '], time:2.554798ms, swizzle: 1024, TFLOPS: 107.59
      f16wmma(mma2x4+...+stage2+swizzle): ['29.96875  ', '47.9375   '], time:2.551412ms, swizzle: 1024, TFLOPS: 107.74
       f16wmma(...+stage3+dsmem+swizzle): ['29.96875  ', '47.9375   '], time:2.589011ms, swizzle: 1024, TFLOPS: 106.17
       f16wmma(...+stage2+dsmem+swizzle): ['29.96875  ', '47.9375   '], time:2.551913ms, swizzle: 1024, TFLOPS: 107.71
                             f16(cublas): ['29.96875  ', '47.9375   '], time:2.633857ms, swizzle: NOOP, TFLOPS: 104.36
                                  f16_th: ['30.078125 ', '47.75     '], time:2.406287ms, swizzle: NOOP, TFLOPS: 114.23(+5.95%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=4096
                    f16x8pack(t8x8+dbuf): ['21.5      ', '42.78125  '], time:11.09659ms, swizzle: NOOP, TFLOPS: 49.54 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['21.5      ', '42.78125  '], time:10.89971ms, swizzle: NOOP, TFLOPS: 50.44 (+1.81%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['21.75     ', '42.1875   '], time:5.250835ms, swizzle: NOOP, TFLOPS: 104.70(+107.58%)
          f16wmma(mma2x4+warp2x4+stage2): ['21.75     ', '42.1875   '], time:5.027651ms, swizzle: NOOP, TFLOPS: 109.35(+4.44%)
        f16wmma(mma2x4+...+stage3+dsmem): ['21.75     ', '42.1875   '], time:5.012679ms, swizzle: NOOP, TFLOPS: 109.67(+0.30%)
        f16wmma(mma2x4+...+stage2+dsmem): ['21.75     ', '42.1875   '], time:5.011177ms, swizzle: NOOP, TFLOPS: 109.71(+0.03%)
      f16wmma(mma2x4+...+stage3+swizzle): ['21.75     ', '42.1875   '], time:5.054998ms, swizzle: 1024, TFLOPS: 108.75
      f16wmma(mma2x4+...+stage2+swizzle): ['21.75     ', '42.1875   '], time:5.016922ms, swizzle: 1024, TFLOPS: 109.58
       f16wmma(...+stage3+dsmem+swizzle): ['21.75     ', '42.1875   '], time:5.134606ms, swizzle: 1024, TFLOPS: 107.07
       f16wmma(...+stage2+dsmem+swizzle): ['21.75     ', '42.1875   '], time:5.019664ms, swizzle: 1024, TFLOPS: 109.52
                             f16(cublas): ['21.75     ', '42.1875   '], time:4.956316ms, swizzle: NOOP, TFLOPS: 110.92(+1.11%)
                                  f16_th: ['21.9375   ', '41.9375   '], time:4.901862ms, swizzle: NOOP, TFLOPS: 112.15(+1.11%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=8192
                    f16x8pack(t8x8+dbuf): ['43.3125   ', '-19.578125'], time:22.18217ms, swizzle: NOOP, TFLOPS: 49.57 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['43.3125   ', '-19.578125'], time:21.54114ms, swizzle: NOOP, TFLOPS: 51.04 (+2.98%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['43.6875   ', '-20.21875 '], time:10.16273ms, swizzle: NOOP, TFLOPS: 108.19(+111.96%)
          f16wmma(mma2x4+warp2x4+stage2): ['43.6875   ', '-20.21875 '], time:9.931492ms, swizzle: NOOP, TFLOPS: 110.71(+2.33%)
        f16wmma(mma2x4+...+stage3+dsmem): ['43.6875   ', '-20.21875 '], time:9.895348ms, swizzle: NOOP, TFLOPS: 111.11(+0.37%)
        f16wmma(mma2x4+...+stage2+dsmem): ['43.6875   ', '-20.21875 '], time:9.879732ms, swizzle: NOOP, TFLOPS: 111.29(+0.16%)
      f16wmma(mma2x4+...+stage3+swizzle): ['43.6875   ', '-20.21875 '], time:9.979248ms, swizzle: 1024, TFLOPS: 110.18
      f16wmma(mma2x4+...+stage2+swizzle): ['43.6875   ', '-20.21875 '], time:9.871578ms, swizzle: 1024, TFLOPS: 111.38(+0.08%)
       f16wmma(...+stage3+dsmem+swizzle): ['43.6875   ', '-20.21875 '], time:10.04981ms, swizzle: 1024, TFLOPS: 109.41
       f16wmma(...+stage2+dsmem+swizzle): ['43.6875   ', '-20.21875 '], time:9.942173ms, swizzle: 1024, TFLOPS: 110.59
                             f16(cublas): ['43.6875   ', '-20.21875 '], time:9.671616ms, swizzle: NOOP, TFLOPS: 113.68(+2.07%)
                                  f16_th: ['43.625    ', '-20.625   '], time:10.07661ms, swizzle: NOOP, TFLOPS: 109.12
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=2048
                    f16x8pack(t8x8+dbuf): ['29.828125 ', '48.15625  '], time:11.16781ms, swizzle: NOOP, TFLOPS: 49.23 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['29.828125 ', '48.15625  '], time:10.68592ms, swizzle: NOOP, TFLOPS: 51.45 (+4.51%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['29.96875  ', '47.9375   '], time:5.304908ms, swizzle: NOOP, TFLOPS: 103.63(+101.43%)
          f16wmma(mma2x4+warp2x4+stage2): ['29.96875  ', '47.9375   '], time:5.048704ms, swizzle: NOOP, TFLOPS: 108.89(+5.07%)
        f16wmma(mma2x4+...+stage3+dsmem): ['29.96875  ', '47.9375   '], time:5.052852ms, swizzle: NOOP, TFLOPS: 108.80
        f16wmma(mma2x4+...+stage2+dsmem): ['29.96875  ', '47.9375   '], time:5.048632ms, swizzle: NOOP, TFLOPS: 108.89(+0.00%)
      f16wmma(mma2x4+...+stage3+swizzle): ['29.96875  ', '47.9375   '], time:5.083298ms, swizzle: 2048, TFLOPS: 108.15
      f16wmma(mma2x4+...+stage2+swizzle): ['29.96875  ', '47.9375   '], time:5.055999ms, swizzle: 2048, TFLOPS: 108.73
       f16wmma(...+stage3+dsmem+swizzle): ['29.96875  ', '47.9375   '], time:5.128002ms, swizzle: 2048, TFLOPS: 107.21
       f16wmma(...+stage2+dsmem+swizzle): ['29.96875  ', '47.9375   '], time:5.059814ms, swizzle: 2048, TFLOPS: 108.65
                             f16(cublas): ['29.96875  ', '47.9375   '], time:4.985952ms, swizzle: NOOP, TFLOPS: 110.26(+1.26%)
                                  f16_th: ['30.078125 ', '47.75     '], time:4.820632ms, swizzle: NOOP, TFLOPS: 114.04(+3.43%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=4096
                    f16x8pack(t8x8+dbuf): ['21.5      ', '42.78125  '], time:22.50843ms, swizzle: NOOP, TFLOPS: 48.85 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['21.5      ', '42.78125  '], time:21.28853ms, swizzle: NOOP, TFLOPS: 51.65 (+5.73%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['21.75     ', '42.1875   '], time:10.19561ms, swizzle: NOOP, TFLOPS: 107.84(+108.80%)
          f16wmma(mma2x4+warp2x4+stage2): ['21.75     ', '42.1875   '], time:9.961318ms, swizzle: NOOP, TFLOPS: 110.38(+2.35%)
        f16wmma(mma2x4+...+stage3+dsmem): ['21.75     ', '42.1875   '], time:9.923768ms, swizzle: NOOP, TFLOPS: 110.80(+0.38%)
        f16wmma(mma2x4+...+stage2+dsmem): ['21.75     ', '42.1875   '], time:9.902667ms, swizzle: NOOP, TFLOPS: 111.03(+0.21%)
      f16wmma(mma2x4+...+stage3+swizzle): ['21.75     ', '42.1875   '], time:9.985685ms, swizzle: 2048, TFLOPS: 110.11
      f16wmma(mma2x4+...+stage2+swizzle): ['21.75     ', '42.1875   '], time:9.863257ms, swizzle: 2048, TFLOPS: 111.48(+0.40%)
       f16wmma(...+stage3+dsmem+swizzle): ['21.75     ', '42.1875   '], time:10.02886ms, swizzle: 2048, TFLOPS: 109.63
       f16wmma(...+stage2+dsmem+swizzle): ['21.75     ', '42.1875   '], time:9.939670ms, swizzle: 2048, TFLOPS: 110.62
                             f16(cublas): ['21.75     ', '42.1875   '], time:9.735059ms, swizzle: NOOP, TFLOPS: 112.94(+1.32%)
                                  f16_th: ['21.9375   ', '41.9375   '], time:9.576606ms, swizzle: NOOP, TFLOPS: 114.81(+1.65%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=8192
                    f16x8pack(t8x8+dbuf): ['43.3125   ', '-19.578125'], time:49.33888ms, swizzle: NOOP, TFLOPS: 44.57 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['43.3125   ', '-19.578125'], time:47.99692ms, swizzle: NOOP, TFLOPS: 45.82 (+2.80%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['43.6875   ', '-20.21875 '], time:21.67944ms, swizzle: NOOP, TFLOPS: 101.43(+121.39%)
          f16wmma(mma2x4+warp2x4+stage2): ['43.6875   ', '-20.21875 '], time:21.39406ms, swizzle: NOOP, TFLOPS: 102.79(+1.33%)
        f16wmma(mma2x4+...+stage3+dsmem): ['43.6875   ', '-20.21875 '], time:21.61624ms, swizzle: NOOP, TFLOPS: 101.73
        f16wmma(mma2x4+...+stage2+dsmem): ['43.6875   ', '-20.21875 '], time:21.42615ms, swizzle: NOOP, TFLOPS: 102.63
      f16wmma(mma2x4+...+stage3+swizzle): ['43.6875   ', '-20.21875 '], time:19.75040ms, swizzle: 2048, TFLOPS: 111.34(+8.32%)
      f16wmma(mma2x4+...+stage2+swizzle): ['43.6875   ', '-20.21875 '], time:19.55401ms, swizzle: 2048, TFLOPS: 112.46(+1.00%)
       f16wmma(...+stage3+dsmem+swizzle): ['43.6875   ', '-20.21875 '], time:19.82777ms, swizzle: 2048, TFLOPS: 110.91
       f16wmma(...+stage2+dsmem+swizzle): ['43.6875   ', '-20.21875 '], time:19.61979ms, swizzle: 2048, TFLOPS: 112.08
                             f16(cublas): ['43.6875   ', '-20.21875 '], time:19.21470ms, swizzle: NOOP, TFLOPS: 114.44(+1.77%)
                                  f16_th: ['43.625    ', '-20.625   '], time:19.41342ms, swizzle: NOOP, TFLOPS: 113.27
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=2048
                    f16x8pack(t8x8+dbuf): ['29.828125 ', '48.15625  '], time:22.39682ms, swizzle: NOOP, TFLOPS: 49.09 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['29.828125 ', '48.15625  '], time:21.21143ms, swizzle: NOOP, TFLOPS: 51.84 (+5.59%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['29.96875  ', '47.9375   '], time:10.27917ms, swizzle: NOOP, TFLOPS: 106.96(+106.35%)
          f16wmma(mma2x4+warp2x4+stage2): ['29.96875  ', '47.9375   '], time:10.05582ms, swizzle: NOOP, TFLOPS: 109.34(+2.22%)
        f16wmma(mma2x4+...+stage3+dsmem): ['29.96875  ', '47.9375   '], time:10.03055ms, swizzle: NOOP, TFLOPS: 109.62(+0.25%)
        f16wmma(mma2x4+...+stage2+dsmem): ['29.96875  ', '47.9375   '], time:10.02089ms, swizzle: NOOP, TFLOPS: 109.72(+0.10%)
      f16wmma(mma2x4+...+stage3+swizzle): ['29.96875  ', '47.9375   '], time:10.13073ms, swizzle: 4096, TFLOPS: 108.53
      f16wmma(mma2x4+...+stage2+swizzle): ['29.96875  ', '47.9375   '], time:9.979963ms, swizzle: 4096, TFLOPS: 110.17(+0.41%)
       f16wmma(...+stage3+dsmem+swizzle): ['29.96875  ', '47.9375   '], time:10.29798ms, swizzle: 4096, TFLOPS: 106.77
       f16wmma(...+stage2+dsmem+swizzle): ['29.96875  ', '47.9375   '], time:10.10079ms, swizzle: 4096, TFLOPS: 108.85
                             f16(cublas): ['29.96875  ', '47.9375   '], time:9.823417ms, swizzle: NOOP, TFLOPS: 111.93(+1.59%)
                                  f16_th: ['30.078125 ', '47.75     '], time:9.698748ms, swizzle: NOOP, TFLOPS: 113.37(+1.29%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=4096
                    f16x8pack(t8x8+dbuf): ['21.5      ', '42.78125  '], time:51.25117ms, swizzle: NOOP, TFLOPS: 42.91 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['21.5      ', '42.78125  '], time:47.79529ms, swizzle: NOOP, TFLOPS: 46.01 (+7.23%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['21.75     ', '42.1875   '], time:22.11267ms, swizzle: NOOP, TFLOPS: 99.45 (+116.14%)
          f16wmma(mma2x4+warp2x4+stage2): ['21.75     ', '42.1875   '], time:21.47891ms, swizzle: NOOP, TFLOPS: 102.38(+2.95%)
        f16wmma(mma2x4+...+stage3+dsmem): ['21.75     ', '42.1875   '], time:22.03938ms, swizzle: NOOP, TFLOPS: 99.78
        f16wmma(mma2x4+...+stage2+dsmem): ['21.75     ', '42.1875   '], time:21.72265ms, swizzle: NOOP, TFLOPS: 101.23
      f16wmma(mma2x4+...+stage3+swizzle): ['21.75     ', '42.1875   '], time:19.75204ms, swizzle: 4096, TFLOPS: 111.33(+8.74%)
      f16wmma(mma2x4+...+stage2+swizzle): ['21.75     ', '42.1875   '], time:19.45090ms, swizzle: 4096, TFLOPS: 113.06(+1.55%)
       f16wmma(...+stage3+dsmem+swizzle): ['21.75     ', '42.1875   '], time:19.89583ms, swizzle: 4096, TFLOPS: 110.53
       f16wmma(...+stage2+dsmem+swizzle): ['21.75     ', '42.1875   '], time:19.64452ms, swizzle: 4096, TFLOPS: 111.94
                             f16(cublas): ['21.75     ', '42.1875   '], time:19.46139ms, swizzle: NOOP, TFLOPS: 112.99
                                  f16_th: ['21.9375   ', '41.9375   '], time:19.11823ms, swizzle: NOOP, TFLOPS: 115.02(+1.74%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=8192
                    f16x8pack(t8x8+dbuf): ['43.3125   ', '-19.578125'], time:105.0558ms, swizzle: NOOP, TFLOPS: 41.86 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['43.3125   ', '-19.578125'], time:102.8327ms, swizzle: NOOP, TFLOPS: 42.77 (+2.16%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
          f16wmma(mma2x4+warp2x4+stage3): ['43.6875   ', '-20.21875 '], time:49.66168ms, swizzle: NOOP, TFLOPS: 88.56 (+107.07%)
          f16wmma(mma2x4+warp2x4+stage2): ['43.6875   ', '-20.21875 '], time:49.83716ms, swizzle: NOOP, TFLOPS: 88.25
        f16wmma(mma2x4+...+stage3+dsmem): ['43.6875   ', '-20.21875 '], time:49.68352ms, swizzle: NOOP, TFLOPS: 88.52
        f16wmma(mma2x4+...+stage2+dsmem): ['43.6875   ', '-20.21875 '], time:49.83155ms, swizzle: NOOP, TFLOPS: 88.26
      f16wmma(mma2x4+...+stage3+swizzle): ['43.6875   ', '-20.21875 '], time:39.48702ms, swizzle: 4096, TFLOPS: 111.38(+25.77%)
      f16wmma(mma2x4+...+stage2+swizzle): ['43.6875   ', '-20.21875 '], time:39.14082ms, swizzle: 4096, TFLOPS: 112.36(+0.88%)
       f16wmma(...+stage3+dsmem+swizzle): ['43.6875   ', '-20.21875 '], time:39.76533ms, swizzle: 4096, TFLOPS: 110.60
       f16wmma(...+stage2+dsmem+swizzle): ['43.6875   ', '-20.21875 '], time:38.84062ms, swizzle: 4096, TFLOPS: 113.23(+0.77%)
                             f16(cublas): ['43.6875   ', '-20.21875 '], time:38.62674ms, swizzle: NOOP, TFLOPS: 113.86(+0.55%)
                                  f16_th: ['43.625    ', '-20.625   '], time:38.53664ms, swizzle: NOOP, TFLOPS: 114.13(+0.23%)
----------------------------------------------------------------------------------------------------------------------------------
root@84d10136-98dc-41c0-a69b-8ff1f802b8cf:hgemm# python3 hgemm.py
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=2048
                    f16x8pack(t8x8+dbuf): ['1.59863281', '-1.5263671'], time:1.404404ms, swizzle: NOOP, TFLOPS: 48.93 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['1.59863281', '-1.5263671'], time:1.327443ms, swizzle: NOOP, TFLOPS: 51.77 (+5.80%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['1.89746094', '-1.4111328'], time:1.488709ms, swizzle: NOOP, TFLOPS: 46.16
                 f16wmma(mma4x2+warp2x4): ['1.89746094', '-1.4111328'], time:0.940060ms, swizzle: NOOP, TFLOPS: 73.10 (+41.21%)
          f16wmma(mma2x4+warp2x4+stage3): ['1.89746094', '-1.4111328'], time:0.698161ms, swizzle: NOOP, TFLOPS: 98.43 (+34.65%)
          f16wmma(mma2x4+warp2x4+stage2): ['1.89746094', '-1.4111328'], time:0.692677ms, swizzle: NOOP, TFLOPS: 99.21 (+0.79%)
        f16wmma(mma2x4+...+stage3+dsmem): ['1.89746094', '-1.4111328'], time:0.693106ms, swizzle: NOOP, TFLOPS: 99.15
        f16wmma(mma2x4+...+stage2+dsmem): ['1.89746094', '-1.4111328'], time:0.693821ms, swizzle: NOOP, TFLOPS: 99.04
      f16wmma(mma2x4+...+stage3+swizzle): ['1.89746094', '-1.4111328'], time:0.693750ms, swizzle: 1024, TFLOPS: 99.06
      f16wmma(mma2x4+...+stage2+swizzle): ['1.89746094', '-1.4111328'], time:0.682759ms, swizzle: 1024, TFLOPS: 100.65(+1.45%)
       f16wmma(...+stage3+dsmem+swizzle): ['1.89746094', '-1.4111328'], time:0.697231ms, swizzle: 1024, TFLOPS: 98.56
       f16wmma(...+stage2+dsmem+swizzle): ['1.89746094', '-1.4111328'], time:0.683140ms, swizzle: 1024, TFLOPS: 100.59
                             f16(cublas): ['1.89746094', '-1.4111328'], time:0.845146ms, swizzle: NOOP, TFLOPS: 81.31
                                  f16_th: ['1.94628906', '-1.4042968'], time:0.660371ms, swizzle: NOOP, TFLOPS: 104.06(+3.39%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=4096
                    f16x8pack(t8x8+dbuf): ['-39.28125 ', '-30.484375'], time:2.793455ms, swizzle: NOOP, TFLOPS: 49.20 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['-39.28125 ', '-30.484375'], time:2.647829ms, swizzle: NOOP, TFLOPS: 51.91 (+5.50%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-39.0     ', '-30.34375 '], time:2.983331ms, swizzle: NOOP, TFLOPS: 46.07
                 f16wmma(mma4x2+warp2x4): ['-39.0     ', '-30.34375 '], time:1.846551ms, swizzle: NOOP, TFLOPS: 74.43 (+43.39%)
          f16wmma(mma2x4+warp2x4+stage3): ['-39.0     ', '-30.34375 '], time:1.372385ms, swizzle: NOOP, TFLOPS: 100.15(+34.55%)
          f16wmma(mma2x4+warp2x4+stage2): ['-39.0     ', '-30.34375 '], time:1.359462ms, swizzle: NOOP, TFLOPS: 101.10(+0.95%)
        f16wmma(mma2x4+...+stage3+dsmem): ['-39.0     ', '-30.34375 '], time:1.358675ms, swizzle: NOOP, TFLOPS: 101.16(+0.06%)
        f16wmma(mma2x4+...+stage2+dsmem): ['-39.0     ', '-30.34375 '], time:1.359844ms, swizzle: NOOP, TFLOPS: 101.07
      f16wmma(mma2x4+...+stage3+swizzle): ['-39.0     ', '-30.34375 '], time:1.361584ms, swizzle: 1024, TFLOPS: 100.94
      f16wmma(mma2x4+...+stage2+swizzle): ['-39.0     ', '-30.34375 '], time:1.341962ms, swizzle: 1024, TFLOPS: 102.42(+1.25%)
       f16wmma(...+stage3+dsmem+swizzle): ['-39.0     ', '-30.34375 '], time:1.366734ms, swizzle: 1024, TFLOPS: 100.56
       f16wmma(...+stage2+dsmem+swizzle): ['-39.0     ', '-30.34375 '], time:1.340460ms, swizzle: 1024, TFLOPS: 102.53(+0.11%)
                             f16(cublas): ['-39.0     ', '-30.34375 '], time:1.504993ms, swizzle: NOOP, TFLOPS: 91.32
                                  f16_th: ['-38.9375  ', '-30.265625'], time:1.293635ms, swizzle: NOOP, TFLOPS: 106.24(+3.62%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=8192
                    f16x8pack(t8x8+dbuf): ['-69.125   ', '14.8359375'], time:5.602812ms, swizzle: NOOP, TFLOPS: 49.06 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['-69.125   ', '14.8359375'], time:5.485749ms, swizzle: NOOP, TFLOPS: 50.11 (+2.13%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-67.9375  ', '14.7734375'], time:5.937862ms, swizzle: NOOP, TFLOPS: 46.29
                 f16wmma(mma4x2+warp2x4): ['-67.9375  ', '14.7734375'], time:3.656244ms, swizzle: NOOP, TFLOPS: 75.18 (+50.04%)
          f16wmma(mma2x4+warp2x4+stage3): ['-67.9375  ', '14.7734375'], time:2.706050ms, swizzle: NOOP, TFLOPS: 101.58(+35.11%)
          f16wmma(mma2x4+warp2x4+stage2): ['-67.9375  ', '14.7734375'], time:2.683877ms, swizzle: NOOP, TFLOPS: 102.42(+0.83%)
        f16wmma(mma2x4+...+stage3+dsmem): ['-67.9375  ', '14.7734375'], time:2.682471ms, swizzle: NOOP, TFLOPS: 102.47(+0.05%)
        f16wmma(mma2x4+...+stage2+dsmem): ['-67.9375  ', '14.7734375'], time:2.685070ms, swizzle: NOOP, TFLOPS: 102.37
      f16wmma(mma2x4+...+stage3+swizzle): ['-67.9375  ', '14.7734375'], time:2.686834ms, swizzle: 1024, TFLOPS: 102.31
      f16wmma(mma2x4+...+stage2+swizzle): ['-67.9375  ', '14.7734375'], time:2.635765ms, swizzle: 1024, TFLOPS: 104.29(+1.77%)
       f16wmma(...+stage3+dsmem+swizzle): ['-67.9375  ', '14.7734375'], time:2.678680ms, swizzle: 1024, TFLOPS: 102.62
       f16wmma(...+stage2+dsmem+swizzle): ['-67.9375  ', '14.7734375'], time:2.631449ms, swizzle: 1024, TFLOPS: 104.46(+0.16%)
                             f16(cublas): ['-67.5625  ', '14.8125   '], time:2.652001ms, swizzle: NOOP, TFLOPS: 103.65
                                  f16_th: ['-67.3125  ', '14.953125 '], time:2.400565ms, swizzle: NOOP, TFLOPS: 114.51(+9.62%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=2048
                    f16x8pack(t8x8+dbuf): ['1.59863281', '-1.5263671'], time:2.663874ms, swizzle: NOOP, TFLOPS: 51.59 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['1.59863281', '-1.5263671'], time:2.546334ms, swizzle: NOOP, TFLOPS: 53.98 (+4.62%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['1.89746094', '-1.4111328'], time:3.093552ms, swizzle: NOOP, TFLOPS: 44.43
                 f16wmma(mma4x2+warp2x4): ['1.89746094', '-1.4111328'], time:1.884722ms, swizzle: NOOP, TFLOPS: 72.92 (+35.10%)
          f16wmma(mma2x4+warp2x4+stage3): ['1.89746094', '-1.4111328'], time:1.326441ms, swizzle: NOOP, TFLOPS: 103.61(+42.09%)
          f16wmma(mma2x4+warp2x4+stage2): ['1.89746094', '-1.4111328'], time:1.315212ms, swizzle: NOOP, TFLOPS: 104.50(+0.85%)
        f16wmma(mma2x4+...+stage3+dsmem): ['1.89746094', '-1.4111328'], time:1.313281ms, swizzle: NOOP, TFLOPS: 104.65(+0.15%)
        f16wmma(mma2x4+...+stage2+dsmem): ['1.89746094', '-1.4111328'], time:1.314806ms, swizzle: NOOP, TFLOPS: 104.53
      f16wmma(mma2x4+...+stage3+swizzle): ['1.89746094', '-1.4111328'], time:1.313948ms, swizzle: 2048, TFLOPS: 104.60
      f16wmma(mma2x4+...+stage2+swizzle): ['1.89746094', '-1.4111328'], time:1.297807ms, swizzle: 2048, TFLOPS: 105.90(+1.19%)
       f16wmma(...+stage3+dsmem+swizzle): ['1.89746094', '-1.4111328'], time:1.319813ms, swizzle: 2048, TFLOPS: 104.14
       f16wmma(...+stage2+dsmem+swizzle): ['1.89746094', '-1.4111328'], time:1.299142ms, swizzle: 2048, TFLOPS: 105.79
                             f16(cublas): ['1.89746094', '-1.4111328'], time:1.454472ms, swizzle: NOOP, TFLOPS: 94.49
                                  f16_th: ['1.94628906', '-1.4042968'], time:1.308298ms, swizzle: NOOP, TFLOPS: 105.05
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=4096
                    f16x8pack(t8x8+dbuf): ['-39.28125 ', '-30.484375'], time:5.483603ms, swizzle: NOOP, TFLOPS: 50.13 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['-39.28125 ', '-30.484375'], time:5.259418ms, swizzle: NOOP, TFLOPS: 52.26 (+4.26%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-39.0     ', '-30.34375 '], time:5.940341ms, swizzle: NOOP, TFLOPS: 46.27
                 f16wmma(mma4x2+warp2x4): ['-39.0     ', '-30.34375 '], time:3.635025ms, swizzle: NOOP, TFLOPS: 75.62 (+44.69%)
          f16wmma(mma2x4+warp2x4+stage3): ['-39.0     ', '-30.34375 '], time:2.593827ms, swizzle: NOOP, TFLOPS: 105.97(+40.14%)
          f16wmma(mma2x4+warp2x4+stage2): ['-39.0     ', '-30.34375 '], time:2.575826ms, swizzle: NOOP, TFLOPS: 106.71(+0.70%)
        f16wmma(mma2x4+...+stage3+dsmem): ['-39.0     ', '-30.34375 '], time:2.576589ms, swizzle: NOOP, TFLOPS: 106.68
        f16wmma(mma2x4+...+stage2+dsmem): ['-39.0     ', '-30.34375 '], time:2.575778ms, swizzle: NOOP, TFLOPS: 106.72(+0.00%)
      f16wmma(mma2x4+...+stage3+swizzle): ['-39.0     ', '-30.34375 '], time:2.575325ms, swizzle: 2048, TFLOPS: 106.74(+0.02%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-39.0     ', '-30.34375 '], time:2.539539ms, swizzle: 2048, TFLOPS: 108.24(+1.41%)
       f16wmma(...+stage3+dsmem+swizzle): ['-39.0     ', '-30.34375 '], time:2.569365ms, swizzle: 2048, TFLOPS: 106.98
       f16wmma(...+stage2+dsmem+swizzle): ['-39.0     ', '-30.34375 '], time:2.530384ms, swizzle: 2048, TFLOPS: 108.63(+0.36%)
                             f16(cublas): ['-39.0     ', '-30.34375 '], time:2.642416ms, swizzle: NOOP, TFLOPS: 104.03
                                  f16_th: ['-38.9375  ', '-30.265625'], time:2.550554ms, swizzle: NOOP, TFLOPS: 107.77
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=8192
                    f16x8pack(t8x8+dbuf): ['-69.125   ', '14.8359375'], time:11.65771ms, swizzle: NOOP, TFLOPS: 47.16 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['-69.125   ', '14.8359375'], time:11.30318ms, swizzle: NOOP, TFLOPS: 48.64 (+3.14%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-67.9375  ', '14.7734375'], time:12.42666ms, swizzle: NOOP, TFLOPS: 44.24
                 f16wmma(mma4x2+warp2x4): ['-67.9375  ', '14.7734375'], time:7.332730ms, swizzle: NOOP, TFLOPS: 74.97 (+54.15%)
          f16wmma(mma2x4+warp2x4+stage3): ['-67.9375  ', '14.7734375'], time:5.269169ms, swizzle: NOOP, TFLOPS: 104.33(+39.16%)
          f16wmma(mma2x4+warp2x4+stage2): ['-67.9375  ', '14.7734375'], time:5.303263ms, swizzle: NOOP, TFLOPS: 103.66
        f16wmma(mma2x4+...+stage3+dsmem): ['-67.9375  ', '14.7734375'], time:5.343055ms, swizzle: NOOP, TFLOPS: 102.89
        f16wmma(mma2x4+...+stage2+dsmem): ['-67.9375  ', '14.7734375'], time:5.364751ms, swizzle: NOOP, TFLOPS: 102.48
      f16wmma(mma2x4+...+stage3+swizzle): ['-67.9375  ', '14.7734375'], time:5.154800ms, swizzle: 2048, TFLOPS: 106.65(+2.22%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-67.9375  ', '14.7734375'], time:5.037450ms, swizzle: 2048, TFLOPS: 109.13(+2.33%)
       f16wmma(...+stage3+dsmem+swizzle): ['-67.9375  ', '14.7734375'], time:5.111980ms, swizzle: 2048, TFLOPS: 107.54
       f16wmma(...+stage2+dsmem+swizzle): ['-67.9375  ', '14.7734375'], time:5.037331ms, swizzle: 2048, TFLOPS: 109.14(+0.00%)
                             f16(cublas): ['-67.9375  ', '14.7734375'], time:5.056142ms, swizzle: NOOP, TFLOPS: 108.73
                                  f16_th: ['-67.375   ', '14.9609375'], time:4.914736ms, swizzle: NOOP, TFLOPS: 111.86(+2.49%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=2048
                    f16x8pack(t8x8+dbuf): ['1.59863281', '-1.5263671'], time:5.383038ms, swizzle: NOOP, TFLOPS: 51.06 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['1.59863281', '-1.5263671'], time:5.182886ms, swizzle: NOOP, TFLOPS: 53.04 (+3.86%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['1.89746094', '-1.4111328'], time:5.978965ms, swizzle: NOOP, TFLOPS: 45.97
                 f16wmma(mma4x2+warp2x4): ['1.89746094', '-1.4111328'], time:3.550386ms, swizzle: NOOP, TFLOPS: 77.42 (+45.98%)
          f16wmma(mma2x4+warp2x4+stage3): ['1.89746094', '-1.4111328'], time:2.566289ms, swizzle: NOOP, TFLOPS: 107.11(+38.35%)
          f16wmma(mma2x4+warp2x4+stage2): ['1.89746094', '-1.4111328'], time:2.549481ms, swizzle: NOOP, TFLOPS: 107.82(+0.66%)
        f16wmma(mma2x4+...+stage3+dsmem): ['1.89746094', '-1.4111328'], time:2.547526ms, swizzle: NOOP, TFLOPS: 107.90(+0.08%)
        f16wmma(mma2x4+...+stage2+dsmem): ['1.89746094', '-1.4111328'], time:2.549314ms, swizzle: NOOP, TFLOPS: 107.82
      f16wmma(mma2x4+...+stage3+swizzle): ['1.89746094', '-1.4111328'], time:2.546262ms, swizzle: 4096, TFLOPS: 107.95(+0.05%)
      f16wmma(mma2x4+...+stage2+swizzle): ['1.89746094', '-1.4111328'], time:2.513575ms, swizzle: 4096, TFLOPS: 109.36(+1.30%)
       f16wmma(...+stage3+dsmem+swizzle): ['1.89746094', '-1.4111328'], time:2.545523ms, swizzle: 4096, TFLOPS: 107.98
       f16wmma(...+stage2+dsmem+swizzle): ['1.89746094', '-1.4111328'], time:2.509880ms, swizzle: 4096, TFLOPS: 109.52(+0.15%)
                             f16(cublas): ['1.89746094', '-1.4111328'], time:2.609992ms, swizzle: NOOP, TFLOPS: 105.32
                                  f16_th: ['1.94628906', '-1.4042968'], time:2.398943ms, swizzle: NOOP, TFLOPS: 114.58(+4.62%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=4096
                    f16x8pack(t8x8+dbuf): ['-39.28125 ', '-30.484375'], time:11.72232ms, swizzle: NOOP, TFLOPS: 46.90 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['-39.28125 ', '-30.484375'], time:11.23323ms, swizzle: NOOP, TFLOPS: 48.94 (+4.35%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-39.0     ', '-30.34375 '], time:15.56153ms, swizzle: NOOP, TFLOPS: 35.33
                 f16wmma(mma4x2+warp2x4): ['-39.0     ', '-30.34375 '], time:7.295370ms, swizzle: NOOP, TFLOPS: 75.36 (+53.98%)
          f16wmma(mma2x4+warp2x4+stage3): ['-39.0     ', '-30.34375 '], time:5.264639ms, swizzle: NOOP, TFLOPS: 104.42(+38.57%)
          f16wmma(mma2x4+warp2x4+stage2): ['-39.0     ', '-30.34375 '], time:5.281233ms, swizzle: NOOP, TFLOPS: 104.10
        f16wmma(mma2x4+...+stage3+dsmem): ['-39.0     ', '-30.34375 '], time:5.325174ms, swizzle: NOOP, TFLOPS: 103.24
        f16wmma(mma2x4+...+stage2+dsmem): ['-39.0     ', '-30.34375 '], time:5.249261ms, swizzle: NOOP, TFLOPS: 104.73(+0.29%)
      f16wmma(mma2x4+...+stage3+swizzle): ['-39.0     ', '-30.34375 '], time:5.101490ms, swizzle: 4096, TFLOPS: 107.76(+2.90%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-39.0     ', '-30.34375 '], time:4.953670ms, swizzle: 4096, TFLOPS: 110.98(+2.98%)
       f16wmma(...+stage3+dsmem+swizzle): ['-39.0     ', '-30.34375 '], time:5.025219ms, swizzle: 4096, TFLOPS: 109.40
       f16wmma(...+stage2+dsmem+swizzle): ['-39.0     ', '-30.34375 '], time:4.955148ms, swizzle: 4096, TFLOPS: 110.95
                             f16(cublas): ['-39.0     ', '-30.34375 '], time:5.001807ms, swizzle: NOOP, TFLOPS: 109.91
                                  f16_th: ['-38.9375  ', '-30.265625'], time:4.873561ms, swizzle: NOOP, TFLOPS: 112.80(+1.64%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=8192
                    f16x8pack(t8x8+dbuf): ['-69.125   ', '14.8359375'], time:24.62918ms, swizzle: NOOP, TFLOPS: 44.64 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['-69.125   ', '14.8359375'], time:25.67009ms, swizzle: NOOP, TFLOPS: 42.83
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-67.9375  ', '14.7734375'], time:31.05800ms, swizzle: NOOP, TFLOPS: 35.40
                 f16wmma(mma4x2+warp2x4): ['-67.9375  ', '14.7734375'], time:14.35520ms, swizzle: NOOP, TFLOPS: 76.59 (+71.57%)
          f16wmma(mma2x4+warp2x4+stage3): ['-67.9375  ', '14.7734375'], time:12.42783ms, swizzle: NOOP, TFLOPS: 88.47 (+15.51%)
          f16wmma(mma2x4+warp2x4+stage2): ['-67.9375  ', '14.7734375'], time:12.27314ms, swizzle: NOOP, TFLOPS: 89.59 (+1.26%)
        f16wmma(mma2x4+...+stage3+dsmem): ['-67.9375  ', '14.7734375'], time:12.41307ms, swizzle: NOOP, TFLOPS: 88.58
        f16wmma(mma2x4+...+stage2+dsmem): ['-67.9375  ', '14.7734375'], time:12.28914ms, swizzle: NOOP, TFLOPS: 89.47
      f16wmma(mma2x4+...+stage3+swizzle): ['-67.9375  ', '14.7734375'], time:10.12926ms, swizzle: 4096, TFLOPS: 108.55(+21.17%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-67.9375  ', '14.7734375'], time:9.821295ms, swizzle: 4096, TFLOPS: 111.95(+3.14%)
       f16wmma(...+stage3+dsmem+swizzle): ['-67.9375  ', '14.7734375'], time:9.950423ms, swizzle: 4096, TFLOPS: 110.50
       f16wmma(...+stage2+dsmem+swizzle): ['-67.9375  ', '14.7734375'], time:9.760189ms, swizzle: 4096, TFLOPS: 112.65(+0.63%)
                             f16(cublas): ['-67.9375  ', '14.7734375'], time:9.659981ms, swizzle: NOOP, TFLOPS: 113.82(+1.04%)
                                  f16_th: ['-67.375   ', '14.9609375'], time:9.674811ms, swizzle: NOOP, TFLOPS: 113.65
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=2048
                    f16x8pack(t8x8+dbuf): ['1.59863281', '-1.5263671'], time:2.714872ms, swizzle: NOOP, TFLOPS: 50.62 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['1.59863281', '-1.5263671'], time:2.663779ms, swizzle: NOOP, TFLOPS: 51.60 (+1.92%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['1.89746094', '-1.4111328'], time:3.077459ms, swizzle: NOOP, TFLOPS: 44.66
                 f16wmma(mma4x2+warp2x4): ['1.89746094', '-1.4111328'], time:1.845455ms, swizzle: NOOP, TFLOPS: 74.47 (+44.34%)
          f16wmma(mma2x4+warp2x4+stage3): ['1.89746094', '-1.4111328'], time:1.331400ms, swizzle: NOOP, TFLOPS: 103.23(+38.61%)
          f16wmma(mma2x4+warp2x4+stage2): ['1.89746094', '-1.4111328'], time:1.315236ms, swizzle: NOOP, TFLOPS: 104.50(+1.23%)
        f16wmma(mma2x4+...+stage3+dsmem): ['1.89746094', '-1.4111328'], time:1.312994ms, swizzle: NOOP, TFLOPS: 104.68(+0.17%)
        f16wmma(mma2x4+...+stage2+dsmem): ['1.89746094', '-1.4111328'], time:1.314711ms, swizzle: NOOP, TFLOPS: 104.54
      f16wmma(mma2x4+...+stage3+swizzle): ['1.89746094', '-1.4111328'], time:1.313352ms, swizzle: 1024, TFLOPS: 104.65
      f16wmma(mma2x4+...+stage2+swizzle): ['1.89746094', '-1.4111328'], time:1.298546ms, swizzle: 1024, TFLOPS: 105.84(+1.11%)
       f16wmma(...+stage3+dsmem+swizzle): ['1.89746094', '-1.4111328'], time:1.319527ms, swizzle: 1024, TFLOPS: 104.16
       f16wmma(...+stage2+dsmem+swizzle): ['1.89746094', '-1.4111328'], time:1.298904ms, swizzle: 1024, TFLOPS: 105.81
                             f16(cublas): ['1.89746094', '-1.4111328'], time:1.470446ms, swizzle: NOOP, TFLOPS: 93.47
                                  f16_th: ['1.94628906', '-1.4042968'], time:1.308560ms, swizzle: NOOP, TFLOPS: 105.03
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=4096
                    f16x8pack(t8x8+dbuf): ['-39.28125 ', '-30.484375'], time:5.532741ms, swizzle: NOOP, TFLOPS: 49.68 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['-39.28125 ', '-30.484375'], time:5.330657ms, swizzle: NOOP, TFLOPS: 51.57 (+3.79%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-39.0     ', '-30.34375 '], time:6.005954ms, swizzle: NOOP, TFLOPS: 45.77
                 f16wmma(mma4x2+warp2x4): ['-39.0     ', '-30.34375 '], time:3.638529ms, swizzle: NOOP, TFLOPS: 75.55 (+46.51%)
          f16wmma(mma2x4+warp2x4+stage3): ['-39.0     ', '-30.34375 '], time:2.594757ms, swizzle: NOOP, TFLOPS: 105.94(+40.23%)
          f16wmma(mma2x4+warp2x4+stage2): ['-39.0     ', '-30.34375 '], time:2.575993ms, swizzle: NOOP, TFLOPS: 106.71(+0.73%)
        f16wmma(mma2x4+...+stage3+dsmem): ['-39.0     ', '-30.34375 '], time:2.576184ms, swizzle: NOOP, TFLOPS: 106.70
        f16wmma(mma2x4+...+stage2+dsmem): ['-39.0     ', '-30.34375 '], time:2.576065ms, swizzle: NOOP, TFLOPS: 106.70
      f16wmma(mma2x4+...+stage3+swizzle): ['-39.0     ', '-30.34375 '], time:2.574872ms, swizzle: 1024, TFLOPS: 106.75(+0.04%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-39.0     ', '-30.34375 '], time:2.543497ms, swizzle: 1024, TFLOPS: 108.07(+1.23%)
       f16wmma(...+stage3+dsmem+swizzle): ['-39.0     ', '-30.34375 '], time:2.570343ms, swizzle: 1024, TFLOPS: 106.94
       f16wmma(...+stage2+dsmem+swizzle): ['-39.0     ', '-30.34375 '], time:2.530217ms, swizzle: 1024, TFLOPS: 108.64(+0.52%)
                             f16(cublas): ['-39.0     ', '-30.34375 '], time:2.658534ms, swizzle: NOOP, TFLOPS: 103.39
                                  f16_th: ['-38.9375  ', '-30.265625'], time:2.551436ms, swizzle: NOOP, TFLOPS: 107.73
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=8192
                    f16x8pack(t8x8+dbuf): ['-69.125   ', '14.8359375'], time:11.10405ms, swizzle: NOOP, TFLOPS: 49.51 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['-69.125   ', '14.8359375'], time:10.68348ms, swizzle: NOOP, TFLOPS: 51.46 (+3.94%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-67.9375  ', '14.7734375'], time:11.77365ms, swizzle: NOOP, TFLOPS: 46.69
                 f16wmma(mma4x2+warp2x4): ['-67.9375  ', '14.7734375'], time:7.174301ms, swizzle: NOOP, TFLOPS: 76.63 (+48.91%)
          f16wmma(mma2x4+warp2x4+stage3): ['-67.9375  ', '14.7734375'], time:5.135965ms, swizzle: NOOP, TFLOPS: 107.04(+39.69%)
          f16wmma(mma2x4+warp2x4+stage2): ['-67.9375  ', '14.7734375'], time:5.070614ms, swizzle: NOOP, TFLOPS: 108.42(+1.29%)
        f16wmma(mma2x4+...+stage3+dsmem): ['-67.9375  ', '14.7734375'], time:5.074000ms, swizzle: NOOP, TFLOPS: 108.35
        f16wmma(mma2x4+...+stage2+dsmem): ['-67.9375  ', '14.7734375'], time:5.070209ms, swizzle: NOOP, TFLOPS: 108.43(+0.01%)
      f16wmma(mma2x4+...+stage3+swizzle): ['-67.9375  ', '14.7734375'], time:5.069398ms, swizzle: 1024, TFLOPS: 108.45(+0.02%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-67.9375  ', '14.7734375'], time:5.006647ms, swizzle: 1024, TFLOPS: 109.81(+1.25%)
       f16wmma(...+stage3+dsmem+swizzle): ['-67.9375  ', '14.7734375'], time:5.085134ms, swizzle: 1024, TFLOPS: 108.11
       f16wmma(...+stage2+dsmem+swizzle): ['-67.9375  ', '14.7734375'], time:5.008268ms, swizzle: 1024, TFLOPS: 109.77
                             f16(cublas): ['-67.9375  ', '14.7734375'], time:5.024623ms, swizzle: NOOP, TFLOPS: 109.41
                                  f16_th: ['-67.3125  ', '14.953125 '], time:4.873943ms, swizzle: NOOP, TFLOPS: 112.79(+2.72%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=2048
                    f16x8pack(t8x8+dbuf): ['1.59863281', '-1.5263671'], time:5.399775ms, swizzle: NOOP, TFLOPS: 50.91 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['1.59863281', '-1.5263671'], time:5.255079ms, swizzle: NOOP, TFLOPS: 52.31 (+2.75%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['1.89746094', '-1.4111328'], time:5.980038ms, swizzle: NOOP, TFLOPS: 45.97
                 f16wmma(mma4x2+warp2x4): ['1.89746094', '-1.4111328'], time:3.570342ms, swizzle: NOOP, TFLOPS: 76.99 (+47.19%)
          f16wmma(mma2x4+warp2x4+stage3): ['1.89746094', '-1.4111328'], time:2.563881ms, swizzle: NOOP, TFLOPS: 107.21(+39.26%)
          f16wmma(mma2x4+warp2x4+stage2): ['1.89746094', '-1.4111328'], time:2.548766ms, swizzle: NOOP, TFLOPS: 107.85(+0.59%)
        f16wmma(mma2x4+...+stage3+dsmem): ['1.89746094', '-1.4111328'], time:2.546596ms, swizzle: NOOP, TFLOPS: 107.94(+0.09%)
        f16wmma(mma2x4+...+stage2+dsmem): ['1.89746094', '-1.4111328'], time:2.548956ms, swizzle: NOOP, TFLOPS: 107.84
      f16wmma(mma2x4+...+stage3+swizzle): ['1.89746094', '-1.4111328'], time:2.545738ms, swizzle: 2048, TFLOPS: 107.98(+0.03%)
      f16wmma(mma2x4+...+stage2+swizzle): ['1.89746094', '-1.4111328'], time:2.519011ms, swizzle: 2048, TFLOPS: 109.12(+1.06%)
       f16wmma(...+stage3+dsmem+swizzle): ['1.89746094', '-1.4111328'], time:2.542185ms, swizzle: 2048, TFLOPS: 108.13
       f16wmma(...+stage2+dsmem+swizzle): ['1.89746094', '-1.4111328'], time:2.507638ms, swizzle: 2048, TFLOPS: 109.62(+0.45%)
                             f16(cublas): ['1.89746094', '-1.4111328'], time:2.602076ms, swizzle: NOOP, TFLOPS: 105.64
                                  f16_th: ['1.94628906', '-1.4042968'], time:2.392387ms, swizzle: NOOP, TFLOPS: 114.90(+4.82%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=4096
                    f16x8pack(t8x8+dbuf): ['-39.28125 ', '-30.484375'], time:10.81132ms, swizzle: NOOP, TFLOPS: 50.85 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['-39.28125 ', '-30.484375'], time:10.45336ms, swizzle: NOOP, TFLOPS: 52.59 (+3.42%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-39.0     ', '-30.34375 '], time:11.73586ms, swizzle: NOOP, TFLOPS: 46.84
                 f16wmma(mma4x2+warp2x4): ['-39.0     ', '-30.34375 '], time:7.009434ms, swizzle: NOOP, TFLOPS: 78.43 (+49.13%)
          f16wmma(mma2x4+warp2x4+stage3): ['-39.0     ', '-30.34375 '], time:5.041909ms, swizzle: NOOP, TFLOPS: 109.04(+39.02%)
          f16wmma(mma2x4+warp2x4+stage2): ['-39.0     ', '-30.34375 '], time:4.980516ms, swizzle: NOOP, TFLOPS: 110.38(+1.23%)
        f16wmma(mma2x4+...+stage3+dsmem): ['-39.0     ', '-30.34375 '], time:4.986071ms, swizzle: NOOP, TFLOPS: 110.26
        f16wmma(mma2x4+...+stage2+dsmem): ['-39.0     ', '-30.34375 '], time:4.980444ms, swizzle: NOOP, TFLOPS: 110.38(+0.00%)
      f16wmma(mma2x4+...+stage3+swizzle): ['-39.0     ', '-30.34375 '], time:4.974246ms, swizzle: 2048, TFLOPS: 110.52(+0.12%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-39.0     ', '-30.34375 '], time:4.922366ms, swizzle: 2048, TFLOPS: 111.69(+1.05%)
       f16wmma(...+stage3+dsmem+swizzle): ['-39.0     ', '-30.34375 '], time:4.996895ms, swizzle: 2048, TFLOPS: 110.02
       f16wmma(...+stage2+dsmem+swizzle): ['-39.0     ', '-30.34375 '], time:4.924964ms, swizzle: 2048, TFLOPS: 111.63
                             f16(cublas): ['-39.0     ', '-30.34375 '], time:4.935336ms, swizzle: NOOP, TFLOPS: 111.39
                                  f16_th: ['-38.9375  ', '-30.265625'], time:4.876470ms, swizzle: NOOP, TFLOPS: 112.74(+0.94%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=8192
                    f16x8pack(t8x8+dbuf): ['-69.125   ', '14.8359375'], time:23.82354ms, swizzle: NOOP, TFLOPS: 46.15 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['-69.125   ', '14.8359375'], time:24.44944ms, swizzle: NOOP, TFLOPS: 44.97
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-67.9375  ', '14.7734375'], time:25.05738ms, swizzle: NOOP, TFLOPS: 43.88
                 f16wmma(mma4x2+warp2x4): ['-67.9375  ', '14.7734375'], time:14.22839ms, swizzle: NOOP, TFLOPS: 77.28 (+67.44%)
          f16wmma(mma2x4+warp2x4+stage3): ['-67.9375  ', '14.7734375'], time:10.64333ms, swizzle: NOOP, TFLOPS: 103.31(+33.68%)
          f16wmma(mma2x4+warp2x4+stage2): ['-67.9375  ', '14.7734375'], time:10.46867ms, swizzle: NOOP, TFLOPS: 105.03(+1.67%)
        f16wmma(mma2x4+...+stage3+dsmem): ['-67.9375  ', '14.7734375'], time:10.57918ms, swizzle: NOOP, TFLOPS: 103.93
        f16wmma(mma2x4+...+stage2+dsmem): ['-67.9375  ', '14.7734375'], time:10.50658ms, swizzle: NOOP, TFLOPS: 104.65
      f16wmma(mma2x4+...+stage3+swizzle): ['-67.9375  ', '14.7734375'], time:9.939336ms, swizzle: 2048, TFLOPS: 110.62(+5.33%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-67.9375  ', '14.7734375'], time:9.870815ms, swizzle: 2048, TFLOPS: 111.39(+0.69%)
       f16wmma(...+stage3+dsmem+swizzle): ['-67.9375  ', '14.7734375'], time:9.955310ms, swizzle: 2048, TFLOPS: 110.44
       f16wmma(...+stage2+dsmem+swizzle): ['-67.9375  ', '14.7734375'], time:9.816360ms, swizzle: 2048, TFLOPS: 112.01(+0.55%)
                             f16(cublas): ['-67.9375  ', '14.7734375'], time:9.654760ms, swizzle: NOOP, TFLOPS: 113.88(+1.67%)
                                  f16_th: ['-67.375   ', '14.9609375'], time:9.672737ms, swizzle: NOOP, TFLOPS: 113.67
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=2048
                    f16x8pack(t8x8+dbuf): ['1.59863281', '-1.5263671'], time:10.90948ms, swizzle: NOOP, TFLOPS: 50.39 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['1.59863281', '-1.5263671'], time:10.57326ms, swizzle: NOOP, TFLOPS: 51.99 (+3.18%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['1.89746094', '-1.4111328'], time:11.80236ms, swizzle: NOOP, TFLOPS: 46.58
                 f16wmma(mma4x2+warp2x4): ['1.89746094', '-1.4111328'], time:6.925582ms, swizzle: NOOP, TFLOPS: 79.38 (+52.67%)
          f16wmma(mma2x4+warp2x4+stage3): ['1.89746094', '-1.4111328'], time:5.084562ms, swizzle: NOOP, TFLOPS: 108.12(+36.21%)
          f16wmma(mma2x4+warp2x4+stage2): ['1.89746094', '-1.4111328'], time:5.022287ms, swizzle: NOOP, TFLOPS: 109.46(+1.24%)
        f16wmma(mma2x4+...+stage3+dsmem): ['1.89746094', '-1.4111328'], time:5.024647ms, swizzle: NOOP, TFLOPS: 109.41
        f16wmma(mma2x4+...+stage2+dsmem): ['1.89746094', '-1.4111328'], time:5.019664ms, swizzle: NOOP, TFLOPS: 109.52(+0.05%)
      f16wmma(mma2x4+...+stage3+swizzle): ['1.89746094', '-1.4111328'], time:5.016016ms, swizzle: 4096, TFLOPS: 109.60(+0.07%)
      f16wmma(mma2x4+...+stage2+swizzle): ['1.89746094', '-1.4111328'], time:4.968237ms, swizzle: 4096, TFLOPS: 110.65(+0.96%)
       f16wmma(...+stage3+dsmem+swizzle): ['1.89746094', '-1.4111328'], time:5.039548ms, swizzle: 4096, TFLOPS: 109.09
       f16wmma(...+stage2+dsmem+swizzle): ['1.89746094', '-1.4111328'], time:4.970836ms, swizzle: 4096, TFLOPS: 110.60
                             f16(cublas): ['1.89746094', '-1.4111328'], time:5.000329ms, swizzle: NOOP, TFLOPS: 109.94
                                  f16_th: ['1.94628906', '-1.4042968'], time:4.831910ms, swizzle: NOOP, TFLOPS: 113.78(+2.82%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=4096
                    f16x8pack(t8x8+dbuf): ['-39.28125 ', '-30.484375'], time:25.65839ms, swizzle: NOOP, TFLOPS: 42.85 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['-39.28125 ', '-30.484375'], time:23.98827ms, swizzle: NOOP, TFLOPS: 45.84 (+6.96%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-39.0     ', '-30.34375 '], time:31.22930ms, swizzle: NOOP, TFLOPS: 35.21
                 f16wmma(mma4x2+warp2x4): ['-39.0     ', '-30.34375 '], time:14.21833ms, swizzle: NOOP, TFLOPS: 77.33 (+68.71%)
          f16wmma(mma2x4+warp2x4+stage3): ['-39.0     ', '-30.34375 '], time:10.81850ms, swizzle: NOOP, TFLOPS: 101.63(+31.43%)
          f16wmma(mma2x4+warp2x4+stage2): ['-39.0     ', '-30.34375 '], time:10.66896ms, swizzle: NOOP, TFLOPS: 103.06(+1.40%)
        f16wmma(mma2x4+...+stage3+dsmem): ['-39.0     ', '-30.34375 '], time:10.91930ms, swizzle: NOOP, TFLOPS: 100.69
        f16wmma(mma2x4+...+stage2+dsmem): ['-39.0     ', '-30.34375 '], time:10.77117ms, swizzle: NOOP, TFLOPS: 102.08
      f16wmma(mma2x4+...+stage3+swizzle): ['-39.0     ', '-30.34375 '], time:9.971523ms, swizzle: 4096, TFLOPS: 110.27(+6.99%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-39.0     ', '-30.34375 '], time:9.851503ms, swizzle: 4096, TFLOPS: 111.61(+1.22%)
       f16wmma(...+stage3+dsmem+swizzle): ['-39.0     ', '-30.34375 '], time:9.935140ms, swizzle: 4096, TFLOPS: 110.67
       f16wmma(...+stage2+dsmem+swizzle): ['-39.0     ', '-30.34375 '], time:9.798574ms, swizzle: 4096, TFLOPS: 112.21(+0.54%)
                             f16(cublas): ['-39.0     ', '-30.34375 '], time:9.702229ms, swizzle: NOOP, TFLOPS: 113.33(+0.99%)
                                  f16_th: ['-38.9375  ', '-30.265625'], time:9.520697ms, swizzle: NOOP, TFLOPS: 115.49(+1.91%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=8192
                    f16x8pack(t8x8+dbuf): ['-69.125   ', '14.8359375'], time:51.99902ms, swizzle: NOOP, TFLOPS: 42.29 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['-69.125   ', '14.8359375'], time:50.50570ms, swizzle: NOOP, TFLOPS: 43.54 (+2.96%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-67.9375  ', '14.7734375'], time:61.79916ms, swizzle: NOOP, TFLOPS: 35.58
                 f16wmma(mma4x2+warp2x4): ['-67.9375  ', '14.7734375'], time:28.18243ms, swizzle: NOOP, TFLOPS: 78.03 (+79.21%)
          f16wmma(mma2x4+warp2x4+stage3): ['-67.9375  ', '14.7734375'], time:24.82872ms, swizzle: NOOP, TFLOPS: 88.57 (+13.51%)
          f16wmma(mma2x4+warp2x4+stage2): ['-67.9375  ', '14.7734375'], time:24.81389ms, swizzle: NOOP, TFLOPS: 88.62 (+0.06%)
        f16wmma(mma2x4+...+stage3+dsmem): ['-67.9375  ', '14.7734375'], time:24.83656ms, swizzle: NOOP, TFLOPS: 88.54
        f16wmma(mma2x4+...+stage2+dsmem): ['-67.9375  ', '14.7734375'], time:24.74703ms, swizzle: NOOP, TFLOPS: 88.86 (+0.27%)
      f16wmma(mma2x4+...+stage3+swizzle): ['-67.9375  ', '14.7734375'], time:19.90499ms, swizzle: 4096, TFLOPS: 110.48(+24.33%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-67.9375  ', '14.7734375'], time:19.56765ms, swizzle: 4096, TFLOPS: 112.38(+1.72%)
       f16wmma(...+stage3+dsmem+swizzle): ['-67.9375  ', '14.7734375'], time:19.70267ms, swizzle: 4096, TFLOPS: 111.61
       f16wmma(...+stage2+dsmem+swizzle): ['-67.9375  ', '14.7734375'], time:19.46005ms, swizzle: 4096, TFLOPS: 113.00(+0.55%)
                             f16(cublas): ['-67.9375  ', '14.7734375'], time:19.52464ms, swizzle: NOOP, TFLOPS: 112.63
                                  f16_th: ['-67.375   ', '14.9609375'], time:19.39635ms, swizzle: NOOP, TFLOPS: 113.37(+0.33%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=2048
                    f16x8pack(t8x8+dbuf): ['1.59863281', '-1.5263671'], time:5.357718ms, swizzle: NOOP, TFLOPS: 51.31 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['1.59863281', '-1.5263671'], time:5.348920ms, swizzle: NOOP, TFLOPS: 51.39 (+0.16%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['1.89746094', '-1.4111328'], time:6.000566ms, swizzle: NOOP, TFLOPS: 45.81
                 f16wmma(mma4x2+warp2x4): ['1.89746094', '-1.4111328'], time:3.570771ms, swizzle: NOOP, TFLOPS: 76.98 (+49.80%)
          f16wmma(mma2x4+warp2x4+stage3): ['1.89746094', '-1.4111328'], time:2.565336ms, swizzle: NOOP, TFLOPS: 107.15(+39.19%)
          f16wmma(mma2x4+warp2x4+stage2): ['1.89746094', '-1.4111328'], time:2.550292ms, swizzle: NOOP, TFLOPS: 107.78(+0.59%)
        f16wmma(mma2x4+...+stage3+dsmem): ['1.89746094', '-1.4111328'], time:2.547144ms, swizzle: NOOP, TFLOPS: 107.92(+0.12%)
        f16wmma(mma2x4+...+stage2+dsmem): ['1.89746094', '-1.4111328'], time:2.550339ms, swizzle: NOOP, TFLOPS: 107.78
      f16wmma(mma2x4+...+stage3+swizzle): ['1.89746094', '-1.4111328'], time:2.546238ms, swizzle: 1024, TFLOPS: 107.95(+0.04%)
      f16wmma(mma2x4+...+stage2+swizzle): ['1.89746094', '-1.4111328'], time:2.522706ms, swizzle: 1024, TFLOPS: 108.96(+0.93%)
       f16wmma(...+stage3+dsmem+swizzle): ['1.89746094', '-1.4111328'], time:2.560997ms, swizzle: 1024, TFLOPS: 107.33
       f16wmma(...+stage2+dsmem+swizzle): ['1.89746094', '-1.4111328'], time:2.536177ms, swizzle: 1024, TFLOPS: 108.38
                             f16(cublas): ['1.89746094', '-1.4111328'], time:2.653121ms, swizzle: NOOP, TFLOPS: 103.61
                                  f16_th: ['1.94628906', '-1.4042968'], time:2.404737ms, swizzle: NOOP, TFLOPS: 114.31(+4.91%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=4096
                    f16x8pack(t8x8+dbuf): ['-39.28125 ', '-30.484375'], time:11.02826ms, swizzle: NOOP, TFLOPS: 49.85 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['-39.28125 ', '-30.484375'], time:10.60945ms, swizzle: NOOP, TFLOPS: 51.82 (+3.95%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-39.0     ', '-30.34375 '], time:11.79709ms, swizzle: NOOP, TFLOPS: 46.60
                 f16wmma(mma4x2+warp2x4): ['-39.0     ', '-30.34375 '], time:7.007050ms, swizzle: NOOP, TFLOPS: 78.46 (+51.41%)
          f16wmma(mma2x4+warp2x4+stage3): ['-39.0     ', '-30.34375 '], time:5.042505ms, swizzle: NOOP, TFLOPS: 109.02(+38.96%)
          f16wmma(mma2x4+warp2x4+stage2): ['-39.0     ', '-30.34375 '], time:4.984068ms, swizzle: NOOP, TFLOPS: 110.30(+1.17%)
        f16wmma(mma2x4+...+stage3+dsmem): ['-39.0     ', '-30.34375 '], time:4.986548ms, swizzle: NOOP, TFLOPS: 110.25
        f16wmma(mma2x4+...+stage2+dsmem): ['-39.0     ', '-30.34375 '], time:4.983687ms, swizzle: NOOP, TFLOPS: 110.31(+0.01%)
      f16wmma(mma2x4+...+stage3+swizzle): ['-39.0     ', '-30.34375 '], time:5.016398ms, swizzle: 1024, TFLOPS: 109.59
      f16wmma(mma2x4+...+stage2+swizzle): ['-39.0     ', '-30.34375 '], time:4.984092ms, swizzle: 1024, TFLOPS: 110.30
       f16wmma(...+stage3+dsmem+swizzle): ['-39.0     ', '-30.34375 '], time:5.056858ms, swizzle: 1024, TFLOPS: 108.71
       f16wmma(...+stage2+dsmem+swizzle): ['-39.0     ', '-30.34375 '], time:4.984569ms, swizzle: 1024, TFLOPS: 110.29
                             f16(cublas): ['-39.0     ', '-30.34375 '], time:4.996538ms, swizzle: NOOP, TFLOPS: 110.03
                                  f16_th: ['-38.9375  ', '-30.265625'], time:4.907298ms, swizzle: NOOP, TFLOPS: 112.03(+1.56%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=8192
                    f16x8pack(t8x8+dbuf): ['-69.125   ', '14.8359375'], time:22.09017ms, swizzle: NOOP, TFLOPS: 49.77 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['-69.125   ', '14.8359375'], time:21.44041ms, swizzle: NOOP, TFLOPS: 51.28 (+3.03%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-67.9375  ', '14.7734375'], time:23.34470ms, swizzle: NOOP, TFLOPS: 47.10
                 f16wmma(mma4x2+warp2x4): ['-67.9375  ', '14.7734375'], time:13.78324ms, swizzle: NOOP, TFLOPS: 79.77 (+55.55%)
          f16wmma(mma2x4+warp2x4+stage3): ['-67.9375  ', '14.7734375'], time:9.940481ms, swizzle: NOOP, TFLOPS: 110.61(+38.66%)
          f16wmma(mma2x4+warp2x4+stage2): ['-67.9375  ', '14.7734375'], time:9.873795ms, swizzle: NOOP, TFLOPS: 111.36(+0.68%)
        f16wmma(mma2x4+...+stage3+dsmem): ['-67.9375  ', '14.7734375'], time:9.887790ms, swizzle: NOOP, TFLOPS: 111.20
        f16wmma(mma2x4+...+stage2+dsmem): ['-67.9375  ', '14.7734375'], time:9.872055ms, swizzle: NOOP, TFLOPS: 111.38(+0.02%)
      f16wmma(mma2x4+...+stage3+swizzle): ['-67.9375  ', '14.7734375'], time:9.961581ms, swizzle: 1024, TFLOPS: 110.38
      f16wmma(mma2x4+...+stage2+swizzle): ['-67.9375  ', '14.7734375'], time:9.813380ms, swizzle: 1024, TFLOPS: 112.04(+0.60%)
       f16wmma(...+stage3+dsmem+swizzle): ['-67.9375  ', '14.7734375'], time:9.958195ms, swizzle: 1024, TFLOPS: 110.41
       f16wmma(...+stage2+dsmem+swizzle): ['-67.9375  ', '14.7734375'], time:9.937167ms, swizzle: 1024, TFLOPS: 110.65
                             f16(cublas): ['-67.9375  ', '14.7734375'], time:9.706902ms, swizzle: NOOP, TFLOPS: 113.27(+1.10%)
                                  f16_th: ['-67.3125  ', '14.953125 '], time:9.992313ms, swizzle: NOOP, TFLOPS: 110.04
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=2048
                    f16x8pack(t8x8+dbuf): ['1.59863281', '-1.5263671'], time:11.18245ms, swizzle: NOOP, TFLOPS: 49.16 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['1.59863281', '-1.5263671'], time:10.62731ms, swizzle: NOOP, TFLOPS: 51.73 (+5.22%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['1.89746094', '-1.4111328'], time:11.85719ms, swizzle: NOOP, TFLOPS: 46.36
                 f16wmma(mma4x2+warp2x4): ['1.89746094', '-1.4111328'], time:6.969070ms, swizzle: NOOP, TFLOPS: 78.89 (+52.49%)
          f16wmma(mma2x4+warp2x4+stage3): ['1.89746094', '-1.4111328'], time:5.083250ms, swizzle: NOOP, TFLOPS: 108.15(+37.10%)
          f16wmma(mma2x4+warp2x4+stage2): ['1.89746094', '-1.4111328'], time:5.020236ms, swizzle: NOOP, TFLOPS: 109.51(+1.26%)
        f16wmma(mma2x4+...+stage3+dsmem): ['1.89746094', '-1.4111328'], time:5.024361ms, swizzle: NOOP, TFLOPS: 109.42
        f16wmma(mma2x4+...+stage2+dsmem): ['1.89746094', '-1.4111328'], time:5.019664ms, swizzle: NOOP, TFLOPS: 109.52(+0.01%)
      f16wmma(mma2x4+...+stage3+swizzle): ['1.89746094', '-1.4111328'], time:5.042648ms, swizzle: 2048, TFLOPS: 109.02
      f16wmma(mma2x4+...+stage2+swizzle): ['1.89746094', '-1.4111328'], time:4.996371ms, swizzle: 2048, TFLOPS: 110.03(+0.47%)
       f16wmma(...+stage3+dsmem+swizzle): ['1.89746094', '-1.4111328'], time:5.109548ms, swizzle: 2048, TFLOPS: 107.59
       f16wmma(...+stage2+dsmem+swizzle): ['1.89746094', '-1.4111328'], time:4.999089ms, swizzle: 2048, TFLOPS: 109.97
                             f16(cublas): ['1.89746094', '-1.4111328'], time:5.016422ms, swizzle: NOOP, TFLOPS: 109.59
                                  f16_th: ['1.94628906', '-1.4042968'], time:4.833221ms, swizzle: NOOP, TFLOPS: 113.75(+3.38%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=4096
                    f16x8pack(t8x8+dbuf): ['-39.28125 ', '-30.484375'], time:22.44675ms, swizzle: NOOP, TFLOPS: 48.98 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['-39.28125 ', '-30.484375'], time:21.31097ms, swizzle: NOOP, TFLOPS: 51.59 (+5.33%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-39.0     ', '-30.34375 '], time:23.37458ms, swizzle: NOOP, TFLOPS: 47.04
                 f16wmma(mma4x2+warp2x4): ['-39.0     ', '-30.34375 '], time:13.62562ms, swizzle: NOOP, TFLOPS: 80.69 (+56.40%)
          f16wmma(mma2x4+warp2x4+stage3): ['-39.0     ', '-30.34375 '], time:9.968185ms, swizzle: NOOP, TFLOPS: 110.30(+36.69%)
          f16wmma(mma2x4+warp2x4+stage2): ['-39.0     ', '-30.34375 '], time:9.902882ms, swizzle: NOOP, TFLOPS: 111.03(+0.66%)
        f16wmma(mma2x4+...+stage3+dsmem): ['-39.0     ', '-30.34375 '], time:9.920740ms, swizzle: NOOP, TFLOPS: 110.83
        f16wmma(mma2x4+...+stage2+dsmem): ['-39.0     ', '-30.34375 '], time:9.903454ms, swizzle: NOOP, TFLOPS: 111.02
      f16wmma(mma2x4+...+stage3+swizzle): ['-39.0     ', '-30.34375 '], time:9.889793ms, swizzle: 2048, TFLOPS: 111.18(+0.13%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-39.0     ', '-30.34375 '], time:9.793472ms, swizzle: 2048, TFLOPS: 112.27(+0.98%)
       f16wmma(...+stage3+dsmem+swizzle): ['-39.0     ', '-30.34375 '], time:9.933280ms, swizzle: 2048, TFLOPS: 110.69
       f16wmma(...+stage2+dsmem+swizzle): ['-39.0     ', '-30.34375 '], time:9.900712ms, swizzle: 2048, TFLOPS: 111.05
                             f16(cublas): ['-39.0     ', '-30.34375 '], time:9.764933ms, swizzle: NOOP, TFLOPS: 112.60(+0.29%)
                                  f16_th: ['-38.9375  ', '-30.265625'], time:9.576892ms, swizzle: NOOP, TFLOPS: 114.81(+1.96%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=8192
                    f16x8pack(t8x8+dbuf): ['-69.125   ', '14.8359375'], time:49.22273ms, swizzle: NOOP, TFLOPS: 44.67 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['-69.125   ', '14.8359375'], time:48.20930ms, swizzle: NOOP, TFLOPS: 45.61 (+2.10%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-67.9375  ', '14.7734375'], time:50.56085ms, swizzle: NOOP, TFLOPS: 43.49
                 f16wmma(mma4x2+warp2x4): ['-67.9375  ', '14.7734375'], time:28.05802ms, swizzle: NOOP, TFLOPS: 78.37 (+71.82%)
          f16wmma(mma2x4+warp2x4+stage3): ['-67.9375  ', '14.7734375'], time:21.29669ms, swizzle: NOOP, TFLOPS: 103.26(+31.75%)
          f16wmma(mma2x4+warp2x4+stage2): ['-67.9375  ', '14.7734375'], time:21.00188ms, swizzle: NOOP, TFLOPS: 104.71(+1.40%)
        f16wmma(mma2x4+...+stage3+dsmem): ['-67.9375  ', '14.7734375'], time:22.03595ms, swizzle: NOOP, TFLOPS: 99.79
        f16wmma(mma2x4+...+stage2+dsmem): ['-67.9375  ', '14.7734375'], time:21.36318ms, swizzle: NOOP, TFLOPS: 102.94
      f16wmma(mma2x4+...+stage3+swizzle): ['-67.9375  ', '14.7734375'], time:19.75164ms, swizzle: 2048, TFLOPS: 111.33(+6.33%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-67.9375  ', '14.7734375'], time:19.44720ms, swizzle: 2048, TFLOPS: 113.08(+1.57%)
       f16wmma(...+stage3+dsmem+swizzle): ['-67.9375  ', '14.7734375'], time:19.70472ms, swizzle: 2048, TFLOPS: 111.60
       f16wmma(...+stage2+dsmem+swizzle): ['-67.9375  ', '14.7734375'], time:19.45617ms, swizzle: 2048, TFLOPS: 113.02
                             f16(cublas): ['-67.9375  ', '14.7734375'], time:19.24154ms, swizzle: NOOP, TFLOPS: 114.29(+1.07%)
                                  f16_th: ['-67.375   ', '14.9609375'], time:19.39365ms, swizzle: NOOP, TFLOPS: 113.39
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=2048
                    f16x8pack(t8x8+dbuf): ['1.59863281', '-1.5263671'], time:21.91691ms, swizzle: NOOP, TFLOPS: 50.17 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['1.59863281', '-1.5263671'], time:21.03848ms, swizzle: NOOP, TFLOPS: 52.26 (+4.18%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['1.89746094', '-1.4111328'], time:23.51725ms, swizzle: NOOP, TFLOPS: 46.75
                 f16wmma(mma4x2+warp2x4): ['1.89746094', '-1.4111328'], time:13.65675ms, swizzle: NOOP, TFLOPS: 80.51 (+54.05%)
          f16wmma(mma2x4+warp2x4+stage3): ['1.89746094', '-1.4111328'], time:10.03389ms, swizzle: NOOP, TFLOPS: 109.58(+36.11%)
          f16wmma(mma2x4+warp2x4+stage2): ['1.89746094', '-1.4111328'], time:9.967756ms, swizzle: NOOP, TFLOPS: 110.31(+0.66%)
        f16wmma(mma2x4+...+stage3+dsmem): ['1.89746094', '-1.4111328'], time:9.974980ms, swizzle: NOOP, TFLOPS: 110.23
        f16wmma(mma2x4+...+stage2+dsmem): ['1.89746094', '-1.4111328'], time:9.966444ms, swizzle: NOOP, TFLOPS: 110.32(+0.01%)
      f16wmma(mma2x4+...+stage3+swizzle): ['1.89746094', '-1.4111328'], time:10.01203ms, swizzle: 4096, TFLOPS: 109.82
      f16wmma(mma2x4+...+stage2+swizzle): ['1.89746094', '-1.4111328'], time:9.968757ms, swizzle: 4096, TFLOPS: 110.30
       f16wmma(...+stage3+dsmem+swizzle): ['1.89746094', '-1.4111328'], time:10.23325ms, swizzle: 4096, TFLOPS: 107.44
       f16wmma(...+stage2+dsmem+swizzle): ['1.89746094', '-1.4111328'], time:9.987878ms, swizzle: 4096, TFLOPS: 110.08
                             f16(cublas): ['1.89746094', '-1.4111328'], time:9.793019ms, swizzle: NOOP, TFLOPS: 112.28(+1.77%)
                                  f16_th: ['1.94628906', '-1.4042968'], time:9.699010ms, swizzle: NOOP, TFLOPS: 113.36(+0.97%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=4096
                    f16x8pack(t8x8+dbuf): ['-39.28125 ', '-30.484375'], time:51.11653ms, swizzle: NOOP, TFLOPS: 43.02 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['-39.28125 ', '-30.484375'], time:48.39284ms, swizzle: NOOP, TFLOPS: 45.44 (+5.63%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-39.0     ', '-30.34375 '], time:62.82320ms, swizzle: NOOP, TFLOPS: 35.00
                 f16wmma(mma4x2+warp2x4): ['-39.0     ', '-30.34375 '], time:28.40590ms, swizzle: NOOP, TFLOPS: 77.41 (+70.36%)
          f16wmma(mma2x4+warp2x4+stage3): ['-39.0     ', '-30.34375 '], time:21.87743ms, swizzle: NOOP, TFLOPS: 100.52(+29.84%)
          f16wmma(mma2x4+warp2x4+stage2): ['-39.0     ', '-30.34375 '], time:21.69656ms, swizzle: NOOP, TFLOPS: 101.35(+0.83%)
        f16wmma(mma2x4+...+stage3+dsmem): ['-39.0     ', '-30.34375 '], time:22.31538ms, swizzle: NOOP, TFLOPS: 98.54
        f16wmma(mma2x4+...+stage2+dsmem): ['-39.0     ', '-30.34375 '], time:21.39074ms, swizzle: NOOP, TFLOPS: 102.80(+1.43%)
      f16wmma(mma2x4+...+stage3+swizzle): ['-39.0     ', '-30.34375 '], time:19.75109ms, swizzle: 4096, TFLOPS: 111.34(+8.30%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-39.0     ', '-30.34375 '], time:19.45185ms, swizzle: 4096, TFLOPS: 113.05(+1.54%)
       f16wmma(...+stage3+dsmem+swizzle): ['-39.0     ', '-30.34375 '], time:19.83349ms, swizzle: 4096, TFLOPS: 110.87
       f16wmma(...+stage2+dsmem+swizzle): ['-39.0     ', '-30.34375 '], time:19.57380ms, swizzle: 4096, TFLOPS: 112.35
                             f16(cublas): ['-39.0     ', '-30.34375 '], time:19.32525ms, swizzle: NOOP, TFLOPS: 113.79(+0.66%)
                                  f16_th: ['-38.9375  ', '-30.265625'], time:19.11854ms, swizzle: NOOP, TFLOPS: 115.02(+1.08%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=8192
                    f16x8pack(t8x8+dbuf): ['-69.125   ', '14.8359375'], time:103.2802ms, swizzle: NOOP, TFLOPS: 42.58 (+0.00%)
                f16x8pack(t8x8+k16+dbuf): ['-69.125   ', '14.8359375'], time:101.1459ms, swizzle: NOOP, TFLOPS: 43.48 (+2.11%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-67.9375  ', '14.7734375'], time:124.0100ms, swizzle: NOOP, TFLOPS: 35.47
                 f16wmma(mma4x2+warp2x4): ['-67.9375  ', '14.7734375'], time:55.98418ms, swizzle: NOOP, TFLOPS: 78.56 (+80.67%)
          f16wmma(mma2x4+warp2x4+stage3): ['-67.9375  ', '14.7734375'], time:49.67854ms, swizzle: NOOP, TFLOPS: 88.53 (+12.69%)
          f16wmma(mma2x4+warp2x4+stage2): ['-67.9375  ', '14.7734375'], time:49.78499ms, swizzle: NOOP, TFLOPS: 88.34
        f16wmma(mma2x4+...+stage3+dsmem): ['-67.9375  ', '14.7734375'], time:49.68118ms, swizzle: NOOP, TFLOPS: 88.53
        f16wmma(mma2x4+...+stage2+dsmem): ['-67.9375  ', '14.7734375'], time:49.85711ms, swizzle: NOOP, TFLOPS: 88.21
      f16wmma(mma2x4+...+stage3+swizzle): ['-67.9375  ', '14.7734375'], time:39.40358ms, swizzle: 4096, TFLOPS: 111.62(+26.08%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-67.9375  ', '14.7734375'], time:39.02866ms, swizzle: 4096, TFLOPS: 112.69(+0.96%)
       f16wmma(...+stage3+dsmem+swizzle): ['-67.9375  ', '14.7734375'], time:39.64929ms, swizzle: 4096, TFLOPS: 110.92
       f16wmma(...+stage2+dsmem+swizzle): ['-67.9375  ', '14.7734375'], time:38.88757ms, swizzle: 4096, TFLOPS: 113.10(+0.36%)
                             f16(cublas): ['-67.9375  ', '14.7734375'], time:38.70918ms, swizzle: NOOP, TFLOPS: 113.62(+0.46%)
                                  f16_th: ['-67.375   ', '14.9609375'], time:38.53211ms, swizzle: NOOP, TFLOPS: 114.14(+0.46%)
----------------------------------------------------------------------------------------------------------------------------------
```
