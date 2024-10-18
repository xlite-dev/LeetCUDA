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

目前最优的实现，在L20上（理论Tensor Cores FP16算力为 119.5 TFLOPS），能达到cuBLAS大概95%~98%左右的性能(105-110 TFLOPS vs 105-115 TFLOPS)，部分case会超越cuBLAS。已知问题为bank conflicts没有完全消除，目前通过padding的方式缓解bank conflicts会导致shared memory浪费，也会影响SM occupancy。并且尚未手工实现Warp swizzle(受限于WMMA API的灵活性以及本人的能力)，后续将会尝试通过MMA PTX实现warp swizzle。

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
                     f16x8pack(t8x8+bcf): ['-54.125   ', '11.21875  '], time:1.423144ms, swizzle: NOOP, TFLOPS: 48.29 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-54.125   ', '11.21875  '], time:1.404595ms, swizzle: NOOP, TFLOPS: 48.92 (+1.32%)
                f16x8pack(t8x8+k16+dbuf): ['-54.125   ', '11.21875  '], time:1.328659ms, swizzle: NOOP, TFLOPS: 51.72 (+5.72%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-53.9375  ', '11.15625  '], time:1.489758ms, swizzle: NOOP, TFLOPS: 46.13
                 f16wmma(mma4x2+warp2x4): ['-53.9375  ', '11.15625  '], time:0.940990ms, swizzle: NOOP, TFLOPS: 73.03 (+41.20%)
          f16wmma(mma2x4+warp2x4+stage3): ['-53.9375  ', '11.15625  '], time:0.695109ms, swizzle: NOOP, TFLOPS: 98.86 (+35.37%)
          f16wmma(mma2x4+warp2x4+stage2): ['-53.9375  ', '11.15625  '], time:0.696945ms, swizzle: NOOP, TFLOPS: 98.60
        f16wmma(mma2x4+...+stage3+dsmem): ['-53.9375  ', '11.15625  '], time:0.699973ms, swizzle: NOOP, TFLOPS: 98.17
        f16wmma(mma2x4+...+stage2+dsmem): ['-53.9375  ', '11.15625  '], time:0.695180ms, swizzle: NOOP, TFLOPS: 98.85
      f16wmma(mma2x4+...+stage3+swizzle): ['-53.9375  ', '11.15625  '], time:0.694012ms, swizzle: 1024, TFLOPS: 99.02 (+0.16%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-53.9375  ', '11.15625  '], time:0.685882ms, swizzle: 1024, TFLOPS: 100.19(+1.19%)
       f16wmma(...+stage3+dsmem+swizzle): ['-53.9375  ', '11.15625  '], time:0.700545ms, swizzle: 1024, TFLOPS: 98.09
       f16wmma(...+stage2+dsmem+swizzle): ['-53.9375  ', '11.15625  '], time:0.685405ms, swizzle: 1024, TFLOPS: 100.26(+0.07%)
                             f16(cublas): ['-53.9375  ', '11.15625  '], time:0.847744ms, swizzle: NOOP, TFLOPS: 81.06
                                  f16_th: ['-53.96875 ', '11.1171875'], time:0.660753ms, swizzle: NOOP, TFLOPS: 104.00(+3.73%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=4096
                     f16x8pack(t8x8+bcf): ['0.89404297', '-13.5625  '], time:2.833724ms, swizzle: NOOP, TFLOPS: 48.50 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['0.89404297', '-13.5625  '], time:2.795863ms, swizzle: NOOP, TFLOPS: 49.16 (+1.35%)
                f16x8pack(t8x8+k16+dbuf): ['0.89404297', '-13.5625  '], time:2.630090ms, swizzle: NOOP, TFLOPS: 52.26 (+6.30%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['1.36816406', '-13.765625'], time:2.964925ms, swizzle: NOOP, TFLOPS: 46.35
                 f16wmma(mma4x2+warp2x4): ['1.36816406', '-13.765625'], time:1.837420ms, swizzle: NOOP, TFLOPS: 74.80 (+43.14%)
          f16wmma(mma2x4+warp2x4+stage3): ['1.36816406', '-13.765625'], time:1.356744ms, swizzle: NOOP, TFLOPS: 101.30(+35.43%)
          f16wmma(mma2x4+warp2x4+stage2): ['1.36816406', '-13.765625'], time:1.356244ms, swizzle: NOOP, TFLOPS: 101.34(+0.04%)
        f16wmma(mma2x4+...+stage3+dsmem): ['1.36816406', '-13.765625'], time:1.360177ms, swizzle: NOOP, TFLOPS: 101.04
        f16wmma(mma2x4+...+stage2+dsmem): ['1.36816406', '-13.765625'], time:1.353812ms, swizzle: NOOP, TFLOPS: 101.52(+0.18%)
      f16wmma(mma2x4+...+stage3+swizzle): ['1.36816406', '-13.765625'], time:1.350283ms, swizzle: 1024, TFLOPS: 101.79(+0.26%)
      f16wmma(mma2x4+...+stage2+swizzle): ['1.36816406', '-13.765625'], time:1.332807ms, swizzle: 1024, TFLOPS: 103.12(+1.31%)
       f16wmma(...+stage3+dsmem+swizzle): ['1.36816406', '-13.765625'], time:1.365542ms, swizzle: 1024, TFLOPS: 100.65
       f16wmma(...+stage2+dsmem+swizzle): ['1.36816406', '-13.765625'], time:1.332807ms, swizzle: 1024, TFLOPS: 103.12
                             f16(cublas): ['1.36816406', '-13.765625'], time:1.488780ms, swizzle: NOOP, TFLOPS: 92.32
                                  f16_th: ['1.39550781', '-13.898437'], time:1.286172ms, swizzle: NOOP, TFLOPS: 106.86(+3.63%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=8192
                     f16x8pack(t8x8+bcf): ['-27.078125', '-48.875   '], time:5.790376ms, swizzle: NOOP, TFLOPS: 47.47 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-27.078125', '-48.875   '], time:5.631232ms, swizzle: NOOP, TFLOPS: 48.81 (+2.83%)
                f16x8pack(t8x8+k16+dbuf): ['-27.078125', '-48.875   '], time:5.383110ms, swizzle: NOOP, TFLOPS: 51.06 (+4.61%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-27.125   ', '-48.5625  '], time:5.904579ms, swizzle: NOOP, TFLOPS: 46.55
                 f16wmma(mma4x2+warp2x4): ['-27.125   ', '-48.5625  '], time:3.653526ms, swizzle: NOOP, TFLOPS: 75.24 (+47.34%)
          f16wmma(mma2x4+warp2x4+stage3): ['-27.125   ', '-48.5625  '], time:2.665686ms, swizzle: NOOP, TFLOPS: 103.12(+37.06%)
          f16wmma(mma2x4+warp2x4+stage2): ['-27.125   ', '-48.5625  '], time:2.672934ms, swizzle: NOOP, TFLOPS: 102.84
        f16wmma(mma2x4+...+stage3+dsmem): ['-27.125   ', '-48.5625  '], time:2.681159ms, swizzle: NOOP, TFLOPS: 102.52
        f16wmma(mma2x4+...+stage2+dsmem): ['-27.125   ', '-48.5625  '], time:2.662348ms, swizzle: NOOP, TFLOPS: 103.25(+0.13%)
      f16wmma(mma2x4+...+stage3+swizzle): ['-27.125   ', '-48.5625  '], time:2.672886ms, swizzle: 1024, TFLOPS: 102.84
      f16wmma(mma2x4+...+stage2+swizzle): ['-27.125   ', '-48.5625  '], time:2.631425ms, swizzle: 1024, TFLOPS: 104.46(+1.18%)
       f16wmma(...+stage3+dsmem+swizzle): ['-27.125   ', '-48.5625  '], time:2.698731ms, swizzle: 1024, TFLOPS: 101.85
       f16wmma(...+stage2+dsmem+swizzle): ['-27.125   ', '-48.5625  '], time:2.622413ms, swizzle: 1024, TFLOPS: 104.82(+0.34%)
                             f16(cublas): ['-27.109375', '-48.9375  '], time:2.655792ms, swizzle: NOOP, TFLOPS: 103.50
                                  f16_th: ['-27.1875  ', '-48.90625 '], time:2.405142ms, swizzle: NOOP, TFLOPS: 114.29(+9.03%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=2048
                     f16x8pack(t8x8+bcf): ['-54.125   ', '11.21875  '], time:2.718329ms, swizzle: NOOP, TFLOPS: 50.56 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-54.125   ', '11.21875  '], time:2.678704ms, swizzle: NOOP, TFLOPS: 51.31 (+1.48%)
                f16x8pack(t8x8+k16+dbuf): ['-54.125   ', '11.21875  '], time:2.537894ms, swizzle: NOOP, TFLOPS: 54.15 (+5.55%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-53.9375  ', '11.15625  '], time:2.964472ms, swizzle: NOOP, TFLOPS: 46.36
                 f16wmma(mma4x2+warp2x4): ['-53.9375  ', '11.15625  '], time:1.844596ms, swizzle: NOOP, TFLOPS: 74.51 (+37.59%)
          f16wmma(mma2x4+warp2x4+stage3): ['-53.9375  ', '11.15625  '], time:1.311135ms, swizzle: NOOP, TFLOPS: 104.82(+40.69%)
          f16wmma(mma2x4+warp2x4+stage2): ['-53.9375  ', '11.15625  '], time:1.317501ms, swizzle: NOOP, TFLOPS: 104.32
        f16wmma(mma2x4+...+stage3+dsmem): ['-53.9375  ', '11.15625  '], time:1.320862ms, swizzle: NOOP, TFLOPS: 104.05
        f16wmma(mma2x4+...+stage2+dsmem): ['-53.9375  ', '11.15625  '], time:1.315283ms, swizzle: NOOP, TFLOPS: 104.49
      f16wmma(mma2x4+...+stage3+swizzle): ['-53.9375  ', '11.15625  '], time:1.308703ms, swizzle: 2048, TFLOPS: 105.02(+0.19%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-53.9375  ', '11.15625  '], time:1.298141ms, swizzle: 2048, TFLOPS: 105.87(+0.81%)
       f16wmma(...+stage3+dsmem+swizzle): ['-53.9375  ', '11.15625  '], time:1.321554ms, swizzle: 2048, TFLOPS: 104.00
       f16wmma(...+stage2+dsmem+swizzle): ['-53.9375  ', '11.15625  '], time:1.297688ms, swizzle: 2048, TFLOPS: 105.91(+0.03%)
                             f16(cublas): ['-53.9375  ', '11.15625  '], time:1.445293ms, swizzle: NOOP, TFLOPS: 95.09
                                  f16_th: ['-53.96875 ', '11.1171875'], time:1.301765ms, swizzle: NOOP, TFLOPS: 105.58
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=4096
                     f16x8pack(t8x8+bcf): ['0.89404297', '-13.5625  '], time:5.457043ms, swizzle: NOOP, TFLOPS: 50.37 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['0.89404297', '-13.5625  '], time:5.358934ms, swizzle: NOOP, TFLOPS: 51.29 (+1.83%)
                f16x8pack(t8x8+k16+dbuf): ['0.89404297', '-13.5625  '], time:5.148506ms, swizzle: NOOP, TFLOPS: 53.39 (+4.09%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['1.36816406', '-13.765625'], time:5.894541ms, swizzle: NOOP, TFLOPS: 46.63
                 f16wmma(mma4x2+warp2x4): ['1.36816406', '-13.765625'], time:3.650927ms, swizzle: NOOP, TFLOPS: 75.29 (+41.02%)
          f16wmma(mma2x4+warp2x4+stage3): ['1.36816406', '-13.765625'], time:2.572631ms, swizzle: NOOP, TFLOPS: 106.85(+41.91%)
          f16wmma(mma2x4+warp2x4+stage2): ['1.36816406', '-13.765625'], time:2.570700ms, swizzle: NOOP, TFLOPS: 106.93(+0.08%)
        f16wmma(mma2x4+...+stage3+dsmem): ['1.36816406', '-13.765625'], time:2.577877ms, swizzle: NOOP, TFLOPS: 106.63
        f16wmma(mma2x4+...+stage2+dsmem): ['1.36816406', '-13.765625'], time:2.567315ms, swizzle: NOOP, TFLOPS: 107.07(+0.13%)
      f16wmma(mma2x4+...+stage3+swizzle): ['1.36816406', '-13.765625'], time:2.557826ms, swizzle: 2048, TFLOPS: 107.47(+0.37%)
      f16wmma(mma2x4+...+stage2+swizzle): ['1.36816406', '-13.765625'], time:2.533388ms, swizzle: 2048, TFLOPS: 108.50(+0.96%)
       f16wmma(...+stage3+dsmem+swizzle): ['1.36816406', '-13.765625'], time:2.584576ms, swizzle: 2048, TFLOPS: 106.35
       f16wmma(...+stage2+dsmem+swizzle): ['1.36816406', '-13.765625'], time:2.533221ms, swizzle: 2048, TFLOPS: 108.51(+0.01%)
                             f16(cublas): ['1.36816406', '-13.765625'], time:2.638125ms, swizzle: NOOP, TFLOPS: 104.19
                                  f16_th: ['1.39550781', '-13.898437'], time:2.550959ms, swizzle: NOOP, TFLOPS: 107.75
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=8192
                     f16x8pack(t8x8+bcf): ['-27.078125', '-48.875   '], time:12.04879ms, swizzle: NOOP, TFLOPS: 45.63 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-27.078125', '-48.875   '], time:11.79327ms, swizzle: NOOP, TFLOPS: 46.62 (+2.17%)
                f16x8pack(t8x8+k16+dbuf): ['-27.078125', '-48.875   '], time:11.44108ms, swizzle: NOOP, TFLOPS: 48.05 (+3.08%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-27.125   ', '-48.5625  '], time:12.29424ms, swizzle: NOOP, TFLOPS: 44.72
                 f16wmma(mma4x2+warp2x4): ['-27.125   ', '-48.5625  '], time:7.309770ms, swizzle: NOOP, TFLOPS: 75.21 (+56.52%)
          f16wmma(mma2x4+warp2x4+stage3): ['-27.125   ', '-48.5625  '], time:5.307912ms, swizzle: NOOP, TFLOPS: 103.57(+37.71%)
          f16wmma(mma2x4+warp2x4+stage2): ['-27.125   ', '-48.5625  '], time:5.165386ms, swizzle: NOOP, TFLOPS: 106.43(+2.76%)
        f16wmma(mma2x4+...+stage3+dsmem): ['-27.125   ', '-48.5625  '], time:5.312108ms, swizzle: NOOP, TFLOPS: 103.49
        f16wmma(mma2x4+...+stage2+dsmem): ['-27.125   ', '-48.5625  '], time:5.298399ms, swizzle: NOOP, TFLOPS: 103.76
      f16wmma(mma2x4+...+stage3+swizzle): ['-27.125   ', '-48.5625  '], time:5.130910ms, swizzle: 2048, TFLOPS: 107.15(+0.67%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-27.125   ', '-48.5625  '], time:5.038666ms, swizzle: 2048, TFLOPS: 109.11(+1.83%)
       f16wmma(...+stage3+dsmem+swizzle): ['-27.125   ', '-48.5625  '], time:5.142164ms, swizzle: 2048, TFLOPS: 106.91
       f16wmma(...+stage2+dsmem+swizzle): ['-27.125   ', '-48.5625  '], time:5.037188ms, swizzle: 2048, TFLOPS: 109.14(+0.03%)
                             f16(cublas): ['-27.125   ', '-48.5625  '], time:5.047488ms, swizzle: NOOP, TFLOPS: 108.92
                                  f16_th: ['-27.203125', '-48.90625 '], time:4.914093ms, swizzle: NOOP, TFLOPS: 111.87(+2.50%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=2048
                     f16x8pack(t8x8+bcf): ['-54.125   ', '11.21875  '], time:5.287313ms, swizzle: NOOP, TFLOPS: 51.99 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-54.125   ', '11.21875  '], time:5.211281ms, swizzle: NOOP, TFLOPS: 52.75 (+1.46%)
                f16x8pack(t8x8+k16+dbuf): ['-54.125   ', '11.21875  '], time:4.944944ms, swizzle: NOOP, TFLOPS: 55.59 (+5.39%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-53.9375  ', '11.15625  '], time:5.902266ms, swizzle: NOOP, TFLOPS: 46.57
                 f16wmma(mma4x2+warp2x4): ['-53.9375  ', '11.15625  '], time:3.550195ms, swizzle: NOOP, TFLOPS: 77.43 (+39.29%)
          f16wmma(mma2x4+warp2x4+stage3): ['-53.9375  ', '11.15625  '], time:2.552223ms, swizzle: NOOP, TFLOPS: 107.70(+39.10%)
          f16wmma(mma2x4+warp2x4+stage2): ['-53.9375  ', '11.15625  '], time:2.559995ms, swizzle: NOOP, TFLOPS: 107.37
        f16wmma(mma2x4+...+stage3+dsmem): ['-53.9375  ', '11.15625  '], time:2.554583ms, swizzle: NOOP, TFLOPS: 107.60
        f16wmma(mma2x4+...+stage2+dsmem): ['-53.9375  ', '11.15625  '], time:2.546525ms, swizzle: NOOP, TFLOPS: 107.94(+0.22%)
      f16wmma(mma2x4+...+stage3+swizzle): ['-53.9375  ', '11.15625  '], time:2.533578ms, swizzle: 4096, TFLOPS: 108.49(+0.51%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-53.9375  ', '11.15625  '], time:2.517557ms, swizzle: 4096, TFLOPS: 109.18(+0.64%)
       f16wmma(...+stage3+dsmem+swizzle): ['-53.9375  ', '11.15625  '], time:2.556943ms, swizzle: 4096, TFLOPS: 107.50
       f16wmma(...+stage2+dsmem+swizzle): ['-53.9375  ', '11.15625  '], time:2.516031ms, swizzle: 4096, TFLOPS: 109.25(+0.06%)
                             f16(cublas): ['-53.9375  ', '11.15625  '], time:2.604794ms, swizzle: NOOP, TFLOPS: 105.53
                                  f16_th: ['-53.96875 ', '11.1171875'], time:2.394223ms, swizzle: NOOP, TFLOPS: 114.81(+5.09%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=4096
                     f16x8pack(t8x8+bcf): ['0.89404297', '-13.5625  '], time:11.72189ms, swizzle: NOOP, TFLOPS: 46.90 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['0.89404297', '-13.5625  '], time:11.64755ms, swizzle: NOOP, TFLOPS: 47.20 (+0.64%)
                f16x8pack(t8x8+k16+dbuf): ['0.89404297', '-13.5625  '], time:11.53805ms, swizzle: NOOP, TFLOPS: 47.65 (+0.95%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['1.36816406', '-13.765625'], time:15.53082ms, swizzle: NOOP, TFLOPS: 35.40
                 f16wmma(mma4x2+warp2x4): ['1.36816406', '-13.765625'], time:7.273221ms, swizzle: NOOP, TFLOPS: 75.59 (+58.64%)
          f16wmma(mma2x4+warp2x4+stage3): ['1.36816406', '-13.765625'], time:5.248022ms, swizzle: NOOP, TFLOPS: 104.75(+38.59%)
          f16wmma(mma2x4+warp2x4+stage2): ['1.36816406', '-13.765625'], time:5.237030ms, swizzle: NOOP, TFLOPS: 104.97(+0.21%)
        f16wmma(mma2x4+...+stage3+dsmem): ['1.36816406', '-13.765625'], time:5.314159ms, swizzle: NOOP, TFLOPS: 103.45
        f16wmma(mma2x4+...+stage2+dsmem): ['1.36816406', '-13.765625'], time:5.238246ms, swizzle: NOOP, TFLOPS: 104.95
      f16wmma(mma2x4+...+stage3+swizzle): ['1.36816406', '-13.765625'], time:5.007338ms, swizzle: 4096, TFLOPS: 109.79(+4.59%)
      f16wmma(mma2x4+...+stage2+swizzle): ['1.36816406', '-13.765625'], time:4.961037ms, swizzle: 4096, TFLOPS: 110.81(+0.93%)
       f16wmma(...+stage3+dsmem+swizzle): ['1.36816406', '-13.765625'], time:5.051326ms, swizzle: 4096, TFLOPS: 108.83
       f16wmma(...+stage2+dsmem+swizzle): ['1.36816406', '-13.765625'], time:4.959821ms, swizzle: 4096, TFLOPS: 110.84(+0.02%)
                             f16(cublas): ['1.36816406', '-13.765625'], time:4.990649ms, swizzle: NOOP, TFLOPS: 110.16
                                  f16_th: ['1.39550781', '-13.898437'], time:4.902839ms, swizzle: NOOP, TFLOPS: 112.13(+1.16%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=8192
                     f16x8pack(t8x8+bcf): ['-27.078125', '-48.875   '], time:24.46112ms, swizzle: NOOP, TFLOPS: 44.95 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-27.078125', '-48.875   '], time:24.18811ms, swizzle: NOOP, TFLOPS: 45.46 (+1.13%)
                f16x8pack(t8x8+k16+dbuf): ['-27.078125', '-48.875   '], time:23.81680ms, swizzle: NOOP, TFLOPS: 46.17 (+1.56%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-27.125   ', '-48.5625  '], time:31.02438ms, swizzle: NOOP, TFLOPS: 35.44
                 f16wmma(mma4x2+warp2x4): ['-27.125   ', '-48.5625  '], time:14.35780ms, swizzle: NOOP, TFLOPS: 76.58 (+65.88%)
          f16wmma(mma2x4+warp2x4+stage3): ['-27.125   ', '-48.5625  '], time:12.45610ms, swizzle: NOOP, TFLOPS: 88.27 (+15.27%)
          f16wmma(mma2x4+warp2x4+stage2): ['-27.125   ', '-48.5625  '], time:12.35179ms, swizzle: NOOP, TFLOPS: 89.02 (+0.84%)
        f16wmma(mma2x4+...+stage3+dsmem): ['-27.125   ', '-48.5625  '], time:12.47189ms, swizzle: NOOP, TFLOPS: 88.16
        f16wmma(mma2x4+...+stage2+dsmem): ['-27.125   ', '-48.5625  '], time:12.34390ms, swizzle: NOOP, TFLOPS: 89.07 (+0.06%)
      f16wmma(mma2x4+...+stage3+swizzle): ['-27.125   ', '-48.5625  '], time:9.990620ms, swizzle: 4096, TFLOPS: 110.05(+23.55%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-27.125   ', '-48.5625  '], time:9.921455ms, swizzle: 4096, TFLOPS: 110.82(+0.70%)
       f16wmma(...+stage3+dsmem+swizzle): ['-27.125   ', '-48.5625  '], time:10.17749ms, swizzle: 4096, TFLOPS: 108.03
       f16wmma(...+stage2+dsmem+swizzle): ['-27.125   ', '-48.5625  '], time:9.977889ms, swizzle: 4096, TFLOPS: 110.19
                             f16(cublas): ['-27.125   ', '-48.5625  '], time:9.716081ms, swizzle: NOOP, TFLOPS: 113.16(+2.11%)
                                  f16_th: ['-27.203125', '-48.90625 '], time:9.708857ms, swizzle: NOOP, TFLOPS: 113.25(+0.07%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=2048
                     f16x8pack(t8x8+bcf): ['-54.125   ', '11.21875  '], time:2.729988ms, swizzle: NOOP, TFLOPS: 50.34 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-54.125   ', '11.21875  '], time:2.678513ms, swizzle: NOOP, TFLOPS: 51.31 (+1.92%)
                f16x8pack(t8x8+k16+dbuf): ['-54.125   ', '11.21875  '], time:2.537012ms, swizzle: NOOP, TFLOPS: 54.17 (+5.58%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-53.9375  ', '11.15625  '], time:3.013658ms, swizzle: NOOP, TFLOPS: 45.61
                 f16wmma(mma4x2+warp2x4): ['-53.9375  ', '11.15625  '], time:1.826381ms, swizzle: NOOP, TFLOPS: 75.25 (+38.91%)
          f16wmma(mma2x4+warp2x4+stage3): ['-53.9375  ', '11.15625  '], time:1.315021ms, swizzle: NOOP, TFLOPS: 104.51(+38.89%)
          f16wmma(mma2x4+warp2x4+stage2): ['-53.9375  ', '11.15625  '], time:1.323676ms, swizzle: NOOP, TFLOPS: 103.83
        f16wmma(mma2x4+...+stage3+dsmem): ['-53.9375  ', '11.15625  '], time:1.323461ms, swizzle: NOOP, TFLOPS: 103.85
        f16wmma(mma2x4+...+stage2+dsmem): ['-53.9375  ', '11.15625  '], time:1.317262ms, swizzle: NOOP, TFLOPS: 104.34
      f16wmma(mma2x4+...+stage3+swizzle): ['-53.9375  ', '11.15625  '], time:1.310777ms, swizzle: 1024, TFLOPS: 104.85(+0.32%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-53.9375  ', '11.15625  '], time:1.299142ms, swizzle: 1024, TFLOPS: 105.79(+0.90%)
       f16wmma(...+stage3+dsmem+swizzle): ['-53.9375  ', '11.15625  '], time:1.323223ms, swizzle: 1024, TFLOPS: 103.87
       f16wmma(...+stage2+dsmem+swizzle): ['-53.9375  ', '11.15625  '], time:1.299834ms, swizzle: 1024, TFLOPS: 105.74
                             f16(cublas): ['-53.9375  ', '11.15625  '], time:1.431703ms, swizzle: NOOP, TFLOPS: 96.00
                                  f16_th: ['-53.96875 ', '11.1171875'], time:1.301836ms, swizzle: NOOP, TFLOPS: 105.57
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=4096
                     f16x8pack(t8x8+bcf): ['0.89404297', '-13.5625  '], time:5.494832ms, swizzle: NOOP, TFLOPS: 50.02 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['0.89404297', '-13.5625  '], time:5.380392ms, swizzle: NOOP, TFLOPS: 51.09 (+2.13%)
                f16x8pack(t8x8+k16+dbuf): ['0.89404297', '-13.5625  '], time:5.194044ms, swizzle: NOOP, TFLOPS: 52.92 (+3.59%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['1.36816406', '-13.765625'], time:5.911517ms, swizzle: NOOP, TFLOPS: 46.50
                 f16wmma(mma4x2+warp2x4): ['1.36816406', '-13.765625'], time:3.633975ms, swizzle: NOOP, TFLOPS: 75.64 (+42.93%)
          f16wmma(mma2x4+warp2x4+stage3): ['1.36816406', '-13.765625'], time:2.575016ms, swizzle: NOOP, TFLOPS: 106.75(+41.12%)
          f16wmma(mma2x4+warp2x4+stage2): ['1.36816406', '-13.765625'], time:2.586603ms, swizzle: NOOP, TFLOPS: 106.27
        f16wmma(mma2x4+...+stage3+dsmem): ['1.36816406', '-13.765625'], time:2.582311ms, swizzle: NOOP, TFLOPS: 106.45
        f16wmma(mma2x4+...+stage2+dsmem): ['1.36816406', '-13.765625'], time:2.569913ms, swizzle: NOOP, TFLOPS: 106.96(+0.20%)
      f16wmma(mma2x4+...+stage3+swizzle): ['1.36816406', '-13.765625'], time:2.561902ms, swizzle: 1024, TFLOPS: 107.29(+0.31%)
      f16wmma(mma2x4+...+stage2+swizzle): ['1.36816406', '-13.765625'], time:2.536106ms, swizzle: 1024, TFLOPS: 108.39(+1.02%)
       f16wmma(...+stage3+dsmem+swizzle): ['1.36816406', '-13.765625'], time:2.587747ms, swizzle: 1024, TFLOPS: 106.22
       f16wmma(...+stage2+dsmem+swizzle): ['1.36816406', '-13.765625'], time:2.536773ms, swizzle: 1024, TFLOPS: 108.36
                             f16(cublas): ['1.36816406', '-13.765625'], time:2.633023ms, swizzle: NOOP, TFLOPS: 104.40
                                  f16_th: ['1.39550781', '-13.898437'], time:2.552318ms, swizzle: NOOP, TFLOPS: 107.70
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=8192
                     f16x8pack(t8x8+bcf): ['-27.078125', '-48.875   '], time:11.24224ms, swizzle: NOOP, TFLOPS: 48.90 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-27.078125', '-48.875   '], time:10.98401ms, swizzle: NOOP, TFLOPS: 50.05 (+2.35%)
                f16x8pack(t8x8+k16+dbuf): ['-27.078125', '-48.875   '], time:10.58707ms, swizzle: NOOP, TFLOPS: 51.93 (+3.75%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-27.125   ', '-48.5625  '], time:11.71593ms, swizzle: NOOP, TFLOPS: 46.92
                 f16wmma(mma4x2+warp2x4): ['-27.125   ', '-48.5625  '], time:7.178306ms, swizzle: NOOP, TFLOPS: 76.59 (+47.49%)
          f16wmma(mma2x4+warp2x4+stage3): ['-27.125   ', '-48.5625  '], time:5.056834ms, swizzle: NOOP, TFLOPS: 108.72(+41.95%)
          f16wmma(mma2x4+warp2x4+stage2): ['-27.125   ', '-48.5625  '], time:5.096650ms, swizzle: NOOP, TFLOPS: 107.87
        f16wmma(mma2x4+...+stage3+dsmem): ['-27.125   ', '-48.5625  '], time:5.098223ms, swizzle: NOOP, TFLOPS: 107.83
        f16wmma(mma2x4+...+stage2+dsmem): ['-27.125   ', '-48.5625  '], time:5.076169ms, swizzle: NOOP, TFLOPS: 108.30
      f16wmma(mma2x4+...+stage3+swizzle): ['-27.125   ', '-48.5625  '], time:5.091643ms, swizzle: 1024, TFLOPS: 107.97
      f16wmma(mma2x4+...+stage2+swizzle): ['-27.125   ', '-48.5625  '], time:5.085301ms, swizzle: 1024, TFLOPS: 108.11
       f16wmma(...+stage3+dsmem+swizzle): ['-27.125   ', '-48.5625  '], time:5.169296ms, swizzle: 1024, TFLOPS: 106.35
       f16wmma(...+stage2+dsmem+swizzle): ['-27.125   ', '-48.5625  '], time:5.046343ms, swizzle: 1024, TFLOPS: 108.94(+0.21%)
                             f16(cublas): ['-27.125   ', '-48.5625  '], time:5.101323ms, swizzle: NOOP, TFLOPS: 107.77
                                  f16_th: ['-27.1875  ', '-48.90625 '], time:4.802632ms, swizzle: NOOP, TFLOPS: 114.47(+5.07%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=2048
                     f16x8pack(t8x8+bcf): ['-54.125   ', '11.21875  '], time:5.380392ms, swizzle: NOOP, TFLOPS: 51.09 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-54.125   ', '11.21875  '], time:5.314040ms, swizzle: NOOP, TFLOPS: 51.73 (+1.25%)
                f16x8pack(t8x8+k16+dbuf): ['-54.125   ', '11.21875  '], time:5.108165ms, swizzle: NOOP, TFLOPS: 53.81 (+4.03%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-53.9375  ', '11.15625  '], time:5.897569ms, swizzle: NOOP, TFLOPS: 46.61
                 f16wmma(mma4x2+warp2x4): ['-53.9375  ', '11.15625  '], time:3.570127ms, swizzle: NOOP, TFLOPS: 76.99 (+43.08%)
          f16wmma(mma2x4+warp2x4+stage3): ['-53.9375  ', '11.15625  '], time:2.553844ms, swizzle: NOOP, TFLOPS: 107.63(+39.79%)
          f16wmma(mma2x4+warp2x4+stage2): ['-53.9375  ', '11.15625  '], time:2.554178ms, swizzle: NOOP, TFLOPS: 107.62
        f16wmma(mma2x4+...+stage3+dsmem): ['-53.9375  ', '11.15625  '], time:2.556610ms, swizzle: NOOP, TFLOPS: 107.52
        f16wmma(mma2x4+...+stage2+dsmem): ['-53.9375  ', '11.15625  '], time:2.549052ms, swizzle: NOOP, TFLOPS: 107.84(+0.19%)
      f16wmma(mma2x4+...+stage3+swizzle): ['-53.9375  ', '11.15625  '], time:2.535676ms, swizzle: 2048, TFLOPS: 108.40(+0.53%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-53.9375  ', '11.15625  '], time:2.518987ms, swizzle: 2048, TFLOPS: 109.12(+0.66%)
       f16wmma(...+stage3+dsmem+swizzle): ['-53.9375  ', '11.15625  '], time:2.558732ms, swizzle: 2048, TFLOPS: 107.43
       f16wmma(...+stage2+dsmem+swizzle): ['-53.9375  ', '11.15625  '], time:2.519011ms, swizzle: 2048, TFLOPS: 109.12
                             f16(cublas): ['-53.9375  ', '11.15625  '], time:2.600860ms, swizzle: NOOP, TFLOPS: 105.69
                                  f16_th: ['-53.96875 ', '11.1171875'], time:2.395844ms, swizzle: NOOP, TFLOPS: 114.73(+5.14%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=4096
                     f16x8pack(t8x8+bcf): ['0.89404297', '-13.5625  '], time:10.90965ms, swizzle: NOOP, TFLOPS: 50.39 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['0.89404297', '-13.5625  '], time:10.92147ms, swizzle: NOOP, TFLOPS: 50.34
                f16x8pack(t8x8+k16+dbuf): ['0.89404297', '-13.5625  '], time:10.43622ms, swizzle: NOOP, TFLOPS: 52.68 (+4.54%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['1.36816406', '-13.765625'], time:11.74299ms, swizzle: NOOP, TFLOPS: 46.82
                 f16wmma(mma4x2+warp2x4): ['1.36816406', '-13.765625'], time:6.996607ms, swizzle: NOOP, TFLOPS: 78.57 (+49.16%)
          f16wmma(mma2x4+warp2x4+stage3): ['1.36816406', '-13.765625'], time:4.981565ms, swizzle: NOOP, TFLOPS: 110.36(+40.45%)
          f16wmma(mma2x4+warp2x4+stage2): ['1.36816406', '-13.765625'], time:5.003023ms, swizzle: NOOP, TFLOPS: 109.88
        f16wmma(mma2x4+...+stage3+dsmem): ['1.36816406', '-13.765625'], time:5.016517ms, swizzle: NOOP, TFLOPS: 109.59
        f16wmma(mma2x4+...+stage2+dsmem): ['1.36816406', '-13.765625'], time:4.997205ms, swizzle: NOOP, TFLOPS: 110.01
      f16wmma(mma2x4+...+stage3+swizzle): ['1.36816406', '-13.765625'], time:4.992318ms, swizzle: 2048, TFLOPS: 110.12
      f16wmma(mma2x4+...+stage2+swizzle): ['1.36816406', '-13.765625'], time:4.994416ms, swizzle: 2048, TFLOPS: 110.07
       f16wmma(...+stage3+dsmem+swizzle): ['1.36816406', '-13.765625'], time:5.106782ms, swizzle: 2048, TFLOPS: 107.65
       f16wmma(...+stage2+dsmem+swizzle): ['1.36816406', '-13.765625'], time:4.989671ms, swizzle: 2048, TFLOPS: 110.18
                             f16(cublas): ['1.36816406', '-13.765625'], time:5.018281ms, swizzle: NOOP, TFLOPS: 109.55
                                  f16_th: ['1.39550781', '-13.898437'], time:4.905271ms, swizzle: NOOP, TFLOPS: 112.07(+1.56%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=8192
                     f16x8pack(t8x8+bcf): ['-27.078125', '-48.875   '], time:23.99902ms, swizzle: NOOP, TFLOPS: 45.81 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-27.078125', '-48.875   '], time:23.68755ms, swizzle: NOOP, TFLOPS: 46.42 (+1.31%)
                f16x8pack(t8x8+k16+dbuf): ['-27.078125', '-48.875   '], time:23.11522ms, swizzle: NOOP, TFLOPS: 47.57 (+2.48%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-27.125   ', '-48.5625  '], time:24.87859ms, swizzle: NOOP, TFLOPS: 44.20
                 f16wmma(mma4x2+warp2x4): ['-27.125   ', '-48.5625  '], time:14.21639ms, swizzle: NOOP, TFLOPS: 77.34 (+62.60%)
          f16wmma(mma2x4+warp2x4+stage3): ['-27.125   ', '-48.5625  '], time:10.91473ms, swizzle: NOOP, TFLOPS: 100.74(+30.25%)
          f16wmma(mma2x4+warp2x4+stage2): ['-27.125   ', '-48.5625  '], time:10.73844ms, swizzle: NOOP, TFLOPS: 102.39(+1.64%)
        f16wmma(mma2x4+...+stage3+dsmem): ['-27.125   ', '-48.5625  '], time:10.91294ms, swizzle: NOOP, TFLOPS: 100.75
        f16wmma(mma2x4+...+stage2+dsmem): ['-27.125   ', '-48.5625  '], time:10.73040ms, swizzle: NOOP, TFLOPS: 102.47(+0.07%)
      f16wmma(mma2x4+...+stage3+swizzle): ['-27.125   ', '-48.5625  '], time:10.05156ms, swizzle: 2048, TFLOPS: 109.39(+6.75%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-27.125   ', '-48.5625  '], time:9.907770ms, swizzle: 2048, TFLOPS: 110.97(+1.45%)
       f16wmma(...+stage3+dsmem+swizzle): ['-27.125   ', '-48.5625  '], time:10.17735ms, swizzle: 2048, TFLOPS: 108.04
       f16wmma(...+stage2+dsmem+swizzle): ['-27.125   ', '-48.5625  '], time:9.921789ms, swizzle: 2048, TFLOPS: 110.82
                             f16(cublas): ['-27.125   ', '-48.5625  '], time:10.46237ms, swizzle: NOOP, TFLOPS: 105.09
                                  f16_th: ['-27.203125', '-48.90625 '], time:9.708380ms, swizzle: NOOP, TFLOPS: 113.25(+2.05%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=2048
                     f16x8pack(t8x8+bcf): ['-54.125   ', '11.21875  '], time:10.90126ms, swizzle: NOOP, TFLOPS: 50.43 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-54.125   ', '11.21875  '], time:10.93275ms, swizzle: NOOP, TFLOPS: 50.29
                f16x8pack(t8x8+k16+dbuf): ['-54.125   ', '11.21875  '], time:10.48004ms, swizzle: NOOP, TFLOPS: 52.46 (+4.02%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-53.9375  ', '11.15625  '], time:11.76428ms, swizzle: NOOP, TFLOPS: 46.73
                 f16wmma(mma4x2+warp2x4): ['-53.9375  ', '11.15625  '], time:6.931257ms, swizzle: NOOP, TFLOPS: 79.32 (+51.20%)
          f16wmma(mma2x4+warp2x4+stage3): ['-53.9375  ', '11.15625  '], time:5.045485ms, swizzle: NOOP, TFLOPS: 108.96(+37.38%)
          f16wmma(mma2x4+warp2x4+stage2): ['-53.9375  ', '11.15625  '], time:5.075860ms, swizzle: NOOP, TFLOPS: 108.31
        f16wmma(mma2x4+...+stage3+dsmem): ['-53.9375  ', '11.15625  '], time:5.089354ms, swizzle: NOOP, TFLOPS: 108.02
        f16wmma(mma2x4+...+stage2+dsmem): ['-53.9375  ', '11.15625  '], time:5.070042ms, swizzle: NOOP, TFLOPS: 108.43
      f16wmma(mma2x4+...+stage3+swizzle): ['-53.9375  ', '11.15625  '], time:5.074930ms, swizzle: 4096, TFLOPS: 108.33
      f16wmma(mma2x4+...+stage2+swizzle): ['-53.9375  ', '11.15625  '], time:5.020570ms, swizzle: 4096, TFLOPS: 109.50(+0.50%)
       f16wmma(...+stage3+dsmem+swizzle): ['-53.9375  ', '11.15625  '], time:5.104279ms, swizzle: 4096, TFLOPS: 107.70
       f16wmma(...+stage2+dsmem+swizzle): ['-53.9375  ', '11.15625  '], time:5.056905ms, swizzle: 4096, TFLOPS: 108.71
                             f16(cublas): ['-53.9375  ', '11.15625  '], time:5.028772ms, swizzle: NOOP, TFLOPS: 109.32
                                  f16_th: ['-53.96875 ', '11.1171875'], time:4.797482ms, swizzle: NOOP, TFLOPS: 114.59(+4.65%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=4096
                     f16x8pack(t8x8+bcf): ['0.89404297', '-13.5625  '], time:24.05524ms, swizzle: NOOP, TFLOPS: 45.71 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['0.89404297', '-13.5625  '], time:23.88441ms, swizzle: NOOP, TFLOPS: 46.03 (+0.72%)
                f16x8pack(t8x8+k16+dbuf): ['0.89404297', '-13.5625  '], time:23.25210ms, swizzle: NOOP, TFLOPS: 47.29 (+2.72%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['1.36816406', '-13.765625'], time:31.20961ms, swizzle: NOOP, TFLOPS: 35.23
                 f16wmma(mma4x2+warp2x4): ['1.36816406', '-13.765625'], time:14.21947ms, swizzle: NOOP, TFLOPS: 77.32 (+63.52%)
          f16wmma(mma2x4+warp2x4+stage3): ['1.36816406', '-13.765625'], time:11.13727ms, swizzle: NOOP, TFLOPS: 98.72 (+27.67%)
          f16wmma(mma2x4+warp2x4+stage2): ['1.36816406', '-13.765625'], time:10.95814ms, swizzle: NOOP, TFLOPS: 100.34(+1.63%)
        f16wmma(mma2x4+...+stage3+dsmem): ['1.36816406', '-13.765625'], time:11.15133ms, swizzle: NOOP, TFLOPS: 98.60
        f16wmma(mma2x4+...+stage2+dsmem): ['1.36816406', '-13.765625'], time:10.84332ms, swizzle: NOOP, TFLOPS: 101.40(+1.06%)
      f16wmma(mma2x4+...+stage3+swizzle): ['1.36816406', '-13.765625'], time:10.05570ms, swizzle: 4096, TFLOPS: 109.34(+7.83%)
      f16wmma(mma2x4+...+stage2+swizzle): ['1.36816406', '-13.765625'], time:9.990954ms, swizzle: 4096, TFLOPS: 110.05(+0.65%)
       f16wmma(...+stage3+dsmem+swizzle): ['1.36816406', '-13.765625'], time:10.14525ms, swizzle: 4096, TFLOPS: 108.38
       f16wmma(...+stage2+dsmem+swizzle): ['1.36816406', '-13.765625'], time:9.966373ms, swizzle: 4096, TFLOPS: 110.32(+0.25%)
                             f16(cublas): ['1.36816406', '-13.765625'], time:9.750556ms, swizzle: NOOP, TFLOPS: 112.76(+2.21%)
                                  f16_th: ['1.39550781', '-13.898437'], time:9.574890ms, swizzle: NOOP, TFLOPS: 114.83(+1.83%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=8192
                     f16x8pack(t8x8+bcf): ['-27.078125', '-48.875   '], time:49.38902ms, swizzle: NOOP, TFLOPS: 44.52 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-27.078125', '-48.875   '], time:49.77235ms, swizzle: NOOP, TFLOPS: 44.18
                f16x8pack(t8x8+k16+dbuf): ['-27.078125', '-48.875   '], time:49.64823ms, swizzle: NOOP, TFLOPS: 44.29
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-27.125   ', '-48.5625  '], time:61.82026ms, swizzle: NOOP, TFLOPS: 35.57
                 f16wmma(mma4x2+warp2x4): ['-27.125   ', '-48.5625  '], time:28.17454ms, swizzle: NOOP, TFLOPS: 78.05 (+75.30%)
          f16wmma(mma2x4+warp2x4+stage3): ['-27.125   ', '-48.5625  '], time:24.88780ms, swizzle: NOOP, TFLOPS: 88.36 (+13.21%)
          f16wmma(mma2x4+warp2x4+stage2): ['-27.125   ', '-48.5625  '], time:24.93255ms, swizzle: NOOP, TFLOPS: 88.20
        f16wmma(mma2x4+...+stage3+dsmem): ['-27.125   ', '-48.5625  '], time:24.93562ms, swizzle: NOOP, TFLOPS: 88.19
        f16wmma(mma2x4+...+stage2+dsmem): ['-27.125   ', '-48.5625  '], time:24.90916ms, swizzle: NOOP, TFLOPS: 88.28
      f16wmma(mma2x4+...+stage3+swizzle): ['-27.125   ', '-48.5625  '], time:20.19155ms, swizzle: 4096, TFLOPS: 108.91(+23.26%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-27.125   ', '-48.5625  '], time:19.95363ms, swizzle: 4096, TFLOPS: 110.21(+1.19%)
       f16wmma(...+stage3+dsmem+swizzle): ['-27.125   ', '-48.5625  '], time:20.31297ms, swizzle: 4096, TFLOPS: 108.26
       f16wmma(...+stage2+dsmem+swizzle): ['-27.125   ', '-48.5625  '], time:19.94469ms, swizzle: 4096, TFLOPS: 110.26(+0.04%)
                             f16(cublas): ['-27.125   ', '-48.5625  '], time:19.17912ms, swizzle: NOOP, TFLOPS: 114.66(+3.99%)
                                  f16_th: ['-27.203125', '-48.90625 '], time:19.39091ms, swizzle: NOOP, TFLOPS: 113.40
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=2048
                     f16x8pack(t8x8+bcf): ['-54.125   ', '11.21875  '], time:5.281758ms, swizzle: NOOP, TFLOPS: 52.04 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-54.125   ', '11.21875  '], time:5.385351ms, swizzle: NOOP, TFLOPS: 51.04
                f16x8pack(t8x8+k16+dbuf): ['-54.125   ', '11.21875  '], time:5.172514ms, swizzle: NOOP, TFLOPS: 53.14 (+2.11%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-53.9375  ', '11.15625  '], time:5.929684ms, swizzle: NOOP, TFLOPS: 46.36
                 f16wmma(mma4x2+warp2x4): ['-53.9375  ', '11.15625  '], time:3.570580ms, swizzle: NOOP, TFLOPS: 76.98 (+44.86%)
          f16wmma(mma2x4+warp2x4+stage3): ['-53.9375  ', '11.15625  '], time:2.553486ms, swizzle: NOOP, TFLOPS: 107.65(+39.83%)
          f16wmma(mma2x4+warp2x4+stage2): ['-53.9375  ', '11.15625  '], time:2.567291ms, swizzle: NOOP, TFLOPS: 107.07
        f16wmma(mma2x4+...+stage3+dsmem): ['-53.9375  ', '11.15625  '], time:2.561354ms, swizzle: NOOP, TFLOPS: 107.32
        f16wmma(mma2x4+...+stage2+dsmem): ['-53.9375  ', '11.15625  '], time:2.554297ms, swizzle: NOOP, TFLOPS: 107.61
      f16wmma(mma2x4+...+stage3+swizzle): ['-53.9375  ', '11.15625  '], time:2.539038ms, swizzle: 1024, TFLOPS: 108.26(+0.57%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-53.9375  ', '11.15625  '], time:2.532052ms, swizzle: 1024, TFLOPS: 108.56(+0.28%)
       f16wmma(...+stage3+dsmem+swizzle): ['-53.9375  ', '11.15625  '], time:2.573370ms, swizzle: 1024, TFLOPS: 106.82
       f16wmma(...+stage2+dsmem+swizzle): ['-53.9375  ', '11.15625  '], time:2.536416ms, swizzle: 1024, TFLOPS: 108.37
                             f16(cublas): ['-53.9375  ', '11.15625  '], time:2.648115ms, swizzle: NOOP, TFLOPS: 103.80
                                  f16_th: ['-53.96875 ', '11.1171875'], time:2.407217ms, swizzle: NOOP, TFLOPS: 114.19(+5.19%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=4096
                     f16x8pack(t8x8+bcf): ['0.89404297', '-13.5625  '], time:11.07091ms, swizzle: NOOP, TFLOPS: 49.66 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['0.89404297', '-13.5625  '], time:11.05723ms, swizzle: NOOP, TFLOPS: 49.72 (+0.12%)
                f16x8pack(t8x8+k16+dbuf): ['0.89404297', '-13.5625  '], time:10.49160ms, swizzle: NOOP, TFLOPS: 52.40 (+5.39%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['1.36816406', '-13.765625'], time:11.71293ms, swizzle: NOOP, TFLOPS: 46.94
                 f16wmma(mma4x2+warp2x4): ['1.36816406', '-13.765625'], time:7.009673ms, swizzle: NOOP, TFLOPS: 78.43 (+49.67%)
          f16wmma(mma2x4+warp2x4+stage3): ['1.36816406', '-13.765625'], time:4.973363ms, swizzle: NOOP, TFLOPS: 110.54(+40.94%)
          f16wmma(mma2x4+warp2x4+stage2): ['1.36816406', '-13.765625'], time:5.029225ms, swizzle: NOOP, TFLOPS: 109.31
        f16wmma(mma2x4+...+stage3+dsmem): ['1.36816406', '-13.765625'], time:5.043435ms, swizzle: NOOP, TFLOPS: 109.00
        f16wmma(mma2x4+...+stage2+dsmem): ['1.36816406', '-13.765625'], time:5.023694ms, swizzle: NOOP, TFLOPS: 109.43
      f16wmma(mma2x4+...+stage3+swizzle): ['1.36816406', '-13.765625'], time:5.074596ms, swizzle: 1024, TFLOPS: 108.33
      f16wmma(mma2x4+...+stage2+swizzle): ['1.36816406', '-13.765625'], time:5.023050ms, swizzle: 1024, TFLOPS: 109.45
       f16wmma(...+stage3+dsmem+swizzle): ['1.36816406', '-13.765625'], time:5.102992ms, swizzle: 1024, TFLOPS: 107.73
       f16wmma(...+stage2+dsmem+swizzle): ['1.36816406', '-13.765625'], time:5.042982ms, swizzle: 1024, TFLOPS: 109.01
                             f16(cublas): ['1.36816406', '-13.765625'], time:4.966020ms, swizzle: NOOP, TFLOPS: 110.70(+0.15%)
                                  f16_th: ['1.39550781', '-13.898437'], time:4.902386ms, swizzle: NOOP, TFLOPS: 112.14(+1.30%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=8192
                     f16x8pack(t8x8+bcf): ['-27.078125', '-48.875   '], time:22.22590ms, swizzle: NOOP, TFLOPS: 49.47 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-27.078125', '-48.875   '], time:22.03605ms, swizzle: NOOP, TFLOPS: 49.90 (+0.86%)
                f16x8pack(t8x8+k16+dbuf): ['-27.078125', '-48.875   '], time:21.20213ms, swizzle: NOOP, TFLOPS: 51.86 (+3.93%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-27.125   ', '-48.5625  '], time:23.33390ms, swizzle: NOOP, TFLOPS: 47.12
                 f16wmma(mma4x2+warp2x4): ['-27.125   ', '-48.5625  '], time:13.81051ms, swizzle: NOOP, TFLOPS: 79.61 (+53.52%)
          f16wmma(mma2x4+warp2x4+stage3): ['-27.125   ', '-48.5625  '], time:9.887790ms, swizzle: NOOP, TFLOPS: 111.20(+39.67%)
          f16wmma(mma2x4+warp2x4+stage2): ['-27.125   ', '-48.5625  '], time:9.963655ms, swizzle: NOOP, TFLOPS: 110.35
        f16wmma(mma2x4+...+stage3+dsmem): ['-27.125   ', '-48.5625  '], time:10.02178ms, swizzle: NOOP, TFLOPS: 109.71
        f16wmma(mma2x4+...+stage2+dsmem): ['-27.125   ', '-48.5625  '], time:9.944844ms, swizzle: NOOP, TFLOPS: 110.56
      f16wmma(mma2x4+...+stage3+swizzle): ['-27.125   ', '-48.5625  '], time:10.11433ms, swizzle: 1024, TFLOPS: 108.71
      f16wmma(mma2x4+...+stage2+swizzle): ['-27.125   ', '-48.5625  '], time:9.967827ms, swizzle: 1024, TFLOPS: 110.31
       f16wmma(...+stage3+dsmem+swizzle): ['-27.125   ', '-48.5625  '], time:10.29257ms, swizzle: 1024, TFLOPS: 106.83
       f16wmma(...+stage2+dsmem+swizzle): ['-27.125   ', '-48.5625  '], time:10.03665ms, swizzle: 1024, TFLOPS: 109.55
                             f16(cublas): ['-27.125   ', '-48.5625  '], time:9.737825ms, swizzle: NOOP, TFLOPS: 112.91(+1.54%)
                                  f16_th: ['-27.1875  ', '-48.90625 '], time:9.758663ms, swizzle: NOOP, TFLOPS: 112.67
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=2048
                     f16x8pack(t8x8+bcf): ['-54.125   ', '11.21875  '], time:10.92875ms, swizzle: NOOP, TFLOPS: 50.30 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-54.125   ', '11.21875  '], time:10.94396ms, swizzle: NOOP, TFLOPS: 50.23
                f16x8pack(t8x8+k16+dbuf): ['-54.125   ', '11.21875  '], time:10.62042ms, swizzle: NOOP, TFLOPS: 51.76 (+2.90%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-53.9375  ', '11.15625  '], time:11.76416ms, swizzle: NOOP, TFLOPS: 46.73
                 f16wmma(mma4x2+warp2x4): ['-53.9375  ', '11.15625  '], time:6.952357ms, swizzle: NOOP, TFLOPS: 79.07 (+52.76%)
          f16wmma(mma2x4+warp2x4+stage3): ['-53.9375  ', '11.15625  '], time:5.056333ms, swizzle: NOOP, TFLOPS: 108.73(+37.50%)
          f16wmma(mma2x4+warp2x4+stage2): ['-53.9375  ', '11.15625  '], time:5.079627ms, swizzle: NOOP, TFLOPS: 108.23
        f16wmma(mma2x4+...+stage3+dsmem): ['-53.9375  ', '11.15625  '], time:5.104613ms, swizzle: NOOP, TFLOPS: 107.70
        f16wmma(mma2x4+...+stage2+dsmem): ['-53.9375  ', '11.15625  '], time:5.085945ms, swizzle: NOOP, TFLOPS: 108.09
      f16wmma(mma2x4+...+stage3+swizzle): ['-53.9375  ', '11.15625  '], time:5.042982ms, swizzle: 2048, TFLOPS: 109.01(+0.26%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-53.9375  ', '11.15625  '], time:5.071973ms, swizzle: 2048, TFLOPS: 108.39
       f16wmma(...+stage3+dsmem+swizzle): ['-53.9375  ', '11.15625  '], time:5.127406ms, swizzle: 2048, TFLOPS: 107.22
       f16wmma(...+stage2+dsmem+swizzle): ['-53.9375  ', '11.15625  '], time:5.060100ms, swizzle: 2048, TFLOPS: 108.65
                             f16(cublas): ['-53.9375  ', '11.15625  '], time:5.015182ms, swizzle: NOOP, TFLOPS: 109.62(+0.55%)
                                  f16_th: ['-53.96875 ', '11.1171875'], time:4.786968ms, swizzle: NOOP, TFLOPS: 114.84(+4.77%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=4096
                     f16x8pack(t8x8+bcf): ['0.89404297', '-13.5625  '], time:22.00777ms, swizzle: NOOP, TFLOPS: 49.96 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['0.89404297', '-13.5625  '], time:22.02115ms, swizzle: NOOP, TFLOPS: 49.93
                f16x8pack(t8x8+k16+dbuf): ['0.89404297', '-13.5625  '], time:20.95642ms, swizzle: NOOP, TFLOPS: 52.47 (+5.02%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['1.36816406', '-13.765625'], time:23.35927ms, swizzle: NOOP, TFLOPS: 47.07
                 f16wmma(mma4x2+warp2x4): ['1.36816406', '-13.765625'], time:13.63372ms, swizzle: NOOP, TFLOPS: 80.65 (+53.71%)
          f16wmma(mma2x4+warp2x4+stage3): ['1.36816406', '-13.765625'], time:10.00251ms, swizzle: NOOP, TFLOPS: 109.92(+36.30%)
          f16wmma(mma2x4+warp2x4+stage2): ['1.36816406', '-13.765625'], time:10.00540ms, swizzle: NOOP, TFLOPS: 109.89
        f16wmma(mma2x4+...+stage3+dsmem): ['1.36816406', '-13.765625'], time:10.05990ms, swizzle: NOOP, TFLOPS: 109.30
        f16wmma(mma2x4+...+stage2+dsmem): ['1.36816406', '-13.765625'], time:9.990119ms, swizzle: NOOP, TFLOPS: 110.06(+0.12%)
      f16wmma(mma2x4+...+stage3+swizzle): ['1.36816406', '-13.765625'], time:10.16755ms, swizzle: 2048, TFLOPS: 108.14
      f16wmma(mma2x4+...+stage2+swizzle): ['1.36816406', '-13.765625'], time:10.07659ms, swizzle: 2048, TFLOPS: 109.12
       f16wmma(...+stage3+dsmem+swizzle): ['1.36816406', '-13.765625'], time:10.25078ms, swizzle: 2048, TFLOPS: 107.26
       f16wmma(...+stage2+dsmem+swizzle): ['1.36816406', '-13.765625'], time:10.07041ms, swizzle: 2048, TFLOPS: 109.18
                             f16(cublas): ['1.36816406', '-13.765625'], time:9.725880ms, swizzle: NOOP, TFLOPS: 113.05(+2.72%)
                                  f16_th: ['1.39550781', '-13.898437'], time:9.557795ms, swizzle: NOOP, TFLOPS: 115.04(+1.76%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=8192
                     f16x8pack(t8x8+bcf): ['-27.078125', '-48.875   '], time:48.27799ms, swizzle: NOOP, TFLOPS: 45.55 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-27.078125', '-48.875   '], time:48.28574ms, swizzle: NOOP, TFLOPS: 45.54
                f16x8pack(t8x8+k16+dbuf): ['-27.078125', '-48.875   '], time:46.86999ms, swizzle: NOOP, TFLOPS: 46.92 (+3.00%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-27.125   ', '-48.5625  '], time:49.85406ms, swizzle: NOOP, TFLOPS: 44.11
                 f16wmma(mma4x2+warp2x4): ['-27.125   ', '-48.5625  '], time:28.04584ms, swizzle: NOOP, TFLOPS: 78.41 (+67.12%)
          f16wmma(mma2x4+warp2x4+stage3): ['-27.125   ', '-48.5625  '], time:21.96478ms, swizzle: NOOP, TFLOPS: 100.12(+27.69%)
          f16wmma(mma2x4+warp2x4+stage2): ['-27.125   ', '-48.5625  '], time:21.81372ms, swizzle: NOOP, TFLOPS: 100.81(+0.69%)
        f16wmma(mma2x4+...+stage3+dsmem): ['-27.125   ', '-48.5625  '], time:22.26374ms, swizzle: NOOP, TFLOPS: 98.77
        f16wmma(mma2x4+...+stage2+dsmem): ['-27.125   ', '-48.5625  '], time:21.87619ms, swizzle: NOOP, TFLOPS: 100.52
      f16wmma(mma2x4+...+stage3+swizzle): ['-27.125   ', '-48.5625  '], time:20.18580ms, swizzle: 2048, TFLOPS: 108.94(+8.06%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-27.125   ', '-48.5625  '], time:19.93291ms, swizzle: 2048, TFLOPS: 110.32(+1.27%)
       f16wmma(...+stage3+dsmem+swizzle): ['-27.125   ', '-48.5625  '], time:20.31307ms, swizzle: 2048, TFLOPS: 108.26
       f16wmma(...+stage2+dsmem+swizzle): ['-27.125   ', '-48.5625  '], time:19.92838ms, swizzle: 2048, TFLOPS: 110.35(+0.02%)
                             f16(cublas): ['-27.125   ', '-48.5625  '], time:19.38226ms, swizzle: NOOP, TFLOPS: 113.46(+2.82%)
                                  f16_th: ['-27.203125', '-48.90625 '], time:19.28415ms, swizzle: NOOP, TFLOPS: 114.03(+0.51%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=2048
                     f16x8pack(t8x8+bcf): ['-54.125   ', '11.21875  '], time:22.19297ms, swizzle: NOOP, TFLOPS: 49.54 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-54.125   ', '11.21875  '], time:22.13892ms, swizzle: NOOP, TFLOPS: 49.66 (+0.24%)
                f16x8pack(t8x8+k16+dbuf): ['-54.125   ', '11.21875  '], time:21.11446ms, swizzle: NOOP, TFLOPS: 52.07 (+4.85%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-53.9375  ', '11.15625  '], time:23.49538ms, swizzle: NOOP, TFLOPS: 46.80
                 f16wmma(mma4x2+warp2x4): ['-53.9375  ', '11.15625  '], time:13.65721ms, swizzle: NOOP, TFLOPS: 80.51 (+54.60%)
          f16wmma(mma2x4+warp2x4+stage3): ['-53.9375  ', '11.15625  '], time:10.14587ms, swizzle: NOOP, TFLOPS: 108.37(+34.61%)
          f16wmma(mma2x4+warp2x4+stage2): ['-53.9375  ', '11.15625  '], time:10.18333ms, swizzle: NOOP, TFLOPS: 107.97
        f16wmma(mma2x4+...+stage3+dsmem): ['-53.9375  ', '11.15625  '], time:10.18404ms, swizzle: NOOP, TFLOPS: 107.96
        f16wmma(mma2x4+...+stage2+dsmem): ['-53.9375  ', '11.15625  '], time:10.21685ms, swizzle: NOOP, TFLOPS: 107.62
      f16wmma(mma2x4+...+stage3+swizzle): ['-53.9375  ', '11.15625  '], time:10.29250ms, swizzle: 4096, TFLOPS: 106.83
      f16wmma(mma2x4+...+stage2+swizzle): ['-53.9375  ', '11.15625  '], time:10.15782ms, swizzle: 4096, TFLOPS: 108.24
       f16wmma(...+stage3+dsmem+swizzle): ['-53.9375  ', '11.15625  '], time:10.33658ms, swizzle: 4096, TFLOPS: 106.37
       f16wmma(...+stage2+dsmem+swizzle): ['-53.9375  ', '11.15625  '], time:10.20550ms, swizzle: 4096, TFLOPS: 107.74
                             f16(cublas): ['-53.9375  ', '11.15625  '], time:9.746479ms, swizzle: NOOP, TFLOPS: 112.81(+4.10%)
                                  f16_th: ['-53.96875 ', '11.1171875'], time:9.699082ms, swizzle: NOOP, TFLOPS: 113.36(+0.49%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=4096
                     f16x8pack(t8x8+bcf): ['0.89404297', '-13.5625  '], time:48.05865ms, swizzle: NOOP, TFLOPS: 45.76 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['0.89404297', '-13.5625  '], time:48.78034ms, swizzle: NOOP, TFLOPS: 45.08
                f16x8pack(t8x8+k16+dbuf): ['0.89404297', '-13.5625  '], time:47.16243ms, swizzle: NOOP, TFLOPS: 46.63 (+1.90%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['1.36816406', '-13.765625'], time:62.86273ms, swizzle: NOOP, TFLOPS: 34.98
                 f16wmma(mma4x2+warp2x4): ['1.36816406', '-13.765625'], time:28.40418ms, swizzle: NOOP, TFLOPS: 77.42 (+66.04%)
          f16wmma(mma2x4+warp2x4+stage3): ['1.36816406', '-13.765625'], time:22.69105ms, swizzle: NOOP, TFLOPS: 96.91 (+25.18%)
          f16wmma(mma2x4+warp2x4+stage2): ['1.36816406', '-13.765625'], time:22.32215ms, swizzle: NOOP, TFLOPS: 98.51 (+1.65%)
        f16wmma(mma2x4+...+stage3+dsmem): ['1.36816406', '-13.765625'], time:22.77216ms, swizzle: NOOP, TFLOPS: 96.57
        f16wmma(mma2x4+...+stage2+dsmem): ['1.36816406', '-13.765625'], time:22.20034ms, swizzle: NOOP, TFLOPS: 99.05 (+0.55%)
      f16wmma(mma2x4+...+stage3+swizzle): ['1.36816406', '-13.765625'], time:20.20931ms, swizzle: 4096, TFLOPS: 108.81(+9.85%)
      f16wmma(mma2x4+...+stage2+swizzle): ['1.36816406', '-13.765625'], time:19.94318ms, swizzle: 4096, TFLOPS: 110.26(+1.33%)
       f16wmma(...+stage3+dsmem+swizzle): ['1.36816406', '-13.765625'], time:20.40047ms, swizzle: 4096, TFLOPS: 107.79
       f16wmma(...+stage2+dsmem+swizzle): ['1.36816406', '-13.765625'], time:19.95811ms, swizzle: 4096, TFLOPS: 110.18
                             f16(cublas): ['1.36816406', '-13.765625'], time:19.22338ms, swizzle: NOOP, TFLOPS: 114.39(+3.74%)
                                  f16_th: ['1.39550781', '-13.898437'], time:19.11091ms, swizzle: NOOP, TFLOPS: 115.07(+0.59%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=8192
                     f16x8pack(t8x8+bcf): ['-7.0742187', '-78.3125  '], time:103.2896ms, swizzle: NOOP, TFLOPS: 42.58 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-7.0742187', '-78.3125  '], time:103.9499ms, swizzle: NOOP, TFLOPS: 42.31
                f16x8pack(t8x8+k16+dbuf): ['-7.0742187', '-78.3125  '], time:102.3885ms, swizzle: NOOP, TFLOPS: 42.95 (+0.88%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-7.0546875', '-80.4375  '], time:124.1894ms, swizzle: NOOP, TFLOPS: 35.41
                 f16wmma(mma4x2+warp2x4): ['-7.0546875', '-80.4375  '], time:55.97629ms, swizzle: NOOP, TFLOPS: 78.57 (+82.91%)
          f16wmma(mma2x4+warp2x4+stage3): ['-7.0546875', '-80.4375  '], time:49.90615ms, swizzle: NOOP, TFLOPS: 88.13 (+12.16%)
          f16wmma(mma2x4+warp2x4+stage2): ['-7.0546875', '-80.4375  '], time:50.10385ms, swizzle: NOOP, TFLOPS: 87.78
        f16wmma(mma2x4+...+stage3+dsmem): ['-7.0546875', '-80.4375  '], time:49.86991ms, swizzle: NOOP, TFLOPS: 88.19 (+0.07%)
        f16wmma(mma2x4+...+stage2+dsmem): ['-7.0546875', '-80.4375  '], time:50.06759ms, swizzle: NOOP, TFLOPS: 87.84
      f16wmma(mma2x4+...+stage3+swizzle): ['-7.0546875', '-80.4375  '], time:40.56155ms, swizzle: 4096, TFLOPS: 108.43(+22.95%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-7.0546875', '-80.4375  '], time:39.84785ms, swizzle: 4096, TFLOPS: 110.37(+1.79%)
       f16wmma(...+stage3+dsmem+swizzle): ['-7.0546875', '-80.4375  '], time:40.76685ms, swizzle: 4096, TFLOPS: 107.88
       f16wmma(...+stage2+dsmem+swizzle): ['-7.0546875', '-80.4375  '], time:39.97714ms, swizzle: 4096, TFLOPS: 110.01
                             f16(cublas): ['-7.0546875', '-80.4375  '], time:38.90087ms, swizzle: NOOP, TFLOPS: 113.06(+2.43%)
                                  f16_th: ['-7.0742187', '-80.875   '], time:38.52620ms, swizzle: NOOP, TFLOPS: 114.16(+0.97%)
----------------------------------------------------------------------------------------------------------------------------------
```
