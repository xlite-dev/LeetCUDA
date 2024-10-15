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
- [X] hgemm_wmma_m16n16k16_naive(WMMA API, Tensor Cores) 
- [X] hgemm_wmma_m16n16k16_mma4x2(Tensor Cores, Tile MMA) 
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4(Tensor Cores, Tile MMA/Warp, pack) 
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_async(Tensor Cores, Tile MMA/Warp, Copy Async) 
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_async_offset(Tensor Cores, Tile MMA/Warp, Copy Async, Pad)
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(Tensor Cores, Tile MMA/Warp, Copy Async, Double Buffers, Pad)  
- [X] hgemm_wmma_m16n16k16_mma4x4_warp2x2x2_dbuf_async(Tensor Cores, Tile MMA/Warp, Copy Async, Double Buffers, Pad)  
- [X] hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async(Tensor Cores, Tile MMA/Warp, Copy Async, Double Buffers, Pad)
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_rbuf_async(Tensor Cores, Tile MMA/Warp, Copy Async, Double/Reg Buffers, Pad)
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_stage2/3/4(Tensor Cores, Tile MMA/Warp, Copy Async, Stage, Pad, Thread block swizzle) 
- [X] PyTorch bindings

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

- RTX 3080

```bash  
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=256
                                           out_f16: ['37.3125     ', '6.1015625   ', '-11.6484375 '], time:6.849790ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['37.25       ', '6.078125    ', '-11.640625  '], time:0.268483ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['37.25       ', '6.078125    ', '-11.640625  '], time:0.245142ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['37.25       ', '6.078125    ', '-11.640625  '], time:0.239062ms
                                        out_f16_th: ['37.25       ', '6.0703125   ', '-11.640625  '], time:0.302052ms
                                   out_f16(cublas): ['37.25       ', '6.078125    ', '-11.640625  '], time:0.394297ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=512
                                           out_f16: ['2.56640625  ', '-3.1953125  ', '56.59375    '], time:16.299248ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['2.5625      ', '-3.16601562 ', '56.625      '], time:0.394201ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['2.5625      ', '-3.16601562 ', '56.625      '], time:0.384831ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['2.5625      ', '-3.16601562 ', '56.625      '], time:0.367522ms
                                        out_f16_th: ['2.56054688  ', '-3.171875   ', '56.53125    '], time:0.519037ms
                                   out_f16(cublas): ['2.5625      ', '-3.16601562 ', '56.625      '], time:0.403666ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=1024
                                           out_f16: ['39.0625     ', '2.04101562  ', '-8.3046875  '], time:34.225392ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['39.21875    ', '2.06640625  ', '-8.3203125  '], time:0.716949ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['39.21875    ', '2.06640625  ', '-8.3203125  '], time:0.722528ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['39.21875    ', '2.06640625  ', '-8.3203125  '], time:0.645280ms
                                        out_f16_th: ['39.21875    ', '2.08398438  ', '-8.328125   '], time:0.852585ms
                                   out_f16(cublas): ['39.21875    ', '2.06640625  ', '-8.3203125  '], time:0.554204ms
------------------------------------------------------------------------------------------------------------------------
```

- NVIDIA L20  
```bash
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=512
                                   f16(naive): ['-6.96484375 ', '4.3203125   ', '-3.99804688 '], time:4.755163ms, TFLOPS: 3.61
                          f16x8pack(t8x8+bcf): ['-6.96484375 ', '4.3203125   ', '-3.99804688 '], time:0.362110ms, TFLOPS: 47.44
                     f16x8pack(t8x8+bcf+dbuf): ['-6.96484375 ', '4.3203125   ', '-3.99804688 '], time:0.356078ms, TFLOPS: 48.25
                     f16x8pack(t8x8+k16+dbuf): ['-6.96484375 ', '4.3203125   ', '-3.99804688 '], time:0.337338ms, TFLOPS: 50.93
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['-6.94921875 ', '4.44140625  ', '-4.0        '], time:0.742650ms, TFLOPS: 23.13
                              f16wmma(mma4x2): ['-6.94921875 ', '4.44140625  ', '-4.0        '], time:0.381040ms, TFLOPS: 45.09
                      f16wmma(mma4x2+warp2x4): ['-6.94921875 ', '4.44140625  ', '-4.0        '], time:0.267457ms, TFLOPS: 64.23
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['-6.94921875 ', '4.44140625  ', '-4.0        '], time:0.199079ms, TFLOPS: 86.30
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-6.94921875 ', '4.44140625  ', '-4.0        '], time:0.199556ms, TFLOPS: 86.09
               f16wmma(mma2x4+warp2x4+stage3): ['-6.94921875 ', '4.44140625  ', '-4.0        '], time:0.191736ms, TFLOPS: 89.60
               f16wmma(mma2x4+warp2x4+stage2): ['-6.94921875 ', '4.44140625  ', '-4.0        '], time:0.194001ms, TFLOPS: 88.56
            f16wmma(warp2x4+...+stage3+dsmem): ['-6.94921875 ', '4.44140625  ', '-4.0        '], time:0.193929ms, TFLOPS: 88.59
            f16wmma(warp2x4+...+stage2+dsmem): ['-6.94921875 ', '4.44140625  ', '-4.0        '], time:0.193309ms, TFLOPS: 88.87
          f16wmma(warp2x4+...+stage3+swizzle): ['-6.94921875 ', '4.44140625  ', '-4.0        '], time:0.193476ms, TFLOPS: 88.80
          f16wmma(warp2x4+...+stage2+swizzle): ['-6.94921875 ', '4.44140625  ', '-4.0        '], time:0.190114ms, TFLOPS: 90.37
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['-6.94921875 ', '4.44140625  ', '-4.0        '], time:0.194001ms, TFLOPS: 88.56
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['-6.94921875 ', '4.44140625  ', '-4.0        '], time:0.190067ms, TFLOPS: 90.39
                                  f16(cublas): ['-6.94921875 ', '4.44140625  ', '-4.0        '], time:0.203537ms, TFLOPS: 84.41
                                       f16_th: ['-6.953125   ', '4.4375      ', '-4.00390625 '], time:0.169372ms, TFLOPS: 101.43
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=1024
                                   f16(naive): ['19.359375   ', '-25.390625  ', '-44.6875    '], time:9.460616ms, TFLOPS: 3.63
                          f16x8pack(t8x8+bcf): ['19.359375   ', '-25.390625  ', '-44.6875    '], time:0.716400ms, TFLOPS: 47.96
                     f16x8pack(t8x8+bcf+dbuf): ['19.359375   ', '-25.390625  ', '-44.6875    '], time:0.705480ms, TFLOPS: 48.70
                     f16x8pack(t8x8+k16+dbuf): ['19.359375   ', '-25.390625  ', '-44.6875    '], time:0.668263ms, TFLOPS: 51.42
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['19.125      ', '-25.1875    ', '-44.25      '], time:1.474428ms, TFLOPS: 23.30
                              f16wmma(mma4x2): ['19.125      ', '-25.1875    ', '-44.25      '], time:0.746893ms, TFLOPS: 46.00
                      f16wmma(mma4x2+warp2x4): ['19.125      ', '-25.1875    ', '-44.25      '], time:0.490164ms, TFLOPS: 70.10
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['19.125      ', '-25.1875    ', '-44.25      '], time:0.363874ms, TFLOPS: 94.43
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['19.125      ', '-25.1875    ', '-44.25      '], time:0.366926ms, TFLOPS: 93.64
               f16wmma(mma2x4+warp2x4+stage3): ['19.125      ', '-25.1875    ', '-44.25      '], time:0.357699ms, TFLOPS: 96.06
               f16wmma(mma2x4+warp2x4+stage2): ['19.125      ', '-25.1875    ', '-44.25      '], time:0.359964ms, TFLOPS: 95.45
            f16wmma(warp2x4+...+stage3+dsmem): ['19.125      ', '-25.1875    ', '-44.25      '], time:0.360035ms, TFLOPS: 95.43
            f16wmma(warp2x4+...+stage2+dsmem): ['19.125      ', '-25.1875    ', '-44.25      '], time:0.359702ms, TFLOPS: 95.52
          f16wmma(warp2x4+...+stage3+swizzle): ['19.125      ', '-25.1875    ', '-44.25      '], time:0.357770ms, TFLOPS: 96.04
          f16wmma(warp2x4+...+stage2+swizzle): ['19.125      ', '-25.1875    ', '-44.25      '], time:0.353789ms, TFLOPS: 97.12
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['19.125      ', '-25.1875    ', '-44.25      '], time:0.357913ms, TFLOPS: 96.00
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['19.125      ', '-25.1875    ', '-44.25      '], time:0.353121ms, TFLOPS: 97.30
                                  f16(cublas): ['19.125      ', '-25.1875    ', '-44.25      '], time:0.345182ms, TFLOPS: 99.54
                                       f16_th: ['19.171875   ', '-25.21875   ', '-44.375     '], time:0.335860ms, TFLOPS: 102.30
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=2048
                                   f16(naive): ['-11.5625    ', '23.6875     ', '-7.58203125 '], time:18.86038ms, TFLOPS: 3.64
                          f16x8pack(t8x8+bcf): ['-11.5625    ', '23.6875     ', '-7.58203125 '], time:1.421093ms, TFLOPS: 48.36
                     f16x8pack(t8x8+bcf+dbuf): ['-11.5625    ', '23.6875     ', '-7.58203125 '], time:1.399636ms, TFLOPS: 49.10
                     f16x8pack(t8x8+k16+dbuf): ['-11.5625    ', '23.6875     ', '-7.58203125 '], time:1.323437ms, TFLOPS: 51.92
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['-11.5546875 ', '23.859375   ', '-7.76171875 '], time:2.928638ms, TFLOPS: 23.46
                              f16wmma(mma4x2): ['-11.5546875 ', '23.859375   ', '-7.76171875 '], time:1.481294ms, TFLOPS: 46.39
                      f16wmma(mma4x2+warp2x4): ['-11.5546875 ', '23.859375   ', '-7.76171875 '], time:0.932502ms, TFLOPS: 73.69
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['-11.5546875 ', '23.859375   ', '-7.76171875 '], time:0.693392ms, TFLOPS: 99.11
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-11.5546875 ', '23.859375   ', '-7.76171875 '], time:0.699138ms, TFLOPS: 98.29
               f16wmma(mma2x4+warp2x4+stage3): ['-11.5546875 ', '23.859375   ', '-7.76171875 '], time:0.685977ms, TFLOPS: 100.18
               f16wmma(mma2x4+warp2x4+stage2): ['-11.5546875 ', '23.859375   ', '-7.76171875 '], time:0.688457ms, TFLOPS: 99.82
            f16wmma(warp2x4+...+stage3+dsmem): ['-11.5546875 ', '23.859375   ', '-7.76171875 '], time:0.690627ms, TFLOPS: 99.50
            f16wmma(warp2x4+...+stage2+dsmem): ['-11.5546875 ', '23.859375   ', '-7.76171875 '], time:0.687241ms, TFLOPS: 99.99
          f16wmma(warp2x4+...+stage3+swizzle): ['-11.5546875 ', '23.859375   ', '-7.76171875 '], time:0.684452ms, TFLOPS: 100.40
          f16wmma(warp2x4+...+stage2+swizzle): ['-11.5546875 ', '23.859375   ', '-7.76171875 '], time:0.676655ms, TFLOPS: 101.56
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['-11.5546875 ', '23.859375   ', '-7.76171875 '], time:0.684618ms, TFLOPS: 100.38
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['-11.5546875 ', '23.859375   ', '-7.76171875 '], time:0.677490ms, TFLOPS: 101.43
                                  f16(cublas): ['-11.5546875 ', '23.859375   ', '-7.76171875 '], time:0.666689ms, TFLOPS: 103.08
                                       f16_th: ['-11.546875  ', '23.90625    ', '-7.81640625 '], time:0.647282ms, TFLOPS: 106.17
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=512
                                   f16(naive): ['8.046875    ', '-19.015625  ', '7.9765625   '], time:9.491753ms, TFLOPS: 3.62
                          f16x8pack(t8x8+bcf): ['8.046875    ', '-19.015625  ', '7.9765625   '], time:0.681471ms, TFLOPS: 50.42
                     f16x8pack(t8x8+bcf+dbuf): ['8.046875    ', '-19.015625  ', '7.9765625   '], time:0.673961ms, TFLOPS: 50.98
                     f16x8pack(t8x8+k16+dbuf): ['8.046875    ', '-19.015625  ', '7.9765625   '], time:0.639867ms, TFLOPS: 53.70
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['8.140625    ', '-19.09375   ', '8.0625      '], time:1.484227ms, TFLOPS: 23.15
                              f16wmma(mma4x2): ['8.140625    ', '-19.09375   ', '8.0625      '], time:0.755119ms, TFLOPS: 45.50
                      f16wmma(mma4x2+warp2x4): ['8.140625    ', '-19.09375   ', '8.0625      '], time:0.517153ms, TFLOPS: 66.44
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['8.140625    ', '-19.09375   ', '8.0625      '], time:0.374794ms, TFLOPS: 91.68
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['8.140625    ', '-19.09375   ', '8.0625      '], time:0.376391ms, TFLOPS: 91.29
               f16wmma(mma2x4+warp2x4+stage3): ['8.140625    ', '-19.09375   ', '8.0625      '], time:0.360202ms, TFLOPS: 95.39
               f16wmma(mma2x4+warp2x4+stage2): ['8.140625    ', '-19.09375   ', '8.0625      '], time:0.364947ms, TFLOPS: 94.15
            f16wmma(warp2x4+...+stage3+dsmem): ['8.140625    ', '-19.09375   ', '8.0625      '], time:0.363087ms, TFLOPS: 94.63
            f16wmma(warp2x4+...+stage2+dsmem): ['8.140625    ', '-19.09375   ', '8.0625      '], time:0.364112ms, TFLOPS: 94.37
          f16wmma(warp2x4+...+stage3+swizzle): ['8.140625    ', '-19.09375   ', '8.0625      '], time:0.363278ms, TFLOPS: 94.58
          f16wmma(warp2x4+...+stage2+swizzle): ['8.140625    ', '-19.09375   ', '8.0625      '], time:0.355386ms, TFLOPS: 96.68
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['8.140625    ', '-19.09375   ', '8.0625      '], time:0.361871ms, TFLOPS: 94.95
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['8.140625    ', '-19.09375   ', '8.0625      '], time:0.355887ms, TFLOPS: 96.55
                                  f16(cublas): ['8.140625    ', '-19.09375   ', '8.0625      '], time:0.340175ms, TFLOPS: 101.01
                                       f16_th: ['8.1015625   ', '-19.078125  ', '8.0703125   '], time:0.318384ms, TFLOPS: 107.92
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=1024
                                   f16(naive): ['-56.09375   ', '35.0625     ', '-17.0625    '], time:18.87712ms, TFLOPS: 3.64
                          f16x8pack(t8x8+bcf): ['-56.09375   ', '35.0625     ', '-17.0625    '], time:1.353096ms, TFLOPS: 50.79
                     f16x8pack(t8x8+bcf+dbuf): ['-56.09375   ', '35.0625     ', '-17.0625    '], time:1.341223ms, TFLOPS: 51.24
                     f16x8pack(t8x8+k16+dbuf): ['-56.09375   ', '35.0625     ', '-17.0625    '], time:1.271581ms, TFLOPS: 54.04
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['-56.28125   ', '35.5        ', '-17.328125  '], time:2.935433ms, TFLOPS: 23.41
                              f16wmma(mma4x2): ['-56.28125   ', '35.5        ', '-17.328125  '], time:1.476883ms, TFLOPS: 46.53
                      f16wmma(mma4x2+warp2x4): ['-56.28125   ', '35.5        ', '-17.328125  '], time:0.953793ms, TFLOPS: 72.05
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['-56.28125   ', '35.5        ', '-17.328125  '], time:0.689148ms, TFLOPS: 99.72
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-56.28125   ', '35.5        ', '-17.328125  '], time:0.694704ms, TFLOPS: 98.92
               f16wmma(mma2x4+warp2x4+stage3): ['-56.28125   ', '35.5        ', '-17.328125  '], time:0.673317ms, TFLOPS: 102.06
               f16wmma(mma2x4+warp2x4+stage2): ['-56.28125   ', '35.5        ', '-17.328125  '], time:0.677633ms, TFLOPS: 101.41
            f16wmma(warp2x4+...+stage3+dsmem): ['-56.28125   ', '35.5        ', '-17.328125  '], time:0.675988ms, TFLOPS: 101.66
            f16wmma(warp2x4+...+stage2+dsmem): ['-56.28125   ', '35.5        ', '-17.328125  '], time:0.676703ms, TFLOPS: 101.55
          f16wmma(warp2x4+...+stage3+swizzle): ['-56.28125   ', '35.5        ', '-17.328125  '], time:0.673818ms, TFLOPS: 101.99
          f16wmma(warp2x4+...+stage2+swizzle): ['-56.28125   ', '35.5        ', '-17.328125  '], time:0.665903ms, TFLOPS: 103.20
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['-56.28125   ', '35.5        ', '-17.328125  '], time:0.674104ms, TFLOPS: 101.94
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['-56.28125   ', '35.5        ', '-17.328125  '], time:0.665497ms, TFLOPS: 103.26
                                  f16(cublas): ['-56.28125   ', '35.5        ', '-17.328125  '], time:0.636100ms, TFLOPS: 108.03
                                       f16_th: ['-56.1875    ', '35.59375    ', '-17.25      '], time:0.618982ms, TFLOPS: 111.02
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=2048
                                   f16(naive): ['9.609375    ', '-6.0546875  ', '79.4375     '], time:37.66117ms, TFLOPS: 3.65
                          f16x8pack(t8x8+bcf): ['9.609375    ', '-6.0546875  ', '79.4375     '], time:2.716755ms, TFLOPS: 50.59
                     f16x8pack(t8x8+bcf+dbuf): ['9.609375    ', '-6.0546875  ', '79.4375     '], time:2.660131ms, TFLOPS: 51.67
                     f16x8pack(t8x8+k16+dbuf): ['9.609375    ', '-6.0546875  ', '79.4375     '], time:2.519488ms, TFLOPS: 54.55
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['9.8515625   ', '-6.15234375 ', '79.0        '], time:5.810904ms, TFLOPS: 23.65
                              f16wmma(mma4x2): ['9.8515625   ', '-6.15234375 ', '79.0        '], time:2.943897ms, TFLOPS: 46.69
                      f16wmma(mma4x2+warp2x4): ['9.8515625   ', '-6.15234375 ', '79.0        '], time:1.843070ms, TFLOPS: 74.57
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['9.8515625   ', '-6.15234375 ', '79.0        '], time:1.323509ms, TFLOPS: 103.84
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['9.8515625   ', '-6.15234375 ', '79.0        '], time:1.335167ms, TFLOPS: 102.94
               f16wmma(mma2x4+warp2x4+stage3): ['9.8515625   ', '-6.15234375 ', '79.0        '], time:1.301765ms, TFLOPS: 105.58
               f16wmma(mma2x4+warp2x4+stage2): ['9.8515625   ', '-6.15234375 ', '79.0        '], time:1.307725ms, TFLOPS: 105.10
            f16wmma(warp2x4+...+stage3+dsmem): ['9.8515625   ', '-6.15234375 ', '79.0        '], time:1.309943ms, TFLOPS: 104.92
            f16wmma(warp2x4+...+stage2+dsmem): ['9.8515625   ', '-6.15234375 ', '79.0        '], time:1.306605ms, TFLOPS: 105.19
          f16wmma(warp2x4+...+stage3+swizzle): ['9.8515625   ', '-6.15234375 ', '79.0        '], time:1.299619ms, TFLOPS: 105.75
          f16wmma(warp2x4+...+stage2+swizzle): ['9.8515625   ', '-6.15234375 ', '79.0        '], time:1.288533ms, TFLOPS: 106.66
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['9.8515625   ', '-6.15234375 ', '79.0        '], time:1.299858ms, TFLOPS: 105.73
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['9.8515625   ', '-6.15234375 ', '79.0        '], time:1.288485ms, TFLOPS: 106.67
                                  f16(cublas): ['9.8515625   ', '-6.15234375 ', '79.0        '], time:1.238131ms, TFLOPS: 111.01
                                       f16_th: ['9.828125    ', '-6.1484375  ', '79.1875     '], time:1.288795ms, TFLOPS: 106.64
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=512
                                   f16(naive): ['-20.03125   ', '-0.70263672 ', '-10.046875  '], time:18.96362ms, TFLOPS: 3.62
                          f16x8pack(t8x8+bcf): ['-20.03125   ', '-0.70263672 ', '-10.046875  '], time:1.335716ms, TFLOPS: 51.45
                     f16x8pack(t8x8+bcf+dbuf): ['-20.03125   ', '-0.70263672 ', '-10.046875  '], time:1.314711ms, TFLOPS: 52.27
                     f16x8pack(t8x8+k16+dbuf): ['-20.03125   ', '-0.70263672 ', '-10.046875  '], time:1.244783ms, TFLOPS: 55.21
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['-20.09375   ', '-0.77539062 ', '-10.0546875 '], time:2.969694ms, TFLOPS: 23.14
                              f16wmma(mma4x2): ['-20.09375   ', '-0.77539062 ', '-10.0546875 '], time:1.496338ms, TFLOPS: 45.93
                      f16wmma(mma4x2+warp2x4): ['-20.09375   ', '-0.77539062 ', '-10.0546875 '], time:0.979781ms, TFLOPS: 70.14
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['-20.09375   ', '-0.77539062 ', '-10.0546875 '], time:0.722885ms, TFLOPS: 95.06
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-20.09375   ', '-0.77539062 ', '-10.0546875 '], time:0.729274ms, TFLOPS: 94.23
               f16wmma(mma2x4+warp2x4+stage3): ['-20.09375   ', '-0.77539062 ', '-10.0546875 '], time:0.691628ms, TFLOPS: 99.36
               f16wmma(mma2x4+warp2x4+stage2): ['-20.09375   ', '-0.77539062 ', '-10.0546875 '], time:0.701689ms, TFLOPS: 97.93
            f16wmma(warp2x4+...+stage3+dsmem): ['-20.09375   ', '-0.77539062 ', '-10.0546875 '], time:0.697326ms, TFLOPS: 98.55
            f16wmma(warp2x4+...+stage2+dsmem): ['-20.09375   ', '-0.77539062 ', '-10.0546875 '], time:0.698852ms, TFLOPS: 98.33
          f16wmma(warp2x4+...+stage3+swizzle): ['-20.09375   ', '-0.77539062 ', '-10.0546875 '], time:0.696325ms, TFLOPS: 98.69
          f16wmma(warp2x4+...+stage2+swizzle): ['-20.09375   ', '-0.77539062 ', '-10.0546875 '], time:0.685858ms, TFLOPS: 100.19
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['-20.09375   ', '-0.77539062 ', '-10.0546875 '], time:0.698733ms, TFLOPS: 98.35
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['-20.09375   ', '-0.77539062 ', '-10.0546875 '], time:0.688934ms, TFLOPS: 99.75
                                  f16(cublas): ['-20.09375   ', '-0.77539062 ', '-10.0546875 '], time:0.638651ms, TFLOPS: 107.60
                                       f16_th: ['-20.09375   ', '-0.76464844 ', '-10.0625    '], time:0.619006ms, TFLOPS: 111.02
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=1024
                                   f16(naive): ['15.8984375  ', '-31.296875  ', '-27.140625  '], time:37.72568ms, TFLOPS: 3.64
                          f16x8pack(t8x8+bcf): ['15.8984375  ', '-31.296875  ', '-27.140625  '], time:2.647995ms, TFLOPS: 51.90
                     f16x8pack(t8x8+bcf+dbuf): ['15.8984375  ', '-31.296875  ', '-27.140625  '], time:2.614283ms, TFLOPS: 52.57
                     f16x8pack(t8x8+k16+dbuf): ['15.8984375  ', '-31.296875  ', '-27.140625  '], time:2.472138ms, TFLOPS: 55.60
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['15.9375     ', '-31.296875  ', '-27.09375   '], time:5.859637ms, TFLOPS: 23.46
                              f16wmma(mma4x2): ['15.9375     ', '-31.296875  ', '-27.09375   '], time:2.942276ms, TFLOPS: 46.71
                      f16wmma(mma4x2+warp2x4): ['15.9375     ', '-31.296875  ', '-27.09375   '], time:1.823592ms, TFLOPS: 75.37
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['15.9375     ', '-31.296875  ', '-27.09375   '], time:1.337838ms, TFLOPS: 102.73
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['15.9375     ', '-31.296875  ', '-27.09375   '], time:1.358222ms, TFLOPS: 101.19
               f16wmma(mma2x4+warp2x4+stage3): ['15.9375     ', '-31.296875  ', '-27.09375   '], time:1.306509ms, TFLOPS: 105.20
               f16wmma(mma2x4+warp2x4+stage2): ['15.9375     ', '-31.296875  ', '-27.09375   '], time:1.314020ms, TFLOPS: 104.59
            f16wmma(warp2x4+...+stage3+dsmem): ['15.9375     ', '-31.296875  ', '-27.09375   '], time:1.310944ms, TFLOPS: 104.84
            f16wmma(warp2x4+...+stage2+dsmem): ['15.9375     ', '-31.296875  ', '-27.09375   '], time:1.317644ms, TFLOPS: 104.31
          f16wmma(warp2x4+...+stage3+swizzle): ['15.9375     ', '-31.296875  ', '-27.09375   '], time:1.311779ms, TFLOPS: 104.77
          f16wmma(warp2x4+...+stage2+swizzle): ['15.9375     ', '-31.296875  ', '-27.09375   '], time:1.297187ms, TFLOPS: 105.95
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['15.9375     ', '-31.296875  ', '-27.09375   '], time:1.305961ms, TFLOPS: 105.24
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['15.9375     ', '-31.296875  ', '-27.09375   '], time:1.293301ms, TFLOPS: 106.27
                                  f16(cublas): ['15.9375     ', '-31.296875  ', '-27.09375   '], time:1.240444ms, TFLOPS: 110.80
                                       f16_th: ['15.9453125  ', '-31.390625  ', '-27.125     '], time:1.204419ms, TFLOPS: 114.11
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=2048
                                   f16(naive): ['-27.109375  ', '55.84375    ', '66.75       '], time:75.27086ms, TFLOPS: 3.65
                          f16x8pack(t8x8+bcf): ['-27.109375  ', '55.84375    ', '66.75       '], time:5.387568ms, TFLOPS: 51.02
                     f16x8pack(t8x8+bcf+dbuf): ['-27.109375  ', '55.84375    ', '66.75       '], time:5.241966ms, TFLOPS: 52.44
                     f16x8pack(t8x8+k16+dbuf): ['-27.109375  ', '55.84375    ', '66.75       '], time:4.958200ms, TFLOPS: 55.44
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['-26.9375    ', '55.78125    ', '65.8125     '], time:11.58077ms, TFLOPS: 23.74
                              f16wmma(mma4x2): ['-26.9375    ', '55.78125    ', '65.8125     '], time:5.858707ms, TFLOPS: 46.92
                      f16wmma(mma4x2+warp2x4): ['-26.9375    ', '55.78125    ', '65.8125     '], time:3.523778ms, TFLOPS: 78.01
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['-26.9375    ', '55.78125    ', '65.8125     '], time:2.578425ms, TFLOPS: 106.61
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-26.9375    ', '55.78125    ', '65.8125     '], time:2.601242ms, TFLOPS: 105.67
               f16wmma(mma2x4+warp2x4+stage3): ['-26.9375    ', '55.78125    ', '65.8125     '], time:2.531552ms, TFLOPS: 108.58
               f16wmma(mma2x4+warp2x4+stage2): ['-26.9375    ', '55.78125    ', '65.8125     '], time:2.541685ms, TFLOPS: 108.15
            f16wmma(warp2x4+...+stage3+dsmem): ['-26.9375    ', '55.78125    ', '65.8125     '], time:2.544617ms, TFLOPS: 108.02
            f16wmma(warp2x4+...+stage2+dsmem): ['-26.9375    ', '55.78125    ', '65.8125     '], time:2.540850ms, TFLOPS: 108.18
          f16wmma(warp2x4+...+stage3+swizzle): ['-26.9375    ', '55.78125    ', '65.8125     '], time:2.536463ms, TFLOPS: 108.37
          f16wmma(warp2x4+...+stage2+swizzle): ['-26.9375    ', '55.78125    ', '65.8125     '], time:2.515792ms, TFLOPS: 109.26
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['-26.9375    ', '55.78125    ', '65.8125     '], time:2.526521ms, TFLOPS: 108.80
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['-26.9375    ', '55.78125    ', '65.8125     '], time:2.506589ms, TFLOPS: 109.66
                                  f16(cublas): ['-26.9375    ', '55.78125    ', '65.8125     '], time:2.399086ms, TFLOPS: 114.58
                                       f16_th: ['-26.953125  ', '55.8125     ', '65.8125     '], time:2.380847ms, TFLOPS: 115.45
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=512
                                   f16(naive): ['-4.796875   ', '-7.30078125 ', '-5.11328125 '], time:9.490036ms, TFLOPS: 3.62
                          f16x8pack(t8x8+bcf): ['-4.796875   ', '-7.30078125 ', '-5.11328125 '], time:0.683236ms, TFLOPS: 50.29
                     f16x8pack(t8x8+bcf+dbuf): ['-4.796875   ', '-7.30078125 ', '-5.11328125 '], time:0.675106ms, TFLOPS: 50.90
                     f16x8pack(t8x8+k16+dbuf): ['-4.796875   ', '-7.30078125 ', '-5.11328125 '], time:0.641155ms, TFLOPS: 53.59
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['-4.72265625 ', '-7.3203125  ', '-5.1484375  '], time:1.479578ms, TFLOPS: 23.22
                              f16wmma(mma4x2): ['-4.72265625 ', '-7.3203125  ', '-5.1484375  '], time:0.754594ms, TFLOPS: 45.53
                      f16wmma(mma4x2+warp2x4): ['-4.72265625 ', '-7.3203125  ', '-5.1484375  '], time:0.518560ms, TFLOPS: 66.26
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['-4.72265625 ', '-7.3203125  ', '-5.1484375  '], time:0.373578ms, TFLOPS: 91.97
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-4.72265625 ', '-7.3203125  ', '-5.1484375  '], time:0.374579ms, TFLOPS: 91.73
               f16wmma(mma2x4+warp2x4+stage3): ['-4.72265625 ', '-7.3203125  ', '-5.1484375  '], time:0.357961ms, TFLOPS: 95.99
               f16wmma(mma2x4+warp2x4+stage2): ['-4.72265625 ', '-7.3203125  ', '-5.1484375  '], time:0.362443ms, TFLOPS: 94.80
            f16wmma(warp2x4+...+stage3+dsmem): ['-4.72265625 ', '-7.3203125  ', '-5.1484375  '], time:0.360584ms, TFLOPS: 95.29
            f16wmma(warp2x4+...+stage2+dsmem): ['-4.72265625 ', '-7.3203125  ', '-5.1484375  '], time:0.361824ms, TFLOPS: 94.96
          f16wmma(warp2x4+...+stage3+swizzle): ['-4.72265625 ', '-7.3203125  ', '-5.1484375  '], time:0.361394ms, TFLOPS: 95.08
          f16wmma(warp2x4+...+stage2+swizzle): ['-4.72265625 ', '-7.3203125  ', '-5.1484375  '], time:0.354862ms, TFLOPS: 96.83
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['-4.72265625 ', '-7.3203125  ', '-5.1484375  '], time:0.361776ms, TFLOPS: 94.98
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['-4.72265625 ', '-7.3203125  ', '-5.1484375  '], time:0.355648ms, TFLOPS: 96.61
                                  f16(cublas): ['-4.72265625 ', '-7.3203125  ', '-5.1484375  '], time:0.337624ms, TFLOPS: 101.77
                                       f16_th: ['-4.703125   ', '-7.3046875  ', '-5.140625   '], time:0.318288ms, TFLOPS: 107.95
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=1024
                                   f16(naive): ['7.26953125  ', '2.328125    ', '7.09765625  '], time:18.89035ms, TFLOPS: 3.64
                          f16x8pack(t8x8+bcf): ['7.26953125  ', '2.328125    ', '7.09765625  '], time:1.346158ms, TFLOPS: 51.05
                     f16x8pack(t8x8+bcf+dbuf): ['7.26953125  ', '2.328125    ', '7.09765625  '], time:1.335239ms, TFLOPS: 51.47
                     f16x8pack(t8x8+k16+dbuf): ['7.26953125  ', '2.328125    ', '7.09765625  '], time:1.266884ms, TFLOPS: 54.24
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['7.36328125  ', '2.42382812  ', '7.0078125   '], time:2.927064ms, TFLOPS: 23.48
                              f16wmma(mma4x2): ['7.36328125  ', '2.42382812  ', '7.0078125   '], time:1.477313ms, TFLOPS: 46.52
                      f16wmma(mma4x2+warp2x4): ['7.36328125  ', '2.42382812  ', '7.0078125   '], time:0.956869ms, TFLOPS: 71.82
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['7.36328125  ', '2.42382812  ', '7.0078125   '], time:0.688672ms, TFLOPS: 99.79
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['7.36328125  ', '2.42382812  ', '7.0078125   '], time:0.693678ms, TFLOPS: 99.07
               f16wmma(mma2x4+warp2x4+stage3): ['7.36328125  ', '2.42382812  ', '7.0078125   '], time:0.673699ms, TFLOPS: 102.00
               f16wmma(mma2x4+warp2x4+stage2): ['7.36328125  ', '2.42382812  ', '7.0078125   '], time:0.676822ms, TFLOPS: 101.53
            f16wmma(warp2x4+...+stage3+dsmem): ['7.36328125  ', '2.42382812  ', '7.0078125   '], time:0.675392ms, TFLOPS: 101.75
            f16wmma(warp2x4+...+stage2+dsmem): ['7.36328125  ', '2.42382812  ', '7.0078125   '], time:0.676298ms, TFLOPS: 101.61
          f16wmma(warp2x4+...+stage3+swizzle): ['7.36328125  ', '2.42382812  ', '7.0078125   '], time:0.672411ms, TFLOPS: 102.20
          f16wmma(warp2x4+...+stage2+swizzle): ['7.36328125  ', '2.42382812  ', '7.0078125   '], time:0.664901ms, TFLOPS: 103.35
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['7.36328125  ', '2.42382812  ', '7.0078125   '], time:0.672364ms, TFLOPS: 102.21
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['7.36328125  ', '2.42382812  ', '7.0078125   '], time:0.664758ms, TFLOPS: 103.38
                                  f16(cublas): ['7.36328125  ', '2.42382812  ', '7.0078125   '], time:0.635671ms, TFLOPS: 108.11
                                       f16_th: ['7.3671875   ', '2.43164062  ', '7.0         '], time:0.618934ms, TFLOPS: 111.03
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=2048
                                   f16(naive): ['78.1875     ', '-31.53125   ', '14.6640625  '], time:37.66205ms, TFLOPS: 3.65
                          f16x8pack(t8x8+bcf): ['78.1875     ', '-31.53125   ', '14.6640625  '], time:2.731585ms, TFLOPS: 50.31
                     f16x8pack(t8x8+bcf+dbuf): ['78.1875     ', '-31.53125   ', '14.6640625  '], time:2.659392ms, TFLOPS: 51.68
                     f16x8pack(t8x8+k16+dbuf): ['78.1875     ', '-31.53125   ', '14.6640625  '], time:2.523469ms, TFLOPS: 54.46
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['77.8125     ', '-31.953125  ', '14.7265625  '], time:5.816221ms, TFLOPS: 23.63
                              f16wmma(mma4x2): ['77.8125     ', '-31.953125  ', '14.7265625  '], time:2.949500ms, TFLOPS: 46.60
                      f16wmma(mma4x2+warp2x4): ['77.8125     ', '-31.953125  ', '14.7265625  '], time:1.816320ms, TFLOPS: 75.67
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['77.8125     ', '-31.953125  ', '14.7265625  '], time:1.323175ms, TFLOPS: 103.87
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['77.8125     ', '-31.953125  ', '14.7265625  '], time:1.334881ms, TFLOPS: 102.96
               f16wmma(mma2x4+warp2x4+stage3): ['77.8125     ', '-31.953125  ', '14.7265625  '], time:1.302218ms, TFLOPS: 105.54
               f16wmma(mma2x4+warp2x4+stage2): ['77.8125     ', '-31.953125  ', '14.7265625  '], time:1.306295ms, TFLOPS: 105.21
            f16wmma(warp2x4+...+stage3+dsmem): ['77.8125     ', '-31.953125  ', '14.7265625  '], time:1.309323ms, TFLOPS: 104.97
            f16wmma(warp2x4+...+stage2+dsmem): ['77.8125     ', '-31.953125  ', '14.7265625  '], time:1.305937ms, TFLOPS: 105.24
          f16wmma(warp2x4+...+stage3+swizzle): ['77.8125     ', '-31.953125  ', '14.7265625  '], time:1.298594ms, TFLOPS: 105.84
          f16wmma(warp2x4+...+stage2+swizzle): ['77.8125     ', '-31.953125  ', '14.7265625  '], time:1.286983ms, TFLOPS: 106.79
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['77.8125     ', '-31.953125  ', '14.7265625  '], time:1.298689ms, TFLOPS: 105.83
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['77.8125     ', '-31.953125  ', '14.7265625  '], time:1.287865ms, TFLOPS: 106.72
                                  f16(cublas): ['77.8125     ', '-31.953125  ', '14.7265625  '], time:1.237010ms, TFLOPS: 111.11
                                       f16_th: ['78.0        ', '-31.90625   ', '14.6328125  '], time:1.290225ms, TFLOPS: 106.52
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=512
                                   f16(naive): ['-2.24609375 ', '39.03125    ', '-9.671875   '], time:18.95790ms, TFLOPS: 3.62
                          f16x8pack(t8x8+bcf): ['-2.24609375 ', '39.03125    ', '-9.671875   '], time:1.339864ms, TFLOPS: 51.29
                     f16x8pack(t8x8+bcf+dbuf): ['-2.24609375 ', '39.03125    ', '-9.671875   '], time:1.315975ms, TFLOPS: 52.22
                     f16x8pack(t8x8+k16+dbuf): ['-2.24609375 ', '39.03125    ', '-9.671875   '], time:1.250886ms, TFLOPS: 54.94
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['-2.24804688 ', '38.9375     ', '-9.6953125  '], time:2.960944ms, TFLOPS: 23.21
                              f16wmma(mma4x2): ['-2.24804688 ', '38.9375     ', '-9.6953125  '], time:1.499247ms, TFLOPS: 45.84
                      f16wmma(mma4x2+warp2x4): ['-2.24804688 ', '38.9375     ', '-9.6953125  '], time:0.979161ms, TFLOPS: 70.18
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['-2.24804688 ', '38.9375     ', '-9.6953125  '], time:0.721836ms, TFLOPS: 95.20
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-2.24804688 ', '38.9375     ', '-9.6953125  '], time:0.729393ms, TFLOPS: 94.21
               f16wmma(mma2x4+warp2x4+stage3): ['-2.24804688 ', '38.9375     ', '-9.6953125  '], time:0.692129ms, TFLOPS: 99.29
               f16wmma(mma2x4+warp2x4+stage2): ['-2.24804688 ', '38.9375     ', '-9.6953125  '], time:0.699949ms, TFLOPS: 98.18
            f16wmma(warp2x4+...+stage3+dsmem): ['-2.24804688 ', '38.9375     ', '-9.6953125  '], time:0.696325ms, TFLOPS: 98.69
            f16wmma(warp2x4+...+stage2+dsmem): ['-2.24804688 ', '38.9375     ', '-9.6953125  '], time:0.698924ms, TFLOPS: 98.32
          f16wmma(warp2x4+...+stage3+swizzle): ['-2.24804688 ', '38.9375     ', '-9.6953125  '], time:0.697708ms, TFLOPS: 98.49
          f16wmma(warp2x4+...+stage2+swizzle): ['-2.24804688 ', '38.9375     ', '-9.6953125  '], time:0.686192ms, TFLOPS: 100.15
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['-2.24804688 ', '38.9375     ', '-9.6953125  '], time:0.695800ms, TFLOPS: 98.76
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['-2.24804688 ', '38.9375     ', '-9.6953125  '], time:0.687861ms, TFLOPS: 99.90
                                  f16(cublas): ['-2.24804688 ', '38.9375     ', '-9.6953125  '], time:0.640630ms, TFLOPS: 107.27
                                       f16_th: ['-2.24023438 ', '38.9375     ', '-9.671875   '], time:0.618004ms, TFLOPS: 111.20
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=1024
                                   f16(naive): ['48.71875    ', '71.0625     ', '39.28125    '], time:37.73891ms, TFLOPS: 3.64
                          f16x8pack(t8x8+bcf): ['48.71875    ', '71.0625     ', '39.28125    '], time:2.655243ms, TFLOPS: 51.76
                     f16x8pack(t8x8+bcf+dbuf): ['48.71875    ', '71.0625     ', '39.28125    '], time:2.603912ms, TFLOPS: 52.78
                     f16x8pack(t8x8+k16+dbuf): ['48.71875    ', '71.0625     ', '39.28125    '], time:2.472829ms, TFLOPS: 55.58
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['48.5625     ', '71.0        ', '39.21875    '], time:5.841517ms, TFLOPS: 23.53
                              f16wmma(mma4x2): ['48.5625     ', '71.0        ', '39.21875    '], time:2.952837ms, TFLOPS: 46.54
                      f16wmma(mma4x2+warp2x4): ['48.5625     ', '71.0        ', '39.21875    '], time:1.830434ms, TFLOPS: 75.09
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['48.5625     ', '71.0        ', '39.21875    '], time:1.340746ms, TFLOPS: 102.51
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['48.5625     ', '71.0        ', '39.21875    '], time:1.352071ms, TFLOPS: 101.65
               f16wmma(mma2x4+warp2x4+stage3): ['48.5625     ', '71.0        ', '39.21875    '], time:1.305294ms, TFLOPS: 105.29
               f16wmma(mma2x4+warp2x4+stage2): ['48.5625     ', '71.0        ', '39.21875    '], time:1.314735ms, TFLOPS: 104.54
            f16wmma(warp2x4+...+stage3+dsmem): ['48.5625     ', '71.0        ', '39.21875    '], time:1.312541ms, TFLOPS: 104.71
            f16wmma(warp2x4+...+stage2+dsmem): ['48.5625     ', '71.0        ', '39.21875    '], time:1.311588ms, TFLOPS: 104.79
          f16wmma(warp2x4+...+stage3+swizzle): ['48.5625     ', '71.0        ', '39.21875    '], time:1.313185ms, TFLOPS: 104.66
          f16wmma(warp2x4+...+stage2+swizzle): ['48.5625     ', '71.0        ', '39.21875    '], time:1.298713ms, TFLOPS: 105.83
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['48.5625     ', '71.0        ', '39.21875    '], time:1.307201ms, TFLOPS: 105.14
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['48.5625     ', '71.0        ', '39.21875    '], time:1.294422ms, TFLOPS: 106.18
                                  f16(cublas): ['48.5625     ', '71.0        ', '39.21875    '], time:1.224398ms, TFLOPS: 112.25
                                       f16_th: ['48.5        ', '71.0625     ', '39.03125    '], time:1.204466ms, TFLOPS: 114.11
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=2048
                                   f16(naive): ['-34.4375    ', '20.515625   ', '15.4296875  '], time:75.25849ms, TFLOPS: 3.65
                          f16x8pack(t8x8+bcf): ['-34.4375    ', '20.515625   ', '15.4296875  '], time:5.287170ms, TFLOPS: 51.99
                     f16x8pack(t8x8+bcf+dbuf): ['-34.4375    ', '20.515625   ', '15.4296875  '], time:5.215668ms, TFLOPS: 52.70
                     f16x8pack(t8x8+k16+dbuf): ['-34.4375    ', '20.515625   ', '15.4296875  '], time:5.070400ms, TFLOPS: 54.21
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['-34.25      ', '20.421875   ', '15.3359375  '], time:11.59327ms, TFLOPS: 23.71
                              f16wmma(mma4x2): ['-34.25      ', '20.421875   ', '15.3359375  '], time:5.877757ms, TFLOPS: 46.77
                      f16wmma(mma4x2+warp2x4): ['-34.25      ', '20.421875   ', '15.3359375  '], time:3.544974ms, TFLOPS: 77.54
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['-34.25      ', '20.421875   ', '15.3359375  '], time:2.578687ms, TFLOPS: 106.60
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-34.25      ', '20.421875   ', '15.3359375  '], time:2.598357ms, TFLOPS: 105.79
               f16wmma(mma2x4+warp2x4+stage3): ['-34.25      ', '20.421875   ', '15.3359375  '], time:2.542924ms, TFLOPS: 108.10
               f16wmma(mma2x4+warp2x4+stage2): ['-34.25      ', '20.421875   ', '15.3359375  '], time:2.551841ms, TFLOPS: 107.72
            f16wmma(warp2x4+...+stage3+dsmem): ['-34.25      ', '20.421875   ', '15.3359375  '], time:2.547001ms, TFLOPS: 107.92
            f16wmma(warp2x4+...+stage2+dsmem): ['-34.25      ', '20.421875   ', '15.3359375  '], time:2.542281ms, TFLOPS: 108.12
          f16wmma(warp2x4+...+stage3+swizzle): ['-34.25      ', '20.421875   ', '15.3359375  '], time:2.528476ms, TFLOPS: 108.71
          f16wmma(warp2x4+...+stage2+swizzle): ['-34.25      ', '20.421875   ', '15.3359375  '], time:2.508544ms, TFLOPS: 109.58
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['-34.25      ', '20.421875   ', '15.3359375  '], time:2.527260ms, TFLOPS: 108.77
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['-34.25      ', '20.421875   ', '15.3359375  '], time:2.508091ms, TFLOPS: 109.60
                                  f16(cublas): ['-34.25      ', '20.421875   ', '15.3359375  '], time:2.400755ms, TFLOPS: 114.50
                                       f16_th: ['-34.5625    ', '20.515625   ', '15.3359375  '], time:2.381229ms, TFLOPS: 115.44
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=512
                                   f16(naive): ['-1.52441406 ', '-14.75      ', '54.28125    '], time:37.89505ms, TFLOPS: 3.63
                          f16x8pack(t8x8+bcf): ['-1.52441406 ', '-14.75      ', '54.28125    '], time:2.650403ms, TFLOPS: 51.86
                     f16x8pack(t8x8+bcf+dbuf): ['-1.52441406 ', '-14.75      ', '54.28125    '], time:2.608299ms, TFLOPS: 52.69
                     f16x8pack(t8x8+k16+dbuf): ['-1.52441406 ', '-14.75      ', '54.28125    '], time:2.482652ms, TFLOPS: 55.36
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['-1.50097656 ', '-14.7109375 ', '54.34375    '], time:5.941462ms, TFLOPS: 23.13
                              f16wmma(mma4x2): ['-1.50097656 ', '-14.7109375 ', '54.34375    '], time:2.982974ms, TFLOPS: 46.07
                      f16wmma(mma4x2+warp2x4): ['-1.50097656 ', '-14.7109375 ', '54.34375    '], time:1.905989ms, TFLOPS: 72.11
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['-1.50097656 ', '-14.7109375 ', '54.34375    '], time:1.408457ms, TFLOPS: 97.58
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-1.50097656 ', '-14.7109375 ', '54.34375    '], time:1.427769ms, TFLOPS: 96.26
               f16wmma(mma2x4+warp2x4+stage3): ['-1.50097656 ', '-14.7109375 ', '54.34375    '], time:1.358413ms, TFLOPS: 101.18
               f16wmma(mma2x4+warp2x4+stage2): ['-1.50097656 ', '-14.7109375 ', '54.34375    '], time:1.369357ms, TFLOPS: 100.37
            f16wmma(warp2x4+...+stage3+dsmem): ['-1.50097656 ', '-14.7109375 ', '54.34375    '], time:1.366710ms, TFLOPS: 100.56
            f16wmma(warp2x4+...+stage2+dsmem): ['-1.50097656 ', '-14.7109375 ', '54.34375    '], time:1.366329ms, TFLOPS: 100.59
          f16wmma(warp2x4+...+stage3+swizzle): ['-1.50097656 ', '-14.7109375 ', '54.34375    '], time:1.363706ms, TFLOPS: 100.78
          f16wmma(warp2x4+...+stage2+swizzle): ['-1.50097656 ', '-14.7109375 ', '54.34375    '], time:1.343870ms, TFLOPS: 102.27
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['-1.50097656 ', '-14.7109375 ', '54.34375    '], time:1.364278ms, TFLOPS: 100.74
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['-1.50097656 ', '-14.7109375 ', '54.34375    '], time:1.348114ms, TFLOPS: 101.95
                                  f16(cublas): ['-1.50097656 ', '-14.7109375 ', '54.34375    '], time:1.248025ms, TFLOPS: 110.13
                                       f16_th: ['-1.50195312 ', '-14.7421875 ', '54.34375    '], time:1.226782ms, TFLOPS: 112.03
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=1024
                                   f16(naive): ['39.59375    ', '-2.73046875 ', '-27.375     '], time:75.44095ms, TFLOPS: 3.64
                          f16x8pack(t8x8+bcf): ['39.59375    ', '-2.73046875 ', '-27.375     '], time:5.309510ms, TFLOPS: 51.77
                     f16x8pack(t8x8+bcf+dbuf): ['39.59375    ', '-2.73046875 ', '-27.375     '], time:5.337309ms, TFLOPS: 51.50
                     f16x8pack(t8x8+k16+dbuf): ['39.59375    ', '-2.73046875 ', '-27.375     '], time:5.068349ms, TFLOPS: 54.23
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['39.65625    ', '-2.828125   ', '-27.265625  '], time:11.67445ms, TFLOPS: 23.55
                              f16wmma(mma4x2): ['39.65625    ', '-2.828125   ', '-27.265625  '], time:5.865192ms, TFLOPS: 46.87
                      f16wmma(mma4x2+warp2x4): ['39.65625    ', '-2.828125   ', '-27.265625  '], time:3.560590ms, TFLOPS: 77.20
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['39.65625    ', '-2.828125   ', '-27.265625  '], time:2.645897ms, TFLOPS: 103.89
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['39.65625    ', '-2.828125   ', '-27.265625  '], time:2.674484ms, TFLOPS: 102.78
               f16wmma(mma2x4+warp2x4+stage3): ['39.65625    ', '-2.828125   ', '-27.265625  '], time:2.590298ms, TFLOPS: 106.12
               f16wmma(mma2x4+warp2x4+stage2): ['39.65625    ', '-2.828125   ', '-27.265625  '], time:2.593374ms, TFLOPS: 105.99
            f16wmma(warp2x4+...+stage3+dsmem): ['39.65625    ', '-2.828125   ', '-27.265625  '], time:2.600646ms, TFLOPS: 105.70
            f16wmma(warp2x4+...+stage2+dsmem): ['39.65625    ', '-2.828125   ', '-27.265625  '], time:2.591443ms, TFLOPS: 106.07
          f16wmma(warp2x4+...+stage3+swizzle): ['39.65625    ', '-2.828125   ', '-27.265625  '], time:2.579808ms, TFLOPS: 106.55
          f16wmma(warp2x4+...+stage2+swizzle): ['39.65625    ', '-2.828125   ', '-27.265625  '], time:2.553772ms, TFLOPS: 107.64
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['39.65625    ', '-2.828125   ', '-27.265625  '], time:2.576017ms, TFLOPS: 106.71
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['39.65625    ', '-2.828125   ', '-27.265625  '], time:2.562522ms, TFLOPS: 107.27
                                  f16(cublas): ['39.65625    ', '-2.828125   ', '-27.265625  '], time:2.421426ms, TFLOPS: 113.52
                                       f16_th: ['39.71875    ', '-2.85351562 ', '-27.328125  '], time:2.398848ms, TFLOPS: 114.59
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=2048
                                   f16(naive): ['7.09765625  ', '4.65625     ', '129.75      '], time:150.4832ms, TFLOPS: 3.65
                          f16x8pack(t8x8+bcf): ['7.09765625  ', '4.65625     ', '129.75      '], time:10.81454ms, TFLOPS: 50.83
                     f16x8pack(t8x8+bcf+dbuf): ['7.09765625  ', '4.65625     ', '129.75      '], time:10.76550ms, TFLOPS: 51.07
                     f16x8pack(t8x8+k16+dbuf): ['7.09765625  ', '4.65625     ', '129.75      '], time:10.29613ms, TFLOPS: 53.39
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['7.3671875   ', '4.4765625   ', '130.5       '], time:23.13179ms, TFLOPS: 23.77
                              f16wmma(mma4x2): ['7.3671875   ', '4.4765625   ', '130.5       '], time:11.75928ms, TFLOPS: 46.75
                      f16wmma(mma4x2+warp2x4): ['7.3671875   ', '4.4765625   ', '130.5       '], time:6.885719ms, TFLOPS: 79.84
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['7.3671875   ', '4.4765625   ', '130.5       '], time:5.109715ms, TFLOPS: 107.59
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['7.3671875   ', '4.4765625   ', '130.5       '], time:5.174016ms, TFLOPS: 106.25
               f16wmma(mma2x4+warp2x4+stage3): ['7.3671875   ', '4.4765625   ', '130.5       '], time:5.019330ms, TFLOPS: 109.53
               f16wmma(mma2x4+warp2x4+stage2): ['7.3671875   ', '4.4765625   ', '130.5       '], time:5.038428ms, TFLOPS: 109.11
            f16wmma(warp2x4+...+stage3+dsmem): ['7.3671875   ', '4.4765625   ', '130.5       '], time:5.068945ms, TFLOPS: 108.46
            f16wmma(warp2x4+...+stage2+dsmem): ['7.3671875   ', '4.4765625   ', '130.5       '], time:5.047750ms, TFLOPS: 108.91
          f16wmma(warp2x4+...+stage3+swizzle): ['7.3671875   ', '4.4765625   ', '130.5       '], time:5.007791ms, TFLOPS: 109.78
          f16wmma(warp2x4+...+stage2+swizzle): ['7.3671875   ', '4.4765625   ', '130.5       '], time:4.964947ms, TFLOPS: 110.73
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['7.3671875   ', '4.4765625   ', '130.5       '], time:5.006980ms, TFLOPS: 109.80
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['7.3671875   ', '4.4765625   ', '130.5       '], time:4.968047ms, TFLOPS: 110.66
                                  f16(cublas): ['7.3671875   ', '4.4765625   ', '130.5       '], time:4.763627ms, TFLOPS: 115.41
                                       f16_th: ['7.37890625  ', '4.5078125   ', '130.25      '], time:4.757499ms, TFLOPS: 115.56
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=512
                                   f16(naive): ['30.125      ', '-0.15991211 ', '16.875      '], time:18.96388ms, TFLOPS: 3.62
                          f16x8pack(t8x8+bcf): ['30.125      ', '-0.15991211 ', '16.875      '], time:1.347732ms, TFLOPS: 50.99
                     f16x8pack(t8x8+bcf+dbuf): ['30.125      ', '-0.15991211 ', '16.875      '], time:1.309132ms, TFLOPS: 52.49
                     f16x8pack(t8x8+k16+dbuf): ['30.125      ', '-0.15991211 ', '16.875      '], time:1.245999ms, TFLOPS: 55.15
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['29.953125   ', '-0.03341675 ', '16.875      '], time:2.950358ms, TFLOPS: 23.29
                              f16wmma(mma4x2): ['29.953125   ', '-0.03341675 ', '16.875      '], time:1.503968ms, TFLOPS: 45.69
                      f16wmma(mma4x2+warp2x4): ['29.953125   ', '-0.03341675 ', '16.875      '], time:0.975728ms, TFLOPS: 70.43
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['29.953125   ', '-0.03341675 ', '16.875      '], time:0.721406ms, TFLOPS: 95.26
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['29.953125   ', '-0.03341675 ', '16.875      '], time:0.729632ms, TFLOPS: 94.18
               f16wmma(mma2x4+warp2x4+stage3): ['29.953125   ', '-0.03341675 ', '16.875      '], time:0.691914ms, TFLOPS: 99.32
               f16wmma(mma2x4+warp2x4+stage2): ['29.953125   ', '-0.03341675 ', '16.875      '], time:0.699710ms, TFLOPS: 98.21
            f16wmma(warp2x4+...+stage3+dsmem): ['29.953125   ', '-0.03341675 ', '16.875      '], time:0.694179ms, TFLOPS: 98.99
            f16wmma(warp2x4+...+stage2+dsmem): ['29.953125   ', '-0.03341675 ', '16.875      '], time:0.697875ms, TFLOPS: 98.47
          f16wmma(warp2x4+...+stage3+swizzle): ['29.953125   ', '-0.03341675 ', '16.875      '], time:0.695395ms, TFLOPS: 98.82
          f16wmma(warp2x4+...+stage2+swizzle): ['29.953125   ', '-0.03341675 ', '16.875      '], time:0.688123ms, TFLOPS: 99.87
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['29.953125   ', '-0.03341675 ', '16.875      '], time:0.698375ms, TFLOPS: 98.40
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['29.953125   ', '-0.03341675 ', '16.875      '], time:0.688290ms, TFLOPS: 99.84
                                  f16(cublas): ['29.953125   ', '-0.03341675 ', '16.875      '], time:0.637674ms, TFLOPS: 107.77
                                       f16_th: ['29.96875    ', '-0.02645874 ', '16.859375   '], time:0.617384ms, TFLOPS: 111.31
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=1024
                                   f16(naive): ['-55.0       ', '30.453125   ', '-24.21875   '], time:37.73779ms, TFLOPS: 3.64
                          f16x8pack(t8x8+bcf): ['-55.0       ', '30.453125   ', '-24.21875   '], time:2.653264ms, TFLOPS: 51.80
                     f16x8pack(t8x8+bcf+dbuf): ['-55.0       ', '30.453125   ', '-24.21875   '], time:2.601957ms, TFLOPS: 52.82
                     f16x8pack(t8x8+k16+dbuf): ['-55.0       ', '30.453125   ', '-24.21875   '], time:2.489256ms, TFLOPS: 55.21
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['-55.0625    ', '30.421875   ', '-24.203125  '], time:5.854630ms, TFLOPS: 23.48
                              f16wmma(mma4x2): ['-55.0625    ', '30.421875   ', '-24.203125  '], time:2.944421ms, TFLOPS: 46.68
                      f16wmma(mma4x2+warp2x4): ['-55.0625    ', '30.421875   ', '-24.203125  '], time:1.830911ms, TFLOPS: 75.07
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['-55.0625    ', '30.421875   ', '-24.203125  '], time:1.339173ms, TFLOPS: 102.63
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-55.0625    ', '30.421875   ', '-24.203125  '], time:1.353573ms, TFLOPS: 101.54
               f16wmma(mma2x4+warp2x4+stage3): ['-55.0625    ', '30.421875   ', '-24.203125  '], time:1.305890ms, TFLOPS: 105.25
               f16wmma(mma2x4+warp2x4+stage2): ['-55.0625    ', '30.421875   ', '-24.203125  '], time:1.314163ms, TFLOPS: 104.58
            f16wmma(warp2x4+...+stage3+dsmem): ['-55.0625    ', '30.421875   ', '-24.203125  '], time:1.313018ms, TFLOPS: 104.67
            f16wmma(warp2x4+...+stage2+dsmem): ['-55.0625    ', '30.421875   ', '-24.203125  '], time:1.311230ms, TFLOPS: 104.82
          f16wmma(warp2x4+...+stage3+swizzle): ['-55.0625    ', '30.421875   ', '-24.203125  '], time:1.306200ms, TFLOPS: 105.22
          f16wmma(warp2x4+...+stage2+swizzle): ['-55.0625    ', '30.421875   ', '-24.203125  '], time:1.292943ms, TFLOPS: 106.30
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['-55.0625    ', '30.421875   ', '-24.203125  '], time:1.305413ms, TFLOPS: 105.28
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['-55.0625    ', '30.421875   ', '-24.203125  '], time:1.292705ms, TFLOPS: 106.32
                                  f16(cublas): ['-55.0625    ', '30.421875   ', '-24.203125  '], time:1.223444ms, TFLOPS: 112.34
                                       f16_th: ['-55.03125   ', '30.40625    ', '-24.1875    '], time:1.204776ms, TFLOPS: 114.08
----------------------------------------------------------------------------------------------------------------------------------
```
