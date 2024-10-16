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
- [X] hgemm_wmma_m16n16k16_mma4x4_warp4x4_stage2/3/4(Tensor Cores, Tile MMA/Warp, Copy Async, Stage, Pad, Thread block swizzle) 
- [X] PyTorch bindings

目前最优的实现，在L20上（理论Tensor Cores FP16算力为 119.5 TFLOPS），能达到cuBLAS大概95%~98%左右的性能(105~110 TFLOPS)，部分case会超越cuBLAS。已知问题为bank conflicts没有完全消除，目前通过padding的方式缓解bank conflicts会导致shared memory浪费，也会影响SM occupancy。并且尚未手工实现Warp swizzle(受限于WMMA API的灵活性以及本人的能力)，后续将会尝试通过MMA PTX实现warp swizzle。

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
                                                       M=16384, N=8192, K=2048
                              f16(naive): ['41.78125  ', '24.546875 '], time:150.4883ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                     f16x8pack(t8x8+bcf): ['41.78125  ', '24.546875 '], time:11.30933ms, swizzle: NOOP, TFLOPS: 48.61 (+1230.66%)
                f16x8pack(t8x8+bcf+dbuf): ['41.78125  ', '24.546875 '], time:11.25104ms, swizzle: NOOP, TFLOPS: 48.86 (+0.52%)
                f16x8pack(t8x8+k16+dbuf): ['41.78125  ', '24.546875 '], time:10.76521ms, swizzle: NOOP, TFLOPS: 51.07 (+4.51%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['41.75     ', '24.78125  '], time:23.30009ms, swizzle: NOOP, TFLOPS: 23.59
                         f16wmma(mma4x2): ['41.75     ', '24.78125  '], time:11.70096ms, swizzle: NOOP, TFLOPS: 46.98
                 f16wmma(mma4x2+warp2x4): ['41.75     ', '24.78125  '], time:6.929373ms, swizzle: NOOP, TFLOPS: 79.34 (+55.36%)
            f16wmma(mma2x4+warp2x4+dbuf): ['41.75     ', '24.78125  '], time:5.218052ms, swizzle: NOOP, TFLOPS: 105.36(+32.80%)
          f16wmma(mma2x4+warp2x4+stage4): ['41.75     ', '24.78125  '], time:5.273580ms, swizzle: NOOP, TFLOPS: 104.25
          f16wmma(mma2x4+warp2x4+stage3): ['41.75     ', '24.78125  '], time:5.188989ms, swizzle: NOOP, TFLOPS: 105.95(+0.56%)
          f16wmma(mma2x4+warp2x4+stage2): ['41.75     ', '24.78125  '], time:5.200719ms, swizzle: NOOP, TFLOPS: 105.71
        f16wmma(mma2x4+...+stage3+dsmem): ['41.75     ', '24.78125  '], time:5.265188ms, swizzle: NOOP, TFLOPS: 104.41
        f16wmma(mma2x4+...+stage3+dsmem): ['41.75     ', '24.78125  '], time:5.212473ms, swizzle: NOOP, TFLOPS: 105.47
        f16wmma(mma2x4+...+stage2+dsmem): ['41.75     ', '24.78125  '], time:5.207681ms, swizzle: NOOP, TFLOPS: 105.57
        f16wmma(mma4x4+...+stage4+dsmem): ['41.75     ', '24.78125  '], time:5.383682ms, swizzle: NOOP, TFLOPS: 102.12
        f16wmma(mma4x4+...+stage3+dsmem): ['41.75     ', '24.78125  '], time:5.479192ms, swizzle: NOOP, TFLOPS: 100.34
        f16wmma(mma4x4+...+stage2+dsmem): ['41.75     ', '24.78125  '], time:5.644440ms, swizzle: NOOP, TFLOPS: 97.40
      f16wmma(mma2x4+...+stage4+swizzle): ['41.75     ', '24.78125  '], time:5.351376ms, swizzle: 2048, TFLOPS: 102.73
      f16wmma(mma2x4+...+stage3+swizzle): ['41.75     ', '24.78125  '], time:5.263781ms, swizzle: 2048, TFLOPS: 104.44
      f16wmma(mma2x4+...+stage2+swizzle): ['41.75     ', '24.78125  '], time:5.228519ms, swizzle: 2048, TFLOPS: 105.15
       f16wmma(...+stage3+dsmem+swizzle): ['41.75     ', '24.78125  '], time:5.311512ms, swizzle: 2048, TFLOPS: 103.50
       f16wmma(...+stage2+dsmem+swizzle): ['41.75     ', '24.78125  '], time:5.231094ms, swizzle: 2048, TFLOPS: 105.09
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['41.75     ', '24.78125  '], time:5.459690ms, swizzle: 2048, TFLOPS: 100.69
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['41.75     ', '24.78125  '], time:5.362582ms, swizzle: 2048, TFLOPS: 102.52
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['41.75     ', '24.78125  '], time:5.480861ms, swizzle: 2048, TFLOPS: 100.30
                             f16(cublas): ['41.75     ', '24.78125  '], time:4.996967ms, swizzle: NOOP, TFLOPS: 110.02(+3.84%)
                                  f16_th: ['41.65625  ', '24.765625 '], time:4.819083ms, swizzle: NOOP, TFLOPS: 114.08(+3.69%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=4096
                              f16(naive): ['70.0      ', '-2.0957031'], time:300.5908ms, swizzle: NOOP, TFLOPS: 3.66  (+0.00%)
                     f16x8pack(t8x8+bcf): ['70.0      ', '-2.0957031'], time:22.41368ms, swizzle: NOOP, TFLOPS: 49.06 (+1241.10%)
                f16x8pack(t8x8+bcf+dbuf): ['70.0      ', '-2.0957031'], time:22.27663ms, swizzle: NOOP, TFLOPS: 49.36 (+0.62%)
                f16x8pack(t8x8+k16+dbuf): ['70.0      ', '-2.0957031'], time:21.95541ms, swizzle: NOOP, TFLOPS: 50.08 (+1.46%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['69.5625   ', '-1.7373046'], time:46.20220ms, swizzle: NOOP, TFLOPS: 23.80
                         f16wmma(mma4x2): ['69.5625   ', '-1.7373046'], time:23.24860ms, swizzle: NOOP, TFLOPS: 47.29
                 f16wmma(mma4x2+warp2x4): ['69.5625   ', '-1.7373046'], time:13.63050ms, swizzle: NOOP, TFLOPS: 80.67 (+61.08%)
            f16wmma(mma2x4+warp2x4+dbuf): ['69.5625   ', '-1.7373046'], time:10.23759ms, swizzle: NOOP, TFLOPS: 107.40(+33.14%)
          f16wmma(mma2x4+warp2x4+stage4): ['69.5625   ', '-1.7373046'], time:10.28592ms, swizzle: NOOP, TFLOPS: 106.89
          f16wmma(mma2x4+warp2x4+stage3): ['69.5625   ', '-1.7373046'], time:10.14952ms, swizzle: NOOP, TFLOPS: 108.33(+0.87%)
          f16wmma(mma2x4+warp2x4+stage2): ['69.5625   ', '-1.7373046'], time:10.17880ms, swizzle: NOOP, TFLOPS: 108.02
        f16wmma(mma2x4+...+stage3+dsmem): ['69.5625   ', '-1.7373046'], time:10.26527ms, swizzle: NOOP, TFLOPS: 107.11
        f16wmma(mma2x4+...+stage3+dsmem): ['69.5625   ', '-1.7373046'], time:10.44397ms, swizzle: NOOP, TFLOPS: 105.28
        f16wmma(mma2x4+...+stage2+dsmem): ['69.5625   ', '-1.7373046'], time:10.38477ms, swizzle: NOOP, TFLOPS: 105.88
        f16wmma(mma4x4+...+stage4+dsmem): ['69.5625   ', '-1.7373046'], time:10.44440ms, swizzle: NOOP, TFLOPS: 105.27
        f16wmma(mma4x4+...+stage3+dsmem): ['69.5625   ', '-1.7373046'], time:10.65192ms, swizzle: NOOP, TFLOPS: 103.22
        f16wmma(mma4x4+...+stage2+dsmem): ['69.5625   ', '-1.7373046'], time:10.97645ms, swizzle: NOOP, TFLOPS: 100.17
      f16wmma(mma2x4+...+stage4+swizzle): ['69.5625   ', '-1.7373046'], time:10.48655ms, swizzle: 2048, TFLOPS: 104.85
      f16wmma(mma2x4+...+stage3+swizzle): ['69.5625   ', '-1.7373046'], time:10.35373ms, swizzle: 2048, TFLOPS: 106.19
      f16wmma(mma2x4+...+stage2+swizzle): ['69.5625   ', '-1.7373046'], time:10.24963ms, swizzle: 2048, TFLOPS: 107.27
       f16wmma(...+stage3+dsmem+swizzle): ['69.5625   ', '-1.7373046'], time:10.40430ms, swizzle: 2048, TFLOPS: 105.68
       f16wmma(...+stage2+dsmem+swizzle): ['69.5625   ', '-1.7373046'], time:10.23306ms, swizzle: 2048, TFLOPS: 107.45
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['69.5625   ', '-1.7373046'], time:10.52174ms, swizzle: 2048, TFLOPS: 104.50
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['69.5625   ', '-1.7373046'], time:10.44552ms, swizzle: 2048, TFLOPS: 105.26
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['69.5625   ', '-1.7373046'], time:10.62233ms, swizzle: 2048, TFLOPS: 103.51
                             f16(cublas): ['69.5625   ', '-1.7373046'], time:9.715437ms, swizzle: NOOP, TFLOPS: 113.17(+4.47%)
                                  f16_th: ['69.8125   ', '-1.7314453'], time:9.576702ms, swizzle: NOOP, TFLOPS: 114.81(+1.45%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=8192
                              f16(naive): ['108.4375  ', '-75.375   '], time:888.0897ms, swizzle: NOOP, TFLOPS: 2.48  (+0.00%)
                     f16x8pack(t8x8+bcf): ['108.4375  ', '-75.375   '], time:48.78988ms, swizzle: NOOP, TFLOPS: 45.07 (+1720.23%)
                f16x8pack(t8x8+bcf+dbuf): ['108.4375  ', '-75.375   '], time:50.18644ms, swizzle: NOOP, TFLOPS: 43.82
                f16x8pack(t8x8+k16+dbuf): ['108.4375  ', '-75.375   '], time:50.16939ms, swizzle: NOOP, TFLOPS: 43.83
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['106.8125  ', '-74.875   '], time:99.48325ms, swizzle: NOOP, TFLOPS: 22.10
                         f16wmma(mma4x2): ['106.8125  ', '-74.875   '], time:51.45063ms, swizzle: NOOP, TFLOPS: 42.74
                 f16wmma(mma4x2+warp2x4): ['106.8125  ', '-74.875   '], time:28.05333ms, swizzle: NOOP, TFLOPS: 78.39 (+73.92%)
            f16wmma(mma2x4+warp2x4+dbuf): ['106.8125  ', '-74.875   '], time:22.40943ms, swizzle: NOOP, TFLOPS: 98.13 (+25.19%)
          f16wmma(mma2x4+warp2x4+stage4): ['106.8125  ', '-74.875   '], time:22.76268ms, swizzle: NOOP, TFLOPS: 96.61
          f16wmma(mma2x4+warp2x4+stage3): ['106.8125  ', '-74.875   '], time:23.64182ms, swizzle: NOOP, TFLOPS: 93.01
          f16wmma(mma2x4+warp2x4+stage2): ['106.8125  ', '-74.875   '], time:22.27849ms, swizzle: NOOP, TFLOPS: 98.71 (+0.59%)
        f16wmma(mma2x4+...+stage3+dsmem): ['106.8125  ', '-74.875   '], time:23.07252ms, swizzle: NOOP, TFLOPS: 95.31
        f16wmma(mma2x4+...+stage3+dsmem): ['106.8125  ', '-74.875   '], time:23.36189ms, swizzle: NOOP, TFLOPS: 94.13
        f16wmma(mma2x4+...+stage2+dsmem): ['106.8125  ', '-74.875   '], time:22.26281ms, swizzle: NOOP, TFLOPS: 98.78 (+0.07%)
        f16wmma(mma4x4+...+stage4+dsmem): ['106.8125  ', '-74.875   '], time:20.58002ms, swizzle: NOOP, TFLOPS: 106.85(+8.18%)
        f16wmma(mma4x4+...+stage3+dsmem): ['106.8125  ', '-74.875   '], time:20.87032ms, swizzle: NOOP, TFLOPS: 105.37
        f16wmma(mma4x4+...+stage2+dsmem): ['106.8125  ', '-74.875   '], time:21.68383ms, swizzle: NOOP, TFLOPS: 101.41
      f16wmma(mma2x4+...+stage4+swizzle): ['106.8125  ', '-74.875   '], time:20.66783ms, swizzle: 2048, TFLOPS: 106.40
      f16wmma(mma2x4+...+stage3+swizzle): ['106.8125  ', '-74.875   '], time:20.34959ms, swizzle: 2048, TFLOPS: 108.06(+1.13%)
      f16wmma(mma2x4+...+stage2+swizzle): ['106.8125  ', '-74.875   '], time:20.61433ms, swizzle: 2048, TFLOPS: 106.67
       f16wmma(...+stage3+dsmem+swizzle): ['106.8125  ', '-74.875   '], time:20.74484ms, swizzle: 2048, TFLOPS: 106.00
       f16wmma(...+stage2+dsmem+swizzle): ['106.8125  ', '-74.875   '], time:20.14210ms, swizzle: 2048, TFLOPS: 109.18(+1.03%)
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['106.8125  ', '-74.875   '], time:20.70202ms, swizzle: 2048, TFLOPS: 106.22
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['106.8125  ', '-74.875   '], time:20.44272ms, swizzle: 2048, TFLOPS: 107.57
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['106.8125  ', '-74.875   '], time:20.94430ms, swizzle: 2048, TFLOPS: 104.99
                             f16(cublas): ['106.8125  ', '-74.875   '], time:19.31245ms, swizzle: NOOP, TFLOPS: 113.87(+4.30%)
                                  f16_th: ['107.25    ', '-75.0     '], time:19.39303ms, swizzle: NOOP, TFLOPS: 113.39
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=2048
                              f16(naive): ['41.78125  ', '24.546875 '], time:300.9659ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                     f16x8pack(t8x8+bcf): ['41.78125  ', '24.546875 '], time:22.34311ms, swizzle: NOOP, TFLOPS: 49.21 (+1247.02%)
                f16x8pack(t8x8+bcf+dbuf): ['41.78125  ', '24.546875 '], time:22.19541ms, swizzle: NOOP, TFLOPS: 49.54 (+0.67%)
                f16x8pack(t8x8+k16+dbuf): ['41.78125  ', '24.546875 '], time:21.97241ms, swizzle: NOOP, TFLOPS: 50.04 (+1.01%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['41.75     ', '24.78125  '], time:46.33269ms, swizzle: NOOP, TFLOPS: 23.73
                         f16wmma(mma4x2): ['41.75     ', '24.78125  '], time:23.37417ms, swizzle: NOOP, TFLOPS: 47.04
                 f16wmma(mma4x2+warp2x4): ['41.75     ', '24.78125  '], time:13.65461ms, swizzle: NOOP, TFLOPS: 80.52 (+60.92%)
            f16wmma(mma2x4+warp2x4+dbuf): ['41.75     ', '24.78125  '], time:10.44301ms, swizzle: NOOP, TFLOPS: 105.29(+30.75%)
          f16wmma(mma2x4+warp2x4+stage4): ['41.75     ', '24.78125  '], time:10.50512ms, swizzle: NOOP, TFLOPS: 104.66
          f16wmma(mma2x4+warp2x4+stage3): ['41.75     ', '24.78125  '], time:10.33954ms, swizzle: NOOP, TFLOPS: 106.34(+1.00%)
          f16wmma(mma2x4+warp2x4+stage2): ['41.75     ', '24.78125  '], time:10.33453ms, swizzle: NOOP, TFLOPS: 106.39(+0.05%)
        f16wmma(mma2x4+...+stage3+dsmem): ['41.75     ', '24.78125  '], time:10.58974ms, swizzle: NOOP, TFLOPS: 103.83
        f16wmma(mma2x4+...+stage3+dsmem): ['41.75     ', '24.78125  '], time:10.66737ms, swizzle: NOOP, TFLOPS: 103.07
        f16wmma(mma2x4+...+stage2+dsmem): ['41.75     ', '24.78125  '], time:10.57136ms, swizzle: NOOP, TFLOPS: 104.01
        f16wmma(mma4x4+...+stage4+dsmem): ['41.75     ', '24.78125  '], time:10.49997ms, swizzle: NOOP, TFLOPS: 104.72
        f16wmma(mma4x4+...+stage3+dsmem): ['41.75     ', '24.78125  '], time:10.74283ms, swizzle: NOOP, TFLOPS: 102.35
        f16wmma(mma4x4+...+stage2+dsmem): ['41.75     ', '24.78125  '], time:11.02094ms, swizzle: NOOP, TFLOPS: 99.77
      f16wmma(mma2x4+...+stage4+swizzle): ['41.75     ', '24.78125  '], time:10.69972ms, swizzle: 4096, TFLOPS: 102.76
      f16wmma(mma2x4+...+stage3+swizzle): ['41.75     ', '24.78125  '], time:10.41080ms, swizzle: 4096, TFLOPS: 105.61
      f16wmma(mma2x4+...+stage2+swizzle): ['41.75     ', '24.78125  '], time:10.37557ms, swizzle: 4096, TFLOPS: 105.97
       f16wmma(...+stage3+dsmem+swizzle): ['41.75     ', '24.78125  '], time:10.46659ms, swizzle: 4096, TFLOPS: 105.05
       f16wmma(...+stage2+dsmem+swizzle): ['41.75     ', '24.78125  '], time:10.34002ms, swizzle: 4096, TFLOPS: 106.34
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['41.75     ', '24.78125  '], time:10.60650ms, swizzle: 4096, TFLOPS: 103.66
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['41.75     ', '24.78125  '], time:10.48388ms, swizzle: 4096, TFLOPS: 104.88
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['41.75     ', '24.78125  '], time:10.66639ms, swizzle: 4096, TFLOPS: 103.08
                             f16(cublas): ['41.75     ', '24.78125  '], time:9.852170ms, swizzle: NOOP, TFLOPS: 111.60(+4.90%)
                                  f16_th: ['41.65625  ', '24.765625 '], time:9.698629ms, swizzle: NOOP, TFLOPS: 113.37(+1.58%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=4096
                              f16(naive): ['70.0      ', '-2.0957031'], time:919.1768ms, swizzle: NOOP, TFLOPS: 2.39  (+0.00%)
                     f16x8pack(t8x8+bcf): ['70.0      ', '-2.0957031'], time:48.46169ms, swizzle: NOOP, TFLOPS: 45.38 (+1796.71%)
                f16x8pack(t8x8+bcf+dbuf): ['70.0      ', '-2.0957031'], time:50.90982ms, swizzle: NOOP, TFLOPS: 43.19
                f16x8pack(t8x8+k16+dbuf): ['70.0      ', '-2.0957031'], time:50.40857ms, swizzle: NOOP, TFLOPS: 43.62
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['69.5625   ', '-1.7373046'], time:152.3369ms, swizzle: NOOP, TFLOPS: 14.44
                         f16wmma(mma4x2): ['69.5625   ', '-1.7373046'], time:63.32268ms, swizzle: NOOP, TFLOPS: 34.73
                 f16wmma(mma4x2+warp2x4): ['69.5625   ', '-1.7373046'], time:28.39894ms, swizzle: NOOP, TFLOPS: 77.43 (+70.65%)
            f16wmma(mma2x4+warp2x4+dbuf): ['69.5625   ', '-1.7373046'], time:22.88904ms, swizzle: NOOP, TFLOPS: 96.07 (+24.07%)
          f16wmma(mma2x4+warp2x4+stage4): ['69.5625   ', '-1.7373046'], time:25.14255ms, swizzle: NOOP, TFLOPS: 87.46
          f16wmma(mma2x4+warp2x4+stage3): ['69.5625   ', '-1.7373046'], time:23.66592ms, swizzle: NOOP, TFLOPS: 92.92
          f16wmma(mma2x4+warp2x4+stage2): ['69.5625   ', '-1.7373046'], time:23.58512ms, swizzle: NOOP, TFLOPS: 93.24
        f16wmma(mma2x4+...+stage3+dsmem): ['69.5625   ', '-1.7373046'], time:25.00433ms, swizzle: NOOP, TFLOPS: 87.95
        f16wmma(mma2x4+...+stage3+dsmem): ['69.5625   ', '-1.7373046'], time:23.75319ms, swizzle: NOOP, TFLOPS: 92.58
        f16wmma(mma2x4+...+stage2+dsmem): ['69.5625   ', '-1.7373046'], time:23.90868ms, swizzle: NOOP, TFLOPS: 91.98
        f16wmma(mma4x4+...+stage4+dsmem): ['69.5625   ', '-1.7373046'], time:20.42579ms, swizzle: NOOP, TFLOPS: 107.66(+12.06%)
        f16wmma(mma4x4+...+stage3+dsmem): ['69.5625   ', '-1.7373046'], time:20.83826ms, swizzle: NOOP, TFLOPS: 105.53
        f16wmma(mma4x4+...+stage2+dsmem): ['69.5625   ', '-1.7373046'], time:21.56798ms, swizzle: NOOP, TFLOPS: 101.96
      f16wmma(mma2x4+...+stage4+swizzle): ['69.5625   ', '-1.7373046'], time:20.91116ms, swizzle: 4096, TFLOPS: 105.16
      f16wmma(mma2x4+...+stage3+swizzle): ['69.5625   ', '-1.7373046'], time:20.43797ms, swizzle: 4096, TFLOPS: 107.59
      f16wmma(mma2x4+...+stage2+swizzle): ['69.5625   ', '-1.7373046'], time:20.32136ms, swizzle: 4096, TFLOPS: 108.21(+0.51%)
       f16wmma(...+stage3+dsmem+swizzle): ['69.5625   ', '-1.7373046'], time:20.83003ms, swizzle: 4096, TFLOPS: 105.57
       f16wmma(...+stage2+dsmem+swizzle): ['69.5625   ', '-1.7373046'], time:20.19472ms, swizzle: 4096, TFLOPS: 108.89(+0.63%)
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['69.5625   ', '-1.7373046'], time:20.56219ms, swizzle: 4096, TFLOPS: 106.94
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['69.5625   ', '-1.7373046'], time:20.30205ms, swizzle: 4096, TFLOPS: 108.32
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['69.5625   ', '-1.7373046'], time:20.77736ms, swizzle: 4096, TFLOPS: 105.84
                             f16(cublas): ['69.5625   ', '-1.7373046'], time:19.34309ms, swizzle: NOOP, TFLOPS: 113.69(+4.40%)
                                  f16_th: ['69.8125   ', '-1.7314453'], time:19.11721ms, swizzle: NOOP, TFLOPS: 115.03(+1.18%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=8192
                              f16(naive): ['108.4375  ', '-75.375   '], time:1835.972ms, swizzle: NOOP, TFLOPS: 2.40  (+0.00%)
                     f16x8pack(t8x8+bcf): ['108.4375  ', '-75.375   '], time:105.7710ms, swizzle: NOOP, TFLOPS: 41.58 (+1635.80%)
                f16x8pack(t8x8+bcf+dbuf): ['108.4375  ', '-75.375   '], time:106.7285ms, swizzle: NOOP, TFLOPS: 41.21
                f16x8pack(t8x8+k16+dbuf): ['108.4375  ', '-75.375   '], time:105.1434ms, swizzle: NOOP, TFLOPS: 41.83 (+0.60%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['106.8125  ', '-74.875   '], time:823.6142ms, swizzle: NOOP, TFLOPS: 5.34
                         f16wmma(mma4x2): ['106.8125  ', '-74.875   '], time:124.7474ms, swizzle: NOOP, TFLOPS: 35.26
                 f16wmma(mma4x2+warp2x4): ['106.8125  ', '-74.875   '], time:56.26049ms, swizzle: NOOP, TFLOPS: 78.17 (+86.89%)
            f16wmma(mma2x4+warp2x4+dbuf): ['106.8125  ', '-74.875   '], time:50.75109ms, swizzle: NOOP, TFLOPS: 86.66 (+10.86%)
          f16wmma(mma2x4+warp2x4+stage4): ['106.8125  ', '-74.875   '], time:50.91655ms, swizzle: NOOP, TFLOPS: 86.38
          f16wmma(mma2x4+warp2x4+stage3): ['106.8125  ', '-74.875   '], time:50.46632ms, swizzle: NOOP, TFLOPS: 87.15 (+0.56%)
          f16wmma(mma2x4+warp2x4+stage2): ['106.8125  ', '-74.875   '], time:50.12691ms, swizzle: NOOP, TFLOPS: 87.74 (+0.68%)
        f16wmma(mma2x4+...+stage3+dsmem): ['106.8125  ', '-74.875   '], time:50.28378ms, swizzle: NOOP, TFLOPS: 87.46
        f16wmma(mma2x4+...+stage3+dsmem): ['106.8125  ', '-74.875   '], time:51.22525ms, swizzle: NOOP, TFLOPS: 85.86
        f16wmma(mma2x4+...+stage2+dsmem): ['106.8125  ', '-74.875   '], time:50.15027ms, swizzle: NOOP, TFLOPS: 87.70
        f16wmma(mma4x4+...+stage4+dsmem): ['106.8125  ', '-74.875   '], time:40.56966ms, swizzle: NOOP, TFLOPS: 108.41(+23.56%)
        f16wmma(mma4x4+...+stage3+dsmem): ['106.8125  ', '-74.875   '], time:41.65627ms, swizzle: NOOP, TFLOPS: 105.58
        f16wmma(mma4x4+...+stage2+dsmem): ['106.8125  ', '-74.875   '], time:42.43867ms, swizzle: NOOP, TFLOPS: 103.63
      f16wmma(mma2x4+...+stage4+swizzle): ['106.8125  ', '-74.875   '], time:41.84768ms, swizzle: 4096, TFLOPS: 105.10
      f16wmma(mma2x4+...+stage3+swizzle): ['106.8125  ', '-74.875   '], time:41.24093ms, swizzle: 4096, TFLOPS: 106.64
      f16wmma(mma2x4+...+stage2+swizzle): ['106.8125  ', '-74.875   '], time:40.60795ms, swizzle: 4096, TFLOPS: 108.31
       f16wmma(...+stage3+dsmem+swizzle): ['106.8125  ', '-74.875   '], time:41.68512ms, swizzle: 4096, TFLOPS: 105.51
       f16wmma(...+stage2+dsmem+swizzle): ['106.8125  ', '-74.875   '], time:40.60430ms, swizzle: 4096, TFLOPS: 108.31
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['106.8125  ', '-74.875   '], time:40.36123ms, swizzle: 4096, TFLOPS: 108.97(+0.52%)
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['106.8125  ', '-74.875   '], time:39.99798ms, swizzle: 4096, TFLOPS: 109.96(+0.91%)
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['106.8125  ', '-74.875   '], time:40.96848ms, swizzle: 4096, TFLOPS: 107.35
                             f16(cublas): ['106.8125  ', '-74.875   '], time:38.68441ms, swizzle: NOOP, TFLOPS: 113.69(+3.40%)
                                  f16_th: ['107.25    ', '-75.0     '], time:39.02029ms, swizzle: NOOP, TFLOPS: 112.71
----------------------------------------------------------------------------------------------------------------------------------
root@84d10136-98dc-41c0-a69b-8ff1f802b8cf:hgemm# python3 hgemm.py
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=2048
                              f16(naive): ['80.9375   ', '5.7734375 '], time:18.87300ms, swizzle: NOOP, TFLOPS: 3.64  (+0.00%)
                     f16x8pack(t8x8+bcf): ['80.9375   ', '5.7734375 '], time:1.420259ms, swizzle: NOOP, TFLOPS: 48.39 (+1228.84%)
                f16x8pack(t8x8+bcf+dbuf): ['80.9375   ', '5.7734375 '], time:1.402926ms, swizzle: NOOP, TFLOPS: 48.98 (+1.24%)
                f16x8pack(t8x8+k16+dbuf): ['80.9375   ', '5.7734375 '], time:1.326084ms, swizzle: NOOP, TFLOPS: 51.82 (+5.79%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['81.1875   ', '5.9296875 '], time:2.925968ms, swizzle: NOOP, TFLOPS: 23.49
                         f16wmma(mma4x2): ['81.1875   ', '5.9296875 '], time:1.487874ms, swizzle: NOOP, TFLOPS: 46.19
                 f16wmma(mma4x2+warp2x4): ['81.1875   ', '5.9296875 '], time:0.940656ms, swizzle: NOOP, TFLOPS: 73.05 (+40.97%)
            f16wmma(mma2x4+warp2x4+dbuf): ['81.1875   ', '5.9296875 '], time:0.700235ms, swizzle: NOOP, TFLOPS: 98.14 (+34.33%)
          f16wmma(mma2x4+warp2x4+stage4): ['81.1875   ', '5.9296875 '], time:0.706553ms, swizzle: NOOP, TFLOPS: 97.26
          f16wmma(mma2x4+warp2x4+stage3): ['81.1875   ', '5.9296875 '], time:0.696659ms, swizzle: NOOP, TFLOPS: 98.64 (+0.51%)
          f16wmma(mma2x4+warp2x4+stage2): ['81.1875   ', '5.9296875 '], time:0.697755ms, swizzle: NOOP, TFLOPS: 98.49
        f16wmma(mma2x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:0.707578ms, swizzle: NOOP, TFLOPS: 97.12
        f16wmma(mma2x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:0.701642ms, swizzle: NOOP, TFLOPS: 97.94
        f16wmma(mma2x4+...+stage2+dsmem): ['81.1875   ', '5.9296875 '], time:0.697278ms, swizzle: NOOP, TFLOPS: 98.55
        f16wmma(mma4x4+...+stage4+dsmem): ['81.1875   ', '5.9296875 '], time:0.717997ms, swizzle: NOOP, TFLOPS: 95.71
        f16wmma(mma4x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:0.732088ms, swizzle: NOOP, TFLOPS: 93.87
        f16wmma(mma4x4+...+stage2+dsmem): ['81.1875   ', '5.9296875 '], time:0.755906ms, swizzle: NOOP, TFLOPS: 90.91
      f16wmma(mma2x4+...+stage4+swizzle): ['81.1875   ', '5.9296875 '], time:0.710105ms, swizzle: 1024, TFLOPS: 96.77
      f16wmma(mma2x4+...+stage3+swizzle): ['81.1875   ', '5.9296875 '], time:0.694751ms, swizzle: 1024, TFLOPS: 98.91 (+0.27%)
      f16wmma(mma2x4+...+stage2+swizzle): ['81.1875   ', '5.9296875 '], time:0.687837ms, swizzle: 1024, TFLOPS: 99.91 (+1.01%)
       f16wmma(...+stage3+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:0.702643ms, swizzle: 1024, TFLOPS: 97.80
       f16wmma(...+stage2+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:0.686645ms, swizzle: 1024, TFLOPS: 100.08(+0.17%)
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:0.723195ms, swizzle: 1024, TFLOPS: 95.02
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:0.718116ms, swizzle: 1024, TFLOPS: 95.69
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:0.732922ms, swizzle: 1024, TFLOPS: 93.76
                             f16(cublas): ['81.1875   ', '5.9296875 '], time:0.850868ms, swizzle: NOOP, TFLOPS: 80.76
                                  f16_th: ['80.875    ', '5.9140625 '], time:0.661969ms, swizzle: NOOP, TFLOPS: 103.81(+3.73%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=4096
                              f16(naive): ['40.09375  ', '-9.3671875'], time:37.69044ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                     f16x8pack(t8x8+bcf): ['40.09375  ', '-9.3671875'], time:2.827191ms, swizzle: NOOP, TFLOPS: 48.61 (+1233.14%)
                f16x8pack(t8x8+bcf+dbuf): ['40.09375  ', '-9.3671875'], time:2.790236ms, swizzle: NOOP, TFLOPS: 49.26 (+1.32%)
                f16x8pack(t8x8+k16+dbuf): ['40.09375  ', '-9.3671875'], time:2.642393ms, swizzle: NOOP, TFLOPS: 52.01 (+5.60%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['40.53125  ', '-9.296875 '], time:5.846190ms, swizzle: NOOP, TFLOPS: 23.51
                         f16wmma(mma4x2): ['40.53125  ', '-9.296875 '], time:2.955460ms, swizzle: NOOP, TFLOPS: 46.50
                 f16wmma(mma4x2+warp2x4): ['40.53125  ', '-9.296875 '], time:1.824450ms, swizzle: NOOP, TFLOPS: 75.33 (+44.83%)
            f16wmma(mma2x4+warp2x4+dbuf): ['40.53125  ', '-9.296875 '], time:1.359963ms, swizzle: NOOP, TFLOPS: 101.06(+34.15%)
          f16wmma(mma2x4+warp2x4+stage4): ['40.53125  ', '-9.296875 '], time:1.365518ms, swizzle: NOOP, TFLOPS: 100.65
          f16wmma(mma2x4+warp2x4+stage3): ['40.53125  ', '-9.296875 '], time:1.353526ms, swizzle: NOOP, TFLOPS: 101.54(+0.48%)
          f16wmma(mma2x4+warp2x4+stage2): ['40.53125  ', '-9.296875 '], time:1.359820ms, swizzle: NOOP, TFLOPS: 101.07
        f16wmma(mma2x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:1.369905ms, swizzle: NOOP, TFLOPS: 100.33
        f16wmma(mma2x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:1.363921ms, swizzle: NOOP, TFLOPS: 100.77
        f16wmma(mma2x4+...+stage2+dsmem): ['40.53125  ', '-9.296875 '], time:1.357817ms, swizzle: NOOP, TFLOPS: 101.22
        f16wmma(mma4x4+...+stage4+dsmem): ['40.53125  ', '-9.296875 '], time:1.375889ms, swizzle: NOOP, TFLOPS: 99.89
        f16wmma(mma4x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:1.405382ms, swizzle: NOOP, TFLOPS: 97.79
        f16wmma(mma4x4+...+stage2+dsmem): ['40.53125  ', '-9.296875 '], time:1.455616ms, swizzle: NOOP, TFLOPS: 94.42
      f16wmma(mma2x4+...+stage4+swizzle): ['40.53125  ', '-9.296875 '], time:1.377558ms, swizzle: 1024, TFLOPS: 99.77
      f16wmma(mma2x4+...+stage3+swizzle): ['40.53125  ', '-9.296875 '], time:1.353383ms, swizzle: 1024, TFLOPS: 101.55(+0.01%)
      f16wmma(mma2x4+...+stage2+swizzle): ['40.53125  ', '-9.296875 '], time:1.335358ms, swizzle: 1024, TFLOPS: 102.92(+1.35%)
       f16wmma(...+stage3+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:1.368594ms, swizzle: 1024, TFLOPS: 100.42
       f16wmma(...+stage2+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:1.335358ms, swizzle: 1024, TFLOPS: 102.92
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:1.385593ms, swizzle: 1024, TFLOPS: 99.19
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:1.376056ms, swizzle: 1024, TFLOPS: 99.88
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:1.407885ms, swizzle: 1024, TFLOPS: 97.62
                             f16(cublas): ['40.53125  ', '-9.296875 '], time:1.462912ms, swizzle: NOOP, TFLOPS: 93.95
                                  f16_th: ['40.0625   ', '-9.2890625'], time:1.288652ms, swizzle: NOOP, TFLOPS: 106.65(+3.62%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=8192
                              f16(naive): ['80.125    ', '81.25     '], time:75.29690ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                     f16x8pack(t8x8+bcf): ['80.125    ', '81.25     '], time:5.812358ms, swizzle: NOOP, TFLOPS: 47.29 (+1195.46%)
                f16x8pack(t8x8+bcf+dbuf): ['80.125    ', '81.25     '], time:5.634522ms, swizzle: NOOP, TFLOPS: 48.78 (+3.16%)
                f16x8pack(t8x8+k16+dbuf): ['80.125    ', '81.25     '], time:5.341577ms, swizzle: NOOP, TFLOPS: 51.46 (+5.48%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['80.4375   ', '81.8125   '], time:11.61527ms, swizzle: NOOP, TFLOPS: 23.67
                         f16wmma(mma4x2): ['80.4375   ', '81.8125   '], time:5.901408ms, swizzle: NOOP, TFLOPS: 46.58
                 f16wmma(mma4x2+warp2x4): ['80.4375   ', '81.8125   '], time:3.632378ms, swizzle: NOOP, TFLOPS: 75.67 (+47.05%)
            f16wmma(mma2x4+warp2x4+dbuf): ['80.4375   ', '81.8125   '], time:2.680444ms, swizzle: NOOP, TFLOPS: 102.55(+35.51%)
          f16wmma(mma2x4+warp2x4+stage4): ['80.4375   ', '81.8125   '], time:2.690529ms, swizzle: NOOP, TFLOPS: 102.16
          f16wmma(mma2x4+warp2x4+stage3): ['80.4375   ', '81.8125   '], time:2.659583ms, swizzle: NOOP, TFLOPS: 103.35(+0.78%)
          f16wmma(mma2x4+warp2x4+stage2): ['80.4375   ', '81.8125   '], time:2.672839ms, swizzle: NOOP, TFLOPS: 102.84
        f16wmma(mma2x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:2.685451ms, swizzle: NOOP, TFLOPS: 102.36
        f16wmma(mma2x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:2.681398ms, swizzle: NOOP, TFLOPS: 102.51
        f16wmma(mma2x4+...+stage2+dsmem): ['80.4375   ', '81.8125   '], time:2.669525ms, swizzle: NOOP, TFLOPS: 102.97
        f16wmma(mma4x4+...+stage4+dsmem): ['80.4375   ', '81.8125   '], time:2.687597ms, swizzle: NOOP, TFLOPS: 102.28
        f16wmma(mma4x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:2.744531ms, swizzle: NOOP, TFLOPS: 100.15
        f16wmma(mma4x4+...+stage2+dsmem): ['80.4375   ', '81.8125   '], time:2.846717ms, swizzle: NOOP, TFLOPS: 96.56
      f16wmma(mma2x4+...+stage4+swizzle): ['80.4375   ', '81.8125   '], time:2.705287ms, swizzle: 1024, TFLOPS: 101.61
      f16wmma(mma2x4+...+stage3+swizzle): ['80.4375   ', '81.8125   '], time:2.663922ms, swizzle: 1024, TFLOPS: 103.19
      f16wmma(mma2x4+...+stage2+swizzle): ['80.4375   ', '81.8125   '], time:2.627706ms, swizzle: 1024, TFLOPS: 104.61(+1.21%)
       f16wmma(...+stage3+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:2.694034ms, swizzle: 1024, TFLOPS: 102.03
       f16wmma(...+stage2+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:2.628540ms, swizzle: 1024, TFLOPS: 104.57
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:2.704906ms, swizzle: 1024, TFLOPS: 101.62
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:2.686285ms, swizzle: 1024, TFLOPS: 102.33
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:2.751755ms, swizzle: 1024, TFLOPS: 99.89
                             f16(cublas): ['80.125    ', '81.8125   '], time:2.625799ms, swizzle: NOOP, TFLOPS: 104.68(+0.07%)
                                  f16_th: ['79.875    ', '81.8125   '], time:2.399659ms, swizzle: NOOP, TFLOPS: 114.55(+9.42%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=2048
                              f16(naive): ['80.9375   ', '5.7734375 '], time:37.66982ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                     f16x8pack(t8x8+bcf): ['80.9375   ', '5.7734375 '], time:2.710223ms, swizzle: NOOP, TFLOPS: 50.71 (+1289.92%)
                f16x8pack(t8x8+bcf+dbuf): ['80.9375   ', '5.7734375 '], time:2.662611ms, swizzle: NOOP, TFLOPS: 51.62 (+1.79%)
                f16x8pack(t8x8+k16+dbuf): ['80.9375   ', '5.7734375 '], time:2.523016ms, swizzle: NOOP, TFLOPS: 54.47 (+5.53%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['81.1875   ', '5.9296875 '], time:5.819272ms, swizzle: NOOP, TFLOPS: 23.62
                         f16wmma(mma4x2): ['81.1875   ', '5.9296875 '], time:2.946019ms, swizzle: NOOP, TFLOPS: 46.65
                 f16wmma(mma4x2+warp2x4): ['81.1875   ', '5.9296875 '], time:1.829028ms, swizzle: NOOP, TFLOPS: 75.14 (+37.94%)
            f16wmma(mma2x4+warp2x4+dbuf): ['81.1875   ', '5.9296875 '], time:1.328682ms, swizzle: NOOP, TFLOPS: 103.44(+37.66%)
          f16wmma(mma2x4+warp2x4+stage4): ['81.1875   ', '5.9296875 '], time:1.334452ms, swizzle: NOOP, TFLOPS: 102.99
          f16wmma(mma2x4+warp2x4+stage3): ['81.1875   ', '5.9296875 '], time:1.308894ms, swizzle: NOOP, TFLOPS: 105.00(+1.51%)
          f16wmma(mma2x4+warp2x4+stage2): ['81.1875   ', '5.9296875 '], time:1.315855ms, swizzle: NOOP, TFLOPS: 104.45
        f16wmma(mma2x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:1.332449ms, swizzle: NOOP, TFLOPS: 103.15
        f16wmma(mma2x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:1.318407ms, swizzle: NOOP, TFLOPS: 104.25
        f16wmma(mma2x4+...+stage2+dsmem): ['81.1875   ', '5.9296875 '], time:1.312303ms, swizzle: NOOP, TFLOPS: 104.73
        f16wmma(mma4x4+...+stage4+dsmem): ['81.1875   ', '5.9296875 '], time:1.417040ms, swizzle: NOOP, TFLOPS: 96.99
        f16wmma(mma4x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:1.446652ms, swizzle: NOOP, TFLOPS: 95.00
        f16wmma(mma4x4+...+stage2+dsmem): ['81.1875   ', '5.9296875 '], time:1.492500ms, swizzle: NOOP, TFLOPS: 92.09
      f16wmma(mma2x4+...+stage4+swizzle): ['81.1875   ', '5.9296875 '], time:1.341366ms, swizzle: 2048, TFLOPS: 102.46
      f16wmma(mma2x4+...+stage3+swizzle): ['81.1875   ', '5.9296875 '], time:1.305580ms, swizzle: 2048, TFLOPS: 105.27(+0.25%)
      f16wmma(mma2x4+...+stage2+swizzle): ['81.1875   ', '5.9296875 '], time:1.295685ms, swizzle: 2048, TFLOPS: 106.07(+0.76%)
       f16wmma(...+stage3+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:1.318740ms, swizzle: 2048, TFLOPS: 104.22
       f16wmma(...+stage2+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:1.295995ms, swizzle: 2048, TFLOPS: 106.05
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:1.427912ms, swizzle: 2048, TFLOPS: 96.25
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:1.419377ms, swizzle: 2048, TFLOPS: 96.83
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:1.447200ms, swizzle: 2048, TFLOPS: 94.97
                             f16(cublas): ['81.1875   ', '5.9296875 '], time:1.416730ms, swizzle: NOOP, TFLOPS: 97.01
                                  f16_th: ['80.875    ', '5.9140625 '], time:1.300287ms, swizzle: NOOP, TFLOPS: 105.70
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=4096
                              f16(naive): ['40.09375  ', '-9.3671875'], time:75.23338ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                     f16x8pack(t8x8+bcf): ['40.09375  ', '-9.3671875'], time:5.409145ms, swizzle: NOOP, TFLOPS: 50.82 (+1290.86%)
                f16x8pack(t8x8+bcf+dbuf): ['40.09375  ', '-9.3671875'], time:5.398583ms, swizzle: NOOP, TFLOPS: 50.92 (+0.20%)
                f16x8pack(t8x8+k16+dbuf): ['40.09375  ', '-9.3671875'], time:5.169630ms, swizzle: NOOP, TFLOPS: 53.17 (+4.43%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['40.53125  ', '-9.296875 '], time:11.64746ms, swizzle: NOOP, TFLOPS: 23.60
                         f16wmma(mma4x2): ['40.53125  ', '-9.296875 '], time:5.885386ms, swizzle: NOOP, TFLOPS: 46.71
                 f16wmma(mma4x2+warp2x4): ['40.53125  ', '-9.296875 '], time:3.624153ms, swizzle: NOOP, TFLOPS: 75.85 (+42.64%)
            f16wmma(mma2x4+warp2x4+dbuf): ['40.53125  ', '-9.296875 '], time:2.594971ms, swizzle: NOOP, TFLOPS: 105.93(+39.66%)
          f16wmma(mma2x4+warp2x4+stage4): ['40.53125  ', '-9.296875 '], time:2.603292ms, swizzle: NOOP, TFLOPS: 105.59
          f16wmma(mma2x4+warp2x4+stage3): ['40.53125  ', '-9.296875 '], time:2.564644ms, swizzle: NOOP, TFLOPS: 107.18(+1.18%)
          f16wmma(mma2x4+warp2x4+stage2): ['40.53125  ', '-9.296875 '], time:2.574896ms, swizzle: NOOP, TFLOPS: 106.75
        f16wmma(mma2x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:2.595067ms, swizzle: NOOP, TFLOPS: 105.92
        f16wmma(mma2x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:2.582883ms, swizzle: NOOP, TFLOPS: 106.42
        f16wmma(mma2x4+...+stage2+dsmem): ['40.53125  ', '-9.296875 '], time:2.571034ms, swizzle: NOOP, TFLOPS: 106.91
        f16wmma(mma4x4+...+stage4+dsmem): ['40.53125  ', '-9.296875 '], time:2.732205ms, swizzle: NOOP, TFLOPS: 100.61
        f16wmma(mma4x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:2.787137ms, swizzle: NOOP, TFLOPS: 98.62
        f16wmma(mma4x4+...+stage2+dsmem): ['40.53125  ', '-9.296875 '], time:2.889394ms, swizzle: NOOP, TFLOPS: 95.13
      f16wmma(mma2x4+...+stage4+swizzle): ['40.53125  ', '-9.296875 '], time:2.616453ms, swizzle: 2048, TFLOPS: 105.06
      f16wmma(mma2x4+...+stage3+swizzle): ['40.53125  ', '-9.296875 '], time:2.562308ms, swizzle: 2048, TFLOPS: 107.28(+0.09%)
      f16wmma(mma2x4+...+stage2+swizzle): ['40.53125  ', '-9.296875 '], time:2.536916ms, swizzle: 2048, TFLOPS: 108.35(+1.00%)
       f16wmma(...+stage3+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:2.587962ms, swizzle: 2048, TFLOPS: 106.21
       f16wmma(...+stage2+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:2.538180ms, swizzle: 2048, TFLOPS: 108.30
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:2.750611ms, swizzle: 2048, TFLOPS: 99.93
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:2.732849ms, swizzle: 2048, TFLOPS: 100.58
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:2.794599ms, swizzle: 2048, TFLOPS: 98.36
                             f16(cublas): ['40.53125  ', '-9.296875 '], time:3.245925ms, swizzle: NOOP, TFLOPS: 84.68
                                  f16_th: ['40.0625   ', '-9.2890625'], time:2.551722ms, swizzle: NOOP, TFLOPS: 107.72
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=8192
                              f16(naive): ['80.125    ', '81.25     '], time:222.0335ms, swizzle: NOOP, TFLOPS: 2.48  (+0.00%)
                     f16x8pack(t8x8+bcf): ['80.125    ', '81.25     '], time:12.09609ms, swizzle: NOOP, TFLOPS: 45.45 (+1735.58%)
                f16x8pack(t8x8+bcf+dbuf): ['80.125    ', '81.25     '], time:11.73336ms, swizzle: NOOP, TFLOPS: 46.85 (+3.09%)
                f16x8pack(t8x8+k16+dbuf): ['80.125    ', '81.25     '], time:11.45088ms, swizzle: NOOP, TFLOPS: 48.01 (+2.47%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['80.4375   ', '81.8125   '], time:24.10671ms, swizzle: NOOP, TFLOPS: 22.81
                         f16wmma(mma4x2): ['80.4375   ', '81.8125   '], time:12.31992ms, swizzle: NOOP, TFLOPS: 44.62
                 f16wmma(mma4x2+warp2x4): ['80.4375   ', '81.8125   '], time:7.307672ms, swizzle: NOOP, TFLOPS: 75.23 (+56.70%)
            f16wmma(mma2x4+warp2x4+dbuf): ['80.4375   ', '81.8125   '], time:5.338072ms, swizzle: NOOP, TFLOPS: 102.99(+36.90%)
          f16wmma(mma2x4+warp2x4+stage4): ['80.4375   ', '81.8125   '], time:5.337977ms, swizzle: NOOP, TFLOPS: 102.99(+0.00%)
          f16wmma(mma2x4+warp2x4+stage3): ['80.4375   ', '81.8125   '], time:5.345630ms, swizzle: NOOP, TFLOPS: 102.84
          f16wmma(mma2x4+warp2x4+stage2): ['80.4375   ', '81.8125   '], time:5.317664ms, swizzle: NOOP, TFLOPS: 103.38(+0.38%)
        f16wmma(mma2x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:5.308198ms, swizzle: NOOP, TFLOPS: 103.57(+0.18%)
        f16wmma(mma2x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:5.315828ms, swizzle: NOOP, TFLOPS: 103.42
        f16wmma(mma2x4+...+stage2+dsmem): ['80.4375   ', '81.8125   '], time:5.390524ms, swizzle: NOOP, TFLOPS: 101.99
        f16wmma(mma4x4+...+stage4+dsmem): ['80.4375   ', '81.8125   '], time:5.447459ms, swizzle: NOOP, TFLOPS: 100.92
        f16wmma(mma4x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:5.499100ms, swizzle: NOOP, TFLOPS: 99.97
        f16wmma(mma4x4+...+stage2+dsmem): ['80.4375   ', '81.8125   '], time:5.708026ms, swizzle: NOOP, TFLOPS: 96.31
      f16wmma(mma2x4+...+stage4+swizzle): ['80.4375   ', '81.8125   '], time:5.176067ms, swizzle: 2048, TFLOPS: 106.21(+2.55%)
      f16wmma(mma2x4+...+stage3+swizzle): ['80.4375   ', '81.8125   '], time:5.103492ms, swizzle: 2048, TFLOPS: 107.72(+1.42%)
      f16wmma(mma2x4+...+stage2+swizzle): ['80.4375   ', '81.8125   '], time:5.038762ms, swizzle: 2048, TFLOPS: 109.11(+1.28%)
       f16wmma(...+stage3+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:5.140471ms, swizzle: 2048, TFLOPS: 106.95
       f16wmma(...+stage2+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:5.040121ms, swizzle: 2048, TFLOPS: 109.08
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:5.422472ms, swizzle: 2048, TFLOPS: 101.38
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:5.388784ms, swizzle: 2048, TFLOPS: 102.02
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:5.483460ms, swizzle: 2048, TFLOPS: 100.26
                             f16(cublas): ['80.4375   ', '81.8125   '], time:5.020642ms, swizzle: NOOP, TFLOPS: 109.50(+0.36%)
                                  f16_th: ['79.875    ', '81.875    '], time:4.912948ms, swizzle: NOOP, TFLOPS: 111.90(+2.19%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=2048
                              f16(naive): ['80.9375   ', '5.7734375 '], time:75.27763ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                     f16x8pack(t8x8+bcf): ['80.9375   ', '5.7734375 '], time:5.416202ms, swizzle: NOOP, TFLOPS: 50.75 (+1289.86%)
                f16x8pack(t8x8+bcf+dbuf): ['80.9375   ', '5.7734375 '], time:5.333566ms, swizzle: NOOP, TFLOPS: 51.54 (+1.55%)
                f16x8pack(t8x8+k16+dbuf): ['80.9375   ', '5.7734375 '], time:5.019426ms, swizzle: NOOP, TFLOPS: 54.76 (+6.26%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['81.1875   ', '5.9296875 '], time:11.65666ms, swizzle: NOOP, TFLOPS: 23.58
                         f16wmma(mma4x2): ['81.1875   ', '5.9296875 '], time:5.896329ms, swizzle: NOOP, TFLOPS: 46.62
                 f16wmma(mma4x2+warp2x4): ['81.1875   ', '5.9296875 '], time:3.528523ms, swizzle: NOOP, TFLOPS: 77.90 (+42.25%)
            f16wmma(mma2x4+warp2x4+dbuf): ['81.1875   ', '5.9296875 '], time:2.582454ms, swizzle: NOOP, TFLOPS: 106.44(+36.63%)
          f16wmma(mma2x4+warp2x4+stage4): ['81.1875   ', '5.9296875 '], time:2.593946ms, swizzle: NOOP, TFLOPS: 105.97
          f16wmma(mma2x4+warp2x4+stage3): ['81.1875   ', '5.9296875 '], time:2.538132ms, swizzle: NOOP, TFLOPS: 108.30(+1.75%)
          f16wmma(mma2x4+warp2x4+stage2): ['81.1875   ', '5.9296875 '], time:2.551817ms, swizzle: NOOP, TFLOPS: 107.72
        f16wmma(mma2x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:2.586984ms, swizzle: NOOP, TFLOPS: 106.25
        f16wmma(mma2x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:2.554607ms, swizzle: NOOP, TFLOPS: 107.60
        f16wmma(mma2x4+...+stage2+dsmem): ['81.1875   ', '5.9296875 '], time:2.546596ms, swizzle: NOOP, TFLOPS: 107.94
        f16wmma(mma4x4+...+stage4+dsmem): ['81.1875   ', '5.9296875 '], time:2.794003ms, swizzle: NOOP, TFLOPS: 98.38
        f16wmma(mma4x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:2.857542ms, swizzle: NOOP, TFLOPS: 96.19
        f16wmma(mma4x4+...+stage2+dsmem): ['81.1875   ', '5.9296875 '], time:2.947998ms, swizzle: NOOP, TFLOPS: 93.24
      f16wmma(mma2x4+...+stage4+swizzle): ['81.1875   ', '5.9296875 '], time:2.611708ms, swizzle: 4096, TFLOPS: 105.25
      f16wmma(mma2x4+...+stage3+swizzle): ['81.1875   ', '5.9296875 '], time:2.533745ms, swizzle: 4096, TFLOPS: 108.49(+0.17%)
      f16wmma(mma2x4+...+stage2+swizzle): ['81.1875   ', '5.9296875 '], time:2.516555ms, swizzle: 4096, TFLOPS: 109.23(+0.68%)
       f16wmma(...+stage3+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:2.558112ms, swizzle: 4096, TFLOPS: 107.45
       f16wmma(...+stage2+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:2.516436ms, swizzle: 4096, TFLOPS: 109.23(+0.00%)
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:2.823567ms, swizzle: 4096, TFLOPS: 97.35
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:2.800917ms, swizzle: 4096, TFLOPS: 98.14
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:2.854800ms, swizzle: 4096, TFLOPS: 96.29
                             f16(cublas): ['81.1875   ', '5.9296875 '], time:2.578425ms, swizzle: NOOP, TFLOPS: 106.61
                                  f16_th: ['80.875    ', '5.9140625 '], time:2.394032ms, swizzle: NOOP, TFLOPS: 114.82(+5.11%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=4096
                              f16(naive): ['40.09375  ', '-9.3671875'], time:229.5666ms, swizzle: NOOP, TFLOPS: 2.39  (+0.00%)
                     f16x8pack(t8x8+bcf): ['40.09375  ', '-9.3671875'], time:12.06429ms, swizzle: NOOP, TFLOPS: 45.57 (+1802.86%)
                f16x8pack(t8x8+bcf+dbuf): ['40.09375  ', '-9.3671875'], time:12.02869ms, swizzle: NOOP, TFLOPS: 45.70 (+0.30%)
                f16x8pack(t8x8+k16+dbuf): ['40.09375  ', '-9.3671875'], time:11.51549ms, swizzle: NOOP, TFLOPS: 47.74 (+4.46%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['40.53125  ', '-9.296875 '], time:37.69237ms, swizzle: NOOP, TFLOPS: 14.59
                         f16wmma(mma4x2): ['40.53125  ', '-9.296875 '], time:15.53285ms, swizzle: NOOP, TFLOPS: 35.39
                 f16wmma(mma4x2+warp2x4): ['40.53125  ', '-9.296875 '], time:7.263112ms, swizzle: NOOP, TFLOPS: 75.69 (+58.55%)
            f16wmma(mma2x4+warp2x4+dbuf): ['40.53125  ', '-9.296875 '], time:5.184030ms, swizzle: NOOP, TFLOPS: 106.05(+40.11%)
          f16wmma(mma2x4+warp2x4+stage4): ['40.53125  ', '-9.296875 '], time:5.358004ms, swizzle: NOOP, TFLOPS: 102.60
          f16wmma(mma2x4+warp2x4+stage3): ['40.53125  ', '-9.296875 '], time:5.296397ms, swizzle: NOOP, TFLOPS: 103.80
          f16wmma(mma2x4+warp2x4+stage2): ['40.53125  ', '-9.296875 '], time:5.298280ms, swizzle: NOOP, TFLOPS: 103.76
        f16wmma(mma2x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:5.421686ms, swizzle: NOOP, TFLOPS: 101.40
        f16wmma(mma2x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:5.360198ms, swizzle: NOOP, TFLOPS: 102.56
        f16wmma(mma2x4+...+stage2+dsmem): ['40.53125  ', '-9.296875 '], time:5.253720ms, swizzle: NOOP, TFLOPS: 104.64
        f16wmma(mma4x4+...+stage4+dsmem): ['40.53125  ', '-9.296875 '], time:5.464339ms, swizzle: NOOP, TFLOPS: 100.61
        f16wmma(mma4x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:5.572032ms, swizzle: NOOP, TFLOPS: 98.66
        f16wmma(mma4x4+...+stage2+dsmem): ['40.53125  ', '-9.296875 '], time:5.793857ms, swizzle: NOOP, TFLOPS: 94.89
      f16wmma(mma2x4+...+stage4+swizzle): ['40.53125  ', '-9.296875 '], time:5.105924ms, swizzle: 4096, TFLOPS: 107.67(+1.53%)
      f16wmma(mma2x4+...+stage3+swizzle): ['40.53125  ', '-9.296875 '], time:5.016398ms, swizzle: 4096, TFLOPS: 109.59(+1.78%)
      f16wmma(mma2x4+...+stage2+swizzle): ['40.53125  ', '-9.296875 '], time:4.971027ms, swizzle: 4096, TFLOPS: 110.59(+0.91%)
       f16wmma(...+stage3+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:5.054545ms, swizzle: 4096, TFLOPS: 108.76
       f16wmma(...+stage2+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:4.962754ms, swizzle: 4096, TFLOPS: 110.78(+0.17%)
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:5.500364ms, swizzle: 4096, TFLOPS: 99.95
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:5.463171ms, swizzle: 4096, TFLOPS: 100.63
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:5.578231ms, swizzle: 4096, TFLOPS: 98.55
                             f16(cublas): ['40.53125  ', '-9.296875 '], time:4.990983ms, swizzle: NOOP, TFLOPS: 110.15
                                  f16_th: ['40.0625   ', '-9.2890625'], time:4.876756ms, swizzle: NOOP, TFLOPS: 112.73(+1.76%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=8192
                              f16(naive): ['80.125    ', '81.25     '], time:458.4778ms, swizzle: NOOP, TFLOPS: 2.40  (+0.00%)
                     f16x8pack(t8x8+bcf): ['80.125    ', '81.25     '], time:24.92465ms, swizzle: NOOP, TFLOPS: 44.11 (+1739.45%)
                f16x8pack(t8x8+bcf+dbuf): ['80.125    ', '81.25     '], time:25.05929ms, swizzle: NOOP, TFLOPS: 43.88
                f16x8pack(t8x8+k16+dbuf): ['80.125    ', '81.25     '], time:24.50480ms, swizzle: NOOP, TFLOPS: 44.87 (+1.71%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['80.4375   ', '81.8125   '], time:198.9140ms, swizzle: NOOP, TFLOPS: 5.53
                         f16wmma(mma4x2): ['80.4375   ', '81.8125   '], time:30.96418ms, swizzle: NOOP, TFLOPS: 35.51
                 f16wmma(mma4x2+warp2x4): ['80.4375   ', '81.8125   '], time:14.35799ms, swizzle: NOOP, TFLOPS: 76.58 (+70.67%)
            f16wmma(mma2x4+warp2x4+dbuf): ['80.4375   ', '81.8125   '], time:12.56763ms, swizzle: NOOP, TFLOPS: 87.49 (+14.25%)
          f16wmma(mma2x4+warp2x4+stage4): ['80.4375   ', '81.8125   '], time:12.43705ms, swizzle: NOOP, TFLOPS: 88.41 (+1.05%)
          f16wmma(mma2x4+warp2x4+stage3): ['80.4375   ', '81.8125   '], time:12.45658ms, swizzle: NOOP, TFLOPS: 88.27
          f16wmma(mma2x4+warp2x4+stage2): ['80.4375   ', '81.8125   '], time:12.36305ms, swizzle: NOOP, TFLOPS: 88.94 (+0.60%)
        f16wmma(mma2x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:12.44227ms, swizzle: NOOP, TFLOPS: 88.37
        f16wmma(mma2x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:12.45524ms, swizzle: NOOP, TFLOPS: 88.28
        f16wmma(mma2x4+...+stage2+dsmem): ['80.4375   ', '81.8125   '], time:12.37652ms, swizzle: NOOP, TFLOPS: 88.84
        f16wmma(mma4x4+...+stage4+dsmem): ['80.4375   ', '81.8125   '], time:10.74585ms, swizzle: NOOP, TFLOPS: 102.32(+15.05%)
        f16wmma(mma4x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:10.94403ms, swizzle: NOOP, TFLOPS: 100.47
        f16wmma(mma4x4+...+stage2+dsmem): ['80.4375   ', '81.8125   '], time:11.31646ms, swizzle: NOOP, TFLOPS: 97.16
      f16wmma(mma2x4+...+stage4+swizzle): ['80.4375   ', '81.8125   '], time:10.22846ms, swizzle: 4096, TFLOPS: 107.50(+5.06%)
      f16wmma(mma2x4+...+stage3+swizzle): ['80.4375   ', '81.8125   '], time:10.05773ms, swizzle: 4096, TFLOPS: 109.32(+1.70%)
      f16wmma(mma2x4+...+stage2+swizzle): ['80.4375   ', '81.8125   '], time:10.00795ms, swizzle: 4096, TFLOPS: 109.86(+0.50%)
       f16wmma(...+stage3+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:10.22062ms, swizzle: 4096, TFLOPS: 107.58
       f16wmma(...+stage2+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:10.06367ms, swizzle: 4096, TFLOPS: 109.26
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:10.81349ms, swizzle: 4096, TFLOPS: 101.68
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:10.69638ms, swizzle: 4096, TFLOPS: 102.79
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:10.91887ms, swizzle: 4096, TFLOPS: 100.70
                             f16(cublas): ['80.4375   ', '81.8125   '], time:9.731626ms, swizzle: NOOP, TFLOPS: 112.98(+2.84%)
                                  f16_th: ['79.875    ', '81.875    '], time:9.737706ms, swizzle: NOOP, TFLOPS: 112.91
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=2048
                              f16(naive): ['80.9375   ', '5.7734375 '], time:37.67905ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                     f16x8pack(t8x8+bcf): ['80.9375   ', '5.7734375 '], time:2.731585ms, swizzle: NOOP, TFLOPS: 50.31 (+1279.38%)
                f16x8pack(t8x8+bcf+dbuf): ['80.9375   ', '5.7734375 '], time:2.710795ms, swizzle: NOOP, TFLOPS: 50.70 (+0.77%)
                f16x8pack(t8x8+k16+dbuf): ['80.9375   ', '5.7734375 '], time:2.539300ms, swizzle: NOOP, TFLOPS: 54.12 (+6.75%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['81.1875   ', '5.9296875 '], time:5.853652ms, swizzle: NOOP, TFLOPS: 23.48
                         f16wmma(mma4x2): ['81.1875   ', '5.9296875 '], time:2.964735ms, swizzle: NOOP, TFLOPS: 46.36
                 f16wmma(mma4x2+warp2x4): ['81.1875   ', '5.9296875 '], time:1.834726ms, swizzle: NOOP, TFLOPS: 74.91 (+38.40%)
            f16wmma(mma2x4+warp2x4+dbuf): ['81.1875   ', '5.9296875 '], time:1.331329ms, swizzle: NOOP, TFLOPS: 103.23(+37.81%)
          f16wmma(mma2x4+warp2x4+stage4): ['81.1875   ', '5.9296875 '], time:1.335597ms, swizzle: NOOP, TFLOPS: 102.90
          f16wmma(mma2x4+warp2x4+stage3): ['81.1875   ', '5.9296875 '], time:1.310873ms, swizzle: NOOP, TFLOPS: 104.85(+1.56%)
          f16wmma(mma2x4+warp2x4+stage2): ['81.1875   ', '5.9296875 '], time:1.316809ms, swizzle: NOOP, TFLOPS: 104.37
        f16wmma(mma2x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:1.333117ms, swizzle: NOOP, TFLOPS: 103.10
        f16wmma(mma2x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:1.319622ms, swizzle: NOOP, TFLOPS: 104.15
        f16wmma(mma2x4+...+stage2+dsmem): ['81.1875   ', '5.9296875 '], time:1.314854ms, swizzle: NOOP, TFLOPS: 104.53
        f16wmma(mma4x4+...+stage4+dsmem): ['81.1875   ', '5.9296875 '], time:1.419210ms, swizzle: NOOP, TFLOPS: 96.84
        f16wmma(mma4x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:1.450371ms, swizzle: NOOP, TFLOPS: 94.76
        f16wmma(mma4x4+...+stage2+dsmem): ['81.1875   ', '5.9296875 '], time:1.491141ms, swizzle: NOOP, TFLOPS: 92.17
      f16wmma(mma2x4+...+stage4+swizzle): ['81.1875   ', '5.9296875 '], time:1.343202ms, swizzle: 1024, TFLOPS: 102.32
      f16wmma(mma2x4+...+stage3+swizzle): ['81.1875   ', '5.9296875 '], time:1.307392ms, swizzle: 1024, TFLOPS: 105.12(+0.27%)
      f16wmma(mma2x4+...+stage2+swizzle): ['81.1875   ', '5.9296875 '], time:1.296067ms, swizzle: 1024, TFLOPS: 106.04(+0.87%)
       f16wmma(...+stage3+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:1.319885ms, swizzle: 1024, TFLOPS: 104.13
       f16wmma(...+stage2+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:1.296615ms, swizzle: 1024, TFLOPS: 106.00
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:1.427412ms, swizzle: 1024, TFLOPS: 96.29
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:1.419162ms, swizzle: 1024, TFLOPS: 96.85
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:1.449465ms, swizzle: 1024, TFLOPS: 94.82
                             f16(cublas): ['81.1875   ', '5.9296875 '], time:1.493287ms, swizzle: NOOP, TFLOPS: 92.04
                                  f16_th: ['80.875    ', '5.9140625 '], time:1.303124ms, swizzle: NOOP, TFLOPS: 105.47
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=4096
                              f16(naive): ['40.09375  ', '-9.3671875'], time:75.24681ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                     f16x8pack(t8x8+bcf): ['40.09375  ', '-9.3671875'], time:5.596494ms, swizzle: NOOP, TFLOPS: 49.12 (+1244.53%)
                f16x8pack(t8x8+bcf+dbuf): ['40.09375  ', '-9.3671875'], time:5.371403ms, swizzle: NOOP, TFLOPS: 51.17 (+4.19%)
                f16x8pack(t8x8+k16+dbuf): ['40.09375  ', '-9.3671875'], time:5.086922ms, swizzle: NOOP, TFLOPS: 54.04 (+5.59%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['40.53125  ', '-9.296875 '], time:11.63885ms, swizzle: NOOP, TFLOPS: 23.62
                         f16wmma(mma4x2): ['40.53125  ', '-9.296875 '], time:5.880379ms, swizzle: NOOP, TFLOPS: 46.74
                 f16wmma(mma4x2+warp2x4): ['40.53125  ', '-9.296875 '], time:3.595781ms, swizzle: NOOP, TFLOPS: 76.44 (+41.47%)
            f16wmma(mma2x4+warp2x4+dbuf): ['40.53125  ', '-9.296875 '], time:2.591347ms, swizzle: NOOP, TFLOPS: 106.08(+38.76%)
          f16wmma(mma2x4+warp2x4+stage4): ['40.53125  ', '-9.296875 '], time:2.598214ms, swizzle: NOOP, TFLOPS: 105.79
          f16wmma(mma2x4+warp2x4+stage3): ['40.53125  ', '-9.296875 '], time:2.558684ms, swizzle: NOOP, TFLOPS: 107.43(+1.28%)
          f16wmma(mma2x4+warp2x4+stage2): ['40.53125  ', '-9.296875 '], time:2.569556ms, swizzle: NOOP, TFLOPS: 106.97
        f16wmma(mma2x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:2.588295ms, swizzle: NOOP, TFLOPS: 106.20
        f16wmma(mma2x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:2.577710ms, swizzle: NOOP, TFLOPS: 106.64
        f16wmma(mma2x4+...+stage2+dsmem): ['40.53125  ', '-9.296875 '], time:2.566719ms, swizzle: NOOP, TFLOPS: 107.09
        f16wmma(mma4x4+...+stage4+dsmem): ['40.53125  ', '-9.296875 '], time:2.727508ms, swizzle: NOOP, TFLOPS: 100.78
        f16wmma(mma4x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:2.783536ms, swizzle: NOOP, TFLOPS: 98.75
        f16wmma(mma4x4+...+stage2+dsmem): ['40.53125  ', '-9.296875 '], time:2.887654ms, swizzle: NOOP, TFLOPS: 95.19
      f16wmma(mma2x4+...+stage4+swizzle): ['40.53125  ', '-9.296875 '], time:2.611708ms, swizzle: 1024, TFLOPS: 105.25
      f16wmma(mma2x4+...+stage3+swizzle): ['40.53125  ', '-9.296875 '], time:2.557611ms, swizzle: 1024, TFLOPS: 107.47(+0.04%)
      f16wmma(mma2x4+...+stage2+swizzle): ['40.53125  ', '-9.296875 '], time:2.532625ms, swizzle: 1024, TFLOPS: 108.53(+0.99%)
       f16wmma(...+stage3+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:2.583837ms, swizzle: 1024, TFLOPS: 106.38
       f16wmma(...+stage2+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:2.532482ms, swizzle: 1024, TFLOPS: 108.54(+0.01%)
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:2.747654ms, swizzle: 1024, TFLOPS: 100.04
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:2.730131ms, swizzle: 1024, TFLOPS: 100.68
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:2.793526ms, swizzle: 1024, TFLOPS: 98.40
                             f16(cublas): ['40.53125  ', '-9.296875 '], time:2.633857ms, swizzle: NOOP, TFLOPS: 104.36
                                  f16_th: ['40.0625   ', '-9.2890625'], time:2.550768ms, swizzle: NOOP, TFLOPS: 107.76
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=8192
                              f16(naive): ['80.125    ', '81.25     '], time:150.4420ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                     f16x8pack(t8x8+bcf): ['80.125    ', '81.25     '], time:11.28554ms, swizzle: NOOP, TFLOPS: 48.71 (+1233.05%)
                f16x8pack(t8x8+bcf+dbuf): ['80.125    ', '81.25     '], time:11.37616ms, swizzle: NOOP, TFLOPS: 48.33
                f16x8pack(t8x8+k16+dbuf): ['80.125    ', '81.25     '], time:10.75201ms, swizzle: NOOP, TFLOPS: 51.13 (+4.96%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['80.4375   ', '81.8125   '], time:23.09622ms, swizzle: NOOP, TFLOPS: 23.80
                         f16wmma(mma4x2): ['80.4375   ', '81.8125   '], time:11.64886ms, swizzle: NOOP, TFLOPS: 47.19
                 f16wmma(mma4x2+warp2x4): ['80.4375   ', '81.8125   '], time:7.186770ms, swizzle: NOOP, TFLOPS: 76.50 (+49.61%)
            f16wmma(mma2x4+warp2x4+dbuf): ['80.4375   ', '81.8125   '], time:5.116152ms, swizzle: NOOP, TFLOPS: 107.45(+40.47%)
          f16wmma(mma2x4+warp2x4+stage4): ['80.4375   ', '81.8125   '], time:5.130147ms, swizzle: NOOP, TFLOPS: 107.16
          f16wmma(mma2x4+warp2x4+stage3): ['80.4375   ', '81.8125   '], time:5.058288ms, swizzle: NOOP, TFLOPS: 108.68(+1.14%)
          f16wmma(mma2x4+warp2x4+stage2): ['80.4375   ', '81.8125   '], time:5.082297ms, swizzle: NOOP, TFLOPS: 108.17
        f16wmma(mma2x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:5.110549ms, swizzle: NOOP, TFLOPS: 107.57
        f16wmma(mma2x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:5.098056ms, swizzle: NOOP, TFLOPS: 107.84
        f16wmma(mma2x4+...+stage2+dsmem): ['80.4375   ', '81.8125   '], time:5.076909ms, swizzle: NOOP, TFLOPS: 108.29
        f16wmma(mma4x4+...+stage4+dsmem): ['80.4375   ', '81.8125   '], time:5.357861ms, swizzle: NOOP, TFLOPS: 102.61
        f16wmma(mma4x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:5.465102ms, swizzle: NOOP, TFLOPS: 100.59
        f16wmma(mma4x4+...+stage2+dsmem): ['80.4375   ', '81.8125   '], time:5.675625ms, swizzle: NOOP, TFLOPS: 96.86
      f16wmma(mma2x4+...+stage4+swizzle): ['80.4375   ', '81.8125   '], time:5.157303ms, swizzle: 1024, TFLOPS: 106.60
      f16wmma(mma2x4+...+stage3+swizzle): ['80.4375   ', '81.8125   '], time:5.145287ms, swizzle: 1024, TFLOPS: 106.85
      f16wmma(mma2x4+...+stage2+swizzle): ['80.4375   ', '81.8125   '], time:5.068826ms, swizzle: 1024, TFLOPS: 108.46
       f16wmma(...+stage3+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:5.206513ms, swizzle: 1024, TFLOPS: 105.59
       f16wmma(...+stage2+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:5.068516ms, swizzle: 1024, TFLOPS: 108.46
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:5.423307ms, swizzle: 1024, TFLOPS: 101.37
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:5.388140ms, swizzle: 1024, TFLOPS: 102.03
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:5.488562ms, swizzle: 1024, TFLOPS: 100.16
                             f16(cublas): ['80.4375   ', '81.8125   '], time:5.087113ms, swizzle: NOOP, TFLOPS: 108.07
                                  f16_th: ['79.875    ', '81.8125   '], time:4.828977ms, swizzle: NOOP, TFLOPS: 113.85(+4.75%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=2048
                              f16(naive): ['80.9375   ', '5.7734375 '], time:75.27191ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                     f16x8pack(t8x8+bcf): ['80.9375   ', '5.7734375 '], time:5.366444ms, swizzle: NOOP, TFLOPS: 51.22 (+1302.64%)
                f16x8pack(t8x8+bcf+dbuf): ['80.9375   ', '5.7734375 '], time:5.368256ms, swizzle: NOOP, TFLOPS: 51.20
                f16x8pack(t8x8+k16+dbuf): ['80.9375   ', '5.7734375 '], time:5.156874ms, swizzle: NOOP, TFLOPS: 53.30 (+4.06%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['81.1875   ', '5.9296875 '], time:11.66641ms, swizzle: NOOP, TFLOPS: 23.56
                         f16wmma(mma4x2): ['81.1875   ', '5.9296875 '], time:5.896234ms, swizzle: NOOP, TFLOPS: 46.62
                 f16wmma(mma4x2+warp2x4): ['81.1875   ', '5.9296875 '], time:3.550696ms, swizzle: NOOP, TFLOPS: 77.42 (+45.24%)
            f16wmma(mma2x4+warp2x4+dbuf): ['81.1875   ', '5.9296875 '], time:2.593255ms, swizzle: NOOP, TFLOPS: 106.00(+36.92%)
          f16wmma(mma2x4+warp2x4+stage4): ['81.1875   ', '5.9296875 '], time:2.605295ms, swizzle: NOOP, TFLOPS: 105.51
          f16wmma(mma2x4+warp2x4+stage3): ['81.1875   ', '5.9296875 '], time:2.546215ms, swizzle: NOOP, TFLOPS: 107.96(+1.85%)
          f16wmma(mma2x4+warp2x4+stage2): ['81.1875   ', '5.9296875 '], time:2.558183ms, swizzle: NOOP, TFLOPS: 107.45
        f16wmma(mma2x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:2.595591ms, swizzle: NOOP, TFLOPS: 105.90
        f16wmma(mma2x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:2.563333ms, swizzle: NOOP, TFLOPS: 107.23
        f16wmma(mma2x4+...+stage2+dsmem): ['81.1875   ', '5.9296875 '], time:2.554893ms, swizzle: NOOP, TFLOPS: 107.59
        f16wmma(mma4x4+...+stage4+dsmem): ['81.1875   ', '5.9296875 '], time:2.791786ms, swizzle: NOOP, TFLOPS: 98.46
        f16wmma(mma4x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:2.851438ms, swizzle: NOOP, TFLOPS: 96.40
        f16wmma(mma4x4+...+stage2+dsmem): ['81.1875   ', '5.9296875 '], time:2.935194ms, swizzle: NOOP, TFLOPS: 93.65
      f16wmma(mma2x4+...+stage4+swizzle): ['81.1875   ', '5.9296875 '], time:2.618432ms, swizzle: 2048, TFLOPS: 104.98
      f16wmma(mma2x4+...+stage3+swizzle): ['81.1875   ', '5.9296875 '], time:2.540159ms, swizzle: 2048, TFLOPS: 108.21(+0.24%)
      f16wmma(mma2x4+...+stage2+swizzle): ['81.1875   ', '5.9296875 '], time:2.523136ms, swizzle: 2048, TFLOPS: 108.94(+0.67%)
       f16wmma(...+stage3+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:2.564334ms, swizzle: 2048, TFLOPS: 107.19
       f16wmma(...+stage2+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:2.532672ms, swizzle: 2048, TFLOPS: 108.53
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:2.834129ms, swizzle: 2048, TFLOPS: 96.99
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:2.812409ms, swizzle: 2048, TFLOPS: 97.74
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:2.852797ms, swizzle: 2048, TFLOPS: 96.35
                             f16(cublas): ['81.1875   ', '5.9296875 '], time:2.618527ms, swizzle: NOOP, TFLOPS: 104.97
                                  f16_th: ['80.875    ', '5.9140625 '], time:2.400469ms, swizzle: NOOP, TFLOPS: 114.51(+5.11%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=4096
                              f16(naive): ['40.09375  ', '-9.3671875'], time:150.3400ms, swizzle: NOOP, TFLOPS: 3.66  (+0.00%)
                     f16x8pack(t8x8+bcf): ['40.09375  ', '-9.3671875'], time:10.91401ms, swizzle: NOOP, TFLOPS: 50.37 (+1277.50%)
                f16x8pack(t8x8+bcf+dbuf): ['40.09375  ', '-9.3671875'], time:10.95380ms, swizzle: NOOP, TFLOPS: 50.19
                f16x8pack(t8x8+k16+dbuf): ['40.09375  ', '-9.3671875'], time:10.41371ms, swizzle: NOOP, TFLOPS: 52.79 (+4.80%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['40.53125  ', '-9.296875 '], time:23.21007ms, swizzle: NOOP, TFLOPS: 23.69
                         f16wmma(mma4x2): ['40.53125  ', '-9.296875 '], time:11.64598ms, swizzle: NOOP, TFLOPS: 47.21
                 f16wmma(mma4x2+warp2x4): ['40.53125  ', '-9.296875 '], time:6.972169ms, swizzle: NOOP, TFLOPS: 78.85 (+49.36%)
            f16wmma(mma2x4+warp2x4+dbuf): ['40.53125  ', '-9.296875 '], time:5.051231ms, swizzle: NOOP, TFLOPS: 108.84(+38.03%)
          f16wmma(mma2x4+warp2x4+stage4): ['40.53125  ', '-9.296875 '], time:5.065608ms, swizzle: NOOP, TFLOPS: 108.53
          f16wmma(mma2x4+warp2x4+stage3): ['40.53125  ', '-9.296875 '], time:5.004525ms, swizzle: NOOP, TFLOPS: 109.85(+0.93%)
          f16wmma(mma2x4+warp2x4+stage2): ['40.53125  ', '-9.296875 '], time:5.029392ms, swizzle: NOOP, TFLOPS: 109.31
        f16wmma(mma2x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:5.068373ms, swizzle: NOOP, TFLOPS: 108.47
        f16wmma(mma2x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:5.043554ms, swizzle: NOOP, TFLOPS: 109.00
        f16wmma(mma2x4+...+stage2+dsmem): ['40.53125  ', '-9.296875 '], time:5.023980ms, swizzle: NOOP, TFLOPS: 109.43
        f16wmma(mma4x4+...+stage4+dsmem): ['40.53125  ', '-9.296875 '], time:5.455899ms, swizzle: NOOP, TFLOPS: 100.76
        f16wmma(mma4x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:5.546998ms, swizzle: NOOP, TFLOPS: 99.11
        f16wmma(mma4x4+...+stage2+dsmem): ['40.53125  ', '-9.296875 '], time:5.740261ms, swizzle: NOOP, TFLOPS: 95.77
      f16wmma(mma2x4+...+stage4+swizzle): ['40.53125  ', '-9.296875 '], time:5.146169ms, swizzle: 2048, TFLOPS: 106.83
      f16wmma(mma2x4+...+stage3+swizzle): ['40.53125  ', '-9.296875 '], time:5.062413ms, swizzle: 2048, TFLOPS: 108.60
      f16wmma(mma2x4+...+stage2+swizzle): ['40.53125  ', '-9.296875 '], time:4.995584ms, swizzle: 2048, TFLOPS: 110.05(+0.18%)
       f16wmma(...+stage3+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:5.057883ms, swizzle: 2048, TFLOPS: 108.69
       f16wmma(...+stage2+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:5.010056ms, swizzle: 2048, TFLOPS: 109.73
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:5.495810ms, swizzle: 2048, TFLOPS: 100.03
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:5.458950ms, swizzle: 2048, TFLOPS: 100.71
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:5.551075ms, swizzle: 2048, TFLOPS: 99.04
                             f16(cublas): ['40.53125  ', '-9.296875 '], time:4.989862ms, swizzle: NOOP, TFLOPS: 110.17(+0.11%)
                                  f16_th: ['40.0625   ', '-9.2890625'], time:4.875206ms, swizzle: NOOP, TFLOPS: 112.77(+2.35%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=8192
                              f16(naive): ['80.125    ', '81.25     '], time:443.9446ms, swizzle: NOOP, TFLOPS: 2.48  (+0.00%)
                     f16x8pack(t8x8+bcf): ['80.125    ', '81.25     '], time:24.43621ms, swizzle: NOOP, TFLOPS: 45.00 (+1716.75%)
                f16x8pack(t8x8+bcf+dbuf): ['80.125    ', '81.25     '], time:24.16708ms, swizzle: NOOP, TFLOPS: 45.50 (+1.11%)
                f16x8pack(t8x8+k16+dbuf): ['80.125    ', '81.25     '], time:23.61018ms, swizzle: NOOP, TFLOPS: 46.57 (+2.36%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['80.4375   ', '81.8125   '], time:48.91512ms, swizzle: NOOP, TFLOPS: 22.48
                         f16wmma(mma4x2): ['80.4375   ', '81.8125   '], time:25.23126ms, swizzle: NOOP, TFLOPS: 43.58
                 f16wmma(mma4x2+warp2x4): ['80.4375   ', '81.8125   '], time:14.22612ms, swizzle: NOOP, TFLOPS: 77.29 (+65.96%)
            f16wmma(mma2x4+warp2x4+dbuf): ['80.4375   ', '81.8125   '], time:10.96324ms, swizzle: NOOP, TFLOPS: 100.29(+29.76%)
          f16wmma(mma2x4+warp2x4+stage4): ['80.4375   ', '81.8125   '], time:10.90235ms, swizzle: NOOP, TFLOPS: 100.85(+0.56%)
          f16wmma(mma2x4+warp2x4+stage3): ['80.4375   ', '81.8125   '], time:10.99100ms, swizzle: NOOP, TFLOPS: 100.04
          f16wmma(mma2x4+warp2x4+stage2): ['80.4375   ', '81.8125   '], time:10.81967ms, swizzle: NOOP, TFLOPS: 101.62(+0.76%)
        f16wmma(mma2x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:10.98363ms, swizzle: NOOP, TFLOPS: 100.10
        f16wmma(mma2x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:11.12866ms, swizzle: NOOP, TFLOPS: 98.80
        f16wmma(mma2x4+...+stage2+dsmem): ['80.4375   ', '81.8125   '], time:11.08555ms, swizzle: NOOP, TFLOPS: 99.18
        f16wmma(mma4x4+...+stage4+dsmem): ['80.4375   ', '81.8125   '], time:10.74411ms, swizzle: NOOP, TFLOPS: 102.34(+0.70%)
        f16wmma(mma4x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:10.93094ms, swizzle: NOOP, TFLOPS: 100.59
        f16wmma(mma4x4+...+stage2+dsmem): ['80.4375   ', '81.8125   '], time:11.31961ms, swizzle: NOOP, TFLOPS: 97.13
      f16wmma(mma2x4+...+stage4+swizzle): ['80.4375   ', '81.8125   '], time:10.33263ms, swizzle: 2048, TFLOPS: 106.41(+3.98%)
      f16wmma(mma2x4+...+stage3+swizzle): ['80.4375   ', '81.8125   '], time:10.13724ms, swizzle: 2048, TFLOPS: 108.46(+1.93%)
      f16wmma(mma2x4+...+stage2+swizzle): ['80.4375   ', '81.8125   '], time:10.01677ms, swizzle: 2048, TFLOPS: 109.77(+1.20%)
       f16wmma(...+stage3+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:10.24608ms, swizzle: 2048, TFLOPS: 107.31
       f16wmma(...+stage2+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:9.971213ms, swizzle: 2048, TFLOPS: 110.27(+0.46%)
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:10.80942ms, swizzle: 2048, TFLOPS: 101.72
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:10.74123ms, swizzle: 2048, TFLOPS: 102.36
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:10.93015ms, swizzle: 2048, TFLOPS: 100.59
                             f16(cublas): ['80.4375   ', '81.8125   '], time:9.702062ms, swizzle: NOOP, TFLOPS: 113.33(+2.77%)
                                  f16_th: ['79.875    ', '81.875    '], time:9.730458ms, swizzle: NOOP, TFLOPS: 113.00
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=2048
                              f16(naive): ['80.9375   ', '5.7734375 '], time:150.5032ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                     f16x8pack(t8x8+bcf): ['80.9375   ', '5.7734375 '], time:11.12394ms, swizzle: NOOP, TFLOPS: 49.42 (+1252.97%)
                f16x8pack(t8x8+bcf+dbuf): ['80.9375   ', '5.7734375 '], time:11.13758ms, swizzle: NOOP, TFLOPS: 49.36
                f16x8pack(t8x8+k16+dbuf): ['80.9375   ', '5.7734375 '], time:10.62729ms, swizzle: NOOP, TFLOPS: 51.73 (+4.67%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['81.1875   ', '5.9296875 '], time:23.26977ms, swizzle: NOOP, TFLOPS: 23.63
                         f16wmma(mma4x2): ['81.1875   ', '5.9296875 '], time:11.70744ms, swizzle: NOOP, TFLOPS: 46.96
                 f16wmma(mma4x2+warp2x4): ['81.1875   ', '5.9296875 '], time:6.896805ms, swizzle: NOOP, TFLOPS: 79.71 (+54.09%)
            f16wmma(mma2x4+warp2x4+dbuf): ['81.1875   ', '5.9296875 '], time:5.144286ms, swizzle: NOOP, TFLOPS: 106.87(+34.07%)
          f16wmma(mma2x4+warp2x4+stage4): ['81.1875   ', '5.9296875 '], time:5.202627ms, swizzle: NOOP, TFLOPS: 105.67
          f16wmma(mma2x4+warp2x4+stage3): ['81.1875   ', '5.9296875 '], time:5.079007ms, swizzle: NOOP, TFLOPS: 108.24(+1.29%)
          f16wmma(mma2x4+warp2x4+stage2): ['81.1875   ', '5.9296875 '], time:5.105161ms, swizzle: NOOP, TFLOPS: 107.69
        f16wmma(mma2x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:5.143880ms, swizzle: NOOP, TFLOPS: 106.88
        f16wmma(mma2x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:5.090236ms, swizzle: NOOP, TFLOPS: 108.00
        f16wmma(mma2x4+...+stage2+dsmem): ['81.1875   ', '5.9296875 '], time:5.110311ms, swizzle: NOOP, TFLOPS: 107.58
        f16wmma(mma4x4+...+stage4+dsmem): ['81.1875   ', '5.9296875 '], time:5.369067ms, swizzle: NOOP, TFLOPS: 102.39
        f16wmma(mma4x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:5.497217ms, swizzle: NOOP, TFLOPS: 100.01
        f16wmma(mma4x4+...+stage2+dsmem): ['81.1875   ', '5.9296875 '], time:5.633306ms, swizzle: NOOP, TFLOPS: 97.59
      f16wmma(mma2x4+...+stage4+swizzle): ['81.1875   ', '5.9296875 '], time:5.239129ms, swizzle: 4096, TFLOPS: 104.93
      f16wmma(mma2x4+...+stage3+swizzle): ['81.1875   ', '5.9296875 '], time:5.041193ms, swizzle: 4096, TFLOPS: 109.05(+0.75%)
      f16wmma(mma2x4+...+stage2+swizzle): ['81.1875   ', '5.9296875 '], time:5.040335ms, swizzle: 4096, TFLOPS: 109.07(+0.02%)
       f16wmma(...+stage3+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:5.149388ms, swizzle: 4096, TFLOPS: 106.76
       f16wmma(...+stage2+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:5.042314ms, swizzle: 4096, TFLOPS: 109.03
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:5.438852ms, swizzle: 4096, TFLOPS: 101.08
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:5.361533ms, swizzle: 4096, TFLOPS: 102.54
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:5.455160ms, swizzle: 4096, TFLOPS: 100.78
                             f16(cublas): ['81.1875   ', '5.9296875 '], time:5.007004ms, swizzle: NOOP, TFLOPS: 109.80(+0.67%)
                                  f16_th: ['80.875    ', '5.9140625 '], time:4.822874ms, swizzle: NOOP, TFLOPS: 113.99(+3.82%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=4096
                              f16(naive): ['40.09375  ', '-9.3671875'], time:458.9131ms, swizzle: NOOP, TFLOPS: 2.40  (+0.00%)
                     f16x8pack(t8x8+bcf): ['40.09375  ', '-9.3671875'], time:24.46501ms, swizzle: NOOP, TFLOPS: 44.94 (+1775.79%)
                f16x8pack(t8x8+bcf+dbuf): ['40.09375  ', '-9.3671875'], time:24.48763ms, swizzle: NOOP, TFLOPS: 44.90
                f16x8pack(t8x8+k16+dbuf): ['40.09375  ', '-9.3671875'], time:23.89101ms, swizzle: NOOP, TFLOPS: 46.02 (+2.40%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['40.53125  ', '-9.296875 '], time:76.20103ms, swizzle: NOOP, TFLOPS: 14.43
                         f16wmma(mma4x2): ['40.53125  ', '-9.296875 '], time:31.38382ms, swizzle: NOOP, TFLOPS: 35.03
                 f16wmma(mma4x2+warp2x4): ['40.53125  ', '-9.296875 '], time:14.28868ms, swizzle: NOOP, TFLOPS: 76.95 (+67.20%)
            f16wmma(mma2x4+warp2x4+dbuf): ['40.53125  ', '-9.296875 '], time:11.25531ms, swizzle: NOOP, TFLOPS: 97.69 (+26.95%)
          f16wmma(mma2x4+warp2x4+stage4): ['40.53125  ', '-9.296875 '], time:11.47220ms, swizzle: NOOP, TFLOPS: 95.84
          f16wmma(mma2x4+warp2x4+stage3): ['40.53125  ', '-9.296875 '], time:11.45608ms, swizzle: NOOP, TFLOPS: 95.98
          f16wmma(mma2x4+warp2x4+stage2): ['40.53125  ', '-9.296875 '], time:11.18333ms, swizzle: NOOP, TFLOPS: 98.32 (+0.64%)
        f16wmma(mma2x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:11.58487ms, swizzle: NOOP, TFLOPS: 94.91
        f16wmma(mma2x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:11.32571ms, swizzle: NOOP, TFLOPS: 97.08
        f16wmma(mma2x4+...+stage2+dsmem): ['40.53125  ', '-9.296875 '], time:11.01408ms, swizzle: NOOP, TFLOPS: 99.83 (+1.54%)
        f16wmma(mma4x4+...+stage4+dsmem): ['40.53125  ', '-9.296875 '], time:10.46371ms, swizzle: NOOP, TFLOPS: 105.08(+5.26%)
        f16wmma(mma4x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:10.66100ms, swizzle: NOOP, TFLOPS: 103.13
        f16wmma(mma4x4+...+stage2+dsmem): ['40.53125  ', '-9.296875 '], time:11.01729ms, swizzle: NOOP, TFLOPS: 99.80
      f16wmma(mma2x4+...+stage4+swizzle): ['40.53125  ', '-9.296875 '], time:10.33980ms, swizzle: 4096, TFLOPS: 106.34(+1.20%)
      f16wmma(mma2x4+...+stage3+swizzle): ['40.53125  ', '-9.296875 '], time:10.16290ms, swizzle: 4096, TFLOPS: 108.19(+1.74%)
      f16wmma(mma2x4+...+stage2+swizzle): ['40.53125  ', '-9.296875 '], time:9.991765ms, swizzle: 4096, TFLOPS: 110.04(+1.71%)
       f16wmma(...+stage3+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:10.27524ms, swizzle: 4096, TFLOPS: 107.01
       f16wmma(...+stage2+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:10.03918ms, swizzle: 4096, TFLOPS: 109.52
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:10.51924ms, swizzle: 4096, TFLOPS: 104.52
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:10.45379ms, swizzle: 4096, TFLOPS: 105.18
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:10.62982ms, swizzle: 4096, TFLOPS: 103.44
                             f16(cublas): ['40.53125  ', '-9.296875 '], time:9.773707ms, swizzle: NOOP, TFLOPS: 112.50(+2.23%)
                                  f16_th: ['40.0625   ', '-9.2890625'], time:9.578633ms, swizzle: NOOP, TFLOPS: 114.79(+2.04%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=8192
                              f16(naive): ['80.125    ', '81.25     '], time:916.5035ms, swizzle: NOOP, TFLOPS: 2.40  (+0.00%)
                     f16x8pack(t8x8+bcf): ['80.125    ', '81.25     '], time:51.11901ms, swizzle: NOOP, TFLOPS: 43.02 (+1692.88%)
                f16x8pack(t8x8+bcf+dbuf): ['80.125    ', '81.25     '], time:51.80337ms, swizzle: NOOP, TFLOPS: 42.45
                f16x8pack(t8x8+k16+dbuf): ['80.125    ', '81.25     '], time:50.93510ms, swizzle: NOOP, TFLOPS: 43.17 (+0.36%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['80.4375   ', '81.8125   '], time:407.6780ms, swizzle: NOOP, TFLOPS: 5.39
                         f16wmma(mma4x2): ['80.4375   ', '81.8125   '], time:62.05229ms, swizzle: NOOP, TFLOPS: 35.44
                 f16wmma(mma4x2+warp2x4): ['80.4375   ', '81.8125   '], time:28.18341ms, swizzle: NOOP, TFLOPS: 78.03 (+80.73%)
            f16wmma(mma2x4+warp2x4+dbuf): ['80.4375   ', '81.8125   '], time:25.30872ms, swizzle: NOOP, TFLOPS: 86.89 (+11.36%)
          f16wmma(mma2x4+warp2x4+stage4): ['80.4375   ', '81.8125   '], time:24.87831ms, swizzle: NOOP, TFLOPS: 88.39 (+1.73%)
          f16wmma(mma2x4+warp2x4+stage3): ['80.4375   ', '81.8125   '], time:24.94282ms, swizzle: NOOP, TFLOPS: 88.16
          f16wmma(mma2x4+warp2x4+stage2): ['80.4375   ', '81.8125   '], time:24.97403ms, swizzle: NOOP, TFLOPS: 88.05
        f16wmma(mma2x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:24.87688ms, swizzle: NOOP, TFLOPS: 88.40 (+0.01%)
        f16wmma(mma2x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:24.91343ms, swizzle: NOOP, TFLOPS: 88.27
        f16wmma(mma2x4+...+stage2+dsmem): ['80.4375   ', '81.8125   '], time:24.97208ms, swizzle: NOOP, TFLOPS: 88.06
        f16wmma(mma4x4+...+stage4+dsmem): ['80.4375   ', '81.8125   '], time:20.58312ms, swizzle: NOOP, TFLOPS: 106.84(+20.86%)
        f16wmma(mma4x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:20.87883ms, swizzle: NOOP, TFLOPS: 105.32
        f16wmma(mma4x4+...+stage2+dsmem): ['80.4375   ', '81.8125   '], time:21.67587ms, swizzle: NOOP, TFLOPS: 101.45
      f16wmma(mma2x4+...+stage4+swizzle): ['80.4375   ', '81.8125   '], time:20.80790ms, swizzle: 4096, TFLOPS: 105.68
      f16wmma(mma2x4+...+stage3+swizzle): ['80.4375   ', '81.8125   '], time:20.47626ms, swizzle: 4096, TFLOPS: 107.39(+0.52%)
      f16wmma(mma2x4+...+stage2+swizzle): ['80.4375   ', '81.8125   '], time:20.23813ms, swizzle: 4096, TFLOPS: 108.66(+1.18%)
       f16wmma(...+stage3+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:20.64583ms, swizzle: 4096, TFLOPS: 106.51
       f16wmma(...+stage2+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:20.18899ms, swizzle: 4096, TFLOPS: 108.92(+0.24%)
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:20.68908ms, swizzle: 4096, TFLOPS: 106.29
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:20.45376ms, swizzle: 4096, TFLOPS: 107.51
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:20.95272ms, swizzle: 4096, TFLOPS: 104.95
                             f16(cublas): ['80.4375   ', '81.8125   '], time:23.40786ms, swizzle: NOOP, TFLOPS: 93.94
                                  f16_th: ['79.875    ', '81.875    '], time:19.36850ms, swizzle: NOOP, TFLOPS: 113.54(+4.24%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=2048
                              f16(naive): ['80.9375   ', '5.7734375 '], time:75.29358ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                     f16x8pack(t8x8+bcf): ['80.9375   ', '5.7734375 '], time:5.523920ms, swizzle: NOOP, TFLOPS: 49.76 (+1263.05%)
                f16x8pack(t8x8+bcf+dbuf): ['80.9375   ', '5.7734375 '], time:5.498003ms, swizzle: NOOP, TFLOPS: 50.00 (+0.47%)
                f16x8pack(t8x8+k16+dbuf): ['80.9375   ', '5.7734375 '], time:5.202889ms, swizzle: NOOP, TFLOPS: 52.83 (+5.67%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['81.1875   ', '5.9296875 '], time:11.68839ms, swizzle: NOOP, TFLOPS: 23.52
                         f16wmma(mma4x2): ['81.1875   ', '5.9296875 '], time:5.898237ms, swizzle: NOOP, TFLOPS: 46.60
                 f16wmma(mma4x2+warp2x4): ['81.1875   ', '5.9296875 '], time:3.558826ms, swizzle: NOOP, TFLOPS: 77.24 (+46.20%)
            f16wmma(mma2x4+warp2x4+dbuf): ['81.1875   ', '5.9296875 '], time:2.595853ms, swizzle: NOOP, TFLOPS: 105.89(+37.10%)
          f16wmma(mma2x4+warp2x4+stage4): ['81.1875   ', '5.9296875 '], time:2.604889ms, swizzle: NOOP, TFLOPS: 105.52
          f16wmma(mma2x4+warp2x4+stage3): ['81.1875   ', '5.9296875 '], time:2.551507ms, swizzle: NOOP, TFLOPS: 107.73(+1.74%)
          f16wmma(mma2x4+warp2x4+stage2): ['81.1875   ', '5.9296875 '], time:2.557659ms, swizzle: NOOP, TFLOPS: 107.47
        f16wmma(mma2x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:2.604603ms, swizzle: NOOP, TFLOPS: 105.54
        f16wmma(mma2x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:2.571511ms, swizzle: NOOP, TFLOPS: 106.89
        f16wmma(mma2x4+...+stage2+dsmem): ['81.1875   ', '5.9296875 '], time:2.565240ms, swizzle: NOOP, TFLOPS: 107.15
        f16wmma(mma4x4+...+stage4+dsmem): ['81.1875   ', '5.9296875 '], time:2.787709ms, swizzle: NOOP, TFLOPS: 98.60
        f16wmma(mma4x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:2.852010ms, swizzle: NOOP, TFLOPS: 96.38
        f16wmma(mma4x4+...+stage2+dsmem): ['81.1875   ', '5.9296875 '], time:2.944684ms, swizzle: NOOP, TFLOPS: 93.35
      f16wmma(mma2x4+...+stage4+swizzle): ['81.1875   ', '5.9296875 '], time:2.630496ms, swizzle: 1024, TFLOPS: 104.50
      f16wmma(mma2x4+...+stage3+swizzle): ['81.1875   ', '5.9296875 '], time:2.576422ms, swizzle: 1024, TFLOPS: 106.69
      f16wmma(mma2x4+...+stage2+swizzle): ['81.1875   ', '5.9296875 '], time:2.532649ms, swizzle: 1024, TFLOPS: 108.53(+0.74%)
       f16wmma(...+stage3+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:2.581381ms, swizzle: 1024, TFLOPS: 106.48
       f16wmma(...+stage2+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:2.533221ms, swizzle: 1024, TFLOPS: 108.51
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:2.838969ms, swizzle: 1024, TFLOPS: 96.82
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:2.823424ms, swizzle: 1024, TFLOPS: 97.36
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:2.869677ms, swizzle: 1024, TFLOPS: 95.79
                             f16(cublas): ['81.1875   ', '5.9296875 '], time:2.845001ms, swizzle: NOOP, TFLOPS: 96.62
                                  f16_th: ['80.875    ', '5.9140625 '], time:2.411317ms, swizzle: NOOP, TFLOPS: 113.99(+5.03%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=4096
                              f16(naive): ['40.09375  ', '-9.3671875'], time:150.4282ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                     f16x8pack(t8x8+bcf): ['40.09375  ', '-9.3671875'], time:11.17117ms, swizzle: NOOP, TFLOPS: 49.21 (+1246.57%)
                f16x8pack(t8x8+bcf+dbuf): ['40.09375  ', '-9.3671875'], time:11.10396ms, swizzle: NOOP, TFLOPS: 49.51 (+0.61%)
                f16x8pack(t8x8+k16+dbuf): ['40.09375  ', '-9.3671875'], time:10.88376ms, swizzle: NOOP, TFLOPS: 50.51 (+2.02%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['40.53125  ', '-9.296875 '], time:23.20601ms, swizzle: NOOP, TFLOPS: 23.69
                         f16wmma(mma4x2): ['40.53125  ', '-9.296875 '], time:11.64853ms, swizzle: NOOP, TFLOPS: 47.20
                 f16wmma(mma4x2+warp2x4): ['40.53125  ', '-9.296875 '], time:6.994080ms, swizzle: NOOP, TFLOPS: 78.60 (+55.61%)
            f16wmma(mma2x4+warp2x4+dbuf): ['40.53125  ', '-9.296875 '], time:5.106353ms, swizzle: NOOP, TFLOPS: 107.66(+36.97%)
          f16wmma(mma2x4+warp2x4+stage4): ['40.53125  ', '-9.296875 '], time:5.128264ms, swizzle: NOOP, TFLOPS: 107.20
          f16wmma(mma2x4+warp2x4+stage3): ['40.53125  ', '-9.296875 '], time:5.025506ms, swizzle: NOOP, TFLOPS: 109.39(+1.61%)
          f16wmma(mma2x4+warp2x4+stage2): ['40.53125  ', '-9.296875 '], time:5.028367ms, swizzle: NOOP, TFLOPS: 109.33
        f16wmma(mma2x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:5.091547ms, swizzle: NOOP, TFLOPS: 107.97
        f16wmma(mma2x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:5.079436ms, swizzle: NOOP, TFLOPS: 108.23
        f16wmma(mma2x4+...+stage2+dsmem): ['40.53125  ', '-9.296875 '], time:5.068445ms, swizzle: NOOP, TFLOPS: 108.47
        f16wmma(mma4x4+...+stage4+dsmem): ['40.53125  ', '-9.296875 '], time:5.452799ms, swizzle: NOOP, TFLOPS: 100.82
        f16wmma(mma4x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:5.563282ms, swizzle: NOOP, TFLOPS: 98.82
        f16wmma(mma4x4+...+stage2+dsmem): ['40.53125  ', '-9.296875 '], time:5.774235ms, swizzle: NOOP, TFLOPS: 95.21
      f16wmma(mma2x4+...+stage4+swizzle): ['40.53125  ', '-9.296875 '], time:5.212688ms, swizzle: 1024, TFLOPS: 105.46
      f16wmma(mma2x4+...+stage3+swizzle): ['40.53125  ', '-9.296875 '], time:5.090284ms, swizzle: 1024, TFLOPS: 108.00
      f16wmma(mma2x4+...+stage2+swizzle): ['40.53125  ', '-9.296875 '], time:5.064606ms, swizzle: 1024, TFLOPS: 108.55
       f16wmma(...+stage3+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:5.226016ms, swizzle: 1024, TFLOPS: 105.20
       f16wmma(...+stage2+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:4.990983ms, swizzle: 1024, TFLOPS: 110.15(+0.69%)
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:5.496168ms, swizzle: 1024, TFLOPS: 100.03
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:5.452466ms, swizzle: 1024, TFLOPS: 100.83
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:5.582737ms, swizzle: 1024, TFLOPS: 98.47
                             f16(cublas): ['40.53125  ', '-9.296875 '], time:4.924607ms, swizzle: NOOP, TFLOPS: 111.63(+1.35%)
                                  f16_th: ['40.0625   ', '-9.2890625'], time:4.886174ms, swizzle: NOOP, TFLOPS: 112.51(+0.79%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=8192
                              f16(naive): ['80.125    ', '81.25     '], time:300.5355ms, swizzle: NOOP, TFLOPS: 3.66  (+0.00%)
                     f16x8pack(t8x8+bcf): ['80.125    ', '81.25     '], time:22.80614ms, swizzle: NOOP, TFLOPS: 48.21 (+1217.78%)
                f16x8pack(t8x8+bcf+dbuf): ['80.125    ', '81.25     '], time:22.48601ms, swizzle: NOOP, TFLOPS: 48.90 (+1.42%)
                f16x8pack(t8x8+k16+dbuf): ['80.125    ', '81.25     '], time:21.63348ms, swizzle: NOOP, TFLOPS: 50.82 (+3.94%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['80.4375   ', '81.8125   '], time:45.94721ms, swizzle: NOOP, TFLOPS: 23.93
                         f16wmma(mma4x2): ['80.4375   ', '81.8125   '], time:23.21996ms, swizzle: NOOP, TFLOPS: 47.35
                 f16wmma(mma4x2+warp2x4): ['80.4375   ', '81.8125   '], time:13.80817ms, swizzle: NOOP, TFLOPS: 79.63 (+56.67%)
            f16wmma(mma2x4+warp2x4+dbuf): ['80.4375   ', '81.8125   '], time:10.21535ms, swizzle: NOOP, TFLOPS: 107.63(+35.17%)
          f16wmma(mma2x4+warp2x4+stage4): ['80.4375   ', '81.8125   '], time:10.24181ms, swizzle: NOOP, TFLOPS: 107.36
          f16wmma(mma2x4+warp2x4+stage3): ['80.4375   ', '81.8125   '], time:10.15005ms, swizzle: NOOP, TFLOPS: 108.33(+0.64%)
          f16wmma(mma2x4+warp2x4+stage2): ['80.4375   ', '81.8125   '], time:10.14647ms, swizzle: NOOP, TFLOPS: 108.36(+0.04%)
        f16wmma(mma2x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:10.23235ms, swizzle: NOOP, TFLOPS: 107.45
        f16wmma(mma2x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:10.13498ms, swizzle: NOOP, TFLOPS: 108.49(+0.11%)
        f16wmma(mma2x4+...+stage2+dsmem): ['80.4375   ', '81.8125   '], time:10.04700ms, swizzle: NOOP, TFLOPS: 109.44(+0.88%)
        f16wmma(mma4x4+...+stage4+dsmem): ['80.4375   ', '81.8125   '], time:10.74309ms, swizzle: NOOP, TFLOPS: 102.35
        f16wmma(mma4x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:10.93523ms, swizzle: NOOP, TFLOPS: 100.55
        f16wmma(mma4x4+...+stage2+dsmem): ['80.4375   ', '81.8125   '], time:11.31312ms, swizzle: NOOP, TFLOPS: 97.19
      f16wmma(mma2x4+...+stage4+swizzle): ['80.4375   ', '81.8125   '], time:10.36229ms, swizzle: 1024, TFLOPS: 106.11
      f16wmma(mma2x4+...+stage3+swizzle): ['80.4375   ', '81.8125   '], time:10.36906ms, swizzle: 1024, TFLOPS: 106.04
      f16wmma(mma2x4+...+stage2+swizzle): ['80.4375   ', '81.8125   '], time:10.21506ms, swizzle: 1024, TFLOPS: 107.64
       f16wmma(...+stage3+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:10.46659ms, swizzle: 1024, TFLOPS: 105.05
       f16wmma(...+stage2+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:10.18373ms, swizzle: 1024, TFLOPS: 107.97
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:10.81142ms, swizzle: 1024, TFLOPS: 101.70
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:10.73374ms, swizzle: 1024, TFLOPS: 102.44
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:10.93778ms, swizzle: 1024, TFLOPS: 100.52
                             f16(cublas): ['80.4375   ', '81.8125   '], time:9.735608ms, swizzle: NOOP, TFLOPS: 112.94(+3.20%)
                                  f16_th: ['79.875    ', '81.8125   '], time:9.948205ms, swizzle: NOOP, TFLOPS: 110.52
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=2048
                              f16(naive): ['80.9375   ', '5.7734375 '], time:150.4915ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                     f16x8pack(t8x8+bcf): ['80.9375   ', '5.7734375 '], time:11.14513ms, swizzle: NOOP, TFLOPS: 49.33 (+1250.29%)
                f16x8pack(t8x8+bcf+dbuf): ['80.9375   ', '5.7734375 '], time:11.25109ms, swizzle: NOOP, TFLOPS: 48.86
                f16x8pack(t8x8+k16+dbuf): ['80.9375   ', '5.7734375 '], time:10.80908ms, swizzle: NOOP, TFLOPS: 50.86 (+3.11%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['81.1875   ', '5.9296875 '], time:23.29092ms, swizzle: NOOP, TFLOPS: 23.60
                         f16wmma(mma4x2): ['81.1875   ', '5.9296875 '], time:11.80624ms, swizzle: NOOP, TFLOPS: 46.56
                 f16wmma(mma4x2+warp2x4): ['81.1875   ', '5.9296875 '], time:6.971907ms, swizzle: NOOP, TFLOPS: 78.85 (+55.04%)
            f16wmma(mma2x4+warp2x4+dbuf): ['81.1875   ', '5.9296875 '], time:5.181264ms, swizzle: NOOP, TFLOPS: 106.10(+34.56%)
          f16wmma(mma2x4+warp2x4+stage4): ['81.1875   ', '5.9296875 '], time:5.202054ms, swizzle: NOOP, TFLOPS: 105.68
          f16wmma(mma2x4+warp2x4+stage3): ['81.1875   ', '5.9296875 '], time:5.136322ms, swizzle: NOOP, TFLOPS: 107.03(+0.87%)
          f16wmma(mma2x4+warp2x4+stage2): ['81.1875   ', '5.9296875 '], time:5.078673ms, swizzle: NOOP, TFLOPS: 108.25(+1.14%)
        f16wmma(mma2x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:5.148530ms, swizzle: NOOP, TFLOPS: 106.78
        f16wmma(mma2x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:5.149078ms, swizzle: NOOP, TFLOPS: 106.77
        f16wmma(mma2x4+...+stage2+dsmem): ['81.1875   ', '5.9296875 '], time:5.123376ms, swizzle: NOOP, TFLOPS: 107.30
        f16wmma(mma4x4+...+stage4+dsmem): ['81.1875   ', '5.9296875 '], time:5.356335ms, swizzle: NOOP, TFLOPS: 102.64
        f16wmma(mma4x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:5.488133ms, swizzle: NOOP, TFLOPS: 100.17
        f16wmma(mma4x4+...+stage2+dsmem): ['81.1875   ', '5.9296875 '], time:5.618929ms, swizzle: NOOP, TFLOPS: 97.84
      f16wmma(mma2x4+...+stage4+swizzle): ['81.1875   ', '5.9296875 '], time:5.223298ms, swizzle: 2048, TFLOPS: 105.25
      f16wmma(mma2x4+...+stage3+swizzle): ['81.1875   ', '5.9296875 '], time:5.177855ms, swizzle: 2048, TFLOPS: 106.17
      f16wmma(mma2x4+...+stage2+swizzle): ['81.1875   ', '5.9296875 '], time:5.139446ms, swizzle: 2048, TFLOPS: 106.97
       f16wmma(...+stage3+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:5.180191ms, swizzle: 2048, TFLOPS: 106.13
       f16wmma(...+stage2+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:5.124926ms, swizzle: 2048, TFLOPS: 107.27
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:5.410528ms, swizzle: 2048, TFLOPS: 101.61
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:5.367803ms, swizzle: 2048, TFLOPS: 102.42
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:5.481576ms, swizzle: 2048, TFLOPS: 100.29
                             f16(cublas): ['81.1875   ', '5.9296875 '], time:5.002284ms, swizzle: NOOP, TFLOPS: 109.90(+1.53%)
                                  f16_th: ['80.875    ', '5.9140625 '], time:4.804801ms, swizzle: NOOP, TFLOPS: 114.42(+4.11%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=4096
                              f16(naive): ['40.09375  ', '-9.3671875'], time:300.5918ms, swizzle: NOOP, TFLOPS: 3.66  (+0.00%)
                     f16x8pack(t8x8+bcf): ['40.09375  ', '-9.3671875'], time:22.47240ms, swizzle: NOOP, TFLOPS: 48.93 (+1237.60%)
                f16x8pack(t8x8+bcf+dbuf): ['40.09375  ', '-9.3671875'], time:22.44052ms, swizzle: NOOP, TFLOPS: 49.00 (+0.14%)
                f16x8pack(t8x8+k16+dbuf): ['40.09375  ', '-9.3671875'], time:21.43564ms, swizzle: NOOP, TFLOPS: 51.29 (+4.69%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['40.53125  ', '-9.296875 '], time:46.19073ms, swizzle: NOOP, TFLOPS: 23.80
                         f16wmma(mma4x2): ['40.53125  ', '-9.296875 '], time:23.26786ms, swizzle: NOOP, TFLOPS: 47.25
                 f16wmma(mma4x2+warp2x4): ['40.53125  ', '-9.296875 '], time:13.66479ms, swizzle: NOOP, TFLOPS: 80.46 (+56.87%)
            f16wmma(mma2x4+warp2x4+dbuf): ['40.53125  ', '-9.296875 '], time:10.21111ms, swizzle: NOOP, TFLOPS: 107.68(+33.82%)
          f16wmma(mma2x4+warp2x4+stage4): ['40.53125  ', '-9.296875 '], time:10.24806ms, swizzle: NOOP, TFLOPS: 107.29
          f16wmma(mma2x4+warp2x4+stage3): ['40.53125  ', '-9.296875 '], time:10.18645ms, swizzle: NOOP, TFLOPS: 107.94(+0.24%)
          f16wmma(mma2x4+warp2x4+stage2): ['40.53125  ', '-9.296875 '], time:10.18936ms, swizzle: NOOP, TFLOPS: 107.91
        f16wmma(mma2x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:10.28721ms, swizzle: NOOP, TFLOPS: 106.88
        f16wmma(mma2x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:10.24394ms, swizzle: NOOP, TFLOPS: 107.33
        f16wmma(mma2x4+...+stage2+dsmem): ['40.53125  ', '-9.296875 '], time:10.16311ms, swizzle: NOOP, TFLOPS: 108.19(+0.23%)
        f16wmma(mma4x4+...+stage4+dsmem): ['40.53125  ', '-9.296875 '], time:10.43620ms, swizzle: NOOP, TFLOPS: 105.36
        f16wmma(mma4x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:10.61503ms, swizzle: NOOP, TFLOPS: 103.58
        f16wmma(mma4x4+...+stage2+dsmem): ['40.53125  ', '-9.296875 '], time:11.00249ms, swizzle: NOOP, TFLOPS: 99.93
      f16wmma(mma2x4+...+stage4+swizzle): ['40.53125  ', '-9.296875 '], time:10.39535ms, swizzle: 2048, TFLOPS: 105.77
      f16wmma(mma2x4+...+stage3+swizzle): ['40.53125  ', '-9.296875 '], time:10.25552ms, swizzle: 2048, TFLOPS: 107.21
      f16wmma(mma2x4+...+stage2+swizzle): ['40.53125  ', '-9.296875 '], time:10.06872ms, swizzle: 2048, TFLOPS: 109.20(+0.94%)
       f16wmma(...+stage3+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:10.38682ms, swizzle: 2048, TFLOPS: 105.86
       f16wmma(...+stage2+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:10.16161ms, swizzle: 2048, TFLOPS: 108.20
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:10.51208ms, swizzle: 2048, TFLOPS: 104.59
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:10.44013ms, swizzle: 2048, TFLOPS: 105.32
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:10.62643ms, swizzle: 2048, TFLOPS: 103.47
                             f16(cublas): ['40.53125  ', '-9.296875 '], time:9.729194ms, swizzle: NOOP, TFLOPS: 113.01(+3.49%)
                                  f16_th: ['40.0625   ', '-9.2890625'], time:9.584045ms, swizzle: NOOP, TFLOPS: 114.72(+1.51%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=8192
                              f16(naive): ['80.125    ', '81.25     '], time:887.9194ms, swizzle: NOOP, TFLOPS: 2.48  (+0.00%)
                     f16x8pack(t8x8+bcf): ['80.125    ', '81.25     '], time:48.97031ms, swizzle: NOOP, TFLOPS: 44.91 (+1713.18%)
                f16x8pack(t8x8+bcf+dbuf): ['80.125    ', '81.25     '], time:49.23377ms, swizzle: NOOP, TFLOPS: 44.66
                f16x8pack(t8x8+k16+dbuf): ['80.125    ', '81.25     '], time:48.24690ms, swizzle: NOOP, TFLOPS: 45.58 (+1.50%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['80.4375   ', '81.8125   '], time:100.2594ms, swizzle: NOOP, TFLOPS: 21.93
                         f16wmma(mma4x2): ['80.4375   ', '81.8125   '], time:50.94373ms, swizzle: NOOP, TFLOPS: 43.17
                 f16wmma(mma4x2+warp2x4): ['80.4375   ', '81.8125   '], time:28.04174ms, swizzle: NOOP, TFLOPS: 78.42 (+72.05%)
            f16wmma(mma2x4+warp2x4+dbuf): ['80.4375   ', '81.8125   '], time:22.53372ms, swizzle: NOOP, TFLOPS: 97.59 (+24.44%)
          f16wmma(mma2x4+warp2x4+stage4): ['80.4375   ', '81.8125   '], time:22.54621ms, swizzle: NOOP, TFLOPS: 97.53
          f16wmma(mma2x4+warp2x4+stage3): ['80.4375   ', '81.8125   '], time:22.51112ms, swizzle: NOOP, TFLOPS: 97.69 (+0.10%)
          f16wmma(mma2x4+warp2x4+stage2): ['80.4375   ', '81.8125   '], time:22.34466ms, swizzle: NOOP, TFLOPS: 98.41 (+0.74%)
        f16wmma(mma2x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:22.47118ms, swizzle: NOOP, TFLOPS: 97.86
        f16wmma(mma2x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:22.63612ms, swizzle: NOOP, TFLOPS: 97.15
        f16wmma(mma2x4+...+stage2+dsmem): ['80.4375   ', '81.8125   '], time:22.29204ms, swizzle: NOOP, TFLOPS: 98.65 (+0.24%)
        f16wmma(mma4x4+...+stage4+dsmem): ['80.4375   ', '81.8125   '], time:20.57881ms, swizzle: NOOP, TFLOPS: 106.86(+8.33%)
        f16wmma(mma4x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:20.88322ms, swizzle: NOOP, TFLOPS: 105.30
        f16wmma(mma4x4+...+stage2+dsmem): ['80.4375   ', '81.8125   '], time:21.68486ms, swizzle: NOOP, TFLOPS: 101.41
      f16wmma(mma2x4+...+stage4+swizzle): ['80.4375   ', '81.8125   '], time:20.73450ms, swizzle: 2048, TFLOPS: 106.06
      f16wmma(mma2x4+...+stage3+swizzle): ['80.4375   ', '81.8125   '], time:20.47457ms, swizzle: 2048, TFLOPS: 107.40(+0.51%)
      f16wmma(mma2x4+...+stage2+swizzle): ['80.4375   ', '81.8125   '], time:20.17295ms, swizzle: 2048, TFLOPS: 109.01(+1.50%)
       f16wmma(...+stage3+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:20.67279ms, swizzle: 2048, TFLOPS: 106.37
       f16wmma(...+stage2+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:20.18678ms, swizzle: 2048, TFLOPS: 108.93
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:20.68634ms, swizzle: 2048, TFLOPS: 106.30
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:20.44701ms, swizzle: 2048, TFLOPS: 107.55
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:20.95601ms, swizzle: 2048, TFLOPS: 104.94
                             f16(cublas): ['80.4375   ', '81.8125   '], time:19.48888ms, swizzle: NOOP, TFLOPS: 112.83(+3.51%)
                                  f16_th: ['79.875    ', '81.875    '], time:19.39394ms, swizzle: NOOP, TFLOPS: 113.39(+0.49%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=2048
                              f16(naive): ['80.9375   ', '5.7734375 '], time:300.9500ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                     f16x8pack(t8x8+bcf): ['80.9375   ', '5.7734375 '], time:22.47061ms, swizzle: NOOP, TFLOPS: 48.93 (+1239.30%)
                f16x8pack(t8x8+bcf+dbuf): ['80.9375   ', '5.7734375 '], time:22.45101ms, swizzle: NOOP, TFLOPS: 48.97 (+0.09%)
                f16x8pack(t8x8+k16+dbuf): ['80.9375   ', '5.7734375 '], time:21.38516ms, swizzle: NOOP, TFLOPS: 51.41 (+4.98%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['81.1875   ', '5.9296875 '], time:46.29857ms, swizzle: NOOP, TFLOPS: 23.75
                         f16wmma(mma4x2): ['81.1875   ', '5.9296875 '], time:23.37560ms, swizzle: NOOP, TFLOPS: 47.04
                 f16wmma(mma4x2+warp2x4): ['81.1875   ', '5.9296875 '], time:13.68629ms, swizzle: NOOP, TFLOPS: 80.34 (+56.25%)
            f16wmma(mma2x4+warp2x4+dbuf): ['81.1875   ', '5.9296875 '], time:10.35296ms, swizzle: NOOP, TFLOPS: 106.20(+32.20%)
          f16wmma(mma2x4+warp2x4+stage4): ['81.1875   ', '5.9296875 '], time:10.51018ms, swizzle: NOOP, TFLOPS: 104.61
          f16wmma(mma2x4+warp2x4+stage3): ['81.1875   ', '5.9296875 '], time:10.28497ms, swizzle: NOOP, TFLOPS: 106.90(+0.66%)
          f16wmma(mma2x4+warp2x4+stage2): ['81.1875   ', '5.9296875 '], time:10.34665ms, swizzle: NOOP, TFLOPS: 106.27
        f16wmma(mma2x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:10.47251ms, swizzle: NOOP, TFLOPS: 104.99
        f16wmma(mma2x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:10.40322ms, swizzle: NOOP, TFLOPS: 105.69
        f16wmma(mma2x4+...+stage2+dsmem): ['81.1875   ', '5.9296875 '], time:10.30566ms, swizzle: NOOP, TFLOPS: 106.69
        f16wmma(mma4x4+...+stage4+dsmem): ['81.1875   ', '5.9296875 '], time:10.49680ms, swizzle: NOOP, TFLOPS: 104.75
        f16wmma(mma4x4+...+stage3+dsmem): ['81.1875   ', '5.9296875 '], time:10.71636ms, swizzle: NOOP, TFLOPS: 102.60
        f16wmma(mma4x4+...+stage2+dsmem): ['81.1875   ', '5.9296875 '], time:11.03622ms, swizzle: NOOP, TFLOPS: 99.63
      f16wmma(mma2x4+...+stage4+swizzle): ['81.1875   ', '5.9296875 '], time:10.60550ms, swizzle: 4096, TFLOPS: 103.67
      f16wmma(mma2x4+...+stage3+swizzle): ['81.1875   ', '5.9296875 '], time:10.48877ms, swizzle: 4096, TFLOPS: 104.83
      f16wmma(mma2x4+...+stage2+swizzle): ['81.1875   ', '5.9296875 '], time:10.33043ms, swizzle: 4096, TFLOPS: 106.43
       f16wmma(...+stage3+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:10.50045ms, swizzle: 4096, TFLOPS: 104.71
       f16wmma(...+stage2+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:10.26101ms, swizzle: 4096, TFLOPS: 107.15(+0.23%)
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:10.60705ms, swizzle: 4096, TFLOPS: 103.66
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:10.47563ms, swizzle: 4096, TFLOPS: 104.96
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['81.1875   ', '5.9296875 '], time:10.67864ms, swizzle: 4096, TFLOPS: 102.96
                             f16(cublas): ['81.1875   ', '5.9296875 '], time:9.836053ms, swizzle: NOOP, TFLOPS: 111.78(+4.32%)
                                  f16_th: ['80.875    ', '5.9140625 '], time:9.700632ms, swizzle: NOOP, TFLOPS: 113.34(+1.40%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=4096
                              f16(naive): ['40.09375  ', '-9.3671875'], time:917.6757ms, swizzle: NOOP, TFLOPS: 2.40  (+0.00%)
                     f16x8pack(t8x8+bcf): ['40.09375  ', '-9.3671875'], time:49.34096ms, swizzle: NOOP, TFLOPS: 44.57 (+1759.87%)
                f16x8pack(t8x8+bcf+dbuf): ['40.09375  ', '-9.3671875'], time:51.09326ms, swizzle: NOOP, TFLOPS: 43.04
                f16x8pack(t8x8+k16+dbuf): ['40.09375  ', '-9.3671875'], time:48.73661ms, swizzle: NOOP, TFLOPS: 45.12 (+1.24%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['40.53125  ', '-9.296875 '], time:151.7885ms, swizzle: NOOP, TFLOPS: 14.49
                         f16wmma(mma4x2): ['40.53125  ', '-9.296875 '], time:63.30149ms, swizzle: NOOP, TFLOPS: 34.74
                 f16wmma(mma4x2+warp2x4): ['40.53125  ', '-9.296875 '], time:28.39467ms, swizzle: NOOP, TFLOPS: 77.44 (+71.64%)
            f16wmma(mma2x4+warp2x4+dbuf): ['40.53125  ', '-9.296875 '], time:22.71130ms, swizzle: NOOP, TFLOPS: 96.83 (+25.02%)
          f16wmma(mma2x4+warp2x4+stage4): ['40.53125  ', '-9.296875 '], time:23.52900ms, swizzle: NOOP, TFLOPS: 93.46
          f16wmma(mma2x4+warp2x4+stage3): ['40.53125  ', '-9.296875 '], time:23.82099ms, swizzle: NOOP, TFLOPS: 92.31
          f16wmma(mma2x4+warp2x4+stage2): ['40.53125  ', '-9.296875 '], time:23.08320ms, swizzle: NOOP, TFLOPS: 95.27
        f16wmma(mma2x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:23.78501ms, swizzle: NOOP, TFLOPS: 92.45
        f16wmma(mma2x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:23.68180ms, swizzle: NOOP, TFLOPS: 92.86
        f16wmma(mma2x4+...+stage2+dsmem): ['40.53125  ', '-9.296875 '], time:22.67043ms, swizzle: NOOP, TFLOPS: 97.00 (+0.18%)
        f16wmma(mma4x4+...+stage4+dsmem): ['40.53125  ', '-9.296875 '], time:20.63658ms, swizzle: NOOP, TFLOPS: 106.56(+9.86%)
        f16wmma(mma4x4+...+stage3+dsmem): ['40.53125  ', '-9.296875 '], time:20.98727ms, swizzle: NOOP, TFLOPS: 104.78
        f16wmma(mma4x4+...+stage2+dsmem): ['40.53125  ', '-9.296875 '], time:21.62094ms, swizzle: NOOP, TFLOPS: 101.71
      f16wmma(mma2x4+...+stage4+swizzle): ['40.53125  ', '-9.296875 '], time:20.98352ms, swizzle: 4096, TFLOPS: 104.80
      f16wmma(mma2x4+...+stage3+swizzle): ['40.53125  ', '-9.296875 '], time:20.61495ms, swizzle: 4096, TFLOPS: 106.67(+0.10%)
      f16wmma(mma2x4+...+stage2+swizzle): ['40.53125  ', '-9.296875 '], time:20.32923ms, swizzle: 4096, TFLOPS: 108.17(+1.41%)
       f16wmma(...+stage3+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:20.67291ms, swizzle: 4096, TFLOPS: 106.37
       f16wmma(...+stage2+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:20.41170ms, swizzle: 4096, TFLOPS: 107.73
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:20.56102ms, swizzle: 4096, TFLOPS: 106.95
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:20.32101ms, swizzle: 4096, TFLOPS: 108.21(+0.04%)
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['40.53125  ', '-9.296875 '], time:20.79734ms, swizzle: 4096, TFLOPS: 105.74
                             f16(cublas): ['40.53125  ', '-9.296875 '], time:19.62115ms, swizzle: NOOP, TFLOPS: 112.07(+3.57%)
                                  f16_th: ['40.0625   ', '-9.2890625'], time:19.12159ms, swizzle: NOOP, TFLOPS: 115.00(+2.61%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=8192
                              f16(naive): ['80.125    ', '81.25     '], time:1832.812ms, swizzle: NOOP, TFLOPS: 2.40  (+0.00%)
                     f16x8pack(t8x8+bcf): ['80.125    ', '81.25     '], time:107.7742ms, swizzle: NOOP, TFLOPS: 40.81 (+1600.60%)
                f16x8pack(t8x8+bcf+dbuf): ['80.125    ', '81.25     '], time:108.6536ms, swizzle: NOOP, TFLOPS: 40.48
                f16x8pack(t8x8+k16+dbuf): ['80.125    ', '81.25     '], time:106.4913ms, swizzle: NOOP, TFLOPS: 41.30 (+1.20%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                          f16wmma(naive): ['80.4375   ', '81.8125   '], time:823.9610ms, swizzle: NOOP, TFLOPS: 5.34
                         f16wmma(mma4x2): ['80.4375   ', '81.8125   '], time:125.1207ms, swizzle: NOOP, TFLOPS: 35.15
                 f16wmma(mma4x2+warp2x4): ['80.4375   ', '81.8125   '], time:56.25746ms, swizzle: NOOP, TFLOPS: 78.18 (+89.29%)
            f16wmma(mma2x4+warp2x4+dbuf): ['80.4375   ', '81.8125   '], time:50.76479ms, swizzle: NOOP, TFLOPS: 86.64 (+10.82%)
          f16wmma(mma2x4+warp2x4+stage4): ['80.4375   ', '81.8125   '], time:50.01738ms, swizzle: NOOP, TFLOPS: 87.93 (+1.49%)
          f16wmma(mma2x4+warp2x4+stage3): ['80.4375   ', '81.8125   '], time:49.92475ms, swizzle: NOOP, TFLOPS: 88.09 (+0.19%)
          f16wmma(mma2x4+warp2x4+stage2): ['80.4375   ', '81.8125   '], time:50.08881ms, swizzle: NOOP, TFLOPS: 87.80
        f16wmma(mma2x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:49.96726ms, swizzle: NOOP, TFLOPS: 88.02
        f16wmma(mma2x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:49.89750ms, swizzle: NOOP, TFLOPS: 88.14 (+0.05%)
        f16wmma(mma2x4+...+stage2+dsmem): ['80.4375   ', '81.8125   '], time:50.08876ms, swizzle: NOOP, TFLOPS: 87.81
        f16wmma(mma4x4+...+stage4+dsmem): ['80.4375   ', '81.8125   '], time:40.74456ms, swizzle: NOOP, TFLOPS: 107.94(+22.46%)
        f16wmma(mma4x4+...+stage3+dsmem): ['80.4375   ', '81.8125   '], time:41.29490ms, swizzle: NOOP, TFLOPS: 106.50
        f16wmma(mma4x4+...+stage2+dsmem): ['80.4375   ', '81.8125   '], time:42.41952ms, swizzle: NOOP, TFLOPS: 103.68
      f16wmma(mma2x4+...+stage4+swizzle): ['80.4375   ', '81.8125   '], time:41.57793ms, swizzle: 4096, TFLOPS: 105.78
      f16wmma(mma2x4+...+stage3+swizzle): ['80.4375   ', '81.8125   '], time:41.13643ms, swizzle: 4096, TFLOPS: 106.91
      f16wmma(mma2x4+...+stage2+swizzle): ['80.4375   ', '81.8125   '], time:40.39962ms, swizzle: 4096, TFLOPS: 108.86(+0.85%)
       f16wmma(...+stage3+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:41.30423ms, swizzle: 4096, TFLOPS: 106.48
       f16wmma(...+stage2+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:40.79089ms, swizzle: 4096, TFLOPS: 107.82
    f16wmma(mma4x4+stage4+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:40.34731ms, swizzle: 4096, TFLOPS: 109.00(+0.13%)
    f16wmma(mma4x4+stage3+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:40.00172ms, swizzle: 4096, TFLOPS: 109.95(+0.86%)
    f16wmma(mma4x4+stage2+dsmem+swizzle): ['80.4375   ', '81.8125   '], time:40.97278ms, swizzle: 4096, TFLOPS: 107.34
                             f16(cublas): ['80.4375   ', '81.8125   '], time:38.84799ms, swizzle: NOOP, TFLOPS: 113.21(+2.97%)
                                  f16_th: ['79.875    ', '81.875    '], time:38.70694ms, swizzle: NOOP, TFLOPS: 113.62(+0.36%)
----------------------------------------------------------------------------------------------------------------------------------
```
