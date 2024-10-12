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
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_stage3(ensor Cores, Tile MMA/Warp, Copy Async, Stage3, Pad) 
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_stage4(ensor Cores, Tile MMA/Warp, Copy Async, Stage4, Pad) 
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

- SASS

```C
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

```bash
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=1024, K=256
                                           out_f16: ['2.49023438  ', '-22.21875   ', '-18.765625  '], time:0.607085ms
                                       out_f16(sk): ['2.49023438  ', '-22.21875   ', '-18.765625  '], time:0.456238ms
                            out_f16x4pack(t4x4bcf): ['2.49023438  ', '-22.21875   ', '-18.765625  '], time:0.071573ms
                         out_f16x4pack(t4x4offset): ['2.49023438  ', '-22.21875   ', '-18.765625  '], time:0.070262ms
                                 out_f16x4(t8x8sk): ['2.49023438  ', '-22.21875   ', '-18.765625  '], time:0.066423ms
                                out_f16x4(t8x8bcf): ['2.49023438  ', '-22.21875   ', '-18.765625  '], time:0.058770ms
                             out_f16x4pack(t8x8sk): ['2.49023438  ', '-22.21875   ', '-18.765625  '], time:0.058794ms
                                out_f16x4pack(bcf): ['2.49023438  ', '-22.21875   ', '-18.765625  '], time:0.054646ms
                         out_f16x4pack(bcf+offset): ['2.49023438  ', '-22.21875   ', '-18.765625  '], time:0.054574ms
                                out_f16x8pack(bcf): ['2.49023438  ', '-22.21875   ', '-18.765625  '], time:0.053334ms
                         out_f16x8pack(bcf+offset): ['2.49023438  ', '-22.21875   ', '-18.765625  '], time:0.052977ms
                           out_f16x8pack(bcf+dbuf): ['2.49023438  ', '-22.21875   ', '-18.765625  '], time:0.049996ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['2.49023438  ', '-22.21875   ', '-18.765625  '], time:0.046873ms
                    out_f16x8pack(k16+dbuf+offset): ['2.49023438  ', '-22.21875   ', '-18.765625  '], time:0.046849ms
                     out_f16x8pack(k16+dbuf+async): ['2.49023438  ', '-22.21875   ', '-18.765625  '], time:0.047898ms
                           out_f16x8pack(k32+dbuf): ['2.49023438  ', '-22.21875   ', '-18.765625  '], time:0.048304ms
                     out_f16x8pack(k32+dbuf+async): ['2.49023438  ', '-22.21875   ', '-18.765625  '], time:0.047326ms
                     out_f16x8pack(k32+dbuf+t16x8): ['2.49023438  ', '-22.21875   ', '-18.765625  '], time:0.050735ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['2.49023438  ', '-22.21875   ', '-18.765625  '], time:0.048423ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['2.49023438  ', '-22.203125  ', '-18.78125   '], time:0.096488ms
                               out_f16wmma(mma4x2): ['2.49023438  ', '-22.203125  ', '-18.78125   '], time:0.054073ms
                       out_f16wmma(mma4x2+warp2x4): ['2.49023438  ', '-22.203125  ', '-18.78125   '], time:0.044775ms
                 out_f16wmma(mma4x2+warp2x4+async): ['2.49023438  ', '-22.203125  ', '-18.78125   '], time:0.039029ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['2.49023438  ', '-22.203125  ', '-18.78125   '], time:0.050640ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['2.49023438  ', '-22.203125  ', '-18.78125   '], time:0.035882ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['2.49023438  ', '-22.203125  ', '-18.78125   '], time:0.040841ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['2.49023438  ', '-22.203125  ', '-18.78125   '], time:0.049567ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['2.49023438  ', '-22.203125  ', '-18.78125   '], time:0.043082ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['2.49023438  ', '-22.203125  ', '-18.78125   '], time:0.042033ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['2.49023438  ', '-22.203125  ', '-18.78125   '], time:0.039911ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['2.49023438  ', '-22.203125  ', '-18.78125   '], time:0.036979ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['2.49023438  ', '-22.203125  ', '-18.78125   '], time:0.037599ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['2.49023438  ', '-22.203125  ', '-18.78125   '], time:0.040388ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['2.49023438  ', '-22.203125  ', '-18.78125   '], time:0.042033ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['2.49023438  ', '-22.203125  ', '-18.78125   '], time:0.035667ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['2.49023438  ', '-22.203125  ', '-18.78125   '], time:0.035477ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['2.49023438  ', '-22.203125  ', '-18.78125   '], time:0.035453ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['2.49023438  ', '-22.203125  ', '-18.78125   '], time:0.036073ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['2.49023438  ', '-22.203125  ', '-18.78125   '], time:0.035095ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['2.49023438  ', '-22.203125  ', '-18.78125   '], time:0.034356ms
                                        out_f16_th: ['2.4921875   ', '-22.21875   ', '-18.765625  '], time:0.026560ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=1024, K=512
                                           out_f16: ['12.9453125  ', '0.20959473  ', '-0.60253906 '], time:1.202178ms
                                       out_f16(sk): ['12.9453125  ', '0.20959473  ', '-0.60253906 '], time:0.896049ms
                            out_f16x4pack(t4x4bcf): ['12.9453125  ', '0.20959473  ', '-0.60253906 '], time:0.140285ms
                         out_f16x4pack(t4x4offset): ['12.9453125  ', '0.20959473  ', '-0.60253906 '], time:0.137687ms
                                 out_f16x4(t8x8sk): ['12.9453125  ', '0.20959473  ', '-0.60253906 '], time:0.121593ms
                                out_f16x4(t8x8bcf): ['12.9453125  ', '0.20959473  ', '-0.60253906 '], time:0.112748ms
                             out_f16x4pack(t8x8sk): ['12.9453125  ', '0.20959473  ', '-0.60253906 '], time:0.111365ms
                                out_f16x4pack(bcf): ['12.9453125  ', '0.20959473  ', '-0.60253906 '], time:0.104547ms
                         out_f16x4pack(bcf+offset): ['12.9453125  ', '0.20959473  ', '-0.60253906 '], time:0.104165ms
                                out_f16x8pack(bcf): ['12.9453125  ', '0.20959473  ', '-0.60253906 '], time:0.101161ms
                         out_f16x8pack(bcf+offset): ['12.9453125  ', '0.20959473  ', '-0.60253906 '], time:0.100636ms
                           out_f16x8pack(bcf+dbuf): ['12.9453125  ', '0.20959473  ', '-0.60253906 '], time:0.095248ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['12.9453125  ', '0.20959473  ', '-0.60253906 '], time:0.088620ms
                    out_f16x8pack(k16+dbuf+offset): ['12.9453125  ', '0.20959473  ', '-0.60253906 '], time:0.088334ms
                     out_f16x8pack(k16+dbuf+async): ['12.9453125  ', '0.20959473  ', '-0.60253906 '], time:0.090837ms
                           out_f16x8pack(k32+dbuf): ['12.9453125  ', '0.20959473  ', '-0.60253906 '], time:0.091362ms
                     out_f16x8pack(k32+dbuf+async): ['12.9453125  ', '0.20959473  ', '-0.60253906 '], time:0.089812ms
                     out_f16x8pack(k32+dbuf+t16x8): ['12.9453125  ', '0.20959473  ', '-0.60253906 '], time:0.095892ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['12.9453125  ', '0.20959473  ', '-0.60253906 '], time:0.091362ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['12.9140625  ', '0.25048828  ', '-0.53125    '], time:0.188065ms
                               out_f16wmma(mma4x2): ['12.9140625  ', '0.25048828  ', '-0.53125    '], time:0.100970ms
                       out_f16wmma(mma4x2+warp2x4): ['12.9140625  ', '0.25048828  ', '-0.53125    '], time:0.075102ms
                 out_f16wmma(mma4x2+warp2x4+async): ['12.9140625  ', '0.25048828  ', '-0.53125    '], time:0.065017ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['12.9140625  ', '0.25048828  ', '-0.53125    '], time:0.093865ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['12.9140625  ', '0.25048828  ', '-0.53125    '], time:0.058627ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['12.9140625  ', '0.25048828  ', '-0.53125    '], time:0.068450ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['12.9140625  ', '0.25048828  ', '-0.53125    '], time:0.085521ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['12.9140625  ', '0.25048828  ', '-0.53125    '], time:0.068545ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['12.9140625  ', '0.25048828  ', '-0.53125    '], time:0.068665ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['12.9140625  ', '0.25048828  ', '-0.53125    '], time:0.065970ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['12.9140625  ', '0.25048828  ', '-0.53125    '], time:0.060391ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['12.9140625  ', '0.25048828  ', '-0.53125    '], time:0.062823ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['12.9140625  ', '0.25048828  ', '-0.53125    '], time:0.067234ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['12.9140625  ', '0.25048828  ', '-0.53125    '], time:0.068736ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['12.9140625  ', '0.25048828  ', '-0.53125    '], time:0.057173ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['12.9140625  ', '0.25048828  ', '-0.53125    '], time:0.057673ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['12.9140625  ', '0.25048828  ', '-0.53125    '], time:0.056839ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['12.9140625  ', '0.25048828  ', '-0.53125    '], time:0.057197ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['12.9140625  ', '0.25048828  ', '-0.53125    '], time:0.055861ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['12.9140625  ', '0.25048828  ', '-0.53125    '], time:0.055289ms
                                        out_f16_th: ['12.9140625  ', '0.24450684  ', '-0.53564453 '], time:0.046158ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=1024, K=1024
                                           out_f16: ['-5.43359375 ', '43.71875    ', '28.484375   '], time:2.384853ms
                                       out_f16(sk): ['-5.43359375 ', '43.71875    ', '28.484375   '], time:1.775241ms
                            out_f16x4pack(t4x4bcf): ['-5.43359375 ', '43.71875    ', '28.484375   '], time:0.277424ms
                         out_f16x4pack(t4x4offset): ['-5.43359375 ', '43.71875    ', '28.484375   '], time:0.272393ms
                                 out_f16x4(t8x8sk): ['-5.43359375 ', '43.71875    ', '28.484375   '], time:0.231647ms
                                out_f16x4(t8x8bcf): ['-5.43359375 ', '43.71875    ', '28.484375   '], time:0.220227ms
                             out_f16x4pack(t8x8sk): ['-5.43359375 ', '43.71875    ', '28.484375   '], time:0.216460ms
                                out_f16x4pack(bcf): ['-5.43359375 ', '43.71875    ', '28.484375   '], time:0.203872ms
                         out_f16x4pack(bcf+offset): ['-5.43359375 ', '43.71875    ', '28.484375   '], time:0.203276ms
                                out_f16x8pack(bcf): ['-5.43359375 ', '43.71875    ', '28.484375   '], time:0.197387ms
                         out_f16x8pack(bcf+offset): ['-5.43359375 ', '43.71875    ', '28.484375   '], time:0.196004ms
                           out_f16x8pack(bcf+dbuf): ['-5.43359375 ', '43.71875    ', '28.484375   '], time:0.185418ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['-5.43359375 ', '43.71875    ', '28.484375   '], time:0.172114ms
                    out_f16x8pack(k16+dbuf+offset): ['-5.43359375 ', '43.71875    ', '28.484375   '], time:0.171638ms
                     out_f16x8pack(k16+dbuf+async): ['-5.43359375 ', '43.71875    ', '28.484375   '], time:0.176215ms
                           out_f16x8pack(k32+dbuf): ['-5.43359375 ', '43.71875    ', '28.484375   '], time:0.177431ms
                     out_f16x8pack(k32+dbuf+async): ['-5.43359375 ', '43.71875    ', '28.484375   '], time:0.173879ms
                     out_f16x8pack(k32+dbuf+t16x8): ['-5.43359375 ', '43.71875    ', '28.484375   '], time:0.187063ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['-5.43359375 ', '43.71875    ', '28.484375   '], time:0.177073ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['-5.34765625 ', '43.40625    ', '28.484375   '], time:0.372100ms
                               out_f16wmma(mma4x2): ['-5.34765625 ', '43.40625    ', '28.484375   '], time:0.194550ms
                       out_f16wmma(mma4x2+warp2x4): ['-5.34765625 ', '43.40625    ', '28.484375   '], time:0.135612ms
                 out_f16wmma(mma4x2+warp2x4+async): ['-5.34765625 ', '43.40625    ', '28.484375   '], time:0.117707ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['-5.34765625 ', '43.40625    ', '28.484375   '], time:0.176740ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['-5.34765625 ', '43.40625    ', '28.484375   '], time:0.104165ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['-5.34765625 ', '43.40625    ', '28.484375   '], time:0.123429ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['-5.34765625 ', '43.40625    ', '28.484375   '], time:0.157404ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['-5.34765625 ', '43.40625    ', '28.484375   '], time:0.123239ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['-5.34765625 ', '43.40625    ', '28.484375   '], time:0.123096ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['-5.34765625 ', '43.40625    ', '28.484375   '], time:0.118470ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-5.34765625 ', '43.40625    ', '28.484375   '], time:0.107288ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['-5.34765625 ', '43.40625    ', '28.484375   '], time:0.113201ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['-5.34765625 ', '43.40625    ', '28.484375   '], time:0.121427ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['-5.34765625 ', '43.40625    ', '28.484375   '], time:0.123143ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['-5.34765625 ', '43.40625    ', '28.484375   '], time:0.099945ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['-5.34765625 ', '43.40625    ', '28.484375   '], time:0.101972ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['-5.34765625 ', '43.40625    ', '28.484375   '], time:0.099802ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['-5.34765625 ', '43.40625    ', '28.484375   '], time:0.099492ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['-5.34765625 ', '43.40625    ', '28.484375   '], time:0.097609ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['-5.34765625 ', '43.40625    ', '28.484375   '], time:0.097060ms
                                        out_f16_th: ['-5.3359375  ', '43.375      ', '28.46875    '], time:0.085664ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=2048, K=256
                                           out_f16: ['19.25       ', '-16.5625    ', '10.71875    '], time:1.206160ms
                                       out_f16(sk): ['19.25       ', '-16.5625    ', '10.71875    '], time:0.905156ms
                            out_f16x4pack(t4x4bcf): ['19.25       ', '-16.5625    ', '10.71875    '], time:0.132132ms
                         out_f16x4pack(t4x4offset): ['19.25       ', '-16.5625    ', '10.71875    '], time:0.129700ms
                                 out_f16x4(t8x8sk): ['19.25       ', '-16.5625    ', '10.71875    '], time:0.129271ms
                                out_f16x4(t8x8bcf): ['19.25       ', '-16.5625    ', '10.71875    '], time:0.109172ms
                             out_f16x4pack(t8x8sk): ['19.25       ', '-16.5625    ', '10.71875    '], time:0.110769ms
                                out_f16x4pack(bcf): ['19.25       ', '-16.5625    ', '10.71875    '], time:0.100756ms
                         out_f16x4pack(bcf+offset): ['19.25       ', '-16.5625    ', '10.71875    '], time:0.100732ms
                                out_f16x8pack(bcf): ['19.25       ', '-16.5625    ', '10.71875    '], time:0.097895ms
                         out_f16x8pack(bcf+offset): ['19.25       ', '-16.5625    ', '10.71875    '], time:0.097179ms
                           out_f16x8pack(bcf+dbuf): ['19.25       ', '-16.5625    ', '10.71875    '], time:0.094295ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['19.25       ', '-16.5625    ', '10.71875    '], time:0.089240ms
                    out_f16x8pack(k16+dbuf+offset): ['19.25       ', '-16.5625    ', '10.71875    '], time:0.089264ms
                     out_f16x8pack(k16+dbuf+async): ['19.25       ', '-16.5625    ', '10.71875    '], time:0.090885ms
                           out_f16x8pack(k32+dbuf): ['19.25       ', '-16.5625    ', '10.71875    '], time:0.090837ms
                     out_f16x8pack(k32+dbuf+async): ['19.25       ', '-16.5625    ', '10.71875    '], time:0.089550ms
                     out_f16x8pack(k32+dbuf+t16x8): ['19.25       ', '-16.5625    ', '10.71875    '], time:0.094700ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['19.25       ', '-16.5625    ', '10.71875    '], time:0.090504ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['19.28125    ', '-16.75      ', '10.75       '], time:0.189352ms
                               out_f16wmma(mma4x2): ['19.28125    ', '-16.75      ', '10.75       '], time:0.102592ms
                       out_f16wmma(mma4x2+warp2x4): ['19.28125    ', '-16.75      ', '10.75       '], time:0.081158ms
                 out_f16wmma(mma4x2+warp2x4+async): ['19.28125    ', '-16.75      ', '10.75       '], time:0.073314ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['19.28125    ', '-16.75      ', '10.75       '], time:0.083399ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['19.28125    ', '-16.75      ', '10.75       '], time:0.066710ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['19.28125    ', '-16.75      ', '10.75       '], time:0.072670ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['19.28125    ', '-16.75      ', '10.75       '], time:0.095248ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['19.28125    ', '-16.75      ', '10.75       '], time:0.076342ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['19.28125    ', '-16.75      ', '10.75       '], time:0.075817ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['19.28125    ', '-16.75      ', '10.75       '], time:0.073934ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['19.28125    ', '-16.75      ', '10.75       '], time:0.067544ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['19.28125    ', '-16.75      ', '10.75       '], time:0.070357ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['19.28125    ', '-16.75      ', '10.75       '], time:0.074458ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['19.28125    ', '-16.75      ', '10.75       '], time:0.076389ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['19.28125    ', '-16.75      ', '10.75       '], time:0.065112ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['19.28125    ', '-16.75      ', '10.75       '], time:0.066447ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['19.28125    ', '-16.75      ', '10.75       '], time:0.065112ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['19.28125    ', '-16.75      ', '10.75       '], time:0.066400ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['19.28125    ', '-16.75      ', '10.75       '], time:0.064301ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['19.28125    ', '-16.75      ', '10.75       '], time:0.063491ms
                                        out_f16_th: ['19.28125    ', '-16.703125  ', '10.75       '], time:0.048089ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=2048, K=512
                                           out_f16: ['11.375      ', '9.1484375   ', '12.0859375  '], time:2.385569ms
                                       out_f16(sk): ['11.375      ', '9.1484375   ', '12.0859375  '], time:1.779795ms
                            out_f16x4pack(t4x4bcf): ['11.375      ', '9.1484375   ', '12.0859375  '], time:0.260401ms
                         out_f16x4pack(t4x4offset): ['11.375      ', '9.1484375   ', '12.0859375  '], time:0.255466ms
                                 out_f16x4(t8x8sk): ['11.375      ', '9.1484375   ', '12.0859375  '], time:0.235415ms
                                out_f16x4(t8x8bcf): ['11.375      ', '9.1484375   ', '12.0859375  '], time:0.212288ms
                             out_f16x4pack(t8x8sk): ['11.375      ', '9.1484375   ', '12.0859375  '], time:0.210261ms
                                out_f16x4pack(bcf): ['11.375      ', '9.1484375   ', '12.0859375  '], time:0.195289ms
                         out_f16x4pack(bcf+offset): ['11.375      ', '9.1484375   ', '12.0859375  '], time:0.195241ms
                                out_f16x8pack(bcf): ['11.375      ', '9.1484375   ', '12.0859375  '], time:0.189018ms
                         out_f16x8pack(bcf+offset): ['11.375      ', '9.1484375   ', '12.0859375  '], time:0.188112ms
                           out_f16x8pack(bcf+dbuf): ['11.375      ', '9.1484375   ', '12.0859375  '], time:0.182366ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['11.375      ', '9.1484375   ', '12.0859375  '], time:0.171685ms
                    out_f16x8pack(k16+dbuf+offset): ['11.375      ', '9.1484375   ', '12.0859375  '], time:0.171161ms
                     out_f16x8pack(k16+dbuf+async): ['11.375      ', '9.1484375   ', '12.0859375  '], time:0.175381ms
                           out_f16x8pack(k32+dbuf): ['11.375      ', '9.1484375   ', '12.0859375  '], time:0.175858ms
                     out_f16x8pack(k32+dbuf+async): ['11.375      ', '9.1484375   ', '12.0859375  '], time:0.172973ms
                     out_f16x8pack(k32+dbuf+t16x8): ['11.375      ', '9.1484375   ', '12.0859375  '], time:0.182748ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['11.375      ', '9.1484375   ', '12.0859375  '], time:0.174761ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['11.3671875  ', '9.1328125   ', '12.0390625  '], time:0.371408ms
                               out_f16wmma(mma4x2): ['11.3671875  ', '9.1328125   ', '12.0390625  '], time:0.193667ms
                       out_f16wmma(mma4x2+warp2x4): ['11.3671875  ', '9.1328125   ', '12.0390625  '], time:0.137043ms
                 out_f16wmma(mma4x2+warp2x4+async): ['11.3671875  ', '9.1328125   ', '12.0390625  '], time:0.125194ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['11.3671875  ', '9.1328125   ', '12.0390625  '], time:0.145102ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['11.3671875  ', '9.1328125   ', '12.0390625  '], time:0.109696ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['11.3671875  ', '9.1328125   ', '12.0390625  '], time:0.124764ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['11.3671875  ', '9.1328125   ', '12.0390625  '], time:0.166702ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['11.3671875  ', '9.1328125   ', '12.0390625  '], time:0.129294ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['11.3671875  ', '9.1328125   ', '12.0390625  '], time:0.129128ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['11.3671875  ', '9.1328125   ', '12.0390625  '], time:0.124359ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['11.3671875  ', '9.1328125   ', '12.0390625  '], time:0.113058ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['11.3671875  ', '9.1328125   ', '12.0390625  '], time:0.119114ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['11.3671875  ', '9.1328125   ', '12.0390625  '], time:0.126457ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['11.3671875  ', '9.1328125   ', '12.0390625  '], time:0.129580ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['11.3671875  ', '9.1328125   ', '12.0390625  '], time:0.106955ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['11.3671875  ', '9.1328125   ', '12.0390625  '], time:0.110793ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['11.3671875  ', '9.1328125   ', '12.0390625  '], time:0.106835ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['11.3671875  ', '9.1328125   ', '12.0390625  '], time:0.108075ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['11.3671875  ', '9.1328125   ', '12.0390625  '], time:0.105524ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['11.3671875  ', '9.1328125   ', '12.0390625  '], time:0.103855ms
                                        out_f16_th: ['11.3515625  ', '9.140625    ', '12.0234375  '], time:0.087261ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=2048, K=1024
                                           out_f16: ['67.1875     ', '-6.73828125 ', '-24.609375  '], time:4.745054ms
                                       out_f16(sk): ['67.1875     ', '-6.73828125 ', '-24.609375  '], time:3.529692ms
                            out_f16x4pack(t4x4bcf): ['67.1875     ', '-6.73828125 ', '-24.609375  '], time:0.516963ms
                         out_f16x4pack(t4x4offset): ['67.1875     ', '-6.73828125 ', '-24.609375  '], time:0.507259ms
                                 out_f16x4(t8x8sk): ['67.1875     ', '-6.73828125 ', '-24.609375  '], time:0.446701ms
                                out_f16x4(t8x8bcf): ['67.1875     ', '-6.73828125 ', '-24.609375  '], time:0.418377ms
                             out_f16x4pack(t8x8sk): ['67.1875     ', '-6.73828125 ', '-24.609375  '], time:0.409269ms
                                out_f16x4pack(bcf): ['67.1875     ', '-6.73828125 ', '-24.609375  '], time:0.384378ms
                         out_f16x4pack(bcf+offset): ['67.1875     ', '-6.73828125 ', '-24.609375  '], time:0.383449ms
                                out_f16x8pack(bcf): ['67.1875     ', '-6.73828125 ', '-24.609375  '], time:0.371885ms
                         out_f16x8pack(bcf+offset): ['67.1875     ', '-6.73828125 ', '-24.609375  '], time:0.368834ms
                           out_f16x8pack(bcf+dbuf): ['67.1875     ', '-6.73828125 ', '-24.609375  '], time:0.358033ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['67.1875     ', '-6.73828125 ', '-24.609375  '], time:0.336599ms
                    out_f16x8pack(k16+dbuf+offset): ['67.1875     ', '-6.73828125 ', '-24.609375  '], time:0.336289ms
                     out_f16x8pack(k16+dbuf+async): ['67.1875     ', '-6.73828125 ', '-24.609375  '], time:0.344133ms
                           out_f16x8pack(k32+dbuf): ['67.1875     ', '-6.73828125 ', '-24.609375  '], time:0.345802ms
                     out_f16x8pack(k32+dbuf+async): ['67.1875     ', '-6.73828125 ', '-24.609375  '], time:0.339437ms
                     out_f16x8pack(k32+dbuf+t16x8): ['67.1875     ', '-6.73828125 ', '-24.609375  '], time:0.359678ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['67.1875     ', '-6.73828125 ', '-24.609375  '], time:0.342917ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['67.125      ', '-6.72265625 ', '-24.875     '], time:0.735950ms
                               out_f16wmma(mma4x2): ['67.125      ', '-6.72265625 ', '-24.875     '], time:0.376177ms
                       out_f16wmma(mma4x2+warp2x4): ['67.125      ', '-6.72265625 ', '-24.875     '], time:0.247979ms
                 out_f16wmma(mma4x2+warp2x4+async): ['67.125      ', '-6.72265625 ', '-24.875     '], time:0.228381ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['67.125      ', '-6.72265625 ', '-24.875     '], time:0.272679ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['67.125      ', '-6.72265625 ', '-24.875     '], time:0.196624ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['67.125      ', '-6.72265625 ', '-24.875     '], time:0.229931ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['67.125      ', '-6.72265625 ', '-24.875     '], time:0.310135ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['67.125      ', '-6.72265625 ', '-24.875     '], time:0.237274ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['67.125      ', '-6.72265625 ', '-24.875     '], time:0.236988ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['67.125      ', '-6.72265625 ', '-24.875     '], time:0.226831ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['67.125      ', '-6.72265625 ', '-24.875     '], time:0.204563ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['67.125      ', '-6.72265625 ', '-24.875     '], time:0.218034ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['67.125      ', '-6.72265625 ', '-24.875     '], time:0.231338ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['67.125      ', '-6.72265625 ', '-24.875     '], time:0.237036ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['67.125      ', '-6.72265625 ', '-24.875     '], time:0.191450ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['67.125      ', '-6.72265625 ', '-24.875     '], time:0.199556ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['67.125      ', '-6.72265625 ', '-24.875     '], time:0.190020ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['67.125      ', '-6.72265625 ', '-24.875     '], time:0.190830ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['67.125      ', '-6.72265625 ', '-24.875     '], time:0.187445ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['67.125      ', '-6.72265625 ', '-24.875     '], time:0.186157ms
                                        out_f16_th: ['67.25       ', '-6.6796875  ', '-24.796875  '], time:0.169420ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=256
                                           out_f16: ['11.0625     ', '10.03125    ', '-14.8515625 '], time:2.403116ms
                                       out_f16(sk): ['11.0625     ', '10.03125    ', '-14.8515625 '], time:1.799965ms
                            out_f16x4pack(t4x4bcf): ['11.0625     ', '10.03125    ', '-14.8515625 '], time:0.253272ms
                         out_f16x4pack(t4x4offset): ['11.0625     ', '10.03125    ', '-14.8515625 '], time:0.248456ms
                                 out_f16x4(t8x8sk): ['11.0625     ', '10.03125    ', '-14.8515625 '], time:0.231075ms
                                out_f16x4(t8x8bcf): ['11.0625     ', '10.03125    ', '-14.8515625 '], time:0.208807ms
                             out_f16x4pack(t8x8sk): ['11.0625     ', '10.03125    ', '-14.8515625 '], time:0.207376ms
                                out_f16x4pack(bcf): ['11.0625     ', '10.03125    ', '-14.8515625 '], time:0.192118ms
                         out_f16x4pack(bcf+offset): ['11.0625     ', '10.03125    ', '-14.8515625 '], time:0.191569ms
                                out_f16x8pack(bcf): ['11.0625     ', '10.03125    ', '-14.8515625 '], time:0.185370ms
                         out_f16x8pack(bcf+offset): ['11.0625     ', '10.03125    ', '-14.8515625 '], time:0.183868ms
                           out_f16x8pack(bcf+dbuf): ['11.0625     ', '10.03125    ', '-14.8515625 '], time:0.181317ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['11.0625     ', '10.03125    ', '-14.8515625 '], time:0.172424ms
                    out_f16x8pack(k16+dbuf+offset): ['11.0625     ', '10.03125    ', '-14.8515625 '], time:0.172448ms
                     out_f16x8pack(k16+dbuf+async): ['11.0625     ', '10.03125    ', '-14.8515625 '], time:0.175357ms
                           out_f16x8pack(k32+dbuf): ['11.0625     ', '10.03125    ', '-14.8515625 '], time:0.175142ms
                     out_f16x8pack(k32+dbuf+async): ['11.0625     ', '10.03125    ', '-14.8515625 '], time:0.172710ms
                     out_f16x8pack(k32+dbuf+t16x8): ['11.0625     ', '10.03125    ', '-14.8515625 '], time:0.181222ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['11.0625     ', '10.03125    ', '-14.8515625 '], time:0.174165ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['11.1015625  ', '10.0078125  ', '-14.8203125 '], time:0.377846ms
                               out_f16wmma(mma4x2): ['11.1015625  ', '10.0078125  ', '-14.8203125 '], time:0.200582ms
                       out_f16wmma(mma4x2+warp2x4): ['11.1015625  ', '10.0078125  ', '-14.8203125 '], time:0.155926ms
                 out_f16wmma(mma4x2+warp2x4+async): ['11.1015625  ', '10.0078125  ', '-14.8203125 '], time:0.134492ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['11.1015625  ', '10.0078125  ', '-14.8203125 '], time:0.157261ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['11.1015625  ', '10.0078125  ', '-14.8203125 '], time:0.118852ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['11.1015625  ', '10.0078125  ', '-14.8203125 '], time:0.129580ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['11.1015625  ', '10.0078125  ', '-14.8203125 '], time:0.179863ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['11.1015625  ', '10.0078125  ', '-14.8203125 '], time:0.137353ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['11.1015625  ', '10.0078125  ', '-14.8203125 '], time:0.137568ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['11.1015625  ', '10.0078125  ', '-14.8203125 '], time:0.134587ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['11.1015625  ', '10.0078125  ', '-14.8203125 '], time:0.120997ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['11.1015625  ', '10.0078125  ', '-14.8203125 '], time:0.127792ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['11.1015625  ', '10.0078125  ', '-14.8203125 '], time:0.135303ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['11.1015625  ', '10.0078125  ', '-14.8203125 '], time:0.137353ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['11.1015625  ', '10.0078125  ', '-14.8203125 '], time:0.116706ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['11.1015625  ', '10.0078125  ', '-14.8203125 '], time:0.121832ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['11.1015625  ', '10.0078125  ', '-14.8203125 '], time:0.116873ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['11.1015625  ', '10.0078125  ', '-14.8203125 '], time:0.116920ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['11.1015625  ', '10.0078125  ', '-14.8203125 '], time:0.115156ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['11.1015625  ', '10.0078125  ', '-14.8203125 '], time:0.111985ms
                                        out_f16_th: ['11.0859375  ', '10.046875   ', '-14.8203125 '], time:0.090480ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=512
                                           out_f16: ['-0.01989746 ', '4.171875    ', '-10.59375   '], time:4.755235ms
                                       out_f16(sk): ['-0.01989746 ', '4.171875    ', '-10.59375   '], time:3.541994ms
                            out_f16x4pack(t4x4bcf): ['-0.01989746 ', '4.171875    ', '-10.59375   '], time:0.500965ms
                         out_f16x4pack(t4x4offset): ['-0.01989746 ', '4.171875    ', '-10.59375   '], time:0.491905ms
                                 out_f16x4(t8x8sk): ['-0.01989746 ', '4.171875    ', '-10.59375   '], time:0.441146ms
                                out_f16x4(t8x8bcf): ['-0.01989746 ', '4.171875    ', '-10.59375   '], time:0.409985ms
                             out_f16x4pack(t8x8sk): ['-0.01989746 ', '4.171875    ', '-10.59375   '], time:0.401831ms
                                out_f16x4pack(bcf): ['-0.01989746 ', '4.171875    ', '-10.59375   '], time:0.375915ms
                         out_f16x4pack(bcf+offset): ['-0.01989746 ', '4.171875    ', '-10.59375   '], time:0.374866ms
                                out_f16x8pack(bcf): ['-0.01989746 ', '4.171875    ', '-10.59375   '], time:0.362110ms
                         out_f16x8pack(bcf+offset): ['-0.01989746 ', '4.171875    ', '-10.59375   '], time:0.359201ms
                           out_f16x8pack(bcf+dbuf): ['-0.01989746 ', '4.171875    ', '-10.59375   '], time:0.356174ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['-0.01989746 ', '4.171875    ', '-10.59375   '], time:0.337267ms
                    out_f16x8pack(k16+dbuf+offset): ['-0.01989746 ', '4.171875    ', '-10.59375   '], time:0.336981ms
                     out_f16x8pack(k16+dbuf+async): ['-0.01989746 ', '4.171875    ', '-10.59375   '], time:0.344467ms
                           out_f16x8pack(k32+dbuf): ['-0.01989746 ', '4.171875    ', '-10.59375   '], time:0.343966ms
                     out_f16x8pack(k32+dbuf+async): ['-0.01989746 ', '4.171875    ', '-10.59375   '], time:0.339818ms
                     out_f16x8pack(k32+dbuf+t16x8): ['-0.01989746 ', '4.171875    ', '-10.59375   '], time:0.356150ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['-0.01989746 ', '4.171875    ', '-10.59375   '], time:0.342083ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['-0.12597656 ', '4.12890625  ', '-10.4921875 '], time:0.742912ms
                               out_f16wmma(mma4x2): ['-0.12597656 ', '4.12890625  ', '-10.4921875 '], time:0.380778ms
                       out_f16wmma(mma4x2+warp2x4): ['-0.12597656 ', '4.12890625  ', '-10.4921875 '], time:0.267506ms
                 out_f16wmma(mma4x2+warp2x4+async): ['-0.12597656 ', '4.12890625  ', '-10.4921875 '], time:0.236797ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['-0.12597656 ', '4.12890625  ', '-10.4921875 '], time:0.280452ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['-0.12597656 ', '4.12890625  ', '-10.4921875 '], time:0.202537ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['-0.12597656 ', '4.12890625  ', '-10.4921875 '], time:0.232458ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['-0.12597656 ', '4.12890625  ', '-10.4921875 '], time:0.321436ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['-0.12597656 ', '4.12890625  ', '-10.4921875 '], time:0.243926ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['-0.12597656 ', '4.12890625  ', '-10.4921875 '], time:0.243282ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['-0.12597656 ', '4.12890625  ', '-10.4921875 '], time:0.236368ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-0.12597656 ', '4.12890625  ', '-10.4921875 '], time:0.212836ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['-0.12597656 ', '4.12890625  ', '-10.4921875 '], time:0.224638ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['-0.12597656 ', '4.12890625  ', '-10.4921875 '], time:0.237775ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['-0.12597656 ', '4.12890625  ', '-10.4921875 '], time:0.243926ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['-0.12597656 ', '4.12890625  ', '-10.4921875 '], time:0.200939ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['-0.12597656 ', '4.12890625  ', '-10.4921875 '], time:0.211167ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['-0.12597656 ', '4.12890625  ', '-10.4921875 '], time:0.199771ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['-0.12597656 ', '4.12890625  ', '-10.4921875 '], time:0.198579ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['-0.12597656 ', '4.12890625  ', '-10.4921875 '], time:0.196958ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['-0.12597656 ', '4.12890625  ', '-10.4921875 '], time:0.191498ms
                                        out_f16_th: ['-0.13024902 ', '4.16015625  ', '-10.546875  '], time:0.169277ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=1024
                                           out_f16: ['17.953125   ', '13.109375   ', '-22.78125   '], time:9.460902ms
                                       out_f16(sk): ['17.953125   ', '13.109375   ', '-22.78125   '], time:7.035208ms
                            out_f16x4pack(t4x4bcf): ['17.953125   ', '13.109375   ', '-22.78125   '], time:0.996184ms
                         out_f16x4pack(t4x4offset): ['17.953125   ', '13.109375   ', '-22.78125   '], time:0.978017ms
                                 out_f16x4(t8x8sk): ['17.953125   ', '13.109375   ', '-22.78125   '], time:0.856137ms
                                out_f16x4(t8x8bcf): ['17.953125   ', '13.109375   ', '-22.78125   '], time:0.811291ms
                             out_f16x4pack(t8x8sk): ['17.953125   ', '13.109375   ', '-22.78125   '], time:0.788999ms
                                out_f16x4pack(bcf): ['17.953125   ', '13.109375   ', '-22.78125   '], time:0.742698ms
                         out_f16x4pack(bcf+offset): ['17.953125   ', '13.109375   ', '-22.78125   '], time:0.739908ms
                                out_f16x8pack(bcf): ['17.953125   ', '13.109375   ', '-22.78125   '], time:0.713420ms
                         out_f16x8pack(bcf+offset): ['17.953125   ', '13.109375   ', '-22.78125   '], time:0.708079ms
                           out_f16x8pack(bcf+dbuf): ['17.953125   ', '13.109375   ', '-22.78125   '], time:0.703835ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['17.953125   ', '13.109375   ', '-22.78125   '], time:0.666380ms
                    out_f16x8pack(k16+dbuf+offset): ['17.953125   ', '13.109375   ', '-22.78125   '], time:0.665474ms
                     out_f16x8pack(k16+dbuf+async): ['17.953125   ', '13.109375   ', '-22.78125   '], time:0.680828ms
                           out_f16x8pack(k32+dbuf): ['17.953125   ', '13.109375   ', '-22.78125   '], time:0.680208ms
                     out_f16x8pack(k32+dbuf+async): ['17.953125   ', '13.109375   ', '-22.78125   '], time:0.671458ms
                     out_f16x8pack(k32+dbuf+t16x8): ['17.953125   ', '13.109375   ', '-22.78125   '], time:0.705290ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['17.953125   ', '13.109375   ', '-22.78125   '], time:0.675941ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['18.0625     ', '13.21875    ', '-22.78125   '], time:1.470852ms
                               out_f16wmma(mma4x2): ['18.0625     ', '13.21875    ', '-22.78125   '], time:0.744677ms
                       out_f16wmma(mma4x2+warp2x4): ['18.0625     ', '13.21875    ', '-22.78125   '], time:0.488257ms
                 out_f16wmma(mma4x2+warp2x4+async): ['18.0625     ', '13.21875    ', '-22.78125   '], time:0.443721ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['18.0625     ', '13.21875    ', '-22.78125   '], time:0.531912ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['18.0625     ', '13.21875    ', '-22.78125   '], time:0.373864ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['18.0625     ', '13.21875    ', '-22.78125   '], time:0.438857ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['18.0625     ', '13.21875    ', '-22.78125   '], time:0.608087ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['18.0625     ', '13.21875    ', '-22.78125   '], time:0.463438ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['18.0625     ', '13.21875    ', '-22.78125   '], time:0.459909ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['18.0625     ', '13.21875    ', '-22.78125   '], time:0.441074ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['18.0625     ', '13.21875    ', '-22.78125   '], time:0.394320ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['18.0625     ', '13.21875    ', '-22.78125   '], time:0.422692ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['18.0625     ', '13.21875    ', '-22.78125   '], time:0.446177ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['18.0625     ', '13.21875    ', '-22.78125   '], time:0.460100ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['18.0625     ', '13.21875    ', '-22.78125   '], time:0.370955ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['18.0625     ', '13.21875    ', '-22.78125   '], time:0.389242ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['18.0625     ', '13.21875    ', '-22.78125   '], time:0.366330ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['18.0625     ', '13.21875    ', '-22.78125   '], time:0.363851ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['18.0625     ', '13.21875    ', '-22.78125   '], time:0.362515ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['18.0625     ', '13.21875    ', '-22.78125   '], time:0.355983ms
                                        out_f16_th: ['18.015625   ', '13.203125   ', '-22.859375  '], time:0.334716ms
------------------------------------------------------------------------------------------------------------------------
```
