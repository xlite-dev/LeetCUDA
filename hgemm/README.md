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
                                                       M=4096, N=1024, K=1024
                                           out_f16: ['-12.8125    ', '3.22070312  ', '-33.96875   '], time:2.384830ms
                                       out_f16(sk): ['-12.8125    ', '3.22070312  ', '-33.96875   '], time:1.775217ms
                            out_f16x4pack(t4x4bcf): ['-12.8125    ', '3.22070312  ', '-33.96875   '], time:0.277352ms
                         out_f16x4pack(t4x4offset): ['-12.8125    ', '3.22070312  ', '-33.96875   '], time:0.272465ms
                                 out_f16x4(t8x8sk): ['-12.8125    ', '3.22070312  ', '-33.96875   '], time:0.231457ms
                                out_f16x4(t8x8bcf): ['-12.8125    ', '3.22070312  ', '-33.96875   '], time:0.220227ms
                             out_f16x4pack(t8x8sk): ['-12.8125    ', '3.22070312  ', '-33.96875   '], time:0.216341ms
                                out_f16x4pack(bcf): ['-12.8125    ', '3.22070312  ', '-33.96875   '], time:0.203967ms
                         out_f16x4pack(bcf+offset): ['-12.8125    ', '3.22070312  ', '-33.96875   '], time:0.203228ms
                                out_f16x8pack(bcf): ['-12.8125    ', '3.22070312  ', '-33.96875   '], time:0.197363ms
                         out_f16x8pack(bcf+offset): ['-12.8125    ', '3.22070312  ', '-33.96875   '], time:0.196075ms
                           out_f16x8pack(bcf+dbuf): ['-12.8125    ', '3.22070312  ', '-33.96875   '], time:0.185442ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['-12.8125    ', '3.22070312  ', '-33.96875   '], time:0.172043ms
                    out_f16x8pack(k16+dbuf+offset): ['-12.8125    ', '3.22070312  ', '-33.96875   '], time:0.171638ms
                     out_f16x8pack(k16+dbuf+async): ['-12.8125    ', '3.22070312  ', '-33.96875   '], time:0.176167ms
                           out_f16x8pack(k32+dbuf): ['-12.8125    ', '3.22070312  ', '-33.96875   '], time:0.178003ms
                     out_f16x8pack(k32+dbuf+async): ['-12.8125    ', '3.22070312  ', '-33.96875   '], time:0.173903ms
                     out_f16x8pack(k32+dbuf+t16x8): ['-12.8125    ', '3.22070312  ', '-33.96875   '], time:0.186944ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['-12.8125    ', '3.22070312  ', '-33.96875   '], time:0.177002ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['-12.8671875 ', '3.18945312  ', '-34.03125   '], time:0.372362ms
                               out_f16wmma(mma4x2): ['-12.8671875 ', '3.18945312  ', '-34.03125   '], time:0.194645ms
                       out_f16wmma(mma4x2+warp2x4): ['-12.8671875 ', '3.18945312  ', '-34.03125   '], time:0.135493ms
                 out_f16wmma(mma4x2+warp2x4+async): ['-12.8671875 ', '3.18945312  ', '-34.03125   '], time:0.117469ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['-12.8671875 ', '3.18945312  ', '-34.03125   '], time:0.177312ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['-12.8671875 ', '3.18945312  ', '-34.03125   '], time:0.103879ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['-12.8671875 ', '3.18945312  ', '-34.03125   '], time:0.123668ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['-12.8671875 ', '3.18945312  ', '-34.03125   '], time:0.157428ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['-12.8671875 ', '3.18945312  ', '-34.03125   '], time:0.123143ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['-12.8671875 ', '3.18945312  ', '-34.03125   '], time:0.123167ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['-12.8671875 ', '3.18945312  ', '-34.03125   '], time:0.118661ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-12.8671875 ', '3.18945312  ', '-34.03125   '], time:0.107241ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['-12.8671875 ', '3.18945312  ', '-34.03125   '], time:0.115657ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['-12.8671875 ', '3.18945312  ', '-34.03125   '], time:0.121546ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['-12.8671875 ', '3.18945312  ', '-34.03125   '], time:0.123215ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['-12.8671875 ', '3.18945312  ', '-34.03125   '], time:0.099921ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['-12.8671875 ', '3.18945312  ', '-34.03125   '], time:0.102186ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['-12.8671875 ', '3.18945312  ', '-34.03125   '], time:0.100017ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['-12.8671875 ', '3.18945312  ', '-34.03125   '], time:0.097775ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['-12.8671875 ', '3.18945312  ', '-34.03125   '], time:0.097585ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['-12.8671875 ', '3.18945312  ', '-34.03125   '], time:0.098562ms
                                        out_f16_th: ['-12.8125    ', '3.234375    ', '-34.0       '], time:0.085568ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=2048, K=256
                                           out_f16: ['-9.671875   ', '11.71875    ', '4.17578125  '], time:1.206255ms
                                       out_f16(sk): ['-9.671875   ', '11.71875    ', '4.17578125  '], time:0.905132ms
                            out_f16x4pack(t4x4bcf): ['-9.671875   ', '11.71875    ', '4.17578125  '], time:0.132132ms
                         out_f16x4pack(t4x4offset): ['-9.671875   ', '11.71875    ', '4.17578125  '], time:0.129557ms
                                 out_f16x4(t8x8sk): ['-9.671875   ', '11.71875    ', '4.17578125  '], time:0.129128ms
                                out_f16x4(t8x8bcf): ['-9.671875   ', '11.71875    ', '4.17578125  '], time:0.109267ms
                             out_f16x4pack(t8x8sk): ['-9.671875   ', '11.71875    ', '4.17578125  '], time:0.110793ms
                                out_f16x4pack(bcf): ['-9.671875   ', '11.71875    ', '4.17578125  '], time:0.100613ms
                         out_f16x4pack(bcf+offset): ['-9.671875   ', '11.71875    ', '4.17578125  '], time:0.100613ms
                                out_f16x8pack(bcf): ['-9.671875   ', '11.71875    ', '4.17578125  '], time:0.097919ms
                         out_f16x8pack(bcf+offset): ['-9.671875   ', '11.71875    ', '4.17578125  '], time:0.097036ms
                           out_f16x8pack(bcf+dbuf): ['-9.671875   ', '11.71875    ', '4.17578125  '], time:0.094295ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['-9.671875   ', '11.71875    ', '4.17578125  '], time:0.089383ms
                    out_f16x8pack(k16+dbuf+offset): ['-9.671875   ', '11.71875    ', '4.17578125  '], time:0.089407ms
                     out_f16x8pack(k16+dbuf+async): ['-9.671875   ', '11.71875    ', '4.17578125  '], time:0.090981ms
                           out_f16x8pack(k32+dbuf): ['-9.671875   ', '11.71875    ', '4.17578125  '], time:0.090885ms
                     out_f16x8pack(k32+dbuf+async): ['-9.671875   ', '11.71875    ', '4.17578125  '], time:0.089526ms
                     out_f16x8pack(k32+dbuf+t16x8): ['-9.671875   ', '11.71875    ', '4.17578125  '], time:0.094724ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['-9.671875   ', '11.71875    ', '4.17578125  '], time:0.090718ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['-9.6640625  ', '11.703125   ', '4.15625     '], time:0.189257ms
                               out_f16wmma(mma4x2): ['-9.6640625  ', '11.703125   ', '4.15625     '], time:0.102663ms
                       out_f16wmma(mma4x2+warp2x4): ['-9.6640625  ', '11.703125   ', '4.15625     '], time:0.081253ms
                 out_f16wmma(mma4x2+warp2x4+async): ['-9.6640625  ', '11.703125   ', '4.15625     '], time:0.073361ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['-9.6640625  ', '11.703125   ', '4.15625     '], time:0.083494ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['-9.6640625  ', '11.703125   ', '4.15625     '], time:0.066638ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['-9.6640625  ', '11.703125   ', '4.15625     '], time:0.072551ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['-9.6640625  ', '11.703125   ', '4.15625     '], time:0.095105ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['-9.6640625  ', '11.703125   ', '4.15625     '], time:0.076222ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['-9.6640625  ', '11.703125   ', '4.15625     '], time:0.076652ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['-9.6640625  ', '11.703125   ', '4.15625     '], time:0.073981ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-9.6640625  ', '11.703125   ', '4.15625     '], time:0.067639ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['-9.6640625  ', '11.703125   ', '4.15625     '], time:0.071239ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['-9.6640625  ', '11.703125   ', '4.15625     '], time:0.073957ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['-9.6640625  ', '11.703125   ', '4.15625     '], time:0.076652ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['-9.6640625  ', '11.703125   ', '4.15625     '], time:0.064993ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['-9.6640625  ', '11.703125   ', '4.15625     '], time:0.066495ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['-9.6640625  ', '11.703125   ', '4.15625     '], time:0.064921ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['-9.6640625  ', '11.703125   ', '4.15625     '], time:0.065398ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['-9.6640625  ', '11.703125   ', '4.15625     '], time:0.064325ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['-9.6640625  ', '11.703125   ', '4.15625     '], time:0.063539ms
                                        out_f16_th: ['-9.6875     ', '11.7109375  ', '4.16015625  '], time:0.048184ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=2048, K=512
                                           out_f16: ['28.1875     ', '-5.23828125 ', '-1.81347656 '], time:2.385688ms
                                       out_f16(sk): ['28.1875     ', '-5.23828125 ', '-1.81347656 '], time:1.779747ms
                            out_f16x4pack(t4x4bcf): ['28.1875     ', '-5.23828125 ', '-1.81347656 '], time:0.260139ms
                         out_f16x4pack(t4x4offset): ['28.1875     ', '-5.23828125 ', '-1.81347656 '], time:0.255561ms
                                 out_f16x4(t8x8sk): ['28.1875     ', '-5.23828125 ', '-1.81347656 '], time:0.235486ms
                                out_f16x4(t8x8bcf): ['28.1875     ', '-5.23828125 ', '-1.81347656 '], time:0.212550ms
                             out_f16x4pack(t8x8sk): ['28.1875     ', '-5.23828125 ', '-1.81347656 '], time:0.210381ms
                                out_f16x4pack(bcf): ['28.1875     ', '-5.23828125 ', '-1.81347656 '], time:0.195575ms
                         out_f16x4pack(bcf+offset): ['28.1875     ', '-5.23828125 ', '-1.81347656 '], time:0.195074ms
                                out_f16x8pack(bcf): ['28.1875     ', '-5.23828125 ', '-1.81347656 '], time:0.189304ms
                         out_f16x8pack(bcf+offset): ['28.1875     ', '-5.23828125 ', '-1.81347656 '], time:0.187898ms
                           out_f16x8pack(bcf+dbuf): ['28.1875     ', '-5.23828125 ', '-1.81347656 '], time:0.182343ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['28.1875     ', '-5.23828125 ', '-1.81347656 '], time:0.171614ms
                    out_f16x8pack(k16+dbuf+offset): ['28.1875     ', '-5.23828125 ', '-1.81347656 '], time:0.171328ms
                     out_f16x8pack(k16+dbuf+async): ['28.1875     ', '-5.23828125 ', '-1.81347656 '], time:0.175500ms
                           out_f16x8pack(k32+dbuf): ['28.1875     ', '-5.23828125 ', '-1.81347656 '], time:0.175858ms
                     out_f16x8pack(k32+dbuf+async): ['28.1875     ', '-5.23828125 ', '-1.81347656 '], time:0.172853ms
                     out_f16x8pack(k32+dbuf+t16x8): ['28.1875     ', '-5.23828125 ', '-1.81347656 '], time:0.182557ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['28.1875     ', '-5.23828125 ', '-1.81347656 '], time:0.174785ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['28.15625    ', '-5.265625   ', '-1.80371094 '], time:0.371289ms
                               out_f16wmma(mma4x2): ['28.15625    ', '-5.265625   ', '-1.80371094 '], time:0.193858ms
                       out_f16wmma(mma4x2+warp2x4): ['28.15625    ', '-5.265625   ', '-1.80371094 '], time:0.137019ms
                 out_f16wmma(mma4x2+warp2x4+async): ['28.15625    ', '-5.265625   ', '-1.80371094 '], time:0.125027ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['28.15625    ', '-5.265625   ', '-1.80371094 '], time:0.145125ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['28.15625    ', '-5.265625   ', '-1.80371094 '], time:0.109458ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['28.15625    ', '-5.265625   ', '-1.80371094 '], time:0.124955ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['28.15625    ', '-5.265625   ', '-1.80371094 '], time:0.166774ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['28.15625    ', '-5.265625   ', '-1.80371094 '], time:0.129032ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['28.15625    ', '-5.265625   ', '-1.80371094 '], time:0.129199ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['28.15625    ', '-5.265625   ', '-1.80371094 '], time:0.124598ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['28.15625    ', '-5.265625   ', '-1.80371094 '], time:0.112963ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['28.15625    ', '-5.265625   ', '-1.80371094 '], time:0.121617ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['28.15625    ', '-5.265625   ', '-1.80371094 '], time:0.126600ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['28.15625    ', '-5.265625   ', '-1.80371094 '], time:0.129390ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['28.15625    ', '-5.265625   ', '-1.80371094 '], time:0.106955ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['28.15625    ', '-5.265625   ', '-1.80371094 '], time:0.110722ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['28.15625    ', '-5.265625   ', '-1.80371094 '], time:0.106573ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['28.15625    ', '-5.265625   ', '-1.80371094 '], time:0.106335ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['28.15625    ', '-5.265625   ', '-1.80371094 '], time:0.105715ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['28.15625    ', '-5.265625   ', '-1.80371094 '], time:0.104737ms
                                        out_f16_th: ['28.171875   ', '-5.2578125  ', '-1.79199219 '], time:0.087309ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=2048, K=1024
                                           out_f16: ['-14.78125   ', '-9.9609375  ', '-66.875     '], time:4.745364ms
                                       out_f16(sk): ['-14.78125   ', '-9.9609375  ', '-66.875     '], time:3.529716ms
                            out_f16x4pack(t4x4bcf): ['-14.78125   ', '-9.9609375  ', '-66.875     '], time:0.516915ms
                         out_f16x4pack(t4x4offset): ['-14.78125   ', '-9.9609375  ', '-66.875     '], time:0.507665ms
                                 out_f16x4(t8x8sk): ['-14.78125   ', '-9.9609375  ', '-66.875     '], time:0.446701ms
                                out_f16x4(t8x8bcf): ['-14.78125   ', '-9.9609375  ', '-66.875     '], time:0.418901ms
                             out_f16x4pack(t8x8sk): ['-14.78125   ', '-9.9609375  ', '-66.875     '], time:0.409174ms
                                out_f16x4pack(bcf): ['-14.78125   ', '-9.9609375  ', '-66.875     '], time:0.384450ms
                         out_f16x4pack(bcf+offset): ['-14.78125   ', '-9.9609375  ', '-66.875     '], time:0.383162ms
                                out_f16x8pack(bcf): ['-14.78125   ', '-9.9609375  ', '-66.875     '], time:0.371957ms
                         out_f16x8pack(bcf+offset): ['-14.78125   ', '-9.9609375  ', '-66.875     '], time:0.368762ms
                           out_f16x8pack(bcf+dbuf): ['-14.78125   ', '-9.9609375  ', '-66.875     '], time:0.358129ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['-14.78125   ', '-9.9609375  ', '-66.875     '], time:0.336647ms
                    out_f16x8pack(k16+dbuf+offset): ['-14.78125   ', '-9.9609375  ', '-66.875     '], time:0.336337ms
                     out_f16x8pack(k16+dbuf+async): ['-14.78125   ', '-9.9609375  ', '-66.875     '], time:0.344276ms
                           out_f16x8pack(k32+dbuf): ['-14.78125   ', '-9.9609375  ', '-66.875     '], time:0.345731ms
                     out_f16x8pack(k32+dbuf+async): ['-14.78125   ', '-9.9609375  ', '-66.875     '], time:0.339556ms
                     out_f16x8pack(k32+dbuf+t16x8): ['-14.78125   ', '-9.9609375  ', '-66.875     '], time:0.359702ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['-14.78125   ', '-9.9609375  ', '-66.875     '], time:0.342917ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['-14.78125   ', '-10.0625    ', '-67.0625    '], time:0.735736ms
                               out_f16wmma(mma4x2): ['-14.78125   ', '-10.0625    ', '-67.0625    '], time:0.376296ms
                       out_f16wmma(mma4x2+warp2x4): ['-14.78125   ', '-10.0625    ', '-67.0625    '], time:0.247860ms
                 out_f16wmma(mma4x2+warp2x4+async): ['-14.78125   ', '-10.0625    ', '-67.0625    '], time:0.228500ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['-14.78125   ', '-10.0625    ', '-67.0625    '], time:0.272346ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['-14.78125   ', '-10.0625    ', '-67.0625    '], time:0.196648ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['-14.78125   ', '-10.0625    ', '-67.0625    '], time:0.229120ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['-14.78125   ', '-10.0625    ', '-67.0625    '], time:0.310302ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['-14.78125   ', '-10.0625    ', '-67.0625    '], time:0.237226ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['-14.78125   ', '-10.0625    ', '-67.0625    '], time:0.237393ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['-14.78125   ', '-10.0625    ', '-67.0625    '], time:0.227165ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-14.78125   ', '-10.0625    ', '-67.0625    '], time:0.204492ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['-14.78125   ', '-10.0625    ', '-67.0625    '], time:0.222588ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['-14.78125   ', '-10.0625    ', '-67.0625    '], time:0.232100ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['-14.78125   ', '-10.0625    ', '-67.0625    '], time:0.237298ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['-14.78125   ', '-10.0625    ', '-67.0625    '], time:0.191355ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['-14.78125   ', '-10.0625    ', '-67.0625    '], time:0.199533ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['-14.78125   ', '-10.0625    ', '-67.0625    '], time:0.190210ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['-14.78125   ', '-10.0625    ', '-67.0625    '], time:0.188971ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['-14.78125   ', '-10.0625    ', '-67.0625    '], time:0.187469ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['-14.78125   ', '-10.0625    ', '-67.0625    '], time:0.187802ms
                                        out_f16_th: ['-14.8125    ', '-10.0546875 ', '-66.9375    '], time:0.169420ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=256
                                           out_f16: ['9.5859375   ', '-6.25       ', '-10.625     '], time:2.403069ms
                                       out_f16(sk): ['9.5859375   ', '-6.25       ', '-10.625     '], time:1.800013ms
                            out_f16x4pack(t4x4bcf): ['9.5859375   ', '-6.25       ', '-10.625     '], time:0.253177ms
                         out_f16x4pack(t4x4offset): ['9.5859375   ', '-6.25       ', '-10.625     '], time:0.248551ms
                                 out_f16x4(t8x8sk): ['9.5859375   ', '-6.25       ', '-10.625     '], time:0.231290ms
                                out_f16x4(t8x8bcf): ['9.5859375   ', '-6.25       ', '-10.625     '], time:0.209022ms
                             out_f16x4pack(t8x8sk): ['9.5859375   ', '-6.25       ', '-10.625     '], time:0.207210ms
                                out_f16x4pack(bcf): ['9.5859375   ', '-6.25       ', '-10.625     '], time:0.192142ms
                         out_f16x4pack(bcf+offset): ['9.5859375   ', '-6.25       ', '-10.625     '], time:0.191307ms
                                out_f16x8pack(bcf): ['9.5859375   ', '-6.25       ', '-10.625     '], time:0.185609ms
                         out_f16x8pack(bcf+offset): ['9.5859375   ', '-6.25       ', '-10.625     '], time:0.183630ms
                           out_f16x8pack(bcf+dbuf): ['9.5859375   ', '-6.25       ', '-10.625     '], time:0.181222ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['9.5859375   ', '-6.25       ', '-10.625     '], time:0.172138ms
                    out_f16x8pack(k16+dbuf+offset): ['9.5859375   ', '-6.25       ', '-10.625     '], time:0.172067ms
                     out_f16x8pack(k16+dbuf+async): ['9.5859375   ', '-6.25       ', '-10.625     '], time:0.175524ms
                           out_f16x8pack(k32+dbuf): ['9.5859375   ', '-6.25       ', '-10.625     '], time:0.175261ms
                     out_f16x8pack(k32+dbuf+async): ['9.5859375   ', '-6.25       ', '-10.625     '], time:0.172925ms
                     out_f16x8pack(k32+dbuf+t16x8): ['9.5859375   ', '-6.25       ', '-10.625     '], time:0.181389ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['9.5859375   ', '-6.25       ', '-10.625     '], time:0.174308ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['9.6015625   ', '-6.24609375 ', '-10.6328125 '], time:0.377917ms
                               out_f16wmma(mma4x2): ['9.6015625   ', '-6.24609375 ', '-10.6328125 '], time:0.199890ms
                       out_f16wmma(mma4x2+warp2x4): ['9.6015625   ', '-6.24609375 ', '-10.6328125 '], time:0.156021ms
                 out_f16wmma(mma4x2+warp2x4+async): ['9.6015625   ', '-6.24609375 ', '-10.6328125 '], time:0.134683ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['9.6015625   ', '-6.24609375 ', '-10.6328125 '], time:0.156760ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['9.6015625   ', '-6.24609375 ', '-10.6328125 '], time:0.118947ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['9.6015625   ', '-6.24609375 ', '-10.6328125 '], time:0.129938ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['9.6015625   ', '-6.24609375 ', '-10.6328125 '], time:0.180268ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['9.6015625   ', '-6.24609375 ', '-10.6328125 '], time:0.137448ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['9.6015625   ', '-6.24609375 ', '-10.6328125 '], time:0.137806ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['9.6015625   ', '-6.24609375 ', '-10.6328125 '], time:0.134492ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['9.6015625   ', '-6.24609375 ', '-10.6328125 '], time:0.121021ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['9.6015625   ', '-6.24609375 ', '-10.6328125 '], time:0.128889ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['9.6015625   ', '-6.24609375 ', '-10.6328125 '], time:0.134206ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['9.6015625   ', '-6.24609375 ', '-10.6328125 '], time:0.137186ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['9.6015625   ', '-6.24609375 ', '-10.6328125 '], time:0.116992ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['9.6015625   ', '-6.24609375 ', '-10.6328125 '], time:0.121951ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['9.6015625   ', '-6.24609375 ', '-10.6328125 '], time:0.117660ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['9.6015625   ', '-6.24609375 ', '-10.6328125 '], time:0.116134ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['9.6015625   ', '-6.24609375 ', '-10.6328125 '], time:0.115180ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['9.6015625   ', '-6.24609375 ', '-10.6328125 '], time:0.111556ms
                                        out_f16_th: ['9.6015625   ', '-6.23828125 ', '-10.640625  '], time:0.090384ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=512
                                           out_f16: ['6.671875    ', '10.109375   ', '-9.484375   '], time:4.755330ms
                                       out_f16(sk): ['6.671875    ', '10.109375   ', '-9.484375   '], time:3.544712ms
                            out_f16x4pack(t4x4bcf): ['6.671875    ', '10.109375   ', '-9.484375   '], time:0.501204ms
                         out_f16x4pack(t4x4offset): ['6.671875    ', '10.109375   ', '-9.484375   '], time:0.491786ms
                                 out_f16x4(t8x8sk): ['6.671875    ', '10.109375   ', '-9.484375   '], time:0.441647ms
                                out_f16x4(t8x8bcf): ['6.671875    ', '10.109375   ', '-9.484375   '], time:0.410080ms
                             out_f16x4pack(t8x8sk): ['6.671875    ', '10.109375   ', '-9.484375   '], time:0.401950ms
                                out_f16x4pack(bcf): ['6.671875    ', '10.109375   ', '-9.484375   '], time:0.376248ms
                         out_f16x4pack(bcf+offset): ['6.671875    ', '10.109375   ', '-9.484375   '], time:0.374746ms
                                out_f16x8pack(bcf): ['6.671875    ', '10.109375   ', '-9.484375   '], time:0.362182ms
                         out_f16x8pack(bcf+offset): ['6.671875    ', '10.109375   ', '-9.484375   '], time:0.359511ms
                           out_f16x8pack(bcf+dbuf): ['6.671875    ', '10.109375   ', '-9.484375   '], time:0.356007ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['6.671875    ', '10.109375   ', '-9.484375   '], time:0.337315ms
                    out_f16x8pack(k16+dbuf+offset): ['6.671875    ', '10.109375   ', '-9.484375   '], time:0.336909ms
                     out_f16x8pack(k16+dbuf+async): ['6.671875    ', '10.109375   ', '-9.484375   '], time:0.344563ms
                           out_f16x8pack(k32+dbuf): ['6.671875    ', '10.109375   ', '-9.484375   '], time:0.344062ms
                     out_f16x8pack(k32+dbuf+async): ['6.671875    ', '10.109375   ', '-9.484375   '], time:0.339699ms
                     out_f16x8pack(k32+dbuf+t16x8): ['6.671875    ', '10.109375   ', '-9.484375   '], time:0.356412ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['6.671875    ', '10.109375   ', '-9.484375   '], time:0.342011ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['6.69921875  ', '10.0078125  ', '-9.4375     '], time:0.743055ms
                               out_f16wmma(mma4x2): ['6.69921875  ', '10.0078125  ', '-9.4375     '], time:0.380993ms
                       out_f16wmma(mma4x2+warp2x4): ['6.69921875  ', '10.0078125  ', '-9.4375     '], time:0.267601ms
                 out_f16wmma(mma4x2+warp2x4+async): ['6.69921875  ', '10.0078125  ', '-9.4375     '], time:0.237703ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['6.69921875  ', '10.0078125  ', '-9.4375     '], time:0.280118ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['6.69921875  ', '10.0078125  ', '-9.4375     '], time:0.202632ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['6.69921875  ', '10.0078125  ', '-9.4375     '], time:0.232768ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['6.69921875  ', '10.0078125  ', '-9.4375     '], time:0.321174ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['6.69921875  ', '10.0078125  ', '-9.4375     '], time:0.243449ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['6.69921875  ', '10.0078125  ', '-9.4375     '], time:0.243592ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['6.69921875  ', '10.0078125  ', '-9.4375     '], time:0.236392ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['6.69921875  ', '10.0078125  ', '-9.4375     '], time:0.212860ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['6.69921875  ', '10.0078125  ', '-9.4375     '], time:0.228286ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['6.69921875  ', '10.0078125  ', '-9.4375     '], time:0.236917ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['6.69921875  ', '10.0078125  ', '-9.4375     '], time:0.243306ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['6.69921875  ', '10.0078125  ', '-9.4375     '], time:0.200629ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['6.69921875  ', '10.0078125  ', '-9.4375     '], time:0.210881ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['6.69921875  ', '10.0078125  ', '-9.4375     '], time:0.198913ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['6.69921875  ', '10.0078125  ', '-9.4375     '], time:0.198007ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['6.69921875  ', '10.0078125  ', '-9.4375     '], time:0.197840ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['6.69921875  ', '10.0078125  ', '-9.4375     '], time:0.190878ms
                                        out_f16_th: ['6.6640625   ', '10.0390625  ', '-9.4609375  '], time:0.169420ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=1024
                                           out_f16: ['3.0546875   ', '35.96875    ', '-7.6953125  '], time:9.460449ms
                                       out_f16(sk): ['3.0546875   ', '35.96875    ', '-7.6953125  '], time:7.026243ms
                            out_f16x4pack(t4x4bcf): ['3.0546875   ', '35.96875    ', '-7.6953125  '], time:0.996542ms
                         out_f16x4pack(t4x4offset): ['3.0546875   ', '35.96875    ', '-7.6953125  '], time:0.978065ms
                                 out_f16x4(t8x8sk): ['3.0546875   ', '35.96875    ', '-7.6953125  '], time:0.855637ms
                                out_f16x4(t8x8bcf): ['3.0546875   ', '35.96875    ', '-7.6953125  '], time:0.811505ms
                             out_f16x4pack(t8x8sk): ['3.0546875   ', '35.96875    ', '-7.6953125  '], time:0.788736ms
                                out_f16x4pack(bcf): ['3.0546875   ', '35.96875    ', '-7.6953125  '], time:0.743079ms
                         out_f16x4pack(bcf+offset): ['3.0546875   ', '35.96875    ', '-7.6953125  '], time:0.740409ms
                                out_f16x8pack(bcf): ['3.0546875   ', '35.96875    ', '-7.6953125  '], time:0.713587ms
                         out_f16x8pack(bcf+offset): ['3.0546875   ', '35.96875    ', '-7.6953125  '], time:0.707984ms
                           out_f16x8pack(bcf+dbuf): ['3.0546875   ', '35.96875    ', '-7.6953125  '], time:0.703931ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['3.0546875   ', '35.96875    ', '-7.6953125  '], time:0.666451ms
                    out_f16x8pack(k16+dbuf+offset): ['3.0546875   ', '35.96875    ', '-7.6953125  '], time:0.665879ms
                     out_f16x8pack(k16+dbuf+async): ['3.0546875   ', '35.96875    ', '-7.6953125  '], time:0.680852ms
                           out_f16x8pack(k32+dbuf): ['3.0546875   ', '35.96875    ', '-7.6953125  '], time:0.680232ms
                     out_f16x8pack(k32+dbuf+async): ['3.0546875   ', '35.96875    ', '-7.6953125  '], time:0.671506ms
                     out_f16x8pack(k32+dbuf+t16x8): ['3.0546875   ', '35.96875    ', '-7.6953125  '], time:0.705028ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['3.0546875   ', '35.96875    ', '-7.6953125  '], time:0.676084ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['3.11328125  ', '36.125      ', '-7.8046875  '], time:1.469350ms
                               out_f16wmma(mma4x2): ['3.11328125  ', '36.125      ', '-7.8046875  '], time:0.744700ms
                       out_f16wmma(mma4x2+warp2x4): ['3.11328125  ', '36.125      ', '-7.8046875  '], time:0.488305ms
                 out_f16wmma(mma4x2+warp2x4+async): ['3.11328125  ', '36.125      ', '-7.8046875  '], time:0.443363ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['3.11328125  ', '36.125      ', '-7.8046875  '], time:0.532341ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['3.11328125  ', '36.125      ', '-7.8046875  '], time:0.373864ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['3.11328125  ', '36.125      ', '-7.8046875  '], time:0.439048ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['3.11328125  ', '36.125      ', '-7.8046875  '], time:0.607681ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['3.11328125  ', '36.125      ', '-7.8046875  '], time:0.460005ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['3.11328125  ', '36.125      ', '-7.8046875  '], time:0.459814ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['3.11328125  ', '36.125      ', '-7.8046875  '], time:0.440431ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['3.11328125  ', '36.125      ', '-7.8046875  '], time:0.394607ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['3.11328125  ', '36.125      ', '-7.8046875  '], time:0.428915ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['3.11328125  ', '36.125      ', '-7.8046875  '], time:0.446892ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['3.11328125  ', '36.125      ', '-7.8046875  '], time:0.459623ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['3.11328125  ', '36.125      ', '-7.8046875  '], time:0.370765ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['3.11328125  ', '36.125      ', '-7.8046875  '], time:0.389671ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['3.11328125  ', '36.125      ', '-7.8046875  '], time:0.366282ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['3.11328125  ', '36.125      ', '-7.8046875  '], time:0.361443ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['3.11328125  ', '36.125      ', '-7.8046875  '], time:0.362182ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['3.11328125  ', '36.125      ', '-7.8046875  '], time:0.355911ms
                                        out_f16_th: ['3.07226562  ', '36.09375    ', '-7.77734375 '], time:0.334620ms
------------------------------------------------------------------------------------------------------------------------
```
