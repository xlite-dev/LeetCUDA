## Learn how to apply SMEM Swizzle for bank conflicts free

- build bin

```bash
make
```

- ncu profile

```bash
ncu --metrics l1tex__data_bank_reads ./mat_trans_swizzle.bin
ncu --metrics l1tex__data_bank_writes ./mat_trans_swizzle.bin
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld ./mat_trans_swizzle.bin
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st ./mat_trans_swizzle.bin

ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld ./hgemm_mma_swizzle.bin 1024 1024 1024 0 1
ncu --metrics sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm ./hgemm_mma_swizzle.bin 1024 1024 1024 0 1
```

log: 

```bash
// bank conflicts free via pad = 8, 拒绝幻想，相信profile
[1490314] hgemm_mma_swizzle.bin@127.0.0.1
  void hgemm_mma_m16n8k16_naive_kernel<16, 8, 16>(__half *, __half *, __half *, int, int, int) (128, 64, 1)x(32, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.avg                 22795.13
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.max                    24576
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.min                    19968
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                  2097152
    -------------------------------------------------------- ----------- ------------

  void hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel<16, 8, 16, 2, 4, 4, 4, 8, 8>(__half *, __half *, __half *, int, int, int) (8, 8, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.avg                        0
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.max                        0
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.min                        0
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    -------------------------------------------------------- ----------- ------------
[1490418] hgemm_mma_swizzle.bin@127.0.0.1
  void hgemm_mma_m16n8k16_naive_kernel<16, 8, 16>(__half *, __half *, __half *, int, int, int) (128, 64, 1)x(32, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    ------------------------------------------------------------------ ----------- ------------
    Metric Name                                                        Metric Unit Metric Value
    ------------------------------------------------------------------ ----------- ------------
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.avg                 22795.13
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.max                    24576
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.min                    18944
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.sum                  2097152
    ------------------------------------------------------------------ ----------- ------------

  void hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel<16, 8, 16, 2, 4, 4, 4, 8, 8>(__half *, __half *, __half *, int, int, int) (8, 8, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    ------------------------------------------------------------------ ----------- ------------
    Metric Name                                                        Metric Unit Metric Value
    ------------------------------------------------------------------ ----------- ------------
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.avg                        0
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.max                        0
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.min                        0
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.sum                        0
    ------------------------------------------------------------------ ----------- ------------
```
