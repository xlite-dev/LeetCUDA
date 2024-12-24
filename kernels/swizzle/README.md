## Learn how to apply SMEM Swizzle for bank conflicts free

- build bin

```bash
make
```

- ncu profile

```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld ./hgemm_mma_swizzle.bin
ncu --metrics sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm ./hgemm_mma_swizzle.bin
```
