# Mat Transpose

## 0x00 说明

包含以下内容：

- [X] mat_transpose_f32_col2row_kernel
- [X] mat_transpose_f32_row2col_kernel
- [X] mat_transpose_f32x4_col2row_kernel(float4向量化版本)
- [X] mat_transpose_f32x4_row2col_kernel(float4向量化版本)
- [X] mat_transpose_f32_diagnonal(对角轴应用于S=K)
- [X] mat_transpose_f32x4_shared_col2row_kernel(float4向量化版本，共享内存)
- [X] mat_transpose_f32x4_shared_row2col_kernel(float4向量化版本，共享内存)
- [X] mat_transpose_f32x4_shared_bcf_col2row_kernel(float4向量化版本，共享内存，去bank conflict)
- [X] mat_transpose_f32x4_shared_bcf_row2col_kernel(float4向量化版本，共享内存，去bank conflict)
- [X] mat_transpose_cute_row2col_reg
- [X] mat_transpose_cute_col2row_reg
- [X] mat_transpose_cute_col_smem
- [X] mat_transpose_cute_row_smem
- [X] mat_transpose_cute_col_smem_swizzled (bank conflict free)
- [X] mat_transpose_cute_row_smem_swizzled
- [X] mat_transpose_cute_row_cvectorized
- [X] mat_transpose_cute_row_cvectorized_swizzled
- [X] mat_transpose_cute_row_rvectorized
- [X] mat_transpose_cute_row_rvectorized_swizzled
- [X] PyTorch bindings

虽然是基础操作但是很适合练手，比矩阵乘法难度低一点但是可以其中可以用到的优化技巧都可以想办法用到这里来。

## 测试

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长: Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada
python3 mat_transpose.py
```

输出:

```bash
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=1024, N=1024
                       out_original: [0.0, 1.0, 1024.0], validate False, time:0.01322293ms
                    out_f32_col2row: [0.0, 1024.0, 1.0], validate True , time:0.03734207ms
                    out_f32_row2col: [0.0, 1024.0, 1.0], validate True , time:0.02030921ms
                out_f32_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.01762843ms
                out_f32_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.01770711ms
                  out_f32_diagnonal: [0.0, 1024.0, 1.0], validate True , time:0.01362014ms
                  out_f32x4_col2row: [0.0, 1024.0, 1.0], validate True , time:0.03673577ms
                  out_f32x4_row2col: [0.0, 1024.0, 1.0], validate True , time:0.01884627ms
              out_f32x4_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.01892900ms
              out_f32x4_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.01024675ms
       out_f32x4_shared_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.01027608ms
       out_f32x4_shared_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.01155329ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.01036501ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.01045060ms
 out_mat_transpose_cute_col2row_reg: [0.0, 1024.0, 1.0], validate True , time:0.01586843ms
 out_mat_transpose_cute_row2col_reg: [0.0, 1024.0, 1.0], validate True , time:0.01302576ms
    out_mat_transpose_cute_col_smem: [0.0, 1024.0, 1.0], validate True , time:0.01425385ms
    out_mat_transpose_cute_row_smem: [0.0, 1024.0, 1.0], validate True , time:0.01313591ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.01398778ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.01285744ms
out_mat_transpose_cute_row_cvectorized: [0.0, 1024.0, 1.0], validate True , time:0.01234102ms
out_mat_transpose_cute_row_rvectorized: [0.0, 1024.0, 1.0], validate True , time:0.01036048ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.01147723ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.01037669ms
out_mat_transpose_cute_row_rvectorized_swizzled_optimized: [0.0, 1024.0, 1.0], validate True , time:0.01087165ms
                         out_f32_th: [0.0, 1024.0, 1.0], validate True , time:0.03082085ms
                out_f32_th_compiled: [0.0, 1024.0, 1.0], validate True , time:0.07333684ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=1024, N=2048
                       out_original: [0.0, 1.0, 2048.0], validate False, time:0.02374721ms
                    out_f32_col2row: [0.0, 2048.0, 1.0], validate True , time:0.07149410ms
                    out_f32_row2col: [0.0, 2048.0, 1.0], validate True , time:0.03805780ms
                out_f32_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.03402877ms
                out_f32_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.03385830ms
                  out_f32x4_col2row: [0.0, 2048.0, 1.0], validate True , time:0.06924939ms
                  out_f32x4_row2col: [0.0, 2048.0, 1.0], validate True , time:0.03661728ms
              out_f32x4_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.03592300ms
              out_f32x4_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.02259111ms
       out_f32x4_shared_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.02224588ms
       out_f32x4_shared_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.02293062ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.02227783ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.02276349ms
 out_mat_transpose_cute_col2row_reg: [0.0, 2048.0, 1.0], validate True , time:0.03074288ms
 out_mat_transpose_cute_row2col_reg: [0.0, 2048.0, 1.0], validate True , time:0.02413964ms
    out_mat_transpose_cute_col_smem: [0.0, 2048.0, 1.0], validate True , time:0.02684236ms
    out_mat_transpose_cute_row_smem: [0.0, 2048.0, 1.0], validate True , time:0.02411056ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.02621555ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.02390814ms
out_mat_transpose_cute_row_cvectorized: [0.0, 2048.0, 1.0], validate True , time:0.02338839ms
out_mat_transpose_cute_row_rvectorized: [0.0, 2048.0, 1.0], validate True , time:0.02213287ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.02302265ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.02224278ms
out_mat_transpose_cute_row_rvectorized_swizzled_optimized: [0.0, 2048.0, 1.0], validate True , time:0.02258778ms
                         out_f32_th: [0.0, 2048.0, 1.0], validate True , time:0.06115103ms
                out_f32_th_compiled: [0.0, 2048.0, 1.0], validate True , time:0.05778503ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=1024, N=4096
                       out_original: [0.0, 1.0, 4096.0], validate False, time:0.04392624ms
                    out_f32_col2row: [0.0, 4096.0, 1.0], validate True , time:0.12588978ms
                    out_f32_row2col: [0.0, 4096.0, 1.0], validate True , time:0.07158637ms
                out_f32_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.06782699ms
                out_f32_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.06651688ms
                  out_f32x4_col2row: [0.0, 4096.0, 1.0], validate True , time:0.13267922ms
                  out_f32x4_row2col: [0.0, 4096.0, 1.0], validate True , time:0.07224798ms
              out_f32x4_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.06865716ms
              out_f32x4_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.04291391ms
       out_f32x4_shared_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.04286146ms
       out_f32x4_shared_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.04337263ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.04302001ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.04323077ms
 out_mat_transpose_cute_col2row_reg: [0.0, 4096.0, 1.0], validate True , time:0.06857824ms
 out_mat_transpose_cute_row2col_reg: [0.0, 4096.0, 1.0], validate True , time:0.04765439ms
    out_mat_transpose_cute_col_smem: [0.0, 4096.0, 1.0], validate True , time:0.05450273ms
    out_mat_transpose_cute_row_smem: [0.0, 4096.0, 1.0], validate True , time:0.04703522ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.05180621ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.04637504ms
out_mat_transpose_cute_row_cvectorized: [0.0, 4096.0, 1.0], validate True , time:0.04552484ms
out_mat_transpose_cute_row_rvectorized: [0.0, 4096.0, 1.0], validate True , time:0.04266071ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.04476309ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.04267573ms
out_mat_transpose_cute_row_rvectorized_swizzled_optimized: [0.0, 4096.0, 1.0], validate True , time:0.04308534ms
                         out_f32_th: [0.0, 4096.0, 1.0], validate True , time:0.11653090ms
                out_f32_th_compiled: [0.0, 4096.0, 1.0], validate True , time:0.11728549ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=1024, N=8192
                       out_original: [0.0, 1.0, 8192.0], validate False, time:0.08478189ms
                    out_f32_col2row: [0.0, 8192.0, 1.0], validate True , time:0.25058746ms
                    out_f32_row2col: [0.0, 8192.0, 1.0], validate True , time:0.14558482ms
                out_f32_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.13590932ms
                out_f32_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.13259697ms
                  out_f32x4_col2row: [0.0, 8192.0, 1.0], validate True , time:0.26455188ms
                  out_f32x4_row2col: [0.0, 8192.0, 1.0], validate True , time:0.16565704ms
              out_f32x4_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.13458133ms
              out_f32x4_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.08459783ms
       out_f32x4_shared_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.08723879ms
       out_f32x4_shared_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.08400655ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.08733034ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.08398318ms
 out_mat_transpose_cute_col2row_reg: [0.0, 8192.0, 1.0], validate True , time:0.13130641ms
 out_mat_transpose_cute_row2col_reg: [0.0, 8192.0, 1.0], validate True , time:0.09233975ms
    out_mat_transpose_cute_col_smem: [0.0, 8192.0, 1.0], validate True , time:0.10645866ms
    out_mat_transpose_cute_row_smem: [0.0, 8192.0, 1.0], validate True , time:0.09159970ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.10360169ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.09012818ms
out_mat_transpose_cute_row_cvectorized: [0.0, 8192.0, 1.0], validate True , time:0.09122086ms
out_mat_transpose_cute_row_rvectorized: [0.0, 8192.0, 1.0], validate True , time:0.08314061ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.08987474ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.08318615ms
out_mat_transpose_cute_row_rvectorized_swizzled_optimized: [0.0, 8192.0, 1.0], validate True , time:0.08382916ms
                         out_f32_th: [0.0, 8192.0, 1.0], validate True , time:0.23323345ms
                out_f32_th_compiled: [0.0, 8192.0, 1.0], validate True , time:0.23502731ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=1024
                       out_original: [0.0, 1.0, 1024.0], validate False, time:0.02391768ms
                    out_f32_col2row: [0.0, 1024.0, 1.0], validate True , time:0.06568789ms
                    out_f32_row2col: [0.0, 1024.0, 1.0], validate True , time:0.03711987ms
                out_f32_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.03264403ms
                out_f32_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.03245735ms
                  out_f32x4_col2row: [0.0, 1024.0, 1.0], validate True , time:0.06936216ms
                  out_f32x4_row2col: [0.0, 1024.0, 1.0], validate True , time:0.03759336ms
              out_f32x4_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.03599405ms
              out_f32x4_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.02248406ms
       out_f32x4_shared_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.02231026ms
       out_f32x4_shared_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.02310872ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.02236223ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.02289176ms
 out_mat_transpose_cute_col2row_reg: [0.0, 1024.0, 1.0], validate True , time:0.03009558ms
 out_mat_transpose_cute_row2col_reg: [0.0, 1024.0, 1.0], validate True , time:0.02510953ms
    out_mat_transpose_cute_col_smem: [0.0, 1024.0, 1.0], validate True , time:0.02802110ms
    out_mat_transpose_cute_row_smem: [0.0, 1024.0, 1.0], validate True , time:0.02494168ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.02729082ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.02456188ms
out_mat_transpose_cute_row_cvectorized: [0.0, 1024.0, 1.0], validate True , time:0.02376342ms
out_mat_transpose_cute_row_rvectorized: [0.0, 1024.0, 1.0], validate True , time:0.02220488ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.02344656ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.02236819ms
out_mat_transpose_cute_row_rvectorized_swizzled_optimized: [0.0, 1024.0, 1.0], validate True , time:0.02258658ms
                         out_f32_th: [0.0, 1024.0, 1.0], validate True , time:0.06107378ms
                out_f32_th_compiled: [0.0, 1024.0, 1.0], validate True , time:0.06182599ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=2048
                       out_original: [0.0, 1.0, 2048.0], validate False, time:0.04422760ms
                    out_f32_col2row: [0.0, 2048.0, 1.0], validate True , time:0.12561965ms
                    out_f32_row2col: [0.0, 2048.0, 1.0], validate True , time:0.07676911ms
                out_f32_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.06741023ms
                out_f32_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.06514311ms
                  out_f32_diagnonal: [0.0, 2048.0, 1.0], validate True , time:0.05187368ms
                  out_f32x4_col2row: [0.0, 2048.0, 1.0], validate True , time:0.13291311ms
                  out_f32x4_row2col: [0.0, 2048.0, 1.0], validate True , time:0.07298565ms
              out_f32x4_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.06908822ms
              out_f32x4_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.04279065ms
       out_f32x4_shared_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.04279661ms
       out_f32x4_shared_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.04364467ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.04293537ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.04339409ms
 out_mat_transpose_cute_col2row_reg: [0.0, 2048.0, 1.0], validate True , time:0.06557107ms
 out_mat_transpose_cute_row2col_reg: [0.0, 2048.0, 1.0], validate True , time:0.05029869ms
    out_mat_transpose_cute_col_smem: [0.0, 2048.0, 1.0], validate True , time:0.05644822ms
    out_mat_transpose_cute_row_smem: [0.0, 2048.0, 1.0], validate True , time:0.04912710ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.05456066ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.04846501ms
out_mat_transpose_cute_row_cvectorized: [0.0, 2048.0, 1.0], validate True , time:0.04663944ms
out_mat_transpose_cute_row_rvectorized: [0.0, 2048.0, 1.0], validate True , time:0.04294229ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.04629183ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.04302597ms
out_mat_transpose_cute_row_rvectorized_swizzled_optimized: [0.0, 2048.0, 1.0], validate True , time:0.04354882ms
                         out_f32_th: [0.0, 2048.0, 1.0], validate True , time:0.11834693ms
                out_f32_th_compiled: [0.0, 2048.0, 1.0], validate True , time:0.11902070ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=4096
                       out_original: [0.0, 1.0, 4096.0], validate False, time:0.08467579ms
                    out_f32_col2row: [0.0, 4096.0, 1.0], validate True , time:0.24970269ms
                    out_f32_row2col: [0.0, 4096.0, 1.0], validate True , time:0.15055704ms
                out_f32_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.13335991ms
                out_f32_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.13127899ms
                  out_f32x4_col2row: [0.0, 4096.0, 1.0], validate True , time:0.26483536ms
                  out_f32x4_row2col: [0.0, 4096.0, 1.0], validate True , time:0.15927362ms
              out_f32x4_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.13693118ms
              out_f32x4_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.08926201ms
       out_f32x4_shared_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.09092736ms
       out_f32x4_shared_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.08591580ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.09117699ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.08632779ms
 out_mat_transpose_cute_col2row_reg: [0.0, 4096.0, 1.0], validate True , time:0.12909770ms
 out_mat_transpose_cute_row2col_reg: [0.0, 4096.0, 1.0], validate True , time:0.09749031ms
    out_mat_transpose_cute_col_smem: [0.0, 4096.0, 1.0], validate True , time:0.11034966ms
    out_mat_transpose_cute_row_smem: [0.0, 4096.0, 1.0], validate True , time:0.09659719ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.10805702ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.09507823ms
out_mat_transpose_cute_row_cvectorized: [0.0, 4096.0, 1.0], validate True , time:0.09092307ms
out_mat_transpose_cute_row_rvectorized: [0.0, 4096.0, 1.0], validate True , time:0.08320594ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.09008026ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.08336163ms
out_mat_transpose_cute_row_rvectorized_swizzled_optimized: [0.0, 4096.0, 1.0], validate True , time:0.08401752ms
                         out_f32_th: [0.0, 4096.0, 1.0], validate True , time:0.23449445ms
                out_f32_th_compiled: [0.0, 4096.0, 1.0], validate True , time:0.23755503ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=8192
                       out_original: [0.0, 1.0, 8192.0], validate False, time:0.16631269ms
                    out_f32_col2row: [0.0, 8192.0, 1.0], validate True , time:0.50284362ms
                    out_f32_row2col: [0.0, 8192.0, 1.0], validate True , time:0.32336164ms
                out_f32_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.27494860ms
                out_f32_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.26218605ms
                  out_f32x4_col2row: [0.0, 8192.0, 1.0], validate True , time:0.51392102ms
                  out_f32x4_row2col: [0.0, 8192.0, 1.0], validate True , time:0.37878942ms
              out_f32x4_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.26178098ms
              out_f32x4_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.16887903ms
       out_f32x4_shared_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.17632246ms
       out_f32x4_shared_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.16928768ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.17651772ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.16609192ms
 out_mat_transpose_cute_col2row_reg: [0.0, 8192.0, 1.0], validate True , time:0.25982738ms
 out_mat_transpose_cute_row2col_reg: [0.0, 8192.0, 1.0], validate True , time:0.19544315ms
    out_mat_transpose_cute_col_smem: [0.0, 8192.0, 1.0], validate True , time:0.21964002ms
    out_mat_transpose_cute_row_smem: [0.0, 8192.0, 1.0], validate True , time:0.19438195ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.21328235ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.19120145ms
out_mat_transpose_cute_row_cvectorized: [0.0, 8192.0, 1.0], validate True , time:0.19285893ms
out_mat_transpose_cute_row_rvectorized: [0.0, 8192.0, 1.0], validate True , time:0.16419482ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.19116139ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.16445112ms
out_mat_transpose_cute_row_rvectorized_swizzled_optimized: [0.0, 8192.0, 1.0], validate True , time:0.16537309ms
                         out_f32_th: [0.0, 8192.0, 1.0], validate True , time:0.46839619ms
                out_f32_th_compiled: [0.0, 8192.0, 1.0], validate True , time:0.46951747ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=1024
                       out_original: [0.0, 1.0, 1024.0], validate False, time:0.04421234ms
                    out_f32_col2row: [0.0, 1024.0, 1.0], validate True , time:0.12696767ms
                    out_f32_row2col: [0.0, 1024.0, 1.0], validate True , time:0.07942963ms
                out_f32_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.06330681ms
                out_f32_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.06264949ms
                  out_f32x4_col2row: [0.0, 1024.0, 1.0], validate True , time:0.13269401ms
                  out_f32x4_row2col: [0.0, 1024.0, 1.0], validate True , time:0.07409143ms
              out_f32x4_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.07032919ms
              out_f32x4_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.04281425ms
       out_f32x4_shared_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.04285693ms
       out_f32x4_shared_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.04353714ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.04288602ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.04316950ms
 out_mat_transpose_cute_col2row_reg: [0.0, 1024.0, 1.0], validate True , time:0.06419706ms
 out_mat_transpose_cute_row2col_reg: [0.0, 1024.0, 1.0], validate True , time:0.05924821ms
    out_mat_transpose_cute_col_smem: [0.0, 1024.0, 1.0], validate True , time:0.06122255ms
    out_mat_transpose_cute_row_smem: [0.0, 1024.0, 1.0], validate True , time:0.05818987ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.06006002ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.05800104ms
out_mat_transpose_cute_row_cvectorized: [0.0, 1024.0, 1.0], validate True , time:0.04861760ms
out_mat_transpose_cute_row_rvectorized: [0.0, 1024.0, 1.0], validate True , time:0.04257965ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.04820013ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.04268837ms
out_mat_transpose_cute_row_rvectorized_swizzled_optimized: [0.0, 1024.0, 1.0], validate True , time:0.04290032ms
                         out_f32_th: [0.0, 1024.0, 1.0], validate True , time:0.12000775ms
                out_f32_th_compiled: [0.0, 1024.0, 1.0], validate True , time:0.12048841ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=2048
                       out_original: [0.0, 1.0, 2048.0], validate False, time:0.08468652ms
                    out_f32_col2row: [0.0, 2048.0, 1.0], validate True , time:0.25145221ms
                    out_f32_row2col: [0.0, 2048.0, 1.0], validate True , time:0.16740918ms
                out_f32_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.12791324ms
                out_f32_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.12523937ms
                  out_f32x4_col2row: [0.0, 2048.0, 1.0], validate True , time:0.26657438ms
                  out_f32x4_row2col: [0.0, 2048.0, 1.0], validate True , time:0.15698409ms
              out_f32x4_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.13683271ms
              out_f32x4_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.08711052ms
       out_f32x4_shared_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.08722329ms
       out_f32x4_shared_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.08619714ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.08731914ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.08600688ms
 out_mat_transpose_cute_col2row_reg: [0.0, 2048.0, 1.0], validate True , time:0.12273955ms
 out_mat_transpose_cute_row2col_reg: [0.0, 2048.0, 1.0], validate True , time:0.12345028ms
    out_mat_transpose_cute_col_smem: [0.0, 2048.0, 1.0], validate True , time:0.12603784ms
    out_mat_transpose_cute_row_smem: [0.0, 2048.0, 1.0], validate True , time:0.12179136ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.12539601ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.12133408ms
out_mat_transpose_cute_row_cvectorized: [0.0, 2048.0, 1.0], validate True , time:0.09775424ms
out_mat_transpose_cute_row_rvectorized: [0.0, 2048.0, 1.0], validate True , time:0.08309197ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.09721851ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.08333635ms
out_mat_transpose_cute_row_rvectorized_swizzled_optimized: [0.0, 2048.0, 1.0], validate True , time:0.08407569ms
                         out_f32_th: [0.0, 2048.0, 1.0], validate True , time:0.24632931ms
                out_f32_th_compiled: [0.0, 2048.0, 1.0], validate True , time:0.24452114ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096
                       out_original: [0.0, 1.0, 4096.0], validate False, time:0.16618252ms
                    out_f32_col2row: [0.0, 4096.0, 1.0], validate True , time:0.50097990ms
                    out_f32_row2col: [0.0, 4096.0, 1.0], validate True , time:0.35535216ms
                out_f32_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.27060771ms
                out_f32_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.25861597ms
                  out_f32_diagnonal: [0.0, 4096.0, 1.0], validate True , time:0.26617074ms
                  out_f32x4_col2row: [0.0, 4096.0, 1.0], validate True , time:0.51177859ms
                  out_f32x4_row2col: [0.0, 4096.0, 1.0], validate True , time:0.31568432ms
              out_f32x4_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.26652837ms
              out_f32x4_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.17010331ms
       out_f32x4_shared_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.17163038ms
       out_f32x4_shared_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.16799879ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.17183828ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.16800833ms
 out_mat_transpose_cute_col2row_reg: [0.0, 4096.0, 1.0], validate True , time:0.26187921ms
 out_mat_transpose_cute_row2col_reg: [0.0, 4096.0, 1.0], validate True , time:0.24576879ms
    out_mat_transpose_cute_col_smem: [0.0, 4096.0, 1.0], validate True , time:0.25347733ms
    out_mat_transpose_cute_row_smem: [0.0, 4096.0, 1.0], validate True , time:0.24340343ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.25243449ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.24239349ms
out_mat_transpose_cute_row_cvectorized: [0.0, 4096.0, 1.0], validate True , time:0.19701624ms
out_mat_transpose_cute_row_rvectorized: [0.0, 4096.0, 1.0], validate True , time:0.16537905ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.19600892ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.16557527ms
out_mat_transpose_cute_row_rvectorized_swizzled_optimized: [0.0, 4096.0, 1.0], validate True , time:0.16697907ms
                         out_f32_th: [0.0, 4096.0, 1.0], validate True , time:0.49465632ms
                out_f32_th_compiled: [0.0, 4096.0, 1.0], validate True , time:0.49463820ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192
                       out_original: [0.0, 1.0, 8192.0], validate False, time:0.32828760ms
                    out_f32_col2row: [0.0, 8192.0, 1.0], validate True , time:1.02160501ms
                    out_f32_row2col: [0.0, 8192.0, 1.0], validate True , time:0.72063184ms
                out_f32_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.53131676ms
                out_f32_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.52935743ms
                  out_f32x4_col2row: [0.0, 8192.0, 1.0], validate True , time:1.03164721ms
                  out_f32x4_row2col: [0.0, 8192.0, 1.0], validate True , time:0.80112934ms
              out_f32x4_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.53537488ms
              out_f32x4_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.34326696ms
       out_f32x4_shared_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.35909820ms
       out_f32x4_shared_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.33556294ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.35932803ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.33499861ms
 out_mat_transpose_cute_col2row_reg: [0.0, 8192.0, 1.0], validate True , time:0.54176092ms
 out_mat_transpose_cute_row2col_reg: [0.0, 8192.0, 1.0], validate True , time:0.48399639ms
    out_mat_transpose_cute_col_smem: [0.0, 8192.0, 1.0], validate True , time:0.49951982ms
    out_mat_transpose_cute_row_smem: [0.0, 8192.0, 1.0], validate True , time:0.48039985ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.49459934ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.47925377ms
out_mat_transpose_cute_row_cvectorized: [0.0, 8192.0, 1.0], validate True , time:0.42132831ms
out_mat_transpose_cute_row_rvectorized: [0.0, 8192.0, 1.0], validate True , time:0.32764554ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.41889501ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.32755017ms
out_mat_transpose_cute_row_rvectorized_swizzled_optimized: [0.0, 8192.0, 1.0], validate True , time:0.33102536ms
                         out_f32_th: [0.0, 8192.0, 1.0], validate True , time:0.99227643ms
                out_f32_th_compiled: [0.0, 8192.0, 1.0], validate True , time:0.99217844ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=1024
                       out_original: [0.0, 1.0, 1024.0], validate False, time:0.08466172ms
                    out_f32_col2row: [0.0, 1024.0, 1.0], validate True , time:0.25207758ms
                    out_f32_row2col: [0.0, 1024.0, 1.0], validate True , time:0.18354273ms
                out_f32_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.12594485ms
                out_f32_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.12563300ms
                  out_f32x4_col2row: [0.0, 1024.0, 1.0], validate True , time:0.27044630ms
                  out_f32x4_row2col: [0.0, 1024.0, 1.0], validate True , time:0.16568422ms
              out_f32x4_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.13587356ms
              out_f32x4_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.08572435ms
       out_f32x4_shared_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.08661747ms
       out_f32x4_shared_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.08503509ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.08664989ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.08469391ms
 out_mat_transpose_cute_col2row_reg: [0.0, 1024.0, 1.0], validate True , time:0.12477684ms
 out_mat_transpose_cute_row2col_reg: [0.0, 1024.0, 1.0], validate True , time:0.11948586ms
    out_mat_transpose_cute_col_smem: [0.0, 1024.0, 1.0], validate True , time:0.12046981ms
    out_mat_transpose_cute_row_smem: [0.0, 1024.0, 1.0], validate True , time:0.11931205ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.11980653ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.11914992ms
out_mat_transpose_cute_row_cvectorized: [0.0, 1024.0, 1.0], validate True , time:0.10827446ms
out_mat_transpose_cute_row_rvectorized: [0.0, 1024.0, 1.0], validate True , time:0.08326149ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.10820341ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.08342290ms
out_mat_transpose_cute_row_rvectorized_swizzled_optimized: [0.0, 1024.0, 1.0], validate True , time:0.08371043ms
                         out_f32_th: [0.0, 1024.0, 1.0], validate True , time:0.27149653ms
                out_f32_th_compiled: [0.0, 1024.0, 1.0], validate True , time:0.27161312ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=2048
                       out_original: [0.0, 1.0, 2048.0], validate False, time:0.16627026ms
                    out_f32_col2row: [0.0, 2048.0, 1.0], validate True , time:0.50467086ms
                    out_f32_row2col: [0.0, 2048.0, 1.0], validate True , time:0.39958572ms
                out_f32_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.26184535ms
                out_f32_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.25405502ms
                  out_f32x4_col2row: [0.0, 2048.0, 1.0], validate True , time:0.51482177ms
                  out_f32x4_row2col: [0.0, 2048.0, 1.0], validate True , time:0.35543489ms
              out_f32x4_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.26602769ms
              out_f32x4_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.16936231ms
       out_f32x4_shared_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.17157674ms
       out_f32x4_shared_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.16717434ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.17186403ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.16655135ms
 out_mat_transpose_cute_col2row_reg: [0.0, 2048.0, 1.0], validate True , time:0.24904299ms
 out_mat_transpose_cute_row2col_reg: [0.0, 2048.0, 1.0], validate True , time:0.24884462ms
    out_mat_transpose_cute_col_smem: [0.0, 2048.0, 1.0], validate True , time:0.25050330ms
    out_mat_transpose_cute_row_smem: [0.0, 2048.0, 1.0], validate True , time:0.24837542ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.25063992ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.24788022ms
out_mat_transpose_cute_row_cvectorized: [0.0, 2048.0, 1.0], validate True , time:0.22211933ms
out_mat_transpose_cute_row_rvectorized: [0.0, 2048.0, 1.0], validate True , time:0.16454816ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.22272992ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.16480446ms
out_mat_transpose_cute_row_rvectorized_swizzled_optimized: [0.0, 2048.0, 1.0], validate True , time:0.16556239ms
                         out_f32_th: [0.0, 2048.0, 1.0], validate True , time:0.56485367ms
                out_f32_th_compiled: [0.0, 2048.0, 1.0], validate True , time:0.56441379ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096
                       out_original: [0.0, 1.0, 4096.0], validate False, time:0.32855296ms
                    out_f32_col2row: [0.0, 4096.0, 1.0], validate True , time:1.02771664ms
                    out_f32_row2col: [0.0, 4096.0, 1.0], validate True , time:0.82746363ms
                out_f32_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.52390122ms
                out_f32_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.53260994ms
                  out_f32x4_col2row: [0.0, 4096.0, 1.0], validate True , time:1.02764344ms
                  out_f32x4_row2col: [0.0, 4096.0, 1.0], validate True , time:0.73395181ms
              out_f32x4_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.53176832ms
              out_f32x4_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.33959198ms
       out_f32x4_shared_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.35376525ms
       out_f32x4_shared_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.33685756ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.35392237ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.33208728ms
 out_mat_transpose_cute_col2row_reg: [0.0, 4096.0, 1.0], validate True , time:0.53561139ms
 out_mat_transpose_cute_row2col_reg: [0.0, 4096.0, 1.0], validate True , time:0.50565004ms
    out_mat_transpose_cute_col_smem: [0.0, 4096.0, 1.0], validate True , time:0.51027036ms
    out_mat_transpose_cute_row_smem: [0.0, 4096.0, 1.0], validate True , time:0.50410295ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.50999451ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.50347900ms
out_mat_transpose_cute_row_cvectorized: [0.0, 4096.0, 1.0], validate True , time:0.44723439ms
out_mat_transpose_cute_row_rvectorized: [0.0, 4096.0, 1.0], validate True , time:0.33080840ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.44853663ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.33074450ms
out_mat_transpose_cute_row_rvectorized_swizzled_optimized: [0.0, 4096.0, 1.0], validate True , time:0.33293080ms
                         out_f32_th: [0.0, 4096.0, 1.0], validate True , time:1.16223931ms
                out_f32_th_compiled: [0.0, 4096.0, 1.0], validate True , time:1.16229939ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192
                       out_original: [0.0, 1.0, 8192.0], validate False, time:0.65326476ms
                    out_f32_col2row: [0.0, 8192.0, 1.0], validate True , time:2.06266975ms
                    out_f32_row2col: [0.0, 8192.0, 1.0], validate True , time:1.63385463ms
                out_f32_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:1.06545210ms
                out_f32_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:1.06657267ms
                  out_f32_diagnonal: [0.0, 8192.0, 1.0], validate True , time:1.11862946ms
                  out_f32x4_col2row: [0.0, 8192.0, 1.0], validate True , time:2.06704783ms
                  out_f32x4_row2col: [0.0, 8192.0, 1.0], validate True , time:1.67259908ms
              out_f32x4_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:1.06197333ms
              out_f32x4_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.69753051ms
       out_f32x4_shared_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.75325727ms
       out_f32x4_shared_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.66890717ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.75367093ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.66564393ms
 out_mat_transpose_cute_col2row_reg: [0.0, 8192.0, 1.0], validate True , time:1.04156613ms
 out_mat_transpose_cute_row2col_reg: [0.0, 8192.0, 1.0], validate True , time:0.98622823ms
    out_mat_transpose_cute_col_smem: [0.0, 8192.0, 1.0], validate True , time:1.01331401ms
    out_mat_transpose_cute_row_smem: [0.0, 8192.0, 1.0], validate True , time:0.98386550ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 8192.0, 1.0], validate True , time:1.00588274ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.98301244ms
out_mat_transpose_cute_row_cvectorized: [0.0, 8192.0, 1.0], validate True , time:0.90383577ms
out_mat_transpose_cute_row_rvectorized: [0.0, 8192.0, 1.0], validate True , time:0.65374827ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.90814710ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.65394568ms
out_mat_transpose_cute_row_rvectorized_swizzled_optimized: [0.0, 8192.0, 1.0], validate True , time:0.65754938ms
                         out_f32_th: [0.0, 8192.0, 1.0], validate True , time:2.29676986ms
                out_f32_th_compiled: [0.0, 8192.0, 1.0], validate True , time:2.29685950ms
----------------------------------------------------------------------------------------------------------------------------------

```
