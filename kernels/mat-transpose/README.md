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
                       out_original: [0.0, 1.0, 1024.0], validate False, time:0.00667048ms
                    out_f32_col2row: [0.0, 1024.0, 1.0], validate True , time:0.01761174ms
                    out_f32_row2col: [0.0, 1024.0, 1.0], validate True , time:0.01008821ms
                out_f32_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.00852585ms
                out_f32_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.00851846ms
                  out_f32_diagnonal: [0.0, 1024.0, 1.0], validate True , time:0.00641012ms
                  out_f32x4_col2row: [0.0, 1024.0, 1.0], validate True , time:0.01742840ms
                  out_f32x4_row2col: [0.0, 1024.0, 1.0], validate True , time:0.00943899ms
              out_f32x4_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.00987363ms
              out_f32x4_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.00499630ms
       out_f32x4_shared_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.00524426ms
       out_f32x4_shared_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.00670385ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.00536251ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.00580740ms
 out_mat_transpose_cute_col2row_reg: [0.0, 1024.0, 1.0], validate True , time:0.00747943ms
 out_mat_transpose_cute_row2col_reg: [0.0, 1024.0, 1.0], validate True , time:0.00599384ms
    out_mat_transpose_cute_col_smem: [0.0, 1024.0, 1.0], validate True , time:0.00989509ms
    out_mat_transpose_cute_row_smem: [0.0, 1024.0, 1.0], validate True , time:0.00534320ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.00869393ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.00555682ms
out_mat_transpose_cute_row_cvectorized: [0.0, 1024.0, 1.0], validate True , time:0.00727987ms
out_mat_transpose_cute_row_rvectorized: [0.0, 1024.0, 1.0], validate True , time:0.00493646ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.00592184ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.00563526ms
                         out_f32_th: [0.0, 1024.0, 1.0], validate True , time:0.01460934ms
                out_f32_th_compiled: [0.0, 1024.0, 1.0], validate True , time:0.03173113ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=1024, N=2048
                       out_original: [0.0, 1.0, 2048.0], validate False, time:0.00685191ms
                    out_f32_col2row: [0.0, 2048.0, 1.0], validate True , time:0.03323030ms
                    out_f32_row2col: [0.0, 2048.0, 1.0], validate True , time:0.01826382ms
                out_f32_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.01468587ms
                out_f32_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.01467657ms
                  out_f32x4_col2row: [0.0, 2048.0, 1.0], validate True , time:0.03329539ms
                  out_f32x4_row2col: [0.0, 2048.0, 1.0], validate True , time:0.01831269ms
              out_f32x4_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.01730418ms
              out_f32x4_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.00699377ms
       out_f32x4_shared_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.00813270ms
       out_f32x4_shared_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.01007080ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.00836945ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.00849128ms
 out_mat_transpose_cute_col2row_reg: [0.0, 2048.0, 1.0], validate True , time:0.01324391ms
 out_mat_transpose_cute_row2col_reg: [0.0, 2048.0, 1.0], validate True , time:0.00926590ms
    out_mat_transpose_cute_col_smem: [0.0, 2048.0, 1.0], validate True , time:0.01706100ms
    out_mat_transpose_cute_row_smem: [0.0, 2048.0, 1.0], validate True , time:0.00816727ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.01489687ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.00876975ms
out_mat_transpose_cute_row_cvectorized: [0.0, 2048.0, 1.0], validate True , time:0.01191163ms
out_mat_transpose_cute_row_rvectorized: [0.0, 2048.0, 1.0], validate True , time:0.00748754ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.00862432ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.00858784ms
                         out_f32_th: [0.0, 2048.0, 1.0], validate True , time:0.02438354ms
                out_f32_th_compiled: [0.0, 2048.0, 1.0], validate True , time:0.04902887ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=1024, N=4096
                       out_original: [0.0, 1.0, 4096.0], validate False, time:0.01132798ms
                    out_f32_col2row: [0.0, 4096.0, 1.0], validate True , time:0.06182218ms
                    out_f32_row2col: [0.0, 4096.0, 1.0], validate True , time:0.03392482ms
                out_f32_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.02772188ms
                out_f32_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.02774906ms
                  out_f32x4_col2row: [0.0, 4096.0, 1.0], validate True , time:0.06245613ms
                  out_f32x4_row2col: [0.0, 4096.0, 1.0], validate True , time:0.03352976ms
              out_f32x4_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.03137302ms
              out_f32x4_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.01143122ms
       out_f32x4_shared_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.01373696ms
       out_f32x4_shared_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.01705766ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.01421714ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.01388431ms
 out_mat_transpose_cute_col2row_reg: [0.0, 4096.0, 1.0], validate True , time:0.02740574ms
 out_mat_transpose_cute_row2col_reg: [0.0, 4096.0, 1.0], validate True , time:0.01666737ms
    out_mat_transpose_cute_col_smem: [0.0, 4096.0, 1.0], validate True , time:0.03182530ms
    out_mat_transpose_cute_row_smem: [0.0, 4096.0, 1.0], validate True , time:0.01621008ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.02744722ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.01582789ms
out_mat_transpose_cute_row_cvectorized: [0.0, 4096.0, 1.0], validate True , time:0.02123284ms
out_mat_transpose_cute_row_rvectorized: [0.0, 4096.0, 1.0], validate True , time:0.01262116ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.01439214ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.01478362ms
                         out_f32_th: [0.0, 4096.0, 1.0], validate True , time:0.04604459ms
                out_f32_th_compiled: [0.0, 4096.0, 1.0], validate True , time:0.04802442ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=1024, N=8192
                       out_original: [0.0, 1.0, 8192.0], validate False, time:0.02622271ms
                    out_f32_col2row: [0.0, 8192.0, 1.0], validate True , time:0.13227582ms
                    out_f32_row2col: [0.0, 8192.0, 1.0], validate True , time:0.06796670ms
                out_f32_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.06429243ms
                out_f32_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.06427002ms
                  out_f32x4_col2row: [0.0, 8192.0, 1.0], validate True , time:0.13971567ms
                  out_f32x4_row2col: [0.0, 8192.0, 1.0], validate True , time:0.06687403ms
              out_f32x4_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.08264017ms
              out_f32x4_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.02314472ms
       out_f32x4_shared_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.02923679ms
       out_f32x4_shared_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.03703094ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.02966428ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.03264546ms
 out_mat_transpose_cute_col2row_reg: [0.0, 8192.0, 1.0], validate True , time:0.05948019ms
 out_mat_transpose_cute_row2col_reg: [0.0, 8192.0, 1.0], validate True , time:0.04069352ms
    out_mat_transpose_cute_col_smem: [0.0, 8192.0, 1.0], validate True , time:0.06174302ms
    out_mat_transpose_cute_row_smem: [0.0, 8192.0, 1.0], validate True , time:0.04106617ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.05320787ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.03881240ms
out_mat_transpose_cute_row_cvectorized: [0.0, 8192.0, 1.0], validate True , time:0.04451799ms
out_mat_transpose_cute_row_rvectorized: [0.0, 8192.0, 1.0], validate True , time:0.02818966ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.03275704ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.03005648ms
                         out_f32_th: [0.0, 8192.0, 1.0], validate True , time:0.09130526ms
                out_f32_th_compiled: [0.0, 8192.0, 1.0], validate True , time:0.04808426ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=1024
                       out_original: [0.0, 1.0, 1024.0], validate False, time:0.00638318ms
                    out_f32_col2row: [0.0, 1024.0, 1.0], validate True , time:0.03563547ms
                    out_f32_row2col: [0.0, 1024.0, 1.0], validate True , time:0.01650119ms
                out_f32_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.01620793ms
                out_f32_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.01620555ms
                  out_f32x4_col2row: [0.0, 1024.0, 1.0], validate True , time:0.03585052ms
                  out_f32x4_row2col: [0.0, 1024.0, 1.0], validate True , time:0.01708913ms
              out_f32x4_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.01859522ms
              out_f32x4_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.00708890ms
       out_f32x4_shared_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.00807047ms
       out_f32x4_shared_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.01017618ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.00838590ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.00867391ms
 out_mat_transpose_cute_col2row_reg: [0.0, 1024.0, 1.0], validate True , time:0.01354027ms
 out_mat_transpose_cute_row2col_reg: [0.0, 1024.0, 1.0], validate True , time:0.00928712ms
    out_mat_transpose_cute_col_smem: [0.0, 1024.0, 1.0], validate True , time:0.01707125ms
    out_mat_transpose_cute_row_smem: [0.0, 1024.0, 1.0], validate True , time:0.00816989ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.01489687ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.00876665ms
out_mat_transpose_cute_row_cvectorized: [0.0, 1024.0, 1.0], validate True , time:0.01165247ms
out_mat_transpose_cute_row_rvectorized: [0.0, 1024.0, 1.0], validate True , time:0.00741124ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.00810361ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.00862789ms
                         out_f32_th: [0.0, 1024.0, 1.0], validate True , time:0.02319264ms
                out_f32_th_compiled: [0.0, 1024.0, 1.0], validate True , time:0.05861235ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=2048
                       out_original: [0.0, 1.0, 2048.0], validate False, time:0.01117110ms
                    out_f32_col2row: [0.0, 2048.0, 1.0], validate True , time:0.06950235ms
                    out_f32_row2col: [0.0, 2048.0, 1.0], validate True , time:0.03485703ms
                out_f32_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.03005457ms
                out_f32_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.03008485ms
                  out_f32_diagnonal: [0.0, 2048.0, 1.0], validate True , time:0.01830244ms
                  out_f32x4_col2row: [0.0, 2048.0, 1.0], validate True , time:0.07024670ms
                  out_f32x4_row2col: [0.0, 2048.0, 1.0], validate True , time:0.03537869ms
              out_f32x4_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.03516817ms
              out_f32x4_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.01145554ms
       out_f32x4_shared_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.01382899ms
       out_f32x4_shared_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.01729012ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.01426935ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.01422572ms
 out_mat_transpose_cute_col2row_reg: [0.0, 2048.0, 1.0], validate True , time:0.02737832ms
 out_mat_transpose_cute_row2col_reg: [0.0, 2048.0, 1.0], validate True , time:0.01676321ms
    out_mat_transpose_cute_col_smem: [0.0, 2048.0, 1.0], validate True , time:0.03183579ms
    out_mat_transpose_cute_row_smem: [0.0, 2048.0, 1.0], validate True , time:0.01636028ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.02742934ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.01592755ms
out_mat_transpose_cute_row_cvectorized: [0.0, 2048.0, 1.0], validate True , time:0.02093101ms
out_mat_transpose_cute_row_rvectorized: [0.0, 2048.0, 1.0], validate True , time:0.01259851ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.01370025ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.01482153ms
                         out_f32_th: [0.0, 2048.0, 1.0], validate True , time:0.04773760ms
                out_f32_th_compiled: [0.0, 2048.0, 1.0], validate True , time:0.05919909ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=4096
                       out_original: [0.0, 1.0, 4096.0], validate False, time:0.02612376ms
                    out_f32_col2row: [0.0, 4096.0, 1.0], validate True , time:0.13792729ms
                    out_f32_row2col: [0.0, 4096.0, 1.0], validate True , time:0.06782484ms
                out_f32_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.06800270ms
                out_f32_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.06801772ms
                  out_f32x4_col2row: [0.0, 4096.0, 1.0], validate True , time:0.14473867ms
                  out_f32x4_row2col: [0.0, 4096.0, 1.0], validate True , time:0.06878757ms
              out_f32x4_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.08687305ms
              out_f32x4_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.02311611ms
       out_f32x4_shared_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.02917051ms
       out_f32x4_shared_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.03698635ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.02948761ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.03269577ms
 out_mat_transpose_cute_col2row_reg: [0.0, 4096.0, 1.0], validate True , time:0.05988336ms
 out_mat_transpose_cute_row2col_reg: [0.0, 4096.0, 1.0], validate True , time:0.04102373ms
    out_mat_transpose_cute_col_smem: [0.0, 4096.0, 1.0], validate True , time:0.06177592ms
    out_mat_transpose_cute_row_smem: [0.0, 4096.0, 1.0], validate True , time:0.04124475ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.05327010ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.03905988ms
out_mat_transpose_cute_row_cvectorized: [0.0, 4096.0, 1.0], validate True , time:0.04370165ms
out_mat_transpose_cute_row_rvectorized: [0.0, 4096.0, 1.0], validate True , time:0.02813411ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.03098154ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.03004074ms
                         out_f32_th: [0.0, 4096.0, 1.0], validate True , time:0.09264803ms
                out_f32_th_compiled: [0.0, 4096.0, 1.0], validate True , time:0.05882740ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=8192
                       out_original: [0.0, 1.0, 8192.0], validate False, time:0.04788351ms
                    out_f32_col2row: [0.0, 8192.0, 1.0], validate True , time:0.26702380ms
                    out_f32_row2col: [0.0, 8192.0, 1.0], validate True , time:0.12641311ms
                out_f32_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.12766623ms
                out_f32_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.12762928ms
                  out_f32x4_col2row: [0.0, 8192.0, 1.0], validate True , time:0.27893376ms
                  out_f32x4_row2col: [0.0, 8192.0, 1.0], validate True , time:0.12813592ms
              out_f32x4_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.16845822ms
              out_f32x4_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.04240894ms
       out_f32x4_shared_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.05404329ms
       out_f32x4_shared_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.06824255ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.05481243ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.06005955ms
 out_mat_transpose_cute_col2row_reg: [0.0, 8192.0, 1.0], validate True , time:0.11488175ms
 out_mat_transpose_cute_row2col_reg: [0.0, 8192.0, 1.0], validate True , time:0.07866359ms
    out_mat_transpose_cute_col_smem: [0.0, 8192.0, 1.0], validate True , time:0.11966443ms
    out_mat_transpose_cute_row_smem: [0.0, 8192.0, 1.0], validate True , time:0.07921481ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.10269117ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.07479906ms
out_mat_transpose_cute_row_cvectorized: [0.0, 8192.0, 1.0], validate True , time:0.08218861ms
out_mat_transpose_cute_row_rvectorized: [0.0, 8192.0, 1.0], validate True , time:0.05100894ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.05682635ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.05597353ms
                         out_f32_th: [0.0, 8192.0, 1.0], validate True , time:0.17199111ms
                out_f32_th_compiled: [0.0, 8192.0, 1.0], validate True , time:0.05846906ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=1024
                       out_original: [0.0, 1.0, 1024.0], validate False, time:0.01135612ms
                    out_f32_col2row: [0.0, 1024.0, 1.0], validate True , time:0.06473422ms
                    out_f32_row2col: [0.0, 1024.0, 1.0], validate True , time:0.03022242ms
                out_f32_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.02885413ms
                out_f32_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.02884769ms
                  out_f32x4_col2row: [0.0, 1024.0, 1.0], validate True , time:0.06478477ms
                  out_f32x4_row2col: [0.0, 1024.0, 1.0], validate True , time:0.03187799ms
              out_f32x4_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.03276491ms
              out_f32x4_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.01148319ms
       out_f32x4_shared_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.01368475ms
       out_f32x4_shared_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.01800251ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.01427579ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.01484346ms
 out_mat_transpose_cute_col2row_reg: [0.0, 1024.0, 1.0], validate True , time:0.02777314ms
 out_mat_transpose_cute_row2col_reg: [0.0, 1024.0, 1.0], validate True , time:0.01661682ms
    out_mat_transpose_cute_col_smem: [0.0, 1024.0, 1.0], validate True , time:0.03186488ms
    out_mat_transpose_cute_row_smem: [0.0, 1024.0, 1.0], validate True , time:0.01607609ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.02742004ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.01581359ms
out_mat_transpose_cute_row_cvectorized: [0.0, 1024.0, 1.0], validate True , time:0.02088809ms
out_mat_transpose_cute_row_rvectorized: [0.0, 1024.0, 1.0], validate True , time:0.01251507ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.01367640ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.01482892ms
                         out_f32_th: [0.0, 1024.0, 1.0], validate True , time:0.04351592ms
                out_f32_th_compiled: [0.0, 1024.0, 1.0], validate True , time:0.06099153ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=2048
                       out_original: [0.0, 1.0, 2048.0], validate False, time:0.02611160ms
                    out_f32_col2row: [0.0, 2048.0, 1.0], validate True , time:0.14095616ms
                    out_f32_row2col: [0.0, 2048.0, 1.0], validate True , time:0.06800532ms
                out_f32_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.06891775ms
                out_f32_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.06890607ms
                  out_f32x4_col2row: [0.0, 2048.0, 1.0], validate True , time:0.14629245ms
                  out_f32x4_row2col: [0.0, 2048.0, 1.0], validate True , time:0.07073379ms
              out_f32x4_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.08693480ms
              out_f32x4_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.02308369ms
       out_f32x4_shared_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.02914500ms
       out_f32x4_shared_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.03913331ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.02951288ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.03415990ms
 out_mat_transpose_cute_col2row_reg: [0.0, 2048.0, 1.0], validate True , time:0.05992270ms
 out_mat_transpose_cute_row2col_reg: [0.0, 2048.0, 1.0], validate True , time:0.04137969ms
    out_mat_transpose_cute_col_smem: [0.0, 2048.0, 1.0], validate True , time:0.06176162ms
    out_mat_transpose_cute_row_smem: [0.0, 2048.0, 1.0], validate True , time:0.04155755ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.05330944ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.03944159ms
out_mat_transpose_cute_row_cvectorized: [0.0, 2048.0, 1.0], validate True , time:0.04380369ms
out_mat_transpose_cute_row_rvectorized: [0.0, 2048.0, 1.0], validate True , time:0.02814078ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.03094649ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.03010845ms
                         out_f32_th: [0.0, 2048.0, 1.0], validate True , time:0.09364438ms
                out_f32_th_compiled: [0.0, 2048.0, 1.0], validate True , time:0.05807614ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096
                       out_original: [0.0, 1.0, 4096.0], validate False, time:0.04761362ms
                    out_f32_col2row: [0.0, 4096.0, 1.0], validate True , time:0.26551914ms
                    out_f32_row2col: [0.0, 4096.0, 1.0], validate True , time:0.12826180ms
                out_f32_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.12965775ms
                out_f32_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.12970257ms
                  out_f32_diagnonal: [0.0, 4096.0, 1.0], validate True , time:0.08692741ms
                  out_f32x4_col2row: [0.0, 4096.0, 1.0], validate True , time:0.27640176ms
                  out_f32x4_row2col: [0.0, 4096.0, 1.0], validate True , time:0.13386226ms
              out_f32x4_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.16657877ms
              out_f32x4_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.04246473ms
       out_f32x4_shared_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.05401349ms
       out_f32x4_shared_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.07208681ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.05456352ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.06306529ms
 out_mat_transpose_cute_col2row_reg: [0.0, 4096.0, 1.0], validate True , time:0.11687827ms
 out_mat_transpose_cute_row2col_reg: [0.0, 4096.0, 1.0], validate True , time:0.07934999ms
    out_mat_transpose_cute_col_smem: [0.0, 4096.0, 1.0], validate True , time:0.11962628ms
    out_mat_transpose_cute_row_smem: [0.0, 4096.0, 1.0], validate True , time:0.07979536ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.10279226ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.07552862ms
out_mat_transpose_cute_row_cvectorized: [0.0, 4096.0, 1.0], validate True , time:0.08200026ms
out_mat_transpose_cute_row_rvectorized: [0.0, 4096.0, 1.0], validate True , time:0.05083799ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.05675173ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.05572939ms
                         out_f32_th: [0.0, 4096.0, 1.0], validate True , time:0.17590189ms
                out_f32_th_compiled: [0.0, 4096.0, 1.0], validate True , time:0.05817580ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192
                       out_original: [0.0, 1.0, 8192.0], validate False, time:0.09065676ms
                    out_f32_col2row: [0.0, 8192.0, 1.0], validate True , time:0.52875924ms
                    out_f32_row2col: [0.0, 8192.0, 1.0], validate True , time:0.24883223ms
                out_f32_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.25138545ms
                out_f32_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.25151587ms
                  out_f32x4_col2row: [0.0, 8192.0, 1.0], validate True , time:0.55523229ms
                  out_f32x4_row2col: [0.0, 8192.0, 1.0], validate True , time:0.25797868ms
              out_f32x4_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.33464503ms
              out_f32x4_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.08137870ms
       out_f32x4_shared_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.10357594ms
       out_f32x4_shared_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.13923311ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.10485888ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.12177515ms
 out_mat_transpose_cute_col2row_reg: [0.0, 8192.0, 1.0], validate True , time:0.22806001ms
 out_mat_transpose_cute_row2col_reg: [0.0, 8192.0, 1.0], validate True , time:0.15632129ms
    out_mat_transpose_cute_col_smem: [0.0, 8192.0, 1.0], validate True , time:0.23541665ms
    out_mat_transpose_cute_row_smem: [0.0, 8192.0, 1.0], validate True , time:0.15710807ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.20194674ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.14867640ms
out_mat_transpose_cute_row_cvectorized: [0.0, 8192.0, 1.0], validate True , time:0.15857029ms
out_mat_transpose_cute_row_rvectorized: [0.0, 8192.0, 1.0], validate True , time:0.09695339ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.10866976ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.10696220ms
                         out_f32_th: [0.0, 8192.0, 1.0], validate True , time:0.33883190ms
                out_f32_th_compiled: [0.0, 8192.0, 1.0], validate True , time:0.07038069ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=1024
                       out_original: [0.0, 1.0, 1024.0], validate False, time:0.02613044ms
                    out_f32_col2row: [0.0, 1024.0, 1.0], validate True , time:0.14240122ms
                    out_f32_row2col: [0.0, 1024.0, 1.0], validate True , time:0.06301308ms
                out_f32_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.06837869ms
                out_f32_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.06840038ms
                  out_f32x4_col2row: [0.0, 1024.0, 1.0], validate True , time:0.14549279ms
                  out_f32x4_row2col: [0.0, 1024.0, 1.0], validate True , time:0.06775117ms
              out_f32x4_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.08498430ms
              out_f32x4_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.02311444ms
       out_f32x4_shared_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.02915359ms
       out_f32x4_shared_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.03713298ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.02951527ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.03300023ms
 out_mat_transpose_cute_col2row_reg: [0.0, 1024.0, 1.0], validate True , time:0.05988598ms
 out_mat_transpose_cute_row2col_reg: [0.0, 1024.0, 1.0], validate True , time:0.04108763ms
    out_mat_transpose_cute_col_smem: [0.0, 1024.0, 1.0], validate True , time:0.06168962ms
    out_mat_transpose_cute_row_smem: [0.0, 1024.0, 1.0], validate True , time:0.04137659ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.05313158ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.03937101ms
out_mat_transpose_cute_row_cvectorized: [0.0, 1024.0, 1.0], validate True , time:0.04361272ms
out_mat_transpose_cute_row_rvectorized: [0.0, 1024.0, 1.0], validate True , time:0.02782607ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.03081965ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 1024.0, 1.0], validate True , time:0.02822685ms
                         out_f32_th: [0.0, 1024.0, 1.0], validate True , time:0.08901119ms
                out_f32_th_compiled: [0.0, 1024.0, 1.0], validate True , time:0.05782843ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=2048
                       out_original: [0.0, 1.0, 2048.0], validate False, time:0.04759407ms
                    out_f32_col2row: [0.0, 2048.0, 1.0], validate True , time:0.27500105ms
                    out_f32_row2col: [0.0, 2048.0, 1.0], validate True , time:0.12144923ms
                out_f32_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.13192320ms
                out_f32_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.13188601ms
                  out_f32x4_col2row: [0.0, 2048.0, 1.0], validate True , time:0.29046535ms
                  out_f32x4_row2col: [0.0, 2048.0, 1.0], validate True , time:0.13048077ms
              out_f32x4_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.17475128ms
              out_f32x4_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.04228449ms
       out_f32x4_shared_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.05392790ms
       out_f32x4_shared_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.06848621ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.05463552ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.06046867ms
 out_mat_transpose_cute_col2row_reg: [0.0, 2048.0, 1.0], validate True , time:0.11590195ms
 out_mat_transpose_cute_row2col_reg: [0.0, 2048.0, 1.0], validate True , time:0.07939339ms
    out_mat_transpose_cute_col_smem: [0.0, 2048.0, 1.0], validate True , time:0.11953950ms
    out_mat_transpose_cute_row_smem: [0.0, 2048.0, 1.0], validate True , time:0.07990146ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.10252953ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.07594895ms
out_mat_transpose_cute_row_cvectorized: [0.0, 2048.0, 1.0], validate True , time:0.08208942ms
out_mat_transpose_cute_row_rvectorized: [0.0, 2048.0, 1.0], validate True , time:0.05093598ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.05676866ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 2048.0, 1.0], validate True , time:0.05595016ms
                         out_f32_th: [0.0, 2048.0, 1.0], validate True , time:0.17065334ms
                out_f32_th_compiled: [0.0, 2048.0, 1.0], validate True , time:0.05812144ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096
                       out_original: [0.0, 1.0, 4096.0], validate False, time:0.09060454ms
                    out_f32_col2row: [0.0, 4096.0, 1.0], validate True , time:0.53396153ms
                    out_f32_row2col: [0.0, 4096.0, 1.0], validate True , time:0.23736954ms
                out_f32_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.25855660ms
                out_f32_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.25850701ms
                  out_f32x4_col2row: [0.0, 4096.0, 1.0], validate True , time:0.56433153ms
                  out_f32x4_row2col: [0.0, 4096.0, 1.0], validate True , time:0.25520277ms
              out_f32x4_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.34376645ms
              out_f32x4_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.08140516ms
       out_f32x4_shared_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.10327840ms
       out_f32x4_shared_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.13143134ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.10436797ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.11549497ms
 out_mat_transpose_cute_col2row_reg: [0.0, 4096.0, 1.0], validate True , time:0.22957921ms
 out_mat_transpose_cute_row2col_reg: [0.0, 4096.0, 1.0], validate True , time:0.15648627ms
    out_mat_transpose_cute_col_smem: [0.0, 4096.0, 1.0], validate True , time:0.23523784ms
    out_mat_transpose_cute_row_smem: [0.0, 4096.0, 1.0], validate True , time:0.15738535ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.20139265ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.14952707ms
out_mat_transpose_cute_row_cvectorized: [0.0, 4096.0, 1.0], validate True , time:0.15815616ms
out_mat_transpose_cute_row_rvectorized: [0.0, 4096.0, 1.0], validate True , time:0.09657621ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.10863233ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 4096.0, 1.0], validate True , time:0.10682678ms
                         out_f32_th: [0.0, 4096.0, 1.0], validate True , time:0.33217621ms
                out_f32_th_compiled: [0.0, 4096.0, 1.0], validate True , time:0.07057214ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192
                       out_original: [0.0, 1.0, 8192.0], validate False, time:0.17657781ms
                    out_f32_col2row: [0.0, 8192.0, 1.0], validate True , time:1.05990601ms
                    out_f32_row2col: [0.0, 8192.0, 1.0], validate True , time:0.46211433ms
                out_f32_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.50515103ms
                out_f32_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.50519037ms
                  out_f32_diagnonal: [0.0, 8192.0, 1.0], validate True , time:0.34358025ms
                  out_f32x4_col2row: [0.0, 8192.0, 1.0], validate True , time:1.10640693ms
                  out_f32x4_row2col: [0.0, 8192.0, 1.0], validate True , time:0.49680948ms
              out_f32x4_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.67155337ms
              out_f32x4_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.15931034ms
       out_f32x4_shared_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.20235109ms
       out_f32x4_shared_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.25867033ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.20493627ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.22880530ms
 out_mat_transpose_cute_col2row_reg: [0.0, 8192.0, 1.0], validate True , time:0.45240569ms
 out_mat_transpose_cute_row2col_reg: [0.0, 8192.0, 1.0], validate True , time:0.31271243ms
    out_mat_transpose_cute_col_smem: [0.0, 8192.0, 1.0], validate True , time:0.46673679ms
    out_mat_transpose_cute_row_smem: [0.0, 8192.0, 1.0], validate True , time:0.31364536ms
out_mat_transpose_cute_col_smem_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.39961338ms
out_mat_transpose_cute_row_smem_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.29810309ms
out_mat_transpose_cute_row_cvectorized: [0.0, 8192.0, 1.0], validate True , time:0.31140137ms
out_mat_transpose_cute_row_rvectorized: [0.0, 8192.0, 1.0], validate True , time:0.18832517ms
out_mat_transpose_cute_row_cvectorized_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.21277142ms
out_mat_transpose_cute_row_rvectorized_swizzled: [0.0, 8192.0, 1.0], validate True , time:0.20939684ms
                         out_f32_th: [0.0, 8192.0, 1.0], validate True , time:0.64468265ms
                out_f32_th_compiled: [0.0, 8192.0, 1.0], validate True , time:0.13783288ms
----------------------------------------------------------------------------------------------------------------------------------

```
