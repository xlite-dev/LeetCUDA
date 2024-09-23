# Reduce

## 0x00 说明

包含以下内容：

- [X] warp_reduce_fp32/fp16/bf16_kernel
- [X] block_reduce_fp32_kernel
- [X] block_all_reduce_sum_f32_f32_kernel
- [X] block_all_reduce_sum_f32x4_f32_kernel(float4向量化版本)
- [X] block_all_reduce_sum_f16_f16_kernel(fp16版本，使用fp16 acc)
- [X] block_all_reduce_sum_f16_f32_kernel(fp16版本，使用fp32 acc)
- [X] block_all_reduce_sum_f16x2_f16_kernel(fp16向量化版本，使用fp16 acc)
- [X] block_all_reduce_sum_f16x2_f32_kernel(fp16向量化版本，使用fp32 acc)
- [X] block_all_reduce_sum_f16x8_pack_f16_kernel(fp16向量化版本，使用fp16 acc, pack)
- [X] block_all_reduce_sum_f16x8_pack_f32_kernel(fp16向量化版本，使用fp32 acc, pack)
- [X] block_all_reduce_sum_bf16_bf16_kernel(bf16版本，使用bf16 acc)
- [X] block_all_reduce_sum_bf16_f32_kernel(bf16版本，使用fp32 acc)
- [X] block_all_reduce_sum_bf16x2_bf16_kernel(bf16向量化版本，使用bf16 acc)
- [X] block_all_reduce_sum_bf16x2_f32_kernel(bf16向量化版本，使用fp32 acc)
- [X] block_all_reduce_sum_fp8_e4m3_f16_kernel(fp8_e4m3版本，使用fp16 acc)
- [X] block_all_reduce_sum_fp8_e5m2_f16_kernel(fp8_e5m2版本，使用fp16 acc)
- [X] block_all_reduce_sum_i8_i32_kernel(i8版本，使用i32 acc)
- [X] PyTorch bindings for block reduce **fp32/fp16/bf16/fp8/i8** kernels

所有支持的block all reduce kernel:

```c++
// packed_type, acc_type, th_type, element_type, n_elements_per_pack, out_type
TORCH_BINDING_REDUCE(f32,        f32,  torch::kFloat32,       float,              1, float)
TORCH_BINDING_REDUCE(f32x4,      f32,  torch::kFloat32,       float,              4, float)
TORCH_BINDING_REDUCE(f16,        f16,  torch::kHalf,          half,               1, float)
TORCH_BINDING_REDUCE(f16,        f32,  torch::kHalf,          half,               1, float)
TORCH_BINDING_REDUCE(f16x2,      f16,  torch::kHalf,          half,               2, float)
TORCH_BINDING_REDUCE(f16x2,      f32,  torch::kHalf,          half,               2, float)
TORCH_BINDING_REDUCE(f16x8_pack, f16,  torch::kHalf,          half,               8, float)
TORCH_BINDING_REDUCE(f16x8_pack, f32,  torch::kHalf,          half,               8, float)
TORCH_BINDING_REDUCE(bf16,       bf16, torch::kBFloat16,      __nv_bfloat16,      1, float)
TORCH_BINDING_REDUCE(bf16,       f32,  torch::kBFloat16,      __nv_bfloat16,      1, float)
TORCH_BINDING_REDUCE(bf16x2,     bf16, torch::kBFloat16,      __nv_bfloat16,      2, float)
TORCH_BINDING_REDUCE(bf16x2,     f32,  torch::kBFloat16,      __nv_bfloat16,      2, float)
TORCH_BINDING_REDUCE(fp8_e4m3,   f16,  torch::kFloat8_e4m3fn, __nv_fp8_storage_t, 1, float)
TORCH_BINDING_REDUCE(fp8_e5m2,   f16,  torch::kFloat8_e5m2,   __nv_fp8_storage_t, 1, float)
TORCH_BINDING_REDUCE(i8,         i32,  torch::kInt8,          int8_t,             1, int32_t)
```

## 测试

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长
export TORCH_CUDA_ARCH_LIST=Ada 
python3 block_all_reduce.py
```

输出:

```bash
--------------------------------------------------------------------------------
       out_f32f32: 1210.16015625  , time:0.05203605ms
     out_f32x4f32: 1210.17163086  , time:0.01288271ms
    out_f32f32_th: 1210.17041016  , time:0.01749063ms
--------------------------------------------------------------------------------
       out_f16f16: 1211.90966797  , time:0.05204892ms
       out_f16f32: 1210.20922852  , time:0.05203557ms
     out_f16x2f32: 1209.62231445  , time:0.02830338ms
     out_f16x2f16: 1209.65844727  , time:0.02812505ms
 out_f16x8packf16: 1208.61621094  , time:0.01047134ms
 out_f16x8packf32: 1210.20263672  , time:0.01042914ms
    out_f16f16_th: 1210.00000000  , time:0.01252866ms
--------------------------------------------------------------------------------
     out_bf16bf16: 1190.73437500  , time:0.05206537ms
      out_bf16f32: 1206.96472168  , time:0.05202317ms
    out_bf16x2f32: 1210.72680664  , time:0.02816391ms
   out_bf16x2bf16: 1218.03125000  , time:0.02937555ms
  out_bf16bf16_th: 1208.00000000  , time:0.01241899ms
--------------------------------------------------------------------------------
    out_f8e4m3f16: 1157.98828125  , time:0.05200744ms
 out_f8e4m3f16_th: 1158.00000000  , time:0.01262856ms
--------------------------------------------------------------------------------
    out_f8e5m2f16: 1535.91503906  , time:0.05203843ms
 out_f8e5m2f16_th: 1535.00000000  , time:0.01274371ms
--------------------------------------------------------------------------------
        out_i8i32: 693            , time:0.03517842ms
     out_i8i32_th: 693            , time:0.05223250ms
--------------------------------------------------------------------------------
```
