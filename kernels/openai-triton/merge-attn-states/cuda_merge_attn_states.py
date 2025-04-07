from torch.utils.cpp_extension import load


lib = load(
    name='merge_attn_states_cuda', 
    sources=['cuda_merge_attn_states.cu'], 
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math"
    ], 
    extra_cflags=['-std=c++17'],
)

merge_attn_states_cuda: callable = lib.merge_attn_states_cuda
