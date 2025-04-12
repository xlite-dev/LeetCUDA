from typing import Optional

import torch
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
        "--generate-line-info -g", # for NCU debugging
        # "--use_fast_math"
    ], 
    extra_cflags=['-std=c++17'],
    verbose=True
)


def merge_attn_states_cuda(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: Optional[torch.Tensor] = None,
) -> None:
   lib.merge_attn_states_cuda(output, output_lse, prefix_output,
                              prefix_lse, suffix_output, suffix_lse)
