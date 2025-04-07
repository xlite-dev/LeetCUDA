"""Adapted from: https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_merge_attn_states.py"""
import triton
import triton.language as tl


# Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
# can be used to combine partial attention results (in the split-KV case)
@triton.jit
def merge_attn_states_triton(
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    output_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse,  # [NUM_HEADS, NUM_TOKENS]
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse,  # [NUM_HEADS, NUM_TOKENS]
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    num_tokens = tl.num_programs(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    p_lse = tl.load(prefix_lse + head_idx * num_tokens + token_idx)
    s_lse = tl.load(suffix_lse + head_idx * num_tokens + token_idx)
    p_lse = float('-inf') if p_lse == float('inf') else p_lse
    s_lse = float('-inf') if s_lse == float('inf') else s_lse

    max_lse = tl.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    out_se = (tl.exp(p_lse) + tl.exp(s_lse))

    if OUTPUT_LSE:
        out_lse = tl.log(out_se) + max_lse
        tl.store(output_lse + head_idx * num_tokens + token_idx, out_lse)

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE
    p_out = tl.load(prefix_output + token_idx * num_heads * HEAD_SIZE +
                    head_idx * HEAD_SIZE + head_arange,
                    mask=head_mask)
    s_out = tl.load(suffix_output + token_idx * num_heads * HEAD_SIZE +
                    head_idx * HEAD_SIZE + head_arange,
                    mask=head_mask)

    # NOTE(woosuk): Be careful with the numerical stability.
    # We should compute the scale first, and then multiply it with the output.
    # Do not multiply the output with tl.exp(p_lse) or tl.exp(s_lse) directly.
    p_scale = tl.exp(p_lse) / out_se
    s_scale = tl.exp(s_lse) / out_se
    out = p_out * p_scale + s_out * s_scale
    tl.store(output + token_idx * num_heads * HEAD_SIZE +
             head_idx * HEAD_SIZE + head_arange,
             out,
             mask=head_mask)


@triton.autotune(
    configs=[
        triton.Config({'num_warps': 2}, num_warps=2),
        triton.Config({'num_warps': 4}, num_warps=4),
        triton.Config({'num_warps': 8}, num_warps=8),
        triton.Config({'num_warps': 16}, num_warps=16),
        triton.Config({'num_warps': 32}, num_warps=32),
    ],
    key=['PADDED_HEAD_SIZE', 'OUTPUT_LSE'],
    use_cuda_graph=False,
)
@triton.jit
def merge_attn_states_triton_opt(
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    output_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse,  # [NUM_HEADS, NUM_TOKENS]
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse,  # [NUM_HEADS, NUM_TOKENS]
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    num_tokens = tl.num_programs(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    p_lse = tl.load(prefix_lse + head_idx * num_tokens + token_idx,
                    cache_modifier=".cg")
    s_lse = tl.load(suffix_lse + head_idx * num_tokens + token_idx,
                    cache_modifier=".cg")
    p_lse = float('-inf') if p_lse == float('inf') else p_lse
    s_lse = float('-inf') if s_lse == float('inf') else s_lse

    max_lse = tl.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    p_se = tl.exp(p_lse)
    s_se = tl.exp(s_lse)
    out_se = (p_se + s_se)

    if OUTPUT_LSE:
        out_lse = tl.log(out_se) + max_lse
        tl.store(output_lse + head_idx * num_tokens + token_idx, out_lse)

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE
    p_out = tl.load(prefix_output + token_idx * num_heads * HEAD_SIZE +
                    head_idx * HEAD_SIZE + head_arange,
                    mask=head_mask, cache_modifier=".cg")
    s_out = tl.load(suffix_output + token_idx * num_heads * HEAD_SIZE +
                    head_idx * HEAD_SIZE + head_arange,
                    mask=head_mask, cache_modifier=".cg")
    # NOTE(woosuk): Be careful with the numerical stability.
    # We should compute the scale first, and then multiply it with the output.
    # Do not multiply the output with tl.exp(p_lse) or tl.exp(s_lse) directly.
    p_scale = p_se / out_se
    s_scale = s_se / out_se
    out = p_out * p_scale + s_out * s_scale
    tl.store(output + token_idx * num_heads * HEAD_SIZE +
             head_idx * HEAD_SIZE + head_arange,
             out,
             mask=head_mask)