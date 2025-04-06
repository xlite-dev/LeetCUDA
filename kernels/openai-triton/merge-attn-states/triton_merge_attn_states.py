"""Adapted from: https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_merge_attn_states.py"""
import torch
import triton
import triton.language as tl


# Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
# can be used to combine partial attention results (in the split-KV case)
@triton.jit
def merge_attn_states_kernel(
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
def merge_attn_states_kernel_opt(
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


def test_merge_attn_states(verbose: bool = False):
    # Set test parameters
    NUM_TOKENS = 1024
    NUM_QUERY_HEADS = 128
    OUTPUT_LSE = verbose
    # Set HEAD_SIZE to a power of 2 in the test code
    HEAD_SIZE = 512
    PADDED_HEAD_SIZE = triton.next_power_of_2(HEAD_SIZE)
    # NUM_WARPS = triton.next_power_of_2(int(PADDED_HEAD_SIZE / 32))  

    # Generate test inputs
    # prefix_lse and suffix_lse contain inf and normal values
    prefix_lse = torch.randn(NUM_QUERY_HEADS, NUM_TOKENS, dtype=torch.float32, device="cuda")
    suffix_lse = torch.randn(NUM_QUERY_HEADS, NUM_TOKENS, dtype=torch.float32, device="cuda")

    # Generate boolean masks
    mask_prefix = torch.rand(NUM_QUERY_HEADS, NUM_TOKENS) < 0.1
    mask_suffix = torch.rand(NUM_QUERY_HEADS, NUM_TOKENS) < 0.1
    # Ensure that the same position is not True at the same time
    combined_mask = torch.logical_and(mask_prefix, mask_suffix)
    mask_prefix = torch.logical_and(mask_prefix, ~combined_mask)
    mask_suffix = torch.logical_and(mask_suffix, ~combined_mask)

    prefix_lse[mask_prefix] = float('inf')
    suffix_lse[mask_suffix] = float('inf')

    # Other input tensors (need to be initialized but no actual calculation needed)
    output = torch.zeros(
        (NUM_TOKENS, NUM_QUERY_HEADS, HEAD_SIZE),
        dtype=torch.float32,
        device="cuda"
    )
    output_lse = torch.zeros(
        (NUM_QUERY_HEADS, NUM_TOKENS),
        dtype=torch.float32,
        device="cuda"
    )
    prefix_output = torch.randn(
        (NUM_TOKENS, NUM_QUERY_HEADS, HEAD_SIZE),
        dtype=torch.float32,
        device="cuda"
    )
    suffix_output = torch.randn(
        (NUM_TOKENS, NUM_QUERY_HEADS, HEAD_SIZE),
        dtype=torch.float32,
        device="cuda"
    )

    output_ref = output.clone()
    output_lse_ref = output_lse.clone()
    torch.cuda.synchronize()

    # Run the Triton kernel
    grid = (NUM_TOKENS, NUM_QUERY_HEADS)  # Program IDs are assigned according to token_idx and head_idx

    # Warmup and measure performance of merge_attn_states_kernel
    warmup_times = 2
    repeat_times = 100
    total_time_kernel = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(warmup_times):
        merge_attn_states_kernel[grid](
            output_ref,
            output_lse_ref,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            HEAD_SIZE,
            PADDED_HEAD_SIZE,
            OUTPUT_LSE,
        )
    torch.cuda.synchronize()

    # Repeat and measure
    start.record()
    for _ in range(repeat_times):
        merge_attn_states_kernel[grid](
            output_ref,
            output_lse_ref,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            HEAD_SIZE,
            PADDED_HEAD_SIZE,
            OUTPUT_LSE,
        )
    end.record()
    torch.cuda.synchronize()
    total_time_kernel += start.elapsed_time(end)

    avg_time_kernel = total_time_kernel / repeat_times

    # Warmup and measure performance of merge_attn_states_kernel_opt
    total_time_kernel_opt = 0
    start_opt = torch.cuda.Event(enable_timing=True)
    end_opt = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(warmup_times):
        merge_attn_states_kernel_opt[grid](
            output,
            output_lse,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            HEAD_SIZE,
            PADDED_HEAD_SIZE,
            OUTPUT_LSE,
            # num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()

    # Repeat and measure
    start_opt.record()
    for _ in range(repeat_times):
        merge_attn_states_kernel_opt[grid](
            output,
            output_lse,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            HEAD_SIZE,
            PADDED_HEAD_SIZE,
            OUTPUT_LSE,
        )
    end_opt.record()
    torch.cuda.synchronize()
    total_time_kernel_opt += start_opt.elapsed_time(end_opt)

    avg_time_kernel_opt = total_time_kernel_opt / repeat_times
    print(f"Average time taken by merge_attn_states_kernel: {avg_time_kernel} ms")
    print(f"Average time taken by merge_attn_states_kernel_opt: {avg_time_kernel_opt} ms")
    best_config = merge_attn_states_kernel_opt.best_config
    print(f"Optimal config for merge_attn_states_kernel_opt: {best_config}, "
          f"keys: {merge_attn_states_kernel_opt.keys}")
    # Verify the results: Check if the inf values in output_lse are replaced with -inf
    if verbose:
      for h in range(NUM_QUERY_HEADS):
          for t in range(NUM_TOKENS):
              p_lse = float('-inf') if prefix_lse[h, t] == float('inf') else prefix_lse[h, t]
              s_lse = float('-inf') if suffix_lse[h, t] == float('inf') else suffix_lse[h, t]
              # Ensure that both p_lse and s_lse are Tensor types
              if isinstance(p_lse, float):
                  p_lse = torch.tensor(p_lse, dtype=torch.float32, device=s_lse.device)
              if isinstance(s_lse, float):
                  s_lse = torch.tensor(s_lse, dtype=torch.float32, device=p_lse.device)
              max_lse = torch.max(p_lse, s_lse)
              out_se = (torch.exp(p_lse - max_lse) + torch.exp(s_lse - max_lse))
              expected_out_lse = torch.log(out_se) + max_lse

              # Print debugging information
              if not torch.allclose(
                  output_lse_ref[h, t],
                  expected_out_lse,
                  rtol=1e-3,
                  atol=1e-3
              ):
                  print(f"Calculation result of Head {h}, Token {t} does not match:")
                  print(f"output_lse: {output_lse_ref[h, t]}")
                  print(f"expected_out_lse: {expected_out_lse}")
                  print(f"p_lse: {p_lse}")
                  print(f"s_lse: {s_lse}")
                  print(f"max_lse: {max_lse}")
                  print(f"out_se: {out_se}")

              # Verify if the calculation of output_lse meets the expected logic
              assert torch.allclose(
                  output_lse_ref[h, t],
                  expected_out_lse,
                  rtol=1e-3,
                  atol=1e-3
              ), f"Calculation result of Head {h}, Token {t} does not match"

              # Verify output
              p_scale = torch.exp(p_lse - max_lse) / out_se
              s_scale = torch.exp(s_lse - max_lse) / out_se
              expected_out = prefix_output[t, h] * p_scale + suffix_output[t, h] * s_scale
              assert torch.allclose(
                  output_ref[t, h],
                  expected_out,
                  rtol=1e-3,
                  atol=1e-3
              ), f"Output calculation result of Head {h}, Token {t} does not match"

    # Verify the correctness of merge_attn_states_kernel_opt
    assert torch.allclose(output, output_ref, rtol=1e-3, atol=1e-3), "Output of merge_attn_states_kernel_opt does not match the reference."
    assert torch.allclose(output_lse, output_lse_ref, rtol=1e-3, atol=1e-3), "Output_lse of merge_attn_states_kernel_opt does not match the reference."

    print("All correctness test passed! All inf values are correctly replaced with -inf.")
   
    if verbose:
        print(prefix_lse)
        print(suffix_lse)
        print(output_lse)


if __name__ == "__main__":
    test_merge_attn_states()
    