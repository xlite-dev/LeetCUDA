import torch
import triton
import argparse
from cuda_merge_attn_states import merge_attn_states_cuda
from triton_merge_attn_states import merge_attn_states_triton


def get_args():
    parser = argparse.ArgumentParser(description="hgemm benchmark")
    parser.add_argument("--num-query-heads", "-q", type=int, default=128, help="num query heads")
    parser.add_argument("--head-size", "-d", type=int, default=512, help="headsize per head")
    parser.add_argument("--num-tokens", "-n", type=int, default=1024, help="num tokens")
    parser.add_argument("--verbose", "-v", action="store_true", help="verbose")
    parser.add_argument("--loop-over-head", "-loop", "-l", action="store_true", help="verbose")
    parser.add_argument("--output-lse", "-lse", action="store_true", help="check output lse")
    parser.add_argument("--warmup", "-w", type=int, default=2, help="warmup")
    parser.add_argument("--repeat", "-r", type=int, default=20, help="repeat")
    return parser.parse_args()


def test_merge_attn_states():
    args = get_args()
    print("-" * 100)
    print(args)
    print("-" * 100)
    # Set test parameters
    NUM_TOKENS = args.num_tokens
    # Num query heads
    NUM_HEADS = args.num_query_heads 
    OUTPUT_LSE = args.output_lse
    # Set HEAD_SIZE to a power of 2 in the test code
    HEAD_SIZE = args.head_size 
    PADDED_HEAD_SIZE = triton.next_power_of_2(HEAD_SIZE)

    # Generate test inputs
    # prefix_lse and suffix_lse contain inf and normal values
    prefix_lse = torch.randn(NUM_HEADS, NUM_TOKENS, dtype=torch.float32, device="cuda")
    suffix_lse = torch.randn(NUM_HEADS, NUM_TOKENS, dtype=torch.float32, device="cuda")

    # Generate boolean masks
    mask_prefix = torch.rand(NUM_HEADS, NUM_TOKENS) < 0.1
    mask_suffix = torch.rand(NUM_HEADS, NUM_TOKENS) < 0.1
    # Ensure that the same position is not True at the same time
    combined_mask = torch.logical_and(mask_prefix, mask_suffix)
    mask_prefix = torch.logical_and(mask_prefix, ~combined_mask)
    mask_suffix = torch.logical_and(mask_suffix, ~combined_mask)

    prefix_lse[mask_prefix] = float('inf')
    suffix_lse[mask_suffix] = float('inf')

    # Other input tensors (need to be initialized but no actual calculation needed)
    output = torch.zeros(
        (NUM_TOKENS, NUM_HEADS, HEAD_SIZE),
        dtype=torch.float32,
        device="cuda"
    )
    output_lse = torch.zeros(
        (NUM_HEADS, NUM_TOKENS),
        dtype=torch.float32,
        device="cuda"
    )
    prefix_output = torch.randn(
        (NUM_TOKENS, NUM_HEADS, HEAD_SIZE),
        dtype=torch.float32,
        device="cuda"
    )
    suffix_output = torch.randn(
        (NUM_TOKENS, NUM_HEADS, HEAD_SIZE),
        dtype=torch.float32,
        device="cuda"
    )

    output_ref = output.clone()
    output_lse_ref = output_lse.clone()

    # Run the Triton kernel
    grid = (NUM_TOKENS, NUM_HEADS)
    # Warmup and measure performance of merge_attn_states_kernel
    warmup_times = args.warmup
    repeat_times = args.repeat
    total_time_kernel = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(warmup_times):
        merge_attn_states_triton[grid](
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
    for _ in range(repeat_times):
        start.record()
        merge_attn_states_triton[grid](
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

    # Warmup and measure performance of merge_attn_states_cuda
    total_time_kernel_cuda = 0
    output_cuda = output.clone()
    output_lse_cuda = output_lse.clone() if OUTPUT_LSE else None
    LOOP_OVER_HEAD = args.loop_over_head

    # Warmup
    for _ in range(warmup_times):
        merge_attn_states_cuda(
            output_cuda,
            output_lse_cuda,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            OUTPUT_LSE,
            LOOP_OVER_HEAD
        )
    torch.cuda.synchronize()

    # Repeat and measure
    for _ in range(repeat_times):
        start.record()
        merge_attn_states_cuda(
            output_cuda,
            output_lse_cuda,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            OUTPUT_LSE,
            LOOP_OVER_HEAD
        )
        end.record()
        torch.cuda.synchronize()
        total_time_kernel_cuda += start.elapsed_time(end)

    avg_time_kernel_cuda = total_time_kernel_cuda / repeat_times
    print(f"Average time taken by Triton merge_attn_states kernel: {avg_time_kernel} ms")
    print(f"Average time taken by   CUDA merge_attn_states kernel: {avg_time_kernel_cuda} ms")
    print("-" * 100)

    if not torch.allclose(output_ref, output_cuda, rtol=1e-3, atol=1e-3):
        diff = torch.abs(output_ref - output_cuda)
        max_diff = torch.max(diff)
        max_diff_index = torch.argmax(diff)
        print(f"Max difference in output: {max_diff} at index {max_diff_index}")
        print(f"Triton output at max diff index: {output_ref.flatten()[max_diff_index]}")
        print(f"CUDA output at max diff index: {output_cuda.flatten()[max_diff_index]}")
    assert torch.allclose(output_ref, output_cuda, rtol=1e-3, atol=1e-3), "Output of Triton and CUDA do not match."
    print("Output of Triton and CUDA all match.")

    if OUTPUT_LSE:
      if not torch.allclose(output_lse_ref, output_lse_cuda, rtol=1e-3, atol=1e-3):
          diff = torch.abs(output_lse_ref - output_lse_cuda)
          max_diff = torch.max(diff)
          max_diff_index = torch.argmax(diff)
          print(f"Max difference in output_lse: {max_diff} at index {max_diff_index}")
          print(f"Triton output_lse at max diff index: {output_lse_ref.flatten()[max_diff_index]}")
          print(f"CUDA output_lse at max diff index: {output_lse_cuda.flatten()[max_diff_index]}")
      assert torch.allclose(output_lse_ref, output_lse_cuda, rtol=1e-3, atol=1e-3), "Output_lse of Triton and CUDA do not match."
      print("Output LSE of Triton and CUDA all match.")

    print("All output values test passed! All inf values are correctly replaced with -inf.")
    print("-" * 100)
    
    if args.verbose:
        print(prefix_lse)
        print(suffix_lse)
        print(output_lse)


if __name__ == "__main__":
    test_merge_attn_states()
    