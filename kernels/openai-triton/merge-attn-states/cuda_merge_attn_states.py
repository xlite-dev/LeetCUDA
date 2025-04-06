import torch
import triton
import triton.language as tl
import torch.utils.cpp_extension

# 编译 CUDA 扩展
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void merge_attn_states_kernel_cuda(
    float* output,
    float* output_lse,
    const float* prefix_output,
    const float* prefix_lse,
    const float* suffix_output,
    const float* suffix_lse,
    int NUM_TOKENS,
    int NUM_HEADS,
    int HEAD_SIZE,
    bool OUTPUT_LSE
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int thread_idx = threadIdx.x;

    float p_lse = prefix_lse[head_idx * NUM_TOKENS + token_idx];
    float s_lse = suffix_lse[head_idx * NUM_TOKENS + token_idx];
    p_lse = std::isinf(p_lse) ? -std::numeric_limits<float>::infinity() : p_lse;
    s_lse = std::isinf(s_lse) ? -std::numeric_limits<float>::infinity() : s_lse;

    const float max_lse = fmaxf(p_lse, s_lse);
    p_lse = p_lse - max_lse;
    s_lse = s_lse - max_lse;
    const float p_se = expf(p_lse);
    const float s_se = expf(s_lse);
    const float out_se = p_se + s_se;

    if (OUTPUT_LSE) {
        float out_lse = logf(out_se) + max_lse;
        output_lse[head_idx * NUM_TOKENS + token_idx] = out_lse;
    }
    
    const int blk_offset = token_idx * NUM_HEADS * HEAD_SIZE + head_idx * HEAD_SIZE;
    const int thr_offset = thread_idx * 4;
    const float p_scale = p_se / out_se;
    const float s_scale = s_se / out_se;
    const float* prefix_output_blk = prefix_output + blk_offset;
    const float* suffix_output_blk = suffix_output + blk_offset;
    float* output_blk = output + blk_offset;

    if (thr_offset < HEAD_SIZE) {
        float4 p_out4 = ((const float4*)(prefix_output_blk))[thr_offset / 4];
        float4 s_out4 = ((const float4*)(suffix_output_blk))[thr_offset / 4];

        float4 out4 = make_float4(
            p_out4.x * p_scale + s_out4.x * s_scale,
            p_out4.y * p_scale + s_out4.y * s_scale,
            p_out4.z * p_scale + s_out4.z * s_scale,
            p_out4.w * p_scale + s_out4.w * s_scale
        );

        ((float4*)(output_blk))[thr_offset / 4] = out4;
    }
}

torch::Tensor merge_attn_states_cuda(
    torch::Tensor output,
    torch::Tensor output_lse,
    torch::Tensor prefix_output,
    torch::Tensor prefix_lse,
    torch::Tensor suffix_output,
    torch::Tensor suffix_lse,
    int HEAD_SIZE,
    int PADDED_HEAD_SIZE,
    bool OUTPUT_LSE
) {
    int NUM_TOKENS = output.size(0);
    int NUM_HEADS = output.size(1);
    assert(HEAD_SIZE % 4 == 0);
    dim3 grid(NUM_TOKENS, NUM_HEADS);
    dim3 block(HEAD_SIZE / 4);
    merge_attn_states_kernel_cuda<<<grid, block>>>(
        output.data_ptr<float>(),
        output_lse.data_ptr<float>(),
        prefix_output.data_ptr<float>(),
        prefix_lse.data_ptr<float>(),
        suffix_output.data_ptr<float>(),
        suffix_lse.data_ptr<float>(),
        NUM_TOKENS,
        NUM_HEADS,
        HEAD_SIZE,
        OUTPUT_LSE
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("merge_attn_states_cuda", &merge_attn_states_cuda, "Merge attention states (CUDA)");
}
"""

module = torch.utils.cpp_extension.load_inline(
    name='merge_attn_states_cuda',
    cpp_sources='',
    cuda_sources=cuda_source,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3'],
    is_python_module=True
)

# Triton 内核
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


def test_merge_attn_states(verbose: bool = False):
    # Set test parameters
    NUM_TOKENS = 1024
    NUM_HEADS = 128
    OUTPUT_LSE = False
    # Set HEAD_SIZE to a power of 2 in the test code
    HEAD_SIZE = 512
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
    grid = (NUM_TOKENS, NUM_HEADS)  # Program IDs are assigned according to token_idx and head_idx

    # Warmup and measure performance of merge_attn_states_kernel
    warmup_times = 2
    repeat_times = 10
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
    for _ in range(repeat_times):
        start.record()
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

    # Warmup and measure performance of merge_attn_states_kernel_cuda
    total_time_kernel_cuda = 0
    output_cuda = output.clone()
    output_lse_cuda = output_lse.clone()

    # Warmup
    for _ in range(warmup_times):
        module.merge_attn_states_cuda(
            output_cuda,
            output_lse_cuda,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            HEAD_SIZE,
            PADDED_HEAD_SIZE,
            OUTPUT_LSE
        )
    torch.cuda.synchronize()

    # Repeat and measure
    for _ in range(repeat_times):
        start.record()
        module.merge_attn_states_cuda(
            output_cuda,
            output_lse_cuda,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            HEAD_SIZE,
            PADDED_HEAD_SIZE,
            OUTPUT_LSE
        )
        end.record()
        torch.cuda.synchronize()
        total_time_kernel_cuda += start.elapsed_time(end)

    avg_time_kernel_cuda = total_time_kernel_cuda / repeat_times
    print(f"Average time taken by Triton merge_attn_states_kernel: {avg_time_kernel} ms")
    print(f"Average time taken by CUDA merge_attn_states_kernel: {avg_time_kernel_cuda} ms")

    # 精度对比
    if not torch.allclose(output_ref, output_cuda, rtol=1e-3, atol=1e-3):
        diff = torch.abs(output_ref - output_cuda)
        max_diff = torch.max(diff)
        max_diff_index = torch.argmax(diff)
        print(f"Max difference in output: {max_diff} at index {max_diff_index}")
        print(f"Triton output at max diff index: {output_ref.flatten()[max_diff_index]}")
        print(f"CUDA output at max diff index: {output_cuda.flatten()[max_diff_index]}")
    assert torch.allclose(output_ref, output_cuda, rtol=1e-3, atol=1e-3), "Output of Triton and CUDA do not match."

    if not torch.allclose(output_lse_ref, output_lse_cuda, rtol=1e-3, atol=1e-3):
        diff = torch.abs(output_lse_ref - output_lse_cuda)
        max_diff = torch.max(diff)
        max_diff_index = torch.argmax(diff)
        print(f"Max difference in output_lse: {max_diff} at index {max_diff_index}")
        print(f"Triton output_lse at max diff index: {output_lse_ref.flatten()[max_diff_index]}")
        print(f"CUDA output_lse at max diff index: {output_lse_cuda.flatten()[max_diff_index]}")
    assert torch.allclose(output_lse_ref, output_lse_cuda, rtol=1e-3, atol=1e-3), "Output_lse of Triton and CUDA do not match."

    print("All out lse test passed! All inf values are correctly replaced with -inf.")
    
    if verbose:
        print(prefix_lse)
        print(suffix_lse)
        print(output_lse)


if __name__ == "__main__":
    test_merge_attn_states()
    