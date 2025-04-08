# Merge Attention States Kernel

## Introduction

Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005, can be used to combine partial attention results (in the split-KV case). The Triton and CUDA kernels here are modified from [vllm/attention/ops/triton_merge_attn_states.py](https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_merge_attn_states.py) and [vllm/pull/16173](https://github.com/vllm-project/vllm/pull/16173). Use CUDA kernel instead of Triton to minimize CPU overhead. Compared to the Triton kernel, the CUDA kernel implemented in this PR can achieve a maximum speedup of over `3x`. End2End (vLLM) performance improved for R1 with PP=3 + TP=8 on L20,  4K IN:1K OUT (TTFT 5687.80 ms -> 5654.02 ms), 8K IN:64 OUT (TTFT  17673.93ms -> 14785.00ms).

- [x] float32
- [x] float16
- [x] bfloat16
- [x] dispatch by scalar_t
- [x] fallback strategy 
- [x] unit tests (performance & correctness)
- [x] end2end test   

 
## Performance

- float32 (performance & correctness)

```bash
pytest -s test_merge_attn_states.py # please install pytest via `pip install pytest`
----------------------------------------------------------------------------------------------------
.
NUM_TOKENS:512, NUM_HEADS:16, HEAD_SIZE:128, DTYPE: torch.float32, Device: NVIDIA L20
 Torch time: 0.150216ms
Triton time: 0.051350ms
  CUDA time: 0.016072ms, Performance: 3.19502x
----------------------------------------------------------------------------------------------------
Output all match, max abs diff:
 (CUDA  vs Triton): 4.76837158203125e-07
(Triton vs Torch) : 4.76837158203125e-07
  (CUDA vs Torch) : 2.384185791015625e-07
----------------------------------------------------------------------------------------------------
Output LSE all match, max abs diff:
(Triton vs Torch) : 4.76837158203125e-07
  (CUDA vs Torch) : 0.0
 (CUDA  vs Triton): 4.76837158203125e-07
----------------------------------------------------------------------------------------------------
All output values test passed! All inf values are correctly replaced with -inf.
----------------------------------------------------------------------------------------------------
```

- float16 (performance & correctness)

```bash
----------------------------------------------------------------------------------------------------
NUM_TOKENS:512, NUM_HEADS:16, HEAD_SIZE:128, DTYPE: torch.float16, Device: NVIDIA L20
 Torch time: 0.149299ms
Triton time: 0.050995ms
  CUDA time: 0.015722ms, Performance: 3.24364x
----------------------------------------------------------------------------------------------------
Output all match, max abs diff:
 (CUDA  vs Triton): 0.0009765625
(Triton vs Torch) : 0.0015368461608886719
  (CUDA vs Torch) : 0.0015368461608886719
----------------------------------------------------------------------------------------------------
Output LSE all match, max abs diff:
(Triton vs Torch) : 2.384185791015625e-07
  (CUDA vs Torch) : 0.0
 (CUDA  vs Triton): 2.384185791015625e-07
----------------------------------------------------------------------------------------------------
All output values test passed! All inf values are correctly replaced with -inf.
----------------------------------------------------------------------------------------------------
```

- bfloat16  (performance & correctness)

```bash
----------------------------------------------------------------------------------------------------
NUM_TOKENS:4096, NUM_HEADS:16, HEAD_SIZE:128, DTYPE: torch.bfloat16, Device: NVIDIA L20
 Torch time: 0.322397ms
Triton time: 0.081408ms
  CUDA time: 0.026824ms, Performance: 3.03489x
----------------------------------------------------------------------------------------------------
Output all match, max abs diff:
 (CUDA  vs Triton): 0.015625
(Triton vs Torch) : 0.011169910430908203
  (CUDA vs Torch) : 0.011169910430908203
----------------------------------------------------------------------------------------------------
Output LSE all match, max abs diff:
(Triton vs Torch) : 2.384185791015625e-07
  (CUDA vs Torch) : 0.0
 (CUDA  vs Triton): 2.384185791015625e-07
----------------------------------------------------------------------------------------------------
All output values test passed! All inf values are correctly replaced with -inf.
----------------------------------------------------------------------------------------------------
```

## End2End test with vLLM

R1 671B with L20x3, PP=3, TP=8, vLLM with [vllm/pull/16173](https://github.com/vllm-project/vllm/pull/16173)

- launch cmd

```bash
# export VLLM_DISABLE_MERGE_ATTN_CUDA_OP=1 # if don't want to use this custom CUDA kernel

nohup python3 -m vllm.entrypoints.openai.api_server \
        --model=/workspace/dev/hf_models/DeepSeek-R1 \
        --dtype=auto \
        --block-size 32 \
        --tokenizer-mode=slow \
        --max-model-len 32768 \
        --max-num-batched-tokens 2048 \
        --tensor-parallel-size 8 \
        --pipeline-parallel-size 3 \
        --gpu-memory-utilization 0.90 \
        --max-num-seqs 128 \
        --trust-remote-code \
        --no-enable-prefix-caching \
        --enable-chunked-prefill=True \
        --disable-custom-all-reduce \
        --port 8862 > vllm.R1.log.3 2>&1 &
```

### 4K IN:1K OUT (TTFT 5687.80 ms -> 5654.02 ms)

- w/o this opt, 4K IN:1K OUT

```bash

Maximum request concurrency: 16
============ Serving Benchmark Result ============
Successful requests:                     32
Benchmark duration (s):                  207.14
Total input tokens:                      131072
Total generated tokens:                  32768
Request throughput (req/s):              0.15
Output token throughput (tok/s):         158.19
Total Token throughput (tok/s):          790.96
---------------Time to First Token----------------
Mean TTFT (ms):                          5687.80
Median TTFT (ms):                        3969.86
P99 TTFT (ms):                           11952.93
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          95.51
Median TPOT (ms):                        96.38
P99 TPOT (ms):                           98.71
---------------Inter-token Latency----------------
Mean ITL (ms):                           95.51
Median ITL (ms):                         89.71
P99 ITL (ms):                            97.03
==================================================
```

- w/ this opt, 4K IN:1K OUT (TTFT 5687.80 ms -> 5654.02 ms)
```bash
Maximum request concurrency: 16
============ Serving Benchmark Result ============
Successful requests:                     32
Benchmark duration (s):                  206.65
Total input tokens:                      131072
Total generated tokens:                  32768
Request throughput (req/s):              0.15
Output token throughput (tok/s):         158.57
Total Token throughput (tok/s):          792.83
---------------Time to First Token----------------
Mean TTFT (ms):                          5654.02
Median TTFT (ms):                        3958.66
P99 TTFT (ms):                           11861.09
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          95.30
Median TPOT (ms):                        95.98
P99 TPOT (ms):                           98.70
---------------Inter-token Latency----------------
Mean ITL (ms):                           95.30
Median ITL (ms):                         89.62
P99 ITL (ms):                            96.89
==================================================
```

### 8K IN:64 OUT (TTFT  17673.93ms -> 14785.00ms)

- w/o this opt, 8K IN:64 OUT

```bash
Maximum request concurrency: 16
============ Serving Benchmark Result ============
Successful requests:                     16
Benchmark duration (s):                  61.38
Total input tokens:                      131072
Total generated tokens:                  1024
Request throughput (req/s):              0.26
Output token throughput (tok/s):         16.68
Total Token throughput (tok/s):          2152.14
---------------Time to First Token----------------
Mean TTFT (ms):                          17673.93
Median TTFT (ms):                        16260.86
P99 TTFT (ms):                           33269.41
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          540.71
Median TPOT (ms):                        558.80
P99 TPOT (ms):                           613.13
---------------Inter-token Latency----------------
Mean ITL (ms):                           540.71
Median ITL (ms):                         268.09
P99 ITL (ms):                            6782.42
==================================================
```

- w/ this opt, 8K IN:64 OUT (TTFT  17673.93ms -> 14785.00ms)

```bash
Maximum request concurrency: 16
============ Serving Benchmark Result ============
Successful requests:                     16
Benchmark duration (s):                  42.52
Total input tokens:                      131072
Total generated tokens:                  1024
Request throughput (req/s):              0.38
Output token throughput (tok/s):         24.08
Total Token throughput (tok/s):          3106.85
---------------Time to First Token----------------
Mean TTFT (ms):                          14785.00
Median TTFT (ms):                        14644.31
P99 TTFT (ms):                           24556.74
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          387.38
Median TPOT (ms):                        388.55
P99 TPOT (ms):                           499.89
---------------Inter-token Latency----------------
Mean ITL (ms):                           387.38
Median ITL (ms):                         271.57
P99 ITL (ms):                            1056.03
==================================================
```
