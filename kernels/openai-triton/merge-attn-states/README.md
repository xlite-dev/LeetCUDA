# Merge Attention States Kernel

## Introduction

Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005, can be used to combine partial attention results (in the split-KV case). The Triton and CUDA kernels here are modified from [vllm/attention/ops/triton_merge_attn_states.py](https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_merge_attn_states.py) and [vllm/pull/16173](https://github.com/vllm-project/vllm/pull/16173). Use CUDA kernel instead of Triton to minimize CPU overhead. Compared to the Triton kernel, the CUDA kernel implemented in this PR can achieve a maximum speedup of over `3x`. End2End performance improved for R1 with PP=3 + TP=8 on L20,  4K IN:1K OUT (TTFT 5687.80 ms -> 5654.02 ms), 8K IN:64 OUT (TTFT  15188.70ms -> 14534.50ms).

- [x] float32
- [x] float16
- [x] bfloat16
- [x] dispatch by scalar_t
- [x] fallback strategy 
- [x] unit tests (performance & correctness)
- [x] end2end test   
- [x] CEval benchmark 

## Performance

- float32 (performance & correctness)

```bash
pytest -s test_merge_attn_states.py
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

## End2End test  

R1 671B with L20x3, PP=3, TP=8

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

### 8K IN:64 OUT (TTFT 15188.70ms -> 14534.50ms)

- w/o this opt, 8K IN:64 OUT

```bash
Maximum request concurrency: 16
============ Serving Benchmark Result ============
Successful requests:                     16
Benchmark duration (s):                  42.55
Total input tokens:                      131072
Total generated tokens:                  1024
Request throughput (req/s):              0.38
Output token throughput (tok/s):         24.07
Total Token throughput (tok/s):          3104.52
---------------Time to First Token----------------
Mean TTFT (ms):                          15188.70
Median TTFT (ms):                        15051.09
P99 TTFT (ms):                           24925.20
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          383.59
Median TPOT (ms):                        385.05
P99 TPOT (ms):                           495.82
---------------Inter-token Latency----------------
Mean ITL (ms):                           383.59
Median ITL (ms):                         268.32
P99 ITL (ms):                            1057.60
==================================================
```

- w/ this opt, 8K IN:64 OUT (TTFT  15188.70ms -> 14534.50ms)

```bash
Maximum request concurrency: 16
============ Serving Benchmark Result ============
Successful requests:                     16
Benchmark duration (s):                  41.58
Total input tokens:                      131072
Total generated tokens:                  1024
Request throughput (req/s):              0.38
Output token throughput (tok/s):         24.63
Total Token throughput (tok/s):          3177.20
---------------Time to First Token----------------
Mean TTFT (ms):                          14534.50
Median TTFT (ms):                        14393.85
P99 TTFT (ms):                           24302.77
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          385.79
Median TPOT (ms):                        387.44
P99 TPOT (ms):                           498.96
---------------Inter-token Latency----------------
Mean ITL (ms):                           385.79
Median ITL (ms):                         270.91
P99 ITL (ms):                            1055.40
==================================================
```

## CEval benchmark  (0.90197884615385)

We use [evalscope](https://github.com/modelscope/evalscope) to run benchmark on `CEval` dataset.

```bash
evalscope eval \
 --model /workspace/dev/hf_models/DeepSeek-R1 \
 --api-url http://0.0.0.0:8862/v1/chat/completions \
 --api-key EMPTY \
 --eval-batch-size 32 \
 --eval-type service \
 --datasets ceval \
 --dataset-args '{"ceval": {"local_path": "/workspace/dev/openllm/benchmarks/data/ceval"}}'
```

Total AverageAccuracy: 0.90197884615385
```bash
2025-04-08 23:22:21,580 - evalscope - INFO - Report table:
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| Model       | Dataset   | Metric          | Subset                                   |   Num |   Score | Cat.0          |
+=============+===========+=================+==========================================+=======+=========+================+
| DeepSeek-R1 | ceval     | AverageAccuracy | modern_chinese_history                   |    23 |  0.8696 | Humanities     |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | ideological_and_moral_cultivation        |    19 |  1      | Humanities     |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | logic                                    |    22 |  0.9091 | Humanities     |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | law                                      |    24 |  0.875  | Humanities     |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | chinese_language_and_literature          |    23 |  0.8261 | Humanities     |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | art_studies                              |    33 |  0.9091 | Humanities     |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | professional_tour_guide                  |    29 |  0.9655 | Humanities     |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | legal_professional                       |    23 |  0.913  | Humanities     |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | high_school_chinese                      |    19 |  0.7895 | Humanities     |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | high_school_history                      |    20 |  0.95   | Humanities     |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | middle_school_history                    |    22 |  0.9545 | Humanities     |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | civil_servant                            |    47 |  0.8723 | Other          |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | sports_science                           |    19 |  0.8947 | Other          |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | plant_protection                         |    22 |  1      | Other          |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | basic_medicine                           |    19 |  1      | Other          |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | clinical_medicine                        |    22 |  0.9091 | Other          |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | urban_and_rural_planner                  |    46 |  0.8913 | Other          |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | accountant                               |    49 |  0.9184 | Other          |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | fire_engineer                            |    31 |  1      | Other          |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | environmental_impact_assessment_engineer |    31 |  0.9032 | Other          |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | tax_accountant                           |    49 |  0.9184 | Other          |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | physician                                |    49 |  0.9184 | Other          |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | computer_network                         |    19 |  0.7895 | STEM           |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | operating_system                         |    19 |  0.8947 | STEM           |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | computer_architecture                    |    21 |  1      | STEM           |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | college_programming                      |    37 |  0.9189 | STEM           |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | college_physics                          |    19 |  0.8947 | STEM           |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | college_chemistry                        |    24 |  0.9167 | STEM           |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | advanced_mathematics                     |    19 |  0.9474 | STEM           |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | probability_and_statistics               |    18 |  0.7778 | STEM           |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | discrete_mathematics                     |    16 |  0.5625 | STEM           |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | electrical_engineer                      |    37 |  0.7027 | STEM           |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | metrology_engineer                       |    24 |  0.9583 | STEM           |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | high_school_mathematics                  |    18 |  0.7778 | STEM           |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | high_school_physics                      |    19 |  1      | STEM           |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | high_school_chemistry                    |    19 |  0.9474 | STEM           |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | high_school_biology                      |    19 |  0.9474 | STEM           |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | middle_school_mathematics                |    19 |  1      | STEM           |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | middle_school_biology                    |    21 |  0.8571 | STEM           |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | middle_school_physics                    |    19 |  1      | STEM           |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | middle_school_chemistry                  |    20 |  1      | STEM           |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | veterinary_medicine                      |    23 |  0.8696 | STEM           |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | college_economics                        |    55 |  0.8727 | Social Science |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | business_administration                  |    33 |  0.8182 | Social Science |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | marxism                                  |    19 |  0.9474 | Social Science |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | mao_zedong_thought                       |    24 |  1      | Social Science |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | education_science                        |    29 |  0.931  | Social Science |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | teacher_qualification                    |    44 |  0.9318 | Social Science |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | high_school_politics                     |    19 |  1      | Social Science |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | high_school_geography                    |    19 |  0.9474 | Social Science |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | middle_school_politics                   |    21 |  1      | Social Science |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
| DeepSeek-R1 | ceval     | AverageAccuracy | middle_school_geography                  |    12 |  0.9167 | Social Science |
+-------------+-----------+-----------------+------------------------------------------+-------+---------+----------------+
```