# Merge Attention States Kernel

## Introduction

Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005, can be used to combine partial attention results (in the split-KV case). The Triton and CUDA kernels here are modified from [vllm/attention/ops/triton_merge_attn_states.py](https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_merge_attn_states.py) and [vllm/pull/16173](https://github.com/vllm-project/vllm/pull/16173). Use CUDA kernel instead of Triton to minimize CPU overhead. Compared to the Triton kernel, the CUDA kernel implemented in this PR can achieve a maximum speedup of over `3x`.End2End performance improved for R1 with PP=3 + TP=8 on L20,  4K IN:1K OUT (TTFT 5687.80 ms -> 5654.02 ms). The performance of reasoning will not degrade.

- [x] float32
- [x] float16
- [x] bfloat16
- [x] dispatch by scalar_t
- [x] fallback strategy 
- [x] unit tests (performance & correctness)
- [x] end2end test   
- [x] CEval benchmark 
- [x] test cascade_flash_attn (used merge_attn_states), passed

## Performance

| tokens | heads | headsize | dtype | Device | torch | triton | cuda | speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 256 | 16 | 128 | float16 | NVIDIA L20 | 0.15288ms | 0.04977ms | 0.01648ms | 3.0196x |
| 512 | 16 | 128 | float16 | NVIDIA L20 | 0.15355ms | 0.05237ms | 0.01659ms | 3.1563x |
| 613 | 16 | 128 | float16 | NVIDIA L20 | 0.15304ms | 0.05099ms | 0.01710ms | 2.9818x |
| 1024 | 16 | 128 | float16 | NVIDIA L20 | 0.15236ms | 0.05207ms | 0.01720ms | 3.0267x |
| 1536 | 16 | 128 | float16 | NVIDIA L20 | 0.16123ms | 0.05714ms | 0.01664ms | 3.4346x |
| 4096 | 16 | 128 | float16 | NVIDIA L20 | 0.32471ms | 0.08289ms | 0.01981ms | 4.1841x |
| 256 | 32 | 128 | float16 | NVIDIA L20 | 0.15212ms | 0.05094ms | 0.01653ms | 3.0810x |
| 512 | 32 | 128 | float16 | NVIDIA L20 | 0.15273ms | 0.05120ms | 0.01731ms | 2.9580x |
| 613 | 32 | 128 | float16 | NVIDIA L20 | 0.15344ms | 0.05269ms | 0.01879ms | 2.8040x |
| 1024 | 32 | 128 | float16 | NVIDIA L20 | 0.17060ms | 0.06185ms | 0.02596ms | 2.3829x |
| 1536 | 32 | 128 | float16 | NVIDIA L20 | 0.21955ms | 0.07167ms | 0.01720ms | 4.1659x |
| 4096 | 32 | 128 | float16 | NVIDIA L20 | 0.71306ms | 0.15442ms | 0.06354ms | 2.4304x |
| 256 | 48 | 128 | float16 | NVIDIA L20 | 0.15206ms | 0.04945ms | 0.01673ms | 2.9554x |
| 512 | 48 | 128 | float16 | NVIDIA L20 | 0.15944ms | 0.05663ms | 0.02166ms | 2.6149x |
| 613 | 48 | 128 | float16 | NVIDIA L20 | 0.16748ms | 0.05924ms | 0.02458ms | 2.4103x |
| 1024 | 48 | 128 | float16 | NVIDIA L20 | 0.21939ms | 0.07404ms | 0.03450ms | 2.1458x |
| 1536 | 48 | 128 | float16 | NVIDIA L20 | 0.38421ms | 0.08924ms | 0.03441ms | 2.5937x |
| 4096 | 48 | 128 | float16 | NVIDIA L20 | 1.02671ms | 0.30397ms | 0.23511ms | 1.2929x |
| 256 | 16 | 128 | bfloat16 | NVIDIA L20 | 0.15253ms | 0.05180ms | 0.01633ms | 3.1715x |
| 512 | 16 | 128 | bfloat16 | NVIDIA L20 | 0.15237ms | 0.05146ms | 0.01643ms | 3.1312x |
| 613 | 16 | 128 | bfloat16 | NVIDIA L20 | 0.15304ms | 0.05243ms | 0.01736ms | 3.0206x |
| 1024 | 16 | 128 | bfloat16 | NVIDIA L20 | 0.15350ms | 0.05191ms | 0.01715ms | 3.0272x |
| 1536 | 16 | 128 | bfloat16 | NVIDIA L20 | 0.16072ms | 0.05668ms | 0.01648ms | 3.4391x |
| 4096 | 16 | 128 | bfloat16 | NVIDIA L20 | 0.32445ms | 0.08197ms | 0.01986ms | 4.1272x |
| 256 | 32 | 128 | bfloat16 | NVIDIA L20 | 0.15314ms | 0.05023ms | 0.01643ms | 3.0571x |
| 512 | 32 | 128 | bfloat16 | NVIDIA L20 | 0.15253ms | 0.05146ms | 0.01720ms | 2.9913x |
| 613 | 32 | 128 | bfloat16 | NVIDIA L20 | 0.15467ms | 0.05417ms | 0.01884ms | 2.8744x |
| 1024 | 32 | 128 | bfloat16 | NVIDIA L20 | 0.17224ms | 0.06221ms | 0.02595ms | 2.3973x |
| 1536 | 32 | 128 | bfloat16 | NVIDIA L20 | 0.22102ms | 0.07240ms | 0.01751ms | 4.1349x |
| 4096 | 32 | 128 | bfloat16 | NVIDIA L20 | 0.71388ms | 0.15248ms | 0.06359ms | 2.3978x |

## Correctness 

- float16 (performance & correctness)

```bash
pytest -s test_merge_attn_states.py
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
<details>
<summary> show more details </summary>
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
</details>

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
4K IN:1K OUT (TTFT 5687.80 ms -> 5654.02 ms), The performance of reasoning will not degrade.

<details>
<summary> show more details </summary>
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

### 8K IN:64 OUT (TTFT 8861.07ms -> 8767.16ms)

- w/o this opt, 8K IN:64 OUT

```bash
Maximum request concurrency: 16
============ Serving Benchmark Result ============
Successful requests:                     48
Benchmark duration (s):                  115.37
Total input tokens:                      393216
Total generated tokens:                  3072
Request throughput (req/s):              0.42
Output token throughput (tok/s):         26.63
Total Token throughput (tok/s):          3434.90
---------------Time to First Token----------------
Mean TTFT (ms):                          8861.07
Median TTFT (ms):                        6167.50
P99 TTFT (ms):                           23576.12
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          454.74
Median TPOT (ms):                        484.97
P99 TPOT (ms):                           504.62
---------------Inter-token Latency----------------
Mean ITL (ms):                           454.74
Median ITL (ms):                         273.69
P99 ITL (ms):                            1065.00
==================================================
```

- w/ this opt, 8K IN:64 OUT (TTFT 8861.07ms -> 8767.16ms)
```bash
Maximum request concurrency: 16
============ Serving Benchmark Result ============
Successful requests:                     48
Benchmark duration (s):                  115.19
Total input tokens:                      393216
Total generated tokens:                  3072
Request throughput (req/s):              0.42
Output token throughput (tok/s):         26.67
Total Token throughput (tok/s):          3440.28
---------------Time to First Token----------------
Mean TTFT (ms):                          8767.16
Median TTFT (ms):                        6170.44
P99 TTFT (ms):                           23594.15
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          455.34
Median TPOT (ms):                        483.54
P99 TPOT (ms):                           504.48
---------------Inter-token Latency----------------
Mean ITL (ms):                           455.34
Median ITL (ms):                         270.61
P99 ITL (ms):                            1066.51
==================================================
```
</details>

## CEval benchmark

We use [evalscope](https://github.com/modelscope/evalscope) to run benchmark on `CEval` dataset. Total AverageAccuracy: 0.90197884615385

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


<details>
<summary> show more details </summary>

```bash
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
</details>

## Test cascade_flash_attn

```bash
pytest -s test_cascade_flash_attn.py
================================================================================== test session starts ===================================================================================
collected 198 items
Running 198 items in this shard

test_cascade_flash_attn.py ..............................................................................................................................ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss

============================================================================ 126 passed, 72 skipped in 1.05s =============================================================================
```
