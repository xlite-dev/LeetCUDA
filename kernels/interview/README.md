# notes-v2.cu — CUDA Kernel 面试背题笔记

[LeetCUDA](https://github.com/xlite-dev/LeetCUDA) 中面试高频 CUDA kernel 的完整实现，共 8 个 Phase、26 个 kernel、25 个 test case。

## 快速开始

```bash
# 编译 (sm_89)
nvcc -std=c++20 -O2 -arch=sm_89 -lcublas -lcuda notes-v2.cu -o notes_v2_sm89.bin

# 运行
CUDA_VISIBLE_DEVICES=7 ./notes_v2_sm89.bin

# 自定义 GEMM 尺寸
./notes_v2_sm89.bin
```

## 测试输出

```
=== notes-v2.cu verification harness ===
| Kernel                              | Max Err      | Pass |
|-------------------------------------|--------------|------|
| BlockReduceAll                      | 1.907349e-06 | PASS |
| Dot                                 | 0.000000e+00 | PASS |
| Dot-Vec4                            | 0.000000e+00 | PASS |
| ReLU                                | 0.000000e+00 | PASS |
| ReLU-Vec4                           | 0.000000e+00 | PASS |
| ElemwiseAdd                         | 0.000000e+00 | PASS |
| ElemwiseAdd-Vec4                    | 0.000000e+00 | PASS |
| Histogram                           | 0.000000e+00 | PASS |
| OnlineSafeSoftmax                   | 3.725290e-09 | PASS |
| SafeSoftmax                         | 1.862645e-09 | PASS |
| NaiveSoftmax                        | 3.725290e-09 | PASS |
| RMSNorm                             | 4.768372e-07 | PASS |
| RMSNorm-Vec4                        | 4.768372e-07 | PASS |
| LayerNorm                           | 4.768372e-07 | PASS |
| LayerNorm-Vec4                      | 3.576279e-07 | PASS |
| RoPE                                | 1.192093e-07 | PASS |
| MatTranspose                        | 0.000000e+00 | PASS |
| MatTransposePadded                  | 0.000000e+00 | PASS |
| SGEMV-K128                          | 9.536743e-07 | PASS |
| SGEMV-K32                           | 9.536743e-07 | PASS |
| SGEMV-K16                           | 2.384186e-07 | PASS |
| SGEMM                               | 7.247925e-05 | PASS |
| SGEMM-Vec4                          | 7.247925e-05 | PASS |
| HGEMM MMA                           | 0.000000e+00 | PASS |
| FlashAttn-SplitQ                    | 1.303628e-04 | PASS |
=== All tests done ===
```
