# CUDA Kernel 面试背题笔记

[notes-v2.cu](notes-v2.cu): 面试中高频出现的 CUDA kernel 的背题版本。

## 快速开始

```bash
git submodule update --init --recursive --force && cd kernels/interview
nvcc -std=c++20 -O2 -arch=sm_89 -lcublas -lcuda notes-v2.cu -o notes_v2_sm89.bin # Ada
nvcc -std=c++20 -O2 -arch=sm_89 -DNOTES_V2_ENABLE_CUTE -I ../../third-party/cutlass/include  \
  -lcublas -lcuda notes-v2.cu -o notes_v2_cute_sm89.bin # Ada + CuTe
nvcc -std=c++20 -O2 -gencode arch=compute_90a,code=sm_90a -DNOTES_V2_ENABLE_WGMMA \
  -DNOTES_V2_ENABLE_CUTE -I ../../third-party/cutlass/include -lcublas -lcuda \
  notes-v2.cu -o notes_v2_sm90.bin # Hopper (H100, H200, etc)
./notes_v2_sm90.bin # NOTE: run notes_v2_sm90.bin for HGEMM TMA + WGMMA on Hopper device.
=== notes-v2.cu verification harness ===
| Kernel                              | Max Err      | Pass |
|-------------------------------------|--------------|------|
| BlockReduce                         | 1.907349e-06 | PASS |
| Dot                                 | 0.000000e+00 | PASS |
| Dot-Vec4                            | 0.000000e+00 | PASS |
| ReLU                                | 0.000000e+00 | PASS |
| ReLU-Vec4                           | 0.000000e+00 | PASS |
| ElemwiseAdd                         | 0.000000e+00 | PASS |
| ElemwiseAdd-Vec4                    | 0.000000e+00 | PASS |
| Histogram                           | 0.000000e+00 | PASS |
| MergeAttnStates                     | 1.788139e-07 | PASS |
| MergeAttnStates-inf                 | 0.000000e+00 | PASS |
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
| SGEMM                               | 0.000000e+00 | PASS |
| SGEMM-Vec4                          | 0.000000e+00 | PASS |
| HGEMM MMA                           | 0.000000e+00 | PASS |
| HGEMM Swizzle + Reg2x               | 0.000000e+00 | PASS |
| HGEMM CuTe                          | 0.000000e+00 | PASS |
| HGEMM WGMMA                         | 0.000000e+00 | PASS |
| FlashAttn-SplitQ                    | 1.646988e-04 | PASS |
=== All tests done ===
```
