## 📖 快速开始 🔥🔥

```bash
git clone https://github.com/xlite-dev/LeetCUDA.git && cd LeetCUDA
git submodule update --init --recursive --force && cd kernels/interview
# Install the latest cuBLAS & cuDNN for bench (remove the old version)
apt remove -y libcublas-cuda-13 libcudnn9-cuda-13 libcudnn9-dev-cuda-13
apt install -y cublas-cuda-13 cudnn9-cuda-13 && apt install -y ccache
```
```bash
# Blackwell (e.g., RTX 5090, PRO 5000/6000, recommended: CUDA>=13.2)
./build.sh --arch sm_120a && ./bin/leetcuda_bench_sm120a.bin --bench
```

```bash
HGEMM: M=8192 N=8192 K=8192   FA: B=1 H=32 N=8192 D=128
| Kernel                                                   | Max Err   | TFLOPS/cu{BLAS,DNN} |
|----------------------------------------------------------|-----------|---------------------|
| HGEMM CuTe Swizzle (S=2, BLK_SW=0, F16Acc)               | 0.000e+00 | 243.2/247.0 (0.98x) |
| HGEMM CuTe Swizzle (S=2, BLK_SW=0, F32Acc)               | 0.000e+00 | 202.6/165.1 (1.23x) |
| HGEMM CuTe Swizzle (S=2, BLK_SW=1, F16Acc)               | 0.000e+00 | 250.5/247.0 (1.01x) |
| HGEMM CuTe Swizzle (S=2, BLK_SW=1, F32Acc)               | 0.000e+00 | 226.2/165.1 (1.37x) |
| HGEMM CuTe Swizzle (S=3, BLK_SW=0, F16Acc)               | 0.000e+00 | 257.7/247.0 (1.04x) |
| HGEMM CuTe Swizzle (S=3, BLK_SW=1, F16Acc)               | 0.000e+00 | 266.4/247.0 (1.08x) |
| HGEMM CuTe Swizzle (S=3, BLK_SW=1, F32Acc)               | 0.000e+00 | 242.9/165.1 (1.47x) |
| FP8 GEMM CuTe NonWS (PerRow x PerCol, 128x256/s2)        | 6.250e+00 | 486.5/163.3 (2.98x) |
| FP8 GEMM CuTe NonWS (PerRow x PerBlk, 128x256/s2)        | 6.250e+00 | 483.4/163.3 (2.96x) |
| FP8 GEMM CuTe NonWS (PerBlk x PerCol, 128x256/s2)        | 6.250e+00 | 487.3/163.3 (2.98x) |
| FP8 GEMM CuTe NonWS (PerBlk x PerBlk, 128x256/s2)        | 6.250e+00 | 484.0/163.3 (2.96x) |
| FP8 GEMM CuTe WS (PerRow x PerCol, 128x256/s2)           | 6.250e+00 | 491.2/163.3 (3.01x) |
| FP8 GEMM+Quant E2E NonWS (PerRow x PerCol, 128x256/s2)   | 6.250e+00 | 316.1/163.3 (1.94x) |
| FP8 GEMM+Quant E2E WS (PerRow x PerCol, 128x256/s2)      | 6.250e+00 | 318.3/163.3 (1.95x) |
| FP8 GEMM+A Quant E2E NonWS (B offline, 128x256/s2)       | 6.250e+00 | 431.1/163.3 (2.64x) |
| FP8 GEMM+A Quant E2E WS (B offline, 128x256/s2)          | 6.250e+00 | 434.3/163.3 (2.66x) |
| FP8 GEMM CuTe NonWS (RC, randn(+-0.25), 128x256/s2)      | 1.328e-01 | 483.3/163.4 (2.96x) |
| FP8 GEMM CuTe WS (RC, randn(+-0.25), 128x256/s2)         | 1.328e-01 | 487.3/163.4 (2.98x) |
| FP4 GEMM CuTe NonWS (level-2 only, 128x256/s6)           | 1.440e-01 | 642.8/163.4 (3.93x) |
| FP4 GEMM CuTe NonWS (level-2 x PerRow, 128x256/s6)       | 1.446e-01 | 642.4/163.4 (3.93x) |
| FP4 GEMM CuTe NonWS (level-2 x PerCol, 128x256/s6)       | 1.446e-01 | 641.3/163.4 (3.92x) |
| FP4 GEMM CuTe NonWS (PerRow x PerCol, 128x256/s6)        | 1.452e-01 | 641.0/163.4 (3.92x) |
| FP4 GEMM CuTe WS (PerRow x PerCol, 128x256/s6)           | 1.452e-01 | 649.2/163.4 (3.97x) |
| FP4 GEMM+Quant E2E NonWS (level-2 only, 128x256/s6)      | 1.440e-01 | 532.7/163.4 (3.26x) |
| FP4 GEMM+Quant E2E NonWS (PerRow x PerCol, 128x256/s6)   | 1.452e-01 | 468.2/163.4 (2.87x) |
| FP4 GEMM+Quant E2E WS (PerRow x PerCol, 128x256/s6)      | 1.452e-01 | 470.5/163.4 (2.88x) |
| FP4 GEMM+A Quant E2E NonWS (B offline, 128x256/s6)       | 1.452e-01 | 541.7/163.4 (3.32x) |
| FP4 GEMM+A Quant E2E WS (B offline, 128x256/s6)          | 1.452e-01 | 546.8/163.4 (3.35x) |
| FP4 GEMM CuTe NonWS (RC, randn(+-0.25), 128x256/s6)      | 1.345e-01 | 644.7/163.3 (3.95x) |
| FP4 GEMM CuTe WS (RC, randn(+-0.25), 128x256/s6)         | 1.345e-01 | 648.4/163.3 (3.97x) |
| FA2 MMA Stages (Sk=1, Pad, F16Acc)                       | 2.441e-04 | 127.8/231.2 (0.55x) |
| FA2 MMA Stages (Sk=2, Pad, F16Acc)                       | 2.441e-04 | 156.2/231.2 (0.68x) |
| FA2 MMA Stages (Sk=1, Pad, F32Acc)                       | 1.526e-05 | 140.8/231.5 (0.61x) |
| FA2 MMA Stages (Sk=2, Pad, F32Acc)                       | 1.526e-05 | 163.4/231.5 (0.71x) |
| FA2 CuTe MMA Stages (Sk=1, F32Acc)                       | 1.526e-05 | 184.2/231.5 (0.80x) |
| FA2 CuTe MMA Stages (Sk=2, F32Acc)                       | 1.526e-05 | 196.4/231.5 (0.85x) |
| FA2 TMA MMA WS (1 Consumer WG) (Sk=1, Sv=1, F16Acc)      | 2.441e-04 | 157.2/231.2 (0.68x) |
| FA2 TMA MMA WS (1 Consumer WG) (Sk=2, Sv=1, F16Acc)      | 2.441e-04 | 188.8/231.2 (0.82x) |
| FA2 TMA MMA WS (1 Consumer WG) (Sk=2, Sv=2, F16Acc)      | 2.441e-04 | 189.7/231.2 (0.82x) |
| FA2 TMA MMA WS (1 Consumer WG) (Sk=2, Sv=1, F32Acc)      | 1.526e-05 | 204.5/231.5 (0.88x) |
| FA3 TMA MMA WS (2 Consumer WG) (Sk=1, Sv=1, F16Acc)      | 1.068e-04 | 206.3/231.2 (0.89x) |
| FA3 TMA MMA WS (2 Consumer WG) (Sk=1, Sv=1, F32Acc)      | 1.526e-05 | 206.0/231.5 (0.89x) |
| FA2 CuTe TMA MMA WS (1 Consumer WG) (Sk=2, Sv=1, F32Acc) | 1.526e-05 | 218.9/231.5 (0.95x) |
| FA2 CuTe TMA MMA WS (1 Consumer WG) (Sk=3, Sv=1, F32Acc) | 1.526e-05 | 221.6/231.5 (0.96x) |
| FA2 CuTe TMA MMA Persistent-CTA WS (D=128)               | 1.526e-05 | 240.2/231.5 (1.04x) |
```