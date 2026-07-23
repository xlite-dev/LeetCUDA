# CUDA Kernel 面试背题笔记

[notes-v2.cu](notes-v2.cu): 面试中高频出现的 CUDA kernel 的背题版本。

## 📖 快速开始 🔥🔥

```bash
git clone https://github.com/xlite-dev/LeetCUDA.git && cd LeetCUDA
git submodule update --init --recursive --force && cd kernels/interview
# Install the latest CUDNN library for benchmarks (remove the old version first)
apt remove -y libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-headers-cuda-13 
apt install -y cudnn9-cuda-13 # Install the latest version for best performance.

# Ada SM_89 + MMA + SMEM Swizzle + Block Swizzle + CuTe (CUDA Toolkit >= 13.2)
nvcc -std=c++20 -O2 -arch=sm_89 -lcublas -lcuda notes-v2.cu -o notes_v2_sm89.bin
nvcc -std=c++20 -O2 -arch=sm_89 -DNOTES_V2_ENABLE_CUTE -I ../../third-party/cutlass/include  \
  -lcublas -lcuda notes-v2.cu -o notes_v2_cute_sm89.bin

# Hopper SM_90a + CuTe + Swizzle + TMA WGMMA WS + CuTe HGEMM (CUDA Toolkit >= 13.2)
nvcc -std=c++20 -O2 -gencode arch=compute_90a,code=sm_90a -DNOTES_V2_ENABLE_WGMMA \
  -DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_TMA_MMA_WS -I ../../third-party/cutlass/include \
  -lcublas -lcuda notes-v2.cu -o notes_v2_sm90a.bin 

# Blackwell SM_120a + CuTe + Swizzle + TMA MMA WS + cuDNN SDPA (CUDA Toolkit >= 13.2):
nvcc -std=c++20 -O2 -gencode arch=compute_120a,code=sm_120a --expt-relaxed-constexpr \
  -DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_TMA_MMA_WS -DNOTES_V2_ENABLE_CUDNN \
  -I ../../third-party/cutlass/include -I ../../third-party/cudnn-frontend/include \
  -L/usr/local/cuda/targets/x86_64-linux/lib/stubs -lcublas -lcudnn -lnvrtc \
  -lcuda notes-v2.cu -o notes_v2_sm120a.bin 
```

```bash
# Then, run the notes_v2_sm120a.bin with bench mode (e.g., NVIDIA RTX 5090, Blackwell SM_120a)
# Baseline: cuBLAS v13.3.0.5-1 (290T); cuDNN v9.25.0.15 SDPA (223T), PyTorch v2.11 SDPA (210T)
# Speedup: Flash-Attention 2/3 -> ~1.32x (F16 Acc vs cuDNN), ~1.01x (F32 Acc vs cuDNN), ~1.07x
# (F32 Acc vs PyTorch SDPA); HGEMM w/ Pipe & SMEM & Block Swizzle -> 1.05x (F16 Acc vs cuBLAS)
./notes_v2_sm120a.bin --bench --mnk 4096,4096,4096 --bhnd 1,32,16384,128 # MMA ACC F16/F32 Acc
| Kernel                                                   | Max Err   | TFLOPS/cu{BLAS,DNN} |
|----------------------------------------------------------|-----------|---------------------|
| HGEMM CuTe Swizzle (S=2, SW=0)                           | 0.000e+00 | 309.2/302.7 (1.02x) |
| HGEMM CuTe Swizzle (S=2, SW=1)                           | 0.000e+00 | 307.6/302.7 (1.02x) |
| HGEMM CuTe Swizzle (S=3, SW=0)                           | 0.000e+00 | 315.0/302.7 (1.04x) |
| HGEMM CuTe Swizzle (S=3, SW=1)                           | 0.000e+00 | 317.4/302.7 (1.05x) |
| FA2 (S=1, Pad, F16Acc)                                   | 1.831e-04 | 220.0/223.1 (0.99x) |
| FA2 (S=2, Pad, F16Acc)                                   | 1.831e-04 | 254.5/223.1 (1.14x) |
| FA2 (S=1, Pad, F32Acc)                                   | 1.526e-05 | 166.6/222.8 (0.75x) |
| FA2 (S=2, Pad, F32Acc)                                   | 1.526e-05 | 179.1/222.8 (0.80x) |
| FA2 TMA MMA WS (1 Consumer WG) (Sk=1, Sv=1, F16Acc)      | 1.831e-04 | 264.0/223.1 (1.18x) |
| FA2 TMA MMA WS (1 Consumer WG) (Sk=2, Sv=1, F16Acc)      | 1.831e-04 | 285.7/223.1 (1.28x) |
| FA2 TMA MMA WS (1 Consumer WG) (Sk=2, Sv=2, F16Acc)      | 1.831e-04 | 289.3/223.1 (1.30x) |
| FA2 TMA MMA WS (1 Consumer WG) (Sk=2, Sv=1, F32Acc)      | 1.526e-05 | 205.6/222.8 (0.92x) |
| FA3 TMA MMA WS (2 Consumer WG) (Sk=1, Sv=1, F16Acc)      | 9.155e-05 | 293.5/223.1 (1.32x) |
| FA3 TMA MMA WS (2 Consumer WG) (Sk=1, Sv=1, F32Acc)      | 1.526e-05 | 220.2/222.8 (0.99x) |
| FA2 CuTe TMA MMA WS (1 Consumer WG) (Sk=2, Sv=1, F32Acc) | 1.526e-05 | 220.2/222.8 (0.99x) |
| FA3 CuTe TMA MMA WS (2 Consumer WG) (Sk=1, Sv=1, F32Acc) | 1.526e-05 | 224.2/222.8 (1.01x) |
```
