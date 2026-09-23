## 📖 快速开始 🔥🔥

```bash
git clone https://github.com/xlite-dev/LeetCUDA.git && cd LeetCUDA
git submodule update --init --recursive --force && cd kernels/interview
# Install the latest CUDNN library for benchmarks (remove the old version first)
apt remove -y libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-headers-cuda-13 
apt install -y cudnn9-cuda-13 ccache # Also install ccache for faster rebuilds

# Build for target architecture (ccache accelerated when available):
./build.sh --arch sm_120a   # Blackwell (RTX 5090 / PRO 5000/6000, CUDA Toolkit >= 13.2)
./build.sh --help           # Show help for build options
```

```bash
# Then, run the notes_v2_sm120a.bin with bench mode (e.g., NVIDIA PRO 5000, Blackwell SM_120a)
./bin/notes_v2_sm120a.bin --bench --mnk 4096,4096,4096 --bhnd 1,32,16384,128 # MMA ACC F16/F32
| Kernel                                                   | Max Err   | TFLOPS/cu{BLAS,DNN} |
|----------------------------------------------------------|-----------|---------------------|
| HGEMM CuTe Swizzle (S=2, BLK_SW=0, F16Acc)               | 0.000e+00 | 229.5/236.9 (0.97x) |
| HGEMM CuTe Swizzle (S=2, BLK_SW=0, F32Acc)               | 0.000e+00 | 213.0/163.7 (1.30x) |
| HGEMM CuTe Swizzle (S=2, BLK_SW=1, F16Acc)               | 0.000e+00 | 231.1/236.9 (0.98x) |
| HGEMM CuTe Swizzle (S=2, BLK_SW=1, F32Acc)               | 0.000e+00 | 217.2/163.7 (1.33x) |
| HGEMM CuTe Swizzle (S=3, BLK_SW=0, F16Acc)               | 0.000e+00 | 245.6/236.9 (1.04x) |
| HGEMM CuTe Swizzle (S=3, BLK_SW=0, F32Acc)               | 0.000e+00 | 242.6/163.7 (1.48x) |
| HGEMM CuTe Swizzle (S=3, BLK_SW=1, F16Acc)               | 0.000e+00 | 246.7/236.9 (1.04x) |
| HGEMM CuTe Swizzle (S=3, BLK_SW=1, F32Acc)               | 0.000e+00 | 243.5/163.7 (1.49x) |
| FA2 MMA Stages (Sk=1, Pad, F16Acc)                       | 1.831e-04 | 131.5/232.4 (0.57x) |
| FA2 MMA Stages (Sk=2, Pad, F16Acc)                       | 1.831e-04 | 157.9/232.4 (0.68x) |
| FA2 MMA Stages (Sk=1, Pad, F32Acc)                       | 1.526e-05 | 145.6/232.4 (0.63x) |
| FA2 MMA Stages (Sk=2, Pad, F32Acc)                       | 1.526e-05 | 166.5/232.4 (0.72x) |
| FA2 CuTe MMA Stages (Sk=1, F32Acc)                       | 1.526e-05 | 189.6/232.4 (0.82x) |
| FA2 CuTe MMA Stages (Sk=2, F32Acc)                       | 1.526e-05 | 197.0/232.4 (0.85x) |
| FA2 TMA MMA WS (1 Consumer WG) (Sk=1, Sv=1, F16Acc)      | 1.831e-04 | 161.2/232.4 (0.69x) |
| FA2 TMA MMA WS (1 Consumer WG) (Sk=2, Sv=1, F16Acc)      | 1.831e-04 | 189.4/232.4 (0.81x) |
| FA2 TMA MMA WS (1 Consumer WG) (Sk=2, Sv=2, F16Acc)      | 1.831e-04 | 190.5/232.4 (0.82x) |
| FA2 TMA MMA WS (1 Consumer WG) (Sk=2, Sv=1, F32Acc)      | 1.526e-05 | 204.9/232.4 (0.88x) |
| FA3 TMA MMA WS (2 Consumer WG) (Sk=1, Sv=1, F16Acc)      | 9.155e-05 | 210.1/232.4 (0.90x) |
| FA3 TMA MMA WS (2 Consumer WG) (Sk=1, Sv=1, F32Acc)      | 1.526e-05 | 210.8/232.4 (0.91x) |
| FA2 CuTe TMA MMA WS (1 Consumer WG) (Sk=2, Sv=1, F32Acc) | 1.526e-05 | 220.0/232.4 (0.95x) |
| FA2 CuTe TMA MMA WS (1 Consumer WG) (Sk=3, Sv=1, F32Acc) | 1.526e-05 | 223.4/232.4 (0.96x) |
| FA2 CuTe TMA MMA Persistent-CTA WS (D=128)               | 1.526e-05 | 242.5/232.4 (1.04x) |
# Speedup: Split-D for large headdim (e.g, D=320) ~2.93x faster than cuDNN SDPA (with F32 Acc)
./bin/notes_v2_sm120a.bin --bench --bhnd 1,32,8192,320 # Split-D for large headdims (e.g, 320)
| Kernel                                                   | Max Err   | TFLOPS/cu{BLAS,DNN} |
|----------------------------------------------------------|-----------|---------------------|
| FA Split-D CuTe TMA MMA WS (D=320, Sk=1, Sv=1)           | 1.526e-05 |  86.6/69.8 (1.24x)  |
| FA Split-D CuTe TMA MMA WS (D=320, Sk=2, Sv=2)           | 1.526e-05 | 139.9/69.8 (2.01x)  |
| FA Split-D CuTe TMA non-WS (D=320, Sk=2, Sv=2)           | 3.052e-05 | 187.9/69.8 (2.69x)  |
| FA Split-D CuTe TMA non-WS (D=320, Sk=2, Sv=3)           | 3.052e-05 | 188.8/69.8 (2.71x)  |
| FA Split-D CuTe TMA non-WS (D=320, Sk=3, Sv=2)           | 3.052e-05 | 204.3/69.8 (2.93x)  |
```