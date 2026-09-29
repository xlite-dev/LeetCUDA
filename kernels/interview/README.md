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