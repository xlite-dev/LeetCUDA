<div align="center">
  <p align="center">
    <h2>📚 LeetCUDA: Modern CUDA Learn Notes with PyTorch for Beginners 🐑</h2>
    <img src='https://github.com/user-attachments/assets/b2578723-b7a7-4d8f-bcd1-5008947b808a' width="360" height="56" >
    <a href="https://hellogithub.com/repository/98348655a96640ca8ddcbc298edc901d" target="_blank"><img src="https://api.hellogithub.com/v1/widgets/recommend.svg?rid=98348655a96640ca8ddcbc298edc901d&claim_uid=ofSCbzTmdeQk3FD&theme=dark" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>
  </p>
  <div align='center'>
      <img src=https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg >
      <img src=https://img.shields.io/badge/Language-CUDA-brightgreen.svg >
      <img src=https://img.shields.io/github/forks/xlite-dev/LeetCUDA.svg?style=dark >
      <img src=https://img.shields.io/github/stars/xlite-dev/LeetCUDA.svg?style=dark >
      <img src=https://img.shields.io/badge/License-GPLv3.0-turquoise.svg >
  </div>
</div>

📚 **LeetCUDA**: It includes **Tensor/CUDA Cores, TF32/F16/BF16/F8**, [📖200+ CUDA Kernels🔥](#cuda-kernel) with PyTorch, [📖100+ LLM/CUDA🔥](#my-blogs-part-1) blogs, [📖HGEMM⚡️](./kernels/hgemm) which can achieve `98%~100%` TFLOPS of **cuBLAS**, and [📖flash-attn⚡️](./kernels/flash-attn) using Tensor Cores with pure MMA PTX. ♥️ Please consider to leave a ⭐️ Star to support me, my bro ~ ♥️

<div align="center">
  <p align="center">
    <a href="#contribute">🔥🔥 PR Welcome: Add Your Kernel to LeetCUDA! Let's make it Awesome together! 🎉🎉</a> <br>
    <a href=https://github.com/xlite-dev/LeetCUDA/graphs/contributors > <img src=https://opencollective.com/leetcuda/contributors.svg height=40px > </a>
  </p>
</div>

## ©️Citations🎉🎉

```BibTeX
@misc{LeetCUDA@2025,
  title={LeetCUDA: A Modern CUDA Learn Notes with PyTorch for Beginners},
  url={https://github.com/xlite-dev/LeetCUDA.git},
  note={Open-source software available at https://github.com/xlite-dev/LeetCUDA.git},
  author={DefTruth and Many Others},
  year={2025}
}
```


## 📖 News 🔥🔥
<div id="news"></div>

- [2026/02] Cache-DiT **[🎉v1.2.1](https://github.com/vipshop/cache-dit)** release is ready, the major updates including: [Ring](https://cache-dit.readthedocs.io/en/latest/user_guide/CONTEXT_PARALLEL) Attention w/ [batched P2P](https://cache-dit.readthedocs.io/en/latest/user_guide/CONTEXT_PARALLEL), [USP](https://cache-dit.readthedocs.io/en/latest/user_guide/CONTEXT_PARALLEL/) (Hybrid Ring and Ulysses), Hybrid 2D and 3D Parallelism (💥[USP + TP](https://cache-dit.readthedocs.io/en/latest/user_guide/HYBRID_PARALLEL/)),  VAE-P Comm overhead reduce.

<div align='center'>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/cache-dit-logo-v2.png height=320px>
</div>

- [2025/01]: **[🤖ffpa-attn](https://github.com/xlite-dev/ffpa-attn.git)** is released! Yet another Faster Flash Prefill Attention with O(1)🎉SRAM complexity for large headdim, **1.8x~3x↑**🎉 vs SDPA EA: [📈L20 ~1.9x↑🎉](https://github.com/xlite-dev/ffpa-attn?tab=readme-ov-file#L1-bench-l20), [📈A30 ~1.8x↑🎉](https://github.com/xlite-dev/ffpa-attn?tab=readme-ov-file#L1-bench-a30),[📈4090 ~2.1x↑🎉](https://github.com/xlite-dev/ffpa-attn?tab=readme-ov-file#L1-bench-4090).

<div align='center'>
<img height="320px" alt="image" src="https://github.com/user-attachments/assets/ed30185b-2e11-4293-832f-43e9003d6ad9" />
</div>

- [2024/12]: **[⚡️HGEMM](https://github.com/xlite-dev/HGEMM.git)** is released! Write HGEMM from scratch using Tensor Cores with **WMMA, MMA and CuTe** API, achieve peak🎉 performance.

## 📖 Contents
<div id="contents"></div>

- [📖 HGEMM-MMA 🎉🎉](#HGEMM-bench)
- [📖 FlashAttention-MMA 🎉🎉](#fa-mma-bench)
  - [📚 Split KV (Basic, FA-1)](#mma-split-kv)
  - [📚 Split Q (Faster, FA-2)](#mma-split-q)
  - [📚 Split Q + Shared KV](#mma-share-kv)
  - [📚 Split Q + Shared QKV](#mma-share-qkv)
  - [📚 Split Q + QK Tiling](#mma-tiling-qk)
  - [📚 Split Q + QKV Tiling](#mma-tiling-qkv)
- [📖 200+ CUDA Kernels 🔥🔥](#cuda-kernel)
  - [📚 Easy ⭐️](#cuda-kernel-easy-medium)
  - [📚 Medium ⭐️⭐️](#cuda-kernel-easy-medium)
  - [📚 Hard ⭐️⭐️⭐️](#cuda-kernel-hard)
  - [📚 Hard+ ⭐️⭐️⭐️⭐️](#cuda-kernel-hard-plus)
  - [📚 Hard++ ⭐⭐⭐️⭐️⭐️](#cuda-kernel-hard-plus)
  - [📚 Triton ⭐⭐⭐️](#triton-kernel)
  - [📚 CUTLASS ⭐⭐⭐️](#cutlass-kernel)
- [📖 100+ LLM/CUDA Blogs 🔥](#my-blogs-part-1)
- [📖 How to Contribute 👀👇](#contribute)


## 📖 HGEMM Benchmark 🎉🎉

<div id="HGEMM-bench"></div>

Currently, on NVIDIA L20, RTX 4090 and RTX 3080 Laptop, compared with cuBLAS's default Tensor Cores algorithm, the `HGEMM (WMMA/MMA/CuTe)` in this repo (`blue`🔵) can achieve `98%~100%` of its (`orange`🟠) performance. Please check [toy-hgemm library⚡️⚡️](./kernels/hgemm) or [HGEMM⚡️⚡️](https://github.com/xlite-dev/HGEMM) repo for more details.

![toy-hgemm-library](https://github.com/user-attachments/assets/962bda14-b494-4423-b8eb-775da9f5503d)

|📚Feature |📚Feature |📚Feature |📚Feature|
|:---:|:---:|:---:|:---:|
|✔️CUDA/**Tensor Cores**|✔️Loop over K|✔️Tile Block(BMxBK)|✔️Tile Threads(T 8x8)|
|✔️WMMA(m16n16k16)|✔️MMA(m16n8k16)|✔️Pack LDST(128 bits)|✔️SMEM Padding|
|✔️Copy Async|✔️Tile MMAs|✔️Tile Warps|✔️**Multi Stages(2~4)**|
|✔️Register Double Buffers|✔️**Block Swizzle**|✔️**Warp Swizzle**|✔️**SMEM Swizzle**(CuTe/MMA)|
|✔️Collective Store(Shfl)|✔️Layout NN|✔️Layout TN|✔️SGEMM FP32/TF32|

## 📖 FA2-MMA Benchmark 🎉🎉

<div id="fa-mma-bench"></div>

I have also implemented **FlashAttention-2** using pure MMA PTX instructions, which supports features such as Multi-Stages, Tile MMA, Tile Warp, Shared KV SMEM, **Fully Shared QKV SMEM**, **Prefetch Q s2r**, **Prefetch K/V g2s**, **QKV Fine-grained Tiling**, Collective Store, etc. Please refer to [flash-attn⚡️⚡️](./kernels/flash-attn) for more details.

![flash-attn-mma](https://github.com/user-attachments/assets/6f66796d-44d5-4ec1-b224-af997bd152b2)

|📚Feature |📚Feature |📚Feature |📚Feature|
|:---:|:---:|:---:|:---:|
|✔️Tensor Cores|✔️Loop over N/D |✔️Tile Block(Br, Bc)|✔️MMA(m16n8k16)|
|✔️Pack LDST(128 bits)|✔️SMEM **Swizzle**/Padding |✔️Copy Async|✔️Tile MMAs|
|✔️Tile Warps|✔️Multi Stages(1/2)|✔️Collective Store(Shfl)|✔️**Split KV/Q**|
|✔️**Shared QKV** SMEM|✔️**Prefetch Q** s2r|✔️**Prefetch KV** g2s|✔️**QKV Fine-grained Tiling**|

Currently, for small-scale attention `(B<=4, H <=48, SeqLen <= 8192, D <= 64)` it can run faster than FA2/SDPA on some Devices. For example, on NVIDIA RTX 3080 Laptop, [📚 Split Q + Fully Shared QKV SMEM](#mma-share-qkv) method can achieve **55 TFLOPS (D=64)** that almost **~1.5x** 🎉 faster than FA2. On NVIDIA L20, 🤖[ffpa-attn](https://github.com/xlite-dev/ffpa-attn) method can achieve **104 TFLOPS (D=512)** that almost **~1.8x** 🎉 faster than SDPA (EFFICIENT ATTENTION). However, for large-scale attention, there remains a performance gap. Stay tuned for updates ~ (MMA Acc F16/F32, softmax Acc F32 vs FA2 MMA/softmax Acc F32, 👇Benchmark)

|Algorithm| (B,H,N,D) | RTX 3080 Laptop | L20 | RTX 4090 |
|:---:|:---:|:---:|:---:|:---:|
|FlashAttention-2|(1,8,8192,64)|37 TFLOPS|100 TFLOPS|145 TFLOPS|
|share-qkv+stage2|(1,8,8192,64)|**55 TFLOPS**|99 TFLOPS|**221 TFLOPS**|
|FlashAttention-2|(1,48,8192,64)|37 TFLOPS|109 TFLOPS|163 TFLOPS|
|share-qkv+stage2|(1,48,8192,64)|**48 TFLOPS**|107 TFLOPS|**224 TFLOPS**|
|SDPA(EFFICIENT ATTENTION)|(1,48,8192,512)|16 TFLOPS|58 TFLOPS|85 TFLOPS|
|🤖[ffpa-attn](https://github.com/xlite-dev/ffpa-attn)|(1,48,8192,512)|**39 TFLOPS**|**104 TFLOPS**|**200 TFLOPS**|
|Precision Errors vs FA2/SDPA| / | max: < ~1e-3 | min: ~0.0 | mean: < ~1e-5 |

The `Split KV` and `Split Q` implementations have been carried out in [flash-attn⚡️⚡️](./kernels/flash-attn) for performance comparison. The `Split KV` method, which involves splitting all QKV across MMA (Warps), is slower than `Split Q` method, which splitting Q across MMA(Warps) and keep access KV for all MMA(Warps).

- 📚 Split KV (Basic, FlashAttention-1)
<div id="mma-split-kv"></div>

```C++
// Split QKV across MMA(Warps) using naive matmul MMA&Warp tiling policy.
// case: The layout of 8 MMA(2x4)  [after] kWarpTileSeqLenQxkWarpTileSeqLenK(2x2) -> 32x2,32x2=64x64:
// |  [64,64]  |    warp_KV 0    |    warp_KV 1    |    warp_KV 2    |    warp_KV 3    |
// | warp_QP 0 |-- MMA 0,MMA 0 --|-- MMA 2,MMA 2 --|-- MMA 4,MMA 4 --|-- MMA 6,MMA 6 --|
// | warp_QP 0 |-- MMA 0,MMA 0 --|-- MMA 2,MMA 2 --|-- MMA 4,MMA 4 --|-- MMA 6,MMA 6 --|
// | warp_QP 1 |-- MMA 1,MMA 1 --|-- MMA 3,MMA 2 --|-- MMA 5,MMA 5 --|-- MMA 7,MMA 7 --|
// | warp_QP 1 |-- MMA 1,MMA 1 --|-- MMA 3,MMA 2 --|-- MMA 5,MMA 5 --|-- MMA 7,MMA 7 --|
__global__ void // Q, K, V, O -> [B, H, N, D]
flash_attn_mma_stages_split_kv_kernel(half* Q, half* K, half* V, half* O, ...);
```

- 📚 Split Q (Faster, FlashAttention-2)
<div id="mma-split-q"></div>

```C++
// Split Q across MMA(Warps) and keep access KV for all MMA(Warps),
// in order to reduce the comm between warps via smem and warp shuffle.
// case: MMA = m16n8k16, Br=16x4=64, Bc=8x8=64, layout: 4 warps
// |   64x64   |      warp_KV 0       |
// | warp_QP 0 | MMA 0 ... MMA 0 (x8) |
// | warp_QP 1 | MMA 1 ... MMA 1 (x8) |
// | warp_QP 2 | MMA 2 ... MMA 2 (x8) |
// | warp_QP 3 | MMA 3 ... MMA 3 (x8) |
__global__ void // Q, K, V, O -> [B, H, N, D]
flash_attn_mma_stages_split_q_kernel(half* Q, half* K, half* V, half* O, ...);
```

- 📚 Split Q + Shared KV SMEM (**1/2 SRAM** vs FA2)
<div id="mma-share-kv"></div>

```C++
// K, V shared the same shared memory, improve block occupancy.
__global__ void // Q, K, V, O -> [B, H, N, D]
flash_attn_mma_stages_split_q_shared_kv_kernel(half* Q, half* K, half* V, half* O, ...);
```

- 📚 Split Q + Fully Shared QKV SMEM (**1/4 SRAM** vs FA2)

<div id="mma-share-qkv"></div>

```C++
// Q, K, V fully shared the same shared memory and prefetch Q s2r, improve block occupancy
// and reduce Q SMEM IO-Access.
__global__ void // Q, K, V, O -> [B, H, N, D]
flash_attn_mma_stages_split_q_shared_qkv_kernel(half* Q, half* K, half* V, half* O, ...);
```

- 📚 Split Q + QK Fine-grained Tiling (**O(16xd) SRAM** vs FA2 **O(4xBrxd) SRAM**, `Headdim -> 1024`)

<div id="mma-tiling-qk"></div>

```C++
// Fine-grained tiling at the MMA level for Q@K^T results in a constant SRAM usage of
// 64 * kMmaAtomK for Q and K. For V, the SRAM complexity is O(kMmaAtomK * d), leading to
// an overall SRAM complexity of O(kMmaAtomK * d). Consequently, this approach allows us to
// extend D (head dimension) up to 1024.
__global__ void // Q, K, V, O -> [B, H, N, D]
flash_attn_mma_stages_split_q_tiling_qk_kernel(half* Q, half* K, half* V, half* O, ...);
```

- 📚 Split Q + Fully QKV Fine-grained Tiling (**O(2xBrx16)~O(1) SRAM** vs FA2 **O(4xBrxd) SRAM**)

<div id="mma-tiling-qkv"></div>

```C++
// Fine-grained tiling at the MMA level for all Q@K^T and P@V results in a constant SRAM usage of
// Br * 16 or Bc * 16 for Q, K, V, leading to an overall SRAM complexity of O(Br * 16). Consequently,
// this approach allows us to run faster than SDPA w or w/o MMA Acc F32.
__global__ void // Q, K, V, O -> [B, H, N, D]
flash_attn_mma_stages_split_q_tiling_qkv_kernel(half* Q, half* K, half* V, half* O, ...);
```

💡NOTE: [📚Split Q + Fully QKV Fine-grained Tiling](#mma-tiling-qkv) has been refactored into 🤖[ffpa-attn](https://github.com/xlite-dev/ffpa-attn).

## 📖 200+ CUDA Kernels 🔥🔥 (Easy -> Hard++) ([©️back👆🏻](#contents))

<div id="cuda-kernel"></div>

The kernels listed here will guide you through a step-by-step progression, ranging from easy to very challenging topics. The **workflow** for each topic will be as follows: custom **CUDA kernel** implementation -> PyTorch **Python bindings** -> Run tests. 👉TIPS: `*` = Tensor Cores (WMMA, MMA, CuTe), otherwise, CUDA Cores; `/` = not supported; `✔️` = supported; `❔` = TODO. Contents are listed as follows:

- [📚 Easy ⭐️](#cuda-kernel-easy-medium)
- [📚 Medium ⭐️⭐️](#cuda-kernel-easy-medium)
- [📚 Hard ⭐️⭐️⭐️](#cuda-kernel-hard)
- [📚 Hard+ ⭐️⭐️⭐️⭐️](#cuda-kernel-hard-plus)
- [📚 Hard++ ⭐⭐⭐️⭐️⭐️](#cuda-kernel-hard-plus)
- [📚 Triton ⭐⭐⭐️](#triton-kernel)
- [📚 CUTLASS ⭐⭐⭐️](#cutlass-kernel)

[📚 Easy](#cuda-kernel-easy-medium) and [📚 Medium](#cuda-kernel-easy-medium) sections cover operations such as `element-wise, mat_trans, warp/block reduce, nms, relu, gelu, swish, layer-norm, rms-norm, online-softmax, dot-prod, embedding` and basic usage for `FP32`, `FP16`, `BF16` and `FP8` . [📚 Hard](#cuda-kernel-hard), [📚 Hard+](#cuda-kernel-hard-plus) and [📚 Hard++](#cuda-kernel-hard-plus) sections delve deeper into advanced topics, primarily focusing on operations like `sgemv, sgemm, hgemv, hgemm and flash-attention`. These sections also provide numerous kernels implemented using Tensor Cores with pure MMA PTX.

### 📚 Easy ⭐️ & Medium ⭐️⭐️  ([©️back👆🏻](#cuda-kernel))
<div id="cuda-kernel-easy-medium"></div>

|📖 CUDA Kernel| 📖 Elem DType| 📖 Acc DType| 📖 Docs | 📖 Level |
|:---|:---|:---|:---|:---|
| ✔️ [elementwise_f32](./kernels/elementwise/elementwise.cu)|f32|/|[link](./kernels/elementwise/)|⭐️|
| ✔️ [elementwise_f32x4](./kernels/elementwise/elementwise.cu)|f32|/|[link](./kernels/elementwise/)|⭐️|
| ✔️ [elementwise_f16](./kernels/elementwise/elementwise.cu)|f16|/|[link](./kernels/elementwise/)|⭐️|
| ✔️ [elementwise_f16x2](./kernels/elementwise/elementwise.cu)|f16|/|[link](./kernels/elementwise/)|⭐️|
| ✔️ [elementwise_f16x8](./kernels/elementwise/elementwise.cu)|f16|/|[link](./kernels/elementwise/)|⭐️|
| ✔️ [elementwise_f16x8_pack](./kernels/elementwise/elementwise.cu)|f16|/|[link](./kernels/elementwise/)|⭐️⭐️|
| ✔️ [histogram_i32](./kernels/histogram/histogram.cu)|i32|/|[link](./kernels/histogram/)|⭐️|
| ✔️ [histogram_i32x4](./kernels/histogram/histogram.cu)|i32|/|[link](./kernels/histogram/)|⭐️|
| ✔️ [sigmoid_f32](./kernels/sigmoid/sigmoid.cu)|f32|/|[link](./kernels/sigmoid/)|⭐️|
| ✔️ [sigmoid_f32x4](./kernels/sigmoid/sigmoid.cu)|f32|/|[link](./kernels/sigmoid/)|⭐️|
| ✔️ [sigmoid_f16](./kernels/sigmoid/sigmoid.cu)|16|/|[link](./kernels/sigmoid/)|⭐️|
| ✔️ [sigmoid_f16x2](./kernels/sigmoid/sigmoid.cu)|f16|/|[link](./kernels/sigmoid/)|⭐️|
| ✔️ [sigmoid_f16x8](./kernels/sigmoid/sigmoid.cu)|f16|/|[link](./kernels/sigmoid/)|⭐️|
| ✔️ [sigmoid_f16x8_pack](./kernels/sigmoid/sigmoid.cu)|f16|/|[link](./kernels/sigmoid/)|⭐️⭐️|
| ✔️ [relu_f32](./kernels/relu/relu.cu)|f32|/|[link](./kernels/relu/)|⭐️|
| ✔️ [relu_f32x4](./kernels/relu/relu.cu)|f32|/|[link](./kernels/relu/)|⭐️|
| ✔️ [relu_f16](./kernels/relu/relu.cu)|f16|/|[link](./kernels/relu/)|⭐️|
| ✔️ [relu_f16x2](./kernels/relu/relu.cu)|f16|/|[link](./kernels/relu/)|⭐️|
| ✔️ [relu_f16x8](./kernels/relu/relu.cu)|f16|/|[link](./kernels/relu/)|⭐️|
| ✔️ [relu_f16x8_pack](./kernels/relu/relu.cu)|f16|/|[link](./kernels/relu/)|⭐️⭐️|
| ✔️ [elu_f32](./kernels/elu/elu.cu)|f32|/|[link](./kernels/elu/)|⭐️|
| ✔️ [elu_f32x4](./kernels/elu/elu.cu)|f32|/|[link](./kernels/elu/)|⭐️|
| ✔️ [elu_f16](./kernels/elu/elu.cu)|f16|/|[link](./kernels/elu/)|⭐️|
| ✔️ [elu_f16x2](./kernels/elu/elu.cu)|f16|/|[link](./kernels/elu/)|⭐️|
| ✔️ [elu_f16x8](./kernels/elu/elu.cu)|f16|/|[link](./kernels/elu/)|⭐️|
| ✔️ [elu_f16x8_pack](./kernels/elu/elu.cu)|f16|/|[link](./kernels/elu/)|⭐️⭐️|
| ✔️ [gelu_f32](./kernels/gelu/gelu.cu)|f32|/|[link](./kernels/gelu/)|⭐️|
| ✔️ [gelu_f32x4](./kernels/gelu/gelu.cu)|f32|/|[link](./kernels/gelu/)|⭐️|
| ✔️ [gelu_f16](./kernels/gelu/gelu.cu)|f16|/|[link](./kernels/gelu/)|⭐️|
| ✔️ [gelu_f16x2](./kernels/gelu/gelu.cu)|f16|/|[link](./kernels/gelu/)|⭐️|
| ✔️ [gelu_f16x8](./kernels/gelu/gelu.cu)|f16|/|[link](./kernels/gelu/)|⭐️|
| ✔️ [gelu_f16x8_pack](./kernels/gelu/gelu.cu)|f16|/|[link](./kernels/gelu/)|⭐️⭐️|
| ✔️ [swish_f32](./kernels/swish/swish.cu)|f32|/|[link](./kernels/swish/)|⭐️|
| ✔️ [swish_f32x4](./kernels/swish/swish.cu)|f32|/|[link](./kernels/swish/)|⭐️|
| ✔️ [swish_f16](./kernels/swish/swish.cu)|f16|/|[link](./kernels/swish/)|⭐️|
| ✔️ [swish_f16x2](./kernels/swish/swish.cu)|f16|/|[link](./kernels/swish/)|⭐️|
| ✔️ [swish_f16x8](./kernels/swish/swish.cu)|f16|/|[link](./kernels/swish/)|⭐️|
| ✔️ [swish_f16x8_pack](./kernels/swish/swish.cu)|f16|/|[link](./kernels/swish/)|⭐️⭐️|
| ✔️ [hardswish_f32](./kernels/hardswish/hardswish.cu)|f32|/|[link](./kernels/hardswish/)|⭐️|
| ✔️ [hardswish_f32x4](./kernels/hardswish/hardswish.cu)|f32|/|[link](./kernels/hardswish/)|⭐️|
| ✔️ [hardswish_f16](./kernels/hardswish/hardswish.cu)|f16|/|[link](./kernels/hardswish/)|⭐️|
| ✔️ [hardswish_f16x2](./kernels/hardswish/hardswish.cu)|f16|/|[link](./kernels/hardswish/)|⭐️|
| ✔️ [hardswish_f16x8](./kernels/hardswish/hardswish.cu)|f16|/|[link](./kernels/hardswish/)|⭐️|
| ✔️ [hardswish_f16x8_pack](./kernels/hardswish/hardswish.cu)|f16|/|[link](./kernels/hardswish/)|⭐️⭐️|
| ✔️ [hardshrink_f32](./kernels/hardshrink/hardshrink.cu)|f32|/|[link](./kernels/hardshrink/)|⭐️|
| ✔️ [hardshrink_f32x4](./kernels/hardshrink/hardshrink.cu)|f32|/|[link](./kernels/hardshrink/)|⭐️|
| ✔️ [hardshrink_f16](./kernels/hardshrink/hardshrink.cu)|f16|/|[link](./kernels/hardshrink/)|⭐️|
| ✔️ [hardshrink_f16x2](./kernels/hardshrink/hardshrink.cu)|f16|/|[link](./kernels/hardshrink/)|⭐️|
| ✔️ [hardshrink_f16x8](./kernels/hardshrink/hardshrink.cu)|f16|/|[link](./kernels/hardshrink/)|⭐️|
| ✔️ [hardshrink_f16x8_pack](./kernels/hardshrink/hardshrink.cu)|f16|/|[link](./kernels/hardshrink/)|⭐️⭐️|
| ✔️ [embedding_f32](./kernels/embedding/embedding.cu)|f32|/|[link](./kernels/embedding/)|⭐️|
| ✔️ [embedding_f32x4](./kernels/embedding/embedding.cu)|f32|/|[link](./kernels/embedding/)|⭐️|
| ✔️ [embedding_f32x4_pack](./kernels/embedding/embedding.cu)|f32|/|[link](./kernels/embedding/)|⭐️|
| ✔️ [embedding_f16](./kernels/embedding/embedding.cu)|f16|/|[link](./kernels/embedding/)|⭐️|
| ✔️ [embedding_f16x2](./kernels/embedding/embedding.cu)|f16|/|[link](./kernels/embedding/)|⭐️|
| ✔️ [embedding_f16x8](./kernels/embedding/embedding.cu)|f16|/|[link](./kernels/embedding/)|⭐️|
| ✔️ [embedding_f16x8_pack](./kernels/embedding/embedding.cu)|f16|/|[link](./kernels/embedding/)|⭐️⭐️|
| ✔️ [mat_trans_f32_col2row{2d}](./kernels/mat-transpose/mat_transpose.cu)|f32|/|[link](./kernels/mat-transpose/)|⭐️|
| ✔️ [mat_trans_f32_row2col{2d}](./kernels/mat-transpose/mat_transpose.cu)|f32|/|[link](./kernels/mat-transpose/)|⭐️|
| ✔️ [mat_trans_f32_diagonal2d](./kernels/mat-transpose/mat_transpose.cu)|f32|/|[link](./kernels/mat-transpose/)|⭐️⭐️|
| ✔️ [mat_trans_f32x4_col2row{2d}](./kernels/mat-transpose/mat_transpose.cu)|f32|/|[link](./kernels/mat-transpose/)|⭐️⭐️|
| ✔️ [mat_trans_f32x4_row2col{2d}](./kernels/mat-transpose/mat_transpose.cu)|f32|/|[link](./kernels/mat-transpose/)|⭐️⭐️|
| ✔️ [mat_trans_cute](./kernels/mat-transpose/mat_transpose_cute.cu)|f32|/|[link](./kernels/mat-transpose/)|⭐️⭐️|
| ✔️ [warp_reduce_{all}](./kernels/reduce/block_all_reduce.cu)|all|all|[link](./kernels/reduce/)|⭐️⭐️|
| ✔️ [block_all_reduce_f32_f32](./kernels/reduce/block_all_reduce.cu)|f32|f32|[link](./kernels/reduce/)|⭐️⭐️|
| ✔️ [block_all_reduce_f32x4_f32](./kernels/reduce/block_all_reduce.cu)|f32|f32|[link](./kernels/reduce/)|⭐️⭐️|
| ✔️ [block_all_reduce_f16_f16](./kernels/reduce/block_all_reduce.cu)|f16|f16|[link](./kernels/reduce/)|⭐️⭐️|
| ✔️ [block_all_reduce_f16_f32](./kernels/reduce/block_all_reduce.cu)|f16|f32|[link](./kernels/reduce/)|⭐️⭐️|
| ✔️ [block_all_reduce_f16x2_f16](./kernels/reduce/block_all_reduce.cu)|f16|f16|[link](./kernels/reduce/)|⭐️⭐️|
| ✔️ [block_all_reduce_f16x2_f32](./kernels/reduce/block_all_reduce.cu)|f16|f32|[link](./kernels/reduce/)|⭐️⭐️|
| ✔️ [block_all_reduce_f16x8_pack_f16](./kernels/reduce/block_all_reduce.cu)|f16|f16|[link](./kernels/reduce/)|⭐️⭐️|
| ✔️ [block_all_reduce_f16x8_pack_f32](./kernels/reduce/block_all_reduce.cu)|f16|f32|[link](./kernels/reduce/)|⭐️⭐️|
| ✔️ [block_all_reduce_bf16_bf16](./kernels/reduce/block_all_reduce.cu)|bf16|bf16|[link](./kernels/reduce/)|⭐️⭐️|
| ✔️ [block_all_reduce_bf16_f32](./kernels/reduce/block_all_reduce.cu)|bf16|f32|[link](./kernels/reduce/)|⭐️⭐️|
| ✔️ [block_all_reduce_bf16x2_bf16](./kernels/reduce/block_all_reduce.cu)|bf16|bf16|[link](./kernels/reduce/)|⭐️⭐️|
| ✔️ [block_all_reduce_bf16x2_f32](./kernels/reduce/block_all_reduce.cu)|bf16|f32|[link](./kernels/reduce/)|⭐️⭐️|
| ✔️ [block_all_reduce_bf16x8_pack_bf16](./kernels/reduce/block_all_reduce.cu)|bf16|bf16|[link](./kernels/reduce/)|⭐️⭐️|
| ✔️ [block_all_reduce_bf16x8_pack_f32](./kernels/reduce/block_all_reduce.cu)|bf16|f32|[link](./kernels/reduce/)|⭐️⭐️|
| ✔️ [block_all_reduce_fp8_e4m3_f16](./kernels/reduce/block_all_reduce.cu)|fp8_e4m3|f16|[link](./kernels/reduce/)|⭐️⭐️⭐️|
| ✔️ [block_all_reduce_fp8_e5m2_f16](./kernels/reduce/block_all_reduce.cu)|fp8_e5m2|f16|[link](./kernels/reduce/)|⭐️⭐️⭐️|
| ✔️ [block_all_reduce_fp8_e4m3x16_pack_f16](./kernels/reduce/block_all_reduce.cu)|fp8_e4m3|f16|[link](./kernels/reduce/)|⭐️⭐️⭐️|
| ✔️ [block_all_reduce_fp8_e5m2x16_pack_f16](./kernels/reduce/block_all_reduce.cu)|fp8_e5m2|f16|[link](./kernels/reduce/)|⭐️⭐️⭐️|
| ✔️ [block_all_reduce_i8_i32](./kernels/reduce/block_all_reduce.cu)|i8|i32|[link](./kernels/reduce/)|⭐️⭐️|
| ✔️ [block_all_reduce_i8x16_pack_i32](./kernels/reduce/block_all_reduce.cu)|i8|i32|[link](./kernels/reduce/)|⭐️⭐️|
| ✔️ [dot_product_f32](./kernels/dot-product/dot_product.cu)|f32|f32|[link](./kernels/dot-product/)|⭐️⭐️|
| ✔️ [dot_product_f32x4](./kernels/dot-product/dot_product.cu)|f32|f32|[link](./kernels/dot-product/)|⭐️⭐️|
| ✔️ [dot_product_f16_f32](./kernels/dot-product/dot_product.cu)|f16|f32|[link](./kernels/dot-product/)|⭐️⭐️|
| ✔️ [dot_product_f16x2_f32](./kernels/dot-product/dot_product.cu)|f16|f32|[link](./kernels/dot-product/)|⭐️⭐️|
| ✔️ [dot_product_f16x8_pack_f32](./kernels/dot-product/dot_product.cu)|f16|f32|[link](./kernels/dot-product/)|⭐️⭐️|
| ✔️ [softmax_f32_per_tok](./kernels/softmax/softmax.cu)|f32|f32|[link](./kernels/softmax/)|⭐️⭐️|
| ✔️ [softmax_f32x4_per_tok](./kernels/softmax/softmax.cu)|f32|f32|[link](./kernels/softmax/)|⭐️⭐️|
| ✔️ [safe_softmax_f32_per_tok](./kernels/softmax/softmax.cu)|f32|f32|[link](./kernels/softmax/)|⭐️⭐️|
| ✔️ [safe_softmax_f32x4_per_tok](./kernels/softmax/softmax.cu)|f32|f32|[link](./kernels/softmax/)|⭐️⭐️|
| ✔️ [safe_softmax_f16_f32_per_tok](./kernels/softmax/softmax.cu)|f16|f32|[link](./kernels/softmax/)|⭐️⭐️|
| ✔️ [safe_softmax_f16x2_f32_per_tok](./kernels/softmax/softmax.cu)|f16|f32|[link](./kernels/softmax/)|⭐️⭐️|
| ✔️ [safe_softmax_f16x8_pack_f32_per_tok](./kernels/softmax/softmax.cu)|f16|f32|[link](./kernels/softmax/)|⭐️⭐️|
| ✔️ [online_safe_softmax_f32_per_token](./kernels/softmax/softmax.cu)|f32|f32|[link](./kernels/softmax/)|⭐️⭐️|
| ✔️ [online_safe_softmax_f32x4_pack_per_tok](./kernels/softmax/softmax.cu)|f32|f32|[link](./kernels/softmax/)|⭐️⭐️|
| ✔️ [rope_f32](./kernels/rope/rope.cu)|f32|f32|[link](./kernels/rope/)|⭐️⭐️|
| ✔️ [rope_f32x4_pack](./kernels/rope/rope.cu)|f32|f32|[link](./kernels/rope/)|⭐️⭐️|
| ✔️ [layer_norm_f32](./kernels/layer-norm/layer_norm.cu)|f32|f32|[link](./kernels/layer-norm/)|⭐️⭐️|
| ✔️ [layer_norm_f32x4](./kernels/layer-norm/layer_norm.cu)|f32|f32|[link](./kernels/layer-norm/)|⭐️⭐️|
| ✔️ [layer_norm_f16_f16](./kernels/layer-norm/layer_norm.cu)|f16|f16|[link](./kernels/layer-norm/)|⭐️⭐️|
| ✔️ [layer_norm_f16x2_f16](./kernels/layer-norm/layer_norm.cu)|f16|f16|[link](./kernels/layer-norm/)|⭐️⭐️|
| ✔️ [layer_norm_f16x8_f16](./kernels/layer-norm/layer_norm.cu)|f16|f16|[link](./kernels/layer-norm/)|⭐️⭐️|
| ✔️ [layer_norm_f16x8_pack_f16](./kernels/layer-norm/layer_norm.cu)|f16|f16|[link](./kernels/layer-norm/)|⭐️⭐️|
| ✔️ [layer_norm_f16x8_pack_f32](./kernels/layer-norm/layer_norm.cu)|f16|f32|[link](./kernels/layer-norm/)|⭐️⭐️|
| ✔️ [layer_norm_f16_f32](./kernels/layer-norm/layer_norm.cu)|f16|f32|[link](./kernels/layer-norm/)|⭐️⭐️|
| ✔️ [rms_norm_f32](./kernels/rms-norm/rms_norm.cu)|f32|f32|[link](./kernels/rms-norm/)|⭐️⭐️|
| ✔️ [rms_norm_f32x4](./kernels/rms-norm/rms_norm.cu)|f32|f32|[link](./kernels/rms-norm/)|⭐️⭐️|
| ✔️ [rms_norm_f16_f16](./kernels/rms-norm/rms_norm.cu)|f16|f16|[link](./kernels/rms-norm/)|⭐️⭐️|
| ✔️ [rms_norm_f16x2_f16](./kernels/rms-norm/rms_norm.cu)|f16|f16|[link](./kernels/rms-norm/)|⭐️⭐️|
| ✔️ [rms_norm_f16x8_f16](./kernels/rms-norm/rms_norm.cu)|f16|f16|[link](./kernels/rms-norm/)|⭐️⭐️|
| ✔️ [rms_norm_f16x8_f32](./kernels/rms-norm/rms_norm.cu)|f16|f32|[link](./kernels/rms-norm/)|⭐️⭐️|
| ✔️ [rms_norm_f16x8_pack_f16](./kernels/rms-norm/rms_norm.cu)|f16|f16|[link](./kernels/rms-norm/)|⭐️⭐️|
| ✔️ [rms_norm_f16x8_pack_f32](./kernels/rms-norm/rms_norm.cu)|f16|f32|[link](./kernels/rms-norm/)|⭐️⭐️|
| ✔️ [rms_norm_f16_f32](./kernels/rms-norm/rms_norm.cu)|f16|f32|[link](./kernels/rms-norm/)|⭐️⭐️|
| ✔️ [nms_f32](./kernels/nms/nms.cu)|f32|/|[link](./kernels/nms)|⭐️⭐️|
| ✔️ [merge_attn_states](./kernels/openai-triton/merge-attn-states/cuda_merge_attn_states.cu)|f16/bf16/f32|f32|[link](./kernels/openai-triton/merge-attn-states)|⭐️⭐️|
| ✔️ [notes v1(deprecated)](./kernels/notes-v1.cu)|f32|f32|/|⭐️⭐️|
| ✔️ [How to use nsys/ncu(timeline/ptx/sass)](./kernels/nvidia-nsight/)|/|/|[link](./kernels/nvidia-nsight/)|⭐️⭐️|

### 📚 Hard ⭐⭐⭐️ ([©️back👆🏻](#cuda-kernel))

<div id="cuda-kernel-hard"></div>

|📖 CUDA Kernel| 📖 Elem DType| 📖 Acc DType| 📖 Docs | 📖 Level |
|:---|:---|:---|:---|:---|
| ✔️ [sgemv_k32_f32](./kernels/sgemv/sgemv.cu)|f32|f32|[link](./kernels/sgemv/)|⭐️⭐️⭐️|
| ✔️ [sgemv_k128_f32x4](./kernels/sgemv/sgemv.cu)|f32|f32|[link](./kernels/sgemv/)|⭐️⭐️⭐️|
| ✔️ [sgemv_k16_f32](./kernels/sgemv/sgemv.cu)|f32|f32|[link](./kernels/sgemv/)|⭐️⭐️⭐️|
| ✔️ [hgemv_k32_f16](./kernels/hgemv/hgemv.cu)|f16|f16|[link](./kernels/hgemv/)|⭐️⭐️⭐️|
| ✔️ [hgemv_k128_f16x4](./kernels/hgemv/hgemv.cu)|f16|f16|[link](./kernels/hgemv/)|⭐️⭐️⭐️|
| ✔️ [hgemv_k16_f16](./kernels/hgemv/hgemv.cu)|f16|f16|[link](./kernels/hgemv/)|⭐️⭐️⭐️|
| ✔️ [sgemm_naive_f32](./kernels/sgemm/sgemm.cu)|f32|f32|[link](./kernels/sgemm/)|⭐️⭐️|
| ✔️ [sgemm_sliced_k_f32](./kernels/sgemm/sgemm.cu)|f32|f32|[link](./kernels/sgemm/)|⭐️⭐️⭐️|
| ✔️ [sgemm_t_8x8_sliced_k_f32x4](./kernels/sgemm/sgemm.cu)|f32|f32|[link](./kernels/sgemm/)|⭐️⭐️⭐️|
| ✔️ [sgemm_t_8x8_sliced_k...bcf](./kernels/sgemm/sgemm.cu)|f32|f32|[link](./kernels/sgemm/)|⭐️⭐️⭐️|
| ✔️ [sgemm_t_8x8_sliced_k...dbuf](./kernels/sgemm/sgemm.cu)|f32|f32|[link](./kernels/sgemm/)|⭐️⭐️⭐️|
| ✔️ [sgemm_t_8x8_sliced_k16...dbuf](./kernels/sgemm/sgemm_async.cu)|f32|f32|[link](./kernels/sgemm/)|⭐️⭐️⭐️|
| ✔️ [sgemm_t_8x8_sliced_k16...async](./kernels/sgemm/sgemm_async.cu)|f32|f32|[link](./kernels/sgemm/)|⭐️⭐️⭐️|
| ✔️ [sgemm_wmma_m16n16k8...stages*](./kernels/sgemm/sgemm_wmma_tf32_stage.cu)|tf32|f32|[link](./kernels/sgemm/)|⭐️⭐️⭐️|
| ✔️ [sgemm_wmma_m16n16k8...swizzle*](./kernels/sgemm/sgemm_wmma_tf32_stage.cu)|tf32|f32|[link](./kernels/sgemm/)|⭐️⭐️⭐️|
| ✔️ [hgemm_naive_f16](./kernels/hgemm/naive/hgemm.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️|
| ✔️ [hgemm_sliced_k_f16](./kernels/hgemm/naive/hgemm.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️⭐️|
| ✔️ [hgemm_t_8x8_sliced_k_f16x4](./kernels/hgemm/hgemm.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️⭐️|
| ✔️ [hgemm_t_8x8_sliced_k_f16x4_pack](./kernels/hgemm/naive/hgemm.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️⭐️|
| ✔️ [hgemm_t_8x8_sliced_k_f16x8_pack](./kernels/hgemm/naive/hgemm.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️⭐️|
| ✔️ [hgemm_t_8x8_sliced_k...dbuf](./kernels/hgemm/naive/hgemm.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️⭐️|
| ✔️ [hgemm_t_8/16x8...k16/32...dbuf](./kernels/hgemm/naive/hgemm_async.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️⭐️|
| ✔️ [hgemm_t_8/16x8...k16/32...async](./kernels/hgemm/naive/hgemm_async.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️⭐️|
| ✔️ [hgemm_wmma_m16n16k16...naive*](./kernels/hgemm/wmma/hgemm_wmma.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️⭐️|
| ✔️ [hgemm_wmma_m16n16k16...mma4x2*](./kernels/hgemm/wmma/hgemm_wmma.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️⭐️|
| ✔️ [hgemm_wmma_m16n16k16...mma4x4*](./kernels/hgemm/wmma/hgemm_wmma.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️⭐️|
| ✔️ [hgemm_wmma_m16n16k16...dbuf*](./kernels/hgemm/wmma/hgemm_wmma.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️⭐️|
| ✔️ [hgemm_wmma_m32n8k16....dbuf*](./kernels/hgemm/wmma/hgemm_wmma.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️⭐️|
| ✔️ [hgemm_wmma_m16n16k16...stages*](./kernels/hgemm/wmma/hgemm_wmma_stage.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️⭐️|
| ✔️ [hgemm_wmma_m16n16k16...swizzle*](./kernels/hgemm/wmma/hgemm_wmma_stage.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️⭐️|
| ✔️ [hgemm_mma_m16n8k16...naive*](./kernels/hgemm/mma/basic/hgemm_mma.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️⭐️|
| ✔️ [hgemm_mma_m16n8k16...mma2x4*](./kernels/hgemm/mma/basic/hgemm_mma.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️⭐️|
| ✔️ [hgemm_mma_m16n8k16...stages*](./kernels/hgemm/mma/basic/hgemm_mma_stage.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️⭐️|
| ✔️ [hgemm_mma_m16n8k16...swizzle*](./kernels/hgemm/mma/basic/hgemm_mma_stage.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️⭐️|
| ✔️ [hgemm_mma_m16n8k16...swizzle{smem}*](./kernels/hgemm/mma/swizzle/hgemm_mma_stage_swizzle.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️⭐️|
| ✔️ [hgemm_mma_m16n8k16...swizzle{tn}{smem}*](./kernels/hgemm/mma/swizzle/hgemm_mma_stage_tn_swizzle_x4.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️⭐️|
| ✔️ [hgemm_mma_stages_swizzle{smem}...cute*](./kernels/hgemm/cutlass/hgemm_mma_stage_tn_cute.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️⭐️|
| ✔️ [hgemm_mma_cublas*](./kernels/hgemm/cublas/hgemm_cublas.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️|

### 📚 Hard+ ⭐️⭐️⭐️⭐️ & Hard++ ⭐️⭐️⭐️⭐️⭐️ ([©️back👆🏻](#cuda-kernel))

- 📚 FlashAttention-2 MMA (MMA Acc F32/F16, swizzle, QKV smem share, fine-grained tiling, etc.🎉)

<div id="cuda-kernel-hard-plus"></div>

|📖 CUDA Kernel| 📖 Elem DType| 📖 Acc DType| 📖 Docs | 📖 Level |
|:---|:---|:---|:---|:---|
| ✔️ [flash_attn_cute(naive)](./kernels/flash-attn/cutlass/flash_attn_cute.cu)|f16|f32|[link](./kernels/flash-attn/)|⭐️⭐️⭐️|
| ✔️ [How to implement MMA smem swizzle*](./kernels/swizzle/mma_simple_swizzle.cu)|f16|f16|[link](./kernels/swizzle)|⭐️⭐️⭐️|
| ✔️ [flash_attn_mma_stages_split_kv*](./kernels/flash-attn/mma/basic/flash_attn_mma_split_kv.cu)|f16|f16|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma_stages_split_q*](./kernels/flash-attn/mma/basic/flash_attn_mma_split_q.cu)|f16|f16|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma_stages...shared_kv*](./kernels/flash-attn/mma/basic/flash_attn_mma_share_kv.cu)|f16|f16|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma_stages...shared_qkv*](./kernels/flash-attn/mma/basic/flash_attn_mma_share_qkv.cu)|f16|f16|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma_stages...tiling_qk*](./kernels/flash-attn/mma/basic/flash_attn_mma_tiling_qk.cu)|f16|f16|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma_stages...tiling_qkv*](./kernels/flash-attn/mma/basic/flash_attn_mma_tiling_qkv.cu)|f16|f16|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma_stages...shared_kv{f32}*](./kernels/flash-attn/mma/basic/flash_attn_mma_share_kv_F32F16F16F32.cu)|f16|f32|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma_stages...shared_qkv{f32}*](./kernels/flash-attn/mma/basic/flash_attn_mma_share_qkv_F32F16F16F32.cu)|f16|f32|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma_stages...tiling_qk{f32}*](./kernels/flash-attn/mma/basic/flash_attn_mma_tiling_qk_F32F16F16F32.cu)|f16|f32|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma_stages...tiling_qkv{f32}*](./kernels/flash-attn/mma/basic/flash_attn_mma_tiling_qkv_F32F16F16F32.cu)|f16|f32|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma...shared_kv{f32}{rr}*](./kernels/flash-attn/mma/others/flash_attn_mma_share_kv_F32F16F16F32_rr.cu)|f16|f32|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma...shared_qkv{f32}{rr}*](./kernels/flash-attn/mma/others/flash_attn_mma_share_qkv_F32F16F16F32_rr.cu)|f16|f32|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma...shared_kv_swizzle{q}*](./kernels/flash-attn/mma/swizzle/flash_attn_mma_share_kv_swizzle_q.cu)|f16|f16|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma...shared_kv_swizzle{qk}*](./kernels/flash-attn/mma/swizzle/flash_attn_mma_share_kv_swizzle_qk.cu)|f16|f16|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma...shared_kv_swizzle{qkv}*](./kernels/flash-attn/mma/swizzle/flash_attn_mma_share_kv_swizzle_qkv.cu)|f16|f16|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma...shared_qkv_swizzle{q}*](./kernels/flash-attn/mma/swizzle/flash_attn_mma_share_qkv_swizzle_q.cu)|f16|f16|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma...shared_qkv_swizzle{qk}*](./kernels/flash-attn/mma/swizzle/flash_attn_mma_share_qkv_swizzle_qk.cu)|f16|f16|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma...shared_qkv_swizzle{qkv}*](./kernels/flash-attn/mma/swizzle/flash_attn_mma_share_qkv_swizzle_qkv.cu)|f16|f16|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma...tiling_qk_swizzle{q}*](./kernels/flash-attn/mma/swizzle/flash_attn_mma_tiling_qk_swizzle_q.cu)|f16|f16|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma...tiling_qk_swizzle{qk}*](./kernels/flash-attn/mma/swizzle/flash_attn_mma_tiling_qk_swizzle_qk.cu)|f16|f16|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma...tiling_qk_swizzle{qkv}*](./kernels/flash-attn/mma/swizzle/flash_attn_mma_tiling_qk_swizzle_qkv.cu)|f16|f16|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma...tiling_qkv_swizzle{q}*](./kernels/flash-attn/mma/swizzle/flash_attn_mma_tiling_qkv_swizzle_q.cu)|f16|f16|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma...tiling_qkv_swizzle{qk}*](./kernels/flash-attn/mma/swizzle/flash_attn_mma_tiling_qkv_swizzle_qk.cu)|f16|f16|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn_mma...tiling_qkv_swizzle{qkv}*](./kernels/flash-attn/mma/swizzle/flash_attn_mma_tiling_qkv_swizzle_qkv.cu)|f16|f16|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn...tiling_qkv_swizzle{q}{f32}*](./kernels/flash-attn/mma/swizzle/flash_attn_mma_tiling_qkv_swizzle_q_F32F16F16F32.cu)|f16|f32|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn...tiling_qkv_swizzle{qk}{f32}*](./kernels/flash-attn/mma/swizzle/flash_attn_mma_tiling_qkv_swizzle_qk_F32F16F16F32.cu)|f16|f32|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [flash_attn...tiling_qkv_swizzle{qkv}{f32}*](./kernels/flash-attn/mma/swizzle/flash_attn_mma_tiling_qkv_swizzle_qkv_F32F16F16F32.cu)|f16|f32|[link](./kernels/flash-attn)|⭐️⭐️⭐️⭐️|

💡NOTE: **rr**: means reduce registers usage (for `d>128`); **f32**: means MMA accumulate with FP32 dtype, otherwise, FP16. softmax Acc dtype is always be FP32 for high precision; **swizzle**: now, only support smem swizzle for MMA.

- 📚 FFPA Attention MMA (**1.8x~3x**🎉faster vs SDPA EA, D > 256, FA2 not supported)

|📖 CUDA Kernel| 📖 Elem DType| 📖 Acc DType| 📖 Docs | 📖 Level |
|:---|:---|:---|:---|:---|
| ✔️ [ffpa_mma_stages_split_q_L1_F16F16F16](https://github.com/xlite-dev/ffpa-attn/blob/main/csrc/cuffpa/ffpa_attn_F16F16F16_L1.cu)|f16|f16|[link](https://github.com/xlite-dev/ffpa-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [ffpa_mma_stages_split_q_L1_F16F16F32](https://github.com/xlite-dev/ffpa-attn/blob/main/csrc/cuffpa/ffpa_attn_F16F16F32_L1.cu)|f16|f32|[link](https://github.com/xlite-dev/ffpa-attn)|⭐️⭐️⭐️⭐️|
| ✔️ [ffpa_mma_stages_split_q_L1_mixed_acc](https://github.com/xlite-dev/ffpa-attn/blob/main/csrc/cuffpa/ffpa_attn_F16F16F32_L1.cu)|f16|QK f32, PV f16|[link](https://github.com/xlite-dev/ffpa-attn)|⭐️⭐️⭐️⭐️|
| ⚠️ [ffpa_mma_stages_split_q_L2_F16F16F16](https://github.com/xlite-dev/ffpa-attn/blob/main/csrc/cuffpa/ffpa_attn_F16F16F16_L2.cu)|f16|f16|[link](https://github.com/xlite-dev/ffpa-attn)|⭐️⭐️⭐️⭐️|
| ⚠️ [ffpa_mma_stages_split_q_L2_F16F16F32](https://github.com/xlite-dev/ffpa-attn/blob/main/csrc/cuffpa/ffpa_attn_F16F16F32_L2.cu)|f16|f32|[link](https://github.com/xlite-dev/ffpa-attn)|⭐️⭐️⭐️⭐️|
| ⚠️ [ffpa_mma_stages_split_q_L2_mixed_acc](https://github.com/xlite-dev/ffpa-attn/blob/main/csrc/cuffpa/ffpa_attn_F16F16F32_L2.cu)|f16|QK f32, PV f16|[link](https://github.com/xlite-dev/ffpa-attn)|⭐️⭐️⭐️⭐️|
| ⚠️ [ffpa_mma_stages_split_q_L3_F16F16F16](https://github.com/xlite-dev/ffpa-attn/blob/main/csrc/cuffpa/ffpa_attn_F16F16F16_L3.cu)|f16|f16|[link](https://github.com/xlite-dev/ffpa-attn)|⭐️⭐️⭐️⭐️|
| ⚠️ [ffpa_mma_stages_split_q_L3_F16F16F32](https://github.com/xlite-dev/ffpa-attn/blob/main/csrc/cuffpa/ffpa_attn_F16F16F32_L3.cu)|f16|f32|[link](https://github.com/xlite-dev/ffpa-attn)|⭐️⭐️⭐️⭐️|
| ⚠️ [ffpa_mma_stages_split_q_L3_mixed_acc](https://github.com/xlite-dev/ffpa-attn/blob/main/csrc/cuffpa/ffpa_attn_F16F16F32_L3.cu)|f16|QK f32, PV f16|[link](https://github.com/xlite-dev/ffpa-attn)|⭐️⭐️⭐️⭐️|

💡NOTE: 🤖[ffpa-attn](https://github.com/xlite-dev/ffpa-attn): 📚FFPA - Yet another Faster Flash Prefill Attention with O(1)🎉SRAM complexity for headdim > 256, **1.8x~3x**🎉faster than SDPA EA: [📈L20 ~1.9x↑🎉](https://github.com/xlite-dev/ffpa-attn?tab=readme-ov-file#L1-bench-l20), [📈 A30 ~1.8x↑🎉](https://github.com/xlite-dev/ffpa-attn?tab=readme-ov-file#L1-bench-a30), [📈3080 ~2.9x↑🎉](https://github.com/xlite-dev/ffpa-attn?tab=readme-ov-file#L1-bench-3080), [📈4090 ~2.1x↑🎉](https://github.com/xlite-dev/ffpa-attn?tab=readme-ov-file#L1-bench-4090).

### 📚 Triton Kernel (OpenAI Triton) ⭐️⭐️⭐️ ([©️back👆🏻](#cuda-kernel))

<div id="triton-kernel"></div>

|📖 Triton Kernel| 📖 Elem DType| 📖 Acc DType| 📖 Docs | 📖 Level |
|:---|:---|:---|:---|:---|
| ✔️ [triton_vector_add_kernel](./kernels/openai-triton/vector-add/)|all|all|[link](./kernels/openai-triton/vector-add/)|⭐️⭐️|
| ✔️ [triton_fused_softmax(multi-stages)](./kernels/openai-triton/fused-softmax/)|f16/bf16/f32|f32|[link](./kernels/openai-triton/fused-softmax/)|⭐️⭐️⭐️|
| ✔️ [triton_fused_layer_norm(forward-pass)](./kernels/openai-triton/layer-norm/)|f16/bf16/f32|f32|[link](./kernels/openai-triton/layer-norm/)|⭐️⭐️⭐️|
| ✔️ [triton_fused_layer_norm(backward-pass)](./kernels/openai-triton/layer-norm/)|f16/bf16/f32|f32|[link](./kernels/openai-triton/layer-norm/)|⭐️⭐️⭐️|
| ✔️ [triton_merge_attn_states_kernel(w/ CUDA)](./kernels/openai-triton/merge-attn-states/)|f16/bf16/f32|f32|[link](./kernels/openai-triton/merge-attn-states/)|⭐️⭐️⭐️|

### 📚 CUTLASS/CuTe Kernel ⭐️⭐️⭐️ ([©️back👆🏻](#cuda-kernel))

<div id="cutlass-kernel"></div>

|📖 CUTLASS/CuTe Kernel| 📖 Elem DType| 📖 Acc DType| 📖 Docs | 📖 Level |
|:---|:---|:---|:---|:---|
| ✔️ [mat_transpose_cute](./kernels/mat-transpose/mat_transpose_cute.cu)|f32|/|[link](./kernels/mat-transpose/)|⭐️⭐️|
| ✔️ [flash_attn_cute(naive)](./kernels/flash-attn/cutlass/flash_attn_cute.cu)|f16|f32|[link](./kernels/flash-attn/)|⭐️⭐️⭐️|
| ✔️ [hgemv_f16_cute_kernel](./kernels/hgemv/hgemv_cute.cu)|f16|f16|[link](./kernels/hgemv/)|⭐️⭐️⭐️|
| ✔️ [hgemv_f16x8_cute_kernel](./kernels/hgemv/hgemv_cute.cu)|f16|f16|[link](./kernels/hgemv/)|⭐️⭐️⭐️|
| ✔️ [hgemv_tensor_core_cute_kernel](./kernels/hgemv/hgemv_cute.cu)|f16|f16|[link](./kernels/hgemv/)|⭐️⭐️⭐️|
| ✔️ [hgemm_mma_stages_swizzle{smem}...cute*](./kernels/hgemm/cutlass/hgemm_mma_stage_tn_cute.cu)|f16|f16|[link](./kernels/hgemm/)|⭐️⭐️⭐️|
| ✔️ [ws_hgemm_naive_cute_kernel](./kernels/ws-hgemm/naive_ws_hgemm_sm8x.cu)|f16|f16|[link](./kernels/ws-hgemm/)|⭐️⭐️⭐️|

## 📖 100+ 高性能计算与分布式-技术博客

<div id="my-blogs-part-1"></div>

### 📚 高性能计算与分布式-个人技术专栏 ([©️back👆🏻](#contents))

|📖 类型-标题|📖 作者| 📖 推荐 |
|:---|:---|:---|
| [[Diffusion推理]📖简短的2025年总结，写在Cache-DiT v1.2.1之际](https://zhuanlan.zhihu.com/p/2001692370358539662)|@DefTruth|⭐️⭐️|
| [[Diffusion推理]📖CacheDiT支持Z-Image分布式推理和缓存加速​​](https://zhuanlan.zhihu.com/p/1978490962742374735)|@DefTruth|⭐️⭐️|
| [[Diffusion推理]📖cache-dit支持FLUX.2分布式推理和Cache](https://zhuanlan.zhihu.com/p/1977698505834379041)|@DefTruth|⭐️⭐️|
| [[Diffusion推理]📖Cache加速-FoCa公式理解记录](https://zhuanlan.zhihu.com/p/1952056591068144338)|@DefTruth|⭐️⭐️⭐|
| [[Diffusion推理]📖cache-dit: BlockAdapter支持HunyuanImage-2.1 Cache加速!](https://zhuanlan.zhihu.com/p/1950849526400263083)|@DefTruth|⭐️⭐️⭐|
| [[Diffusion推理]📖cache-dit + Qwen-Image-Lightning 实现 3.5 steps 推理!](https://zhuanlan.zhihu.com/p/1948696529180295613)|@DefTruth|⭐️⭐️⭐|
| [[Diffusion推理]📖cache-dit: Wan2.2-MoE 2.4x 推理加速!](https://zhuanlan.zhihu.com/p/1943976514321380955)|@DefTruth|⭐️⭐️⭐|
| [[Diffusion推理]📖cache-dit: Qwen-Image-Edit 2x 无损加速!](https://zhuanlan.zhihu.com/p/1941503245764792443)|@DefTruth|⭐️⭐️⭐|
| [[Diffusion推理]📖cache-dit: Qwen-Image 1.5x 无损加速!](https://zhuanlan.zhihu.com/p/1938547315221705644)|@DefTruth|⭐️⭐️⭐|
| [[Diffusion推理]📖Cache加速-TaylorSeer算法简析](https://zhuanlan.zhihu.com/p/1937477466475197176)|@DefTruth|⭐️⭐️⭐|
| [[Diffusion推理]📖DiT推理加速综述: Caching](https://zhuanlan.zhihu.com/p/711223667)|@DefTruth|⭐️⭐️⭐|
| [[Triton编程][基础]📖Triton极简入门: Triton Vector Add](https://zhuanlan.zhihu.com/p/1902778199261291694)|@DefTruth|⭐️⭐️⭐|
| [[Triton编程][基础]📖Triton Fused Softmax Kernel详解: 从Python源码到PTX](https://zhuanlan.zhihu.com/p/1899562146477609112)|@DefTruth|⭐️⭐️⭐|
| [[Triton编程][基础]📖vLLM Triton Merge Attention States Kernel详解](https://zhuanlan.zhihu.com/p/1904937907703243110)|@DefTruth|⭐️⭐️⭐|
| [[Triton编程][进阶]📖vLLM Prefix Prefill Triton Kernel图解](https://zhuanlan.zhihu.com/p/695799736)|@DefTruth|⭐️⭐️⭐️|
| [[张量/序列并行]📖序列并行: BPT、Ring-Attention及Striped-Attention笔记](https://zhuanlan.zhihu.com/p/6456708235)|@DefTruth|⭐️⭐️⭐|
| [[vLLM实践][算子]📖vLLM算子开发流程：”保姆级“详细记录](https://zhuanlan.zhihu.com/p/1892966682634473987)|@DefTruth|⭐️⭐️⭐|
| [[vLLM实践][万字]📖vLLM + DeepSeek-R1 671B 多机部署及修Bug笔记](https://zhuanlan.zhihu.com/p/29950052712)|@DefTruth|⭐️⭐️⭐|
| [[Attention优化]📖FFPA(Split-D): FA2无限HeadDim扩展，2x↑🎉 vs SDPA EA](https://zhuanlan.zhihu.com/p/13975660308)|@DefTruth|⭐️⭐️⭐️|
| [[CUDA基础][开篇]📖LeetCUDA: v3.0 大升级-面试刷题不迷路](https://zhuanlan.zhihu.com/p/19862356369)|@DefTruth|⭐️⭐️⭐⭐️|
| [[分布式训推][张量/序列并行]📖图解DeepSpeed-Ulysses&Megatron-LM TP/SP](https://zhuanlan.zhihu.com/p/5750410146)|@DefTruth|⭐️⭐️|
| [[VLM推理优化][InternVL系列]📖InternLM2/.../InternVL1.5系列笔记: 核心点解析](https://zhuanlan.zhihu.com/p/702481058)|@DefTruth|⭐️⭐️|
| [[LLM推理优化][TensorRT-LLM][5w字]📖TensorRT-LLM部署调优-指北](https://zhuanlan.zhihu.com/p/699333691)|@DefTruth|⭐️⭐️⭐️|
| [[LLM推理优化][KV Cache优化]📖GQA/YOCO/CLA/MLKV: 层内和层间KV Cache共享](https://zhuanlan.zhihu.com/p/697311739)|@DefTruth|⭐️⭐️|
| [[LLM推理优化][Prefill优化][万字]📖图解vLLM Automatic Prefix Caching: TTFT优化](https://zhuanlan.zhihu.com/p/693556044)|@DefTruth|⭐️⭐️⭐️|
| [[LLM推理优化][Attention优化]📖图解:从Online-Softmax到FlashAttention V1/V2/V3](https://zhuanlan.zhihu.com/p/668888063)|@DefTruth|⭐️⭐️⭐️|
| [[LLM推理优化][Decoding优化]📖原理&图解FlashDecoding/FlashDecoding++](https://zhuanlan.zhihu.com/p/696075602)|@DefTruth|⭐️⭐️|
| [[VLM推理优化][LLaVA系列]📖CLIP/LLaVA/LLaVA1.5/VILA笔记: 核心点解析](https://zhuanlan.zhihu.com/p/683137074)|@DefTruth|⭐️⭐️|
| [[LLM推理优化][Attention优化][万字]📖TensorRT MHA/Myelin vs FlashAttention-2](https://zhuanlan.zhihu.com/p/678873216)|@DefTruth|⭐️⭐️⭐️|
| [[LLM推理优化][PTX汇编]📖CUDA 12 PTX汇编: PRMT指令详解-通用模式](https://zhuanlan.zhihu.com/p/660630414)|@DefTruth|⭐️|
| [[LLM推理优化][PTX汇编]📖CUDA 12 PTX汇编: LOP3指令详解](https://zhuanlan.zhihu.com/p/659741469)|@DefTruth|⭐️|
| [[LLM推理优化][CUDA][3w字]📖高频面试题汇总-大模型手撕CUDA](https://zhuanlan.zhihu.com/p/678903537)|@DefTruth|⭐️⭐️⭐️|
| [[LLM推理优化][Weight Only]📖WINT8/4-(00): 通俗易懂讲解-快速反量化算法](https://zhuanlan.zhihu.com/p/657072856)|@DefTruth|⭐️⭐️|
| [[LLM推理优化][Weight Only]📖WINT8/4-(01): PRMT指令详解及FT源码解析](https://zhuanlan.zhihu.com/p/657070837)|@DefTruth|⭐️⭐️|
| [[LLM推理优化][Weight Only]📖WINT8/4-(02): 快速反量化之INT8转BF16](https://zhuanlan.zhihu.com/p/657073159)|@DefTruth|⭐️⭐️|
| [[LLM推理优化][Weight Only]📖WINT8/4-(03): LOP3指令详解及INT4转FP16/BF16](https://zhuanlan.zhihu.com/p/657073857)|@DefTruth|⭐️⭐️|
| [[LLM推理优化][LLM Infra整理]📖100+篇: 大模型推理各方向新发展整理](https://zhuanlan.zhihu.com/p/693680304)|@DefTruth|⭐️⭐️|
| [[LLM推理优化][LLM Infra整理]📖30+篇: LLM推理论文集-500页PDF](https://zhuanlan.zhihu.com/p/669777159)|@DefTruth|⭐️⭐️|
| [[LLM推理优化][LLM Infra整理]📖FlashDecoding++: 比FlashDecoding还要快！](https://zhuanlan.zhihu.com/p/665022589)|@DefTruth|⭐️|
| [[LLM推理优化][LLM Infra整理]📖TensorRT-LLM开源，TensorRT 9.1也来了](https://zhuanlan.zhihu.com/p/662361469)|@DefTruth|⭐️|
| [[LLM推理优化][LLM Infra整理]📖20+篇: LLM推理论文集-300页PDF](https://zhuanlan.zhihu.com/p/658091768)|@DefTruth|⭐️⭐️|
| [[LLM推理优化][LLM Infra整理]📖PagedAttention论文新鲜出炉](https://zhuanlan.zhihu.com/p/617015570)|@DefTruth|⭐️|
| [[推理部署][CV/NLP]📖FastDeploy三行代码搞定150+ CV、NLP模型部署](https://zhuanlan.zhihu.com/p/581326442)|@DefTruth|⭐️|
| [[推理部署][CV]📖如何在lite.ai.toolkit(3.6k+ stars)中增加您的模型？](https://zhuanlan.zhihu.com/p/523876625)|@DefTruth|⭐️⭐️|
| [[推理部署][CV]📖美团 YOLOv6 ORT/MNN/TNN/NCNN C++推理部署](https://zhuanlan.zhihu.com/p/533643238)|@DefTruth|⭐️⭐️|
| [[推理部署][ONNX]📖ONNX推理加速技术文档-杂记](https://zhuanlan.zhihu.com/p/524023964)|@DefTruth|⭐️|
| [[推理部署][TensorFlow]📖Mac源码编译TensorFlow C++指北](https://zhuanlan.zhihu.com/p/524013615)|@DefTruth|⭐️|
| [[推理部署][CV]📖1Mb!头部姿态估计: FSANet，一个小而美的模型(C++)](https://zhuanlan.zhihu.com/p/447364201)|@DefTruth|⭐️|
| [[推理部署][CV]📖opencv+ffmpeg编译打包全解指南](https://zhuanlan.zhihu.com/p/472115312)|@DefTruth|⭐️⭐️|
| [[推理部署][CV]📖RobustVideoMatting视频抠图静态ONNX模型转换](https://zhuanlan.zhihu.com/p/459088407)|@DefTruth|⭐️|
| [[推理部署][CV]📖190Kb!SSRNet年龄检测详细解读（含C++工程）](https://zhuanlan.zhihu.com/p/462762797)|@DefTruth|⭐️|
| [[推理部署][CV]📖MGMatting(CVPR2021)人像抠图C++应用记录](https://zhuanlan.zhihu.com/p/464732042)|@DefTruth|⭐️|
| [[推理部署][CV]📖超准确人脸检测(带关键点)YOLO5Face C++工程详细记录](https://zhuanlan.zhihu.com/p/461878005)|@DefTruth|⭐️⭐️|
| [[推理部署][ORT]📖解决: ONNXRuntime(Python) GPU 部署配置记录](https://zhuanlan.zhihu.com/p/457484536)|@DefTruth|⭐️|
| [[推理部署][CV]📖记录SCRFD(CVPR2021)人脸检测C++工程化(含docker镜像)](https://zhuanlan.zhihu.com/p/455165568)|@DefTruth|⭐️⭐️|
| [[推理部署][NCNN]📖野路子：记录一个解决onnx转ncnn时op不支持的trick](https://zhuanlan.zhihu.com/p/451446147)|@DefTruth|⭐️|
| [[推理部署][CV]📖升级版NanoDet-Plus MNN/TNN/NCNN/ORT C++工程记录](https://zhuanlan.zhihu.com/p/450586647)|@DefTruth|⭐️⭐️|
| [[推理部署][CV]📖超轻量级NanoDet MNN/TNN/NCNN/ORT C++工程记录](https://zhuanlan.zhihu.com/p/443419387)|@DefTruth|⭐️|
| [[推理部署][CV]📖详细记录MGMatting之MNN、TNN和ORT C++移植](https://zhuanlan.zhihu.com/p/442949027)|@DefTruth|⭐️⭐️|
| [[推理部署][CV]📖YOLOX NCNN/MNN/TNN/ONNXRuntime C++工程简记](https://zhuanlan.zhihu.com/p/447364122)|@DefTruth|⭐️|
| [[推理部署][TNN]📖手动修改YoloX的tnnproto记录-TNN](https://zhuanlan.zhihu.com/p/425668734)|@DefTruth|⭐️|
| [[推理部署][ORT]📖全网最详细 ONNXRuntime C++/Java/Python 资料！](https://zhuanlan.zhihu.com/p/414317269)|@DefTruth|⭐️|
| [[推理部署][CV]📖RobustVideoMatting: C++工程化记录-实现篇](https://zhuanlan.zhihu.com/p/413280488)|@DefTruth|⭐️⭐️|
| [[推理部署][CV]📖RobustVideoMatting: C++工程化记录-应用篇](https://zhuanlan.zhihu.com/p/412491918)|@DefTruth|⭐️⭐️|
| [[推理部署][ORT]📖ONNXRuntime C++ CMake 工程分析及编译](https://zhuanlan.zhihu.com/p/411887386)|@DefTruth|⭐️⭐️|
| [[推理部署][ORT]📖如何使用ORT C++ API处理NCHW和NHWC输入？](https://zhuanlan.zhihu.com/p/524230808)|@DefTruth|⭐️|
| [[推理部署][TNN]📖tnn-convert搭建简记-YOLOP转TNN](https://zhuanlan.zhihu.com/p/431418709)|@DefTruth|⭐️|
| [[推理部署][CV]📖YOLOP ONNXRuntime C++工程化记录](https://zhuanlan.zhihu.com/p/411651933)|@DefTruth|⭐️⭐️|
| [[推理部署][NCNN]📖超有用NCNN参考资料整理](https://zhuanlan.zhihu.com/p/449765328)|@DefTruth|⭐️|
| [[推理部署][MNN]📖超有用MNN参考资料整理](https://zhuanlan.zhihu.com/p/449761992)|@DefTruth|⭐️|
| [[推理部署][TNN]📖超有用TNN参考资料整理](https://zhuanlan.zhihu.com/p/449769615)|@DefTruth|⭐️|
| [[推理部署][ONNX]📖超有用ONNX参考资料整理](https://zhuanlan.zhihu.com/p/449773663)|@DefTruth|⭐️|
| [[推理部署][ONNX]📖超有用ONNX模型结构参考资料整理](https://zhuanlan.zhihu.com/p/449775926)|@DefTruth|⭐️|
| [[推理部署][OpenCV-DNN]📖超有用OpenCV-DNN参考资料整理](https://zhuanlan.zhihu.com/p/449778377)|@DefTruth|⭐️|
| [[推理部署][Tensorflow]📖超有用Tensorflow C++工程化知识点](https://zhuanlan.zhihu.com/p/449788027)|@DefTruth|⭐️|
| [[推理部署][模型转换]📖深度学习模型转换资料整理](https://zhuanlan.zhihu.com/p/449759361)|@DefTruth|⭐️|
| [[技术随笔][C++][CMake]📖超有用CMake参考资料整理](https://zhuanlan.zhihu.com/p/449779892)|@DefTruth|⭐️⭐️|
| [[技术随笔][C++][3W字]📖静态链接和静态库实践指北-原理篇](https://zhuanlan.zhihu.com/p/595527528)|@DefTruth|⭐️⭐️⭐️|
| [[技术随笔][C++]📖Mac下C++内存检查指北(Valgrind VS Asan)](https://zhuanlan.zhihu.com/p/508470880)|@DefTruth|⭐️|
| [[技术随笔][CV]📖torchlm: 人脸关键点检测库](https://zhuanlan.zhihu.com/p/467211561)|@DefTruth|⭐️⭐️|
| [[技术随笔][ML]📖《统计学习方法-李航: 笔记-从原理到实现-基于R》](https://zhuanlan.zhihu.com/p/684885595)|@DefTruth|⭐️⭐️|
| [[技术随笔][Git]📖如何优雅地git clone和git submodule？](https://zhuanlan.zhihu.com/p/639136221)|@DefTruth|⭐️|
| [[技术随笔][3D]📖人脸重建3D参考资料整理](https://zhuanlan.zhihu.com/p/524034741)|@DefTruth|⭐️|
| [[技术随笔][3D]📖BlendShapes参考资料整理](https://zhuanlan.zhihu.com/p/524036145)|@DefTruth|⭐️|
| [[技术随笔][3D]📖从源码安装Pytorch3D详细记录及学习资料](https://zhuanlan.zhihu.com/p/512347464)|@DefTruth|⭐️|
| [[技术随笔][ML]📖200页:《统计学习方法：李航》笔记 -从原理到实现](https://zhuanlan.zhihu.com/p/461520847)|@DefTruth|⭐️⭐️|

### 📚 高性能计算与分布式-技术博客推荐 ([©️back👆🏻](#contents))

<div id="other-blogs"></div>

💡说明: 本小节整理一些自己比较喜欢的文章。欢迎大家提PR推荐更多优秀的文章！

|📖 类型-标题|📖 作者| 📖 推荐 |
|:---|:---|:---|
| [[cute系列详解][入门]📖cutlass cute 101](https://zhuanlan.zhihu.com/p/660379052)|@朱小霖|⭐️⭐️⭐️|
| [[cute系列详解][入门]📖CUTLASS 2.x & CUTLASS 3.x Intro 学习笔记](https://zhuanlan.zhihu.com/p/710516489)|@BBuf|⭐️⭐️⭐️|
| [[cute系列详解][入门]📖写给大家看的 CuTe 教程：tiled copy](https://zhuanlan.zhihu.com/p/1930389542784964333)|@竹熙佳处|⭐️⭐️⭐️|
| [[cute系列详解][入门]📖写给大家看的 CuTe 教程：tiled mma](https://zhuanlan.zhihu.com/p/1937145378446226159)|@竹熙佳处|⭐️⭐️⭐️|
| [[cute系列详解][入门]📖写给大家看的 CuTe 教程：Layout Compose & Inverse](https://zhuanlan.zhihu.com/p/1962625273636845008)|@竹熙佳处|⭐️⭐️⭐️|
| [[cute系列详解][入门]📖写给大家看的 CuTe 教程: Layout Product & Divide](https://zhuanlan.zhihu.com/p/1971945267294111573)|@竹熙佳处|⭐️⭐️⭐️|
| [[cute系列详解][入门]📖写给大家看的 CuTe 教程：TMA Copy](https://zhuanlan.zhihu.com/p/2003198909405763007)|@竹熙佳处|⭐️⭐️⭐️|
| [[cute系列详解][入门]📖写给进阶开发的 CuTe 笔记：permutationMNK 参数](https://zhuanlan.zhihu.com/p/1973526710105419953)|@竹熙佳处|⭐️⭐️⭐️|
| [[cute系列详解][Layout]📖cute 之 Layout](https://zhuanlan.zhihu.com/p/661182311)|@reed|⭐️⭐️⭐️|
| [[cute系列详解][Layout]📖cute Layout 的代数和几何解释](https://zhuanlan.zhihu.com/p/662089556)|@reed|⭐️⭐️⭐️|
| [[cute系列详解][Tensor]📖cute 之 Tensor](https://zhuanlan.zhihu.com/p/663093816)|@reed|⭐️⭐️⭐️|
| [[cute系列详解][MMA]📖cute 之 MMA抽象](https://zhuanlan.zhihu.com/p/663092747)|@reed|⭐️⭐️⭐️|
| [[cute系列详解][Copy]📖cute 之 Copy抽象](https://zhuanlan.zhihu.com/p/666232173)|@reed|⭐️⭐️⭐️|
| [[cute系列详解][Swizzle]📖cute 之 Swizzle](https://zhuanlan.zhihu.com/p/671419093)|@reed|⭐️⭐️⭐️|
| [[cute系列详解][Swizzle]📖cute Swizzle细谈](https://zhuanlan.zhihu.com/p/684250988)|@进击的Killua|⭐️⭐️⭐️|
| [[cute系列详解][Swizzle]📖cutlass swizzle机制解析（一）](https://zhuanlan.zhihu.com/p/710337546)|@Titus|⭐️⭐️⭐️|
| [[cute系列详解][Swizzle]📖cutlass swizzle机制解析（二）](https://zhuanlan.zhihu.com/p/711398930)|@Titus|⭐️⭐️⭐️|
| [[cute系列详解][Swizzle]📖CUDA避免smem bank conflict的swizzle机制解析](https://zhuanlan.zhihu.com/p/4746910252)|@frankshi|⭐️⭐️⭐️|
| [[cute系列详解][Swizzle]📖布局代数实战：Swizzle自动推导](https://zhuanlan.zhihu.com/p/1941306442683515068)|@melonedo|⭐️⭐️⭐️|
| [[cute系列详解][GEMM]📖cute 之 简单GEMM实现](https://zhuanlan.zhihu.com/p/667521327)|@reed|⭐️⭐️⭐️|
| [[cute系列详解][GEMM]📖cute 之 GEMM流水线](https://zhuanlan.zhihu.com/p/665082713)|@reed|⭐️⭐️⭐️|
| [[cute系列详解][GEMM]📖cute 之 高效GEMM实现](https://zhuanlan.zhihu.com/p/675308830)|@reed|⭐️⭐️⭐️|
| [[cute系列详解][GEMM]📖GEMM流水线: single/multi-stage、pipeline](https://zhuanlan.zhihu.com/p/712451053)|@Titus|⭐️⭐️⭐️|
| [[cute系列详解][GEMM]📖GEMM细节分析(一): ldmatrix的选择](https://zhuanlan.zhihu.com/p/702818267)|@Anonymous|⭐️⭐️⭐️|
| [[cute系列详解][GEMM]📖GEMM细节分析(二): TiledCopy与cp.async](https://zhuanlan.zhihu.com/p/703560147)|@Anonymous|⭐️⭐️⭐️|
| [[cute系列详解][GEMM]📖GEMM细节分析(三): Swizzle<B,M,S>参数取值](https://zhuanlan.zhihu.com/p/713713957)|@Anonymous|⭐️⭐️⭐️|
| [[cute系列详解][实践]📖Hopper Mixed GEMM的CUTLASS实现笔记](https://zhuanlan.zhihu.com/p/714378343)|@BBuf|⭐️⭐️⭐️|
| [[cute系列详解][实践]📖CUTLASS CuTe实战(一): 基础](https://zhuanlan.zhihu.com/p/690703999)|@进击的Killua|⭐️⭐️⭐️|
| [[cute系列详解][实践]📖CUTLASS CuTe实战(二): 应用](https://zhuanlan.zhihu.com/p/692078624)|@进击的Killua|⭐️⭐️⭐️|
| [[cute系列详解][实践]📖FlashAttention fp8实现（ada架构)](https://zhuanlan.zhihu.com/p/712314257)|@shengying.wei|⭐️⭐️⭐️|
| [[cute系列详解][实践]📖FlashAttention 笔记: tiny-flash-attention解读](https://zhuanlan.zhihu.com/p/708867810)|@shengying.wei|⭐️⭐️⭐️|
| [[cute系列详解][实践]📖使用cutlass cute复现flash attention](https://zhuanlan.zhihu.com/p/696323042)|@66RING|⭐️⭐️⭐️|
| [[cutlass教程][入门]📖cutlass 基本认知](https://zhuanlan.zhihu.com/p/677616101)|@JoeNomad|⭐️⭐️⭐️|
| [[cutlass教程][入门]📖cutlass 软件架构](https://zhuanlan.zhihu.com/p/678915618)|@JoeNomad|⭐️⭐️⭐️|
| [[cutlass教程][入门]📖CUTLASS 基础介绍](https://zhuanlan.zhihu.com/p/671324125)|@进击的Killua|⭐️⭐️⭐️|
| [[cutlass教程][入门]📖乱谈CUTLASS GTC2020 SLIDES](https://zhuanlan.zhihu.com/p/674693873)|@zzk again|⭐️⭐️⭐️|
| [[cutlass教程][深入]📖cutlass block swizzle 和 tile iterator](https://zhuanlan.zhihu.com/p/679929705)|@JoeNomad|⭐️⭐️⭐️|
| [[cutlass教程][深入]📖cutlass bank conflict free的smem layout](https://zhuanlan.zhihu.com/p/681966685)|@JoeNomad|⭐️⭐️⭐️|
| [[cutlass教程][深入]📖cutlass 多级流水线](https://zhuanlan.zhihu.com/p/687397095)|@JoeNomad|⭐️⭐️⭐️|
| [[GPU指令集架构][精解]📖NVidia GPU指令集架构-前言](https://zhuanlan.zhihu.com/p/686198447)|@reed|⭐️⭐️⭐️|
| [[GPU指令集架构][精解]📖NVidia GPU指令集架构-寄存器](https://zhuanlan.zhihu.com/p/688616037)|@reed|⭐️⭐️⭐️|
| [[GPU指令集架构][精解]📖NVidia GPU指令集架构-Load和Cache](https://zhuanlan.zhihu.com/p/692445145)|@reed|⭐️⭐️⭐️|
| [[GPU指令集架构][精解]📖NVidia GPU指令集架构-浮点运算](https://zhuanlan.zhihu.com/p/695667044)|@reed|⭐️⭐️⭐️|
| [[GPU指令集架构][精解]📖NVidia GPU指令集架构-整数运算](https://zhuanlan.zhihu.com/p/700921948)|@reed|⭐️⭐️⭐️|
| [[GPU指令集架构][精解]📖NVidia GPU指令集架构-比特和逻辑操作](https://zhuanlan.zhihu.com/p/712356884)|@reed|⭐️⭐️⭐️|
| [[GPU指令集架构][精解]📖NVidia GPU指令集架构-Warp级和Uniform操作](https://zhuanlan.zhihu.com/p/712357647)|@reed|⭐️⭐️⭐️|
| [[CUDA优化][入门]📖CUDA 入门的正确姿势：how-to-optimize-gemm](https://zhuanlan.zhihu.com/p/478846788)|@白牛|⭐️⭐️⭐️|
| [[CUDA优化][入门]📖CUDA（一）：CUDA 编程基础](https://zhuanlan.zhihu.com/p/645330027)|@紫气东来|⭐️⭐️⭐️|
| [[CUDA优化][入门]📖CUDA（二）：GPU的内存体系及其优化指南](https://zhuanlan.zhihu.com/p/654027980)|@紫气东来|⭐️⭐️⭐️|
| [[CUDA优化][实践]📖CUDA（三）：通用矩阵乘法：从入门到熟练](https://zhuanlan.zhihu.com/p/657632577)|@紫气东来|⭐️⭐️⭐️|
| [[CUDA优化][实践]📖ops(1)：LayerNorm 算子的 CUDA 实现与优化](https://zhuanlan.zhihu.com/p/694974164)|@紫气东来|⭐️⭐️⭐️|
| [[CUDA优化][实践]📖ops(2)：SoftMax算子的 CUDA 实现](https://zhuanlan.zhihu.com/p/695307283)|@紫气东来|⭐️⭐️⭐️|
| [[CUDA优化][实践]📖ops(3)：Cross Entropy 的 CUDA 实现](https://zhuanlan.zhihu.com/p/695594396)|@紫气东来|⭐️⭐️⭐️|
| [[CUDA优化][实践]📖ops(4)：AdamW 优化器的 CUDA 实现](https://zhuanlan.zhihu.com/p/695611950)|@紫气东来|⭐️⭐️⭐️|
| [[CUDA优化][实践]📖ops(5)：激活函数与残差连接的 CUDA 实现](https://zhuanlan.zhihu.com/p/695703671)|@紫气东来|⭐️⭐️⭐️|
| [[CUDA优化][实践]📖ops(6)：embedding 层与 LM head 层的 CUDA 实现](https://zhuanlan.zhihu.com/p/695785781)|@紫气东来|⭐️⭐️⭐️|
| [[CUDA优化][实践]📖ops(7)：self-attention 的 CUDA 实现及优化 (上)](https://zhuanlan.zhihu.com/p/695898274)|@紫气东来|⭐️⭐️⭐️|
| [[CUDA优化][实践]📖ops(8)：self-attention 的 CUDA 实现及优化 (下)](https://zhuanlan.zhihu.com/p/696197013)|@紫气东来|⭐️⭐️⭐️|
| [[CUDA优化][实践]📖CUDA（四）：使用 CUDA 实现 Transformer 结构](https://zhuanlan.zhihu.com/p/694416583)|@紫气东来|⭐️⭐️⭐️|
| [[CUDA优化][Copy]📖Async Copy及Memory Barrier指令的功能与实现](https://zhuanlan.zhihu.com/p/685168850)|@Frank Wang|⭐️⭐️⭐️|
| [[CUDA优化][GEMV]📖深入浅出GPU优化系列：gemv优化](https://zhuanlan.zhihu.com/p/494144694)|@有了琦琦的棍子|⭐️⭐️⭐️|
| [[CUDA优化][实践]📖CUDA element-wise 算子详解](https://zhuanlan.zhihu.com/p/1888630735520391519)|@懒蚂蚁呀不嘿|⭐️⭐️⭐️|
| [[CUDA优化][实践]📖CUDA transpose 算子详解](https://zhuanlan.zhihu.com/p/1899760505733756129)|@懒蚂蚁呀不嘿|⭐️⭐️⭐️|
| [[CUDA优化][实践]📖CUDA reduce 算子详解](https://zhuanlan.zhihu.com/p/1905661893739283464)|@懒蚂蚁呀不嘿|⭐️⭐️⭐️|
| [[CUDA优化][实践]📖CUDA GEMM 算子详解](https://zhuanlan.zhihu.com/p/1910636263666610461)|@懒蚂蚁呀不嘿|⭐️⭐️⭐️|
| [[Tensor Cores]📖Nvidia Tensor Core初探](https://zhuanlan.zhihu.com/p/620185229)|@木子知|⭐️⭐️⭐️|
| [[Tensor Cores]📖Nvidia Tensor Core-WMMA API编程入门](https://zhuanlan.zhihu.com/p/620766588)|@木子知|⭐️⭐️⭐️|
| [[Tensor Cores]📖Nvidia Tensor Core-MMA PTX编程入门](https://zhuanlan.zhihu.com/p/621855199)|@木子知|⭐️⭐️⭐️|
| [[Tensor Cores]📖CUDA Ampere Tensor Core HGEMM 矩阵乘法优化](https://zhuanlan.zhihu.com/p/555339335)|@nicholaswilde|⭐️⭐️⭐️|
| [[GPU通信架构][精解]📖NVIDIA GPGPU（四）- 通信架构](https://zhuanlan.zhihu.com/p/680262016)|@Bruce|⭐️⭐️⭐️|
| [[torch.compile][原理]📖Torch.compile流程解析: 介绍](https://zhuanlan.zhihu.com/p/9418379234)|@StarCap|⭐️⭐️⭐️|
| [[torch.compile][原理]📖Torch.compile流程解析: TorchDynamo](https://zhuanlan.zhihu.com/p/9640728231)|@StarCap|⭐️⭐️⭐️|
| [[torch.compile][原理]📖Torch.compile流程解析: AOTAutograd](https://zhuanlan.zhihu.com/p/9997263922)|@StarCap|⭐️⭐️⭐️|
| [[torch.compile][原理]📖Torch.compile流程解析: TorchInductor](https://zhuanlan.zhihu.com/p/11224299472)|@StarCap|⭐️⭐️⭐️|
| [[torch.compile][原理]📖Torch.compile流程解析: 算子融合](https://zhuanlan.zhihu.com/p/21053905491)|@StarCap|⭐️⭐️⭐️|
| [[torch.compile][实践]📖Torch.compile使用指南](https://zhuanlan.zhihu.com/p/620163218)|@jhang|⭐️⭐️⭐️|
| [[torch.compile][实践]📖Torch.compile详细示例解析教程](https://zhuanlan.zhihu.com/p/855291863)|@Bbuf|⭐️⭐️⭐️|
| [[torch.compile][原理]📖一文搞懂TorchDynamo原理](https://zhuanlan.zhihu.com/p/630933479)|@吾乃阿尔法|⭐️⭐️⭐️|
| [[torch.compile][原理]📖理解torch.compile基本原理和使用方式](https://zhuanlan.zhihu.com/p/12712224407)|@俯仰|⭐️⭐️⭐️|

## ©️License ([©️back👆🏻](#contents))

<div id="License"></div>

GNU General Public License v3.0

## 🎉Contribute ([©️back👆🏻](#contents))

<div id="contribute"></div>

How to contribute? Star this repo or check [🌤🌤CONTRIBUTE🎉🎉](./CONTRIBUTE.md).

<div align='center'>
<a href="https://star-history.com/#xlite-dev/LeetCUDA&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=xlite-dev/LeetCUDA&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=xlite-dev/LeetCUDA&type=Date" />
   <img width=400 height=300 alt="Star History Chart" src="https://api.star-history.com/svg?repos=xlite-dev/LeetCUDA&type=Date" />
 </picture>
</a>
</div>

## 📖 References ([©️back👆🏻](#contents))
<div id="ref"></div>

- [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal)
- [tiny-flash-attention](https://github.com/66RING/tiny-flash-attention)
- [cute-gemm](https://github.com/reed-lau/cute-gemm)
- [cutlass_flash_atten_fp8](https://github.com/weishengying/cutlass_flash_atten_fp8)
- [cuda_learning](https://github.com/ifromeast/cuda_learning)
- [cuda_hgemm](https://github.com/Bruce-Lee-LY/cuda_hgemm)
- [cuda-tensorcore-hgemm](https://github.com/nicolaswilde/cuda-tensorcore-hgemm)
- [How_to_optimize_in_GPU](https://github.com/Liu-xiandong/How_to_optimize_in_GPU/tree/master/sgemv)
- [how-to-optim-algorithm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda)
- [cute_gemm](https://github.com/weishengying/cute_gemm)
- [cutlass](https://github.com/NVIDIA/cutlass)
