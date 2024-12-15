## ⚡️⚡️FlashAttention-2 MMA: Write FlashAttention using Tensor Cores with pure MMA PTX 

![flash-attn-mma](https://github.com/user-attachments/assets/6f66796d-44d5-4ec1-b224-af997bd152b2)

|CUDA Cores|Loop over Seqlen/HeadDim |Tile Block (Br, Bc, Bd)|MMA (m16n8k16)|
|:---:|:---:|:---:|:---:|
|✔️|✔️|✔️|✔️|
|Pack LDST (pack 128 bits)|SMEM Padding|Copy Async (cp.async.cg/ca)|Tile MMA (More Threads)
|✔️|✔️|✔️|✔️|
|Tile Warp (More Values)|Multi Stages (1/2)|Collective Store (Warp Shuffle & Reg Reuse)|**Split KV/Q**|
|✔️|✔️|✔️|✔️|

This repository's implementation of FlashAttention is intended solely for learning CUDA programming. For optimal performance, please use the official [flash-attention](https://github.com/Dao-AILab/flash-attention). Currently, for small-scale attention (SeqLen <= 4096), the flash-attention-mma implemented in this repository matches the performance of the official FA. However, for large-scale attention computations, there remains a significant performance gap. Performance optimizations are ongoing; stay tuned for updates.

## 📖 FlashAttetion MMA Kernels

The `Split KV` and `Split Q` implementations have been carried out in [flash-attention-mma⚡️⚡️](.) for performance comparison. The `Split KV` method, which involves splitting all QKV across MMA (Warps) using a naive matmul (MMA) and Warp tiling policy, is slower compared to the `Split Q` policy, which splitting Q across MMA(Warps) and keep access KV for all MMA(Warps).

- Split KV (Basic, FlashAttention-1)

```C++
// Split QKV across MMA(Warps) using naive matmul MMA&Warp tiling policy.
// case: The layout of 8 MMA(2x4)  [after] kWarpTileSeqLenQxkWarpTileSeqLenK(2x2) -> 32x2,32x2=64x64: 
// |  [64,64]  |    warp_KV 0    |    warp_KV 1    |    warp_KV 2    |    warp_KV 3    |
// | warp_QP 0 |-- MMA 0,MMA 0 --|-- MMA 2,MMA 2 --|-- MMA 4,MMA 4 --|-- MMA 6,MMA 6 --|
// | warp_QP 0 |-- MMA 0,MMA 0 --|-- MMA 2,MMA 2 --|-- MMA 4,MMA 4 --|-- MMA 6,MMA 6 --|
// | warp_QP 1 |-- MMA 1,MMA 1 --|-- MMA 3,MMA 2 --|-- MMA 5,MMA 5 --|-- MMA 7,MMA 7 --|
// | warp_QP 1 |-- MMA 1,MMA 1 --|-- MMA 3,MMA 2 --|-- MMA 5,MMA 5 --|-- MMA 7,MMA 7 --|
template<
         const int kHeadDim,          // Headdim, 32,64,128     
         const int kMmaAtomM,         // MMA Atom M, 16
         const int kMmaAtomN,         // MMA Atom N, 8
         const int kMmaAtomK,         // MMA Atom K, 16
         const int kMmaTileSeqLenQ,   // 2, more MMA(warp), M=16*2=32, Q@K^T=[Br(M), d(K)]@[d(K),  Bc(N)]  
         const int kMmaTileSeqLenK,   // 4, more MMA(warp), N=8*4= 32, Q@K^T=[Br(M), d(K)]@[d(K),  Bc(N)]    
         const int kMmaTileSeqLenP,   // 2, more MMA(warp), M=16*2=32, P@V  =[Br(M),Bc(K)]@[Bc(K), d(N) ]
         const int kMmaTileHeadDimV,  // 4, more MMA(warp), N=8*4= 32, P@V  =[Br(M),Bc(K)]@[Bc(K), d(N) ]       
         const int kWarpTileSeqLenQ,  // 2, more values, M, Br=32*2=64, matmul M 
         const int kWarpTileSeqLenK,  // 2, more values, N, Bc=32*2=64, matmul N
         const int kWarpTileSeqLenP,  // 2, more values, M, Br=32*2=64, matmul M
         const int kWarpTileHeadDimV, // 2, more values, N, d=32*(1|2|3|4|...)=32|64|96|128|...
         const int kStage,            // only support 1 or 2 now.
         const int kPad               // 0,8              
         >
__global__ void 
flash_attn_mma_stages_split_kv_kernel(half* Q, // [B, H, N, D]
                                      half* K, // [B, H, D, N] K^T transposed 
                                      half* V, // [B, H, N, D] 
                                      half* O, // [B, H, N, D] 
                                      int QKV_seqlen);
```

- Split Q (Faster, FlashAttention-2)

```C++
// Split Q across MMA(Warps) and keep access KV for all MMA(Warps),
// in order to reduce the comm between warps via smem and warp shuffle.
// case: MMA = m16n8k16, Br=16x4=64, Bc=8x8=64, layout: 4 warps
// |   64x64   |      warp_KV 0       |
// | warp_QP 0 | MMA 0 ... MMA 0 (x8) |
// | warp_QP 1 | MMA 1 ... MMA 1 (x8) |
// | warp_QP 2 | MMA 2 ... MMA 2 (x8) |
// | warp_QP 3 | MMA 3 ... MMA 3 (x8) |
template<
         const int kHeadDim,          // Headdim, 32,64,128     
         const int kMmaAtomM,         // MMA Atom M, 16
         const int kMmaAtomN,         // MMA Atom N, 8
         const int kMmaAtomK,         // MMA Atom K, 16
         const int kMmaTileSeqLenQ,   // 4, more MMA(warp), M=16*4=64, Q@K^T=[Br(M), d(K)]@[d(K),  Bc(N)]  
         const int kMmaTileSeqLenK,   // 1, more MMA(warp), N=8*1 =8,  Q@K^T=[Br(M), d(K)]@[d(K),  Bc(N)]    
         const int kMmaTileSeqLenP,   // 4, more MMA(warp), M=16*4=64, P@V  =[Br(M),Bc(K)]@[Bc(K), d(N) ]
         const int kMmaTileHeadDimV,  // 1, more MMA(warp), N=8*1 =8,  P@V  =[Br(M),Bc(K)]@[Bc(K), d(N) ]       
         const int kWarpTileSeqLenQ,  // 1, more values, M, Br=64*1=64, matmul M 
         const int kWarpTileSeqLenK,  // 8, more values, N, Bc=8*8 =64, matmul N
         const int kWarpTileSeqLenP,  // 1, more values, M, Br=64*1=64, matmul M
         const int kWarpTileHeadDimV, // 8, more values, N, d=8*(1|2|3|4|...)=8|...|32|64|96|128|...
         const int kStage,            // only support 1 or 2 now.
         const int kPad               // 0,8           
         >
__global__ void
flash_attn_mma_stages_split_q_kernel(half* Q, // [B, H, N, D]
                                     half* K, // [B, H, D, N] K^T transposed 
                                     half* V, // [B, H, N, D] 
                                     half* O, // [B, H, N, D] 
                                     int QKV_seqlen);
```

## 📖 Contents

- [📖 Prerequisites](#prerequisites)
- [📖 Installation](#install)
- [📖 Performance](#perf)
- [📖 Python Testing](#test)

## 📖 Prerequisites
<div id="prerequisites"></div>  

- flash-attention >= 2.6
- PyTorch >= 2.0, CUDA >= 12.0
- Recommended: PyTorch 2.5.1, CUDA 12.5

## 📖 Installation  
<div id="install"></div>    

```bash
pip install flash-attn --no-build-isolation # need offical flash-attention for comparison
```

## 📖 Performance
<div id="perf"></div>  

Currently, for small-scale attention (SeqLen <= 4096), the flash-attention-mma implemented in this repository matches the performance of the official FA version. However, for large-scale attention computations, there remains a significant performance gap. Performance optimizations are ongoing; stay tuned for updates.

## 📖 Python Testing  
<div id="test"></div>  

```bash
# Volta, Ampere, Ada, Hopper, ...
python3 -m pip install flash-attn --no-build-isolation
export TORCH_CUDA_ARCH_LIST=Ada 
python3 flash_attn_mma.py --D 64 # test all default settings for D=64
```

- B=2, H=2, N=4096, D=64
  
```bash
python3 flash_attn_mma.py --B 2 --H 2 --D 64 --N 4096
```

- B=2, H=2, N=4096, D=128
  
```bash
python3 flash_attn_mma.py --B 2 --H 2 --D 128 --N 4096
```

- B=2, H=2, N=1024, D=128
  
```bash
python3 flash_attn_mma.py --B 2 --H 2 --D 128 --N 1024
```

- B=2, H=2, N=8192, D=64
```bash
python3 flash_attn_mma.py --B 2 --H 2 --D 64 --N 8192
```
