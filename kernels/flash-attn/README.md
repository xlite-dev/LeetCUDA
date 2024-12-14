## ⚡️⚡️FlashAttention-2 MMA: Write FlashAttention using Tensor Cores with pure MMA PTX 

|CUDA Cores|Loop over Seqlen/HeadDim |Tile Block (Br, Bc, Bd)|MMA (m16n8k16)|
|:---:|:---:|:---:|:---:|
|✔️|✔️|✔️|✔️|
|Pack LDST (pack 128 bits)|SMEM Padding|Copy Async (cp.async.cg/ca)|Tile MMA (More Threads)
|✔️|✔️|✔️|✔️|
|Tile Warp (More Values)|Multi Stages (1/2)|Collective Store (Warp Shuffle & Reg Reuse)|Split KV/Q|
|✔️|✔️|✔️|✔️|

This repository's implementation of FlashAttention is intended solely for learning CUDA programming. For optimal performance, please use the official [flash-attention](https://github.com/Dao-AILab/flash-attention). Currently, for small-scale attention (SeqLen <= 4096), the flash-attention-mma implemented in this repository matches the performance of the official FA version. However, for large-scale attention computations, there remains a significant performance gap. Performance optimizations are ongoing; stay tuned for updates.

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

## 📖 Performance

urrently, for small-scale attention (SeqLen <= 4096), the flash-attention-mma implemented in this repository matches the performance of the official FA version. However, for large-scale attention computations, there remains a significant performance gap. Performance optimizations are ongoing; stay tuned for updates.

- B=2, H=2, N=4096, D=64
  
```bash
python3 flash_attn_mma.py --B 2 --H 2 --D 64 --N 4096
----------------------------------------------------------------------------------------------------
          B: batch_size, H: n_head, N: seq_len, D: head_dim, seed: 8942, Warmup: 2, Iters: 10
----------------------------------------------------------------------------------------------------
                         B=2, H=2, N=4096, D=64, Warmup: 2, Iters: 10
      naive(unfused): ['-0.03945923 ', '0.01776123  ', '0.02627563  '], time:1.318264ms
          mma(naive): ['-0.03945923 ', '0.01774597  ', '0.02626038  '], time:9.853077ms
         mma(stage1): ['-0.03945923 ', '0.01776123  ', '0.02624512  '], time:0.336719ms
         mma(stage2): ['-0.03945923 ', '0.01776123  ', '0.02624512  '], time:0.304818ms
             (flash): ['-0.03945923 ', '0.01776123  ', '0.02626038  '], time:0.328016ms
----------------------------------------------------------------------------------------------------
```

- B=2, H=2, N=4096, D=128
  
```bash
python3 flash_attn_mma.py --B 2 --H 2 --D 128 --N 4096
----------------------------------------------------------------------------------------------------
          B: batch_size, H: n_head, N: seq_len, D: head_dim, seed: 2806, Warmup: 2, Iters: 10
----------------------------------------------------------------------------------------------------
                         B=2, H=2, N=4096, D=128, Warmup: 2, Iters: 10
      naive(unfused): ['0.00286484  ', '-0.00598907 ', '-0.02156067 '], time:1.377940ms
          mma(naive): ['0.00284004  ', '-0.00598526 ', '-0.02157593 '], time:19.166064ms
         mma(stage1): ['0.00284004  ', '-0.00598526 ', '-0.02156067 '], time:0.678110ms
         mma(stage2): ['0.00284004  ', '-0.00598526 ', '-0.02156067 '], time:0.659609ms
             (flash): ['0.0028553   ', '-0.00598145 ', '-0.02156067 '], time:0.548506ms
----------------------------------------------------------------------------------------------------
```

- B=2, H=2, N=1024, D=128
  
```bash
python3 flash_attn_mma.py --B 2 --H 2 --D 128 --N 1024
----------------------------------------------------------------------------------------------------
          B: batch_size, H: n_head, N: seq_len, D: head_dim, seed: 4166, Warmup: 2, Iters: 10
----------------------------------------------------------------------------------------------------
                         B=2, H=2, N=1024, D=128, Warmup: 2, Iters: 10
      naive(unfused): ['-0.02110291 ', '0.04946899  ', '-0.04928589 '], time:0.145769ms
          mma(naive): ['-0.02116394 ', '0.04946899  ', '-0.04946899 '], time:1.236653ms
         mma(stage1): ['-0.02114868 ', '0.04943848  ', '-0.04943848 '], time:0.070930ms
         mma(stage2): ['-0.02114868 ', '0.04943848  ', '-0.04943848 '], time:0.069165ms
             (flash): ['-0.02113342 ', '0.04949951  ', '-0.04931641 '], time:0.151205ms
----------------------------------------------------------------------------------------------------
```

- B=2, H=2, N=8192, D=64
```bash
python3 flash_attn_mma.py --B 2 --H 2 --D 64 --N 8192
----------------------------------------------------------------------------------------------------
          B: batch_size, H: n_head, N: seq_len, D: head_dim, seed: 434, Warmup: 2, Iters: 10
----------------------------------------------------------------------------------------------------
                         B=2, H=2, N=8192, D=64, Warmup: 2, Iters: 10
      naive(unfused): ['-0.00259781 ', '-0.00584412 ', '-0.00161552 '], time:5.139947ms
          mma(naive): ['-0.00258827 ', '-0.00583267 ', '-0.00162792 '], time:39.265347ms
         mma(stage1): ['-0.00261307 ', '-0.00583267 ', '-0.00162888 '], time:1.131415ms
         mma(stage2): ['-0.00261307 ', '-0.00583267 ', '-0.00162888 '], time:1.082253ms
             (flash): ['-0.00259209 ', '-0.00584793 ', '-0.00160122 '], time:0.786042ms
----------------------------------------------------------------------------------------------------
```

## 📖 More tests   
```bash
# Volta, Ampere, Ada, Hopper, ...
pip install flash-attn
export TORCH_CUDA_ARCH_LIST=Ada 
python3 flash_attn_mma.py
```

- NVIDIA L20
```bash
python3 flash_attn_mma.py --N 4096 --B 2 --H 2 --D 128
----------------------------------------------------------------------------------------------------
          B: batch_size, H: n_head, N: seq_len, D: head_dim, seed: 762, Warmup: 2, Iters: 10
----------------------------------------------------------------------------------------------------
                         B=2, H=2, N=4096, D=128, Warmup: 2, Iters: 10
      naive(unfused): ['0.06402588  ', '0.01030731  ', '0.02693176  '], time:1.380467ms
          mma(naive): ['0.06408691  ', '0.01036835  ', '0.0269165   '], time:19.160128ms
         mma(stage1): ['0.06390381  ', '0.01038361  ', '0.02685547  '], time:0.681663ms
         mma(stage2): ['0.06390381  ', '0.01038361  ', '0.02685547  '], time:0.661945ms
             (flash): ['0.06402588  ', '0.01029968  ', '0.02694702  '], time:0.550222ms
----------------------------------------------------------------------------------------------------
```
