# 全书记号表（RFC-A.1 定稿冻结，2026-09-11）

> 规范：正文公式符号遵循本表；源码变量映射列已经 grep 实测核对（行号为冻结版 anchors.yaml 基线）。
> 修订须走 BOOK_PLAN 显式修订流程，并同步全书 grep 别名（RFC-J.6）。

## GEMM 维度与分块

| 记号 | 含义 | 源码对应（实测） |
|---|---|---|
| $M,N,K$ | GEMM 行/列/归约维 | kernel 参数 `M,N,K`；bench `--mnk` |
| $B_M,B_N,B_K$ | block tile 尺寸 | `BM/BN/BK`：sgemm.cuh L34-36（naive 32）、L109-111（vec4 128）、L234-236（WMMA 128）；hgemm.cuh L137-139、L442-444；CuTe 模板参数 hgemm.cuh L807 |
| $W_M,W_N$ | warp tile 尺寸 | `kMmaTileM/kMmaTileN`（hgemm.cuh L123-124、L427-428，warp 排布数×mma 尺寸）；WMMA 版 `WARP_TILE_M/WARP_TILE_N`（sgemm.cuh L234-235） |
| $T_M,T_N$ | thread tile（寄存器 tile） | `kValTileM/kValTileN`（hgemm.cuh） |
| $S$ | 流水级数（kStages） | `kStage`（hgemm.cuh L808）、`KStage`（CuTe，hgemm.cuh L1235）、`kStagesQK/kStagesV`（ffpa_attn.cuh L80） |

## Attention 记号

| 记号 | 含义 | 源码对应（实测） |
|---|---|---|
| $B,H,N,D$ | batch / heads / seqlen / head dim | bench `--bhnd`；`kHeadDim`（模板参数，flash_attn.cuh、ffpa_attn.cuh） |
| $B_r,B_c$ | Q block / KV block 行数 | 手写 FA：`kMmaTileSeqLenQ/kMmaTileSeqLenK`（flash_attn.cuh L124 附近）；CuTe FA/FFPA：`kBr/kBc`（flash_attn.cuh L2544-2545、ffpa_attn.cuh L93-94、L395-396） |
| $m,\ell$ | online softmax running max / sum | `block_row_max/block_row_sum`（base.cuh softmax、flash_attn.cuh；更新中间量 `block_row_max_new/block_row_sum_new`） |
| $\mathrm{LSE}$ | log-sum-exp | `merge_attn_states`（base.cuh L437）的合并输入/输出 |
| $O,P,S$ | 输出 / 概率 / 打分矩阵 | `O/P/S`（flash_attn/ffpa kernel 参数；fragment `tCrO` 等） |
| $d_c$ | D-chunk（Split-D 64 宽块） | ffpa split-D 的 head dim 切分；`kHeadDim` 模板参数按 64 分块 |

## 布局与代数（CuTe，colex 序）

| 记号 | 含义 | 源码对应 |
|---|---|---|
| $s_i,c_i$ | 第 $i$ mode 的 stride / coord | `Shape/Stride`、`make_shape/make_stride`、`make_coord` |
| $\mathrm{idx}$ | 线性化 offset：$\mathrm{idx}=\sum_i s_i c_i$ | `layout(coord)`、`cosize` |
| $L_A\circ L_B$ | layout compose | `composition/compose`（hgemm.cuh L1257） |
| $L^{-1}$ | layout inverse | `colex` 逆映射；`LayoutLeft/LayoutRight` |
| $\mathrm{Swizzle}\langle B,M,S\rangle$ | 位异或重排 | `Swizzle<B,M,S>`、`SwizzleBMS`（common.cuh 手写版 L110-348） |

## 执行配置

| 记号 | 含义 | 源码对应 |
|---|---|---|
| — | block 线程数 | `kNumThreads`（base.cuh L445、flash_attn.cuh L124） |
| — | warp 大小 32 | `kWarpSize`（common.cuh） |
| — | lane / warp / block id | `lane_id/warp_id`、`threadIdx/blockIdx` |
