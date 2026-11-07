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

## SVDQuant（W4A4）记号（2026-11-07 增补，cache-dit@3295dc6 基线）

> 术语约定：第 28 章的 smooth-K/V 是 softmax 不变性平移；本部分"平滑/通道缩放平滑"
> 一律指 SmoothQuant 意义的逐输入通道对角缩放 $\widehat X=XS^{-1}$、$\widehat W=WS$。
> $L_1$/$L_2$ 按本书 $W[N,K]$ 方向定义（$L_1$=up、$L_2$=down），与 SVDQuant 原文
> （$W[m\times n]$、$L_1$=down）角色相反，交叉阅读不可只换形状。

| 记号 | 含义 | 源码对应（cache-dit 3295dc6） |
|---|---|---|
| $X,W,Y$ | 激活 $[M,K]$ / 权重 $[N,K]$ / 输出 $Y=XW^{\top}$（PyTorch 权重方向） | `input`/`wgt`/`out`（gemm_w4a4.h L59-62） |
| $S=\mathrm{diag}(s)$ | 通道缩放平滑因子（离线校准），迁移强度 $\alpha$ | `smooth`（ producer 参数）；`smooth_factor` |
| $\widehat X,\widehat W$ | 平滑后激活/权重，$\widehat X\widehat W^{\top}=XW^{\top}$ 精确成立 | `quant[n]`（除 smooth 后 fragment） |
| $L_1,L_2,R$ | 低秩因子（$L_1\in\mathbb{R}^{N\times r}$ up、$L_2\in\mathbb{R}^{r\times K}$ down）与残差 $R=\widehat W-L_1L_2$ | `proj_up`/`proj_down`（离线）；`lora_up`/`lora_down`（kernel） |
| $r$ / rank | 低秩 rank（kernel 按 16 分档，0--128） | `rank`；模板 `kRankTiles*16` |
| $\widetilde L_2=L_2S^{-1}$ | 预乘 $S^{-1}$ 的 down 因子（低秩分支直接消费原始 $X$） | packing 的 `lora_down /= smooth` |
| $D$ | 低秩 down 投影输出 $D=\widehat XL_2^{\top}=X\widetilde L_2^{\top}$，$[M,r]$ | `lora_act`（producer 输出）/`lora_act_in`（GEMM 输入） |
| SFA / SFB | 激活/权重侧 per-16 元素 FP8_E4M3（ue4m3）scale | `ascales` / `wscales` |
| v1 pack | 256 行 MMA-fragment 交织的激活/scale pack（G2R 直读） | `qact`/`ascales`（quantize_w4a4.cu L209-220） |
| `kNative`/`kRowMajor` | $D$ 到达 GEMM 的两种 layout：FP32 fragment+gather / BF16 行主序+TMA | `enum class LoraLayout`（gemm_w4a4.h L14-17） |
| $\alpha$、wcscales | 残差积的标量/逐输出列权重 scale，**只乘 NVFP4 主积** | `alpha`/`wcscales`（gemm_w4a4.cu） |
| $s=\alpha\cdot\mathrm{wcscales}$ | LoRA-first 代数补偿的列 scale（先除后乘） | `apply_inverse_scale`/`apply_residual_scale` |
| `lora_scales` | per-16-rank 折叠系数（`kNative` gather 在线施加；`kRowMajor` 已由 producer 折入） | `F32DLoraScales.values[8]` |
