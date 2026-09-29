<!--
author: reed
author_id: reed-84-49
url: https://zhuanlan.zhihu.com/p/2038747362596737621
column: 
published: 2026-05-15
fetched: 2026-09-29
images: 4
-->

# FP8 Attention 中的精度优化：逆序计算与缩放因子选择

LLM 模型规模的持续增长，低精度计算逐渐成为提升训练和推理吞吐的核心手段。Hopper 架构在 Tensor Core 上首次提供了对 FP8（包括 E4M3 与 E5M2）的原生支持，FlashAttention-3 在此基础上把 attention 算子的中间结果也整体下放到 FP8。但 FP8 与 BF16 / FP16 在数值表示上的差距是数量级的，E4M3 仅用 3 比特描述尾数，相对精度只有 $2^{-3} = 12.5\%$，比 FP16 差了一到两个数量级。FP16 / BF16 时代被认为无关紧要的实现细节，例如 KV block 的迭代方向、softmax 之后 P 的量化缩放因子取值，到了 FP8 下都会被显著放大并影响最终精度。

本文围绕 Hopper 上的主流 FP8 FlashAttention 布局（O 累加器全程 FP32、P 显式 cast 为 E4M3）讨论这两个问题，给出对应的工程修法：KV block 逆向迭代与$S = 256 = 2^8$ 的静态量化 scale。文章结构上，首先回顾 FP8 E4M3 的数值结构以及 Attention Sink 在正向迭代中如何使 P 矩阵塌陷到 subnormal 区域之下，从机理上引出逆向计算策略；然后从 IEEE 754 浮点数学与 E4M3 数轴几何两个角度证立 $S = 256$ 是最优解；最后对照主流 FP8 attention 实现的具体选择，并给出对照实验。

需要事先指出的是，Attention Sink 也有训练侧的解决路径，例如 learnable sink token、clipped softmax 等，本文走的是另一条正交方向，即在 sink 已经存在的预训练模型上从 kernel 层规避 FP8 量化引入的精度退化，不需要重新训练。

## Attention Sink 与 P 的精度退化

Self-Attention 是 Transformer 的核心算子，

$$O = \mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V.$$

实际实现中 $K$、$V$ 沿序列维度切分为若干 block，通过 Online Softmax 维护 running max 和 running sum，逐 block 迭代完成精确归一化。Hopper 上 FP8 FlashAttention 的精度策略是 $Q$、$K$ 在第一次矩阵乘前已是 FP8，矩阵乘累加器与输出 $O$ 全程保持 FP32，softmax 之后的 $P$ 是 FP32，在送入 $P \cdot V$ 之前需要显式 cast 为 FP8 E4M3。$P$ 是链路中真正发生 FP8 量化的张量，本文讨论的精度问题都围绕它展开。

### FP8 E4M3 的精度结构

E4M3 的位宽布局为 1 位符号、4 位指数、3 位尾数；最大正值 $448 = 7 \times 2^6$，最小正规格化数 $2^{-6}$，最小正非规格化数 $2^{-9} \approx 1.95 \times 10^{-3}$。

![img-1](https://pic3.zhimg.com/v2-f9d39a7134f05c812e0d39075a62a602_r.jpg)
（图注：Figure 1. FP8 E4M3 Bit Fields Layout）

3 位尾数意味着任意一段 binade $[2^n, 2^{n 1})$ 内只有 $2^3 = 8$ 个均匀分布的可表示值，相对精度 $2^{-3} = 12.5\%$。E4M3 的字段定义如图 1 所示，正半轴可表示值的分布如图 2 所示。

![img-2](https://pic2.zhimg.com/v2-6e977f3f7eb65f6d5fc0f00dd0df9721_r.jpg)
（图注：Figure 2. FP8 E4M3 Representable Values (Positive Axis)）

E4M3 的可表示值绝对间距（LSB）随 binade 上行成倍变大，到 $[256, 512)$ 这一段 LSB 已经达到 32。一个连续实数被量化到 FP8 时，其量化误差直接由它落入的 binade 决定。下表给出后文几何分析会反复用到的几个关键 binade。

| binade | LSB | 半 LSB（最坏量化误差） |
| --- | --- | --- |
| [2^-9, 2^-6)（subnormal） | 2^-9 | 2^-10 |
| [64, 128) | 8 | 4 |
| [128, 256) | 16 | 8 |
| [256, 512)（含 max_normal = 448） | 32 | 16 |

### Attention Sink 与 P 塌陷

Attention Sink 是经过充分训练的 Transformer 中一个被广泛观察到的现象，序列开头的若干 token 会获得异常高的注意力权重，对应到 logit 矩阵 $S = QK^T / \sqrt{d_k}$ 上则表现为序列首部若干列的 score 显著大于其他位置。多项实证研究表明，主流预训练 LLM 中的 sink 强度 $\Delta = S_{\text{sink}} - S_{\text{normal}}$ 在 context 长度数千量级时大致落在 $\Delta \in [6, 13]$ 区间内。

下面分析 Online Softmax 在正向迭代时如何与 sink 相互作用。每处理一个新的 KV block 走五步：（i）算局部 score $S_{\text{local}}$；（ii）更新全局最大值 $m_{\text{new}} = \max(m_{\text{old}}, m_{\text{local}})$；（iii）算修正因子 $\alpha = \exp(m_{\text{old}} - m_{\text{new}})$；（iv）算局部概率 $P_{\text{local}} = \exp(S_{\text{local}} - m_{\text{new}})$ 并 cast 为 FP8 E4M3；（v）更新 FP32 累加器 $O_{\text{new}} = \alpha \cdot O_{\text{old}}   P_{\text{local}}^{\text{fp8}} \cdot V_{\text{local}}^{\text{fp8}}$。

在标准的正向迭代中，sink 位于 $\text{Block}_0$，第一步就把 $m_{\text{global}}$ 推到 $\Delta$ 量级。后续正常 block 的局部 score 大致分布在 $[-1, 1]$，因此从第二步开始 $P_{\text{local}} \sim \exp(-\Delta)$。只要 $\Delta \gtrsim 9$，$\exp(-\Delta) \lesssim 10^{-4}$ 已经低于 E4M3 subnormal 下界 $2 \times 10^{-3}$，P 中所有非 sink 位置的元素在 cast 时被舍入为零，最终 $P^{\text{fp8}} \cdot V$ 的结果只剩 sink token 一列的贡献，正常 token 的注意力信息在 cast 这一步整体丢失。

值得指出的是，这一精度退化的根源不在累加器（O 全程在 FP32 上累积），而在 P 这张需要 cast 到 FP8 的中间张量上；同时 P 塌陷是 sink 强度的临界现象，弱 sink 下 cast 仍能保留相对结构，只有当 sink 强到把 P 推到 subnormal 之下才出现整列归零的灾难。本文的实验在 $\Delta_{\text{sink}} = 12$ 下进行验证，对应主流预训练模型中较为典型的 sink 量级。

## 逆序计算策略

针对正向迭代下由 Attention Sink 引发的 P 塌陷，一个直接而有效的解决方案是将分块迭代顺序反转：$\text{Block}_n \to \text{Block}_{n-1} \to \cdots \to \text{Block}_0$。Online Softmax 在结合律下与迭代顺序无关，无限精度下结果严格等价，逆序不改变算法的数学正确性，差异只体现在有限精度下舍入误差的累积方式。

逆序对 P 塌陷的规避机理如下。逆序迭代从 $\text{Block}_n$ 开始，由于序列中绝大多数 token 的 attention score 处于一个相对均匀且温和的范围内，前 $n$ 步迭代过程中 $m_{\text{global}}$ 始终维持在一个较小的水平（如 $m_{\text{global}} \approx 0.6$），对应的 $P_{\text{local}}$ 元素分布在 $[\exp(-2), 1] \approx [0.14, 1]$ 区间内，完全落在 E4M3 normal 区域，cast 误差只来自正常的 round-to-nearest，没有 subnormal 截断。当迭代行进到最后一步遇到 sink 时，$m_{\text{global}}$ 才发生一次大跳变，此时之前累积的 $O_{\text{old}}$ 需要乘以一个极小的修正因子 $\alpha$，但 O 全程在 FP32 上累积，FP32 的 23 位尾数足以保留 $O_{\text{old}}$ 各分量之间的相对结构；同时这一步的 P 来自 sink block 自身，元素值整体接近 1，cast 到 E4M3 后保持高精度。逆序将 P 塌陷到 subnormal 之下这一灾难性事件完全规避，将精度退化压缩到最后一步可控的 $\alpha \cdot O_{\text{old}}$ 缩放上。

具体到实现，逆序计算的修改非常简洁，仅需将 KV block 的迭代循环方向反转：

```
# 标准 FlashAttention (Forward Order)
for j in range(0, num_kv_blocks):           # 0, 1, 2, ..., n
    # online softmax + P cast to FP8 + P_fp8 @ V

# 逆序 FlashAttention (Reverse Order)
for j in range(num_kv_blocks - 1, -1, -1):  # n, n-1, ..., 1, 0
    # 算法逻辑完全一致
```

需要指出的是，FlashAttention 系列（FA2 / FA3 / FA4）的 KV 主循环本身就采用逆序方向，但其源码注释给出的动机是 mask phase 拆分与寄存器节省，与精度无关。本节给这条已经存在的工程默认补上一条独立的精度依据，使它可以被单独地施加到 hpc-ops、TensorRT-LLM XQA 等仍走正向 K-loop 的实现上。

## 缩放因子 256 的选取

除迭代顺序外，FP8 Attention 中另一个直接影响精度的实现选择是 P 在 cast 之前的静态缩放因子 $S$。完整的计算路径为

$$O = \frac{\big(P \cdot S\big)_{\text{fp8}} \cdot V_{\text{fp8}}}{S}.$$

$S$ 的选择决定整条链路的精度上限。本节从 IEEE 754 浮点数学与 E4M3 数轴几何两个角度证立 $S = 256$ 是最优解，并就实现层面对前两条的具体落地做一点澄清。

### 二次幂缩放使乘除运算 bit-exact

在 IEEE 754 浮点数体系中，乘以或除以 $2^k$ 是精确操作，其只改变 exponent 字段、不动 mantissa 字段，不引入任何舍入误差。$S = 2^k$ 时整条链路上的 $P \cdot S$ 与 $O / S$ 都是 bit-exact 的，整条公式中只有 P 到 FP8 的 cast 这一步引入误差，其余所有缩放与反缩放都位精确。当 $S$ 取非二次幂（如 250、300、448）时则不然，每一次 $P \cdot S$ 与 $O / S$ 都是真正的 FP32 浮点乘除，每次额外引入约 $2^{-23}$ 量级的尾数舍入误差。值得注意的是，E4M3 的最大正值 $448 = 7 \times 2^6$ 并不是二次幂，主流实现采用的 amax/448 类取值不享受这条性质。

### dp(S) 锯齿波在二次幂中锁定 S = 256

仅满足二次幂这一条件还不能唯一锁定最优 $S$，需要再引入一层”最大表示误差”分析。对 P 中任意 $p \in [0, 1]$，乘 $S$ 之后落在 $[0, S]$ 内某条 binade 上，cast 到 FP8 时被舍入到该 binade 的可表示值，反映回原域的最坏量化误差为

$$dp(S) = \frac{\max\limits_{x \in [0,\, \min(S,\, 448)]} \mathrm{LSB}_{\mathrm{E4M3}}(x)}{S},$$

即 $[0, S]$ 内最稀疏 binade 的 LSB 除以 $S$。它给出 P 中任一元素在 cast 后能保证的最坏精度。把 $dp(S)$ 在 $S \in [2, 2048]$ 上画出来，便得到图 3。

![img-3](https://pic4.zhimg.com/v2-fd76a9b0075ae26004ae59b24d067233_r.jpg)
（图注：Figure 3. FP8 Quantization Error Analysis）

曲线的形状揭示了 $S$ 选取的全部几何结构：（i）每个 binade $[2^k, 2^{k 1})$ 内 $dp(S) = 2^{k-3} / S$ 单调下降，跨过二次幂边界后跳变并重新下降，形成锯齿波；（ii）每个二次幂边界 $S = 2^k$ 处 $dp(2^k) = 2^{-4} \approx 0.0625$，对所有 $k$ 都是同一个值，构成下包络，任意非二次幂 $S$ 的 $dp(S)$ 严格大于 $2^{-4}$、最坏可达 $2^{-3}$（一倍差距）；（iii）$S > 448$ 时 amax 越过 max_normal 被 clamp，$dp(S) = 1 - 448 / S$ 随 $S$ 上升迅速劣化。

由此从图 3 可以读出 $S = 256$ 的最优性：在 $S \le 448$ 不溢出约束下，二次幂候选 $\{2, 4, \ldots, 128, 256\}$ 都给出最低的 $dp = 2^{-4}$，落在下包络上；主流实现采用的 $S = 448$ 落在 binade $[256, 512)$ 内部，$dp = 32 / 448 \approx 0.0714$，比 $S = 256$ 高约 14%。落在下包络上的二次幂候选不止一个，最大的那个 $k$ 还附带一个额外好处：能落入 E4M3 normal 区的 P 小值范围最广。E4M3 normal 区下界为 $2^{-6}$，反映回原域为 $2^{-6-k}$，$S$ 越大该阈值越低。$S = 256$ 把该下界推到 $2^{-14} \approx 6.1 \times 10^{-5}$，比 $S = 128$ 的 $1.22 \times 10^{-4}$ 低一倍。

S=256可以最小化最大误差，S=448可以最小化subnormal，优先级上”最大误差”在前、”保护 subnormal”在后：因为 $O = P \cdot V$ 的求和结构使 amax 量级元素的误差以高权重传递到最终输出，而 subnormal 量级元素自身数值小、对 $O$ 的贡献本就被压低。 据此先用 $dp(S)$   下包络这条硬条件淘汰所有非二次幂候选（包括 $S = 448$），再在通过的二次幂候选中取最大 $k$，得到 $S = 256$ 这一同时兼顾两项目标的唯一最优解。

### 实现层补充：精度优势就是工程优势的全部

第三条仅作为对实现层的澄清。Hopper SASS 没有直接计算 $e^x$ 的指令，只有计算 $2^x$ 的多功能单元指令 MUFU.EX2，因此主流 FlashAttention 实现把 $\log_2 e$ 折进 softmax_scale、用 $\exp_2$ 计算 softmax；P 在 cast 之前要乘上 $S$，落在实现上就是给 MUFU.EX2 的输入项再附加一个常数 $\log_2 S$，finalize 阶段则乘上 $1/S$。

需要说明的是，GPU 上浮点乘法（包括 FFMA）本身已经非常高效，乘以 $256$ 与乘以 $1/256$ 在指令开销上和乘以任意其他浮点常数没有差别——不存在”$2^k$ 走整数 bit-shift 比浮点除法快”这类硬件 trick。$S$ 取 $2^k$ 与取 $448$ 在性能上等价，唯一的差别就是上面理由一里讲过的那一条：当 $S = 2^k$ 时 $S$ 和 $1/S$ 都是 IEEE 754 下精确可表示的浮点常数、且对 finite 数做乘法是 exponent-only 操作，不引入任何舍入；而 $1/448$ 不是 IEEE 754 精确可表示的浮点数，对应的乘法每次都引入约 $2^{-23}$ 量级的 round-to-nearest 舍入。也就是说，$S = 2^k$ 在实现层面相对非二次幂的全部优势就是精度上的优势，不存在额外的指令级好处。

将以上几条性质合在一起，$S = 256$ 是同时满足下列三条条件的唯一取值：（i）二次幂使 $\times S$ 与 $\times (1/S)$ 都是 IEEE 754 下精确可表示的常数、对 finite 数做浮点乘法是 exponent-only 操作不引入任何舍入；（ii）落在 $dp(S)$ 锯齿波下包络 $2^{-4}$ 上，最大表示误差最小；（iii）在 $S \le 448$ 不溢出约束下取最大值，最大化 normal 区对小值的覆盖。$S = 128$ 浪费一半 normal 区动态范围；$S = 512$ 越过 max_normal 触发 clipping；$S = 448$ 同时违反 (i) 与 (ii)，与 $S = 256$ 在 $dp(S)$ 上有约 14% 的差距，反映到平方阶的输出 MSE 上约为 $(1.14)^2 - 1 \approx 30\%$。

## 主流实现对比与实验验证

### 主流 FP8 Attention 实现的设计选择

下表汇总几个主流 FlashAttention 系列实现及外部 FP8 attention kernel 在 KV 顺序与 P scale 上的具体选择，均从源码中实际读出。

| 实现 | KV 顺序 | P 量化 scale | 2^k | 累加器 |
| --- | --- | --- | --- | --- |
| FA2 / FA3 (BF16) | 逆向 | — (P 不 cast) | — | FP32 |
| FA3 / FA4 (FP8) | 逆向 | S = 256 = 2^8 | 是 | FP32 |
| Tencent hpc-ops | 正向 | S = 1（直接 cast） | 是 | FP32 |
| FlashInfer | 逆向 | S = 448（贴 max_normal） | 否 | FP32 |
| TensorRT-LLM XQA | 正向 | S = 448（贴 max_normal） | 否 | FP32 |
| SageAttention2 | 正向 | S = 448（per-block） | 否 | FP32 |
| SageAttention2++ | 正向 | S = 112（FP16 累加器约束） | 否 | FP16 |
| 本文 | 逆向 | S = 256 | 是 | FP32 |

具体来看：

- FlashAttention 系列。逆向 K-loop 从 FA2 起就已存在，FA3 / FA4 沿用；$S = 2^8$ 的 P-scaling 由 FA3 引入并被 FA4 在 SM100 路径上原样保留。两条选择源码注释给出的动机均与精度论证无关：逆向是为 mask phase 拆分与寄存器节省，$S = 2^8$ 的注释只说”use more of the FP8 range to reduce underflow”，即把 P 从 $[0, 1]$ 抬到 $[0, 256]$ 以减少下溢，但并未对”为何取 $2^8$ 而不是其他 $2^k$“给出几何或精度论证。这两条工程默认在 FP8 + Attention Sink 场景下精度上的好处是本文关注的副作用。

- 直接 cast 路线（hpc-ops，等价于 $S = 1$）。其 prefill kernel 对 softmax 之后的 P 直接 cast 为 E4M3，不引入显式缩放因子；K-loop 正向，P×V 用 FP32 累加器，O 在 epilogue 乘 v-scale 后 cast 为 BF16 写回 HBM。该实现对 sink 非常敏感，sink 让正向第一步把 $m_{\text{global}}$ 钉死后，后续 P 元素整体落到 subnormal 之下被 cast 归零。

- 贴 max_normal 路线（FlashInfer）。softmax 之后对 P 乘 E4M3 max_normal（即 448）后 cast 到 FP8，再在 epilogue 中除回，等价于 $S = 448$；K-loop 逆向，与其 BF16 路径完全同构，纯属沿用 FA3 模板。

- 贴 max_normal 路线（TensorRT-LLM XQA）。NVIDIA TensorRT-LLM 的 generation 阶段使用自研的 XQA kernel，与 FA3 / FlashInfer 完全独立。K-loop 正向，P 在 cast 前乘固定常数 448，后续在 epilogue 反 scale。源码注释中给出 scale 选择的设计哲学：”把 softmax 输出值域 $[0, 1]$ 的上界 1 直接映射到 E4M3 满量程 448”。

- 贴 max_normal 路线（SageAttention2 / 2++）。SageAttention-2对 P 采用 per-block 静态 $S = 448$；SageAttention-2++将 P×V 累加器换成 FP16，由溢出约束 $|32 \cdot P V| \le 65504$ 反推出 $P_r = 112$、$V_r = 4.5$，实质等价于 $S = 112$。

$S = 256$ 与 $S = 448$ 是两个独立的设计选择，对应两条不同的设计逻辑：256 由”二次幂、$dp(S)$ 下包络、不溢出取最大 $k$“三条几何条件唯一锁定；448 由”贴 max_normal、对任意 amax 不溢出”的工程兼容性单条考量得出。

### 实验：两项优化的有效性验证

为定量验证两项优化的具体效用，构造一组对照实验：以 “Forward + $S = 1$” 作为基准（对应 hpc-ops 当前实现），与 “Reverse + $S = 1$“（仅施加优化 1）、”Forward + $S = 448$“（amax/448 路线）、”Forward + $S = 256$“（仅施加优化 2）、”Reverse + $S = 256$“（两项优化叠加）四种配置在同一组合成 attention 输入上对比 MSE。实验在 Hopper FP8 layout 下模拟，注入 $\Delta_{\text{sink}} = 12$ 的 Attention Sink，序列长度从 512 扫到 8192，每个配置重复 5 次取均值。

![img-4](https://pic1.zhimg.com/v2-0e39fb91423d3247198b1eebf71de62a_r.jpg)
（图注：Figure 4. FP8 attention output MSE under Attention Sink）

图 4 给出完整的对照结果，可以观察到四个事实：

1. 基准严重退化。Forward + $S = 1$ 的 MSE 比其他四种配置高 1~3 个数量级，正是 P 元素被 sink 推到 subnormal 之下后 cast 归零的直接体现。

2. 逆序与加 scale 各自独立修复了基准。Reverse + $S = 1$ 与 Forward + $S = 448$ 都把 MSE 从 $10^{-4}$ 量级压回 $10^{-7} \sim 10^{-6}$ 量级，长序列上落到同一数量级。逆序通过控制 $m_{\text{global}}$ 轨迹避免 P 落入 subnormal，加 scale 则把 P 整体抬高远离 subnormal 下界，两条路径对修复 P 塌陷是等价的。

3. $S = 256$ 严格优于 $S = 448$。长序列上 Forward + $S = 256$ 比 Forward + $S = 448$ 低约 30%（seq_len=8192 上 $1.4 \times 10^{-7}$ vs $1.8 \times 10^{-7}$），与 $dp(S)$ 几何预测一致。

4. 两项优化叠加与单独 $S = 256$ 几乎重合。Reverse + $S = 256$ 与 Forward + $S = 256$ 差距均在 1% 以内，落在同一精度下界 $dp = 2^{-4}$ 上。这从实验上印证了两项优化修复的是同一个机理（subnormal 塌陷），叠加不再带来乘性增益。

综合而言，逆序计算主要在 P scale 不足以保护的实现路径下发挥作用（hpc-ops 的 $S = 1$、TensorRT-LLM XQA 的正向 K-loop），单独施加即可把 MSE 改善 1~3 个数量级；$S = 256$ 在已有 scale 保护的路径下相对 $S = 448$ 这类非二次幂取值给出约 30% 的进一步改善。两项优化机理不同、可独立施加，叠加饱和到同一下界。两者实现成本均极低，前者反转 for 循环方向，后者把 scale 从 $1/448$ 改为 $1/256$。

## 总结

本文从 FP8 E4M3 的数值结构出发，分析了 Hopper FP8 Attention 主流实现路径下的两个核心精度问题，并分别给出对应的工程优化。第一个问题是 Attention Sink 在正向迭代中引发的 P 塌陷，本文给出 KV block 逆序迭代作为解决方案；第二个问题是 P 在 cast 时的缩放因子选取，本文从 IEEE 754 浮点数学与 $dp(S)$ 锯齿波几何两个角度证立 $S = 256 = 2^8$ 是 $S \le 448$ 不溢出约束下的唯一最优解。最后通过对照实验定量验证了两项优化的有效性，并对照了 FlashAttention-3 / 4、FlashInfer、TensorRT-LLM XQA、Tencent hpc-ops、SageAttention2 / 2++ 等主流实现在 KV 顺序与 P scale 这两项实现选择上的具体取值。

## 参考

NVidia GPU指令集架构-浮点运算(https://zhuanlan.zhihu.com/p/695667044)

https://en.wikipedia.org/wiki/Binade(https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Binade)

https://github.com/Dao-AILab/flash-attention(https://link.zhihu.com/?target=https%3A//github.com/Dao-AILab/flash-attention)

https://github.com/flashinfer-ai/flashinfer(https://link.zhihu.com/?target=https%3A//github.com/flashinfer-ai/flashinfer)

https://github.com/NVIDIA/TensorRT-LLM(https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/TensorRT-LLM)

https://github.com/Tencent/hpc-ops(https://link.zhihu.com/?target=https%3A//github.com/Tencent/hpc-ops)

https://github.com/thu-ml/sageattention(https://link.zhihu.com/?target=https%3A//github.com/thu-ml/sageattention)
