# 第零层 + 四层自审案例集

这些案例来自 LeetCUDA 第 20–26c 章的实际复审，以及 R25/R26 对主书与
cute-zhihu 的全量复审。每条都按“坏句 → 为什么错 → 证据 → 修法”记录，
可直接当作对照样本。

## 第零层案例：数理逻辑

> 先看“前提能不能推到结论”；这里错了，后面的定义/API/资源/图注都不用看。

### 1. `size == cosize` 不等于双射

**坏句**：

```text
size 与 cosize 相等时，布局 compact（无空洞、双射）。
```

**为什么错**：

`(2,2,2):(1,1,5)` 的 `size = cosize = 8`，但像只有
`{0,1,2,5,6,7}`：3、4 缺失，且 8 个坐标只有 6 个不同 offset，存在重复。
等式只说明值域跨度与定义域基数相同，不保证单射或无洞。

**证据**：

- `cute/layout.hpp` 的 `coshape` 按 `1 + Σ|stride|(shape-1)` 计算；
- 反例逐坐标枚举。

**修法**：

```text
对单射、非负 stride 的 compact 布局，size == cosize；
反过来不成立，给出 (2,2,2):(1,1,5) 反例。
```

### 2. `O_c` 不是求和分量

**坏句**：

```text
Phase 4 把 8 个 chunk 的 O 累加器相加：O = Σ_c O_c。
```

**为什么错**：

Split-D 切的是输出 D 维，每个 `O_c` 对应互不相交的 64 列输出块；
各块用同一行的 softmax 统计量独立归一化后写回自己的列块。相加会把不同
输出列混在一起，维度也不对。

**证据**：

- `ffpa_attn.cuh` Phase 4 的 `local_tile(mO, ..., v_chunk)`；
- 每个 `v_chunk` 单独 `convert_type` 与写回。

**修法**：

```text
O 的 D 维被切成互不相交的列块；Phase 4 对每个 O_c 独立归一化并写回。
S 是归约维上的部分和，O 是输出维上的分块，两者不能都叫“归约”。
```

### 3. 对称 stages 的“任何 D”断言

**坏句**：

```text
对称的 (S,S) 在任何 D 都取不到最优。
```

**为什么错**：

同一章的表格里，D=384 与 D=512 的最优恰好是 `(2,2)`。坏句与自己的表冲突。
正确说法是“不存在一个固定的对称组合在所有 D 都最优”，不是“对称永远不好”。

**修法**：

```text
D <= 320 最优 (3,2)；D=384/512 回到 (2,2)；D=448 是 (2,3)。
launcher 暴露独立 S_K/S_V，按 D 分段选参。
```

### L0-4. “每行非空”推不出“首轮不 NaN”

**坏句**：

```text
定理假设每行至少有一个有限分数（非空行），所以首轮 rescale 一定安全。
```

**为什么错**：

前提只保证“存在有限分数”，不保证“首个被处理的块里有有限分数”。如果该行
首个 KV 块整块被 mask，则 `m1=-∞`，首轮 rescale 就是
`e^{-∞-(-∞)}=NaN`，后续 `0×NaN` 会污染输出。全 mask 行与“非空行的空前缀”
是两个不同边界。

**证据**：

- ch15 定理 15.1 的证明基础；IEEE 754 `-∞-(-∞)=NaN`；
- 源码用 `-INFINITY` 初始化 `m`，不是 `-FLT_MAX`；
- 工程若改用 `-FLT_MAX`，首轮不 NaN，但全 mask 行最终仍是 `ℓ=0`、`0/0`。

**修法**：

```text
补前提：首个被处理块非空；否则实现必须跳过空前缀，或用有限哨兵清洗。
全 mask 行单独归入调用方保证/尾部处理，不混进“非空行”定理。
```

### L0-5. “W 次写 + W 次读”漏掉了 W²

**坏句**：

```text
smem 流量只有 W 次写 + W 次读（各 4B）。
```

**为什么错**：

写是 W 个 float；但 W 个 warp 每个都要读取 W 个部分和，读侧是 W² 个 float
（W 条读事务，每条 W 个值）。把“读事务数”和“读取元素数”混成同一个 W，
会把读流量低估 W 倍。

**证据**：

- `base.cuh` 的 `block_reduce_sum`：warp0 的每个 lane 读取全部 W 个
  warp 部分和；
- 对比全 smem 朴素树形归约的 `log2(T)` 轮往返。

**修法**：

```text
写：W 个 float；读：W 条事务 × W 个 float = W² 个 float。
要比较的是字节/事务总量，不是“W 次写 + W 次读”这种对称句式。
```

## 第一层案例：定义、指称与边界

### L1-1. `make_tensor_like` 不是 non-owning view

**坏句/表项**：

```text
Heap non-owning: make_tensor(ptr, ...); make_tensor_like(tensor);
make_fragment_like(tensor);
```

**为什么错**：

`make_tensor_like` / `make_fragment_like` 会构造**自有存储**的 register
tensor（`ArrayEngine`），不是“给一个指针 + layout”的 non-owning view。
把它们和 `make_tensor(ptr, ...)` 放在同一行，会让读者以为它们只是换 layout。

**证据**：

- `cute/tensor_impl.hpp` 的 `make_tensor_like` 走 owning/ArrayEngine；
- `make_tensor(ptr, shape, stride)` 才是 non-owning。

**修法**：

```text
non-owning 行只留 make_tensor(ptr, ...)；
make_tensor_like/make_fragment_like 归入 owning register tensor，
并说明它们会分配寄存器存储。
```

### L1-2. 寄存器分配粒度是每线程 8，不是 4

**坏句**：

```text
寄存器文件按 kernel 所需寄存器数分配，分配粒度为 4；
示例中每个线程看到 R0-R3。
```

**为什么错**：

sm_70 及以后按每 warp 256 个 32-bit 寄存器分配，即每线程 8 个；4 是更老
架构的粒度，或者只是图中为讲解画的小例子，不能写成当前架构的分配粒度。

**证据**：

- 全书 occupancy 口径按每线程 8 个寄存器算；
- 图 2-b 的 R0–R3 只是示例，不是架构约束。

**修法**：

```text
写“每 warp 256 个 32-bit 寄存器（每线程 8 个）”；图里的 R0-R3 标注为
示例，不冒充分配粒度。
```

## 第二层案例：API 与硬件契约

### 4. `cp.async.wait_group N` 的等待语义

**坏句**：

```text
wait_group N 等的是栈顶往下恰好 N 组未完成；
尾部只剩 V 组时误用 wait<1>，会等一个不存在的组、等很久后仍能算完。
```

**为什么错**：

PTX 的语义是等待到**最多只有最近 N 组**仍未完成。它不会等一个“不存在的组”，
也不会因为组数不足而挂起。尾部只剩 V 组时 `wait<1>` 允许这组 V 仍未完成，
可能立即放行，读到未就绪的 V；这是正确性竞态，不是性能问题。

**证据**：

- PTX ISA `cp.async.wait_group`；
- `flash_attn.cuh` 实现 1 的 L2743-2754：尾部退回 `wait<0>`。

**修法**：

```text
wait_group N：最多只有最近 N 组未完成；较早提交的组已完成。
尾部必须 wait<0>，否则 V 未就绪。
```

### 5. `_1{}` 不是“沿 minor 维搬满”

**坏句**：

```text
make_tma_copy(..., _1{}) 声明沿 minor 维一次搬满。
```

**为什么错**：

`make_tma_copy` 的第五个参数是 `cluster_size`，此处等于 1；
box 几何由 `CTA_Tiler`（`Shape<...>`）给出，不是 `_1{}` 的职责。
`local_tile` 也没有这个第五参。

**证据**：

- `cute/atom/copy_traits_sm90_tma.hpp` 的
  `make_tma_copy(CopyOp, gtensor, slayout, cta_tiler, cluster_size)`；
- 被审源码 `make_tma_copy(..., Shape<...>{}, _1{})`。

**修法**：

```text
box 几何由 Shape<Int<kBr>, Int<kHeadDim>> 给出；
_1{} 是 cluster_size=1，不是 minor 维搬满声明。
```

### 6. `partition_fragment_B` 不是 smem 指针划分

**坏句**：

```text
partition_fragment_B(sV_stg) 用带 swizzle 的视图划分 smem 数据指针。
```

**为什么错**：

它建立的是每线程私有的**寄存器 fragment**布局，返回 rmem tensor；
S2R 的 smem 地址由 `partition_S(sVt_stg)` 提供。把前者当地址运算，
会把“谁持有哪个逻辑元素”和“从哪个 smem 地址读”混为一谈。

**证据**：

- `ffpa_attn.cuh` L341-345：
  `tCrVStorage = thr_mma_pv.partition_fragment_B(sV_stg)`，
  `tCrV = make_tensor(tCrVStorage.data(), tCrV_layout)`，
  `tVsVt = s2r_thr_v.partition_S(sVt_stg)`；
- CuTe `partition_fragment_B` 的返回类型是 rmem tensor。

**修法**：

```text
partition_fragment_B：线程--值归属 + 寄存器存储形状；
tCrV_layout：nonswizzle 逻辑布局；
partition_S：带 swizzle 的 S2R smem 地址。
三者分工不同。
```

### 7. TMA OOB、`ldmatrix`、对齐的边界

**坏句**：

```text
TMA 对 OOB 行不写 smem，残留值由 mask 覆盖；
ldmatrix 的地址通道原生支持 swizzle；
TMA smem 基址一律 1024B 对齐。
```

**为什么错**：

- TMA 默认 zero-fill，越界元素写 0，不是保留旧值；mask 仍必要，因为 0 不是
  `-inf`，会参与 softmax 归一化。
- `ldmatrix` 没有原生 swizzle 操作数，只接收每 lane 地址；XOR 在软件地址中算。
- TMA 32/64/128B swizzle 的基址要求分别是 256/512/1024B，不是一律 1024B。

**证据**：

- PTX TMA fill mode 0 / CUTLASS `OOB_FILL_NONE`；
- `ldmatrix` 指令签名；
- `copy_sm90_desc.hpp` / TMA swizzle 转换表。

**修法**：

```text
TMA OOB = zero-fill；mask 排除 0，不是“覆盖残留”。
ldmatrix 的 XOR 由软件地址完成。
对齐按 swizzle 档位写，不写死 1024B。
```

### 8. `cute::gemm` 会处理 `MMA_K`

**坏句**：

```text
cute::gemm 只展开 M/N，不碰 K；K 方向必须作者手写。
```

**为什么错**：

对 `(V,M,K) × (V,N,K)` fragment，`gemm` 会按 `tiled_mma` 展开
`MMA_M × MMA_N × MMA_K`，其中 K 循环遍历 `size<2>(A)`。作者手写的是
外层 K-tile 搬运循环，以及可选的 `k_step` 寄存器双缓冲；
这是实现选择，不是 `gemm` 的能力限制。

**证据**：

- `cute/algorithm/gemm.hpp` 的 `gemm` 重载；
- `hgemm.cuh` 的调用与 `k_step` 循环。

**修法**：

```text
MMA_K 在 gemm 内展开；外层 K-tile 仍需搬运；
显式 k_step 是寄存器双缓冲的实现选择。
```

## 第三层案例：资源与类型

### 9. 256 个逻辑累加值 vs 255 个寄存器

**坏句**：

```text
D=512 时每线程 8×32=256 个 float 累加器，是可用寄存器的一半。
```

**为什么错**：

算术 256 没错，但“寄存器”是物理单位。每线程最多 255 个 32-bit 寄存器；
256 个 f32 逻辑值本身已超过硬上限，再计地址、流水、临时量，必然 spill。
“可用寄存器的一半”把 255 的上限与 512 的配额混了，也与实际 spill 结论矛盾。

**证据**：

- 硬件 255 个 32-bit 寄存器/线程；
- `ffpa_attn.cuh` 的 `o_acc_storage[kDChunks][32]` 与 `-Xptxas -v` 的 spill 记录。

**修法**：

```text
256 个 f32 逻辑累加值，已超过 255 寄存器上限，必然 spill；
Split-D 本身不降低 acc_O 总量，只改变存放与循环结构。
```

### 10. `convert_type` 产生新 fragment，Aregs 才零拷贝

**坏句**：

```text
P 是 S 的零拷贝重解释：convert_type 只改类型解释，
P 与 S 共用同一批物理寄存器。
```

**为什么错**：

`convert_type` 用 `NumericArrayConverter` 做 f32→f16 数值转换，产生新的
f16 fragment；Aregs 随后的 `convert_layout_acc_Aregs` 才是不搬数据的布局
重解释。免掉的是 smem 往返，不是类型转换本身；不能承诺物理寄存器复用。

**证据**：

- `ffpa_attn.cuh` L301-303；
- 第 25 章 `convert_type` / Aregs 两段源码与 `NumericArrayConverter` 语义。

**修法**：

```text
softmax 写回 S；
convert_type 做数值转换并生成 f16 P；
Aregs 只重解释 P 的布局；
全程不落 smem，但不叫“原寄存器零拷贝”。
```

### 11. VCPB 是整除与容量双条件

**坏句**：

```text
D=320、默认 (2,2) 时 smem 只够一个 128×64 O 分量，所以 VCPB=1。
```

**为什么错**：

64 KB 池能放 4 个 16 KB 分量；但 `compute_vchunks_per_batch` 要求批量整除
`kDChunksV=5`，候选只有 1 或 5。5 个分量要 80 KB，超过 64 KB，因此退到
VCPB=1。不是“只能放一个”，而是“可放的 4 不整除 5，5 又装不下”。

**证据**：

- `ffpa_attn.cuh` 的 `compute_vchunks_per_batch`；
- `kSmemElems = 32768` half（64 KB），单个分量 8192 half（16 KB）。

**修法**：

```text
容量 64KB 可放 4 个分量，但批量必须整除 5；
1 或 5 中只有 1 满足容量，所以 VCPB=1。
```

### 12. 每个 M-warp 的私有副本

**坏句**：

```text
V 的 B fragment 是共享寄存器，每线程只算一份。
```

**为什么错**：

PV 的 B 数据对 4 个 M-warp 都要可见，每个 warp 各持一份寄存器副本；
每线程的元素数按自己的 partition 算，但 CTA 总寄存器成本包含 4 份重复。
“共享寄存器”既不是物理事实，也会低估 CTA 压力。

**修法**：

```text
每个 warp 一份私有 B fragment；
每线程按 partition 持有固定值；
评估 CTA 总量时乘上 M-warp 副本数。
```

## 第四层案例：表达一致性

### 13. 正文已改，图注还停在旧模型

**坏句**：

```text
正文：convert_type 生成新的 f16 fragment。
图 26-1：Phase 2 就地覆盖，不新增。
```

**为什么错**：

图和图注也是技术断言；读者会以图为准。正文与图注冲突时，不能只改正文。
本例中 softmax 确实写回 S，但 P 是新 f16 fragment；“不新增”只对
“不新增 smem”成立，不对寄存器成立。

**修法**：

```text
图 26-1：S → P；softmax 写回 S；convert_type 生成新 P fragment。
图注同步写 S 32 f32、P 32 f16。
```

### 14. 图注的行号/阶段指称错位

**坏句**：

```text
图 21：蓝色两行 = 第 1 轮（rows 0–1）。
```

**为什么错**：

表行是线程 t，不是数据行 m。4 个线程每轮都参与；第 1 轮覆盖数据行
m=0,1，第 2 轮覆盖 m=2,3。把表行当轮次会让读者以为只有两个线程工作。

**修法**：

```text
蓝色四行 = 4 个线程全体；
第 1 轮共同覆盖数据行 m=0,1，第 2 轮覆盖 m=2,3。
```

### 15. 流水图的 stage 覆盖关系

**坏句**：

```text
S=2 时 K1 的 TMA 可以复写 stage 0。
```

**为什么错**：

prefill 只装 K0；主循环第一个 tile 发 K1 到 stage 1。复写 stage 0 的是
K2（以及 V2），且 K2 要等 QK0 的 `k_empty[0]`，V2 要等 PV0 的
`v_empty[0]`。把 K1 画成复写 stage 0 会误导流水深度与 barrier 相位。

**修法**：

```text
prefill: K0 → V0；
稳态: 每 tile 先 V 后 K；
S=2: K2/V2 复写 stage 0，等 QK0/PV0 的 empty。
```

### 16. persistent 的收益不能张冠李戴

**坏句**：

```text
persistent CTA 消除 wave 量化尾损，并让 Q 跨 q-tile 持久。
```

**为什么错**：

在等时 tile、静态认领下，2048 个 tile / 110 个 CTA 的 makespan 仍是 19，
与非 persistent 的 19 个 wave 相同；减少的是 launch/CTA 调度和状态重建。
Q 在每个 q-tile 仍要重新装载，“跨 q-tile 持久”不成立。收益归因应写成
调度开销下降与 CTA 内屏障/流水状态复用。

**修法**：

```text
persistent 复用 CTA 内屏障与流水状态、降低调度开销；
等时 tile 的最大 tile 数仍决定 makespan；
Q 在每个 q-tile 重新装载。
```

## R25/R26 增补案例：跨层与 API

### 17. `cute::gemm` 四参重载的参数顺序

**坏句/代码**：

```cpp
cute::gemm(tiled_mma, trC, trA, trB);   // 把 C 当 A，A 当 B，B 当 C
```

**为什么错**：

CuTe 的四参重载是 `gemm(mma, A, B, C)`，不是 `(mma, C, A, B)`。五参重载才是
`gemm(mma, D, A, B, C)`。同一篇 `zhuxi-02` 后文已经写了正确的
`gemm(tiled_mma, trA, trB, trC)`，前文却反了——属于同文自相矛盾。

**证据**：

- `cute/algorithm/gemm.hpp:83`：`gemm(MMA_Atom<MMA>, Tensor A, Tensor B, Tensor C)`；
- 同文件 `:127` 的五参版本是 `(mma, D, A, B, C)`；
- `zhuxi-02` 的循环展开版也必须写成 `(mma, trA(_,i_m,_), trB(_,i_n,_), trC(_,i_m,i_n))`。

**修法**：

```text
先确认重载参数个数，再按 (A, B, C) 或 (D, A, B, C) 排；不要靠“看起来像输出”猜。
```

### 18. listing 截断在表达式中间

**坏句/代码**：

```text
\lstinputlisting[linerange={2786-2798}]{../flash_attn.cuh}
```

源码 2796–2799 是：

```cpp
auto mO = make_tensor(make_gmem_ptr(output),
                      make_shape(rows, Int<kHeadDim>{}),
                      make_stride(Int<kHeadDim>{}, _1{}));
```

range 停在 `make_shape(...),`，缺 `make_stride` 和右括号，读者看到的是半个表达式。

**为什么错**：

LaTeX 不会因为代码 fragment 不完整而报错；`PDF/RTD` 都能过，只有逐条对源码
行号才会发现。同类：`ch17` 标题承诺“释放 V”，但 1260–1306 不含
`empty_V.arrive()`，必须延伸到 1312。

**证据**：

- `flash_attn.cuh:2796-2803`；`flash_attn.cuh:1307-1312`；
- 标题里的 `Lxxxx` 与 `linerange` 必须逐字一致。

**修法**：

```text
listing 必须结束在完整语句/分支/函数片段；至少检查：
末行不是以 ',' '(' '=' 结尾，下一行不是 '}' ')' 收尾，标题行号与 range 相同。
```

### 19. 表行名必须对应实际实现

**坏句**：

```text
naive & 512 & 256 & 0.065 & 4.13 & 6.2 & 516
```

**为什么错**：

同章解析 naive 的 AI 是 0.25（2 FLOP / 8B），而这一行的模型带宽 516 GB/s
对应 AI=8 的 32×32 `sgemm`。表里的 `naive` 不是分析模型，而是 32×32 实现；
不改标签会让读者以为 naive 能跑到 4 TFLOPS。

**证据**：

- `sgemm.cuh` 的 `BM=BN=BK=32`；
- 表内 516/642/683 GB/s 与 32×32 tiling 的流量模型一致；
- 同章 4.76×/5.5T 等结论也都按这组实测行计算。

**修法**：

```text
表行改 sgemm，表注注明 32×32；正文的“naive”只保留给 AI=0.25 的解析模型，
所有实测比较改成“32×32 版 sgemm”。
```

### 20. phase 周期必须按 C 与 Sk 算

**坏句**：

```text
D=128 时 phase 每 1 个 kv_tile 翻转；D=192 时周期 2。
```

**为什么错**：

phase 周期是 `lcm(C, 2·Sk)/C`：D=128 → C=2、Sk=2 → 2；D=192 → C=3、Sk=2
→ 4。写成 1/2 与同章表 19-phase 直接冲突。

**证据**：

- `ffpa_attn.cuh` 的 chunk 数 `C = D/64` 与 `phase = (idx / Sk) & 1`；
- 同章表 19-phase 的 2/4。

**修法**：

```text
先算 C 与 lcm(C, 2Sk)/C，再写每档周期；不要从一个 D 的例子外推所有 D。
```

### 21. GMMA 原子是位置相关语义，不是普通 layout

**坏句**：

```text
把 GMMA::Layout_K_SW128_Atom<half> 铺到 64×64 tile 上，
layout(r,c) 应与 64r+8·((c/8)⊕(r mod 8))+(c mod 8) 逐点相等。
```

**为什么错**：

GMMA 的 smem atom 是指针语义，`layout(r,c)` 只给相对偏移，真正的物理地址
还依赖基址；直接把它当普通 layout 求值，只有 1024/4096 命中。正确验证是把
`Swizzle<3,4,3>` 作用在**字节地址**上，再做 4096 点比较。

**证据**：

- 本书 `tests/ch23_swizzle_tma.cu` 的 case E1/E2/E3：E1/E2 4096 点全中，
  E3 明确断言裸 `layout(r,c)` 非等价、只有 1024/4096 命中。

**修法**：

```text
区分“layout 代数”和“smem 指针 + swizzle 的位置相关语义”；
验证时先构造字节地址，再套 Swizzle，不要直接对 GMMA atom 的 layout(r,c) 求值。
```

### 22. R2S 的 dtype 转换不在 `UniversalCopy`

**坏句**：

```text
R2SCopyAtomC 是 UniversalCopy<int>，用 UniversalCopy 顺带完成类型转换。
```

**为什么错**：

`UniversalCopy<int>` 只做 32-bit 搬运；真正的 f32→f16 转换发生在更前面的
R2R staging（`make_tensor_like<T>` + 普通 `cute::copy`），R2S 只把
thread-local payload 写进 smem。默认 F16 acc 且 `T=half` 时，这次 cast 还是 no-op。

**证据**：

- `hgemm.cuh` 的 `r2r_acc` / `make_tensor_like<T>` 与后续
  `r2s_tiled_copy_c` 调用点；
- `Copy_Atom<UniversalCopy<int>, T>` 的 `T` 只描述元素类型，不执行数值转换。

**修法**：

```text
把“数值转换（R2R）”和“寄存器→smem 搬运（R2S）”分开写；
不要因为 Copy_Atom 带 T 就认为它在做 cast。
```

### 23. `logical_divide` 非整除时 Size 会变大

**坏句**：

```text
对 A 做 divide 得到的 C，和 A 的 Size 是一样的。
```

**为什么错**：

只有 tiler 整除 A 各维时才相同。非整除时 Rest 维取 ceil：
`logical_divide((10):(1), (4):(1))` 得到 `((4,1),3)`（tile 4、rest 3），
Size=12>10；多出的位置是越界元素，`local_tile` 的尾部 tile 正是靠这个 ceil
覆盖工作。

**证据**：

- CuTe `logical_divide` 的 Rest 维用 `ceil_div`；
- 反例逐坐标枚举 10 个元素，Rest 需要 3 个 4-wide tile。

**修法**：

```text
加“tiler 整除”前提；不整除时明确 Rest=ceil、Size 变大、末块含越界；
local_tile 的边界守卫要单独讲。
```

### 24. “4 个值一次 STG.128”先算 dtype 宽度

**坏句**：

```text
thread0 拿到连续的 4 个值，STG.128 就可以执行了。
```

**为什么错**：

STG.128 是 16B。4 个 fp64 是 32B，需要两次；4 个 fp32 才是 16B；4 个 fp16
只有 8B，对应 STG.64。`zhuxi-06` 用 fp64 画图，但 store 字长收益属于
4 字节累加器；fp64 的 thread0 置换前本来就有两个连续 double（16B），置换
并不增加它的 store 字长。

**证据**：

- STG.128 是 16B，`4×8B=32B` 至少需要两次；`4×4B=16B` 才是一次；
- `zhuxi-06` 的 `SM80_8x8x4_F64F64F64F64_TN` 是 fp64 atom，置换前 thread0
  已有两个连续 double（16B）。

**修法**：

```text
先算 N×sizeof(dtype)，再选 STG.32/64/128；不要用“值个数”代替字节数。
```

### 25. Task agent 的 high 候选也要复核

**坏候选**：

```text
ch18 的 β = e^{-∞-m0} = 0，若 m0=-∞ 会产生 NaN，因此该段错误。
```

**为什么不是错误**：

该段讨论的是“Tc=1 导致 WG1 为空、WG0 非空”的 dense 路径，m0 有限；
真正的全 mask 行由 ch26b 的空行分支处理。把 `m0=-∞` 套进来是换了前提。

**教训**：

```text
agent 可以扩大覆盖，但不能代替判定。每条 high 必须回到源码前提/定义域；
举证责任在“主张错误”的一侧，不在原文。
```

## 案例使用方式

审阅时不要只搜索原句。把每个案例抽象成检查问题：

1. 这个等式是双向的吗？反例是什么？
2. 这个 API 参数的作用域和单位是什么？签名在哪？
3. 这里是逻辑值、寄存器、smem 字节还是 CTA 总副本？
4. 正文改了，图注/表/问答/标题跟上了吗？

先过第零层（推导链/量词/反例），再过四层；全部过一遍，再写“通过”。
