<!--
author: 赵尚春
author_id: tongsanpang
url: https://zhuanlan.zhihu.com/p/2052843307584295385
column: CUDA-365
published: 2026-06-23
fetched: 2026-11-22
images: 36
-->

# GTC 2026 技术分析：通过 CUTLASS Python 在 Blackwell GPU 上实现 GEMM 峰值 Tensor Core 性能

## 第一章：引言与动机

本文基于 GTC 2026 演讲 "Achieve Peak Tensor Core Performance for GEMM on Blackwell via CUTLASS Python" 进行深入技术分析。该演讲由 NVIDIA 工程师 Linfeng Zheng 和 Albert Di 主讲，系统性地展示了如何使用 CUTLASS Python 在 Blackwell 架构（B200 GPU）上逐步优化 GEMM（通用矩阵乘法）运算，最终达到接近硬件理论峰值的 Tensor Core 性能。视频见：[通过CUTLASS Python 在Blackwell 上实现GEMM 的峰值Tensor Core 性能](https://www.bilibili.com/video/BV1egGs6dE54?spm_id_from=333.788.videopod.sections&vd_source=0d656ebac9f6ff3cdced8f1272b08374) 。

### 1.1 为什么选择 CUTLASS Python？

![img-1](https://pic2.zhimg.com/v2-e13a37d626bf1ccdac2055f42dc809b9_r.jpg)

**达到 SOL（Speed of Light）性能的挑战：**

要在现代 GPU 上达到 SOL 性能，开发者需要使用更低层级的抽象来精细控制硬件特性。特别是 Blackwell 架构引入的 **PDL（Programmatic Dependent Launch）** 等高级特性，需要细粒度的控制才能充分发挥其潜力。

**为什么选择 CUTLASS：**

- **Day-0 可用性：** CUTLASS 在新硬件发布的第一天即可使用，提供完全的硬件控制能力
- **表达力强的抽象层：** 其抽象设计能够在各种场景下都获得高性能
- **模块化和可扩展设计：** 历经多代 GPU 架构验证，设计稳健可靠

**为什么选择 Python 接口：**

- **C++ 模板编程不便：** CUTLASS C++ 的模板元编程对开发者极不友好，代码晦涩难懂
- **编译时间过长：** C++ 模板会导致极长的编译时间，严重影响开发迭代效率
- **Python 生态兼容：** 深度学习领域已全面拥抱 Python 生态系统，Python 接口能更好地融入现有工作流

### 1.2 CUTLASS Python 的优势

![img-2](https://pica.zhimg.com/v2-0b62cc4888dc04aaf1595bf183a6e728_r.jpg)

**性能对比数据（8K×8K×8K GEMM）：**

CUTLASS Python 相较于传统方式实现了令人印象深刻的优势：

- **性能方面：** 在 8K×8K×8K GEMM 场景下达到了极高的 TFLOPS 吞吐量（图中 "Higher is better" 方向）
- **编译速度：** 编译时间相较 C++ 实现缩短超过 **100 倍**（图中 "Lower is better" 方向）

这说明 CUTLASS Python 不仅没有牺牲运行时性能，还极大地改善了开发体验和迭代速度。JIT（Just-In-Time）编译和缓存机制使得 Python 接口在首次编译后可以达到与 C++ 相当的运行时性能。

### 1.3 CUTLASS Python 架构概览

![img-3](https://pic4.zhimg.com/v2-287edaf4face9f64855d7bcb2933b625_r.jpg)

**CUTLASS Python 的分层架构：**

CUTLASS Python 基于 **CuTe DSL（Domain Specific Language）** 构建，形成了清晰的层次结构：

**底层 - CuTe Atoms：**

- TMEM store/load（Tensor Memory 操作）
- TMA（Tensor Memory Accelerator）
- MMA SM90/SM100（Matrix Multiply-Accumulate 指令）

**中间层 - Tiled Copy & Tiled MMA：**

- 提供分块拷贝和分块矩阵乘法的抽象
- 支持不同架构的 Collective 操作（SM90 TMA SS、SM100 TMA、SM100 ATMEM 等）

**Collective Layer：**

- SM90 GEMM TMA Warp Specialized（Ping-pong/Cooperative 模式）
- SM100 GEMM TMA Warp Specialized
- SM100 GEMM TMA A from TMEM

**Device Layer（最高层）：**

- Universal Interfaces
- Dense GEMM、Grouped GEMM 等

**开发生态支持：**

- **开发工具：** DL Pack、JIT Caching、TVM FFI、AOT Compilation、多 Python 版本支持、Auto-Tuning、静态/动态参数
- **平台支持：** X86-64 & Arm
- **调试工具：** 编译时/运行时 Print、MLIR/PTX/Cubin Dumping、Source Location Tracking、Logging System
- **指导资源：** Core Examples、Tutorials for Beginners

本演讲的重点聚焦在 **Tutorials for Beginners** 部分，帮助初学者从零开始理解 GEMM 优化。

### 1.4 CUTLASS Python 生态现状

![img-4](https://pica.zhimg.com/v2-65322aeea80e9a4dd106de2916af94ea_r.jpg)

**CUTLASS Python 的快速增长：**

- 单日下载峰值达 **200K 次**
- 累计下载量达 **560 万次**
- 已被多个知名项目采用：**FAv4**（Flash Attention v4）和 **QuACK**
- CuTe DSL 内核已集成到 **TRTLLM**（TensorRT-LLM）和 **FlashInfer** 中

随着越来越多的示例被添加到代码仓库中，开发者能够更快速地上手和精通 CUTLASS Python 编程。

### 1.5 Tutorial GEMM 的必要性

![img-5](https://pic1.zhimg.com/v2-f93fa1a3046aa6427280a0781f484e58_r.jpg)

**现有示例的复杂性问题：**

尽管 CUTLASS Python 提供了大量示例，但存在以下障碍：

- 每个示例都超过 **2000 行代码**，对初学者来说门槛过高
- 初学者难以理解：
- 如何实现一个基本的 GEMM
- 哪些硬件特性已被启用
- 对于不同问题规模，应该启用哪些特性来提升性能

这正是 **Tutorial GEMM** 系列的价值所在——提供循序渐进的学习路径。

### 1.6 GEMM 的基本特性

![img-6](https://picx.zhimg.com/v2-079931ff6c952012042917d6cdc98fbd_r.jpg)

**GEMM 的四大核心特性：**

**可并行性（Parallelizable）：**

- 输出矩阵 C 的每个元素 $$C_{i,j$$ 可以独立计算
- 有助于充分利用 GPU 中的所有 SM（Streaming Multiprocessor）

**数据可复用性（Data-reusable）：**

- C 的第 i 行需要 A 矩阵的第 i 行 $$A_$$ 的相同数据
- C 的第 j 列需要 B 矩阵的第 j 列 $$B_$$ 的相同数据
- 这为减轻 IO 压力提供了可能性

**块友好性（Block-friendly）：**

- 一组元素可以被一起处理
- 有助于减轻每个 SM 的 IO 压力
- 为使用 MMA 指令提供了可能性

**瓶颈灵活性（Bottleneck-flexible）：**

- 不同问题规模下瓶颈各异
- 要达到 SOL 性能，通常需要针对不同问题规模采用不同策略

### 1.7 Tutorial GEMM 的贡献

![img-7](https://pic3.zhimg.com/v2-0d0f2aa133e29ccaaf84d1f59b595c5a_r.jpg)

**Tutorial GEMM 系列的主要贡献：**

提供了一系列渐进式优化的教程，每个仅需 200-300 行代码：

| 教程名称 | 优化技术 | 解决的问题 |
| --- | --- | --- |
| fp16gemm0.py | 基础 GEMM | 基线实现 |
| fp16gemm1.py | 启用 tcgen05.2cta | 延迟隐藏能力不足 |
| fp16gemm2.py | Warp Specialization + TMA Store | 波内优化 |
| fp16gemm3/3_1.py | Static/Dynamic Persistent | 波间切换优化 |
| fp16gemm4.py | Preferred/Fallback Clusters | 充分利用 SM |
| fp16gemm5.py | PDL | 内核重叠 |
| fp16gemm6.py | TMA L2 Prefetch | DRAM 延迟 |

## 第二章：基础 GEMM 概览

本章详细解析基础 GEMM（Tutorial 0）的实现原理，这是所有后续优化的基础。

### 2.1 基础 GEMM 工作流程

![img-8](https://pic4.zhimg.com/v2-552df3649a908f2bd983cdd77d406615_r.jpg)

**整体计算流程：**

基础 GEMM 的工作流程分为以下关键步骤：

**Step 1 - 创建 MMA Tile 视图（`cute.local_tile()`）：**

将全局内存（GMEM）中的大矩阵按照 MMA Tile 大小进行分块。在图中可以看到，矩阵 C 被划分为 M Tiles × N Tiles 的网格，每个 CTA 负责计算其中一个 tile（如图中标注的 C[3,1]）。对应的 A 矩阵取出相应的行 tile，B 矩阵取出相应的列 tile。

**Step 2 - 使用 TiledMMA 分区（`thr_mma.partition_A()`）：**

将 MMA Tile 进一步分配给具体的线程。TiledMMA 抽象定义了线程如何协作完成矩阵乘法运算，每个线程负责 tile 中的特定元素子集。

**Step 3 - 为 TMA 分区 SMEM/GMEM（`tma_partition()`）：**

Tensor Memory Accelerator（TMA）是 Blackwell 架构的硬件加速器，负责高效地在全局内存和共享内存之间传输数据。这一步为 TMA 操作准备好源和目标的内存分区。

**Step 4 - TMA 从 GMEM 拷贝到 SMEM（`cute.copy()`）：**

使用 TMA 硬件单元将数据从全局内存异步加载到共享内存（SMEM）。图中的 "A buffer" 和 "B buffer" 即为 SMEM 中的缓冲区。通过多缓冲（multi-buffering）策略，可以在计算一个 K block 的同时加载下一个 K block 的数据。

**Step 5 - 创建 MMA Fragment（`tiled_mma.make_fragment_A()`）：**

从 SMEM 中创建 MMA 操作所需的数据片段（fragment）。在 Blackwell 架构中，这些片段存储在 TMEM（Tensor Memory）中，这是一种专为 Tensor Core 优化的内存层级。

**Step 6 - 执行 MMA（`cute.gemm()`）：**

调用 Tensor Core 执行矩阵乘累加运算。结果存储在 TMEM 中的累加器（Accumulator）里。整个 K 维度的循环通过 "K blocks" 迭代，每次处理一个 MMA Tile K 大小的数据。

**Step 7 - 存储结果：TMEM → RMEM → GMEM（`cute.copy()`）：**

将最终计算结果从 TMEM 传输到寄存器（RMEM），再从寄存器写回全局内存。这是基础版本中最简单的 epilogue 实现。

### 2.2 数据分块与内存层次（续）

![img-9](https://pic3.zhimg.com/v2-2af6ed8bb0f0d4b07796035fd2b56daa_r.jpg)

**分块策略的详细说明：**

图中展示了数据如何在不同内存层次间流动：

- **GMEM → SMEM：** 通过 TMA 将 A 和 B 矩阵的 tile 加载到 SMEM 的 buffer 中
- **SMEM → TMEM/RMEM：** 通过 MMA Fragment 的创建过程，数据被送入 Tensor Core 的输入缓冲
- **TMEM → GMEM：** 计算完成后将累加器结果写回全局内存

关键的 tiling 参数：

- **MMA Tile M/N/K：** 定义了 Tensor Core 单次 MMA 操作的大小
- **K Tiles：** 沿 K 维度的分块数量，决定了主循环的迭代次数
- **ab_stages：** SMEM 中的缓冲区数量，决定了 software pipeline 的深度

### 2.3 基础 GEMM 性能基线

![img-10](https://picx.zhimg.com/v2-c6225dd49e3d5b6cde5eb3dea9e4bb3f_r.jpg)

**内核配置：**

- 非 Warp Specialization 模式
- MMA tiler 大小：128×256×64（M×N×K）
- ab_stages = 4（SMEM 缓冲区数量）
- Epilogue：TMEM → RMEM → GMEM（最基本的写回方式）

**B200 FP16 峰值性能分析：**

B200 GPU 的理论峰值性能为 **2.25 petaFLOPS**（@ 1.9GHz），但实际可达峰值取决于输入数据类型：

| 输入数据 | 频率 | PFLOPS |
| --- | --- | --- |
| Zeros（全零） | ~1.8 GHz | ~2.18 |
| Random Int（随机整数） | ~1.6 GHz | ~1.94 |
| Random Float（随机浮点） | ~1.35 GHz | ~1.64 |

这是因为 GPU 的 DVFS（Dynamic Voltage and Frequency Scaling）机制会根据功耗和温度动态调整频率。随机浮点数据的功耗最大，因此频率最低。

**性能数据：**

从图表来看，Tutorial 0 在不同问题规模下的性能：

- 2K×2K×2K：相对较低，受 kernel launch 开销影响
- 4K×4K×4K：逐步提升
- 8K×8K×8K：约 1200+ TFLOPS
- 16K×16K×16K：约 1300+ TFLOPS

基线性能为 **1.64 petaFLOPS（1640 TFLOPS）**，这为后续优化提供了起点。

## 第三章：渐进式 GEMM 优化——计算密集型场景

本章是整个演讲的核心部分，详细展示了如何从基线性能逐步优化到接近理论峰值。

### 3.1 优化路径总览

![img-11](https://pic3.zhimg.com/v2-2093a2788957f995ebf975be7beae1ac_r.jpg)

该图展示了从基线到峰值性能的完整优化路径，每一步优化都叠加在前一步之上，最终在 NVIDIA B200 上达到接近 SOL 的性能。

### 3.2 Tutorial 0：Software Pipeline 与延迟隐藏

![img-12](https://pic4.zhimg.com/v2-f10e597c68609b5f133fb708d3e771ff_r.jpg)

**Software Pipeline 的核心原理：**

DRAM 的加载延迟通常在 **数百到上千个时钟周期**，如果不做隐藏，Tensor Core 将在等待数据时处于空闲状态。Software Pipeline 通过以下机制隐藏延迟：

**多缓冲机制（Multi-buffering）：**

- 使用多个 SMEM 缓冲区（ab_stages 个）来交替存放 TMA 加载的数据
- 受 SMEM 容量限制，缓冲区数量有限

**预取策略（Prefetching）：**

- 通常预取 `ab_stages - 2` 个 stage 的数据
- 延迟隐藏能力 = `(ab_stages - 1) × one_stage_mma_inst_time`

**代码实现的关键结构：**

```python
for k_tile_idx in cutlass.range(num_k_tiles, prefetch_stages=ab_stages - 2):
    # Issue TMA loads
    ab_empty = ab_producer.acquire_and_advance()
    cute.copy(...)
    # Execute one K-block worth of MMA instructions
    ab_full = ab_consumer.wait_and_advance()
    cute.gemm(...)
    # Signal buffer consumed
    ab_full.release()
```

这里使用了 **Producer-Consumer** 模型：

- Producer 负责 TMA 加载，获取空缓冲区后发起异步拷贝
- Consumer 负责 MMA 计算，等待缓冲区数据就绪后执行计算
- 计算完成后释放缓冲区供 Producer 复用

### 3.3 Tutorial 0 的不足：延迟隐藏能力有限

![img-13](https://picx.zhimg.com/v2-681b5a5fd42c08a1020122951476feff_r.jpg)

**Tutorial 0 的瓶颈分析：**

对于 Tutorial 0 的配置（128×256×64 tile size）：

- 受 SMEM 容量限制，ab_stages 最多为 4
- 延迟隐藏能力：`(4 - 1) × 512 = 1536 cycles`

**完美隐藏条件：** latency_hiding_ability >=geq data_loading_latenc

然而实际测量表明：

- 数据加载延迟约为 **1800 cycles**
- 由于 1536 < 1800 cycles，**数据加载延迟无法被完全隐藏**

**NCU（Nsight Compute）分析结果：**

- 波内计算吞吐量仅约 **72%**
- 图中右侧的时间线清楚展示了当 ab_stages=4 时，MMA 执行存在明显的停顿（stall），等待数据加载完成

这个分析直接引出了 Tutorial 1 的优化方向：如何增加延迟隐藏能力。

### 3.4 Tutorial 1：启用 2CTA（tcgen05.2cta）

![img-14](https://pic1.zhimg.com/v2-4381b96f22f4b2cf96da96c2df710002_r.jpg)

**Blackwell 2CTA 技术原理：**

Blackwell 架构引入了一项重要的硬件创新——**将 Tensor Core 执行扩展到 2 个 CTA（Cooperative Thread Array）**：

**核心概念：**

- 一对 CTA 在同一 TPC（Texture Processing Cluster）的 2 个 SM 上协同执行一次 MMA
- 2×1 cluster 包含 1 个 CTA pair
- 4×4 cluster 包含 8 个 CTA pairs（2×4 布局）

**执行机制：**

- 2×1 CTA pair 中的 CTA 0 是 "leader CTA"
- Leader CTA 选出一个线程为两个 SM/CTA 发射 MMA 指令（`tcgen05.mma`）
- A、B 矩阵和累加器在 2 个 SM 之间均匀分割
- 每个 SM 读写其本地的 SMEM/TMEM

**内存布局（如图所示）：**

- SM 0 和 SM 1 各持有 A 矩阵的一半（SMEM 或 TMEM）
- B 矩阵在两个 SM 之间共享（关键优化点！）
- 累加器分别存储在两个 SM 的 TMEM 中
- MMA 维度：M=256（跨 2 个 SM），K 和 N 维度在两个 SM 内处理

### 3.5 Tutorial 1：2CTA 带来的 SMEM 优化

![img-15](https://pica.zhimg.com/v2-c335a6a2a332bc04ca4c0677ccbe0f1a_r.jpg)

**2CTA 的配置变化：**

启用 2CTA 后，关键参数发生如下变化：

- `mma_tiler_mnk`：128×256×64 → 256×256×64
- `cta_tiler_mnk`：128×256×64（每个 CTA 的实际 tile 不变）
- `cluster_shape_mnk`：(1,1,1) → (2,1,1)

**SMEM 节省的关键原因：**

由于 **B 矩阵在 2 个 SM 之间共享**，每个 SM 只需要存储一半的 B 数据：

| 配置项 | 1 CTA SMEM (KB) | 2 CTA SMEM (KB) |
| --- | --- | --- |
| one_stage_A | 16 | 16 |
| one_stage_B | 32 | 16 |
| one_stage_all | 48 | 32 |
| num_ab_stages | 227 // 48 = 4 | 227 // 32 = 7 |

**延迟隐藏能力提升：**

- 原来：(4-1) × 512 = 1536 cycles
- 现在：(7-1) × 512 = **3072 cycles**

ab_stages 从 4 提升到 7，延迟隐藏能力翻倍，远超 1800 cycles 的加载延迟，理论上可以完全隐藏 DRAM 延迟。

### 3.6 Tutorial 1：性能结果

![img-16](https://picx.zhimg.com/v2-6478ac074363d4431e6593a9b48921ad_r.jpg)

**1CTA vs 2CTA 性能对比：**

- **2K×2K×2K：** 0.96x（略有下降）——因为 prologue 和 epilogue 占比较大，2CTA 引入的同步开销反而成为负担
- **4K×4K×4K：** 1.05x 提升
- **8K×8K×8K：** 1.08x 提升——主循环占比增大，延迟隐藏优势显现
- **16K×16K×16K：** 1.02x（提升有限）——性能受限于 **L2 缓存命中率**

**关键洞察：** 2CTA 优化的收益在中等规模（4K-8K）最为明显。对于小规模，同步开销主导；对于大规模，L2 缓存成为新的瓶颈。这体现了 GEMM 优化中"瓶颈灵活性"的特点——不同规模有不同的性能瓶颈。

### 3.7 Tutorial 2：Epilogue 优化——TMA Store

![img-17](https://pic4.zhimg.com/v2-adf4652fa604cba8fb5a3ea7994d98a5_r.jpg)

**问题背景：**

在 Tutorial 1 中，结果从 RMEM 直接存储到 GMEM（STG 指令）。但由于连续线程的数据在 GMEM 中并不连续，这导致了非合并（non-coalesced）的内存访问，严重浪费带宽。

**优化方案：STG → TMA S2G + Sub-tiling**

Tutorial 2 将 epilogue 改为通过 TMA 进行合并存储：

- Step 1：TMEM → RMEM（T2R）
- Step 2：RMEM → SMEM（R2S）
- Step 3：SMEM → GMEM（TMA S2G，合并访问）

**Sub-tiling 策略：**

为了节省 SMEM 和寄存器使用量，采用子分块（sub-tiling）方式：将 epilogue 的输出分成多个子块，每次只处理一个子块的 T2R → R2S → TMA S2G 流水线。如图中所示，4 个子块依次执行，R2S 和 TMA S2G 可以重叠。

**核心代码结构：**

```python
for subtile_idx in cutlass.range(subtile_cnt):
    # TMEM -> RMEM
    cute.copy(tiled_copy_t2r, tCtC, tCrC)
    # Type conversion
    tCrC_out.store(tCrC.load().to(io_dtype))
    # RMEM -> SMEM
    cute.copy(tiled_copy_r2s, tCrC_out, tCsC)
    # Memory fence
    cute.arch.fence_view_async_shared()
    epilogue_sync_barrier.arrive_and_wait()
    # SMEM -> GMEM (TMA)
    if warp_idx == epilogue_warp_ids[0]:
        cute.copy(tma_c.atom, tCsC, tCgC)
```

### 3.8 Tutorial 2：Epilogue 时间线优化效果

![img-18](https://pic4.zhimg.com/v2-456b0beb4bbd13b14876ea72c944b4a1_r.jpg)

**时间线对比：**

该图清晰展示了 epilogue 优化前后的执行时间线：

- **优化前：** Epilogue 阶段占据较长时间，因为非合并内存访问导致大量带宽浪费
- **优化后：** Epilogue 时间显著缩短，TMA 合并存储的效率远高于逐线程存储

这一优化对小 K 值（epilogue 占比较大）的场景尤为重要。

### 3.9 Tutorial 2：Warp Specialization

![img-19](https://pic3.zhimg.com/v2-addc7642de4a777388624a188613ae26_r.jpg)

**Software Pipeline vs Warp Specialization：**

除了 Software Pipeline 预取机制外，**Warp Specialization** 提供了另一种隐藏延迟的方式：

**核心思想：** 将不同类型的指令分配给不同的 warp 执行，实现任务级并行。

**Warp 分配方案：**

```python
epilogue_warp_ids = (0, 1, 2, 3)  # 4 个 warp 处理 epilogue
mma_warp_id = 4                    # 1 个 warp 处理 MMA
tma_warp_id = 5                    # 1 个 warp 处理 TMA
threads_per_cta = 32 * 6           # 共 6 个 warp = 192 线程
```

**优势：**

- **更好的指令发射效率：** 不同类型的指令（TMA load、MMA compute、TMA store）由专用 warp 发射，避免指令队列拥塞
- **隐藏 epilogue 开销：** 当 MMA warp 还在计算时，epilogue warp 可以处理上一次迭代的结果写回

**执行模式：**

```python
if warp_idx == tma_warp_id:
    ...  # 负责所有 TMA 加载操作
if warp_idx == mma_warp_id:
    ...  # 负责所有 MMA 计算操作
if warp_idx < epilogue_warp_ids:
    ...  # 负责结果写回操作
```

### 3.10 Tutorial 2：性能提升

![img-20](https://picx.zhimg.com/v2-cb4ca3c07c32565fa9071ba2ff42a92f_r.jpg)

**Warp Specialization + TMA Store 的性能收益：**

**大 K 场景（左图，KxKxK）：**

- 2K×2K×2K：1.18x 提升
- 4K×4K×4K：1.11x 提升
- 8K×8K×8K：1.03x 提升
- 16K×16K×16K：1.00x（几乎无提升）

对于大 K 值，mainloop 占据绝大比例，因此 epilogue 和 prologue 的优化效果不明显。 **小 K 场景（右图，KxKx256）：**

- 2K×2K×256：1.51x 提升
- 4K×4K×256：1.51x 提升
- 8K×8K×256：1.53x 提升
- 16K×16K×256：1.57x 提升

**关键发现：** 当 K 维度较小时，epilogue 占据了内核执行时间的大比例，Warp Specialization + TMA Store 的优化效果极为显著（超过 1.5x）。这在实际 LLM 推理中的 prefill 阶段（通常 K=hidden_dim，相对较小）非常有价值。

### 3.11 Tutorial 3：Persistent Kernel——波间重叠优化

![img-21](https://pic3.zhimg.com/v2-6e0e4f5fb8b8af250fd52bea60c80486_r.jpg)

**波间切换的问题：**

在非 Persistent 模式下，GEMM 的 tile 被分成多个 "wave" 执行：

- 当一个 wave 中的所有 CTA 执行完毕后，下一个 wave 的 CTA 才会被调度
- 这导致每个 wave 的 **prologue 和 epilogue 都暴露在关键路径**上
- wave 切换时 Tensor Core 利用率极低

图中展示了这种"阶梯式"执行模式：每个 wave 的 prologue（数据预取）和 epilogue（结果写回）都无法与其他 wave 重叠，造成严重的计算资源浪费。

### 3.12 Tutorial 3：Static Persistent Scheduler

![img-22](https://pic3.zhimg.com/v2-87eeba72c1404cfc5f6964e60d1688b2_r.jpg)

**Persistent Kernel 的核心思想：**

让 CTA 一直驻留在 SM 上，处理完一个 tile 后直接处理下一个 tile，而不是退出后重新调度：

- **节省 (#waves-1) 个 prologue：** 整个 kernel 只需要开始时的一次 prologue
- **Wave i-1 的 epilogue 与 Wave i 的 MMA 重叠：** Warp Specialization 使得上一个 wave 的结果写回可以与下一个 wave 的计算同时进行

**关键约束：**

Warp Specialization 和 Persistent Kernel **必须配合使用**：

- **仅 Warp Specialization：** prologue 和 epilogue 仍然暴露（如 Tutorial 2）
- **仅 Persistent：** epilogue 不容易隐藏
- **两者结合：** 实现最大化重叠，如图中理想时间线所示

**理想执行时间线：** 只有第一个 wave 的 prologue 和最后一个 wave 的 epilogue 暴露，中间所有的 prologue 和 epilogue 都被计算所隐藏。

### 3.13 Tutorial 3：NCU 验证

![img-23](https://pic3.zhimg.com/v2-4e4e82776736462a6de5b148e98a28f6_r.jpg)

**Nsight Compute 性能分析验证：**

通过 NCU 的时间线视图，可以清晰看到 Persistent Kernel 优化的效果：

- **Non-Persistent Kernel：** prologue 和 epilogue 都明确暴露在时间线中，每个 wave 都有明显的非计算开销
- **Persistent Kernel：** 仅第一个 prologue 和最后一个 epilogue 暴露，中间部分计算连续进行

这从微架构层面验证了 Persistent Kernel 的优化效果。

### 3.14 Tutorial 3：性能数据

![img-24](https://pica.zhimg.com/v2-1c6088bddd9431b550ad8ce33e68b7fe_r.jpg)

**Static Persistent Scheduler 的性能收益：**

**大 K 场景（左图）：**

- 2K×2K×2K：0.99x（几乎不变）
- 4K×4K×4K：1.10x
- 8K×8K×8K：1.03x
- 16K×16K×16K：1.00x

**小 K 场景（右图）：**

- 2K×2K×256：0.95x（略有下降，wave 数量不足以摊薄开销）
- 4K×4K×256：1.51x
- 8K×8K×256：1.74x
- 16K×16K×256：**1.84x**

**性能规律：** Persistent Kernel 的收益与 epilogue 在总执行时间中的占比正相关。当 epilogue 占比大时（小 K），收益显著（最高 1.84x）；当 epilogue 占比小时（大 K），收益有限。

### 3.15 Tutorial 4：Preferred 和 Fallback Clusters

![img-25](https://pic2.zhimg.com/v2-7d1483ef7c35aa744c17adb51fc96a7b_r.jpg)

**16K×16K×16K 性能瓶颈分析：**

为什么 Tutorial 3 在 16K×16K×16K 上没有收益？答案是 **L2 缓存命中率低**：

- B200 的 L2 缓存大小约为 120MB
- 16K×16K×16K GEMM 的工作集远超 L2 容量
- 导致 L2 命中率仅 **32.89%**

**大 Cluster 的优势：** 使用更大的 cluster 布局可以缓解 L2 压力：

- **Programmatic Multicast：** 一次 TMA 加载可以将数据存入同一 cluster 内多个 CTA 的共享内存，如图中所示，一次加载 → 多个 SM 受益，大幅减少 L2 流量
- **CTA Rasterization：** 大 cluster 布局配合 CTA 光栅化遍历顺序，提高 L2 命中率

**大 Cluster 的问题：**

量化效应（quantization）可能导致 SM 利用率不足：

- 例如，2×2 cluster 在 18 个 SM 的 GPU 上只能使用 16 个 SM，2 个 SM 空闲

**Blackwell 的解决方案——双 Cluster 策略：**

- **Preferred Cluster（大）：** 获得更高的 TMA multicast 收益
- **Fallback Cluster（小）：** 在 preferred cluster 不满足对齐要求时使用，提升 SM 占用率

### 3.16 Tutorial 4：性能结果

![img-26](https://pic4.zhimg.com/v2-1956122510a2928f6e67f175a7dfe187_r.jpg)

**L2 命中率改善：**

| MxNxK | 无 Preferred Cluster | 有 Preferred Cluster |
| --- | --- | --- |
| 2K×2K×2K | 97.25% | 97.07% |
| 4K×4K×4K | 81.29% | 81.49% |
| 8K×8K×8K | 46.17% | 73.63% |
| 16K×16K×16K | 32.89% | 51.13% |

**性能提升：**

- 2K×2K×2K：0.99x（L2 已经很高，无需优化）
- 4K×4K×4K：1.00x
- 8K×8K×8K：**1.09x**
- 16K×16K×16K：**1.26x**

**关键结论：** Preferred/Fallback Clusters 优化主要针对 L2 缓存受限的大规模 GEMM 场景。8K 和 16K 的 L2 命中率分别从 46%→74% 和 33%→51%，性能提升显著。最终 16K 场景达到约 1597 TFLOPS。

## 第四章：其他代表性优化场景

本章讨论两个特殊但在实际生产中非常重要的优化场景。

### 4.1 代表性场景概览

![img-27](https://pic1.zhimg.com/v2-cc2622206b7df2fc4f27c0f8ccd91e7e_r.jpg)

**两大代表性场景：**

**场景 1 - 多流并行（Multi-stream Parallelism）：**

- 静态 Persistent Scheduler 无法获取 SM 占用信息，可能导致性能下降
- 解决方案：Tutorial 3_1 - 启用 **Dynamic Scheduler（CLC）**

**场景 2 - 短时 Kernel（Kernel with Short Duration）：**

- Prologue 和 Epilogue 占比过大
- 解决方案：尝试将前一个 kernel 的 epilogue 与当前 kernel 的 prologue 重叠
- Tutorial 6：启用 **Programmatic Dependent Launch（PDL）**

### 4.2 Tutorial 3-1：Cluster Launch Control（CLC）

![img-28](https://picx.zhimg.com/v2-ea64f72b338be38c47372a560ffe3f0f_r.jpg)

**三种调度模式对比：**

| 特性 | Legacy Mode | Static Persistent | Dynamic Persistent |
| --- | --- | --- | --- |
| 节省波间切换开销 | ❌ | ✅ | ✅ |
| 抢占（Preemption） | ❌ | ❌ | ✅ |
| 负载均衡 | ❌ | ❌ | ✅ |

**CLC 的核心机制：**

Blackwell 引入的 **Cluster Launch Control（CLC）** 实现了动态调度：

- 每个 SM 尝试取消（cancel）尚未执行的 thread block，直到没有可用工作
- 空闲 SM 可以主动 "窃取" 忙碌 SM 的任务（work stealing）
- 相比 Static Persistent 的固定分配，CLC 能够更好地适应运行时的资源变化

**Static Persistent 的问题：**

Static Persistent Scheduler 在编译时就确定了 tile 到 SM 的映射。当部分 SM 资源不可用时（例如多流并行场景），会导致严重的负载不均衡。

### 4.3 Tutorial 3-1：多流并行场景对比

![img-29](https://pic2.zhimg.com/v2-a084f0b6f8bac5e530bbdf824da2e4b3_r.jpg)

该图展示了两种 Persistent 模式在多流并行场景下的行为差异：

- **Static Persistent：** tile 被静态分配到固定的 SM，当部分 SM 被其他 stream 的 kernel 占用时，分配到那些 SM 上的 tile 只能等待
- **Dynamic Persistent：** 使用 CLC 动态分配，空闲 SM 可以立即获取新的 tile 执行，实现更好的负载均衡

### 4.4 Tutorial 3-1：性能数据

![img-30](https://pic3.zhimg.com/v2-04c8b4b7846bbc9023c2116922ccda48_r.jpg)

**测试配置：**

- 背景 kernel（Background Kernel）占用 20 个 SM
- 前台 kernel（Foreground Kernel）使用剩余的 SM

**从 NSYS 视图可以看到：** 前台 kernel 和背景 kernel 并发执行，前台 kernel 能够利用背景 kernel 未占用的 SM 资源。

**性能对比（Static vs Dynamic）：**

- 2K×2K×2K：1.01x（128 个可用 SM 足以同时容纳所有 tile，无需动态调度）
- 4K×4K×4K：**1.36x**
- 8K×8K×8K：**1.29x**
- 16K×16K×16K：**1.34x**

**实践价值：** 在实际 LLM 推理中，多个 kernel 并发执行非常常见（例如通信与计算重叠）。Dynamic Persistent Scheduler 在这种场景下能够带来 1.3x 以上的性能提升，具有极高的实用价值。

### 4.5 Tutorial 6：PDL 原理

![img-31](https://pic1.zhimg.com/v2-cde5c42e68d361284df8612e8c5ad3d0_r.jpg)

**Programmatic Dependent Launch（PDL）的核心原理：**

PDL 是一种使同一 stream 中的依赖 kernel **实现重叠执行**的技术。

**适用场景：** 当 kernel 的 prologue 和 epilogue 主导执行时间时（即短时 kernel），通过提前启动次级 kernel 的 prologue 并与主 kernel 重叠，可以有效提升性能。

**执行模型（如图所示）：**

- **Without PDL：** primary kernel → secondary kernel → other kernel，串行执行
- **With PDL：** primary kernel 的 epilogue 与 secondary kernel 的 prologue 重叠

传统 CUDA stream 语义要求同一 stream 内的 kernel 严格按序执行。PDL 打破了这一限制，允许在前一个 kernel 接近完成时就启动下一个 kernel 的初始化工作。

### 4.6 Tutorial 6：PDL 实现

![img-32](https://pic4.zhimg.com/v2-9d974809d2da0924aad50fac64e58257_r.jpg)

**实现 PDL 需要 3 个关键步骤：**

**Step 1 - Primary Kernel 中声明可以启动后续 kernel：**

```python
@cute.kernel()
def primary_kernel(...):
    # prologue & mainloop
    ...
    # 告知 runtime 可以提前启动下一个 kernel
    cute.arch.griddepcontrol_launch_dependents()
    # epilogue
    ...
```

`griddepcontrol_launch_dependents()` 是一个**集体操作**——当所有 block 都执行到这个 API 时，runtime 才会实际启动下一个 kernel。

**Step 2 - Secondary Kernel 中等待前序 kernel 完成：**

```python
@cute.kernel()
def secondary_kernel(...):
    # prologue
    ...
    # 等待前序 kernel 完成且内存操作可见
    cute.arch.griddepcontrol_wait()
    # mainloop & epilogue
    ...
```

`griddepcontrol_wait()` 确保了数据一致性——secondary kernel 的 mainloop 不会在 primary kernel 的内存操作完全可见之前开始。

**Step 3 - Launch 时启用 PDL：**

```python
kernel(...).launch(
    grid=grid,
    block=[threads, 1, 1],
    stream=stream,
    use_pdl=True,  # 关键参数
)
```

### 4.7 Tutorial 6：PDL 应用设计

![img-33](https://pic2.zhimg.com/v2-c882d00e0a45092bb6b0559cc23a3bdf_r.jpg)

**实际应用案例：Dequantization + GEMM**

本 Tutorial 设计了一个实际的 PDL 应用场景：

**两个 Kernel 的流水线：**

1. **Dequantization Kernel（Primary）：**
2. Prologue：加载量化的 B 矩阵（i8 格式）
3. Mainloop：执行反量化操作
4. Epilogue：将 fp16 格式的 B 矩阵写回 + 调用 `griddepcontrol_launch_dependents()`
5. **GEMM Kernel（Secondary）：**
6. Preamble/Prologue：预加载 A 矩阵等准备工作
7. Dependency：调用 `griddepcontrol_wait()` 等待 fp16 B 可见
8. Mainloop + Epilogue：执行标准 GEMM

**重叠效果：**

- GEMM Kernel 的 prologue（加载 A 矩阵、初始化）与 Dequantization Kernel 的最后阶段重叠
- 整体端到端延迟 = Dequant 全部 + GEMM 除去 prologue 的部分

### 4.8 Tutorial 6：PDL 性能

![img-34](https://picx.zhimg.com/v2-190de1f1f80dffa651fb2625150753e3_r.jpg)

**NSYS 视图验证：**

从 NSYS 的时间线视图中，在 mnk=256×16384×128 的场景下，可以清楚看到两个 kernel 的重叠执行。

**性能数据（Dequantization + GEMM 总时间）：**

- 256×16K×64：**1.12x** speedup
- 256×16K×128：**1.16x** speedup
- 256×16K×256：**1.13x** speedup
- 256×16K×512：**1.10x** speedup

**适用场景分析：** PDL 在以下条件下效果最佳：(1) Kernel 执行时间短；(2) Prologue 占比较大；(3) 有明确的数据依赖关系。在 LLM 推理中，量化推理场景下的 Dequant+GEMM 正是典型应用。

## 第五章：总结

### 5.1 优化技术全景总结

![img-35](https://pic3.zhimg.com/v2-1cc1ff61a6802674dcf8737cac44c84a_r.jpg)

**CUTLASS Python 的定位：**

CUTLASS Python 是一个强大的 kernel 编写工具，能够达到峰值 Tensor Core 性能。它提供了一系列面向初学者的 GEMM 教程，从一个仅需 200-300 行的基础 GEMM 开始逐步深入。

**计算密集型场景优化总结（基线：1640 TFLOPS @ 1.35GHz）：**

| 优化技术 | 原理 | 达到 TFLOPS |
| --- | --- | --- |
| 2CTA + Multicast | 2CTA 减少 B tensor SMEM 大小以允许更大的 ab stage；TMA multicast 减少 L2 流量 | 1393 |
| Warp Specialization + TMA STG | CTA 内 warp 间任务并行；指令可以更高效地发射 | 1429 |
| Persistent Scheduler | Persistent cluster 在 kernel 执行期间驻留 GPU，处理多个 tile，隐藏 prologue/epilogue 开销 | 1468 |
| Preferred & Fallback Clusters | 大 cluster 获得更高 TMA multicast + 小 cluster 填充设备 | 1597 |

**其他代表性场景：**

- **Dynamic Persistent Scheduler（CLC）：** 用于多流并行，最高 **1.36x** 加速
- **PDL：** 用于 kernel 重叠执行，最高 **1.16x** 加速

### 5.2 技术演进全景

![img-36](https://picx.zhimg.com/v2-331ca66bd6061d19da20699a57009c5f_r.jpg)

**推荐参考资源：**

- "Programming Blackwell Tensor Cores with CUTLASS" - 深入 Blackwell Tensor Core 编程
- "Enable Tensor Core Programming in Python with CUTLASS 4.0" - CUTLASS 4.0 Python 接口详解

这些资源提供了更深层次的技术细节，建议有兴趣的读者进一步阅读。
