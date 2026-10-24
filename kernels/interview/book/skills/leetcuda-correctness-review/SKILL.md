---
name: leetcuda-correctness-review
description: 对 LeetCUDA 技术书做“第零层（数理逻辑）+ 四层”正确性自审：先验推导链与量词/蕴含，再查定义/指称、API/硬件契约、资源/类型计数、表达一致性。当任务涉及修改 chapters/*.tex 后的自审、审阅 CuTe/CUDA/PTX 事实断言、排查“公式对了但名词/作用域/寄存器口径错了”、检查图注/表格/面试问答与正文冲突、commit 或发布前验收、复核“零拷贝/必然/所有/唯一/最优/寄存器/共享/周期”等高风险表述时使用。产出逐条 claim ledger、边界矩阵、回修清单和残余风险，不以“推导自洽”代替“指称正确”。书稿改动后强制编译所有受影响 PDF 入口（主 book、cute-zhihu 等）并跑 ReadTheDocs 三层验收，不能只编主 book。
user-invocable: true
---

# LeetCUDA 第零层 + 四层正确性自审

## 0. 定位书根

按以下顺序定位 LeetCUDA 仓库根，记为 `<LeetCUDA_DIR>`：

1. 环境变量 `LEETCUDA_DIR`
2. 当前目录或父目录下的 `LeetCUDA/kernels/interview/book`
3. `~/LeetCUDA/kernels/interview/book`
4. 当前工作区中唯一含 `kernels/interview/book/chapters/` 的目录

书根：

```text
<LeetCUDA_DIR>/kernels/interview/book
```

本 skill 只审书稿，不改冻结源码。源码问题记录到 `book/CHECKLOG.md` 或正文勘误，
不在 `.cuh` / `bench/*.cu` 上直接修。

## 1. 何时使用

在以下场景强制加载：

- 修改 `book/chapters/chNN*.tex`、图、表、面试问答之后；
- 修改 `book/cute-zhihu/`、`book/appendices/` 或任何独立 PDF 入口之后；
- 新增章节、增补章、重构某段类型/算法解释之后；
- 每轮 review 产生书稿改动、准备宣告完成之前（必须连带跑所有受影响 PDF + RTD）；
- commit、发布、生成 PDF、同步 ReadTheDocs 之前；
- 审阅 CuTe / CUDA / PTX / 性能口径断言；
- 用户要求“自审”“复核”“逐项检查改前改后”。

不适用：

- 只改错别字、纯排版、无技术断言的文字；
- 需要 GPU 实测才能判定的性能数字。此时只能标注“未复测”，不能伪造结论。

## 2. 核心原则

先过第零层，再区分四类断言：

| 层 | 类型 | 例子 | 需要的证据 |
|---|---|---|---|
| L0 | 数理逻辑 | 从前提 P 到结论 Q 每一步是否成立；`A⇒B` 还是 `B⇒A`；全称/存在；边界/反例 | 逐步推导、证明、反例 |
| L1 | 定义/指称 | `size`/`cosize` 的定义、`P` 到底指哪个对象、tile 边界 | 定义、索引、术语表 |
| L2 | API/硬件契约 | `wait_group N` 等到什么、`_1{}` 是什么 | 头文件、PTX、源码调用点 |
| L3 | 资源/类型计数 | 256 个逻辑值占多少寄存器、P 是否新 fragment | 单位表、dtype、生命周期、源码 |
| L4 | 表达一致性 | 图注是否与正文冲突、问答是否沿用旧结论 | 全书交叉检索、术语表 |

**第零层不通过，直接停。** 它回答“给定这些前提，结论是否真的推得出来”；
第一层回答“前提、定义、指称本身是否正确”。两层都过只说明数学内部闭合，
后三层必须去找外部事实，不能继续在同一套叙述里反复读。遇到“公式没错但
读起来顺”的句子，优先怀疑推导跳步、量词方向、名词指称、作用域或单位。

## 2.1 当前正确性状态与审核者校准（2026-10-25 快照）

**这不是永久结论，也不是“零错误”证明。** 它只描述当前工作树（R28）在
第零层 + 四层审查下的置信状态；书稿再改、源码/版本变化后必须重审。

| 文档 | 总体判断 | 最强 | 最弱 | 不能宣称 |
|---|---|---|---|---|
| 主 book | 结构性正确性高；发布候选/高质量教学稿 | L0 核心推导、L1 术语、L2 源码/PTX 锚定 | 性能数字、少数 medium 推导/归因 | 零错误；所有性能已复测；形式化证明 |
| cute-zhihu | 概念正确性中高；高质量学习资料/参考级 | layout/TMA/mbarrier/proxy 叙事 | 文章特定代码/表格/术语；未逐页 PDF 验收 | CUTLASS 规范；所有代码可编译；零错误 |

当前构建状态（只证明“产物能生成/页面可渲染”，不证明内容正确）：

```text
book.pdf：588 页，0 error / 0 missing，Overfull 16
cute-zhihu.pdf：248 页，0 error / 0 missing，Overfull 6
RTD：英文图集 1875 条 / 1792 单元，代码 771/771，公式 121/121
visual_check：93 页，需处理 0
```

### 审核者的四条校准规则

1. **先对齐前提、定义域和版本。** 换掉原句的“每行非空 / 首块非空 / Bc=128 /
   sm_80 / CUTLASS 版本”等前提再造反例，反例不算数；先问“这句话在它的
   上下文里是否成立”。
2. **举证责任在主张错误的人。** 判 false 必须有源码行、PTX 段落、逐步反例
   或可复现实验；找不到证据只能标 `medium / 未验证`，不能直接改书。
3. **旧 PASS 和 agent high 都不是证明。** 历史构建通过只证明产物能生成；
   agent 的 high 只是候选，必须由主 agent 回源码/PTX 复核。反过来，
   “未发现错误”也不等于“已证明正确”。
4. **不要为了统一而改。** 同一术语可能在不同章节有合法的不同口径；先判
   “同一断言”还是“不同断言”。只有同一断言才同步；不同断言保留并注明。

### 已知容易误判的点（false-positive guard）

- **ch18 的 `β=e^{-∞-m0}=0`**：该段是 `Tc=1`、WG1 为空、WG0 非空的
  dense 路径，`m0` 有限；把 `m0=-∞` 套进来是换了前提，不能判错。真正的
  全 mask 行由空行分支处理。
- **“相对步长” vs “相对误差上界”**：e4m3 的 `2^{-3}` 是 binade 下沿的相对
  ULP，`2^{-4}` 是 RN 相对误差上界；这是术语/量纲口径，不是算术错误。
  不要把 `2^{-4}` 的结论当成 `2^{-3}` 的步长来判错。
- **`P` 的 zero-copy / `convert_type`**：布局重解释与 dtype 转换是两件事。
  说“布局层 zero-copy”不等于说“数值转换不产生新 fragment”；不要把两者
  混成一句判错。
- **未复测的性能数字**：静态表里的 TFLOPS/ms/spill/speedup 没有被复测时，
  只能标“未复测”，不能因为“和我预期不同”判 false。
- **`agent` 的 high 候选**：像 ch18 这种“看起来会 NaN、但前提有限”的句子，
  必须回到定义域；不能因为 agent 说 high 就改。

### 当前残余 medium（不要重复当 high 报）

- 主 book：ch17/ch18 的 129/257 arrive 计数归因；ch26c 的 VCPB STSM 指令
  计数；ch27 “四个误差源”只展开三项；ch33 smooth-v +1.0% 的机制归因；
  ch36 误差预算表的 3.24%/10.17% 合成口径。
- cute-zhihu：`reed-02` product 表的 logical 行读法；`zhuxi-04`
  “tiled_product 在 cutlass 没被用到”的版本相关断言；`zhuxi-05` 8-bit
  字段与 ≤256 的口径；`zhuxi-03` “一一映射/单射”术语。

这些是“证据不足/口径未定”，不是“已确认错误”。下一轮要么补证据改，
要么继续保留 medium；不能为了清零而强行改。

## 3. 标准工作流

### Step 1：建立改动清单

先收集范围，不急着读结论：

```bash
cd <LeetCUDA_DIR>
git diff --name-only -- kernels/interview/book/chapters
git diff --check -- kernels/interview/book/chapters
git --no-pager diff -U2 -- <changed.tex>
```

对每个 hunk 登记：

- 文件与行号；
- 原句与新句；
- 它属于哪一层；
- 原句错在“算错”“说过头”“名词错”“单位错”还是“图注残留”。

图注、表格、面试问答、section title 必须和图/正文一起进清单，不能当附属品。

### Step 2：写 claim ledger

每条技术断言一行，模板见
`references/claim-ledger-template.md`。最低字段：

```text
ID | 文件:行 | 断言 | 层级 | 类型 | 证据 | 判定 | 处置
```

证据等级从高到低：

1. 定义/定理与反例；
2. CUTLASS/CuTe 头文件、PTX ISA、CUDA 文档的具体段落；
3. 被审源码的调用点；
4. host-only 编译探针 / `static_assert`；
5. GPU 测试日志（只能证明该 case 跑通，不能证明解释正确）；
6. 无证据（标 `未验证`）。

没有证据的断言不能写成“已确认”。

### Step 3：第零层 + 四层审查

按顺序做。第零层没闭合，不进入第一层；第一层没闭合，不进入第二层。

---

## 第零层：数理逻辑正确性（最高优先级）

**目标**：给定书中自己写出的前提和定义，结论是否真的由它们推出；每一步
等号、不等号、量词、分类、归纳是否成立。它不负责“定义是否符合硬件”（那是
第一/二层），只负责**推导链有没有断**。

**操作**：

1. 把段落改写成 `P1, P2, ... ⇒ Q`；中间步骤逐条编号，禁止从第一行直接跳到
   结论。
2. 逐项检查代数：移项、代入、约分、消去、取对数/指数、求和/期望/方差、
   归一化；分母非零、开方非负、log 定义域。
3. 检查量词与蕴含方向：`A⇒B` 不自动给 `B⇒A`；`所有`/`存在`；充分/必要/
   充要；全称断言必须给证明或反例。
4. 检查证明结构：归纳基例/归纳步、分类是否完备、反证是否真的矛盾、有没有
   循环论证或隐藏前提。
5. 区分恒等、近似、上界、经验值；数值代入通过不等于恒等成立。
6. 找不到反例只能写“未找到反例”，不能写“已证明”。

**具体示例**：

- 坏：`cosize == size`，所以布局无空洞、是双射。
  反例：`(2,2,2):(1,1,5)` 的 `size == cosize == 8`，像却是
  `{0,1,2,5,6,7}`，有重复、缺 3 和 4。
  改：只有单射且非负 stride 的 compact 布局才有 `cosize == size`；反过来不成立。
- 坏：`O = Σ_c O_c`，所以 Phase 4 要把所有 chunk 加起来。
  实际：每个 `O_c` 是输出 D 维上互不相交的列块，Phase 4 各自归一化写回。
  证据：`ffpa_attn.cuh` Phase 4 的 `local_tile(..., v_chunk)` 与逐块写回。
- 坏：`对称 (S,S) 在任何 D 都取不到最优`。
  实际：表格里 D=384/512 的最优恰好是 `(2,2)`；应写成“不存在一个固定对称组合
  在所有 D 最优”，并列出分段表。
- 坏：`每行至少一个有限分数 ⇒ 首轮 rescale 不会 NaN`。
  反例：该行首个被处理的块整块 mask，`-∞-(-∞)=NaN`。前提只保证“存在有限
  分数”，不保证“首块非空”；必须补“首块非空/跳过空前缀/有限哨兵”。
- 坏：`D=128 时 phase 每 1 个 kv_tile 翻转；D=192 时周期 2`。
  实际周期是 `lcm(C, 2·Sk)/C`：D=128→C=2→2，D=192→C=3→4。
  这是算术/分类推导错，不是图注笔误。

**通过标准**：每个 `⇒/iff/所有/存在` 都有逐步依据；全称断言有证明或反例；
恒等/近似/上界已分开；没有“从前提跳不到结论”的句子。

---

## 第一层：定义、指称与边界

**目标**：术语是否有定义，指称是否落到正确对象，定义域、索引和边界是否完整。
它**不重复**第零层的逐步推导，只检查推导所使用的“词”和“对象”对不对。

**操作**：

1. 给每个术语定义；区分同名不同物（逻辑值/寄存器/fragment；ULP/误差上界；
   layout/atom/pointer；layout 重解释/dtype 转换）。
2. 检查主语和代词：这句话说的是 tensor、fragment、指令、线程、buffer 还是
   整个 CTA。
3. 检查索引/边界：0/1 基、inclusive/exclusive、shape/stride、坐标→offset、
   tile 边界、尾块。
4. 填边界矩阵：0、1、负 stride、混合符号 stride、不整除、尾部 stage、
   全 mask、非 2 的幂、空 tile。
5. 找不到反例不等于证明；把“未找到”和“已证明”分开写。

**具体示例**：

- 坏：`P` 是原 f32 寄存器的零拷贝重解释。
  实际：`convert_type` 做数值转换并生成新的 f16 fragment；随后的
  `convert_layout_acc_Aregs` 才是不搬数据的布局重解释。免掉的是 smem 往返，
  不是类型转换本身。
- 坏：`P 的 smem 往返成本随 chunk 数 C 增长`。
  实际：P 的转换发生在完整 QK+softmax 之后，每个 `kv_tile` 一次；C 是 D 方向
  chunk 数，不乘这笔成本。
- 坏：`logical_divide` 前后 Size 相同。
  实际：只有 tiler 整除时相同；不整除时 Rest 取 ceil，`(10):(1) / (4):(1)`
  的 Size 是 12。先补“整除”前提，再谈等式。

**通过标准**：所有术语有定义、指称唯一；定义域/边界表已填；没有把不同对象
混名，也没有把“布局重解释”和“数值转换”当成同一件事。

---

## 第二层：API 与硬件契约

**目标**：每个函数名、指令名、参数、架构门槛、默认行为都能指到头文件/PTX/调用点。

**操作**：

1. 查函数签名和参数顺序，不靠名字猜。
2. 查参数单位：元素、字节、bit、stage、cluster、thread、value。
3. 查作用域：本线程 / warp / CTA / cluster / gmem。
4. 查完成语义：wait 等到什么、谁来 arrive、相位怎么翻。
5. 查默认值和目标架构：OOB fill、swizzle、alignment、sm90/sm120 分支。
6. 查源码调用点：书里说的参数是否真是这个参数。

**具体示例**：

- 坏：`cp.async.wait_group N` 等“恰好 N 组未完成”，尾部会等一个不存在的组。
  实际：它等到**最多只有最近 N 组**仍未完成；尾部只剩 V 组时误用 `wait<1>`，
  允许这组 V 仍未完成，可能读到未就绪的 V。证据：PTX ISA
  `cp.async.wait_group` 语义 + `flash_attn.cuh` 的尾部 `wait<0>` 分支。
- 坏：`make_tma_copy(..., _1{})` 的 `_1{}` 是“沿 minor 维一次搬满”。
  实际：它是第五个参数 `cluster_size`，此处为 1。box 几何由 `CTA_Tiler` 给出。
  证据：`copy_traits_sm90_tma.hpp` 的 `make_tma_copy` 签名与注释。
- 坏：`partition_fragment_B(sV_stg)` 划分了 smem 数据指针。
  实际：它建立每线程私有的寄存器 fragment 布局；S2R 的 smem 地址由
  `partition_S(sVt_stg)` 提供。证据：`ffpa_attn.cuh` L341-345 与 CuTe 返回类型。
- 坏：TMA 对 OOB 行“不写 smem，保留旧值”。
  实际：TMA descriptor 默认 zero-fill，越界元素写 0；mask 仍必要，因为 0 不是
  `-inf`。证据：PTX TMA fill mode 0 + CUTLASS `OOB_FILL_NONE` 路径。
- 坏：`cute::gemm` 不处理 K，K 方向必须手写。
  实际：对 `(V,M,K)×(V,N,K)` fragment，`gemm` 会展开 `MMA_K`；手写的是外层
  K-tile 搬运和寄存器双缓冲。证据：`cute/algorithm/gemm.hpp`。

**通过标准**：每条 API/硬件断言都有 `文件:行` 或 `PTX 章节`；架构/版本限定写清。

---

## 第三层：资源与类型计数

**目标**：把“逻辑值、寄存器、fragment、smem 字节、CTA 总副本”分开算，禁止混用。

**操作**：

1. 建表：对象 | dtype | 逻辑值/线程 | 寄存器/线程 | smem 字节 | CTA 总副本 | 生命周期。
2. 判断“每线程”还是“CTA 总计”。
3. 判断“数值转换产生新 fragment”还是“只重解释布局”。
4. 判断“共享”还是“每个 warp 各一份私有副本”。
5. 判断“每个 chunk 一次”还是“每个 KV tile 一次”。
6. 对照硬上限：255 个 32-bit 寄存器/线程、smem opt-in 上限、TMA 16B 原子等。

**具体示例**：

- 坏：`8 chunk × 32 = 256` 是“可用寄存器的一半”。
  实际：已经是 256 个 f32 逻辑值，超过每线程 255 个 32-bit 寄存器硬上限，
  再计临时量必然 spill。
- 坏：`P` 是原 f32 寄存器的零拷贝重解释。
  实际：`convert_type` 做数值转换并生成新的 f16 fragment；随后的
  `convert_layout_acc_Aregs` 才是不搬数据的布局重解释。免掉的是 smem 往返，
  不是类型转换本身。
- 坏：V 的 B fragment 是“共享寄存器”。
  实际：每个 M-warp 各有一份私有副本，增加的是 CTA 总寄存器成本；不能按
  “共享一份”估算每线程压力。
- 坏：D=320、默认 `(2,2)` 时“smem 只够一个 O 分量，所以 VCPB=1”。
  实际：池能放 4 个 `128×64` 分量；但批必须整除 `kDChunksV=5`，可选 1 或 5，
  5 个需要 80 KB、预算只有 64 KB，所以退到 VCPB=1。证据：
  `compute_vchunks_per_batch` 的整除与容量双条件。

**通过标准**：所有“寄存器/共享/一半/总量/周期”都能在资源表里找到一行，且单位一致。

---

## 第四层：表达一致性

**目标**：正文、图注、表、问答、标题、术语使用同一套事实，不留下第二套旧说法。

**操作**：

1. 维护冻结术语表：逻辑值、寄存器、fragment、smem view、copy atom、stage、
   per-call、CTA 总量、私有副本。
2. 把图注、表标题、面试问答当正文审，不因为“只是说明”就跳过。
3. 对修改点做同义/反义检索：`零拷贝`、`原地`、`不新增`、`共享`、`私有`、
   `总量`、`周期`、`只够`、`只接受`、`消除`、`最优`。
4. 对流水图，按源码调用顺序重画：prefill 顺序、主循环顺序、consumer 顺序、
   stage 覆盖关系。
5. 检查 label/ref、旧 equation label、旧章节引用是否还有指向。

**具体示例**：

- 坏：正文改成“`convert_type` 生成新 f16 fragment”，图 26-1 仍写
  “Phase 2 就地覆盖，不新增”。
  改：图与图注统一为“softmax 写回 S；`convert_type` 生成新 P”。
- 坏：图 21 写“蓝色两行 = 第 1 轮”，但四个线程每轮共同参与。
  改：“蓝色四行 = 四个线程全体；第 1 轮覆盖数据行 m=0,1，第 2 轮覆盖 m=2,3”。
- 坏：26b 流水图把“K1 复写 stage 0”画成依赖。
  实际：S=2 时复写 stage 0 的是 K2/V2，且要等 QK0/PV0 的 empty。
  证据：`flash_attn.cuh` 的预取位置与 `k_empty`/`v_empty` 相位。
- 坏：persistent 的收益写成“Q 跨 q-tile 持久”。
  实际：Q 每个 q-tile 仍要重新装载；收益是减少 launch/CTA 调度、复用 CTA 内
  屏障与流水状态。

**通过标准**：全书检索高风险术语只剩一种口径；图/表/问答与正文不冲突；
旧 label 没有悬空引用。

### Step 4：全量轮次的分片复核（R25/R26 经验）

单轮通读容易只覆盖“主结论”。全量轮次（主书 + cute-zhihu）建议分片并行，
但**判定权必须留在主 agent**：

1. **分片**：主书按 `ch00–14 / ch15–19b+附录 / ch20–26c / ch27–37` 四片；
   cute-zhihu 按 `reed-* / zhuxi-* + frankshi` 两片。
2. **给 agent 的约束**：只读、不改文件；先跑 `scan_risky_claims.py`；逐章覆盖
   正文、图注、表、问答、listing；每条 high 必须带 `文件:行 + 外部证据 +
   最小修法`；预先告知旧轮已修项，避免把旧账当新发现。
3. **主 agent 复核**：agent 的 high 逐条回源码/PTX/头文件，确认后才改；
   medium 只记录，不为了“看起来统一”强行改。误报实例：ch18 的
   `β=e^{-∞-m0}=0` 看似 NaN，但 dense 路径 `m0` 有限，原句正确，不能改。
4. **按断言合并**：把不同 agent 命中的“同一断言的不同出现位置”合并成一条，
   不要按文件去重；否则正文、图注、表注、问答会各修一半。
5. **记录来源**：在 claim ledger 处置列旁标 `agent-A high → 主 agent 复核`、
   `机械扫描 → 人工确认`；未复核候选不能进回修清单。

### Step 5：修复传播与四表面检查

一条结论通常至少出现在五个表面：**正文/公式、图与图注、表与表注、
listing/代码注释、面试问答/标题**。任何一处修完，必须做同源检索：

```bash
# 用原句里最稳定的关键词/公式片段/数字，不要只搜整句
rg -n '<关键词>|<公式片段>|<数字>' kernels/interview/book/chapters \
  kernels/interview/book/cute-zhihu/chapters
```

对每个命中判“同一断言”还是“不同断言”：

- 同一断言 → 同步修，或在 ledger 里写清“该处保留的另一种合法口径”；
- 不同断言 → 不要为了统一而改；
- 证据不足 → 标 medium，写入 `audit-history.md`，不要硬改。

R25 的典型传播缺口：R22 只改了 ch27 图注，正文仍写“无额外数学误差源”；
ch15 只改了定理/remark，坑二仍把 NaN 归到最终除法；ch09 只改了部分表述，
实测表行仍叫 `naive`。这些都不是新事实，而是**修复没有扩散到同源表面**。

## 4. 边界矩阵

先过第零层，再用边界矩阵服务 L0/L1：L0 用它找反例推翻全称断言，L1 用它补
定义域和尾块。每次至少填一张，按对象选择：

```text
对象 | 普通 | 尾部/最后 tile | 全 mask/空 | 不整除 | 符号/stride 异常 | 架构/版本边界
```

典型行：

- layout：`size/cosize`、负 stride、混合符号、`compact` 反例；
- cp.async：`kStagesK=1`、尾部、只剩 V 组；
- TMA：32/64/128B swizzle、M=4/5/6、OOB、cluster size；
- 寄存器：D=320/384/448/512、F16/F32 累加、spill；
- mask：整行 `-inf`、整个 q-tile `Tc_eff=0`、Nkv<Nq；
- persistent：等时 tile、causal 反向、tile 数不整除 CTA 数。

## 5. 验证（按改动路径构建所有受影响入口）

先做静态检查，定位问题：

```bash
cd <LeetCUDA_DIR>/kernels/interview/book
git diff --check
# IDE/语言诊断：read_lints
rg -n '零拷贝|原地|不新增|共享寄存器|唯一手段|消除.*尾损|任何.*最优|必然.*0|可用寄存器的一半|不碰 K' \
  chapters cute-zhihu/chapters
python3 skills/leetcuda-correctness-review/scripts/scan_risky_claims.py \
  chapters appendices
```

**静态检查不是验收，主 book 的 PDF 也不等于全部书稿。** 每轮 review 只要
改动任一书稿入口（正文、图注、表、TikZ、封面、独立分册、RTD 词典或转换器），
就必须先判定构建矩阵，再编译**所有受影响入口**，不能只跑主 `book/build.sh`。

### 5.0 先判定构建矩阵

| 改动路径 | 必须构建/验收 |
|---|---|
| `book/chapters/`、`book/appendices/`、`book/book.tex`、主 book 图/封面 | `book/build.sh` |
| `book/cute-zhihu/chapters/`、`cute-zhihu.tex`、`cute-zhihu/figures/`、`cute-zhihu/cover-tikz.tex` | `cd book/cute-zhihu && bash build.sh` |
| `docs/readthedocs/**`（转换器、模板、词典、CSS） | RTD `build.sh` + `visual_check` |
| 同时改了主 book 与 cute-zhihu | 两个 PDF 都编 + RTD 全量 |

`cute-zhihu.pdf` 是独立 PDF 入口，不会被主 `book/build.sh` 编译；主 book 的
`book.pdf` 也不会被 `cute-zhihu/build.sh` 编译。改哪个入口就编哪个，两个都改
就都编；RTD 合集包含两者，只要有一处内容改动，最终仍要跑 RTD 全量。

### 5.1 PDF 构建闸门（所有受影响 PDF）

**主 book：**

```bash
cd <LeetCUDA_DIR>/kernels/interview/book
./build.sh
```

**cute-zhihu 分册：**

```bash
cd <LeetCUDA_DIR>/kernels/interview/book/cute-zhihu
bash build.sh
```

两个入口的通过标准相同：

- 退出码 0；
- 0 条 `^! ` LaTeX error；
- 0 条 undefined reference / undefined citation；
- `Missing characters: 0`；
- 记录 `Overfull` 数量与页数；
- 用 `pdftotext` 抽查本轮改动的关键句/公式确实进入 PDF（不是只编了旧源）。

注意 `cute-zhihu/build.sh` 只打印 `errors/missingchar/overfull/pages`，xelatex
即使有 `!` 也可能返回 0；必须自己 grep `.tmp_build2.log`，不能只看脚本退出码。

页数相对上一轮的变化本身不是错误：补全 listing、增加图内文字都可能增页；
要解释变化来源，而不是只报一个数字。`cute-zhihu.pdf` 是入库文件，重编后要
把新 PDF 一起纳入改动；`cover-tikz.pdf` 按仓库约定不入库。

### 5.2 ReadTheDocs 三层验收闸门

```bash
conda activate cdit          # 或 conda run -n cdit ...
cd <LeetCUDA_DIR>/docs/readthedocs
pip install -r requirements.txt          # 首次或依赖变更后
python -m playwright install chromium     # 首次跑 visual_check 前
./build.sh                   # convert → sphinx-build → convert.verify
python tools/visual_check.py --html build/html --out build/shots/chapters
# 需要代理时追加 --proxy http://localhost:7890
```

通过标准：

1. `./build.sh` 内的 `convert.verify` 全项通过（文件级）；
2. `visual_check` 无 MathJax 报错、无可见字面残迹、无控制台报错、无 404；
3. 逐页截图/接触表检查图片、表格、过宽公式没有撑出正文栏（浏览器级 + 版式几何）。

**最终验收必须全量**：不得用 `--skip-tikz`、`--only` 或复用旧 SVG 冒充验收；
这些只允许用于中间调试。任一项失败，本轮状态只能是“待修”，不能宣称完成。
push/PR 仍由用户决定，但本地 build + RTD 验收不能因此省略。

### 5.3 图与封面的额外必做项

- 改了 TikZ 图内中文文字或新增含中文图：**先跑一次全量 RTD build**，让它报
  “英文图集 17xx/1792”，再跑
  `python -m tools.translate_figures --proxy ...` 补
  `i18n/figures-en.json`，然后重跑全量 build；否则 RTD `post_build` 的英文图集
  核对必红。不要只跑 `translate_figures.py --report`：它读的是旧的
  `build/tmp/tikz`，图刚改过时会假绿。
- 图内文字换措辞会换 key；新增条目要和书稿一起提交，不能只更新 TikZ 源。
- 改了主 book 封面文案：重编译 `figures/misc/cover.pdf`；英文封面另跑
  `python -m tools.build_cover_en`，更新 `_static/figures-en/cover.png`。
- 改了 cute-zhihu 封面：重编 `cute-zhihu/cover-tikz.tex` 后再编分册；
  `cover-tikz.pdf` 不入库，但它是分册 PDF 的输入，不能拿旧的冒充。

### 5.4 记录与收尾

自审结束必须给出：

1. 通过项；
2. 自审新发现并回修的项；
3. 未验证项与残余风险；
4. 改动文件清单与 `git diff --check` / lint 结果；
5. 每个受影响 PDF 的结果：`book.pdf`、`cute-zhihu.pdf`（分别列页数、
   Overfull、error/undefined/missing）；
6. RTD `build.sh` + `visual_check` 结果（verify 项、MathJax/console/404/版式）。

然后把本轮记录**追加**到 `references/audit-history.md`（不要覆盖旧轮次），
写清日期、范围、方法、发现、修正、证据、所有受影响 PDF 与 RTD 的验收结果。
若本轮确实没有书稿改动（例如只改 skill 文档），必须明确写
“本轮无书稿改动，未触发任何 PDF/RTD 闸门”，不能假装跑过。

## 6. 高风险词与停止条件

出现以下词，暂停并追问“证据是什么”：

```text
零拷贝、原地、不新增、共享、私有、唯一、必然、所有、任何、恰好、等价、
只要…就、一半、只够、只接受、只覆盖、消除、最优、无损、完全重叠、无额外、
相对步长、相对误差上界、STG.128、归约、Size 相同、不重不漏、一次搬满、
因此、所以、由此可得、可推出、代入、约去、消去、当且仅当、充分、必要、充要
```

高风险句型：

- “因此/所以/由此可得” → 把中间步骤补出来，检查有没有跳步；
- “当且仅当/充分必要” → 两个方向都验，不能只证一边；
- “代入/约去/消去” → 检查分母、定义域、是否真的同量；
- “上界/近似/经验值” → 不能写成“恒等/精确”；
- “假设…则…” → 检查结论是否真的用到该假设，还是只用到更弱的前提；
- “A 就等于 B” → 检查双向；
- “所有/任何” → 找边界与反例；
- “只/唯一” → 找替代路径；
- “寄存器/共享/总量” → 查单位与归属；
- “零拷贝/原地” → 查是否产生新 dtype fragment；
- “N 个值一次 STG.128” → 先算 `N × sizeof(dtype)`，再选指令宽度；
- “divide 前后 Size 相同 / 不重不漏” → 查 tiler 是否整除，不整除时 Rest 取 ceil；
- “相对步长” → 问清是 `ULP/值` 还是 `RN 相对误差上界`，两者在 binade 内差 2 倍；
- “listing 行区间” → 查是否截在表达式/分支/`make_tensor` 中间，标题行号是否与 range 一致；
- “周期/图样” → 查完整 XOR/相位周期，不按单行猜；
- “消除尾损/达到上限” → 查 makespan 与吞吐模型。

这些词不是禁词，而是**必须带证据的断言**。

## 7. 输出模板

```text
自审范围：<文件/commit/章节>
改动清单：<N hunks，含图注/问答>
第零层结论：<推导链全过 / 发现 N 处跳步、反向蕴含或反例>
结论：<通过项> / <新发现问题> / <未验证项>

claim ledger：
| ID | 位置 | 断言 | 层级 | 证据 | 判定 | 处置 |

同源传播：
- 检索词：<关键词/公式/数字> → <命中位置> → <已同步/保留 medium/证据不足>

agent 候选（如有）：
- high N 条 → 复核通过 X 条、否决 Y 条（否决理由）

逐章结果：
- 第 N 章：L0 数理逻辑：...；L1 定义/指称：...；L2 API/硬件：...；L3 资源/类型：...；L4 图注/问答：...
- 图/表/问答：...

回修：
- <原句> → <新句>（依据；L0/L1/L2/L3/L4）

验证：
- git diff --check：...
- lint：...
- 受影响 PDF：book.pdf / cute-zhihu.pdf（逐个列 exit / pages / Overfull / error / undefined / missing）
- RTD build.sh + convert.verify：<pass/fail，失败项>
- visual_check：<MathJax / console / 404 / 版式>
- GPU：未做/已做（与 PDF/RTD 验收分开记）
残余风险：...
```

## 8. 参考资源

- `references/claim-ledger-template.md`：claim ledger、边界矩阵、自审报告模板。
- `references/casebook.md`：实际抓出的错误案例（含 R25/R26 增补），每条有坏句、
  真值、证据、修法；覆盖 layout、API、listing、表标签、phase、GMMA、R2S、
  divide、STG 宽度与 agent 误报。
- `references/audit-history.md`：append-only 轮次日志（R1 起），记录每轮触发、
  范围、发现、修正、验证与残余风险；每轮自审结束后追加。
- `scripts/scan_risky_claims.py`：书稿高风险断言候选扫描；只做候选列表，
  不能替代人工判定。

## 9. 禁止事项

- 不用“推导自洽”代替 API/硬件事实核查。
- 不用单个数值例子代替推导/证明；数值代入通过只说明该点成立。
- 不把近似、上界、经验值写成恒等、精确或“必然”。
- 不只验算结论/末行，跳过中间代数步骤；尤其禁止“从第一行直接到结论”。
- 不把 `A⇒B` 当成 `B⇒A`；不把“存在”当成“任意”。
- 不靠函数名、变量名、注释猜行为。
- 不把历史 PASS 当日志语义证明。
- 不只审正文，漏掉图注、表、问答、标题。
- 不用 `git diff --check` / lint / 静态扫描代替任何受影响 PDF 入口和 RTD 三层验收。
- 不只编主 `book/build.sh`，漏掉 `cute-zhihu/build.sh` 等独立分册。
- 不在没跑所有受影响 PDF 与 RTD `build.sh` + `visual_check` 时声称书稿改动已完成。
- GPU 性能未复测就写“未复测”，不能拿 PDF/RTD 通过冒充性能已验收。
- 不直接改冻结源码；只改书稿或记 CHECKLOG。
- 不为了“看起来一致”把互相矛盾的段落强行改成同一句，先确定哪一句有证据。

## 10. 与 book 工作流的关系

在 `write-leetcuda-book` 的验收步骤中，内容改动完成后加载本 skill：

1. 先按 `references/claim-ledger-template.md` 建账；
2. **先跑第零层（数理逻辑）**：推导链、量词/蕴含、边界/反例；
3. 再跑四层：定义/指称、API/硬件、资源/类型、表达一致性；
4. 把回修写回章节；
5. 跑 `remove-tech-ai-tone` 做文风自检；
6. **强制**按 §5.0 构建矩阵跑所有受影响 PDF：主 book 跑
   `<LeetCUDA_DIR>/kernels/interview/book/build.sh`；cute-zhihu 跑
   `cd <LeetCUDA_DIR>/kernels/interview/book/cute-zhihu && bash build.sh`；
7. **强制**跑 `<LeetCUDA_DIR>/docs/readthedocs/build.sh` +
   `tools/visual_check.py`，按 §5.2 记录三层验收；
8. 把本轮记录追加到 `references/audit-history.md`，把每个受影响 PDF 与
   RTD 的结果一并写入。

第 6、7 步不能跳过。只改 skill 文档、没有书稿改动时，第 8 步必须写明
“本轮无书稿改动，未触发任何 PDF/RTD 闸门”。
