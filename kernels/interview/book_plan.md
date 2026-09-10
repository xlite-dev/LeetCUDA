# LeetCUDA 面试笔记成书计划

把 `kernels/interview/` 下分散在 `.cuh` 源码注释里的教学内容，重构为一本**按难度递进**的中文 CUDA Kernel 技术书，最终产出 PDF。

## 一、总体设计

### 1.1 现状

| 现状 | 问题 |
|---|---|
| 教学内容全部以中文注释形式内嵌在 7 个 `.cuh` 文件（共 8449 行） | 无章节、无目录、无讲解 |
| `tex/notes-v2.tex` 仅 180 行，正文只有 7 条 `\lstinputlisting` 整篇引入源码 | 是"带语法高亮的源码打印本"，不是书 |
| `notes-v2.cu` 4915 行 test/bench harness | 与教学正文混杂 |
| `tex/notes-v2_chunks.tex` 未包含 `ffpa_attn.cuh` | PDF 内容滞后于代码 |

但已有两条高质量资产可直接复用：

**资产一：教学骨架已天然 easy→hard**

- 源码中已存在 `Phase 0–8` 主轴，以及 `Level 1–5` 优化阶梯
- `notes-v2.cu` 的 `main()` 默认调用序严格从易到难（block_reduce → dot → relu → … → sgemm → hgemm → flash_attn）
- `sgemm.cuh` 头部已有 GEMM 优化五层金字塔与 Roofline 定量分析

**资产二：中文排版管线已就绪**

`tex/notes-v2.tex` 的 XeLaTeX + `ctexart` + `listings` 配置，含：

- VSCode 风格 8 色配色（`vscBackground` / `vscKeyword` / `vscComment` 等）
- `DejaVu Sans Mono` + `Noto Sans Mono CJK SC` 双等宽字体
- 256 条关键字表（一级 80 条 CUDA 内建 + 二级 176 条项目内核/CuTe 标识符）
- `texmf.cnf` 内存扩容配置

### 1.2 为什么选 LaTeX 而非 Markdown 管线

曾评估 Markdown → HTML → PDF 的管线，放弃。原因是它无法满足技术书的三个硬需求：

| 需求 | Markdown 管线 | LaTeX 管线 |
|---|---|---|
| 中文正文字体 | 需自行配置 CJK 字体栈，易出豆腐块 | `ctexbook` 原生支持，按字符类自动切换字体 |
| 代码语法高亮 + 行号 | 需引入外部高亮库，语言标签易丢失 | `listings` 原生支持，关键字表已在用 |
| 数学公式（Roofline / online softmax） | 需 KaTeX/MathJax，PDF 导出不稳定 | `amsmath` 原生支持 |

且现有 `tex/` 管线**已验证可构建**（`notes-v2.pdf` 存在），扩展它比新建管线风险低得多。

### 1.3 决策清单

| # | 决策 | 说明 |
|---|---|---|
| D1 | PDF 管线 = **XeLaTeX + ctexbook + listings** | 扩展现有 `tex/` 配置 |
| D2 | 正文形态 = **每章新写中文讲解 + 精选源码** | 不做"源码打印本"；代码用 `\lstinputlisting[linerange={a-b}]` 按行范围裁剪嵌入 |
| D3 | 收录范围 = **只收 `.cuh` 教学内容** | `notes-v2.cu` 的 test/bench harness 与 `bench_*.cu` 工具不入正文，性能数据进附录 |
| D4 | `ctexart` → **`ctexbook`**（`[9pt,openany]`） | 需要 `\part`/`\chapter`/TOC/PDF 书签；`openany` 避免每章前插入空白页 |
| D5 | 输出 PDF **不入库** | 可再生、体积大；`.aux`/`.log`/`.toc`/`.out` 一律 gitignore |
| D6 | 保留现有 `tex/notes-v2.pdf` | 作为"全量源码打印本"与书并存 |

## 二、章节结构（从易到难）

三个 Part 对应三档能力：**会写正确的 kernel → 会写快的 kernel → 会写工业级 kernel**。

### Part I 基础篇：原语、访存与归约

源：`base.cuh`（909 行）。目标：建立 CUDA 编程模型与性能分析的基本功。

| 章 | 标题 | 行范围 | 难度 |
|---|---|---|---|
| 1 | GPU 架构、内存层级与 Roofline 模型 | `base.cuh` L1–86 | 入门 |
| 2 | 归约原语：Warp Reduce / Block Reduce / Dot Product | L87–305 | 入门 |
| 3 | 向量化访存与原子操作：Elementwise / Histogram | L306–435 | 入门 |
| 4 | 归约的应用：merge_attn_states（LSE 合并） | L436–519 | 基础 |
| 5 | Softmax 三级递进：naive → safe → online | L520–665 | 基础 ★ |
| 6 | 归一化：RMSNorm 与 LayerNorm | L666–801 | 基础 |
| 7 | RoPE 与矩阵转置（Bank Conflict 专题） | L802–909 | 基础 |

第 5 章是全书第一个关键节点：online softmax 是 FlashAttention 的数学前提，后续 Part III 反复回引。

### Part II GEMM 篇：从 CUDA Core 到 Tensor Core

源：`sgemv.cuh`（102 行）、`sgemm.cuh`（434 行）、`hgemm.cuh`（2100 行）。目标：完整走一遍 GEMM 优化五层金字塔。

| 章 | 标题 | 源 / 行范围 | 难度 |
|---|---|---|---|
| 8 | SGEMV：内存受限算子的三种划分 | `sgemv.cuh` 全篇 | 基础 |
| 9 | SGEMM（一）：Block Tile → Thread Tile → 向量化 | `sgemm.cuh` L8–183 | 进阶 |
| 10 | SGEMM（二）：TF32 WMMA 与 cp.async 双缓冲 | `sgemm.cuh` L184–434 | 进阶 |
| 11 | HGEMM（一）：mma.sync m16n8k16 + ldmatrix + 多级流水 | `hgemm.cuh` L5–397 | 进阶 |
| 12 | HGEMM（二）：XOR Swizzle、寄存器双缓冲、Block Swizzle | L399–716 | 进阶 ★ |
| 13 | HGEMM（三）：用 CuTe 重新表达（TiledMMA / TiledCopy） | L718–1427 | 高级 |
| 14 | HGEMM（四）：WGMMA m64n128k16 + TMA + Warp Specialization | L1428–1862 | 专家 |
| 15 | HGEMM（五）：SM120 上的 TMA + mma.sync 组合 | L1863–2100 | 专家 |

第 12 章是第二个关键节点：寄存器双缓冲与 swizzle 是后续所有高性能 kernel 的通用手法。第 14/15 章按架构分流（Hopper 的 WGMMA 路径 vs SM120 的 TMA+mma.sync 路径），读者可按自己的卡选读。

### Part III Attention 篇：FlashAttention 2/3 与大 head_dim

源：`flash_attn.cuh`（3490 行）、`ffpa_attn.cuh`（641 行）。目标：掌握 attention 的完整优化史。

| 章 | 标题 | 源 / 行范围 | 难度 |
|---|---|---|---|
| 16 | FlashAttention（一）：Split-Q + 多级流水的 MMA 实现 | `flash_attn.cuh` L5–791 | 专家 |
| 17 | FlashAttention（二）：TMA + Warp Specialization | L792–1440 | 专家 |
| 18 | FlashAttention（三）：FA3 双 Consumer WG + setmaxnreg | L1441–2494 | 专家 |
| 19 | FlashAttention（四）：CuTe 版本对照 | L2495–3490 | 专家 |
| 20 | FFPA Split-D：大 head_dim 的分块注意力 | `ffpa_attn.cuh` 全篇 | 专家 |

第 19 章承载"手写 PTX vs 高层抽象"的方法论对比：同一算法在 CuTe 下有 3 个实现，与 Part II 的手写版本互为镜像。

### 附录

| 附录 | 内容 | 源 |
|---|---|---|
| A | 基础设施工具箱：PTX MMA/WGMMA 宏、XOR Swizzle v1/v2、TMA/mbarrier helper、TensorMap、setmaxnreg | `common.cuh`（773 行） |
| B | 性能数据（SM120A 实测，vs cuBLAS / cuDNN / PyTorch SDPA） | `README.md` |
| C | 构建与运行：`build.sh`、CLI 参数、9 种 FA layout | `build.sh` + `README.md` |
| D | 难度索引：topic ↔ 源文件行号对照表 | 全书 |

`common.cuh` 放附录而非正文，是因为它是**被引用的基础设施**而非独立教学单元——正文各章按需回引，附录集中呈现完整实现。

## 三、每章统一模板

1. **本章导读** — 学什么、前置章节、预计篇幅
2. **问题引入与动机** — 为什么需要这个算子/这项优化
3. **设计决策与 WHY** — 提炼源码中的 `Phase` / `Level` 注释
4. **关键源码** — `\lstinputlisting[linerange={a-b}]` 裁剪嵌入，配逐段讲解
5. **Roofline 与性能分析** — 算术强度公式 + 实测数据对照
6. **常见坑** — bank conflict / 寄存器溢出 / mbarrier 误用 / 流水线排空
7. **面试要点速查** — Q&A 形式
8. **小结与下一章衔接**

## 四、文件规划

```
kernels/interview/book/
├── book.tex          # 主文件：preamble + \tableofcontents + part/chapter 骨架
├── preamble.tex      # 共享导言区（从 tex/notes-v2.tex 抽取）
├── build.sh          # 两遍 xelatex + 清理 aux
├── chapters/         # ch01-*.tex … ch20-*.tex
└── appendices/       # appA-*.tex … appD-*.tex
```

### 复用（从 `tex/notes-v2.tex` 抽取）

- 8 色调色板定义
- `\lstset{...}` 全文配置：`language=C`、`numbers=left`、`breaklines=true`、`columns=fullflexible`、`keepspaces=true`
- 两级 `morekeywords`（共 256 条 CUDA/CuTe 标识符）
- 字体设置：`\setmonofont{DejaVu Sans Mono}` + `\setCJKmonofont{Noto Sans Mono CJK SC}`
- 构建方式：`TEXMFCNF=./: xelatex -interaction=nonstopmode <file>.tex`

### 新增宏包

`amsmath` / `amssymb`（Roofline 与 online softmax 公式）、`booktabs`（三线表）、`hyperref`（书签与交叉引用）、`caption`、`enumitem`。

## 五、执行步骤

### Phase A — 脚手架（独立可验证）

1. 建 `book/` 目录树
2. 抽 `preamble.tex`（复用上表配置 + 新增宏包）
3. `book.tex` 写入 Part / Chapter 骨架
4. `build.sh` 实现两遍 xelatex
5. **出口条件**：空骨架能编译出带正确 TOC 的 PDF

Phase A 必须先独立验证通过再写正文，否则排版问题会与内容问题混在一起难以定位。

### Phase B — 章节正文

- B1 = Ch1–7（基础篇）、B2 = Ch8–15（GEMM 篇）、B3 = Ch16–20（Attention 篇）、B4 = 附录 A–D
- 四批可并行起草，最后统一一次润色 pass 保持一致语气与术语
- 每批完成后即时编译该部分，避免错误累积

### Phase C — 收尾

1. 补全 `\ref` 交叉引用、目录与索引
2. 两遍编译确认 TOC / 页码 / PDF 书签
3. `pdftotext` 抽查渲染：中文正常、代码无横向溢出
4. 与现有 `tex/notes-v2.pdf` 对比确认无回归

## 六、风险与对策

| # | 风险 | 对策 |
|---|---|---|
| R1 | 根 `.gitignore` 含 `*.tex` 规则，新建章节 `.tex` 会被静默忽略（现有 `tex/*.tex` 属历史遗留跟踪） | Phase A 前在根 `.gitignore` 追加 `!kernels/interview/book/**`，或对新增文件用 `git add -f` |
| R2 | `listings` 整篇排版 2100 / 3490 行的大文件会耗尽 TeX 内存 | **禁止整篇 `\lstinputlisting`**，一律 `linerange` 裁剪；必要时继续调大 `texmf.cnf` |
| R3 | CJK 等宽字体缺失，PDF 出现豆腐块 | 执行前预检 `fc-list \| grep -i "Noto Sans Mono CJK SC"`；缺失则改 `Noto Sans Mono CJK` / `Source Han Mono` |
| R4 | 长代码行溢出 A4 页宽 | 现有 `breaklines=true` + `columns=fullflexible` 已处理，靠 Phase C 抽查把关 |
| R5 | 全量两遍编译耗时长（单遍预计超过 120s，正文补齐后更久） | 后台执行，避免频繁轮询；完成后系统会通知 |
| R6 | 源码行范围随代码演进而失效，导致 `linerange` 越界静默丢内容 | Phase C 加脚本校验每章 `linerange` 上界不超过源文件实际行数；本书只引用不修改源文件 |

## 七、验收标准

- `book.pdf` 生成成功，页数不少于 250
- TOC 包含全部 20 章 + 4 附录，页码正确，PDF 书签可用
- `pdftotext` 抽查无豆腐块，中文渲染正常
- 代码块带语法高亮与行号，无横向溢出
- 编译日志无 `! LaTeX Error`，`Overfull \hbox` 数量可控

## 八、范围边界

**包含**：`kernels/interview/book/` 下的 LaTeX 源与构建脚本、PDF 产出。

**不包含**：KDP 元数据生成、封面设计、HTML 输出、Amazon 上架流程。

**不改动**：现有 `notes-v2.cu`、`base.cuh` 等教学源码。书通过 `linerange` 引用它们，源文件保持为唯一事实来源。

## 九、待确认事项

1. **书名与署名**：建议《CUDA Kernel 面试背题笔记（进阶版）》/ LeetCUDA Project，沿用现有 `tex` 的标题风格
2. **PDF 是否入库**：建议不入库（可再生、体积大）
3. **现有 `tex/notes-v2.pdf` 的去留**：建议保留，作为"全量源码打印本"与书并存
