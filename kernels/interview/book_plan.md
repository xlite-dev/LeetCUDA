# LeetCUDA 面试笔记成书计划

> **⚠️ 本文档（v1 draft）已被 [`book/skills/write-leetcuda-book/BOOK_PLAN.md`](book/skills/write-leetcuda-book/BOOK_PLAN.md) 取代**（RFC-0.9，2026-09-11）。仅保留作历史参考，规范与执行跟踪一律以新文档为准。

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
| D7 | 每章标注**编译宏**与**架构限定** | 第 13–20 章源码整段位于 `#if defined(NOTES_V2_ENABLE_*)` 内，读者没有这个映射就无法复现 |
| D8 | 代码清单统一 `firstnumber=auto` | 正文与附录 D 会引用"源码第 N 行"，行号必须与源文件一致 |

## 二、章节结构（从易到难）

三个 Part 对应三档能力：**会写正确的 kernel → 会写快的 kernel → 会写工业级 kernel**。

> **行范围约定**（所有 `linerange` 必须满足）：
> 1. 起点与终点**不得落在注释块中间**（否则读者看到半句话）；
> 2. 范围内出现的 `#if` / `#endif` **必须成对**，禁止孤立指令行被印出；跨章共享的宏块，在章内文字中说明其边界；
> 3. 需要跳过指令行或空段时用不连续范围 `linerange={a-b,d-e}`；
> 4. 补全文件名，不依赖"继承上文源文件"的隐式规则。
>
> 下表行范围已按上述四条逐一核验。

### Part I 基础篇：原语、访存与归约

源：`base.cuh`（909 行）。目标：建立 CUDA 编程模型与性能分析的基本功。

| 章 | 标题 | 行范围 | 难度 |
|---|---|---|---|
| 1 | GPU 架构、内存层级与 Roofline 模型 | `base.cuh` L1–86 | 入门 |
| 2 | 归约原语：Warp Reduce / Block Reduce / Dot Product | `base.cuh` L87–305 | 入门 |
| 3 | 向量化访存与原子操作：Elementwise / Histogram | `base.cuh` L306–386 | 入门 |
| 4 | 归约的应用：merge_attn_states（LSE 合并） | `base.cuh` L387–519 | 基础 |
| 5 | Softmax 三级递进：naive → safe → online | `base.cuh` L520–665 | 基础 ★ |
| 6 | 归一化：RMSNorm 与 LayerNorm | `base.cuh` L666–801 | 基础 |
| 7 | RoPE 与矩阵转置（Bank Conflict 专题） | `base.cuh` L802–909 | 基础 |

第 3/4 章的边界在 `base.cuh:386/387`：第 4 章的说明注释（含 LSE 合并的 5 步公式推导）从 L387 开始，L436 的 `// source:` 行只是注释块末行，若以 L436 起算会把注释块腰斩。

第 5 章是全书第一个关键节点：online softmax 是 FlashAttention 的数学前提，后续 Part III 反复回引。

### Part II GEMM 篇：从 CUDA Core 到 Tensor Core

源：`sgemv.cuh`（102 行）、`sgemm.cuh`（434 行）、`hgemm.cuh`（2100 行）。目标：完整走一遍 GEMM 优化五层金字塔。

| 章 | 标题 | 源 / 行范围 | 难度 |
|---|---|---|---|
| 8 | SGEMV：内存受限算子的三种划分 | `sgemv.cuh` L1–102 | 基础 |
| 9 | SGEMM（一）：Block Tile → 向量化 → 4×4 Thread Tile | `sgemm.cuh` L6–183 | 进阶 |
| 10 | SGEMM（二）：TF32 WMMA 与 cp.async 双缓冲 | `sgemm.cuh` L184–434 | 进阶 |
| 11 | HGEMM（一）：mma.sync m16n8k16 + ldmatrix + 多级流水 | `hgemm.cuh` L3–397 | 进阶 |
| 12 | HGEMM（二）：XOR Swizzle、寄存器双缓冲、Block Swizzle | `hgemm.cuh` L399–716 | 进阶 ★ |
| 13 | HGEMM（三）：用 CuTe 重新表达（TiledMMA / TiledCopy） | `hgemm.cuh` L718–1427 | 高级 |
| 14 | HGEMM（四）：WGMMA m64n128k16 + TMA + Warp Specialization | `hgemm.cuh` L1428–1858 | 专家 |
| 15 | HGEMM（五）：SM120 上的 TMA + mma.sync 组合 | `hgemm.cuh` L1859–2100 | 专家 |

第 9 章标题已更正：源码中并不存在独立的 "Thread Tile" 版本，`sgemm`（L33）是一元素/线程的 Block Tile 版，`sgemm_vec4`（L108）才是 4×4 thread tile + 向量化版。

第 11/12 章的分工需在章内说明：`hgemm_mma_stages_tn`（L130）已含 Block Swizzle（L135–139），第 12 章讲的是在它基础上叠加 XOR Swizzle 与寄存器双缓冲的 `hgemm_mma_stages_tn_swizzle`（L435）。**XOR Swizzle 的实现体在 `common.cuh:236–340`**（附录 A），第 12 章只讲 `hgemm.cuh:389–396` 的派发与 `kColStride` 语义，需前向引用附录 A。

第 14/15 章的宏块边界：`#if NOTES_V2_ENABLE_WGMMA` 在 L1428–1857，`#if NOTES_V2_ENABLE_TMA_MMA_WS` 在 L1859–2100；两章各自完整包含一对，无悬空指令。

### Part III Attention 篇：FlashAttention 2/3 与大 head_dim

源：`flash_attn.cuh`（3490 行）、`ffpa_attn.cuh`（641 行）。目标：掌握 attention 的完整优化史。

| 章 | 标题 | 源 / 行范围 | 难度 |
|---|---|---|---|
| 16 | FlashAttention（一）：Split-Q + 多级流水的 MMA 实现 | `flash_attn.cuh` L5–790 | 专家 |
| 17 | FlashAttention（二）：TMA + Warp Specialization | `flash_attn.cuh` L792–1440 | 专家 |
| 18 | FlashAttention（三）：FA3 双 Consumer WG | `flash_attn.cuh` L1441–2192 | 专家 |
| 19 | FlashAttention（四）：CuTe 基础设施 + 3 个实现对照 | `flash_attn.cuh` L2196–3488 | 专家 |
| 20 | FFPA Split-D：大 head_dim 的分块注意力 | `ffpa_attn.cuh` L1–641 | 专家 |

第 16 章止于 L790：L791 是 `#if defined(NOTES_V2_ENABLE_TMA_MMA_WS)`，第 17/18 两章同处这一个宏块（L791–2194）内部，故两章的引文都不含指令行，需在章内文字里注明所属宏与块边界。第 18 章范围已收紧到 L2192，避免把 L2194 的 `#endif` 孤零零印出。

第 19 章范围自 L2196 起（原文 L2495 落在注释句中间），并**把 `fa_cute` 基础设施一并纳入本章**——它是三个 CuTe 实现的共同依赖，原方案将其划归第 18 章会造成章节题目与内容错位、且隐藏依赖方向。本章含 5 对 `#if/#endif`（2196/2491、2493/2804、2807/3124、3126/3445、3447/3488），全部成对。

第 18 章标题已去掉 setmaxnreg 的强主张：源码中 `NOTES_V2_ENABLE_SETMAXNREGS`（common.cuh:462–475）在 `build.sh` 的四种 arch 里**都未定义**，默认展开为 `((void)0)`；且 sm_120a 上 ptxas 会以 C7506 丢弃 `setmaxnreg`。章内需声明该宏的开启方式与 sm_90a 限定。

### 各章编译宏与架构限定

读者照书复现时必须知道每个 kernel 由哪个宏开启、需要哪种卡：

| 章 | 编译宏 | 架构限定 |
|---|---|---|
| 1–12 | 无（纯 CUDA C++ / WMMA / mma.sync） | 全部 |
| 12 | 另可选 `NOTES_V2_ENABLE_SWIZZLE_V2` | 全部 |
| 13 | `NOTES_V2_ENABLE_CUTE` | 全部 |
| 14 | `NOTES_V2_ENABLE_WGMMA` | 仅 sm_90a |
| 15 | `NOTES_V2_ENABLE_TMA_MMA_WS` | sm_90a / sm_120a，且要求 CUDART ≥ 13.0 |
| 16 | 无 | 全部 |
| 17–18 | `NOTES_V2_ENABLE_TMA_MMA_WS` | sm_90a / sm_120a |
| 18 | 另可选 `NOTES_V2_ENABLE_SETMAXNREGS`（默认关闭） | sm_90a（sm_120a 上被 ptxas 丢弃） |
| 19 | `NOTES_V2_ENABLE_CUTE` | 全部 |
| 20 | `NOTES_V2_ENABLE_CUTE` + `NOTES_V2_ENABLE_TMA_MMA_WS` | sm_90a / sm_120a |

`NOTES_V2_ENABLE_TMA_MMA_WS` 在 CUDART < 13.0 时由 `common.cuh:21–23` 直接 `#error`，这是硬门槛。

### 附录

| 附录 | 内容 | 源 |
|---|---|---|
| A | 基础设施工具箱：PTX MMA/WGMMA 宏、XOR Swizzle v1/v2、TMA/mbarrier helper、TensorMap、setmaxnreg | `common.cuh`（773 行） |
| B | 性能数据（SM120A 实测，vs cuBLAS / cuDNN / PyTorch SDPA） | `README.md` |
| C | 构建与运行：`build.sh`、CLI 参数、8 种 FA layout + all | `notes-v2.cu` + `build.sh` + `README.md` |
| D | 难度索引：topic ↔ 源文件行号对照表 | 全书 |

附录 A 虽是附录，却是 **Part II 的前置依赖**（`swizzle_v1/v2_impl`、`LDMATRIX_X4`、`CP_ASYNC_*`、`WGMMA_*` 宏都在其中），正文第 11/12/15/19/20 章必须前向引用它，否则代码解释会悬空。

附录 C 的 FA layout 取值真实出处是 `notes-v2.cu`：`enum class FALayout`（L43）与 `--fa-layout` 解析（L4697–4715），取值为 `all` + 8 种 layout（pad / swizzle-q / swizzle-k / swizzle-v / swizzle-qk / swizzle-qv / swizzle-kv / swizzle）。`build.sh` 与 `README.md` 中均无 "layout" 字样。

附录 B 的性能数据必须标注口径：README 的表是 **SM120a**、cuBLAS v13.3.0.5、cuDNN v9.25.0.15 的实测；且表里 "FA2 TMA MMA WS (1 Consumer WG)" 这类名称与源码函数名已不对应，需补一列"对应章节/函数名"。

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
- 两级 `morekeywords`（共 256 条 CUDA/CuTe 标识符）——**需先审计再复用**，见 Phase A 第 3 步
- 字体设置：`\setmonofont{DejaVu Sans Mono}` + `\setCJKmonofont{Noto Sans Mono CJK SC}`
- 构建方式：`TEXMFCNF=../tex/ xelatex -interaction=nonstopmode <file>.tex`

### 路径与工作目录约定

- 构建时 **CWD = `kernels/interview/book/`**
- 引源码用相对路径 `\lstinputlisting{../base.cuh}` 等（与 `tex/notes-v2_chunks.tex` 的 `../common.cuh` 同构；`listings` 按进程 CWD 解析路径）
- `texmf.cnf` **不复制**，直接复用 `../tex/texmf.cnf`，故 `TEXMFCNF=../tex/:`（若写成 `./`，从 `book/` 运行时该文件不存在，**配置会静默失效而不报错**）

## 五、执行步骤

### Phase A — 脚手架（独立可验证）

1. 建 `book/` 目录树
2. 抽 `preamble.tex`（复用上表配置 + 新增宏包），代码清单设 `firstnumber=auto`
3. **审计并扩充关键字表**：现有 256 条已过期——`sgemm_thread_tile_vec4`、`rope_f32_kernel`、`mat_transpose_f32_row2col2d_kernel`、`block_all_reduce_sum`、`hgemv_k32_f16_kernel`、`hgemm_t_8x8_sliced_k_f16x4_kernel`、`flash_attn_mma_stages_split_q_kernel` 在当前 `.cuh` 中已 0 命中（实际名称为 `sgemm_vec4`、`rope`、`mat_transpose`、`block_reduce_all`）；同时需**补入 CuTe/FFPA 标识符**：`fa_cute::*`、`FFPAAttnSplitDCuTeTraits`、`FlashAttn3CuTeTraits`、`gemm_ss`/`gemm_rs`、`convert_layout_acc_rowcol`、`swizzle_v1_impl`/`swizzle_v2_impl`、`SwizzleBMS`、`create_tensor_map`、`NOTES_V2_REG_ALLOC`/`NOTES_V2_DEALLOC`。否则第 13/19/20 章与附录 A 的代码基本不高亮
4. `book.tex` 写入 Part / Chapter 骨架
5. `build.sh` 实现两遍 xelatex（`TEXMFCNF=../tex/`）
6. **出口条件**：空骨架能编译出带正确 TOC 的 PDF

Phase A 必须先独立验证通过再写正文，否则排版问题会与内容问题混在一起难以定位。

### Phase B — 章节正文

- B1 = Ch1–7（基础篇）、B2 = Ch8–15（GEMM 篇）、B3 = Ch16–20（Attention 篇）、B4 = 附录 A–D
- 四批可并行起草，但**先冻结术语与记号表**，最后统一一次润色 pass
- **每章长度预算（决定 250 页达标与否）**：纯代码按实测密度约 64 行/页，8432 行引文仅折合 **≈131 页**；故需**每章新写中文讲解 ≥ 5 页（约 3500–5000 字）**，20 章合计 ≥ 100 页，才能达到全书 ≥ 250 页
- **B1 完成后外推全书页数**：若外推不足 250 页，按章补写讲解后再继续 B2/B3
- 每批完成后即时编译该部分，避免错误累积
- 附录 D 的行号索引必须等 B1–B3 范围定稿后才能生成，**不可并行**

### Phase C — 收尾

1. 补全 `\ref` 交叉引用、目录与索引
2. 两遍编译确认 TOC / 页码 / PDF 书签
3. `pdftotext` 抽查渲染：中文正常、代码无横向溢出
4. **内容锚点断言**（而非只校验上界）：每章记录源文件 SHA256 + 范围内首行/末行必须匹配的文本，用脚本断言。仅校验"上界不越界"无法发现"边界落在注释块中间""孤儿 `#if/#endif`""内容随源码漂移"三类问题
5. 逐章核对：范围内每一行都能在 `tex/notes-v2_chunks.tex` 的 7 个整篇 listing 中找到（第 20 章除外，现有 PDF 不含 `ffpa_attn.cuh`）

## 六、风险与对策

| # | 风险 | 对策 |
|---|---|---|
| R1 | 根 `.gitignore` 含 `*.tex` 规则，新建章节 `.tex` 会被静默忽略（现有 `tex/*.tex` 属历史遗留跟踪）；且 `*.toc`/`*.out`/`*.pdf` 未被忽略 | 追加 `!kernels/interview/book/**/*.tex` 与 `kernels/interview/book/*.{pdf,toc,out}`。**不要用 `!kernels/interview/book/**`**——按 gitignore"后匹配优先"，那会把 book/ 下的 PDF/aux 一并解禁，违反 D5 |
| R2 | `listings` 整篇排版大文件会耗尽 TeX 内存 | 仅 `hgemm.cuh`/`flash_attn.cuh` 禁止整篇引用，一律 `linerange`（可用逗号分段跳过 `#if/#endif`）；第 8/20 章与附录 A 属整篇引用，需 `firstnumber=auto`。**注意：现代 TeX Live 中 `main_memory` 是编译期常量，`texmf.cnf` 对该项多半无效**，真正的护栏是 `linerange` |
| R3 | CJK 等宽字体缺失，PDF 出现豆腐块 | 预检 `fc-list \| grep -i "Noto Sans Mono CJK SC"`；缺失则改 `Noto Sans Mono CJK` / `Source Han Mono SC` |
| R4 | 长代码行溢出 A4 页宽 | 现有 `breaklines=true` + `columns=fullflexible` 已处理，靠 Phase C 抽查把关 |
| R5 | 全量两遍编译耗时长 | 后台执行，避免频繁轮询 |
| R6 | 源码行范围随代码演进而失效，`linerange` 越界静默丢内容 | 见 R7 |
| R7 | 行范围**语义**漂移（越界只是最轻的一种）：边界落进注释块、宏块被切断、内容整体位移 | 每章记录源文件 SHA256 + 首/末行锚点文本；Phase C 用脚本断言（而非只校验上界）；索引表禁止使用"全篇"式范围 |

## 七、验收标准

- `book.pdf` 生成成功，页数不少于 250（其中中文讲解 ≥ 100 页，每章 ≥ 5 页；纯代码约 131 页）
- TOC 包含全部 20 章 + 4 附录，页码正确，PDF 书签可用
- **引文无悬空预处理指令**：逐章 grep 生成正文的 `linerange` 文本，确认 `#if`/`#endif` 成对
- `pdftotext` 抽查无豆腐块，中文渲染正常
- 代码块带语法高亮与行号（且行号与源文件一致），无横向溢出
- 每章标注了编译宏与架构限定（见 §二 映射表）
- 编译日志无 `! LaTeX Error`，`Overfull \hbox` 数量可控

## 八、范围边界

**包含**：`kernels/interview/book/` 下的 LaTeX 源与构建脚本、PDF 产出。

**不包含**：KDP 元数据生成、封面设计、HTML 输出、Amazon 上架流程。

**不改动**：现有 `notes-v2.cu`、`base.cuh` 等教学源码。书通过 `linerange` 引用它们，源文件保持为唯一事实来源。

## 九、待确认事项

1. **书名与署名**：建议《CUDA Kernel 面试背题笔记（进阶版）》/ LeetCUDA Project，沿用现有 `tex` 的标题风格
2. **PDF 是否入库**：建议不入库（可再生、体积大）
3. **现有 `tex/notes-v2.pdf` 的去留**：建议保留，作为"全量源码打印本"与书并存

## 十、环境要求

本节列出**构建与复现所需的环境**。开发机（本 workspace 所在机器）未安装 XeLaTeX 与 Noto CJK 字体，构建需在上述环境齐备的机器上进行。

### 10.1 排版工具链（构建 PDF 的硬前提）

| 组件 | 要求 | 用途 |
|---|---|---|
| TeX Live | ≥ 2023，须含 XeTeX 引擎 | `ctex` 依赖 XeLaTeX，pdfLaTeX 无法处理中文字体 |
| `xelatex` | 包 `texlive-xetex` | 唯一编译器 |
| `ctex` | 包 `texlive-lang-chinese` | 中文文档类 `ctexbook` |
| 其他宏包 | `texlive-latex-extra`（`listings`/`caption`/`enumitem`）、`texlive-fonts-recommended`、`texlive-fonts-extra`（`booktabs`/`amsmath`/`hyperref` 多在基础集合） | 排版 |
| `fontconfig` | 系统包 | 字体发现，缺失时 fontspec 报错 |

Debian / Ubuntu 参考安装：

```bash
apt-get update && apt-get install -y \
  texlive-xetex texlive-lang-chinese texlive-latex-extra \
  texlive-fonts-recommended texlive-fonts-extra fontconfig
```

安装后自检（三项都应返回路径，空则为失败）：

```bash
xelatex --version
kpsewhich ctexbook.cls
kpsewhich listings.sty
```

### 10.2 字体（缺则中文/代码出现豆腐块）

| 字体 | 用途 | 缺失后果 |
|---|---|---|
| `Noto Sans Mono CJK SC` | `\setCJKmonofont`：代码块内的中文注释 | 中文注释整片豆腐块 |
| `DejaVu Sans Mono` | `\setmonofont`：代码本体 | 回退到默认字体，行宽失控、溢出页边 |
| CJK 正文字体（`Noto Serif CJK SC` 或 `Noto Sans CJK SC`） | `ctex` 中文正文 | 正文豆腐块 |

```bash
apt-get install -y fonts-noto-cjk fonts-dejavu-core
fc-list | grep -i "Noto Sans Mono CJK SC"
fc-list | grep -i "DejaVu Sans Mono"
```

若 `Noto Sans Mono CJK SC` 不可得，可改用 `Noto Sans Mono CJK` 或 `Source Han Mono SC`（需同步修改 `preamble.tex`）。

### 10.3 源码复现环境（想跑通书中 kernel 时需要）

本书只讲解既有内核，**写作不依赖 CUDA**；但读者要复现书中的性能数据需要：

| 组件 | 要求 |
|---|---|
| CUDA Toolkit | ≥ 13.2（`build.sh` 的 `--arch sm_89/sm_90a/sm_120a` 与 cuDNN 9 依赖此版本） |
| GPU 架构 | sm_89 (Ada) / sm_90a (Hopper) / sm_120a (Blackwell)。**第 14 章仅 sm_90a；第 15/17/18/20 章需 sm_90a 或 sm_120a** |
| cuDNN | 9.x（`cudnn9-cuda-13`），bench 的 SDPA 对照需要 |
| CUTLASS | 仓库子模块 `third-party/cutlass`（`-I ../../third-party/cutlass/include`） |
| cudnn-frontend | 仓库子模块 `third-party/cudnn-frontend`（`-I ../../third-party/cudnn-frontend/include`） |
| ccache | 可选，显著加速重建 |
| Python 3 + PyTorch | 仅 `bench_sdpa.py` 需要（PyTorch SDPA / flash-attn baseline） |

构建与运行：

```bash
# 子模块
git submodule update --init --recursive --force

# 编译（在 kernels/interview/ 下）
./build.sh --arch sm_120a      # 或 sm_89 / sm_90a / all

# 运行
./notes_v2_sm120a.bin --bench --mnk 4096,4096,4096 --bhnd 1,32,16384,128
```

### 10.4 构建本书

```bash
cd kernels/interview/book
TEXMFCNF=../tex/: xelatex -interaction=nonstopmode book.tex   # 第一遍：生成 .aux/.toc
TEXMFCNF=../tex/: xelatex -interaction=nonstopmode book.tex   # 第二遍：解析目录与交叉引用
```

`book/build.sh` 封装上述两步并清理中间文件。要点：

- CWD 必须是 `book/`，源码引用写作 `../base.cuh` 形式
- `TEXMFCNF=../tex/` 指向现有扩容配置；写成 `./` 会因 `book/` 下无 `texmf.cnf` 而**静默失效**
- 改动章节后需重跑两遍，否则页码与目录不更新
