---
name: leetcuda-cpp-kernel
description: >-
  LeetCUDA 中文技术书（584 页，XeLaTeX 源）按需查阅 skill——写、优化、调试或
  review CUDA C++/PTX kernel 时的权威参考路由层。当任务涉及：GPU 架构/Roofline/
  occupancy、向量化与 coalescing、warp/block reduce、softmax（online/LSE merge）、
  SGEMV/SGEMM/HGEMM 阶梯优化、mma.sync/ldmatrix/WMMA、XOR/block swizzle、cp.async
  多级流水、TMA/mbarrier/WGMMA（Hopper）、SM120 TMA+warp specialization、
  FlashAttention FA1/FA2/FA3 实现、SM120 持久化 FA（persist-D 超越 cuDNN）、
  split-D 大 head_dim、CuTe Layout/Tensor/
  TiledCopy/TiledMMA（含官方白皮书译注、colfax 范畴论译注、cute-zhihu 合集三份
  深度参考）、FP8/FP4 量化 GEMM（e4m3 per-row/per-block 前处理、NVFP4 两级量化、
  block-scaled MMA m16n8k64.kind::mxf4nvf4 与 SF 流水）、FP8/FP4(NVFP4) 量化注意力、
  nsys/ncu 性能分析、cuobjdump/
  PTX/SASS 取证（含 setmaxnreg 与 shared::cluster/cta 陷阱）、CUDA 面试题时使用。
  生产级 attention 算子参考首选 ffpa-attn（sm_120 CuTe fp16/fp8/fp4 峰值性能：
  fp16 超 FA-2、fp8 持平略优 Sage2、fp4 明显优于 Sage3，含 large head_dim 专门
  优化）。ch00 profiling 与 ch01 架构/Roofline 的总体指导精编内嵌，加载即读；
  其余章节只做"任务 → 章节/图"的路由，agent 按需读取书源文件。
user-invocable: true
---

# leetcuda-cpp-kernel — ch00/ch01 内嵌 + LeetCUDA 按需路由

本 skill 是 **LeetCUDA 开源技术书** 的总体指导与路由层：ch00 profiling 和
ch01 架构/Roofline 的可操作内容已精编内嵌在文末，加载本文件即完成前置阅读；
其余书稿（chapters/*.tex + figures + appendices + colfax/cute-zhihu）仍按需读取。

## 第一步：解析 <LeetCUDA_DIR>

按以下顺序定位 LeetCUDA 仓库根（记为 `<LeetCUDA_DIR>`），**全程环境无关**：

1. 环境变量 `LEETCUDA_DIR`（若已设置则直接使用）
2. 从当前工作目录及其父目录、兄弟目录中查找名为 `LeetCUDA/` 且含
   `kernels/interview/` 的目录（如 `./LeetCUDA`、`../LeetCUDA`、monorepo 内
   兄弟目录布局）
3. 常见位置探测：`~/LeetCUDA`
4. 本地没有则 clone（浅克隆即可）：
   `git clone --depth 1 https://github.com/xlite-dev/LeetCUDA.git <工作区>/LeetCUDA`

书根（本 skill 全部资源相对它描述）：

```
<LeetCUDA_DIR>/kernels/interview/book
```

目录结构：

| 路径（相对书根） | 内容 |
|---|---|
| `chapters/chNN-<slug>.tex` | 42 个 tex（41 章正文 + `_template.tex` 章模板）：ch00–ch37 主线 38 章 + ch19b 白皮书导读 + ch26b/ch26c 增补章（中文，XeLaTeX 源；含行号锚定的源码解析、踩坑记录、实测性能表） |
| `chapters/wp/wp0-7.tex` | ch19b 分节正文：CuTe 官方白皮书（Cris Cecka, arXiv:2603.02298）完整中文译注 |
| `figures/drawio/fig-<chNN>-<n>-<slug>/` | **历史留档**：旧版 `.drawio`/`gen.py`/`*.png` 三件套（章内图已全部迁 TikZ inline，仅作旧版对照，勿当现行图源） |
| `figures/ffpa/`、`figures/tikz/`、`figures/misc/` | ffpa bench 图留档 / wp6 白皮书 2 张官方架构截图 / 封面 `cover.pdf` 等杂项 |
| `colfax/` | 独立译注文档：Colfax《Categorical Foundations for CuTe Layouts》（arXiv:2601.05972）中译，源在 `colfax/sec/`，成品 `colfax/colfax-cute-zh.pdf` |
| `cute-zhihu/` | 独立合集文档：reed（13 篇）+ 竹熙佳处（7 篇）CuTe 知乎系列忠实整理，源在 `cute-zhihu/chapters/`，成品 `cute-zhihu/cute-zhihu.pdf` |
| `references/` | `zhihu-inventory.md`（附录 E 数据源）+ `fulltext/`（7 篇知乎全文 markdown，可直接 grep） |
| `appendices/appA..appE` | 见下文附录路由 |
| `tests/` | 31 个单章正确性测试（CPU fp64 对拍，`build_tests.sh --arch sm_120a --all`）；ch00（工具章）、ch15（原理章）、ch27-33（Part VI 改走 `ffpa_attn.bench` parity）没有 .cu |
| `book.pdf` | 编译成品（584 页，可直接 pdftotext 按页抽取） |

## ⚠️ 第 1.5 步：强制约束——加载即读（优先级高于"按需读取纪律"）

以下两类内容在 **skill 加载后、开始任务分析/动手前必须已经阅读**，不得等路由命中才读：

1. **ch00 + ch01——所有任务无条件内嵌已读**：两章的可操作内容已精编内嵌于文末
   “内嵌总体指导”两节，加载 `SKILL.md` 即完成阅读，覆盖先 nsys 后 ncu、
   bench 口径、warmup/repeat、GPU 执行模型、Roofline 与 occupancy。不要再
   求 agent 重复外部读取这两个 `.tex`；只有需要完整案例叙述或核对同步时才回读源文件。
2. **Part IV CuTe 全部章节——涉及 CuTe 开发的任务必读**：任务涉及 CuTe
   开发的任一形态（写/改/优化/调试 CuTe C++ kernel，CuTe DSL，或出现
   CuTe/CUTLASS/CuTe Layout/Tensor/TiledCopy/TiledMMA/Swizzle 等关键词）
   时，必须先读 Part IV **全部**章节，再开始任务：
   `ch19b-cute-whitepaper-zh.tex`（含 `chapters/wp/wp0-7.tex` 分节）、
   `ch20-cute-layout.tex`、`ch21-cute-tensor-tiledcopy.tex`、
   `ch22-cute-tiledmma.tex`、`ch23-cute-swizzle-tma.tex`、
   `ch24-cute-hgemm.tex`、`ch25-cute-flash-attn.tex`、`ch26-cute-ffpa.tex`、
   `ch26b-cute-persist-d-flash-attn.tex`、`ch26c-cute-split-d-sm120.tex`。
   原因：CuTe 各章类型代数强耦合（Layout → Tensor → TiledCopy/TiledMMA →
   Swizzle/TMA → 实战章），只读单章会漏掉跨章依赖，导致布局/引擎误用。

本节与"第三步：按需读取纪律"第 1 条（一次只读 1-2 章）冲突时，**以本节
为准**；必读章之外的内容仍按需读取。

## 第二步：任务路由表

先按任务关键词找到章，再读该章 tex（**先 `grep -n '\\\\section'` 看目录，再按行区间读**，
不要整章全文读入）。**全书章内配图已 100% 迁为 TikZ inline**（2026-10-19 复核：
`chapters/*.tex` 194 处 + `chapters/wp/*.tex` 34 处 = 228 处 `tikzpicture`，
ch00–ch37 每章 2-11 处；`chapters/` 内 `\includegraphics` 全部为
`% was:` 历史注释；全书活跃图片引用仅 3 处——封面 `figures/misc/cover.pdf` +
ch19b/wp6 的 2 张官方架构截图），现行图源都在章节 tex 内。

### Part I-II：CUDA 基础与 GEMM 阶梯（配图全部 TikZ 内联在各章 tex 内）

| 任务/症状 | 章 |
|---|---|
| 优化前方法论：先 nsys 后 ncu、torch.profiler、warmup/repeat 口径 | **已内嵌**：文末“ch00-profiling” |
| 架构/执行模型/warp 调度/Roofline/Occupancy、寄存器与 255 墙 | **已内嵌**：文末“ch01-arch-roofline” |
| warp shuffle 蝴蝶归约、block reduce、dot product | `ch02-reduce-dot.tex` |
| 向量化（f32x4/f16x8 pack）、coalescing、原子操作 | `ch03-vectorize-atomic.tex` |
| softmax 三级递进（naive→safe→online） | `ch04-softmax.tex` |
| LSE 与分块合并 merge_attn_states | `ch05-lse-merge.tex` |
| RMSNorm/LayerNorm（单遍统计） | `ch06-norm.tex` |
| RoPE、矩阵转置、bank conflict 与 padding | `ch07-rope-transpose.tex` |
| SGEMV 三种划分（memory-bound 专题） | `ch08-sgemv.tex` |
| SGEMM 阶梯一：block tile/Vec4/thread tile、双缓冲 | `ch09-sgemm-tiling.tex` |
| SGEMM 阶梯二：TF32 WMMA、cp.async 多级流水 | `ch10-sgemm-tf32-wmma.tex` |
| HGEMM：mma.sync m16n8k16、ldmatrix、fragment 布局 | `ch11-hgemm-mma.tex` |
| XOR swizzle、寄存器双缓冲、block swizzle、L2 复用 | `ch12-hgemm-swizzle.tex` |

### Part III：Hopper/SM120 与 FlashAttention（配图全部 TikZ 内联）

| 任务/症状 | 章 |
|---|---|
| TMA（CUtensorMap/cp.async.bulk.tensor）、mbarrier 协议、WGMMA descriptor | `ch13-hopper-tma-wgmma.tex` |
| SM120：TMA+mma.sync+warp specialization、setmaxnreg 池数学与 **shared::cluster/cta 取证小节（C7506/C7508 根因）** | `ch14-sm120-tma-ws.tex` |
| Attention 数学、online softmax 流水、FA1/FA2/FA3 演进 | `ch15-attn-math.tex` |
| FA2（一）：Split-Q + MMA 多级流水、Casual mask | `ch16-fa2-splitq-mma.tex` |
| FA2（二）：TMA + WS 双流水、mbarrier 拓扑、坑七（setmaxnreg 约束） | `ch17-fa2-tma-ws.tex` |
| FA3：双 consumer warpgroup、寄存器再分配 | `ch18-fa3-dual-consumer.tex` |
| FFPA Split-D：大 head_dim（D>256）分块、两阶段 merge | `ch19-ffpa-split-d.tex` |

### Part IV：CuTe（CUTLASS）（配图全部 TikZ 内联）

| 任务/症状 | 章 |
|---|---|
| CuTe 官方白皮书中文译注（第一手定义）：布局表示、张量、布局代数（拼接/合并/复合/补/切分/分块/逆） | `ch19b-cute-whitepaper-zh.tex`（分节在 `chapters/wp/`，wp6 另含 2 张官方架构截图） |
| Layout 基础与代数（colex/mode/compose/inverse/product/divide） | `ch20-cute-layout.tex` |
| Tensor 与 TiledCopy（thrval 引擎、g2s 分解） | `ch21-cute-tensor-tiledcopy.tex` |
| TiledMMA 与 fragment 布局 | `ch22-cute-tiledmma.tex` |
| Swizzle<B,M,S> 与 TMA copy | `ch23-cute-swizzle-tma.tex` |
| CuTe HGEMM 实战（kStage 流水、三级划分） | `ch24-cute-hgemm.tex` |
| CuTe FlashAttention 三实现对照 | `ch25-cute-flash-attn.tex` |
| CuTe FFPA Split-D 类型代数 | `ch26-cute-ffpa.tex` |
| **SM120 持久化 FlashAttention**（persist-D、WS 1+1、persistent CTA、scale 融合；全书唯一超越 cuDNN SDPA 的 attention kernel，240.1/230.2 = 1.04×） | `ch26b-cute-persist-d-flash-attn.tex`（书内第 28 章） |
| **SM120 大 head_dim non-WS Split-D**（tile 128×128、256T 全员 MMA + tid=0 内联 TMA、K/V stages 解耦：(3,2) D320=204.1T=2.93× cuDNN；ffpa-attn split_d 同源教学集，含寄存器 spill 机理） | `ch26c-cute-split-d-sm120.tex`（书内第 29 章） |

### CuTe 进阶深度参考（colfax / cute-zhihu，独立文档）

当 ch20-26 的讲解不够用时，按问题深度依次下钻：

| 需求 | 位置 |
|---|---|
| 布局代数的第一手定义与动机（NVIDIA 官方白皮书，含 PyCuTe 参考实现） | 主书 `ch19b`（`chapters/wp/wp0-7.tex`） |
| 布局代数的数学严格化：范畴论视角（Tuple/Nest 范畴、可处理布局、复合/逻辑乘积/逻辑切分定理）+ 范畴论入门 | `colfax/sec/cf-ch2*.tex`、`cf-ch3*.tex`、`cf-ch4*.tex`、`cf-appa.tex` |
| reed 系列 13 篇：Layout → 代数与几何解释 → Tensor → Copy/MMA 抽象 → Swizzle → Hopper/mbarrier/TMA → 21bit TMA descriptor → simple/pipeline/efficient GEMM | `cute-zhihu/chapters/reed-01~13-*.tex` |
| 竹熙佳处系列 7 篇：tiled copy / tiled mma / compose & inverse / product & divide / TMA copy / permutationMNK / async pipeline | `cute-zhihu/chapters/zhuxi-01~07-*.tex` |
| 知乎文章全文快速 grep（7 篇已落 markdown） | `references/fulltext/*.md` |

查阅顺序建议：概念不清先读主书章 → 定义有疑义查 `ch19b` 白皮书 → 需要数学
严格性（定理证明/范畴论）查 colfax → 想看教学式推导与作者视角查 cute-zhihu。

### Part V：FP8/FP4 HGEMM 篇（量化矩阵乘，PDF 打印章号 30-33；配图全部 TikZ 内联）

| 任务/症状 | 章 |
|---|---|
| e4m3 格式、per-row/per-block 量化、量化前处理 kernel（A/B 两侧量化链逐 bit） | `ch34-fp8-quant-gemm.tex` |
| FP8 CuTe 主 GEMM、在线反量化、cuBLAS BF16 对比、tile 档位扫描 | `ch35-fp8-gemm-cute.tex` |
| NVFP4 线格式（e2m1 数据 + ue4m3 SF）、512 B SF 原子 gmem 布局、两级折叠恒等式、误差模型 | `ch36-fp4-gemm-quant.tex` |
| block-scaled MMA（`mma.sync...m16n8k64.kind::mxf4nvf4.scale_vec::4X`）、SF smem 流水、workspace 六段布局、形状自适应 tile、cuBLAS 对比 | `ch37-fp4-gemm-cute.tex` |

教学源码在 **LeetCUDA 本仓** `fp8_gemm.cuh` / `fp4_gemm.cuh`（在线量化 GEMM，非
ffpa-attn 仓）；bench 入口 `./bin/leetcuda_bench_sm120a.bin --fp8-gemm <N>` /
`--fp4-gemm <N>` / `--fp4-gemm-sweep <N>`。

### Part VI：FP8/FP4 Attention 篇（量化注意力，PDF 打印章号 34-40，文件名 ch27-33；配图全部 TikZ 内联）

| 任务/症状 | 章 |
|---|---|
| 浮点位域、量化格点、ESS 分解等数学基础 | `ch27-quant-attn-math.tex` |
| FP8 量化前处理链（per-tensor/per-block/per-channel） | `ch28-fp8-quant-aux.tex` |
| FP8 persist-D 主 kernel、scale 折叠、reorg-free | `ch29-fp8-persist-d.tex` |
| FP8 split-D 与 M4N2 TiledMMA（大 D 两堵墙） | `ch30-fp8-split-d-m4n2.tex` |
| NVFP4 格式、量化链、KV perm32 置换 | `ch31-fp4-nvfp4-quant.tex` |
| FP4 persist-D、两级 P 量化 | `ch32-fp4-persist-d.tex` |
| FP8/FP4 bench 矩阵、精度方法论、竞品对照 | `ch33-fp8-fp4-bench.tex` |

**配套生产级参考：[ffpa-attn](https://github.com/vipshop/ffpa-attn) 仓库。**
Part VI 各章的教学 kernel 均源自 ffpa-attn（书锚定其某个 commit 的冻结基线），
而 ffpa-attn 本身是 **sm_120（Blackwell）下 CuTe fp16/fp8/fp4 attention 算子的
峰值性能实现**，可作为高质量 attention 算子参考库直接使用：

- **fp16**：超过 FA-2 标准实现；
- **fp8**：持平甚至略优于 SageAttention2；
- **fp4（NVFP4）**：明显优于 SageAttention3；
- **large head_dim 专门优化**：split-D / M4N2 TiledMMA 家族覆盖 D>256 大 D 场景
  （教学对照 ch19/ch26/ch30）。

其余生产级沉淀：native/CuTe kernel 家族与特性矩阵、NHD 布局零拷贝（packed 与
strided 通用）、量化数学与 scale 折叠、性能 RFC 与已证伪清单、bench 与精度验证
方法论。深度任务应**书章 + ffpa-attn 源码/文档联合使用**：

- 仓库定位（记为 `<FFPA_ATTN_DIR>`）：环境变量 `FFPA_ATTN_DIR` → 当前目录及
  父/兄弟目录查找 `ffpa-attn/` → `~/ffpa-attn` → 均无则
  `git clone --depth 1 https://github.com/vipshop/ffpa-attn.git`
- **优先联合 `<FFPA_ATTN_DIR>/.github/skills/ffpa-cuda-understand` skill**：
  该 skill 是 ffpa-attn CUDA 后端的全景知识库（架构分发链路、kernel 家族、
  量化数学原理、RFC 进度与验证方法论），与本书的 Part V/VI 章节互补——书讲
  原理推导与教学实现，ffpa-cuda-understand 讲生产实现与工程决策。

### 附录（快速事实查询）

| 需求 | 附录 |
|---|---|
| common.cuh 逐段解析（TMA/mbarrier/setmaxnreg/WGMMA 宏封装、swizzle） | `appendices/appA-common-toolbox.tex` |
| 全书性能数据汇总（各卡 TFLOPS 基线表；ch11/12/14/16/17/18/24/25 章内另有 `tab:chNN-perf` 实测表） | `appendices/appB-perf-data.tex` |
| 构建指南：build.sh 架构×宏矩阵、-arch 目标选择、book/tests 用法、**setmaxnreg 保留条件** | `appendices/appC-build-guide.tex` |
| 源码索引：kernel 源文件+行号定位表 | `appendices/appD-source-index.tex` |
| 参考文献与延伸阅读（含 PTX ISA 章节映射） | `appendices/appE-references.tex` |

### 代码文件 references（kernels/interview 源码直定位）

按任务路由到章后，配套可参考源码直接按此表打开（行数为实测值，权威口径是
`book/scripts/anchors.yaml`：`frozen_at 2026-09-29` / `repo_commit 3b52c5c`，含
全部 `.cuh` 与 bench/ 下 10 个 `.cu`——bench_leetcuda.cu + 9 模块 .cu；
行号级章节映射见 `appendices/appD-source-index.tex`，其 permalink 基准为老
commit `6cd32de`，该 commit 里 `ffpa_attn.cuh` 只有前 641 行）：

| 文件（相对 `<LeetCUDA_DIR>/kernels/interview/`） | 行数 | 内容 | 对应章 |
|---|---|---|---|
| `base.cuh` | 909 | 架构/Roofline 速查、warp/block 归约与 dot、向量化与原子操作、softmax 三级递进、merge_attn_states、RMS/LayerNorm、RoPE 与转置 | ch01–07 |
| `sgemv.cuh` | 102 | SGEMV 三种划分（warp-per-row K32/K128/K16） | ch08 |
| `sgemm.cuh` | 434 | SGEMM 阶梯（block-tile/Vec4/双缓冲 → TF32 WMMA） | ch09–10 |
| `hgemm.cuh` | 2100 | mma.sync m16n8k16 与 ldmatrix、XOR swizzle、TMA/WGMMA/mbarrier、SM120 TMA+WS、CuTe 对照片段与 CuTe HGEMM 实战 | ch11–14、ch20–21、ch23–24 |
| `flash_attn.cuh` | 4179 | FA 原理头注释、FA2 split-Q+MMA、FA2 TMA+WS、FA3 双 consumer、CuTe FA 三实现、persist-D（`flash_attn_cute_persist_d_sm120` L3635 起） | ch15–18、ch25、ch26b |
| `ffpa_attn.cuh` | 1307 | FFPA Split-D 全篇、双 TiledMMA/traits（M4N2 对照）；L643-1306 为 SM120 大 head_dim non-WS split-D TMA 教学集（ffpa-attn `split_d.cuh` 移植，与 ch26c 同源） | ch19、ch22、ch26、ch26c |
| `fp8_gemm.cuh` | 958 | e4m3 量化前处理（per-row/per-block，L49-300）+ CuTe 主 GEMM、在线反量化（L302-957） | ch34-35 |
| `fp4_gemm.cuh` | 1298 | NVFP4 在线量化 GEMM：A/B 两级量化 kernel 与误差模型（L55-393）+ block-scaled MMA 主 kernel、SF 流水、workspace（L395-1297）；形状自适应入口 `fp4_gemm_use_wide_tile`（MNK≥4096 用 128×256/s6，否则保守 128×128/s4） | ch36-37 |
| `common.cuh` | 803 | TMA/mbarrier/setmaxnreg/WGMMA 宏封装与 swizzle 工具箱（逐段解析见 appA） | ch12–14、appA |
| `bench/bench_leetcuda.cu` | 440 | main+CLI 总装（多 TU）：extern 声明 + 模板实例化 wrapper 调用；host test/bench 拆入 `bench/{base,sgemv,sgemm,hgemm,fp8_gemm,fp4_gemm,flash_attn,ffpa_attn}.cu` + `bench/utils.cu` 共享符号（~30 kernel 的 WHY+HOW 注释与 10 Phase 递进在 `.cuh` 内）；`--bench` 辅助工具与 `--fp8-gemm`/`--fp4-gemm`/`--fp4-gemm-sweep` GEMM 专项 bench CLI 不变（`bin/leetcuda_bench_*.bin` 源，`build.sh --arch sm_120a --jobs 8` 并行编译） | 全书总装、appC |
| `bench/deprecated/bench_attn.cu` | — | （已弃用）FA2 CuTe TMA+MMA+WS vs cuDNN SDPA 专项 bench | ch00、ch17/25 |
| `bench/deprecated/bench_ffpa.cu` | — | （已弃用）FFPA Split-D attention 专项 bench | ch19、ch33 |
| `bench/deprecated/bench_sgemm.cu` | — | （已弃用）`sgemm.cuh` 全 kernel 性能+精度 bench | ch09–12 |
| `bench/deprecated/bench_sdpa.py` | — | （已弃用）PyTorch SDPA 参照计时 | ch00、appB |
| `book/tests/chNN_*.cu` | — | 每章最小正确性测试（CPU fp64 对拍），`build_tests.sh --arch sm_120a --all` | 各章 |

Part VI（ch27–33，量化 attention）教学 kernel 源码在 `<FFPA_ATTN_DIR>/csrc/cuffpa/`
（appD 内附逐章 GitHub permalink）；Part V（ch34–37，量化 GEMM）教学 kernel 源码在
本仓 `kernels/interview/fp8_gemm.cuh` 与 `fp4_gemm.cuh`。

## 第三步：按需读取纪律

1. **先路由后读取**：一次只读任务相关的 1-2 章；章内先看 `\section/\subsection`
   目录再定位行区间（每章 500-900 行，全文读入浪费上下文）。
   **例外**：ch00/ch01 已在文末内嵌，无需外部读取；CuTe 任务的 Part IV
   全部章节仍必须加载即读，不受本条 1-2 章限制。
2. **源码交叉**：书按行号引用 `<LeetCUDA_DIR>/kernels/interview/*.cuh` 冻结基线
   （common/sgemv/sgemm/hgemm/flash_attn/ffpa_attn/fp8_gemm/fp4_gemm.cuh），需要完整实现时用
   `appendices/appD-source-index.tex` 定位再读源文件。
3. **图**：章内图全部为 TikZ inline（`tikzpicture` 就在章节 tex 内）。看图：
   `pdftotext book.pdf -` 按图题关键词定位页码 → `pdftoppm -r 150 -png` 整页
   渲染 → view 工具目检。改图：直接改 tex 内 `tikzpicture`，**改后必须真实渲染
   （standalone 抽图或重建 PDF）+ view 视觉验收无文文/图文重叠**（铁律与配方见
   write-leetcuda-book skill 图管线）。`figures/drawio/`、`figures/ffpa/` 下
   png 为历史留档，仅作旧版对照。
4. **验证 kernel 正确性**：优先跑 `book/tests/`（CPU fp64 对拍，无 cuBLAS 依赖）；
   性能验收用 `bash build.sh --arch sm_120a && ./bin/leetcuda_bench_sm120a.bin --bench
   --bhnd 1,32,8192,128 [--bench-all]`；FP8/FP4 GEMM 专项用 `--fp8-gemm <N>` /
   `--fp4-gemm <N>`（形状自适应行）/ `--fp4-gemm-sweep <N>`（tile 档位扫描）
   （产物落 `bin/`，bench 辅助工具源在
   `kernels/interview/bench/`；构建细节见 appC）。
5. **PDF 兜底**：tex 源不适合读时（如只要结论），`pdftotext book.pdf -` 按页抽取；
   colfax 与 cute-zhihu 两份独立文档同理（`pdftotext colfax/colfax-cute-zh.pdf -`、
   `pdftotext cute-zhihu/cute-zhihu.pdf -`）。
6. 不要修改书的内容除非任务明确要求；本 skill 与书保持同步演进。

## 已知重要结论速查（细节见对应章节）

- **setmaxnreg 被 ptxas 静默丢弃的真根因**：`cp.async.bulk.tensor` 目的地址写
  `shared::cluster` 会被视为 implicit extern 边界（C7506）；写 `shared::cta` 则
  sm_120a/sm_120f 均保留。另需 `__launch_bounds__(N, 1)`（否则 C7508）。取证过程
  见 ch14 专门小节；SASS 助记符是 `USETMAXREG`（grep `setmaxnreg` 查 SASS 会假阴性）。
- **bench A/B 口径**：同轮 SDPA 参照一致性、min-of-N、PSNR/Max-Err 双指标——见
  ch00 与 appB。
- **GPU 规格口径（2026-09-22 运行时实测核准）**：PRO 5000 = **110 SM**
  （`torch.cuda.get_device_properties().multi_processor_count`）；历史误传
  "96 SM" 已全书清除（96 系与 Full GB202 裸 die 的 TPC 数混淆）；5090 = 170 SM、
  PRO 6000 = 188 SM。硬件规格一律以运行时实测为准，书内不写其他卡的 SM 数。
- **性能基线速查（PRO 5000，2026-09-22 `--bench --bench-all`）**：ch26b persist-D
  240.1 TFLOPS = 1.04× cuDNN SDPA（230.2）；HGEMM CuTe Swizzle 追平/超越 cuBLAS
  （最高 1.08× f16、1.48× f32）；明细见各章 `tab:ch11/12/14/16/17/18/24/25-perf`。
- **FP8/FP4 GEMM 基线（PRO 5000，cuBLAS BF16 参照 163.1/163.4 T @ 4096³/8192³）**：
  FP8 ws 128×256/s2 kernel-only 443.7/487.8 T（2.72×/2.99×），e2e 含量化
  186.5/314.6 T；NVFP4 形状自适应（MNK≥4096 宽档 128×256/s6，否则保守
  128×128/s4）：4096³ kernel-only 603.4（NonWS）/614.1 T（WS）=3.69×/3.76×、
  e2e 408.2 T **反超 FP8 e2e**；8192³ 639.6/648.6 T、e2e 467.4 T；4096³ B 离线
  550.5/561.2 T（= kernel 的 91-93%）。量化成本 109 µs 是加性的，宽档下占 e2e
  32%——占比上升是主 kernel 变快所致，不是量化变慢（ch37 有防误读句）。

## 内嵌总体指导：ch00-profiling（正文精编）

> 来源：`chapters/ch00-profiling.tex`。保留算子开发必须立即执行的规则、命令、
> 指标与判据；完整案例叙述和 TikZ 图仍可按需回读源文件。

### 1. 优化闭环与三个反例

**profile 永远都是最重要的。没有 NCU 数据，不改 kernel。** 正确循环是：

```
nsys 定位谁慢 → ncu 解释为什么慢 → 单变量修改 → 复测 diff → 指标闭环
```

- 感觉是装载瓶颈，实际 `math_pipe_throttle=1.32`、`wait=0.82`、
  `long_scoreboard=0.04`：瓶颈是 softmax→PV 依赖链；加深流水/减指令/缩 tile
  均无效，拆成 2 CTA/SM 才得到约 5% 收益。
- 主 kernel 已占 e2e 86–92%，辅助链的优化上限只有 8–14%；先做 nsys 可避免
  在非关键 kernel 上浪费 ncu 时间。
- 融合 bias 注入反而慢 4.2 ms：LDS 进入 consumer 关键路径，`sleeping` 从
  3.5% 升到 22.3%。指标能否决直觉优化。

### 2. nsys 先行，ncu 跟进

- **nsys** 是 trace：看 kernel 占比、launch gap、CPU 是否供不上、流是否重叠、
  fallback。开销低，报告时间可解释真实时间线。
- **ncu** 是 counter replay：看单 kernel 的 SOL、occupancy、stall、cache、
  指令 mix。锁频、多遍回放、串行化，**报告耗时不能当 bench 时间**。
- 只对 nsys 排名前 1–2 的 kernel 做 `ncu --set full`；每次只改一处，无效或
  负优化立即回退。

```bash
nsys profile -t cuda --sample=none --cpuctxsw=none \
  -o .tmp/ch00/report.nsys-rep python bench.py
nsys stats -r cuda_gpu_kern_sum report.nsys-rep  # 优化谁
nsys stats -r cuda_gpu_trace   report.nsys-rep  # gap / overlap

ncu --set full -o .tmp/ch00/report_v1 --target-processes all ./kernel_app
ncu --import report_v1.ncu-rep --page raw --csv > report_v1.csv
ncu --set full -o rep --kernel-name regex:persist_d_ws_fwd_cute_fp8 \
  --target-processes all ./app
ncu --import rep.ncu-rep --page raw 2>&1 | \
  grep -A3 "warps_issue_stalled_\(wait\|math_pipe_throttle\|sleeping\)"
```

采集纪律：

- 只采需要域和最短窗口；大模型 e2e 不加限制采集曾写满 `/tmp`（500 GiB+）。
  `TMPDIR` 和输出放项目 `.tmp/`，单域、最小 repeat、事后清理；路径过长会撞
  Unix socket 107 字节限制。
- 只要 kernel 时间表优先 `torch.profiler`；窗口化用
  `--capture-range=cudaProfilerApi` 配合 `torch.cuda.profiler.start()/stop()`。
- `--kernel-name` 前缀过滤必须写 `regex:`；具体计数器必须 `--page raw`，
  不要把 rule-based CSV 当原始 metric。
- 计数器权限不足可在同微架构代理 GPU 上分析架构级结论；绝对性能必须回目标
  硬件重测。
- A/B 先校验公共参照（通常 ±1%）；抖动大用 min-of-N，profiling 配置与 bench
  逐 flag 对齐。GPU 干扰会让两轮共同膨胀，不能直接算加速比。

### 3. 指标字典与瓶颈决策

| 维度 | 读法与行动 |
|---|---|
| SOL `SM% / MEM%` | SM 高 MEM 低 = compute-bound；反之 memory-bound；双低是 latency-bound，继续查 stall；双高接近上限 |
| occupancy | 区分理论/达到值；限制源看 registers、smem、block/warp slots。occupancy 是手段，不是 KPI |
| 寄存器/spill | 每线程上限 255；看 ptxas `-v`、cuobjdump STACK、SASS LDL/STL 与 local bytes。主要代价常是延迟暴露和指令增多，不只是带宽 |
| `wait` | 固定延迟依赖；拆依赖链、增并行 warp、重排指令 |
| `long_scoreboard` | 等 gmem/TMA；cp.async/TMA、prefetch、加深 stage |
| `short_scoreboard` | 等 smem/MIO；优化布局、减少 smem 往返 |
| `math_pipe_throttle` | 数学管线排队；必须与 `pipe_tensor` 交叉验证，不能直接断言 Tensor Core 饱和 |
| `sleeping/barrier` | 休眠/栅栏；结合忙闲比判断 WS 失衡、退避和同步结构 |
| `not_selected` | 就绪但未被选，通常是健康冗余并行度 |
| L1/L2/DRAM | 同时看命中率、吞吐和合并度。DRAM 高先查 L2 miss；L2 高而 L1 低查 smem/L1 复用 |
| bank conflict | 先看占比和每 warp/cycle 重放，绝对次数大不等于值得修 |
| pipe mix | `pipe_fma/alu` 高而 `pipe_tensor` 低，考虑 MMA；tensor 高且排队下降才支持 TC 利用改善 |

经验阈值不是硬边界：SOL 相差 >20 可分流，双低 <40 查 latency，双高 >60
接近平衡；DRAM >70% 偏带宽墙，L2 hit <50% 且 DRAM >40% 查工作集/局部性，
L1 hit <20% 且 L2 高查 smem 复用。连续 32-bit warp 访问理想为 4 个 32B sector；
明显超标说明流量被访问模式放大。

资源预算实例：384T WS 启动静态上限约 168 regs/thread；稳态
`128×32 + 256×232 = 63488` regs，第二个 CTA 需 126976 > 65536，故只能
1 CTA/SM、12/48 warps，理论 occupancy 25%。M8N1 的 O 累加器约为
`kBr×D/256` regs/thread；`kBr=128,D=512` 已达 256，必须换 M4N2 等结构。

### 4. 实战判据

- 主 kernel 占 86–92%，aux 已达约 1.04 TB/s 时，融合 aux 只省 launch、不减流量；
  大 N wall 差 <0.2%，应转向主 kernel 或引擎 overlap。
- spill 总量大但 local 带宽低时，真正问题可能是 warp 太少、LDL 延迟藏不住。
- `setmaxnreg` 是否保留与 `sm_120a/f` 后缀无关；TMA 目的地址写
  `shared::cluster` 会触发 C7506，写 `shared::cta` 才保留；缺
  `__launch_bounds__(N,1)` 可触发 C7508。
- 最终结论必须标 GPU/架构/日期和计时口径；ncu 时间、不同卡数字、fallback
  kernel、不同 bench 配置都不能混作同一性能口径。

## 内嵌总体指导：ch01-arch-roofline（正文精编）

> 来源：`chapters/ch01-arch-roofline.tex`。保留架构、AI、Roofline、occupancy
> 的公式与使用边界；图示和完整问答可按需回读源文件。

### 1. 执行模型与资源层级

CPU 追求单任务低延迟，GPU 追求海量任务高吞吐，靠 warp 切换而非复杂分支预测
隐藏延迟。同一算法在 GPU 上的主要分类是：

- **memory-bound**：AI 低于机器 ridge point；
- **compute-bound**：AI 高于 ridge point；
- **latency-bound**：并行度不足或依赖链过长，延迟无法隐藏。

SM 速查：warp = 32 threads，是调度/指令发射单元；典型 SM 有 4 个 scheduler，
每个每周期最多发射 1 条指令。寄存器文件每 SM 为 65536×32-bit = 256 KB；
寄存器、smem、block/warp 槽位共同限制驻留量。smem/L1 是片上 SRAM 的不同角色，
block 用 smem 显式交换数据，L1 是透明缓存。

资源例子：64 regs/thread × 256T = 64 KB/block，因此 256 KB 寄存器文件最多
驻留 4 个这种 block。SM120 本机 per-block smem opt-in 上限为 99 KB；不要把
Hopper 的 228 KB 或其他架构上限直接搬用。

内存层级趋势是越靠近 SM，带宽越高、容量越小。H100 数量级参考：寄存器
100+ TB/s、L1/SMEM 约 19 TB/s、L2 约 12 TB/s、HBM3 理论 3.35 TB/s。
L1/L2 数字是微基准量级，不是与 HBM 官方 spec 同精度的常数。优化的共同方向是
让复用数据驻留更高层：寄存器 shuffle、smem tiling、TMA/WS 流水都服务于此。

### 2. 算术强度与 Roofline

\[
AI=\frac{\mathrm{FLOPs}}{\mathrm{Bytes}},\qquad
P=\min(P_{peak},AI\times BW),\qquad AI^*=\frac{P_{peak}}{BW}.
\]

- FLOPs 是运算总量，不是 FLOP/s；乘加各记 1 次。Bytes 是目标存储层级不可再
  省的读写下界，必须包含输出写回；寄存器中间量不计入 HBM Bytes。
- `AI < AI*` 为 memory-bound，`AI > AI*` 为 compute-bound。比较低前必须统一
  dense/sparse、数据精度和实际存储层级口径。

典型推导：

| 算子 | FLOPs / Bytes（fp32） | AI |
|---|---|---:|
| GEMM，M=N=K | `2K³ / [4(MK+KN+MN)]` | `K/6`；K=4096 时 682.67 |
| GEMV | `2MK / [4(MK+K+M)]` | 大 MK 时约 0.5 |
| softmax（本章口径） | 约 `5N / 8N` | 0.625；其他章节可用不同省算口径 |

H100 示例：FP16 TC dense `989/3.35≈295`，FP32 `67/3.35≈20`。AI=100 在
FP32 屋顶是 compute-bound，在 FP16 TC 屋顶仍是 memory-bound；所以迁移到
Tensor Core 不保证更快，若带宽没变，memory-bound 仍受带宽限制。

### 3. Occupancy 与延迟隐藏

\[
\mathrm{occupancy}=\frac{\text{active warps / SM}}{\text{max warps / SM}}.
\]

warp A 等 HBM 时 scheduler 切到 B/C，发射口才不空转。HBM 常见数量级约
500–800 cycles。实际 occupancy 取寄存器、smem、block/warp 槽位约束的最小值，
并受 tail 和负载不均影响。大 thread tile、双缓冲、setmaxnreg 都是在寄存器/
smem/并行度之间换工作点；**低 occupancy + 高 ILP 可优于高 occupancy + 低 ILP**。

### 4. 优化路线与本机标尺

基础路线：coalescing → tiling/复用 → vectorize → thread tile → bank/swizzle →
pipeline → Tensor Core → warp specialization → TMA。不要跳步用高级特性掩盖
算法或访存问题；先用 ch00 的指标证明瓶颈，再选择对应层级。

RTX PRO 5000（sm_120a，2026-09-11 实测）：110 SM、L2 96 MB、smem/block
opt-in 99 KB；GDDR7 理论带宽 `14 GHz×384 bit×2 = 1344 GB/s`；FP32 峰值估算
`128 FMA/clk/SM×110×2.376 GHz×2 = 66.9 TFLOPS`，ridge≈49.8 FLOP/B。
本机复测以此卡为默认口径，其他 GPU/归档数据必须另标。

16 MB 数据可装进 96 MB L2，ch03 因而测得 1862 GB/s > HBM 理论 1344 GB/s：
这里测到的是 L2 而不是 HBM。标定 HBM 时工作集应明显大于 L2，并确认稳定
命中状态；Roofline 的 BW 必须用数据实际所在层级的带宽。

### 5. 高频错误与测试锚点

- Bytes 漏写回：会把 memory-bound 错算成 compute-bound。
- sparse 峰值当 dense：ridge 差 2 倍，bound 类别可能翻转。
- occupancy 当唯一 KPI：忽略 ILP、tile 和流水收益。
- FLOP 与 FLOPS 混用：AI 分子是运算量，Roofline 纵轴是速率。
- 小数据标 HBM：可能实际测到 L2；容量能装下也不等于永不换出。

`book/tests/ch01_roofline.cu` 是 host 侧标尺测试，检查 GEMM/GEMV/softmax/
eadd/relu/dot 的 AI 和本机资源。参考值依次为 682.67、0.5、0.625、1/12、
1/8、0.25。运行方式：`cd book/tests && ./build_tests.sh --ch ch01`。
