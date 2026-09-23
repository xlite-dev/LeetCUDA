---
name: leetcuda-cpp-kernel
description: >-
  LeetCUDA 中文技术书（530 页，XeLaTeX 源）按需查阅 skill——写、优化、调试或
  review CUDA C++/PTX kernel 时的权威参考路由层。当任务涉及：GPU 架构/Roofline/
  occupancy、向量化与 coalescing、warp/block reduce、softmax（online/LSE merge）、
  SGEMV/SGEMM/HGEMM 阶梯优化、mma.sync/ldmatrix/WMMA、XOR/block swizzle、cp.async
  多级流水、TMA/mbarrier/WGMMA（Hopper）、SM120 TMA+warp specialization、
  FlashAttention FA1/FA2/FA3 实现、SM120 持久化 FA（persist-D 超越 cuDNN）、
  split-D 大 head_dim、CuTe Layout/Tensor/
  TiledCopy/TiledMMA（含官方白皮书译注、colfax 范畴论译注、cute-zhihu 合集三份
  深度参考）、FP8/FP4(NVFP4) 量化注意力、nsys/ncu 性能分析、cuobjdump/
  PTX/SASS 取证（含 setmaxnreg 与 shared::cluster/cta 陷阱）、CUDA 面试题时使用。
  生产级 attention 算子参考首选 ffpa-attn（sm_120 CuTe fp16/fp8/fp4 峰值性能：
  fp16 超 FA-2、fp8 持平略优 Sage2、fp4 明显优于 Sage3，含 large head_dim 专门
  优化）。本 skill 不复述书的内容，只做"任务 → 章节/图"的路由，agent 按需读取
  书源文件。
user-invocable: true
---

# leetcuda-cpp-kernel — LeetCUDA 书按需查阅（薄转发层）

本 skill 是 **LeetCUDA 开源技术书** 的路由层：书本身就是参考资源（chapters/*.tex
正文 + figures 配图 + appendices 附录 + colfax/cute-zhihu 两份独立深度参考文档），
skill 只负责把任务路由到正确的章节，**不重写、不复述书内已有内容**。

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
| `chapters/chNN-<slug>.tex` | 36 章正文（ch00–ch33、ch26b、ch26c）+ ch19b 白皮书导读 + `_template.tex` 章模板（中文，XeLaTeX 源；含行号锚定的源码解析、踩坑记录、实测性能表） |
| `chapters/wp/wp0-7.tex` | ch19b 分节正文：CuTe 官方白皮书（Cris Cecka, arXiv:2603.02298）完整中文译注 |
| `figures/drawio/fig-<chNN>-<n>-<slug>/` | 每章配图：`*.png`（用 view 看）、`*.drawio`/`gen.py`（可改后重导出） |
| `figures/ffpa/`、`figures/tikz/`、`figures/misc/` | matplotlib bench 图 / TikZ 图 / 封面等杂项 |
| `colfax/` | 独立译注文档：Colfax《Categorical Foundations for CuTe Layouts》（arXiv:2601.05972）中译，源在 `colfax/sec/`，成品 `colfax/colfax-cute-zh.pdf` |
| `cute-zhihu/` | 独立合集文档：reed（13 篇）+ 竹熙佳处（7 篇）CuTe 知乎系列忠实整理，源在 `cute-zhihu/chapters/`，成品 `cute-zhihu/cute-zhihu.pdf` |
| `references/` | `zhihu-inventory.md`（附录 E 数据源）+ `fulltext/`（7 篇知乎全文 markdown，可直接 grep） |
| `appendices/appA..appE` | 见下文附录路由 |
| `tests/` | 每章最小正确性测试（CPU fp64 对拍，`build_tests.sh --arch sm_120a --all`） |
| `book.pdf` | 编译成品（530 页，可直接 pdftotext 按页抽取） |

## 第二步：任务路由表

先按任务关键词找到章，再读该章 tex（**先 `grep -n '\\\\section'` 看目录，再按行区间读**，
不要整章全文读入）。配图路径中的 `NN` 与章号一致。

### Part I-II：CUDA 基础与 GEMM 阶梯

| 任务/症状 | 章 | 配图 |
|---|---|---|
| 优化前方法论：先 nsys 后 ncu、torch.profiler、warmup/repeat 口径 | `ch00-profiling.tex` | — |
| 架构/执行模型/warp 调度/Roofline/Occupancy、寄存器与 255 墙 | `ch01-arch-roofline.tex` | fig-1-1/1-2 |
| warp shuffle 蝴蝶归约、block reduce、dot product | `ch02-reduce-dot.tex` | fig-2-1 |
| 向量化（f32x4/f16x8 pack）、coalescing、原子操作 | `ch03-vectorize-atomic.tex` | fig-3-1 |
| softmax 三级递进（naive→safe→online） | `ch04-softmax.tex` | fig-4-1 |
| LSE 与分块合并 merge_attn_states | `ch05-lse-merge.tex` | fig-5-1 |
| RMSNorm/LayerNorm（单遍统计） | `ch06-norm.tex` | — |
| RoPE、矩阵转置、bank conflict 与 padding | `ch07-rope-transpose.tex` | fig-7-1 |
| SGEMV 三种划分（memory-bound 专题） | `ch08-sgemv.tex` | fig-8-1 |
| SGEMM 阶梯一：block tile/Vec4/thread tile、双缓冲 | `ch09-sgemm-tiling.tex` | fig-9-1, fig-10-1 |
| SGEMM 阶梯二：TF32 WMMA、cp.async 多级流水 | `ch10-sgemm-tf32-wmma.tex` | fig-10-1 |
| HGEMM：mma.sync m16n8k16、ldmatrix、fragment 布局 | `ch11-hgemm-mma.tex` | fig-11-1/11-2a |
| XOR swizzle、寄存器双缓冲、block swizzle、L2 复用 | `ch12-hgemm-swizzle.tex` | fig-12-1/12-2/12-3 |

### Part III：Hopper/SM120 与 FlashAttention

| 任务/症状 | 章 | 配图 |
|---|---|---|
| TMA（CUtensorMap/cp.async.bulk.tensor）、mbarrier 协议、WGMMA descriptor | `ch13-hopper-tma-wgmma.tex` | fig-13-1/13-2/13-3 |
| SM120：TMA+mma.sync+warp specialization、setmaxnreg 池数学与 **shared::cluster/cta 取证小节（C7506/C7508 根因）** | `ch14-sm120-tma-ws.tex` | fig-14-1/14-2 |
| Attention 数学、online softmax 流水、FA1/FA2/FA3 演进 | `ch15-attn-math.tex` | fig-15-1/15-2 |
| FA2（一）：Split-Q + MMA 多级流水、Casual mask | `ch16-fa2-splitq-mma.tex` | fig-16-1 |
| FA2（二）：TMA + WS 双流水、mbarrier 拓扑、坑七（setmaxnreg 约束） | `ch17-fa2-tma-ws.tex` | fig-17-1/17-2 |
| FA3：双 consumer warpgroup、寄存器再分配 | `ch18-fa3-dual-consumer.tex` | fig-18-1 |
| FFPA Split-D：大 head_dim（D>256）分块、两阶段 merge | `ch19-ffpa-split-d.tex` | fig-19-1/19-2 |

### Part IV：CuTe（CUTLASS）

| 任务/症状 | 章 | 配图 |
|---|---|---|
| CuTe 官方白皮书中文译注（第一手定义）：布局表示、张量、布局代数（拼接/合并/复合/补/切分/分块/逆） | `ch19b-cute-whitepaper-zh.tex`（分节在 `chapters/wp/`） | — |
| Layout 基础与代数（colex/mode/compose/inverse/product/divide） | `ch20-cute-layout.tex` | fig-20-1~20-9 |
| Tensor 与 TiledCopy（thrval 引擎、g2s 分解） | `ch21-cute-tensor-tiledcopy.tex` | fig-21-1~21-4 |
| TiledMMA 与 fragment 布局 | `ch22-cute-tiledmma.tex` | fig-22-1/22-2 |
| Swizzle<B,M,S> 与 TMA copy | `ch23-cute-swizzle-tma.tex` | fig-23-1/23-2 |
| CuTe HGEMM 实战（kStage 流水、三级划分） | `ch24-cute-hgemm.tex` | fig-24-2/24-3 |
| CuTe FlashAttention 三实现对照 | `ch25-cute-flash-attn.tex` | fig-25-1 |
| CuTe FFPA Split-D 类型代数 | `ch26-cute-ffpa.tex` | fig-26-1 |
| **SM120 持久化 FlashAttention**（persist-D、WS 1+1、persistent CTA、scale 融合；全书唯一超越 cuDNN SDPA 的 attention kernel，240.1/230.2 = 1.04×） | `ch26b-cute-persist-d-flash-attn.tex`（书内第 28 章） | TikZ 内联 |
| **SM120 大 head_dim non-WS Split-D**（tile 128×128、256T 全员 MMA + tid=0 内联 TMA、K/V stages 解耦：(3,2) D320=204.1T=2.93× cuDNN；ffpa-attn split_d 同源教学集，含寄存器 spill 机理） | `ch26c-cute-split-d-sm120.tex`（书内第 29 章） | TikZ 内联 |

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

### Part V：FP8/FP4 量化注意力

| 任务/症状 | 章 | 配图 |
|---|---|---|
| 浮点位域、量化格点、ESS 分解等数学基础 | `ch27-quant-attn-math.tex` | fig-27-1/27-2/27-3 |
| FP8 量化前处理链（per-tensor/per-block/per-channel） | `ch28-fp8-quant-aux.tex` | fig-28-1/28-2 |
| FP8 persist-D 主 kernel、scale 折叠、reorg-free | `ch29-fp8-persist-d.tex` | fig-29-1/29-2/29-3 |
| FP8 split-D 与 M4N2 TiledMMA（大 D 两堵墙） | `ch30-fp8-split-d-m4n2.tex` | fig-30-1/30-2 |
| NVFP4 格式、量化链、KV perm32 置换 | `ch31-fp4-nvfp4-quant.tex` | fig-31-1/31-2/31-3 |
| FP4 persist-D、两级 P 量化 | `ch32-fp4-persist-d.tex` | fig-32-1/32-2 |
| FP8/FP4 bench 矩阵、精度方法论、竞品对照 | `ch33-fp8-fp4-bench.tex` | `figures/ffpa/` |

**配套生产级参考：[ffpa-attn](https://github.com/xlite-dev/ffpa-attn) 仓库。**
Part V 各章的教学 kernel 均源自 ffpa-attn（书锚定其某个 commit 的冻结基线），
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
  `git clone --depth 1 https://github.com/xlite-dev/ffpa-attn.git`
- **优先联合 `<FFPA_ATTN_DIR>/.github/skills/ffpa-cuda-understand` skill**：
  该 skill 是 ffpa-attn CUDA 后端的全景知识库（架构分发链路、kernel 家族、
  量化数学原理、RFC 进度与验证方法论），与本书的 Part V 章节互补——书讲
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

按任务路由到章后，配套可参考源码直接按此表打开（行数为实测值；行号级
冻结映射见 `appendices/appD-source-index.tex`，基准 commit `6cd32de`）：

| 文件（相对 `<LeetCUDA_DIR>/kernels/interview/`） | 行数 | 内容 | 对应章 |
|---|---|---|---|
| `base.cuh` | 909 | 架构/Roofline 速查、warp/block 归约与 dot、向量化与原子操作、softmax 三级递进、merge_attn_states、RMS/LayerNorm、RoPE 与转置 | ch01–07 |
| `sgemv.cuh` | 102 | SGEMV 三种划分（warp-per-row K32/K128/K16） | ch08 |
| `sgemm.cuh` | 434 | SGEMM 阶梯（block-tile/Vec4/双缓冲 → TF32 WMMA） | ch09–10 |
| `hgemm.cuh` | 2100 | mma.sync m16n8k16 与 ldmatrix、XOR swizzle、TMA/WGMMA/mbarrier、SM120 TMA+WS、CuTe 对照片段与 CuTe HGEMM 实战 | ch11–14、ch20–21、ch23–24 |
| `flash_attn.cuh` | 4179 | FA 原理头注释、FA2 split-Q+MMA、FA2 TMA+WS、FA3 双 consumer、CuTe FA 三实现、persist-D（`flash_attn_cute_persist_d_sm120` L3635 起） | ch15–18、ch25、ch26b |
| `ffpa_attn.cuh` | 641 | FFPA Split-D 全篇、双 TiledMMA/traits（M4N2 对照） | ch19、ch22、ch26 |
| `common.cuh` | 803 | TMA/mbarrier/setmaxnreg/WGMMA 宏封装与 swizzle 工具箱（逐段解析见 appA） | ch12–14、appA |
| `notes-v2.cu` | 5219 | 面试背题主编译单元：include 全部 `.cuh`，~30 kernel 的 WHY+HOW 注释与 10 Phase 递进，`--bench` harness（`bin/notes_v2_*.bin` 源） | 全书总装、appC |
| `bench/bench_attn.cu` | — | FA2 CuTe TMA+MMA+WS vs cuDNN SDPA 专项 bench | ch00、ch17/25 |
| `bench/bench_ffpa.cu` | — | FFPA Split-D attention 专项 bench | ch19、ch33 |
| `bench/bench_sgemm.cu` | — | `sgemm.cuh` 全 kernel 性能+精度 bench | ch09–12 |
| `bench/bench_sdpa.py` | — | PyTorch SDPA 参照计时 | ch00、appB |
| `book/tests/chNN_*.cu` | — | 每章最小正确性测试（CPU fp64 对拍），`build_tests.sh --arch sm_120a --all` | 各章 |

Part V（ch27–33）教学 kernel 源码在 `<FFPA_ATTN_DIR>/csrc/cuffpa/`（appD 内
附逐章 GitHub permalink）。

## 第三步：按需读取纪律

1. **先路由后读取**：一次只读任务相关的 1-2 章；章内先看 `\section/\subsection`
   目录再定位行区间（每章 500-900 行，全文读入浪费上下文）。
2. **源码交叉**：书按行号引用 `<LeetCUDA_DIR>/kernels/interview/*.cuh` 冻结基线
   （common/sgemv/sgemm/hgemm/flash_attn/ffpa_attn.cuh），需要完整实现时用
   `appendices/appD-source-index.tex` 定位再读源文件。
3. **图**：`figures/drawio/fig-*/` 下 png 用 view 工具直接看；需要改图时改同目录
   `gen.py`/`.drawio` 后用 `drawio-headless -x -f png -s 3` 重导出（输出落回同目录）。
4. **验证 kernel 正确性**：优先跑 `book/tests/`（CPU fp64 对拍，无 cuBLAS 依赖）；
   性能验收用 `bash build.sh --arch sm_120a && ./bin/notes_v2_sm120a.bin --bench
   --bhnd 1,32,8192,128 [--bench-all]`（产物落 `bin/`，bench harness 源在
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
