---
name: leetcuda-cpp-kernel
description: >-
  LeetCUDA 中文技术书（426 页，XeLaTeX 源）按需查阅 skill——写、优化、调试或
  review CUDA C++/PTX kernel 时的权威参考路由层。当任务涉及：GPU 架构/Roofline/
  occupancy、向量化与 coalescing、warp/block reduce、softmax（online/LSE merge）、
  SGEMV/SGEMM/HGEMM 阶梯优化、mma.sync/ldmatrix/WMMA、XOR/block swizzle、cp.async
  多级流水、TMA/mbarrier/WGMMA（Hopper）、SM120 TMA+warp specialization、
  FlashAttention FA1/FA2/FA3 实现、split-D 大 head_dim、CuTe Layout/Tensor/
  TiledCopy/TiledMMA、FP8/FP4(NVFP4) 量化注意力、nsys/ncu 性能分析、cuobjdump/
  PTX/SASS 取证（含 setmaxnreg 与 shared::cluster/cta 陷阱）、CUDA 面试题时使用。
  本 skill 不复述书的内容，只做"任务 → 章节/图"的路由，agent 按需读取书源文件。
user-invocable: true
---

# leetcuda-cpp-kernel — LeetCUDA 书按需查阅（薄转发层）

本 skill 是 **LeetCUDA 开源技术书** 的路由层：书本身就是参考资源（chapters/*.tex
正文 + figures 配图 + appendices 附录），skill 只负责把任务路由到正确的章节，
**不重写、不复述书内已有内容**。

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
| `chapters/chNN-<slug>.tex` | 34 章正文（中文，XeLaTeX 源；含行号锚定的源码解析、踩坑记录、性能数据） |
| `figures/drawio/fig-<chNN>-<n>-<slug>/` | 每章配图：`*.png`（用 view 看）、`*.drawio`/`gen.py`（可改后重导出） |
| `figures/ffpa/`、`figures/tikz/`、`figures/misc/` | matplotlib bench 图 / TikZ 图 / 封面等杂项 |
| `appendices/appA..appE` | 见下文附录路由 |
| `tests/` | 每章最小正确性测试（CPU fp64 对拍，`build_tests.sh --arch sm_120a --all`） |
| `book.pdf` | 编译成品（426 页，可直接 pdftotext 按页抽取） |

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
| Layout 基础与代数（colex/mode/compose/inverse/product/divide） | `ch20-cute-layout.tex` | fig-20-1~20-9 |
| Tensor 与 TiledCopy（thrval 引擎、g2s 分解） | `ch21-cute-tensor-tiledcopy.tex` | fig-21-1~21-4 |
| TiledMMA 与 fragment 布局 | `ch22-cute-tiledmma.tex` | fig-22-1/22-2 |
| Swizzle<B,M,S> 与 TMA copy | `ch23-cute-swizzle-tma.tex` | fig-23-1/23-2 |
| CuTe HGEMM 实战（kStage 流水、三级划分） | `ch24-cute-hgemm.tex` | fig-24-2/24-3 |
| CuTe FlashAttention 三实现对照 | `ch25-cute-flash-attn.tex` | fig-25-1 |
| CuTe FFPA Split-D 类型代数 | `ch26-cute-ffpa.tex` | fig-26-1 |

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
而 ffpa-attn 本身沉淀了大量 **FP8/FP4 Attention 与 large head\_dim attention
的生产级最佳实践**（native/CuTe fp16/fp8/fp4 kernel 家族、特性矩阵与限制、
布局零拷贝、split-D 大 D 技术、量化数学、性能 RFC 与已证伪清单、bench 与
精度验证方法论）。深度任务应**书章 + ffpa-attn 源码/文档联合使用**：

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
| 全书性能数据汇总（各卡 TFLOPS 基线表） | `appendices/appB-perf-data.tex` |
| 构建指南：build.sh 架构×宏矩阵、-arch 目标选择、book/tests 用法、**setmaxnreg 保留条件** | `appendices/appC-build-guide.tex` |
| 源码索引：kernel 源文件+行号定位表 | `appendices/appD-source-index.tex` |
| 参考文献与延伸阅读（含 PTX ISA 章节映射） | `appendices/appE-references.tex` |

## 第三步：按需读取纪律

1. **先路由后读取**：一次只读任务相关的 1-2 章；章内先看 `\section/\subsection`
   目录再定位行区间（每章 500-900 行，全文读入浪费上下文）。
2. **源码交叉**：书按行号引用 `<LeetCUDA_DIR>/kernels/interview/*.cuh` 冻结基线
   （common/sgemv/sgemm/hgemm/flash_attn/ffpa_attn.cuh），需要完整实现时用
   `appendices/appD-source-index.tex` 定位再读源文件。
3. **图**：`figures/drawio/fig-*/` 下 png 用 view 工具直接看；需要改图时改同目录
   `gen.py`/`.drawio` 后用 `drawio-headless -x -f png -s 3` 重导出（输出落回同目录）。
4. **验证 kernel 正确性**：优先跑 `book/tests/`（CPU fp64 对拍，无 cuBLAS 依赖）；
   性能验收用 `build.sh --arch sm_120f && ./notes_v2_sm120f.bin --bench ...`
   （构建细节见 appC）。
5. **PDF 兜底**：tex 源不适合读时（如只要结论），`pdftotext book.pdf -` 按页抽取。
6. 不要修改书的内容除非任务明确要求；本 skill 与书保持同步演进。

## 已知重要结论速查（细节见对应章节）

- **setmaxnreg 被 ptxas 静默丢弃的真根因**：`cp.async.bulk.tensor` 目的地址写
  `shared::cluster` 会被视为 implicit extern 边界（C7506）；写 `shared::cta` 则
  sm_120a/sm_120f 均保留。另需 `__launch_bounds__(N, 1)`（否则 C7508）。取证过程
  见 ch14 专门小节；SASS 助记符是 `USETMAXREG`（grep `setmaxnreg` 查 SASS 会假阴性）。
- **bench A/B 口径**：同轮 SDPA 参照一致性、min-of-N、PSNR/Max-Err 双指标——见
  ch00 与 appB。
