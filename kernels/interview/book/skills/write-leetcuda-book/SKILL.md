---
name: write-leetcuda-book
description: >-
  把 LeetCUDA kernels/interview 教学源码（base/sgemv/sgemm/hgemm/flash_attn/ffpa_attn.cuh）写成中文 CUDA 技术书（4 Part 26 章，XeLaTeX→PDF）。当任务涉及：撰写或修改书稿章节（chapters/chNN-*.tex）、每章最小测试（book/tests/chNN_*.cu）、知乎资料收集与提炼、drawio 图重建或新建、源码注释核查、勾选 RFC 进度、构建/审校 book.pdf 时使用。规范源=BOOK_PLAN.md（章节结构/每章 DoD 八条/源码冻结/容差表/图片管线），执行跟踪=RFC.md（RFC-0..J 里程碑 + 26 章卡片 + 图清单 + 知乎参考种子表）。
user-invocable: true
---

# write-leetcuda-book — LeetCUDA 成书工作流

本 skill 目录即唯一事实源，两份核心文档（同目录）：

| 文件 | 角色 |
|---|---|
| `BOOK_PLAN.md` | **规范源**：26 章结构、每章模板与 DoD 八条、素材/测试/图表/公式工程、LaTeX 管线、风险与验收 |
| `RFC.md` | **执行跟踪**：RFC-0..J 里程碑 checkbox、每章执行卡片、图清单登记表、知乎参考种子表 |

绝对路径：`/workspace/dev/vipshop/LeetCUDA/kernels/interview/book/skills/write-leetcuda-book/`（已软链到 `/workspace/dev/vipshop/.github/skills/write-leetcuda-book`）。

## 工作流（每次任务）

1. **领任务**：读 `RFC.md` 里程碑总览表 + 对应章节的执行卡片，按依赖顺序（0→A→(B∥C)→D→E→F→(G∥H∥I)→J）取未勾选项。
2. **读规范**：`BOOK_PLAN.md` §3 章节卡片（源码区间/宏/公式/图/参考/测试映射）+ §4 模板与 DoD + 相关工程节。
3. **执行**：章节任务四合一 = 正文 tex + 最小测试 .cu + 注释核查 + 增量 bench。
4. **验收**：对照 §4.2 DoD 八条逐条自检（编译零 error / 锚点断言 / CHECKLOG / 测试 PASS / bench 落盘 / 图表登记 / 延伸阅读 / pdftotext 抽查）。
5. **回写**：RFC.md 勾选 `- [x]` + 追加日期；发现问题记 CHECKLOG / RFC 对应位置。

## 硬规则速查（违反必返工）

1. **源码冻结**：`*.cuh` / notes-v2.cu 在 RFC-A 登记 SHA256 后**零改动**。注释错误只记 `book/CHECKLOG.md` + 正文「勘误与考据」，不回写源码。
2. **正文形态**：原理讲解 + 关键代码段（每段 ≤40 行，`linerange` 多段裁剪，`firstnumber=auto`）；完整代码给附录 D 的 **GitHub commit permalink**；每章正文（非 listings）≥3500 字。
3. **测试**：`book/tests/chNN_*.cu` 无 cuBLAS/cuDNN 依赖，CPU fp64 参考 + 容差三档（F32Acc 1e-3 / F16Acc 5e-2 / TF32 1e-2），规模 ≤512，arch 不在位输出 SKIP；行数 ≤300（基础章）/≤500（复杂章）。
4. **知乎素材**：五步法（枚举→全文落 `zhihu-analysis/`→提炼→成文不照抄→附录 E 汇总）；**图片直接引用+出处标注**（作者/文章/链接/日期），原图归档 `book/figures/zhihu/`；水印/作者角标/平台 logo 一律标记为非内容元素，不得进入书内图。
5. **drawio 管线**（知乎图→正式图主路径）：`drawio-reconstruction` skill 为主（inventory→重建→审查闭环），`drawio-diagram-builder`/`drawio-flow-forge` 辅助（ASCII 升级/新建）。**执行模式：主 agent 直接做，不派 task agent**（闭环子代理实测易卡死，用户 2026-09-11 拍板）。审查结论如实记 audit（自审需标注）。
6. **行文风格（作者基线）**：参考 @DefTruth《图解:从Online-Softmax到FlashAttention V1/V2/V3》与《WINT8/4》系列——**原理一定要讲细**：公式逐项展开、指令逐 bit/逐字段解释、先直觉后形式化再代码；中文叙述、术语保留英文。其余可自由发挥。
7. **注释核查五类**：F1 架构事实（Prog Guide+cutlass skill arch guides）/ F2 PTX（本地 ptx-docs）/ F3 数学（独立推导）/ F4 性能断言（标来源）/ F5 历史陈述（对原论文）。
8. **构建**：CWD=`book/`，`TEXMFCNF=../tex/: xelatex -interaction=nonstopmode book.tex` ×2（或 `./build.sh`）；本机已有 xelatex+ctexbook+Noto CJK（已验证）。

## 工具链速查

```bash
# drawio headless 导出（已装 v31.4.5 deb + xvfb；Ubuntu noble 无 apt 包，须用 GitHub releases deb）
DRAWIO_PATH=/usr/local/bin/drawio-headless \
  python3 /workspace/dev/vipshop/.github/skills/drawio-reconstruction/scripts/export_drawio.py in.drawio out.png
python3 /workspace/dev/vipshop/.github/skills/drawio-reconstruction/scripts/check_drawio.py in.drawio
# VS Code drawio 插件（用户已装）：用于人工查看/微调 .drawio；自动导出仍走上面 CLI

# 知乎 CLI（已授权；--count 上限 10）
/root/.local/share/zhihu-cli/current/zhihu-cli search zhihu --query "..." --count 10
python3 /workspace/dev/vipshop/.github/skills/zhihu/scripts/fetch_column_fulltext.py <专栏URL>   # 枚举/取全文

# 源码锚点断言（RFC-A 后可用）
python3 /workspace/dev/vipshop/LeetCUDA/kernels/interview/book/scripts/verify_anchors.py --ch chNN
```

## 目录结构（book/，相对本 skill 的 `../../`）

```
kernels/interview/book/
├── book.tex / preamble.tex / build.sh      # RFC-0 产出
├── chapters/chNN-*.tex … appendices/       # RFC-C..I 产出
├── figures/{ascii/,zhihu/,drawio/,tikz/}
├── tests/{common_test.h, chNN_*.cu, build_tests.sh}
├── notes/ references/ scripts/ CHECKLOG.md
├── skills/write-leetcuda-book/             # 本 skill（规范+跟踪）
└── .tmp/                                   # 测试/bench 临时产物（按任务建子目录）
```

## Examples

- `examples/drawio-recon-m1/`：drawio 重建 smoke test 固化样例（重建件 + headless 导出预览 + audit + 复现命令）。正式重建任务照此流程执行；原图在 drawio-reconstruction skill 的 `examples/m1.png`（路径引用，不复制）。

## 关联 skill

- `drawio-reconstruction`（知乎图重建主路径）、`drawio-diagram-builder` / `drawio-flow-forge`（新建图）
- `zhihu`（资料五步法）
- `cuda-cpp-kernel`（ptx-docs=F2 核查源）、`cutlass-cpp-kernel`（arch guides=F1 核查源）、`cuda-auto-tune`（bench 复测方法论）
- `ffpa-cuda-understand`（ch19/26 大 D split-D 背景）
