---
name: write-leetcuda-book
description: >-
  把 LeetCUDA kernels/interview 教学源码（base/sgemv/sgemm/hgemm/flash_attn/ffpa_attn.cuh）+ ffpa-attn CuTe sm_120 源码（csrc/cuffpa/cute fp8/fp4，commit 861d75e）写成中文 CUDA 技术书（5 Part 36 章 + CuTe 白皮书导读 + 5 附录，XeLaTeX→PDF，终态 530 页零 error）。RFC-0..M 已全部完成（2026-09-23），当前任务形态 = 增补章（RFC-L/M 模式）/复审/勘误/图重绘。当任务涉及：撰写或修改书稿章节（chapters/chNN*.tex）、每章最小测试（book/tests/chNN_*.cu）、ffpa_attn.bench 验证、知乎资料收集与提炼、TikZ inline 图新建（主路径）与存量 drawio 图维护、源码注释核查、勾选 RFC 进度、构建/审校 book.pdf 时使用。规范源=BOOK_PLAN.md（章节结构/每章 DoD 八条/源码冻结/容差表/图片管线），执行跟踪=RFC.md（RFC-0..M 里程碑 + 各章卡片 + 图清单 + 知乎参考种子表）。
user-invocable: true
---

# write-leetcuda-book — LeetCUDA 成书工作流

本 skill 目录即唯一事实源，两份核心文档（同目录）：

| 文件 | 角色 |
|---|---|
| `BOOK_PLAN.md` | **规范源**：5 Part 35 章+导读结构、每章模板与 DoD 八条（含 Part V 特化 DoD）、素材/测试/图表/公式工程、LaTeX 管线、风险与验收 |
| `RFC.md` | **执行跟踪**：RFC-0..M 里程碑 checkbox（B 已取消）、每章执行卡片、图清单登记表、知乎参考种子表 |

绝对路径：`/workspace/dev/vipshop/LeetCUDA/kernels/interview/book/skills/write-leetcuda-book/`（已软链到 `/workspace/dev/vipshop/.github/skills/write-leetcuda-book`）。

## 工作流（每次任务）

1. **领任务**：读 `RFC.md` 里程碑总览表 + 对应章节的执行卡片。RFC-0..M 已全部收口（RFC-B 于 2026-09-18 取消，职能归附录 E + drawio 全量重建），存量只剩零星复审项（如 B.5 RoPE 参考）；新需求 = 用户新点名的增补章/增强项，按 RFC-L/M 模式新开里程碑条目（源码整合→测试→bench→正文→接线→验收→code review→用户增强）逐项推进。
2. **读规范**：`BOOK_PLAN.md` §3 章节卡片（源码区间/宏/公式/图/参考/测试映射）+ §4 模板与 DoD + 相关工程节。
3. **执行**：章节任务四合一 = 正文 tex + 最小测试 .cu + 注释核查 + 增量 bench。
4. **验收**：对照 §4.2 DoD 八条逐条自检（编译零 error / 锚点断言 / CHECKLOG / 测试 PASS / bench 落盘 / 图表登记 / 延伸阅读 / pdftotext 抽查）。
5. **回写**：RFC.md 勾选 `- [x]` + 追加日期；发现问题记 CHECKLOG / RFC 对应位置。

## 硬规则速查（违反必返工）

1. **源码冻结**：`*.cuh` / notes-v2.cu 在 RFC-A 登记 SHA256 后**零改动**。注释错误只记 `book/CHECKLOG.md` + 正文「勘误与考据」，不回写源码。
2. **正文形态**：原理讲解 + 关键代码段（每段 ≤40 行，`linerange` 多段裁剪，`firstnumber=auto`）；完整代码给附录 D 的 **GitHub commit permalink**；每章正文（非 listings）≥3500 字。
3. **测试**：`book/tests/chNN_*.cu` 无 cuBLAS/cuDNN 依赖，CPU fp64 参考 + 容差三档（F32Acc 1e-3 / F16Acc 5e-2 / TF32 1e-2），规模 ≤512，arch 不在位输出 SKIP；行数 ≤300（基础章）/≤500（复杂章）。
4. **知乎素材**：五步法（枚举→全文落 `zhihu-analysis/`→提炼→成文不照抄→附录 E 汇总）；**图片直接引用+出处标注**（作者/文章/链接/日期），原图归档 `book/figures/zhihu/`；水印/作者角标/平台 logo 一律标记为非内容元素，不得进入书内图。
5. **图管线（TikZ 主路径）**：新图默认 **TikZ inline** 写在章节 tex 内，用 `tikz-diagrams` skill（模板/编译渲染/视觉 QA 工具链；ch26b 六图先例，上游 5be940f 亦将存量图大规模迁为 tikzpicture）。drawio 降为**存量维护**路径（ch20-32 仍 36 处 png 引用）：`drawio-reconstruction` skill 为主（inventory→重建→审查闭环），`drawio-diagram-builder`/`drawio-flow-forge` 辅助。**执行模式：主 agent 直接做，不派 task agent**（闭环子代理实测易卡死，用户 2026-09-11 拍板）。审查结论如实记 audit（自审需标注）。drawio 质量三查（XML well-formed + COVER=0 + 渲染宽不越界，见「drawio 制图铁律」）；`.drawio` 源文件必须入库（用户要求可手改，勿只提交 PNG）。
6. **行文风格（作者基线）**：参考 @DefTruth《图解:从Online-Softmax到FlashAttention V1/V2/V3》与《WINT8/4》系列——**原理一定要讲细**：公式逐项展开、指令逐 bit/逐字段解释、先直觉后形式化再代码；中文叙述、术语保留英文。其余可自由发挥。
7. **注释核查五类**：F1 架构事实（Prog Guide+cutlass skill arch guides）/ F2 PTX（本地 ptx-docs）/ F3 数学（独立推导）/ F4 性能断言（标来源）/ F5 历史陈述（对原论文）。
8. **构建**：CWD=`book/`，`TEXMFCNF=../tex/: xelatex -interaction=nonstopmode book.tex` ×2（或 `./build.sh`）；本机已有 xelatex+ctexbook+Noto CJK（已验证）。TOC 全量装载报 `main_memory capacity exceeded`（PDF 目录丢失）时须 `fmtutil-sys --byfmt xelatex` 重建 fmt——`texmf.cnf` 的 main_memory 仅 fmt 生成期生效，改配置不重建无效（2026-09-22 L.9 根因）。

## 工具链速查

```bash
# drawio headless 导出（已装 v31.4.5 deb + xvfb；Ubuntu noble 无 apt 包，须用 GitHub releases deb）
DRAWIO_PATH=/usr/local/bin/drawio-headless \
  python3 /workspace/dev/vipshop/.github/skills/drawio-reconstruction/scripts/export_drawio.py in.drawio out.png
python3 /workspace/dev/vipshop/.github/skills/drawio-reconstruction/scripts/check_drawio.py in.drawio
# 也可直接：/usr/local/bin/drawio-headless -x -s 3 -o out.png in.drawio
# VS Code drawio 插件（用户已装）：用于人工查看/微调 .drawio；自动导出仍走上面 CLI

# 全书图几何体检（book/scripts/，2026-09 新增；改动后必须复检至 0）
python3 scripts/check_overlap.py figures/drawio/*/*.drawio   # 文字覆盖（COVER = 后画实心框盖住先画 text）
python3 scripts/check_text_fit.py figures/drawio/*/*.drawio  # 渲染宽 vs geometry 宽 + 页宽越界

# 知乎 CLI（已授权；--count 上限 10）
/root/.local/share/zhihu-cli/current/zhihu-cli search zhihu --query "..." --count 10
python3 /workspace/dev/vipshop/.github/skills/zhihu/scripts/fetch_column_fulltext.py <专栏URL>   # 枚举/取全文

# 源码锚点断言（RFC-A 后可用）
python3 /workspace/dev/vipshop/LeetCUDA/kernels/interview/book/scripts/verify_anchors.py --ch chNN
```

## drawio 制图铁律（存量图维护适用；2026-09 全书返工沉淀）

**生成器结构**：每图一份自包含 `gen.py`（`F` 类 box/text/line + 输出 `python3 gen.py <输出文件全名>`，如 `gen.py fig-12-2.drawio`——误传目录名会产生孤儿文件）。批量期的权威源在 `.tmp/drawio/gen_batch_{a,b1,b2,c1,c2,d,e}.py`（**`.gitignore` 忽略**），每图目录的 gen.py 是其回流拷贝（入库、自包含）。**改图必须改 `.tmp` 权威源再回流**，直接改目录版会在下一轮批量重跑时被静默覆盖（fig-14-2 实测踩中）；回流后用 `cmp -s` 校验。

**XML 铁律**：
1. `value` 禁裸 `<` / `&`：`Tile<64,64,16>`、`Swizzle<3,4,3>`、`->`、`R^16` 类写法的 `<` 必须转义 `&lt;`。裸 `<` 毁掉 well-formed → drawio **静默**渲染成细条（fig-26-1 只输出 2522x74，书里等于废图）。
2. 每个生成器末尾做 `ET.fromstring(xml)` 自检；全书定期 `python3 -c "ET.parse"` 全量扫（本次 2/39 破损）。
3. `text` 元素**不自动换行**：geometry 宽小于渲染宽，文字直接溢出色框（甚至越出页宽）。长注释必须手工拆多行，或把 geometry 宽改到 ≥ 估算渲染宽。
4. 宽度估算：CJK/全角 × `fontSize×1.02`，ASCII × `fontSize×0.58`（含全角标点按 CJK 计）。`check_text_fit.py` 用此公式报 `[OOB]`（越页宽）/`[COLLIDE]`（侵入右邻框）。
5. z-order：后画的**实心 box 会盖住**先画的 text。右上角注释卡与左侧网格文字相交时，把文字移出卡区（`check_overlap.py` 的 `COVER` 即此类）。
6. 导出体检：`PNG 宽 ≈ pageWidth × 3`（`-s 3`），高宽比 ≈ pageHeight/pageWidth。比例异常 = 渲染崩了（先查 XML well-formed），不要直接 `includegraphics`。

**常见布局病**（本轮回修）：说明行（y 40–60）与列头框（y 50+）净空不足 → 原点下移 16–26px；卡片内文字 409px 渲染塞 300px 框 → 框加宽 + 拆行；左表右侧的 `...` 居左文本 geometry 过宽侵入相邻框 → 宽度收到渲染宽；「标题 + 图」跨页 → 图示节前 `\needspace{图高+30mm}`（mm = `width_tw × 181 × png_h/png_w`，`preamble.tex` 已 `\usepackage{needspace}`）。

**其它约定**：CJK 禁 `bold`（易糊）；两位数格宽 ≥54px；纯色缩略格无文字；`.drawio`/`.png`/`gen.py` 三件套全部入库。

## 目录结构（book/，相对本 skill 的 `../../`）

```
kernels/interview/book/
├── book.tex / preamble.tex / build.sh      # RFC-0 产出
├── chapters/ch00-ch33 + ch19b/ch26b/ch26c 增补 + wp/ 白皮书导读 … appendices/  # RFC-C..M 产出
├── figures/{ascii,zhihu,drawio,ffpa,misc,tikz}/
├── tests/{common_test.h, chNN_*.cu, build_tests.sh}
├── notes/ references/ scripts/ CHECKLOG.md
├── skills/write-leetcuda-book/             # 本 skill（规范+跟踪）
└── .tmp/                                   # 测试/bench 临时产物（按任务建子目录）
```

## Examples

- `examples/drawio-recon-m1/`：drawio 重建 smoke test 固化样例（重建件 + headless 导出预览 + audit + 复现命令）。正式重建任务照此流程执行；原图在 drawio-reconstruction skill 的 `examples/m1.png`（路径引用，不复制）。

## 关联 skill

- `tikz-diagrams`（**图主路径**：TikZ inline 写在章节 tex 内——全书 36 章已含 176 处 tikzpicture（ch26b/ch26c 全 TikZ 先例 + 上游 5be940f 大迁移）；提供模板/编译渲染/视觉 QA 工具链，产出 .tex+.pdf+.png）
- `drawio-reconstruction`（drawio **存量图维护**：ch20-32 仍有 36 处 drawio png 引用，重建走 inventory→审查闭环）、`drawio-diagram-builder` / `drawio-flow-forge`（存量 drawio 修改/新建）
- `zhihu`（资料五步法）
- `cuda-cpp-kernel`（ptx-docs=F2 核查源）、`cutlass-cpp-kernel`（arch guides=F1 核查源）、`cuda-auto-tune`（bench 复测方法论）
- `ffpa-cuda-understand`（ch19/26/26c 大 D split-D 背景）
