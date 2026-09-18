# LeetCUDA Book RFC（长期执行跟踪表）

> 配套 [`BOOK_PLAN.md`](BOOK_PLAN.md)（规范源）使用。本文件只做**执行跟踪**：里程碑、每章任务、图清单、参考清单。
> 规范（章节结构/模板/DoD/容差/风险）一律以 BOOK_PLAN.md 为准，本文件不重复定义，只登记状态。

## 0. 使用说明

- 每完成一项：`- [ ]` → `- [x]`，并在条目尾追加 `（YYYY-MM-DD）`。
- 每个章节子任务（C1..F7）的完成 = BOOK_PLAN §4.2 单章 DoD 八条全过。
- 后续会话/任务从此文件领取工作项，按依赖顺序执行；开工前先读 BOOK_PLAN.md 对应章节。
- 新发现的问题（注释错误/行号漂移/资料失效）登记到对应位置，不改动 BOOK_PLAN 规范除非明确修订。

## 1. 里程碑总览

| 里程碑 | 内容 | 依赖 | 状态 |
|---|---|---|---|
 |
| RFC-A | 冻结件（记号表/模板/锚点脚本/源码冻结） | RFC-0 | 完成（2026-09-11）|
| RFC-B | 知乎资料全集（专栏枚举+图片归档） | RFC-0 | 取消（2026-09-18：被附录 E 成表 + drawio 全量重建取代，详见 §4 注记）|
| RFC-C | Part I 基础篇 ch1-7 | RFC-A（素材按需 RFC-B） | 完成（2026-09-11，91 页/42 PASS）|
| RFC-D | Part II GEMM 篇 ch8-14 | RFC-C | 完成（2026-09-11/14，218 页/77 PASS）|
| RFC-E | Part III Attention 篇 ch15-19 | RFC-D | 完成（2026-09-14，324 页/139 PASS）|
| RFC-F | Part IV CuTe 篇 ch20-26（原理章先行） | RFC-E | 完成（2026-09-14，324 页/139 PASS）|
| RFC-G | 性能数据汇总（附录 B 成表） | RFC-C..F 各章增量数据 | 完成（2026-09-16，随 K-验收 appB B.1-B.7 成表；G.2 口径见 §9）|
| RFC-H | 图表升级（知乎图 drawio 重建为主 + ASCII→drawio 新建） | 章节完成 | 完成（2026-09-16，69 图全正式/64 gen.py 入库；H.0/H.5 TikZ 路径取消）|
| RFC-I | 附录 A-E | RFC-C..F（D 需收口） | 完成（2026-09-16，随 K-验收收口）|
| RFC-J | 全书集成审校与验收 | 全部 | 进行中（J.9/J.10 完成 2026-09-16；J.1-J.8 随 2026-09-18 十六-agent 数理复审收口）|
| RFC-K | Part V FP8/FP4 Attention 篇 ch27-33（ffpa-attn CuTe sm_120） | RFC-E/F（前置章节） | 完成（K0-K7 + K-验收 2026-09-16：全书 425 页零 error 零 undefined、Overfull 清零）|

依赖图（执行实况）：`RFC-0 → RFC-A → RFC-C → RFC-D → RFC-E → RFC-F → (RFC-G ∥ RFC-H ∥ RFC-I) → RFC-J`；RFC-B 于 2026-09-18 取消（素材职能被附录 E + drawio 全量重建接管）；`RFC-K 依赖 RFC-E/F 素材，与 RFC-G/H/I 并行，完成后并入 RFC-J 收口`

## 2. RFC-0 脚手架

- [x] 0.1 建 `book/` 目录树：chapters/ appendices/ figures/{ascii,zhihu,tikz}/ tests/ notes/ references/ scripts/（2026-09-11）
- [x] 0.2 `preamble.tex`：从 `tex/notes-v2.tex` 抽取 8 色调色板、字体、`\lstset`；`ctexart`→`ctexbook[9pt,openany]`；新增 graphicx/amsmath/amsthm/booktabs/enumitem；代码清单 `firstnumber=auto`（2026-09-11）
- [x] 0.3 `book.tex`：4 Part + 26 章 + 5 附录骨架（空章节），`\tableofcontents` + hyperref 书签（2026-09-11）
- [x] 0.4 关键字表审计：删 0 命中旧名（`sgemm_thread_tile_vec4`/`rope_f32_kernel`/`mat_transpose_f32_row2col2d_kernel`/`block_all_reduce_sum`/`hgemv_k32_f16_kernel`/`hgemm_t_8x8_sliced_k_f16x4_kernel`/`flash_attn_mma_stages_split_q_kernel`/`softmax_f32_per_token_kernel` 系），补 BOOK_PLAN §8.2 清单中的新 kernel/CuTe/FFPA 标识符（2026-09-11）
- [x] 0.5 `build.sh`：两遍 xelatex（`TEXMFCNF=../tex/`）+ 中间文件清理（2026-09-11）
- [x] 0.6 `tests/common_test.h` + `tests/build_tests.sh`（含 `-I ../../third-party/cutlass/include`）+ `tests/README.md`（2026-09-11）
- [x] 0.7 `.gitignore` 追加 BOOK_PLAN §8.3 规则；`git check-ignore -v` 自检通过；`git status` 确认章节 .tex 未被忽略（2026-09-11）
- [x] 0.8 `book/CHECKLOG.md` 空表（列：位置/原文/问题类别 F1-F5/证据/建议/状态）（2026-09-11）
- [x] 0.9 draft `../../../book_plan.md` 顶部加「已被 book/skills/write-leetcuda-book/BOOK_PLAN.md 取代」横幅（2026-09-11）
- [x] 0.10 **验收**：空骨架两遍编译出带正确 TOC/书签的 `book.pdf`；无 `! LaTeX Error`（2026-09-11）

## 3. RFC-A 冻结件

- [x] A.1 全书记号表定稿（BOOK_PLAN §7.4 初稿 → 补全源码变量映射列 → 冻结）（2026-09-11）
- [x] A.2 每章模板固化成 `chapters/_template.tex`（10 节骨架 + DoD 自审 checklist 注释块）（2026-09-11）
- [x] A.3 **源码冻结宣告**：`scripts/anchors.yaml` 登记各 `.cuh`/`notes-v2.cu` 的 SHA256 + 每章引用区间（首/末行锚点文本）；此后源文件零改动（2026-09-11）
- [x] A.4 `scripts/verify_anchors.py`：校验 SHA256、锚点文本、范围内 `#if/#endif` 成对；`--ch chNN` 单章模式（2026-09-11）
- [x] A.5 附录 D 索引 v1：topic ↔ file:line ↔ GitHub commit permalink（记录当前 commit hash）（2026-09-11）
- [x] A.6 **验收**：verify_anchors.py 全绿；模板章编译通过（2026-09-11）

## 4. RFC-B 知乎资料全集

> **2026-09-18 取消说明（`[-]` = 取消，不再执行）**：全书已成稿（426 页），本里程碑的资料职能已被实际产物接管——参考条目由附录 E（`appendices/appE-references.tex`，官方文档/论文/知乎三组 298 行）承载；知乎图片全部由 drawio 正式图替换（69 图全「正式」，正式图不含水印/角标）；已取回全文留存于 `book/references/fulltext/`（7 篇）与 `book/references/zhihu-inventory.md`，原图归档于 `figures/zhihu/`（gitignore 不入库，各含 meta.md）。写作期实际采用「按需检索全文、提炼不照抄」路径，全量枚举/全量存档不再有收益。B.5 转入 2026-09-18 复审（O2 agent 核 ch7 延伸阅读，缺则补 1 条 RoPE 参考）。

- [-] B.1 `fetch_column_fulltext.py` 枚举 @reed、@竹熙佳处 专栏全部文章（frankshi 如有专栏同）；产出 `book/references/zhihu-inventory.md`（§15 种子表并入，标注「已核/待核」） —— 部分落地（inventory 已存在、fulltext/ 7 篇），全集枚举取消，条目职能归附录 E（2026-09-18）
- [-] B.2 增量确认：README 外新文章全部入册（已知：reed《TMA Descriptor 第21bit》p/2037200219700449995、《程序控制和原子操作》p/712357443） —— 两篇均已在 §15 种子表并经 ch1/ch13/ch23 引用，取消（2026-09-18）
- [-] B.3 补充作者关键文章全文取回（@melonedo 除法、@Anonymous Layout 技巧+GEMM 细节三篇、@可怕的杰瑞、@weishengying、@水木皇工仔、@Arthur、@Titus、@进击的Killua、@66RING、@shengying.wei 等），存 `zhihu-analysis/` —— 写作期按需检索已覆盖（竹熙佳处四篇全文在 references/fulltext/），全量存档取消（2026-09-18）
- [-] B.4 图片归档规范落地：`figures/zhihu/<author>-<slug>/` + 元数据 sidecar（来源 URL/引用章节/替换状态=否） —— 归档+meta.md 已实际执行（11 目录），「替换状态」字段因全部替换为 drawio 正式图而废弃，取消（2026-09-18）
- [-] B.4a drawio 重建管线试点（**知乎真实图**，含水印场景）：取 1 张候选图（建议 FIG-21-1 tv-layout-grid，原图先归档）走全流程（水印标记→重建→导出→audit），验收按 BOOK_PLAN §7.2 单图 DoD 六条。工具链已由本 skill `examples/drawio-recon-m1/`（2026-09-11 smoke test：CLI v31.4.5 官方 deb + `/usr/local/bin/drawio-headless` wrapper，CJK 渲染 OK）验证，此项只差水印场景实操 —— 管线已在 64 张正式图上全面验证（含水印图源的 FIG-20-4..20-9），试点取消（2026-09-18）
- [ ] B.5 RoPE 参考补充检索（ch7 当前无主参考） —— 转入 2026-09-18 复审 O2 核查 ch7 延伸阅读，缺则补 1 条
- [-] B.6 附录 E 表结构定稿（作者/标题/URL/对应章节/引用日期/图片数） —— 被 appE 实际成表结构取代（官方文档/论文/知乎三组+主题分组），取消原六列设计（2026-09-18）
- [-] B.7 **验收**：inventory 覆盖 §15 全部条目且每章「主参考」都有 URL；每篇已核文章在 zhihu-analysis/ 有存档 —— 全量存档要求取消；出处标注 100% 由 J.5 承接验收（2026-09-18）
- 注：素材按 Part I→II→III→IV 章序优先交付（章节任务不被卡死）

## 5. RFC-C Part I 基础篇（ch1-7）

> 每项格式：`[ ] 编号 chNN 标题 | 源码区间 | 宏/arch | 测试文件 ← notes-v2.cu 抽取源 | 必收图`。DoD=BOOK_PLAN §4.2。

- [x] C1 ch01 GPU 架构/Roofline | base.cuh L1-86 扩写 | 无宏/全 arch | ch01_roofline.cu（AI 计算演示，host 为主） | FIG-1-1 内存层级、FIG-1-2 roofline（2026-09-11）
- [x] C2 ch02 Warp/Block Reduce 与 Dot | base.cuh L87-305 | 无宏 | ch02_reduce.cu ← test_block_reduce@L510 + test_dot@L545 | FIG-2-1 shuffle 蝶形（2026-09-11）
- [x] C3 ch03 向量化与原子操作 | base.cuh L306-386 | 无宏 | ch03_elementwise.cu ← test_relu@L596/test_elementwise@L641/test_histogram@L689 | FIG-3-1 coalescing 对比（2026-09-11）
- [x] C4 ch04 Softmax 三级递进★ | base.cuh L520-665 | 无宏 | ch04_softmax.cu ← test_softmax@L847 | FIG-4-1 online 数据流（2026-09-11）
- [x] C5 ch05 LSE 与 merge_attn_states | base.cuh L387-519 | 无宏 | ch05_merge_attn.cu ← test_merge_attn_states@L723 | FIG-5-1 分块合并（2026-09-11）
- [x] C6 ch06 RMSNorm/LayerNorm | base.cuh L666-801 | 无宏 | ch06_norm.cu ← test_rms_norm@L920/test_layer_norm@L978 | —（2026-09-11）
- [x] C7 ch07 RoPE 与转置·Bank Conflict | base.cuh L802-909 | 无宏 | ch07_rope_transpose.cu ← test_rope@L1042/test_mat_transpose@L1098/@1144 | FIG-7-1 bank conflict vs padding（2026-09-11）
- [x] C-验收 Part I 全部章节编译进 book.pdf；7 个测试全 PASS；每章 DoD 勾选留档（2026-09-11）

## 6. RFC-D Part II GEMM 篇（ch8-14）

- [x] D1 ch08 SGEMV 三种划分 | sgemv.cuh L1-102 | 无宏 | ch08_sgemv.cu ← test_sgemv@L1190 | FIG-8-1 warp-per-row（2026-09-11）
- [x] D2 ch09 SGEMM 阶梯一 | sgemm.cuh L6-183 | 无宏 | ch09_sgemm.cu ← test_sgemm@L1289（CPU fp64 参考+F32Acc 档） | FIG-9-1 四级 tiling（2026-09-11）
- [x] D3 ch10 SGEMM 阶梯二 TF32 WMMA | sgemm.cuh L184-434 | 无宏 | ch10_sgemm_tf32.cu ← test_sgemm@L1289（TF32 档） | FIG-10-1 双缓冲时序（2026-09-11）
- [x] D4 ch11 HGEMM mma.sync | hgemm.cuh L3-397 | 无宏（SM80+ mma） | ch11_hgemm_mma.cu ← test_hgemm_mma@L1416（F16Acc 档） | FIG-11-1 ldmatrix、FIG-11-2 fragment 布局（2026-09-11）
- [x] D5 ch12 HGEMM Swizzle 三件套 | hgemm.cuh L399-716 + common.cuh L110-348 | 可选 `NOTES_V2_ENABLE_SWIZZLE_V2` | ch12_hgemm_swizzle.cu ← test_hgemm_swizzle@L1491 + test_swizzle_equiv@L4657 | **FIG-12-1 smem swizzle 前后排布（用户点名）**、**FIG-12-2 block swizzle layout（用户点名）**、FIG-12-3 XOR 位运算（2026-09-11）
- [x] D6 ch13 Hopper：TMA+mbarrier+WGMMA | hgemm.cuh L1428-1857 + common.cuh L350-773 | `NOTES_V2_ENABLE_WGMMA`/仅 sm_90a | ch13_hgemm_wgmma.cu ← test_hgemm_wgmma@L1648（本机编译级验证+SKIP 标注） | FIG-13-1 descriptor 位域、FIG-13-2 warpgroup 数据流、FIG-13-3 mbarrier 状态机（2026-09-11）
- [x] D7 ch14 SM120 TMA+mma.sync+WS | hgemm.cuh L1859-2100 + common.cuh setmaxnreg | `NOTES_V2_ENABLE_TMA_MMA_WS`/sm_90a+sm_120a | ch14_hgemm_tma_ws.cu ← test_hgemm_tma_mma_ws@L1810 | FIG-14-1 TMA box、FIG-14-2 producer/consumer 时序（2026-09-11）
- [x] D-验收 Part II 编译+测试（ch13 sm_120a 自动 SKIP 路径验证）+每章 DoD 留档（2026-09-14 随 Part III 验收：139 PASS/0 FAIL）

## 7. RFC-E Part III Attention 篇（ch15-19）

- [x] E1 ch15 Attention 数学与 FA 原理（新写） | flash_attn.cuh L5-110 头注释+FA2/FA3 论文 | 无宏 | 无独立 kernel 测试（公式推导章；引用 ch04/05 测试） | FIG-15-1 attention 分块流水（2026-09-11）
- [x] E2 ch16 FA2 Split-Q+MMA | flash_attn.cuh L5-790 | 无宏 | ch16_fa2_mma.cu ← test_flash_attn@L1895 | FIG-16-1 split-Q warp 布局（2026-09-11）
- [x] E3 ch17 FA2 TMA+WS | flash_attn.cuh L792-1440（宏块 L791-2194 内） | `NOTES_V2_ENABLE_TMA_MMA_WS` | ch17_fa2_tma_ws.cu ← test_flash_attn_tma_mma_ws_impl@L2041 | FIG-17-1 双流水时序（2026-09-11）
- [x] E4 ch18 FA3 双 Consumer | flash_attn.cuh L1441-2192（同宏块） | 同上+setmaxnreg 宏说明 | ch18_fa3.cu ← test_flash_attn_3_tma_ws_impl@L2318 | FIG-18-1 双 consumer 角色（2026-09-14）
- [x] E5 ch19 FFPA Split-D | ffpa_attn.cuh L1-641 | `NOTES_V2_ENABLE_CUTE`(+TMA_MMA_WS) | ch19_ffpa_split_d.cu ← bench_ffpa.cu 内 test 逻辑抽取（8 PASS） | FIG-19-1 D-chunk 切分、FIG-19-2 两阶段合并（2026-09-14）
- [x] E-验收 Part III 编译+测试+每章 DoD 留档；ch15 公式全部独立推导核验（F3）（2026-09-14：全书 324 页零错误，测试 139 PASS）

## 8. RFC-F Part IV CuTe 篇（ch20-26，原理章 F1-F4 先行）

- [x] F0 ch00 性能分析先于优化：nsys 与 ncu | 新写（工具方法论章，通用化实战素材；无 kernel 测试） | 无 | 无 | 表 tab:00-stall（stall 字典）+ tab:00-tree（决策速查） | 2026-09-14 用户点名新增，基础篇开篇
- [x] F1 ch20 CuTe Layout 基础与代数 | 新写+对照 hgemm.cuh L788-820 | `NOTES_V2_ENABLE_CUTE` | ch20_layout.cu（host 端 CuTe 布局断言：compose/inverse/product/divide 坐标一致性） | FIG-20-1 坐标映射、FIG-20-2 mode 分组
- [x] F2 ch21 CuTe Tensor 与 TiledCopy | 新写+对照 g2s/s2r copy | 同上 | ch21_tiled_copy.cu（host 端 TV-layout 断言，6 case） | FIG-21-1 thread×value 网格
- [x] F3 ch22 CuTe TiledMMA 与 fragment | 新写+对照 FFPAAttnSplitDCuTeTraits L31-79 | 同上 | ch22_tiled_mma.cu（host 端 partition 一致性断言，6 case） | FIG-22-1 TiledMMA partition
- [x] F4 ch23 CuTe Swizzle 与 TMA | 新写+对照 SW128 atom 用法 | 同上 | ch23_swizzle_tma.cu（swizzle 位运算 host 断言；TMA 部分编译级，7 case） | FIG-23-1 SW128 atom 排布
- [x] F5 ch24 CuTe HGEMM | hgemm.cuh L718-1427（宏块 L779-1427 内） | 同上 | ch24_hgemm_cute.cu ← test_hgemm_cute@L1568（F16/F32 双组 PASS） | FIG-24-1 手写 vs CuTe 对照表（核心资产）、FIG-24-2 流水时序
- [x] F6 ch25 CuTe FA 三实现对照 | flash_attn.cuh L2196-3488（5 对宏块） | `NOTES_V2_ENABLE_CUTE`+`TMA_MMA_WS` | ch25_fa_cute.cu（3 PASS） | FIG-25-1 三实现结构对照
- [x] F7 ch26 CuTe FFPA Split-D | ffpa_attn.cuh（重点 L31-79+L81/L380 两 kernel） | 同上 | ch26_ffpa_cute.cu ← bench_ffpa.cu（CuTe 版 test，10 项 PASS） | FIG-26-1 双 TiledMma 数据流
- [x] F-验收 Part IV 编译+测试+每章 DoD 留档；F1-F4 完成前不开 F5-F7（2026-09-14：全书 324 页零错误，139 PASS；RFC-K 用户评审驱动的 CuTe 风格重构+勘误融入+drawio 图升级进行中）

## 9. RFC-G 性能数据汇总（降级为汇总任务；数据来自各章 DoD 第 5 条）

- [x] G.1 汇总 `.tmp/book-bench/ch*/` 数据 → 统一口径表（GPU/驱动/库版本/日期/形状/TFLOPS/对照基线）（2026-09-16：appB B.1-B.6 成表）
- [-] G.2 复跑 README 主表形状（`--bench --mnk 4096,4096,4096 --bhnd 1,32,16384,128` 与 D=320 split-D）核对与 README 口径差异 —— 取消单跑（2026-09-18）：appB 采用「README/5090 引用口径 + PRO 5000 本机实测口径并列标注」，每行注明硬件/来源，无需在本机复刻 5090 主表
- [x] G.3 附录 B 成表：README 数据 + 复测数据并列，标注差异原因（F4 类核查）（2026-09-16：含 B.7 Part V 六行明细与四条主线总结）
- [x] G.4 **验收**：附录 B 每行有出处；差异 >5% 的行有解释（2026-09-16，2026-09-18 R7 复审三方交叉）

## 10. RFC-H 图表升级（v4，2026-09-14：生成器单路径，主 agent 直做）

> 管线规范=BOOK_PLAN §7.2 v4：tex 内规格为内容源 → python 生成器（figures/drawio/<id>/gen.py 入库）
> → drawio-headless -s 3 → view_image + PIL 裁剪复核（≥2 轮）→ audit.md → tex 接线 → 图清单回写。
> **PDF 缩放硬验收**：两位数格 ≥54px、字号 ≥12、CJK 禁 bold——铁律全集 /memories/repo/leetcuda-book-drawio.md。
> 知乎图 reconstruction 路径已恢复（2026-09-14 打通浏览器登录态抓图：fetch→base64→`figures/zhihu/<author>-<slug>/` 本地归档 + meta.md，gitignore 不入库）→ vision inventory → gen.py 重建 → 体检（COVER/COLLIDE=0）→ view 直读验收。首批 FIG-20-4..20-7（竹熙佳处 Layout Compose & Inverse）。
> **2026-09-18 收口状态**：图清单 69 条全部「正式」（drawio 64 张 gen.py 全入库 + FIG-24-1 LaTeX 表 + FIG-33-1/2 引用图 + FIG-33-3/4 本机 plot）；49 份 audit.md（ch1-26 批次），Part V 15 张（fig-27-1..32-2）复核结论原记于 RFC K1-K6 条目，audit.md 补录列入本轮复审收口。知乎原图归档 11 目录（figures/zhihu/，gitignore 不入库，各含 meta.md）。

- [-] H.0 TikZ/pgfplots 模板（仅函数曲线/数据图：roofline、吞吐曲线；配色/字体与全书一致） —— 取消（2026-09-18）：roofline 由 drawio 折线 fig-1-2 承担，吞吐/性能曲线由 bench 脚本 matplotlib plot（fig-33-3/4）承担，TikZ 路径无收益
- [x] H.1 必收图批次（用户点名，优先）：FIG-12-1 smem swizzle、FIG-12-2 block swizzle——生成器新建（规格=ch12 ASCII 图 + caption，可参 common.cuh L185-235 布局表）（2026-09-14）
- [x] H.2 基础章批次：FIG-1-1/2-1/3-1/4-1/5-1/7-1/8-1（生成器新建）（2026-09-14）
- [x] H.3 GEMM 章批次：FIG-9-1/10-1/11-1/11-2a/12-3/13-1/13-2/13-3/14-1/14-2（2026-09-14）
- [x] H.4 FA/CuTe 章批次：FIG-15-1/15-2/16-1/17-1/17-2/18-1/19-1/19-2/20-2/20-3/21-1/21-2/21-3/24-2/25-1（2026-09-14 全部完成，35/35 正式）
- [-] H.5 函数/数据图批次（TikZ）：FIG-1-2 roofline 等 —— 取消（2026-09-18）：同 H.0，由 drawio/matplotlib 替代
- [x] H.6 生成器从 .tmp/drawio/ 迁移到 figures/drawio/<id>/gen.py（入库）；每张完成后图清单状态回写「正式」（2026-09-14/16：64 目录 gen.py 全部入库，.tmp 权威源按铁律回流）
- [x] H.7 **验收**：图清单全部条目状态=「正式」（69/69）；audit.md 含生成器参数与复核结论（49 份已入库，Part V 15 份补录中，2026-09-18）；PDF 抽查缩放可读性（K-验收 2026-09-16 通过）

## 11. RFC-I 附录 A-E

- [x] I.1 附录 A common.cuh 工具箱（773 行分块解析）（2026-09-16：appA 181 行，8 段工具族地图+行号区间表）
- [x] I.2 附录 B ← RFC-G 产出合入（2026-09-16：appB 218 行，B.1-B.7 含 Part V 明细）
- [x] I.3 附录 C 构建/CLI/测试指南 + 章×宏×arch 矩阵（BOOK_PLAN §1.4 扩展）+ notes-v2.cu 角色声明 + 环境安装命令（2026-09-16：appC 192 行，含 C.2 ffpa-attn 安装/bench 段）
- [x] I.4 附录 D 源码索引收口（26 章定稿后刷新 commit permalink）（2026-09-16：34 行双仓表 LeetCUDA@1c2c1e0 + ffpa-attn@861d75e；2026-09-18 R8 复审）
- [x] I.5 附录 E ← RFC-B inventory 合入（2026-09-16：appE 298 行，官方文档/论文/知乎三组；RFC-B 取消后为参考条目唯一事实源）
- [x] I.6 **验收**：五个附录编译进书、TOC 正确（2026-09-16 K-验收，2026-09-18 复审 R7/R8）

## 12. RFC-J 全书集成审校与验收

- [ ] J.1 交叉引用/`\ref` 全部解析；TOC/页码/PDF 书签核对
- [ ] J.2 verify_anchors.py 全量跑（26 章+附录 A）
- [ ] J.3 pdftotext 全书抽查：无豆腐块、无横向溢出、每章正文 ≥3500 字统计
- [ ] J.4 引文 `#if/#endif` 成对 grep 断言
- [ ] J.5 引用图出处标注 100% 检查
- [ ] J.6 术语一致性 pass（记号表为准；全书 grep 常见别名）
- [ ] J.7 页数验收（≥280）+ 无 `! LaTeX Error` + Overfull 可控
- [ ] J.8 **验收**：对照 BOOK_PLAN §11 全书级标准逐条勾选
- [x] J.9 行越界专项：全书 Overfull \hbox 114 处（75 处 ≥10pt，最大 112pt）→ ≥1pt 清零（2026-09-16）。手段：preamble `\emergencystretch=2em`（114→52）；`\texttt` 断点注入脚本（camelCase/`\_`/`/`/`.`/`::`/`(`/`<`，门槛=真实断点切分后 run≥12，52→11，另修复 3 处裸 `\allowbreak ` 空格伪影导致的 `ENABLE_ TMA` 渲染变形）；`xurl`（\url 长链接断行）；剩余 11 处 editorial 精修（拆 run-in 粗体、`file~Lxxx` 改空格、表格 \footnotesize、align 续行列对齐修正）。终态仅余 1 处 0.88pt（<1pt 不可见，保留）。工具与清单：`book/.tmp/layout-fix/{map_overfull.py, inject2.py, overfull-map.md}`。页数 324→354
- [x] J.10 全书内容审校（2026-09-16，commit 4caaed6+4517bb0）：(1) 日志排版——ch13/14/17/19/25 的 quote+texttt 伪代码块统一转 `lstlisting[style=console]`，恢复真实日志文本；(2) 本地路径清理——删除 16 处 `.tmp/book-bench|agent-chNN` 等读者不可见路径（保留 ch00 教学示例与模板注释）；(3) 4 份 GLM-5.3 审校报告（`book/.tmp/review/agent1-4.md`，77 条）应用 76 条：数学/事实错误均独立复算或源码/PTX 文档核实（重点：ch12 寄存器占用 R≈220→110、ch17 三处 linerange 错位互换、ch25/26 C fragment 行列 PTX ISA Figure 83 双重验证、ch15 FA3 作者名）；语言类清理生造词；跳过 12-7（低置信度冲突绝对计数）。顺手修复存量 undefined 引用 4 处（ch02 补 3 个 subsection label、ch17 ch:2→ch:02）。重建验证：0 编译错误、Overfull ≥1pt 为 0、undefined 引用为 0。遗留：fig-3-1/fig-12-2 图内数字与修正后正文不一致需重绘；ch16_fa2_mma.cu:5 头注释 grid 描述未改（测试文件）

## 12-K. RFC-K Part V FP8/FP4 Attention 篇（ch27-33）

> 规范源：BOOK_PLAN §3 Part V 卡片 + §4.2 Part V 特化 DoD。代码链路 = ffpa-attn repo（锚定 commit `861d75e`，`csrc/cuffpa/cute/{fp8,fp4}` + `cute/{softmax,hadamard}.cuh`，只讲 sm_120）。验证 = `python -m ffpa_attn.bench` / `bench/bench_{fp8,fp4}.py`（PRO 5000，`CUDA_VISIBLE_DEVICES=7`，ffpa-attn 0.2.5.dev25 editable 已装）。原理参考 = ffpa-cuda-understand skill §11（推导级）+ `references/papers/`（SA1/2/2++/3、FA-2/3/4 论文文本）。性能图资源 = ffpa-attn `docs/assets/perf/{fp8,fp4}/`（5090）+ `docs/assets/{ffpa-split-d,mma}.png`。

- [x] K0 Part V 章节规划落盘：BOOK_PLAN §3 Part V 表（7 章卡片）+ 附录 B/C/D/E 增补 + §4.2 Part V 特化 DoD + 本 RFC-K 卡片（2026-09-16）
- [x] K0.1 `book.tex` 加 Part V `\part` + ch27-33 空章骨架（含 `chapters/_template.tex` 复制）；两遍编译通过（2026-09-16：365 页零 error；Overfull 3 处为 ch14/ch21/ch24 存量，登记 K-验收清偿）
- [x] K0.2 性能图资源归档：`docs/assets/perf/{fp8,fp4}/*.png` + `ffpa-split-d.png`/`mma.png` 复制到 `book/figures/ffpa/` + 出处 sidecar（ffpa-attn README / 5090 / URL / 引用章节）；附录 E 预登记（2026-09-16：15 图 + meta.md + SA1/2/2++/3、FA-4 论文条目 + ffpa-attn 仓库条目；bench smoke 口径确立：`--show-allclose` ✅ + timing 表，fp8 D128 self-attn 1.70x/365T 落 .tmp/book-bench/k0-smoke/）
- [x] K1 ch27 量化注意力的数学基础 | 新写原理章（fp8_pscale.cuh 头注释 + softmax.cuh 对照） | 前置 4,5,15 | FIG-27-1 量化格式位域、FIG-27-2 粒度谱系表、FIG-27-3 per-stage 误差分解 | 参考 SA1/SA2/SA2++/SA3/FA-4 论文 + @DefTruth WINT8/4 | 验证：`ffpa_attn.bench --cuda-impl fp8 --D 128`（quant 基线口径展示）（2026-09-16：正文 10 节 + 相对步长定理证明 + 3 表 + 3 图（gen.py 入库，COVER=0/渲染越界 0/3x 导出指纹验收）；锚点 L17-37/L111-138/L44-63 grep 核对；bench 基线 .tmp/book-bench/k0-smoke/ 1.70x allclose✅；全书 375 页零 error、ch27 Overfull 清零）
- [x] K2 ch28 FP8(一)量化前处理链 | `cute/fp8/{smooth_k(165),smooth_v(175),quantize_fp8(1024)}.cuh` + `cute/hadamard.cuh(202)` | 前置 20-23,27 | FIG-28-1 前处理 pipeline、FIG-28-2 smoothing 三不变性 | 验证：`--fp8-smooth-k/--fp8-smooth-v/--fp8-q-quant-method per-thread` knob A/B bench（2026-09-16：正文 10 节（三不变性推导+WHT+fragment 粒度+VTPerm+NHD 零拷贝）+ 2 图（COVER=0/text_fit 0/3x 导出指纹验收）；锚点 smooth_k L12-52/L56-84、quantize_fp8 L29-79/L172-201/L695-741、hadamard L1-32/L40-92 grep 核对，注释核查无错误（85→50us/2.6x RMSE/64-scale 排布均与实现/论文一致）；bench knob A/B 五组 + parity 三组全 allclose✅（smk 1.70x/smv 1.72x/per-thread 1.65x）.tmp/book-bench/ch28/；全书 385 页零 error、ch28 Overfull 清零（余 2.4pt 微量））
- [x] K3 ch29 FP8(二)persist-D 主 kernel ★ | `cute/fp8/sm_120/persist_d.cuh(1036)` + `fp8_pscale.cuh(304)` + `softmax.cuh(344)` + `reg2reg_8b.cuh(163)` + `attn_traits.cuh(313)` | 前置 17,18,22,23,28 | FIG-29-1 scale 折叠数据流、FIG-29-2 persist-D WS 布局、FIG-29-3 reorg-free 打包对照 | 验证：`--cuda-impl fp8 --tasks self causal --D {64,128,192,256}` parity+timing（2026-09-16：正文 10 节（δ_Qδ_K 折 exp2+三步协议+lazy rescale 溢出界+f16 PV 域推导+置换不变性+traits/WS 复用/五 Phase/reorg-free 布局推导）+ 3 图（COVER=0/text_fit 0/3x 导出指纹验收）；锚点 persist_d L37-66/L193-215/L406-409/L502-522/L583-601/L723-732/L833-903/L886-894/L904-1031/L1020-1031、reg2reg L14-34/L96-141 grep 核对，code review 6 处行号偏差修正后无实质错误；bench D=64 1.26x/D=128 1.70x/causal 1.45x parity allclose✅，**D=192 发现正确性 bug（kBc=64 变体，坏行 row%128∈[64,127]），已记 memory + 书稿 29.7.1 定位方法论五步法**，.tmp/book-bench/ch29/；全书 395 页零 error）
- [x] K4 ch30 FP8(三)split-D 与 M4N2 | `cute/fp8/sm_120/{split_d,split_d_m4n2}.cuh` | 前置 19,26,29 | FIG-30-1 寄存器压力曲线（README mma.png）、FIG-30-2 split-D 切分（ffpa-split-d.png）、FIG-30-3 M4N2 fragment 划分 | 验证：`--cuda-impl fp8 --D {320,512,768,1024}`（cross-point 展示）（2026-09-16：正文 10 节（两堵墙动机+dispatch 三分+D-chunk 数学+O_acc/寄存器模型 M8N1 vs M4N2+三 Phase 结构+M4N2 三笔新代价（P SMEM roundtrip/跨 N-warp softmax 单 barrier/两级 reg 模型）+causal 精度定性+PC-7/PC-8 旋钮史）+ 2 图（FIG-30-1 split-D 两堵墙与三分路由、FIG-30-2 M4N2 布局与 softmax 协议；COVER=0/text_fit 0/3x 导出指纹验收，原计划 3 图并为 2 图）；锚点 split_d L39-42/L105-108/L124-125/L229-245/L267-274/L446-483/L495-503/L609-618/L678-756、split_d_m4n2 L120-121/L300-307/L638-641/L685-707/L970、softmax L74-163、traits L152-157/L279-284 grep 核对，code review（4 实质错误+9 行号+6 表述）全部修正；bench D={256,320,512} self 1.30x/3.76x/2.98x causal 1.12x/3.31x/2.57x，dense 全绿、causal D=320/512 临界失败按已知 ESS 特性定性（头注释 L39-42 佐证+hybrid 解法方向），本机无 768+ 编译实例引用 5090 交叉点数据，.tmp/book-bench/ch30/；配套 [H] 图文错乱修复（float 包+7 处）+全书 inflight/DoD 用词清理；全书 403 页零 error）
- [x] K5 ch31 FP4(一)NVFP4 格式与量化链 | `cute/fp4/{quantize_fp4(1642),delta_s(451),fp4_gemm(276),attn_traits(658)}.cuh` | 前置 27,28 | FIG-31-1 NVFP4 1×16 block+SF、FIG-31-2 delta_s rank-1 修正、FIG-31-3 kv_perm32 | 验证：`--cuda-impl fp4 --D {64,128}`（fused 量化链）（2026-09-16：正文 10 节（e2m1 动态范围墙+blockscale MMA 语义+ΔS rank-1 恒等式+smooth-V 恒等式+pad 零贡献+WHT 线性性+4 线程/token 映射+SF atom 写序+fused 单 launch D≤128 与蝶形拆分+delta_s wmma/CuTe-TMA 双实现+kv_perm32 与 perm-aware masking+subbyte 陷阱）+ 3 图（COVER=0/text_fit 0/3x 导出验收）；锚点 quantize_fp4 L32-56/L109-320/L158-172/L195-235/L293/L322-496/L342-343/L414/L437-455/L667-701/L1024+/delta_s L2-8/L37-161/L163+/fp4_gemm L23-26/traits L30-37/L51-63 grep 核对，code review（4 实质错误：π 闭式末项 %2、V^T 不置换补偿在 P 打包、1.4→0.3ms 出处、M_b 释义 + 8 行号 + 4 表述）全部修正；bench D={64,128,192,256} self 1.49x/2.03x/2.05x/2.11x causal 1.23x/1.61x/1.59x/1.49x 16 格全绿，knob：hadamard 零开销/bitwise 一致、smooth-v +2.8%、MXFP8 PV +37.7%，.tmp/book-bench/ch31/；全书 411 页零 error）
- [x] K6 ch32 FP4(二)persist-D 主 kernel ★ | `cute/fp4/sm_120/persist_d.cuh(1113)` + `fp4_pscale.cuh(793)` + `fp4_gemm.cuh(276)` | 前置 29,31 | FIG-32-1 两级 P 量化域、FIG-32-2 persist-D 主循环数据流 | 验证：`--cuda-impl fp4 --tasks self causal --D {64,128,192,256}` + `--fp4-pv-mm-type fp8` 变体（2026-09-16：正文 10 节（P 三重困境+blockscale 约束动机 / 两级 P 量化数学：eq 两级方案+online 退化（rowmax≡1 → 全局常数 1/2688 折 exp2）+归一化折进 exp2（rcp/FMUL 段消失）+lazy rescale 与 lse 2688 消去+mxfp8 契约 / WS 骨架与 grid 契约+barrier 清单与永不 re-init / Q 驻留+历史 bug（cicc 折叠未初始化 asm 操作数）/ 主循环五步（gemm_ss zip 对+bias dequant 域注入+perm-aware 谓词+融合 softmax+gemm_rs A 不过 smem）/ epilogue（smooth-V 加回恒等式+STSM 批量 staging）/ mxfp8 表+fp4 vs fp8 口径+已证伪三项 / adapter 整体性+评估口径）+ 2 图（COVER=0/text_fit 0/3x 导出指纹验收）；锚点 persist_d L6-53/L92-120/L200-213/L722/L747-753/L757-758/L811-814/L819-822/L835-902/L903-916/L921-927/L937-940/L968-975/L981-1066、pscale L2-5/L117-128/L160-167/L169-175/L182-186/L314-346/L327-331/L390-397/L417、gemm L91-218、traits L19-28/L62-65 grep 核对，code review（1 硬数学错误：eq 32.2 常数符号（源码注释 L118 同款笔误，正文修正为 +log2(448·6) 并注记）+1 文件引用（fp4_gemm→fp4_pscale pack 注释）+23 处行号漂移）全部修正；bench mxfp8 变体 D={64,128,192} vs fp4 基线 +28~39%（1.07x-1.56x）parity allclose✅，.tmp/book-bench/ch32/；全书 416 页零 error、Overfull 存量 7（非 Part V 5 处+ch30 2 处，K-验收清偿））
- [x] K7 ch33 FP8/FP4 性能实战 | `bench/bench_{fp8,fp4}.py` + bench CLI 全 knob | 前置 29,30,32 | FIG-33-x README 5090 speedup 引用图（fp8/fp4 各 3-4 张）+ 本机 PRO 5000 plot | 验证：完整 task 矩阵（self/causal/gqa/cross × D{64..512}）+ knob 矩阵实验 + 图落盘（2026-09-16：正文 10 节（度量三陷阱 / speedup+有效 TFLOPS+E2E 摊销 eq+精度四指标与 tol 分档依据 / 两套 bench 两层 + 测量纪律代码化 + CLI knob 全集（含 --cuda-impl fp8 默认 Q/K/V 全 per-block 与 bench_fp8.py preset 的区分）/ 4 图 / fp8 矩阵 D 三段形态 + fp4 矩阵 D≥128 稳定 2x + knob 矩阵（数值通路 ±2.3% 中性、粒度升 per-thread/per-channel 1.7-2.7%）+ 竞品与硬件对照（Sage3 causal rel_err 0.59 vs hybrid 0.14、5090 804T vs PRO 5000 523T）/ 选用决策树 + Cache-DiT×FFPA 三要点 / 坑 6 / 面试 7 / 实验 4）+ 2 引用图（5090 fp8/fp4 D128）+ 2 本机 plot（bench_fp8/fp4.py 生成，8K/16K）；锚点 bench_fp8 L1-20/L57/L79-122/L125-140/L153-160/L256-259/L270-274/L276/L344-347、bench_fp4 L8-10、_bench L347-386/L387-490/L648-694、_runner_fwd L413-419 grep 核对，code review（5 硬错误：E2E 摊销 1.15x→1.22x、CLI 默认粒度 per-block（表 33.3 两行 no-op 对照重跑真实 per-thread/per-channel 数据）、SDPA 209T→213T、±1.5%→最差 2.6%、plot N 口径 4K/8K/16K→8K/16K（低分辨率图误读教训）+ 5 锚点 + 9 表述）全部修正；bench fp8 16 格（D=512 causal fp16 ❌ 已知 ESS 特性）+ fp4 16 格全绿 + knob 2 轮（第二轮补真实粒度 A/B 与 ch28 28.8 的 355T 记录精确互证）+ e2e bench_fp8/fp4.py 全场景，.tmp/book-bench/ch33/；顺手修复 ch28 fig:11-frag→fig:11-2a 未定义引用（存量 bug）；全书 424 页零 error 零 undefined、Overfull 存量 7）
- [x] K-验收 Part V 集成：全书两遍编译零 error、Overfull ≥1pt 为 0、无 undefined 引用；附录 B（fp8/fp4 数据段）+ 附录 D（ffpa-attn `861d75e` 索引段）+ 附录 C（ffpa-attn 安装/bench 指南）更新；RFC.md 图清单登记；每章 DoD 勾选留档（2026-09-16 完成收口：全书 425 页、0 error、0 undefined、**Overfull ≥1pt 全量清零**（存量 7 处：ch14 mbarrier 长指令串 allowbreak 重组、ch21 延伸阅读 URL 项语序调换、ch23 Q4 句式精简、ch27 表 27-sage 首列 p{1.0→1.2cm}（SA2++ 列宽不足真因）、ch28 坑 1 措辞精简、ch30 dispatch 句语序 + 临界失败段两段拆分（该段单行不 break 的排版异常以拆段收口））；附录 B 增 B.7 Part V 明细表（ch28-33 六行 + 5090 非本地口径注）+ 主线四（量化通路）；附录 C 增 C.2 ffpa-attn 安装/bench 段（pip editable/build.sh sm_120f/小 D 环境变量/CLI+竞品两层）；附录 D 增 第27-33 七行 ffpa-attn permalink（861d75ee87…，路径 fp8/fp4 子目录核对修正 3 处）；pdftotext 抽查附录与 Part V 渲染）

执行序：K0 → K1 → K2 → K3 → (K4 ∥ K5) → K6 → K7 → K-验收（K4/K5 无文件冲突可并行；每章完成即独立 commit）

## 13. 图清单登记表（写作期持续更新）

| FIG | 章 | slug | 类型(A/B/D/C1/C2, 见 BOOK_PLAN §7.1) | 状态(占位/引用/重建完成/正式) | 出处或源文件 |
|---|---|---|---|---|---|
| FIG-1-1 | 1 | mem-hierarchy | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-1-1-mem-hierarchy/ |
| FIG-1-2 | 1 | roofline | C1 新建（drawio 折线，2026-09-14） | 正式 | figures/drawio/fig-1-2-roofline/ |
| FIG-2-1 | 2 | shuffle-butterfly | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-2-1-shuffle-butterfly/ |
| FIG-3-1 | 3 | coalescing | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-3-1-coalescing/ |
| FIG-4-1 | 4 | online-softmax-flow | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-4-1-softmax-flow/ |
| FIG-5-1 | 5 | merge-states | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-5-1-merge-states/ |
| FIG-7-1 | 7 | bank-conflict-padding | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-7-1-bank-pad/ |
| FIG-8-1 | 8 | warp-per-row | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-8-1-warp-per-row/ |
| FIG-9-1 | 9 | gemm-tiling-4level | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-9-1-gemm-tiling/ |
| FIG-10-1 | 10 | double-buffer | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-10-1-double-buffer/ |
| FIG-11-1 | 11 | ldmatrix | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-11-1-ldmatrix/ |
| FIG-11-2 | 11 | mma-fragment | C1 新建（fig-11-2a，2026-09-14） | 正式 | figures/drawio/fig-11-2a-fragment/ |
| FIG-12-1 | 12 | smem-swizzle-before-after | C1 新建（用户点名，2026-09-14） | 正式 | figures/drawio/fig-12-1-smem-swizzle/ |
| FIG-12-2 | 12 | block-swizzle-layout | C1 新建（用户点名，2026-09-14） | 正式 | figures/drawio/fig-12-2-block-swizzle/ |
| FIG-12-3 | 12 | xor-bitwise | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-12-3-xor-bits/ |
| FIG-13-1 | 13 | wgmma-desc-bitfield | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-13-1-wgmma-desc/ |
| FIG-13-2 | 13 | warpgroup-dataflow | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-13-2-ws-dataflow/ |
| FIG-13-3 | 13 | mbarrier-statemachine | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-13-3-mbarrier/ |
| FIG-14-1 | 14 | tma-box | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-14-1-tma-box/ |
| FIG-14-2 | 14 | producer-consumer-timeline | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-14-2-pc-timeline/ |
| FIG-15-1 | 15 | attention-block-pipeline | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-15-1-online-softmax/ |
| FIG-15-2 | 15 | fa1-fa2-fa3-timeline | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-15-2-fa1-fa2-fa3/ |
| FIG-16-1 | 16 | split-q-warp-layout | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-16-1-split-q-warp/ |
| FIG-17-1 | 17 | fa2-dual-pipeline | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-17-1-dual-pipeline/ |
| FIG-17-2 | 17 | mbarrier-topology | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-17-2-mbarrier-topo/ |
| FIG-18-1 | 18 | fa3-dual-consumer | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-18-1-fa3-dual-consumer/ |
| FIG-19-1 | 19 | split-d-chunk | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-19-1-split-d-chunk/ |
| FIG-19-2 | 19 | split-d-merge | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-19-2-two-stage-merge/ |
| FIG-20-1 | 20 | layout-coord-mapping | B→D（reed/竹熙佳处） | 正式 | figures/drawio/fig-20-1-colex/
| FIG-20-2 | 20 | mode-grouping | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-20-2-mode-ops/ |
| FIG-20-3 | 20 | layout-convert-routes | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-20-3-layout-convert/ |
| FIG-20-4 | 20 | tensor-layout-engine-addr | B→D（竹熙佳处 compose&inverse，2026-09-14） | 正式 | figures/drawio/fig-20-4-tensor-addr/ |
| FIG-20-5 | 20 | compose-3step-flow | B→D（竹熙佳处，2026-09-14） | 正式 | figures/drawio/fig-20-5-compose-flow/ |
| FIG-20-6 | 20 | inverse-with-shape | B→D（竹熙佳处，2026-09-14） | 正式 | figures/drawio/fig-20-6-inverse-reshape/ |
| FIG-20-7 | 20 | left-right-inverse | B→D（竹熙佳处，2026-09-14） | 正式 | figures/drawio/fig-20-7-left-right-inverse/ |
| FIG-20-8 | 20 | layout-product | B→D（竹熙佳处 product&divide，2026-09-14） | 正式 | figures/drawio/fig-20-8-layout-product/ |
| FIG-20-9 | 20 | layout-divide | B→D（竹熙佳处，2026-09-14） | 正式 | figures/drawio/fig-20-9-layout-divide/ |
| FIG-21-1 | 21 | tv-layout-grid | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-21-1-engine-layout/ |
| FIG-21-2 | 21 | tv-hand-table | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-21-2-tv-table/ |
| FIG-21-3 | 21 | g2s-thread-grid | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-21-3-g2s-grid/ |
| FIG-21-4 | 21 | tiled-copy-flow | B→D（竹熙佳处 tiled copy，2026-09-14） | 正式 | figures/drawio/fig-21-4-tiled-copy-flow/ |
| FIG-22-1 | 22 | tiledmma-partition | B→D（竹熙佳处 tiled mma） | 正式 | figures/drawio/fig-22-1-tiledmma/ |
| FIG-22-2 | 22 | mma-atom-fragments | B→D（竹熙佳处 tiled mma，2026-09-14） | 正式 | figures/drawio/fig-22-2-mma-atom-fragments/ |
| FIG-23-1 | 23 | sw128-atom | B→D（竹熙佳处/reed） | 正式 | figures/drawio/fig-23-1-sw128/ |
| FIG-23-2 | 23 | tma-copy-flow | B→D（竹熙佳处 TMA Copy，2026-09-14） | 正式 | figures/drawio/fig-23-2-tma-copy-flow/ |
| FIG-24-1 | 24 | handwritten-vs-cute-table | C2（LaTeX 表格） | 正式 | （保持 tabular 形态） |
| FIG-24-2 | 24 | kstage2-pipeline | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-24-2-kstage-pipeline/ |
| FIG-24-3 | 24 | gemm-3level-partition | B→D（reed 简单 GEMM，2026-09-14） | 正式 | figures/drawio/fig-24-3-gemm-3level-partition/ |
| FIG-25-1 | 25 | fa-cute-3impl-compare | C1 新建（2026-09-14） | 正式 | figures/drawio/fig-25-1-cute-struct/ |
| FIG-26-1 | 26 | ffpa-dual-tiledmma | A→C1 | 正式 | figures/drawio/fig-26-1-dual-mma/
| FIG-27-1 | 27 | fp-bitfield | C1 新建（2026-09-16） | 正式 | figures/drawio/fig-27-1-fp-bitfield/ |
| FIG-27-2 | 27 | quant-points | C1 新建（2026-09-16） | 正式 | figures/drawio/fig-27-2-quant-points/ |
| FIG-27-3 | 27 | ess-decompose | C1 新建（2026-09-16） | 正式 | figures/drawio/fig-27-3-ess-decompose/ |
| FIG-28-1 | 28 | quant-pipeline | C1 新建（2026-09-16） | 正式 | figures/drawio/fig-28-1-quant-pipeline/ |
| FIG-28-2 | 28 | invariants | C1 新建（2026-09-16） | 正式 | figures/drawio/fig-28-2-invariants/ |
| FIG-29-1 | 29 | scale-folding | C1 新建（2026-09-16） | 正式 | figures/drawio/fig-29-1-scale-folding/ |
| FIG-29-2 | 29 | persist-d-ws | C1 新建（2026-09-16） | 正式 | figures/drawio/fig-29-2-persist-d-ws/ |
| FIG-29-3 | 29 | reorg-free | C1 新建（2026-09-16） | 正式 | figures/drawio/fig-29-3-reorg-free/ |
| FIG-30-1 | 30 | split-d-walls | C1 新建（2026-09-16，原计划 mma 曲线+split-D 切分两图并为一张：两堵墙+dispatch 三分路由） | 正式 | figures/drawio/fig-30-1-split-d-walls/ |
| FIG-30-2 | 30 | m4n2-softmax | C1 新建（2026-09-16，M4N2 (4,2,1) 布局与跨 N-warp softmax 单 barrier 协议） | 正式 | figures/drawio/fig-30-2-m4n2-softmax/ |
| FIG-31-1 | 31 | nvfp4-block | C1 新建（2026-09-16，NVFP4 1×16 块 + e2m1/ue4m3 位域 + blockscale MMA 语义） | 正式 | figures/drawio/fig-31-1-nvfp4-block/ |
| FIG-31-2 | 31 | fp4-pipeline | C1 新建（2026-09-16，前处理链数据流：均值先行→量化→ΔS→主 kernel，D≤128 fused 路径） | 正式 | figures/drawio/fig-31-2-fp4-pipeline/ |
| FIG-31-3 | 31 | kv-perm32 | C1 新建（2026-09-16，32 列窗口双射置换表 + perm-aware masking 正误对照） | 正式 | figures/drawio/fig-31-3-kv-perm32/ |
| FIG-32-1 | 32 | two-level-p | C1 新建（2026-09-16，两级 P 量化域拉伸：online softmax 退化第一级为全局常数 1/2688 折 exp2 shift，第二级 per-16 组 ue4m3） | 正式 | figures/drawio/fig-32-1-two-level-p/ |
| FIG-32-2 | 32 | persist-d-loop | C1 新建（2026-09-16，persist-D 主循环数据流：gemm_ss→bias/masking→融合 softmax→gemm_rs→lazy rescale 五步与 barrier 交叠） | 正式 | figures/drawio/fig-32-2-persist-d-loop/ |
| FIG-33-1 | 33 | 5090-fp8-speedup | C2 引用（2026-09-16，ffpa-attn README 5090 fp8 D=128 speedup：FFPA-FP8 441T vs Sage2 394T vs SDPA 291T） | 正式（引用） | figures/ffpa/fp8/ |
| FIG-33-2 | 33 | 5090-fp4-speedup | C2 引用（2026-09-16，ffpa-attn README 5090 fp4 D=128 speedup：FFPA-FP4 804T/2.60x） | 正式（引用） | figures/ffpa/fp4/ |
| FIG-33-3 | 33 | pro5000-fp8-tflops | C3 本机 plot（2026-09-16，bench_fp8.py 生成：PRO 5000 D=128 fp16 七场景 × 8K/16K，FFPA/Sage/SDPA 三系列） | 正式 | figures/ffpa/fp8/fig-33-3-pro5000-fp8-tflops.png（生成脚本 = ffpa-attn bench/bench_fp8.py） |
| FIG-33-4 | 33 | pro5000-fp4-tflops | C3 本机 plot（2026-09-16，bench_fp4.py 生成：PRO 5000 同形状，FFPA-FP4/FP8/Sage3/SDPA 四系列） | 正式 | figures/ffpa/fp4/fig-33-4-pro5000-fp4-tflops.png（生成脚本 = ffpa-attn bench/bench_fp4.py） |

## 14. CHECKLOG 摘要镜像（明细在 book/CHECKLOG.md）

| 日期 | 位置 | 类别 | 摘要 | 状态 |
|---|---|---|---|---|
| 2026-09-11 | ffpa_attn.cuh 头注释 L17-18 | F4 | 性能口径 PRO 5000 与 README（5090）不一致，成书需统一口径标注 | 待 ch19 核查 |

## 15. 知乎参考种子清单（RFC-B 并入 zhihu-inventory.md）

> URL 前缀省略 `https://zhuanlan.zhihu.com/p/`。「章」= 主参考章节。

| 作者 | 文章 | id | 章 |
|---|---|---|---|
| 竹熙佳处 | 写给大家看的 CuTe 教程：tiled copy | 1930389542784964333 | 21 |
| 竹熙佳处 | 写给大家看的 CuTe 教程：tiled mma | 1937145378446226159 | 22 |
| 竹熙佳处 | CuTe 教程：Layout Compose & Inverse | 1962625273636845008 | 20 |
| 竹熙佳处 | CuTe 教程：Layout Product & Divide | 1971945267294111573 | 20 |
| 竹熙佳处 | CuTe 教程：TMA Copy | 2003198909405763007 | 13,14,23 |
| 竹熙佳处 | CuTe 笔记：permutationMNK 参数 | 1973526710105419953 | 24 |
| reed | cute 之 Layout | 661182311 | 20 |
| reed | cute Layout 的代数和几何解释 | 662089556 | 20 |
| reed | cute 之 Tensor | 663093816 | 21 |
| reed | cute 之 MMA 抽象 | 663092747 | 22 |
| reed | cute 之 Copy 抽象 | 666232173 | 21 |
| reed | cute 之 Swizzle | 671419093 | 12,23 |
| reed | cute 之 TMA Descriptor 编码与隐藏的第 21bit | 2037200219700449995 | 13,23 |
| reed | cute 之 简单 GEMM 实现 | 667521327 | 24 |
| reed | cute 之 GEMM 流水线 | 665082713 | 24 |
| reed | cute 之 高效 GEMM 实现 | 675308830 | 24 |
| reed | GPU 指令集架构-前言 | 686198447 | 1 |
| reed | GPU 指令集架构-寄存器 | 688616037 | 1 |
| reed | GPU 指令集架构-Load 和 Cache | 692445145 | 1,11 |
| reed | GPU 指令集架构-浮点运算 | 695667044 | 1 |
| reed | GPU 指令集架构-整数运算 | 700921948 | 1 |
| reed | GPU 指令集架构-比特和逻辑操作 | 712356884 | 1 |
| reed | GPU 指令集架构-Warp 级和 Uniform 操作 | 712357647 | 1 |
| reed | GPU 指令集架构-程序控制和原子操作 | 712357443 | 1,3 |
| frankshi | CUDA shared memory 避免 bank conflict 的 swizzling 机制解析 | 4746910252 | **12（主）**,7,23 |
| melonedo | Cute 布局代数实战：除法 | 1970274785691936058 | 20 |
| melonedo | 布局代数实战：Swizzle 自动推导 | 1941306442683515068 | 12,23 |
| Anonymous | 介绍一个关于 CuTe Layout 变换的小技巧 | 2006000375463961170 | 20,24 |
| Anonymous | GEMM 细节分析(一)：ldmatrix 的选择 | 702818267 | 11 |
| Anonymous | GEMM 细节分析(二)：TiledCopy 与 cp.async | 703560147 | 21 |
| Anonymous | GEMM 细节分析(三)：Swizzle<B,M,S> 参数取值 | 713713957 | 12,23 |
| 可怕的杰瑞 | Cute TiledMMA 简单理解 | 1991908850132088026 | 22 |
| weishengying | cute swizzle | 706796240 | 23 |
| 水木皇工仔 | 基于 CuTe 理解 swizzle, LDSM, MMA | 934430036 | 21,22 |
| Arthur | Bank Conflict 自动消除：Swizzle 技术原理解析 | 2042049499263136652 | 12 |
| Titus | cutlass swizzle 机制解析（一）（二） | 710337546 / 711398930 | 12,23 |
| Titus | GEMM 流水线：single/multi-stage、pipeline | 712451053 | 24,25 |
| 进击的Killua | cute Swizzle 细谈 | 684250988 | 12,23 |
| 进击的Killua | CUTLASS CuTe 实战（一）基础/（二）应用 | 690703999 / 692078624 | 24 |
| 朱小霖 | cutlass cute 101 | 660379052 | 20,24 |
| BBuf | CUTLASS 2.x & 3.x Intro 学习笔记 | 710516489 | 24 |
| BBuf | Hopper Mixed GEMM 的 CUTLASS 实现笔记 | 714378343 | 24 |
| 66RING | 使用 cutlass cute 复现 flash attention | 696323042 | 25 |
| shengying.wei | FlashAttention 笔记：tiny-flash-attention 解读 | 708867810 | 25 |
| shengying.wei | FlashAttention fp8 实现（ada 架构） | 712314257 | 25 |
| JoeNomad | cutlass block swizzle 和 tile iterator | 679929705 | 12 |
| JoeNomad | cutlass bank conflict free 的 smem layout | 681966685 | 12 |
| JoeNomad | cutlass 多级流水线 | 687397095 | 24 |
| 木子知 | Nvidia Tensor Core 初探 / WMMA API / MMA PTX 编程入门 | 620185229 / 620766588 / 621855199 | 10 / 10 / 11 |
| nicholaswilde | CUDA Ampere Tensor Core HGEMM 优化 | 555339335 | 11 |
| Frank Wang | Async Copy 及 Memory Barrier 指令的功能与实现 | 685168850 | 10,13,14,17 |
| 紫气东来 | CUDA（一）编程基础 /（二）内存体系 /（三）GEMM 从入门到熟练 | 645330027 / 654027980 / 657632577 | 1 / 1 / 9 |
| 紫气东来 | ops(1) LayerNorm / ops(2) SoftMax / ops(5) 激活与残差 / ops(7)(8) self-attention 上下 | 694974164 / 695307283 / 695703671 / 695898274+696197013 | 6 / 4 / 3 / 15 |
| 白牛 | CUDA 入门的正确姿势：how-to-optimize-gemm | 478846788 | 9 |
| 有了琦琦的棍子 | 深入浅出 GPU 优化系列：gemv 优化 | 494144694 | 8 |
| 懒蚂蚁呀不嘿 | CUDA element-wise / transpose / reduce 算子详解 | 1888630735520391519 / 1899760505733756129 / 1905661893739283464 | 3 / 7 / 2 |
| DefTruth | 图解：从 Online-Softmax 到 FlashAttention V1/V2/V3 | 668888063 | **15（主）**,4 |
| DefTruth | FFPA(Split-D)：FA2 无限 HeadDim 扩展 | 13975660308 | 19,26 |
| DefTruth | vLLM Triton Merge Attention States Kernel 详解 | 1904937907703243110 | 5 |
| DefTruth | LeetCUDA v3.0 大升级（项目自述） | 19862356369 | 1,附录C |
| DefTruth | WINT8/4-(00)~(03) 快速反量化系列 | 657072856 / 657070837 / 657073159 / 657073857 | **全书行文风格基线**（BOOK_PLAN §4.3，与图解 FA 系列同为风格样本） |
