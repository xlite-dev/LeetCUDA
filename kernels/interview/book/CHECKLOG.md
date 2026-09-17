# CHECKLOG — 源码注释核查记录

> 规范：BOOK_PLAN §5.3。源码冻结（RFC-A.3 `scripts/anchors.yaml` 登记 SHA256）后，发现的注释错误一律记录于此，**不回写源码**；正文以「勘误与考据」框注呈现。
> 类别：F1 架构事实 / F2 PTX 语义 / F3 数学 / F4 性能断言 / F5 历史陈述。

| 日期 | 位置 | 原文摘要 | 类别 | 证据 | 建议 | 状态 |
|---|---|---|---|---|---|---|
| 2026-09-14 | ffpa_attn.cuh L432/L484 | `NOTES_V2_REG_DEALLOC(40)/ALLOC(255)`：255 非 8 的倍数，违反 PTX ISA §9.7.19.5（imm ∈ [24,256] 且 8 倍数）；宏默认关闭展开为 ((void)0) 故从未暴露 | F2 | ptx-docs 9.7.19.5 + FA3 论文约束链（NVCC 每线程 ≤255，合法顶格 248）+ ffpa-attn repo persist_d 原版 32/232 | 已改为 32/232（对齐 ffpa-attn repo），ch19 五处表述同步（含 sm120 丢弃条件限定）；anchors.yaml SHA 已更新 | 已回写源码+入正文 |
| 2026-09-14 | ch14 表格/正文 | 「默认目标 sm_120a 丢弃 setmaxnreg」缺「与 TMA 同 kernel」条件，与 PTX ISA Target ISA Notes（sm_120a 在支持列表）矛盾 | F2 | ptx-docs Target ISA Notes：sm_90a/100a/110a/120a + 100f/110f/120f | 三处改为条件表述+精确支持列表 | 已入正文 |
| 2026-09-11 | ffpa_attn.cuh L17-18 头注释 | 性能口径为 PRO 5000 | F4 | README 同项数据口径为 RTX 5090 | ch19 正文双口径标注（PRO 5000 复测 + README 5090 引用） | 待 ch19 核查 |
| 2026-09-11 | sgemv.cuh L17 | 「否则内层循环 kNumWarps 次」——kNumWarps 不存在，实际 NUM_ITERS=ceil(K/32) | F3 | 轮2 agent 独立核校 | ch08 正文勘误框 | 已入正文 |
| 2026-09-11 | sgemv.cuh L91-98 | sgemv_k16 M%8!=0 时半 warp 守卫分叉，全掩码 __shfl_xor_sync 属 UB（实践碰巧正确） | F2 | CUDA 语义+轮2 agent 分析 | ch08 勘误+坑节 | 已入正文 |
| 2026-09-11 | sgemm.cuh L18-19 | 金字塔 AI 公式量纲不一致：B_K 分子分母相消，32×32 tile AI=8 而非 4；L84-85 同族 | F3 | 轮2 agent 独立推导 | ch09 勘误框（定性结论不变） | 已入正文 |
| 2026-09-11 | sgemm.cuh L193/L238/L242-243 | smem 字节数 16KB 实为 20736B；1024 floats 实为 1536/1056 | F1 | 模板参数计算 | ch10 勘误框 | 已入正文 |
| 2026-09-11 | sgemm.cuh L196-198 | 「每 MMA 64 TF32 MAC、4096 MAC/cycle」不能由配置推出 | F1 | m16n16k8=2048 MAC/tile 推导 | ch10 勘误框 | 已入正文 |
| 2026-09-11 | hgemm.cuh L31-32 | 「ldmatrix 默认加载 col-major」与 PTX 文档相反（.trans 才是 column-major） | F2 | ptx-docs 9-instruction-set | ch11 勘误框（重要） | 已入正文 |
| 2026-09-11 | common.cuh L124-125 | 「swizzle 后 1-way conflict-free」仅 32B 行宽成立；BK=64/128B 行宽下 NCU 实测仍 4-way（297,628 次/launch，模型预测 2.33 vs 实测 2.32） | F3 | 轮2 agent NCU 实证+数学模型 | ch12 勘误框（重要）；128B 零冲突需 swizzle<64>/SW128 | 已入正文 |
| 2026-09-11 | common.cuh L344-347 | 「swizzle 公开派发器」注释位置悬置，实际在 hgemm.cuh L388-396 | F5 | 轮2 agent 考据 | ch12 考据框 | 已入正文 |
| 2026-09-11 | flash_attn.cuh L7-13 | 「三板斧」（tiling/online softmax/recomputation）挂在 FA2 名下属归属错位：按 FA1 论文（arXiv:2205.14135）为 FA1 贡献；FA2 主题是 non-matmul FLOPs/并行度/split-Q | F5 | FA1/FA2 原论文摘要比对 | ch15 勘误框 | 已入正文 |
| 2026-09-11 | flash_attn.cuh L19-21 | scale 按块乘进 S（FA1 式）；FA2 论文 Algorithm 1 预乘进 Q（数学等价，非错误） | F5/考据 | FA2 论文 §3.2 | ch15 考据框（等价性说明） | 已入正文 |
