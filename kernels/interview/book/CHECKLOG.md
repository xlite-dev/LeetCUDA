# CHECKLOG — 源码注释核查记录

> 规范：BOOK_PLAN §5.3。源码冻结（RFC-A.3 `scripts/anchors.yaml` 登记 SHA256）后，发现的注释错误一律记录于此，**不回写源码**；正文以「勘误与考据」框注呈现。
> 类别：F1 架构事实 / F2 PTX 语义 / F3 数学 / F4 性能断言 / F5 历史陈述。

| 日期 | 位置 | 原文摘要 | 类别 | 证据 | 建议 | 状态 |
|---|---|---|---|---|---|---|
| 2026-09-24 | ch00 §1.8.2 正文 / ch19 两实现对照表 / tests/ch26b_fa_persist_d.cu 头注释 | 「`setmaxnreg` 在 sm_120a 上会被 ptxas C7506 静默忽略，必须用 sm_120f 构建」：把 arch 后缀当成变量，与 ch14/ch18/ch26b 已修正的结论自相矛盾 | F2 | 2026-09-22 终版矩阵（最小 probe + notes-v2 全量重编）：sm_120a 与 sm_120f 均 6/6 保留 USETMAXREG、零 C750x；真正变量是 TMA dst 状态空间（cluster→C7506）与 `__launch_bounds__(N,1)`（缺→C7508） | 三处改为「与 arch 后缀无关 + 两个静默丢弃触发」表述；全书 setmaxnreg 表述一致性复查（ch13/ch14/ch17/ch18/ch19/ch26b/ch26c/ch29/ch30 + appA/appC 无残留） | 已入正文 |
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
| 2026-09-18 | hgemm.cuh L1248 | 「一个 8×32 的逻辑 tile 正好对应 8 行 × 128B」：32 half/行 = 64B，应是 8 行 × 64B（Swizzle\<3,3,3\>，2^(M+S)=64B 行宽） | F3 | 行宽公式 + CUTLASS `Layout_K_SW128_Atom` 对照 | ch23 §23.3 已批评该注释 | 已入正文 |
| 2026-09-18 | flash_attn.cuh L2255-2256（L2213 同族） | 「GMMA 128B swizzle atom: (8,8) layout + Swizzle\<3,4,3\>」「8 行 × 8 half = 128B，恰好一个 swizzle 周期」：CUTLASS atom 是 (8,64):(64,1)，周期 1024B=8 行×128B | F3 | CUTLASS atom 定义 + 周期公式 2^(M+S+B)=1024B | ch25 §25.x 勘误框 | 已入正文 |
| 2026-09-18 | flash_attn.cuh L2213/L2263/L2266 | 三处注释把 `make_layout(Shape<D,64>, GenRowMajor{})` 称「col-major (D,64)」：B 实为 row-major，与 V^T col-major QKV 复合后的结果才是 col-major | F3 | 源码显式 GenRowMajor + 复合布局 (64,64):(1,64) 推导 | ch25 eq (25-vt) 解读处勘误注 | 已入正文 |
| 2026-09-18 | ffpa fp8/quantize_fp8.cuh L697（自相矛盾见 L702） | 注释「paired via shfl_xor(amax, 8)」；实际代码 L739-740 为 shfl_xor 1（行内两半）+ shfl_xor 16（{r,r+8} 行对），同段 L702 自写 shfl_xor 16 | F3 | 逐行对代码 | ch28 §28.4.4 listing 原样引用，正文叙述用正确的 16 | 仅登记 |
| 2026-09-18 | ffpa fp4/fp4_pscale.cuh L117 | 注释 `P2 = exp2(... + log2(1/(448*6))) in [0, 2688]`：符号与值域矛盾，应为 `+ log2(448*6)`（2688=448×6） | F3 | 值域反推 + 代码实际行为 | ch32 正文已指出笔误并按代码记述 | 已入正文 |
| 2026-09-18 | ffpa fp8/sm_120/split_d.cuh L613 | 「cannot hide the extra tensor-pipe pressure」与上文「raises math_pipe_throttle」矛盾：rescale 折叠进吸收 FFMA 增加的是 math/FMA pipe 压力，不是 tensor pipe | F4 | NCU math_pipe_throttle stall 口径 | ch30 §30.4 忠实沿袭源注释，未改措辞 | 仅登记 |
| 2026-09-18 | common.cuh L494-495 | v4.0.0（6cd32de）合并残留：注释被截断成乱码「The function templatS_V2_ENABLE_SETMAXNREGS: / default builds」，与 L491-493 语义重复 | F5 | git 比对 PR #522 合并 | 书稿 listing 不抽取该行；建议上报上游修复 | 仅登记 |
