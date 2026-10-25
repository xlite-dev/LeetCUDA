# 审核与修正历史（append-only）

> 用途：记录每一轮书稿审核的触发原因、范围、证据、发现和修正，方便日后复查。
> 规则：只在末尾追加新轮次；不覆盖或删除旧记录。若旧结论被推翻，新增
> “R<N> correction to R<M>-xx” 并注明旧条目的状态，保留原记录。
>
> **名称变更**：R23 起本 skill 名为 `leetcuda-correctness-review`；R1–R5 中的
> `leetcuda-correctness-sellf-review` 是历史名称，当前路径与软链以 R23 为准。

## 基线

| 项 | 值 |
|---|---|
| 仓库 | `/home/qiuyanjun/develop/LeetCUDA` |
| 分支 | `dev` |
| 基线 commit | `bbcb1cda7159d04ed33f44fec80a758e9f1bab48` |
| 记录日期 | 2026-10-24 |
| 工作区状态 | 章节改动未提交；`book/skills/` 下新增 skill 未提交 |
| 审核范围 | `book/chapters/ch20`–`ch26c` 共 9 章 |
| 冻结源码 | `.cuh` / `bench/*.cu` 未改；只读书稿与源码 |

改动文件：

```text
kernels/interview/book/chapters/ch20-cute-layout.tex
kernels/interview/book/chapters/ch21-cute-tensor-tiledcopy.tex
kernels/interview/book/chapters/ch22-cute-tiledmma.tex
kernels/interview/book/chapters/ch23-cute-swizzle-tma.tex
kernels/interview/book/chapters/ch24-cute-hgemm.tex
kernels/interview/book/chapters/ch25-cute-flash-attn.tex
kernels/interview/book/chapters/ch26-cute-ffpa.tex
kernels/interview/book/chapters/ch26b-cute-persist-d-flash-attn.tex
kernels/interview/book/chapters/ch26c-cute-split-d-sm120.tex
kernels/interview/book/skills/write-leetcuda-book/SKILL.md
kernels/interview/book/skills/leetcuda-correctness-sellf-review/（新增）
```

## 轮次总览

| 轮次 | 日期 | 触发 | 范围 | 结果 |
|---|---|---|---|---|
| R1 | 2026-10-24 | 修订既有高/中置信度审计问题 | ch20–ch26c | 完成内容修正；但仍留有 7 处同源残留 |
| R2 | 2026-10-24 | 用户要求“逐项自审改前改后” | 9 章全部 hunk | 逐项复核 R1；新发现并回修 7 处 |
| R3 | 2026-10-24 | 复盘“为何前几轮数理审查未发现” | 方法论 | 不改书稿；确定四层审查模型 |
| R4 | 2026-10-24 | 用户要求固化为 skill | book/skills + .github/skills | 新建 skill、接线、软链、校验通过 |
| R5 | 2026-10-24 | 用户要求记录每轮审核详情 | references/ | 新增本文件；不改章节 |
| R22 | 2026-10-24 | 用户要求全书逐章正确性自审 | 40 章 + 附录 A–E；白皮书只读 | 9 处高置信表述/计数错误已回修；未编译 PDF、未跑 GPU |
| R23 | 2026-10-24 | 用户要求改名并强化验收 | skill 文档 + write-leetcuda-book 接线 | skill 改名；新增“有书稿改动必须跑 book/build.sh + RTD 三层验收”闸门 |
| R24 | 2026-10-24 | 用户要求按新 skill 执行验收 | PDF + RTD 全量 build + visual_check | PDF 通过；RTD 首跑 2 项 FAIL，补 42 条图集词条 + 修 1 条公式词典后重跑全绿；visual_check 93 页 0 需处理 |
| R25 | 2026-10-24 | 用户要求主书二轮全量自检 | 36 章 + 附录；4 个只读 Task agent | 回修 20 余处高/中置信问题；PDF 588 页、RTD 1875 条词典、visual_check 93 页 0 需处理 |
| R26 | 2026-10-24 | 用户要求 cute-zhihu 全量自检 | 34 章；2 个只读 Task agent | 回修 6 条高置信 + 4 条中置信；与 R25 共用 PDF/RTD 闸门 |
| R27 | 2026-10-25 | 用户要求加入“第零层：数理逻辑正确性” | skill 文档 + 扫描脚本 + 案例库 | 新增 L0 推导链/量词/反例层；L1 重构为定义/指称/边界；无书稿改动，未触发构建闸门 |
| R28 | 2026-10-25 | cute-zhihu 改动后要求编译，并要求 skill 强调非主 book 入口 | 新增构建矩阵；重编 cute-zhihu.pdf | cute-zhihu 248 页 / 0 error / 0 missing / Overfull 6；skill 增加“所有受影响 PDF 都要编” |
| R29 | 2026-10-25 | 用户要求把当前正确性判断写进 skill，防止后续审核者盲目自信误判 | skill 文档 + 审核者校准 | 记录主 book/cute-zhihu 置信状态、false-positive guard、残余 medium；无书稿改动，未触发构建闸门 |

---

## R1 — 内容修正轮

### 触发

根据既有审计报告，修正第 20–26c 章的高/中置信度错误。重点是“公式能自洽，
但指称、作用域、单位或硬件契约不对”的句子。

### 方法

- 逐章读源码、CUTLASS/CuTe 头文件、PTX 文档；
- 对公式类断言做反例/边界检查；
- 对实现类断言回到调用点；
- 只改书稿，不改冻结源码。

### 分章修正

| 章节 | 修正类型 | 代表性问题 → 修正 |
|---|---|---|
| ch20 Layout | 定义/定理、图 | `cosize` 缺绝对值；`size==cosize` 被当双射判据；composition 被说成“不再是函数复合”；`make_layout` 被归入 product；左右逆图与 colex 图残留 |
| ch21 TiledCopy | API/资源 | `ldmatrix` 的 source 连续 8 值被当成 `rA[0..7]`；错误 ValLayout 被说成自动降级 8 条 16-bit；duplication 被归因到每线程 fragment 放大 |
| ch22 TiledMMA | 类型/资源 | `convert_type` 被称零拷贝；256 个 f32 被说成“可用寄存器一半”；PV 小 N 被说成控制总 `acc_O` |
| ch23 Swizzle/TMA | 硬件契约 | TMA 门槛写成 sm80；`ldmatrix` 被说成原生 swizzle；退化 SW128 周期写成 2 行；TMA 对齐写死 1024B；`M=4` 绝对化 |
| ch24 HGEMM | API/资源 | `cute::gemm` 被说成不处理 K；256 累加器口径错误；BK=32 被说成完整 128B 行宽；`group_modes` 后 `i+j` 对象错误 |
| ch25 FlashAttention | 语义/硬件 | rowcol 行列顺序错；P 的转换被称零拷贝；`wait_group N` 语义反；尾部 `wait<1>` 被说成“等不存在的组”；V/K 发射顺序错；`cid` 写成 XOR 1 |
| ch26 FFPA | 算法/类型/资源 | `O=ΣO_c` 错误；P 的 smem 成本按 C 放大；k8 复用被否定；`partition_fragment_B` 被当 smem 指针划分；256 逻辑值被当常驻寄存器；`wait<Sk-2>` 组数错；stage 维写成只能指针算术 |
| ch26b persist-D | 语义/图/测试 | 全 mask 行被说成必然输出 0；persistent 被说成消除 wave 尾损；`SmemLayoutKVt` 内层写成 col-major；`_1{}` 被说成 minor 维搬满；流水图 K1 复写 stage0；setmaxnreg 被说成 WS 唯一手段 |
| ch26c split-D sm120 | 硬件/实验 | SW64 单行 64B 被说成一个完整周期；TMA OOB 被说成保留旧值；`(S,S)` 被说成任何 D 都非最优；`(4,2)` smem 写成 112KB；`_1{}` 被说成 `local_tile` 参数；D=320 VCPB 被写成固定 5 |

### 结果

- 高/中置信度问题已落到正文；
- 但 R1 结束后仍有旧模型的“第二套说法”藏在图注、问答和个别解释里；
- 未编译 PDF、未跑 GPU；性能数字保留原书实测口径。

### 轮末验证

```text
read_lints：9 个 tex 文件 0 error / 0 warning
git diff --check：通过
git status：仅 9 章和 write-leetcuda-book/SKILL.md 被修改
```

---

## R2 — 逐项自审轮

### 触发

用户要求：“这次改动项挺多的，改完后你逐项自审改前改后的正确性。”

### 方法

- 对 9 章逐一读 `git diff -U2/-U3`，按 hunk 对照“改前 → 改后”；
- 用源码/头文件/PTX 作为仲裁，而不是继续复述正文；
- 把图注、图例、表标题、面试问答也当技术断言；
- 不跑 GPU，不编译 PDF。

### 自审新发现并回修

| ID | 位置 | 改前 | 改后 | 层级 | 证据 |
|---|---|---|---|---|---|
| R2-01 | ch20 cosize 段 | 删掉旧的 compact 定义后仍使用“compact” | 补“值域连续且双射（下文称 compact）” | L1 定义 | 术语一致性；避免悬空定义 |
| R2-02 | ch21 小例子图注 | “蓝色两行 = 第 1 轮；rows 0–1” | “蓝色四行 = 4 线程全体；第 1 轮覆盖数据行 m=0,1” | L4 图注 | 4 线程每轮共同参与；表行是线程号 |
| R2-03 | ch26 图 26-1 | “Phase 2 就地覆盖，不新增”；图注“softmax 就地覆盖为 P” | “softmax 写回 S；convert_type 生成新 P”；S 32 f32、P 32 f16 | L4 图注 + L3 类型 | `convert_type` 产生新 fragment；Aregs 才重解释 |
| R2-04 | ch26 Q5 | 问“每个线程为 O 花多少寄存器”，答案说逻辑值 | 问“持有多少 O 逻辑累加值” | L3 口径 | 256 逻辑值会 spill，不能直接叫寄存器数 |
| R2-05 | ch26b P 段 | “P（f32）经 convert_type 压成 fp16，再原地重排” | “f32 S 经 convert_type 产生 f16 P；Aregs 重解释；不把转换叫零拷贝” | L3 类型 | `NumericArrayConverter` + Aregs 两段语义 |
| R2-06 | ch26b persistent 收益 | “复用 Q 寄存器与流水状态”“Q 跨 q-tile 持久” | “复用 CTA 内屏障与流水状态；Q 每个 q-tile 仍重装” | L2/L4 | 源码每个 q-tile 都发 Q TMA；Q 只在 KV 循环内持久 |
| R2-07 | ch26c VCPB | “默认 (2,2) smem 只够一个 O 分量” | “64KB 池可放 4 个分量；但批量必须整除 5，5 个装不下，故 VCPB=1” | L3 资源 | `compute_vchunks_per_batch` 的容量 + 整除双条件 |

### 逐章复核结论

- **ch20**：cosize/composition/tiler/product/图 20-7、20-8 全部通过；R2-01 补定义。
- **ch21**：source/rA 区分、ValLayout 约束、duplication 归属通过；R2-02 修图注。
- **ch22**：convert_type、256、PV 小 N 通过，无需再改。
- **ch23**：sm90/TMA 对齐、ldmatrix、周期、tile_to_shape、M=5/6 通过。
- **ch24**：gemm K、256、BK=32、group_modes 通过。
- **ch25**：rowcol/4-lane、wait_group、V/K 顺序、cid、k8 通过。
- **ch26**：O_c、P smem、partition_fragment_B、spill、wait、stage 通过；R2-03/04 修图与问答。
- **ch26b**：全 mask、persistent makespan、KVt、_1、流水图通过；R2-05/06 修 P 与 persistent 归因。
- **ch26c**：SW64、OOB、对称分段、96KB、`_1` 通过；R2-07 修 VCPB 解释。

### 轮末验证

```text
read_lints：9 个文件 0 error / 0 warning
git diff --check：通过
旧结论检索：eq:20-product、Phase 2 就地覆盖、S / P、smem 只够一个、Q 跨 q-tile 等 0 命中
未做：PDF 编译、GPU 实测
```

### 残余风险

- TikZ 图的真实字宽/重叠/分页未目视验收；
- 性能数字未复测，只保留原书口径；
- 部分“未验证”属于需要 GPU 或 PDF 才能确认的事项，不能写成已验收。

---

## R3 — 漏检复盘轮（不改书稿）

### 触发

用户问：“之前做了几轮数理都没发现这些问题，为什么？”

### 结论

不是推导错误，而是审查目标错配：

1. 前几轮检查公式自洽，本轮检查名词/作用域/单位/API 契约；
2. 错误集中在硬件/API 边界，不在算法主线；
3. “零拷贝/原地/共享/一半/只够/必然”等顺口简写掩盖了指称错误；
4. 同一套材料与心智模型会重复同一盲点；
5. 测试 PASS 只证明 case 跑通，不证明解释正确；
6. 图注、表、问答被当成附属材料，未进入主审路径。

### 产出

- 形成四层审查模型：定义/定理、API/硬件契约、资源/类型计数、表达一致性；
- 作为 R4 新建 skill 的骨架；
- 本轮不改章节。

---

## R4 — Skill 固化轮

### 触发

用户要求把四层自审做成 `leetcuda-correctness-sellf-review` skill，
保存到 book skill，并软链到 `.github/skills` 给 CodeBuddy/Copilot 使用。

### 新增/修改

```text
book/skills/leetcuda-correctness-sellf-review/SKILL.md
book/skills/leetcuda-correctness-sellf-review/references/casebook.md
book/skills/leetcuda-correctness-sellf-review/references/claim-ledger-template.md
book/skills/leetcuda-correctness-sellf-review/references/audit-history.md（R5）
book/skills/leetcuda-correctness-sellf-review/scripts/scan_risky_claims.py
book/skills/write-leetcuda-book/SKILL.md（验收步骤 + 关联 skill）
```

软链：

```text
/home/qiuyanjun/develop/.github/skills/leetcuda-correctness-sellf-review
  -> /home/qiuyanjun/develop/LeetCUDA/kernels/interview/book/skills/leetcuda-correctness-sellf-review

/home/qiuyanjun/.codebuddy/skills/leetcuda-correctness-sellf-review
  -> /home/qiuyanjun/develop/.github/skills/leetcuda-correctness-sellf-review
```

### 验证

```text
quick_validate.py：Skill is valid!
scan_risky_claims.py --help：通过
scan_risky_claims.py ch25/ch26c：可输出候选行
read_lints：0 error / 0 warning
```

### 备注

- 目录与 frontmatter 按用户给定的 `sellf` 拼写保留；若后续确认是 `self`
  笔误，需同时改目录、frontmatter、两处软链和 write skill 中的引用。
- 扫描器只产出候选，不能替代人工判定。

---

## R5 — 审计历史文档轮

### 触发

用户要求：在 `references/` 中记录每一轮审核与修正详情，方便日后复查。

### 变更

- 新增本文件 `references/audit-history.md`；
- 在 `SKILL.md` 中把本文件列入参考资源；
- 在 `write-leetcuda-book` 的验收步骤中要求把每轮自审结果追加到本文件；
- 本轮不改章节内容。

### 验证

- 只新增/更新 skill 文档；
- 不触碰 `.cuh` / `bench/*.cu`；
- 不改已记录的历史轮次。

---

## 后续轮次模板

复制到文件末尾，不要改写上面的旧记录。

````markdown
## R<N> — YYYY-MM-DD — <触发>

### 范围与基线

- commit/worktree：
- 文件：
- 未做：

### 方法

- ...

### 新发现与修正

| ID | 位置 | 改前 | 改后 | 层级 | 证据 |
|---|---|---|---|---|---|
| R<N>-01 | ... | ... | ... | L1/L2/L3/L4 | ... |

### 复核结论

- 通过：
- 保留：
- 未验证：

### 验证

```text
git diff --check：
read_lints：
扫描器：
PDF/GPU：
```

### 残余风险

- ...
````

## 快速复查索引

| 想复查什么 | 先看 |
|---|---|
| 四层怎么审 | `SKILL.md` 第 3 节 |
| 具体坏句/真值/修法 | `references/casebook.md` |
| 一轮审了哪些 claim | 本文件对应轮次 + `claim-ledger-template.md` |
| 当前残余风险 | 本文件最后一轮的“残余风险” |
| 自动找候选 | `scripts/scan_risky_claims.py` |

---

## R22 — 2026-10-24 — 全书逐章正确性自审

### 范围与基线

- 范围：`chapters/ch00`–`ch37` 共 40 章（`_template.tex` 与
  `ch19b-cute-whitepaper-zh.tex` 不在修改范围），加 `appendices/appA`–`appE`。
- CuTe 白皮书：只读 `chapters/wp/wp0`–`wp7`，作为 shape/stride/co-layout、
  coalesce、composition、complement、left/right inverse、product/tiling 的
  术语基准；不修改。
- 基线：`bbcb1cda7159d04ed33f44fec80a758e9f1bab48`（`dev`），工作区已有
  ch20–26c 的未提交修订。
- 方法：不使用 sub agent；逐章读结构、承重公式、表、图注与问答，结合
  `scan_risky_claims.py` 候选行，回到 `.cuh`/头文件/PTX/定义核对。
- 未做：GPU 实测、性能复测、PDF 编译与 TikZ 目检。

### 新发现与修正

| ID | 位置 | 改前 | 改后 | 层级 | 证据 |
|---|---|---|---|---|---|
| R22-01 | ch04 §4.8 坑三 | `__expf`/`__fdividef` 快速数学“通常无损” | 精度通常够用但非无损；`__expf` 2+ ulp、`__fdividef` 为近似除法 | L2/L4 | CUDA math API 语义；参考/回归必须用 `expf` |
| R22-02 | ch06 §6.3.5 | `rsqrtf` 的 2 ulp 误差“是无损加速” | 精度代价可忽略的近似加速，不能称无损 | L2/L3 | `rsqrtf` 为 MUFU 近似；2 ulp 非零 |
| R22-03 | ch07 图 7-5 图内与图注 | PAD=1 读侧“16 个 bank 各 2 线程” | 16 线程落在 8 个 bank、每 bank 2 线程 | L4 | `68tx mod32=4tx`，tx=0..15 只产生 8 个不同 bank |
| R22-04 | ch08 §8.3.3 + 8.8 坑二 | `K % 128 == 0` 被写成 float4 16B 对齐的必要条件 | 对齐只需基址 16B + `K % 4 == 0` + 步进 4 float；`K % 128 == 0` 是无尾块覆盖条件，不满足先越界而非 misaligned | L1/L2 | `sgemv.cuh` k128 地址 `a+mK+4(...)`；行起点对齐与 warp 覆盖是两件事 |
| R22-05 | ch15 定理 15.1、证明基础、remark (3) | 初始化 `m=-∞` 时首轮 rescale 一律为 0；空行只在最终除法出 NaN | 定理补“每行至少一个有限分数”；首个块整块 `-∞` 时 `-∞-(-∞)` 立即 NaN，后续 `0×NaN` 会污染输出 | L1/L2 | IEEE 754：`-∞-(-∞)=NaN`；与 ch26b 全 mask 结论对齐 |
| R22-06 | ch27 图 27-2 结论 | “除 round() 外无任何额外数学误差源” | 实数代数下无额外误差源；浮点乘加、exp2、低位宽累加仍有舍入 | L3/L4 | 代数等价不等于浮点执行无舍入；与 ch29/ch34 口径一致 |
| R22-07 | ch29 图内结论与图注 | 同 R22-06 的“无额外误差源” | 加“实数代数下”限定，并列出浮点舍入来源 | L3/L4 | 同 R22-06；避免图与正文两套口径 |
| R22-08 | ch36 §36.2.1 | 8 个 e2m1 幅值点在 `[0,6]` 上“平均步长 6/8=0.75” | 8 个码点只有 7 个间隔，平均间距 `6/7≈0.857`；`6/8=0.75` 是 8 个 Voronoi 格的平均格宽 | L1/L3 | 相邻码点计数；后文误差表按 8 个格积分，格宽口径保留 |
| R22-09 | ch37 §37.3.1 | 整 warp 的 SF 字节数（A 64B+B 32B）与每线程 A 数据 16B 相比，称“SF 比数据还多” | 按同口径：唯一 SF 96B 对数据 768B = 1/8；每线程 SF 8B 对数据 24B ≈ 1/3；不存在“SF 比数据多” | L2/L3 | 作用域/单位核对：A 16B/线程×32=512B，B 8B/线程×32=256B |

### 逐章结论

| 章节 | 结论 |
|---|---|
| ch00–ch03 | 未发现新的高置信错误 |
| ch04 | R22-01 回修 |
| ch05 | 未发现新的高置信错误 |
| ch06 | R22-02 回修 |
| ch07 | R22-03 回修 |
| ch08 | R22-04 回修 |
| ch09–ch14 | 未发现新的高置信错误；第 8 章的覆盖/对齐口径已同步到 GEMM 相关说明 |
| ch15 | R22-05 回修 |
| ch16–ch19 | 未发现新的高置信错误；Split-Q/Split-KV/Split-D 的 m/ℓ/O 语义复核通过 |
| ch20–ch26c | 复核 R1/R2 的修订与图注；本轮无新增改动 |
| ch27 | R22-06 回修 |
| ch28 | 未发现新的高置信错误 |
| ch29 | R22-07 回修 |
| ch30–ch35 | 未发现新的高置信错误；M8N1/M4N2 的 `D/(2N_w)` 与 per-SM 池占比复核通过 |
| ch36 | R22-08 回修 |
| ch37 | R22-09 回修 |
| appA–appE | 抽查源码行号、测试数（30 个可执行/218 PASS）与性能表口径；未发现新的高置信错误 |

“未发现”只表示静态四层审查未抓到高置信错误，不是形式化证明。

### 验证

```text
read_lints：本轮修改的 9 个 tex 文件 0 error / 0 warning
git diff --check：通过
残留检索：已删除“rsqrtf 无损”“16 个 bank 各 2”“平均步长 6/8”
         “SF 操作数字节数比数据还多”“无任何额外数学误差源”等原句
未做：GPU 测试、性能复测、PDF 编译、TikZ 目检
```

### 残余风险

- 性能数字、NCU 计数、竞品对照未复测；只核对了算式与口径。
- TikZ 图的实际字宽、换行和分页未编译目检；本轮只改了图内文字。
- 部分 API/硬件断言沿用 R21/R2 的证据链，本轮未逐条重读全部头文件。
- 白皮书只作为术语基准，未做译文校勘。
- “逐章通过”不等于全书形式化正确；真正发布前仍需 PDF 构建 + 抽样 GPU 实测。

---

## R23 — 2026-10-24 — skill 改名与每轮验收闸门

### 触发

用户要求把 skill 名称从 `leetcuda-correctness-sellf-review` 改为
`leetcuda-correctness-review`，并强调：每轮 review 只要有改动，就必须走
`book/build.sh` 与 ReadTheDocs 验收流程，不能跳过。

### 改动

```text
book/skills/leetcuda-correctness-sellf-review/ → leetcuda-correctness-review/
book/skills/leetcuda-correctness-review/SKILL.md
book/skills/write-leetcuda-book/SKILL.md
.github/skills/leetcuda-correctness-review（新软链）
~/.codebuddy/skills/leetcuda-correctness-review（新软链）
```

- `SKILL.md`：frontmatter `name` 改为新名；§5 重写为两道强制闸门：
  ① `kernels/interview/book/build.sh`；②
  `docs/readthedocs/build.sh` + `tools/visual_check.py` 三层验收。
- 明确最终验收必须全量，不得用 `--skip-tikz` / `--only` 冒充；TikZ 中文
  图必须补 `i18n/figures-en.json`，封面改动必须重编 `cover.pdf` /
  `build_cover_en`。
- 输出模板新增 `book/build.sh`、RTD `build.sh + convert.verify`、
  `visual_check` 三行；禁止事项新增“不得用 lint/diff 代替 PDF/RTD 验收”。
- `write-leetcuda-book` 验收步骤与关联 skill 改为新名，并把 build + RTD
  写成宣告完成前的固定顺序。
- 历史 R1–R5 保留旧名，只在本文件顶部加改名说明；不重写旧记录。

### 验证

```text
quick_validate.py：Skill is valid!（源目录与用户软链各一次）
use_skill leetcuda-correctness-review：可加载，base dir 指向用户软链
read_lints：0 error / 0 warning
旧软链已删除，新软链指向 book 源目录
本轮只改 skill 文档，无书稿改动，未触发 book/build.sh 与 RTD 闸门
```

### 残余风险

- 新闸门尚未在下一轮书稿改动中实际执行，首次执行时需按 §5 记录完整输出。
- 历史 R22 的“未编译 PDF、未跑 GPU”保留为事实记录；R23 起的新轮次不得
  再以“静态审查通过”代替 PDF/RTD 验收。

---

## R24 — 2026-10-24 — 按新 skill 执行 PDF + RTD 全量验收

### 触发

用户要求按 `leetcuda-correctness-review` §5 执行验收。本轮不是新的书稿
内容审查，而是对当前未提交书稿改动（R1/R2/R22）执行两道强制闸门。

### 闸门 1：PDF 构建

```text
命令：cd <LeetCUDA_DIR>/kernels/interview/book && ./build.sh
exit：0
页数：586
Overfull hbox >=1pt：16
Missing characters：0
LaTeX error / undefined reference / undefined citation：0
```

日志：`.tmp/acceptance-r24/book-build.log`

### 闸门 2：ReadTheDocs 全量构建

首跑 **FAIL**（退出码 1）：

```text
[FAIL] 英文图集完整：词典覆盖 1750/1792
[FAIL] 英译词典匹配产物：公式未命中 1、死条目 1
```

修复：

1. 运行 `python tools/translate_figures.py --proxy http://127.0.0.1:7890`
   补齐 42 条图内中文英译；词典 1822 → 1864 条，覆盖 1792/1792，
   英文图 196/196。
2. 公式词典把旧 key `(2_{\text{行}},2_{\text{列}})` 改为正文当前的
   `(2_{\text{列}},2_{\text{行}})`，英文 value 同步改为
   `(2_{\text{cols}},2_{\text{rows}})`。正文顺序与
   `convert_layout_acc_rowcol` 的 `logical_divide` 映射一致，故修词典，
   不回退正文。

重跑 **PASS**（退出码 0）：

```text
build succeeded
内容核对全部 [OK]（“书目录编号核对”因本地无 book.toc 跳过）
英文图集完整：词典 1864 条，英文图 196/196 张
英译词典匹配产物：代码块 771/771；公式 121/121
```

日志：`.tmp/acceptance-r24/rtd-build.log`（首跑）、
`.tmp/acceptance-r24/rtd-build-rerun.log`（重跑）

### 闸门 3：浏览器验收

```text
命令：python tools/visual_check.py --html build/html \
      --out build/shots/chapters --sheet build/shots/chapters-sheet.png \
      --proxy http://127.0.0.1:7890
exit：0
合计 93 页，需处理 0 页：[]
MathJax 报错 0、公式残迹 0、控制台/资源失败 0、表格/图片偏离 0、越界公式 0
```

日志：`.tmp/acceptance-r24/visual-check.log`
接触表：`docs/readthedocs/build/shots/chapters-sheet.png`

### 改动文件

```text
docs/readthedocs/i18n/figures-en.json   +42 条图内英译
docs/readthedocs/_static/math-en.js     1 条公式 key/value 同步
kernels/interview/book/book.pdf         重新构建（586 页）
```

本轮未改章节正文、未改冻结源码。`figures-en.json` 与 `math-en.js`
属于 RTD 构建词典，不是书稿内容。

### 残余风险

- 新增 42 条英译由现有机器翻译产线生成，只验证了占位符完整、可编译、
  页面覆盖；英文语义与英文模式下的图内排版未逐图人工润色。
- PDF 的 Overfull hbox ≥1pt 为 16，比 R21 基线 13 多 3；`build.sh` 不以
  Overfull 判失败，本轮未逐条定位，建议下一轮排版审查处理。
- 接触表整张过大（约 686M 像素），无法在会话中整张目检；以
  `visual_check` 的逐页 93 页几何断言为准。
- `书目录编号核对` 因 `book/build.sh` 清理了 `book.toc` 而跳过；H1 章号
  已核对，但目录层级本身未在本轮核对。
- 未 push、未触发 RTD 远端构建；本地验收通过不等于线上已部署。

---

## R25 — 2026-10-24 — 主书第二轮全量自审（Task agent 四路复核）

### 触发与方法

用户要求对 LeetCUDA 主 book 再做一轮全量自审，并明确 subagent 已可用。
本轮派 4 个只读 Task agent，分别覆盖：

```text
A: ch00–ch14
B: ch15–ch19b + 附录 A–E
C: ch20–ch26c
D: ch27–ch37
```

每个 agent 先读 `SKILL.md`/`casebook.md`，跑 `scan_risky_claims.py`，再逐章
回源码/CuTe 头文件/PTX 复核；主 agent 对所有 high 候选逐条重验后再改，
没有直接采用 agent 结论。R22/R24 已修旧账不重复计。

### 高置信回修

| 位置 | 问题 | 证据 | 修法 |
|---|---|---|---|
| ch09 表 9-bench | 实测行名 `naive` 与同章 AI=0.25 的解析 naive 混用；实测行其实是 32×32 `sgemm` | `sgemm.cuh:33` BM=BN=BK=32；表内模型带宽 516/642/683 与该实现一致 | 表行改 `sgemm`，表注注明 32×32；相关 4.76×/5.5T/边界条件等表述同步改名 |
| ch06 §6.2 | “四个 kernel 统一 block=128”与 vec4 版 block=32 冲突 | `base.cuh:699/766` 默认 `128/4`；测试 launch 为 32 | 改为标量版 128、vec4 版 32 |
| ch19 图 19-1 | D=128/D=192 的 phase 周期写成 1/2 | 同章表 19-phase 为 2/4；源码 `ffpa_attn.cuh:443-445` | 改为 2/4，与 lcm(C,2Sk)/C 一致 |
| ch17 §17.4 listing | 标题承诺“释放 V”，listing 1260-1306 不含 `empty_V.arrive()` | `flash_attn.cuh:1307-1312` | 延伸到 1312 |
| ch15 定理 15.1/remark/坑二 | 证明把“首块全 mask”误称“全 mask 行”；坑二仍把 NaN 归到最终除法 | IEEE `-inf-(-inf)=NaN`；源码 `-INFINITY` 初始化 | 统一为“首块空前缀/全 mask 行”两种情形，区分 -∞ 与 -FLT_MAX |
| ch16 §16.4 | “K/V 每线程管 4 行”方向写反 | `flash_attn.cuh:150-153`：256/64=4 线程/行 | 改为每 4 线程管一行 |
| ch20 §4.8/图 20-3 | 把 f32→f16 的 `convert_type` 混进“同一批寄存器/zero-copy” | `flash_attn.cuh:2762-2766`；ch22/ch25 已有相反口径 | 限定为“布局层 zero-copy”，图注补 dtype 与 convert_type |
| ch20 §4.8 | “列方向拆成两半、后一半拼进 MMA mode”与公式/`logical_divide` 相反 | 式 20-fa2-tri；`cute/layout.hpp:1559`；mode 是 n mod 2 | 改为“按模 2 交错拆，前一个因子 2 并入 MMA mode” |
| ch25 listing | 2786-2798 截断在 `make_tensor` 中间 | `flash_attn.cuh:2796-2803` | 延伸到 2803，正文行号同步 |
| ch23 §3.4 | 称 GMMA 原子裸 `layout(r,c)` 必须与硬件 swizzle 逐点相等 | 本书测试 `ch23_swizzle_tma.cu:194-202` 断言 0<match<4096；实测 1024/4096 | 改为按字节地址比较 `Swizzle<3,4,3>`，并注明原子是位置相关语义 |
| ch21 §3.7 | 把 R→S 的 dtype 转换归给 `UniversalCopy` | `hgemm.cuh:1097-1124`：转换在 R2R staging，R2S atom 只写 32-bit | 改为 R2R staging 完成 cast；默认 F16 acc 下是 no-op |
| ch26b listing | 标题 L3866 与 range 3864 不一致 | `flash_attn.cuh:3864-3881` | 标题改为 L3864-3881 |
| ch27 §27.3.5 | 正文仍写“除 round() 外没有额外数学误差源” | 同章图 27-2 已限定实数代数；浮点仍有 exp2/乘加/累加舍入 | 改为“实数代数下无额外代数误差源” |
| ch27 §27.3.6 | lazy rescale 称“端到端无损” | `softmax.cuh:50-56`：stale max 改变发射值与舍入；fp8 还受 2^τ 饱和 | 改为实数代数等价；浮点有舍入，fp8 有饱和 |
| ch27 §27.3.4 | “约束全部落在 V 侧、P 保持 448”漏了 Bc=128 | `persist_d.cuh:741-747`：Bc=128/f16 acc 时 P=224；launch D≤128→Bc=128 | 按 Bc≤64/128 分档，说明主场景 D≤128 两侧同时收窄 |
| ch27 图 27-1 | 图注称“全部 16 个码点在数轴标出”，图只画正半轴 8 个 | 图内 `foreach 0..7`；同图注“正半轴含 0 仅 8 个” | 改为“正半轴 8 个，负半轴镜像共 16 个” |
| ch29 导读/图示 | 两处“除 round() 外不引入任何额外误差源”未随 R22 图注同步 | 同章图内已限定实数代数 | 两处同步为实数代数 + 浮点舍入 |
| ch27/31/34/36 | “相对步长”有时指 ULP/值，有时指 RN 误差上界（2^-4） | ch27 定理；ch34 `u=2^-3`；ch36 已写 e2m1 下沿/上沿 | ch27 表列改“相对误差上界”并加 ULP 范围注；ch31 全部改“相对误差上界”；ch34/36 改“相对 ULP 下沿/上沿”，误差模型明确取下沿保守值 |

### 中置信回修

- ch01 softmax FLOPs 清单补“max 比较不计 FLOP”；“AI 的上限 0.625”改为
  “该口径下 AI=0.625”，并注明 ch04 的 2 FLOP/元素口径得 0.25。
- ch02 `block_reduce_sum` 的 smem 流量改为“W 次写 + W 条读事务
  （合计 W² 个 float）”，与源码每个 warp 都读全部部分和一致。
- ch04 “乐观口径”改为“最省 FLOP 口径”。
- ch14 `setmaxnreg` 的 `.inc/.dec` 方向改为“当前 ≥ 目标 / 当前 ≤ 目标”。
- ch16 首轮三元判断不再归因于“避免 NaN”；dense 路径 rowmax 恒有限。
- ch20 图 20-8 图注把 `make_layout(A,B)` 的 stride 误写成 shape，改为
  `((2,2),(2,2)){:}((2,1),(8,4))`。

### 未回修 / 残余

- ch17 vs ch18 对 129/257 arrive 计数的归因不同（线程对齐 vs 线程数），
  属解释口径问题，未改正文。
- ch26c “每线程 32C 次标量 store → VCPB 次 STSM”的指令计数依赖 r2s atom
  打包，未在 GPU 上数，保留为待核。
- ch27 “四个误差源”只展开三项；ch33 smooth-v +1.0% 的机制归因；ch36
  误差预算表中 3.24% 与 10.17% 的合成口径，均为 medium，未改。
- 全部 TFLOPS/ms/spill 数字未复测；GPU 性能不在本轮闸门内。

### 闸门

```text
PDF：exit 0，588 页，Overfull >=1pt = 16，Missing characters = 0，
     LaTeX error / undefined reference / undefined citation = 0
RTD：exit 0，英文图集 1875 条 / 1792 单元（1792/1792），
     代码块 771/771，公式 121/121
visual_check：exit 0，93 页，需处理 0 页
```

日志：`.tmp/acceptance-r25/book-build-rerun.log`、
`.tmp/acceptance-r25/rtd-build-final.log`、
`.tmp/acceptance-r25/visual-check-final.log`。
首轮 RTD 因 8 条新图内文字缺词典失败；补 8 条后 ch31 术语再改，又补 3 条，
最终 1875 条全绿。另用 Playwright 对 ch09 表、ch19/20/27/31/34 图做了
元素级截图目检。

---

## R26 — 2026-10-24 — cute-zhihu 全量自审（Task agent 两路复核）

### 方法

2 个只读 Task agent：`reed-*` 24 章、`zhuxi-* + frankshi-01` 8 章。两路都
先跑风险扫描，再回 CUTLASS/CuTe 头文件与 PTX；主 agent 对 high 逐条复核。

### 高置信回修

| 位置 | 问题 | 证据 | 修法 |
|---|---|---|---|
| zhuxi-02 §3 | 两处 `cute::gemm(tiled_mma, trC, trA, trB)` 参数序反了 | `cute/algorithm/gemm.hpp:83` 四参是 `(mma,A,B,C)`；同章 L222 已写对 | 两处改为 `(mma, trA, trB, trC)` |
| zhuxi-06 §3 | fp64 例子里 4 个值说成一次 STG.128 | 4×8B=32B；sm_80 的 v4.f64 不可用；fp64 thread0 置换前已有 2 个连续 double | 补“fp64 需两次 STG.128；fp32 4 值正好一次；fp16 4 值 STG.64” |
| zhuxi-04 §4 | “divide 前后 Size 相同”缺整除前提 | `logical_divide((10):(1),(4):(1))` 的 Rest 取 ceil，Size=12 | 改为整除时相等；不整除时 Rest 取 ceil，末块含越界 |
| reed-03 Tensor 表 | `make_tensor_like/make_fragment_like` 同时列在 Heap non-owning 与 Stack owning | `tensor_impl.hpp:369-443` 走 owning/ArrayEngine | 从 Heap non-owning 行删除 |
| reed-isa-02 §2 | 寄存器分配粒度写 4 | 全书 occupancy 口径按每线程 8（每 warp 256）；sm_70+ | 改为每线程 8；4 只是图 2-b 示例 |
| reed-08 §2 | 位域图引用图 4，实际是图 5 | 同文件 Figure-4/5 caption | 改为图 5 |

### 中置信回修

- reed-isa-04：FMA 翻倍的是 FLOP 吞吐，指令吞吐与 FADD/FMUL 相同。
- reed-06：offset 上界 0–256 改为 0–255，cosize=256。
- frankshi-01：TMA 只要求行字节数 ≤ SWIZZLE_SIZE，等号是本章简化假设。
- zhuxi-07：`producer_commit` 改为 `(state, bytes)`；TMA 场景为 NOP，
  不 arrive full barrier。

### 未回修 / 残余

- reed-02 product 表中 logical 行 shape 的 by-mode/单 Layout 两种读法仍有
  歧义，标 medium；未改，避免把作者原意改反。
- zhuxi-04 “tiled_product 在 cutlass 没被用到”是版本相关绝对断言；
  zhuxi-05 “8-bit 字段 ≤256”与字段存值-1 的口径；zhuxi-03 “一一映射/
  单射”术语，均保留为 medium。
- cute-zhihu 的性能/实测数字未复测；本轮只做静态与构建验收。

### 闸门

与 R25 共用：PDF 588 页、Overfull 16、缺字 0；RTD 1875 条词典、
1792/1792、代码 771/771、公式 121/121；visual_check 93 页 0 需处理。
本轮还修改了 ch31 图内文字，词典从 1872 增至 1875，重跑后全绿。

### 经验沉淀

本轮方法已写回 skill 本体：`SKILL.md` §3 的 Step 4（分片并行/候选复核/误报
控制）与 Step 5（修复传播/四表面）、§5.3（TikZ 改字后先全量 build 再补词典）、
§6（相对步长、STG.128、归约、Size 相同、不重不漏等触发词）、§7（同源传播与
agent 候选复核输出）；案例库新增 17–25（gemm 顺序、listing 截断、表行名、
phase 周期、GMMA 原子、R2S cast、divide ceil、STG 宽度、agent 误报）；
claim ledger 模板加入“同源位置写全”和“未复核候选不得回修”。

### 本轮触及文件

> 其中 ch07/ch08/ch20–ch26c/ch27/ch29/ch36/ch37 等文件同时含 R1/R2/R22
> 的旧改动；不能把整份 `git diff` 都归到 R25/R26。下面只列本轮实际编辑过的文件。

```text
book/chapters/ch01,ch02,ch04,ch06,ch09,ch14,ch15,ch16,ch17,ch19,
ch20,ch21,ch23,ch25,ch26b,ch27,ch29,ch31,ch34,ch36
book/cute-zhihu/chapters/frankshi-01,reed-03,reed-06,reed-08,
reed-isa-02,reed-isa-04,zhuxi-02,zhuxi-04,zhuxi-06,zhuxi-07
docs/readthedocs/i18n/figures-en.json（+11 条）
book/book.pdf（588 页）
```

未改冻结 `.cuh`/`bench/*.cu`；未 push。

---

## R27 — 2026-10-25 — skill 增加第零层：数理逻辑正确性

### 触发

用户要求在 skill 中加入最重要的“第零层：数理逻辑正确性”，放在原有四层之前。

### 改动

- `SKILL.md`：
  - 标题与 description 改为“第零层 + 四层”；
  - 核心原则表新增 L0：推导链、量词/蕴含、边界/反例；
  - Step 3 改为“第零层 + 四层审查”：L0 检查 `P1..Pn ⇒ Q` 每一步、代数变形、
    量词方向、证明结构、恒等/近似/上界；
  - 原第一层重构为“定义、指称与边界”，去掉与 L0 重复的逐步推导；
  - §4 边界矩阵说明服务 L0/L1；§6 增加“因此/所以/代入/约去/消去/上界”等
    高风险词与句型；§7 输出模板新增第零层结论与逐章 L0；§9 增加“不跳步、
    不单向冒充双向、不把近似/上界写成恒等”；§10 改为先跑 L0 再跑四层。
- `references/claim-ledger-template.md`：新增 L0 示例、L0-L4 层级定义、按层
  重排证据等级、报告模板加入第零层结论与 L0 行、快速判断卡加入推导连接词、
  最小操作顺序先跑 L0。
- `references/casebook.md`：改名为“第零层 + 四层案例集”；原案例 1–3 归入
  L0，新增 L0-4（ch15 首块全 mask 前提缺口）、L0-5（ch02 `W²` 读流量）、
  L1-1（`make_tensor_like` 是 owning）、L1-2（寄存器粒度 8 而非 4）。
- `scripts/scan_risky_claims.py`：新增 `L0-logic` 候选（因此/可推出/代入/
  约去/消去/当且仅当/充分必要等）。
- `write-leetcuda-book/SKILL.md`：验收步骤改为“第零层 + 四层”。

### 验证

```text
quick_validate.py：Skill is valid!
read_lints：0 error / 0 warning
scan_risky_claims.py：exit 0，L0-logic 70 条候选（只作候选，不自动改）
git diff --check：通过
```

本轮只改 skill 文档与扫描脚本，**无书稿改动，未触发 book/build.sh 与
RTD build/visual_check 闸门**；上一轮 R25/R26 的 PDF/RTD 结果仍然有效。

### 残余风险

- L0 的 `scan_risky_claims.py` 规则是句型候选：`消去/约去/因此` 会命中大量
  合法推导，必须人工判，不能按命中数当错误数。
- “第零层”和“第一层”仍可能在“双向断言”上重叠：L0 判“推导方向是否成立”，
  L1 判“定义/指称是否对”；遇到争议时先判 L0，再补 L1 定义，不要反复改同一句。
- 案例 1–3 编号沿用旧号、L0-4/L0-5 与 L1-1/L1-2 使用层内编号；这是为了
  保持 append-only 历史，不追求全书统一编号。

---

## R28 — 2026-10-25 — 非主 book 入口也必须编译 + cute-zhihu PDF

### 触发

用户指出：cute-zhihu 本轮也有改动，`cute-zhihu.pdf` 也需要重新编译；并要求
把“非主 book、有修改也要编译验证”写进 skill。

### 实际构建

```text
cd kernels/interview/book/cute-zhihu && bash build.sh
exit：0
errors：0
missingchar：0
overfull：6
Output written on cute-zhihu.pdf (248 pages)
```

`cute-zhihu.pdf` 是入库文件：45963600 → 45968368 bytes（248 页）。
`cover-tikz.pdf` 按仓库约定不入库。

用 `pdftotext` 定位修改页并检查关键文本：

```text
gemm(tiled_mma, trA, trB, trC)   pages 117, 120
divide ceil / Size=12            page 136
两次 STG.128 / 4×8B=32B          page 155
producer_commit(state, bytes)    pages 158–159
每线程 8 个                      page 168
```

并渲染 p117/p136/p155/p158/p168 目检：代码块未越界、正文段落未溢出、
表格/图注正常。日志：`.tmp/acceptance-r27/cute-zhihu-build.log`；
截图：`.tmp/acceptance-r28/shots/`。

### skill 改动

- `SKILL.md` §5 改为“按改动路径构建所有受影响入口”：
  - 新增 §5.0 构建矩阵：主 book → `book/build.sh`；cute-zhihu →
    `cute-zhihu/build.sh`；RTD 词典/转换器 → RTD 全量；两处都改 → 两个 PDF。
  - §5.1 改为“所有受影响 PDF”闸门，明确 `cute-zhihu/build.sh` 即使 xelatex
    有 `!` 也可能返回 0，必须手动 grep `.tmp_build2.log`。
  - §5.4、§7、§10 改为逐个记录 `book.pdf` / `cute-zhihu.pdf`，不能只写
    `book/build.sh`。
  - §1 触发场景加入 `book/cute-zhihu/`、`book/appendices/` 等独立入口。
  - §9 禁止事项加入“不只编主 book、漏掉独立分册”。
- `references/claim-ledger-template.md`：报告模板的 PDF/RTD 与验证区改为
  逐个列受影响 PDF。
- `write-leetcuda-book/SKILL.md`：验收步骤改为按构建矩阵编所有受影响 PDF。

### 验证

```text
quick_validate.py：Skill is valid!
read_lints：0 error / 0 warning
py_compile scan_risky_claims.py：OK
git diff --check：通过
cute-zhihu build.sh：exit 0 / 248 pages / 0 error / 0 missing / Overfull 6
```

本轮没有新的主书正文改动；R25 的主 `book.pdf`（588 页）与 RTD 结果仍然有效，
因此没有重复跑主书/ RTD 闸门。**但 cute-zhihu 入口已单独编译并验收，不能
用主书 PDF 代替。**

### 残余风险

- `cute-zhihu/build.sh` 没有像主 `book/build.sh` 那样把 error/missing/overfull
  做成硬失败；当前靠人工 grep 和退出码双检。后续可考虑给该脚本加
  `if grep -q '^!' .tmp_build2.log; then exit 1; fi`，但本轮未改冻结构建脚本。
- 只抽查了本轮修改相关页（117/136/155/158/168），没有逐页目检 248 页；
  `cute-zhihu` 没有独立的 `visual_check` 管线，版式验收仍以抽样渲染为主。
- `cute-zhihu.pdf` 是 44MB 二进制，diff 只显示大小变化，代码审查应看
  `chapters/*.tex` 与 `pdftotext` 抽查，不要试图 review 二进制。

---

## R29 — 2026-10-25 — 记录当前正确性判断，防止后续审核者误判

### 触发

用户要求把“主 book / cute-zhihu 的正确性判断”写进 skill，避免后续审核者
盲目自信，把本来正确的内容判成错误。

### 改动

`SKILL.md` 新增 §2.1“当前正确性状态与审核者校准（2026-10-25 快照）”：

- 总体判断：
  - 主 book：结构性正确性高，发布候选/高质量教学稿；不能宣称零错误、
    所有性能已复测或形式化证明。
  - cute-zhihu：概念正确性中高，高质量学习资料/参考级；不能当 CUTLASS
    规范，也不能说所有代码可编译。
- 当前构建状态快照（只证明产物/页面，不证明内容）：
  `book.pdf` 588 页 / 0 error / 0 missing / Overfull 16；
  `cute-zhihu.pdf` 248 页 / 0 error / 0 missing / Overfull 6；
  RTD 1875 条 / 1792 单元、代码 771/771、公式 121/121；
  `visual_check` 93 页 0 需处理。
- 四条校准规则：先对齐前提/定义域/版本；举证责任在主张错误者；旧 PASS
  与 agent high 不是证明；不要为了统一而误伤不同断言。
- false-positive guard：
  - ch18 `β=e^{-∞-m0}=0` 在 dense 路径 `m0` 有限，不是错误；
  - 相对 ULP 2^-3 与 RN 误差上界 2^-4 是不同量，不能互判错；
  - `P` 的布局 zero-copy 与 `convert_type` 是两件事；
  - 未复测性能数字只能标“未复测”，不能判 false；
  - agent 的 high 必须回定义域复核。
- 残余 medium：主 book 的 ch17/18 归因、ch26c VCPB、ch27 四误差源、ch33
  +1%、ch36 误差预算；cute-zhihu 的 reed-02 product 表、zhuxi-04
  tiled_product、zhuxi-05 8-bit 字段、zhuxi-03 术语。明确这些是
  “证据不足/口径未定”，不是“已确认错误”，不能为清零硬改。

### 验证

```text
quick_validate.py：Skill is valid!
read_lints：0 error / 0 warning
git diff --check：通过
本轮只改 skill 文档，无书稿改动，未触发任何 PDF/RTD 闸门
```

### 残余风险

- 这是 **2026-10-25 工作树快照**，不是永久结论；书稿或源码变动后必须重审，
  不能拿 R29 当“以后都不会错”的护身符。
- 构建状态数字会随版本/页面变化；它们只用于“当前产物可生成”的记录，
  不是内容正确性证明。
- 残余 medium 可能在未来被证明为真错误；R29 只约束“证据不足时不要硬改”，
  不豁免补证据的责任。

---

## R30 — 2026-10-25 — CuTe 白皮书纯翻译：只通读，不逐条审

### 触发

用户要求明确：`ch19b` 的 CuTe 白皮书章节是纯翻译，可以当作正确，
不做技术审核；但必须读一遍，作为后续 CuTe 相关章节审核的基础知识。

### 改动

`SKILL.md` 新增 §1.1“pure translation 白皮书：只通读，不逐条审”：

- 对象：`chapters/ch19b-cute-whitepaper-zh.tex`（《导读：CuTe 布局表示与代数
  （NVIDIA 官方白皮书中文译注）》）。
- **不进入 L0–L4 逐条审核**，不要求 claim ledger/反例/外部证据；
  不因译文措辞与本书其他章节不同而判错。
- **必须通读一遍**，作为 layout/atom/tensor/composition 的术语与概念基线；
  审 ch20–ch26c、cute-zhihu 的 CuTe 章节前先读。
- 仍做编译/排版检查；技术断言不作为审核对象。
- 下游章节与白皮书冲突时，以 CUTLASS/CuTe 头文件、PTX、源码为准；
  记录为“下游与基线冲突”，不要反推翻译错，也不要把白皮书当规范去改其他章节。
- Step 1 建立改动清单时，ch19b 的 hunk 不登记 L0–L4 技术 claim；只记
  “白皮书基线已通读/排版检查”。
- §2.1 当前状态表加例外说明；§7 输出模板新增“白皮书基线：ch19b 已通读/未读”。
- description 加一句：ch19b 白皮书纯翻译只通读作基线，不做逐条审核。

### 验证

```text
quick_validate.py：Skill is valid!
read_lints：0 error / 0 warning
git diff --check：通过
本轮只改 skill 文档，无书稿改动，未触发任何 PDF/RTD 闸门
```

### 残余风险

- “纯翻译可当正确”是用户给定的编辑约定，不等于逐字对照过官方英文源；
  若未来发现译文与官方源/头文件冲突，仍按“下游与基线冲突”记录，不静默改。
- 白皮书仍参与 PDF/RTD 编译与排版检查；这并不把它重新变成技术审核对象。
- 后续审 CuTe 章节时如果没读白皮书，不能声称“已按基线审过”。

## R31–R37（2026-10-25）：六轮全量自审 + 定点勘误

- 范围：主 book 38 章 + ch19b（纯翻译通读豁免）+ 附录 5 + cute-zhihu 32 章（约 4.4 万行）。
- 方法：主 agent 逐章审核（不派 task agent），R31 全量 L0–L4 → R32 量词/推导专轮 → R33 源码锚点与 PTX 契约轮 → R34 图注/跨章一致轮 → R35 独立脚本数值复算轮 → R36 收敛复核 → R37 定点修复与验收。
- 发现：18 处观察（3 数值笔误 + 15 表述口径），无 blocking 错误；2 处初判经复核为误报并撤回（ch17 “Sk 最优/饱和”实有复测表 Sk=3 支撑；ch22 三元组为 gemm (M,N,K) 口径非 tile_size）。
- 修复（commit `4506dff`，14 文件 +32/−26）：ch20 f(4)=1；ch27 预算线 2047 精确值（图/图注/正文）+ SA1 按论文改 2 倍并注明 FA3 仅 Hopper（papers/SageAttention1.txt L62-64/L76-78/L382）；ch32 加速比 1.55x/1.47x + 硬编码表号改 \ref{tab:31-bench}（3 处）；ch02 归纳步类切分方向；ch03 T=B 持平/T<B 增至 B/T；ch04 二级归约 lane 守卫（base.cuh L654-656）+ 实数域/fp32 舍入限定；ch09 图注节选口径；ch16 图注 lane 公式 mod 8；ch22 gemm 维度标注；ch15 前置条件；ch26b 独立 A/B 不可相加；ch00 mpt 与 pipe_tensor 交叉验证；figures-en.json 换键并删旧键。
- 证据：主 book 两遍 xelatex 零 error、588 页（与 HEAD 基线 worktree 构建同页数）、Overfull 17 与基线一致、0 缺字、无未定义引用；RTD ./build.sh 全绿（英文图集 196/196）；tools.visual_check 93 页 0 问题；成品 pdftotext 抽检旧错文本归零、ch32 引用渲染为表 38.2。
- 流程教训：同一文件多处编辑须顺序执行（并行编辑丢过 ch04 一处，复核 grep 才暴露）；硬编码“表 X.Y”会随章号漂移错引，一律 \ref；Review Plan/Code Review 各拦下一处误改（ch17 误删有据句、ch32 错表号），保留其拦截价值；图内中文标签改动必须同步 figures-en.json 否则英文图集必红。
- 台账：book/.tmp/math-audit-r31/ledger.md（R31–R36 逐章结论与复算记录）。

## R38–R42（2026-10-25）：R37 修复后五轮深度复审（每轮轮末独立复审）

- 范围：主 book 38 章 + 附录 5；不改书稿，每轮结果经 Step 3.5 深度复审后才进入下一轮。
- 五轮视角与结论：R38 修复回归轮（12 被改章全文/精读，16/16 修复验证正确、零回归；新发现 ch00 小结传播缺口 1 HIGH + 表格单元 1 LOW 经复审降级）；R39 L0 深化轮（9 条核心形式化链独立重推全对 + 180 处量词命中抽样核语境，零新发现）；R40 L1/L2 轮（源码锚点 48 处全量核对 + PTX 契约 3 组 + persist_d.cuh 跨仓 constexpr 判定逐字吻合，零新发现；2 处提取疑点系正则截断误报）；R41 L3/L4 轮（修复主题全表面传播扫描仅余 ch00 一处、资源计数表独立复算全对、Q&A/小结抽样全对）；R42 数值+收敛轮（舍入带区间相交判据，ch32 五格与 ch31 争议格全带内，确认 R37"带内不动"处置正确）。
- 复审实证：Step 3.5 规则 5 两次拦截脚本假阳性（R41 ch17 smem 六格=脚本 pad 口径错；R42 首版判据错被锚点自审当场拦截）；规则 2 拦截 1 处 medium 误判（ch00 表格单元反向自辩成立，降级 LOW）。
- 章节裁定：33 章 + appA/B/C/E 正确；带观察：ch00（HIGH 小结传播缺口待回修 + LOW）、ch15/ch07/ch06（LOW/ultra-low 措辞）、ch27/appD（MEDIUM 未验证，不改）。
- 本轮无书稿改动，未触发任何 PDF/RTD 闸门。台账追加于 book/.tmp/math-audit-r31/ledger.md。

## R43（2026-10-25）：high/low 发现逐条复审 + 确证修复

- 范围：R38–R42 收敛出的 5 项 high/low 发现（ch00×2、ch15、ch07、ch06）逐条深度复审（扩大回读 + 反向自辩），确证才改。
- 复审结论：**确证 3 项**（ch00 小结"mpt 高说明 TC 已饱和"与例三直接矛盾；ch00 表格"计算饱和→减 FLOPs"行动指导口径不完整；ch15 交叉项拿 E[q_iq_j] 当理由属"真命题+错对象"）；**撤回 2 项**（ch07 复数等式在复数域严格成立、Re 已在括号注明，非隐含；ch06 "约 2-3 次"与"约 3/5 FLOP"口径相容，AI 0.3/0.6 与 3/8、5/8 自洽）。
- 修复：ch00 小结与表格改"数学管线排队/饱和 + pipe_tensor 交叉验证"口径；ch15 交叉项改为 E[q_ik_iq_jk_j]=E[q_ik_i]E[q_jk_j]=0 因子化。
- Code Review（可提交）6 项 🟢 建议经回读原文复核后采纳 5 项：表格补"未确认饱和→拆链增并行 warp"分支（决策树第 2 步）、"TC"展开为 Tensor Core、正文拆句（not_selected 健康 vs mpt 排队）、小结例三数据标 sm_89、ch15 假设句显式补 q⊥k（E[s] 与因子化的承重前提）；≠ 的 ToUnicode 提取异常为非本次引入的字体问题，不在本轮范围。
- 验收：主 book 两遍 xelatex，589 页（+1，新增 3 行文字顺延）、0 error、0 缺字、Overfull 17 与基线一致、0 undefined；pdftotext 抽检三处新句在、旧错句（"TC 已饱和"/"好信号：计算饱和"/"在正常工作"）归零；RTD build.sh 34 项 OK（英文图集 196/196）；visual_check 93 页 0 问题。cute-zhihu 无改动，未触发其 PDF 闸门。
- 流程记录：本轮再次出现"复审推翻审核结论"两例（ch07/ch06），印证 Step 3.5 的必要性；Code Review 建议同样先回读原文再采纳。

## R44–R48（2026-10-25）：第二轮五视角深度审核（每轮轮末独立复审）

- 范围：主 book 38 章 + 附录 5（R43 修复后状态，基线 8d92a8b）；只审不改。
- 五轮视角与结论：R44 叙事因果链（因/所以/导致 逐章核链路，含 ch37 25 条）——零确证，1 LOW 观察（ch37 图注"容忍度更低"相对 FP8 kBK=128 基准 vs 正文"不是等数据"绝对实测，可并存，自辩撤回）；R45 L1 术语指称（逻辑值/私有副本/原地/stage/chunk/Phase/atom/per-call）——零确证，1 LOW（per-call 首现未定义但上下文自足，撤回）；R46 图内/表注数字（ch30/31/36/37 图内数字与表内数据独立复算）——零新发现；R47 listing 端点（174 个 listing 标题行号 100% 一致，10 处端点告警全为分支边界/连续分段/说明性裁剪/正则误报）——零新发现；R48 问答/小结/标题计数 + 收敛——计数声明全对，小结数字全复算吻合（含 ch37 91–93% 分母口径）。
- 章节裁定：33 章 + appA/B/C/E 正确；2 条 LOW 观察（ch37/ch22，不改）；既有 medium 维持（ch17/ch18 arrive 计数、ch26c STSM 计数、ch27 四个误差源/254 码点、ch33 归因、ch36 合成口径、appD 双基准）。
- 本轮无书稿改动，未触发任何 PDF/RTD 闸门；台账追加于 book/.tmp/math-audit-r31/ledger.md。

## R49–R53（2026-10-25）：第二轮全文逐章通读（每轮 8 章 + 附录，轮末复审）

- 覆盖：R49 ch00–ch07+appA；R50 ch08–ch15+appB；R51 ch16–ch23+appC；R52 ch24–ch29+appD；R53 ch30–ch37+appE。主 agent 逐章全文（仅滤 TikZ 绘图指令，保留节点文字/图注/表格/问答/listing 标题）。
- 发现：仅 1 处 LOW——appA `create_tensor_map` listing 标题 `L765-785` 与区间 `764-785` 差 1 行（已改 764-785）。同时暴露 R47 的覆盖缺口：其 174 个 listing 只扫 `chapters/`、未扫 `appendices/`；补扫附录 5 个 listing 后确认仅此 1 处不一致。
- 复核口径：所有章节的数值表/公式/资源账均现场复算（AI/ridge、smem/寄存器池、加速比、误差界、descriptor 位域、swizzle 周期、barrier 计数），与既有台账一致；ch12/ch13/ch14/ch16/ch17 等长章采用"两遍分段读全"。
- 验收：主 book 589 页 / 0 error / 0 缺字 / Overfull 17 与基线一致 / 0 undefined；`pdftotext` 命中新标题；RTD `build.sh` 34 项 OK、无 FAIL；visual_check 93 页 0 问题。cute-zhihu 无改动未触发。
- 结论：五轮零新确证问题（除上述 1 处 LOW），全书维持"结构正确、可发布候选"状态。

## R49–R58（2026-10-25）：cute-zhihu 专项十轮深度审核（只审不改）

- 范围：`cute-zhihu/chapters/` 32 章 7179 行全量；主 book 不动；不派 sub agent，主 agent 逐章审核；
  用户明确"只审核总结，不改任何内容"，全程只读。
- 十轮视角：R49 全量通读 L0–L4 → R50 L0 量词/边界 → R51 L1 术语指称 → R52 L2 源码/PTX 实证
  （12 锚点）→ R53 L3 计数 → R54 L4 一致性 → R55 独立脚本数值复算 → R56 scan_risky_claims →
  R57 源码摘录逐行核对 → R58 收敛裁定。每轮结果先过 Step 3.5 深度复审再进下一轮。
- 实证证据（本地）：cute int_tuple.hpp:196（depth）、copy_sm90_desc.hpp:83
  （set_barrier_transaction_bytes=mbarrier.arrive.expect_tx）、copy_sm100_tma.hpp:45
  （Sm100MmaPeerBitMask=0xFEFFFFFF）、cutlass/arch/barrier.h:907（umma_arrive_2x1SM_sm0）、
  copy_atom.hpp:495（make_tiled_copy）、layout.hpp:1798/1324/206（tile_to_shape/left_inverse/
  with_shape）、copy_traits.hpp:66（UniversalCopy）、copy_traits_sm75.hpp:42（LDSM_N）、
  mma_sm75.hpp:52（SM75_16x8x8）、PTX 9-instruction-set 12237-12282（fence.proxy 方向）、
  PTX 13510-13512（mbarrier 远端支持矩阵）。
- 确证发现 7 组（不改，记录）：①reed-02 depth=2 vs cute depth()=3（HIGH，L1/L2 口径冲突）；
  ②reed-isa-02 "predication=分支预测"术语错误+流水线清空归因（MEDIUM-HIGH）；
  ③reed-04 路径2 "L1→smem" 通路（MEDIUM）；④reed-11 求和上限 off-by-one（LOW-MEDIUM）；
  ⑤reed-mem-01 barrier.h:912 行号漂移（LOW）；⑥LOW 措辞 6 处；⑦未验证 8 组。
- 撤回 5 条误报：zhuxi-05 "缺 arrive"（expect_tx 自带 arrive 实证）、R55 脚本 3 条 MISMATCH
  （舍入带内）、及 R49 初判 1 处降级。既有 medium 4 条维持（reed-02 product logical 行、
  zhuxi-03 一一映射/单射、zhuxi-04 tiled_product 版本断言、zhuxi-05 8-bit/≤256）。
- 章节裁定：22 章正确、10 章带确证观察、0 blocking；维持"概念正确性中高、参考级"判断。
- 数值复算：24 项独立复算全对（含 frankshi 2031616/2097152、reed-10 bit21 表 7 行、
  blackwell 带宽表、fp8 dp(S) 全链、-7 的 20-bit 补码）。
- 本轮无书稿改动，未触发任何 PDF/RTD 闸门。台账：book/.tmp/cute-zhihu-audit/ledger.md。

## R59（2026-10-25）：cute-zhihu 问题条目复审 + 确证修复（11 文件 18 行）

- 复审推翻 3 条原判：reed-02 depth=2（HIGH 误报，cute depth leaf=0，书中正确）、
  reed-02 product logical 行（medium 误报）、reed-04 路径2（cp.async.cg/.ca 真实硬件区分）。
- 修复 13 处：isa-02 predication×5、reed-11 求和上限+TileB^T、isa-04 偏置指数×2+denormal、
  isa-05 negate 条件、isa-03 local memory 澄清、mem-01 去行号、zhuxi-04 tiled_product 证伪改写、
  reed-01 反向访问、reed-03 Int<M> 补{}、reed-10 标题[2]、声明页同步"最小订正"口径。
- 教训：①LaTeX 转义形态（Int$<$M$>$）会让字面 grep 漏检，撤回结论前必须按 tex 形态复核；
  ②Review Plan 纠出 2 条漏检+1 条同错第二处（isa-04 L25）+1 条改法新错（zhuxi-04"作者写作时"）；
  ③CUTLASS 升 v4.8.1 后全锚点复核，barrier.h 907→966 证明"引用文件路径不写行号"是对的。
- 验收：PDF 248 页/0 error/Overfull 6 同基线；pdftotext 旧句 0 新句全中；RTD 全量+verify 全 OK；
  visual_check 12 页全 OK。未 commit。

## R54–R58（2026-10-25）：第三轮五视角深度审核（每轮轮末独立复审）

- 范围：主 book 38 章 + 附录 5（基线 279e364）；只审不改；主 agent 逐项核验，未派 sub agent。
- 五轮视角与结论：R54 交叉引用指称（815 处 \ref{ch:NN} + 全部 N.M[.K] 节号 + 39 处命名指称 + 6 条公式指称抽样）——节号 0 越界、命名 39/39 匹配、抽样全在位；R55 数值复算（41 处比值断言带舍入带判定）——40 精确吻合，1 处（ch26c 2.71x）带内不动且四源一致；R56 图注-正文-表三角（括号平衡解析 caption，孤立数字 19 逐条判读 + 同量词冲突扫描）——零矛盾，2 冲突候选经操作数复算自辩（ch26c 580/88=6.6 倍、ch00 100KB/32KB=3 blocks）；R57 测试声明（31 个 tests/*.cu 对照书内容差/case/PASS 数字）——容差三档逐字一致、case 字符串与形状全对、编译级通过；本机无 CUDA 设备，PASS 误差数字标"运行期未复测"；R58 高风险词（scan_risky_claims ~1100 候选，聚焦 L4-figure 10 + L2-wait 21 全量判读）——30/31 与 PTX/源码一致，1 LOW 观察（ch29 考据段 wait<2S-1> 简写，源码实为 2kS-1/2kS-2 双常量，指路段合理简写）。
- 方法论记录：Step 3.5 规则 5 本轮四批拦截脚本假阳性（ch19b stem 抢 ch19 计数键、grep 反斜杠转义、ulp 精度过严、caption 正则吞嵌套表）；未有一条流入结论。
- 章节裁定：全部维持"正确"（38 章 + 5 附录）；1 LOW 观察不改；既有 medium 6 项原样保留。
- 本轮无书稿改动，未触发任何 PDF/RTD 闸门。
