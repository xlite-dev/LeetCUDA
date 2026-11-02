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

## R60（2026-10-25）：全量逐章证据链审核 + 63 项修复 + 三 PDF/RTD 全验收

- 范围：主 book 41 个 chapter input（含 ch19b、ch26b、ch26c）+ 8 个白皮书子文件 + 附录 A–E，共 54 个 TeX、40487 行；基线 `1932ac2`，开工工作树 clean。`ch19b`/wp 只通读作术语基线，不做逐条技术审核。
- 与历史轮次的差别：不是看 diff 或旧 ledger，而是把审核单位从“章节”改成“claim 表面”。逐章全文覆盖正文/公式、TikZ 节点、图注、表/表注、面试问答/小结/标题、listing、tests/bench、CLI/backend、附录 permalink、RTD 三套词典。旧 PASS 只用于防误报，不作为本轮证明。
- 机械校验：226 个外部 listing（178 command + 48 environment）检查 `.cu/.cuh` 路径与单段/多段 linerange，无效 0；53 条附录 permalink 按各自 commit 逐条核验，发现 2 条 pinned 快照无效并修复；重算 AI/ridge、ceil、SF 字节、寄存器池、加速比；发现 ch04 bench 启动参数只传 `N` 而非 `S*N`；核对 861d75e 的 CLI choices 与 backend assert，确认 Q/K per-channel 是“CLI 可传、CUDA 后端拒绝”的假可运行档。
- 候选与裁决：67 条候选；撤回 4 条（C-20 TMA 完成时序图省略、C-32 kBlockSwizzle 误读为组数、C-56 WHT 归一化折叠、C-59 旧 bug 复现未写 checkout）；保留并修复 63 条（HIGH 3：ch04 bench、ch13/14/17 expect-tx、ch32 P2 图内符号；MEDIUM 33；LOW 27）。C-14 经 PTX tx-count 复核后收窄：多报残留正值挂死；少报若命中 0 提前放行、跨到负值也挂死；ch26b 单 bulk 少 1 字节原句正确，撤回该处误判。
- 主要修复类型：图内符号/图注与正文冲突；LSE“可逆打包”与 Split-D“LSE 合并”语义；bias 与 RMSNorm 的因果；128-bit/sector 与 request/sector 口径；全局 AI 漏 C；PAD 无法消除固定 k 冲突；expect-tx/phase 协议；ceil 与 per-block SF 数；fp16/bf16 基线；e4m3/e2m1 数值语义；level-1 不能保证所有组不下溢；13824→8704 落点；P2 图内符号；sm_120f 仅 SM12x family；两条 permalink；CLI per-channel。
- 修复与词典：36 个 TeX/测试文件 + 2 个 RTD 词典 + 3 个重建 PDF，共 41 文件（213 insertions / 185 deletions）。改图内中文后 RTD 图集词典少 8 条，用 `python -m tools.translate_figures --limit 8` 补齐；改 appC 的 sm_120f 文案后 code-en 词典漂移 1 条，同步 key/value；强制 TikZ 重建后全部命中。
- 验收：主 book 589 页 / 0 error / 0 missing / Overfull 16；cute-zhihu 248 页 / 0 error / 0 missing / Overfull 6；colfax 204 页 / 0 error / 0 missing / Overfull 0（原环境缺 biblatex/logreq，装用户 TeX 树并用 TL2023 临时 biber）。RTD `./build.sh --force-tikz` 全绿：英文图集 1883 条覆盖 1792/1792、英文图 196/196、code 771/771、math 121/121；`visual_check` 93 页全部 `[OK]`，需处理 0。未运行 GPU 测试/bench；ch04 新 bench 数据标未复测。
- 方法论结论：多查出的原因主要是覆盖面、证据标准和反向自辩三者叠加，不是书稿突然变差。63 项中 HIGH 仅 3 项，27 项是 LOW 表述/图表/测试覆盖问题；同时有 4 项误报，说明“查得多”必须配套 confidence/severity 分层，不能把候选数当错误数。该模式已沉淀到 `references/full-book-claim-audit-playbook.md` 和 SKILL §2.2。
- 本轮 skill 更新本身无书稿改动，未额外触发 PDF/RTD；上述验收对应本轮书稿修复。未 commit。

## R61（2026-10-26）：全量逐章再审视（只读，无书稿改动）

- 基线 `e749510`（R60 修复后），开工工作树 clean。方法：主 agent 逐章通读/定点核查，不派 subagent；先读 R60 全量 diff 做回归面挑选，再按章复算数字与推导。
- 覆盖：主 book 41 个 chapter input（含 ch19b）+ 附录 B 数值交叉复算；ch20–ch26c、ch29–ch32、ch35–ch37 以性能/寄存器/误差节定点核查，ch19b 与 wp/ 只作术语基线。
- 候选 7 条 → 保留 6 条（MEDIUM 1 / LOW 5），撤回 1 条。无 blocking 级错误。
  - MEDIUM：ch03 图 3-4 的 TikZ 节点与图注、以及 Q3 仍写“指令瓶颈从访存侧转移到计算侧 / HBM 直读收益更大”，与 R60 修正后的正文（“占比本身不能判定瓶颈转移”“工作集大于 L2 不单独保证收益”）同页冲突——R60 的 L4 传播缺口。
  - LOW：ch04 Q2 仍以“实测”口吻给 21%，而表 4-bench 已标“未复测/历史记录”；ch28 性能节“±2%”vs ch33 同测量 −2.7%；ch31 把 ue8m0 的**相对步长 100%** 与 ue4m3 的**RN 误差上界 6.25%** 并置（两数各自对，同口径应为 12.5%/100% 或 6.25%/50%）；appB 转录漂移（3.144 vs 正文 3.145、2.71× vs 2.72×）；ch07 硬编码“第 7 节/5.2 节”缺“本章”前缀。
  - 撤回：ch16 K/V 加载映射的线程数疑点（源码 L147-176 复核为 `tid/4` 定行，与 B_c=64 自洽）。
- 复算锚点（全部自洽）：ch01 六算子 AI 与测试输出；ch02 98304×(7,3)=691067/297628；ch09 3.145 ms→5.46 T、683 GB/s；ch11 76.55/28.19；ch19 384D 字节与 AI 42.7→64；ch27/28/31/34/36 的 2^{-(m+1)}、11.02/10.06/10.17/14.38、σ_ε 表与 s²=48；ch29/30/32/33 的 4BHN²D 折算与加速比。
- 本轮**无书稿改动**，未触发任何 PDF / RTD 闸门（未构建、未跑 visual_check）。台账：`book/.tmp/math-audit-r61/ledger.md`。

### R61 定点修复（同日）：P-1

- 改 `chapters/ch03-vectorize-atomic.tex` 三处（图 3-4 TikZ 节点、图注、Q3），把 R60 只改到正文的结论同步到图与问答：由“指令瓶颈从访存侧转移到计算侧”改为“占比变化不等于瓶颈转移，本章实测仍受 L2/互联限制”；Q3 的“HBM 直读场景更大”改为“须按层级实测，不能直接外推”。
- 图内中文换键 → `i18n/figures-en.json` 删旧键补新键（单行 diff，占位符 ⟦0⟧–⟦7⟧ 对齐，value 的反斜杠-n 换行标记与 key 同型）；`build.sh` 复跑后英文图集 195/195、verify 全绿。
- 验收：book.pdf 589 页 / 0 error / 0 missing char / Overfull 16（=基线）/ lint 三面 0 fail；pdftotext 三处新句落地；第 55 页 160dpi 目检图内文本不压图元；RTD `build.sh` 全项 OK；`visual_check` 93 页需处理 0。
- P-2…P-6（ch04 21% 口径、ch28/ch33 ±2% vs −2.7%、ch31 SF 步长/误差上界并置、appB 两处转录漂移、ch07 节号缺“本章”）经用户确认本轮不改。

## R62（2026-10-26）：机械面全量一次（收尾轮，含 5 处修复）

- 手段：脚本化复核四张机械面（脚本 `book/.tmp/r62/check_mech.py`），不再靠通读：227 处 listing 的路径/linerange/title 行号、53 条附录 permalink（按各自 commit 用 `git cat-file`+`git show` 核行数）、9 处 tests 文件名引用、appB 74 个三位以上有效数字回指正文；配合仓库既有 `scripts/verify_anchors.py`（33 章 SHA 冻结 + 首末行文本比对）与 RTD `convert.verify` 三套词典。
- 结果：4 条真缺陷（全为「行号引用/代码块截断」，R60 同一类），已修；机械面其余全绿。
  - ch26b:492 title `L3940-3967` 与 `linerange={3934-3967}` 差 6 行，且起始行截在 `{` 之后 → 改为 `3932-3967`（含注释与 `{`，括号配平），title 同步 `L3932-3967`。
  - ch26c:150 title `L736-756` 与 `linerange={735-756}` 差 1 行，起始行截在 `using SmemLayoutV =` 的续行 → 改为 `730-756`，title 同步 `L730-756`。
  - ch37:550 `linerange={958-964}` 末行是 producer 分支的 `if (tid < kProducerThreads) {`（孤立开括号，且在「修复只有一行」的语义之外）→ 改为 `958-963`，title 同步。
  - ch25:850 三段式第三段 `3462-3482` 末行是 `for (...; linear += blockDim.x) {` 循环头，循环体被切掉 → 改为 `3462-3486`，title 与正文 ch25:864 的 `L3462--3482` 一并同步。
- 假阳性清单（已逐条判读，不改）：`_template.tex` 2 条（模板文件，不进 book.pdf）；ch13:332 只列 kernel 签名（title 明示）；ch13:405 `======` 分隔注释；ch14:313 `#endif`；ch17:529 切在 for 头但其后紧跟续列（阶段一/3b 后半拆两段）；ch35:311 `} else {` 是 producer/consumer 分支切换点；appB 11 处数字未命中里 9 处是显示取整（0.042/0.785/0.144/0.0190/0.54/4.19）或跨章归属（appB:219 的 1.64× 归第 33 章）。
- 顺带：extend ch26b 段引入了一行新中文注释 → `_static/code-en.json` 补 1 条英译，`convert.verify` 代码块词典回到 772/772。
- 验收：book.pdf 589 页 / 0 error / 0 missing char / Overfull 16（=基线）/ lint 三面 0 fail；`verify_anchors` 33 章 ALL GREEN；RTD `build.sh` 全项 OK（英文图 195/195、代码块 772/772、公式 121/121）；`visual_check` 93 页需处理 0。
- 未改（用户已定）：P-2 ch04 21% 口径、P-3 ch28 vs ch33 ±2%、P-4 ch31 SF 步长/误差上界、P-5 appB 转录漂移（3.144 vs 3.145、2.71 vs 2.72，本轮机械面再次确认为真）、P-6 ch07 节号缺「本章」。

### 收尾结论

机械面与通读面前后两轮（R61 通读 + R62 机械）合计只产出 1 MEDIUM（已修）+ 9 LOW，其中 4 条 LOW 是行号/代码块截断（已修）。**正确性自审到此收账**：后续只按「改哪章审哪章」做定点复核，全书通读与全书机械扫描都不再例行；残余风险已转为性能数字复测（ch04/ch13/ch19/ch32）与 6 条已知 medium，均需 GPU 或补证据，不属审稿可解。

## R63（2026-10-27）：ch00–19b + 附录严重问题定点修复

- 范围：静态审核确认的 4 项问题，按用户要求全部修复；只改书稿、RTD 词典与验收记录，不改冻结 CUDA 源码。
- WGMMA descriptor：ch13/ch14/ch23 统一改为「非 1024B 边界必须显式写非零 `base_offset`；当前 `make_smem_desc` 置 0 并配合 1024B 对齐」，删除 `start_address` 自动补偿和当前 WGMMA 实现只需 128B 的说法。ch14 旧「核校通过」改为带 PTX 证据的勘误。
- ch19 cp.async：修正 V 预取与 smem 两处资源/流水断言。关键反向复核结论：QK/V 不是两条独立等待队列，`cp.async.wait_group N` 只保留最近 N 组并要求所有更早组完成，所以即使 `Sv>=3`，Phase 0 的 V 也不能横跨后续 QK/softmax；`Sv>=3` 的真实 V 重叠只在 Phase 3 内。图 19-4 灰底改为逻辑间隔，默认 `Sk=Sv=2` 明确无跨计算重叠，同时删除对应性能机制归因。smem 改为只由 `Sk/Sv` 决定，`C` 只影响迭代与 O 寄存器。
- appB：`1.4→0.3ms` 从 fused 单 launch / `N=8192` 改为 ΔS WMMA 替换 torch 链 / `N=16384`；为消除新增表格 Overfull，将不可断行的 `bmm+transpose+cast` 改为可断行列举。
- 同源传播：`ch23-cute-swizzle-tma.tex` 一并修正；图 19-4 的 4 个 `figures-en.json` 词条同步换 key/value，未改 listing，代码块与公式词典无需变更。
- 验证：主 `book.pdf` 589 页 / 0 error / 0 missing char / Overfull 16（基线一致），跨文件 lint 0 fail；`verify_anchors.py` 33 章 ALL GREEN。RTD 全量 `build.sh` 全项 OK（英文图 195/195、图词典 1922、代码块 772/772、公式 121/121）；`visual_check` 93 页 MathJax/残迹/console/资源失败/page_overflow 均 0。图 19-4 以 180dpi 渲染并放大检查，无文字或箭头重叠。
- 未验证：未重跑性能 benchmark；历史性能数字只修正归属，不视为复测。ch13 不做 sm_90a 运行验收。cute-zhihu 源未改，不触发其独立 PDF 构建；RTD 已包含该集合的全量核对。

## R64（2026-10-27）：ch20–37 技术问题与 CuTe 白皮书翻译修复

- 范围：按用户要求修复审核报告一、二中的全部问题；主 agent 完成，不使用 subagent，不改 CUDA 源码。
- 主书 6 项：ch20 将块数比例限定为连续单位 stride 且整除，一般 rest 由 complement 决定；主机端 static_assert 验证 `logical_divide(10:1,4:1)` 为 shape `(4,3)`、size `12`。ch26 统一 PV 单次视图 `[64,16]`、完整 O chunk `[64,64]` 与 chunk 内 4 个 N-tile。ch27 把相对误差上界限定到规格化域，补 E2M1 的 0.75/0.25 反例。ch34 改正 E4M3 仅 `E=1111,M=111` 为 NaN。ch37 将 `BK=64` 归为教学实现选择，官方 SM120 CuTe DSL 的 BK=128/256 作为反证。ch26b/ch35/ch37 按 CUDA non-exited 线程语义重写 NamedBarrier 理由，保留实现。
- 白皮书：修复 `up to N`、`typically means`、`enables the weak congruence condition` 三处误译；补回结论漏句。新增 `wp7-references.tex`，按官方英文 PDF 实际列出的 29 条及原顺序补齐参考文献（不是把含 35 条的原始 .bib 全量照搬）。
- RTD 图词典同步替换 7 个 key/value，覆盖率恢复 1791/1791；代码块与公式词典无需变更。
- 验收自审：ch20 初版曾把 ceil(size ratio) 推广到任意 stride，复核后收紧。ch36 UE4M3 的 134.3→128 候选经码点复算撤回：该 binade 步长为 16，邻点是 128/144，原书正确。
- RTD 特殊问题：`enumerate` 带 label 选项被 Pandoc/MyST 转成字面 `::: enumerate`，文件级 verify 未捕获；直接检查 HTML 后改为显式 `[1]`–`[29]` 编号段落。最终 HTML 编号序列完整且无指令残迹。这提醒我们：新增列表结构除全量构建外还要检查实际 HTML 呈现。
- 验证：主 `book.pdf` 591 页 / 0 error / 0 undefined / 缺字 0 / Overfull 15；4 张修改图 180dpi 渲染并放大无重叠。RTD 全量全项 OK（英文图 195/195、图词典 1922、code 772/772、math 121/121）；visual_check 93 页 MathJax/残迹/console/资源失败/page_overflow 均 0；anchors 33 章 ALL GREEN。
- 未验证：未重跑性能 benchmark，未做 sm_90a 运行验收；本轮为书稿与翻译修复。

## R65（2026-10-28 起 / 2026-10-29 补完）：主 book 第四轮全量正确性复核（用户要求「没问题就收账」）

- 触发：用户在 R63/R64 之后要求对主 book 再做一轮全量阅读审核，用于决定是否停止例行全书审核。主 agent 独立完成（不使用 subagent）。
- 范围：`book.tex` 实际 input 的 41 个 chapter（含 ch19b 白皮书，仅作术语基线）+ 附录 A–E；`cute-zhihu` 不在本轮范围。
- 方法（含一次返工）：先跑机械面（`lint_book.py` citations/tables/figures、`verify_anchors.py`、listing 路径与 linerange、permalink 按各自 commit 校验、tests 引用、appB 数字回指），再按章做 L0–L4。**首轮 ch00–ch15 逐章全文通读、ch16–ch37 与附录改用 claim 表面抽取，被用户当场质问「为什么不通读全文」；随后补读 ch16–ch37 + 附录 A–E 全文（分 20 余次 read_file），并据此修正台账与结论。** 教训已固化在 `memory/feedback_leetcuda_full_read.md`：说「全量」就必须逐章 `read_file` 读全文，不得用抽取代替，也不得只靠「报告里写明覆盖范围」免责。
- 结果：机械面零 FAIL；逐章复算零 HIGH/MEDIUM，**新增 3 项 LOW**：① ch25:166 用 D=64 的 cosize（相加仅 32 KB）去解释 80 KB（实为 D=128 + Sk=2 的 40960 half）；② ch34:224 图注「$\lceil B_N/128\rceil$ … $B_N{=}256$ 时为 3 个」与同句算术（=2）及 §34.4.1「横跨两块」矛盾；③ ch37:995 的 186.5/314.6 与 ch35 表 186.8/315.1 属转录漂移。**三项 LOW 于 2026-10-29 按用户指示全部回修**（ch25 保留 D=64 实测并补 case C 的 D=128+K 两级算式；ch34 改「2 个，n 起始未按 128 对齐时最多 3 个」；ch37 统一 186.8/315.1）。补读段复算自洽的例子：ch16 AI=B_r=128 与 45 KB smem、ch17 158.7→190.4 +20%、ch18 129 arrive 与池 376≤512、ch19 AI=42.7 与 33792≤65536、ch20 cosize/compose/zd 双射、ch22 512 守恒与 acc_O=256、ch23 三档周期与 8×32 退化、ch24 48 KiB 与两套累加 128 regs、ch26 EURepeat 与 256>255、ch26b 2048=18×110+68、ch26c 16(S_K+S_V) KB 与 spill 表、ch28 192/96 MiB 与 π⁻¹、ch29 64512 与 π/π⁻¹ 双向、ch30 D/(2N_w) 与 D·M_w/4096、ch31 4.5 bit 与 π 32 列表、ch32 −11.3923/−2.585、ch33 全表比值与 1.22×、ch34 σ_ε 与 4σ 指纹、ch35 441.9/443.7 与 4.83→4.56 GB、ch36 0.145833→11.02%→14.38%、ch37 9216 B 与 +31%。
- 撤回/维持：R62 已列的 P-5 appB 转录漂移（3.144 vs 3.145、2.71 vs 2.72）本轮再次复算——3.1445 属 tie 在舍入带内，2.7155 应落 2.72；两项均按用户 R62「不改」裁定维持，不计为新增缺陷。
- 已知残余（不属审稿可解）：6 条 medium（ch26c VCPB STSM 计数、ch27「四个误差源」只列三项且内部仿真未归档、ch33 smooth-v +1.0% 机制、ch36 3.24%/10.17% 合成口径、ch17 Sk 饱和、ch29 wait 考据）仍需 GPU 复测或补证据；ch13/ch19/ch32 的性能数字未复测。
- 验收：**回修后跑满三层闸门**——`book/build.sh` 591 页 / 0 error / 0 undefined / 缺字 0 / Overfull 15（与 R64 基线一致）；`lint_book.py` citations 0 fail（1 白名单 WARN）、tables 0、figures 0；RTD 全量 `./build.sh`（含 TikZ，未用 `--skip-tikz`）+ `convert.verify` 全项 OK + `tools.visual_check` 93 页 / 需处理 0；三页新文本在 HTML 中逐条命中。`CHECKLOG.md` 共 6 行记录（2 行 P-5 复核 + 汇总 + ch25 + ch34 + ch37 + 回修验收），逐章台账在 `book/.tmp/r65/ledger-shardA.md`。
- 结论：主 book 结构性正确性高，可作为发布候选；**建议停止例行全书通读**，后续改为「改哪章审哪章」的定点复核 + 发布前三层构建验收。

## R66（2026-11-01）：ch03 histogram 锚点定点复核与 2 处回修

- 触发：用户 @ `base.cuh:381`（histogram kernel）要求评估主 book 当前正确性，随后要求复审并修复发现的问题。主 agent 独立完成，不使用 subagent。
- 范围：`chapters/ch03-vectorize-atomic.tex` 全章 L0–L4 定点复核（histogram 所在章），非全书通读；`cute-zhihu` 不在范围。
- 发现 2 项（1 B + 1 C），经反向自辩确认真问题后回修：
  - **F1（B，L3 资源/类型计数）**：ch03:152–154 向量化公式把 grid 的 block 数误标为「线程」——「标量版 `N/T` 线程、vec4 版 `N/(4T)` 线程 × 2 条访存指令」。证据：`base.cuh` L315–316（relu 标量版 Block 256 → 共 N 线程）、L326–327（relu_vec4 Block 64 → 共 N/4 线程）；同章 L190/L478 明确 `N/T` 为 block 数；且与 L156「warp 级 `N/32`」（隐含标量版 N 线程）矛盾。回修为「`N` 线程 / `N/4` 线程」，与下文「线程数降 1/4」「warp 级 N/32→N/128」自洽。
  - **F2（C，L4 表达一致性）**：ch03:402「histogram 比 elementwise 慢两个数量级（26.7 vs 2654 GB/s，0.63 ms vs 0.019 ms）」把时间口径（0.63/0.019≈33×）与「两个数量级」（100×）混用。回修为「慢约 33 倍（0.63 vs 0.019 ms）、有效带宽低约两个数量级（26.7 vs 2654 GB/s）」。同章 L187/L222/L228 的「低两个数量级」均修饰带宽（99×），本就正确，未动。
- 同源传播：F1 公式改动使 RTD `_static/math-en.js` 旧公式 key 成死条目（convert.verify 报「公式 121 命中 120 + 死条目 1」）；已同步替换该条 key/value（`\frac{N}{T}`→`N`、`\frac{N}{4T}`→`\frac{N}{4}`），恢复 121/121。未改图/TikZ，figures-en/code-en 无需变更。
- 验收（三层闸门全绿）：主 `book/build.sh` 591 页 / 0 error / 0 undefined / 缺字 0 / Overfull 15（与 R65 基线一致）；`lint_book.py` citations 0 fail（1 白名单 WARN）、tables 0、figures 0；RTD 全量 `./build.sh` + `convert.verify` 全项 OK（英文图 195/195、图词典 1922、代码块 772/772、公式 121/121）；`visual_check` 93 页 / 需处理 0。
- 未验证：未重跑性能 benchmark；ch03 实测数字（26.7 GB/s、0.63 ms、2.6 ns/op）沿用既有口径，不视为复测。cute-zhihu 源未改，不触发其独立 PDF 构建。
- 结论：主 book 维持 R65「发布候选」判定；本轮 2 项 B/C 已修，未推翻收账结论，但印证「改哪章审哪章」定点复核仍有低级别产出。

## R67（2026-11-01）：用户重开主 book 全量审核，17 个根因回修

- 触发与范围：用户明确要求继续全量审核，不能沿用 R65「收账」跳过。主 agent 独立完成，不使用 subagent。按当前 `book.tex` 实际输入全文阅读 54 个 `.tex`：主章 40（ch00–ch37 + ch26b/ch26c）、白皮书 9（ch19b+wp0–wp7，只作翻译基线）、附录 5。图、图注、表、Q&A、listing 与测试说明均纳入阅读。
- 结果：确认并回修 17 个根因，涉及 ch01/ch06/ch15/ch17/ch18/ch20/ch22/ch26/ch26c/ch30/ch31/ch34/ch35/appB。主要是数学量词与维度、API 边界、图注/Q&A 与正文/表格脱节：LN/RMSNorm bias 前提、相近大数 cancellation、FA2 仍保留 online rescale、F32Acc 的 P 在 PV 前转 half、setmaxnreg 暂时不足只等待、CuTe 单调性看进位增量而非 stride 排序、MMA_N 补 ThrN、D=448 stages 例外、大 D 下 O(Bc) 占比方向、NVFP4 K 方向与 π、误差分布前提、s4 扫描结论和 appB min-of-N 口径。
- 同源传播：ch15 图注与 ch18 正文、ch06/ch20/ch26c Q&A、ch35 与 appB 同步；ch31 图内 `π^{-1}→π` 使英文图 key 的数学占位符由两个减为一个，`figures-en.json` 已同步。未改 CUDA 源码、`book.tex` 结构、listing 或性能原始数字。
- 撤回：TF32 随机误差候选经独立复算为 0.0030070，原书正确；128B 单事务与 32B sector 是本书并存的两种口径；Part V/VI 结构候选由旧目录快照造成，重读当前 `book.tex` 后撤回且未改主文件。
- 验收：anchors 与 lint 全绿，permalinks 53 个、测试引用和 appB 回指通过；主 PDF 591 页 / 0 error / 0 undefined / 缺字 0 / Overfull 15；RTD 全量 build（未跳 TikZ）2518/2518 source blocks、英文图 1791/1791，其余全项 OK；浏览器 visual_check 93/93 页需处理 0。逐章与撤回台账在 `book/.tmp/r67/`。
- 未验证：未重跑 GPU 性能 benchmark；ch13/sm_90a 仍仅静态核验。cute-zhihu 源未改，不触发独立 PDF；RTD 全量已覆盖其页面。

## R68（2026-11-01）：快速全量自审，零新增确证错误

- 范围：按用户要求只做快速全量，不再逐字重读 54 个输入。主 agent 对当前 chapters+appendices 跑机械检查和高风险扫描，并逐 hunk 复审 R67 当前技术 diff；白皮书仍只作翻译基线。
- 覆盖与结果：扫描 1115 条候选，人工复核其中 311 条高优先级命中（数理逻辑/图注/wait 106 条，全称/双射/整除/零拷贝等 205 条）。lint 0 fail、anchors 33 章全绿、permalinks 53 与 tests refs 9 均为 FAIL 0；listing/appB 告警均为既有脚本口径，无新增失败。新增确证书稿错误 0，书稿回修 0。
- R67 公式反证：对 ch20 单调性充要条件穷举 rank 1–3、shape 1–4、stride 0–8 共 47988 个 flat 非负 stride 布局，单调性与条件全部一致。脚本首轮 5820 个 mismatch 是把 size-1 维的不存在进位也纳入条件；按原文“实际发生的进位”修正脚本量词后 mismatch=0，候选撤回，不改书稿。
- 验证边界：本轮无书稿改动，未触发 PDF/RTD 闸门；R67 的 PDF/RTD/visual_check 结果只作既有基线，不冒充本轮新跑。GPU benchmark 与 sm_90a 运行仍未验证。台账在 `book/.tmp/r68/summary.md`。

## R69（2026-11-02）：用户要求再开一轮"全量深度审核"，只审不改，9 项候选落盘

- 触发与范围：用户明确"继续做新一轮全量深度审核，发现问题就记录、不修改"。主 agent 独立完成，不使用 subagent。按当前 `book.tex` 输入**逐文件全文通读 41 章 + 5 附录**（含 TikZ 节点、图注、表、Q&A、listing 标题与测试说明）；ch19b 的 8 个 wp 译文按 skill §1.1 只作翻译基线，本轮未逐段重读（已在报告"未验证边界"声明）。
- 机械层：`verify_anchors.py` 33 章全绿；`lint_book.py` 三类 0 fail；appD 53 条 permalink 与 appB 回指 OK；appB 派生量扫描 11 个 token 均既有口径。**无结构性回归**。
- 结果：9 项候选 + 3 项 observation，**全部落在 L4 表达一致性/口径层，无 A 级（改变结论）项**：
  - C-03（B 级·同章自相矛盾）ch02:272「bench 显示这就是真瓶颈」vs ch02:469–472「由总时长反推，不能单独当因果证据，仍需减少 G 的 A/B」——前句说过头，appB 已按后句口径写。
  - C-05 ch12:379–380「Tn ≤ ⌈R/Tm⌉ 时收益趋零（4096³ 仅 +0.3%）」——4096³ 的 Tn=32 > ⌈R/Tm⌉=4，判据与例子互斥；真实机制是 96 MB L2 容量（37.7/24.1 MB 都装得下 vs 116/24.1 MB）。
  - C-04 ch03:454「慢百倍（99 倍差）」把带宽比当时间比（实测时间 33×、带宽 99×），R66 只回修了 §perf 侧。
  - C-02 ch01:403/455–458「Part I 前三章」——Part I 含 ch00，故前三章是 ch00/01/02，但三条手段落在 ch03/ch03/ch02。
  - C-07 appE:58「A. Milakov」应为「M. Milakov」（arXiv:1805.02867 作者 Maxim），与 ch04:650 自相矛盾。
  - C-08 ch35:691 扫描表 SKIP 注记 `---(123)`，按同列公式应为 120（不影响 SKIP 结论），需核打印源。
  - C-09 ch37:750 图注「FP4 只比 FP8 快 6~40%」——4096³ 默认档 +5.0%、8192³ 默认档 −0.6%（表口径 −1.5%），低端不成立。
  - C-06（observation）ch15:216 的 209.5 TFLOPS（F32Acc）与 ch15/16/17 的 ridge≈300（F16Acc≈419 T）是两个口径，算术自洽但全书未点明 2× 关系。
  - O-1 ch29 图 29-reorg-free 的 π(s) 公式符号绑定未声明（按 t=lane&3 代入得 π(3)=6，与正文 π(3)=9 矛盾；按 r=s&3、t=s>>2 则一致）；O-2 ch26c:49 与 appD:55/anchors 的 642–1307 vs 643–1306 边界约定差异；O-3 ch27 与 ch30 的 Bc 分档未点明是两个家族。
- 反向自辩后**撤回 7 项**（ch04 case A 的 N 语义、ch00 slack 2048、ch15 ridge 冲突、ch36 逐格误差表两格矩、ch30 交叉点 vs ch19、ch35 O staging 取等、ch31 π 闭式），全部经二次复算/回章核对后判为原书正确——记录在报告第五节，供下轮不要重复怀疑。
- 验证边界：**本轮无书稿改动**，未触发 PDF/RTD 闸门；GPU 未复测（所有"实测"数字只做量纲与横向一致性核对）；sm_90a 仍仅静态核验。报告与复算记录在 `book/.tmp/r69/report.md`。

## R70（2026-11-02）：R69 自审结论再复核，9 个根因回修并通过三层验收

- 触发与范围：用户要求继续深度审核 R69 的「自审修订结果」，对确有问题的项进行修正。主 agent 独立复核，不使用 subagent；逐项回读现行正文、同章后文、冻结源码、全书同源表格与 PDF/网页产物，不直接照搬 R69 的建议修法。
- 复核中修正的两条审核方法：① C-02 不改成含混的「前四章」，而在 ch01 正文/TikZ/图注直接标明 ch02/ch03；② C-06 不把推测的「F16Acc=2×F32Acc」写成新断言，只删除全仓无出处、且被 ch24 同书 F32Acc 实测 226.0/242.4 TFLOPS 超越的 209.5「峰值」及其 0.66 ms 换算，保留有实测依据的 166.6 TFLOPS/0.83 ms。
- 回修 9 个根因：ch01 的 Part I 章数口径；ch02 把「真瓶颈」降为「与原子瓶颈一致、归因尚未闭合」；ch03 区分耗时 33× 与带宽 99×；ch12 删除与 4096³ 互斥的 `Tn≤⌈R/Tm⌉` 判据，改用 37.7/24.1 MB 与 116.4/24.1 MB 的 L2 footprint 解释；ch15 删除 209.5 峰值；appE 改 `M. Milakov`；ch35 按 traits 公式把 `(123)` 改为 `(120)`；ch37 图注改为默认档 +5%/−1%、宽 tile +35%～40%；ch29 将 π 改为只含 s 的等价闭式，消除与同图 `t=lane&3` 的符号冲突。π 的 16 点像与逆置换已穷举核对。
- 边界与传播：O-2 的行区间边界约定和 O-3 的 Bc 家族限定不改。两处 TikZ 文案已同步 `figures-en.json`；在线翻译遇 429 后，两条新掩码 key 由主 agent 按当前 TikZ 源手工迁移，覆盖率恢复 1791/1791。未改冻结 CUDA 源码、`book.tex` 结构或 cute-zhihu 源。
- 验收：`git diff --check` 通过；anchors 33 章 ALL GREEN；lint citations/tables/figures 均 0 fail（1 条既有白名单 WARN）；permalink 53 与 tests refs 9 均 FAIL 0，appB 11 个提示仍为既有派生量口径。主 PDF **591 页 / 0 error / 0 undefined / 缺字 0 / Overfull 15**，`pdftotext` 逐项命中新文本。RTD 全量 build（未跳 TikZ）第二轮全项 OK：英文图 195/195、代码 772/772、公式 121/121；`visual_check` **93 页 / 需处理 0**，ch01/ch29 改图另作单图截图复核，无文字或公式越框。
- 未验证：未重跑 GPU 性能 benchmark，所有保留的实测数字不视为本轮复测；sm_90a 仍仅静态核验。逐项证据、修法和验收日志见 `book/.tmp/r69/report.md` 第十节及同目录日志。

## R71（2026-11-02）：逐章全量再审核，7 项确证 + 3 项观察（只审不改）

- 触发：用户要求“再做一轮逐章全量深度审核，尽量找出隐藏的错误”。审核对象为 `3dbd618` + 含 R70 未提交回修的当前工作树；主 agent 独立完成，不使用 subagent。逐文件全文通读 41 章 + 5 附录（正文、TikZ、图注、表、问答、listing 标题），ch19b 的 wp 译文按 skill §1.1 只作基线。
- 机械层：anchors 33 章全绿；lint 三类 0 fail（1 条既有白名单 WARN）；appB 11 个提示仍为既有派生量口径；permalink 53、tests refs 9 均 FAIL 0。
- 7 项确证（全部有同章/跨章/源码一级证据）：
  - **R71-01 ch02 图 fig:2-2** 画了两次 `__syncthreads()`（L409/L424），但 `base.cuh:189` 的 `block_reduce_sum` 只有一次，正文 L342、问答 L578-580、步数式都写 1 次；第二次同步在协议上无对应。
  - **R71-02 ch12 Q5 L1053-1056** 仍用 `Tn>√R` 解释 swizzle 收益，4096³ 的 Tn=32 远超 √110≈10.5 却只 +0.3%；R70 已把正文 L379-382 改成 L2 footprint 口径（37.7/24.1 MB vs 116.4/24.1 MB），Q5 未同步，且 `Tn×ceil(R/Tn)` 在 Tn>R 时也不是常驻集。
  - **R71-03 ch33 L544-545** 称“最差 non-aligned-dense 8K FFPA 快 2.6%”，但同章表 L370 是 341.1 vs 342.3（FFPA 慢 0.35%），16384 慢 1.18%；最大胜差实际是 cross-dense 的 +4.0%。
  - **R71-04 ch35 L896-897** 把 `384×168=64.5K` 套到“三个配置”，但 non-WS launch 是 256 线程（`fp8_gemm.cuh:376-377`、`:825`），256×168=43K，2 CTA=86K>64K 才仍限 1 CTA；只有两个 WS 配置是 384 线程。
  - **R71-05 ch36 L575-579** 的“数值位”列混用口径：bf16 写 8（含隐含位的精度），e4m3 写 3、e2m1 写 1（显式尾数）；同表 RMS 用 p=8/4/2。两种口径都无法同时解释 8/3/1。
  - **R71-06 ch37 L695/L757/L952/L987** 把“M,N,K 全 ≥4096”写成 `$MNK\ge4096$`；按字面乘积 64³ 也进宽档。`fp4_gemm.cuh:1289-1295` 注释为“MNK 全 >=4096”，实现是三维 &&。
  - **R71-07 ch37 L925** 说 FP8“2 级 > 3 级，+7.7%”，但 7.7% 在 ch35 是 `ws128×256/s2` 对 `ws128²/s3` 的 tile+深度混合差；同几何纯深度差为 128² +2.8%、64×128 +8.3%。
- 3 项观察：O-1 ch12:423 与 ch23:218-219 的 Swizzle 位宽字符串与 M=4,S=3 差 7 个“不变”位（源 `common.cuh:263` 同源，可能是不按比例示意）；O-2 ch36 L54-61 把 e2m1 的 e=0 binade 拆成两行，与“同一区间两码点”句不自洽；O-3 appD:46 的 ch26c 只链到 bench，未链 `ffpa_attn.cuh L642-1307` 内核本体，脚注 L643-1306 又差一行。
- 反向自辩剔除：ch13 base_offset、ch23 SW32/64/128 M/S 表、ch36 逐格误差表、ch36 10.17% 增量、ch33 cross 0.76x、ch35 2048³ 宽档反例、ch37 R70 新措辞、ch11 ldmatrix sm_75 语句均确认无错，详见报告第四节。
- 验证边界：**本轮只审不改**，未触发 PDF/RTD/visual_check；未重跑 GPU benchmark；sm_90a 仍仅静态核验。台账与逐条证据在 `book/.tmp/r71/report.md`。

## R72–R76（2026-11-02）：按五个透镜续审五轮，新增 1 项书稿硬错 + 2 项观察 + 1 项源码注释冲突

- 触发：用户在 R71 后要求“再继续审5轮”。R71 已完成 41 章+5 附录全文通读；本轮不重复同一读法，而是五轮不同透镜各自覆盖全书对应表面：R72 数值表/百分比、R73 源码契约/listing、R74 公式边界、R75 资源位宽计数、R76 回归与残余风险。主 agent 独立完成，不使用 subagent。
- R72（数值表）：ch24/25/27/29–33/35/37 全部性能表与 appB 逐格复算，**0 项新硬错**；ch25 的 0.80/0.85/0.95/0.96、ch30 的 1.30/3.76/2.99、ch31 的 1.49–2.11、ch35/37 的 24+18 行全部在显示舍入带内。唯一观察是 ch33:460-461 的 “D512 2.98–3.16x” 漏写 causal 2.57x\* 与 cross 1.90x，属区间口径不完整。
- R73（源码契约）：`check_mech.py` listings 227（含模板）、真实书稿 226 FAIL0，permalinks 53 FAIL0、tests refs 9 FAIL0；`tests/*.cu` 31 个扣除 ch13（sm_120a 自动跳过）与 ch15（无测试）正是附录 C 的 30 个可执行；ch37 的 17 条断言 = 7 主 case + 4 离线×2 + 2 tile。新增 **SRC-1**：冻结源码 `fp4_gemm.cuh:61-64` 写 e2m1 “相邻码点间距=2^{e-1}、相对步长最大25%（区间下沿）”，实际间距是 2^{e-2}、相对步长下沿是 **50%**；书稿 ch36:66-70 的 50%/25% 正确，源码注释两处各差一倍，互相掩盖。源码不改，记录待 CHECKLOG。
- R74（公式边界）：重核 CuTe left/right inverse 定义、`raked_product` 的 `t=4i+⌊j'/8⌋,v=j'%8` 与 `4i+128v+j`、`logical_divide` 非整除 Size=12、ch27/29 的 `P̃=P·v_s·448` 与 `Ṽ=V/v_s` 消去、ch28/31/32 lse 符号、ch36 逐格误差表。**0 项新错误**。其中“P×448 少一个 vs”被 ch29:126-131 的 `P̃` 定义反证并撤回。
- R75（资源计数与位宽）：R71 的 7 项全部复现；新增 **N-1（LOW，高置信）**：ch36:54-61 的 e2m1 表把 `e=0` 的同一 subnormal binade `[0,1)` 拆成 `[0,0.5]`/`[0.5,1]` 两行，和紧随其后的“同一区间内两个码点间距=区间半宽”自相矛盾；若按 Voronoi 应是 `[0,0.25]`/`[0.25,0.75]`，当前两组数都不对。仅动机表，不影响后面的 eq:36-grid。O-1（ch12:423/ch23:218-219 的 Swizzle 位字符串 7 位“不变”间隔与 M=4,S=3 不符）仍保留为观察，可能是不按比例示意。
- R76（回归）：R71 的 7 项全部仍开放；机械层 anchors 33 章 ALL GREEN、lint 三类 FAIL0（1既有WARN）。另发现 **R76-T1 审核工具假 FAIL**：`check_mech.py` 把不在 `book.tex` 里的 `chapters/_template.tex` 纳入，模板示例 listing 无真实路径导致 “listings FAIL1”；排除模板后 226 处 FAIL0。R71 记录中的 listing 数应以此更正，不能把模板告警当书稿错误。
- 结论：R72–R76 **0 项 A 级**，新增书稿硬错仅 N-1（ch36 表头区间），新增观察 N-2/O-1，新增源码侧 SRC-1 与工具侧 R76-T1；R71 的 7 项（ch02 双同步、ch12 Q5、ch33 2.6%、ch35 384×168、ch36 数值位、ch37 MNK、ch37 7.7%）仍是当前最值得修的书稿项。已知残余 medium 不重复升级。
- 验证边界：本轮无书稿改动，未触发 PDF/RTD/visual_check；未重跑 GPU benchmark；sm_90a 仅静态。台账在 `book/.tmp/r72-76/report.md`。

## R77–R86（2026-11-02）：十轮续审，0 项新 A/B 公式错误，但仍有 8 项开放

- 触发：用户明确“再做10轮”，并指出此前“没有错误”的结论不可信。本轮按十个独立证据面执行：机械层、Part I–II 数值、Part III–IV 数值、Part V 数值、源码契约、Part I–II 公式、Part III–IV 公式、Part V 公式、资源/位宽计数、回归。**不把“未找到”写成“无错误”**；主 agent 独立完成。
- R77 机械层：书稿真实 226 处 listing FAIL0（`check_mech.py` 因把 `chapters/_template.tex` 纳入而假报1）；permalink 53、tests refs 9 FAIL0；tests 31 个扣 ch13/ch15=30 可执行；appB 74 数中 11 个既有舍入/派生提示。R76-T1 假阳性已记录。
- R78–R80 数值：ch00–ch14、ch15–ch26c、ch27–ch37 与 appB 的表逐格复算，0 项新硬错；新增 **N-4（ch37:994-996）**：部署口径 B-off E2E 550.5~561.2T 只覆盖 4096³，8192³ 实为 541.1/549.7T（主 kernel 的84~85%），属范围口径不完整，observation。
- R81 源码契约：129/257、TMA descriptor、WGMMA base_offset、setmaxnreg、SF 布局、FP4 wide-tile 三维判据均通过；**SRC-1** 仍开放（`fp4_gemm.cuh:61-64` 的 e2m1 间距 `2^{e-1}`/相对步长25% 实际应为 `2^{e-2}`/50%，书稿正确，源码未改）。
- R82–R84 公式：CuTe inverse/raked/divide、P×448 与 V/vs 消去、FA2 split-Q、split-D、two-level P、e2m1 逐格误差模型全部复核；此前“P×448 少乘 vs”的怀疑被 ch29:126-131 反证并撤回。0 项新公式错误。
- R85 计数/位宽：R71 的 ch02/ch12/ch33/ch35/ch36/ch37 七项全部复现；N-1（ch36 e2m1 表把 `e=0` binade 拆成两行）仍开放。O-1（ch12/ch23 位串 7 位间隔与 M4,S3 不符）仍为观察，不升级为硬错。
- R86 回归：书稿无改动；机械层 anchors/lint PASS，真实 listing 226 FAIL0；R71 七项 + N-1 = **8 项开放**，另有 O-1/O-2/O-3、SRC-1、R76-T1 与 6 项已知 medium。**本轮没有发现新的 A/B 级公式错误，也不宣布零错误**。
- 验证边界：本轮无书稿/源码改动，未触发 PDF/RTD/visual_check；GPU 未复测。台账在 `book/.tmp/r77-86/report.md`；若下一轮继续，应先从 8 项开放项与 6 项 medium 开始，而不是重读已通过的表。

## R87（2026-11-02）：对 R71–R86 错误项逐项深度复核，6 项硬错 + 2 项降级 + 4 项观察

- 触发：用户要求“对这些错误项继续逐项深入审核，避免误判”。本轮不是新全量轮，而是把 R71–R86 的 14 个开放候选逐条回到原文/源码/定义/同章表格，按“证明原句其实成立”的方向复审；主 agent 独立完成，不用 subagent。
- 复核后**降级 2 项**（此前判得过重）：
  - R71-04 ch35 `384×168`：原文的“三个配置分配168”是真的；384×168 只对两个 WS 配置成立，non-WS 是256×168=43K，但两个 CTA 86K>64K，`1 CTA/SM` 结论仍成立。属主语/括号混合，不是资源结论错，降为观察。
  - R71-06 ch37 `$MNK\ge4096$`：GPU 性能语境中 MNK 常作三元组简称，源码注释也写“MNK 全 >=4096”；按字面乘积会与“小形状回落”矛盾，但属记号歧义，不是事实错。降为观察，建议改 `min(M,N,K)`。
- **维持 6 项书稿硬错**：ch02 图 fig:2-2 双 sync（源码/正文/问答只有1次）；ch12 Q5 的 `Tn>√R` 旧判据（4096³ Tn=32 却 +0.3%）；ch33 “最差 nonaligned 8K FFPA快2.6%”与表 341.1/342.3（慢0.35%）不一致（是 L4 表文冲突，不能反推实际性能一定慢）；ch36 “数值位”列8/3/1混用精度与显式尾数（应为8/4/2或7/3/1）；ch36 e2m1 表把 e=0 同一 binade 拆成两行；ch37 FP8 “2级>3级+7.7%”把 ch35 的 tile+深度混合差当纯深度差（同几何纯深度约2.8%/8.3%）。
- **4 项观察**：ch33 D512 区间2.98–3.16x漏 causal2.57x*/cross1.90x；ch12:423/ch23:218-219 位串7位间隔与M4,S3不符（可能示意）；appD ch26c 缺 `ffpa_attn.cuh L642-1307` 内核链接；ch37 B-off 550.5~561.2T/91~93%只对4096³成立。均不升为硬错。
- **源码/工具**：`fp4_gemm.cuh:61-64` 的 e2m1 间距2^{e-1}/25%实际应为2^{e-2}/50%，书稿正确，源码注释待CHECKLOG；`check_mech.py` 把 `_template.tex` 计入造成 listing FAIL1 假阳性，真实226处FAIL0。
- 净结果：此前“8项开放”更正为 **6项书稿硬错 + 2项降级观察 + 4项观察 + 1项源码注释 + 1项工具假阳性**。本轮未改书稿/源码，未触发PDF/RTD；C3 的实际性能仍需原始 bench 日志，不能由表反推。台账 `book/.tmp/r87/report.md`。

## R88（2026-11-02）：剩余 6 项错误项逐项反证复核，全部保留但性质分层

- 触发：用户要求“再对剩余这6项深入审核，避免误判”。对 R87 的 6 项逐条做“先证明原句成立”的反证式复核；主 agent 独立完成。
- C1 ch02 图双同步：源码/正文/问答/步数式都只有1次；第二道屏障无保护对象，确证 LOW。
- C2 ch12 Q5 `Tn>√R`：4096³ Tn=32 远超√110≈10.5却仅+0.3%，正文已改用 L2 footprint，Q&A 旧句确证 LOW。
- C3 ch33 2.6%：同表 nonaligned 8K 是 341.1 vs342.3（慢0.35%），16K慢1.18%，全仓无2.6%原始输出；判 **L4 表文证据链冲突**，不能反推实际性能一定错，LOW。
- C4 ch36 数值位：8/3/1 无法用单一“显式尾数位”或“含隐含位精度”解释；同表 RMS 用 p=8/4/2。确证 LOW。
- C5 ch36 e2m1 表：`000/001` 同为 e=0、同属 [0,1)，表却拆成 [0,0.5]/[0.5,1]；Voronoi 应为 [0,0.25]/[0.25,0.75]，确证 LOW。
- C6 ch37 7.7%：ch35 的 7.7% 是 ws128×256/s2 对 ws128²/s3 的 tile+深度混合差，同几何纯深度约2.8%/8.3%；确证为归因措辞错误，LOW。
- 本轮没有撤回项，也没有新的 A/B 级错误；C3 的边界明确为“表内不可复现”，不是“实测一定慢”。未改书稿/源码，未触发PDF/RTD。台账 `book/.tmp/r88/report.md`。

## R89（2026-11-02）：用户授权后回修 R88 的 6 项确证错误并通过三层验收

- 触发：用户要求“对这些错误逐条确认，确实有错的进行修改，没错的保留现状”。主 agent 逐项重读 R88 的 6 项后只修这 6 项；R87 已降级的 ch35 `384×168` 与 ch37 `MNK` 不改，4 项观察不纳入。
- 回修：ch02 图 fig:2-2 删除第二次 `__syncthreads()`；ch12 Q5 改为 L2 footprint 判据；ch33 将无表内依据的“non-aligned 8K FFPA 快2.6%”改为表内 cross-dense +4.0%、non-aligned 两档慢0.4%/1.2%；ch36 “数值位”改“精度（含隐含位）”，bf16/e4m3/e2m1+ue4m3 改 8/4/2(+4)；ch36 e2m1 的 `000/001` 统一到共享的 e=0 binade `[0,1]`；ch37 将 FP8 7.7% 改为 tile+深度混合差并补同几何纯深度 2.8%/8.3%。
- 细节防误修：括号中的 `ue4m3` 也必须按精度 p=4 处理，所以最终是 `2(+4)`，不是只把 e2m1 改成 2 而保留显式尾数 3；e2m1 前两行保留 8 个码点、第二行用“同上”，与后续“同一区间两码点间距=半宽”一致。
- 验收：`git diff --check` 通过；anchors 33 章 ALL GREEN；lint 三类 0 fail（1 既有 WARN）；permalinks 53/tests refs 9 FAIL0，appB 11 个既有派生提示。主 PDF **591 页 / 0 error / 0 undefined / 缺字 0 / Overfull 15**，pdftotext 逐项命中新文本。RTD 全量 build（未跳 TikZ）全项 OK：英文图 195/195、代码 772/772、公式 121/121、词典 1922；`visual_check` 全部页面 MathJax/资源/表格/图/越界 0 问题。图 2-2 仍保留一次同步，词典无需删单元。
- 边界：未重跑 GPU benchmark；ch33 的旧 2.6% 无原始日志，因此删除绝对断言而不是补写新性能结论；SRC-1 源码注释不回写，R76-T1 为工具假阳性。台账 `book/.tmp/r89/report.md`。

## R90–R99（2026-11-02）：十轮深度审核，新增 ch26c 区间外推错误

- 触发：用户在 R89 回修后要求“再做10轮深度审核”。按十个证据面覆盖 Part I–II 表、Attention/CuTe 表、FP8/FP4 表、源码/PTX、公式边界、计数与回归；主 agent 独立完成，不使用 subagent。
- **新发现 N-5（LOW，L4 区间/数据一致性）**：ch26c 把 D=384 的结论外推到 D≥384。
  - 表 `ch26c:684-690`：D=384 (3,3) spill=2184、TFLOPS136.0（-27.4%）；D=512 (3,3) spill=**1876**、TFLOPS163.4（**-4.4%**），而 D=512 (2,2) spill=2460。
  - `ch26c:741` 图注写“D≥384 后 stages 加深进一步推高溢出（(3,2)1676、(3,3)2184，均高于(2,2)）”；D=512 时 (3,3)1876 反而低于 (2,2)2460。
  - `ch26c:760` 正文写“D≥384 时 … (3,3)下降27%”；D=512 实际为 -4.4%。结论“D≥384 最优回到(2,2)/(2,3)”仍成立，错的是对 spill 方向与降幅的区间外推。
  - 修法：把 741 的“均高于”限定为 D=384；把 760 拆成 “D=384 -27.4%、D=512 -4.4%”。表与图数据不动。
- R90–R99 其余九轮：Part I–II ch09/ch10/ch11 表、ch27 ESS、ch30 D/吞吐、ch31/ch32/ch33/ch35/ch37 表、源码/PTX 契约、公式边界均复算通过；0 项新 A/B 级错误。R89 的 6 项修改无回退，4 项观察与 SRC-1/R76-T1 状态不变。
- 验证边界：本轮无书稿改动，未触发 PDF/RTD；N-5 等下一轮回修。台账 `book/.tmp/r90-99/report.md`。

## R100（2026-11-02）：复审并修复 N-5（ch26c 的 D≥384 区间外推）

- 触发：用户要求“复审新发现的错误，如果确实是错误就修复”。仅复审 R90–R99 新增的 N-5，主 agent 独立完成，不使用 subagent。
- 复审证据：表和 `fig:26c-spill` 一致给出 D=384 `(3,3)` spill=2184 B、相对 `(2,2)` 下降 27.4%；D=512 `(3,3)` spill=1876 B，低于 `(2,2)` 的 2460 B，TFLOPS 仅下降 4.4%。原文把 D=384 的“更高 spill”和“下降 27%”写成 D≥384 的普遍规律，确证为 LOW 的 L4 区间/数据一致性错误；最优组合回落的结论不变。
- 回修：只改 `chapters/ch26c-cute-split-d-sm120.tex` 两处。图注限定为 D=384 实测点并补 D=512 spill 反例；正文分别列出 D=384 的 15.5%/27.4% 与 D=512 的 4.4%。表格、TikZ 数据、其他章节、冻结源码和 RTD 词典不改。
- 验收：`git diff --check` 通过，anchors 33 章 ALL GREEN，lint 三类 0 fail（1 条既有 WARN）；主 PDF 591 页 / 0 error / 0 undefined / 缺字 0 / Overfull 15；RTD 全量 build 全项 OK（英文图 195/195、代码 772/772、公式 121/121）；`visual_check` 93 页需处理 0。`check_mech` 的模板 listing FAIL 仍为已知假阳性，permalinks 53/tests refs 9 FAIL 0。
- 边界：未重跑 GPU benchmark；未找到可推翻书内表格/图数据的原始实验日志；未提交 commit。台账 `book/.tmp/r100/report.md`。

## R101–R110（2026-11-02）：十轮深度审核，新增 ch19 的 smem 阈值区间错误 + 附录 B 三处漂移

- 触发：用户在 R100 后要求“再做 10 轮深度审核，尽量发现隐藏错误”。按十个证据面推进（Part III 数学 / 大 D 阈值 / CuTe 布局与位模型 / TiledMMA 计数 / ch26c 协议 / 量化数学 / 量化性能表 / 附录 B 回指 / 机械面 / 高风险词与图注独有数字）。主 agent 独立完成，不使用 subagent。
- **N-1（MEDIUM，L1/L4 区间）**：`ch19:82` 写“$D\ge512$ 时连 Q/K/V 各一份都放不下”，但同段公式 $3\cdot64\cdot D\cdot2$ B 给出 $D{=}320\to120$ KB、$D{=}384\to144$ KB，均超 99 KB；书中又写明 $D{=}256$ 的 96 KB「恰好只够各一份」，越界点应是 **320**（$D$ 取 64 的倍数）。D=320 正是 Part V 反复使用的形状。修法：改 `D≥320` 并补「D=320 已到 120 KB」；表 19-walls 与 Q1 举例（D=512）不动。
- **N-2/N-3/N-4（LOW，附录 B 转录）**：① 424 μs/次的 fp8 量化链被标为「第34」，该数字只出现在 ch35 §8.2（736−312），ch34 无性能节，按附录 B 自述的“转录自各章性能分析节”改为「第35」；② 2.71×→2.72×（76.55/28.19=2.7155，ch11 正文即 2.72）；③ 3.144→3.145 ms（与 ch09 表一致）。
- **N-5（LOW，同章漂移）**：`ch33:494` 的「4.96 vs 8.13ms」与本章 `tab:33-fp4-matrix`/`tab:33-fp8-matrix` 的 4.97/8.12 不一致，统一到表值并加表引用。
- **N-6（LOW，交叉引用）**：`ch21:69` 导读把“Tensor 是什么”指向 4.1 节（kernel 骨架），定义实际在 3.1 节（标题即「Tensor：engine 与 layout」），改为「3.1 节；4.1 节是它的 kernel 实例」。
- **未改观察**：① `ch30:50` 把「D≤224」归因于 smem（Q 槽位复用），但按 `ch29` 的 `kSmemElems`（复用后 smem=(Sk+Sv)·kBc·D，D>128 时 kBc=64），D=224 只有 42 KB、D=512 也仅 96 KB，公式给不出 224；缺 `persist_d.cuh` 完整 traits，不硬改。② `ch27:486`「SA2++ 两侧各收窄一半」在“相对可行点集两端 (448,2.25)/(112,9) 取中点”的读法下成立，保留。
- **不重复报**：节号体例（252 处文件序号 vs 印刷 1-based，R21 已登记不改；本轮只记录 6 处反而写成印刷号：ch10:156、ch13:858、ch15:711、ch16:732、ch31:161/197）；ch26c VCPB、ch27 四误差源、ch33 smooth-v、ch36 3.24/10.17、ch29 wait 考据仍为残余 medium。
- 验收：`git diff --check` 通过；lint 三类 0 fail（1 条既有 WARN）；anchors 33 章 ALL GREEN；主 PDF **591 页 / 0 error / 0 undefined / 缺字 0 / Overfull 15**（与基线一致），`pdftotext` 逐条命中 5 处新文本；RTD 全量 build（未跳 TikZ）`convert.verify` 全 OK（英文图 195/195、代码 772/772、公式 121/121、词典 1922）；`visual_check` 93 页需处理 0。未重跑 GPU benchmark，性能数字只做书内一致性核对。台账 `book/.tmp/r101-110/report.md`。

## R112（2026-11-02）：colfax 中文译注分册首轮正确性审核

- 触发：用户要求按 `leetcuda-correctness-review` 对 `book/colfax` 做正确性审核。主 agent 独立完成，不使用 subagent。
- 范围：`colfax-cute-zh.tex` 实际 input 的 16 个 TeX 文件、11,906 行；全文阅读正文、证明、例子、TikZ 图注、标签与引用。外部对照：arXiv:2601.05972 官方 `FinalVersion.tex` 与 NVlabs/CuTe（PyCuTe `d8e31a7`）。
- 结论：翻译本身总体忠实，但原文/译文含多处数理与转录错误。确认并回修约 68 组内容问题，另消除 12 处活跃 label 冲突；重点包括：ch2b “Φ 非减⇒已排序”缺正步长前提（反例 `(3,2):(0,0)`）、全零步长已合并判据 `m≤1` 错、ch2c N-补兼容性把定义域写成 `[0,N1)`、秩 1 复合 `td_i→ed_i`、ch2d Q-代入切分指标与 `P_r/P_m`、ch2e 非退化定义缺谓词、ch2f 紧凑例子 stride 8→9、ch3a `α≠1→≠*`、ch3b/ch3e 切分未要求 g 可求补、ch3e m=1 合并漏 `α(1)=*` 与例子陪域、ch4a 相互加细算法尾部漏 X/末项、ch4b 逻辑切分结果 `((2,2),(2,2))→((2,2),(2,8))`。
- 反向自辩：ch4a 四个复合例题经 PyCuTe 复算全部正确，撤回；ch3d “行首冒号”是 read_file 行号前缀误判；逻辑乘积证明的 `cosize(g)≤size(codomain g)` 对 tuple morphism 成立，撤回。
- 验收：`colfax/build.sh`（临时 biber）205 页 / 0 error / 0 undefined / 缺字 0 / Overfull≥1pt 0；活跃 label 重复 0、悬空 0；`git diff --check` 通过；PyCuTe 复算复合/切分/乘积/N-补/紧凑性示例与修订文本一致。colfax 不在 RTD 源集合，未触发 RTD；未改主 book / cute-zhihu。
- 残余：ch2f 两个只给形状、未给步长的逻辑切分例子无法唯一还原，保留 observation。报告与台账 `book/.tmp/colfax-audit/`。
