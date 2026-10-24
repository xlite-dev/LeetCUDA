# Claim Ledger 与自审报告模板

## 1. Claim ledger

每条技术断言一行。图注、表标题、面试问答、section title 也要登记。

| ID | 位置 | 原断言 | 层级 | 类型 | 证据 | 判定 | 处置 |
|---|---|---|---|---|---|---|---|
| C-000 | ch15:定理 15.1 | 每行非空 ⇒ 首轮 rescale 不会 NaN | L0 | 推导/前提 | 首块全 mask 反例：`-∞-(-∞)=NaN` | false | 补“首块非空/跳过空前缀” |
| C-001 | ch20:128 | `size == cosize` 即双射 | L0 | 量词/双向 | `(2,2,2):(1,1,5)` 反例 | false | 改写并补 compact 定义 |
| C-002 | ch25:544 | `wait_group N` 等恰好 N 组 | L2 | PTX | PTX `cp.async.wait_group` | false | 改为“最多最近 N 组未完成” |
| C-003 | ch26:301 | P 与 S 共用原寄存器 | L3 | 类型 | `NumericArrayConverter` 生成新 fragment | false | 区分转换与 Aregs |
| C-004 | ch26b:fig | K1 复写 stage 0 | L4 | 图注 | 源码预取位置 `k_write` | false | K2/V2 复写 stage 0 |
| C-005 | ch26c:449 | 默认 (2,2) 只够 1 个 O 分量 | L3 | 资源 | 64KB/16KB + 整除 5 | false | 容量与整除双条件 |
| C-006 | ch26:301 | P 是原 f32 寄存器的零拷贝重解释 | L1 | 指称/类型 | `convert_type` 生成新 f16 fragment | false | 区分布局重解释与 dtype 转换 |

字段说明：

- **层级**：`L0` 数理逻辑（推导/量词/蕴含/反例）；`L1` 定义/指称/边界；
  `L2` API/硬件；`L3` 资源/类型；`L4` 表达一致性。
- **类型**：推导、量词、定义、定理、API、硬件、资源、测量、图注、术语。
- **证据**：文件:行、PTX 章节、头文件签名、host 探针、GPU 日志、`未验证`。
- **判定**：`true`、`false`、`overstrong`（过强）、`undefined`（术语未定义）、
  `needs-test`。
- **处置**：保留、改写、补定义、补边界、标未验证、回修图注。

同一断言在正文、图注、表注、listing、问答多处出现时，**位置列写全**，例如
`ch27:379 + fig-27-2 + ch29:21`，不要只记首个位置；否则修一处、漏三处。
全量轮次在处置列后加来源备注：`agent-A high → 主 agent 复核`、
`机械扫描 → 人工确认`；未复核候选不得进回修清单。

### 证据等级

1. L0 推导逐步复核 + 证明/反例；
2. L1 定义/边界反例；
3. CUTLASS/CuTe 头文件、PTX ISA、CUDA 文档的具体段落；
4. 被审源码的调用点 / 资源表 / host-only 编译探针 / `static_assert`；
5. GPU 测试日志（只证明该 case 跑通）；
6. 无证据 → 必须标 `未验证`。

## 2. 边界矩阵

按章节内容选择行，不要只填“普通”。

| 对象 | 普通 | 尾部 | 空/全 mask | 不整除 | 符号/stride 异常 | 架构/版本 |
|---|---|---|---|---|---|---|
| Layout | `(4,4):(1,4)` | — | size 0 | — | 负 stride / 混合符号 | — |
| cp.async | kStagesK=2 | 最后 kStagesK-1 tile | 无 K 组 | — | — | sm80+ |
| TMA | 128B swizzle | 最后 box | OOB | box 不整除 | — | sm90+ / M=5,6 |
| 寄存器 | D=320 | 最后 chunk | C=0 | D 不整除 64 | F16/F32 累加 | sm120 |
| mask | 部分列 mask | 最后 kv tile | 整行 -inf | Nkv 非 kBc 倍数 | Nkv<Nq | — |
| persistent | 等时 tile | 最后 wave | 0 tile | tile 数不整除 CTA | causal 反向 | — |

## 3. 自审报告模板

```text
# 第零层 + 四层自审报告

范围：<文件列表 / commit / 章节>
PDF/RTD：<有书稿改动 → 按 §5.0 编所有受影响 PDF + RTD build/visual_check；无改动 → 写明未触发>
未做：<GPU 实测 / ...>

## 第零层结论

- 推导链：<全过 / 发现 N 处跳步、反向蕴含、隐藏前提>
- 反例/边界：<...>
- 恒等/近似/上界：<是否已分开>

## 结论

- 通过：<N> 条
- 自审新发现并回修：<N> 条
- 未验证：<N> 条

## 逐章结果

### 第 N 章

- L0 数理逻辑：
- L1 定义/指称/边界：
- L2 API/硬件：
- L3 资源/类型：
- L4 图注/问答/术语：

### 图/表/问答

- ...

## 自审新发现（回修）

1. <位置>：<原句> → <新句>；依据：<证据>。

## 未验证与残余风险

- ...

## 验证

- `git diff --check`：...
- lint：...
- 旧结论检索：...
- 受影响 PDF：`book.pdf` / `cute-zhihu.pdf`（逐个列 exit / pages / Overfull / error / undefined / missing）
- RTD `build.sh` + `convert.verify`：<pass/fail，失败项>
- `visual_check`：<MathJax / console / 404 / 版式>
- GPU：未做/已做（性能验收单独记）
```

## 4. 快速判断卡

| 看到 | 先问 | 常见错法 |
|---|---|---|
| 因此 / 所以 / 由此可得 | 中间步骤是什么？ | 从第一行直接跳到结论 |
| 代入 / 约去 / 消去 | 分母非零？定义域允许？两边同量？ | 约掉零/无穷，或消去不同对象 |
| 上界 / 近似 / 经验值 | 是恒等、上界还是实测？ | 把 2^-4 的误差上界写成 2^-3 的 ULP |
| 等于 / 等价 / 当且仅当 | 两个方向都成立吗？ | 把单向条件写成 iff |
| 所有 / 任何 / 唯一 | 边界和替代路径？ | 漏掉尾部、非整除、空输入 |
| 寄存器 / 一半 / 总量 | 逻辑值还是物理寄存器？每线程还是 CTA？ | 单位混用 |
| 零拷贝 / 原地 | 是否产生新 dtype fragment？ | 把数值转换叫重解释 |
| 共享 | 每 warp 私有副本还是 CTA 一份？ | 低估总寄存器 |
| 周期 / 图样 | 完整 XOR 周期跨几行？ | 按单行宽度猜周期 |
| 消除尾损 / 达到上限 | makespan 与吞吐模型是什么？ | 用局部优化替代全局 |
| 图注 | 它和正文是同一个结论吗？ | 正文改了图注没改 |

## 5. 最小操作顺序

1. `git diff -U2` 收集所有 hunk；
2. 登记 claim ledger；
3. **L0 推导链/量词/反例**；
4. L1 定义/指称/边界；
5. L2 头文件/PTX/调用点；
6. L3 资源表；
7. L4 图注/问答/术语检索；
8. 回修；
9. `git diff --check` + lint + 旧结论扫描；
10. 有书稿改动 → `book/build.sh`（PDF 闸门）；
11. 有书稿改动 → `docs/readthedocs/build.sh` + `tools/visual_check.py`；
12. 写残余风险；两道闸门结果一并记入 audit-history。
