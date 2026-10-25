# 全量逐章证据链审核 playbook（R60 沉淀）

## 1. 适用场景

当目标不是“检查这次 diff 有没有改错”，而是“尽量找出书里残留的隐藏错误”时使用：

- 发布前全书正确性审计；
- 多轮局部修复后，怀疑旧审核只覆盖了主结论；
- 用户明确要求逐章、逐条、由主 agent 自己审；
- 需要重建“哪些章节真的没有发现问题”的基线；
- 需要把审核结果转成批量回修和完整构建验收。

不适用于：typo、纯排版、只改一个公式的小修。小修仍按 `SKILL.md` 的常规 L0–L4 流程走。

## 2. 为什么这种方式能多查出隐藏错误

传统审核容易只看：

- 章节主推导；
- 被修改的 diff；
- 历史 ledger 中已经列过的风险项；
- 代码 listing 的标题，而不看实际源码；
- 公式本身，而不看图注、问答、实验说明和工具选项。

R60 把“章节审阅”改成“claim 审阅”，并强制覆盖每个断言的八个主要表面：

| 表面 | 典型隐藏错误 |
|---|---|
| 正文与公式 | 符号、量词、充分/必要、分母、单位 |
| TikZ 节点与图注 | 图内仍写旧符号，正文已改；图注的计数/方向与图不一致 |
| 表与表注 | 列口径混用、ceil/floor、历史数据被当作新复测 |
| 面试问答、小结、标题 | 沿用旧结论；公式在 Q&A 中多一个因子 |
| listing / 代码注释 | 标题行号对、实际 linerange 或调用参数错 |
| 测试、bench、CLI、backend | 测试能过但 benchmark 传错参数；CLI 接受但后端 assert 拒绝 |
| 附录与 permalink | 用当前工作树验证冻结 commit 的链接；行区间根本不存在 |
| RTD 词典 | 改了图内中文/代码注释，却只更新 tex，不更新 figures/code/math 词典 |

**核心判断**：错误数量增加，通常不是书突然变差，而是审核范围从“主结论”扩到了“所有可验证表面”。

## 3. 开始前必须固定的事实

1. `git rev-parse HEAD` 与工作树状态；
2. 书稿基线 commit，不能把未提交修改混进“历史事实”；
3. 从 `book.tex` 解析实际 chapter input，而不是用 `chapters/*.tex` 目录猜测范围；
4. 记录 `book.tex` 未直接列出、但由章节二次 input 的文件（如 `ch19b` 的 `wp0–wp7`）；
5. 记录外部源码快照：
   - 主仓 `.cuh`：当前工作树 + `verify_anchors.py`；
   - ffpa-attn：书中 pinned commit；
   - CUTLASS：本地版本；
   - PTX：本地文档版本；
   - appD：每条 permalink 使用它 URL 里的 commit，不用 HEAD。
6. 明确不计入技术审核的文件（例如纯翻译白皮书）和仍要做的排版/构建检查。

## 4. 执行流程

### Step 1：建立 100% 覆盖清单

范围至少包括：

```text
41 个主 book chapter input
8 个 ch19b 白皮书子文件（只通读，不审技术）
5 个附录 A–E
所有 \lstinputlisting 的源码路径与 linerange
所有测试/bench/CLI/backend 入口
所有附录 permalink
```

为每个文件记录：

```text
order | key | path | lines | first_read | first_status | second_review | notes
```

`first_status` 只能是：

- `no-issue`：本轮未发现问题；
- `candidate`：有候选；
- `observation`：证据不足/未复测；
- `whitepaper-NA`：纯翻译豁免。

不要在没读完、没复审时写 `no-issue`。

### Step 2：逐章完整阅读

按 `book.tex` 顺序读，不跳章。每章至少覆盖：

- 正文、公式、推导链；
- 图内节点文字、图注、TikZ 的标注；
- 表头、单元格、表注；
- 面试问答、小结、实验建议；
- listing 标题与实际 linerange；
- 跨章引用、章号、表号、图号；
- 本轮的修改是否传播到这些表面。

第零层仍按 `SKILL.md` 的 L0 先过；L0 不闭合时，先记录推导缺口，不要用后续术语解释掩盖。

### Step 3：写 claim ledger

一条 claim 一行，至少：

```text
ID | 位置 | 原断言 | 层级 | 证据 | 首轮判定 | 反向自辩 | 最终判定 | 处置
```

要求：

- 位置写 `文件:行`；行号必须回读当前 HEAD 校准；
- 原断言尽量保留原文关键短语，便于检索；
- 证据必须能指向源码行、PTX 段落、头文件、反例或独立算术；
- 首轮候选不能直接进入修复清单；
- 同一 root cause 在不同章节的出现合并成一条，避免按文件重复计数。

### Step 4：机械校验优先

人工阅读无法可靠覆盖的量，用机械校验兜底：

1. **listing**：解析所有 `\lstinputlisting` 和 `\begin{lstinputlisting}`，检查 `.cu/.cuh` 路径、标题、单段/多段 `linerange` 是否越界；
2. **permalink**：对每条 URL 用 `git -C <repo> show <commit>:<path>` 检查文件和行范围，不能只看当前工作树；
3. **anchors**：跑 `verify_anchors.py`，但只把它当锚点有效性，不替代 pinned commit 检查；
4. **算术**：对 AI/ridge、scale 数、SF 字节、寄存器池、ceil/floor、加速比独立复算；
5. **测试/bench**：不仅看 case/PASS 字符串，还要看 `launch` 的参数是否和全量尺寸一致；
6. **CLI/backend**：CLI choices 是“可输入”，不是“可运行”；要到 backend 的 assert / dispatch 确认；
7. **RTD 词典**：改图内中文/代码注释/公式后，检查 figures/code/math 三套词典和强制 TikZ 重建。

### Step 5：按 root cause 合并候选

典型 root cause：

- 绝对化措辞：`永远/唯一/全部/必然/消除/无损`；
- 单位混淆：bit/byte、per-thread/CTA、request/sector、stage/prefetch depth；
- 资源计数：尾部 ceil、块数、SF 原子数、M4N2 平均预算；
- 协议方向：expect-tx 多报/少报、相位、wait_group；
- 跨章口径：Split-D 的 LSE 合并、fp16/bf16 基线、V buffer；
- 复现/工具：bench 参数、CLI/backend、permalink、历史数据标注；
- 证据强度：从趋势/单一极值推出“零均值/全部来自/单调”。

合并时写清“同一断言”还是“不同断言”；不同断言不能为了统一而强行改。

### Step 6：反向自辩与置信度校准（强制）

每条候选进入修复前，先以“证明原句其实是对的”为目标：

1. 扩大回读范围：复测表、附录、图内数据、跨章引用；
2. 换口径自辩：前提、定义域、单位、指称、版本；
3. 数值过舍入带：显示值 ±0.5 ulp 的区间；
4. 回原文核对转述；
5. 让 Review Plan / Code Review 交叉复核，但它们的建议仍是候选。

R60 的实际教训：

- `kBlockSwizzle` 被误读成组数 `S`，源码证明它是 0/1 开关，撤回；
- `expect-tx` 少报不必然是“提前放行”，单 bulk 少 1 字节会落负挂死，收窄；
- “pad 列贡献为零”存在“补零对收缩净贡献为零”的合法读法，降级；
- 旧 bug 复现未写明 checkout，但章首已 pin 源码，撤回。

**置信度与严重度分开记录**：

```text
confidence: high / medium / low
severity: HIGH / MEDIUM / LOW
```

HIGH 不等于一定改；LOW 不等于一定不改。证据确凿的表述问题也应修，证据不足的严重问题也不能硬改。

### Step 7：修复传播

修一条 claim 后至少检查：

- 正文、公式、图内文字、图注、表、问答、小结；
- 同章其它位置与跨章相同口径；
- tests/bench/CLI/backend/附录 permalink；
- RTD 三套词典（figures/code/math）。

改动原则：

- 先改公式/定义，再改解释；
- 公式和图注必须同批改；
- 历史性能数据无法复测时，明确标“旧口径/未复测”，不要伪造更新；
- 只在 Book 内改，不直接改冻结源码；源码 bug 记 CHECKLOG 或正文勘误；
- 每次修改后回读目标区域，确认工具结果不是假成功。

### Step 8：构建矩阵与三层验收

书稿改动后的最小闸门：

```text
主 book PDF
cute-zhihu PDF
colfax PDF（如存在且受影响）
RTD 全量 ./build.sh --force-tikz
RTD visual_check 全部页面
```

通过标准：

- PDF：exit0、0 LaTeX error、0 undefined、0 missing char，记录页数与 Overfull；
- RTD 文件级：全部 `[OK]`，词典/图集/表格/锚点均通过；
- RTD 浏览器级：MathJax error=0、资源失败=0、可见残迹=0、越界公式=0、需处理页面=0；
- GPU 未复测就明确写“未复测”，不能用 PDF/RTD 绿灯代替性能验收。

如果环境缺宏包（如 biblatex/logreq），可装用户 TeX 树或用临时解包 biber，但不要改系统包；把环境处理写进验收记录。

## 5. 输出模板

```text
范围：<commit>，<N> 个 TeX 文件、<M> 行、<K> 个 listing/permalink
方法：逐章 L0–L4 + 全表面 claim ledger + 机械校验 + 反向自辩
候选：<N> 条（confidence/severity 分布）
撤回：<N> 条及理由
确认并修复：<N> 条（H/M/L）
未验证/残余：<N> 条
逐章状态：no-issue / candidate / observation / NA 清单
静态：diff check、lint、listing、permalink
PDF：所有入口的 pages/error/missing/Overfull
RTD：verify 项、词典数量、visual_check 页数
说明：本轮无书稿改动时不触发 PDF/RTD，或本轮已改并列出全部构建
```

## 6. R60 的实际量化结果（可作规模参考）

- 主 book：41 chapter input + 5 附录 + 8 白皮书子文件；54 个 TeX 文件、40487 行；
- 候选 67 条；撤回 4 条（C-20/C-32/C-56/C-59）；保留并修复 63 条（HIGH 3、MEDIUM 33、LOW 27）；
- listing：226 个（178 command + 48 environment），路径/行范围无效 0；
- permalink：53 条，审核时 2 条 pinned 快照无效，修复为有效 commit；
- 修复：36 个 TeX/测试文件 + 2 个 RTD 词典 + 3 个 PDF；
- PDF：主 book 589 页、cute-zhihu 248 页、colfax 204 页；
- RTD：英文图集 1883 条覆盖 1792/1792、英文图 196/196、code 771/771、math 121/121；visual_check 93 页需处理 0。

**结论**：这种方式的收益主要来自“覆盖面 + 证据标准 + 反向自辩”三者叠加，不是单纯“读得更慢”。缺点是候选量和人工复核成本高，下一轮必须继续做严重度/置信度分层，避免把 LOW 表述问题当成 63 个 blocking 错误。
