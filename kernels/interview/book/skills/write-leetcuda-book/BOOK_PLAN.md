# LeetCUDA 成书计划（BOOK_PLAN v2）

> 本文档取代 [`../../../book_plan.md`](../../../book_plan.md)（v1 draft，保留作历史参考）。
> 本文档是后续所有章节写作任务的**规范源**；执行进度跟踪见 [`RFC.md`](RFC.md)。
> 基线日期：2026-09-11。源码基线：LeetCUDA `dev` 分支（RFC-A 登记 SHA256 后冻结）。
> 修订 v3（2026-09-11）：图片管线升级为 **drawio 重建主路径**（§5.2、§7.2），引入 drawio-reconstruction / drawio-diagram-builder / drawio-flow-forge 三 skill 分工。

---

## 0. 任务定位

把 `kernels/interview/` 下内嵌在 `.cuh` 中文注释里的教学内容，重构成一本**按难度递进**的 CUDA Kernel 技术书（XeLaTeX → PDF）。

**做什么**：新写中文讲解 + 精选关键代码段；每章配最小测试；注释正确性核查；知乎资料提炼；公式与图示；长期 RFC。
**不做什么**：不改教学源码逻辑（源码冻结，见 §5.3）；不做 KDP/封面/上架；不输出 HTML。

---

## 1. 预研事实基线（2026-09-11 核验）

### 1.1 构建环境（修正 v1 draft §10）

| 组件 | 状态 |
|---|---|
| xelatex（TeX Live） | ✅ 本机 `/usr/bin/xelatex` |
| ctexbook.cls | ✅ `/usr/share/texlive/.../ctexbook.cls` |
| Noto Sans Mono CJK SC | ✅ fc-list 2 个命中 |
| DejaVu Sans Mono | ✅ fc-list 4 个命中 |

结论：**本地可直接构建 PDF**。v1 draft「开发机缺环境」的结论已过时；§10 安装命令保留进附录 C 供其他机器复现。

### 1.2 Git 现状

- LeetCUDA 是独立 git repo（工作分支 `dev`），workspace 根 `/workspace/dev/vipshop/LeetCUDA`。
- `.gitignore`：L24 `*.tex`、L26 `pdfs`、L32 `*.aux`。`tex/` 下 `.tex/.pdf/.aux` 均为历史跟踪文件。
- 新建 `book/**/*.tex` 会被 `*.tex` 静默忽略 → RFC-0 加负向解禁规则（见 §8.3）。

### 1.3 源码结构核验表（行号三点交叉验证：Phase 注释 / `#if` 宏边界 / `__global__` 行号）

| 文件 | 行数 | 功能块 → 行区间（kernel 名 @ 行号） |
|---|---|---|
| common.cuh | 773 | 基础宏/kWarpSize L1-31；MMA PTX 宏 L32-109（Phase 7b-1 L32）；XOR swizzle v1/v2 L110-348（`NOTES_V2_ENABLE_SWIZZLE_V2` 门控 L151、派发器 L344）；TMA/mbarrier helpers L350-489（`#if WGMMA‖TMA_MMA_WS` L350 起）；setmaxnreg L462-475（`NOTES_V2_ENABLE_SETMAXNREGS` L469）；WGMMA 宏 L490+（Phase 7d L492）；TMA descriptor 位域注释 L500-700；TensorMap helpers |
| base.cuh | 909 | Phase0 面试速查 L4-86；Phase1 归约 L88-305（warp L102、block L169、`block_reduce_all` @223、`dot` @251、`dot_vec4` @279）；Phase2 elementwise L307-520（`relu`@318、`relu_vec4`@330、`elementwise_add`@347/@358、`histogram`@381、`merge_attn_states`@437）；Phase3 softmax/norm L521-801（Level1 naive L537、Level2 safe L554、Level3 online L580-665；rms L667；ln L724）；Phase4 rope L803-836（`rope`@817）；Phase5 转置 L837-909（`mat_transpose`@858、`mat_transpose_padded`@871） |
| sgemv.cuh | 102 | `sgemv_k32`@24、`sgemv_k128`@53、`sgemv_k16`@85（warp-per-row） |
| sgemm.cuh | 434 | 五层金字塔注释 L7-21（AI 公式 L18-20）；`sgemm`@33（Block Tile）；`sgemm_vec4`@108（Vec4+Thread Tile 4×4）；Phase7a+ L185；`f32x4_tf32x4_kernel`@205；`sgemm_tf32`@229（WMMA+cp.async 双缓冲） |
| hgemm.cuh | 2100 | Phase7b `hgemm_mma_stages_tn`@129；swizzle 版 L399-716（`#if SWIZZLE_V2` L391，kernel @434）；CuTe L718-1427（`#if CUTE` L779-1427，Phase7c L718、7c-1 L788、kernel `hgemm_mma_stages_tn_cute`@821、launch wrapper L1197）；WGMMA L1428-1857（`hgemm_wgmma_stages_tn`@1477，仅 sm_90a）；TMA+WS L1859-2100（`hgemm_tma_mma_ws_tn`@1874，SM120） |
| flash_attn.cuh | 3490 | FA2 MMA `flash_attn_mma_stages_split_q`@112（L5-790）；`#if TMA_MMA_WS` L791-2194：FA2 TMA+WS `flash_attn_tma_mma_ws_stages_split_q`@889（L792-1440）、FA3 `flash_attn_3_tma_ws_stages_split_q`@1516（L1441-2192）；`#if CUTE` 5 块 L2196-3488：fa_cute traits（`FlashAttn2CuTeTraits` L2226、`FlashAttn3CuTeTraits` L2296）、FA2 CuTe MMA `flash_attn_mma_stages_split_q_cute`@2532、FA2 CuTe TMA+WS `flash_attn_tma_mma_ws_split_q_cute`@2811、FA3 CuTe `flash_attn_3_tma_mma_ws_split_q_cute`@3130、TMA smoke `flash_attn_3_cute_tma_copy_smoke`@3449 |
| ffpa_attn.cuh | 641 | 头部设计注释 L1-29；`FFPAAttnSplitDCuTeTraits` L31-79（QK Tile<64,64,16> / PV Tile<64,16,16>）；cp.async 版 `ffpa_split_d_cute`@81-375；TMA+WS 版 `ffpa_attn_tma_mma_ws_split_d_cute`@380-641。头注释性能口径为 PRO 5000（与 README 的 5090 口径不同，F4 类核查样本） |
| notes-v2.cu | 4915 | FALayout enum L43；cute 系 test L102-510；test_* L510-2470；bench_* L2470-4370；`test_swizzle_equiv` L4657；main L4668（CLI：`--bench*`/`--fa-layout`/`--fa2-cute`/`--fa3-cute`/`--fa2-cute-cpasync`/`--fa3-cute-tma-smoke`/`--tma-mma-ws`/`--swizzle-eq-check`）；文件尾 L4836-4915 有各 arch 快速编译命令 |
| bench_*.cu/py | 1737 | bench_attn.cu 711、bench_ffpa.cu 649（含 ffpa test）、bench_sgemm.cu 254、bench_sdpa.py 123 |

**注意**：notes-v2.cu 的 GEMM/FA test 以 cuBLAS/cuDNN 为参考（如 `test_hgemm_mma` 用 `CUBLAS_COMPUTE_16F`+`COMPUTE_32F` 双参考）；最小测试需换 CPU fp64 参考（见 §6）。

### 1.4 编译宏 × arch 矩阵（build.sh 实测）

| arch | CUTE | WGMMA | TMA_MMA_WS | CUDNN | 备注 |
|---|---|---|---|---|---|
| sm_86 | ✔ | ✘ | ✘ | ✔ | Ampere |
| sm_89 | ✔ | ✘ | ✘ | ✔ | Ada |
| sm_90a | ✔ | ✔ | ✔ | ✔ | Hopper；`TMA_MMA_WS` 需 CUDART≥13.0（common.cuh L21-23 `#error`） |
| sm_120a | ✔ | ✘ | ✔ | ✔ | Blackwell（本机 PRO 5000） |

`NOTES_V2_ENABLE_SWIZZLE_V2` 与 `NOTES_V2_ENABLE_SETMAXNREGS` 默认**都不开**（后者 sm_120a 上被 ptxas 以 C7506 丢弃）。

### 1.5 知乎资料资产

- CLI 已安装已授权（`/root/.local/share/zhihu-cli/current/zhihu-cli`）；专栏枚举用 skill 的 `fetch_column_fulltext.py`。
- 根 README「技术博客推荐」约 80 篇入册；**README 之外已发现增量**：@reed《cute 之 TMA Descriptor编码与隐藏的第21bit》(p/2037200219700449995)、《NVidia GPU指令集架构-程序控制和原子操作》(p/712357443)；关联作者：@melonedo（布局代数·除法）、@Anonymous（Layout 变换技巧）、@可怕的杰瑞（TiledMMA 简单理解）、@weishengying（cute swizzle）、@水木皇工仔（swizzle/LDSM/MMA）、@Arthur（swizzle 自动消除）。
- 主参考：CuTe 原理 = @reed + @竹熙佳处；swizzle = @frankshi（417 赞）。
- RFC-B 执行全量专栏枚举（本计划阶段不下载全文）。

### 1.6 本地 skills 与 memory 素材

| 资源 | 用途 |
|---|---|
| cuda-cpp-kernel skill 的 `references/ptx-docs/` | F2 类 PTX 指令语义核查（mma/ldmatrix/cp.async/wgmma/mbarrier/setmaxnreg 等） |
| cutlass-cpp-kernel skill 的 `sm89/90/120-optimization-guide.md` 等 | F1 类架构事实核查 |
| cuda-auto-tune skill + 用户 memory「nsys first」方法论 | 附录 B 性能复测流程 |
| memory：setmaxnreg 满池死锁 | ch18 常见坑素材（alloc/dealloc 总和必须 <65536 且留 ≥2048 slack） |
| memory：partition_fragment 未初始化陷阱 | ch22 常见坑素材（rm fragment 必须显式 copy，undef 会被 cicc 折叠为 0） |
| ffpa-cuda-understand skill | ch19/26 大 D split-D 背景知识 |

---

## 2. 相对 v1 draft 的决策变更

| # | v1 draft | v2（本文档） | 理由 |
|---|---|---|---|
| D1 | 正文≈每章大量 `linerange` 引文（代码 ~131 页） | 原理讲解+关键代码段（每段 ≤40 行）；完整代码→附录 D **commit permalink** | 用户要求「适合阅读的原理理解+关键代码段」 |
| D2 | CuTe 分散在 ch13/19/20 | **Part IV 独立成篇**：原理 4 章（ch20-23）+应用 3 章（ch24-26） | 用户要求专门大章 |
| D3 | Part I 按文件物理序（merge_attn_states 在 softmax 前） | **softmax 先于 merge_attn_states**（ch4→ch5） | LSE 合并以 softmax 概念为前提 |
| D4 | 无每章测试（依赖 notes-v2.cu） | **book/tests/chNN_*.cu 最小测试** + common_test.h | 用户要求取代巨型 harness |
| D5 | 注释错误→未定义处置 | **源码冻结**：SHA256 登记后零改动；错误只记 `book/CHECKLOG.md`+正文勘误 | 锚点/linerange 稳定性优先（审查 A1） |
| D6 | TMA 系统讲解在 ch14（晚于 WGMMA ch13） | ch13 前半先讲 TMA descriptor+mbarrier，ch14 回指 | 教学顺序修正（审查 A2） |
| D7 | RFC-G 一次性复测全部性能 | 每章 DoD 含**增量 bench 复测**；RFC-G 只做汇总 | 时序修正（审查 A4） |
| D8 | 图片「下载存档仅内部参考，默认重绘」 | **知乎图片直接引用+出处标注**（作者/文章/链接/日期）；RFC-H 逐个评估替换 | 用户决策 2026-09-11 |
| D9 | 页数目标 250 页（代码 ~131 页） | ≥280 页；**每章正文（非 listings）≥3500 字**（pdftotext 可测） | 口径可测化（审查 B2） |
| D10 | 附录 D=行号对照表 | 附录 D=源码索引 + **GitHub commit permalink**（防 branch 漂移，审查 B8） |

---

## 3. 书稿结构：4 Part · 26 章 · 5 附录

> 每章卡片字段：**源码区间**（引用范围）/ **kernel** / **前置** / **公式** / **图** / **参考** / **测试**。行号以 §1.3 核验表为准；RFC-A 登记锚点断言。

### Part I 基础篇：编程模型、访存与归约（base.cuh）

| 章 | 标题 | 源码区间 | 前置 | 核心公式 | 关键图 | 主参考 |
|---|---|---|---|---|---|---|
| 1 | GPU 架构、执行模型与 Roofline | base.cuh L1-86（扩写） | — | Roofline $P=\min(P_{peak}, AI\cdot BW)$；$AI=\frac{\text{FLOPs}}{\text{Bytes}}$；occupancy | 内存层级图、roofline 曲线 | @reed GPU 指令集系列；@紫气东来 CUDA(一)(二)；cutlass skill arch guides |
| 2 | 归约原语：Warp/Block Reduce 与 Dot | base.cuh L87-305 | 1 | 蝶形归约步数 $\log_2 32$；`__shfl_xor_sync` 掩码 | shuffle 蝶形图、block 两级归约图 | @懒蚂蚁呀不嘿 reduce 详解 |
| 3 | 向量化访存与原子操作 | base.cuh L306-386 | 1 | coalescing 事务合并；向量化带宽公式 | 合并与非合并访存对比图 | @懒蚂蚁呀不嘿 element-wise；@紫气东来 ops(5) |
| 4 | Softmax 三级递进：naive→safe→online ★ | base.cuh L520-665 | 2 | $\text{softmax}(x)_i=\frac{e^{x_i}}{\sum_j e^{x_j}}$；safe（减 max）；online 递推 $m^{new}=\max(m,x)$、$l^{new}=l\,e^{m-m^{new}}+e^{x-m^{new}}$ | 3-pass→2-pass→1-pass 数据流图 | @紫气东来 ops(2)；@DefTruth 图解 FA（前半） |
| 5 | LSE 与分块合并：merge_attn_states | base.cuh L387-519 | 4 | LSE 定义；合并 $l=e^{m_a-m}l_a+e^{m_b-m}l_b$（以源码 5 步推导为准） | 两个分块状态合并示意 | @DefTruth vLLM Merge Attention States |
| 6 | 归一化：RMSNorm 与 LayerNorm | base.cuh L666-801 | 2 | $\text{RMSNorm}(x)=\frac{x}{\sqrt{\frac1K\sum x_k^2+\epsilon}}\odot g$；LayerNorm 含 $\mu,\sigma$ | 1-pass vs 2-pass 归约结构 | @紫气东来 ops(1) |
| 7 | RoPE 与矩阵转置（Bank Conflict 专题） | base.cuh L802-909 | 3 | RoPE 旋转矩阵 $R(\theta)$（分块对角/复数两种形式） | bank conflict vs padding 图 | @懒蚂蚁呀不嘿 transpose；RoPE 参考 RFC-B 补充 |

### Part II GEMM 篇：从 CUDA Core 到 Tensor Core

| 章 | 标题 | 源码区间 | 前置 | 核心公式 | 关键图 | 主参考 |
|---|---|---|---|---|---|---|
| 8 | SGEMV：memory-bound 的三种划分 | sgemv.cuh L1-102 | 1,3 | $AI\approx\frac{2K}{2K\cdot 4B}\approx0.5$ → memory-bound | warp-per-row 划分图 | @有了琦琦的棍子 gemv 优化 |
| 9 | SGEMM 阶梯一：Block Tile→Vec4→Thread Tile | sgemm.cuh L6-183 | 3 | 五层金字塔 AI 逐级公式（sgemm.cuh L18-20 已有，核校后引用） | grid→block→warp→thread 四级 tiling 图 | @白牛 how-to-optimize-gemm；@紫气东来 CUDA(三) |
| 10 | SGEMM 阶梯二：TF32 WMMA 与 cp.async 双缓冲 | sgemm.cuh L184-434 | 9 | TF32 数值格式（10bit 尾数）误差界 | 双缓冲流水时序图 | @木子知 WMMA；@Frank Wang Async Copy |
| 11 | HGEMM：mma.sync m16n8k16 + ldmatrix + 多级流水 | hgemm.cuh L3-397 | 9,10 | mma m16n8k16 每指令 FLOPs；fragment 寄存器布局 | ldmatrix 图、a/b/c fragment 布局图、kStages 流水图 | @木子知 MMA PTX；@Anonymous GEMM 细节(一)；@reed Load 和 Cache |
| 12 | HGEMM：XOR Swizzle、寄存器双缓冲、Block Swizzle | hgemm.cuh L399-716 + common.cuh L110-348 | 11 | swizzle 地址变换 $addr'=addr\oplus mask$ 及可逆性；**block swizzle 坐标映射** | **smem swizzle 前后排布图**（用户点名）、**block swizzle block layout 图**（用户点名）、XOR 位运算示意 | **@frankshi（主）**；@reed cute Swizzle；@Titus(一)(二)；@进击的Killua；@Anonymous GEMM 细节(三)；@JoeNomad block swizzle |
| 13 | Hopper 之路：TMA、mbarrier 与 WGMMA | hgemm.cuh L1428-1857 + common.cuh L350-773（descriptor 位域 L500-700） | 11 | **WGMMA smem descriptor 64bit 编码逐字段**；SS/RS 模式；mbarrier arrive_tx/wait 事务模型 | warpgroup 数据流图、descriptor 位域图、mbarrier 状态机 | @reed TMA descriptor 第21bit；@竹熙佳处 TMA Copy；cutlass skill sm90 guide；ptx-docs wgmma/mbarrier |
| 14 | SM120：TMA + mma.sync + Warp Specialization | hgemm.cuh L1859-2100 + common.cuh setmaxnreg L462-475 | 13 | producer/consumer 到达数计算；setmaxnreg 池数学（含满池死锁界） | TMA box 图、producer/consumer 时序图 | @竹熙佳处 TMA Copy；@Frank Wang；cutlass skill sm120 guide |

### Part III Attention 篇：FlashAttention 2/3 与大 head_dim

| 章 | 标题 | 源码区间 | 前置 | 核心公式 | 关键图 | 主参考 |
|---|---|---|---|---|---|---|
| 15 | Attention 数学基础与 FlashAttention 原理（新写，公式密集承上启下章） | flash_attn.cuh L5-110 头注释 + FA2/FA3 论文 | 4,5,9 | $\text{Attn}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$；FA2 online rescale $O\leftarrow O\,\mathrm{diag}(e^{m_{old}-m_{new}})+PV$；分块循环不变量；复杂度 $O(N^2 d)$ FLOPs / $O(N)$ 显存 | attention 分块计算示意（$B_r\times B_c$ tile 流水） | **@DefTruth 图解 Online-Softmax→FA V1/V2/V3（主）**；@紫气东来 ops(7)(8) |
| 16 | FA2(一)：Split-Q + MMA 多级流水 | flash_attn.cuh L5-790 | 15 | split-Q warp 行划分数学；每 warp 独立 online softmax | **split-Q warp 布局图**（README ASCII 图可升级）、Q/K/V fragment 流转图 | @DefTruth 图解 FA；FA2 论文 |
| 17 | FA2(二)：TMA + Warp Specialization | flash_attn.cuh L792-1440（宏块 L791-2194 内，引文不含指令行） | 13,16 | mbarrier arrive_count 协议（源码 L808 起 257 协议注释） | producer/consumer 双流水时序图 | @Frank Wang；@竹熙佳处 TMA Copy |
| 18 | FA3：双 Consumer Warpgroup | flash_attn.cuh L1441-2192 | 17 | 2 consumer WG 分工数学；softmax 与 MMA overlap；setmaxnreg 池数学（alloc/dealloc 和 <65536，留 ≥2048） | 双 consumer 角色图 | FA3 论文；setmaxnreg 满 pool 死锁案例（repo memory） |
| 19 | FFPA Split-D：大 head_dim 分块注意力 | ffpa_attn.cuh L1-641 | 15,22 | $S=QK^\top=\sum_{d\text{-chunk}}Q_{:d}K_{:d}^\top$；PV 按 D-chunk 分块 + LSE 跨 chunk 合并 | D-chunk 切分图、两阶段合并图 | @DefTruth FFPA(Split-D)；ffpa-cuda-understand skill |

### Part IV CuTe 篇：布局代数与现代化 Kernel 工程（原理 4 章 + 应用 3 章）

| 章 | 标题 | 源码区间 | 前置 | 核心公式/概念 | 关键图 | 主参考 |
|---|---|---|---|---|---|---|
| 20 | CuTe(一)：Layout 基础与代数 | 新写（对照 hgemm.cuh CuTe types L788-820） | 9 | 第一节 Shape/Stride/坐标→offset（colex 序线性化 $idx=\sum_i s_i c_i$）；再 Compose/Inverse/Product/Divide 形式化定义 | layout 坐标映射图、mode 分组图 | **@reed Layout+代数几何解释；@竹熙佳处 Compose&Inverse、Product&Divide**；@melonedo 除法实战；@Anonymous Layout 技巧 |
| 21 | CuTe(二)：Tensor 与 TiledCopy | 新写（对照 g2s/s2r copy 代码） | 20 | TV-layout / MN-layout 对偶；Copy_Atom→TiledCopy→partition 链 | thread×value 网格图 | **@竹熙佳处 tiled copy（主）**；@reed Copy 抽象；@Anonymous GEMM 细节(二) |
| 22 | CuTe(三)：TiledMMA 与 fragment | 新写（对照 FFPAAttnSplitDCuTeTraits L31-79） | 21 | TiledMMA = Atom×Layout×Tile；EURepeat；`convert_layout_acc_rowcol`；**未初始化陷阱** | TiledMMA partition 图 | **@竹熙佳处 tiled mma（主）**；@reed MMA 抽象；@可怕的杰瑞；@水木皇工仔 |
| 23 | CuTe(四)：Swizzle 与 TMA | 新写（对照 GMMA::Layout_K_SW128_Atom 用法） | 12,20 | `Swizzle<B,M,S>` 位运算；SW128 atom；make_tma_copy；**descriptor 第 21bit**；与 ch12 手写 XOR 对照 | SW128 atom 排布图、smem swizzle 前后排布（回指 ch12） | **@reed Swizzle+TMA descriptor；@竹熙佳处 TMA Copy**；@weishengying；@进击的Killua/@Titus |
| 24 | CuTe 应用(一)：HGEMM | hgemm.cuh L718-1427（宏块 L779-1427 内） | 11,12,21,22 | gemm_ss；SmemLayoutAtom+tile_to_shape；多级流水 CuTe 表达 | 与 ch11/12 手写版**对照表**（本章核心资产） | @reed 简单/高效 GEMM、GEMM 流水线；@朱小霖 cute 101；@进击的Killua 实战 |
| 25 | CuTe 应用(二)：FlashAttention 三实现对照 | flash_attn.cuh L2196-3488（5 对宏块，见 §1.3） | 17,18,21-23 | fa_cute traits 复用结构；cp.async 版/TMA+WS 版/FA3 版演进 | 三实现结构对照图 | @66RING cute 复现 FA；@shengying.wei tiny-flash-attention；@Titus GEMM 流水线 |
| 26 | CuTe 应用(三)：FFPA Split-D | ffpa_attn.cuh 全篇（重点 L31-79+两 kernel） | 19,22,23 | 双 TiledMma（QK Tile<64,64,16>/PV Tile<64,16,16>）；cp.async 与 TMA 两版对照 | 双 TiledMma 数据流图 | @DefTruth FFPA 文；ffpa-cuda-understand skill |

### 附录

| 附录 | 内容 | 源 |
|---|---|---|
| A | 基础设施工具箱：MMA/WGMMA PTX 宏、swizzle v1/v2、TMA/mbarrier helpers、TensorMap、setmaxnreg | common.cuh 773 行 |
| B | 性能数据与口径：README SM120a 表 + 补「对应章节/函数名」列 + 复测数据（统一注明 GPU/库版本/日期） | README + RFC-G |
| C | 构建、运行与最小测试指南：build.sh、CLI、**章×编译宏×arch 开关矩阵**（§1.4 扩展）、cutlass include、notes-v2.cu 角色声明（保留为集成 bench harness） | build.sh、notes-v2.cu |
| D | 源码索引：topic ↔ file:line ↔ **GitHub commit permalink**（固定 hash）——正文的完整代码入口 | 全书 |
| E | 参考资料全集：知乎作者/文章/链接/对应章节/引用日期/图片引用清单 | RFC-B |

---

## 4. 每章统一模板（10 节）与单章 DoD

### 4.1 模板

1. **本章导读** — 学什么、前置章节、源码区间（file:line）、编译宏与架构限定、预计篇幅
2. **问题与动机** — 为什么需要这个算子/优化
3. **数学原理** — 公式（amsmath 编号，符号遵循记号表 §7.4）
4. **设计与实现** — 原理讲解 + 关键代码段（每段 ≤40 行，`linerange` 多段裁剪；完整实现链接附录 D）
5. **图示** — ASCII 草图（起步）或知乎引用图（带出处）
6. **性能分析** — Roofline/算术强度 + 本章复测数据对照
7. **勘误与考据**（精简框注） — 本章注释核查中发现的源码注释问题（全量记录在 `book/CHECKLOG.md`）
8. **常见坑**
9. **面试要点速查** — Q&A
10. **最小测试与动手实验** — 对应 `tests/chNN_*.cu`，编译/运行/预期输出

### 4.2 单章 DoD（完成定义，八条全过才勾 RFC）

1. `chapters/chNN-*.tex` 完成，全书编译零 `! LaTeX Error`
2. 锚点断言通过（`scripts/verify_anchors.py`：源文件 SHA256 + 各 linerange 首/末行文本匹配 + 范围内 `#if/#endif` 成对）
3. 注释核查完成并记录 `book/CHECKLOG.md`（正文只留精简勘误）
4. `tests/chNN_*.cu` 编译并运行 PASS（或目标 arch 不在位时输出 SKIP）
5. 本章 kernel bench 复测一次，数据落 `.tmp/book-bench/chNN/`（增量，供附录 B 汇总）
6. 本章公式清单、图清单登记进 RFC.md 登记表
7. 延伸阅读（2-5 篇知乎参考）就位
8. 单章自审 checklist + 本章 pdftotext 抽查（无豆腐块、无横向溢出）

### 4.3 行文风格（作者风格基线，2026-09-11 用户定）

以 @DefTruth 的《图解:从Online-Softmax到FlashAttention V1/V2/V3》与《WINT8/4》系列为风格基线——**原理一定要讲细**，其余可自由发挥：

- **公式逐项展开**：不跳步。每个符号首次出现即定义（对照记号表），推导逐步给中间量（如 online softmax 的 $m/\ell/O$ 联动更新逐块推）。
- **指令逐 bit/逐字段解释**：涉及 PTX/SASS 的内容给到位域图（如 WGMMA descriptor 逐字段、LOP3 真值表式讲法、PRMT byte 选择模式）。
- **先直觉后形式化再代码**：为什么需要 → 数学表达 → 关键代码段；三者之间显式衔接。
- **中文叙述、术语保留英文**（warp/swizzle/mbarrier 等不硬译）；图解优先于长段文字。
- 风格样本：图解 FA 系列（p/668888063）、WINT8/4-(00)~(03)（p/657072856 / 657070837 / 657073159 / 657073857）。

---

## 5. 素材工程

### 5.1 知乎资料工作流（zhihu skill 五步法）

1. **检索/枚举**（RFC-B）：`fetch_column_fulltext.py <专栏URL>` 枚举 @reed、@竹熙佳处、@frankshi（若有专栏）全部文章；补充作者按 §1.5 清单。产出 `book/references/zhihu-inventory.md`（含 README 外增量）。
2. **取全文**（每章任务内）：有价值文章取全文，存档 `zhihu-analysis/`（skill 要求目录），提炼笔记进 `book/notes/chNN-notes.md`。
3. **对照提炼**：文中方法/图与本书代码逐点对照，标记可采纳/不适用/存疑，**不照抄文字**。
4. **成文引用**：正文用自己叙述+我们的代码；章末「延伸阅读」列参考。
5. **汇总**：附录 E 全集表（作者/标题/URL/对应章节/引用日期）。

**素材交付顺序**：按 Part I→II→III→IV 章序优先交付（RFC-B 内部约定），避免章节任务被素材卡死。

### 5.2 图片策略（v3，2026-09-11：drawio 重建为主路径）

两级策略：
1. **引用级（过渡/兜底）**：知乎文章原图直接引用，每图标出处：`图 X.Y：标题（图片来源：@作者《文章标题》，链接，访问日期）`。
2. **重建级（主路径）**：用 drawio 管线（§7.2）把原图重建为**可编辑 .drawio + 高清导出图**，书内优先使用重建图；caption 改为「重建自 @作者《文章》」——重建≠原创，出处标注保留。

- 原图归档 `book/figures/zhihu/<author>-<slug>/`（元数据 sidecar：来源 URL/引用章节/替换状态）——防外链失效，兼作重建参考图。
- **水印规则（强制）**：知乎图常带作者水印/平台角标。drawio 重建的 inventory 阶段必须把水印、作者签名、平台 logo 标记为**非内容元素**（不进入重建产物）；独立 Reviewer 验收项必含「无水印/角标残留」；复杂视觉需 crop 时用 `crop_assist.py --exclude` 框除水印区域，水印压在主体无法避开时改用原生元素重画该部分。
- 文字内容不照抄（提炼重写）。
- LaTeX 实现：`graphicx` + figure 环境，出处写进 caption。

### 5.3 注释核查与源码冻结

**源码冻结（硬规则）**：RFC-A 登记各 `.cuh`/notes-v2.cu 的 SHA256 之后，**源文件零改动**。发现的注释错误一律记入 `book/CHECKLOG.md`（记录：位置/原文/问题/证据/建议），正文以「勘误与考据」框注呈现。**不回写源码**——任何回写都会使全书 linerange 与锚点断言失效。

核查五类及证据来源：

| 类别 | 内容 | 证据来源（优先级序） |
|---|---|---|
| F1 架构事实 | SM 数/smem 容量/寄存器文件/Tensor Core 吞吐/频率 | CUDA C++ Programming Guide → cutlass skill 各 arch guide → 官方 spec；正文必须标注具体架构 |
| F2 PTX/SASS 指令语义 | mma/ldmatrix/cp.async/wgmma/mbarrier/setmaxnreg 等 | 本地 ptx-docs（`.github/skills/cuda-cpp-kernel/references/ptx-docs/9-instruction-set`） |
| F3 数学 | LSE 合并/swizzle 可逆性/online softmax 递推 | 独立推导验证 |
| F4 性能断言 | 注释中的 TFLOPS/加速比 | 标注数据来源（README 口径/复测口径）；不可复现的弱化为定性描述。已知样本：ffpa_attn.cuh 头注释 PRO 5000 vs README 5090 口径差 |
| F5 历史陈述 | FA1/2/3 演进、硬件代际 | FA2/FA3 原论文 + NVIDIA 官方博客 |

### 5.4 章节参考映射总表

见 §3 各章卡片「主参考」列。汇总口径：CuTe 原理=@reed+@竹熙佳处；swizzle=@frankshi（主）+@reed/@Titus/@进击的Killua（辅）；FA 数学=@DefTruth 图解系列。完整 URL 清单由 RFC-B 生成 `book/references/zhihu-inventory.md` 维护。

---

## 6. 最小测试工程

### 6.1 目录与规范

```
kernels/interview/book/tests/
├── common_test.h        # 共享：CPU fp64 参考、容差判定、PASS/FAIL 输出、arch SKIP 宏
├── build_tests.sh       # --arch sm_120a [--ch ch04] [--all]；带 -I ../../third-party/cutlass/include
├── ch01_arch.cu … ch26_ffpa_split_d_cute.cu   # 每章一个（见 RFC.md 执行卡片）
└── README.md
```

- 每个 `chNN_*.cu`：单文件独立编译；`init → kernel → CPU 参考 → PASS/FAIL`；无 cuBLAS/cuDNN 依赖（正确性对照用 CPU fp64）。
- 行数分级：基础章（ch1-8）≤300 行；GEMM/FA/CuTe 章（ch9-26）≤500 行；共享逻辑进 `common_test.h`。
- arch-gated kernel 用宏保护，目标 arch 不在位时打印 `SKIP(chNN): requires sm_90a` 并返回 0。
- 测试规模上限 ≤512（CPU 三重循环时限约束）。
- 从 notes-v2.cu `test_*` 抽最小逻辑（抽取源行号见 RFC.md 执行卡片）；notes-v2.cu **保留**为集成 bench harness（角色分工写入附录 C）。

### 6.2 容差分级表（common_test.h 实现）

| 档 | 适用 | 判定 |
|---|---|---|
| F32Acc | fp32 累加路径（sgemm 系列、FA F32Acc） | `max_abs_err ≤ 1e-3`（vs CPU fp64，K≤512） |
| F16Acc | fp16 in/out + fp16 累加（hgemm F16Acc、FA F16Acc） | `max_abs_err ≤ 5e-2` |
| TF32 | sgemm_tf32 | `max_abs_err ≤ 1e-2` |

（首版阈值；ch 任务如需调整须在 RFC 勾选项注明理由。）

---

## 7. 图表与公式工程

### 7.1 图的类型与生命周期（v3）

| 类 | 形态 | 用途 | 产出路径 |
|---|---|---|---|
| A. ASCII 草图 | `verbatim`，注释风格 | 写作期占位 | RFC-H → C1/C2 |
| B. 知乎引用图 | 原图+出处 caption | 过渡期引用、重建参考 | RFC-H → D |
| D. drawio 重建图 | .drawio 源+高清导出 | **知乎图的正式形态**（可编辑、无水印） | drawio-reconstruction |
| C1. drawio 新建图 | .drawio 源+高清导出 | 结构/布局/流程类正式图（ASCII 升级） | drawio-diagram-builder（高保真）/ drawio-flow-forge（快速） |
| C2. TikZ/pgfplots | LaTeX 内嵌 | 函数曲线/数据图（roofline、吞吐曲线） | 手写 TikZ |

每图登记 RFC.md 图清单：`FIG-<ch>-<n> | slug | 类型 | 状态（占位/引用/重建完成/正式）| 出处或源文件`。

**必收图（用户点名）**：smem swizzle 前后排布（ch12/23）、block swizzle block layout（ch12）。

### 7.2 drawio 管线（v4，2026-09-14：生成器单路径，主 agent 直做）

**主路径（全部图）**：tex 内 ASCII 规格/caption 规格为内容源 → python 生成器
（`figures/drawio/<FIG-id>/gen.py`，入库不进 .tmp）→ `drawio-headless -s 3`
导出 PNG（300dpi）→ `view_image` 整图 + PIL 裁剪放大复核细节（≥2 轮迭代）
→ `audit.md`（生成器参数/迭代轮次/复核结论）→ tex 接线（label 锚点整块替换）
→ RFC 图清单状态回写 → 分批 commit。

**PDF 缩放硬验收（A5 版心 ~10cm）**：两位数格 ≥54px 宽、正文字号 ≥12；
CJK 一律禁 bold/italic（字体 fallback 缺字形=方框）；value 禁 `&#10;`；
下标纯文本（c0/c1）；折线箭头显式 mxPoint；渲染铁律全集见
`/memories/repo/leetcuda-book-drawio.md`。

| 辅助 skill | 角色 | 何时使用 |
|---|---|---|
| drawio-reconstruction | 知乎原图高保真重建（多 agent 闭环） | **降级为可选**：确有高质量原图且结构复杂时；当前 figures/zhihu/ 无归档图，不构成主路径 |
| drawio-diagram-builder / drawio-flow-forge | 从零新建辅助 | 复杂自绘图/概念流程图；与生成器路径产出同规范 |

**产物规范**：`figures/drawio/<FIG-id>/`（如 `fig-12-1-smem-swizzle/`）内含 `gen.py`（python 生成器，参数化重制的源码，**入库**）、`<stem>.drawio`、`<stem>.png`（≥300dpi：CLI `-s 3` 导出）、`<stem>.audit.md`；导出图入 LaTeX。生成器不再放 .tmp（被 gitignore，清理即丢失）。

**导出工具链（2026-09-11 已验证可用）**：
- drawio CLI v31.4.5，经官方 deb 安装（**Ubuntu noble 官方源无 `drawio` 包**，来源=`jgraph/drawio-desktop` GitHub releases deb；skill README 的 `apt install drawio` 写法不适用于 noble）
- root 容器无头导出必须 `xvfb-run` + `--no-sandbox`：已建统一 wrapper `/usr/local/bin/drawio-headless`；skill 脚本经 `DRAWIO_PATH=/usr/local/bin/drawio-headless` 全链路可用（`export_drawio.py`/`check_drawio.py` 已实测）；dbus 报错为噪音可忽略
- 兜底：`drawio-diagram-builder/scripts/serve_drawio_preview.py` + 集成浏览器截图
- VS Code drawio 插件（用户已装，hediet.vscode-drawio）：用于**人工查看/微调** agent 产出的 `.drawio` 与逐字校对预览；自动化导出仍走上面 CLI，不依赖插件

**执行模式（2026-09-11 smoke test 结论，用户拍板）**：drawio 重建由**主 agent 直接完成**（inventory→重建→导出→自审），**不派 task agent**——多 agent 闭环实测易卡死超时；audit 如实记录执行模式（自审需标注「coordinator 自审」）。正式重建的小字文本须对照放大 crop 逐字核对（流程演示样例：本 skill `examples/drawio-recon-m1/`，含重建件/导出预览/audit/复现命令）。

**水印规则**：见 §5.2；重建 Reviewer 验收项必含「无水印/作者角标/平台 logo 残留」。

**单图重建 DoD**：①原图已归档 figures/zhihu/；②inventory 完成且水印类元素标记为非内容；③reconstruction 全流程闭环（独立 Reviewer PASS）+ audit.md 完整（producer/reviewer id + sha256）；④导出 PNG 入 figures/drawio/<FIG-id>/；⑤LaTeX 引用切换为重建图，caption 出处改为「重建自 @作者《文章》」；⑥RFC 图清单状态更新为「重建完成」。

### 7.3 公式规范

- `amsmath`，重要公式编号；行内公式不滥用。
- 每章「数学原理」节为公式密集区；代码变量与公式符号的对应在首次出现处说明。

### 7.4 全书记号表（初稿，RFC-A 定稿冻结）

| 记号 | 含义 | 源码对应（示例，RFC-A 补全） |
|---|---|---|
| $M,N,K$ | GEMM 维度 | `M,N,K` 参数 |
| $B_M,B_N,B_K$ | block tile 尺寸 | `kMmaTileM/N` 等（待登记） |
| $W_M,W_N$ | warp tile | — |
| $T_M,T_N$ | thread tile | — |
| $B,H,N,D$ | attention batch/heads/seqlen/headdim | `--bhnd` 参数 |
| $B_r,B_c$ | Q block / KV block 尺寸 | `kMmaTileSeqLenQ×…`（待登记） |
| $m,\ell$ | online softmax running max / running sum | `block_row_max` 等 |
| $\mathrm{LSE}$ | log-sum-exp | — |
| $d_c$ | D-chunk（Split-D 64 宽块） | `kHeadDim%64` |
| $s_i,c_i$ | layout 第 i mode 的 stride/coord（colex） | cute `Shape/Stride` |

---

## 8. LaTeX 管线

### 8.1 目录结构

```
kernels/interview/book/
├── BOOK_PLAN.md / RFC.md / CHECKLOG.md
├── book.tex            # ctexbook[9pt,openany] + \tableofcontents + part/chapter 骨架
├── preamble.tex        # 从 tex/notes-v2.tex 抽取：8 色调色板/字体/lstset；+graphicx/amsthm/booktabs
├── build.sh            # 两遍 xelatex（TEXMFCNF=../tex/）+ 清理
├── chapters/ch01-….tex … ch26-….tex
├── appendices/appA-….tex … appE-….tex
├── figures/{ascii/,zhihu/,drawio/,tikz/}
├── tests/  notes/  references/  scripts/verify_anchors.py
```

### 8.2 复用与修正

- 代码清单：`firstnumber=auto`（行号与源文件一致）+ `linerange` 多段（跳过 `#if/#endif`）；禁止整篇引用 hgemm/flash_attn。
- `texmf.cnf` 复用（`TEXMFCNF=../tex/`），但注明：现代 TeX Live 中 `main_memory` 是编译期常量，真正护栏是 linerange。
- 关键字表审计：删除 0 命中旧名（`sgemm_thread_tile_vec4`、`rope_f32_kernel`、`block_all_reduce_sum`、`flash_attn_mma_stages_split_q_kernel` 等），补入 §1.3 实际 kernel 名与 CuTe/FFPA 标识符（`fa_cute::*`、`FFPAAttnSplitDCuTeTraits`、`gemm_ss/gemm_rs`、`convert_layout_acc_rowcol`、`swizzle_v1/v2_impl`、`SwizzleBMS`、`create_tensor_map`、`NOTES_V2_REG_ALLOC/DEALLOC`、`ffpa_split_d_cute`、`hgemm_tma_mma_ws_tn`、`hgemm_wgmma_stages_tn`、`flash_attn_3_tma_mma_ws_split_q_cute` 等）。

### 8.3 gitignore 追加（RFC-0）

```gitignore
!kernels/interview/book/**/*.tex
kernels/interview/book/**/*.pdf
kernels/interview/book/**/*.aux
kernels/interview/book/**/*.toc
kernels/interview/book/**/*.out
kernels/interview/book/**/*.log
```

（不要写 `!kernels/interview/book/**`——会把 PDF/aux 一并解禁。）

### 8.4 构建

```bash
cd kernels/interview/book
TEXMFCNF=../tex/: xelatex -interaction=nonstopmode book.tex   # ×2 遍
# 或 ./build.sh
```

CWD 必须是 `book/`；源码引用写 `../base.cuh` 形式。

---

## 9. 页数与工作量预算

- 26 章 × 每章 9-13 页（讲解 6-9 + 代码段 2-4；CuTe 原理 4 章可达 12-15）≈ 270-320 页
- 附录 40-60 页 → 全书 ≈ 300-360 页（验收下限 280）
- 每章正文（非 listings）≥3500 字（pdftotext 统计口径：提取非代码环境的中文文本）

---

## 10. 风险与对策

| # | 风险 | 对策 |
|---|---|---|
| R1 | `*.tex` gitignore 静默忽略新章节 | RFC-0 负向解禁（§8.3）+ `git check-ignore -v` 自检 |
| R2 | listings 大文件耗尽 TeX 内存 | 禁整篇引用；一律 linerange |
| R3 | CJK 字体缺失豆腐块 | 本机已验证存在；附录 C 保留安装+预检命令 |
| R4 | 代码行溢出 | `breaklines`+`columns=fullflexible` 已有；每章 pdftotext 抽查 |
| R5 | 全量编译慢 | 后台执行不轮询；日常可单章临时编译 |
| R6/R7 | 行号越界/语义漂移 | 锚点断言脚本入每章 DoD；源码冻结（D5）杜绝漂移源头 |
| R8 | 版权（知乎内容/图） | 图片两级策略：过渡期直接引用+出处标注 → drawio 重建为主路径（无水印、可编辑，caption 保留「重建自 @作者」）；文字不照抄；引用图本地归档 |
| R9 | 注释错误传染书稿 | 五类核查流程+CHECKLOG |
| R10 | 行号漂移 | 源码冻结+SHA256+锚点断言 |
| R11 | 环境 | 本机可构建 PDF；drawio CLI v31.4.5 已装（xvfb wrapper 就绪）；安装文档保留附录 C |
| R12 | zhihu 限额/反爬 | 专栏 API 优先+浏览器兜底；额度异常即停 |
| R13 | ch13 无 Hopper 卡 | 编译级验证+标注复现条件（可选 AutoDL H800） |
| R14 | 引用图外链失效 | 本地归档 `figures/zhihu/`，出版不依赖外链 |
| R15 | drawio 重建成本高（单图=多 agent 闭环）；Electron headless 环境脆弱 | 按章分批（2-4 张/批）领取，重建期间书内先用 B 类引用图过渡；headless 走 `/usr/local/bin/drawio-headless` wrapper（xvfb+--no-sandbox），dbus 报错视为噪音；wrapper 失效时浏览器截图兜底 |

---

## 11. 验收标准

**全书级**：`book.pdf` ≥280 页；TOC 含 26 章+5 附录，页码正确，PDF 书签可用；每章正文 ≥3500 字；逐章 grep 引文文本 `#if/#endif` 成对；pdftotext 抽查无豆腐块；代码行号与源文件一致；每章标注编译宏与架构；引用图 100% 有出处标注；编译日志无 `! LaTeX Error`、Overfull hbox 可控。

**每章级**：§4.2 DoD 八条。

**RFC 级**：RFC.md 每项有验收条目、可勾选、有完成日期。

---

## 12. 范围边界

**包含**：`book/` 下 LaTeX 源、测试、脚本、RFC、CHECKLOG、参考资料清单、PDF 产出。
**不包含**：KDP 元数据、封面设计、HTML 输出、上架流程。
**不改动**：`*.cuh`、notes-v2.cu、bench_*.cu 等教学源码（**源码冻结**，§5.3）。书通过 linerange 引用，源文件保持唯一事实来源。

---

## 13. 待确认事项

1. 书名与署名：暂定《CUDA Kernel 面试背题笔记（进阶版）》/ LeetCUDA Project（沿用 tex 标题风格）
2. PDF 是否入库：默认不入库（gitignore §8.3 已忽略）
3. `tex/notes-v2.pdf` 去留：保留，与书并存（「全量源码打印本」）
