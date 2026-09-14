# 知乎参考资料库存（Appendix E 数据源）

> 生成：2026-09-11 · RFC-B（知乎资料全集）· 维护者：主 agent / 资料工程师
> 数据来源：①知乎开放平台官方 CLI 检索（`zhihu-cli search zhihu`，2026-09-11 共 61 次检索调用，原始结果见 `zhihu-analysis/leetcuda-book-refs-search-raw-2026-09-11.json`）；②LeetCUDA README「高性能计算与分布式-技术博客推荐」表（作者自校第一手清单）；③RFC.md §15 种子清单；④账号本人收藏夹（`zhihu-cli me favorites items`，仅用于交叉核对，未落盘）。
> **状态口径**：
> - **【已核】**＝ 2026-09-11 经知乎搜索接口返回该条目，标题/作者/URL 一致（存在性核验通过）。
> - **【已核-README】**＝ 仅 LeetCUDA README 推荐表收录，本次检索未返回（旧文降权）。
> - **【待核】**＝ 本任务未能确认存在性。
> - **【新增】**＝ 不在 RFC §15 种子表、经本轮检索补充入册。
> - ⚠️ 本轮**未获得任何正文全文**（环境限制，见 §5）；「已核」仅代表可检索，不代表已取文。

## 0. 附录 E 最终表结构（B.6 定稿）

| 字段 | 类型 | 说明 | 示例 |
|---|---|---|---|
| 作者 | string | 知乎作者名（`@handle`，以搜索接口 `AuthorName` 为准） | `@reed` |
| 文章标题 | string | 原文标题（去掉「- 知乎」后缀） | `cute 之 Layout` |
| URL | url | 规范链接，`https://zhuanlan.zhihu.com/p/<id>`；回答类为 `https://www.zhihu.com/answer/<id>` | `https://zhuanlan.zhihu.com/p/661182311` |
| 对应章节 | string | `主参考`章 + `辅助`章，逗号分隔；风格样本标注用途 | `20`、`12（主），7，23` |
| 引用日期 | date | 首次入册日期（本表统一 2026-09-11；后续新增按实际日期） | `2026-09-11` |
| 图片数 | int | 书稿实际引用/归档的该文图片数；初值 `—`，B.4 图片归档（`figures/zhihu/`）完成后回填 | `—` |

排版规则：一行一篇文章（同作者多篇分行）；URL 使用原文链接（不带 `utm_*` 参数）；`图片数` 回填时以 `figures/zhihu/<author>-<slug>/meta.md` 的登记数一致。

## 1. RFC §15 种子清单逐条核验（61 行）

> 状态口径：**【已核】**= 2026-09-11 经知乎开放平台搜索接口返回该条目（标题/作者/URL 一致）；**【已核-README】**= 仅 LeetCUDA README 第一手清单收录（本次检索未返回，推测为旧文降权）；**【待核】**= 无任何来源。URL 前缀 `https://zhuanlan.zhihu.com/p/`。

| # | 作者 | 文章 | URL | 章 | 状态 | 核验证据（赞/评） |
|---|---|---|---|---|---|---|
| 1 | @竹熙佳处 | 写给大家看的 CuTe 教程：tiled copy | [p/1930389542784964333](https://zhuanlan.zhihu.com/p/1930389542784964333) | 21 | 【已核】 | 赞334 / 更新 本篇文章的后续， 写给大家看的 CuTe 教程：tiled mma 已发布，有兴趣的读者可以在本文读完后查看 动机 |
| 2 | @竹熙佳处 | 写给大家看的 CuTe 教程：tiled mma | [p/1937145378446226159](https://zhuanlan.zhihu.com/p/1937145378446226159) | 22 | 【已核】 | 赞193 / 前序知识 在上一篇  写给大家看的 CuTe 教程：tiled copy 我们介绍了 CuTe 中的  tiled co |
| 3 | @竹熙佳处 | CuTe 教程：Layout Compose & Inverse | [p/1962625273636845008](https://zhuanlan.zhihu.com/p/1962625273636845008) | 20 | 【已核】 | 赞155 / 虽然我们在本文开头说，只需要初中级别的数学知识就能读懂本篇文章，但是考虑到有些读者可能想要进一步挖掘 CuTe 之下的数 |
| 4 | @竹熙佳处 | CuTe 教程：Layout Product & Divide | [p/1971945267294111573](https://zhuanlan.zhihu.com/p/1971945267294111573) | 20 | 【已核】 | 赞98 / 其本意即是来由 SmemLayoutAtom 通过 tile_to_shape 来构建大的 SmemLayoutA，这样 |
| 5 | @竹熙佳处 | CuTe 教程：TMA Copy | [p/2003198909405763007](https://zhuanlan.zhihu.com/p/2003198909405763007) | 13,14,23 | 【已核】 | 赞144 / 更简单的，CuTe 已经提供了足够简洁的上层函数如 make_tma_copy / get_tma_tensor / c |
| 6 | @竹熙佳处 | CuTe 笔记：permutationMNK 参数 | [p/1973526710105419953](https://zhuanlan.zhihu.com/p/1973526710105419953) | 24 | 【已核】 | 赞102 / 我们观察 mma 的执行过程与 thread 中存放 C 的 reg 数据，可以如 Fig.2 所示： 如我们所预期的， |
| 7 | @reed | cute 之 Layout | [p/661182311](https://zhuanlan.zhihu.com/p/661182311) | 20 | 【已核】 | 赞696 / ・第一阶段BLAS的row/col-major + leading dimension描述阶段； ・第二阶段Tensor |
| 8 | @reed | cute Layout 的代数和几何解释 | [p/662089556](https://zhuanlan.zhihu.com/p/662089556) | 20 | 【已核】 | 赞419 / 如图5所示，Layout乘法在集合上可以认为是按照图示的形式进行的，在Layout x Layout的计算中，首先将La |
| 9 | @reed | cute 之 Tensor | [p/663093816](https://zhuanlan.zhihu.com/p/663093816) | 21 | 【已核】 | 赞296 / 前面的章节介绍了 cute Layout以及 Layout的代数和几何解释，Layout描述了数据的排列和底层存储位置关 |
| 10 | @reed | cute 之 MMA 抽象 | [p/663092747](https://zhuanlan.zhihu.com/p/663092747) | 22 | 【已核】 | 赞310 / cute的MMA核心数据结构及其相互关系 cute作为高性能的原语表示和抽象，其直接面向mma实现，对数据和计算进行很好 |
| 11 | @reed | cute 之 Copy 抽象 | [p/666232173](https://zhuanlan.zhihu.com/p/666232173) | 21 | 【已核】 | 赞264 / copy函数是拷贝的实际执行函数，调用该函数会触发线程级别的拷贝的发生，完成线程指令的执行，实现src到dst到数据拷贝 |
| 12 | @reed | cute 之 Swizzle | [p/671419093](https://zhuanlan.zhihu.com/p/671419093) | 12,23 | 【已核】 | 赞574 / 图7中左侧的逻辑矩阵可认为是icol为1的共享内存，其在共享内存对应一个bank，即矩阵的逻辑位置为  ，我们可以通过对 |
| 13 | @reed | cute 之 TMA Descriptor 编码与隐藏的第 21bit | [p/2037200219700449995](https://zhuanlan.zhihu.com/p/2037200219700449995) | 13,23 | 【已核】 | 赞135 / 方法一：创建模板时强制清零 bit21 在调用 cuTensorMapEncodeTiled 后、将 descripto |
| 14 | @reed | cute 之 简单 GEMM 实现 | [p/667521327](https://zhuanlan.zhihu.com/p/667521327) | 24 | 【已核】 | 赞334 / 其中get_slice函数能够将TiledMMA能力根据具体的线程id得到每一个线程所需要的layout信息。利用par |
| 15 | @reed | cute 之 GEMM 流水线 | [p/665082713](https://zhuanlan.zhihu.com/p/665082713) | 24 | 【已核】 | 赞354 / 前面文章我们介绍了cute的 Copy抽象、 MMA抽象，基于这些抽象我们进行了 简单的GEMM实现。从逻辑上而言，cu |
| 16 | @reed | cute 之 高效 GEMM 实现 | [p/675308830](https://zhuanlan.zhihu.com/p/675308830) | 24 | 【已核】 | 赞295 / 此时，我们便可以得到如下主机端代码 其中前三行选择了MMA指令形成了Atom能力，然后定义了对该Atom能力的重复方法（ |
| 17 | @reed | GPU 指令集架构-前言 | [p/686198447](https://zhuanlan.zhihu.com/p/686198447) | 1 | 【已核】 | 赞246 / 2017年的ACM图灵奖颁给了John L. Hennessy和David A. Patterson表彰他们在计算机架构 |
| 18 | @reed | GPU 指令集架构-寄存器 | [p/688616037](https://zhuanlan.zhihu.com/p/688616037) | 1 | 【已核】 | 赞258 / 指令集是软件和硬件沟通的“词汇”，在这个沟通过程中，具体的输入输出参数体现为寄存器，作为芯片上最基础的存储机构，了解它是 |
| 19 | @reed | GPU 指令集架构-Load 和 Cache | [p/692445145](https://zhuanlan.zhihu.com/p/692445145) | 1,11 | 【已核】 | 赞266 / 前文介绍了 NVidia GPU指令集架构中的 寄存器部分，对于一个GPU程序而言，这些寄存器数据最初来自于外部存储结构 |
| 20 | @reed | GPU 指令集架构-浮点运算 | [p/695667044](https://zhuanlan.zhihu.com/p/695667044) | 1 | 【已核】 | 赞138 / 计算机算数是计算机工程的一个重要分支，现代计算类的软件多是构造在浮点运算之上的。了解浮点数和浮点运算对于我们理计算类任务 |
| 21 | @reed | GPU 指令集架构-整数运算 | [p/700921948](https://zhuanlan.zhihu.com/p/700921948) | 1 | 【已核】 | 赞74 / 前文我们介绍了NVidia GPU CUDA Core上的 浮点运算指令，CUDA Core除了提供浮点能力外还提供了整 |
| 22 | @reed | GPU 指令集架构-比特和逻辑操作 | [p/712356884](https://zhuanlan.zhihu.com/p/712356884) | 1 | 【已核】 | 赞93 / 前面文章我们介绍了 NVidia GPU中常用的 浮点运算和 整数运算指令，除了浮点和整数运算外，比特和逻辑操作也是重要 |
| 23 | @reed | GPU 指令集架构-Warp 级和 Uniform 操作 | [p/712357647](https://zhuanlan.zhihu.com/p/712357647) | 1 | 【已核】 | 赞204 / 在 NVidia GPU介绍寄存器时，我们介绍了Uniform寄存器，该寄存器是Warp Level的，即所有的lane |
| 24 | @reed | GPU 指令集架构-程序控制和原子操作 | [p/712357443](https://zhuanlan.zhihu.com/p/712357443) | 1,3 | 【已核】 | 赞88 / 线程退出（EXIT） 由于GPU是一个多线程设备，大部分情况下不同的线程会执行同样的指令，有些时候其中的某些线程并不需要 |
| 25 | @frankshi | CUDA shared memory 避免 bank conflict 的 swizzling 机制解析 | p/4746910252 | **12（主）**,7,23 | 【已核-README】 | 本次搜索未返回，URL 来自 LeetCUDA README 推荐表 |
| 26 | @melonedo | Cute 布局代数实战：除法 | [p/1970274785691936058](https://zhuanlan.zhihu.com/p/1970274785691936058) | 20 | 【已核】 | 赞21 / 提取区域 CuTe 中，division 的大部分用法都是一个固定的模式：给定一个作用于小区域的功能，需要应用于一个较大 |
| 27 | @melonedo | 布局代数实战：Swizzle 自动推导 | [p/1941306442683515068](https://zhuanlan.zhihu.com/p/1941306442683515068) | 12,23 | 【已核】 | 赞370 / 这两天研究手搓GEMM，算swizzle给我算烦了，实在受不了写了个C++库自动计算swizzle布局。  GitHub |
| 28 | @Anonymous | 介绍一个关于 CuTe Layout 变换的小技巧 | [p/2006000375463961170](https://zhuanlan.zhihu.com/p/2006000375463961170) | 20,24 | 【已核】 | 赞79 / 背景 在回家的路上偶然间刷到了 @竹熙佳处 的文章： 这篇文章介绍了CuTe Layout代数中的Compose与Inv |
| 29 | @Anonymous | GEMM 细节分析(一)：ldmatrix 的选择 | [p/702818267](https://zhuanlan.zhihu.com/p/702818267) | 11 | 【已核】 | 赞208 / 关于ldmatrix指令更详细的解释，请参考 PTX ISA。 A/B Layout 上文我们提到——ldmatrix指 |
| 30 | @Anonymous | GEMM 细节分析(二)：TiledCopy 与 cp.async | [p/703560147](https://zhuanlan.zhihu.com/p/703560147) | 21 | 【已核】 | 赞191 / Prologue 在 上一篇文章中，我们分析了从Shared Memory向Registers拷贝矩阵的ldmatrix |
| 31 | @Anonymous | GEMM 细节分析(三)：Swizzle<B,M,S> 参数取值 | [p/713713957](https://zhuanlan.zhihu.com/p/713713957) | 12,23 | 【已核】 | 赞166 / 关于Swizzle的原理，网上已经有大量的技术博客对其进行了解读。此类博客中，大多数都是以一个固定的共享内存逻辑Layo |
| 32 | @可怕的杰瑞 | Cute TiledMMA 简单理解 | [p/1991908850132088026](https://zhuanlan.zhihu.com/p/1991908850132088026) | 22 | 【已核】 | 赞16 / #include <cute/tensor.hpp> #include <iostream> #include <str |
| 33 | @weishengying | cute swizzle | [p/706796240](https://zhuanlan.zhihu.com/p/706796240) | 23 | 【已核】 | 赞95 / #include <thrust/host_vector.h> #include <thrust/device_vect |
| 34 | @水木皇工仔 | 基于 CuTe 理解 swizzle, LDSM, MMA | [p/934430036](https://zhuanlan.zhihu.com/p/934430036) | 21,22 | 【已核】 | 赞30 / #include <thrust/host_vector.h> #include <thrust/device_vect |
| 35 | @Arthur | Bank Conflict 自动消除：Swizzle 技术原理解析 | [p/2042049499263136652](https://zhuanlan.zhihu.com/p/2042049499263136652) | 12 | 【已核】 | 赞43 / 当你写 CUDA kernel 时，是否遇到过这种情况：明明用了 shared memory，性能却不如预期？很可能你遇 |
| 36 | @Titus | cutlass swizzle 机制解析（一）（二） | [p/710337546](https://zhuanlan.zhihu.com/p/710337546) | 12,23 | 【已核】 | 赞181 / 在cutlass GEMM中，swizzle机制主要起到两种作用： ・Thread Block Swizzle：利用局部 ｜ 其余 id: 711398930 |
| 37 | @Titus | GEMM 流水线：single/multi-stage、pipeline | [p/712451053](https://zhuanlan.zhihu.com/p/712451053) | 24,25 | 【已核】 | 赞174 / ・thread block tiling：将一定大小的block从global memory搬运到shared memo |
| 38 | @进击的Killua | cute Swizzle 细谈 | [p/684250988](https://zhuanlan.zhihu.com/p/684250988) | 12,23 | 【已核】 | 赞98 / 这样的话如果逻辑layout直接存储映射到物理结构上，访问一个子矩阵就会出现大量的bank conflict，导致性能急 |
| 39 | @进击的Killua | CUTLASS CuTe 实战（一）基础/（二）应用 | [p/690703999](https://zhuanlan.zhihu.com/p/690703999) | 24 | 【已核】 | 赞181 / 这里罗列下Tensor中最常用的一些API和使用方式。 下面给出tensor的demo代码，其中handle_regis ｜ 其余 id: 692078624 |
| 40 | @朱小霖 | cutlass cute 101 | [p/660379052](https://zhuanlan.zhihu.com/p/660379052) | 20,24 | 【已核】 | 赞542 / 随着英伟达的显卡功能越来越复杂，cuda 也变得越来越难写，大家都要开始学 cutlass 这样的官方工具库了。在 cu |
| 41 | @BBuf | CUTLASS 2.x & 3.x Intro 学习笔记 | [p/710516489](https://zhuanlan.zhihu.com/p/710516489) | 24 | 【已核】 | 赞184 / CUTLASS GEMM模板中有大量可以调节和设置的模板参数，这些参数的设置会高度影响Kernel性能。这个分享将为大家 |
| 42 | @BBuf | Hopper Mixed GEMM 的 CUTLASS 实现笔记 | [p/714378343](https://zhuanlan.zhihu.com/p/714378343) | 24 | 【已核】 | 赞78 / 补充：   ・WGMMA: Hopper Warp Group MMA (矩阵乘累加操作) 解释：这是Hopper架构上 |
| 43 | @66RING | 使用 cutlass cute 复现 flash attention | [p/696323042](https://zhuanlan.zhihu.com/p/696323042) | 25 | 【已核】 | 赞318 / 而cutlass cute则把原本需要手写的thread协同工作的代码抽象封装好了, 如需要协同做拷贝时可以make_t |
| 44 | @shengying.wei | FlashAttention 笔记：tiny-flash-attention 解读 | [p/708867810](https://zhuanlan.zhihu.com/p/708867810) | 25 | 【已核】 | 赞123 / using MMA_Atom_Arch = std::conditional_t<         std::is_sa |
| 45 | @shengying.wei | FlashAttention fp8 实现（ada 架构） | [p/712314257](https://zhuanlan.zhihu.com/p/712314257) | 25 | 【已核】 | 赞189 / 读者可自行打印验证 ALayout 和 BLayout 与 SM89_16x8x32_F32F8F8F32_E4M3_T |
| 46 | @JoeNomad | cutlass block swizzle 和 tile iterator | [p/679929705](https://zhuanlan.zhihu.com/p/679929705) | 12 | 【已核】 | 赞119 / 首先我们先说明一下shared memory是怎么被load的，我们每个threadblock的大小会被等分到每个war |
| 47 | @JoeNomad | cutlass bank conflict free 的 smem layout | [p/681966685](https://zhuanlan.zhihu.com/p/681966685) | 12 | 【已核】 | 赞115 / 开篇 大家好，我是joe，在上一篇我们分析block swizzle和iterator计算global ptr下标的逻辑 |
| 48 | @JoeNomad | cutlass 多级流水线 | [p/687397095](https://zhuanlan.zhihu.com/p/687397095) | 24 | 【已核】 | 赞118 / # before @T.prim_func def simple_compute(A: T.Buffer[(16, 16 |
| 49 | @木子知 | Nvidia Tensor Core 初探 / WMMA API / MMA PTX 编程入门 | [p/620185229](https://zhuanlan.zhihu.com/p/620185229) | 10 / 10 / 11 | 【已核】 | 赞52 / 3.4 Hopper Tensor Core 第四代Tensor Core使用新的8位浮点精度（FP8），可为万亿参数模 ｜ 其余 id: 620766588, 621855199 |
| 50 | @nicholaswilde | CUDA Ampere Tensor Core HGEMM 优化 | [p/555339335](https://zhuanlan.zhihu.com/p/555339335) | 11 | 【已核】 | 赞367 / 测试结果如下图所示，循环展开比不循环展开时又提高了约15 TFLOPS，基本全面超过cublas。在M = N = K  |
| 51 | @Frank Wang | Async Copy 及 Memory Barrier 指令的功能与实现 | [p/685168850](https://zhuanlan.zhihu.com/p/685168850) | 10,13,14,17 | 【已核】 | 赞122 / 异步操作在 CUDA 编程模型中的含义是由某个 CUDA 线程启动的操作，该操作可以像另一个线程一样异步执行。为了保证数 |
| 52 | @紫气东来 | CUDA（一）编程基础 /（二）内存体系 /（三）GEMM 从入门到熟练 | [p/657632577](https://zhuanlan.zhihu.com/p/657632577) | 1 / 1 / 9 | 【已核】 | 赞632 / 接下来分析计算复杂度，假设 的形状是  ， 的形状是  ，则  形状是  。其中主要的部分是  矩阵相乘，根据矩阵乘法的 ｜ 其余 id: 645330027, 654027980 |
| 53 | @紫气东来 | ops(1) LayerNorm / ops(2) SoftMax / ops(5) 激活与残差 / ops(7)(8) self-attention 上下 | [p/694974164](https://zhuanlan.zhihu.com/p/694974164) | 6 / 4 / 3 / 15 | 【已核】 | 赞158 / extern __shared__ float shared[]; // size = 2 * C      names ｜ 其余 id: 695307283, 695703671, 695898274, 696197013 |
| 54 | @白牛 | CUDA 入门的正确姿势：how-to-optimize-gemm | p/478846788 | 9 | 【已核-README】 | 本次搜索未返回，URL 来自 LeetCUDA README 推荐表 |
| 55 | @有了琦琦的棍子 | 深入浅出 GPU 优化系列：gemv 优化 | [p/494144694](https://zhuanlan.zhihu.com/p/494144694) | 8 | 【已核】 | 赞283 / 本篇文章是深入浅出GPU优化系列的第4个专题，主要是介绍如何对gemv算法进行优化。gemv，即矩阵向量乘，即计算一个矩 |
| 56 | @懒蚂蚁呀不嘿 | CUDA element-wise / transpose / reduce 算子详解 | [p/1888630735520391519](https://zhuanlan.zhihu.com/p/1888630735520391519) | 3 / 7 / 2 | 【已核】 | 赞47 / 据此，可以写出使用 float4 进行访存的 elementwise 算子代码： 其中，float4 类型转换被定义为了 ｜ 其余 id: 1899760505733756129, 1905661893739283464 |
| 57 | @DefTruth | 图解：从 Online-Softmax 到 FlashAttention V1/V2/V3 | [p/668888063](https://zhuanlan.zhihu.com/p/668888063) | **15（主）**,4 | 【已核】 | 赞2782 / 这部分还没有完全理解...，我暂且理解成，从QK^T矩阵乘分块的角度看，FA1会导致cutlass gemm产生这种wa |
| 58 | @DefTruth | FFPA(Split-D)：FA2 无限 HeadDim 扩展 | [p/13975660308](https://zhuanlan.zhihu.com/p/13975660308) | 19,26 | 【已核】 | 赞116 / # QK^T MMA ACC F32 + PV MMA ACC F16, NVIDIA 4090 export ENAB |
| 59 | @DefTruth | vLLM Triton Merge Attention States Kernel 详解 | [p/1904937907703243110](https://zhuanlan.zhihu.com/p/1904937907703243110) | 5 | 【已核】 | 赞64 / 0x00 前言 本文介绍vLLM中Triton Merge Attention States Kernel的实现，与 p |
| 60 | @DefTruth | LeetCUDA v3.0 大升级（项目自述） | [p/19862356369](https://zhuanlan.zhihu.com/p/19862356369) | 1,附录C | 【已核】 | 赞945 / 作为Modern CUDA-Learn-Notes， xlite-dev/LeetCUDA又怎么能少了FlashAtte |
| 61 | @DefTruth | WINT8/4-(00)~(03) 快速反量化系列 | [p/657072856](https://zhuanlan.zhihu.com/p/657072856) | **全书行文风格基线**（BOOK_PLAN §4.3，与图解 FA 系列同为风格样本） | 【已核】 | 赞177 / 0x00 前言  关键词：GEMM + Dequantize Fuse、Fast Int8ToFloat16、Weigh ｜ 其余 id: 657070837, 657073159, 657073857 |

## 2. LeetCUDA README 外新增条目（B.2）

> 口径：README 推荐表（约 2026-09 前）未收录、本轮经搜索确认的新文章；下表为与书稿章节直接相关的精选（其余候选见原始检索 JSON）。全部为【新增】。

| # | 作者 | 文章 | URL | 章 | 状态 | 备注 |
|---|---|---|---|---|---|---|
| 1 | @竹熙佳处 | 写给大家看的 CuTe 教程: async pipeline | [p/2024832248386528275](https://zhuanlan.zhihu.com/p/2024832248386528275) | 24,25 | 【新增】 | 赞 87 / 评 8；tiled copy→tiled mma→TMA Copy 之后的新篇，ch24/25 流水线叙述可对标 |
| 2 | @可怕的杰瑞 | Cute TiledMMA 极简理解(二) | [p/1992328297888126320](https://zhuanlan.zhihu.com/p/1992328297888126320) | 22 | 【新增】 | 种子表 (一) p/1991908850132088026 的续篇 |
| 3 | @可怕的杰瑞 | Cute 极简教程 | [p/1991248525925819668](https://zhuanlan.zhihu.com/p/1991248525925819668) | 20-22 | 【新增】 | 导览型，可作 ch20 引入的通俗对照 |
| 4 | @melonedo | 从线性布局的视角看 CuTe 布局 | [p/1937977743569577920](https://zhuanlan.zhihu.com/p/1937977743569577920) | 20 | 【新增】 | 赞 45；线性布局视角，可与「除法和 Swizzle 自动推导」并读 |
| 5 | @melonedo | Swizzle 布局的直观解释和推导 | [p/1935799939990008359](https://zhuanlan.zhihu.com/p/1935799939990008359) | 12,23 | 【新增】 | 与种子表 1941306442683515068（自动推导）互补 |
| 6 | @北漂小菜鸡 | CuTe 教程: Layout | [p/1997872253799519607](https://zhuanlan.zhihu.com/p/1997872253799519607) | 20 | 【新增】 | 另有《Swizzle 工作原理》[p/1996740631024924636](https://zhuanlan.zhihu.com/p/1996740631024924636)（章 12） |
| 7 | @DeclK | CUTLASS CUTE 系列 1-5 | [p/1983922984805742098](https://zhuanlan.zhihu.com/p/1983922984805742098) | 20-24 | 【新增】 | 系列：1 Layout Algebra / 2 MMA&COPY 抽象 / 3 补充材料 / 4 GEMM 核心优化 / 5 Hopper tma&wgmma（另有 8 hpc-ops）；可作 CuTe 篇系统性替代读物 |
| 8 | @水木皇工仔 | 基于 CUTE 的 GEMM 优化【2】—— 高效 GEMM 实现，超越 Cublas 20% | [p/696028389](https://zhuanlan.zhihu.com/p/696028389) | 24 | 【新增】 | 种子表 934430036（swizzle/LDSM/MMA）的续作方向 |
| 9 | @哈密瓜 | cutlass cute 中 GEMM 的理解和使用 | [p/1892237069151085950](https://zhuanlan.zhihu.com/p/1892237069151085950) | 24 | 【新增】 | 工程视角 API 使用 |
| 10 | @皓月争曦 | 动手 Attention 优化 3：理解 Bank Conflict 及 Cutlass Swizzle | [p/9840919069](https://zhuanlan.zhihu.com/p/9840919069) | 12 | 【新增】 | 赞 157；ch12 「bank 冲突 → XOR 置换」推导可选补充 |
| 11 | @Hardy | CUDA Kernel: Tile、Layout 与 Swizzle | [p/2065840973092067294](https://zhuanlan.zhihu.com/p/2065840973092067294) | 12,20 | 【新增】 | 2026 年新文，概念串联完整 |
| 12 | @德布劳钦 | 从一个反直觉的 Case 说起——A Deep Dive into CUDA Shared Memory Bank Conflict | [p/1990531281377833323](https://zhuanlan.zhihu.com/p/1990531281377833323) | 12 | 【新增】 | 反直觉 case 切入，适合作为「常见坑」素材 |
| 13 | @zzk again | DeepGEMM transpose swizzle 推导 | [p/2031778895150720504](https://zhuanlan.zhihu.com/p/2031778895150720504) | 12,23 | 【新增】 | 与 swizzle 参数取值互证 |
| 14 | @韭浪 | Optimal Generic Swizzling | [p/2005574080691185656](https://zhuanlan.zhihu.com/p/2005574080691185656) | 12,23 | 【新增】 | 通用 swizzle 最优化，进阶补充 |
| 15 | @WingEdge777 | [CUDA 优化实战] hgemm - 超越 cuBLAS: Tensor-core、cp.async、ldmatrix | [p/2017717179840214041](https://zhuanlan.zhihu.com/p/2017717179840214041) | 11 | 【新增】 | 与 ch11 手写 HGEMM 同层次，可作对照数据 |
| 16 | @郑思泽 | [施工中] 在 Hopper GPU 上实现 CuBLAS 90% 性能的 GEMM | [p/695589046](https://zhuanlan.zhihu.com/p/695589046) | 13,24 | 【新增】 | Hopper GEMM 工程细节，ch13 备选 |

## 3. RoPE 参考补充（B.5）

> ch7 此前无 RoPE 主参考（BOOK_PLAN §3 标注「RoPE 参考 RFC-B 补充」）。本轮检索 query：`RoPE CUDA 优化 旋转位置编码`、`旋转位置编码 RoPE kernel 实现 CUDA`（2026-09-11）。

| # | 作者 | 文章 | URL | 章 | 状态 | 备注 |
|---|---|---|---|---|---|---|
| 1 | （CUDA 优化实战系列，作者字段未返回） | [CUDA 优化实战] RoPE - 手写算子的作用之 kernel fusion：减少访存次数、减少启动开销的优化技巧 | [p/2011045579652895890](https://zhuanlan.zhihu.com/p/2011045579652895890) | 7（主补充） | 【新增】 | 赞 28；与 ch9/11 已引「CUDA 优化实战」系列（sgemm/hgemm）同源文风，直接对应 ch7 的 RoPE kernel 融合（sin/cos 查表 + 一次访存）主题 |
| 2 | （FlashInfer 系列，作者字段未返回） | FlashInfer 系列: RoPE / RMSNorm / Sampling / SiLU-GLU / 量化 | [p/2069450791727936671](https://zhuanlan.zhihu.com/p/2069450791727936671) | 7（备选） | 【新增】 | 赞 8；工业实现视角，可作「生产环境 RoPE 如何写」对照 |
| 3 | @水木 | 手写 LLM 推理框架-TFFInfer 项目解析(十二)—CUDA 核心算子实现(上): Flash Attention、RoPE 与 RMSNorm | [p/2055037235888649938](https://zhuanlan.zhihu.com/p/2055037235888649938) | 7,16（备选） | 【新增】 | 赞 10；整链算子实现 |
| 4 | （作者字段未返回） | RoPE 旋转位置编码: Meta 与 Hugging Face 两种代码实现详解 | [p/697282166](https://zhuanlan.zhihu.com/p/697282166) | 7（原理备选） | 【新增】 | 赞 27；公式与两种布局（interleaved / half-split）对照 |

**推荐**：#1 作为 ch7「RoPE 与矩阵转置」的知乎侧主参考（kernel 融合角度与本章「bank conflict vs padding」主题互补），#2 作实现对照；原理部分 #4 备查。

## 4. 每章「主参考」URL 覆盖表（B.7 验收项一）

> 对照 BOOK_PLAN §3 各章卡片「主参考」列。✅=至少一条有可核验 URL；论文类主参考（FA2/FA3）不计入知乎覆盖，单列说明。

| 章 | 主参考（BOOK_PLAN §3） | 库存中对应 URL（示例） | 覆盖 |
|---|---|---|---|
| 1 | @reed GPU 指令集系列；@紫气东来 CUDA(一)(二) | [p/686198447](https://zhuanlan.zhihu.com/p/686198447)、[p/688616037](https://zhuanlan.zhihu.com/p/688616037)、[p/692445145](https://zhuanlan.zhihu.com/p/692445145)、[p/695667044](https://zhuanlan.zhihu.com/p/695667044)、[p/700921948](https://zhuanlan.zhihu.com/p/700921948)、[p/712356884](https://zhuanlan.zhihu.com/p/712356884)、[p/712357647](https://zhuanlan.zhihu.com/p/712357647)、[p/712357443](https://zhuanlan.zhihu.com/p/712357443)；[p/645330027](https://zhuanlan.zhihu.com/p/645330027)、[p/654027980](https://zhuanlan.zhihu.com/p/654027980) | ✅ |
| 2 | @懒蚂蚁呀不嘿 reduce 详解 | [p/1905661893739283464](https://zhuanlan.zhihu.com/p/1905661893739283464) | ✅ |
| 3 | @懒蚂蚁呀不嘿 element-wise；@紫气东来 ops(5) | [p/1888630735520391519](https://zhuanlan.zhihu.com/p/1888630735520391519)；[p/695703671](https://zhuanlan.zhihu.com/p/695703671) | ✅ |
| 4 | @紫气东来 ops(2)；@DefTruth 图解 FA（前半） | [p/695307283](https://zhuanlan.zhihu.com/p/695307283)；[p/668888063](https://zhuanlan.zhihu.com/p/668888063) | ✅ |
| 5 | @DefTruth vLLM Merge Attention States | [p/1904937907703243110](https://zhuanlan.zhihu.com/p/1904937907703243110) | ✅ |
| 6 | @紫气东来 ops(1) | [p/694974164](https://zhuanlan.zhihu.com/p/694974164) | ✅ |
| 7 | @懒蚂蚁呀不嘿 transpose；RoPE 参考（B.5 补充） | [p/1899760505733756129](https://zhuanlan.zhihu.com/p/1899760505733756129)；[p/2011045579652895890](https://zhuanlan.zhihu.com/p/2011045579652895890)（本轮新增） | ✅（本轮补齐） |
| 8 | @有了琦琦的棍子 gemv 优化 | [p/494144694](https://zhuanlan.zhihu.com/p/494144694) | ✅ |
| 9 | @白牛 how-to-optimize-gemm；@紫气东来 CUDA(三) | [p/478846788](https://zhuanlan.zhihu.com/p/478846788)（README 源）；[p/657632577](https://zhuanlan.zhihu.com/p/657632577) | ✅ |
| 10 | @木子知 WMMA；@Frank Wang Async Copy | [p/620766588](https://zhuanlan.zhihu.com/p/620766588)；[p/685168850](https://zhuanlan.zhihu.com/p/685168850) | ✅ |
| 11 | @木子知 MMA PTX；@Anonymous GEMM 细节(一)；@reed Load 和 Cache | [p/621855199](https://zhuanlan.zhihu.com/p/621855199)；[p/702818267](https://zhuanlan.zhihu.com/p/702818267)；[p/692445145](https://zhuanlan.zhihu.com/p/692445145) | ✅ |
| 12 | **@frankshi（主）**；@reed Swizzle；@Titus(一)(二)；@进击的Killua；@Anonymous(三)；@JoeNomad block swizzle | [p/4746910252](https://zhuanlan.zhihu.com/p/4746910252)（README 源）；[p/671419093](https://zhuanlan.zhihu.com/p/671419093)；[p/710337546](https://zhuanlan.zhihu.com/p/710337546)、[p/711398930](https://zhuanlan.zhihu.com/p/711398930)；[p/684250988](https://zhuanlan.zhihu.com/p/684250988)；[p/713713957](https://zhuanlan.zhihu.com/p/713713957)；[p/679929705](https://zhuanlan.zhihu.com/p/679929705) | ✅ |
| 13 | @reed TMA descriptor 第 21bit；@竹熙佳处 TMA Copy | [p/2037200219700449995](https://zhuanlan.zhihu.com/p/2037200219700449995)；[p/2003198909405763007](https://zhuanlan.zhihu.com/p/2003198909405763007) | ✅ |
| 14 | @竹熙佳处 TMA Copy；@Frank Wang | [p/2003198909405763007](https://zhuanlan.zhihu.com/p/2003198909405763007)；[p/685168850](https://zhuanlan.zhihu.com/p/685168850) | ✅ |
| 15 | **@DefTruth 图解 Online-Softmax→FA（主）**；@紫气东来 ops(7)(8) | [p/668888063](https://zhuanlan.zhihu.com/p/668888063)；[p/695898274](https://zhuanlan.zhihu.com/p/695898274)、[p/696197013](https://zhuanlan.zhihu.com/p/696197013) | ✅ |
| 16 | @DefTruth 图解 FA；FA2 论文 | [p/668888063](https://zhuanlan.zhihu.com/p/668888063)（FA2 论文 arXiv:2307.08691 非知乎源） | ✅ |
| 17 | @Frank Wang；@竹熙佳处 TMA Copy | [p/685168850](https://zhuanlan.zhihu.com/p/685168850)；[p/2003198909405763007](https://zhuanlan.zhihu.com/p/2003198909405763007) | ✅ |
| 18 | FA3 论文（arXiv:2407.08608）为主 | 知乎侧候选：[p/668888063](https://zhuanlan.zhihu.com/p/668888063)、[p/13975660308](https://zhuanlan.zhihu.com/p/13975660308) | ✅（论文为主，知乎侧为辅） |
| 19 | **@DefTruth FFPA(Split-D)**；ffpa-cuda-understand skill | [p/13975660308](https://zhuanlan.zhihu.com/p/13975660308) | ✅ |
| 20 | **@reed Layout + 代数几何**；**@竹熙佳处 Compose&Inverse、Product&Divide**；@melonedo 除法；@Anonymous 技巧 | [p/661182311](https://zhuanlan.zhihu.com/p/661182311)、[p/662089556](https://zhuanlan.zhihu.com/p/662089556)；[p/1962625273636845008](https://zhuanlan.zhihu.com/p/1962625273636845008)、[p/1971945267294111573](https://zhuanlan.zhihu.com/p/1971945267294111573)；[p/1970274785691936058](https://zhuanlan.zhihu.com/p/1970274785691936058)；[p/2006000375463961170](https://zhuanlan.zhihu.com/p/2006000375463961170) | ✅ |
| 21 | **@竹熙佳处 tiled copy（主）**；@reed Copy 抽象；@Anonymous(二) | [p/1930389542784964333](https://zhuanlan.zhihu.com/p/1930389542784964333)；[p/666232173](https://zhuanlan.zhihu.com/p/666232173)；[p/703560147](https://zhuanlan.zhihu.com/p/703560147) | ✅ |
| 22 | **@竹熙佳处 tiled mma（主）**；@reed MMA 抽象；@可怕的杰瑞；@水木皇工仔 | [p/1937145378446226159](https://zhuanlan.zhihu.com/p/1937145378446226159)；[p/663092747](https://zhuanlan.zhihu.com/p/663092747)；[p/1991908850132088026](https://zhuanlan.zhihu.com/p/1991908850132088026)；[p/934430036](https://zhuanlan.zhihu.com/p/934430036) | ✅ |
| 23 | **@reed Swizzle + TMA descriptor**；@竹熙佳处 TMA Copy；@weishengying；@进击的Killua/@Titus | [p/671419093](https://zhuanlan.zhihu.com/p/671419093)、[p/2037200219700449995](https://zhuanlan.zhihu.com/p/2037200219700449995)；[p/2003198909405763007](https://zhuanlan.zhihu.com/p/2003198909405763007)；[p/706796240](https://zhuanlan.zhihu.com/p/706796240)；[p/684250988](https://zhuanlan.zhihu.com/p/684250988)、[p/710337546](https://zhuanlan.zhihu.com/p/710337546) | ✅ |
| 24 | @reed 简单/高效 GEMM、GEMM 流水线；@朱小霖 cute 101；@进击的Killua 实战 | [p/667521327](https://zhuanlan.zhihu.com/p/667521327)、[p/675308830](https://zhuanlan.zhihu.com/p/675308830)、[p/665082713](https://zhuanlan.zhihu.com/p/665082713)；[p/660379052](https://zhuanlan.zhihu.com/p/660379052)；[p/690703999](https://zhuanlan.zhihu.com/p/690703999)、[p/692078624](https://zhuanlan.zhihu.com/p/692078624) | ✅ |
| 25 | @66RING cute 复现 FA；@shengying.wei tiny-flash-attention；@Titus GEMM 流水线 | [p/696323042](https://zhuanlan.zhihu.com/p/696323042)；[p/708867810](https://zhuanlan.zhihu.com/p/708867810)；[p/712451053](https://zhuanlan.zhihu.com/p/712451053) | ✅ |
| 26 | @DefTruth FFPA 文；ffpa-cuda-understand skill | [p/13975660308](https://zhuanlan.zhihu.com/p/13975660308) | ✅ |

**结论：26/26 章主参考均有可核验 URL（ch18 以论文为主，知乎侧给辅读）。**

## 5. 正文取回状态（B.3）：环境受限说明 + 逐篇清单

### 5.1 环境受限结论（2026-09-11 实测）

| 路径 | 结果 | 证据 |
|---|---|---|
| 文章页 HTML（`zhuanlan.zhihu.com/p/<id>`） | ❌ 403/反爬壳 | 直取仅返回 ~650B 反爬页面（含 `zh-zse-ck` 元标签），无正文 |
| 公开 v4 文章接口（`/api/v4/articles/<id>`） | ❌ 403（需 x-zse-96 签名） | 带浏览器 UA / 首页 cookie 均 403 |
| 集成浏览器 | ❌ 登录墙 | 打开文章页跳转 `account/unhuman`（「请您登录后查看更多专业优质内容」） |
| 专栏列表 API（`/api/v4/columns/<token>/articles`，免签名） | ⚠️ 可用但无 token | 端点本身可访问（不存在 token 返回 404），但目标作者（reed / 竹熙佳处 / DefTruth / frankshi）专栏 token 无法获得：搜索接口不索引专栏页、作者页需登录、`web.archive.org` 在本网络不可达、按作者名猜 token 全部 404 |
| 外部检索（Bing/archive.org 等） | ❌ 网络受限 | `web.archive.org` / `r.jina.ai` 等连接超时 |

**结论**：本环境（无知乎登录态 + 反爬拦截）**无法取回任何文章正文**，因此本轮不产出 `zhihu-analysis/<author>-<slug>.md` 全文存档；已落盘搜索级证据（标题/作者/URL/摘要/赞评），见 `zhihu-analysis/leetcuda-book-refs-2026-09-11.md` 与 `...-raw-...json`。补救路径见 §5.3。

### 5.2 「主参考优先」逐篇状态（18 篇 = 任务清单 17 项 + Titus 流水线）

| # | 目标文章 | URL | 用途 | 全文存档 |
|---|---|---|---|---|
| 1 | @DefTruth 图解 Online-Softmax→FA V1/V2/V3 | [p/668888063](https://zhuanlan.zhihu.com/p/668888063) | ch15 主参考（必取） | 【待核·环境受限】 |
| 2 | @DefTruth FFPA(Split-D) | [p/13975660308](https://zhuanlan.zhihu.com/p/13975660308) | ch19/26 主参考（必取） | 【待核·环境受限】 |
| 3 | @frankshi swizzle 机制解析 | [p/4746910252](https://zhuanlan.zhihu.com/p/4746910252) | ch12 主参考（必取） | 【待核·环境受限】 |
| 4 | @竹熙佳处 tiled copy | [p/1930389542784964333](https://zhuanlan.zhihu.com/p/1930389542784964333) | ch21 主参考 | 【待核·环境受限】 |
| 5 | @竹熙佳处 tiled mma | [p/1937145378446226159](https://zhuanlan.zhihu.com/p/1937145378446226159) | ch22 主参考 | 【待核·环境受限】 |
| 6 | @reed TMA Descriptor 第21bit | [p/2037200219700449995](https://zhuanlan.zhihu.com/p/2037200219700449995) | ch13 主参考 | 【待核·环境受限】 |
| 7 | @reed cute 之 Layout | [p/661182311](https://zhuanlan.zhihu.com/p/661182311) | ch20 主参考 | 【待核·环境受限】 |
| 8 | @reed cute Layout 的代数和几何解释 | [p/662089556](https://zhuanlan.zhihu.com/p/662089556) | ch20 主参考 | 【待核·环境受限】 |
| 9 | @reed cute 之 MMA 抽象 | [p/663092747](https://zhuanlan.zhihu.com/p/663092747) | ch22 主参考 | 【待核·环境受限】 |
| 10 | @reed cute 之 Copy 抽象 | [p/666232173](https://zhuanlan.zhihu.com/p/666232173) | ch21 主参考 | 【待核·环境受限】 |
| 11 | @reed cute 之 Swizzle | [p/671419093](https://zhuanlan.zhihu.com/p/671419093) | ch12/23 主参考 | 【待核·环境受限】 |
| 12 | @reed cute 之 简单 GEMM 实现 | [p/667521327](https://zhuanlan.zhihu.com/p/667521327) | ch24 主参考 | 【待核·环境受限】 |
| 13 | @reed cute 之 GEMM 流水线 | [p/665082713](https://zhuanlan.zhihu.com/p/665082713) | ch24 主参考 | 【待核·环境受限】 |
| 14 | @melonedo 布局代数实战：除法 | [p/1970274785691936058](https://zhuanlan.zhihu.com/p/1970274785691936058) | ch20 辅助 | 【待核·环境受限】 |
| 15 | @白牛 how-to-optimize-gemm | [p/478846788](https://zhuanlan.zhihu.com/p/478846788) | ch9 主参考 | 【待核·环境受限】 |
| 16 | @有了琦琦的棍子 gemv 优化 | [p/494144694](https://zhuanlan.zhihu.com/p/494144694) | ch8 主参考 | 【待核·环境受限】 |
| 17 | @Frank Wang Async Copy | [p/685168850](https://zhuanlan.zhihu.com/p/685168850) | ch10/13/17 | 【待核·环境受限】 |
| 18 | @Titus GEMM 流水线 | [p/712451053](https://zhuanlan.zhihu.com/p/712451053) | ch24/25 | 【待核·环境受限】 |

### 5.3 补救路径（二选一）

**路径 A（推荐，一次登录换全部）**：在集成浏览器登录一次知乎（任一账号）→ 逐篇打开文章页 → 页面快照落盘后用 skill 脚本提取：
```bash
python3 /workspace/dev/vipshop/.github/skills/zhihu/scripts/extract_fulltext.py <snapshot-file> /workspace/dev/vipshop/zhihu-analysis/<author>-<slug>.md
```

**路径 B（批量，需专栏 URL）**：登录后打开任一作者主页取得其专栏 URL（`zhuanlan.zhihu.com/column/<token>`），免签名批量拉全量文章：
```bash
python3 /workspace/dev/vipshop/.github/skills/zhihu/scripts/fetch_column_fulltext.py <专栏URL或token> --out /workspace/dev/vipshop/zhihu-analysis
# 单篇：加 --id <文章id>；标题匹配：--title <子串>
```
> 已知专栏 token 时无需登录，可直接批量取回（本脚本自带公式图片 → `$LaTeX$` 还原）。

## 6. 复现与维护

```bash
CLI=/root/.local/share/zhihu-cli/current/zhihu-cli

# 单篇存在性核验（本表 §1 的做法）
$CLI search zhihu --query "<文章标题>" --count 10

# 本账号收藏夹交叉核对（可选）
$CLI me favorites lists --limit 20
$CLI me favorites items --url-token <UrlToken> --limit 20

# 全文取回（需已知专栏 URL / 登录态）
python3 .github/skills/zhihu/scripts/fetch_column_fulltext.py <专栏URL> --id <文章id> \
  --out /workspace/dev/vipshop/zhihu-analysis
```

维护约定：
1. 新增参考文章 → 追加到 §1/§2 对应表（沿用 §0 字段），并同步 `附录 E` 时引用本表。
2. 每篇取回正文后：①在 `zhihu-analysis/` 落盘 `<author>-<slug>.md`；②本表加「存档路径」列或备注；③图片按 `figures/zhihu/README.md` 规范归档并回填「图片数」。
3. 引用日期一律以首次入册日期为准（本表统一 2026-09-11）。
