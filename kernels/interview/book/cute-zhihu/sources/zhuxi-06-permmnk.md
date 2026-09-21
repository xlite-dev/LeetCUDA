<!--
author: 竹熙佳处
author_id: zhuxi
url: https://zhuanlan.zhihu.com/p/1973526710105419953
column: CuTe 教程专栏
published: 1763341893
updated: 1766583767
fetched: 2026-09-20 (browser js-initialData)
images: 3
-->

# 写给进阶开发的 CuTe 笔记：tiled mma 的 permutationMNK 参数

### 前言

在之前的写给大家看的 CuTe 教程：tiled mma(https://zhuanlan.zhihu.com/p/1937145378446226159)[1]文章中，我们介绍了 permutation MNK （以下记作 PermMNK）作为 tiler 的作用，但是并没有介绍其真正 “permutation” 的功能。

在文章中我们提到，在 cutlass repo 的 issue: What is PermutationMNK in TiledMMA in CUTLASS 3.4 changes?(https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/cutlass/discussions/1345) [2] 中，CuTe 的作者 Cris Cecka(https://link.zhihu.com/?target=https%3A//github.com/ccecka) 进行了解答，其可以做到让每个 thread 做 mma 时可以更灵活的指定自己所负责的数据。但是笔者觉得，其介绍仍然延续了 CuTe 官方文档类似的问题：缺少足够的上下文以及一步步 reasoning 的过程，导致理解起来仍然非常不直观。在本文中，我们尝试用更直白的语言，帮助读者一同理解这个参数的含义以及用法。

值得注意的是，虽然 permutation 的功能在绝大多数使用 CuTe 的场景下都用不到，但是其在一些特殊场景下可能能有奇效。因此本系列文章主要面向 cute 进阶开发者收录于写给进阶开发者的 CuTe 教程(https://zhuanlan.zhihu.com/column/c_1973695125898150444)[3]，作为 写给大家看的 cute 专栏(https://zhuanlan.zhihu.com/column/c_1946580149119202264) [4]的补充，主要用来解释 CuTe 中不易理解但具有潜力的细节。读者朋友们有感兴趣的内容可以在评论区提出，我们在后续的文章中进行选择梳理。

### 动机

我们以 ampere-style fp16 mma 为例，考虑一个场景：一个 warp 完成 2 次 16x8x16 的 mma，得到 16x16 的 C，然后 store 到 global mem。在一些场景下，例如 M & N 很大，K 很小的场景下，这时候 STG 指令的发射可能构成瓶颈，因此我们希望 store 的时候，用到更大字长的 STG，如 STG.128 来提升 store 的效率。

但是如我们在 tiled mma 篇章中介绍的，mma 指令决定了我们的 thread 存放的数据最多只能写出连续的两个 value，如 Fig.1 所示，如果是 float 数值，最大能用的 STG 指令只能是 STG.64。

![img-1](https://pic3.zhimg.com/v2-c7ed6863853970d9462dc6f104251b60_r.jpg)
*Figure.1 16x8x16 FP16 mma 指令 C 矩阵对应的 thread-value 映射关系。可以看到每个 thread 会得到 C 矩阵中特定位置的数据*

这时候，我们希望 thread0 要是能得到 C 矩阵中的连续的 4 个 value 就好了。这其实就是 permutationMNK 被提出的动机。类似的情况还包含 w4a8 混合精度 gemm，fp8 attention（QK 乘之后的 P 矩阵作为 fp8 mma 输入，需要 4 个数连续），也可以尝试用这个方法。

### 如何使用 PermMNK

在 issue [2] 中，Cris Cecka(https://link.zhihu.com/?target=https%3A//github.com/ccecka)以 fp64 mma 来作为实例以方便可视化，与上述我们给的例子本质是一样的。为展示方便，我们仍然以其在该 issue 中例子上进行分析。首先，在常规的用法中，我们指定 permMNK 为一个普通的 tile 时，对应代码如下：

TiledMMA tiled_mma = make_tiled_mma(SM80_8x8x4_F64F64F64F64_TN{},
                                     Layout<Shape<_1,_1,_1>>{},     // AtomLayout
                                     Tile<_8,_16,_8>{});            // Tiler

我们观察 mma 的执行过程与 thread 中存放 C 的 reg 数据，可以如 Fig.2 所示：

![img-2](https://pic1.zhimg.com/v2-88922c8b3f65fa81f50249e6536d9e60_r.jpg)
*figure2. 常规调用 2 次 mma，thread0 拿到的 C 矩阵数据对应坐标为{(0, 0), (0, 1), (0, 8), (0, 9)}，无法作为连续的数据写出*

如我们所预期的，每个 thread 拿到的数据在写出时是不连续的。原因即是我们遵循 mma 的标准做法：对 B 矩阵进行了连续的读取，并做了两次连续的 mma。

但是，如果我们可以**让两条 mma 指令，不处理 A & B 矩阵在物理上连续的数据块，而是交错式的取数据，计算的结果也交错写出到 C 矩阵中；然后再执行第二条 mma，依旧是读交错的位置，得到的结果交错写入 C reg 矩阵。**这个过程如 Fig.3 所示：

![img-3](https://pica.zhimg.com/v2-d222aa0567503601b91ba3d4a0219e6a_r.jpg)
*figure2. 交错调用 2 次 mma，thread0 拿到的 C 矩阵数据对应坐标为 {(0, 0), (0, 1), (0, 2), (0, 3)}，可以作为连续的数据写出*

我们发现，通过这样计算出来的 C 矩阵，各个 thead 能够拿到连续的 4 个数，STG.128 就可以执行了！那么这个写法，对应到 permutationMNK 的参数如下，其就是在 N 维度做了重排：

TiledMMA tiled_mma = make_tiled_mma(SM80_8x8x4_F64F64F64F64_TN{},
                                     Layout<Shape<_1,_1,_1>>{},     // AtomLayout
                                     Tile<_8,                       // Permutation on M, equivalent to 8:1, identity
                                     Layout<Shape <_2,_4,_2>, 
                                             Stride<_1,_4,_2>>, // Permutation on N, size 16
                                      _8>{});                   // Permutation on K, equivalent to 8:1, identity

那么再考虑，用于 s2r copy 的 LDSM 指令在这个时候是否还可以使用？笔者认为是可以的，因为虽然我们在 N 方向上有一些交错访问，但是交错的粒度，都是连续的 2*16 fp16 元素，因此 LDSM 仍然可以使用。不过，由于从 shared mem 中 load 的数据不再是紧密排列的了，常规的 swizzle BMS 参数需要重新调整一下才能做到 bank-conflict-free。

进一步的，我们思考，想要完成上述功能，除了 permMNK 之外，原本的 g2s -> s2r -> gemm 的 mainloop 代码还需要做什么改动吗？笔者认为应该是不需要的，我们观察 permMNK 在 tiled mma 的具体实现，其是通过 logical_divide 改动了 TV-layout，其作用会自然而然通过 partition 和 retile 的逻辑传递到 s2r tiled copy 上。

thrfrg_C(CTensor&& ctensor) const
{
  CUTE_STATIC_ASSERT_V(rank(ctensor) >= Int<2>{});
  // Reorder the tensor for the TiledAtom
  auto t_tile = make_tile(permutation_mnk<0>(),
                          permutation_mnk<1>());
  auto t_tensor = logical_divide(ctensor, t_tile);                 // (PermM,PermN)

  // ...
}

为了验证 "permMNK 修改并不影响 ldmatrix 以及 mainloop 流程" 的想法，我们给出验证代码，有兴趣的读者可以尝试复现：

https://github.com/chrisHuxi/CuTe-coding-toolings/blob/feature/permMNK_tiled_mma/permMNK_example/perm_tiled_mma.cu

此外，有读者朋友@6666(https://www.zhihu.com/people/940468f485e92a6af2c696581a440e60)在评论区也给出了 CuTeDSL 的实现，发现在设置 permMNK 后再调用 ldmatrix 会产生报错，笔者初版判断是 CuTeDSL 本身的适配问题，待我们研究清楚后再更新结论到本文中，有熟悉 CuTeDSL 的读者也欢迎在评论区讨论。

### 总结

本文中，我们介绍了 Tiled mma 中 permutationMNK 参数的作用。其通过修改 TV-layout，做到对每个 thread 负责输出 C 矩阵的位置进行微调，从而赋予了 load / store 时更多的灵活性。从这个例子我们可以看到，CuTe 的抽象确实足够完备，但高度的抽象也隐藏了诸多细节，确实不易理解，希望我们的梳理能够帮助大家进一步加深对 CuTe 的理解。

### Reference

[1] 写给大家看的 CuTe 教程：tiled mma(https://zhuanlan.zhihu.com/p/1937145378446226159)

[2] [QST] What is PermutationMNK in TiledMMA in CUTLASS 3.4 changes?(https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/cutlass/discussions/1345)

[3] 写给进阶开发看的 CuTe 教程(https://zhuanlan.zhihu.com/column/c_1973695125898150444)

[4] 写给大家看的 CuTe 教程(https://zhuanlan.zhihu.com/column/c_1946580149119202264)
