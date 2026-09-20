<!--
author: 竹熙佳处
author_id: zhuxi
url: https://zhuanlan.zhihu.com/p/1971945267294111573
column: CuTe 教程专栏
published: 1764252521
updated: 1766583755
fetched: 2026-09-20 (browser js-initialData)
images: 9
-->

# 写给大家看的 CuTe 教程: Layout Product & Divide

### 动机

在之前的文章中，我们梳理了 Layout 的两种代数方法：**Compose**​ 与**Inverse **[1]。本文我们将继续介绍 Layout 代数中另外两个重要的代数变换：**Product**​ 与**Divide**。

在介绍这两个操作之前，我们先思考一下 CuTe 引入Product​ 和Divide​ 的目的是什么。在介绍 Tiled Copy 的文章[2]中，我们提到了需要构建 Thread 与 Data 之间的映射关系。我们以 4 个 Thread 组成的 thread block 拷贝 4×4 数据块为例，如 Fig.1 所示：

![img-1](https://pic1.zhimg.com/v2-793572c56162b01689b81f4cca9a3098_r.jpg)
*Figure.1 tiled copy 中 thread 与 data 的映射关系。以 4 thread copy 4x4 data 为例，每个 thread 大字长访存，单次 copy 2 个 data，并且在行方向上重复 2 次。*

我们编写 CUDA Kernel 来完成 tiled copy 过程的过程，本质上是在确定一个 Thread Block 每次所访问的数据范围，然后通过重复扩展，直至不重不漏的覆盖整个数据块。

也就是说，我们实际上是将 “单个 Thread Block 访问行为” 按照某种 “模式” 进行重复。读过本专栏其他文章的读者会发现，这种 “重复” 的行为在我们介绍 Tiled Copy 和 Tiled MMA[3] 中曾反复出现，为此我们提到过一个 CuTe 中的重要观点：**Atom 之外，皆是重复**。

现在我们将“模式”与“重复”的理念引入到 CuTe Layout 中，就可以发现：我们可以指定一个小 Layout 作为一种模式，将这个模式**重复**多次可以形成更大的 Layout，这就是今天要介绍的**Layout Product**。

除了将 Layout 扩大，自然的，我们也需要具备从一个大 Layout 中按照某种模式抽取出一个小 Layout 的能力。例如，在访问大数据块时，我们需要让每个 Thread Block 知道自己每一轮要访问的数据。也就是说，我们指定一个小 Layout 作为模式，将大 Layout **拆解**成若干部分，这即 **Layout Divide**​。

我们观察 CuTe 上层的函数，如 partition / local_tile / tile_to_shape  等，其实都是 product 以及 divide 的变体。更进一步，make tiled copy / mma 等核心函数中，product & divide 也是被反复用到，并配合我们在上一篇文章中提及的 compose & inverse，构成了 CuTe 体系中千变万化的代数变换。

值得一提的是，如果说 compose 和 inverse 还可以借助数学中的函数复合以及逆函数的概念进行理解，product 和 divide 的思想则更是总结为一个图形上的理解，因此接下来我们更多借助图像的方式，尽可能形象的来说明这两个 CuTe 中最难说明白的代数变换。

### Layout 的乘法 Product

Product的目的是将一个 layout 扩大。 给定：

$\text{LayoutC} = \text{product}(\text{LayoutA}, \text{LayoutB})$

其计算过程可以描述为：

- 将 $\text{A}$ 补全为一个连续紧密的 $\text{A}'$；
- 参考 $\text{A}$ 的 “模式”，从 $\text{A}'$ 中选择出一系列 layout 集合，我们将这些集合记为 $\text{A}$**-like Layouts**。
- 按照**Col-Major 方式**访问 $\text{B}$，得到一个 offset 的序列，我们按照  $\text{B}$  的 offset 定义的顺序，将 $\text{A}$-like Layouts 按照一定的规律排列在一起，形成最终的 $\text{C}$。

这里的“一定的规律”，则是 CuTe 中定义的 logical_product / blocked_product / raked_product / zipped_product / tile_product 等 Product 函数具体决定的。

例如，如果把每个$\text{A}$-like Layouts视为 $\text{C}$ 的一列，按顺序排列在一起，那么就是 CuTe 中的 **logical_product**。

我们以图形的方式解释 logical_product 的过程，如 Fig.1 所示：

![img-2](https://pica.zhimg.com/v2-d4aa979fd26746f7c28782f56e6bdd12_r.jpg)
*Figure.1 logical_product 计算过程：首先将 A 补全到紧密 layout，再从紧密 layout 中找出 A-like layouts，最后按照 B 的 col-major 顺序取 A-like layouts，按列排在一起。*

其他 Product 变体函数的基本语义与 logical_product 是一致的，各个函数的区别只是在最后形成 $\text{C}$ 时将 shape 以及 stride 维度给调换了。例如，除了 logical_product，在 cutlass 中比较常见的 product 函数还有：

- **blocked_product**：其填充 $\text{A}$ 并提取 $\text{A}$-like Layouts 的过程和 logical_product 的过程一致。最后构成 $\text{C}$ 时，不再将 $\text{A}$-like Layouts 排成一列，而是尝试把 $A$-like Layouts **作为一个 “block” 聚集在一起**，这个过程如 Fig.2 所示。事实上，这种排列方式也和我们直觉上的 layout 重复更契合，所以其在 cutlass 中被广泛使用。

![img-3](https://pic2.zhimg.com/v2-da7121b2baa99c6a4e21f9648fa677d7_r.jpg)
*Figure.2 blocked_product 计算过程：首先将 A 补全到紧密 layout，再从紧密 layout 中找出 A-like layouts，最后按照 B 的 col-major 顺序取 A-like layouts，按 block 形式排在一起。*

- **raked_product**：其过程为 blocked_product 类似，只是在最后按 “block” 聚集的时候，是优先排成类似 B 的形式。如 Fig3. 所示。

![img-4](https://pica.zhimg.com/v2-5a11ff9f59a5248b308c9360ceb500ce_r.jpg)
*Figure.3 raked_product 计算过程：首先将 A 补全到紧密 layout 再从紧密 layout 中找出 A-like layouts，再按照 B 的 col-major 顺序取 A-like layouts，按 block 形式排在一起，最后进行一次重排，形成 raked 形状。*

直觉上，raked_product类似在 blocked_product 的基础上，把 $\text{C}$ 做了一次一个隔一个的排列（rake 中文即是耙子，类似耙钉的排列，还怪形象的）。当然归根溯源，Raked 这个命名，实际上是来源于并行计算理论中的 cyclic 分布，通常用来均衡各个处理器上的任务数，如 Fig.4 [4]。

![img-5](https://pic2.zhimg.com/v2-4b99509d728e51756cc738fbc0bced37_r.jpg)
*Figure.4 并行计算理论中 block 分布以及 cyclic 分布。*

此外，product 还支持多维的形式，即：

$\text{LayoutC} = \text{product(LayoutA, (LayoutB}_0\text{, LayoutB}_1\text{, ...))}$

其中 $ \text{(LayoutB}_0,  \text{LayoutB}_1 \text{, ...)}$ 不再是一个单独的 layout，而是一个多个 layout 组成的 tuple，通常被称为 **Tiler**。

借助这种 Tiler 形式，可以做到 $\text{B}_i$ 之间不相互耦合。笔者感觉这类场景对于 product 来说其实没有实际应用的场景，在 divide 部分用的比较多。因此，我们看到CuTe 文档中虽然提到了几种 tiler 相关的 product 函数，其实在 cutlass 并没有被用到。出于严谨，我们列出各函数对应的区别，读者有个印象即可：

Layout Shape : (M, N, L, ...)
Tiler Shape  : <TileM, TileN>

logical_product : ((M,TileM), (N,TileN), L, ...)
zipped_product  : ((M,N), (TileM,TileN,L,...))
tiled_product   : ((M,N), TileM, TileN, L, ...)
flat_product    : (M, N, TileM, TileN, L, ...)

回顾 Layout Product 的过程，其实就是**在尝试将 A layout 的模式进行重复，以覆盖一个紧密排列的空间**。我们其实能感觉到，这和我们平时写 cuda 代码的时候是很类似的。以 tiled copy 为例，首先我们规定了一个 thread block 选取的 data tile 作为一个 “模式”，然后我们将这个模式重复，直到填满整个紧密排列的内存空间，不重不漏的完成所有数据的 copy。

### Product 用法案例：tile_to_shape

Product 在 cutlass 中被广泛用于 layout 构建，以 blocked_product 为例，其有一个常用的变体：**tile_to_shape** 函数，区别在于 blocked_product 关注的是 layoutA 被重复多少次，tile_to_shape 关注重复后能达到多大的 shape。其具体代码如下：

template <class Shape, class Stride,
          class TrgShape, class ModeOrder = LayoutLeft>
CUTE_HOST_DEVICE constexpr
auto
tile_to_shape(Layout<Shape,Stride> const& block,
              TrgShape             const& trg_shape,
              ModeOrder            const& ord_shape = {})
{
  CUTE_STATIC_ASSERT_V(rank(block) <= rank(trg_shape), "Rank of layout must be <= rank of target shape.");
  constexpr int R = rank_v<TrgShape>;

  auto padded_block = append<R>(block);

  auto block_shape  = product_each(shape(padded_block));
  auto target_shape = product_each(shape(trg_shape));

  // Assert proper division
  if constexpr (is_static<decltype(target_shape)>::value) {
    CUTE_STATIC_ASSERT_V(evenly_divides(target_shape, block_shape),
                         "tile_to_shape: block shape does not divide the target shape.");
  }

  auto product_shape = ceil_div(target_shape, block_shape);

  return blocked_product(padded_block, make_ordered_layout(product_shape, ord_shape));
}

通过 tile_to_shape，我们可以拿一个小的 atom layout 来填满我们目标 shape 来获得的新 layout，这样我们在 atom layout 上做的变换，就能通过重复作用到整个目标 layout。例如，我们使用 shared memory swizzle 的时候，通常会写出如下代码：

using SmemLayoutAtom = decltype(composition(
      Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
      make_layout(make_shape(Int<8>{}, Int<64>{}),
                  make_stride(Int<64>{}, Int<1>{}))));
using SmemLayoutA = decltype(
      tile_to_shape(SmemLayoutAtom{},
                    make_shape(Int<128>{}, Int<64>{}, Int<kStage>{})));

其本意即是来由 SmemLayoutAtom 通过 tile_to_shape 来构建大的 SmemLayoutA，这样只要我们保证 atom layout 中的 swizzle 能做到 bank-conflict-free，就能借由 product 扩散到整个 shared memory 的 bank-conflict-free。

### Layout 的除法 Divide

如果说 Product的逻辑是：通过重复一个 “模式” 来扩大 一个 Layout。那么，Divide 的逻辑就是：通过选择 “模式” 来**拆解**一个大 Layout。

我们仍然以 Tiled Copy 为例：在乘法的例子中，我们知道一个 Thread block 负责的 Layout，可以通过重复来覆盖更大的 Layout。

在 Thread block 之下，我们还需要具体让每一个 Thread 知道自己负责的数据的坐标。这时候，就需要我们对 Thread block 负责的 Layout 进行**拆解**。

因此，**Layout Divide** 的计算过程即是：

给定一个大的待分解的 $ \text{LayoutA}$ ，然后给出一个基础的 “模式” $\text{LayoutB}$ ，二者做 divide：

$  \text{LayoutC = divide(LayoutA, LayoutB)}$

代表我们通过在 $\text{A}$ 上去找 $\text{B}$** 的 offset **决定的 $\text{B}$ **-like** **layouts**，最后再把这些 $\text{B}$-like layouts 组合在一起，得到 $\text{C}$，从而**确保原本的 **$\text{A}$** 被不重不漏地拆分**。

以 **logical_divide** 为例，我们用图示来描述这个过程，如 Fig.4：

![img-6](https://pic1.zhimg.com/v2-38efad739051563f464eced8de143c7a_r.jpg)
*Figure.4  logical_divide 计算过程：按照 B col-major 访问得到 offset，映射到 A 中找出 B-like layouts，然后将所有 B-like layouts 按列排在一起。*

值得一提的是，对 $\text{A}$ 做 $\text{divide}$ 得到的 $\text{C}$ ，和 $\text{A}$ 的 Size 是一样的，只是映射关系发生了改变。这一点在直觉上和我们在初等数学中学到的 $\text{A / B}$ 是不同的。我们可以把它类比成整数除法，但是将商和除数都保留。如：$10 / 5 = (5, 2)$。

与 Product 类似，CuTe 中也定义了多种不同的 divide 函数，包括 logical_divide、zipped_divide、flat_divide、tile_divide。它们之间的区别仍然在于：最如何选择和排列那些被拆解出来的 $\text{B}$**-like** **Layouts**。

与 Layout Product 不同，divide 的一维形式我们用得相对较少。相反，它的**多维形式**反而更容易理解；在多维形式中，我们通常把 $\text{LayoutB}$ 指定为一个**Tiler**，即：

$\text{LayoutC} = \text{divide(LayoutA, (LayoutB}_0\text{, LayoutB}_1\text{, ...))}$

在这种形式下，我们可以将 $\text{A}$ 的 **Shape **和我们希望选择的“模式” 完全解耦开，真正做到我们想要指定行或者列要按照什么 “模式” 去选择数据。

什么意思呢，我们来做个对比：

在一维的场景中，如果我们给定的 $\text{A}$ 的 shape 发生了改变，这时即使 $\text{B}$ 保持完全一致，我们按照 $\text{B}$ 的 offset 去找出来的 $\text{B}$-like Layouts 也会随之发生变化。而这，正是我们不希望看到的。

我们以 $\text{divide}$ 操作中最常用的 $\text{zipped_divide}$ 为例。当同一个 $\text{B}$ 分别作用在具有不同 $\text{shape}$ 的 $\text{A}$ 和 $\text{A}'$ 上时，其结果的差异性，如 Fig.5 中所展示的。

![img-7](https://pic2.zhimg.com/v2-10f03a0698da0708c57cee5e4ad5b34b_r.jpg)
*Figure.5  一维 zipped_divide 计算过程，相同 layoutB 作用在不同 shape layoutA 上的对比，我们选择的 “模式” 发生了改变。*

作为对比，我们讨论二维场景下的情况。二维 divide 通常会被表达为以下形式：

$\text{LayoutC = divide(LayoutA, TilerB)}$

其中 $\text{TilerB}$ 是一个由 2 个彼此独立的 $\text{Layout}$ 所构成的 $\text{tuple}$：

$\text{TlerB} = \text{(LayoutB}_0\text{, LayoutB}_1\text{)}$

在这种情况下，$\text{divide}$ 的过程随之调整为：将 $\text{A}$ 的每一个维度，分对这个 $\text{TilerB}$ 中的对应 $\text{LayoutB}_i$ 执行 $\text{divide}$ 操作，以完成数据的选取。

通过这种机制，我们可以确保对 $\text{TilerB}$ 的定义不受到 $\text{LayoutA}$ 的 $\text{shape}$ 影响。即，当保持 $\text{TilerB}$ 不变，并将其作用于两个具有不同 shape 的 $\text{LayoutA}$ 和 $\text{LayoutA}'$ 上时，其选择的模式便能保持一致。具体的计算过程，如 Fig.6 所示。

![img-8](https://pic2.zhimg.com/v2-331923ea603956aa0a4919bae292c19d_r.jpg)
*Figure.6 二维 zipped_divide 计算过程，相同 layoutB 作用在不同 shape layoutA 上的对比，可以看到我们选择的 “模式” 具有一致性。*

因此，我们可以总结一个使用上的惯例：$\text{divide}$** 操作通常侧重于使用多维形式，其中以二维形式最为常见。而 **$\text{product}$** 操作则多使用一维形式。**

除了前面提到的 $\text{zipped_divide}$ 之外，常见的多维 $\text{divide}$ 变体还包括：

$\text{logical_divide}, \text{flat_divide}, \text{tile_divide}$

这些函数在根据 $\text{TilerB}$ 从 $\text{layoutA}$ 中选取数据的逻辑上是完全一致的。

它们之间的区别主要体现在最后组成 $\text{layoutC}$ 的阶段：各个函数会通过调整 $\text{layoutC}$ 元素的顺序，或者调整括号的加减，从而影响最终输出 $\text{layoutC}$ 的 $\text{shape}$。具体对比，如下表所示：

Layout Shape : (M, N, L, ...)
Tiler Shape  : <TileM, TileN>

logical_divide : ((TileM,RestM), (TileN,RestN), L, ...)
zipped_divide  : ((TileM,TileN), (RestM,RestN,L,...))
tiled_divide   : ((TileM,TileN), RestM, RestN, L, ...)
flat_divide    : (TileM, TileN, RestM, RestN, L, ...)

### Divide 用法案例：local_partition & local_tile

我们注意到，如果我们将 Fig.6 中的 $\text{TilerB}$ 中的 $\text{(LayoutB}_0\text{, LayoutB}_1\text{)}$ 设定为两个紧密排列的 layout，那么我们就能发现 $\text{zipped_divide}$ 之后的 $\text{layoutC}$，**每一列就是 **$\text{layoutA}$** 中连续的小一块元素，每一行就是 **$\text{layoutA}$**  中块内相同位置的元素，**如 Fig.7 所示：

![img-9](https://picx.zhimg.com/v2-e00eb8e8e52d3056338cd99782a99ac1_r.jpg)
*Figure.7 zipped_divide 的使用案例。可以看到对结果 layoutC 分别做行方向上的 slice，可以得到 A中连续的小一块元素；以做列方向上的 slice，可以得到每一块块内相同位置的元素。*

对于一些实际使用过 CuTe 的读者来说，可能立刻会联想到一个熟悉的场景：“这个功能，不就和我们在 $\text{tiled copy}$ 操作中用到的 $\text{local_tile}$ 和 $\text{local_partition}$ 的功能是一样的吗？”

没错，这并非巧合。实际上当我们去观察 $\text{local_tile}$ 和 $\text{local_partition}$ 这两个函数（也叫 inner_partition 和 outer_partition）的具体内部实现时，我们会发现，它们正是通过 $\text{zipped_divide}$ 来完成核心的数据重排和分块逻辑的。divide 的类似用法遍布整个 CuTe 的代码中，可以说是最常用的代数变换了。

// ...
local_tile(Tensor    && tensor,
           Tiler const& tilewowozzZZ,   // tiler to apply
           Coord const& coord)   // coord to slice into "remainder"
{
  return inner_partition(static_cast<Tensor&&>(tensor),
                         tiler,
                         coord);
}

// ...
inner_partition(Tensor    && tensor,
                Tiler const& tiler,
                Coord const& coord)
{
  auto tensor_tiled = zipped_divide(static_cast<Tensor&&>(tensor), tiler);
  constexpr int R0 = decltype(rank<0>(tensor_tiled))::value;
  // ...
}

// ...
local_partition(Tensor                     && tensor,
                Layout<LShape,LStride> const& tile,    // coord -> index
                Index                  const& index)   // index to slice for
{
  static_assert(is_integral<Index>::value);
  return outer_partition(static_cast<Tensor&&>(tensor),
                         product_each(shape(tile)),
                         tile.get_flat_coord(index));
}

// ...
outer_partition(Tensor    && tensor,
                Tiler const& tiler,
                Coord const& coord)
{
  auto tensor_tiled = zipped_divide(static_cast<Tensor&&>(tensor), tiler);
  constexpr int R1 = decltype(rank<1>(tensor_tiled))::value;
  // ... 

}

### 总结

至此，我们已经完整地介绍了 $\text{layout}$ 代数中的核心操作：$\text{product}$ 和 $\text{divide}$。结合我们在上一篇中讨论的：$\text{compose}$ 和 $\text{inverse}$，事实上，这四个基本操作就构成了$\text{CuTe layout}$ 代数的核心骨架。

阅读 Cutlass 代码的过程中，你会发现 $\text{layout}$ 代数会以各种变体形式反复出现。除了我们本文中提到的 $\text{tile_to_shape}$、$\text{local_partition}$、$\text{local_tile}$ 等，更常见的情况是，一个实际的功能函数，往往是多个代数变换步骤的组合。

例如，在 $\text{tiled copy}$ 中，我们能看到 $\text{raked_product}$、$\text{inverse}$ 和 $\text{compose}$ 的联合使用。在 $\text{tiled mma}$ 中，则用到了多次的 $\text{logical_divide}$、$\text{zipped_divide}$ 和 $\text{compose}$ 等操作。

熟练掌握这些基本的 $\text{layout}$ 代数操作，能够让我们对 $\text{CuTe}$ 这套工具的使用更加得心应手。一旦融会贯通，你就能够在理解高性能算子实现的道路上，觉醒一种**“通透世界”**的能力，清晰地看透复杂数据排布背后的逻辑，成为真正的高性能算子大师。

本系列其他文章导引：

写给大家看的 CuTe 教程：tiled copy

写给大家看的 CuTe 教程：tiled mma

写给大家看的 CuTe 教程：Layout Compose & Inverse
