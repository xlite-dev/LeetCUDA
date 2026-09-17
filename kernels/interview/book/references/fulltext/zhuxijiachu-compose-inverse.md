# 写给大家看的 CuTe 教程：Layout Compose & Inverse

URL: https://zhuanlan.zhihu.com/p/1962625273636845008

（全文经浏览器取回，作者保留所有权利；本书仅作学习引用与写作风格参照）

目录
动机

在本专栏此前的两篇文章中，我们梳理了 CuTe 中的两大模块：tiled copy 和 tiled mma 以及对应的 thread-value 的映射关系。

以这两大模块为基础，实际上就可以完成调用 tensor core 来完成矩阵计算的流程了。然而我们知道，只会简单的使用这些函数并无法做到自定义高性能 CuTe kernel。

例如，在 flash atten v2/v3 中，Q @ K.t 的结果 S 矩阵，首先需要做一个 online softmax 得到 P，然后以 P 为输入完成 P @ V。其中可以发现，P 的数据分布是按照 mma-C 矩阵来排列的，而紧接着的 P@V 又要求其作为 mma 的输入按照 mma-A 矩阵的数据分布来排列。这就需要我们对 layout 进行变换，其目的是为了让我们的数据按照我们希望的形式进行排列，以适应不同的接口要求（cute::gemm）。

另一个例子是，w4a8 混合精度 gemm，由于 load 数据(int4) 与实际计算所需的计算 int8/fp8 数据类型不一致，load 数据之后的 layout 也需要变换；更多的例子还包含，2:4 稀疏 gemm，implict im2col gemm 等等，也都需要对 layout 变换有一定认知才能实现。

更细节的，CuTe 的所有上层接口如 make_tiled_copy/mma，partition，retile 等等逻辑，都来自 layout 的变换，了解 layout 变换的逻辑，不仅有助于我们使用 CuTe 的时候更得心应手，实际也具备将这样一套代数系统迁移到任意类 gpu 芯片上的潜力。

而 NV 的每一代芯片在 CuTe 这套表达体系下依旧能适配的很好。那么对于国产芯片，其能适配到什么程度呢？或许只有在我们更深入理解 CuTe layout 代数的原理之后，才能回答这个问题。

那么，是时候来探索一下 CuTe 体系中最神秘的 layout 代数了。虽然我们把 layout 说的如此重要，但是相信我，正如很多宏观物理现象最终可以被表述为牛顿三公式一样，cutlass 庞大的代码规模下，其蕴含的 layout 代数，只需要高中级别的数学知识即可理解。

## Layout 的定义 & 作用

NV GPU 体系下，gemm kernel 常常需要对 Tile 数据进行读写，我们访问数据时，实际上是通过 base ptr 加一个位置偏移来获得的。因此最基础版本的 layout，实际上就是为了解决逻辑上的 (x, y) 坐标转变为一个物理的指针的偏移位置这样一个问题。

带着这样的目的，我们给出 Layout 的定义：Layout 由 shape 和 stride 构成，给定一个输入坐标 (x, y)，可以获得一个 1D 的数值，我们记作 offset。Layout 本质上可以理解为一个函数，这个函数的的计算只是整数的乘法和加法，所需要的参数就是我们给定的 shape 和 stride，即，计算指针偏移 offset 的过程，即是坐标和我们定义 layout 给定的的 stride 的对应位置相乘并相加。

为不失一般性，CuTe 中将 base_ptr 这类可以通过加偏移来访问数据的实体称为 Engine，类似 c++ 中 iterator。Tensor / Engine / Layout 的关系以及作用，如 Fig1. 所示。

Figure1. 一个 2 维 Tensor 的例子，其由一个 base_ptr (i.e. egine) 和一个 layout 组成。layout 本质上上是一个以 2 维坐标（m, n）为输入，1 维 offset 为输出的一个函数，这个 offset 配合 base_ptr 得到我们实际想要访问的数据

由于 layout 是一个函数，那么我们参考高中数学中函数的一些基础操作，就能理解 layout 代数实际上是在干什么了。

## Layout 的复合 compose

我们回顾一下函数的复合的定义：给定一个函数， y = f(x) ，再给定一个函数 z = g(y) ，那么 g 复合 f 可以理解为两次连续的映射， （x -> y -> z） ，即， z = g(f(x)) ， f 为内层函数。

同样的，对于 layout 来说，给定 layout A： (m, n) -> offset_0，layout B: (p, q) -> offset_1，其复合过程也就可以理解为连续两次映射。我们记作：

layoutC = compose(layoutA, layoutB) = layoutA(layoutB) 。

那么给定一个内层函数 layoutB 的输入 (p, q)，我们来看看如何得到 compose 后 layoutC 对应的输出。首先我们知道， offset_0 = p * s_p + q * s_q，（其中 s_p 、 s_q 为 layout B 的 stride），这一步计算是平凡的，接下来就是 offset_0 作为输入传给 layout A，但是 layout A 的输入需要是 2D 的 (m, n)，怎么办？

事实上，很多初学者会在这里感到困惑，就是没注意到一个 CuTe 中有点反直觉的关键信息：即，对于任意维度的 layout 来说，其输入总是可以兼容 1D 坐标的，这个 1D 坐标会根据这个 layout 的 shape，按照 col-major 的方式来转换为多维坐标，并且这个转换和 stride 无关。

以 layoutA 为例，其 shape 为 (M, N)：

1D 坐标 offset_0 可通过如下公式转换为 2D 坐标： m = offset_0 % M，n = offset_0 / M ；再将转换后的 (m, n) 传入 layoutA，计算得到 offset_1 = m * s_m + n * s_n（其中 s_m 、 s_n 为 layoutA 的 stride）。

Figure2. layoutC = layoutA(layoutB) 的计算过程。对该式子中的内层 layoutB 的一个输入 (4, 1) 为例，首先计算出通过 layoutB 的 offset0 = 9；然后将其按照 col-major 方式转换为 2D 坐标，再输入到 layoutA 中得到 offset1 = 6，即是 layoutC(4, 1) 所需要的输出

我们打印出 layoutA/B/C，可以验证 layoutC 与我们构造的结果一致。

    auto layoutA = make_layout(make_shape(4, 4), make_stride(4, 1));
    auto layoutB = make_layout(make_shape(8, 2), make_stride(2, 1));
    auto layoutC = composition(layoutA, layoutB);

    if (cute::thread0()) {
      print_latex(layoutA); print("\n"); // % Layout: (4,4):(4,1)
      print_latex(layoutB); print("\n"); // % Layout: (8,2):(2,1)
      print_latex(layoutC); print("\n"); // % Layout: ((2,4),(2,1)):((8,1),(4,1))
    }

在理解 compose 的语义即是连续映射后，我们可以进一步观察到，因为 layoutC 的输入来源于 layoutB，因此其最后产生的输出个数一定与 layoutB 是一致的，因此我们可以理解为 layoutC 的 shape 在形式上与 layoutB 的 shape 相同；但是其 stride 会变得比较奇怪，甚至需要通过拆分 shape 中的维度来匹配 stride。例如，上述变换后的 layoutC，实际上是需要写成 ((2,4),(2,1)):((8,1),(4,1)) 的。

Layout compose 在很多地方都可以用到。例如，我们在之前的tiled copy 一文中提到的 TV-layout：给定 (t-id, v-id) 得到 (m-id, n-id)，这个 (m-id, n-id) 只是逻辑上的坐标，我们想要真实拿到实际能访问的物理地址，必须给其引入一个 tensor 本身的 layout，完成 (m-id, n-id) -> offset = m-id * s_m + n-id * s_n 的转换。

对 tensor 来说，其 layout 代表的是 (m, n) -> global offset 的映射。因此，可以通过一个 compose(tensor.layout(), tv-layout) 来让每个 thread 找到自己需要的数据，其中蕴含的映射关系是 (t, v) -> (m, n) -> global/shared offset。

这样的设计，既可以让我们变换 layout 时只关注坐标空间，又可以在实际使用中，通过 compose 不同的 tensor layout，让同一个 (m-id, n-id) 可以正确访问 global / shared / reg tensor 中的数值。

另一个例子就是为了避免 g2s copy 时 bank conflict 的 swizzle 的 compose。在朴素 cuda 实现中，我们引入 swizzle 逻辑时，需要关注我们读到的 global data 放到 shared data 哪个位置。但是引入了 layout compose，我们就可以做到对 swizzle 后的 tensor 仍然用正常逻辑下的 (m-id, n-id) 的坐标体系去访问到正确数据，虽然实际的物理地址已经改变了，但是我们不再 care 了。

具体来说，siwzzle 代表的是一个 offset -> offset' 的一个映射，我们对一个 shm tensor 做 swizzle，蕴含的映射是： (m, n) -> offset -> offset' ，因此能看到我们总是 swizzle 作为 A-layout 去做 compose 的，如下所示：

    static constexpr int kSwizzleB = 3;
    static constexpr int kSwizzleM = 3;
    static constexpr int kSwizzleS = 3;
    using SmemLayoutAtomC = decltype(composition(Swizzle<kSwizzleB, kSwizzleM, kSwizzleS>{},
                                     make_layout(make_shape(Int<8>{}, Int<64>{}), make_stride(Int<64>{}, Int<1>{}))));

## Layout 的逆 inverse

我们回顾一下函数的逆的定义：给定一个函数， y = f(x) ，其逆函数 g 可以做到给出 f 的输出 y 得到其输入 x，即 x = g(y) 。对于 layout 来说，inverse 操作可以完成输入和输出的调换。

因此，给定 layout A： (m, n) -> offset_0，其 inverse 过程即是给定 offset_0 可以得到（m, n）。又因为 offset0 可以 col-major 转换为新的多维坐标，实际上就可以做到 offset，view as (p, q) 的操作，进而构建出 offset -> (p,q) 这样一个新的映射。当然这个 view as (p, q) 的过程，需要进行一次 with_shape 操作，本质上就是一次 compose。

Figure3. layoutB = left_inverse(layoutA).with_shape 的计算过程。给定 layoutA 中的一个输入(2, 1)，其 col-major 1D 坐标为 6；我们首先通过 layoutA 得到 offset0 = 9，然后将输入输出调换为 9 to 6；然后引入 with_shape 将其排列成 layoutB 的 shape

    auto layoutA = make_layout(make_shape(_4{}, _4{}), make_stride(_4{}, _1{}));
    auto layoutA_inv = left_inverse(layoutA);
    auto layoutA_inv_with_shape = layoutA_inv.with_shape(make_shape(_8{}, _2{}));

另外，我们观察到 inverse 操作总是会带一个 with_shape 这样的操作，只单纯做 inverse 其实也能得到一个 layout（即，我们例子中的 layoutA_inv），但是这个 layout 的 shape 其实并不重要，我们只需要理解成原始 layout 输入和输出的调换组成的一组集合即可。

通常 inverse 后面会有一个进一步变换的步骤，才能让 inverse 的 layout 成为一个有语义的 layout。带上 with_shape，则可以让这列数字按照 col-major 排列成我们指定的 shape。事实上，如果我们查看 CuTe 中 with_shape 函数的代码，也能看到 with_shape(P, Q) 就是将 inverse 后的 layout，进一步 compose 一个 (P, Q) : (1, P) layout。

    template <class OtherShape>
    CUTE_HOST_DEVICE constexpr
    auto
    with_shape(OtherShape const& shape) const {
      return composition(*this, make_layout(shape));
    }

除了 with_shape，我们也可以 compose 其他 layout 来完成具备语义的各种操作。

了解清楚 inverse 语义后，我们来看 CuTe 的具体函数。CuTe 对 inverse 提供了 left_inverse & right_inverse 两个函数。二者在给定的 layout 为一一映射且连续的场景下（即，每一个输入都对应一个不同的输出，且输出空间是连续的），left_inverse 等价于 right_inverse。事实上，CuTe 代码中用到 inverse 的场景绝大多数都是这样一一映射且连续的，因此大部分情况下 right_inverse 和 left_inverse 可以相互替换。

那么如果我们给定的 layout 不是一一映射，即，多个输入都对应一个相同的输出时（但要求输出空间连续），如，broadcast 访问场景，其 inverse 其实是会丢失一部分信息的。虽然通过 left_inverse & right_inverse 也能得到一个 layout，但是其只能说保留了一部分信息，如 Fig.4 (a) 所示，inv_layout 并不能保留 layout 的所有输入。

如果我们给定的 layout 是一一映射，但是输出空间不连续，即，每个输入都对应一个不同的输出，但输出空间不连续，如，跨步访问场景，这时候 right_inverse 只会把第一段连续输出的 layout 进行一次 inverse，left_inverse 则保留了完整的原始 layout 的输出到输入的映射，所以一般来说 left inverse 会保留更多的信息。

Figure4. broadcast access 以及 stride access 下，left_inverse & right_inverse 的输出对比。可知，在 broadcast 场景下，二者结果一致但会丢失一部分原 layout 的映射关系；stride 场景下，left_inverse 构造出了原 layout 的整个陪域的映射关系，right_inverse 只构造了第一段连续值域的映射关系

总结来说，left inverse 是对 layout 的所有潜在输出（即，陪域）做 inverse，right inverse 则是在 layout 实际的"一部分"输出做 inverse，且这个"一部分"是实际输出空间（即，值域）中连续的一段。

当我们想要对一个非紧凑或非连续的 layout 进行 inverse 操作时，需要想清楚目的是什么，再选择是用 left 还是 right。另外需要强调一点，当我们给的 layout stride 不是编译期常量时，left_inverse 会报错，而 right_inverse 则会跳过这些非编译期常量的维度来构造 inverse layout。

虽然我们在本文开头说，只需要初中级别的数学知识就能读懂本篇文章，但是考虑到有些读者可能想要进一步挖掘 CuTe 之下的数学原理，我们也简单提一下：left inverse 和 right inverse 的概念其实来源于集合论，并且可以证明在两个集合满足双射的前提下，left inverse 等价于 right inverse；当集合仅满足单射时（即，一一映射，但是并不是每个输出都能找到输入），只存在 left inverse；而仅满足满射时（即，每个输出都必然能找到一个输入，但有可能多个输入对应一个输出），只存在 right inverse。

而求一个集合的逆的过程，可以通过 two-line notation -> swap two-line -> 排序的方法来完成。事实上，我们观察 CuTe 实现 inverse 函数的过程中就用到了排序，原因就在此。

    template <class Shape, class Stride>
    CUTE_HOST_DEVICE constexpr
    auto
    left_inverse(Layout<Shape,Stride> const& layout)
    {
      // Flatten and filter shape-1
      auto clayout = coalesce(layout);
      auto lstride = wrap(clayout.stride());
      auto lshape  = wrap(clayout.shape());

      // Prefix product of the shape
      auto preprod_shape = cute::fold(lshape, cute::tuple<_1>{}, [](auto c, auto vi) { return append(c, vi*back(c)); });

      // Sort by strides
      static_assert(is_static<decltype(lstride)>::value, "Left inverse requires static strides.");
      using Sorted = detail::SortByKey<decltype(lstride), tuple_seq<decltype(lstride)>>;
      auto sorted_seq = typename Sorted::val_type{};

      // ...
    }

## Layout compose & Inverse 实战

在笔者刚开始接触 CuTe 时，常常会困惑于一件事：这些 layout 代数究竟有什么用？大家对于 CuTe 官方文档吐槽也主要因为其完全没有前后联系的抛出一堆概念和代数运算，然后指望用户自己从庞大的代码库中理解其用法，人为制造出一条陡峭到天上的学习曲线。

作为一篇写给大家看的教程，我们不急于输出更多的概念，先结合 Flash atten v2 中 A-layout 到 C-layout 转换这个例子，来感受一下 layout compose & inverse 的用法和其蕴含的潜力。

我们知道在 Flash atten v2 的实现中，核心是 P = Q @ K.T 以及 O = P @ V 两个矩阵连乘（为简化说明，我们先忽略 softmax 的运算），在这个过程中 P 即是 QK 矩阵乘的输出，此时其符合 mma accumulator tensor 的 layout；也是 PV 矩阵乘的输入，此时其又需要符合 mma A tensor 的 layout。我们知道，tiled mma 的 layout 要求 A 和 C tensor 的 layout 是不同的，直接将 C 作为 A tensor 输入到 cute::gemm 函数中会编译报错。那么需要怎么对 C 进行转换呢？

我们从 C = A @ B.T 这样一个简单的例子，延伸出一个简化的矩阵连乘来说明。即，我们想完成 C = A @ B.T，D = C @ B.T 这样一个计算：

    template <typename T, int kTileM, int kTileN, int kTileK, typename TiledMMA>
    __global__ void simple_kernel(T *Cptr, const T *Aptr, const T *Bptr, int m,
                                  int n, int k) {
      Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k),
                             make_stride(k, Int<1>{}));
      Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k),
                             make_stride(k, Int<1>{}));
      Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n),
                             make_stride(n, Int<1>{}));

      int ix = blockIdx.x;
      int iy = blockIdx.y;

      Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
      Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
      Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
      //  gA(kTileM, kTileK, num_tile_k)
      //  gB(kTileN, kTileK, num_tile_k)
      //  gC(kTileM, kTileN)

      TiledMMA tiled_mma;
      auto thr_mma = tiled_mma.get_slice(threadIdx.x);
      auto tgA_g2r = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K, num_tile_k)
      auto tgB_g2r = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K, num_tile_k)
      auto tgC_g2r = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)

      auto trA_mma = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
      auto trB_mma = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
      auto trC_mma = thr_mma.partition_fragment_C(gC(_, _));    // (MMA, MMA_M, MMA_N)
      auto trD_mma = thr_mma.partition_fragment_C(gC(_, _));    // (MMA, MMA_M, MMA_N)

      clear(trC_mma);
      clear(trD_mma);

      int num_tile_k = size<2>(gA);
    #pragma unroll 1
      for (int itile = 0; itile < num_tile_k; ++itile) {
        cute::copy(tgA_g2r(_, _, _, itile), trA_mma);
        cute::copy(tgB_g2r(_, _, _, itile), trB_mma);

        // first we compute C = A @ B.T, it's fine
        cute::gemm(tiled_mma, trC_mma, trA_mma, trB_mma, trC_mma);

        auto trC_as_A_mma = trC_mma;

        // TODO: here we need to convert trC as trC_as_A,
        // because trC_mma has shape of ((2, 2), MMA_M, MMA_N)
        // while trA_mma needs shape of ((2, 2, 2), MMA_M, MMA_N / 2)
        // how to convert layout?

        // therefore, we got compile error: layout not acceptable for cute::gemm
        cute::gemm(tiled_mma, trD_mma, trC_as_A_mma, tBrB_mma, trD_mma);
      }
    }

以 16x8x16 fp16 mma atom 为例，trC 作为 accumulator layout 为 ((2, 2), MMA_M, MMA_N)，而下一次 gemm 所需要其作为 A-tensor，需要其 layout 为 ((2, 2, 2), MMA_M, MMA_N / 2)。

二者的数据本身是完全一致的，因此我们需要对 trC layout 进行一次变换。熟悉 flash atten v2 的读者可能已经发现，Tri-dao 在其实现中使用了 logical_divide 对 layout 在几何层面上进行了一次重排，我们将其过程拆解出来如下：

        // tri-dao method: logic-divide
        auto acc_layout_div = logical_divide(
            trC_mma.layout(),
            Shape<Underscore, Underscore, _2>{}); // ((2, 2), MMA_M, (2, MMA_N / 2)))
        auto a_layout = make_layout(
            make_layout(get<0>(acc_layout_div), get<2, 0>(acc_layout_div)),
            get<1>(acc_layout_div), get<2, 1>(acc_layout_div));
        // (((2, 2), 2), MMA_M, MMA_N / 2)
        auto trC_as_A_mma = make_tensor(trC_mma.data(), a_layout);

        // result correct
        cute::gemm(tiled_mma, trD_mma, trC_as_A_mma, tBrB_mma, trD_mma);

然而，reed 大师早已看穿了一切。

更优雅的，我们可以利用 inverse + compose 来从代数变换的角度上完成这个变换。其原理为:

我们已知 trC layout 是一个 (x_acc, y_acc) -> offset_0 的映射，且其是一个连续的一一映射，因此我们可以对其求 inverse（ right & left 均可），得到 offset_0 -> (x_acc, y_acc) 的映射；

然后，我们 compose 一个 (x_a, y_a) -> offset_0 的映射，即，trA layout，则可以得到 (x_a, y_a) -> offset_0 -> (x_acc, y_acc) 的映射；

最后，因为我们的目标是得到 (x_a, y_a) 在 trC 空间中的 offset_1，因此我们还需要 compose 一个 trC_mma tensor 的 layout，完成 (x_a, y_a) -> offset_0 -> (x_acc, y_acc) -> offset_1 变换，整个过程我们展示为如下代码：

        // reed method: layout-algebra
        auto C_tensor_for_partition = make_tensor_like(gC(_, _));
        auto trC_as_A_layout = thr_mma.partition_A(C_tensor_for_partition).layout();
        auto trC_as_C_layout = thr_mma.partition_C(C_tensor_for_partition).layout();

        auto acc_layout_inv = left_inverse(trC_as_C_layout);
        // (x_acc, y_acc) -> offset0 => (sorted) offset0 ->  (x_acc, y_acc)

        auto a_layout_algebra = acc_layout_inv.compose(trC_as_A_layout);
        // (x_a, y_a) -> offset0 -> (x_acc, y_acc)

        // trC: (x_acc, y_acc) -> offset1, compose as B
        auto trC_as_A_mma = trC_mma.compose(a_layout_algebra);
        // (x_a, y_a) -> offset0 -> (x_acc, y_acc) -> offset1

        // result correct
        cute::gemm(tiled_mma, trD_mma, trC_as_A_mma, tBrB_mma, trD_mma);

我们建议感兴趣的读者可以尝试跑通这个例子，并验证一下 tri-dao 和 reed 两者方法的结果，并尝试在更多的 mma 规格上测试，来感受 layout 代数所带来的语义上的便利性。

## 结论

我们在本文中梳理了 CuTe layout 代数中 compose 以及 inverse 的计算过程以及用法，其基本思想是将 layout 视为一种函数，因此我们可以借助函数中的 compose & inverse 的逻辑来理解 CuTe 的代数体系。

进一步的，我们结合 flash atten v2 中 A-C-layout 转换例子，也能看到利用代数方法能让我们完成一些自定义 layout 变换，真正做到随心所欲写 CuTe，而不止于调用 cutlass 现成的功能。

接下来的文章我们将继续探讨 layout 代数中的 product 和 divide 逻辑，之所以将其分在不同的章节中，是因为我们认为 product 和 divide 无法从函数的角度推导，而是更倾向于一种几何形式的构造。

## 文内图片URL

https://pic3.zhimg.com/v2-bf7c9f94b379ddfcfce2f8d3471c6914_1440w.jpg
https://pic3.zhimg.com/v2-9a38888c7c414cdb509d220f31f07910_1440w.jpg
https://pic3.zhimg.com/v2-76c6acc93de6242c35ac1b83801c001a_1440w.jpg
https://pic3.zhimg.com/v2-3efb4adf38cc8873ae4e14c085a8c39c_1440w.jpg
