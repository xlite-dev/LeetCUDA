<!--
author: reed
author_id: reed-84-49
url: https://zhuanlan.zhihu.com/p/686198447
column: 
published: 2024-03-10
fetched: 2026-09-29
images: 2
-->

# NVidia GPU指令集架构-前言

2017年的ACM图灵奖颁给了John L. Hennessy和David A. Patterson表彰他们在计算机架构开拓性的贡献。在他们共同撰写的文章“计算机架构的新黄金时代”中，详细介绍了指令集架构的发展和未来的机遇点。此处我们引用文中的指令集架构（Instruction Set Architecture）定义：“Software talks to hardware through a vocabulary called an instruction set architecture (ISA)“，和DSA（Domain Specific Architecture）的优势：1. 针对特定领域更高效的并行模式，2. 对内存层级更有效的利用，3. 某些场景可以使用更低的精度，4. DSL（Domain Specific Language）可以更好的暴露硬件能力。

对于高性能计算（High Performance Computing）和以深度学习（Deep Learning）为核心承载的人工智能，NVidia GPU在算力输出方面扮演着极其重要的角色。NVida GPU可以认为是图形和深度学习领域的DSA架构，即Domain为Graphic和Deep Learning的Specific Architecture，自然地前文提到的DAS架构的优势NVidia GPU也都有享用。指令集是软件和硬件沟通的“词汇”，我们必须用这套语言体系来和计算机硬件进行沟通，将我们的算法逻辑变换成硬件可理解的“词汇”，硬件通过这套词汇便可以得到我们所需要的算法逻辑结果。如图1所示，指令集体现了硬件能力的暴露，通过指令集暴露了硬件提供的加法计算、减法计算和乘法计算能力，图中的硬件没有提供除法计算能力，则软件在遇到除法问题时需要用其他指令来模拟实现。

![img-1](https://pic3.zhimg.com/v2-44a1e0ca0ccd6750d723420a75f3e2b8_r.jpg)
（图注：Figure-1. Instruction Set Architecture）

指令集架构是硬件能力的暴露，了解指令集架构能够让我们更清楚的知晓硬件所提供的基础能力，更好的辅助我们选择硬件友好的算法，同时了解指令集架构可以让我们选择更高效的指令，继而提升软件运行效率。

对于NVidia GPU而言，其软件部分的核心语言为CUDA，硬件架构的指令在不同代际是不同的（如Tesla，Fermi, Keper, Maxwell, Pascal, Volta, Turing, Ampere, Hopper，Blackwell），本文将基于Ampere架构进行介绍。图2展示了NVidia GPU由软件编程到硬件可执行的编译流程，其中CUDA是用户可编程语言，使用该语言编写的程序可以通过NVCC编译器编译成PTX指令，PTX指令是硬件无关的（实际上也有版本）用于屏蔽硬件差异，PTX指令通过PTXAS汇编工具可以汇编成SASS指令，SASS指令可以交由硬件执行，SASS指令是硬件相关的。我们所要研究的指令集则指SASS指令这一层次，因为该层次的指令直接能被硬件执行，是硬件能力的最基本抽象。

![img-2](https://picx.zhimg.com/v2-430382b0832c2ab58e96fdb97c8c1277_r.jpg)
（图注：Figure-2. NVidia GPU Complation and Execution Phase）

值得注意的是，NVidia并没有官方介绍SASS的文档，只在介绍cuda binary utilities的时候简要提及，本文内容多为利用cuobjdump反汇编libcublas、libcublasLt、libfft等官方库、利用ncu分析实际用例得到，同时和PTX文档相互对应印证，其中也不乏猜测和臆断的部分。

本系列文章将以指令分类的形式介绍NVidia Ampere GPU的指令集架构，同时在介绍具体指令时会结合实际场景分析潜在的性能风险和优化点。在文章结构方面主要有包含寄存器部分，该部分分为通用寄存器、特殊寄存器、Uniform寄存器、Predication寄存器；常量Cache部分，整数操作指令、浮点操作指令、BIT操作和逻辑指令、特殊函数指令；分支和控制指令；数据加载和存储指令；Warp Level指令；Atomic指令。希望读者了解完该系列文章后能够回答类似如下问题：“SASS中PRMT指令很多，是如何产生的，如何优化？”，“有了half类型为什么还需要half2类型？”，“整数除法是如何实现的？”，“浮点除法的效率如何？”，“Flash Attention中的快速EXP指数是怎么回事，在什么场景下结果会有问题？”。

本文为该系列的前言，后续部分我们将分不同章节重点介绍如下内容

- 通用寄存器

- 特殊寄存器

- Uniform寄存器

- Predication寄存器

- 常量Cache

- 整数操作指令

- 浮点操作指令

- BIT操作和逻辑指令

- 特殊函数指令

- 分支和控制指令

- 数据加载和存储指令

- Warp Level指令

- Atomic指令

总结

指令集架构是软件和硬件沟通的“词汇”，熟悉并了解这些词汇（ISA）才能更好的将软件问题通过这些指令映射到硬件上，同时了解NVidia GPU ISA也能帮助我们设计我们自己的GPU、NPU ISA。

参考

https://amturing.acm.org/byyear.cfm(https://link.zhihu.com/?target=https%3A//amturing.acm.org/byyear.cfm)

https://dl.acm.org/doi/10.1145/3282307(https://link.zhihu.com/?target=https%3A//dl.acm.org/doi/10.1145/3282307)

https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html(https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/cuda-binary-utilities/index.html)
