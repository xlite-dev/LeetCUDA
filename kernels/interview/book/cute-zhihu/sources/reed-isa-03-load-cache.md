<!--
author: reed
author_id: reed-84-49
url: https://zhuanlan.zhihu.com/p/692445145
column: 
published: 2024-04-15
fetched: 2026-09-29
images: 2
-->

# NVidia GPU指令集架构-Load和Cache

前文介绍了NVidia GPU指令集架构(https://zhuanlan.zhihu.com/p/686198447)中的寄存器部分(https://zhuanlan.zhihu.com/p/688616037)，对于一个GPU程序而言，这些寄存器数据最初来自于外部存储结构，如何将数据从外部存储结构搬运到寄存器，以及在搬运过程中经过哪些Cache对程序的效率有重要影响。本文将围绕数据搬运展开，重点介绍NVidia GPU的存储层次和Cache层级，以及这些存储层级之间的数据搬运指令和Cache控制指令。文章结构方面，首先介绍了NVidia GPU上的计算单元和存储存储，然后介绍了Cache和Shared Memory机构，再后介绍了Load和Cache相关的指令已经一些特殊的Cache和预取行为，最后对文章进行了总结。

## NVidia GPU的计算单元和存储层级

GPU是一个高度并行的设备，其上装配了大量的计算单元，为了更好的组织这些计算单元，更充分的利用数据局部性（Spatial Locality）和支持数据规约和同步能力，GPU采用层次化的组织形式。整体而言，在Ampere及其之前的架构在计算单元组织方面可以分为三个层次：最小的计算机构为SubCore其负责线程束（warp）的执行，4个SubCore组成一个SM（Stream Multiprocessor），多个SM组织成GPU设备（device），根据设备设计规格的不同SM数目是其最直观的区别，如数据中心级A100和消费级GeForce RTX3080、GeForce RTX3090卡都是Ampere架构，在计算单元层面的最大区别就是所装配的SM数目。就计算单元具体而言（如图1），SubCore中包含有核心计算单元Tensor Core用以完成矩阵计算、CUDA Core用以完成向量化的浮点数和整数的乘加运算、SFU（special function unit）特殊函数单元用以完成sin、exp2、sqrt、rcp等超越函数，同时SubCore内有程序状态核心存储机构寄存器文件（Register File），还有具有广播语义的常量内存（图中标记为constant cache），SubCore内还有用于外部数据加载和存储用的Load Store Unit。SubCore中还有warp scheduler、branch unit、FP64等单元（他们对于理解指令集架构没有特别明显的作用，我们就没有单独标出）。对于存储层次而言，SubCore内有寄存器和Constant Cache，SM内有4个SubCore共享的L1 Cache和 Shared Memory。所有的SM通过交叉开关（CrossBar）共享L2 Cache。分Slice的L2 Cache通过Memory Controller和外部存储（独立的Die）HBM（high bandwidth memory）或则GDDR（Graphic DDR SDRAM）相连。数据中心级卡A100和消费级GeForce RTX 3080的另一个重要不同则是内存介质技术（HBM vs GDDR）。

![img-1](https://picx.zhimg.com/v2-1b64c3fa661c6045c5323abb897080b7_r.jpg)
（图注：Figure-1. NVidia GPU存储层级和Cache层级）

软件的是对物理硬件的抽象描述，对于CUDA编程而言，如图2，我们基本上可以将物理的SubCore映射到CUDA编程中的warp；将硬件上的SM映射为CUDA编程中的thread block；将多个SM组成的Device映射到CUDA编程中的grid。只是软件在向硬件映射的时候多了调度的逻辑。即一个物理的（physical）subcore可以运行多个warp，一个SM可以运行多个thread block，有限的SM组成的device可以运行远超其硬件数目的grid。

![img-2](https://pic4.zhimg.com/v2-d062e39beaaafb8d6ee04a07a65e2b93_r.jpg)
（图注：Figure-2. Hardware and its Software Abstraction）

## GPU中的Cache和Shared Memory机构

Cache是时间局部性和空间局部性的一条重要的实践方案，也是20世纪90年代研究最多的课题。对于高度追求并行处理的GPU而言，Cache是其重要的存储结构，其能极大的提升程序效率。GPU编程中（CUDA编程），数据最初被存储在全局内存中（global memory），而核心的计算发生在SubCore中的计算单元内。前文提过GPU是Load-Store架构（寄存器-寄存器架构），所以计算单元只能访问SubCore内的寄存器（和constant cache）。如果需要外部数据，必须通过Load指令将数据加载到寄存器中，GPU利用局部性原理在global memory和寄存器之间设置了两层Cache机构：L2 Cache和L1 Cache。其中L2 Cache在A100下为40MB，其数据被所有SM共享。L1 Cache被装配在每一个SM中，被SM中的4个SubCore共享。当SubCore中的Load Store Unit产生对全局内存的数据访问请求时，L1 Cache会查看当前请求在之前是否被请求过，如果之前请求过该数据并且没有被清除掉，那么L1 Cache命中可以直接返回该数据。如果L1 Cache之前没有被请求或者数据被清除导致miss，则L1 Cache会对L2 Cache发送请求，此时如果L2 Cache命中，则立即返回，只有两级Cache都miss的情况下才会对全局内存产生请求。L1/2 Cache的带宽和访问Latency相比HBM要好很多，适当的调整数据访问逻辑，提升数据局部性可以更充分的利用各级Cache，极大的提升程序效率。

Cache是不可编程的存储空间，其命中和清除逻辑由硬件自行控制，有些时候程序书写着可以更精准的管理共享数据，并且可以选择更合适的时机对数据进行更新和同步，为此NVidia GPU在SM级别提供了可编程的Shared Memory存储机构，它是一片可寻址的地址空间，同时提供了Load Store以及同步数据可见性的能力，这样当有一段数据需要反复使用时，则用户可以显式的加载到共享内存中，然后反复读取，避免了对低层级内存的访问压力。从Fermi架构开始，L1 Cache和Shared Memory共享同一份后端存储空间，只在前端做tag命中和地址判定，提供了相对灵活的配置能力，用户根据具体的使用场景决策将这份空间多分给L1 Cache还是自己手动可以控制的Shared Memory。

结合上面的介绍我们不难发现，从不同的维度去看数据加载有不同的形式：

1. 软件概念：Global -> Shared -> Register

2. 物理概念：HBM(Global) -> SRAM(L2 Cache) -> SRAM(L1 Cache) -> SRAM(Register File)

3. 片上概念：OffChip(Global) -> OnChip(L2 Cache) -> OnChip(L1 Cache) -> OnChip(Register File)

4. 共享层次：SMs(Global) -> SMs(L2 Cache) -> SM(L1 Cache) -> SM's SubCore(Register File)

## 数据Load指令

数据加载相关的指令整体如下：

```
LD, LDG, LDS, LDSM, LDL
```

其中LD指令（LoaD），是通用的（编译器在编译时无法推导地址空间类型的数据加载），如果编译时可以明确的知晓地址空间类型则使用有类型的加载指令LDG（LoaD Global memory），LDS（LoaD Shared memory），LDSM（LoaD Shared Matrix），LDL（LoaD Local memory）等，具体地

全局内存到寄存器：

```
LDG.类型.向量.Cache控制.L2预取
```

加载全局内存地址中的数据到寄存器，通过modifier可以配置加载时的数据宽度，如8bit数据，16bit数据，128bit数据等。同时可以配置各层级的Cache的bypass等情况，也可以配置是否对数据向L2中预取。单就指令而言向量化的数据加载（或叫大字长加载）如LDG.128是NVidia GPU支持的最大的加载指令，一条指令可以加载128bit数据，对于同等规模的数据使用更宽的加载指令可以减少warp对指令的调度次数，减少调度开销，减少MIO queue的事务数，避免由于queue满而造成阻塞。除了单指令位宽，更高效的数据加载还需要考虑合并访存（关于合并访存的优势我们后续会专门出一个"软件优化的硬件解释"系列中详细介绍）。

全局内存到共享内存：

```
LDGSTS, LDGDEPBAR, DEPBAR.LE SB0, 0x1 
```

异步读取数据全局内存并将结果存储到共享内存LDGSTS（LoaD Global memory STore Shared memory），可以实现不经过寄存器的全局内存到共享内存数据搬运，可以减少寄存器的使用和依赖，它在矩阵计算中，尤其Multi Stage的矩阵计算中有重要作用，可以参考cute之GEMM流水线(https://zhuanlan.zhihu.com/p/665082713)的异步拷贝章节。同时该指令要结合Barrier设置和等待指令（LDGDEPBAR，DEPBAR）协同使用。另外该指令在加载数据时可以指定是否在L1进行Cache，和对L2进行数据预取。

共享内存到寄存器：

```
LDS.类型.向量化，LDSM.块.转置
```

LDS的modifer可以设置数据位宽信息，和LDG类似高位宽的指令可以减少warp指令的调度数减少MIO queue中的事务数目，防止queue满引起的阻塞。LDSM为warp级协作指令，完成共享内存到寄存器的数据加载，进而将这些寄存器feed给Tensor Core指令完成矩阵计算，更细节的介绍可以参考cute之Copy抽象(https://zhuanlan.zhihu.com/p/666232173)和ldmatrix指令优势介绍(https://www.zhihu.com/question/600927104/answer/3029266372)。

局部数组和寄存器溢出：

```
LDL
```

目前认为有三种情况可以引入Local Memory：1. 当线程计算需要局部数组，并且数组的下标不能被编译时计算时；2. 单线程的寄存器使用数目超过255；3. 访问kernel数组常量时使用了不能被编译时确定的索引。Local Memory是CUDA编程中的一个概念，它的物理实体是全局内存中的一段。当上面情况发生时，每一个线程都会被分配一段全局内存来作为数据空间，由于数据需要对全局内存进行读写，一般而言对于线程数比较多的场景，其开销很大，除非万不得已，应该尽量能避免Local  Memory的使用。

## 广播语义的常量Cache

除了Load全局内存和共享内存，我们提到了SubCore中有constant cache机构，虽然它的名字是cache，但其本质是有名对象（通过地址标识）和寄存器更类似，它提供了一种广播语义，即warp内的所有线程都访问同一个数据时，它的访问速度和寄存器一样快，所以其可以被直接编码在指令操作数中。同时我们知道kernel的参数需要广播给所有的执行线程，所以其也是使用constant cache实现，可以实现高效的广播语义。另外我们可以实现device端的可编程常量（__constant__ __device_ int a;）。当warp内的线程访问同一个constant位置时，其是确定的latency的（和访问寄存器一样），但是当不同的线程访问的位置不同时，则是串行效果，是一个变Latency指令，所以SASS层级提供了LDC指令，用以实现不同线程访问不同的constant位置。

## 寄存器reuse和Prefetch

除了以上常规的存储机构和Cache，在计算单元流水线中，已经加载进计算单元流水线的数据也可以复用，其体现为寄存器的reuse，我们可以把它当做寄存器cache，它可以减少寄存器带宽压力一定程度上降低功耗，如

```
R1.reuse
```

除了前面提到的Load指令可以做伴随的数据预取，SASS还提供了显式设置L1/2 Cache的预取指令，如（CCTL = Cache ConTroL）

```
CCTL.E.PF2
```

## 总结

本文介绍了NVidia GPU的内存层次和Cache机构和指令集架构中的Load和Cache相关指令，了解这些指令能够更好的理解硬件在数据搬运时的行为，充分且合理的利用这些指令和Cache可以提升数据的搬运效率，指导数据搬运相关的优化。

## 参考

https://www.qidian.com/book/1031795831/(https://link.zhihu.com/?target=https%3A//www.qidian.com/book/1031795831/)

超标量处理器设计: 9787302347071: 姚永斌: Books(https://link.zhihu.com/?target=https%3A//www.amazon.com/%25E8%25B6%2585%25E6%25A0%2587%25E9%2587%258F%25E5%25A4%2584%25E7%2590%2586%25E5%2599%25A8%25E8%25AE%25BE%25E8%25AE%25A1-%25E5%25A7%259A%25E6%25B0%25B8%25E6%2596%258C/dp/B00JFJTI2I/ref%3Dmonarch_sidesheet)

https://pc.watch.impress.co.jp/docs/column/kaigai/1275220.html(https://link.zhihu.com/?target=https%3A//pc.watch.impress.co.jp/docs/column/kaigai/1275220.html)

https://pc.watch.impress.co.jp/video/pcw/docs/1275/220/p3.pdf(https://link.zhihu.com/?target=https%3A//pc.watch.impress.co.jp/video/pcw/docs/1275/220/p3.pdf)

reed：cute 之 GEMM流水线(https://zhuanlan.zhihu.com/p/665082713)

reed：cute 之 Copy抽象(https://zhuanlan.zhihu.com/p/666232173)

tensorcore中ldmatrix指令的优势是什么？(https://www.zhihu.com/question/600927104/answer/3029266372)
