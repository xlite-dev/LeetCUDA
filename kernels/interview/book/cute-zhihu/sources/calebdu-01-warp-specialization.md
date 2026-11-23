<!--
author: CalebDu
author_id: qi-qi-qi-qi-98-47
url: https://zhuanlan.zhihu.com/p/1941232816185644696
column:
published: 2025-08-19
fetched: 2026-11-22
images: 7
-->

# WarpSpecialization for Blackwell

## 前言：

本文旨在结合cutlass的源码，学习一下blackwell 架构的GPU的新特性如何在cutlass中应用。本期的主题是WarpSpecialization 在Blackwell 架构上的应用。参考的主要实现是include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp 和include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp，（不同的gemm实现的细节有差异）。

## WarpSpecialization：

在先前的文章([Nvidia Cute 实战-WarpSpecialization Gemm](https://zhuanlan.zhihu.com/p/8488333314) 和 [Nvidia Cute 实战-WarpSpecialization Gemm for Hopper](https://zhuanlan.zhihu.com/p/1905383022901059783)) 中，介绍了Hopper和Ampere架构上Gemm WarpSpecialization(WASP) 的实现，简单的来说WASP的目的就是把复杂的指令流解耦，让warp负责各自解耦后的子任务，避免不同指令流之间的互相阻塞，使得warp scheduler更高的调度指令发射。

![img-1](https://pica.zhimg.com/v2-85a108ab4d53e2c9dde69fa3a9530088_r.jpg)

（图注：Programming Blackwell Tensor Cores with CUTLASS GTC 25）

在介绍Blackwell 的WarpSpecialization 实现之前，我们先简单回顾一下Hopper上的WASP的实现，在Hopper上的WASP pingpong实现 launch 3个warp group(12个warp），一个warp group 中的两个warp作为生产者分别load mainloop和epilog 的数据，另外两个warp group作为消费者计算 生产者load的mainloop数据，通过tma pipeline进行生产者/消费者之间的同步。为了通过mainloop的tensorcore计算来掩盖epilog的操作，两个warp group之间利用sequence ordered pipeline来保证两个warp group mainloop/epilog 的overlap。以上的提到的pipeline 都是基于ptx mbarrier指令进行封装的，整一个WASP的核心也就是基于mbarrier。

![img-2](https://pic3.zhimg.com/v2-b360a09ef5019c1217eede8488097e50_r.jpg)

（图注：warp role）

在之前的文章[Dynamic Cluster and Scheduler for Blackwell](https://zhuanlan.zhihu.com/p/1940858379905001339)里介绍了Blackwell 的新特性Dynamic Scheduler, scheduler warp作为生产者调用CLC ptx 指令去动态的获取后续执行的CTA id，获取到底CTA id存储在smem给所有warp进行计算next tile info。

此外， 不同于Hopper 架构的wgmma tensorcore，Blackwell 全新的**tcgen5** tensorcore指令不再需要一个warp group参与，只需要一个thread 发起tcgen5指令。同时tcgen5 的completion_mechanism 直接由mbarrier 进行控制，不再是Hopper上调用wgmma.wait_group 指令控制tensor core计算的完成（与mbarrier分离），这样在Blackwell上tcgen5的tensor core计算同样可以通过pipeline来控制。

![img-3](https://pic2.zhimg.com/v2-a8a8de6628e58bc4dce476cdf0c90a39_r.jpg)

![img-4](https://pic1.zhimg.com/v2-fa8b46defbabc101fd7ec1d8b3c173d0_r.jpg)

（图注：https://www.bilibili.com/video/BV11tMwznEmo）

在Blackwell上WASP需要以下的Pipeline，来进行不同warp之间的同步

- **mainloop_pipeline:**负责mainloop 数据搬运和计算之间的同步,生产者:main load warp,消费者:mma warp
- **epilog_load_pipeline:**负责epilog 数据搬运和计算之间的同步,生产者:epilog load warp,消费者:epilog warp
- **clc_pipeline:**负责CLC response的获取和对应tile info的计算,生产者:scheduler warp,消费者:所有 warp
- **clc_throttle_pipeline:**控制scheduler warp 的调度下一个tile的速度,防止tile info 踩踏，生产者:main load warp,消费者:scheduler warp
- **accumulator_pipeline:**负责tcgen5 mma结果的完成和epilog 计算之间的同步,生产者: mma warp,消费者:epilog warp
- **load_order_pipeline:**保证prolog第一个tile mainloop load之后，再发射epilog load

![img-5](https://pic1.zhimg.com/v2-16336e2a9e50b5ad96da4aeb4dc8bbfc_r.jpg)

下图简单的画了Blackwell WASP 不同的warp如何基于pipeline进行通信实现同步，图中的缩写代表producer_acquire(**PA**), producer_commit(**PC**),consumer_wait(**CW),**consumer_release(**CR**), mbar(**mbar_expect_tx,**TMA pipeline 不需要调用**PC**), 虚线的PA 代表最初的pipeline state **phase=1** 第一次wait不触发。

![img-6](https://pic3.zhimg.com/v2-1376d97e6b2d202ffc983981606a4984_r.jpg)

![img-7](https://pic1.zhimg.com/v2-4b50bce05a18bcc3f50fcefbdfdd45e0_r.jpg)

（图注：gemm::CollectiveBuilder）

include/cutlass/pipeline/sm100_pipeline.hpp 基于include/cutlass/pipeline/sm90_pipeline.hpp 中的pipeline，针对blackwell的场景，封装了新的pipeline， 如**PipelineUmmaAsync**(继承PipelineAsync, **tcgen5.commit**重载producer_commit，用于accumulator_pipeline) , **PipelineTmaUmmaAsync**(继承PipelineTmaAsync, **tcgen5.commit** 重载consumer_release，用于mainloop_pipeline), **PipelineCLCFetchAsync(**用于clc_pipeline)

## 总结

随着GPU引入越来越多DSA的特性，为了达到最佳的SOL性能，需要引入更复杂的pipeline来尽可能隐藏延迟和打满计算单元的吞吐。基于WASP的异步编程与pipeline通信是未来GPU获得最优性能的必要手段。
