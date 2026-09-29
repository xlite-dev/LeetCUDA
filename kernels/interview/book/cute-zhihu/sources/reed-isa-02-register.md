<!--
author: reed
author_id: reed-84-49
url: https://zhuanlan.zhihu.com/p/688616037
column: 
published: 2024-03-23
fetched: 2026-09-29
images: 3
-->

# NVidia GPU指令集架构-寄存器

指令集是软件和硬件沟通的“词汇”(https://zhuanlan.zhihu.com/p/686198447)，在这个沟通过程中，具体的输入输出参数体现为寄存器，作为芯片上最基础的存储机构，了解它是学习具体指令的基础。本文将重点介绍NVidia GPU上的寄存器体系。针对不同的场景和目的这些寄存器包括：通用寄存器、特殊寄存器、Predicate寄存器、Uniform寄存器。文章结构方面本文首先通过介绍Load-Store架构引入寄存器然后介绍了寄存器表示的程序状态和GPU中延迟隐藏，之后分别介绍了NVidia GPU中的各类寄存器。

## Load-Store架构

Load-store架构，也叫做寄存器-寄存器架构，是计算机指令集体系架构中的重要形式。该架构中，所有的计算类指令的源操作数和目的操作数都必须是寄存器。内存和寄存器之间的通信通过独立的Load和Store指令完成。Load-store概念是RISC（Reduced Instruction Set Computer）架构的基本概念之一。NVidia Ampere及其之前的GPGPU架构基本符合Load Store架构的定义（有一点不符合的是常量内存），由于GPU为了提高内存的访问效率和数据局部性，内存层次相较于传统的CPU架构会多一些，所以其Load/Store指令也区分为全局内存（global memory）、共享内存（shared memory）、寄存器溢出或者局部动态寻址数组引入的Local Memory以及针对Tensor Core数据搬运的指令ldmatrix。整体Load、Store指令和非Load、Store指令的操作数情况如下表格所示

| 指令 | 类型 | 目标操作位置 | 源操作位置 |
| --- | --- | --- | --- |
| LDG | Load | 寄存器 | 全局内存 |
| STG | Store | 全局内存 | 寄存器 |
| LDS | Load | 寄存器 | 共享内存 |
| STS | Store | 共享内存 | 寄存器 |
| LDL | Load | 寄存器 | 局部内存 |
| STL | Store | 局部内存 | 寄存器 |
| LDSM | Load | 寄存器 | 共享内存 |
| 非Load/Store指令 | 算数指令 | 寄存器 | 寄存器 |

Load-Store架构是一种在许多现代处理器设计中广泛采用的数据访问模型，特别是在精简指令集计算机（RISC）体系结构中。这种架构的主要优势包括：

1. 简化的指令集：- Load-Store架构通过限制指令集只对寄存器进行操作，大大简化了指令集的设计和编码。这使得处理器的设计更加直观，易于理解和实现。

2. 高效的流水线设计：  - 由于所有的内存访问都是通过Load和Store指令完成的，这使得处理器的流水线设计更加简单和高效。处理器可以独立地优化计算和内存访问操作，减少资源冲突和提高吞吐量。

3. 编译器优化： - 显式的Load和Store指令使得编译器更容易进行优化，如指令重排、冗余消除等，以减少不必要的内存访问次数，从而提高程序的运行效率。

4. 提高内存访问效率：- 通过显式的Load和Store指令，可以更有效地管理内存访问，减少内存带宽的浪费，并提高内存访问的效率。

5. 寄存器的高效利用：- 由于Load-Store架构鼓励使用寄存器来存储数据，这有助于减少对慢速内存的访问，从而提高程序的执行速度。

6. 清晰的指令格式：- Load-Store架构通常采用三地址格式，这意味着大多数指令只涉及三个操作数（源操作数、目标操作数和立即数），这使得指令格式更加一致和易于处理。

7. 并行处理能力：  - 由于Load-Store架构的简单性和高效的流水线设计，它特别适合于实现并行处理，这对于现代多核和众核处理器来说是一个重要的特性。

8. 更好的错误隔离： - 在Load-Store架构中，由于内存访问和计算操作是分开的，这有助于在发生内存访问错误时更好地隔离错误，因为错误不会影响到计算操作。

总的来说，Load-Store架构通过其简单性、高效的流水线设计和编译器优化能力，为现代处理器提供了高性能和高能效的计算环境。这种架构在现代微处理器设计中非常流行。

从Hopper架构开始Tensor Core相关指令方面引入了wgmma指令，此计算类指令可以直接读取共享内存上的数据进行计算，而不必要求操作数必须在寄存器上，其则突破了Load Store架构的约束，但对于vector（SIMT）类的计算依然遵循。

## 寄存器表达的程序状态机

![img-1](https://pic4.zhimg.com/v2-8f7665ea36db49c1c8d52b4bd100d8db_r.jpg)
（图注：Figure 1. Register File in Sub-Core of SM）

图1为NVidia Ampere的SM（Stream Multiprocessor）的架构描述，可以看到一个SM包含4个Sub Core（或者成为Sub Partition），每个Sub Core中包含16,384个32bit的寄存器（通用寄存器）。我们知道CUDA GPU中的调度单位为一个warp，每个warp包含32个lane，这32个lane共享同样的代码，所以他们的寄存器在各自的线程看来也是同样的，将16,384个寄存器分配给32个lane我们可以得到512组寄存器。如图2-a所示，我们可以认为Sub core上的寄存器是一个二维结构横向为32个lane，纵向为512组。当SM执行某个kernel时，会将这些寄存器分配给不同的warp（warp组成block），图2-b展示了每个warp分配4个寄存器时，各个lane得到的本地的寄存器的名称和不同的warp组，对于每一个需要被执行的warp，寄存器文件按照kernel所需要的寄存器数目（分配粒度为4）分配给不同的warp，示例中对于每一个线程看到的寄存器为R0-R3（图中标示为thread-view）。

![img-2](https://pica.zhimg.com/v2-9dd02f9d548dafff211c048a4f248148_r.jpg)
（图注：Figure 2. Register File and its Warp View and Thread View）

这种寄存器分配模式构成了CUDA中最重要的延迟隐藏机制，那就是程序的执行状态都在寄存器中被记录和表示，warp调度单元选择一个可以执行的单元，利用执行单元（如图1中的FP32单元）读取特定warp对应的寄存器中的数据，将结果写入到这些寄存器中。如某个warp遇到了Load指令需要较多cycle才能获取到数据，而warp调度器则可以切换到其他数据已经ready需要进行计算任务的warp执行，即换一组寄存器表的状态即可（类似于CPU的线程调度）。如此便做到了执行单元只有一份，但是表达程序运行状态的存储单元却有多份，并且这种切换执行是十分轻量的，其和CPU中的超线程是类似的：一份执行单元，多份寄存器状态。通过warp切换来达到延迟隐藏的目的。物理硬件上寄存器文件一般使用SRAM（static random access memory）实现。

## 通用寄存器

上面提到的寄存器文件即为通用寄存器，通用寄存器是CUDA中的重要存储结构，其也是GPU上最高效的存储结构，通用寄存器可读可写，非Load/Store指令只能对寄存器进行操作，即源操作数和目的操作数都为寄存器。单个寄存器的位宽为32bit。每个线程所私有的寄存器最多可以为255个，在SASS表示中，以R为前缀，表示为R0-R255。由于寄存器的位宽为32bit，有时候算法需要更宽的数据存储结构时，则采用连续的多个寄存器组来完成，在SASS编码中只体现首寄存器编号，其余的寄存器不体现在编码中。如F2F.F64.F32 R4, R2; 表示32bit浮点数到64bit浮点数的类型转换(float to float: float32 to float64)，其中目标寄存器需要使用两个连续的寄存器R4R5来存储double值，源寄存器R2存储float值。另外约定R255寄存器为常零寄存器，在SASS中表示为RZ（Register ZERO）。

```
R0, R1, R2, R3, ..., R251, R252, R253, R254, R255(RZ)
```

## 特殊寄存器

特殊寄存器（Special Register）一般是只读的，用于标识该执行单元的定位信息，如线程号，线程块号等，需要通过特定的指令来读取这些寄存器，常见的特殊寄存器如下，其中SR_TID表示cuda thread block内的线程id即cuda编程中的threadIdx，SR_CTAID表示线程块id即cuda编程中的blockIdx，还有其他的获取硬件SM id、时间信息等的特殊寄存器，如下

```
SR_TID.X, SR_TID.Y, SR_TID.Z,
SR_CTAID.X, SR_CTAID.Y, SR_CTAID.Z,
SR_VIRTUALSMID, SR_LANEID, SR_LEMASK, SR_LTMASK, SR_GEMASK,
SR_CLOCKLO, SR_CLOCKHI
SR_GLOBALTIMERLO, SR_GLOBALTIMERHI
SRZ
SR_PM0-7
SR_SMEMSZ
```

具体的寄存器意义，可以参考PTX文档中的Special Register章节。

## Predicate寄存器

Predication技术是GPU架构中用于实现分支预测（branch prediction）的一种技术。在GPU中，predication通常用于控制线程束（warp）中线程的执行流程，而不是传统CPU中的分支预测。Predicatie寄存器的优势主要体现在以下几个方面：

1. 提高分支效率：在GPU中，predication寄存器允许整个线程束根据单一的预测结果来决定是继续执行还是跳过某些指令。这种方式可以减少每个线程单独进行分支决策的开销，从而提高分支处理的效率。

2. 减少分支开销：由于predication寄存器的存在，GPU可以在一个统一的控制下管理线程束的执行路径，这减少了每个线程单独处理分支的需要，从而降低了分支操作的总体开销。

3. 优化指令流水线：通过predication，GPU可以更好地管理指令流水线，减少由于分支预测错误导致的流水线清空（pipeline flush）和重新取指令（instruction fetching）的需要。

4. 提升并行执行效率：在GPU的并行计算中，线程束中的线程通常会执行相同的指令流。Predication寄存器允许线程束作为一个整体来响应分支指令，这样可以保持线程束内部的同步性，减少由于分支分叉导致的执行效率下降。

5. 简化编程模型：对于程序员来说，predication寄存器提供了一种更为简单的方法来控制线程束的行为，而不需要在代码中显式地处理复杂的同步和分支逻辑。

6. 增强硬件的灵活性：通过使用predication寄存器，GPU的硬件设计可以更加灵活地处理分支密集型的程序，特别是在处理大量并行线程时，可以有效地管理线程束的执行状态。

7. 减少能耗：由于减少了分支预测错误和相关的流水线清空，predication寄存器有助于降低能耗，这对于功耗敏感的移动和嵌入式设备尤为重要。

总的来说，predication寄存器是GPU架构中用于提高分支处理效率和优化线程束执行的关键技术。它通过简化分支控制逻辑，减少了分支预测的开销，并提高了GPU在执行并行计算任务时的性能和能效。具体的一个分支代码在不使用predicate和使用predicate寄存器后的效果如图3所示：

![img-3](https://pic4.zhimg.com/v2-37b353dbf6d35bc8dd2c3953bd401485_r.jpg)
（图注：Figure 3. Predication avoid pipeline stall）

具体地，Predicate寄存器主要有两个作用：一个是作为指令的执行条件放在指令的开始，如@P6 FADD R5 R5 R28; 和@!P1 FADD R11 R11 R17; 用以指示该条指令为条件指令，只有当@后的predicat运算结果为True时才会执行，反之为False时，该指令不产生执行副作用；二是predicate可以作为操作数参与特定指令的运算，如FMNMX R9 RZ R6 !PT;。Predicate寄存器为每个线程私有，每个线程最多可使用8个，在SASS表示中名称以P为前缀，表示为P0-P7，其中P0-P6为常规的读写Predicate寄存，P7为常真寄存器，即它始终为True，SASS中标示为PT（Predicate True）：

```
P0, P1, P2, P3, P4, P5, P6, P7(PT) 
```

## Uniform寄存器

以上的通用寄存器和predicate寄存器在SIMT编程模式看都是线程私有的寄存器，有些场景一个warp内的所有lane会执行完全相同的逻辑或者做reduce等功能（CUDA对应__reduce_sync类函数），NV提供了Uniform寄存器、Uniform Predicate和相应的指令来完成Warp Level的公共计算。使用Uniform寄存器可以减少对私有寄存器的使用量，继而可以减少warp对通用寄存器的使用，使得SM上有机会运行更多的warp提升并发度，同时由于Warp Level不需要向量化的执行单元，也能减少整体芯片功耗。在SASS层面，Uniform寄存器以UR作为前缀，单个warp最多可用64个，Uniform Predicate以UP作为前缀，单个warp最多可用7个，和通用寄存器、Predicate类似，最后一个寄存器UR63为常零寄存器，SASS中表示为URZ（Uniform Register ZERO），类似地，也有常真Uniform Predicate UPT（Uniform Predicate True）。

Uniform 寄存器表示如下：

```
UR0, UR1, UR2, UR3, ..., UR60, UR61, UR62, UR63（URZ）
```

Uniform Predicate表示如下：

```
UP0, UP1, UP2, UP3, UP4, UP5, UP6, UP7(UPT)
```

## 总结

寄存器是GPU中重要的存储结构，GPU通过提供大量的寄存器实现高并发的延迟隐藏模型，对于单线程而言使用较少的寄存器可以提升SM上能并发的Warp数，提升效率并发度，同时NV限制了单个线程能使用寄存器的上限为255个32bit；NV针对线程块索引等提供了特殊的寄存器用于获取逻辑坐标；Preicate寄存器可以实现小块代码段的更高效的流水线效果。Uniform寄存器可以实现warp level一致的计算逻辑和reduce功能，其可以减少通用寄存器的使用提升能效。

## 参考

https://eng.libretexts.org/Bookshelves/Computer_Science/Programming_Languages/Introduction_to_Assembly_Language_Programming(https://link.zhihu.com/?target=https%3A//eng.libretexts.org/Bookshelves/Computer_Science/Programming_Languages/Introduction_to_Assembly_Language_Programming%253A_From_Soup_to_Nuts%253A_ARM_Edition_%28Kann%29/04%253A_New_Page/4.04%253A_New_Page)

https://en.wikipedia.org/wiki/Load%E2%80%93store_architecture(https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Load%25E2%2580%2593store_architecture)

https://en.wikipedia.org/wiki/Predication_(computer_architecture)(https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Predication_%28computer_architecture%29)

https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf(https://link.zhihu.com/?target=https%3A//www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)

https://course.ece.cmu.edu/~ece740/f13/lib/exe/fetch.php?media=onur-740-fall13-module7.4.2-predicated-execution.pdf(https://link.zhihu.com/?target=https%3A//course.ece.cmu.edu/~ece740/f13/lib/exe/fetch.php%3Fmedia%3Donur-740-fall13-module7.4.2-predicated-execution.pdf)

https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers(https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/parallel-thread-execution/index.html%23special-registers)
