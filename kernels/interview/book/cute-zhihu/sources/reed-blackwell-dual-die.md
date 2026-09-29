<!--
author: reed
author_id: reed-84-49
url: https://zhuanlan.zhihu.com/p/2051639750541546085
column: 
published: 2026-06-20
fetched: 2026-09-29
images: 9
-->

# Blackwell 双 die 拓扑：SM 与地址的 die 映射及跨 die 访存代价

单 die 面积有两条硬约束——光刻机一次曝光的 reticle limit（约 800 mm²）划下面积墙，良率又随面积超线性下降使得大 die 成本陡峭。把一颗大芯片拆成两颗较小的 die 再用高带宽互联缝合是绕开这两条约束的标准做法，代价是引入一道 die 边界：原本片内均匀的访存被分成”本地”与”跨边界”两种。

Blackwell 是 NVIDIA 第一代沿这条路线、把多颗 die 封装为单一逻辑 GPU 的数据中心架构。一颗 Blackwell GPU 内部是两颗 reticle 尺寸的 die，由 NVIDIA 的片间高带宽互联NV-HBI（NVIDIA High-Bandwidth Interface）连接，对主机呈现为单一设备、单一地址空间、单一逻辑 L2。这套封装内的结构与多路 CPU 服务器上的 NUMA（Non-Uniform Memory Access）同构：一颗 die 连同它本地的 HBM 对应一个 socket，NV-HBI 对应 socket 间的 UPI / Infinity Fabric——从某个 SM 访问本 die 的 HBM，与访问另一颗 die 的 HBM，代价不应相同。

本文围绕三件事展开：确定每个 SM（Streaming Multiprocessor）归属哪颗 die；确定任意一个虚拟地址背后的字节由哪颗 die 持有；并量化当二者错配、访问跨越 die 边界时所付出的代价。我们所用的全部手段都是常规 CUDA 代码，不借助任何特权接口或硬件文档之外的信息。

行文上，我们首先介绍双 die 拓扑及其硬件特点；之后给出测量单次冷 HBM 访问延迟的原语，并先用它把每个 SM 标定到所在 die；继而反过来以标定好的 SM 为量尺，反推出地址到 die 的三层确定性结构，它可以从虚拟地址闭式求解；随后考察跨卡推广性；最后基于 SM 与地址这两套 die 归属构造亲和与反亲和访问，测出二者的只读带宽差并解释其放大机理，再讨论走向真实工程的适用范围与局限，对全文进行总结。

## 一、Blackwell 双 die 的拓扑与硬件特点

Blackwell 的双 die 拓扑如图 1 所示。两颗对称的 die 各含 4 个 GPC（Graphics Processing Cluster）、约 74 个 SM、2 个 L2 partition 和 4 个内存控制器，各自连接 4 个 HBM3e（High Bandwidth Memory）堆栈、提供约 4 TB/s 的本地带宽；两颗 die 间由 NV-HBI 互连，聚合带宽不低于 10 TB/s，整颗芯片峰值带宽约 8 TB/s（HGX机型7.7TB/s， NVL72机型8TB/s）。本文的测量在 B200 上进行，其规格为合计 148 个 SM、8 个 GPC、4 个 L2 partition（是 Hopper 的两倍）、8 个 HBM3e 堆栈。

![img-1](https://pic3.zhimg.com/v2-8d4f9cb46f363c6edeefa0ecd2df94dc_r.jpg)
（图注：Figure 1. Blackwell single-package 2-die NUMA topology.）

值得强调的是 L2 的位置：4 个 L2 partition 并非集中一处，而是每颗 die 各占两个，与该 die 的内存控制器同侧——也就是说 L2 是按 die 切分的。当一颗 die 上的访问需要另一颗 die 所缓存或所拥有的数据时，无论是取数本身还是维持两侧 L2 的一致性，相关流量都必须经 NV-HBI。因此 NV-HBI 承载的不只是远端 HBM 的读写，还包括跨 die 的缓存一致性通信，它是这颗封装内 NUMA 唯一的跨域通路。

这套拓扑有一个反直觉的带宽关系：NV-HBI 的聚合带宽（≥ 10 TB/s）高于单颗 die 的本地 HBM 带宽（≈ 4 TB/s），这让带宽在 Blackwell 上不是一个方便的观测量。在 CPU NUMA 上，跨 socket 访问既增延迟又降带宽，把数据绑到远端 socket 跑个带宽测试即可暴露远端；而在 Blackwell 上，NV-HBI 比单 die 的本地带宽还宽，只要访存大致均摊在两颗 die 上，无非是两组内存控制器各出一半力，聚合带宽并不会因为”跨了 die”而明显下降。要想用带宽把 die 边界显示出来，就必须人为控制每次访问落在哪颗 die 上，而 CUDA 并没有提供按 die 申请内存的接口——这道障碍要到下一节才着手解决。相比之下，单次访问的延迟在本地与远端之间存在可测差异，即跨 die 多出的那一跳——因此本文以延迟而非带宽为主信号，这是整套方法论的起点。

顺带一提：NVIDIA 把双 die 的细节封在驱动之内、对程序员仅暴露”一颗逻辑 GPU”的抽象，并未提供查询 SM 到 die 或地址到 die 的接口。本文的全部 die 归属信息都靠用户态微基准反推。

## 二、测量原语：一次干净的冷 HBM 访问延迟

既然可观测的信号是延迟，整套方法就建立在一个原语之上：测准从某个指定 SM 发出、到某个指定地址的单次冷 HBM 访问延迟。这个原语本身就是后文所有结论的量具，如图 2 所示。要让它真正”测准”，前三件事必须同时解决：

1. 冷访问。 测的必须是真正的冷 HBM 访问，而非缓存命中。每次计时前先流式扫过远超 L2 容量的无关数据以驱逐缓存，并令计时核只执行一次访问、不做预热，保证第一跳就是冷访问。

2. 不被预取或访存重叠掩盖。 由单个线程串行执行依赖式访问——每一拍的地址依赖上一拍的返回值——所测周期即真实的 load-to-use 延迟。

3. 归属到确定的物理 SM。 通过申请超过单 SM 一半容量的动态共享内存，保证一个 SM 上只驻留一个线程块，再令该块读取自身 %smid 上报，延迟即归属到该 SM。

第四件事是单位稳定性：GPU 频率随 DVFS（Dynamic Voltage and Frequency Scaling）波动，因此主度量始终采用对频率不变的 cycles/hop，并在每段测量前后各测一次实际频率以剔除存疑样本。

计时核心可浓缩为：

```
int v = 0;
int64_t t0 = clock64();
do {
  v = ldg_cv(input);
} while (v < 0);          // 强制 load 完成（retire）后才出循环
int64_t t1 = clock64();
output[0] = v;            // sink 写，阻止编译器消除计时
```

这里的 do-while + sink 写是关键：没有它们，ptxas 会把第二次 clock64 排到 LDG 发射之后立即执行，而 LDG 是异步 retire 的，计时只会捕获到 LDG 发射（issue）的时刻，而非数据真正返回（retire）的时刻。加上二者，t1 - t0 才干净地报告周期级的冷 HBM 延迟。

![img-2](https://pic1.zhimg.com/v2-1b80c9e5143b13ca3684821ee5d3bdfc_r.jpg)
（图注：Figure 2. Measuring one cold-HBM access latency (SM s to address a).）

## 三、SM 到 die：干净的 74/74 划分

反推的出发点是一个足够小的地址：小到一次 16 B 的访问，其背后的字节必然完整落在某一颗 die 上，不会横跨 die 边界。固定这样一个地址，用全部 148 个 SM 各自访问它一次，每个 SM 测得的冷延迟就只取决于它与这个地址是否同 die——同 die 的 SM 走本地通路，跨 die 的 SM 要多走一趟 NV-HBI。

把 148 个延迟画成直方图，结果如图 3：两条不重叠的 band，跨 die 比同 die 多约 +400 cyc（约 +220 ns），每条各含 74 个 SM。快的一条是”SM 与该地址同 die”，慢的一条是”SM 需跨 NV-HBI 才能取到”。零重叠意味着对这个特定地址而言，die 归属是确定的、可二值判定的。

![img-3](https://picx.zhimg.com/v2-698e79e29680b67ecbd38d578f69288b_r.jpg)
（图注：Figure 3. SM-to-die latency histogram — two non-overlapping bands, 74 / 74.）

单个地址给出一次两组划分，还不足以断定这个二分就是 SM 的固有属性——它有可能只是该地址的偶然。为此换一批彼此独立、落在不同 die 上的地址，各自重做同一探测：每个地址都给出同样清晰的两组 band，而且同一个 SM 在所有地址上始终落在与之相对的同一组里（它对本 die 地址快、对另一 die 地址慢，从不摇摆）。这说明被探出的二分不依附于某个具体地址，而是 SM 自身的稳定属性，即它所在的 die。由此这张 SM 到 die 的标签表是确定且可复用的——对任意一个新地址，只要做一次单 SM cold load 探测，就足以读出该地址在哪颗 die，进而所有 SM 都拿到自己的 die 标签。

把 SM 到 die 标号按物理 smid 排布，得到图 4 的映射。它是确定的、周期性的、由架构固定：在前若干 smid 区间内每 16 个 SM 为一行、行间模式一致，越过中部边界后模式发生规律性旋转。这张映射可以一次性导出，并以一张常量表的形式供后续 kernel 在入口直接查表使用。

![img-4](https://picx.zhimg.com/v2-be841e030db73ce38593224f18c0edc5_r.jpg)
（图注：Figure 4. SM-to-die map (each cell is one of 148 SMs, labeled with its smid).）

## 四、地址到 die：一个三层确定性结构

有了 SM 到 die 的标签表，量尺与被测对象就互换了：现在反过来，用一颗已知 die 的 SM 去探测内存——它对某地址的延迟落在哪条 band，就直接读出该地址在哪颗 die。沿连续地址滑动这把量尺，便能勾勒出内存的 die 归属如何随地址变化。结果并非杂乱无章，而是一个三层、完全确定的结构，可以从地址 + 一个 per-allocation 的常量闭式求解：最底层是 die 分配的原子粒度，中间层是 2 MiB 块内由地址位奇偶决定的图样，最上层是块的极性如何随分配排布。下面把这三层粒度记作 G0、G1、G2（G 取 granularity），自底向上逐层拆开。

G0：4 KiB 是 die 分配的原子单位。 用一颗已知 die 的 SM，以 16 B 步长细扫一段地址（图 5）：die 标签的边界精确落在 4 KiB 处，且是单步翻转——偏移 4095 与 4096 之间延迟差直接由 +400 cyc 跳到 −400 cyc，没有过渡区。每个 4 KiB 段整段属于同一颗 die，跨过 4 KiB 边界即翻 die。

![img-5](https://pic3.zhimg.com/v2-8bc8caf0089aab99b92d1e6e0d052562_r.jpg)
（图注：Figure 5. 4 KiB is the atomic die-assignment unit — single-step boundary.）

G1：2 MiB polarity tile，由 8 个地址位的奇偶决定。 把尺度放大到 256 MiB，每 4 KiB 取一个样本得到图 6：横轴是该样本在 2 MiB 块内的位置（512 个 4 KiB 段），纵轴是 128 个连续的 2 MiB 块。图中可以同时看到两件事——4 KiB 段的 die 图样确实以 2 MiB 为周期重复（每一行内部呈现同样形态的细密交错），而相邻行之间只在两种互补的图案间交替。我们把这样一个 2 MiB 块称为 chunk。每个 chunk 内恰好 256 段属 die-0、256 段属 die-1，且归属服从一个确定的 8 bit hash：

```
mask 0x1EF000 = bits {12,13,14,15, 17,18,19,20}  (bit 16 被跳过)
die_intra(addr) = popcount(addr & 0x1EF000) & 1
```

我们在 128 个 chunk、65 376 个干净样本上验证，符合率 100%。沿一个 chunk 内 512 个段的序号看过去，die 标签就是该序号在这 8 位上的 popcount 奇偶——一条 Thue-Morse 样的交替序列。用地址位奇偶做这种细粒度交织，其效果（也是这类交织 hash 的惯常用途）是把任意规则步长的访问尽量均摊到两颗 die 上：固定的访存 stride 不会与 die 划分共振、把访问系统性地压到单颗 die，从而避免某种特定 pattern 让负载在两颗 die 间严重倾斜。同样的”打散”思路在更大尺度上由 G2 延续：G1 在 chunk 内打散 4 KiB 段，G2 在 chunk 之间打散 polarity，两层叠加让任意 stride 的访问都难以持续偏向单颗 die。这个不对称的掩码——4 个连续位、跳过 bit 16、再 4 个连续位——有一个鲜明的后果：相距 64 KiB 的两个地址落在同一颗 die（popcount 不变），要相距 128 KiB 才翻 die（bit 17 翻转）。bit 16 的跳过本身是个尚未完全解释的现象，最多只能说”很可能是某种 HBM / FBPA 条带 hash 在 64 KiB 处相消”。每个 chunk 只有两种互补的内部排布，二者恰好互为按位取反——我们用一个 1-bit 的 polarity（极性）记录每个 chunk 落在哪一种：polarity = 0 表示 chunk 内 seg 0 在 die-0、整块按 base 图样；polarity = 1 是它的取反副本。整块的 die 图样即 die_intra(addr) ^ tile_polarity。

![img-6](https://pic4.zhimg.com/v2-5eb8ee9c76788b358fd2031f16e4a799_r.jpg)
（图注：Figure 6. Per-2 MiB chunk pattern — two complementary layouts, 256/256 split.）

G2：chunk 的 polarity 取自一条固定序列，分配只决定从哪里切入。 G0、G1 已经把一个 chunk 内部的 die 图样确定为 popcount(addr & 0x1EF000) & 1，只差每个 chunk 的那一位 polarity。把一大段 cudaMalloc 出来、逐个 chunk 探测 polarity，会发现这一位并不是每次分配现掷的硬币，而是早已排好的一条固定序列（下称 canonical 序列；后文将看到，它在判据里就是那张 4096 位的 chunk_polarity[] 表，加上 8 GiB 反相位的扩展）：cudaMalloc(4 GiB) 得到的 2048 位 polarity，与依次 cuMemCreate(2 MiB) 申请 2048 个块得到的 2048 位逐位一致。这说明 polarity 不属于某次分配，而是物理 chunk 池本身的属性——池里每个物理块的 polarity 是固定的，分配只是按顺序从池中取块，因而必然复刻池的 polarity 排列。注意”按顺序”这一点是关键：同一段 VA 在两次分配中得到的不一定是同一段物理 chunk，因此 polarity 序列也未必相同（具体规律见图 7）。

剩下的唯一自由度，是一次分配从这条序列的哪里切入。它只由分配从池中第几个块开始取决定，这个起点就是全局 cursor。图 7 用几组分配模式验证了这个模型：顺序分配时，cursor 每次前进一个分配的大小（E1、E2）；cudaFree 把取走的块还回池中，下一次同尺寸分配优先复用刚空出的位置，于是反复 alloc/free 会在少数几个起点之间轮转——对 ≥ 1 GiB 的分配实测为两个 zone 交替（E3、E4）。换言之，只要拿到一次分配的起点 offset，整段地址的 polarity 就能照着 canonical 序列查出来。

![img-7](https://pic3.zhimg.com/v2-6dac12219be636e245ce09a8dc6dd31e_r.jpg)
（图注：Figure 7. Each allocation enters one canonical sequence at a cursor offset; free + realloc round-robins K=2 zones.）

三层合起来，就是一个对任意地址都成立的闭式判据。先做几个记号：把内存按 2 MiB 切成连续的块，第 k 个 2 MiB 块的位置记作 chunk_idx = k（即该块从分配基址起的 2 MiB 偏移序号；对地址 addr，它就是 ((addr - alloc_base) >> 21) + O，其中 O 是该分配进入序列的起点）。判据是：

```
die(addr) = popcount(addr & 0x1EF000) & 1       // G0+G1：8 位地址奇偶
          ^ chunk_polarity[chunk_idx mod 4096]  // G2：512 字节常量
          ^ (chunk_idx >> 12) & 1               // G2：8 GiB 反相位
```

三项各自来源不同：

第一项是 G0、G1 在 8 个地址位上的闭式算术，对任意 VA 立即可算。

第二项 chunk_polarity[] 是这条 polarity 序列的核心。它是一张只有 4096 位（= 512 字节）的常量表，每位描述一个 2 MiB chunk 的极性，覆盖连续 8 GiB（= 4096 个 chunk）；G1 里的 tile_polarity 即第 c 个 chunk 在这里查到的那一位异或上 8 GiB 反相位，等价于 chunk_polarity[c mod 4096] ^ ((c >> 12) & 1)。它没有更短的闭式描述——4096 位的具体内容须靠测量得出，但所有更大尺度的结构都从这 4096 位推出：100 GiB 实测显示，pol[c] == chunk_polarity[c mod 4096] ^ ((c >> 12) & 1) 在 51200 个 chunk 上有 51182 匹配（剩余 18 位是探针噪声），即整段 100 GiB 严格服从这条公式。

第三项就是上面那条公式里的 (chunk_idx >> 12) & 1：每越过一个 8 GiB 边界，整张 chunk_polarity 表”翻转一次”——前 8 GiB 用 base、后 8 GiB 用 ~base、再后 8 GiB 又回到 base，如此正负正负地铺满整个内存空间。这就是为什么 4096 位的 base 表加一位奇偶，足以描述 B200 全部 192 GiB 内存里所有 chunk 的极性。

这张 4096 位常量是 per-arch 的：跨卡测得 8191⁄8192 一致（剩 1 位是探针噪声），不是 per-card；本仓库的 polarity_scan 工具一次性扫出后即可作为代码内常量永久使用。每次分配进入序列的起点 O 则是 per-allocation 的：分配入口对 buffer 的前 ~32 个 chunk 各做一次单 SM 探测、与常量表做最佳匹配即可定出，整个过程不到一秒。

至此地址到 die 完全确定：先做 8 位掩码 popcount，再查 512 字节表的一位，最后与 8 GiB 奇偶位异或，这三步操作的结果就是该地址所在的 die。

至此 SM 到 die、地址到 die 两套映射都已建立。下一节先验证它们的跨卡推广性，第六节再用它们做实测。

## 五、跨卡：结构普适，标号逐卡

这套结构能否跨卡复用，关系到映射推导的成本能否一次性摊销。在同一张卡上重复测量，每次都得到完全一致的 SM 到 die 映射，对固定设备它是确定性的。在多张 B200 上分别跑同样的 smid_map 探测后会发现：双 die 的结构本身普适——每张卡都呈现严格的 74⁄74 划分与同量级的跨 die 延迟差；但具体的 smid 到 die 的标号逐卡不同，如图 8 所示，差异既可能微小、也可能涉及绝大多数 SM。

![img-8](https://pic4.zhimg.com/v2-73d139b652ab26099d4bc59db4e850b9_r.jpg)
（图注：Figure 8. SM-to-die labels across multiple B200s — the 74/74 partition is universal, but the smid-to-die labels vary per device.）

这在直觉上是合理的：双 die 的划分是硅片层面的硬件常量，而 smid 只是运行时暴露物理 SM 的逻辑编号，很可能在制造或固件阶段为良率与 binning（如屏蔽个别失效 SM）而设定，NVIDIA 并未文档化。其工程后果是明确的：SM 到 die 映射不能跨卡假定一致，必须按设备各自推导——即便两张卡碰巧一致，也不能据此推广。好在同卡推导一次即稳定，多 GPU 作业由每个进程各自推导本地映射即可。

## 六、die 亲和放置与只读带宽

把前面两套 die 归属——SM 在哪颗 die、某段地址在哪颗 die——合到一起用于优化，最直接的形式是让每个线程块只访问它所在 die 上的数据。

要干净地验证这一点，需要一段 die 归属已知、且最好处处一致的内存。借助第四节的闭式判据可以直接构造：用 VMM（Virtual Memory Management）接口逐个申请 2 MiB 物理块、探测其 polarity，只保留 polarity = 0 的块拼成一整段 global 内存。这样整段内存的 polarity 恒为 0，闭式判据退化为 die(addr) = popcount(addr & 0x1EF000) & 1——不查表就能算出任一地址在哪颗 die，且每个 4 KiB 段的 die 归属在整段内存里都按同一规则排布。

在这段内存上跑一个只读的聚合 kernel：每个线程流式扫过分给它的那些 4 KiB 段、把读到的字节累加进寄存器，工作集远超 L2、不写回，因而带宽被压到饱和、GB/s 直接反映读带宽。kernel 配置为 <<<148, 1024>>>（用 200 KiB 动态共享内存占满 SM，强制每个 SM 只驻留 1 个块，从而 148 个块铺满 148 个 SM）。我们只改”哪个块去读哪些段”这一张对应表，构造三种配置：

```
die-affinity： 每个 SM 只读本 die 的段（用第三节的 SM 到 die 表 + 闭式判据配对）
cudaMalloc：   内存换成默认 cudaMalloc（polarity 不再恒为 0），块仍读自己那一段，
               于是约一半的段随机落在跨 die
die-anti：     每个 SM 只读对侧 die 的段，每条 LDG 都跨 NV-HBI
```

三种配置共用同一个 kernel 二进制，grid、block、寄存器数、指令数、搬运字节数全部相同，唯一的变量是每条 LDG 实际打到本 die 还是对侧 die。结果如图 9、下表：

| 映射 | 输入带宽 | 占 8 TB/s 峰值(*实际7.7TB/s) |
| --- | --- | --- |
| die-affinity | 6226 GB/s | 77.8% |
| cudaMalloc | 5800 GB/s | 72.5% |
| die-anti | 4396 GB/s | 55.0% |

affinity 对 anti = +42% 带宽，而 cudaMalloc 干净地落在中段，正是随机极性在段级混合同 die / 跨 die 所预期的位置。这一收益完全来自把计算与数据放在同一颗 die，代价仅是 kernel 入口一次 SM 到 die 查表。

这个差并非来自单次访问变快。第一节已经指出，访存一旦在两颗 die 间均摊，聚合带宽就看不出 die 边界；affinity 与 anti 的差之所以出现，是因为它们经过的路径不同。两种配置在 HBM 控制器侧的负载是对等的——affinity 让各 die 的 SM 读各自 die 的段，anti 让各 die 的 SM 全部读对侧 die 的段，但两侧的 HBM 控制器组都各承担一半总流量；差异只在中间这一跳：anti 配置下每条 LDG 都要经 NV-HBI、并在请求方与归属方两颗 die 的 L2 各做一次 sector 查找（下文 ncu 段落的 1.5×/2× 放大正是这条路径的直接指纹）。

决定饱和吞吐的不是单次延迟，而是每个资源的排队：由 Little 定律，吞吐 ≈ 在飞请求数 ÷ 有效服务时间。anti 配置下每条请求的有效服务时间因跨 die 多一跳路径与额外的 L2 查找而上升，且 NV-HBI 聚合带宽（标称 ≥ 10 TB/s）虽然超过单 die 本地带宽，但被双向流量同时占用。两者叠加，使得一个空载下并不显眼的单次延迟差，在饱和下被放大成可观的带宽差。跨 die 惩罚的本质，不是单次访问慢了多少，而是同一份请求被两次占用了沿途的资源，至于哪一环（NV-HBI 链路、两侧 L2 partition、还是再下游）先成为瓶颈，本文未做穷尽对比；带宽收益则是把延迟探测给出的 die 映射用于回避这种占用之后的结果。

![img-9](https://pic1.zhimg.com/v2-f1d05b3edd0549b61237c6d4cc9c30c0_r.jpg)
（图注：Figure 9. Affinity / cudaMalloc / anti bandwidth, and the L2 sector amplification from ncu.）

ncu 的指标进一步确认了 NV-HBI 的角色。三种配置下 SM 侧指标（指令数、warp 占用、scoreboard 压力）完全相同，DRAM 实际读字节也相同；唯一随跨 die 比例变化的是 L2 sector 放大：L1 在每种配置下都发出相同的 sector 读，而 L2 服务的 sector 数恰好是 1× / 1.5× / 2×：

```
L2 sectors / L1 sectors = 1 + cross_die_fraction
  affinity ( 0% 跨die): 1.00×
  cudaMalloc (~50%)   : 1.50×
  anti (100%)         : 2.00×
```

最自然的解读是：每条越过 die 分界线的 LDG 要在两颗 die 各自的 L2 partition各做一次 sector 查找——请求方 die 的 L2（未命中、转发）和归属方 die 的 L2（未命中、下到 HBM）。这就把 SM 到内存的路径定性为：SM 接的是本 die 的 L2，L2 才是判定”这条 line 不归我、转发出去”的单元；换言之，一次跨 die 访问要顺着”SM、本 die L2、NV-HBI、远 die L2、HBM”这条链走完——NV-HBI 不是挂在旁边的总线，而是把两颗 die 的 L2 缝合成一块分布式缓存的链路。 需要提醒的是：本节为了让“每个 SM 唯一对应一个 block，读自己die 那一份”这件事干净成立、把所有差异都归因到 die 边界本身，选了 148 × 1024 + 200 KiB 动态共享内存这种“每 SM 仅驻留 1个block”的形态，它不是带宽最优的，提高 occupancy 或使用TMA可以把绝对带宽再次提高，但本节的解释机理（NV-HBI 路径上的双 L2 partition 占用、L2 sector 1× / 1.5× / 2× 放大）对那些形态同样成立。

## 七、工程可用性与局限

放置本身不依赖调度器：每个块在入口读自身 %smid、查出所在 die、再选对应 die 的数据区，调度顺序不可预知也无妨。即使不做精细的逐块亲和，仅把数据均分到两颗 die 轮流使用，就已显著优于所有访问倾斜到单颗 die 的退化情形（这并不需要谁主动制造——一个对 die 拓扑无感的分配器加上恰好与 polarity 共振的访存 stride 就够了）：均分让两颗 die 的 HBM 同时出力，倾斜则受限于单 die 带宽。精细亲和是在此之上再逼近峰值。

要落到真实工程，有两处待解决。其一在分配侧：前文测的是 cudaMalloc / cuMemCreate，而业务代码通常经框架内存池（如 PyTorch caching allocator）、cudaMallocAsync 或 unified memory 拿到显存——这些接口下 polarity 序列与 cursor 是否同样确定，本文未单独验证；可行的折中是把 die 感知下沉到分配器——底层按 die 维护两个内存池、对上层仍暴露常规接口，分配时按调用方所在 die 选池。其二在地址侧：地址到 die 虽可闭式求解，但仍需先确定某段数据在哪颗 die——或由分配器在分配时记录，或用 canonical 序列加偏移探测推算，或如第六节用 VMM 拼出 polarity 恒为 0 的内存。如何最低成本地拿到地址归属，是走向通用的关键一环。

几处本文未覆盖、不应当作已解决：sub-1 GiB 分配的 zone 数 K、内存压力与多进程下的行为、以及第七节提到的非 cudaMalloc 分配路径（PyTorch caching allocator、cudaMallocAsync、unified memory）下 polarity / cursor 是否同样确定。最后，整套方法成立的前提是硬件确实以足够粗的粒度按 die 划分内存；若未来某代在更细粒度上交织、跨 die 惩罚低于可分辨阈值，则 die 感知放置不再有意义，应转而以聚合带宽为优化目标——本文的测量流程本身也能识别这种情形。

## 总结

本文把”Blackwell 跨 die 访问究竟有何代价”这一难以直接观测的问题，构建为一条可测量、可证伪的链路：

- 以一个测准单次冷 HBM 访问延迟的原语为量具，先固定一个必落在单一 die 的小地址、用 148 个 SM 各探一次，把每个 SM 标定到所在 die；再反过来以已知 die 的 SM 为量尺去扫地址，反推内存的 die 归属——这条自举链是整套方法的骨架；

- SM 到 die 是干净的 74⁄74：两条零重叠的延迟 band，且换多个独立地址复测划分始终一致，说明它是 SM 的稳定物理属性而非寻址假象；

- 地址到 die 是一个三层确定性结构 die(addr) = popcount(addr & 0x1EF000) & 1 ^ chunk_polarity[chunk_idx mod 4096] ^ (chunk_idx >> 12) & 1——4 KiB 原子段、2 MiB polarity tile（bit 16 被跳过，64 KiB 同 die / 128 KiB 翻 die）、以及一张 4096 位（512 字节）per-arch 常量加 8 GiB 反相位，整段 192 GiB 内存的 die 归属由它完全描述；

- 双 die 结构跨卡普适，但 smid 标号逐卡不同，SM 到 die 必须逐设备推导；

- 一个空载下约 +220 ns 的单次延迟差，在饱和下经排队效应放大为 +42% 的聚合读带宽差，ncu 中表现为 L2 sector 放大 1× / 1.5× / 2× = 1 + 跨 die 比例，表明 NV-HBI 位于 L2 fabric 之上。

就方法而言，最关键的两点是：在这套封装内 NUMA 上，可观测的主信号是延迟而非带宽；SM 到 die 映射必须逐设备推导。就落地而言，分配粒度与地址归属是两处尚待打磨的环节。此外还有一个本文未能测到的二阶收益：基准全程是冷的（L2 命中率 ≈ 0），而跨 die 访问会瞬态在两颗 die 的 L2 各占一个 line。极限来看，全跨 die 下每条 cache line 都在两边各占一个槽位，die 亲和等价于把 L2 有效容量翻倍——有待一个工作集驻留的 kernel 去量化。

## 参考

- Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking(https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1804.06826)

- Microbenchmarking NVIDIA’s Blackwell Architecture: An in-depth Architectural Analysis(https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2512.02189)

- John L. Hennessy, David A. Patterson. Computer Architecture: A Quantitative Approach(https://link.zhihu.com/?target=https%3A//dl.acm.org/doi/book/10.5555/1999263)

- NVIDIA Blackwell Architecture Technical Brief(https://link.zhihu.com/?target=https%3A//resources.nvidia.com/en-us-blackwell-architecture)

- NVIDIA Blackwell Datasheet(https://link.zhihu.com/?target=https%3A//www.primeline-solutions.com/media/categories/server/nach-gpu/nvidia-hgx-h200/nvidia-blackwell-b200-datasheet.pdf)

- CUDA C++ Programming Guide — Virtual Memory Management(https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/cuda-c-programming-guide/index.html%23virtual-memory-management)

- Nsight Compute — Metrics Reference（ltst_sectors / drambytes_read）(https://link.zhihu.com/?target=https%3A//docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

- Non-uniform memory access — Wikipedia(https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Non-uniform_memory_access)

- Little’s law — Wikipedia(https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Little%2527s_law)

- Thue-Morse Sequence(https://link.zhihu.com/?target=https%3A//mathworld.wolfram.com/Thue-MorseSequence.html)
