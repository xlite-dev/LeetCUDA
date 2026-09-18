# fig-32-1-two-level-p（drawio，2026-09）

- 内容：两级 P 量化——问题（P∈[0,1] 行内分布剧变，直接块量化浪费 ue4m3 窄域）与方案（第一级行拉伸 2688=448×6、第二级 per-16 组 absmax ue4m3），域拉伸链三数轴、online softmax 使第一级退化成常数折进 exp2（编译期常数 −11.392/−2.585），附同链双归约、三守卫、lazy rescale 与 mxfp8 PV 变体域常量切换（2688→448）。
- 生成器：gen.py（入库副本，drawio-headless -s 3 导出 PNG）。
- 正文章节：ch32《FP4（二）：persist-D 主 Kernel 与两级 P 量化》。
- 执行模式：主 agent 直做（drawio 管线程序化生成）。
