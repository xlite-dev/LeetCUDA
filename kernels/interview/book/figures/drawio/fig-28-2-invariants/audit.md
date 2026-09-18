# fig-28-2-invariants（drawio，2026-09）

- 内容：三条「改数值、保输出」的不变变换泳道——smooth-K（softmax 行平移不变，唯一代价 lse 修正）、smooth-V（均值剥离，P·1=1 精确还原）、Hadamard 旋转（正交 whitening 摊平 outlier），各带推导式与迷你示意图，底部结论：摊平 outlier 保 block scale 收敛且非近似。
- 生成器：gen.py（入库副本，drawio-headless -s 3 导出 PNG）。
- 正文章节：ch28《FP8（一）：量化前处理链》。
- 执行模式：主 agent 直做（drawio 管线程序化生成）。
