# fig-29-1-scale-folding（drawio，2026-09）

- 内容：fp8 persist-D 的 scale 折叠全链数据流——主链 Q8/K8→QK^T MMA→log2 域 online softmax→P perm pack→PV MMA 上三条 scale 流（δ_Qδ_K 折 exp2 系数、v_s 消去 P̃=P·v_s·448、epilogue 一步 O=o_acc·(1/448)/rowsum 收尾），附发射域纪律三条硬约束与 rowsum MMA 气泡藏法。
- 生成器：gen.py（入库副本，drawio-headless -s 3 导出 PNG）。
- 正文章节：ch29《FP8（二）：persist-D 主 Kernel 与 Scale 折叠代数》。
- 执行模式：主 agent 直做（drawio 管线程序化生成）。
