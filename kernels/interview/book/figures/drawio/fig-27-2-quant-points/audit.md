# fig-27-2-quant-points（drawio，2026-09）

- 内容：attention 主链（Q/K→QK^T MMA→softmax→P→PV MMA→O）上的三个量化点（离线 Q/K、在线 P、离线 V per-channel）与三条 scale 流：δ_Qδ_K 折进 exp2 系数、v_s 在 P 发射侧精确消去、p_s（fixed 1/448 vs per-row）epilogue 收尾，附溢出/互斥约束。
- 生成器：gen.py（入库副本，drawio-headless -s 3 导出 PNG）。
- 正文章节：ch27《量化注意力的数学基础》。
- 执行模式：主 agent 直做（drawio 管线程序化生成）。
