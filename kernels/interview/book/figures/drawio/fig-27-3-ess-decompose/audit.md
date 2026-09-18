# fig-27-3-ess-decompose（drawio，2026-09）

- 内容：ESS 误差模型左右两半——左半 dense 行 vs causal row[0] 的 P 分布/ESS/幅度对比（ESS≈3000 vs 1，15× 绝对误差差距全来自幅度），右半 per-stage 误差分解条形（V 0.19 > QK 0.13 > P 0.11）与「单换 QK 无效、早行须全链 fp16」的 hybrid 依据。
- 生成器：gen.py（入库副本，drawio-headless -s 3 导出 PNG）。
- 正文章节：ch27《量化注意力的数学基础》。
- 执行模式：主 agent 直做（drawio 管线程序化生成）。
