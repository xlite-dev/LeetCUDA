# fig-28-1-quant-pipeline（drawio，2026-09）

- 内容：fp8 前处理链三泳道 aux kernel——Q（wht_qk_kernel→quantize_fp8_q）、K（kv_col_sum→fused quantize_fp8_k）、V（v_col_stats→quantize_fp8_vt smem 转置+VTPerm）汇入 Fp8QuantizedInputs workspace，供 persist-D/split-D/M4N2 三家族共享，底部标注融合原则与带宽账（~1.04 TB/s 贴峰）。
- 生成器：gen.py（入库副本，drawio-headless -s 3 导出 PNG）。
- 正文章节：ch28《FP8（一）：量化前处理链》。
- 执行模式：主 agent 直做（drawio 管线程序化生成）。
