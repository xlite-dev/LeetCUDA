# fig-32-2-persist-d-loop（drawio，2026-09）

- 内容：fp4 persist-D 主循环数据流——work loop 的 dense persistent / causal block-per-work grid 契约、producer（128T）装载与反向 barrier 全集（mbarrier 永不 re-init）、smem stages（Vᵀ slot0 可别名 Q 区、O staging 叠 Q/K 区）、consumer 每 kv tile 五步（add_delta_s→gemm_ss_fp4→masking(perm-aware)→softmax_with_quant→gemm_rs_fp4），及 Q 寄存器驻留（含历史 cicc bug）与 epilogue。
- 生成器：gen.py（入库副本，drawio-headless -s 3 导出 PNG）。
- 正文章节：ch32《FP4（二）：persist-D 主 Kernel 与两级 P 量化》。
- 执行模式：主 agent 直做（drawio 管线程序化生成）。
