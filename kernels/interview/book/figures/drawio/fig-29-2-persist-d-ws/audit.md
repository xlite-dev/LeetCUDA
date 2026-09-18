# fig-29-2-persist-d-ws（drawio，2026-09）

- 内容：fp8 persist-D WS 主 kernel 结构——warpgroup 128P+256C 与 setmaxnreg 寄存器分配、smem 朴素 80KB→kPersistQs2r Q 区复用 64KB、producer 装载序（V 先于 K、K stage0 等 q_consumed）、consumer 每 tile 五 Phase 主循环，底部附证伪清单与已落地优化。
- 生成器：gen.py（入库副本，drawio-headless -s 3 导出 PNG）。
- 正文章节：ch29《FP8（二）：persist-D 主 Kernel 与 Scale 折叠代数》。
- 执行模式：主 agent 直做（drawio 管线程序化生成）。
