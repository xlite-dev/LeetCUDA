# fig-31-2-fp4-pipeline（drawio，2026-09）

- 内容：fp4 前处理链五层数据流——①均值先行（q_m/k̄/q_km/v_m）→②量化 kernel（减 bias、WHT fused、1×16 块 absmax、cvt e2m1/ue4m3，D≤128 三段合一个 launch）→③输出 workspace（e2m1 数据+ue4m3 SF，K 按 perm32 写序、V 转置布局）→④ΔS rank-1 修正（免物化 K−k̄）→⑤fp4 persist-D 主 kernel。
- 生成器：gen.py（入库副本，drawio-headless -s 3 导出 PNG）。
- 正文章节：ch31《FP4（一）：NVFP4 格式与量化链》。
- 执行模式：主 agent 直做（drawio 管线程序化生成）。
