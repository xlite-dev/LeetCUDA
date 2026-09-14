# FIG-26-1 audit（双 TiledMma 数据流）

- 生成器：`.tmp/drawio/gen_fig26_1.py`
- 迭代：2 轮。C1 两个标签（P (Aregs) / row_scale）被箭头穿越 → C2 移到线外。
- 结构：左栏 QK chunk 循环（蓝）| 中栏 Phase 2 rowcol→P（橙）| 右栏 PV chunk 循环（绿）；实线 S→P→O 主数据流 + 虚线 row_scale 耦合 rescale；底部寄存器账本三卡（S/P 32、O[C][32] 256、V frag dup）。
- 配色沿用系列色：蓝 QK / 橙 softmax / 绿 PV。
