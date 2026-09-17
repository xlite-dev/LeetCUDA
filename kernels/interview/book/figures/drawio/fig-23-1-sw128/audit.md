# FIG-23-1 audit（SW128 原子 XOR 置换网格）

- 生成器：`.tmp/drawio/gen_fig23_1.py`
- 迭代：2 轮。C1 右侧行注释与公式栏重叠（x=560..710 vs 640..890）→ C2 note 列缩窄至 558-668、公式栏移 x=690；迁移线 label 改底部统一图例行。
- 结构：8×8 网格（格值 = j XOR r），r=1/2/4 行蓝底高亮 + 三条显式坐标折线示例 c0 迁移；右侧公式 smem = 64r + 8(j XOR r) + (c mod 8)。
- 铁律复用：CJK 不 bold；公式框纯 ASCII 可 bold；折线箭头用显式 mxPoint（不绑 cell 锚点，避免漂移）。
