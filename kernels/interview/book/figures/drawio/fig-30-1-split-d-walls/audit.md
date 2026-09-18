# fig-30-1-split-d-walls（drawio，2026-09）

- 内容：大 D 的两堵墙与 dispatch 三分路由——顶部 persist-D D≤224 / split-D M8N1 224<D<768 / M4N2 D≥768（交叉点 D=768）；墙一 smem K tile 字节 O(D)、墙二寄存器 O 累加器 D/(2N_w) 及压力模型（N_w 消去、只有减 M_w 有效），附实测交叉数据与 split-D 精确性论证。
- 生成器：gen.py（入库副本，drawio-headless -s 3 导出 PNG）。
- 正文章节：ch30《FP8（三）：split-D 与 M4N2——大 D 的两堵墙》。
- 执行模式：主 agent 直做（drawio 管线程序化生成）。
