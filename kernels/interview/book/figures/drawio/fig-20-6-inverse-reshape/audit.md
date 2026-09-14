# FIG-20-6 audit（inverse 与 with_shape 重排）

- 重建自：@竹熙佳处《写给大家看的 CuTe 教程：Layout Compose & Inverse》
  （zhuanlan.zhihu.com/p/1962625273636845008，2026-09-14；原图本地留存 fig-3.jpg）
- 生成器：`gen.py`（本目录，源 `.tmp/drawio/gen_batch_f.py`）
- 原图要点：A (4,4):(4,1) 的 (2,1)=9 → inverse 查表 6↔9 互逆 → 1×16 竖条
  （index 9 处高亮 6）→ with_shape(8,2) col-major 折回 → B(1,1)=6。
- 迭代：3 处修复——①副标题 771px 横穿竖条 → 拆两行（x≤530）；
  ②橙色 reorder 标签 x=620 压竖条格 → 移入竖条与 layoutB 间走廊（x=664，箭头改道 x=720）；
  ③重导出后 view 复验通过。
