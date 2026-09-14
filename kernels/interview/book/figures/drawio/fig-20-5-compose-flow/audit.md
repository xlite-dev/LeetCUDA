# FIG-20-5 audit（compose 三步流）

- 重建自：@竹熙佳处《写给大家看的 CuTe 教程：Layout Compose & Inverse》
  （zhuanlan.zhihu.com/p/1962625273636845008，2026-09-14；原图本地留存 fig-2.jpg）
- 生成器：`gen.py`（本目录，源 `.tmp/drawio/gen_batch_f.py`）
- 原图要点：compose(layoutA, layoutB) 三步——C(4,1) 查 B 得 offset 9；
  9 按 A 的 shape (4,4) col-major 换算回 (1,2)；A(1,2)=6 回填 C。
  四网格（C 提问态/C 结果态/B/A）+ 橙紫绿三色虚线 step 箭头。
- 迭代：1 轮过检；view 验收通过（紫色线与 layoutB 标签贴边 ~2px，可接受）。
