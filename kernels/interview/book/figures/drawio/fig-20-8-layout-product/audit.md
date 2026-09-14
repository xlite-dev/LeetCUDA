# FIG-20-8 audit（layout product 嵌套构造）

- 重建自：@竹熙佳处《写给大家看的 CuTe 教程：Layout Product & Divide》
  （zhuanlan.zhihu.com/p/1971945267294111573，2026-09-14；原图本地留存
  `figures/zhihu/zhuxijiachu-layout-product-divide/fig-2.jpg`）
- 生成器：`gen.py`（本目录，源 `.tmp/drawio/gen_batch_pd.py`）
- 语义要点：A=(2,2):(2,1) ⊗ B=(2,2):(2,1) → ((2,2),(2,2)):((8,4),(2,1))；
  值 = 8m+4n+2p+q，几何 = 每块是 B 的拷贝、块偏移 = a_idx×4 →
  行值 0,1,4,5 / 2,3,6,7 / 8,9,12,13 / 10,11,14,15（非 row-major，教学点）。
- 迭代：1 轮过检（COVER/COLLIDE=0），view 直读验收通过。
- 高亮验证：值 11 = 块(1,0) 内 (1,1)。

## 2026-09-14 术语修订（code review）
- 块扫描顺序实为 A 的值序（row-major，左上→右上→左下→右下），非 colex 序
  （colex 会给出左上→左下→右上→右下与不同块偏移）；「线性序号」改「求值结果」。
  图内数值不变（已逐格复核）。
