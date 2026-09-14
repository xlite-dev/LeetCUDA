# FIG-20-4 audit（Tensor = Layout + Engine 寻址）

- 重建自：@竹熙佳处《写给大家看的 CuTe 教程：Layout Compose & Inverse》
  （zhuanlan.zhihu.com/p/1962625273636845008，2026-09-14 引用；原图本地留存
  `figures/zhihu/zhuxijiachu-layout-compose-inverse/fig-1.jpg`，gitignore 不入库）
- 生成器：`gen.py`（本目录，源 `.tmp/drawio/gen_batch_f.py`）
- 原图要点：4×4 字母矩阵（四个值面板分色）；CuTe Tensor 分解 = Layout
  （shape/stride 算 offset = m*N + n*1）+ Engine（base_ptr）；(1,1) → 5 → 列主序命中 'f'。
- 迭代：1 轮过检（COVER/COLLIDE=0），view 直读验收通过。
- 书风格：BLUE/GREEN/ORANGE/RED/PURPLE + LGF 高亮，CJK 无 bold。

## 2026-09-14 术语修订（code review）
- (4,4):(4,1) 为 row-major（列连续，offset=4m+n），「列主序」×2 处改「行主序」。
