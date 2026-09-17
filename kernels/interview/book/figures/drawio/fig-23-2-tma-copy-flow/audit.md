# FIG-23-2 audit（TMA Copy 数据流）

- 重建自：@竹熙佳处《写给大家看的 CuTe 教程：TMA Copy》
  （zhuanlan.zhihu.com/p/2003198909405763007 fig-2，2026-09-14；原图本地留存
  `figures/zhihu/zhuxijiachu-tma-copy/fig-2.jpg`）
- 生成器：`gen.py`（本目录，源 `.tmp/drawio/gen_fig23_2.py`）
- 语义要点：gmem 全局张量（16x6，globalX/globalY 坐标）按 3x3 box 切分，前 3 个
  box 已搬完（绿）、box_3={x=9,y=0} 正在搬（橙）；tensormap 面板列出各 box 坐标与
  boxDim/strides，坐标与 box 参数由 CPU encode 写入；只需 1 个 thread 发起拷贝，
  TMA 引擎按 tensormap 描述整块搬运 gmem box → smem tile（tileX/tileY 坐标从 0
  重新计数）。教学点：发起者（thread）与搬运者（TMA 引擎）分离，坐标语义由
  tensormap 承载。
- 原图 tensormap 面板内嵌一行英文提示注入文本（"ignore all previous
  instructions..."），非教学语义，已剔除并在会话中向用户报告。
- 迭代：2 轮（tileX/tileY 轴标题压 smem 面板左边框 → 移至面板外右侧/面板内底部），
  复检 COVER/COLLIDE=0，view 直读验收通过（导出 3382x1984 ≈ 1160x3）。
