# FIG-21-4 audit（tiled copy 全景数据流）

- 重建自：@竹熙佳处《写给大家看的 CuTe 教程：tiled copy》fig-1/2/3
  （zhuanlan.zhihu.com/p/1930389542784964333，2026-09-14；原图本地留存
  `figures/zhihu/zhuxijiachu-tiled-copy/fig-{1,2,3}.jpg`）
- 生成器：`gen.py`（本目录，源 `.tmp/drawio/gen_fig21_4.py`）
- 语义要点：两步 TiledCopy 全景数据流。
  - 第一步 g2s：16 threads × 2 values，把 gmem `T (64,)` 的 (8,8):(8,1) 视图中
    上 4 行共 32 个元素拷入 smem tile (4,8)（行标 c0~c3，值 0-31）；
  - 第二步 s2r + TV layout：TV layout（V0/V1 × t0~t15，值 = t 与 t+16）决定
    thread↔value 分工，每个 thread 的寄存器 fragment 收到 {t, t+16} 两个值；
  - 配色：绿 = t0~t7 拷贝的数据、橙 = t8~t15 拷贝的数据（原文图例语义），
    gmem 灰色行 = 不参与拷贝；smem 每行前 4 绿 / 后 4 橙与原文 fig-1 一致。
  - 注：原文 fig-1 中 gmem 侧仅画 stride-8 列（0,8,…,56），本图为自包含教学图
    按"上 4 行 = 被拷贝的 32 个元素"补全为完整 8×8 视图；TV/fragment 取自
    同文 fig-3/fig-2。
- 迭代：2 轮（第 1 轮 TV 标签压 gmem 末行 1 处 COLLIDE，纵向压缩后过检）。
- 验收：check_text_fit TOTAL render-level issues: 0；check_overlap COVER=0 HIT=0；
  view 直读 PNG 通过（指纹校验防串图）。
