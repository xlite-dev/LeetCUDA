# FIG-22-2 audit（MMA atom thread-value 布局）

- 重建自：@竹熙佳处《写给大家看的 CuTe 教程：tiled mma》fig-1
  （zhuanlan.zhihu.com/p/1937145378446226159，2026-09-14；原图本地留存
  `figures/zhihu/zhuxijiachu-tiled-mma/fig-1.jpg`，5822×3306）
- 生成器：`gen.py`（本目录，源 `.tmp/drawio/gen_fig22_2.py`）
- 原图真实结构（view_image 因 image budget 超限曾返回占位符导致误读，最终以
  gcmp vision 两次独立读图仲裁 + 像素投影定位为准）：单个 16×8 主网格
  （M16/N16 尺寸箭头、行省略压缩为 5 可见行 + 红点省略号），单元格标注
  thread-N 且相邻两格同 thread；行 m=0（T0-T3）→ m=1（T4-T7）→ ⋮ → m=7
  （T28-T31）→ m=8（T0-T3 重复）→ ⋮ → m=15（T28-T31）；两处 thread-0 格
  红虚线高亮，紫虚线引出右侧 2×2 callout：thread==0 拿到 data-(0,0)(0,1)
  (8,0)(8,1) 共 4 值。原图无 A/B/C 三面板（任务主题由本图卡片补足）。
- 重建要点：主网格忠实复刻（T=thread、8 色循环、红虚线双框、双引线、N/M
  箭头、省略行）；右侧新增三卡：thread-0 callout（4 data = fragment）、
  m16n8k16 每 thread fragment 大小（A 8 值 / B 4 值 / C 4 值 + 守恒式
  32×16=512）、atom 最小可复制单元（mma.sync 一条指令 = 32 threads 协作；
  复制 → M32×N16 tile、映射不变）；左下 atom 复制示意（2×2 → 64 threads）。
- 语义：thread t → 行 t/4 与 t/4+8、列 2(t%4) 与 2(t%4)+1，共 4 格；行 m
  与 m+8 同组 threads（T 编号重复）——PTX m16n8 C/D（及 k8 的 A）布局。
- 迭代：2 轮。v1 引线 L1 穿 n0 表头与副标题、L2 水平段穿 m15 行、省略号贴
  行 → v2 走线绕外（L1 顶走 y=60，L2 缝隙走 y=350）、dots 上移、m15 下移。
  体检 TOTAL 0 / COVER 0（HIT 均为卡片内文字，合法）；view 直读验收通过。
- 配色：thread 8 色循环（红 #dc2626/蓝 #2171b5/绿 #2e8540/橙 #d97706/
  紫 #6a51a3/青 #0e7490/玫 #be185d/橄榄 #4d7c0f，浅底同族）；A/B/C 卡片
  蓝绿橙（与 fig-22-1 系列一致）。
