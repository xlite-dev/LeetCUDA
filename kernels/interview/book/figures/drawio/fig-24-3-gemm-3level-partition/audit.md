# FIG-24-3 audit（CuTe GEMM 三层 Partition）

- 重建自：@reed《cute 之 简单 GEMM 实现》
  （zhuanlan.zhihu.com/p/667521327，2026-09-14；原图本地留存
  `figures/zhihu/reed-simple-gemm/fig-3.jpg`，1967×1175 手绘稿）
- 生成器：`gen.py`（本目录，源 `.tmp/drawio/gen_fig24_3.py`）
- 语义要点：输出矩阵 C 的三层切分——`local_tile` 按 (blockIdx.y, blockIdx.x)
  从 gmem 切出 CTA tile (128,128)；warp 布局 (2,2) 下 `local_partition`
  给每个 warp 分 warp tile (64,64)；warp tile = 4×8 个 16×8 MMA 输出 tile，
  `partition_fragment_C` 为每 thread 生成累加片段（v0,v1 在第 i 行、
  v2,v3 在第 i+8 行的 2×2 四值）。层级换算：128×128 = 4×(64×64) =
  128 threads × 128 值/thread。
- 布局：三面板链（gmem 矩阵 → CTA tile 2×2 warp → warp tile 4×8 MMA 格 +
  thread value 放大泡）+ 每层 API 徽章 + 底部层级链；三层底色
  LBF（CTA）/ LOF（warp）/ LGF（thread value 追踪格）。
- 迭代：2 轮。第 1 轮发现 rbox 漏写 `as="geometry"` 且
  `partition_fragment_C` 标签实际渲染宽于盒（溢出到页外，导出宽 3632>3480），
  补齐属性并加宽盒后导出 3362×1658（内容边界 1080×550，页内）。
- 体检：check_text_fit TOTAL=0；check_overlap COVER=0（HIT 均为宿主格内
  文字叠放）；view 直读验收通过。
