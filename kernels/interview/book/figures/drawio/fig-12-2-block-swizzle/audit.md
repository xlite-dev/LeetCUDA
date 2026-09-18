# FIG-12-2 block swizzle（drawio 新建，2026-09-14）

- 内容：C-tile grid 光栅化前后对照——default 2D grid 宽条带 resident（W=110 x H=1）
  vs 3D swizzle（bx = z*gridDim.x + x）近方块 resident（16 x 7，同 R≈110 blocks，
  7 组 x 16 槽）；
  底部 L2 footprint 公式对比（14208 vs 2944 K*2B，4.8x）+ PRO 5000 实测 L2 hit
  64.45% -> 96.15%、端到端 117.9 -> 122.9 TFLOPS（+4.2%，2048x16384x4096 HGEMM，
  正文 §12.3.8 口径）。
- 修订 2026-09-18（二轮复审）：常驻集模型 R=220 -> R≈110（110 SM x 1 block/SM），
  H/W/F 与倍率（16640/3840/4.3x）同步改正文 §12.3.8（14208/2944/4.8x）。
- 执行模式：主 agent 直做（生成器 gen.py -> drawio-headless -s 3 导出 -> PIL 裁剪
  两区复核 2 轮），coordinator 自审。
- 迭代记录：v1 底部三卡用 \n 双行 value 违反渲染铁律 -> 拆 box+text 叠放 v2 通过。
- 规格：缩略格 26x22（纯填色无文字）；文字卡字号 >= 12.5；画布 1120x660。
