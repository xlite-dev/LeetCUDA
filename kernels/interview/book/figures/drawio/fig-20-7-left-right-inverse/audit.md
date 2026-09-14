# FIG-20-7 audit（left/right inverse 两情形）

- 重建自：@竹熙佳处《写给大家看的 CuTe 教程：Layout Compose & Inverse》
  （zhuanlan.zhihu.com/p/1962625273636845008，2026-09-14；原图本地留存 fig-4.jpg）
- 生成器：`gen.py`（本目录，源 `.tmp/drawio/gen_batch_f.py`）
- 原图要点：(a) broadcast stride(1,0)：left/right inverse 均 shape(4,1):stride(1,0)；
  (b) stride(4,1)：left_inv 保 (4,4):(4,1)，right_inv 降为 (2,1):(4,0)——
  stride 0 维度被投影去重，即 CuTe 判定可逆性的依据。
- 迭代：1 处修复——四个 left/right_inverse 标签侵入结果网格 → 左移至 x=150/700；
  复检 0，view 验收通过。

## 2026-09-14 数理修订（review 发现）
- 原图数值来自知乎原文，与 CuTe v4.6.1 实测不符：v4.6.1 的 left/right_inverse
  stride 恒为 shape 前缀积（无 stride-0 输出）。已按实测重画：
  (a) 两逆均为 (4):(1)（去重投影）；(b) 两逆均为 (4,4):(4,1)（布局双射且对合）。
  caption 同步改并注明「数值按 CuTe v4.6.1 实测校正」。
