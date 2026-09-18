# fig-30-2-m4n2-softmax（drawio，2026-09）

- 内容：M4N2 (4,2,1) atom 的 8-warp 网格（4 M-warp × 2 N-warp，peer=warp_id⊕4）、P 的 SMEM roundtrip（e4m3 写不了 stmatrix → DefaultCopy），以及跨 N-warp softmax 的 smem_exchange 单 barrier 协议——max/sum 两段分置防 RAW 覆写，三步流程只有一次显式 syncthreads（sum 发布搭 P roundtrip 便车）。
- 生成器：gen.py（入库副本，drawio-headless -s 3 导出 PNG）。
- 正文章节：ch30《FP8（三）：split-D 与 M4N2——大 D 的两堵墙》。
- 执行模式：主 agent 直做（drawio 管线程序化生成）。
