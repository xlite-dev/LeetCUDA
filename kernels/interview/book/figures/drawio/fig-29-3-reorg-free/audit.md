# fig-29-3-reorg-free（drawio，2026-09）

- 内容：P→A-fragment 布局重排两方案对照——方案 A 跨 lane ReorgC8bitToA8bit（16×shfl+32×byte_perm，压在关键路径）vs 方案 B 就地 PackC8bitToA8bitPermVT（归约轴双射不变性 Σ_k P[m,k]V[k,n] 对 π 不变，V^T 列置换预铺），底部 launcher 编译期常量 reorg_free 配对契约（错配=静默错值）。
- 生成器：gen.py（入库副本，drawio-headless -s 3 导出 PNG）。
- 正文章节：ch29《FP8（二）：persist-D 主 Kernel 与 Scale 折叠代数》。
- 执行模式：主 agent 直做（drawio 管线程序化生成）。
