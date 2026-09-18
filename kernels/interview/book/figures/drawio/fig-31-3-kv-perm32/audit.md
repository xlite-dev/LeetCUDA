# fig-31-3-kv-perm32（drawio，2026-09）

- 内容：kv_perm32 的 32 列窗口双射置换表（取数步长 0,1,8,9,16,17,24,25,… 双指数节奏）与 perm-aware masking 正误对照——causal 判定必须 pos=kv_tile·Bc+π(j) 回语义域，直接用存储列 j 即上游 SA3 真实 bug（N=512 max_abs 3.3 量级错误），附 fragment 视角（步长-8 取数变连续存储列）与「同一张 π 表贯穿三处」契约。
- 生成器：gen.py（入库副本，drawio-headless -s 3 导出 PNG）。
- 正文章节：ch31《FP4（一）：NVFP4 格式与量化链》。
- 执行模式：主 agent 直做（drawio 管线程序化生成）。
