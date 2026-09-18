# fig-31-1-nvfp4-block（drawio，2026-09）

- 内容：NVFP4 1×16 microscaling 块——16 个 e2m1 数据（4 bit，值域 ±{0.5,1,1.5,2,3,4,6}）共用 1 字节 ue4m3 组尺度（均摊 4.5 bit/元素）的位域，下半 blockscale MMA 语义（SM120 m16n32k64 mxf4nvf4、SF 旁路、K=64 对齐、pad 列零贡献）与尺度工程三件套（microscaling/smoothing/hadamard）。
- 生成器：gen.py（入库副本，drawio-headless -s 3 导出 PNG）。
- 正文章节：ch31《FP4（一）：NVFP4 格式与量化链》。
- 执行模式：主 agent 直做（drawio 管线程序化生成）。
