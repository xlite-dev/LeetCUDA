# figures/ffpa/ 来源登记（RFC-K0.2，2026-09-16）

来源仓库：ffpa-attn（https://github.com/xlite-dev/ffpa-attn），锚定 commit `861d75e`。
测试硬件：NVIDIA GeForce RTX 5090（sm_120）；原图生成脚本与口径见 ffpa-attn README
「Performance」节与 `bench/bench_fp8.py` / `bench/bench_fp4.py`。

| 文件 | 源路径（ffpa-attn） | 引用章节 |
|---|---|---|
| fp8/ffpa_speedup_..._D{64,128,256,320,512,768}_T.png | docs/assets/perf/fp8/ | ch33（fp8 speedup 主图）；D768 另用于 ch30 cross-point |
| fp4/ffpa_speedup_..._D{64,128,192,256,320,512}_T.png | docs/assets/perf/fp4/ | ch33（fp4 speedup 主图） |
| ffpa-split-d.png | docs/assets/ffpa-split-d.png | ch19、ch30（split-D 切分示意） |
| mma.png | docs/assets/mma.png | ch30（寄存器压力 / M4N2 vs M8N1 交叉） |

注意：fp8 与 fp4 的 speedup 图**同名**（目录不同），必须按子目录归档引用，
严禁平铺复制（同名静默覆盖）。

使用约定：
- 直接引用 + 正文出处标注（「图源：ffpa-attn README，RTX 5090 实测」），不二次描摹；
- 本机 PRO 5000 复测数据以 matplotlib 重绘（口径见 ch33 与附录 B），与本目录 5090
  数据并列展示，注明 GPU 差异；
- 商用/再分发需遵守 ffpa-attn 仓库许可证。
