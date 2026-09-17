#!/usr/bin/env python3
# fig-28-1-quant-pipeline — fp8 前处理链：三泳道 aux kernel 与量化 workspace
import xml.etree.ElementTree as ET
import sys

def mk(parts, W, H, did):
    return ('<mxfile host="app.diagrams.net"><diagram id="%s" name="%s">'
            '<mxGraphModel dx="800" dy="600" grid="0" page="1" pageWidth="%d" pageHeight="%d" math="0" shadow="0">'
            '<root><mxCell id="0"/><mxCell id="1" parent="0"/>%s'
            '</root></mxGraphModel></diagram></mxfile>') % (did, did, W, H, ''.join(parts))

class F:
    def __init__(self):
        self.p = []
    def box(self, x, y, w, h, t="", fill="#ffffff", stroke="#bcccdc", fs=13, bold=False, fc="#1f2933"):
        st = ('rounded=0;whiteSpace=wrap;html=1;fillColor=%s;strokeColor=%s;'
              'fontSize=%d;fontColor=%s;align=center;verticalAlign=middle;fontFamily=Helvetica') % (fill, stroke, fs, fc)
        if bold: st += ';fontStyle=1'
        self.p.append('<mxCell value="%s" style="%s" vertex="1" parent="1">'
                      '<mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry"/></mxCell>' % (t, st, x, y, w, h))
    def text(self, x, y, w, h, t, fs=13, fc="#102a43", bold=False, align="left"):
        st = 'text;html=1;align=%s;verticalAlign=middle;fontSize=%d;fontColor=%s;fontFamily=Helvetica' % (align, fs, fc)
        if bold: st += ';fontStyle=1'
        self.p.append('<mxCell value="%s" style="%s" vertex="1" parent="1">'
                      '<mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry"/></mxCell>' % (t, st, x, y, w, h))
    def line(self, pts, color="#2171b5", width=2, dashed=0, arrow_end=1):
        ae = "blockThin;endFill=1" if arrow_end else "none"
        st = 'endArrow=%s;html=1;strokeColor=%s;strokeWidth=%d;rounded=0;dashed=%d' % (ae, color, width, dashed)
        self.p.append('<mxCell value="" style="%s" edge="1" parent="1">'
                      '<mxGeometry relative="1" as="geometry"><Array as="points">'
                      % st + ''.join('<mxPoint x="%d" y="%d"/>' % (x, y) for x, y in pts[1:-1])
                      + '</Array><mxPoint x="%d" y="%d" as="sourcePoint"/>'
                        '<mxPoint x="%d" y="%d" as="targetPoint"/></mxGeometry></mxCell>' % (pts[0][0], pts[0][1], pts[-1][0], pts[-1][1]))

BLUE, GREEN, ORANGE, RED, PURPLE = "#2171b5", "#2e8540", "#d97706", "#dc2626", "#6a51a3"
LBF, LGF, LOF, LRF, LPF = "#f0f6fb", "#c9e4c8", "#fde9d0", "#fbe5e5", "#f1eaf8"
INK = "#102a43"
GRAY = "#9fb3c8"

W, H = 1160, 710

f = F()
f.text(24, 8, 1112, 30, "fp8 前处理链：三泳道 aux kernel 与量化 workspace（约 13 个 kernel / forward）", fs=17, bold=True)

# ============ 输入列（左） ============
f.box(24, 96, 130, 46, "Q  fp16/bf16&#10;(B,H,N,D_og)", fill=LBF, stroke=BLUE, fs=13.5, bold=True, fc=BLUE)
f.box(24, 176, 130, 46, "K  fp16/bf16&#10;(B,H,N,D_og)", fill=LBF, stroke=BLUE, fs=13.5, bold=True, fc=BLUE)
f.box(24, 256, 130, 46, "V  fp16/bf16&#10;(B,H,N,D_og)", fill=LBF, stroke=BLUE, fs=13.5, bold=True, fc=BLUE)
f.box(24, 330, 130, 40, "NHD view&#10;零拷贝寻址", fill="#f7f9fb", stroke=GRAY, fs=12.5, fc="#48586a")
f.line([(89, 330), (89, 302)], color=GRAY, width=1.5, dashed=1, arrow_end=0)

# ============ Q 泳道（y≈96） ============
f.box(200, 88, 190, 62, "wht_qk_kernel（可选）&#10;Hadamard 蝶形旋转&#10;显式开启（fp4 收益大）", fill=LPF, stroke=PURPLE, fs=12.5)
f.box(434, 88, 220, 62, "quantize_fp8_q&#10;per-block: 1 scale/128 行&#10;per-thread: 64 scale/128 行", fill=LGF, stroke=GREEN, fs=12.5)
f.box(700, 96, 150, 46, "Q8 e4m3/int8&#10;(B,H,N,D) 行主序", fill=LOF, stroke=ORANGE, fs=12.5, bold=True)
f.line([(154, 119), (200, 119)], color=INK, width=2)
f.line([(390, 119), (434, 119)], color=INK, width=2)
f.line([(654, 119), (700, 119)], color=INK, width=2)

# ============ K 泳道（y≈176，先统计后融合量化） ============
f.box(200, 168, 190, 62, "kv_col_sum + finalize&#10;两阶段列均值 k̄&#10;~50us（无 atomics）", fill=LPF, stroke=PURPLE, fs=12.5)
f.box(434, 168, 220, 62, "quantize_fp8_k（fused）&#10;fp32 域减 k̄ 再量化&#10;不物化 K&#39; 中间量", fill=LGF, stroke=GREEN, fs=12.5)
f.box(700, 176, 150, 46, "K8 e4m3/int8&#10;(B,H,N,D) 行主序", fill=LOF, stroke=ORANGE, fs=12.5, bold=True)
f.line([(154, 199), (200, 199)], color=INK, width=2)
f.line([(390, 199), (434, 199)], color=INK, width=2)
f.line([(654, 199), (700, 199)], color=INK, width=2)
# km_f32 侧输出（lse 用）
f.line([(544, 230), (544, 252), (700, 252)], color=PURPLE, width=1.5, dashed=1)
f.text(706, 242, 198, 18, "km_f32（lse 修正用）", fs=12, fc=PURPLE, bold=True)

# ============ V 泳道（y≈300，统计 + 转置量化） ============
f.box(200, 322, 190, 62, "v_col_stats + finalize&#10;per-channel: sum/max/min&#10;amax=max(|max−μ|,|min−μ|)", fill=LPF, stroke=PURPLE, fs=12)
f.box(434, 322, 220, 62, "quantize_fp8_vt&#10;smem staging 转置 (N,D)→(D,N)&#10;kPad=16 · VTPerm 列置换", fill=LGF, stroke=GREEN, fs=12.5)
f.box(700, 322, 150, 62, "V8^T e4m3&#10;flat [B·Hkv·D, N_pad]&#10;TMA 16B 对齐", fill=LOF, stroke=ORANGE, fs=12, bold=True)
f.line([(154, 353), (200, 353)], color=INK, width=2)
f.line([(390, 353), (434, 353)], color=INK, width=2)
f.line([(654, 353), (700, 353)], color=INK, width=2)

# ============ scale 汇流（右侧） ============
f.box(910, 96, 226, 288, "", fill="#f7f9fb", stroke=GRAY)
f.text(926, 106, 196, 22, "Fp8QuantizedInputs", fs=14.5, bold=True, fc=INK)
f.text(926, 136, 196, 20, "q8, k8, vt8（workspace）", fs=12.5, fc="#48586a")
f.text(926, 160, 196, 20, "q_scale / k_scale（粒度随 knob）", fs=12.5, fc="#48586a")
f.text(926, 184, 196, 20, "v_scale: (bh,n_rb) 或 (bh,D)", fs=12.5, fc="#48586a")
f.text(926, 208, 196, 20, "km_f32 → epilogue lse 修正", fs=12.5, fc=PURPLE)
f.text(926, 232, 196, 20, "vm → O 加回均值行", fs=12.5, fc=PURPLE)
f.text(926, 264, 196, 18, "布局由消费方决定：", fs=12, bold=True, fc="#1e5631")
f.text(926, 284, 196, 18, "行主序 / 列主序 /", fs=12, bold=True, fc="#1e5631")
f.text(926, 304, 196, 18, "fragment 槽位", fs=12, bold=True, fc="#1e5631")
f.line([(850, 119), (910, 140)], color=ORANGE, width=2)
f.line([(850, 199), (910, 199)], color=ORANGE, width=2)
f.line([(850, 353), (910, 300)], color=ORANGE, width=2)
f.line([(946, 384), (946, 420)], color=INK, width=2.5)
f.box(820, 420, 320, 50, "persist-D / split-D / M4N2&#10;三个 fp8 主 kernel 家族共享（ch29-30）", fill=LBF, stroke=BLUE, fs=13, bold=True, fc=BLUE)

# ============ 融合原则框（底部） ============
f.box(24, 500, 1112, 92, "", fill=LGF, stroke=GREEN)
f.text(44, 508, 1072, 22, "融合原则：统计先行、变换融进量化 —— 永远不物化 K&#39;=K−1k̄^T 或 V&#39; 中间张量（省一整轮 DRAM 往返）", fs=14, fc=INK, bold=True)
f.text(44, 534, 1072, 22, "单遍寄存器驻留：16B 向量读进寄存器后 amax 与量化同遍完成，全局内存只读一次", fs=13.5, fc=INK)
f.text(44, 560, 1072, 22, "VTPerm 列置换：为 ch30 的 reorg-free PV 打包预铺列序（launcher 强制配对，防静默错值）", fs=13.5, fc=INK)

f.text(24, 608, 1112, 20, "带宽账（B1 H32 N8192 D128）：输入 ~192MB fp16 → 输出 ~96MB fp8；聚合有效带宽 ~1.04 TB/s（贴峰）", fs=13.5, fc="#48586a")
f.text(24, 632, 1112, 20, "占比：aux 链 ~0.3ms vs 主 kernel 3.0ms（N=8192）≈10%；N 翻倍时 O(N) vs O(N²) → 占比衰减", fs=13.5, fc="#48586a")
f.text(24, 676, 1112, 20, "对照 ffpa-attn(861d75e)：cute/fp8/{smooth_k, smooth_v, quantize_fp8, prepare_inputs, input_layout}.cuh", fs=12.5, fc="#627d98")

xml = mk(f.p, W, H, "fig-28-1")
ET.fromstring(xml)
out = sys.argv[1] if len(sys.argv) > 1 else "fig-28-1-quant-pipeline.drawio"
with open(out, "w") as fh:
    fh.write(xml)
print("written", out)
