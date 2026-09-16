#!/usr/bin/env python3
# fig-29-1-scale-folding — fp8 persist-D 的 scale 折叠全链数据流
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

W, H = 1160, 700

f = F()
f.text(24, 8, 1112, 30, "scale 折叠全链：δ_Qδ_K 折 exp2 系数 · v_s 在 P 发射域精确消去 · p_scale 收 epilogue", fs=17, bold=True)

# ===== 主数据流（中部行）=====
y0 = 130
f.box(24, y0, 100, 52, "Q8·δ_Q", fill=LBF, stroke=BLUE, fs=14, bold=True, fc=BLUE)
f.box(24, y0 + 66, 100, 52, "K8·δ_K", fill=LBF, stroke=BLUE, fs=14, bold=True, fc=BLUE)
f.box(190, y0 + 32, 170, 56, "QK^T MMA&#10;(f8f8f32 / s8s8s32)", fill=LGF, stroke=GREEN, fs=13, bold=True, fc="#1e5631")
f.box(424, y0 + 32, 190, 56, "log2 域 online softmax&#10;(exp2 偏移折叠)", fill=LGF, stroke=GREEN, fs=13, bold=True, fc="#1e5631")
f.box(700, y0 + 32, 150, 56, "P 打包 e4m3x2&#10;(perm pack)", fill=LOF, stroke=ORANGE, fs=13, bold=True, fc="#8a4b08")
f.box(910, y0 + 32, 130, 56, "PV MMA&#10;(f8f8f32/f16)", fill=LGF, stroke=GREEN, fs=13, bold=True, fc="#1e5631")
f.line([(124, y0 + 26), (190, y0 + 50)], color=INK, width=2)
f.line([(124, y0 + 92), (190, y0 + 70)], color=INK, width=2)
f.line([(360, y0 + 60), (424, y0 + 60)], color=INK, width=2)
f.line([(614, y0 + 60), (700, y0 + 60)], color=INK, width=2)
f.line([(850, y0 + 60), (910, y0 + 60)], color=INK, width=2)
# V^T from below into PV
f.box(880, y0 + 130, 90, 46, "V8^T·(1/v_s)&#10;(gmem 预量化)", fill=LBF, stroke=BLUE, fs=12.5, bold=True, fc=BLUE)
f.line([(925, y0 + 130), (950, y0 + 88)], color=INK, width=2)

# ===== scale 流 1：δ_Qδ_K → exp2 系数 =====
f.line([(500, y0 + 32), (500, y0 + 6), (560, y0 + 6), (560, y0 + 28)], color=PURPLE, width=2)
f.text(190, y0 - 40, 380, 20, "s_dequant = δ_Q·δ_K·scale·log2 e（单 fp32）", fs=13, bold=True, fc=PURPLE)
f.text(190, y0 - 18, 380, 18, "tile max 归约后一次乘入（kMaxScaleAfter 再省 Bc 次 FMUL）", fs=12, fc="#627d98")

# ===== scale 流 2：v_s 消去 =====
f.line([(880, y0 + 153), (820, y0 + 153), (770, y0 + 92)], color=BLUE, width=2)
f.text(470, y0 + 128, 330, 20, "softmax 直接发射 P̃ = P·v_s·448", fs=13, bold=True, fc=BLUE)
f.text(470, y0 + 148, 340, 20, "exp_offset = log2(v_s·448) 折进 exp2 偏移", fs=12.5, fc="#48586a")
f.text(470, y0 + 168, 340, 20, "(P·v_s/p_s)·(V/v_s) = PV/p_s：v_s 精确消去", fs=12.5, fc="#48586a")
f.text(470, y0 + 190, 360, 18, "gmem 的 V̂ 原样复用（kernel 不读 V 原件）", fs=12, fc="#627d98")

# ===== scale 流 3：p_scale 收尾（epilogue 框）=====
f.box(24, 400, 640, 120, "", fill=LRF, stroke=RED)
f.text(40, 408, 608, 20, "epilogue 一步收尾（fixed：p_s = 1/448 编译期常数）", fs=13.5, bold=True, fc=RED)
f.text(40, 432, 608, 20, "O = o_acc · (1/448) / rowsum        ← 反量化+归一化一步完成", fs=13, fc=INK)
f.text(40, 456, 608, 20, "per-channel V：O ×= vs_d[col]/p_quant_scale，再 += v̄_d（smooth-V）", fs=13, fc=INK)
f.text(40, 480, 608, 20, "lse = (rowmax + log2 rowsum)·ln2  +  scale·δ_Q·q_km（smooth-K 修正）", fs=13, fc=INK)
f.line([(975, y0 + 88), (975, 340), (400, 340), (400, 400)], color=RED, width=2, dashed=1)
f.text(430, 320, 420, 18, "o_acc 全程单一 448× 域（fixed 统一域是 rescale 折进 FFMA 的前提）", fs=12, bold=True, fc=RED)

# ===== 发射域纪律框（右下）=====
f.box(700, 400, 436, 120, "", fill=LPF, stroke=PURPLE)
f.text(716, 408, 404, 20, "发射域纪律（三条硬约束）", fs=13.5, bold=True, fc=PURPLE)
f.text(716, 432, 404, 20, "① lazy rescale 与满量程 P 互斥：2^T 膨胀撞 satfinite", fs=12.5, fc=INK)
f.text(716, 456, 404, 20, "② f16 PV 累加域：kBc·448·2.25 ≤ 65504（kBc=128 时 P 收 224）", fs=12.5, fc=INK)
f.text(716, 480, 404, 20, "③ per-row p_s 满量程 ⟹ 无 2^T headroom ⟹ 与 ① 互斥", fs=12.5, fc=INK)

# ===== 底部结论 =====
f.box(24, 548, 1112, 62, "", fill=LGF, stroke=GREEN)
f.text(44, 554, 1072, 22, "结论：除 Q/K/V/P 各自的 round() 外，无任何额外数学误差源——所有反量化都是精确代数操作", fs=14, bold=True, fc="#1e5631")
f.text(44, 580, 1072, 20, "对照 ffpa-attn(861d75e)：fp8_pscale.cuh L72-300 三步协议 · persist_d.cuh Phase 1-5 · softmax.cuh L44-63", fs=12.5, fc="#627d98")

f.text(24, 630, 1112, 20, "粒度 × scale 读取：per-block ks 每 tile 一标量；per-thread ks 按 lane%4 列组查 4 槽表、qs 按 {r,r+8} 行对查 64 槽表（零 shuffle）", fs=12.5, fc="#48586a")
f.text(24, 656, 1112, 20, "rowsum：tensor-core rowsum MMA（B 全 1）藏进 PV tensor pipe 气泡，CUDA-core FADD 版本实测 -4%", fs=12.5, fc="#48586a")

xml = mk(f.p, W, H, "fig-29-1")
ET.fromstring(xml)
out = sys.argv[1] if len(sys.argv) > 1 else "fig-29-1-scale-folding.drawio"
with open(out, "w") as fh:
    fh.write(xml)
print("written", out)
