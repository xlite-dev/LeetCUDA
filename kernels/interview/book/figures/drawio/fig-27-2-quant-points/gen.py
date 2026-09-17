#!/usr/bin/env python3
# fig-27-2-quant-points — attention 的三个量化点与 scale 流向
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
LBF, LGF, LOF, LRF = "#f0f6fb", "#c9e4c8", "#fde9d0", "#fbe5e5"
INK = "#102a43"
GRAY = "#9fb3c8"

W, H = 1160, 700

f = F()
f.text(24, 8, 1112, 30, "attention 的三个量化点与 scale 流向（无显式反量化 pass）", fs=17, bold=True)

# 主链（y=110 行）：Q K → QK MMA → S → softmax → P → PV MMA → O
y0 = 110
f.box(24, y0, 90, 52, "Q", fill=LBF, stroke=BLUE, fs=17, bold=True, fc=BLUE)
f.box(24, y0 + 64, 90, 52, "K", fill=LBF, stroke=BLUE, fs=17, bold=True, fc=BLUE)
f.box(190, y0 + 32, 150, 56, "QK^T MMA", fill=LGF, stroke=GREEN, fs=16, bold=True, fc="#1e5631")
f.box(420, y0 + 32, 60, 56, "Ŝ", fill="#ffffff", stroke=GRAY, fs=17, bold=True, fc=INK)
f.box(540, y0 + 32, 150, 56, "softmax&#10;(exp2 域)", fill=LGF, stroke=GREEN, fs=15, bold=True, fc="#1e5631")
f.box(740, y0 + 32, 60, 56, "P", fill="#ffffff", stroke=GRAY, fs=17, bold=True, fc=INK)
f.box(860, y0 + 32, 150, 56, "PV MMA", fill=LGF, stroke=GREEN, fs=16, bold=True, fc="#1e5631")
f.box(1060, y0 + 32, 76, 56, "O", fill=LOF, stroke=ORANGE, fs=17, bold=True, fc="#8a4b08")
f.line([(114, y0 + 26), (190, y0 + 56)], color=INK, width=2)
f.line([(114, y0 + 90), (190, y0 + 66)], color=INK, width=2)
f.line([(340, y0 + 60), (420, y0 + 60)], color=INK, width=2)
f.line([(480, y0 + 60), (540, y0 + 60)], color=INK, width=2)
f.line([(690, y0 + 60), (740, y0 + 60)], color=INK, width=2)
f.line([(800, y0 + 60), (860, y0 + 60)], color=INK, width=2)
f.line([(1010, y0 + 60), (1060, y0 + 60)], color=INK, width=2)

# V 从下方进入 PV MMA
f.box(890, y0 + 130, 90, 52, "V", fill=LBF, stroke=BLUE, fs=17, bold=True, fc=BLUE)
f.line([(935, y0 + 130), (935, y0 + 92)], color=INK, width=2)

# --- 量化点 1 标注（Q/K 离线，右上图区）---
f.box(560, 36, 576, 62, "", fill="#f7f9fb", stroke=GRAY)
f.text(572, 42, 552, 20, "量化点 1（离线）：Q, K → e4m3 / int8", fs=13.5, bold=True, fc=INK)
f.text(572, 64, 552, 20, "per-thread 粒度与 mma fragment 对齐 → scale 零 shuffle 使用", fs=13, fc="#48586a")

# scale 流 1：δ_Qδ_K 折进 exp2 系数（softmax 上方）
f.line([(480, y0 + 8), (640, y0 + 8), (640, y0 + 28)], color=PURPLE, width=2)
f.text(190, y0 - 22, 360, 20, "s_dequant = scale·δ_Q·δ_K·log2 e（单个 fp32）", fs=13, bold=True, fc=PURPLE)

# scale 流 2：v_s 折进 P 发射（V 左侧图区）
f.line([(890, y0 + 156), (700, y0 + 156), (770, y0 + 92)], color=BLUE, width=2)
f.text(470, y0 + 130, 400, 20, "v_s 折进 P 发射：P̃ = P·v_s·448", fs=13, bold=True, fc=BLUE)
f.text(470, y0 + 152, 410, 20, "(P·v_s/p_s)·(V/v_s) = PV/p_s —— v_s 精确消去", fs=12.5, fc="#48586a")
f.text(830, y0 + 186, 310, 20, "量化点 3（离线）：V → e4m3（per-channel）", fs=13, bold=True, fc=INK)
f.text(830, y0 + 208, 310, 20, "V̂ = V/v_s 驻留 gmem，原样复用", fs=12.5, fc="#48586a")

# --- 量化点 2 标注（P 在线，左下图区；先画框后画文字，避免 COVER）---
f.box(24, 330, 520, 112, "", fill=LRF, stroke=RED)
f.text(38, 338, 492, 20, "量化点 2（在线）：P 每 tile 重算 → 无法离线", fs=13.5, bold=True, fc=RED)
f.text(38, 360, 492, 20, "fixed：p_s ≡ 1/448（编译期常数），o_acc 单一域直接累加；", fs=13, fc="#48586a")
f.text(38, 382, 492, 20, "epilogue 一步 O = o_acc·(1/448)/ℓ 完成反量化+归一化", fs=13, fc="#48586a")
f.text(38, 404, 492, 20, "per-row：p_s[row] = rowmax(P)/448，满量程精度最优（费寄存器）", fs=13, fc="#48586a")
f.line([(770, y0 + 92), (770, 330), (548, 330)], color=RED, width=2, dashed=1)

# scale 流 3：p_scale 收尾（O 下方弧线）
f.line([(1035, y0 + 92), (1035, y0 + 118), (1098, y0 + 118), (1098, y0 + 92)], color=ORANGE, width=2, dashed=1)
f.text(620, y0 + 96, 400, 20, "p_s 收尾：O += MMA·p_s（fixed：并入 epilogue 常数）", fs=13, bold=True, fc=ORANGE)

# --- 底部要点框（先框后文字）---
f.box(24, 470, 1112, 96, "", fill="#f7f9fb", stroke=GRAY)
f.text(44, 478, 1072, 22, "要点 1：δ_Qδ_K 折进 exp2 系数（softmax 前一次乘入，kMaxScaleAfter 再省 B_c 次 FMUL）", fs=14, fc=INK)
f.text(44, 502, 1072, 22, "要点 2：v_s 在 P 侧精确消去 → gmem 的 V̂ 不动；p_s 只在 epilogue 出现一次", fs=14, fc=INK)
f.text(44, 526, 1072, 22, "结论：除量化本身 round() 外无任何额外数学误差源——所有反量化都是精确代数操作", fs=14, bold=True, fc="#1e5631")

f.text(24, 580, 1112, 20, "溢出约束（fixed + FA-4 lazy rescale）：2^T · amax(V) ≤ 448，T=4 ⇒ amax(V) ≤ 28，残余超调 satfinite 兜底", fs=13.5, fc=RED)
f.text(24, 604, 1112, 20, "互斥约束：per-row 满量程发射没有 2^T headroom —— per-row 与 lazy rescale 不能同时开", fs=13.5, fc=RED)
f.text(24, 648, 1112, 20, "对照 ffpa-attn：cute/fp8/fp8_pscale.cuh L17-37（三步协议）；kernel 化展开见 ch29（persist-D）", fs=12.5, fc="#627d98")

xml = mk(f.p, W, H, "fig-27-2")
ET.fromstring(xml)
out = sys.argv[1] if len(sys.argv) > 1 else "fig-27-2-quant-points.drawio"
with open(out, "w") as fh:
    fh.write(xml)
print("written", out)
