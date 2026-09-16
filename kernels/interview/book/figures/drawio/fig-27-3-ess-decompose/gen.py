#!/usr/bin/env python3
# fig-27-3-ess-decompose — ESS 误差模型与 per-stage 误差分解
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

W, H = 1160, 620

f = F()
f.text(24, 8, 1112, 30, "ESS 误差模型：causal 早行的绝对误差放大 与 per-stage 误差分解", fs=17, bold=True)
f.text(24, 42, 1112, 22, "Var(O[i,d]) = σ_V² · Σ_j P[i,j]² = σ_V² / ESS_i —— 输出幅度 ≈ 3.1/√ESS（D=128, σ_V=1）", fs=14, fc="#48586a", align="center")

# ================= 左半：dense vs causal 早行 =================
f.box(24, 80, 530, 36, "左：同一模型，两类行（ε = 5% 恒定）", fill="#d9e2ec", stroke="#486581", fs=14.5, bold=True, fc=INK)

# dense 行卡片
f.box(24, 128, 530, 190, "", fill="#f7f9fb", stroke="#9fb3c8")
f.text(40, 136, 500, 22, "dense 行：P ≈ uniform over 8192", fs=14.5, bold=True, fc=INK)
# P 分布示意（16 根矮柱）
bx, by = 40, 166
for i in range(16):
    f.box(bx + i * 16, by + 28 - 12, 12, 12, "", fill=BLUE, stroke=BLUE)
f.line([(bx, by + 28), (bx + 256, by + 28)], color="#9fb3c8", width=1, arrow_end=0)
f.text(310, 158, 240, 20, "ESS ≈ 3000", fs=14, bold=True, fc=BLUE)
f.text(310, 180, 240, 20, "幅度 ≈ 0.05", fs=14, fc="#48586a")
f.text(310, 202, 240, 20, "绝对误差 ≈ 0.003", fs=14, fc="#48586a")
f.text(310, 224, 240, 20, "（max_abs 实测 0.015）", fs=12.5, fc="#627d98")
f.text(40, 236, 260, 20, "输出像大样本均值", fs=12.5, fc="#627d98")

# causal row[0] 卡片
f.box(24, 330, 530, 190, "", fill=LRF, stroke=RED)
f.text(40, 338, 500, 22, "causal row[0]：P = one-hot（只能看自己）", fs=14.5, bold=True, fc=RED)
bx, by = 40, 368
for i in range(16):
    hgt = 64 if i == 0 else 4
    f.box(bx + i * 16, by + 64 - hgt, 12, hgt, "", fill=RED if i == 0 else "#e7ecf3", stroke=RED if i == 0 else "#e7ecf3")
f.line([(bx, by + 64), (bx + 256, by + 64)], color="#9fb3c8", width=1, arrow_end=0)
f.text(310, 360, 240, 20, "ESS = 1", fs=14, bold=True, fc=RED)
f.text(310, 382, 240, 20, "幅度 ≈ 3.1", fs=14, fc="#48586a")
f.text(310, 404, 240, 20, "绝对误差 ≈ 0.16", fs=14, fc="#48586a")
f.text(310, 426, 240, 20, "（max_abs 实测 0.22）", fs=12.5, fc="#627d98")
f.text(40, 438, 260, 20, "输出 ≈ 单个 V 行", fs=12.5, fc="#627d98")

# 中间结论箭头
f.line([(554, 300), (590, 300)], color=INK, width=2)
f.text(24, 530, 530, 22, "15× 的 abs 差距全部来自幅度；相对误差两者都是 ~5%", fs=13.5, bold=True, fc=INK)
f.text(24, 554, 530, 20, "（causal 5.5% vs dense 9.1%，PyTorch 仿真 B1H32N8192D128）", fs=12.5, fc="#627d98")

# ================= 右半：per-stage 误差分解 =================
f.box(578, 80, 558, 36, "右：误差贡献分解（独立 ε × 幅度）", fill="#d9e2ec", stroke="#486581", fs=14.5, bold=True, fc=INK)

f.box(578, 128, 558, 200, "", fill="#f7f9fb", stroke="#9fb3c8")
f.text(594, 136, 520, 22, "单级贡献（早行幅度 ~1.3 下）", fs=14, bold=True, fc=INK)
# 水平条形：V 0.19 / QK 0.13 / P 0.11
bars = [("V 量化", 0.19, RED), ("QK 量化", 0.13, ORANGE), ("P 量化", 0.11, BLUE)]
maxw = 330
for i, (name, v, c) in enumerate(bars):
    yy = 168 + i * 44
    f.text(594, yy, 90, 22, name, fs=13.5, fc=INK)
    f.box(690, yy, int(v / 0.19 * maxw), 26, "", fill=c, stroke=c)
    f.text(690 + int(v / 0.19 * maxw) + 10, yy, 90, 22, "%.2f" % v, fs=14, bold=True, fc=c)
f.text(594, 300, 520, 20, "V 0.19 > QK 0.13 > P 0.11 —— V 量化是最大单项", fs=13.5, bold=True, fc=INK)

f.box(578, 340, 558, 120, "", fill=LOF, stroke=ORANGE)
f.text(594, 348, 526, 22, "推论（hybrid 的理论依据）：", fs=14, bold=True, fc="#8a4b08")
f.text(594, 372, 526, 20, "「早行只换 QK 回 fp16」无效——V 项不动；", fs=13.5, fc="#48586a")
f.text(594, 394, 526, 20, "早行必须全链 fp16（hybrid 行分割，ch29）", fs=13.5, fc="#48586a")
f.text(594, 416, 526, 20, "或 per-channel V scale 压 V 相对误差（ch28）", fs=13.5, fc="#48586a")

f.text(578, 480, 558, 22, "n_early 不需很大：ESS 极小的行集中在前段（前 256 行已覆盖）", fs=13, fc="#48586a")
f.text(578, 504, 558, 22, "fp4 同源：ε 大一个量级，长尾由 V 主导、与行号无关", fs=13, fc="#48586a")

f.text(24, 580, 1112, 20, "对照 ffpa-attn：cute/fp8/sm_120/persist_d.cuh L19-49 头注释（PyTorch sim 数字与 ESS 推导）", fs=12.5, fc="#627d98")

xml = mk(f.p, W, H, "fig-27-3")
ET.fromstring(xml)
out = sys.argv[1] if len(sys.argv) > 1 else "fig-27-3-ess-decompose.drawio"
with open(out, "w") as fh:
    fh.write(xml)
print("written", out)
