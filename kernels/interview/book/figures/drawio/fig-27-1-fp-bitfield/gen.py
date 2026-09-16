#!/usr/bin/env python3
# fig-27-1-fp-bitfield — 四种量化格式的位域与值域对照（int8/e4m3/e2m1/ue4m3）
import xml.etree.ElementTree as ET

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

BLUE, GREEN, ORANGE, RED, GRAY = "#2171b5", "#2e8540", "#d97706", "#dc2626", "#829ab1"
LBF, LGF, LOF, LRF = "#f0f6fb", "#c9e4c8", "#fde9d0", "#fbe5e5"
INK = "#102a43"

W, H = 1160, 660

f = F()
f.text(24, 10, 1112, 30, "四种量化格式的位域与值域（蓝 = 符号 S / 橙 = 指数 E / 绿 = 尾数 M）", fs=17, bold=True)

# 行模板：行名 + bit 格 + 注释
def bits(x0, y0, spec, cw=44, chh=40):
    # spec: list of (label, fill, stroke)
    x = x0
    for lab, fill, stroke in spec:
        f.box(x, y0, cw, chh, lab, fill=fill, stroke=stroke, fs=15, bold=True, fc=INK)
        x += cw
    return x

# --- row 1: int8 ---
y = 70
f.text(24, y + 6, 120, 30, "int8（对称）", fs=15, bold=True)
xe = bits(150, y, [("d7", LBF, BLUE)] * 8)
f.text(xe + 24, y - 4, 999 - xe, 22, "均匀整数网格 ±127；无指数/尾数之分", fs=13.5, fc="#48586a")
f.text(xe + 24, y + 20, 999 - xe, 22, "误差只依赖步长 δ：RMS = δ/√12（绝对误差恒定）", fs=13.5, fc="#48586a")
f.text(1010, y + 6, 130, 30, "SA1：QK", fs=13, fc=GRAY)

# --- row 2: e4m3 ---
y = 190
f.text(24, y + 6, 120, 30, "e4m3（FP8）", fs=15, bold=True)
spec = [("S", LBF, BLUE)] + [("E", LOF, ORANGE)] * 4 + [("M", LGF, GREEN)] * 3
xe = bits(150, y, spec)
f.text(xe + 24, y - 4, 999 - xe, 22, "bias=7；max finite = 1.75×2^8 = 448（E 全 1 时仅 M=111 为 NaN）", fs=13.5, fc="#48586a")
f.text(xe + 24, y + 20, 999 - xe, 22, "相对步长 2^-(m+1) = 2^-4 = 6.25%（相对误差恒定）", fs=13.5, fc="#48586a")
f.text(1010, y + 6, 140, 30, "SA2：Q/K/V/P", fs=13, fc=GRAY)

# --- row 3: e2m1 ---
y = 310
f.text(24, y + 6, 120, 30, "e2m1（FP4）", fs=15, bold=True)
spec = [("S", LBF, BLUE)] + [("E", LOF, ORANGE)] * 2 + [("M", LGF, GREEN)] * 1
xe = bits(150, y, spec)
f.text(xe + 24, y - 4, 999 - xe, 22, "仅 16 个码点；正值 {0.5, 1, 1.5, 2, 3, 4, 6}——一个倍频程只有两格", fs=13.5, fc="#48586a")
f.text(xe + 24, y + 20, 999 - xe, 22, "相对步长 2^-2 = 25%；动态范围最窄 → smoothing 强制", fs=13.5, fc="#48586a")
f.text(1010, y + 6, 140, 30, "SA3：Q/K/V/P", fs=13, fc=GRAY)

# e2m1 数轴（等距排列示清楚；注明非等比）
axis_y = 386
f.line([(150, axis_y), (1010, axis_y)], color=GRAY, width=1, arrow_end=0)
vals = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
xs = [150 + i * 118 for i in range(len(vals))]
for v, x in zip(vals, xs):
    f.line([(x, axis_y - 6), (x, axis_y + 6)], color=GRAY, width=1, arrow_end=0)
    lab = "0" if v == 0 else ("%g" % v)
    f.text(x - 24, axis_y + 8, 48, 18, lab, fs=12.5, fc="#48586a", align="center")
f.text(150, axis_y + 28, 860, 18, "e2m1 全部正值格点（等距排列以示清楚；倍频程 [2,4) 内只有 {2,3} 两个格点）", fs=12.5, fc=GRAY)

# --- row 4: ue4m3 ---
y = 470
f.text(24, y + 6, 120, 30, "ue4m3", fs=15, bold=True)
spec = [("E", LOF, ORANGE)] * 4 + [("M", LGF, GREEN)] * 3
xe = bits(150, y, spec)
f.text(xe + 24, y - 4, 999 - xe, 22, "无符号：2^-6 ~ 448；用作 NVFP4 的 per-16 block scale（SF）", fs=13.5, fc="#48586a")
f.text(xe + 24, y + 20, 999 - xe, 22, "scale 乘性作用：两级相对误差相加（两级 P 量化核算的账，ch31）", fs=13.5, fc="#48586a")
f.text(1010, y + 6, 140, 30, "SA3：SF", fs=13, fc=GRAY)

# --- 定理框 ---
f.box(24, 556, 1112, 84, "", fill="#f7f9fb", stroke="#9fb3c8")
f.text(44, 564, 1072, 26, "相对步长定理：规格化数 x ∈ [2^e, 2^(e+1))，m 位尾数 ⇒ ULP = 2^(e−m)，RN 误差 ≤ ULP/2", fs=14.5, bold=True, fc=INK)
f.text(44, 592, 1072, 26, "相对误差 ≤ 2^(e−m−1) / 2^e = 2^−(m+1)：与 e 无关——e4m3 ≈ 6.25%，e2m1 ≈ 25%；int8 无此性质（均匀网格）", fs=14.5, fc="#48586a")

xml = mk(f.p, W, H, "fig-27-1")
ET.fromstring(xml)  # well-formed 自检
import sys
out = sys.argv[1] if len(sys.argv) > 1 else "fig-27-1-fp-bitfield.drawio"
with open(out, "w") as fh:
    fh.write(xml)
print("written", out)
