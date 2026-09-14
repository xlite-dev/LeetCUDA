#!/usr/bin/env python3
# gen_batch_f.py — FIG-20-4..20-7：知乎 @竹熙佳处《Layout Compose & Inverse》四图重建
# 来源：figures/zhihu/zhuxijiachu-layout-compose-inverse/fig-{1,2,3,4}.jpg（1440px 原图已归档）
# 重建原则：保留教学语义（网格/数字/追踪高亮/步骤箭头），配色映射为书内系列色；水印剔除
import os, sys
import xml.etree.ElementTree as ET

parts = []
def P(s): parts.append(s)

def box(x, y, w, h, t="", fill="#ffffff", stroke="#bcccdc", fs=13, bold=False, fc="#1f2933", dashed=0):
    st = (f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
          f"fontSize={fs};fontColor={fc};align=center;verticalAlign=middle;fontFamily=Helvetica")
    if bold: st += ";fontStyle=1"
    if dashed: st += ";dashed=1"
    P(f'<mxCell value="{t}" style="{st}" vertex="1" parent="1">'
      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def text(x, y, w, h, t, fs=13, fc="#102a43", bold=False):
    st = f"text;html=1;align=left;verticalAlign=middle;fontSize={fs};fontColor={fc};fontFamily=Helvetica"
    if bold: st += ";fontStyle=1"
    P(f'<mxCell value="{t}" style="{st}" vertex="1" parent="1">'
      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def arrow(pts, color="#48586a", width=2, dashed=0, both=0):
    st = (f"endArrow=blockThin;endFill=1;html=1;strokeColor={color};strokeWidth={width};rounded=0;")
    if both: st = "startArrow=blockThin;startFill=1;" + st
    if dashed: st += "dashed=1;"
    P(f'<mxCell value="" style="{st}" edge="1" parent="1">'
      f'<mxGeometry relative="1" as="geometry"><Array as="points">'
      + ''.join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts[1:-1])
      + f'</Array><mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint"/>'
        f'<mxPoint x="{pts[-1][0]}" y="{pts[-1][1]}" as="targetPoint"/></mxGeometry></mxCell>')

BLUE, GREEN, ORANGE, RED, PURPLE = "#2171b5", "#2e8540", "#d97706", "#dc2626", "#6a51a3"
LBF, LGF, LOF, LRF = "#bfd7ea", "#c9e4c8", "#fde9d0", "#fdeaea"
GRAY = "#e7ecf3"

def grid(x0, y0, cw, ch, nr, nc, vals=None, hi=None, stroke="#bcccdc", fs=12.5):
  """vals: nr*nc 列表或 None；hi: (r,c) 高亮格"""
  for r in range(nr):
    for c in range(nc):
      v = "" if vals is None else str(vals[r * nc + c])
      f, s, fc = "#ffffff", stroke, "#1f2933"
      if hi and (r, c) == hi:
        f, s, fc = LGF, GREEN, "#1e5631"
      box(x0 + c * cw, y0 + r * ch, cw - 2, ch - 2, v, fill=f, stroke=s, fs=fs, fc=fc)

def axis_h(x1, x2, y, label):
  arrow([(x1, y), (x2, y)], color="#627d98", width=1.2, both=1)
  text(x1, y - 20, x2 - x1, 18, label, fs=13, fc="#627d98")

def axis_v(y1, y2, x, label):
  arrow([(x, y1), (x, y2)], color="#627d98", width=1.2, both=1)
  text(x - 30, (y1 + y2) // 2 - 9, 24, 18, label, fs=13, fc="#627d98")

def fig_20_4():
  parts.clear()
  text(30, 12, 1060, 26, "CuTe Tensor = Layout + Engine：坐标 -&gt; offset -&gt; 解引用取数", fs=17, bold=True)
  text(30, 42, 1060, 20, "Layout 只定义「坐标 -&gt; 偏移」的映射规则，Engine 提供可寻址的存储（base_ptr）；二者相加才是一次真实的访存", fs=12.5, fc="#48586a")
  # 4x4 字母网格：行列板双色编码 a-h / A-H
  GX, GY, CW, CH = 190, 130, 66, 50
  plate0 = [RED, PURPLE, BLUE, ORANGE]     # 行 0/2 色板
  plate1 = ["#846358", "#2f9e44", "#b8860b", "#0e7490"]  # 行 1/3 色板
  letters = ["a","b","c","d","e","f","g","h","A","B","C","D","E","F","G","H"]
  for r in range(4):
    for c in range(4):
      L = letters[r * 4 + c]
      col = (plate0 if r in (0, 2) else plate1)[c]
      f, fc = "#ffffff", col
      if L == "f": f, fc = LGF, "#1e5631"
      box(GX + c * CW, GY + r * CH, CW - 2, CH - 2, f"'{L}'", fill=f, stroke=col, fs=14, fc=fc)
  axis_h(GX, GX + 4 * CW, GY - 18, "N")
  axis_v(GY, GY + 4 * CH, GX - 22, "M")
  # 网格下方定义
  text(GX, GY + 4 * CH + 14, 300, 20, "layoutA（逻辑视图，元素带引号）", fs=12, fc=RED)
  # 右侧算式块
  TX = 560
  text(TX, 120, 420, 20, "shape: (M, N), i.e. (4, 4)", fs=14, fc=BLUE)
  text(TX, 148, 420, 20, "stride: (N, 1), i.e. (4, 1)", fs=14, fc=BLUE)
  text(TX, 196, 420, 20, "offset = m * N + n * 1", fs=15, fc=RED, bold=True)
  text(TX, 240, 420, 20, "e.g. given input (1, 1),", fs=13.5, fc=BLUE)
  text(TX, 268, 420, 20, "we got offset = 1 * 4 + 1 * 1 = 5", fs=13.5, fc=BLUE)
  # 左下：Tensor 分解
  box(40, 440, 140, 36, "CuTe Tensor", fill="#ffffff", stroke=GREEN, fs=14, fc=GREEN, bold=True)
  arrow([(180, 458), (250, 458), (250, 402)], color=RED, width=1.6, dashed=1)
  arrow([(250, 458), (250, 514)], color=RED, width=1.6, dashed=1)
  box(260, 384, 150, 36, "Layout", fill="#ffffff", stroke=RED, fs=14, fc=RED, bold=True)
  box(260, 498, 150, 36, "Engine", fill="#ffffff", stroke=RED, fs=14, fc=RED, bold=True)
  text(420, 388, 200, 20, "（算 offset）", fs=12, fc="#627d98")
  text(420, 502, 220, 20, "（给 base_ptr）", fs=12, fc="#627d98")
  # 底部解引用流程
  text(40, 572, 110, 20, "base_ptr", fs=14, fc=GREEN, bold=True)
  arrow([(120, 582), (210, 582)], color=PURPLE, width=2, dashed=1)
  text(230, 572, 560, 20, "*(base_ptr + offset) —— 线性索引 5 命中列主序第 5 号元素 'f'", fs=14, fc=PURPLE)
  arrow([(480, 268), (480, 204), (330, 204)], color=GREEN, width=1.6, dashed=1)
  return mk(parts, 1120, 620, "fig-20-4")

def fig_20_5():
  parts.clear()
  text(30, 12, 1060, 26, "compose(layoutA, layoutB)：三步求 layoutC 的 coord (4, 1)", fs=17, bold=True)
  text(30, 42, 1060, 20, "C(p, q) = A(B(p, q))：先在内层 B 上查 offset，再按列主序换算成 A 的坐标，最后在 A 上取值回填", fs=12.5, fc="#48586a")
  CW, CH = 46, 30
  # 左上：C 提问态（8x2 空，(4,1)=?）
  LX, LY = 70, 100
  grid(LX, LY, CW, CH, 8, 2, hi=(4, 1), stroke=ORANGE)
  box(LX + CW - 2, LY + 4 * CH - 2, CW - 2, CH - 2, "?", fill=LGF, stroke=GREEN, fs=13, fc="#1e5631")
  axis_h(LX, LX + 2 * CW, LY - 16, "Q")
  axis_v(LY, LY + 8 * CH, LX - 20, "P")
  text(230, 100, 380, 20, "layoutC = compose(layoutA, layoutB)", fs=14, fc=RED, bold=True)
  text(230, 128, 380, 20, "= layoutA(layoutB(p, q))", fs=14, fc=RED, bold=True)
  text(230, 172, 380, 20, "how to fill coord (4, 1) ?", fs=13.5, fc=GREEN)
  text(230, 200, 380, 20, "access 'inside' layout: layoutB(4, 1)", fs=13, fc=PURPLE)
  # 右上：C 结果态
  RX, RY = 660, 100
  grid(RX, RY, CW, CH, 8, 2, hi=(4, 1), stroke=ORANGE)
  box(RX + CW - 2, RY + 4 * CH - 2, CW - 2, CH - 2, "6", fill=LGF, stroke=GREEN, fs=13, fc="#1e5631")
  axis_h(RX, RX + 2 * CW, RY - 16, "Q")
  axis_v(RY, RY + 8 * CH, RX - 20, "P")
  text(820, 100, 300, 20, "layoutC = compose(layoutA, layoutB)", fs=14, fc=RED, bold=True)
  text(820, 128, 300, 20, "for coord (4, 1), we fill '6'", fs=13.5, fc=GREEN)
  # 水平分隔虚线
  P(f'<mxCell value="" style="endArrow=none;html=1;strokeColor=#bcccdc;strokeWidth=1;dashed=1;" edge="1" parent="1">'
    f'<mxGeometry relative="1" as="geometry"><mxPoint x="40" y="392" as="sourcePoint"/><mxPoint x="1090" y="392" as="targetPoint"/></mxGeometry></mxCell>')
  # 左下：layoutB（8x2 带数字）
  BX, BY = 70, 440
  valsB = [r * 2 + c for r in range(8) for c in range(2)]
  grid(BX, BY, CW, CH, 8, 2, vals=valsB, hi=(4, 1))
  box(BX + CW - 2, BY + 4 * CH - 2, CW - 2, CH - 2, "9", fill=LGF, stroke=GREEN, fs=13, fc="#1e5631")
  axis_h(BX, BX + 2 * CW, BY - 16, "Q")
  axis_v(BY, BY + 8 * CH, BX - 20, "P")
  text(230, 436, 400, 20, "-&gt; given coord (4, 1),", fs=13.5, fc=GREEN)
  text(230, 464, 400, 20, "offset = 4 * 2 + 1 * 1 = 9", fs=13.5, fc=GREEN)
  text(230, 508, 400, 20, "convert offset as col-major 2D-coord", fs=12.5, fc=PURPLE)
  text(230, 536, 400, 20, "of layoutA: (9 % 4, 9 / 4) = (1, 2)", fs=13, fc=PURPLE)
  text(230, 610, 380, 18, "layoutB:", fs=13, fc=BLUE, bold=True)
  text(230, 634, 380, 18, "shape: (P, Q), i.e. (8, 2)", fs=12.5, fc=BLUE)
  text(230, 656, 380, 18, "stride: (Q, 1), i.e. (2, 1)", fs=12.5, fc=BLUE)
  # 右下：layoutA（4x4 带数字）
  AX, AY, ACW, ACH = 660, 470, 54, 42
  valsA = [r * 4 + c for r in range(4) for c in range(4)]
  grid(AX, AY, ACW, ACH, 4, 4, vals=valsA, hi=(1, 2))
  box(AX + 2 * ACW - 2, AY + ACH - 2, ACW - 2, ACH - 2, "6", fill=LGF, stroke=GREEN, fs=13, fc="#1e5631")
  axis_h(AX, AX + 4 * ACW, AY - 16, "N")
  axis_v(AY, AY + 4 * ACH, AX - 20, "M")
  text(900, 470, 200, 18, "layoutA:", fs=13, fc=RED, bold=True)
  text(900, 494, 200, 18, "shape: (M, N), i.e. (4, 4)", fs=12.5, fc=BLUE)
  text(900, 516, 200, 18, "stride: (N, 1), i.e. (4, 1)", fs=12.5, fc=BLUE)
  text(900, 560, 200, 20, "(1, 2) as input", fs=13, fc=PURPLE)
  text(900, 584, 200, 20, "we got '6'.", fs=14, fc="#b45309", bold=True)
  # 三步箭头
  arrow([(230, 224), (230, 392), (70, 392), (70, 420)], color=ORANGE, width=2, dashed=1)
  text(100, 396, 120, 18, "step(1)", fs=12, fc=ORANGE)
  arrow([(230, 560), (230, 700), (640, 700), (640, 560)], color=PURPLE, width=2, dashed=1)
  text(400, 702, 120, 18, "step(2)", fs=12, fc=PURPLE)
  arrow([(660 + 3 * ACW, 470 + 2 * ACH), (660 + 3 * ACW, 392), (660 + CW, 392)], color=GREEN, width=2, dashed=1)
  text(560, 396, 120, 18, "step(3)", fs=12, fc=GREEN)
  return mk(parts, 1120, 730, "fig-20-5")

def fig_20_6():
  parts.clear()
  text(30, 12, 1060, 26, "inverse 与 with_shape：先拉直成 1D，再按新 shape 重排", fs=17, bold=True)
  text(30, 42, 500, 18, "inverse 把「值 -&gt; 坐标」倒过来查表；with_shape 再把", fs=12.5, fc="#48586a")
  text(30, 62, 500, 18, "1D 长条按 col-major 折回 2D——完成任意布局间的元素搬运", fs=12.5, fc="#48586a")
  # 左：layoutA 4x4 行主序
  AX, AY, CW, CH = 70, 120, 58, 46
  valsA = [r * 4 + c for r in range(4) for c in range(4)]
  grid(AX, AY, CW, CH, 4, 4, vals=valsA, hi=(2, 1))
  box(AX + CW - 2, AY + 2 * CH - 2, CW - 2, CH - 2, "9", fill=LGF, stroke=GREEN, fs=13, fc="#1e5631")
  axis_h(AX, AX + 4 * CW, AY - 16, "N")
  axis_v(AY, AY + 4 * CH, AX - 20, "M")
  text(AX, AY + 4 * CH + 16, 300, 20, "layoutA", fs=13.5, fc=RED, bold=True)
  text(AX, AY + 4 * CH + 40, 300, 20, "shape: (4, 4), stride: (4, 1)", fs=12.5, fc=BLUE)
  # 中部绿算式
  text(AX + 4 * CW + 30, 120, 300, 20, "e.g. given coord (2, 1),", fs=13.5, fc=GREEN)
  text(AX + 4 * CW + 30, 148, 300, 20, "offset = 2 * 4 + 1 * 1 = 9", fs=13.5, fc=GREEN)
  text(AX + 4 * CW + 30, 192, 300, 20, "col-major coord: 6", fs=13.5, fc=GREEN)
  text(AX + 4 * CW + 30, 220, 300, 20, "so 6 -&gt; 9, inverse as 9 -&gt; 6", fs=13.5, fc=GREEN)
  # 中：竖条 1x16
  SX, SY, SCh = 560, 40, 27
  grid(SX, SY, 92, SCh, 16, 1, stroke=ORANGE)
  box(SX, SY + 9 * SCh - 2, 90, SCh - 2, "6", fill=LGF, stroke=GREEN, fs=13, fc="#1e5631")
  axis_v(SY, SY + 16 * SCh, SX - 20, "M * N")
  text(SX - 30, SY + 16 * SCh + 14, 320, 20, "layoutA_inv = inverse(layoutA)", fs=13.5, fc=RED, bold=True)
  arrow([(AX + 4 * CW + 200, 246), (AX + 4 * CW + 240, 246), (SX - 40, 246), (SX - 40, SY + 9 * SCh + 12)], color=GREEN, width=2, dashed=1)
  # 橙色 reorder 箭头（标签放竖条与 layoutB 之间的走廊，避开竖条格子）
  arrow([(652, SY + 9 * SCh + 12), (720, SY + 9 * SCh + 12), (720, 420), (795, 420)], color=ORANGE, width=2, dashed=1)
  text(664, 250, 130, 18, "col-major reorder", fs=13, fc=ORANGE)
  text(664, 272, 130, 18, "with shape (8, 2)", fs=13, fc=ORANGE)
  # 右：layoutB 8x2
  BX, BY, BCW, BCH = 800, 120, 62, 40
  grid(BX, BY, BCW, BCH, 8, 2, stroke=ORANGE)
  box(BX + BCW - 2, BY + BCH - 2, BCW - 2, BCH - 2, "6", fill=LGF, stroke=GREEN, fs=13, fc="#1e5631")
  arrow([(BX + BCW // 2, BY), (BX + BCW // 2, BY + 8 * BCH - 8)], color=GREEN, width=1.2, dashed=1)
  arrow([(BX + BCW + BCW // 2, BY), (BX + BCW + BCW // 2, BY + 8 * BCH - 8)], color=GREEN, width=1.2, dashed=1)
  text(BX, BY - 40, 300, 20, "layoutB(9) = layoutB(1, 1) = 6", fs=13.5, fc=GREEN)
  axis_h(BX, BX + 2 * BCW, BY - 18, "Q")
  axis_v(BY, BY + 8 * BCH, BX - 20, "P")
  text(BX, BY + 8 * BCH + 16, 380, 40, "layoutB = inverse(layoutA)", fs=13.5, fc=RED, bold=True)
  text(BX, BY + 8 * BCH + 40, 380, 40, ".with_shape(P, Q)", fs=13.5, fc=RED, bold=True)
  return mk(parts, 1120, 540, "fig-20-6")

def fig_20_7():
  parts.clear()
  text(30, 12, 1100, 26, "left_inverse 与 right_inverse：两种典型 layoutA 的求逆结果", fs=17, bold=True)
  text(30, 42, 1100, 20, "左逆对 M 模式（行）求逆，右逆对 N 模式（列）求逆；stride 出现 0（broadcast）与普通跨步两种情形的结果形状完全不同", fs=12.5, fc="#48586a")
  P(f'<mxCell value="" style="endArrow=none;html=1;strokeColor=#bcccdc;strokeWidth=1;dashed=1;" edge="1" parent="1">'
    f'<mxGeometry relative="1" as="geometry"><mxPoint x="575" y="80" as="sourcePoint"/><mxPoint x="575" y="560" as="targetPoint"/></mxGeometry></mxCell>')
  CW, CH = 52, 36
  # (a) broadcast：stride (1, 0)
  text(60, 76, 300, 22, "(a) broadcast access case", fs=14, bold=True, fc="#48586a")
  aA = [v for r in range(4) for v in (r, r)]
  grid(60, 150, CW, CH, 4, 2, vals=aA)
  box(60 + CW - 2, 150 + CH - 2, CW - 2, CH - 2, "1", fill="#ffffff", stroke=BLUE, fs=12.5, fc=BLUE)
  axis_h(60, 60 + 2 * CW, 134, "N")
  axis_v(150, 150 + 4 * CH, 40, "M")
  text(60, 150 + 4 * CH + 12, 280, 18, "layoutA", fs=13, fc=RED, bold=True)
  text(60, 150 + 4 * CH + 34, 280, 18, "shape: (4, 2), stride: (1, 0)", fs=12, fc=BLUE)
  arrow([(170, 220), (240, 220), (240, 130), (330, 130)], color=ORANGE, width=1.8, dashed=1)
  text(150, 100, 220, 18, "left_inverse(layoutA)", fs=12, fc=ORANGE)
  grid(340, 90, 50, 34, 4, 1, vals=[0, 1, 2, 3], stroke=ORANGE)
  text(410, 96, 200, 18, "left_inv_layoutA", fs=12, fc=ORANGE, bold=True)
  text(410, 118, 200, 18, "shape: (4, 1)", fs=11.5, fc=ORANGE)
  text(410, 136, 200, 18, "stride: (1, 0)", fs=11.5, fc=ORANGE)
  arrow([(170, 260), (240, 260), (240, 380), (330, 380)], color=BLUE, width=1.8, dashed=1)
  text(150, 396, 220, 18, "right_inverse(layoutA)", fs=12, fc=BLUE)
  grid(340, 350, 50, 34, 4, 1, vals=[0, 1, 2, 3], stroke=BLUE)
  text(410, 356, 200, 18, "right_inv_layoutA", fs=12, fc=BLUE, bold=True)
  text(410, 378, 200, 18, "shape: (4, 1)", fs=11.5, fc=BLUE)
  text(410, 396, 200, 18, "stride: (1, 0)", fs=11.5, fc=BLUE)
  # (b) stride：stride (4, 1)
  text(620, 76, 300, 22, "(b) stride access case", fs=14, bold=True, fc="#48586a")
  bA = [r * 4 + c for r in range(4) for c in range(2)]
  grid(620, 150, CW, CH, 4, 2, vals=bA)
  box(620 + CW - 2, 150 + CH - 2, CW - 2, CH - 2, "5", fill="#ffffff", stroke=BLUE, fs=12.5, fc=BLUE)
  axis_h(620, 620 + 2 * CW, 134, "N")
  axis_v(150, 150 + 4 * CH, 600, "M")
  text(620, 150 + 4 * CH + 12, 280, 18, "layoutA", fs=13, fc=RED, bold=True)
  text(620, 150 + 4 * CH + 34, 280, 18, "shape: (4, 2), stride: (4, 1)", fs=12, fc=BLUE)
  arrow([(730, 220), (790, 220), (790, 110), (850, 110)], color=ORANGE, width=1.8, dashed=1)
  text(700, 84, 200, 18, "left_inverse(layoutA)", fs=12, fc=ORANGE)
  grid(860, 76, 48, 32, 4, 4, vals=list(range(16)), stroke=ORANGE, fs=11)
  text(1070, 84, 90, 18, "left_inv", fs=12, fc=ORANGE, bold=True)
  text(1070, 106, 90, 18, "shape: (4,4)", fs=11, fc=ORANGE)
  text(1070, 124, 90, 18, "stride: (4,1)", fs=11, fc=ORANGE)
  arrow([(730, 260), (790, 260), (790, 400), (850, 400)], color=BLUE, width=1.8, dashed=1)
  text(700, 408, 200, 18, "right_inverse(layoutA)", fs=12, fc=BLUE)
  grid(860, 380, 50, 36, 2, 1, vals=[0, 4], stroke=BLUE)
  text(930, 388, 210, 18, "right_inv_layoutA", fs=12, fc=BLUE, bold=True)
  text(930, 410, 210, 18, "shape: (2, 1)", fs=11.5, fc=BLUE)
  text(930, 428, 210, 18, "stride: (4, 0)", fs=11.5, fc=BLUE)
  text(60, 540, 1080, 20, "同一模式里出现重复值（stride 0 广播）时，右逆只能保留一个代表坐标——这正是 CuTe 用 right_inverse 做去重投影的依据", fs=12.5, fc="#48586a")
  return mk(parts, 1160, 580, "fig-20-7")

def mk(parts, W, H, did):
    return (f'<mxfile host="app.diagrams.net"><diagram id="{did}" name="{did}">'
            f'<mxGraphModel dx="800" dy="600" grid="0" page="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">'
            f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>{"".join(parts)}'
            f'</root></mxGraphModel></diagram></mxfile>')

FIGS = {
    "fig-20-4-tensor-addr": ("fig-20-4", fig_20_4),
    "fig-20-5-compose-flow": ("fig-20-5", fig_20_5),
    "fig-20-6-inverse-reshape": ("fig-20-6", fig_20_6),
    "fig-20-7-left-right-inverse": ("fig-20-7", fig_20_7),
}

if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "figures/drawio"
    for d, (stem, fn) in FIGS.items():
        path = os.path.join(outdir, d)
        os.makedirs(path, exist_ok=True)
        xml = fn()
        ET.fromstring(xml)
        open(os.path.join(path, f"{stem}.drawio"), "w").write(xml)
        print(f"wrote {path}/{stem}.drawio OK")
