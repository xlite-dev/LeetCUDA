#!/usr/bin/env python3
# gen_batch_pd.py — FIG-20-8/20-9：知乎 @竹熙佳处《Layout Product & Divide》两图重建
# 来源：figures/zhihu/zhuxijiachu-layout-product-divide/fig-{2,7}.jpg（原图已本地归档）
# 重建原则：保留教学语义（product 块拷贝拼接 / divide 子块重组值排布不变），配色映射为书内系列色
import os, sys

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
LBF, LGF, LOF, LRF, GRAY = "#bfd7ea", "#c9e4c8", "#fde9d0", "#fdeaea", "#e7ecf3"
# 块色板（四块）：蓝 / 橙 / 灰 / 红 + 对应描边
PLATES = [(LBF, BLUE), (LOF, "#b45309"), (GRAY, "#627d98"), (LRF, RED)]

def blocks(x0, y0, cw, ch, br, bc, sw=144, sh=104):
  """画 br x bc 个块底色 + 块描边（每块 sw x sh）"""
  for m in range(br):
    for n in range(bc):
      fill, stroke = PLATES[m * bc + n]
      box(x0 + n * sw, y0 + m * sh, sw - 2, sh - 2, fill=fill, stroke=stroke)

def grid(x0, y0, cw, ch, nr, nc, vals=None, hi=None, stroke="#bcccdc", fs=12.5):
  for r in range(nr):
    for c in range(nc):
      v = "" if vals is None else str(vals[r * nc + c])
      f, s, fc = "none", stroke, "#1f2933"
      if hi and (r, c) == hi:
        f, s, fc = LGF, GREEN, "#1e5631"
      box(x0 + c * cw, y0 + r * ch, cw - 2, ch - 2, v, fill=f, stroke=s, fs=fs, fc=fc)

def fig_20_8():
  parts.clear()
  text(30, 12, 1100, 26, "layout product：外层定块位、内层定块内，size 相乘成嵌套布局", fs=17, bold=True)
  text(30, 42, 1080, 20, "A 的每个坐标位放一份 B 的拷贝：块偏移 = A 的线性序号 * size(B)，块内排布即 B", fs=12.5, fc="#48586a")
  # 左上 layoutA
  AX, AY, CW, CH = 70, 130, 56, 46
  grid(AX, AY, CW, CH, 2, 2, vals=[0, 1, 2, 3], stroke=RED, fs=14)
  text(AX, AY + 2 * CH + 12, 220, 20, "layoutA", fs=13.5, fc=RED, bold=True)
  text(AX, AY + 2 * CH + 34, 220, 20, "shape: (2, 2), stride: (2, 1)", fs=12.5, fc=BLUE)
  text(AX + 2 * CW + 14, AY + CH - 12, 24, 18, "m", fs=12, fc="#627d98")
  text(AX - 16, AY + CH + 6, 16, 18, "n", fs=12, fc="#627d98")
  # 符号
  text(80, 288, 120, 30, "product", fs=14, fc="#48586a", bold=True)
  # 左下 layoutB
  BX, BY = 70, 350
  grid(BX, BY, CW, CH, 2, 2, vals=[0, 1, 2, 3], stroke=BLUE, fs=14)
  text(BX, BY + 2 * CH + 12, 220, 20, "layoutB", fs=13.5, fc=BLUE, bold=True)
  text(BX, BY + 2 * CH + 34, 220, 20, "shape: (2, 2), stride: (2, 1)", fs=12.5, fc=BLUE)
  text(BX + 2 * CW + 14, BY + CH - 12, 24, 18, "p", fs=12, fc="#627d98")
  text(BX - 16, BY + CH + 6, 16, 18, "q", fs=12, fc="#627d98")
  # 中部示意：A 每格 -> 一份 B
  arrow([(320, 176), (370, 176), (370, 396), (330, 396)], color=PURPLE, width=2, dashed=1)
  text(300, 240, 180, 40, "每格放一份 B", fs=12.5, fc=PURPLE)
  # 右大网格：块拷贝拼接值排布
  GX, GY, GW, GH = 560, 110, 72, 52
  blocks(GX, GY, GW, GH, 2, 2, sw=2 * GW, sh=2 * GH)
  vals = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
  grid(GX, GY, GW, GH, 4, 4, vals=vals, hi=(3, 1), stroke="#8aa2b8", fs=14)
  text(GX, GY + 4 * GH + 14, 420, 20, "layoutA (x) layoutB", fs=13.5, fc=RED, bold=True)
  text(GX, GY + 4 * GH + 38, 460, 20, "shape: ((2,2), (2,2)),  stride: ((8,4), (2,1))", fs=12.5, fc=BLUE)
  text(GX, GY + 4 * GH + 62, 460, 20, "块序 = A 的 colex 序：左上 -&gt; 右上 -&gt; 左下 -&gt; 右下", fs=12.5, fc="#48586a")
  # 右侧算式
  text(920, 130, 220, 20, "value = 8m + 4n", fs=13.5, fc="#48586a")
  text(920, 154, 220, 20, "        + 2p + 1q", fs=13.5, fc="#48586a")
  text(920, 200, 220, 44, "e.g. (m,n)=(1,0)", fs=13.5, fc=GREEN)
  text(920, 224, 220, 20, "       (p,q)=(1,1)", fs=13.5, fc=GREEN)
  text(920, 258, 220, 20, "8+0+2+1 = 11", fs=14, fc=GREEN, bold=True)
  text(920, 300, 220, 40, "flatten 后仍合法;", fs=12.5, fc="#48586a")
  text(920, 322, 220, 20, "TiledCopy / TiledMMA", fs=12.5, fc="#48586a")
  text(920, 344, 220, 20, "的层级结构由 product 搭出", fs=12.5, fc="#48586a")
  return mk(parts, 1160, 470, "fig-20-8")

def fig_20_9():
  parts.clear()
  text(30, 12, 1100, 26, "logical_divide：按 2x2 子块重组为嵌套布局（值排布不变）", fs=17, bold=True)
  text(30, 42, 1080, 20, "divide 是 product 的逆：外层 (2,2):(8,2) 索引块，内层 (2,2):(4,1) 索引块内——同一网格换一种分组解释", fs=12.5, fc="#48586a")
  # 左：原布局 L
  LX, LY, CW, CH = 70, 130, 58, 46
  grid(LX, LY, CW, CH, 4, 4, vals=[r * 4 + c for r in range(4) for c in range(4)], stroke="#bcccdc", fs=13)
  text(LX, LY + 4 * CH + 14, 300, 20, "L = (4, 4) : (4, 1)", fs=13.5, fc=RED, bold=True)
  text(LX, LY + 4 * CH + 38, 300, 20, "16 格 row-major 0-15", fs=12.5, fc="#48586a")
  # 中：divider
  DX, DY = 390, 170
  grid(DX, DY, 40, 34, 2, 2, stroke=ORANGE, fs=12)
  text(DX - 20, DY + 2 * 34 + 12, 180, 20, "divide by (2, 2)", fs=13.5, fc=ORANGE, bold=True)
  arrow([(DX + 100, DY + 34), (DX + 150, DY + 34)], color=ORANGE, width=2, dashed=1)
  text(DX - 20, DY - 34, 200, 20, "每维切成 2 段", fs=12.5, fc="#48586a")
  # 右：嵌套结果（值排布不变 row-major）
  GX, GY, GW, GH = 620, 110, 66, 50
  blocks(GX, GY, GW, GH, 2, 2, sw=2 * GW, sh=2 * GH)
  vals = [r * 4 + c for r in range(4) for c in range(4)]
  grid(GX, GY, GW, GH, 4, 4, vals=vals, hi=(3, 1), stroke="#8aa2b8", fs=14)
  text(GX, GY + 4 * GH + 14, 480, 20, "divide(L, (2,2))", fs=13.5, fc=RED, bold=True)
  text(GX, GY + 4 * GH + 38, 500, 20, "shape: ((2,2), (2,2)),  stride: ((4,1), (8,2))", fs=12.5, fc=BLUE)
  # 外层/内层指示（网格上方图例行 + 小箭头）
  text(GX, GY - 36, 250, 20, "块位 (2,2):(8,2)", fs=12.5, fc=PURPLE)
  text(GX + 270, GY - 36, 260, 20, "块内 (2,2):(4,1)", fs=12.5, fc=GREEN)
  arrow([(GX + 110, GY - 14), (GX + GW + 30, GY + 4)], color=PURPLE, width=2, dashed=1)
  arrow([(GX + 420, GY - 14), (GX + GW + 16, GY + GH - 16)], color=GREEN, width=2, dashed=1)
  # 算式（与 20-8 同物理格对比）
  text(GX, GY + 4 * GH + 74, 500, 20, "e.g. 同格 (3,1): 块(1,0) 内(1,1) -&gt; 8*1 + 2*0 + 4*1 + 1 = 13", fs=13.5, fc=GREEN, bold=True)
  text(GX, GY + 4 * GH + 98, 520, 20, "对比 product：其同物理格值为 11——product 生成新值排布，divide 只改分组", fs=12.5, fc="#48586a")
  return mk(parts, 1160, 500, "fig-20-9")

def mk(parts, W, H, did):
  xml = ('<mxfile host="app.diagrams.net"><diagram id="%s" name="Page-1">'
         '<mxGraphModel dx="800" dy="600" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" '
         'arrows="1" fold="1" page="1" pageScale="1" pageWidth="%d" pageHeight="%d" math="0" shadow="0">'
         '<root><mxCell id="0"/><mxCell id="1" parent="0"/>%s</root></mxGraphModel></diagram></mxfile>'
         ) % (did, W, H, ''.join(parts))
  outdir = os.path.join(sys.argv[1] if len(sys.argv) > 1 else "figures/drawio",
                        {"fig-20-8": "fig-20-8-layout-product",
                         "fig-20-9": "fig-20-9-layout-divide"}[did])
  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, did + ".drawio")
  open(path, "w").write(xml)
  return path

if __name__ == "__main__":
  for f in (fig_20_8, fig_20_9):
    print("wrote", f(), "OK")
