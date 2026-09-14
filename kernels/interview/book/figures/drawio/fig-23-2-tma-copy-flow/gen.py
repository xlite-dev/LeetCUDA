#!/usr/bin/env python3
# gen_fig23_2.py — FIG-23-2：知乎 @竹熙佳处《写给大家看的 CuTe 教程：TMA Copy》fig-2 重建
# 来源：figures/zhihu/zhuxijiachu-tma-copy/fig-2.jpg（原图已本地归档）
# 重建原则：保留教学语义（tensormap 描述坐标/box 参数；1 个 thread 发起；TMA 引擎整块搬
# gmem box -> smem tile），配色映射为书内系列色。原图 tensormap 面板内嵌有一行可疑英文
# 注入文本（"ignore all previous instructions..."），非教学语义，重建时剔除。
import os, sys

parts = []
def P(s): parts.append(s)

def box(x, y, w, h, t="", fill="#ffffff", stroke="#bcccdc", fs=13, bold=False, fc="#1f2933", dashed=0, sw=1):
    st = (f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={sw};"
          f"fontSize={fs};fontColor={fc};align=center;verticalAlign=middle;fontFamily=Helvetica")
    if bold: st += ";fontStyle=1"
    if dashed: st += ";dashed=1"
    P(f'<mxCell value="{t}" style="{st}" vertex="1" parent="1">'
      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def ellipse(x, y, w, h, t, fill="#ffffff", stroke="#bcccdc", fs=13, bold=False, fc="#1f2933"):
    st = (f"ellipse;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
          f"fontSize={fs};fontColor={fc};align=center;verticalAlign=middle;fontFamily=Helvetica")
    if bold: st += ";fontStyle=1"
    P(f'<mxCell value="{t}" style="{st}" vertex="1" parent="1">'
      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def text(x, y, w, h, t, fs=13, fc="#102a43", bold=False):
    st = f"text;html=1;align=left;verticalAlign=middle;fontSize={fs};fontColor={fc};fontFamily=Helvetica"
    if bold: st += ";fontStyle=1"
    P(f'<mxCell value="{t}" style="{st}" vertex="1" parent="1">'
      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def arrow(pts, color="#48586a", width=2, dashed=0):
    st = f"endArrow=blockThin;endFill=1;html=1;strokeColor={color};strokeWidth={width};rounded=0;"
    if dashed: st += "dashed=1;"
    P(f'<mxCell value="" style="{st}" edge="1" parent="1">'
      f'<mxGeometry relative="1" as="geometry"><Array as="points">'
      + ''.join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts[1:-1])
      + f'</Array><mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint"/>'
        f'<mxPoint x="{pts[-1][0]}" y="{pts[-1][1]}" as="targetPoint"/></mxGeometry></mxCell>')

BLUE, GREEN, ORANGE, RED, PURPLE = "#2171b5", "#2e8540", "#d97706", "#dc2626", "#6a51a3"
LBF, LGF, LOF, LRF, GRAY = "#bfd7ea", "#c9e4c8", "#fde9d0", "#fdeaea", "#e7ecf3"
DKG, DKO, DKGN = "#1e5631", "#b45309", "#48586a"

def fig_23_2():
  parts.clear()
  text(30, 10, 1100, 26, "TMA Copy 数据流：一个 thread 发起，TMA 引擎按 tensormap 描述搬运整个 box", fs=17)
  text(30, 38, 1100, 20, "CPU 将张量形状与 box 参数 encode 进 tensormap；gmem 的每个 box 由 TMA 引擎整块搬入 smem tile", fs=12.5, fc=DKGN)
  # ---- gmem 全局张量：16 列 x 6 行，box 切分 3x3，前 3 个 box 已搬完，box_3 正在搬 ----
  GX, GY, CW, CH, NC, NR = 60, 116, 44, 44, 16, 6
  text(60, 74, 220, 20, "gmem 全局张量", fs=14, fc=BLUE)
  text(456, 74, 160, 20, "当前 box (3, 3)", fs=12.5, fc=DKO)
  text(757, 94, 50, 18, "globalX", fs=12, fc="#627d98")
  for r in range(NR):
    for c in range(NC):
      if r < 3 and c < 9: f, s = LGF, GREEN
      elif r < 3 and c < 12: f, s = LOF, ORANGE
      else: f, s = "#ffffff", "#bcccdc"
      box(GX + c * CW, GY + r * CH, CW - 2, CH - 2, fill=f, stroke=s)
  for g in range(3):
    box(GX + 132 * g, GY, 132, 132, fill="none", stroke=GREEN, sw=2.5)
  box(GX + 132 * 3, GY, 132, 132, fill="none", stroke=ORANGE, sw=3.5)
  for c in range(NC):
    text(GX + CW * c + 13, 94, 18, 18, str(c), fs=12, fc="#627d98")
  for r in range(NR):
    text(34, GY + CH * r + 13, 18, 18, str(r), fs=12, fc="#627d98")
  text(24, 384, 66, 18, "globalY", fs=12, fc="#627d98")
  # ---- tensormap 面板：box 坐标 + 描述参数（CPU encode） ----
  box(812, 74, 338, 260, fill="#ffffff", stroke="#8aa2b8")
  text(832, 84, 300, 20, "tensormap (CPU encode)", fs=14, fc=PURPLE, bold=True)
  text(832, 116, 300, 18, "box_0 = {x = 0, y = 0}", fs=12.5, fc="#1f2933")
  text(832, 142, 300, 18, "box_1 = {x = 3, y = 0}", fs=12.5, fc="#1f2933")
  text(832, 168, 300, 18, "box_2 = {x = 6, y = 0}", fs=12.5, fc="#1f2933")
  text(832, 194, 300, 18, "box_3 = {x = 9, y = 0}  ← 当前", fs=12.5, fc=DKO)
  text(832, 224, 300, 18, "boxDim = (3, 3)", fs=12.5, fc="#1f2933")
  text(832, 248, 310, 18, "globalStrides / elementStrides", fs=12.5, fc="#1f2933")
  text(832, 278, 310, 18, "坐标 / box 大小 / 形状等由", fs=12, fc=DKGN)
  text(832, 298, 310, 18, "CPU encode 写入 tensormap", fs=12, fc=DKGN)
  # ---- TMA 引擎 + 1 thread 发起 ----
  box(640, 412, 260, 90, "TMA 引擎", fill=PURPLE, stroke="#4a1d6e", fs=15, fc="#ffffff")
  ellipse(928, 415, 80, 70, "1 thread", fill=LRF, stroke=RED, fs=12.5, bold=True, fc=RED)
  text(908, 492, 200, 18, "只需 1 个 thread 发起", fs=12, fc=RED)
  # ---- smem tile 面板：12 列 x 3 行，box 依次落地 ----
  box(50, 456, 544, 214, fill=LBF, stroke=BLUE)
  text(60, 462, 200, 20, "smem tile", fs=14, fc=BLUE, bold=True)
  SX, SY = 60, 516
  for r in range(3):
    for c in range(12):
      f, s = (LGF, GREEN) if c < 9 else (LOF, ORANGE)
      box(SX + c * CW, SY + r * CH, CW - 2, CH - 2, fill=f, stroke=s)
  for g in range(3):
    box(SX + 132 * g, SY, 132, 132, fill="none", stroke=GREEN, sw=2.5)
  box(SX + 132 * 3, SY, 132, 132, fill="none", stroke=ORANGE, sw=3.5)
  for c in range(12):
    text(SX + CW * c + 13, 494, 18, 18, str(c), fs=12, fc="#3e5c76")
  text(600, 494, 50, 18, "tileX", fs=12, fc="#3e5c76")
  for r in range(3):
    text(34, SY + CH * r + 13, 18, 18, str(r), fs=12, fc="#3e5c76")
  text(56, 652, 50, 18, "tileY", fs=12, fc="#3e5c76")
  text(118, 652, 320, 18, "tile 坐标从 0 重新计数", fs=12, fc=DKGN)
  # ---- 数据流箭头 ----
  arrow([(522, 380), (522, 440), (638, 440)], color=ORANGE, width=2.5)
  arrow([(836, 334), (872, 411)], color=PURPLE, width=2)
  arrow([(640, 470), (588, 548)], color=BLUE, width=2.5)
  arrow([(928, 450), (902, 450)], color=RED, width=2, dashed=1)
  text(384, 415, 140, 18, "搬运整个 box", fs=12.5, fc=DKO)
  text(884, 352, 150, 18, "encode 坐标/参数", fs=12, fc=PURPLE)
  text(640, 530, 150, 18, "写入 smem tile", fs=12.5, fc=BLUE)
  # ---- 图例 ----
  leg = [(LGF, GREEN, "已搬完的 box 数据", DKG), (LOF, ORANGE, "正在搬运的 box 数据", DKO),
         ("#ffffff", "#bcccdc", "未搬运的数据", DKGN), (PURPLE, PURPLE, "TMA 引擎", PURPLE),
         (LRF, RED, "发起拷贝的 1 个 thread", RED)]
  for i, (f, s, t, fc) in enumerate(leg):
    y = 548 + 26 * i
    box(720, y, 18, 18, fill=f, stroke=s)
    text(746, y - 2, 240, 18, t, fs=12.5, fc=fc)
  return mk(parts, 1160, 680, "fig-23-2")

def mk(parts, W, H, did):
  xml = ('<mxfile host="app.diagrams.net"><diagram id="%s" name="Page-1">'
         '<mxGraphModel dx="800" dy="600" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" '
         'arrows="1" fold="1" page="1" pageScale="1" pageWidth="%d" pageHeight="%d" math="0" shadow="0">'
         '<root><mxCell id="0"/><mxCell id="1" parent="0"/>%s</root></mxGraphModel></diagram></mxfile>'
         ) % (did, W, H, ''.join(parts))
  outdir = os.path.join(sys.argv[1] if len(sys.argv) > 1 else "figures/drawio",
                        {"fig-23-2": "fig-23-2-tma-copy-flow"}[did])
  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, did + ".drawio")
  open(path, "w").write(xml)
  return path

if __name__ == "__main__":
  print("wrote", fig_23_2(), "OK")
