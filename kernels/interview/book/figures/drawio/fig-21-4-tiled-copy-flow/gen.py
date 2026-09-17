#!/usr/bin/env python3
# gen_fig21_4.py — FIG-21-4：知乎 @竹熙佳处《写给大家看的 CuTe 教程：tiled copy》fig-1/2/3 重建
# 来源：figures/zhihu/zhuxijiachu-tiled-copy/fig-{1,2,3}.jpg（原图已本地归档）
# 语义：两步 TiledCopy 全景——gmem tile (8,8):(8,1) 上 4 行 32 元素 --g2s(16 thr x 2 val)-->
#       smem tile (4,8) 0-31 --s2r + TV layout--> 每 thread 寄存器 fragment {t, t+16}
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

def textc(x, y, w, h, t, fs=12, fc="#627d98"):
    st = f"text;html=1;align=center;verticalAlign=middle;fontSize={fs};fontColor={fc};fontFamily=Helvetica"
    P(f'<mxCell value="{t}" style="{st}" vertex="1" parent="1">'
      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def arrow(pts, color="#48586a", width=2.5, dashed=0):
    st = f"endArrow=blockThin;endFill=1;html=1;strokeColor={color};strokeWidth={width};rounded=0;"
    if dashed: st += "dashed=1;"
    P(f'<mxCell value="" style="{st}" edge="1" parent="1">'
      f'<mxGeometry relative="1" as="geometry"><Array as="points">'
      + ''.join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts[1:-1])
      + f'</Array><mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint"/>'
        f'<mxPoint x="{pts[-1][0]}" y="{pts[-1][1]}" as="targetPoint"/></mxGeometry></mxCell>')

BLUE, GREEN, ORANGE, RED, PURPLE = "#2171b5", "#2e8540", "#d97706", "#dc2626", "#6a51a3"
LBF, LGF, LOF, LRF, GRAY = "#bfd7ea", "#c9e4c8", "#fde9d0", "#fdeaea", "#e7ecf3"
OSTROKE = "#b45309"

CW = 46  # cell size

def fig_21_4():
  parts.clear()
  # ---- 网格与框（先画，文本一律后画，避免 COVER）----
  # gmem tile: 8x8, 值 = 8r + c；上 4 行参与拷贝（左 4 列绿 / 右 4 列橙），下 4 行灰
  GX, GY = 56, 134
  for r in range(8):
    for c in range(8):
      v = 8 * r + c
      if r < 4:
        fill = LGF if c < 4 else LOF
        box(GX + c * CW, GY + r * CW, CW - 2, CW - 2, str(v), fill=fill, fs=13)
      else:
        box(GX + c * CW, GY + r * CW, CW - 2, CW - 2, str(v), fill=GRAY, fs=13, fc="#627d98")
  # smem tile: 4x8, 值 = 8r + c（0-31）；每行前 4 绿(t0~t7) / 后 4 橙(t8~t15)
  SX, SY = 510, 228
  for r in range(4):
    for c in range(8):
      fill = LGF if c < 4 else LOF
      box(SX + c * CW, SY + r * CW, CW - 2, CW - 2, str(8 * r + c), fill=fill, fs=13)
  # TV layout: 2 行(V0/V1) x 16 列(t0~t15)；V0 = t, V1 = t+16；t<8 绿 / t>=8 橙
  TX, TY = 88, 544
  for r in range(2):
    for t in range(16):
      v = t if r == 0 else t + 16
      fill = LGF if t < 8 else LOF
      box(TX + t * CW, TY + r * CW, CW - 2, CW - 2, str(v), fill=fill, fs=12.5)
  # register fragments: 16 个 thread 盒（4 列 x 4 行），值对 {t, t+16}
  FX, FY = 932, 158
  for t in range(16):
    col, row = t % 4, t // 4
    fill, stroke = (LGF, GREEN) if t < 8 else (LOF, OSTROKE)
    box(FX + col * 53, FY + row * 50, 46, 26, "%d %d" % (t, t + 16), fill=fill, stroke=stroke, fs=12.5)
  # 图例色块
  box(866, 524, 16, 16, fill=LGF, stroke=GREEN)
  box(866, 552, 16, 16, fill=LOF, stroke=OSTROKE)
  box(866, 580, 16, 16, fill=GRAY, stroke="#9fb3c8")

  # ---- 文本（后画）----
  text(30, 8, 1120, 26, "tiled copy 全景数据流：gmem tile → smem tile → register fragment", fs=17)
  text(30, 36, 1120, 20, "两步 TiledCopy：g2s 把 gmem 中 32 个数据拷到 smem；s2r 依 TV layout 分发到各 thread 的寄存器 fragment", fs=12.5, fc="#48586a")
  # gmem 标注
  text(30, 58, 120, 20, "gmem tile", fs=13.5, fc=BLUE, bold=True)
  text(30, 78, 400, 18, "T (64,) 的 (8,8):(8,1) 视图；上 4 行 32 个元素参与拷贝", fs=11.5, fc="#48586a")
  for c in range(8):
    textc(GX + c * CW, 114, CW - 2, 16, str(c), fs=11.5)
  for r in range(8):
    textc(30, GY + r * CW + 13, 22, 16, str(r), fs=11.5)
  # smem 标注
  text(478, 162, 110, 20, "smem tile", fs=13.5, fc=BLUE, bold=True)
  text(600, 164, 240, 18, "← 第一步：TiledCopy g2s", fs=12.5, fc=GREEN)
  text(478, 184, 320, 16, "16 threads × 2 values / thread", fs=11, fc="#48586a")
  for c in range(8):
    textc(SX + c * CW, 208, CW - 2, 16, str(c), fs=11.5)
  for r in range(4):
    text(478, SY + r * CW + 14, 28, 16, "c%d" % r, fs=12, fc="#1e5631")
  # fragment 标注
  text(932, 100, 170, 20, "register fragment", fs=13.5, fc=BLUE, bold=True)
  text(932, 122, 230, 18, "← 第二步：TiledCopy s2r", fs=12.5, fc=PURPLE)
  for t in range(16):
    col, row = t % 4, t // 4
    textc(FX + col * 53, FY + row * 50 - 14, 46, 13, "t%d" % t, fs=11, fc="#48586a")
  # TV layout 标注
  text(56, 510, 460, 20, "TV layout：决定哪个 thread 拷贝哪些数据", fs=13, fc=PURPLE)
  for t in range(16):
    textc(TX + t * CW, 528, CW - 2, 15, "t%d" % t, fs=11)
  for r in range(2):
    textc(60, TY + r * CW + 15, 24, 16, "V%d" % r, fs=11.5)
  # 图例文本
  text(890, 522, 240, 18, "t0~t7 拷贝的数据", fs=12, fc="#1e5631")
  text(890, 550, 240, 18, "t8~t15 拷贝的数据", fs=12, fc="#92400e")
  text(890, 578, 260, 18, "gmem 灰色格：不参与拷贝", fs=12, fc="#627d98")

  # ---- 箭头 ----
  arrow([(GX + 8 * CW + 4, 318), (SX - 6, 318)])
  arrow([(SX + 8 * CW + 4, 262), (FX - 4, 262)])
  return mk(parts, 1160, 640, "fig-21-4")

def mk(parts, W, H, did):
  xml = ('<mxfile host="app.diagrams.net"><diagram id="%s" name="Page-1">'
         '<mxGraphModel dx="800" dy="600" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" '
         'arrows="1" fold="1" page="1" pageScale="1" pageWidth="%d" pageHeight="%d" math="0" shadow="0">'
         '<root><mxCell id="0"/><mxCell id="1" parent="0"/>%s</root></mxGraphModel></diagram></mxfile>'
         ) % (did, W, H, ''.join(parts))
  outdir = os.path.join(sys.argv[1] if len(sys.argv) > 1 else "figures/drawio",
                        "fig-21-4-tiled-copy-flow")
  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, did + ".drawio")
  open(path, "w").write(xml)
  return path

if __name__ == "__main__":
  print("wrote", fig_21_4(), "OK")
